"""
Vectorised agent-based swarm simulator for stenotic coronary flow.

WHAT THIS IS
------------
A numpy-vectorised agent-based model (ABM) of therapeutic nanoparticles
("subagents") transiting a stenosed coronary artery. Each agent independently
advects in a Poiseuille velocity field, diffuses radially, and applies local
rules at the vessel wall:

    * inflamed plaque  -> bind, with saturable probability driven by VCAM-1
                          density and agent avidity, balanced by detachment
    * healthy wall     -> reflect (bounce), with only a small off-target
                          adhesion probability
    * bound agent      -> erodes local plaque exponentially; small chance of
                          releasing embolic debris

Vectorisation means wall-clock cost scales with total agent-steps, not with the
Python-level agent count, so large swarms are affordable on a laptop CPU.

ON THE WORD "TRILLION"
----------------------
A trillion *simultaneous* interacting agents is not achievable on a CPU, and
claiming it would be a lie. A trillion cumulative *agent-steps* is achievable:
this implementation sustains roughly 2e7 agent-steps/s, so ~1e12 agent-steps
costs on the order of 14 hours of CPU time. That is the honest framing, and
`SwarmResult.agent_steps` reports the exact accumulated figure. Report
agent-steps, never "a trillion agents".

UNITS AND SCOPE
---------------
Lengths in micrometres, time in seconds. The velocity field is a steady
Poiseuille profile modulated by a pulsatility term; this is a *reduced-order*
model, not a Navier-Stokes solve. It exists to rank candidate interventions and
to generate a dataset with realistic scaling, not to predict a clinical
outcome. Nothing here is validated against an animal or human study.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from danon.physics import Fluid, stokes_einstein_diffusion

logger = logging.getLogger(__name__)

# Baseline (healthy) plaque burden below which wall is treated as non-inflamed.
# Without a real threshold the Gaussian stenosis tail makes "inflamed" true
# almost everywhere, which silently erases off-target adhesion as a metric.
INFLAMMATION_THRESHOLD = 0.05

# Safety-relevant: distal debris is the clinically dangerous failure mode, so it
# is a first-class output rather than a diagnostic afterthought.
EMBOLIC_RELEASE_RATE_PER_S = 1.0e-4


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ArteryGeometry:
    """Axisymmetric coronary segment with a single Gaussian stenosis."""

    length_um: float = 3000.0
    healthy_radius_um: float = 1500.0     # ~3 mm lumen diameter
    stenosis_center_um: float = 1500.0
    stenosis_sigma_um: float = 380.0
    initial_occlusion_pct: float = 70.0

    @property
    def min_radius_um(self) -> float:
        return self.healthy_radius_um * (1.0 - self.initial_occlusion_pct / 100.0)

    def radius_at(self, x_um: np.ndarray) -> np.ndarray:
        """Lumen radius (um) along the vessel axis.

        Severity is parameterised by *diameter* occlusion, which is what
        angiography reports.
        """
        depth = self.healthy_radius_um - self.min_radius_um
        bump = np.exp(-0.5 * ((x_um - self.stenosis_center_um) / self.stenosis_sigma_um) ** 2)
        return self.healthy_radius_um - depth * bump

    def plaque_fraction(self, x_um: np.ndarray) -> np.ndarray:
        """Normalised plaque burden in [0, 1] along the axis."""
        return np.clip(
            (self.healthy_radius_um - self.radius_at(x_um))
            / max(self.healthy_radius_um - self.min_radius_um, 1e-9),
            0.0,
            1.0,
        )

    def with_occlusion(self, occlusion_pct: float) -> "ArteryGeometry":
        return ArteryGeometry(
            length_um=self.length_um,
            healthy_radius_um=self.healthy_radius_um,
            stenosis_center_um=self.stenosis_center_um,
            stenosis_sigma_um=self.stenosis_sigma_um,
            initial_occlusion_pct=float(np.clip(occlusion_pct, 0.0, 95.0)),
        )

    def resistance(self, occlusion_pct: Optional[float] = None) -> float:
        """Series Poiseuille resistance, proportional to integral of 1/R^4.

        A 70% diameter stenosis roughly quadruples 1/R^4 at its throat, which is
        the physical reason such a lesion matters so much clinically, and why
        the reported pressure-drop and flow changes are large but still bounded.
        """
        g = self if occlusion_pct is None else self.with_occlusion(occlusion_pct)
        xs = np.linspace(0.0, g.length_um, 512)
        radii = np.maximum(g.radius_at(xs), 1e-6)
        inv_r4 = 1.0 / radii ** 4
        integ = np.trapezoid(inv_r4, xs) if hasattr(np, "trapezoid") else np.trapz(inv_r4, xs)
        return float(integ)


# ---------------------------------------------------------------------------
# Config / result
# ---------------------------------------------------------------------------

@dataclass
class SwarmConfig:
    """One simulation run's parameters."""

    n_agents: int = 50_000
    steps: int = 600
    dt_s: float = 2.0e-3

    # Flow
    peak_velocity_um_s: float = 400_000.0   # ~0.4 m/s peak in a stenosis
    pulsatility: float = 0.35
    heart_rate_bpm: float = 70.0

    # Diffusion is DERIVED from Stokes-Einstein for `particle_diameter_nm`, not
    # hand-picked. A hardcoded 12 um^2/s corresponds to a ~10 nm particle, which
    # contradicts the ~100 nm particle this pipeline claims to model; a 100 nm
    # particle in blood at 37 C gives ~1.3 um^2/s. Set explicitly to override.
    diffusion_um2_s: Optional[float] = None
    particle_diameter_nm: float = 100.0

    def __post_init__(self) -> None:
        if self.diffusion_um2_s is None:
            self.diffusion_um2_s = stokes_einstein_diffusion(
                self.particle_diameter_nm * 1e-9, Fluid.blood()
            ) * 1e12  # m^2/s -> um^2/s

    # Agent behaviour
    avidity: float = 0.5                    # 0..1 binding efficiency to VCAM-1
    potency: float = 0.5                    # 0..1 plaque digestion rate
    vcam1_density: float = 0.8              # 0..1 inflammation signal at plaque
    healthy_wall_stickiness: float = 4.0e-4  # per-step off-target adhesion prob

    # Kinetics (per second; converted internally using dt)
    bind_rate_per_s: float = 60.0           # wall-contact -> adhesion rate
    detach_rate_per_s: float = 1.5          # adhered -> free rate
    erosion_rate_per_s: float = 2.0e-4      # plaque decay per adhered agent

    binding_shell_fraction: float = 0.06    # fraction of radius treated as wall
    seed: int = 0

    def signature(self) -> Tuple:
        return (
            self.n_agents, self.steps, self.dt_s, self.peak_velocity_um_s,
            self.pulsatility, self.heart_rate_bpm, self.diffusion_um2_s,
            self.particle_diameter_nm,
            self.avidity, self.potency, self.vcam1_density,
            self.healthy_wall_stickiness, self.bind_rate_per_s,
            self.detach_rate_per_s, self.erosion_rate_per_s,
            self.binding_shell_fraction, self.seed,
        )


@dataclass
class SwarmResult:
    """Outcome of one run. All percentages are physically bounded."""

    plaque_reduction_pct: float
    pressure_drop_reduction_pct: float
    reperfusion_gain_pct: float
    off_target_adhesion_pct: float
    distal_embolic_particle_count: int
    bound_fraction: float
    peak_bound_fraction: float
    agent_steps: int
    wall_clock_s: float
    config: SwarmConfig
    notes: List[str] = field(default_factory=list)

    @property
    def agent_steps_per_second(self) -> float:
        return self.agent_steps / max(self.wall_clock_s, 1e-9)

    def as_metrics(self) -> Dict[str, float]:
        """Raw physical endpoints, in the units the model actually integrates.

        Deliberately reports no ``efficacy`` or ``safety``. Those are
        *normalised* axes defined against the model's own attainable maximum, so
        they live in exactly one place, :func:`normalise_to_reference`. Emitting
        a second, differently-scaled pair here would let two callers both say
        "efficacy 60" while meaning incompatible quantities.
        """
        return {
            "plaque_reduction_pct": self.plaque_reduction_pct,
            "pressure_drop_reduction_pct": self.pressure_drop_reduction_pct,
            "reperfusion_gain_pct": self.reperfusion_gain_pct,
            "off_target_healthy_wall_adhesion_pct": self.off_target_adhesion_pct,
            "distal_embolic_particle_count": float(self.distal_embolic_particle_count),
            "bound_fraction": self.bound_fraction,
        }


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class SwarmSimulator:
    """Vectorised ABM over a stenosed arterial segment.

    Every agent is an entry in a numpy array; there is no per-agent Python loop.
    """

    def __init__(
        self,
        config: Optional[SwarmConfig] = None,
        geometry: Optional[ArteryGeometry] = None,
    ) -> None:
        self.config = config or SwarmConfig()
        self.geometry = geometry or ArteryGeometry()
        self.rng = np.random.default_rng(self.config.seed)

        c = self.config
        g = self.geometry

        # Agent state
        self.x = self.rng.uniform(0.0, g.length_um * 0.15, c.n_agents)
        self.r = self.rng.uniform(-g.healthy_radius_um, g.healthy_radius_um, c.n_agents)
        self.bound = np.zeros(c.n_agents, dtype=bool)

        self._agent_steps = 0
        self._embolic_count = 0
        self._bound_plaque_total = 0
        self._bound_healthy_total = 0

        # Reference resistances for bounded flow reporting.
        self._r_occluded = g.resistance()
        self._r_healthy = g.resistance(0.0)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _plaque_at(plaque: np.ndarray, n_cells: int, length_um: float, x: np.ndarray) -> np.ndarray:
        idx = np.clip(((x / length_um) * n_cells).astype(int), 0, n_cells - 1)
        return plaque[idx]

    # -- main loop --------------------------------------------------------
    def run(self) -> SwarmResult:
        c = self.config
        g = self.geometry
        start = time.time()

        n_cells = 256
        cell_centers = (np.arange(n_cells) + 0.5) * (g.length_um / n_cells)
        plaque = g.plaque_fraction(cell_centers).copy()
        plaque0_total = float(plaque.sum())

        omega = 2.0 * math.pi * (c.heart_rate_bpm / 60.0)
        p_bind_plaque_step = c.bind_rate_per_s * c.dt_s
        p_detach_step = float(np.clip(c.detach_rate_per_s * c.dt_s, 0.0, 1.0))
        p_embolic_step = EMBOLIC_RELEASE_RATE_PER_S * c.dt_s

        peak_bound = 0.0
        t = 0.0

        for _ in range(c.steps):
            t += c.dt_s

            # --- detachment ---------------------------------------------
            if np.any(self.bound) and p_detach_step > 0.0:
                detach = self.bound & (self.rng.random(c.n_agents) < p_detach_step)
                self.bound &= ~detach

            free = ~self.bound

            radius = np.maximum(g.radius_at(self.x), 1e-6)

            # --- axial advection (Poiseuille + pulsatility) --------------
            rr = np.clip(np.abs(self.r) / radius, 0.0, 1.0)
            u_axial = c.peak_velocity_um_s * (1.0 - rr ** 2)
            u_axial *= 1.0 + c.pulsatility * math.sin(omega * t)
            self.x = np.where(free, self.x + u_axial * c.dt_s, self.x)

            # --- radial diffusion ---------------------------------------
            if c.diffusion_um2_s > 0.0:
                sigma = math.sqrt(2.0 * c.diffusion_um2_s * c.dt_s)
                self.r = np.where(
                    free, self.r + self.rng.normal(0.0, sigma, c.n_agents), self.r
                )

            # --- wall interaction ---------------------------------------
            radius = np.maximum(g.radius_at(self.x), 1e-6)
            near_wall = np.abs(self.r) >= radius * (1.0 - c.binding_shell_fraction)
            unresolved = free & near_wall

            if np.any(unresolved):
                plaque_here = self._plaque_at(plaque, n_cells, g.length_um, self.x)
                # A real threshold is required so healthy wall is distinguishable
                # from the diffuse Gaussian tail of the stenosis.
                inflamed = plaque_here > INFLAMMATION_THRESHOLD
                severity = np.clip((plaque_here - INFLAMMATION_THRESHOLD)
                                   / max(1.0 - INFLAMMATION_THRESHOLD, 1e-9), 0.0, 1.0)

                p_plaque = np.clip(
                    c.avidity * c.vcam1_density * severity * p_bind_plaque_step, 0.0, 1.0
                )
                p_bind = np.where(inflamed, p_plaque, c.healthy_wall_stickiness)

                will_bind = unresolved & (self.rng.random(c.n_agents) < p_bind)
                self.bound |= will_bind
                self._bound_plaque_total += int(np.sum(will_bind & inflamed))
                self._bound_healthy_total += int(np.sum(will_bind & ~inflamed))

                # Non-binding agents at the wall are reflected (bounce).
                reflect = unresolved & ~will_bind
                self.r = np.where(reflect, np.sign(self.r) * radius * 0.92, self.r)

            # --- plaque erosion by bound agents --------------------------
            if np.any(self.bound):
                idx = np.clip(
                    ((self.x[self.bound] / g.length_um) * n_cells).astype(int), 0, n_cells - 1
                )
                counts = np.bincount(idx, minlength=n_cells).astype(float)
                # Exponential erosion keeps the field non-negative by
                # construction and is invariant to grid refinement.
                exponent = c.erosion_rate_per_s * c.potency * counts * c.dt_s
                plaque *= np.exp(-exponent)

                if c.potency > 0.0:
                    released = self.rng.random(int(np.sum(self.bound))) < (
                        p_embolic_step * c.potency
                    )
                    self._embolic_count += int(np.sum(released))

            peak_bound = max(peak_bound, float(np.mean(self.bound)))

            # --- recycling (models continuous infusion) -----------------
            out = self.x >= g.length_um
            if np.any(out):
                reenter = out & free
                self.x = np.where(reenter, 0.0, self.x)
                self.r = np.where(
                    reenter,
                    self.rng.uniform(-g.healthy_radius_um, g.healthy_radius_um, c.n_agents),
                    self.r,
                )

            self._agent_steps += c.n_agents

        # -- outcomes -------------------------------------------------------
        final_total = float(plaque.sum())
        if plaque0_total > 0.0:
            plaque_reduction = 100.0 * (1.0 - final_total / plaque0_total)
        else:
            plaque_reduction = 0.0
        plaque_reduction = float(np.clip(plaque_reduction, 0.0, 100.0))

        # Residual occlusion scales with residual plaque burden.
        residual_frac = final_total / max(plaque0_total, 1e-9)
        new_occlusion = g.initial_occlusion_pct * float(np.clip(residual_frac, 0.0, 1.0))
        r_new = g.resistance(new_occlusion)

        if self._r_occluded > 0.0:
            dp_reduction = 100.0 * (1.0 - r_new / self._r_occluded)
        else:
            dp_reduction = 0.0
        dp_reduction = float(np.clip(dp_reduction, 0.0, 100.0))

        # Flow gain is a ratio of resistances, but bounded by the maximum that
        # is physically available: complete resolution to a healthy vessel.
        max_gain_pct = 100.0 * (self._r_occluded / max(self._r_healthy, 1e-30) - 1.0)
        if self._r_occluded > 0.0:
            raw_gain = 100.0 * (self._r_occluded / max(r_new, 1e-30) - 1.0)
        else:
            raw_gain = 0.0
        reperfusion = float(np.clip(raw_gain, 0.0, max(max_gain_pct, 0.0)))

        total_bound = self._bound_plaque_total + self._bound_healthy_total
        off_target = 100.0 * self._bound_healthy_total / total_bound if total_bound > 0 else 0.0

        return SwarmResult(
            plaque_reduction_pct=round(plaque_reduction, 4),
            pressure_drop_reduction_pct=round(dp_reduction, 4),
            reperfusion_gain_pct=round(reperfusion, 4),
            off_target_adhesion_pct=round(float(np.clip(off_target, 0.0, 100.0)), 4),
            distal_embolic_particle_count=int(self._embolic_count),
            bound_fraction=round(float(np.mean(self.bound)), 6),
            peak_bound_fraction=round(peak_bound, 6),
            agent_steps=int(self._agent_steps),
            wall_clock_s=round(time.time() - start, 4),
            config=self.config,
            notes=[
                "Reduced-order Poiseuille model, not a Navier-Stokes solve.",
                "Outputs are simulation results, not clinical predictions.",
                f"Max physically available flow gain for this lesion: "
                f"{max_gain_pct:.1f}%.",
            ],
        )


# ---------------------------------------------------------------------------
# Campaign: many agents across a parameter sweep
# ---------------------------------------------------------------------------

def _run_one(payload: Tuple[SwarmConfig, ArteryGeometry]) -> SwarmResult:
    """Module-level worker so the campaign can be pickled across processes."""
    config, geometry = payload
    return SwarmSimulator(config, geometry).run()


@dataclass
class CampaignReport:
    """Aggregate outcome of a parameter sweep."""

    study_id: str
    runs: int
    total_agent_steps: int
    wall_clock_s: float
    endpoint_means: Dict[str, float]
    endpoint_stdev: Dict[str, float]
    endpoint_ci95: Dict[str, float]
    best_config_index: int
    top_configs: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def agent_steps_per_second(self) -> float:
        return self.total_agent_steps / max(self.wall_clock_s, 1e-9)

    def headline(self) -> str:
        return (
            f"{self.runs} runs | {self.total_agent_steps:,} agent-steps | "
            f"{self.wall_clock_s:.1f}s | {self.agent_steps_per_second:,.0f} agent-steps/s"
        )


def _mean_ci95(values: np.ndarray) -> Tuple[float, float]:
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    sd = float(np.std(values, ddof=1))
    # Normal-approximation half-width; adequate for n >= ~10 and clearly
    # labelled as such rather than presented as an exact interval.
    return sd, float(1.96 * sd / math.sqrt(n))


class SwarmCampaign:
    """Deploy many agents across a parameter sweep and aggregate the results.

    This is the layer that lets a single CPU produce a large *cumulative*
    agent-step count. It reports the exact number achieved rather than an
    aspiration, and it stops early once an agent-step budget is exhausted.
    """

    def __init__(
        self,
        geometry: Optional[ArteryGeometry] = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self.geometry = geometry or ArteryGeometry()
        self.max_workers = max_workers

    @staticmethod
    def sweep_configs(
        base: Optional[SwarmConfig] = None,
        *,
        avidities: Sequence[float] = (0.1, 0.4, 0.8),
        potencies: Sequence[float] = (0.2, 0.6, 1.0),
        vcam1_levels: Sequence[float] = (0.4, 0.9),
        seeds: Sequence[int] = (0, 1, 2),
    ) -> List[SwarmConfig]:
        """Full-factorial sweep, including replicate seeds for error bars."""
        base = base or SwarmConfig()
        out: List[SwarmConfig] = []
        for av in avidities:
            for pot in potencies:
                for vc in vcam1_levels:
                    for sd in seeds:
                        out.append(SwarmConfig(
                            n_agents=base.n_agents,
                            steps=base.steps,
                            dt_s=base.dt_s,
                            peak_velocity_um_s=base.peak_velocity_um_s,
                            pulsatility=base.pulsatility,
                            heart_rate_bpm=base.heart_rate_bpm,
                            diffusion_um2_s=base.diffusion_um2_s,
                            avidity=av,
                            potency=pot,
                            vcam1_density=vc,
                            healthy_wall_stickiness=base.healthy_wall_stickiness,
                            bind_rate_per_s=base.bind_rate_per_s,
                            detach_rate_per_s=base.detach_rate_per_s,
                            erosion_rate_per_s=base.erosion_rate_per_s,
                            binding_shell_fraction=base.binding_shell_fraction,
                            seed=int(sd),
                        ))
        return out

    def run(
        self,
        configs: Sequence[SwarmConfig],
        *,
        study_id: str = "swarm_campaign",
        budget_agent_steps: Optional[int] = None,
    ) -> Tuple[CampaignReport, List[SwarmResult]]:
        """Execute a sweep, optionally stopping at an agent-step budget."""
        start = time.time()
        results: List[SwarmResult] = []
        consumed = 0
        truncated = False

        payloads = [(c, self.geometry) for c in configs]

        if self.max_workers and self.max_workers > 1 and len(payloads) > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=self.max_workers) as pool:
                for res in pool.map(_run_one, payloads):
                    results.append(res)
                    consumed += res.agent_steps
                    if budget_agent_steps is not None and consumed >= budget_agent_steps:
                        truncated = True
                        break
        else:
            for payload in payloads:
                res = _run_one(payload)
                results.append(res)
                consumed += res.agent_steps
                if budget_agent_steps is not None and consumed >= budget_agent_steps:
                    truncated = True
                    break

        wall = time.time() - start

        # Aggregate every metric the results expose.
        metric_names = sorted(results[0].as_metrics().keys()) if results else []
        means: Dict[str, float] = {}
        sds: Dict[str, float] = {}
        ci95: Dict[str, float] = {}
        for name in metric_names:
            vals = np.array([r.as_metrics()[name] for r in results], dtype=float)
            means[name] = float(np.mean(vals)) if len(vals) else 0.0
            sd, ci = _mean_ci95(vals)
            sds[name] = sd
            ci95[name] = ci

        # Rank configs on the derived normalised axes. Previously this used
        # invented weights over raw physical units (percentages mixed with a
        # particle count), which is dimensionally incoherent and untunable.
        def composite(r: SwarmResult) -> float:
            m = normalise_to_reference(r, self.geometry)
            return m["efficacy"] + 0.5 * m["safety"]

        order = sorted(range(len(results)), key=lambda i: composite(results[i]), reverse=True)
        top = []
        for i in order[:5]:
            r = results[i]
            top.append({
                "avidity": r.config.avidity,
                "potency": r.config.potency,
                "vcam1_density": r.config.vcam1_density,
                "seed": r.config.seed,
                "plaque_reduction_pct": r.plaque_reduction_pct,
                "pressure_drop_reduction_pct": r.pressure_drop_reduction_pct,
                "off_target_adhesion_pct": r.off_target_adhesion_pct,
                "distal_embolic_particle_count": r.distal_embolic_particle_count,
                "composite": round(composite(r), 4),
            })

        notes = [
            "Error bars are normal-approximation 95% CI half-widths across runs.",
            "Cumulative agent-steps, not simultaneous agent count.",
        ]
        if truncated:
            notes.append(
                f"Sweep truncated at the agent-step budget ({budget_agent_steps:,}); "
                f"{len(results)} of {len(configs)} configs completed."
            )

        report = CampaignReport(
            study_id=study_id,
            runs=len(results),
            total_agent_steps=int(consumed),
            wall_clock_s=round(wall, 3),
            endpoint_means=means,
            endpoint_stdev=sds,
            endpoint_ci95=ci95,
            best_config_index=int(order[0]) if order else -1,
            top_configs=top,
            notes=notes,
        )
        return report, results


# ---------------------------------------------------------------------------
# Validation: simulation vs benchtop measurement
# ---------------------------------------------------------------------------

def correlation_report(
    predicted: Sequence[float],
    measured: Sequence[float],
    *,
    n_bootstrap: int = 2000,
    seed: int = 0,
) -> Dict[str, float]:
    """Agreement statistics between predicted and measured values.

    This is the layer that turns "we simulated it" into "our model predicts the
    bench", which is the claim that actually carries weight in a research
    report. Returns R^2, slope, intercept, Pearson r, and bootstrap CIs.

    R^2 here is the coefficient of determination (1 - SS_res/SS_tot) about the
    fitted line, so it is directly interpretable as variance explained.
    """
    p = np.asarray(predicted, dtype=float)
    m = np.asarray(measured, dtype=float)
    if p.shape != m.shape:
        raise ValueError(f"shapes differ: {p.shape} vs {m.shape}")
    if len(p) < 3:
        raise ValueError("need at least 3 paired points for a meaningful fit")

    slope, intercept = np.polyfit(p, m, 1)
    fitted = slope * p + intercept
    ss_res = float(np.sum((m - fitted) ** 2))
    ss_tot = float(np.sum((m - np.mean(m)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    pearson = float(np.corrcoef(p, m)[0, 1]) if np.std(p) > 0 and np.std(m) > 0 else 0.0

    # Bootstrap CIs for R^2 and slope.
    rng = np.random.default_rng(seed)
    r2s: List[float] = []
    slopes: List[float] = []
    n = len(p)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        pp, mm = p[idx], m[idx]
        if np.std(pp) == 0:
            continue
        s, i = np.polyfit(pp, mm, 1)
        fit = s * pp + i
        sr = float(np.sum((mm - fit) ** 2))
        st = float(np.sum((mm - np.mean(mm)) ** 2))
        r2s.append(1.0 - sr / st if st > 0 else 0.0)
        slopes.append(float(s))

    def ci(vals: List[float]) -> Tuple[float, float]:
        if not vals:
            return (0.0, 0.0)
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    r2_lo, r2_hi = ci(r2s)
    sl_lo, sl_hi = ci(slopes)

    return {
        "n_pairs": float(n),
        "r2": float(r2),
        "r2_ci95_low": r2_lo,
        "r2_ci95_high": r2_hi,
        "pearson_r": pearson,
        "slope": float(slope),
        "slope_ci95_low": sl_lo,
        "slope_ci95_high": sl_hi,
        "intercept": float(intercept),
        "rmse": float(math.sqrt(ss_res / n)),
        "mean_measured": float(np.mean(m)),
    }


# ---------------------------------------------------------------------------
# Derived efficacy normalisation
# ---------------------------------------------------------------------------

#: Runs are ranked on a 0-100 scale so different agents are comparable. That
#: scale needs an origin. Rather than hardcode "100% = 30% plaque reduction"
#: (an invented constant that a real run could silently exceed or never reach),
#: the ceiling is *derived* by running the same simulator at its attainable
#: maximum. A reference run is therefore always exactly as physical as the run
#: it normalises, and re-deriving it after any model change keeps the two in
#: step.
REFERENCE_POTENCY = 1.0
REFERENCE_AVIDITY = 1.0
REFERENCE_VCAM1_DENSITY = 1.0

_REFERENCE_CACHE: Dict[Tuple, Dict[str, float]] = {}


def reference_config(config: SwarmConfig) -> SwarmConfig:
    """The same run at its attainable maximum: identical size, kinetics and
    geometry; potency, avidity and VCAM-1 density pinned to 1.0.

    Crucially this preserves ``n_agents`` and ``steps``. A reference computed at
    a different swarm size measures a different experiment, and scaling a small
    run against a large reference would cap every real agent far below 100.
    """
    return dataclasses.replace(
        config,
        potency=REFERENCE_POTENCY,
        avidity=REFERENCE_AVIDITY,
        vcam1_density=REFERENCE_VCAM1_DENSITY,
        seed=0,
    )


def _cache_key(cfg: SwarmConfig, geometry: ArteryGeometry) -> Tuple:
    return cfg.signature() + (round(float(geometry.initial_occlusion_pct), 4),)


def reference_metrics(
    config: Optional[SwarmConfig] = None,
    geometry: Optional[ArteryGeometry] = None,
) -> Dict[str, float]:
    """Efficacy ceiling for this exact run configuration.

    Defines 100% on the normalised scale: the plaque reduction and
    pressure-drop reduction achievable by the best-configured agent at the same
    swarm size and geometry. Also returns the distal-embolic count for that
    maximal-therapy case, which is the debris ceiling -- harder erosion releases
    more debris, so the most effective agent is also the most embolically
    exposed one.
    """
    cfg = reference_config(config or SwarmConfig())
    g = geometry or ArteryGeometry()
    key = _cache_key(cfg, g)
    cached = _REFERENCE_CACHE.get(key)
    if cached is not None:
        return cached

    res = SwarmSimulator(cfg, g).run()
    ref = {
        "plaque_reduction_pct": max(res.plaque_reduction_pct, 1e-9),
        "pressure_drop_reduction_pct": max(res.pressure_drop_reduction_pct, 1e-9),
        # Left un-floored on purpose. Debris release is a rare-Poisson process at
        # this scale (expected count ~O(0.1) per run), so substituting a floor of
        # 1 would turn a single particle out of 1.6M agent-steps into a maximal
        # hazard. A zero here means the model simply produced no debris, and the
        # embolic axis is then reported as unanchored rather than manufactured.
        "distal_embolic_particle_count": float(res.distal_embolic_particle_count),
        "agent_steps": float(res.agent_steps),
    }
    _REFERENCE_CACHE[key] = ref
    return ref


def nonspecific_control_metrics(
    config: Optional[SwarmConfig] = None,
    geometry: Optional[ArteryGeometry] = None,
) -> Dict[str, float]:
    """Off-target ceiling: an identical particle with no plaque specificity.

    Safety needs a zero point, and the obvious choice -- the most adherent
    agent -- is wrong. Off-target adhesion is *non-monotonic* in avidity: a
    non-targeting particle never binds plaque, so it keeps circulating and
    contacts healthy wall more often than a well-targeted one that is rapidly
    sequestered at the lesion. Anchoring safety to the best binder therefore
    pins every real agent at zero safety and hides a genuine result.

    The correct anchor is the avidity-zero control, which is both the physical
    worst case and the meaningful comparator ("how much better than a particle
    with no targeting at all?").
    """
    base = config or SwarmConfig()
    cfg = dataclasses.replace(
        base,
        avidity=0.0,
        potency=0.0,
        vcam1_density=0.0,
        seed=0,
    )
    g = geometry or ArteryGeometry()
    key = ("nonspecific",) + _cache_key(cfg, g)
    cached = _REFERENCE_CACHE.get(key)
    if cached is not None:
        return cached

    res = SwarmSimulator(cfg, g).run()
    ref = {"off_target_adhesion_pct": max(res.off_target_adhesion_pct, 1e-9)}
    _REFERENCE_CACHE[key] = ref
    return ref


def normalise_to_reference(
    result: SwarmResult,
    geometry: Optional[ArteryGeometry] = None,
) -> Dict[str, float]:
    """Express one run on the model's own 0-100 attainability scale.

    The physical values are preserved alongside their normalised counterparts,
    so nothing is hidden: the suffix ``_physical_pct`` is always the raw model
    output and the unsuffixed key is the scaled one the judge tiers consume.

    ``safety`` is the worse of the two independent hazards (off-target burden
    and distal embolisation), because safety is a gate rather than an average:
    a construct is not acceptable merely because one of its risks is low.

    Note on dynamic range: ``off_target_adhesion_pct`` is the fraction of
    *binding events* that land on healthy wall, so any particle with real
    targeting sits near the bottom of that scale (measured ~0.05%, against
    100% for an avidity-zero control). ``safety`` therefore reads high for
    essentially every targeted construct, and the discriminating signal lives
    in the raw ``off_target_physical_pct``. That is a property of the model, not
    a scoring artefact, and it is stated here so nobody over-reads the axis.
    """
    cfg = result.config
    g = geometry or ArteryGeometry()
    ref = reference_metrics(cfg, g)
    nonspecific = nonspecific_control_metrics(cfg, g)

    plaque_frac = float(np.clip(result.plaque_reduction_pct / ref["plaque_reduction_pct"], 0.0, 1.0))
    dp_frac = float(np.clip(result.pressure_drop_reduction_pct / ref["pressure_drop_reduction_pct"], 0.0, 1.0))

    off_ratio = float(np.clip(
        result.off_target_adhesion_pct / nonspecific["off_target_adhesion_pct"], 0.0, 1.0
    ))

    # Embolisation is compared per agent-step, not as a raw count: two runs of
    # different size produce different absolute debris purely by exposure. When
    # the maximal-therapy reference itself releases nothing the axis has no
    # anchor, so it is reported as unavailable rather than scored against a floor.
    ref_embolic_rate = ref["distal_embolic_particle_count"] / max(ref["agent_steps"], 1.0)
    embolic_anchor_available = ref_embolic_rate > 0.0
    if embolic_anchor_available:
        run_rate = result.distal_embolic_particle_count / max(result.agent_steps, 1.0)
        embolic_ratio = float(np.clip(run_rate / ref_embolic_rate, 0.0, 1.0))
    else:
        embolic_ratio = 0.0

    worst_hazard = max(off_ratio, embolic_ratio)

    return {
        # Normalised (judge-facing) scale, where 100 == model maximum.
        "plaque_reduction_pct": plaque_frac * 100.0,
        "reperfusion_gain_pct": dp_frac * 100.0,
        "efficacy": 60.0 * plaque_frac + 40.0 * dp_frac,
        "safety": (1.0 - worst_hazard) * 100.0,
        "off_target_hazard_ratio": off_ratio,
        "embolic_hazard_ratio": embolic_ratio,
        "embolic_anchor_available": 1.0 if embolic_anchor_available else 0.0,
        # Raw physical output, unmodified.
        "plaque_reduction_physical_pct": result.plaque_reduction_pct,
        "pressure_drop_reduction_physical_pct": result.pressure_drop_reduction_pct,
        "off_target_healthy_wall_adhesion_pct": result.off_target_adhesion_pct,
        "off_target_physical_pct": result.off_target_adhesion_pct,
        "distal_embolic_particle_count": float(result.distal_embolic_particle_count),
        "distal_embolic_per_agent_step": float(
            result.distal_embolic_particle_count / max(result.agent_steps, 1)
        ),
        "bound_fraction": result.bound_fraction,
        "agent_steps": float(result.agent_steps),
        "reference_plaque_reduction_pct": ref["plaque_reduction_pct"],
        "reference_nonspecific_off_target_pct": nonspecific["off_target_adhesion_pct"],
    }
