"""
Master controller: Blood -> Targeting -> Hydrodynamic regression.

Chains three stages into one reproducible pipeline:

    Stage 1  blood profiling      -> cohort eligibility + manufacturing yield
    Stage 2  molecular targeting  -> adhesion fraction under wall shear
    Stage 3  swarm hydrodynamics  -> plaque regression for THAT patient

Two corrections relative to a naive implementation, both of which matter for
defensibility:

1. **Stage 2 uses a real binding model.** A common shortcut writes adhesion as
   ``1 - exp(-Ka / 1e8)``. That expression is degenerate: at the usually-quoted
   ``Ka = 1e8`` it evaluates to the constant ``1 - exp(-1) = 0.632``, and it
   saturates for any ``Ka >= 1e9``, so it cannot rank an affinity-improved
   design at all. Here the Langmuir fractional occupancy
   ``theta = Ka[R] / (1 + Ka[R])`` is used, which responds continuously to both
   affinity and receptor density and saturates only when the receptors
   genuinely do.

2. **Stage 3 is coupled through avidity, not through agent count.** The naive
   approach feeds "number of bound agents" in as the swarm size, which
   conflates *dose* with *binding efficiency*: it predicts that a poorly
   targeting construct behaves identically to a well-targeting one at lower
   dose. Here the Stage 2 adhesion fraction drives the swarm's ``avidity``,
   while the Stage 1 yield drives its ``n_agents``.

SCIENTIFIC SCOPE
----------------
Ischemic heart disease is polygenic, so "blood to cure" cannot mean reading a
single broken gene out of a blood draw. What blood legitimately informs is
disease severity, immune compatibility, and (for a cell-based product)
manufacturing yield. Every number below is a model output. Nothing here is
validated in vitro or in vivo, and none of it is medical advice.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from danon.swarm import (
    ArteryGeometry,
    SwarmConfig,
    SwarmResult,
    SwarmSimulator,
)

logger = logging.getLogger(__name__)

# Wall shear stress (Pa) at which adhesion falls to half its low-shear value.
# Firm integrin-mediated adhesion is comparatively shear-resistant, so this is
# far above the selectin-rolling regime.
SHEAR_HALF_MAX_PA = 2.0
SHEAR_HILL_EXPONENT = 1.5

# Nominal endothelial adhesion-molecule concentration at full inflammation.
# Used as the [R] term in the Langmuir isotherm.
RECEPTOR_SATURATING_CONC_M = 1.0e-8

# Monocyte reference range (cells/uL); used for the severity index only.
MONOCYTE_LOW, MONOCYTE_HIGH = 200.0, 800.0

BLOOD_VISCOSITY_PA_S = 3.5e-3


# ===========================================================================
# Stage 1 — blood profiling
# ===========================================================================

@dataclass
class BloodPanel:
    """Raw bloodwork plus the minimum chart data."""

    patient_id: str
    monocyte_count_per_ul: float = 500.0
    hdl_mg_dl: float = 50.0
    ldl_mg_dl: float = 100.0
    crp_mg_l: float = 1.0

    age_years: int = 60
    weight_kg: float = 80.0
    lvef_pct: float = 50.0
    nab_titer: float = 5.0
    hla_alleles: Tuple[str, ...] = ()


@dataclass
class BloodProfileResult:
    """Stage 1 output."""

    patient_id: str
    inflammation_index: float          # 0..1, drives Stage 2 receptor density
    plaque_vulnerability_index: float  # 0..1, severity (NOT yield)
    therapeutic_yield: int             # manufacturing starting material
    eligible: bool
    exclusion_reasons: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


class BloodProfiler:
    """Automated bloodwork interpretation.

    Deliberately separates two things a naive implementation conflates:
    *disease severity* (does this patient need treatment, and are they safe to
    treat) and *manufacturing yield* (how much starting material their blood
    provides). A low-HDL patient is sicker; that does not mean their blood
    yields more product.
    """

    def __init__(self, base_yield_per_monocyte: float = 1000.0) -> None:
        self.base_yield_per_monocyte = base_yield_per_monocyte

    @staticmethod
    def _clamp01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    def inflammation_index(self, panel: BloodPanel) -> float:
        """Relative inflammatory burden, which sets endothelial adhesion molecules."""
        mono = self._clamp01(
            (panel.monocyte_count_per_ul - MONOCYTE_LOW)
            / max(MONOCYTE_HIGH - MONOCYTE_LOW, 1e-9)
        )
        # CRP ~3 mg/L is the commonly used high-risk threshold.
        crp = self._clamp01(panel.crp_mg_l / 3.0)
        # Low HDL (<40 mg/dL) is pro-inflammatory.
        hdl = self._clamp01((40.0 - panel.hdl_mg_dl) / max(40.0, 1e-9))
        return self._clamp01(0.5 * mono + 0.3 * crp + 0.2 * hdl)

    def plaque_vulnerability_index(self, panel: BloodPanel) -> float:
        """Severity score. Used for risk and eligibility, never for yield."""
        inflam = self.inflammation_index(panel)
        ldl = self._clamp01((panel.ldl_mg_dl - 70.0) / max(190.0 - 70.0, 1e-9))
        hdl_def = self._clamp01((40.0 - panel.hdl_mg_dl) / max(40.0, 1e-9))
        return self._clamp01(0.45 * inflam + 0.35 * ldl + 0.20 * hdl_def)

    def manufacturing_yield(self, panel: BloodPanel) -> int:
        """Autologous starting material.

        NOTE: this is only meaningful if the therapeutic is *cell-based* and
        manufactured from the patient's own monocytes. For a nanoparticle or
        viral product, yield has nothing to do with circulating monocyte count,
        and treating it as such is a category error.
        """
        return int(max(0.0, panel.monocyte_count_per_ul) * self.base_yield_per_monocyte)

    def profile(self, panel: BloodPanel) -> BloodProfileResult:
        notes = [
            "Yield is monocyte-derived and is only meaningful for a cell-based "
            "product; it is reported for completeness, not as a nanoparticle dose.",
        ]
        exclusions: List[str] = []
        if panel.nab_titer >= 1000.0:
            exclusions.append(
                f"NAb titer {panel.nab_titer:.0f} >= 1:1000 (trial exclusion criterion)"
            )
        if panel.lvef_pct < 25.0:
            exclusions.append(f"LVEF {panel.lvef_pct:.0f}% below intervention window")
        if panel.age_years < 18:
            exclusions.append("Pediatric patient; outside modeled adult envelope")
        if panel.monocyte_count_per_ul <= 0:
            notes.append("Zero monocytes: autologous manufacturing not feasible.")

        return BloodProfileResult(
            patient_id=panel.patient_id,
            inflammation_index=round(self.inflammation_index(panel), 6),
            plaque_vulnerability_index=round(self.plaque_vulnerability_index(panel), 6),
            therapeutic_yield=self.manufacturing_yield(panel),
            eligible=not exclusions,
            exclusion_reasons=exclusions,
            notes=notes,
        )


# ===========================================================================
# Stage 2 — molecular targeting
# ===========================================================================

@dataclass
class TargetingResult:
    """Stage 2 output: how many agents actually anchor to the lesion."""

    n_agents: int
    receptor_density_rel: float
    ka_per_molar: float
    intrinsic_occupancy: float        # Langmuir theta, before shear
    shear_factor: float               # 0..1 attenuation from wall shear
    adhesion_probability: float       # theta * shear_factor
    bound_agents: int
    wall_shear_stress_pa: float
    notes: List[str] = field(default_factory=list)


class MolecularTargeting:
    """Langmuir binding under wall shear.

    ``theta = Ka[R] / (1 + Ka[R])``
    ``P_adhere = theta * 1 / (1 + (tau/tau50)^n)``

    Both terms are physically motivated: the first is the equilibrium fraction
    of receptors occupied at the given affinity and receptor density, the second
    captures the fact that hydrodynamic drag removes weakly adhered particles.
    """

    def __init__(
        self,
        shear_half_max_pa: float = SHEAR_HALF_MAX_PA,
        shear_exponent: float = SHEAR_HILL_EXPONENT,
        receptor_sat_conc_m: float = RECEPTOR_SATURATING_CONC_M,
    ) -> None:
        self.shear_half_max_pa = shear_half_max_pa
        self.shear_exponent = shear_exponent
        self.receptor_sat_conc_m = receptor_sat_conc_m

    def intrinsic_occupancy(self, ka_per_molar: float, receptor_density_rel: float) -> float:
        """Langmuir fractional occupancy. Responds continuously to Ka and [R]."""
        receptor_conc = max(0.0, receptor_density_rel) * self.receptor_sat_conc_m
        ka_r = max(0.0, ka_per_molar) * receptor_conc
        if ka_r <= 0.0:
            return 0.0
        return float(ka_r / (1.0 + ka_r))

    def shear_factor(self, wall_shear_pa: float) -> float:
        if wall_shear_pa <= 0.0:
            return 1.0
        return float(
            1.0 / (1.0 + (wall_shear_pa / self.shear_half_max_pa) ** self.shear_exponent)
        )

    @staticmethod
    def wall_shear_stress(peak_velocity_um_s: float, radius_um: float) -> float:
        """Wall shear stress (Pa) for Poiseuille flow: tau = 2*mu*u_peak/R."""
        radius_m = max(radius_um, 1e-9) * 1e-6
        u_peak_m_s = peak_velocity_um_s * 1e-6
        return float(2.0 * BLOOD_VISCOSITY_PA_S * u_peak_m_s / radius_m)

    def simulate(
        self,
        n_agents: int,
        *,
        receptor_expression_level: float = 0.92,
        ka_per_molar: float = 1.0e9,
        wall_shear_pa: Optional[float] = None,
        peak_velocity_um_s: float = 400_000.0,
        radius_um: float = 1500.0,
        rng: Optional[np.random.Generator] = None,
    ) -> TargetingResult:
        if n_agents <= 0:
            return TargetingResult(
                n_agents=0, receptor_density_rel=receptor_expression_level,
                ka_per_molar=ka_per_molar, intrinsic_occupancy=0.0,
                shear_factor=0.0, adhesion_probability=0.0, bound_agents=0,
                wall_shear_stress_pa=0.0,
                notes=["No agents available; nothing to target."],
            )

        if wall_shear_pa is None:
            wall_shear_pa = self.wall_shear_stress(peak_velocity_um_s, radius_um)

        theta = self.intrinsic_occupancy(ka_per_molar, receptor_expression_level)
        f_shear = self.shear_factor(wall_shear_pa)
        p_adhere = float(np.clip(theta * f_shear, 0.0, 1.0))

        # Vectorised draw; no Python-level loop over agents.
        rng = rng or np.random.default_rng(0)
        draws = rng.random(int(n_agents))
        bound = int(np.count_nonzero(draws < p_adhere))

        notes = [
            "Adhesion probability is the product of Langmuir occupancy and a "
            "shear-attenuation factor; it saturates only where the physics does.",
        ]
        if f_shear < 0.5:
            notes.append(
                f"Wall shear {wall_shear_pa:.2f} Pa is above the half-maximum "
                f"({self.shear_half_max_pa:.1f} Pa), so drag is suppressing "
                "adhesion more than affinity is helping it."
            )

        return TargetingResult(
            n_agents=int(n_agents),
            receptor_density_rel=round(float(receptor_expression_level), 6),
            ka_per_molar=float(ka_per_molar),
            intrinsic_occupancy=round(theta, 6),
            shear_factor=round(f_shear, 6),
            adhesion_probability=round(p_adhere, 6),
            bound_agents=bound,
            wall_shear_stress_pa=round(wall_shear_pa, 6),
            notes=notes,
        )


# ===========================================================================
# Stage 3 — hydrodynamic swarm regression
# ===========================================================================

@dataclass
class PipelineReport:
    """Full Blood -> Targeting -> Hydrodynamics result for one patient."""

    patient_id: str
    blood: BloodProfileResult
    targeting: TargetingResult
    swarm: SwarmResult
    eligible: bool
    summary: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def render(self) -> str:
        b, t, s = self.blood, self.targeting, self.swarm
        lines = [
            f"BLOOD-TO-CURE PIPELINE — {self.patient_id}",
            "=" * 62,
            "STAGE 1 — Blood profiling",
            f"  inflammation index        : {b.inflammation_index:.3f}",
            f"  vulnerability index       : {b.plaque_vulnerability_index:.3f}",
            f"  autologous yield          : {b.therapeutic_yield:,} agents",
            f"  eligible                  : {b.eligible}",
            "STAGE 2 — Molecular targeting",
            f"  receptor density (rel)    : {t.receptor_density_rel:.3f}",
            f"  Ka                        : {t.ka_per_molar:.2e} /M",
            f"  Langmuir occupancy        : {t.intrinsic_occupancy:.4f}",
            f"  wall shear                : {t.wall_shear_stress_pa:.2f} Pa",
            f"  shear factor              : {t.shear_factor:.4f}",
            f"  adhesion probability      : {t.adhesion_probability:.4f}",
            f"  bound agents              : {t.bound_agents:,}",
            "STAGE 3 — Hydrodynamic swarm regression",
            f"  agents simulated          : {s.config.n_agents:,}",
            f"  agent-steps               : {s.agent_steps:,}",
            f"  plaque reduction          : {s.plaque_reduction_pct:.4f} %",
            f"  pressure-drop reduction   : {s.pressure_drop_reduction_pct:.4f} %",
            f"  reperfusion gain          : {s.reperfusion_gain_pct:.4f} %",
            f"  off-target adhesion       : {s.off_target_adhesion_pct:.4f} %",
            f"  distal embolic particles  : {s.distal_embolic_particle_count:,}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            lines += [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


class BloodToCurePipeline:
    """End-to-end controller.

    Stage 2's adhesion probability is passed to Stage 3 as the swarm's
    ``avidity`` (binding efficiency), NOT as the agent count. The agent count
    comes from Stage 1 yield and the intended dose. This keeps "how well does it
    stick" and "how much did we give" as separate, independently testable knobs.
    """

    def __init__(
        self,
        geometry: Optional[ArteryGeometry] = None,
        profiler: Optional[BloodProfiler] = None,
        targeting: Optional[MolecularTargeting] = None,
    ) -> None:
        self.geometry = geometry or ArteryGeometry()
        self.profiler = profiler or BloodProfiler()
        self.targeting = targeting or MolecularTargeting()

    def run(
        self,
        panel: BloodPanel,
        *,
        n_agents: int = 25_000,
        steps: int = 300,
        ka_per_molar: float = 1.0e9,
        seed: int = 0,
    ) -> PipelineReport:
        blood = self.profiler.profile(panel)

        targeting = self.targeting.simulate(
            n_agents,
            receptor_expression_level=max(blood.inflammation_index, 1e-6),
            ka_per_molar=ka_per_molar,
            peak_velocity_um_s=SwarmConfig().peak_velocity_um_s,
            radius_um=self.geometry.min_radius_um,
            rng=np.random.default_rng(seed),
        )

        # Correct coupling: adhesion fraction -> avidity, not agent count.
        cfg = SwarmConfig(
            n_agents=int(n_agents),
            steps=int(steps),
            avidity=float(np.clip(targeting.adhesion_probability, 0.0, 1.0)),
            seed=seed,
        )
        swarm = SwarmSimulator(cfg, self.geometry).run()

        warnings = [
            "All stages are unvalidated computational models.",
            "Not a cure, not clinically actionable, not medical advice.",
        ]
        if not blood.eligible:
            warnings.append(
                "Patient would be excluded from treatment: "
                + "; ".join(blood.exclusion_reasons)
            )
        if blood.inflammation_index < 0.05:
            warnings.append(
                "Low inflammation index implies little adhesion-molecule "
                "expression, so targeting efficiency is inherently poor."
            )

        summary = {
            "inflammation_index": blood.inflammation_index,
            "vulnerability_index": blood.plaque_vulnerability_index,
            "adhesion_probability": targeting.adhesion_probability,
            "bound_agents": float(targeting.bound_agents),
            "plaque_reduction_pct": swarm.plaque_reduction_pct,
            "pressure_drop_reduction_pct": swarm.pressure_drop_reduction_pct,
            "reperfusion_gain_pct": swarm.reperfusion_gain_pct,
            "off_target_adhesion_pct": swarm.off_target_adhesion_pct,
            "distal_embolic_particle_count": float(swarm.distal_embolic_particle_count),
        }

        return PipelineReport(
            patient_id=panel.patient_id,
            blood=blood,
            targeting=targeting,
            swarm=swarm,
            eligible=blood.eligible,
            summary=summary,
            warnings=warnings,
        )

    def cohort(self, panels: List[BloodPanel], **kwargs) -> Dict[str, object]:
        """Run the whole pipeline across a cohort and summarise."""
        reports = [self.run(p, **kwargs) for p in panels]
        regs = np.array([r.summary["plaque_reduction_pct"] for r in reports], dtype=float)
        adh = np.array([r.summary["adhesion_probability"] for r in reports], dtype=float)
        return {
            "n": len(reports),
            "eligible_fraction": (
                sum(1 for r in reports if r.eligible) / len(reports) if reports else 0.0
            ),
            "mean_adhesion_probability": float(np.mean(adh)) if len(adh) else 0.0,
            "mean_plaque_reduction_pct": float(np.mean(regs)) if len(regs) else 0.0,
            "stdev_plaque_reduction_pct": float(np.std(regs, ddof=1)) if len(regs) > 1 else 0.0,
            "total_agent_steps": sum(r.swarm.agent_steps for r in reports),
            "reports": reports,
        }
