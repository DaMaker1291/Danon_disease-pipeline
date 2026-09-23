"""
Parametric robustness of the adaptive collateral-switch policy.

WHAT THIS MODULE ANSWERS
------------------------
Every growth, kill, transport and cost constant in the engine is ASSUMED
(bench calibration is the missing tier). So the honest question is not
"does the policy win?" (it does, measured) but "how much of that win
survives if my assumed constants are wrong?" This module sweeps each
load-bearing constant across a scale grid, re-runs the PAIRED experiment
(policy vs MTD at equal seeds, same perturbed world, same geometry, same
r0) and reports, per perturbation, whether the policy still wins.

WHAT IT CANNOT ANSWER
---------------------
It is a structural robustness statement: it says WHICH constants the
conclusion is sensitive to, and by how much. It cannot say whether the
assumed constants are the true ones -- perturbing a wrong number by
+/-50% and finding the answer stable tells you the conclusion is
insensitive to that number, not that the number is right. Constants that
turn out to matter must be measured (danon.validation tier 3).

ENDPOINTS (one always defined, one often not)
---------------------------------------------
PRIMARY: mean total tumour burden over the horizon, paired per seed, reported
as a median policy/MTD ratio. Every replicate has one, so no cell is ever
silently dropped -- a cell where the policy leaves more tumour is a loss.

SECONDARY: progression-free survival. Informative only where BOTH arms
progressed inside the horizon. A policy arm that never progresses has no
median, and comparing its censored value against the control's would be a
manufactured win -- such cells report "pfs: no info" rather than a result.

PERTURBATION TARGETS (explicit semantics, not just numbers)
-----------------------------------------------------------
tumour_growth      scale growth_S and growth_R by the same factor: the
                   proliferation timescale moves, the resistance cost
                   RATIO is preserved.
drug_a_kill        scale Drug A's kill rate on BOTH clones:
                   the selective pressure that breeds resistance.
collateral_b_kill  scale Drug B's kill rate on R only: the collateral
                   sensitivity the whole strategy is built on. This is
                   the parameter the method's validity actually rests on.
drug_transport     scale both drugs' diffusion coefficient: how deep
                   drug penetrates, i.e. how big the sanctuary pockets are.
resistance_cost    scale the fitness penalty R pays, at fixed growth_S:
                   a cost-free resistant clone would break the mechanism
                   entirely (there would be nothing to exploit).
intake_r0          scale the patient's initial resistant fraction
                   (clamped to a representable range): how much the
                   schedule depends on getting intake right.

The scale grid includes 1.0, so every sweep contains the unperturbed
reference cell its other cells are read against.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.evosteer import (
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    CloneParams,
    CollateralSwitchPolicy,
    DrugParams,
    MTDPolicy,
    TumorEvolutionSimulator,
    TumorGeometry,
)


# ---------------------------------------------------------------------------
# Perturbation semantics
# ---------------------------------------------------------------------------


TARGETS: Tuple[str, ...] = (
    "tumour_growth",
    "drug_a_kill",
    "collateral_b_kill",
    "drug_transport",
    "resistance_cost",
    "intake_r0",
)

DEFAULT_SCALES: Tuple[float, ...] = (0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0)

# A policy that leaves >= 75% of the control's mean burden has not materially
# changed tumour control, even though it is technically "better". The
# threshold is a modelling choice, stated here and in every report, so a
# reader can move it and re-read the same table.
MATERIAL_EFFECT_RATIO: float = 0.75

_TARGET_MEANING: Dict[str, str] = {
    "tumour_growth": "proliferation timescale (growth_S, growth_R scaled together)",
    "drug_a_kill": "Drug A selective pressure (kill on both clones)",
    "collateral_b_kill": "Drug B kill on the resistant clone (the mechanism)",
    "drug_transport": "drug diffusion / penetration depth",
    "resistance_cost": "fitness penalty paid by the resistant clone",
    "intake_r0": "patient's initial resistant fraction (intake uncertainty)",
}


def perturb(
    target: str,
    scale: float,
    clones: CloneParams = DEFAULT_CLONES,
    drugs: DrugParams = DEFAULT_DRUGS,
    r0: float = 0.10,
) -> Tuple[CloneParams, DrugParams, float]:
    """Return (clones, drugs, r0) with `target` scaled by `scale`.

    Unknown targets and non-positive scales raise; a perturbation that
    would produce unrepresentable parameters (negative growth, saturating
    cost) is clamped only where the clamp is physically meaningful and
    documented, never silently.
    """
    if target not in TARGETS:
        raise ValueError(f"unknown perturbation target {target!r}; known: {TARGETS}")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"scale must be finite and > 0, got {scale!r}")

    if target == "tumour_growth":
        return (
            dc_replace(
                clones,
                growth_S=clones.growth_S * scale,
                growth_R=clones.growth_R * scale,
            ),
            drugs,
            r0,
        )

    if target == "drug_a_kill":
        return (
            clones,
            dc_replace(
                drugs,
                kill_max_S=(drugs.kill_max_S[0] * scale, drugs.kill_max_S[1]),
                kill_max_R=(drugs.kill_max_R[0] * scale, drugs.kill_max_R[1]),
            ),
            r0,
        )

    if target == "collateral_b_kill":
        return (
            clones,
            dc_replace(drugs, kill_max_R=(drugs.kill_max_R[0], drugs.kill_max_R[1] * scale)),
            r0,
        )

    if target == "drug_transport":
        return (
            clones,
            dc_replace(
                drugs,
                diffusion=(drugs.diffusion[0] * scale, drugs.diffusion[1] * scale),
            ),
            r0,
        )

    if target == "resistance_cost":
        base_cost = 1.0 - (clones.growth_R / clones.growth_S)
        if base_cost <= 0.0:
            raise ValueError(
                "resistance_cost perturbation needs a positive base cost "
                f"(growth_R/growth_S = {clones.growth_R / clones.growth_S:.3f})"
            )
        scaled_cost = min(0.95, base_cost * scale)  # cost >= 1 is not representable
        return (
            dc_replace(clones, growth_R=clones.growth_S * (1.0 - scaled_cost)),
            drugs,
            r0,
        )

    # intake_r0: bounded to a representable resistant fraction.
    scaled_r0 = float(min(0.5, max(1e-4, r0 * scale)))
    return clones, drugs, scaled_r0


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerturbedOutcome:
    """One target x scale cell of the sweep, both arms, paired seeds.

    Two endpoints are carried because one of them can always be computed and
    the other cannot:

    * TUMOUR CONTROL -- mean total burden over the horizon, per replicate.
      Always defined, so it is the primary criterion: a cell where the
      policy leaves more tumour than MTD is a loss, full stop.
    * PFS -- progression time, censored at the horizon. Informative only
      when BOTH arms progressed (a fully controlled policy arm has no
      median); those cells are reported as non-discriminating on PFS rather
      than scored as a win or a loss.
    """

    target: str
    scale: float
    policy_pfs: Tuple[float, ...]
    mtd_pfs: Tuple[float, ...]
    policy_mean_burden: Tuple[float, ...]
    mtd_mean_burden: Tuple[float, ...]
    policy_median_pfs: float
    mtd_median_pfs: float
    policy_toxicity: float
    mtd_toxicity: float
    # The horizon is carried on the cell so a censored replicate can never be
    # mistaken for an event (and therefore for a win) anywhere downstream.
    horizon_days: float = 365.0

    @property
    def policy_events(self) -> int:
        """Replicates whose progression happened INSIDE the horizon."""
        return int(sum(1 for t in self.policy_pfs if t < self.horizon_days))

    @property
    def mtd_events(self) -> int:
        return int(sum(1 for t in self.mtd_pfs if t < self.horizon_days))

    @property
    def pfs_discriminating(self) -> bool:
        """True when BOTH arms recorded at least one progression."""
        return self.policy_events > 0 and self.mtd_events > 0

    @property
    def extension(self) -> Optional[float]:
        """PFS extension (policy - MTD)/MTD; None when PFS cannot discriminate."""
        if not self.pfs_discriminating or self.mtd_median_pfs <= 0:
            return None
        return (self.policy_median_pfs - self.mtd_median_pfs) / self.mtd_median_pfs

    @property
    def paired_burden_ratios(self) -> Tuple[float, ...]:
        """Per-seed policy/MTD mean-burden ratio (< 1 = policy controls better)."""
        out = []
        for p, m in zip(self.policy_mean_burden, self.mtd_mean_burden):
            if m <= 0.0:
                out.append(1.0 if p <= 0.0 else float("inf"))
            else:
                out.append(p / m)
        return tuple(out)

    @property
    def burden_ratio(self) -> float:
        """Median paired burden ratio: the primary, always-defined criterion."""
        return float(np.median(self.paired_burden_ratios))

    @property
    def policy_wins(self) -> bool:
        """Primary criterion: policy leaves less tumour than MTD at equal seeds."""
        return self.burden_ratio < 1.0

    @property
    def material_win(self) -> bool:
        """Policy leaves <= MATERIAL_EFFECT_RATIO of the control's mean burden."""
        return self.burden_ratio <= MATERIAL_EFFECT_RATIO

    @property
    def toxicity_reduction(self) -> Optional[float]:
        if self.mtd_toxicity <= 0:
            return None
        return (self.mtd_toxicity - self.policy_toxicity) / self.mtd_toxicity


@dataclass
class SensitivityReport:
    """The full sweep plus the statements it does and does not support."""

    case_id: str
    r0: float
    days: int
    seeds: Tuple[int, ...]
    scales: Tuple[float, ...]
    outcomes: List[PerturbedOutcome] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    # -- aggregation --------------------------------------------------------

    def by_target(self) -> Dict[str, List[PerturbedOutcome]]:
        out: Dict[str, List[PerturbedOutcome]] = {}
        for o in self.outcomes:
            out.setdefault(o.target, []).append(o)
        for v in out.values():
            v.sort(key=lambda o: o.scale)
        return out

    def n_cells(self) -> int:
        return len(self.outcomes)

    def n_cells_pfs_discriminating(self) -> int:
        return sum(1 for o in self.outcomes if o.pfs_discriminating)

    def n_cells_policy_wins(self) -> int:
        """Cells where the policy leaves less tumour than MTD (primary criterion)."""
        return sum(1 for o in self.outcomes if o.policy_wins)

    def n_cells_material_wins(self) -> int:
        """Cells where the policy clears the material-effect threshold."""
        return sum(1 for o in self.outcomes if o.material_win)

    def n_cells_pfs_wins(self) -> int:
        """Cells where PFS could be compared AND the policy's was longer."""
        return sum(
            1
            for o in self.outcomes
            if o.pfs_discriminating and o.policy_median_pfs > o.mtd_median_pfs
        )

    def break_scale(self, target: str) -> Optional[float]:
        """Highest scale (< 1.0) at which the policy stops MATTERING.

        The operational statement: "the strategy keeps a material tumour-
        control advantage until this constant falls to X% of its assumed
        value". Uses the material-effect threshold, not the bare
        "any improvement counts" comparison, because a 9% burden reduction
        is not a therapy. Returns None when every sub-reference cell is a
        material win.
        """
        cells = [o for o in self.by_target().get(target, []) if o.scale < 1.0]
        broken = [o.scale for o in cells if not o.material_win]
        return max(broken) if broken else None

    @staticmethod
    def effect(o: PerturbedOutcome) -> float:
        """Effect size on the primary endpoint: 1 - burden ratio (>0 = policy better).

        Defined for EVERY cell (unlike PFS extension), so ranking targets by
        this number never silently drops the cells that matter most -- the
        ones where the policy controls the tumour outright.
        """
        return 1.0 - o.burden_ratio

    def reference_effect(self) -> Optional[float]:
        """Effect at scale 1.0 of the mechanism target, else of any scale-1.0 cell."""
        for o in self.outcomes:
            if o.scale == 1.0 and o.target == "collateral_b_kill":
                return self.effect(o)
        for o in self.outcomes:
            if o.scale == 1.0:
                return self.effect(o)
        return None

    def swing(self, target: str) -> Optional[float]:
        """Largest |effect - reference effect| within a target's row.

        Ranks targets by how much the conclusion moves when they move: the
        headline robustness number.
        """
        cells = self.by_target().get(target, [])
        ref = next((self.effect(o) for o in cells if o.scale == 1.0), None)
        if ref is None:
            return None
        return max(abs(self.effect(o) - ref) for o in cells) if cells else None

    def failures(self) -> List[PerturbedOutcome]:
        """Cells where the policy did NOT beat MTD on the primary criterion."""
        return [o for o in self.outcomes if not o.policy_wins]

    def most_sensitive_target(self) -> Optional[str]:
        ranked = [(self.swing(t), t) for t in self.by_target()]
        ranked = [(s, t) for s, t in ranked if s is not None]
        if not ranked:
            return None
        return max(ranked)[1]

    def failure_scales(self, target: str) -> List[float]:
        """Every scale in a target's row where the policy lost on burden."""
        return [
            o.scale for o in self.by_target().get(target, []) if not o.policy_wins
        ]

    # -- presentation -------------------------------------------------------

    def summary_table(self) -> str:
        lines = [
            f"PARAMETRIC ROBUSTNESS SWEEP -- {self.case_id}, r0={self.r0:.3f}, "
            f"{self.days} d horizon, {len(self.seeds)} paired seeds",
            f"primary criterion (mean tumour burden, always defined): "
            f"policy better in {self.n_cells_policy_wins()}/{self.n_cells()} cells, "
            f"materially better in {self.n_cells_material_wins()}/{self.n_cells()}",
            f"material-effect threshold: burden ratio <= {MATERIAL_EFFECT_RATIO:.2f}",
            "",
            f"{'target':<18s} {'scale':>6s} {'pol burden':>11s} {'mtd burden':>11s} "
            f"{'ratio':>6s} {'tox':>7s}  verdict",
        ]
        for target, cells in self.by_target().items():
            for o in cells:
                tox = "n/a" if o.toxicity_reduction is None else f"{o.toxicity_reduction:.1%}"
                pol_b = float(np.median(o.policy_mean_burden))
                mtd_b = float(np.median(o.mtd_mean_burden))
                if not o.material_win:
                    verdict = "lost material control" if not o.policy_wins else "marginal only"
                else:
                    verdict = "material win"
                lines.append(
                    f"{target:<18s} {o.scale:>6.2f} {pol_b:>11.2f} {mtd_b:>11.2f} "
                    f"{o.burden_ratio:>6.2f} {tox:>7s}  {verdict}"
                )
        lines.append("")
        lines.append(f"most sensitive target: {self.most_sensitive_target()}")
        lines.append(
            f"PFS (secondary) comparable in {self.n_cells_pfs_discriminating()}/"
            f"{self.n_cells()} cells; policy longer in {self.n_cells_pfs_wins()}"
        )
        for target in self.by_target():
            b = self.break_scale(target)
            if b is not None:
                lines.append(
                    f"  {target}: material control lost at or below scale {b:.2f} "
                    f"({_TARGET_MEANING.get(target, '')})"
                )
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)

    def to_records(self) -> List[Dict[str, object]]:
        return [
            {
                "case_id": self.case_id,
                "target": o.target,
                "scale": o.scale,
                "policy_median_pfs": o.policy_median_pfs,
                "mtd_median_pfs": o.mtd_median_pfs,
                "extension": o.extension,
                "toxicity_reduction": o.toxicity_reduction,
                "policy_wins": o.policy_wins,
                "burden_ratio": o.burden_ratio,
                "paired_burden_ratios": list(o.paired_burden_ratios),
                "pfs_discriminating": o.pfs_discriminating,
                "policy_events": o.policy_events,
                "mtd_events": o.mtd_events,
                "policy_pfs": list(o.policy_pfs),
                "mtd_pfs": list(o.mtd_pfs),
                "policy_mean_burden": list(o.policy_mean_burden),
                "mtd_mean_burden": list(o.mtd_mean_burden),
            }
            for o in self.outcomes
        ]


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------


def run_sensitivity(
    geometry: TumorGeometry,
    r0: float,
    case_id: str = "geometry",
    targets: Sequence[str] = TARGETS,
    scales: Sequence[float] = DEFAULT_SCALES,
    days: int = 240,
    seeds: Sequence[int] = (101, 102),
    theta_on: float = 0.15,
    theta_off: float = 0.05,
    clones: CloneParams = DEFAULT_CLONES,
    drugs: DrugParams = DEFAULT_DRUGS,
    log: Optional[callable] = None,
) -> SensitivityReport:
    """Run the paired policy-vs-MTD experiment on a perturbed-parameter grid.

    Both arms see the SAME perturbed world at the SAME seed, so each cell is
    a paired comparison, not two independent samples. The 1.0 scale is the
    unperturbed reference; every other cell is read against it.
    """
    if days < 2:
        raise ValueError("need at least 2 days")
    if not seeds:
        raise ValueError("need at least one seed")
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        raise ValueError(f"unknown perturbation targets: {unknown}")

    report = SensitivityReport(
        case_id=case_id,
        r0=r0,
        days=days,
        seeds=tuple(seeds),
        scales=tuple(scales),
        notes=[
            "both arms share the perturbed world and the seeds: each cell is "
            "paired, so the sign of the difference is more informative than "
            "either median",
            "medians over few replicates are coarse; the load-bearing claim "
            "is the sign pattern across the grid, not any single cell's value",
            "the horizon must be long enough for the MTD control to progress; "
            "cells where neither arm progressed are reported as "
            "non-discriminating rather than scored as losses",
            "the MTD control moves with the perturbation too -- this answers "
            "'does the policy still beat standard dosing in this world', not "
            "'does this world behave like a patient'",
            "constant values remain ASSUMED; robustness to a wrong constant "
            "is not evidence the constant is right",
        ],
    )

    for target in targets:
        for scale in scales:
            p_clones, p_drugs, p_r0 = perturb(target, scale, clones, drugs, r0)
            sim = TumorEvolutionSimulator(geometry, clones=p_clones, drugs=p_drugs)
            policy_pfs: List[float] = []
            mtd_pfs: List[float] = []
            policy_burden: List[float] = []
            mtd_burden: List[float] = []
            policy_tox: List[float] = []
            mtd_tox: List[float] = []
            for seed in seeds:
                traj_p = sim.run(
                    CollateralSwitchPolicy(theta_on, theta_off),
                    days=days,
                    initial_resistant_fraction=p_r0,
                    seed=seed,
                )
                traj_m = sim.run(
                    MTDPolicy(),
                    days=days,
                    initial_resistant_fraction=p_r0,
                    seed=seed,
                )
                policy_pfs.append(
                    float(traj_p.pfs_day) if traj_p.pfs_day is not None else float(days)
                )
                mtd_pfs.append(
                    float(traj_m.pfs_day) if traj_m.pfs_day is not None else float(days)
                )
                policy_burden.append(
                    float(np.mean(traj_p.burden_S + traj_p.burden_R))
                )
                mtd_burden.append(float(np.mean(traj_m.burden_S + traj_m.burden_R)))
                policy_tox.append(float(traj_p.toxicity))
                mtd_tox.append(float(traj_m.toxicity))

            outcome = PerturbedOutcome(
                target=target,
                scale=float(scale),
                policy_pfs=tuple(policy_pfs),
                mtd_pfs=tuple(mtd_pfs),
                policy_mean_burden=tuple(policy_burden),
                mtd_mean_burden=tuple(mtd_burden),
                policy_median_pfs=float(np.median(policy_pfs)),
                mtd_median_pfs=float(np.median(mtd_pfs)),
                policy_toxicity=float(np.mean(policy_tox)),
                mtd_toxicity=float(np.mean(mtd_tox)),
                horizon_days=float(days),
            )
            report.outcomes.append(outcome)
            if log is not None:
                ext = "n/a" if outcome.extension is None else f"{outcome.extension:+.1%}"
                verdict = (
                    "material win"
                    if outcome.material_win
                    else ("marginal" if outcome.policy_wins else "LOSES")
                )
                log(
                    f"  {target:<18s} x{scale:<5.2f} burden ratio "
                    f"{outcome.burden_ratio:>6.2f}  PFS {outcome.policy_median_pfs:>5.0f} "
                    f"vs {outcome.mtd_median_pfs:>5.0f} d (ext {ext:>8s})  {verdict}"
                )
    return report


# ---------------------------------------------------------------------------
# Honesty record
# ---------------------------------------------------------------------------


def sensitivity_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "paired_design_equal_seeds": EvidenceStatus.ESTABLISHED,
        "perturbation_semantics": EvidenceStatus.ESTABLISHED,   # defined in code, enforceable
        "assumed_constant_values": EvidenceStatus.UNKNOWN,       # inherited from evosteer
        "collateral_sensitivity_in_this_tumour": EvidenceStatus.UNKNOWN,  # bench-measurable
        "sweep_replaces_bench_calibration": EvidenceStatus.PROVEN_FAILED,
    }


def what_sensitivity_cannot_prove() -> List[str]:
    return [
        "that the assumed constants are correct: perturbing a wrong value "
        "and finding stability shows insensitivity, not validity",
        "patient-level outcomes: every cell is a model run, not a trial",
        "that the collateral mechanism exists in human GBM in vivo: it is an "
        "in-vitro-documented phenomenon whose in-vivo magnitude is UNKNOWN",
        "statistical power: with few replicates the medians are coarse and "
        "adjacent cells may be indistinguishable noise",
    ]
