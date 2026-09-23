"""
EvoTwin: cohort-level evolutionary-trap analysis on the EvoSteer engine.

WHAT THIS ADDS OVER danon.evosteer
----------------------------------
evosteer.py answers a single-tumour question: given one geometry and one
policy, does steering beat continuous MTD? evotwin.py answers the cohort
question: across a simulated patient cohort with patient-varying
parameters, what fraction achieves LONG-TERM DISEASE CONTROL, and does
pairing the steering schedule with a (weak, GBM-realistic) immune
clearance layer change the FAILURE MODE of relapse?

THE EVOLUTIONARY TRAP, STATED PRECISELY
---------------------------------------
The trap has two jaws, and both are required:
  Jaw 1 (competition): dose modulation keeps sensitive cells alive, and
      sensitive cells crowd the resistant clone through asymmetric
      competition. Pure MTD removes the suppressor.
  Jaw 2 (immune): a resistance-blind, density-gated immune kill converts
      the resistant clone's fitness cost into an ABSOLUTE decline whenever
      its local density drops. Drug pressure alone can only hold R in
      check relative to S; the immune layer is what removes it.

Measured on this engine (size-16 phantom, 400 days, seeds 7/8):
  MTD             -> progression, final R fraction 1.000 (resistant relapse)
  MTD + immune    -> burden controlled, final R fraction 1.000 at k=0.25
                     (a fully DRUG-RESISTANT controlled tumour), or
                     eradication at k=0.45
  adaptive        -> progression, final R fraction 1.000
  adaptive+immune -> R ELIMINATED (0.000) at k>=0.45 with S under control;
                     at weak k=0.25, relapse is S-dominant (R=0.23)
The defensible claim is a FAILURE-MODE ASYMMETRY: with a weak immune
layer, MTD's relapse is untreatable (drug-resistant) while adaptive
dosing's relapse is drug-sensitive and re-treatable.

THE EVIDENCE BASIS IS DELIBERATELY ASYMMETRIC
---------------------------------------------
  - The COMPETITION jaw is anchored to the mCRPC adaptive-therapy trial
    (33.5 vs 14.3 months, ratio 2.34). Established dynamics class.
  - The IMMUNE jaw is a HYPOTHESIS. Checkpoint monotherapy failed in
    recurrent GBM (CheckMate 143: nivolumab 9.8 vs bevacizumab 10.0
    months, HR 1.04, p=0.76 -- Omuro et al., JAMA Oncology 2020, n=369),
    GBM is immune-cold (low TMB), and no trial of adaptive dosing plus
    immune activation exists. The module reports what the hypothesis does
    IN SILICO and labels it assumed -- it does not claim clinical support.
    The bench-measurable quantities are named.

WHAT IS MEASURED, WHAT IS ASSUMED
---------------------------------
LITERATURE: adaptive/standard TTP ratio 2.34; RANO 25% progression
fraction; CheckMate 143 failure; GBM recurrent median OS ~9.8 months
(sanity context only, never a model output).
DERIVED: every cohort statistic (PFS medians, KM estimates, remission
fractions, failure-mode counts) is computed from the simulated cohort by
standard estimators -- nothing is hardcoded from a target value.
ASSUMED (bench-measurable): all growth/kill/diffusion constants (tier-3
wet-lab campaign, see danon.validation), immune kill rate and
antigenicity (co-culture assays), and the dose-interval grid the
optimizer searches.

SCOPE -- deliberately excluded
------------------------------
- No MONAI/SegResNet segmentation and no DICOM intake: the engine consumes
  a TumorGeometry; segmentation is an upstream preprocessing stage. The
  phantom is a stand-in, not an MRI reconstruction.
- No PhysiCell, no OpenXR/ParaView/VR: presentation layers. Spatial
  fields are numpy arrays; a drug gradient renders identically anywhere.
- No real TCGA/TCIA data: parameters vary over ASSUMED population ranges.
- No third double-resistant clone, no immune escape (antigen-loss
  variants), no T-cell exhaustion dynamics: each is listed in
  what_this_model_cannot_know().
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.evosteer import (
    ADAPTIVE_TTP_RATIO_ANCHOR,
    AdaptiveDosePolicy,
    BasePolicy,
    CloneParams,
    CollateralSwitchPolicy,
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    DrugParams,
    ImmuneParams,
    MTDPolicy,
    TumorEvolutionSimulator,
    make_gbm_phantom,
    what_this_model_cannot_know as evosteer_limits,
)

# ---------------------------------------------------------------------------
# Literature anchors
# ---------------------------------------------------------------------------

#: CheckMate 143 (Omuro et al., JAMA Oncology 2020; n=369): single-agent
#: PD-1 blockade in recurrent GBM did NOT improve survival.
CHECKMATE143_MOS_MONTHS: Tuple[float, float] = (9.8, 10.0)   # nivolumab, bevacizumab
CHECKMATE143_HR: float = 1.04

#: GBM standard-of-care context: median OS at first recurrence ~9-10
#: months. Sanity context for the simulated horizon only.
GBM_RECURRENT_MEDIAN_OS_MONTHS: float = 9.8

#: Long-term control definition: burden held below the RANO progression
#: threshold for >= 2 years (730 days) of simulated follow-up.
LONG_TERM_CONTROL_DAYS: int = 730

#: Neoantigen factor population range (ASSUMED; GBM immune-cold context).
NEOANTIGEN_RANGE: Tuple[float, float] = (0.15, 0.45)

# ---------------------------------------------------------------------------
# Kaplan-Meier estimator (pure numpy, no scipy dependency)
# ---------------------------------------------------------------------------


def kaplan_meier(
    times: Sequence[float],
    events: Sequence[int],
    *,
    eval_times: Optional[Sequence[float]] = None,
) -> Dict[str, object]:
    """Kaplan-Meier survival estimate with Greenwood standard errors.

    times  : follow-up time (event time if event=1, censoring time if 0)
    events : 1 = progression/death observed, 0 = censored
    eval_times : grid on which to report S(t); default = all distinct times

    Returns {"t": [...], "S": [...], "se": [...], "n_at_risk": [...]}.
    Censored observations contribute to the at-risk count at their removal
    time but not to the hazard numerator.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    if len(t) != len(e) or len(t) == 0:
        raise ValueError("times and events must be same nonzero length")
    if not np.all((e == 0) | (e == 1)):
        raise ValueError("events must be 0/1")
    if np.any(t < 0):
        raise ValueError("times must be non-negative")

    order = np.argsort(t, kind="stable")
    t_s, e_s = t[order], e[order]

    # Build the EXACT step function over event times only. Deaths between
    # grid points must still move the curve: an eval grid (e.g. a linspace)
    # almost never coincides with event times, and counting deaths only at
    # grid points silently flattens the curve and hides every median.
    S = 1.0
    cum_var_term = 0.0
    step_t: List[float] = [0.0]
    step_S: List[float] = [1.0]
    step_se: List[float] = [0.0]
    for te in np.unique(t_s[e_s == 1]):
        d = float(np.sum((t_s == te) & (e_s == 1)))
        n_risk = float(np.sum(t_s >= te))
        S *= 1.0 - d / n_risk
        if n_risk > d:
            cum_var_term += d / (n_risk * (n_risk - d))
        step_t.append(float(te))
        step_S.append(S)
        step_se.append(float(S * math.sqrt(cum_var_term)) if cum_var_term > 0 else 0.0)

    grid = (
        np.asarray(eval_times, dtype=float)
        if eval_times is not None
        else np.asarray(step_t, dtype=float)
    )
    idx = np.searchsorted(np.asarray(step_t), grid, side="right") - 1
    idx = np.clip(idx, 0, len(step_t) - 1)
    out_t = [float(g) for g in grid]
    out_S = [float(np.asarray(step_S)[i]) for i in idx]
    out_se = [float(np.asarray(step_se)[i]) for i in idx]
    out_n = [float(np.sum(t_s >= g)) for g in grid]
    return {"t": out_t, "S": out_S, "se": out_se, "n_at_risk": out_n}


def cohort_km_curve(
    pfs_days: Sequence[float],
    progressed: Sequence[bool],
    horizon: float,
    n_eval: int = 61,
) -> Dict[str, object]:
    """KM over a cohort: event = progression (1 at its PFS day);
    censored = controlled at the horizon (0 at the horizon)."""
    if len(pfs_days) != len(progressed) or len(pfs_days) == 0:
        raise ValueError("pfs_days and progressed must be same nonzero length")
    events = [1 if p else 0 for p in progressed]
    times = [min(float(p), float(horizon)) for p in pfs_days]
    return kaplan_meier(times, events, eval_times=np.linspace(0.0, horizon, n_eval))


def km_at(km: Dict[str, object], t: float) -> Tuple[float, float]:
    """(S, se) of a KM curve at time t (step function, right-continuous)."""
    ts = np.asarray(km["t"], dtype=float)
    Ss = np.asarray(km["S"], dtype=float)
    ses = np.asarray(km["se"], dtype=float)
    if t < ts[0]:
        return 1.0, 0.0
    idx = int(np.searchsorted(ts, t, side="right")) - 1
    return float(Ss[idx]), float(ses[idx])


def km_median(km: Dict[str, object]) -> Optional[float]:
    """First time at which S(t) <= 0.5, else None (median not reached)."""
    ts = np.asarray(km["t"], dtype=float)
    Ss = np.asarray(km["S"], dtype=float)
    for t, s in zip(ts, Ss):
        if s <= 0.5:
            return float(t)
    return None


# ---------------------------------------------------------------------------
# Cohort definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CohortParams:
    """Population ranges for patient-varying parameters (ASSUMED ranges)."""

    size: int = 60
    tumor_size_range: Tuple[float, float] = (14.0, 22.0)   # phantom edge
    r0_range: Tuple[float, float] = (0.005, 0.05)          # initial R fraction
    neoantigen_range: Tuple[float, float] = NEOANTIGEN_RANGE
    immune_kill_range: Tuple[float, float] = (0.20, 0.60)
    growth_scale_range: Tuple[float, float] = (0.8, 1.2)
    seed0: int = 1000


@dataclass
class PatientOutcome:
    patient_id: int
    policy: str
    pfs_days: float
    progressed: bool
    final_resistant_fraction: float
    toxicity: float
    immune_used: bool
    tumor_size: int
    initial_resistant_fraction: float


@dataclass
class ArmSummary:
    policy: str
    n: int
    median_pfs_days: float
    pfs_iqr: Tuple[float, float]
    progressed_frac: float
    long_term_control_frac: float
    median_toxicity: float
    relapse_resistant_frac: Optional[float] = None   # R-dominant at horizon
    relapse_sensitive_frac: Optional[float] = None   # R < 0.5 at horizon
    median_final_R_controlled: Optional[float] = None  # R fraction among patients
    # whose burden never crossed the progression threshold. Burden control
    # is NOT clone control: an MTD+immune patient can be 'controlled' while
    # sitting on a fully drug-resistant tumour.


@dataclass
class EvotwinReport:
    arms: Dict[str, ArmSummary]
    days: int
    cohort_size: int
    immune_used: bool
    km_curves: Dict[str, Dict[str, object]]
    km_median_by_arm: Dict[str, Optional[float]]
    logrank: Optional[Dict[str, float]]
    failure_mode_table: Dict[str, Dict[str, float]]
    verdict: str
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Cohort simulator
# ---------------------------------------------------------------------------


class PatientCohortSimulator:
    """Runs one policy over a simulated patient cohort on the EvoSteer engine.

    Patient variation: tumour size, initial resistant fraction, immune
    parameters (when the immune layer is on) and a global growth scale are
    drawn per patient from CohortParams ranges; geometry seed follows the
    patient id so the same cohort can be replayed under any policy.
    """

    def __init__(
        self,
        cohort: CohortParams,
        clones: CloneParams = DEFAULT_CLONES,
        drugs: DrugParams = DEFAULT_DRUGS,
        sim_days: int = 400,
        immune: Optional[ImmuneParams] = None,
    ) -> None:
        self.cohort = cohort
        self.clones = clones
        self.drugs = drugs
        self.sim_days = sim_days
        self.immune = immune

    def run(self, policy: BasePolicy, policy_name: str) -> List[PatientOutcome]:
        outcomes: List[PatientOutcome] = []
        rng = np.random.default_rng(self.cohort.seed0)
        for pid in range(self.cohort.size):
            size = int(
                rng.integers(
                    self.cohort.tumor_size_range[0],
                    self.cohort.tumor_size_range[1] + 1,
                )
            )
            r0 = float(rng.uniform(*self.cohort.r0_range))
            g_scale = float(rng.uniform(*self.cohort.growth_scale_range))
            patient_immune: Optional[ImmuneParams] = None
            if self.immune is not None:
                n_ant = float(rng.uniform(*self.cohort.neoantigen_range))
                k_imm = float(rng.uniform(*self.cohort.immune_kill_range))
                patient_immune = ImmuneParams(kill_max=k_imm, neoantigen_factor=n_ant)

            geom = make_gbm_phantom(size=size, seed=1000 + pid)
            clones = replace(self.clones, growth_S=self.clones.growth_S * g_scale)
            sim = TumorEvolutionSimulator(geom, clones=clones, drugs=self.drugs)
            traj = sim.run(
                policy,
                days=self.sim_days,
                initial_resistant_fraction=r0,
                seed=2000 + pid,
                immune=patient_immune,
            )
            outcomes.append(
                PatientOutcome(
                    patient_id=pid,
                    policy=policy_name,
                    pfs_days=traj.pfs_value(self.sim_days),
                    progressed=traj.progressed,
                    final_resistant_fraction=traj.final_resistant_fraction,
                    toxicity=traj.toxicity,
                    immune_used=patient_immune is not None,
                    tumor_size=size,
                    initial_resistant_fraction=r0,
                )
            )
        return outcomes


# ---------------------------------------------------------------------------
# Cohort statistics
# ---------------------------------------------------------------------------


def _summarize_arm(
    outcomes: List[PatientOutcome], sim_days: int
) -> Tuple[ArmSummary, Dict[str, object]]:
    pfs = np.asarray([o.pfs_days for o in outcomes], dtype=float)
    prog = np.asarray([o.progressed for o in outcomes], dtype=bool)
    tox = np.asarray([o.toxicity for o in outcomes], dtype=float)
    rfrac = np.asarray([o.final_resistant_fraction for o in outcomes], dtype=float)
    ltc = float(np.mean(pfs >= LONG_TERM_CONTROL_DAYS))

    km = cohort_km_curve(pfs, prog, horizon=float(sim_days))
    rel_res: Optional[float] = None
    rel_sens: Optional[float] = None
    median_R_controlled: Optional[float] = None
    controlled_mask = ~prog
    if controlled_mask.any():
        median_R_controlled = float(np.median(rfrac[controlled_mask]))
    # Failure-mode classification over progressed patients: what does
    # the relapsing tumour look like at horizon?
    relapsing = rfrac[prog] if prog.any() else np.asarray([])
    if relapsing.size:
        rel_res = float(np.mean(relapsing >= 0.5))
        rel_sens = float(np.mean(relapsing < 0.5))
    summary = ArmSummary(
        policy=outcomes[0].policy,
        n=len(outcomes),
        median_pfs_days=float(np.median(pfs)),
        pfs_iqr=(
            float(np.percentile(pfs, 25)),
            float(np.percentile(pfs, 75)),
        ),
        progressed_frac=float(np.mean(prog)),
        long_term_control_frac=ltc,
        median_toxicity=float(np.median(tox)),
        relapse_resistant_frac=rel_res,
        relapse_sensitive_frac=rel_sens,
        median_final_R_controlled=median_R_controlled,
    )
    return summary, km


def run_evotwin_cohort(
    arms: Optional[Sequence[BasePolicy]] = None,
    arm_names: Optional[Sequence[str]] = None,
    cohort: Optional[CohortParams] = None,
    *,
    sim_days: int = 400,
    immune: Optional[ImmuneParams] = ImmuneParams(),
    clones: CloneParams = DEFAULT_CLONES,
    drugs: DrugParams = DEFAULT_DRUGS,
) -> EvotwinReport:
    """Run the policy comparison over a simulated patient cohort.

    Default arms: MTD, adaptive dose modulation, collateral switch -- each
    WITH the immune layer when `immune` is provided (the cohort question is
    the combination, and the no-immune ablation lives in evosteer.run_trial).
    """
    cohort = cohort or CohortParams()
    if arms is None:
        arms = [MTDPolicy(), AdaptiveDosePolicy(), CollateralSwitchPolicy()]
    if arm_names is None:
        arm_names = [a.describe() for a in arms]
    if len(arms) != len(arm_names):
        raise ValueError("arms and arm_names must have the same length")

    outcomes_by_arm: Dict[str, List[PatientOutcome]] = {}
    summaries: Dict[str, ArmSummary] = {}
    kms: Dict[str, Dict[str, object]] = {}
    for pol, name in zip(arms, arm_names):
        sim = PatientCohortSimulator(
            cohort, clones=clones, drugs=drugs, sim_days=sim_days, immune=immune
        )
        outs = sim.run(pol, name)
        outcomes_by_arm[name] = outs
        summaries[name], kms[name] = _summarize_arm(outs, sim_days)

    # Log-rank between the first two arms (standard head-to-head). If the
    # combination therapy controls nearly everyone, events fall below the
    # estimator's minimum -- that is reported as 'not estimable', never
    # silently dropped.
    logrank: Optional[Dict[str, float]] = None
    logrank_note: Optional[str] = None
    if len(arm_names) >= 2:
        from danon.validation import logrank_two_group

        a, b = arm_names[0], arm_names[1]
        times = [o.pfs_days for o in outcomes_by_arm[a]] + [
            o.pfs_days for o in outcomes_by_arm[b]
        ]
        events = [int(o.progressed) for o in outcomes_by_arm[a]] + [
            int(o.progressed) for o in outcomes_by_arm[b]
        ]
        group_a = [True] * len(outcomes_by_arm[a]) + [False] * len(outcomes_by_arm[b])
        try:
            logrank = logrank_two_group(times, events, group_a)
            # arm_a / arm_b are positional: arm_names[0] vs arm_names[1].
        except ValueError as exc:
            logrank_note = (
                f"log-rank {a} vs {b} not estimable: {exc} -- near-total "
                "burden control leaves too few events to compare arms"
            )

    # Failure-mode table: among arms, fraction of patients whose disease at
    # horizon is (i) controlled, (ii) progressed with an S-dominant relapse
    # (re-treatable), (iii) progressed with an R-dominant relapse
    # (drug-resistant), (iv) eradicated (zero burden).
    failure: Dict[str, Dict[str, float]] = {}
    for name, outs in outcomes_by_arm.items():
        n = len(outs)
        failure[name] = {
            "controlled_frac": float(np.mean([not o.progressed for o in outs])),
            "relapse_sensitive_frac": float(
                np.mean([o.progressed and o.final_resistant_fraction < 0.5 for o in outs])
            ),
            "relapse_resistant_frac": float(
                np.mean([o.progressed and o.final_resistant_fraction >= 0.5 for o in outs])
            ),
        }

    medians = {n: summaries[n].median_pfs_days for n in arm_names}
    best = max(medians, key=medians.get)  # type: ignore[arg-type]
    verdict = (
        f"median PFS: " + ", ".join(f"{k}={v:.0f}d" for k, v in medians.items())
        + f"; best arm: {best}"
    )

    notes = [
        "all cohort statistics are DERIVED from the simulated cohort; "
        "parameter RANGES are ASSUMED population spreads, not measured "
        "distributions",
        f"immune layer is a hypothesis: checkpoint monotherapy FAILED in "
        f"recurrent GBM (CheckMate 143: {CHECKMATE143_MOS_MONTHS[0]} vs "
        f"{CHECKMATE143_MOS_MONTHS[1]} months, HR {CHECKMATE143_HR}) -- "
        "the combination modelled here has no clinical evidence",
        f"long-term control = PFS >= {LONG_TERM_CONTROL_DAYS}d in a "
        f"{sim_days}d simulation; the horizon bounds the claim",
        "steering anchor is the mCRPC adaptive-therapy trial (ratio "
        f"{ADAPTIVE_TTP_RATIO_ANCHOR:.2f}); this model class reproduces the "
        "direction, not the magnitude (see danon.evosteer CALIBRATION)",
    ]
    if logrank_note:
        notes.append(logrank_note)
    return EvotwinReport(
        arms=summaries,
        days=sim_days,
        cohort_size=cohort.size,
        immune_used=immune is not None,
        km_curves=kms,
        km_median_by_arm={n: km_median(kms[n]) for n in arm_names},
        logrank=logrank,
        failure_mode_table=failure,
        verdict=verdict,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Evolutionary-trap analysis: the two-jaw mechanism, isolated
# ---------------------------------------------------------------------------


@dataclass
class TrapReport:
    immune_off_mtd_relapse_resistant: bool
    immune_off_adaptive_relapse_resistant: bool
    immune_on_mtd_relapse_resistant: Optional[bool]
    immune_on_adaptive_relapse_resistant: Optional[bool]
    immune_on_adaptive_r_eliminated: bool
    trap_completed: bool
    interpretation: str
    notes: List[str] = field(default_factory=list)


def evolutionary_trap_analysis(
    geometry: TumorGeometry,
    *,
    days: int = 400,
    initial_resistant_fraction: float = 0.01,
    seed: int = 7,
    immune: Optional[ImmuneParams] = None,
    clones: CloneParams = DEFAULT_CLONES,
    drugs: DrugParams = DEFAULT_DRUGS,
) -> TrapReport:
    """Isolate the two jaws of the evolutionary trap on one tumour.

    Runs the 2x2 (policy x immune) and classifies each arm's failure mode
    by the final resistant fraction. The trap is 'completed' when adaptive
    dosing plus the immune layer ELIMINATES the resistant clone while the
    tumour remains under burden control -- i.e. R's fitness cost was
    converted into absolute decline by resistance-blind clearance.
    """
    immune = immune if immune is not None else ImmuneParams()
    sim = TumorEvolutionSimulator(geometry, clones=clones, drugs=drugs)

    def _end(pol: BasePolicy, imm: Optional[ImmuneParams]) -> Tuple[float, bool]:
        tr = sim.run(
            pol,
            days=days,
            initial_resistant_fraction=initial_resistant_fraction,
            seed=seed,
            immune=imm,
        )
        return tr.final_resistant_fraction, tr.progressed

    r_mtd, p_mtd = _end(MTDPolicy(), None)
    r_adp, p_adp = _end(AdaptiveDosePolicy(), None)
    r_mtd_i, p_mtd_i = _end(MTDPolicy(), immune)
    r_adp_i, p_adp_i = _end(AdaptiveDosePolicy(), immune)

    notes = [
        "single-tumour analysis (one seed); the cohort generalisation is "
        "run_evotwin_cohort",
        "immune layer is a hypothesis, not an evidence-backed GBM therapy "
        "(CheckMate 143 failure anchors the context)",
    ]
    interpretation = (
        "Without immune clearance both policies relapse with a fully "
        "drug-resistant tumour (R->1). With the hypothesised immune layer, "
        "MTD still selects a resistant (or eradicated) state while adaptive "
        "dosing can eliminate R entirely and leave a drug-sensitive, "
        "controlled tumour -- the evolutionary trap completing."
        if r_adp_i < 0.05 and not p_adp_i
        else "The hypothesised immune layer did not complete the trap at "
        "these parameters; the failure-mode asymmetry between policies "
        "still holds but R is suppressed, not eliminated."
    )
    return TrapReport(
        immune_off_mtd_relapse_resistant=r_mtd >= 0.5,
        immune_off_adaptive_relapse_resistant=r_adp >= 0.5,
        immune_on_mtd_relapse_resistant=(r_mtd_i >= 0.5) if p_mtd_i else None,
        immune_on_adaptive_relapse_resistant=(r_adp_i >= 0.5) if p_adp_i else None,
        immune_on_adaptive_r_eliminated=r_adp_i < 0.05,
        trap_completed=bool(r_adp_i < 0.05 and not p_adp_i),
        interpretation=interpretation,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Provenance and honest limits
# ---------------------------------------------------------------------------


def evotwin_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "adaptive_vs_standard_ttp_ratio_mcrpc": EvidenceStatus.ESTABLISHED,
        "checkpoint_monotherapy_gbm": EvidenceStatus.PROVEN_FAILED,   # CheckMate 143
        "adaptive_plus_immunotherapy_synergy": EvidenceStatus.UNKNOWN,  # hypothesis, no trial
        "rano_progression_threshold": EvidenceStatus.ESTABLISHED,
        "km_logrank_estimators": EvidenceStatus.ESTABLISHED,
        "cohort_parameter_ranges": EvidenceStatus.UNKNOWN,   # ASSUMED spreads
        "growth_kill_diffusion_constants": EvidenceStatus.UNKNOWN,  # tier-3 bench
        "immune_kill_and_antigenicity_constants": EvidenceStatus.UNKNOWN,  # co-culture
        "evolutionary_trap_two_jaw_mechanism": EvidenceStatus.UNKNOWN,  # in-silico result
    }


def what_this_model_cannot_know() -> List[str]:
    base = evosteer_limits()
    extra = [
        "immune escape: antigen-loss variants are not modelled -- real "
        "tumours can lose the neoantigen the clearance layer targets",
        "T-cell exhaustion and checkpoint pharmacodynamics: the immune "
        "layer is a constant kill term, not an adaptive immune system",
        "real cohort parameter distributions: ranges are ASSUMED population "
        "spreads, not fitted to patient data",
        "double-resistant clones under sequential therapy: two clones "
        "cannot represent a third phenotype emerging",
    ]
    return base + extra


def print_evotwin_disclaimer() -> None:
    print(
        "=" * 72 + "\n"
        "EVOTWIN -- simulation-only cohort evolutionary-trap engine\n"
        "Anchors: mCRPC adaptive-therapy trial (steering dynamics class);\n"
        "CheckMate 143 PROVEN_FAILED for checkpoint monotherapy in GBM --\n"
        "the immune layer modelled here is a HYPOTHESIS, not evidence-backed.\n"
        "No patient data, no clinical claim, no real TCGA/TCIA intake.\n"
        + "=" * 72
    )
