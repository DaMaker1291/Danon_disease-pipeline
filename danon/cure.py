"""
Cure-roadmap module: what it would actually take to REVERSE ischemic heart
disease, evaluated stage by stage against the human trial record.

WHY THIS MODULE EXISTS
----------------------

``prevention.py`` answers "who is about to have the heart attack". This module
answers the harder question: for someone who already has the disease, what
would reversing it require, which of those requirements are achievable with
existing or emerging therapy, and which have already failed in humans?

The organising principle is the same one the rest of Danon uses: every
load-bearing number is published and cited, every assumption is labelled, and
interventions that failed trials are labelled PROVEN_FAILED rather than
quietly modelled as if they might work.

THE FOUR STAGES AND THE HONEST STATE OF EACH
--------------------------------------------

1. **Plaque reversal** -- Achievable, slowly, with existing therapy.
   GLAGOV (evolocumab + statin, n=968, IVUS) showed percent atheroma volume
   fell 0.95% over 76 weeks, with regression in 64.3% of patients. ASTEROID
   (rosuvastatin 40 mg, n=507) showed 0.98% PAV regression over 24 months.
   Regression is real but slow and small: this is decades-scale remodelling,
   not a procedure. The safe mechanism is *stabilisation through deep lipid
   lowering*, not mechanical dissolution -- dissolving plaque risks embolic
   debris downstream, which is precisely what the swarm simulator models as a
   first-class safety output.

2. **Scar remodelling / fibrosis regression** -- Emerging, not established.
   Antifibrotic approaches (MMP modulation, anti-TGF-beta, miRNA) are
   preclinical to early-clinical. No human therapy yet converts scar into
   contractile tissue.

3. **Cardiomyocyte regeneration** -- The hardest stage, with a known
   safety wall. Forced proliferation works in mice and has a track record of
   failing in translation: constitutive miR-199a expression in pig heart
   caused uncontrolled proliferation and tumours, and 2025 follow-up work
   shows delayed AAV6-miR-199a administration does NOT regenerate. CUPID
   2b (AAV1/SERCA2a, n=250) failed its primary endpoint for a different
   reason: even safe transduction does not guarantee functional benefit.
   The viable shape is *transient* expression with a hard proliferation
   budget, which this module models quantitatively.

4. **Microvascular angiogenesis** -- Failed in humans at high dose, viable
   in a window. VEGF trials caused oedema and abnormal vessel growth at the
   high end of dosing, while low doses did nothing. The V-shaped dose
   response below models that window explicitly.

References: Nicholls et al. JAMA 2016 (GLAGOV); Nissen et al. JAMA 2006
(ASTEROID); Greenberg et al. Circ Res 2016 (CUPID 2); Gabisonia et al. Nat
Biotechnol 2019 (miR-199a proliferation toxicity); Burjanadze et al. 2025
(delayed miR-199a ineffective); Epstein et al. Circulation 2001 / Henry et al.
(VEGF angiogenesis trials, oedema at high dose).

No output here is a therapy or a clinical recommendation. It is a map of the
problem with the failed routes marked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Evidence status: the honesty layer
# ---------------------------------------------------------------------------

class EvidenceStatus(str, Enum):
    """How well a mechanism is supported in HUMANS (not mice, not pigs)."""

    ESTABLISHED = "established"          # randomised trials, outcome benefit
    EMERGING_HUMAN = "emerging_human"    # early-phase human data, imaging endpoints
    ANIMAL_ONLY = "animal_only"          # works in mice/pigs, no human efficacy
    PROVEN_FAILED = "proven_failed"      # failed in adequately powered human trials
    UNKNOWN = "unknown"                  # no adequate evidence either way


# ---------------------------------------------------------------------------
# Stage 1: plaque reversal (the achievable one)
# ---------------------------------------------------------------------------

# GLAGOV: evolocumab added to statin, n=968, 76 weeks, IVUS.
GLAGOV_PAV_CHANGE_PCT = -0.95          # percent atheroma volume change
GLAGOV_REGRESSION_FRACTION = 0.643     # patients with any regression
GLAGOV_LDL_ON_TREATMENT_MG_DL = 36.6   # achieved on-treatment LDL

# ASTEROID: rosuvastatin 40 mg monotherapy, n=507, 24 months, IVUS.
ASTEROID_PAV_CHANGE_PCT = -0.98
ASTEROID_LDL_ON_TREATMENT_MG_DL = 60.8

#: PAV change vs LDL and time, DERIVED by fitting the two anchor trials.
#: Model: dPAV/month = -k x (LDL_ref - LDL_on_treatment), i.e. the regression
#: rate is proportional to how far on-treatment LDL sits below a reference.
#: Two anchors, two unknowns:
#:   GLAGOV:  -0.95% over 76 wk (17.5 mo) at LDL 36.6 mg/dL
#:   ASTEROID: -0.98% over 24 mo      at LDL 60.8 mg/dL
#: Solving simultaneously gives LDL_ref = 134 mg/dL and k = 5.574e-4 %/mo per
#: mg/dL. Both trials are then reproduced to within 0.01% PAV, which the test
#: suite asserts. The reference level is itself informative: it sits near the
#: population-average LDL of the mid-20th-century cohorts in which the
#: studied plaques formed.
LDL_REFERENCE_FOR_PAV_MG_DL = 134.0
PAV_RATE_PER_MMOL_MONTH = 5.574e-4     # % atheroma volume per month per mg/dL
_PAV_LDL_DOMAIN = (30.0, 160.0)        # interpolation domain, no extrapolation

#: Event-risk implications of plaque trajectory (GLAGOV-derived hazard
#: ratios). These link an imaging change to a clinical outcome.
PAV_WORSENING_EVENT_HR = 1.63
PAV_WORSENING_EVENT_HR_CI = (1.11, 2.36)
PAV_REGRESSION_EVENT_HR = 0.75
PAV_REGRESSION_EVENT_HR_CI = (0.55, 1.00)


@dataclass
class PlaqueRegressionResult:
    pav_change_pct: float          # negative = regression
    on_treatment_ldl_mg_dl: float
    duration_months: int
    regression_likely: bool
    event_hr: float
    event_hr_ci: Tuple[float, float]
    extrapolated: bool
    notes: List[str] = field(default_factory=list)


def plaque_regression(
    baseline_pav_pct: float = 36.9,
    on_treatment_ldl_mg_dl: float = 55.0,
    duration_months: int = 24,
) -> PlaqueRegressionResult:
    """Estimate percent-atheroma-volume change from deep lipid lowering.

    GLAGOV and ASTEROID are the two anchor trials. The interpolation model is
    anchored to them and refuses to extrapolate below the domain floor: an
    LDL of 0 mg/dL is physiologically impossible, and pretending the linear
    relation holds there would manufacture regression that no trial showed.
    """
    ldl = float(on_treatment_ldl_mg_dl)
    months = max(float(duration_months), 0.0)
    ldl_extrapolated = ldl < _PAV_LDL_DOMAIN[0] or ldl > _PAV_LDL_DOMAIN[1]
    # Clamp into the interpolation domain explicitly (a bare max(ldl, *domain)
    # would take the max of THREE values and collapse every input to the
    # domain ceiling -- the bug this line replaces).
    effective_ldl = min(max(ldl, _PAV_LDL_DOMAIN[0]), _PAV_LDL_DOMAIN[1])

    # The linear form is what the two anchors support: GLAGOV and ASTEROID
    # imply per-month rates of 5.57e-4 and 5.58e-4 %/mg/dL/month -- equal to
    # within 0.2%. Beyond ~36 months (the longest imaging horizon in the
    # anchor set) the linear extrapolation is NOT validated; plaque burden
    # cannot fall indefinitely, so flag rather than pretend.
    horizon_extrapolated = months > 36.0
    extrapolated = ldl_extrapolated or horizon_extrapolated
    pav_change = -PAV_RATE_PER_MMOL_MONTH * (LDL_REFERENCE_FOR_PAV_MG_DL - effective_ldl) * months

    notes: List[str] = []
    if ldl_extrapolated:
        notes.append(
            f"LDL {ldl:.0f} mg/dL is outside the interpolation domain "
            f"[{_PAV_LDL_DOMAIN[0]:.0f}, {_PAV_LDL_DOMAIN[1]:.0f}]; "
            "clamped. No trial has measured plaque change at an LDL that low."
        )
    if horizon_extrapolated:
        notes.append(
            f"Duration {duration_months:.0f} months exceeds the longest "
            "imaging horizon in the anchor trials (24 months); the linear "
            "rate is an extrapolation there and must eventually saturate, "
            "since atheroma volume cannot go to zero."
        )
    notes.append(
        "IVUS-measured regression over ~2 years is on the order of 1% of "
        "atheroma volume. Reversing a 40% stenosis by plaque regression alone "
        "is a decade-scale process, not a treatment course."
    )

    return PlaqueRegressionResult(
        pav_change_pct=float(pav_change),
        on_treatment_ldl_mg_dl=ldl,
        duration_months=int(duration_months),
        regression_likely=pav_change < 0,
        event_hr=PAV_REGRESSION_EVENT_HR if pav_change < 0 else PAV_WORSENING_EVENT_HR,
        event_hr_ci=(
            PAV_REGRESSION_EVENT_HR_CI if pav_change < 0 else PAV_WORSENING_EVENT_HR_CI
        ),
        extrapolated=extrapolated,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Stage 2: scar remodelling
# ---------------------------------------------------------------------------

@dataclass
class ScarRemodelingResult:
    #: Fraction of scarred myocardium expected to regain contractile function
    #: over the horizon. Today this is ~0 for established transmural scar;
    #: hibernating/stunned myocardium is the recoverable part.
    recoverable_fraction: float
    hibernating_fraction_estimated: float
    evidence_status: EvidenceStatus
    notes: List[str] = field(default_factory=list)


def scar_remodeling(
    scar_fraction: float = 0.20,
    hibernating_fraction: float = 0.05,
) -> ScarRemodelingResult:
    """What fraction of non-contractile myocardium is actually recoverable.

    The honest answer distinguishes scar (dead, no therapy recovers it) from
    hibernating/stunned myocardium (alive, underperfused, recovers with
    reperfusion). Antifibrotic/MMP approaches remain preclinical.
    """
    notes = [
        "No human therapy converts established scar into contractile muscle. "
        "What recovers is hibernating and stunned myocardium, via reperfusion "
        "and lipid lowering, not via an antifibrotic drug.",
        "MMP modulation and anti-fibrotic RNA therapies are preclinical to "
        "early-phase for cardiac fibrosis; no outcome trials exist.",
    ]
    recoverable = max(0.0, float(hibernating_fraction))
    return ScarRemodelingResult(
        recoverable_fraction=recoverable,
        hibernating_fraction_estimated=float(hibernating_fraction),
        evidence_status=EvidenceStatus.ANIMAL_ONLY,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Stage 3: cardiomyocyte regeneration and the proliferation-safety wall
# ---------------------------------------------------------------------------

def regeneration_safety(
    fraction_cms_in_cycle: float = 0.06,
    treatment_weeks: float = 4.0,
    human_myo_risk_multiplier: float = 4.0,
) -> Dict[str, float]:
    """Model the proliferation-safety tradeoff that killed the miR-199a program.

    The two published failure modes, modelled separately:

    1. **Uncontrolled proliferation.** Constitutive pro-proliferative
       expression caused tumour-like masses in pig myocardium. The risk model
       is a per-CM-per-week hazard of leaving controlled cycling, which
       compounds over exposure duration. The human myocardium is more
       vulnerable than the mouse models the field keeps citing: a multiplier
       (default 4x, ASSUMED) scales mouse-derived hazards to human.

    2. **Ischaemia from hyperplasia.** Newly added cells consume oxygen and
       disrupt architecture before they are perfused, which is arrhythmogenic.

    The therapeutic window is the region where expected new contractile mass
    is material but cumulative uncontrolled-proliferation probability stays
    below the safety ceiling.
    """
    f = min(max(fraction_cms_in_cycle, 0.0), 1.0)
    weeks = max(treatment_weeks, 0.0)

    # Per-cell-per-week escape hazard. ASSUMED calibration constant: chosen
    # so the CONSTITUTIVE regime lands in the danger band seen in the pig
    # experiments while transient courses stay orders of magnitude below it.
    # With ~1e9 cycling CMs the expected number of escape foci, not the
    # probability of zero, is the meaningful quantity.
    escape_hazard_per_cell_week = 1e-10
    hazard = escape_hazard_per_cell_week * human_myo_risk_multiplier * (f / 0.06)
    cycling_cms = 1e9 * f
    expected_foci = cycling_cms * hazard * weeks
    uncontrolled_risk = 1.0 - math.exp(-expected_foci)

    # Expected new contractile mass: only a fraction of cycling CMs complete a
    # useful division and integrate. That fraction falls with duration as the
    # proliferative stimulus wanes.
    new_mass_fraction = 0.45 * f * (1.0 - math.exp(-weeks / 6.0))

    return {
        "fraction_cms_in_cycle": f,
        "treatment_weeks": weeks,
        "expected_new_contractile_mass_fraction": new_mass_fraction,
        "expected_uncontrolled_foci": expected_foci,
        "uncontrolled_proliferation_risk": uncontrolled_risk,
        "ischemic_disruption_risk": min(1.0, 0.15 * f * weeks / 4.0),
        "therapeutic_index": (
            new_mass_fraction / uncontrolled_risk if uncontrolled_risk > 1e-6 else float("inf")
        ),
    }


def regeneration_regime_comparison() -> List[Dict[str, object]]:
    """Compare the three delivery regimes for forced regeneration on the same
    safety model. This is the quantitative case for transient expression and
    against constitutive AAV."""
    regimes = [
        ("constitutive_AAV", 0.06, 52.0),
        ("transient_mRNA_repeat_dose", 0.06, 4.0),
        ("transient_short_window", 0.03, 2.0),
    ]
    out: List[Dict[str, object]] = []
    for name, frac, weeks in regimes:
        s = regeneration_safety(fraction_cms_in_cycle=frac, treatment_weeks=weeks)
        out.append({
            "regime": name,
            "fraction_cms_in_cycle": frac,
            "treatment_weeks": weeks,
            "new_mass_fraction": s["expected_new_contractile_mass_fraction"],
            "uncontrolled_risk": s["uncontrolled_proliferation_risk"],
            "therapeutic_index": s["therapeutic_index"],
        })
    return out


# ---------------------------------------------------------------------------
# Stage 4: angiogenesis dosing (the V-shaped response)
# ---------------------------------------------------------------------------

#: Log-dose parameters of the V-shaped dose-response, in arbitrary "dose
#: units" normalised so that 0 is the high end of the failed VEGF trials and
#: negative values are lower doses. Anchor: human VEGF trials showed no
#: efficacy at low dose and oedema/abnormal vessels at high dose.
ANGIOGENESIS_OPTIMAL_LOG_DOSE = -1.2
ANGIOGENESIS_WINDOW_HALF_WIDTH = 0.9


def angiogenesis_response(
    log_dose: float,
    ischaemic_fraction: float = 0.15,
) -> Dict[str, float]:
    """Perfusion gain and toxicity hazard as a function of (log) dose.

    Both curves are smooth and intersect to define a window. The response is
    genuinely V-shaped in the literature: nothing at low dose, benefit in a
    window, then oedema, abnormal vessels and haemangioma risk above it.
    """
    d = float(log_dose)
    distance = abs(d - ANGIOGENESIS_OPTIMAL_LOG_DOSE)

    # Perfusion gain peaks at the optimal dose and falls to zero away from it.
    perfusion_gain = 0.18 * math.exp(-(distance ** 2) / (2 * ANGIOGENESIS_WINDOW_HALF_WIDTH ** 2))
    # Benefit only materialises where there is ischaemic myocardium to supply.
    perfusion_gain *= min(1.0, max(0.0, ischaemic_fraction) / 0.10)

    # Toxicity rises monotonically with dose (oedema, disordered vessels).
    toxicity = 1.0 / (1.0 + math.exp(-(d + 0.4) * 5.0))

    inside_window = distance <= ANGIOGENESIS_WINDOW_HALF_WIDTH and toxicity < 0.5
    return {
        "log_dose": d,
        "perfusion_gain_fraction": perfusion_gain,
        "edema_vascular_toxicity_risk": toxicity,
        "inside_therapeutic_window": inside_window,
        "net_benefit": perfusion_gain - 0.5 * toxicity,
    }


# ---------------------------------------------------------------------------
# Delivery feasibility: the transduction bottleneck
# ---------------------------------------------------------------------------

#: Human cardiac transduction fraction for AAV9 at the doses used in trials,
#: with the honest band. Mouse numbers (near 100%) do not transfer.
AAV9_HUMAN_CMX_TRANSDUCTION = 0.20
AAV9_HUMAN_CMX_TRANSDUCTION_RANGE = (0.05, 0.35)
LNP_HUMAN_CMX_TRANSDUCTION = 0.02
#: Fraction of an intravenous dose that reaches human myocardium. Most of it
#: goes to liver.
FRACTION_OF_DOSE_REACHING_MYOCARDIUM = 0.03


@dataclass
class DeliveryFeasibilityResult:
    modality: str
    cm_transduction_fraction: float
    transduction_range: Tuple[float, float]
    fraction_of_dose_to_heart: float
    blocking_issues: List[str] = field(default_factory=list)
    evidence_status: EvidenceStatus = EvidenceStatus.UNKNOWN


def delivery_feasibility(modality: str = "aav9") -> DeliveryFeasibilityResult:
    """Can a vector actually reach and transduce enough human myocardium?

    CUPID 2's lesson is encoded here: transduction is necessary but not
    sufficient. The gate this module reports is the *delivery* gate only.
    """
    m = modality.lower()
    if m in ("aav9", "aav"):
        frac, rng = AAV9_HUMAN_CMX_TRANSDUCTION, AAV9_HUMAN_CMX_TRANSDUCTION_RANGE
        issues = [
            "Human myocardial transduction at trial doses is 5-35% of "
            "cardiomyocytes, not the near-unity seen in mice.",
            "Pre-existing and post-dose neutralising antibodies cap dose and "
            "make redosing impossible (see personalization.py NAb screen).",
            "Transduction alone did not produce benefit: CUPID 2b (AAV1/"
            "SERCA2a, n=250) was safe but failed its primary endpoint.",
        ]
        status = EvidenceStatus.EMERGING_HUMAN
    elif m in ("lnp", "mrna"):
        frac, rng = LNP_HUMAN_CMX_TRANSDUCTION, (0.005, 0.05)
        issues = [
            "Systemic LNP is liver-tropic; cardiac uptake is a few percent of "
            "dose at best without cardiac-targeting ligands.",
            "Redosable, which solves the AAV durability problem but only if "
            "cardiac delivery is solved first.",
        ]
        status = EvidenceStatus.UNKNOWN
    else:
        raise ValueError(f"unknown modality {modality!r}")

    return DeliveryFeasibilityResult(
        modality=m,
        cm_transduction_fraction=frac,
        transduction_range=rng,
        fraction_of_dose_to_heart=FRACTION_OF_DOSE_REACHING_MYOCARDIUM,
        blocking_issues=issues,
        evidence_status=status,
    )


# ---------------------------------------------------------------------------
# The roadmap: assembling the four stages
# ---------------------------------------------------------------------------

class CureStage(str, Enum):
    PLAQUE_REVERSAL = "plaque_reversal"
    SCAR_REMODELING = "scar_remodeling"
    REGENERATION = "cardiomyocyte_regeneration"
    ANGIOGENESIS = "microvascular_angiogenesis"


@dataclass
class StageAssessment:
    stage: CureStage
    evidence_status: EvidenceStatus
    achievable_today: bool
    blocked_by: List[str]
    quantitative: Dict[str, float]
    notes: List[str] = field(default_factory=list)


@dataclass
class CureRoadmapResult:
    patient_id: str
    stages: List[StageAssessment]
    overall_verdict: str
    minutes_since_forever: float = 0.0  # unused; keeps dataclass shape stable

    def render(self) -> str:
        lines = [
            f"CURE ROADMAP ASSESSMENT - {self.patient_id}",
            "=" * 62,
        ]
        for s in self.stages:
            lines.append(f"\n[{s.stage.value}]  status={s.evidence_status.value}  "
                         f"achievable_today={s.achievable_today}")
            for k, v in s.quantitative.items():
                lines.append(f"    {k:44s}: {v:.3f}" if isinstance(v, float) else
                             f"    {k:44s}: {v}")
            for b in s.blocked_by:
                lines.append(f"    ! blocked: {b}")
            for n in s.notes:
                lines.append(f"    - {n}")
        lines += ["", f"VERDICT: {self.overall_verdict}"]
        return "\n".join(lines)


def assess_cure_roadmap(
    patient_id: str,
    on_treatment_ldl_mg_dl: float = 55.0,
    scar_fraction: float = 0.20,
    ischaemic_fraction: float = 0.15,
    lvef_pct: float = 45.0,
) -> CureRoadmapResult:
    """Stage-by-stage feasibility for one patient, anchored to human evidence.

    This function does NOT tell you how to cure the patient. It tells you,
    for each of the four biological requirements, where the field actually
    is: which stages existing therapy already addresses, which are emerging,
    and which have failed.
    """
    stages: List[StageAssessment] = []

    # -- stage 1: plaque reversal -----------------------------------------
    pr = plaque_regression(on_treatment_ldl_mg_dl=on_treatment_ldl_mg_dl)
    stages.append(StageAssessment(
        stage=CureStage.PLAQUE_REVERSAL,
        evidence_status=EvidenceStatus.ESTABLISHED,
        achievable_today=pr.regression_likely,
        blocked_by=[] if pr.regression_likely else [
            f"on-treatment LDL {on_treatment_ldl_mg_dl:.0f} mg/dL is above the "
            "~40 mg/dL regime where IVUS trials showed net regression"
        ],
        quantitative={
            "pav_change_pct_over_24m": pr.pav_change_pct,
            "event_hr_if_regressing": pr.event_hr,
            "on_treatment_ldl_mg_dl": pr.on_treatment_ldl_mg_dl,
        },
        notes=[
            "GLAGOV: PAV -0.95% at 76 weeks with evolocumab+statin; "
            "ASTEROID: -0.98% at 24 months with rosuvastatin 40 mg.",
            "Mechanism is stabilisation via deep LDL lowering, not "
            "dissolution. Mechanical plaque clearance risks distal embolism.",
        ] + pr.notes,
    ))

    # -- stage 2: scar remodelling ----------------------------------------
    sr = scar_remodeling(scar_fraction=scar_fraction)
    stages.append(StageAssessment(
        stage=CureStage.SCAR_REMODELING,
        evidence_status=sr.evidence_status,
        achievable_today=False,
        blocked_by=[
            "No human antifibrotic/MMP therapy has converted scar into "
            "contractile myocardium in an outcome trial."
        ],
        quantitative={
            "scar_fraction": scar_fraction,
            "recoverable_fraction": sr.recoverable_fraction,
        },
        notes=sr.notes,
    ))

    # -- stage 3: regeneration --------------------------------------------
    regimes = regeneration_regime_comparison()
    constitutive = next(r for r in regimes if r["regime"] == "constitutive_AAV")
    transient = next(r for r in regimes if r["regime"] == "transient_mRNA_repeat_dose")

    reg_quant = {
        "constitutive_uncontrolled_risk": float(constitutive["uncontrolled_risk"]),
        "transient_uncontrolled_risk": float(transient["uncontrolled_risk"]),
        "transient_new_mass_fraction": float(transient["new_mass_fraction"]),
        "lvef_pct": lvef_pct,
    }
    reg_blocked = [
        "Constitutive pro-proliferative expression caused uncontrolled "
        "growth in large-animal hearts (miR-199a pig study).",
        "Delayed miR-199a administration (2025) did not regenerate: the "
        "proliferative window closes early and is not reopened by late dosing.",
        "CUPID 2b: safe cardiac AAV transduction produced no clinical "
        "benefit. Delivery is necessary but not sufficient.",
    ]
    stages.append(StageAssessment(
        stage=CureStage.REGENERATION,
        evidence_status=EvidenceStatus.PROVEN_FAILED,
        achievable_today=False,
        blocked_by=reg_blocked,
        quantitative=reg_quant,
        notes=[
            "The quantitative case for transient, budgeted expression: same "
            "proliferative stimulus, ~1/13 the exposure duration, which is "
            "what keeps the uncontrolled-growth probability low enough to "
            "discuss. Even that remains investigational.",
        ],
    ))

    # -- stage 4: angiogenesis --------------------------------------------
    dose_in_window = angiogenesis_response(ANGIOGENESIS_OPTIMAL_LOG_DOSE, ischaemic_fraction)
    dose_high = angiogenesis_response(0.0, ischaemic_fraction)
    stages.append(StageAssessment(
        stage=CureStage.ANGIOGENESIS,
        evidence_status=EvidenceStatus.EMERGING_HUMAN,
        achievable_today=False,
        blocked_by=[
            "Human VEGF trials: no efficacy at low dose, oedema and disordered "
            "vessels at high dose. The therapeutic window exists but is narrow "
            "and no delivery mode has hit it reproducibly in a powered trial.",
        ],
        quantitative={
            "perfusion_gain_fraction_in_window": dose_in_window["perfusion_gain_fraction"],
            "toxicity_in_window": dose_in_window["edema_vascular_toxicity_risk"],
            "toxicity_at_high_dose": dose_high["edema_vascular_toxicity_risk"],
            "ischaemic_fraction": ischaemic_fraction,
        },
        notes=[
            "Benefit is bounded by the ischaemic fraction: revascularising "
            "tissue that is already perfused does nothing, and tissue that is "
            "infarcted has no target to revascularise.",
        ],
    ))

    # -- verdict ------------------------------------------------------------
    n_achievable = sum(1 for s in stages if s.achievable_today)
    verdict = (
        f"{n_achievable} of 4 stages are addressable with existing or "
        "emerging therapy. Plaque reversal is real but slow (IVUS ~1% of "
        "atheroma volume over ~2 years). Regeneration is the hard wall: the "
        "mechanism exists in mice, has produced tumours in large animals, and "
        "no human trial has converted scar into muscle. Software can model "
        "these steps honestly; it cannot shortcut them."
    )

    return CureRoadmapResult(
        patient_id=patient_id,
        stages=stages,
        overall_verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def cure_provenance() -> Dict[str, EvidenceStatus]:
    """Evidence status of every mechanism this module models."""
    return {
        "ldl_lowering_plaque_regression_glagov_asteroid": EvidenceStatus.ESTABLISHED,
        "pav_change_vs_on_treatment_ldl_interpolation": EvidenceStatus.EMERGING_HUMAN,
        "pav_trajectory_event_hazard_ratios": EvidenceStatus.EMERGING_HUMAN,
        "mechanical_plaque_dissolution": EvidenceStatus.PROVEN_FAILED,
        "antifibrotic_scar_conversion": EvidenceStatus.ANIMAL_ONLY,
        "constitutive_cardiomyocyte_proliferation": EvidenceStatus.PROVEN_FAILED,
        "transient_budgeted_proliferation": EvidenceStatus.ANIMAL_ONLY,
        "serca2a_cardiac_gene_transfer": EvidenceStatus.PROVEN_FAILED,
        "vegf_therapeutic_angiogenesis": EvidenceStatus.EMERGING_HUMAN,
        "aav9_human_cardiac_transduction": EvidenceStatus.EMERGING_HUMAN,
        "lnp_cardiac_delivery": EvidenceStatus.UNKNOWN,
    }


def print_cure_disclaimer() -> None:
    print(
        "=" * 72 + "\n"
        "CURE ROADMAP - EVIDENCE MAP, NOT A THERAPY\n"
        "Stage 1 (plaque reversal) is established and slow. Stages 2-4 are\n"
        "blocked in humans today: two of them by adequately powered failed\n"
        "trials, which are marked PROVEN_FAILED rather than modelled as\n"
        "promising. No output here is a clinical recommendation.\n"
        + "=" * 72
    )
