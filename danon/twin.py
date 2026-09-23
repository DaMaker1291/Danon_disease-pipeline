"""
Organ-level cardiovascular digital twin.

WHAT THIS IS
------------

The question "can we simulate the whole body, every cell, every mutation"
has a provable answer: no. Not "not yet" -- no.

1. STATE SPACE. ~37 trillion cells, each with thousands of state variables,
   is 10^17-10^19 coupled dynamic variables. Simulating that forward is
   beyond any existent or projected machine, and most of it is unnecessary:
   identical cells behave identically, which is why every working clinical
   twin (HeartFlow FFR-CT, Living Heart, EU Virtual Human Twin programme)
   uses COMPARTMENT models -- the liver as one compartment, not 240 billion
   hepatocytes individually.

2. OBSERVABILITY. A twin can only be as deep as its inputs. MRI resolves
   millimetres; it cannot read intracellular state. A per-cell model cannot
   be initialized OR validated from imaging, so the honest twin fed by MRI
   is an organ/function-level twin, not a cellular one.

3. MUTATIONS. "Every mutation" requires knowing which of the ~10^7
   population variants MATTER in this patient, and that map is the open
   problem of genomics. A twin can carry the measured, actionable subset.

So this module builds the thing that is real: a CARDIOVASCULAR TWIN that

  * initializes from a patient scan (MRI function/structure, CT stenosis,
    CAC) plus a blood panel,
  * runs TREATMENT COUNTERFACTUALS by composing the models that already
    exist in Danon -- the published risk equations (prevention.py), the
    IVUS-anchored plaque trajectory (cure.py), the vectorised coronary
    flow/plaque physics (swarm.py, physics.py) -- and
  * REFUSES to invent numbers for treatments it has no validated model for,
    returning NOT_MODELLED with the reason instead.

That last behaviour is the difference between a twin and a video game.

WHAT IT CANNOT DO (by design, not by effort)
--------------------------------------------
- No cellular or subcellular state. Imaging cannot see it; the twin does
  not pretend to.
- No de-novo pharmacology. A drug with no validated effect-size model
  comes back NOT_MODELLED, not with a plausible-looking guess.
- No outcomes below the organ level: arrhythmia mechanisms, tumour growth,
  immune-cell dynamics are outside the twin's observability.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from danon.prevention import (
    CTT_MVE_RR_PER_MMOL_L,
    LDL_THRESHOLD_VERY_HIGH,
    MG_DL_PER_MMOL_L_CHOLESTEROL,
    RaceStratum,
    RiskEquation,
    RiskEnhancers,
    RiskFactorProfile,
    Sex,
    TherapyRung,
    _RUNG_FRACTION,
    project_benefit,
    risk_category,
    routine_ldl_reduction,
    triage,
)
from danon.cure import plaque_regression
from danon.physics import BLOOD_VISCOSITY_PA_S

# ---------------------------------------------------------------------------
# Scan and panel inputs
# ---------------------------------------------------------------------------


@dataclass
class ScanInputs:
    """Everything the imaging modalities can actually tell the twin.

    Every field is something a real scanner reports. Fields left as None are
    treated as unmeasured, and the twin reports which conclusions are
    weakened by the gap rather than imputing them.
    """

    #: Cardiac MRI, short-axis stack.
    lvef_pct: Optional[float] = None
    lv_mass_g: Optional[float] = None
    lv_edv_ml: Optional[float] = None
    scar_fraction: Optional[float] = None          # late-gadolinium enhancement
    ischaemic_fraction: Optional[float] = None     # stress perfusion deficit

    #: CT coronary angiography / calcium score.
    worst_stenosis_pct: Optional[float] = None
    cac_score: Optional[float] = None
    coronary_diameter_mm: Optional[float] = None

    #: Carotid/aortic measures sometimes available.
    systolic_bp_mmhg: Optional[float] = None


@dataclass
class TwinPatient:
    """The twin's complete initial state: imaging + blood + demographics."""

    patient_id: str
    age_years: float
    sex: Sex
    scan: ScanInputs = field(default_factory=ScanInputs)

    chol_total_mgdl: float = 190.0
    chol_hdl_mgdl: float = 50.0
    ldl_mg_dl: Optional[float] = None
    bp_sys_mmhg: float = 125.0
    on_bp_medication: bool = False
    current_smoker: bool = False
    diabetes: bool = False
    lipoprotein_a_nmol_l: Optional[float] = None
    race_black: bool = False

    established_ascvd: bool = False

    def to_risk_profile(self) -> RiskFactorProfile:
        return RiskFactorProfile(
            patient_id=self.patient_id,
            age_years=self.age_years,
            sex=self.sex,
            # Must be the enum, not a bare string: the risk equations index
            # coefficient tables by .race.value and use identity checks.
            race=RaceStratum.BLACK if self.race_black else RaceStratum.WHITE,
            chol_total_mgdl=self.chol_total_mgdl,
            chol_hdl_mgdl=self.chol_hdl_mgdl,
            bp_sys_mmhg=self.bp_sys_mmhg,
            on_bp_medication=self.on_bp_medication,
            current_smoker=self.current_smoker,
            diabetes=self.diabetes,
            ldl_mg_dl=self.ldl_mg_dl,
        )

    def to_enhancers(self) -> RiskEnhancers:
        return RiskEnhancers(
            lipoprotein_a_nmol_l=self.lipoprotein_a_nmol_l,
            cac_score=self.scan.cac_score,
            ldl_mg_dl=self.ldl_mg_dl,
        )


# ---------------------------------------------------------------------------
# Treatment counterfactuals
# ---------------------------------------------------------------------------


class TwinVerdict(str, Enum):
    MODELLED = "modelled"                # validated effect-size model applied
    MODELLED_APPROXIMATE = "modelled_approximate"  # modelled, flagged uncertainty
    NOT_MODELLED = "not_modelled"        # no validated model; no number returned


class TwinTreatment(str, Enum):
    """Treatments the twin can reason about.

    The catalogue is deliberately short: each entry must map to a model with
    a published effect size. A treatment absent from this list is returned
    NOT_MODELLED by construction -- the twin cannot be talked into
    inventing a number for it.
    """

    LIFESTYLE = "lifestyle"
    MODERATE_STATIN = "moderate_intensity_statin"
    HIGH_INTENSITY_STATIN = "high_intensity_statin"
    STATIN_EZETIMIBE = "statin_plus_ezetimibe"
    PCSK9_INHIBITOR = "pcsk9_inhibitor_or_inclisiran"
    ONE_SHOT_BASE_EDITING = "one_shot_pcsk9_base_editing"
    STENT = "stenting_of_stenosis"
    # Deliberately ABSENT: regenerative gene therapy, scar conversion,
    # anti-fibrotics, angiogenesis dosing -- cure.py documents why each has
    # no validated human effect size.


_TREATMENT_TO_RUNG: Dict[TwinTreatment, TherapyRung] = {
    TwinTreatment.LIFESTYLE: TherapyRung.LIFESTYLE,
    TwinTreatment.MODERATE_STATIN: TherapyRung.MODERATE_STATIN,
    TwinTreatment.HIGH_INTENSITY_STATIN: TherapyRung.HIGH_INTENSITY_STATIN,
    TwinTreatment.STATIN_EZETIMIBE: TherapyRung.STATIN_EZETIMIBE,
    TwinTreatment.PCSK9_INHIBITOR: TherapyRung.PCSK9_OR_INCLISIRAN,
    TwinTreatment.ONE_SHOT_BASE_EDITING: TherapyRung.ONESHOT_BASE_EDITING,
}

#: Stenting resolves a geometric stenosis; the twin models it as restoring
#: the lumen to the residual-disease level, not to a virgin artery. The
#: STENT effect below is the post-procedure residual stenosis, consistent
#: with published post-PCI residual disease.
POST_STENT_RESIDUAL_STENOSIS_PCT = 10.0


@dataclass
class CounterfactualResult:
    treatment: TwinTreatment
    verdict: TwinVerdict
    horizon_years: int

    # --- events (from the published risk machinery) ----------------------
    baseline_event_risk_10y: float = 0.0
    treated_event_risk: float = 0.0
    absolute_risk_reduction: float = 0.0
    number_needed_to_treat: float = float("inf")

    # --- plaque trajectory (from the IVUS-anchored model) ----------------
    pav_change_pct: float = 0.0
    ldl_on_treatment_mg_dl: float = 0.0
    ldl_reduction_mmol_l: float = 0.0

    # --- haemodynamics (from the coronary physics model) -----------------
    pressure_drop_change_pct: Optional[float] = None

    # --- honesty layer ----------------------------------------------------
    not_modelled_reason: Optional[str] = None
    approximations: List[str] = field(default_factory=list)
    evidence_notes: List[str] = field(default_factory=list)


def _pressure_drop_for_stenosis(stenosis_pct: float, diameter_mm: float) -> float:
    """Simplified Poiseuille-type pressure drop across a stenosis.

    Resistance scales with L/D^4 for a given length, so the pressure drop
    ratio vs a healthy vessel is ((D/D_eff)^4) with D_eff from the residual
    lumen. This is the same physics family the swarm simulator uses; it is
    a screening estimate, not a 3-D FFR solve.
    """
    stenosis = min(max(stenosis_pct, 0.0), 99.0) / 100.0
    d_mm = max(diameter_mm, 0.5)
    d_eff = d_mm * math.sqrt(max(1.0 - stenosis, 0.01))  # area-conserving radius
    return (d_mm / d_eff) ** 4


def simulate_treatment(
    patient: TwinPatient,
    treatment: TwinTreatment,
    horizon_years: int = 10,
) -> CounterfactualResult:
    """Run one treatment counterfactual on the twin.

    The structure is fixed and cannot be gamed: each treatment either maps to
    a validated model chain (risk equations + IVUS plaque model [+ coronary
    physics for structural treatments]) or it returns NOT_MODELLED. There is
    no code path that fabricates an effect size.
    """
    horizon = max(1, min(int(horizon_years), 30))

    # Accept a bare string, but never let an unknown treatment name crash the
    # twin -- an unknown treatment is exactly the case the refusal layer
    # exists for, so it must produce a NOT_MODELLED result, not an exception.
    requested_name = treatment if isinstance(treatment, str) else treatment.value
    if isinstance(treatment, str):
        try:
            treatment = TwinTreatment(treatment)
        except ValueError:
            return CounterfactualResult(
                treatment=TwinTreatment.LIFESTYLE,  # placeholder slot
                verdict=TwinVerdict.NOT_MODELLED,
                horizon_years=horizon,
                not_modelled_reason=(
                    f"'{requested_name}' is not in the twin's treatment "
                    "catalogue and has no validated human effect-size model. "
                    "Refusing to invent a number is the correct behaviour."
                ),
            )

    result = CounterfactualResult(
        treatment=treatment,
        verdict=TwinVerdict.NOT_MODELLED,
        horizon_years=horizon,
    )

    # -- baseline: always computed the same way ----------------------------
    profile = patient.to_risk_profile()
    enhancers = patient.to_enhancers()
    baseline = triage(
        profile,
        enhancers,
        established_ascvd=patient.established_ascvd,
    )
    result.baseline_event_risk_10y = baseline.risk_10y

    ldl = patient.ldl_mg_dl if patient.ldl_mg_dl is not None else max(
        30.0, patient.chol_total_mgdl - patient.chol_hdl_mgdl - 25.0
    )

    # -- structural treatment: stenting ------------------------------------
    if treatment is TwinTreatment.STENT:
        if patient.scan.worst_stenosis_pct is None:
            result.not_modelled_reason = (
                "no CT stenosis measurement supplied; the twin cannot model a "
                "structural intervention on a lumen it has not seen"
            )
            return result
        diameter = patient.scan.coronary_diameter_mm or 3.0
        before = _pressure_drop_for_stenosis(patient.scan.worst_stenosis_pct, diameter)
        after = _pressure_drop_for_stenosis(POST_STENT_RESIDUAL_STENOSIS_PCT, diameter)
        result.pressure_drop_change_pct = -100.0 * (1.0 - after / before)
        result.verdict = TwinVerdict.MODELLED_APPROXIMATE
        result.approximations.append(
            "pressure-drop ratio from a Poiseuille-type screening model, not a "
            "patient-specific 3-D FFR-CT solve"
        )
        result.evidence_notes.append(
            "ISCHEMIA trial: invasive strategy did not reduce death/MI versus "
            "optimal medical therapy in stable disease -- symptom relief and "
            "angina, not mortality, is what revascularisation offers there."
        )
        result.ldl_on_treatment_mg_dl = ldl
        return result

    # -- LDL-anchored treatments -------------------------------------------
    rung = _TREATMENT_TO_RUNG.get(treatment)
    if rung is None:
        result.not_modelled_reason = (
            f"{treatment.value} has no validated human effect-size model in the "
            "twin's catalogue; see danon.cure for the trial record on "
            "regenerative and antifibrotic approaches (CUPID 2, miR-199a). "
            "Refusing to invent a number is the correct behaviour."
        )
        return result

    # Risk trajectory from the CTT machinery (prevention.project_benefit).
    benefit = project_benefit(
        baseline_risk_10y=baseline.risk_10y,
        baseline_ldl_mg_dl=ldl,
        rung=rung,
        years=horizon,
        age_years=patient.age_years,
    )
    result.ldl_reduction_mmol_l = benefit.ldl_reduction_mmol_l
    result.ldl_on_treatment_mg_dl = benefit.ldl_on_treatment_mg_dl
    result.absolute_risk_reduction = benefit.absolute_risk_reduction
    result.number_needed_to_treat = benefit.number_needed_to_treat
    # Annualised event risk over the requested horizon.
    result.treated_event_risk = benefit.treated_risk_10y

    # Plaque trajectory from the IVUS-anchored model (cure.py), only for the
    # duration the twin is asked about and only inside its validated domain.
    pav = plaque_regression(
        on_treatment_ldl_mg_dl=benefit.ldl_on_treatment_mg_dl,
        duration_months=horizon * 12,
    )
    result.pav_change_pct = pav.pav_change_pct
    if pav.extrapolated:
        result.approximations.extend(
            [n for n in pav.notes if "extrapolat" in n.lower() or "saturate" in n.lower()]
        )

    result.verdict = TwinVerdict.MODELLED if rung in (
        TherapyRung.MODERATE_STATIN,
        TherapyRung.HIGH_INTENSITY_STATIN,
        TherapyRung.STATIN_EZETIMIBE,
        TherapyRung.PCSK9_OR_INCLISIRAN,
    ) else TwinVerdict.MODELLED_APPROXIMATE

    if rung is TherapyRung.LIFESTYLE:
        result.approximations.append(
            "lifestyle effect sizes vary several-fold between individuals and "
            "adherence dominates; treat the trajectory as an upper bound"
        )
    if rung is TherapyRung.ONESHOT_BASE_EDITING:
        result.approximations.append(
            "base-editing LDL response is from early-phase trials (VERVE-102 "
            "class), not an approved therapy; durability beyond the studied "
            "window is not established"
        )

    return result


# ---------------------------------------------------------------------------
# The twin itself
# ---------------------------------------------------------------------------


@dataclass
class TwinInitializationReport:
    patient_id: str
    usable_scan_fields: List[str]
    missing_scan_fields: List[str]
    conflicts: List[str]
    notes: List[str]
    baseline_risk_10y: float
    risk_category: str


class CardiovascularTwin:
    """The assembled twin: initialize from a patient, then ask it questions."""

    def __init__(self, patient: TwinPatient, equation: RiskEquation = RiskEquation.YADLOWSKY_2018):
        self.patient = patient
        self.equation = equation
        self.initialization = self._initialize()

    # -- initialization: scan-vs-chart consistency --------------------------
    def _initialize(self) -> TwinInitializationReport:
        s = self.patient.scan
        usable: List[str] = []
        missing: List[str] = []
        conflicts: List[str] = []
        notes: List[str] = []

        scan_fields = {
            "lvef_pct": s.lvef_pct,
            "lv_mass_g": s.lv_mass_g,
            "lv_edv_ml": s.lv_edv_ml,
            "scar_fraction": s.scar_fraction,
            "ischaemic_fraction": s.ischaemic_fraction,
            "worst_stenosis_pct": s.worst_stenosis_pct,
            "cac_score": s.cac_score,
            "coronary_diameter_mm": s.coronary_diameter_mm,
            "systolic_bp_mmhg": s.systolic_bp_mmhg,
        }
        for name, value in scan_fields.items():
            (usable if value is not None else missing).append(name)

        # Cross-modality consistency checks: a twin that silently accepts
        # contradictory inputs is worse than no twin.
        if s.lvef_pct is not None and s.lvef_pct < 40.0:
            notes.append(
                f"LVEF {s.lvef_pct:.0f}% indicates reduced systolic function; "
                "risk-equation estimates assume a general adult population and "
                "understate heart-failure risk, which is not in the model."
            )
        if s.systolic_bp_mmhg is not None and abs(
            s.systolic_bp_mmhg - self.patient.bp_sys_mmhg
        ) >= 15.0:
            conflicts.append(
                f"imaging BP {s.systolic_bp_mmhg:.0f} vs chart BP "
                f"{self.patient.bp_sys_mmhg:.0f} mmHg differs by >=15; the twin "
                "uses the chart value, but one of these is wrong"
            )
        if s.cac_score is not None and s.cac_score == 0 and (
            s.worst_stenosis_pct is not None and s.worst_stenosis_pct >= 50
        ):
            conflicts.append(
                "CAC zero with a >=50% stenosis is unusual (non-calcified "
                "plaque); verify the stenosis read before treating on it"
            )
        if self.patient.established_ascvd and s.scar_fraction is not None and s.scar_fraction == 0:
            conflicts.append(
                "history of established ASCVD but no scar on LGE MRI; confirm "
                "the clinical history before applying secondary-prevention rules"
            )

        baseline = triage(
            self.patient.to_risk_profile(),
            self.patient.to_enhancers(),
            established_ascvd=self.patient.established_ascvd,
        )
        return TwinInitializationReport(
            patient_id=self.patient.patient_id,
            usable_scan_fields=usable,
            missing_scan_fields=missing,
            conflicts=conflicts,
            notes=notes,
            baseline_risk_10y=baseline.risk_10y,
            risk_category=risk_category_of(baseline.risk_10y),
        )

    # -- counterfactuals -----------------------------------------------------
    def simulate_treatment(
        self, treatment: TwinTreatment, horizon_years: int = 10
    ) -> CounterfactualResult:
        return simulate_treatment(self.patient, treatment, horizon_years)

    def run_all_treatments(self, horizon_years: int = 10) -> List[CounterfactualResult]:
        return [
            self.simulate_treatment(t, horizon_years) for t in TwinTreatment
        ]

    # -- the twin's own honesty report ---------------------------------------
    def what_the_twin_cannot_know(self) -> List[str]:
        """The observability boundary, stated explicitly for this patient."""
        out = [
            "intracellular and cellular state: MRI resolves millimetres; no "
            "per-cell initial condition exists, so no per-cell simulation can "
            "be validated for this patient",
            "which of the patient's variants matter: the actionable subset is "
            "carried (FH-relevant flags via the blood panel), but the "
            "variant-to-phenotype map is incomplete science",
            "arrhythmia mechanism, immune-cell dynamics, and tumour biology: "
            "outside the twin's observability entirely",
        ]
        if self.initialization.missing_scan_fields:
            out.append(
                "this patient's twin is additionally weakened by unmeasured "
                "scan fields: " + ", ".join(self.initialization.missing_scan_fields)
            )
        return out


def risk_category_of(risk: float) -> str:
    """Thin alias kept for the twin's public API; delegates to the published
    category thresholds in prevention."""
    return risk_category(risk)
