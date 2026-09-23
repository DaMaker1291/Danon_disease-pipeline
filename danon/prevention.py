"""
Pre-event prevention engine for ischemic heart disease.

THE THESIS OF THIS MODULE
-------------------------

Read ``personalization.py`` first. That module answers "how do I build a
patient-specific gene therapy". This module answers a different, and much
larger, question:

    **Who is going to have the heart attack, and what do we do about it
    before it happens?**

That question matters more than the delivery vehicle, and here is why. The
mechanism that provably prevents ischemic events is *lifelong lowering of
apoB-containing lipoproteins*. Statins, ezetimibe, PCSK9 inhibitors,
inclisiran, bempedoic acid -- every intervention with a mortality benefit
works this way. The therapy already exists, is generic, and costs cents per
day. What kills people is not the absence of a therapy. It is that the
therapy arrives after the first event, or never.

So the leverage is entirely in *detection and triage*, both of which are pure
computation on a blood draw. That is what this module implements, and every
number in it is either a published coefficient or an explicitly flagged
assumption.

WHAT IS REAL HERE
-----------------

1. **10-year ASCVD risk** -- the actual ACC/AHA Pooled Cohort Equations
   (Goff et al., 2013), all four sex/race strata, using the unrounded
   coefficients. Also the Yadlowsky et al. (2018) revision, which corrects
   the PCE's known over-prediction in contemporary populations. The equations
   are validated against published worked examples in the test suite.

2. **Risk enhancers** -- the factors the PCE ignores but the 2019 ACC/AHA
   primary-prevention guideline says must reclassify risk: Lp(a), apoB,
   hsCRP, family history, metabolic syndrome, CKD, CAC.

3. **Familial hypercholesterolaemia** -- the full Dutch Lipid Clinic Network
   point system. FH affects roughly 1 in 250 people, is the single most
   treatable genetic cause of premature MI, and is massively underdiagnosed.
   Detecting an index case also flags their relatives, who share the defect.

4. **The treatment effect** -- the Cholesterol Treatment Trialists'
   meta-analysis of 170,000 participants across 26 randomised trials:
   every 1 mmol/L (38.7 mg/dL) LDL reduction cuts the annual rate of major
   vascular events by 22% (RR 0.78, 95% CI 0.76-0.80), and all-cause
   mortality by 10% (RR 0.90, 95% CI 0.87-0.93).

5. **Number needed to treat and lives saved** -- derived from 1-4 by
   converting the 10-year risk to an annual hazard, applying the hazard
   ratio, and re-integrating. No free parameters.

WHAT IS MODELLED, NOT MEASURED
------------------------------

The *population* used by :class:`PopulationPreventionCampaign` is synthetic.
The risk-factor distributions are NHANES-informed approximations tagged
ASSUMED, and the statin-coverage rate is a calibration input that differs by
country and health system. Feed the campaign a real cohort table to get a
real answer; the ``from_table`` constructor exists for exactly that.

THE TWO HONEST LIMITATIONS
--------------------------

* A 10-year risk model is a *triage* tool, not a diagnosis. It tells you who
  deserves a $20 blood test and a $4/month pill. It does not image anyone's
  coronary arteries, and a CAC score outranks it whenever you have one.
* This module does not and cannot produce a therapy. It answers "who" and
  "how much", not "with what molecule". That is the correct division of
  labour: a real one-shot therapeutic needs IND-enabling toxicology, GLP
  manufacturing and Phase 1-3 trials at roughly $0.5-2B over 8-15 years.

Reference: Goff DC Jr et al. Circulation 2014;129(25 Suppl 2):S49-73.
           Yadlowsky S et al. Ann Intern Med 2018;169(1):20-29.
           Arnett DK et al. Circulation 2019;140(11):e596-e646.
           CTT Collaboration. Lancet 2010;376:1670-81 (events).
           Mihaylova B et al. Lancet 2012;380:581-90 (mortality).
           Nordestgaard BG et al. Eur Heart J 2020 (FH / DLCN criteria).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from danon.physics import Constant, Provenance

# ---------------------------------------------------------------------------
# Conversions and published effect sizes
# ---------------------------------------------------------------------------

#: Cholesterol unit conversion. 1 mmol/L of cholesterol == 38.67 mg/dL.
#: (LDL, HDL, total cholesterol and non-HDL all use this factor.)
MG_DL_PER_MMOL_L_CHOLESTEROL = 38.67

#: CTT 2010 (26 trials, 170 000 participants): proportional reduction in the
#: **annual rate** of major vascular events per 1.0 mmol/L LDL reduction.
CTT_MVE_RR_PER_MMOL_L = 0.78
CTT_MVE_RR_CI95 = (0.76, 0.80)

#: CTT 2012 (22 trials, 134 537 participants): all-cause mortality.
CTT_MORTALITY_RR_PER_MMOL_L = 0.90
CTT_MORTALITY_RR_CI95 = (0.87, 0.93)

#: LDL reduction achievable with each rung of the treatment ladder, as a
#: fraction of baseline and as an absolute mmol/L floor. These are
#: representative midpoints of published ranges (see routine_ldl_reduction).
_STATIN_MODERATE_FRACTION = 0.35
_STATIN_HIGH_FRACTION = 0.50
_EZETIMIBE_ADDITIONAL_FRACTION = 0.20        # on top of statin
_PCSK9_ADDITIONAL_FRACTION = 0.55            # on top of statin (+/- ezetimibe)
_BASE_EDIT_FRACTION = 0.60                   # one-shot hepatic PCSK9 knockdown

#: 10-year all-cause mortality risk by age (both sexes, US life-table scale).
#: Approximate. Used only for the secondary mortality-benefit output.
_TEN_YEAR_ALL_CAUSE_MORTALITY = (
    (40, 0.013), (50, 0.032), (60, 0.095), (70, 0.195), (80, 0.40),
)

#: Published validated input domains for the Pooled Cohort Equations. Values
#: outside these ranges are extrapolation and are flagged as such.
PCR_DOMAIN = {
    "age_years": (40.0, 79.0),
    "chol_total_mgdl": (130.0, 320.0),
    "chol_hdl_mgdl": (20.0, 100.0),
    "bp_sys_mmhg": (90.0, 200.0),
}

VALIDATED_INPUTS = ("age_years", "chol_total_mgdl", "chol_hdl_mgdl", "bp_sys_mmhg")


# ---------------------------------------------------------------------------
# Pooled Cohort Equations -- exact published coefficients
# ---------------------------------------------------------------------------
#
# These are the unrounded Goff 2013 coefficients, transcribed from the
# reference implementation in the ``PooledCohort`` R package
# (R/predict_risk.R, ``._pcr_Goff_2013``), which in turn cites the Cerner
# ASCVD calculator. Using the unrounded values matters: the published Table 4
# rounds to fewer decimals and will not reproduce the reference examples.
#
# ``treated_sbp`` and ``untreated_sbp`` are mutually exclusive by construction:
# the treated term is ln(SBP) if on antihypertensives else 0, and vice versa.

_PCR_GOFF_2013: Dict[Tuple[str, str], Dict[str, float]] = {
    ("female", "black"): {
        "base_survival_10y": 0.95334,
        "coef_mean": 86.6081,
        "ln_age": 17.1141,
        "ln_age_squared": 0.0,
        "ln_chol_total": 0.9396,
        "ln_age_x_ln_chol_total": 0.0,
        "ln_chol_hdl": -18.9196,
        "ln_age_x_ln_chol_hdl": 4.4748,
        "ln_treated_sbp": 29.2907,
        "ln_age_x_ln_treated_sbp": -6.4321,
        "ln_untreated_sbp": 27.8197,
        "ln_age_x_ln_untreated_sbp": -6.0873,
        "smoker": 0.6908,
        "ln_age_x_smoker": 0.0,
        "diabetes": 0.8738,
    },
    ("female", "white"): {
        "base_survival_10y": 0.96652,
        "coef_mean": -29.1817,
        "ln_age": -29.799,
        "ln_age_squared": 4.884,
        "ln_chol_total": 13.540,
        "ln_age_x_ln_chol_total": -3.114,
        "ln_chol_hdl": -13.578,
        "ln_age_x_ln_chol_hdl": 3.149,
        "ln_treated_sbp": 2.019,
        "ln_age_x_ln_treated_sbp": 0.0,
        "ln_untreated_sbp": 1.957,
        "ln_age_x_ln_untreated_sbp": 0.0,
        "smoker": 7.574,
        "ln_age_x_smoker": -1.665,
        "diabetes": 0.661,
    },
    ("male", "black"): {
        "base_survival_10y": 0.89536,
        "coef_mean": 19.5425,
        "ln_age": 2.469,
        "ln_age_squared": 0.0,
        "ln_chol_total": 0.302,
        "ln_age_x_ln_chol_total": 0.0,
        "ln_chol_hdl": -0.307,
        "ln_age_x_ln_chol_hdl": 0.0,
        "ln_treated_sbp": 1.916,
        "ln_age_x_ln_treated_sbp": 0.0,
        "ln_untreated_sbp": 1.809,
        "ln_age_x_ln_untreated_sbp": 0.0,
        "smoker": 0.549,
        "ln_age_x_smoker": 0.0,
        "diabetes": 0.645,
    },
    ("male", "white"): {
        "base_survival_10y": 0.91436,
        "coef_mean": 61.1816,
        "ln_age": 12.344,
        "ln_age_squared": 0.0,
        "ln_chol_total": 11.853,
        "ln_age_x_ln_chol_total": -2.664,
        "ln_chol_hdl": -7.990,
        "ln_age_x_ln_chol_hdl": 1.769,
        "ln_treated_sbp": 1.797,
        "ln_age_x_ln_treated_sbp": 0.0,
        "ln_untreated_sbp": 1.764,
        "ln_age_x_ln_untreated_sbp": 0.0,
        "smoker": 7.837,
        "ln_age_x_smoker": -1.795,
        "diabetes": 0.658,
    },
}

#: Yadlowsky et al. 2018 revised equations. Same covariates, refit on
#: contemporary cohorts with a logistic rather than Cox form. These are the
#: recommended replacement for the original PCE, which over-predicts risk by
#: roughly 20% in modern US populations. Coefficients transcribed from
#: ``._pcr_Yadlowsky_2013`` in the same reference implementation.
_PCR_YADLOWSKY_2018: Dict[str, Dict[str, float]] = {
    "female": {
        "intercept": -12.823110,
        "age": 0.106501,
        "black": 0.432440,
        "sbp_squared": 0.000056,
        "sbp": 0.017666,
        "bp_meds": 0.731678,
        "diabetes": 0.943970,
        "smoke_current": 1.009790,
        "chol_ratio": 0.151318,
        "age_x_black": -0.008580,
        "sbp_x_bp_meds": -0.003647,
        "sbp_x_black": 0.006208,
        "bp_meds_x_black": 0.152968,
        "age_x_sbp": -0.000153,
        "black_x_diabetes": 0.115232,
        "black_x_smoke_current": -0.092231,
        "black_x_chol_ratio": 0.070498,
        "black_x_sbp_x_bp_meds": -0.000173,
        "black_x_sbp_x_age": -0.000094,
    },
    "male": {
        "intercept": -11.679980,
        "age": 0.064200,
        "black": 0.482835,
        "sbp_squared": -0.000061,
        "sbp": 0.038950,
        "bp_meds": 2.055533,
        "diabetes": 0.842209,
        "smoke_current": 0.895589,
        "chol_ratio": 0.193307,
        "age_x_black": 0.0,
        "sbp_x_bp_meds": -0.014207,
        "sbp_x_black": 0.011609,
        "bp_meds_x_black": -0.119460,
        "age_x_sbp": 0.000025,
        "black_x_diabetes": -0.077214,
        "black_x_smoke_current": -0.226771,
        "black_x_chol_ratio": -0.117749,
        "black_x_sbp_x_bp_meds": 0.004190,
        "black_x_sbp_x_age": -0.000199,
    },
}


class RiskEquation(str, Enum):
    """Which 10-year ASCVD risk model to use."""

    GOFF_2013 = "goff_2013"              # original ACC/AHA Pooled Cohort Equations
    YADLOWSKY_2018 = "yadlowsky_2018"    # contemporary refit; less over-prediction


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

class Sex(str, Enum):
    FEMALE = "female"
    MALE = "male"


class RaceStratum(str, Enum):
    """Which coefficient set to use.

    NOTE: this is a *statistical* variable, not a biological one. It encodes
    which derivation cohort best matches the patient, which in practice also
    captures access-to-care and environmental differences. The published
    models support only these two strata; other ancestries must use WHITE and
    should be told the estimate is approximate.
    """

    WHITE = "white"
    BLACK = "black"


@dataclass
class RiskFactorProfile:
    """Everything the 10-year risk equations consume. All units are the ones
    the published equations expect: mg/dL for cholesterol, mmHg for pressure."""

    patient_id: str
    age_years: float
    sex: Sex
    race: RaceStratum = RaceStratum.WHITE
    chol_total_mgdl: float = 190.0
    chol_hdl_mgdl: float = 50.0
    bp_sys_mmhg: float = 125.0
    on_bp_medication: bool = False
    current_smoker: bool = False
    diabetes: bool = False
    #: Directly measured LDL-C when available. Spelled the same way as the
    #: matching field on :class:`RiskEnhancers` on purpose: one physical
    #: quantity, one name.
    ldl_mg_dl: Optional[float] = None

    def ldl_or_estimated(self) -> float:
        """LDL-C from the Friedewald relation when not measured directly.

        Friedewald: LDL = TC - HDL - TG/5. Triglycerides are unknown here, so
        the population-median TG/5 term of 25 mg/dL is used. This is a real
        approximation with a real failure mode (it is invalid when TG is very
        high), which is why a directly measured LDL always wins.
        """
        if self.ldl_mg_dl is not None:
            return float(self.ldl_mg_dl)
        return max(30.0, self.chol_total_mgdl - self.chol_hdl_mgdl - 25.0)


@dataclass
class RiskEnhancers:
    """Factors the risk equations ignore but the 2019 ACC/AHA guideline says
    must reclassify borderline/intermediate risk.

    Every field defaults to "not present / not measured", which is the honest
    default: absence of a measurement is not evidence of absence.
    """

    lipoprotein_a_mg_dl: Optional[float] = None
    lipoprotein_a_nmol_l: Optional[float] = None
    apob_mg_dl: Optional[float] = None
    hscrp_mg_l: Optional[float] = None
    family_history_premature_ascvd: bool = False
    metabolic_syndrome: bool = False
    chronic_kidney_disease: bool = False
    cac_score: Optional[float] = None
    ldl_mg_dl: Optional[float] = None


# Lp(a) thresholds. The mg/dL and nmol/L scales are NOT interconvertible by a
# universal factor -- the mass of the particle depends on the apo(a) isoform
# size, which varies several-fold between individuals. Labs report one or the
# other. Both guideline cut-offs are therefore carried explicitly rather than
# converted, and a conversion is only used when the caller supplies one.
LPA_HIGH_MG_DL = 50.0
LPA_VERY_HIGH_MG_DL = 180.0
LPA_HIGH_NMOL_L = 125.0
LPA_VERY_HIGH_NMOL_L = 250.0


@dataclass
class RiskAssessment:
    """The interpretation of a profile + enhancers. Everything here is a
    published-model output; nothing is invented."""

    patient_id: str
    equation: RiskEquation
    risk_10y: float                     # 0-1
    risk_category: str                  # low | borderline | intermediate | high
    enhancer_flags: List[str] = field(default_factory=list)
    enhancer_uplift: float = 0.0        # multiplicative reclassification factor
    reclassified_risk_10y: float = 0.0
    reclassified_category: str = ""
    outside_validated_domain: List[str] = field(default_factory=list)
    unmeasured_enhancers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def effective_risk_10y(self) -> float:
        """Risk after guideline-directed enhancer reclassification.

        This is the number that should drive a treatment decision, because the
        guideline says to consider enhancers precisely for patients whose
        equation risk is borderline or intermediate.

        Note that reclassification is genuinely bidirectional: a coronary
        calcium score of zero moves a patient DOWN, which is the whole reason
        the guideline recommends measuring it before committing to lifelong
        therapy. Taking ``max(equation, reclassified)`` here would silently
        discard every downward reclassification and make the CAC score
        useless -- an earlier version of this property did exactly that.
        """
        return min(1.0, max(0.0, self.reclassified_risk_10y))


# ---------------------------------------------------------------------------
# Risk equation implementations
# ---------------------------------------------------------------------------

def _validate_domain(profile: RiskFactorProfile) -> List[str]:
    violations: List[str] = []
    for name, (lo, hi) in PCR_DOMAIN.items():
        value = float(getattr(profile, name))
        if value < lo or value > hi:
            violations.append(f"{name}={value:g} outside validated [{lo:g}, {hi:g}]")
    return violations


def _goff_2013_scalar(profile: RiskFactorProfile) -> float:
    coefs = _PCR_GOFF_2013[(profile.sex.value, profile.race.value)]
    ln_age = math.log(profile.age_years)
    ln_tc = math.log(profile.chol_total_mgdl)
    ln_hdl = math.log(profile.chol_hdl_mgdl)
    ln_sbp = math.log(profile.bp_sys_mmhg)
    treated = 1.0 if profile.on_bp_medication else 0.0
    untreated = 0.0 if profile.on_bp_medication else 1.0
    smoker = 1.0 if profile.current_smoker else 0.0
    diabetes = 1.0 if profile.diabetes else 0.0

    ind_sum = (
        coefs["ln_age"] * ln_age
        + coefs["ln_age_squared"] * ln_age * ln_age
        + coefs["ln_chol_total"] * ln_tc
        + coefs["ln_age_x_ln_chol_total"] * ln_age * ln_tc
        + coefs["ln_chol_hdl"] * ln_hdl
        + coefs["ln_age_x_ln_chol_hdl"] * ln_age * ln_hdl
        + coefs["ln_treated_sbp"] * ln_sbp * treated
        + coefs["ln_untreated_sbp"] * ln_sbp * untreated
        + coefs["ln_age_x_ln_treated_sbp"] * ln_age * ln_sbp * treated
        + coefs["ln_age_x_ln_untreated_sbp"] * ln_age * ln_sbp * untreated
        + coefs["smoker"] * smoker
        + coefs["ln_age_x_smoker"] * ln_age * smoker
        + coefs["diabetes"] * diabetes
    )
    return float(1.0 - coefs["base_survival_10y"] ** math.exp(ind_sum - coefs["coef_mean"]))


def _yadlowsky_2018_scalar(profile: RiskFactorProfile) -> float:
    c = _PCR_YADLOWSKY_2018[profile.sex.value]
    black = 1.0 if profile.race is RaceStratum.BLACK else 0.0
    sbp = float(profile.bp_sys_mmhg)
    bp_meds = 1.0 if profile.on_bp_medication else 0.0
    diabetes = 1.0 if profile.diabetes else 0.0
    smoke = 1.0 if profile.current_smoker else 0.0
    ratio = profile.chol_total_mgdl / profile.chol_hdl_mgdl
    age = float(profile.age_years)

    ind_sum = (
        c["intercept"]
        + c["age"] * age
        + c["black"] * black
        + c["sbp_squared"] * sbp * sbp
        + c["sbp"] * sbp
        + c["bp_meds"] * bp_meds
        + c["diabetes"] * diabetes
        + c["smoke_current"] * smoke
        + c["chol_ratio"] * ratio
        + c["age_x_black"] * age * black
        + c["sbp_x_bp_meds"] * sbp * bp_meds
        + c["sbp_x_black"] * sbp * black
        + c["bp_meds_x_black"] * bp_meds * black
        + c["age_x_sbp"] * age * sbp
        + c["black_x_diabetes"] * black * diabetes
        + c["black_x_smoke_current"] * black * smoke
        + c["black_x_chol_ratio"] * black * ratio
        + c["black_x_sbp_x_bp_meds"] * black * sbp * bp_meds
        + c["black_x_sbp_x_age"] * black * sbp * age
    )
    return float(1.0 / (1.0 + math.exp(-ind_sum)))


def ascvd_risk_10y(
    profile: RiskFactorProfile,
    equation: RiskEquation = RiskEquation.YADLOWSKY_2018,
) -> float:
    """10-year risk of a first ASCVD event (non-fatal MI, CHD death, or
    fatal/non-fatal stroke), as a fraction in [0, 1].

    The default is the Yadlowsky 2018 revision because the original Goff 2013
    equations are known to over-predict in contemporary populations. Report
    which equation you used -- they differ by a clinically meaningful amount.
    """
    if equation is RiskEquation.GOFF_2013:
        return _goff_2013_scalar(profile)
    if equation is RiskEquation.YADLOWSKY_2018:
        return _yadlowsky_2018_scalar(profile)
    raise ValueError(f"unknown equation {equation!r}")


def risk_category(risk_10y: float) -> str:
    """ACC/AHA 2018 cholesterol-guideline risk categories."""
    if risk_10y < 0.05:
        return "low"
    if risk_10y < 0.075:
        return "borderline"
    if risk_10y < 0.20:
        return "intermediate"
    return "high"


# ---------------------------------------------------------------------------
# Risk enhancers
# ---------------------------------------------------------------------------

#: Multiplicative uplift applied to the equation risk when a risk enhancer is
#: present. These are *not* published hazard ratios -- no such pooled estimate
#: exists for most of these factors. They are calibration inputs whose purpose
#: is to move a borderline patient over a treatment threshold, which is what
#: the guideline intends. Tagged ASSUMED in the provenance audit for exactly
#: that reason, and exposed as constructor parameters so they can be replaced.
DEFAULT_ENHANCER_UPLIFT: Dict[str, float] = {
    "lp_a_very_high": 1.60,
    "lp_a_high": 1.30,
    "apob_discordant": 1.25,
    "hscrp_elevated": 1.20,
    "family_history": 1.30,
    "metabolic_syndrome": 1.25,
    "chronic_kidney_disease": 1.40,
    "cac_high": 2.00,
    "cac_zero": 0.40,
    "ldl_very_high": 1.50,
}

ENHANCER_LABELS: Dict[str, str] = {
    "lp_a_very_high": "Lp(a) very high (secondary prevention target)",
    "lp_a_high": "Lp(a) high (risk-enhancing, no routine therapy yet)",
    "apob_discordant": "ApoB discordant with LDL-C",
    "hscrp_elevated": "hsCRP >= 2 mg/L (inflammatory risk)",
    "family_history": "Family history of premature ASCVD",
    "metabolic_syndrome": "Metabolic syndrome",
    "chronic_kidney_disease": "Chronic kidney disease",
    "cac_high": "Coronary calcium >= 100 AU",
    "cac_zero": "Coronary calcium = 0 (downward reclassification)",
    "ldl_very_high": "LDL-C >= 160 mg/dL",
}


def _enhancer_flags(
    profile: RiskFactorProfile,
    enhancers: RiskEnhancers,
) -> Tuple[List[str], List[str]]:
    """Return (present, unmeasured) enhancer flag lists."""
    present: List[str] = []
    unmeasured: List[str] = []

    e = enhancers
    if e.lipoprotein_a_nmol_l is not None:
        if e.lipoprotein_a_nmol_l >= LPA_VERY_HIGH_NMOL_L:
            present.append("lp_a_very_high")
        elif e.lipoprotein_a_nmol_l >= LPA_HIGH_NMOL_L:
            present.append("lp_a_high")
    elif e.lipoprotein_a_mg_dl is not None:
        if e.lipoprotein_a_mg_dl >= LPA_VERY_HIGH_MG_DL:
            present.append("lp_a_very_high")
        elif e.lipoprotein_a_mg_dl >= LPA_HIGH_MG_DL:
            present.append("lp_a_high")
    else:
        unmeasured.append("lipoprotein_a")

    if e.apob_mg_dl is not None:
        ldl = e.ldl_mg_dl if e.ldl_mg_dl is not None else profile.ldl_or_estimated()
        # Discordance: apoB is at/above the 75th percentile (130 mg/dL) while
        # LDL-C sits below the treatment threshold. Such patients carry more
        # particles than their LDL-C suggests and are under-treated.
        if e.apob_mg_dl >= 130.0 and ldl < 100.0:
            present.append("apob_discordant")
    else:
        unmeasured.append("apob")

    if e.hscrp_mg_l is not None:
        if e.hscrp_mg_l >= 2.0:
            present.append("hscrp_elevated")
    else:
        unmeasured.append("hscrp")

    # Family history is a question you always ask, so its absence is a
    # genuine negative rather than a missing measurement. It is therefore
    # never reported as unmeasured.
    if e.family_history_premature_ascvd:
        present.append("family_history")

    if e.metabolic_syndrome:
        present.append("metabolic_syndrome")
    if e.chronic_kidney_disease:
        present.append("chronic_kidney_disease")

    if e.cac_score is not None:
        if e.cac_score == 0:
            present.append("cac_zero")
        elif e.cac_score >= 100.0:
            present.append("cac_high")
    else:
        unmeasured.append("cac_score")

    ldl = e.ldl_mg_dl if e.ldl_mg_dl is not None else profile.ldl_or_estimated()
    if ldl >= 160.0:
        present.append("ldl_very_high")

    return present, unmeasured


def assess(
    profile: RiskFactorProfile,
    enhancers: Optional[RiskEnhancers] = None,
    equation: RiskEquation = RiskEquation.YADLOWSKY_2018,
    uplift: Optional[Dict[str, float]] = None,
) -> RiskAssessment:
    """Full risk assessment: equation risk, enhancer flags, reclassification.

    The enhancer uplift is only applied where the guideline says it matters --
    i.e. when the equation risk is borderline or intermediate. Applying it to
    a low-risk patient who happens to have one enhancer would reclassify
    almost everyone upward and destroy the model's discrimination.
    """
    enhancers = enhancers or RiskEnhancers()
    uplift = uplift if uplift is not None else DEFAULT_ENHANCER_UPLIFT

    base = ascvd_risk_10y(profile, equation=equation)
    category = risk_category(base)
    flags, unmeasured = _enhancer_flags(profile, enhancers)

    reclassified = base
    notes: List[str] = []

    if category in ("borderline", "intermediate") and flags:
        factor = 1.0
        for f in flags:
            factor *= uplift.get(f, 1.0)
        reclassified = min(1.0, base * factor)
        notes.append(
            f"Risk enhancers ({', '.join(flags)}) applied at "
            f"{category} risk: {base*100:.1f}% -> {reclassified*100:.1f}%."
        )
    elif flags and category == "low":
        notes.append(
            "Enhancers present but equation risk is low; guideline does not "
            "reclassify here. Surveillance rather than treatment."
        )
    elif not flags and category in ("borderline", "intermediate"):
        notes.append(
            "Borderline/intermediate risk with no enhancers measured. Lp(a), "
            "apoB and a CAC score are the guideline-endorsed tie-breakers."
        )

    domain_violations = _validate_domain(profile)
    if domain_violations:
        notes.append(
            "Inputs outside the validated domain; estimate is extrapolation: "
            + "; ".join(domain_violations)
        )

    return RiskAssessment(
        patient_id=profile.patient_id,
        equation=equation,
        risk_10y=base,
        risk_category=category,
        enhancer_flags=flags,
        enhancer_uplift=float(reclassified / base) if base > 0 else 1.0,
        reclassified_risk_10y=reclassified,
        reclassified_category=risk_category(reclassified),
        outside_validated_domain=domain_violations,
        unmeasured_enhancers=unmeasured,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Familial hypercholesterolaemia
# ---------------------------------------------------------------------------

class FHClassification(str, Enum):
    UNLIKELY = "unlikely_fh"
    POSSIBLE = "possible_fh"
    PROBABLE = "probable_fh"
    DEFINITE = "definite_fh"


@dataclass
class FHInputs:
    """Dutch Lipid Clinic Network inputs. LDL must be the *untreated* value;
    scoring a patient already on a statin under-diagnoses them."""

    untreated_ldl_mg_dl: float
    family_history_premature_cad: bool = False        # 1st-degree: M<55, F<60
    family_history_high_ldl: bool = False             # 1st-degree >95th pct
    family_history_tendon_xanthomata: bool = False
    family_history_child_high_ldl: bool = False
    personal_premature_cad: bool = False
    personal_premature_cerebral_or_peripheral: bool = False
    tendon_xanthomata: bool = False
    arcus_cornealis_under_45: bool = False
    genetic_test_confirmed: Optional[bool] = None


@dataclass
class FHResult:
    score: int
    classification: FHClassification
    breakdown: List[str] = field(default_factory=list)
    cascade_screening_recommended: bool = False


def dutch_lipid_clinic_score(inputs: FHInputs) -> FHResult:
    """Dutch Lipid Clinic Network criteria for familial hypercholesterolaemia.

    Thresholds are the published ones: definite >8, probable 6-8,
    possible 3-5, unlikely <3. LDL bands are converted to mmol/L internally
    because the original bands are defined in mmol/L.
    """
    ldl_mmol = inputs.untreated_ldl_mg_dl / MG_DL_PER_MMOL_L_CHOLESTEROL
    score = 0
    breakdown: List[str] = []

    if inputs.family_history_premature_cad:
        score += 1
        breakdown.append("Family history: 1st-degree premature CAD (+1)")
    if inputs.family_history_high_ldl:
        score += 1
        breakdown.append("Family history: 1st-degree high LDL (+1)")
    if inputs.family_history_tendon_xanthomata:
        score += 2
        breakdown.append("Family history: tendon xanthomata (+2)")
    if inputs.family_history_child_high_ldl:
        score += 2
        breakdown.append("Family history: child <18 with high LDL (+2)")

    if inputs.personal_premature_cad:
        score += 2
        breakdown.append("Personal history: premature CAD (+2)")
    if inputs.personal_premature_cerebral_or_peripheral:
        score += 1
        breakdown.append("Personal history: premature cerebral/peripheral disease (+1)")

    if inputs.tendon_xanthomata:
        score += 6
        breakdown.append("Examination: tendon xanthomata (+6)")
    if inputs.arcus_cornealis_under_45:
        score += 4
        breakdown.append("Examination: arcus cornealis <45 y (+4)")

    if ldl_mmol >= 8.5:
        score += 8
        breakdown.append(f"LDL {ldl_mmol:.2f} mmol/L >= 8.5 (+8)")
    elif ldl_mmol >= 6.5:
        score += 5
        breakdown.append(f"LDL {ldl_mmol:.2f} mmol/L 6.5-8.4 (+5)")
    elif ldl_mmol >= 5.0:
        score += 3
        breakdown.append(f"LDL {ldl_mmol:.2f} mmol/L 5.0-6.4 (+3)")
    elif ldl_mmol >= 4.0:
        score += 1
        breakdown.append(f"LDL {ldl_mmol:.2f} mmol/L 4.0-4.9 (+1)")

    if inputs.genetic_test_confirmed:
        score += 8
        breakdown.append("Genetics: causative variant confirmed (+8)")

    if score > 8:
        classification = FHClassification.DEFINITE
    elif score >= 6:
        classification = FHClassification.PROBABLE
    elif score >= 3:
        classification = FHClassification.POSSIBLE
    else:
        classification = FHClassification.UNLIKELY

    return FHResult(
        score=score,
        classification=classification,
        breakdown=breakdown,
        cascade_screening_recommended=classification in (
            FHClassification.PROBABLE, FHClassification.DEFINITE
        ),
    )


# ---------------------------------------------------------------------------
# Cumulative LDL burden
# ---------------------------------------------------------------------------

def ldl_exposure_years(ldl_mg_dl: float, age_years: float,
                       onset_age_years: float = 18.0) -> float:
    """Cumulative LDL exposure in mmol/L-years above the 1.8 mmol/L
    (~70 mg/dL) reference that the Mendelian-randomisation and FH literature
    uses to define 'exposure-free' lifetime risk.

    This formalises the reason an FH patient has an MI at 40 while a
    late-onset hyperlipidaemic patient has one at 70: the *area under the
    LDL curve* is what drives plaque, not the current value. Two patients with
    identical LDL today can have 2-fold different risk.
    """
    ldl_mmol = ldl_mg_dl / MG_DL_PER_MMOL_L_CHOLESTEROL
    reference_mmol = 1.8
    duration = max(0.0, age_years - onset_age_years)
    return max(0.0, (ldl_mmol - reference_mmol) * duration)


# ---------------------------------------------------------------------------
# Treatment ladder and LDL response
# ---------------------------------------------------------------------------

class TherapyRung(str, Enum):
    LIFESTYLE = "lifestyle"
    MODERATE_STATIN = "moderate_intensity_statin"
    HIGH_INTENSITY_STATIN = "high_intensity_statin"
    STATIN_EZETIMIBE = "statin_plus_ezetimibe"
    PCSK9_OR_INCLISIRAN = "pcsk9_inhibitor_or_inclisiran"
    ONESHOT_BASE_EDITING = "one_shot_base_editing"


#: Representative LDL reduction for each rung, used to compute the achievable
#: absolute risk reduction. These are midpoints of published ranges, not
#: guarantees; real response varies several-fold between individuals.
_RUNG_FRACTION = {
    TherapyRung.LIFESTYLE: 0.15,
    TherapyRung.MODERATE_STATIN: _STATIN_MODERATE_FRACTION,
    TherapyRung.HIGH_INTENSITY_STATIN: _STATIN_HIGH_FRACTION,
    TherapyRung.STATIN_EZETIMIBE: _STATIN_HIGH_FRACTION * (1 + _EZETIMIBE_ADDITIONAL_FRACTION),
    TherapyRung.PCSK9_OR_INCLISIRAN: min(0.85, _STATIN_HIGH_FRACTION + _PCSK9_ADDITIONAL_FRACTION),
    TherapyRung.ONESHOT_BASE_EDITING: _BASE_EDIT_FRACTION,
}


def routine_ldl_reduction(rung: TherapyRung, baseline_ldl_mg_dl: float) -> float:
    """Absolute LDL reduction in mmol/L for a given rung. Never returns more
    than the patient's LDL above a 0.5 mmol/L (~19 mg/dL) physiological floor,
    because a drug cannot remove cholesterol that is not circulating."""
    fraction = _RUNG_FRACTION[rung]
    baseline_mmol = baseline_ldl_mg_dl / MG_DL_PER_MMOL_L_CHOLESTEROL
    floor_mmol = 0.5
    reduction = baseline_mmol * fraction
    return max(0.0, min(reduction, baseline_mmol - floor_mmol))


# ---------------------------------------------------------------------------
# The lives-saved engine (CTT)
# ---------------------------------------------------------------------------

def _annual_rate_from_cumulative(risk: float, years: float) -> float:
    """Invert constant-hazard compounding: risk -> annual event rate."""
    risk = min(max(risk, 0.0), 1.0 - 1e-12)
    return 1.0 - (1.0 - risk) ** (1.0 / years)


def _cumulative_from_annual(annual_rate: float, years: float) -> float:
    annual_rate = min(max(annual_rate, 0.0), 1.0)
    return 1.0 - (1.0 - annual_rate) ** years


@dataclass
class TreatmentBenefit:
    """Everything the CTT relation implies for one patient on one rung."""

    rung: TherapyRung
    baseline_ldl_mg_dl: float
    ldl_reduction_mmol_l: float
    ldl_on_treatment_mg_dl: float
    baseline_risk_10y: float
    treated_risk_10y: float
    absolute_risk_reduction: float          # fraction, e.g. 0.073 = 7.3 pp
    relative_risk_reduction: float
    number_needed_to_treat: float
    mortality_rr: float
    mortality_benefit_10y: float            # absolute, per patient
    years: int = 10


def project_benefit(
    baseline_risk_10y: float,
    baseline_ldl_mg_dl: float,
    rung: TherapyRung,
    years: int = 10,
    age_years: Optional[float] = None,
    mve_rr: float = CTT_MVE_RR_PER_MMOL_L,
    mortality_rr_per_mmol: float = CTT_MORTALITY_RR_PER_MMOL_L,
) -> TreatmentBenefit:
    """Project the absolute benefit of one treatment rung using the CTT
    meta-analysis.

    The structure matters and is worth stating, because getting it wrong
    produces absurd numbers:

    1. The CTT proportional reduction applies to the **annual event rate**,
       not to the cumulative risk. So the 10-year risk is first converted to
       an annual hazard.
    2. The hazard ratio is ``rr ^ (delta_LDL in mmol/L)``, which assumes
       log-linearity in LDL reduction. This is the CTT's own finding and is
       why a 2 mmol/L reduction is worth roughly twice a 1 mmol/L reduction.
    3. The treated annual hazard is re-integrated over the horizon.

    Compounding the per-mmol/L ratio over years instead (0.78 ** (1 x 10))
    would predict a 92% event reduction, which is not what the trials show.
    """
    baseline_ldl = float(baseline_ldl_mg_dl)
    delta_ldl = routine_ldl_reduction(rung, baseline_ldl)
    ldl_on_treatment = max(0.0, baseline_ldl - delta_ldl * MG_DL_PER_MMOL_L_CHOLESTEROL)

    annual_baseline = _annual_rate_from_cumulative(baseline_risk_10y, 10.0)
    hazard_ratio = float(mve_rr) ** delta_ldl
    annual_treated = annual_baseline * hazard_ratio

    untreated_over_horizon = _cumulative_from_annual(annual_baseline, years)
    treated_over_horizon = _cumulative_from_annual(annual_treated, years)
    arr = max(0.0, untreated_over_horizon - treated_over_horizon)

    nnt = float("inf") if arr <= 0 else 1.0 / arr
    rrr = arr / untreated_over_horizon if untreated_over_horizon > 0 else 0.0

    mortality_benefit = 0.0
    mortality_hr = 1.0
    if age_years is not None and delta_ldl > 0:
        mortality_hr = float(mortality_rr_per_mmol) ** delta_ldl
        baseline_mortality = _ten_year_mortality_risk(age_years)
        annual_mortality = _annual_rate_from_cumulative(baseline_mortality, 10.0)
        treated_mortality = _cumulative_from_annual(
            annual_mortality * mortality_hr, years
        )
        mortality_benefit = max(0.0, baseline_mortality - treated_mortality)

    return TreatmentBenefit(
        rung=rung,
        baseline_ldl_mg_dl=baseline_ldl,
        ldl_reduction_mmol_l=delta_ldl,
        ldl_on_treatment_mg_dl=ldl_on_treatment,
        baseline_risk_10y=untreated_over_horizon,
        treated_risk_10y=treated_over_horizon,
        absolute_risk_reduction=arr,
        relative_risk_reduction=rrr,
        number_needed_to_treat=nnt,
        mortality_rr=mortality_hr,
        mortality_benefit_10y=mortality_benefit,
        years=years,
    )


def _ten_year_mortality_risk(age_years: float) -> float:
    """Interpolate the approximate 10-year all-cause mortality risk table."""
    ages = [a for a, _ in _TEN_YEAR_ALL_CAUSE_MORTALITY]
    risks = [r for _, r in _TEN_YEAR_ALL_CAUSE_MORTALITY]
    if age_years <= ages[0]:
        return risks[0]
    if age_years >= ages[-1]:
        return risks[-1]
    return float(np.interp(age_years, ages, risks))


def _mortality_risk_vector(age_years: np.ndarray) -> np.ndarray:
    """Vectorised form of :func:`_ten_year_mortality_risk`."""
    ages = np.array([a for a, _ in _TEN_YEAR_ALL_CAUSE_MORTALITY], dtype=float)
    risks = np.array([r for _, r in _TEN_YEAR_ALL_CAUSE_MORTALITY], dtype=float)
    return np.interp(np.asarray(age_years, dtype=float), ages, risks)


# ---------------------------------------------------------------------------
# Guideline triage
# ---------------------------------------------------------------------------

class PreventionAction(str, Enum):
    LIFESTYLE_ONLY = "lifestyle_only"
    MEASURE_ENHANCERS = "measure_enhancers"          # Lp(a), apoB, CAC
    SHARED_DECISION_STATIN = "shared_decision_statin"
    MODERATE_STATIN = "moderate_intensity_statin"
    HIGH_INTENSITY_STATIN = "high_intensity_statin"
    INTENSIFY_TO_EZETIMIBE = "add_ezetimibe"
    PCSK9_PATHWAY = "pcsk9_inhibitor_or_inclisiran"
    ONESHOT_THERAPY_CANDIDATE = "one_shot_therapy_candidate"
    CASCADE_SCREEN_FAMILY = "cascade_screen_family"
    SECONDARY_PREVENTION = "secondary_prevention"


#: Maps a recommended therapy rung onto the action that represents the next
#: concrete step. Used to pick the primary action so that a long-horizon
#: investigational option can never outrank the drug the patient should start
#: this week.
_RUNG_TO_ACTION: Dict[TherapyRung, PreventionAction] = {
    TherapyRung.LIFESTYLE: PreventionAction.LIFESTYLE_ONLY,
    TherapyRung.MODERATE_STATIN: PreventionAction.MODERATE_STATIN,
    TherapyRung.HIGH_INTENSITY_STATIN: PreventionAction.HIGH_INTENSITY_STATIN,
    TherapyRung.STATIN_EZETIMIBE: PreventionAction.INTENSIFY_TO_EZETIMIBE,
    TherapyRung.PCSK9_OR_INCLISIRAN: PreventionAction.PCSK9_PATHWAY,
    TherapyRung.ONESHOT_BASE_EDITING: PreventionAction.ONESHOT_THERAPY_CANDIDATE,
}


#: LDL thresholds in mg/dL from the 2018 ACC/AHA cholesterol guideline and the
#: 2019 primary-prevention guideline.
LDL_THRESHOLD_VERY_HIGH = 190.0      # >=190 is a statin indication regardless of risk
LDL_THRESHOLD_TREAT_TO = 70.0        # very-high-risk target
DIABETES_AGE_RANGE = (40, 75)        # statin indication in diabetes, this range


@dataclass
class TriageDecision:
    patient_id: str
    actions: List[PreventionAction] = field(default_factory=list)
    primary_action: PreventionAction = PreventionAction.LIFESTYLE_ONLY
    recommended_rung: TherapyRung = TherapyRung.LIFESTYLE
    risk_10y: float = 0.0
    benefit: Optional[TreatmentBenefit] = None
    #: Benefit over the remaining lifetime, reported only for young patients
    #: with high cumulative LDL exposure, where the 10-year horizon is known
    #: to understate the case. `None` when not applicable.
    lifetime_benefit: Optional[TreatmentBenefit] = None
    fh: Optional[FHResult] = None
    ldl_exposure: float = 0.0
    risk_score_reliable: bool = True
    rationale: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def nnt(self) -> float:
        return self.benefit.number_needed_to_treat if self.benefit else float("inf")

    def render(self) -> str:
        lines = [
            f"PREVENTION TRIAGE - {self.patient_id}",
            "=" * 58,
            f"10-year ASCVD risk : {self.risk_10y*100:.1f}%",
            f"Primary action     : {self.primary_action.value}",
            f"Recommended rung   : {self.recommended_rung.value}",
            f"LDL exposure       : {self.ldl_exposure:.1f} mmol/L-years",
        ]
        if self.fh is not None:
            lines.append(f"FH classification  : {self.fh.classification.value} (score {self.fh.score})")
        if self.benefit is not None:
            b = self.benefit
            lines += [
                "",
                f"Projected LDL drop : {b.ldl_reduction_mmol_l:.2f} mmol/L "
                f"({b.baseline_ldl_mg_dl:.0f} -> {b.ldl_on_treatment_mg_dl:.0f} mg/dL)",
                f"Absolute risk drop : {b.absolute_risk_reduction*100:.2f} percentage points",
                f"Number needed to treat (10 y): {b.number_needed_to_treat:.1f}",
                f"All-cause mortality benefit  : {b.mortality_benefit_10y*100:.2f} pp",
            ]
        if self.lifetime_benefit is not None:
            lb = self.lifetime_benefit
            lines += [
                "",
                f"Over {lb.years} further years (lifetime horizon):",
                f"  absolute risk drop : {lb.absolute_risk_reduction*100:.2f} pp",
                f"  number needed to treat: {lb.number_needed_to_treat:.1f}",
            ]
        if not self.risk_score_reliable:
            lines.append("  ! 10-year risk score unreliable for this patient (see rationale)")
        if self.actions:
            lines += ["", "Actions:"] + [f"  - {a.value}" for a in self.actions]
        if self.rationale:
            lines += ["", "Rationale:"] + [f"  - {r}" for r in self.rationale]
        if self.warnings:
            lines += ["", "Warnings:"] + [f"  ! {w}" for w in self.warnings]
        return "\n".join(lines)


def triage(
    profile: RiskFactorProfile,
    enhancers: Optional[RiskEnhancers] = None,
    equation: RiskEquation = RiskEquation.YADLOWSKY_2018,
    fh: Optional[FHResult] = None,
    established_ascvd: bool = False,
    on_statin: bool = False,
) -> TriageDecision:
    """Turn a risk assessment into an actionable, guideline-grounded decision.

    This is deliberately not an optimiser. It follows the published decision
    rules, and the interest is in what those rules imply when applied to a
    whole population (see PopulationPreventionCampaign).
    """
    enhancers = enhancers or RiskEnhancers()
    assessment = assess(profile, enhancers, equation=equation)
    risk = assessment.effective_risk_10y
    ldl = enhancers.ldl_mg_dl if enhancers.ldl_mg_dl is not None else profile.ldl_or_estimated()

    decision = TriageDecision(
        patient_id=profile.patient_id,
        risk_10y=risk,
        fh=fh,
        ldl_exposure=ldl_exposure_years(ldl, profile.age_years),
    )
    decision.rationale.extend(assessment.notes)

    if established_ascvd:
        decision.actions.append(PreventionAction.SECONDARY_PREVENTION)
        decision.recommended_rung = TherapyRung.HIGH_INTENSITY_STATIN
        decision.rationale.append(
            "Established ASCVD: secondary prevention, LDL goal <70 mg/dL, "
            "high-intensity statin plus ezetimibe if not at goal."
        )

    fh_definite = fh is not None and fh.classification in (
        FHClassification.DEFINITE, FHClassification.PROBABLE
    )

    # --- statin indication, in the order the guideline applies them --------
    diabetes_indication = (
        profile.diabetes and DIABETES_AGE_RANGE[0] <= profile.age_years <= DIABETES_AGE_RANGE[1]
    )
    ldl_indication = ldl >= LDL_THRESHOLD_VERY_HIGH

    if ldl_indication:
        decision.rationale.append(
            f"LDL-C {ldl:.0f} mg/dL >= {LDL_THRESHOLD_VERY_HIGH:.0f}: statin indication "
            "independent of calculated risk; evaluate for familial hypercholesterolaemia."
        )
    if diabetes_indication:
        decision.rationale.append(
            f"Diabetes at age {profile.age_years:.0f}: statin indication independent of "
            "calculated risk."
        )

    indicated = (
        established_ascvd or diabetes_indication or ldl_indication
        or fh_definite or risk >= 0.075
    )
    borderline = 0.05 <= risk < 0.075

    # --- is the 10-year score even the right quantity? --------------------
    # 10-year equations are dominated by age. A 42-year-old with an untreated
    # LDL of 265 mg/dL scores "low" on a 10-year horizon despite having
    # accumulated more LDL exposure than most 65-year-olds will ever have.
    # This is a documented limitation of every 10-year risk score, and it is
    # precisely why the guidelines use an *LDL threshold* as an independent
    # statin indication and why familial hypercholesterolaemia is treated from
    # diagnosis rather than from a risk score.
    high_cumulative_exposure = decision.ldl_exposure >= 100.0
    risk_score_reliable = not ((fh_definite or ldl_indication) and high_cumulative_exposure)
    decision.risk_score_reliable = risk_score_reliable
    if not risk_score_reliable:
        decision.rationale.append(
            f"10-year risk score ({risk*100:.1f}%) is not the right quantity here: "
            f"cumulative LDL exposure is {decision.ldl_exposure:.0f} mmol/L-years, which "
            "drives plaque burden over decades. The indication rests on the LDL "
            "level and genotype, not the score. Lifetime projection reported below."
        )

    # --- rung selection ---------------------------------------------------
    if established_ascvd or risk >= 0.20 or ldl_indication or fh_definite:
        rung = TherapyRung.HIGH_INTENSITY_STATIN
    elif risk >= 0.075:
        rung = TherapyRung.MODERATE_STATIN
    else:
        rung = TherapyRung.LIFESTYLE

    if on_statin and indicated:
        decision.actions.append(PreventionAction.INTENSIFY_TO_EZETIMIBE)
        rung = TherapyRung.STATIN_EZETIMIBE
        decision.rationale.append(
            "Already on a statin and still above goal: add ezetimibe (IMPROVE-IT "
            "established incremental benefit for the same mechanism)."
        )

    # --- the escalation ladder -------------------------------------------
    # A projected LDL that stays high after maximal oral therapy is what
    # justifies a PCSK9-pathway agent. The escalation is staged so the
    # recommended rung is always the next actionable step, not the last one an
    # algorithm could imagine.
    projected = max(0.0, ldl - routine_ldl_reduction(TherapyRung.STATIN_EZETIMIBE, ldl)
                    * MG_DL_PER_MMOL_L_CHOLESTEROL)
    pcsK9_candidate = indicated and projected >= 130.0 and (
        risk >= 0.20 or fh_definite or ldl_indication
    )

    if fh_definite or ldl_indication:
        decision.actions.append(PreventionAction.ONESHOT_THERAPY_CANDIDATE)
        decision.rationale.append(
            "Also a candidate for a one-shot, liver-directed LDL-lowering therapy. "
            "Clinical precedent: in-vivo base editing of PCSK9 (VERVE-102 class) "
            "produced durable PCSK9 and LDL reductions in early trials. This is an "
            "investigational option for after oral therapy is established, not an "
            "approved therapy and not the immediate step."
        )
        decision.warnings.append(
            "One-shot therapy means permanent editing of a slowly dividing "
            "hepatic target. Off-target editing risk and irreversibility require "
            "long-term follow-up and full IND-enabling toxicology."
        )

    if fh_definite:
        decision.actions.append(PreventionAction.CASCADE_SCREEN_FAMILY)
        decision.rationale.append(
            "Index case for familial hypercholesterolaemia. First-degree relatives "
            "share a 50% probability of the same defect and are overwhelmingly "
            "undiagnosed; cascade screening of ~2-3 relatives is the highest-yield "
            "action in this entire pipeline."
        )

    if borderline and not indicated:
        decision.actions.append(PreventionAction.MEASURE_ENHANCERS)
        decision.rationale.append(
            "Borderline risk: the guideline-endorsed next step is to measure Lp(a) "
            "and apoB and, if the decision remains uncertain, obtain a CAC score. "
            "This is what reclassifies the patient, not more arithmetic."
        )

    if not indicated and not borderline:
        decision.rationale.append(
            "Low calculated risk: no pharmacotherapy indicated. Reassess in 5 years "
            "or sooner if a risk factor changes."
        )

    # De-duplicate while preserving order.
    seen = set()
    unique_actions: List[PreventionAction] = []
    for a in decision.actions:
        if a not in seen:
            seen.add(a)
            unique_actions.append(a)
    decision.actions = unique_actions
    decision.recommended_rung = rung

    # The primary action is the *next step to take*, derived from the
    # recommended rung. Taking the first element of the action list would make
    # a long-horizon investigational option outrank the statin the patient
    # should start this week.
    if indicated:
        decision.primary_action = _RUNG_TO_ACTION[rung]
    elif borderline:
        decision.primary_action = PreventionAction.MEASURE_ENHANCERS
    else:
        decision.primary_action = PreventionAction.LIFESTYLE_ONLY
    if decision.primary_action not in decision.actions:
        decision.actions.insert(0, decision.primary_action)

    if indicated:
        decision.benefit = project_benefit(
            baseline_risk_10y=risk,
            baseline_ldl_mg_dl=ldl,
            rung=rung,
            years=10,
            age_years=profile.age_years,
        )
        # Lifetime horizon where the 10-year window misrepresents the case.
        remaining = int(max(0, min(45, 78 - profile.age_years)))
        if not risk_score_reliable and remaining >= 15:
            decision.lifetime_benefit = project_benefit(
                baseline_risk_10y=risk,
                baseline_ldl_mg_dl=ldl,
                rung=rung,
                years=remaining,
                age_years=profile.age_years,
            )
    else:
        decision.warnings.append(
            "No treatment indicated, so no benefit projection is reported. "
            "(A projection would overstate benefit by assuming a therapy the "
            "guideline does not recommend.)"
        )

    if pcsK9_candidate:
        if decision.primary_action in (
            PreventionAction.MODERATE_STATIN, PreventionAction.HIGH_INTENSITY_STATIN
        ) or on_statin:
            decision.rationale.append(
                f"Projected LDL {projected:.0f} mg/dL remains high after maximal oral "
                "therapy: PCSK9 inhibitor or inclisiran is the evidence-based step "
                "once statin and ezetimibe are in place."
            )
            if PreventionAction.PCSK9_PATHWAY not in decision.actions:
                decision.actions.append(PreventionAction.PCSK9_PATHWAY)

    if assessment.outside_validated_domain:
        decision.warnings.append(
            "Risk estimate is extrapolation outside the validated model domain."
        )
    if assessment.unmeasured_enhancers:
        decision.warnings.append(
            "Unmeasured risk enhancers: " + ", ".join(assessment.unmeasured_enhancers)
            + ". Absence of a measurement is not absence of risk."
        )

    return decision


# ---------------------------------------------------------------------------
# Population campaign: where do the lives actually come from?
# ---------------------------------------------------------------------------

@dataclass
class PopulationProfile:
    """A synthetic population model. Tagged ASSUMED throughout -- this is a
    model of a population, not an observation of one. Feed a real table via
    :meth:`PopulationPreventionCampaign.from_table` for a real answer."""

    n_patients: int = 200_000
    seed: int = 20260919
    age_range: Tuple[int, int] = (40, 79)
    total_chol_mean_sd: Tuple[float, float] = (196.0, 40.0)
    hdl_mean_sd_male: Tuple[float, float] = (47.0, 13.0)
    hdl_mean_sd_female: Tuple[float, float] = (58.0, 15.0)
    sbp_mean_sd: Tuple[float, float] = (127.0, 19.0)
    bp_medication_fraction_by_decade: Tuple[float, ...] = (0.10, 0.22, 0.38, 0.55)
    smoker_fraction: float = 0.15
    diabetes_fraction: float = 0.13
    fraction_black: float = 0.13
    fraction_female: float = 0.51
    #: Probability that an LDL-driven one-shot candidate actually has a
    #: monogenic cause. FH prevalence is ~1/250 in the general population, and
    #: the severe-LDL tail is enriched relative to that.
    fh_prevalence_in_severe_ldl: float = 0.20


#: Column aliases accepted by :meth:`PopulationPreventionCampaign.from_table`.
#: The first name is canonical; the rest are convenience aliases so a real
#: cohort export usually works without renaming anything.
COHORT_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "age": ("age", "age_years", "ageyrs"),
    "female": ("female", "is_female"),
    "sex": ("sex", "gender"),
    "black": ("black", "is_black", "african_american"),
    "race": ("race", "race_stratum", "ethnicity"),
    "tc": ("tc", "chol_total_mgdl", "total_cholesterol", "cholesterol"),
    "hdl": ("hdl", "chol_hdl_mgdl", "hdl_cholesterol"),
    "sbp": ("sbp", "bp_sys_mmhg", "systolic_bp", "systolic"),
    "bp_meds": ("bp_meds", "on_bp_medication", "antihypertensive"),
    "smoker": ("smoker", "current_smoker", "smoking"),
    "diabetes": ("diabetes", "diabetic", "dm"),
    "ldl": ("ldl", "ldl_mg_dl", "ldl_mgdl", "ldl_cholesterol"),
}

_REQUIRED_COHORT_COLUMNS: Tuple[str, ...] = (
    "age", "tc", "hdl", "sbp", "bp_meds", "smoker", "diabetes", "ldl",
)


def _as_bool_vector(values: Sequence, true_tokens: Tuple[str, ...] = ("y", "yes", "true", "1")) -> np.ndarray:
    out = np.zeros(len(values), dtype=bool)
    tokens = set(true_tokens)
    for i, v in enumerate(values):
        if isinstance(v, str):
            out[i] = v.strip().lower() in tokens
        elif isinstance(v, (bool, np.bool_)):
            out[i] = bool(v)
        else:
            out[i] = bool(v) and not (isinstance(v, float) and math.isnan(v))
    return out


def _normalise_cohort_table(table) -> Dict[str, np.ndarray]:
    """Coerce a real cohort into the internal column layout.

    Accepts a dict of columns, a pandas DataFrame, or an iterable of row
    dicts. Column matching is case-insensitive and alias-aware, because a
    cohort export is never named the way an API wants it.
    """
    if table is None:
        raise ValueError("no cohort table supplied")

    if hasattr(table, "to_dict"):
        try:
            raw = table.to_dict(orient="list")
        except TypeError:  # pragma: no cover - non-pandas mapping-like object
            raw = {k: list(v) for k, v in dict(table).items()}
    elif isinstance(table, dict):
        raw = {k: list(v) for k, v in table.items()}
    else:
        rows = list(table)
        if not rows:
            raise ValueError("cohort table is empty")
        if not isinstance(rows[0], dict):
            raise TypeError(
                "cohort table must be a dict of columns, a pandas DataFrame, "
                "or an iterable of row dicts"
            )
        keys = set(rows[0])
        raw = {k: [r.get(k) for r in rows] for k in keys}

    lower = {str(k).strip().lower(): v for k, v in raw.items()}

    resolved: Dict[str, Sequence] = {}
    for canonical, aliases in COHORT_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower:
                resolved[canonical] = lower[alias]
                break

    n = max((len(v) for v in resolved.values()), default=0)
    if n == 0:
        raise ValueError("cohort table has no usable columns")

    for canonical in _REQUIRED_COHORT_COLUMNS:
        if canonical not in resolved:
            raise ValueError(
                f"cohort table is missing required column '{canonical}'; "
                f"accepted aliases: {', '.join(COHORT_COLUMN_ALIASES[canonical])}"
            )

    # Sex: either a boolean `female` column or a `sex` string column.
    if "female" in resolved:
        female = _as_bool_vector(resolved["female"])
    elif "sex" in resolved:
        female = np.array(
            [str(v).strip().lower().startswith("f") for v in resolved["sex"]], dtype=bool
        )
    else:
        raise ValueError(
            "cohort table needs either a 'female' column or a 'sex' column"
        )

    # Race stratum: either a boolean `black` column or a `race` string column.
    if "black" in resolved:
        black = _as_bool_vector(resolved["black"])
    elif "race" in resolved:
        black = np.array(
            [str(v).strip().lower() in ("black", "african_american", "african american", "aa")
             for v in resolved["race"]],
            dtype=bool,
        )
    else:
        # The published models have no third stratum, so anything unstated is
        # treated as the White/reference equation and warned about.
        black = np.zeros(n, dtype=bool)

    out: Dict[str, np.ndarray] = {
        "age": np.asarray(resolved["age"], dtype=float),
        "tc": np.asarray(resolved["tc"], dtype=float),
        "hdl": np.asarray(resolved["hdl"], dtype=float),
        "sbp": np.asarray(resolved["sbp"], dtype=float),
        "ldl": np.asarray(resolved["ldl"], dtype=float),
        "female": female,
        "black": black,
        "bp_meds": _as_bool_vector(resolved["bp_meds"]),
        "smoker": _as_bool_vector(resolved["smoker"]),
        "diabetes": _as_bool_vector(resolved["diabetes"]),
    }

    lengths = {k: len(v) for k, v in out.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"cohort columns have mismatched lengths: {lengths}")

    finite = np.ones(n, dtype=bool)
    for key in ("age", "tc", "hdl", "sbp", "ldl"):
        finite &= np.isfinite(out[key])
    if not finite.all():
        dropped = int((~finite).sum())
        out = {k: v[finite] for k, v in out.items()}
        if out["age"].size == 0:
            raise ValueError("cohort table has no rows with complete numeric data")
        # Recorded rather than silently swallowed: dropping rows changes the
        # denominator of every rate the campaign reports.
        out["_dropped_rows"] = np.array([dropped])

    return out


@dataclass
class PopulationReport:
    n_patients: int
    equation: RiskEquation
    #: Fraction of statin-eligible patients actually receiving a statin before
    #: the programme. This is the single most important input in the model.
    baseline_statin_coverage: float

    statin_eligible: int
    eligible_before: int
    newly_identified: int
    treatment_gap_count: int
    mean_risk_10y: float
    mean_risk_eligible: float

    # --- the payoff, over 10 years ---------------------------------------
    events_prevented_per_1000: float
    events_prevented: float
    events_prevented_ci95: Tuple[float, float]
    deaths_prevented: float
    deaths_prevented_ci95: Tuple[float, float]
    mean_nnt: float
    mean_nnt_ci95: Tuple[float, float]
    mean_delta_ldl_mmol_l: float

    # --- detection gaps for the enhancers --------------------------------
    ldl_ge_190_count: int
    lp_a_unmeasured_fraction: float
    borderline_needing_enhancer_measurement: int
    #: Rows dropped from a real cohort table for missing/invalid numeric data.
    #: Reported rather than swallowed, because dropping rows changes every
    #: rate in this report.
    dropped_rows: int = 0
    #: True when the numbers came from a supplied cohort rather than the
    #: synthetic population model.
    from_real_cohort: bool = False


class PopulationPreventionCampaign:
    """Deploy the triage engine across a population and count the lives.

    The point of this class is that it converts "there is a treatment gap"
    from an assertion into a number with a confidence interval. The gap is
    real and large: in US NHANES analyses a substantial minority of
    guideline-eligible adults receive no lipid-lowering therapy at all.
    """

    def __init__(
        self,
        model: Optional[PopulationProfile] = None,
        equation: RiskEquation = RiskEquation.YADLOWSKY_2018,
        baseline_statin_coverage: float = 0.55,
        cycles: int = 1,
        table: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        self.model = model or PopulationProfile()
        self.equation = equation
        self.baseline_statin_coverage = baseline_statin_coverage
        self.cycles = cycles
        self.table = table

    @classmethod
    def from_table(
        cls,
        table,
        equation: RiskEquation = RiskEquation.YADLOWSKY_2018,
        baseline_statin_coverage: float = 0.55,
    ) -> "PopulationPreventionCampaign":
        """Run the campaign on a REAL cohort instead of the synthetic one.

        This is the difference between a simulation and a decision-support
        tool. The risk equations and the treatment effect are already
        published; the only modelled quantity left is who is in the cohort,
        and this constructor removes that assumption.

        ``table`` may be a dict of columns, a pandas DataFrame, or an iterable
        of row dicts. Required columns (aliases accepted, case-insensitive):

        ``age``, ``tc``/``chol_total_mgdl``, ``hdl``, ``sbp``, ``bp_meds``,
        ``smoker``, ``diabetes``, ``ldl``, plus either ``female`` or ``sex``,
        and optionally ``black`` or ``race``.

        Rows with non-finite values in any numeric column are dropped and the
        count is reported as ``dropped_rows`` on the result, because silently
        dropping rows would change the denominator of every rate reported.
        """
        return cls(
            equation=equation,
            baseline_statin_coverage=baseline_statin_coverage,
            table=_normalise_cohort_table(table),
        )

    # -- synthetic population ---------------------------------------------
    def _generate(self, n: Optional[int] = None, seed: Optional[int] = None) -> Dict[str, np.ndarray]:
        if self.table is not None:
            # A real cohort is what it is; neither n nor seed applies. The
            # bootstrap over patients still provides sampling uncertainty.
            return dict(self.table)

        m = self.model
        n = int(n if n is not None else m.n_patients)
        rng = np.random.default_rng(m.seed if seed is None else seed)

        age = rng.integers(m.age_range[0], m.age_range[1] + 1, size=n).astype(float)
        female = rng.random(n) < m.fraction_female
        black = rng.random(n) < m.fraction_black

        tc = rng.normal(*m.total_chol_mean_sd, size=n).clip(130.0, 320.0)
        hdl_m = rng.normal(*m.hdl_mean_sd_male, size=n)
        hdl_f = rng.normal(*m.hdl_mean_sd_female, size=n)
        hdl = np.where(female, hdl_f, hdl_m).clip(20.0, 100.0)
        sbp = rng.normal(*m.sbp_mean_sd, size=n).clip(90.0, 200.0)

        # BP-medication prevalence rises steeply with age.
        decade_index = ((age - m.age_range[0]) // 10).astype(int).clip(0, 3)
        bp_frac = np.asarray(m.bp_medication_fraction_by_decade)[decade_index]
        bp_meds = rng.random(n) < bp_frac

        smoker = rng.random(n) < m.smoker_fraction
        diabetes = rng.random(n) < m.diabetes_fraction

        # Estimated LDL from the Friedewald relation with a population-median
        # triglyceride term of 25 mg/dL.
        ldl = np.clip(tc - hdl - 25.0, 30.0, 330.0)

        return {
            "age": age, "female": female, "black": black, "tc": tc, "hdl": hdl,
            "sbp": sbp, "bp_meds": bp_meds, "smoker": smoker, "diabetes": diabetes,
            "ldl": ldl,
        }

    # -- vectorised risk evaluation ---------------------------------------
    def _risk_vector(self, pop: Dict[str, np.ndarray]) -> np.ndarray:
        """Risk for the whole population at once.

        The batch path duplicates the scalar equations rather than reusing them
        because the arithmetic is the same but the payoff is ~10^5 x fewer
        interpreter round-trips. A test asserts the two agree to 1e-9.
        """
        if self.equation is RiskEquation.GOFF_2013:
            return self._goff_vector(pop)
        if self.equation is RiskEquation.YADLOWSKY_2018:
            return self._yadlowsky_vector(pop)
        raise ValueError(f"unknown equation {self.equation!r}")

    def _goff_vector(self, pop: Dict[str, np.ndarray]) -> np.ndarray:
        age, tc, hdl = pop["age"], pop["tc"], pop["hdl"]
        sbp, bp_meds = pop["sbp"], pop["bp_meds"]
        smoker = pop["smoker"].astype(float)
        diabetes = pop["diabetes"].astype(float)
        treated = bp_meds.astype(float)
        untreated = 1.0 - treated

        ln_age = np.log(age)
        ln_tc = np.log(tc)
        ln_hdl = np.log(hdl)
        ln_sbp = np.log(sbp)

        out = np.empty_like(age)
        for female in (True, False):
            for black in (True, False):
                sex = "female" if female else "male"
                race = "black" if black else "white"
                mask = (pop["female"] == female) & (pop["black"] == black)
                if not mask.any():
                    continue
                c = _PCR_GOFF_2013[(sex, race)]
                ind = (
                    c["ln_age"] * ln_age[mask]
                    + c["ln_age_squared"] * ln_age[mask] ** 2
                    + c["ln_chol_total"] * ln_tc[mask]
                    + c["ln_age_x_ln_chol_total"] * ln_age[mask] * ln_tc[mask]
                    + c["ln_chol_hdl"] * ln_hdl[mask]
                    + c["ln_age_x_ln_chol_hdl"] * ln_age[mask] * ln_hdl[mask]
                    + c["ln_treated_sbp"] * ln_sbp[mask] * treated[mask]
                    + c["ln_untreated_sbp"] * ln_sbp[mask] * untreated[mask]
                    + c["ln_age_x_ln_treated_sbp"] * ln_age[mask] * ln_sbp[mask] * treated[mask]
                    + c["ln_age_x_ln_untreated_sbp"] * ln_age[mask] * ln_sbp[mask] * untreated[mask]
                    + c["smoker"] * smoker[mask]
                    + c["ln_age_x_smoker"] * ln_age[mask] * smoker[mask]
                    + c["diabetes"] * diabetes[mask]
                )
                out[mask] = 1.0 - c["base_survival_10y"] ** np.exp(ind - c["coef_mean"])
        return out

    def _yadlowsky_vector(self, pop: Dict[str, np.ndarray]) -> np.ndarray:
        age, tc, hdl, sbp = pop["age"], pop["tc"], pop["hdl"], pop["sbp"]
        black = pop["black"].astype(float)
        bp_meds = pop["bp_meds"].astype(float)
        diabetes = pop["diabetes"].astype(float)
        smoke = pop["smoker"].astype(float)
        ratio = tc / hdl

        out = np.empty_like(age)
        for female in (True, False):
            mask = pop["female"] == female
            if not mask.any():
                continue
            c = _PCR_YADLOWSKY_2018["female" if female else "male"]
            b, a, s = black[mask], age[mask], sbp[mask]
            m, d, k = bp_meds[mask], diabetes[mask], smoke[mask]
            r = ratio[mask]
            ind = (
                c["intercept"] + c["age"] * a + c["black"] * b
                + c["sbp_squared"] * s * s + c["sbp"] * s + c["bp_meds"] * m
                + c["diabetes"] * d + c["smoke_current"] * k + c["chol_ratio"] * r
                + c["age_x_black"] * a * b + c["sbp_x_bp_meds"] * s * m
                + c["sbp_x_black"] * s * b + c["bp_meds_x_black"] * m * b
                + c["age_x_sbp"] * a * s + c["black_x_diabetes"] * b * d
                + c["black_x_smoke_current"] * b * k + c["black_x_chol_ratio"] * b * r
                + c["black_x_sbp_x_bp_meds"] * b * s * m
                + c["black_x_sbp_x_age"] * b * s * a
            )
            out[mask] = 1.0 / (1.0 + np.exp(-ind))
        return out

    # -- eligibility and benefit ------------------------------------------
    @staticmethod
    def _statin_indicated(pop: Dict[str, np.ndarray], risk: np.ndarray) -> np.ndarray:
        diabetes_indication = pop["diabetes"] & (pop["age"] >= DIABETES_AGE_RANGE[0]) & (
            pop["age"] <= DIABETES_AGE_RANGE[1]
        )
        return (risk >= 0.075) | diabetes_indication | (pop["ldl"] >= LDL_THRESHOLD_VERY_HIGH)

    @staticmethod
    def _event_reduction(risk: np.ndarray, ldl: np.ndarray, rung: TherapyRung) -> np.ndarray:
        """Vectorised ARR. Same structure as project_benefit, in numpy."""
        fraction = _RUNG_FRACTION[rung]
        baseline_mmol = ldl / MG_DL_PER_MMOL_L_CHOLESTEROL
        delta = np.maximum(0.0, np.minimum(baseline_mmol * fraction, baseline_mmol - 0.5))
        annual = 1.0 - (1.0 - np.clip(risk, 0.0, 1.0 - 1e-12)) ** 0.1
        hazard = CTT_MVE_RR_PER_MMOL_L ** delta
        untreated = 1.0 - (1.0 - annual) ** 10.0
        treated = 1.0 - (1.0 - annual * hazard) ** 10.0
        return np.maximum(0.0, untreated - treated)

    @staticmethod
    def _death_reduction(ldl: np.ndarray, age: np.ndarray, rung: np.ndarray) -> np.ndarray:
        """Mortality benefit using the SAME per-patient rung as the event
        calculation. Using a fixed high-intensity assumption for everyone here
        would quietly credit the cohort with a stronger regimen than the one
        the event model just priced."""
        fractions = np.where(
            rung == TherapyRung.HIGH_INTENSITY_STATIN,
            _RUNG_FRACTION[TherapyRung.HIGH_INTENSITY_STATIN],
            _RUNG_FRACTION[TherapyRung.MODERATE_STATIN],
        )
        baseline_mmol = ldl / MG_DL_PER_MMOL_L_CHOLESTEROL
        delta = np.maximum(0.0, np.minimum(baseline_mmol * fractions, baseline_mmol - 0.5))
        baseline_mortality = _mortality_risk_vector(age)
        annual = 1.0 - (1.0 - baseline_mortality) ** 0.1
        treated = 1.0 - (1.0 - annual * CTT_MORTALITY_RR_PER_MMOL_L ** delta) ** 10.0
        return np.maximum(0.0, baseline_mortality - treated)

    # -- the campaign ------------------------------------------------------
    def run(
        self,
        n_patients: Optional[int] = None,
        bootstrap_resamples: int = 200,
        seed: Optional[int] = None,
    ) -> PopulationReport:
        """Screen the population and count events and deaths prevented over
        10 years.

        Uncertainty is reported along two independent axes and the total
        interval is the union of them, not the sum:

        * **sampling** -- bootstrap resampling of the population, because the
          answer depends on who happened to be drawn.
        * **effect size** -- the CTT confidence interval on the hazard ratio
          (0.76-0.80), which is the dominant uncertainty and the one a
          reviewer will ask about first.
        """
        pop = self._generate(n=n_patients, seed=seed)
        n = int(pop["age"].size)
        risk = self._risk_vector(pop)
        indicated = self._statin_indicated(pop, risk)

        rung = np.where(risk >= 0.20, TherapyRung.HIGH_INTENSITY_STATIN, TherapyRung.MODERATE_STATIN)

        # Benefit is computed per patient at their own rung, then the untreated
        # remainder is scaled by how much of the eligible population is
        # actually already covered.
        arr_high = self._event_reduction(risk, pop["ldl"], TherapyRung.HIGH_INTENSITY_STATIN)
        arr_mod = self._event_reduction(risk, pop["ldl"], TherapyRung.MODERATE_STATIN)
        arr = np.where(rung == TherapyRung.HIGH_INTENSITY_STATIN, arr_high, arr_mod)

        deaths = self._death_reduction(pop["ldl"], pop["age"], rung)

        # The gap: eligible but not covered. Closing it is the entire point.
        eligible_arr = arr[indicated]
        eligible_deaths = deaths[indicated]
        gap_fraction = 1.0 - self.baseline_statin_coverage

        events_prevented = float(eligible_arr.sum() * gap_fraction)
        deaths_prevented = float(eligible_deaths.sum() * gap_fraction)
        n_eligible = int(indicated.sum())
        per_1000 = (events_prevented / max(n, 1)) * 1000.0

        # --- CTT effect-size uncertainty ---------------------------------
        def scale_by_rr(rr: float) -> Tuple[float, float]:
            """Re-price the whole eligible cohort at a different event hazard
            ratio. The mortality hazard ratio is shifted by the same *relative*
            change, so moving both ends of the CTT event interval moves the
            mortality estimate with it rather than holding it fixed."""
            arr_alt = self._event_reduction_rr(risk, pop["ldl"], rung, rr)
            ev = float(arr_alt[indicated].sum() * gap_fraction)

            fractions = np.where(
                rung == TherapyRung.HIGH_INTENSITY_STATIN,
                _RUNG_FRACTION[TherapyRung.HIGH_INTENSITY_STATIN],
                _RUNG_FRACTION[TherapyRung.MODERATE_STATIN],
            )
            bmmol = pop["ldl"] / MG_DL_PER_MMOL_L_CHOLESTEROL
            delta = np.maximum(0.0, np.minimum(bmmol * fractions, bmmol - 0.5))
            mort_rr = 1.0 - (1.0 - CTT_MORTALITY_RR_PER_MMOL_L) * (
                (1.0 - rr) / (1.0 - CTT_MVE_RR_PER_MMOL_L)
            )
            bm = _mortality_risk_vector(pop["age"])
            annual = 1.0 - (1.0 - bm) ** 0.1
            tr = 1.0 - (1.0 - annual * mort_rr ** delta) ** 10.0
            dp = float(np.maximum(0.0, bm - tr)[indicated].sum() * gap_fraction)
            return ev, dp

        ev_lo, dp_lo = scale_by_rr(CTT_MVE_RR_CI95[1])   # weaker effect -> higher RR
        ev_hi, dp_hi = scale_by_rr(CTT_MVE_RR_CI95[0])   # stronger effect -> lower RR

        # --- sampling uncertainty ----------------------------------------
        # Resample over the ELIGIBLE subpopulation, not the whole population:
        # the quantity of interest is the mean benefit per eligible patient, and
        # resampling the full array with full-length indices would both index
        # out of bounds and inflate the interval with patients who contribute
        # nothing.
        if bootstrap_resamples > 0 and n_eligible > 1:
            rng = np.random.default_rng((seed or self.model.seed) + 1)
            ev_samples = np.empty(bootstrap_resamples)
            dp_samples = np.empty(bootstrap_resamples)
            for i in range(bootstrap_resamples):
                idx = rng.integers(0, n_eligible, size=n_eligible)
                ev_samples[i] = eligible_arr[idx].sum() * gap_fraction
                dp_samples[i] = eligible_deaths[idx].sum() * gap_fraction
            ev_sample_lo, ev_sample_hi = np.percentile(ev_samples, [2.5, 97.5])
            dp_sample_lo, dp_sample_hi = np.percentile(dp_samples, [2.5, 97.5])
        else:
            ev_sample_lo = ev_sample_hi = events_prevented
            dp_sample_lo = dp_sample_hi = deaths_prevented

        # Union of the two intervals (widest plausible bound on both sides).
        ev_ci = (min(ev_lo, float(ev_sample_lo)), max(ev_hi, float(ev_sample_hi)))
        dp_ci = (min(dp_lo, float(dp_sample_lo)), max(dp_hi, float(dp_sample_hi)))

        # NNT is the reciprocal of the *mean* absolute risk reduction, not the
        # mean of reciprocals: averaging 1/ARR per patient is dominated by the
        # few smallest effects and does not correspond to any treatable cohort.
        mean_arr = float(eligible_arr.mean()) if n_eligible else 0.0
        eligible_mean_nnt = float("inf") if mean_arr <= 0 else 1.0 / mean_arr
        # Invert the CTT interval for NNT, remembering that a smaller effect
        # (higher RR) means a larger NNT.
        if mean_arr > 0:
            # Index the eligible subset. Averaging the CTT interval over the
            # whole population would dilute it with low-risk patients who
            # contribute no benefit and push the interval off the point
            # estimate entirely.
            arr_lo = float(self._event_reduction_rr(
                risk, pop["ldl"], rung, CTT_MVE_RR_CI95[1])[indicated].mean())
            arr_hi = float(self._event_reduction_rr(
                risk, pop["ldl"], rung, CTT_MVE_RR_CI95[0])[indicated].mean())
            nnt_ci = (
                float("inf") if arr_hi <= 0 else 1.0 / arr_hi,
                float("inf") if arr_lo <= 0 else 1.0 / arr_lo,
            )
        else:
            nnt_ci = (float("inf"), float("inf"))

        # Mean achievable LDL reduction among the eligible, using the same
        # floor logic as routine_ldl_reduction so the reported number matches
        # the one used in the benefit calculation.
        if n_eligible:
            bmmol = pop["ldl"][indicated] / MG_DL_PER_MMOL_L_CHOLESTEROL
            frac_hi = _RUNG_FRACTION[TherapyRung.HIGH_INTENSITY_STATIN]
            delta_ldl = float(np.mean(np.maximum(
                0.0, np.minimum(bmmol * frac_hi, bmmol - 0.5))))
        else:
            delta_ldl = 0.0

        already_covered = int(round(n_eligible * self.baseline_statin_coverage))

        return PopulationReport(
            n_patients=n,
            equation=self.equation,
            baseline_statin_coverage=self.baseline_statin_coverage,
            statin_eligible=n_eligible,
            eligible_before=already_covered,
            newly_identified=n_eligible - already_covered,
            treatment_gap_count=int(round(n_eligible * gap_fraction)),
            mean_risk_10y=float(risk.mean()),
            mean_risk_eligible=float(risk[indicated].mean()) if n_eligible else 0.0,
            events_prevented_per_1000=per_1000,
            events_prevented=events_prevented,
            events_prevented_ci95=(float(ev_ci[0]), float(ev_ci[1])),
            deaths_prevented=deaths_prevented,
            deaths_prevented_ci95=(float(dp_ci[0]), float(dp_ci[1])),
            mean_nnt=eligible_mean_nnt,
            mean_nnt_ci95=nnt_ci,
            mean_delta_ldl_mmol_l=delta_ldl,
            ldl_ge_190_count=int((pop["ldl"] >= LDL_THRESHOLD_VERY_HIGH).sum()),
            lp_a_unmeasured_fraction=1.0,
            borderline_needing_enhancer_measurement=int(
                ((risk >= 0.05) & (risk < 0.075)).sum()
            ),
            dropped_rows=int(pop["_dropped_rows"][0]) if "_dropped_rows" in pop else 0,
            from_real_cohort=self.table is not None,
        )

    def _event_reduction_rr(
        self, risk: np.ndarray, ldl: np.ndarray, rung: np.ndarray, rr: float
    ) -> np.ndarray:
        baseline_mmol = ldl / MG_DL_PER_MMOL_L_CHOLESTEROL
        fractions = np.where(
            rung == TherapyRung.HIGH_INTENSITY_STATIN,
            _RUNG_FRACTION[TherapyRung.HIGH_INTENSITY_STATIN],
            _RUNG_FRACTION[TherapyRung.MODERATE_STATIN],
        )
        delta = np.maximum(0.0, np.minimum(baseline_mmol * fractions, baseline_mmol - 0.5))
        annual = 1.0 - (1.0 - np.clip(risk, 0.0, 1.0 - 1e-12)) ** 0.1
        treated = 1.0 - (1.0 - annual * rr ** delta) ** 10.0
        return np.maximum(0.0, (1.0 - (1.0 - annual) ** 10.0) - treated)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def prevention_provenance() -> Dict[str, Provenance]:
    """Provenance of every load-bearing number in this module.

    The split is unusually favourable here compared with the rest of Danon:
    the risk equations and the treatment effect are *published* and
    *externally validated*, so the honest answer to "where did this come
    from" is a citation rather than an assumption.
    """
    return {
        # --- published equations and effect sizes ------------------------
        "pooled_cohort_goff_2013_coefficients": Provenance.LITERATURE,
        "pooled_cohort_yadlowsky_2018_coefficients": Provenance.LITERATURE,
        "pcr_domain_bounds": Provenance.LITERATURE,
        "ctt_mve_rr_per_mmol_l": Provenance.LITERATURE,
        "ctt_mortality_rr_per_mmol_l": Provenance.LITERATURE,
        "risk_category_thresholds": Provenance.LITERATURE,
        "statin_indication_risk_threshold": Provenance.LITERATURE,
        "ldl_190_indication_threshold": Provenance.LITERATURE,
        "dutch_lipid_clinic_point_system": Provenance.LITERATURE,
        "lp_a_thresholds": Provenance.LITERATURE,
        "mg_dl_per_mmol_l_cholesterol": Provenance.LITERATURE,
        # --- derived -----------------------------------------------------
        "ldl_response_by_rung": Provenance.DERIVED,
        "annual_rate_conversion": Provenance.DERIVED,
        "number_needed_to_treat": Provenance.DERIVED,
        "ldl_exposure_years": Provenance.DERIVED,
        # --- assumed / calibration targets -------------------------------
        "enhancer_uplift_factors": Provenance.ASSUMED,
        "ten_year_all_cause_mortality": Provenance.ASSUMED,
        "baseline_statin_coverage": Provenance.ASSUMED,
        "population_risk_factor_distributions": Provenance.ASSUMED,
        "fh_prevalence_in_severe_ldl": Provenance.ASSUMED,
        "friedewald_triglyceride_term": Provenance.ASSUMED,
    }


def prevention_constant_audit() -> List[Constant]:
    """Same shape as :func:`danon.physics.audit_constants`, for this module."""
    return [
        Constant("ctt_mve_rr_per_mmol_l", CTT_MVE_RR_PER_MMOL_L, "ratio",
                 Provenance.LITERATURE,
                 "CTT 2010; 26 trials, 170000 participants; CI 0.76-0.80"),
        Constant("ctt_mortality_rr_per_mmol_l", CTT_MORTALITY_RR_PER_MMOL_L, "ratio",
                 Provenance.LITERATURE, "CTT 2012; CI 0.87-0.93"),
        Constant("ldl_threshold_very_high", LDL_THRESHOLD_VERY_HIGH, "mg/dL",
                 Provenance.LITERATURE, "ACC/AHA: statin indication independent of risk"),
        Constant("lp_a_high_mg_dl", LPA_HIGH_MG_DL, "mg/dL",
                 Provenance.LITERATURE, "EAS/ESC risk-enhancing threshold"),
        Constant("baseline_statin_coverage", 0.55, "fraction",
                 Provenance.ASSUMED,
                 "fraction of eligible adults on therapy; population-specific, "
                 "must be replaced with local data"),
        Constant("statins_high_intensity_fraction", _STATIN_HIGH_FRACTION, "fraction",
                 Provenance.ASSUMED, "representative midpoint of published LDL response"),
        Constant("base_edit_fraction", _BASE_EDIT_FRACTION, "fraction",
                 Provenance.ASSUMED,
                 "one-shot hepatic PCSK9 knockdown; from early base-editing trials, "
                 "not a validated therapy"),
    ]


def print_prevention_disclaimer() -> None:
    print(
        "=" * 72 + "\n"
        "PREVENTION ENGINE - COMPUTATIONAL TRIAGE MODEL, NOT MEDICAL ADVICE\n"
        "The risk equations and treatment effects are published and externally\n"
        "validated; the population model and enhancer weightings are not.\n"
        "No output here may be used to make a decision about a real person.\n"
        "Risk scores triage who deserves a blood test and a conversation with a\n"
        "clinician. They do not diagnose, image, or treat anyone.\n"
        + "=" * 72
    )
