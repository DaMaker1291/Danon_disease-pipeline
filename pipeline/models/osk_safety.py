import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Aligned with Life Biosciences ER-100 clinical trial (first patient dosed June 9, 2026)
# ER-100 uses an AAV2 vector with a systemic doxycycline-controlled switch.
# The vector persists long-term; the doxycycline activation trigger is tightly
# controlled to an 8-week (56-day) course to prevent continuous overexpression.
OSK_FACTORS = {
    "Oct4": {
        "gene_id": "POU5F1",
        "function": "Master pluripotency transcription factor",
        "cancer_risk_if_prolonged": "high",
        "safe_expression_window_days": (14, 56),
        "lethality_if_constitutive": "teratoma formation",
    },
    "Sox2": {
        "gene_id": "SOX2",
        "function": "Neural stem cell maintenance + pluripotency",
        "cancer_risk_if_prolonged": "high",
        "safe_expression_window_days": (14, 56),
        "lethality_if_constitutive": "squamous cell carcinoma risk",
    },
    "Klf4": {
        "gene_id": "KLF4",
        "function": "Pluripotency induction + tumor suppression",
        "cancer_risk_if_prolonged": "moderate",
        "safe_expression_window_days": (14, 56),
        "lethality_if_constitutive": "context-dependent (oncogene in some tissues)",
    },
}

# ER-100 clinical parameters: doxycycline switch active for 56 days
# Vector is AAV2 — persists indefinitely, but expression only when doxycycline is present
DOXYCYCLINE_THRESHOLDS = {
    "max_activation_days": 56,
    "min_activation_days": 14,
    "optimal_activation_days": 42,
    "doxycycline_half_life_hours": 18,
    "max_expression_level": 1.0,
    "target_expression_level": 0.5,
    "vector_persistence_decay_per_year": 0.05,
    "rejuvenation_efficiency_at_42d": 0.85,
    "cancer_risk_at_56d": 0.08,
    "cancer_risk_at_70d": 0.25,
    "cancer_risk_at_90d": 0.55,
    "cancer_risk_at_180d": 0.85,
}


@dataclass
class OSKDoxycyclineProfile:
    days_on_dox: float
    oct4_level: float
    sox2_level: float
    klf4_level: float
    total_expression: float
    rejuvenation_score: float
    cancer_risk: float
    is_safe: bool
    within_dox_window: bool
    vector_present: bool


class OSKDoxycyclineSwitch:
    """Models the Life Biosciences ER-100 AAV2 + doxycycline-controlled OSK system.

    The AAV2 vector integrates and persists long-term (>95% after 1 year).
    OSK expression is only active while the patient takes oral doxycycline
    (systemic activation switch). The clinical protocol specifies an 8-week
    (56-day) doxycycline course to limit oncogenic risk from continuous OSK.
    """

    def __init__(self, max_days: float = 56, target_days: float = 42):
        self.max_days = max_days
        self.target_days = target_days
        self.thresholds = DOXYCYCLINE_THRESHOLDS

    def simulate_expression(self, expression_level: float, days_on_dox: float,
                            years_since_delivery: float = 0.0) -> OSKDoxycyclineProfile:
        peak = min(expression_level, self.thresholds["max_expression_level"])

        # AAV2 vector persistence: ~95% after 1 year, slow linear decay
        vector_titer = 1.0 - self.thresholds["vector_persistence_decay_per_year"] * years_since_delivery
        vector_present = vector_titer > 0.5

        # Expression is proportional to doxycycline presence (switch on/off)
        # No natural degradation decay — the switch is what controls it
        dox_factor = np.exp(-np.log(2) * 0 / self.thresholds["doxycycline_half_life_hours"])  # dox is actively maintained
        oct4 = peak * vector_titer * dox_factor
        sox2 = peak * vector_titer * dox_factor * 0.95
        klf4 = peak * vector_titer * dox_factor * 0.90

        total = (oct4 + sox2 + klf4) / 3.0

        rejuvenation = self._compute_rejuvenation(total, days_on_dox)
        cancer_risk = self._compute_cancer_risk(days_on_dox, total)

        is_safe = (days_on_dox <= self.max_days and
                   cancer_risk < self.thresholds["cancer_risk_at_56d"] * 1.5 and
                   vector_present)
        within_dox_window = (self.thresholds["min_activation_days"] <=
                             days_on_dox <= self.max_days)

        return OSKDoxycyclineProfile(
            days_on_dox=days_on_dox,
            oct4_level=float(oct4),
            sox2_level=float(sox2),
            klf4_level=float(klf4),
            total_expression=float(total),
            rejuvenation_score=float(rejuvenation),
            cancer_risk=float(cancer_risk),
            is_safe=is_safe,
            within_dox_window=within_dox_window,
            vector_present=vector_present,
        )

    def _compute_rejuvenation(self, expression: float, days: float) -> float:
        opt_days = self.thresholds["optimal_activation_days"]
        time_factor = np.exp(-0.5 * ((days - opt_days) / 14) ** 2)
        expr_factor = np.exp(-0.5 * ((expression - 0.5) / 0.2) ** 2)
        return float(np.clip(time_factor * expr_factor, 0, 1))

    def _compute_cancer_risk(self, days: float, expression: float) -> float:
        if days <= 14:
            base_risk = 0.01
        elif days <= 28:
            base_risk = 0.03
        elif days <= 56:
            base_risk = self.thresholds["cancer_risk_at_56d"]
        elif days <= 70:
            base_risk = self.thresholds["cancer_risk_at_70d"]
        elif days <= 90:
            base_risk = self.thresholds["cancer_risk_at_90d"]
        else:
            base_risk = self.thresholds["cancer_risk_at_180d"]

        # Higher expression amplifies risk when doxycycline is taken too long
        expr_modifier = 1.0 + 0.5 * (expression - 0.5)
        return float(np.clip(base_risk * expr_modifier, 0, 1))

    def score_candidate(self, expression_level: float, days_on_dox: float,
                        years_since_delivery: float = 0.0) -> dict:
        profile = self.simulate_expression(expression_level, days_on_dox, years_since_delivery)

        safety_score = 1.0 - profile.cancer_risk
        efficacy_score = profile.rejuvenation_score
        window_compliance = 1.0 if profile.within_dox_window else 0.0

        composite = (
            0.40 * safety_score +
            0.35 * efficacy_score +
            0.25 * window_compliance
        )

        return {
            "composite_score": float(np.clip(composite, 0, 1)),
            "safety_score": float(safety_score),
            "efficacy_score": float(efficacy_score),
            "window_compliance": float(window_compliance),
            "profile": profile,
        }

    def constrain_generation(self, candidate, max_days: float = 56) -> dict:
        expression_level = getattr(candidate, "osk_expression_level", 0.5)
        days_on_dox = getattr(candidate, "doxycycline_days", 42)
        years_delivery = getattr(candidate, "years_since_delivery", 0.0)

        score = self.score_candidate(expression_level, min(days_on_dox, max_days), years_delivery)

        candidate.osk_safety_score = score["safety_score"]
        candidate.osk_efficacy_score = score["efficacy_score"]
        candidate.osk_composite = score["composite_score"]
        candidate.osk_cancer_risk = score["profile"].cancer_risk
        candidate.osk_duration_compliant = score["profile"].within_dox_window

        return score


class OSKPenalizedFitness:
    def __init__(self, max_days: float = 56, penalty_weight: float = 0.3):
        self.switch = OSKDoxycyclineSwitch(max_days=max_days)
        self.penalty_weight = penalty_weight

    def __call__(self, base_fitness: float, expression_level: float,
                 days_on_dox: float, years_since_delivery: float = 0.0) -> float:
        score = self.switch.score_candidate(expression_level, days_on_dox, years_since_delivery)
        penalty = (1.0 - score["composite_score"]) * self.penalty_weight
        penalized = base_fitness * (1.0 - penalty)
        return float(np.clip(penalized, 0, 1))
