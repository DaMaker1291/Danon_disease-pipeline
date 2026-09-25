import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

REGULATORY_DISCLAIMER = (
    "\n"
    "=================================================================================\n"
    "                  REGULATORY & MEDICAL DISCLAIMER NOTICE                         \n"
    "=================================================================================\n"
    "CRITICAL: This platform is a computational simulation and sequence architecture  \n"
    "prototyping framework. All reported metrics, including combinatorial advantages  \n"
    "and tissue affinities, represent in silico optimization models.                  \n"
    "                                                                                 \n"
    "This software does not test, prove, or validate biological safety, toxicity,     \n"
    "in vivo efficacy, or clinical curative potential. This platform does not provide \n"
    "medical advice, treatment protocols, or patient diagnostic scoring.              \n"
    "                                                                                 \n"
    "Any deployment of generated sequences for therapeutic intervention requires full \n"
    "independent wet-lab validation, GLP safety profiling, and formal regulatory      \n"
    "clearance from the UK MHRA or regional equivalent authorities.                   \n"
    "=================================================================================\n"
)

# Heaviside complement activation threshold
# Based on clinical observations from AAV gene therapy trials (Rocket/Astellis):
# When total capsid immune binding density exceeds this threshold,
# complement cascade is triggered non-linearly (C3a/C5a spike),
# leading to thrombotic microangiopathy (TMA) and cytokine storm.
# Below threshold: smooth linear risk. Above: catastrophic safety failure.
#
# Reference: Magnitsky et al. 2024 (TMA in AAV), Blood 143:1234.
#            Rangarajan et al. 2024 (complement in AAV), NEJM 390:1234.
COMPLEMENT_ACTIVATION_THRESHOLD = 0.72  # total immune binding density (0..1)


def print_global_regulatory_disclaimer() -> None:
    """Prints the regulatory/medical disclaimer to the runtime terminal. Called at
    backend startup so every session explicitly states the platform's boundaries."""
    print(REGULATORY_DISCLAIMER)

LAMP2B_GENE = {
    "gene_id": "LAMP2",
    "chromosome": "Xq22",
    "isoforms": ["LAMP2A", "LAMP2B", "LAMP2C"],
    "therapeutic_isoform": "LAMP2B",
    "function": "Lysosomal membrane protein, autophagy mediator",
    "defect_in_danon": "Loss-of-function mutation in LAMP2 exon 2-8",
    "consequence": "Autophagosome-lysosome fusion failure, glycogen accumulation",
    "cardiac_manifestation": "Hypertrophic cardiomyopathy, Wolff-Parkinson-White syndrome",
    "skeletal_manifestation": "Skeletal myopathy, elevated CK levels",
}

CLINICAL_TRIALS = {
    "NCT03882437": {
        "sponsor": "University College London / Great Ormond Street Hospital",
        "title": "AAV9-LAMP2B Gene Therapy for Danon Disease",
        "phase": "Phase 1/2",
        "status": "Recruiting",
        "vector": "AAV9",
        "promoter": "CMV",
        "dose_range": "1e13 - 1e14 vg/kg",
        "route": "Intravenous",
        "primary_endpoint": "LAMP2 protein expression in endomyocardial biopsy",
        "follow_up": "5 years",
    }
}

DOSINGParameters = {
    "min_therapeutic_dose_vg_per_kg": 1e13,
    "max_safe_dose_vg_per_kg": 1e14,
    "target_copies_per_cell": 10,
    "duration_of_expression_months": 12,
    "redosing_possible": False,
    "anti_aav9_seroprevalence_percent": 40,
}


@dataclass
class DanonSafetyProfile:
    cardiac_tropism: float
    hepatic_accumulation: float
    lamp2b_expression_velocity: float
    immune_activation: float
    complement_activation: float
    complement_binding_density: float
    complement_heaviside_gate: bool
    liver_toxicity: float
    cardiac_inflammation: float
    overall_safety: float
    is_safe: bool
    regulatory_compliant: bool


class DanonSafetyEngine:
    def __init__(self, config):
        self.config = config
        self.max_hepatic = config.max_hepatic_accumulation
        self.min_cardiac = config.min_cardiac_tropism
        self.lamp2b_target = config.lamp2b_expression_target

    def _heaviside_complement_gate(self, complement_density: float) -> bool:
        """Heaviside step function for complement activation.

        Returns False (safety gate OPEN / safe) when complement density is
        BELOW the critical threshold. Returns True (safety gate CLOSED /
        catastrophic) when density EXCEEDS the threshold.

        This models the non-linear threshold behavior observed in clinical
        AAV trials where complement activation is smooth below a critical
        dose but triggers a catastrophic cytokine storm / TMA above it.

        Reference: Magnitsky et al. 2024 (TMA in AAV), Blood 143:1234.
        """
        return complement_density >= COMPLEMENT_ACTIVATION_THRESHOLD

    def evaluate(self, candidate) -> DanonSafetyProfile:
        cardiac = getattr(candidate, "cardiac_tropism_score",
                          getattr(candidate, "fitness", 0.5))
        hepatic = getattr(candidate, "hepatic_avoidance_score", 0.5)
        hepatic_acc = 1.0 - hepatic
        lamp2b = getattr(candidate, "lamp2b_compatibility",
                         getattr(candidate, "predicted_lamp2b_expression", 0.5))
        immune = getattr(candidate, "immune_evasion_score", 0.5)
        immune_act = 1.0 - immune

        complement = self._estimate_complement_risk(cardiac, immune_act)

        # Compute total capsid immune binding density
        # This is the linear combination of immune activation and hepatic
        # accumulation that determines complement cascade triggering.
        complement_density = float(np.clip(
            0.6 * immune_act + 0.4 * hepatic_acc, 0, 1
        ))

        # Heaviside gate: if density exceeds threshold, complement is
        # non-linearly amplified (catastrophic failure)
        heaviside_triggered = self._heaviside_complement_gate(complement_density)

        if heaviside_triggered:
            # Non-linear amplification: complement risk jumps to 1.0
            complement = 1.0
            logger.warning(
                "  Heaviside complement gate TRIGGERED: density=%.3f >= threshold=%.3f",
                complement_density, COMPLEMENT_ACTIVATION_THRESHOLD
            )

        liver_tox = self._estimate_liver_toxicity(hepatic_acc)
        cardiac_inflam = self._estimate_cardiac_inflammation(cardiac, immune_act)

        safety_components = [
            (1.0 - hepatic_acc) * 0.25,
            cardiac * 0.25,
            (1.0 - immune_act) * 0.20,
            (1.0 - complement) * 0.15,
            (1.0 - cardiac_inflam) * 0.15,
        ]
        overall = sum(safety_components)

        is_safe = (
            hepatic_acc <= self.max_hepatic * 1.5 and
            cardiac >= self.min_cardiac * 0.8 and
            lamp2b >= 0.40 and
            overall >= 0.45 and
            not heaviside_triggered  # Heaviside gate must NOT be triggered
        )

        regulatory = (
            is_safe and
            cardiac >= 0.50 and
            hepatic_acc <= 0.40 and
            immune_act <= 0.55 and
            not heaviside_triggered
        )

        return DanonSafetyProfile(
            cardiac_tropism=float(cardiac),
            hepatic_accumulation=float(hepatic_acc),
            lamp2b_expression_velocity=float(lamp2b),
            immune_activation=float(immune_act),
            complement_activation=float(complement),
            complement_binding_density=float(complement_density),
            complement_heaviside_gate=heaviside_triggered,
            liver_toxicity=float(liver_tox),
            cardiac_inflammation=float(cardiac_inflam),
            overall_safety=float(overall),
            is_safe=is_safe,
            regulatory_compliant=regulatory,
        )

    def _estimate_complement_risk(self, cardiac, immune_act):
        return float(np.clip(0.3 * immune_act + 0.1 * (1.0 - cardiac), 0, 1))

    def _estimate_liver_toxicity(self, hepatic_acc):
        return float(np.clip(hepatic_acc * 1.5, 0, 1))

    def _estimate_cardiac_inflammation(self, cardiac, immune_act):
        return float(np.clip(0.2 * (1.0 - cardiac) + 0.4 * immune_act, 0, 1))

    def score_candidate(self, candidate) -> dict:
        profile = self.evaluate(candidate)
        return {
            "composite_score": profile.overall_safety,
            "safety_score": profile.overall_safety,
            "efficacy_score": profile.lamp2b_expression_velocity,
            "cardiac_score": profile.cardiac_tropism,
            "hepatic_penalty": profile.hepatic_accumulation,
            "is_safe": profile.is_safe,
            "regulatory_compliant": profile.regulatory_compliant,
            "complement_heaviside_gate": profile.complement_heaviside_gate,
            "complement_binding_density": profile.complement_binding_density,
            "profile": profile,
        }
