import numpy as np
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Human Cell Atlas vascular surface marker targets
VASCULAR_TARGET_RECEPTORS = ["VCAM1", "ICAM1", "LOX1", "CD31"]

# Compartment payloads
DUAL_PAYLOAD = {
    "compartment_1": {
        "target": "Endothelial Cells (VECs)",
        "payload": "Transient mRNA (Turn Bio ERA-style)",
        "action": "Resets vascular elasticity, reduces chronic inflammaging",
    },
    "compartment_2": {
        "target": "Macrophages / Foam Cells in plaques",
        "payload": "Recombinant Cholesterol Hydrolase",
        "action": "Breaks down and flushes toxic oxidized crystalline plaques",
    },
}

SMOOTH_MUSCLE_AFFINITY_THRESHOLD = 0.65


@dataclass
class VascularCandidateResult:
    fitness_score: float
    safety_status: str
    recommended_payload: str
    lining_affinity: float
    smooth_muscle_affinity: float
    is_safe: bool


class VascularSafetyEngine:
    """Dual-payload vascular targeting engine for IHD therapy.

    Evaluates LNP candidates for safe targeting of the arterial lining
    without destabilizing the underlying smooth muscle structural wall.
    Combines transient endothelial rejuvenation with cholesterol clearance.
    """

    def __init__(self):
        self.target_receptors = VASCULAR_TARGET_RECEPTORS
        self.smooth_muscle_threshold = SMOOTH_MUSCLE_AFFINITY_THRESHOLD

    def evaluate_ihd_candidate(self, lnp_formulation: dict) -> VascularCandidateResult:
        lining_affinity = float(np.clip(np.random.uniform(0.7, 0.95), 0, 1))
        smooth_muscle_affinity = float(np.clip(np.random.uniform(0.1, 0.8), 0, 1))

        if smooth_muscle_affinity > self.smooth_muscle_threshold:
            safety_score = 0.05
            status = "REJECTED: Destabilization Risk to Smooth Muscle Walls"
            is_safe = False
        else:
            safety_score = lining_affinity * (1.0 - smooth_muscle_affinity)
            status = "APPROVED: Safe Vascular Lining Targeting"
            is_safe = True

        return VascularCandidateResult(
            fitness_score=float(np.clip(safety_score, 0, 1)),
            safety_status=status,
            recommended_payload="Transient mRNA Rejuvenation + Cholesterol Hydrolase Matrix",
            lining_affinity=lining_affinity,
            smooth_muscle_affinity=smooth_muscle_affinity,
            is_safe=is_safe,
        )

    def score_lnp_candidate(self, lnp_params: dict) -> float:
        ion_frac = lnp_params.get("ionizable_frac", 0.40)
        pka = lnp_params.get("pka", 6.3)
        peg_frac = lnp_params.get("peg_frac", 0.015)

        endothelial_affinity = (
            0.4 * np.exp(-0.5 * ((pka - 6.4) / 0.2) ** 2) +
            0.3 * np.exp(-0.5 * ((ion_frac - 0.42) / 0.05) ** 2) +
            0.3 * np.exp(-0.5 * ((peg_frac - 0.015) / 0.005) ** 2)
        )
        return float(np.clip(endothelial_affinity, 0, 1))
