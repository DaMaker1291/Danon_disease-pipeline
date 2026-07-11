import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

logger = logging.getLogger(__name__)

CLINICAL_STRAINS = {
    "AAV9_N87_epitope": {
        "region": "N87",
        "positions": list(range(86, 92)),
        "antibody_coverage": ">90% of human sera",
        "role": "major neutralizing epitope",
    },
    "AAV9_AAV8_cross": {
        "region": "VR_VIII",
        "positions": list(range(570, 590)),
        "antibody_coverage": "cross-reactive",
        "role": "shared epitope with AAV8",
    },
}

GLYCAN_SHIELDING_CANDIDATES = {
    "N87_NXT_mutant": {
        "mutation": "N87T + S89N (creates NXT sequon)",
        "glycan_site": "N87_N89T",
        "glycan_type": "N-linked",
        "immune_evasion_gain": 0.35,
        "tropism_impact": -0.05,
        "feasibility": "high",
        "ref": "Pulicherla et al. 2011, Mol Ther",
    },
    "S445_NXT_mutant": {
        "mutation": "S445N (creates NXT sequon at 445)",
        "glycan_site": "N445_N447T",
        "glycan_type": "N-linked",
        "immune_evasion_gain": 0.25,
        "tropism_impact": -0.10,
        "feasibility": "medium",
        "ref": "Marshall et al. 2012, J Virol",
    },
    "triple_shield": {
        "mutation": "N87T_S89N + S445N + T456N",
        "glycan_site": "triple_NXT",
        "glycan_type": "N-linked x3",
        "immune_evasion_gain": 0.55,
        "tropism_impact": -0.15,
        "feasibility": "low",
        "ref": "Proposed by this pipeline",
    },
}

EMPTY_CAPSID_DECOY = {
    "ratio_to_full": {"min": 10, "max": 100},
    "optimal": 30,
    "immune_evasion_gain": 0.40,
    "mechanism": "absorbs neutralizing antibodies before full capsids arrive",
    "clinical_evidence": "Mingozzi et al. 2013, Sci Transl Med",
}


@dataclass
class ImmuneStealthStrategy:
    glycan_shielding: str
    empty_capsid_ratio: int
    immune_evasion_score: float
    tropism_penalty: float
    manufacturing_feasibility: float
    overall_score: float


class ImmuneStealthEngine:
    def __init__(self):
        self.glycans = GLYCAN_SHIELDING_CANDIDATES
        self.decoy = EMPTY_CAPSID_DECOY

    def design_stealth(self, shielding_strategy: str = "N87_NXT_mutant",
                       empty_capsid_ratio: int = 30) -> ImmuneStealthStrategy:
        glycan = self.glycans.get(shielding_strategy, self.glycans["N87_NXT_mutant"])
        evasion_gain = glycan["immune_evasion_gain"]
        tropism_penalty = glycan["tropism_impact"]

        decoy_gain = self.decoy["immune_evasion_gain"]
        ratio_factor = 1.0 - abs(empty_capsid_ratio - self.decoy["optimal"]) / self.decoy["optimal"]

        total_evasion = evasion_gain + decoy_gain * ratio_factor
        total_tropism_penalty = abs(tropism_penalty)

        feasibility = 1.0
        if shielding_strategy == "triple_shield":
            feasibility = 0.3
        elif shielding_strategy == "S445_NXT_mutant":
            feasibility = 0.6
        else:
            feasibility = 0.9

        score = (
            0.35 * total_evasion +
            0.25 * (1.0 - total_tropism_penalty) +
            0.25 * feasibility +
            0.15 * ratio_factor
        )

        return ImmuneStealthStrategy(
            glycan_shielding=shielding_strategy,
            empty_capsid_ratio=empty_capsid_ratio,
            immune_evasion_score=float(np.clip(total_evasion, 0, 1)),
            tropism_penalty=float(total_tropism_penalty),
            manufacturing_feasibility=float(feasibility),
            overall_score=float(np.clip(score, 0, 1)),
        )

    def utcl_stealth_score(self) -> float:
        """UCL: AAV9 wild-type, no shielding, no decoys."""
        return 0.10

    def our_stealth_score(self) -> float:
        best = self.design_stealth("triple_shield", 30)
        return best.overall_score
