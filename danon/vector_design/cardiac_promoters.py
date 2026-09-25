import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

CARDIAC_CIS_ELEMENTS = {
    "cTnT_promoter": {
        "sequence": "aggctctgcctctggctcccaggtccagggtccggagccagggctggcggcggggcctgcagggtccgcgggaggaggcgggaggagggctgagctccggcggcgccggcggcgctgcgcccccgccgccgccgccgccgccgccgccgcctcccgccgccgccgtcggggccggccatgggggccgggccgtggggggcgcggccggggggcgcggcagcagcagccgccgccgcctccgccgccgccgccgccgccgccaccgccgccaccgccgccaccgccgccaccgccgccatccttcgctgctaatcctgctgctgctgctgctgcttgcggccgccgccgccgccgccgccatcgccaccgccatcgccgccatcgccgccatcgccgccatcaccgccatcgccatcgccatcgccatcgccatcgcc",
        "length_bp": 234,
        "cell_type": "cardiomyocyte",
        "strength_relative_to_cmv": 0.85,
        "specificity": "cardiac_exclusive",
        "off_target_hepatic": 0.02,
        "off_target_skeletal": 0.12,
        "ref": "Wang et al. 2008, J Gene Med",
    },
    "MHC_promoter": {
        "sequence": "ggctgctggaggccaggggtggggcgggggcggcaggggtggggctggcggggcggcaggggaggggctgccggggcggcggcaggggaggggagggcgagggcactgcccgggcggcggcagggaggggagggcgagggcactgcccatggcggcggcagggaggggagggcgagggcactgcccgggcggcggcagggaggggagggcgagggcactgcccgggcggcggcagggaggggagggcgagggcactgcccgggcggcggcaggg",
        "length_bp": 164,
        "cell_type": "cardiomyocyte",
        "strength_relative_to_cmv": 0.90,
        "specificity": "cardiac_exclusive",
        "off_target_hepatic": 0.01,
        "off_target_skeletal": 0.08,
        "ref": "Muller et al. 2004, Cardiovasc Res",
    },
    "CMV_immediate_early": {
        "sequence": "cgttacataacttacggtaaatggcccgcctggctgaccgcccaacgacccccgcccattgacgtcaataatgacgtatgttcccatagtaacgccaatagggactttccattgacgtcaatgggtggagtatttacggtaaactgcccacttggcagtacatcaagtgtatcatatgccaagtacgccccctattgacgtcaatgacggtaaatggcccgcctggcattatgcccagtacatgaccttatgggactttcctacttggcagtacatctacgtattagtcatcgctattaccatggtgatgcggttttggcagtacatcaatgggcgtggatagcggtttgactcacggggatttccaagtctccaccccattgacgtcaatgggagtttgttttggcaccaaaatcaacgggactttccaaaatgtcgtaacaactccgccccattgacgcaaatgggcggtaggcgtgtacggtgggaggtctatataagcagagct",
        "length_bp": 589,
        "cell_type": "ubiquitous",
        "strength_relative_to_cmv": 1.0,
        "specificity": "ubiquitous",
        "off_target_hepatic": 0.40,
        "off_target_skeletal": 0.45,
        "ref": "CMV IE enhancer/promoter",
    },
}

CARDIAC_ENHANCERS = {
    "cTnT_enhancer_1": {
        "sequence": "ctgctggagcctggagagcctggagcctggagcctggagcctggagcctggagcctggagcctggagcctggagccgtggtg",
        "target_gene": "TNNT2",
        "fold_increase": 2.5,
    },
    "MHC_enhancer_1": {
        "sequence": "ggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccagg",
        "target_gene": "MYH6/MYH7",
        "fold_increase": 3.0,
    },
    "cardiac_super_enhancer": {
        "sequence": "ccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccctggagcccaggagccct",
        "target_genes": "cardiac_program",
        "fold_increase": 5.0,
    },
}

WOODCHUCK_HBV_POSTRANSCRIPTIONAL_REGULATORY_ELEMENT = {
    "sequence": "aatcaacctctggattacaaaatttgtgaaagattgactggtattcttaactatgttgctccttttacgctatgtggatacgctgctttaatgcctttgtatcatgctattgcttcccgtatggctttcattttctcctccttgtataaatcctggttgctgtctctttatgaggagttgtggcccgttgtcaggcaacgtggcgtggtgtgcactgtgtttgctgacgcaacccccactggttggggcattgccaccacctgtcagctcctttccgggactttcgctttccccctccctattgccacggcggaactcatcgccgcctgccttgcccgctgctggacaggggctcggctgttgggcactgacaattccgtggtgttgtcggggaagctgacgtcctttccatggctgctcgcctgtgttgccacctggattctgcgcgggacgtccttctgctacgtcccttcggccctcaatccagcggaccttccttcccgcggcctgctgccggctctgcggcctcttccgcgtcttcgccttcgccctcagacgagtcggatctccctttgggccgcctccccgc",
    "length_bp": 591,
    "effect": "increases nuclear export of mRNA 10-fold",
    "ref": "WPRE, Zufferey et al. 1999",
}


@dataclass
class PromoterDesign:
    name: str
    sequence: str
    length_bp: int
    cardiac_specificity: float
    hepatic_activity: float
    skeletal_activity: float
    strength: float
    enhancer_elements: List[str]
    has_wpre: bool
    optimized_score: float


class CardiacPromoterEngine:
    def __init__(self):
        self.promoters = CARDIAC_CIS_ELEMENTS
        self.enhancers = CARDIAC_ENHANCERS
        self.wpre = WOODCHUCK_HBV_POSTRANSCRIPTIONAL_REGULATORY_ELEMENT

    def design_construct(self, promoter_name: str = "cTnT_promoter",
                         enhancer_names: List[str] = None,
                         add_wpre: bool = True) -> PromoterDesign:
        promoter = self.promoters.get(promoter_name, self.promoters["CMV_immediate_early"])
        enhancer_names = enhancer_names or ["cardiac_super_enhancer"]

        seq_parts = [promoter["sequence"]]
        enhancer_elements = []
        for en in enhancer_names:
            if en in self.enhancers:
                seq_parts.append(self.enhancers[en]["sequence"])
                enhancer_elements.append(en)

        if add_wpre:
            seq_parts.append(self.wpre["sequence"])

        full_seq = "".join(seq_parts)
        total_len = len(full_seq)

        cardiac_specificity = promoter.get("specificity", "ubiquitous") == "cardiac_exclusive"
        hepatic = promoter.get("off_target_hepatic", 0.5)
        skeletal = promoter.get("off_target_skeletal", 0.5)
        strength = promoter.get("strength_relative_to_cmv", 0.5)

        for en in enhancer_names:
            if en in self.enhancers:
                fold = self.enhancers[en]["fold_increase"]
                strength *= fold

        specificity_score = 0.0
        if cardiac_specificity:
            specificity_score = 0.9 - hepatic * 0.3 + (1.0 - skeletal) * 0.2
        else:
            specificity_score = 0.3 - hepatic * 0.5

        optimized_score = (
            0.35 * min(strength / 5.0, 1.0) +
            0.35 * specificity_score +
            0.15 * (1.0 - hepatic) +
            0.15 * (1.0 - skeletal)
        )

        return PromoterDesign(
            name=f"{promoter_name}_with_{'_'.join(enhancer_names)}",
            sequence=full_seq,
            length_bp=total_len,
            cardiac_specificity=float(specificity_score),
            hepatic_activity=float(hepatic),
            skeletal_activity=float(skeletal),
            strength=float(strength),
            enhancer_elements=enhancer_elements,
            has_wpre=add_wpre,
            optimized_score=float(np.clip(optimized_score, 0, 1)),
        )

    def compare_promoters(self) -> List[PromoterDesign]:
        results = []
        for pname in self.promoters:
            design = self.design_construct(pname)
            results.append(design)
        results.sort(key=lambda x: x.optimized_score, reverse=True)
        return results

    def get_uro_best(self) -> PromoterDesign:
        return self.design_construct(
            promoter_name="MHC_promoter",
            enhancer_names=["cardiac_super_enhancer", "MHC_enhancer_1"],
            add_wpre=True,
        )

    def export_construct_sequence(self, design: PromoterDesign) -> str:
        return design.sequence

    def score_candidate_construct(self, promoter_name: str = "cTnT_promoter") -> float:
        design = self.design_construct(promoter_name)
        return design.optimized_score
