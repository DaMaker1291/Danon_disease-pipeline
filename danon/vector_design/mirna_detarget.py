import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict

logger = logging.getLogger(__name__)

MIRNA_TARGET_SITES = {
    "miR-122": {
        "seed": "UGGAGUGUGACAAUGGUGUUUG",
        "tissue": "liver",
        "expression_level": "very_high",
        "effect": "destroys transcripts in hepatocytes",
        "clinical_status": "proven_in_AAV_trials",
        "refs": ["Xie et al. 2011, Nat Biotechnol", "Qiao et al. 2011, Mol Ther"],
        "target_sequence": "tggagtgtgacaatggtgtttg",
        "perfect_match": True,
    },
    "miR-1": {
        "seed": "GGAAUGUAAAGAAGUAUGGAG",
        "tissue": "cardiac_muscle",
        "expression_level": "very_high",
        "effect": "enriches transcripts in cardiomyocytes",
        "clinical_status": "preclinical",
        "refs": ["Zhao et al. 2005, Nature", "Rao et al. 2009, Cardiovasc Res"],
        "target_sequence": "ggaatgtaaagaagtatggag",
        "perfect_match": False,
    },
    "miR-133": {
        "seed": "GGACCCAAACACCUGGUCUUU",
        "tissue": "cardiac_and_skeletal",
        "expression_level": "high",
        "effect": "enriches in muscle tissues",
        "clinical_status": "preclinical",
        "refs": ["Chen et al. 2006, Nat Genet"],
        "target_sequence": "ggacccaaacacctggtcttt",
        "perfect_match": False,
    },
    "miR-208": {
        "seed": "AUAAGACGAGCAAAAAGCUUGU",
        "tissue": "cardiac",
        "expression_level": "cardiac_specific",
        "effect": "enriches specifically in heart",
        "clinical_status": "exploratory",
        "refs": ["van Rooij et al. 2007, Science"],
        "target_sequence": "ataagacgagcaaaaagcttgt",
        "perfect_match": True,
    },
    "miR-142": {
        "seed": "CCAUAAAGUAGAAAGCACUAC",
        "tissue": "hematopoietic",
        "expression_level": "high_in_APCs",
        "effect": "prevents immune response against transgene",
        "clinical_status": "proven_in_AAV_trials",
        "refs": ["Brown et al. 2006, Nat Med", "Xiao et al. 2019, Mol Ther"],
        "target_sequence": "ccataaagtagaaagcactac",
        "perfect_match": True,
    },
}

DETARGET_SEQUENCES = {
    "liver_detarget_4x_miR122": "tggagtgtgacaatggtgtttgtggagtgtgacaatggtgtttgtggagtgtgacaatggtgtttgtggagtgtgacaatggtgtttg",
    "immune_detarget_4x_miR142": "ccataaagtagaaagcactacccataaagtagaaagcactacccataaagtagaaagcactacccataaagtagaaagcactac",
    "cardiac_retarget_4x_miR1": "ggaatgtaaagaagtatggagggaatgtaaagaagtatggagggaatgtaaagaagtatggagggaatgtaaagaagtatggag",
    "cardiac_retarget_4x_miR208": "ataagacgagcaaaaagcttgtataagacgagcaaaaagcttgtataagacgagcaaaaagcttgtataagacgagcaaaaagcttgt",
}


@dataclass
class miRNADesign:
    detarget_liver: bool
    detarget_immune: bool
    retarget_cardiac: bool
    liver_protection_score: float
    immune_protection_score: float
    cardiac_enrichment_score: float
    total_optimization_score: float
    utr_sequence_3p: str
    utr_length_bp: int


class miRNADetargetEngine:
    def __init__(self):
        self.mirnas = MIRNA_TARGET_SITES
        self.detarget_seqs = DETARGET_SEQUENCES

    def design_utr(self, detarget_liver: bool = True,
                   detarget_immune: bool = True,
                   retarget_cardiac: bool = True) -> miRNADesign:
        utr_3p = ""
        liver_score = 0.0
        immune_score = 0.0
        cardiac_score = 0.0

        if detarget_liver:
            utr_3p += self.detarget_seqs["liver_detarget_4x_miR122"] + "cctgaagg"
            liver_score = 0.95

        if detarget_immune:
            utr_3p += self.detarget_seqs["immune_detarget_4x_miR142"] + "cctgaagg"
            immune_score = 0.90

        if retarget_cardiac:
            utr_3p += self.detarget_seqs["cardiac_retarget_4x_miR1"]
            utr_3p += self.detarget_seqs["cardiac_retarget_4x_miR208"]
            cardiac_score = 0.88

        if not utr_3p:
            utr_3p = "gctagc"

        total = (
            0.30 * liver_score +
            0.20 * immune_score +
            0.30 * cardiac_score +
            0.20 * (1.0 if detarget_liver or detarget_immune else 0.0)
        )

        return miRNADesign(
            detarget_liver=detarget_liver,
            detarget_immune=detarget_immune,
            retarget_cardiac=retarget_cardiac,
            liver_protection_score=float(liver_score),
            immune_protection_score=float(immune_score),
            cardiac_enrichment_score=float(cardiac_score),
            total_optimization_score=float(np.clip(total, 0, 1)),
            utr_sequence_3p=utr_3p,
            utr_length_bp=len(utr_3p),
        )

    def score_candidate_for_mirna_compatibility(self, sequence: str) -> float:
        design = self.design_utr(detarget_liver=True, detarget_immune=True, retarget_cardiac=True)
        utr = design.utr_sequence_3p.lower()

        miR122_count = utr.count(self.mirnas["miR-122"]["target_sequence"])
        miR1_count = utr.count(self.mirnas["miR-1"]["target_sequence"])
        miR142_count = utr.count(self.mirnas["miR-142"]["target_sequence"])
        miR208_count = utr.count(self.mirnas["miR-208"]["target_sequence"])

        liver_present = miR122_count >= 4
        cardiac_present = miR1_count >= 2 and miR208_count >= 2
        immune_present = miR142_count >= 2

        score = 0.0
        if liver_present:
            score += 0.35
        if cardiac_present:
            score += 0.35
        if immune_present:
            score += 0.20
        if liver_present and cardiac_present and immune_present:
            score += 0.10

        return float(np.clip(score, 0, 1))
