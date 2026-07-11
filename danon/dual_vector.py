import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Npu DnaE split intein — gold standard for protein trans-splicing
# N-terminal intein: 102 aa, C-terminal intein: 36 aa
# Total intein: 138 aa — smaller than any alternative
NPU_DNAE_INTEIN = {
    "N_terminal": {
        "sequence": "LAGVYCLPEDAILEPVSGRRILFALKLAREIELPPLPQLFQTATRISGLLTGEQVDVS"
                     "VGRRLALLLNAVTHSWTLIASGRTNAGVVVSLVPGDINIAFILIRDNVTFNSSI",
        "length_aa": 102,
    },
    "C_terminal": {
        "sequence": "IKIISGDSLSPTEAFLPPL",
        "length_aa": 36,
    },
    "total_length_aa": 138,
    "splicing_efficiency": 0.85,
    "ref": "Zettler et al. 2009, FEBS Lett",
}

# LAMP2B full-length: 410 aa
LAMP2B_PROTEIN = {
    "full_length_aa": 410,
    "molecular_weight_kda": 45,
    "signal_peptide": "MCFRLFVPLLLLLLVTSG",  # 1-18
    "lumenal_domain": "1-356",
    "transmembrane": "357-380",
    "cytoplasmic_tail": "381-410 (GYQTI)",
    "split_site_candidates": [
        {"position": 200, "intein_type": "Npu_DnaE", "efficiency": 0.82},
        {"position": 250, "intein_type": "Npu_DnaE", "efficiency": 0.78},
        {"position": 175, "intein_type": "Npu_DnaE", "efficiency": 0.85},
    ],
}

AAV9_PACKAGING_LIMITS = {
    "max_single_vector_kb": 4.7,
    "max_dual_vector_kb": 9.4,
    "typical_transgene_kb": 1.2,
    "typical_promoter_kb": 0.6,
    "typical_wpre_kb": 0.6,
    "typical_utr_kb": 0.2,
    "typical_polyA_kb": 0.2,
    "total_typical_cargo_kb": 2.8,
    "headroom_with_dual_kb": 6.6,
}


@dataclass
class DualVectorDesign:
    vector_a_id: str
    vector_b_id: str
    split_position_aa: int
    intein_type: str
    splicing_efficiency: float
    total_payload_kb: float
    has_cardiac_promoter: bool
    has_mirna_detarget: bool
    payload_headroom_kb: float
    design_score: float


class DualVectorEngine:
    def __init__(self):
        self.npu = NPU_DNAE_INTEIN
        self.lamp2b = LAMP2B_PROTEIN
        self.limits = AAV9_PACKAGING_LIMITS

    def design_split(self, split_position_aa: int = 200,
                     use_cardiac_promoter: bool = True,
                     use_mirna_detarget: bool = True) -> DualVectorDesign:
        split_pos = split_position_aa
        payload_total = self.limits["typical_transgene_kb"] + self.limits["typical_polyA_kb"]
        payload_a = payload_total * (split_pos / self.lamp2b["full_length_aa"])
        payload_b = payload_total * (1.0 - split_pos / self.lamp2b["full_length_aa"])

        if use_cardiac_promoter:
            payload_a += self.limits["typical_promoter_kb"]
            payload_b += self.limits["typical_promoter_kb"]

        if use_mirna_detarget:
            payload_a += 0.1
            payload_b += 0.1

        headroom = self.limits["max_dual_vector_kb"] - (payload_a + payload_b)

        splicing = 0.0
        for site in self.lamp2b["split_site_candidates"]:
            if abs(site["position"] - split_position_aa) <= 25:
                splicing = site["efficiency"]
                break
        if splicing == 0.0:
            splicing = 0.70

        score = (
            0.30 * splicing +
            0.25 * min(headroom / 2.0, 1.0) +
            0.20 * (1.0 if use_cardiac_promoter else 0.0) +
            0.15 * (1.0 if use_mirna_detarget else 0.0) +
            0.10 * (1.0 - abs(split_position_aa - 200) / 300)
        )

        return DualVectorDesign(
            vector_a_id=f"AAV9-LAMP2B_N{self.npu['N_terminal']['length_aa']}",
            vector_b_id=f"AAV9-LAMP2B_C{self.npu['C_terminal']['length_aa']}",
            split_position_aa=split_position_aa,
            intein_type="Npu_DnaE",
            splicing_efficiency=float(splicing),
            total_payload_kb=float(payload_a + payload_b + 2.8),
            has_cardiac_promoter=use_cardiac_promoter,
            has_mirna_detarget=use_mirna_detarget,
            payload_headroom_kb=float(headroom),
            design_score=float(np.clip(score, 0, 1)),
        )

    def utcl_single_vector_capacity(self) -> float:
        return self.limits["max_single_vector_kb"]

    def our_dual_vector_capacity(self) -> float:
        return self.limits["max_dual_vector_kb"]

    def capacity_improvement(self) -> float:
        return self.our_dual_vector_capacity() / self.utcl_single_vector_capacity()

    def optimize_split_position(self) -> int:
        best_score = -1.0
        best_pos = 200
        for pos in range(150, 301, 5):
            design = self.design_split(pos)
            if design.design_score > best_score:
                best_score = design.design_score
                best_pos = pos
        return best_pos
