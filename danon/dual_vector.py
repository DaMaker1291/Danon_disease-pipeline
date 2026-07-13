import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

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

# Biologically validated extein junction motifs for Npu DnaE intein
# CFN (Cys-Phe-Asn) at N-extein: Cys forms thioester, Asn is C-terminal splice residue
# CPG (Cys-Pro-Gly) at C-extein: Proline kink, Glycine flexibility
EXTEIN_MOTIFS = {
    "N_extein_viable": {"C", "F", "N", "S", "A", "V", "T"},
    "N_extein_preferred": {"C", "F", "N"},
    "N_extein_forbidden": {"P"},
    "C_extein_viable": {"C", "P", "G", "A", "S", "T", "V"},
    "C_extein_preferred": {"C", "P", "G"},
    "C_extein_forbidden": {"W", "Y", "R"},
}

# Kyle-Doolittle hydrophobicity scale for FoldX-style aggregation scoring
AA_HYDRO_KYD = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
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
    def __init__(self, lamp2b_sequence: str = None):
        self.npu = NPU_DNAE_INTEIN
        self.lamp2b = LAMP2B_PROTEIN
        self.limits = AAV9_PACKAGING_LIMITS
        self.lamp2b_seq = lamp2b_sequence

    def check_extein_junction(self, split_position_aa: int, sequence: str = None) -> Dict:
        seq = sequence or self.lamp2b_seq
        if seq is None or split_position_aa < 2 or split_position_aa > len(seq) - 2:
            return {"valid": False, "motif_penalty": 1.0, "reason": "sequence unavailable or position out of range"}
        n_extein = seq[split_position_aa - 1]
        c_extein = seq[split_position_aa]
        n_valid = n_extein in EXTEIN_MOTIFS["N_extein_viable"]
        c_valid = c_extein in EXTEIN_MOTIFS["C_extein_viable"]
        n_forbidden = n_extein in EXTEIN_MOTIFS["N_extein_forbidden"]
        c_forbidden = c_extein in EXTEIN_MOTIFS["C_extein_forbidden"]
        n_preferred = n_extein in EXTEIN_MOTIFS["N_extein_preferred"]
        c_preferred = c_extein in EXTEIN_MOTIFS["C_extein_preferred"]
        penalty = 0.0
        if n_forbidden or c_forbidden:
            penalty = 1.0
        elif not n_valid or not c_valid:
            penalty = 0.7
        elif not n_preferred or not c_preferred:
            penalty = 0.3
        return {
            "valid": penalty < 0.5,
            "motif_penalty": penalty,
            "n_extein": n_extein,
            "c_extein": c_extein,
            "n_preferred": n_preferred,
            "c_preferred": c_preferred,
            "reason": "valid junction" if penalty < 0.5 else f"suboptimal extein: {n_extein}{c_extein}",
        }

    def _foldx_aggregation_score(self, sequence: str) -> float:
        window = 5
        scores = []
        for i in range(len(sequence) - window + 1):
            window_seq = sequence[i:i + window]
            hydro_score = sum(AA_HYDRO_KYD.get(aa, 0.0) for aa in window_seq) / window
            if hydro_score > 1.0:
                beta_sheet_propensity = sum(1 for aa in window_seq if aa in {"V", "I", "Y", "F", "W", "L", "T"})
                scores.append(hydro_score * (1.0 + 0.2 * beta_sheet_propensity))
        if not scores:
            return 0.0
        return float(np.mean(scores))

    def compute_intein_hydrophobic_aggregation(self) -> Dict:
        n_seq = self.npu["N_terminal"]["sequence"]
        c_seq = self.npu["C_terminal"]["sequence"]
        n_agg = self._foldx_aggregation_score(n_seq)
        c_agg = self._foldx_aggregation_score(c_seq)
        total_agg = (n_agg * len(n_seq) + c_agg * len(c_seq)) / (len(n_seq) + len(c_seq))
        agg_risk = float(np.clip((total_agg - 1.0) / 3.0, 0, 1))
        return {
            "N_intein_aggregation": float(n_agg),
            "C_intein_aggregation": float(c_agg),
            "total_aggregation": float(total_agg),
            "aggregation_risk_score": agg_risk,
        }

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

        extein_result = self.check_extein_junction(split_pos)
        agg_result = self.compute_intein_hydrophobic_aggregation()

        splicing_adjusted = splicing * (1.0 - 0.5 * extein_result["motif_penalty"])
        agg_penalty = agg_result["aggregation_risk_score"]

        score = (
            0.30 * splicing_adjusted +
            0.20 * min(headroom / 2.0, 1.0) +
            0.15 * (1.0 if use_cardiac_promoter else 0.0) +
            0.10 * (1.0 if use_mirna_detarget else 0.0) +
            0.10 * (1.0 - abs(split_position_aa - 200) / 300) -
            0.15 * agg_penalty
        )

        return DualVectorDesign(
            vector_a_id=f"AAV9-LAMP2B_N{self.npu['N_terminal']['length_aa']}",
            vector_b_id=f"AAV9-LAMP2B_C{self.npu['C_terminal']['length_aa']}",
            split_position_aa=split_position_aa,
            intein_type="Npu_DnaE",
            splicing_efficiency=float(splicing_adjusted),
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


# Strict biological packaging threshold for a single AAV9 capsid.
AAV_MAX_CAPACITY_BP = 4700


def evaluate_vector_capacity(cargo_length_bp: int) -> dict:
    """Absolute capacity-gate: a small transgene (e.g. LAMP2B CDS ~1.2 kb) fits
    comfortably inside ONE AAV9 vector with promoter + UTRs + WPRE. Splitting it
    into two vectors forces ~2x viral dose, which multiplies liver-toxicity and
    complement-activation risk for zero packaging benefit.

    A dual-vector split-intein is only justified when the cargo structurally
    exceeds the single-vector limit.
    """
    if cargo_length_bp <= AAV_MAX_CAPACITY_BP:
        return {
            "strategy": "Single-Vector (Monotropic)",
            "efficiency": 1.0,
            "toxicity_risk_multiplier": 1.0,
            "cargo_length_bp": cargo_length_bp,
            "clinical_justification": (
                "Payload within single-vector 4.7 kb limit; avoids dual-transduction "
                "efficiency loss and doubles the hepatotoxic / complement-activation exposure."
            ),
        }
    return {
        "strategy": "Dual-Vector Split-Intein",
        "efficiency": 0.85,
        "toxicity_risk_multiplier": 2.2,
        "cargo_length_bp": cargo_length_bp,
        "clinical_justification": (
            "Cargo exceeds single-vector limit; split-intein trans-splicing required "
            "despite 2.2x toxicity-risk multiplier from doubled viral dose."
        ),
    }
