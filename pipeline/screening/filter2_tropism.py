import logging
import numpy as np
from pipeline.screening import Filter
from pipeline.generation.aav_generator import AAVCandidate

logger = logging.getLogger(__name__)

AGING_SPECIFIC_RECEPTORS = {
    "cardiac": {
        "primary": {"name": "integrin_alpha7_beta1", "positions": [265, 270, 275], "affinity": -8.2},
        "secondary": {"name": "SIRP_alpha", "positions": [448, 453, 458], "affinity": -7.5},
        "aging_markers": {"name": "ICAM1", "positions": [380, 385, 390], "affinity": -7.8},
        "senescence_receptor": {"name": "uPAR", "positions": [520, 525, 530], "affinity": -7.0},
    },
    "neuronal": {
        "primary": {"name": "AAVR_KIAA0319L", "positions": [265, 270, 275], "affinity": -9.0},
        "secondary": {"name": "lamr_receptor", "positions": [458, 463, 468], "affinity": -6.8},
        "aging_markers": {"name": "RAGE", "positions": [400, 405, 410], "affinity": -7.2},
        "senescence_receptor": {"name": "p75NTR", "positions": [570, 575, 580], "affinity": -6.5},
    },
    "joint_cartilage": {
        "primary": {"name": "CD44", "positions": [325, 330, 335], "affinity": -7.8},
        "secondary": {"name": "integrin_alpha1_beta1", "positions": [265, 270, 275], "affinity": -7.0},
        "aging_markers": {"name": "MMP13_receptor", "positions": [450, 455, 460], "affinity": -6.8},
        "senescence_receptor": {"name": "NOTCH1", "positions": [580, 585, 590], "affinity": -6.5},
    },
    "skeletal_muscle": {
        "primary": {"name": "dystroglycan", "positions": [265, 270, 275], "affinity": -8.5},
        "secondary": {"name": "integrin_alphaV_beta3", "positions": [448, 453], "affinity": -7.2},
        "aging_markers": {"name": "TGF_beta_R1", "positions": [380, 385], "affinity": -7.0},
        "senescence_receptor": {"name": "IGF1R", "positions": [520, 525], "affinity": -6.8},
    },
    "hepatic": {
        "primary": {"name": "AAVR", "positions": [265, 270, 275], "affinity": -9.5},
        "secondary": {"name": "heparan_sulfate", "positions": [458, 463], "affinity": -8.0},
        "aging_markers": {"name": "ASGPR", "positions": [380, 385], "affinity": -7.5},
        "senescence_receptor": {"name": "CD133", "positions": [520, 525], "affinity": -6.8},
    },
    "renal": {
        "primary": {"name": "megalin", "positions": [265, 270, 275], "affinity": -8.0},
        "secondary": {"name": "cubilin", "positions": [448, 453], "affinity": -7.5},
        "aging_markers": {"name": "Klotho", "positions": [380, 385], "affinity": -7.0},
        "senescence_receptor": {"name": "AT1R", "positions": [520, 525], "affinity": -6.5},
    },
}

SPATIAL_WEIGHTS = {
    "cardiac": {"sialic_acid": 0.30, "lamr": 0.25, "charge": 0.15, "stability": 0.15, "size": 0.15},
    "neuronal": {"sialic_acid": 0.20, "lamr": 0.30, "charge": 0.20, "stability": 0.15, "size": 0.15},
    "joint_cartilage": {"sialic_acid": 0.25, "lamr": 0.20, "charge": 0.25, "stability": 0.15, "size": 0.15},
    "skeletal_muscle": {"sialic_acid": 0.25, "lamr": 0.25, "charge": 0.10, "stability": 0.20, "size": 0.20},
    "hepatic": {"sialic_acid": 0.35, "lamr": 0.15, "charge": 0.10, "stability": 0.20, "size": 0.20},
    "renal": {"sialic_acid": 0.30, "lamr": 0.20, "charge": 0.15, "stability": 0.20, "size": 0.15},
}

CHARGE_PROFILE = {
    "A": 0.0, "C": 0.1, "D": -1.0, "E": -1.0, "F": 0.0,
    "G": 0.0, "H": 0.5, "I": 0.0, "K": 1.0, "L": 0.0,
    "M": 0.0, "N": 0.0, "P": 0.0, "Q": 0.0, "R": 1.0,
    "S": 0.0, "T": 0.0, "V": 0.0, "W": 0.0, "Y": 0.0,
}

# Vascular dual-compartment targets for IHD therapy
# Compartment 1: Endothelial lining (rejuvenation payload)
# Compartment 2: Macrophage/foam cells in plaques (cholesterol hydrolase payload)
VASCULAR_RECEPTORS = {
    "endothelial": {
        "primary": {"name": "VCAM1", "positions": [265, 270, 275], "affinity": -8.5,
                     "role": "inflamed endothelium homing"},
        "secondary": {"name": "ICAM1", "positions": [380, 385, 390], "affinity": -7.8,
                       "role": "leukocyte adhesion site"},
        "homing": {"name": "CD31", "positions": [448, 453], "affinity": -8.0,
                    "role": "endothelial cell junction integrity"},
        "aging_markers": {"name": "LOX1", "positions": [520, 525], "affinity": -7.2,
                           "role": "oxidized LDL receptor, plaque indicator"},
    },
    "macrophage_foam_cell": {
        "primary": {"name": "SR_A", "positions": [265, 270, 275], "affinity": -8.0,
                     "role": "scavenger receptor, plaque uptake"},
        "secondary": {"name": "CD36", "positions": [380, 385], "affinity": -7.5,
                       "role": "foam cell formation marker"},
        "aging_markers": {"name": "TLR4", "positions": [448, 453], "affinity": -7.0,
                           "role": "inflammatory cascade trigger"},
        "senescence_receptor": {"name": "ABCA1", "positions": [520, 525], "affinity": -6.8,
                                 "role": "cholesterol efflux pump"},
    },
    "smooth_muscle": {
        "primary": {"name": "SMA", "positions": [265, 270, 275], "affinity": -8.2,
                     "role": "contractile phenotype marker"},
        "secondary": {"name": "SM22alpha", "positions": [380, 385], "affinity": -7.0,
                       "role": "structural rigidity marker"},
        "aging_markers": {"name": "OPN", "positions": [448, 453], "affinity": -6.8,
                           "role": "synthetic phenotype (destabilization risk)"},
        "senescence_receptor": {"name": "MMP9", "positions": [520, 525], "affinity": -6.5,
                                 "role": "WARNING: aneurysm destabilization marker"},
    },
}

VASCULAR_SPATIAL_WEIGHTS = {
    "endothelial": {"sialic_acid": 0.25, "lamr": 0.20, "charge": 0.20, "stability": 0.15, "size": 0.20},
    "macrophage_foam_cell": {"sialic_acid": 0.20, "lamr": 0.25, "charge": 0.15, "stability": 0.20, "size": 0.20},
    "smooth_muscle": {"sialic_acid": 0.20, "lamr": 0.15, "charge": 0.20, "stability": 0.25, "size": 0.20},
}

SMOOTH_MUSCLE_AFFINITY_THRESHOLD = 0.70


class SpatialTranscriptomicsIntegrator:
    def __init__(self):
        self.tissue_aging_profiles = {
            "cardiac": {"aging_index": 0.72, "senescence_cells": 0.15, "inflammation": 0.68},
            "neuronal": {"aging_index": 0.80, "senescence_cells": 0.12, "inflammation": 0.55},
            "joint_cartilage": {"aging_index": 0.85, "senescence_cells": 0.20, "inflammation": 0.78},
            "skeletal_muscle": {"aging_index": 0.65, "senescence_cells": 0.10, "inflammation": 0.50},
            "hepatic": {"aging_index": 0.60, "senescence_cells": 0.08, "inflammation": 0.45},
            "renal": {"aging_index": 0.70, "senescence_cells": 0.14, "inflammation": 0.60},
        }

    def get_aging_priority(self, tissue: str) -> float:
        profile = self.tissue_aging_profiles.get(tissue, {})
        aging = profile.get("aging_index", 0.5)
        senescence = profile.get("senescence_cells", 0.1)
        return float(np.clip(0.6 * aging + 0.4 * senescence, 0, 1))


class TropismFilter(Filter):
    name = "tropism"

    def __init__(self, target_tissues: list[str] = None, avoid_tissues: list[str] = None):
        self.target_tissues = target_tissues or ["cardiac", "neuronal", "joint_cartilage"]
        self.avoid_tissues = avoid_tissues or ["hepatic"]
        self.spatial_integrator = SpatialTranscriptomicsIntegrator()

    def score(self, candidate: AAVCandidate) -> float:
        seq = candidate.sequence

        target_scores = []
        for tissue in self.target_tissues:
            if tissue not in SPATIAL_WEIGHTS:
                continue
            weights = SPATIAL_WEIGHTS[tissue]
            score = self._compute_tissue_score(seq, tissue, weights)
            aging_bonus = self.spatial_integrator.get_aging_priority(tissue) * 0.15
            target_scores.append(score + aging_bonus)

        avoid_scores = []
        for tissue in self.avoid_tissues:
            if tissue not in SPATIAL_WEIGHTS:
                continue
            weights = SPATIAL_WEIGHTS[tissue]
            score = self._compute_tissue_score(seq, tissue, weights)
            avoid_scores.append(score)

        avg_target = np.mean(target_scores) if target_scores else 0.5
        avg_avoid = np.mean(avoid_scores) if avoid_scores else 0.5

        specificity = avg_target * (1.0 - 0.7 * avg_avoid)
        return float(np.clip(specificity, 0, 1))

    def _compute_tissue_score(self, seq: str, tissue: str, weights: dict) -> float:
        scores = {}
        scores["sialic_acid"] = self._score_sialic_acid_binding(seq, tissue)
        scores["lamr"] = self._score_lamr_binding(seq, tissue)
        scores["charge"] = self._score_positive_charge(seq)
        scores["stability"] = self._score_capsid_stability(seq)
        scores["size"] = self._score_size_complementarity(seq)

        total = 0.0
        for feature, weight in weights.items():
            total += weight * scores.get(feature, 0.5)
        return total

    def _score_sialic_acid_binding(self, seq: str, tissue: str) -> float:
        receptors = AGING_SPECIFIC_RECEPTORS.get(tissue, {})
        primary = receptors.get("primary", {})
        positions = primary.get("positions", [265, 270, 275])

        score = 0.0
        count = 0
        for pos in positions:
            idx = pos - 263
            if 0 <= idx < len(seq):
                aa = seq[idx]
                if aa in ["R", "K", "H", "Y", "W"]:
                    score += 1.0
                elif aa in ["D", "E", "N", "Q"]:
                    score += 0.3
                else:
                    score += 0.6
                count += 1
        return score / max(count, 1)

    def _score_lamr_binding(self, seq: str, tissue: str) -> float:
        receptors = AGING_SPECIFIC_RECEPTORS.get(tissue, {})
        secondary = receptors.get("secondary", {})
        positions = secondary.get("positions", [458, 463, 468])

        score = 0.0
        count = 0
        for pos in positions:
            idx = pos - 263
            if 0 <= idx < len(seq):
                aa = seq[idx]
                if aa in ["K", "R", "H", "Y"]:
                    score += 1.0
                elif aa in ["D", "E"]:
                    score += 0.2
                else:
                    score += 0.5
                count += 1
        return score / max(count, 1)

    def _score_positive_charge(self, seq: str) -> float:
        positive_charge = sum(1 for aa in seq if CHARGE_PROFILE.get(aa, 0.0) > 0)
        ratio = positive_charge / max(len(seq), 1)
        return float(np.clip(ratio * 3, 0, 1))

    def _score_capsid_stability(self, seq: str) -> float:
        hydrophobic = sum(1 for aa in seq if aa in ["A", "I", "L", "M", "F", "W", "Y", "V"])
        charged = sum(1 for aa in seq if aa in ["D", "E", "K", "R", "H"])
        balance = 1.0 - abs(hydrophobic - charged) / max(len(seq), 1)
        return float(np.clip(balance, 0, 1))

    def _score_size_complementarity(self, seq: str) -> float:
        hydrophobic_core = sum(1 for aa in seq if aa in ["A", "V", "I", "L", "M", "F", "W"])
        surface = sum(1 for aa in seq if aa in ["D", "E", "K", "R", "H", "N", "Q", "S", "T"])
        ratio = hydrophobic_core / max(surface, 1)
        return float(np.clip(1.0 - abs(ratio - 1.5) / 3, 0, 1))

    def passes(self, candidate: AAVCandidate, threshold: float) -> bool:
        return self.score(candidate) >= threshold

    def filter_aav_stream(self, candidates_iter, threshold: float, target_count: int):
        total_tested = 0
        total_passed = 0
        for batch in candidates_iter:
            passed = []
            for c in batch:
                score = self.score(c)
                if score >= threshold:
                    passed.append(c)
            total_tested += len(batch)
            total_passed += len(passed)
            if total_tested % 100000 == 0:
                logger.info("Tropism: %d/%d passed (%.4f%%)",
                            total_passed, total_tested,
                            100 * total_passed / max(total_tested, 1))
            yield passed
            if total_passed >= target_count:
                break


class VascularTropismFilter(Filter):
    """Dual-compartment vascular tropism filter for IHD therapy.

    Compartment 1 (Endothelial): Targets inflamed arterial lining for
    transient OSK reprogramming to restore vascular elasticity.
    Compartment 2 (Macrophage/Foam Cell): Targets plaque-embedded cells
    for cholesterol hydrolase delivery to physically dissolve plaques.

    Safety constraint: Penalizes candidates with high smooth muscle
    affinity to prevent aneurysm from loss of structural rigidity.
    """
    name = "vascular_tropism"

    def __init__(self, target_compartments: list[str] = None):
        self.target_compartments = target_compartments or ["endothelial", "macrophage_foam_cell"]
        self.avoid_compartments = ["smooth_muscle"]

    def score(self, candidate) -> float:
        seq = getattr(candidate, "sequence", "")
        if not seq:
            return 0.0

        target_scores = []
        for compartment in self.target_compartments:
            if compartment not in VASCULAR_SPATIAL_WEIGHTS:
                continue
            weights = VASCULAR_SPATIAL_WEIGHTS[compartment]
            score = self._compute_compartment_score(seq, compartment, weights)
            target_scores.append(score)

        avoid_scores = []
        smooth_muscle_affinity = 0.0
        for compartment in self.avoid_compartments:
            if compartment not in VASCULAR_SPATIAL_WEIGHTS:
                continue
            weights = VASCULAR_SPATIAL_WEIGHTS[compartment]
            score = self._compute_compartment_score(seq, compartment, weights)
            avoid_scores.append(score)
            smooth_muscle_affinity = score

        avg_target = np.mean(target_scores) if target_scores else 0.5
        avg_avoid = np.mean(avoid_scores) if avoid_scores else 0.5

        specificity = avg_target * (1.0 - 0.7 * avg_avoid)

        if smooth_muscle_affinity > SMOOTH_MUSCLE_AFFINITY_THRESHOLD:
            logger.debug("SMOOTH MUSCLE PENALTY: affinity=%.3f > %.2f -> score *= 0.10",
                         smooth_muscle_affinity, SMOOTH_MUSCLE_AFFINITY_THRESHOLD)
            specificity *= 0.10

        return float(np.clip(specificity, 0, 1))

    def _compute_compartment_score(self, seq: str, compartment: str, weights: dict) -> float:
        scores = {}
        scores["sialic_acid"] = self._score_vascular_binding(seq, compartment, "primary")
        scores["lamr"] = self._score_vascular_binding(seq, compartment, "secondary")
        scores["charge"] = self._score_positive_charge(seq)
        scores["stability"] = self._score_capsid_stability(seq)
        scores["size"] = self._score_size_complementarity(seq)

        total = 0.0
        for feature, weight in weights.items():
            total += weight * scores.get(feature, 0.5)
        return total

    def _score_vascular_binding(self, seq: str, compartment: str, receptor_type: str) -> float:
        receptors = VASCULAR_RECEPTORS.get(compartment, {})
        receptor = receptors.get(receptor_type, {})
        positions = receptor.get("positions", [265, 270, 275])

        score = 0.0
        count = 0
        for pos in positions:
            idx = pos - 263
            if 0 <= idx < len(seq):
                aa = seq[idx]
                if aa in ["R", "K", "H", "Y", "W"]:
                    score += 1.0
                elif aa in ["D", "E", "N", "Q"]:
                    score += 0.3
                else:
                    score += 0.6
                count += 1
        return score / max(count, 1)

    def _score_positive_charge(self, seq: str) -> float:
        positive_charge = sum(1 for aa in seq if CHARGE_PROFILE.get(aa, 0.0) > 0)
        ratio = positive_charge / max(len(seq), 1)
        return float(np.clip(ratio * 3, 0, 1))

    def _score_capsid_stability(self, seq: str) -> float:
        hydrophobic = sum(1 for aa in seq if aa in ["A", "I", "L", "M", "F", "W", "Y", "V"])
        charged = sum(1 for aa in seq if aa in ["D", "E", "K", "R", "H"])
        balance = 1.0 - abs(hydrophobic - charged) / max(len(seq), 1)
        return float(np.clip(balance, 0, 1))

    def _score_size_complementarity(self, seq: str) -> float:
        hydrophobic_core = sum(1 for aa in seq if aa in ["A", "V", "I", "L", "M", "F", "W"])
        surface = sum(1 for aa in seq if aa in ["D", "E", "K", "R", "H", "N", "Q", "S", "T"])
        ratio = hydrophobic_core / max(surface, 1)
        return float(np.clip(1.0 - abs(ratio - 1.5) / 3, 0, 1))

    def passes(self, candidate, threshold: float) -> bool:
        return self.score(candidate) >= threshold
