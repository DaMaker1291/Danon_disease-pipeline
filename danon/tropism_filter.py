import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CARDIAC_RECEPTORS = {
    "integrin_alpha7_beta1": {
        "positions": [568, 572, 576, 580, 585],
        "affinity": -9.93,
        "kd_range_nm": (50, 200),
        "role": "cardiac myocyte homing via VR-VIII",
    },
    "SIRP_alpha": {
        "positions": [490, 495, 500, 505, 510],
        "affinity": -10.24,
        "kd_range_nm": (40, 80),
        "role": "immune checkpoint on cardiomyocytes via VR-VII",
    },
    "ICAM1": {
        "positions": [445, 450, 455, 460, 465],
        "affinity": -9.68,
        "kd_range_nm": (100, 200),
        "role": "inflamed cardiac endothelium via VR-IV",
    },
    "dystroglycan": {
        "positions": [575, 580, 585, 590, 595],
        "affinity": -10.58,
        "kd_range_nm": (20, 60),
        "role": "skeletal + cardiac muscle membrane anchor via VR-IX",
    },
}

HEPATIC_RECEPTORS = {
    "AAVR": {
        "positions": [397, 402, 408, 414, 420],
        "affinity": -12.0,
        "kd_range_nm": (2, 5),
        "role": "primary AAV receptor via VR-V",
    },
    "galactose": {
        "positions": [451, 454, 457, 460, 462],
        "affinity": -10.86,
        "kd_range_nm": (10, 50),
        "role": "hepatocyte galactose recognition via VR-IV",
    },
    "heparan_sulfate": {
        "positions": [448, 451, 454, 457, 460],
        "affinity": -9.43,
        "kd_range_nm": (100, 500),
        "role": "heparan sulfate proteoglycan binding via VR-IV",
    },
}

SPATIAL_WEIGHTS = {
    "cardiac_myocytes": 0.25,
    "skeletal_myocytes": 0.20,
    "vascular_endothelium": 0.05,
    "hepatic": 0.35,
    "lung": 0.10,
    "cns": 0.05,
}

CHARGE_PROFILE = {
    "A": 0.0, "C": 0.1, "D": -1.0, "E": -1.0, "F": 0.0,
    "G": 0.0, "H": 0.5, "I": 0.0, "K": 1.0, "L": 0.0,
    "M": 0.0, "N": 0.0, "P": 0.0, "Q": 0.0, "R": 1.0,
    "S": 0.0, "T": 0.0, "V": 0.0, "W": 0.0, "Y": 0.0,
}


class DanonTropismFilter:
    def __init__(self, config):
        self.config = config
        self.target_tissues = config.target_tissues
        self.avoid_tissues = config.avoid_tissues
        self.min_cardiac = config.min_cardiac_tropism
        self.max_hepatic = config.max_hepatic_accumulation

    def score(self, candidate) -> float:
        seq = getattr(candidate, "sequence", "")
        if not seq:
            return 0.0

        target_scores = []
        for tissue in self.target_tissues:
            if tissue not in SPATIAL_WEIGHTS:
                continue
            weights = SPATIAL_WEIGHTS[tissue]
            score = self._compute_tissue_score(seq, tissue, weights)
            target_scores.append(score)

        avoid_scores = []
        for tissue in self.avoid_tissues:
            if tissue not in SPATIAL_WEIGHTS:
                continue
            weights = SPATIAL_WEIGHTS[tissue]
            score = self._compute_tissue_score(seq, tissue, weights)
            avoid_scores.append(score)

        avg_target = np.mean(target_scores) if target_scores else 0.5
        avg_avoid = np.mean(avoid_scores) if avoid_scores else 0.5

        specificity = avg_target * (1.0 - 0.65 * avg_avoid)
        return float(np.clip(specificity, 0, 1))

    def passes(self, candidate, threshold: float = None) -> bool:
        threshold = threshold or self.min_cardiac
        cardiac = getattr(candidate, "cardiac_tropism_score", self.score(candidate))
        hepatic = getattr(candidate, "hepatic_avoidance_score", 1.0 - self._compute_hepatic_score(getattr(candidate, "sequence", "")))
        return cardiac >= threshold and hepatic >= (1.0 - self.max_hepatic * 1.5)

    def _compute_tissue_score(self, seq, tissue, weight):
        charge_comp = self._score_charge_complementarity(seq, tissue)
        accessibility = self._score_surface_accessibility(seq, tissue)
        steric = self._score_steric_clash(seq, tissue)
        raw_score = 0.45 * charge_comp + 0.35 * accessibility + 0.20 * steric
        return float(np.clip(raw_score * weight, 0, 1))

    def _get_tissue_receptors(self, tissue):
        if tissue in ["cardiac_myocytes", "skeletal_myocytes", "vascular_endothelium"]:
            return CARDIAC_RECEPTORS
        return HEPATIC_RECEPTORS

    def _score_charge_complementarity(self, seq, tissue):
        receptors = self._get_tissue_receptors(tissue)
        receptor = list(receptors.values())[0] if receptors else {}
        positions = receptor.get("positions", [])
        if not positions:
            return 0.5
        favorable = 0.0
        for pos in positions:
            idx = pos - 263
            if 0 <= idx < len(seq):
                aa = seq[idx]
                charge = CHARGE_PROFILE.get(aa, 0.0)
                if tissue in ["cardiac_myocytes", "skeletal_myocytes", "vascular_endothelium"]:
                    favorable += 1.0 if charge > 0 else (0.5 if charge == 0 else 0.2)
                else:
                    favorable += 1.0 if charge < 0 else (0.5 if charge == 0 else 0.2)
        return float(np.clip(favorable / max(len(positions), 1), 0, 1))

    def _score_surface_accessibility(self, seq, tissue):
        receptors = self._get_tissue_receptors(tissue)
        receptor = list(receptors.values())[0] if receptors else {}
        positions = receptor.get("positions", [])
        if not positions:
            return 0.5
        hydrophilic = {"D", "E", "K", "N", "Q", "R", "S", "T"}
        accessible = 0.0
        for pos in positions:
            idx = pos - 263
            if 0 <= idx < len(seq):
                aa = seq[idx]
                if aa in hydrophilic:
                    accessible += 1.0
                elif aa in {"A", "G", "V", "P", "L", "I", "M"}:
                    accessible += 0.3
                else:
                    accessible += 0.6
        return float(np.clip(accessible / max(len(positions), 1), 0, 1))

    def _score_steric_clash(self, seq, tissue):
        receptors = self._get_tissue_receptors(tissue)
        receptor = list(receptors.values())[0] if receptors else {}
        positions = receptor.get("positions", [])
        if not positions:
            return 0.5
        bulky = {"F", "W", "Y", "M", "I"}
        small = {"G", "A", "S"}
        score = 0.0
        for pos in positions:
            idx = pos - 263
            if 0 <= idx < len(seq):
                aa = seq[idx]
                if aa in bulky:
                    score += 0.3
                elif aa in small:
                    score += 1.0
                else:
                    score += 0.7
        return float(np.clip(score / max(len(positions), 1), 0, 1))

    def _compute_hepatic_score(self, seq):
        if not seq:
            return 0.5
        return self._score_charge_complementarity(seq, "hepatic")

    def filter_stream(self, candidates_iter, threshold=None):
        threshold = threshold or self.min_cardiac
        total_tested = 0
        total_passed = 0
        for batch in candidates_iter:
            passed = []
            for c in batch:
                if self.passes(c, threshold):
                    passed.append(c)
            total_tested += len(batch)
            total_passed += len(passed)
            if total_tested % 10000 == 0:
                logger.info("Danon Tropism: %d/%d passed (%.2f%%)",
                            total_passed, total_tested,
                            100 * total_passed / max(total_tested, 1))
            yield passed
