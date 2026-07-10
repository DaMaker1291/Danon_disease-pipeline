import logging
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CARDIAC_RECEPTORS = {
    "integrin_alpha7_beta1": {"positions": [265, 270, 275], "affinity": -8.2,
                               "role": "cardiac myocyte homing"},
    "SIRP_alpha": {"positions": [448, 453, 458], "affinity": -7.5,
                    "role": "immune checkpoint on cardiomyocytes"},
    "ICAM1": {"positions": [380, 385, 390], "affinity": -7.8,
              "role": "inflamed cardiac endothelium"},
    "dystroglycan": {"positions": [520, 525, 530], "affinity": -8.5,
                      "role": "skeletal + cardiac muscle membrane anchor"},
}

HEPATIC_RECEPTORS = {
    "AAVR": {"positions": [265, 270, 275], "affinity": -9.5},
    "heparan_sulfate": {"positions": [458, 463], "affinity": -8.0},
    "ASGPR": {"positions": [380, 385], "affinity": -7.5},
}

SPATIAL_WEIGHTS = {
    "cardiac_myocytes": {"sialic_acid": 0.30, "lamr": 0.25, "charge": 0.15, "stability": 0.15, "size": 0.15},
    "skeletal_myocytes": {"sialic_acid": 0.25, "lamr": 0.25, "charge": 0.10, "stability": 0.20, "size": 0.20},
    "vascular_endothelium": {"sialic_acid": 0.25, "lamr": 0.20, "charge": 0.20, "stability": 0.15, "size": 0.20},
    "hepatic": {"sialic_acid": 0.35, "lamr": 0.15, "charge": 0.10, "stability": 0.20, "size": 0.20},
    "renal": {"sialic_acid": 0.30, "lamr": 0.20, "charge": 0.15, "stability": 0.20, "size": 0.15},
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

        specificity = avg_target * (1.0 - 0.7 * avg_avoid)
        return float(np.clip(specificity, 0, 1))

    def passes(self, candidate, threshold: float = None) -> bool:
        threshold = threshold or self.min_cardiac
        score = self.score(candidate)
        hepatic_score = self._compute_hepatic_score(getattr(candidate, "sequence", ""))
        return score >= threshold and hepatic_score <= self.max_hepatic

    def _compute_tissue_score(self, seq, tissue, weights):
        scores = {}
        scores["sialic_acid"] = self._score_binding(seq, tissue, "primary")
        scores["lamr"] = self._score_binding(seq, tissue, "secondary")
        scores["charge"] = self._score_positive_charge(seq)
        scores["stability"] = self._score_stability(seq)
        scores["size"] = self._score_size(seq)
        total = sum(weights.get(f, 0) * scores.get(f, 0.5) for f in weights)
        return total

    def _score_binding(self, seq, tissue, receptor_type):
        if tissue in ["cardiac_myocytes", "skeletal_myocytes", "vascular_endothelium"]:
            receptors = CARDIAC_RECEPTORS
        else:
            receptors = HEPATIC_RECEPTORS

        receptor_list = list(receptors.values())
        if receptor_type == "secondary" and len(receptor_list) > 1:
            receptor = receptor_list[1]
        else:
            receptor = receptor_list[0] if receptor_list else {}

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

    def _compute_hepatic_score(self, seq):
        if not seq:
            return 0.5
        return self._score_binding(seq, "hepatic", "primary")

    def _score_positive_charge(self, seq):
        positive = sum(1 for aa in seq if CHARGE_PROFILE.get(aa, 0.0) > 0)
        return float(np.clip(positive / max(len(seq), 1) * 3, 0, 1))

    def _score_stability(self, seq):
        hydrophobic = sum(1 for aa in seq if aa in ["A", "I", "L", "M", "F", "W", "Y", "V"])
        charged = sum(1 for aa in seq if aa in ["D", "E", "K", "R", "H"])
        return float(np.clip(1.0 - abs(hydrophobic - charged) / max(len(seq), 1), 0, 1))

    def _score_size(self, seq):
        core = sum(1 for aa in seq if aa in ["A", "V", "I", "L", "M", "F", "W"])
        surface = sum(1 for aa in seq if aa in ["D", "E", "K", "R", "H", "N", "Q", "S", "T"])
        ratio = core / max(surface, 1)
        return float(np.clip(1.0 - abs(ratio - 1.5) / 3, 0, 1))

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
