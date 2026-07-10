import logging
import numpy as np
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)

IONIZABLE_LIPIDS = {
    "DLin-MC3-DMA": {"tails": 18, "unsaturation": 2, "pka": 6.44},
    "SM-102": {"tails": 14, "unsaturation": 1, "pka": 6.27},
    "ALC-0315": {"tails": 18, "unsaturation": 2, "pka": 6.09},
    "DODAP": {"tails": 18, "unsaturation": 1, "pka": 6.44},
    "DLin-DMA": {"tails": 18, "unsaturation": 2, "pka": 6.84},
    "cKK-E11": {"tails": 14, "unsaturation": 1, "pka": 6.07},
}

PEG_LIPIDS = {"DMG-PEG2000": 2000, "DSPC-PEG2000": 2000, "DSPE-PEG2000": 2000}
HELPER_LIPIDS = {"DSPC": 0, "DPPC": 1, "DOPE": 2, "POPC": 3}


@dataclass
class DanonLNPCandidate:
    candidate_id: int
    ionizable_lipid: str
    peg_lipid: str
    helper_lipid: str
    ionizable_frac: float
    peg_frac: float
    helper_frac: float
    cholesterol_frac: float
    tail_length: int
    unsaturation: int
    pka: float
    cardiac_delivery_score: float = 0.0
    hepatic_avoidance_score: float = 0.0
    endosomal_escape_score: float = 0.0
    immune_activation: float = 0.0
    particle_size_nm: float = 0.0
    fitness: float = 0.0


class DanonLNPGenerator:
    def __init__(self, config):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)

    def generate_candidates(self, batch_id: int, batch_size: int) -> List[DanonLNPCandidate]:
        candidates = []
        rng = np.random.RandomState(batch_id)
        for i in range(batch_size):
            global_id = batch_id * batch_size + i
            c = self._generate_single(rng, global_id)
            candidates.append(c)
        return candidates

    def _generate_single(self, rng, global_id):
        ion_name = rng.choice(list(IONIZABLE_LIPIDS.keys()))
        ion_info = IONIZABLE_LIPIDS[ion_name]
        peg_name = rng.choice(list(PEG_LIPIDS.keys()))
        helper_name = rng.choice(list(HELPER_LIPIDS.keys()))

        ion_frac = rng.uniform(0.30, 0.50)
        peg_frac = rng.uniform(0.005, 0.025)
        cholesterol_frac = rng.uniform(0.25, 0.40)
        helper_frac = 1.0 - ion_frac - peg_frac - cholesterol_frac

        tail_length = max(8, min(24, int(ion_info["tails"] + rng.randint(-4, 5))))
        unsaturation = max(0, min(6, int(ion_info["unsaturation"] + rng.randint(-1, 2))))
        pka = np.clip(ion_info["pka"] + rng.normal(0, 0.3), 5.5, 7.0)

        return DanonLNPCandidate(
            candidate_id=global_id, ionizable_lipid=ion_name,
            peg_lipid=peg_name, helper_lipid=helper_name,
            ionizable_frac=ion_frac, peg_frac=peg_frac,
            helper_frac=helper_frac, cholesterol_frac=cholesterol_frac,
            tail_length=tail_length, unsaturation=unsaturation, pka=pka,
        )

    def score_candidates(self, candidates: List[DanonLNPCandidate]) -> List[DanonLNPCandidate]:
        for c in candidates:
            c.cardiac_delivery_score = self._cardiac_delivery(c)
            c.hepatic_avoidance_score = 1.0 - self._hepatic_affinity(c)
            c.endosomal_escape_score = self._endosomal_escape(c)
            c.particle_size_nm = self._predict_size(c)
            c.immune_activation = self._immune_activation(c)

            size_score = np.exp(-0.5 * ((c.particle_size_nm - 80) / 30) ** 2)

            c.fitness = (
                0.30 * c.cardiac_delivery_score +
                0.20 * c.hepatic_avoidance_score +
                0.20 * c.endosomal_escape_score +
                0.15 * size_score +
                0.15 * (1.0 - c.immune_activation)
            )
        return candidates

    def _cardiac_delivery(self, c):
        ion_score = np.exp(-0.5 * ((c.ionizable_frac - 0.40) / 0.08) ** 2)
        pka_score = np.exp(-0.5 * ((c.pka - 6.3) / 0.2) ** 2)
        tail_score = np.exp(-0.5 * ((c.tail_length - 16) / 4) ** 2)
        peg_penalty = 1.0 - 3.0 * c.peg_frac
        return float(np.clip(0.3 * ion_score + 0.3 * pka_score + 0.2 * tail_score + 0.2 * peg_penalty, 0, 1))

    def _hepatic_affinity(self, c):
        pka_high = np.exp(-0.5 * ((c.pka - 6.5) / 0.3) ** 2)
        peg_shielding = 1.0 - min(1.0, c.peg_frac * 40)
        return float(np.clip(0.5 * pka_high + 0.5 * peg_shielding, 0, 1))

    def _endosomal_escape(self, c):
        pka_escape = np.exp(-0.5 * ((c.pka - 6.4) / 0.25) ** 2)
        unsat_escape = np.exp(-0.5 * ((c.unsaturation - 2) / 1.5) ** 2)
        helper_escape = 1.0 if c.helper_lipid == "DOPE" else 0.6
        return float(np.clip(0.4 * pka_escape + 0.3 * unsat_escape + 0.3 * helper_escape, 0, 1))

    def _predict_size(self, c):
        base = 50.0
        size = base + c.peg_frac * 2000 + c.tail_length * 1.5 - c.cholesterol_frac * 30 - c.ionizable_frac * 20
        return float(np.clip(size, 20, 200))

    def _immune_activation(self, c):
        ion_charge = 1.0 / (1.0 + np.exp(-(c.pka - 6.35) * 10))
        return float(np.clip(0.3 * ion_charge + 0.2 * (c.ionizable_frac - 0.3), 0, 1))

    def stream_candidates(self, total: int, batch_size: int):
        num_batches = (total + batch_size - 1) // batch_size
        for batch_id in range(num_batches):
            current_batch_size = min(batch_size, total - batch_id * batch_size)
            candidates = self.generate_candidates(batch_id, current_batch_size)
            scored = self.score_candidates(candidates)
            yield scored
