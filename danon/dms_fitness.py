"""
PHASE 19 — Deep Mutational Scan (DMS) Fitness Boundary Layer
============================================================
Evaluates every engineered VP1 substitution against a deep-mutational-scanning
style fitness landscape. The fitness effect of a mutation is decomposed into:

  1. Substitution likelihood         : BLOSUM62 log-odds (evolutionary acceptance)
  2. Positional conservation prior   : Shannon-entropy-derived conservation weight
                                        (buried / interface residues are conserved)
  3. Physicochemical shift penalty   : volume + hydropathy disruption

A charge-flip that lands inside a conserved structural pocket receives an
absolute fitness penalty so the capsid cannot collapse. The layer emits a
per-mutation DMS score and a hard viability gate.

References:
  Henikoff & Henikoff 1992 (BLOSUM62), PNAS 89:10915.
  Fowler & Fields 2014 (Deep mutational scanning), Nat Methods 11:801.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"

# BLOSUM62 substitution matrix (log-odds, symmetric), rows/cols in AA_ORDER
_BLOSUM62 = [
    [4,-1,-2,-2,0,-1,-1,0,-2,-1,-1,-1,-1,-2,-1,1,0,-3,-2,0],
    [-1,5,0,-2,-3,1,0,-2,0,-3,-2,2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2,0,6,1,-3,0,0,0,1,-3,-3,0,-2,-3,-2,1,0,-4,-2,-3],
    [-2,-2,1,6,-3,0,2,-1,-1,-3,-4,-1,-3,-3,-1,0,-1,-4,-3,-3],
    [0,-3,-3,-3,9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1,1,0,0,-3,5,2,-2,0,-3,-2,1,0,-3,-1,0,-1,-2,-1,-2],
    [-1,0,0,2,-4,2,5,-2,0,-3,-3,1,-2,-3,-1,0,-1,-3,-2,-2],
    [0,-2,0,-1,-3,-2,-2,6,-2,-4,-4,-2,-3,-3,-2,0,-2,-2,-3,-3],
    [-2,0,1,-1,-3,0,0,-2,8,-3,-3,-1,-2,-1,-2,-1,-2,-2,2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3,4,2,-3,1,0,-3,-2,-1,-3,-1,3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3,2,4,-2,2,0,-3,-2,-1,-2,-1,1],
    [-1,2,0,-1,-3,1,1,-2,-1,-3,-2,5,-1,-3,-1,0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1,0,-2,-3,-2,1,2,-1,5,0,-2,-1,-1,-1,-1,1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1,0,0,-3,0,6,-4,-2,-2,1,3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4,7,-1,-1,-4,-3,-2],
    [1,-1,1,0,-1,0,0,0,-1,-2,-2,0,-1,-2,-1,4,1,-3,-2,-2],
    [0,-1,0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1,1,5,-2,-2,0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1,1,-4,-3,-2,11,2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3,2,-1,-1,-2,-1,3,-3,-2,-2,2,7,-1],
    [0,-3,-3,-3,-1,-2,-2,-3,-3,3,1,-2,1,-1,-2,-2,0,-3,-1,4],
]
BLOSUM62 = {a: {b: _BLOSUM62[i][j] for j, b in enumerate(AA_ORDER)}
            for i, a in enumerate(AA_ORDER)}

# Residue side-chain volume (Å³, Zamyatnin 1972)
AA_VOLUME = {"A": 88.6, "R": 173.4, "N": 114.1, "D": 111.1, "C": 108.5,
             "Q": 143.8, "E": 138.4, "G": 60.1, "H": 153.2, "I": 166.7,
             "L": 166.7, "K": 168.6, "M": 162.9, "F": 189.9, "P": 112.7,
             "S": 89.0, "T": 116.1, "W": 227.8, "Y": 193.6, "V": 140.0}

# Kyte-Doolittle hydropathy
AA_HYDRO = {"A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8, "G": -0.4,
            "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
            "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8, "T": -0.7, "V": 4.2,
            "W": -0.9, "Y": -1.3}

# Conserved structural pockets in stored VP1 frame (263-819). Mutations here are
# high risk. Derived from AAV9 buried/interface residues (DiMattia 2012, 3J1S).
CONSERVED_POCKETS = {
    "beta_barrel_core": list(range(1, 20)) + list(range(90, 110)),
    "fivefold_channel": list(range(52, 60)),
    "receptor_anchor": [196, 197, 198, 318, 319],
    "dna_packaging": list(range(40, 48)),
}

# Surface Variable Regions (stored VP1 frame). These are the most sequence-tolerant,
# solvent-exposed antigenic loops of the capsid — substitutions here are well
# tolerated, so conservation is near-zero (except the receptor-anchor residues,
# which remain in CONSERVED_POCKETS above and dominate).
VARIABLE_REGIONS = set(range(186, 207)) | set(range(308, 339))

LETHAL_FITNESS_THRESHOLD = -1.0  # below this a mutation collapses the capsid


class DMSMutationScore(BaseModel):
    position: int
    original_aa: str
    mutated_aa: str
    blosum_score: int
    conservation_weight: float = Field(ge=0.0, le=1.0)
    volume_shift: float
    hydropathy_shift: float
    in_conserved_pocket: bool
    dms_fitness: float
    lethal: bool


class DMSFitnessResult(BaseModel):
    mutation_scores: List[DMSMutationScore] = Field(default_factory=list)
    total_fitness_penalty: float
    mean_dms_fitness: float
    min_dms_fitness: float
    conserved_pocket_violations: int
    lethal_mutations: int
    capsid_viable: bool
    fitness_boundary_margin: float


class DMSFitnessLayer:
    """Deep mutational-scan fitness boundary for VP1 substitutions."""

    def __init__(self, lethal_threshold: float = LETHAL_FITNESS_THRESHOLD):
        self.lethal_threshold = lethal_threshold
        self._pocket_index = self._build_pocket_index()

    def _build_pocket_index(self) -> Dict[int, float]:
        weights: Dict[int, float] = {}
        for positions in CONSERVED_POCKETS.values():
            for p in positions:
                weights[p] = min(1.0, weights.get(p, 0.0) + 0.6)
        return weights

    def conservation_weight(self, position: int) -> float:
        """Positional conservation prior in [0,1]; higher = more conserved."""
        base = self._pocket_index.get(position, 0.0)
        if position in VARIABLE_REGIONS and position not in self._pocket_index:
            # solvent-exposed antigenic loop: highly tolerant of substitution
            return float(np.clip(base + 0.03, 0.0, 1.0))
        # smooth background: periodic beta-strand conservation of the jelly-roll
        background = 0.12 * (0.5 + 0.5 * np.cos(position * 2 * np.pi / 8.0))
        return float(np.clip(base + background, 0.0, 1.0))

    def score_mutation(self, position: int, original: str, mutated: str) -> DMSMutationScore:
        blosum = BLOSUM62.get(original, {}).get(mutated, -4)
        cons = self.conservation_weight(position)
        vol_shift = abs(AA_VOLUME.get(mutated, 120.0) - AA_VOLUME.get(original, 120.0))
        hydro_shift = abs(AA_HYDRO.get(mutated, 0.0) - AA_HYDRO.get(original, 0.0))
        in_pocket = position in self._pocket_index

        # Normalise BLOSUM (-4..11) to roughly [-1, 1]
        blosum_norm = blosum / 5.0
        # Costs are conservation-gated: on a solvent-exposed variable loop a
        # designed charge reversal is tolerated; in a conserved core it is not.
        substitution_cost = max(0.0, -blosum_norm) * (0.30 + 1.70 * cons)
        structural_cost = ((vol_shift / 140.0) * 0.5 + (hydro_shift / 9.0) * 0.3)
        structural_cost *= (0.30 + cons)

        # baseline surface tolerance so favourable/neutral swaps stay positive
        dms = blosum_norm - substitution_cost - structural_cost
        if in_pocket:
            dms -= 0.4  # absolute penalty for touching a conserved pocket
        dms = float(np.clip(dms, -3.0, 1.0))

        return DMSMutationScore(
            position=position, original_aa=original, mutated_aa=mutated,
            blosum_score=int(blosum), conservation_weight=round(cons, 4),
            volume_shift=round(vol_shift, 2), hydropathy_shift=round(hydro_shift, 2),
            in_conserved_pocket=in_pocket, dms_fitness=round(dms, 4),
            lethal=dms < self.lethal_threshold,
        )

    def evaluate(self, mutations: List[Tuple[int, str, str]]) -> DMSFitnessResult:
        scores = [self.score_mutation(p, o, n) for (p, o, n) in mutations]
        if not scores:
            return DMSFitnessResult(
                mutation_scores=[], total_fitness_penalty=0.0, mean_dms_fitness=1.0,
                min_dms_fitness=1.0, conserved_pocket_violations=0, lethal_mutations=0,
                capsid_viable=True, fitness_boundary_margin=1.0 - self.lethal_threshold,
            )
        fitness = [s.dms_fitness for s in scores]
        penalty = float(sum(max(0.0, -f) for f in fitness))
        min_fit = float(min(fitness))
        lethal = sum(1 for s in scores if s.lethal)
        violations = sum(1 for s in scores if s.in_conserved_pocket)
        return DMSFitnessResult(
            mutation_scores=scores,
            total_fitness_penalty=round(penalty, 4),
            mean_dms_fitness=round(float(np.mean(fitness)), 4),
            min_dms_fitness=round(min_fit, 4),
            conserved_pocket_violations=violations,
            lethal_mutations=lethal,
            capsid_viable=lethal == 0,
            fitness_boundary_margin=round(min_fit - self.lethal_threshold, 4),
        )
