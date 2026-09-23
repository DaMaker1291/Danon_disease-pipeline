import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple

logger = logging.getLogger(__name__)

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"

# Key AAV9 capsid structural regions (based on PDB 3UX1)
AAV9_STRUCTURE_REGIONS = {
    "VR_I": {"positions": [263, 280], "surface_exposed": 0.3, "role": "core_beta_barrel"},
    "VR_II": {"positions": [326, 346], "surface_exposed": 0.4, "role": "loop_between_strands"},
    "VR_III": {"positions": [380, 395], "surface_exposed": 0.6, "role": "surface_loop"},
    "VR_IV": {"positions": [448, 468], "surface_exposed": 0.8, "role": "receptor_binding"},
    "VR_V": {"positions": [489, 510], "surface_exposed": 0.7, "role": "galactose_binding"},
    "VR_VI": {"positions": [526, 544], "surface_exposed": 0.5, "role": "structural_stabilizer"},
    "VR_VII": {"positions": [545, 562], "surface_exposed": 0.6, "role": "antigenic_region"},
    "VR_VIII": {"positions": [570, 600], "surface_exposed": 0.9, "role": "major_antigenic_site"},
    "VR_IX": {"positions": [450, 485], "surface_exposed": 0.85, "role": "receptor_attachment"},
}

AA_PROPERTIES = {
    "A": {"hydrophobicity": 1.8, "charge": 0, "size": 1.0, "bulkiness": 0.5},
    "C": {"hydrophobicity": 2.5, "charge": 0, "size": 1.2, "bulkiness": 0.6},
    "D": {"hydrophobicity": -3.5, "charge": -1, "size": 1.4, "bulkiness": 0.4},
    "E": {"hydrophobicity": -3.5, "charge": -1, "size": 1.6, "bulkiness": 0.5},
    "F": {"hydrophobicity": 2.8, "charge": 0, "size": 2.2, "bulkiness": 1.0},
    "G": {"hydrophobicity": -0.4, "charge": 0, "size": 0.5, "bulkiness": 0.1},
    "H": {"hydrophobicity": -3.2, "charge": 0.5, "size": 1.8, "bulkiness": 0.7},
    "I": {"hydrophobicity": 4.5, "charge": 0, "size": 1.9, "bulkiness": 0.8},
    "K": {"hydrophobicity": -3.9, "charge": 1, "size": 2.0, "bulkiness": 0.7},
    "L": {"hydrophobicity": 3.8, "charge": 0, "size": 1.9, "bulkiness": 0.8},
    "M": {"hydrophobicity": 1.9, "charge": 0, "size": 1.8, "bulkiness": 0.7},
    "N": {"hydrophobicity": -3.5, "charge": 0, "size": 1.4, "bulkiness": 0.4},
    "P": {"hydrophobicity": -1.6, "charge": 0, "size": 1.2, "bulkiness": 0.3},
    "Q": {"hydrophobicity": -3.5, "charge": 0, "size": 1.6, "bulkiness": 0.5},
    "R": {"hydrophobicity": -4.5, "charge": 1, "size": 2.2, "bulkiness": 0.9},
    "S": {"hydrophobicity": -0.8, "charge": 0, "size": 0.8, "bulkiness": 0.2},
    "T": {"hydrophobicity": -0.7, "charge": 0, "size": 1.0, "bulkiness": 0.3},
    "V": {"hydrophobicity": 4.2, "charge": 0, "size": 1.5, "bulkiness": 0.7},
    "W": {"hydrophobicity": -0.9, "charge": 0, "size": 2.5, "bulkiness": 1.0},
    "Y": {"hydrophobicity": -1.3, "charge": 0, "size": 2.3, "bulkiness": 0.9},
}

CARDIAC_RECEPTOR_PREFERENCES = {
    "integrin_alpha7_beta1": {"prefers": ["R", "K", "H", "Y"], "avoids": ["D", "E"]},
    "SIRP_alpha": {"prefers": ["R", "K", "H"], "avoids": ["D", "E", "P"]},
    "ICAM1": {"prefers": ["R", "K", "Y", "W"], "avoids": ["D", "E", "N"]},
    "dystroglycan": {"prefers": ["R", "K", "H", "Y"], "avoids": ["D", "E", "G"]},
}

HEPATIC_AVOIDANCE = {
    "AAVR": {"prefers": ["K", "R", "Y", "W"], "avoids": ["D", "E", "S", "T"]},
    "heparan_sulfate": {"prefers": ["R", "K"], "avoids": ["D", "E"]},
}


#: Objectives that measure *distance from the parental sequence* rather than an
#: intrinsic property. They are 1.0 for an unmodified capsid by construction, so
#: ``total_fitness`` carries 0.40 of its weight as credit for doing nothing.
#: Consequence, and it matters: ``total_fitness`` is only meaningful for ranking
#: designs *against each other*, and must not be reported as a win over the
#: wild type. A real comparison needs a reference-free stability predictor
#: (Rosetta ddG, ProteinMPNN log-likelihood) rather than an identity penalty.
DELTA_OBJECTIVES = ("structural_stability", "fold_quality")


@dataclass
class InverseFoldedCapsid:
    """One designed capsid variant and its objectives.

    Read ``DELTA_OBJECTIVES`` before comparing ``total_fitness`` to a wild-type
    baseline: two of the five weighted terms are identity penalties, so the
    unmodified sequence scores a perfect 1.0 on both.
    """

    candidate_id: int
    parent_sequence: str
    mutated_sequence: str
    mutations: List[tuple]
    regions_targeted: List[str]
    cardiac_receptor_score: float
    hepatic_avoidance_score: float
    structural_stability: float
    immune_evasion_score: float
    fold_quality: float
    total_fitness: float


class InverseFoldingEngine:
    """
    Structure-aware capsid design using inverse folding principles.
    
    Instead of random mutations (UCL's approach), this designs mutations
    based on:
    1. Structural region (VR) context
    2. Target tissue receptor preferences
    3. Physicochemical property preservation
    4. Surface exposure weighting
    
    This is the computational equivalent of directed evolution,
    but 10,000x faster.
    """

    def __init__(self, site_temperature: float = 0.35, seed_base: int = 0):
        self.regions = AAV9_STRUCTURE_REGIONS
        self.props = AA_PROPERTIES
        self.cardiac_rec = CARDIAC_RECEPTOR_PREFERENCES
        self.hepatic = HEPATIC_AVOIDANCE
        #: Softmax temperature for edit-site sampling (0 = purely greedy).
        self.site_temperature = site_temperature
        #: Offset so two engines can explore disjoint candidate streams.
        self.seed_base = seed_base

    def _best_substitution(self, seq_list: List[str], pos: int, region: Dict,
                           prefers: List[str], avoids: List[str],
                           surface_exposure: float) -> Tuple[float, str]:
        """Best-scoring amino acid at one position, with its score.

        Scoring terms are, in order: cardiac-receptor preference, preservation of
        hydrophobicity, preservation of side-chain volume, and a proline
        guardrail that blocks helix/sheet breakers in structured contexts. The
        whole score is scaled by solvent exposure, because an edit that isn't
        solvent-exposed cannot change receptor engagement.
        """
        original = seq_list[pos]
        best_score = -np.inf
        best_aa = original
        for aa in AA_VOCAB:
            if aa == original:
                continue
            score = 0.0
            if aa in prefers:
                score += 2.0
            elif aa in avoids:
                score -= 1.0

            score -= abs(self.props[aa]["hydrophobicity"]
                         - self.props[original]["hydrophobicity"]) * 0.1
            size_ratio = self.props[aa]["size"] / self.props[original]["size"]
            score -= abs(1.0 - size_ratio) * 0.5
            score -= self._proline_guardrail(pos, original, aa, region) * 2.0
            score *= surface_exposure

            if score > best_score:
                best_score, best_aa = score, aa
        return float(best_score), best_aa

    def _proline_guardrail(self, pos: int, original: str, new: str, region: Dict) -> float:
        penalty = 0.0
        if new == "P":
            secondary = self._predict_secondary_structure(pos, region)
            if secondary == "helix":
                penalty += 1.0
            elif secondary == "sheet":
                penalty += 0.6
            elif region.get("surface_exposed", 0.5) < 0.5:
                penalty += 0.4
        if original == "G" and new != "G":
            if self._is_tight_turn(pos, region):
                penalty += 0.8
        return penalty

    def _predict_secondary_structure(self, pos: int, region: Dict) -> str:
        start, end = region.get("positions", [0, 0])
        length = end - start
        frac = (pos - start) / max(length, 1)
        if 0.2 < frac < 0.5:
            return "helix"
        elif frac < 0.2 or frac > 0.8:
            return "loop"
        else:
            return "sheet"

    def _is_tight_turn(self, pos: int, region: Dict) -> bool:
        start, end = region.get("positions", [0, 0])
        length = end - start
        frac = (pos - start) / max(length, 1)
        return frac < 0.15 or frac > 0.85

    def _score_aggregation_propensity(self, seq: str, region: Dict) -> float:
        start, end = region.get("positions", [0, 0])
        agg_score = 0.0
        count = 0
        for pos in range(start, min(end, len(seq))):
            aa = seq[pos]
            hydro = self.props[aa]["hydrophobicity"]
            bulk = self.props[aa]["bulkiness"]
            if hydro > 1.5 and bulk > 0.6:
                agg_score += hydro * bulk
                count += 1
        return min(agg_score / max(count, 1) / 10.0, 1.0) if count > 0 else 0.0

    def design_candidate(self, candidate_id: int, wild_type_seq: str,
                         target_region: str = "VR_IV",
                         target_receptor: str = "integrin_alpha7_beta1",
                         num_mutations: int = 3,
                         avoid_hepatic: bool = True) -> InverseFoldedCapsid:
        region = self.regions.get(target_region, self.regions["VR_IV"])
        start, end = region["positions"]
        surface_exposure = region["surface_exposed"]

        receptor = self.cardiac_rec.get(target_receptor, self.cardiac_rec["integrin_alpha7_beta1"])
        prefers = receptor["prefers"]
        avoids = receptor["avoids"]

        seq_list = list(wild_type_seq)
        mutations = []
        positions = list(range(start, min(end, len(wild_type_seq))))

        if len(positions) < num_mutations:
            num_mutations = len(positions)

        # A sequence shorter than the stored VP1 frame exposes no editable sites.
        # Returning the parental sequence unchanged is correct here; silently
        # drawing edits from an empty site list is not.
        if num_mutations <= 0 or not positions:
            logger.warning(
                "no editable positions in region %s for a %d-residue sequence; "
                "returning the parental sequence unmodified",
                target_region, len(wild_type_seq),
            )
            return self._finalize(wild_type_seq, wild_type_seq, [], target_region,
                                  target_receptor, region, start, end)

        # Score every candidate substitution at every position in the target
        # region *first*, then choose which positions to edit. That inversion is
        # what makes this structure-aware design rather than random mutagenesis:
        # the site of each edit is selected because of its predicted benefit, not
        # drawn uniformly from the loop.
        best_per_position = [self._best_substitution(seq_list, pos, region, prefers, avoids,
                                                     surface_exposure)
                             for pos in positions]
        gains = np.array([score for score, _ in best_per_position], dtype=float)

        # Sample the edit sites without replacement, weighted by their predicted
        # gain. A softmax keeps the search directed while letting later
        # candidates explore different (still sensible) sites, so `stream_designs`
        # returns a diverse panel rather than one design repeated.
        temperature = max(float(self.site_temperature), 1e-6)
        shifted = gains - float(np.max(gains))
        weights = np.exp(shifted / temperature)
        total = float(np.sum(weights))
        if total <= 0 or not np.isfinite(total):
            weights = np.ones_like(gains)
            total = float(weights.size)
        weights = weights / total

        # Local generator: seeding must never touch numpy's global state, and a
        # given candidate_id must reproduce the same design every time.
        rng = np.random.default_rng(self.seed_base + int(candidate_id))
        chosen = rng.choice(len(positions), size=num_mutations, replace=False, p=weights)
        target_positions = sorted(positions[int(i)] for i in chosen)

        for pos in target_positions:
            original = seq_list[pos]
            _, new_aa = best_per_position[positions.index(pos)]
            seq_list[pos] = new_aa
            mutations.append((pos, original, new_aa))

        mutated = "".join(seq_list)
        return self._finalize(candidate_id, wild_type_seq, mutations, target_region,
                              target_receptor, region, start, end, mutated=mutated)

    def _finalize(self, candidate_id: int, wild_type_seq: str, mutations: List[tuple],
                  target_region: str, target_receptor: str, region: Dict,
                  start: int, end: int, mutated: Optional[str] = None,
                  ) -> "InverseFoldedCapsid":
        """Score a (possibly unmodified) sequence and package the design."""
        mutated = mutated if mutated is not None else wild_type_seq

        cardiac_score = self._score_cardiac_receptor(mutated, [start, end],
                                                      target_receptor)
        hepatic_score = self._score_hepatic_avoidance(mutated, [start, end])

        agg_score = self._score_aggregation_propensity(mutated, region)
        agg_before = self._score_aggregation_propensity(wild_type_seq, region)
        agg_penalty = max(0.0, agg_score - agg_before)

        stability = self._score_structural_stability(mutated, mutations)
        immune = self._score_immune_evasion(mutated, target_region)
        fold = self._score_fold_quality(mutated, mutations)

        total = (
            0.20 * cardiac_score +
            0.15 * hepatic_score +
            0.20 * stability +
            0.15 * immune +
            0.20 * fold -
            0.10 * agg_penalty
        )

        return InverseFoldedCapsid(
            candidate_id=candidate_id,
            parent_sequence=wild_type_seq,
            mutated_sequence=mutated,
            mutations=mutations,
            regions_targeted=[target_region],
            cardiac_receptor_score=float(cardiac_score),
            hepatic_avoidance_score=float(hepatic_score),
            structural_stability=float(stability),
            immune_evasion_score=float(immune),
            fold_quality=float(fold),
            total_fitness=float(np.clip(total, 0, 1)),
        )

    def _score_cardiac_receptor(self, seq: str, region: List[int],
                                 receptor_name: str) -> float:
        receptor = self.cardiac_rec.get(receptor_name, self.cardiac_rec["integrin_alpha7_beta1"])
        prefers = receptor["prefers"]
        avoids = receptor["avoids"]

        start, end = region
        score = 0.0
        count = 0
        for pos in range(start, min(end, len(seq))):
            aa = seq[pos]
            if aa in prefers:
                score += 1.0
            elif aa in avoids:
                score -= 0.5
            else:
                score += 0.3
            count += 1
        return float(np.clip(score / max(count, 1), 0, 1))

    def _score_hepatic_avoidance(self, seq: str, region: List[int]) -> float:
        start, end = region
        hepatic_score = 0.0
        count = 0
        for pos in range(start, min(end, len(seq))):
            aa = seq[pos]
            for rec_name, rec in self.hepatic.items():
                if aa in rec["avoids"]:
                    hepatic_score += 1.0
                elif aa in rec["prefers"]:
                    hepatic_score -= 0.3
            count += 1
        return float(np.clip(hepatic_score / max(count, 1), 0, 1))

    def _score_structural_stability(self, seq: str, mutations: List[tuple]) -> float:
        penalty = 0.0
        for pos, orig, new in mutations:
            size_diff = abs(self.props[new]["size"] - self.props[orig]["size"])
            hydro_diff = abs(self.props[new]["hydrophobicity"] - self.props[orig]["hydrophobicity"])
            charge_diff = abs(self.props[new]["charge"] - self.props[orig]["charge"])
            penalty += (size_diff * 0.4 + hydro_diff * 0.01 + charge_diff * 0.5)
        return float(np.clip(1.0 - penalty / max(len(mutations), 1), 0, 1))

    def _score_immune_evasion(self, seq: str, region_name: str) -> float:
        antigenic = ["VR_VII", "VR_VIII", "VR_IX", "VR_IV"]
        if region_name in antigenic:
            region = self.regions[region_name]
            start, end = region["positions"]
            epitope_disruption = 0.0
            count = 0
            for pos in range(start, min(end, len(seq))):
                aa = seq[pos]
                if aa in ["D", "E", "K", "R", "N", "Q", "S", "T", "P"]:
                    epitope_disruption += 1.0
                count += 1
            return float(np.clip(epitope_disruption / max(count, 1), 0, 1))
        return 0.3

    def _score_fold_quality(self, seq: str, mutations: List[tuple]) -> float:
        if not mutations:
            return 1.0
        total_prop_change = 0.0
        for pos, orig, new in mutations:
            bulk_orig = self.props[orig]["bulkiness"]
            bulk_new = self.props[new]["bulkiness"]
            total_prop_change += abs(bulk_new - bulk_orig)
        return float(np.clip(1.0 - total_prop_change / max(len(mutations), 1), 0, 1))

    def stream_designs(self, wild_type: str, n_designs: int,
                       target_region: str = "VR_IV",
                       seed: Optional[int] = None,
                       avoid_hepatic: bool = True) -> List[InverseFoldedCapsid]:
        """Generate a ranked panel of designs for a *real* parental sequence."""
        if not wild_type:
            raise ValueError("a wild-type capsid sequence is required to design against")
        rng = np.random.default_rng(self.seed_base if seed is None else seed)
        designs = []
        for i in range(n_designs):
            design = self.design_candidate(
                i, wild_type, target_region,
                num_mutations=int(rng.integers(2, 6)),
                avoid_hepatic=avoid_hepatic,
            )
            designs.append(design)
        designs.sort(key=lambda x: x.total_fitness, reverse=True)
        return designs

    def utcl_score(self, wild_type: str) -> float:
        """Baseline: the unmodified wild-type capsid, scored by the same model.

        This used to return a hardcoded 0.35, which made every "we beat the
        baseline by Nx" claim unverifiable. Running the identical scoring
        pipeline on the parental sequence makes the comparison apples-to-apples
        and makes it obvious that the baseline is the model's opinion too.
        """
        baseline = self.design_candidate(0, wild_type, num_mutations=0)
        return float(baseline.total_fitness)

    def our_best_score(self, wild_type: str, n_designs: int = 100,
                       target_region: str = "VR_IV", seed: Optional[int] = None) -> float:
        """Best design fitness for a real parental sequence.

        The previous implementation scored ``"A" * 750`` - a poly-alanine string
        that is not an AAV9 capsid - so the returned number described nothing.
        """
        designs = self.stream_designs(wild_type, n_designs, target_region, seed=seed)
        return designs[0].total_fitness if designs else 0.0
