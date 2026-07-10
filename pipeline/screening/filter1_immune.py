import logging
import numpy as np
import math
from pipeline.screening import Filter
from pipeline.generation.aav_generator import AAVCandidate

logger = logging.getLogger(__name__)

KNOWN_ANTIBODIES = {
    "AAV2_Ab4": {
        "paratope_residues": ["R585", "Y587", "N589"],
        "epitope_region": (570, 600),
        "binding_affinity": -10.2,
        "paratope_charges": [1.0, 0.0, 0.0],
        "paratope_hydrophobicity": [-4.5, -1.3, -3.5],
    },
    "AAV2_Ab58": {
        "paratope_residues": ["R471", "D473", "K475"],
        "epitope_region": (460, 490),
        "binding_affinity": -9.8,
        "paratope_charges": [1.0, -1.0, 1.0],
        "paratope_hydrophobicity": [-4.5, -3.5, -3.9],
    },
    "AAV8_Ab1": {
        "paratope_residues": ["Y266", "N268", "K270"],
        "epitope_region": (260, 290),
        "binding_affinity": -8.5,
        "paratope_charges": [0.0, 0.0, 1.0],
        "paratope_hydrophobicity": [-1.3, -3.5, -3.9],
    },
    "AAV9_Ab3": {
        "paratope_residues": ["K459", "R461", "E463"],
        "epitope_region": (450, 475),
        "binding_affinity": -9.1,
        "paratope_charges": [1.0, 1.0, -1.0],
        "paratope_hydrophobicity": [-3.9, -4.5, -3.5],
    },
}

CHARGE_PROFILE = {
    "A": 0.0, "C": 0.1, "D": -1.0, "E": -1.0, "F": 0.0,
    "G": 0.0, "H": 0.5, "I": 0.0, "K": 1.0, "L": 0.0,
    "M": 0.0, "N": 0.0, "P": 0.0, "Q": 0.0, "R": 1.0,
    "S": 0.0, "T": 0.0, "V": 0.0, "W": 0.0, "Y": 0.0,
}

HYDROPHOBICITY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

AA_VDW_RADIUS = {
    "A": 1.8, "C": 2.1, "D": 2.4, "E": 2.6, "F": 3.0,
    "G": 1.5, "H": 2.7, "I": 2.9, "K": 3.1, "L": 2.9,
    "M": 3.0, "N": 2.5, "P": 2.2, "Q": 2.6, "R": 3.2,
    "S": 2.0, "T": 2.2, "V": 2.7, "W": 3.4, "Y": 3.2,
}


class StructuralBindingPredictor:
    def __init__(self):
        self.k_e = 332.0
        self.distance_cutoff = 12.0

    def compute_binding_energy(self, epitope_seq: str, ab_info: dict) -> float:
        if not epitope_seq:
            return 0.0

        electrostatic = 0.0
        vdw = 0.0
        hbond = 0.0

        paratope_charges = ab_info.get("paratope_charges", [0, 0, 0])
        paratope_hydro = ab_info.get("paratope_hydrophobicity", [0, 0, 0])

        for i, aa in enumerate(epitope_seq[:len(paratope_charges)]):
            aa_charge = CHARGE_PROFILE.get(aa, 0.0)
            aa_hydro = HYDROPHOBICITY.get(aa, 0.0)
            aa_radius = AA_VDW_RADIUS.get(aa, 2.5)

            distance = 3.5 + i * 1.2
            charge_product = aa_charge * paratope_charges[min(i, len(paratope_charges)-1)]
            electrostatic += self.k_e * charge_product / max(distance, 1.0)

            hydro_product = aa_hydro * paratope_hydro[min(i, len(paratope_hydro)-1)]
            vdw += -0.5 * hydro_product / (distance ** 2)

            if aa in ["D", "E", "N", "Q", "S", "T", "H", "K", "R", "Y"]:
                hbond += -1.5

        energy = electrostatic * 0.01 + vdw + hbond * 0.1
        return energy

    def compute_steric_hindrance(self, seq: str, epitope_start: int, epitope_end: int) -> float:
        hindrance = 0.0
        for i in range(epitope_start, min(epitope_end, len(seq))):
            if i >= len(seq):
                break
            aa = seq[i]
            radius = AA_VDW_RADIUS.get(aa, 2.5)
            protrusion = max(0, radius - 2.0)
            hindrance += protrusion * 0.3

            if aa in ["W", "F", "Y", "R", "K"]:
                hindrance += 0.2

        normalized = hindrance / max(epitope_end - epitope_start, 1)
        return float(np.clip(normalized, 0, 1))

    def predict_escape_probability(self, seq: str, ab_name: str) -> float:
        if ab_name not in KNOWN_ANTIBODIES:
            return 0.5

        ab_info = KNOWN_ANTIBODIES[ab_name]
        epitope_start, epitope_end = ab_info["epitope_region"]
        epitope_start_rel = max(0, epitope_start - 263)
        epitope_end_rel = min(len(seq), epitope_end - 263)

        if epitope_start_rel >= epitope_end_rel or epitope_end_rel > len(seq):
            return 0.8

        epitope_seq = seq[epitope_start_rel:epitope_end_rel]
        binding_energy = self.compute_binding_energy(epitope_seq, ab_info)
        steric = self.compute_steric_hindrance(seq, epitope_start_rel, epitope_end_rel)

        wild_type_energy = ab_info["binding_affinity"]
        energy_escape = 1.0 - np.exp(min(binding_energy - wild_type_energy, 10))
        steric_escape = steric

        charge_mismatches = 0
        for i, aa in enumerate(epitope_seq):
            if i < len(ab_info.get("paratope_charges", [])):
                aa_charge = CHARGE_PROFILE.get(aa, 0.0)
                ab_charge = ab_info["paratope_charges"][i]
                if aa_charge * ab_charge < 0:
                    charge_mismatches += 1
        charge_escape = charge_mismatches / max(len(epitope_seq), 1)

        return float(np.clip(
            0.40 * energy_escape + 0.30 * steric_escape + 0.30 * charge_escape, 0, 1))


class ImmuneEvasionFilter(Filter):
    name = "immune_evasion"

    def __init__(self, antibody_panel: list[str] = None):
        self.antibody_panel = antibody_panel or list(KNOWN_ANTIBODIES.keys())
        self.binding_predictor = StructuralBindingPredictor()

    def score(self, candidate: AAVCandidate) -> float:
        seq = candidate.sequence
        escape_scores = []

        for ab_name in self.antibody_panel:
            escape = self.binding_predictor.predict_escape_probability(seq, ab_name)
            escape_scores.append(escape)

        avg_escape = np.mean(escape_scores) if escape_scores else 0.5

        charge_score = self._compute_surface_charge_avoidance(seq)
        epitope_masking = self._compute_epitope_masking_score(seq)
        glycan_shielding = self._compute_glycan_shielding(seq)

        final = (0.35 * avg_escape + 0.20 * charge_score +
                 0.25 * epitope_masking + 0.20 * glycan_shielding)
        return float(np.clip(final, 0, 1))

    def _compute_surface_charge_avoidance(self, seq: str) -> float:
        net_charge = sum(CHARGE_PROFILE.get(aa, 0.0) for aa in seq)
        avg_charge = net_charge / max(len(seq), 1)
        return float(np.exp(-2.0 * abs(avg_charge)))

    def _compute_epitope_masking_score(self, seq: str) -> float:
        glycan_positions = []
        for i, aa in enumerate(seq):
            if aa in ["N", "S", "T"] and 0 < i < len(seq) - 1:
                if seq[i + 1] != "P":
                    glycan_positions.append(i)

        mask_coverage = len(glycan_positions) / max(len(seq), 1)
        return float(np.clip(mask_coverage * 5, 0, 1))

    def _compute_glycan_shielding(self, seq: str) -> float:
        nx_sites = 0
        for i in range(len(seq) - 1):
            if seq[i] == "N" and seq[i+1] != "P":
                nx_sites += 1

        density = nx_sites / max(len(seq), 1)
        return float(np.clip(density * 15, 0, 1))

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
                    c.surface_score = score
                    passed.append(c)
            total_tested += len(batch)
            total_passed += len(passed)
            if total_tested % 100000 == 0:
                logger.info("Immune evasion: %d/%d passed (%.4f%%)",
                            total_passed, total_tested,
                            100 * total_passed / max(total_tested, 1))
            yield passed
            if total_passed >= target_count:
                break
