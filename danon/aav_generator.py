import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

WILD_TYPE_AAV9_CAPSID = (
    "ADLGNSSGVNYYNLKGSTLPNRLSFPGAITYHTYNQESQVPEN"
    "YIAPKLYDLPSMFAPATVKAPLNIQKRTQYTLTHSGSNPTTAG"
    "HPITNFYVPVTGTTLTTNISLPQYVNVPVVYKMQTTKYEDGVL"
    "PVRGSIMQTYQVSSYSTNWQIQVTLQFNTTSEVQPVFEVVYTR"
    "QVQGRVILPDVDKNITQLIHCINEMINTFNYNKLIVTPPMQLNN"
    "YTYWHQLQPEQNFQVKTTTTSVNVNFTITGQVPAQFVVTRNVNT"
    "MVTMKMQTTASSGSTARSFEKVRQYHTDKSGTLPRYVLQISSV"
    "NTYGTQTRVIESLKENAQFGQVGAITYTDIENTLQVHTANQVLK"
    "NTTIYAGTNLHTYIQENLSPASQSVATAFITKYVSKRVKAEGES"
    "SITYLWEILNNKMDQIRVQVNGVQVNINTTVQAVTALMINTIYV"
    "QTNITTITLQEKNITLSVTKLNEQVNATVQIHTISGSIIGPGQN"
    "NAVTKLQVTAGATANITVQNVTLDNQVTQRVKVSYVNAGGTNTT"
    "TFTLKVLPDKVINTYRGTHATRYSNFSLKIGSSN"
)

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"

CAPSID_STRUCTURAL_INTERFACES = {
    "VP1_VP2_interface": list(range(263, 340)),
    "VP1_VP3_interface": list(range(340, 450)),
    "spike_region_VR_VIII": list(range(570, 600)),
    "spike_region_VR_IX": list(range(450, 485)),
    "receptor_binding_patch": list(range(263, 290)),
    "epitope_hotspot_1": list(range(450, 475)),
    "epitope_hotspot_2": list(range(570, 590)),
}

AA_RADIUS = {
    "A": 1.8, "C": 2.1, "D": 2.4, "E": 2.6, "F": 3.0,
    "G": 1.5, "H": 2.7, "I": 2.9, "K": 3.1, "L": 2.9,
    "M": 3.0, "N": 2.5, "P": 2.2, "Q": 2.6, "R": 3.2,
    "S": 2.0, "T": 2.2, "V": 2.7, "W": 3.4, "Y": 3.2,
}

HYDROPHOBICITY_KYTE = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
    "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
    "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
    "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}


@dataclass
class DanonAAVCandidate:
    candidate_id: int
    sequence: str
    mutations: List[tuple]
    esm_score: float = 0.0
    stability_score: float = 0.0
    surface_score: float = 0.0
    structural_score: float = 0.0
    interface_integrity: float = 0.0
    packing_density: float = 0.0
    fitness: float = 0.0

    cardiac_tropism_score: float = 0.0
    skeletal_muscle_score: float = 0.0
    hepatic_avoidance_score: float = 0.0
    lamp2b_compatibility: float = 0.0
    immune_evasion_score: float = 0.0


class DanonAAVGenerator:
    def __init__(self, config):
        self.config = config
        self.max_seq_len = config.max_seq_len
        self.rng = np.random.RandomState(config.random_seed)

    def generate_candidates(self, batch_id: int, batch_size: int) -> List[DanonAAVCandidate]:
        candidates = []
        rng = np.random.RandomState(batch_id)
        for i in range(batch_size):
            global_id = batch_id * batch_size + i
            seq, mutations = self._mutate_capsid(rng)
            candidate = DanonAAVCandidate(
                candidate_id=global_id, sequence=seq, mutations=mutations
            )
            candidates.append(candidate)
        return candidates

    def _mutate_capsid(self, rng):
        seq_list = list(WILD_TYPE_AAV9_CAPSID)
        mutations = []
        num_mutations = rng.poisson(5)
        positions = list(range(len(seq_list)))
        target_positions = rng.choice(
            positions, size=min(num_mutations, len(positions)), replace=False
        )
        for pos in target_positions:
            original = seq_list[pos]
            possible = [aa for aa in AA_VOCAB if aa != original]
            new_aa = rng.choice(possible)
            seq_list[pos] = new_aa
            mutations.append((pos, original, new_aa))
        return "".join(seq_list), mutations

    def score_candidates(self, candidates: List[DanonAAVCandidate]) -> List[DanonAAVCandidate]:
        for c in candidates:
            c.structural_score = self._compute_structural(c.sequence, c.mutations)
            c.cardiac_tropism_score = self._compute_cardiac_tropism(c.sequence)
            c.skeletal_muscle_score = self._compute_skeletal_muscle(c.sequence)
            c.hepatic_avoidance_score = 1.0 - self._compute_hepatic_affinity(c.sequence)
            c.immune_evasion_score = self._compute_immune_evasion(c.sequence)
            c.lamp2b_compatibility = self._compute_lamp2b_compat(c.sequence)
            c.stability_score = self._compute_stability(c.sequence)

            c.fitness = (
                0.30 * c.cardiac_tropism_score +
                0.15 * c.skeletal_muscle_score +
                0.20 * c.hepatic_avoidance_score +
                0.15 * c.immune_evasion_score +
                0.10 * c.lamp2b_compatibility +
                0.10 * c.structural_score +
                0.05 * c.stability_score
            )
        return candidates

    def _compute_cardiac_tropism(self, seq: str) -> float:
        positions = [265, 270, 275, 458, 463]
        score = 0.0
        count = 0
        for pos in positions:
            idx = pos - 263
            if 0 <= idx < len(seq):
                aa = seq[idx]
                if aa in ["R", "K", "H", "Y", "W"]:
                    score += 1.0
                elif aa in ["D", "E"]:
                    score += 0.3
                else:
                    score += 0.6
                count += 1
        positive_ratio = sum(1 for aa in seq if aa in ["R", "K", "H"]) / max(len(seq), 1)
        charge_bonus = min(positive_ratio * 5, 0.3)
        return float(np.clip(score / max(count, 1) + charge_bonus, 0, 1))

    def _compute_skeletal_muscle(self, seq: str) -> float:
        positions = [265, 270, 380, 385]
        score = 0.0
        count = 0
        for pos in positions:
            idx = pos - 263
            if 0 <= idx < len(seq):
                aa = seq[idx]
                if aa in ["R", "K", "H", "Y"]:
                    score += 0.9
                elif aa in ["D", "E"]:
                    score += 0.2
                else:
                    score += 0.5
                count += 1
        return float(np.clip(score / max(count, 1), 0, 1))

    def _compute_hepatic_affinity(self, seq: str) -> float:
        positions = [265, 270, 275, 458, 463]
        score = 0.0
        count = 0
        for pos in positions:
            idx = pos - 263
            if 0 <= idx < len(seq):
                aa = seq[idx]
                if aa in ["R", "K", "Y", "W"]:
                    score += 0.8
                elif aa in ["D", "E", "N", "Q"]:
                    score += 0.5
                else:
                    score += 0.3
                count += 1
        return float(np.clip(score / max(count, 1), 0, 1))

    def _compute_immune_evasion(self, seq: str) -> float:
        epitope_positions = list(range(450, 475)) + list(range(570, 590))
        masked = 0
        total = 0
        for pos in epitope_positions:
            if pos < len(seq):
                aa = seq[pos]
                if aa in ["D", "E", "K", "R", "N", "Q", "S", "T", "P"]:
                    masked += 1
                total += 1
        return float(np.clip(masked / max(total, 1), 0, 1))

    def _compute_lamp2b_compat(self, seq: str) -> float:
        aa_composition = {}
        for aa in seq:
            aa_composition[aa] = aa_composition.get(aa, 0) + 1
        hydrophobic_ratio = sum(aa_composition.get(aa, 0) for aa in ["A", "I", "L", "M", "F", "W", "V"]) / max(len(seq), 1)
        score = np.exp(-0.5 * ((hydrophobic_ratio - 0.45) / 0.1) ** 2)
        return float(np.clip(score, 0, 1))

    def _compute_stability(self, seq: str) -> float:
        hydrophobic = sum(1 for aa in seq if aa in ["A", "I", "L", "M", "F", "W", "Y", "V"])
        charged = sum(1 for aa in seq if aa in ["D", "E", "K", "R", "H"])
        balance = 1.0 - abs(hydrophobic - charged) / max(len(seq), 1)
        return float(np.clip(balance, 0, 1))

    def _compute_structural(self, seq: str, mutations: list) -> float:
        interface_mutations = 0
        for pos, orig, new in mutations:
            for positions in CAPSID_STRUCTURAL_INTERFACES.values():
                if pos in positions:
                    radius_penalty = abs(AA_RADIUS.get(new, 2.0) - AA_RADIUS.get(orig, 2.0))
                    interface_mutations += radius_penalty * 0.4
        penalty = interface_mutations / max(len(CAPSID_STRUCTURAL_INTERFACES), 1)
        return float(np.clip(1.0 - penalty, 0, 1))

    def stream_candidates(self, total: int, batch_size: int):
        num_batches = (total + batch_size - 1) // batch_size
        for batch_id in range(num_batches):
            current_batch_size = min(batch_size, total - batch_id * batch_size)
            candidates = self.generate_candidates(batch_id, current_batch_size)
            scored = self.score_candidates(candidates)
            yield scored
