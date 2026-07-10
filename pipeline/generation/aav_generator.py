import logging
import numpy as np
import torch
import math
from typing import Iterator
from dataclasses import dataclass, field
from esm import pretrained
from pipeline.config import GenerationConfig

logger = logging.getLogger(__name__)

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}

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

CAPSID_STRUCTURAL_INTERFACES = {
    "VP1_VP2_interface": list(range(263, 340)),
    "VP1_VP3_interface": list(range(340, 450)),
    "spike_region_VR_VIII": list(range(570, 600)),
    "spike_region_VR_IX": list(range(450, 485)),
    "receptor_binding_patch": list(range(263, 290)),
    "epitope_hotspot_1": list(range(450, 475)),
    "epitope_hotspot_2": list(range(570, 590)),
}

ICOSAHEDRAL_SYMMETRY_AXES = {
    "5_fold": [0.0, 0.0, 1.0],
    "3_fold": [0.0, 0.9428, 0.3333],
    "2_fold": [0.7071, 0.7071, 0.0],
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
class AAVCandidate:
    candidate_id: int
    sequence: str
    mutations: list[tuple[int, str, str]]
    esm_score: float = 0.0
    stability_score: float = 0.0
    surface_score: float = 0.0
    structural_score: float = 0.0
    interface_integrity: float = 0.0
    packing_density: float = 0.0
    fitness: float = 0.0


class CapsidStructuralAnalyzer:
    def __init__(self):
        self.interface_positions = CAPSID_STRUCTURAL_INTERFACES

    def compute_interface_integrity(self, seq: str, mutations: list) -> float:
        if not mutations:
            return 1.0

        interface_mutations = 0
        total_interfaces = len(self.interface_positions)

        for pos, orig, new in mutations:
            for interface_name, positions in self.interface_positions.items():
                if pos in positions:
                    radius_penalty = abs(AA_RADIUS.get(new, 2.0) - AA_RADIUS.get(orig, 2.0))
                    hydro_penalty = abs(HYDROPHOBICITY_KYTE.get(new, 0) - HYDROPHOBICITY_KYTE.get(orig, 0)) / 8.0
                    interface_mutations += radius_penalty * 0.4 + hydro_penalty * 0.6

        penalty = interface_mutations / max(total_interfaces, 1)
        return float(np.clip(1.0 - penalty, 0, 1))

    def compute_packing_density(self, seq: str) -> float:
        if not seq:
            return 0.0

        radii = [AA_RADIUS.get(aa, 2.0) for aa in seq]
        avg_radius = np.mean(radii)
        std_radius = np.std(radii)

        volume_fraction = sum((4/3) * math.pi * r**3 for r in radii) / (len(seq) * (2 * avg_radius) ** 3)
        packing_score = np.exp(-0.5 * ((volume_fraction - 0.65) / 0.1) ** 2)

        charge_balance = sum(1 for aa in seq if aa in ["D", "E"]) - sum(1 for aa in seq if aa in ["K", "R"])
        charge_ratio = abs(charge_balance) / max(len(seq), 1)
        charge_score = np.exp(-3.0 * charge_ratio)

        return float(np.clip(0.6 * packing_score + 0.4 * charge_score, 0, 1))

    def compute_structural_score(self, seq: str, mutations: list) -> float:
        interface_score = self.compute_interface_integrity(seq, mutations)
        packing_score = self.compute_packing_density(seq)

        mutation_count = len(mutations)
        sparsity_penalty = np.exp(-0.05 * mutation_count)

        hydro_profile = [HYDROPHOBICITY_KYTE.get(aa, 0) for aa in seq]
        hydro_variance = np.var(hydro_profile)
        hydro_score = np.exp(-0.1 * abs(hydro_variance - 2.0))

        return float(np.clip(
            0.35 * interface_score + 0.25 * packing_score +
            0.20 * sparsity_penalty + 0.20 * hydro_score,
            0, 1
        ))


class StructureConditionedGenerator:
    def __init__(self):
        self.aa_vocabulary = list(AA_VOCAB)

    def generate_near_interface(self, seq: str, interface_name: str, num_mutations: int,
                                rng: np.random.RandomState) -> tuple[str, list]:
        positions = CAPSID_STRUCTURAL_INTERFACES.get(interface_name, [])
        if not positions:
            return seq, []

        seq_list = list(seq)
        mutations = []

        available = [p for p in positions if p < len(seq_list)]
        target_positions = rng.choice(available, size=min(num_mutations, len(available)), replace=False)

        for pos in target_positions:
            original = seq_list[pos]
            orig_hydro = HYDROPHOBICITY_KYTE.get(original, 0)
            candidates = [aa for aa in self.aa_vocabulary if aa != original]
            candidates.sort(key=lambda aa: abs(HYDROPHOBICITY_KYTE.get(aa, 0) - orig_hydro))
            new_aa = candidates[rng.randint(0, min(5, len(candidates)))]
            seq_list[pos] = new_aa
            mutations.append((pos, original, new_aa))

        return "".join(seq_list), mutations

    def generate_surface_masked(self, seq: str, epitope_positions: list,
                                rng: np.random.RandomState) -> tuple[str, list]:
        seq_list = list(seq)
        mutations = []

        for pos in epitope_positions:
            if pos >= len(seq_list):
                continue
            original = seq_list[pos]
            candidates = [aa for aa in self.aa_vocabulary
                         if aa != original and AA_RADIUS.get(aa, 2.0) > AA_RADIUS.get(original, 2.0)]
            if candidates:
                new_aa = rng.choice(candidates)
                seq_list[pos] = new_aa
                mutations.append((pos, original, new_aa))

        return "".join(seq_list), mutations


class AAVGenerator:
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.mutation_rate = config.aav_mutation_rate
        self.variable_regions = config.aav_variable_regions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.alphabet = None
        self.structural_analyzer = CapsidStructuralAnalyzer()
        self.structure_generator = StructureConditionedGenerator()

    def load_model(self):
        logger.info("Loading ESM-2 protein language model (8M params for fast testing)...")
        self.model, self.alphabet = pretrained.esm2_t6_8M_UR50D()
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("ESM-2 loaded on %s", self.device)

    def generate_candidates(self, batch_id: int, batch_size: int) -> list[AAVCandidate]:
        candidates = []
        rng = np.random.RandomState(batch_id)

        for i in range(batch_size):
            global_id = batch_id * batch_size + i
            strategy = rng.choice(["random", "interface", "surface_masked"], p=[0.4, 0.35, 0.25])

            if strategy == "random":
                seq, mutations = self._mutate_capsid(rng)
            elif strategy == "interface":
                interface = rng.choice(list(CAPSID_STRUCTURAL_INTERFACES.keys()))
                num_muts = rng.poisson(3)
                seq, mutations = self.structure_generator.generate_near_interface(
                    WILD_TYPE_AAV9_CAPSID, interface, num_muts, rng
                )
            else:
                epitope_positions = []
                for positions in CAPSID_STRUCTURAL_INTERFACES.values():
                    subset = [p for p in positions if p < len(WILD_TYPE_AAV9_CAPSID)]
                    epitope_positions.extend(rng.choice(subset, size=min(3, len(subset)), replace=False))
                seq, mutations = self.structure_generator.generate_surface_masked(
                    WILD_TYPE_AAV9_CAPSID, epitope_positions, rng
                )

            candidate = AAVCandidate(
                candidate_id=global_id, sequence=seq, mutations=mutations
            )
            candidates.append(candidate)

        return candidates

    def _mutate_capsid(self, rng: np.random.RandomState) -> tuple[str, list]:
        seq_list = list(WILD_TYPE_AAV9_CAPSID)
        mutations = []
        num_mutations = rng.poisson(len(self.variable_regions) * self.mutation_rate)
        target_positions = rng.choice(
            self.variable_regions,
            size=min(num_mutations, len(self.variable_regions)),
            replace=False,
        )
        for pos in target_positions:
            if pos >= len(seq_list):
                continue
            original = seq_list[pos]
            possible = [aa for aa in AA_VOCAB if aa != original]
            new_aa = rng.choice(possible)
            seq_list[pos] = new_aa
            mutations.append((pos, original, new_aa))
        return "".join(seq_list), mutations

    def score_candidates(self, candidates: list[AAVCandidate]) -> list[AAVCandidate]:
        if self.model is None:
            self.load_model()

        batch_seqs = [c.sequence for c in candidates]
        all_tokens = [self.alphabet.encode(seq) for seq in batch_seqs]
        max_len = max(len(t) for t in all_tokens)
        padded = np.zeros((len(all_tokens), max_len + 2), dtype=np.int64)
        for i, tokens in enumerate(all_tokens):
            padded[i, :len(tokens)] = tokens

        token_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)

        with torch.no_grad():
            results = self.model(token_tensor, repr_layers=[6], return_contacts=False)
            logits = results["logits"]
            representations = results["representations"][6]

        for i, candidate in enumerate(candidates):
            seq_len = len(candidate.sequence)
            seq_logits = logits[i, 1:seq_len + 1]
            log_probs = torch.log_softmax(seq_logits, dim=-1)
            token_ids = token_tensor[i, 1:seq_len + 1]
            token_log_probs = log_probs[torch.arange(seq_len, device=self.device), token_ids]
            ppl = torch.exp(-token_log_probs.mean()).item()
            candidate.esm_score = -np.log(ppl)

            candidate.stability_score = self._compute_stability(representations[i])
            candidate.interface_integrity = self.structural_analyzer.compute_interface_integrity(
                candidate.sequence, candidate.mutations
            )
            candidate.packing_density = self.structural_analyzer.compute_packing_density(candidate.sequence)
            candidate.structural_score = self.structural_analyzer.compute_structural_score(
                candidate.sequence, candidate.mutations
            )

            candidate.fitness = (
                0.30 * candidate.esm_score +
                0.25 * candidate.stability_score +
                0.20 * candidate.structural_score +
                0.15 * candidate.interface_integrity +
                0.10 * candidate.packing_density
            )

        return candidates

    def _compute_stability(self, representation: torch.Tensor) -> float:
        mean_rep = representation.mean(dim=0)
        std_rep = representation.std(dim=0).mean().item()
        return float(np.clip(std_rep, 0, 1))

    def stream_candidates(self, total: int, batch_size: int) -> Iterator[list[AAVCandidate]]:
        num_batches = (total + batch_size - 1) // batch_size
        for batch_id in range(num_batches):
            current_batch_size = min(batch_size, total - batch_id * batch_size)
            candidates = self.generate_candidates(batch_id, current_batch_size)
            scored = self.score_candidates(candidates)
            yield scored
