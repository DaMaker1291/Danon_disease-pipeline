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
        def _score_aavr_binding(region: str) -> float:
            charge_residues = sum(1 for aa in region if aa in ["R", "K", "D", "E"])
            polar_residues = sum(1 for aa in region if aa in ["S", "T", "N", "Q", "H"])
            hydrophobic = sum(1 for aa in region if aa in ["A", "V", "I", "L", "M", "F", "W", "Y"])
            n = len(region)
            if n == 0:
                return 0.0
            charge_score = charge_residues / n
            polar_score = polar_residues / n
            hydro_score = hydrophobic / n
            return float(np.clip(0.4 * charge_score + 0.4 * polar_score + 0.2 * hydro_score, 0, 1))

        def _score_galactose_binding(region: str) -> float:
            glycan_contact = sum(1 for aa in region if aa in ["N", "S", "T", "D", "E", "Q", "R", "K"])
            polar_fraction = glycan_contact / max(len(region), 1)
            asparagine_count = region.count("N")
            asn_bonus = min(asparagine_count / max(len(region), 1), 0.3)
            return float(np.clip(0.7 * polar_fraction + 0.3 * asn_bonus, 0, 1))

        def _score_integrin_binding(region: str) -> float:
            rgd_like = 0
            for i in range(len(region) - 2):
                motif = region[i:i+3]
                if motif in ["RGD", "DGR", "RGE", "SGD", "RGD"]:
                    rgd_like += 1
            r_count = region.count("R")
            d_count = region.count("D")
            e_count = region.count("E")
            charged_fraction = (r_count + d_count + e_count) / max(len(region), 1)
            rgd_bonus = min(rgd_like * 0.3, 0.6)
            return float(np.clip(0.5 * charged_fraction + rgd_bonus, 0, 1))

        def _score_surface_charge_balance(seq: str) -> float:
            n = len(seq)
            if n == 0:
                return 0.0
            positive = sum(1 for aa in seq if aa in ["R", "K", "H"])
            negative = sum(1 for aa in seq if aa in ["D", "E"])
            balance = 1.0 - abs(positive - negative) / n
            density = (positive + negative) / n
            optimal_density = 0.25
            density_fit = 1.0 - abs(density - optimal_density) / optimal_density
            return float(np.clip(0.5 * balance + 0.5 * density_fit, 0, 1))

        aavr_region = seq[134:157] if len(seq) > 157 else seq[134:]
        galactose_region = seq[188:199] if len(seq) > 199 else seq[188:]
        integrin_region = seq[305:322] if len(seq) > 322 else seq[305:]

        aavr_score = _score_aavr_binding(aavr_region)
        galactose_score = _score_galactose_binding(galactose_region)
        integrin_score = _score_integrin_binding(integrin_region)
        charge_balance = _score_surface_charge_balance(seq)

        return float(np.clip(
            0.3 * aavr_score + 0.3 * galactose_score + 0.2 * integrin_score + 0.2 * charge_balance,
            0, 1
        ))

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
        def _score_hspg_binding(region: str) -> float:
            n = len(region)
            if n == 0:
                return 0.0
            positive_residues = sum(1 for aa in region if aa in ["R", "K"])
            basic_cluster = 0
            for i in range(len(region)):
                if region[i] in ["R", "K"]:
                    neighbors = sum(1 for j in range(max(0, i-2), min(n, i+3)) if region[j] in ["R", "K"])
                    if neighbors >= 2:
                        basic_cluster += 1
            cluster_fraction = basic_cluster / n
            charge_density = positive_residues / n
            return float(np.clip(0.6 * charge_density + 0.4 * cluster_fraction, 0, 1))

        def _score_asgpr_galactose(region: str) -> float:
            n = len(region)
            if n == 0:
                return 0.0
            glycan_residues = sum(1 for aa in region if aa in ["N", "S", "T", "D", "E"])
            asparagine = region.count("N")
            glycan_fraction = glycan_residues / n
            asn_contribution = min(asparagine / n, 0.4)
            return float(np.clip(0.5 * glycan_fraction + 0.5 * asn_contribution, 0, 1))

        hspg_region = seq[185:197] if len(seq) > 197 else seq[185:]
        asgpr_region = seq[188:199] if len(seq) > 199 else seq[188:]

        hspg_score = _score_hspg_binding(hspg_region)
        asgpr_score = _score_asgpr_galactose(asgpr_region)

        hepatic_entry = 0.6 * hspg_score + 0.4 * asgpr_score
        return float(np.clip(hepatic_entry, 0, 1))

    def _compute_immune_evasion(self, seq: str) -> float:
        epitope_vr_iv = list(range(187, 207))
        epitope_vr_viii = list(range(307, 327))
        epitope_vr_vii = list(range(282, 297))

        charged_polar = set(["D", "E", "K", "R", "N", "Q", "S", "T", "H", "P"])
        all_epitope = epitope_vr_iv + epitope_vr_viii + epitope_vr_vii

        epitope_masked = 0
        epitope_total = 0
        for pos in all_epitope:
            if pos < len(seq):
                aa = seq[pos]
                epitope_total += 1
                if aa in charged_polar:
                    epitope_masked += 1

        evasion_score = epitope_masked / max(epitope_total, 1)

        glycan_sequons = 0
        surface_positions = set(epitope_vr_iv + epitope_vr_viii + epitope_vr_vii)
        for i in range(len(seq) - 2):
            if i in surface_positions:
                motif = seq[i:i+3]
                if len(motif) == 3:
                    nxst = (motif[0] == "N" and motif[1] != "P" and motif[2] in ["S", "T"])
                    if nxst:
                        glycan_sequons += 1

        glycan_bonus = min(glycan_sequons * 0.05, 0.2)
        return float(np.clip(evasion_score + glycan_bonus, 0, 1))

    def _compute_lamp2b_compat(self, seq: str) -> float:
        def _score_tm_helix_length(region: str) -> float:
            optimal_length = 24
            actual_length = len(region)
            deviation = abs(actual_length - optimal_length)
            return float(np.clip(1.0 - deviation / optimal_length, 0, 1))

        def _score_flanking_charges(region: str) -> float:
            n = len(region)
            if n == 0:
                return 0.0
            n_term = region[:6] if n >= 6 else region
            c_term = region[-6:] if n >= 6 else region
            positive_n = sum(1 for aa in n_term if aa in ["R", "K", "H"])
            positive_c = sum(1 for aa in c_term if aa in ["R", "K", "H"])
            negative_n = sum(1 for aa in n_term if aa in ["D", "E"])
            negative_c = sum(1 for aa in c_term if aa in ["D", "E"])
            n_inside_positive = positive_n > negative_n
            c_inside_positive = positive_c > negative_c
            if n_inside_positive and c_inside_positive:
                return 1.0
            elif n_inside_positive or c_inside_positive:
                return 0.7
            else:
                return 0.3

        def _score_cargo_margin(seq: str) -> float:
            n = len(seq)
            if n == 0:
                return 0.0
            max_capacity = 4700
            current_capacity = n * 3
            payload_limit = 2200
            available = max_capacity - current_capacity
            margin = available - payload_limit
            if margin >= 500:
                return 1.0
            elif margin >= 0:
                return 0.7
            else:
                return max(0.0, 0.3 + margin / 1000)

        tm_region = seq[94:117] if len(seq) > 117 else seq[94:]

        length_score = _score_tm_helix_length(tm_region)
        charge_score = _score_flanking_charges(tm_region)
        cargo_score = _score_cargo_margin(seq)

        return float(np.clip(
            0.4 * length_score + 0.3 * charge_score + 0.3 * cargo_score,
            0, 1
        ))

    def _compute_stability(self, seq: str) -> float:
        def _score_hydrophobic_core(seq: str) -> float:
            n = len(seq)
            if n == 0:
                return 0.0
            hydrophobic_core = ["A", "V", "I", "L", "M", "F", "W", "Y"]
            hydrophobic_count = sum(1 for aa in seq if aa in hydrophobic_core)
            fraction = hydrophobic_count / n
            optimal_fraction = 0.30
            deviation = abs(fraction - optimal_fraction)
            return float(np.clip(1.0 - deviation / optimal_fraction, 0, 1))

        def _score_disulfide_potential(seq: str) -> float:
            cys_positions = [i for i, aa in enumerate(seq) if aa == "C"]
            if len(cys_positions) < 2:
                return 0.5
            compatible_pairs = 0
            for i in range(len(cys_positions)):
                for j in range(i+1, len(cys_positions)):
                    pos_i = cys_positions[i]
                    pos_j = cys_positions[j]
                    separation = abs(pos_i - pos_j)
                    if 8 <= separation <= 16:
                        compatible_pairs += 1
            pair_score = min(compatible_pairs * 0.25, 1.0)
            return float(np.clip(pair_score, 0, 1))

        def _score_proline_tolerance(seq: str) -> float:
            n = len(seq)
            if n == 0:
                return 0.0
            prolines = [i for i, aa in enumerate(seq) if aa == "P"]
            if not prolines:
                return 0.8
            loop_positions = set()
            for region in CAPSID_STRUCTURAL_INTERFACES.values():
                loop_positions.update(region)
            prolines_in_loops = sum(1 for p in prolines if p in loop_positions)
            prolines_outside = len(prolines) - prolines_in_loops
            total = len(prolines)
            if total == 0:
                return 0.8
            loop_fraction = prolines_in_loops / total
            outside_penalty = prolines_outside * 0.1
            return float(np.clip(loop_fraction - outside_penalty, 0, 1))

        def _score_charge_complementarity(seq: str) -> float:
            n = len(seq)
            if n == 0:
                return 0.0
            positive = [i for i, aa in enumerate(seq) if aa in ["R", "K", "H"]]
            negative = [i for i, aa in enumerate(seq) if aa in ["D", "E"]]
            if not positive or not negative:
                return 0.3
            salt_bridges = 0
            for p_pos in positive:
                for p_neg in negative:
                    separation = abs(p_pos - p_neg)
                    if 4 <= separation <= 12:
                        salt_bridges += 1
            bridge_score = min(salt_bridges / max(len(positive), 1), 1.0)
            return float(np.clip(bridge_score, 0, 1))

        core_score = _score_hydrophobic_core(seq)
        disulfide_score = _score_disulfide_potential(seq)
        proline_score = _score_proline_tolerance(seq)
        charge_score = _score_charge_complementarity(seq)

        return float(np.clip(
            0.3 * core_score + 0.2 * disulfide_score + 0.2 * proline_score + 0.3 * charge_score,
            0, 1
        ))

    def _compute_structural(self, seq: str, mutations: list) -> float:
        def _score_vr_loop_flexibility(seq: str) -> float:
            vr_regions = [list(range(187, 207)), list(range(282, 297)), list(range(307, 327))]
            total_gly_pro = 0
            total_residues = 0
            for region in vr_regions:
                start = min(region)
                end = max(region) + 1
                loop_seq = seq[start:end] if end <= len(seq) else seq[start:]
                for aa in loop_seq:
                    total_residues += 1
                    if aa in ["G", "P"]:
                        total_gly_pro += 1
            if total_residues == 0:
                return 0.5
            fraction = total_gly_pro / total_residues
            optimal_fraction = 0.25
            deviation = abs(fraction - optimal_fraction)
            return float(np.clip(1.0 - deviation / optimal_fraction, 0, 1))

        def _score_interface_packing(seq: str, mutations: list) -> float:
            interface_scores = []
            for interface_name, positions in CAPSID_STRUCTURAL_INTERFACES.items():
                start = min(positions)
                end = max(positions) + 1
                interface_seq = seq[start:end] if end <= len(seq) else seq[start:]
                hydrophobic_core = ["A", "V", "I", "L", "M", "F", "W"]
                hydrophobic_count = sum(1 for aa in interface_seq if aa in hydrophobic_core)
                fraction = hydrophobic_count / max(len(interface_seq), 1)
                interface_scores.append(fraction)
            avg_fraction = np.mean(interface_scores) if interface_scores else 0.0
            optimal_fraction = 0.35
            return float(np.clip(1.0 - abs(avg_fraction - optimal_fraction) / optimal_fraction, 0, 1))

        def _score_radius_of_gyration(seq: str, mutations: list) -> float:
            if not mutations:
                return 1.0
            wt_radii = [AA_RADIUS.get(orig, 2.0) for _, orig, _ in mutations]
            mut_radii = [AA_RADIUS.get(new, 2.0) for _, _, new in mutations]
            if not wt_radii:
                return 1.0
            wt_avg = np.mean(wt_radii)
            mut_avg = np.mean(mut_radii)
            deviation = abs(mut_avg - wt_avg)
            return float(np.clip(1.0 - deviation / 1.0, 0, 1))

        vr_score = _score_vr_loop_flexibility(seq)
        packing_score = _score_interface_packing(seq, mutations)
        rog_score = _score_radius_of_gyration(seq, mutations)

        return float(np.clip(
            0.3 * vr_score + 0.4 * packing_score + 0.3 * rog_score,
            0, 1
        ))

    def stream_candidates(self, total: int, batch_size: int):
        num_batches = (total + batch_size - 1) // batch_size
        for batch_id in range(num_batches):
            current_batch_size = min(batch_size, total - batch_id * batch_size)
            candidates = self.generate_candidates(batch_id, current_batch_size)
            scored = self.score_candidates(candidates)
            yield scored
