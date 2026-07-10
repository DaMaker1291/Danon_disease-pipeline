import logging
import numpy as np
from pipeline.screening import Filter
from pipeline.generation.aav_generator import AAVCandidate
from pipeline.generation.lnp_generator import LNPCandidate

logger = logging.getLogger(__name__)

CODON_OPTIMIZATION_TABLE = {
    "A": {"optimized": 6.3, "frequency": 0.078},
    "C": {"optimized": 5.2, "frequency": 0.020},
    "D": {"optimized": 5.9, "frequency": 0.049},
    "E": {"optimized": 6.3, "frequency": 0.062},
    "F": {"optimized": 4.6, "frequency": 0.037},
    "G": {"optimized": 5.4, "frequency": 0.067},
    "H": {"optimized": 4.2, "frequency": 0.023},
    "I": {"optimized": 5.9, "frequency": 0.054},
    "K": {"optimized": 5.8, "frequency": 0.063},
    "L": {"optimized": 7.0, "frequency": 0.094},
    "M": {"optimized": 2.2, "frequency": 0.024},
    "N": {"optimized": 4.6, "frequency": 0.039},
    "P": {"optimized": 5.7, "frequency": 0.048},
    "Q": {"optimized": 4.5, "frequency": 0.039},
    "R": {"optimized": 4.7, "frequency": 0.052},
    "S": {"optimized": 6.3, "frequency": 0.069},
    "T": {"optimized": 5.5, "frequency": 0.055},
    "V": {"optimized": 5.9, "frequency": 0.066},
    "W": {"optimized": 1.3, "frequency": 0.013},
    "Y": {"optimized": 2.8, "frequency": 0.029},
}

NLS_SEQUENCES = {
    "nls_svk": "PKKKRKV",
    "nls_nls4": "KKKL",
    "nls_bipartite": "KRKKFNQKK",
    "nls_classical": "KKKR",
}

NUCLEAR_IMPORT_SIGNALS = ["PKKKRKV", "KKKL", "KKKR", "KKKRK", "RKKRR"]


class TransductionEfficiencyFilter(Filter):
    name = "transduction_efficiency"

    def __init__(self, min_transduction: float = 0.3, max_transduction: float = 0.7):
        self.min_transduction = min_transduction
        self.max_transduction = max_transduction

    def score(self, candidate: AAVCandidate) -> float:
        seq = candidate.sequence

        nls_score = self._score_nuclear_localization(seq)
        codon_score = self._score_codon_optimization(seq)
        stability_score = self._score_expression_stability(seq)
        promoter_compatibility = self._score_promoter_compatibility(seq)
        nuclear_escape = self._score_nuclear_escape(seq)

        final = (
            0.25 * nls_score +
            0.20 * codon_score +
            0.20 * stability_score +
            0.20 * promoter_compatibility +
            0.15 * nuclear_escape
        )
        return float(np.clip(final, 0, 1))

    def _score_nuclear_localization(self, seq: str) -> float:
        best_score = 0.0
        for nls_name, nls_seq in NLS_SEQUENCES.items():
            if nls_seq in seq:
                best_score = max(best_score, 1.0)
            else:
                k_r_count = sum(1 for aa in seq if aa in ["K", "R"])
                k_r_ratio = k_r_count / max(len(seq), 1)
                partial_score = min(1.0, k_r_ratio * 10)
                best_score = max(best_score, partial_score * 0.7)

        return best_score

    def _score_codon_optimization(self, seq: str) -> float:
        total_score = 0.0
        for aa in seq:
            info = CODON_OPTIMIZATION_TABLE.get(aa, {"optimized": 4.0})
            normalized = info["optimized"] / 7.0
            total_score += normalized
        avg_score = total_score / max(len(seq), 1)
        return float(np.clip(avg_score, 0, 1))

    def _score_expression_stability(self, seq: str) -> float:
        gc_content = sum(1 for aa in seq if aa in ["A", "T", "G", "C"]) / max(len(seq), 1)
        gc_score = 1.0 - abs(gc_content - 0.5) / 0.5

        rare_codons = sum(1 for aa in seq if aa in ["C", "G", "A"])
        rare_ratio = rare_codons / max(len(seq), 1)
        rare_score = 1.0 - rare_ratio

        stability = 0.5 * gc_score + 0.5 * rare_score
        return float(np.clip(stability, 0, 1))

    def _score_promoter_compatibility(self, seq: str) -> float:
        cpg_islands = self._detect_cpg_islands(seq)
        tts_signal = self._detect_poly_a_signal(seq)
        splice_sites = self._detect_splice_sites(seq)

        compat = 0.4 * (1.0 - cpg_islands) + 0.3 * tts_signal + 0.3 * (1.0 - splice_sites)
        return float(np.clip(compat, 0, 1))

    def _detect_cpg_islands(self, seq: str) -> float:
        window = 20
        cpg_counts = []
        for i in range(0, len(seq) - window, window // 2):
            chunk = seq[i:i + window]
            cg = sum(1 for j in range(len(chunk) - 1) if chunk[j:j+2] in ["CG", "GC"])
            cpg_counts.append(cg / max(len(chunk), 1))
        avg_cpg = np.mean(cpg_counts) if cpg_counts else 0
        return float(np.clip(avg_cpg * 5, 0, 1))

    def _detect_poly_a_signal(self, seq: str) -> float:
        a_count = sum(1 for aa in seq if aa == "A")
        a_ratio = a_count / max(len(seq), 1)
        return float(np.clip(a_ratio * 3, 0, 1))

    def _detect_splice_sites(self, seq: str) -> float:
        splice_motifs = ["GT", "AG", "GG", "AT"]
        count = 0
        for i in range(len(seq) - 1):
            if seq[i:i+2] in splice_motifs:
                count += 1
        ratio = count / max(len(seq), 1)
        return float(np.clip(ratio * 2, 0, 1))

    def _score_nuclear_escape(self, seq: str) -> float:
        nls_signals = sum(
            1 for signal in NUCLEAR_IMPORT_SIGNALS if signal in seq
        )
        basic_residues = sum(1 for aa in seq if aa in ["K", "R"])
        basic_ratio = basic_residues / max(len(seq), 1)

        escape = min(1.0, nls_signals * 0.3 + basic_ratio * 5)
        return float(np.clip(escape, 0, 1))

    def passes(self, candidate: AAVCandidate, threshold: float) -> bool:
        score = self.score(candidate)
        return score >= threshold

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
                logger.info(
                    "Transduction: %d/%d passed (%.4f%%)",
                    total_passed, total_tested,
                    100 * total_passed / max(total_tested, 1),
                )
            yield passed
            if total_passed >= target_count:
                break


class LNPTransductionFilter(Filter):
    name = "lnp_transduction"

    def __init__(self, min_transduction: float = 0.3, max_transduction: float = 0.7):
        self.min_transduction = min_transduction
        self.max_transduction = max_transduction

    def score(self, candidate: LNPCandidate) -> float:
        pka_score = self._score_pka(candidate.pka)
        composition_score = self._score_composition(candidate)
        peg_balance = self._score_peg_balance(candidate.peg_frac)

        final = 0.4 * pka_score + 0.3 * composition_score + 0.3 * peg_balance
        return float(np.clip(final, 0, 1))

    def _score_pka(self, pka: float) -> float:
        return float(np.exp(-0.5 * ((pka - 6.35) / 0.15) ** 2))

    def _score_composition(self, c: LNPCandidate) -> float:
        ion_score = np.exp(-0.5 * ((c.ionizable_frac - 0.40) / 0.08) ** 2)
        chol_score = np.exp(-0.5 * ((c.cholesterol_frac - 0.35) / 0.05) ** 2)
        return float(np.clip(0.5 * ion_score + 0.5 * chol_score, 0, 1))

    def _score_peg_balance(self, peg_frac: float) -> float:
        return float(np.exp(-0.5 * ((peg_frac - 0.015) / 0.005) ** 2))

    def passes(self, candidate: LNPCandidate, threshold: float) -> bool:
        return self.score(candidate) >= threshold
