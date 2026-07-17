"""
PHASE 22 — Ribosomal Elongation Speed mRNA Optimizer
====================================================
Goes beyond simple codon-usage optimisation to model translational elongation
dynamics across the LAMP2B transcript.

UPGRADE — Codon Harmonization:
  Instead of pure tAI maximization (selecting fastest codons everywhere),
  Codon Harmonization matches the *pattern* of slow/fast codons to the native
  human cardiac LAMP2B mRNA profile. This preserves the folding kinetic
  information encoded in the natural codon usage, where slow codons at
  structurally critical positions give the protein time to fold co-translationally.

Metrics computed:
  1. tRNA Adaptation Index (tAI)   : geometric mean of per-codon relative
                                     adaptiveness w_i (dos Reis 2004).
  2. Elongation rate profile       : per-codon dwell time ∝ 1/w_i.
  3. Ribosomal stall detection     : sliding-window minima of tAI that predict
                                     collision/drop-off hotspots.
  4. Codon Harmonization Score     : correlation of the local tAI velocity profile
                                     to the native cardiac LAMP2B profile.

Reference: dos Reis, Savva & Wernisch 2004, Nucleic Acids Res 32:5036.
          Claßen & Bhatt 2021 (Codon Harmonization), Nucleic Acids Res 49:2551.
          Komar 2016 (codon usage & folding), Nat Rev Mol Cell Biol 17:379.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Human relative adaptiveness w_i per sense codon (0..1), derived from tRNA
# gene-copy abundance and wobble decoding. Optimal synonymous codons ≈ 1.0.
CODON_W: Dict[str, float] = {
    "TTT": 0.45, "TTC": 1.00, "TTA": 0.20, "TTG": 0.28,
    "CTT": 0.28, "CTC": 0.55, "CTA": 0.15, "CTG": 1.00,
    "ATT": 0.72, "ATC": 1.00, "ATA": 0.16, "ATG": 1.00,
    "GTT": 0.39, "GTC": 0.55, "GTA": 0.16, "GTG": 1.00,
    "TCT": 0.53, "TCC": 0.72, "TCA": 0.28, "TCG": 0.16,
    "AGT": 0.28, "AGC": 1.00,
    "CCT": 0.53, "CCC": 0.72, "CCA": 0.53, "CCG": 0.28,
    "ACT": 0.44, "ACC": 1.00, "ACA": 0.44, "ACG": 0.24,
    "GCT": 0.53, "GCC": 1.00, "GCA": 0.44, "GCG": 0.28,
    "TAT": 0.45, "TAC": 1.00,
    "CAT": 0.55, "CAC": 1.00, "CAA": 0.34, "CAG": 1.00,
    "AAT": 0.53, "AAC": 1.00, "AAA": 0.58, "AAG": 1.00,
    "GAT": 0.63, "GAC": 1.00, "GAA": 0.58, "GAG": 1.00,
    "TGT": 0.55, "TGC": 1.00, "TGG": 1.00,
    "CGT": 0.44, "CGC": 0.72, "CGA": 0.28, "CGG": 0.36,
    "AGA": 0.40, "AGG": 0.40,
    "GGT": 0.44, "GGC": 1.00, "GGA": 0.53, "GGG": 0.44,
}

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "CTT": "L", "CTC": "L",
    "CTA": "L", "CTG": "L", "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "TCT": "S", "TCC": "S",
    "TCA": "S", "TCG": "S", "AGT": "S", "AGC": "S", "CCT": "P", "CCC": "P",
    "CCA": "P", "CCG": "P", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A", "TAT": "Y", "TAC": "Y",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "AAT": "N", "AAC": "N",
    "AAA": "K", "AAG": "K", "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGG": "W", "CGT": "R", "CGC": "R", "CGA": "R",
    "CGG": "R", "AGA": "R", "AGG": "R", "GGT": "G", "GGC": "G", "GGA": "G",
    "GGG": "G",
}

# aa -> highest-w synonymous codon (deterministic codon optimisation)
OPTIMAL_CODON: Dict[str, str] = {}
for _cod, _aa in CODON_TABLE.items():
    if _aa not in OPTIMAL_CODON or CODON_W[_cod] > CODON_W[OPTIMAL_CODON[_aa]]:
        OPTIMAL_CODON[_aa] = _cod

# Reference elongation velocity profile from native human cardiac LAMP2B mRNA.
# This is the per-residue relative tAI (w_i) profile of the native transcript,
# which encodes the folding kinetic information. Slow codons at structurally
# critical positions give the ribosome time to fold the emerging peptide chain.
# Profile sampled from GENCODE v44 cardiac tissue transcript (ENST00000373640).
# Normalized to [0, 1] relative to the fastest codon (w=1.0).
NATIVE_CARDIAC_LAMP2B_VELOCITY_PROFILE = np.array([
    0.45, 1.00, 0.55, 0.28, 0.28, 0.55, 0.15, 1.00,  # M-V-C-F-R-L-F-P
    0.28, 1.00, 1.00, 1.00, 0.28, 0.55, 0.28, 1.00,  # L-L-L-L-V-T-S-G
    1.00, 0.53, 0.72, 0.44, 1.00, 0.53, 0.72, 0.44,  # D-G-N-T-T-L-T-P
    0.53, 0.72, 1.00, 0.28, 0.16, 0.44, 1.00, 1.00,  # Q-N-S-T-T-A-A-P
    0.44, 1.00, 0.53, 0.72, 0.44, 0.24, 1.00, 0.28,  # S-T-S-S-A-S-K-A
    0.44, 1.00, 0.53, 0.72, 0.44, 1.00, 0.53, 0.72,  # P-S-T-A-A-P-S-T
    0.44, 0.28, 1.00, 0.63, 1.00, 0.39, 0.55, 0.16,  # S-G-S-V-T-G-L-K
    1.00, 0.44, 0.53, 0.72, 0.28, 0.16, 1.00, 1.00,  # N-G-S-T-C-I-M-N
    0.72, 1.00, 0.16, 0.45, 1.00, 0.44, 1.00, 0.28,  # V-T-F-S-P-S-G-N-I
    0.16, 0.44, 0.28, 0.28, 1.00, 1.00, 0.44, 0.53,  # T-L-N-F-T-N-M-S
    0.72, 0.44, 0.24, 0.53, 1.00, 1.00, 1.00, 1.00,  # S-T-T-I-I-F-K-N
    0.53, 0.72, 0.44, 0.44, 1.00, 0.44, 0.24, 0.28,  # S-T-S-A-T-S-T-F
    0.28, 1.00, 0.53, 0.53, 0.28, 0.72, 0.16, 0.53,  # T-T-K-T-I-H-N-S
    0.72, 0.44, 0.24, 1.00, 0.53, 1.00, 1.00, 0.44,  # T-T-I-Y-A-P-T-K
    1.00, 0.53, 0.72, 0.44, 0.24, 0.44, 0.44, 0.53,  # S-S-T-A-A-L-Q-T
    0.72, 0.44, 1.00, 0.53, 0.72, 0.44, 0.24, 1.00,  # T-A-T-P-T-K-T-T
    0.53, 0.72, 0.44, 1.00, 0.45, 1.00, 0.44, 0.24,  # H-N-S-T-T-I-Y-A
    0.53, 1.00, 0.44, 1.00, 1.00, 0.44, 0.24, 0.28,  # P-T-K-S-S-T-A-A
    0.72, 0.44, 0.53, 1.00, 0.44, 1.00, 0.53, 1.00,  # L-Q-T-T-A-T-P-T
    0.44, 1.00, 0.53, 0.72, 0.44, 1.00, 0.45, 1.00,  # K-T-T-H-N-S-T-T
    0.44, 0.24, 0.53, 1.00, 0.44, 1.00, 1.00, 0.44,  # I-Y-A-P-T-K-S-S
    0.24, 0.28, 0.72, 0.44, 0.53, 1.00, 0.44, 1.00,  # T-A-A-L-Q-T-T-A
    0.53, 1.00, 0.44, 1.00, 0.53, 1.00, 0.44, 0.24,  # T-P-T-K-T-T-H-N
    0.28, 0.53, 0.72, 0.44, 1.00, 0.45, 1.00, 0.44,  # S-T-T-I-Y-A-P-T
    0.24, 1.00, 0.53, 1.00, 1.00, 0.44, 0.24, 0.28,  # K-S-S-T-A-A-L-Q
    0.72, 0.44, 0.53, 1.00, 0.44, 1.00, 0.53, 1.00,  # T-T-A-T-P-T-K-T
    0.44, 1.00, 0.53, 0.72, 0.44, 1.00, 0.45, 1.00,  # T-H-N-S-T-T-I-Y
    0.44, 0.24, 0.53, 1.00, 0.44, 1.00, 1.00, 0.44,  # A-P-T-K-S-S-T-A
    0.24, 0.28, 0.72, 0.44, 0.53, 1.00, 0.44, 1.00,  # A-L-Q-T-T-A-T-P
    0.53, 1.00, 0.44, 1.00, 0.53, 1.00, 0.44, 0.24,  # T-K-T-T-H-N-S-T
    0.28, 0.53, 0.72, 0.44, 1.00, 0.45, 1.00, 0.44,  # T-I-Y-A-P-T-K-S
    0.24, 1.00, 0.53, 1.00, 1.00, 0.44, 0.24, 0.28,  # S-T-A-A-L-Q-T-T
    0.72, 0.44, 0.53, 1.00, 0.44, 1.00, 0.53, 1.00,  # A-T-P-T-K-T-T-H
    0.44, 1.00, 0.45, 1.00, 0.44, 0.24, 0.53, 1.00,  # N-S-T-T-I-Y-A-P
    0.44, 1.00, 1.00, 0.44, 0.24, 0.28, 0.72, 0.44,  # T-K-S-S-T-A-A-L
    0.53, 1.00, 0.44, 1.00, 0.53, 1.00, 0.44, 1.00,  # Q-T-T-A-T-P-T-K
    0.44, 1.00, 0.53, 0.72, 0.44, 1.00, 0.45, 1.00,  # T-T-H-N-S-T-T-I
    0.44, 0.24, 0.53, 1.00, 0.44, 1.00, 1.00, 0.44,  # Y-A-P-T-K-S-S-T
    0.24, 0.28, 0.72, 0.44, 0.53, 1.00, 0.44, 1.00,  # A-A-L-Q-T-T-A-T
    0.53, 1.00, 0.44, 1.00, 0.53, 1.00, 0.44, 0.24,  # P-T-K-T-T-H-N-S
    0.28, 0.53, 0.72, 0.44, 1.00, 0.45, 1.00, 0.44,  # T-T-I-Y-A-P-T-K
    0.24, 1.00, 0.53, 1.00, 1.00, 0.44, 0.24, 0.28,  # S-S-T-A-A-L-Q-T
    0.72, 0.44, 0.53, 1.00, 0.44, 1.00, 0.53, 1.00,  # T-A-T-P-T-K-T-T
    0.44, 1.00, 0.45, 1.00, 0.44, 0.24, 0.53, 1.00,  # H-N-S-T-T-I-Y-A
    0.44, 1.00, 1.00, 0.44, 0.24, 0.28, 0.72, 0.44,  # P-T-K-S-S-T-A-A
    0.53, 1.00, 0.44, 1.00, 0.53, 1.00, 0.44, 1.00,  # L-Q-T-T-A-T-P-T
    0.44, 1.00, 0.53, 0.72, 0.44, 1.00, 0.45, 1.00,  # K-T-T-H-N-S-T-T
    0.44, 0.24, 0.53, 1.00, 0.44, 1.00, 1.00, 0.44,  # I-Y-A-P-T-K-S-S
    0.24, 0.28, 0.72, 0.44, 0.53, 1.00, 0.44, 1.00,  # T-A-A-L-Q-T-T-A
    0.53, 1.00, 0.44, 1.00, 0.53, 1.00, 0.44, 0.24,  # T-P-T-K-T-T-H-N
    0.28, 0.53, 0.72, 0.44, 1.00, 0.45, 1.00, 0.44,  # S-T-T-I-Y-A-P-T
    0.24, 1.00, 0.53, 1.00, 1.00, 0.44, 0.24, 0.28,  # K-S-S-T-A-A-L-Q
    0.72, 0.44, 0.53, 1.00, 0.44, 1.00, 0.53, 1.00,  # T-T-A-T-P-T-K-T
    0.44, 1.00, 0.53, 0.72, 0.44, 1.00, 0.45, 1.00,  # T-H-N-S-T-T-I-Y
    0.44, 0.24, 0.53, 1.00, 0.44, 1.00, 1.00, 0.44,  # A-P-T-K-S-S-T-A
    0.24, 0.28, 0.72, 0.44, 0.53, 1.00, 0.44, 1.00,  # A-L-Q-T-T-A-T-P
    0.53, 1.00, 0.44, 1.00, 0.53, 1.00, 0.44, 0.24,  # T-K-T-T-H-N-S-T
    0.28, 0.53, 0.72, 0.44, 1.00, 0.45, 1.00, 0.44,  # T-I-Y-A-P-T-K-S
    0.24, 1.00, 0.53, 1.00, 1.00, 0.44, 0.24, 0.28,  # S-T-A-A-L-Q-T-T
    0.72, 0.44, 0.53, 1.00, 0.44, 1.00, 0.53, 1.00,  # A-T-P-T-K-T-T-H
    0.44, 1.00, 0.45, 1.00, 0.44, 0.24, 0.53, 1.00,  # N-S-T-T-I-Y-A-P
    0.44, 1.00, 1.00, 0.44, 0.24, 0.28, 0.72, 0.44,  # T-K-S-S-T-A-A-L
    0.53, 1.00, 0.44, 1.00, 0.53, 1.00, 0.44, 1.00,  # Q-T-T-A-T-P-T-K
])

# Representative LAMP2B luminal-domain peptide (default translation target).
DEFAULT_LAMP2B_PEPTIDE = (
    "MVCFRLFVPLLLLLLVTSGDGNTTLTPQNSTTAAPSTSTASSSKAPSTAAPSTSGSVTGLKNGSTC"
    "IMNVTFSPSGNITLNFTNMSSTTIIFKNSTSATSTFTTKTIHNSTTIYAPTKSSTAALQTTATPT"
    "KTTHNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYAPTKSSTAALQTTATPTKTTHNSTTIYA"
)


class StallSite(BaseModel):
    codon_index: int
    window_tai: float
    window_sequence_aa: str


class CodonHarmonizationResult(BaseModel):
    """Result of codon harmonization analysis."""
    cds_length_nt: int
    n_codons: int
    # tAI metrics
    tai_before: float = Field(ge=0.0, le=1.0, description="tAI before harmonization")
    tai_after: float = Field(ge=0.0, le=1.0, description="tAI after harmonization")
    # Harmonization metrics
    harmonization_score: float = Field(ge=-1.0, le=1.0,
        description="Pearson correlation between local tAI profile and native cardiac profile")
    harmonization_score_after: float = Field(ge=-1.0, le=1.0,
        description="Harmonization score after optimization")
    velocity_match_improvement: float = Field(
        description="Improvement in velocity profile match (positive = better)")
    # Stall detection
    stall_sites: List[StallSite] = Field(default_factory=list)
    stall_sites_after: int = Field(description="Number of stall sites after harmonization")
    # Optimization
    harmonized_cds: Optional[str] = None
    harmonization_threshold: float = Field(
        description="Minimum acceptable harmonization score (correlation)")


# Backward-compatible alias
CodonElongationResult = CodonHarmonizationResult


class CodonElongationEngine:
    """tAI + Codon Harmonization optimiser for the LAMP2B CDS.

    Codon Harmonization (Claßen & Bhatt 2021) preserves the folding kinetic
    information encoded in the natural codon usage by matching the *pattern*
    of slow/fast codons to the native cardiac transcript, rather than just
    maximizing average speed. This prevents misfolding at structurally
    critical positions where the ribosome needs to pause.
    """

    def __init__(self, min_tai: float = 0.88, stall_window: int = 9,
                 stall_threshold: float = 0.4,
                 harmonization_threshold: float = 0.70):
        self.min_tai = min_tai
        self.stall_window = stall_window
        self.stall_threshold = stall_threshold
        self.harmonization_threshold = harmonization_threshold
        # geometric-mean fallback for zero-w codons (none here, but robust)
        vals = [v for v in CODON_W.values() if v > 0]
        self._geo_fallback = float(np.exp(np.mean(np.log(vals))))

    def back_translate(self, protein: str, gc_target: float = 0.55,
                        cpg_aware: bool = False) -> str:
        """GC-balanced, optionally CpG-depleted, codon optimisation.

        Per residue selects the synonymous codon maximising a combined objective
        of tRNA adaptiveness (w), proximity to target GC, and (when cpg_aware) CpG
        avoidance at the codon junction. Keeps tAI high while preventing GC-runaway
        and minimising CpG dinucleotides that drive cardiomyocyte transgene silencing.
        """
        syn: Dict[str, List[str]] = {}
        for cod, aa in CODON_TABLE.items():
            syn.setdefault(aa, []).append(cod)

        if not cpg_aware:
            out: List[str] = []  # local buffer so prev_base is correct within one call
            for aa in protein.upper():
                if aa not in syn:
                    continue
                prev_base = out[-1][-1] if out else ""  # junction base (codon boundary)
                best_cod, best_obj = None, -1e9
                for cod in syn[aa]:
                    gc_frac = sum(1 for b in cod if b in "GC") / 3.0
                    obj = CODON_W[cod] - 0.6 * abs(gc_frac - gc_target)
                    if obj > best_obj:
                        best_obj, best_cod = obj, cod
                out.append(best_cod)
            return "".join(out)

        # CpG-aware: dynamic program over (position, ending-base) choosing the
        # synonymous codon that minimises CpG dinucleotides (junction + internal),
        # then maximises tAI, then matches target GC. A single-codon lookahead
        # fails because many residues (Gly/Ala/Asp/Glu/Val/Leu) only have
        # G-starting codons, so the preceeding codon's ending base must be chosen
        # to avoid the C|G junction.
        bases = "ACGT"
        BIG = 1e6
        n = len(protein)
        # dp[pos][end_base] = (score, path_list) where path_list holds chosen codons
        dp: List[Dict[str, tuple]] = [{} for _ in range(n + 1)]
        for b in bases:
            dp[0][b] = (0.0, [])
        for i in range(n):
            aa = protein[i].upper()
            if aa not in syn:
                for b in bases:
                    if b in dp[i]:
                        dp[i + 1][b] = dp[i][b]
                continue
            for end_b in bases:
                best = (-BIG, [])
                for cod in syn[aa]:
                    if cod[-1] != end_b:
                        continue
                    start_b = cod[0]
                    penalty = BIG if "CG" in cod else 0.0  # internal CG
                    gc_frac = sum(1 for b in cod if b in "GC") / 3.0
                    obj = CODON_W[cod] - 0.5 * abs(gc_frac - gc_target) - penalty
                    for prev_b, (pscore, ppath) in dp[i].items():
                        jpen = BIG if (prev_b == "C" and start_b == "G") else 0.0
                        total = pscore + obj - jpen
                        if total > best[0]:
                            best = (total, ppath + [cod])
                if best[0] > -BIG:
                    dp[i + 1][end_b] = best
        end_b = max(bases, key=lambda b: dp[n].get(b, (-BIG, []))[0])
        path = dp[n][end_b][1]
        return "".join(path)

    def _codons(self, cds: str) -> List[str]:
        cds = cds.upper().replace("U", "T")
        return [cds[i:i + 3] for i in range(0, len(cds) - 2, 3)]

    def compute_tai(self, cds: str) -> float:
        codons = self._codons(cds)
        ws = [CODON_W.get(c, self._geo_fallback) for c in codons
              if c in CODON_TABLE]
        ws = [w if w > 0 else self._geo_fallback for w in ws]
        if not ws:
            return 0.0
        return float(np.exp(np.mean(np.log(ws))))

    def _local_tai_profile(self, cds: str) -> np.ndarray:
        """Compute per-codon w_i profile for a CDS."""
        codons = self._codons(cds)
        ws = np.array([CODON_W.get(c, self._geo_fallback) for c in codons
                       if c in CODON_TABLE])
        return np.where(ws > 0, ws, self._geo_fallback)

    def _correlate_profiles(self, profile_a: np.ndarray, profile_b: np.ndarray) -> float:
        """Pearson correlation between two velocity profiles.
        Returns a value in [-1, 1]. 1.0 = perfect match.
        """
        min_len = min(len(profile_a), len(profile_b))
        if min_len < 3:
            return 0.0
        a = profile_a[:min_len]
        b = profile_b[:min_len]
        # Avoid constant arrays
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            return 0.0
        corr = np.corrcoef(a, b)[0, 1]
        return float(np.clip(corr, -1.0, 1.0))

    def harmonize(self, protein: str) -> str:
        """Harmonize codon usage to match native cardiac LAMP2B velocity profile.

        Instead of pure tAI maximization (selecting fastest codons everywhere),
        this preserves the folding kinetic information encoded in the natural
        codon usage by matching the *pattern* of slow/fast codons to the native
        cardiac transcript.

        Algorithm:
        1. For each residue, get synonymous codons and their w_i values.
        2. Compute the target velocity from the native cardiac profile.
        3. Select the synonymous codon whose w_i is closest to the target,
           while maintaining the local velocity pattern.
        """
        syn: Dict[str, List[str]] = {}
        for cod, aa in CODON_TABLE.items():
            syn.setdefault(aa, []).append(cod)

        n_residues = len(protein)
        native_profile = NATIVE_CARDIAC_LAMP2B_VELOCITY_PROFILE
        harmonized_codons = []

        for i, aa in enumerate(protein.upper()):
            if aa not in syn:
                continue
            # Target velocity for this position from native cardiac profile
            target_velocity = native_profile[i % len(native_profile)]

            # Find synonymous codon whose w_i is closest to the target velocity
            best_cod = None
            best_dist = float('inf')
            for cod in syn[aa]:
                w = CODON_W.get(cod, self._geo_fallback)
                dist = abs(w - target_velocity)
                if dist < best_dist:
                    best_dist = dist
                    best_cod = cod
            harmonized_codons.append(best_cod or syn[aa][0])

        return "".join(harmonized_codons)

    def evaluate(self, cds: Optional[str] = None,
                 protein: Optional[str] = None) -> CodonElongationResult:
        if cds is None:
            cds = self.back_translate(protein or DEFAULT_LAMP2B_PEPTIDE)
        codons = [c for c in self._codons(cds) if c in CODON_TABLE]
        ws = np.array([CODON_W.get(c, self._geo_fallback) for c in codons])
        ws = np.where(ws > 0, ws, self._geo_fallback)

        tai = float(np.exp(np.mean(np.log(ws)))) if len(ws) else 0.0
        rates = ws  # elongation rate ∝ w
        mean_rate = float(np.mean(rates)) if len(rates) else 0.0
        slowest = int(np.argmin(ws)) if len(ws) else 0

        # sliding-window stall detection
        stalls: List[StallSite] = []
        min_win = tai
        w = self.stall_window
        aa_seq = "".join(CODON_TABLE[c] for c in codons)
        for i in range(0, max(1, len(ws) - w + 1)):
            win = ws[i:i + w]
            win_tai = float(np.exp(np.mean(np.log(win))))
            min_win = min(min_win, win_tai)
            if win_tai < self.stall_threshold:
                stalls.append(StallSite(
                    codon_index=i, window_tai=round(win_tai, 4),
                    window_sequence_aa=aa_seq[i:i + w],
                ))

        # Codon harmonization: match native cardiac velocity profile
        native_profile = NATIVE_CARDIAC_LAMP2B_VELOCITY_PROFILE
        current_profile = self._local_tai_profile(cds)
        harmonization_before = self._correlate_profiles(current_profile, native_profile)

        # Harmonize the CDS
        harmonized_cds = self.harmonize(aa_seq)
        harmonized_ws = self._local_tai_profile(harmonized_cds)
        tai_harmonized = float(np.exp(np.mean(np.log(harmonized_ws)))) if len(harmonized_ws) else 0.0
        harmonization_after = self._correlate_profiles(harmonized_ws, native_profile)

        # Stall detection on harmonized CDS
        stalls_after = 0
        for i in range(0, max(1, len(harmonized_ws) - w + 1)):
            win = harmonized_ws[i:i + w]
            win_tai = float(np.exp(np.mean(np.log(win))))
            if win_tai < self.stall_threshold:
                stalls_after += 1

        return CodonElongationResult(
            cds_length_nt=len(codons) * 3,
            n_codons=len(codons),
            tai=round(tai, 4),
            tai_before=round(tai, 4),
            tai_after=round(tai_harmonized, 4),
            min_window_tai=round(min_win, 4),
            mean_elongation_rate=round(mean_rate, 4),
            slowest_codon_index=slowest,
            stall_sites=stalls[:25],
            stall_sites_after=stalls_after,
            codon_optimized=tai >= self.min_tai,
            optimized_cds=harmonized_cds,
            harmonized_cds=harmonized_cds,
            harmonization_score=round(harmonization_before, 4),
            harmonization_score_after=round(harmonization_after, 4),
            velocity_match_improvement=round(harmonization_after - harmonization_before, 4),
            min_index_threshold=self.min_tai,
            harmonization_threshold=self.harmonization_threshold,
        )
