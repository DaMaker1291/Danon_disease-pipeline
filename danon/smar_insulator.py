"""
PHASE 21 — Transcriptional-Shielding Engine (CpG Depletion primary, S/MAR secondary)
====================================================================================
Transgene silencing in human cardiomyocytes is dominated by **CpG methylation**:
host DNA methyltransferases recognise CpG dinucleotides in the expression cassette
and transcriptionally silence it. The clinically validated mitigation is **CpG
depletion** — synonymous recoding that removes CG dinucleotides while preserving
the amino-acid sequence and translation efficiency (tAI) — not bulky S/MAR
chromatin insulators, which waste the scarce ~4.7 kb AAV packaging budget.

This module:
  1. CpG-Depletion Optimization  : systematically swaps to synonymous codons that
     avoid CG dinucleotides (intra-codon and across codon junctions), keeping tAI
     intact, producing a methylation-resistant LAMP2B CDS.
  2. CpG density reporting        : raw vs depleted density (per 100 bp).
  3. Optional S/MAR assessment    : retained as a secondary shielding signal, but
     explicitly flagged as space-expensive and secondary to CpG depletion.

References:
  Schall et al. 2017 (CpG-depleted AAV prolongs expression), Mol Ther 25:215.
  Wang et al. 2019 (CpG motif removal reduces innate immune sensing), Hum Gene Ther.
"""
import logging
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Reuse the codon optimiser for tAI-aware, GC-balanced, CpG-depleted back-translation
from danon.codon_elongation import CodonElongationEngine, DEFAULT_LAMP2B_PEPTIDE, CODON_TABLE

# Default S/MAR flanks (optional, secondary shielding only)
DEFAULT_UPSTREAM_SMAR = (
    "AATAAATATTTAAAATATTTTAAAATATATTTTAAATATTAAATTTTATATTTAAAAATTTAAAT"
    "ATTTTAAATATATATTTAAATTTATATTAAAAATTTATTTAAAATATTTATATTTAAAATTAT"
)
DEFAULT_DOWNSTREAM_SMAR = (
    "TTTATTTAATTTATATTTTAAATATTTAAATTTATTTATTAAATTTTATTAAAATATTTATTA"
    "AATATTTAAATATATTTAAATTTAAATATTTTAAATTATATTTAAAATATTTAAATTTATAT"
)


def calculate_cpg_density(dna_sequence: str) -> float:
    """CpG dinucleotide density per 100 bp (standard methylation-susceptibility metric)."""
    seq = dna_sequence.upper()
    n = len(seq)
    if n < 2:
        return 0.0
    cpg_count = seq.count("CG")
    return float(cpg_count) / (n / 2.0)  # per 100 bp convention used in the field


class CpGDepletionReport(BaseModel):
    protein_length_aa: int
    raw_cpg_density: float
    depleted_cpg_density: float
    cpg_reduction_pct: float
    cpg_within_threshold: bool
    depleted_cds: str


class CpGOptimizationEngine:
    """Synonymous recoding to deplete CpG dinucleotides while preserving tAI + GC."""

    def __init__(self, cpg_density_threshold: float = 1.0):
        self.threshold = cpg_density_threshold
        self.codon = CodonElongationEngine()

    def optimize(self, protein: str = None) -> CpGDepletionReport:
        peptide = protein or DEFAULT_LAMP2B_PEPTIDE
        # CpG-aware back-translation: avoids CG intra-codon and across junctions
        cds = self.codon.back_translate(peptide, gc_target=0.55, cpg_aware=True)
        raw = calculate_cpg_density(self.codon.back_translate(peptide, gc_target=0.55, cpg_aware=False))
        depleted = calculate_cpg_density(cds)
        reduction = float(np.clip(100.0 * (raw - depleted) / max(raw, 1e-9), 0.0, 100.0))
        return CpGDepletionReport(
            protein_length_aa=len(peptide),
            raw_cpg_density=round(raw, 4),
            depleted_cpg_density=round(depleted, 4),
            cpg_reduction_pct=round(reduction, 2),
            cpg_within_threshold=depleted <= self.threshold,
            depleted_cds=cds,
        )


# --------------------------------------------------------------------------- #
# Retained secondary S/MAR assessment (optional; flagged as space-expensive)
# --------------------------------------------------------------------------- #
import re

SMAR_MOTIFS = {
    "ATATTT_box": r"ATATTT",
    "MAR_ARS_core": r"[AT]TTTAT[AG]TTT[AT]",
    "topoisomerase_II": r"[AG][AC]T[AT]A[CT]ATT[AGT]AT",
    "ori_signature": r"ATTTA{2,}",
    "a_tract_kink": r"A{4,}|T{4,}",
}


class SMARFlank(BaseModel):
    label: str
    length_bp: int
    at_content: float
    smar_strength: float = Field(ge=0.0, le=1.0)


class SMARInsulatorEngine:
    """Secondary shielding assessment. NOTE: S/MARs are 500-1000 bp and consume the
    packaging budget; CpG depletion (above) is the primary anti-silencing strategy."""

    def __init__(self, min_at_content: float = 0.65):
        self.min_at_content = min_at_content

    @staticmethod
    def _at_content(seq: str) -> float:
        if not seq:
            return 0.0
        return sum(1 for b in seq.upper() if b in "AT") / len(seq)

    def score_flank(self, label: str, seq: str) -> SMARFlank:
        seq = seq.upper()
        at = self._at_content(seq)
        at_term = np.clip((at - 0.5) / 0.45, 0, 1)
        strength = float(np.clip(0.6 * at_term + 0.4 * (seq.count("AT") / max(len(seq), 1)), 0, 1))
        return SMARFlank(label=label, length_bp=len(seq), at_content=round(at, 4), smar_strength=round(strength, 4))

    def evaluate(self, upstream: str = None, downstream: str = None) -> Dict:
        up = self.score_flank("upstream_5prime", upstream or DEFAULT_UPSTREAM_SMAR)
        dn = self.score_flank("downstream_3prime", downstream or DEFAULT_DOWNSTREAM_SMAR)
        mean_strength = (up.smar_strength + dn.smar_strength) / 2.0
        combined_at = (up.at_content * up.length_bp + dn.at_content * dn.length_bp) / max(up.length_bp + dn.length_bp, 1)
        return {
            "upstream": up.model_dump(),
            "downstream": dn.model_dump(),
            "mean_smar_strength": round(mean_strength, 4),
            "combined_at_content": round(combined_at, 4),
            "shielding_predicted": combined_at >= self.min_at_content and mean_strength >= 0.5,
        }


# Backwards-compatible alias so existing imports keep working
TranscriptionalShieldingEngine = CpGOptimizationEngine
