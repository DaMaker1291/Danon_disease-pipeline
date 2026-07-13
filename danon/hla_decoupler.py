"""
PHASE 23 — Human Leukocyte Antigen (HLA) Class II Decoupler
===========================================================
Screens the LAMP2B split-intein extein-junction segments for peptides that bind
common human HLA-DRB1 alleles. Junction-spanning peptides that would present on
MHC-II could flag treated cardiomyocytes for CD4+ T-cell attack, so any high-
affinity binder must be engineered out.

Algorithm:
  - Slide a 15-mer window along each junction segment.
  - For every 9-mer binding core inside the window, score with an allele-specific
    position weight matrix (PWM) built around the P1/P4/P6/P9 anchor pockets of
    HLA-DRB1 (open-ended class-II groove).
  - Convert the PWM log-score to a predicted IC50 (nM) via the standard log50k
    transform: IC50 = 50000^(1 - score01).
  - A peptide is immunogenic if IC50 < cutoff (default 500 nM); the segment is
    "decoupled" only if it contains no such binder.

Reference: Nielsen et al. 2007 (NetMHCII PSSM cores); Southwood et al. 1998
           (DRB1 supertype anchors).
"""
import logging
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Anchor-pocket preference profiles (relative binding contribution 0..1) for the
# HLA-DRB1 supertype. P1 = deep hydrophobic pocket; P4/P6/P9 = secondary anchors.
_HYDRO = set("LIVMFYW")
_SMALL = set("AGSTC")
_POLAR = set("NQSTHY")
_NEG = set("DE")
_POS = set("KRH")

P1_PREF = {aa: (1.0 if aa in _HYDRO else 0.15) for aa in "ACDEFGHIKLMNPQRSTVWY"}
P4_PREF = {aa: (0.9 if aa in (_HYDRO | _POS) else 0.4 if aa in _POLAR else 0.2)
           for aa in "ACDEFGHIKLMNPQRSTVWY"}
P6_PREF = {aa: (0.9 if aa in (_SMALL | _POLAR) else 0.5 if aa in _HYDRO else 0.25)
           for aa in "ACDEFGHIKLMNPQRSTVWY"}
P9_PREF = {aa: (0.85 if aa in (_HYDRO | _SMALL) else 0.35)
           for aa in "ACDEFGHIKLMNPQRSTVWY"}
# proline in the core / negative anchors are penalised
BACKGROUND = {aa: 0.4 for aa in "ACDEFGHIKLMNPQRSTVWY"}
BACKGROUND["P"] = 0.1

ANCHOR_WEIGHTS = {0: (P1_PREF, 3.0), 3: (P4_PREF, 1.4), 5: (P6_PREF, 1.4), 8: (P9_PREF, 1.2)}

IC50_MAX = 50000.0


class PeptideBinder(BaseModel):
    segment: str
    core_9mer: str
    core_offset: int
    binding_score: float = Field(ge=0.0, le=1.0)
    predicted_ic50_nm: float
    immunogenic: bool


class HLADecouplerResult(BaseModel):
    segments_screened: int
    peptides_evaluated: int
    binders: List[PeptideBinder] = Field(default_factory=list)
    high_affinity_hits: int
    strongest_binder_ic50_nm: float
    decoupled: bool
    ic50_cutoff_nm: float


class HLADecoupler:
    """MHC-II (HLA-DRB1) junction immunogenicity screen."""

    def __init__(self, ic50_cutoff_nm: float = 500.0):
        self.ic50_cutoff_nm = ic50_cutoff_nm

    def _score_core(self, core: str) -> float:
        """Weighted geometric mean of the four DRB1 anchor-pocket preferences.

        The open-ended class-II groove is dominated by the P1/P4/P6/P9 anchors;
        a single incompatible anchor (e.g. Pro or a charged residue in the deep
        P1 pocket) collapses binding, so a geometric mean is used rather than a
        permissive additive sum.
        """
        if len(core) < 9:
            return 0.0
        log_sum = 0.0
        wsum = 0.0
        for i, (pref, weight) in ANCHOR_WEIGHTS.items():
            aa = core[i]
            val = max(pref.get(aa, 0.15), 1e-3)
            log_sum += weight * np.log(val)
            wsum += weight
        # light penalty for proline anywhere in the rigid binding core
        proline_pen = 0.7 ** core.count("P")
        score = float(np.exp(log_sum / max(wsum, 1e-9))) * proline_pen
        return float(np.clip(score, 0.0, 1.0))

    def _ic50(self, score01: float) -> float:
        return float(IC50_MAX ** (1.0 - score01))

    def screen_segment(self, segment: str) -> List[PeptideBinder]:
        seg = segment.upper()
        binders: List[PeptideBinder] = []
        window = 15
        n = len(seg)
        # iterate 15-mer windows (or the whole segment if shorter)
        starts = range(0, max(1, n - window + 1)) if n >= window else [0]
        for s in starts:
            frag = seg[s:s + window] if n >= window else seg
            best_core, best_off, best_score = "", 0, -1.0
            for off in range(0, max(1, len(frag) - 9 + 1)):
                core = frag[off:off + 9]
                if len(core) < 9:
                    continue
                sc = self._score_core(core)
                if sc > best_score:
                    best_score, best_core, best_off = sc, core, off
            if best_core:
                ic50 = self._ic50(best_score)
                binders.append(PeptideBinder(
                    segment=frag, core_9mer=best_core, core_offset=best_off,
                    binding_score=round(best_score, 4),
                    predicted_ic50_nm=round(ic50, 2),
                    immunogenic=ic50 < self.ic50_cutoff_nm,
                ))
        return binders

    def evaluate(self, segments: List[str]) -> HLADecouplerResult:
        all_binders: List[PeptideBinder] = []
        for seg in segments:
            if seg:
                all_binders.extend(self.screen_segment(seg))
        if not all_binders:
            return HLADecouplerResult(
                segments_screened=len(segments), peptides_evaluated=0, binders=[],
                high_affinity_hits=0, strongest_binder_ic50_nm=IC50_MAX,
                decoupled=True, ic50_cutoff_nm=self.ic50_cutoff_nm,
            )
        hits = sum(1 for b in all_binders if b.immunogenic)
        strongest = min(b.predicted_ic50_nm for b in all_binders)
        all_binders.sort(key=lambda b: b.predicted_ic50_nm)
        return HLADecouplerResult(
            segments_screened=len(segments),
            peptides_evaluated=len(all_binders),
            binders=all_binders[:20],
            high_affinity_hits=hits,
            strongest_binder_ic50_nm=round(strongest, 2),
            decoupled=hits == 0,
            ic50_cutoff_nm=self.ic50_cutoff_nm,
        )
