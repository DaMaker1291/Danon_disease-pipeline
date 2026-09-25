"""
PHASE 24 — High-Fidelity Automated Synthesis Error Screen
=========================================================
Scans the dual-plasmid AAV payload DNA for sequence features that cause gene-
synthesis and viral-packaging failures at commercial providers (Twist, PackGene,
GenScript) or during rAAV production:

  1. Global + windowed GC content     : extreme or highly variable GC breaks
                                        oligo assembly (allowable window, e.g. 40–65%)
  2. Homopolymer runs                 : long single-base tracts (>8 nt) cause
                                        polymerase slippage / sequencing dropouts
  3. Inverted repeats (hairpins)      : self-complementary stems mimic ITRs and
                                        stall synthesis / recombine
  4. Direct/tandem repeats            : promote misannealing and deletion
  5. Extreme GC micro-windows         : GC-rich stretches (>75%) that resist
                                        denaturation

Emits a synthesizability verdict and per-feature evidence.
"""
import logging
from typing import Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_COMP = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}


def revcomp(seq: str) -> str:
    return "".join(_COMP.get(b, "N") for b in reversed(seq.upper()))


class GCWindow(BaseModel):
    start: int
    end: int
    gc: float


class RepeatFeature(BaseModel):
    kind: str
    start: int
    length: int
    sequence: str


class SynthesisResult(BaseModel):
    length_bp: int
    gc_content: float
    gc_min_window: float
    gc_max_window: float
    out_of_bounds_windows: List[GCWindow] = Field(default_factory=list)
    homopolymer_runs: List[RepeatFeature] = Field(default_factory=list)
    inverted_repeats: List[RepeatFeature] = Field(default_factory=list)
    direct_repeats: List[RepeatFeature] = Field(default_factory=list)
    n_hard_failures: int
    synthesizable: bool
    gc_window_bounds: List[float]


class SynthesisGuard:
    """Commercial synthesis / packaging feasibility screen for payload DNA."""

    def __init__(self, gc_window: Tuple[float, float] = (40.0, 65.0),
                 window_size: int = 50, homopolymer_min: int = 9,
                 hairpin_stem_min: int = 12, hairpin_loop_max: int = 12,
                 direct_repeat_min: int = 12):
        self.gc_lo, self.gc_hi = gc_window
        self.window_size = window_size
        self.homopolymer_min = homopolymer_min
        self.hairpin_stem_min = hairpin_stem_min
        self.hairpin_loop_max = hairpin_loop_max
        self.direct_repeat_min = direct_repeat_min

    @staticmethod
    def _gc(seq: str) -> float:
        if not seq:
            return 0.0
        return 100.0 * sum(1 for b in seq if b in "GCgc") / len(seq)

    def _gc_windows(self, seq: str) -> Tuple[List[GCWindow], float, float]:
        out: List[GCWindow] = []
        gmin, gmax = 100.0, 0.0
        w = self.window_size
        step = max(1, w // 2)
        for i in range(0, max(1, len(seq) - w + 1), step):
            win = seq[i:i + w]
            gc = self._gc(win)
            gmin, gmax = min(gmin, gc), max(gmax, gc)
            if gc < self.gc_lo or gc > self.gc_hi:
                out.append(GCWindow(start=i, end=i + w, gc=round(gc, 2)))
        return out, round(gmin, 2), round(gmax, 2)

    def _homopolymers(self, seq: str) -> List[RepeatFeature]:
        runs: List[RepeatFeature] = []
        i = 0
        n = len(seq)
        while i < n:
            j = i
            while j < n and seq[j] == seq[i]:
                j += 1
            if j - i >= self.homopolymer_min:
                runs.append(RepeatFeature(kind=f"poly-{seq[i]}", start=i,
                                          length=j - i, sequence=seq[i:j]))
            i = j
        return runs

    def _inverted_repeats(self, seq: str) -> List[RepeatFeature]:
        """Detect stem>=k self-complementary arms with a short intervening loop."""
        found: List[RepeatFeature] = []
        n = len(seq)
        k = self.hairpin_stem_min
        seen_starts = set()
        step = 3
        for i in range(0, n - k, step):
            arm = seq[i:i + k]
            rc = revcomp(arm)
            # search for the arm's reverse-complement downstream within loop range
            search_lo = i + k
            search_hi = min(n - k, i + k + self.hairpin_loop_max + k)
            idx = seq.find(rc, search_lo, search_hi + k)
            if idx != -1 and i not in seen_starts:
                seen_starts.add(i)
                found.append(RepeatFeature(
                    kind="inverted_repeat", start=i,
                    length=(idx + k) - i, sequence=seq[i:idx + k],
                ))
        return found[:30]

    def _direct_repeats(self, seq: str) -> List[RepeatFeature]:
        found: List[RepeatFeature] = []
        n = len(seq)
        k = self.direct_repeat_min
        seen = set()
        step = 4
        for i in range(0, n - k, step):
            unit = seq[i:i + k]
            if unit in seen:
                continue
            idx = seq.find(unit, i + k)
            if idx != -1:
                seen.add(unit)
                found.append(RepeatFeature(
                    kind="direct_repeat", start=i, length=k, sequence=unit,
                ))
        return found[:30]

    def evaluate(self, dna: str) -> SynthesisResult:
        seq = "".join(b for b in dna.upper() if b in "ACGT")
        gc = round(self._gc(seq), 2)
        windows, gmin, gmax = self._gc_windows(seq)
        homo = self._homopolymers(seq)
        inverted = self._inverted_repeats(seq)
        direct = self._direct_repeats(seq)

        # hard failures: out-of-bound global GC, any long hairpin, >2 homopolymers
        hard = 0
        if gc < self.gc_lo or gc > self.gc_hi:
            hard += 1
        hard += len(inverted)
        hard += max(0, len(homo) - 2)

        return SynthesisResult(
            length_bp=len(seq),
            gc_content=gc,
            gc_min_window=gmin,
            gc_max_window=gmax,
            out_of_bounds_windows=windows[:40],
            homopolymer_runs=homo[:30],
            inverted_repeats=inverted,
            direct_repeats=direct,
            n_hard_failures=hard,
            synthesizable=hard == 0,
            gc_window_bounds=[self.gc_lo, self.gc_hi],
        )
