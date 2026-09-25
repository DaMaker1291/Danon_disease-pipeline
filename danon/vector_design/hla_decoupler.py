"""
PHASE 23 — Human Leukocyte Antigen (HLA) Decoupler
===========================================================
Screens therapeutic payload segments for peptides that bind human HLA alleles.
Peptides presented on MHC-II can flag transduced cardiomyocytes for CD4+ T-cell
attack; peptides presented on MHC-I can flag them for CD8+ cytotoxicity. Any
high-affinity binder must be engineered out.

Two screening modes:

  1. **Supertype mode** (default, allele-agnostic): screens against the
     HLA-DRB1 supertype anchor profile. This is the original behaviour and is
     appropriate for cohort-level design work.

  2. **Patient mode**: pass `patient_alleles=` and each segment is screened
     against that specific haplotype. A peptide counts as immunogenic if *any*
     of the patient's alleles presents it, because presentation on any allele is
     a liability for that patient.

Algorithm:
  - Slide a 15-mer (class II) / 9-mer (class I) window along each segment.
  - For every core inside the window, score with an allele-specific position
    weight matrix built around the anchor pockets of the groove: P1/P4/P6/P9 for
    the open-ended class-II groove, P2/P9 for the closed class-I groove.
  - Convert the PWM log-score to a predicted IC50 (nM) via the standard log50k
    transform: IC50 = 50000^(1 - score01).
  - A peptide is immunogenic if IC50 < cutoff (default 500 nM); the segment is
    "decoupled" only if it contains no such binder.

IMPORTANT LIMITATION
--------------------
The anchor tables below are hand-built supertype approximations in the spirit of
Nielsen et al. 2007 and the Southwood/Sette supertype motifs. They are NOT a
substitute for a trained predictor such as NetMHCpan or MHCflurry, and they do
not model proteasomal processing, TAP transport, or TCR recognition. Treat
per-allele output as indicative, not predictive. A `predictor` hook is exposed
so a real model can be substituted without changing callers.

Reference: Nielsen et al. 2007 (NetMHCII PSSM cores); Southwood et al. 1998
           (DRB1 supertype anchors); Sette & Sidney 1999 (class I supertypes).
"""
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Amino-acid property sets
# ---------------------------------------------------------------------------
_HYDRO = set("LIVMFYW")
_SMALL = set("AGSTC")
_POLAR = set("NQSTHY")
_NEG = set("DE")
_POS = set("KRH")
_AROMATIC = set("FYWH")
_ALIPHATIC = set("LIVMA")

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def _pref(favoured, good=(), neutral=(), penalty=()):
    """Build an anchor preference dict: 1.0 favoured, 0.5 good, 0.15 neutral."""
    table = {}
    for aa in ALPHABET:
        if aa in favoured:
            table[aa] = 1.0
        elif aa in good:
            table[aa] = 0.5
        elif aa in penalty:
            table[aa] = 0.05
        else:
            table[aa] = 0.15
    return table


# ---------------------------------------------------------------------------
# Class II — HLA-DRB1 supertypes (P1 deep pocket dominates)
# ---------------------------------------------------------------------------
P1_PREF = {aa: (1.0 if aa in _HYDRO else 0.15) for aa in ALPHABET}
P4_PREF = {aa: (0.9 if aa in (_HYDRO | _POS) else 0.4 if aa in _POLAR else 0.2)
           for aa in ALPHABET}
P6_PREF = {aa: (0.9 if aa in (_SMALL | _POLAR) else 0.5 if aa in _HYDRO else 0.25)
           for aa in ALPHABET}
P9_PREF = {aa: (0.85 if aa in (_HYDRO | _SMALL) else 0.35) for aa in ALPHABET}
BACKGROUND = {aa: 0.4 for aa in ALPHABET}
BACKGROUND["P"] = 0.1

# The original DRB1 supertype profile — preserved exactly for backwards
# compatibility with cohort-level callers and existing tests.
ANCHOR_WEIGHTS = {0: (P1_PREF, 3.0), 3: (P4_PREF, 1.4), 5: (P6_PREF, 1.4), 8: (P9_PREF, 1.2)}


def _dr_supertype(p1_fav, p4_fav, p6_fav, p9_fav) -> Dict[int, Tuple[Dict[str, float], float]]:
    return {
        0: (_pref(p1_fav, good=_HYDRO), 3.0),
        3: (_pref(p4_fav, good=_ALIPHATIC), 1.4),
        5: (_pref(p6_fav, good=_SMALL | _POLAR), 1.4),
        8: (_pref(p9_fav, good=_ALIPHATIC), 1.2),
    }


DRB1_SUPERTYPES: Dict[str, Dict[int, Tuple[Dict[str, float], float]]] = {
    "DR1":  _dr_supertype(_AROMATIC | _SMALL, _ALIPHATIC, _SMALL, _ALIPHATIC),
    "DR3":  _dr_supertype(_HYDRO, _ALIPHATIC | _NEG, _SMALL, _ALIPHATIC),
    "DR4":  _dr_supertype(_AROMATIC | _HYDRO, _ALIPHATIC, _SMALL, _ALIPHATIC),
    "DR5":  _dr_supertype(_AROMATIC | _HYDRO, _ALIPHATIC, _SMALL | _POLAR, _ALIPHATIC),
    "DR7":  _dr_supertype(_AROMATIC | {"L", "I", "M"}, _ALIPHATIC, _SMALL, _ALIPHATIC),
    "DR9":  _dr_supertype(_AROMATIC | _HYDRO, _AROMATIC | _ALIPHATIC, _SMALL, _AROMATIC | _ALIPHATIC),
    "DR15": _dr_supertype(_AROMATIC | _HYDRO, _ALIPHATIC, _SMALL, _SMALL),
}

# ---------------------------------------------------------------------------
# Class I — HLA-A / HLA-B supertypes (P2 and P9 anchors, closed groove)
# ---------------------------------------------------------------------------
def _class_i_supertype(p2_fav, p9_fav, p2_good=(), p9_good=()) -> Dict[int, Tuple[Dict[str, float], float]]:
    return {
        1: (_pref(p2_fav, good=p2_good, penalty="P"), 2.6),   # P2
        2: (_pref(_SMALL | _HYDRO, good=_POLAR), 0.8),        # P3 secondary
        8: (_pref(p9_fav, good=p9_good), 2.2),                # P9
    }


CLASS_I_SUPERTYPES: Dict[str, Dict[int, Tuple[Dict[str, float], float]]] = {
    "A1":  _class_i_supertype(_SMALL, _AROMATIC, p9_good=_HYDRO),
    "A2":  _class_i_supertype(_ALIPHATIC | {"T", "Q"}, _ALIPHATIC, p2_good=_SMALL),
    "A3":  _class_i_supertype(_ALIPHATIC | _SMALL, _POS, p9_good=_ALIPHATIC),
    "B7":  _class_i_supertype(set("P"), _ALIPHATIC, p2_good=_SMALL),
    "B44": _class_i_supertype(set("E"), _AROMATIC | _HYDRO, p9_good=_ALIPHATIC),
}

# Allele -> supertype. Matched on the field-1 (e.g. "02" of A*02:01).
_DRB1_TO_SUPERTYPE = {
    "01": "DR1", "10": "DR1",
    "03": "DR3", "11": "DR3", "13": "DR3", "14": "DR3",
    "04": "DR4", "08": "DR4",
    "12": "DR5",
    "07": "DR7",
    "09": "DR9",
    "15": "DR15", "16": "DR15",
}
_A_TO_SUPERTYPE = {
    "01": "A1", "25": "A1", "26": "A1", "32": "A1", "43": "A1", "66": "A1",
    "02": "A2", "68": "A2", "69": "A2",
    "03": "A3", "11": "A3", "31": "A3", "33": "A3",
}
_B_TO_SUPERTYPE = {
    "07": "B7", "35": "B7", "51": "B7", "53": "B7", "54": "B7", "55": "B7", "56": "B7", "67": "B7",
    "37": "B44", "40": "B44", "41": "B44", "44": "B44",
}

IC50_MAX = 50000.0

# Class II peptides are longer and the groove is open-ended.
CLASS_II_WINDOW = 15
CLASS_I_WINDOW = 9
# Fraction of random nonamers that bind any given allele (used to contextualise
# hit counts). Order-of-magnitude only.
_RANDOM_BINDER_RATE = 0.02


class AlleleAnchorSet(BaseModel):
    """Anchor profile resolved for one patient allele."""

    allele: str
    locus: str                      # "I" or "II"
    supertype: str
    core_len: int = 9
    window: int = CLASS_II_WINDOW
    # position index -> (preference table, weight)
    anchors: Dict[int, Tuple[Dict[str, float], float]] = Field(default_factory=dict)
    resolved: bool = True           # False when we had to fall back

    class Config:
        arbitrary_types_allowed = True


def resolve_allele(allele: str) -> AlleleAnchorSet:
    """Map an HLA allele string onto a supertype anchor profile.

    Accepts forms like "HLA-A*02:01", "A*02:01", "HLA-DRB1*15:01", "DRB1*15".
    Unknown alleles fall back to the DRB1 supertype with resolved=False so the
    caller can see that the call was not allele-specific.
    """
    raw = allele.strip().upper().replace("HLA-", "")
    if "*" not in raw:
        # A bare supertype name, e.g. "DR15" or "A2".
        if raw in DRB1_SUPERTYPES:
            return AlleleAnchorSet(
                allele=allele, locus="II", supertype=raw,
                anchors=DRB1_SUPERTYPES[raw], window=CLASS_II_WINDOW,
            )
        if raw in CLASS_I_SUPERTYPES:
            return AlleleAnchorSet(
                allele=allele, locus="I", supertype=raw,
                anchors=CLASS_I_SUPERTYPES[raw], window=CLASS_I_WINDOW,
            )
        return AlleleAnchorSet(
            allele=allele, locus="II", supertype="DRB1_supertype",
            anchors=ANCHOR_WEIGHTS, window=CLASS_II_WINDOW, resolved=False,
        )

    locus_part, _, rest = raw.partition("*")
    field1 = rest.split(":")[0].split("-")[0]

    if "DRB" in locus_part:
        st = _DRB1_TO_SUPERTYPE.get(field1)
        if st:
            return AlleleAnchorSet(
                allele=allele, locus="II", supertype=st,
                anchors=DRB1_SUPERTYPES[st], window=CLASS_II_WINDOW,
            )
    elif "DP" in locus_part or "DQ" in locus_part:
        # DP/DQ are not modelled; fall back rather than silently mis-typing.
        return AlleleAnchorSet(
            allele=allele, locus="II", supertype="DRB1_supertype",
            anchors=ANCHOR_WEIGHTS, window=CLASS_II_WINDOW, resolved=False,
        )
    elif locus_part.endswith("A"):
        st = _A_TO_SUPERTYPE.get(field1)
        if st:
            return AlleleAnchorSet(
                allele=allele, locus="I", supertype=st,
                anchors=CLASS_I_SUPERTYPES[st], window=CLASS_I_WINDOW,
            )
    elif locus_part.endswith("B"):
        st = _B_TO_SUPERTYPE.get(field1)
        if st:
            return AlleleAnchorSet(
                allele=allele, locus="I", supertype=st,
                anchors=CLASS_I_SUPERTYPES[st], window=CLASS_I_WINDOW,
            )

    return AlleleAnchorSet(
        allele=allele, locus="II", supertype="DRB1_supertype",
        anchors=ANCHOR_WEIGHTS, window=CLASS_II_WINDOW, resolved=False,
    )


class PeptideBinder(BaseModel):
    segment: str
    core_9mer: str
    core_offset: int
    binding_score: float = Field(ge=0.0, le=1.0)
    predicted_ic50_nm: float
    immunogenic: bool
    # Allele-aware extensions (default to the cohort-level DRB1 supertype call)
    allele: str = "DRB1_supertype"
    locus: str = "II"
    supertype: str = "DRB1_supertype"


class HLADecouplerResult(BaseModel):
    segments_screened: int
    peptides_evaluated: int
    binders: List[PeptideBinder] = Field(default_factory=list)
    high_affinity_hits: int
    strongest_binder_ic50_nm: float
    decoupled: bool
    ic50_cutoff_nm: float
    # Per-patient extensions
    alleles_screened: List[str] = Field(default_factory=list)
    per_allele_hits: Dict[str, int] = Field(default_factory=dict)
    dominant_allele: str = ""
    unresolved_alleles: List[str] = Field(default_factory=list)


class HLADecoupler:
    """MHC immunogenicity screen, cohort-level by default and per-patient on request.

    Args:
        ic50_cutoff_nm: class II binder threshold.
        patient_alleles: optional haplotype, e.g. ("HLA-A*02:01", "HLA-DRB1*15:01").
            When supplied, screening becomes allele-specific and a peptide is a
            hit if any of the patient's alleles presents it.
        class_i_cutoff_nm: separate (usually tighter) threshold for class I.
        predictor: optional callable `(core, allele_set) -> score in [0,1]` to
            substitute a trained model for the built-in PWM.
    """

    def __init__(
        self,
        ic50_cutoff_nm: float = 500.0,
        patient_alleles: Optional[Tuple[str, ...]] = None,
        class_i_cutoff_nm: float = 500.0,
        predictor: Optional[Callable[[str, AlleleAnchorSet], float]] = None,
    ) -> None:
        self.ic50_cutoff_nm = ic50_cutoff_nm
        self.class_i_cutoff_nm = class_i_cutoff_nm
        self.predictor = predictor
        self.patient_alleles: Tuple[str, ...] = tuple(patient_alleles or ())
        self._allele_sets: List[AlleleAnchorSet] = [
            resolve_allele(a) for a in self.patient_alleles
        ]

    # -- scoring ----------------------------------------------------------
    def _score_core(
        self,
        core: str,
        anchors: Optional[Dict[int, Tuple[Dict[str, float], float]]] = None,
    ) -> float:
        """Weighted geometric mean of the anchor-pocket preferences.

        The open-ended class-II groove is dominated by the P1/P4/P6/P9 anchors;
        a single incompatible anchor (e.g. Pro or a charged residue in the deep
        P1 pocket) collapses binding, so a geometric mean is used rather than a
        permissive additive sum. Class I uses the P2/P9 anchors of the closed
        groove but the same geometric-mean logic applies.
        """
        if len(core) < 9:
            return 0.0
        table = anchors if anchors is not None else ANCHOR_WEIGHTS
        log_sum = 0.0
        wsum = 0.0
        for i, (pref, weight) in table.items():
            if i >= len(core):
                continue
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

    def _score_for_allele(self, core: str, allele_set: AlleleAnchorSet) -> float:
        if self.predictor is not None:
            return float(np.clip(self.predictor(core, allele_set), 0.0, 1.0))
        return self._score_core(core, allele_set.anchors)

    # -- screening --------------------------------------------------------
    def screen_segment(
        self, segment: str, allele: Optional[str] = None
    ) -> List[PeptideBinder]:
        """Screen one segment. Without `allele`, uses the DRB1 supertype.

        With `allele`, uses that allele's anchors and window length.
        """
        seg = segment.upper()
        if allele is None:
            allele_set = AlleleAnchorSet(
                allele="DRB1_supertype", locus="II", supertype="DRB1_supertype",
                anchors=ANCHOR_WEIGHTS, window=CLASS_II_WINDOW,
            )
        else:
            allele_set = resolve_allele(allele)

        binders: List[PeptideBinder] = []
        window = allele_set.window
        n = len(seg)
        starts = range(0, max(1, n - window + 1)) if n >= window else [0]
        cutoff = self.class_i_cutoff_nm if allele_set.locus == "I" else self.ic50_cutoff_nm

        for s in starts:
            frag = seg[s:s + window] if n >= window else seg
            best_core, best_off, best_score = "", 0, -1.0
            for off in range(0, max(1, len(frag) - 9 + 1)):
                core = frag[off:off + 9]
                if len(core) < 9:
                    continue
                sc = self._score_for_allele(core, allele_set)
                if sc > best_score:
                    best_score, best_core, best_off = sc, core, off
            if best_core:
                ic50 = self._ic50(best_score)
                binders.append(PeptideBinder(
                    segment=frag, core_9mer=best_core, core_offset=best_off,
                    binding_score=round(best_score, 4),
                    predicted_ic50_nm=round(ic50, 2),
                    immunogenic=ic50 < cutoff,
                    allele=allele_set.allele,
                    locus=allele_set.locus,
                    supertype=allele_set.supertype,
                ))
        return binders

    def screen_patient(self, segments: List[str]) -> List[PeptideBinder]:
        """Screen every segment against every patient allele.

        A binder call on any allele is retained, so `evaluate` over this list
        reflects the patient's true presentation liability.
        """
        if not self._allele_sets:
            out: List[PeptideBinder] = []
            for seg in segments:
                if seg:
                    out.extend(self.screen_segment(seg))
            return out

        out = []
        for seg in segments:
            if not seg:
                continue
            worst: Optional[PeptideBinder] = None
            for allele_set in self._allele_sets:
                hits = self.screen_segment(seg, allele=allele_set.allele)
                for b in hits:
                    out.append(b)
                    if worst is None or b.predicted_ic50_nm < worst.predicted_ic50_nm:
                        worst = b
        return out

    def evaluate(
        self, segments: List[str], alleles: Optional[Tuple[str, ...]] = None
    ) -> HLADecouplerResult:
        """Aggregate a segment list into a decoupling verdict.

        `alleles` overrides the instance-level haplotype for this call.
        """
        allele_list: Tuple[str, ...] = tuple(alleles) if alleles is not None else self.patient_alleles
        if allele_list and tuple(allele_list) != self.patient_alleles:
            allele_sets = [resolve_allele(a) for a in allele_list]
        else:
            allele_sets = self._allele_sets

        if allele_sets:
            all_binders: List[PeptideBinder] = []
            for seg in segments:
                if not seg:
                    continue
                for a_set in allele_sets:
                    all_binders.extend(self.screen_segment(seg, allele=a_set.allele))
        else:
            all_binders = []
            for seg in segments:
                if seg:
                    all_binders.extend(self.screen_segment(seg))

        per_allele: Dict[str, int] = {}
        for b in all_binders:
            if b.immunogenic:
                per_allele[b.allele] = per_allele.get(b.allele, 0) + 1
        dominant = max(per_allele.items(), key=lambda kv: kv[1])[0] if per_allele else ""

        unresolved = sorted({a.allele for a in allele_sets if not a.resolved})

        if not all_binders:
            return HLADecouplerResult(
                segments_screened=len(segments), peptides_evaluated=0, binders=[],
                high_affinity_hits=0, strongest_binder_ic50_nm=IC50_MAX,
                decoupled=True, ic50_cutoff_nm=self.ic50_cutoff_nm,
                alleles_screened=list(allele_list), per_allele_hits=per_allele,
                dominant_allele=dominant, unresolved_alleles=unresolved,
            )

        # A peptide presented by several patient alleles is a stronger liability,
        # so deduplicate cores but keep the strongest (lowest IC50) call.
        by_core: Dict[str, PeptideBinder] = {}
        for b in all_binders:
            prev = by_core.get(b.core_9mer)
            if prev is None or b.predicted_ic50_nm < prev.predicted_ic50_nm:
                by_core[b.core_9mer] = b
        unique = list(by_core.values())

        hits = sum(1 for b in unique if b.immunogenic)
        strongest = min(b.predicted_ic50_nm for b in unique)
        unique.sort(key=lambda b: b.predicted_ic50_nm)
        return HLADecouplerResult(
            segments_screened=len(segments),
            peptides_evaluated=len(unique),
            binders=unique[:20],
            high_affinity_hits=hits,
            strongest_binder_ic50_nm=round(strongest, 2),
            decoupled=hits == 0,
            ic50_cutoff_nm=self.ic50_cutoff_nm,
            alleles_screened=list(allele_list),
            per_allele_hits=per_allele,
            dominant_allele=dominant,
            unresolved_alleles=unresolved,
        )

    # -- patient-level reporting -----------------------------------------
    def patient_report(self, segments: List[str]) -> Dict:
        """Compact summary for the personalization / manufacturing layer."""
        result = self.evaluate(segments)
        deimmunization_targets = [
            b.core_9mer for b in result.binders if b.immunogenic
        ]
        return {
            "alleles_screened": result.alleles_screened,
            "unresolved_alleles": result.unresolved_alleles,
            "segs": result.segments_screened,
            "hits": result.high_affinity_hits,
            "decoupled": result.decoupled,
            "dominant_allele": result.dominant_allele,
            "per_allele_hits": result.per_allele_hits,
            "strongest_ic50_nm": result.strongest_binder_ic50_nm,
            "deimmunization_targets": deimmunization_targets,
            "personalized": bool(result.alleles_screened)
            and not result.unresolved_alleles,
        }
