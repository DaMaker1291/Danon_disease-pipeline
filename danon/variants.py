"""The variant channel: real measured allele fractions, and what they support.

WHY THIS MODULE EXISTS
----------------------
`danon.walkforward` refuses a k-bin allele-frequency spectrum because the engine
carries exactly two clone bins. That refusal was correct and it was also a dead
end: the GLASS release is a live, public, paired resource whose mutation records
carry `tumorAltCount`/`tumorRefCount`, so a per-variant ALLELE FRACTION is
measurable from real patient specimens today. This module carries that measured
quantity and states, in code, exactly how far it can be pushed.

MEASURED, NOT DECLARED
----------------------
* An allele fraction is computed from counts: ``AF = alt / (alt + ref)``. It is
  never accepted as a declared number, so a corrupted or truncated record cannot
  enter as a plausible-looking fraction.
* A specimen's spectrum is the set of its measured variants, one allele fraction
  each, keyed by (gene, chromosome, start, ref, alt).
* The primary -> recurrence change is measured per shared variant. Nothing is
  interpolated to a day number: the release labels specimens by kind, not by day.

WHAT CAN BE PROVED FROM IT, AND WHAT CANNOT
-------------------------------------------
Write ``f`` for the fraction of TUMOUR CELLS carrying the variant and ``v/t`` for
the variant's share of the copies at that locus. Then ``A_t = f * (v / t)``, and
since ``v <= t`` always, ``A_t <= f``. So a LOWER bound on the clone fraction is
provable with **no copy-number assumption at all**:

    f >= A_t, and A_t = AF / purity under one declared assumption (the normal
    compartment contributes no copies at that locus -- i.e. the variant is
    somatic).

An UPPER bound needs the local copy number, which this release does not carry, so
no upper bound is published (`UPPER_BOUND_REFUSED`). A point clone fraction is
refused for the same reason (`POINT_FRACTION_REFUSED`), and the mapping of a
k-bin spectrum onto the engine's two bins is refused because the summarisation
rule does not exist (`TWO_BIN_SUMMARY_REFUSED`).

A lower bound is what the first research question needs: whether an earlier
sample's molecular state predicts a later change, scored against a baseline that
does not have the earlier sample. A bound can be scored; a number asserted from
an assumption cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from danon.cure import EvidenceStatus

#: The declared assumption that turns a measured allele fraction into a tumour
#: allele fraction. Somatic here means the normal compartment contributes no
#: copies at that locus; a germline variant would break it, which is why the
#: release's `mutationStatus` is carried rather than assumed.
SOMATIC_ASSUMPTION = (
    "somatic: the normal compartment contributes no copies at this locus, so "
    "the tumour allele fraction is AF_observed / purity"
)

#: The provable step. No copy number is needed for it, and none is assumed.
LOWER_BOUND_DERIVATION = (
    "A_t = f * (variant copies / total copies) and variant copies <= total "
    "copies, so A_t <= f: the measured tumour allele fraction is a lower bound "
    "on the fraction of tumour cells carrying the variant"
)

UPPER_BOUND_REFUSED = (
    "refused: an upper bound on the clone fraction needs the local copy number "
    "(f = A_t * total/variant copies), and the mutation records of this release "
    "carry no copy number or purity field -- so only a lower bound is published"
)

POINT_FRACTION_REFUSED = (
    "refused: a point clone fraction would require a copy number, a purity and "
    "a cellularity model, none of which this release carries; a lower bound is "
    "published instead of a point"
)

TWO_BIN_SUMMARY_REFUSED = (
    "refused: this spectrum has k bins and the engine has 2, so it cannot be "
    "scored as a clone state -- danon.walkforward.clonal_jsd refuses a support "
    "of a different size by construction, and no summarisation rule from a "
    "k-bin allele-frequency spectrum onto a two-bin cell-population fraction "
    "exists in this repository"
)


#: The DECLARED gene panel this channel reads. Declared, not "the genome": the
#: mutations endpoint of the GLASS mutation profile requires a gene filter (an
#: unfiltered query answers HTTP 400), so the panel is exactly this list and a
#: change outside it is invisible here. The genes are the ones a glioma
#: evolution-and-resistance route names: the mismatch-repair and MGMT axis
#: (temozolomide resistance and hypermutation), the receptor and PI3K/PTEN axis,
#: the cell-cycle axis, and the IDH/ATRX/TP53 lineage axis.
VARIANT_PANEL: Tuple[Tuple[str, int], ...] = (
    ("MGMT", 4255),
    ("MSH6", 2956),
    ("MSH2", 4436),
    ("MSH3", 4437),
    ("MLH1", 4292),
    ("PMS2", 5395),
    ("POLE", 5426),
    ("TP53", 7157),
    ("EGFR", 1956),
    ("PTEN", 5728),
    ("IDH1", 3417),
    ("IDH2", 3418),
    ("ATRX", 546),
    ("NF1", 4763),
    ("PIK3CA", 5290),
    ("PIK3R1", 5295),
    ("RB1", 5925),
    ("CDKN2A", 1029),
    ("CDK4", 1019),
    ("MDM2", 4193),
    ("MDM4", 4194),
    ("PDGFRA", 5156),
    ("TERT", 7015),
    ("BRAF", 673),
    ("CIC", 23152),
    ("FUBP1", 8880),
    ("SETD2", 29072),
)


class VariantError(ValueError):
    """A variant record, spectrum or mapping that cannot be supported."""


# ---------------------------------------------------------------------------
# Records and spectra: the measured layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariantRecord:
    """One measured variant in one specimen. Counts, not declared fractions."""

    patient_id: str
    sample_id: str
    timepoint: str            # the specimen's kind, e.g. "Primary"/"Recurrence"
    gene: str
    protein_change: str
    chromosome: str
    start: int
    ref_allele: str
    alt_allele: str
    alt_count: int
    ref_count: int
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("patient_id", "sample_id", "timepoint", "gene"):
            if not str(getattr(self, name)).strip():
                raise VariantError(f"{name} must be a non-empty string")
        for name in ("alt_count", "ref_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise VariantError(f"{name} must be an int, got {type(value).__name__}")
            if value < 0:
                raise VariantError(f"{name} cannot be negative, got {value}")
        if self.alt_count + self.ref_count <= 0:
            raise VariantError(
                "zero sequencing depth: an allele fraction is not defined, and a "
                "record with no reads must be dropped rather than entered as 0.0"
            )

    @property
    def depth(self) -> int:
        return self.alt_count + self.ref_count

    @property
    def allele_fraction(self) -> float:
        """The measured fraction of reads carrying the variant."""
        return self.alt_count / self.depth

    @property
    def key(self) -> Tuple[str, str, int, str, str]:
        return (
            self.gene,
            str(self.chromosome),
            int(self.start),
            str(self.ref_allele),
            str(self.alt_allele),
        )

    @property
    def describe(self) -> str:
        return (
            f"{self.gene} {self.protein_change} in {self.sample_id}: "
            f"{self.alt_count}/{self.depth} reads, AF {self.allele_fraction:.3f}"
        )


@dataclass(frozen=True)
class Spectrum:
    """Every measured variant in one specimen (one patient, one timepoint)."""

    patient_id: str
    timepoint: str
    sample_id: str
    records: Tuple[VariantRecord, ...] = ()

    def __post_init__(self) -> None:
        for name in ("patient_id", "timepoint", "sample_id"):
            if not str(getattr(self, name)).strip():
                raise VariantError(f"{name} must be a non-empty string")
        for rec in self.records:
            if rec.patient_id != self.patient_id:
                raise VariantError(
                    f"spectrum for {self.patient_id} carries a record for "
                    f"{rec.patient_id}: a spectrum is one patient's specimen"
                )
            if rec.sample_id != self.sample_id:
                raise VariantError(
                    f"spectrum for {self.sample_id} carries a record from "
                    f"{rec.sample_id}: a spectrum is one specimen"
                )

    def __len__(self) -> int:
        return len(self.records)

    def keyed(self) -> Dict[Tuple[str, str, int, str, str], VariantRecord]:
        """Records by variant key; a repeated key in one specimen is an error."""
        out: Dict[Tuple[str, str, int, str, str], VariantRecord] = {}
        for rec in self.records:
            if rec.key in out:
                raise VariantError(
                    f"variant {rec.key} appears twice in {self.sample_id}: a "
                    "spectrum cannot carry the same locus twice"
                )
            out[rec.key] = rec
        return out

    def allele_fractions(self) -> Dict[Tuple[str, str, int, str, str], float]:
        return {k: r.allele_fraction for k, r in self.keyed().items()}


# ---------------------------------------------------------------------------
# The primary -> recurrence change: the measured longitudinal quantity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedChange:
    """What actually changed between two specimens of the same patient.

    Every field is measured from the paired spectra. There is no fitness, no
    clonal reconstruction and no day number: the release labels specimen KIND,
    so 'primary versus recurrence' is the only pairing the data licenses.
    """

    patient_id: str
    primary: str
    recurrence: str
    shared: Tuple[str, ...] = ()
    gained: Tuple[str, ...] = ()
    lost: Tuple[str, ...] = ()
    af_primary: Dict[str, float] = field(default_factory=dict)
    af_recurrence: Dict[str, float] = field(default_factory=dict)
    af_delta: Dict[str, float] = field(default_factory=dict)

    @property
    def n_shared(self) -> int:
        return len(self.shared)

    @property
    def n_gained(self) -> int:
        return len(self.gained)

    @property
    def n_lost(self) -> int:
        return len(self.lost)

    def rising(self) -> Tuple[str, ...]:
        return tuple(k for k in self.shared if self.af_delta[k] > 0)


def _label(key: Tuple[str, str, int, str, str]) -> str:
    gene, chrom, start, ref, alt = key
    return f"{gene}:{chrom}:{start}:{ref}>{alt}"


def paired_change(primary: Spectrum, recurrence: Spectrum) -> PairedChange:
    """Measure the change between two specimens of the SAME patient."""
    if primary.patient_id != recurrence.patient_id:
        raise VariantError(
            "a paired change needs two specimens of one patient, got "
            f"{primary.patient_id} and {recurrence.patient_id}"
        )
    if primary.sample_id == recurrence.sample_id:
        raise VariantError(
            f"both spectra are specimen {primary.sample_id}: a patient with one "
            "specimen has no longitudinal change to measure"
        )
    a = primary.allele_fractions()
    b = recurrence.allele_fractions()
    shared = tuple(_label(k) for k in sorted(set(a) & set(b)))
    gained = tuple(_label(k) for k in sorted(set(b) - set(a)))
    lost = tuple(_label(k) for k in sorted(set(a) - set(b)))
    return PairedChange(
        patient_id=primary.patient_id,
        primary=primary.sample_id,
        recurrence=recurrence.sample_id,
        shared=shared,
        gained=gained,
        lost=lost,
        af_primary={_label(k): a[k] for k in sorted(set(a) & set(b))},
        af_recurrence={_label(k): b[k] for k in sorted(set(a) & set(b))},
        af_delta={_label(k): b[k] - a[k] for k in sorted(set(a) & set(b))},
    )


# ---------------------------------------------------------------------------
# The one mapping the data licenses: a lower bound, with its assumption named
# ---------------------------------------------------------------------------


def clone_fraction_lower_bound(
    allele_fraction: float, purity: Optional[float] = None
) -> Tuple[float, str]:
    """A GUARANTEED lower bound on the fraction of tumour cells carrying the
    variant, with the derivation returned beside it.

    ``purity`` is the tumour-cell fraction of the specimen. It is optional
    precisely because this release does not carry it: without it the bound is the
    measured tumour allele fraction itself (the weaker but assumption-light
    ``f >= A_t``), and with it the declared somatic assumption gives
    ``A_t = AF / purity``. An upper bound is never returned -- see
    :data:`UPPER_BOUND_REFUSED`.
    """
    af = float(allele_fraction)
    if not 0.0 <= af <= 1.0:
        raise VariantError(f"an allele fraction must be in [0, 1], got {af}")
    if purity is None:
        return af, (
            f"f >= AF = {af:.4f} (no purity supplied, so the tumour allele "
            "fraction is taken as the measured one); " + LOWER_BOUND_DERIVATION
        )
    p = float(purity)
    if not 0.0 < p <= 1.0:
        raise VariantError(f"purity must be in (0, 1], got {p}")
    if af > p:
        raise VariantError(
            f"AF {af:.4f} exceeds purity {p:.4f}: a tumour compartment that is "
            "p of the specimen cannot carry this allele fraction, so the purity "
            "and the measurement disagree and no bound is published"
        )
    return af / p, (
        f"f >= AF/purity = {af:.4f}/{p:.4f} = {af / p:.4f} under the declared "
        f"assumption ({SOMATIC_ASSUMPTION}); " + LOWER_BOUND_DERIVATION
    )


def summarise_onto_two_bins(_spectrum: Spectrum) -> Tuple[float, float]:
    """Always refuses. Kept as a function so the refusal is importable and
    probeable rather than a comment nobody can test."""
    raise VariantError(TWO_BIN_SUMMARY_REFUSED)


# ---------------------------------------------------------------------------
# Cohort-level measurement (used by _glass_variant_cohort.py)
# ---------------------------------------------------------------------------


def spectra_from_records(records: Iterable[VariantRecord]) -> List[Spectrum]:
    """One spectrum per (patient, sample), in a deterministic order."""
    grouped: Dict[Tuple[str, str], List[VariantRecord]] = {}
    for rec in records:
        grouped.setdefault((rec.patient_id, rec.sample_id), []).append(rec)
    out: List[Spectrum] = []
    for (patient_id, sample_id), recs in sorted(grouped.items()):
        timepoints = sorted({r.timepoint for r in recs})
        out.append(
            Spectrum(
                patient_id=patient_id,
                sample_id=sample_id,
                timepoint="|".join(timepoints) if timepoints else "unlabelled",
                records=tuple(sorted(recs, key=lambda r: r.key)),
            )
        )
    return out


def paired_changes(spectra: Sequence[Spectrum]) -> List[PairedChange]:
    """Every primary -> recurrence pair the spectra actually provide.

    Only pairs whose timepoint labels differ are formed, and only when one
    specimen is labelled primary and the other recurrent: a pair of two
    primaries is not a longitudinal change, it is a duplicate.
    """
    by_patient: Dict[str, List[Spectrum]] = {}
    for spec in spectra:
        by_patient.setdefault(spec.patient_id, []).append(spec)
    out: List[PairedChange] = []
    for patient_id, specs in sorted(by_patient.items()):
        primaries = [s for s in specs if "primary" in s.timepoint.lower()]
        recurrents = [
            s
            for s in specs
            if "recurrence" in s.timepoint.lower() or "recurrent" in s.timepoint.lower()
        ]
        for p in sorted(primaries, key=lambda s: s.sample_id):
            for r in sorted(recurrents, key=lambda s: s.sample_id):
                if p.sample_id != r.sample_id:
                    out.append(paired_change(p, r))
    return out


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def variant_channel_provenance() -> Dict[str, EvidenceStatus]:
    """How well each link in this channel is supported, stated per link."""
    return {
        # The counts are in the public processed release and are read live.
        "allele_fraction_from_read_counts": EvidenceStatus.ESTABLISHED,
        # The lower bound is arithmetic, not a model.
        "clone_fraction_lower_bound": EvidenceStatus.ESTABLISHED,
        # The somatic assumption is declared, not measured for these specimens.
        "somatic_assumption": EvidenceStatus.UNKNOWN,
        # A per-patient purity is not in this release.
        "tumour_purity": EvidenceStatus.UNKNOWN,
        "point_clone_fraction": EvidenceStatus.PROVEN_FAILED,
        "upper_bound_on_clone_fraction": EvidenceStatus.PROVEN_FAILED,
        # No rule exists here for k bins -> the engine's two.
        "two_bin_summarisation": EvidenceStatus.PROVEN_FAILED,
        # Nothing in this channel is a treatment recommendation or an outcome.
        "clinical_benefit": EvidenceStatus.UNKNOWN,
        "engine_predicts_patients": EvidenceStatus.UNKNOWN,
    }


def what_this_channel_cannot_know() -> str:
    """Stated so no number here is read as a clone size, a cause or an outcome."""
    return (
        "This channel measures allele fractions in bulk specimens and lower "
        "bounds on the fraction of tumour cells carrying a variant. It is NOT a "
        "clonal reconstruction: two variants at the same lower bound may sit in "
        "the same cells or in different cells, and nothing here resolves that. "
        "It carries no copy number, no purity, no treatment record and no "
        "outcome, so it cannot say whether a change was caused by therapy, by "
        "sampling or by drift, and it cannot be scored as the engine's clone "
        "state. A rising variant is a measurement, not a prediction."
    )
