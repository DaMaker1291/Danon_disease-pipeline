"""Longitudinal MRI availability manifests: schema, validation, census.

WHAT THIS MODULE IS FOR
-----------------------
Stream 4 of the clinical datastream spec (sequential MRI -> observed clonal
velocity) needs two dated scans of the same patient carrying the same usable
sequence. Whether such a pair exists is a property of the COHORT, and it is
recorded in availability manifests such as the CFB-GBM ``_mri_availability``
table: one row per (patient, timepoint), a flag per sequence, and the offset
from t0 in weeks.

That table is what makes the longitudinal stream measurable at all, so it is
ingested here as a first-class input rather than kept as a note in a README.
The mapping is validated and refused on a schema mismatch; nothing is guessed
from a mis-shaped file.

PROVENANCE (read from the collection page; DOI recorded, not asserted)
---------------------------------------------------------------------
    collection  CFB-GBM -- "Pre and post treatment MRI and radiotherapy plans
                of patients with glioblastoma: the CFB-GBM cohort"
    publisher   The Cancer Imaging Archive (TCIA); Centre Francois Baclesse
    DOI         10.7937/v9pn-2f72 (Data Citation Required)
    cohort      264 patients, 208.01 GB, MR/CT/RTDOSE + clinical + radiomics
    schema      pre-treatment MRI at t0 for all patients; post-treatment MRI
                at ~4 months (t1) and ~6 months (t2) when available
    NBIA API    NOT SERVED -- ``getSeries``/``getPatient`` return HTTP 200 with
                an empty body for this collection, and it is absent from the
                156 collections in ``getCollectionValues``. The images are not
                reachable through the endpoint this repo probes, which is
                precisely why the availability table matters: it enumerates
                what WOULD be there if the collection is downloaded.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not download anything, and it claims nothing about image content. A row
with ``t1gd = 1`` records that a T1-Gd series EXISTS for that patient and
timepoint -- not that it is registered, artefact-free, or comparable to the
other timepoint. The census counts pairs; measuring a velocity field needs the
images, and the scanner-change count below is reported because a pair spanning
two machines is a confound for exactly that measurement.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# --------------------------------------------------------------------------- #
# Declared constants
# --------------------------------------------------------------------------- #

AVAILABILITY_DIR = "data/cfb_gbm"

#: The committed manifest. 20 KB of per-(patient, timepoint) sequence flags.
CFB_GBM_MANIFEST = os.path.join(
    AVAILABILITY_DIR, "CFB-GBM_mri_availability_v02_20260129.tsv"
)

CFB_GBM_DOI = "10.7937/v9pn-2f72"
CFB_GBM_SOURCE = (
    "CFB-GBM, The Cancer Imaging Archive (Centre Francois Baclesse), "
    f"DOI {CFB_GBM_DOI} -- Data Citation Required"
)

#: The sequence a longitudinal pair must both carry. T1-Gd is the
#: contrast-enhanced T1 that the registration and delta-volume step needs, and
#: it is the acquisition the engine's own cached t1ce proxy corresponds to.
REFERENCE_SEQUENCE = "t1gd"

#: Minimum interval between the two scans of a pair, in weeks. 4 weeks is the
#: same floor as ``danon.multimodal.MIN_FOLLOWUP_INTERVAL_DAYS`` (28 days),
#: declared in weeks here because the manifest's own column is in weeks.
MIN_PAIR_INTERVAL_WEEKS = 4

#: Columns the census cannot be computed without. A file missing any of these
#: is refused rather than interpreted.
REQUIRED_COLUMNS: Tuple[str, ...] = (
    "id_patient",
    "temporality",
    "time_diff_t0 (weeks)",
)

#: Per-sequence availability flag columns, in the manifest's own order.
SEQUENCE_COLUMNS: Tuple[str, ...] = (
    "t1gd",
    "t1eg",
    "t1tse",
    "t2star",
    "t2tse",
    "flair",
    "adc",
)

MACHINE_COLUMN = "mri_machine"
TEMPORALITY_ORDER = ("t0", "t1", "t2")


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AvailabilityRow:
    """One (patient, timepoint) with its sequence flags."""

    patient_id: str
    timepoint: str
    offset_weeks: int
    sequences: Dict[str, bool]
    machine: str = ""

    def has(self, sequence: str) -> bool:
        return bool(self.sequences.get(sequence, False))


@dataclass(frozen=True)
class AvailabilityCensus:
    """Measured counts over a manifest. Every field is a count or a declared
    parameter; no field is an estimate."""

    source: str
    n_rows: int
    n_patients: int
    n_timepoints_by_label: Dict[str, int]
    n_patients_multi_timepoint: int
    sequence: str
    min_interval_weeks: int
    n_usable_pairs: int
    n_patients_with_usable_pair: int
    n_same_machine_pairs: int
    n_cross_machine_pairs: int
    n_patients_only_same_machine: int
    n_patients_with_cross_machine_pair: int
    interval_weeks_min: Optional[int]
    interval_weeks_median: Optional[float]
    interval_weeks_max: Optional[int]
    n_machines: Dict[str, int]
    n_patients_per_sequence: Dict[str, int]
    """Distinct patients carrying the sequence at both ends of a qualifying
    interval. Equal to ``n_patients_with_usable_pair`` for the reference
    sequence, since that predicate defines both."""
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [
            f"availability census -- {self.source}",
            f"  rows {self.n_rows}, patients {self.n_patients}, "
            f"timepoints {self.n_timepoints_by_label}",
            f"  patients with >1 timepoint: {self.n_patients_multi_timepoint}",
            f"  reference sequence {self.sequence}, min interval "
            f"{self.min_interval_weeks} week(s)",
            f"  usable pairs: {self.n_usable_pairs} "
            f"({self.n_patients_with_usable_pair} patients)",
            f"  pairs same machine: {self.n_same_machine_pairs}, cross machine: "
            f"{self.n_cross_machine_pairs}",
            f"  interval weeks min/median/max: {self.interval_weeks_min}/"
            f"{self.interval_weeks_median}/{self.interval_weeks_max}",
            f"  scanners: {self.n_machines}",
            f"  patients with the sequence at 2+ timepoints, by sequence: "
            f"{self.n_patients_per_sequence}",
        ]
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "n_rows": self.n_rows,
            "n_patients": self.n_patients,
            "n_timepoints_by_label": self.n_timepoints_by_label,
            "n_patients_multi_timepoint": self.n_patients_multi_timepoint,
            "reference_sequence": self.sequence,
            "min_interval_weeks": self.min_interval_weeks,
            "n_usable_pairs": self.n_usable_pairs,
            "n_patients_with_usable_pair": self.n_patients_with_usable_pair,
            "n_same_machine_pairs": self.n_same_machine_pairs,
            "n_cross_machine_pairs": self.n_cross_machine_pairs,
            "n_patients_only_same_machine": self.n_patients_only_same_machine,
            "n_patients_with_cross_machine_pair": self.n_patients_with_cross_machine_pair,
            "interval_weeks_min": self.interval_weeks_min,
            "interval_weeks_median": self.interval_weeks_median,
            "interval_weeks_max": self.interval_weeks_max,
            "n_machines": self.n_machines,
            "n_patients_per_sequence": self.n_patients_per_sequence,
            "notes": self.notes,
        }


def _flag(value: object) -> bool:
    return str(value or "").strip() == "1"


def looks_like_availability_manifest(text: str) -> bool:
    """True when a header line carries the manifest's own columns.

    Used by the intake endpoint to route a dropped file, and mirrored in
    ``frontend/src/utils/availability.ts`` so the browser makes the same
    decision. Content-based, not extension-based: the CFB-GBM table has arrived
    with ``.tsv`` and could arrive as ``.txt``.
    """
    text = text.lstrip("\ufeff")  # a BOM is not a column name
    first = text.splitlines()[0] if text.splitlines() else ""
    if "\t" not in first:
        return False
    cells = [c.strip().lower() for c in first.split("\t")]
    return all(c in cells for c in ("id_patient", "temporality")) and any(
        c.startswith("time_diff") for c in cells
    )


def census_from_text(text: str, *, source: str = "uploaded manifest") -> AvailabilityCensus:
    """Census a manifest supplied as text, validating the same way as the file
    loader. Raises ``ValueError`` on a schema mismatch.

    The BOM is stripped because the file loader opens with ``utf-8-sig``: a
    spreadsheet export that carries one must not be refused here as "missing
    column id_patient" while the same bytes load fine from disk.
    """
    lines = [l for l in text.lstrip("\ufeff").splitlines() if l.strip()]
    if not lines:
        raise ValueError("manifest is empty")
    header = tuple(c.strip() for c in lines[0].split("\t"))
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    offset_col = next((c for c in header if c.startswith("time_diff")), None)
    if offset_col is None:
        missing.append("time_diff_t0 (weeks)")
    if missing:
        raise ValueError(
            f"manifest is missing required column(s) {missing}; found {list(header)}"
        )
    index = {c: i for i, c in enumerate(header)}
    rows: List[AvailabilityRow] = []
    for lineno, line in enumerate(lines[1:], start=2):
        cells = line.split("\t")
        pid = cells[index["id_patient"]].strip() if len(cells) > index["id_patient"] else ""
        if not pid:
            continue
        raw_offset = cells[index[offset_col]].strip() if len(cells) > index[offset_col] else ""
        try:
            offset = int(float(raw_offset))
        except ValueError as exc:
            raise ValueError(
                f"line {lineno}: time_diff column is not numeric ({raw_offset!r})"
            ) from exc
        rows.append(
            AvailabilityRow(
                patient_id=pid,
                timepoint=(
                    cells[index["temporality"]].strip()
                    if len(cells) > index["temporality"] else ""
                ),
                offset_weeks=offset,
                sequences={
                    c: _flag(cells[index[c]]) for c in SEQUENCE_COLUMNS if c in index
                },
                machine=(
                    cells[index[MACHINE_COLUMN]].strip()
                    if MACHINE_COLUMN in index and len(cells) > index[MACHINE_COLUMN] else ""
                ),
            )
        )
    if not rows:
        raise ValueError("manifest contains no usable rows")
    return mri_availability_census(rows, source=source)


def load_mri_availability_manifest(
    path: str = CFB_GBM_MANIFEST,
) -> List[AvailabilityRow]:
    """Parse and validate an availability manifest.

    Raises ``FileNotFoundError`` if absent and ``ValueError`` on any schema
    mismatch (missing required column, non-numeric offset, duplicate
    patient/timepoint). A census computed from a file we had to guess at would
    be worse than no census.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no availability manifest at {path}. This module will not invent "
            f"one; supply the cohort's own table."
        )
    with open(path, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        header = tuple(reader.fieldnames or ())
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            raise ValueError(
                f"availability manifest {path} is missing required column(s) "
                f"{missing}; found {list(header)}"
            )
        rows: List[AvailabilityRow] = []
        seen: set = set()
        for lineno, raw in enumerate(reader, start=2):
            pid = str(raw.get("id_patient") or "").strip()
            if not pid:
                continue
            tp = str(raw.get("temporality") or "").strip()
            raw_offset = str(raw.get("time_diff_t0 (weeks)") or "").strip()
            try:
                offset = int(float(raw_offset))
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{lineno}: time_diff_t0 (weeks) is not numeric "
                    f"({raw_offset!r})"
                ) from exc
            key = (pid, tp)
            if key in seen:
                raise ValueError(
                    f"{path}:{lineno}: duplicate row for patient {pid} "
                    f"timepoint {tp!r}"
                )
            seen.add(key)
            rows.append(
                AvailabilityRow(
                    patient_id=pid,
                    timepoint=tp,
                    offset_weeks=offset,
                    sequences={c: _flag(raw.get(c)) for c in SEQUENCE_COLUMNS},
                    machine=str(raw.get(MACHINE_COLUMN) or "").strip(),
                )
            )
    if not rows:
        raise ValueError(f"availability manifest {path} contains no rows")
    return rows


# --------------------------------------------------------------------------- #
# Census
# --------------------------------------------------------------------------- #


def _median(values: Sequence[int]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return float(ordered[n // 2])
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def mri_availability_census(
    rows_or_path: Union[str, Sequence[AvailabilityRow], None] = None,
    *,
    sequence: str = REFERENCE_SEQUENCE,
    min_interval_weeks: int = MIN_PAIR_INTERVAL_WEEKS,
    source: str = CFB_GBM_SOURCE,
) -> AvailabilityCensus:
    """Count timepoints and usable longitudinal pairs over a whole manifest.

    A pair counts when the same patient carries ``sequence`` at two timepoints
    whose offset difference is at least ``min_interval_weeks``. Pairs are
    additionally split by scanner, because a pair spanning two machines is a
    confound for a delta-volume measurement rather than a clean follow-up.
    """
    if rows_or_path is None:
        rows = load_mri_availability_manifest()
    elif isinstance(rows_or_path, str):
        rows = load_mri_availability_manifest(rows_or_path)
    else:
        rows = list(rows_or_path)

    by_patient: Dict[str, List[AvailabilityRow]] = {}
    for row in rows:
        by_patient.setdefault(row.patient_id, []).append(row)

    labels: Dict[str, int] = {}
    machines: Dict[str, int] = {}
    for row in rows:
        labels[row.timepoint] = labels.get(row.timepoint, 0) + 1
        if row.machine:
            machines[row.machine] = machines.get(row.machine, 0) + 1

    n_patients_multi = 0
    n_usable = 0
    n_same = 0
    n_cross = 0
    intervals: List[int] = []
    patients_with_pair: set = set()
    patients_only_same: set = set()
    patients_cross: set = set()
    # A SET of patient ids per sequence, not a running total: the field is
    # named for patients and read as patients, so a patient with three
    # qualifying timepoints must count once, not three times (its pairs).
    per_sequence: Dict[str, set] = {c: set() for c in SEQUENCE_COLUMNS}
    for pid, prows in by_patient.items():
        if len(prows) > 1:
            n_patients_multi += 1
        ordered = sorted(prows, key=lambda r: r.offset_weeks)
        patient_had_pair = False
        patient_had_cross = False
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                gap = ordered[j].offset_weeks - ordered[i].offset_weeks
                if gap < min_interval_weeks:
                    continue
                # Per-sequence count: does this patient have the sequence at
                # both ends of some qualifying interval?
                for col in SEQUENCE_COLUMNS:
                    if ordered[i].has(col) and ordered[j].has(col):
                        per_sequence[col].add(pid)
                if ordered[i].has(sequence) and ordered[j].has(sequence):
                    patient_had_pair = True
                    n_usable += 1
                    intervals.append(gap)
                    if ordered[i].machine and ordered[i].machine == ordered[j].machine:
                        n_same += 1
                    else:
                        n_cross += 1
                        patient_had_cross = True
        if patient_had_pair:
            patients_with_pair.add(pid)
            (patients_cross if patient_had_cross else patients_only_same).add(pid)

    notes = [
        "counts describe what EXISTS in the manifest, not what is usable: the "
        "images are not downloaded, so no registration, artefact or "
        "comparability check has been performed",
        f"{len(patients_cross)} patient(s) have at least one pair spanning two "
        "scanners, a confound for a delta-volume measurement; "
        f"{len(patients_only_same)} have same-scanner pairs only",
        "CFB-GBM is not served by the NBIA API this repo probes "
        "(getSeries/getPatient return an empty body and the collection is "
        "absent from getCollectionValues), so activating stream 4 needs the "
        "collection downloaded rather than queried",
    ]

    return AvailabilityCensus(
        source=source,
        n_rows=len(rows),
        n_patients=len(by_patient),
        n_timepoints_by_label=labels,
        n_patients_multi_timepoint=n_patients_multi,
        sequence=sequence,
        min_interval_weeks=min_interval_weeks,
        n_usable_pairs=n_usable,
        n_patients_with_usable_pair=len(patients_with_pair),
        n_same_machine_pairs=n_same,
        n_cross_machine_pairs=n_cross,
        n_patients_only_same_machine=len(patients_only_same),
        n_patients_with_cross_machine_pair=len(patients_cross),
        interval_weeks_min=min(intervals) if intervals else None,
        interval_weeks_median=_median(intervals),
        interval_weeks_max=max(intervals) if intervals else None,
        n_machines=machines,
        n_patients_per_sequence={c: len(s) for c, s in per_sequence.items()},
        notes=notes,
    )


def main() -> int:
    """Print the census for the committed manifest (no network, no download)."""
    census = mri_availability_census()
    print(census.describe())
    print()
    print("stream-4 activation: a pair above is required; images are NOT cached.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
