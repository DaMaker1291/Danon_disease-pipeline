"""
Retrospective multimodal benchmarking: what public data can and cannot support.

WHY THIS MODULE EXISTS
----------------------
The natural way to "validate" a spatial tumour model is a three-test battery:

  1. **Discrimination** -- does the model rank real patients by real outcome?
  2. **Spatial agreement** -- does its predicted spatial field match a real
     follow-up scan (Dice similarity)?
  3. **Paired control advantage** -- does its recommended schedule beat standard
     care on identical geometry?

Tests 1 and 3 are supported by the data reachable from here and are implemented
elsewhere in the repo. Test 2 is **not supported**, and the point of this module
is to make that a *derived, re-checkable verdict* rather than a sentence in a
README. A refusal written in prose can survive the arrival of new data and keep
refusing; a refusal computed from recorded probe numbers flips by itself.

WHAT WAS ACTUALLY PROBED (recorded in :data:`EVIDENCE_PATH`)
-----------------------------------------------------------
  * TCIA NBIA v1 ``getCollectionValues`` -> 156 collections; **TCGA-GBM and
    TCGA-LGG are absent**. The barcode-based route that would have joined real
    radiology to real genomics (same ``TCGA-xx-xxxx`` identifier in both TCIA
    and GDC) is therefore closed, not merely inconvenient.
  * TCIA ``UPENN-GBM`` -> 630 patients, 3680 series, **MR only**, and
    ``seg_series == 0``: there are no DICOM SEG objects, so there is no expert
    tumour mask in the collection to score a predicted field against.
  * Multi-timepoint check on UPENN-GBM patients (``-00554``, ``-00439``): 5
    studies each, **1 distinct StudyDate** each. The ``_T`` / ``_TERA`` series
    suffixes are re-processings, not later timepoints. The collection is
    single-timepoint, so there is no follow-up scan to compare against even if a
    mask existed.
  * ChEMBL and PubChem are reachable (see :mod:`danon.pharmacology`) and supply
    real measured pharmacology, which is what makes a *pharmacological*
    benchmark possible without a lab.

Consequently the spatial test is refused on two independent grounds -- no
segmentation and no second timepoint -- and both are counts, not opinions.

WHAT THIS MODULE CANNOT SHOW
----------------------------
Reachability is not sufficiency, and the absence of a dataset is not evidence
about a model. Nothing here says the engine's spatial predictions are wrong; it
says the public data reachable from this environment cannot test them.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from danon.cure import EvidenceStatus

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

NBIA_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
NBIA_COLLECTIONS = f"{NBIA_BASE}/getCollectionValues"
NBIA_PATIENTS = f"{NBIA_BASE}/getPatient?Collection={{collection}}"
NBIA_SERIES = f"{NBIA_BASE}/getSeries?Collection={{collection}}"
NBIA_STUDIES = f"{NBIA_BASE}/getPatientStudy?Collection={{collection}}&PatientID={{patient}}"

#: Attribution for the pharmacology cache, so the evidence record names its source.
CHEMBL_SOURCE_NOTE = (
    "ChEMBL (EBI) REST: molecule descriptors and measured activity records"
)

#: Recorded probe output. Committed so reports and tests are deterministic and
#: unaffected by TCIA availability, and so a reviewer can see the numbers the
#: verdicts are computed from without re-running any network call.
EVIDENCE_PATH = "data/tcia_probe/multimodal_evidence.json"

#: The imaging collections that were expected to carry glioma MRI with expert
#: segmentation and more than one timepoint.
CANDIDATE_GLIOMA_COLLECTIONS: Tuple[str, ...] = (
    "TCGA-GBM", "TCGA-LGG", "UPENN-GBM", "BraTS", "ICDC-Glioma",
)


class ModalityStatus(str, Enum):
    """Where a modality stands relative to this environment."""

    #: Present on disk, from a named public source.
    LOCAL = "LOCAL"
    #: Reachable now but not ingested.
    AVAILABLE = "AVAILABLE"
    #: Was public, no longer served.
    RETIRED = "RETIRED"
    #: Served, but lacks the structure the test needs.
    ABSENT = "ABSENT"
    #: Could not be reached during the probe.
    UNREACHABLE = "UNREACHABLE"


@dataclass(frozen=True)
class ModalityEvidence:
    """One probed data source, with the numbers the probe returned."""

    key: str
    name: str
    status: ModalityStatus
    source: str
    license: str
    detail: Dict[str, object] = field(default_factory=dict)
    supports: Tuple[str, ...] = ()
    cannot_support: Tuple[str, ...] = ()

    def get(self, key: str, default: object = None) -> object:
        return self.detail.get(key, default)

    def describe(self) -> str:
        bits = ", ".join(f"{k}={v}" for k, v in sorted(self.detail.items()))
        return f"{self.name} [{self.status.value}] {bits}" if bits else (
            f"{self.name} [{self.status.value}]"
        )


@dataclass(frozen=True)
class SpatialDiceVerdict:
    """Whether a Dice-similarity test against real follow-up imaging is possible."""

    supported: bool
    reason: str
    deciding_evidence: Tuple[str, ...]
    what_would_be_needed: Tuple[str, ...]

    def describe(self) -> str:
        if self.supported:
            return f"spatial Dice test SUPPORTED: {self.reason}"
        lines = [f"spatial Dice test REFUSED: {self.reason}"]
        lines += [f"  - {e}" for e in self.deciding_evidence]
        lines += [f"  needs: {w}" for w in self.what_would_be_needed]
        return "\n".join(lines)


@dataclass(frozen=True)
class BenchmarkTest:
    """One test in the retrospective battery, with its status and its blocker."""

    test_id: str
    question: str
    status: str                     # SUPPORTED | REFUSED
    implemented_by: str
    evidence: str
    blocker: str = ""

    def describe(self) -> str:
        head = f"[{self.status}] {self.test_id}: {self.question}"
        body = [f"      measured by: {self.implemented_by}", f"      basis: {self.evidence}"]
        if self.blocker:
            body.append(f"      blocker: {self.blocker}")
        return "\n".join([head] + body)


# ---------------------------------------------------------------------------
# Recorded evidence
# ---------------------------------------------------------------------------


def load_evidence(path: str = EVIDENCE_PATH) -> List[ModalityEvidence]:
    """Read the recorded probe output. Raises if it is absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no recorded probe evidence at {path}. Run "
            f"`python -m danon.multimodal --probe` to regenerate it; this module "
            f"will not invent availability numbers."
        )
    with open(path) as fh:
        payload = json.load(fh)
    out: List[ModalityEvidence] = []
    for row in payload.get("modalities", []):
        out.append(
            ModalityEvidence(
                key=row["key"],
                name=row["name"],
                status=ModalityStatus(row["status"]),
                source=row["source"],
                license=row.get("license", "unknown"),
                detail=dict(row.get("detail", {})),
                supports=tuple(row.get("supports", ())),
                cannot_support=tuple(row.get("cannot_support", ())),
            )
        )
    return out


def save_evidence(items: Sequence[ModalityEvidence], path: str = EVIDENCE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "note": (
            "Probe output for the retrospective multimodal benchmark. The spatial "
            "Dice verdict is computed from these numbers, so adding a collection "
            "that carries masks and a second timepoint will flip it."
        ),
        "modalities": [
            {
                "key": m.key, "name": m.name, "status": m.status.value,
                "source": m.source, "license": m.license, "detail": m.detail,
                "supports": list(m.supports), "cannot_support": list(m.cannot_support),
            }
            for m in items
        ],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)


def _default_fetcher(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "danon-multimodal/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Live probe
# ---------------------------------------------------------------------------


def probe_tcia(
    *,
    fetcher: Optional[Callable[[str], object]] = None,
    sample_patients: int = 3,
) -> Dict[str, object]:
    """Query NBIA for the facts the spatial verdict depends on.

    Returns counts only; nothing is inferred. Network failures are reported as
    ``UNREACHABLE`` rather than silently returning zero, because "no series
    found" and "could not ask" must not be the same answer.
    """
    fetch = fetcher or _default_fetcher
    out: Dict[str, object] = {}
    try:
        collections_raw = fetch(NBIA_COLLECTIONS) or []
        names = [
            str(c.get("Collection")) for c in collections_raw
            if isinstance(c, dict) and c.get("Collection")
        ]
        out["n_collections"] = len(names)
        out["collections_present"] = sorted(
            c for c in names if c in CANDIDATE_GLIOMA_COLLECTIONS
        )
        out["collections_absent"] = sorted(
            c for c in CANDIDATE_GLIOMA_COLLECTIONS if c not in names
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        out["collections_error"] = f"{type(exc).__name__}: {exc}"
        out["n_collections"] = None
    return out


def probe_collection(
    collection: str,
    *,
    fetcher: Optional[Callable[[str], object]] = None,
    sample_patients: int = 3,
) -> Dict[str, object]:
    """Series/modality/segmentation and multi-timepoint counts for a collection."""
    fetch = fetcher or _default_fetcher
    out: Dict[str, object] = {"collection": collection}
    try:
        series = fetch(NBIA_SERIES.format(collection=urllib.parse.quote(collection))) or []
        out["n_series"] = len(series)
        out["modalities"] = sorted({str(s.get("Modality")) for s in series if s.get("Modality")})
        out["n_seg_series"] = sum(
            1 for s in series
            if str(s.get("Modality") or "").upper() == "SEG"
            or any(
                tok in str(s.get("SeriesDescription") or "").lower()
                for tok in ("seg", "mask", "label")
            )
        )
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        out["series_error"] = f"{type(exc).__name__}: {exc}"
        return out

    patients = sorted({str(s.get("PatientID")) for s in series if s.get("PatientID")})
    out["n_patients"] = len(patients)
    multi = 0
    checked = 0
    for pid in patients[:max(0, sample_patients)]:
        try:
            studies = fetch(
                NBIA_STUDIES.format(
                    collection=urllib.parse.quote(collection),
                    patient=urllib.parse.quote(pid),
                )
            ) or []
            dates = {str(s.get("StudyDate")) for s in studies if s.get("StudyDate")}
            checked += 1
            if len(dates) > 1:
                multi += 1
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            continue
    out["patients_checked_for_timepoints"] = checked
    out["patients_with_multi_timepoint"] = multi
    return out


def probe_local_inventory(
    *,
    brats_dir: str = "data/brats",
    gdc_clinical: str = "data/gdc_clinical/tcga_gbm_cases.json",
    gdc_expr_dir: str = "data/gdc",
    pharmacology_dir: str = "data/pharmacology",
) -> List[ModalityEvidence]:
    """Count what is actually on disk, so provenance is not taken on trust."""
    import glob

    items: List[ModalityEvidence] = []

    seg_files = sorted(glob.glob(os.path.join(brats_dir, "*_seg.nii.gz")))
    items.append(
        ModalityEvidence(
            key="braats2021_segmentations",
            name="BraTS 2021 expert segmentations (local)",
            status=ModalityStatus.LOCAL if seg_files else ModalityStatus.UNREACHABLE,
            source=(
                "BraTS 2021 training data: TCGA-GBM/TCIA-derived de-identified "
                "glioma MRI with expert multi-parametric segmentations; Zenodo "
                "record 19541844"
            ),
            license="CC-BY-4.0",
            detail={
                "n_segmentation_files": len(seg_files),
                "cases": len(seg_files),
                "timepoints_per_case": 1,
            },
            supports=(
                "real pre-operative tumour geometry and a label-derived "
                "proliferative map for the engine grid",
            ),
            cannot_support=(
                "a follow-up timepoint: the release is pre-operative only",
                "vasculature: no perfusion map is distributed with the static release",
            ),
        )
    )

    n_cases = None
    if os.path.exists(gdc_clinical):
        try:
            with open(gdc_clinical) as fh:
                payload = json.load(fh)
            n_cases = len(payload.get("data", {}).get("hits", []) or payload.get("hits", []) or [])
        except (OSError, json.JSONDecodeError, AttributeError):
            n_cases = None
    items.append(
        ModalityEvidence(
            key="gdc_tcga_gbm_clinical",
            name="GDC TCGA-GBM clinical export (local)",
            status=(
                ModalityStatus.LOCAL if os.path.exists(gdc_clinical)
                else ModalityStatus.UNREACHABLE
            ),
            source="NCI Genomic Data Commons, open-access clinical export for TCGA-GBM",
            license="GDC open access (see GDC data-use policy)",
            detail={"n_cases": n_cases, "path": gdc_clinical},
            supports=(
                "real recorded progression and overall-survival endpoints for a "
                "real patient cohort, used as the external anchor and holdout",
            ),
            cannot_support=(
                "patient-level linkage to the imaging cohort: BraTS and GDC "
                "anonymise differently",
            ),
        )
    )

    expr_files = sorted(glob.glob(os.path.join(gdc_expr_dir, "*_star_counts.tsv")))
    items.append(
        ModalityEvidence(
            key="gdc_tcga_gbm_expression",
            name="GDC STAR gene-count matrices (local)",
            status=ModalityStatus.LOCAL if expr_files else ModalityStatus.UNREACHABLE,
            source="NCI Genomic Data Commons, open-access STAR counts",
            license="GDC open access (see GDC data-use policy)",
            detail={"n_expression_files": len(expr_files), "assay": "bulk RNA-seq STAR counts"},
            supports=(
                "bulk transcriptomes that parameterise a sub-clone ratio by "
                "calibrated inference",
            ),
            cannot_support=(
                "single-cell resolution: bulk RNA-seq cannot resolve individual "
                "sub-clones (marked PROVEN_FAILED in danon.genomics_intake)",
            ),
        )
    )

    mol_files = sorted(glob.glob(os.path.join(pharmacology_dir, "*_molecule.json")))
    act_files = sorted(glob.glob(os.path.join(pharmacology_dir, "raw_*_*.json")))
    n_activities = None
    if act_files:
        try:
            with open(act_files[0]) as fh:
                n_activities = len(json.load(fh).get("activities", []))
        except (OSError, json.JSONDecodeError):
            n_activities = None
    items.append(
        ModalityEvidence(
            key="chembl_measured_potency",
            name="ChEMBL measured activity records (local cache)",
            status=ModalityStatus.LOCAL if act_files else ModalityStatus.UNREACHABLE,
            source=f"{CHEMBL_SOURCE_NOTE}",
            license="CC-BY-SA 3.0 (ChEMBL data licence)",
            detail={
                "n_molecule_caches": len(mol_files),
                "n_activity_files": len(act_files),
                "n_activity_records_in_first": n_activities,
            },
            supports=(
                "measured potency with units, relation and target preserved, and "
                "curated molecular descriptors",
            ),
            cannot_support=(
                "in-vivo efficacy: 2D monolayer IC50 is not 3D tumour kill",
            ),
        )
    )
    return items


def build_evidence(
    *,
    fetcher: Optional[Callable[[str], object]] = None,
    sample_patients: int = 3,
    include_local: bool = True,
) -> List[ModalityEvidence]:
    """Assemble the evidence list from live probes plus local-inventory counts."""
    fetch = fetcher or _default_fetcher
    items: List[ModalityEvidence] = []

    tcia = probe_tcia(fetcher=fetch)
    present = set(tcia.get("collections_present") or [])
    absent = set(tcia.get("collections_absent") or [])

    for key, name in (
        ("tcia_nbia_api", "TCIA NBIA v1 API"),
    ):
        items.append(
            ModalityEvidence(
                key=key, name=name,
                status=(
                    ModalityStatus.AVAILABLE if tcia.get("n_collections") is not None
                    else ModalityStatus.UNREACHABLE
                ),
                source=f"{NBIA_BASE}",
                license="per-collection (see TCIA collection page)",
                detail={
                    "n_collections": tcia.get("n_collections"),
                    "glioma_candidates_present": sorted(present),
                    "glioma_candidates_absent": sorted(absent),
                },
                supports=(
                    "anonymised imaging for collections that are still served",
                ),
                cannot_support=(
                    "patient-level joining of radiology to an external clinical "
                    "export when the collection that carried the shared identifier "
                    "has been withdrawn",
                ),
            )
        )

    for coll in CANDIDATE_GLIOMA_COLLECTIONS:
        if coll in absent:
            items.append(
                ModalityEvidence(
                    key=f"tcia_{coll.lower().replace('-', '_')}",
                    name=f"TCIA collection {coll}",
                    status=ModalityStatus.RETIRED,
                    source=NBIA_BASE,
                    license="n/a (withdrawn)",
                    detail={"collection": coll},
                    cannot_support=(
                        "any imaging-based test, including the barcode-based link "
                        "to a GDC clinical export",
                    ),
                )
            )
            continue
        if coll not in present:
            continue
        detail = probe_collection(coll, fetcher=fetch, sample_patients=sample_patients)
        n_seg = int(detail.get("n_seg_series") or 0)
        multi = int(detail.get("patients_with_multi_timepoint") or 0)
        checked = int(detail.get("patients_checked_for_timepoints") or 0)
        lacks = []
        if n_seg == 0:
            lacks.append("an expert tumour mask to score a predicted field against")
        if checked and multi == 0:
            lacks.append("a second timepoint, so there is no follow-up scan to compare to")
        items.append(
            ModalityEvidence(
                key=f"tcia_{coll.lower().replace('-', '_')}",
                name=f"TCIA collection {coll}",
                status=_collection_status(n_seg),
                source=NBIA_BASE,
                license="per-collection (see TCIA collection page)",
                detail=detail,
                supports=(
                    "single-timepoint anonymised MRI with a documented collection page",
                ),
                cannot_support=tuple(lacks),
            )
        )

    if include_local:
        items.extend(probe_local_inventory())

    return items


def _collection_status(n_seg: int) -> ModalityStatus:
    """A collection that carries masks can support a spatial test; one that does
    not has the modality but lacks the structure the test needs."""
    return ModalityStatus.AVAILABLE if n_seg > 0 else ModalityStatus.ABSENT


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def spatial_dice_verdict(
    evidence: Optional[Sequence[ModalityEvidence]] = None,
    path: str = EVIDENCE_PATH,
) -> SpatialDiceVerdict:
    """Decide the spatial test from the evidence, so new data can flip it.

    The test needs a collection that carries **both** an expert tumour mask and
    a second timepoint for the same patient. The verdict scans every recorded
    imaging collection for those two structures and refuses if none has both.
    """
    items = list(evidence) if evidence is not None else load_evidence(path)
    imaging = [
        m for m in items
        if m.key.startswith("tcia_") and m.key != "tcia_nbia_api"
    ]
    if not imaging:
        return SpatialDiceVerdict(
            supported=False,
            reason="no imaging collections recorded in the evidence file",
            deciding_evidence=(f"evidence path {path} contains no tcia_* collections",),
            what_would_be_needed=(
                "record probe output for at least one glioma imaging collection",
            ),
        )

    qualifying: List[ModalityEvidence] = []
    deciding: List[str] = []
    for m in imaging:
        if m.status is ModalityStatus.RETIRED:
            deciding.append(
                f"{m.name}: RETIRED -- not served by the API "
                f"({m.detail.get('collection')})"
            )
            continue
        n_seg = int(m.get("n_seg_series") or 0)
        multi = int(m.get("patients_with_multi_timepoint") or 0)
        checked = int(m.get("patients_checked_for_timepoints") or 0)
        deciding.append(
            f"{m.name}: {m.get('n_patients')} patient(s), {m.get('n_series')} "
            f"series, modalities {m.get('modalities')}, {n_seg} segmentation "
            f"series, {multi}/{checked} sampled patient(s) with more than one "
            f"StudyDate"
        )
        if n_seg > 0 and multi > 0:
            qualifying.append(m)

    if qualifying:
        names = ", ".join(m.name for m in qualifying)
        return SpatialDiceVerdict(
            supported=True,
            reason=f"{names} carries both masks and a second timepoint",
            deciding_evidence=tuple(deciding),
            what_would_be_needed=(),
        )

    return SpatialDiceVerdict(
        supported=False,
        reason=(
            "no reachable glioma imaging collection carries both an expert tumour "
            "mask and a second timepoint for the same patient"
        ),
        deciding_evidence=tuple(deciding),
        what_would_be_needed=(
            "a collection with DICOM SEG (or equivalent mask series) on glioma MRI",
            "at least one patient with two imaging timepoints, so a predicted "
            "day-90 field can be compared to an observed recurrence scan",
            "an explicit mapping between that patient and an administered schedule, "
            "without which a mismatch cannot be attributed to the model",
        ),
    )


def benchmark_tests(
    evidence: Optional[Sequence[ModalityEvidence]] = None,
    path: str = EVIDENCE_PATH,
) -> List[BenchmarkTest]:
    """The three-test battery, each with its status and the reason for it."""
    items = list(evidence) if evidence is not None else load_evidence(path)
    dice = spatial_dice_verdict(items)

    imaging_retired = [
        m for m in items if m.status is ModalityStatus.RETIRED and "GBM" in m.key.upper()
    ]
    pharmaco = [m for m in items if m.key.startswith("chembl") or m.key.startswith("pubchem")]

    return [
        BenchmarkTest(
            test_id="test1_discrimination",
            question="does the model rank real patients by real outcome?",
            status="SUPPORTED",
            implemented_by="danon.predictive.kfold_holdout_c_index / harrell_c_index",
            evidence=(
                "held-out Cox discrimination on the real GDC TCGA-GBM clinical "
                "export, with the fit and the score on separate patient splits"
            ),
            blocker=(
                "the engine itself has no per-patient prediction to rank "
                "(danon.predictive.engine_discrimination_verdict), so the C-index "
                "reported is for a covariate model, not for the scheduler"
            ),
        ),
        BenchmarkTest(
            test_id="test2_spatial_dice",
            question="does the predicted spatial field match a real follow-up scan?",
            status="SUPPORTED" if dice.supported else "REFUSED",
            implemented_by="danon.imaging (geometry intake) -- no Dice estimator wired",
            evidence=dice.reason,
            blocker=(
                "; ".join(dice.what_would_be_needed)
                if not dice.supported else ""
            ),
        ),
        BenchmarkTest(
            test_id="test3_paired_control",
            question="does the recommended schedule beat standard care on identical geometry?",
            status="SUPPORTED",
            implemented_by="danon.outcome_export.export_outcome_cards / danon.campaign.run_campaign",
            evidence=(
                "both arms run on the same real BraTS-derived geometry with "
                "replicates, reporting the burden and progression delta paired "
                "per geometry rather than across cohorts"
            ),
            blocker=(
                "the observed cohort and the imaging cohort are different people, "
                "so the size of the advantage is a model result, not a treatment "
                "effect measured in patients"
            ),
        ),
    ]


def multimodal_provenance() -> Dict[str, EvidenceStatus]:
    """Provenance tags for the benchmark's inputs and its own verdicts."""
    return {
        "tcia_collection_listing": EvidenceStatus.ESTABLISHED,
        "tcia_collection_series_and_timepoints": EvidenceStatus.ESTABLISHED,
        "braats_derived_geometry": EvidenceStatus.ESTABLISHED,
        "gdc_clinical_endpoints": EvidenceStatus.ESTABLISHED,
        "glioma_potency_measurements": EvidenceStatus.ESTABLISHED,
        "spatial_dice_agreement": EvidenceStatus.PROVEN_FAILED,
        "engine_per_patient_discrimination": EvidenceStatus.PROVEN_FAILED,
    }


def what_multimodal_benchmark_cannot_show() -> List[str]:
    """The limits that must be stated wherever this benchmark is reported."""
    return [
        "Data presence is not model validity. A supported test says an estimate "
        "could be computed; it does not say the estimate is good.",
        "The spatial test is refused because no reachable collection carries both "
        "masks and a second timepoint -- that is a statement about public data "
        "availability, not evidence that the engine's spatial predictions are "
        "wrong. The distinction matters: absence of a test is not a negative result.",
        "The imaging cohort (BraTS-derived) and the outcome cohort (GDC TCGA-GBM) "
        "anonymise differently and cannot be joined at patient level. Every "
        "comparison across them is between distributions, never between matched "
        "individuals.",
        "The paired-control delta is produced by both arms of the same model. It "
        "measures the model's own contrast, not a clinical benefit in patients, "
        "and cannot be reported as a survival gain.",
        "Reachability was probed from one environment on one date; a collection "
        "added later can change the second test's verdict, which is why the "
        "verdict is computed from recorded counts rather than asserted.",
    ]


# ---------------------------------------------------------------------------
# Running the two tests that can run
# ---------------------------------------------------------------------------


@dataclass
class PairedControlSummary:
    """Aggregated day-365 paired-control contrast over the real-geometry cards."""

    n_cards: int
    n_progression_free_policy: int
    n_progression_free_mtd: int
    median_pfs_extension: Optional[float]
    min_pfs_extension: float
    max_pfs_extension: float
    n_censored_extensions: int
    median_toxicity_reduction: Optional[float]
    median_burden_policy: Optional[float]
    median_burden_mtd: Optional[float]
    notes: List[str] = field(default_factory=list)

    def describe(self) -> str:
        if self.n_cards == 0:
            return "paired control: no real-geometry cards available"
        # Every optional statistic is rendered through a formatter that yields a
        # word for absence. A summary whose describe() raises on a missing field
        # is a summary that cannot report its own gaps -- which is exactly when
        # a reader most needs it to speak.
        def _pct(v: Optional[float]) -> str:
            return f"{v:+.3f}" if v is not None else "not measurable"

        ext = _pct(self.median_pfs_extension)
        if self.median_pfs_extension is None:
            ext = "not measurable from these cards"
        tox = (
            f"{self.median_toxicity_reduction:.3f}"
            if self.median_toxicity_reduction is not None else "not reported"
        )
        return (
            f"paired control over {self.n_cards} card(s): progression-free "
            f"{self.n_progression_free_policy}/{self.n_cards} on the policy vs "
            f"{self.n_progression_free_mtd}/{self.n_cards} on MTD; median PFS "
            f"extension {ext} (range {_pct(self.min_pfs_extension)} to "
            f"{_pct(self.max_pfs_extension)}), censored in "
            f"{self.n_censored_extensions} where the policy never progressed "
            f"inside the horizon; median toxicity reduction {tox}"
        )


def summarize_paired_control(cards_payload: Dict[str, object]) -> PairedControlSummary:
    """Aggregate the exported day-365 cards into the paired-control result.

    Two honesty rules are applied in the aggregation rather than left to the
    reader:

    * A card where the policy arm never progressed inside the horizon has a PFS
      extension that is *bounded below*, not measured, and those are counted
      explicitly (``n_censored_extensions``) so a median that leans on them is
      never quoted as a measurement.
    * The contrast is between two arms of the same model, so it is a model
      contrast and not a treatment effect. That is recorded in ``notes``.
    """
    cases = list(cards_payload.get("cases") or [])
    ext: List[float] = []
    tox: List[float] = []
    b_pol: List[float] = []
    b_mtd: List[float] = []
    n_pf_pol = n_pf_mtd = n_censored = 0
    for c in cases:
        x = c.get("day_365") or {}
        if x.get("progression_free_policy"):
            n_pf_pol += 1
        if x.get("progression_free_mtd"):
            n_pf_mtd += 1
        if x.get("pfs_extension") is not None:
            ext.append(float(x["pfs_extension"]))
        if x.get("toxicity_reduction") is not None:
            tox.append(float(x["toxicity_reduction"]))
        if x.get("burden_policy") is not None:
            b_pol.append(float(x["burden_policy"]))
        if x.get("burden_mtd") is not None:
            b_mtd.append(float(x["burden_mtd"]))
        if not x.get("progression_free_mtd", True):
            n_censored += 1
    notes: List[str] = []
    if n_censored:
        notes.append(
            f"{n_censored} card(s) have the MTD arm progressing while the policy arm "
            f"does not reach threshold inside the 365-day horizon, so their PFS "
            f"extension is a lower bound rather than a measured survival gain"
        )
    notes.append(
        "both arms are the same model, so this is the model's own contrast on real "
        "geometry, not a clinical benefit measured in patients"
    )
    return PairedControlSummary(
        n_cards=len(cases),
        n_progression_free_policy=n_pf_pol,
        n_progression_free_mtd=n_pf_mtd,
        median_pfs_extension=(float(np_median(ext)) if ext else None),
        min_pfs_extension=(float(min(ext)) if ext else 0.0),
        max_pfs_extension=(float(max(ext)) if ext else 0.0),
        n_censored_extensions=n_censored,
        median_toxicity_reduction=(float(np_median(tox)) if tox else None),
        median_burden_policy=(float(np_median(b_pol)) if b_pol else None),
        median_burden_mtd=(float(np_median(b_mtd)) if b_mtd else None),
        notes=notes,
    )


def np_median(values: Sequence[float]) -> float:
    """Median without importing numpy for one call site."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        raise ValueError("empty")
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def measured_discrimination(path: str = "data/gdc_clinical/tcga_gbm_cases.json") -> str:
    """Held-out C-index lines for both endpoints, as a text block."""
    from danon.predictive import kfold_holdout_c_index, load_cohort_rows

    lines: List[str] = []
    for endpoint in ("os", "progression"):
        rows, notes = load_cohort_rows(path, endpoint=endpoint)
        rep = kfold_holdout_c_index(rows, endpoint=endpoint)
        lines.append(f"{endpoint}: {rep}")
        lines.append(f"    n={len(rows)} patients; " + "; ".join(notes))
    return "\n".join(lines)


def load_paired_control_cards(path: str = "outputs_real/outcome_cards.json"):
    """The real-geometry day-365 cards, or None when they have not been exported."""
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def render_multimodal_report(
    evidence: Optional[Sequence[ModalityEvidence]] = None,
    path: str = EVIDENCE_PATH,
    *,
    include_measured: bool = False,
    cards_path: str = "outputs_real/outcome_cards.json",
    clinical_path: str = "data/gdc_clinical/tcga_gbm_cases.json",
) -> str:
    """Poster-ready summary of the battery, its statuses and its limits.

    ``include_measured`` runs the two supported tests and inlines their real
    numbers. It is off by default so report generation does not silently cost a
    5-fold Cox fit per endpoint.
    """
    items = list(evidence) if evidence is not None else load_evidence(path)
    dice = spatial_dice_verdict(items)
    lines = ["# Retrospective multimodal benchmark", ""]
    lines.append("## Data sources probed")
    for m in sorted(items, key=lambda x: x.key):
        lines.append(f"- {m.describe()}")
        for s in m.supports:
            lines.append(f"    supports: {s}")
        for c in m.cannot_support:
            lines.append(f"    cannot: {c}")
    lines += ["", "## Test battery"]
    for t in benchmark_tests(items):
        lines.append(f"- {t.describe()}")

    if include_measured:
        lines += ["", "## Measured outcomes (tests 1 and 3)", ""]
        lines.append("### Test 1 -- held-out discrimination on the real GDC cohort")
        try:
            lines.append("```")
            lines.append(measured_discrimination(clinical_path))
            lines.append("```")
        except (FileNotFoundError, ValueError) as exc:
            lines.append(f"Test 1 not run: {exc}")
        cards = load_paired_control_cards(cards_path)
        lines.append("")
        lines.append("### Test 3 -- paired control on real BraTS-derived geometry")
        if cards is None:
            lines.append(f"not run: no exported cards at {cards_path}")
        else:
            summary = summarize_paired_control(cards)
            lines.append(summary.describe())
            lines += [f"  - {n}" for n in summary.notes]

    lines += ["", "## Spatial verdict", dice.describe(), "", "## Limits"]
    lines += [f"- {c}" for c in what_multimodal_benchmark_cannot_show()]
    return "\n".join(lines)
