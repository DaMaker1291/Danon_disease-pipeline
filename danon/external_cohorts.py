"""
External cohorts: five proposed online-only verification pipelines, and what
each can actually bear.

WHY THIS MODULE EXISTS
----------------------
A pitch proposes five purely-online verifications, each named dataset and each
with a target number attached: an out-of-distribution C-index above 0.62 on
CGGA, a Jensen-Shannon divergence below 0.12 against single-cell state
proportions, an R^2 of 0.93 or better against GDSC dose response, an inverse
tensor recovery above 0.999 on a BrainWeb phantom, and a Dice of 0.88 or better
against BraTS sub-regions. This module does the same thing for that pitch that
`danon/tumor_board.py` does for a dashboard mock-up: it binds each pipeline to
data that exists, measures the one that can be measured here, and refuses the
rest in writing, carrying the proposed figure as the thing being refused.

ADULT-IN-THE-ROOM
-----------------
None of the five, measured, says anything about the treatment engine. The
engine has no estimable patient-level C-index (`check_engine_c_index_is_not_claimed`
exists for that reason). An external C-index on age, sex and diagnosis -- which
is what pipeline 1 can produce -- transports a *covariate model*; it does not
validate a 3D simulation, and a single external cohort cannot prove "zero
overfitting". That sentence is carried in the record's notes verbatim so the
claim check can require it.

PIPELINE 1 -- MEASURED HERE.
CGGA (Chinese Glioma Genome Atlas) publishes its clinical tables as direct
downloads (no login). Two releases are cached under `data/cgga/`: mRNAseq_325
and mRNAseq_693, both 2020-05-06. The cohort is read, the same Cox model the
repository fits on TCGA-GBM (`age_years`, `male`, `is_glioblastoma`; see
`danon/predictive.py`) is fitted on TCGA once and then applied to CGGA
**without changing a coefficient**. What that yields is recorded, not chosen:
whatever C-index the transport gives is the number.

Two corrections to the pitch's use of this dataset follow from the file itself:
the release named "325" holds 325 clinical rows of which 85 carry `Histology
== "GBM"`, so "n=325 Glioblastoma" is not what the file contains; and the
larger 693-row release exists. Both files are scored here so the correction is
visible rather than asserted.

PIPELINES 2-5 -- REFUSED, WITH THE REASON NAMED.
Not because open data is unavailable, but because the metric has nothing to
attach to: there is no engine quantity that is a four-state cell composition
(and single-cell inference from bulk is already refused by
`danon/genomics_intake.py`); the engine's EC50 has no declared unit, so an R^2
against a molar IC50 is a type error, and 34 of 34 temozolomide IC50s in GDSC2
are extrapolations beyond the highest concentration tested
(`danon/eval_targets.py`); the repository has no spatial inverse-tensor
estimator, and injecting a tensor and recovering it is a self-consistency
check, not a validation against measured tissue; and the Dice test needs an
expert mask plus a second timepoint for the same patient, which the live
`spatial_dice_verdict` says no probed collection carries.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not download anything at import or check time. The cohort script
`_cgga_ood_cohort.py` fetches the two files and writes
`outputs_real/cgga_ood.json`; the claim check reads the cache and recomputes.
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.eval_targets import TargetStatus

# ---------------------------------------------------------------------------
# The cached external release
# ---------------------------------------------------------------------------

CGGA_325_ZIP = os.path.join("data", "cgga", "CGGA.mRNAseq_325_clinical.20200506.txt.zip")
CGGA_693_ZIP = os.path.join("data", "cgga", "CGGA.mRNAseq_693_clinical.20200506.txt.zip")

CGGA_SOURCE_325 = (
    "https://www.cgga.org.cn/download?file=download/20200506/"
    "CGGA.mRNAseq_325_clinical.20200506.txt.zip&type=mRNAseq_325_clinical&time=20200506"
)
CGGA_SOURCE_693 = (
    "https://www.cgga.org.cn/download?file=download/20200506/"
    "CGGA.mRNAseq_693_clinical.20200506.txt.zip&type=mRNAseq_693_clinical&time=20200506"
)
CGGA_CITATION = (
    "CGGA: Zhao Z, Zhang KN, Wang Q, et al. Chinese Glioma Genome Atlas (CGGA): "
    "A Comprehensive Resource with Functional Genomic Data from Chinese Glioma "
    "Patients. Genomics Proteomics Bioinformatics. 2021;19(1):1-12. "
    "doi:10.1016/j.gpb.2020.10.005. Clinical releases 2020-05-06."
)

#: Neftel et al. Cell 2019 single-cell cohort. The GEO supplementary directory
#: was listed on 2026-09-30: `GSE131928_RAW.tar` is 638 MB and the cell-to-tumour
#: mapping workbook is 828 KB. The workbook maps cells to tumours, NOT to the
#: four cellular states, so the state fractions the pitch needs are not in the
#: file the pitch points at.
GSE131928_SUPPL_URL = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131928/suppl/"
GSE131928_RAW_BYTES = 668_991_488        # observed listing, 2019-07-08
GSE131928_MAPPING_BYTES = 847_872        # observed listing, 2019-05-29

#: Records written by `_cgga_ood_cohort.py`.
CGGA_OOD_RECORD = os.path.join("outputs_real", "cgga_ood.json")

#: The sentence the claim check requires in the record, so the transport test
#: can never be described as an engine validation.
ENGINE_CAVEAT = (
    "this validates a covariate model on external patients; it is not a "
    "validation of the treatment engine"
)

#: The figures the pitch proposes. Kept here so a correction can name the thing
#: it is replacing, and so no check can pass while one of them is reported as a
#: result.
PITCH_FIGURES: Dict[str, str] = {
    "cgga_ood_c_index": "CGGA out-of-distribution C-index 0.6412",
    "cgga_gbm_n": "CGGA 'n=325 Glioblastoma'",
    "scrna_jsd": "single-cell state-shift JSD < 0.12",
    "in_vitro_r2": "in-vitro IC50 replicator fit R^2 >= 0.938",
    "inverse_recovery_r2": "inverse tensor recovery R^2 >= 0.999",
    "brats_dice": "3D sub-region Dice >= 0.88",
}


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Reading the CGGA clinical table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CggaRow:
    cgga_id: str
    prs_type: str
    histology: str
    grade: str
    age_years: float
    male: int
    os_days: float
    event: int
    idh: str
    mgmt: str

    @property
    def is_gbm(self) -> int:
        return 1 if self.histology == "GBM" else 0


@dataclass
class CggaCohort:
    """One CGGA clinical release as read, with what was dropped and why."""

    path: str
    rows: List[CggaRow] = field(default_factory=list)
    n_table_rows: int = 0
    dropped_no_age: int = 0
    dropped_no_endpoint: int = 0
    dropped_no_sex: int = 0
    sex_values_normalised: int = 0
    #: Histology of the rows that were dropped, so a count of "GBM rows in the
    #: file" can be stated separately from "GBM rows with a usable endpoint".
    dropped_histology: Dict[str, int] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def n_events(self) -> int:
        return sum(r.event for r in self.rows)

    def histology_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.rows:
            counts[r.histology] = counts.get(r.histology, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def prs_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self.rows:
            counts[r.prs_type] = counts.get(r.prs_type, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def primary(self) -> List[CggaRow]:
        return [r for r in self.rows if r.prs_type.lower() == "primary"]

    def primary_gbm(self) -> List[CggaRow]:
        return [r for r in self.primary() if r.is_gbm]


def _read_zip_table(path: str) -> str:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if not n.startswith("__MACOSX") and not n.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"{path}: expected one table in the archive, found {names}")
        return z.read(names[0]).decode("utf-8", "replace")


def _column(header: Sequence[str], *prefixes: str) -> int:
    for i, h in enumerate(header):
        for p in prefixes:
            if h.startswith(p):
                return i
    raise KeyError(f"none of {prefixes} present in CGGA header {list(header)}")


def _num(text: str) -> Optional[float]:
    text = text.strip()
    if not text or text.upper() in ("NA", "N/A", "NAN"):
        return None
    try:
        return float(text)
    except ValueError:
        return None


def read_cgga_clinical(path: str = CGGA_325_ZIP) -> CggaCohort:
    """Read a CGGA clinical release. Nothing is imputed; drops are counted.

    The header is matched by prefix because the censor column is spelled
    `Censor (alive=0; dead=1)` and the sex column carries trailing spaces in
    some rows of the 693 release ("Female ", "Male  "), which a naive split
    leaves in place.
    """
    try:
        raw = _read_zip_table(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"CGGA clinical release not cached at {path}")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f"{path}: no data rows")
    header = lines[0].split("\t")
    i_id = _column(header, "CGGA_ID")
    i_prs = _column(header, "PRS_type")
    i_hist = _column(header, "Histology")
    i_grade = _column(header, "Grade")
    i_sex = _column(header, "Gender")
    i_age = _column(header, "Age")
    i_os = _column(header, "OS")
    i_cen = _column(header, "Censor")
    i_idh = _column(header, "IDH")
    i_mgmt = _column(header, "MGMTp")

    cohort = CggaCohort(path=path, n_table_rows=len(lines) - 1)
    for line in lines[1:]:
        parts = line.split("\t")
        get = lambda i: parts[i].strip() if i < len(parts) else ""  # noqa: E731
        histology = get(i_hist)
        age = _num(get(i_age))
        if age is None:
            cohort.dropped_no_age += 1
            cohort.dropped_histology[histology] = (
                cohort.dropped_histology.get(histology, 0) + 1
            )
            continue
        os_days = _num(get(i_os))
        censor = _num(get(i_cen))
        if os_days is None or censor is None or os_days <= 0:
            cohort.dropped_no_endpoint += 1
            cohort.dropped_histology[histology] = (
                cohort.dropped_histology.get(histology, 0) + 1
            )
            continue
        # Sex is compared BEFORE stripping: the 693 release carries 'Female '
        # and 'Male  ' in a handful of rows, and a reader that silently strips
        # them cannot report that it did.
        sex_raw = parts[i_sex] if i_sex < len(parts) else ""
        sex = sex_raw.strip()
        if sex != sex_raw:
            cohort.sex_values_normalised += 1
        if sex == "Male":
            male = 1
        elif sex == "Female":
            male = 0
        else:
            cohort.dropped_no_sex += 1
            cohort.dropped_histology[histology] = (
                cohort.dropped_histology.get(histology, 0) + 1
            )
            continue
        cohort.rows.append(
            CggaRow(
                cgga_id=get(i_id),
                prs_type=get(i_prs),
                histology=histology,
                grade=get(i_grade),
                age_years=float(age),
                male=male,
                os_days=float(os_days),
                event=1 if censor >= 0.5 else 0,
                idh=get(i_idh),
                mgmt=get(i_mgmt),
            )
        )

    cohort.notes.append(
        f"{cohort.n_table_rows} table rows; {cohort.n} usable; "
        f"{cohort.n_events} events, {cohort.n - cohort.n_events} censored"
    )
    if cohort.dropped_no_age:
        cohort.notes.append(f"{cohort.dropped_no_age} dropped: no age")
    if cohort.dropped_no_endpoint:
        cohort.notes.append(
            f"{cohort.dropped_no_endpoint} dropped: no usable OS/censor"
        )
    if cohort.dropped_no_sex:
        cohort.notes.append(f"{cohort.dropped_no_sex} dropped: unrecognised sex value")
    if cohort.sex_values_normalised:
        cohort.notes.append(
            f"{cohort.sex_values_normalised} sex value(s) carried padding and were "
            f"stripped before mapping"
        )
    cohort.notes.append(
        f"histology counts read from the file: {cohort.histology_counts()}"
    )
    cohort.notes.append(f"PRS_type counts read from the file: {cohort.prs_counts()}")
    return cohort


# ---------------------------------------------------------------------------
# Pipeline 1: the transport test
# ---------------------------------------------------------------------------


@dataclass
class TcgaReference:
    """The frozen model: Cox fit on the cached TCGA cohort, once."""

    n: int
    n_events: int
    names: Tuple[str, ...]
    beta: Tuple[float, ...]
    hazard_ratios: Dict[str, float]
    internal_c: float
    internal_c_sd: float
    notes: List[str] = field(default_factory=list)


@dataclass
class ExternalTest:
    """One subset of one external cohort, scored with frozen coefficients."""

    label: str
    n: int
    n_events: int
    c_index: float
    age_only_c_index: float
    note: str = ""

    @property
    def estimable(self) -> bool:
        return bool(np.isfinite(self.c_index))


def fit_tcga_reference(
    tcga_path: Optional[str] = None,
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> TcgaReference:
    """Fit the repository's Cox model on TCGA and record what it is.

    The fit uses exactly the design `danon.predictive` uses -- age, sex, and a
    glioblastoma dummy -- so the external test is of the model this repository
    already reports, not of a model built for the external test.
    """
    from danon.predictive import (
        GDC_CLINICAL_PATH,
        cox_fit,
        kfold_holdout_c_index,
        load_cohort_rows,
    )

    path = tcga_path or GDC_CLINICAL_PATH
    rows, notes = load_cohort_rows(path, endpoint="os")
    X = np.array([[r.age_years, r.male, r.is_glioblastoma] for r in rows], dtype=float)
    time = np.array([r.time_days for r in rows], dtype=float)
    event = np.array([r.event for r in rows], dtype=int)
    fit = cox_fit(X, time, event, ("age_years", "male", "is_glioblastoma"))
    if not fit.converged:
        raise RuntimeError("TCGA reference fit did not converge")
    holdout = kfold_holdout_c_index(rows, n_folds=n_folds, seed=seed, endpoint="os")
    return TcgaReference(
        n=len(rows),
        n_events=int(event.sum()),
        names=tuple(fit.names),
        beta=tuple(float(b) for b in fit.coefficients),
        hazard_ratios=dict(fit.hazard_ratios),
        internal_c=float(holdout.mean_multivariable),
        internal_c_sd=float(holdout.sd_multivariable),
        notes=list(notes),
    )


def _score(ref: TcgaReference, rows: Sequence[CggaRow], label: str, note: str) -> ExternalTest:
    from danon.predictive import harrell_c_index

    if not rows:
        return ExternalTest(label, 0, 0, float("nan"), float("nan"), "no rows")
    X = np.array(
        [[r.age_years, r.male, r.is_gbm] for r in rows], dtype=float
    )
    time = np.array([r.os_days for r in rows], dtype=float)
    event = np.array([r.event for r in rows], dtype=int)
    beta = np.array(ref.beta, dtype=float)
    risk = X @ beta
    c = harrell_c_index(risk, time, event)
    age_c = harrell_c_index(X[:, 0], time, event)
    return ExternalTest(
        label=label,
        n=len(rows),
        n_events=int(event.sum()),
        c_index=float(c.c_index) if c.estimable else float("nan"),
        age_only_c_index=float(age_c.c_index) if age_c.estimable else float("nan"),
        note=note,
    )


@dataclass
class CggaOodReport:
    reference: TcgaReference
    cohorts: Dict[str, CggaCohort]
    tests: Dict[str, List[ExternalTest]]
    notes: List[str]


def cgga_ood_report(
    paths: Sequence[str] = (CGGA_325_ZIP, CGGA_693_ZIP),
    *,
    tcga_path: Optional[str] = None,
) -> CggaOodReport:
    """Apply the TCGA-fitted model to CGGA, unchanged, and report what it gives.

    Two subsets per file: all primary gliomas (where the glioblastoma dummy is
    informative, and any grade is present) and primary GBM only (the subset the
    pitch names -- but there the dummy is constant, so what is transported is
    age and sex, which the note says).
    """
    ref = fit_tcga_reference(tcga_path=tcga_path)
    cohorts: Dict[str, CggaCohort] = {}
    tests: Dict[str, List[ExternalTest]] = {}
    for path in paths:
        key = os.path.basename(path)
        cohort = read_cgga_clinical(path)
        cohorts[key] = cohort
        primary = cohort.primary()
        primary_gbm = cohort.primary_gbm()
        tests[key] = [
            _score(
                ref,
                primary,
                "primary gliomas, any grade",
                "the glioblastoma dummy is informative here; scores age, sex and "
                "histological grade band",
            ),
            _score(
                ref,
                primary_gbm,
                "primary GBM only",
                "the glioblastoma dummy is constant on this subset, so the "
                "transported score is age and sex",
            ),
        ]
    notes = [
        ENGINE_CAVEAT,
        "no coefficient is refitted on CGGA: the fit is the TCGA cohort's and "
        "is applied unchanged, which is what makes this a transport test rather "
        "than a second internal validation",
        "one external cohort at one metric is evidence of transport, not proof "
        "of 'zero overfitting': a model with no signal would read C~0.5, and a "
        "weak model can transport at the same 0.55-0.65 band",
        "the model carries no treatment covariates; CGGA and TCGA differ in era "
        "and regimen, so this compares cohorts, not treatments",
        "no imaging is linked to these rows, so nothing here is a spatial or "
        "engine validation",
    ]
    return CggaOodReport(reference=ref, cohorts=cohorts, tests=tests, notes=notes)


# ---------------------------------------------------------------------------
# Pipelines 2-5: refusals with the live reason
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExternalPipeline:
    index: int
    name: str
    dataset: str
    source: str
    proposed_metric: str
    status: TargetStatus
    detail: str
    recorded_instead: Optional[str] = None

    @property
    def refused(self) -> bool:
        return self.status is not TargetStatus.MEASURED


def external_pipeline_matrix(cgga_paths: Sequence[str] = (CGGA_325_ZIP, CGGA_693_ZIP)) -> List[ExternalPipeline]:
    """The five pipelines, with pipeline 1 measured and 2-5 refused live."""
    from danon.eval_targets import in_vitro_target
    from danon.genomics_intake import genomics_provenance

    # --- 1. external cohort transport (measured, or refused if uncached) -----
    cached = [p for p in cgga_paths if os.path.exists(p)]
    if cached:
        report = cgga_ood_report(tuple(cached))
        parts = []
        for key, tests in report.tests.items():
            for t in tests:
                parts.append(
                    f"{key} {t.label}: C={t.c_index:.4f} (n={t.n}, "
                    f"{t.n_events} events)"
                )
        pipeline_1 = ExternalPipeline(
            index=1,
            name="External cohort generalization (out of distribution)",
            dataset="CGGA clinical releases mRNAseq_325 / mRNAseq_693",
            source=CGGA_SOURCE_325 + " | " + CGGA_SOURCE_693,
            proposed_metric="out-of-distribution C-index",
            status=TargetStatus.MEASURED,
            detail=(
                "cohorts cached under data/cgga; " + "; ".join(parts) +
                f". Frozen TCGA reference: n={report.reference.n}, "
                f"C={report.reference.internal_c:.4f} +/- "
                f"{report.reference.internal_c_sd:.4f} internally. "
                + ENGINE_CAVEAT
            ),
            recorded_instead=None,
        )
    else:
        pipeline_1 = ExternalPipeline(
            index=1,
            name="External cohort generalization (out of distribution)",
            dataset="CGGA clinical releases mRNAseq_325 / mRNAseq_693",
            source=CGGA_SOURCE_325,
            proposed_metric="out-of-distribution C-index",
            status=TargetStatus.REFUSED_NO_DATA,
            detail=(
                f"the CGGA clinical releases are not cached ({cgga_paths[0]}); "
                f"run _cgga_ood_cohort.py, which downloads them from "
                f"www.cgga.org.cn and writes {CGGA_OOD_RECORD}"
            ),
        )

    # --- 2. single-cell state shift ------------------------------------------
    single_cell_refused = (
        genomics_provenance()["single_cell_clone_fractions_from_bulk"]
        is EvidenceStatus.PROVEN_FAILED
    )
    pipeline_2 = ExternalPipeline(
        index=2,
        name="Single-cell clonal dynamics",
        dataset="GEO GSE131928 (Neftel et al. Cell 2019)",
        source=GSE131928_SUPPL_URL,
        proposed_metric="Jensen-Shannon divergence between predicted and "
        "measured cell-state proportions",
        status=TargetStatus.REFUSED_NO_DATA,
        detail=(
            f"the dataset is reachable ({GSE131928_RAW_BYTES / 1e6:.0f} MB raw "
            f"archive, {GSE131928_MAPPING_BYTES / 1e6:.1f} MB cell-to-tumour "
            f"mapping workbook), but the metric has nothing to attach to: the "
            f"mapping workbook carries cells against tumours, not the four "
            f"cellular states, and no module here predicts a four-state "
            f"composition under therapy. Single-cell clone fractions from bulk "
            f"expression are already PROVEN_FAILED in danon/genomics_intake.py "
            f"({single_cell_refused}); a JSD would require both a state "
            f"classifier and a therapy-selection model this repository does not "
            f"have"
        ),
    )

    # --- 3. in-vitro dose response -------------------------------------------
    in_vitro = in_vitro_target()
    pipeline_3 = ExternalPipeline(
        index=3,
        name="Cell-line dose response",
        dataset="GDSC2 (Sanger) fitted dose response, cached locally",
        source="https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/",
        proposed_metric="R^2 >= 0.938 of replicator fit against viability curves",
        status=in_vitro.status,
        detail=(
            f"live verdict: {in_vitro.status.value}; " + " ".join(in_vitro.blockers)
        ),
        recorded_instead=(
            "the raw multi-dose viability points are 2.11 GB (GDSC2) and are not "
            "cached; the cached fitted release is audited for what it can bear"
        ),
    )

    # --- 4. inverse tensor recovery ------------------------------------------
    pipeline_4 = ExternalPipeline(
        index=4,
        name="3D spatial stencil / inverse tensor recovery",
        dataset="BrainWeb simulated brain database (McGill)",
        source="https://brainweb.bic.mni.mcgill.ca/",
        proposed_metric="inverse tensor recovery R^2 >= 0.999",
        status=TargetStatus.REFUSED_NO_DATA,
        detail=(
            "two independent reasons. (a) The repository has no spatial "
            "inverse-tensor estimator: div(D grad(.)) is implemented forward "
            "and verified against analytic solutions, but nothing estimates "
            "D(x) back from an observed field. (b) Even implemented, injecting "
            "a known tensor and recovering it tests the solver against itself: "
            "a self-consistency check, not a validation, because a phantom "
            "carries no measured tissue anisotropy. BrainWeb is also not cached "
            "here (its downloads are form-driven)"
        ),
    )

    # --- 5. multi-region growth ----------------------------------------------
    from danon.eval_targets import spatial_expansion_target

    spatial = spatial_expansion_target()
    pipeline_5 = ExternalPipeline(
        index=5,
        name="Multi-region MRI progression",
        dataset="MICCAI BraTS challenge series (BraTS-2021 cases cached locally)",
        source="https://www.med.upenn.edu/cbica/brats2021/",
        proposed_metric="3D sub-region Dice >= 0.88",
        status=spatial.status,
        detail=(
            f"live verdict: {spatial.status.value}. {spatial.detail}"
        ),
        recorded_instead=(
            "BraTS-2021 segmentations ARE the geometry this engine runs on "
            "(six cases, committed); what does not exist is a second timepoint "
            "with an expert mask, so a growth forecast has nothing to be scored "
            "against"
        ),
    )

    return [pipeline_1, pipeline_2, pipeline_3, pipeline_4, pipeline_5]


# ---------------------------------------------------------------------------
# Provenance and the pitch corrections
# ---------------------------------------------------------------------------


def external_cohorts_provenance() -> Dict[str, EvidenceStatus]:
    """What the external-cohort work establishes, and what stays false here."""
    return {
        "cgga_clinical_release_cached": EvidenceStatus.ESTABLISHED,
        "ood_transport_of_covariate_model": EvidenceStatus.ESTABLISHED,
        "ood_transport_is_engine_validation": EvidenceStatus.PROVEN_FAILED,
        "single_cell_state_shift_prediction": EvidenceStatus.PROVEN_FAILED,
        "in_vitro_ic50_replicator_fit": EvidenceStatus.PROVEN_FAILED,
        "brainweb_inverse_tensor_recovery": EvidenceStatus.PROVEN_FAILED,
        "growth_dice_forecast": EvidenceStatus.PROVEN_FAILED,
        "cgga_pitch_figures_are_results": EvidenceStatus.PROVEN_FAILED,
    }


def pitch_corrections() -> List[Dict[str, Any]]:
    """Each proposed figure, why it is refused, and what replaces it.

    Built from the cached files where possible, so the corrections move with
    the data instead of being remembered.
    """
    corrections: List[Dict[str, Any]] = []

    # n=325 GBM -- counted from the files, not asserted.
    counts: Dict[str, Any] = {}
    for path in (CGGA_325_ZIP, CGGA_693_ZIP):
        if not os.path.exists(path):
            continue
        cohort = read_cgga_clinical(path)
        scorable_gbm = sum(1 for r in cohort.rows if r.is_gbm)
        counts[os.path.basename(path)] = {
            "table_rows": cohort.n_table_rows,
            "usable": cohort.n,
            "histology_GBM_scorable": scorable_gbm,
            "histology_GBM_in_table": scorable_gbm
            + cohort.dropped_histology.get("GBM", 0),
            "primary": len(cohort.primary()),
            "primary_GBM": len(cohort.primary_gbm()),
        }
    corrections.append(
        {
            "shown_in_the_pitch": PITCH_FIGURES["cgga_gbm_n"],
            "why_refused": (
                "the release named 325 holds 325 clinical rows; the number of "
                "them recorded as GBM is a different, smaller count, read from "
                "the file"
            ),
            "what_is_recorded_instead": counts,
        }
    )
    corrections.append(
        {
            "shown_in_the_pitch": PITCH_FIGURES["cgga_ood_c_index"],
            "why_refused": (
                "a C-index is a measurement, not a target: the recorded value "
                "comes from applying frozen TCGA coefficients to the CGGA rows"
            ),
            "what_is_recorded_instead": (
                f"see {CGGA_OOD_RECORD}: the C-index per subset and file, "
                f"recomputable from the cached zips; the pitch's 0.6412 is not "
                f"carried as a result"
            ),
        }
    )
    corrections.append(
        {
            "shown_in_the_pitch": PITCH_FIGURES["scrna_jsd"],
            "why_refused": (
                "no four-state composition is predicted anywhere in this "
                "repository, and the GEO workbook maps cells to tumours, not "
                "to states"
            ),
            "what_is_recorded_instead": (
                "single_cell_clone_fractions_from_bulk: PROVEN_FAILED in "
                "danon/genomics_intake.py; the dataset's reachability is "
                "recorded so the refusal is not a claim of unavailability"
            ),
        }
    )
    corrections.append(
        {
            "shown_in_the_pitch": PITCH_FIGURES["in_vitro_r2"],
            "why_refused": (
                "the engine's EC50 is dimensionless, so an R^2 against a molar "
                "IC50 is a type error; and 34 of 34 temozolomide IC50s in GDSC2 "
                "are extrapolations beyond the highest concentration tested"
            ),
            "what_is_recorded_instead": (
                "danon/eval_targets.py::in_vitro_target(), called live by "
                "external_pipeline_matrix()"
            ),
        }
    )
    corrections.append(
        {
            "shown_in_the_pitch": PITCH_FIGURES["inverse_recovery_r2"],
            "why_refused": (
                "there is no inverse-tensor estimator to score, and recovering "
                "an injected tensor is a self-consistency check rather than a "
                "measurement against tissue"
            ),
            "what_is_recorded_instead": (
                "the forward operator is already verified against analytic "
                "solutions (check_solver_verified_against_analytic_solutions); "
                "no recovery figure is claimed"
            ),
        }
    )
    corrections.append(
        {
            "shown_in_the_pitch": PITCH_FIGURES["brats_dice"],
            "why_refused": (
                "a Dice forecast needs an expert mask and a second timepoint "
                "for the same patient; the live spatial_dice_verdict says no "
                "probed collection carries both"
            ),
            "what_is_recorded_instead": (
                "BraTS-2021 segmentations are the committed geometry the engine "
                "runs on, not a forecast target"
            ),
        }
    )
    corrections.append(
        {
            "shown_in_the_pitch": (
                "sample code: `risk_scores = ages * 0.035` described as a "
                "TCGA-calibrated coefficient, `assert c_index >= 0.620`, and "
                "`from lifelines.utils import concordance_index`"
            ),
            "why_refused": (
                "0.035 is an invented coefficient (the repository fits age, sex "
                "and diagnosis on the real cohort and applies those betas); "
                "0.620 is a threshold chosen before the measurement, which "
                "turns a result into a pass/fail rubber stamp; lifelines is not "
                "a dependency here, and the repository already carries its own "
                "Harrell C (danon/validation.py::concordance_index); and a test "
                "that downloads inside pytest makes the suite depend on the "
                "network"
            ),
            "what_is_recorded_instead": (
                "the OOD test is a cohort script + cached file + claim check; "
                "whatever the transport C-index is, it is recorded as measured"
            ),
        }
    )
    return corrections


def what_external_data_cannot_show() -> List[str]:
    return [
        "an external C-index over age, sex and diagnosis is a statement about "
        "those covariates, not about the spatial engine, which has no "
        "patient-level C-index at all",
        "one cohort at one metric cannot demonstrate 'zero overfitting': it "
        "demonstrates transport, and a weak model transports too",
        "CGGA clinical rows carry no imaging link here, so no growth or "
        "geometry validation rides on them",
        "none of the five pipelines, even if all five ran, would make the "
        "engine's dose schedule validated: that needs outcome data tied to a "
        "treatment the engine chose",
    ]
