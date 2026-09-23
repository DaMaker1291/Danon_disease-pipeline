"""
Evaluation targets: three proposed ways to reach 0.90+, and what each can bear.

WHY THIS MODULE EXISTS
----------------------
A published argument for raising predictive performance is to *change the
estimand*: stop predicting survival, and predict 3D spatial expansion, molecular
responder status, or in-vitro dose response instead, each of which is claimed to
have a ceiling above 0.90. The argument is sound in general -- different
estimands do have different ceilings -- but a ceiling is not a result, and each
target has to be checked against data that exists. Doing that here gives three
different answers, and only one of them is a number.

TARGET 1 -- 3D spatial expansion (claimed Dice 0.91-0.94): NO DATA.
The Dice test needs paired imaging: a baseline scan, a follow-up scan, and an
expert mask on both. None is retrievable -- `danon.multimodal.spatial_dice_verdict`
derives that from live collection counts (TCGA-GBM, TCGA-LGG and BraTS are
retired from the NBIA API; UPENN-GBM is single-timepoint with zero segmentation
series). This module does not restate that; it calls the verdict and reports
whatever it currently says, so if a segmented collection ever appears the target
starts reporting numbers instead of staying refused.

TARGET 2 -- molecular responder classification (claimed AUC 0.91-0.94): NO LABEL.
`best_overall_response` is present as a field on the diagnosis node and is
**empty for all 617 TCGA-GBM cases**. The nearest available label is
`disease_response`, whose vocabulary is `WT-With Tumor` / `TF-Tumor Free` /
`Unknown` -- a disease-status flag recorded at follow-up, not a response to
therapy. Used as a binary label it yields 23 positives in 529 cases (4.3%), it
is *anti-correlated* with survival (median OS 229 days for TF against 374 for
WT), and 11 cases carry both values at different timepoints so "clean" labels
exist for only part of the cohort. A baseline covariate model reaches out-of-fold
AUC 0.6155. The reported ceiling is not the issue; the label is.

TARGET 3 -- in-vitro dose response (claimed R^2 > 0.93): INCOMMENSURABLE, AND MOST
OF THE TARGET VALUES ARE EXTRAPOLATED. GDSC is reachable (this module reads the
40 MB GDSC2 fitted dose-response release), so the data gap is not the blocker.
Two other things are:

1. The engine's EC50 has **no declared unit** -- `DrugParams` documents
   "conc/day at full-dose vessels" -- so it cannot be compared to a molar IC50.
   `danon.pharmacology.compare_engine_ec50` already refuses for this reason. An
   R^2 between a dimensionless number and a micromolar IC50 is not a fit, it is a
   type error.
2. Independently, **76.0% of GDSC2 GBM IC50s lie above the highest concentration
   tested**, so they are extrapolations rather than measurements. For
   temozolomide it is **34 of 34**. The same defect shows up in the ChEMBL
   records already in this repository, where 19 of 50 glioma temozolomide
   records are right-censored (`>`). Two independent sources agreeing that this
   particular quantity is mostly a lower bound is the reason
   `gdsc_extrapolation` is a refusal and not a caveat.

NOT CLAIMED
-----------
Nothing here raises any number anywhere else in the repository. Refusing three
targets does not improve the fourth.
"""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.predictive import GDC_CLINICAL_PATH, load_cohort_rows

#: GDSC2 fitted dose response from the Sanger current release. The raw
#: dose-response files are 0.78 GB (GDSC1) and 2.11 GB (GDSC2), which is why the
#: fitted summary is what is cached; the raw files would be needed to test the
#: engine's Hill *slope*, and that is recorded as a gap rather than skipped.
GDSC2_FITTED_PATH = "data/gdsc/GDSC2_fitted_dose_response.csv"
GDSC2_FITTED_SOURCE = (
    "https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/"
    "GDSC2_fitted_dose_response_24Jul22.csv"
)
GDSC2_RAW_BYTES = 2_112_924_313
GDSC1_RAW_BYTES = 778_875_447

#: GDSC tissue code for the glioma/CNS lines.
GDSC_GBM_TISSUE = "GBM"

#: The response vocabulary actually present in the GDC export.
RESPONSE_WITH_TUMOR = "WT"
RESPONSE_TUMOR_FREE = "TF"

#: EGFRvIII is an in-frame deletion of exons 2-7 -- a STRUCTURAL variant, not a
#: point mutation -- so it cannot appear in a Masked Somatic Mutation (SNV/indel)
#: file. It would need Structural Variation calls or RNA-seq junction reads.
EGFRVIII_IS_STRUCTURAL = True
EGFRVIII_REACHABILITY_PATH = "data/gdc_clinical/egfrviii_reachability.json"

#: The dosing policies that exist. All closed-form, none learned.
RULE_BASED_POLICIES: Dict[str, str] = {
    "MTDPolicy": "returns the constant (1.0, 0.0)",
    "AdaptiveDosePolicy": "one comparison against the initial burden",
    "CollateralSwitchPolicy": "a threshold switch on the resistant fraction",
}


class TargetStatus(str, Enum):
    """Why a target does or does not have a number."""

    MEASURED = "measured"
    REFUSED_NO_DATA = "refused_no_data"
    REFUSED_NO_LABEL = "refused_no_label"
    REFUSED_INCOMMENSURABLE = "refused_incommensurable"
    REFUSED_ESTIMAND = "refused_estimand"


@dataclass(frozen=True)
class TargetVerdict:
    """One proposed target, its metric, and what it can actually bear."""

    name: str
    claimed: str
    metric: str
    status: TargetStatus
    measured: Optional[float]
    n: int
    detail: str
    blockers: List[str] = field(default_factory=list)

    @property
    def has_number(self) -> bool:
        return self.status is TargetStatus.MEASURED and self.measured is not None

    @property
    def display(self) -> str:
        """The measured value, at enough precision not to read as perfect.

        An R^2 of 0.999951 formatted to four decimals prints "1.0000", which
        reads as a flawless fit while a measurable residual remains. Anything
        above 0.999 is shown at six decimals so the residual survives the
        format; `measured` still returns the full float.
        """
        if self.measured is None:
            return "no number"
        if abs(self.measured) > 0.999:
            return f"{self.measured:.6f}"
        return f"{self.measured:.4f}"

    def describe(self) -> str:
        value = self.display
        return (
            f"{self.name}: claimed {self.claimed} on {self.metric} -> "
            f"{self.status.value.upper()} ({value}, n={self.n}). {self.detail}"
        )


# ---------------------------------------------------------------------------
# Target 1 -- spatial expansion
# ---------------------------------------------------------------------------


def spatial_expansion_target() -> TargetVerdict:
    """The Dice target, coupled to the live data verdict rather than restated."""
    from danon.multimodal import load_evidence, spatial_dice_verdict

    verdict = spatial_dice_verdict()
    imaging = [
        m for m in load_evidence()
        if str(getattr(m, "key", "")).startswith("tcia_")
        and getattr(m, "key", "") != "tcia_nbia_api"
    ]
    masks = sum(int(getattr(m, "n_seg_series", 0) or 0) for m in imaging)
    checked = sum(
        int(getattr(m, "patients_checked_for_timepoints", 0) or 0) for m in imaging
    )
    if verdict.supported:
        return TargetVerdict(
            name="3D spatial tumour expansion",
            claimed="Dice 0.91-0.94",
            metric="Dice similarity coefficient",
            status=TargetStatus.MEASURED,
            measured=None,
            n=checked,
            detail=(
                "a segmented collection with a second timepoint is now retrievable, "
                "so this target should be measured rather than refused; measure it "
                "and report the number"
            ),
        )
    return TargetVerdict(
        name="3D spatial tumour expansion",
        claimed="Dice 0.91-0.94",
        metric="Dice similarity coefficient",
        status=TargetStatus.REFUSED_NO_DATA,
        measured=None,
        n=checked,
        detail=(
            f"{len(imaging)} imaging collection(s) probed, {masks} segmentation "
            f"series, {checked} patient(s) checked for a second timepoint. A Dice "
            f"score needs a baseline scan, a follow-up scan and an expert mask on "
            f"both; none of the three is retrievable"
        ),
        blockers=[
            "TCGA-GBM, TCGA-LGG and BraTS are retired from the NBIA API",
            "UPENN-GBM is single-timepoint and carries no segmentation series",
            "no local case has a second dated timepoint",
        ],
    )


# ---------------------------------------------------------------------------
# Target 2 -- responder classification
# ---------------------------------------------------------------------------


def _response_kind(value: object) -> Optional[str]:
    text = str(value or "")
    if text.startswith(RESPONSE_TUMOR_FREE):
        return RESPONSE_TUMOR_FREE
    if text.startswith(RESPONSE_WITH_TUMOR):
        return RESPONSE_WITH_TUMOR
    return None


@dataclass(frozen=True)
class ResponderLabelAudit:
    """Does a responder label exist, and what is it if so?"""

    n_cases: int
    n_best_overall_response: int
    n_with_any_disease_response: int
    n_clean_label: int
    n_positive: int
    n_ambiguous: int
    n_neither: int

    @property
    def positive_fraction(self) -> float:
        return self.n_positive / self.n_clean_label if self.n_clean_label else float("nan")

    @property
    def majority_accuracy(self) -> float:
        if not self.n_clean_label:
            return float("nan")
        return max(self.positive_fraction, 1.0 - self.positive_fraction)

    @property
    def verdict(self) -> str:
        return (
            f"`best_overall_response` records: {self.n_best_overall_response} of "
            f"{self.n_cases}. The nearest label is a disease-status flag with "
            f"{self.n_positive} positives in {self.n_clean_label} clean cases "
            f"({100 * self.positive_fraction:.1f}%), {self.n_ambiguous} case(s) "
            f"carrying both values at different timepoints, {self.n_neither} with "
            f"neither. Majority-class accuracy is "
            f"{100 * self.majority_accuracy:.2f}%, which is where a '94% accuracy' "
            f"claim would come from without any model at all"
        )


def audit_responder_label(path: str = GDC_CLINICAL_PATH) -> ResponderLabelAudit:
    """Measure the label that exists, and the one that was proposed."""
    with open(path) as fh:
        hits = json.load(fh).get("hits", [])

    n_bor = 0
    n_any = 0
    clean = 0
    positive = 0
    ambiguous = 0
    neither = 0
    for hit in hits:
        for dx in hit.get("diagnoses") or []:
            if dx.get("best_overall_response") is not None:
                n_bor += 1
        kinds = {
            k for k in (
                _response_kind(fu.get("disease_response"))
                for fu in (hit.get("follow_ups") or [])
            ) if k is not None
        }
        if kinds:
            n_any += 1
        if len(kinds) > 1:
            ambiguous += 1
        elif kinds == {RESPONSE_TUMOR_FREE}:
            clean += 1
            positive += 1
        elif kinds == {RESPONSE_WITH_TUMOR}:
            clean += 1
        else:
            neither += 1

    return ResponderLabelAudit(
        n_cases=len(hits),
        n_best_overall_response=n_bor,
        n_with_any_disease_response=n_any,
        n_clean_label=clean,
        n_positive=positive,
        n_ambiguous=ambiguous,
        n_neither=neither,
    )


def _logistic_fit(X: np.ndarray, y: np.ndarray, iters: int = 60, ridge: float = 1e-6) -> np.ndarray:
    """Newton-Raphson logistic regression with a small ridge."""
    X = np.column_stack([np.ones(len(X)), X])
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ beta)))
        w = p * (1.0 - p) + 1e-9
        grad = X.T @ (y - p) - ridge * beta
        hess = (X * w[:, None]).T @ X + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.abs(step).max() < 1e-9:
            break
    return beta


def _auc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties counted as half."""
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).sum() + 0.5 * (diff == 0).sum()) / (len(pos) * len(neg)))


@dataclass(frozen=True)
class ResponderTarget:
    """The honest version of the responder target on the label that exists."""

    label: ResponderLabelAudit
    n_patients: int
    n_positive: int
    auc: float
    auc_ci: Tuple[float, float]
    median_os_positive_days: float
    median_os_negative_days: float

    @property
    def label_tracks_survival_backwards(self) -> bool:
        """True when 'tumour free' patients died sooner than the others.

        That inversion means the flag is confounded by follow-up timing -- it is
        recorded at follow-up, so a patient who died early may carry a
        tumour-free reading while long survivors are dominated by later
        with-tumour readings -- and not a treatment-response label.
        """
        return self.median_os_positive_days < self.median_os_negative_days

    def describe(self) -> str:
        return (
            f"responder target on the reachable label: out-of-fold AUC = "
            f"{self.auc:.4f} (95% CI {self.auc_ci[0]:.4f}-{self.auc_ci[1]:.4f}, "
            f"n={self.n_patients}, {self.n_positive} positive). Median OS: "
            f"{self.median_os_positive_days:.0f} d for tumour-free against "
            f"{self.median_os_negative_days:.0f} d for with-tumour"
            + (
                " -- INVERTED, so the flag is confounded by when it was recorded"
                if self.label_tracks_survival_backwards
                else ""
            )
        )


def responder_target(
    *,
    clinical_path: str = GDC_CLINICAL_PATH,
    n_folds: int = 5,
    seed: int = 0,
    n_bootstrap: int = 200,
) -> ResponderTarget:
    """Fit the baseline covariates to the reachable label and score it out of fold."""
    label_audit = audit_responder_label(clinical_path)
    with open(clinical_path) as fh:
        hits = json.load(fh).get("hits", [])
    label: Dict[str, int] = {}
    for hit in hits:
        kinds = {
            k for k in (
                _response_kind(fu.get("disease_response"))
                for fu in (hit.get("follow_ups") or [])
            ) if k is not None
        }
        if kinds == {RESPONSE_TUMOR_FREE}:
            label[str(hit.get("submitter_id"))] = 1
        elif kinds == {RESPONSE_WITH_TUMOR}:
            label[str(hit.get("submitter_id"))] = 0

    rows, _ = load_cohort_rows(clinical_path, endpoint="os")
    index = {r.submitter_id: r for r in rows}
    keep = [sid for sid in label if sid in index]
    X = np.array(
        [[index[s].age_years, index[s].male, index[s].is_glioblastoma] for s in keep],
        dtype=float,
    )
    y = np.array([label[s] for s in keep], dtype=int)

    rng = np.random.RandomState(seed)
    oof = np.full(len(y), np.nan)
    for fold in np.array_split(rng.permutation(len(y)), n_folds):
        test = np.zeros(len(y), dtype=bool)
        test[fold] = True
        train = ~test
        if y[train].sum() < 3 or y[test].sum() < 1:
            continue
        beta = _logistic_fit(X[train], y[train])
        oof[test] = np.column_stack([np.ones(test.sum()), X[test]]) @ beta

    scored = np.isfinite(oof)
    auc = _auc(oof[scored], y[scored])

    boot: List[float] = []
    idx = np.where(scored)[0]
    for _ in range(n_bootstrap):
        take = rng.choice(idx, size=len(idx), replace=True)
        if y[take].sum() in (0, len(take)):
            continue
        boot.append(_auc(oof[take], y[take]))
    ci = (
        (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        if boot
        else (float("nan"), float("nan"))
    )

    times = {s: index[s].time_days for s in keep}
    pos_t = [times[s] for s, yy in zip(keep, y) if yy == 1]
    neg_t = [times[s] for s, yy in zip(keep, y) if yy == 0]
    return ResponderTarget(
        label=label_audit,
        n_patients=int(scored.sum()),
        n_positive=int(y[scored].sum()),
        auc=auc,
        auc_ci=ci,
        median_os_positive_days=float(statistics.median(pos_t)) if pos_t else float("nan"),
        median_os_negative_days=float(statistics.median(neg_t)) if neg_t else float("nan"),
    )


# ---------------------------------------------------------------------------
# Target 3 -- in-vitro dose response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GdscAudit:
    """What the GDSC2 fitted release contains for glioma lines."""

    path: str
    n_rows: int
    n_drugs: int
    n_cell_lines: int
    n_gbm_rows: int
    n_gbm_drugs: int
    n_gbm_lines: int
    n_inside_range: int
    n_above_max_tested: int
    n_below_min_tested: int
    temozolomide_rows: int
    temozolomide_above_max_tested: int

    @property
    def extrapolated_fraction(self) -> float:
        total = self.n_inside_range + self.n_above_max_tested + self.n_below_min_tested
        return self.n_above_max_tested / total if total else float("nan")

    @property
    def verdict(self) -> str:
        return (
            f"{self.n_gbm_rows} glioma-row(s) over {self.n_gbm_drugs} drug(s) and "
            f"{self.n_gbm_lines} line(s); "
            f"{100 * self.extrapolated_fraction:.1f}% of glioma IC50s lie above the "
            f"highest concentration tested, and for temozolomide it is "
            f"{self.temozolomide_above_max_tested} of {self.temozolomide_rows}"
        )


def load_gdsc_glioma(path: str = GDSC2_FITTED_PATH) -> GdscAudit:
    """Read the cached GDSC2 fitted release and audit the glioma subset.

    The extrapolation count is the point. An `LN_IC50` whose exponential exceeds
    the plate's maximum tested concentration is a model extrapolation, not a
    measurement, and those values are what an in-vitro R^2 would be fitted to.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GDSC fitted release not cached at {path}; source {GDSC2_FITTED_SOURCE}"
        )

    def _f(value: object) -> Optional[float]:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None

    n_rows = 0
    drugs = set()
    lines = set()
    gbm_rows = 0
    gbm_drugs = set()
    gbm_lines = set()
    inside = above = below = 0
    tmz_rows = 0
    tmz_above = 0

    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            drugs.add(row.get("DRUG_NAME"))
            lines.add(row.get("CELL_LINE_NAME"))
            if (row.get("TCGA_DESC") or "") != GDSC_GBM_TISSUE:
                continue
            gbm_rows += 1
            gbm_drugs.add(row.get("DRUG_NAME"))
            gbm_lines.add(row.get("CELL_LINE_NAME"))
            ln_ic50 = _f(row.get("LN_IC50"))
            max_conc = _f(row.get("MAX_CONC"))
            min_conc = _f(row.get("MIN_CONC"))
            is_tmz = (row.get("DRUG_NAME") or "").strip().lower() == "temozolomide"
            if is_tmz:
                tmz_rows += 1
            if ln_ic50 is None or max_conc is None:
                continue
            ic50 = math.exp(ln_ic50)
            if ic50 > max_conc:
                above += 1
                if is_tmz:
                    tmz_above += 1
            elif min_conc is not None and ic50 < min_conc:
                below += 1
            else:
                inside += 1

    return GdscAudit(
        path=path,
        n_rows=n_rows,
        n_drugs=len(drugs),
        n_cell_lines=len(lines),
        n_gbm_rows=gbm_rows,
        n_gbm_drugs=len(gbm_drugs),
        n_gbm_lines=len(gbm_lines),
        n_inside_range=inside,
        n_above_max_tested=above,
        n_below_min_tested=below,
        temozolomide_rows=tmz_rows,
        temozolomide_above_max_tested=tmz_above,
    )


def in_vitro_target(audit: Optional[GdscAudit] = None, *, path: str = GDSC2_FITTED_PATH) -> TargetVerdict:
    """The GDSC R^2 target: reachable data, two independent blockers."""
    from danon.pharmacology import engine_ec50_unit

    unit = engine_ec50_unit()
    try:
        gdsc = audit if audit is not None else load_gdsc_glioma(path)
        n = gdsc.n_gbm_rows
        reachability = gdsc.verdict
        extr = gdsc.extrapolated_fraction
        tmz_above, tmz_rows = gdsc.temozolomide_above_max_tested, gdsc.temozolomide_rows
    except FileNotFoundError as exc:
        n = 0
        reachability = str(exc)
        extr = float("nan")
        tmz_above = tmz_rows = 0

    blockers = []
    if unit is None:
        blockers.append(
            "the engine's EC50 has no declared concentration unit, so an R^2 "
            "against a molar IC50 is incommensurable rather than inaccurate"
        )
    if np.isfinite(extr) and extr > 0.5:
        blockers.append(
            f"{100 * extr:.1f}% of glioma IC50s are extrapolated beyond the highest "
            f"concentration tested ({tmz_above} of {tmz_rows} for temozolomide), so "
            f"the target values are largely lower bounds"
        )
    blockers.append(
        f"testing the engine's Hill slope (it fixes n=1) needs the raw "
        f"dose-response points: {GDSC2_RAW_BYTES / 1e9:.2f} GB for GDSC2 and "
        f"{GDSC1_RAW_BYTES / 1e9:.2f} GB for GDSC1, not cached here"
    )

    status = (
        TargetStatus.REFUSED_INCOMMENSURABLE
        if unit is None
        else TargetStatus.REFUSED_ESTIMAND
    )
    return TargetVerdict(
        name="in-vitro cell-line dose response",
        claimed="R^2 > 0.93 vs GDSC IC50",
        metric="R^2 against measured IC50",
        status=status,
        measured=None,
        n=n,
        detail=reachability,
        blockers=blockers,
    )


# ---------------------------------------------------------------------------
# Spatial metric: correct degenerate-case semantics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiceResult:
    """Dice with an explicit estimability flag, mirroring `Concordance`.

    The conventional shortcut `if total_volume == 0: return 1.0` scores a
    prediction of *nothing* at 1.0 whenever the truth is also empty. That inverts
    the meaning of the metric in exactly the case where the model has made no
    prediction at all -- the same failure mode as a verdict that renders
    "could not tell" as "passed". An empty comparison is not perfect; it is
    undefined, and it is reported as such here.
    """

    coefficient: Optional[float]
    estimable: bool
    n_predicted: int
    n_true: int
    n_intersection: int
    reason: str = ""

    def __str__(self) -> str:
        if not self.estimable:
            return (
                f"Dice not estimable ({self.reason}); predicted {self.n_predicted} "
                f"voxel(s), truth {self.n_true}"
            )
        return (
            f"Dice {self.coefficient:.4f} over predicted {self.n_predicted} / "
            f"truth {self.n_true} / intersection {self.n_intersection} voxels"
        )


def dice_similarity(predicted: np.ndarray, truth: np.ndarray) -> DiceResult:
    """Dice coefficient with an explicit undefined case.

    Undefined when both masks are empty (no prediction was made), and 0.0 when
    the prediction is empty but the truth is not (that is a real miss).
    """
    pred = np.asarray(predicted) > 0
    tru = np.asarray(truth) > 0
    if pred.shape != tru.shape:
        raise ValueError("predicted and truth must have the same shape")
    n_pred = int(pred.sum())
    n_true = int(tru.sum())
    n_int = int((pred & tru).sum())
    if n_pred + n_true == 0:
        return DiceResult(
            coefficient=None,
            estimable=False,
            n_predicted=0,
            n_true=0,
            n_intersection=0,
            reason="both masks empty: no prediction was made, so overlap is undefined",
        )
    return DiceResult(
        coefficient=float(2.0 * n_int / (n_pred + n_true)),
        estimable=True,
        n_predicted=n_pred,
        n_true=n_true,
        n_intersection=n_int,
    )


def dice_degenerate_case_report() -> str:
    """What the shortcut convention scores, against what it should score."""
    empty = np.zeros((4, 4, 4))
    truth = np.zeros((4, 4, 4))
    truth[1, 1, 1] = 1
    both_empty_shortcut = 1.0
    both_empty_actual = dice_similarity(empty, empty)
    miss = dice_similarity(empty, truth)
    return (
        f"predicting nothing against an empty truth: the `total_volume == 0 -> 1.0` "
        f"shortcut returns {both_empty_shortcut:.1f} (a PERFECT score for no "
        f"prediction), while the flagged version returns "
        f"{'not estimable' if not both_empty_actual.estimable else 'a number'}. "
        f"Predicting nothing against a non-empty truth is a real miss and scores "
        f"{miss.coefficient:.1f}"
    )


# ---------------------------------------------------------------------------
# EGFRvIII: a structural variant is not in the open mutation data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EgfrviiiReachability:
    """Whether EGFRvIII status is obtainable from open GDC data."""

    n_structural_variation_files: int
    n_structural_variation_controlled: int
    n_open_maf_files: int
    n_maf_controlled: int
    source: str

    @property
    def reachable_open(self) -> bool:
        return self.n_structural_variation_controlled < self.n_structural_variation_files

    @property
    def verdict(self) -> str:
        if self.reachable_open:
            return (
                "some structural-variation files are open, so EGFRvIII may be "
                "callable; verify and record the call method"
            )
        return (
            f"EGFRvIII is an in-frame deletion of exons 2-7, a STRUCTURAL variant, "
            f"so it cannot appear in the {self.n_open_maf_files} open Masked Somatic "
            f"Mutation files (SNV/indel). All {self.n_structural_variation_files} "
            f"structural-variation files for TCGA-GBM are controlled access, so "
            f"EGFRvIII status is not obtainable from open data"
        )

    def __str__(self) -> str:
        return (
            f"EGFRvIII: {self.n_structural_variation_files} SV files "
            f"({self.n_structural_variation_controlled} controlled), "
            f"{self.n_open_maf_files} open MAF -- {self.verdict}"
        )


def probe_egfrviii_reachability(
    *,
    project: str = "TCGA-GBM",
    fetcher: Optional[object] = None,
    cache_path: str = EGFRVIII_REACHABILITY_PATH,
    use_cache: bool = True,
) -> EgfrviiiReachability:
    """Count SV and MAF files by access level for the project."""
    if use_cache:
        try:
            with open(cache_path) as fh:
                return EgfrviiiReachability(**json.load(fh))
        except (OSError, ValueError, TypeError):
            pass

    def count(data_category: str) -> Tuple[int, int]:
        query = urllib.parse.urlencode(
            {
                "filters": json.dumps(
                    {
                        "op": "and",
                        "content": [
                            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
                            {"op": "in", "content": {"field": "data_category", "value": [data_category]}},
                        ],
                    }
                ),
                "size": "0",
                "facets": "access,data_type",
            }
        )
        url = f"https://api.gdc.cancer.gov/files?{query}"
        payload = json.load(urllib.request.urlopen(url, timeout=60))
        total = int(payload["data"]["pagination"]["total"])
        buckets = payload["data"]["aggregations"]["access"]["buckets"]
        controlled = sum(int(b["doc_count"]) for b in buckets if b["key"] != "open")
        return total, controlled

    sv_total, sv_controlled = count("Structural Variation")
    maf_total, maf_controlled = count("Simple Nucleotide Variation")
    result = EgfrviiiReachability(
        n_structural_variation_files=sv_total,
        n_structural_variation_controlled=sv_controlled,
        n_open_maf_files=maf_total - maf_controlled,
        n_maf_controlled=maf_controlled,
        source="https://api.gdc.cancer.gov/files",
    )
    try:
        with open(cache_path, "w") as fh:
            json.dump(result.__dict__, fh, indent=2, sort_keys=True)
    except OSError:
        pass
    return result


# ---------------------------------------------------------------------------
# The RL tier
# ---------------------------------------------------------------------------


def rl_policy_target() -> TargetVerdict:
    """The claimed RL convergence rate, against the policies that exist.

    No reinforcement-learning agent is present: every dosing policy is a
    closed-form rule, and `requirements.txt` carries `ray[default]` for the
    design-arm bandit (`danon.active_learner`), which selects protein candidates
    rather than dosing a simulated patient. The metric itself is the deeper
    problem -- `normalized reward retention` is defined by whoever writes the
    reward and the normalization, so it cannot be falsified by any data. This
    repository's comparable quantity is anchored externally instead: a paired
    MTD control on the same geometry, a RANO progression threshold, and a
    censored lower bound.
    """
    from danon.evosteer import AdaptiveDosePolicy, CollateralSwitchPolicy, MTDPolicy

    present = {
        cls.__name__: RULE_BASED_POLICIES[name]
        for name, cls in (
            ("MTDPolicy", MTDPolicy),
            ("AdaptiveDosePolicy", AdaptiveDosePolicy),
            ("CollateralSwitchPolicy", CollateralSwitchPolicy),
        )
    }
    # A literal constant dose is the clearest evidence that nothing is learned.
    from danon.evosteer import PolicyState

    constant = MTDPolicy().dose(PolicyState(day=0, burden=1.0, burden_initial=1.0, resistant_fraction=0.0))
    return TargetVerdict(
        name="RL dosing-policy convergence",
        claimed="95.4% normalised reward retention",
        metric="reward retention over Monte Carlo training episodes",
        status=TargetStatus.REFUSED_NO_DATA,
        measured=None,
        n=0,
        detail=(
            f"no reinforcement-learning agent exists in the repository. The three "
            f"dosing policies are closed-form: "
            f"{'; '.join(f'{k} {v}' for k, v in present.items())}. "
            f"MTDPolicy.dose() returns {constant} for every state, so nothing is "
            f"learned, trained, or converged"
        ),
        blockers=[
            "no RL agent, environment or training loop is present",
            "the three dosing policies are closed-form rules, one of which is a constant",
            "`normalized reward retention` is self-defined: the reward and the "
            "normalization are both chosen by the author, so the number cannot be "
            "falsified by any measurement",
            "the repo's comparable quantity is externally anchored instead: paired "
            "MTD control on identical geometry against a RANO threshold, reported "
            "as a censored lower bound",
        ],
    )


# ---------------------------------------------------------------------------
# The one tier that does have a number
# ---------------------------------------------------------------------------


def solver_convergence_target() -> TargetVerdict:
    """The PDE stencil tier -- the only proposed metric that is measurable here.

    Reported as MEASURED because it genuinely is one, and labelled at every
    surface as **synthetic planted ground truth**: the field is generated from
    the analytic point-source solution, which knows nothing about this
    codebase's grid or stencil, so the recovery error is the engine's
    discretisation error rather than a linear solve inverting an operator it was
    handed. That makes it a real solver check. It is not a patient result, and
    it is not the same kind of quantity as the C-index.
    """
    from danon.recovery import transport_recovery

    tr = transport_recovery()
    return TargetVerdict(
        name="PDE stencil convergence (SYNTHETIC, no patients)",
        claimed="R^2 >= 0.9999",
        metric="R^2 against analytic point-source ground truth",
        status=TargetStatus.MEASURED,
        measured=float(tr.r_squared),
        n=int(tr.n_equations),
        detail=(
            f"D planted {tr.planted_d:.4f} -> recovered {tr.recovered_d:.4f} "
            f"({100 * tr.d_relative_error:.2f}% error); lambda planted "
            f"{tr.planted_lambda:.4f} -> recovered {tr.recovered_lambda:.4f} "
            f"({100 * tr.lambda_relative_error:.2f}% error) over "
            f"{tr.n_equations:,} equations. This validates the solver; it says "
            f"nothing about patients, and a high value here raises no number in "
            f"any other tier"
        ),
        blockers=[
            "the ground truth is synthetic: no patient appears in this tier",
            "the reported R^2 is a fit residual on an analytic field, not a "
            "predictive accuracy on any clinical outcome",
            "it may not be averaged with, added to, or used to raise the "
            "overall-survival C-index",
        ],
    )


# ---------------------------------------------------------------------------
# The honest ceiling table
# ---------------------------------------------------------------------------


def target_verdicts(
    *,
    with_gdsc: bool = True,
    with_responder: bool = True,
    with_solver: bool = False,
) -> List[TargetVerdict]:
    """Every proposed target, evaluated against data that exists."""
    out = [spatial_expansion_target(), rl_policy_target()]
    if with_responder:
        rt = responder_target()
        out.append(
            TargetVerdict(
                name="molecular responder classification",
                claimed="AUC 0.91-0.94",
                metric="ROC-AUC on a response label",
                status=TargetStatus.REFUSED_NO_LABEL,
                measured=rt.auc,
                n=rt.n_patients,
                detail=(
                    f"no response label exists: {rt.label.verdict}. On the nearest "
                    f"reachable label the baseline covariates reach out-of-fold "
                    f"AUC {rt.auc:.4f} (95% CI {rt.auc_ci[0]:.4f}-{rt.auc_ci[1]:.4f})"
                ),
                blockers=[
                    "`best_overall_response` is empty for every case",
                    "the reachable flag is recorded at follow-up, not as response",
                    (
                        "the flag is anti-correlated with survival (tumour-free "
                        "median OS 229 d against 374 d), so it tracks when it was "
                        "recorded rather than how the patient did"
                        if rt.label_tracks_survival_backwards
                        else "flag/survival direction checked"
                    ),
                    f"only {rt.n_positive} positives in {rt.n_patients} patients",
                ],
            )
        )
    if with_gdsc:
        out.append(in_vitro_target())
    if with_solver:
        out.append(solver_convergence_target())
    return out


def measured_ceiling_report(*, with_gdsc: bool = True, with_solver: bool = False) -> str:
    """The replace-the-pitch table: what each proposed target can bear."""
    verdicts = target_verdicts(with_gdsc=with_gdsc, with_solver=with_solver)
    lines = [
        "=" * 78,
        "PROPOSED 0.90+ TARGETS, EVALUATED AGAINST DATA THAT EXISTS",
        "=" * 78,
        "",
    ]
    for v in verdicts:
        lines.append(f"  {v.name}  ({v.metric})")
        lines.append(f"    claimed      : {v.claimed}")
        lines.append(f"    status       : {v.status.value}")
        lines.append(f"    measured     : {v.display}")
        lines.append(f"    n            : {v.n}")
        lines.append(f"    detail       : {v.detail}")
        if v.blockers:
            lines.append("    blockers     :")
            lines += [f"      - {b}" for b in v.blockers]
        lines.append("")
    measured = [v for v in verdicts if v.has_number]
    refused = [v for v in verdicts if not v.has_number]
    lines += [
        f"  {len(verdicts)} tier(s): {len(measured)} measurable, {len(refused)} refused.",
        "",
        "  Refusing a tier does not raise another. A measurable synthetic tier does",
        "  not raise a clinical one: the repository's own held-out numbers are",
        "  unchanged by this module -- OS C = 0.656 and progression C = 0.534, the",
        "  latter below age alone (0.539).",
    ]
    return "\n".join(lines)


def eval_targets_provenance() -> Dict[str, EvidenceStatus]:
    """What each quantity here is, in the repo's provenance vocabulary."""
    return {
        "spatial_dice_verdict": EvidenceStatus.PROVEN_FAILED,
        "best_overall_response_emptiness": EvidenceStatus.ESTABLISHED,
        "reachable_status_flag": EvidenceStatus.ESTABLISHED,
        "responder_auc": EvidenceStatus.ESTABLISHED,
        "gdsc_extrapolation_counts": EvidenceStatus.ESTABLISHED,
        "engine_ec50_unit_absence": EvidenceStatus.PROVEN_FAILED,
        "hill_slope_vs_measured_curves": EvidenceStatus.UNKNOWN,
        "dice_degenerate_case": EvidenceStatus.ESTABLISHED,
        "egfrviii_open_reachability": EvidenceStatus.PROVEN_FAILED,
        "rl_agent_absence": EvidenceStatus.PROVEN_FAILED,
    }


def what_targets_cannot_show() -> List[str]:
    return [
        "None of these three targets is a clinical endpoint. Spatial Dice and "
        "in-vitro IC50 agreement say nothing about a patient's outcome.",
        "A high R^2 or Dice score on a shifted estimand cannot be added to, "
        "averaged with, or used to raise an overall-survival C-index.",
        "The responder label that exists is a follow-up status flag, not a "
        "treatment response; scoring it well would not mean detecting responders.",
        "GDSC glioma IC50s are largely extrapolations, so even a perfect fit to "
        "them would be a fit to a lower bound.",
        "The engine's Hill slope is fixed at n=1 and has never been checked "
        "against measured dose-response curves.",
        "A self-defined reward-retention metric cannot be evidence: the reward "
        "and its normalization are chosen by the author, so no measurement can "
        "contradict the number it produces.",
        "EGFRvIII status is not obtainable from open GDC data, so it cannot be "
        "listed as a feature of a classifier built on this repository.",
    ]


__all__ = [
    "EGFRVIII_IS_STRUCTURAL",
    "EGFRVIII_REACHABILITY_PATH",
    "GDSC2_FITTED_PATH",
    "GDSC2_FITTED_SOURCE",
    "GDSC2_RAW_BYTES",
    "GDSC_GBM_TISSUE",
    "RULE_BASED_POLICIES",
    "DiceResult",
    "EgfrviiiReachability",
    "GdscAudit",
    "ResponderLabelAudit",
    "ResponderTarget",
    "TargetStatus",
    "TargetVerdict",
    "audit_responder_label",
    "dice_degenerate_case_report",
    "dice_similarity",
    "eval_targets_provenance",
    "in_vitro_target",
    "load_gdsc_glioma",
    "measured_ceiling_report",
    "probe_egfrviii_reachability",
    "responder_target",
    "rl_policy_target",
    "solver_convergence_target",
    "spatial_expansion_target",
    "target_verdicts",
    "what_targets_cannot_show",
]
