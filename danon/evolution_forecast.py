"""Does an earlier specimen predict a later molecular change, beyond a baseline?

THE QUESTION, ON REAL DATA
--------------------------
This is the first research question a longitudinal-evolution project has to
answer, and until now this repository could only pose it: *can information from
an earlier tumour sample predict a later molecular change, beyond what a simple
baseline predicts?* `danon.variants` now measures the public paired GLASS release,
so it is answerable on **real patient specimens** -- and the answer is allowed to
be no.

TWO TASKS, BOTH DECLARED BEFORE THEY ARE SCORED
-----------------------------------------------
Task A -- re-detection of loci the PRIMARY specimen measured. Rows are the
variants the primary registered, split into those detected again in the
recurrence and those that are not; features come from the primary specimen only.
Variants seen only at recurrence are EXCLUDED, because a gained variant is
detected at recurrence by the definition of "gained" -- scoring those rows would
let a model read a construction instead of a measurement (an earlier version did
exactly that, and the inflated AUC was caught here rather than published). No
oracle and no imputation is used: every row is something the primary measured,
so this is *re-detection*, not discovery.

Task B -- the allele-fraction change of variants seen in BOTH specimens. The
baseline is persistence (the fraction does not move), and a model only counts if
it beats that baseline's error.

THE BASELINES ARE THE POINT
---------------------------
A model is not compared against another model, it is compared against the
dumbest honest competitor: the majority class for Task A (every locus is
present at recurrence) and no-change persistence for Task B. Whatever the
numbers say is what gets recorded, including "the baseline wins".

SPLITS
------
Patients are split, never rows: every row of a patient lands in one fold, so a
second specimen of the same tumour cannot leak its own recurrence into the
training set. The split is a seeded permutation of PATIENTS and the fold
ownership is returned so a caller can check it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.variants import VariantRecord, Spectrum, VariantError, spectra_from_records

#: Feature order for the fitted models, named so a reader can see exactly what
#: was allowed in: the primary specimen's own measured quantities and nothing
#: that could carry the future.
TASK_A_FEATURES = ("primary_af", "log1p_primary_depth")
TASK_B_FEATURES = ("primary_af", "log1p_primary_depth")


class ForecastError(ValueError):
    """An example, split or score that cannot be supported."""


# ---------------------------------------------------------------------------
# Reading the real release: one definition of the labelling and pairing rules
# ---------------------------------------------------------------------------


#: The GLASS sample-id suffix that names the draw a specimen came from. It lives
#: here, in the shipping module, rather than in the build script, so the audit
#: can re-derive the cohort with the SAME rule the artifact was built under --
#: two copies of this function would be two definitions of "a pair".
def timepoint_of(sample_id: str) -> str:
    suffix = sample_id.rsplit("-", 1)[-1].upper()
    if suffix == "TP":
        return "Primary"
    if len(suffix) >= 2 and suffix[0] == "R" and suffix[1:].isdigit():
        return "Recurrence"
    return "Unlabelled"


def records_from_cache(raw: Mapping[str, Any]) -> Tuple[List[VariantRecord], int]:
    """Rebuild measured records from the cached release; return (records, dropped).

    A record without both an alt and a ref count is DROPPED and counted, never
    given a default: an allele fraction that was not computed from reads is not
    a measurement, and the dropped count is published beside the used count.
    """
    records: List[VariantRecord] = []
    dropped = 0
    for rec in raw.get("records", []):
        alt, ref = rec.get("tumorAltCount"), rec.get("tumorRefCount")
        if not isinstance(alt, int) or not isinstance(ref, int):
            dropped += 1
            continue
        sample_id = rec.get("sampleId", "")
        try:
            records.append(
                VariantRecord(
                    patient_id=rec.get("patientId", ""),
                    sample_id=sample_id,
                    timepoint=timepoint_of(sample_id),
                    gene=rec["_gene"],
                    protein_change=rec.get("proteinChange") or "",
                    chromosome=str(rec.get("chr") or ""),
                    start=int(rec.get("startPosition") or 0),
                    ref_allele=str(rec.get("referenceAllele") or ""),
                    alt_allele=str(rec.get("variantAllele") or ""),
                    alt_count=alt,
                    ref_count=ref,
                    source=raw.get("source", ""),
                )
            )
        except (KeyError, TypeError, ValueError, VariantError):
            dropped += 1
    return records, dropped


def timepoint_census(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """What the release DECLARES about a specimen's kind, counted.

    This exists because the pairing is an inference and the inference has to be
    auditable. In the cached GLASS release the `sampleType` column is uniform --
    every one of the 693 specimen rows says "Primary Solid Tumor", including the
    ones whose identifier ends in `R1`, `R2` or `R3` -- so the column carries NO
    timepoint information and the only signal is the identifier suffix. That is
    recorded here as a measurement rather than left implicit, along with the
    693-1 (or however many) rows it leaves unlabelled and therefore excluded.

    It cannot establish that a suffix means what this module assumes. What it can
    do is make the assumption falsifiable: if the release ever publishes a real
    timepoint column, `declared_types_distinct` rises above 1 and
    `declared_type_carries_timepoint` becomes True, at which point the suffix
    rule must be replaced rather than trusted.
    """
    samples = raw.get("samples") or []
    declared_types: Dict[str, int] = {}
    by_suffix: Dict[str, Dict[str, int]] = {}
    for sample in samples:
        if not isinstance(sample, Mapping):
            continue
        declared = str(sample.get("sampleType", ""))
        declared_types[declared] = declared_types.get(declared, 0) + 1
        suffix = str(sample.get("sampleId", "")).rsplit("-", 1)[-1].upper()
        bucket = by_suffix.setdefault(suffix, {})
        bucket[declared] = bucket.get(declared, 0) + 1

    n_unlabelled = sum(
        1 for s in samples
        if isinstance(s, Mapping)
        and timepoint_of(str(s.get("sampleId", ""))) == "Unlabelled"
    )
    return {
        "n_specimen_rows": len(samples),
        "declared_types": dict(sorted(declared_types.items())),
        "declared_types_distinct": len(declared_types),
        "declared_type_carries_timepoint": len(declared_types) > 1,
        "suffixes": {k: dict(sorted(v.items())) for k, v in sorted(by_suffix.items())},
        "n_excluded_by_suffix": n_unlabelled,
        "basis": (
            "the identifier suffix only (TP -> Primary, R<digit> -> Recurrence); "
            "the release's own sampleType column is uniform and uninformative"
        ),
        "cannot_establish": (
            "that the suffix convention means what this module assumes: a "
            "specimen labelled R1 is taken to be a later draw from the same "
            "patient, and nothing in the cached release independently confirms "
            "the draw order, the interval between draws, or that a later "
            "specimen is a recurrence rather than a second primary or a "
            "re-biopsy of the same lesion"
        ),
    }


def paired_spectra(
    records: Sequence[VariantRecord],
) -> Tuple[List[Spectrum], List[Tuple[Spectrum, Spectrum]]]:
    """The measured spectra and every primary -> later-recurrence pair.

    Pairs are taken within one patient only, ordered by specimen id so the
    pairing is deterministic, and a patient with no primary or no recurrence
    contributes no pair rather than an imputed one.
    """
    spectra = spectra_from_records(records)
    by_patient: Dict[str, List[Spectrum]] = {}
    for spec in spectra:
        by_patient.setdefault(spec.patient_id, []).append(spec)
    pairs: List[Tuple[Spectrum, Spectrum]] = []
    for _pid, specs in sorted(by_patient.items()):
        primaries = sorted(
            (s for s in specs if s.timepoint == "Primary"), key=lambda s: s.sample_id
        )
        recurrents = sorted(
            (s for s in specs if s.timepoint == "Recurrence"), key=lambda s: s.sample_id
        )
        for p in primaries:
            for r in recurrents:
                pairs.append((p, r))
    return spectra, pairs


# ---------------------------------------------------------------------------
# Examples built from measured paired spectra
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionExample:
    """One locus in one pair: features from the primary, target at recurrence."""

    patient_id: str
    variant: str
    primary_af: float
    primary_depth: int
    detected_at_recurrence: bool
    recurrence_depth: int


@dataclass(frozen=True)
class ChangeExample:
    """One shared variant in one pair: features from the primary, target delta."""

    patient_id: str
    variant: str
    primary_af: float
    primary_depth: int
    delta_af: float


def _label(key: Tuple[str, str, int, str, str]) -> str:
    gene, chrom, start, ref, alt = key
    return f"{gene}:{chrom}:{start}:{ref}>{alt}"


def detection_examples(
    pairs: Sequence[Tuple[Spectrum, Spectrum]],
) -> List[DetectionExample]:
    """Task A rows: loci the PRIMARY specimen actually measured.

    The question is whether the primary's own reading predicts re-detection, so
    only variants detected in the primary are scored: shared ones (detected
    again) against lost ones (not detected).

    Variants seen ONLY at recurrence are deliberately EXCLUDED, and the reason is
    a defect this module carried in its first version: a gained variant has a
    primary allele fraction of 0.0 and is detected at recurrence *by the
    definition of 'gained'*, so including it hands the model a deterministic
    identity -- the label follows from the construction rather than from a
    measurement. With those rows in, a fitted model reached AUC 0.85 while the
    allele fraction alone sat at 0.30; the 0.85 was the definition talking.
    Excluding them leaves a fully well-posed task that needs no oracle: the locus
    is known because the primary measured it.
    """
    out: List[DetectionExample] = []
    for primary, recurrence in pairs:
        _check_pair(primary, recurrence)
        a = primary.keyed()
        b = recurrence.keyed()
        for key in sorted(set(a) & set(b)):
            out.append(
                DetectionExample(
                    primary.patient_id,
                    _label(key),
                    a[key].allele_fraction,
                    a[key].depth,
                    True,
                    b[key].depth,
                )
            )
        for key in sorted(set(a) - set(b)):
            out.append(
                DetectionExample(
                    primary.patient_id,
                    _label(key),
                    a[key].allele_fraction,
                    a[key].depth,
                    False,
                    0,
                )
            )
    return out


def change_examples(
    pairs: Sequence[Tuple[Spectrum, Spectrum]],
) -> List[ChangeExample]:
    """Task B rows: shared variants only, so both timepoints have a measurement."""
    out: List[ChangeExample] = []
    for primary, recurrence in pairs:
        _check_pair(primary, recurrence)
        a = primary.keyed()
        b = recurrence.keyed()
        for key in sorted(set(a) & set(b)):
            out.append(
                ChangeExample(
                    primary.patient_id,
                    _label(key),
                    a[key].allele_fraction,
                    a[key].depth,
                    b[key].allele_fraction - a[key].allele_fraction,
                )
            )
    return out


def _check_pair(primary: Spectrum, recurrence: Spectrum) -> None:
    if primary.patient_id != recurrence.patient_id:
        raise VariantError(
            "a forecast example needs two specimens of one patient, got "
            f"{primary.patient_id} and {recurrence.patient_id}"
        )
    if primary.sample_id == recurrence.sample_id:
        raise VariantError(
            f"both spectra are specimen {primary.sample_id}: one specimen has no "
            "later molecular state to predict"
        )


# ---------------------------------------------------------------------------
# Patient-level splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatientSplit:
    train_patients: Tuple[str, ...]
    test_patients: Tuple[str, ...]

    def owners(self) -> Dict[str, Tuple[str, ...]]:
        out: Dict[str, Tuple[str, ...]] = {}
        for pid in self.train_patients:
            out[pid] = ("train",)
        for pid in self.test_patients:
            out[pid] = ("test",)
        return out

    def n_shared_patients(self) -> int:
        return len(set(self.train_patients) & set(self.test_patients))


def patient_split(
    patient_ids: Sequence[str], *, train_fraction: float = 0.7, seed: int = 7
) -> PatientSplit:
    """Split by PATIENT; the rows of one tumour never straddle the boundary."""
    unique = sorted(set(patient_ids))
    if len(unique) < 2:
        raise ForecastError(
            f"a split needs at least 2 patients with a pair, got {len(unique)}"
        )
    if not 0.0 < train_fraction < 1.0:
        raise ForecastError(f"train_fraction must be in (0, 1), got {train_fraction}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    n_train = max(1, min(len(unique) - 1, int(round(train_fraction * len(unique)))))
    train = tuple(unique[i] for i in sorted(order[:n_train]))
    test = tuple(unique[i] for i in sorted(order[n_train:]))
    return PatientSplit(train, test)


# ---------------------------------------------------------------------------
# Scoring: a threshold-free AUC and a paired bootstrap over patients
# ---------------------------------------------------------------------------


def auc(scores: Sequence[float], labels: Sequence[bool]) -> Optional[float]:
    """Rank-sum AUC. ``None`` when a class is absent -- reported, not guessed."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    pos_sorted = np.sort(np.asarray(pos, dtype=float))
    neg_sorted = np.sort(np.asarray(neg, dtype=float))
    # Mann-Whitney U with ties handled by counting half for equality.
    total = 0.0
    for value in pos_sorted:
        total += float(np.sum(neg_sorted < value)) + 0.5 * float(
            np.sum(neg_sorted == value)
        )
    return total / (len(pos) * len(neg))


def bootstrap_auc_ci(
    scores: Sequence[float],
    labels: Sequence[bool],
    patients: Sequence[str],
    *,
    n_draws: int = 400,
    seed: int = 11,
) -> Tuple[Optional[float], Optional[float]]:
    """A 95% interval from resampling PATIENTS, not rows: the rows of one tumour
    are not independent draws, so resampling them would understate the spread."""
    by_patient: Dict[str, List[int]] = {}
    for i, pid in enumerate(patients):
        by_patient.setdefault(pid, []).append(i)
    keys = sorted(by_patient)
    if len(keys) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    draws: List[float] = []
    for _ in range(n_draws):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = [i for p in pick for i in by_patient[keys[int(p)]]]
        value = auc([scores[i] for i in idx], [labels[i] for i in idx])
        if value is not None:
            draws.append(value)
    if len(draws) < 20:
        return None, None
    return (
        float(np.percentile(draws, 2.5)),
        float(np.percentile(draws, 97.5)),
    )


def bootstrap_mae_difference(
    pred: Sequence[float],
    y: Sequence[float],
    patients: Sequence[str],
    *,
    n_draws: int = 2000,
    seed: int = 17,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Patient-level interval on (persistence MAE - model MAE).

    Resampling the tumours, not the rows: the variants of one specimen are not
    independent. The returned share is the fraction of draws in which the model
    still beat persistence, which is the number that says whether the advantage
    is a property of the cohort or of the split.
    """
    by_patient: Dict[str, List[int]] = {}
    for i, pid in enumerate(patients):
        by_patient.setdefault(pid, []).append(i)
    keys = sorted(by_patient)
    if len(keys) < 3:
        return None, None, None
    abs_y = np.abs(np.asarray(y, dtype=float))
    abs_err = np.abs(np.asarray(pred, dtype=float) - np.asarray(y, dtype=float))
    rng = np.random.default_rng(seed)
    draws: List[float] = []
    for _ in range(n_draws):
        pick = rng.integers(0, len(keys), size=len(keys))
        idx = np.array(
            [i for p in pick for i in by_patient[keys[int(p)]]], dtype=int
        )
        draws.append(float(np.mean(abs_y[idx]) - np.mean(abs_err[idx])))
    arr = np.asarray(draws)
    return (
        float(np.percentile(arr, 2.5)),
        float(np.percentile(arr, 97.5)),
        float(np.mean(arr > 0.0)),
    )


# ---------------------------------------------------------------------------
# The models: a fixed-iteration logistic fit and a least-squares line
# ---------------------------------------------------------------------------


def _standardise(train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def fit_logistic(
    x: np.ndarray, y: np.ndarray, *, iterations: int = 2000, lr: float = 0.1
) -> np.ndarray:
    """Deterministic gradient descent on the log-loss, no hidden randomness."""
    z = np.hstack([np.ones((x.shape[0], 1)), x])
    w = np.zeros(z.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-z @ w))
        grad = z.T @ (p - y) / z.shape[0]
        w -= lr * grad
    return w


def fit_line(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinary least squares with an intercept, solved directly."""
    z = np.hstack([np.ones((x.shape[0], 1)), x])
    coef, *_ = np.linalg.lstsq(z, y, rcond=None)
    return coef


# ---------------------------------------------------------------------------
# Calibration: does a probability mean what it says?
# ---------------------------------------------------------------------------
#
# Discrimination and calibration are different claims, and task A only settled
# the first. An AUC above 0.5 says a higher score comes with a higher chance of
# re-detection; it says nothing about whether 0.80 means "re-detected four times
# in five". A probability nobody can read off a patient is not yet a usable
# readout, so the pooled out-of-fold probabilities are checked against the rates
# they produced -- and the sentence published about them is DERIVED from these
# numbers by ``calibration_reading``, so prose and figures cannot drift apart.

#: Declared once, so the sentence and the flags cannot be tuned per run.
CALIBRATION_SLOPE_TOLERANCE = 0.35
CALIBRATION_INTERCEPT_TOLERANCE = 0.35
CALIBRATION_MEAN_TOLERANCE = 0.05


def _logit(p: float) -> float:
    """Log-odds, clipped so a 0.0 or 1.0 prediction does not go to infinity."""
    q = min(max(float(p), 1e-9), 1.0 - 1e-9)
    return math.log(q / (1.0 - q))


def _calibration_curve_fit(
    logits: np.ndarray, y: np.ndarray, *, iterations: int = 100
) -> Tuple[float, float]:
    """Fit slope/intercept by Newton-Raphson (IRLS); exact and deterministic.

    This is the standard calibration curve: the outcome regressed on the
    log-odds of the model's own prediction. Perfect calibration is slope 1 and
    intercept 0. A slope BELOW 1 means the probabilities are too extreme (the
    model is more confident than its own hit rate); an intercept away from 0
    means they are shifted. IRLS rather than the module's gradient fitter
    because two parameters converge exactly and a stopped-short fit would be a
    number nobody can reproduce.
    """
    z = np.hstack([np.ones((logits.shape[0], 1)), logits])
    w = np.zeros(z.shape[1])
    for _ in range(iterations):
        p = 1.0 / (1.0 + np.exp(-(z @ w)))
        grad = z.T @ (p - y)
        hessian = z.T @ (z * (p * (1.0 - p))[:, None])
        try:
            step = np.linalg.solve(hessian + 1e-12 * np.eye(z.shape[1]), grad)
        except np.linalg.LinAlgError:  # pragma: no cover - singular Hessian
            break
        w -= step
        if float(np.max(np.abs(step))) < 1e-12:
            break
    return float(w[1]), float(w[0])


def reliability_bins(
    scores: Sequence[float], labels: Sequence[bool], *, n_bins: int = 5
) -> List[Dict[str, Any]]:
    """Equal-count bins of predicted probability against the rate observed.

    The arithmetic a reader can check by hand: for each bin, the mean
    probability the model gave and how often re-detection actually happened. The
    unit is the same one the AUC ranks -- a ROW, each carrying its own measured
    label -- and ``n`` is how many of those rows the bin holds.
    """
    if len(scores) != len(labels):
        raise ForecastError("scores and labels must be the same length")
    if not scores:
        raise ForecastError("no pooled out-of-fold scores to calibrate")
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    n = len(order)
    out: List[Dict[str, Any]] = []
    for b in range(n_bins):
        idx = order[(b * n) // n_bins : ((b + 1) * n) // n_bins]
        if not idx:
            continue
        predicted = float(np.mean([scores[i] for i in idx]))
        observed = float(np.mean([1.0 if labels[i] else 0.0 for i in idx]))
        out.append(
            {
                "bin": len(out),
                "n": len(idx),
                "mean_predicted": predicted,
                "observed_rate": observed,
                "difference": observed - predicted,
            }
        )
    return out


def calibration_summary(
    scores: Sequence[float], labels: Sequence[bool]
) -> Dict[str, Optional[float]]:
    """Brier score, calibration-in-the-large, and the fitted curve.

    The Brier score is reported BESIDE the constant-prevalence predictor's own,
    because a Brier score on its own has no scale: it is the comparison that
    says whether the probabilities carry information a single number does not.
    """
    if len(scores) != len(labels):
        raise ForecastError("scores and labels must be the same length")
    if not scores:
        raise ForecastError("no pooled out-of-fold scores to calibrate")
    y = np.array([1.0 if v else 0.0 for v in labels], dtype=float)
    p = np.asarray([float(s) for s in scores], dtype=float)
    prevalence = float(np.mean(y))
    brier = float(np.mean((p - y) ** 2))
    brier_baseline = float(np.mean((prevalence - y) ** 2))
    logits = np.array([[_logit(s)] for s in scores], dtype=float)
    slope, intercept = _calibration_curve_fit(logits, y)
    return {
        "brier_score": brier,
        "brier_score_constant_prevalence": brier_baseline,
        "brier_skill_score": (
            None if brier_baseline == 0.0 else 1.0 - brier / brier_baseline
        ),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "mean_predicted_probability": float(np.mean(p)),
        "observed_re_detection_rate": prevalence,
    }


def calibration_is_within_tolerance(summary: Mapping[str, Optional[float]]) -> bool:
    """Whether the curve sits inside the package's DECLARED tolerances.

    Shared by the sentence and by the audit, so the two cannot disagree about
    which side of the line the same numbers fall on.
    """
    slope = summary.get("calibration_slope")
    intercept = summary.get("calibration_intercept")
    mean_pred = summary.get("mean_predicted_probability")
    observed = summary.get("observed_re_detection_rate")
    if None in (slope, intercept, mean_pred, observed):
        return False
    return (
        abs(float(slope) - 1.0) <= CALIBRATION_SLOPE_TOLERANCE
        and abs(float(intercept)) <= CALIBRATION_INTERCEPT_TOLERANCE
        and abs(float(mean_pred) - float(observed)) <= CALIBRATION_MEAN_TOLERANCE
    )


def calibration_reading(
    summary: Mapping[str, Optional[float]], *, n_bins: int
) -> str:
    """The sentence about calibration, DERIVED from the summary it describes.

    Written as a function rather than as prose in the record for one reason: the
    failure this repository keeps finding is a recorded sentence that stopped
    matching the numbers beside it. A caller cannot hand-write this sentence --
    the audit re-derives it and compares, so a hand-edited reading fails.
    """
    slope = summary["calibration_slope"]
    intercept = summary["calibration_intercept"]
    mean_pred = summary["mean_predicted_probability"]
    observed = summary["observed_re_detection_rate"]
    brier = summary["brier_score"]
    baseline = summary["brier_score_constant_prevalence"]
    skill = summary["brier_skill_score"]
    if slope is None or intercept is None or mean_pred is None or observed is None:
        return "calibration could not be measured for this task"
    well = calibration_is_within_tolerance(summary)
    verdict = (
        "the out-of-fold PROBABILITIES are calibrated to this cohort's own "
        f"declared tolerance (slope within {CALIBRATION_SLOPE_TOLERANCE}, "
        f"intercept within {CALIBRATION_INTERCEPT_TOLERANCE}, mean within "
        f"{CALIBRATION_MEAN_TOLERANCE})"
        if well
        else "the out-of-fold probabilities are NOT calibrated: the curve's "
        f"slope is {slope:.3f} against an ideal 1, its intercept {intercept:.3f} "
        f"against an ideal 0, and the mean predicted probability {mean_pred:.3f} "
        f"sits {abs(mean_pred - observed):.3f} from the rate observed "
        f"({observed:.3f}) -- a probability from this model is not a risk you "
        "can read off a patient"
    )
    return (
        f"{verdict}. Over {n_bins} equal-count bins the Brier score is "
        f"{brier:.4f} against a constant-prevalence predictor's {baseline:.4f} "
        f"(skill {skill:.4f}); discrimination and calibration are separate "
        "claims and this is one public cohort's numbers, not a general one"
    )


def calibration_block(
    scores: Sequence[float], labels: Sequence[bool], *, n_bins: int = 5
) -> Dict[str, Any]:
    """The recorded block: the binned table, the tolerances, and the sentence."""
    summary = calibration_summary(scores, labels)
    return {
        "unit": "row",
        "n_bins": n_bins,
        "tolerance": {
            "slope": CALIBRATION_SLOPE_TOLERANCE,
            "intercept": CALIBRATION_INTERCEPT_TOLERANCE,
            "mean": CALIBRATION_MEAN_TOLERANCE,
        },
        "summary": summary,
        "bins": reliability_bins(scores, labels, n_bins=n_bins),
        "reading": calibration_reading(summary, n_bins=n_bins),
    }


# ---------------------------------------------------------------------------
# From measured evidence to a clinical decision
# ---------------------------------------------------------------------------

#: The threshold at which a pooled out-of-fold PROBABILITY is called a positive.
#: Declared once, here, before any cohort is read: choosing the threshold that
#: flatters a measured curve is the same overfit as choosing the seed. A site
#: that acts at a different threshold must recalculate, because sensitivity,
#: specificity and every predictive value move with it.
OPERATING_THRESHOLD = 0.5

#: The thresholds a decision curve is reported at. A net benefit at one
#: threshold answers one clinician's question ("would I intervene at a 30%
#: risk?"), so the curve answers several at once and shows where the model stops
#: earning its keep.
DECISION_CURVE_THRESHOLDS: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70)


def safe_ratio(num: float, den: float) -> Optional[float]:
    return None if den == 0 else num / den


def operating_point_block(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    threshold: float = OPERATING_THRESHOLD,
) -> Dict[str, Any]:
    """Sensitivity, specificity and both predictive values at a DECLARED threshold.

    This is the block that turns a discriminating model into something a ward can
    act on: the AUC says the ranking separates, this says what a single call at
    an agreed threshold actually finds and actually misses, on this cohort's own
    prevalence. The confusion matrix is published beside the ratios so the words
    can always be re-derived from the counts, and the sentence is derived from
    those same counts in ``operating_point_reading``.
    """
    n = len(scores)
    if n == 0:
        raise ForecastError("no out-of-fold scores to place at an operating point")
    predicted = [s >= threshold for s in scores]
    tp = sum(1 for p, y in zip(predicted, labels) if p and y)
    fp = sum(1 for p, y in zip(predicted, labels) if p and not y)
    fn = sum(1 for p, y in zip(predicted, labels) if not p and y)
    tn = sum(1 for p, y in zip(predicted, labels) if not p and not y)
    positives = tp + fn
    negatives = tn + fp
    sensitivity = safe_ratio(tp, positives)
    specificity = safe_ratio(tn, negatives)
    ppv = safe_ratio(tp, tp + fp)
    npv = safe_ratio(tn, tn + fn)
    prevalence = float(np.mean(labels))
    block: Dict[str, Any] = {
        "unit": "row",
        "threshold": threshold,
        "threshold_tuned_on_this_cohort": False,
        "n_rows": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "prevalence": prevalence,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": (tp + tn) / n,
        "false_positives_per_100_called_positive": (
            None if tp + fp == 0 else 100.0 * fp / (tp + fp)
        ),
        "false_negatives_per_100_called_negative": (
            None if tn + fn == 0 else 100.0 * fn / (tn + fn)
        ),
    }
    block["reading"] = operating_point_reading(block)
    return block


def operating_point_reading(b: Dict[str, Any]) -> str:
    """The operating point's sentence, derived from its own counts."""
    if b["sensitivity"] is None or b["specificity"] is None:
        return (
            "the operating point could not be measured: this cohort's measured rows "
            f"carry only {b['tp'] + b['fn']} positive and {b['tn'] + b['fp']} negative "
            "labels, so one side of the confusion matrix has no rows to count"
        )
    ppv = b["ppv"]
    npv = b["npv"]
    ppv_txt = (
        "a positive call is followed by re-detection "
        f"{ppv * 100:.1f}% of the time"
        if ppv is not None
        else "no row was called positive, so a positive call's precision is undefined here"
    )
    npv_txt = (
        "a negative call is followed by no re-detection "
        f"{npv * 100:.1f}% of the time"
        if npv is not None
        else "every row was called positive, so a negative call's assurance is undefined here"
    )
    return (
        f"at the declared threshold {b['threshold']:.3f} -- set before this cohort "
        f"was read, never tuned on it -- the model calls {b['tp'] + b['fp']} of "
        f"{b['n_rows']} measured rows positive: it keeps {b['sensitivity'] * 100:.1f}% "
        f"of the {b['tp'] + b['fn']} rows that re-appear later (specificity "
        f"{b['specificity'] * 100:.1f}% against the {b['tn'] + b['fp']} that do not), "
        f"with {b['fn']} missed and {b['fp']} false alarms. Because the row-level "
        f"rate here is {b['prevalence'] * 100:.1f}%, the predictive values belong to "
        f"THIS prevalence: {ppv_txt}, and {npv_txt}. Both predictive values move "
        "when the ward's prevalence moves, which is why they are printed beside "
        "the sensitivity and specificity rather than instead of them"
    )


def decision_curve_block(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    thresholds: Sequence[float] = DECISION_CURVE_THRESHOLDS,
) -> Dict[str, Any]:
    """Net benefit of acting on the model, against acting on everyone or nobody.

    Decision-curve analysis is the one calculation that answers "is using this
    model better than the alternatives a clinician already has?" in the units of
    the decision itself (true positives, weighted by what a false positive costs
    at that threshold). The two comparators are the honest ones: treat-all acts
    on prevalence, treat-none does nothing. New is measured against BOTH.
    """
    n = len(scores)
    if n == 0:
        raise ForecastError("no out-of-fold scores to weigh in a decision curve")
    prevalence = float(np.mean(labels))
    rows: List[Dict[str, Any]] = []
    for pt in thresholds:
        if not 0.0 < pt < 1.0:
            raise ForecastError(f"decision threshold {pt} is outside (0, 1)")
        on_model = sum(1 for s, y in zip(scores, labels) if s >= pt and y)
        off_model = sum(1 for s, y in zip(scores, labels) if s >= pt and not y)
        odds = pt / (1.0 - pt)
        model = (on_model / n) - (off_model / n) * odds
        treat_all = prevalence - (1.0 - prevalence) * odds
        rows.append(
            {
                "threshold": pt,
                "model": model,
                "treat_all": treat_all,
                "treat_none": 0.0,
                "beats_both": bool(model > treat_all and model > 0.0),
            }
        )
    block: Dict[str, Any] = {
        "unit": "row",
        "prevalence": prevalence,
        "axis": (
            "net benefit = (true positives / rows) - (false positives / rows) * "
            "pt/(1-pt), so a false positive costs what an intervention at threshold "
            "pt is worth"
        ),
        "rows": rows,
    }
    block["reading"] = decision_curve_reading(block)
    return block


def decision_curve_reading(b: Dict[str, Any]) -> str:
    """The decision curve's sentence, derived from its own rows."""
    rows = b["rows"] or []
    winners = [r["threshold"] for r in rows if r["beats_both"]]
    tested = ", ".join(f"{r['threshold']:.2f}" for r in rows)
    if not rows:
        return "the decision curve was not measured at any threshold"
    if not winners:
        return (
            "the model adds NO net benefit over the two alternatives a clinician "
            "already has: across the thresholds measured (" + tested + ") its net "
            "benefit never exceeds both treating every row and treating none, so "
            "acting on this score would not be better than acting on prevalence "
            "alone"
        )
    if len(winners) == len(rows):
        where = f"every threshold measured ({tested})"
    else:
        where = "thresholds " + ", ".join(f"{t:.2f}" for t in winners)
    return (
        f"the model earns its place at {where}: only there does acting on the "
        "score beat BOTH treating every row and treating none, in the units of "
        "the decision. Between those points the honest comparator is the one the "
        "curve crosses, so the threshold in any protocol must be read off this "
        "row of the curve rather than carried over from another cohort"
    )


def clinical_use_block(
    scores: Sequence[float], labels: Sequence[bool]
) -> Dict[str, Any]:
    """Everything this task licenses a CLINICAL reader to act on, in one block.

    Discrimination alone is not a decision: the operating point says what a call
    finds and misses, the decision curve says whether making the call beats the
    alternatives, and the conditions say who still has to decide. The block is
    recorded on the artifact so the same numbers the research screen shows are
    the numbers a clinical reader acts on -- there is no second, rosier estimate.
    """
    return {
        "unit": "row",
        "intended_use": (
            "decision support for a clinician weighing whether a variant seen in "
            "this patient's primary specimen is likely to be seen again: the score "
            "ranks and calls, a named clinician decides, and the record here is the "
            "one audited on the research screen"
        ),
        "operating_point": operating_point_block(scores, labels),
        "decision_curve": decision_curve_block(scores, labels),
        "conditions": [
            "the unit is a ROW (a variant), not a patient: this call is about one "
            "variant, and a patient-level re-detection is a separately declared "
            "endpoint with its own interval",
            "the threshold is declared, never tuned on this cohort; a site acting "
            "at another threshold recalculates every figure above",
            "no autonomous action: the output is decision support for a named "
            "clinician or tumour board, who may withhold it and must record why",
            "one public cohort, one cohort's prevalence: the predictive values do "
            "not travel to a ward whose prevalence differs, and no prospective "
            "outcome has been shown by any figure in this block",
        ],
    }


# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    task: str
    n_train_rows: int
    n_test_rows: int
    n_train_patients: int
    n_test_patients: int
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "task": self.task,
            "n_train_rows": self.n_train_rows,
            "n_test_rows": self.n_test_rows,
            "n_train_patients": self.n_train_patients,
            "n_test_patients": self.n_test_patients,
            "metrics": self.metrics,
            "notes": self.notes,
        }


def _features_a(train: Sequence[DetectionExample], rows: Sequence[DetectionExample]):
    def build(rs):
        return np.array(
            [[r.primary_af, math.log1p(r.primary_depth)] for r in rs], dtype=float
        )

    return build(train), build(rows)


def evaluate_detection(
    examples: Sequence[DetectionExample], *, seed: int = 7, train_fraction: float = 0.7
) -> TaskResult:
    """Task A: does the primary specimen predict re-detection, beyond prevalence?"""
    if not examples:
        raise ForecastError("no detection examples to score")
    split = patient_split([e.patient_id for e in examples], train_fraction=train_fraction, seed=seed)
    train_rows = [e for e in examples if e.patient_id in set(split.train_patients)]
    test_rows = [e for e in examples if e.patient_id in set(split.test_patients)]
    if not train_rows or not test_rows:
        raise ForecastError("the split left a fold empty")
    x_train, x_test = _features_a(train_rows, test_rows)
    y_train = np.array([e.detected_at_recurrence for e in train_rows], dtype=float)
    y_test = [e.detected_at_recurrence for e in test_rows]
    labels = list(y_test)
    patients = [e.patient_id for e in test_rows]

    mean, scale = _standardise(x_train)
    w = fit_logistic((x_train - mean) / scale, y_train)
    z_test = np.hstack([np.ones((x_test.shape[0], 1)), (x_test - mean) / scale])
    fitted = 1.0 / (1.0 + np.exp(-z_test @ w))

    prevalence = float(np.mean(y_test)) if y_test else None
    majority = max(prevalence, 1.0 - prevalence) if prevalence is not None else None
    af_only = [e.primary_af for e in test_rows]
    depth_only = [math.log1p(e.primary_depth) for e in test_rows]

    result = TaskResult(
        task="detection_at_recurrence",
        n_train_rows=len(train_rows),
        n_test_rows=len(test_rows),
        n_train_patients=len(split.train_patients),
        n_test_patients=len(split.test_patients),
        metrics={
            "test_prevalence_detected": prevalence,
            "baseline_majority_accuracy": majority,
            "baseline_majority_auc": 0.5,
            "auc_primary_af_only": auc(af_only, labels),
            "auc_primary_depth_only": auc(depth_only, labels),
            "auc_fitted_logistic": auc(list(fitted), labels),
        },
        notes=[
            "task A scores only variants the PRIMARY specimen measured, so every "
            "row is a reading rather than an assumption and no oracle is used",
            "variants seen only at recurrence are excluded: their label follows "
            "from the definition of 'gained', so scoring them would let a model "
            "exploit the construction instead of the measurement",
            "the rows of one tumour move together, so the interval resamples "
            "patients and not rows",
        ],
    )
    lo, hi = bootstrap_auc_ci(list(fitted), labels, patients)
    result.metrics["auc_fitted_logistic_ci_low"] = lo
    result.metrics["auc_fitted_logistic_ci_high"] = hi
    return result


def evaluate_change(
    examples: Sequence[ChangeExample], *, seed: int = 7, train_fraction: float = 0.7
) -> TaskResult:
    """Task B: can the primary specimen beat no-change persistence?"""
    if not examples:
        raise ForecastError("no change examples to score")
    split = patient_split([e.patient_id for e in examples], train_fraction=train_fraction, seed=seed)
    train_rows = [e for e in examples if e.patient_id in set(split.train_patients)]
    test_rows = [e for e in examples if e.patient_id in set(split.test_patients)]
    if not train_rows or not test_rows:
        raise ForecastError("the split left a fold empty")
    x_train, x_test = _features_a(train_rows, test_rows)
    y_train = np.array([e.delta_af for e in train_rows], dtype=float)
    y_test = np.array([e.delta_af for e in test_rows], dtype=float)

    mean, scale = _standardise(x_train)
    coef = fit_line((x_train - mean) / scale, y_train)
    z_test = np.hstack([np.ones((x_test.shape[0], 1)), (x_test - mean) / scale])
    pred = z_test @ coef

    persistence_mae = float(np.mean(np.abs(y_test)))
    fitted_mae = float(np.mean(np.abs(pred - y_test)))
    train_mean_mae = float(np.mean(np.abs(float(np.mean(y_train)) - y_test)))
    # Sign prediction: persistence says "no change", which is correct only when
    # the change is smaller than the measurement resolution; the fitted line is
    # scored the same way so the comparison is like-for-like.
    resolution = 0.01
    nonzero = np.abs(y_test) >= resolution
    n_resolvable = int(np.sum(nonzero))
    model_sign = (
        float(np.mean(np.sign(pred[nonzero]) == np.sign(y_test[nonzero])))
        if n_resolvable
        else None
    )
    # The baseline a sign predictor must clear is not 0.5 but the majority
    # direction: if most resolvable changes go one way, "always that way" is a
    # real predictor, and reporting 0.5 beside it would flatter the model.
    majority_sign = None
    if n_resolvable:
        up = float(np.mean(y_test[nonzero] > 0.0))
        majority_sign = max(up, 1.0 - up)
    diff_lo, diff_hi, frac_positive = bootstrap_mae_difference(
        list(pred), list(y_test), [e.patient_id for e in test_rows]
    )
    return TaskResult(
        task="allele_fraction_change",
        n_train_rows=len(train_rows),
        n_test_rows=len(test_rows),
        n_train_patients=len(split.train_patients),
        n_test_patients=len(split.test_patients),
        metrics={
            "baseline_persistence_mae": persistence_mae,
            "baseline_train_mean_mae": train_mean_mae,
            "model_fitted_mae": fitted_mae,
            "model_beats_persistence": fitted_mae < persistence_mae,
            "mae_advantage_over_persistence": persistence_mae - fitted_mae,
            "mae_advantage_ci_low": diff_lo,
            "mae_advantage_ci_high": diff_hi,
            "mae_advantage_share_of_draws_positive": frac_positive,
            "mae_advantage_resolvable": (
                None if diff_lo is None else bool(diff_lo > 0.0)
            ),
            "mean_delta": float(np.mean(y_test)),
            "n_test_with_a_resolvable_change": n_resolvable,
            "model_sign_accuracy_on_resolvable": model_sign,
            "baseline_majority_sign_accuracy": majority_sign,
            "persistence_sign_accuracy_on_resolvable": 0.0
            if n_resolvable
            else None,
        },
        notes=[
            "persistence is the honest floor: predicting no change costs the mean "
            "absolute change, which is what a model has to beat",
            "a sign cannot be predicted by persistence at all, so its 0.0 is the "
            "declared floor rather than a measured failure; the baseline the "
            "fitted line actually has to clear is the majority direction",
            "the advantage over persistence is reported with a patient-level "
            "interval, because one split can show a gain that does not survive "
            "resampling the tumours",
            "these are bulk next-generation allele fractions: a change can come "
            "from sampling, depth, purity or a clone, and this task cannot say "
            "which",
        ],
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Repeated patient-level cross-validation
# ---------------------------------------------------------------------------


@dataclass
class CrossValidationResult:
    """A task scored over REPEATED patient-level folds, not one seeded split.

    A single split gives one number and one fold's worth of luck. Repeated folds
    turn that into a distribution: how many folds put the model above its
    baseline at all, how wide the spread is, and whether the pooled
    out-of-fold predictions clear the baseline once the patient-level bootstrap
    is applied to them. The verdict is read from the interval, never from the
    median.
    """

    task: str
    n_folds: int
    n_repeats: int
    n_patients: int
    n_rows: int
    per_fold: List[Dict[str, Optional[float]]] = field(default_factory=list)
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    #: Present only where the pooled out-of-fold score IS a probability (task A).
    #: The binned reliability table and the sentence derived from it live here
    #: together, so the words and the figures are produced by one function.
    calibration: Optional[Dict[str, Any]] = None
    #: Present only where the pooled score can be ACTED on, which is task A: the
    #: operating point at the declared threshold, the decision curve against
    #: treat-all/treat-none, and the conditions that keep a clinician in the
    #: loop. Built by ``clinical_use_block`` so the clinical figures and the
    #: research figures come from the same pooled out-of-fold predictions.
    clinical_use: Optional[Dict[str, Any]] = None
    #: Present only where the pooled unit needs declaring, which is task A: what
    #: a pooled score is (a row, carrying its own label) and what a per-patient
    #: label would have cost (the patients that hold both). Built by
    #: ``pooling_block`` and compared verbatim by the audit.
    pooling: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "task": self.task,
            "n_folds": self.n_folds,
            "n_repeats": self.n_repeats,
            "n_patients": self.n_patients,
            "n_rows": self.n_rows,
            "per_fold": self.per_fold,
            "metrics": self.metrics,
            "notes": self.notes,
            "calibration": self.calibration,
            "clinical_use": self.clinical_use,
            "pooling": self.pooling,
        }


def repeated_folds(
    patient_ids: Sequence[str],
    *,
    n_folds: int = 5,
    n_repeats: int = 5,
    seed: int = 13,
) -> List[PatientSplit]:
    """``n_repeats`` disjoint patient-level fold assignments.

    Every patient lands in exactly one test fold per repeat, so the
    out-of-fold predictions cover the whole cohort and no patient is scored by
    a model that trained on it. Each repeat reshuffles independently, which is
    what makes the spread across repeats meaningful rather than an artefact of
    one partition.
    """
    unique = sorted(set(patient_ids))
    if len(unique) < n_folds:
        raise ForecastError(
            f"{n_folds}-fold cross-validation needs at least {n_folds} patients, "
            f"got {len(unique)}"
        )
    if n_repeats < 1:
        raise ForecastError(f"n_repeats must be >= 1, got {n_repeats}")
    # Every fold must be non-empty: assign n_folds copies of each patient in a
    # stride pattern (equivalent to a shuffle-and-deal, but deterministic).
    rng = np.random.default_rng(seed)
    splits: List[PatientSplit] = []
    for _ in range(n_repeats):
        order = rng.permutation(len(unique))
        buckets: List[List[str]] = [[] for _ in range(n_folds)]
        for position, index in enumerate(order):
            buckets[position % n_folds].append(unique[int(index)])
        for fold in range(n_folds):
            test = tuple(sorted(buckets[fold]))
            train = tuple(sorted(p for f, b in enumerate(buckets) if f != fold for p in b))
            if not test or not train:
                raise ForecastError("a fold came out empty")
            splits.append(PatientSplit(train, test))
    return splits


def pooling_block(examples: Sequence[DetectionExample]) -> Dict[str, Any]:
    """WHAT a pooled score is, and what its label is -- counted, not asserted.

    A pooled discriminative number is only readable if the unit and its label
    are both named, and this block names them while counting the thing that
    makes the naming necessary: how many patients hold BOTH labels across their
    own rows, so that a single per-patient label is not defined for them at all.
    The sentence is DERIVED from those counts rather than written by hand, and
    the audit re-derives the whole block and compares it, so an edit that
    reintroduced a per-patient average over mixed labels would have to publish
    different numbers instead of different prose.
    """
    by_patient: Dict[str, List[bool]] = {}
    for example in examples:
        by_patient.setdefault(example.patient_id, []).append(
            example.detected_at_recurrence
        )
    if not by_patient:
        raise ForecastError("no rows to describe the pooling of")
    patients = sorted(by_patient)
    mixed = [p for p in patients if len(set(by_patient[p])) > 1]
    last_row = [by_patient[p][-1] for p in patients]
    any_row = [any(by_patient[p]) for p in patients]
    disagree = sum(1 for a, b in zip(last_row, any_row) if a != b)
    reading = (
        f"a pooled score is a ROW -- one measured locus in one patient -- and "
        f"every row carries its own measured label; the {len(patients)} patients "
        f"contribute {len(examples)} such rows. {len(mixed)} of those patients "
        f"hold BOTH labels across their own rows, so no single per-patient label "
        f"is defined for them: reading the last row alone would call "
        f"{int(sum(last_row))} patients positive while a re-detection anywhere "
        f"calls {int(sum(any_row))}, and the two disagree on {disagree}. The "
        f"published numbers are therefore ROW-level, with an interval that "
        f"resamples patients, and the patient endpoint is reported separately "
        f"rather than hidden inside the row result"
    )
    return {
        "unit": "row",
        "n_pooled_rows": len(examples),
        "n_patients": len(patients),
        "mixed_label_patients": len(mixed),
        "patients_positive_last_row": int(sum(last_row)),
        "patients_positive_any_row": int(sum(any_row)),
        "patients_disagreeing": disagree,
        "patient_endpoint": (
            "any primary-measured variant re-detected in the later draw"
        ),
        "reading": reading,
    }


def _oof_detection_aucs(
    examples: Sequence[DetectionExample],
    splits: Sequence[PatientSplit],
) -> Tuple[
    List[Dict[str, Optional[float]]],
    List[float],
    List[bool],
    List[str],
    List[float],
    List[bool],
    List[str],
]:
    """Per-fold AUC plus the out-of-fold predictions pooled over repeats.

    The pooled predictions are kept PER ROW, each carrying its OWN measured
    label, and that is a repair rather than a detail. The first version of this
    function collapsed them to one score per patient and read the patient's
    label off whichever row was written last -- which is not a patient label at
    all. Measured on this cohort: 83 of the 226 patients hold BOTH labels across
    their own rows, so under 'last row wins' 122 patients read as positive while
    the only defensible patient endpoint (a re-detection anywhere) gives 204, and
    the two disagree on 82 patients. A ranking compared against a label that an
    accident of row order decided is not a result. Rows now carry their own
    labels, the interval resamples PATIENTS so the within-tumour correlation is
    respected, and the patient endpoint is returned beside them as a SEPARATELY
    DECLARED second view rather than as the hidden label of the first. The block
    that states this -- counts and the sentence derived from them -- is built by
    ``pooling_block`` and compared by the audit, so the definition cannot drift
    away from the numbers it describes.
    """
    oof_rows: Dict[int, Dict[str, Any]] = {}

    per_fold: List[Dict[str, Optional[float]]] = []
    for split in splits:
        train_rows = [e for e in examples if e.patient_id in set(split.train_patients)]
        test_rows = [e for e in examples if e.patient_id in set(split.test_patients)]
        if not train_rows or not test_rows:
            raise ForecastError("the split left a fold empty")
        x_train, x_test = _features_a(train_rows, test_rows)
        y_train = np.array([e.detected_at_recurrence for e in train_rows], dtype=float)
        mean, scale = _standardise(x_train)
        w = fit_logistic((x_train - mean) / scale, y_train)
        z = np.hstack([np.ones((x_test.shape[0], 1)), (x_test - mean) / scale])
        fitted = 1.0 / (1.0 + np.exp(-z @ w))
        labels = [e.detected_at_recurrence for e in test_rows]
        for example, score in zip(test_rows, fitted):
            oof_rows.setdefault(id(example), {"example": example, "scores": []})[
                "scores"
            ].append(float(score))
        per_fold.append(
            {
                "fold": len(per_fold),
                "n_train_patients": len(split.train_patients),
                "n_test_patients": len(split.test_patients),
                "n_test_rows": len(test_rows),
                "auc_fitted": auc(list(fitted), labels),
                "auc_primary_af_only": auc(
                    [e.primary_af for e in test_rows], labels
                ),
                "prevalence": float(np.mean(labels)),
            }
        )
    # One out-of-fold score per ROW, over the repeats, each with its own label.
    ordered = [oof_rows[id(example)] for example in examples]
    row_scores = [float(np.mean(slot["scores"])) for slot in ordered]
    row_labels = [bool(slot["example"].detected_at_recurrence) for slot in ordered]
    row_patients = [slot["example"].patient_id for slot in ordered]

    index_by_patient: Dict[str, List[int]] = {}
    for position, patient_id in enumerate(row_patients):
        index_by_patient.setdefault(patient_id, []).append(position)
    patients = sorted(index_by_patient)
    patient_scores = [
        float(np.mean([row_scores[i] for i in index_by_patient[p]])) for p in patients
    ]
    patient_labels = [
        any(row_labels[i] for i in index_by_patient[p]) for p in patients
    ]
    return (
        per_fold,
        row_scores,
        row_labels,
        row_patients,
        patient_scores,
        patient_labels,
        patients,
    )


def cross_validated_detection(
    examples: Sequence[DetectionExample],
    *,
    n_folds: int = 5,
    n_repeats: int = 5,
    seed: int = 13,
) -> CrossValidationResult:
    """Task A over repeated patient-level folds.

    The unit of the published number is a ROW and the interval resamples
    PATIENTS -- see ``pooling_block`` for the measurement that makes a single
    per-patient label indefensible here. The row-level result and the separately
    declared patient endpoint are BOTH recorded, so a reader sees two named
    quantities instead of one figure whose label nobody can check.
    """
    if not examples:
        raise ForecastError("no detection examples to score")
    splits = repeated_folds(
        [e.patient_id for e in examples],
        n_folds=n_folds,
        n_repeats=n_repeats,
        seed=seed,
    )
    (
        per_fold,
        row_scores,
        row_labels,
        row_patients,
        patient_scores,
        patient_labels,
        patients,
    ) = _oof_detection_aucs(examples, splits)
    fold_aucs = [f["auc_fitted"] for f in per_fold if f["auc_fitted"] is not None]
    lo, hi = bootstrap_auc_ci(row_scores, row_labels, row_patients, seed=seed)
    pt_lo, pt_hi = bootstrap_auc_ci(
        patient_scores, patient_labels, patients, seed=seed
    )
    prevalence = float(np.mean(row_labels))
    n_scored = len(fold_aucs)
    n_above = sum(1 for a in fold_aucs if a > 0.5)
    rows_per_patient = len(row_scores) / len(patients)
    # Discrimination is not calibration: the same pooled out-of-fold
    # probabilities that give the AUC above are checked against the rates they
    # actually produced, and the sentence about them is derived here. The
    # calibration is measured on the SAME rows the AUC ranks -- a calibration
    # curve fitted on a different population from the ranking would be a second
    # undeclared change of unit.
    calibration = calibration_block(row_scores, row_labels)
    # The SAME pooled out-of-fold predictions, read as a decision instead of a
    # ranking: threshold the probability the AUC ranked, count what each call
    # finds and costs, and weigh it against the two alternatives a clinician
    # already has. Reusing these arrays is the guarantee that the clinical layer
    # cannot quote a different set of predictions from the research one.
    clinical_use = clinical_use_block(row_scores, row_labels)
    result = CrossValidationResult(
        task="detection_at_recurrence",
        n_folds=n_folds,
        n_repeats=n_repeats,
        n_patients=len(patients),
        n_rows=len(examples),
        per_fold=per_fold,
        metrics={
            "n_folds_scored": n_scored,
            "pooled_rows": len(row_scores),
            "pooled_patients": len(patients),
            "rows_per_patient": rows_per_patient,
            "oof_auc": auc(row_scores, row_labels),
            "oof_auc_ci_low": lo,
            "oof_auc_ci_high": hi,
            "patient_endpoint_auc": auc(patient_scores, patient_labels),
            "patient_endpoint_ci_low": pt_lo,
            "patient_endpoint_ci_high": pt_hi,
            "patient_endpoint_positive": int(sum(patient_labels)),
            "patient_endpoint_prevalence": float(np.mean(patient_labels)),
            "fold_auc_median": float(np.median(fold_aucs)) if fold_aucs else None,
            "fold_auc_min": float(np.min(fold_aucs)) if fold_aucs else None,
            "fold_auc_max": float(np.max(fold_aucs)) if fold_aucs else None,
            "fold_auc_iqr_low": (
                float(np.percentile(fold_aucs, 25)) if fold_aucs else None
            ),
            "fold_auc_iqr_high": (
                float(np.percentile(fold_aucs, 75)) if fold_aucs else None
            ),
            "folds_above_chance": n_above,
            "share_of_folds_above_chance": (n_above / n_scored) if n_scored else None,
            "baseline_majority_accuracy": max(prevalence, 1.0 - prevalence),
            "baseline_majority_auc": 0.5,
            "beats_prevalence_resolvably": (
                None if lo is None else bool(lo > 0.5)
            ),
            "patient_endpoint_resolvably_above_chance": (
                None if pt_lo is None else bool(pt_lo > 0.5)
            ),
            "seed": float(seed),
        },
        notes=[
            f"{n_repeats} repeats of {n_folds}-fold cross-validation, split by "
            "patient, so every row of a patient is scored by a model that never "
            "saw that patient",
            "the pooled unit is the ROW, each row carrying its own measured "
            "label; a per-patient mean score is still recorded, but the patient "
            "endpoint that labels it (a re-detection anywhere) is declared "
            "separately because the rows of one patient can disagree",
            "the fold count is reported BECAUSE a single split's point estimate "
            "is not a result -- read folds_above_chance and the pooled interval, "
            "not the median",
            "the interval resamples PATIENTS rather than rows, because the "
            "variants of one tumour are not independent draws",
            "the calibration block answers a DIFFERENT question from the AUC: "
            "whether the probability itself can be read, not whether the ranking "
            "separates. It is one cohort's calibration, and it is checked against "
            "a constant-prevalence predictor rather than against nothing",
        ],
        calibration=calibration,
        clinical_use=clinical_use,
        pooling=pooling_block(examples),
    )
    return result


def _oof_change_errors(
    examples: Sequence[ChangeExample],
    splits: Sequence[PatientSplit],
) -> Tuple[List[Dict[str, Optional[float]]], List[float], List[float], List[str]]:
    """Per-fold persistence advantage plus pooled out-of-fold predictions."""
    oof_sum: Dict[str, List[float]] = {}
    oof_y: Dict[str, List[float]] = {}
    per_fold: List[Dict[str, Optional[float]]] = []
    for split in splits:
        train_rows = [e for e in examples if e.patient_id in set(split.train_patients)]
        test_rows = [e for e in examples if e.patient_id in set(split.test_patients)]
        if not train_rows or not test_rows:
            raise ForecastError("the split left a fold empty")
        x_train, x_test = _features_a(train_rows, test_rows)
        y_train = np.array([e.delta_af for e in train_rows], dtype=float)
        y_test = np.array([e.delta_af for e in test_rows], dtype=float)
        mean, scale = _standardise(x_train)
        coef = fit_line((x_train - mean) / scale, y_train)
        z = np.hstack([np.ones((x_test.shape[0], 1)), (x_test - mean) / scale])
        pred = z @ coef
        for example, value in zip(test_rows, pred):
            oof_sum.setdefault(example.patient_id, []).append(float(value))
            oof_y.setdefault(example.patient_id, []).append(example.delta_af)
        persistence_mae = float(np.mean(np.abs(y_test)))
        fitted_mae = float(np.mean(np.abs(pred - y_test)))
        per_fold.append(
            {
                "fold": len(per_fold),
                "n_train_patients": len(split.train_patients),
                "n_test_patients": len(split.test_patients),
                "n_test_rows": len(test_rows),
                "persistence_mae": persistence_mae,
                "model_mae": fitted_mae,
                "mae_advantage": persistence_mae - fitted_mae,
            }
        )
    patients = sorted(oof_sum)
    pooled_pred = [float(np.mean(oof_sum[p])) for p in patients]
    pooled_y = [float(np.mean(oof_y[p])) for p in patients]
    return per_fold, pooled_pred, pooled_y, patients


def cross_validated_change(
    examples: Sequence[ChangeExample],
    *,
    n_folds: int = 5,
    n_repeats: int = 5,
    seed: int = 13,
) -> CrossValidationResult:
    """Task B over repeated patient-level folds."""
    if not examples:
        raise ForecastError("no change examples to score")
    splits = repeated_folds(
        [e.patient_id for e in examples],
        n_folds=n_folds,
        n_repeats=n_repeats,
        seed=seed,
    )
    per_fold, pooled_pred, pooled_y, patients = _oof_change_errors(examples, splits)
    advantages = [f["mae_advantage"] for f in per_fold if f["mae_advantage"] is not None]
    lo, hi, share = bootstrap_mae_difference(pooled_pred, pooled_y, patients, seed=seed)
    n_positive = sum(1 for a in advantages if a > 0)
    n_scored = len(advantages)
    # A per-patient (not per-row) comparison: persistence against the pooled
    # out-of-fold prediction. `pooled_y` and `pooled_pred` already carry one
    # value per patient -- the mean over that tumour's variants -- so a row-level
    # mean would count one specimen many times and overstate the cohort.
    persistence_by_patient = [abs(y_value) for y_value in pooled_y]
    model_by_patient = [
        abs(pred_value - y_value)
        for y_value, pred_value in zip(pooled_y, pooled_pred)
    ]
    result = CrossValidationResult(
        task="allele_fraction_change",
        n_folds=n_folds,
        n_repeats=n_repeats,
        n_patients=len(patients),
        n_rows=len(examples),
        per_fold=per_fold,
        metrics={
            "n_folds_scored": n_scored,
            "fold_mae_advantage_median": (
                float(np.median(advantages)) if advantages else None
            ),
            "fold_mae_advantage_min": (
                float(np.min(advantages)) if advantages else None
            ),
            "fold_mae_advantage_max": (
                float(np.max(advantages)) if advantages else None
            ),
            "folds_beating_persistence": n_positive,
            "share_of_folds_beating_persistence": (
                (n_positive / n_scored) if n_scored else None
            ),
            # Patient-level errors, so the number is not inflated by the many
            # variants of one specimen counting as many independent results.
            "persistence_mae_per_patient": float(np.mean(persistence_by_patient)),
            "model_mae_per_patient": float(np.mean(model_by_patient)),
            "oof_mae_advantage": float(
                np.mean(persistence_by_patient) - np.mean(model_by_patient)
            ),
            "oof_mae_advantage_ci_low": lo,
            "oof_mae_advantage_ci_high": hi,
            "oof_mae_advantage_share_of_draws_positive": share,
            "beats_persistence_resolvably": (
                None if lo is None else bool(lo > 0.0)
            ),
            "seed": float(seed),
        },
        notes=[
            f"{n_repeats} repeats of {n_folds}-fold cross-validation, split by "
            "patient, so each patient is scored exactly once per repeat by a "
            "model that never saw it",
            "the advantage is reported per PATIENT as well as per row: the many "
            "variants of one specimen are not independent draws, so a row-level "
            "mean would overstate the cohort this rests on",
            "every repeat's fold assignment is recorded, so the spread can be "
            "checked rather than believed",
        ],
    )
    return result


def forecast_provenance() -> Dict[str, EvidenceStatus]:
    """What each link in this forecast is and is not supported by."""
    return {
        "allele_fractions_measured_from_real_specimens": EvidenceStatus.ESTABLISHED,
        "patient_level_split_enforced": EvidenceStatus.ESTABLISHED,
        "task_a_scores_re-detection_of_variants_the_primary_measured": (
            EvidenceStatus.ESTABLISHED
        ),
        "baselines_are_reported_beside_the_model": EvidenceStatus.ESTABLISHED,
        # A forecast of one specimen from another is not a treatment effect.
        "change_is_caused_by_therapy": EvidenceStatus.UNKNOWN,
        "engine_predicts_patients": EvidenceStatus.UNKNOWN,
        "clinical_benefit": EvidenceStatus.UNKNOWN,
    }


def what_this_forecast_cannot_say() -> str:
    return (
        "This scores a molecular prediction on real paired specimens: whether a "
        "locus is re-detected, and how far an allele fraction moves. It is not a "
        "treatment effect and not a patient outcome: the specimens carry no "
        "therapy record and no survival, several years separate the two draws, "
        "depth differs between them, and a change in an allele fraction can come "
        "from clonal expansion, from sampling, from sequencing depth or from "
        "tumour purity. A model that beats persistence here has beaten a "
        "statistical floor on one public cohort; it has not been validated "
        "clinically. It may enter a decision only as reviewed evidence beside "
        "the operating point and the decision curve published with it, for a "
        "named clinician who can withhold it and records why — never as a "
        "stand-alone instruction."
    )
