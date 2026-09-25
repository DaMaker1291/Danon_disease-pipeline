"""
Covariate intake: which of the proposed upgrades are real, and what they cost.

WHY THIS MODULE EXISTS
----------------------
Four covariates are routinely proposed to lift a glioblastoma C-index --
MGMT promoter methylation, 3D sub-region segmentation, Karnofsky performance
status, and restriction to a standardised treatment cohort. Each is implemented
or refused here, and each carries a **measured** effect on the real cohort
rather than an estimated one. An estimated gain is the easiest number in a
project like this to invent, so no estimate appears in this module.

THE MEASUREMENT THAT DOMINATES ALL FOUR
---------------------------------------
Before any covariate is added, the endpoint has to be right. The cohort loader
in `danon.predictive` read the censoring time from `follow_ups[].days_to_last_
follow_up`, a field that does not exist; the day is `follow_ups[].days_to_follow_up`
and `diagnoses[].days_to_last_follow_up`. The consequence was that the censoring
branch could never fire: every patient with a usable endpoint was coded as an
event, and the 102 patients recorded *Alive* were dropped.

    endpoint     before (no censoring)      after (corrected)
    OS           n=491  0 censored  C=0.643   n=595  104 censored  C=0.656
    progression  n=267  0 censored  C=0.590   n=594  222 censored  C=0.534

Discarding every censored patient deletes exactly the patients who survived
longest. On progression that inflated the C-index by +0.056 and hid the fact
that the multivariable model does not beat age alone. Fixing the endpoint is
worth more than all four covariates combined.

KPS IS PRESENT BUT TEMPORALLY AMBIGUOUS
---------------------------------------
`karnofsky_performance_status` is in the cached export -- 476 records over the
617 cases. Two properties decide whether it can be used:

* Every KPS record sits in a timepoint that has **no** day stamp
  (`Preoperative`, `Prior to Adjuvant Therapy`, `Post Adjuvant Therapy`,
  `Not Reported`, `Other`), while every timepoint that carries a day stamp
  (`Follow-up`, `Post Initial Treatment`, `Last Contact`) has **no** KPS.
  Timepoint category is therefore the only temporal anchor, and the gate below
  is what makes the covariate usable at all.
* Values are strings and include `'Unknown'`, which must parse to missing. A
  `'Unknown'` read as 0 is a KPS of 0 -- i.e. dead.

Univariate KPS as a lone predictor, measured on the corrected OS cohort:

    pre-treatment  (Preoperative / Prior to Adjuvant Therapy)  n=247  C=0.597
    post-adjuvant  (recorded AFTER treatment)                  n=101  C=0.684

A post-treatment KPS "predicts" survival better than a pre-treatment one. That
is not a better predictor; it is a measurement partly determined by the outcome
(a patient who deteriorated has a low post-treatment KPS). `kps_leakage_contrast`
measures this rather than asserting it, and `baseline_kps` refuses post-baseline
timepoints outright.

MGMT IS REACHABLE; SUB-REGION SEGMENTATION IS NOT
-------------------------------------------------
450 open GDC `Methylation Beta Value` files exist for TCGA-GBM covering 200
patients (155 on the 450k platform). `extract_mgmt_beta` reads one and returns
the canonical promoter-probe betas; the format is `probe_id<TAB>beta` over
486,427 probes with `NA` for missing. Segmented sub-regions remain refused --
`danon.multimodal.spatial_dice_verdict` derives that from live collection
counts, and nothing here changes it.

References:
  Bady P et al. Acta Neuropathol 2012;124(4):547-560 (MGMT-STP27; probe pair
  cg12434587/cg12981137 and the average-beta threshold).
  Mansouri A et al. Neuro Oncol 2019;21(12):1519-1528 (KPS prognostic value).
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.predictive import (
    GDC_CLINICAL_PATH,
    PatientRow,
    cox_fit,
    harrell_c_index,
    load_cohort_rows,
)

# ---------------------------------------------------------------------------
# KPS: the temporal gate
# ---------------------------------------------------------------------------

#: KPS timepoints that precede the treatment decision. Only these may be used
#: as a baseline covariate. `Post Adjuvant Therapy` is deliberately excluded:
#: it is measured after treatment and partially encodes the outcome.
KPS_BASELINE_TIMEPOINTS = frozenset({"Preoperative", "Prior to Adjuvant Therapy"})

#: Timepoints recorded after treatment started, kept for the leakage contrast.
KPS_POST_BASELINE_TIMEPOINTS = frozenset({"Post Adjuvant Therapy"})

#: Strings that mean "no measurement". Reading any of these as 0 would record a
#: KPS of 0, which is a dead patient.
KPS_UNUSABLE = frozenset({"unknown", "not reported", "not available", "na", "null", ""})

KPS_RANGE = (0.0, 100.0)


def parse_kps(value: object) -> Optional[float]:
    """A KPS string to a number, or None. `'Unknown'` is missing, not zero."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in KPS_UNUSABLE:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    lo, hi = KPS_RANGE
    if not (lo <= number <= hi):
        return None
    return number


@dataclass(frozen=True)
class TimepointCensus:
    """Where KPS records live, and whether those timepoints carry a day stamp."""

    timepoint: str
    n_records: int
    n_kps_records: int
    n_numeric_kps: int
    median_day: Optional[float]
    max_day: Optional[float]
    #: True when this timepoint has any `days_to_follow_up`. The timepoints that
    #: hold KPS records are exactly the ones that do not, which is why the
    #: timepoint category and not a day stamp is the only temporal anchor.
    carries_day_stamp: bool

    @property
    def is_baseline(self) -> bool:
        return self.timepoint in KPS_BASELINE_TIMEPOINTS


def kps_timepoint_census(cases: Sequence[dict]) -> List[TimepointCensus]:
    """Per-timepoint KPS counts and day-stamp availability.

    Returned sorted by descending record count. This is the table that justifies
    the baseline gate, and it is computed rather than asserted.
    """
    tally: Dict[str, Dict[str, List[float]]] = {}
    for hit in cases:
        for fu in hit.get("follow_ups") or []:
            tp = fu.get("timepoint_category") or "None"
            row = tally.setdefault(tp, {"n": 0, "kps": 0, "numeric": 0, "days": []})
            row["n"] += 1
            if fu.get("karnofsky_performance_status") is not None:
                row["kps"] += 1
            if parse_kps(fu.get("karnofsky_performance_status")) is not None:
                row["numeric"] += 1
            day = fu.get("days_to_follow_up")
            if day is not None:
                row["days"].append(float(day))

    out: List[TimepointCensus] = []
    for tp, row in tally.items():
        days = row["days"]
        out.append(
            TimepointCensus(
                timepoint=tp,
                n_records=row["n"],
                n_kps_records=row["kps"],
                n_numeric_kps=row["numeric"],
                median_day=float(np.median(days)) if days else None,
                max_day=max(days) if days else None,
                carries_day_stamp=bool(days),
            )
        )
    out.sort(key=lambda c: -c.n_records)
    return out


def _pick_kps(hit: dict, timepoints: Iterable[str], *, take_max: bool = True) -> Optional[float]:
    best: Optional[float] = None
    for fu in hit.get("follow_ups") or []:
        if (fu.get("timepoint_category") or "") not in timepoints:
            continue
        value = parse_kps(fu.get("karnofsky_performance_status"))
        if value is None:
            continue
        if best is None:
            best = value
        elif take_max:
            best = max(best, value)
        else:
            best = min(best, value)
    return best


def baseline_kps(cases: Sequence[dict]) -> Dict[str, float]:
    """Pre-treatment KPS per case. Post-baseline timepoints are refused."""
    out: Dict[str, float] = {}
    for hit in cases:
        value = _pick_kps(hit, KPS_BASELINE_TIMEPOINTS)
        if value is not None:
            out[str(hit.get("submitter_id"))] = value
    return out


def post_baseline_kps(cases: Sequence[dict]) -> Dict[str, float]:
    """Post-treatment KPS per case. Only for the leakage contrast -- never a feature."""
    out: Dict[str, float] = {}
    for hit in cases:
        value = _pick_kps(hit, KPS_POST_BASELINE_TIMEPOINTS)
        if value is not None:
            out[str(hit.get("submitter_id"))] = value
    return out


@dataclass(frozen=True)
class KpsCoverage:
    n_cohort: int
    n_baseline: int
    n_post_baseline: int
    n_any_numeric: int
    median_baseline: Optional[float]
    min_baseline: Optional[float]
    max_baseline: Optional[float]

    @property
    def baseline_fraction(self) -> float:
        return self.n_baseline / self.n_cohort if self.n_cohort else float("nan")

    def __str__(self) -> str:
        return (
            f"KPS: {self.n_baseline}/{self.n_cohort} patients have a pre-treatment "
            f"value ({100 * self.baseline_fraction:.1f}%), {self.n_post_baseline} have "
            f"a post-treatment value, {self.n_any_numeric} have a numeric value "
            f"anywhere; baseline range {self.min_baseline:.0f}-{self.max_baseline:.0f}, "
            f"median {self.median_baseline:.0f}"
        )


def kps_coverage(cases: Sequence[dict]) -> KpsCoverage:
    """How many patients actually carry each flavour of KPS."""
    baseline = baseline_kps(cases)
    post = post_baseline_kps(cases)
    numeric: Dict[str, float] = {}
    for hit in cases:
        value = _pick_kps(hit, KPS_BASELINE_TIMEPOINTS | KPS_POST_BASELINE_TIMEPOINTS | {"Not Reported", "Other"}, take_max=True)
        if value is not None:
            numeric[str(hit.get("submitter_id"))] = value
    values = sorted(baseline.values())
    return KpsCoverage(
        n_cohort=len(cases),
        n_baseline=len(baseline),
        n_post_baseline=len(post),
        n_any_numeric=len(numeric),
        median_baseline=float(np.median(values)) if values else None,
        min_baseline=min(values) if values else None,
        max_baseline=max(values) if values else None,
    )


# ---------------------------------------------------------------------------
# Cross-validated effect of one covariate, measured on the corrected endpoint
# ---------------------------------------------------------------------------

#: The baseline design. Every covariate is judged against this, on the same
#: patient subset, so a selection effect cannot masquerade as a covariate gain.
BASE_COVARIATES: Tuple[str, ...] = ("age_years", "male", "is_glioblastoma")


@dataclass(frozen=True)
class _FoldScores:
    """Held-out scores for one fitted design."""

    mean_c: float
    sd_c: float
    age_only_c: float
    n_folds_used: int
    hazard_ratios: Optional[Dict[str, float]]


@dataclass(frozen=True)
class CovariateEffect:
    """A covariate's measured effect, with the cost of requiring it."""

    name: str
    n_patients: int
    n_events: int
    n_censored: int
    base_c: float
    base_sd: float
    augmented_c: float
    augmented_sd: float
    age_only_c: float
    hazard_ratio: Optional[float]
    n_dropped_by_availability: int
    n_folds_used: int = 0
    n_folds_requested: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.augmented_c - self.base_c

    @property
    def folds_complete(self) -> bool:
        return self.n_folds_requested > 0 and self.n_folds_used == self.n_folds_requested

    @property
    def verdict(self) -> str:
        if not (np.isfinite(self.base_c) and np.isfinite(self.augmented_c)):
            # A covariate that separates the outcome almost perfectly makes the
            # partial likelihood unbounded, so no fold converges and the honest
            # answer is that nothing was estimable -- not a large gain.
            if self.n_patients and not self.folds_complete:
                return "not estimable: the augmented fit did not converge"
            return "not estimable"
        if not self.folds_complete:
            return f"not estimable: only {self.n_folds_used} of {self.n_folds_requested} folds converged"
        # Fold-to-fold spread is the yardstick: a delta inside one sd of the
        # base fit is not distinguishable from fold noise on this cohort.
        if abs(self.delta) < max(self.base_sd, 1e-9):
            return "no measurable gain"
        return "gain exceeds fold noise" if self.delta > 0 else "loss exceeds fold noise"

    def __str__(self) -> str:
        hr = f"{self.hazard_ratio:.4f}" if self.hazard_ratio else "n/a"
        return (
            f"{self.name}: n={self.n_patients} ({self.n_events} events, "
            f"{self.n_censored} censored)  base C={self.base_c:.4f}+/-{self.base_sd:.4f} "
            f"-> +covariate C={self.augmented_c:.4f}+/-{self.augmented_sd:.4f}  "
            f"delta={self.delta:+.4f}  HR={hr}  "
            f"folds={self.n_folds_used}/{self.n_folds_requested}  [{self.verdict}]"
        )


def _cv(
    rows: Sequence[PatientRow],
    values: Optional[Sequence[float]] = None,
    *,
    n_folds: int = 5,
    seed: int = 0,
    covariate_name: Optional[str] = None,
) -> _FoldScores:
    """Held-out scores for the base design, optionally plus one covariate.

    Fold assignment is a seeded permutation of the same rows either way, so the
    base and augmented designs see identical splits and the difference between
    them is the covariate rather than a different partition.
    """
    nan = _FoldScores(float("nan"), float("nan"), float("nan"), 0, None)
    if not rows:
        return nan
    columns = [[r.age_years, r.male, r.is_glioblastoma] for r in rows]
    names: Tuple[str, ...] = BASE_COVARIATES
    if values is not None:
        columns = [c + [float(v)] for c, v in zip(columns, values)]
        names = BASE_COVARIATES + (covariate_name or "covariate",)
    X = np.asarray(columns, dtype=float)
    t = np.array([r.time_days for r in rows], dtype=float)
    e = np.array([r.event for r in rows], dtype=int)

    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(len(rows)), n_folds)
    scores: List[float] = []
    ages: List[float] = []
    last_hazard: Optional[Dict[str, float]] = None
    for fold in folds:
        test = np.zeros(len(rows), dtype=bool)
        test[fold] = True
        train = ~test
        if e[train].sum() < 3 or len(fold) < 3:
            continue
        fit = cox_fit(X[train], t[train], e[train], names)
        if not fit.converged:
            continue
        c = harrell_c_index(fit.risk(X[test]), t[test], e[test])
        if not c.estimable:
            continue
        age_c = harrell_c_index(X[test, 0], t[test], e[test])
        scores.append(c.c_index)
        if age_c.estimable:
            ages.append(age_c.c_index)
        last_hazard = fit.hazard_ratios
    if not scores:
        return nan
    sd = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
    return _FoldScores(
        mean_c=float(np.mean(scores)),
        sd_c=sd,
        age_only_c=float(np.mean(ages)) if ages else float("nan"),
        n_folds_used=len(scores),
        hazard_ratios=last_hazard,
    )


def covariate_effect(
    rows: Sequence[PatientRow],
    values: Dict[str, float],
    *,
    name: str,
    n_folds: int = 5,
    seed: int = 0,
) -> CovariateEffect:
    """Measure a covariate's held-out effect on the subset that carries it.

    The comparison is paired within the subset: base design and augmented design
    are fitted on the *same* patients. The subset itself is a selection, so the
    subset's base C is reported alongside -- if the base C moves when the subset
    shrinks, that move is selection, not signal.
    """
    subset = [r for r in rows if r.submitter_id in values]
    dropped = len(rows) - len(subset)
    if not subset:
        return CovariateEffect(
            name=name, n_patients=0, n_events=0, n_censored=0,
            base_c=float("nan"), base_sd=float("nan"),
            augmented_c=float("nan"), augmented_sd=float("nan"),
            age_only_c=float("nan"), hazard_ratio=None,
            n_dropped_by_availability=dropped,
            n_folds_used=0, n_folds_requested=n_folds,
            notes=["no patient in the cohort carries this covariate; refused"],
        )
    base = _cv(subset, None, n_folds=n_folds, seed=seed)
    aug = _cv(
        subset,
        [values[r.submitter_id] for r in subset],
        n_folds=n_folds,
        seed=seed,
        covariate_name=name,
    )
    events = sum(r.event for r in subset)
    notes = [f"{dropped} of {len(rows)} cohort patients dropped: covariate absent"]
    if dropped:
        notes.append(
            "requiring this covariate changes the cohort, so base C on the "
            "subset is not comparable to base C on the full cohort"
        )
    if len(subset) and aug.n_folds_used < n_folds:
        notes.append(
            f"the augmented design converged in only {aug.n_folds_used} of {n_folds} "
            f"folds; a covariate that nearly separates the outcome leaves the "
            f"partial likelihood unbounded, so a nan here means 'not estimable', "
            f"not 'very large'"
        )
    return CovariateEffect(
        name=name,
        n_patients=len(subset),
        n_events=events,
        n_censored=len(subset) - events,
        base_c=base.mean_c,
        base_sd=base.sd_c,
        augmented_c=aug.mean_c,
        augmented_sd=aug.sd_c,
        age_only_c=base.age_only_c,
        hazard_ratio=(aug.hazard_ratios or {}).get(name),
        n_dropped_by_availability=dropped,
        n_folds_used=aug.n_folds_used,
        n_folds_requested=n_folds,
        notes=notes,
    )


@dataclass(frozen=True)
class LeakageContrast:
    """The same statistic computed from a pre- and a post-treatment measurement."""

    baseline_c: float
    baseline_n: int
    post_baseline_c: float
    post_baseline_n: int

    @property
    def verdict(self) -> str:
        if not (np.isfinite(self.baseline_c) and np.isfinite(self.post_baseline_c)):
            return "not estimable"
        if self.post_baseline_c <= self.baseline_c:
            return "no leakage detected in this cohort"
        return (
            f"LEAKAGE: a post-treatment KPS outranks a pre-treatment one by "
            f"{self.post_baseline_c - self.baseline_c:+.4f} C. The post-treatment "
            f"value is partly determined by the outcome, so it must not be used "
            f"as a baseline feature"
        )

    def __str__(self) -> str:
        return (
            f"univariate KPS: pre-treatment C={self.baseline_c:.4f} (n={self.baseline_n}) "
            f"vs post-adjuvant C={self.post_baseline_c:.4f} (n={self.post_baseline_n}) "
            f"-- {self.verdict}"
        )


def kps_leakage_contrast(
    rows: Sequence[PatientRow], cases: Sequence[dict]
) -> LeakageContrast:
    """Measure how much a post-treatment KPS overstates discrimination.

    KPS is a risk *marker*: higher is better, so it enters the C-index negated.
    """
    index = {r.submitter_id: r for r in rows}

    def univariate(values: Dict[str, float]) -> Tuple[float, int]:
        ids = [sid for sid in values if sid in index]
        if len(ids) < 10:
            return float("nan"), len(ids)
        risk = -np.array([values[sid] for sid in ids], dtype=float)
        t = np.array([index[sid].time_days for sid in ids], dtype=float)
        e = np.array([index[sid].event for sid in ids], dtype=int)
        c = harrell_c_index(risk, t, e)
        return (c.c_index if c.estimable else float("nan")), len(ids)

    base_c, base_n = univariate(baseline_kps(cases))
    post_c, post_n = univariate(post_baseline_kps(cases))
    return LeakageContrast(
        baseline_c=base_c, baseline_n=base_n, post_baseline_c=post_c, post_baseline_n=post_n
    )


# ---------------------------------------------------------------------------
# MGMT promoter methylation: real files, real probes
# ---------------------------------------------------------------------------

#: Canonical MGMT promoter probe pair (Bady 2012, MGMT-STP27). The probe
#: selection is literature-derived, not derived from the file, so it is marked
#: ASSUMED: a different probe set would give a different beta.
MGMT_PROBES: Dict[str, str] = {
    "cg12434587": "MGMT promoter, TSS200",
    "cg12981137": "MGMT promoter, TSS1500",
}

#: Average-beta threshold for calling a tumour methylated. The value is
#: model-specific and publications differ; it is a parameter, not a fact.
MGMT_METHYLATION_CUTOFF = 0.358

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
MGMT_REACHABILITY_PATH = "data/gdc_methylation/reachability.json"

#: The cached clinical export carries no treatment node at all -- fetching cases
#: with `expand=diagnoses.treatments` and projecting only that subtree is what
#: makes treatment exposure measurable offline.
TREATMENT_RECORDS_PATH = "data/gdc_clinical/tcga_gbm_treatments.json"
TCGA_GBM_PROJECT = "TCGA-GBM"


@dataclass(frozen=True)
class MgmtBetaResult:
    """A real MGMT beta call read from a GDC methylation file."""

    source_path: str
    submitter_id: Optional[str]
    probe_values: Dict[str, float]
    missing_probes: List[str]
    mean_beta: Optional[float]
    cutoff: float
    call: str
    n_probes_in_file: int

    def __str__(self) -> str:
        vals = ", ".join(f"{p}={v:.4f}" for p, v in sorted(self.probe_values.items()))
        mean = f"{self.mean_beta:.4f}" if self.mean_beta is not None else "n/a"
        return (
            f"MGMT {self.submitter_id or self.source_path}: {vals} "
            f"mean={mean} cutoff={self.cutoff:.3f} -> {self.call}"
        )


def extract_mgmt_beta(
    path: str,
    *,
    probes: Optional[Dict[str, str]] = None,
    cutoff: float = MGMT_METHYLATION_CUTOFF,
    submitter_id: Optional[str] = None,
) -> MgmtBetaResult:
    """Read a GDC `Methylation Beta Value` file and call MGMT.

    The format is two columns, `probe_id<TAB>beta`, over ~486k probes with `NA`
    for missing values and no header. A probe that is absent or `NA` is reported
    as missing; it is never imputed, because imputing a promoter beta would
    fabricate the very covariate under test.
    """
    probe_set = dict(probes or MGMT_PROBES)
    values: Dict[str, float] = {}
    n_lines = 0
    with open(path) as fh:
        for line in fh:
            n_lines += 1
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            probe = parts[0]
            if probe not in probe_set:
                continue
            raw = parts[1].strip()
            if raw.upper() in ("NA", "NAN", ""):
                continue
            try:
                values[probe] = float(raw)
            except ValueError:
                continue
    missing = [p for p in probe_set if p not in values]
    usable = [v for v in values.values() if 0.0 <= v <= 1.0]
    mean_beta = float(np.mean(usable)) if usable else None
    if mean_beta is None:
        call = "not callable: no usable probe"
    else:
        call = "methylated" if mean_beta >= cutoff else "unmethylated"
    return MgmtBetaResult(
        source_path=path,
        submitter_id=submitter_id,
        probe_values=values,
        missing_probes=missing,
        mean_beta=mean_beta,
        cutoff=cutoff,
        call=call,
        n_probes_in_file=n_lines,
    )


@dataclass(frozen=True)
class MgmtReachability:
    """How many TCGA-GBM patients have a methylation file, by platform."""

    n_beta_files: int
    n_patients_total: int
    n_patients_450k: int
    n_patients_27k: int
    all_open_access: bool
    source: str

    @property
    def verdict(self) -> str:
        if self.n_patients_450k == 0:
            return "MGMT not reachable: no 450k beta files"
        return (
            f"MGMT reachable for {self.n_patients_450k} patients on the 450k "
            f"platform ({self.n_patients_total} across both platforms). Adding MGMT "
            f"therefore caps the fitting cohort at {self.n_patients_450k} patients"
        )

    def __str__(self) -> str:
        return (
            f"GDC methylation: {self.n_beta_files} beta files, "
            f"{self.n_patients_total} patients, 450k={self.n_patients_450k} "
            f"27k={self.n_patients_27k}, open={self.all_open_access} -- {self.verdict}"
        )


def _default_fetcher(url: str, timeout: int = 60) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return json.loads(fh.read().decode("utf-8"))


def probe_mgmt_reachability(
    *,
    project: str = "TCGA-GBM",
    fetcher: Optional[Callable[[str], dict]] = None,
    cache_path: str = MGMT_REACHABILITY_PATH,
    use_cache: bool = True,
) -> MgmtReachability:
    """Count real MGMT beta-value files, 450k and 27k, from the live GDC API."""
    if use_cache:
        try:
            with open(cache_path) as fh:
                blob = json.load(fh)
            return MgmtReachability(**blob)
        except (OSError, ValueError, TypeError):
            pass

    fetch = fetcher or _default_fetcher

    def count(platform: Optional[str]) -> Tuple[int, int, bool]:
        content = [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Methylation Beta Value"]}},
        ]
        if platform:
            content.append(
                {"op": "in", "content": {"field": "platform", "value": [platform]}}
            )
        query = urllib.parse.urlencode(
            {
                "filters": json.dumps({"op": "and", "content": content}),
                "size": "0",
                "facets": "cases.submitter_id,access",
            }
        )
        payload = fetch(f"{GDC_FILES_ENDPOINT}?{query}")
        data = payload["data"]
        aggs = data["aggregations"]
        n_files = int(data["pagination"]["total"])
        n_patients = len(aggs["cases.submitter_id"]["buckets"])
        access = aggs.get("access", {}).get("buckets", [])
        all_open = all(b["key"] == "open" for b in access) if access else False
        return n_files, n_patients, all_open

    n_beta, _n_any, _ = count(None)
    n_450_files, n_450, open_450 = count("Illumina Human Methylation 450")
    _, n_27, _ = count("Illumina Human Methylation 27")
    result = MgmtReachability(
        n_beta_files=n_beta,
        n_patients_total=_n_any,
        n_patients_450k=n_450,
        n_patients_27k=n_27,
        all_open_access=open_450,
        source=GDC_FILES_ENDPOINT,
    )
    try:
        with open(cache_path, "w") as fh:
            json.dump(result.__dict__, fh, indent=2, sort_keys=True)
    except OSError:
        pass
    return result


# ---------------------------------------------------------------------------
# Treatment exposure: reachable, but exposure is not protocol
# ---------------------------------------------------------------------------

#: Agents whose presence, with radiotherapy, identifies a Stupp-like regimen.
STUPP_ALKYLATOR = "temozolomide"
RADIOTHERAPY_MARKERS: Tuple[str, ...] = ("radiation", "radiotherapy")


@dataclass(frozen=True)
class TreatmentExposure:
    """What the real treatment records can and cannot support."""

    n_cases: int
    n_records: int
    n_cases_with_agent: int
    n_cases_rt_plus_tmz: int
    n_records_with_start_day: int
    median_records_per_case: float
    top_agents: List[Tuple[str, int]]

    @property
    def can_restrict_to_protocol(self) -> bool:
        """False: not enough treatment records carry an ordered start day."""
        if not self.n_records:
            return False
        return self.stamped_fraction >= 0.9

    @property
    def stamped_fraction(self) -> float:
        return self.n_records_with_start_day / self.n_records if self.n_records else float("nan")

    @property
    def verdict(self) -> str:
        if not self.n_records:
            return (
                "no treatment records loaded; the cached clinical export carries "
                "none, so this measurement needs the expanded cases fetch"
            )
        return (
            f"{self.n_cases_rt_plus_tmz}/{self.n_cases} cases record radiotherapy plus "
            f"temozolomide somewhere in their history, but only "
            f"{100 * self.stamped_fraction:.1f}% of treatment "
            f"records carry a start day, so 'received temozolomide' cannot be "
            f"separated from first-line therapy. This is a treatment-EXPOSURE "
            f"restriction, not a protocol restriction"
        )

    def __str__(self) -> str:
        return (
            f"treatments: {self.n_records} records over {self.n_cases} cases, "
            f"{self.n_cases_with_agent} with a named agent, "
            f"{self.n_cases_rt_plus_tmz} with RT+temozolomide, "
            f"{self.n_records_with_start_day} records day-stamped -- {self.verdict}"
        )


def probe_treatment_records(
    *,
    project: str = TCGA_GBM_PROJECT,
    fetcher: Optional[Callable[[str], dict]] = None,
    cache_path: str = TREATMENT_RECORDS_PATH,
    use_cache: bool = True,
) -> List[dict]:
    """Fetch every case with its treatment subtree, caching a slim projection.

    The projection keeps `{submitter_id, diagnoses:[{treatments:[...]}]}` so the
    cache is reviewable and stays a few megabytes rather than tens.
    """
    if use_cache:
        try:
            with open(cache_path) as fh:
                return list(json.load(fh).get("hits", []))
        except (OSError, ValueError, TypeError):
            pass

    fetch = fetcher or (lambda url: _default_fetcher(url, timeout=120))
    hits: List[dict] = []
    offset = 0
    page = 500
    while True:
        query = urllib.parse.urlencode(
            {
                "filters": json.dumps(
                    {"op": "in", "content": {"field": "project.project_id", "value": [project]}}
                ),
                "size": str(page),
                "from": str(offset),
                "expand": "diagnoses.treatments",
            }
        )
        payload = fetch(f"{GDC_CASES_ENDPOINT}?{query}")
        batch = payload["data"]["hits"]
        hits.extend(
            {
                "submitter_id": c.get("submitter_id"),
                "diagnoses": [
                    {"treatments": dx.get("treatments") or []}
                    for dx in (c.get("diagnoses") or [])
                ],
            }
            for c in batch
        )
        offset += len(batch)
        total = int(payload["data"]["pagination"]["total"])
        if not batch or offset >= total:
            break

    try:
        with open(cache_path, "w") as fh:
            json.dump(
                {
                    "source": f"{GDC_CASES_ENDPOINT}?expand=diagnoses.treatments&project={project}",
                    "n_cases": len(hits),
                    "hits": hits,
                },
                fh,
            )
    except OSError:
        pass
    return hits


def treatment_exposure(hits: Sequence[dict]) -> TreatmentExposure:
    """Summarise the treatment records actually present in an export."""
    from collections import Counter

    agents: Counter = Counter()
    n_records = 0
    n_stamped = 0
    with_agent = 0
    rt_tmz = 0
    counts: List[int] = []

    for hit in hits:
        records = [
            tr for dx in (hit.get("diagnoses") or []) for tr in (dx.get("treatments") or [])
        ]
        counts.append(len(records))
        n_records += len(records)
        text = json.dumps(records).lower()
        if any(marker in text for marker in RADIOTHERAPY_MARKERS) and STUPP_ALKYLATOR in text:
            rt_tmz += 1
        found = False
        for tr in records:
            if tr.get("days_to_treatment_start") is not None:
                n_stamped += 1
            agent = tr.get("therapeutic_agents")
            if agent:
                agents[str(agent)] += 1
                found = True
        if found:
            with_agent += 1

    return TreatmentExposure(
        n_cases=len(hits),
        n_records=n_records,
        n_cases_with_agent=with_agent,
        n_cases_rt_plus_tmz=rt_tmz,
        n_records_with_start_day=n_stamped,
        median_records_per_case=float(np.median(counts)) if counts else float("nan"),
        top_agents=agents.most_common(8),
    )


# ---------------------------------------------------------------------------
# The full design matrix: every term, with its own resolution
# ---------------------------------------------------------------------------


def _standard_errors(
    X: np.ndarray, t: np.ndarray, e: np.ndarray, beta: np.ndarray
) -> Optional[np.ndarray]:
    """Observed-information standard errors at the fitted coefficients.

    `CoxFit` carries coefficients but not their uncertainty, and a coefficient
    without an SE cannot be compared with zero. The information matrix is the
    one the fit already uses, evaluated at the solution, so this adds no second
    likelihood and no second convention.
    """
    from danon.predictive import _breslow_objective

    try:
        _, _, information = _breslow_objective(
            np.asarray(beta, dtype=float),
            np.asarray(X, dtype=float),
            np.asarray(t, dtype=float),
            np.asarray(e, dtype=int),
        )
        covariance = np.linalg.inv(np.asarray(information, dtype=float))
    except (np.linalg.LinAlgError, ValueError, TypeError):
        return None
    diagonal = np.diag(covariance)
    if not np.all(np.isfinite(diagonal)) or np.any(diagonal < 0.0):
        return None
    return np.sqrt(diagonal)


@dataclass(frozen=True)
class DesignMatrixEffect:
    """One covariate appended to the baseline design, with every term's HR and SE.

    `covariate_effect` answers "does held-out discrimination move?". This
    answers the prior question -- "did the covariate actually enter the model,
    and with what coefficient?" -- because a covariate can be in the design
    matrix and still be indistinguishable from zero, and that is a different
    finding from being absent.
    """

    name: str
    covariates: Tuple[str, ...]
    n_patients: int
    n_events: int
    n_censored: int
    coefficients: Dict[str, float]
    standard_errors: Dict[str, float]
    effect: CovariateEffect
    notes: List[str] = field(default_factory=list)

    @property
    def hazard_ratios(self) -> Dict[str, float]:
        return {n: float(math.exp(b)) for n, b in self.coefficients.items()}

    @property
    def z_scores(self) -> Dict[str, float]:
        return {
            n: float(b / self.standard_errors[n])
            for n, b in self.coefficients.items()
            if n in self.standard_errors and self.standard_errors[n] > 0
        }

    @property
    def hazard_ratio(self) -> Optional[float]:
        b = self.coefficients.get(self.name)
        return float(math.exp(b)) if b is not None else None

    @property
    def z(self) -> Optional[float]:
        return self.z_scores.get(self.name)

    def hazard_ratio_per(self, per: float) -> Optional[float]:
        """HR per `per` units, from the same fit rather than a second one."""
        b = self.coefficients.get(self.name)
        return float(math.exp(per * b)) if b is not None else None

    @property
    def verdict(self) -> str:
        z = self.z
        hr = self.hazard_ratio
        if hr is None or z is None:
            return "not estimable: the augmented fit did not converge"
        if abs(z) < 2.0:
            return (
                f"in the design matrix but indistinguishable from no effect "
                f"(HR={hr:.4f}, z={z:+.2f}, {self.effect.verdict})"
            )
        return (
            f"in the design matrix with a {abs(z):.1f}-SE coefficient "
            f"(HR={hr:.4f}, {self.effect.verdict})"
        )

    def __str__(self) -> str:
        terms = ", ".join(
            f"{n}={self.hazard_ratios[n]:.4f}"
            + (f"+/-{self.standard_errors[n]:.4f}" if n in self.standard_errors else "")
            for n in self.covariates
        )
        return (
            f"{self.name} in the design matrix: n={self.n_patients} "
            f"({self.n_events} events, {self.n_censored} censored); "
            f"terms [{terms}]; delta_C={self.effect.delta:+.4f} "
            f"(fold sd {self.effect.base_sd:.4f}, "
            f"{self.effect.n_folds_used}/{self.effect.n_folds_requested} folds) "
            f"[{self.verdict}]"
        )


def design_matrix_effect(
    rows: Sequence[PatientRow],
    values: Dict[str, float],
    *,
    name: str,
    n_folds: int = 5,
    seed: int = 0,
    direction: Optional[str] = None,
) -> DesignMatrixEffect:
    """Fit the baseline design plus one covariate and report every term.

    The covariate enters the *same* design as age, sex and diagnosis, on the
    same patients, so its coefficient is adjusted for them. `direction` records
    the expected sign in words, because a hazard ratio printed without its
    expected direction invites the reader to pick whichever reading flatters it.
    """
    effect = covariate_effect(rows, values, name=name, n_folds=n_folds, seed=seed)
    covariates = BASE_COVARIATES + (name,)
    subset = [r for r in rows if r.submitter_id in values]
    notes = list(effect.notes)
    if direction:
        notes.append(f"expected direction: {direction}")
    if not subset:
        return DesignMatrixEffect(
            name=name,
            covariates=covariates,
            n_patients=0,
            n_events=0,
            n_censored=0,
            coefficients={},
            standard_errors={},
            effect=effect,
            notes=notes + ["no patient in the cohort carries this covariate; refused"],
        )

    X = np.column_stack(
        [
            [r.age_years for r in subset],
            [r.male for r in subset],
            [r.is_glioblastoma for r in subset],
            [float(values[r.submitter_id]) for r in subset],
        ]
    )
    t = np.array([r.time_days for r in subset], dtype=float)
    e = np.array([r.event for r in subset], dtype=int)
    fit = cox_fit(X, t, e, covariates)
    coefficients: Dict[str, float] = {}
    standard_errors: Dict[str, float] = {}
    if fit.converged:
        coefficients = {n: float(b) for n, b in zip(fit.names, fit.coefficients)}
        se = _standard_errors(X, t, e, fit.coefficients)
        if se is not None:
            standard_errors = {n: float(s) for n, s in zip(fit.names, se)}
    else:
        notes.append(
            "the full-sample augmented fit did not converge; a covariate that "
            "nearly separates the outcome leaves the partial likelihood unbounded"
        )
    return DesignMatrixEffect(
        name=name,
        covariates=covariates,
        n_patients=len(subset),
        n_events=int(e.sum()),
        n_censored=len(subset) - int(e.sum()),
        coefficients=coefficients,
        standard_errors=standard_errors,
        effect=effect,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Restricting to the patients who record a standard regimen
# ---------------------------------------------------------------------------


def has_rt_plus_alkylator(hit: dict) -> bool:
    """Whether one case's treatment records mention radiotherapy and temozolomide."""
    records = [
        tr for dx in (hit.get("diagnoses") or []) for tr in (dx.get("treatments") or [])
    ]
    text = json.dumps(records).lower()
    return any(m in text for m in RADIOTHERAPY_MARKERS) and STUPP_ALKYLATOR in text


@dataclass(frozen=True)
class TreatmentRestriction:
    """The measured cost of keeping only patients who record RT + temozolomide.

    Two numbers are kept apart on purpose. `cohort_membership_delta` is what
    happens to the *baseline* score when the cohort is cut down -- a selection
    effect. The `effects` are measured inside the restricted cohort, paired on
    the same patients. A gain measured without the first number is a selection
    effect wearing a covariate's name.
    """

    n_cohort: int
    n_restricted: int
    base_c_full: float
    base_sd_full: float
    base_c_restricted: float
    base_sd_restricted: float
    age_only_c_restricted: float
    age_only_c_full: float
    effects: Tuple[CovariateEffect, ...]
    exposure: Optional[TreatmentExposure]
    notes: List[str] = field(default_factory=list)

    @property
    def cohort_membership_delta(self) -> float:
        return self.base_c_restricted - self.base_c_full

    @property
    def verdict(self) -> str:
        if not (np.isfinite(self.base_c_full) and np.isfinite(self.base_c_restricted)):
            return "not estimable"
        if abs(self.cohort_membership_delta) >= self.base_sd_full:
            return (
                f"restriction alone moves the baseline by "
                f"{self.cohort_membership_delta:+.4f} C, which is outside one fold "
                f"sd ({self.base_sd_full:.4f}): this is cohort selection, and any "
                f"covariate delta inside it is not separable from it"
            )
        return (
            f"restriction moves the baseline by {self.cohort_membership_delta:+.4f} C, "
            f"inside one fold sd ({self.base_sd_full:.4f})"
        )

    def __str__(self) -> str:
        lines = [
            f"treatment restriction (RT + temozolomide): kept {self.n_restricted} of "
            f"{self.n_cohort} patients; base C {self.base_c_full:.4f}+/-{self.base_sd_full:.4f} "
            f"(all) -> {self.base_c_restricted:.4f}+/-{self.base_sd_restricted:.4f} "
            f"(restricted), cohort-membership delta "
            f"{self.cohort_membership_delta:+.4f}; age-only on the restricted set "
            f"{self.age_only_c_restricted:.4f} [{self.verdict}]"
        ]
        for eff in self.effects:
            lines.append(f"  within the restricted set: {eff}")
        return "\n".join(lines)


def treatment_restriction_effect(
    rows: Sequence[PatientRow],
    hits: Sequence[dict],
    *,
    covariates: Sequence[Tuple[str, Dict[str, float]]] = (),
    n_folds: int = 5,
    seed: int = 0,
    exposure: Optional[TreatmentExposure] = None,
) -> TreatmentRestriction:
    """Measure a regimen restriction as cohort selection first, covariate second."""
    credited = {h.get("submitter_id") for h in hits if has_rt_plus_alkylator(h)}
    kept = [r for r in rows if r.submitter_id in credited]
    full = _cv(rows, None, n_folds=n_folds, seed=seed)
    restricted = _cv(kept, None, n_folds=n_folds, seed=seed)
    effects = tuple(
        covariate_effect(kept, values, name=name, n_folds=n_folds, seed=seed)
        for name, values in covariates
    )
    notes = [
        f"{len(credited)} case(s) in the treatment export record radiotherapy plus "
        f"{STUPP_ALKYLATOR}; {len(rows) - len(kept)} cohort patient(s) dropped for "
        f"not recording it",
        "a record is an exposure, not an ordered protocol: start days are missing "
        "for most treatment records, so first-line therapy cannot be separated "
        "from salvage therapy in this export",
    ]
    return TreatmentRestriction(
        n_cohort=len(rows),
        n_restricted=len(kept),
        base_c_full=full.mean_c,
        base_sd_full=full.sd_c,
        base_c_restricted=restricted.mean_c,
        base_sd_restricted=restricted.sd_c,
        age_only_c_restricted=restricted.age_only_c,
        age_only_c_full=full.age_only_c,
        effects=effects,
        exposure=exposure,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# The ceiling: the best model the data actually supports
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CeilingReport:
    """Pooled out-of-fold discrimination for the fullest model available.

    This exists to answer one question with a measurement instead of a target:
    how high can held-out discrimination on this cohort actually go? The answer
    is what it is, and a requested target above it is a target the data cannot
    carry.
    """

    model: str
    covariates: Tuple[str, ...]
    n_patients: int
    n_events: int
    n_censored: int
    pooled_c: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    age_only_c: float
    majority_class_accuracy: float
    n_folds_used: int
    n_folds_requested: int
    notes: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.pooled_c):
            return "not estimable"
        if self.n_folds_used < self.n_folds_requested:
            return f"not estimable: only {self.n_folds_used} of {self.n_folds_requested} folds converged"
        inside = self.ci_low <= 0.5 <= self.ci_high
        chance = "the interval includes 0.5" if inside else "the interval excludes 0.5"
        return (
            f"pooled out-of-fold C = {self.pooled_c:.4f} "
            f"(95% bootstrap CI {self.ci_low:.4f}-{self.ci_high:.4f}, {chance}); "
            f"age alone {self.age_only_c:.4f}. Majority-class accuracy on the same "
            f"cohort is {100 * self.majority_class_accuracy:.2f}%, which is what an "
            f"'accuracy' claim of that size would mean without any model"
        )

    def __str__(self) -> str:
        return (
            f"ceiling [{self.model}]: covariates {list(self.covariates)}; "
            f"n={self.n_patients} ({self.n_events} events, {self.n_censored} censored); "
            f"{self.verdict}"
        )


def predictive_ceiling(
    rows: Sequence[PatientRow],
    features: Sequence[Tuple[str, Dict[str, float]]],
    *,
    model: str = "baseline + all available covariates",
    n_folds: int = 5,
    seed: int = 0,
    n_bootstrap: int = 400,
    bootstrap_seed: int = 0,
) -> CeilingReport:
    """Pooled out-of-fold C for the fullest design, with a patient bootstrap CI.

    Only patients carrying every covariate are used, so the number is honest
    about what it costs. Pooled out-of-fold risks are used rather than an
    average of per-fold scores, because the average hides how many patients each
    fold actually scored.
    """
    names = [name for name, _ in features]
    subset = [
        r for r in rows if all(r.submitter_id in values for _, values in features)
    ]
    covariates = BASE_COVARIATES + tuple(names)
    if not subset:
        return CeilingReport(
            model=model, covariates=covariates, n_patients=0, n_events=0, n_censored=0,
            pooled_c=float("nan"), ci_low=float("nan"), ci_high=float("nan"),
            n_bootstrap=0, age_only_c=float("nan"),
            majority_class_accuracy=float("nan"), n_folds_used=0,
            n_folds_requested=n_folds,
            notes=["no patient carries every covariate; refused"],
        )

    X = np.column_stack(
        [[r.age_years for r in subset], [r.male for r in subset],
         [r.is_glioblastoma for r in subset]]
        + [[float(values[r.submitter_id]) for r in subset] for _, values in features]
    )
    t = np.array([r.time_days for r in subset], dtype=float)
    e = np.array([r.event for r in subset], dtype=int)

    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(len(subset)), n_folds)
    risks = np.full(len(subset), np.nan)
    n_used = 0
    for fold in folds:
        test = np.zeros(len(subset), dtype=bool)
        test[fold] = True
        train = ~test
        if e[train].sum() < 3 or len(fold) < 3:
            continue
        fit = cox_fit(X[train], t[train], e[train], covariates)
        if not fit.converged:
            continue
        risks[test] = fit.risk(X[test])
        n_used += 1
    scored = np.isfinite(risks)
    pooled = harrell_c_index(risks[scored], t[scored], e[scored])
    age_c = harrell_c_index(X[scored, 0], t[scored], e[scored])

    boot: List[float] = []
    if pooled.estimable:
        brng = np.random.RandomState(bootstrap_seed)
        n_scored = int(scored.sum())
        for _ in range(n_bootstrap):
            pick = brng.randint(0, n_scored, n_scored)
            c = harrell_c_index(risks[scored][pick], t[scored][pick], e[scored][pick])
            if c.estimable:
                boot.append(c.c_index)
    arr = np.asarray(boot, dtype=float)
    n_events = int(e[scored].sum())
    n_scored_total = int(scored.sum())
    majority = (
        max(n_events, n_scored_total - n_events) / n_scored_total
        if n_scored_total
        else float("nan")
    )
    notes = [
        f"{len(rows) - len(subset)} of {len(rows)} cohort patient(s) dropped: at "
        f"least one covariate absent",
        f"risks pooled out of fold; {len(arr)} of {n_bootstrap} bootstrap resamples "
        f"were estimable",
        "a C-index of 0.90 would mean 90% of random patient pairs ordered "
        "correctly; accuracy on a binary outcome is a different quantity, and "
        "the majority class already scores the value above with no model",
    ]
    return CeilingReport(
        model=model,
        covariates=covariates,
        n_patients=n_scored_total,
        n_events=n_events,
        n_censored=n_scored_total - n_events,
        pooled_c=pooled.c_index if pooled.estimable else float("nan"),
        ci_low=float(np.percentile(arr, 2.5)) if arr.size else float("nan"),
        ci_high=float(np.percentile(arr, 97.5)) if arr.size else float("nan"),
        n_bootstrap=len(arr),
        age_only_c=age_c.c_index if age_c.estimable else float("nan"),
        majority_class_accuracy=float(majority),
        n_folds_used=n_used,
        n_folds_requested=n_folds,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------


def covariate_upgrade_report(
    *,
    clinical_path: str = GDC_CLINICAL_PATH,
    endpoint: str = "os",
    n_folds: int = 5,
    seed: int = 0,
    n_bootstrap: int = 400,
    treatment_path: str = TREATMENT_RECORDS_PATH,
    mgmt_values: Optional[Dict[str, float]] = None,
    mgmt_beta_values: Optional[Dict[str, float]] = None,
    mgmt_beta_source: str = "",
    treatment_hits: Optional[Sequence[dict]] = None,
) -> str:
    """The four proposed upgrades, each implemented or refused, each measured."""
    with open(clinical_path) as fh:
        cases = json.load(fh).get("hits", [])
    rows, notes = load_cohort_rows(clinical_path, endpoint=endpoint)

    lines: List[str] = []
    lines.append("=" * 74)
    lines.append("COVARIATE UPGRADES: implemented or refused, each measured")
    lines.append("=" * 74)
    lines.append("")
    for note in notes:
        lines.append(f"  {note}")
    lines.append("")

    lines.append("-- upgrade 3: pre-treatment KPS beside age and sex --")
    kps = baseline_kps(cases)
    kps_design = design_matrix_effect(
        rows, kps, name="kps_baseline", n_folds=n_folds, seed=seed,
        direction="higher KPS is better, so a hazard ratio below 1 is expected",
    )
    lines.append(f"  {kps_design}")
    hr10 = kps_design.hazard_ratio_per(10.0)
    if hr10 is not None:
        lines.append(f"  the same fit, reported per +10 KPS points: HR={hr10:.4f}")
    lines.append(f"  univariate contrast: {kps_leakage_contrast(rows, cases)}")
    for note in kps_design.notes:
        lines.append(f"      note: {note}")
    lines.append("")

    lines.append("-- upgrade 1: MGMT promoter methylation --")
    mgmt_design: Optional[DesignMatrixEffect] = None
    if mgmt_values:
        mgmt_design = design_matrix_effect(
            rows, mgmt_values, name="mgmt_methylated", n_folds=n_folds, seed=seed,
            direction="methylated MGMT is silenced, so a hazard ratio below 1 is expected",
        )
        lines.append(f"  binary published call: {mgmt_design}")
        for note in mgmt_design.notes:
            lines.append(f"      note: {note}")
    else:
        lines.append("  binary published call: not loaded in this run")
    beta_design: Optional[DesignMatrixEffect] = None
    if mgmt_beta_values:
        beta_design = design_matrix_effect(
            rows, mgmt_beta_values, name="mgmt_beta", n_folds=n_folds, seed=seed,
            direction="a higher promoter beta means less repair, so a hazard ratio below 1 is expected",
        )
        lines.append(
            f"  continuous beta ({mgmt_beta_source or 'source not recorded'}): {beta_design}"
        )
        for note in beta_design.notes:
            lines.append(f"      note: {note}")
    else:
        lines.append("  continuous beta: not loaded in this run")
    lines.append("")

    hits = list(treatment_hits or [])
    lines.append("-- upgrade 4: restrict to patients recording RT + temozolomide --")
    restriction = treatment_restriction_effect(
        rows,
        hits,
        covariates=[("kps_baseline", kps)]
        + ([("mgmt_methylated", mgmt_values)] if mgmt_values else []),
        n_folds=n_folds,
        seed=seed,
        exposure=treatment_exposure(hits),
    )
    for line in str(restriction).splitlines():
        lines.append(f"  {line}")
    for note in restriction.notes:
        lines.append(f"      note: {note}")
    lines.append("")

    lines.append("-- upgrade 2: 3D sub-region segmentation --")
    lines.append(
        "  refused, unchanged: no collection with expert masks is retrievable, "
        "and danon.multimodal.spatial_dice_verdict derives that from live counts"
    )
    lines.append("")

    lines.append("-- the ceiling: what the fullest model the data supports reaches --")
    ceiling_features: List[Tuple[str, Dict[str, float]]] = [
        ("kps_baseline", kps),
    ]
    if mgmt_values:
        ceiling_features.append(("mgmt_methylated", mgmt_values))
    if mgmt_beta_values:
        ceiling_features.append(("mgmt_beta", mgmt_beta_values))
    ceiling = predictive_ceiling(
        rows, ceiling_features, n_folds=n_folds, seed=seed, n_bootstrap=n_bootstrap
    )
    lines.append(f"  {ceiling}")
    for note in ceiling.notes:
        lines.append(f"      note: {note}")
    lines.append("")
    lines.append(
        "Every delta above is paired on the patients its subset keeps, and every"
    )
    lines.append(
        "subset change is reported with its own baseline, so a change in n is"
    )
    lines.append("never reported as a change in signal.")
    return "\n".join(lines)



def cohort_feature_report(
    *,
    clinical_path: str = GDC_CLINICAL_PATH,
    endpoint: str = "os",
    n_folds: int = 5,
    seed: int = 0,
    mgmt_reachability: Optional[MgmtReachability] = None,
) -> str:
    """Everything above, measured on the corrected endpoint, as text."""
    with open(clinical_path) as fh:
        cases = json.load(fh).get("hits", [])
    rows, notes = load_cohort_rows(clinical_path, endpoint=endpoint)

    lines: List[str] = []
    lines.append("=" * 74)
    lines.append("COVARIATE INTAKE: measured effects on the corrected endpoint")
    lines.append("=" * 74)
    lines.append("")
    lines.append(f"endpoint: {endpoint}")
    for note in notes:
        lines.append(f"  {note}")
    lines.append("")

    lines.append("-- KPS timepoint census (why the baseline gate exists) --")
    for row in kps_timepoint_census(cases):
        stamp = f"median day {row.median_day:.0f}" if row.median_day is not None else "NO DAY STAMP"
        flag = " <-- baseline" if row.is_baseline else ""
        lines.append(
            f"  {row.timepoint:26s} records={row.n_records:5d} kps={row.n_kps_records:4d} "
            f"numeric={row.n_numeric_kps:4d}  {stamp}{flag}"
        )
    lines.append("")
    lines.append(f"-- {kps_coverage(cases)}")
    lines.append(f"-- {kps_leakage_contrast(rows, cases)}")
    lines.append("")

    lines.append("-- measured effect of each proposed covariate --")
    effects = [
        covariate_effect(rows, baseline_kps(cases), name="kps_baseline", n_folds=n_folds, seed=seed),
    ]
    for eff in effects:
        lines.append(f"  {eff}")
        for note in eff.notes:
            lines.append(f"      note: {note}")
    lines.append("")
    lines.append("  Not measured here, and why:")
    if mgmt_reachability is not None:
        lines.append(
            f"    mgmt_methylation     -- {mgmt_reachability.n_beta_files} open beta "
            f"files exist across {mgmt_reachability.n_patients_total} patients "
            f"({mgmt_reachability.n_patients_450k} on 450k), so wiring MGMT in caps "
            f"the fitting cohort at {mgmt_reachability.n_patients_450k} patients -- "
            f"fewer than the {len(rows)} the baseline design uses. The delta must be "
            f"measured on that subset, not asserted"
        )
    else:
        lines.append(
            "    mgmt_methylation     -- reachability not probed in this run; call "
            "probe_mgmt_reachability() to get real patient counts"
        )
    lines.append(
        "    subregion_segmentation -- refused: no collection with expert masks is "
        "retrievable (danon.multimodal.spatial_dice_verdict derives this from live counts)"
    )
    lines.append("")

    treatments: List[dict] = []
    try:
        with open(TREATMENT_RECORDS_PATH) as fh:
            treatments = list(json.load(fh).get("hits", []))
    except (OSError, ValueError, TypeError):
        pass
    exposure = treatment_exposure(treatments)
    lines.append(f"-- {exposure}")
    if exposure.top_agents:
        lines.append(
            "     top agents: "
            + ", ".join(f"{a} ({n})" for a, n in exposure.top_agents[:6])
        )
    lines.append("")
    if mgmt_reachability is not None:
        lines.append(f"-- {mgmt_reachability}")
        lines.append("")

    lines.append("A covariate whose delta is inside the fold-to-fold sd is reported as")
    lines.append("'no measurable gain'. Nothing in this report is an estimate.")
    return "\n".join(lines)


def cohort_feature_provenance() -> Dict[str, EvidenceStatus]:
    """What each quantity here is, in the repo's provenance vocabulary."""
    return {
        "kps_baseline_values": EvidenceStatus.ESTABLISHED,      # read from the GDC export
        "kps_timepoint_gate": EvidenceStatus.ESTABLISHED,       # justified by the census
        "kps_measured_delta_c": EvidenceStatus.ESTABLISHED,     # computed, not estimated
        "kps_leakage_contrast": EvidenceStatus.ESTABLISHED,     # computed
        "mgmt_probe_selection": EvidenceStatus.UNKNOWN,         # literature-derived, not file-derived
        "mgmt_methylation_cutoff": EvidenceStatus.UNKNOWN,      # model-specific
        "mgmt_beta_values": EvidenceStatus.ESTABLISHED,         # read from real files
        "treatment_exposure_counts": EvidenceStatus.ESTABLISHED,
        "treatment_protocol_restriction": EvidenceStatus.PROVEN_FAILED,  # start days missing
        "subregion_segmentation": EvidenceStatus.PROVEN_FAILED,          # no masks retrievable
    }


# ---------------------------------------------------------------------------
# Design calculation: what a cohort of a given size can resolve at all
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvableDelta:
    """The smallest incremental C-index a cohort of this size can resolve.

    THIS IS A DESIGN CALCULATION, NOT A MEASUREMENT OF ANY STREAM. It answers
    the question that precedes every covariate proposal: given a cohort of
    this size drawn from this population, how much does the held-out C move
    for no reason at all? That movement is the floor a claimed gain has to
    clear, and it is what turns "we would need more data" into a number.

    Two floors are reported, because they answer two different questions:

    * ``se_across_draws`` -- the spread of the held-out C over independent
      cohorts of this size. This is the classical standard error, and it is
      the yardstick for "would a different cohort of the same size reproduce
      this gain?"
    * ``mean_fold_sd`` -- the spread across the k folds *within* one cohort,
      which is the yardstick `covariate_effect.verdict` already uses. At a
      small n the folds are few and heterogeneous, so this is the more
      conservative floor.

    Quoting only the conservative floor would overstate the data requirement;
    quoting only the optimistic one would launder fold noise as signal.
    """

    n_patients: int
    n_draws: int
    n_folds: int
    mean_c: float
    se_across_draws: float
    mean_fold_sd: float
    age_only_c: float
    exhaustive_draws: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def floor_optimistic(self) -> float:
        return self.se_across_draws

    @property
    def floor_conservative(self) -> float:
        return self.mean_fold_sd

    @property
    def estimable(self) -> bool:
        return self.n_draws > 1 and np.isfinite(self.se_across_draws)

    def resolves(self, delta: float) -> bool:
        """Whether a claimed gain ``delta`` clears the optimistic floor."""
        return self.estimable and abs(delta) > self.floor_optimistic

    def verdict(self) -> str:
        if not self.estimable:
            return "not estimable: too few draws produced a finite held-out C"
        return (
            f"resolves |delta_C| > {self.floor_optimistic:.4f} "
            f"(1 SE over {self.n_draws} draws); fold yardstick {self.floor_conservative:.4f}"
        )

    def __str__(self) -> str:
        return (
            f"n={self.n_patients} over {self.n_draws} draws ({self.n_folds}-fold): "
            f"C={self.mean_c:.4f} +/- {self.se_across_draws:.4f} (age alone "
            f"{self.age_only_c:.4f}), mean fold sd {self.mean_fold_sd:.4f} -- "
            f"{self.verdict()}"
        )


def resolvable_delta_c_index(
    rows: Sequence[PatientRow],
    *,
    n_patients: int,
    n_draws: int = 24,
    n_folds: int = 5,
    base_seed: int = 7000,
) -> ResolvableDelta:
    """Resample cohorts of ``n_patients`` from real rows and measure the floor.

    Each draw is an independent sample of ``n_patients`` WITHOUT replacement
    from the real cohort, followed by a fresh k-fold split, so patient
    sampling and split variation are both present -- which is what a real
    cohort of that size would present.

    When ``n_patients`` equals the row count there is only one distinct draw,
    so the loop samples permutations and the spread measures split variation
    alone. That case is flagged in ``exhaustive_draws`` and carries a note,
    because the two spreads are not interchangeable: an SE measured on the
    full cohort is not the SE a smaller cohort would have.

    The floor moves with the model, not only with n -- a design with more
    free parameters has a higher floor at the same n -- so this is measured
    for the design actually being reported (age, sex, GBM flag), not in the
    abstract.
    """
    if n_patients < n_folds * 5:
        raise ValueError(
            f"{n_patients} patients is too few for a {n_folds}-fold split "
            f"(the existing house rule is {n_folds * 5})"
        )
    if n_patients > len(rows):
        raise ValueError(
            f"cannot draw {n_patients} patients from a cohort of {len(rows)}"
        )
    if n_draws < 2:
        raise ValueError("n_draws must be at least 2 to have a spread at all")

    means: List[float] = []
    fold_sds: List[float] = []
    ages: List[float] = []
    for draw in range(n_draws):
        rng = np.random.RandomState(base_seed + draw)
        idx = rng.choice(len(rows), size=n_patients, replace=False)
        subset = [rows[i] for i in idx]
        scores = _cv(subset, None, n_folds=n_folds, seed=draw)
        if not np.isfinite(scores.mean_c):
            continue
        means.append(scores.mean_c)
        fold_sds.append(scores.sd_c)
        if np.isfinite(scores.age_only_c):
            ages.append(scores.age_only_c)

    exhaustive = n_patients == len(rows)
    notes = [
        f"{n_draws - len(means)} of {n_draws} draws produced no finite held-out C and "
        f"were dropped"
    ] if len(means) < n_draws else []
    if exhaustive:
        notes.append(
            "n_patients equals the cohort size, so every draw is the same patient "
            "set: the spread measures split variation only, not cohort sampling"
        )
    notes.append(
        "design calculation: this bounds what a cohort of this size could show, "
        "it does not measure any stream, covariate, or treatment effect"
    )

    if len(means) < 2:
        return ResolvableDelta(
            n_patients=n_patients, n_draws=len(means), n_folds=n_folds,
            mean_c=float("nan"), se_across_draws=float("nan"),
            mean_fold_sd=float("nan"), age_only_c=float("nan"),
            exhaustive_draws=exhaustive, notes=notes,
        )

    m = np.asarray(means, dtype=float)
    return ResolvableDelta(
        n_patients=n_patients,
        n_draws=len(means),
        n_folds=n_folds,
        mean_c=float(m.mean()),
        se_across_draws=float(m.std(ddof=1)),
        mean_fold_sd=float(np.mean(fold_sds)),
        age_only_c=float(np.mean(ages)) if ages else float("nan"),
        exhaustive_draws=exhaustive,
        notes=notes,
    )


def what_covariate_intake_cannot_show() -> List[str]:
    """Stated plainly so no section of the report is read as clinical evidence."""
    return [
        "A higher C-index from a covariate is a statement about the covariate, "
        "not about the engine. The engine consumes neither KPS nor MGMT by default.",
        "MGMT betas read here are from donor tumours, not from the patient being "
        "modelled, so they calibrate a distribution rather than a case.",
        "Treatment exposure is not treatment protocol: without reliable start days, "
        "a temozolomide record cannot be separated from salvage therapy.",
        "No covariate in this module has been shown to change any patient's outcome.",
    ]


__all__ = [
    "BASE_COVARIATES",
    "KPS_BASELINE_TIMEPOINTS",
    "KPS_POST_BASELINE_TIMEPOINTS",
    "KPS_UNUSABLE",
    "MGMT_METHYLATION_CUTOFF",
    "MGMT_PROBES",
    "CovariateEffect",
    "KpsCoverage",
    "LeakageContrast",
    "MgmtBetaResult",
    "MgmtReachability",
    "ResolvableDelta",
    "TimepointCensus",
    "TreatmentExposure",
    "baseline_kps",
    "cohort_feature_provenance",
    "cohort_feature_report",
    "covariate_effect",
    "extract_mgmt_beta",
    "kps_coverage",
    "kps_leakage_contrast",
    "kps_timepoint_census",
    "parse_kps",
    "post_baseline_kps",
    "probe_mgmt_reachability",
    "resolvable_delta_c_index",
    "treatment_exposure",
    "what_covariate_intake_cannot_show",
]
