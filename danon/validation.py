"""
Validation tiers: what can be tested without a lab, and what cannot.

A model is not validated by being impressive. This module implements the three
experimental proxies for testing a computational model against real cancer,
with the statistics each tier needs, and it refuses to produce a statistic that
its data cannot support:

**Tier 1 -- public cohort (no lab).** A risk score is validated on real
patients, with the fit and the score kept apart: `patient_level_split` is
patient-level and deterministic, and every statistic reported by
`validate_survival_stratification` is computed on the held-out split only.

**Tier 2 -- public pharmacology (no lab).** Predicted IC50s are compared with
measured ones from public screens (GDSC / CTRP). Agreement is reported on the
log10 scale because potency is multiplicative, and the *bias* is reported
separately because R^2 is shift-invariant: a 1000x uM-vs-nM units error leaves
R^2 untouched and is visible only in the bias term.

**Tier 3 -- cell-line wet lab.** Pre-registered predictions with acceptance
ranges, so a result cannot be re-interpreted after the fact, plus per-line
caveats (U87-MG lineage drift, LN229 p53 background) that are part of the
interpretation rather than footnotes.

**Tier 4 -- patient-derived.** Organoids / xenografts. Represented in the tier
ladder and not implemented: this repository has no such data.

The statistics are implemented here rather than imported so their failure modes
are visible. Two regressions are pinned by `tests/test_validation.py` and worth
naming: an early `concordance_index` counted *subjects* in the denominator while
accumulating *pairs* in the numerator and returned C = 2.92 on a demo cohort,
and the C-index and hazard ratio must always agree about a bad score's direction
-- a statistic that flatters a sign-flipped model is worse than no statistic.

References:
  Harrell FE et al. Evaluating the ability of a prognostic model. JAMA 1982.
  Mantel N. Evaluation of survival data and two new rank order statistics.
  Cancer Chemother Rep 1966;50(3):163-170 (log-rank).
  Iorio F et al. A landscape of pharmacogenomic interactions in cancer. Cell 2016
  (GDSC); Seashore-Ludlow B et al. Cancer Cell 2015 (CTRP).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus

# ---------------------------------------------------------------------------
# Tier ladder
# ---------------------------------------------------------------------------


class ValidationTier(str, Enum):
    """The four experimental proxies, weakest-to-strongest evidence."""

    PUBLIC_COHORT = "public_cohort"
    PUBLIC_PHARMACOLOGY = "public_pharmacology"
    CELL_LINE_WETLAB = "cell_line_wetlab"
    PATIENT_DERIVED = "patient_derived"


def tier_overview() -> List[Dict[str, str]]:
    """The ladder, with what each tier proves and what it cannot."""
    return [
        {
            "tier": ValidationTier.PUBLIC_COHORT.value,
            "name": "Public cohort, retrospective (TCGA / GDC)",
            "requires_lab": "False",
            "data": "de-identified clinical and molecular exports",
            "proves": (
                "that the model's risk score transfers to held-out real "
                "patients better than chance"
            ),
            "cannot_prove": (
                "causality or treatment benefit: these patients were not "
                "randomised to the model's recommendation"
            ),
        },
        {
            "tier": ValidationTier.PUBLIC_PHARMACOLOGY.value,
            "name": "Public pharmacology (GDSC / CTRP IC50 screens)",
            "requires_lab": "False",
            "data": "measured potency across cell lines and compounds",
            "proves": (
                "that predicted drug potency agrees with measured potency on the "
                "same lines and compounds, including its systematic bias"
            ),
            "cannot_prove": (
                "in-vivo effect: two screens differ in assay, seeding density "
                "and growth medium, and cross-screen comparability is limited"
            ),
        },
        {
            "tier": ValidationTier.CELL_LINE_WETLAB.value,
            "name": "Cell-line wet lab, pre-registered predictions",
            "requires_lab": "True",
            "data": "predictions with acceptance ranges, generated before the run",
            "proves": (
                "that a quantitative in-silico prediction survives contact with "
                "living cells inside a pre-declared range"
            ),
            "cannot_prove": (
                "patient relevance: established lines carry lineage drift "
                "(U87-MG) and pathway backgrounds (LN229 p53) that differ from "
                "primary tumours"
            ),
        },
        {
            "tier": ValidationTier.PATIENT_DERIVED.value,
            "name": "Patient-derived organoids / xenografts",
            "requires_lab": "True",
            "data": "primary patient material under ethics approval",
            "proves": (
                "that the prediction holds in material derived from the actual "
                "tumour type, including patient-to-patient variability"
            ),
            "cannot_prove": (
                "clinical benefit: organoids lack immune and stromal context, "
                "and ex-vivo dosing is not a trial"
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Tier 1: the concordance index
# ---------------------------------------------------------------------------


def concordance_index(
    times: Sequence[float],
    events: Sequence[int],
    risk_scores: Sequence[float],
) -> float:
    """Harrell's C for right-censored data.

    A pair is comparable when the earlier of the two is an observed event; a
    pair whose earlier member was censored is uninformative and excluded. The
    pair is concordant when the higher risk score failed first; tied scores
    contribute 0.5. The denominator is therefore the number of comparable
    PAIRS -- the bug this function exists to prevent was dividing a pair count
    by a subject count, which yielded C > 1.

    Raises `ValueError` when no pair is comparable, because C is then undefined
    rather than 0.5.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    r = np.asarray(risk_scores, dtype=float)
    if not (len(t) == len(e) == len(r)):
        raise ValueError("times, events and risk_scores must have equal length")
    if len(t) == 0:
        raise ValueError("empty cohort")
    if np.any(~np.isfinite(r)):
        raise ValueError("risk scores must be finite")
    if np.any(t < 0):
        raise ValueError("negative survival times are not representable")

    comparable = 0
    concordant = 0.0
    n = len(t)
    for i in range(n):
        for j in range(i + 1, n):
            if t[i] == t[j]:
                if e[i] and e[j]:
                    comparable += 1
                    concordant += 0.5 if r[i] != r[j] else 0.5
                continue
            first, second = (i, j) if t[i] < t[j] else (j, i)
            if not e[first]:
                continue
            comparable += 1
            if r[first] > r[second]:
                concordant += 1.0
            elif r[first] == r[second]:
                concordant += 0.5
    if comparable == 0:
        raise ValueError(
            "no comparable pairs: every pair is censored before an observed "
            "event, so the concordance index is undefined"
        )
    return float(concordant / comparable)


# ---------------------------------------------------------------------------
# Tier 1: patient-level holdout
# ---------------------------------------------------------------------------


def patient_level_split(
    patient_ids: Sequence[str],
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Deterministic, disjoint, exhaustive patient-level split.

    Rows are never split: every measurement of a patient stays on one side.
    The permutation is seeded, so the same cohort and seed always give the same
    split -- a validation result that cannot be reproduced is not a result.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(
            f"test_fraction must be strictly between 0 and 1, got {test_fraction!r}"
        )
    ids = list(patient_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate patient ids in cohort")
    if len(ids) < 5:
        raise ValueError(
            f"need at least 5 patients to split (got {len(ids)}); a smaller "
            f"cohort cannot support a held-out estimate"
        )
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(ids))
    n_test = max(1, int(round(len(ids) * test_fraction)))
    test_idx = sorted(order[:n_test].tolist())
    train_idx = sorted(order[n_test:].tolist())
    return ([ids[i] for i in train_idx], [ids[i] for i in test_idx])


# ---------------------------------------------------------------------------
# Tier 1: scale check
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScaleCheck:
    plausible_transformed: bool
    message: str
    median_value: float


def log2_transform_check(values: Sequence[float]) -> ScaleCheck:
    """Is this expression sample on a log scale, or still raw counts?

    Comparing a log2 expression vector against raw microarray counts produces
    a plausible-looking correlation with the wrong sign, so the scale is
    checked before any statistic is computed. The test is deliberately crude
    and states its rule: a median above 50, or any negative value, is not a
    log2 scale.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        raise ValueError("empty expression sample")
    median = float(np.median(arr))
    if np.any(arr < 0):
        return ScaleCheck(
            False,
            "sample contains negative values: not a log2 expression scale, and "
            "not a valid raw-count vector either -- check the processing step",
            median,
        )
    if median > 50.0:
        return ScaleCheck(
            False,
            f"median {median:.1f} is on the RAW COUNTS scale, not log2: apply "
            f"log2(x + 1) before comparing against a log-scale model",
            median,
        )
    return ScaleCheck(
        True,
        f"median {median:.2f} is plausible for a log2-transformed expression "
        f"scale",
        median,
    )


# ---------------------------------------------------------------------------
# Tier 1: the log-rank test
# ---------------------------------------------------------------------------


def logrank_two_group(
    times: Sequence[float],
    events: Sequence[int],
    group_a: Sequence[bool],
) -> Dict[str, float]:
    """Mantel-Cox log-rank comparing two groups, with a hazard ratio.

    Returns `logrank_p_value`, `chi_square`, `observed_a`, `expected_a`,
    `hazard_ratio` and counts. Raises `ValueError` when the test is not
    estimable: no events, or no variance in the risk-set composition, which
    happens when one arm has every event censored.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    a = np.asarray(group_a, dtype=bool)
    if not (len(t) == len(e) == len(a)):
        raise ValueError("times, events and group_a must have equal length")
    if e.sum() == 0:
        raise ValueError("no events: the log-rank test is not estimable")
    if a.all() or (~a).all():
        raise ValueError("both groups must be non-empty")

    o_a = e_a = 0.0
    var = 0.0
    for time in np.unique(t[e == 1]):
        at_risk = t >= time
        n_a = float(np.sum(at_risk & a))
        n_b = float(np.sum(at_risk & ~a))
        n = n_a + n_b
        d_a = float(np.sum((t == time) & a & (e == 1)))
        d_b = float(np.sum((t == time) & ~a & (e == 1)))
        d = d_a + d_b
        if n < 2 or d == 0:
            continue
        e_a_t = d * n_a / n
        o_a += d_a
        e_a += e_a_t
        # hypergeometric variance of d_a given the risk set
        if n > 1:
            var += (d * (n_a / n) * (1 - n_a / n) * (n - d)) / (n - 1)

    if var <= 0.0:
        raise ValueError(
            "zero variance across event times: no information to compare the "
            "arms (one group contributes no events)"
        )
    chi2 = (o_a - e_a) ** 2 / var
    p = math.erfc(math.sqrt(max(chi2, 0.0) / 2.0))

    o_b = float(e.sum()) - o_a
    e_b = float(e.sum()) - e_a
    if o_a > 0 and o_b > 0 and e_a > 0 and e_b > 0:
        hr = (o_a / e_a) / (o_b / e_b)
    else:
        hr = float("inf") if o_a > 0 else 0.0
    return {
        "logrank_p_value": float(p),
        "chi_square": float(chi2),
        "observed_a": float(o_a),
        "expected_a": float(e_a),
        "observed_b": float(o_b),
        "expected_b": float(e_b),
        "variance": float(var),
        "hazard_ratio": float(hr),
        "n_a": int(a.sum()),
        "n_b": int((~a).sum()),
        "n_events": int(e.sum()),
    }


# ---------------------------------------------------------------------------
# Tier 1: end-to-end survival validation
# ---------------------------------------------------------------------------


def _cox_univariate_hr(
    time: np.ndarray, event: np.ndarray, score: np.ndarray,
    *, max_iter: int = 100, tol: float = 1e-10,
) -> float:
    """Hazard ratio per unit risk score, by Newton-Raphson on Breslow's
    partial likelihood (tied event times approximate the risk set).

    Kept univariate so it cannot disagree with the concordance index about the
    direction of an effect. The weights must carry `beta`: a gradient evaluated
    at a fixed beta is a constant, and iterating on it diverges to a confidently
    wrong hazard ratio -- which is exactly how a well-ordered score once came
    back with HR < 1.
    """
    event_idx = np.where(event == 1)[0]
    beta = 0.0
    for _ in range(max_iter):
        grad = 0.0
        info = 0.0
        for i in event_idx:
            at_risk = time >= time[i]
            s = score[at_risk]
            z = beta * s
            z = z - z.max()
            w = np.exp(z)
            s0 = float(w.sum())
            mean = float((s * w).sum() / s0)
            var = float(((s ** 2) * w).sum() / s0) - mean ** 2
            grad += score[i] - mean
            info += max(var, 0.0)
        if info <= 0.0:
            raise ValueError("no information for a hazard ratio in this subset")
        step = grad / info
        beta += step
        if abs(step) < tol:
            break
    return float(math.exp(beta))


@dataclass
class CohortValidationResult:
    c_index: float
    c_index_interpretation: str
    hazard_ratio: float
    logrank_p_value: float
    n_patients: int
    n_events: int
    split: Tuple[List[str], List[str]]
    scale_check: Optional[ScaleCheck] = None
    notes: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.c_index <= 0.5:
            return "score does not discriminate on held-out patients"
        if self.logrank_p_value < 0.05:
            return "score discriminates on held-out patients at p < 0.05"
        return "apparent discrimination does not reach significance"


def validate_survival_stratification(
    patient_ids: Sequence[str],
    times: Sequence[float],
    events: Sequence[int],
    risk_scores: Mapping[str, float],
    *,
    expression_sample: Optional[Sequence[float]] = None,
    test_fraction: float = 0.3,
    seed: int = 0,
    min_events: int = 8,
) -> CohortValidationResult:
    """Held-out validation of a per-patient risk score.

    Order of operations matters and is enforced: input integrity first, then a
    patient-level split, then every statistic on the held-out side only.
    """
    ids = list(patient_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate patient ids in cohort")
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    if not (len(ids) == len(t) == len(e)):
        raise ValueError("patient_ids, times and events must have equal length")
    missing = [p for p in ids if p not in risk_scores]
    if missing:
        raise ValueError(
            f"risk_scores missing for {len(missing)} patients (e.g. {missing[0]}): "
            f"a held-out score cannot be invented for an unseen patient"
        )
    scores = np.array([float(risk_scores[p]) for p in ids], dtype=float)
    bad = [ids[i] for i in np.where(~np.isfinite(scores))[0]]
    if bad:
        raise ValueError(
            f"risk score is NaN or infinite for {len(bad)} patients (e.g. "
            f"{bad[0]}): imputing it would fabricate a prediction"
        )
    if int(e.sum()) < min_events:
        raise ValueError(
            f"only {int(e.sum())} events in the cohort; at least {min_events} "
            f"are needed for a stratified comparison to mean anything"
        )

    train, test = patient_level_split(ids, test_fraction=test_fraction, seed=seed)
    idx = {p: i for i, p in enumerate(ids)}
    test_idx = np.array([idx[p] for p in test], dtype=int)
    t_test, e_test, s_test = t[test_idx], e[test_idx], scores[test_idx]
    if int(e_test.sum()) < 2:
        raise ValueError(
            f"held-out split has {int(e_test.sum())} events: too few to score"
        )

    c = concordance_index(t_test, e_test, s_test)
    hr = _cox_univariate_hr(t_test, e_test, s_test)
    median = float(np.median(s_test))
    group_high = s_test >= median
    lr = logrank_two_group(t_test, e_test, group_high)

    notes = [
        f"all statistics computed on the held-out split only "
        f"({len(test)} of {len(ids)} patients)"
    ]
    scale_check = None
    if expression_sample is not None:
        scale_check = log2_transform_check(expression_sample)
        if not scale_check.plausible_transformed:
            notes.append(scale_check.message)

    if c <= 0.5:
        interpretation = (
            "AT OR BELOW CHANCE on held-out patients: the score orders outcomes "
            "no better than a coin toss, and its hazard ratio must point the "
            "other way"
        )
    elif c < 0.6:
        interpretation = (
            "weakly above chance on held-out patients: real but small "
            "discrimination"
        )
    else:
        interpretation = "clearly ABOVE CHANCE on held-out patients"

    return CohortValidationResult(
        c_index=float(c),
        c_index_interpretation=interpretation,
        hazard_ratio=float(hr),
        logrank_p_value=float(lr["logrank_p_value"]),
        n_patients=len(test),
        n_events=int(e_test.sum()),
        split=(train, test),
        scale_check=scale_check,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Tier 2: IC50 benchmarking against public screens
# ---------------------------------------------------------------------------

MIN_IC50_PAIRS = 8
MIN_IC50_LINES = 5
MIN_IC50_DRUGS = 3


@dataclass(frozen=True)
class IC50Pair:
    """One predicted/measured potency pair, in micromolar."""

    cell_line: str
    drug: str
    predicted_ic50_um: float
    measured_ic50_um: float

    def __post_init__(self) -> None:
        for name in ("predicted_ic50_um", "measured_ic50_um"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0.0:
                raise ValueError(
                    f"{name} must be finite and positive (got {v!r}): IC50 is a "
                    f"concentration on a log scale"
                )
        if not self.cell_line or not self.drug:
            raise ValueError("cell_line and drug must be named")


@dataclass
class IC50ValidationResult:
    r2: float
    spearman_rho: float
    bias_log10: float
    ci95: Tuple[float, float]
    n_pairs: int
    n_distinct_lines: int
    n_distinct_drugs: int
    notes: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if abs(self.bias_log10) > 0.3:
            return (
                f"agreement is high but systematically biased by "
                f"{self.bias_log10:+.2f} log10 units -- check units and assay "
                f"conditions before calling this supportive"
            )
        if self.r2 >= 0.6:
            return "supportive: predicted potency tracks measured potency"
        return "not supportive: predicted potency does not track measurement"


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt(float((a ** 2).sum()) * float((b ** 2).sum()))
    if denom == 0.0:
        raise ValueError("zero variance in one of the vectors: r is undefined")
    return float((a * b).sum() / denom)


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    # average ranks inside tie groups
    sorted_vals = values[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def validate_ic50_predictions(
    pairs: Sequence[IC50Pair],
    *,
    n_boot: int = 500,
    seed: int = 0,
) -> IC50ValidationResult:
    """Compare predicted and measured potency across lines and compounds.

    Refuses sparse, duplicated, or too-narrow inputs rather than reporting a
    correlation computed from a handful of points: a potency benchmark whose
    "drugs" column has one level is a cell-line ranking, not a pharmacology
    validation.
    """
    pairs = list(pairs)
    if len(pairs) < MIN_IC50_PAIRS:
        raise ValueError(
            f"too few pairs ({len(pairs)}): at least {MIN_IC50_PAIRS} predicted/"
            f"measured pairs are needed before a correlation is meaningful"
        )
    seen = set()
    for p in pairs:
        key = (p.cell_line, p.drug)
        if key in seen:
            raise ValueError(
                f"duplicate measurement for {p.cell_line} x {p.drug}: one pair "
                f"per line-compound combination, or the benchmark double-counts"
            )
        seen.add(key)
    lines = {p.cell_line for p in pairs}
    drugs = {p.drug for p in pairs}
    if len(lines) < MIN_IC50_LINES:
        raise ValueError(
            f"only {len(lines)} distinct cell lines: at least {MIN_IC50_LINES} "
            f"are needed so the correlation is not one cell line's idiosyncrasy"
        )
    if len(drugs) < MIN_IC50_DRUGS:
        raise ValueError(
            f"only {len(drugs)} distinct drugs: at least {MIN_IC50_DRUGS} are "
            f"needed so the benchmark tests pharmacology, not one compound"
        )

    pred = np.array([math.log10(p.predicted_ic50_um) for p in pairs])
    meas = np.array([math.log10(p.measured_ic50_um) for p in pairs])
    r = _pearson(pred, meas)
    r2 = float(r ** 2)
    bias = float((pred - meas).mean())
    rho = _pearson(_ranks(pred), _ranks(meas))

    rng = np.random.RandomState(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.randint(0, len(pairs), len(pairs))
        try:
            boots.append(_pearson(pred[pick], meas[pick]) ** 2)
        except ValueError:
            continue
    if boots:
        lo, hi = np.percentile(boots, [2.5, 97.5])
        ci95 = (float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0)))
    else:
        ci95 = (float("nan"), float("nan"))

    notes = [
        f"R^2 and bias are computed in log10 space over {len(pairs)} pairs "
        f"across {len(lines)} lines and {len(drugs)} compounds"
    ]
    if abs(bias) > 0.3:
        notes.append(
            f"systematic bias of {bias:+.3f} log10 units between prediction and "
            f"measurement: R^2 cannot see a constant shift, so this bias log10 "
            f"term is what catches a units error (a 1000x uM-vs-nM mistake "
            f"appears here as +3.0)"
        )
    if ci95[0] < 0.5 <= r2:
        notes.append(
            "the bootstrap 95% interval on R^2 reaches below 0.5, so the "
            "agreement is not resolved by this many pairs"
        )
    return IC50ValidationResult(
        r2=r2,
        spearman_rho=float(rho),
        bias_log10=bias,
        ci95=ci95,
        n_pairs=len(pairs),
        n_distinct_lines=len(lines),
        n_distinct_drugs=len(drugs),
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Tier 3: pre-registered wet-lab plan
# ---------------------------------------------------------------------------

#: Curated caveats for established lines. Every entry must be a documented
#: property of that line, not a generic disclaimer.
CELL_LINE_CAVEATS: Dict[str, str] = {
    "U87-MG": (
        "U87-MG: reported to have drifted from its original isolate, with "
        "lineage markers inconsistent with the originating tumour; state the "
        "passage number and STR profile with any viability result"
    ),
    "LN229": (
        "LN229: p53-mutant background, so apoptosis-based readouts conflate "
        "the drug effect with a p53-independent response"
    ),
    "T98G": (
        "T98G: p53-mutant and highly radioresistant; viability changes may "
        "reflect growth arrest rather than death"
    ),
    "A172": (
        "A172: carries a PTEN deletion; PI3K-pathway readouts are not "
        "generalizable to PTEN-wild-type tumours"
    ),
    "U251": (
        "U251: widely used but genomically distinct from U373, which it was "
        "long conflated with; confirm the identity before comparing to "
        "published results"
    ),
}


@dataclass(frozen=True)
class WetLabPrediction:
    """One prediction that must be declared before the experiment runs."""

    statement: str
    outcome_measure: str
    predicted_value: float
    acceptable_range: Tuple[float, float]
    conditions: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lo, hi = self.acceptable_range
        if not lo < hi:
            raise ValueError(
                f"acceptable_range must be a strict interval (lo < hi), got "
                f"{self.acceptable_range!r}: a zero-width range makes the "
                f"prediction unfalsifiable"
            )
        if not np.isfinite(self.predicted_value):
            raise ValueError("predicted_value must be finite")
        if not self.statement or not self.outcome_measure:
            raise ValueError("statement and outcome_measure must be non-empty")


@dataclass
class WetLabValidationPlan:
    predictions: List[WetLabPrediction]
    cell_lines: List[str]
    caveats: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def falsifiable(self) -> bool:
        return bool(self.predictions) and all(
            p.acceptable_range[0] < p.acceptable_range[1] for p in self.predictions
        )


def design_wetlab_validation(
    predictions: Sequence[WetLabPrediction],
    cell_lines: Optional[Sequence[str]] = None,
    *,
    notes: Optional[Sequence[str]] = None,
) -> WetLabValidationPlan:
    """Assemble a pre-registered plan, with the caveats that bound it."""
    predictions = list(predictions)
    if not predictions:
        raise ValueError(
            "no predictions: a wet-lab plan without pre-registered, falsifiable "
            "predictions cannot validate anything after the fact"
        )
    lines = [str(l) for l in (cell_lines or [])]
    caveats: List[str] = []
    for line in lines:
        caveats.append(
            CELL_LINE_CAVEATS.get(
                line.upper(),
                f"{line}: no curated lineage caveat in this repository -- verify "
                f"identity (STR), mycoplasma status and passage number before "
                f"interpreting any result",
            )
        )
    caveats.append(
        "results outside the pre-declared acceptable_range are reported as "
        "failed predictions, not re-interpreted"
    )
    plan_notes = list(notes or [])
    plan_notes.append(
        "predictions and ranges must be committed before the experiment is run"
    )
    return WetLabValidationPlan(
        predictions=predictions, cell_lines=lines, caveats=caveats, notes=plan_notes
    )


# ---------------------------------------------------------------------------
# Honesty envelope
# ---------------------------------------------------------------------------


def validation_provenance() -> Dict[str, EvidenceStatus]:
    """What each validation tier's output is worth.

    Two entries are `PROVEN_FAILED` on purpose: they are the claims this
    project made and should not make again.
    """
    return {
        # tier 1
        "independent_cohort_survival_validation": EvidenceStatus.ESTABLISHED,
        "heldout_split_integrity": EvidenceStatus.ESTABLISHED,
        "treatment_benefit_from_retrospective_data": EvidenceStatus.PROVEN_FAILED,
        # tier 2
        "gdsc_ctrp_ic50_benchmarking": EvidenceStatus.ESTABLISHED,
        "cross_screen_ic50_comparability": EvidenceStatus.PROVEN_FAILED,
        # tier 3
        "cell_line_fidelity_u87": EvidenceStatus.PROVEN_FAILED,
        "cell_line_predictive_value_for_patients": EvidenceStatus.UNKNOWN,
        # tier 4
        "patient_derived_material": EvidenceStatus.UNKNOWN,
    }


def what_validation_cannot_prove() -> List[str]:
    return [
        "that a schedule extends a human life: retrospective cohorts were not "
        "randomised to the model's recommendation, and no prospective trial of "
        "this engine exists",
        "that a surviving correlation is causal: cohort covariates are "
        "confounded with treatment era, resection extent and molecular subtype",
        "that cross-screen IC50 agreement means the assay would reproduce: two "
        "screens differ in seeding density, medium and readout",
        "that a cell-line result transfers to a patient: established lines "
        "carry lineage drift and pathway backgrounds that primary tumours do not",
        "anything about an individual: every statistic here is a population "
        "summary with a confidence interval, not a prognosis",
    ]
