"""
Predictive measurement: what can be measured about this engine, and what cannot.

`danon.validation` covers the tier ladder (cohort, pharmacology, wet lab).
This module holds the measurements that can be run **now**, offline, against
data already in the repository -- and, just as importantly, the ones that
cannot be run at all.

1. **Held-out discrimination of a clinical baseline -- measurable.** A Cox
   proportional-hazards model is fitted here by Newton-Raphson on the Breslow
   partial likelihood over real TCGA-GBM patients (the cached GDC clinical
   export) with real covariates (age, sex, glioblastoma diagnosis), and scored
   by Harrell's C-index on held-out folds. This validates the *statistical
   machinery* on real patients. It is not a result about the engine.

2. **The engine's patient-level discrimination -- not measurable.** No dataset
   here pairs (baseline tumour state, treatment given, outcome) for the same
   patient, and the engine's arms are deterministic functions of geometry and
   schedule rather than patient covariates: it emits one curve per simulation.
   A C-index over such curves is 0.5 by construction and would carry no
   information. The module returns that verdict with its reason rather than a
   number that looks like validation.

3. **Robustness under input noise -- measurable.** The adaptive policy is
   re-run against paired MTD controls in Gaussian-perturbed worlds (growth,
   transport, intake resistance fraction) and the stability rate is reported
   per noise level.

4. **Physical invariants on real trajectories -- measurable.** Non-negativity,
   carrying capacity, and the discrete reaction-diffusion balance of the
   engine's own concentration field, checked with an independently written
   Laplacian.

NOTHING HERE IS EVIDENCE THAT A PATIENT BENEFITS
------------------------------------------------
Layer 1 shows the statistic works on real patients; layer 3 shows internal
robustness; layer 4 shows the solver keeps its own physics. None of that is
clinical validation. `what_predictive_measurement_cannot_prove()` says so, and
`predictive_provenance()` marks every entry.

References:
  Harrell FE et al. JAMA 1982;247(18):2543-2546 (concordance).
  Cox DR. JRSS-B 1972;34(2):187-202 (partial likelihood, Breslow handling of
  ties).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.evosteer import (
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    CloneParams,
    CollateralSwitchPolicy,
    DrugParams,
    MTDPolicy,
    SimulationTrajectory,
    TumorEvolutionSimulator,
    TumorGeometry,
)
from danon.retrospective import GDC_CLINICAL_PATH
from danon.validation import concordance_index

# ---------------------------------------------------------------------------
# 1. The statistic, wrapped around the single implementation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Concordance:
    """Harrell's C, with the pair counts that make it interpretable."""

    c_index: float
    n: int
    n_events: int

    @property
    def estimable(self) -> bool:
        return np.isfinite(self.c_index)

    def __str__(self) -> str:
        if not self.estimable:
            return f"C-index not estimable (n={self.n})"
        return f"C-index {self.c_index:.3f} ({self.n} patients, {self.n_events} events)"


def harrell_c_index(
    risk: Sequence[float],
    time: Sequence[float],
    event: Sequence[int],
) -> Concordance:
    """Harrell's C, delegating to the one implementation in the repository.

    A second implementation would eventually disagree with the first, and the
    only reason to have one is that `danon.validation`'s version is pinned by
    tests covering the denominator bug it was written to fix.
    """
    n = len(time)
    n_events = int(np.sum(event))
    try:
        c = concordance_index(time, event, risk)
    except ValueError:
        c = float("nan")
    return Concordance(c_index=float(c), n=n, n_events=n_events)


# ---------------------------------------------------------------------------
# 2. Cox proportional hazards, fitted for real
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoxFit:
    coefficients: np.ndarray
    names: Tuple[str, ...]
    log_likelihood: float
    iterations: int
    converged: bool

    def risk(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=float) @ self.coefficients

    @property
    def hazard_ratios(self) -> Dict[str, float]:
        return {n: float(math.exp(b)) for n, b in zip(self.names, self.coefficients)}


def _breslow_objective(
    beta: np.ndarray, X: np.ndarray, time: np.ndarray, event: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Negative Breslow partial log-likelihood with gradient and Hessian.

    Observations are ordered by *ascending* time, so at each event time the
    risk set {j : T_j >= t} is a suffix of the ordered sample. Ordering the
    other way inverts every risk set and the fit converges to the mirror image
    of the truth (measured: coefficients -0.555/+0.164 for a true +0.8/-0.4),
    which is why `tests/test_predictive.py` pins recovery of a known signal
    rather than only checking that the objective is self-consistent.

    Ties in event time are handled the Breslow way.
    """
    eta = X @ beta
    eta = eta - eta.max()
    w = np.exp(eta)
    order = np.argsort(time, kind="stable")
    n, p = X.shape

    ll = 0.0
    grad = np.zeros(p)
    hess = np.zeros((p, p))
    i = 0
    while i < n:
        j = i
        while j < n and time[order[j]] == time[order[i]]:
            j += 1
        block = order[i:j]
        risk_set = order[i:]
        s0 = w[risk_set].sum()
        s1 = (X[risk_set] * w[risk_set, None]).sum(axis=0)
        s2 = (
            X[risk_set][:, :, None] * X[risk_set][:, None, :] * w[risk_set][:, None, None]
        ).sum(axis=0)
        d = int(event[block].sum())
        if d:
            failed = block[event[block] == 1]
            ll += float(eta[failed].sum()) - d * math.log(max(s0, 1e-300))
            mean = s1 / max(s0, 1e-300)
            grad += X[failed].sum(axis=0) - d * mean
            hess -= d * (s2 / max(s0, 1e-300) - np.outer(mean, mean))
        i = j
    return -ll, -grad, -hess


def cox_fit(
    X: np.ndarray,
    time: Sequence[float],
    event: Sequence[int],
    names: Sequence[str],
    *,
    max_iter: int = 60,
    tol: float = 1e-9,
    l2: float = 1e-6,
) -> CoxFit:
    """Fit a Cox PH model by Newton-Raphson with a small ridge for stability."""
    X = np.asarray(X, dtype=float)
    t = np.asarray(time, dtype=float)
    e = np.asarray(event, dtype=int)
    if X.ndim != 2 or X.shape[0] != len(t) or len(t) != len(e):
        raise ValueError("X must be (n_samples, n_features) matching time and event")
    if e.sum() == 0:
        raise ValueError("no events: a Cox model is not identifiable")
    beta = np.zeros(X.shape[1])
    ll_prev = -np.inf
    iterations = 0
    for it in range(1, max_iter + 1):
        iterations = it
        nll, grad, hess = _breslow_objective(beta, X, t, e)
        hess = hess + l2 * np.eye(X.shape[1])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return CoxFit(beta, tuple(names), -nll, it, False)
        beta = beta - step
        nll_new, _, _ = _breslow_objective(beta, X, t, e)
        if not np.isfinite(nll_new):
            return CoxFit(beta, tuple(names), float("nan"), it, False)
        if abs(ll_prev - (-nll_new)) < tol:
            return CoxFit(beta, tuple(names), float(-nll_new), it, True)
        ll_prev = -nll_new
    return CoxFit(beta, tuple(names), float(ll_prev), iterations, False)


# ---------------------------------------------------------------------------
# 3. Real cohort covariates and the held-out check
# ---------------------------------------------------------------------------

GBM_PRIMARY_DIAGNOSIS = "Glioblastoma"


def _last_contact_day(hit: dict, diagnosis: dict) -> Optional[float]:
    """Latest day a patient was known alive, from the nodes that carry it.

    The GDC case model puts the follow-up day in two different places, and a
    patient may populate either:

    * `follow_ups[].days_to_follow_up`     -- present on every one of the 2,482
      follow-up records in the cached TCGA-GBM export.
    * `diagnoses[].days_to_last_follow_up` -- present on the diagnosis node.

    There is **no** `days_to_last_follow_up` on follow-up records, which is
    where an earlier version of this loader looked. Because of that, the
    censoring branch could never fire: all 491 patients with a usable OS
    endpoint were coded as events and the 102 patients recorded as *Alive*
    were dropped. Censoring is not cosmetic -- discarding every censored
    patient removes exactly the patients who survived longest, which inflated
    the reported progression C-index from a corrected 0.53 to 0.59 (see
    `danon/cohort_features.py`, which reports both).
    """
    days: List[float] = []
    for fu in hit.get("follow_ups") or []:
        d = fu.get("days_to_follow_up")
        if d is not None:
            days.append(float(d))
    d = diagnosis.get("days_to_last_follow_up")
    if d is not None:
        days.append(float(d))
    return max(days) if days else None


@dataclass(frozen=True)
class PatientRow:
    submitter_id: str
    age_years: float
    male: int
    is_glioblastoma: int
    time_days: float
    event: int


def load_cohort_rows(
    path: str = GDC_CLINICAL_PATH,
    *,
    endpoint: str = "os",
) -> Tuple[List[PatientRow], List[str]]:
    """Per-patient rows from the cached GDC export, with what was dropped.

    `endpoint` selects overall survival (`os`) or recorded progression.
    Patients with no usable age or no usable endpoint are counted in the notes
    rather than imputed.

    A patient is an event when the record says so and is otherwise censored at
    last contact. That includes `vital_status` values of `Alive`, `None` and
    `Not Reported`: for the latter two the vital status is unknown, and
    censoring at the last recorded follow-up is the standard treatment. If no
    patient is censored the notes carry a warning, because a C-index computed
    only among events is biased upward.
    """
    if endpoint not in ("os", "progression"):
        raise ValueError("endpoint must be 'os' or 'progression'")
    with open(path) as fh:
        payload = json.load(fh)

    rows: List[PatientRow] = []
    notes: List[str] = []
    dropped_age = dropped_endpoint = 0
    dead_without_day = 0

    for h in payload.get("hits", []):
        demo = h.get("demographic") or {}
        dx = (h.get("diagnoses") or [{}])[0]
        age = demo.get("age_at_index", dx.get("age_at_diagnosis"))
        if age is None:
            dropped_age += 1
            continue
        age = float(age)
        if age > 130.0:              # GDC sometimes stores days, not years
            age = age / 365.25
        male = 1 if str(demo.get("sex_at_birth") or "").lower().startswith("male") else 0
        is_gbm = 1 if str(dx.get("primary_diagnosis") or "").startswith(GBM_PRIMARY_DIAGNOSIS) else 0

        time: Optional[float] = None
        event = 0
        if endpoint == "os":
            if str(demo.get("vital_status") or "") == "Dead" and demo.get("days_to_death") is not None:
                time, event = float(demo["days_to_death"]), 1
            else:
                if str(demo.get("vital_status") or "") == "Dead":
                    # Recorded Dead with no `days_to_death`: there is no event
                    # time to use, so the patient is censored at last contact.
                    # Counted rather than silently absorbed.
                    dead_without_day += 1
                time = _last_contact_day(h, dx)
        else:
            # Any recorded progression or recurrence is the event; the earliest
            # such day is taken. `days_to_progression_free` is retained for
            # exports that spell the field that way.
            for fu in h.get("follow_ups") or []:
                for key in ("days_to_progression", "days_to_progression_free", "days_to_recurrence"):
                    d = fu.get(key)
                    if d is not None and float(d) > 0:
                        time = float(d) if time is None else min(time, float(d))
                        event = 1
            if dx.get("days_to_recurrence") is not None and float(dx["days_to_recurrence"]) > 0:
                d = float(dx["days_to_recurrence"])
                time = d if time is None else min(time, d)
                event = 1
            if time is None:
                # No progression recorded: the patient is censored at last
                # contact. Without this branch every patient is an event and
                # the C-index is computed only among progressors.
                time = _last_contact_day(h, dx)
        if time is None or time <= 0:
            dropped_endpoint += 1
            continue
        rows.append(
            PatientRow(
                submitter_id=str(h.get("submitter_id") or h.get("id") or "unknown"),
                age_years=age,
                male=male,
                is_glioblastoma=is_gbm,
                time_days=float(time),
                event=int(event),
            )
        )

    n_events = sum(r.event for r in rows)
    n_censored = len(rows) - n_events
    notes.append(f"{len(rows)} patients with a usable {endpoint} endpoint")
    notes.append(f"{n_events} events, {n_censored} censored at last contact")
    if n_censored == 0:
        notes.append(
            "WARNING: no censored patients -- every patient is coded as an "
            "event, so discrimination is measured only among events and is "
            "biased upward"
        )
    if dead_without_day:
        notes.append(
            f"{dead_without_day} patient(s) recorded Dead with no days_to_death: "
            f"censored at last contact because there is no event time"
        )
    if dropped_age:
        notes.append(f"{dropped_age} dropped: no age recorded")
    if dropped_endpoint:
        notes.append(f"{dropped_endpoint} dropped: no usable time or endpoint")
    return rows, notes


def _design(rows: Sequence[PatientRow]) -> Tuple[np.ndarray, Tuple[str, ...]]:
    X = np.array([[r.age_years, r.male, r.is_glioblastoma] for r in rows], dtype=float)
    return X, ("age_years", "male", "is_glioblastoma")


@dataclass
class HoldoutReport:
    endpoint: str
    n_folds: int
    n_patients: int
    n_events: int
    fold_c_indices: List[float] = field(default_factory=list)
    age_only_c_indices: List[float] = field(default_factory=list)
    mean_multivariable: float = float("nan")
    sd_multivariable: float = float("nan")
    mean_age_only: float = float("nan")
    hazard_ratios: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.mean_multivariable):
            return "not estimable"
        if self.mean_multivariable >= 0.60:
            return (
                "real but weak discrimination on held-out patients, consistent "
                "with the published prognostic value of age and diagnosis in GBM"
            )
        return "no usable held-out discrimination from these covariates"

    def __str__(self) -> str:
        return (
            f"{self.endpoint}: C = {self.mean_multivariable:.3f} +/- "
            f"{self.sd_multivariable:.3f} over {len(self.fold_c_indices)} folds "
            f"(age alone {self.mean_age_only:.3f}) -- {self.verdict}"
        )


def kfold_holdout_c_index(
    rows: Sequence[PatientRow],
    *,
    n_folds: int = 5,
    seed: int = 0,
    endpoint: str = "os",
) -> HoldoutReport:
    """Fit on k-1 folds, score the held-out fold; every patient scored once.

    Fold assignment is a seeded permutation, so the split is reproducible and
    no patient is ever scored by a model that saw them.
    """
    if n_folds < 2:
        raise ValueError("need at least 2 folds")
    if len(rows) < n_folds * 5:
        raise ValueError(f"{len(rows)} patients is too few for {n_folds} folds")
    X, names = _design(rows)
    time = np.array([r.time_days for r in rows], dtype=float)
    event = np.array([r.event for r in rows], dtype=int)

    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(len(rows)), n_folds)

    rep = HoldoutReport(
        endpoint=endpoint, n_folds=n_folds, n_patients=len(rows), n_events=int(event.sum())
    )
    for fold in folds:
        test = np.zeros(len(rows), dtype=bool)
        test[fold] = True
        train = ~test
        if event[train].sum() < 3 or len(fold) < 3:
            rep.notes.append(
                f"fold skipped: only {int(event[train].sum())} training events"
            )
            continue
        fit = cox_fit(X[train], time[train], event[train], names)
        if not fit.converged:
            rep.notes.append("fold skipped: model did not converge")
            continue
        c = harrell_c_index(fit.risk(X[test]), time[test], event[test])
        age_c = harrell_c_index(X[test, 0], time[test], event[test])
        if not c.estimable:
            rep.notes.append("fold skipped: no comparable pairs in the held-out set")
            continue
        rep.fold_c_indices.append(c.c_index)
        rep.age_only_c_indices.append(age_c.c_index)
        rep.hazard_ratios = fit.hazard_ratios

    if rep.fold_c_indices:
        rep.mean_multivariable = float(np.mean(rep.fold_c_indices))
        rep.sd_multivariable = (
            float(np.std(rep.fold_c_indices, ddof=1)) if len(rep.fold_c_indices) > 1 else 0.0
        )
        rep.mean_age_only = float(np.mean(rep.age_only_c_indices))
    rep.notes.append(
        "this validates the statistical pipeline on real patients; it is not a "
        "validation of the treatment engine"
    )
    return rep


def load_clinical_holdout(
    path: str = GDC_CLINICAL_PATH,
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> HoldoutReport:
    """Convenience: real cohort rows in, held-out discrimination out."""
    rows, _ = load_cohort_rows(path)
    return kfold_holdout_c_index(rows, n_folds=n_folds, seed=seed)


# ---------------------------------------------------------------------------
# 4. The engine's own discrimination: the honest verdict
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EngineDiscrimination:
    estimable: bool
    c_index: Optional[float]
    reason: str
    what_would_be_needed: List[str]

    def __str__(self) -> str:
        if self.estimable:
            return f"C-index {self.c_index:.3f}"
        return f"engine patient-level C-index not estimable: {self.reason}"


def engine_discrimination_verdict() -> EngineDiscrimination:
    """Why this engine has no patient-level C-index to report.

    Both conditions are structural rather than a shortage of downloads:

    1. No dataset here pairs (baseline spatial tumour state, treatment actually
       administered, observed outcome) for the same patient. The BraTS imaging
       cases and the TCGA-GBM clinical export anonymise differently, which is
       why `realdata.py` runs a factorial over the two cohorts instead of
       pretending they match.

       The usual objection -- "join them by TCGA barcode" -- was checked and is
       closed: TCIA no longer serves TCGA-GBM or TCGA-LGG (probed; see
       `danon.multimodal.EVIDENCE_PATH`), so the shared identifier route has no
       imaging collection to read from.
    2. The engine emits a **burden trajectory**, not a risk score in clinical
       time. Mapping one to the other requires a calibrated endpoint
       translation (the model's endpoint is a RANO-fraction burden-threshold
       crossing; the clinical endpoint is a radiologist's recorded progression
       date), and no such translation is fitted here.

       An earlier version of this verdict also claimed the engine "emits one
       survival curve per simulation rather than one per patient", which is
       **false and was asserted without checking**. The engine's forecast does
       vary by patient geometry: the three real BraTS geometries in
       `outputs_real/outcome_cards.json` produce MTD-arm day-365 burdens of
       1384, 2244 and 3661 units and median progression times of 365, 293 and
       288 days (re-measured after the extinction/detection floors landed;
       the previous values were 1386, 2259, 3677 and 365, 292, 288, so the
       floors move this quantity by under 1%). So the engine *can* produce a
       patient-varying prediction. What is missing is the cohort to score it
       against, and the endpoint translation above -- not patient-dependence
       in the model.
    """
    return EngineDiscrimination(
        estimable=False,
        c_index=None,
        reason=(
            "no patient-level dataset pairs baseline tumour state, administered "
            "schedule and observed outcome, and the engine emits a burden "
            "trajectory whose endpoint (a RANO-fraction threshold crossing) is "
            "not yet translated into the clinical endpoint (recorded "
            "progression date) that a C-index would require"
        ),
        what_would_be_needed=[
            "a cohort where the imaging geometry, the administered schedule and "
            "the observed progression date all belong to one patient",
            "a per-patient link between a real MRI and a real molecular profile. "
            "The barcode route (TCIA TCGA-GBM imaging joined to the GDC export on "
            "the same TCGA identifier) was the obvious candidate and is gone: "
            "TCIA no longer serves TCGA-GBM or TCGA-LGG. The reachable glioma "
            "collections anonymise separately from GDC, so no current pairing can "
            "be joined at patient level (probed; danon.multimodal records the "
            "counts)",
            "a calibrated endpoint translation from the model's burden-threshold "
            "crossing to the clinical progression date. The engine's forecast "
            "already varies across patients by geometry (verified on the real "
            "geometries), so patient-dependence is not the blocker -- the "
            "unfitted model-to-clinic endpoint mapping is",
        ],
    )


# ---------------------------------------------------------------------------
# 5. Monte Carlo robustness: does the policy survive noisy inputs?
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoiseLevelResult:
    sigma: float
    n_replicates: int
    stable_fraction: float
    median_burden_ratio: float
    median_pfs_gain_days: float
    n_lower_burden_policy: int
    n_lower_burden_mtd: int


@dataclass
class StabilityReport:
    days: int
    n_replicates_per_level: int
    levels: List[NoiseLevelResult] = field(default_factory=list)
    targets: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if not self.levels:
            return "not run"
        worst = max(self.levels, key=lambda l: l.sigma)
        pct = 100 * worst.stable_fraction
        if worst.stable_fraction >= 0.9:
            return f"policy kept a material advantage in {pct:.0f}% of worlds at {100 * worst.sigma:.0f}% noise"
        if worst.stable_fraction >= 0.5:
            return f"policy advantage eroded but held in {pct:.0f}% of worlds at {100 * worst.sigma:.0f}% noise"
        return f"policy advantage did not survive {100 * worst.sigma:.0f}% noise ({pct:.0f}% of worlds)"

    def __str__(self) -> str:
        lines = [f"Monte Carlo stability ({self.days} d horizon):"]
        for l in self.levels:
            lines.append(
                f"  {100 * l.sigma:>3.0f}% noise: stable in "
                f"{100 * l.stable_fraction:>3.0f}% of {l.n_replicates} worlds, "
                f"median burden ratio {l.median_burden_ratio:.3f}, "
                f"median PFS gain {l.median_pfs_gain_days:+.0f} d"
            )
        lines.append(f"  verdict: {self.verdict}")
        return "\n".join(lines)


def _noisy_world(
    rng: np.random.RandomState,
    sigma: float,
    clones: CloneParams,
    drugs: DrugParams,
    r0: float,
) -> Tuple[CloneParams, DrugParams, float]:
    """One Gaussian-perturbed world: growth, transport and intake fraction.

    These are the parameters a patient's data would actually carry uncertainty
    about. Clamped to physically admissible values, and the clamps are part of
    the model's stated behaviour rather than hidden: growth and diffusion stay
    positive, and the intake resistant fraction stays inside (0, 1).
    """
    g = float(np.clip(rng.normal(1.0, sigma), 0.05, 3.0))
    d = float(np.clip(rng.normal(1.0, sigma), 0.05, 3.0))
    r = float(np.clip(r0 * rng.normal(1.0, sigma), 1e-4, 0.9))
    return (
        replace(clones, growth_S=clones.growth_S * g, growth_R=clones.growth_R * g),
        replace(drugs, diffusion=(drugs.diffusion[0] * d, drugs.diffusion[1] * d)),
        r,
    )


def monte_carlo_policy_stability(
    geometry: TumorGeometry,
    *,
    sigmas: Sequence[float] = (0.05, 0.10, 0.20),
    n_replicates: int = 8,
    days: int = 180,
    seed0: int = 0,
    initial_resistant_fraction: float = 0.08,
    clones: CloneParams = DEFAULT_CLONES,
    drugs: DrugParams = DEFAULT_DRUGS,
    stable_ratio: float = 0.5,
) -> StabilityReport:
    """Paired noise test: same perturbed world, both arms, same seed.

    A world counts as *stable* when the adaptive policy ends at a materially
    lower tumour burden than its paired MTD control (< `stable_ratio` x). Burden
    is used rather than PFS because PFS is censored whenever the policy controls
    the tumour for the whole horizon, which would score censoring as a win.
    """
    if n_replicates < 1:
        raise ValueError("need at least one replicate")
    report = StabilityReport(
        days=days,
        n_replicates_per_level=int(n_replicates),
        targets=["tumour_growth", "drug_transport", "intake_r0"],
    )
    for sigma in sigmas:
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        ratios: List[float] = []
        gains: List[float] = []
        n_pol = n_mtd = 0
        for rep in range(n_replicates):
            rng = np.random.RandomState(seed0 + 1000 * int(round(sigma * 100)) + rep)
            c_i, d_i, r_i = _noisy_world(rng, sigma, clones, drugs, initial_resistant_fraction)
            sim = TumorEvolutionSimulator(geometry, clones=c_i, drugs=d_i)
            mtd = sim.run(MTDPolicy(), days=days, initial_resistant_fraction=r_i, seed=rep)
            pol = sim.run(
                CollateralSwitchPolicy(), days=days,
                initial_resistant_fraction=r_i, seed=rep,
            )
            b_mtd = float(mtd.burden_S[-1] + mtd.burden_R[-1])
            b_pol = float(pol.burden_S[-1] + pol.burden_R[-1])
            ratios.append(b_pol / max(b_mtd, 1e-12))
            gains.append(pol.pfs_value(days) - mtd.pfs_value(days))
            n_pol += int(b_pol < b_mtd)
            n_mtd += int(b_mtd <= b_pol)
        report.levels.append(
            NoiseLevelResult(
                sigma=float(sigma),
                n_replicates=int(n_replicates),
                stable_fraction=float(np.mean([r < stable_ratio for r in ratios])),
                median_burden_ratio=float(np.median(ratios)),
                median_pfs_gain_days=float(np.median(gains)),
                n_lower_burden_policy=n_pol,
                n_lower_burden_mtd=n_mtd,
            )
        )
    report.notes.append(
        f"a world is stable when burden_policy < {stable_ratio} x burden_mtd at "
        f"day {days}; that is a model outcome, not a survival benefit"
    )
    report.notes.append(
        "noise is applied to growth, transport and intake r0 only: the "
        "collateral kill mechanism is left at its calibrated value, which is "
        "the assumption the whole result rests on"
    )
    return report


# ---------------------------------------------------------------------------
# 6. Invariants on real trajectories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InvariantViolation:
    invariant: str
    day: Optional[int]
    detail: str
    value: float


def _laplacian(f: np.ndarray) -> np.ndarray:
    """Three-point stencil, written independently of the simulator.

    Deliberately *not* imported from the engine: an invariant check that reuses
    the operator it is checking cannot detect an operator bug.
    """
    lap = np.zeros_like(f, dtype=float)
    lap[1:-1, :, :] += f[:-2, :, :] - 2 * f[1:-1, :, :] + f[2:, :, :]
    lap[:, 1:-1, :] += f[:, :-2, :] - 2 * f[:, 1:-1, :] + f[:, 2:, :]
    lap[:, :, 1:-1] += f[:, :, :-2] - 2 * f[:, :, 1:-1] + f[:, :, 2:]
    return lap


def check_invariants(
    traj: SimulationTrajectory,
    geometry: TumorGeometry,
    *,
    clones: CloneParams = DEFAULT_CLONES,
    drugs: DrugParams = DEFAULT_DRUGS,
    tolerance: float = 1e-6,
) -> List[InvariantViolation]:
    """Physical invariants over one real run.

    Four families, each falsifiable and each checked against the engine's own
    recorded output:

    * **non-negativity** -- densities, doses and concentration fields cannot go
      negative, and must be finite;
    * **carrying capacity** -- total density cannot exceed K per tumour voxel
      (`K * n_tumour_voxels` in total);
    * **PDE-operator consistency** -- the final concentration field must satisfy
      a discrete reaction-diffusion balance with the engine's own transport
      constants, `D |lap C| <= source + clearance * C_max`;
    * **trajectory shape** -- series lengths match the horizon and `pfs_day`
      lies inside it.
    """
    out: List[InvariantViolation] = []

    # Shape first: a ragged trajectory makes every cross-series comparison
    # below ill-defined, and crashing on it would report a checker failure
    # instead of the violation it is.
    series = {name: np.asarray(getattr(traj, name), dtype=float) for name in ("burden_S", "burden_R", "dose_A", "dose_B")}
    lengths = {name: len(arr) for name, arr in series.items()}
    reference = lengths["burden_S"]
    ragged = {name: n for name, n in lengths.items() if n != reference}
    if ragged:
        out.append(
            InvariantViolation(
                "trajectory_shape",
                None,
                f"series lengths disagree: burden_S has {reference}, "
                f"{ragged}",
                float(reference),
            )
        )

    for name, arr in series.items():
        if np.any(arr < -tolerance):
            idx = int(np.argmin(arr))
            out.append(
                InvariantViolation(
                    "non_negativity", idx, f"{name} went negative", float(arr[idx])
                )
            )
        if not np.all(np.isfinite(arr)):
            out.append(
                InvariantViolation(
                    "finiteness", None, f"{name} contains non-finite values", float("nan")
                )
            )

    cap_total = clones.carrying_capacity * float(geometry.tumor_mask.sum())
    if ragged:
        # the sums below would broadcast silently against the wrong days
        return out
    total = series["burden_S"] + series["burden_R"]
    over = np.where(total > cap_total * (1.0 + 1e-3))[0]
    if len(over):
        day = int(over[0])
        out.append(
            InvariantViolation(
                "carrying_capacity",
                day,
                f"total density {total[day]:.6g} exceeded K x tumour voxels "
                f"= {cap_total:.6g}",
                float(total[day] / cap_total),
            )
        )

    for field_name, (D, k) in (
        ("final_conc_A", (drugs.diffusion[0], drugs.clearance[0])),
        ("final_conc_B", (drugs.diffusion[1], drugs.clearance[1])),
    ):
        field = getattr(traj, field_name, None)
        if field is None:
            continue
        C = np.asarray(field, dtype=float)
        if np.any(C < -tolerance):
            out.append(
                InvariantViolation(
                    "non_negativity", None,
                    f"{field_name} has negative concentration", float(C.min()),
                )
            )
        bound = drugs.source_strength + k * float(C.max()) + tolerance
        worst = float((D * np.abs(_laplacian(C))).max())
        if worst > bound * (1.0 + 1e-3):
            out.append(
                InvariantViolation(
                    "pde_operator_consistency",
                    None,
                    f"{field_name}: max |D*lap(C)| = {worst:.4g} exceeds source + "
                    f"clearance*C_max = {bound:.4g}",
                    worst / bound,
                )
            )

    if reference != traj.days:
        out.append(
            InvariantViolation(
                "trajectory_shape", None,
                f"burden_S has {reference} entries, days={traj.days}",
                float(reference),
            )
        )
    if traj.pfs_day is not None and not (0 <= traj.pfs_day < traj.days):
        out.append(
            InvariantViolation(
                "trajectory_shape", int(traj.pfs_day),
                "pfs_day outside the simulated horizon", float(traj.pfs_day),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Honesty envelope
# ---------------------------------------------------------------------------


def predictive_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "c_index_implementation": EvidenceStatus.ESTABLISHED,
        "cox_partial_likelihood_fit": EvidenceStatus.ESTABLISHED,
        "real_patient_covariates": EvidenceStatus.ESTABLISHED,
        "heldout_discrimination_of_clinical_baseline": EvidenceStatus.EMERGING_HUMAN,
        "engine_patient_level_discrimination": EvidenceStatus.UNKNOWN,
        "policy_stability_under_noise": EvidenceStatus.UNKNOWN,
        "invariants_on_simulated_trajectories": EvidenceStatus.ESTABLISHED,
        "clinical_benefit": EvidenceStatus.UNKNOWN,
    }


def what_predictive_measurement_cannot_prove() -> List[str]:
    return [
        "that the engine predicts any individual patient's course: no paired "
        "geometry/schedule/outcome dataset exists here, and the model emits one "
        "curve per simulation",
        "that the noise test bounds real biological uncertainty: the "
        "perturbations are Gaussian and parameter-local, so model-form error is "
        "invisible to it",
        "that the invariants make the model right: a model can conserve mass, "
        "stay non-negative and satisfy its own operator while being wrong about "
        "the biology",
        "that a C-index on clinical covariates transfers to the engine: it "
        "validates the statistic on real patients and nothing else",
        "that any schedule here extends a human life: prospective trials do that",
    ]
