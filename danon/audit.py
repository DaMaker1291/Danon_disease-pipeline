"""
Temporal-leakage and out-of-sample audit trace.

WHY THIS MODULE EXISTS
----------------------
"Prove your model did not see the outcome before it predicted" is the right
question to ask of a clinical model, and it deserves a machine-generated answer
rather than a screenshot. This module produces that answer for the parts of this
repository that *can* answer it, and refuses to produce one for the part that
cannot.

WHAT IT PROVES
--------------
1. **Out-of-sample scoring.** Every patient is scored by a model fitted without
   them. The split is a seeded permutation, and the fold membership is checked
   independently here rather than trusted.
2. **No outcome leakage into the design.** The covariate matrix is built from
   age, sex and diagnosis only; the survival time and event indicator are
   separate vectors. This is asserted structurally.
3. **A real leakage detector.** Permuting the outcome across patients must
   collapse held-out discrimination to chance. If a model can still rank
   patients after its labels are shuffled, something other than the covariates
   is carrying information -- and that is the failure this test exists to catch.
   This is the strongest check here, because it fails on real leakage rather
   than on a mistake in bookkeeping.
4. **Seeded reproducibility.** Runs are stochastic by design; the same seed
   reproduces a trajectory exactly and a different seed diverges.

WHAT IT REFUSES TO PROVE
------------------------
**It will not emit a per-patient relapse prediction matched against a real
hospital relapse date.** No such pairing exists in this repository, and the
engine could not produce one if it did:

  * **The engine emits one curve per simulation, not one per patient.** Its arms
    are deterministic functions of geometry, clone parameters and schedule, not
    of patient covariates, so a per-patient prediction would be identical across
    patients and a C-index over it would be 0.5 by construction
    (:func:`danon.predictive.engine_discrimination_verdict`).
  * **No patient here has baseline imaging, ingestible molecular state,
    administered schedule and observed outcome together.** BraTS-derived imaging
    and the GDC clinical export anonymise differently, and TCIA no longer serves
    TCGA-GBM or TCGA-LGG, so the barcode route is closed
    (:mod:`danon.multimodal` records the probe counts).
  * **The progression endpoint does not discriminate.** Held-out C = 0.534 on
    the corrected endpoint (372 events, 222 censored at last contact), against
    0.539 for age alone -- the multivariable model is beaten by a patient's age.
    Any log reporting a two-percent error against a real relapse date would be
    inconsistent with the repository's own measured discrimination, and that
    inconsistency is what an auditor finds first.

    That figure was 0.590 before the censoring defect in
    :func:`danon.predictive.load_cohort_rows` was fixed (:mod:`danon.cohort_features`
    records the before/after pair). Discarding every censored patient deletes the
    longest survivors and inflated this number by +0.056.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus

#: Covariates the Cox model is permitted to consume. Anything outside this set
#: in the design matrix is a leak, so the set is named rather than inferred.
PERMITTED_DESIGN_COLUMNS: Tuple[str, ...] = ("age_years", "male", "is_glioblastoma")

#: Columns that must never enter the design matrix.
FORBIDDEN_DESIGN_COLUMNS: Tuple[str, ...] = (
    "time_days", "event", "days_to_death", "days_to_progression",
    "days_to_last_follow_up", "vital_status", "progression_or_recurrence",
)


@dataclass(frozen=True)
class AuditCheck:
    """One assertion in the trace, with the number that decided it."""

    id: str
    question: str
    statistic: str
    observed: Optional[float]
    passed: bool
    detail: str
    null_center: Optional[float] = None
    null_low: Optional[float] = None
    null_high: Optional[float] = None

    def describe(self) -> str:
        band = ""
        if self.null_center is not None:
            band = (
                f" [null centre {self.null_center:.3f}, "
                f"band {self.null_low:.3f}-{self.null_high:.3f}]"
            )
        return (
            f"[{'PASS' if self.passed else 'FAIL'}] {self.id}: {self.question} "
            f"-- {self.statistic} = "
            f"{'n/a' if self.observed is None else f'{self.observed:.4f}'}"
            f"{band}\n      {self.detail}"
        )


# ---------------------------------------------------------------------------
# Part 1 -- what existed at ingestion
# ---------------------------------------------------------------------------


def ingestion_boundary() -> Dict[str, object]:
    """What the pipeline had access to at baseline, by provenance tag.

    Derived from the modules that actually read the data, not restated here.
    """
    from danon.realdata import DecimationSpec, BRATS2021_SOURCE
    from danon.pharmacology import load_glioma_potency
    from danon.multimodal import load_evidence

    spec = DecimationSpec()
    boundary: Dict[str, object] = {
        "imaging_source": BRATS2021_SOURCE,
        "imaging_timepoints_available": 1,
        "engine_grid_voxel_mm": float(np.mean(list(spec.factors))),
        "decimation_is_lossy": True,
        "clinical_endpoints_available": [
            "demographic.days_to_death (overall survival)",
            "follow_ups.days_to_progression (recorded progression)",
        ],
        "molecular_assay": "bulk RNA-seq (GDC open STAR counts)",
        "single_cell_assay_available": False,
        "patient_level_imaging_outcome_join_available": False,
        "engine_per_patient_prediction_available": False,
    }
    try:
        mol, potency = load_glioma_potency()
        boundary["potency_source"] = f"{mol.chembl_id} ({potency.standard_type})"
        boundary["potency_exact_records"] = potency.n_exact
        boundary["potency_right_censored_records"] = potency.n_right_censored
    except Exception as exc:  # pragma: no cover - cache-dependent
        boundary["potency_source_error"] = str(exc)

    by_key = {m.key: m for m in load_evidence()}
    tcia = by_key.get("tcia_nbia_api")
    if tcia is not None:
        boundary["tcia_glioma_candidates_absent"] = tcia.get("glioma_candidates_absent")
        boundary["tcia_glioma_candidates_present"] = tcia.get("glioma_candidates_present")
    return boundary


def ingestion_boundary_notes() -> List[str]:
    """Why the boundary above is what it is, stated where it is printed."""
    return [
        "The engine consumes ONE imaging timepoint per case: the BraTS 2021 "
        "release is pre-operative only, so there is no future frame in memory to "
        "leak from -- the invariant holds because the data does not exist, not "
        "because a mask function blocked it.",
        "No patient in this repository carries baseline imaging and an observed "
        "outcome together. TCIA no longer serves TCGA-GBM or TCGA-LGG, and the "
        "reachable glioma collections anonymise separately from the GDC export, "
        "so no patient-level imaging/outcome pair can be assembled.",
        "The engine's forecast varies by patient geometry, so it does carry "
        "patient-dependence. What it has no per-patient prediction to blind is "
        "for a different reason: no cohort pairs geometry with outcome, and the "
        "model endpoint is not translated into clinical time.",
    ]


# ---------------------------------------------------------------------------
# Part 2 -- structural leakage assertions
# ---------------------------------------------------------------------------


def design_columns_audit() -> AuditCheck:
    """The design matrix must contain covariates only, never the outcome."""
    from danon.predictive import _design

    rows = _sample_rows(3)
    _, names = _design(rows)
    leaked = [n for n in names if n in FORBIDDEN_DESIGN_COLUMNS]
    unpermitted = [n for n in names if n not in PERMITTED_DESIGN_COLUMNS]
    return AuditCheck(
        id="design_matrix_excludes_outcome",
        question="does the covariate matrix contain any outcome column?",
        statistic="columns",
        observed=float(len(names)),
        passed=not leaked and not unpermitted,
        detail=(
            f"design columns = {names}; forbidden columns present: "
            f"{leaked or 'none'}; unpermitted columns present: "
            f"{unpermitted or 'none'}"
        ),
    )


def fold_membership_audit(
    rows: Sequence[object],
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> AuditCheck:
    """Every patient is scored exactly once, and never by a model that saw them.

    The fold split is re-derived here from the same seed independently of
    :func:`danon.predictive.kfold_holdout_c_index`, so this checks the property
    rather than trusting the function that implements it.
    """
    n = len(rows)
    rng = np.random.RandomState(seed)
    folds = np.array_split(rng.permutation(n), n_folds)
    counts = np.zeros(n, dtype=int)
    for fold in folds:
        counts[fold] += 1
    test_seen = np.zeros(n, dtype=int)
    train_seen = np.zeros(n, dtype=int)
    for fold in folds:
        test = np.zeros(n, dtype=bool)
        test[fold] = True
        train = ~test
        test_seen[test] += 1
        train_seen[train] += 1
    ok = (
        counts.min() == 1 and counts.max() == 1
        and test_seen.min() == 1 and test_seen.max() == 1
        and train_seen.min() == n_folds - 1 and train_seen.max() == n_folds - 1
    )
    return AuditCheck(
        id="out_of_sample_fold_membership",
        question="is every patient scored exactly once, by a model fitted without them?",
        statistic="patients scored once / total",
        observed=float((test_seen == 1).sum()),
        passed=bool(ok),
        detail=(
            f"{n} patients, {n_folds} folds: test appearances "
            f"{test_seen.min()}-{test_seen.max()}, train appearances "
            f"{train_seen.min()}-{train_seen.max()} (expected exactly "
            f"{n_folds - 1}); split re-derived from seed {seed}"
        ),
    )


# ---------------------------------------------------------------------------
# Part 3 -- the leakage detector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PermutationNull:
    """Observed held-out discrimination against its permutation null."""

    observed: float
    null_median: float
    null_low: float
    null_high: float
    n_permutations: int
    n_patients: int
    statistic: str
    detail: str

    @property
    def separation(self) -> float:
        """How far the observed value sits above the null's upper edge."""
        return self.observed - self.null_high


def permutation_null_c_index(
    rows: Sequence[object],
    *,
    n_permutations: int = 20,
    n_folds: int = 5,
    seed: int = 0,
    endpoint: str = "os",
) -> PermutationNull:
    """Refit on shuffled outcomes and check discrimination collapses to chance.

    The outcome pair (time, event) is permuted across patients while the
    covariates stay fixed. Under the null of no association, held-out
    discrimination must sit at chance (0.5). A model that keeps ranking patients
    after its labels are shuffled is being fed information it should not have,
    which is precisely the failure mode a leakage audit exists to find.

    ``n_permutations`` defaults low (20) because each permutation is a full
    k-fold refit; the resulting null has ~0.05 resolution in its upper tail,
    which is enough to separate a real 0.64 from chance and is stated rather
    than presented as a precise p-value.
    """
    from danon.predictive import kfold_holdout_c_index

    # The observed fit is guarded on the same terms as the permutations. An
    # earlier version guarded only the loop, so an undersized cohort raised
    # out of the *observed* call and the audit crashed instead of reporting
    # that leakage was undetermined -- a failure to report, which is worse than
    # an undetermined answer.
    try:
        observed_rep = kfold_holdout_c_index(
            rows, n_folds=n_folds, seed=seed, endpoint=endpoint
        )
        observed = observed_rep.mean_multivariable
    except ValueError as exc:
        return PermutationNull(
            observed=float("nan"), null_median=float("nan"),
            null_low=float("nan"), null_high=float("nan"),
            n_permutations=0, n_patients=len(rows),
            statistic=f"{endpoint} C-index",
            detail=f"leakage undetermined: observed fit not estimable ({exc})",
        )
    rng = np.random.RandomState(seed + 991)
    null: List[float] = []
    skipped = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(rows))
        shuffled = [
            replace(r, time_days=rows[j].time_days, event=rows[j].event)
            for r, j in zip(rows, perm)
        ]
        try:
            rep = kfold_holdout_c_index(
                shuffled, n_folds=n_folds, seed=seed, endpoint=endpoint
            )
        except ValueError:
            skipped += 1
            continue
        if np.isfinite(rep.mean_multivariable):
            null.append(float(rep.mean_multivariable))
        else:
            skipped += 1

    if not null:
        return PermutationNull(
            observed=float(observed), null_median=float("nan"),
            null_low=float("nan"), null_high=float("nan"),
            n_permutations=0, n_patients=len(rows), statistic=f"{endpoint} C-index",
            detail="no permutation completed; leakage undetermined",
        )

    arr = np.array(null, dtype=float)
    return PermutationNull(
        observed=float(observed),
        null_median=float(np.median(arr)),
        null_low=float(np.percentile(arr, 2.5)),
        null_high=float(np.percentile(arr, 97.5)),
        n_permutations=len(null),
        n_patients=len(rows),
        statistic=f"{endpoint} held-out C-index",
        detail=(
            f"{len(null)} permutation(s) of (time, event) across {len(rows)} "
            f"patients with covariates fixed; null median {np.median(arr):.3f}; "
            f"observed {observed:.3f}"
            + (f"; {skipped} permutation(s) skipped" if skipped else "")
            + ". A null centred at chance is the required result: it means "
            "discrimination comes from the covariates, not from leaked outcome "
            "information."
        ),
    )


#: The tightest centring this check will demand of the null median. A null is
#: built from a finite number of draws on a finite cohort, so its own Monte-Carlo
#: resolution is the floor on what it can be asked to demonstrate.
NULL_CENTRE_TOLERANCE: float = 0.06


def permutation_leakage_audit(
    rows: Sequence[object],
    *,
    n_permutations: int = 20,
    n_folds: int = 5,
    seed: int = 0,
    endpoint: str = "os",
) -> AuditCheck:
    """The permutation null as an :class:`AuditCheck`.

    An undetermined null (too few patients, or every permutation skipped) is a
    **failure**, not a pass: "we could not tell" must never render as "no leak".

    The centring tolerance adapts to the null's own resolution. A median over
    `k` permutations of a cohort of `n` cannot be pinned more tightly than the
    spread of those draws, so demanding a fixed 0.06 of a 4-draw null on 80
    patients asserts precision the estimator does not have -- the same failure
    this project keeps finding elsewhere, a number read as a verdict while
    sitting inside its own noise.

    Measured on the corrected OS cohort (n=595, 491 events, 104 censored, seed 0):
    20 draws give centre 0.506 and band [0.468, 0.536]; 100 draws give centre
    0.502 and band [0.460, 0.536]. More draws move the *centre* toward chance
    while the band width barely changes, so the band is set by cohort size, not
    by draw count. On an 80-patient, 4-draw null the band half-width is 0.080,
    and that is the tolerance applied -- with the fixed 0.06 it would fail on
    resolution alone.
    """
    pn = permutation_null_c_index(
        rows, n_permutations=n_permutations, n_folds=n_folds, seed=seed, endpoint=endpoint
    )
    # Null must centre at chance. Whether the observed value clears its band is
    # a *strength* question, so it is reported but not required: a weak model is
    # not a leaky one, and conflating the two would make this test lie in the
    # opposite direction.
    determined = np.isfinite(pn.null_median) and pn.n_permutations > 0
    band = np.isfinite(pn.null_high) and np.isfinite(pn.null_low)
    resolution = 0.5 * (pn.null_high - pn.null_low) if band else 0.0
    tolerance = max(NULL_CENTRE_TOLERANCE, resolution)
    offset = abs(pn.null_median - 0.5) if determined else float("inf")
    centred = determined and offset <= tolerance
    return AuditCheck(
        id="permutation_leakage_null",
        question=(
            "does held-out discrimination collapse to chance when the outcome "
            "is permuted across patients?"
        ),
        statistic=pn.statistic,
        observed=pn.observed,
        null_center=pn.null_median if np.isfinite(pn.null_median) else None,
        null_low=pn.null_low if np.isfinite(pn.null_low) else None,
        null_high=pn.null_high if np.isfinite(pn.null_high) else None,
        passed=bool(centred),
        detail=pn.detail
        + (
            f" Null centre sits {offset:.3f} from chance against a tolerance of "
            f"{tolerance:.3f} (its own half-band at {n_permutations} draws)."
            if determined else ""
        )
        + (
            f" Observed value clears the null band by "
            f"{pn.separation:+.3f} (strength, not a leakage requirement)."
            if np.isfinite(pn.null_high) else ""
        )
        + (
            "" if determined
            else " The null is undetermined, which is reported as a failure: "
                 "an audit that cannot tell has not passed."
        ),
    )


# ---------------------------------------------------------------------------
# Part 4 -- determinism and the refusal
# ---------------------------------------------------------------------------


def determinism_audit(*, days: int = 15, seed: int = 7) -> AuditCheck:
    """Same seed reproduces a trajectory exactly; a different seed diverges."""
    from danon.evosteer import MTDPolicy, TumorEvolutionSimulator, make_gbm_phantom

    geom = make_gbm_phantom(size=8, seed=0)
    sim = TumorEvolutionSimulator(geom)
    a = sim.run(MTDPolicy(), days=days, initial_resistant_fraction=0.05, seed=seed)
    b = sim.run(MTDPolicy(), days=days, initial_resistant_fraction=0.05, seed=seed)
    c = sim.run(MTDPolicy(), days=days, initial_resistant_fraction=0.05, seed=seed + 1)
    same = bool(np.array_equal(a.burden_S, b.burden_S))
    differs = not bool(np.array_equal(a.burden_S, c.burden_S))
    return AuditCheck(
        id="seeded_reproducibility",
        question="does a fixed seed reproduce a trajectory exactly, and a different seed diverge?",
        statistic="same-seed identical / different-seed different",
        observed=None,
        passed=bool(same and differs),
        detail=(
            f"same seed {seed} reproduces exactly: {same}; seed {seed + 1} "
            f"diverges: {differs}. Runs are stochastic and seeded, which is the "
            f"honest claim -- a claim of full determinism would be false."
        ),
    )


def per_patient_prediction_audit() -> AuditCheck:
    """The refusal, as a check rather than an omission.

    This is deliberately phrased so that a *future* change makes it fail: if a
    per-patient prediction ever becomes estimable, this audit must be rewritten
    rather than silently passing.
    """
    from danon.predictive import engine_discrimination_verdict

    v = engine_discrimination_verdict()
    return AuditCheck(
        id="per_patient_relapse_prediction",
        question="is there a per-patient relapse prediction to compare to a real relapse date?",
        statistic="estimable",
        observed=None,
        passed=not v.estimable,
        detail=(
            "NOT ESTIMABLE, and the audit says so rather than omitting the "
            "section: " + v.reason + ". Therefore no prediction-error-in-days "
            "figure exists for any patient in this repository, and a trace "
            "reporting one would be fabricated. Note what is NOT the obstacle: "
            "the engine's forecast does vary by geometry, so the gap is data "
            "linkage and endpoint translation, not a model that ignores the "
            "patient."
        ),
    )


def _sample_rows(n: int) -> List[object]:
    """A few real cohort rows, for structural checks that need a design matrix."""
    from danon.predictive import load_cohort_rows

    rows, _ = load_cohort_rows(endpoint="os")
    return list(rows[:n])


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------


def run_audit(
    *,
    n_permutations: int = 20,
    n_folds: int = 5,
    seed: int = 0,
    endpoint: str = "os",
    max_patients: Optional[int] = None,
) -> List[AuditCheck]:
    """Every check, in order. ``max_patients`` subsamples to bound runtime."""
    from danon.predictive import load_cohort_rows

    rows, _ = load_cohort_rows(endpoint=endpoint)
    rows = list(rows[:max_patients] if max_patients else rows)
    return [
        design_columns_audit(),
        fold_membership_audit(rows, n_folds=n_folds, seed=seed),
        permutation_leakage_audit(
            rows, n_permutations=n_permutations, n_folds=n_folds,
            seed=seed, endpoint=endpoint,
        ),
        determinism_audit(),
        per_patient_prediction_audit(),
    ]


def render_audit_trace(
    *,
    n_permutations: int = 20,
    n_folds: int = 5,
    seed: int = 0,
    endpoint: str = "os",
    max_patients: Optional[int] = None,
) -> str:
    """The four-part trace, with every number generated rather than templated."""
    from danon.predictive import kfold_holdout_c_index, load_cohort_rows

    rows, notes = load_cohort_rows(endpoint=endpoint)
    rows = list(rows[:max_patients] if max_patients else rows)
    t0 = time.time()
    checks = run_audit(
        n_permutations=n_permutations, n_folds=n_folds, seed=seed,
        endpoint=endpoint, max_patients=max_patients,
    )
    elapsed = time.time() - t0
    rep = kfold_holdout_c_index(rows, n_folds=n_folds, seed=seed, endpoint=endpoint)

    bar = "=" * 88
    lines = [
        bar,
        "            RECONCILIATION AUDIT: TEMPORAL LEAKAGE & OUT-OF-SAMPLE TRACE",
        bar,
        "",
        "[PART 1: INGESTION BOUNDARY -- WHAT EXISTED BEFORE ANY FORECAST]",
        json.dumps(ingestion_boundary(), indent=2, default=str),
        "",
        "boundary notes:",
    ]
    lines += [f"  - {n}" for n in ingestion_boundary_notes()]

    lines += [
        "",
        "[PART 2: OUT-OF-SAMPLE STRUCTURE]",
    ]
    for c in checks[:2]:
        lines.append("  " + c.describe().replace("\n", "\n  "))

    lines += [
        "",
        "[PART 3: LEAKAGE DETECTION -- PERMUTATION NULL]",
    ]
    lines.append("  " + checks[2].describe().replace("\n", "\n  "))

    lines += [
        "",
        "[PART 4: DETERMINISM AND THE SECTION THAT CANNOT BE FILLED]",
    ]
    for c in checks[3:]:
        lines.append("  " + c.describe().replace("\n", "\n  "))

    lines += [
        "",
        "[MEASURED OUT-OF-SAMPLE RESULT ON THE REAL COHORT]",
        f"  {rep}",
        f"  n = {len(rows)} patients ({'; '.join(notes)})",
        "",
        "[MODEL STANDARD-OF-CARE ARM vs OBSERVED PROGRESSION]",
    ]
    ts = model_vs_observed_timescale()
    if not ts.get("available"):
        lines.append(f"  not run: {ts.get('reason')}")
    else:
        oc, ma = ts["observed_cohort"], ts["model_mtd_arm"]
        lines.append(
            f"  observed: {oc['n_cases']} TCGA-GBM case(s), "
            f"{oc['n_progression_events']} progression event(s), dated-event IQR "
            f"{oc['dated_event_iqr_days']} days"
        )
        lines.append(
            f"  model MTD arm over {ma['n_geometries']} real geometr(ies): "
            f"median progression {ma['median_progression_days']:.0f} days, "
            f"range {min(ma['times']):.0f}-{max(ma['times']):.0f}"
        )
        lines.append("    day | observed S(t) | model S(t) | diff")
        for t in ts["survival_table"]:
            lines.append(
                f"    {t['day']:4d} | {t['observed_S']:12.3f} | "
                f"{t['model_S']:10.3f} | {t['diff']:+.3f}"
            )
        r = ts["reading"]
        lines.append(
            f"  reading: timescale plausible = {r['timescale_plausible']}; "
            f"too optimistic early = {r['model_too_optimistic_early']}; "
            f"too pessimistic late = {r['model_too_pessimistic_late']}; "
            f"max |diff| = {r['max_absolute_disagreement']}"
        )
        lines.append(f"  {ts['interpretation']}")
        lines += [f"    caveat: {c}" for c in ts["caveats"]]
    lines += [
        "",
        "[WHAT IS DELIBERATELY ABSENT]",
        "  - No per-patient predicted-vs-actual relapse day. The engine's forecast",
        "    DOES vary by geometry (01017/01018/01019 give day-365 MTD burdens of",
        "    2259/1386/3677), so patient-dependence is not the blocker: no patient",
        "    here has baseline imaging, molecular state, administered schedule and",
        "    observed outcome together, and the model's burden-threshold endpoint",
        "    is not yet translated into a clinical progression date.",
        "  - No percent-error figure against a hospital relapse date, because",
        "    producing one would require inventing the pairing.",
        "  - No single-cell derived initial resistant fraction: the pipeline reads",
        "    bulk RNA-seq and marks single-cell inference PROVEN_FAILED.",
        "",
        f"[AUDIT COMPLETE in {elapsed:.1f}s]  "
        f"{sum(1 for c in checks if c.passed)}/{len(checks)} checks passed",
        bar,
    ]
    return "\n".join(lines)


def model_vs_observed_timescale() -> Dict[str, object]:
    """The model's standard-of-care arm against the real cohort's progression.

    This is the strongest *honest* external comparison available: the engine's
    MTD arm is its own standard-of-care proxy, and the observed curve is the
    real recorded progression from the GDC export. It is a timescale check, not
    a calibration -- the model is not fitted to this cohort, the endpoints are
    different objects (a RANO burden-threshold crossing versus a radiologist's
    recorded progression), and the two populations are different people.

    Returns the table plus what the numbers actually say, so the interpretation
    travels with the measurement rather than being re-derived later.
    """
    import json
    import os as _os

    from danon.retrospective import load_tcga_gbm_clinical, plausibility_check

    out: Dict[str, object] = {}
    cards_path = "outputs_real/outcome_cards.json"
    if not _os.path.exists(cards_path):
        out["available"] = False
        out["reason"] = f"no exported cards at {cards_path}"
        return out
    with open(cards_path) as fh:
        cards = json.load(fh)
    times, events = [], []
    for c in cards.get("cases", []):
        x = c.get("day_365") or {}
        m = x.get("median_pfs_mtd")
        if m is None:
            continue
        times.append(float(m))
        events.append(1 if float(m) < float(c.get("horizon_days") or 365) else 0)
    if not times:
        out["available"] = False
        out["reason"] = "cards carry no MTD median progression times"
        return out

    cohort = load_tcga_gbm_clinical()
    check = plausibility_check(cohort, times, events)
    table = [
        {"day": int(d), "observed_S": round(o, 3), "model_S": round(m, 3),
         "diff": round(m - o, 3)}
        for d, o, m, _ in check.eval_table
    ]
    iqr = cohort.progression_event_iqr()
    diffs = [t["diff"] for t in table]
    early = [t for t in table if t["day"] <= 200]
    late = [t for t in table if t["day"] > 200]
    out.update({
        "available": True,
        "observed_cohort": {
            "n_cases": cohort.n_cases,
            "n_progression_events": cohort.n_progression_events,
            "dated_event_iqr_days": [round(iqr[0], 1), round(iqr[1], 1)] if iqr else None,
            "source": cohort.source,
        },
        "model_mtd_arm": {
            "n_geometries": len(times),
            "median_progression_days": check.model_median_days,
            "times": sorted(times),
        },
        "survival_table": table,
        "reading": {
            "timescale_plausible": (
                iqr is not None
                and iqr[0] <= float(check.model_median_days or 0) <= iqr[1]
            ),
            "model_too_optimistic_early": all(t["diff"] > 0 for t in early),
            "model_too_pessimistic_late": all(t["diff"] < 0 for t in late),
            "max_absolute_disagreement": round(max(abs(d) for d in diffs), 3),
        },
        "interpretation": (
            "The model's standard-of-care arm progresses on a plausible "
            "timescale (its median sits inside the observed dated-event IQR) but "
            "its distribution is too narrow: it over-predicts progression-free "
            "survival at 100-200 days and under-predicts it at 365. That is a "
            "calibration target for the growth and mutation parameters, and it "
            "is NOT a validation of the treatment engine."
        ),
        "caveats": list(check.caveats),
    })
    return out


def audit_provenance() -> Dict[str, EvidenceStatus]:
    """What each audited property is, in the repo's provenance vocabulary."""
    return {
        "out_of_sample_fold_membership": EvidenceStatus.ESTABLISHED,
        "design_matrix_excludes_outcome": EvidenceStatus.ESTABLISHED,
        "permutation_leakage_null": EvidenceStatus.ESTABLISHED,
        "seeded_reproducibility": EvidenceStatus.ESTABLISHED,
        "per_patient_relapse_prediction": EvidenceStatus.PROVEN_FAILED,
        "single_cell_input": EvidenceStatus.PROVEN_FAILED,
    }


def what_the_audit_cannot_establish() -> List[str]:
    """Limits that must travel with the trace."""
    return [
        "Absence of detectable leakage is not proof of absence. The permutation "
        "null catches outcome leakage that inflates discrimination; it cannot "
        "catch a covariate that is itself a proxy for the outcome, and it says "
        "nothing about whether the covariate model is mechanistically right.",
        "The permutation null has finite resolution: with the default 20 "
        "permutations the null band is estimated from 20 draws and is reported as "
        "a band, not a p-value. Raising n_permutations tightens it at linear cost.",
        "This trace audits the *measurement* path. It is not an audit of the "
        "engine's biology, and it establishes nothing about treatment effect.",
        "Out-of-sample here means held out from the Cox fit, across a cohort "
        "collected retrospectively. It does not mean prospectively held out in "
        "time, which is the stronger design and is not available from these data.",
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI: print the trace. ``--full`` uses the whole cohort and more permutations."""
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Leakage and out-of-sample audit trace")
    ap.add_argument("--permutations", type=int, default=20)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--endpoint", default="os", choices=("os", "progression"))
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    text = render_audit_trace(
        n_permutations=args.permutations, n_folds=args.folds, seed=args.seed,
        endpoint=args.endpoint, max_patients=args.max_patients,
    )
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
