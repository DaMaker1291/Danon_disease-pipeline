"""Tests for the predictive-measurement module.

The load-bearing test here is `test_cox_recovers_a_known_signal`. During
development the fit was self-consistent -- gradient and Hessian both matched
finite differences -- while producing coefficients with the *wrong sign*
because observations were ordered by descending time, which inverts every risk
set. Checking internal consistency would have passed. Only recovery of a known
truth catches that, so that is what is pinned.

The invariant tests are pinned on purpose-built bad trajectories as well as
real ones: a detector that never fires is indistinguishable from a detector
that cannot fire.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from danon.evosteer import (
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    MTDPolicy,
    SimulationTrajectory,
    TumorEvolutionSimulator,
    make_gbm_phantom,
)
from danon.predictive import (
    Concordance,
    EngineDiscrimination,
    check_invariants,
    cox_fit,
    engine_discrimination_verdict,
    harrell_c_index,
    kfold_holdout_c_index,
    load_cohort_rows,
    monte_carlo_policy_stability,
    predictive_provenance,
    what_predictive_measurement_cannot_prove,
)


def _phantom(size: int = 10, seed: int = 0):
    return make_gbm_phantom(size=size, seed=seed)


# ---------------------------------------------------------------------------
# The statistic
# ---------------------------------------------------------------------------


class TestConcordance:
    def test_perfect_and_antiperfect_ordering(self):
        times = [10.0, 20.0, 30.0, 40.0, 50.0]
        events = [1] * 5
        assert harrell_c_index([5, 4, 3, 2, 1], times, events).c_index == pytest.approx(1.0)
        assert harrell_c_index([1, 2, 3, 4, 5], times, events).c_index == pytest.approx(0.0)

    def test_all_censored_is_not_estimable_rather_than_a_number(self):
        c = harrell_c_index([1, 2, 3], [5, 5, 5], [0, 0, 0])
        assert not c.estimable
        assert "not estimable" in str(c)

    def test_delegates_to_the_single_implementation_in_validation(self):
        from danon.validation import concordance_index

        rng = np.random.RandomState(0)
        times = rng.exponential(10, 40).tolist()
        events = rng.randint(0, 2, 40).tolist()
        risk = rng.normal(size=40).tolist()
        assert harrell_c_index(risk, times, events).c_index == pytest.approx(
            concordance_index(times, events, risk)
        )


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


class TestCoxFit:
    def test_cox_recovers_a_known_signal(self):
        rng = np.random.RandomState(0)
        n = 600
        X = rng.normal(size=(n, 2))
        time = rng.exponential(1.0 / np.exp(0.8 * X[:, 0] - 0.4 * X[:, 1]))
        fit = cox_fit(X, time, np.ones(n, dtype=int), ("a", "b"))
        assert fit.converged
        assert fit.coefficients[0] == pytest.approx(0.8, abs=0.2)
        assert fit.coefficients[1] == pytest.approx(-0.4, abs=0.2)
        assert fit.hazard_ratios["a"] > 1.0 > fit.hazard_ratios["b"]

    def test_cox_recovers_the_signal_under_censoring(self):
        rng = np.random.RandomState(1)
        n = 800
        X = rng.normal(size=(n, 2))
        time = rng.exponential(1.0 / np.exp(0.8 * X[:, 0] - 0.4 * X[:, 1]))
        censor = rng.exponential(1.5, n)
        event = (time <= censor).astype(int)
        fit = cox_fit(X, np.minimum(time, censor), event, ("a", "b"))
        assert fit.coefficients[0] == pytest.approx(0.8, abs=0.3)
        assert fit.coefficients[1] == pytest.approx(-0.4, abs=0.3)

    def test_no_events_is_refused(self):
        rng = np.random.RandomState(0)
        X = rng.normal(size=(20, 2))
        with pytest.raises(ValueError, match="not identifiable"):
            cox_fit(X, np.arange(1, 21, dtype=float), np.zeros(20, dtype=int), ("a", "b"))

    def test_length_mismatch_is_refused(self):
        rng = np.random.RandomState(0)
        X = rng.normal(size=(20, 2))
        with pytest.raises(ValueError, match="matching"):
            cox_fit(X, np.arange(1, 21, dtype=float), np.ones(5, dtype=int), ("a", "b"))


# ---------------------------------------------------------------------------
# The real cohort
# ---------------------------------------------------------------------------


class TestRealCohortHoldout:
    @pytest.fixture(scope="class")
    def rows(self):
        rows, notes = load_cohort_rows()
        assert rows, f"no cohort rows: {notes}"
        return rows

    def test_cohort_has_real_patients_and_events(self, rows):
        assert len(rows) > 300
        assert sum(r.event for r in rows) > 100
        assert all(r.time_days > 0 for r in rows)

    def test_heldout_discrimination_exceeds_chance(self, rows):
        rep = kfold_holdout_c_index(rows, n_folds=5)
        assert len(rep.fold_c_indices) == 5
        assert rep.mean_multivariable > 0.55
        assert rep.sd_multivariable < 0.1
        assert "weak discrimination" in rep.verdict

    def test_older_age_carries_higher_hazard(self, rows):
        rep = kfold_holdout_c_index(rows, n_folds=5)
        assert rep.hazard_ratios["age_years"] > 1.0

    def test_progression_endpoint_is_reported_separately(self):
        rows, notes = load_cohort_rows(endpoint="progression")
        assert rows and any("progression" in n for n in notes)
        rep = kfold_holdout_c_index(rows, n_folds=5, endpoint="progression")
        assert 0.5 < rep.mean_multivariable < 0.7

    def test_too_few_patients_for_the_requested_folds_is_refused(self, rows):
        with pytest.raises(ValueError, match="too few"):
            kfold_holdout_c_index(rows[:12], n_folds=5)

    def test_bad_endpoint_name_is_refused(self):
        with pytest.raises(ValueError, match="endpoint"):
            load_cohort_rows(endpoint="response")


class TestEngineDiscriminationVerdict:
    def test_engine_c_index_is_declared_not_estimable_with_reasons(self):
        v = engine_discrimination_verdict()
        assert isinstance(v, EngineDiscrimination)
        assert v.estimable is False
        assert v.c_index is None
        assert "no patient-level dataset pairs baseline tumour state" in str(v)
        assert len(v.what_would_be_needed) >= 3

    def test_verdict_names_endpoint_translation_as_the_blocker(self):
        """The blocker is the unfitted model-to-clinic mapping, not the model."""
        v = engine_discrimination_verdict()
        text = str(v) + " ".join(v.what_would_be_needed)
        assert "RANO-fraction" in text
        assert "endpoint" in text

    def test_engine_forecast_does_vary_by_patient_geometry(self):
        """A false claim that used to be asserted in this repo, now pinned.

        ``engine_discrimination_verdict`` once said the engine "emits one
        survival curve per simulation rather than one per patient". That was
        asserted without checking and is wrong: the engine's forecast varies
        with the patient's geometry, so patient-dependence is not the reason a
        C-index is unavailable. This test holds the correction in place by
        measuring the spread across real geometries rather than reading prose.
        """
        import json
        import os

        path = "outputs_real/outcome_cards.json"
        if not os.path.exists(path):
            import pytest as _pytest

            _pytest.skip("no exported outcome cards on disk")
        with open(path) as fh:
            cards = json.load(fh)
        by_geometry: dict = {}
        for c in cards["cases"]:
            by_geometry.setdefault(c["case_id"], set()).add(
                round(float(c["day_365"]["burden_mtd"]), 1)
            )
        assert len(by_geometry) >= 2
        medians = {min(v) for v in by_geometry.values()}
        assert len(medians) >= 2, (
            "the engine produced identical MTD burdens across distinct "
            "geometries, so a per-patient prediction genuinely is impossible"
        )
        # And the verdict must not claim otherwise.
        assert "one survival curve per simulation" not in str(
            engine_discrimination_verdict()
        )


# ---------------------------------------------------------------------------
# Monte Carlo stability
# ---------------------------------------------------------------------------


class TestMonteCarloStability:
    def test_paired_noise_report_is_well_formed(self):
        geom = _phantom(size=8)
        rep = monte_carlo_policy_stability(
            geom, sigmas=(0.10,), n_replicates=2, days=40
        )
        assert len(rep.levels) == 1
        lvl = rep.levels[0]
        assert lvl.n_replicates == 2
        assert 0.0 <= lvl.stable_fraction <= 1.0
        assert lvl.n_lower_burden_policy + lvl.n_lower_burden_mtd == 2
        assert lvl.median_burden_ratio > 0.0
        assert "model outcome, not a survival benefit" in " ".join(rep.notes)

    def test_zero_replicates_and_negative_sigma_are_refused(self):
        geom = _phantom(size=8)
        with pytest.raises(ValueError, match="at least one replicate"):
            monte_carlo_policy_stability(geom, n_replicates=0, days=30)
        with pytest.raises(ValueError, match="sigma must be positive"):
            monte_carlo_policy_stability(geom, sigmas=(-0.1,), n_replicates=1, days=30)

    def test_noise_is_applied_to_the_named_targets_only(self):
        """The collateral mechanism must not be perturbed: the report says so."""
        geom = _phantom(size=8)
        rep = monte_carlo_policy_stability(geom, sigmas=(0.2,), n_replicates=1, days=30)
        assert rep.targets == ["tumour_growth", "drug_transport", "intake_r0"]
        assert any("collateral kill mechanism" in n for n in rep.notes)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


class TestInvariants:
    @pytest.fixture(scope="class")
    def run_and_geometry(self):
        geom = _phantom(size=10)
        sim = TumorEvolutionSimulator(geom)
        traj = sim.run(MTDPolicy(), days=25, initial_resistant_fraction=0.08, seed=0)
        return traj, geom

    def test_real_trajectory_has_no_violations(self, run_and_geometry):
        traj, geom = run_and_geometry
        assert check_invariants(traj, geom) == []

    def test_negative_density_is_detected(self, run_and_geometry):
        traj, geom = run_and_geometry
        bad = replace(traj, burden_S=traj.burden_S.copy())
        bad.burden_S[5] = -1e-3
        v = check_invariants(bad, geom)
        assert any(x.invariant == "non_negativity" and x.day == 5 for x in v)

    def test_carrying_capacity_overflow_is_detected(self, run_and_geometry):
        traj, geom = run_and_geometry
        bad = replace(traj, burden_S=(traj.burden_S + 1e6).copy())
        v = check_invariants(bad, geom)
        assert any(x.invariant == "carrying_capacity" for x in v)

    def test_pde_residual_detector_fires_on_an_impossible_field(self, run_and_geometry):
        traj, geom = run_and_geometry
        chaos = np.zeros_like(geom.tumor_mask, dtype=float)
        # a checkerboard field is maximally inconsistent with its own Laplacian
        chaos[::2, ::2, ::2] = 1e6
        bad = replace(traj, final_conc_A=chaos)
        v = check_invariants(bad, geom)
        assert any(x.invariant == "pde_operator_consistency" for x in v)

    def test_wrong_series_length_is_detected(self, run_and_geometry):
        traj, geom = run_and_geometry
        bad = replace(traj, burden_S=traj.burden_S[:-1].copy())
        v = check_invariants(bad, geom)
        assert any(x.invariant == "trajectory_shape" for x in v)

    def test_laplacian_matches_a_quadratic_field(self):
        """The independent stencil must still be a correct Laplacian."""
        from danon.predictive import _laplacian

        n = 8
        idx = np.arange(n)
        X, Y, Z = np.meshgrid(idx, idx, idx, indexing="ij")
        f = (X ** 2 + Y ** 2 + Z ** 2).astype(float)
        interior = _laplacian(f)[1:-1, 1:-1, 1:-1]
        assert np.allclose(interior, 6.0)


# ---------------------------------------------------------------------------
# Honesty envelope
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_engine_discrimination_and_benefit_stay_unknown(self):
        prov = predictive_provenance()
        assert prov["engine_patient_level_discrimination"].value == "unknown"
        assert prov["clinical_benefit"].value == "unknown"
        assert prov["c_index_implementation"].value == "established"

    def test_limits_name_each_layer_s_shortcoming(self):
        text = " ".join(what_predictive_measurement_cannot_prove()).lower()
        assert "model-form error" in text
        assert "prospective trials" in text
        assert "no paired" in text
