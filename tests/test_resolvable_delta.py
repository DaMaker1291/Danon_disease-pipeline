"""Tests for the cohort-resolution design calculation.

The point of `resolvable_delta_c_index` is to convert "we would need more
data" into a number, so the tests pin the two properties that make the number
meaningful:

* the floor falls as the cohort grows (otherwise it is not a sampling
  statistic at all), and
* the two floors stay ordered -- the fold yardstick is never the tighter of
  the two, because folds at a small n are few and heterogeneous.

Plus the refusals, since a design calculation that silently returns a floor
for an impossible cohort is worse than one that raises.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from danon.cohort_features import ResolvableDelta, resolvable_delta_c_index
from danon.predictive import PatientRow

CLINICAL_PATH = "data/gdc_clinical/tcga_gbm_cases.json"


def _synthetic_cohort(n: int = 300, seed: int = 0):
    """Rows with a real age-and-diagnosis effect, so the statistic is real.

    Survival is generated from a hazard that depends on age and on GBM status,
    which is the same design the function scores. A fixture with no signal
    would make every floor look identical and prove nothing.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for i in range(n):
        age = float(rng.uniform(30, 85))
        gbm = int(rng.rand() < 0.85)
        base = 900.0 * np.exp(-0.06 * (age - 55))
        if gbm == 0:
            base *= 1.6
        event = int(rng.rand() < 0.8)
        t = (
            base * float(rng.uniform(0.4, 1.6))
            if event
            else base * float(rng.uniform(1.0, 3.0))
        )
        rows.append(
            PatientRow(f"P{i}", age, int(rng.rand() < 0.6), gbm, max(t, 1.0), event)
        )
    return rows


class TestRefusals:
    def test_cohort_too_small_for_the_split(self):
        with pytest.raises(ValueError, match="too few"):
            resolvable_delta_c_index(_synthetic_cohort(120), n_patients=20)

    def test_cannot_draw_more_than_the_cohort_holds(self):
        with pytest.raises(ValueError, match="cannot draw"):
            resolvable_delta_c_index(_synthetic_cohort(120), n_patients=500)

    def test_one_draw_has_no_spread(self):
        with pytest.raises(ValueError, match="n_draws"):
            resolvable_delta_c_index(_synthetic_cohort(120), n_patients=100, n_draws=1)


class TestFloorBehaves:
    def test_floor_falls_as_the_cohort_grows(self):
        rows = _synthetic_cohort(300)
        small = resolvable_delta_c_index(rows, n_patients=60, n_draws=10)
        large = resolvable_delta_c_index(rows, n_patients=240, n_draws=10)
        assert small.estimable and large.estimable
        assert small.se_across_draws > 1.5 * large.se_across_draws
        assert small.floor_optimistic > large.floor_optimistic

    def test_fold_yardstick_is_never_the_tighter_floor(self):
        rows = _synthetic_cohort(240)
        r = resolvable_delta_c_index(rows, n_patients=120, n_draws=5)
        assert r.floor_optimistic <= r.floor_conservative

    def test_resolves_uses_the_optimistic_floor(self):
        r = ResolvableDelta(
            n_patients=168, n_draws=10, n_folds=5, mean_c=0.64,
            se_across_draws=0.022, mean_fold_sd=0.061, age_only_c=0.65,
        )
        assert r.resolves(0.030)
        assert not r.resolves(0.020)
        assert not r.resolves(-0.020)

    def test_not_estimable_is_stated_not_zeroed(self):
        r = ResolvableDelta(
            n_patients=168, n_draws=1, n_folds=5, mean_c=float("nan"),
            se_across_draws=float("nan"), mean_fold_sd=float("nan"),
            age_only_c=float("nan"),
        )
        assert not r.estimable
        assert "not estimable" in r.verdict()
        assert not r.resolves(0.5)

    def test_exhaustive_draws_are_flagged_as_split_only(self):
        rows = _synthetic_cohort(200)
        r = resolvable_delta_c_index(rows, n_patients=200, n_draws=5)
        assert r.exhaustive_draws
        assert any("split variation only" in n for n in r.notes)
        assert any("not measure any stream" in n for n in r.notes)

    def test_sampling_draws_are_not_flagged_exhaustive(self):
        rows = _synthetic_cohort(240)
        r = resolvable_delta_c_index(rows, n_patients=120, n_draws=4)
        assert not r.exhaustive_draws
        assert not any("split variation only" in n for n in r.notes)


@pytest.mark.skipif(
    not os.path.exists(CLINICAL_PATH), reason="cached GDC clinical export absent"
)
class TestRealCohortAnchor:
    def test_floor_at_the_manifest_cohort_size_is_finite_and_wide(self):
        from danon.predictive import load_cohort_rows

        rows, _ = load_cohort_rows()
        r = resolvable_delta_c_index(rows, n_patients=168, n_draws=5)
        assert r.estimable
        # A 168-patient cohort cannot pin the ceiling to a hair's width; the
        # committed artifact measures 0.0219 over 24 draws, and the assertion
        # band only says the number is in that regime.
        assert 0.005 < r.floor_optimistic < 0.06
        assert 0.55 < r.mean_c < 0.72
        assert r.floor_conservative > r.floor_optimistic

    def test_committed_artifact_matches_the_manifest_count(self):
        from danon.availability import mri_availability_census
        from danon.clinical_twin import stream4_resolution_report

        if not os.path.exists("outputs_real/stream4_resolution.json"):
            pytest.skip("stream-4 design calculation not built")
        payload = stream4_resolution_report()
        assert (
            payload["resolution"]["n_patients"]
            == mri_availability_census().n_patients_with_usable_pair
        )
        assert "DESIGN calculation" in payload["what_this_is"]
        assert len(payload["what_this_cannot_show"]) >= 3
