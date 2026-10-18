"""The closed-loop twin: what the assimilation audit may and may not say.

These tests exist because the audit's OUTPUT is a sentence about accuracy, and a
sentence is the easiest thing in a repository to make true by accident. So the
tests do not check that assimilation "works" -- they check the parts that would
make a wrong number look right: that the two metrics are the declared formulas
(hand-computed here, not read back from the module), that both arms are scored
over the same days, that a schedule the design cannot honour is refused instead
of silently trimmed, and that the prior the ensemble starts from can actually
contain the truth (the version-2 failure: a prior whose support excluded the
truth could never be reached by a linear analysis, and the arm lost).

The log-space burden correction is tested directly, because its predecessor was
a ratio-of-fractions rescale that threw the burden half a decade off the truth
when the fraction was near 1. That failure is cheap to reintroduce and invisible
in an aggregate score, so the invariant is pinned here: correcting the burden
multiplies the resistant field by 10**delta, and nothing else.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from danon.assimilation import (
    BURDEN_FLOOR,
    ClosedLoopTracker,
    DEFAULT_GEOM_SEED,
    DEFAULT_OBSERVATION_DAYS,
    GROWTH_BOUNDS,
    PRIOR_GROWTH_HALF_WIDTH,
    PRIOR_GROWTH_MEAN,
    TRUTH_GROWTH_R,
    TRUTH_GROWTH_S,
    Member,
    Observation,
    _apply_burden,
    _member_vector,
    assimilation_twin_audit,
    assimilation_twin_validation,
    log_burden_accuracy_pct,
    log10_burden,
    resistant_fraction,
    tracking_accuracy_pct,
)
from danon.evosteer import CloneParams, MTDPolicy, TumorEvolutionSimulator, make_gbm_phantom


def _member(n_s=1000.0, n_r=10.0) -> Member:
    s = np.full((4, 4), n_s / 16.0)
    r = np.full((4, 4), n_r / 16.0)
    return Member(
        params=CloneParams(growth_S=0.04, growth_R=0.05, mutation_rate=1e-4),
        seed=1,
        N_S=s,
        N_R=r,
    )


# ---------------------------------------------------------------------------
# The metrics are the declared formulas, checked by hand
# ---------------------------------------------------------------------------


def test_fraction_accuracy_is_the_mean_absolute_error_in_points():
    tracked = [0.0, 0.5, 1.0]
    truth = [0.0, 0.4, 1.0]
    # |0| + |0.1| + |0| over three days = 0.1/3
    assert tracking_accuracy_pct(tracked, truth) == pytest.approx(
        100.0 * (1.0 - 0.1 / 3.0)
    )


def test_burden_accuracy_scores_one_full_decade_as_zero():
    # One decade of error on a single day scores 0 for that day.
    assert log_burden_accuracy_pct([1.0], [2.0]) == pytest.approx(0.0)
    # Half a decade scores 50.
    assert log_burden_accuracy_pct([1.0], [1.5]) == pytest.approx(50.0)
    assert log_burden_accuracy_pct([1.0], [1.0]) == pytest.approx(100.0)


def test_mismatched_or_empty_curves_are_refused():
    with pytest.raises(ValueError, match="equal length"):
        tracking_accuracy_pct([0.1, 0.2], [0.1])
    with pytest.raises(ValueError, match="equal length"):
        log_burden_accuracy_pct([], [])
    with pytest.raises(ValueError, match="finite"):
        tracking_accuracy_pct([float("nan")], [0.1])


def test_the_fraction_and_burden_curves_are_derived_not_stored():
    f = resistant_fraction([90.0, 0.0], [10.0, 0.0])
    assert f[0] == pytest.approx(0.1)
    assert f[1] == 0.0, "a zero-mass day must be zero, not a division blow-up"
    b = log10_burden([100.0, 0.0])
    assert b[0] == pytest.approx(2.0)
    assert b[1] == pytest.approx(math.log10(BURDEN_FLOOR))


# ---------------------------------------------------------------------------
# The prior must be able to reach the truth
# ---------------------------------------------------------------------------


def test_the_prior_support_contains_the_truth_growth_rates():
    lo = PRIOR_GROWTH_MEAN - PRIOR_GROWTH_HALF_WIDTH
    hi = PRIOR_GROWTH_MEAN + PRIOR_GROWTH_HALF_WIDTH
    assert lo >= GROWTH_BOUNDS[0], "the prior must be physically simulable"
    assert hi <= GROWTH_BOUNDS[1]
    assert lo <= min(TRUTH_GROWTH_S, TRUTH_GROWTH_R)
    assert hi >= max(TRUTH_GROWTH_S, TRUTH_GROWTH_R), (
        "a linear analysis interpolates inside the ensemble support; a prior "
        "that excludes the truth is the version-2 bug"
    )


# ---------------------------------------------------------------------------
# The burden correction is multiplicative in log space
# ---------------------------------------------------------------------------


def test_correcting_the_burden_multiplies_the_field_by_ten_to_the_delta():
    m = _member(n_r=500.0)
    before = float(m.N_R.sum())
    _apply_burden(m, math.log10(before) + 0.5)
    assert float(m.N_R.sum()) == pytest.approx(before * 10.0 ** 0.5, rel=1e-9)
    assert np.all(m.N_S == np.full((4, 4), 1000.0 / 16.0)), (
        "the sensitive field is not what the observations constrain"
    )


def test_a_near_saturated_fraction_no_longer_moves_the_mass_by_a_fraction_ratio():
    """The version-3 failure, expressed as an invariant.

    A state at fraction 0.9996 has an R:S ratio of ~2500:1; the same state one
    part in ten thousand higher has ~10000:1. Rescaling mass to match that
    fraction ratio multiplies the burden by 4 -- this test pins that the burden
    correction is driven by the analysed BURDEN, so a half-decade update is the
    only thing that can move it a half decade.
    """
    m = _member(n_s=9996.0, n_r=4.0)
    before = float(m.N_R.sum())
    target = math.log10(before) - 0.05
    _apply_burden(m, target)
    assert float(m.N_R.sum()) == pytest.approx(10.0 ** target, rel=1e-9)
    assert float(m.N_R.sum()) == pytest.approx(before * 10.0 ** -0.05, rel=1e-9)


def test_a_zero_resistant_field_is_seeded_rather_than_left_uncorrected():
    m = _member(n_r=0.0)
    _apply_burden(m, 1.0)
    assert float(m.N_R.sum()) == pytest.approx(10.0)
    assert np.all(np.isfinite(m.N_R))


def test_an_extinct_state_and_a_non_finite_burden_are_refused():
    with pytest.raises(ValueError, match="extinct"):
        _apply_burden(_member(n_s=0.0, n_r=0.0), 1.0)
    with pytest.raises(ValueError, match="finite"):
        _apply_burden(_member(), float("nan"))


def test_the_augmented_state_vector_carries_the_resistant_log_burden_last():
    m = _member(n_r=1000.0)
    v = _member_vector(m)
    assert v.shape == (4,)
    assert v[3] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "obs, match",
    [
        ((14, 14, 28), "unique"),
        ((0, 14, 28), "1 <= day < horizon"),
        ((14, 28, 42), "1 <= day < horizon"),
    ],
)
def test_a_schedule_the_design_cannot_honour_is_refused(obs, match):
    with pytest.raises(ValueError, match=match):
        assimilation_twin_audit(horizon_days=42, observation_days=obs)


# ---------------------------------------------------------------------------
# The tracker: the same code path a live feed would use
# ---------------------------------------------------------------------------


def _tracker_and_truthtell(horizon=21, obs_day=21):
    """A tracker with a wrong prior, and a measurement taken from the truth."""
    geom = make_gbm_phantom(size=16, seed=3)
    truth_params = CloneParams(growth_S=0.05, growth_R=0.06, mutation_rate=1e-4)
    init = TumorEvolutionSimulator(geom, clones=truth_params).seed(0.01, seed=4)
    traj = TumorEvolutionSimulator(geom, clones=truth_params).run(
        MTDPolicy(), days=horizon, seed=11, initial_state=init
    )
    frac = resistant_fraction(traj.burden_S, traj.burden_R)[obs_day - 1]
    logb = log10_burden(traj.burden_R)[obs_day - 1]
    # Spread matters: an ensemble whose members share one parameter vector has
    # no parameter covariance, so no Kalman gain, so nothing to learn from.
    members = [
        Member(
            params=CloneParams(
                growth_S=0.018 + 0.001 * i,
                growth_R=0.018 + 0.001 * i,
                mutation_rate=1e-4,
            ),
            seed=100 + 17 * i,
            N_S=init[0].copy(),
            N_R=init[1].copy(),
        )
        for i in range(10)
    ]
    tracker = ClosedLoopTracker(geom, members, rng=np.random.default_rng(5))
    obs = Observation(day=obs_day, resistant_fraction=float(frac), log10_resistant_burden=float(logb))
    return tracker, obs


def test_the_tracker_refuses_an_out_of_order_measurement():
    tracker, obs = _tracker_and_truthtell()
    tracker.step(obs)
    with pytest.raises(ValueError, match="not after the current"):
        tracker.step(obs)
    with pytest.raises(ValueError, match="not after the current"):
        tracker.step(Observation(day=obs.day - 1, resistant_fraction=0.1, log10_resistant_burden=1.0))


def test_the_tracker_refuses_to_advance_backwards():
    tracker, _ = _tracker_and_truthtell()
    with pytest.raises(ValueError, match="not after the current"):
        tracker.advance_to(0)


def test_one_step_reduces_the_error_against_a_true_measurement():
    """The analysis must move the tracked burden toward a true measurement.

    Deliberately NOT asserted: that the fraction also improves. The analysed
    state carries the resistant burden and the parameters, so a correction
    moves R while the sensitive field stays where the wrong prior grew it, and
    the fraction can therefore move off (the module records this, and the
    version that tried to place S as well measured worse). Asserting a
    fraction improvement here would be asserting something the design does
    not do.
    """
    tracker, obs = _tracker_and_truthtell()
    report = tracker.step(obs)
    assert report["day"] == obs.day
    assert report["burden_error_after"] < report["burden_error_before"]
    assert tracker.day == obs.day
    assert tracker.observations == [obs]
    assert report["ensemble_mean_growth_R_after"] > report["ensemble_mean_growth_R_before"], (
        "a prior below the truth must be moved UP by a true measurement"
    )


def test_an_ensemble_with_one_shared_parameter_vector_cannot_learn():
    """No spread means no covariance means no gain -- the honest NULL result.

    This is why the twin's prior is drawn per member; an ensemble copied from a
    single parameter vector would look like a working filter that never moves,
    and every accuracy number it produced would be the blind arm's.
    """
    geom = make_gbm_phantom(size=16, seed=3)
    init = TumorEvolutionSimulator(geom).seed(0.01, seed=4)
    same = CloneParams(growth_S=0.02, growth_R=0.02, mutation_rate=1e-4)
    members = [
        Member(params=same, seed=100 + 17 * i, N_S=init[0].copy(), N_R=init[1].copy())
        for i in range(10)
    ]
    tracker = ClosedLoopTracker(geom, members, rng=np.random.default_rng(5))
    truth = CloneParams(growth_S=0.05, growth_R=0.06, mutation_rate=1e-4)
    traj = TumorEvolutionSimulator(geom, clones=truth).run(
        MTDPolicy(), days=21, seed=11, initial_state=init
    )
    obs = Observation(
        day=21,
        resistant_fraction=float(resistant_fraction(traj.burden_S, traj.burden_R)[20]),
        log10_resistant_burden=float(log10_burden(traj.burden_R)[20]),
    )
    report = tracker.step(obs)
    assert report["ensemble_mean_growth_R_after"] == pytest.approx(0.02)
    assert report["ensemble_mean_growth_S_after"] == pytest.approx(0.02)


def test_the_tracker_needs_an_ensemble_that_can_carry_a_covariance():
    geom = make_gbm_phantom(size=16, seed=3)
    one = _member()
    with pytest.raises(ValueError, match="at least 4 members"):
        ClosedLoopTracker(geom, [one], rng=np.random.default_rng(0))


def test_track_returns_one_value_per_day_of_the_horizon():
    tracker, obs = _tracker_and_truthtell(horizon=21, obs_day=14)
    frac, logb = tracker.track([obs], horizon_days=21)
    assert frac.shape == (21,)
    assert logb.shape == (21,)
    assert np.all(np.isfinite(frac)) and np.all(np.isfinite(logb))


# ---------------------------------------------------------------------------
# The twin, end to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_twin():
    return assimilation_twin_audit(
        horizon_days=42,
        observation_days=(14, 28),
        ensemble_size=8,
        geom_seed=DEFAULT_GEOM_SEED,
    )


def test_the_twin_scores_both_arms_over_the_same_days(small_twin):
    curves = small_twin["curves"]
    n = small_twin["design"]["horizon_days"]
    for key in (
        "truth_fraction",
        "blind_fraction",
        "assimilated_fraction",
        "truth_log10_burden",
        "blind_log10_burden",
        "assimilated_log10_burden",
    ):
        assert len(curves[key]) == n, f"{key} is not on the common day grid"
    assert curves["day"] == list(range(1, n + 1))


def test_the_recorded_observation_days_are_the_ones_assimilated(small_twin):
    curves = small_twin["curves"]
    assert [o["day"] for o in curves["observations"]] == [14, 28]
    assert [int(r["day"]) for r in small_twin["metrics"]["per_observation"]] == [14, 28]
    assert small_twin["design"]["observation_days"] == [14, 28]


def test_the_reading_follows_the_measured_numbers_rather_than_a_preference(small_twin):
    m = small_twin["metrics"]
    reading = small_twin["reading"]
    if m["burden_gain_points"] > 0:
        assert "improved" in reading
        assert f"{m['blind_burden_accuracy_pct']:.1f}%" in reading
        assert f"{m['assimilated_burden_accuracy_pct']:.1f}%" in reading
    else:
        assert "did NOT beat" in reading


def test_the_record_never_claims_patient_tracking(small_twin):
    refusals = " ".join(small_twin["refusals"]).lower()
    assert "no patient telemetry" in refusals
    assert "unknown" in refusals
    assert "synthetic" in small_twin["reading"].lower()
    design = small_twin["design"]
    assert design["kind"].startswith("synthetic twin")
    assert design["blind_prior"]["support_contains_truth"] is True
    assert design["both_arms_share_day0_fields"] is True


def test_the_truth_is_not_the_blind_arms_own_miss(small_twin):
    """If both arms were identical the comparison would be vacuous."""
    c = small_twin["curves"]
    assert np.max(np.abs(np.array(c["blind_log10_burden"]) - np.array(c["truth_log10_burden"]))) > 0.05
    assert np.max(
        np.abs(np.array(c["assimilated_log10_burden"]) - np.array(c["blind_log10_burden"]))
    ) > 0.01, "the correction did nothing at all -- the arms are the same run"


# ---------------------------------------------------------------------------
# Held-out validation
# ---------------------------------------------------------------------------


def test_validation_refuses_to_score_on_the_seed_that_chose_the_configuration():
    with pytest.raises(ValueError, match="development seed"):
        assimilation_twin_validation(geom_seeds=(DEFAULT_GEOM_SEED, 5))


def test_validation_reports_the_worst_seed_not_only_the_mean():
    rec = assimilation_twin_validation(
        geom_seeds=(5, 7),
        horizon_days=28,
        observation_days=(14,),
        ensemble_size=8,
    )
    s = rec["summary"]
    gains = [r["burden_gain_points"] for r in rec["runs"]]
    assert s["seeds_total"] == 2
    assert s["min_burden_gain_points"] == pytest.approx(min(gains))
    assert s["mean_burden_gain_points"] == pytest.approx(sum(gains) / 2)
    assert s["seeds_where_assimilation_beats_blind"] == sum(1 for g in gains if g > 0)
    assert "Synthetic twin only" in rec["reading"]
    if s["seeds_where_assimilation_beats_blind"] < s["seeds_total"]:
        assert "LOST" in rec["reading"]


def test_the_default_schedule_is_the_declared_fourteen_day_one():
    assert DEFAULT_OBSERVATION_DAYS == (14, 28, 42, 56)


# ---------------------------------------------------------------------------
# The published sentence must be this record's numbers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]


def test_the_registry_sentence_carries_this_record_s_numbers():
    """A claim that quotes a number nothing re-derives is the failure mode.

    The claim sentence in `danon/claims.py` states the mean and worst-case
    burden accuracy. Those numbers live in the held-out record, so the sentence
    must still be quoting them: if the record is regenerated and the sentence
    is not, this fails rather than publishing a stale figure.
    """
    from danon.claims import CHECKS, CLAIMS

    path = ROOT / "outputs_real" / "assimilation_twin_holdout.json"
    if not path.exists():
        pytest.skip(f"{path} not built (run _run_assimilation_holdout.py)")
    rec = json.loads(path.read_text(encoding="utf-8"))
    s = rec["summary"]
    claim = next(c for c in CLAIMS if c.id == "closed_loop_assimilation_is_measured")
    for key in (
        "mean_assimilated_burden_accuracy_pct",
        "mean_blind_burden_accuracy_pct",
        "min_assimilated_burden_accuracy_pct",
    ):
        assert f"{s[key]:.1f}%" in claim.sentence, f"sentence does not quote {key}"
    assert claim.check in CHECKS, "the claim's check is not in the registry's checks"
    assert claim.check == "closed_loop_assimilation_measured"


def test_the_published_sentence_never_calls_it_a_patient_result():
    from danon.claims import CLAIMS

    claim = next(c for c in CLAIMS if c.id == "closed_loop_assimilation_is_measured")
    text = (claim.sentence + " " + claim.limit).lower()
    assert "no patient telemetry" in text or "not a patient result" in text
    assert "unknown" in text, "real-tumour tracking accuracy must stay UNKNOWN"
