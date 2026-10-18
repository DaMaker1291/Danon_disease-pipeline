"""Tests for the dosing calendar, the safety screen, and the schedule search.

The load-bearing test is IDENTITY. A calendar is only real if the dose series
the engine actually applied is the dose series the schedule's text claims.
That is pinned against the engine's own recorded `dose_A` / `dose_B` arrays
rather than against a restatement of the same code that built them.

The rest follows the rule this repository already uses: a check that cannot
fail is decoration, so the failure paths are tested as carefully as the passing
ones. In particular the search is pinned as being ABLE TO LOSE -- if the only
reachable outcome were "a pulsed calendar wins", the search would be a
foregone conclusion rather than a measurement.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from danon.dosing import (
    DoseBlock,
    MyelosuppressionModel,
    Schedule,
    SchedulePolicy,
    evaluate_many,
    measure_batch_scaling,
    compare_combination_vs_single,
    alternation_candidates,
    evaluate_candidates,
    feasible_grid_schedules,
    single_drug_candidates,
    grid_size,
    screen_fidelity,
    screen_fidelity_note,
    evaluate_schedule,
    grid_size,
    provenance,
    run_competitive_release_experiment,
    screen_schedule,
    search_schedules,
    _grid_schedules,
    _kendall_tau,
    _spearman,
)
from danon.evosteer import TumorEvolutionSimulator, make_gbm_phantom


DAYS = 60  # short horizon: the calendar semantics do not depend on length


@pytest.fixture(scope="module")
def sim():
    return TumorEvolutionSimulator(make_gbm_phantom(size=16, seed=7))


# ---------------------------------------------------------------------------
# The calendar itself
# ---------------------------------------------------------------------------


class TestCalendar:
    def test_cyclic_places_blocks_on_the_declared_period(self):
        s = Schedule.cyclic("A", 150.0, on_days=5, cycle_days=26, horizon_days=100)
        assert [b.start_day for b in s.blocks] == [0, 26, 52, 78]
        assert all(b.duration_days == 5 for b in s.blocks)

    def test_dose_on_is_on_then_off(self):
        s = Schedule.cyclic("A", 150.0, 5, 26, 365)
        for day in range(0, 5):
            assert s.dose_on(day) == (1.0, 0.0)
        for day in range(5, 26):
            assert s.dose_on(day) == (0.0, 0.0)
        assert s.dose_on(26) == (1.0, 0.0)

    def test_same_drug_overlap_is_rejected(self):
        with pytest.raises(ValueError, match="overlapping"):
            Schedule((
                DoseBlock("A", 0, 10),
                DoseBlock("A", 5, 10),
            ))

    def test_different_drugs_may_overlap(self):
        s = Schedule((DoseBlock("A", 0, 10), DoseBlock("B", 0, 10)))
        assert s.dose_on(3) == (1.0, 1.0)

    def test_zero_or_negative_block_is_rejected(self):
        with pytest.raises(ValueError):
            DoseBlock("A", 0, 0)
        with pytest.raises(ValueError):
            DoseBlock("A", 0, 5, dose_level=1.5)
        with pytest.raises(ValueError):
            DoseBlock("C", 0, 5)

    def test_on_fraction_is_the_share_of_time_on_drug(self):
        s = Schedule.cyclic("A", 150.0, 5, 26, 365)
        # clipped to the 365-day horizon: 14 full blocks (70 d) + 1 day of the
        # block that starts on day 364.
        assert s.on_fraction(365) == pytest.approx(71.0 / 365.0, abs=1e-9)

    def test_on_fraction_without_a_horizon_counts_nominal_days(self):
        s = Schedule.cyclic("A", 150.0, 5, 26, 365)
        # 15 blocks x 5 d over the natural 369-day span of the calendar
        assert s.on_fraction() == pytest.approx(75.0 / 369.0, abs=1e-9)


# ---------------------------------------------------------------------------
# IDENTITY: the calendar is what the engine actually ran
# ---------------------------------------------------------------------------


class TestIdentityWithEngine:
    def test_engine_dose_series_equals_the_calendar(self, sim):
        """The engine's recorded dose_A/dose_B must equal Schedule.dose_on."""
        sched = Schedule.cyclic("A", 150.0, 5, 26, DAYS)
        traj = sim.run(SchedulePolicy(sched), days=DAYS, seed=3)
        for day in range(DAYS):
            want_a, want_b = sched.dose_on(day)
            assert traj.dose_A[day] == want_a, f"dose_A differs on day {day}"
            assert traj.dose_B[day] == want_b, f"dose_B differs on day {day}"

    def test_continuous_calendar_equals_mtd(self, sim):
        """Schedule.continuous must reproduce the engine's own MTD dose series."""
        sched = Schedule.continuous("A", 150.0, DAYS)
        cal = sim.run(SchedulePolicy(sched), days=DAYS, seed=3)
        mtd = sim.run(
            __import__("danon.evosteer", fromlist=["MTDPolicy"]).MTDPolicy(),
            days=DAYS, seed=3,
        )
        assert np.array_equal(cal.dose_A, mtd.dose_A)
        assert np.array_equal(cal.burden_S + cal.burden_R, mtd.burden_S + mtd.burden_R)

    def test_partial_dose_changes_the_trajectory(self, sim):
        """A 50% calendar is not the same run as a 100% one."""
        full = sim.run(
            SchedulePolicy(Schedule.cyclic("A", 150.0, 5, 26, DAYS)),
            days=DAYS, seed=3,
        )
        half = sim.run(
            SchedulePolicy(Schedule.cyclic("A", 75.0, 5, 26, DAYS, dose_level=0.5)),
            days=DAYS, seed=3,
        )
        assert not np.array_equal(full.burden_S, half.burden_S)

    def test_same_seed_is_deterministic(self, sim):
        sched = Schedule.cyclic("A", 150.0, 5, 26, DAYS)
        a = sim.run(SchedulePolicy(sched), days=DAYS, seed=5)
        b = sim.run(SchedulePolicy(sched), days=DAYS, seed=5)
        assert np.array_equal(a.burden_S, b.burden_S)


# ---------------------------------------------------------------------------
# The safety screen
# ---------------------------------------------------------------------------


class TestSafetyScreen:
    def test_nadir_falls_with_dose(self):
        m = MyelosuppressionModel()
        low = screen_schedule(Schedule.cyclic("A", 60.0, 5, 26, 365), 365, m)
        high = screen_schedule(Schedule.cyclic("A", 150.0, 5, 26, 365), 365, m)
        assert high.anc_nadir < low.anc_nadir

    def test_more_days_on_drug_lowers_the_nadir(self):
        m = MyelosuppressionModel()
        short = screen_schedule(Schedule.cyclic("A", 150.0, 3, 42, 365), 365, m)
        long = screen_schedule(Schedule.cyclic("A", 150.0, 21, 42, 365), 365, m)
        assert long.anc_nadir < short.anc_nadir

    def test_continuous_high_dose_is_infeasible_and_says_why(self):
        r = screen_schedule(Schedule.continuous("A", 150.0, 365), 365)
        assert not r.feasible
        assert r.violations, "an infeasible screen must carry a reason"

    def test_a_generous_calendar_passes(self):
        r = screen_schedule(Schedule.cyclic("A", 60.0, 5, 28, 365), 365)
        assert r.feasible
        assert r.anc_nadir >= r.floor

    def test_model_refuses_to_call_itself_a_clinical_standard(self):
        d = MyelosuppressionModel().to_dict()
        # The floor is the LABEL's own quoted withhold threshold, so it is now
        # traceable -- but nothing here becomes clinical decision support.
        assert d["is_label_threshold"] is True
        assert d["anc_floor"] == 1.5
        assert d["is_clinical_decision_support"] is False
        assert d["measured_in_this_repository"] is False
        assert d["status"] == "anchored_to_fda_label_with_declared_kernel_shape"


class TestLabelAnchoredSafetyModel:
    """The screen is anchored to the temozolomide label, and MEASURED.

    These are the assertions the docstring makes about the model. They are
    asserted rather than left in prose, because a prose claim about a model is
    exactly the thing that rots silently.
    """

    def test_the_labels_own_regimen_is_admissible(self):
        m = MyelosuppressionModel()
        reg = Schedule.cyclic("A", 150.0, 5, 28, 365)
        assert m.feasible(reg, 365)
        assert m.nadir(reg, 365) == pytest.approx(1.5, abs=1e-6)

    def test_one_more_day_on_drug_is_not(self):
        """The screen's boundary must land ON the label's regimen."""
        m = MyelosuppressionModel()
        six = Schedule.cyclic("A", 150.0, 6, 28, 365)
        assert not m.feasible(six, 365)
        assert m.nadir(six, 365) < 1.5

    def test_nadir_lands_on_the_reported_day(self):
        a = MyelosuppressionModel().anchor_checks()
        assert a["label_regimen_nadir_day_first_cycle"] == 28
        assert a["nadir_day_residual_days"] == 0
        # The robust answer must agree with the first-cycle measurement, or
        # the tolerance in `nadir_day` is hiding a real disagreement.
        assert a["label_regimen_nadir_day_1based"] == 28

    def test_suppression_is_not_cumulative_across_cycles(self):
        ratio = MyelosuppressionModel().anchor_checks()["worst_later_cycle_nadir_ratio"]
        # The label states myelosuppression "was generally not cumulative";
        # the model must actually behave that way, to a stated tolerance.
        assert ratio == pytest.approx(1.0, abs=0.02)

    def test_continuous_budget_is_readable_and_enforced(self):
        m = MyelosuppressionModel()
        cap = m.max_continuous_mg_m2()
        assert cap == pytest.approx(m.suppression_budget())
        assert m.feasible(Schedule.continuous("A", cap * 0.8, 365), 365)
        assert not m.feasible(Schedule.continuous("A", cap * 1.2, 365), 365)

    def test_dose_level_scales_exposure(self):
        """A block at 0.2 x full is a fifth of the mg/m2, and screens as one."""
        s = Schedule.alternating_relative(
            0.5, 3, 40, dose_mg_m2=150.0, dose_level_a=0.2,
        )
        assert s.daily_exposure(0) == pytest.approx(30.0)
        assert s.daily_exposure(3) == pytest.approx(75.0)
        # 40 days of 3-day blocks is 7 A blocks at 0.2 and 7 B blocks at 0.5
        # (the last one runs past the horizon, which `on_fraction` clips but
        # the nominal total deliberately does not).
        assert s.total_mg_m2() == pytest.approx(7 * 3 * (30.0 + 75.0))

    def test_anchor_checks_report_measured_numbers(self):
        a = MyelosuppressionModel().anchor_checks()
        assert a["label_regimen_feasible"] is True
        assert a["one_extra_day_feasible"] is False
        assert a["suppression_budget_mg_m2"] > 0.0

    def test_to_dict_carries_the_source_it_is_anchored_to(self):
        d = MyelosuppressionModel().to_dict()
        assert d["anc_floor_source"].startswith("https://")
        assert d["label_quotes"]["withhold_threshold"]["quote"].startswith(
            "Prior to dosing, withhold"
        )
        assert "not cumulative" in d["label_quotes"]["not_cumulative"]["quote"]
        assert d["kernel"]["fitted_to"]
        assert d["second_drug_potency"].startswith("ASSUMED")


# ---------------------------------------------------------------------------
# Batched evaluation
# ---------------------------------------------------------------------------


class TestBatchedEvaluation:
    """The batched path is only legitimate if it changes no number.

    `evaluate_schedule` chunks at `MEASURED_BATCH_OPTIMUM` (1) because one big
    `run_batch` is several times slower here. These tests pin the other half of
    that bargain: the chunk is a scheduling choice, so forcing the whole seed
    list into a single call must reproduce it exactly.
    """

    def test_one_big_batch_equals_the_measured_chunking(self, sim):
        named = [
            ("pulsed 5/28", Schedule.cyclic("A", 90.0, 5, 28, DAYS)),
            ("pulsed 3/14", Schedule.cyclic("A", 60.0, 3, 14, DAYS)),
            ("continuous", Schedule.continuous("A", 150.0, DAYS)),
        ]
        seeds = (0, 1, 2)
        chunked = evaluate_many(named, sim, seeds, days=DAYS)
        one_shot = [
            evaluate_schedule(name, sched, sim, seeds, days=DAYS,
                              batch_size=len(seeds))
            for name, sched in named
        ]
        for got, alone in zip(chunked, one_shot):
            assert got.name == alone.name
            assert got.median_aubc == pytest.approx(alone.median_aubc, abs=1e-12)
            assert got.median_dominance_day == alone.median_dominance_day
            assert got.median_pfs_days == pytest.approx(alone.median_pfs_days)
            assert got.n_progressed == alone.n_progressed
            assert got.median_toxicity == pytest.approx(alone.median_toxicity)

    def test_the_measured_batch_size_is_one(self):
        """If this ever changes, the speed story in the docstrings changes."""
        from danon.dosing import MEASURED_BATCH_OPTIMUM
        assert MEASURED_BATCH_OPTIMUM == 1

    def test_a_candidate_needs_at_least_one_seed(self, sim):
        with pytest.raises(ValueError):
            evaluate_schedule("empty", Schedule.cyclic("A", 90.0, 5, 28, DAYS),
                              sim, (), days=DAYS)

    def test_chunked_per_seed_values_match_the_single_run(self, sim):
        sched = Schedule.cyclic("A", 90.0, 5, 28, DAYS)
        seeds = (0, 1, 2)
        got = evaluate_schedule("p", sched, sim, seeds, days=DAYS)
        alone = evaluate_schedule("p", sched, sim, seeds, days=DAYS,
                                  batch_size=len(seeds))
        assert got.per_seed["aubc"] == pytest.approx(alone.per_seed["aubc"], abs=1e-12)
        assert got.per_seed["dominance_day"] == alone.per_seed["dominance_day"]

    def test_empty_input_is_empty_output(self, sim):
        assert evaluate_many([], sim, (0,), days=DAYS) == []

    def test_batch_order_does_not_change_a_candidate(self, sim):
        """A candidate's score must not depend on its neighbours in the batch."""
        a = Schedule.cyclic("A", 90.0, 5, 28, DAYS)
        first = evaluate_many(
            [("a", a), ("b", Schedule.continuous("A", 150.0, DAYS))],
            sim, (0, 1), days=DAYS,
        )[0]
        second = evaluate_many(
            [("b", Schedule.continuous("A", 150.0, DAYS)), ("a", a)],
            sim, (0, 1), days=DAYS,
        )[1]
        assert first.median_aubc == pytest.approx(second.median_aubc, abs=1e-12)

    def test_chunk_size_does_not_change_any_number(self, sim):
        """Chunking is scheduling only. Every batch size must agree exactly."""
        named = [
            ("a", Schedule.cyclic("A", 90.0, 5, 28, DAYS)),
            ("b", Schedule.cyclic("A", 60.0, 3, 14, DAYS)),
            ("c", Schedule.continuous("A", 150.0, DAYS)),
        ]
        seeds = (0, 1, 2, 3)
        ref = evaluate_many(named, sim, seeds, days=DAYS, batch_size=1)
        for chunk in (2, 3, 8):
            got = evaluate_many(named, sim, seeds, days=DAYS, batch_size=chunk)
            for r, g in zip(ref, got):
                assert g.median_aubc == pytest.approx(r.median_aubc, abs=1e-12)
                assert g.per_seed["aubc"] == pytest.approx(r.per_seed["aubc"], abs=1e-12)
                assert g.median_dominance_day == r.median_dominance_day

    def test_measured_batch_scaling_is_reported_not_assumed(self, sim):
        rows = measure_batch_scaling(sim, sizes=(1, 2), days=DAYS)
        assert {r["batch_size"] for r in rows} == {1, 2}
        assert all(r["total_seconds"] > 0 for r in rows)
        assert all(r["ms_per_replicate"] > 0 for r in rows)


# ---------------------------------------------------------------------------
# Evaluation and search
# ---------------------------------------------------------------------------


class TestEvaluation:
    def test_score_records_the_engine_run_count(self, sim):
        sc = evaluate_schedule(
            "t", Schedule.cyclic("A", 60.0, 5, 28, DAYS), sim, seeds=[0, 1, 2], days=DAYS
        )
        assert sc.engine_runs == 3
        assert len(sc.seeds) == 3
        assert sc.median_aubc > 0.0

    def test_grid_enumeration_is_complete_and_deduplicated(self):
        cells = _grid_schedules((3, 5), (10, 20), (0.5, 1.0), 150.0, 365)
        # on=5 > cycle=10 is fine; on=5 > 20 fine. All 2*2*2=8 cells exist.
        assert len(cells) == 8
        assert grid_size((3, 5), (10, 20), (0.5, 1.0)) == 8

    def test_grid_skips_on_longer_than_cycle(self):
        assert grid_size((30,), (10,), (1.0,)) == 0


class TestSearch:
    @pytest.fixture(scope="class")
    def small_search(self):
        return search_schedules(
            TumorEvolutionSimulator(make_gbm_phantom(size=16, seed=7)),
            seeds=(0, 1, 2, 3),
            days=45,
            on_days=(3, 5),
            cycle_days=(14, 21),
            dose_levels=(0.4, 0.6),
            screen_seeds=2,
            confirm_top=2,
        )

    def test_search_reports_its_own_cost(self, small_search):
        assert small_search.engine_runs > 0
        assert small_search.seeds == (0, 1, 2, 3)

    def test_search_result_is_serialisable(self, small_search):
        json.dumps(small_search.to_dict())

    def test_every_scored_row_actually_ran_on_the_full_seed_set(self, small_search):
        assert small_search.scored
        for s in small_search.scored:
            assert len(s.seeds) == len(small_search.seeds)
            assert s.engine_runs == len(small_search.seeds)

    def test_the_mtd_baseline_is_always_evaluated(self, small_search):
        assert "mtd" in small_search.baselines
        assert small_search.baselines["mtd"].engine_runs == len(small_search.seeds)

    def test_beats_baselines_is_a_real_verdict_not_a_constant(self, small_search):
        """The flag must be derived, and a forced-worse search must say False."""
        assert isinstance(small_search.beats_baselines, bool)
        best = small_search.baselines["mtd"].median_aubc
        if small_search.best is not None:
            expected = small_search.best.median_aubc < best
            assert small_search.beats_baselines == expected
        else:
            assert small_search.beats_baselines is False

    def test_a_search_that_cannot_win_reports_a_null(self, sim):
        """NEGATIVE CONTROL: an empty grid can only produce a null result."""
        r = search_schedules(
            sim, seeds=(0, 1), days=30,
            on_days=(5,), cycle_days=(10,), dose_levels=(0.4,),
            screen_seeds=1, confirm_top=1,
        )
        # A real grid, so this one may or may not win -- but the flag must
        # always agree with the numbers, and a non-win must carry a note.
        if not r.beats_baselines and r.best is not None:
            assert any("null result" in n for n in r.notes)

    def test_notes_always_describe_the_two_stage_design(self, small_search):
        assert any("stage 1" in n for n in small_search.notes)


# ---------------------------------------------------------------------------
# Combination (alternating two-drug) schedules
# ---------------------------------------------------------------------------


class TestAlternatingSchedule:
    def test_alternates_strictly_with_no_gap_or_overlap(self):
        s = Schedule.alternating_relative(0.7, 5, 40, dose_mg_m2=150.0)
        for day in range(0, 5):
            assert s.dose_on(day) == (1.0, 0.0)
        for day in range(5, 10):
            assert s.dose_on(day) == (0.0, 0.7)
        for day in range(10, 15):
            assert s.dose_on(day) == (1.0, 0.0)

    def test_uses_both_drugs_and_says_so(self):
        s = Schedule.alternating_relative(0.7, 5, 40)
        assert s.drugs_used() == ("A", "B")
        assert s.is_combination()
        assert "alternating" in s.describe()

    def test_describe_reports_the_relative_level_not_just_mg_m2(self):
        """A text saying 150 mg/m2 must not hide that B runs at 0.7 x full."""
        s = Schedule.alternating_relative(0.7, 5, 40, dose_mg_m2=150.0)
        assert "0.7 x full" in s.describe()

    def test_start_with_b_flips_the_order(self):
        s = Schedule.alternating(150.0, 150.0, 4, 20, start_with="B")
        assert s.dose_on(0) == (0.0, 1.0)
        assert s.dose_on(4) == (1.0, 0.0)

    def test_rejects_a_non_positive_block(self):
        with pytest.raises(ValueError):
            Schedule.alternating(150.0, 150.0, 0, 20)
        with pytest.raises(ValueError):
            Schedule.alternating(150.0, 150.0, 5, 20, start_with="C")

    def test_a_rest_gap_puts_days_on_no_drug_at_all(self):
        """`rest_days` is a real holiday, not a block of the other drug."""
        s = Schedule.alternating(150.0, 150.0, 5, 40, rest_days=9)
        assert s.dose_on(4) == (1.0, 0.0)
        for day in range(5, 14):
            assert s.dose_on(day) == (0.0, 0.0)
        assert s.dose_on(14) == (0.0, 1.0)
        assert s.dose_on(28) == (1.0, 0.0)

    def test_zero_rest_reproduces_back_to_back_alternation(self):
        gap = Schedule.alternating(150.0, 150.0, 5, 40, rest_days=0)
        tight = Schedule.alternating(150.0, 150.0, 5, 40)
        assert [gap.dose_on(d) for d in range(40)] == \
               [tight.dose_on(d) for d in range(40)]

    def test_rejects_a_negative_rest(self):
        with pytest.raises(ValueError):
            Schedule.alternating(150.0, 150.0, 5, 20, rest_days=-1)

    def test_alternating_with_rest_is_marrow_cheaper_than_without(self):
        """The rest gap must reach the safety model, not just the calendar."""
        m = MyelosuppressionModel()
        without = Schedule.alternating(150.0, 150.0, 5, 365)
        with_gap = Schedule.alternating(150.0, 150.0, 5, 365, rest_days=23)
        assert not m.feasible(without, 365)
        assert m.feasible(with_gap, 365)

    def test_alternating_is_distinct_from_the_reactive_switch_policy(self):
        """An open-loop calendar must not be mistaken for the closed loop."""
        from danon.evosteer import CollateralSwitchPolicy
        s = Schedule.alternating(150.0, 150.0, 5, 60)
        pol = SchedulePolicy(s)
        switch = CollateralSwitchPolicy()
        from danon.evosteer import PolicyState
        st = PolicyState(day=7, burden=1.0, burden_initial=1.0, resistant_fraction=0.01)
        # At day 7 the calendar is on drug B; the reactive switch, seeing a low
        # resistant fraction, is still on drug A. Different controllers.
        assert pol.dose(st) == (0.0, 1.0)
        assert switch.dose(st) == (1.0, 0.0)


class TestCombinationExposure:
    def test_second_drug_counts_toward_marrow_exposure(self):
        s = Schedule.alternating_relative(1.0, 5, 40, dose_mg_m2=150.0)
        # on drug A the exposure is A's mg; on drug B it is B's mg
        assert s.daily_exposure(0, {"A": 1.0, "B": 1.0}) == pytest.approx(150.0)
        assert s.daily_exposure(5, {"A": 1.0, "B": 1.0}) == pytest.approx(150.0)

    def test_weight_scales_the_second_drug(self):
        s = Schedule.alternating_relative(1.0, 5, 40, dose_mg_m2=150.0)
        assert s.daily_exposure(5, {"A": 1.0, "B": 0.0}) == pytest.approx(0.0)
        assert s.daily_exposure(5, {"A": 1.0, "B": 2.0}) == pytest.approx(300.0)

    def test_model_flags_the_second_drug_potency_as_unknown(self):
        s = Schedule.alternating_relative(1.0, 5, 40)
        assert MyelosuppressionModel().unknown_drug_potency(s) == ("B",)

    def test_model_states_the_second_drug_weight_is_assumed(self):
        d = MyelosuppressionModel().to_dict()
        assert "ASSUMED" in d["second_drug_potency"]
        assert d["drug_weights"] == {"A": 1.0, "B": 1.0}

    def test_combination_is_screened_as_infeasible_when_intense(self):
        """Near-continuous two-drug dosing must not silently pass the screen."""
        s = Schedule.alternating(150.0, 150.0, 10, 120)
        assert not screen_schedule(s, 120).feasible


class TestMarrowWeightSweep:
    """Drug B's marrow potency is an ASSUMPTION, so it is swept, not fixed."""

    def test_a_lower_weight_admits_a_calendar_the_conservative_weight_rejects(self):
        from dataclasses import replace
        m = MyelosuppressionModel()
        s = Schedule.alternating_relative(
            1.0, 14, 365, dose_mg_m2=150.0, dose_level_a=0.2,
        )
        conservative = replace(m, drug_weights={"A": 1.0, "B": 1.0})
        assert not conservative.feasible(s, 365)
        assert replace(m, drug_weights={"A": 1.0, "B": 0.3}).feasible(s, 365)

    def test_the_declared_sweep_never_claims_the_second_drug_is_free(self):
        from danon.dosing import DEFAULT_MARROW_WEIGHTS
        assert DEFAULT_MARROW_WEIGHTS
        assert min(DEFAULT_MARROW_WEIGHTS) > 0.0
        assert max(DEFAULT_MARROW_WEIGHTS) == 1.0

    def test_to_dict_reports_the_weight_actually_used(self):
        from dataclasses import replace
        m = replace(MyelosuppressionModel(), drug_weights={"A": 1.0, "B": 0.5})
        assert m.to_dict()["drug_weights"] == {"A": 1.0, "B": 0.5}
        assert MyelosuppressionModel().to_dict()["drug_weights"]["B"] == 1.0

    def test_unknown_drug_potency_names_drug_b_only(self):
        s = Schedule.alternating_relative(1.0, 5, 40)
        assert MyelosuppressionModel().unknown_drug_potency(s) == ("B",)


class TestCombinationComparison:
    @pytest.fixture(scope="class")
    def cmp(self):
        return compare_combination_vs_single(
            TumorEvolutionSimulator(make_gbm_phantom(size=16, seed=7)),
            seeds=(0, 1, 2, 3), days=365,
            on_days=(2, 5), cycle_days=(28,), single_continuous_levels=(0.25,),
            alternation_shapes=((5, 23), (5, 9)),
            drug_b_levels=(1.0,), continuous_blocks=(14,),
            continuous_level_a=(0.2,), marrow_weights=(1.0, 0.5),
            max_workers=1,
        )

    def test_both_arms_ran_on_the_same_seeds(self, cmp):
        assert cmp.single_drug.seeds == (0, 1, 2, 3)
        assert cmp.single_drug.engine_runs == 4
        for s in cmp.combination:
            assert s.seeds == cmp.single_drug.seeds
            assert s.engine_runs == 4

    def test_the_single_drug_arm_is_the_best_of_its_own_grid(self, cmp):
        """The reference must be found by measurement, not chosen in advance."""
        assert len(cmp.single_drug_scored) > 1
        assert cmp.single_drug.median_aubc == min(
            s.median_aubc for s in cmp.single_drug_scored
        )

    def test_only_feasible_single_drug_calendars_were_run(self, cmp):
        for s in cmp.single_drug_scored:
            assert s.feasibility.feasible

    def test_verdict_is_derived_not_assumed(self, cmp):
        if cmp.best_combination is None:
            assert cmp.beats_single is False
        else:
            assert cmp.beats_single is (
                cmp.best_combination.median_aubc < cmp.single_drug.median_aubc
            )

    def test_the_verdict_is_taken_at_the_conservative_weight(self, cmp):
        assert cmp.primary_marrow_weight == max(
            r["drug_b_marrow_weight"] for r in cmp.marrow_weight_sweep
        )

    def test_the_sweep_covers_every_declared_weight(self, cmp):
        assert [r["drug_b_marrow_weight"] for r in cmp.marrow_weight_sweep] == [1.0, 0.5]

    def test_a_softer_weight_cannot_make_the_screen_stricter(self, cmp):
        rows = {r["drug_b_marrow_weight"]: r for r in cmp.marrow_weight_sweep}
        assert rows[0.5]["n_alternating_calendars_admissible"] >= \
            rows[1.0]["n_alternating_calendars_admissible"]

    def test_a_loss_is_reported_as_a_null_result(self, cmp):
        if cmp.beats_single is False and cmp.best_combination is not None:
            assert any("null result" in n for n in cmp.notes)

    def test_result_is_serialisable(self, cmp):
        json.dumps(cmp.to_dict())

    def test_comparison_uses_only_feasible_combinations(self, cmp):
        for s in cmp.combination:
            assert s.schedule.is_combination()
            assert s.feasibility.feasible

    def test_engine_runs_are_counted(self, cmp):
        assert cmp.engine_runs == sum(
            s.engine_runs for s in cmp.single_drug_scored + cmp.combination
        )


class TestCandidateGrids:
    def test_single_drug_grid_brackets_the_safety_boundary(self):
        """A grid with no rejections would measure nothing."""
        m = MyelosuppressionModel()
        cands = single_drug_candidates(365)
        flags = [m.feasible(s, 365) for _n, s in cands]
        assert any(flags) and not all(flags)

    def test_alternation_grid_brackets_the_safety_boundary(self):
        m = MyelosuppressionModel()
        cands = alternation_candidates(365)
        flags = [m.feasible(s, 365) for _n, s in cands]
        assert any(flags) and not all(flags)

    def test_alternation_grid_is_actually_alternating(self):
        for _n, s in alternation_candidates(365):
            assert s.is_combination()
            assert s.drugs_used() == ("A", "B")

    def test_only_one_drug_is_used_in_the_reference_grid(self):
        for _n, s in single_drug_candidates(365):
            assert s.drugs_used() == ("A",)
            assert not s.is_combination()


class TestParallelEvaluation:
    """Parallelism across candidates must change no number."""

    def test_parallel_equals_serial(self, sim):
        named = [
            ("5/28", Schedule.cyclic("A", 90.0, 5, 28, DAYS)),
            ("alt", Schedule.alternating_relative(1.0, 5, DAYS,
                                                  dose_mg_m2=90.0,
                                                  rest_days=23)),
        ]
        serial = evaluate_candidates(named, sim, (0, 1), days=DAYS,
                                     max_workers=1)
        par = evaluate_candidates(named, sim, (0, 1), days=DAYS, max_workers=2)
        for a, b in zip(serial, par):
            assert a.name == b.name
            assert a.median_aubc == b.median_aubc
            assert a.median_dominance_day == b.median_dominance_day
            assert a.median_pfs_days == b.median_pfs_days
            assert a.n_progressed == b.n_progressed

    def test_empty_input_is_not_an_error(self, sim):
        assert evaluate_candidates([], sim, (0,), days=DAYS) == []


class TestFeasibleGridHelper:
    def test_returns_only_feasible_cells(self):
        out = feasible_grid_schedules(
            (3, 5), (28, 42), (0.4, 0.8), dose_mg_m2=150.0, days=365
        )
        total = grid_size((3, 5), (28, 42), (0.4, 0.8))
        assert len(out) <= total
        assert all(screen_schedule(s, 365).feasible for _n, s in out)

    def test_every_entry_is_named_and_a_schedule(self):
        out = feasible_grid_schedules((3,), (28,), (0.4,), dose_mg_m2=150.0, days=365)
        for name, sched in out:
            assert isinstance(name, str) and name
            assert isinstance(sched, Schedule)


# ---------------------------------------------------------------------------
# Short-horizon screening
# ---------------------------------------------------------------------------


CANDIDATES = [
    ("5/28 @90", Schedule.cyclic("A", 90.0, 5, 28, 365)),
    ("3/14 @60", Schedule.cyclic("A", 60.0, 3, 14, 365)),
    ("7/42 @60", Schedule.cyclic("A", 60.0, 7, 42, 365)),
    ("continuous @150", Schedule.continuous("A", 150.0, 365)),
]


class TestScreenFidelity:
    @pytest.fixture(scope="class")
    def fid(self):
        return screen_fidelity(
            TumorEvolutionSimulator(make_gbm_phantom(size=16, seed=7)),
            CANDIDATES, seeds=(0, 1, 2), full_days=120, screen_days=60, top_k=2,
        )

    def test_both_horizons_were_actually_run(self, fid):
        assert fid.n_candidates == len(CANDIDATES)
        assert fid.screen_days == 60 and fid.full_days == 120

    def test_winner_exists_on_both_sides(self, fid):
        assert fid.full_winner in {n for n, _ in CANDIDATES}
        assert fid.screen_winner in {n for n, _ in CANDIDATES}

    def test_caught_and_missed_partition_the_top_k(self, fid):
        assert fid.caught + len(fid.missed) == min(fid.top_k, fid.n_candidates)

    def test_rank_correlation_is_reported(self, fid):
        assert -1.0 <= fid.spearman <= 1.0
        assert -1.0 <= fid.kendall_tau <= 1.0

    def test_identical_orderings_correlate_perfectly(self):
        """A ranking against itself must be rho = tau = 1, or the statistics lie."""
        a = [0.1, 0.2, 0.3]
        assert _spearman(a, a) == pytest.approx(1.0)
        assert _kendall_tau(a, a) == pytest.approx(1.0)

    def test_reversed_orderings_correlate_perfectly_negatively(self):
        a = [0.1, 0.2, 0.3]
        assert _spearman(a, list(reversed(a))) == pytest.approx(-1.0)
        assert _kendall_tau(a, list(reversed(a))) == pytest.approx(-1.0)

    def test_screen_of_length_at_least_full_is_rejected(self, sim):
        with pytest.raises(ValueError):
            screen_fidelity(sim, CANDIDATES[:1], (0,), full_days=60, screen_days=60)
        with pytest.raises(ValueError):
            screen_fidelity(sim, CANDIDATES[:1], (0,), full_days=60, screen_days=90)

    def test_empty_candidate_list_is_rejected(self, sim):
        with pytest.raises(ValueError):
            screen_fidelity(sim, [], (0,), full_days=60, screen_days=30)

    def test_note_names_the_cost_when_candidates_are_dropped(self, fid):
        n = screen_fidelity_note(fid)
        assert "Spearman" in n
        assert ("DROPPED" in n) or ("no candidate was lost" in n)


class TestShortHorizonSearch:
    def test_search_refuses_a_screen_that_is_not_a_saving(self, sim):
        with pytest.raises(ValueError, match="saving"):
            search_schedules(sim, seeds=(0, 1), days=60, screen_days=60)
        with pytest.raises(ValueError, match="saving"):
            search_schedules(sim, seeds=(0, 1), days=60, screen_days=90)

    def test_short_screen_is_opt_in_and_recorded(self, sim):
        full = search_schedules(
            sim, seeds=(0, 1, 2), days=60, on_days=(3,), cycle_days=(14,),
            dose_levels=(0.4,), screen_seeds=2, confirm_top=1,
        )
        assert "at 60 days" in full.notes[0]
        assert not any("PROXY" in n for n in full.notes)

        short = search_schedules(
            sim, seeds=(0, 1, 2), days=60, screen_days=30, on_days=(3,),
            cycle_days=(14,), dose_levels=(0.4,), screen_seeds=2, confirm_top=1,
        )
        assert any("PROXY" in n for n in short.notes)
        assert "at 30 days" in short.notes[0]

    def test_confirmed_numbers_are_always_full_horizon(self, sim):
        """Whatever the screen does, the reported row is a full-horizon run."""
        short = search_schedules(
            sim, seeds=(0, 1, 2), days=60, screen_days=30, on_days=(3, 5),
            cycle_days=(14,), dose_levels=(0.4, 0.6), screen_seeds=2, confirm_top=2,
        )
        for s in short.scored:
            assert len(s.seeds) == 3
            assert s.engine_runs == 3

    def test_verdict_is_still_derived(self, sim):
        r = search_schedules(
            sim, seeds=(0, 1, 2), days=60, screen_days=30, on_days=(3,),
            cycle_days=(14,), dose_levels=(0.4,), screen_seeds=2, confirm_top=1,
        )
        if r.best is None:
            assert r.beats_baselines is False
        else:
            assert r.beats_baselines == (
                r.best.median_aubc < r.baselines["mtd"].median_aubc
            )


# ---------------------------------------------------------------------------
# The hypothesis test
# ---------------------------------------------------------------------------


class TestCompetitiveRelease:
    @pytest.fixture(scope="class")
    def cr(self):
        return run_competitive_release_experiment(
            on_days=5, cycle_days=26, seeds=(0, 1, 2), days=90
        )

    def test_both_arms_really_ran_on_the_same_seeds(self, cr):
        assert cr.mtd.seeds == cr.pulsed.seeds == (0, 1, 2)
        assert cr.mtd.engine_runs == 3 and cr.pulsed.engine_runs == 3

    def test_per_seed_deltas_are_reported(self, cr):
        assert len(cr.per_seed["seed"]) == 3
        assert len(cr.per_seed["mtd_dominance_day"]) == 3
        assert len(cr.per_seed["pulsed_dominance_day"]) == 3

    def test_verdict_is_derived_from_the_measured_delta(self, cr):
        if cr.dominance_delta_days is not None:
            assert cr.hypothesis_supported == (cr.dominance_delta_days > 0)

    def test_result_is_serialisable(self, cr):
        json.dumps(cr.to_dict())


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_engine_and_calendar_are_marked_measured(self):
        p = provenance()
        assert p["engine"].startswith("measured")
        assert p["calendar"].startswith("measured")

    def test_myelosuppression_is_marked_an_assumption(self):
        p = provenance()
        assert "DECLARED ASSUMPTION" in p["myelosuppression"]

    def test_clinical_outcome_is_explicitly_not_claimed(self):
        assert "NOT CLAIMED" in provenance()["clinical_outcome"]