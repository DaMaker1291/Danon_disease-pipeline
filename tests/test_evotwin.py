"""Tests for the EvoTwin cohort module.

Pins: the Kaplan-Meier estimator against analytic values, the evolutionary
trap's two-jaw mechanism on the reference phantom, the cohort-level
failure-mode asymmetry (MTD+immune controls burden but not the clone;
adaptive+immune eliminates the resistant clone), and the honest provenance
(CheckMate 143 failure, immune synergy as UNKNOWN hypothesis).
"""

from __future__ import annotations

import numpy as np
import pytest

from danon.cure import EvidenceStatus
from danon.evosteer import make_gbm_phantom
from danon.evotwin import (  # re-exports MTDPolicy / AdaptiveDosePolicy
    CHECKMATE143_HR,
    CHECKMATE143_MOS_MONTHS,
    AdaptiveDosePolicy,
    CohortParams,
    MTDPolicy,
    cohort_km_curve,
    evolutionary_trap_analysis,
    evotwin_provenance,
    kaplan_meier,
    km_at,
    km_median,
    run_evotwin_cohort,
    what_this_model_cannot_know,
)


# ---------------------------------------------------------------------------
# Kaplan-Meier estimator
# ---------------------------------------------------------------------------


def test_km_analytic_reference():
    # times [1,2,3], events [1,1,0]: S = 2/3 at t=1, 1/3 at t=2, holds at t=3.
    km = kaplan_meier([1, 2, 3], [1, 1, 0])
    S = dict(zip(km["t"], km["S"]))
    assert abs(S[1.0] - 2 / 3) < 1e-12
    assert abs(S[2.0] - 1 / 3) < 1e-12
    assert abs(S[2.0] - 1 / 3) < 1e-12
    assert abs(S[0.0] - 1.0) < 1e-12  # curve starts at 1
    # Censoring creates no step: S holds at 1/3 through t=3.
    assert abs(km_at(km, 3.0)[0] - 1 / 3) < 1e-12
    # Survival is non-increasing; Greenwood SE positive after first event.
    assert all(s2 <= s1 for s1, s2 in zip(km["S"], km["S"][1:]))
    assert km["se"][1] > 0
    assert km_median(km) == 2.0
    s, se = km_at(km, 2.5)
    assert abs(s - 1 / 3) < 1e-12 and se > 0
    # Before the first event: S=1, se=0.
    s0, se0 = km_at(km, 0.5)
    assert s0 == 1.0 and se0 == 0.0


def test_km_edge_cases_and_validation():
    # All censored: survival stays at 1 and median is never reached.
    km = kaplan_meier([5, 5, 5], [0, 0, 0])
    assert all(abs(s - 1.0) < 1e-12 for s in km["S"])
    assert km_median(km) is None
    # All events: survival hits exactly 0 at the last event.
    km3 = kaplan_meier([1, 1, 1], [1, 1, 1])
    assert km3["S"][-1] == 0.0
    with pytest.raises(ValueError):
        kaplan_meier([1, 2], [1])
    with pytest.raises(ValueError):
        kaplan_meier([1, 2], [1, 2])
    with pytest.raises(ValueError):
        kaplan_meier([1, 2], [1, -1])
    with pytest.raises(ValueError):
        kaplan_meier([-1, 2], [1, 0])


def test_cohort_km_curve_censors_at_horizon():
    # Progressed patients become events at their PFS day; controlled ones
    # are censored AT the horizon (never past it).
    km = cohort_km_curve([50.0, 250.0, 250.0], [True, False, False], horizon=250.0)
    assert km["t"][-1] == 250.0
    assert abs(km["S"][-1] - 2 / 3) < 1e-12
    with pytest.raises(ValueError):
        cohort_km_curve([], [], horizon=100.0)


# ---------------------------------------------------------------------------
# Evolutionary trap (single tumour, reference phantom)
# ---------------------------------------------------------------------------


def test_evolutionary_trap_two_jaw_mechanism():
    gbm = make_gbm_phantom(size=16, seed=3)
    trap = evolutionary_trap_analysis(gbm, days=400, seed=7)
    # Jaw 1 alone (competition, no immune): BOTH policies relapse resistant.
    assert trap.immune_off_mtd_relapse_resistant is True
    assert trap.immune_off_adaptive_relapse_resistant is True
    # Jaw 2 (immune) plus steering: the resistant clone is ELIMINATED.
    assert trap.immune_on_adaptive_r_eliminated is True
    assert trap.trap_completed is True
    assert "hypothesis" in " ".join(trap.notes)


# ---------------------------------------------------------------------------
# Cohort runs (small, fast cohort; module-scoped to share across asserts)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cohort():
    return CohortParams(size=6, tumor_size_range=(12, 14), seed0=500)


@pytest.fixture(scope="module")
def cohort_no_immune(small_cohort):
    return run_evotwin_cohort(
        arms=[MTDPolicy(), AdaptiveDosePolicy()],
        cohort=small_cohort,
        sim_days=250,
        immune=None,
    )


@pytest.fixture(scope="module")
def cohort_immune(small_cohort):
    return run_evotwin_cohort(
        arms=[MTDPolicy(), AdaptiveDosePolicy()],
        cohort=small_cohort,
        sim_days=250,
    )


def test_no_immune_cohort_everyone_progresses_resistant(cohort_no_immune):
    rep = cohort_no_immune
    for name, arm in rep.arms.items():
        assert arm.progressed_frac == 1.0
        # Without the immune layer, every relapse is drug-resistant.
        assert arm.relapse_resistant_frac == 1.0
    # Steering extends median PFS (direction of the mCRPC anchor).
    assert (
        rep.arms["adaptive_dose"].median_pfs_days
        >= rep.arms["mtd"].median_pfs_days
    )
    # With 12 events, the log-rank comparison is estimable.
    assert rep.logrank is not None
    assert 0.0 <= rep.logrank["logrank_p_value"] <= 1.0


def test_immune_cohort_failure_mode_asymmetry(cohort_immune):
    rep = cohort_immune
    for name, arm in rep.arms.items():
        # The immune cohort controls burden across the board...
        assert arm.progressed_frac == 0.0
    mtd = rep.arms["mtd"]
    adp = rep.arms["adaptive_dose"]
    # ...but ONLY steering controls the CLONE: MTD+immune 'controlled'
    # patients sit on fully drug-resistant tumours; adaptive+immune
    # patients have the resistant clone essentially eliminated.
    # These are CLONE COMPOSITIONS (ungated), because the claim is about what
    # the surviving population is made of -- the gated clinical fraction reads
    # 0.0 for both arms here, since both are held below the detection floor,
    # and that gating is exactly what stops 'eradicated' from being read as
    # 'resistance took over'.
    assert mtd.median_final_R_controlled is not None
    assert mtd.median_final_R_controlled > 0.9
    assert adp.median_final_R_controlled is not None
    assert adp.median_final_R_controlled < 0.1
    # Near-total control leaves too few events for log-rank: reported as
    # not estimable, never silently dropped.
    assert rep.logrank is None
    assert any("not estimable" in n for n in rep.notes)


def test_km_median_reported_when_reached(cohort_no_immune):
    # Everyone progresses inside the window -> KM medians exist.
    for name, med in cohort_no_immune.km_median_by_arm.items():
        assert med is not None, f"{name}: median should be reached"
        assert 0 < med <= 250


# ---------------------------------------------------------------------------
# Provenance and honest limits
# ---------------------------------------------------------------------------


def test_provenance_pins_checkmate_and_hypothesis():
    prov = evotwin_provenance()
    # Checkpoint monotherapy in GBM failed: pinned as PROVEN_FAILED.
    assert prov["checkpoint_monotherapy_gbm"] is EvidenceStatus.PROVEN_FAILED
    assert CHECKMATE143_MOS_MONTHS == (9.8, 10.0)
    assert CHECKMATE143_HR == 1.04
    # The steering+immune combination is a hypothesis, not evidence.
    assert prov["adaptive_plus_immunotherapy_synergy"] is EvidenceStatus.UNKNOWN
    # And the trap mechanism itself is an in-silico result.
    assert prov["evolutionary_trap_two_jaw_mechanism"] is EvidenceStatus.UNKNOWN


def test_cannot_know_extends_evosteer():
    limits = what_this_model_cannot_know()
    from danon.evosteer import what_this_model_cannot_know as base_limits

    assert len(limits) > len(base_limits())
    joined = " ".join(limits)
    assert "antigen-loss" in joined
    assert "exhaustion" in joined
    assert "ASSUMED" in joined
