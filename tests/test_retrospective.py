"""Tests for the real-clinical-outcome ingestion and plausibility check.

Pins: cohort reconstruction rules (earliest dated progression is the event,
last follow-up censors it), the survival step lookup, the anchor check on a
fixture that is deliberately off-anchor, the refusal paths, the
non-validation language, and the campaign-log to (times, events) conversion
including its censoring-window refusal.
"""

from __future__ import annotations

import json
import os

import pytest

from danon.retrospective import (
    GDC_CLINICAL_PATH,
    PUBLISHED_GBM_OS_BAND_DAYS,
    anchor_check,
    load_campaign_log,
    load_tcga_gbm_clinical,
    model_curve_from_campaign_log,
    plausibility_check,
    reconstruct_cohort,
    retrospective_provenance,
    what_this_check_cannot_do,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _payload():
    return {
        "source": "fixture payload",
        "hits": [
            {  # progressed, died -> both endpoints eventful
                "submitter_id": "P1",
                "follow_ups": [
                    {"progression_or_recurrence": "Yes", "days_to_progression": 150.0,
                     "days_to_follow_up": 160.0},
                    {"progression_or_recurrence": "Yes", "days_to_progression": 90.0,
                     "days_to_follow_up": 100.0},
                ],
                "demographic": {"vital_status": "Dead", "days_to_death": 400.0},
            },
            {  # never progressed, alive -> censored on both
                "submitter_id": "P2",
                "follow_ups": [{"progression_or_recurrence": "No", "days_to_follow_up": 275.0}],
                "demographic": {"vital_status": "Alive"},
            },
            {  # progressed, alive -> event on PFS, censored on OS
                "submitter_id": "P3",
                "follow_ups": [
                    {"progression_or_recurrence": "Yes", "days_to_progression": 320.0,
                     "days_to_follow_up": 330.0}
                ],
                "demographic": {"vital_status": "Alive"},
            },
            {  # no follow-up data at all -> in n_cases, in no curve
                "submitter_id": "P4",
                "demographic": {},
            },
        ],
    }


@pytest.fixture()
def cohort():
    return reconstruct_cohort(_payload(), source="fixture payload")


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def test_reconstruction_rules(cohort):
    assert cohort.n_cases == 4
    # P1 (earliest dated progression wins), P3 eventful; P2 censored
    assert cohort.progression_events == (1, 0, 1)
    assert cohort.progression_times == (90.0, 275.0, 320.0)
    assert cohort.n_progression_events == 2
    assert cohort.n_progression_censored == 1
    # OS: P1 event, P2 and P3 censored at last follow-up
    assert cohort.os_events == (1, 0, 0)
    assert cohort.os_times == (400.0, 275.0, 330.0)
    # the case with no follow-up contributed to n_cases but to no curve
    assert len(cohort.cases) == 4
    assert cohort.cases[3].progression_time is None
    assert cohort.cases[3].os_time is None


def test_event_iqr_ignores_censored_times(cohort):
    lo, hi = cohort.progression_event_iqr()
    # only two dated events exist (90 and 320), so the quartiles are the
    # linear interpolation between them -- the point of the test is that the
    # censored 275.0 does NOT enter this distribution
    assert lo == pytest.approx(147.5)
    assert hi == pytest.approx(262.5)
    assert lo > 90.0 - 1e-9 and hi < 320.0 + 1e-9


def test_step_lookup(cohort):
    assert cohort.survival_at(cohort.km_progression, 0.0) == pytest.approx(1.0)
    assert cohort.survival_at(cohort.km_progression, 90.0) == pytest.approx(2 / 3)
    assert cohort.survival_at(cohort.km_progression, 300.0) == pytest.approx(2 / 3)
    # at the last event time one subject is still at risk, so the curve drops
    # to zero there -- the censored case does not change that
    assert cohort.survival_at(cohort.km_progression, 320.0) == pytest.approx(0.0)


def test_reconstruction_refuses_empty_payloads():
    with pytest.raises(ValueError, match="no usable progression records"):
        reconstruct_cohort({"hits": [{"submitter_id": "X", "demographic": {}}]}, "empty")
    with pytest.raises(ValueError, match="no usable survival records"):
        reconstruct_cohort(
            {
                "hits": [
                    {
                        "submitter_id": "X",
                        "follow_ups": [{"progression_or_recurrence": "Yes",
                                        "days_to_progression": 10.0}],
                        "demographic": {},
                    }
                ]
            },
            "no-os",
        )


# ---------------------------------------------------------------------------
# Anchor check
# ---------------------------------------------------------------------------


def test_anchor_check_flags_an_off_anchor_reconstruction(cohort):
    inside = anchor_check(cohort, band_days=(300.0, 600.0))
    assert inside.ok
    assert "INSIDE" in inside.message()

    outside = anchor_check(cohort, band_days=(10.0, 20.0))
    assert not outside.ok
    assert "OUTSIDE" in outside.message()


def test_anchor_check_handles_an_unreached_median():
    payload = {
        "hits": [
            {
                "submitter_id": "A",
                "follow_ups": [{"progression_or_recurrence": "No",
                                "days_to_follow_up": 30.0}],
                "demographic": {"vital_status": "Alive"},
            },
            {
                "submitter_id": "B",
                "follow_ups": [{"progression_or_recurrence": "Yes",
                                "days_to_progression": 40.0}],
                "demographic": {"vital_status": "Dead", "days_to_death": 50.0},
            },
        ]
    }
    c = reconstruct_cohort(payload, "small")
    check = anchor_check(c, band_days=(330.0, 550.0))
    assert check.median_os_days == pytest.approx(50.0)
    assert not check.ok


# ---------------------------------------------------------------------------
# Plausibility check
# ---------------------------------------------------------------------------


def test_plausibility_check_reports_gaps_and_never_validates(cohort):
    model_times = [40.0, 60.0, 80.0, 100.0, 365.0]
    model_events = [1, 1, 1, 1, 0]
    chk = plausibility_check(cohort, model_times, model_events, label="unit")
    assert chk.model_n == 5
    assert chk.model_events == 4
    assert len(chk.eval_table) == 3
    assert chk.max_abs_gap >= 0.0
    assert chk.median_within_event_iqr in (True, False)
    verdict = chk.verdict()
    assert "not validation" in verdict
    assert "no patient matching" in verdict
    table = chk.summary_table()
    assert "PLAUSIBILITY CHECK" in table
    assert "max |gap|" in table


def test_plausibility_check_refuses_an_empty_model_arm(cohort):
    with pytest.raises(ValueError, match="non-empty"):
        plausibility_check(cohort, [], [])
    with pytest.raises(ValueError, match="events must be 0/1"):
        plausibility_check(cohort, [10.0], [2])


# ---------------------------------------------------------------------------
# Campaign log -> model arm
# ---------------------------------------------------------------------------


def test_curve_from_campaign_log_tags_censoring():
    entries = [
        {"case_id": "C1", "pfs_mtd_replicates": [200.0, 240.0, 365.0],
         "pfs_policy_replicates": [365.0, 365.0, 365.0]},
        {"case_id": "C2", "pfs_mtd_replicates": [300.0], "pfs_policy_replicates": [280.0]},
    ]
    t, e = model_curve_from_campaign_log(entries, "mtd", horizon_days=365.0)
    assert t == [200.0, 240.0, 365.0, 300.0]
    assert e == [1, 1, 0, 1]
    t, e = model_curve_from_campaign_log(entries, "policy", horizon_days=365.0)
    assert e == [0, 0, 0, 1]
    assert sum(e) == 1


def test_curve_from_campaign_log_refusals():
    with pytest.raises(ValueError, match="arm must be"):
        model_curve_from_campaign_log([], "adaptive")
    with pytest.raises(ValueError, match="has no pfs_mtd_replicates"):
        model_curve_from_campaign_log([{"case_id": "C"}], "mtd")
    with pytest.raises(ValueError, match="exceeds the stated horizon"):
        model_curve_from_campaign_log(
            [{"case_id": "C", "pfs_mtd_replicates": [400.0]}], "mtd", horizon_days=365.0
        )
    with pytest.raises(ValueError, match="no replicate values"):
        model_curve_from_campaign_log([{"case_id": "C", "pfs_mtd_replicates": []}], "mtd")


# ---------------------------------------------------------------------------
# Honesty record
# ---------------------------------------------------------------------------


def test_provenance_and_limits_are_stated():
    prov = retrospective_provenance()
    assert prov["tcga_gbm_clinical_export_real"].value == "established"
    assert prov["model_timescale_matches_patients"].value == "unknown"
    assert prov["this_check_is_clinical_validation"].value == "proven_failed"
    limits = what_this_check_cannot_do()
    assert any("no pair to score" in l for l in limits)
    assert any("trials" not in l for l in limits)


# ---------------------------------------------------------------------------
# Against the real cached export (skipped when the data is not present)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(GDC_CLINICAL_PATH),
    reason="GDC clinical export not downloaded in this checkout",
)
def test_real_cohort_hits_the_published_anchor():
    """The ingestion must recover the published GBM median OS band.

    This is the one real-data assertion in the repo that is allowed to fail
    loudly: if the reconstruction stops hitting a textbook number, the
    data-handling path is broken.
    """
    cohort = load_tcga_gbm_clinical()
    assert cohort.n_cases > 500
    assert cohort.n_progression_events > 100
    check = anchor_check(cohort)
    assert check.ok, check.message()
    assert PUBLISHED_GBM_OS_BAND_DAYS[0] <= check.median_os_days <= PUBLISHED_GBM_OS_BAND_DAYS[1]


@pytest.mark.skipif(
    not os.path.exists(GDC_CLINICAL_PATH) or not os.path.exists("outputs_real/campaign_log.json"),
    reason="needs both the GDC export and a finished real-data campaign",
)
def test_real_plausibility_check_runs_end_to_end():
    cohort = load_tcga_gbm_clinical()
    entries = load_campaign_log()
    times, events = model_curve_from_campaign_log(entries, "mtd")
    chk = plausibility_check(cohort, times, events, label="campaign MTD arm")
    assert chk.model_events > 0
    assert "not validation" in chk.verdict()
    json.dumps(chk.eval_table)  # the table must be serialisable for the poster
