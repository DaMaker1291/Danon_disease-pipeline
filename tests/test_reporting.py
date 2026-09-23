"""Tests for the Phase-3 deliverable module (plots, ledger, validation packet).

Pins: the figure guards (no figure from an empty artifact), the hazard-ratio
refusal when one arm has no events (the degenerate 1e17 artifact), the
single-event log-rank refusal, and that the validation packet is assembled
from the artifacts rather than from literals.
"""

from __future__ import annotations

import json
import os

import pytest

matplotlib = pytest.importorskip("matplotlib")

from danon.reporting import (  # noqa: E402
    campaign_bar_plot,
    km_plot,
    sensitivity_curve_plot,
    validation_packet,
)
from danon.retrospective import reconstruct_cohort  # noqa: E402


def _cohort():
    payload = {
        "source": "fixture",
        "hits": [
            {"submitter_id": "A",
             "follow_ups": [{"progression_or_recurrence": "Yes",
                             "days_to_progression": 120.0, "days_to_follow_up": 130.0}],
             "demographic": {"vital_status": "Dead", "days_to_death": 400.0}},
            {"submitter_id": "B",
             "follow_ups": [{"progression_or_recurrence": "No",
                             "days_to_follow_up": 300.0}],
             "demographic": {"vital_status": "Alive"}},
        ],
    }
    return reconstruct_cohort(payload, "fixture")


def test_km_plot_refuses_hazard_ratio_when_an_arm_has_no_events(tmp_path):
    out = str(tmp_path / "km.png")
    res = km_plot(
        [365.0, 365.0, 365.0], [0, 0, 0],
        [200.0, 240.0, 300.0], [1, 1, 1],
        out,
    )
    assert res["hazard_ratio"] is None
    assert res["policy_events"] == 0 and res["mtd_events"] == 3
    assert os.path.exists(out)


def test_km_plot_reports_hazard_ratio_when_both_arms_event(tmp_path):
    res = km_plot(
        [300.0, 365.0, 365.0], [1, 0, 0],
        [150.0, 200.0, 250.0], [1, 1, 1],
        str(tmp_path / "km2.png"),
    )
    assert res["hazard_ratio"] is not None


def test_km_plot_refuses_when_the_logrank_is_not_estimable(tmp_path):
    with pytest.raises(ValueError, match="not estimable"):
        km_plot(
            [365.0, 365.0], [0, 0],
            [300.0, 360.0], [0, 0],
            str(tmp_path / "km3.png"),
        )


def test_campaign_bar_plot_refuses_an_empty_log(tmp_path):
    with pytest.raises(ValueError, match="empty campaign figure"):
        campaign_bar_plot([], str(tmp_path / "cells.png"))


def test_campaign_bar_plot_renders_from_log_records(tmp_path):
    records = [
        {"case_id": "BraTS2021_01017", "r0": 0.127,
         "pfs_extension_median": 0.44, "toxicity_reduction_median": 0.154},
        {"case_id": "BraTS2021_01018", "r0": 0.236,
         "pfs_extension_median": 0.11, "toxicity_reduction_median": 0.117},
    ]
    out = str(tmp_path / "cells.png")
    assert campaign_bar_plot(records, out) == out
    assert os.path.getsize(out) > 0


def test_sensitivity_curve_plot_refuses_unknown_case(tmp_path):
    with pytest.raises(ValueError, match="no sensitivity records"):
        sensitivity_curve_plot(
            [{"case_id": "X", "target": "collateral_b_kill", "scale": 1.0,
              "burden_ratio": 0.5}],
            str(tmp_path / "s.png"),
            case_id="Y",
        )


def test_sensitivity_curve_plot_renders(tmp_path):
    records = [
        {"case_id": "X", "target": "collateral_b_kill", "scale": s,
         "burden_ratio": r}
        for s, r in ((0.02, 0.91), (0.10, 0.26), (1.0, 0.02))
    ]
    out = str(tmp_path / "s.png")
    assert sensitivity_curve_plot(records, out, case_id="X") == out
    assert os.path.getsize(out) > 0


def test_validation_packet_reads_numbers_from_artifacts(tmp_path):
    cohort = _cohort()
    out = str(tmp_path / "packet.txt")
    text = validation_packet(
        cohort,
        [90.0, 150.0, 365.0],
        [1, 1, 0],
        sensitivity_records=[
            {"target": "collateral_b_kill", "scale": 1.0, "burden_ratio": 0.02,
             "policy_wins": True},
            {"target": "collateral_b_kill", "scale": 0.02, "burden_ratio": 1.4,
             "policy_wins": False},
        ],
        test_count=573,
        out_path=out,
    )
    assert os.path.exists(out)
    assert f"n={len(cohort.progression_times)}" in text
    assert "INSIDE" in text or "OUTSIDE" in text
    assert "LAYER 3" in text
    assert "1/2" in text  # one of the two supplied cells is a win
    assert "LOSES" in text
    assert "573" in text
    assert "WHAT THIS PACKET DOES NOT SHOW" in text
    json.dumps(text)  # trivially serialisable, but keep the contract explicit


def test_validation_packet_states_when_a_layer_is_missing(tmp_path):
    text = validation_packet(_cohort(), [90.0], [1], out_path=str(tmp_path / "p2.txt"))
    assert "no sensitivity records supplied" in text
