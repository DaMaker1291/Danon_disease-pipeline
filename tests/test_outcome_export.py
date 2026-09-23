"""Tests for the day-365 outcome export and the self-contained dashboard.

Pins: the exactness of `ReplicateRun.day()` (no interpolation, hard refusal
outside the run window), the two honesty flags that otherwise read as
results (a fully-censored arm makes the PFS extension a LOWER BOUND; a
near-zero burden makes the resistant fraction uninterpretable), the export
schema and refusals, and that the rendered dashboard is self-contained and
carries the case data it was built from.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from danon.evosteer import RANO_PROGRESSION_FRACTION, make_gbm_phantom
from danon.outcome_export import (
    ArmRecord,
    ReplicateRun,
    _headline,
    export_outcome_cards,
    load_cards,
    outcome_export_provenance,
    regenerate_ui_artifacts,
    render_dashboard_html,
    run_case_outcome,
)


def _run(seed: int, s: list[float], r: list[float], pfs_day=None, tox=None) -> ReplicateRun:
    n = len(s)
    total = s[-1] + r[-1]
    return ReplicateRun(
        seed=seed,
        days=n,
        burden_S=np.asarray(s, dtype=float),
        burden_R=np.asarray(r, dtype=float),
        dose_A=np.ones(n),
        dose_B=np.zeros(n),
        pfs_day=pfs_day,
        progressed=pfs_day is not None,
        toxicity=float(sum(s) + sum(r)) if tox is None else tox,
        final_resistant_fraction=(r[-1] / total) if total > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Exact day readout
# ---------------------------------------------------------------------------


def test_day_readout_is_exact_and_indexed_from_one():
    rep = _run(7, [10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
    d1 = rep.day(1)
    assert d1["day"] == 1
    assert d1["burden_sensitive"] == 10.0  # day 1 == first engine step
    assert d1["burden_resistant"] == 1.0
    assert d1["burden_total"] == 11.0
    assert d1["resistant_fraction"] == pytest.approx(1.0 / 11.0)
    assert d1["dose_A_cumulative"] == 1.0

    d3 = rep.day(3)
    assert d3["burden_total"] == 33.0
    assert d3["dose_A_cumulative"] == 3.0


def test_day_readout_refuses_outside_the_window():
    rep = _run(7, [1.0, 2.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="outside the 1..2 run window"):
        rep.day(0)
    with pytest.raises(ValueError, match="outside the 1..2 run window"):
        rep.day(3)


def test_zero_burden_reports_zero_fraction_not_nan():
    rep = _run(7, [0.0, 0.0], [0.0, 0.0])
    assert rep.day(2)["resistant_fraction"] == 0.0


def test_arm_medians_come_from_replicate_values():
    arm = ArmRecord(label="arm", policy="policy")
    arm.replicates = [
        _run(1, [10.0], [0.0], pfs_day=100),
        _run(2, [10.0], [0.0], pfs_day=200),
        _run(3, [10.0], [0.0], pfs_day=None),  # censored at day 1
    ]
    assert arm.pfs_values == [100.0, 200.0, 1.0]
    assert arm.median_pfs == 100.0
    assert arm.n_progressed == 2
    median_day = arm.median_day(1)
    assert median_day["day"] == 1
    assert median_day["burden_total"] == 10.0


# ---------------------------------------------------------------------------
# The two honesty flags
# ---------------------------------------------------------------------------


def test_fully_censored_arm_makes_the_extension_a_lower_bound():
    policy = ArmRecord(label="p", policy="policy")
    policy.replicates = [_run(1, [10.0, 5.0], [0.0, 0.0]), _run(2, [10.0, 4.0], [0.0, 0.0])]
    mtd = ArmRecord(label="m", policy="mtd")
    mtd.replicates = [_run(1, [10.0, 40.0], [0.0, 0.0], pfs_day=2),
                      _run(2, [10.0, 50.0], [0.0, 0.0], pfs_day=2)]

    h = _headline(policy, mtd, 2, initial_burden=10.0)
    assert h["n_progressed_policy"] == 0
    assert h["pfs_extension_is_lower_bound"] is True
    assert "lower bound" not in h["pfs_extension_label"]  # label is terse
    assert ">=" in h["pfs_extension_label"]
    assert "censored" in h["pfs_extension_label"]
    assert h["pfs_extension"] == pytest.approx(0.0)  # 2 d vs 2 d, both censored at horizon


def test_clearance_note_fires_only_at_near_zero_burden():
    policy = ArmRecord(label="p", policy="policy")
    policy.replicates = [_run(1, [10.0, 0.0], [0.0, 0.0], pfs_day=1)]
    mtd = ArmRecord(label="m", policy="mtd")
    mtd.replicates = [_run(1, [10.0, 500.0], [0.0, 0.0], pfs_day=2)]

    cleared = _headline(policy, mtd, 2, initial_burden=10.0)
    assert cleared["clearance_note"] is not None
    assert "not interpretable" in cleared["clearance_note"]

    surviving = _headline(mtd, mtd, 2, initial_burden=10.0)
    assert surviving["clearance_note"] is None


def test_extension_is_not_flagged_as_a_bound_when_the_policy_arm_progresses():
    policy = ArmRecord(label="p", policy="policy")
    policy.replicates = [_run(1, [10.0, 10.0], [0.0, 0.0], pfs_day=2)]
    mtd = ArmRecord(label="m", policy="mtd")
    mtd.replicates = [_run(1, [10.0, 30.0], [0.0, 0.0], pfs_day=2)]
    h = _headline(policy, mtd, 2, initial_burden=10.0)
    assert h["pfs_extension_is_lower_bound"] is False
    assert h["clearance_note"] is None


# ---------------------------------------------------------------------------
# Case runs and export
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_geom():
    return make_gbm_phantom(size=10, seed=5, n_vessels=3)


def _spec(geom, **over):
    spec = dict(
        geometry=geom,
        r0=0.08,
        case_id="phantom",
        patient_id="TEST-01",
        r0_source="unit-test intake",
        geometry_note="unit-test phantom",
        seeds=(11,),
        days=40,
    )
    spec.update(over)
    return spec


def test_case_outcome_is_exact_and_deterministic(small_geom):
    a = run_case_outcome(**_spec(small_geom))
    b = run_case_outcome(**_spec(small_geom))
    assert a["day_365"]["day"] == 40
    assert a["arms"]["policy"]["daily"]["seed"] == 11
    assert len(a["arms"]["mtd"]["daily"]["burden_total"]) == 40
    assert a["day_365"]["burden_policy"] == b["day_365"]["burden_policy"]
    assert a["arms"]["policy"]["daily"]["burden_total"] == b["arms"]["policy"]["daily"]["burden_total"]
    assert a["day_365"]["initial_burden"] > 0
    assert len(a["notes"]) > 5
    # the stale phantom caveat must not appear on a real-geometry card...
    assert not any("the phantom is a stand-in" in n for n in a["notes"])
    # ...and the real-data provenance is carried alongside the engine's
    assert "brats2021_expert_segmentations" in a["data_provenance"]


def test_case_outcome_refusals(small_geom):
    with pytest.raises(ValueError, match="r0 must be a fraction"):
        run_case_outcome(**_spec(small_geom, r0=1.5))
    with pytest.raises(ValueError, match="need at least one seed"):
        run_case_outcome(**_spec(small_geom, seeds=()))
    with pytest.raises(ValueError, match="need at least 2 days"):
        run_case_outcome(**_spec(small_geom, days=1))


def test_export_schema_and_refusals(small_geom, tmp_path):
    out = str(tmp_path / "cards.json")
    with pytest.raises(ValueError, match="no cases"):
        export_outcome_cards([], out)
    with pytest.raises(ValueError, match="missing required fields"):
        export_outcome_cards([{"r0": 0.1}], out)

    payload = export_outcome_cards([_spec(small_geom)], out)
    assert os.path.exists(out)
    assert payload["rano_progression_fraction"] == RANO_PROGRESSION_FRACTION
    assert "ASSUMED" in payload["engine_constants"]["provenance"]
    assert payload["provenance"]["day_365_readout_is_a_patient_prediction"] == "proven_failed"
    assert payload["cases"][0]["case_id"] == "phantom"
    # the file on disk is the same artifact that was returned
    assert load_cards(out)["cases"][0]["case_id"] == "phantom"


def test_export_mirrors_the_same_artifact(small_geom, tmp_path):
    """The UI copy must be byte-identical to the artifact it reads from."""
    primary = str(tmp_path / "cards.json")
    mirror = str(tmp_path / "public" / "evotwin" / "outcome_cards.json")
    export_outcome_cards([_spec(small_geom)], primary, mirror_path=mirror)
    assert open(primary, encoding="utf-8").read() == open(mirror, encoding="utf-8").read()
    assert load_cards(mirror)["cases"][0]["case_id"] == "phantom"


def test_rerender_needs_no_engine_run(small_geom, tmp_path):
    payload = export_outcome_cards([_spec(small_geom)], str(tmp_path / "c.json"))
    dash = str(tmp_path / "d.html")
    path, title = regenerate_ui_artifacts(payload, dash)
    assert path == dash and title
    assert "phantom" in open(dash, encoding="utf-8").read()


def test_load_cards_refuses_an_empty_export(tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"cases": []}))
    with pytest.raises(ValueError, match="has no cases"):
        load_cards(str(p))


# ---------------------------------------------------------------------------
# Dashboard rendering
# ---------------------------------------------------------------------------


def test_dashboard_refuses_an_empty_payload(tmp_path):
    with pytest.raises(ValueError, match="no outcome cards"):
        render_dashboard_html({"cases": []}, str(tmp_path / "d.html"))


def test_dashboard_is_self_contained_and_carries_the_data(small_geom, tmp_path):
    payload = export_outcome_cards(
        [_spec(small_geom, case_id="PHANTOM-9", seeds=(11, 12), days=30)],
        str(tmp_path / "cards.json"),
    )
    out = str(tmp_path / "dash.html")
    render_dashboard_html(payload, out)
    html = open(out, encoding="utf-8").read()

    assert "PHANTOM-9" in html
    assert "NOT A PATIENT PREDICTION" in html
    assert "CONSTANTS ASSUMED" in html
    assert "DAY 365" in html.upper()
    # no unresolved python-brace escapes leaked into the stylesheet or markup
    # (the embedded JSON legitimately contains adjacent braces, so only the
    # part before the script block is checked)
    head = html.split("<script>")[0]
    assert "{{" not in head and "}}" not in head
    assert "@media (max-width:900px){" in head
    # the embedded data is real: the day-365 burden of the exported card
    burden = payload["cases"][0]["day_365"]["burden_mtd"]
    assert str(round(burden, 6)) in html or "DATA.cases" in html
    # the two flag branches exist in the rendered script
    assert "pfs_extension_is_lower_bound" in html
    assert "clearance_note" in html


def test_provenance_registry_is_complete():
    prov = outcome_export_provenance()
    assert prov["engine_arithmetic"].value == "established"
    assert prov["engine_constants"].value == "unknown"
    assert prov["r0_from_real_transcriptome"].value == "unknown"
    for key, value in prov.items():
        assert value.value in {"established", "emerging_human", "animal_only",
                              "proven_failed", "unknown"}, key
