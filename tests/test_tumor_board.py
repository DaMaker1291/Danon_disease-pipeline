"""The dashboard must not be able to show a number that nothing produced.

A clinical dashboard is the easiest place in a software project to publish an
invention, because a badge LOOKS like data whatever wrote it. The mock-up this
module answers proposed an MGMT beta, a Ki-67 index, a 5-ALA residual, a
recurrence date per arm, "972 tests passing", "28/28 claims", "FDA-PCCP bounds:
NOMINAL", a 45 s -> 12 ms GPU step, an FNO surrogate and a 60 FPS WebGPU stream.
None of those exist in this repository.

So the tests are written as refusals first:

* every field the mock-up proposed is either absent or present ONLY inside a
  refusal that names it, carries the reason, and says what is recorded instead;
* no recorded (non-refused) field may contain any of the mock-up's values --
  that is the mutation this file exists to catch, and it is checked over the
  whole artifact, not over the fields a human remembered to look at;
* the counts and the hash are re-derived, so the artifact cannot carry a
  remembered number: the test-function count must equal the registry's own
  counter, and the engine hash must match a fresh hashing of the four files it
  names, verified file by file with the recorded byte lengths;
* the acceleration block must equal the perf record it cites, entry for entry,
  so the panel cannot drift from `outputs_real/perf.json`.

The last group is the honesty envelope: provenance must call every GPU,
surrogate, frame-rate and regulatory field PROVEN_FAILED, and the module's own
"what this cannot show" list must say so in words.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from danon.cure import EvidenceStatus
from danon.tumor_board import (
    ENGINE_SOURCE_FILES,
    ROOT,
    build_tumor_board,
    engine_source_hash,
    tumor_board_provenance,
    what_a_tumor_board_cannot_show,
)

#: The values the mock-up displayed, each of which must be refused. Kept as the
#: literal strings a reader would see on that screen, because the test's job is
#: to catch one of THEM reappearing -- not a paraphrase of one.
MOCKUP_VALUES = (
    "0.82",
    "28%",
    "0.42",
    "58",
    "KPS 90",
    "Day 144",
    "365+",
    "972",
    "28/28",
    "NOMINAL",
    "12 ms",
    "45 s",
    "R^2",
    "0.95",
    "60 FPS",
    "10,000",
    "Ray",
    "CUDA",
    "e7f3a9",
)


@pytest.fixture(scope="module")
def board():
    """The live board, built from the recorded artifacts in the tree."""
    return build_tumor_board()


def _walk(node, path=""):
    """Every (path, string) leaf, for whole-artifact scanning."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{path}/{key}")
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            yield from _walk(value, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def _without_refusals(board):
    return {k: v for k, v in board.items() if k != "refusals"}


# ---------------------------------------------------------------------------
# Every mock-up value is refused, and only there
# ---------------------------------------------------------------------------


def test_every_mockup_field_is_refused(board):
    shown = " | ".join(r["shown_in_the_mockup"] for r in board["refusals"])
    for needle in (
        "Day 144",
        "972 tests",
        "28/28 claims",
        "RTX 4090",
        "neural operator",
        "10,000-trajectory",
        "WebGPU",
        "FDA-PCCP",
    ):
        assert needle in shown, f"{needle!r} is not refused anywhere: {shown}"


def test_each_refusal_carries_a_reason_and_substitute(board):
    for refusal in board["refusals"]:
        assert refusal["shown_in_the_mockup"].strip()
        assert len(refusal["why_refused"]) > 40, refusal
        assert "what_is_recorded_instead" in refusal


def test_no_recorded_field_repeats_a_mockup_value(board):
    """The mutation this file exists for: a mock-up number pasted into a panel.

    Three parts of the artifact are refusals in their own right rather than
    recorded values -- the not-recorded panel, the acceleration block's own
    "what is not measured" list, and the hash (whose hex digits collide with
    short numbers like 58) -- so they are excluded here and tested where they
    belong. Everything else must be free of every mock-up value.
    """
    exempt = (
        "/patient_panel/not_recorded",
        "/acceleration/not_measured",
        "/engine_source_hash/",
    )
    offenders = []
    for path, text in _walk(_without_refusals(board)):
        if any(fragment in path for fragment in exempt):
            continue
        for needle in MOCKUP_VALUES:
            if needle in text:
                # `0.82` and friends legitimately appear inside a refusal text
                # elsewhere, but these are the NOT-refused panels.
                offenders.append((path, needle, text[:90]))
    assert not offenders, f"unrefused panels carry mock-up values: {offenders}"


def test_the_refused_fields_name_the_module_that_would_produce_them(board):
    fields = {f["field"]: f for f in board["patient_panel"]["not_recorded"]}
    for name, module in (
        ("MGMT promoter methylation beta", "danon/mgmt.py"),
        ("Ki-67 index", "danon.histology.ki67_fraction"),
        ("5-ALA fluorescent residual volume", "danon/clinical_twin.py"),
    ):
        assert name in fields, fields.keys()
        row = fields[name]
        assert module in f"{row['status']} {row['detail']}", (name, row)
        assert fields[name]["quoted_in_the_dashboard_mockup"], name
        assert fields[name]["would_come_from"], name


def test_no_recurrence_date_is_published_per_arm(board):
    panel = board["dosing_panel"]
    assert panel["no_recurrence_date"]
    arms = panel["arms"]
    assert set(arms) == {"mtd", "policy"}
    for arm in arms.values():
        assert "median_pfs_days" in arm
        assert "recurrence_day" not in arm
        assert "recurrence" not in arm
    # A median that is a censored horizon must be labelled as a lower bound: if
    # no replicate progressed, the number is the horizon, not an event time.
    if arms["policy"]["n_progressed"] == 0:
        assert panel["pfs_extension_is_lower_bound"] is True


# ---------------------------------------------------------------------------
# The numbers that ARE shown are re-derived, not remembered
# ---------------------------------------------------------------------------


def test_test_count_equals_the_registrys_own_counter(board):
    from danon.claims import _suite_function_count

    total, classes = _suite_function_count()
    assert board["audit_panel"]["test_functions"] == total
    assert board["audit_panel"]["in_classes"] == classes
    assert total > 0


def test_claims_panel_matches_the_recorded_audit(board):
    with open(Path(ROOT) / "outputs_real" / "app_artifacts" / "claims.json", encoding="utf-8") as fh:
        recorded = json.load(fh)["data"]
    panel = board["audit_panel"]["claims"]
    assert panel["passed"] == recorded["n_passed"]
    assert panel["total"] == recorded["n_claims"]
    assert panel["failed"] == recorded["n_failed"]
    assert panel["failed_ids"] == [
        row["id"] for row in recorded["claims"] if not row["supported"]
    ]
    assert panel["total"] != 28, "the mock-up's 28 is not this registry's size"


def test_engine_hash_matches_a_fresh_hash_of_the_named_files(board):
    recorded = board["audit_panel"]["engine_source_hash"]
    assert recorded["algorithm"] == "sha256"
    digest = hashlib.sha256()
    for row in recorded["files"]:
        blob = (Path(ROOT) / row["path"]).read_bytes()
        assert len(blob) == row["bytes"], row["path"]
        assert hashlib.sha256(blob).hexdigest() == row["sha256"], row["path"]
        digest.update(row["path"].encode("utf-8"))
        digest.update(blob)
    assert digest.hexdigest() == recorded["combined_sha256"]
    assert [r["path"] for r in recorded["files"]] == list(ENGINE_SOURCE_FILES)
    # It is not a hash of the whole tree: an unrelated edit must not move it.
    assert "glob" not in json.dumps(recorded).lower()
    assert engine_source_hash()["combined_sha256"] == recorded["combined_sha256"]


def test_acceleration_block_equals_the_perf_record_it_cites(board):
    with open(Path(ROOT) / "outputs_real" / "perf.json", encoding="utf-8") as fh:
        perf = json.load(fh)
    measured = board["audit_panel"]["acceleration"]
    for op in perf["operators"]:
        key = " x ".join(str(n) for n in op["shape"])
        assert measured["laplacian_ratio_by_grid"][key] == op["laplacian_ratio"]
        assert measured["divergence_ratio_by_grid"][key] == op["divergence_ratio"]
    assert measured["day_ratio_scalar"] == perf["day"]["scalar"]["ratio"]
    assert measured["day_ratio_heterogeneous"] == perf["day"]["heterogeneous"]["ratio"]
    assert measured["batch_per_replicate_ratio"] == perf["throughput"][
        "batch_per_replicate_streams"
    ]["ratio"]
    for workers, row in perf["throughput"]["processes"]["by_workers"].items():
        assert measured["process_ratio_by_workers"][workers] == row["ratio"]
    assert measured["measured_at"] == perf["measured_at"]
    assert measured["statistic"] == perf["method"]["statistic"]
    assert list(measured["not_measured"]) == list(perf["what_is_not_measured"])
    assert measured["bit_identity_holds"]["fields"] is True
    assert measured["bit_identity_holds"]["trajectory"] is True


def test_kinetics_tipping_day_agrees_with_the_recorded_curves(board):
    kinetics = board["kinetics_panel"]
    if not kinetics["resistant_fraction"]:
        pytest.skip("no recorded evolution artifact to re-derive from")
    for arm, curve in kinetics["resistant_fraction"].items():
        expected = next((i for i, v in enumerate(curve) if v > 0.5), None)
        assert kinetics["tipping_day"][arm] == expected, arm
        if expected is not None:
            assert curve[expected] > 0.5
            assert all(v <= 0.5 for v in curve[:expected])


def test_volume_panel_counts_the_frames_it_recorded(board):
    volume = board["volume_panel"]
    if not volume["arm_frames"]:
        pytest.skip("no recorded evolution artifact to re-derive from")
    with open(
        Path(ROOT) / "outputs_real" / "app_artifacts" / "evolution.json", encoding="utf-8"
    ) as fh:
        evolution = json.load(fh)["data"]
    assert volume["frame_days"] == evolution["frame_days"]
    assert volume["grid"] == evolution["grid"]
    for arm in ("policy", "mtd"):
        assert volume["arm_frames"][arm]["n_frames"] == len(
            evolution["arms"][arm]["frames_vti_b64"]
        )
        assert volume["arm_frames"][arm]["pfs_day"] == evolution["arms"][arm]["pfs_day"]


def test_a_supplied_claims_payload_is_what_the_bar_reports():
    """A full build must print THIS run's audit, not the file it is about to write."""
    payload = {
        "data": {
            "n_passed": 11,
            "n_claims": 12,
            "n_failed": 1,
            "claims": [{"id": "a_claim", "supported": False}],
        }
    }
    board = build_tumor_board(claims_payload=payload)
    panel = board["audit_panel"]["claims"]
    assert panel["passed"] == 11
    assert panel["total"] == 12
    assert panel["failed"] == 1
    assert panel["failed_ids"] == ["a_claim"]


def test_a_dependent_artifact_is_built_after_the_one_it_reads(monkeypatch):
    """The order is a dependency, and the builder passes its payload through."""
    import danon.app_artifacts as aa

    calls = []
    seen = {}

    def fake_claims():
        calls.append("claims")
        return {"data": {"n_passed": 7, "n_claims": 7, "n_failed": 0, "claims": []}}

    def fake_board():
        calls.append("tumorboard")
        seen["payload"] = aa._BUILT.get("claims")
        return {"data": {}}

    monkeypatch.setitem(aa.BUILDERS, "claims", fake_claims)
    monkeypatch.setitem(aa.BUILDERS, "tumorboard", fake_board)
    aa.build_app_artifacts(keys=["tumorboard", "claims"])
    assert calls == ["claims", "tumorboard"]
    assert seen["payload"] == {
        "data": {"n_passed": 7, "n_claims": 7, "n_failed": 0, "claims": []}
    }


def test_a_missing_record_raises_rather_than_filling_in(monkeypatch, tmp_path):
    import danon.tumor_board as tb

    monkeypatch.setattr(tb, "OUTCOME_CARDS", "outputs_real/definitely_not_here.json")
    with pytest.raises(FileNotFoundError) as excinfo:
        tb.build_tumor_board()
    assert "build it first" in str(excinfo.value)


# ---------------------------------------------------------------------------
# The honesty envelope
# ---------------------------------------------------------------------------


def test_provenance_refuses_every_thing_the_mockup_showed():
    prov = tumor_board_provenance()
    for key in (
        "any_gpu_figure",
        "any_fno_surrogate_figure",
        "any_webgpu_frame_rate",
        "any_fda_status",
    ):
        assert prov[key] is EvidenceStatus.PROVEN_FAILED, key
    assert prov["patient_panel.not_recorded"] is EvidenceStatus.UNKNOWN
    assert prov["engine_parameters"] is EvidenceStatus.UNKNOWN
    assert prov["audit_panel.acceleration"] is EvidenceStatus.ESTABLISHED


def test_the_audit_panel_reports_no_regulatory_status(board):
    assert board["audit_panel"]["no_regulatory_status"]
    recorded = json.dumps(_without_refusals(board)).lower()
    for banned in ("fda_cleared", "clearance_status", "is_approved", "nominal", "approved"):
        assert banned not in recorded, banned
    # The words appear only where they are being REFUSED: the audit panel's own
    # sentence names them to say there is nothing behind them here.
    for word in ("pccp", "fda"):
        if word in recorded:
            assert word in board["audit_panel"]["no_regulatory_status"].lower(), word


def test_the_module_states_what_it_cannot_show():
    lines = what_a_tumor_board_cannot_show()
    assert len(lines) >= 3
    joined = " ".join(lines).lower()
    for topic in ("gpu", "patient", "regulatory"):
        assert topic in joined, topic
