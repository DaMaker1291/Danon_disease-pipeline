"""The stress-test matrix's own tests: the validator is validated.

The release gate's credibility rests on the matrix itself being sound, so
these tests pin four things: every shipped probe passes on the honest
layers, the record is complete machine-readable output, the gate BLOCKS when
a probe is made to fail (proven by a fabricated failing probe, not by
breaking the detector), and the published state band matches the weighted
least-squares formula it claims to implement -- the check that would have
caught the sqrt(W)-too-narrow band before it was measured.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from danon.robustness import (
    CALIBRATION_FLOOR,
    ProbeResult,
    ProbeStatus,
    PROBES,
    RECORD_VERSION,
    release_gate,
    robustness_record,
    run_robustness_matrix,
    write_robustness_record,
)


@pytest.fixture(scope="module")
def matrix():
    """One matrix run for the whole module: the probes are the expensive part."""
    return run_robustness_matrix()


def test_every_shipped_probe_passes_on_the_honest_layers(matrix):
    failing = [(r.probe_id, r.detail, r.error) for r in matrix if r.status is not ProbeStatus.PASS]
    assert failing == [], failing


def test_matrix_covers_every_planned_challenge_class(matrix):
    """The probe IDs must span the planner's classes: data quality (D),
    measurement variation (M), biological variation (B), failure/safety (F),
    band calibration (U) and the observer layer (O)."""
    ids = [r.probe_id for r in matrix]
    assert any(i.startswith("D") for i in ids), "missing data-quality probes"
    assert any(i.startswith("M") for i in ids), "missing measurement-variation probes"
    assert any(i.startswith("B") for i in ids), "missing biological-variation probes"
    assert any(i.startswith("F") for i in ids), "missing failure/safety probes"
    assert any(i.startswith("U") for i in ids), "missing band-calibration probes"
    assert any(i.startswith("O") for i in ids), "missing observer probes"
    assert any(i.startswith("V") for i in ids), "missing variant-channel probes"


def test_record_is_machine_readable_and_versioned(matrix, tmp_path):
    path = tmp_path / "record.json"
    record = write_robustness_record(matrix, str(path))
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == record
    assert record["record_version"] == RECORD_VERSION == "robustness_record_v3"
    assert record["aggregate"]["n_probes"] == len(PROBES)
    assert record["aggregate"]["gate"] in ("PASS", "BLOCKED")
    assert record["scope"].startswith("synthetic-only")
    assert "observer" in record["scope"]


def test_the_calibration_floor_is_the_pre_declared_one(matrix):
    """The floor is part of the declared protocol, and the measured coverage
    is recorded beside it -- a number, not an assertion."""
    assert CALIBRATION_FLOOR == 0.80
    u1 = next(r for r in matrix if r.probe_id == "U1_band_calibration")
    for label, run in u1.metrics.items():
        assert run["coverage"] >= CALIBRATION_FLOOR, (label, run)
        assert run["n_cases"] == 40
        assert run["mean_band_width"] is not None and run["mean_band_width"] > 0


def test_published_band_sigma_matches_the_weighted_least_squares_formula():
    """The band is a claim about a formula, and the formula is checkable.

    Var(slope) = s2 / sum_i w_i (x_i - xbar)^2, with s2 = sum_i w_i resid_i^2 /
    dof estimated from the residuals. An earlier form divided s2 by W a second
    time, shrinking the published sigma by sqrt(W) (~40x at this window) and
    turning the published "95% band" into one that contained the true rate in
    1 of 40 synthetic cases. This test pins the formula, not the incident.
    """
    from danon.detect.state import estimate_state

    rate, sigma, n, window = 0.003, 2.0, 60, 10
    rng = np.random.default_rng(0)
    days = np.arange(n, dtype=float) * 7.0
    values = 10.0 * np.exp(rate * days) + rng.normal(0.0, sigma, n)
    sigmas = np.full(n, sigma)
    est = estimate_state(values, days, sigmas)
    assert not est.is_insufficient

    last = slice(n - window, n)
    x = days[last]
    y = np.log(values[last])
    w = 1.0 / (sigmas[last] / values[last]) ** 2
    W = float(w.sum())
    xbar = float((w * x).sum() / W)
    ybar = float((w * y).sum() / W)
    sum_wxx = float((w * (x - xbar) ** 2).sum())
    slope = float((w * (x - xbar) * (y - ybar)).sum() / sum_wxx)
    resid = y - (slope * x + (ybar - slope * xbar))
    s2 = float((w * resid**2).sum() / (window - 2))
    expected_sigma = float(np.sqrt(s2 / sum_wxx))

    assert est.rate_per_day == pytest.approx(slope, rel=1e-6, abs=0.0)
    assert est.rate_sigma == pytest.approx(expected_sigma, rel=1e-6, abs=0.0)
    # The shrunk form would report expected_sigma / sqrt(W), ~40x smaller here.
    assert est.rate_sigma * np.sqrt(W) > 10 * expected_sigma
    assert est.band_low <= rate <= est.band_high


def test_gate_blocks_on_a_fabricated_failure():
    good = ProbeResult("honest_probe", "passes", ProbeStatus.PASS, "ok")
    bad = ProbeResult("fabricated_failure", "fails", ProbeStatus.FAIL, "broke on purpose")
    gate = release_gate([good, bad])
    assert gate["gate"] == "BLOCKED"
    assert "fabricated_failure" in gate["failing_probes"]


def test_gate_blocks_on_a_probe_error():
    good = ProbeResult("honest_probe", "passes", ProbeStatus.PASS, "ok")
    err = ProbeResult("crashed_probe", "crashed", ProbeStatus.ERROR, "", error="boom")
    gate = release_gate([good, err])
    assert gate["gate"] == "BLOCKED"


def test_gate_passes_when_every_probe_passes(matrix):
    assert release_gate(matrix)["gate"] == "PASS"


def test_probe_crash_is_recorded_as_error_not_pass():
    """A probe that raises is itself a failure: the runner wraps it."""

    def broken():
        raise RuntimeError("deliberate")

    saved = list(PROBES)
    try:
        import danon.robustness as rb

        rb.PROBES = PROBES.__class__([broken])
        entry = rb.run_robustness_matrix()[0]
        assert entry.status is ProbeStatus.ERROR
        assert "deliberate" in entry.error
    finally:
        import danon.robustness as rb

        rb.PROBES = saved


def test_uncertainty_probe_uses_the_state_estimate_refusal(matrix):
    """B3 must exercise the honest insufficient-evidence path, not a crash."""
    b3 = next(r for r in matrix if r.probe_id == "B3_out_of_distribution")
    assert b3.status is ProbeStatus.PASS
    assert "no positive value" in b3.detail


def test_observer_probes_record_a_recovery_and_a_refusal(matrix):
    """The observer layer is probed in both directions: a stream it can
    explain is estimated, and one it cannot is refused -- recorded as
    verdicts, not as prose."""
    o1 = next(r for r in matrix if r.probe_id == "O1_observer_recovery")
    o2 = next(r for r in matrix if r.probe_id == "O2_observer_unidentified_refusal")
    assert o1.status is ProbeStatus.PASS and o1.metrics["n_accepted"] > 0
    assert o2.status is ProbeStatus.PASS
    assert o2.metrics["verdict"] == "refused_unidentified"


def test_observer_units_error_is_refused_rather_than_estimated(matrix):
    """A percent-scale VAF must not be read as a fraction."""
    o3 = next(r for r in matrix if r.probe_id == "O3_observer_units_error")
    assert o3.status is ProbeStatus.PASS
    assert o3.metrics["scale_factor"] == 100.0
    assert o3.metrics["verdict"] != "estimated"


def test_observer_degeneracy_invariant_is_recorded(matrix):
    """No ESTIMATED verdict may violate the observer's own criterion -- and
    how close the widest probe stream came is recorded either way."""
    o5 = next(r for r in matrix if r.probe_id == "O5_observer_degeneracy_invariant")
    assert o5.status is ProbeStatus.PASS
    assert o5.metrics["worst_width_fraction"] < 0.9
    assert o5.metrics["n_estimated"] > 0


@pytest.mark.parametrize("probe_id", [
    "D1_missing_observations",
    "D5_irregular_timestamps",
    "B3_out_of_distribution",
    "F3_failed_solver",
    "U1_band_calibration",
    "U4_band_monotone_in_noise",
    "O1_observer_recovery",
    "O5_observer_degeneracy_invariant",
    "V1_variant_af_from_counts",
    "V4_variant_pairing_contracts",
])
def test_key_probes_report_a_detail_string(matrix, probe_id):
    r = next(x for x in matrix if x.probe_id == probe_id)
    assert isinstance(r.detail, str) and r.detail, probe_id


def test_variant_probes_record_the_refusals_they_enforce(matrix):
    """The channel's refusals are probed as outcomes: only a lower bound is
    published, and the engine's own scorer refuses a k-bin support beside it."""
    v2 = next(r for r in matrix if r.probe_id == "V2_variant_bounds_are_bounded")
    assert v2.status is ProbeStatus.PASS
    assert v2.metrics["summarisation_refused"] is True
    assert v2.metrics["engine_support_refused"] is True
    assert v2.metrics["lower_bound_purity_0.8"] > v2.metrics["lower_bound_no_purity"]
