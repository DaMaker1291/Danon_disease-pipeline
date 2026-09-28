"""Tests for the variance budget: the algebra, the refusals, and the record.

The first class of test is the one that matters: a decomposition is only worth
trusting if it recovers a variance it was not told. So the arithmetic is checked
against a construction whose between- and within-components are known on paper
and asserted EXACTLY (to float noise), and separately against a stochastic
construction where the recovery is asserted at the estimator's own sampling
error rather than at a tolerance chosen to make it pass.

The second class pins the refusals. A decomposition that silently clips a
negative variance component, or reports a fraction when there is no spread to
attribute, would manufacture a finding out of an artifact -- which is the exact
failure this module exists to avoid.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from danon.cure import EvidenceStatus
from danon.evosteer import make_gbm_phantom
from danon.mgmt import MgmtBetaPanel
from danon.uncertainty import (
    ARM_POLICIES,
    READOUTS,
    Band,
    UncertaintyError,
    budget_from_groups,
    mgmt_input_distribution,
    percentile_band,
    propagate_mgmt_uncertainty,
    render_uncertainty_report,
    run_input_grid,
    uncertainty_provenance,
    variance_budget,
    what_this_decomposition_cannot_show,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "outputs_real" / "uncertainty_budget.json"


# ---------------------------------------------------------------------------
# The arithmetic, against a construction known on paper
# ---------------------------------------------------------------------------


def test_decomposition_is_exact_on_a_construction_known_on_paper():
    """Zero within-noise and a chosen mean spread: every term is analytic.

    With groups [m_i - w, m_i + w], each group's mean is exactly m_i and its
    squared deviations are w^2 each, so

        MSW = 2 w^2        MSB = 2 * s^2(m)        (r = 2)

    and the variance components are therefore MSW and s^2(m) - w^2 with no
    estimation error at all. A decomposition that cannot reproduce these is
    wrong, not noisy.
    """
    k, w = 40, 0.7
    means = np.linspace(0.0, 5.0, k)
    groups = [[m - w, m + w] for m in means]

    budget = variance_budget(groups)
    sample_var = float(np.var(means, ddof=1))
    assert budget.df_between == k - 1
    assert budget.df_within == k * 1
    assert budget.ms_within == pytest.approx(2 * w * w, rel=1e-12)
    assert budget.ms_between == pytest.approx(2 * sample_var, rel=1e-12)
    assert budget.component_within == pytest.approx(2 * w * w, rel=1e-12)
    assert budget.component_between == pytest.approx(sample_var - w * w, rel=1e-9)
    assert budget.input_fraction == pytest.approx(
        (sample_var - w * w) / (sample_var - w * w + 2 * w * w), rel=1e-9
    )


def test_zero_within_noise_leaves_the_whole_variance_to_the_input():
    """With no within-noise the input owns all of it, to the last digit.

    MSB = r * s^2(means) and MSW = 0, so (MSB - MSW)/r = s^2(means) exactly --
    the sample variance of the inputs, unattenuated.
    """
    values = (1.0, 2.0, 4.0, 8.0)
    budget = variance_budget([[m, m, m] for m in values])
    assert budget.ms_within == 0.0
    assert budget.component_within == 0.0
    assert budget.component_between == pytest.approx(
        float(np.var(values, ddof=1)), rel=1e-12
    )
    assert budget.input_fraction == 1.0
    # MSW is zero, so an F ratio would be infinite; that is not a number to report.
    assert budget.f_ratio is None


def test_recovery_of_planted_variances_is_within_the_estimators_own_error():
    """Planted between-variance 4 and within-variance 1, recovered approximately.

    The tolerance is the standard error the estimator actually has at this
    sample size, not a number tuned until the test passed: SE of MSW is
    sigma^2 * sqrt(2/df).
    """
    rng = np.random.default_rng(0)
    k, r = 60, 3
    between = rng.normal(0.0, 2.0, size=k)
    groups = [[b + rng.normal(0.0, 1.0) for _ in range(r)] for b in between]
    budget = variance_budget(groups)

    se_within = 1.0 * math.sqrt(2.0 / budget.df_within)
    assert abs(budget.component_within - 1.0) < 4 * se_within
    assert budget.component_between is not None
    assert abs(budget.component_between - 4.0) < 3.0
    assert budget.input_fraction is not None
    assert abs(budget.input_fraction - 0.8) < 0.12
    assert budget.f_ratio is not None and budget.f_ratio > 4.0


def test_a_negative_component_is_reported_not_clipped():
    """No input effect: the estimator's sign is the verdict, and it must survive.

    Constructed so MSB is exactly zero rather than merely small: every group is
    symmetric about the same point, so every group mean is identical whatever
    the within-scatter is. Then (MSB - MSW)/r < 0 by construction, and a
    decomposition that returned 0.0 here would be inventing a floor.
    """
    rng = np.random.default_rng(1)
    groups = [[-w, w] for w in rng.uniform(0.5, 2.0, size=40)]
    budget = variance_budget(groups)
    assert budget.ms_between == 0.0
    assert budget.component_between is None, "a negative component was clipped to a number"
    assert budget.input_fraction is None
    assert budget.f_ratio == 0.0
    assert (budget.ms_between - budget.ms_within) / budget.n_replicates < 0.0


def test_a_readout_identical_everywhere_is_degenerate_not_a_finding():
    budget = variance_budget([[365.0, 365.0] for _ in range(12)])
    assert budget.is_degenerate
    assert budget.input_fraction is None
    row = budget_from_groups("mtd", {"pfs_days": [[365.0, 365.0]] * 12, "final_burden": [[1.0, 2.0]] * 12})[0]
    assert "every donor and seed returned the same value" in row.verdict()


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_partial_or_absent_designs_are_refused_by_name():
    with pytest.raises(UncertaintyError, match="nothing to decompose"):
        variance_budget([])
    with pytest.raises(UncertaintyError, match="at least 2 inputs"):
        variance_budget([[1.0, 2.0]])
    with pytest.raises(UncertaintyError, match="at least 2 replicates"):
        variance_budget([[1.0], [2.0]])
    with pytest.raises(UncertaintyError, match="same number of replicates"):
        variance_budget([[1.0], [2.0, 3.0]])


def test_a_band_refuses_samples_that_would_read_as_certainty():
    with pytest.raises(UncertaintyError, match="empty sample"):
        percentile_band([])
    with pytest.raises(UncertaintyError, match="at least 2 outcomes"):
        percentile_band([7.0])


def test_band_is_the_2_5_and_97_5_percentiles():
    band = percentile_band(list(range(100)))
    assert band.lo == pytest.approx(2.475)
    assert band.hi == pytest.approx(96.525)
    assert band.median == 49.5
    assert band.n == 100
    assert band.width == pytest.approx(94.05)


def test_run_grid_refuses_an_unknown_arm_or_a_single_replicate():
    geom = make_gbm_phantom(size=10)
    with pytest.raises(UncertaintyError, match="arm must be one of"):
        run_input_grid(geom, [0.1, 0.6], arm="magic", days=2, replicates=2, initial_resistant_fraction=0.01)
    with pytest.raises(UncertaintyError, match="at least 2 to measure"):
        run_input_grid(geom, [0.1, 0.6], arm="mtd", days=2, replicates=1, initial_resistant_fraction=0.01)
    with pytest.raises(UncertaintyError, match="at least 2 inputs"):
        run_input_grid(geom, [0.1], arm="mtd", days=2, replicates=2, initial_resistant_fraction=0.01)


def test_the_arm_names_are_the_engines_own_classes():
    from danon.evosteer import CollateralSwitchPolicy, MTDPolicy

    assert ARM_POLICIES == {"mtd": MTDPolicy, "steering": CollateralSwitchPolicy}


# ---------------------------------------------------------------------------
# The real path: a short run through the engine
# ---------------------------------------------------------------------------


def _panel(betas: dict[str, float]) -> MgmtBetaPanel:
    return MgmtBetaPanel(
        profile="test",
        probes={"p1": "probe one"},
        betas={pid: {"p1": b} for pid, b in betas.items()},
        source="test fixture (not the real panel)",
        n_records=len(betas),
        notes=[],
    )


def test_input_distribution_uses_the_panels_own_accessor_and_refuses_a_degenerate_one():
    panel = _panel({"a": 0.1, "b": 0.4, "c": 0.9})
    values, source = mgmt_input_distribution(panel)
    assert list(values) == [0.1, 0.4, 0.9]
    assert source == str(panel)
    with pytest.raises(UncertaintyError, match="at least 2"):
        mgmt_input_distribution(_panel({"a": 0.5}))


def test_a_short_propagation_runs_both_arms_and_labels_itself():
    panel = _panel({"a": 0.02, "b": 0.75})
    report = propagate_mgmt_uncertainty(
        make_gbm_phantom(size=10),
        panel,
        geometry_label="phantom 10 (test)",
        arms=("mtd", "steering"),
        days=4,
        replicates=2,
        initial_resistant_fraction=0.01,
    )
    assert report.is_credible_interval is False
    assert set(ARM_POLICIES) == {row.arm for row in report.arms}
    assert {row.readout for row in report.arms} == set(READOUTS)
    assert len(report.arms) == len(ARM_POLICIES) * len(READOUTS)

    for row in report.arms:
        assert row.propagated_band.n == 4  # 2 donors x 2 seeds
        assert row.input_span.n == 2
        assert row.propagated_band.lo <= row.propagated_band.median <= row.propagated_band.hi
        assert row.budget.n_inputs == 2 and row.budget.n_replicates == 2
        # A fraction may only exist when the component does.
        if row.budget.input_fraction is not None:
            between = row.budget.component_between
            assert between is not None and between >= 0.0
            assert 0.0 <= row.budget.input_fraction <= 1.0

    text = render_uncertainty_report(report)
    assert "credible interval: False" in text
    assert "VARIANCE BUDGET" in text
    for item in what_this_decomposition_cannot_show():
        assert item[:40] in text


def test_beta_reaches_the_engine_so_the_inputs_are_not_all_the_same():
    """Guard against a wiring bug that maps every donor to the same clone params.

    Read on the burden, not on PFS: over a 60-day horizon on a small phantom
    NEITHER donor progresses, so PFS is the horizon for both and would be
    identical even with the wiring correct. The burden separates them.
    """
    from danon.mgmt import clone_params_for_mgmt_beta

    geom = make_gbm_phantom(size=8)
    assert (
        clone_params_for_mgmt_beta(0.0).drug_sensitivity_scale_R
        != clone_params_for_mgmt_beta(0.9).drug_sensitivity_scale_R
    )
    groups = run_input_grid(
        geom,
        [0.0, 0.9],
        arm="mtd",
        days=60,
        replicates=2,
        initial_resistant_fraction=0.01,
    )
    assert groups["final_burden"][0] != groups["final_burden"][1], (
        "a maximally methylated and a fully unmethylated donor produced identical "
        "burdens: the beta is not reaching the clone parameters"
    )


def test_the_provenance_marks_the_credible_interval_as_refused():
    prov = uncertainty_provenance()
    assert prov["credible_interval"] is EvidenceStatus.PROVEN_FAILED
    assert prov["r0_mapping"] is EvidenceStatus.UNKNOWN
    assert prov["variance_decomposition"] is EvidenceStatus.ESTABLISHED


# ---------------------------------------------------------------------------
# The committed record
# ---------------------------------------------------------------------------


def test_committed_record_recomputes_its_own_verdicts():
    """The record's sentence and its mean squares must agree, or it is stale."""
    if not RECORD.exists():
        pytest.skip("uncertainty record not built (run _uncertainty_cohort.py)")
    record = json.loads(RECORD.read_text(encoding="utf-8"))

    assert record["is_credible_interval"] is False
    assert record["n_donors"] == 140, "the panel the record claims is not the panel"
    assert record["input_source"].startswith("MGMT beta panel")
    assert [(r["arm"], r["readout"]) for r in record["arms"]] == [
        (arm, readout) for arm in ("mtd", "steering") for readout in READOUTS
    ]

    for row in record["arms"]:
        b = row["budget"]
        between = (b["ms_between"] - b["ms_within"]) / record["replicates"]
        if b["is_degenerate"]:
            # Degeneracy is tested FIRST: with both mean squares zero the
            # arithmetic component is exactly 0.0, and reading that as a
            # resolved input would be the clipping this module refuses.
            assert b["ms_between"] == 0.0 and b["ms_within"] == 0.0
            assert b["input_fraction"] is None
            assert "no spread to attribute" in row["verdict"]
        elif between <= 0.0:
            assert b["component_between"] is None, f"{row['arm']}/{row['readout']} clipped"
            assert b["input_fraction"] is None
            assert "NOT resolvable" in row["verdict"]
        else:
            assert b["component_between"] == pytest.approx(between, rel=1e-9)
            assert b["input_fraction"] == pytest.approx(
                between / (between + b["ms_within"]), rel=1e-9
            )
            assert "accounts for" in row["verdict"]
        # The propagated band must contain the input span's median: the
        # replicates can only widen what the inputs alone produce.
        assert row["propagated_band"]["n"] == record["n_donors"] * record["replicates"]
        assert row["input_span"]["n"] == record["n_donors"]

    degenerate = [
        r for r in record["arms"] if r["arm"] == "steering" and r["readout"] == "pfs_days"
    ][0]
    assert degenerate["budget"]["is_degenerate"], (
        "the switching arm is expected to saturate at the horizon in this design; if "
        "it no longer does, the design changed and the record must be rebuilt"
    )
