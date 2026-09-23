"""Tests for the MGMT beta -> resistance bridge and its engine measurement.

Three classes of assertion, kept separate:

* **The mapping.** ``mgmt_resistance`` implements
  ``r0 = Baseline_Resistance * (1 - beta)`` exactly, refuses out-of-range
  betas rather than clamping them, and at ``beta = 0`` reproduces the
  calibrated default clones -- a re-parameterisation of the existing model,
  not a second competing one.
* **The engine measurement.** A live two-beta sweep on a small phantom must
  move MTD progression in the methylated direction, and the committed sweep
  artifact must record the full quantile measurement it claims.
* **The real panel.** The cached cBioPortal betas are fractions in [0, 1],
  and the quantile betas fed to the engine are the panel's own quantiles.
"""

from __future__ import annotations

import math
import os

import pytest

from danon.evosteer import DEFAULT_CLONES, MTDPolicy
from danon.mgmt import (
    BASELINE_RESISTANCE,
    clone_params_for_mgmt_beta,
    load_mgmt_beta_panel,
    mgmt_resistance,
    mgmt_resistance_sweep,
    render_mgmt_report,
    resistance_audit,
)

PANEL_PATH = "data/gdc_methylation/cbioportal_mgmt_probes.json"
ARTIFACT = "outputs_real/mgmt_resistance.txt"

pytestmark = pytest.mark.skipif(
    not os.path.exists(PANEL_PATH), reason="cached cBioPortal beta panel absent"
)


@pytest.fixture(scope="module")
def small_sweep():
    """A two-beta MTD-only sweep: fast, and enough to show direction."""
    return mgmt_resistance_sweep(
        (0.0, 0.8),
        beta_source="test fixture",
        geometry_size=8,
        policies=[MTDPolicy()],
        n_replicates=2,
        seed0=0,
    )


# ---------------------------------------------------------------------------
# The mapping itself
# ---------------------------------------------------------------------------


class TestMapping:
    def test_formula_is_the_requested_one(self):
        assert mgmt_resistance(0.0) == pytest.approx(BASELINE_RESISTANCE)
        assert mgmt_resistance(1.0) == pytest.approx(0.0)
        assert mgmt_resistance(0.25) == pytest.approx(BASELINE_RESISTANCE * 0.75)

    def test_monotone_decreasing_in_beta(self):
        r0s = [mgmt_resistance(b / 10) for b in range(11)]
        assert all(a > b for a, b in zip(r0s, r0s[1:]))

    @pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), "0.5", None])
    def test_invalid_beta_is_refused_not_clamped(self, bad):
        with pytest.raises(ValueError):
            mgmt_resistance(bad)  # type: ignore[arg-type]

    def test_beta_zero_reproduces_the_calibrated_default(self):
        # The bridge is a re-parameterisation: at beta = 0 it must be the
        # existing model exactly, or the sweep would compare two rival models.
        assert clone_params_for_mgmt_beta(0.0) == DEFAULT_CLONES

    def test_bridge_sets_sensitivity_scale(self):
        clones = clone_params_for_mgmt_beta(0.5)
        assert clones.drug_sensitivity_scale_R == pytest.approx(
            1.0 - BASELINE_RESISTANCE * 0.5
        )

    def test_audit_splits_reachable_from_ceiling(self):
        audit = resistance_audit()
        assert 0.0 < audit.reachable_span <= audit.total_resistance_unmethylated
        assert 0.0 < audit.ceiling_share < 1.0
        assert "kill ceiling" in audit.verdict


# ---------------------------------------------------------------------------
# The real panel
# ---------------------------------------------------------------------------


class TestRealPanel:
    def test_panel_betas_are_fractions(self):
        panel = load_mgmt_beta_panel()
        betas = list(panel.betas_of(panel.complete_patients()).values())
        assert betas, "the cached panel loaded but produced no betas"
        assert all(0.0 <= b <= 1.0 for b in betas)

    def test_quantile_betas_are_valid_engine_inputs(self):
        panel = load_mgmt_beta_panel()
        for beta in panel.quantiles((0.0, 0.25, 0.5, 0.75, 1.0)):
            r0 = mgmt_resistance(beta)
            assert math.isfinite(r0) and 0.0 <= r0 <= BASELINE_RESISTANCE


# ---------------------------------------------------------------------------
# The engine measurement
# ---------------------------------------------------------------------------


class TestSweep:
    def test_scale_r_follows_the_mapping(self, small_sweep):
        for name, arms in small_sweep.by_policy.items():
            for arm in arms:
                assert arm.scale_R == pytest.approx(1.0 - mgmt_resistance(arm.beta))
                assert name  # every arm is labelled by its policy

    def test_mtd_moves_in_the_methylated_direction(self, small_sweep):
        arms = small_sweep.by_policy["mtd"]
        by_beta = {a.beta: a for a in arms}
        low, high = by_beta[0.0], by_beta[0.8]
        # A fully methylated tumour (lower r0) must not progress earlier.
        assert high.median_pfs_days >= low.median_pfs_days

    def test_artifact_records_the_committed_measurement(self):
        assert os.path.exists(ARTIFACT), (
            "outputs_real/mgmt_resistance.txt is the sweep's measurement record; "
            "regenerate it rather than asserting an unrecorded number"
        )
        text = open(ARTIFACT).read()
        assert "r0 = Baseline_Resistance * (1 - beta)" in text
        assert "mtd" in text and "beta=" in text

    def test_report_renders_all_sections(self, small_sweep):
        text = render_mgmt_report(
            panel=load_mgmt_beta_panel(),
            sweep=small_sweep,
            audit=resistance_audit(),
        )
        assert "MGMT PROMOTER METHYLATION -> RESISTANCE TERM, MEASURED" in text
        assert "What this cannot show:" in text
