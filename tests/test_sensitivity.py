"""Tests for the parametric robustness sweep.

Pins: perturbation semantics (each target moves only what it claims to),
refusal paths, the always-defined burden criterion, the PFS non-discrimination
rule (an all-censored policy arm must never be scored as a loss or a win on
PFS), material-effect break detection, and determinism of the engine sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

from danon.evosteer import DEFAULT_CLONES, DEFAULT_DRUGS, make_gbm_phantom
from danon.sensitivity import (
    MATERIAL_EFFECT_RATIO,
    PerturbedOutcome,
    SensitivityReport,
    TARGETS,
    perturb,
    run_sensitivity,
    sensitivity_provenance,
    what_sensitivity_cannot_prove,
)


# ---------------------------------------------------------------------------
# Perturbation semantics
# ---------------------------------------------------------------------------


def test_each_target_moves_only_what_it_claims():
    base = perturb("collateral_b_kill", 1.0)
    assert base[0] == DEFAULT_CLONES and base[1] == DEFAULT_DRUGS

    c, d, r0 = perturb("tumour_growth", 0.5)
    assert c.growth_S == pytest.approx(DEFAULT_CLONES.growth_S * 0.5)
    assert c.growth_R == pytest.approx(DEFAULT_CLONES.growth_R * 0.5)
    # the resistance-cost RATIO is preserved, which is the point of scaling both
    assert c.growth_R / c.growth_S == pytest.approx(
        DEFAULT_CLONES.growth_R / DEFAULT_CLONES.growth_S
    )
    assert d == DEFAULT_DRUGS and r0 == 0.10

    c, d, _ = perturb("drug_a_kill", 2.0)
    assert c == DEFAULT_CLONES
    assert d.kill_max_S[0] == pytest.approx(DEFAULT_DRUGS.kill_max_S[0] * 2.0)
    assert d.kill_max_R[0] == pytest.approx(DEFAULT_DRUGS.kill_max_R[0] * 2.0)
    assert d.kill_max_S[1] == DEFAULT_DRUGS.kill_max_S[1]
    assert d.kill_max_R[1] == DEFAULT_DRUGS.kill_max_R[1]

    c, d, _ = perturb("collateral_b_kill", 0.25)
    assert d.kill_max_R[1] == pytest.approx(DEFAULT_DRUGS.kill_max_R[1] * 0.25)
    assert d.kill_max_R[0] == DEFAULT_DRUGS.kill_max_R[0]
    assert d.kill_max_S == DEFAULT_DRUGS.kill_max_S

    c, d, _ = perturb("drug_transport", 3.0)
    assert d.diffusion == pytest.approx(
        (DEFAULT_DRUGS.diffusion[0] * 3.0, DEFAULT_DRUGS.diffusion[1] * 3.0)
    )
    assert c == DEFAULT_CLONES


def test_resistance_cost_scaling_and_clamp():
    c, _, _ = perturb("resistance_cost", 2.0)
    base_cost = 1.0 - DEFAULT_CLONES.growth_R / DEFAULT_CLONES.growth_S
    assert c.growth_S == DEFAULT_CLONES.growth_S
    assert 1.0 - c.growth_R / c.growth_S == pytest.approx(min(0.95, base_cost * 2.0))

    # a very large scale must not produce a negative growth rate
    c, _, _ = perturb("resistance_cost", 1e6)
    assert c.growth_R > 0.0
    assert 1.0 - c.growth_R / c.growth_S == pytest.approx(0.95)


def test_intake_r0_is_bounded_and_refuses_nothing_silently():
    _, _, small = perturb("intake_r0", 0.001, r0=0.1)
    _, _, big = perturb("intake_r0", 1e6, r0=0.1)
    assert small == pytest.approx(1e-4)
    assert big == pytest.approx(0.5)
    _, _, same = perturb("intake_r0", 1.0, r0=0.1)
    assert same == pytest.approx(0.1)


def test_perturb_refuses_bad_target_and_scale():
    with pytest.raises(ValueError, match="unknown perturbation target"):
        perturb("tumour_growth_lol", 1.0)
    with pytest.raises(ValueError, match="scale must be finite and > 0"):
        perturb("tumour_growth", 0.0)
    with pytest.raises(ValueError, match="scale must be finite and > 0"):
        perturb("tumour_growth", float("nan"))


# ---------------------------------------------------------------------------
# Cell arithmetic: the always-defined criterion and the PFS rule
# ---------------------------------------------------------------------------


def _cell(**kw) -> PerturbedOutcome:
    defaults = dict(
        target="collateral_b_kill",
        scale=1.0,
        policy_pfs=(365.0, 365.0),
        mtd_pfs=(200.0, 260.0),
        policy_mean_burden=(80.0, 84.0),
        mtd_mean_burden=(800.0, 840.0),
        policy_median_pfs=365.0,
        mtd_median_pfs=230.0,
        policy_toxicity=30.0,
        mtd_toxicity=36.0,
        horizon_days=365.0,
    )
    defaults.update(kw)
    return PerturbedOutcome(**defaults)


def test_burden_criterion_is_defined_even_when_pfs_is_not(  ):
    cell = _cell()
    assert cell.burden_ratio == pytest.approx(
        float(np.median([80.0 / 800.0, 84.0 / 840.0]))
    )
    assert cell.policy_wins
    assert cell.material_win
    # policy arm is fully censored -> PFS cannot be compared, and the cell
    # must NOT report an extension
    assert cell.policy_events == 0
    assert not cell.pfs_discriminating
    assert cell.extension is None


def test_all_censored_policy_arm_is_not_a_pfs_win_or_loss():
    cell = _cell()
    assert cell.policy_median_pfs > cell.mtd_median_pfs  # numerically longer...
    assert cell.extension is None  # ...but not reported as an extension


def test_material_threshold_and_loss_paths():
    loser = _cell(
        policy_mean_burden=(790.0, 800.0), mtd_mean_burden=(800.0, 840.0)
    )
    assert loser.burden_ratio > MATERIAL_EFFECT_RATIO
    assert loser.policy_wins  # technically better
    assert not loser.material_win  # but not materially

    beaten = _cell(policy_mean_burden=(900.0, 950.0))
    assert not beaten.policy_wins
    assert not beaten.material_win


def test_zero_control_burden_is_not_an_infinite_loss():
    cell = _cell(mtd_mean_burden=(0.0, 0.0), policy_mean_burden=(0.0, 0.0))
    assert cell.paired_burden_ratios == (1.0, 1.0)
    assert not (cell.policy_wins and cell.burden_ratio < 1.0)


def test_pfs_discriminating_requires_events_in_both_arms():
    both = _cell(policy_pfs=(200.0, 220.0), policy_median_pfs=210.0)
    assert both.pfs_discriminating
    assert both.extension == pytest.approx((210.0 - 230.0) / 230.0)

    control_only = _cell(policy_pfs=(365.0, 365.0))
    assert not control_only.pfs_discriminating
    assert control_only.extension is None


# ---------------------------------------------------------------------------
# Report aggregation
# ---------------------------------------------------------------------------


def test_report_counts_break_scales_and_failures():
    cells = [
        _cell(scale=1.0),
        _cell(scale=0.25),
        # collateral fails at 0.05: still technically better, not material
        _cell(scale=0.05, policy_mean_burden=(790.0, 800.0)),
        # and genuinely loses below that
        _cell(scale=0.02, policy_mean_burden=(900.0, 910.0)),
    ]
    rep = SensitivityReport(
        case_id="synthetic",
        r0=0.1,
        days=365,
        seeds=(1, 2),
        scales=(1.0, 0.25, 0.05, 0.02),
        outcomes=cells,
    )
    assert rep.n_cells() == 4
    assert rep.n_cells_material_wins() == 2
    assert rep.n_cells_policy_wins() == 3
    assert rep.break_scale("collateral_b_kill") == pytest.approx(0.05)
    assert [o.scale for o in rep.failures()] == [pytest.approx(0.02)]
    assert rep.failure_scales("collateral_b_kill") == pytest.approx([0.02])
    assert rep.most_sensitive_target() == "collateral_b_kill"
    text = rep.summary_table()
    assert "material-effect threshold" in text
    assert "lost material control" in text


def test_report_never_claims_a_break_when_every_cell_holds():
    rep = SensitivityReport(
        case_id="synthetic",
        r0=0.1,
        days=365,
        seeds=(1,),
        scales=(1.0, 0.5),
        outcomes=[_cell(scale=1.0), _cell(scale=0.5)],
    )
    assert rep.break_scale("collateral_b_kill") is None
    assert "material control lost" not in rep.summary_table()


# ---------------------------------------------------------------------------
# The sweep itself, on a tiny phantom
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_geom():
    return make_gbm_phantom(size=10, seed=3, n_vessels=3)


def test_sweep_runs_and_is_deterministic(small_geom):
    kwargs = dict(
        geometry=small_geom,
        r0=0.08,
        case_id="phantom",
        targets=("collateral_b_kill", "intake_r0"),
        scales=(0.5, 1.0),
        days=60,
        seeds=(5,),
    )
    a = run_sensitivity(**kwargs)
    b = run_sensitivity(**kwargs)
    assert a.n_cells() == 4
    assert [o.burden_ratio for o in a.outcomes] == [
        o.burden_ratio for o in b.outcomes
    ]
    assert [o.policy_mean_burden for o in a.outcomes] == [
        o.policy_mean_burden for o in b.outcomes
    ]
    # the reference cell must be identical across targets (same world)
    ref = [o for o in a.outcomes if o.scale == 1.0]
    assert len(ref) == 2
    assert ref[0].policy_mean_burden == ref[1].policy_mean_burden
    assert ref[0].mtd_mean_burden == ref[1].mtd_mean_burden
    assert ref[0].horizon_days == 60.0


def test_sweep_refuses_bad_input(small_geom):
    with pytest.raises(ValueError, match="need at least 2 days"):
        run_sensitivity(small_geom, r0=0.1, days=1)
    with pytest.raises(ValueError, match="need at least one seed"):
        run_sensitivity(small_geom, r0=0.1, days=10, seeds=())
    with pytest.raises(ValueError, match="unknown perturbation targets"):
        run_sensitivity(small_geom, r0=0.1, days=10, seeds=(1,), targets=("nope",))


def test_records_carry_the_criteria(small_geom):
    rep = run_sensitivity(
        small_geom,
        r0=0.08,
        targets=("collateral_b_kill",),
        scales=(1.0,),
        days=40,
        seeds=(3,),
    )
    rec = rep.to_records()[0]
    for key in (
        "burden_ratio",
        "paired_burden_ratios",
        "policy_wins",
        "pfs_discriminating",
        "policy_mean_burden",
        "mtd_mean_burden",
    ):
        assert key in rec


# ---------------------------------------------------------------------------
# Honesty record
# ---------------------------------------------------------------------------


def test_provenance_and_limits_are_stated():
    prov = sensitivity_provenance()
    assert prov["assumed_constant_values"].value == "unknown"
    assert prov["sweep_replaces_bench_calibration"].value == "proven_failed"
    limits = what_sensitivity_cannot_prove()
    assert any("insensitivity, not validity" in l for l in limits)
    assert any("in vivo" in l for l in limits)
    assert all(isinstance(l, str) and l for l in limits)
    assert set(TARGETS) == {
        "tumour_growth",
        "drug_a_kill",
        "collateral_b_kill",
        "drug_transport",
        "resistance_cost",
        "intake_r0",
    }
