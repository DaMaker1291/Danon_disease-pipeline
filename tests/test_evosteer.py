"""Tests for the EvoSteer spatial evolutionary-scheduling module.

Dynamic tests run small phantoms (16^3, <=300 simulated days) and pin the
qualitative invariants measured during calibration: MTD selects for
resistance, adaptive dosing extends PFS and cuts toxicity, the collateral
switch's advantage is contingent on the drug-pair network, and the
documented low-resistant-fraction failure regime is recorded as NOT
reproduced (the provenance test pins that honesty explicitly).
"""

from __future__ import annotations

import numpy as np
import pytest

from danon.cure import EvidenceStatus
from danon.evosteer import (
    ADAPTIVE_TTP_RATIO_ANCHOR,
    MEASURED_ADAPTIVE_MTD_RATIO_BAND,
    AdaptiveDosePolicy,
    CollateralSwitchPolicy,
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    DrugParams,
    MTDPolicy,
    PolicyKind,
    PolicyState,
    SimulationTrajectory,
    TumorEvolutionSimulator,
    evosteer_provenance,
    make_gbm_phantom,
    make_policy,
    run_trial,
    sanctuary_sequestration,
    threshold_sweep,
    what_this_model_cannot_know,
)

MTD_NAME = MTDPolicy().describe()
ADP_NAME = AdaptiveDosePolicy().describe()


@pytest.fixture(scope="module")
def gbm():
    return make_gbm_phantom(size=16, seed=3)


# ---------------------------------------------------------------------------
# Geometry and initial conditions
# ---------------------------------------------------------------------------


def test_phantom_invariants(gbm):
    assert gbm.tumor_mask.shape == (16, 16, 16)
    assert gbm.tumor_mask.sum() > 200
    # Hypoxic core proliferates slower than the rim.
    r = np.sqrt(
        (np.arange(16)[:, None, None] - 7.5) ** 2
        + (np.arange(16)[None, :, None] - 7.5) ** 2
        + (np.arange(16)[None, None, :] - 7.5) ** 2
    )
    core = gbm.tumor_mask & (r < 0.55 * 0.38 * 16)
    rim = gbm.tumor_mask & (r > 0.85 * 0.38 * 16)
    assert core.any() and rim.any()
    assert gbm.proliferative[core].max() < gbm.proliferative[rim].min()
    # Vasculature exists and reaches the tumour region.
    assert gbm.vessels.max() == 1.0
    assert (gbm.vessels > 0).sum() > 50


def test_seed_layout(gbm):
    sim = TumorEvolutionSimulator(gbm)
    n_s, n_r = sim.seed(0.01, seed=11)
    total = n_s.sum() + n_r.sum()
    assert abs(n_r.sum() / total - 0.01) < 0.002
    # R is a compact local patch (~2% of tumour voxels), not sprinkled.
    n_r_voxels = int((n_r > 0).sum())
    assert 9 <= n_r_voxels <= 0.03 * gbm.tumor_mask.sum()
    # Seeded below spatial carrying capacity so dynamics can run.
    assert n_s.max() <= DEFAULT_CLONES.carrying_capacity * 0.66
    # Fraction 0.0 is a DEFINED configuration, not an error: no resistant
    # clone at diagnosis, so any resistance in the run has to be created by
    # mutation. Without it, 'resistance evolves' is untestable -- the
    # clustered-seed path always places R in at least 9 voxels.
    n_s0, n_r0 = sim.seed(0.0, seed=11)
    assert n_r0.sum() == 0.0
    assert n_s0.sum() > 0.0
    with pytest.raises(ValueError):
        sim.seed(1.0)
    with pytest.raises(ValueError):
        sim.seed(-0.1)


# ---------------------------------------------------------------------------
# Policy logic (no simulation)
# ---------------------------------------------------------------------------


def _state(burden, r_frac, b_init=100.0, day=0):
    return PolicyState(day=day, burden=burden, burden_initial=b_init, resistant_fraction=r_frac)


def test_collateral_switch_hysteresis():
    pol = CollateralSwitchPolicy(theta_on=0.15, theta_off=0.05)
    # Below off-threshold -> Drug A. Above on-threshold -> Drug B.
    assert pol.dose(_state(100.0, 0.01)) == (1.0, 0.0)
    assert pol.dose(_state(100.0, 0.20)) == (0.0, 1.0)
    # Between thresholds: hysteresis holds the current regime (no chatter).
    assert pol.dose(_state(100.0, 0.10)) == (0.0, 1.0)
    # Falling back below the off-threshold returns to Drug A.
    assert pol.dose(_state(100.0, 0.04)) == (1.0, 0.0)
    # reset() clears per-trial state.
    pol.reset()
    assert pol.dose(_state(100.0, 0.10)) == (1.0, 0.0)
    with pytest.raises(ValueError):
        CollateralSwitchPolicy(theta_on=0.15, theta_off=0.15)


def test_mtd_and_adaptive_policies():
    assert MTDPolicy().dose(_state(50.0, 0.9)) == (1.0, 0.0)
    adp = AdaptiveDosePolicy()
    assert adp.dose(_state(101.0, 0.01, b_init=100.0)) == (1.0, 0.0)
    assert adp.dose(_state(100.0, 0.01, b_init=100.0)) == (0.0, 0.0)
    assert adp.dose(_state(99.0, 0.01, b_init=100.0)) == (0.0, 0.0)


def test_make_policy_factory():
    assert make_policy(PolicyKind.MTD).kind is PolicyKind.MTD
    assert make_policy(PolicyKind.ADAPTIVE).kind is PolicyKind.ADAPTIVE
    sw = make_policy(PolicyKind.COLLATERAL_SWITCH, theta_on=0.3)
    assert sw.theta_on == 0.3
    with pytest.raises(ValueError):
        make_policy("not-a-kind")


# ---------------------------------------------------------------------------
# Dynamics (small phantoms, pinned against the calibration runs)
# ---------------------------------------------------------------------------


def test_mtd_drives_resistant_dominance(gbm):
    sim = TumorEvolutionSimulator(gbm)
    traj = sim.run(MTDPolicy(), days=150, initial_resistant_fraction=0.01, seed=7)
    assert traj.resistant_dominance_day is not None
    assert traj.resistant_dominance_day < 60
    assert traj.final_resistant_fraction > 0.5


def test_adaptive_extends_pfs_reduces_toxicity_and_delays_dominance(gbm):
    rep = run_trial(
        gbm,
        arms=[MTDPolicy(), AdaptiveDosePolicy()],
        days=300,
        n_replicates=3,
        initial_resistant_fraction=0.01,
        seed0=40,
    )
    mtd, adp = rep.arms[MTD_NAME], rep.arms[ADP_NAME]
    # Direction: steering beats continuous MTD.
    assert adp.median_pfs_days > mtd.median_pfs_days
    # Magnitude: inside the MEASURED model-class band, below the 2.34 anchor.
    assert MEASURED_ADAPTIVE_MTD_RATIO_BAND[0] <= rep.adaptive_vs_mtd_pfs_ratio
    assert rep.adaptive_vs_mtd_pfs_ratio <= MEASURED_ADAPTIVE_MTD_RATIO_BAND[1]
    assert ADAPTIVE_TTP_RATIO_ANCHOR > MEASURED_ADAPTIVE_MTD_RATIO_BAND[1]
    # Toxicity proxy: adaptive doses a fraction of the time MTD does.
    assert adp.toxicity_reduction_vs_mtd > 0.2
    # MTD selects for resistance much earlier than adaptive dosing.
    assert mtd.median_resistance_dominance_day is not None
    assert adp.median_resistance_dominance_day is None or (
        adp.median_resistance_dominance_day > mtd.median_resistance_dominance_day
    )


def test_collateral_switch_controls_tumour_at_defaults(gbm):
    sim = TumorEvolutionSimulator(gbm)
    traj = sim.run(CollateralSwitchPolicy(), days=300, initial_resistant_fraction=0.01, seed=40)
    assert traj.pfs_value(300) >= 300.0  # censored: no progression
    assert traj.progressed is False
    assert traj.final_resistant_fraction < 0.5


def test_threshold_sweep_contingency_on_collateral_network(gbm):
    rep = threshold_sweep(
        gbm,
        thetas=(0.10, 0.40),
        days=200,
        n_replicates=2,
        seed0=100,
    )
    assert rep.best_theta in rep.thetas
    assert rep.best_band[0] <= rep.best_theta <= rep.best_band[1]
    assert len(rep.median_pfs) == len(rep.thetas)
    assert len(rep.median_final_resistant_fraction) == len(rep.thetas)
    # PFS is censored in BOTH networks at this horizon: burden control is a
    # property of the policy shape, not the drug-pair network. The network's
    # effect shows in whether the resistant clone is ever cleared.
    assert max(rep.no_collateral_median_pfs) == max(rep.median_pfs)
    # Without collateral sensitivity the resistant clone runs away:
    # final R fraction must be far higher than with a true collateral pair.
    assert max(rep.no_collateral_final_resistant_fraction) > 0.4
    assert max(rep.median_final_resistant_fraction) < 0.4
    assert any("resistant fraction" in n or "contingent" in n for n in rep.notes)


def test_sanctuary_sequestration_under_mtd(gbm):
    sim = TumorEvolutionSimulator(gbm)
    # Measure mid-trial: the metric degrades to ~0 once R fills every voxel.
    traj = sim.run(
        MTDPolicy(), days=60, initial_resistant_fraction=0.01, seed=7, keep_fields=True
    )
    contrast = sanctuary_sequestration(traj, gbm)
    # The resistant clone is enriched where Drug A is weakest.
    assert contrast > 0.05
    with pytest.raises(ValueError):
        sanctuary_sequestration(
            sim.run(MTDPolicy(), days=20, initial_resistant_fraction=0.01, seed=1), gbm
        )


def test_run_is_deterministic_given_seed(gbm):
    sim = TumorEvolutionSimulator(gbm)
    a = sim.run(AdaptiveDosePolicy(), days=80, initial_resistant_fraction=0.01, seed=5)
    b = sim.run(AdaptiveDosePolicy(), days=80, initial_resistant_fraction=0.01, seed=5)
    assert np.array_equal(a.burden_S, b.burden_S)
    assert np.array_equal(a.burden_R, b.burden_R)


def test_dose_logs_and_toxicity_bookkeeping(gbm):
    sim = TumorEvolutionSimulator(gbm)
    traj = sim.run(MTDPolicy(), days=100, initial_resistant_fraction=0.01, seed=2)
    assert traj.dose_A.max() <= 1.0 and traj.dose_A.min() >= 0.0
    assert traj.dose_B.max() <= 1.0 and traj.dose_B.min() >= 0.0
    # MTD: full dose every day -> toxicity equals days * weight_A.
    assert traj.toxicity == pytest.approx(100.0 * DEFAULT_DRUGS.toxicity_weight[0])


def test_pfs_censoring_semantics():
    traj = SimulationTrajectory(
        policy="mtd",
        days=10,
        burden_S=np.zeros(10),
        burden_R=np.zeros(10),
        dose_A=np.ones(10),
        dose_B=np.zeros(10),
        pfs_day=None,
        progressed=False,
        initial_resistant_fraction=0.01,
        toxicity=10.0,
        final_resistant_fraction=0.0,
        resistant_dominance_day=None,
    )
    assert traj.pfs_value(365) == 365.0
    traj.pfs_day = 7
    assert traj.pfs_value(365) == 7.0


def test_run_validates_inputs(gbm):
    sim = TumorEvolutionSimulator(gbm)
    with pytest.raises(ValueError):
        sim.run(MTDPolicy(), days=1)


# ---------------------------------------------------------------------------
# Honest limits
# ---------------------------------------------------------------------------


def test_provenance_pins_failure_regime_as_not_reproduced():
    prov = evosteer_provenance()
    # The literature anchor is established...
    assert prov["adaptive_therapy_failure_regime_low_resistant_fraction"] is (
        EvidenceStatus.ESTABLISHED
    )
    # ...but THIS spatial model could not reproduce it, and says so.
    assert prov["failure_regime_reproduction_in_spatial_abm"] is EvidenceStatus.PROVEN_FAILED
    # cure.py's EvidenceStatus has no ASSUMED member: unmeasured constants
    # are UNKNOWN (no adequate evidence either way) until benched.
    assert prov["gbm_growth_and_kill_constants"] is EvidenceStatus.UNKNOWN


def test_what_this_model_cannot_know():
    limits = what_this_model_cannot_know()
    assert len(limits) >= 4
    joined = " ".join(limits)
    assert "immune" in joined
    assert "sanctuar" in joined or "vascular" in joined
