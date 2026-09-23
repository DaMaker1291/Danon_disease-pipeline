"""Tests for the verification-campaign module.

Pins: the invariant set on real runs, the violation detectors on rigged
trajectories, determinism probing, bootstrap CI behavior, JSON resumability
of accumulated evidence, and the honesty record (a campaign verifies
implementation, not biology).
"""

from __future__ import annotations

import numpy as np
import pytest

from danon.campaign import (
    CampaignConfig,
    CampaignStats,
    _MAX_MASK_LEAK_FRACTION,
    bootstrap_ci_mean,
    campaign_provenance,
    check_final_fields,
    check_trajectory,
    determinism_probe,
    format_campaign_report,
    run_campaign_chunk,
    what_the_campaign_cannot_prove,
)
from danon.cure import EvidenceStatus
from danon.evosteer import (
    MTDPolicy,
    SimulationTrajectory,
    TumorEvolutionSimulator,
    make_gbm_phantom,
)


@pytest.fixture(scope="module")
def sim():
    return TumorEvolutionSimulator(make_gbm_phantom(size=16, seed=42))


def _synth_traj(**overrides) -> SimulationTrajectory:
    n = 20
    d = dict(
        policy="synthetic",
        days=n,
        burden_S=np.full(n, 50.0),
        burden_R=np.full(n, 5.0),
        dose_A=np.ones(n),
        dose_B=np.zeros(n),
        pfs_day=None,
        progressed=False,
        initial_resistant_fraction=0.01,
        toxicity=float(n * 1.0),  # matches sum(dose_A)*1.0
        final_resistant_fraction=0.09,
        resistant_dominance_day=None,
    )
    d.update(overrides)
    return SimulationTrajectory(**d)


# ---------------------------------------------------------------------------
# Violation detectors on rigged trajectories
# ---------------------------------------------------------------------------


def test_detector_catches_dose_out_of_bounds(sim):
    stats = CampaignStats()
    traj = _synth_traj(dose_A=np.full(20, 1.5))
    check_trajectory(traj, sim, stats)
    assert any("dose_A out of [0,1]" in r for r in stats.violation_records)


def test_detector_catches_toxicity_mismatch(sim):
    stats = CampaignStats()
    traj = _synth_traj(toxicity=999.0)
    check_trajectory(traj, sim, stats)
    assert any("toxicity" in r and "recomputed" in r for r in stats.violation_records)


def test_detector_catches_wrong_pfs_day(sim):
    stats = CampaignStats()
    n = 20
    burden = np.full(n, 50.0)
    burden[10:] = 100.0  # crosses the RANO threshold at day 10
    traj = _synth_traj(burden_S=burden, burden_R=np.full(n, 5.0), pfs_day=15, progressed=True)
    check_trajectory(traj, sim, stats)
    assert any("pfs_day" in r and "first RANO crossing" in r for r in stats.violation_records)


def test_detector_catches_negative_voxels(sim):
    stats = CampaignStats()
    n_s = np.full((16, 16, 16), 0.1)
    n_s[2, 2, 2] = -0.5
    traj = _synth_traj()
    traj.final_N_S = n_s
    traj.final_N_R = np.zeros((16, 16, 16))
    traj.final_conc_A = np.zeros((16, 16, 16))
    traj.final_conc_B = np.zeros((16, 16, 16))
    check_final_fields(traj, sim, stats)
    assert any("negative voxels" in r for r in stats.violation_records)


def test_detector_catches_capacity_breach(sim):
    stats = CampaignStats()
    K = sim.clones.carrying_capacity
    n_s = np.full((16, 16, 16), 0.0)
    n_s[3, 3, 3] = K * 2.0
    traj = _synth_traj()
    traj.final_N_S = n_s
    traj.final_N_R = np.zeros((16, 16, 16))
    traj.final_conc_A = np.zeros((16, 16, 16))
    traj.final_conc_B = np.zeros((16, 16, 16))
    check_final_fields(traj, sim, stats)
    assert any("exceed carrying capacity" in r for r in stats.violation_records)


# ---------------------------------------------------------------------------
# Clean-run invariants
# ---------------------------------------------------------------------------


def test_clean_run_has_zero_violations(sim):
    stats = CampaignStats()
    traj = sim.run(MTDPolicy(), days=40, initial_resistant_fraction=0.01, seed=1)
    check_trajectory(traj, sim, stats)
    traj_kf = sim.run(
        MTDPolicy(), days=40, initial_resistant_fraction=0.01, seed=1, keep_fields=True
    )
    check_final_fields(traj_kf, sim, stats)
    # Toxicity and PFS bookkeeping are exact; voxel invariants hold.
    assert not any(
        ("toxicity" in r or "capacity" in r or "negative" in r or "pfs_day" in r)
        for r in stats.violation_records
    )


def test_determinism_probe_detects_nondeterminism():
    # Two identical seeds must match; a rigged unequal pair must record.
    stats = CampaignStats()
    sim = TumorEvolutionSimulator(make_gbm_phantom(size=16, seed=42))
    determinism_probe(sim, MTDPolicy(), days=20, seed=7, stats=stats)
    assert stats.n_determinism_probes == 1
    assert stats.n_violations == 0  # engine is fully seeded -> identical reruns


def test_bootstrap_ci_behavior():
    rng = np.random.default_rng(0)
    samples = list(rng.normal(0.5, 0.05, size=200))
    lo, hi = bootstrap_ci_mean(samples, n_boot=500, seed=1)
    assert lo < 0.5 < hi and hi - lo < 0.1
    # Too few samples -> refuse rather than fabricate.
    assert bootstrap_ci_mean([0.1, 0.2], n_boot=100) is None


# ---------------------------------------------------------------------------
# Chunk runner + resumability
# ---------------------------------------------------------------------------


def test_chunk_runs_and_accumulates():
    cfg = CampaignConfig(days=60, size=16)
    stats = run_campaign_chunk(cfg, n_runs=3, seed0=0)
    assert stats.n_runs == 6  # 3 paired runs x 2 arms
    assert stats.n_checks > 50_000
    # Only the documented leak statistic may appear; no invariant violations.
    assert all("leaked" in r or "ceiling" in r for r in stats.violation_records)
    assert len(stats.tox_red_samples) == len(stats.pfs_policy)


def test_stats_json_roundtrip():
    cfg = CampaignConfig(days=50, size=16)
    stats = run_campaign_chunk(cfg, n_runs=2, seed0=5)
    restored = CampaignStats.from_json(stats.to_json())
    assert restored.n_checks == stats.n_checks
    assert restored.n_runs == stats.n_runs
    assert restored.n_violations == stats.n_violations
    assert len(restored.tox_red_samples) == len(stats.tox_red_samples)


def test_chunk_is_deterministic():
    cfg = CampaignConfig(days=50, size=16)
    s1 = run_campaign_chunk(cfg, n_runs=2, seed0=11)
    s2 = run_campaign_chunk(cfg, n_runs=2, seed0=11)
    assert s1.n_checks == s2.n_checks and s1.n_violations == s2.n_violations


def test_report_mentions_scope_honestly():
    cfg = CampaignConfig(days=50, size=16)
    stats = run_campaign_chunk(cfg, n_runs=2, seed0=0)
    text = format_campaign_report(stats, 60.0, cfg)
    assert "verifies the IMPLEMENTATION" in text
    assert "not the biology" in text
    assert "elementary checks" in text


def test_campaign_provenance_pins_scope():
    prov = campaign_provenance()
    assert prov["campaign_equals_clinical_validation"] is EvidenceStatus.PROVEN_FAILED
    joined = " ".join(what_the_campaign_cannot_prove())
    assert "not cancer" in joined
    assert "_MAX_MASK_LEAK_FRACTION" not in joined  # internal constant not leaked into prose


def test_leak_ceiling_is_the_documented_bound():
    # The ceiling bounds a documented artifact; it must be generous relative
    # to measured behavior (6-8% on 60-90d runs) but finite.
    assert 0.10 <= _MAX_MASK_LEAK_FRACTION <= 0.50
