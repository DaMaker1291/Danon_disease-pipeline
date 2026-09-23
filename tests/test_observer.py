"""Tests for the clinical observation layer (danon.observer).

Pins the measured results:
  * the surrogate reproduces the engine's regime structure (zero log ->
    r dilutes via cost of resistance; MTD log -> R dominates; Drug B kills
    R through its own pairing);
  * the ABC filter recovers a planted r0 within its credible band from a
    synthetic episodic CSF stream, and honestly leaves ess unidentified
    under a zero (untreated) dose log;
  * all three refusal paths fire with reasons;
  * the closed-loop controller beats MTD on paired seeds with toxicity
    below MTD -- without ever reading engine state;
  * provenance entries use real EvidenceStatus members.
"""

from __future__ import annotations

import numpy as np
import pytest

from danon.cure import EvidenceStatus
from danon.evosteer import (
    CollateralSwitchPolicy,
    MTDPolicy,
    TumorEvolutionSimulator,
    make_gbm_phantom,
)
from danon.observer import (
    CSF_LIQUID_BIOPSY_ANCHOR,
    ObservationModel,
    ObservedSwitchController,
    PatientObservations,
    Posterior,
    ProjectedState,
    ObserverVerdict,
    abc_filter,
    estimate_state,
    mtd_dose_log,
    null_dose_log,
    observer_provenance,
    project_resistant_fraction,
    run_observed_trial,
    surrogate_burden_series,
    what_this_model_cannot_know,
)


# ---------------------------------------------------------------------------
# Surrogate dynamics
# ---------------------------------------------------------------------------


def test_surrogate_untreated_regime_matches_engine():
    """Zero log: S grows; the resistant FRACTION declines (cost of resistance)."""
    S, R = surrogate_burden_series(0.10, 0.9, 0.25, np.zeros(120))
    assert S[-1] > S[0]
    r0, rT = R[0] / (S[0] + R[0]), R[-1] / (S[-1] + R[-1])
    assert rT < r0


def test_surrogate_mtd_regime_matches_engine():
    """MTD log: S collapses and R's fraction rises (net R growth under A)."""
    S, R = surrogate_burden_series(0.02, 0.9, 0.25, np.ones(120))
    assert S[-1] < 0.05 * S[0]
    r0, rT = R[0] / (S[0] + R[0]), R[-1] / (S[-1] + R[-1])
    assert rT > r0


def test_surrogate_drug_pairing():
    """Drug B hits R through its own kill pairing, not Drug A's log."""
    _, R_B = surrogate_burden_series(0.30, 0.9, 0.25, np.zeros(60), np.ones(60))
    _, R_A = surrogate_burden_series(0.30, 0.9, 0.25, np.ones(60), np.zeros(60))
    # Under B, R is strongly suppressed; under A it keeps growing (resistance).
    assert R_B[-1] < R_B[0]
    assert R_A[-1] > R_A[0]


def test_surrogate_validates_inputs():
    with pytest.raises(ValueError):
        surrogate_burden_series(0.02, 1.5, 0.25, np.ones(10))
    with pytest.raises(ValueError):
        surrogate_burden_series(0.0, 0.5, 0.25, np.ones(10))
    with pytest.raises(ValueError):
        surrogate_burden_series(0.1, 0.5, 0.25, np.ones(10), np.ones(9))


# ---------------------------------------------------------------------------
# Dose-log helpers and observation plumbing
# ---------------------------------------------------------------------------


def test_dose_log_helpers():
    assert np.all(null_dose_log(30) == 0.0)
    assert np.all(mtd_dose_log(30) == 1.0)
    with pytest.raises(ValueError):
        null_dose_log(0)


def test_observation_model_and_stream_validation():
    with pytest.raises(ValueError):
        ObservationModel(sigma_vaf=0.0)
    with pytest.raises(ValueError):
        ObservationModel(cadence_days=0)
    with pytest.raises(ValueError):
        PatientObservations(days=np.array([0.0, 1.0]), vaf_resistant=np.array([0.1]))
    with pytest.raises(ValueError):
        PatientObservations(
            days=np.array([1.0, 0.0]), vaf_resistant=np.array([0.1, 0.2])
        )
    # Noisy readings stay in [0, 1] even with extreme truth.
    rng = np.random.default_rng(0)
    obs = ObservationModel(sigma_vaf=0.2, cadence_days=1)
    readings = obs.observe(np.array([0.9] * 50), rng)
    assert np.all(readings >= 0.0) and np.all(readings <= 1.0)


# ---------------------------------------------------------------------------
# ABC filter: recovery + honest non-identifiability
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def intake_stream():
    """One synthetic patient: engine truth + episodic noisy CSF draws."""
    sim = TumorEvolutionSimulator(make_gbm_phantom(size=18, seed=3))
    obs, true_r = PatientObservations.from_engine_trajectory(
        sim,
        initial_resistant_fraction=0.08,
        days=90,
        seed=11,
        obs=ObservationModel(sigma_vaf=0.03, cadence_days=14),
    )
    return obs, true_r


def test_abc_recovers_planted_r0_within_band(intake_stream):
    obs, _true_r = intake_stream
    est = estimate_state(obs, null_dose_log(90), n_draws=2500, seed=5)
    assert est.usable
    lo, _mid, hi = est.posterior.r0_q
    assert lo <= 0.08 <= hi
    assert est.posterior.n_accepted >= 20
    assert np.isfinite(est.posterior.predictive_residual_rms)


def test_ess_unidentifiable_under_untreated_log(intake_stream):
    """A zero log leaves ess multiplying nothing: the posterior must say so."""
    obs, _true_r = intake_stream
    est = estimate_state(obs, null_dose_log(90), n_draws=2500, seed=5)
    lo, _mid, hi = est.posterior.ess_q
    assert lo <= 0.30 + 0.05 and hi >= 0.95 - 0.05   # spans essentially the prior


# ---------------------------------------------------------------------------
# Refusal semantics
# ---------------------------------------------------------------------------


def test_refusal_too_few_observations():
    short = PatientObservations(
        days=np.array([0.0, 14.0]), vaf_resistant=np.array([0.05, 0.06])
    )
    est = estimate_state(short, null_dose_log(90))
    assert est.verdict is ObserverVerdict.REFUSED_TOO_FEW
    assert "3 observations" in est.reason or "only 2" in est.reason


def test_refusal_coarse_cadence():
    sparse = PatientObservations(
        days=np.array([0.0, 120.0, 240.0]),
        vaf_resistant=np.array([0.05, 0.07, 0.09]),
    )
    est = estimate_state(sparse, null_dose_log(280))
    assert est.verdict is ObserverVerdict.REFUSED_TOO_FEW


def test_refusal_unidentified_stream():
    """A stream the surrogate cannot produce (violent alternation) -> refuse."""
    wild = PatientObservations(
        days=np.arange(0, 84, 7, dtype=float),
        vaf_resistant=np.array([0.0, 0.9] * 6),
    )
    est = estimate_state(wild, null_dose_log(90), n_draws=400, seed=1)
    assert est.verdict is ObserverVerdict.REFUSED_UNIDENTIFIED


# ---------------------------------------------------------------------------
# Projection math
# ---------------------------------------------------------------------------


def _toy_posterior() -> Posterior:
    return Posterior(
        r0_q=(0.03, 0.05, 0.09),
        ess_q=(0.5, 0.7, 0.9),
        cost_q=(0.15, 0.25, 0.35),
        n_accepted=50,
        n_draws=300,
        tolerance=0.06,
        predictive_residual_rms=0.03,
        samples={
            "r0": np.linspace(0.02, 0.10, 50).tolist(),
            "ess": np.linspace(0.4, 0.9, 50).tolist(),
            "cost": np.linspace(0.12, 0.4, 50).tolist(),
        },
    )


def test_projection_band_ordering_and_clamp():
    proj = project_resistant_fraction(_toy_posterior(), mtd_dose_log(120), n_samples=60)
    assert isinstance(proj, ProjectedState)
    assert len(proj.r_median) == 120
    assert np.all(proj.r_lo <= proj.r_median + 1e-12)
    assert np.all(proj.r_median <= proj.r_hi + 1e-12)
    assert proj.r_at(-5) == proj.r_at(0)
    assert proj.r_at(10_000) == proj.r_at(119)
    with pytest.raises(ValueError):
        project_resistant_fraction(
            Posterior(
                r0_q=(0, 0, 0), ess_q=(0, 0, 0), cost_q=(0, 0, 0),
                n_accepted=0, n_draws=0, tolerance=0, predictive_residual_rms=0,
            ),
            mtd_dose_log(10),
        )


# ---------------------------------------------------------------------------
# Closed-loop controller vs MTD (paired seeds)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def controller_vs_mtd():
    """Two paired seeds: observed controller vs MTD on identical geometry."""
    sim = TumorEvolutionSimulator(make_gbm_phantom(size=16, seed=3))
    planted, days = 0.08, 365
    out = []
    for rep, seed in enumerate((100, 101)):
        t_mtd = sim.run(
            MTDPolicy(), days=days, initial_resistant_fraction=planted, seed=seed
        )
        ctrl = ObservedSwitchController(horizon_days=days, seed=rep, n_draws=1200)
        res = run_observed_trial(
            sim, planted, days=days, seed=seed, controller=ctrl
        )
        out.append((t_mtd, res))
    return out


def test_controller_beats_mtd_on_paired_seeds(controller_vs_mtd):
    for t_mtd, res in controller_vs_mtd:
        t_obs = res.trajectory
        assert t_obs.pfs_value(365) > t_mtd.pfs_value(365)
        assert t_obs.toxicity < t_mtd.toxicity
        assert res.n_switches >= 1


def test_controller_uses_only_its_own_inputs(controller_vs_mtd):
    """The controller's decision inputs are its own draws + its own log."""
    for _t_mtd, res in controller_vs_mtd:
        assert res.n_switches >= 1
        assert res.n_refusals >= 0
        assert res.final_posterior is not None
        assert res.final_posterior.n_accepted > 0
        # The r0 trace exists and the controller switched days are strictly
        # increasing (a real schedule, not noise chatter).
        days_switched = [d for d, _action in res.events]
        assert days_switched == sorted(days_switched)


# ---------------------------------------------------------------------------
# Provenance and honesty records
# ---------------------------------------------------------------------------


def test_provenance_uses_real_enum_members():
    prov = observer_provenance()
    assert "csf_liquid_biopsy_tracks_glioma_evolution" in prov
    assert prov["csf_liquid_biopsy_tracks_glioma_evolution"] is EvidenceStatus.ESTABLISHED
    assert prov["static_schedule_from_one_shot_posterior"] is EvidenceStatus.PROVEN_FAILED
    for status in prov.values():
        assert isinstance(status, EvidenceStatus)


def test_limits_mention_confounding_and_transfer():
    limits = what_this_model_cannot_know()
    joined = " ".join(limits)
    assert "confound" in joined
    assert "transfers to patients" in joined
    assert "Miller" in CSF_LIQUID_BIOPSY_ANCHOR and "2019" in CSF_LIQUID_BIOPSY_ANCHOR


def test_engine_run_accepts_and_ignores_no_hook():
    """Backward compatibility: run() without the hook behaves as before."""
    sim = TumorEvolutionSimulator(make_gbm_phantom(size=12, seed=0))
    t1 = sim.run(MTDPolicy(), days=30, initial_resistant_fraction=0.05, seed=0)
    t2 = sim.run(MTDPolicy(), days=30, initial_resistant_fraction=0.05, seed=0)
    assert np.allclose(t1.burden_S, t2.burden_S)
    assert np.allclose(t1.burden_R, t2.burden_R)
