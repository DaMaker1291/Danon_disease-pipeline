"""Tests for the vectorised swarm ABM, campaign harness and validation maths.

Runs are kept deliberately small so the suite stays fast; the physics assertions
are about direction and bounds, not absolute magnitudes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from danon.swarm import (
    ArteryGeometry,
    SwarmCampaign,
    SwarmConfig,
    SwarmSimulator,
    correlation_report,
    INFLAMMATION_THRESHOLD,
)

FAST = dict(n_agents=4000, steps=120)


@pytest.fixture
def geometry() -> ArteryGeometry:
    return ArteryGeometry()


def run(**kwargs):
    cfg = SwarmConfig(**(FAST | kwargs))
    return SwarmSimulator(cfg, ArteryGeometry()).run()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_radius_is_narrowest_at_stenosis_centre(self, geometry: ArteryGeometry) -> None:
        xs = np.array([0.0, geometry.stenosis_center_um, geometry.length_um])
        radii = geometry.radius_at(xs)
        assert radii[1] == pytest.approx(geometry.min_radius_um, rel=1e-6)
        assert radii[0] > radii[1] and radii[2] > radii[1]

    def test_occlusion_percentage_sets_min_radius(self) -> None:
        g = ArteryGeometry(initial_occlusion_pct=70.0)
        assert g.min_radius_um == pytest.approx(0.30 * g.healthy_radius_um, rel=1e-9)

    def test_plaque_fraction_peaks_at_one(self, geometry: ArteryGeometry) -> None:
        peak = float(geometry.plaque_fraction(np.array([geometry.stenosis_center_um]))[0])
        assert peak == pytest.approx(1.0, rel=1e-6)

    def test_resistance_rises_with_occlusion(self) -> None:
        light = ArteryGeometry(initial_occlusion_pct=20.0).resistance()
        heavy = ArteryGeometry(initial_occlusion_pct=80.0).resistance()
        assert heavy > light


# ---------------------------------------------------------------------------
# Simulator behaviour
# ---------------------------------------------------------------------------


class TestSwarmPhysics:
    def test_zero_potency_yields_no_effect(self) -> None:
        r = run(potency=0.0)
        assert r.plaque_reduction_pct == 0.0
        assert r.pressure_drop_reduction_pct == 0.0
        assert r.reperfusion_gain_pct == 0.0
        assert r.distal_embolic_particle_count == 0

    def test_plaque_reduction_increases_with_potency(self) -> None:
        series = [run(potency=p).plaque_reduction_pct for p in (0.2, 0.5, 1.0)]
        for a, b in zip(series, series[1:]):
            assert b >= a, f"not monotonic in potency: {series}"
        assert series[-1] > series[0]

    def test_binding_is_not_total(self) -> None:
        """A model where every agent immediately sticks carries no information."""
        r = run(avidity=0.6, potency=0.5)
        assert r.bound_fraction < 1.0
        assert 0.0 <= r.bound_fraction <= 1.0

    def test_lower_avidity_reduces_plaque_effect(self) -> None:
        low = run(avidity=0.05, potency=0.8).plaque_reduction_pct
        high = run(avidity=0.9, potency=0.8).plaque_reduction_pct
        assert high > low

    def test_off_target_adhesion_is_measurable_and_distinct(self) -> None:
        """The inflammation threshold must let healthy-wall adhesion register."""
        r = run(avidity=0.05, potency=0.5)
        assert 0.0 <= r.off_target_adhesion_pct <= 100.0
        assert INFLAMMATION_THRESHOLD > 0.0
        assert r.off_target_adhesion_pct > 0.0

    def test_embolism_tracks_potency(self) -> None:
        """Aggressive digestion is the mechanism for embolic debris."""
        low = run(potency=0.1, steps=400).distal_embolic_particle_count
        high = run(potency=1.0, steps=400).distal_embolic_particle_count
        assert high >= low

    def test_all_percentages_bounded(self) -> None:
        for pot in (0.0, 0.5, 1.0):
            r = run(potency=pot)
            for name, val in (
                ("plaque", r.plaque_reduction_pct),
                ("dp", r.pressure_drop_reduction_pct),
                ("reperfusion", r.reperfusion_gain_pct),
                ("off_target", r.off_target_adhesion_pct),
            ):
                assert 0.0 <= val <= 100.0, f"{name}={val} out of range"

    def test_reperfusion_never_exceeds_physical_ceiling(self) -> None:
        """Flow gain cannot exceed complete resolution to a healthy vessel."""
        g = ArteryGeometry()
        ceiling = 100.0 * (g.resistance() / g.resistance(0.0) - 1.0)
        r = SwarmSimulator(SwarmConfig(**FAST, potency=1.0), g).run()
        assert r.reperfusion_gain_pct <= ceiling + 1e-6

    def test_agent_steps_accounted_exactly(self) -> None:
        cfg = SwarmConfig(n_agents=3000, steps=50)
        r = SwarmSimulator(cfg, ArteryGeometry()).run()
        assert r.agent_steps == 3000 * 50

    def test_deterministic_for_fixed_seed(self) -> None:
        a = run(seed=42, potency=0.7)
        b = run(seed=42, potency=0.7)
        assert a.plaque_reduction_pct == b.plaque_reduction_pct
        assert a.bound_fraction == b.bound_fraction

    def test_metrics_match_endpoint_vocabulary(self) -> None:
        m = run().as_metrics()
        for key in (
            "plaque_reduction_pct",
            "pressure_drop_reduction_pct",
            "reperfusion_gain_pct",
            "off_target_healthy_wall_adhesion_pct",
        ):
            assert key in m
        # as_metrics() reports raw physical values only; the normalised
        # efficacy/safety axes come from normalise_to_reference alone, so a
        # single definition of each exists.
        assert "efficacy" not in m
        assert "safety" not in m

    def test_normalised_axes_are_bounded_and_derived(self) -> None:
        from danon.swarm import normalise_to_reference

        g = ArteryGeometry()
        r = SwarmSimulator(SwarmConfig(**FAST), g).run()
        m = normalise_to_reference(r, g)
        for key in ("efficacy", "safety", "plaque_reduction_pct", "reperfusion_gain_pct"):
            assert 0.0 <= m[key] <= 100.0
        # The physical values must survive unmodified alongside the scaled ones.
        assert m["plaque_reduction_physical_pct"] == r.plaque_reduction_pct
        assert m["off_target_physical_pct"] == r.off_target_adhesion_pct


# ---------------------------------------------------------------------------
# Campaign
# ---------------------------------------------------------------------------


class TestCampaign:
    def test_sweep_is_full_factorial_with_replicates(self) -> None:
        cfgs = SwarmCampaign.sweep_configs(
            SwarmConfig(**FAST),
            avidities=(0.2, 0.8),
            potencies=(0.3, 0.9),
            vcam1_levels=(0.5,),
            seeds=(0, 1),
        )
        assert len(cfgs) == 2 * 2 * 1 * 2

    def test_campaign_reports_error_bars(self) -> None:
        campaign = SwarmCampaign()
        cfgs = SwarmCampaign.sweep_configs(
            SwarmConfig(**FAST), avidities=(0.2, 0.8), potencies=(0.4,), vcam1_levels=(0.7,), seeds=(0, 1)
        )
        report, results = campaign.run(cfgs, study_id="t")
        assert report.runs == len(cfgs)
        assert report.total_agent_steps == sum(r.agent_steps for r in results)
        for key in report.endpoint_means:
            assert key in report.endpoint_ci95
            assert report.endpoint_ci95[key] >= 0.0
        # with replicates, at least one endpoint should show spread
        assert any(v > 0 for v in report.endpoint_stdev.values())

    def test_campaign_respects_agent_step_budget(self) -> None:
        campaign = SwarmCampaign()
        cfgs = SwarmCampaign.sweep_configs(SwarmConfig(**FAST), seeds=(0, 1, 2, 3))
        budget = 2 * 4000 * 120
        report, results = campaign.run(cfgs, budget_agent_steps=budget)
        assert len(results) < len(cfgs)
        assert any("truncated" in n.lower() for n in report.notes)

    def test_campaign_ranks_configs(self) -> None:
        campaign = SwarmCampaign()
        cfgs = SwarmCampaign.sweep_configs(
            SwarmConfig(**FAST), avidities=(0.1, 0.9), potencies=(0.9,), vcam1_levels=(0.7,), seeds=(0,)
        )
        report, _ = campaign.run(cfgs)
        assert report.top_configs
        assert report.top_configs[0]["composite"] >= report.top_configs[-1]["composite"]

    def test_headline_mentions_agent_steps(self) -> None:
        campaign = SwarmCampaign()
        report, _ = campaign.run(SwarmCampaign.sweep_configs(SwarmConfig(**FAST), seeds=(0,)))
        assert "agent-steps" in report.headline()


# ---------------------------------------------------------------------------
# Validation maths
# ---------------------------------------------------------------------------


class TestCorrelationReport:
    def test_perfect_agreement_gives_r2_one(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        rep = correlation_report(x, x, n_bootstrap=200)
        assert rep["r2"] == pytest.approx(1.0, abs=1e-9)
        assert rep["slope"] == pytest.approx(1.0, abs=1e-9)
        assert rep["intercept"] == pytest.approx(0.0, abs=1e-9)

    def test_recovers_known_slope(self) -> None:
        x = np.linspace(1.0, 10.0, 12)
        y = 0.9 * x + 0.5
        rep = correlation_report(x, y, n_bootstrap=200)
        assert rep["slope"] == pytest.approx(0.9, abs=1e-6)
        assert rep["intercept"] == pytest.approx(0.5, abs=1e-6)

    def test_ci_brackets_estimate(self) -> None:
        rng = np.random.default_rng(0)
        x = np.linspace(1.0, 20.0, 20)
        y = 0.9 * x + rng.normal(0.0, 0.5, 20)
        rep = correlation_report(x, y, n_bootstrap=400)
        assert rep["r2_ci95_low"] <= rep["r2"] <= rep["r2_ci95_high"]
        assert rep["slope_ci95_low"] <= rep["slope"] <= rep["slope_ci95_high"]

    def test_uncorrelated_data_gives_low_r2(self) -> None:
        rng = np.random.default_rng(3)
        x = rng.normal(0.0, 1.0, 40)
        y = rng.normal(0.0, 1.0, 40)
        rep = correlation_report(x, y, n_bootstrap=200)
        assert rep["r2"] < 0.5

    def test_rejects_mismatched_lengths(self) -> None:
        with pytest.raises(ValueError):
            correlation_report([1.0, 2.0, 3.0], [1.0, 2.0])

    def test_rejects_too_few_points(self) -> None:
        with pytest.raises(ValueError):
            correlation_report([1.0, 2.0], [1.0, 2.0])

    def test_rmse_reported(self) -> None:
        rep = correlation_report([1.0, 2.0, 3.0, 4.0], [1.1, 1.9, 3.2, 3.8], n_bootstrap=100)
        assert rep["rmse"] >= 0.0
        assert rep["n_pairs"] == 4.0
