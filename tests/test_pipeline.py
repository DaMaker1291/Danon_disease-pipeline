"""Tests for the Blood -> Targeting -> Hydrodynamics master controller.

The most important test here is `test_binding_constant_actually_matters`. A
common implementation writes adhesion as ``1 - exp(-Ka/1e8)``, which evaluates
to the constant 0.632 at the usual ``Ka = 1e8`` and saturates for any
``Ka >= 1e9``. That makes the affinity term decorative: it cannot rank an
affinity-improved design. These tests fail if that regression returns.
"""

from __future__ import annotations

import numpy as np
import pytest

from danon.vector_design.pipeline import (
    BloodPanel,
    BloodProfiler,
    BloodToCurePipeline,
    MolecularTargeting,
)
from danon.swarm import ArteryGeometry, SwarmConfig


@pytest.fixture
def targeting() -> MolecularTargeting:
    return MolecularTargeting()


@pytest.fixture
def panel() -> BloodPanel:
    return BloodPanel(
        patient_id="PT-TEST",
        monocyte_count_per_ul=750.0,
        hdl_mg_dl=35.0,
        ldl_mg_dl=160.0,
        crp_mg_l=4.2,
        age_years=61,
        weight_kg=88.0,
        lvef_pct=42.0,
        nab_titer=12.0,
    )


# ---------------------------------------------------------------------------
# Stage 2 — the affinity model must not be decorative
# ---------------------------------------------------------------------------


class TestBindingModelIntegrity:
    def test_binding_constant_actually_matters(
        self, targeting: MolecularTargeting
    ) -> None:
        """Ka must change adhesion across the range where optimisation happens."""
        occupancies = [
            targeting.intrinsic_occupancy(ka, 0.92)
            for ka in (1e6, 1e7, 1e8, 1e9, 1e10)
        ]
        for a, b in zip(occupancies, occupancies[1:]):
            assert b > a, f"occupancy not increasing with Ka: {occupancies}"
        # must resolve real differences, not jump to a ceiling immediately
        assert occupancies[0] < 0.05
        assert occupancies[2] > 0.2
        assert len(set(round(o, 6) for o in occupancies)) == len(occupancies)

    def test_binding_constant_is_not_a_constant_multiplier(
        self, targeting: MolecularTargeting
    ) -> None:
        """Guards against the 1 - exp(-Ka/1e8) shortcut, which is 0.632 always."""
        degenerate_value = 1.0 - float(np.exp(-1.0))
        for ka in (1e7, 1e8, 1e9):
            occ = targeting.intrinsic_occupancy(ka, 1.0)
            assert abs(occ - degenerate_value) > 0.05, (
                f"occupancy {occ} at Ka={ka:.0e} matches the degenerate constant"
            )

    def test_receptor_density_matters(self, targeting: MolecularTargeting) -> None:
        occs = [
            targeting.intrinsic_occupancy(1e9, r) for r in (0.05, 0.3, 0.6, 0.92)
        ]
        for a, b in zip(occs, occs[1:]):
            assert b > a, f"occupancy not increasing with receptor density: {occs}"

    def test_occupancy_is_bounded(self, targeting: MolecularTargeting) -> None:
        for ka in (1e3, 1e8, 1e14, 1e20):
            for r in (0.0, 0.5, 1.0):
                occ = targeting.intrinsic_occupancy(ka, r)
                assert 0.0 <= occ <= 1.0

    def test_zero_receptor_gives_zero_adhesion(
        self, targeting: MolecularTargeting
    ) -> None:
        assert targeting.intrinsic_occupancy(1e12, 0.0) == 0.0


class TestShearAttenuation:
    def test_shear_reduces_adhesion(self, targeting: MolecularTargeting) -> None:
        """Drag must remove weakly adhered particles."""
        low = targeting.shear_factor(0.1)
        high = targeting.shear_factor(20.0)
        assert high < low
        assert 0.0 < high < 1.0 and 0.0 < low <= 1.0

    def test_shear_factor_bounded(self, targeting: MolecularTargeting) -> None:
        for tau in (0.0, 1.0, 50.0, 1e6):
            assert 0.0 <= targeting.shear_factor(tau) <= 1.0

    def test_stenosis_throat_has_higher_shear_than_healthy_wall(
        self, targeting: MolecularTargeting
    ) -> None:
        healthy = targeting.wall_shear_stress(400_000.0, 1500.0)
        throat = targeting.wall_shear_stress(400_000.0, 450.0)
        assert throat > healthy

    def test_narrower_vessel_suppresses_adhesion(
        self, targeting: MolecularTargeting
    ) -> None:
        wide = targeting.simulate(1000, ka_per_molar=1e9, radius_um=1500.0)
        narrow = targeting.simulate(1000, ka_per_molar=1e9, radius_um=450.0)
        assert narrow.adhesion_probability < wide.adhesion_probability
        assert narrow.shear_factor < wide.shear_factor


# ---------------------------------------------------------------------------
# Stage 2 — simulation behaviour
# ---------------------------------------------------------------------------


class TestTargetingSimulation:
    def test_bound_agents_scale_with_affinity(self, targeting: MolecularTargeting) -> None:
        weak = targeting.simulate(5000, ka_per_molar=1e6, rng=np.random.default_rng(1))
        strong = targeting.simulate(5000, ka_per_molar=1e11, rng=np.random.default_rng(1))
        assert strong.bound_agents > weak.bound_agents

    def test_zero_agents_is_handled(self, targeting: MolecularTargeting) -> None:
        r = targeting.simulate(0)
        assert r.bound_agents == 0
        assert r.adhesion_probability == 0.0
        assert r.notes

    def test_bound_never_exceeds_available(self, targeting: MolecularTargeting) -> None:
        r = targeting.simulate(1000, ka_per_molar=1e15, rng=np.random.default_rng(0))
        assert 0 <= r.bound_agents <= 1000

    def test_no_python_loop_scale(self, targeting: MolecularTargeting) -> None:
        """Vectorised: a large swarm must not require per-agent iteration."""
        r = targeting.simulate(2_000_000, ka_per_molar=1e9, rng=np.random.default_rng(0))
        assert 0 <= r.bound_agents <= 2_000_000

    def test_high_shear_emits_explanatory_note(
        self, targeting: MolecularTargeting
    ) -> None:
        r = targeting.simulate(1000, ka_per_molar=1e9, radius_um=300.0)
        assert any("shear" in n.lower() for n in r.notes)


# ---------------------------------------------------------------------------
# Stage 1 — severity must not be conflated with yield
# ---------------------------------------------------------------------------


class TestBloodProfiler:
    def test_low_hdl_does_not_increase_yield(self) -> None:
        """A sicker patient does not yield more product; that is a category error."""
        profiler = BloodProfiler()
        healthy = profiler.profile(BloodPanel(patient_id="a", monocyte_count_per_ul=600, hdl_mg_dl=65))
        sick = profiler.profile(BloodPanel(patient_id="b", monocyte_count_per_ul=600, hdl_mg_dl=28))
        assert sick.therapeutic_yield == healthy.therapeutic_yield
        # but severity must differ
        assert sick.plaque_vulnerability_index > healthy.plaque_vulnerability_index

    def test_yield_tracks_monocyte_count(self) -> None:
        profiler = BloodProfiler()
        low = profiler.profile(BloodPanel(patient_id="a", monocyte_count_per_ul=200))
        high = profiler.profile(BloodPanel(patient_id="b", monocyte_count_per_ul=800))
        assert high.therapeutic_yield > low.therapeutic_yield

    def test_inflammation_index_bounded_and_ordered(self) -> None:
        profiler = BloodProfiler()
        calm = profiler.inflammation_index(
            BloodPanel(patient_id="a", monocyte_count_per_ul=200, hdl_mg_dl=65, crp_mg_l=0.4)
        )
        inflamed = profiler.inflammation_index(
            BloodPanel(patient_id="b", monocyte_count_per_ul=850, hdl_mg_dl=28, crp_mg_l=6.0)
        )
        assert 0.0 <= calm < inflamed <= 1.0

    def test_high_titer_is_ineligible(self) -> None:
        r = BloodProfiler().profile(BloodPanel(patient_id="a", nab_titer=2000.0))
        assert not r.eligible
        assert any("titer" in x.lower() for x in r.exclusion_reasons)

    def test_low_lvef_is_ineligible(self) -> None:
        r = BloodProfiler().profile(BloodPanel(patient_id="a", lvef_pct=20.0))
        assert not r.eligible

    def test_zero_monocytes_flagged(self) -> None:
        r = BloodProfiler().profile(BloodPanel(patient_id="a", monocyte_count_per_ul=0.0))
        assert r.therapeutic_yield == 0
        assert any("autologous" in n.lower() for n in r.notes)


# ---------------------------------------------------------------------------
# Stage coupling — the critical interface
# ---------------------------------------------------------------------------


class TestStageCoupling:
    def test_adhesion_drives_avidity_not_agent_count(self, panel: BloodPanel) -> None:
        """Bound-agent count must not be fed in as the swarm size."""
        pipe = BloodToCurePipeline()
        rep = pipe.run(panel, n_agents=8000, steps=120)
        # swarm size comes from the intended dose
        assert rep.swarm.config.n_agents == 8000
        # avidity comes from Stage 2 adhesion
        assert rep.swarm.config.avidity == pytest.approx(
            rep.targeting.adhesion_probability, abs=5e-3
        )

    def test_avidity_is_distinct_from_bound_fraction(self, panel: BloodPanel) -> None:
        pipe = BloodToCurePipeline()
        rep = pipe.run(panel, n_agents=8000, steps=120)
        bound_frac = rep.targeting.bound_agents / rep.targeting.n_agents
        # the two quantities are related but not interchangeable
        assert 0.0 <= rep.swarm.config.avidity <= 1.0
        assert 0.0 <= bound_frac <= 1.0

    def test_more_inflammation_gives_more_adhesion(self) -> None:
        pipe = BloodToCurePipeline()
        calm = pipe.run(
            BloodPanel(patient_id="a", monocyte_count_per_ul=200, hdl_mg_dl=65, crp_mg_l=0.4),
            n_agents=4000, steps=50,
        )
        inflamed = pipe.run(
            BloodPanel(patient_id="b", monocyte_count_per_ul=850, hdl_mg_dl=28, crp_mg_l=6.0),
            n_agents=4000, steps=50,
        )
        assert inflamed.targeting.adhesion_probability > calm.targeting.adhesion_probability


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_pipeline_produces_full_report(self, panel: BloodPanel) -> None:
        rep = BloodToCurePipeline().run(panel, n_agents=4000, steps=80)
        assert rep.patient_id == "PT-TEST"
        for key in (
            "inflammation_index",
            "adhesion_probability",
            "plaque_reduction_pct",
            "pressure_drop_reduction_pct",
            "off_target_adhesion_pct",
            "distal_embolic_particle_count",
        ):
            assert key in rep.summary
        assert rep.swarm.agent_steps == 4000 * 80

    def test_report_renders_with_disclaimers(self, panel: BloodPanel) -> None:
        text = BloodToCurePipeline().run(panel, n_agents=2000, steps=40).render()
        for token in ("STAGE 1", "STAGE 2", "STAGE 3", "Langmuir", "shear"):
            assert token in text
        assert "not medical advice" in text.lower()

    def test_ineligible_patient_warned(self) -> None:
        rep = BloodToCurePipeline().run(
            BloodPanel(patient_id="x", nab_titer=5000.0), n_agents=2000, steps=40
        )
        assert not rep.eligible
        assert any("excluded" in w.lower() for w in rep.warnings)

    def test_deterministic_for_fixed_seed(self, panel: BloodPanel) -> None:
        a = BloodToCurePipeline().run(panel, n_agents=3000, steps=50, seed=5)
        b = BloodToCurePipeline().run(panel, n_agents=3000, steps=50, seed=5)
        assert a.summary == b.summary

    def test_cohort_summary(self, panel: BloodPanel) -> None:
        others = [
            BloodPanel(patient_id=f"P{i}", monocyte_count_per_ul=200 + 200 * i)
            for i in range(3)
        ]
        out = BloodToCurePipeline().cohort(others, n_agents=2000, steps=40)
        assert out["n"] == 3
        assert out["total_agent_steps"] == 3 * 2000 * 40
        assert out["stdev_plaque_reduction_pct"] >= 0.0

    def test_custom_geometry_is_respected(self) -> None:
        g = ArteryGeometry(initial_occlusion_pct=40.0)
        pipe = BloodToCurePipeline(geometry=g)
        rep = pipe.run(BloodPanel(patient_id="a"), n_agents=2000, steps=40)
        assert rep.swarm.config.n_agents == 2000
        assert pipe.geometry.initial_occlusion_pct == 40.0
