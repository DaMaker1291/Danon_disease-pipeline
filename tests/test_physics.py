"""Tests for the first-principles hemodynamics module.

Two kinds of assertion here:

* **Correctness** - does the derived physics reproduce known physiology?
  Coronary Reynolds numbers must be laminar, Womersley numbers must land in the
  published 2-4 window, and LAD flow must come out near physiological values.
  These catch unit errors and formula mistakes.

* **Similarity** - does the rig solver actually produce a rig that matches the
  target dimensionless groups? If not, the benchtop experiment cannot validate
  the simulation and the whole project premise collapses.
"""

from __future__ import annotations

import math

import pytest

from danon.physics import (
    BLOOD_DENSITY_KG_M3,
    BLOOD_VISCOSITY_PA_S,
    BODY_TEMPERATURE_K,
    BenchRigSolver,
    Fluid,
    Provenance,
    angular_frequency,
    audit_constants,
    characterize,
    coronary_presets,
    flow_rate_ml_per_min,
    poiseuille_resistance,
    provenance_summary,
    reynolds_number,
    shear_rate,
    stokes_einstein_diffusion,
    volume_flow_rate,
    wall_shear_stress,
)


@pytest.fixture
def blood() -> Fluid:
    return Fluid.blood()


# ---------------------------------------------------------------------------
# Stokes-Einstein
# ---------------------------------------------------------------------------


class TestStokesEinstein:
    def test_matches_hand_calculation(self, blood: Fluid) -> None:
        """D = kT/(6 pi mu a) for a 100 nm particle in blood at 37 C."""
        expected = (1.380649e-23 * BODY_TEMPERATURE_K) / (
            6.0 * math.pi * BLOOD_VISCOSITY_PA_S * 50e-9
        )
        got = stokes_einstein_diffusion(100e-9, blood)
        assert got == pytest.approx(expected, rel=1e-9)
        assert got * 1e12 == pytest.approx(1.298, abs=0.01)

    def test_diffusion_falls_with_particle_size(self, blood: Fluid) -> None:
        small = stokes_einstein_diffusion(10e-9, blood)
        large = stokes_einstein_diffusion(1000e-9, blood)
        assert small > large

    def test_diffusion_falls_with_viscosity(self) -> None:
        water = stokes_einstein_diffusion(100e-9, Fluid.glycerol_water(0.0))
        thick = stokes_einstein_diffusion(100e-9, Fluid.glycerol_water(60.0))
        assert thick < water

    def test_hindered_diffusion_is_lower(self, blood: Fluid) -> None:
        free = stokes_einstein_diffusion(100e-9, blood)
        hindered = stokes_einstein_diffusion(100e-9, blood, hindered=True)
        assert hindered < free

    def test_rejects_zero_diameter(self, blood: Fluid) -> None:
        with pytest.raises(ValueError):
            stokes_einstein_diffusion(0.0, blood)


# ---------------------------------------------------------------------------
# Physiology sanity checks
# ---------------------------------------------------------------------------


class TestPhysiologicalSanity:
    def test_coronary_flow_is_laminar(self, blood: Fluid) -> None:
        for seg in coronary_presets().values():
            g = characterize(seg, blood)
            assert g.reynolds < 1000.0, f"{seg.name}: Re={g.reynolds} not laminar"

    def test_reynolds_order_of_magnitude(self, blood: Fluid) -> None:
        g = characterize(coronary_presets()["LAD_proximal"], blood)
        assert 100.0 <= g.reynolds <= 400.0

    def test_womersley_is_physiological(self, blood: Fluid) -> None:
        """Coronary Womersley numbers are reported in the ~2-4 range."""
        g = characterize(coronary_presets()["LAD_proximal"], blood)
        assert 1.5 <= g.womersley <= 4.0

    def test_wall_shear_stress_in_arterial_range(self, blood: Fluid) -> None:
        """Healthy coronary wall shear stress is of order 1-3 Pa."""
        g = characterize(coronary_presets()["LAD_proximal"], blood)
        assert 0.5 <= g.shear_stress_pa <= 4.0

    def test_stenosis_raises_shear_dramatically(self, blood: Fluid) -> None:
        healthy = characterize(coronary_presets()["LAD_proximal"], blood)
        stenotic = characterize(coronary_presets()["LAD_stenotic"], blood)
        assert stenotic.shear_stress_pa > 5.0 * healthy.shear_stress_pa

    def test_lad_flow_rate_is_physiological(self, blood: Fluid) -> None:
        """Total coronary flow ~250 mL/min; the LAD takes roughly half.

        This is the assertion that catches the mL/s vs mL/min unit error.
        """
        g = characterize(coronary_presets()["LAD_proximal"], blood)
        assert 60.0 <= g.flow_rate_ml_min <= 200.0, (
            f"LAD flow {g.flow_rate_ml_min} mL/min is not physiological"
        )

    def test_peclet_is_large(self, blood: Fluid) -> None:
        """Advection utterly dominates diffusion for nanoparticles in arteries."""
        g = characterize(coronary_presets()["LAD_proximal"], blood, particle_diameter_m=100e-9)
        assert g.peclet > 1e6


# ---------------------------------------------------------------------------
# Flow relations
# ---------------------------------------------------------------------------


class TestFlowRelations:
    def test_reynolds_scales_linearly_with_velocity(self, blood: Fluid) -> None:
        a = reynolds_number(0.1, 3e-3, blood)
        b = reynolds_number(0.2, 3e-3, blood)
        assert b == pytest.approx(2 * a, rel=1e-9)

    def test_shear_stress_relation(self, blood: Fluid) -> None:
        tau = wall_shear_stress(0.2, 1.5e-3, blood)
        assert tau == pytest.approx(4 * BLOOD_VISCOSITY_PA_S * 0.2 / 1.5e-3, rel=1e-9)

    def test_shear_rate_is_quaternary_over_radius(self) -> None:
        assert shear_rate(0.2, 1e-3) == pytest.approx(800.0, rel=1e-9)

    def test_resistance_scales_as_radius_to_the_minus_four(self) -> None:
        r1 = poiseuille_resistance(3.5e-3, 0.01, 1e-3)
        r2 = poiseuille_resistance(3.5e-3, 0.01, 5e-4)
        assert r2 / r1 == pytest.approx(16.0, rel=1e-6)

    def test_flow_rate_unit_conversion(self) -> None:
        """10 mL/min through a 1 mm radius tube: verify the arithmetic."""
        q_m3_s = flow_rate_ml_per_min(1.0, 1e-3) / (1e6 * 60.0)
        assert q_m3_s == pytest.approx(math.pi * 1e-6, rel=1e-9)
        assert volume_flow_rate(1.0, 1e-3) == pytest.approx(math.pi * 1e-6, rel=1e-9)

    def test_angular_frequency(self) -> None:
        assert angular_frequency(60.0) == pytest.approx(2 * math.pi, rel=1e-9)


# ---------------------------------------------------------------------------
# Blood analog
# ---------------------------------------------------------------------------


class TestBloodAnalog:
    def test_37_percent_glycerol_matches_blood_viscosity(self) -> None:
        a = Fluid.glycerol_water(37.0)
        assert a.viscosity_pa_s == pytest.approx(BLOOD_VISCOSITY_PA_S, rel=0.05)

    def test_water_table_lookup(self) -> None:
        assert Fluid.glycerol_water(0.0).viscosity_pa_s == pytest.approx(1.0e-3, rel=1e-9)

    def test_viscosity_increases_with_glycerol(self) -> None:
        vals = [Fluid.glycerol_water(p).viscosity_pa_s for p in (0, 20, 37, 60)]
        for a, b in zip(vals, vals[1:]):
            assert b > a

    def test_density_increases_with_glycerol(self) -> None:
        assert Fluid.glycerol_water(50.0).density_kg_m3 > Fluid.glycerol_water(0.0).density_kg_m3

    def test_kinematic_viscosity(self) -> None:
        f = Fluid.glycerol_water(37.0)
        assert f.kinematic_viscosity_m2_s == pytest.approx(
            f.viscosity_pa_s / f.density_kg_m3, rel=1e-9
        )


# ---------------------------------------------------------------------------
# Rig derivation — the point of the module
# ---------------------------------------------------------------------------


class TestRigSolver:
    def test_solver_reproduces_target_groups(self) -> None:
        """A rig is only useful if it matches Re, Wo and tau simultaneously."""
        target = coronary_presets()["LAD_proximal"]
        solver = BenchRigSolver(Fluid.glycerol_water(37.0))
        rig = solver.solve(target)
        report = solver.verify(target, rig)
        assert report.max_error < 1e-3
        assert report.is_matched()

    def test_matched_analog_runs_at_human_heart_rate(self) -> None:
        """The headline result: a viscosity-matched analog needs no scaling."""
        solver = BenchRigSolver(Fluid.glycerol_water(37.0))
        rig = solver.solve(coronary_presets()["LAD_proximal"])
        assert 55.0 <= rig.pulse_frequency_bpm <= 85.0
        assert 2.5 <= rig.channel_diameter_mm <= 4.5

    def test_water_requires_scaling_and_faster_pulsing(self) -> None:
        """Water is too thin, so the rig must be scaled down and sped up."""
        water_rig = BenchRigSolver(Fluid.glycerol_water(0.0)).solve(
            coronary_presets()["LAD_proximal"]
        )
        analog_rig = BenchRigSolver(Fluid.glycerol_water(37.0)).solve(
            coronary_presets()["LAD_proximal"]
        )
        assert water_rig.channel_diameter_mm < analog_rig.channel_diameter_mm
        assert water_rig.pulse_frequency_bpm > analog_rig.pulse_frequency_bpm

    def test_all_analogs_achieve_matched_similarity(self) -> None:
        target = coronary_presets()["LAD_proximal"]
        for pct in (0.0, 20.0, 37.0):
            solver = BenchRigSolver(Fluid.glycerol_water(pct))
            rig = solver.solve(target)
            assert solver.verify(target, rig).is_matched(), f"{pct}% failed"

    def test_flow_rate_is_achievable_with_a_benchtop_pump(self) -> None:
        solver = BenchRigSolver(Fluid.glycerol_water(37.0))
        rig = solver.solve(coronary_presets()["LAD_proximal"])
        # peristaltic and syringe pumps comfortably cover this range
        assert 1.0 <= rig.flow_rate_ml_min <= 1000.0

    def test_rig_shear_matches_target_shear(self) -> None:
        target = coronary_presets()["LAD_proximal"]
        solver = BenchRigSolver(Fluid.glycerol_water(37.0))
        rig = solver.solve(target)
        target_tau = characterize(target, Fluid.blood()).shear_stress_pa
        assert rig.wall_shear_stress_pa == pytest.approx(target_tau, rel=1e-3)

    def test_solver_reports_scaling_notes(self) -> None:
        rig = BenchRigSolver(Fluid.glycerol_water(0.0)).solve(
            coronary_presets()["LAD_proximal"]
        )
        assert rig.notes
        assert any("pulse" in n.lower() or "channel" in n.lower() or "Re=" in n for n in rig.notes)

    def test_stenotic_target_gives_narrower_or_lower_flow_rig(self) -> None:
        solver = BenchRigSolver(Fluid.glycerol_water(37.0))
        healthy = solver.solve(coronary_presets()["LAD_proximal"])
        stenotic = solver.solve(coronary_presets()["LAD_stenotic"])
        # the stenotic lesion carries much higher shear, so the rig must too
        assert stenotic.wall_shear_stress_pa > healthy.wall_shear_stress_pa


# ---------------------------------------------------------------------------
# Constant provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_audit_lists_constants(self) -> None:
        consts = audit_constants()
        assert len(consts) >= 8
        assert all(c.unit for c in consts)
        assert all(c.note for c in consts)

    def test_audit_flags_the_unmeasured_parameters(self) -> None:
        """The things the rig must measure must be visibly tagged ASSUMED."""
        assumed = [c.name for c in audit_constants() if c.provenance is Provenance.ASSUMED]
        for expected in ("shear_half_max_pa", "plaque_erosion_rate_per_s"):
            assert expected in assumed

    def test_physiological_values_are_tagged_literature(self) -> None:
        by_name = {c.name: c for c in audit_constants()}
        assert by_name["blood_viscosity_pa_s"].provenance is Provenance.LITERATURE
        assert by_name["lad_diameter_mm"].provenance is Provenance.LITERATURE

    def test_provenance_summary_counts_match(self) -> None:
        summary = provenance_summary()
        assert sum(summary.values()) == len(audit_constants())
        assert summary.get("assumed", 0) > 0
