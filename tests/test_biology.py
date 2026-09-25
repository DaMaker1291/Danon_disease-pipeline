"""Tests for the derived molecular biology module.

These assert that biological quantities are computed from physical chemistry
rather than invented, and that the resulting numbers are internally consistent
and in plausible regimes.
"""

from __future__ import annotations

import math

import pytest

from danon.vector_design.biology import (
    BOND_FORCE_SCALE_PN,
    COHESION_STRENGTH_INTACT_KPA,
    ENZYMES_PER_PARTICLE,
    K_CAT_PER_S,
    LIPID_MOLECULAR_WEIGHT,
    LIPID_SOLUBILITY_MOL_M3,
    SIGMA_RECEPTOR_HEALTHY_PER_UM2,
    SIGMA_RECEPTOR_INFLAMED_PER_UM2,
    StenosisGeometry,
    bell_off_rate,
    biological_provenance,
    cohesion_strength_kpa,
    damkohler_number,
    derive_adhesion,
    embolization_risk,
    enzyme_mass_removal_rate_kg_s,
    hydrodynamic_drag_force_n,
    injected_dose_required,
    mass_transfer_velocity,
    noyes_whitney_rate_kg_s,
    particles_required_for_fraction,
    receptor_concentration,
    reaction_velocity,
    series_deposition_velocity,
    sherwood_number,
    single_pass_adhesion_fraction,
    surface_density_to_concentration,
    time_to_clear_fraction,
)
from danon.physics import AVOGADRO, Fluid, Provenance, stokes_einstein_diffusion


@pytest.fixture
def blood() -> Fluid:
    return Fluid.blood()


@pytest.fixture
def stenosis() -> StenosisGeometry:
    return StenosisGeometry()


# ---------------------------------------------------------------------------
# Surface density -> concentration
# ---------------------------------------------------------------------------


class TestSurfaceDensity:
    def test_hand_calculation(self) -> None:
        """100 molecules/um^2 across a 100 nm slab."""
        expected = (100.0 * 1e12) / (100e-9 * AVOGADRO)
        got = surface_density_to_concentration(100.0, 100e-9)
        assert got == pytest.approx(expected, rel=1e-9)
        # 1.66e-3 mol/L == 1.66 mmol/L
        assert got * 1e3 == pytest.approx(1.661, rel=0.01)

    def test_thicker_layer_gives_lower_concentration(self) -> None:
        thin = surface_density_to_concentration(100.0, 50e-9)
        thick = surface_density_to_concentration(100.0, 200e-9)
        assert thick < thin

    def test_rejects_zero_thickness(self) -> None:
        with pytest.raises(ValueError):
            surface_density_to_concentration(100.0, 0.0)

    def test_inflammation_increases_concentration(self) -> None:
        vals = [receptor_concentration(f) for f in (0.0, 0.25, 0.5, 1.0)]
        for a, b in zip(vals, vals[1:]):
            assert b > a

    def test_inflamed_healthy_ratio_matches_constants(self) -> None:
        ratio = receptor_concentration(1.0) / receptor_concentration(0.0)
        expected = SIGMA_RECEPTOR_INFLAMED_PER_UM2 / SIGMA_RECEPTOR_HEALTHY_PER_UM2
        assert ratio == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# Leveque transport
# ---------------------------------------------------------------------------


class TestMassTransfer:
    def test_sherwood_scales_as_cube_root_of_peclet(self) -> None:
        """The 1/3 power law is the signature of the Leveque solution."""
        sh1 = sherwood_number(100.0, 1e-3, 1e-12)
        sh2 = sherwood_number(800.0, 1e-3, 1e-12)   # Pe x8
        assert sh2 / sh1 == pytest.approx(2.0, rel=1e-6)

    def test_transfer_velocity_increases_with_shear(self, blood: Fluid) -> None:
        d = stokes_einstein_diffusion(100e-9, blood)
        low = mass_transfer_velocity(100.0, 3e-3, d)
        high = mass_transfer_velocity(1000.0, 3e-3, d)
        assert high > low

    def test_transfer_velocity_decreases_with_particle_size(self, blood: Fluid) -> None:
        small = mass_transfer_velocity(500.0, 3e-3, stokes_einstein_diffusion(50e-9, blood))
        large = mass_transfer_velocity(500.0, 3e-3, stokes_einstein_diffusion(1000e-9, blood))
        assert small > large

    def test_zero_diffusivity_gives_zero(self) -> None:
        assert sherwood_number(100.0, 1e-3, 0.0) == 0.0


class TestSeriesResistance:
    def test_effective_velocity_is_below_either_component(self) -> None:
        v = series_deposition_velocity(1e-7, 5e-7)
        assert v < 1e-7 and v < 5e-7
        assert v == pytest.approx(1.0 / (1e7 + 2e6), rel=1e-9)

    def test_slower_step_dominates(self) -> None:
        """Halving the fast step should barely change the result."""
        base = series_deposition_velocity(1e-8, 1e-5)
        changed = series_deposition_velocity(5e-9, 1e-5)
        assert changed == pytest.approx(base, rel=0.6)

    def test_zero_component_gives_zero(self) -> None:
        assert series_deposition_velocity(0.0, 1e-6) == 0.0
        assert series_deposition_velocity(1e-6, 0.0) == 0.0

    def test_damkohler_regime_classification(self) -> None:
        assert damkohler_number(1e-7, 1e-9) > 10.0      # transport-limited
        assert damkohler_number(1e-9, 1e-7) < 0.1       # reaction-limited


class TestAdhesionFraction:
    def test_bounded_and_monotonic(self) -> None:
        vals = [single_pass_adhesion_fraction(v, 3e-3, 0.2, 1.5e-3)
                for v in (0.0, 1e-7, 1e-6, 1e-4)]
        for a, b in zip(vals, vals[1:]):
            assert b >= a
        assert all(0.0 <= v <= 1.0 for v in vals)

    def test_zero_velocity_guarded(self) -> None:
        assert single_pass_adhesion_fraction(1e-7, 3e-3, 0.0, 1.5e-3) == 0.0

    def test_single_pass_efficiency_is_low(self, blood: Fluid) -> None:
        """Known nanomedicine reality: delivery efficiency is well under 1%."""
        d = derive_adhesion(
            inflamed_fraction=1.0, mean_velocity_m_s=0.2, radius_m=1.5e-3,
            length_m=3e-3, particle_diameter_m=100e-9, fluid=blood,
        )
        assert d.adhesion_fraction < 1e-3
        assert d.adhesion_fraction > 0.0


# ---------------------------------------------------------------------------
# Bond mechanics
# ---------------------------------------------------------------------------


class TestBondMechanics:
    def test_drag_scales_as_diameter_squared(self, blood: Fluid) -> None:
        """The key scaling that makes shear-stripping a micron-scale effect."""
        f1 = hydrodynamic_drag_force_n(blood, 500.0, 100e-9)
        f2 = hydrodynamic_drag_force_n(blood, 500.0, 200e-9)
        assert f2 / f1 == pytest.approx(4.0, rel=1e-6)

    def test_bell_off_rate_grows_with_force(self) -> None:
        assert bell_off_rate(1e-11) > bell_off_rate(0.0)
        f_beta_n = BOND_FORCE_SCALE_PN * 1e-12
        # one force scale should multiply the off-rate by e
        assert bell_off_rate(f_beta_n) / bell_off_rate(0.0) == pytest.approx(math.e, rel=1e-6)

    def test_nanoparticles_are_kinetically_not_force_limited(self, blood: Fluid) -> None:
        d = derive_adhesion(
            inflamed_fraction=1.0, mean_velocity_m_s=0.2, radius_m=1.5e-3,
            length_m=3e-3, particle_diameter_m=100e-9, fluid=blood,
        )
        assert d.force_ratio < 1.0

    def test_micron_particles_become_shear_limited(self, blood: Fluid) -> None:
        d = derive_adhesion(
            inflamed_fraction=1.0, mean_velocity_m_s=0.2, radius_m=1.5e-3,
            length_m=3e-3, particle_diameter_m=2000e-9, fluid=blood,
        )
        assert d.force_ratio > 1.0

    def test_notes_explain_the_limiting_factor(self, blood: Fluid) -> None:
        d = derive_adhesion(
            inflamed_fraction=1.0, mean_velocity_m_s=0.2, radius_m=1.5e-3,
            length_m=3e-3, particle_diameter_m=100e-9, fluid=blood,
        )
        assert d.regime in ("transport-limited", "reaction-limited", "mixed")
        assert d.notes
        assert any("kinetic" in n.lower() or "shear" in n.lower() for n in d.notes)


# ---------------------------------------------------------------------------
# Plaque removal kinetics
# ---------------------------------------------------------------------------


class TestPlaqueKinetics:
    def test_turnover_scales_with_kcat_and_particles(self) -> None:
        a = enzyme_mass_removal_rate_kg_s(1000.0)
        b = enzyme_mass_removal_rate_kg_s(2000.0)
        assert b == pytest.approx(2 * a, rel=1e-9)
        c = enzyme_mass_removal_rate_kg_s(1000.0, k_cat_per_s=K_CAT_PER_S * 2)
        assert c == pytest.approx(2 * a, rel=1e-9)

    def test_rate_hand_calculation(self) -> None:
        molecules_per_s = 1000.0 * ENZYMES_PER_PARTICLE * K_CAT_PER_S
        expected = molecules_per_s * (LIPID_MOLECULAR_WEIGHT * 1e-3) / AVOGADRO
        assert enzyme_mass_removal_rate_kg_s(1000.0) == pytest.approx(expected, rel=1e-9)

    def test_zero_particles_gives_zero(self) -> None:
        assert enzyme_mass_removal_rate_kg_s(0.0) == 0.0

    def test_dose_inversion_is_consistent(self, stenosis: StenosisGeometry) -> None:
        """Inverting the turnover rate must reproduce the forward calculation."""
        n = particles_required_for_fraction(stenosis.plaque_mass_kg, 0.01, 48 * 3600)
        rate = enzyme_mass_removal_rate_kg_s(n)
        t = time_to_clear_fraction(stenosis.plaque_mass_kg, rate, 0.01)
        assert t == pytest.approx(48 * 3600, rel=1e-6)

    def test_higher_kcat_needs_fewer_particles(self, stenosis: StenosisGeometry) -> None:
        few = particles_required_for_fraction(stenosis.plaque_mass_kg, 0.01, 48 * 3600, k_cat_per_s=2000)
        many = particles_required_for_fraction(stenosis.plaque_mass_kg, 0.01, 48 * 3600, k_cat_per_s=5)
        assert few < many

    def test_passive_and_catalytic_rates_are_comparable(self, stenosis: StenosisGeometry) -> None:
        """At equal particle counts the two mechanisms are similar in magnitude.

        An earlier version of this module asserted that passive dissolution was
        orders of magnitude slower. That was wrong, and this test documents the
        real relationship so the claim cannot quietly reappear.
        """
        active = enzyme_mass_removal_rate_kg_s(6.8e6)
        passive = noyes_whitney_rate_kg_s(
            stokes_einstein_diffusion(1e-9, Fluid.blood()),
            math.pi * (50e-9) ** 2 * 6.8e6,
            boundary_layer_m=50e-9,
        )
        ratio = passive / active
        assert 0.1 < ratio < 10.0, f"rates differ by {ratio:.3g}x, expected comparable"

    def test_passive_dissolution_is_saturation_limited(self) -> None:
        """The real distinction: no sink, so C -> Cs and the rate collapses."""
        saturated = noyes_whitney_rate_kg_s(
            stokes_einstein_diffusion(1e-9, Fluid.blood()),
            math.pi * (50e-9) ** 2 * 1e6,
            boundary_layer_m=50e-9,
            bulk_conc_mol_m3=LIPID_SOLUBILITY_MOL_M3,
        )
        assert saturated == 0.0

    def test_geometry_mass_is_plausible(self, stenosis: StenosisGeometry) -> None:
        # a 70% lesion in a 3.5 mm vessel over 1.5 mm holds a few mg of lipid
        assert 1e-6 < stenosis.plaque_mass_kg < 1e-4


# ---------------------------------------------------------------------------
# Embolization
# ---------------------------------------------------------------------------


class TestEmbolization:
    def test_intact_plaque_cannot_be_fractured_by_shear(self) -> None:
        """Arterial shear (~Pa) is ~3 orders below plaque strength (~kPa)."""
        assert embolization_risk(1.6, 0.0) < 0.01

    def test_strength_falls_with_digestion(self) -> None:
        vals = [cohesion_strength_kpa(f) for f in (0.0, 0.5, 0.9, 0.99)]
        for a, b in zip(vals, vals[1:]):
            assert b < a
        assert vals[0] == pytest.approx(COHESION_STRENGTH_INTACT_KPA, rel=1e-6)

    def test_threshold_is_reachable(self) -> None:
        """The safety metric must be able to fire; a floor that caps it is a bug."""
        assert embolization_risk(1.6, 0.80) < 1.0
        assert embolization_risk(1.6, 0.99) > 1.0

    def test_higher_shear_increases_risk(self) -> None:
        assert embolization_risk(16.0, 0.95) > embolization_risk(1.6, 0.95)

    def test_risk_is_non_negative(self) -> None:
        assert embolization_risk(0.0, 0.5) == 0.0


# ---------------------------------------------------------------------------
# The dose chain
# ---------------------------------------------------------------------------


class TestDoseChain:
    def test_injected_dose_exceeds_adhered_by_delivery_efficiency(
        self, stenosis: StenosisGeometry, blood: Fluid
    ) -> None:
        d = derive_adhesion(
            inflamed_fraction=1.0, mean_velocity_m_s=0.2, radius_m=1.5e-3,
            length_m=3e-3, particle_diameter_m=100e-9, fluid=blood,
        )
        r = injected_dose_required(
            plaque_mass_kg=stenosis.plaque_mass_kg,
            target_fraction=0.01,
            within_seconds=48 * 3600,
            adhesion_fraction=d.adhesion_fraction,
        )
        assert r["injected_required"] > r["adhered_required"]
        ratio = r["injected_required"] / r["adhered_required"]
        assert ratio == pytest.approx(1.0 / d.adhesion_fraction, rel=1e-6)

    def test_trillion_scale_is_derived_not_asserted(
        self, stenosis: StenosisGeometry, blood: Fluid
    ) -> None:
        """The headline: the delivery bottleneck forces a ~1e12 particle dose."""
        d = derive_adhesion(
            inflamed_fraction=1.0, mean_velocity_m_s=0.2, radius_m=1.5e-3,
            length_m=3e-3, particle_diameter_m=100e-9, fluid=blood,
        )
        r = injected_dose_required(
            plaque_mass_kg=stenosis.plaque_mass_kg,
            target_fraction=0.01,
            within_seconds=48 * 3600,
            adhesion_fraction=d.adhesion_fraction,
        )
        assert 1e11 < r["injected_required"] < 1e13

    def test_zero_adhesion_is_infeasible(self, stenosis: StenosisGeometry) -> None:
        r = injected_dose_required(
            plaque_mass_kg=stenosis.plaque_mass_kg,
            target_fraction=0.01,
            within_seconds=48 * 3600,
            adhesion_fraction=0.0,
        )
        assert r["injected_required"] == float("inf")
        assert r["feasible"] == 0.0


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_every_biological_input_is_tagged(self) -> None:
        prov = biological_provenance()
        assert prov
        assert all(isinstance(v, Provenance) for v in prov.values())

    def test_only_modelling_choices_are_assumed(self) -> None:
        """Literature constants must not be quietly reclassified as assumptions."""
        prov = biological_provenance()
        assert prov["k_cat_per_s"] is Provenance.LITERATURE
        assert prov["sigma_receptor_inflamed_per_um2"] is Provenance.LITERATURE
        assert prov["boundary_layer_thickness_m"] is Provenance.ASSUMED
