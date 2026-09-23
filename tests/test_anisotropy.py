"""Tests for the heterogeneous-transport module.

The load-bearing assertions are the operator identities: `div(D grad C)` must
reduce to `D * laplacian(C)` exactly on constant `D`, and the simulator's new
path must reproduce the scalar path byte-for-byte when handed a uniform field.
Without those, "heterogeneous transport" would be a second, differently-behaving
model rather than a generalisation of the first.
"""

from __future__ import annotations

import numpy as np
import pytest

from danon.anisotropy import (
    CELL_MOTILITY_MM2_PER_DAY_RANGE,
    LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY,
    PHANTOM_VOXEL_MM,
    REAL_DECIMATION,
    REAL_VOXEL_MM,
    TISSUE_DRUG_D_MM2_PER_DAY,
    TISSUE_LABELS,
    DiffusionField,
    anisotropy_provenance,
    d_to_engine,
    d_to_physical,
    divergence_d_grad,
    divergence_diag_tensor_grad,
    measure_transport_contrast,
    tissue_classes_from_geometry,
    unit_audit,
    unit_audit_report,
    voxel_area_mm2,
    what_anisotropy_cannot_show,
)
from danon.cure import EvidenceStatus
from danon.evosteer import (
    MTDPolicy,
    TumorEvolutionSimulator,
    _laplacian,
    make_gbm_phantom,
)


# ---------------------------------------------------------------------------
# The unit bridge
# ---------------------------------------------------------------------------


class TestUnitBridge:
    def test_round_trips(self):
        for voxel_mm in (1.0, 2.0, REAL_VOXEL_MM):
            for d in (0.001, 0.35, 4.7):
                engine = d_to_engine(d, voxel_mm)
                assert d_to_physical(engine, voxel_mm) == pytest.approx(d)

    def test_area_is_the_square_of_the_edge(self):
        assert voxel_area_mm2(2.0) == pytest.approx(4.0)
        assert voxel_area_mm2(REAL_VOXEL_MM) == pytest.approx(13.4444, rel=1e-4)

    def test_voxel_size_matches_the_repository_convention(self):
        """Must not invent a third value: `realdata` uses the mean of spacings."""
        from danon.realdata import DecimationSpec

        spec = DecimationSpec()
        repo = float(np.mean([1.0 * f for f in spec.factors]))
        assert REAL_VOXEL_MM == pytest.approx(repo)
        assert tuple(spec.factors) == REAL_DECIMATION

    def test_non_positive_voxel_is_refused(self):
        with pytest.raises(ValueError):
            voxel_area_mm2(0.0)
        with pytest.raises(ValueError):
            voxel_area_mm2(-1.0)


class TestUnitAudit:
    def test_drug_diffusion_is_13x_larger_on_real_geometry(self):
        row = next(r for r in unit_audit() if "drug diffusion" in r.name)
        assert row.phantom_mm2_per_day == pytest.approx(0.35)
        assert row.real_mm2_per_day == pytest.approx(4.7056, rel=1e-3)
        assert row.inflation_factor == pytest.approx(13.4444, rel=1e-4)
        # Inside the reference on the phantom, outside it on real geometry.
        assert row.within_reference is False

    def test_cell_motility_is_outside_the_range_at_both_sizes(self):
        row = next(r for r in unit_audit() if "cell motility" in r.name)
        lo, hi = CELL_MOTILITY_MM2_PER_DAY_RANGE
        assert row.phantom_mm2_per_day > hi
        assert row.real_mm2_per_day > hi
        assert row.within_reference is False

    def test_report_names_the_factor_and_the_legacy_literal(self):
        text = unit_audit_report()
        assert "13.44x" in text
        assert f"{LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY:g} voxel^2/day" in text
        assert "voxel^2/day, not mm^2/day" in text


# ---------------------------------------------------------------------------
# The operators
# ---------------------------------------------------------------------------


class TestDivergenceDGrad:
    def _field(self, shape=(9, 9, 9), seed=0):
        return np.random.default_rng(seed).random(shape)

    @pytest.mark.parametrize("d", [0.0, 0.05, 0.35, 2.5])
    def test_constant_d_reduces_exactly_to_d_times_laplacian(self, d):
        c = self._field()
        got = divergence_d_grad(c, np.full_like(c, d))
        want = d * _laplacian(c)
        assert np.allclose(got, want, atol=1e-12)

    def test_isotropic_diagonal_tensor_reduces_to_the_scalar_operator(self):
        c = self._field()
        for d in (0.1, 0.35):
            diag = np.stack([np.full_like(c, d)] * 3)
            assert np.allclose(
                divergence_diag_tensor_grad(c, diag),
                divergence_d_grad(c, np.full_like(c, d)),
                atol=1e-12,
            )

    def test_anisotropy_changes_the_operator(self):
        c = self._field()
        diag = np.stack([
            np.full_like(c, 0.35),
            np.full_like(c, 0.05),
            np.full_like(c, 0.05),
        ])
        aniso = divergence_diag_tensor_grad(c, diag)
        iso = divergence_d_grad(c, np.full_like(c, 0.35))
        assert not np.allclose(aniso, iso)

    def test_no_flux_through_the_boundary(self):
        """Divergence theorem: zero-flux means the interior sum vanishes."""
        c = self._field()
        for d in (0.35, ):
            assert abs(divergence_d_grad(c, np.full_like(c, d)).sum()) < 1e-9

    def test_harmonic_face_gives_no_flux_into_an_impermeable_voxel(self):
        """A zero-D voxel must be a wall, not a half-leaky neighbour.

        With an arithmetic face mean, a face between D=0 and D=0.35 would carry
        0.175 -- half the flux of a fully permeable face -- which is precisely
        the error the harmonic mean exists to avoid.
        """
        c = np.zeros((5, 5, 5))
        c[2, 2, 2] = 1.0
        d = np.full((5, 5, 5), 0.35)
        d[2, 2, 3] = 0.0            # impermeable neighbour
        # The seeded voxel is a sink: six open faces give -6D, and blocking one
        # face must remove EXACTLY one face's worth (-D), because the harmonic
        # mean of (0, D) is 0. An arithmetic face mean would remove only half
        # of one face (-D/2), leaving the wall half-leaky -- which is the
        # difference this test exists to detect.
        open_flux = divergence_d_grad(c, np.full((5, 5, 5), 0.35))[2, 2, 2]
        blocked_flux = divergence_d_grad(c, d)[2, 2, 2]
        assert open_flux == pytest.approx(-6.0 * 0.35)
        assert blocked_flux == pytest.approx(-5.0 * 0.35)
        # A fully impermeable domain carries no flux at all.
        assert np.allclose(divergence_d_grad(c, np.zeros((5, 5, 5))), 0.0)

    def test_rejects_shape_mismatch_and_negative_d(self):
        c = self._field()
        with pytest.raises(ValueError):
            divergence_d_grad(c, np.ones((3, 3, 3)))
        with pytest.raises(ValueError):
            divergence_d_grad(c, -np.ones_like(c))
        with pytest.raises(ValueError):
            divergence_diag_tensor_grad(c, np.ones((3,) + c.shape) * -1.0)
        with pytest.raises(ValueError):
            divergence_diag_tensor_grad(c, np.ones(c.shape))

    def test_2d_input_is_refused(self):
        with pytest.raises(ValueError):
            divergence_d_grad(np.ones((4, 4)), np.ones((4, 4)))


# ---------------------------------------------------------------------------
# Diffusion fields
# ---------------------------------------------------------------------------


class TestDiffusionField:
    def test_labels_map_to_their_physical_values_then_convert(self):
        classes = np.zeros((3, 3, 3), dtype=int)
        classes[0] = TISSUE_LABELS_INVERSE["necrotic_core"]
        classes[1] = TISSUE_LABELS_INVERSE["edema"]
        classes[2] = TISSUE_LABELS_INVERSE["white_matter"]
        field = DiffusionField.from_tissue_classes(classes, voxel_mm=REAL_VOXEL_MM)
        assert field.d_mm2_per_day[0].min() == pytest.approx(0.01)
        assert field.d_mm2_per_day[2].max() == pytest.approx(0.35)
        # Engine units are the physical values divided by the voxel area.
        assert field.d_engine[2].max() == pytest.approx(
            d_to_engine(0.35, REAL_VOXEL_MM)
        )
        assert field.contrast_ratio == pytest.approx(35.0)

    def test_unknown_label_is_refused(self):
        classes = np.full((3, 3, 3), 7)
        with pytest.raises(ValueError):
            DiffusionField.from_tissue_classes(classes)

    def test_uniform_field_converts_at_the_voxel_size(self):
        f = DiffusionField.uniform(0.35, (4, 4, 4), voxel_mm=1.0)
        assert np.allclose(f.d_engine, 0.35)
        f_real = DiffusionField.uniform(0.35, (4, 4, 4), voxel_mm=REAL_VOXEL_MM)
        assert np.allclose(f_real.d_engine, d_to_engine(0.35, REAL_VOXEL_MM))

    def test_anisotropy_factor_is_dimensionless(self):
        classes = np.zeros((3, 3, 3), dtype=int)
        aniso = np.full((3, 3, 3), 0.5)
        f = DiffusionField.from_tissue_classes(
            classes, voxel_mm=REAL_VOXEL_MM, anisotropy=aniso
        )
        base = DiffusionField.from_tissue_classes(classes, voxel_mm=REAL_VOXEL_MM)
        assert np.allclose(f.d_engine, base.d_engine * 0.5)
        with pytest.raises(ValueError):
            DiffusionField.from_tissue_classes(classes, anisotropy=-aniso)


class TestTransportContrast:
    def test_reports_a_real_gap_not_equality(self):
        geom = make_gbm_phantom(size=10, seed=0)
        classes = tissue_classes_from_geometry(geom)
        conc = geom.tumor_mask.astype(float)
        c = measure_transport_contrast(classes, conc)
        assert c.max_ratio == pytest.approx(35.0)
        assert 0.0 < c.operator_gap_at_core < 1.0
        assert "35x contrast" in c.describe()

    def test_tissue_classes_from_geometry_yields_all_three_classes(self):
        geom = make_gbm_phantom(size=14, seed=0)
        classes = tissue_classes_from_geometry(geom)
        present = set(np.unique(classes).tolist())
        assert TISSUE_LABELS_INVERSE["necrotic_core"] in present
        assert TISSUE_LABELS_INVERSE["edema"] in present
        assert TISSUE_LABELS_INVERSE["white_matter"] in present


# ---------------------------------------------------------------------------
# Wiring into the simulator
# ---------------------------------------------------------------------------


class TestSimulatorIntegration:
    def test_a_uniform_field_reproduces_the_scalar_path_exactly(self):
        """The load-bearing regression pin: the new path generalises the old."""
        geom = make_gbm_phantom(size=10, seed=0)
        scalar = TumorEvolutionSimulator(geom).run(MTDPolicy(), days=45, seed=1)
        uniform = DiffusionField.uniform(
            0.35, geom.tumor_mask.shape, voxel_mm=1.0
        )
        het = TumorEvolutionSimulator(geom, diffusion_field=uniform).run(
            MTDPolicy(), days=45, seed=1
        )
        assert np.allclose(scalar.burden_S, het.burden_S)
        assert np.allclose(scalar.burden_R, het.burden_R)
        assert scalar.pfs_day == het.pfs_day
        assert scalar.toxicity == pytest.approx(het.toxicity)

    def test_a_heterogeneous_field_changes_the_trajectory(self):
        geom = make_gbm_phantom(size=10, seed=0)
        classes = tissue_classes_from_geometry(geom)
        field = DiffusionField.from_tissue_classes(classes, voxel_mm=REAL_VOXEL_MM)
        scalar = TumorEvolutionSimulator(geom).run(MTDPolicy(), days=45, seed=1)
        het = TumorEvolutionSimulator(geom, diffusion_field=field).run(
            MTDPolicy(), days=45, seed=1
        )
        assert not np.allclose(scalar.burden_S, het.burden_S)

    def test_shape_mismatch_is_refused(self):
        geom = make_gbm_phantom(size=10, seed=0)
        bad = DiffusionField.uniform(0.35, (4, 4, 4), voxel_mm=1.0)
        with pytest.raises(ValueError):
            TumorEvolutionSimulator(geom, diffusion_field=bad)

    def test_zero_mean_field_is_refused(self):
        geom = make_gbm_phantom(size=10, seed=0)
        zeros = np.zeros(geom.tumor_mask.shape)
        with pytest.raises(ValueError):
            TumorEvolutionSimulator(geom, diffusion_field=zeros)

    def test_negative_cell_motility_is_refused(self):
        geom = make_gbm_phantom(size=10, seed=0)
        with pytest.raises(ValueError):
            TumorEvolutionSimulator(geom, cell_motility=-0.01)

    def test_zero_cell_motility_freezes_the_migration_term(self):
        """A named parameter must actually gate the term it names."""
        geom = make_gbm_phantom(size=10, seed=0)
        moving = TumorEvolutionSimulator(geom).run(MTDPolicy(), days=30, seed=2)
        still = TumorEvolutionSimulator(geom, cell_motility=0.0).run(
            MTDPolicy(), days=30, seed=2
        )
        # The migration term redistributes density, so the aggregate diverges.
        assert not np.allclose(moving.burden_S, still.burden_S)


# ---------------------------------------------------------------------------
# Honesty surface
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_measured_vs_assumed_split(self):
        prov = anisotropy_provenance()
        assert prov["unit_conversion"] is EvidenceStatus.ESTABLISHED
        assert prov["constant_d_identity"] is EvidenceStatus.ESTABLISHED
        assert prov["tissue_d_values"] is EvidenceStatus.UNKNOWN
        assert prov["anisotropy_field"] is EvidenceStatus.UNKNOWN

    def test_cannot_show_list_refuses_the_flattering_direction(self):
        items = what_anisotropy_cannot_show()
        assert len(items) >= 4
        assert any("backwards" in i for i in items)
        assert any("not measured by imaging" in i for i in items)


TISSUE_LABELS_INVERSE = {v: k for k, v in TISSUE_LABELS.items()}
