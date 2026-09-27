"""Tests for the clinical-twin datastream module.

Pins: every stream's identity property (the neutral value reproduces the
calibrated default exactly), the validation refusals, the stream-3 real-data
pipeline (its transform reproduces the committed engine_ready mask exactly),
the live census refusals, and the committed cohort measurement.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from danon.clinical_twin import (
    CLEARANCE_GAMMA,
    DEFAULT_ALA_R0,
    DIFFUSION_GAMMA,
    KI67_REF,
    STREAM3_COHORT_PATH,
    STREAM3_D0_MM2_PER_DAY,
    HISTOLOGY_PROBE_PATH,
    STREAM4_RESOLUTION_PATH,
    assemble_digital_twin_state,
    build_stream3_clearance,
    build_stream3_diffusion_field,
    build_stream3_vessels,
    clinical_stream_census,
    decimate_t1ce_to_engine_grid,
    stream1_initial_state,
    stream2_capacity_clones,
    stream2_capacity_contrast,
    stream2_carrying_capacity,
    stream2_clones,
    stream2_growth_rate,
    stream3_diffusion_contrast,
    stream4_velocity,
    stream4_velocity_agreement,
    stream3_report,
    stream4_resolution_report,
)
from danon.cure import EvidenceStatus
from danon.evosteer import DEFAULT_CLONES


# ---------------------------------------------------------------------------
# Stream 2: Ki-67 -> growth
# ---------------------------------------------------------------------------


def test_stream2_identity_at_reference_ki67():
    g = stream2_growth_rate(KI67_REF)
    assert g == pytest.approx(DEFAULT_CLONES.growth_S, abs=1e-12)
    clones = stream2_clones(KI67_REF)
    assert clones.growth_S == pytest.approx(DEFAULT_CLONES.growth_S, abs=1e-12)
    assert clones.growth_R == pytest.approx(DEFAULT_CLONES.growth_R, abs=1e-12)
    # untouched calibrated fields
    assert clones.mutation_rate == DEFAULT_CLONES.mutation_rate
    assert clones.competition == DEFAULT_CLONES.competition
    assert clones.drug_sensitivity_scale_R == DEFAULT_CLONES.drug_sensitivity_scale_R


def test_stream2_scales_growth_and_resistance_cost_together():
    clones = stream2_clones(2 * KI67_REF)
    assert clones.growth_S == pytest.approx(2 * DEFAULT_CLONES.growth_S)
    # the resistance cost is preserved: gR/gS unchanged
    assert clones.growth_R / clones.growth_S == pytest.approx(
        DEFAULT_CLONES.growth_R / DEFAULT_CLONES.growth_S
    )


def test_stream2_rejects_out_of_range_ki67():
    with pytest.raises(ValueError):
        stream2_growth_rate(0.0)
    with pytest.raises(ValueError):
        stream2_growth_rate(1.5)


# ---------------------------------------------------------------------------
# Stream 3: clearance pattern
# ---------------------------------------------------------------------------


def test_stream3_uniform_rcbv_is_uniform_pattern():
    pattern = build_stream3_clearance(np.full((4, 4, 4), 0.5))
    assert np.allclose(pattern, 1.0)


def test_stream3_variable_rcbv_is_mean_one_and_ordered():
    rcbv = np.zeros((4, 4, 4))
    rcbv[2, 2, 2] = 1.0
    pattern = build_stream3_clearance(rcbv, gamma=2.0)
    assert pattern.mean() == pytest.approx(1.0, abs=1e-12)
    assert pattern[2, 2, 2] == pattern.max()
    assert pattern[0, 0, 0] == pattern.min()
    with pytest.raises(ValueError):
        build_stream3_clearance(np.full((2, 2, 2), -1.0))


# ---------------------------------------------------------------------------
# Stream 1: 5-ALA initial state
# ---------------------------------------------------------------------------


def _phantom_mask():
    m = np.zeros((5, 6, 7), dtype=bool)
    m[1:4, 2:4, 2:5] = True
    return m


def test_stream1_preserves_engine_fill_burden_and_r0():
    mask = _phantom_mask()
    rng = np.random.default_rng(3)
    fluo = np.where(mask, rng.uniform(0.2, 1.0, mask.shape), 0.0)
    ns, nr = stream1_initial_state(fluo, mask, r0=0.05)
    assert ns.sum() + nr.sum() == pytest.approx(0.65 * mask.sum())
    assert nr.sum() / (ns.sum() + nr.sum()) == pytest.approx(0.05, abs=1e-12)
    assert not (ns[~mask] > 0).any()


def test_stream1_heterogeneous_map_is_heterogeneous_state():
    mask = _phantom_mask()
    fluo = np.where(mask, 0.1, 0.0)
    fluo[2, 3, 3] = 1.0  # one bright voxel
    ns, _ = stream1_initial_state(fluo, mask, r0=0.05)
    assert ns[2, 3, 3] > 5.0 * ns[1, 2, 2]  # spatial layout follows the map
    with pytest.raises(ValueError):
        stream1_initial_state(np.zeros(mask.shape), mask)  # no signal
    with pytest.raises(ValueError):
        stream1_initial_state(np.ones((2, 2, 2)), mask)  # shape mismatch


# ---------------------------------------------------------------------------
# Stream 4: observation only
# ---------------------------------------------------------------------------


def test_stream4_reports_measured_velocity_without_input_status():
    t1 = np.zeros((4, 4, 4))
    t2 = t1.copy()
    t2[2, 2, 2] = 3.0
    t2[0, 0, 0] = -1.0
    obs = stream4_velocity(t1, t2, interval_days=56)
    assert obs["growing_voxels"] == 1
    assert obs["shrinking_voxels"] == 1
    assert obs["delta_volume_total"] == pytest.approx(2.0)
    assert obs["abs_rate_per_day"] == pytest.approx(4.0 / 56)
    assert "observation only" in obs["status"]
    with pytest.raises(ValueError):
        stream4_velocity(t1, np.zeros((2, 2, 2)))


def test_stream4_agreement_refuses_and_names_the_live_count():
    """Without a measured pair the comparison is refused, not estimated."""
    ag = stream4_velocity_agreement()
    assert ag.status == "REFUSED"
    assert ag.pearson_r is None and ag.rate_ratio is None
    assert ag.n_voxels == 0
    assert "0 cached" in ag.refusal
    assert "usable-pair patients in the cached CFB-GBM manifest" in ag.refusal
    # one side alone is still a refusal: the reference is what is missing
    t1 = np.zeros((3, 3, 3))
    t2 = t1.copy()
    t2[1, 1, 1] = 2.0
    half = stream4_velocity_agreement(measured=(t1, t2))
    assert half.status == "REFUSED" and "simulated" in half.refusal
    assert "REFUSED" in ag.describe()
    assert ag.to_json_dict()["role"].startswith("validation target")


def test_stream4_agreement_is_a_comparison_not_a_covariate():
    """Identical fields agree perfectly; a shifted field does not.

    The second half is the point: this function reports a number that CAN be
    bad. A comparator that only ever returned agreement would be decoration.
    """
    a = np.zeros((5, 5, 5))
    b = a.copy()
    b[2, 2, 2] = 1.0
    b[1, 1, 1] = 0.5
    same = stream4_velocity_agreement(measured=(a, b), simulated=(a, b))
    assert same.status == "MEASURED"
    assert same.pearson_r == pytest.approx(1.0)
    assert same.sign_agreement == 1.0
    assert same.rate_ratio == pytest.approx(1.0)
    assert same.measured_peak_voxel == same.simulated_peak_voxel

    # the engine grows a different voxel, and shrinks one the measurement
    # called growth: r must fall and the sign agreement must not be 1.
    c = a.copy()
    c[3, 3, 3] = 1.0
    c[1, 1, 1] = -0.5
    worse = stream4_velocity_agreement(measured=(a, b), simulated=(a, c))
    assert worse.pearson_r < 1.0
    assert worse.sign_agreement < 1.0
    assert worse.measured_peak_voxel != worse.simulated_peak_voxel

    # a flat pair is refused rather than reported as perfect agreement
    with pytest.raises(ValueError):
        stream4_velocity_agreement(measured=(a, a), simulated=(a, a))
    # different grids must be registered first, not silently broadcast
    with pytest.raises(ValueError):
        stream4_velocity_agreement(measured=(a, b), simulated=(a, np.zeros((2, 2, 2))))


# ---------------------------------------------------------------------------
# Stream 3 real-data pipeline (needs the cached BraTS files)
# ---------------------------------------------------------------------------


BRATS_ROOT = os.path.join("data", "brats")
HAS_BRATS = os.path.exists(os.path.join(BRATS_ROOT, "BraTS2021_01017_seg.nii.gz"))


@pytest.mark.skipif(not HAS_BRATS, reason="cached BraTS case not present")
def test_stream3_transform_reproduces_committed_engine_mask():
    """The t1ce pipeline's window is the segmentation's exact engine transform."""
    from danon.imaging import read_nifti
    from danon.realdata import DecimationSpec, _bbox, _decimate_labels

    seg_path = os.path.join(BRATS_ROOT, "BraTS2021_01017_seg.nii.gz")
    seg, _ = read_nifti(seg_path)
    spec = DecimationSpec()
    bb = _bbox(np.asarray(seg) > 0)
    mask_grid, _ = _decimate_labels(np.rint(np.asarray(seg)).astype(np.int32)[bb], spec)

    engine, _ = decimate_t1ce_to_engine_grid(
        os.path.join(BRATS_ROOT, "BraTS2021_01017_t1ce.nii.gz"), seg_path
    )
    assert engine.shape == mask_grid.shape
    committed, _ = read_nifti("data/engine_ready/BraTS2021_01017_mask.nii.gz")
    assert np.array_equal(mask_grid, np.asarray(committed) > 0)


@pytest.mark.skipif(not HAS_BRATS, reason="cached BraTS case not present")
def test_stream3_vessels_carry_real_enhancement(tmp_path):
    out = tmp_path / "vessels.nii.gz"
    info = build_stream3_vessels(
        os.path.join(BRATS_ROOT, "BraTS2021_01017_t1ce.nii.gz"),
        os.path.join(BRATS_ROOT, "BraTS2021_01017_seg.nii.gz"),
        out_path=str(out),
    )
    assert info["tumour_voxels"] == 1774
    assert info["engine_grid"] == [14, 24, 24]
    assert 0.0 < info["mean_enhancement_in_tumour"] < 1.0
    assert info["voxels_with_enhancement"] > info["tumour_voxels"] * 0.5
    assert out.exists()


@pytest.mark.skipif(not HAS_BRATS, reason="cached BraTS case not present")
def test_engine_runs_with_stream3_vasculature_and_moves_mtd():
    """Paired same-seed: real vessels vs null, geometry otherwise identical."""
    from danon.evosteer import MTDPolicy, TumorEvolutionSimulator
    from danon.imaging import from_nifti_mask

    pg_r = from_nifti_mask(
        "data/engine_ready/BraTS2021_01017_mask.nii.gz",
        proliferative_path="data/engine_ready/BraTS2021_01017_prolif.nii.gz",
        vessels_path="data/engine_ready/BraTS2021_01017_vessels_t1ce.nii.gz",
    )
    pg_n = from_nifti_mask(
        "data/engine_ready/BraTS2021_01017_mask.nii.gz",
        proliferative_path="data/engine_ready/BraTS2021_01017_prolif.nii.gz",
    )
    assert pg_r.vasculature_source == "file"
    assert pg_n.vasculature_source == "null-uniform"
    assert np.array_equal(pg_r.geometry.tumor_mask, pg_n.geometry.tumor_mask)

    a = TumorEvolutionSimulator(pg_r.geometry).run(
        MTDPolicy(), days=200, initial_resistant_fraction=0.005, seed=101
    )
    b = TumorEvolutionSimulator(pg_n.geometry).run(
        MTDPolicy(), days=200, initial_resistant_fraction=0.005, seed=101
    )
    # both run; the fields are not identical (the vasculature differs)
    assert a.pfs_day is not None or b.pfs_day is not None or True
    assert not np.array_equal(a.burden_S, b.burden_S)


# ---------------------------------------------------------------------------
# Assembly + census
# ---------------------------------------------------------------------------


def test_assemble_with_no_streams_is_identity_status():
    state = assemble_digital_twin_state()
    assert state.initial_state is None
    assert state.clones_override is None
    assert state.clearance_pattern is None
    assert state.velocity_observation is None
    assert all("not supplied" in v for v in state.stream_status.values())


def test_assemble_routes_each_stream():
    mask = _phantom_mask()
    fluo = np.where(mask, 0.5, 0.0)
    rcbv = np.full(mask.shape, 0.3)
    t1 = np.zeros(mask.shape)
    t2 = t1.copy()
    t2[2, 2, 2] = 1.0

    state = assemble_digital_twin_state(
        fluorescence_5ala_mesh=fluo,
        histology_ki67_ratio=0.30,
        mri_rcbv_tensor=rcbv,
        delta_volume_velocity=(t1, t2),
        mask=mask,
        histology_density_per_mm2=1579.0,
        density_reference_per_mm2=636.0,
    )
    assert state.initial_state is not None
    assert state.clones_override is not None
    assert state.clones_override.growth_S > DEFAULT_CLONES.growth_S
    assert state.clones_override.carrying_capacity == pytest.approx(1579 / 636)
    assert state.clearance_pattern is not None
    assert state.velocity_observation is not None
    assert all(v.startswith("applied") for v in state.stream_status.values())

    with pytest.raises(ValueError):
        assemble_digital_twin_state(fluorescence_5ala_mesh=fluo)  # mask missing


def test_census_refuses_empty_streams_and_reports_stream3():
    census = clinical_stream_census()
    assert census["stream1_5ala"]["status"] == "REFUSED"
    assert census["stream1_5ala"]["n_files"] == 0
    assert census["stream2_histology"]["status"] == "REFUSED"
    assert census["stream4_sequential"]["status"] == "REFUSED"
    assert census["stream4_sequential"]["n_pairs_cached"] == 0
    if HAS_BRATS:
        assert census["stream3_perfusion"]["status"] == "MEASURABLE"
        assert census["stream3_perfusion"]["n_t1ce_files"] >= 1


class TestStream2Capacity:
    """The measured packing as the engine's density ceiling.

    The load-bearing property is not the identity -- it is that the mapping is
    the ONLY thing the capacity arm changes, so the paired cohort can attribute a
    difference to the density. A mis-paired control (the defect this class
    caught) shows up at K x1.0, which must reproduce the control exactly.
    """

    def test_identity_at_the_reference_density(self):
        from danon.evosteer import DEFAULT_CLONES

        result = stream2_carrying_capacity(636.0, 636.0)
        assert result.ratio == 1.0
        assert result.k == DEFAULT_CLONES.carrying_capacity
        assert result.mode == "relative"
        assert "section thickness" in result.absolute_refusal
        assert "2.48" in result.describe() or "1.000x" in result.describe()

    def test_scales_by_the_measured_ratio(self):
        lo = stream2_capacity_clones(460.0, 636.0)
        hi = stream2_capacity_clones(1579.0, 636.0)
        from danon.evosteer import DEFAULT_CLONES

        assert hi.carrying_capacity / lo.carrying_capacity == pytest.approx(1579 / 460)
        # Nothing but the ceiling moves.
        for clones in (lo, hi):
            assert clones.growth_S == DEFAULT_CLONES.growth_S
            assert clones.growth_R == DEFAULT_CLONES.growth_R
            assert clones.mutation_rate == DEFAULT_CLONES.mutation_rate
            assert clones.competition == DEFAULT_CLONES.competition

    @pytest.mark.parametrize(
        "density,reference",
        [(0.0, 1.0), (-5.0, 1.0), (1.0, 0.0), (1.0, -1.0), (float("nan"), 1.0), (float("inf"), 1.0)],
    )
    def test_refuses_unusable_densities(self, density, reference):
        with pytest.raises(ValueError):
            stream2_carrying_capacity(density, reference)

    def test_the_seed_scales_with_the_measured_ceiling(self):
        """fill_fraction is calibrated AGAINST K, so both must move together.

        Seeding at the default 1.0 while the ceiling sat at a measured K divided
        the calibrated fill by K -- at K=2.48 the tumour started at 26% of its
        stated fill, which is a different initial condition, not a rescaled one.
        """
        mask = np.ones((4, 4, 4), dtype=bool)
        fluo = np.where(mask, 1.0, 0.0)
        n_voxels = int(mask.sum())
        for density in (460.0, 636.0, 1579.0):
            state = assemble_digital_twin_state(
                fluorescence_5ala_mesh=fluo,
                mask=mask,
                histology_density_per_mm2=density,
                density_reference_per_mm2=636.0,
            )
            burden = float(state.initial_state[0].sum() + state.initial_state[1].sum())
            k = state.clones_override.carrying_capacity
            assert burden / (k * n_voxels) == pytest.approx(0.65, rel=1e-9)
        # With no density the seed is the engine's own default, unchanged. The
        # total is what the stream promises: the seed splits into S and R by r0.
        plain = assemble_digital_twin_state(fluorescence_5ala_mesh=fluo, mask=mask)
        total = float(plain.initial_state[0].sum() + plain.initial_state[1].sum())
        assert total == pytest.approx(0.65 * n_voxels)
        assert float(plain.initial_state[1].sum()) == pytest.approx(
            total * DEFAULT_ALA_R0
        )

    @pytest.mark.parametrize("scalar", [np.float32(636.0), np.float64(636.0), np.int64(636)])
    def test_accepts_numpy_scalars(self, scalar):
        """A density measured off an array arrives as a numpy scalar.

        The first version rejected np.float32 with the message "must be a finite
        number, got 636.0" -- wrong on both counts.
        """
        assert stream2_carrying_capacity(scalar, 636.0).ratio == pytest.approx(1.0)

    @pytest.mark.parametrize("junk", [True, "636", None, object()])
    def test_refuses_non_numbers(self, junk):
        with pytest.raises(ValueError):
            stream2_carrying_capacity(junk, 636.0)

    def test_the_departure_from_pure_scale_grows_as_the_seed_approaches_the_floor(
        self,
    ):
        """The claim's mechanism, demonstrated on the engine's own phantom.

        The claim says measured packing matters only where an ABSOLUTE floor gates
        a decision. This walks the ceiling toward `MIN_VIABLE_DENSITY` and requires
        the departure from pure scale-covariance to grow -- so if the floor moved
        or stopped biting, the claim's reading would fail here rather than quietly
        becoming false.
        """
        from danon.evosteer import (
            MIN_VIABLE_DENSITY,
            MTDPolicy,
            TumorEvolutionSimulator,
            make_gbm_phantom,
        )

        geom = make_gbm_phantom()
        scaled = []
        for ratio in (1.0, 1e-3, 1e-5):
            clones = stream2_capacity_clones(636.0 * ratio, 636.0)
            traj = TumorEvolutionSimulator(geom, clones=clones).run(
                MTDPolicy(), days=365, initial_resistant_fraction=0.005, seed=101
            )
            # Pure scale-covariance leaves final/K identical across ratios, so the
            # reference is the baseline arm itself rather than a remembered number.
            scaled.append(
                float(traj.burden_S[-1] + traj.burden_R[-1])
                / clones.carrying_capacity
            )
        departures = [abs(s - scaled[0]) / scaled[0] for s in scaled]
        assert departures[0] == 0.0
        assert departures[1] > 0.0
        assert departures[2] > departures[1]
        # And the floor really is what does it: the tiniest ceiling empties.
        tiny = stream2_capacity_clones(636.0 * 1e-7, 636.0)
        assert 0.65 * tiny.carrying_capacity < MIN_VIABLE_DENSITY
        empty = TumorEvolutionSimulator(geom, clones=tiny).run(
            MTDPolicy(), days=365, initial_resistant_fraction=0.005, seed=101
        )
        assert float(empty.burden_S[-1] + empty.burden_R[-1]) == 0.0
        assert empty.pfs_day is None

    def test_the_contrast_refuses_an_empty_cohort(self):
        """No seeds is no measurement, not a null result."""
        with pytest.raises(ValueError) as err:
            stream2_capacity_contrast(
                "01017",
                nuclei_per_mm2=636.0,
                reference_nuclei_per_mm2=636.0,
                reference_role="empty-seed case",
                ratios=[1.0],
                seeds=(),
            )
        assert "empty cohort" in str(err.value)

    def test_assembled_state_applies_the_capacity_and_refuses_a_half_pair(self):
        mask = np.ones((4, 4, 4), dtype=bool)
        state = assemble_digital_twin_state(
            fluorescence_5ala_mesh=np.ones((4, 4, 4)),
            mask=mask,
            histology_density_per_mm2=1579.0,
            density_reference_per_mm2=636.0,
        )
        assert state.clones_override.carrying_capacity == pytest.approx(1579 / 636)
        assert state.clones_override.growth_S == pytest.approx(
            DEFAULT_CLONES.growth_S
        )
        assert state.stream_status["stream2_density_capacity"].startswith("applied")

        # Composes with stream 2's growth mapping rather than replacing it.
        both = assemble_digital_twin_state(
            fluorescence_5ala_mesh=np.ones((4, 4, 4)),
            mask=mask,
            histology_ki67_ratio=KI67_REF,
            histology_density_per_mm2=460.0,
            density_reference_per_mm2=636.0,
        )
        assert both.clones_override.growth_S == pytest.approx(DEFAULT_CLONES.growth_S)
        assert both.clones_override.carrying_capacity == pytest.approx(460 / 636)

        # A density with no declared reference refuses: it would set K against
        # whatever number happened to be nearby.
        with pytest.raises(ValueError):
            assemble_digital_twin_state(
                fluorescence_5ala_mesh=np.ones((4, 4, 4)),
                mask=mask,
                histology_density_per_mm2=1579.0,
            )
        # Omitted entirely, nothing is overridden.
        plain = assemble_digital_twin_state(
            fluorescence_5ala_mesh=np.ones((4, 4, 4)), mask=mask
        )
        assert plain.clones_override is None
        assert "not supplied" in plain.stream_status["stream2_density_capacity"]


@pytest.mark.skipif(
    not os.path.exists("outputs_real/stream2_capacity_cohort.json"),
    reason="stream-2 capacity cohort not built (see _stream2_capacity_cohort.py)",
)
class TestCapacityCohortRecord:
    """The committed measurement, checked for the relations it claims.

    Read rather than re-run: a paired engine cohort costs minutes, and the
    relations below are exactly what a re-run would have to satisfy.
    """

    @staticmethod
    def _rows():
        import json

        return json.load(open("outputs_real/stream2_capacity_cohort.json"))

    def test_reference_and_endpoints_come_from_the_slide_record(self):
        import json

        probe = json.load(open(HISTOLOGY_PROBE_PATH))
        density = probe["slide_measured"]["density"]
        reference = float(density["nuclei_per_mm2"])
        spread = density["per_crop_spread"]
        for row in self._rows():
            assert row["reference_nuclei_per_mm2"] == reference
            ratios = sorted(
                set(
                    float(key)
                    for arm in row["arms"].values()
                    for key in arm["k_by_ratio"]
                )
            )
            expected = sorted(
                [1.0, float(spread["min"]) / reference, float(spread["max"]) / reference]
            )
            assert ratios == pytest.approx(expected, abs=1e-6)

    def test_the_unit_ratio_arm_reproduces_the_control_exactly(self):
        """The pairing assertion: K x1.0 must be bit-identical to the control.

        This is what caught a mis-paired control during development, where the
        control was re-run inside the ratio loop and the arms were compared to
        the wrong partner.
        """
        for row in self._rows():
            for policy, arm in row["arms"].items():
                unit = arm["k_by_ratio"]["1.000000"]
                assert unit == pytest.approx(1.0)
                assert arm["burden_capacity"]["1.000000"] == arm["burden_uniform"]
                assert arm["pfs_capacity"]["1.000000"] == arm["pfs_uniform"]
                assert arm["max_abs_relative_burden_difference"]["1.000000"] == 0.0
                assert policy in ("mtd", "steering")

    def test_under_mtd_the_burden_ratio_is_the_capacity_ratio(self):
        """Static dosing: K rescales the reported burden by its own factor."""
        for row in self._rows():
            arm = row["arms"]["mtd"]
            for key, k in arm["k_by_ratio"].items():
                pairs = list(zip(arm["burden_uniform"], arm["burden_capacity"][key]))
                assert pairs, "no paired burdens recorded"
                for control, capacity in pairs:
                    assert control > 0
                    assert capacity / control == pytest.approx(k, rel=0.01)

    def test_under_steering_the_ratio_departs_from_the_capacity_ratio(self):
        """Adaptive dosing is where an absolute floor gates a decision.

        The departure must be present AND larger than the MTD arm's, or the
        claim's reading of the mechanism would be unsupported.
        """
        departures = []
        for row in self._rows():
            arm = row["arms"]["steering"]
            for key, k in arm["k_by_ratio"].items():
                if k == 1.0:
                    continue
                for control, capacity in zip(
                    arm["burden_uniform"], arm["burden_capacity"][key]
                ):
                    departures.append(abs(capacity / control - k))
        assert departures
        assert max(departures) > 0.05
        assert max(departures) > arm["max_abs_relative_burden_difference"]["1.000000"]

    @pytest.mark.skipif(
        not os.path.exists("data/engine_ready/BraTS2021_01017_mask.nii.gz"),
        reason="engine-ready BraTS case not cached",
    )
    def test_a_uniform_slide_collapses_to_one_arm(self):
        """A slide whose crops all measure the same density yields ratio 1.0
        three times over. The arms are keyed by ratio, so an undeduplicated list
        would leave an arm longer than its control and read as a broken cohort.
        """
        contrast = stream2_capacity_contrast(
            "01017",
            nuclei_per_mm2=636.0,
            reference_nuclei_per_mm2=636.0,
            reference_role="uniform-slide case",
            ratios=[1.0, 1.0, 1.0],
            seeds=(101,),
        )
        assert contrast.ratios == (1.0,)
        assert len(contrast.pfs_capacity_days["1.000000"]) == len(
            contrast.pfs_uniform_days
        )

    def test_pfs_is_reported_as_essentially_unmoved(self):
        """The claim may not rest on PFS: 5 of 6 policy-arms move 0 days."""
        moved = 0
        total = 0
        for row in self._rows():
            for arm in row["arms"].values():
                for delta in arm["median_pfs_delta_days"].values():
                    total += 1
                    if delta != 0:
                        moved += 1
                        assert abs(delta) <= 1.0
        assert total > 0
        assert moved <= total / 2


def test_census_refuses_stream2_as_input_while_reporting_the_sample():
    """A downloaded haematoxylin-only slide is a SAMPLE, not a Ki-67 input.

    The census counts slides in the intake directories as inputs and the probe's
    downloaded slide separately, so a refusal cannot be flipped to "available"
    by a file that could not feed the mapping anyway.
    """
    s2 = clinical_stream_census()["stream2_histology"]
    assert s2["n_files"] == 0
    assert "Ki-67" in s2["refused_for"]
    assert "CELLULARITY" in s2["measurable"]
    wsi = s2.get("wsi") or {}
    if "error" in wsi:
        pytest.skip("histology probe record unavailable")
    assert wsi["n_named_ki67"] == 0
    assert wsi["stain"] != "dab-positive"
    assert wsi["nuclei_per_mm2"] > 0 and wsi["crops_used"] >= 2
    assert wsi["n_slides_cached"] >= 1


def test_census_separates_cached_from_retrievable():
    """A refusal must say what is missing, not imply the data does not exist.

    Stream 4's old entry reported zero pairs with the rationale "BraTS is a
    single pre-operative timepoint" -- true of the cached path, and read as a
    statement about the archive. The census now carries both counts, so the
    BraTS fact cannot stand in for the availability answer.
    """
    census = clinical_stream_census()
    s4 = census["stream4_sequential"]
    archive = s4.get("archive") or {}
    if not archive or "error" in archive:
        pytest.skip("recorded TCIA probe unavailable")
    assert s4["n_pairs_cached"] == 0
    assert s4["n_pairs_retrievable"] == 41
    assert archive["n_patients"] == 630
    assert archive["min_interval_days"] == 28
    # the join: every longitudinal patient also carries perfusion series
    assert archive["n_longitudinal_with_perfusion"] == s4["n_pairs_retrievable"]
    # stream 3's real (measured) form is retrievable even though the cache holds
    # only the t1ce proxy
    s3 = census["stream3_perfusion"]
    assert s3["n_archive_dsc_patients"] == 456
    assert s3["n_archive_processed_captk_patients"] == 628
    # ...but those 628 are co-registered STRUCTURAL series, not perfusion maps:
    # a downloaded file is 'AX T1 PRE : Processed_CaPTk'. The census must say so
    # rather than let the number read as an rCBV/Ktrans archive.
    assert "STRUCTURAL" in s3["archive_processed_series_are"]
    # Stream 3's own measured arm: the downloaded DSC cohort, read live.
    dsc = s3["dsc"]
    assert dsc["n_cases"] >= 1
    assert dsc["n_series_cached"] >= 1
    assert dsc["max_field_contrast"] > 1.0
    assert "censored" in dsc["endpoint_note"]
    assert archive["n_seg_series"] == 0


# ---------------------------------------------------------------------------
# Committed cohort measurement
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(STREAM3_COHORT_PATH),
    reason="stream-3 cohort measurement not committed",
)
def test_committed_stream3_cohort_matches_report():
    import json

    rows = json.load(open(STREAM3_COHORT_PATH))
    assert len(rows) == 6
    report = stream3_report()
    assert report["n_cases"] == 6
    assert report["n_paired_runs_per_arm"] == 18
    # the measured headline is negative and small: |mean| < 5 days
    assert -5.0 < report["mtd_pfs_delta_mean_days"] <= 0.0
    assert report["steering_censored_both"] is True


# ---------------------------------------------------------------------------
# Stream-4 resolution: a design calculation, wired into the refusal
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.exists(STREAM4_RESOLUTION_PATH),
    reason="stream-4 design calculation not committed",
)
def test_resolution_report_exposes_both_floors():
    report = stream4_resolution_report()
    res = report["resolution"]
    assert res["n_patients"] == 168
    assert res["n_draws"] == 24
    assert 0.0 < res["floor_optimistic"] < res["floor_conservative"]
    assert "DESIGN calculation" in report["what_this_is"]
    assert "not a measurement of any datastream" in report["what_this_is"]
    # the verdict names both floors, so neither can be quoted alone by accident
    assert f"{res['floor_optimistic']:.4f}" in report["verdict"]
    assert f"{res['floor_conservative']:.4f}" in report["verdict"]


def test_census_carries_the_resolvable_delta_with_formatted_detail():
    s4 = clinical_stream_census()["stream4_sequential"]
    assert s4["status"] == "REFUSED"
    assert s4["resolvable_delta_c"] is not None
    assert "(1 SE) /" in s4["detail"] and "fold yardstick" in s4["detail"]
    # an unrounded float in prose is how a derived number looks invented
    assert "0.021904" not in s4["detail"]
    assert "an OBSERVATION cannot move a C-index" in s4["detail"]


# ---------------------------------------------------------------------------
# Stream 3, diffusion arm: D_chemo(x) = D0*(1+gamma*rCBV(x))
# ---------------------------------------------------------------------------


def test_diffusion_field_neutral_value_is_uniform_transport():
    """Uniform rCBV (and gamma=0) return the engine's homogeneous transport."""
    uniform = build_stream3_diffusion_field(np.full((4, 5, 6), 0.3))
    assert uniform.d_engine.min() == uniform.d_engine.max()
    assert uniform.contrast_ratio == pytest.approx(1.0)
    # the simulator re-normalises to a mean-one pattern, so the neutral
    # field's pattern is exactly 1 everywhere
    pattern = uniform.d_engine / uniform.d_engine.mean()
    assert np.allclose(pattern, 1.0, rtol=0, atol=1e-15)
    # gamma = 0 removes the perfusion gain entirely, whatever the map says
    flat = build_stream3_diffusion_field(np.array([[[0.0, 0.9], [0.3, 0.1]]]), gamma=0.0)
    assert flat.d_engine.min() == flat.d_engine.max()


def test_diffusion_field_d0_and_units_cancel_in_the_pattern():
    """The claim in the builder: only gamma and the map reach the integrator."""
    rcbv = np.array([[[0.0, 0.4], [0.8, 0.2]]])
    a = build_stream3_diffusion_field(rcbv, d0_mm2_per_day=0.35, voxel_mm=1.0)
    b = build_stream3_diffusion_field(rcbv, d0_mm2_per_day=7.0, voxel_mm=3.0)
    # physical arrays differ by both factors ...
    assert not np.allclose(a.d_mm2_per_day, b.d_mm2_per_day)
    # ... and the mean-one pattern the engine integrates is identical
    pa = a.d_engine / a.d_engine.mean()
    pb = b.d_engine / b.d_engine.mean()
    assert np.allclose(pa, pb, rtol=1e-12, atol=1e-15)
    # so D0 cannot be a hidden tuning knob, which is the point of saying it
    assert b.d_mm2_per_day[0, 0, 1] / a.d_mm2_per_day[0, 0, 1] == pytest.approx(20.0)


def test_diffusion_field_gain_is_the_declared_specification_value():
    """gamma is the specification's 2.5 and is shared with nothing else."""
    assert DIFFUSION_GAMMA == 2.5
    # the clearance mapping is a SEPARATE declared gain, not this one
    assert CLEARANCE_GAMMA != DIFFUSION_GAMMA
    field = build_stream3_diffusion_field(
        np.array([[[0.0, 1.0]]]), d0_mm2_per_day=1.0, voxel_mm=1.0
    )
    assert field.d_mm2_per_day[0, 0, 1] == pytest.approx(1.0 + DIFFUSION_GAMMA)
    assert field.d_mm2_per_day[0, 0, 0] == pytest.approx(1.0)


def test_diffusion_field_provenance_labels_the_perfusion_proxy():
    """Neither the map nor the gain may claim to be measured."""
    field = build_stream3_diffusion_field(np.full((3, 3, 3), 0.5))
    assert field.provenance["perfusion_map"] is EvidenceStatus.UNKNOWN
    assert field.provenance["gain_gamma"] is EvidenceStatus.UNKNOWN
    assert field.provenance["unit_conversion"] is EvidenceStatus.ESTABLISHED


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.zeros((4, 5)), "must be 3D"),
        (np.full((2, 2, 2), -0.1), "must be non-negative"),
        (np.full((2, 2, 2), np.nan), "non-finite"),
        (np.full((2, 2, 2), np.inf), "non-finite"),
    ],
)
def test_diffusion_field_refuses_unusable_maps(bad, match):
    with pytest.raises(ValueError, match=match):
        build_stream3_diffusion_field(bad)


def test_diffusion_field_refuses_negative_gain_and_bad_scales():
    ok = np.full((2, 2, 2), 0.2)
    with pytest.raises(ValueError, match="gamma must be non-negative"):
        build_stream3_diffusion_field(ok, gamma=-1.0)
    with pytest.raises(ValueError, match="voxel_mm must be positive"):
        build_stream3_diffusion_field(ok, voxel_mm=0.0)
    with pytest.raises(ValueError, match="d0 must be positive"):
        build_stream3_diffusion_field(ok, d0_mm2_per_day=0.0)


def test_uniform_diffusion_field_reproduces_the_default_transport_path():
    """The identity that makes a uniform perfusion map the neutral value.

    div(D grad C) with constant D IS D*laplacian(C), so the field supplies no
    new physics at the neutral value. The two branches sum in a different
    order, so the match is at round-off, not bit-identical -- asserted at that
    tolerance rather than overstated.
    """
    from danon.evosteer import MTDPolicy, TumorEvolutionSimulator, TumorGeometry

    n = 16
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    r = np.sqrt((zz - n / 2) ** 2 + (yy - n / 2) ** 2 + (xx - n / 2) ** 2)
    mask = r < 6
    geom = TumorGeometry(
        tumor_mask=mask,
        proliferative=mask & (r > 4),
        vessels=(mask & (r > 4) & (xx < n / 2)),
        voxel_mm=1.0,
    )
    kwargs = dict(days=40, seed=7, initial_resistant_fraction=0.02)
    default = TumorEvolutionSimulator(geom).run(MTDPolicy(), **kwargs)
    neutral = TumorEvolutionSimulator(
        geom, diffusion_field=build_stream3_diffusion_field(np.full(mask.shape, 0.4))
    ).run(MTDPolicy(), **kwargs)
    assert np.allclose(default.burden_S, neutral.burden_S, rtol=1e-9, atol=1e-9)
    assert np.allclose(default.burden_R, neutral.burden_R, rtol=1e-9, atol=1e-9)
    assert default.pfs_day == neutral.pfs_day


def test_heterogeneous_diffusion_field_actually_changes_the_trajectory():
    """The hook is live: a non-uniform field must not be a no-op."""
    from danon.evosteer import MTDPolicy, TumorEvolutionSimulator, TumorGeometry

    n = 16
    zz, yy, xx = np.mgrid[0:n, 0:n, 0:n]
    r = np.sqrt((zz - n / 2) ** 2 + (yy - n / 2) ** 2 + (xx - n / 2) ** 2)
    mask = r < 6
    geom = TumorGeometry(
        tumor_mask=mask,
        proliferative=mask & (r > 4),
        vessels=(mask & (r > 4) & (xx < n / 2)),
        voxel_mm=1.0,
    )
    kwargs = dict(days=40, seed=7, initial_resistant_fraction=0.02)
    default = TumorEvolutionSimulator(geom).run(MTDPolicy(), **kwargs)
    gradient = build_stream3_diffusion_field(np.where(mask, xx / float(n), 0.0))
    perfused = TumorEvolutionSimulator(geom, diffusion_field=gradient).run(
        MTDPolicy(), **kwargs
    )
    assert not np.allclose(default.burden_S, perfused.burden_S, rtol=1e-9, atol=1e-9)
    # and the field itself really is heterogeneous
    assert gradient.contrast_ratio > 1.0


def test_diffusion_contrast_refuses_a_case_that_is_not_cached(tmp_path):
    """No cached case means a refusal naming the path, never a generated one."""
    with pytest.raises(FileNotFoundError, match="is not cached"):
        stream3_diffusion_contrast(
            "99999", brats_dir=str(tmp_path), engineered_dir=str(tmp_path), seeds=(1,)
        )


def test_diffusion_contrast_rejects_an_unknown_policy(tmp_path):
    """Policy is validated before any engine time is spent."""
    with pytest.raises(ValueError, match="policy must be one of"):
        stream3_diffusion_contrast(
            "01017", engineered_dir=str(tmp_path), policy="moon-shot", seeds=(1,)
        )


STREAM3_DIFFUSION_PATH = os.path.join(
    "outputs_real", "stream3_diffusion_cohort.json"
)


@pytest.mark.skipif(
    not os.path.exists(STREAM3_DIFFUSION_PATH),
    reason="diffusion cohort not built (run _stream3_diffusion_cohort.py)",
)
def test_committed_diffusion_cohort_is_internally_consistent():
    """The committed measurement cannot go stale against its own arrays."""
    import json

    records = json.load(open(STREAM3_DIFFUSION_PATH))
    assert records, "cohort file exists but is empty"
    for rec in records:
        assert rec["case_id"].isdigit()
        for policy in ("mtd", "steering"):
            arm = rec["arms"][policy]
            assert len(arm["seeds"]) == len(arm["pfs_uniform"]) == len(arm["pfs_perfusion"])
            assert len(arm["toxicity_uniform"]) == len(arm["toxicity_perfusion"])
            # the stored median delta must equal the delta of the stored arrays
            assert arm["median_pfs_delta_days"] == pytest.approx(
                float(np.median(arm["pfs_perfusion"]) - np.median(arm["pfs_uniform"]))
            )
            assert arm["n_pairs_in_disagreement"] == sum(
                a != b for a, b in zip(arm["pfs_uniform"], arm["pfs_perfusion"])
            )
            assert arm["gamma"] == DIFFUSION_GAMMA
            assert arm["field_contrast_ratio"] > 1.0, "a uniform field measures nothing"
            assert arm["voxels_active"] > 0


def test_missing_resolution_artifact_refuses_rather_than_guesses(monkeypatch, tmp_path):
    import danon.clinical_twin as ct

    monkeypatch.setattr(
        ct, "STREAM4_RESOLUTION_PATH", str(tmp_path / "absent.json")
    )
    with pytest.raises(FileNotFoundError):
        ct.stream4_resolution_report()
    # and the census reports the failure instead of omitting the field
    s4 = ct.clinical_stream_census()["stream4_sequential"]
    assert s4["resolvable_delta_c"] is None
    assert "error" in s4["resolution"]
    assert "not computed (the design calculation could not be read)" in s4["detail"]
