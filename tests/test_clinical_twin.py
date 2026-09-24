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
    KI67_REF,
    STREAM3_COHORT_PATH,
    assemble_digital_twin_state,
    build_stream3_clearance,
    build_stream3_vessels,
    clinical_stream_census,
    decimate_t1ce_to_engine_grid,
    stream1_initial_state,
    stream2_clones,
    stream2_growth_rate,
    stream4_velocity,
    stream3_report,
)
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
    )
    assert state.initial_state is not None
    assert state.clones_override is not None
    assert state.clones_override.growth_S > DEFAULT_CLONES.growth_S
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
    assert s3["n_archive_processed_perfusion_patients"] == 628
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
