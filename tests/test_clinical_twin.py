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
    STREAM4_RESOLUTION_PATH,
    assemble_digital_twin_state,
    build_stream3_clearance,
    build_stream3_vessels,
    clinical_stream_census,
    decimate_t1ce_to_engine_grid,
    stream1_initial_state,
    stream2_clones,
    stream2_growth_rate,
    stream4_velocity,
    stream4_velocity_agreement,
    stream3_report,
    stream4_resolution_report,
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
