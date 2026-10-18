"""Tests for the grid-artifact measurements: the partition, the drift, the record.

The properties pinned here are structural, so they stay true after the shipped
decimation rule is repaired: the module's own axis-aligned block vote must keep
a tumour cube that a wrong grouping loses, must be exactly the partition a brute
force loop computes, and must deflate a real tumour while the max-pool
alternative inflates it.

What the shipped rule currently produces is NOT asserted to be wrong either --
it is *measured*, compared against the axis-aligned vote voxel for voxel, and
recorded. `test_record_recomputes_its_own_verdicts` re-derives that comparison
live and requires the committed record to agree with it, so when the rule is
fixed the record must be rebuilt rather than quietly kept.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from danon.cure import EvidenceStatus
from danon.grid_artifact import (
    GRID_LEVELS,
    SHIPPED_LEVEL,
    _boundary_voxels,
    _geometry_at_level,
    _spread,
    _upsample_nearest,
    block_decimation,
    committed_derived_audit,
    decimation_geometry,
    derived_round_trip,
    grid_artifact_provenance,
    grid_resolution_cohort,
    matched_diffusion,
    physical_diffusion_mm2_per_day,
    render_grid_artifact_report,
    render_round_trip_report,
    shipped_decimation,
    what_the_grid_cannot_show,
)
from danon.imaging import read_nifti, write_nifti

ROOT = Path(__file__).resolve().parents[1]
CASE = "BraTS2021_01017"
SEG = ROOT / "data" / "brats" / f"{CASE}_seg.nii.gz"
DERIVED = ROOT / "data" / "engine_ready" / f"{CASE}_mask.nii.gz"
RECORD = ROOT / "outputs_real" / "grid_artifact.json"


# ---------------------------------------------------------------------------
# The partition itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factors", [(2, 2, 2), (3, 1, 2)])
def test_block_vote_is_exactly_a_brute_force_axis_aligned_majority(factors):
    """Every engine voxel owns one axis-aligned block, checked by hand."""
    rng = np.random.default_rng(4)
    labels = rng.choice([0, 2, 4], size=(12, 9, 8)).astype(np.int32)
    mask, labels_grid, _ = block_decimation(labels, factors)
    fx, fy, fz = factors
    nx, ny, nz = 12 // fx, 9 // fy, 8 // fz
    assert labels_grid.shape == (nx, ny, nz)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                block = labels[
                    i * fx : (i + 1) * fx, j * fy : (j + 1) * fy, k * fz : (k + 1) * fz
                ].ravel()
                tumour = int((block > 0).sum())
                expected = tumour * 2 > block.size
                assert bool(mask[i, j, k]) is expected
                if expected:
                    # The label comes from the tumour contributors only, so a
                    # background count that ties a tumour label cannot erase a
                    # majority-tumour block.
                    values, counts = np.unique(block[block > 0], return_counts=True)
                    order = np.argsort(-counts, kind="stable")
                    assert labels_grid[i, j, k] == int(values[order[0]])
                else:
                    assert labels_grid[i, j, k] == 0


def test_the_axis_aligned_vote_keeps_a_cube_the_wrong_grouping_loses():
    """Miniature of the defect, with no real data involved.

    A 2x2x2 cube of tumour at the origin is exactly one engine voxel under (2,2,2).
    The axis-aligned vote keeps it. `realdata._decimate_labels` groups the source
    volume differently and returns an EMPTY mask for this input, which is the
    smallest statement of the problem this module exists to expose: a lesion seed
    that vanishes before the simulation starts.
    """
    labels = np.zeros((4, 6, 8), dtype=np.int32)
    labels[0:2, 0:2, 0:2] = 1
    aligned, _, _ = block_decimation(labels, (2, 2, 2))
    shipped, _ = shipped_decimation(labels, (2, 2, 2))
    assert aligned[0, 0, 0]
    assert int(aligned.sum()) == 1
    # Measured, not asserted as a requirement: the shipped rule's answer.
    assert int(shipped.sum()) == 0


def test_nearest_upsampling_inverts_the_partition_voxel_for_voxel():
    rng = np.random.default_rng(7)
    labels = rng.choice([0, 1, 2, 4], size=(8, 6, 4)).astype(np.int32)
    factors = (2, 2, 2)
    mask, _, _ = block_decimation(labels, factors)
    up = _upsample_nearest(mask, factors)
    assert up.shape == labels.shape
    for i in range(2):
        for j in range(3):
            for k in range(2):
                block = up[
                    i * 2 : i * 2 + 2, j * 2 : j * 2 + 2, k * 2 : k * 2 + 2
                ]
                assert bool(block.all()) is bool(mask[i, j, k])


def test_partition_probe_answers_a_question_the_array_comparison_cannot():
    """A fully-tumour axis-aligned cube must survive as its own engine voxel.

    This isolates the GROUPING from the label tie rule: every voxel of the cube
    carries the same label, so no majority count can be tied by background. The
    probe reports where the shipped rule sends that cube, and the accounting has
    to add up to the number of cubes probed.
    """
    from danon.grid_artifact import partition_probe

    probe = partition_probe((24, 24, 24), (2, 2, 2), n_samples=48)
    assert probe["n_probed"] == 48
    assert (
        probe["n_landed_in_their_own_engine_voxel"]
        + probe["n_landed_elsewhere"]
        + probe["n_dropped_entirely"]
        == probe["n_probed"]
    )
    assert probe["is_a_block_partition"] is (
        probe["n_landed_in_their_own_engine_voxel"] == probe["n_probed"]
    )
    assert probe["shape"] == [24, 24, 24]
    assert probe["grid"] == [12, 12, 12]


def test_maxpool_contains_the_majority_vote_and_can_only_grow():
    rng = np.random.default_rng(11)
    labels = rng.choice([0, 2, 4], size=(8, 8, 8)).astype(np.int32)
    mask, _, maxpool = block_decimation(labels, (2, 2, 2))
    assert np.all(maxpool[mask])
    assert int(maxpool.sum()) >= int(mask.sum())


# ---------------------------------------------------------------------------
# The boundary counter
# ---------------------------------------------------------------------------


def test_boundary_counter_counts_faces_and_treats_the_array_border_as_outside():
    solid = np.zeros((5, 5, 5), dtype=bool)
    solid[1:4, 1:4, 1:4] = True  # 3x3x3: every voxel but the centre touches out
    assert _boundary_voxels(solid) == 26

    single = np.zeros((5, 5, 5), dtype=bool)
    single[2, 2, 2] = True
    assert _boundary_voxels(single) == 1

    # A 2x2x2 array of tumour: every voxel is a corner, so all 8 touch the
    # outside. In a 3x3x3 array of tumour the centre has six tumour faces, so it
    # is the one voxel that is NOT boundary.
    assert _boundary_voxels(np.ones((2, 2, 2), dtype=bool)) == 8
    assert _boundary_voxels(np.ones((3, 3, 3), dtype=bool)) == 26

    with pytest.raises(ValueError, match="must be 3D"):
        _boundary_voxels(np.ones((2, 2), dtype=bool))


# ---------------------------------------------------------------------------
# The geometry of a real case
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def geometry():
    return decimation_geometry(str(SEG), levels=GRID_LEVELS)


def test_majority_vote_deflates_and_the_maxpool_alternative_inflates(geometry):
    """The direction of both biases, on a real expert segmentation."""
    for name, level in geometry["levels"].items():
        assert level["volume_bias"] < 0.0, name
        assert level["maxpool_volume_bias"] > 0.0, name
        assert level["merged_voxels"] > 0, name
        assert 0.0 < level["merged_fraction_of_tumour"] < 1.0, name
        assert 0.0 < level["dice_vs_source"] <= 1.0, name


def test_the_vote_tracks_the_source_volume_far_better_than_the_naive_scaling(geometry):
    """A strict-majority vote should be close to unbiased, and it is.

    The decimated volume is within a few percent of the source's own 1 mm
    volume at every level -- so the transform is not the reason a volume read
    back from a derived file is wrong. That is a separate, measured defect.
    """
    for name, level in geometry["levels"].items():
        assert abs(level["volume_bias"]) < 0.05, name
        assert level["dice_vs_source"] > 0.80, name
    # Resolution costs detail monotonically here: finer grids overlap more.
    dice = [geometry["levels"][n]["dice_vs_source"] for n in GRID_LEVELS]
    assert dice == sorted(dice, reverse=True)


def test_volume_is_the_voxel_count_times_the_decimated_spacing(geometry):
    for name, level in geometry["levels"].items():
        voxel_mm3 = float(np.prod(level["decimated_spacing_mm"]))
        assert level["voxel_mm3"] == pytest.approx(voxel_mm3)
        assert level["voxel_mm"] == pytest.approx(voxel_mm3 ** (1.0 / 3.0))
        assert level["volume_mm3"] == pytest.approx(level["tumour_voxels"] * voxel_mm3)
        assert level["volume_bias"] == pytest.approx(
            level["volume_mm3"] / level["source_volume_mm3"] - 1.0
        )


def test_a_level_that_loses_every_enhancing_voxel_is_refused_not_substituted():
    """No silent fallback: a level whose labels changed rule would confound."""
    labels = np.zeros((8, 8, 8), dtype=np.int32)
    labels[2:6, 2:6, 2:6] = 2  # edema only, label 4 absent
    with pytest.raises(ValueError, match="no enhancing voxels"):
        _geometry_at_level(labels, (2, 2, 2), (2.0, 2.0, 2.0))
    with pytest.raises(ValueError, match="empty tumour mask"):
        _geometry_at_level(np.zeros((8, 8, 8), dtype=np.int32), (2, 2, 2), (2.0, 2.0, 2.0))


def test_renderings_carry_the_measured_numbers(geometry):
    text = render_grid_artifact_report(geometry)
    assert "GRID ARTIFACT" in text
    assert "shipped" in text
    assert f"{geometry['source_tumour_voxels']:,}" in text


# ---------------------------------------------------------------------------
# The diffusivity bridge
# ---------------------------------------------------------------------------


def test_matched_diffusion_is_the_identity_at_the_anchor_and_scales_by_area():
    assert matched_diffusion(1.0) == pytest.approx(0.35)
    assert matched_diffusion(2.0) == pytest.approx(0.35 / 4.0)
    for voxel_mm in (0.5, 1.0, 2.0, 3.634, 7.268):
        d = matched_diffusion(voxel_mm)
        assert physical_diffusion_mm2_per_day(d, voxel_mm) == pytest.approx(0.35)
    assert matched_diffusion(3.634) < matched_diffusion(2.0) < matched_diffusion(1.0)
    with pytest.raises(ValueError, match="must be positive"):
        matched_diffusion(0.0)


# ---------------------------------------------------------------------------
# The committed derived volumes
# ---------------------------------------------------------------------------


def test_round_trip_measures_both_defects_separately():
    audit = derived_round_trip(str(SEG), str(DERIVED))
    stored, _ = read_nifti(str(DERIVED))
    assert audit["tumour_voxels"] == int((np.asarray(stored) > 0).sum())
    # The array question and the header question are independent measurements.
    assert isinstance(audit["stored_agrees_with_aligned_block_vote"], bool)
    assert isinstance(audit["spacing_consistent"], bool)
    assert audit["stored_agrees_with_aligned_block_vote"] is (
        audit["stored_disagreeing_voxels"] == 0
    )
    # Dice and the disagreement count are two readings of one overlap, so the
    # record cannot report one without implying the other.
    total = audit["tumour_voxels"] + audit["axis_aligned_tumour_voxels"]
    assert audit["stored_disagreeing_voxels"] == int(
        round(total * (1.0 - audit["stored_dice_vs_aligned"]))
    )
    assert audit["volume_report_factor"] == pytest.approx(
        audit["traced_volume_mm3"] / audit["reported_volume_mm3"]
    )


def test_committed_audit_covers_every_case_with_a_cached_source():
    audit = committed_derived_audit()
    assert audit["n_audited"] >= 6
    assert audit["unaudited_no_cached_source"] == []
    assert audit["n_with_array_disagreement"] == sum(
        1 for e in audit["cases"].values() if not e["stored_agrees_with_aligned_block_vote"]
    )
    assert audit["n_with_spacing_mismatch"] == sum(
        1 for e in audit["cases"].values() if not e["spacing_consistent"]
    )
    assert audit["all_arrays_are_the_axis_aligned_block_vote"] is (
        audit["n_with_array_disagreement"] == 0
    )
    assert audit["all_headers_carry_the_decimated_spacing"] is (
        audit["n_with_spacing_mismatch"] == 0
    )
    assert "audited" in render_round_trip_report(audit)


def test_round_trip_refuses_a_file_from_a_different_grid(tmp_path):
    arr, _ = read_nifti(str(DERIVED))
    wrong = tmp_path / "wrong.nii.gz"
    write_nifti(str(wrong), np.asarray(arr)[:, :, :5], voxel_sizes=(1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="not this case's engine grid"):
        derived_round_trip(str(SEG), str(wrong))


# ---------------------------------------------------------------------------
# The outcome cohort
# ---------------------------------------------------------------------------


def test_spread_refuses_a_single_run_and_flags_a_degenerate_readout():
    assert _spread([1.0, 3.0])["spread_rel"] == pytest.approx(1.0)
    assert _spread([2.0, 2.0])["degenerate"] is True
    assert _spread([2.0, 2.0])["spread_rel"] == 0.0
    with pytest.raises(ValueError, match="at least two runs"):
        _spread([1.0])


def test_cohort_runs_end_to_end_and_reports_physics_per_configuration():
    """A miniature cohort: one level, one arm, two seeds, a short horizon."""
    report = grid_resolution_cohort(
        str(SEG),
        levels={"fine_2p0mm": (2, 2, 2)},
        seeds=(101, 102),
        days=30,
        modes={"noise_off": 0.0},
        arms=("mtd",),
    )
    assert report["reference_config"] == "fine_2p0mm_matched_D"
    cfg = report["configs"]["fine_2p0mm_matched_D"]
    assert cfg["physical_d_mm2_per_day"] == pytest.approx(0.35)
    declared = report["configs"]["fine_2p0mm_declared_D"]
    assert declared["physical_d_mm2_per_day"] == pytest.approx(0.35 * 4.0)
    # A single level leaves exactly one contrast: its own declared-D config
    # against its matched-D config. No grid effect is measurable with one grid.
    assert list(report["contrasts"]["noise_off"]) == [
        "fine_2p0mm_declared_D_vs_fine_2p0mm_matched_D"
    ]
    spread = report["spreads"]["noise_off"]["fine_2p0mm_matched_D"]["mtd"]["pfs"]
    assert set(spread) == {"min", "max", "mean", "spread_rel", "degenerate"}
    assert spread["degenerate"] is (spread["min"] == spread["max"])


def test_provenance_and_refusals_are_explicit():
    prov = grid_artifact_provenance()
    assert prov["adaptive_refinement"] is EvidenceStatus.PROVEN_FAILED
    assert prov["sub_voxel_recovery_of_discarded_detail"] is EvidenceStatus.PROVEN_FAILED
    assert prov["outcome_drift_across_grids"] is EvidenceStatus.ESTABLISHED
    refusals = what_the_grid_cannot_show()
    assert any("adaptive" in r for r in refusals)
    assert any("scalar D" in r for r in refusals)


# ---------------------------------------------------------------------------
# The committed record
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def record():
    if not RECORD.exists():
        pytest.skip("outputs_real/grid_artifact.json not built (run _grid_artifact_cohort.py)")
    with RECORD.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_record_recomputes_its_own_verdicts(record):
    """The committed record must agree with a live re-measurement.

    This is the test that notices a repaired decimation rule: the record holds
    what the shipped rule produced at the time it was built, and a live
    recomputation that disagrees means the record -- not the code -- is stale.
    """
    live = decimation_geometry(record["case"], levels=GRID_LEVELS)
    assert live["source_tumour_voxels"] == record["geometry"]["source_tumour_voxels"]
    for name in GRID_LEVELS:
        stored = record["geometry"]["levels"][name]
        now = live["levels"][name]
        for key in (
            "tumour_voxels",
            "shipped_tumour_voxels",
            "shipped_disagreeing_voxels",
            "merged_voxels",
            "maxpool_tumour_voxels",
            "boundary_voxels",
        ):
            assert stored[key] == now[key], (name, key)
        for key in ("volume_bias", "dice_vs_source", "shipped_dice_vs_aligned", "voxel_mm3"):
            assert stored[key] == pytest.approx(now[key]), (name, key)
        assert stored["shipped_agrees_voxelwise"] == now["shipped_agrees_voxelwise"]
    assert (
        record["geometry"]["shipped_rule_is_the_axis_aligned_block_vote"]
        == live["shipped_rule_is_the_axis_aligned_block_vote"]
    )
    assert record["geometry"]["shipped_rule_dice_vs_axis_aligned"] == pytest.approx(
        live["shipped_rule_dice_vs_axis_aligned"]
    )

    live_audit = committed_derived_audit()
    stored_audit = record["round_trip"]
    assert stored_audit["n_audited"] == live_audit["n_audited"]
    assert stored_audit["unaudited_no_cached_source"] == []
    for case, entry in live_audit["cases"].items():
        assert case in stored_audit["cases"]
        stored = stored_audit["cases"][case]
        assert stored["tumour_voxels"] == entry["tumour_voxels"]
        assert stored["stored_disagreeing_voxels"] == entry["stored_disagreeing_voxels"]
        assert stored["axis_aligned_tumour_voxels"] == entry["axis_aligned_tumour_voxels"]
        assert stored["stored_dice_vs_aligned"] == pytest.approx(
            entry["stored_dice_vs_aligned"]
        )
        assert stored["spacing_consistent"] == entry["spacing_consistent"]
        assert stored["volume_report_factor"] == pytest.approx(
            entry["volume_report_factor"]
        )
    assert (
        stored_audit["all_arrays_are_the_axis_aligned_block_vote"]
        == live_audit["all_arrays_are_the_axis_aligned_block_vote"]
    )
    assert (
        stored_audit["all_headers_carry_the_decimated_spacing"]
        == live_audit["all_headers_carry_the_decimated_spacing"]
    )


def test_record_reports_the_two_defects_rather_than_hiding_them(record):
    """Whatever the live verdicts are, the record states them and their counts."""
    audit = record["round_trip"]
    assert audit["n_with_array_disagreement"] == sum(
        1 for e in audit["cases"].values() if not e["stored_agrees_with_aligned_block_vote"]
    )
    assert audit["n_with_spacing_mismatch"] == sum(
        1 for e in audit["cases"].values() if not e["spacing_consistent"]
    )
    # Whether they are defects or clean, the numbers must be recorded, and a
    # mismatch must carry a factor > 1 (the file would understate its tumour).
    for entry in audit["cases"].values():
        if entry["spacing_consistent"]:
            assert entry["volume_report_factor"] == pytest.approx(1.0, rel=1e-6)
        else:
            assert entry["volume_report_factor"] > 1.0
        if entry["stored_agrees_with_aligned_block_vote"]:
            assert entry["stored_dice_vs_aligned"] == pytest.approx(1.0)
        else:
            assert entry["stored_dice_vs_aligned"] < 1.0


def test_record_cohort_arithmetic_is_the_arithmetic_it_reports(record):
    cohort = record["cohort"]
    assert cohort["seeds"]
    assert set(cohort["modes"]) == {"shipped_noise", "noise_off"}
    assert cohort["reference_config"] in cohort["configs"]
    ref_level = cohort["configs"][cohort["reference_config"]]["level"]
    for mode, per_config in cohort["contrasts"].items():
        for name, arms in per_config.items():
            candidate_cfg = name.split("_vs_")[0]
            for arm, readouts in arms.items():
                for metric, entry in readouts.items():
                    base = np.asarray(entry["base"], dtype=float)
                    cand = np.asarray(entry["candidate"], dtype=float)
                    assert base.tolist() == pytest.approx(
                        cohort["runs"][mode][cohort["reference_config"]][arm][
                            "readouts"
                        ][metric]
                    )
                    assert cand.tolist() == pytest.approx(
                        cohort["runs"][mode][candidate_cfg][arm]["readouts"][metric]
                    )
                    assert entry["mean_delta"] == pytest.approx(float((cand - base).mean()))
                    assert entry["max_abs_delta"] == pytest.approx(
                        float(np.abs(cand - base).max())
                    )
                    assert entry["n_seeds"] == len(cohort["seeds"])
                    spread = cohort["spreads"][mode][cohort["reference_config"]][arm][
                        metric
                    ]
                    assert entry["reference_spread_rel"] == pytest.approx(
                        spread["spread_rel"]
                    )
                    assert entry["reference_spread_degenerate"] is spread["degenerate"]
                    if entry["delta_over_spread"] is None:
                        assert spread["degenerate"] or base.mean() == 0.0
                    else:
                        scale = abs(float(base.mean()))
                        assert entry["delta_over_spread"] == pytest.approx(
                            abs(entry["mean_delta"]) / scale / spread["spread_rel"]
                        )
    # The configurations cover every level, matched and declared.
    for level in GRID_LEVELS:
        assert f"{level}_matched_D" in cohort["configs"]
        assert f"{level}_declared_D" in cohort["configs"]
    # The declared configurations all share one voxel-unit value; the matched
    # ones span grids at ONE physical value.
    declared = {
        cohort["configs"][k]["d_voxel"]
        for k in cohort["configs"]
        if k.endswith("_declared_D")
    }
    assert declared == {0.35}
    matched_physical = [
        cohort["configs"][k]["physical_d_mm2_per_day"]
        for k in cohort["configs"]
        if k.endswith("_matched_D")
    ]
    assert len(matched_physical) == len(GRID_LEVELS)
    for value in matched_physical:
        assert value == pytest.approx(0.35)
    assert cohort["reference_config"].startswith(ref_level)


def test_the_shipped_rule_is_measured_against_the_vote_not_assumed(record):
    """The headline number of this pass, pinned as a measured comparison."""
    shipped = record["geometry"]["levels"][SHIPPED_LEVEL]
    assert shipped["shipped_agrees_voxelwise"] is (
        shipped["shipped_disagreeing_voxels"] == 0
    )
    probe = record["geometry"]["partition_probe"]
    assert probe["n_probed"] > 0
    assert (
        probe["n_landed_in_their_own_engine_voxel"]
        + probe["n_landed_elsewhere"]
        + probe["n_dropped_entirely"]
        == probe["n_probed"]
    )
    assert probe["is_a_block_partition"] is (
        probe["n_landed_in_their_own_engine_voxel"] == probe["n_probed"]
    )
    if shipped["shipped_agrees_voxelwise"]:
        assert shipped["shipped_dice_vs_aligned"] == pytest.approx(1.0)
    else:
        assert shipped["shipped_dice_vs_aligned"] < 1.0
        assert shipped["shipped_disagreeing_voxels"] > 0
    live = decimation_geometry(record["case"], levels=GRID_LEVELS)
    assert live["shipped_rule_is_the_axis_aligned_block_vote"] is (
        live["levels"][SHIPPED_LEVEL]["shipped_disagreeing_voxels"] == 0
    )
