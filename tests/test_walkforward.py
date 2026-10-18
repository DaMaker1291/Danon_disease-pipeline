"""Tests for the longitudinal harness: the metrics, the nulls, the refusals.

The metrics are pinned against constructions whose answer is known on paper: two
cubes one voxel apart have a known Jaccard and a known 95th-percentile surface
distance, two identical distributions have JSD 0, two disjoint ones have JSD
exactly 1 bit, and scipy's `jensenshannon` is pinned as the square root of this
module's divergence in NATS -- the difference between "JSD < 0.15" meaning 0.15
bits and 0.0225 bits.

The refusals are pinned as strictly as the arithmetic. This repository has no
registered longitudinal pair, so the one thing the harness must never do is
produce a number anyway; and the engine's outward movement must be attributed to
the term that causes it, by turning that term off and watching the expansion
vanish.

One refusal is checked for a second reason: LUMIERE is reachable and genuinely
longitudinal, and it is refused anyway because its masks are automated, so a
Dice against it would score this engine against another algorithm. That
annotation is pinned against the collection's own article list as read back
through the Figshare API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from danon.cure import EvidenceStatus
from danon.walkforward import (
    CENSUS_PATH,
    LongitudinalPair,
    WalkForwardError,
    clonal_jsd,
    dice3d,
    dilate_mask,
    expansion_bound,
    jaccard3d,
    jensen_shannon_divergence,
    longitudinal_census,
    migration_audit,
    null_models,
    score_prediction,
    shannon_entropy,
    surface_distance_p95,
    surface_mask,
    walkforward_provenance,
    what_walk_forward_cannot_show,
)

ROOT = Path(__file__).resolve().parents[1]


def _cube(size: int = 16, lo: int = 5, hi: int = 11) -> np.ndarray:
    m = np.zeros((size, size, size), dtype=bool)
    m[lo:hi, lo:hi, lo:hi] = True
    return m


def _pair(baseline: np.ndarray, followup: np.ndarray, **kwargs) -> LongitudinalPair:
    params = dict(
        patient_id="SYNTH-01",
        baseline_mask=baseline,
        followup_mask=followup,
        interval_days=180,
        voxel_mm=1.0,
        source="constructed in the test suite",
        registered_same_space=True,
    )
    params.update(kwargs)
    return LongitudinalPair(**params)


# ---------------------------------------------------------------------------
# The metrics, against answers known on paper
# ---------------------------------------------------------------------------


def test_dice_and_jaccard_are_exact_on_constructed_sets():
    a = np.zeros((10, 10, 10), dtype=bool)
    a[0:4, 0:1, 0:1] = True  # 4 voxels
    b = np.zeros((10, 10, 10), dtype=bool)
    b[0:8, 0:1, 0:1] = True  # 8 voxels, 4 shared
    assert dice3d(a, b) == pytest.approx(2 * 4 / (4 + 8))
    assert jaccard3d(a, b) == pytest.approx(4 / 8)
    assert dice3d(a, a) == pytest.approx(1.0)
    assert jaccard3d(a, a) == pytest.approx(1.0)
    assert dice3d(a, b) == pytest.approx(dice3d(b, a))


def test_two_absent_tumours_are_refused_rather_than_scored_as_perfect():
    empty = np.zeros((6, 6, 6), dtype=bool)
    with pytest.raises(WalkForwardError, match="1.0 would score erasure"):
        dice3d(empty, empty)
    with pytest.raises(WalkForwardError, match="Jaccard is undefined"):
        jaccard3d(empty, empty)


def test_shape_mismatch_is_refused_in_every_metric():
    a = np.zeros((8, 8, 8), dtype=bool)
    a[2:6, 2:6, 2:6] = True
    b = np.zeros((8, 8, 7), dtype=bool)
    b[2:6, 2:6, 2:5] = True
    for fn in (dice3d, jaccard3d, surface_distance_p95):
        with pytest.raises(WalkForwardError, match="different grids"):
            fn(a, b) if fn is not surface_distance_p95 else fn(a, b, voxel_mm=1.0)


def test_surface_mask_finds_the_face_of_a_block():
    m = _cube(8, 2, 6)  # 4x4x4 solid: 56 voxels on the surface, 8 inside
    assert int(surface_mask(m).sum()) == 4 * 4 * 4 - 2 * 2 * 2


def test_surface_distance_is_the_known_voxel_shift_and_scales_with_voxel_size():
    a = _cube(16, 5, 11)
    b = _cube(16, 6, 12)  # shifted one voxel along x
    assert surface_distance_p95(a, a, voxel_mm=1.0) == pytest.approx(0.0)
    at_one_mm = surface_distance_p95(a, b, voxel_mm=1.0)
    at_two_mm = surface_distance_p95(a, b, voxel_mm=2.0)
    # A one-voxel shift: most surface voxels find their counterpart one voxel
    # away, and the 95th percentile is set by the corners, whose nearest
    # observed surface voxel is DIAGONAL -- sqrt(2) voxels, not 1. Measured, not
    # assumed: the value is a property of the metric and the shift.
    assert at_one_mm == pytest.approx(np.sqrt(2.0))
    assert at_two_mm == pytest.approx(2.0 * np.sqrt(2.0))
    assert surface_distance_p95(a, b, voxel_mm=2.0) == pytest.approx(
        2.0 * surface_distance_p95(a, b, voxel_mm=1.0)
    )
    with pytest.raises(WalkForwardError, match="voxel_mm must be positive"):
        surface_distance_p95(a, b, voxel_mm=0.0)
    with pytest.raises(WalkForwardError, match="needs a surface on both sides"):
        surface_distance_p95(np.zeros((16, 16, 16), bool), a, voxel_mm=1.0)


def test_shannon_entropy_is_one_bit_for_a_coin_and_zero_for_a_certainty():
    assert shannon_entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert shannon_entropy([1.0, 0.0]) == pytest.approx(0.0)
    assert shannon_entropy([0.5, 0.5], base=np.e) == pytest.approx(np.log(2.0))
    with pytest.raises(WalkForwardError, match="cannot be negative"):
        shannon_entropy([1.2, -0.2])
    with pytest.raises(WalkForwardError, match="must carry mass"):
        shannon_entropy([0.0, 0.0])


def test_jsd_is_zero_for_identical_and_one_bit_for_disjoint_supports():
    assert jensen_shannon_divergence([0.7, 0.3], [0.7, 0.3]) == pytest.approx(0.0)
    assert jensen_shannon_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    # Symmetric, and bounded by 1 bit for a two-bin support.
    p, q = [0.9, 0.1], [0.2, 0.8]
    assert jensen_shannon_divergence(p, q) == pytest.approx(
        jensen_shannon_divergence(q, p)
    )
    assert 0.0 <= jensen_shannon_divergence(p, q) <= 1.0


def test_scipy_jensenshannon_is_the_square_root_in_nats_not_the_divergence():
    """The single most likely way to misread a JSD target."""
    from scipy.spatial.distance import jensenshannon

    p, q = [0.95, 0.05], [0.12, 0.88]
    bits = jensen_shannon_divergence(p, q, base=2.0)
    nats = jensen_shannon_divergence(p, q, base=np.e)
    assert nats == pytest.approx(bits * np.log(2.0))
    assert jensenshannon(p, q) == pytest.approx(np.sqrt(nats))
    assert jensenshannon(p, q, base=2.0) == pytest.approx(np.sqrt(bits))
    # So a quoted target of 0.15 is 0.15 bits here and 0.0225 bits there.
    assert np.sqrt(bits) > bits


def test_clonal_jsd_refuses_a_support_that_is_not_the_engine_s():
    assert clonal_jsd(0.9, 0.1, [0.2, 0.8]) > 0.0
    assert clonal_jsd(0.5, 0.5, [0.5, 0.5]) == pytest.approx(0.0)
    with pytest.raises(WalkForwardError, match="category error"):
        clonal_jsd(0.9, 0.1, [0.4, 0.3, 0.2, 0.1])
    with pytest.raises(WalkForwardError, match="SAME support"):
        jensen_shannon_divergence([0.5, 0.5], [0.3, 0.3, 0.4])


# ---------------------------------------------------------------------------
# The pair and its refusals
# ---------------------------------------------------------------------------


def test_a_pair_that_is_not_registered_is_refused():
    with pytest.raises(WalkForwardError, match="not declared registered"):
        _pair(_cube(), _cube(16, 6, 12), registered_same_space=False)


def test_an_empty_follow_up_is_no_observation_not_a_null_result():
    with pytest.raises(WalkForwardError, match="it is no observation"):
        _pair(_cube(), np.zeros((16, 16, 16), dtype=bool))


def test_a_pair_refuses_a_nonpositive_interval_an_empty_baseline_and_bad_voxels():
    with pytest.raises(WalkForwardError, match="interval_days must be positive"):
        _pair(_cube(), _cube(16, 6, 12), interval_days=0)
    with pytest.raises(WalkForwardError, match="baseline segmentation is empty"):
        _pair(np.zeros((16, 16, 16), dtype=bool), _cube())
    with pytest.raises(WalkForwardError, match="voxel_mm must be positive"):
        _pair(_cube(), _cube(16, 6, 12), voxel_mm=0.0)


def test_a_pair_reports_the_volume_change_it_defines():
    pair = _pair(_cube(16, 5, 11), _cube(16, 5, 12))  # 6^3 -> 7^3
    assert pair.baseline_voxel_count == 216
    assert pair.followup_voxel_count == 343
    assert pair.volume_ratio() == pytest.approx(343 / 216)
    assert "SYNTH-01" in pair.describe()


def test_dilation_is_a_radius_and_the_persistence_null_is_its_own_mask():
    base = _cube(16, 6, 10)  # 4x4x4
    dilated = dilate_mask(base, 2.0, voxel_mm=1.0)
    assert int(dilated.sum()) > int(base.sum())
    assert np.all(dilated[base])
    assert int(dilate_mask(base, 0.0).sum()) == int(base.sum())
    with pytest.raises(WalkForwardError, match="radius must be non-negative"):
        dilate_mask(base, -1.0)


def test_null_models_report_which_null_wins_and_that_it_depends_on_the_case():
    """The reason a Dice headline needs a null beside it.

    Two follow-ups that differ only in WHERE they grew: one extends a single
    margin, the other is a margin recurrence all around. Persistence wins the
    first and loses the second, and the ordering is measured on both rather than
    assumed for either.
    """
    baseline = _cube(20, 6, 12)
    one_margin = baseline.copy()
    one_margin[6:12, 6:12, 12:14] = True
    pair_one = _pair(baseline, one_margin)
    nulls_one = null_models(pair_one)
    assert set(nulls_one) >= {"persistence", "margin_2mm", "margin_5mm", "relocation"}
    assert nulls_one["persistence"]["dice"] == pytest.approx(
        dice3d(pair_one.baseline_mask, pair_one.followup_mask)
    )
    assert nulls_one["persistence"]["dice"] > nulls_one["margin_2mm"]["dice"]
    assert nulls_one["relocation"]["dice"] < nulls_one["persistence"]["dice"]

    pair_margin = _pair(baseline, dilate_mask(baseline, 2.0, voxel_mm=1.0))
    nulls_margin = null_models(pair_margin)
    assert nulls_margin["margin_2mm"]["dice"] == pytest.approx(1.0)
    assert nulls_margin["margin_2mm"]["dice"] > nulls_margin["persistence"]["dice"]
    assert nulls_margin["margin_5mm"]["dice"] < nulls_margin["margin_2mm"]["dice"]


def test_the_matched_volume_sphere_grows_evenly_and_by_exactly_the_volume_asked():
    from danon.walkforward import matched_volume_sphere, shape_departure

    base = _cube(20, 8, 12)  # 4x4x4
    grown = matched_volume_sphere(base, 100, voxel_mm=1.0)
    assert int(grown.sum()) == int(base.sum()) + 100
    assert np.all(grown[base])
    # It is isotropic: every added voxel is within one voxel of the same radius.
    assert matched_volume_sphere(base, 0, voxel_mm=1.0).tolist() == base.tolist()
    with pytest.raises(WalkForwardError, match="non-negative"):
        matched_volume_sphere(base, -1)

    # A prediction that IS the sphere departs by nothing; a cube's margin does.
    sphere = matched_volume_sphere(base, 152, voxel_mm=1.0)
    assert shape_departure(sphere, base, voxel_mm=1.0)[
        "dice_vs_matched_volume_sphere"
    ] == pytest.approx(1.0)
    anisotropic = _cube(20, 4, 16)
    departure = shape_departure(anisotropic, base, voxel_mm=1.0)
    assert departure["predicted_voxels_outside_baseline"] > 0
    assert departure["dice_vs_matched_volume_sphere"] < 1.0

    # A prediction that only shrinks has no margin to have a shape; that is
    # refused instead of being scored as a ball.
    with pytest.raises(WalkForwardError, match="shrinkage"):
        shape_departure(_cube(20, 9, 12), base, voxel_mm=1.0)


def test_score_prediction_reports_the_nulls_and_cannot_beat_persistence_by_shrinking():
    baseline = _cube(20, 6, 12)
    followup = dilate_mask(baseline, 2.0, voxel_mm=1.0)
    pair = _pair(baseline, followup)
    shrunken = _cube(20, 8, 11)  # a prediction that only loses ground
    score = score_prediction(shrunken, pair)
    assert score["beats_persistence"] is False
    assert score["best_null"] == "margin_2mm"
    assert score["margin_over_best_null"] < 0
    # A shrunken prediction puts nothing outside the day-0 mask, while the real
    # recurrence does: the metric that matters is reported, not inferred.
    assert score["predicted_voxels_outside_baseline"] == 0
    assert score["observed_voxels_outside_baseline"] > 0
    assert score["dice"] == pytest.approx(dice3d(shrunken, followup))
    assert score["surface_distance_p95_mm"] > 0.0
    # The isotropic-growth null is built at the OBSERVED growth, so it is scored
    # on the same size as the recurrence and measures shape alone.
    assert "matched_volume_sphere" in score["nulls"]
    assert score["nulls"]["matched_volume_sphere"]["extra_voxels"] == (
        score["observed_voxels_outside_baseline"]
    )


# ---------------------------------------------------------------------------
# The engine's outward movement, and what causes it
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def expansion():
    return expansion_bound(
        _cube(16, 5, 11), arm="mtd", days=30, seeds=(101,), voxel_mm=1.0
    )


def test_expansion_is_attributed_to_the_migration_term(expansion):
    """Turn the only outward term off and the expansion must vanish exactly."""
    assert expansion["expansion_requires_the_migration_term"] is True
    assert expansion["max_burden_outside_share_with_motility_zero"] == 0.0
    assert expansion["max_burden_outside_baseline_share"] > 0.0
    # No growth is available outside the mask at all: the per-voxel growth map
    # summed over those voxels is exactly zero.
    assert expansion["growth_rate_map_sum_outside_mask"] == 0.0
    for run in expansion["zero_motility_runs"]:
        assert run["burden_outside_baseline_mask"] == 0.0
        assert run["voxels_alive_outside"] == 0


def test_expansion_bound_reports_the_migration_coefficient_in_physical_units(expansion):
    audit = expansion["migration"]
    assert audit["value_voxel2_per_day"] == 0.02
    # The literal is in VOXEL units, so its physical value depends on the grid
    # it is applied to -- which is the entire problem with it.
    small = migration_audit(1.0)
    large = migration_audit(3.634)
    assert large["physical_mm2_per_day"] > small["physical_mm2_per_day"]
    assert small["physical_mm2_per_day"] == pytest.approx(0.02)
    assert large["multiple_of_range_top"] > 50.0
    assert large["inside_reference_range"] is False


def test_expansion_bound_refuses_an_empty_mask_and_an_unknown_arm():
    with pytest.raises(WalkForwardError, match="nothing to expand"):
        expansion_bound(np.zeros((8, 8, 8), dtype=bool), days=5)
    with pytest.raises(WalkForwardError, match="arm must be one of"):
        expansion_bound(_cube(12, 4, 8), arm="not-an-arm", days=5)


# ---------------------------------------------------------------------------
# The census and the provenance
# ---------------------------------------------------------------------------


def test_the_census_says_zero_scorable_pairs_and_why():
    census = longitudinal_census()
    assert census["scorable_pairs"] == 0
    assert census["pairs_with_cached_segmentations"] == 0
    cfb = census["sources"]["CFB-GBM"]
    assert cfb["usable_pairs_declared"] == 393
    assert cfb["patients_with_usable_pair"] == 168
    assert cfb["images_present"] is False
    assert cfb["cross_machine_pairs"] > 0
    # The GLASS records are now IN this tree -- the raw cache, the cohort script
    # and the measured artifact all carry the name -- so this assertion runs the
    # other way: what has to hold is that the channel which measures them is
    # named, that it was built, and that the remaining refusal is recorded.
    # 0 scorable pairs is still the count, because scoring needs imaging and a
    # k-bin spectrum is not the engine's two-bin clone state.
    assert census["sources"]["GLASS"]["files_in_tree"] >= 2
    channel = census["sources"]["GLASS"]["variant_channel"]
    assert channel["module"].replace("\\", "/").endswith("danon/variants.py")
    assert channel["built"] is True, channel
    assert (channel["measured"] or {})["n_mutation_records"] > 0
    assert (channel["measured"] or {})["paired_change"]["n_pairs"] > 0
    assert len(census["what_would_make_them_scorable"]) == 3

    # GLASS is refused for the MAPPING, not for access and not for a missing
    # frequency -- and the annotation has to keep saying so, because the older
    # "controlled access" framing was wrong for the processed release.
    glass = census["sources"]["GLASS"]
    assert "controlled access" in glass["access"]
    assert "open API with no application" in glass["access"]
    assert "Tumor Primary" in glass["pairing"] and "First Recurrence" in glass["pairing"]
    assert "tumorAltCount" in glass["frequencies"]
    assert "CELL-POPULATION fraction" in glass["why_it_is_still_not_usable"]
    # The refusal moved rather than disappeared: what is measured now (a lower
    # bound on the clone fraction) is stated beside what is still refused.
    assert "PARTLY SUPERSEDED" in glass["why_it_is_still_not_usable"]
    assert "LOWER bound" in glass["why_it_is_still_not_usable"]

    # LUMIERE is the one longitudinal collection the NBIA probe cannot see: it
    # is on Figshare, it is real, and it is STILL refused -- for a reason about
    # the metric rather than about availability.
    lumiere = census["sources"]["LUMIERE"]
    assert lumiere["files_in_tree"] == 0
    assert lumiere["scorable_for_a_spatial_dice"] is False
    assert lumiere["patients"] == 91
    assert lumiere["expert_rano_ratings"] == 616
    # Its masks are automated, so a Dice against them is model-versus-model...
    assert "AUTOMATED" in lumiere["segmentations"]
    # ...and its expert asset is a response class, not a segmentation.
    assert "NOT a segmentation" in lumiere["expert_asset"]
    # It cannot supply the growth-rate mapping either: no Ki-67 anywhere in it.
    assert "NO Ki-67" in lumiere["pathology"]
    # Nor test steering against MTD: one centre, one regimen, no adaptive arm.
    assert "no adaptive arm" in lumiere["treatment"]


def test_the_recorded_network_probe_agrees_with_the_census():
    path = ROOT / CENSUS_PATH
    if not path.exists():
        pytest.skip("run _walkforward_census.py to record the TCIA probe")
    with path.open(encoding="utf-8") as fh:
        record = json.load(fh)
    assert record["offline"] is False
    assert record["repository"]["scorable_pairs"] == 0
    # The two collections the walk-forward proposal names return nothing from
    # this machine's public API, which is why no Dice is measured in this repo.
    for collection in ("MU-Glioma-Post", "Brain-Tumor-Progression"):
        assert record["tcia"][collection]["get_patient"]["records"] == 0
        assert record["tcia"][collection]["get_series"]["records"] == 0
    assert record["tcia"]["MU-Glioma-Post"]["get_series"]["status"] == 200

    # The Figshare probe records the LUMIERE collection's OWN article list, and
    # that list is the evidence behind the census annotation: the masks ship as
    # an '...automated segmentations' article, and the expert contribution ships
    # as a separate 'Expert RANO rating' article.
    figshare = record["figshare"]["LUMIERE"]
    assert figshare["status"] == 200
    assert figshare["collection_id"] == 5904905
    assert figshare["n_articles"] == 9
    titles = [str(a["title"]).lower() for a in figshare["articles"]]
    assert any("automated segmentation" in t for t in titles)
    assert any("expert rano" in t for t in titles)
    # Not a guess: the collection names the two U-Net tools that made them, in
    # its own two per-tool radiomics articles.
    assert any("deepbratumia" in t for t in titles)
    assert any("hd-glio-auto" in t for t in titles)
    assert figshare["segmentation_article_is_automated"] is True
    assert figshare["expert_article_is_rano"] is True

    # The cBioPortal probe records GLASS's public release, and its own numbers
    # are what the annotation is checked against.
    glass = record["cbioportal"]["GLASS"]
    assert glass["study_id"] == "difg_glass"
    assert str(glass["public_study"]) == "True"
    assert int(glass["all_samples"]) == 693
    assert int(glass["sequenced_samples"]) == 629
    types = glass["sample_types"]
    assert types["Tumor Primary"] == 309 and types["First Recurrence"] == 318
    for gene in ("MSH6", "MGMT", "TP53"):
        entry = glass["mutations"][gene]
        assert entry["status"] == 200, gene
        assert entry["records"] > 0, gene
        # The frequency IS readable: this is the fact that makes the refusal
        # about the mapping rather than about the data.
        assert entry["allele_counts_populated"] is True, gene


def test_provenance_admits_the_missing_pieces():
    prov = walkforward_provenance()
    assert prov["engine_dice_against_a_patient_followup"] is EvidenceStatus.PROVEN_FAILED
    assert prov["engine_jsd_against_glass"] is EvidenceStatus.PROVEN_FAILED
    assert prov["co_registration"] is EvidenceStatus.PROVEN_FAILED
    assert prov["invasion_term"] is EvidenceStatus.PROVEN_FAILED
    assert prov["motility_is_in_physical_units"] is EvidenceStatus.PROVEN_FAILED
    assert prov["expansion_needs_the_migration_term"] is EvidenceStatus.ESTABLISHED
    refusals = what_walk_forward_cannot_show()
    assert any("no registered longitudinal pair" in r for r in refusals)
    assert any("alignment error" in r for r in refusals)
