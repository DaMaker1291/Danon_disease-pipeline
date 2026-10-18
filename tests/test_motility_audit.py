"""Tests for the motility-coefficient audit.

The audit's claim is narrow and falsifiable: the engine's migration coefficient is
per-voxel, so it means different physics on different grids; the consequence is
measured; and the engine's default has NOT been changed underneath the release
artifacts. These tests re-derive each part rather than trusting the artifact.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from danon.anisotropy import (
    CELL_MOTILITY_MM2_PER_DAY_RANGE,
    LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY,
    PHANTOM_VOXEL_MM,
    REAL_VOXEL_MM,
    d_to_physical,
)
from danon.evosteer import TumorEvolutionSimulator, make_gbm_phantom

ARTIFACT = os.path.join("outputs_real", "motility_audit.json")


class TestTheConstantIsGridDependent:
    def test_the_shipped_coefficient_is_a_bare_literal(self):
        assert LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY == 0.02

    def test_one_literal_two_physical_diffusivities(self):
        phantom = d_to_physical(LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY, PHANTOM_VOXEL_MM)
        real = d_to_physical(LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY, REAL_VOXEL_MM)
        assert phantom == pytest.approx(0.02)
        assert real == pytest.approx(0.2689, abs=5e-4)
        assert real / phantom == pytest.approx(
            (REAL_VOXEL_MM / PHANTOM_VOXEL_MM) ** 2, rel=1e-9
        )

    def test_the_shipped_value_is_out_of_range_on_both_grids(self):
        lo, hi = CELL_MOTILITY_MM2_PER_DAY_RANGE
        phantom = d_to_physical(LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY, PHANTOM_VOXEL_MM)
        real = d_to_physical(LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY, REAL_VOXEL_MM)
        assert phantom > hi
        assert real > hi
        # and the physical-on-real-grid settings do land back inside the range
        assert lo <= d_to_physical(hi / (REAL_VOXEL_MM / PHANTOM_VOXEL_MM) ** 2, REAL_VOXEL_MM) <= hi


class TestTheDefaultWasNotSilentlyRetuned:
    def test_a_default_simulator_still_uses_the_legacy_literal(self):
        sim = TumorEvolutionSimulator(make_gbm_phantom(size=24, seed=12345))
        assert sim._cell_motility is None

    def test_an_override_is_accepted_and_validated(self):
        geom = make_gbm_phantom(size=24, seed=12345)
        assert TumorEvolutionSimulator(geom, cell_motility=0.0)._cell_motility == 0.0
        with pytest.raises(ValueError):
            TumorEvolutionSimulator(geom, cell_motility=-1.0)


class TestTheArtifactIsAnchoredAndHasContent:
    @pytest.fixture(scope="class")
    def artifact(self):
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ARTIFACT
        )
        if not os.path.exists(path):
            pytest.skip("motility audit not built")
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def test_default_arm_reproduces_the_release_1_winner(self, artifact):
        assert artifact["release_1_anchor"]["reproduced_by_default_arm"] is True
        rows = {r["label"]: r for r in artifact["settings"]}
        assert rows["legacy_default"]["winner"]["median_aubc"] == pytest.approx(
            artifact["release_1_anchor"]["median_aubc"], rel=1e-12
        )

    def test_physical_columns_re_derive_from_the_live_conversion(self, artifact):
        rows = {r["label"]: r for r in artifact["settings"]}
        row = rows["legacy_default"]
        assert row["phantom_mm2_per_day"] == pytest.approx(
            d_to_physical(LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY, PHANTOM_VOXEL_MM),
            rel=1e-12,
        )
        assert row["real_mm2_per_day"] == pytest.approx(
            d_to_physical(LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY, REAL_VOXEL_MM),
            rel=1e-12,
        )

    def test_the_sensitivity_is_not_a_non_result(self, artifact):
        rows = [r for r in artifact["settings"] if r["label"] != "legacy_default"]
        assert max(abs(r["aubc_delta_vs_default"]) for r in rows) > 0.01
        ratios = [
            v["aubc_ratio_winner_over_reference"]
            for v in artifact["reference_arm"].values()
            if v.get("aubc_ratio_winner_over_reference") is not None
        ]
        assert len(ratios) >= 2
        assert max(ratios) - min(ratios) > 0.005

    def test_conclusions_are_robust_where_the_number_is_not(self, artifact):
        """PFS and dominance are pinned in EVERY setting, including motility off."""
        for row in artifact["settings"]:
            winner = row["winner"]
            assert winner["median_pfs_days"] == 365.0
            assert winner["median_dominance_day"] == 365.0

    def test_the_artifact_carries_its_own_limits(self, artifact):
        assert len(artifact["cannot_show"]) >= 3
        # the defect names why nothing raises, rather than only stating the numbers
        assert "voxel" in artifact["defect"]["why_nothing_raises"]
        assert artifact["operating_point"]["seeds"] == list(range(30))
        assert artifact["operating_point"]["days"] == 365
