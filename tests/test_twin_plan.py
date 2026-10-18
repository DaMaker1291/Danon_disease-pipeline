"""Tests for the digital-twin planning bridge and its API endpoint.

Pins: the end-to-end mask->plan chain, the paired-control structure (same
seeds), VTI export validity, the tumour-preserving downsample, and the API
contract including its error paths.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from danon.evosteer import make_gbm_phantom
from danon.imaging import read_nifti, write_nifti


@pytest.fixture(scope="module")
def mask_path(tmp_path_factory):
    geom = make_gbm_phantom(size=24, seed=3)
    p = tmp_path_factory.mktemp("twin") / "mask.nii.gz"
    write_nifti(str(p), geom.tumor_mask.astype("u1"))
    return str(p)


@pytest.fixture(scope="module")
def plan(mask_path):
    from danon.twin_plan import plan_from_mask

    return plan_from_mask(mask_path, days=120, n_replicates=3, export_vti=True)


class TestPlanFromMask:
    def test_daily_curves_length_matches_horizon(self, plan):
        assert len(plan.daily_curves["days"]) == 120
        for key in ("burden_S", "burden_R", "dose_A", "dose_B", "mtd_burden_total"):
            assert len(plan.daily_curves[key]) == 120

    def test_paired_metrics_present(self, plan):
        pm = plan.paired_metrics
        assert pm["n_replicates"] == 3
        assert len(pm["policy_pfs_replicates"]) == 3
        assert len(pm["mtd_pfs_replicates"]) == 3
        assert pm["policy_median_pfs_days"] > 0
        assert pm["mtd_median_pfs_days"] > 0

    def test_geometry_summary_measured_from_file(self, plan):
        g = plan.geometry_summary
        assert g["tumour_voxels"] > 0
        assert g["grid_shape_xyz"] == list(plan_daily_shape(plan))
        assert 0.0 < g["budget_fraction"] <= 1.0
        assert g["vasculature_source"] == "null-uniform"

    def test_provenance_labels_constants_as_assumed(self, plan):
        assert "ASSUMED" in plan.provenance["constants"]
        assert "NOT patient outcome predictions" in plan.provenance["predictions"]
        assert "NOT named clinical agents" in plan.provenance["drugs"]

    def test_notes_carry_schedule_disclaimer(self, plan):
        joined = " ".join(plan.notes)
        assert "NOT clinically observable" in joined

    def test_vti_exported_when_requested(self, plan):
        assert plan.vti_bytes is not None
        assert plan.vti_bytes.startswith(b"<?xml")
        assert b"<ImageData" in plan.vti_bytes
        assert b'Name="sensitive"' in plan.vti_bytes
        assert b'Name="tumour_mask"' in plan.vti_bytes

    def test_vti_suppressible(self, mask_path):
        from danon.twin_plan import plan_from_mask

        r = plan_from_mask(mask_path, days=60, n_replicates=1, export_vti=False)
        assert r.vti_bytes is None

    def test_days_bounds_enforced(self, mask_path):
        from danon.twin_plan import plan_from_mask

        with pytest.raises(ValueError, match="days"):
            plan_from_mask(mask_path, days=0)
        with pytest.raises(ValueError, match="days"):
            plan_from_mask(mask_path, days=366)

    def test_replicate_bounds_enforced(self, mask_path):
        from danon.twin_plan import plan_from_mask

        with pytest.raises(ValueError, match="n_replicates"):
            plan_from_mask(mask_path, days=60, n_replicates=11)

    def test_json_payload_is_serializable(self, plan):
        import json

        blob = json.dumps(plan.to_json_dict())
        assert "schedule_text" in blob
        assert "paired_metrics" in blob


class TestDownsample:
    def test_small_mask_passes_through(self):
        from danon.twin_plan import downsample_mask_to_budget

        m = np.zeros((32, 32, 32), dtype=bool)
        m[10:20, 10:20, 10:20] = True
        out, factors = downsample_mask_to_budget(m)
        assert out.shape == m.shape
        assert factors == []
        assert out.sum() == m.sum()

    def test_oversize_mask_halved_and_tumour_preserved(self):
        from danon.twin_plan import downsample_mask_to_budget

        m = np.zeros((512, 512, 512), dtype=bool)
        m[200:300, 200:300, 100:200] = True
        out, factors = downsample_mask_to_budget(m)
        assert out.size <= 4_000_000
        assert len(factors) >= 1
        assert int(out.sum()) > 0  # a max-pool over tumour stays tumour

    def test_single_voxel_survives_max_pooling(self):
        from danon.twin_plan import downsample_mask_to_budget

        m = np.zeros((512, 512, 512), dtype=bool)
        m[3, 7, 9] = True
        out, _ = downsample_mask_to_budget(m)
        assert int(out.sum()) == 1


class TestDerivedSpacing:
    """A downsampled file's header must carry the DOWNSAMPLED spacing.

    Writing the source spacing back understates every physical volume taken
    from the derived file by 8x per halving -- the defect `danon.grid_artifact`
    records against the committed derived volumes (a 1 mm header over 48 mm^3
    voxels), and what `danon.realdata` avoids by building `source_vox * factors`.
    """

    def test_scales_spacing_by_the_applied_factor(self):
        from danon.twin_plan import derived_voxel_sizes

        assert derived_voxel_sizes((1.0, 1.0, 2.0), [2, 2]) == (4.0, 4.0, 8.0)
        assert derived_voxel_sizes((0.8, 0.8, 3.0), []) == (0.8, 0.8, 3.0)

    def test_rejects_nonpositive_factors(self):
        from danon.twin_plan import derived_voxel_sizes

        with pytest.raises(ValueError, match="positive"):
            derived_voxel_sizes((1.0, 1.0, 1.0), [2, 0])

    def test_derived_volume_never_understates_the_source(self):
        from danon.twin_plan import derived_voxel_sizes, downsample_mask_to_budget

        m = np.zeros((128, 128, 128), dtype=bool)
        m[40:80, 40:80, 40:80] = True  # 64,000 tumour voxels at 1 mm
        out, factors = downsample_mask_to_budget(m, budget=40_000)
        assert factors, "the budget should force at least one halving"
        assert out.size <= 40_000
        sizes = derived_voxel_sizes((1.0, 1.0, 1.0), factors)
        assert float(np.prod(sizes)) == 64.0  # two halvings -> 4 mm per axis
        source_mm3 = float(m.sum()) * 1.0 ** 3
        derived_mm3 = float(out.sum()) * float(np.prod(sizes))
        assert derived_mm3 >= source_mm3


class TestInputProvenance:
    """A recorded run must be re-identifiable from the file it was given."""

    def test_digest_is_over_the_bytes_as_received(self, mask_path):
        from danon.imaging import mask_bytes_provenance

        raw = open(mask_path, "rb").read()
        prov = mask_bytes_provenance(raw)
        assert prov["input_sha256"] == hashlib.sha256(raw).hexdigest()
        assert prov["original_grid_shape_xyz"] == [
            int(s) for s in read_nifti(mask_path)[0].shape
        ]
        assert prov["source_voxel_mm"] > 0
        assert len(prov["source_voxel_sizes_mm"]) == 3

    def test_a_flipped_payload_byte_moves_the_digest(self, tmp_path):
        from danon.imaging import mask_bytes_provenance

        p = tmp_path / "m.nii"
        write_nifti(str(p), np.ones((8, 8, 8), dtype="u1"))
        raw = bytearray(p.read_bytes())
        before = mask_bytes_provenance(bytes(raw))["input_sha256"]
        raw[-1] ^= 0x01  # last payload byte; the header is untouched
        assert mask_bytes_provenance(bytes(raw))["input_sha256"] != before

    def test_compressed_input_is_hashed_compressed(self, tmp_path):
        from danon.imaging import mask_bytes_provenance

        arr = np.ones((8, 8, 8), dtype="u1")
        plain, gz = tmp_path / "m.nii", tmp_path / "m.nii.gz"
        write_nifti(str(plain), arr)
        write_nifti(str(gz), arr)
        raw_gz = gz.read_bytes()
        prov = mask_bytes_provenance(raw_gz)
        assert prov["input_sha256"] == hashlib.sha256(raw_gz).hexdigest()
        # same array, same declared geometry -- different bytes, different id
        assert prov["input_sha256"] != mask_bytes_provenance(plain.read_bytes())[
            "input_sha256"
        ]
        assert prov["original_grid_shape_xyz"] == list(arr.shape)


def plan_daily_shape(plan):
    return plan.geometry_summary["grid_shape_xyz"]


class TestApiEndpoint:
    def test_plan_endpoint_roundtrip(self, mask_path):
        from fastapi.testclient import TestClient
        from api_server import app

        client = TestClient(app)
        with open(mask_path, "rb") as fh:
            r = client.post(
                "/api/twin/plan",
                files={"file": ("mask.nii.gz", fh.read())},
            )
        assert r.status_code == 200
        d = r.json()
        for key in ("geometry", "scheduleText", "dailyCurves", "pairedMetrics",
                    "events", "notes", "provenance", "vtiBase64"):
            assert key in d, key
        assert len(d["vtiBase64"]) > 1000
        # The input's provenance is reported even when nothing was downsampled.
        g = d["geometry"]
        assert g["originalGridShapeXyz"] == g["gridShapeXyz"]
        assert g["sourceVoxelMm"] > 0
        assert len(g["sourceVoxelSizesMm"]) == 3
        assert len(g["inputSha256"]) == 64
        # The digest is also carried in the provenance record, so the run is
        # tied to the file even for a reader who never looks at geometry.
        assert d["provenance"]["inputSha256"] == g["inputSha256"]
        assert "downsampleFactors" not in g
        assert "downsamplePerAxis" not in g

    def test_endpoint_rejects_non_nifti(self):
        from fastapi.testclient import TestClient
        from api_server import app

        client = TestClient(app)
        r = client.post("/api/twin/plan", files={"file": ("x.txt", b"hello")})
        assert r.status_code == 415

    def test_endpoint_rejects_garbage_nifti(self):
        from fastapi.testclient import TestClient
        from api_server import app

        client = TestClient(app)
        r = client.post(
            "/api/twin/plan", files={"file": ("x.nii", b"\x00" * 352)}
        )
        assert r.status_code == 422

    def test_endpoint_rejects_empty_mask(self, tmp_path):
        from fastapi.testclient import TestClient
        from api_server import app
        from danon.imaging import write_nifti

        p = tmp_path / "empty.nii.gz"
        write_nifti(str(p), np.zeros((8, 8, 8), dtype="u1"))
        client = TestClient(app)
        with open(p, "rb") as fh:
            r = client.post("/api/twin/plan", files={"file": ("empty.nii.gz", fh.read())})
        assert r.status_code == 422
