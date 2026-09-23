"""Tests for the real-imaging intake module.

Pins: NIfTI-1 round-trip fidelity, refusal paths (truncation, bad magic,
empty/oversized masks, mismatched maps), provenance labeling of the two
derived maps (null-uniform vasculature = sanctuary-free by construction),
and that the engine actually consumes the resulting geometry.
"""

from __future__ import annotations

import numpy as np
import pytest

from danon.cure import EvidenceStatus
from danon.evosteer import (
    CollateralSwitchPolicy,
    MTDPolicy,
    TumorEvolutionSimulator,
    make_gbm_phantom,
)
from danon.imaging import (
    MAX_TOTAL_VOXELS,
    read_nifti,
    write_nifti,
    from_nifti_mask,
    imaging_provenance,
    imaging_what_this_model_cannot_know,
)


@pytest.fixture(scope="module")
def mask_file(tmp_path_factory):
    phantom = make_gbm_phantom(size=16, seed=3)
    path = str(tmp_path_factory.mktemp("nii") / "mask.nii.gz")
    write_nifti(path, phantom.tumor_mask.astype(np.uint8), voxel_sizes=(1.0, 1.0, 1.0))
    return path, phantom


def test_nifti_roundtrip_preserves_mask(mask_file):
    path, phantom = mask_file
    data, hdr = read_nifti(path)
    assert hdr.magic == "n+1"
    assert data.shape == phantom.tumor_mask.shape
    assert int((data > 0).sum()) == int(phantom.tumor_mask.sum())
    # Binary masks survive uint8 round-trip exactly.
    assert np.array_equal((data > 0), phantom.tumor_mask)


def test_write_refuses_out_of_range(tmp_path):
    arr = np.zeros((4, 4, 4), dtype=np.float64)
    arr[1, 1, 1] = 300.0
    with pytest.raises(ValueError):
        write_nifti(str(tmp_path / "x.nii"), arr)


def test_read_refuses_garbage(tmp_path):
    p = str(tmp_path / "garbage.nii")
    with open(p, "wb") as fh:
        fh.write(b"\x00" * 100)
    with pytest.raises(ValueError):
        read_nifti(p)


def test_read_refuses_truncated_data(tmp_path):
    # Valid header promising 4096 voxels, body cut short. Uncompressed on
    # purpose: a tiny binary mask gzips below the truncation point, so the
    # cut must apply to raw bytes to actually truncate.
    phantom = make_gbm_phantom(size=16, seed=1)
    p = str(tmp_path / "trunc.nii")
    write_nifti(p, phantom.tumor_mask.astype(np.uint8))
    with open(p, "rb") as fh:
        blob = fh.read()
    with open(p, "wb") as fh:
        fh.write(blob[:352 + 100])
    with pytest.raises(ValueError):
        read_nifti(p)


def test_from_nifti_mask_labels_provenance_honestly(mask_file):
    path, phantom = mask_file
    pg = from_nifti_mask(path)
    assert pg.n_tumour_voxels == int(phantom.tumor_mask.sum())
    assert pg.vasculature_source == "null-uniform"
    assert pg.proliferative_source == "distance-proxy"
    notes = pg.notes()
    joined = " ".join(notes)
    # The sanctuary-free consequence is stated, not hidden.
    assert "NULL MODEL" in joined
    assert "abolishes diffusion sanctuaries" in joined
    # The distance proxy is labelled as a proxy, not a measurement.
    assert "not a measurement" in joined


def test_from_nifti_mask_refuses_empty(tmp_path):
    p = str(tmp_path / "empty.nii.gz")
    write_nifti(p, np.zeros((8, 8, 8), dtype=np.uint8))
    with pytest.raises(ValueError):
        from_nifti_mask(p)


def test_from_nifti_mask_refuses_oversized(tmp_path, monkeypatch):
    phantom = make_gbm_phantom(size=16, seed=2)
    p = str(tmp_path / "big.nii.gz")
    write_nifti(p, phantom.tumor_mask.astype(np.uint8))
    monkeypatch.setattr("danon.imaging.MAX_TOTAL_VOXELS", 10)
    with pytest.raises(ValueError):
        from_nifti_mask(p)


def test_supplied_maps_flow_through(mask_file, tmp_path):
    path, phantom = mask_file
    rng = np.random.default_rng(0)
    prolif = np.where(phantom.tumor_mask, rng.uniform(0.2, 1.0, phantom.tumor_mask.shape), 0.0)
    vessels = np.where(phantom.tumor_mask, rng.uniform(0.3, 1.0, phantom.tumor_mask.shape), 0.0)
    pp = str(tmp_path / "prolif.nii.gz")
    vp = str(tmp_path / "vess.nii.gz")
    write_nifti(pp, np.clip(prolif * 255, 0, 255).astype(np.uint8))
    write_nifti(vp, np.clip(vessels * 255, 0, 255).astype(np.uint8))
    pg = from_nifti_mask(path, proliferative_path=pp, vessels_path=vp)
    assert pg.proliferative_source == "file"
    assert pg.vasculature_source == "file"
    # Normalized into [0, 1].
    assert pg.geometry.vessels.max() <= 1.0 and pg.geometry.vessels.max() > 0.5
    assert "NULL MODEL" not in " ".join(pg.notes())


def test_mismatched_map_shape_refused(mask_file, tmp_path):
    path, _ = mask_file
    small = (np.random.default_rng(0).uniform(size=(8, 8, 8)) * 255).astype(np.uint8)
    pp = str(tmp_path / "wrong.nii.gz")
    write_nifti(pp, small)
    with pytest.raises(ValueError):
        from_nifti_mask(path, proliferative_path=pp)


def test_engine_consumes_real_mask_geometry(mask_file):
    path, _ = mask_file
    pg = from_nifti_mask(path)
    sim = TumorEvolutionSimulator(pg.geometry)
    tr = sim.run(
        CollateralSwitchPolicy(), days=30, initial_resistant_fraction=0.01, seed=5
    )
    assert tr.days == 30
    assert np.all(np.isfinite(tr.burden_S)) and np.all(tr.burden_S >= 0)
    # MTD runs too.
    tr_m = sim.run(MTDPolicy(), days=30, initial_resistant_fraction=0.01, seed=5)
    assert tr_m.days == 30


def test_imaging_provenance_and_limits():
    prov = imaging_provenance()
    assert prov["null_vasculature_sanctuary_free"] is EvidenceStatus.PROVEN_FAILED
    assert prov["proliferative_distance_proxy"] is EvidenceStatus.UNKNOWN
    assert prov["nifti_1_format"] is EvidenceStatus.ESTABLISHED
    joined = " ".join(imaging_what_this_model_cannot_know())
    assert "sanctuary" in joined.lower()
    assert "upstream" in joined
