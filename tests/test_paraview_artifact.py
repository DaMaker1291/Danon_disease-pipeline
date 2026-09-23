"""Tests for the paraview app artifact.

Pins: the payload embeds the committed VTI byte-identically (sha256 match),
the measured statistics agree with an independent read of the same file,
and the builder raises rather than inventing a placeholder when the file
is absent.
"""

from __future__ import annotations

import base64
import hashlib

import pytest

from danon.app_artifacts import (
    PARAVIEW_VTI_PATH,
    build_paraview_artifact,
)
from danon.schedule import read_fields_vti


@pytest.fixture(scope="module")
def artifact():
    if not __import__("os").path.exists(PARAVIEW_VTI_PATH):
        pytest.skip("committed BraTS VTI artifact not present")
    return build_paraview_artifact()


def test_vti_payload_is_byte_identical_to_the_committed_file(artifact):
    raw = open(PARAVIEW_VTI_PATH, "rb").read()
    d = artifact["data"]
    assert d["vti_size_bytes"] == len(raw)
    assert d["vti_sha256"] == hashlib.sha256(raw).hexdigest()
    assert base64.b64decode(d["vti_base64"]) == raw


def test_field_stats_match_an_independent_read(artifact):
    vti = read_fields_vti(PARAVIEW_VTI_PATH)
    d = artifact["data"]
    assert d["grid"]["dims"] == list(vti.dims)
    assert d["grid"]["spacing_mm"] == vti.spacing_mm
    assert d["grid"]["tumour_voxels"] == int(vti.mask.sum())
    assert d["arrays_present"] == sorted(vti.fields)
    for name, st in d["field_stats"].items():
        arr = vti.fields[name]
        assert st["min"] == float(arr.min())
        assert st["max"] == float(arr.max())


def test_headline_matches_the_paraview_doc(artifact):
    """The 35x sanctuary contrast and R-fraction 1.00 come from PARAVIEW.md."""
    d = artifact["data"]
    assert d["headline"]["resistant_fraction"] == 1.0
    assert d["headline"]["diffusion_contrast_ratio"] == 35.0
    assert "simulation" in d["captions"]["short"].lower()


def test_builder_raises_when_the_artifact_is_missing(tmp_path, monkeypatch):
    """A missing file raises; it must never fabricate a placeholder volume."""
    import danon.app_artifacts as aa

    monkeypatch.setattr(aa, "PARAVIEW_VTI_PATH", str(tmp_path / "nope.vti"))
    with pytest.raises(FileNotFoundError, match="ParaView artifact missing"):
        aa.build_paraview_artifact()


def test_registered_in_builders():
    from danon.app_artifacts import BUILDERS

    assert BUILDERS["paraview"] is build_paraview_artifact
