"""Tests for the committed evolution artifact the 365-day player renders.

These pin the *deploy* contract, not the physics (the physics is pinned by
the claims registry and the engine tests): the payload must stay
browser-loadable, its frame budget must be self-consistent, the early
switch window must stay densely sampled, and every frame must really be a
standalone ParaView .vti on the artifact's declared grid.

Why a size test exists at all: the builder's frame budget is the one knob
that can silently turn this feature into a 35 MB download, and the failure
mode is invisible locally (it works on a fast laptop, then crawls on Pages).
"""

from __future__ import annotations

import base64
import json
import os

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVOLUTION_JSON = os.path.join(
    REPO_ROOT, "frontend", "public", "artifacts", "evolution.json"
)

# What the player can carry in a browser: the committed payload's budget plus
# headroom. Raising build_evolution_artifact(max_frames=...) past this is a
# deploy decision, so it has to be made deliberately.
MAX_DEPLOY_BYTES = 24 * 1024 * 1024


@pytest.fixture(scope="module")
def artifact():
    if not os.path.exists(EVOLUTION_JSON):
        pytest.skip("committed evolution artifact not present")
    with open(EVOLUTION_JSON) as fh:
        return json.load(fh)


def test_payload_stays_within_the_deploy_budget():
    size = os.path.getsize(EVOLUTION_JSON)
    assert size <= MAX_DEPLOY_BYTES, (
        f"{size / 1e6:.1f} MB exceeds the {MAX_DEPLOY_BYTES / 1e6:.0f} MB budget the "
        "player loads on tab open; lower build_evolution_artifact(max_frames=...) "
        "rather than committing the larger payload"
    )


def test_frame_budget_is_self_consistent(artifact):
    d = artifact["data"]
    b = d["budget"]
    days = d["frame_days"]
    assert b["days"] == d["days"] == 365
    assert days[0] == 0 and days[-1] == 365
    assert days == sorted(set(days)), "frame days must be strictly increasing"
    assert len(days) <= b["max_frames"]
    assert b["frames_per_arm"] == len(days)
    for arm in ("policy", "mtd"):
        a = d["arms"][arm]
        assert a["frame_days"] == days, "both arms must share one frame schedule"
        assert len(a["frames_vti_b64"]) == len(days)


def test_dense_window_covers_the_switch(artifact):
    """The adaptive switch happens in the first weeks; frames must be dense
    there or the transition the whole tab exists to show is under-sampled."""
    d = artifact["data"]
    dense_until = d["budget"]["dense_until"]
    dense = [day for day in d["frame_days"] if day <= dense_until]
    gaps = [b - a for a, b in zip(dense, dense[1:])]
    assert len(dense) >= 10, dense
    assert all(g <= 5 for g in gaps), (dense, gaps)


def test_every_frame_is_a_standalone_vti_on_the_declared_grid(artifact, tmp_path):
    from danon.schedule import read_fields_vti

    d = artifact["data"]
    dims = tuple(int(v) for v in d["grid"])
    n_points = dims[0] * dims[1] * dims[2]
    for arm in ("policy", "mtd"):
        raw = base64.b64decode(d["arms"][arm]["frames_vti_b64"][-1])
        assert raw[:4] == b"<?xm", "frame is not an XML/VTK document"
        p = tmp_path / f"{arm}_horizon.vti"
        p.write_bytes(raw)
        vti = read_fields_vti(str(p))
        assert tuple(vti.dims) == dims
        assert vti.spacing_mm == pytest.approx(d["spacing_mm"])
        assert {"sensitive", "resistant"} <= set(vti.fields)
        assert int(vti.mask.size) == n_points
        assert int(vti.mask.sum()) > 0, "a frame with no tumour voxels is not a state"


def test_static_maps_are_carried_once_and_match_the_grid(artifact):
    """Geometry constants ship once per payload; per-frame copies are exactly
    the bloat that made the first version of this artifact 58 MB."""
    d = artifact["data"]
    n_points = d["grid"][0] * d["grid"][1] * d["grid"][2]
    assert {"tissue_class", "diffusion_mm2_per_day", "proliferative", "vessels"} <= set(
        d["static_maps"]
    )
    for name, b64 in d["static_maps"].items():
        assert len(base64.b64decode(b64)) == n_points * 4, name
