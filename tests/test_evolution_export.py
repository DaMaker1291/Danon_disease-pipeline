"""Tests for the per-day evolution capture and export chain.

Pins the things an evolution viewer must never get wrong:

- field_hook identity: run(..., field_hook=recorder) is bit-identical to
  the same run without the hook (recording cannot perturb physics);
- frame/trajectory consistency: snapshot day d aggregates equal the run's
  own burden arrays (frame 0 = the seed, frame days = post-update state,
  the horizon frame checked against the hook's final aggregate);
- corruption detection: a frame with perturbed fields must be rejected by
  FrameSet.validate_against_trajectory;
- .pvd validity: monotone times, sorted file list, well-formed XML;
- static geometry maps: byte-identical to the include_tissue=True frame's
  own derived arrays;
- paired payload: shared frame days, verified resistant fractions, and the
  pvd series writing real files that the reader can round-trip.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from danon.evolution_export import (
    FrameSet,
    capture_frames,
    export_paired_evolution,
    resistant_fraction_gated,
    snapshot_days,
    static_geometry_maps,
    write_pvd,
    write_pvd_series,
)
from danon.evosteer import MTDPolicy, TumorEvolutionSimulator, make_gbm_phantom


@pytest.fixture(scope="module")
def phantom():
    return make_gbm_phantom(size=14, seed=3)


# ---------------------------------------------------------------------------
# snapshot schedule
# ---------------------------------------------------------------------------


def test_snapshot_days_shape():
    days = snapshot_days(365)
    assert days[0] == 0 and days[-1] == 365
    assert days == sorted(set(days)), "days must be strictly increasing"
    # Dense early: every day 0..dense window present.
    assert days[:11] == list(range(11))
    # Budget respected.
    assert len(days) <= 110
    # Late frames step by at most 7.
    gaps = np.diff(days)
    assert gaps.max() <= 7


def test_snapshot_days_short_horizons():
    for horizon in (2, 5, 30):
        days = snapshot_days(horizon)
        assert days[0] == 0 and days[-1] == horizon
        assert days == sorted(set(days))


def test_snapshot_days_budget_thins():
    days = snapshot_days(365, dense_until=60, max_frames=40)
    assert len(days) <= 40
    assert days[0] == 0 and days[-1] == 365


# ---------------------------------------------------------------------------
# field_hook identity (the engine-level guarantee)
# ---------------------------------------------------------------------------


def test_field_hook_identity(phantom):
    sim = TumorEvolutionSimulator(phantom)
    plain = sim.run(MTDPolicy(), days=25, seed=101)
    recorded = []
    with_hook = sim.run(
        MTDPolicy(), days=25, seed=101,
        field_hook=lambda d, S, R, cA, cB: recorded.append((d, S.copy(), R.copy())),
    )
    assert np.array_equal(plain.burden_S, with_hook.burden_S)
    assert np.array_equal(plain.burden_R, with_hook.burden_R)
    assert plain.pfs_day == with_hook.pfs_day
    assert plain.toxicity == with_hook.toxicity
    assert [d for d, _s, _r in recorded] == list(range(25))


# ---------------------------------------------------------------------------
# capture + validation
# ---------------------------------------------------------------------------


def test_capture_frames_match_trajectory(phantom):
    fs, traj = capture_frames(phantom, r0=0.01, seed=101, days=30, arm="mtd")
    # Frame 0 is the seeded state: equals burden_[0].
    S0, R0, cA0, cB0 = fs.fields[0]
    assert abs(float(S0.sum() + R0.sum()) - float(traj.burden_S[0] + traj.burden_R[0])) < 1e-9
    # Drug fields start at zero (pre-dose).
    assert float(cA0.max()) == 0.0 and float(cB0.max()) == 0.0
    # Frame 5 is the state after 5 updates: equals burden_[5].
    S5, R5, _cA5, _cB5 = fs.fields[5]
    assert abs(float(S5.sum()) - float(traj.burden_S[5])) < 1e-6 * max(1.0, float(traj.burden_S[5]))
    # Frame days land on the snapshot schedule.
    from danon.evolution_export import snapshot_days as _sd
    assert fs.days == _sd(30)
    # Validation passes on the untouched capture.
    fs.validate_against_trajectory(traj)


def test_capture_frames_detect_perturbation(phantom):
    fs, traj = capture_frames(phantom, r0=0.01, seed=101, days=20, arm="mtd")
    # Corrupt one frame's fields: the validator must refuse.
    d = fs.days[len(fs.days) // 2]
    S, R, cA, cB = fs.fields[d]
    fs.fields[d] = (S * 1.5, R, cA, cB)
    with pytest.raises(ValueError, match="frame burden"):
        fs.validate_against_trajectory(traj)


def test_capture_frames_rejects_bad_args(phantom):
    with pytest.raises(ValueError, match="arm"):
        capture_frames(phantom, r0=0.01, seed=0, days=10, arm="bogus")
    with pytest.raises(ValueError, match="r0"):
        capture_frames(phantom, r0=1.5, seed=0, days=10, arm="mtd")


# ---------------------------------------------------------------------------
# .pvd collection
# ---------------------------------------------------------------------------


def test_write_pvd_wellformed(tmp_path):
    vtis = [str(tmp_path / f"f{i}.vti") for i in range(4)]
    times = [0.0, 2.0, 4.0, 7.0]
    pvd = write_pvd(str(tmp_path / "evo.pvd"), vtis, times)
    tree = ET.parse(pvd)
    root = tree.getroot()
    assert root.tag == "VTKFile"
    sets = root.findall("./Collection/DataSet")
    assert len(sets) == 4
    got_times = [float(s.get("timestep")) for s in sets]
    assert got_times == sorted(times) == times
    for s, v in zip(sets, vtis):
        assert s.get("file").endswith("f.vti") or s.get("file") in [
            "f0.vti", "f1.vti", "f2.vti", "f3.vti"
        ]


def test_write_pvd_rejects_unsorted(tmp_path):
    with pytest.raises(ValueError, match="monotone"):
        write_pvd(str(tmp_path / "e.pvd"), ["a", "b"], [3.0, 1.0])
    with pytest.raises(ValueError, match="equal length"):
        write_pvd(str(tmp_path / "e.pvd"), ["a"], [1.0, 2.0])


# ---------------------------------------------------------------------------
# static geometry maps
# ---------------------------------------------------------------------------


def test_static_maps_match_frame_arrays(phantom):
    """The once-per-payload maps are byte-identical to what an
    include_tissue=True frame carries."""
    import base64

    from danon.imaging import from_nifti_mask  # noqa: F401  (parity guard)

    from danon.evolution_export import export_evolution_frames

    exp = export_evolution_frames(
        phantom, r0=0.01, seed=101, days=12, arm="mtd", dense_until=4,
        include_tissue=True,
    )
    raw = base64.b64decode(exp.frames_vti_b64[0])
    # Extract the tissue_class array from the frame VTI XML.
    import re

    text = raw.decode("ascii")
    m = re.search(r'Name="tissue_class"[^>]*>([^<]+)<', text)
    frame_tissue = np.frombuffer(base64.b64decode(m.group(1)), dtype="<f4")
    statics = static_geometry_maps(phantom)
    payload_tissue = np.frombuffer(
        base64.b64decode(statics["tissue_class"]), dtype="<f4"
    )
    assert np.array_equal(frame_tissue, payload_tissue)


# ---------------------------------------------------------------------------
# paired payload
# ---------------------------------------------------------------------------


def test_export_paired_evolution_payload(phantom):
    payload = export_paired_evolution(
        phantom, r0=0.01, seed=101, days=25, dense_until=8
    )
    assert payload["seed"] == 101 and payload["days"] == 25
    assert payload["arms"]["policy"]["frame_days"] == payload["frame_days"]
    assert payload["arms"]["mtd"]["frame_days"] == payload["frame_days"]
    n = len(payload["frame_days"])
    assert len(payload["arms"]["policy"]["frames_vti_b64"]) == n
    assert len(payload["arms"]["mtd"]["frames_vti_b64"]) == n
    assert "static_maps" in payload
    # Daily curves: full length, and the resistant fraction matches them.
    for arm in ("policy", "mtd"):
        a = payload["arms"][arm]
        assert len(a["burden_S"]) == 25 and len(a["dose_A"]) == 25
        rf = payload["resistant_fraction"][arm]
        assert len(rf) == 25
        for d in (0, 7, 24):
            s, r = a["burden_S"][d], a["burden_R"][d]
            # GATED against the engine's detection floor, not a raw ratio:
            # below ~1e8 cells R/(S+R) is 1.0 for a tumour that is gone, and
            # the chart must not report that as resistance.
            assert rf[d] == pytest.approx(resistant_fraction_gated(s, r))


def test_export_paired_evolution_frames_are_valid_vti(phantom):
    """Each frame decodes through the package's own VTI reader."""
    import base64
    import tempfile
    import os

    from danon.schedule import read_fields_vti

    payload = export_paired_evolution(
        phantom, r0=0.01, seed=101, days=15, dense_until=5
    )
    b64 = payload["arms"]["mtd"]["frames_vti_b64"][-1]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "frame.vti")
        with open(p, "wb") as fh:
            fh.write(base64.b64decode(b64))
        vti = read_fields_vti(p)
        assert set(("sensitive", "resistant", "drug_A", "drug_B")) <= set(vti.fields)
        assert vti.mask.sum() > 0


def test_write_pvd_series_on_disk(phantom, tmp_path):
    pvd = write_pvd_series(
        str(tmp_path / "series"), phantom, r0=0.01, seed=101, days=12,
        arm="mtd", dense_until=4,
    )
    assert pvd.endswith("evolution.pvd")
    tree = ET.parse(pvd)
    sets = tree.getroot().findall("./Collection/DataSet")
    assert len(sets) == len(snapshot_days(12, dense_until=4))
    # Every referenced file exists on disk.
    import os

    root_dir = os.path.dirname(pvd)
    for s in sets:
        assert os.path.exists(os.path.join(root_dir, s.get("file")))
