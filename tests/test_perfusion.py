"""Tests for the DICOM reader and the DSC -> relative-CBV chain.

Pins the things this path must never get wrong:

- reading a real DICOM encoding, both uncompressed and encapsulated JPEG 2000,
  and refusing a transfer syntax it cannot decode rather than mis-decoding it;
- the first-pass window being chosen from the volume-mean curve *after* the
  head-to-tail drift line is removed -- the failure that made the first real
  series call frame 0 the bolus arrival and integrate 45 of its 50 frames;
- recovering a CBV ratio the pipeline was handed (3:1 planted), and refusing a
  series with no bolus, no TR/TE, one slice, or unequal time courses;
- the real downloaded series, when it is cached: the checks that need the actual
  archive are skipped rather than restated when the data is not on disk.
"""

from __future__ import annotations

import io
import os
import struct

import numpy as np
import pytest

from danon.dicomio import (
    DicomError,
    read_series,
    read_slice,
    series_slice_paths,
)
from danon.perfusion import (
    DSC_MIN_TIMEPOINTS,
    DscSeries,
    PerfusionError,
    build_dsc_series,
    planted_cbv_recovery,
    relative_cbv,
)

REAL_SERIES = os.path.join("data", "tcia", "perfusion", "UPENN-GBM-00576")

_LONG_VRS = {"OB", "OW", "OF", "SQ", "UT", "UN", "OD", "OL"}


# ---------------------------------------------------------------------------
# A minimal writer, so the reader is tested on bytes rather than on faith
# ---------------------------------------------------------------------------


def _element(group: int, element: int, vr: str, value: bytes) -> bytes:
    head = struct.pack("<HH", group, element) + vr.encode("ascii")
    if vr in _LONG_VRS:
        return head + b"\x00\x00" + struct.pack("<I", len(value)) + value
    if len(value) % 2:
        value = value + (b"\x00" if vr != "UI" else b"\x00")
    return head + struct.pack("<H", len(value)) + value


def _text(vr: str, text: str) -> bytes:
    raw = text.encode("ascii")
    if len(raw) % 2:
        raw += b"\x00" if vr == "UI" else b" "
    return raw


def _us(value: int) -> bytes:
    return struct.pack("<H", value)


def _write_dicom(
    path: str,
    *,
    z: float,
    instance: int = 1,
    acquisition_time: str = "120000",
    rows: int = 4,
    cols: int = 4,
    tr_ms: str = "1830",
    te_ms: str = "51.4",
    transfer_syntax: str = "1.2.840.10008.1.2.1",
    frames: np.ndarray | None = None,
    encapsulated: bool = False,
    with_sequence: bool = False,
) -> None:
    """Write a small explicit-VR-LE DICOM file the reader must accept."""
    meta = (
        _element(0x0002, 0x0001, "OB", b"\x00\x01")
        + _element(0x0002, 0x0002, "UI", _text("UI", "1.2.840.10008.5.1.4.1.1.4"))
        + _element(0x0002, 0x0003, "UI", _text("UI", f"1.2.826.0.1.3680043.8.498.{instance}"))
        + _element(0x0002, 0x0010, "UI", _text("UI", transfer_syntax))
    )
    body = (
        _element(0x0008, 0x0060, "CS", _text("CS", "MR"))
        + _element(0x0008, 0x0032, "TM", _text("TM", acquisition_time))
        + _element(0x0008, 0x103E, "LO", _text("LO", "PERFUSION GRE"))
        + _element(0x0018, 0x0080, "DS", _text("DS", tr_ms))
        + _element(0x0018, 0x0081, "DS", _text("DS", te_ms))
        + _element(0x0020, 0x0013, "IS", _text("IS", str(instance)))
        + _element(0x0020, 0x0032, "DS", _text("DS", f"0\\0\\{z}"))
        + _element(0x0028, 0x0002, "US", _us(1))
        + _element(0x0028, 0x0010, "US", _us(rows))
        + _element(0x0028, 0x0011, "US", _us(cols))
        + _element(0x0028, 0x0030, "DS", _text("DS", "1.796875\\1.796875"))
        + _element(0x0028, 0x0100, "US", _us(16))
        + _element(0x0028, 0x0103, "US", _us(0))
    )
    if with_sequence:
        # An undefined-length sequence before the pixel data: the reader must
        # step over it exactly, which is what UPENN-GBM's headers require.
        item = struct.pack("<HHI", 0xFFFE, 0xE000, 0xFFFFFFFF)
        item += _element(0x0008, 0x0100, "SH", _text("SH", "ORIGINAL"))
        item += struct.pack("<HHI", 0xFFFE, 0xE00D, 0)
        body += _element(0x0008, 0x1032, "SQ", b"")[0:8] + b"\xff\xff\xff\xff"
        body += item + struct.pack("<HHI", 0xFFFE, 0xE0DD, 0)

    if frames is None:
        frames = np.arange(rows * cols, dtype=np.uint16).reshape(1, rows, cols)
    if encapsulated:
        from PIL import Image

        fragments = b""
        for frame in frames:
            buffer = io.BytesIO()
            Image.fromarray(np.asarray(frame, dtype=np.uint16)).save(
                buffer, format="JPEG2000", irreversible=False
            )
            code = buffer.getvalue()
            if len(code) % 2:
                code += b"\x00"
            fragments += struct.pack("<HHI", 0xFFFE, 0xE000, len(code)) + code
        pixel = (
            struct.pack("<HH", 0x7FE0, 0x0010)
            + b"OB\x00\x00"
            + b"\xff\xff\xff\xff"
            + struct.pack("<HHI", 0xFFFE, 0xE000, 0)  # empty Basic Offset Table
            + fragments
            + struct.pack("<HHI", 0xFFFE, 0xE0DD, 0)
        )
    else:
        payload = np.asarray(frames, dtype="<u2").tobytes()
        pixel = (
            struct.pack("<HH", 0x7FE0, 0x0010)
            + b"OW\x00\x00"
            + struct.pack("<I", len(payload))
            + payload
        )

    meta_len = len(meta)
    meta = _element(0x0002, 0x0000, "UL", struct.pack("<I", meta_len)) + meta
    with open(path, "wb") as fh:
        fh.write(b"\x00" * 128 + b"DICM" + meta + body + pixel)


def _series_dir(tmp_path, n_slices: int = 3, n_timepoints: int = 3, **kw):
    """A tiny DSC series: signal constant per timepoint, geometry as written."""
    directory = tmp_path / "series"
    directory.mkdir(exist_ok=True)
    for s in range(n_slices):
        for t in range(n_timepoints):
            frames = np.full((1, 4, 4), 1000 + 10 * t, dtype=np.uint16)
            _write_dicom(
                str(directory / f"{s:03d}_{t:03d}.dcm"),
                z=float(s) * 6.5,
                instance=t,
                acquisition_time=f"1200{t:02d}",
                frames=frames,
                **kw,
            )
    return str(directory)


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------


class TestDicomReader:
    def test_uncompressed_round_trip(self, tmp_path):
        path = str(tmp_path / "one.dcm")
        frames = np.arange(16, dtype=np.uint16).reshape(1, 4, 4)
        _write_dicom(path, z=-1.25, instance=7, frames=frames)
        s = read_slice(path)
        assert s.transfer_syntax == "1.2.840.10008.1.2.1"
        assert s.z_position == pytest.approx(-1.25)
        assert s.pixel_spacing == pytest.approx((1.796875, 1.796875))
        assert s.tags["tr_ms"] == "1830" and s.tags["te_ms"] == "51.4"
        assert s.frames.shape == (1, 4, 4)
        assert np.array_equal(s.frames[0], frames[0].astype(np.float32))

    def test_encapsulated_jpeg2000_round_trip(self, tmp_path):
        path = str(tmp_path / "j2k.dcm")
        frames = (np.arange(16, dtype=np.uint16).reshape(1, 4, 4) * 97) % 4000
        _write_dicom(
            path,
            z=0.0,
            frames=frames,
            encapsulated=True,
            transfer_syntax="1.2.840.10008.1.2.4.90",
        )
        s = read_slice(path)
        assert s.transfer_syntax == "1.2.840.10008.1.2.4.90"
        assert s.tags["n_fragments"] == 1
        # JPEG 2000 LOSSLESS: the samples must come back exactly.
        assert np.array_equal(s.frames[0], frames[0].astype(np.float32))

    def test_undefined_length_sequence_is_stepped_over(self, tmp_path):
        path = str(tmp_path / "seq.dcm")
        _write_dicom(path, z=3.5, with_sequence=True)
        s = read_slice(path)
        assert s.frames is not None and s.frames.shape == (1, 4, 4)

    def test_unsupported_transfer_syntax_is_refused_by_name(self, tmp_path):
        path = str(tmp_path / "jpeg.dcm")
        _write_dicom(path, z=0.0, transfer_syntax="1.2.840.10008.1.2.4.50")
        with pytest.raises(DicomError) as err:
            read_slice(path)
        assert "1.2.840.10008.1.2.4.50" in str(err.value)

    def test_missing_magic_is_refused(self, tmp_path):
        path = tmp_path / "not.dcm"
        path.write_bytes(b"not a dicom file" * 10)
        with pytest.raises(DicomError) as err:
            read_slice(str(path))
        assert "DICM" in str(err.value)

    def test_non_dicom_directory_is_refused(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(DicomError):
            series_slice_paths(str(tmp_path / "empty"))

    def test_read_series_reads_every_file(self, tmp_path):
        directory = _series_dir(tmp_path, n_slices=2, n_timepoints=2)
        slices = read_series(directory)
        assert len(slices) == 4
        assert {round(s.z_position, 2) for s in slices} == {0.0, 6.5}


# ---------------------------------------------------------------------------
# The DSC -> rCBV chain
# ---------------------------------------------------------------------------


class TestPlantedRecovery:
    def test_recovers_a_planted_ratio_exactly_without_noise(self):
        out = planted_cbv_recovery(true_ratio=3.0, noise=0.0)
        assert out["recovered_ratio"] == pytest.approx(3.0, rel=1e-6)
        assert out["window_frames"] >= 5

    def test_recovers_a_planted_ratio_under_signal_noise(self):
        out = planted_cbv_recovery(true_ratio=3.0, noise=0.002, seed=1)
        assert out["relative_error"] < 0.05

    def test_a_different_planted_ratio_is_also_recovered(self):
        out = planted_cbv_recovery(true_ratio=1.6, noise=0.002, seed=2)
        assert out["recovered_ratio"] == pytest.approx(1.6, rel=0.05)


class TestSeriesAssembly:
    def test_reads_a_synthetic_series_into_time_major_order(self, tmp_path):
        directory = _series_dir(tmp_path, n_slices=3, n_timepoints=21)
        s = build_dsc_series(directory)
        assert s.shape == (21, 3, 4, 4)  # [t, z, y, x]
        assert s.tr_s == pytest.approx(1.83)
        assert s.te_s == pytest.approx(0.0514)
        # Later timepoints carry a higher signal (1000 + 10t) in this phantom.
        assert s.signal[-1].mean() > s.signal[0].mean()

    def test_partial_series_is_refused_not_zero_filled(self, tmp_path):
        directory = _series_dir(tmp_path, n_slices=2, n_timepoints=3)
        # Delete one file: the slice positions now hold unequal frame counts.
        victims = sorted(n for n in os.listdir(directory) if n.endswith(".dcm"))
        os.remove(os.path.join(directory, victims[-1]))
        with pytest.raises(PerfusionError) as err:
            build_dsc_series(directory)
        assert "partial" in str(err.value)

    def test_single_slice_position_is_refused(self, tmp_path):
        directory = _series_dir(tmp_path, n_slices=1, n_timepoints=25)
        with pytest.raises(PerfusionError) as err:
            build_dsc_series(directory)
        assert "slice position" in str(err.value)

    def test_too_few_frames_is_refused(self, tmp_path):
        directory = _series_dir(tmp_path, n_slices=2, n_timepoints=3)
        with pytest.raises(PerfusionError) as err:
            build_dsc_series(directory)
        assert str(DSC_MIN_TIMEPOINTS) in str(err.value)

    def test_missing_te_is_refused(self, tmp_path):
        directory = _series_dir(tmp_path, n_slices=2, n_timepoints=25, te_ms="")

        with pytest.raises(PerfusionError) as err:
            build_dsc_series(directory)
        assert "TE" in str(err.value)


class TestCbvIntegration:
    def _series(self, curve: np.ndarray, n_y: int = 4, n_x: int = 4) -> DscSeries:
        signal = np.broadcast_to(curve[:, None, None, None], (curve.size, 1, n_y, n_x))
        return DscSeries(
            signal=np.array(signal, dtype=np.float32),
            tr_s=1.83,
            te_s=0.0514,
            pixel_mm=(2.0, 2.0),
            slice_mm=None,
            source="test",
        )

    def test_bolus_is_found_where_it_is(self):
        n = 40
        curve = np.full(n, 1000.0)
        curve[12:20] -= 200.0  # a bolus: signal drop -> positive dR2*
        r = relative_cbv(self._series(curve), baseline_frames=5, tail_frames=5)
        assert 10 <= r.peak_frame <= 22
        assert r.arrival_frame <= r.peak_frame <= r.end_frame
        assert r.window_frames < n / 2  # not integrating the whole acquisition

    def test_a_flat_series_has_no_bolus_and_refuses(self):
        r = self._series(np.full(40, 1000.0))
        with pytest.raises(PerfusionError) as err:
            relative_cbv(r)
        assert "no bolus" in str(err.value)

    def test_drift_is_removed_rather_than_integrated(self):
        """A drifting baseline with a bolus on top must not become one big bolus.

        This is the defect the first real series exposed: a monotone drift made
        the volume-mean curve call frame 0 the arrival and integrate 45 of 50
        frames. The planted drift here carries a real bolus at frames 15-22.
        """
        n = 50
        drift = np.linspace(0.0, -0.35, n)  # signal drifts down 35%
        bolus = np.zeros(n)
        bolus[15:23] = -0.25
        signal = 1000.0 * (1.0 + drift + bolus)
        r = relative_cbv(self._series(signal), baseline_frames=5, tail_frames=5)
        assert 13 <= r.peak_frame <= 24
        assert r.arrival_frame >= 8  # NOT frame 0
        assert r.window_frames <= 20  # NOT most of the acquisition
        assert r.drift_peak_ratio < 1.0

    def test_reference_tissue_normalises_to_one(self):
        n = 40
        curve = np.full(n, 1000.0)
        curve[12:20] -= 200.0
        r = relative_cbv(self._series(curve), baseline_frames=5, tail_frames=5)
        assert float(np.median(r.rcbv[r.cbv_raw > 0])) == pytest.approx(1.0, rel=1e-6)
        assert r.reference_voxels > 0

    def test_negative_dR2_is_clipped_not_signed(self):
        n = 40
        curve = np.full(n, 1000.0)
        curve[12:20] -= 200.0
        r = relative_cbv(self._series(curve), baseline_frames=5, tail_frames=5)
        assert r.rcbv.min() >= 0.0
        assert np.all(np.isfinite(r.rcbv))


# ---------------------------------------------------------------------------
# The real archive, when it is cached
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.isdir(REAL_SERIES),
    reason="real UPENN-GBM DSC series not cached (see _perfusion_cohort.py)",
)
class TestRealSeries:
    def test_real_series_geometry_and_bolus(self):
        s = build_dsc_series(REAL_SERIES)
        assert s.shape == (50, 14, 128, 128)
        assert s.tr_s == pytest.approx(1.83)
        assert s.te_s == pytest.approx(0.0514)
        r = relative_cbv(s)
        assert r.arrival_frame < r.peak_frame < r.end_frame
        assert r.window_frames < s.n_timepoints / 2
        # Half of this series' raw volume-mean peak is drift, which is why the
        # drift line exists; pin that it is removed rather than integrated.
        assert 0.2 < r.drift_peak_ratio < 0.8
        assert r.stats()["contrast_p95_over_median"] > 2.0
