"""Real DSC perfusion -> relative CBV, and the refusals that keep it honest.

WHAT THIS REPLACES
------------------
Stream 3 previously reached D_chemo(x) through a *proxy*: contrast enhancement on
a t1ce volume, which is vascular but not perfusion. The archive serves the
measured form -- raw dynamic-susceptibility-contrast series -- and
`danon.dicomio` can now read them, so this module turns one into a relative CBV
map by the textbook route:

    S(t)  --(baseline)-->  dR2*(t) = -ln(S(t)/S0)/TE
          --(first pass)-->  CBV_raw = sum over the bolus of dR2*(t) * TR
          --(reference)-->   rCBV = CBV_raw / median(over reference tissue)

WHAT IS MEASURED, AND WHAT IS STILL ASSUMED
-------------------------------------------
Measured: the whole time course, per voxel, from the patient's own acquisition.
Assumed-and-labelled: the baseline window (``DSC_BASELINE_FRAMES``), where the
first pass is judged to start and end (``DSC_ARRIVAL_FRACTION``,
``DSC_FIRST_PASS_FRACTION``), and the reference tissue (voxels whose baseline
signal is above ``DSC_REFERENCE_PERCENTILE`` of the volume). Those are declared
constants, returned inside every result, because no AIF deconvolution happens
here: this is relative CBV by first-pass integration, not absolute perfusion and
not Ktrans. A quantitative Ktrans map needs an arterial input function and a
pharmacokinetic model, neither of which is invented here.

THE ONE THING THAT WOULD MAKE THIS A LIE
----------------------------------------
Integration windows chosen *after* looking at the map. The window comes from the
volume-mean curve, and the recovered ratio is checked against a planted phantom
with a known CBV ratio (`planted_cbv_recovery`, pinned by test at the measured
tolerance) -- a pipeline that cannot recover a known 3:1 ratio has no business
reporting an unknown one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "DSC_BASELINE_FRAMES",
    "DSC_ARRIVAL_FRACTION",
    "DSC_FIRST_PASS_FRACTION",
    "DSC_REFERENCE_PERCENTILE",
    "DSC_MIN_TIMEPOINTS",
    "PerfusionError",
    "DscSeries",
    "CbvResult",
    "build_dsc_series",
    "relative_cbv",
    "planted_cbv_recovery",
]

#: Frames averaged for the pre-bolus baseline S0. DECLARED, not measured: a
#: real AIF-based analysis selects the baseline per voxel and fits the bolus.
DSC_BASELINE_FRAMES = 5

#: Frames averaged at the END of the series to estimate the post-bolus level.
#: A DSC acquisition drifts: on the first real series this module read, the
#: volume-mean dR2*(t) sat at 0.37 at frame 0 and never returned to zero, so a
#: criterion that assumed a flat pre-bolus plateau called frame 0 the bolus
#: arrival and integrated 45 of 50 frames -- the drift itself, not the bolus.
#: The fix is the textbook one: subtract the straight line between the pre- and
#: post-bolus levels, and integrate what is left above it.
DSC_TAIL_FRAMES = 5

#: The first pass is judged to start at the last frame before the volume-mean
#: peak whose value is at or below this fraction of the peak.
DSC_ARRIVAL_FRACTION = 0.10

#: ...and to end at the first frame after the peak that falls back to this
#: fraction of it. Recirculation makes the choice convention-dependent; it is
#: declared so the number cannot hide behind its own criterion.
DSC_FIRST_PASS_FRACTION = 0.30

#: Reference tissue: voxels whose baseline signal exceeds this percentile of the
#: volume, i.e. tissue that is receiving blood, used as the rCBV = 1 reference.
DSC_REFERENCE_PERCENTILE = 70.0

#: Fewer frames than this and a first pass cannot be separated from noise.
DSC_MIN_TIMEPOINTS = 20


class PerfusionError(RuntimeError):
    """A series this module will not turn into a perfusion map, and why."""


@dataclass(frozen=True)
class DscSeries:
    """A 4D DSC acquisition: signal[t, z, y, x] plus the units it is in."""

    signal: np.ndarray
    tr_s: float
    te_s: float
    pixel_mm: Tuple[float, float]
    slice_mm: Optional[float]
    source: str
    n_files: int = 0
    warnings: Tuple[str, ...] = ()

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return tuple(int(v) for v in self.signal.shape)  # type: ignore[return-value]

    @property
    def n_timepoints(self) -> int:
        return int(self.signal.shape[0])

    @property
    def n_slices(self) -> int:
        return int(self.signal.shape[1])

    def provenance(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "n_files": self.n_files,
            "shape_tzyx": list(self.shape),
            "tr_s": self.tr_s,
            "te_s": self.te_s,
            "pixel_mm": list(self.pixel_mm),
            "slice_mm": self.slice_mm,
            "signal_units": "DICOM stored values (arbitrary, not rescaled to HU)",
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True)
class CbvResult:
    """A relative-CBV map with the window and constants that produced it."""

    rcbv: np.ndarray  # [z, y, x], median-of-reference-tissue == 1
    cbv_raw: np.ndarray
    volume_curve: np.ndarray  # volume-mean dR2*(t) after the drift line is removed
    peak_frame: int
    arrival_frame: int
    end_frame: int
    baseline_frames: int
    tail_frames: int
    tr_s: float
    te_s: float
    reference_voxels: int
    drift_peak_ratio: float = 0.0
    provenance: Dict[str, object] = field(default_factory=dict)

    @property
    def window_frames(self) -> int:
        return int(self.end_frame - self.arrival_frame + 1)

    def stats(self) -> Dict[str, object]:
        """Distribution statistics of the map, on voxels with a real curve."""
        r = self.rcbv
        perfused = r[r > 0]
        return {
            "voxels": int(r.size),
            "voxels_positive": int(perfused.size),
            "median": float(np.median(perfused)) if perfused.size else 0.0,
            "p95": float(np.percentile(perfused, 95.0)) if perfused.size else 0.0,
            "max": float(perfused.max()) if perfused.size else 0.0,
            # The number a "tumour is hyperperfused" claim would rest on.
            "contrast_p95_over_median": (
                float(np.percentile(perfused, 95.0) / np.median(perfused))
                if perfused.size and np.median(perfused) > 0
                else None
            ),
            "fraction_above_1_5x": (
                float((perfused > 1.5).mean()) if perfused.size else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Series assembly
# ---------------------------------------------------------------------------


def _series_files(path: str) -> List[str]:
    from danon.dicomio import series_slice_paths

    return series_slice_paths(path)


def build_dsc_series(path: str) -> DscSeries:
    """Read a downloaded DSC series directory into signal[t, z, y, x].

    Refuses, by name, when the acquisition cannot support a first pass:
    mixed row/column counts, missing TR or TE, a single slice position, fewer
    frames than `DSC_MIN_TIMEPOINTS`, or slice positions that do not partition
    the files into equal-length time courses (a partial download, which would
    otherwise produce a time course with a hole in it).
    """
    from danon.dicomio import DicomError, read_slice

    files = _series_files(path)
    slices = []
    for f in files:
        try:
            slices.append(read_slice(f))
        except DicomError as exc:
            raise PerfusionError(f"unreadable DICOM slice: {exc}") from exc

    tr_values = {s.tags.get("tr_ms") for s in slices}
    te_values = {s.tags.get("te_ms") for s in slices}
    if len(tr_values) != 1 or None in tr_values:
        raise PerfusionError(
            f"series declares TR values {sorted(map(str, tr_values))}: a time "
            f"course needs one"
        )
    if len(te_values) != 1 or None in te_values:
        raise PerfusionError(
            f"series declares TE values {sorted(map(str, te_values))}: dR2* "
            f"needs one TE to convert signal to concentration"
        )
    try:
        tr_s = float(str(next(iter(tr_values)))) / 1000.0
        te_s = float(str(next(iter(te_values)))) / 1000.0
    except ValueError as exc:
        raise PerfusionError(f"TR/TE are not numeric: {exc}") from exc
    if tr_s <= 0.0 or te_s <= 0.0:
        raise PerfusionError(f"TR={tr_s}s and TE={te_s}s must both be positive")

    spacings = {s.pixel_spacing for s in slices}
    if len(spacings) != 1:
        raise PerfusionError(
            f"series mixes pixel spacings {sorted(map(str, spacings))}"
        )
    pixel_mm = next(iter(spacings))
    if pixel_mm is None:
        raise PerfusionError("series declares no (0028,0030) PixelSpacing")

    shapes = {None if s.frames is None else s.frames.shape[1:] for s in slices}
    if len(shapes) != 1 or None in shapes:
        raise PerfusionError(f"series mixes in-plane shapes {sorted(map(str, shapes))}")

    groups: Dict[float, List] = {}
    for s in slices:
        z = s.z_position
        if z is None:
            raise PerfusionError(
                f"{os.path.basename(s.path)}: no (0020,0032) ImagePositionPatient, "
                f"so slices cannot be ordered"
            )
        groups.setdefault(round(z, 2), []).append(s)

    if len(groups) < 2:
        raise PerfusionError(
            f"series holds {len(groups)} slice position(s); a volumetric rCBV "
            f"map needs more than one"
        )

    counts = {len(v) for v in groups.values()}
    if len(counts) != 1:
        raise PerfusionError(
            f"slice positions hold unequal frame counts {sorted(counts)}; this "
            f"is a partial series and its time course would have a hole in it"
        )
    n_timepoints = next(iter(counts))
    if n_timepoints < DSC_MIN_TIMEPOINTS:
        raise PerfusionError(
            f"{n_timepoints} frames per slice is below the {DSC_MIN_TIMEPOINTS} "
            f"needed to separate a first pass from noise"
        )

    warnings: List[str] = []
    per_slice_frames: List[np.ndarray] = []
    for z in sorted(groups):
        ordered = sorted(groups[z], key=lambda s: s.time_key)
        stack = np.stack([np.asarray(s.frames, dtype=np.float32) for s in ordered])
        if stack.shape[0] != n_timepoints:
            raise PerfusionError(
                f"slice at z={z} produced {stack.shape[0]} frames, expected "
                f"{n_timepoints}"
            )
        if stack.shape[1] != 1:
            warnings.append(
                f"slice z={z} holds {stack.shape[1]} frames per file; they are "
                f"treated as one timepoint each"
            )
        per_slice_frames.append(stack.mean(axis=1))  # [t, y, x]

    signal = np.stack(per_slice_frames, axis=1)  # [t, z, y, x]
    zs = sorted(groups)
    diffs = np.diff(zs)
    slice_mm = float(np.median(diffs)) if diffs.size else None
    if diffs.size and not np.allclose(diffs, diffs[0], atol=1e-3):
        warnings.append(
            f"slice positions are not evenly spaced (gaps {np.round(diffs, 2).tolist()} mm); "
            f"rCBV is integrated in time, so this does not enter the map"
        )

    return DscSeries(
        signal=signal,
        tr_s=tr_s,
        te_s=te_s,
        pixel_mm=pixel_mm,  # type: ignore[arg-type]
        slice_mm=slice_mm,
        source=str(path),
        n_files=len(files),
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# The indicator-dilution step
# ---------------------------------------------------------------------------


def _dR2_star(signal: np.ndarray, baseline_frames: int, te_s: float) -> np.ndarray:
    """dR2*(t) = -ln(S(t)/S0)/TE, with S0 the mean of the first frames.

    Voxels whose baseline is non-positive are masked out (NaN): a logarithm of
    a zero baseline is not a concentration, it is a division by nothing.
    """
    baseline = signal[:baseline_frames].mean(axis=0)
    ok = baseline > 0.0
    ratio = np.full(signal.shape, np.nan, dtype=np.float64)
    ratio[:, ok] = signal[:, ok] / baseline[ok]
    np.clip(ratio, 1e-6, None, out=ratio)
    return -np.log(ratio) / te_s


def _linear_drift(level_start: np.ndarray, level_end: np.ndarray, n: int) -> np.ndarray:
    """The straight line from a start level to an end level, per voxel.

    Returned as [t, ...] with the same trailing shape as the levels. Signed:
    a series that drifts downward has a negative slope, and subtracting it is
    what keeps the drift out of the integrated area.
    """
    if n < 2:
        raise PerfusionError("a drift line needs at least two frames")
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float64)
    shape = (n,) + (1,) * np.ndim(level_start)
    return level_start + (level_end - level_start) * ramp.reshape(shape)


def relative_cbv(
    series: DscSeries,
    *,
    baseline_frames: int = DSC_BASELINE_FRAMES,
    tail_frames: int = DSC_TAIL_FRAMES,
    arrival_fraction: float = DSC_ARRIVAL_FRACTION,
    first_pass_fraction: float = DSC_FIRST_PASS_FRACTION,
    reference_percentile: float = DSC_REFERENCE_PERCENTILE,
) -> CbvResult:
    """First-pass integration of dR2*(t) into a relative-CBV map.

    The bolus is located on the volume-mean curve, not per voxel: the window is
    a property of the acquisition. A per-voxel window would let every voxel pick
    the integration limits that flatter it, which is the usual way a perfusion
    pipeline reports a ratio it did not measure. What IS per voxel is the linear
    baseline that each voxel's curve is measured above.
    """
    signal = np.asarray(series.signal, dtype=np.float64)
    if signal.ndim != 4:
        raise PerfusionError(f"signal must be [t, z, y, x], got {signal.shape}")
    if baseline_frames < 2:
        raise PerfusionError("baseline_frames must be at least 2")
    if signal.shape[0] <= baseline_frames + tail_frames + 2:
        raise PerfusionError(
            f"{signal.shape[0]} frames leaves no bolus between a "
            f"{baseline_frames}-frame head and a {tail_frames}-frame tail"
        )

    dR2 = _dR2_star(signal, baseline_frames, series.te_s)
    with np.errstate(invalid="ignore"):
        raw_curve = np.nanmean(dR2, axis=(1, 2, 3))
    if not np.all(np.isfinite(raw_curve)):
        raise PerfusionError(
            "volume-mean curve is not finite: every voxel has a non-positive "
            "baseline, so there is no signal to normalise against"
        )

    # Per-voxel baseline: the straight line from the head level to the tail
    # level. The bolus is what rises above it.
    head = dR2[:baseline_frames].mean(axis=0)
    tail = dR2[-tail_frames:].mean(axis=0)
    drift = _linear_drift(head, tail, signal.shape[0])
    residual = dR2 - drift
    with np.errstate(invalid="ignore"):
        volume_curve = np.nanmean(residual, axis=(1, 2, 3))
    raw_peak = float(np.nanmax(raw_curve)) if np.isfinite(raw_curve).any() else 0.0

    peak_frame = int(np.argmax(volume_curve))
    peak_value = float(volume_curve[peak_frame])
    if peak_value <= 0.0:
        raise PerfusionError(
            "no bolus separates from the drift: after the head-to-tail baseline "
            "is removed, the volume-mean dR2* never rises above zero"
        )

    before = np.where(volume_curve[: peak_frame + 1] <= arrival_fraction * peak_value)[0]
    arrival_frame = int(before[-1]) if before.size else 0
    after = np.where(volume_curve[peak_frame:] <= first_pass_fraction * peak_value)[0]
    end_frame = (
        int(peak_frame + after[0])
        if after.size
        else int(np.argmin(volume_curve[peak_frame:]) + peak_frame)
    )

    if end_frame <= arrival_frame:
        raise PerfusionError(
            f"first-pass window collapsed to frames [{arrival_frame}, {end_frame}]: "
            f"the bolus does not separate from the baseline"
        )

    window = residual[arrival_frame : end_frame + 1]
    cbv_raw = np.nansum(window, axis=0) * series.tr_s  # [z, y, x]
    cbv_raw = np.where(np.isfinite(cbv_raw), cbv_raw, 0.0)
    cbv_raw = np.clip(cbv_raw, 0.0, None)

    baseline = signal[:baseline_frames].mean(axis=0)
    positive = baseline[baseline > 0.0]
    if positive.size == 0:
        raise PerfusionError("no voxel has a positive baseline: cannot choose a reference")
    threshold = float(np.percentile(positive, reference_percentile))
    reference_voxels = (baseline >= threshold) & (cbv_raw > 0.0)
    if int(reference_voxels.sum()) == 0:
        raise PerfusionError(
            "the reference-tissue mask is empty (no voxel above the baseline "
            "percentile carries a bolus)"
        )
    reference = float(np.median(cbv_raw[reference_voxels]))
    if reference <= 0.0:
        raise PerfusionError("reference-tissue median CBV is zero: cannot normalise")

    rcbv = cbv_raw / reference
    return CbvResult(
        rcbv=rcbv,
        cbv_raw=cbv_raw,
        volume_curve=volume_curve,
        peak_frame=peak_frame,
        arrival_frame=arrival_frame,
        end_frame=end_frame,
        baseline_frames=baseline_frames,
        tail_frames=tail_frames,
        drift_peak_ratio=(
            float(peak_value / raw_peak) if raw_peak > 0.0 else 0.0
        ),
        tr_s=series.tr_s,
        te_s=series.te_s,
        reference_voxels=int(reference_voxels.sum()),
        provenance={
            "method": "first-pass integration of dR2*(t), no AIF deconvolution",
            "not_claimed": [
                "absolute CBV in ml/100g (needs an arterial input function)",
                "Ktrans or permeability (needs a pharmacokinetic model)",
                "a tumour segmentation (this map is not voxel-registered to one)",
            ],
            "declared_constants": {
                "baseline_frames": baseline_frames,
                "tail_frames": tail_frames,
                "arrival_fraction": arrival_fraction,
                "first_pass_fraction": first_pass_fraction,
                "reference_percentile": reference_percentile,
            },
            "series": series.provenance(),
        },
    )


def planted_cbv_recovery(
    *,
    true_ratio: float = 3.0,
    n_timepoints: int = 50,
    tr_s: float = 1.83,
    te_s: float = 0.0514,
    noise: float = 0.0,
    seed: int = 0,
) -> Dict[str, float]:
    """Does this pipeline recover a CBV ratio it was handed?

    Two tissue compartments, one with ``true_ratio`` times the other's first-pass
    area (a gamma-variate indicator curve, the standard shape), forward-modelled
    through S(t) = S0 * exp(-TE * C(t)) and read back by `relative_cbv` -- the
    same function the real path calls. Without this, "rCBV" is a number with
    nothing behind it; with it, the number has a measured error bar and a scale.
    """
    from danon.perfusion import DscSeries  # self-import keeps this honest

    rng = np.random.default_rng(seed)
    t = np.arange(n_timepoints, dtype=float) * tr_s

    def gamma_variate(scale: float) -> np.ndarray:
        # Peak near the 12th frame at TR=1.83 s, i.e. a plausible ~20 s bolus.
        arrival, width = 12.0, 3.0
        x = np.clip(t / tr_s - arrival, 0.0, None)
        peak = (width - 1.0) / width * 1.0  # unnormalised; scale is what matters
        curve = (x ** (width - 1.0)) * np.exp(-x) / max(peak, 1e-9)
        return scale * curve

    low, high = gamma_variate(1.0), gamma_variate(true_ratio)
    n_y, n_x = 8, 8
    shape = (n_timepoints, 1, n_y, n_x)
    conc = np.zeros(shape, dtype=np.float64)
    conc[:, 0, : n_y // 2, :] = low[:, None, None]
    conc[:, 0, n_y // 2 :, :] = high[:, None, None]
    s0 = 1000.0
    signal = s0 * np.exp(-te_s * conc)
    if noise:
        signal = signal * (1.0 + rng.normal(0.0, noise, size=signal.shape))

    series = DscSeries(
        signal=signal.astype(np.float32),
        tr_s=tr_s,
        te_s=te_s,
        pixel_mm=(2.0, 2.0),
        slice_mm=None,
        source="planted phantom",
        n_files=0,
    )
    result = relative_cbv(series)
    top = result.rcbv[0, : n_y // 2, :]
    bottom = result.rcbv[0, n_y // 2 :, :]
    recovered = float(np.median(bottom) / np.median(top))
    return {
        "true_ratio": float(true_ratio),
        "recovered_ratio": recovered,
        "relative_error": abs(recovered - true_ratio) / true_ratio,
        "arrival_frame": float(result.arrival_frame),
        "peak_frame": float(result.peak_frame),
        "end_frame": float(result.end_frame),
        "window_frames": float(result.window_frames),
        "peak_is_after_baseline": float(result.peak_frame > DSC_BASELINE_FRAMES),
    }
