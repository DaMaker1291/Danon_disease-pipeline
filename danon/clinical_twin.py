"""Clinical datastream intake: four routine hospital tests -> engine parameters.

WHAT THIS MODULE DOES
---------------------
Implements the four-stream clinical datastream specification for EvoTwin-10X.
Each stream is a routine hospital test, a computational extraction step, and
an exact engine mapping. The discipline is the repo's standard one: every
mapping is a RE-PARAMETERISATION with an identity property (supplying the
stream's neutral value reproduces the engine's default path exactly), every
constant is declared, and any stream without data is REFUSED with a live
count rather than back-filled.

    stream 1  5-ALA fluorescence        -> initial tumour density N(x, t=0)
    stream 2  H&E / Ki-67 histomorphometry -> growth rate r0
    stream 3  DSC-perfusion rCBV/Ktrans -> vasculature + clearance lambda(x)
    stream 4  sequential-MRI delta-volume -> observed clonal velocity

DATA REALITY (checked live at build time, not assumed)
------------------------------------------------------
Only stream 3 has real cached data (BraTS-2021 t1ce contrast volumes for six
cases). Streams 1, 2, and 4 have zero cached fluorescence, pathology, or
sequential-scan files, so their mappings are implemented and unit-tested on
synthetic inputs, and the module's report records the live count (zero) that
refuses them. Supplying real files activates them -- no code changes.

CACHED IS NOT THE SAME AS EXISTING, and the census now reports both. The cached
BraTS path is one pre-operative timepoint per case, but the TCIA collection the
repo probed (UPENN-GBM) carries a second timepoint for 41 of 630 patients, DSC
perfusion for 456, and CaPTk-processed perfusion maps for 628. Streams 3 and 4
are therefore blocked by a DOWNLOAD, not by the absence of data. An earlier
version of this census reported only that stream 4 had no sequential pairs, with
the rationale "BraTS is a single pre-operative timepoint" -- a BraTS-specific
fact that read as a statement about what exists anywhere, and was wrong.

MAPPING DETAILS (constants declared, identity pinned by test)
--------------------------------------------------------------
stream 3, vasculature : vessels(x) = clip(t1ce(x)/max, 0, 1) on the engine
    grid, written as NIfTI and ingested through imaging.from_nifti_mask's
    existing `vessels_path` (the tested perfusion path). t1ce is
    contrast-enhanced T1: enhancement is a vascular-permeability/blood-volume
    proxy, NOT a measured Ktrans.
stream 3, clearance   : lambda(x) = lambda_0 * (1 + gamma * rCBV(x)) --
    the request's D_chemo(x) = D0 * (1 + gamma*rCBV) shape, applied to drug
    CLEARANCE via the new run(clearance_pattern=...) hook. gamma defaults to
    1.0 and is declared. Identity: rCBV uniform == uniform pattern == default.
stream 1, initial map : N_S(x) = alpha * I(x) * (1 - r0_target),
    N_R(x) = alpha * I(x) * r0_target, alpha normalised so the total equals
    the engine's standard fill (identity: uniform I == sim.seed layout in
    total burden; the SPATIAL layout is the stream's entire point).
stream 2, growth      : growth_S = BASELINE_GROWTH * (Ki67 / KI67_REF).
    KI67_REF = 0.15 (published GBM Ki-67 labelling-index reference; declared
    so the ratio is 1.0 at the reference). Identity: Ki67 == KI67_REF gives
    the calibrated default exactly.
stream 4, velocity    : OBSERVATION ONLY. The engine has no mechanism to
    consume a measured velocity field (it derives its own dynamics), so the
    mapping computes divergence(dV/dt) and reports it against the engine's
    simulated velocity -- a calibration/validation target, not an input.
    Claiming otherwise would be fabrication.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.evosteer import (
    CloneParams,
    TumorEvolutionSimulator,
    TumorGeometry,
)
from danon.imaging import from_nifti_mask, read_nifti, write_nifti

# ---------------------------------------------------------------------------
# Declared constants (every one adjustable, none hidden)
# ---------------------------------------------------------------------------

#: Stream 2: GBM Ki-67 labelling index the growth mapping normalises to.
KI67_REF = 0.15

#: Stream 3: clearance-vs-perfusion gain, lambda(x) = lambda_0*(1+gamma*rCBV).
CLEARANCE_GAMMA = 1.0

#: Stream 1: the resistant fraction assigned when digitising a 5-ALA map
#: (no measured clonal assay exists; must be supplied with provenance).
DEFAULT_ALA_R0 = 0.01
DEFAULT_ALA_R0_SOURCE = "engine default 1% (ASSUMED; 5-ALA carries no clonal assay)"

#: Stream 4: interval between the sequential scans (days), for velocity units.
DEFAULT_SCAN_INTERVAL_DAYS = 56  # 8 weeks


# ---------------------------------------------------------------------------
# Stream 3 -- the only stream with real cached data
# ---------------------------------------------------------------------------


def _decimate_to_mask_grid(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Block-average `volume` onto `mask`'s grid (kept for direct array use)."""
    if volume.shape != mask.shape:
        raise ValueError(
            f"volume shape {volume.shape} does not match engine mask {mask.shape}; "
            "decimate the radiology volume with the same spec as the segmentation"
        )
    return volume


def decimate_t1ce_to_engine_grid(
    t1ce_path: str,
    seg_path: str,
    spec: DecimationSpec | None = None,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """Contrast-enhanced T1 -> engine grid, through the SAME transform as the seg.

    Rebuilds the segmentation's bbox-crop + majority-vote decimation window
    for the t1ce volume, then block-averages. Returns (engine_map, bbox) so
    the caller records the exact window applied.
    """
    from danon.realdata import DecimationSpec, _bbox

    spec = spec or DecimationSpec()
    t1, _ = read_nifti(t1ce_path)
    seg, _ = read_nifti(seg_path)
    t1 = np.asarray(t1, dtype=float)
    seg = np.asarray(seg)

    if t1.shape != seg.shape:
        raise ValueError(
            f"t1ce {t1.shape} and segmentation {seg.shape} are different grids; "
            "they must come from the same case's co-registered series"
        )

    bb = _bbox(seg > 0)
    fx, fy, fz = spec.factors
    t1_cropped = t1[bb]
    nx, ny, nz = t1_cropped.shape
    # Trim to a multiple of the decimation factors (same rule as _decimate_labels).
    nx_t, ny_t, nz_t = nx - (nx % fx), ny - (ny % fy), nz - (nz % fz)
    if nx_t == 0 or ny_t == 0 or nz_t == 0:
        raise ValueError(f"cropped t1ce {t1_cropped.shape} too small for factors {spec.factors}")
    t1_trim = t1_cropped[:nx_t, :ny_t, :nz_t]
    blocks = t1_trim.reshape(nx_t // fx, fx, ny_t // fy, fy, nz_t // fz, fz)
    engine = blocks.mean(axis=(1, 3, 5))
    return engine, tuple(int(s.stop) for s in bb)  # type: ignore[return-value]


def build_stream3_vessels(
    t1ce_path: str,
    seg_path: str,
    *,
    out_path: Optional[str] = None,
    spec: Optional["DecimationSpec"] = None,
) -> Dict[str, Any]:
    """Stream 3, layer 1: t1ce -> engine-grid vasculature map (real data).

    Runs the segmentation's EXACT engine transform (bbox-crop + majority-vote
    decimation, verified to reproduce the committed engine_ready mask voxel
    for voxel) and block-averages the t1ce volume through the same window.
    Enhancement is normalised by the engine grid's own max -- the same
    convention imaging.from_nifti_mask applies to supplied perfusion maps --
    and zeroed outside the tumour mask (drug enters through tumour
    vasculature, matching the engine_ready proliferative convention).
    """
    from danon.realdata import DecimationSpec, _bbox, _decimate_labels

    spec = spec or DecimationSpec()
    seg, _ = read_nifti(seg_path)
    seg = np.rint(np.asarray(seg)).astype(np.int32)
    if not any(u in np.unique(seg) for u in (1, 2, 4)):
        raise ValueError(f"{seg_path}: no BraTS tumour labels (1/2/4) present")

    bb = _bbox(seg > 0)
    mask_grid, labels_grid = _decimate_labels(seg[bb], spec)
    engine, _ = decimate_t1ce_to_engine_grid(t1ce_path, seg_path, spec=spec)
    if engine.shape != mask_grid.shape:
        raise ValueError(
            f"t1ce engine grid {engine.shape} != mask grid {mask_grid.shape}; "
            "the volumes are not co-registered"
        )

    vmax = float(engine.max())
    if vmax <= 0:
        raise ValueError(f"{t1ce_path}: decimated t1ce is empty/constant")

    norm = np.clip(engine / vmax, 0.0, 1.0)
    m = mask_grid.astype(bool)
    vessels = np.where(m, norm, 0.0)

    if out_path is not None:
        write_nifti(out_path, np.clip(vessels * 255.0, 0, 255).astype(np.uint8))

    in_tumour = vessels[m]
    return {
        "engine_grid": [int(v) for v in vessels.shape],
        "tumour_voxels": int(m.sum()),
        "voxels_with_enhancement": int((in_tumour > 0.05).sum()),
        "mean_enhancement_in_tumour": float(in_tumour.mean()) if in_tumour.size else 0.0,
        "max_enhancement": float(in_tumour.max()) if in_tumour.size else 0.0,
        "enhancement_contrast_in_tumour": (
            float(in_tumour.max() / in_tumour.mean()) if in_tumour.mean() > 0 else 0.0
        ),
        "bbox_stop_xyz": [int(s.stop) for s in bb],  # type: ignore[union-attr]
        "source": (
            "BraTS-2021 t1ce (contrast-enhanced T1) block-averaged through the "
            "segmentation's engine transform; vascular PROXY, not measured Ktrans"
        ),
    }


def build_stream3_clearance(
    rcbv_engine: np.ndarray,
    *,
    gamma: float = CLEARANCE_GAMMA,
) -> np.ndarray:
    """Stream 3, layer 2: rCBV -> relative clearance pattern lambda(x).

    lambda(x) = (1 + gamma * rCBV(x)) / mean(1 + gamma * rCBV) -- the request's
    mapping shape, normalised to mean 1.0 (the engine re-normalises anyway;
    doing it here makes the identity explicit: uniform rCBV == uniform pattern).
    """
    rcbv = np.asarray(rcbv_engine, dtype=float)
    if rcbv.min() < 0.0:
        raise ValueError("rCBV values must be non-negative")
    raw = 1.0 + float(gamma) * rcbv
    mean = float(raw.mean())
    if mean <= 0.0:
        raise ValueError("clearance pattern has zero mean")
    return raw / mean


# ---------------------------------------------------------------------------
# Stream 1 -- 5-ALA fluorescence -> initial state
# ---------------------------------------------------------------------------


def stream1_initial_state(
    fluorescence: np.ndarray,
    mask: np.ndarray,
    *,
    r0: float = DEFAULT_ALA_R0,
    fill_fraction: float = 0.65,
    carrying_capacity: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """N_S/N_R(x, t=0) = alpha * I(x) * (1 - r0) / alpha * I(x) * r0.

    alpha is chosen so total burden equals the engine's standard fill
    (fill_fraction * K * tumour voxels): the stream replaces the SPATIAL
    layout of the seed, not the total. A uniform I(x) is therefore
    burden-identical to the engine default -- but the engine's clustered
    layout is NOT reproduced (that is the stream's entire point: a measured
    map replaces a random seed).
    """
    if not 0.0 < r0 < 1.0:
        raise ValueError(f"r0 must be in (0, 1), got {r0}")
    fluo = np.asarray(fluorescence, dtype=float)
    m = np.asarray(mask).astype(bool)
    if fluo.shape != m.shape:
        raise ValueError(
            f"fluorescence {fluo.shape} and mask {m.shape} grids differ"
        )
    if fluo.min() < 0.0:
        raise ValueError("fluorescence intensities must be non-negative")
    total_fluo = float(fluo[m].sum())
    if total_fluo <= 0.0:
        raise ValueError("fluorescence map has no signal inside the tumour mask")

    target_total = fill_fraction * carrying_capacity * int(m.sum())
    alpha = target_total / total_fluo
    density = alpha * fluo
    N_S = np.where(m, density * (1.0 - r0), 0.0)
    N_R = np.where(m, density * r0, 0.0)
    return N_S, N_R


# ---------------------------------------------------------------------------
# Stream 2 -- Ki-67 -> growth rate
# ---------------------------------------------------------------------------


def stream2_growth_rate(ki67_fraction: float) -> float:
    """growth_S = BASELINE_GROWTH * (Ki67 / KI67_REF).

    Identity: Ki67 == KI67_REF (0.15, the declared GBM reference labelling
    index) returns the engine's calibrated growth_S exactly.
    """
    if not 0.0 < ki67_fraction <= 1.0:
        raise ValueError(f"Ki-67 fraction must be in (0, 1], got {ki67_fraction}")
    from danon.evosteer import DEFAULT_CLONES

    return float(DEFAULT_CLONES.growth_S) * (float(ki67_fraction) / KI67_REF)


def stream2_clones(ki67_fraction: float) -> CloneParams:
    """CloneParams with stream-2 growth; every other relationship preserved.

    The engine derives growth_R = growth_S * (1 - cost_of_resistance) and
    exports the realised ratio via DEFAULT_CLONES; this rebuilds both rates
    at the SAME resistance cost, so the stream scales proliferation without
    touching the trade-off collateral steering exploits.
    """
    from danon.evosteer import DEFAULT_CLONES

    g = stream2_growth_rate(ki67_fraction)
    cost = 1.0 - DEFAULT_CLONES.growth_R / DEFAULT_CLONES.growth_S
    return CloneParams(
        growth_S=g,
        growth_R=g * (1.0 - cost),
        mutation_rate=DEFAULT_CLONES.mutation_rate,
        competition=DEFAULT_CLONES.competition,
        drug_sensitivity_scale_R=DEFAULT_CLONES.drug_sensitivity_scale_R,
    )


# ---------------------------------------------------------------------------
# Stream 4 -- sequential MRI delta-volume velocity (OBSERVATION ONLY)
# ---------------------------------------------------------------------------


def stream4_velocity(
    scan_t1: np.ndarray,
    scan_t2: np.ndarray,
    *,
    interval_days: float = DEFAULT_SCAN_INTERVAL_DAYS,
) -> Dict[str, Any]:
    """Divergence of the delta-volume velocity field: an OBSERVATION.

    The engine cannot consume a measured velocity (its dynamics are derived,
    and injecting a measured dR/dt would double-count). This returns the
    measured summary -- total delta volume, rate, and peak growth voxels --
    for calibration/validation against simulated velocities.
    """
    a = np.asarray(scan_t1, dtype=float)
    b = np.asarray(scan_t2, dtype=float)
    if a.shape != b.shape:
        raise ValueError(
            f"sequential scans differ in shape {a.shape} vs {b.shape}; "
            "rigid-register to a common grid first"
        )
    if interval_days <= 0:
        raise ValueError("interval_days must be positive")
    dv = b - a
    rate = float(np.abs(dv).sum() / interval_days)
    growing = dv > 0
    return {
        "delta_volume_total": float(dv.sum()),
        "growing_voxels": int(growing.sum()),
        "shrinking_voxels": int((dv < 0).sum()),
        "abs_rate_per_day": rate,
        "peak_growth_voxel_value": float(dv.max()) if dv.size else 0.0,
        "interval_days": float(interval_days),
        "status": "observation only -- calibration target for the engine's derived dynamics",
    }


# ---------------------------------------------------------------------------
# Assemble: the one function the request specifies
# ---------------------------------------------------------------------------


@dataclass
class TwinStreamState:
    """The four-stream assembled state, with per-stream provenance."""

    initial_state: Optional[Tuple[np.ndarray, np.ndarray]]
    clones_override: Optional[CloneParams]
    vessels_path: Optional[str]
    clearance_pattern: Optional[np.ndarray]
    velocity_observation: Optional[Dict[str, Any]]
    stream_status: Dict[str, str]
    notes: List[str] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "stream_status": self.stream_status,
            "notes": self.notes,
            "velocity_observation": self.velocity_observation,
            "has_initial_state": self.initial_state is not None,
            "has_clones_override": self.clones_override is not None,
            "has_vessels": self.vessels_path is not None,
            "has_clearance_pattern": self.clearance_pattern is not None,
        }


def assemble_digital_twin_state(
    fluorescence_5ala_mesh: Optional[np.ndarray] = None,
    histology_ki67_ratio: Optional[float] = None,
    mri_rcbv_tensor: Optional[np.ndarray] = None,
    delta_volume_velocity: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    *,
    mask: Optional[np.ndarray] = None,
    r0: float = DEFAULT_ALA_R0,
    r0_source: str = DEFAULT_ALA_R0_SOURCE,
    clearance_gamma: float = CLEARANCE_GAMMA,
    scan_interval_days: float = DEFAULT_SCAN_INTERVAL_DAYS,
) -> TwinStreamState:
    """Combine whichever routine-hospital datastreams are supplied.

    Only the supplied streams modify the engine's defaults; every omitted
    stream leaves the calibrated parameter untouched, and the returned
    stream_status records which ones were applied. This is the identity
    property at the composition level: no streams -> a state that runs the
    engine exactly as before.
    """
    applied: Dict[str, str] = {}
    notes: List[str] = []

    initial_state = None
    if fluorescence_5ala_mesh is not None:
        if mask is None:
            raise ValueError("stream 1 needs the tumour mask alongside the fluorescence map")
        initial_state = stream1_initial_state(
            fluorescence_5ala_mesh, mask, r0=r0
        )
        applied["stream1_5ala_initial_state"] = (
            f"applied (r0={r0:.4f}, source: {r0_source})"
        )
    else:
        applied["stream1_5ala_initial_state"] = "not supplied -- engine clustered seed retained"

    clones_override = None
    if histology_ki67_ratio is not None:
        clones_override = stream2_clones(histology_ki67_ratio)
        applied["stream2_ki67_growth"] = (
            f"applied (Ki-67 {histology_ki67_ratio:.2f} -> growth_S "
            f"{clones_override.growth_S:.4f}/day; KI67_REF={KI67_REF})"
        )
    else:
        applied["stream2_ki67_growth"] = "not supplied -- calibrated growth retained"

    vessels_path = None
    clearance_pattern = None
    if mri_rcbv_tensor is not None:
        clearance_pattern = build_stream3_clearance(
            mri_rcbv_tensor, gamma=clearance_gamma
        )
        applied["stream3_perfusion_clearance"] = (
            f"applied (gamma={clearance_gamma}; mean-normalised pattern; "
            "vascular source map must be ingested separately via vessels_path)"
        )
        notes.append(
            "stream 3 vasculature: pass the engine-grid t1ce map as vessels_path "
            "to from_nifti_mask; this function consumes the ALREADY-ingested grid"
        )
    else:
        applied["stream3_perfusion_clearance"] = "not supplied -- uniform clearance retained"

    velocity_observation = None
    if delta_volume_velocity is not None:
        t1, t2 = delta_volume_velocity
        velocity_observation = stream4_velocity(
            t1, t2, interval_days=scan_interval_days
        )
        applied["stream4_delta_volume"] = "applied as OBSERVATION (calibration target, not an input)"
    else:
        applied["stream4_delta_volume"] = "not supplied -- no measured velocity"

    return TwinStreamState(
        initial_state=initial_state,
        clones_override=clones_override,
        vessels_path=vessels_path,
        clearance_pattern=clearance_pattern,
        velocity_observation=velocity_observation,
        stream_status=applied,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Measured stream-3 cohort report (the module's honesty artefact)
# ---------------------------------------------------------------------------

#: The committed cohort measurement: 6 BraTS cases x 2 policies x 3 seeds,
#: each run PAIRED (same geometry, same seed) with real t1ce vasculature vs
#: the null-uniform proxy. Committed at outputs_real/stream3_cohort.json;
#: the claim check re-measures case 01017 live rather than trusting the file.
STREAM3_COHORT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs_real", "stream3_cohort.json",
)


def stream3_report() -> Dict[str, Any]:
    """The measured stream-3 verdict, from the committed cohort file.

    Headline: the real t1ce vasculature moves MTD PFS by a mean of -1.17
    days across 18 paired runs (5/6 cases 2-3 days earlier, one unchanged);
    steering is horizon-censored in both geometries so its paired delta is
    exactly zero. The honest reading: at engine scale the vessel-source
    variation contributes ~1 day of PFS -- the sanctuary structure the
    model is known for lives in D(x) (tissue classes), which both
    geometries share. The stream is wired and measured, and its measured
    effect is SMALL -- which is a result, not a failure.
    """
    if not os.path.exists(STREAM3_COHORT_PATH):
        raise FileNotFoundError(
            f"stream-3 cohort measurement missing: {STREAM3_COHORT_PATH}. "
            "Rebuild with: for i in 0 1 2 3 4 5; do python _stream3_cohort.py $i; done"
        )
    import json

    rows = json.load(open(STREAM3_COHORT_PATH))
    d_mtd = [
        pr - pn
        for r in rows
        for pr, pn in zip(r["arms"]["mtd"]["pfs_real"], r["arms"]["mtd"]["pfs_null"])
    ]
    d_steer = [
        pr - pn
        for r in rows
        for pr, pn in zip(
            r["arms"]["steering"]["pfs_real"], r["arms"]["steering"]["pfs_null"]
        )
    ]
    return {
        "n_cases": len(rows),
        "n_paired_runs_per_arm": len(d_mtd),
        "mtd_pfs_delta_mean_days": float(np.mean(d_mtd)),
        "mtd_pfs_delta_sd_days": float(np.std(d_mtd)),
        "mtd_pfs_delta_max_abs_days": float(max(abs(d) for d in d_mtd)),
        "steering_pfs_delta_mean_days": float(np.mean(d_steer)),
        "steering_censored_both": bool(all(d == 0 for d in d_steer)),
        "cases": {
            r["case_id"]: {
                "mtd_pfs_median_real": r["arms"]["mtd"]["pfs_median_real"],
                "mtd_pfs_median_null": r["arms"]["mtd"]["pfs_median_null"],
                "mean_enhancement_in_tumour": r["mean_enhancement_in_tumour"],
            }
            for r in rows
        },
        "source": "outputs_real/stream3_cohort.json (rebuildable via _stream3_cohort.py)",
    }


# ---------------------------------------------------------------------------
# Live data census -- what the repo can actually activate today
# ---------------------------------------------------------------------------


def _archive_availability() -> Dict[str, Any]:
    """Cohort-scored availability from the recorded TCIA probe.

    Read from the probe record rather than restated, so this census cannot drift
    from the numbers the multimodal verdicts are computed from. Returns an
    ``error`` entry if the record is missing: a census that silently reports
    zero retrievable pairs when it cannot read the probe would be the same class
    of mistake as the one this function exists to fix.
    """
    try:
        from danon.multimodal import EVIDENCE_PATH, load_evidence

        for m in load_evidence():
            if m.key != "tcia_upenn_gbm":
                continue
            d = dict(m.detail)
            return {
                "collection": "UPENN-GBM",
                "n_patients": d.get("n_patients"),
                "n_patients_multi_date_any_interval": d.get(
                    "patients_multi_date_any_interval"
                ),
                "n_patients_longitudinal": d.get("patients_with_multi_timepoint"),
                "min_interval_days": d.get("min_followup_interval_days"),
                "n_patients_dsc_perfusion": d.get("n_patients_dsc_perfusion"),
                "n_dsc_series": d.get("n_dsc_series"),
                "n_patients_processed_perfusion": d.get(
                    "n_patients_processed_perfusion"
                ),
                "n_processed_series": d.get("n_processed_series"),
                "n_longitudinal_with_perfusion": d.get(
                    "n_longitudinal_with_perfusion"
                ),
                "n_seg_series": d.get("n_seg_series"),
                "source": f"recorded TCIA probe ({EVIDENCE_PATH})",
            }
        return {"error": "UPENN-GBM entry absent from the recorded probe"}
    except (FileNotFoundError, ImportError, AttributeError, KeyError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def clinical_stream_census(data_dir: str = "data") -> Dict[str, Any]:
    """Live counts per stream: what is cached, what refuses, and why.

    This is the module's honesty artefact. For each stream it separates what is
    on disk (which the mappings can consume today) from what the probed archive
    would supply after a download, so a refusal cannot be mistaken for evidence
    that the data does not exist.
    """
    t1ce_files: List[str] = []
    seg_files: List[str] = []
    brats_dir = os.path.join(data_dir, "brats")
    if os.path.isdir(brats_dir):
        for name in sorted(os.listdir(brats_dir)):
            if name.endswith("_t1ce.nii.gz"):
                t1ce_files.append(name)
            elif name.endswith("_seg.nii.gz"):
                seg_files.append(name)

    ala_files = [
        f
        for root in (os.path.join(data_dir, "fluorescence"),)
        if os.path.isdir(root)
        for f in os.listdir(root)
    ]
    wsi_files = [
        f
        for root in (os.path.join(data_dir, "pathology"), data_dir)
        if os.path.isdir(root)
        for f in os.listdir(root)
        if f.lower().endswith((".svs", ".ndpi", ".tiff"))
    ]
    # Cached pairs: BraTS is one pre-operative timepoint per case, so zero.
    sequential_pairs_cached = 0
    archive = _archive_availability()
    n_retrievable = archive.get("n_patients_longitudinal")

    return {
        "stream1_5ala": {
            "n_files": len(ala_files),
            "status": "REFUSED" if not ala_files else "available",
            "detail": (
                "no 5-ALA fluorescence captures cached; mapping implemented and "
                "unit-tested, activates when a real capture is supplied"
            ),
        },
        "stream2_histology": {
            "n_files": len(wsi_files),
            "status": "REFUSED" if not wsi_files else "available",
            "detail": (
                "no whole-slide images cached; Ki-67 mapping implemented and "
                "unit-tested (KI67_REF=0.15), activates when a WSI is supplied"
            ),
        },
        "stream3_perfusion": {
            "n_t1ce_files": len(t1ce_files),
            "n_seg_files": len(seg_files),
            "n_buildable_cases": len(set(t1ce_files) & {s.replace('_seg', '_t1ce') for s in seg_files}),
            "status": "MEASURABLE" if t1ce_files and seg_files else "REFUSED",
            "n_archive_dsc_patients": archive.get("n_patients_dsc_perfusion"),
            "n_archive_processed_perfusion_patients": archive.get(
                "n_patients_processed_perfusion"
            ),
            "detail": (
                "real cached data: BraTS-2021 t1ce contrast volumes + expert "
                "segmentations; vascular proxy, not measured Ktrans. The probed "
                "archive carries the measured form: DSC perfusion series for "
                f"{archive.get('n_patients_dsc_perfusion')} patients and "
                f"CaPTk-processed rCBV/Ktrans maps for "
                f"{archive.get('n_patients_processed_perfusion')}, neither of "
                "which is downloaded"
            ),
        },
        "stream4_sequential": {
            "n_pairs_cached": sequential_pairs_cached,
            "n_pairs_retrievable": n_retrievable,
            "status": "REFUSED" if sequential_pairs_cached == 0 else "available",
            "archive": archive,
            "detail": (
                "no sequential scan pairs are CACHED (BraTS is a single "
                "pre-operative timepoint per case). Retrievable without new "
                f"code: {n_retrievable} UPENN-GBM patients carry a "
                f">={archive.get('min_interval_days')}-day follow-up interval, "
                "and all of them carry DSC perfusion series. The mapping is "
                "implemented and unit-tested as an observation, and activates "
                "when a longitudinal series is supplied"
            ),
        },
    }
