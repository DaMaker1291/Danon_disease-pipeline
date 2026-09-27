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
                                        + diffusivity D_chemo(x)
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
repo probed (UPENN-GBM) carries a second timepoint for 41 of 630 patients and
raw DSC perfusion for 456. (It also carries 628 patients' worth of
CaPTk-PROCESSED series -- but those are co-registered STRUCTURAL volumes, not
rCBV/Ktrans maps: the smallest was downloaded and read in this session and is
'AX T1 PRE : Processed_CaPTk', TR 777 ms, one frame per file. An earlier
version of this docstring called them perfusion maps.) Streams 3 and 4 were
therefore blocked by a DOWNLOAD, not by the absence of data -- and stream 3 is
no longer blocked: `danon/perfusion.py` reads a DSC series into a relative-CBV
map, and the cohort in `outputs_real/perfusion_cohort.json` is measured on
downloaded series. An earlier
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
stream 3, diffusion   : D_chemo(x) = D0 * (1 + gamma * rCBV(x)) through the
    engine's own `diffusion_field` hook, via the tested anisotropy bridge.
    gamma defaults to 2.5 (the specification's value) and is declared. Two
    properties are stated in the builder rather than left implicit: the
    field is reduced to a mean-one relative pattern, so D0 and the unit
    conversion CANCEL exactly and gamma is this stream's only live constant;
    and an rCBV map is scalar, so the result is isotropic per voxel -- a
    diffusion tensor needs DTI and is not fabricated from perfusion. The
    clearance and diffusion mappings are SEPARATE declared mappings sharing a
    formula shape; applying one physical parameter's shape to another is
    visible here rather than buried.
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

import math
import numbers
import os
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

#: The committed whole-slide probe: the density stream 2's capacity mapping
#: reads its reference and endpoints from, so no number is chosen by hand.
HISTOLOGY_PROBE_PATH = os.path.join("outputs_real", "histology_probe.json")

#: The paired stream-2 capacity cohort: uniform K vs packing-derived K, on real
#: cached geometry, with the sweep's endpoints read from the probe above.
CAPACITY_COHORT_PATH = os.path.join("outputs_real", "stream2_capacity_cohort.json")

#: Stream 3: clearance-vs-perfusion gain, lambda(x) = lambda_0*(1+gamma*rCBV).
CLEARANCE_GAMMA = 1.0

#: Stream 3: diffusion-vs-perfusion gain, D_chemo(x) = D0*(1+gamma*rCBV(x)).
#: The specification writes 2.5. DECLARED, not measured: nothing cached here
#: pairs a perfusion map with a measured drug diffusivity in the same tissue,
#: so the gain is a modelling choice and is labelled one wherever it appears.
DIFFUSION_GAMMA = 2.5

#: Stream 3: reference absolute diffusivity for the physical array only. It
#: cannot move any result -- the simulator reduces a supplied field to a
#: mean-one RELATIVE pattern (see `TumorEvolutionSimulator.__init__`), so D0
#: cancels out of the dynamics exactly. Kept so `d_mm2_per_day` stays
#: inspectable; the engine's calibrated value is 0.35 voxel^2/day at 1 mm.
STREAM3_D0_MM2_PER_DAY = 0.35

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
        # The map itself: every statistic below is derived from it, and the
        # diffusion arm needs the array rather than a round-trip through disk.
        "vessels": vessels,
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


def build_stream3_diffusion_field(
    rcbv_engine: np.ndarray,
    *,
    gamma: float = DIFFUSION_GAMMA,
    voxel_mm: float = 1.0,
    d0_mm2_per_day: float = STREAM3_D0_MM2_PER_DAY,
) -> "DiffusionField":
    """Stream 3, layer 2b: rCBV -> per-voxel drug diffusivity D_chemo(x).

    D_chemo(x) = D0 * (1 + gamma * rCBV(x)) -- the specification's own formula,
    converted to engine units by the tested `danon.anisotropy` bridge.

    TWO THINGS THE FORMULA IMPLIES THAT ARE WORTH STATING PLAINLY

    1. **D0 and the unit conversion cancel.** The simulator reduces any
       supplied field to a mean-one relative pattern, so only
       `(1 + gamma*rCBV)/mean(1 + gamma*rCBV)` reaches the integrator. D0 is
       therefore NOT a free parameter of this stream and no result can be
       tuned by it; the mapping's only live constant is `gamma`.
    2. **It is isotropic per voxel.** An rCBV map is scalar -- it carries a
       magnitude and no direction -- so it cannot produce a diffusion TENSOR.
       Directionality needs DTI and is not fabricated here; the DTI path
       already exists as `DiffusionField.from_tissue_classes(anisotropy=...)`.

    Identity: uniform rCBV reproduces the engine's homogeneous transport -- the
    SAME operator (div(D grad C) with constant D equals D*laplacian(C)), so the
    neutral value is recovered. It is not bit-identical: the two branches sum in
    a different order, and the measured gap on a 60-day synthetic run is ~6e-14
    on the burden arrays. Pinned by test at that tolerance rather than claimed
    as exact equality.
    """
    from danon.anisotropy import DiffusionField, voxel_area_mm2

    rcbv = np.asarray(rcbv_engine, dtype=float)
    if rcbv.ndim != 3:
        raise ValueError(f"rCBV map must be 3D, got shape {rcbv.shape}")
    if not np.all(np.isfinite(rcbv)):
        raise ValueError("rCBV map contains non-finite values")
    if rcbv.min() < 0.0:
        raise ValueError("rCBV values must be non-negative")
    if float(gamma) < 0.0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    if float(voxel_mm) <= 0.0:
        raise ValueError(f"voxel_mm must be positive, got {voxel_mm}")
    if float(d0_mm2_per_day) <= 0.0:
        raise ValueError(f"d0 must be positive, got {d0_mm2_per_day}")
    raw = 1.0 + float(gamma) * rcbv
    if float(raw.mean()) <= 0.0:
        raise ValueError("diffusion field has zero mean: no transport at all")
    physical = float(d0_mm2_per_day) * raw
    return DiffusionField(
        d_engine=physical / voxel_area_mm2(float(voxel_mm)),
        d_mm2_per_day=physical,
        voxel_mm=float(voxel_mm),
        provenance={
            # t1ce enhancement is a vascular proxy, not a measured rCBV series.
            "perfusion_map": EvidenceStatus.UNKNOWN,
            "gain_gamma": EvidenceStatus.UNKNOWN,
            "unit_conversion": EvidenceStatus.ESTABLISHED,
        },
    )


# ---------------------------------------------------------------------------
# Stream 3, diffusion arm -- what the perfusion-derived field changes, measured
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DiffusionContrast:
    """Homogeneous vs perfusion-derived transport: same geometry, same seeds.

    The only thing that differs between the two arms is the transport field,
    so a difference in outcome is attributable to the mapping and not to a
    change of geometry, seed, policy, or initial condition.
    """

    case_id: str
    n_voxels: int
    gamma: float
    field_min_mm2_per_day: float
    field_max_mm2_per_day: float
    field_contrast_ratio: float
    voxels_active: int
    policy: str
    seeds: Tuple[int, ...]
    pfs_uniform_days: Tuple[int, ...]
    pfs_perfusion_days: Tuple[int, ...]
    toxicity_uniform: Tuple[float, ...]
    toxicity_perfusion: Tuple[float, ...]

    @property
    def median_pfs_delta_days(self) -> float:
        return float(
            np.median(self.pfs_perfusion_days) - np.median(self.pfs_uniform_days)
        )

    @property
    def n_pairs_in_disagreement(self) -> int:
        """Pairs where the two arms' PFS differ (the resolution of this test)."""
        return int(
            sum(a != b for a, b in zip(self.pfs_uniform_days, self.pfs_perfusion_days))
        )

    def describe(self) -> str:
        return (
            f"case {self.case_id} ({self.n_voxels} voxels, {self.voxels_active} carrying "
            f"perfusion): gamma={self.gamma:g} gives "
            f"{self.field_min_mm2_per_day:.4g}-{self.field_max_mm2_per_day:.4g} mm^2/day "
            f"({self.field_contrast_ratio:.1f}x contrast); {self.policy} over "
            f"seeds {list(self.seeds)}: PFS {self.median_pfs_delta_days:+.1f} d median "
            f"paired delta, {self.n_pairs_in_disagreement}/{len(self.seeds)} pairs differ"
        )


def stream3_diffusion_contrast(
    case_id: str,
    *,
    brats_dir: str = "data/brats",
    engineered_dir: str = "data/engine_ready",
    gamma: float = DIFFUSION_GAMMA,
    policy: str = "steering",
    seeds: Tuple[int, ...] = (101, 102, 103),
    days: int = 365,
    r0: float = 0.005,
) -> DiffusionContrast:
    """Pair the uniform and perfusion-derived transport arms on one real case.

    Same protocol as the committed stream-3 cohort: BraTS-2021 t1ce through the
    segmentation's own engine transform, engine-ready mask and proliferation,
    fixed seeds, paired so a difference is attributable to the field alone.
    """
    from danon.evosteer import CollateralSwitchPolicy, MTDPolicy

    policies = {"mtd": MTDPolicy, "steering": CollateralSwitchPolicy}
    if policy not in policies:
        raise ValueError(f"policy must be one of {sorted(policies)}, got {policy!r}")

    mask_file = os.path.join(engineered_dir, f"BraTS2021_{case_id}_mask.nii.gz")
    prolif_file = os.path.join(engineered_dir, f"BraTS2021_{case_id}_prolif.nii.gz")
    t1ce = os.path.join(brats_dir, f"BraTS2021_{case_id}_t1ce.nii.gz")
    seg = os.path.join(brats_dir, f"BraTS2021_{case_id}_seg.nii.gz")
    for path in (mask_file, prolif_file, t1ce, seg):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} is not cached; this measurement runs on the committed "
                f"BraTS-2021 case files, not on generated input"
            )

    vessels = build_stream3_vessels(t1ce, seg)
    rcbv = np.asarray(vessels["vessels"], dtype=float)
    field = build_stream3_diffusion_field(rcbv, gamma=gamma)
    page = from_nifti_mask(
        mask_file,
        proliferative_path=prolif_file,
        vessels_path=os.path.join(engineered_dir, f"BraTS2021_{case_id}_vessels_t1ce.nii.gz"),
    )
    pfs_u: List[int] = []
    pfs_p: List[int] = []
    tox_u: List[float] = []
    tox_p: List[float] = []
    for seed in seeds:
        for arm, sim in (
            (pfs_u, TumorEvolutionSimulator(page.geometry)),
            (pfs_p, TumorEvolutionSimulator(page.geometry, diffusion_field=field)),
        ):
            traj = sim.run(
                policies[policy](), days=days, initial_resistant_fraction=r0, seed=seed
            )
            arm.append(int(traj.pfs_day) if traj.pfs_day is not None else days)
            (tox_u if arm is pfs_u else tox_p).append(float(traj.toxicity))

    lo = float(field.d_mm2_per_day[field.d_mm2_per_day > 0].min())
    hi = float(field.d_mm2_per_day.max())
    return DiffusionContrast(
        case_id=case_id,
        n_voxels=int(rcbv.size),
        gamma=float(gamma),
        field_min_mm2_per_day=lo,
        field_max_mm2_per_day=hi,
        field_contrast_ratio=(hi / lo) if lo > 0 else float("inf"),
        voxels_active=int(np.count_nonzero(rcbv)),
        policy=policy,
        seeds=tuple(seeds),
        pfs_uniform_days=tuple(pfs_u),
        pfs_perfusion_days=tuple(pfs_p),
        toxicity_uniform=tuple(tox_u),
        toxicity_perfusion=tuple(tox_p),
    )


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


@dataclass(frozen=True)
class VelocityAgreement:
    """A MEASURED delta-volume field compared with the engine's own.

    This is the only lawful use of stream 4 in this repository, and it is why
    the field is called ``agreement`` and not ``resistance_drift``. A measured
    velocity is a VALIDATION TARGET for the dynamics the engine derives; it is
    never an input to them, because injecting a measured dR/dt into a model that
    already derives a dR/dt double-counts the same rate twice. Nothing here is a
    covariate: a comparison between two fields cannot move a discrimination
    metric, however good the agreement turns out to be.

    ``status`` is MEASURED only when both sides were supplied. Otherwise the
    object carries the live count that refuses it, and no comparison statistics
    at all -- a correlation against an absent reference is a fabricated number,
    which is the failure mode this module exists to avoid.
    """

    status: str
    interval_days: float
    n_voxels: int
    measured_abs_rate_per_day: Optional[float] = None
    simulated_abs_rate_per_day: Optional[float] = None
    rate_ratio: Optional[float] = None
    pearson_r: Optional[float] = None
    sign_agreement: Optional[float] = None
    measured_peak_voxel: Optional[int] = None
    simulated_peak_voxel: Optional[int] = None
    refusal: Optional[str] = None

    def describe(self) -> str:
        if self.status != "MEASURED":
            return f"REFUSED: {self.refusal}"
        r = "n/a (no variance)" if self.pearson_r is None else f"{self.pearson_r:+.4f}"
        return (
            f"MEASURED over {self.n_voxels:,} voxels: measured "
            f"{self.measured_abs_rate_per_day:.4g}/day vs simulated "
            f"{self.simulated_abs_rate_per_day:.4g}/day (ratio "
            f"{self.rate_ratio:.3f}), Pearson r = {r}, sign agreement "
            f"{self.sign_agreement:.3f}. Validation target: it cannot move a "
            "discrimination metric."
        )

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "interval_days": self.interval_days,
            "n_voxels": self.n_voxels,
            "measured_abs_rate_per_day": self.measured_abs_rate_per_day,
            "simulated_abs_rate_per_day": self.simulated_abs_rate_per_day,
            "rate_ratio": self.rate_ratio,
            "pearson_r": self.pearson_r,
            "sign_agreement": self.sign_agreement,
            "measured_peak_voxel": self.measured_peak_voxel,
            "simulated_peak_voxel": self.simulated_peak_voxel,
            "refusal": self.refusal,
            "role": (
                "validation target for the engine's derived velocity -- never an "
                "input and never a covariate"
            ),
        }


def stream4_velocity_agreement(
    measured: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    simulated: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    *,
    interval_days: float = DEFAULT_SCAN_INTERVAL_DAYS,
) -> VelocityAgreement:
    """Compare a measured delta-volume field with the engine's simulated one.

    Both sides are two same-grid volumes separated by ``interval_days``. The
    statistics are computed over the voxels where EITHER field is non-zero: a
    correlation over a mostly-empty grid is dominated by the shared zeros and
    would report agreement that neither field earned.

    Call it with neither pair and it refuses, naming the live count that
    refuses it -- so the refusal is reproducible rather than asserted.
    """
    if measured is None or simulated is None:
        missing = [
            name
            for name, value in (("measured", measured), ("simulated", simulated))
            if value is None
        ]
        # The counts in a refusal must BE the census's counts, so this reads the
        # census rather than restating them: a second copy of a live number is
        # the drift this repository keeps finding.
        stream4 = clinical_stream_census().get("stream4_sequential", {})
        cached = stream4.get("n_pairs_cached")
        retrievable = stream4.get("n_pairs_retrievable")
        archive = stream4.get("archive") or {}
        manifest = stream4.get("manifest") or {}
        in_manifest = manifest.get("n_patients_with_usable_pair")
        return VelocityAgreement(
            status="REFUSED",
            interval_days=float(interval_days),
            n_voxels=0,
            refusal=(
                f"no {', '.join(missing)} sequential-scan pair: "
                f"{cached} cached, {retrievable} retrievable from the probed "
                f"{archive.get('collection', 'archive')}, {in_manifest} usable-pair "
                f"patients in the cached CFB-GBM manifest. A correlation against "
                "an absent reference is not reported."
            ),
        )

    m1, m2 = measured
    s1, s2 = simulated
    if np.asarray(m1).shape != np.asarray(s1).shape:
        raise ValueError(
            f"measured grid {np.asarray(m1).shape} and simulated grid "
            f"{np.asarray(s1).shape} differ; register both to one grid first"
        )

    m_stats = stream4_velocity(m1, m2, interval_days=interval_days)
    s_stats = stream4_velocity(s1, s2, interval_days=interval_days)

    dm = np.asarray(m2, dtype=float) - np.asarray(m1, dtype=float)
    ds = np.asarray(s2, dtype=float) - np.asarray(s1, dtype=float)
    both_zero = (dm == 0.0) & (ds == 0.0)
    active = ~both_zero
    n_active = int(active.sum())
    if n_active == 0:
        raise ValueError(
            "both fields are flat: there is no change to compare, and the "
            "apparent perfect agreement would be an artefact of an empty grid"
        )

    x = dm[active]
    y = ds[active]
    sign_agreement = float((np.sign(x) == np.sign(y)).mean())
    pearson: Optional[float]
    if x.std() == 0.0 or y.std() == 0.0:
        pearson = None
    else:
        pearson = float(np.corrcoef(x, y)[0, 1])

    sim_rate = float(s_stats["abs_rate_per_day"])
    return VelocityAgreement(
        status="MEASURED",
        interval_days=float(interval_days),
        n_voxels=n_active,
        measured_abs_rate_per_day=float(m_stats["abs_rate_per_day"]),
        simulated_abs_rate_per_day=sim_rate,
        rate_ratio=(
            float(m_stats["abs_rate_per_day"]) / sim_rate if sim_rate else None
        ),
        pearson_r=pearson,
        sign_agreement=sign_agreement,
        measured_peak_voxel=int(np.argmax(dm)),
        simulated_peak_voxel=int(np.argmax(ds)),
    )


# ---------------------------------------------------------------------------
# Assemble: the one function the request specifies
# ---------------------------------------------------------------------------


@dataclass
class TwinStreamState:
    """The four-stream assembled state, with per-stream provenance.

    `initial_state` is a PAIR with `clones_override`: a measured density scales
    the ceiling and the seed together (the fill fraction is calibrated against
    K), so running the seed against the default clones over-saturates the tumour
    by K. Pass `clones=state.clones_override` to the simulator, or neither.
    """

    initial_state: Optional[Tuple[np.ndarray, np.ndarray]]
    clones_override: Optional[CloneParams]
    vessels_path: Optional[str]
    clearance_pattern: Optional[np.ndarray]
    velocity_observation: Optional[Dict[str, Any]]
    stream_status: Dict[str, str]
    #: Stream 3's diffusion arm: the per-voxel D_chemo(x) the simulator takes as
    #: `diffusion_field=`. Separate from `clearance_pattern` because they are
    #: separate physical parameters that happen to share a formula shape.
    diffusion_field: Optional["DiffusionField"] = None
    notes: List[str] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        field = self.diffusion_field
        return {
            "stream_status": self.stream_status,
            "notes": self.notes,
            "velocity_observation": self.velocity_observation,
            "has_initial_state": self.initial_state is not None,
            "has_clones_override": self.clones_override is not None,
            "has_vessels": self.vessels_path is not None,
            "has_clearance_pattern": self.clearance_pattern is not None,
            "has_diffusion_field": field is not None,
            "diffusion_field": None if field is None else {
                "shape": list(field.shape),
                "voxel_mm": field.voxel_mm,
                "d_mm2_per_day_min": float(field.d_mm2_per_day.min()),
                "d_mm2_per_day_max": float(field.d_mm2_per_day.max()),
                "contrast_ratio": field.contrast_ratio,
                "provenance": {
                    k: v.value if hasattr(v, "value") else str(v)
                    for k, v in field.provenance.items()
                },
            },
        }


@dataclass(frozen=True)
class CarryCapacityResult:
    """The measured packing as a relative per-voxel density ceiling."""

    ratio: float
    k: float
    baseline_k: float
    density_per_mm2: float
    reference_per_mm2: float
    mode: str
    provenance: str
    absolute_refusal: str

    def describe(self) -> str:
        return (
            f"K = {self.k:.4g} ({self.ratio:.3f}x the {self.baseline_k:g} baseline) "
            f"from {self.density_per_mm2:g} nuclei/mm^2 against a "
            f"{self.reference_per_mm2:g}/mm^2 reference"
        )


def stream2_carrying_capacity(
    nuclei_per_mm2: float,
    reference_nuclei_per_mm2: float,
    *,
    baseline_k: Optional[float] = None,
) -> CarryCapacityResult:
    """Measured cell packing -> the engine's per-voxel density ceiling, RELATIVELY.

    Call it with a density and the reference it is meant relative to; the result
    is the ratio the engine's K is scaled by, in the mean-one relative-pattern
    convention the transport field already uses. An ABSOLUTE ceiling is refused:
    the slide counts nuclei per unit area and the engine's K is per unit volume,
    and the archive's slides record no section thickness (see
    `danon.histology.cells_per_mm2_to_per_mm3`), so that route would set the
    tumour's density from an assumed Z.

    This is not a scale-only knob. The engine carries two ABSOLUTE floors --
    `MIN_VIABLE_DENSITY` (a voxel holding less than half a cell is emptied) and
    `DETECTION_BURDEN_DENSITY` (below it no policy may act) -- so moving K moves
    the seeded tumour's distance from those floors, and where a floor bites the
    trajectory changes. Where it does not, it does not; that is what the paired
    contrast measures rather than assumes.
    """
    from danon.evosteer import DEFAULT_CLONES

    k0 = float(DEFAULT_CLONES.carrying_capacity if baseline_k is None else baseline_k)
    for name, value in (
        ("nuclei_per_mm2", nuclei_per_mm2),
        ("reference_nuclei_per_mm2", reference_nuclei_per_mm2),
    ):
        # numbers.Real, not (int, float): a density measured off an array is as
        # likely to arrive as np.float32 as as a Python float, and rejecting a
        # numpy scalar while claiming it "is not a finite number" was both wrong
        # and unhelpful. A bool is a Real too, and is not a measurement.
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            raise ValueError(
                f"{name} must be a real number, got {type(value).__name__}"
            )
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite, got {value!r}")
        if float(value) <= 0.0:
            raise ValueError(f"{name} must be positive, got {value!r}")
    if k0 <= 0.0:
        raise ValueError(f"baseline carrying capacity must be positive, got {k0!r}")

    ratio = float(nuclei_per_mm2) / float(reference_nuclei_per_mm2)
    return CarryCapacityResult(
        ratio=ratio,
        k=k0 * ratio,
        baseline_k=k0,
        density_per_mm2=float(nuclei_per_mm2),
        reference_per_mm2=float(reference_nuclei_per_mm2),
        mode="relative",
        provenance=(
            "declared: K is scaled by the measured areal density ratio, taking "
            "the slide's own density as the reference (mean-one convention)"
        ),
        absolute_refusal=(
            "an absolute cells/mm^3 ceiling is not derivable from a whole-slide "
            "image: the archive's slides record MPP and AppMag but no section "
            "thickness, so the areal-to-volumetric division has no Z to divide by"
        ),
    )


def stream2_capacity_clones(
    nuclei_per_mm2: float,
    reference_nuclei_per_mm2: float,
    *,
    base: Optional[CloneParams] = None,
) -> CloneParams:
    """The clone parameters with the measured packing as the density ceiling.

    Same shape as `stream2_clones`: everything except `carrying_capacity` is the
    base set, so a paired run differs in K and nothing else.
    """
    from danon.evosteer import DEFAULT_CLONES

    result = stream2_carrying_capacity(nuclei_per_mm2, reference_nuclei_per_mm2)
    return replace(
        base if base is not None else DEFAULT_CLONES, carrying_capacity=result.k
    )


@dataclass(frozen=True)
class CapacityContrast:
    """Uniform ceiling vs packing-derived ceiling: same geometry, same seeds.

    The only thing that differs between the arms is `clones.carrying_capacity`,
    so a difference in outcome is attributable to the density mapping rather than
    to geometry, seed, policy or initial resistant fraction.
    """

    case_id: str
    n_voxels: int
    density_per_mm2: float
    reference_per_mm2: float
    reference_role: str
    policy: str
    seeds: Tuple[int, ...]
    ratios: Tuple[float, ...]
    k_by_ratio: Dict[str, float]
    pfs_uniform_days: Tuple[int, ...]
    pfs_capacity_days: Dict[str, Tuple[int, ...]]
    burden_uniform: Tuple[float, ...]
    burden_capacity: Dict[str, Tuple[float, ...]]

    def median_pfs_delta_days(self, ratio: float) -> float:
        return float(
            np.median(self.pfs_capacity_days[_key(ratio)])
            - np.median(self.pfs_uniform_days)
        )

    def max_abs_relative_burden_difference(self, ratio: float) -> float:
        """Largest |burden_capacity - burden_uniform| / burden_uniform, paired.

        The readout with resolution when every arm is censored: a censored PFS
        cannot distinguish arms, so the claim must rest on the burden when it
        cannot rest on PFS.
        """
        pairs = zip(self.burden_uniform, self.burden_capacity[_key(ratio)])
        return float(
            max(
                (
                    abs(b - u) / u if u > 0 else (0.0 if b == 0 else float("inf"))
                    for u, b in pairs
                ),
                default=0.0,
            )
        )

    def n_pairs_in_disagreement(self, ratio: float) -> int:
        return int(
            sum(
                a != b
                for a, b in zip(self.pfs_uniform_days, self.pfs_capacity_days[_key(ratio)])
            )
        )

    def describe(self) -> str:
        parts = []
        for ratio in self.ratios:
            parts.append(
                f"K x{ratio:.3f}: PFS delta "
                f"{self.median_pfs_delta_days(ratio):+.1f} d, "
                f"burden up to {self.max_abs_relative_burden_difference(ratio):.2%}"
            )
        return (
            f"case {self.case_id} ({self.n_voxels} voxels, {self.policy}, "
            f"{len(self.seeds)} paired seeds): " + "; ".join(parts)
        )


def _key(ratio: float) -> str:
    """Ratio -> JSON-safe key, so records round-trip without losing precision."""
    return f"{float(ratio):.6f}"


def stream2_capacity_contrast(
    case_id: str,
    *,
    nuclei_per_mm2: float,
    reference_nuclei_per_mm2: float,
    reference_role: str,
    ratios: Sequence[float],
    brats_dir: str = "data/brats",
    engineered_dir: str = "data/engine_ready",
    policy: str = "steering",
    seeds: Tuple[int, ...] = (101, 102, 103),
    days: int = 365,
    r0: float = 0.005,
) -> CapacityContrast:
    """Run the uniform-K control against packing-derived K on one real case.

    Paired by seed and policy on identical engine-ready geometry, so K is the
    only difference. K itself is set by `stream2_capacity_clones`, i.e. by the
    measured density ratio, never by hand.
    """
    from danon.evosteer import CollateralSwitchPolicy, MTDPolicy

    policies = {"mtd": MTDPolicy, "steering": CollateralSwitchPolicy}
    if policy not in policies:
        raise ValueError(f"policy must be one of {sorted(policies)}, got {policy!r}")
    if not ratios:
        raise ValueError("at least one K ratio is required")
    if not seeds:
        raise ValueError(
            "no seeds: an empty cohort is not a null result, it is no measurement"
        )
    # A perfectly uniform slide yields ratio 1.0 for its min AND its max crop, so
    # the distinct ratios a caller passes can repeat. Dedupe on the SAME key the
    # arms are keyed by, so the ratio list and `clones_by_ratio` cannot disagree:
    # deduping on exact floats while keying at six decimals would leave an arm
    # short one key and run the same arm twice.
    ratios = list({_key(ratio): float(ratio) for ratio in ratios}.values())

    mask_file = os.path.join(engineered_dir, f"BraTS2021_{case_id}_mask.nii.gz")
    prolif_file = os.path.join(engineered_dir, f"BraTS2021_{case_id}_prolif.nii.gz")
    for path in (mask_file, prolif_file):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} is not cached; this measurement runs on the committed "
                f"BraTS-2021 case files, not on generated input"
            )
    page = from_nifti_mask(mask_file, proliferative_path=prolif_file)

    pfs_u: List[int] = []
    bur_u: List[float] = []
    pfs_c: Dict[str, List[int]] = {_key(r): [] for r in ratios}
    bur_c: Dict[str, List[float]] = {_key(r): [] for r in ratios}
    # Built once per ratio and reused across seeds: CloneParams is frozen, so an
    # arm's clones are the same object every replicate.
    clones_by_ratio = {
        _key(ratio): stream2_capacity_clones(
            nuclei_per_mm2 * ratio, reference_nuclei_per_mm2
        )
        for ratio in ratios
    }
    k_by_ratio = {
        key: clones.carrying_capacity for key, clones in clones_by_ratio.items()
    }
    def run(sim: TumorEvolutionSimulator, seed: int) -> Tuple[int, float]:
        traj = sim.run(
            policies[policy](), days=days, initial_resistant_fraction=r0, seed=seed
        )
        pfs = int(traj.pfs_day) if traj.pfs_day is not None else days
        # The trajectory exposes per-day burden arrays; the end-of-horizon total
        # is their last element, the quantity the engine aggregates internally as
        # `final_burden`.
        burden = (
            float(traj.burden_S[-1] + traj.burden_R[-1]) if len(traj.burden_S) else 0.0
        )
        return pfs, burden

    # The control runs ONCE per seed, and every arm is paired to that same run.
    # Re-running the control inside the ratio loop left `bur_u` one element long
    # per ratio while each arm held one per seed, so the paired differences were
    # computed against the wrong partner. The K x1.0 arm -- which must reproduce
    # the control exactly -- is what exposed it.
    for seed in seeds:
        pfs, burden = run(TumorEvolutionSimulator(page.geometry), seed)
        pfs_u.append(pfs)
        bur_u.append(burden)
        for ratio in ratios:
            pfs, burden = run(
                TumorEvolutionSimulator(
                    page.geometry, clones=clones_by_ratio[_key(ratio)]
                ),
                seed,
            )
            pfs_c[_key(ratio)].append(pfs)
            bur_c[_key(ratio)].append(burden)

    return CapacityContrast(
        case_id=case_id,
        n_voxels=int(page.geometry.tumor_mask.size),
        density_per_mm2=float(nuclei_per_mm2),
        reference_per_mm2=float(reference_nuclei_per_mm2),
        reference_role=reference_role,
        policy=policy,
        seeds=tuple(int(s) for s in seeds),
        ratios=tuple(float(r) for r in ratios),
        k_by_ratio=k_by_ratio,
        pfs_uniform_days=tuple(pfs_u),
        pfs_capacity_days={k: tuple(v) for k, v in pfs_c.items()},
        burden_uniform=tuple(bur_u),
        burden_capacity={k: tuple(v) for k, v in bur_c.items()},
    )


def assemble_digital_twin_state(
    fluorescence_5ala_mesh: Optional[np.ndarray] = None,
    histology_ki67_ratio: Optional[float] = None,
    mri_rcbv_tensor: Optional[np.ndarray] = None,
    delta_volume_velocity: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    *,
    mask: Optional[np.ndarray] = None,
    histology_density_per_mm2: Optional[float] = None,
    density_reference_per_mm2: Optional[float] = None,
    r0: float = DEFAULT_ALA_R0,
    r0_source: str = DEFAULT_ALA_R0_SOURCE,
    clearance_gamma: float = CLEARANCE_GAMMA,
    diffusion_gamma: float = DIFFUSION_GAMMA,
    voxel_mm: float = 1.0,
    scan_interval_days: float = DEFAULT_SCAN_INTERVAL_DAYS,
) -> TwinStreamState:
    """Combine whichever routine-hospital datastreams are supplied.

    Only the supplied streams modify the engine's defaults; every omitted
    stream leaves the calibrated parameter untouched, and the returned
    stream_status records which ones were applied. This is the identity
    property at the composition level: no streams -> a state that runs the
    engine exactly as before.

    `mri_rcbv_tensor` feeds TWO parameters, because the specification's
    formula shape appears against both and they are not interchangeable:
    `clearance_pattern` (drug removal) and `diffusion_field` (drug transport).
    They are returned as separate fields so a caller can supply one without
    the other, and so the engine's `run(clearance_pattern=...)` and
    `TumorEvolutionSimulator(diffusion_field=...)` never receive a value
    derived for the wrong parameter.
    """
    applied: Dict[str, str] = {}
    notes: List[str] = []

    initial_state = None
    clones_override = None
    if histology_ki67_ratio is not None:
        clones_override = stream2_clones(histology_ki67_ratio)
        applied["stream2_ki67_growth"] = (
            f"applied (Ki-67 {histology_ki67_ratio:.2f} -> growth_S "
            f"{clones_override.growth_S:.4f}/day; KI67_REF={KI67_REF})"
        )
    else:
        applied["stream2_ki67_growth"] = "not supplied -- calibrated growth retained"

    # Measured cell packing -> the per-voxel density ceiling. Relative by
    # construction (the slide's own density is the reference); the absolute
    # route needs a section thickness the archive's slides do not record.
    from danon.evosteer import DEFAULT_CLONES

    if (histology_density_per_mm2 is None) != (density_reference_per_mm2 is None):
        raise ValueError(
            "stream 2's density mapping needs BOTH histology_density_per_mm2 and "
            "density_reference_per_mm2: a density with no declared reference "
            "would set K against whatever number happened to be nearby"
        )
    capacity_k = DEFAULT_CLONES.carrying_capacity
    if histology_density_per_mm2 is not None:
        # Through the stream's own function, on top of whatever the growth
        # mapping already produced, so there is ONE place that decides how a
        # measured density lands on clones.
        clones_override = stream2_capacity_clones(
            histology_density_per_mm2,
            density_reference_per_mm2,
            base=clones_override if clones_override is not None else DEFAULT_CLONES,
        )
        capacity = stream2_carrying_capacity(
            histology_density_per_mm2, density_reference_per_mm2
        )
        capacity_k = capacity.k
        applied["stream2_density_capacity"] = f"applied ({capacity.describe()})"
    else:
        applied["stream2_density_capacity"] = (
            "not supplied -- calibrated carrying capacity retained"
        )

    # Stream 1 seeds LAST, because the seed scales with the ceiling: `fill_fraction`
    # is CALIBRATED against K, so seeding at the default 1.0 while the ceiling sits
    # at a measured K would silently divide the fill by K -- at K=2.48 the tumour
    # would start at 26% of its stated fill. The engine's own `seed()` scales the
    # same way, so an unmeasured run is unchanged (capacity_k is the baseline).
    if fluorescence_5ala_mesh is not None:
        if mask is None:
            raise ValueError(
                "stream 1 needs the tumour mask alongside the fluorescence map"
            )
        initial_state = stream1_initial_state(
            fluorescence_5ala_mesh, mask, r0=r0, carrying_capacity=capacity_k
        )
        applied["stream1_5ala_initial_state"] = (
            f"applied (r0={r0:.4f}, source: {r0_source})"
        )
    else:
        applied["stream1_5ala_initial_state"] = (
            "not supplied -- engine clustered seed retained"
        )

    vessels_path = None
    clearance_pattern = None
    diffusion_field = None
    if mri_rcbv_tensor is not None:
        clearance_pattern = build_stream3_clearance(
            mri_rcbv_tensor, gamma=clearance_gamma
        )
        diffusion_field = build_stream3_diffusion_field(
            mri_rcbv_tensor, gamma=diffusion_gamma, voxel_mm=voxel_mm
        )
        applied["stream3_perfusion_clearance"] = (
            f"applied (gamma={clearance_gamma}; mean-normalised pattern; "
            "vascular source map must be ingested separately via vessels_path)"
        )
        applied["stream3_perfusion_diffusion"] = (
            f"applied (gamma={diffusion_gamma}; {diffusion_field.contrast_ratio:.1f}x "
            f"contrast in D(x); pass as TumorEvolutionSimulator(diffusion_field=...))"
        )
        notes.append(
            "stream 3 vasculature: pass the engine-grid t1ce map as vessels_path "
            "to from_nifti_mask; this function consumes the ALREADY-ingested grid"
        )
        notes.append(
            f"stream 3 diffusion: D0 and the unit conversion cancel in the "
            f"engine's mean-one pattern, so gamma={diffusion_gamma} is the only "
            f"constant that moves this parameter"
        )
    else:
        applied["stream3_perfusion_clearance"] = "not supplied -- uniform clearance retained"
        applied["stream3_perfusion_diffusion"] = "not supplied -- homogeneous transport retained"

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
        diffusion_field=diffusion_field,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Measured stream-3 cohort report (the module's honesty artefact)
# ---------------------------------------------------------------------------

#: The committed stream-3 DIFFUSION cohort: 6 cases x 2 policies x 3 seeds,
#: each run paired (same geometry, same seed, same vessels source) with the
#: uniform-transport arm. Committed at outputs_real/stream3_diffusion_cohort.json;
#: rebuildable per case via `python _stream3_diffusion_cohort.py INDEX`.
STREAM3_DIFFUSION_COHORT_PATH = os.path.join(
    "outputs_real", "stream3_diffusion_cohort.json"
)

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


#: The committed stream-4 design calculation: whether a cohort of the
#: manifest's size has the resolution to show anything at all. Built by
#: `_stream4_resolution.py` from the real cached cohort.
STREAM4_RESOLUTION_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs_real", "stream4_resolution.json",
)


def stream4_resolution_report() -> Dict[str, Any]:
    """What a cohort of the manifest's size can resolve -- a DESIGN calculation.

    Stream 4 has no cached images, so nothing about delta-volume can be
    measured. But the question that comes BEFORE the download is answerable
    today: a cohort of 168 patients (the manifest's usable-pair count) has a
    held-out C-index that moves by 0.0219 across independent draws of that
    size from the real cohort, and its fold yardstick is 0.0610. Any stream
    whose claimed gain sits under those numbers cannot be shown to contribute
    at this n, whatever its mechanism is.

    This does not measure velocity, registration error or scanner
    comparability, and it is not evidence that stream 4 carries a signal. It
    is the resolution of the measurement, which is what decides whether the
    download is worth performing.
    """
    if not os.path.exists(STREAM4_RESOLUTION_PATH):
        raise FileNotFoundError(
            f"stream-4 design calculation missing: {STREAM4_RESOLUTION_PATH}. "
            "Rebuild with: python _stream4_resolution.py"
        )
    import json

    payload = json.load(open(STREAM4_RESOLUTION_PATH))
    res = payload["resolution"]
    payload["verdict"] = (
        f"a cohort of {res['n_patients']} resolves |delta_C| > "
        f"{res['floor_optimistic']:.4f} (1 SE over {res['n_draws']} draws); its "
        f"fold yardstick is {res['floor_conservative']:.4f}, and the full cached "
        f"cohort (n={payload['source_n_patients']}) has a fold sd of "
        f"{payload['source_base_fold_sd']:.4f} at C={payload['source_base_c']:.4f}"
    )
    return payload


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


def _stream4_resolution() -> Dict[str, Any]:
    """The committed design calculation, or the reason it could not be read."""
    try:
        return stream4_resolution_report()
    except (FileNotFoundError, ImportError, KeyError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _perfusion_cohort_evidence(data_dir: str = "data") -> Optional[Dict[str, Any]]:
    import json

    """The committed DSC perfusion cohort, plus the series cached on disk.

    Read from the record rather than restated, and paired with a live count of
    the downloaded series directories: a cohort artefact whose series have been
    deleted is a number with nothing behind it, and this function is the one
    place that notices.
    """
    path = os.path.join("outputs_real", "perfusion_cohort.json")
    cached = 0
    root = os.path.join(data_dir, "tcia", "perfusion")
    if os.path.isdir(root):
        for name in os.listdir(root):
            if any(n.endswith(".dcm") for n in os.listdir(os.path.join(root, name))):
                cached += 1
    try:
        with open(path) as fh:
            rows = json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(rows, list) or not rows:
        return None
    return {
        "record": path,
        "n_cases": len(rows),
        "n_series_cached": cached,
        "patients": [r.get("patient_id") for r in rows],
        "selection": "the smallest DSC series by listed FileSize, fixed before outcomes",
        "endpoint_note": (
            "PFS is censored in every arm of this cohort (no pair progressed); "
            "the informative readout is the final burden, which differs between "
            "transport arms"
        ),
        "max_field_contrast": max(
            (
                arm.get("field_contrast_ratio", 0.0)
                for r in rows
                for arm in (r.get("engine", {}).get("arms", {}) or {}).values()
            ),
            default=0.0,
        ),
        "largest_burden_effect": max(
            (
                arm.get("max_abs_relative_burden_difference", 0.0)
                for r in rows
                for arm in (r.get("engine", {}).get("arms", {}) or {}).values()
            ),
            default=0.0,
        ),
    }


def _availability_manifest() -> Dict[str, Any]:
    """Counts from a cached availability manifest, if one is present.

    An availability manifest is per-(patient, timepoint) sequence flags: it is
    what makes stream 4 measurable at all, because it says which patients have
    two scannable timepoints. Read from the file rather than restated, and
    reported as an error if absent -- a census that silently omits what it
    could not read is the failure mode this whole module's honesty section is
    about.
    """
    try:
        from danon.availability import CFB_GBM_MANIFEST, mri_availability_census

        if not os.path.exists(CFB_GBM_MANIFEST):
            return {"error": f"no cached manifest at {CFB_GBM_MANIFEST}"}
        return mri_availability_census().to_json_dict()
    except (FileNotFoundError, ImportError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _histology_evidence(data_dir: str = "data") -> Optional[Dict[str, Any]]:
    """The committed whole-slide probe, plus the slides cached on disk.

    Read from the record rather than restated, and paired with a live count of
    cached slide files: a probe whose slide has been deleted is a number with
    nothing behind it. Returns None when the record is absent or unreadable, so
    the census can say that instead of implying a measurement.
    """
    import json

    path = os.path.join("outputs_real", "histology_probe.json")
    cached = 0
    root = os.path.join(data_dir, "tcia", "wsiprode")
    if os.path.isdir(root):
        cached = sum(
            1 for n in os.listdir(root) if n.lower().endswith((".svs", ".tif", ".tiff"))
        )
    try:
        with open(path) as fh:
            record = json.load(fh)
    except (OSError, ValueError):
        return None
    slide = record.get("slide_measured") or {}
    census = record.get("census") or {}
    density = slide.get("density") or {}
    stain = slide.get("stain") or {}
    if not density or not stain:
        return None
    return {
        "record": path,
        "n_slides_cached": cached,
        "n_slide_images_retrievable": census.get("n_slide_images"),
        "n_named_ki67": census.get("n_named_ki67"),
        "n_named_ihc": census.get("n_named_ihc"),
        "slide": {
            "file": slide.get("file"),
            "bytes": slide.get("bytes"),
            "pixels": (slide.get("slide") or {}).get("pixels"),
            "mpp_um": (slide.get("slide") or {}).get("mpp_um"),
            "app_mag": (slide.get("slide") or {}).get("app_mag"),
        },
        "stain": stain.get("stain"),
        "brown_fraction": stain.get("brown_fraction"),
        "nuclei_per_mm2": density.get("nuclei_per_mm2"),
        "nuclei": density.get("nuclei"),
        "crops_used": density.get("crops_used"),
        "per_crop_spread": density.get("per_crop_spread"),
        "median_nucleus_area_um2": density.get("median_nucleus_area_um2"),
        "measured": density.get("measures"),
        "ki67_refusal": (slide.get("ki67_refusal") or "").split(":")[0] or None,
        "selection": record.get("selection"),
    }


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
    # Slides sitting in the intake directories -- what the mapping could consume
    # today. The probe's downloaded slide lives under data/tcia/wsiprode and is
    # counted separately as a SAMPLE (see _histology_evidence): it is
    # haematoxylin-only, so it is not a Ki-67 input and must not flip stream 2
    # to 'available'.
    wsi_files = [
        f
        for root in (os.path.join(data_dir, "pathology"), data_dir)
        if os.path.isdir(root)
        for f in os.listdir(root)
        if f.lower().endswith((".svs", ".ndpi", ".tiff"))
    ]
    # Cached pairs: BraTS is one pre-operative timepoint per case, so zero.
    # A manifest does not change that -- it enumerates what a download WOULD
    # yield, and the count below is the sequence-level version of the same
    # question the probe answered from scan dates alone.
    sequential_pairs_cached = 0
    archive = _archive_availability()
    perfusion_evidence = _perfusion_cohort_evidence(data_dir)
    histology_evidence = _histology_evidence(data_dir)
    n_retrievable = archive.get("n_patients_longitudinal")
    manifest = _availability_manifest()
    n_manifest_pairs = manifest.get("n_patients_with_usable_pair")
    resolution = _stream4_resolution()
    _res = resolution.get("resolution") or {}
    floor_se = _res.get("floor_optimistic")
    floor_fold = _res.get("floor_conservative")
    floor_txt = (
        f"{floor_se:.4f} (1 SE) / {floor_fold:.4f} (fold yardstick)"
        if isinstance(floor_se, float) and isinstance(floor_fold, float)
        else "not computed (the design calculation could not be read)"
    )

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
            "refused_for": "the Ki-67 ratio (r0 = baseline * Ki67/KI67_REF)",
            "counts": (
                "n_files is slides in the intake directories (inputs); "
                "wsi.n_slides_cached is the probe's downloaded sample, which is "
                "not a Ki-67 input"
            ),
            "measurable": (
                "nuclei density per mm^2 -- CELLULARITY, which is not a "
                "proliferation index and does not enter r0"
            ),
            "wsi": histology_evidence if histology_evidence is not None else {
                "error": "histology probe record unreadable (run _histology_probe.py)"
            },
            # Where the measured density goes. The numbers live in the probe and
            # cohort records and in the bound claim; the census only points.
            "capacity": {
                "mapping": "stream2_carrying_capacity (relative; mean-one convention)",
                "absolute_ceiling": (
                    "REFUSED -- a whole-slide image counts nuclei per unit AREA "
                    "and these slides record no section thickness to divide by"
                ),
                "cohort_record": (
                    CAPACITY_COHORT_PATH if os.path.exists(CAPACITY_COHORT_PATH) else None
                ),
            },
            "detail": (
                "the Ki-67 numerator does not exist in this source, so r0 is "
                "refused and the mapping stays unit-tested (KI67_REF=0.15) "
                "until a real Ki-67 slide is supplied. What the archive DOES "
                "hold is whole-slide images: "
                f"{(histology_evidence or {}).get('n_slide_images_retrievable')} "
                "TCGA-GBM slide images at the live GDC count, "
                f"{(histology_evidence or {}).get('n_named_ki67')} of them named "
                "as Ki-67. The smallest was downloaded and read: "
                f"AppMag {(histology_evidence or {}).get('slide', {}).get('app_mag')}, "
                f"MPP {(histology_evidence or {}).get('slide', {}).get('mpp_um')} "
                f"um/px, {(histology_evidence or {}).get('stain')} (brown fraction "
                f"{(histology_evidence or {}).get('brown_fraction')}), "
                f"{(histology_evidence or {}).get('nuclei_per_mm2')} nuclei/mm^2 "
                f"over {(histology_evidence or {}).get('crops_used')} crops. "
                "A density is not a labelling index: substituting it for the "
                "Ki-67 fraction would fabricate the growth rate it feeds."
            ),
        },
        "stream3_perfusion": {
            "n_t1ce_files": len(t1ce_files),
            "n_seg_files": len(seg_files),
            "n_buildable_cases": len(set(t1ce_files) & {s.replace('_seg', '_t1ce') for s in seg_files}),
            "status": "MEASURABLE" if t1ce_files and seg_files else "REFUSED",
            "n_archive_dsc_patients": archive.get("n_patients_dsc_perfusion"),
            "n_archive_processed_captk_patients": archive.get(
                "n_patients_processed_perfusion"
            ),
            "archive_processed_series_are": (
                "CaPTk-processed STRUCTURAL volumes (verified by download), not "
                "rCBV/Ktrans maps; the census key keeps its legacy name"
            ),
            "dsc": perfusion_evidence if perfusion_evidence is not None else {
                "error": "perfusion cohort record unreadable"
            },
            "detail": (
                "real cached data: BraTS-2021 t1ce contrast volumes + expert "
                "segmentations -- a vascular proxy, not measured perfusion. The "
                "MEASURED arm is now separate and real: raw DSC series from the "
                f"archive ({archive.get('n_patients_dsc_perfusion')} patients "
                "carry one; a series downloads and was read), turned into a "
                "relative-CBV map by `danon/perfusion.py` and measured on "
                f"{(perfusion_evidence or {}).get('n_cases', 0)} downloaded "
                "series. Do not confuse the archive's 628 CaPTk-processed "
                "series with perfusion: those are co-registered structural "
                "volumes."
            ),
        },
        "stream4_sequential": {
            "n_pairs_cached": sequential_pairs_cached,
            "n_pairs_retrievable": n_retrievable,
            "status": "REFUSED" if sequential_pairs_cached == 0 else "available",
            "archive": archive,
            "manifest": manifest,
            "n_patients_with_usable_pair_in_manifest": n_manifest_pairs,
            "resolution": resolution,
            "resolvable_delta_c": resolution.get("resolution", {}).get(
                "floor_optimistic"
            ),
            "detail": (
                "no sequential scan pairs are CACHED (BraTS is a single "
                "pre-operative timepoint per case). Two independent sources say "
                "how many exist: the probed archive supports "
                f"{n_retrievable} UPENN-GBM patients with a "
                f">={archive.get('min_interval_days')}-day follow-up interval, "
                "and the cached CFB-GBM availability manifest records "
                f"{n_manifest_pairs} patients with a usable "
                f"{manifest.get('reference_sequence')} pair at least "
                f"{manifest.get('min_interval_weeks')} weeks apart "
                f"({manifest.get('n_usable_pairs')} pairs). Neither archive is "
                "downloaded, so the mapping -- implemented and unit-tested as "
                "an observation -- has nothing to run on yet. Note that an "
                "OBSERVATION cannot move a C-index at all: stream 4 would need "
                "a declared hazard mapping before it could be scored, and at "
                "the manifest's n such a mapping would have to clear a floor of "
                f"{floor_txt} "
                "to be distinguishable from cohort noise"
            ),
        },
    }
