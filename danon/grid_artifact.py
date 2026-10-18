"""What the engine's fixed grid does to a real tumour, measured.

WHY THIS MODULE EXISTS
----------------------
The engine runs on a uniform grid and has no adaptive refinement, so a clinical
segmentation (BraTS is 240x240x155 at 1 mm) is reduced to an engine grid before
anything is simulated. Everything the twin says about SPACE is said on that
coarser grid. Until now that reduction was described ("bbox-crop + majority-vote
decimation, lossy") and never measured. This module measures it in three
independent ways that do not depend on each other:

1. THE RULE, AGAINST THE RULE IT SAYS IT IS.
   `danon.realdata._decimate_labels` documents itself as a per-engine-voxel
   majority vote over the fx*fy*fz source voxels of an axis-aligned block. Its
   reshape chain groups the source volume differently, so this module computes
   the AXIS-ALIGNED block vote from first principles and compares the two
   arrays voxel for voxel. The disagreement is the measurement.

2. THE GEOMETRY EACH RULE PRODUCES.
   Physical tumour volume at each grid level against the source segmentation's
   own 1 mm volume. Majority vote is a DEFLATING rule (a block stays tumour only
   if more than half its source voxels were tumour), so the direction and the
   size of the bias are measured rather than asserted. The alternative rule --
   `danon.twin_plan.downsample_mask_to_budget`, the block-MAX pool that
   `api_server` applies to an oversize upload -- is measured on the same blocks;
   the voxels where the two rules disagree are exactly the voxels where
   max-pooling merges tumour into the healthy tissue around it.

3. THE OUTCOME.
   A geometry bias only matters if it moves a trial. The same real tumour is
   discretized at three levels on the AXIS-ALIGNED grid and run under both arms,
   with the physical diffusivity held fixed across levels and also left at the
   engine's declared voxel-unit value, and the endpoint drift is reported
   against the cohort's own seed-to-seed spread -- the same "is the artifact
   bigger than the noise?" yardstick `tests/test_benchmarks.py` uses for the
   clearance bias.

WHAT THIS MODULE DOES NOT DO
----------------------------
- It cannot make the artifact disappear. Adaptive refinement does not exist
  here; this module makes the fixed grid's cost visible and bounded.
- It does not recalibrate anything. The absolute value of `DEFAULT_DRUGS`
  remains a declared assumption (see `danon.evosteer`); what is measured here is
  how an outcome changes when the same declared constants are applied to
  different grids.
- The segmentation is real BraTS-2021 expert annotation, but the proliferative
  map derived from it is label-based (enhancing 1.0 / edema 0.6 / necrosis 0.2),
  so the spatial detail being discarded is radiological, not biological.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.evosteer import (
    DEFAULT_DRUGS,
    TumorEvolutionSimulator,
    TumorGeometry,
)
from danon.imaging import read_nifti
from danon.realdata import (
    DecimationSpec,
    _bbox,
    _decimate_labels,
    _label_proliferation,
)
from danon.uncertainty import ARM_POLICIES

#: The engine's declared diffusivity (voxel^2/day) and the spacing its own
#: docstring anchors it to. Used only to convert the constant between the
#: engine's voxel units and mm^2/day when comparing grids.
DECLARED_DIFFUSION_VOXEL2_PER_DAY = float(DEFAULT_DRUGS.diffusion[0])
DECLARED_DIFFUSION_ANCHOR_MM = 1.0

#: The discretizations every statement in this module is measured on.
#: `engine_3p6mm` is the SHIPPED grid: `DecimationSpec.factors` default.
GRID_LEVELS: Mapping[str, Tuple[int, int, int]] = {
    "fine_2p0mm": (2, 2, 2),
    "engine_3p6mm": (4, 4, 3),
    "coarse_7p3mm": (8, 8, 6),
}

#: The level the shipped ingestion path produces (`DecimationSpec()` default).
SHIPPED_LEVEL = "engine_3p6mm"

#: Where the shipped derived volumes live, for the round-trip audit.
DERIVED_DIR = os.path.join("data", "engine_ready")
SOURCE_SEG_DIR = os.path.join("data", "brats")

#: Both arms of the paired trial, and the seeds every configuration shares (the
#: same seed index is the same clustered subclone at every grid level).
COHORT_SEEDS: Tuple[int, ...] = (101, 102, 103, 104, 105, 106)
COHORT_DAYS = 365
COHORT_R0 = 0.01

#: The two stochastic regimes the endpoint drift is measured in.
#: `shipped_noise` is the engine default; `noise_off` removes the per-voxel
#: growth field, so the surviving spread is the initial-condition spread alone
#: and cannot be confused with the noise field's own resolution.
OUTCOME_MODES: Mapping[str, float] = {"shipped_noise": 0.05, "noise_off": 0.0}

#: The endpoint readouts contrasted across grids.
READOUTS: Tuple[str, ...] = ("pfs", "mean_density", "final_resistant_fraction")


# ---------------------------------------------------------------------------
# The two partitions
# ---------------------------------------------------------------------------


def _block_grid(factors: Tuple[int, int, int], shape: Sequence[int]) -> Tuple[int, int, int]:
    fx, fy, fz = factors
    sx, sy, sz = (int(v) for v in shape)
    nx, ny, nz = sx // fx, sy // fy, sz // fz
    if min(nx, ny, nz) == 0:
        raise ValueError(f"decimation {factors} collapses grid {tuple(shape)}")
    return nx, ny, nz


def block_decimation(
    labels: np.ndarray,
    factors: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """    AXIS-ALIGNED block majority vote over the fx*fy*fz source voxels.

    Engine voxel (i, j, k) owns source voxels
    (i*fx .. i*fx+fx-1, j*fy .., k*fz ..): the partition the shipped rule's
    docstring describes. Two decisions are kept separate, because collapsing
    them loses tumour:

      * IS this block tumour? -- yes iff a STRICT majority of the fx*fy*fz
        contributors were tumour. That is the whole of the mask rule.
      * WHICH tumour label does it carry? -- the majority among the tumour
        contributors only (ties toward the lower label).

    A single argmax over {0, 1, 2, 4} answers the second question and then gets
    used for the first, which mislabels a block whose background count TIES the
    count of one tumour label (5 tumour of 8 as 3 background + 2 enhancing): the
    argmax goes to background and a majority-tumour block is dropped.

    Returns (majority_mask, labels_grid, maxpool_mask); the max-pool arm uses
    the identical blocks, so a difference between the two rules is the rule and
    nothing else.
    """
    fx, fy, fz = factors
    nx, ny, nz = _block_grid(factors, labels.shape)
    trimmed = labels[: nx * fx, : ny * fy, : nz * fz]
    # (nx, ny, nz, fx, fy, fz): the block axes MUST be trailing, otherwise the
    # flat reinterpretation groups contiguous runs instead of cubes.
    blocks = (
        trimmed.reshape(nx, fx, ny, fy, nz, fz)
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(nx, ny, nz, -1)
    )
    n_contrib = blocks.shape[-1]
    tumour_votes = np.stack([(blocks == L).sum(axis=-1) for L in (1, 2, 4)], axis=-1)
    is_tumour = tumour_votes.sum(axis=-1) * 2 > n_contrib
    dominant = np.array([1, 2, 4])[np.argmax(tumour_votes, axis=-1)]
    labels_grid = np.where(is_tumour, dominant, 0).astype(np.int32)
    return labels_grid > 0, labels_grid, (blocks > 0).any(axis=-1)


def shipped_decimation(
    labels: np.ndarray, factors: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Whatever `danon.realdata._decimate_labels` produces today.

    Kept as a call into the shipped function rather than a copy, so the
    comparison in `measure_level` cannot silently drift from the code under
    audit: if the shipped function changes, the measured disagreement changes.
    """
    return _decimate_labels(labels, DecimationSpec(factors=factors))


def partition_probe(
    shape: Sequence[int],
    factors: Tuple[int, int, int],
    *,
    n_samples: int = 200,
    seed: int = 0,
) -> Dict[str, Any]:
    """Which source voxels the shipped rule groups together, one block at a time.

    The array comparison in `measure_level` cannot separate two different
    failures: a wrong PARTITION, and a labelling convention that answers 'which
    label?' with a background count that ties a tumour label and then uses that
    answer for 'is this tumour?'. This probe isolates the partition, because it
    puts the same label on every voxel of an axis-aligned block:

      fill one fx*fy*fz cube with label 4, ask the shipped rule where it lands.

    Under an axis-aligned partition the answer is the cube's own engine voxel.
    Anything else is the grouping, with no tie rule involved: a single labelled
    voxel would be dropped by the strict-majority rule whatever the partition.
    """
    fx, fy, fz = factors
    nx, ny, nz = _block_grid(factors, shape)
    rng = np.random.default_rng(seed)
    n = min(int(n_samples), nx * ny * nz)
    picks = rng.choice(nx * ny * nz, size=n, replace=False)
    landed_aligned = 0
    landed_elsewhere = 0
    lost = 0
    max_displacement = 0
    for flat in picks:
        i, j, k = np.unravel_index(int(flat), (nx, ny, nz))
        probe = np.zeros(tuple(int(v) for v in shape), dtype=np.int32)
        probe[
            i * fx : (i + 1) * fx,
            j * fy : (j + 1) * fy,
            k * fz : (k + 1) * fz,
        ] = 4
        out, _ = shipped_decimation(probe, factors)
        owners = np.argwhere(out > 0)
        if len(owners) == 1 and tuple(int(v) for v in owners[0]) == (i, j, k):
            landed_aligned += 1
        elif len(owners) == 1:
            landed_elsewhere += 1
            o = owners[0]
            max_displacement = max(
                max_displacement,
                int(max(abs(int(o[0]) - i), abs(int(o[1]) - j), abs(int(o[2]) - k))),
            )
        else:
            lost += 1
    return {
        "factors": list(factors),
        "shape": [int(v) for v in shape],
        "grid": [nx, ny, nz],
        "n_probed": int(n),
        "n_landed_in_their_own_engine_voxel": int(landed_aligned),
        "share_landed": float(landed_aligned / n) if n else float("nan"),
        "n_landed_elsewhere": int(landed_elsewhere),
        "n_dropped_entirely": int(lost),
        "max_displacement_in_engine_voxels": int(max_displacement),
        "is_a_block_partition": bool(n and landed_aligned == n),
    }


def _upsample_nearest(mask: np.ndarray, factors: Tuple[int, int, int]) -> np.ndarray:
    """Exact inverse of the axis-aligned partition, for a per-source-voxel view."""
    fx, fy, fz = factors
    return np.repeat(np.repeat(np.repeat(np.asarray(mask, bool), fx, 0), fy, 1), fz, 2)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return float("nan")
    return 2.0 * float(np.logical_and(a, b).sum()) / denom


def _boundary_voxels(mask: np.ndarray) -> int:
    """Voxels of `mask` with at least one face neighbour outside the tumour.

    Counted in voxels, never converted to mm^2: on an anisotropic grid a
    boundary-voxel count is not a surface area, and inventing one would be
    exactly the kind of number this module exists to avoid. Voxels on the array
    border count as boundary (their outside neighbour is not tumour).
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 3:
        raise ValueError(f"mask must be 3D, got shape {m.shape}")
    padded = np.pad(m, 1, constant_values=False)
    interior = np.ones_like(m, dtype=bool)
    for axis in range(3):
        for delta in (-1, 1):
            sl = [slice(1, -1)] * 3
            sl[axis] = slice(1 + delta, padded.shape[axis] - 1 + delta)
            interior &= padded[tuple(sl)]
    return int(m.sum() - np.logical_and(m, interior).sum())


# ---------------------------------------------------------------------------
# Measurement 1+2 -- the two rules, and the geometry each produces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LevelGeometry:
    """One discretization of one real tumour, with its measured distortion."""

    name: str
    factors: Tuple[int, int, int]
    input_grid: Tuple[int, int, int]
    grid: Tuple[int, int, int]
    source_spacing_mm: Tuple[float, float, float]
    decimated_spacing_mm: Tuple[float, float, float]
    #: Volume-equivalent isotropic edge: (fx*fy*fz)^(1/3) * source spacing.
    voxel_mm: float
    voxel_mm3: float
    source_volume_mm3: float

    # -- the axis-aligned block vote: the rule the docstring describes -------
    tumour_voxels: int
    volume_mm3: float
    volume_bias: float
    dice_vs_source: float
    boundary_voxels: int
    boundary_fraction: float
    maxpool_tumour_voxels: int
    maxpool_volume_mm3: float
    maxpool_volume_bias: float
    merged_voxels: int
    merged_fraction_of_tumour: float

    # -- what the shipped rule produces on the same source volume -----------
    shipped_tumour_voxels: int
    shipped_volume_mm3: float
    shipped_volume_bias: float
    #: Overlap between the shipped rule's occupancy and the axis-aligned
    #: block vote's, on the engine grid. 1.0 means the shipped function is
    #: doing what it says it does.
    shipped_dice_vs_aligned: float
    shipped_agrees_voxelwise: bool
    shipped_disagreeing_voxels: int

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, tuple):
                out[key] = [float(v) if isinstance(v, float) else int(v) for v in value]
            else:
                out[key] = value
        return out


def measure_level(
    labels: np.ndarray,
    source_spacing_mm: Sequence[float],
    factors: Tuple[int, int, int],
    name: str,
    *,
    source_volume_mm3: float,
) -> LevelGeometry:
    """Decimate a cropped label volume by both rules and measure both."""
    aligned, _, maxpool = block_decimation(labels, factors)
    shipped, _ = shipped_decimation(labels, factors)
    if shipped.shape != aligned.shape:
        raise ValueError(
            f"the shipped decimation produced grid {shipped.shape} but the "
            f"axis-aligned one produced {aligned.shape}; the two rules are no "
            "longer comparable and the audit must be rewritten, not patched"
        )

    fx, fy, fz = factors
    spacing = tuple(float(s) for s in source_spacing_mm)
    decimated = tuple(s * f for s, f in zip(spacing, (fx, fy, fz)))
    voxel_mm3 = float(np.prod(decimated))

    tumour = int(aligned.sum())
    shipped_tumour = int(shipped.sum())
    boundary = _boundary_voxels(aligned)
    merged = int(np.logical_and(maxpool, ~aligned).sum())

    nx, ny, nz = _block_grid(factors, labels.shape)
    trimmed = labels[: nx * fx, : ny * fy, : nz * fz]
    truth = trimmed > 0
    upsampled = _upsample_nearest(aligned, factors)
    disagreeing = int(np.logical_xor(aligned, shipped).sum())

    return LevelGeometry(
        name=name,
        factors=(int(fx), int(fy), int(fz)),
        input_grid=tuple(int(v) for v in labels.shape),  # type: ignore[assignment]
        grid=tuple(int(v) for v in aligned.shape),  # type: ignore[assignment]
        source_spacing_mm=spacing,  # type: ignore[arg-type]
        decimated_spacing_mm=decimated,  # type: ignore[arg-type]
        voxel_mm=float(voxel_mm3 ** (1.0 / 3.0)),
        voxel_mm3=voxel_mm3,
        source_volume_mm3=float(source_volume_mm3),
        tumour_voxels=tumour,
        volume_mm3=tumour * voxel_mm3,
        volume_bias=(tumour * voxel_mm3 / source_volume_mm3 - 1.0)
        if source_volume_mm3
        else float("nan"),
        dice_vs_source=_dice(upsampled, truth),
        boundary_voxels=boundary,
        boundary_fraction=(boundary / tumour if tumour else float("nan")),
        maxpool_tumour_voxels=int(maxpool.sum()),
        maxpool_volume_mm3=float(int(maxpool.sum()) * voxel_mm3),
        maxpool_volume_bias=(int(maxpool.sum()) * voxel_mm3 / source_volume_mm3 - 1.0)
        if source_volume_mm3
        else float("nan"),
        merged_voxels=merged,
        merged_fraction_of_tumour=(merged / tumour if tumour else float("nan")),
        shipped_tumour_voxels=shipped_tumour,
        shipped_volume_mm3=shipped_tumour * voxel_mm3,
        shipped_volume_bias=(shipped_tumour * voxel_mm3 / source_volume_mm3 - 1.0)
        if source_volume_mm3
        else float("nan"),
        shipped_dice_vs_aligned=_dice(aligned, shipped),
        shipped_agrees_voxelwise=bool(disagreeing == 0),
        shipped_disagreeing_voxels=disagreeing,
    )


def decimation_geometry(
    seg_path: str,
    *,
    levels: Mapping[str, Tuple[int, int, int]] = GRID_LEVELS,
) -> Dict[str, Any]:
    """Measure both decimation rules on a real BraTS segmentation.

    The source volume is the segmentation's own voxel count times its measured
    header spacing -- no conversion constant is introduced.
    """
    labels, hdr = read_nifti(seg_path)
    labels = np.rint(np.asarray(labels)).astype(np.int32)
    if not any(u in np.unique(labels) for u in (1, 2, 4)):
        raise ValueError(f"{seg_path}: no BraTS tumour labels (1/2/4) present")
    source_spacing = tuple(float(abs(v)) for v in hdr.voxel_sizes)
    source_voxels = int((labels > 0).sum())
    source_volume = float(source_voxels * np.prod(source_spacing))
    bb = _bbox(labels > 0)
    cropped = labels[bb]

    out: Dict[str, Any] = {
        "seg_path": seg_path,
        "source_grid": [int(v) for v in labels.shape],
        "source_spacing_mm": list(source_spacing),
        "source_tumour_voxels": source_voxels,
        "source_volume_mm3": source_volume,
        "bbox_stop_xyz": [int(s.stop) for s in bb],
        "levels": {},
    }
    for name in levels:
        level = measure_level(
            cropped,
            source_spacing,
            levels[name],
            name,
            source_volume_mm3=source_volume,
        )
        out["levels"][name] = level.as_dict()

    if SHIPPED_LEVEL in levels:
        out["partition_probe"] = partition_probe(cropped.shape, levels[SHIPPED_LEVEL])
    shipped = out["levels"][SHIPPED_LEVEL] if SHIPPED_LEVEL in out["levels"] else None
    out["shipped_level"] = SHIPPED_LEVEL
    out["shipped_rule_is_the_axis_aligned_block_vote"] = (
        bool(shipped["shipped_agrees_voxelwise"]) if shipped else None
    )
    out["shipped_rule_dice_vs_axis_aligned"] = (
        float(shipped["shipped_dice_vs_aligned"]) if shipped else None
    )
    return out


# ---------------------------------------------------------------------------
# Measurement 3 -- does the discretization move a trial?
# ---------------------------------------------------------------------------


def matched_diffusion(
    voxel_mm: float,
    *,
    d_declared_voxel2_per_day: float = DECLARED_DIFFUSION_VOXEL2_PER_DAY,
    anchor_mm: float = DECLARED_DIFFUSION_ANCHOR_MM,
) -> float:
    """Voxel-unit diffusivity that preserves the DECLARED physical diffusivity.

    The engine's constant is documented as voxel^2/day at `anchor_mm`, so the
    physical value is `d * anchor_mm^2`, and a grid at `voxel_mm` reproduces it
    with `d * (anchor_mm / voxel_mm)^2`.

    Two honest caveats. (i) The engine integrates `div(D grad C)` with an
    INDEX-space Laplacian, so it treats every grid as isotropic in index units;
    for the shipped (4, 4, 3) factors no scalar D is simultaneously right along
    all three axes, and `voxel_mm` here is the volume-equivalent isotropic edge.
    (ii) Nothing in the shipped pipeline rescales D when it decimates, which is
    precisely what the `declared_D` configurations measure.
    """
    if voxel_mm <= 0.0:
        raise ValueError(f"voxel_mm must be positive, got {voxel_mm}")
    return d_declared_voxel2_per_day * (anchor_mm / float(voxel_mm)) ** 2


def physical_diffusion_mm2_per_day(d_voxel: float, voxel_mm: float) -> float:
    """The same constant read in physical units: d_voxel * voxel_mm^2."""
    return float(d_voxel) * float(voxel_mm) ** 2


def _geometry_at_level(
    cropped_labels: np.ndarray,
    factors: Tuple[int, int, int],
    decimated_spacing_mm: Sequence[float],
) -> TumorGeometry:
    """The engine geometry the ingestion path INTENDS to build at this level.

    Uses the axis-aligned block vote (`block_decimation`) plus the shipped
    proliferative mapping (`_label_proliferation`); what the shipped grouping
    rule does to the same source volume is reported separately by
    `decimation_geometry`. Refuses rather than falling back to the distance
    proxy: a level that lost every enhancing voxel would silently change the
    proliferative rule between configurations and confound a grid comparison
    with a geometry change.
    """
    mask, labels_grid, _ = block_decimation(cropped_labels, factors)
    if int(mask.sum()) == 0:
        raise ValueError(f"decimation {factors} produced an empty tumour mask")
    prolif, from_labels = _label_proliferation(labels_grid, mask)
    if not from_labels:
        raise ValueError(
            f"decimation {factors} retained no enhancing voxels, so the "
            "proliferative map would change rule between grid levels; refusing "
            "to compare grids with different geometry"
        )
    # Null-uniform vasculature, exactly as the shipped engine-ready cases carry
    # (BraTS ships no perfusion map), held identical across levels by
    # construction so it cannot contribute to a grid difference.
    vessels = np.where(mask, 0.5, 0.0)
    return TumorGeometry(
        tumor_mask=mask,
        proliferative=prolif,
        vessels=vessels,
        voxel_mm=float(np.mean(decimated_spacing_mm)),
    )


def _run_one(
    geom: TumorGeometry,
    d_voxel: float,
    arm: str,
    seed: int,
    *,
    days: int,
    r0: float,
    growth_noise: float,
) -> Dict[str, Any]:
    drugs = type(DEFAULT_DRUGS)(
        **{**DEFAULT_DRUGS.__dict__, "diffusion": (d_voxel, d_voxel)}
    )
    sim = TumorEvolutionSimulator(geom, drugs=drugs)
    traj = sim.run(
        ARM_POLICIES[arm](),
        days=days,
        initial_resistant_fraction=r0,
        seed=seed,
        growth_noise=growth_noise,
    )
    n_vox = int(geom.tumor_mask.sum())
    total = float(traj.burden_S[-1] + traj.burden_R[-1])
    return {
        "pfs": float(traj.pfs_value(days)),
        # Density per tumour voxel: the grid-invariant reading of burden.
        # Carrying capacity is per voxel and the seed is a FRACTION of it, so a
        # block-averaged coarse voxel carries the same density as the fine
        # voxels it averages, while a raw burden sum would differ by the voxel
        # volume ratio and look like a 8x effect that is not there.
        "mean_density": total / n_vox,
        "final_resistant_fraction": float(traj.final_resistant_fraction),
        "voxels": n_vox,
    }


def _spread(values: Sequence[float]) -> Dict[str, Any]:
    """Relative seed spread, or an explicit degenerate flag.

    A readout that is constant across seeds has no spread to compare against; it
    is reported as degenerate rather than as a zero, because "the artifact is 0%
    of the spread" and "there is no spread" are different statements.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        raise ValueError(f"a spread needs at least two runs, got {arr.size}")
    lo, hi, mean = float(arr.min()), float(arr.max()), float(arr.mean())
    if mean == 0.0 or hi == lo:
        return {"min": lo, "max": hi, "mean": mean, "spread_rel": 0.0, "degenerate": True}
    return {
        "min": lo,
        "max": hi,
        "mean": mean,
        "spread_rel": (hi - lo) / abs(mean),
        "degenerate": False,
    }


def grid_resolution_cohort(
    seg_path: str,
    *,
    levels: Mapping[str, Tuple[int, int, int]] = GRID_LEVELS,
    seeds: Sequence[int] = COHORT_SEEDS,
    days: int = COHORT_DAYS,
    r0: float = COHORT_R0,
    modes: Mapping[str, float] = OUTCOME_MODES,
    arms: Sequence[str] = ("mtd", "steering"),
) -> Dict[str, Any]:
    """Run both arms at every grid level and report the drift against the spread.

    Configurations, all anchored on the declared physical diffusivity
    (`DECLARED_DIFFUSION_VOXEL2_PER_DAY` at `DECLARED_DIFFUSION_ANCHOR_MM`):

      <level>_matched_D   the same physical D on every grid -- isolates the
                          discretization itself;
      <level>_declared_D  D left at the engine's voxel-unit value, which is what
                          the shipped pipeline does today -- isolates the
                          consequence of one constant meaning different physics
                          on different grids.
    """
    geom_report = decimation_geometry(seg_path, levels=levels)
    labels, _ = read_nifti(seg_path)
    labels = np.rint(np.asarray(labels)).astype(np.int32)
    cropped = labels[_bbox(labels > 0)]

    geoms: Dict[str, TumorGeometry] = {}
    spacing_by_level: Dict[str, Tuple[float, float, float]] = {}
    for name in levels:
        spacing = tuple(
            float(s) * f for s, f in zip(geom_report["source_spacing_mm"], levels[name])
        )
        spacing_by_level[name] = spacing  # type: ignore[assignment]
        geoms[name] = _geometry_at_level(cropped, levels[name], spacing)

    # The contrast reference is the FIRST level given, which for the shipped
    # mapping is the finest grid: the closest thing to the source segmentation
    # the engine can integrate.
    reference = next(iter(levels))

    configs: Dict[str, Dict[str, Any]] = {}
    for name in levels:
        voxel_mm = geom_report["levels"][name]["voxel_mm"]
        configs[f"{name}_matched_D"] = {
            "level": name,
            "d_voxel": matched_diffusion(voxel_mm),
            "rule": "physical diffusivity preserved across grids",
        }
        configs[f"{name}_declared_D"] = {
            "level": name,
            "d_voxel": DECLARED_DIFFUSION_VOXEL2_PER_DAY,
            "rule": "engine constant used as-is (what the shipped pipeline does)",
        }
    for cfg in configs.values():
        voxel_mm = geom_report["levels"][cfg["level"]]["voxel_mm"]
        cfg["physical_d_mm2_per_day"] = physical_diffusion_mm2_per_day(
            cfg["d_voxel"], voxel_mm
        )

    runs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for mode, noise in modes.items():
        runs[mode] = {}
        for cfg_name, cfg in configs.items():
            runs[mode][cfg_name] = {}
            for arm in arms:
                per_seed = [
                    _run_one(
                        geoms[cfg["level"]],
                        cfg["d_voxel"],
                        arm,
                        int(s),
                        days=days,
                        r0=r0,
                        growth_noise=noise,
                    )
                    for s in seeds
                ]
                runs[mode][cfg_name][arm] = {
                    "seeds": [int(s) for s in seeds],
                    "readouts": {
                        metric: [r[metric] for r in per_seed] for metric in READOUTS
                    },
                    "voxels": per_seed[0]["voxels"],
                }

    spreads: Dict[str, Any] = {}
    for mode in modes:
        spreads[mode] = {}
        for cfg_name in configs:
            spreads[mode][cfg_name] = {}
            for arm in arms:
                spreads[mode][cfg_name][arm] = {
                    metric: _spread(runs[mode][cfg_name][arm]["readouts"][metric])
                    for metric in READOUTS
                }

    contrasts: Dict[str, Any] = {}
    for mode in modes:
        contrasts[mode] = {}
        ref = f"{reference}_matched_D"
        for cfg_name in configs:
            if cfg_name == ref:
                continue
            contrasts[mode][cfg_name + "_vs_" + ref] = {}
            for arm in arms:
                entry: Dict[str, Any] = {}
                for metric in READOUTS:
                    base = np.asarray(runs[mode][ref][arm]["readouts"][metric], float)
                    cand = np.asarray(runs[mode][cfg_name][arm]["readouts"][metric], float)
                    delta = cand - base
                    scale = abs(float(base.mean())) if base.mean() else 0.0
                    spread_rel = spreads[mode][ref][arm][metric]["spread_rel"]
                    entry[metric] = {
                        "base": base.tolist(),
                        "candidate": cand.tolist(),
                        "mean_delta": float(delta.mean()),
                        "mean_rel_delta": float(delta.mean() / scale) if scale else None,
                        "max_abs_delta": float(np.abs(delta).max()),
                        "seeds_moving_the_same_way": int(
                            max(int((delta > 0).sum()), int((delta < 0).sum()))
                        ),
                        "n_seeds": int(delta.size),
                        "reference_spread_rel": spread_rel,
                        "reference_spread_degenerate": spreads[mode][ref][arm][metric][
                            "degenerate"
                        ],
                        "delta_over_spread": (
                            float(abs(delta.mean()) / scale / spread_rel)
                            if scale and spread_rel
                            else None
                        ),
                    }
                contrasts[mode][cfg_name + "_vs_" + ref][arm] = entry

    return {
        "seg_path": seg_path,
        "geometry": geom_report,
        "days": days,
        "r0": r0,
        "seeds": [int(s) for s in seeds],
        "modes": dict(modes),
        "configs": configs,
        "reference_config": f"{reference}_matched_D",
        "level_spacing_mm": {k: list(v) for k, v in spacing_by_level.items()},
        "runs": runs,
        "spreads": spreads,
        "contrasts": contrasts,
        "declared_diffusion_voxel2_per_day": DECLARED_DIFFUSION_VOXEL2_PER_DAY,
        "declared_diffusion_anchor_mm": DECLARED_DIFFUSION_ANCHOR_MM,
    }


# ---------------------------------------------------------------------------
# The round-trip audit -- what a committed file actually holds
# ---------------------------------------------------------------------------


def derived_round_trip(
    seg_path: str,
    derived_mask_path: str,
    *,
    factors: Tuple[int, int, int] = GRID_LEVELS[SHIPPED_LEVEL],
) -> Dict[str, Any]:
    """Read back a committed derived mask and check BOTH geometry claims.

    Two independent things can be wrong with a derived engine-ready volume, and
    both are silent because neither changes the voxel count:

    (a) the array can be a different partition of the source than the axis-
        aligned block vote the pipeline documents (the array is wrong);
    (b) the header can keep the SOURCE spacing instead of the decimated spacing,
        so any physical volume read back from it is off by the product of the
        factors (the header is wrong).

    This measures both, using the source segmentation's own voxel volume as the
    reference. No constant is introduced and no case is special-cased.
    """
    src_labels, src_hdr = read_nifti(seg_path)
    src_labels = np.rint(np.asarray(src_labels)).astype(np.int32)
    source_spacing = tuple(float(abs(v)) for v in src_hdr.voxel_sizes)
    source_volume = float((src_labels > 0).sum() * np.prod(source_spacing))
    bb = _bbox(src_labels > 0)
    level = measure_level(
        src_labels[bb],
        source_spacing,
        factors,
        "round_trip",
        source_volume_mm3=source_volume,
    )
    aligned, _, _ = block_decimation(src_labels[bb], factors)

    mask, mhdr = read_nifti(derived_mask_path)
    stored = np.asarray(mask) > 0
    header_spacing = tuple(float(abs(v)) for v in mhdr.voxel_sizes)
    if stored.shape != aligned.shape:
        raise ValueError(
            f"{derived_mask_path} holds grid {stored.shape} but the source "
            f"decimates to {aligned.shape}; the file is not this case's "
            "engine grid at these factors"
        )
    voxels = int(stored.sum())
    reported = float(voxels * np.prod(header_spacing))
    traced = float(voxels * np.prod(level.decimated_spacing_mm))
    return {
        "seg_path": seg_path,
        "derived_mask_path": derived_mask_path,
        "factors": list(factors),
        "source_spacing_mm": list(source_spacing),
        "expected_spacing_mm": list(level.decimated_spacing_mm),
        "header_spacing_mm": list(header_spacing),
        "tumour_voxels": voxels,
        "axis_aligned_tumour_voxels": int(aligned.sum()),
        "stored_agrees_with_aligned_block_vote": bool(
            int(np.logical_xor(stored, aligned).sum()) == 0
        ),
        "stored_disagreeing_voxels": int(np.logical_xor(stored, aligned).sum()),
        "stored_dice_vs_aligned": _dice(stored, aligned),
        "reported_volume_mm3": reported,
        "traced_volume_mm3": traced,
        "source_volume_mm3": source_volume,
        "volume_report_factor": (traced / reported) if reported else None,
        "spacing_consistent": bool(
            np.allclose(header_spacing, level.decimated_spacing_mm, rtol=1e-6)
        ),
        "volume_bias_vs_source": (traced / source_volume - 1.0) if source_volume else None,
    }


def committed_derived_audit(
    *,
    derived_dir: str = DERIVED_DIR,
    seg_dir: str = SOURCE_SEG_DIR,
    cases: Optional[Sequence[str]] = None,
    factors: Tuple[int, int, int] = GRID_LEVELS[SHIPPED_LEVEL],
) -> Dict[str, Any]:
    """The round-trip audit over every committed case whose source is cached.

    Cases whose source segmentation is not in the repository are REPORTED as
    unaudited rather than skipped quietly: an audit that silently covers less
    than it claims is the failure mode this whole module is about.
    """
    if cases is None:
        cases = sorted(
            f[: -len("_mask.nii.gz")]
            for f in os.listdir(derived_dir)
            if f.endswith("_mask.nii.gz") and not f.endswith(".nii_mask.nii.gz")
        )
    entries: Dict[str, Any] = {}
    unaudited: List[str] = []
    for case in cases:
        seg = os.path.join(seg_dir, f"{case}_seg.nii.gz")
        derived = os.path.join(derived_dir, f"{case}_mask.nii.gz")
        if not (os.path.exists(seg) and os.path.exists(derived)):
            unaudited.append(case)
            continue
        entries[case] = derived_round_trip(seg, derived, factors=factors)
    return {
        "derived_dir": derived_dir,
        "factors": list(factors),
        "n_audited": len(entries),
        "unaudited_no_cached_source": unaudited,
        "cases": entries,
        "all_arrays_are_the_axis_aligned_block_vote": bool(
            entries and all(e["stored_agrees_with_aligned_block_vote"] for e in entries.values())
        ),
        "all_headers_carry_the_decimated_spacing": bool(
            entries and all(e["spacing_consistent"] for e in entries.values())
        ),
        "n_with_array_disagreement": sum(
            1 for e in entries.values() if not e["stored_agrees_with_aligned_block_vote"]
        ),
        "n_with_spacing_mismatch": sum(
            1 for e in entries.values() if not e["spacing_consistent"]
        ),
    }


# ---------------------------------------------------------------------------
# Rendering, provenance, refusals
# ---------------------------------------------------------------------------


def render_grid_artifact_report(report: Mapping[str, Any]) -> str:
    """A text rendering of `decimation_geometry` output."""
    lines = [
        "GRID ARTIFACT -- what the engine's fixed uniform grid does to a real tumour",
        "=" * 88,
        f"source segmentation : {report['seg_path']}",
        f"source grid         : {report['source_grid']} at {report['source_spacing_mm']} mm",
        f"source tumour       : {report['source_tumour_voxels']:,} voxels = "
        f"{report['source_volume_mm3']:,.1f} mm^3",
        "",
        f"{'level':15s} {'factors':>10s} {'grid':>14s} {'mm/vox':>7s} {'vote mm3':>11s} "
        f"{'bias':>9s} {'dice':>6s} {'maxpool bias':>13s} {'merged':>7s} "
        f"{'shipped':>9s} {'rule dice':>9s}",
    ]
    for name, lv in report["levels"].items():
        lines.append(
            f"{name:15s} {str(tuple(lv['factors'])):>10s} "
            f"{str(tuple(lv['grid'])):>14s} {lv['voxel_mm']:7.3f} "
            f"{lv['volume_mm3']:11,.1f} {lv['volume_bias'] * 100:+8.3f}% "
            f"{lv['dice_vs_source']:6.4f} {lv['maxpool_volume_bias'] * 100:+12.3f}% "
            f"{lv['merged_fraction_of_tumour'] * 100:6.2f}% "
            f"{lv['shipped_tumour_voxels']:9,d} {lv['shipped_dice_vs_aligned']:9.4f}"
        )
    lines += [
        "",
        "vote   : the axis-aligned fx*fy*fz block majority vote, computed from",
        "         first principles -- what `_decimate_labels` says it does.",
        "bias   : physical tumour volume that vote reports, vs the source",
        "         segmentation's own 1 mm volume. Negative = the vote DELETES",
        "         tumour; a strict-majority rule cannot inflate.",
        "dice   : overlap between the vote's occupancy (nearest-upsampled to the",
        "         source grid) and the source segmentation.",
        "maxpool: the alternative block-MAX rule (twin_plan.downsample_mask_to_",
        "         budget, used by api_server for oversize uploads); 'merged' is",
        "         the fraction of the vote's tumour that max-pooling adds by",
        "         labelling blocks where the majority said background.",
        "shipped: what `realdata._decimate_labels` actually returns today.",
        "rule   : Dice between the shipped rule and the axis-aligned vote on the",
        "         same blocks. 1.0 means the shipped function does what it says.",
    ]
    return "\n".join(lines)


def render_round_trip_report(audit: Mapping[str, Any]) -> str:
    """A text rendering of `committed_derived_audit` output."""
    lines = [
        "DERIVED-VOLUME ROUND TRIP -- what the committed engine-ready masks hold",
        "=" * 88,
        f"{'case':18s} {'voxels':>8s} {'vote voxels':>12s} {'dice':>7s} "
        f"{'disagree':>9s} {'header mm':>18s} {'vol factor':>10s}",
    ]
    for case, e in audit["cases"].items():
        factor = e["volume_report_factor"]
        lines.append(
            f"{case:18s} {e['tumour_voxels']:8,d} {e['axis_aligned_tumour_voxels']:12,d} "
            f"{e['stored_dice_vs_aligned']:7.4f} {e['stored_disagreeing_voxels']:9,d} "
            f"{str(tuple(e['header_spacing_mm'])):>18s} "
            f"{(f'{factor:.3f}x' if factor else 'n/a'):>10s}"
        )
    lines += [
        "",
        f"audited: {audit['n_audited']} case(s); unaudited (no cached source): "
        f"{audit['unaudited_no_cached_source'] or 'none'}",
        f"arrays that are NOT the axis-aligned block vote: "
        f"{audit['n_with_array_disagreement']}",
        f"headers that do NOT carry the decimated spacing: {audit['n_with_spacing_mismatch']}",
        "vol factor: traced volume / volume reported from the file's own header.",
        "            >1 means the file understates the tumour it holds.",
    ]
    return "\n".join(lines)


def grid_artifact_provenance() -> Dict[str, EvidenceStatus]:
    """What is established about this grid, and what is measurably not.

    `axis_aligned_decimation` is the verdict on the SHIPPED rule, and it is a
    measurement rather than a stylistic preference: `decimation_geometry`
    compares `realdata._decimate_labels` against the axis-aligned block vote its
    own docstring describes, voxel for voxel. It therefore has to move when that
    rule is repaired -- `claims.py::check_grid_artifact_is_measured` enforces the
    consistency in both directions so neither a stale PROVEN_FAILED nor a
    premature ESTABLISHED can survive.
    """
    return {
        "source_segmentations_are_expert_brats_labels": EvidenceStatus.ESTABLISHED,
        "decimation_partition_is_deterministic": EvidenceStatus.ESTABLISHED,
        "physical_tumour_volume_at_each_grid": EvidenceStatus.ESTABLISHED,
        "outcome_drift_across_grids": EvidenceStatus.ESTABLISHED,
        "axis_aligned_decimation": EvidenceStatus.PROVEN_FAILED,
        "derived_volume_headers_carry_the_decimated_spacing": EvidenceStatus.PROVEN_FAILED,
        "proliferative_map_is_label_derived": EvidenceStatus.UNKNOWN,
        "adaptive_refinement": EvidenceStatus.PROVEN_FAILED,
        "sub_voxel_recovery_of_discarded_detail": EvidenceStatus.PROVEN_FAILED,
    }


def what_the_grid_cannot_show() -> List[str]:
    return [
        "the detail decimation destroyed: a fine sanctuary pocket smaller than a "
        "majority block is gone, and no measurement here can bring it back",
        "an anisotropic grid as one scalar diffusivity: the engine's Laplacian is "
        "isotropic in index space, so on the (4, 4, 3) factors no scalar D is "
        "simultaneously right along all three axes",
        "whether the drift is clinically material: the endpoint compared is a "
        "simulated PFS under uncalibrated constants, not a patient's",
        "adaptive mesh refinement: this module measures the fixed grid's cost, it "
        "does not remove it",
    ]
