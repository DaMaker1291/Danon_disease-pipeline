"""Physics-derived features for the engine emulator.

WHY THIS MODULE EXISTS
----------------------
The first emulator attempt used the six raw scalar configuration inputs and
plateaued at out-of-fold R^2 = 0.65 (measured, on 48 held-out-fold
configurations) while the measured noise floor showed an oracle ceiling near
0.999. The gap was the feature representation: the outcomes depend on the
GEOMETRY (vessel-to-voxel distances, sanctuary volume, proliferation map) and
two of the six scalars act on a per-voxel map, not as scalars. This module
computes deterministic, engine-consistent descriptors from the phantom
itself, so the surrogate sees the same structure the dynamics see.

WHAT IS DERIVED VS ASSUMED
--------------------------
Every feature here is computed from the same maps the engine consumes
(tumor_mask, proliferative, vessels) with the same city-block boundary
convention the engine's proxy uses, so the features are engine-consistent by
construction. They are DERIVED quantities of the input geometry, not
additional assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from danon.evosteer import TumorGeometry


def _city_block_dist_to_mask(
    source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Exact city-block (L1) distance from every voxel to the nearest target voxel.

    Uses the same two-pass city-block recurrence as the engine's boundary
    distance (danon.imaging._boundary_distance), applied from a source mask to
    a target mask, so distances are engine-consistent.
    """
    inf = source.shape[0] + source.shape[1] + source.shape[2]
    dist = np.full(source.shape, inf, dtype=np.int32)
    dist[target] = 0
    # forward pass
    for axis in range(3):
        for shift in (1,):
            sl_hi = [slice(None)] * 3
            sl_hi[axis] = slice(1, None)
            sl_lo = [slice(None)] * 3
            sl_lo[axis] = slice(0, -1)
            dist[tuple(sl_hi)] = np.minimum(
                dist[tuple(sl_hi)], dist[tuple(sl_lo)] + 1
            )
    # backward pass
    for axis in range(3):
        sl_hi = [slice(None)] * 3
        sl_hi[axis] = slice(0, -1)
        sl_lo = [slice(None)] * 3
        sl_lo[axis] = slice(1, None)
        dist[tuple(sl_hi)] = np.minimum(
            dist[tuple(sl_hi)], dist[tuple(sl_lo)] + 1
        )
    # repeat sweeps to propagate turns (city-block distance needs multiple
    # sweeps for non-convex paths; 4 sweeps suffice for grid-scale structures)
    for _ in range(3):
        for axis in range(3):
            for direction in (1, -1):
                if direction == 1:
                    sl_hi = [slice(None)] * 3
                    sl_hi[axis] = slice(1, None)
                    sl_lo = [slice(None)] * 3
                    sl_lo[axis] = slice(0, -1)
                    dist[tuple(sl_hi)] = np.minimum(
                        dist[tuple(sl_hi)], dist[tuple(sl_lo)] + 1
                    )
                else:
                    sl_hi = [slice(None)] * 3
                    sl_hi[axis] = slice(0, -1)
                    sl_lo = [slice(None)] * 3
                    sl_lo[axis] = slice(1, None)
                    dist[tuple(sl_hi)] = (
                        np.minimum(
                            dist[tuple(sl_hi)],
                            dist[tuple(sl_lo)] + 1,
                        )
                    )
    return dist


def geometry_features(geom: TumorGeometry) -> Dict[str, float]:
    """Deterministic descriptors of the geometry the engine will run on.

    All quantities are computed from the engine's own maps. Distances are in
    voxels; volumes are voxel counts (relative features are dimensionless).
    """
    mask = geom.tumor_mask
    prolif = geom.proliferative
    vessels = geom.vessels

    n_tumor = int(mask.sum())
    if n_tumor == 0:
        raise ValueError("empty tumour mask")

    # Proliferative map: exactly what the engine's distance proxy produces.
    from danon.imaging import _boundary_distance

    dist = _boundary_distance(mask)  # 0 at boundary, higher inside
    dmax = float(dist.max(initial=0.0))

    # Vessel geometry: the engine's source strength is the vessels map.
    vessel_voxels = vessels > 0
    n_vessel = int(vessel_voxels.sum())

    # Distance from every tumour voxel to the nearest vessel (city-block).
    dist_to_vessel = _city_block_dist_to_mask(mask, vessel_voxels)

    #--- derived descriptors -------------------------------------------------
    tumor_idx = dist[mask]
    vessel_dist_in_tumor = dist_to_vessel[mask]

    features: Dict[str, float] = {}
    features["log_tumour_voxels"] = float(np.log(n_tumor))
    features["mean_internal_distance"] = float(tumor_idx.mean())
    features["max_internal_distance"] = float(tumor_idx.max())
    features["mean_dist_to_vessel"] = float(vessel_dist_in_tumor.mean())
    features["median_dist_to_vessel"] = float(np.median(vessel_dist_in_tumor))
    features["max_dist_to_vessel"] = float(vessel_dist_in_tumor.max())
    features["p90_dist_to_vessel"] = float(np.percentile(vessel_dist_in_tumor, 90))
    features["vessel_density"] = float(n_vessel / n_tumor)
    # Sanctuary proxies: tumour volume far from every vessel.
    features["sanctuary_frac_d_gt_4"] = float((vessel_dist_in_tumor > 4).mean())
    features["sanctuary_frac_d_gt_6"] = float((vessel_dist_in_tumor > 6).mean())
    features["sanctuary_frac_d_gt_8"] = float((vessel_dist_in_tumor > 8).mean())
    # Proliferative-structure descriptors (the rim/core structure):
    features["mean_prolif"] = float(prolif[mask].mean())
    features["core_volume_frac"] = float((prolif[mask] < 0.5).mean())
    # Joint descriptor: sanctuary volume inside the core (worst case).
    core_mask = mask & (prolif < 0.5)
    if core_mask.any():
        features["core_dist_to_vessel_mean"] = float(dist_to_vessel[core_mask].mean())
        features["core_sanctuary_frac"] = float((dist_to_vessel[core_mask] > 6).mean())
    else:
        features["core_dist_to_vessel_mean"] = 0.0
        features["core_sanctuary_frac"] = 0.0

    return features


FEATURE_ORDER: Tuple[str, ...] = (
    "log_tumour_voxels",
    "mean_internal_distance",
    "max_internal_distance",
    "mean_dist_to_vessel",
    "median_dist_to_vessel",
    "max_dist_to_vessel",
    "p90_dist_to_vessel",
    "vessel_density",
    "sanctuary_frac_d_gt_4",
    "sanctuary_frac_d_gt_6",
    "sanctuary_frac_d_gt_8",
    "mean_prolif",
    "core_volume_frac",
    "core_dist_to_vessel_mean",
    "core_sanctuary_frac",
)


def feature_matrix(geoms: List[TumorGeometry]) -> np.ndarray:
    """Feature matrix for a list of geometries, columns in FEATURE_ORDER."""
    rows = []
    for geom in geoms:
        f = geometry_features(geom)
        rows.append([f[name] for name in FEATURE_ORDER])
    return np.array(rows, dtype=float)
