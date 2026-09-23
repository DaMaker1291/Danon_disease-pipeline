"""Real-patient-data intake: BraTS 2021 (TCGA/TCIA-derived) MRI segmentations.

WHAT THIS MODULE DOES
---------------------
Closes the pipeline's documented input gap: geometry and clone-fraction
inputs were synthetic (phantom + provenance-tagged intake parameter). This
module feeds the engine from REAL de-identified patient imaging:

    BraTS 2021 training data
      = TCGA-GBM / TCIA patient MRIs (pre-operative, multi-institutional),
        expert multi-parametric segmentations, released CC-BY-4.0 via
        Zenodo record 19541844 (Menze/Bakas et al. lineage).

Per case:
    1. The expert segmentation (labels 1=necrotic core, 2=peritumoral
       edema, 4=enhancing tumour) is CROPPED to the tumour bounding box and
       DECIMATED to the engine grid (majority vote per engine voxel).
    2. The engine-grid mask and label-derived proliferation map are written
       as NIfTI-1 files and ingested through the TESTED intake path
       (`imaging.from_nifti_mask`), so every guard (budget, emptiness,
       provenance notes) applies unchanged.
    3. Vasculature has NO real map in the BraTS static release: the
       null-uniform proxy is used and its sanctuary-free limitation is
       carried on the geometry notes verbatim.

WHAT IS DERIVED vs MEASURED (provenance discipline)
---------------------------------------------------
  * tumour geometry          : MEASURED (expert segmentation, downsampled)
  * proliferative map        : DERIVED from radiological labels (enhancing
                               rim -> high, edema -> mid, necrosis -> low).
                               Radiological labels correlate with, but are
                               NOT, proliferation assays.
  * vasculature              : NULL PROXY (sanctuaries abolished by
                               construction -- recorded, not hidden)
  * the decimation is a documented lossy step; engine grids are ~4 mm
    effective voxel size and engine-scale (<= a few hundred thousand
    voxels), not radiology-scale.

Every load-bearing entry is in `realdata_provenance()`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.imaging import (
    PatientGeometry,
    read_nifti,
    from_nifti_mask,
    write_nifti,
)

#: Source of the imaging data (license + lineage for the SRC/README record).
BRATS2021_SOURCE = (
    "BraTS 2021 training data: TCGA-GBM/TCIA-derived de-identified patient "
    "glioma MRI with expert segmentations; Zenodo record 19541844 "
    "(CC-BY-4.0); original collection TCGA-GBM via The Cancer Imaging Archive"
)

#: BraTS 2021 label semantics (published standard).
BRATS_LABELS = {
    1: "necrotic tumour core",
    2: "peritumoral edema",
    4: "enhancing tumour",
}


@dataclass(frozen=True)
class DecimationSpec:
    """How a radiology grid is reduced to an engine grid."""

    factors: Tuple[int, int, int] = (4, 4, 3)   # (x, y, z) strides
    crop_to_bbox: bool = True

    def describe(self) -> str:
        return (
            f"bbox-crop x stride {self.factors[0]}x{self.factors[1]}x"
            f"{self.factors[2]} majority-vote decimation"
        )


@dataclass
class RealCaseGeometry:
    """One real patient case made engine-ready, with its derivation record."""

    case_id: str
    patient_geometry: PatientGeometry
    engine_grid: Tuple[int, int, int]
    source_grid: Tuple[int, int, int]
    decimation: str
    voxel_mm: float
    label_counts_source: Dict[int, int]
    prolif_from_labels: bool
    mask_file: str
    prolif_file: str
    source: str = BRATS2021_SOURCE
    extra_notes: List[str] = field(default_factory=list)

    def notes(self) -> List[str]:
        out = list(self.patient_geometry.notes())
        out.append(
            f"geometry measured from expert segmentation; grid reduced by "
            f"{self.decimation} (source {self.source_grid[0]}x{self.source_grid[1]}x"
            f"{self.source_grid[2]} -> engine {self.engine_grid[0]}x"
            f"{self.engine_grid[1]}x{self.engine_grid[2]})"
        )
        if self.prolif_from_labels:
            out.append(
                "proliferative map DERIVED from radiological labels "
                "(enhancing=1.0, edema=0.6, necrosis=0.2): correlates with "
                "biology, is not a proliferation assay"
            )
        else:
            out.append("proliferative map: distance proxy (labels not usable)")
        out.append(
            "vasculature: null-uniform proxy -- no perfusion map exists in "
            "the release; sanctuary claims do not apply to these runs"
        )
        out.extend(self.extra_notes)
        return out


def _bbox(mask: np.ndarray, pad: int = 2) -> Tuple[slice, slice, slice]:
    """Bounding box of nonzero voxels with a small pad, clipped to the grid."""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        raise ValueError("empty mask: no nonzero voxels")
    lo = np.maximum(idx.min(axis=0) - pad, 0)
    hi = np.minimum(idx.max(axis=0) + 1 + pad, np.array(mask.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]


def _decimate_labels(labels: np.ndarray, spec: DecimationSpec) -> Tuple[np.ndarray, np.ndarray]:
    """Majority-vote decimation of a label volume to the engine grid.

    Returns (mask, label_grid). Each engine voxel takes the majority label
    of the fx*fy*fz contributing source voxels (ties resolve to the lower
    label, i.e. toward background), AND is kept only if a strict majority
    of contributors were tumour -- conservative at the boundary.
    """
    fx, fy, fz = spec.factors
    sx, sy, sz = labels.shape
    nx, ny, nz = sx // fx, sy // fy, sz // fz
    if nx == 0 or ny == 0 or nz == 0:
        raise ValueError(
            f"decimation {spec.factors} collapses grid {labels.shape}; use smaller factors"
        )
    trimmed = labels[: nx * fx, : ny * fy, : nz * fz]
    flat = trimmed.reshape(nx, fx, ny, fy, nz, fz).reshape(nx, ny, nz, -1)
    # Votes for labels 0, 1, 2, 4 per engine voxel.
    votes = np.stack([(flat == L).sum(axis=-1) for L in (0, 1, 2, 4)], axis=-1)
    mapped = np.array([0, 1, 2, 4])[np.argmax(votes, axis=-1)]
    n_contrib = flat.shape[-1]
    tumour_votes = n_contrib - votes[..., 0]
    mapped = np.where(tumour_votes * 2 > n_contrib, mapped, 0)  # strict majority
    labels_grid = mapped.astype(np.int32)
    mask = labels_grid > 0
    return mask.astype(np.uint8), labels_grid


def _label_proliferation(labels_grid: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, bool]:
    """Proliferation map DERIVED from radiological labels.

    enhancing (4) -> 1.0, edema (2) -> 0.6, necrosis (1) -> 0.2. If the
    decimated grid lost all enhancing voxels the map is not usable and the
    caller falls back to the distance proxy.
    """
    if int((labels_grid == 4).sum()) == 0:
        return np.zeros_like(mask, dtype=float), False
    prolif = np.zeros(labels_grid.shape, dtype=float)
    prolif[labels_grid == 1] = 0.20
    prolif[labels_grid == 2] = 0.60
    prolif[labels_grid == 4] = 1.00
    prolif[~mask] = 0.0
    return prolif, True


def prepare_brats_case(
    seg_path: str,
    case_id: Optional[str] = None,
    out_dir: str = "data/engine_ready",
    spec: DecimationSpec = DecimationSpec(),
    max_engine_voxels: int = 1_500_000,
) -> RealCaseGeometry:
    """Turn one BraTS segmentation into engine-ready NIfTI + PatientGeometry.

    Writes <out_dir>/<case>_mask.nii.gz and <case>_prolif.nii.gz on the
    engine grid, ingests them through imaging.from_nifti_mask, and returns
    the case record. Raises (never silently degrades) on empty/budget
    problems.
    """
    if case_id is None:
        stem = os.path.basename(seg_path)
        for ext in (".nii.gz", ".nii", ".gz"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        case_id = stem.replace("_seg", "")
    os.makedirs(out_dir, exist_ok=True)

    labels_src, hdr = read_nifti(seg_path)
    labels_src = np.rint(np.asarray(labels_src)).astype(np.int32)
    uniq, cnts = np.unique(labels_src, return_counts=True)
    label_counts = {int(u): int(c) for u, c in zip(uniq, cnts)}
    if not any(u in label_counts for u in (1, 2, 4)):
        raise ValueError(f"{seg_path}: no BraTS tumour labels (1/2/4) present")

    src_grid = labels_src.shape
    if spec.crop_to_bbox:
        bb = _bbox(labels_src > 0)
        labels_src = labels_src[bb]

    mask_grid, labels_grid = _decimate_labels(labels_src, spec)
    if mask_grid.sum() == 0:
        raise ValueError(f"{case_id}: decimation produced an empty tumour mask")
    if int(np.prod(mask_grid.shape)) > max_engine_voxels:
        raise ValueError(
            f"{case_id}: engine grid {mask_grid.shape} exceeds budget "
            f"{max_engine_voxels:,}; increase decimation factors"
        )

    prolif_map, from_labels = _label_proliferation(labels_grid, mask_grid > 0)
    if not from_labels:
        # Distance-proxy fallback: reuse imaging's tested proxy by simply
        # not supplying a proliferative file to from_nifti_mask.
        prolif_file = None
    else:
        prolif_file = os.path.join(out_dir, f"{case_id}_prolif.nii.gz")
        write_nifti(prolif_file, np.clip(prolif_map * 255.0, 0, 255).astype(np.uint8))

    mask_file = os.path.join(out_dir, f"{case_id}_mask.nii.gz")
    write_nifti(mask_file, mask_grid)

    source_vox = tuple(float(hdr.voxel_sizes[i]) * spec.factors[i] for i in range(3))
    voxel_mm = float(np.mean(source_vox))
    # Trust the measured source spacing (decimated), not the written header.
    # PatientGeometry/TumorGeometry are frozen: correct immutably.
    pg_ingested = from_nifti_mask(mask_file, proliferative_path=prolif_file)
    corrected_geom = replace(pg_ingested.geometry, voxel_mm=voxel_mm)
    pg = replace(pg_ingested, geometry=corrected_geom, voxel_mm=voxel_mm)

    return RealCaseGeometry(
        case_id=case_id,
        patient_geometry=pg,
        engine_grid=tuple(int(v) for v in mask_grid.shape),  # type: ignore[assignment]
        source_grid=tuple(int(v) for v in src_grid),  # type: ignore[assignment]
        decimation=spec.describe(),
        voxel_mm=pg.voxel_mm,
        label_counts_source=label_counts,
        prolif_from_labels=from_labels,
        mask_file=mask_file,
        prolif_file=prolif_file or "",
        extra_notes=[f"engine voxel ~{pg.voxel_mm:.2f} mm (decimated)"],
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def realdata_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "brats2021_expert_segmentations": EvidenceStatus.ESTABLISHED,
        "tcga_gbm_collection_lineage": EvidenceStatus.ESTABLISHED,
        "decimation_majority_vote_lossy": EvidenceStatus.UNKNOWN,
        "proliferation_from_radiological_labels": EvidenceStatus.UNKNOWN,
        "null_vasculature_no_sanctuaries": EvidenceStatus.PROVEN_FAILED,
    }


def realdata_what_this_model_cannot_know() -> List[str]:
    return [
        "vascular structure: BraTS ships no perfusion maps, so drug "
        "delivery is the null-uniform proxy and sanctuary dynamics are "
        "NOT studyable on these cases (supply Ktrans maps to enable them)",
        "clone genotype: the segmentation says nothing about drug "
        "resistance; the initial resistant fraction must come from "
        "genomics (see danon.genomics_intake) and stays the weakest input",
        "proliferation rates: radiological labels correlate with but do "
        "not measure proliferation; the label-derived map is a proxy",
        "radiotherapy/therapy history of each TCGA case before imaging",
        "engine-grid resolution: majority-vote decimation discards "
        "sub-voxel anatomy; fine sanctuary structure cannot survive it",
    ]
