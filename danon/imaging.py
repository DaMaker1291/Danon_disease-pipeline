"""
Real-imaging intake: NIfTI-1 I/O and TumorGeometry construction from a
segmentation mask.

WHY THIS MODULE EXISTS
----------------------
Every deliverable downstream (engine, cohort layer, schedule compiler) runs
on a synthetic phantom. The pipeline's declared input boundary was
"geometry is the phantom". This module closes the geometry half of that gap:
a real brain-MRI segmentation mask (DICOM -> NIfTI via any converter, or a
BraTS-style mask) becomes a TumorGeometry the engine actually simulates.

NIfTI-1 I/O is implemented with the standard library (struct + gzip) and
numpy only -- no nibabel dependency -- so the intake path is fully testable
in this repo. The format is the published NIfTI-1 standard (single-file
.nii, magic "n+1"), which is what every segmentation pipeline exports.

THE HONESTY BOUNDARY (load-bearing)
-----------------------------------
A segmentation mask gives exactly ONE of the three TumorGeometry maps:
tumor_mask. The other two are NOT measurable from a structural mask and are
derived here as explicit, labelled proxies:

proliferative -- derived from the city-block distance to the tumour
    boundary (rim -> high, core -> low). This is a SPATIAL PROXY for
    proliferation, not a measurement. The real map comes from enhancing-rim
    segmentation (T1-gd) or DWI/ADC; supply it via proliferative_path.

vessels -- DEFAULT IS THE NULL VASCULATURE MODEL: a uniform drug-inlet
    level inside the tumour. Uniform vasculature abolishes diffusion
    sanctuaries BY CONSTRUCTION, so sanctuary-dependent results (persistent
    resistant pockets under MTD) cannot be claimed from the null default.
    The real map comes from DCE-MRI perfusion (Ktrans); supply it via
    vessels_path. This is stated in PatientGeometry.notes() and in
    imaging_provenance().

So the pipeline's input boundary moves from "phantom only" to "real mask +
declared proxy maps", and the notes carried on every PatientGeometry make
the difference impossible to blur in a report.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import struct
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from danon.cure import EvidenceStatus
from danon.evosteer import TumorGeometry

# The engine's per-day cost scales with total voxels (diffusion steps over
# the full grid). Refuse absurd inputs up front rather than hanging a run.
MAX_TOTAL_VOXELS = 4_000_000

_NIFTI_DATATYPES = {
    2: np.dtype("uint8"),
    4: np.dtype("<i2"),
    8: np.dtype("<i4"),
    16: np.dtype("<f4"),
    64: np.dtype("<f8"),
    256: np.dtype("i1"),
    512: np.dtype("<u2"),
    768: np.dtype("<u4"),
}


# ---------------------------------------------------------------------------
# NIfTI-1 header
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NiftiHeader:
    dims: Tuple[int, int, int]              # x, y, z (dim[1:4])
    voxel_sizes: Tuple[float, float, float]  # pixdim[1:4]
    datatype_code: int
    vox_offset: float
    scl_slope: float
    scl_inter: float
    little_endian: bool
    magic: str


def _parse_header(raw: bytes) -> NiftiHeader:
    if len(raw) < 348:
        raise ValueError(f"NIfTI header truncated: got {len(raw)} bytes, need 348")
    sizeof_le = struct.unpack_from("<i", raw, 0)[0]
    sizeof_be = struct.unpack_from(">i", raw, 0)[0]
    if sizeof_le == 348:
        endian = "<"
    elif sizeof_be == 348:
        endian = ">"
    else:
        raise ValueError(f"not a NIfTI-1 header (sizeof_hdr={sizeof_le}/{sizeof_be})")
    dims = struct.unpack_from(endian + "8h", raw, 40)
    if dims[0] != 3:
        raise ValueError(f"only 3D NIfTI supported, dim[0]={dims[0]}")
    if any(d <= 0 for d in dims[1:4]):
        raise ValueError(f"invalid dimensions {dims[1:4]}")
    datatype = struct.unpack_from(endian + "h", raw, 70)[0]
    if datatype not in _NIFTI_DATATYPES:
        raise ValueError(f"unsupported NIfTI datatype code {datatype}")
    pixdim = struct.unpack_from(endian + "8f", raw, 76)
    vox_offset = struct.unpack_from(endian + "f", raw, 108)[0]
    scl_slope = struct.unpack_from(endian + "f", raw, 112)[0]
    scl_inter = struct.unpack_from(endian + "f", raw, 116)[0]
    magic = raw[344:348].decode("ascii", errors="replace").rstrip("\x00")
    if magic != "n+1":
        raise ValueError(
            f"only single-file NIfTI-1 supported (magic={magic!r}; "
            "convert ANALYZE/two-file pairs with any NIfTI converter)"
        )
    if math.isnan(vox_offset) or vox_offset < 0:
        raise ValueError(f"invalid vox_offset {vox_offset}")
    return NiftiHeader(
        dims=(int(dims[1]), int(dims[2]), int(dims[3])),
        voxel_sizes=(abs(pixdim[1]) or 1.0, abs(pixdim[2]) or 1.0, abs(pixdim[3]) or 1.0),
        datatype_code=int(datatype),
        vox_offset=float(vox_offset),
        scl_slope=float(scl_slope),
        scl_inter=float(scl_inter),
        little_endian=(endian == "<"),
        magic=magic,
    )


def mask_bytes_provenance(raw: bytes) -> Dict[str, Any]:
    """Identity of an uploaded mask: its bytes, and its header's own geometry.

    A recorded run must be tied to the exact file it was computed from. The
    engine runs on a temp copy, so the uploaded bytes are the only record of
    the input; this returns the fields that make that record checkable:

    * ``input_sha256`` over the bytes **as received** -- a ``.nii.gz`` is
      hashed compressed, not re-serialised, so the digest identifies the file
      the caller sent rather than a normalised rewrite of it;
    * ``original_grid_shape_xyz`` -- the dims the header declares, in the same
      x-fastest (x, y, z) order the engine indexes, so it equals
      ``mask.shape`` for the array this file yields;
    * ``source_voxel_sizes_mm`` and their mean ``source_voxel_mm``, which is
      the scalar ``PatientGeometry.voxel_mm`` is derived from.

    Computed from the header alone, without materialising the data array, so
    reporting the identity cannot alter what a run does. Shared by the API and
    the reproducibility check so the two cannot drift on what "the input" is.
    """
    blob = gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw
    hdr = _parse_header(blob[:348])
    sizes = tuple(float(v) for v in hdr.voxel_sizes)
    return {
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "original_grid_shape_xyz": [int(d) for d in hdr.dims],
        "source_voxel_sizes_mm": [sizes[0], sizes[1], sizes[2]],
        "source_voxel_mm": float(np.mean(sizes)),
    }


def read_nifti(path: str) -> Tuple[np.ndarray, NiftiHeader]:
    """Read a NIfTI-1 file (.nii or .nii.gz) into a numpy array.

    Returns (data, header). data has shape (x, y, z) with x varying fastest
    in memory (Fortran order), matching the engine's [x, y, z] indexing and
    the .vti exporter's convention. Scl slope/inter are applied when present.
    """
    with open(path, "rb") as fh:
        raw_head = fh.read(2)
    if raw_head[:2] == b"\x1f\x8b":
        with gzip.open(path, "rb") as fh:
            blob = fh.read()
    else:
        with open(path, "rb") as fh:
            blob = fh.read()

    hdr = _parse_header(blob[:348])
    dtype = _NIFTI_DATATYPES[hdr.datatype_code]
    if not hdr.little_endian:
        dtype = dtype.newbyteorder(">")
    n_vox = hdr.dims[0] * hdr.dims[1] * hdr.dims[2]
    start = int(hdr.vox_offset)
    need = start + n_vox * dtype.itemsize
    if len(blob) < need:
        raise ValueError(
            f"data truncated: file has {len(blob)} bytes, header promises {need}"
        )
    flat = np.frombuffer(blob, dtype=dtype, count=n_vox, offset=start)
    data = flat.reshape(hdr.dims, order="F")
    if hdr.scl_slope not in (0.0, 1.0) and not math.isnan(hdr.scl_slope):
        data = data * np.float64(hdr.scl_slope) + np.float64(hdr.scl_inter)
    return np.asarray(data), hdr


def write_nifti(
    path: str,
    data: np.ndarray,
    voxel_sizes: Tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> str:
    """Write a 3D uint8 NIfTI-1 single file (.nii, or .nii.gz if named so).

    Byte-reproducible: the same array written twice produces the same bytes,
    including for .nii.gz (no gzip timestamp, no embedded filename).
    """
    arr = np.asarray(data)
    if arr.ndim != 3:
        raise ValueError(f"expected a 3D array, got ndim={arr.ndim}")
    if arr.max(initial=0) > 255 or arr.min(initial=0) < 0:
        raise ValueError("write_nifti stores uint8: clip/scale data to [0, 255] first")
    out = np.ascontiguousarray(arr.astype(np.uint8))
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)                      # sizeof_hdr
    hdr[39:40] = b"\x00"                                     # dim_info
    struct.pack_into("<8h", hdr, 40, 3, *out.shape, 1, 1, 1, 1)
    struct.pack_into("<h", hdr, 70, 2)                       # datatype = uint8
    struct.pack_into("<h", hdr, 72, 8)                       # bitpix
    struct.pack_into("<8f", hdr, 76, 1.0, *voxel_sizes, 1.0, 1.0, 1.0, 1.0)
    struct.pack_into("<f", hdr, 108, 352.0)                  # vox_offset
    struct.pack_into("<f", hdr, 112, 1.0)                    # scl_slope
    struct.pack_into("<f", hdr, 116, 0.0)                    # scl_inter
    hdr[123:124] = b"\x02"                                   # xyzt_units = mm
    hdr[344:348] = b"n+1\x00"                                # single-file magic
    payload = out.astype("<u1").tobytes(order="F")           # x-fastest
    blob = bytes(hdr) + b"\x00\x00\x00\x00" + payload
    if path.endswith(".gz"):
        # mtime=0 and no embedded filename, so the output is byte-reproducible:
        # plain `gzip.open` stamps the current time into the header and writes
        # the file's own name, which makes two writers of the SAME array
        # produce different bytes. Nothing reads either field back.
        with open(path, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as fh:
                fh.write(blob)
    else:
        with open(path, "wb") as fh:
            fh.write(blob)
    return path


# ---------------------------------------------------------------------------
# Numpy-only city-block distance transform (erosion layers)
# ---------------------------------------------------------------------------


def _boundary_distance(mask: np.ndarray, max_layers: int = 512) -> np.ndarray:
    """Distance (in voxels, 6-neighbour city-block) from the mask boundary,
    for voxels inside mask; 0 outside. Erosion-layer implementation: each
    6-neighbour erosion peels one voxel layer, so layer k has distance k."""
    current = mask.astype(bool).copy()
    dist = np.zeros(mask.shape, dtype=np.int32)
    for layer in range(1, max_layers + 1):
        eroded = current.copy()
        for axis in range(3):
            for shift in (-1, 1):
                eroded &= np.roll(current, shift, axis=axis)
        # Volumes touching the array edge are boundary by convention.
        for axis in range(3):
            sl_hi = [slice(None)] * 3
            sl_hi[axis] = slice(-1, None)
            sl_lo = [slice(None)] * 3
            sl_lo[axis] = slice(0, 1)
            eroded[tuple(sl_hi)] = False
            eroded[tuple(sl_lo)] = False
        dist[current & ~eroded] = layer - 1
        if not eroded.any():
            break
        current = eroded
    dist[current] = max_layers
    return dist


# ---------------------------------------------------------------------------
# Geometry construction with provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatientGeometry:
    """A TumorGeometry built from real imaging, with its derivation record.

    geometry             : the engine-ready TumorGeometry.
    vasculature_source   : 'file' when a perfusion map was supplied, else the
                           null-uniform default -- which ABOLISHES SANCTUARIES
                           BY CONSTRUCTION (see module docstring).
    proliferative_source : 'file' when supplied, else the distance-to-boundary
                           proxy (a spatial proxy, NOT a proliferation assay).
    """

    geometry: TumorGeometry
    mask_path: str
    n_tumour_voxels: int
    voxel_mm: float
    vasculature_source: str
    proliferative_source: str

    def notes(self) -> List[str]:
        out = [
            f"tumour mask from {self.mask_path} ({self.n_tumour_voxels:,} voxels, "
            f"{self.voxel_mm:.2f} mm isotropic)",
        ]
        if self.vasculature_source == "file":
            out.append("vasculature: supplied perfusion map (normalized to [0,1])")
        else:
            out.append(
                "vasculature: NULL MODEL -- uniform drug inlet inside the tumour. "
                "This abolishes diffusion sanctuaries BY CONSTRUCTION: results "
                "about sanctuary pockets do not apply. Supply a DCE-MRI Ktrans "
                "map for vascular structure."
            )
        if self.proliferative_source == "file":
            out.append("proliferative map: supplied (normalized to [0,1])")
        else:
            out.append(
                "proliferative map: DISTANCE PROXY from the tumour boundary "
                "(rim high, core low) -- a spatial proxy, not a measurement; "
                "supply a T1-gd enhancing-rim or ADC-derived map instead"
            )
        return out


def _load_map(path: str, mask: np.ndarray, name: str) -> np.ndarray:
    arr, hdr = read_nifti(path)
    if arr.shape != mask.shape:
        raise ValueError(
            f"{name} map {path} has shape {arr.shape} but mask has {mask.shape}; "
            "resample to the mask grid first"
        )
    arr = arr.astype(np.float64)
    vmax = float(arr.max(initial=0.0))
    if vmax <= 0:
        raise ValueError(f"{name} map {path} is empty/constant")
    if vmax > 1.5:  # heuristic: raw intensity scale -> normalize to [0, 1]
        arr = arr / vmax
    return np.clip(arr, 0.0, 1.0)


def from_nifti_mask(
    mask_path: str,
    proliferative_path: Optional[str] = None,
    vessels_path: Optional[str] = None,
) -> PatientGeometry:
    """Build an engine-ready TumorGeometry from a real segmentation mask.

    mask_path          : NIfTI-1 file, nonzero = tumour (any uint8/int label
                         map or binary mask works; labels are binarized).
    proliferative_path : optional real proliferation/enhancing map; otherwise
                         the distance proxy is derived (labelled as such).
    vessels_path       : optional real perfusion map (e.g. normalized Ktrans);
                         otherwise the NULL uniform vasculature is used and
                         the sanctuary caveat is attached to the geometry.
    """
    mask_arr, hdr = read_nifti(mask_path)
    mask = mask_arr > 0
    n_tumour = int(mask.sum())
    if n_tumour == 0:
        raise ValueError(f"{mask_path}: no nonzero voxels -- empty or wrong image?")
    total = int(np.prod(mask.shape))
    if total > MAX_TOTAL_VOXELS:
        raise ValueError(
            f"{mask_path}: {total:,} total voxels exceeds the engine budget "
            f"({MAX_TOTAL_VOXELS:,}). Downsample the mask grid (the engine's "
            "diffusion cost scales with grid volume) and re-run."
        )

    if proliferative_path is not None:
        prolif_arr = _load_map(proliferative_path, mask, "proliferative")
        prolif_src = "file"
    else:
        dist = _boundary_distance(mask)
        dmax = float(dist.max(initial=0.0))
        prolif_arr = np.where(
            mask, np.clip(0.2 + 0.8 * (dist / dmax if dmax > 0 else 0.0), 0.0, 1.0), 0.0
        )
        prolif_src = "distance-proxy"

    if vessels_path is not None:
        vessels_arr = _load_map(vessels_path, mask, "vessels")
        vessels_src = "file"
    else:
        # Null vasculature: uniform inlet. Honest, but sanctuary-free by
        # construction -- flagged in PatientGeometry.notes().
        vessels_arr = np.where(mask, 0.5, 0.0)
        vessels_src = "null-uniform"

    voxel_mm = float(np.mean(hdr.voxel_sizes))
    geom = TumorGeometry(
        tumor_mask=mask,
        proliferative=prolif_arr,
        vessels=vessels_arr,
        voxel_mm=voxel_mm,
    )
    return PatientGeometry(
        geometry=geom,
        mask_path=mask_path,
        n_tumour_voxels=n_tumour,
        voxel_mm=voxel_mm,
        vasculature_source=vessels_src,
        proliferative_source=prolif_src,
    )


# Backwards-friendly alias so the engine contract stays the single entry point.
TumorGeometry.from_nifti_mask = classmethod(lambda cls, *a, **k: from_nifti_mask(*a, **k).geometry)


def imaging_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "nifti_1_format": EvidenceStatus.ESTABLISHED,        # published standard
        "mask_binarization": EvidenceStatus.ESTABLISHED,     # lossless for binary masks
        "proliferative_distance_proxy": EvidenceStatus.UNKNOWN,   # not a measurement
        "null_vasculature_sanctuary_free": EvidenceStatus.PROVEN_FAILED,  # by construction
        "perfusion_map_normalization": EvidenceStatus.UNKNOWN,    # max-normalize heuristic
    }


def imaging_what_this_model_cannot_know() -> List[str]:
    return [
        "true vasculature from a structural mask: the null-uniform default "
        "removes every sanctuary pocket, so MTD-vs-steering SPATIAL "
        "mechanisms cannot be studied without a supplied perfusion map",
        "true proliferation: the distance proxy encodes geometry, not biology",
        "any MRI intensity meaning: this module never interprets pixel "
        "values as tissue -- the segmentation producing the mask is an "
        "upstream, separately-validated model",
        "sub-voxel structure: the engine runs at the mask's grid; partial-"
        "volume effects at the boundary are invisible",
    ]
