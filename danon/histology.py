"""Whole-slide histomorphometry, and the refusal that belongs with it.

WHAT THIS CAN MEASURE
---------------------
TCGA-GBM carries whole-slide images (2,053 of them at the live GDC count), and
they are readable with what this environment already declares: Aperio SVS is a
tiled TIFF, so Pillow reads its tiles and OpenCV does the segmentation. Both are
recorded in requirements.txt, and neither is imported at module load -- a missing
one raises `HistologyError` naming itself rather than an ImportError mid-run.
From one slide this
module reports **nuclei density per mm^2** -- nuclei detected over the tissue
area, converted with the slide's own microns-per-pixel -- pooled over several
crops so the number is not a property of wherever the first crop landed.

WHAT IT REFUSES, AND WHY THAT IS THE MAIN RESULT
------------------------------------------------
Stream 2's specification maps biopsy pathology to a growth rate through

    r0 = baseline * (Ki-67 count / total nuclei)

and that ratio needs a **Ki-67 immunohistochemistry** slide: Ki-67 positive
nuclei over all nuclei. The archive's GBM slides are not that. Checked live and
against a downloaded file: none of the 2,053 file names identifies a Ki-67 or
IHC slide, and the smallest one is `AppMag = 5`, `MPP = 2.0037`, stained
haematoxylin-only with no DAB brown anywhere in seven crops. So the numerator of
that ratio does not exist in this source, and `ki67_fraction` refuses rather than
substituting nuclei density -- a density is not a proliferation index, and
swapping one for the other would turn a refusal into a fabricated r0.

An H&E slide still measures something real (cellularity), which is what this
module reports and labels.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "HAEMATOXYLIN_HUE",
    "NUCLEUS_AREA_PX",
    "TISSUE_SATURATION",
    "HistologyError",
    "SlideMeta",
    "read_slide_metadata",
    "cells_per_mm2_to_per_mm3",
    "classify_stain",
    "measure_nuclei_density",
    "ki67_fraction",
]

#: OpenCV hue band (0-179) that haematoxylin-stained nuclei occupy.
HAEMATOXYLIN_HUE = (105, 175)

#: DAB (immunohistochemistry) hue band, for telling a stained slide from an
#: unstained-counterstain-only one when no IHC is expected.
DAB_HUE = (5, 30)

#: Connected-component area limits, in pixels, for a nucleus. At MPP 2.0 um a
#: 10 um nucleus is ~25 px; the band admits the range seen on real slides while
#: rejecting single-pixel noise and merged clumps.
NUCLEUS_AREA_PX = (8, 400)

#: Saturation above which a pixel counts as tissue rather than glass.
TISSUE_SATURATION = 40


class HistologyError(RuntimeError):
    """A slide this module will not measure, with the reason it will not."""


@dataclass(frozen=True)
class SlideMeta:
    """What a slide says about itself, before any pixels are read."""

    path: str
    width: int
    height: int
    mpp_um: Optional[float]
    app_mag: Optional[float]
    levels: int
    format: str
    description: str = ""
    section_thickness_um: Optional[float] = None

    def provenance(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "pixels": [self.width, self.height],
            "mpp_um": self.mpp_um,
            "app_mag": self.app_mag,
            "pyramid_levels": self.levels,
            "format": self.format,
            "mm2_per_pixel": (self.mpp_um / 1000.0) ** 2 if self.mpp_um else None,
            "section_thickness_um": self.section_thickness_um,
        }


def _require_cv2():
    """OpenCV, or a stated refusal. Segmentation is the one step that needs it."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - present in this environment
        raise HistologyError(
            f"whole-slide segmentation needs OpenCV, which is not installed "
            f"({exc}); `pip install opencv-python` (see requirements.txt)"
        ) from exc
    return cv2


def _open_slide(path: str):
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is a hard dependency
        raise HistologyError(f"{path}: Pillow is unavailable ({exc})") from exc
    Image.MAX_IMAGE_PIXELS = None
    try:
        return Image.open(path)
    except Exception as exc:
        raise HistologyError(
            f"{path}: not readable as a whole-slide image "
            f"({type(exc).__name__}: {exc})"
        ) from exc


def read_slide_metadata(path: str) -> SlideMeta:
    """Read a slide's own geometry. No MPP means no mm^2, which refuses later."""
    im = _open_slide(path)
    raw = im.tag_v2.get(270) if hasattr(im, "tag_v2") else None
    description = ""
    if isinstance(raw, bytes):
        description = raw.decode("latin-1")
    elif isinstance(raw, str):
        description = raw
    mpp = re.search(r"MPP\s*=\s*([0-9.]+)", description)
    mag = re.search(r"AppMag\s*=\s*([0-9.]+)", description)
    # Section thickness is the missing Z of a whole-slide image: it is what would
    # turn an areal nuclei count into a volumetric one. Aperio writes scan
    # geometry (MPP, AppMag, StripeWidth) and only sometimes a thickness, so this
    # is read when present and reported as None when absent rather than assumed.
    thick = re.search(r"(?:Slice)?Thickness\s*=\s*([0-9.]+)", description)
    return SlideMeta(
        path=path,
        width=int(im.size[0]),
        height=int(im.size[1]),
        mpp_um=float(mpp.group(1)) if mpp else None,
        app_mag=float(mag.group(1)) if mag else None,
        levels=int(getattr(im, "n_frames", 1)),
        format=str(im.format or "unknown"),
        description=description,
        section_thickness_um=float(thick.group(1)) if thick else None,
    )


def cells_per_mm2_to_per_mm3(
    nuclei_per_mm2: float, section_thickness_um: Optional[float]
) -> float:
    """Areal nuclei density -> volumetric, or a refusal naming the missing Z.

    A whole-slide image counts nuclei per unit *area*; the engine's carrying
    capacity is a *volumetric* cell ceiling. The conversion is a division by the
    section thickness, which the archive's slides do not record (their metadata
    carries MPP and AppMag and no thickness field), so this raises rather than
    assuming a section thickness -- assuming one would set the tumour's density
    from a habit instead of from the slide.
    """
    if section_thickness_um is None or section_thickness_um <= 0:
        raise HistologyError(
            f"cannot convert {nuclei_per_mm2:g} nuclei/mm^2 to a volumetric "
            f"density without the section thickness: a whole-slide image counts "
            f"per unit AREA, and the slides here record MPP and AppMag but no "
            f"thickness field. Any absolute cells/mm^3 from this archive would be "
            f"an assumed section thickness wearing a measurement's units"
        )
    if not np.isfinite(nuclei_per_mm2) or nuclei_per_mm2 <= 0:
        raise HistologyError(f"nuclei density must be positive, got {nuclei_per_mm2!r}")
    return float(nuclei_per_mm2) / (float(section_thickness_um) / 1000.0)


def classify_stain(rgb: np.ndarray, *, sample_px: int = 2_000_000) -> Dict[str, object]:
    """Decide, from hue fractions, whether DAB brown is present.

    Haematoxylin-only tissue (H&E, or a counterstain-only section) has no brown.
    A Ki-67 slide is *brown-positive* where nuclei stain, so the absence of
    brown over a decent sample is what rules a slide out as the input stream 2
    needs -- it is not a judgement about the slide's quality.
    """
    cv2 = _require_cv2()

    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] < 3:
        raise HistologyError(f"expected an RGB image, got shape {a.shape}")
    flat = a.reshape(-1, 3)
    if flat.shape[0] > sample_px:
        step = flat.shape[0] // sample_px + 1
        flat = flat[::step]
    hsv = cv2.cvtColor(flat.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    hue, saturation, value = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    tissue = (saturation > TISSUE_SATURATION) & (value > 40)
    if int(tissue.sum()) < 100:
        raise HistologyError("no tissue pixels found (saturation never rises)")
    h = hue[tissue]
    brown = float(((h >= DAB_HUE[0]) & (h <= DAB_HUE[1])).mean())
    haematoxylin = float(((h >= HAEMATOXYLIN_HUE[0]) & (h <= HAEMATOXYLIN_HUE[1])).mean())
    if brown > 0.15:
        stain = "dab-positive"
    elif haematoxylin > 0.6:
        stain = "haematoxylin-only"
    else:
        stain = "unclassified"
    return {
        "stain": stain,
        "brown_fraction": brown,
        "haematoxylin_fraction": haematoxylin,
        "tissue_pixels_sampled": int(tissue.sum()),
        "rule": (
            "stain is called from hue fractions over tissue pixels: >15% DAB brown "
            "is dab-positive; >60% haematoxylin with almost no brown is "
            "haematoxylin-only"
        ),
    }


def measure_nuclei_density(
    path: str,
    *,
    crops: Optional[Sequence[Tuple[int, int]]] = None,
    crop_px: int = 1024,
    min_tissue_fraction: float = 0.02,
    meta: Optional[SlideMeta] = None,
) -> Dict[str, object]:
    """Nuclei per mm^2 over several crops of one slide.

    Refuses without an MPP (a count in pixels is not a density), when no crop
    holds tissue, and when the whole slide yields no nucleus -- each with the
    reason, because a zero here would otherwise read as "this tissue has no
    cells" rather than "this measurement failed".
    """
    cv2 = _require_cv2()

    meta = meta or read_slide_metadata(path)
    if meta.mpp_um is None or meta.mpp_um <= 0:
        raise HistologyError(
            f"{path}: the slide states no MPP, so nuclei counts cannot be "
            f"converted to per-mm^2 and nothing here is comparable across slides"
        )
    px_mm2 = (meta.mpp_um / 1000.0) ** 2

    im = _open_slide(path)
    if crops is None:
        step = max(crop_px, meta.width // 4)
        crops = [
            (x, y)
            for y in range(crop_px // 2, meta.height - crop_px // 2, step)
            for x in range(crop_px // 2, meta.width - crop_px // 2, step)
        ][:7]

    per_crop: List[Dict[str, object]] = []
    total_nuclei = 0
    total_mm2 = 0.0
    tissue_pixels = 0
    nucleus_areas: List[int] = []
    for (x, y) in crops:
        if x + crop_px > meta.width or y + crop_px > meta.height:
            continue
        rgb = np.asarray(
            im.crop((x, y, x + crop_px, y + crop_px)).convert("RGB")
        )
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        tissue = ((hsv[..., 1] > TISSUE_SATURATION) & (hsv[..., 2] > 40)).astype(np.uint8)
        fraction = float(tissue.mean())
        if fraction < min_tissue_fraction:
            continue
        nuclei_mask = (
            (hsv[..., 0] >= HAEMATOXYLIN_HUE[0])
            & (hsv[..., 0] <= HAEMATOXYLIN_HUE[1])
            & (hsv[..., 1] > 55)
            & (hsv[..., 2] > 40)
        ).astype(np.uint8)
        _n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
            nuclei_mask, connectivity=8
        )
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = (areas >= NUCLEUS_AREA_PX[0]) & (areas <= NUCLEUS_AREA_PX[1])
        nuclei = int(keep.sum())
        area_mm2 = float(tissue.sum()) * px_mm2
        total_nuclei += nuclei
        total_mm2 += area_mm2
        tissue_pixels += int(tissue.sum())
        nucleus_areas.extend(int(a) for a in areas[keep])
        per_crop.append(
            {
                "origin_xy": [int(x), int(y)],
                "nuclei": nuclei,
                "tissue_mm2": round(area_mm2, 4),
                "nuclei_per_mm2": round(nuclei / area_mm2) if area_mm2 > 0 else 0,
                "tissue_fraction": round(fraction, 4),
            }
        )

    if not per_crop:
        raise HistologyError(
            f"{path}: none of the sampled crops holds tissue above "
            f"{min_tissue_fraction:.0%}; there is nothing to count"
        )
    if total_nuclei == 0:
        raise HistologyError(
            f"{path}: {len(per_crop)} tissue crops yielded no nucleus in the "
            f"haematoxylin band -- check the stain before reading this as empty "
            f"tissue"
        )

    rates = [c["nuclei_per_mm2"] for c in per_crop]
    median_area = float(np.median(nucleus_areas)) if nucleus_areas else 0.0
    return {
        "path": path,
        "slide": meta.provenance(),
        "nuclei": total_nuclei,
        "tissue_mm2": round(total_mm2, 4),
        "nuclei_per_mm2": round(total_nuclei / total_mm2) if total_mm2 > 0 else 0,
        "crops_used": len(per_crop),
        "per_crop": per_crop,
        "per_crop_spread": {
            "min": int(min(rates)),
            "max": int(max(rates)),
            "ratio": round(max(rates) / min(rates), 2) if min(rates) > 0 else None,
        },
        "median_nucleus_area_px": median_area,
        "median_nucleus_area_um2": round(median_area * (meta.mpp_um**2), 1),
        "measures": (
            "nuclei density in haematoxylin-stained tissue -- CELLULARITY, not a "
            "proliferation index"
        ),
        "not_measured": [
            "Ki-67 fraction (needs a Ki-67 IHC slide; see ki67_fraction)",
            "a mitosis count, a grade, or any WHO criterion",
            "anything about the patient the slide came from",
        ],
    }


def ki67_fraction(stain: Dict[str, object]) -> float:
    """Refuse: the ratio stream 2 needs cannot come from these slides.

    `r0 = baseline * (Ki-67 count / total nuclei)` has a numerator that only a
    Ki-67 stain supplies. Substituting a nuclei density would make the growth
    rate a function of the wrong quantity while looking like a measurement, so
    this raises and says what would be needed instead.
    """
    called = str(stain.get("stain"))
    if called != "dab-positive":
        raise HistologyError(
            f"Ki-67 fraction is not measurable from a '{called}' slide: the "
            f"numerator (Ki-67 positive nuclei) requires a Ki-67 "
            f"immunohistochemistry slide, and TCGA-GBM's slide images are not "
            f"that (0 of 2,053 file names identify Ki-67 or IHC; a downloaded "
            f"one is AppMag 5, MPP 2.0037, haematoxylin-only). No r0 is derived"
        )
    raise HistologyError(
        "a slide called dab-positive still needs the Ki-67-versus-counterstain "
        "split validated against a pathologist's count before it may drive r0; "
        "that validation does not exist here"
    )
