"""Tests for the biopsy stream: what it measures, and what it refuses.

The load-bearing test is the synthetic slide: circles of a known count and radius
on a white background, written with an MPP, so the reported nuclei per mm^2 can be
checked against arithmetic rather than against itself. The refusal tests are the
other half -- a Ki-67 fraction must not be derivable from a slide that carries no
Ki-67 stain, in any of the ways someone might try.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image

from danon.histology import (
    HistologyError,
    classify_stain,
    ki67_fraction,
    measure_nuclei_density,
    read_slide_metadata,
)

REAL_SLIDE = "data/tcia/wsiprode"
HAEMATOXYLIN_RGB = (110, 90, 180)
DAB_RGB = (150, 110, 60)
#: Counterstained tissue. Saturated enough to count as tissue, hued away from the
#: haematoxylin band so it is not mistaken for a nucleus.
TISSUE_RGB = (240, 180, 185)


def _write_slide(path: str, rgb: np.ndarray, *, mpp: str | None = "2.0", mag: str = "20"):
    """Write an RGB array as a TIFF slide, optionally declaring MPP/AppMag."""
    Image.MAX_IMAGE_PIXELS = None
    info = {}
    if mpp is not None:
        # Tag 270 (ImageDescription) is where Aperio puts MPP and AppMag.
        info[270] = (
            f"Aperio Image Library v12.0\n{rgb.shape[1]}x{rgb.shape[0]} "
            f"[0,0 {rgb.shape[1]}x{rgb.shape[0]}] (256x256) JPEG/RGB Q=70"
            f"|AppMag = {mag}|MPP = {mpp}|"
        )
    Image.fromarray(rgb.astype(np.uint8)).save(path, format="TIFF", tiffinfo=info)


def _nuclei_slide(n: int = 200, size: int = 1024, radius: int = 3, seed: int = 7) -> np.ndarray:
    """A tissue field of `size` with `n` randomly placed nuclei on it.

    The background matters: the reported density is nuclei over *tissue* area, so
    a slide whose only stained pixels are the nuclei would read 1/median nucleus
    area and could not tell a dense slide from a sparse one.
    """
    import cv2

    rng = np.random.default_rng(seed)
    img = np.full((size, size, 3), TISSUE_RGB, dtype=np.uint8)
    spots = rng.integers(radius + 2, size - radius - 2, size=(n, 2))
    for (x, y) in spots:
        cv2.circle(img, (int(x), int(y)), radius, HAEMATOXYLIN_RGB, thickness=-1)
    return img


class TestSyntheticDensity:
    def test_recovers_a_planted_nucleus_count_and_density(self, tmp_path):
        n, radius, mpp = 200, 3, 2.0
        path = str(tmp_path / "planted.tif")
        _write_slide(path, _nuclei_slide(n=n, radius=radius), mpp=str(mpp))

        out = measure_nuclei_density(
            path, crops=[(0, 0)], crop_px=1024, min_tissue_fraction=0.001
        )
        # Every drawn disc is one connected component of pi*r^2 ~ 28 px.
        assert out["nuclei"] == pytest.approx(n, rel=0.05)

        # Density must equal count over tissue area, and here the tissue is the
        # whole 1024^2 crop, converted with the slide's own MPP.
        expected = n / (1024 * 1024 * (mpp / 1000.0) ** 2)
        assert out["nuclei_per_mm2"] == pytest.approx(expected, rel=0.1)
        assert out["tissue_mm2"] == pytest.approx(1024 * 1024 * (mpp / 1000.0) ** 2, rel=0.1)
        assert out["median_nucleus_area_um2"] == pytest.approx(
            np.pi * radius**2 * mpp**2, rel=0.2
        )

    def test_a_denser_slide_reads_denser(self, tmp_path):
        """Four times the nuclei on the same tissue reads four times denser.

        Non-overlapping placement on purpose: the rate is count over *tissue*
        area, so nuclei that clump into components past the area band are lost,
        and a denser-but-clumped slide can read sparser. That is the band doing
        its job, not a density to calibrate against.
        """
        import cv2

        def lattice(step: int, size: int = 1024) -> np.ndarray:
            img = np.full((size, size, 3), TISSUE_RGB, dtype=np.uint8)
            for y in range(step // 2, size, step):
                for x in range(step // 2, size, step):
                    cv2.circle(img, (x, y), 3, HAEMATOXYLIN_RGB, thickness=-1)
            return img

        sparse = str(tmp_path / "sparse.tif")
        dense = str(tmp_path / "dense.tif")
        _write_slide(sparse, lattice(100), mpp="2.0")
        _write_slide(dense, lattice(50), mpp="2.0")
        a = measure_nuclei_density(sparse, crops=[(0, 0)], crop_px=1024, min_tissue_fraction=0.001)
        b = measure_nuclei_density(dense, crops=[(0, 0)], crop_px=1024, min_tissue_fraction=0.001)
        assert a["nuclei"] == pytest.approx(100, rel=0.05)
        assert b["nuclei"] == pytest.approx(400, rel=0.05)
        assert b["nuclei_per_mm2"] == pytest.approx(4 * a["nuclei_per_mm2"], rel=0.1)


class TestRefusals:
    def test_a_slide_without_mpp_refuses(self, tmp_path):
        path = str(tmp_path / "nompp.tif")
        _write_slide(path, _nuclei_slide(n=50), mpp=None)
        meta = read_slide_metadata(path)
        assert meta.mpp_um is None
        with pytest.raises(HistologyError) as err:
            measure_nuclei_density(path, crops=[(0, 0)])
        assert "MPP" in str(err.value)

    def test_a_file_that_is_not_a_slide_refuses(self, tmp_path):
        path = tmp_path / "notaslide.svs"
        path.write_bytes(b"this file is not an image at all\n" + os.urandom(64))
        with pytest.raises(HistologyError):
            read_slide_metadata(str(path))

    def test_an_empty_slide_refuses_rather_than_reporting_zero(self, tmp_path):
        path = str(tmp_path / "blank.tif")
        _write_slide(path, np.full((256, 256, 3), 255, dtype=np.uint8), mpp="2.0")
        with pytest.raises(HistologyError) as err:
            measure_nuclei_density(path, crops=[(0, 0)], crop_px=256)
        assert "nothing to count" in str(err.value)


class TestStainAndKi67:
    def test_haematoxylin_only_is_called(self):
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        img[8:56, 8:56] = HAEMATOXYLIN_RGB
        assert classify_stain(img)["stain"] == "haematoxylin-only"

    def test_dab_is_called_when_brown_dominates(self):
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        img[8:56, 8:56] = DAB_RGB
        out = classify_stain(img)
        assert out["stain"] == "dab-positive"
        assert out["brown_fraction"] > 0.15

    def test_ki67_fraction_refuses_on_a_haematoxylin_slide(self):
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        img[8:56, 8:56] = HAEMATOXYLIN_RGB
        stain = classify_stain(img)
        with pytest.raises(HistologyError) as err:
            ki67_fraction(stain)
        message = str(err.value)
        assert "Ki-67" in message and "numerator" in message

    def test_ki67_fraction_refuses_even_on_a_dab_slide(self):
        """A brown slide is not enough: no pathologist-anchored split exists."""
        img = np.full((64, 64, 3), 255, dtype=np.uint8)
        img[8:56, 8:56] = DAB_RGB
        with pytest.raises(HistologyError) as err:
            ki67_fraction(classify_stain(img))
        assert "pathologist" in str(err.value)


@pytest.mark.skipif(
    not (os.path.isdir(REAL_SLIDE) and any(
        n.endswith((".svs", ".tif", ".tiff")) for n in os.listdir(REAL_SLIDE)
    )),
    reason="real TCGA-GBM slide not cached (see _histology_probe.py)",
)
class TestRealSlide:
    def _path(self) -> str:
        return os.path.join(
            REAL_SLIDE,
            sorted(n for n in os.listdir(REAL_SLIDE) if n.endswith(".svs"))[0],
        )

    def test_real_slide_is_haematoxylin_only_at_low_magnification(self):
        meta = read_slide_metadata(self._path())
        assert meta.mpp_um == pytest.approx(2.0037)
        assert meta.app_mag == pytest.approx(5.0)
        assert meta.width > 1000 and meta.height > 1000

    def test_real_slide_density_is_plausible_and_multi_crop(self):
        out = measure_nuclei_density(
            self._path(),
            crops=[(500, 500), (1200, 1800), (2500, 3200)],
            crop_px=1024,
        )
        assert out["crops_used"] >= 2
        assert 50 < out["nuclei_per_mm2"] < 20_000
        assert out["nuclei"] > 100
        # The refusal must name what is missing, on the real slide too.
        from PIL import Image as _Image

        _Image.MAX_IMAGE_PIXELS = None
        im = _Image.open(self._path())
        stain = classify_stain(np.asarray(im.crop((3000, 1500, 4024, 2524)).convert("RGB")))
        assert stain["stain"] == "haematoxylin-only"
        with pytest.raises(HistologyError):
            ki67_fraction(stain)
