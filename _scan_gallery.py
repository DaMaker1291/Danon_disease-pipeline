"""Render the committed REAL BraTS-2021 scans for the dashboard's MRI Scans view.

Every pixel in these PNGs comes from data/brats/*_{t1ce,seg}.nii.gz -- real
patient MRI committed to this repository. The red/yellow contours are the
BraTS expert segmentation as provided, NOT a model output. The dashboard used
to show charts and engine fields but never an actual scan; this script is the
fix, and it draws nothing that was simulated.

Writes (byte-identical mirrors, the repository's convention):
  outputs_real/scans/*.png                        canonical images
  outputs_real/app_artifacts/scans.json           canonical index
  frontend/public/artifacts/scans/*.png           served copies
  frontend/public/artifacts/scans.json            served index

Run: python _scan_gallery.py
"""

import datetime
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from danon.imaging import read_nifti

BRATS_DIR = "data/brats"
ENGINE_DIR = "data/engine_ready"
OUT_DIR = "outputs_real/scans"
ARTIFACT_DIR = "outputs_real/app_artifacts"
MIRROR_DIR = "frontend/public/artifacts"
MIRROR_IMG_DIR = os.path.join(MIRROR_DIR, "scans")
MIRROR_MASK_DIR = os.path.join(MIRROR_DIR, "masks")

CASES = [
    "BraTS2021_01017",
    "BraTS2021_01018",
    "BraTS2021_01019",
    "BraTS2021_01020",
    "BraTS2021_01021",
    "BraTS2021_01022",
]
STANDARD_CASE = "BraTS2021_01017"

TUMOUR_RED = "#ff4136"
ENHANCING_YELLOW = "#ffd400"


def window(vol: np.ndarray, lo: float = 1.0, hi: float = 99.5) -> np.ndarray:
    """Percentile window for display only; never written back to data."""
    a, b = np.percentile(vol, [lo, hi])
    return np.clip((vol - a) / (b - a + 1e-9), 0.0, 1.0)


def best_slice(seg: np.ndarray, axis: int) -> int:
    other = tuple(ax for ax in range(3) if ax != axis)
    return int(np.argmax((seg > 0).sum(axis=other)))


def draw(ax, img2d: np.ndarray, mask2d: np.ndarray, seg2d: np.ndarray) -> None:
    ax.imshow(img2d, cmap="gray", origin="upper")
    if (mask2d > 0).any():
        ax.contour(
            mask2d.astype(float),
            levels=[0.5],
            colors=[TUMOUR_RED],
            linewidths=1.4,
            origin="upper",
        )
    if (seg2d == 4).any():
        ax.contour(
            (seg2d == 4).astype(float),
            levels=[0.5],
            colors=[ENHANCING_YELLOW],
            linewidths=1.2,
            origin="upper",
        )
    ax.set_axis_off()


def load_case(case: str):
    t1ce, _ = read_nifti(os.path.join(BRATS_DIR, f"{case}_t1ce.nii.gz"))
    seg, _ = read_nifti(os.path.join(BRATS_DIR, f"{case}_seg.nii.gz"))
    return t1ce, seg


def engine_mask_voxels(case: str):
    path = os.path.join(ENGINE_DIR, f"{case}_mask.nii.gz")
    if not os.path.exists(path):
        return None, None
    mask, _ = read_nifti(path)
    return int((mask > 0).sum()), [int(v) for v in mask.shape]


def render_case(case: str) -> dict:
    t1ce, seg = load_case(case)
    k = best_slice(seg, axis=2)

    fig, ax = plt.subplots(figsize=(4.4, 4.4), dpi=140)
    draw(
        ax,
        window(t1ce[:, :, k]).T,
        (seg[:, :, k] > 0).T,
        seg[:, :, k].T,
    )
    fname = f"{case}_axial.png"
    fig.savefig(os.path.join(OUT_DIR, fname), facecolor="black", bbox_inches="tight")
    plt.close(fig)

    labels = {
        str(v): int((seg == v).sum()) for v in (1, 2, 4) if int((seg == v).sum()) > 0
    }
    eng_voxels, eng_grid = engine_mask_voxels(case)
    return {
        "case": case,
        "image": fname,
        # The engine-ready mask is <1 KB, so it is served too: the dashboard
        # can POST it to the live API and show a real run next to the scan.
        "engine_mask_file": f"{case}_mask.nii.gz",
        "plane": "axial",
        "slice_index": k,
        "grid_shape_xyz": [int(v) for v in t1ce.shape],
        "seg_voxels_by_label": labels,
        "seg_voxels_total": int((seg > 0).sum()),
        "engine_mask_voxels": eng_voxels,
        "engine_grid_xyz": eng_grid,
        "is_standard_case": case == STANDARD_CASE,
    }


def render_montage(case: str) -> dict:
    t1ce, seg = load_case(case)
    views = [
        ("axial", 2, "A"),
        ("coronal", 1, "C"),
        ("sagittal", 0, "S"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6), dpi=140)
    for ax, (name, axis, letter) in zip(axes, views):
        k = best_slice(seg, axis=axis)
        others = tuple(a for a in range(3) if a != axis)
        sl = np.take(seg, k, axis=axis)
        im = np.take(t1ce, k, axis=axis)
        draw(ax, window(im).T, (sl > 0).T, sl.T)
        ax.set_title(f"{name}  (slice {k})", color="white", fontsize=11)
    fig.suptitle(
        f"{case} · real T1ce MRI + BraTS expert segmentation (red: whole tumour, "
        f"yellow: enhancing)",
        color="white",
        fontsize=12,
    )
    fig.patch.set_facecolor("black")
    fname = f"{case}_views.png"
    fig.savefig(os.path.join(OUT_DIR, fname), facecolor="black", bbox_inches="tight")
    plt.close(fig)
    return {"file": fname, "case": case, "views": [v[0] for v in views]}


def mirror(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(src, "rb") as fh:
        blob = fh.read()
    with open(dst, "wb") as fh:
        fh.write(blob)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MIRROR_IMG_DIR, exist_ok=True)
    os.makedirs(MIRROR_MASK_DIR, exist_ok=True)

    cases = [render_case(c) for c in CASES]
    montage = render_montage(STANDARD_CASE)

    for name in os.listdir(OUT_DIR):
        if name.endswith(".png"):
            mirror(
                os.path.join(OUT_DIR, name), os.path.join(MIRROR_IMG_DIR, name)
            )

    for c in cases:
        mirror(
            os.path.join(ENGINE_DIR, c["engine_mask_file"]),
            os.path.join(MIRROR_MASK_DIR, c["engine_mask_file"]),
        )

    data = {
        "standard_case": STANDARD_CASE,
        "montage": montage,
        "cases": cases,
        "disclaimer": (
            "Images are real pre-operative patient scans committed to this "
            "repository (data/brats, 6 BraTS-2021 cases). The contours are the "
            "provided expert segmentation, not a model prediction -- no engine "
            "output is drawn on these images. The repository holds no "
            "post-operative imaging (claim #49), and orientation follows the "
            "NIfTI array order as stored."
        ),
    }
    artifact = {
        "key": "scans",
        "title": "Real MRI scans committed to this repository",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
        "source": "_scan_gallery.py over data/brats/*_{t1ce,seg}.nii.gz (BraTS-2021, cached locally)",
        "notes": [
            "Every pixel comes from the committed NIfTI files; matplotlib renders, nothing is simulated.",
            "Contours are the BraTS expert labels (1 = necrotic core, 2 = oedema, 4 = enhancing); no prediction is shown.",
            "engine_mask_voxels is the downsampled mask the engine actually ran on, from data/engine_ready/.",
            "engine_mask_file is that same mask served as a static asset, so 'Run live plan' posts the real mask to the local API instead of asking for a file you do not have.",
        ],
        "provenance": {
            "scan_pixels": "ESTABLISHED",
            "segmentation_contours": "ESTABLISHED",
            "engine_grid_voxels": "ESTABLISHED",
            "model_prediction_on_these_images": "PROVEN_FAILED",
        },
        "data": data,
    }
    blob = json.dumps(artifact, indent=1)
    for path in (
        os.path.join(ARTIFACT_DIR, "scans.json"),
        os.path.join(MIRROR_DIR, "scans.json"),
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(blob)

    print(f"wrote {len(cases)} case images + montage to {OUT_DIR}/")
    print(f"wrote scans.json to {ARTIFACT_DIR}/ and {MIRROR_DIR}/")
    for c in cases:
        print(
            f"  {c['case']}: axial {c['slice_index']}, seg {c['seg_voxels_total']} vox, "
            f"engine {c['engine_mask_voxels']} @ {c['engine_grid_xyz']}"
        )


if __name__ == "__main__":
    main()
