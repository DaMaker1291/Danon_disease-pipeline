"""Perfusion cohort: real downloaded DSC series -> rCBV -> paired transport arms.

Usage: python _perfusion_cohort.py CASE_INDEX [CASE_INDEX ...]

Selection rule (fixed before any outcome is seen): the N smallest DSC series by
the FileSize the TCIA listing reports, taken in ascending order of FileSize. The
rule is declared here because picking series by how their map turns out is the
one way this measurement could be made to say anything the author wanted.

Per case, four things happen:

  1. the series is downloaded from TCIA (NBIA v1 `getImage`) unless already
     cached at data/tcia/perfusion/<patient>/;
  2. `danon.perfusion` reads it into signal[t, z, y, x] and integrates the first
     pass into a relative-CBV map on the series' OWN grid;
  3. that map is block-averaged onto an engine-scale grid and thresholded into a
     perfusion-defined region of interest. It is NOT an expert tumour
     segmentation: no segmentation is co-registered to this DSC grid, so the
     cohort reports a ROI result and no claim about a patient's tumour;
  4. the engine runs paired -- identical geometry, proliferation, vessel source
     and seeds, with the only difference being the transport field (the measured
     D(x) vs the uniform default) -- for both policies, 3 seeds, 365 days.

Everything is merged into outputs_real/perfusion_cohort.json.
"""

import json
import os
import sys
import time
import zipfile

import numpy as np

from danon.clinical_twin import DIFFUSION_GAMMA, build_stream3_diffusion_field
from danon.imaging import from_nifti_mask, write_nifti
from danon.perfusion import build_dsc_series, relative_cbv

LISTING = "data/tcia/probe/upenn_series.json"
LISTING_URL = (
    "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
    "getSeries?Collection=UPENN-GBM"
)
IMAGE_URL = (
    "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
    "getImage?SeriesInstanceUID={uid}"
)
CACHE = "data/tcia/perfusion"
OUT = "outputs_real/perfusion_cohort.json"

#: Engine-grid decimation of the DSC grid (x, y, z). Declared: 128x128x14 ->
#: 32x32x7 voxels, the scale the committed BraTS arms already run at.
DECIMATION = (4, 4, 2)

#: The perfusion-defined ROI: engine voxels whose rCBV is at or above this
#: multiple of the map's median. DECLARED -- see the module docstring: this is
#: hyperperfused tissue, not a tumour segmentation.
ROI_RCBV_MULTIPLE = 1.5

POLICIES = ("mtd", "steering")
SEEDS = (101, 102, 103)
DAYS = 365


def dsc_series_candidates(limit: int) -> list:
    """The `limit` smallest DSC series in the listing, by reported FileSize."""

    def label(row: dict) -> str:
        return f"{row.get('ProtocolName') or ''} {row.get('SeriesDescription') or ''}".lower()

    if not os.path.exists(LISTING):
        os.makedirs(os.path.dirname(LISTING), exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve(LISTING_URL, LISTING)
    rows = json.load(open(LISTING))
    dsc = [r for r in rows if any(t in label(r) for t in ("perf", "bolus"))]
    picked, seen = [], set()
    for row in sorted(dsc, key=lambda r: int(r.get("FileSize") or 0)):
        pid = row["PatientID"]
        if pid in seen:
            continue
        seen.add(pid)
        picked.append(row)
        if len(picked) == limit:
            break
    if len(picked) < limit:
        raise SystemExit(f"listing holds {len(picked)} DSC patients, wanted {limit}")
    return picked


def fetch_series(row: dict) -> str:
    """Download and unpack one series; returns its local directory."""
    dest = os.path.join(CACHE, row["PatientID"])
    if os.path.isdir(dest) and any(n.endswith(".dcm") for n in os.listdir(dest)):
        return dest
    os.makedirs(dest, exist_ok=True)
    import urllib.request

    zip_path = os.path.join(dest, "_series.zip")
    urllib.request.urlretrieve(IMAGE_URL.format(uid=row["SeriesInstanceUID"]), zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith((".dcm", ".dicom", ".ima")):
                open(os.path.join(dest, os.path.basename(name)), "wb").write(
                    z.read(name)
                )
    os.remove(zip_path)
    return dest


def block_mean(volume: np.ndarray, factors: tuple) -> np.ndarray:
    """Block-average a [z, y, x] volume by (x, y, z) factors, trimming the edge."""
    fz, fy, fx = factors[2], factors[1], factors[0]
    nz, ny, nx = volume.shape
    nz_t, ny_t, nx_t = nz - nz % fz, ny - ny % fy, nx - nx % fx
    if min(nz_t, ny_t, nx_t) == 0:
        raise SystemExit(f"decimation {factors} collapses grid {volume.shape}")
    trimmed = volume[:nz_t, :ny_t, :nx_t]
    blocks = trimmed.reshape(nz_t // fz, fz, ny_t // fy, fy, nx_t // fx, fx)
    return blocks.mean(axis=(1, 3, 5))


def run_case(index: int, limit: int) -> dict:
    row = dsc_series_candidates(limit)[index]
    t0 = time.time()
    series_dir = fetch_series(row)
    series = build_dsc_series(series_dir)
    cbv = relative_cbv(series)
    rcbv = np.asarray(cbv.rcbv, dtype=float)
    rcbv_dec = block_mean(rcbv, DECIMATION)
    stats = cbv.stats()

    threshold = float(np.median(rcbv_dec) * ROI_RCBV_MULTIPLE)
    mask = rcbv_dec >= threshold

    os.makedirs("data/tcia/engine_ready", exist_ok=True)
    stem = f"UPENN_GBM_{row['PatientID'].split('-')[-1]}"
    mask_path = f"data/tcia/engine_ready/{stem}_roi.nii.gz"
    perf_path = f"data/tcia/engine_ready/{stem}_rcbv.nii.gz"
    write_nifti(mask_path, mask.astype(np.uint8))
    write_nifti(perf_path, np.clip(rcbv_dec * 40.0, 0, 255).astype(np.uint8))

    field = build_stream3_diffusion_field(rcbv_dec, gamma=DIFFUSION_GAMMA)
    page = from_nifti_mask(mask_path, vessels_path=perf_path)

    from danon.evosteer import CollateralSwitchPolicy, MTDPolicy, TumorEvolutionSimulator

    policies = {"mtd": MTDPolicy, "steering": CollateralSwitchPolicy}
    arms = {}
    for policy in POLICIES:
        u, p = {}, {}
        for seed in SEEDS:
            for tgt, sim in (
                (u, TumorEvolutionSimulator(page.geometry)),
                (p, TumorEvolutionSimulator(page.geometry, diffusion_field=field)),
            ):
                traj = sim.run(
                    policies[policy](), days=DAYS, initial_resistant_fraction=0.005, seed=seed
                )
                tgt.setdefault("pfs", []).append(
                    int(traj.pfs_day) if traj.pfs_day is not None else DAYS
                )
                tgt.setdefault("progressed", []).append(bool(traj.progressed))
                # Continuous readouts: a censored endpoint cannot separate arms in
                # which nothing progressed, so the burden is what carries resolution.
                tgt.setdefault("final_burden", []).append(
                    float(traj.burden_S[-1] + traj.burden_R[-1])
                )
                tgt.setdefault("final_r_fraction", []).append(
                    float(traj.final_resistant_fraction)
                )
                tgt.setdefault("toxicity", []).append(float(traj.toxicity))

        def rel_diffs(a: list, b: list) -> list:
            return [
                abs(y - x) / max(abs(x), 1e-30) for x, y in zip(a, b)
            ]

        burden_u, burden_p = u["final_burden"], p["final_burden"]
        diffs = rel_diffs(burden_u, burden_p)
        d = float(field.d_mm2_per_day[field.d_mm2_per_day > 0].min())
        h = float(field.d_mm2_per_day.max())
        arms[policy] = {
            "pfs_uniform": u["pfs"],
            "pfs_measured": p["pfs"],
            "progressed_uniform": u["progressed"],
            "progressed_measured": p["progressed"],
            "final_burden_uniform": burden_u,
            "final_burden_measured": burden_p,
            "final_r_fraction_uniform": u["final_r_fraction"],
            "final_r_fraction_measured": p["final_r_fraction"],
            "toxicity_uniform": u["toxicity"],
            "toxicity_measured": p["toxicity"],
            "pfs_informed_pairs": int(
                sum(
                    a or b
                    for a, b in zip(u["progressed"], p["progressed"])
                )
            ),
            "n_pairs_in_disagreement": int(sum(a != b for a, b in zip(u["pfs"], p["pfs"]))),
            # The null's resolution: the largest relative burden difference the
            # arms produced, which bounds any effect this contrast could have.
            "max_abs_relative_burden_difference": float(max(diffs)) if diffs else 0.0,
            "max_abs_r_fraction_difference": float(
                max(abs(y - x) for x, y in zip(u["final_r_fraction"], p["final_r_fraction"]))
            ),
            "field_min_mm2_per_day": d,
            "field_max_mm2_per_day": h,
            "field_contrast_ratio": (h / d) if d > 0 else float("inf"),
            "seeds": list(SEEDS),
        }
        a = arms[policy]
        print(
            f"  {row['PatientID']} {policy}: field {a['field_contrast_ratio']:.1f}x, "
            f"progressed pairs {a['pfs_informed_pairs']}/{len(SEEDS)}, "
            f"PFS deltas {a['n_pairs_in_disagreement']} differing, "
            f"final burden differs by at most {a['max_abs_relative_burden_difference']:.3%}",
            flush=True,
        )

    return {
        "patient_id": row["PatientID"],
        "series_uid": row["SeriesInstanceUID"],
        "series_description": row.get("SeriesDescription"),
        "bytes_listed": int(row.get("FileSize") or 0),
        "listing_url": LISTING_URL,
        "n_files": series.n_files,
        "dsc": series.provenance(),
        "cbv": {
            "peak_frame": cbv.peak_frame,
            "arrival_frame": cbv.arrival_frame,
            "end_frame": cbv.end_frame,
            "window_frames": cbv.window_frames,
            "drift_peak_ratio": cbv.drift_peak_ratio,
            "reference_voxels": cbv.reference_voxels,
            **stats,
        },
        "engine": {
            "decimation_xyz": list(DECIMATION),
            "grid_shape_zyx": [int(v) for v in rcbv_dec.shape],
            "roi_rcbv_multiple": ROI_RCBV_MULTIPLE,
            "roi_threshold_rcbv": threshold,
            "roi_voxels": int(mask.sum()),
            "total_voxels": int(mask.size),
            "mask_path": mask_path,
            "perfusion_path": perf_path,
            "geometry_source": "from_nifti_mask(mask, vessels_path=measured rCBV)",
            "arms": arms,
        },
        "honesty": {
            "roi_is_not_a_segmentation": (
                "the ROI is rCBV >= "
                f"{ROI_RCBV_MULTIPLE}x median on the decimated DSC grid; no expert "
                "segmentation is co-registered to this grid, so this is a "
                "perfusion-defined region, not a tumour"
            ),
            "not_claimed": [
                "absolute CBV or Ktrans",
                "a voxel-wise link between this map and any BraTS case",
                "survival of any patient",
            ],
        },
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    limit = int(os.environ.get("PERFUSION_COHORT_LIMIT", "3"))
    merged = json.load(open(OUT)) if os.path.exists(OUT) else []
    for index in (int(a) for a in sys.argv[1:]):
        record = run_case(index, limit)
        merged = [m for m in merged if m["patient_id"] != record["patient_id"]]
        merged.append(record)
        merged.sort(key=lambda m: m["bytes_listed"])
        with open(OUT, "w") as fh:
            json.dump(merged, fh, indent=1)
        print(f"{record['patient_id']} done in {record['seconds']}s ({len(merged)} cases)")


if __name__ == "__main__":
    main()
