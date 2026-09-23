"""Paired stream-3 cohort: real t1ce vasculature vs null-uniform, 6 BraTS cases.

Usage: python _stream3_cohort.py CASE_INDEX
Runs one case (12 engine runs: 2 policies x 2 geometries x 3 seeds, paired by
seed and geometry), merges into outputs_real/stream3_cohort.json. Run for
indices 0..5, then read the file for the cohort summary.

Same protocol as the day-365 outcome cards: r0 from the intake mapping,
seeds 101-103, 365-day horizon, paired same-seed comparison so a change in
vasculature is never confused with a change in geometry.
"""

import json
import os
import sys

import numpy as np

from danon.clinical_twin import build_stream3_vessels
from danon.evosteer import CollateralSwitchPolicy, MTDPolicy, TumorEvolutionSimulator
from danon.imaging import from_nifti_mask

CASES = ["01017", "01018", "01019", "01020", "01021", "01022"]
SEEDS = [101, 102, 103]
R0 = 0.005
OUT = "outputs_real/stream3_cohort.json"


def run_case(case: str) -> dict:
    seg = f"data/brats/BraTS2021_{case}_seg.nii.gz"
    t1ce = f"data/brats/BraTS2021_{case}_t1ce.nii.gz"
    mask_file = f"data/engine_ready/BraTS2021_{case}_mask.nii.gz"
    prolif_file = f"data/engine_ready/BraTS2021_{case}_prolif.nii.gz"
    vessels_file = f"data/engine_ready/BraTS2021_{case}_vessels_t1ce.nii.gz"

    info = build_stream3_vessels(t1ce, seg)

    pg_real = from_nifti_mask(
        mask_file, proliferative_path=prolif_file, vessels_path=vessels_file
    )
    pg_null = from_nifti_mask(mask_file, proliferative_path=prolif_file)
    geom_r, geom_n = pg_real.geometry, pg_null.geometry

    # The comparison must be vasculature-only: identical mask and proliferation.
    assert np.array_equal(geom_r.tumor_mask, geom_n.tumor_mask)
    assert np.array_equal(geom_r.proliferative, geom_n.proliferative)

    arms = {}
    for label, policy in (("mtd", MTDPolicy()), ("steering", CollateralSwitchPolicy())):
        pfs = {"real": [], "null": []}
        tox = {"real": [], "null": []}
        for seed in SEEDS:
            for geom_key, geom in (("real", geom_r), ("null", geom_n)):
                traj = TumorEvolutionSimulator(geom).run(
                    policy, days=365, initial_resistant_fraction=R0, seed=seed
                )
                pfs[geom_key].append(traj.pfs_day if traj.pfs_day is not None else 365)
                tox[geom_key].append(float(traj.toxicity))
        arms[label] = {
            "pfs_real": pfs["real"],
            "pfs_null": pfs["null"],
            "toxicity_real": tox["real"],
            "toxicity_null": tox["null"],
            "pfs_median_real": float(np.median(pfs["real"])),
            "pfs_median_null": float(np.median(pfs["null"])),
            "toxicity_median_real": float(np.median(tox["real"])),
            "toxicity_median_null": float(np.median(tox["null"])),
        }

    return {
        "case_id": case,
        "engine_grid": info["engine_grid"],
        "tumour_voxels": info["tumour_voxels"],
        "mean_enhancement_in_tumour": info["mean_enhancement_in_tumour"],
        "max_enhancement": info["max_enhancement"],
        "voxels_with_enhancement": info["voxels_with_enhancement"],
        "r0": R0,
        "seeds": SEEDS,
        "arms": arms,
    }


def main() -> None:
    idx = int(sys.argv[1])
    case = CASES[idx]
    record = run_case(case)

    merged = []
    if os.path.exists(OUT):
        merged = json.load(open(OUT))
    merged = [m for m in merged if m["case_id"] != case]
    merged.append(record)
    merged.sort(key=lambda m: m["case_id"])
    with open(OUT, "w") as fh:
        json.dump(merged, fh, indent=1)

    mtd = record["arms"]["mtd"]
    st = record["arms"]["steering"]
    print(
        f"{record['case_id']}: MTD real {mtd['pfs_median_real']:.0f}d "
        f"vs null {mtd['pfs_median_null']:.0f}d | steering real "
        f"{st['pfs_median_real']:.0f}d vs null {st['pfs_median_null']:.0f}d "
        f"({len(merged)}/{len(CASES)} cases done)"
    )


if __name__ == "__main__":
    main()
