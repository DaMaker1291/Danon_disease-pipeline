"""Paired stream-3 diffusion cohort: perfusion-derived vs uniform transport.

Usage: python _stream3_diffusion_cohort.py CASE_INDEX [CASE_INDEX ...]
Runs each case (2 policies x 2 transport arms x 3 seeds = 12 engine runs, paired
by policy and seed) and merges into outputs_real/stream3_diffusion_cohort.json.

The ONLY thing that differs between the arms is the transport field: same
engine-ready mask, same proliferation, same vessels source term, same seeds.
That is what makes a difference attributable to the mapping in
`build_stream3_diffusion_field` rather than to geometry or sampling.

Same protocol as `_stream3_cohort.py` and the day-365 outcome cards: real
BraTS-2021 t1ce through the segmentation's own engine transform, seeds 101-103,
365-day horizon, r0 from the intake mapping.
"""

import json
import os
import sys
import time

import numpy as np

from danon.clinical_twin import stream3_diffusion_contrast

CASES = ["01017", "01018", "01019", "01020", "01021", "01022"]
OUT = "outputs_real/stream3_diffusion_cohort.json"


def run_case(case: str) -> dict:
    record = {"case_id": case, "arms": {}}
    for policy in ("mtd", "steering"):
        c = stream3_diffusion_contrast(case, policy=policy)
        record["arms"][policy] = {
            "gamma": c.gamma,
            "n_voxels": c.n_voxels,
            "voxels_active": c.voxels_active,
            "field_min_mm2_per_day": c.field_min_mm2_per_day,
            "field_max_mm2_per_day": c.field_max_mm2_per_day,
            "field_contrast_ratio": c.field_contrast_ratio,
            "seeds": list(c.seeds),
            "pfs_uniform": list(c.pfs_uniform_days),
            "pfs_perfusion": list(c.pfs_perfusion_days),
            "toxicity_uniform": list(c.toxicity_uniform),
            "toxicity_perfusion": list(c.toxicity_perfusion),
            "median_pfs_delta_days": c.median_pfs_delta_days,
            "n_pairs_in_disagreement": c.n_pairs_in_disagreement,
        }
        print("  " + c.describe(), flush=True)
    return record


def main() -> None:
    merged = []
    if os.path.exists(OUT):
        merged = json.load(open(OUT))
    for idx in (int(a) for a in sys.argv[1:]):
        case = CASES[idx]
        t0 = time.time()
        record = run_case(case)
        merged = [m for m in merged if m["case_id"] != case]
        merged.append(record)
        merged.sort(key=lambda m: m["case_id"])
        with open(OUT, "w") as fh:
            json.dump(merged, fh, indent=1)
        print(f"{case} done in {time.time() - t0:.0f}s ({len(merged)}/{len(CASES)} cases)")


if __name__ == "__main__":
    main()
