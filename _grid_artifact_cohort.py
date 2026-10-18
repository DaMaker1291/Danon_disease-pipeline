"""Measure the grid artifact: the decimation rule, the header, the outcome.

Writes outputs_real/grid_artifact.{json,txt}. Three measurements, all on real
BraTS-2021 expert segmentations (data/brats), none of which depends on another:

  1. what `realdata._decimate_labels` actually produces, against the
     axis-aligned block majority vote its own docstring describes;
  2. the physical tumour volume each rule reports, and what the committed
     engine-ready files say their voxels are (header spacing);
  3. whether the grid resolution moves a paired trial, measured against the
     cohort's own seed-to-seed spread.

Run:  python _grid_artifact_cohort.py
"""

import json
import os
import time

from danon.grid_artifact import (
    GRID_LEVELS,
    committed_derived_audit,
    decimation_geometry,
    grid_resolution_cohort,
    render_grid_artifact_report,
    render_round_trip_report,
)

CASE = os.path.join("data", "brats", "BraTS2021_01017_seg.nii.gz")
OUT_JSON = os.path.join("outputs_real", "grid_artifact.json")
OUT_TXT = os.path.join("outputs_real", "grid_artifact.txt")


def main() -> None:
    t0 = time.time()
    print("== measurement 1+2: the rules and the geometry ==", flush=True)
    geometry = decimation_geometry(CASE, levels=GRID_LEVELS)
    print(render_grid_artifact_report(geometry), flush=True)
    print(f"[{time.time() - t0:.1f}s]", flush=True)

    print("\n== the committed derived volumes ==", flush=True)
    round_trip = committed_derived_audit()
    print(render_round_trip_report(round_trip), flush=True)
    print(f"[{time.time() - t0:.1f}s]", flush=True)

    print("\n== measurement 3: does the grid move a trial? ==", flush=True)
    cohort = grid_resolution_cohort(CASE)
    ref = cohort["reference_config"]
    for mode, per_config in cohort["contrasts"].items():
        print(f"\n[{mode}] reference = {ref}", flush=True)
        print(
            f"{'contrast':38s} {'arm':9s} {'readout':26s} {'rel delta':>12s} "
            f"{'ref spread':>11s} {'d/spread':>10s}",
            flush=True,
        )
        for name, arms in per_config.items():
            for arm, readouts in arms.items():
                for metric, entry in readouts.items():
                    rel = entry["mean_rel_delta"]
                    ratio = entry["delta_over_spread"]
                    print(
                        f"{name:38s} {arm:9s} {metric:26s} "
                        f"{(f'{rel * 100:+.4f}%' if rel is not None else 'n/a'):>12s} "
                        f"{entry['reference_spread_rel'] * 100:10.4f}% "
                        f"{(f'{ratio:.3f}' if ratio is not None else 'degenerate'):>10s}",
                        flush=True,
                    )
    print(f"\n[{time.time() - t0:.1f}s total]", flush=True)

    os.makedirs("outputs_real", exist_ok=True)
    report = {
        "case": CASE,
        "geometry": geometry,
        "round_trip": round_trip,
        "cohort": cohort,
        "seconds": time.time() - t0,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(render_grid_artifact_report(geometry))
        fh.write("\n\n")
        fh.write(render_round_trip_report(round_trip))
    print(f"wrote {OUT_JSON} and {OUT_TXT}")


if __name__ == "__main__":
    main()
