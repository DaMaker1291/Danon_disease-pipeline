"""The semi-synthetic 5-ALA standard, run once and recorded.

Plants fluorescence fields on the committed BraTS-2021 case's pre-operative rim,
sends them through the real stream-1 intake, runs the engine under MTD with the
same seeds for every layout, and writes what moved. Every entry is synthetic;
the record says so and the claim check requires it to keep saying so.

Writes outputs_real/fluorescence_standard.json
Run:  python _fluorescence_standard_cohort.py
"""

import dataclasses
import datetime
import json
import os

from danon.external_cohorts import file_sha256
from danon.fluorescence_standard import (
    MASK_PATH,
    PROLIF_PATH,
    STANDARD_CASE,
    STANDARD_RECORD,
    SYNTHETIC_DISCLAIMER,
    VESSELS_PATH,
    fluorescence_standard_provenance,
    layout_contrast,
    layout_summary,
    standard_refusals,
    standard_report,
    stream1_still_refused,
    what_a_synthetic_input_cannot_show,
)


def main() -> None:
    report = standard_report()
    summary = layout_summary(report)
    contrast = layout_contrast(report)
    refused, census_detail = stream1_still_refused()

    payload = {
        "generated_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "case": STANDARD_CASE,
            "files": {
                os.path.basename(p): {
                    "path": p,
                    "sha256": file_sha256(p),
                    "bytes": os.path.getsize(p),
                }
                for p in (MASK_PATH, PROLIF_PATH, VESSELS_PATH)
            },
            "note": (
                "expert BraTS-2021 segmentation and its derived engine-ready "
                "maps; PRE-OPERATIVE only, so the planted field sits on the "
                "tumour rim and not on any resection cavity"
            ),
        },
        "synthetic_disclaimer": SYNTHETIC_DISCLAIMER,
        "geometry": {
            "grid": list(report.grid),
            "tumour_voxels": report.tumour_voxels,
            "voxel_mm": report.voxel_mm,
        },
        "identity": report.identity,
        "descriptors": [dataclasses.asdict(d) for d in report.descriptors],
        "runs": [dataclasses.asdict(r) for r in report.runs],
        "summary": summary,
        "layout_contrast": contrast,
        "stream1_census": {"still_refused": refused, "detail": census_detail},
        "refusals": standard_refusals(),
        "provenance": {k: v.value for k, v in fluorescence_standard_provenance().items()},
        "notes": report.notes,
        "what_a_synthetic_input_cannot_show": what_a_synthetic_input_cannot_show(),
    }

    os.makedirs(os.path.dirname(STANDARD_RECORD), exist_ok=True)
    with open(STANDARD_RECORD, "w") as fh:
        json.dump(payload, fh, indent=1)

    ident = report.identity
    print(
        f"identity: stream1 uniform total {ident['stream1_uniform_total']:.4f} vs "
        f"engine target {ident['engine_target_total']:.4f} "
        f"(|diff| {ident['absolute_difference']:.2e})",
        flush=True,
    )
    for d in report.descriptors:
        print(" ", d.describe(), flush=True)
    for row in summary:
        print(
            f"  {row['mode']:<8s} {row['days']:>3d}d  median PFS "
            f"{row['median_pfs_day']:.1f}, median burden "
            f"{row['median_final_burden']:.2f}, spread "
            f"{row['final_burden_spread']:.2f}"
            + (
                f", paired delta vs default "
                f"{row['median_paired_delta_vs_default']:+.2f}"
                if row["median_paired_delta_vs_default"] is not None
                else ""
            ),
            flush=True,
        )
    for days, row in contrast.items():
        print(
            f"  stream-1 modes at {days}d: median burden "
            f"{row['median_final_burden']:.2f}, relative spread "
            f"{100 * row['relative_spread']:.3f}%",
            flush=True,
        )
    print(f"stream 1 census still refused: {refused} ({census_detail})", flush=True)
    print(f"wrote {STANDARD_RECORD}", flush=True)


if __name__ == "__main__":
    main()
