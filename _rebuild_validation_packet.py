"""Rebuild outputs_real/validation_packet.txt from live measurements.

Run after any change to the engine's dynamics: the packet's layer-3 robustness
counts and its layer-2/layer-4 control-arm numbers all come from the engine, so
a physics change silently invalidates them otherwise.

Usage: PYTHONPATH=. python _rebuild_validation_packet.py
"""

from __future__ import annotations

import json

from danon.imaging import from_nifti_mask
from danon.reporting import validation_packet
from danon.retrospective import (
    load_campaign_log,
    load_tcga_gbm_clinical,
    model_curve_from_campaign_log,
)
from danon.sensitivity import run_sensitivity

CASE_IDS = ("BraTS2021_01017", "BraTS2021_01018")
R0 = 0.005
DAYS = 365
TEST_COUNT = 1022  # tests/ + danon/multiagent/ collected


def _val(x):
    return x() if callable(x) else x


def main() -> None:
    records = []
    for cid in CASE_IDS:
        geom = from_nifti_mask(
            f"data/engine_ready/{cid}_mask.nii.gz",
            proliferative_path=f"data/engine_ready/{cid}_prolif.nii.gz",
        ).geometry
        report = run_sensitivity(geom, R0, case_id=cid, days=DAYS, seeds=(101,))
        for o in _val(report.outcomes):
            records.append(
                {
                    "target": o.target,
                    "scale": float(o.scale),
                    "burden_ratio": float(o.burden_ratio),
                    "policy_wins": bool(_val(o.policy_wins)),
                }
            )
        print(f"  swept {cid}", flush=True)
    print(f"sensitivity records: {len(records)}", flush=True)

    cohort = load_tcga_gbm_clinical()
    times, events = model_curve_from_campaign_log(load_campaign_log(), arm="mtd")
    print(f"model MTD arm: n={len(times)}, events={sum(events)}", flush=True)

    validation_packet(
        cohort,
        times,
        events,
        sensitivity_records=records,
        test_count=TEST_COUNT,
        out_path="outputs_real/validation_packet.txt",
    )
    json.dump(records, open("/tmp/sens_records.json", "w"))
    print("PACKET_DONE", flush=True)


if __name__ == "__main__":
    main()
