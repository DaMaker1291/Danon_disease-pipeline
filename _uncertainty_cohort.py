"""Propagate the real 140-donor MGMT beta distribution through both arms.

Usage: python _uncertainty_cohort.py [MAX_DONORS]

Runs every (donor beta, seed) cell on the committed BraTS-2021 engine-ready
geometry for the MTD and switching arms, decomposes the outcome variance into
the input-driven and stochastic parts, and writes
outputs_real/uncertainty_budget.{txt,json}.

The geometry is committed, so a fresh clone can reproduce this without any
download; the betas are read from the cached per-probe panel the same way the
rest of the repository reads them.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np

from danon.imaging import from_nifti_mask
from danon.mgmt import load_mgmt_beta_panel, clone_params_for_mgmt_beta
from danon.uncertainty import (
    READOUTS,
    propagate_mgmt_uncertainty,
    render_uncertainty_report,
    uncertainty_provenance,
    what_this_decomposition_cannot_show,
)

CASE = "BraTS2021_01017"
ENGINE_READY = os.path.join("data", "engine_ready")
DAYS = 365
# Two replicates, not three, on purpose: the verdict turns on df_between (the
# number of donors, i.e. how well the cohort's own distribution is sampled),
# not on df_within. 140 donors x 2 seeds and 93 x 3 cost the same wall time and
# 140 x 2 estimates the distribution better, so every donor in the panel is
# used. The stochastic component is still estimated with 140 pooled df.
REPLICATES = 2
R0 = 0.01  # the MGMT sweep's own default, so the two records are comparable
MAX_DONORS = int(sys.argv[1]) if len(sys.argv) > 1 else None
OUT_JSON = os.path.join("outputs_real", "uncertainty_budget.json")
OUT_TXT = os.path.join("outputs_real", "uncertainty_budget.txt")


def main() -> int:
    mask = os.path.join(ENGINE_READY, f"{CASE}_mask.nii.gz")
    prolif = os.path.join(ENGINE_READY, f"{CASE}_prolif.nii.gz")
    for path in (mask, prolif):
        if not os.path.exists(path):
            raise SystemExit(
                f"{path} is not present; this measurement runs on the committed "
                "engine-ready case, not on generated input"
            )
    page = from_nifti_mask(mask, proliferative_path=prolif)

    panel = load_mgmt_beta_panel()
    if MAX_DONORS is not None:
        keep = panel.patient_ids[:MAX_DONORS]
        panel = type(panel)(
            profile=panel.profile,
            probes=panel.probes,
            betas={pid: panel.betas[pid] for pid in keep},
            source=panel.source,
            n_records=panel.n_records,
            notes=list(panel.notes),
        )
    print(f"geometry: {CASE} ({page.n_tumour_voxels:,} tumour voxels)")
    print(f"{panel}")
    print(f"running {panel.n_patients} donors x {REPLICATES} seeds x 2 arms ...")

    t0 = time.time()
    report = propagate_mgmt_uncertainty(
        page.geometry,
        panel,
        geometry_label=f"{CASE} (committed BraTS-2021 engine-ready mask + proliferation)",
        arms=("mtd", "steering"),
        days=DAYS,
        replicates=REPLICATES,
        initial_resistant_fraction=R0,
    )
    elapsed = time.time() - t0

    text = render_uncertainty_report(report)
    print()
    print(text)
    print()
    print(
        f"ran {panel.n_patients * REPLICATES * 2} engine runs in {elapsed:.1f} s "
        f"({elapsed / max(panel.n_patients * REPLICATES * 2, 1):.2f} s/run)"
    )

    payload = {
        "geometry": report.geometry,
        "n_voxels": report.n_voxels,
        "n_tumour_voxels": report.n_tumour_voxels,
        "days": report.days,
        "initial_resistant_fraction": report.initial_resistant_fraction,
        "input_quantity": report.input_quantity,
        "input_source": report.input_source,
        "n_donors": len(report.inputs),
        "input_band": asdict(report.input_band),
        "readouts": list(READOUTS),
        "replicates": REPLICATES,
        "is_credible_interval": report.is_credible_interval,
        "elapsed_seconds": elapsed,
        "arms": [
            {
                "arm": row.arm,
                "readout": row.readout,
                "propagated_band": asdict(row.propagated_band),
                "input_span": asdict(row.input_span),
                "verdict": row.verdict(),
                "budget": {
                    **{
                        k: v
                        for k, v in asdict(row.budget).items()
                    },
                    "f_ratio": row.budget.f_ratio,
                    "component_between": row.budget.component_between,
                    "component_within": row.budget.component_within,
                    "input_fraction": row.budget.input_fraction,
                    "is_degenerate": row.budget.is_degenerate,
                },
            }
            for row in report.arms
        ],
        "provenance": {k: v.value for k, v in uncertainty_provenance().items()},
        "cannot_show": list(what_this_decomposition_cannot_show()),
    }
    payload["input_band_lo"] = report.input_band.lo
    payload["input_band_hi"] = report.input_band.hi

    os.makedirs("outputs_real", exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
        fh.write("\n")
    print(f"wrote {OUT_TXT} and {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
