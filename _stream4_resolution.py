"""What the manifest's stream-4 cohort can resolve at all (design calc, ~1 min).

Usage: python _stream4_resolution.py
Writes outputs_real/stream4_resolution.json.

This does not measure delta-volume, perfusion, fluorescence or Ki-67. It
measures the resolution of the *cohort* the CFB-GBM availability manifest
describes, by resampling the real cached GBM cohort down to that size. The
number it produces is the floor a claimed incremental C-index has to clear
before a download of 168 patients' images is worth performing.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from danon.availability import (
    CFB_GBM_DOI,
    CFB_GBM_MANIFEST,
    mri_availability_census,
)
from danon.cohort_features import _cv, resolvable_delta_c_index
from danon.predictive import load_cohort_rows

OUT_PATH = os.path.join("outputs_real", "stream4_resolution.json")
N_DRAWS = 24


def main() -> int:
    t0 = time.time()
    census = mri_availability_census()
    n_manifest = int(census.n_patients_with_usable_pair)
    rows, load_notes = load_cohort_rows()
    print(f"manifest describes {n_manifest} patients with a usable pair; "
          f"cached cohort n={len(rows)}")

    full = _cv(rows, None, n_folds=5, seed=0)
    print(f"full cohort: {full}")

    resolution = resolvable_delta_c_index(
        rows, n_patients=n_manifest, n_draws=N_DRAWS, n_folds=5
    )
    print(f"manifest cohort: {resolution}")

    payload = {
        "what_this_is": (
            "a DESIGN calculation, not a measurement of any datastream: the "
            "spread of the held-out C-index over independent cohorts of the "
            "manifest's size, resampled from the cached real cohort"
        ),
        "source_cohort": "cached GDC clinical export (TCGA-GBM), overall survival",
        "source_n_patients": len(rows),
        "source_base_c": float(full.mean_c),
        "source_base_fold_sd": float(full.sd_c),
        "source_age_only_c": float(full.age_only_c),
        "manifest_source": census.source,
        "manifest_doi": CFB_GBM_DOI,
        "manifest_path": CFB_GBM_MANIFEST,
        "manifest_n_patients_with_usable_pair": n_manifest,
        "manifest_n_usable_pairs": int(census.n_usable_pairs),
        "resolution": {
            "n_patients": resolution.n_patients,
            "n_draws": resolution.n_draws,
            "n_folds": resolution.n_folds,
            "mean_c": resolution.mean_c,
            "se_across_draws": resolution.se_across_draws,
            "mean_fold_sd": resolution.mean_fold_sd,
            "age_only_c": resolution.age_only_c,
            "floor_optimistic": resolution.floor_optimistic,
            "floor_conservative": resolution.floor_conservative,
            "exhaustive_draws": resolution.exhaustive_draws,
            "notes": resolution.notes,
        },
        "load_notes": list(load_notes),
        "built_seconds": round(time.time() - t0, 1),
        "rebuild_command": "python _stream4_resolution.py",
        "what_this_cannot_show": [
            "It does not measure delta-volume, registration error, or scanner "
            "comparability: the CFB-GBM images are not downloaded.",
            "It says nothing about progression-free survival; the cached cohort "
            "is scored on overall survival, which is the endpoint the "
            "availability manifest does not carry at all.",
            "A resolved floor is a permission to look, not evidence that any "
            "stream carries a signal.",
        ],
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"wrote {OUT_PATH} ({os.path.getsize(OUT_PATH)} B) in "
          f"{time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
