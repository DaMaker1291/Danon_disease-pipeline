"""Which mechanisms of the engine actually move its answer, on a real case.

Geometry: the committed engine-ready BraTS-2021 case (expert segmentation, label
derived proliferation) with the t1ce-derived transport field, so the transport
ablation has a field to remove. Everything is paired on seeds; the effect size is
reported in units of the intact arm's own seed-to-seed spread.

Writes outputs_real/ablation_cohort.json
Run:  python _ablation_cohort.py
"""

import json
import os
import time

from danon.ablation import MECHANISMS, ablate_all, what_an_ablation_cannot_show
from danon.clinical_twin import build_stream3_diffusion_field
from danon.imaging import from_nifti_mask

CASE = "BraTS2021_01017"
MASK = f"data/engine_ready/{CASE}_mask.nii.gz"
PROLIF = f"data/engine_ready/{CASE}_prolif.nii.gz"
VESSELS = f"data/engine_ready/{CASE}_vessels_t1ce.nii.gz"
OUT = os.path.join("outputs_real", "ablation_cohort.json")

#: The engine voxel is 4x4x3 mm; the committed file's header says 1 mm because
#: the derived-volume writer predates the fix (see danon.grid_artifact). For this
#: use the defect is inert: the simulator reduces any supplied field to a
#: mean-one RELATIVE pattern, so the absolute scale cancels and only the spatial
#: variation reaches the integrator. Passed explicitly so the provenance says so.
TRUE_VOXEL_MM = 3.634

SEEDS = (101, 102, 103)
DAYS = 365


def main() -> None:
    t0 = time.time()
    patient = from_nifti_mask(MASK, proliferative_path=PROLIF, vessels_path=VESSELS)
    geom = patient.geometry
    # The geometry's OWN vessel map: `_load_map` normalizes a raw uint8 file to
    # [0, 1] with a max-normalisation, and feeding the un-normalized 0-255 file
    # straight in would inflate the field's contrast by two orders of magnitude.
    # Using the map the rest of the pipeline uses is the only way the ablation's
    # intact arm is the pipeline's arm.
    field = build_stream3_diffusion_field(geom.vessels, voxel_mm=TRUE_VOXEL_MM)
    print(
        f"geometry: {int(geom.tumor_mask.sum())} tumour voxels on "
        f"{geom.tumor_mask.shape}, transport field "
        f"{float(field.d_engine.min()):.4g}-{float(field.d_engine.max()):.4g} "
        f"engine units",
        flush=True,
    )

    # WHY THE FIELD IS PASSED TO ONE MECHANISM ONLY. The transport ablation needs
    # a field to remove. The other ablations need the field held CONSTANT within
    # their own pair -- which the paired design gives them either way -- and the
    # heterogeneous transport branch costs a ~50-fold finer diffusion sub-step
    # (its pattern raises the stability peak), so carrying it through all seven
    # mechanisms multiplies the cohort's cost for no extra attribution. The
    # configuration actually used is recorded per mechanism rather than described
    # in prose.
    out = ablate_all(
        geom,
        mechanisms=list(MECHANISMS),
        diffusion_field=None,
        field_for=("heterogeneous_transport",),
        field=field,
        seeds=SEEDS,
        days=DAYS,
    )
    out["case"] = CASE
    out["mask_path"] = MASK
    out["vessels_path"] = VESSELS
    out["transport_field_voxel_mm"] = TRUE_VOXEL_MM
    out["transport_field_contrast"] = float(
        field.d_engine.max() / field.d_engine.min()
    )
    out["seeds"] = list(SEEDS)
    out["days"] = DAYS
    out["seconds"] = time.time() - t0
    out["what_this_cannot_show"] = what_an_ablation_cannot_show()

    print(f"\n{'mechanism':26s} {'arm':9s} {'readout':26s} {'delta/spread':>12s} verdict")
    for name, report in out["mechanisms"].items():
        for arm, entry in report["arms"].items():
            for readout, values in entry["readouts"].items():
                ratio = values["delta_over_spread"]
                print(
                    f"{name:26s} {arm:9s} {readout:26s} "
                    f"{(f'{ratio:12.3f}' if ratio is not None else '           -')} "
                    f"{values['verdict']}",
                    flush=True,
                )
    for name, reason in out["skipped"].items():
        print(f"SKIPPED {name}: {reason}")

    leaderboard = sorted(
        (
            (name, len(report["load_bearing_on"]))
            for name, report in out["mechanisms"].items()
        ),
        key=lambda kv: -kv[1],
    )
    out["leaderboard"] = [
        {"mechanism": name, "n_load_bearing_readouts": n} for name, n in leaderboard
    ]
    print(f"\nload-bearing readouts per mechanism: {leaderboard}")
    print(f"total {out['seconds']:.1f}s")

    os.makedirs("outputs_real", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
