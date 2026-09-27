"""Paired stream-2 cohort: uniform carrying capacity vs measured packing.

Usage: python _stream2_capacity_cohort.py CASE_INDEX [CASE_INDEX ...]
Runs each case and merges into outputs_real/stream2_capacity_cohort.json.

The reference and the K ratios are READ from the committed histology probe, not
chosen here: the reference is that slide's pooled density and the endpoints are
its own crop-to-crop extremes, so the sweep spans exactly the heterogeneity the
slide measured. Nothing differs between arms except `clones.carrying_capacity`:
same engine-ready mask, same proliferation, same seeds, same policy, same
initial resistant fraction. That is what makes any difference attributable to
the density mapping rather than to geometry or sampling.

The hypothesis under test comes from the engine's own code, not from hope: the
dynamics are scale-covariant in (seed, K), but two ABSOLUTE floors are not --
MIN_VIABLE_DENSITY (a voxel under half a cell is emptied) and
DETECTION_BURDEN_DENSITY (below it no policy may act). So a K rescale can only
matter where a floor bites. This measures whether it does.
"""

import json
import os
import sys
import time

from danon.clinical_twin import (
    HISTOLOGY_PROBE_PATH,
    stream2_capacity_contrast,
    stream2_carrying_capacity,
)

CASES = ["01017", "01018", "01019", "01020", "01021", "01022"]
SEEDS = (101, 102)
OUT = "outputs_real/stream2_capacity_cohort.json"


def density_endpoints() -> tuple:
    """(reference, role, ratios) taken from the committed histology probe."""
    with open(HISTOLOGY_PROBE_PATH) as fh:
        record = json.load(fh)
    density = record["slide_measured"]["density"]
    reference = float(density["nuclei_per_mm2"])
    spread = density["per_crop_spread"]
    ratios = [
        1.0,
        float(spread["min"]) / reference,
        float(spread["max"]) / reference,
    ]
    role = (
        f"the slide's own pooled density ({reference:g} nuclei/mm^2 over "
        f"{density['crops_used']} crops); endpoints are its measured crop "
        f"extremes {spread['min']}-{spread['max']}/mm^2"
    )
    return reference, role, ratios


def run_case(case: str, reference: float, role: str, ratios: list) -> dict:
    record = {"case_id": case, "arms": {}}
    for policy in ("mtd", "steering"):
        c = stream2_capacity_contrast(
            case,
            nuclei_per_mm2=reference,
            reference_nuclei_per_mm2=reference,
            reference_role=role,
            ratios=ratios,
            policy=policy,
            seeds=SEEDS,
        )
        record["arms"][policy] = {
            "policy": c.policy,
            "seeds": list(c.seeds),
            "ratios": list(c.ratios),
            "k_by_ratio": c.k_by_ratio,
            "pfs_uniform": list(c.pfs_uniform_days),
            "pfs_capacity": {k: list(v) for k, v in c.pfs_capacity_days.items()},
            "burden_uniform": list(c.burden_uniform),
            "burden_capacity": {k: list(v) for k, v in c.burden_capacity.items()},
            "median_pfs_delta_days": {
                f"{r:.6f}": c.median_pfs_delta_days(r) for r in c.ratios
            },
            "max_abs_relative_burden_difference": {
                f"{r:.6f}": c.max_abs_relative_burden_difference(r) for r in c.ratios
            },
            "n_pairs_in_disagreement": {
                f"{r:.6f}": c.n_pairs_in_disagreement(r) for r in c.ratios
            },
            "n_voxels": c.n_voxels,
        }
        print("  " + c.describe(), flush=True)
    return record


def main() -> None:
    reference, role, ratios = density_endpoints()
    print(
        f"reference {reference:g} nuclei/mm^2 (role: {role[:60]}...); "
        f"K ratios {[round(r, 4) for r in ratios]}",
        flush=True,
    )
    for ratio in ratios:
        result = stream2_carrying_capacity(reference * ratio, reference)
        print(f"  K x{ratio:.4f} -> {result.k:.4f}", flush=True)

    merged = json.load(open(OUT)) if os.path.exists(OUT) else []
    # Refuse to mix references. Every row's sweep is anchored to the density the
    # probe recorded when that row was run, so appending a case after the probe
    # changed would leave one file holding two different references -- and the
    # claim's check would (correctly) fail on it rather than silently averaging.
    stale = [
        row.get("case_id")
        for row in merged
        if float(row.get("reference_nuclei_per_mm2", -1.0)) != reference
    ]
    if stale:
        raise SystemExit(
            f"{OUT} holds cases measured against a different reference "
            f"({', '.join(str(c) for c in stale)}); the slide record now gives "
            f"{reference:g} nuclei/mm^2. Delete the record and re-run every case."
        )
    for idx in (int(a) for a in sys.argv[1:]):
        case = CASES[idx]
        if any(row.get("case_id") == case for row in merged):
            print(f"  {case}: already in {OUT}, skipping", flush=True)
            continue
        t0 = time.time()
        record = run_case(case, reference, role, ratios)
        record["reference_nuclei_per_mm2"] = reference
        record["reference_role"] = role
        record["seeds"] = list(SEEDS)
        merged.append(record)
        with open(OUT, "w") as fh:
            json.dump(merged, fh, indent=1)
        print(f"  {case}: {time.time() - t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
