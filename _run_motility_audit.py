"""Price the cell-motility coefficient: grid dependence, and measured sensitivity.

THE DEFECT THIS MEASURES. `TumorEvolutionSimulator` integrates a cell-migration
term whose coefficient is `LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY = 0.02`, in
voxel^2/day. A per-voxel coefficient is a different physical diffusivity on every
grid:

  * phantom (1.0 mm voxels)        -> 0.020 mm^2/day
  * real decimated (3.667 mm voxels) -> 0.269 mm^2/day   (13.44x)

against a literature cell-motility range of 0.001-0.005 mm^2/day. So the shipped
value is 4-20x above range where the calibration anchors were set, and 54-269x
above range on the real-geometry path -- and nothing raises, because `voxel_mm`
is bookkeeping rather than a conversion. `danon/anisotropy.py` already PRICES
this in its unit audit; what no artifact recorded is whether it changes any
result, which is the question this runner answers.

WHAT IT DOES. Runs the release-1 winning calendar (read out of the artifact, not
retyped) on 30 seeds at the release-1 operating point under six motility
settings: the shipped default, motility off, the two ends of the literature
range expressed on the PHANTOM grid, and the two ends expressed so that they are
in range on the REAL grid (i.e. divided by the 13.44x grid factor). It also runs
the release-1 best single-drug reference arm at two of those settings, so the
ratio the claims actually quote can be checked for motility dependence.

WHAT IT REFUSES. It does not change the engine's default. Changing it would
invalidate `outputs_real/dosing.json` (release 1) and
`outputs_real/release2_sweep.json` (release 2), which are 90 minutes and 24
minutes of measurement respectively, and the honest sequence is to price the
change first and let the decision to re-run be an explicit one.

USAGE
-----
python _run_motility_audit.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "outputs_real", "motility_audit.json")
DAYS = 365
SEEDS = tuple(range(30))
R0 = 0.01
NOISE = 0.05


# ---------------------------------------------------------------------------
# The declared settings, in voxel^2/day, with their physical value on each grid
# ---------------------------------------------------------------------------

def _settings() -> List[Dict[str, Any]]:
    from danon.anisotropy import (
        CELL_MOTILITY_MM2_PER_DAY_RANGE,
        LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY as LEGACY,
        PHANTOM_VOXEL_MM,
        REAL_VOXEL_MM,
        d_to_physical,
    )

    lo, hi = CELL_MOTILITY_MM2_PER_DAY_RANGE
    grid_factor = (REAL_VOXEL_MM / PHANTOM_VOXEL_MM) ** 2

    def row(
        label: str,
        value: Optional[float],
        note: str,
        effective: Optional[float] = None,
    ) -> Dict[str, Any]:
        """One declared setting.

        `value is None` means "no override, use the engine's default", which is
        how the anchored arm is expressed. `effective` carries the number that
        default actually IS, so the row still shows a reader its physical value
        on both grids instead of a dash they would have to work out.
        """
        eff = value if value is not None else effective
        if eff is None:
            phantom = real = None
        else:
            phantom = d_to_physical(eff, PHANTOM_VOXEL_MM)
            real = d_to_physical(eff, REAL_VOXEL_MM)
        return {
            "label": label,
            "voxel2_per_day": value,
            "effective_voxel2_per_day": eff,
            "phantom_mm2_per_day": phantom,
            "real_mm2_per_day": real,
            "phantom_in_literature_range": (
                None if phantom is None else bool(lo <= phantom <= hi)
            ),
            "real_in_literature_range": (
                None if real is None else bool(lo <= real <= hi)
            ),
            "note": note,
        }

    return [
        row("legacy_default", None,
            "the shipped value: `None` keeps the engine's historical 0.02 "
            "voxel^2/day, so this arm is the calibration anchor everything else "
            "is compared against",
            effective=LEGACY),
        row("off", 0.0,
            "motility disabled: the control for 'does the term matter at all'"),
        row("literature_max_on_phantom", hi,
            "the top of the literature range read as a phantom-grid value; "
            "still 13.44x the range if the same literal runs on the real grid"),
        row("literature_min_on_phantom", lo,
            "the bottom of the literature range read as a phantom-grid value"),
        row("literature_max_on_real_grid", hi / grid_factor,
            "the top of the literature range converted to voxel^2/day for the "
            "3.667 mm real grid: the setting that is physical on real geometry"),
        row("literature_min_on_real_grid", lo / grid_factor,
            "the bottom of the literature range on the real grid"),
    ]


# ---------------------------------------------------------------------------
# The measurement (module-level so a worker process can pickle it)
# ---------------------------------------------------------------------------

def _release1_schedules() -> Dict[str, Any]:
    from danon.dosing import DoseBlock, Schedule

    with open(os.path.join(ROOT, "outputs_real", "dosing.json"), encoding="utf-8") as fh:
        rel1 = json.load(fh)
    combo = rel1["combination"]

    def build(schedule: Dict[str, Any]) -> Schedule:
        return Schedule(tuple(
            DoseBlock(
                drug=b["drug"],
                start_day=b["start_day"],
                duration_days=b["duration_days"],
                dose_level=b["dose_level"],
                dose_mg_m2=b["dose_mg_m2"],
            )
            for b in schedule["blocks"]
        ))

    return {
        "winner": build(combo["best_combination"]["schedule"]),
        "reference": build(combo["single_drug_best"]["schedule"]),
        "winner_name": combo["best_combination"]["name"],
        "reference_name": combo["single_drug_best"]["name"],
        "rel1_winner_aubc": combo["best_combination"]["median_aubc"],
        "rel1_winner_pfs": combo["best_combination"]["median_pfs_days"],
    }


def _one(payload: Tuple[str, Optional[float], str, Tuple[int, ...]]) -> Dict[str, Any]:
    """One (setting, arm) cell. Module-level so ProcessPoolExecutor can ship it."""
    import hashlib

    import numpy as np

    from danon.dosing import MyelosuppressionModel, evaluate_schedule
    from danon.evosteer import TumorEvolutionSimulator, make_gbm_phantom

    label, motility, arm, seeds = payload
    # The schedules are rebuilt per worker from the artifact rather than pickled
    # with the payload: same source of truth, smaller messages.
    schedules = _release1_schedules()
    schedule = schedules[arm]
    sim = TumorEvolutionSimulator(
        make_gbm_phantom(size=24, seed=12345), cell_motility=motility
    )
    score = evaluate_schedule(
        arm, schedule, sim, seeds, days=DAYS,
        initial_resistant_fraction=R0, growth_noise=NOISE,
        myelo=MyelosuppressionModel(),
    )
    geoms = [t for t in ()]  # placeholder: geometry is fixed, see note below
    _ = geoms
    return {
        "setting": label,
        "arm": arm,
        "n_seeds": len(seeds),
        "median_aubc": score.median_aubc,
        "iqr_aubc": list(score.iqr_aubc),
        "median_pfs_days": score.median_pfs_days,
        "median_dominance_day": score.median_dominance_day,
        "n_progressed": score.n_progressed,
        "feasible": bool(score.feasibility.feasible),
        "anc_nadir": score.feasibility.anc_nadir,
        "per_seed_aubc": (score.per_seed or {}).get("aubc"),
        "seed_set_sha256": hashlib.sha256(
            ",".join(str(s) for s in seeds).encode()
        ).hexdigest()[:16],
        "numpy": np.__version__,
    }


def main() -> int:
    started = time.time()
    print("=" * 78)
    print("MOTILITY COEFFICIENT: GRID DEPENDENCE AND MEASURED SENSITIVITY")
    print("=" * 78, flush=True)

    base = _release1_schedules()
    print(f"calendar : {base['winner_name']}")
    print(f"reference: {base['reference_name']}")
    print(f"seeds    : {len(SEEDS)} at r0={R0}, growth_noise={NOISE}, {DAYS} d\n",
          flush=True)

    settings = _settings()
    print(f"{'setting':30s} {'vox^2/d':>9s} {'phantom':>9s} {'real':>9s} "
          f"{'phantom ok':>10s} {'real ok':>8s}")
    print("-" * 78)
    for s in settings:
        v = ("default" if s["voxel2_per_day"] is None
             else f"{s['voxel2_per_day']:.6g}")
        ph = "-" if s["phantom_mm2_per_day"] is None else f"{s['phantom_mm2_per_day']:.4g}"
        re_ = "-" if s["real_mm2_per_day"] is None else f"{s['real_mm2_per_day']:.4g}"
        print(f"{s['label']:30s} {v:>9s} {ph:>9s} {re_:>9s} "
              f"{str(s['phantom_in_literature_range']):>10s} "
              f"{str(s['real_in_literature_range']):>8s}", flush=True)

    jobs: List[Tuple[str, Optional[float], str, Tuple[int, ...]]] = []
    for s in settings:
        jobs.append((s["label"], s["voxel2_per_day"], "winner", SEEDS))
    # The reference arm is measured where it decides the question: at the shipped
    # default (the anchor) and with motility off (the extreme).
    for label in ("legacy_default", "off"):
        value = next(s["voxel2_per_day"] for s in settings if s["label"] == label)
        jobs.append((label, value, "reference", SEEDS))

    print(f"\nrunning {len(jobs)} cells ({len(SEEDS)} seeds each) over "
          f"{min(4, len(jobs))} worker processes ...", flush=True)
    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(4, len(jobs))) as pool:
        futures = {pool.submit(_one, job): job for job in jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            res = fut.result()
            results.append(res)
            print(f"  done {job[2]:9s} @ {job[0]:30s} "
                  f"aubc {res['median_aubc']:.6f} pfs {res['median_pfs_days']:.0f}",
                  flush=True)

    by_arm = {"winner": {}, "reference": {}}
    for r in results:
        by_arm[r["arm"]][r["setting"]] = r

    anchor = by_arm["winner"]["legacy_default"]
    print("\n" + "=" * 78)
    print("SENSITIVITY, relative to the shipped default")
    print("=" * 78)
    print(f"{'setting':30s} {'median AUBC':>12s} {'vs default':>11s} "
          f"{'PFS':>6s} {'dominance':>10s}")
    print("-" * 78)
    rows: List[Dict[str, Any]] = []
    for s in settings:
        r = by_arm["winner"].get(s["label"])
        if r is None:
            continue
        delta = (r["median_aubc"] - anchor["median_aubc"]) / anchor["median_aubc"]
        print(f"{s['label']:30s} {r['median_aubc']:12.6f} {delta:+10.1%} "
              f"{r['median_pfs_days']:6.0f} {r['median_dominance_day']:10.0f}",
              flush=True)
        rows.append({
            **{k: s[k] for k in (
                "label", "voxel2_per_day", "effective_voxel2_per_day",
                "phantom_mm2_per_day", "real_mm2_per_day",
                "phantom_in_literature_range", "real_in_literature_range", "note",
            )},
            "winner": r,
            "aubc_delta_vs_default": delta,
        })

    ref_rows = {}
    for label, r in by_arm["reference"].items():
        w = by_arm["winner"].get(label)
        ref_rows[label] = {
            **r,
            "aubc_ratio_winner_over_reference": (
                None if not r["median_aubc"] else w["median_aubc"] / r["median_aubc"]
            ),
        }

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what_this_is": (
            "A measured sensitivity of the release-1 winning calendar to the "
            "engine's cell-motility coefficient, plus that coefficient's "
            "physical value on each grid. It prices a defect, it does not fix "
            "one: the engine's default is unchanged so the release-1 and "
            "release-2 artifacts remain valid."
        ),
        "defect": {
            "constant": "danon/anisotropy.py::LEGACY_CELL_MIGRATION_VOXEL2_PER_DAY",
            "value_voxel2_per_day": 0.02,
            "grid_dependence": (
                "a per-voxel coefficient is a different physical diffusivity on "
                "every grid: 0.020 mm^2/day on 1.0 mm phantom voxels and 0.269 "
                "mm^2/day on 3.667 mm decimated real voxels, a factor of 13.44"
            ),
            "literature_range_mm2_per_day": [0.001, 0.005],
            "shipped_is_above_range_on_phantom_by": 4.0,
            "shipped_is_above_range_on_real_grid_by": 53.8,
            "why_nothing_raises": (
                "voxel_mm is bookkeeping in this engine, not a conversion, so "
                "moving to a real grid silently rescales the physics"
            ),
            "priced_before_this_artifact": (
                "danon/anisotropy.py::unit_audit_report already reports the two "
                "physical values and the range; no artifact had measured what "
                "the choice does to a result"
            ),
        },
        "operating_point": {
            "schedule": base["winner_name"],
            "reference": base["reference_name"],
            "seeds": list(SEEDS),
            "days": DAYS,
            "initial_resistant_fraction": R0,
            "growth_noise": NOISE,
            "geometry": "24^3 phantom, seed 12345 (the release-1 substrate)",
        },
        "release_1_anchor": {
            "median_aubc": base["rel1_winner_aubc"],
            "median_pfs_days": base["rel1_winner_pfs"],
            "reproduced_by_default_arm": (
                None if anchor is None
                else abs(anchor["median_aubc"] - base["rel1_winner_aubc"]) < 1e-9
            ),
        },
        "settings": rows,
        "reference_arm": ref_rows,
        "wall_seconds": round(time.time() - started, 1),
        "cannot_show": [
            "This measures ONE calendar at ONE operating point on ONE synthetic "
            "phantom. It is not a sensitivity of the engine in general.",
            "No real-geometry run is in this artifact: the real-grid column is a "
            "unit conversion, not a simulation on a 3.667 mm grid.",
            "The literature range is a range for cell motility in tissue; whether "
            "glioma cells in this model's substrate should be at 0.001 or 0.005 "
            "is not decided here.",
        ],
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, OUT)

    print(f"\nreference arm (winner/single ratio):", flush=True)
    for label, r in ref_rows.items():
        ratio = r["aubc_ratio_winner_over_reference"]
        print(f"  {label:16s} ratio {ratio:.4f} "
              f"(release-1 quoted 0.0332)" if ratio else f"  {label}: n/a", flush=True)
    print(f"\nwrote {OUT}  ({payload['wall_seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
