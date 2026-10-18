"""Release-2: the definitive *in silico* reversal map.

WHAT THIS MEASURES. Release 1 (`outputs_real/dosing.json`) answered one
question at one operating point: at the declared resistant fraction (0.01) and
growth noise (0.05), does an alternating two-drug calendar beat the best
single-drug calendar? It does, by a wide margin. Release 2 asks the question a
clinician actually has to answer before believing it:

    HOW FAR CAN THAT OPERATING POINT MOVE BEFORE THE WIN GOES AWAY, AND
    WHERE DOES THE CALENDAR STOP BEING SAFE?

Two boundaries matter, and they pull in opposite directions:

  * LOWER boundary (efficacy) -- too little intensity and the alternating
    calendar stops holding the resistant clone down. Measured by the engine.
  * UPPER boundary (safety) -- too much intensity and the marrow screen breaks
    the FDA label's ANC floor. Measured by the screen, which costs no engine
    time, so it is mapped on a fine grid.

The winner is not re-discovered here. It is READ from the release-1 artifact
and re-run at declared intensities, which is why this is a reversal map and
not a second search. Every cell is paired: the winner arm and the release-1
best single-drug arm run on the SAME seeds, so the only difference between
them is the calendar. The reconstruction is checked against release-1: at
intensity 1.0 this script's winner arm reproduces the release-1 AUBC exactly.

USAGE
-----
python _release2_sweep.py            # declared grid, 30 seeds (the release-1 seeds)
python _release2_sweep.py --quick    # 6 seeds, for smoke-testing the plumbing
python _release2_sweep.py --groups 2 # at most 2 (r0, noise) groups, then exit

Runtime is measured, not guessed: sequential cost here is 3.2 s per seed-run
and the measured 4-worker candidate pool runs at 3.6x, so one (resistant
fraction, noise) group is about four minutes and the declared grid is about
half an hour. Partial results are written after every group and a re-run
resumes instead of restarting, and `--groups` exists because a long unattended
background run on this box is not reliable: staged invocations that finish and
exit keep the artifact, whereas a killed background job can leave workers
frozen mid-future. A stage that stops early leaves `partial: true` in the
artifact, which the registry check refuses -- an unfinished map cannot support
a boundary claim.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from danon.dosing import (  # noqa: E402
    DoseBlock,
    MEASURED_WORKER_OPTIMUM,
    MyelosuppressionModel,
    Schedule,
    evaluate_candidates,
    engine_identity,
    provenance,
    screen_schedule,
)
from danon.evosteer import TumorEvolutionSimulator, make_gbm_phantom  # noqa: E402

# The SAME engine substrate release 1 ran on. A reversal map measured on a
# different phantom would not be a reversal map of the release-1 result.
PHANTOM_SIZE = 24
PHANTOM_SEED = 12345
OUT = ROOT / "outputs_real" / "release2_sweep.json"

# ---------------------------------------------------------------------------
# The declared release-2 grid. Declared here, printed into the artifact, and
# not tuned after the fact: a grid adjusted until the win survives is not a
# reversal map.
# ---------------------------------------------------------------------------
GRID = {
    # Multiplier on the RELEASE-1 WINNER'S OWN engine levels. 1.0 is exactly
    # the release-1 calendar (verified: AUBC 0.0433, the release-1 number).
    # The axis is centred on the free screen's own boundary, which this script
    # measures first: the conservative marrow weight breaks the ANC floor just
    # above 1.1, so 1.25 and 1.5 are the cells where efficacy and safety pull
    # apart. At 2.0 drug B's level would hit the engine's [0,1] cap and the
    # calendar is unsafe under every declared marrow weight, so the axis stops
    # where the information stops.
    "intensity": (0.5, 0.75, 1.0, 1.25, 1.5),
    # Pre-existing resistance, the axis that decides whether alternation has
    # anything to exploit: 0.001 is nearly all sensitive, 0.3 is a tumour that
    # starts out mostly resistant.
    "resistant_fraction": (0.001, 0.01, 0.1, 0.3),
    "growth_noise": (0.05, 0.2),
    # The upper (safety) boundary is analytic and free, so it is mapped finer
    # than the engine axis.
    "feasibility_intensity_curve": tuple(
        round(0.05 * i, 3) for i in range(1, 41)
    ),
    "marrow_weights_swept": (1.0, 0.5),
}
DAYS = 365
ALL_SEEDS = tuple(range(30))
QUICK_SEEDS = tuple(range(6))


def build_sim() -> TumorEvolutionSimulator:
    return TumorEvolutionSimulator(
        make_gbm_phantom(size=PHANTOM_SIZE, seed=PHANTOM_SEED)
    )


def write_artifact(payload: dict) -> None:
    """Temp file + os.replace, the same atomicity `_run_dosing.py` uses, so an
    interrupted sweep never leaves a half-written artifact behind."""
    os.makedirs(OUT.parent, exist_ok=True)
    tmp = str(OUT) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, OUT)


def load_release1() -> dict:
    path = ROOT / "outputs_real" / "dosing.json"
    if not path.exists():
        raise SystemExit("outputs_real/dosing.json missing; release-1 not run")
    return json.loads(path.read_text(encoding="utf-8"))


def _schedule_from_dict(blocks: list) -> Schedule:
    return Schedule(tuple(
        DoseBlock(
            drug=b["drug"],
            start_day=b["start_day"],
            duration_days=b["duration_days"],
            dose_level=b["dose_level"],
            dose_mg_m2=b["dose_mg_m2"],
        )
        for b in blocks
    ))


def winner_shape(rel1: dict) -> dict:
    """Read the release-1 winner's shape OUT of the artifact.

    Nothing about the calendar is hardcoded here on purpose: if the release-1
    winner changes, this sweep follows it instead of silently sweeping a
    calendar nobody ran.
    """
    best = rel1["combination"]["best_combination"]
    blocks = sorted(best["schedule"]["blocks"], key=lambda b: b["start_day"])
    a_blocks = [b for b in blocks if b["drug"] == "A"]
    b_blocks = [b for b in blocks if b["drug"] == "B"]
    if not a_blocks or not b_blocks:
        raise SystemExit("release-1 winner is not an alternating calendar")
    block_days = a_blocks[0]["duration_days"]
    period_a = (
        a_blocks[1]["start_day"] - a_blocks[0]["start_day"]
        if len(a_blocks) > 1 else 2 * block_days
    )
    # `Schedule.alternating` advances one block per turn, so drug A recurs every
    # 2 * (block + rest) days.
    rest_days = max(0, period_a // 2 - block_days)
    return {
        "name": best["name"],
        "dose_mg_m2": a_blocks[0]["dose_mg_m2"],
        "dose_level_a": a_blocks[0]["dose_level"],
        "dose_level_b": b_blocks[0]["dose_level"],
        "block_days": block_days,
        "rest_days": rest_days,
    }


def winner_at(shape: dict, intensity: float) -> Schedule:
    return Schedule.alternating_relative(
        drug_b_dose_level=shape["dose_level_b"] * intensity,
        block_days=shape["block_days"],
        horizon_days=DAYS,
        dose_mg_m2=shape["dose_mg_m2"],
        dose_level_a=shape["dose_level_a"] * intensity,
        rest_days=shape["rest_days"],
    )


def reference_schedule(rel1: dict) -> tuple[str, Schedule]:
    single = rel1["combination"]["single_drug_best"]
    return single["name"], _schedule_from_dict(single["schedule"]["blocks"])


def feasibility_curve(shape: dict, myelo: MyelosuppressionModel) -> dict:
    """Where the marrow screen breaks, at fine resolution, for free.

    The screen is analytic -- no engine run -- so the upper (safety) boundary
    is mapped on the same fine axis the engine axis cannot afford.
    """
    out = {}
    for weight in GRID["marrow_weights_swept"]:
        model = replace(myelo, drug_weights={**dict(myelo.drug_weights), "B": weight})
        rows = []
        for intensity in GRID["feasibility_intensity_curve"]:
            sched = winner_at(shape, intensity)
            report = screen_schedule(sched, DAYS, model)
            rows.append({
                "intensity": intensity,
                "anc_nadir": round(report.anc_nadir, 4),
                "feasible": bool(report.feasible),
            })
        feasible = [r["intensity"] for r in rows if r["feasible"]]
        out[str(weight)] = {
            "drug_b_marrow_weight": weight,
            "rows": rows,
            "max_feasible_intensity": max(feasible) if feasible else None,
            "note": (
                "the screen's ANC floor is the FDA label threshold; drug B's "
                "marrow potency per mg/m2 is an assumption and is swept, never "
                "set to 0.0 to make a calendar pass"
            ),
        }
    return out


def cell_status(winner, reference) -> dict:
    """The verdict for one paired cell, with the sign of the effect named."""
    w_aubc, r_aubc = winner.median_aubc, reference.median_aubc
    ratio = w_aubc / r_aubc if r_aubc else float("inf")
    delta_dom = winner.median_dominance_day - reference.median_dominance_day
    delta_pfs = winner.median_pfs_days - reference.median_pfs_days
    if not winner.feasibility.feasible:
        status = "INFEASIBLE"
    elif ratio >= 1.0 or delta_dom <= 0.0:
        status = "REVERSED"
    elif winner.median_pfs_days >= DAYS:
        status = "SATURATES"
    else:
        status = "WINS"
    return {
        "status": status,
        "aubc_ratio_winner_over_reference": round(ratio, 4),
        "dominance_delta_days": round(delta_dom, 1),
        "pfs_delta_days": round(delta_pfs, 1),
        "winner_anc_nadir": round(winner.feasibility.anc_nadir, 3),
        "reference_anc_nadir": round(reference.feasibility.anc_nadir, 3),
    }


def arm_summary(score) -> dict:
    # median_aubc and its IQR are stored at FULL precision, not rounded for
    # display: the registry check that ties this map back to release 1 compares
    # the intensity-1.0 cell against the release-1 number, and a 4-decimal
    # rounding would make a reproduction fail by arithmetic rather than by
    # science.
    return {
        "name": score.name,
        "median_aubc": score.median_aubc,
        "iqr_aubc": list(score.iqr_aubc),
        "median_dominance_day": score.median_dominance_day,
        "median_pfs_days": score.median_pfs_days,
        "median_toxicity": score.median_toxicity,
        "n_progressed": score.n_progressed,
        "feasible": bool(score.feasibility.feasible),
        "anc_nadir": round(score.feasibility.anc_nadir, 3),
        "engine_runs": score.engine_runs,
    }


def main() -> None:
    quick = "--quick" in sys.argv
    seeds = QUICK_SEEDS if quick else ALL_SEEDS
    # Bounded stages: see the module docstring for why this exists.
    max_groups = 0
    if "--groups" in sys.argv:
        max_groups = int(sys.argv[sys.argv.index("--groups") + 1])
    processed = 0
    stopped_early = False
    t_start = time.time()
    rel1 = load_release1()
    shape = winner_shape(rel1)
    ref_name, ref_sched = reference_schedule(rel1)
    combo = rel1["combination"]
    notes = list(combo.get("notes", []))

    myelo = MyelosuppressionModel()
    sim = build_sim()
    print("=" * 78)
    print("RELEASE 2: REVERSAL MAP" + ("  [QUICK MODE]" if quick else ""))
    print("=" * 78)
    print(f"release-1 winner : {shape['name']}")
    print(f"  shape          : A level {shape['dose_level_a']:g}, B level "
          f"{shape['dose_level_b']:g}, block {shape['block_days']} d, rest "
          f"{shape['rest_days']} d at {shape['dose_mg_m2']:g} mg/m2 face")
    print(f"reference arm    : {ref_name}")
    print(f"seeds            : {len(seeds)} ({'quick' if quick else 'declared'})")
    print(f"grid             : intensity {GRID['intensity']} x r0 "
          f"{GRID['resistant_fraction']} x noise {GRID['growth_noise']}")
    print(f"workers          : {MEASURED_WORKER_OPTIMUM} (measured optimum)")

    out_path = OUT
    out = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "geometry": {
            "grid": list(sim.geom.tumor_mask.shape),
            "tumour_voxels": int(sim.geom.tumor_mask.sum()),
            "voxel_mm": 1.0,
            "phantom_seed": PHANTOM_SEED,
            "is_patient_imaging": False,
        },
        "release_1_delivered": "outputs_real/dosing.json",
        "release_2_grid": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in GRID.items()},
        "seeds": list(seeds),
        "partial": True,
        "release_1_winner": {
            "schedule_text": shape["name"],
            "beats_single": combo.get("beats_single"),
            "aubc_ratio": combo.get("aubc_ratio_combination_over_single"),
            "dominance_delta_days": combo.get("dominance_delta_days"),
            "notes": notes,
        },
        "release_1_reference_arm": {"name": ref_name, "seeds": list(seeds)},
        "feasibility": feasibility_curve(shape, myelo),
        "cells": [],
        "reversals": [],
        "engine_runs": 0,
    }

    # Resume: keep completed groups from a prior partial run of the same grid.
    done_groups = set()
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if (prev.get("seeds") == list(seeds)
                    and prev.get("release_2_grid", {}).get("intensity")
                    == out["release_2_grid"]["intensity"]):
                out["cells"] = prev.get("cells", [])
                done_groups = {
                    (c["resistant_fraction"], c["growth_noise"])
                    for c in out["cells"]
                }
                out["engine_runs"] = prev.get("engine_runs", 0)
                print(f"resuming: {len(done_groups)} groups already on disk")
        except (json.JSONDecodeError, KeyError):
            pass

    for r0 in GRID["resistant_fraction"]:
        for noise in GRID["growth_noise"]:
            if (r0, noise) in done_groups:
                print(f"\n[skip] r0={r0} noise={noise} (already measured)")
                continue
            if max_groups and processed >= max_groups:
                stopped_early = True
                break
            print(f"\n[group] r0={r0} noise={noise}")
            t_group = time.time()
            cands = [
                (f"release1_winner x{intensity:g}", winner_at(shape, intensity))
                for intensity in GRID["intensity"]
            ] + [(f"release1_best_single[{ref_name}]", ref_sched)]

            def progress(done, total, name, _t=t_group):  # noqa: ANN001
                print(f"    {done}/{total} arms ({name}) "
                      f"{time.time() - _t:6.0f}s", flush=True)

            try:
                scores = evaluate_candidates(
                    cands, sim, seeds, days=DAYS,
                    initial_resistant_fraction=r0, growth_noise=noise,
                    myelo=myelo, max_workers=MEASURED_WORKER_OPTIMUM,
                    progress=progress,
                )
            except Exception as exc:  # noqa: BLE001
                # A worker pool can die mid-run (observed: BrokenProcessPool).
                # The numbers do not depend on how the work was scheduled, so
                # the group is re-run sequentially rather than lost.
                print(f"  [group fallback] {type(exc).__name__}: {exc}; "
                      f"re-running this group with 1 worker", flush=True)
                scores = evaluate_candidates(
                    cands, sim, seeds, days=DAYS,
                    initial_resistant_fraction=r0, growth_noise=noise,
                    myelo=myelo, max_workers=1,
                    progress=progress,
                )
            reference = scores[-1]
            out["engine_runs"] += reference.engine_runs * len(cands)
            for intensity, score in zip(GRID["intensity"], scores[:-1]):
                cell = {
                    "intensity": intensity,
                    "resistant_fraction": r0,
                    "growth_noise": noise,
                    "winner": arm_summary(score),
                    "reference": arm_summary(reference),
                    **cell_status(score, reference),
                }
                out["cells"].append(cell)
                print(f"  intensity {intensity:g}: {cell['status']:10s} "
                      f"aubc_ratio {cell['aubc_ratio_winner_over_reference']:8.4f} "
                      f"dom_delta {cell['dominance_delta_days']:8.1f} "
                      f"pfs_delta {cell['pfs_delta_days']:7.1f} "
                      f"nadir {cell['winner_anc_nadir']:.2f}")
            out["wall_seconds"] = round(time.time() - t_start, 1)
            write_artifact(out)
            processed += 1
            print(f"  [group done in {time.time() - t_group:.0f}s; partial "
                  f"artifact written]", flush=True)
        if stopped_early:
            break

    if stopped_early:
        total = (len(GRID["intensity"]) * len(GRID["resistant_fraction"])
                 * len(GRID["growth_noise"]))
        out["wall_seconds"] = round(time.time() - t_start, 1)
        write_artifact(out)
        print(f"\n[stage stop] {processed} group(s) this invocation; the partial "
              f"artifact holds {len(out['cells'])} of {total} cell(s). Re-run "
              f"_release2_sweep.py (optionally with --groups) to continue; the "
              f"artifact stays `partial: true` until every cell is measured.",
              flush=True)
        return

    # ------------------------------------------------------------------
    # Summary: the two boundaries, stated as boundaries rather than as prose.
    # ------------------------------------------------------------------
    cells = out["cells"]
    safeties = {
        w: out["feasibility"][w]["max_feasible_intensity"]
        for w in out["feasibility"]
    }
    wins = [c for c in cells if c["status"] == "WINS"]
    reversals = [c for c in cells if c["status"] == "REVERSED"]
    infeasible = [c for c in cells if c["status"] == "INFEASIBLE"]
    saturating = [c for c in cells if c["status"] == "SATURATES"]

    # Lowest intensity at which the winner still wins, per (r0, noise) group.
    low_boundary = {}
    for r0 in GRID["resistant_fraction"]:
        for noise in GRID["growth_noise"]:
            group = [c for c in cells
                     if c["resistant_fraction"] == r0
                     and c["growth_noise"] == noise]
            group.sort(key=lambda c: c["intensity"])
            holds = [c["intensity"] for c in group if c["status"] in
                     ("WINS", "SATURATES")]
            low_boundary[f"r0={r0}|noise={noise}"] = (
                min(holds) if holds else None
            )

    out["summary"] = {
        "cells": len(cells),
        "wins": len(wins),
        "saturates": len(saturating),
        "reversed": len(reversals),
        "infeasible": len(infeasible),
        "max_feasible_intensity_by_marrow_weight": safeties,
        "lowest_winning_intensity_per_group": low_boundary,
        "verdict": (
            "the alternating calendar's advantage and its safety bracket are "
            "both reported as boundaries; a cell that reverses or breaks the "
            "ANC floor is a limit of the claim, not a footnote to it"
        ),
    }
    out["reversals"] = [
        {
            "intensity": c["intensity"],
            "resistant_fraction": c["resistant_fraction"],
            "growth_noise": c["growth_noise"],
            "status": c["status"],
            "aubc_ratio_winner_over_reference":
                c["aubc_ratio_winner_over_reference"],
            "dominance_delta_days": c["dominance_delta_days"],
            "winner_anc_nadir": c["winner_anc_nadir"],
        }
        for c in cells if c["status"] in ("REVERSED", "INFEASIBLE")
    ]
    out["engine_identity"] = engine_identity()
    out["provenance"] = provenance()
    out["wall_seconds"] = round(time.time() - t_start, 1)
    out["partial"] = False
    write_artifact(out)

    print("\n" + "=" * 78)
    print("RELEASE 2 SUMMARY")
    print("=" * 78)
    print(f"cells {len(cells)}: {len(wins)} WINS, {len(saturating)} SATURATES, "
          f"{len(reversals)} REVERSED, {len(infeasible)} INFEASIBLE")
    for w, v in safeties.items():
        print(f"  max feasible intensity at drug-B marrow weight {w}: {v}")
    print(f"  lowest winning intensity per group: {low_boundary}")
    print(f"\nartifact: outputs_real/release2_sweep.json "
          f"({out['wall_seconds']:.0f}s, {out['engine_runs']} engine runs)")


if __name__ == "__main__":
    main()
