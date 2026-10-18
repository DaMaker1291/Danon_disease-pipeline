"""Run the real dosing experiments and write `outputs_real/dosing.json`.

Stages, runnable separately so each fits a bounded slot:

    python _run_dosing.py cr           # the competitive-release hypothesis test
    python _run_dosing.py search        # the schedule grid search
    python _run_dosing.py fidelity      # short-horizon screen calibration
    python _run_dosing.py workers       # measure candidate-level parallelism
    python _run_dosing.py combination   # alternating two-drug vs single drug

Both stages read the existing artifact, replace only their own section, and
write atomically (temp file + os.replace), so an interrupted run never leaves
a half-written artifact behind. Every number written is engine output.
"""
import json
import os
import sys
import time

from danon.dosing import (
    MEASURED_WORKER_OPTIMUM,
    MyelosuppressionModel,
    Schedule,
    engine_identity,
    measure_worker_scaling,
    provenance,
    run_competitive_release_experiment,
    search_schedules,
)
from danon.evosteer import TumorEvolutionSimulator, make_gbm_phantom

OUT = "outputs_real/dosing.json"


def _base() -> dict:
    geom = make_gbm_phantom(size=24, seed=12345)
    return {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "geometry": {
            "grid": list(geom.tumor_mask.shape),
            "tumour_voxels": int(geom.tumor_mask.sum()),
            "voxel_mm": 1.0,
            "phantom_seed": 12345,
            "is_patient_imaging": False,
        },
        "myelosuppression_model": MyelosuppressionModel().to_dict(),
        "engine_identity": engine_identity(),
        "provenance": provenance(),
    }


def _write(payload: dict) -> None:
    os.makedirs("outputs_real", exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, OUT)
    print(f"wrote {OUT}", flush=True)


def _load() -> dict:
    if os.path.exists(OUT):
        try:
            with open(OUT, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            print("existing artifact unreadable; starting a new one", flush=True)
    return {}


def run_cr() -> None:
    payload = _load() or _base()
    payload.update(_base())
    print("[cr] competitive-release experiment, 30 held-out seeds ...", flush=True)
    t = time.time()
    res = run_competitive_release_experiment(seeds=range(30), days=365)
    payload["competitive_release"] = res.to_dict()
    payload.setdefault("wall_seconds", {})
    payload["wall_seconds"]["competitive_release"] = round(time.time() - t, 1)
    _write(payload)
    print(
        f"      MTD AUBC {res.mtd.median_aubc:.4f} vs pulsed "
        f"{res.pulsed.median_aubc:.4f}; dominance delta "
        f"{res.dominance_delta_days:+.0f} d; supported={res.hypothesis_supported}",
        flush=True,
    )
    for n in res.notes:
        print(f"      note: {n}", flush=True)


def run_combination() -> None:
    """Alternating two-drug calendars vs the best single-drug calendar.

    Both arms are grids on the same seeds on the same engine, so the calendar is
    the only thing that differs. Drug B's marrow potency is an ASSUMPTION, so
    the verdict is taken at the conservative weight and swept. Reported
    whichever way it comes out.
    """
    from danon.dosing import compare_combination_vs_single

    payload = _load() or _base()
    payload.update(_base())
    geom = make_gbm_phantom(size=24, seed=12345)
    sim = TumorEvolutionSimulator(geom)
    myelo = MyelosuppressionModel()

    print(f"[combination] label-anchored safety screen: the label's own regimen "
          f"{myelo.label_regimen(365).describe()} -> nadir "
          f"{myelo.nadir(myelo.label_regimen(365), 365):.2f} on day "
          f"{myelo.nadir_day(myelo.label_regimen(365), 365)}, "
          f"one extra day on drug -> "
          f"{myelo.nadir(Schedule.cyclic('A', 150.0, 6, 28, 365), 365):.2f} "
          f"(infeasible)", flush=True)
    print(f"[combination] highest continuous marrow burden the screen permits: "
          f"{myelo.max_continuous_mg_m2():.1f} mg/m2/day", flush=True)
    print("[combination] alternating A/B vs best single-drug, 30 seeds ...",
          flush=True)
    print(f"      {MEASURED_WORKER_OPTIMUM} worker processes "
          f"(measured optimum; see the worker_scaling section)", flush=True)

    def progress(done: int, total: int, name: str) -> None:
        print(f"      [{done:2d}/{total}] {name[:70]}", flush=True)

    t = time.time()
    res = compare_combination_vs_single(
        sim, seeds=range(30), days=365, dose_mg_m2=150.0, myelo=myelo,
        progress=progress,
    )
    payload["combination"] = res.to_dict()
    payload.setdefault("wall_seconds", {})
    payload["wall_seconds"]["combination"] = round(time.time() - t, 1)
    _write(payload)

    print(f"      single-drug arm: {len(res.single_drug_scored)} calendars; best "
          f"{res.single_drug.name} AUBC {res.single_drug.median_aubc:.4f} "
          f"(dominance day {res.single_drug.median_dominance_day:.0f})", flush=True)
    if res.best_combination:
        print(f"      best alternating at weight {res.primary_marrow_weight:g}: "
              f"{res.best_combination.name} AUBC "
              f"{res.best_combination.median_aubc:.4f} "
              f"(dominance day {res.best_combination.median_dominance_day:.0f})",
              flush=True)
    print(f"      {len(res.combination)} of {res.n_candidates} alternating "
          f"calendars admissible at the primary weight; "
          f"{res.n_screened_out} inadmissible at every weight; "
          f"beats_single={res.beats_single}", flush=True)
    print("      marrow-weight sweep:", flush=True)
    for r in res.marrow_weight_sweep:
        if r["best_name"] is None:
            print(f"        w_B={r['drug_b_marrow_weight']:<5g} "
                  f"{r['n_alternating_calendars_admissible']:2d} admissible, "
                  f"no best -> null", flush=True)
            continue
        print(f"        w_B={r['drug_b_marrow_weight']:<5g} "
              f"{r['n_alternating_calendars_admissible']:2d} admissible, "
              f"AUBC {r['best_median_aubc']:.4f} "
              f"ratio {r['aubc_ratio_over_single']:.4f} "
              f"beats_single={r['beats_single']}", flush=True)
    for s in res.combination[:8]:
        print(f"        {s.name[:78]:78s} AUBC {s.median_aubc:.4f} "
              f"dom {s.median_dominance_day:5.0f}", flush=True)
    for n in res.notes:
        print(f"      note: {n}", flush=True)


def run_workers() -> None:
    """MEASURE candidate-level worker scaling and write it to the artifact.

    Same reason `measure_batch_scaling` exists: a worker count chosen by
    intuition rather than by measurement is a guess with a process pool
    attached, and on a bandwidth-bound grid the guess is usually wrong.
    """
    payload = _load() or _base()
    payload.update(_base())
    geom = make_gbm_phantom(size=24, seed=12345)
    sim = TumorEvolutionSimulator(geom)
    print("[workers] measuring candidate-level parallelism ...", flush=True)
    t = time.time()
    rows = measure_worker_scaling(sim)
    payload["worker_scaling"] = rows
    payload.setdefault("wall_seconds", {})
    payload["wall_seconds"]["worker_scaling"] = round(time.time() - t, 1)
    _write(payload)
    for r in rows:
        if "measured_worker_optimum" in r:
            print(f"      optimum={r['measured_worker_optimum']} workers, "
                  f"speedup {r['speedup_vs_one_worker']}x, linear="
                  f"{r['is_speedup_linear_in_workers']}", flush=True)
        else:
            print(f"      workers={r['workers']:<2d} "
                  f"{r['seconds_per_candidate']:6.2f}s/candidate", flush=True)
    print(f"      numbers identical across worker counts: "
          f"{rows[0].get('numbers_identical_across_worker_counts')}", flush=True)


def run_fidelity() -> None:
    """Measure what a short-horizon screen mis-ranks, on the REAL grid.

    This is a calibration run: every feasible candidate is executed at BOTH
    horizons on the same seeds, so the two rankings can be compared directly.
    It costs more than the search it is calibrating -- that is the point, it is
    run once to decide whether short screening is safe here.
    """
    from danon.dosing import feasible_grid_schedules, grid_size, screen_fidelity
    from danon.dosing import screen_fidelity_note

    payload = _load() or _base()
    payload.update(_base())
    geom = make_gbm_phantom(size=24, seed=12345)
    sim = TumorEvolutionSimulator(geom)
    myelo = MyelosuppressionModel()

    scheds = feasible_grid_schedules(
        on_days=(3, 5, 7, 10), cycle_days=(28, 42), dose_levels=(0.4, 0.6, 0.8),
        dose_mg_m2=150.0, days=365, myelo=myelo,
    )
    print(
        f"[fidelity] {len(scheds)} feasible candidates "
        f"(grid was {grid_size((3, 5, 7, 10), (28, 42), (0.4, 0.6, 0.8))}); "
        f"running each at 365 and at 120 days ...",
        flush=True,
    )
    t = time.time()
    fid = screen_fidelity(
        sim, scheds, seeds=range(6), full_days=365, screen_days=120, top_k=3,
        myelo=myelo,
    )
    payload["screen_fidelity"] = fid.to_dict()
    payload["screen_fidelity"]["note"] = screen_fidelity_note(fid)
    payload.setdefault("wall_seconds", {})
    payload["wall_seconds"]["screen_fidelity"] = round(time.time() - t, 1)
    _write(payload)

    print(f"      Spearman {fid.spearman:.4f}  Kendall {fid.kendall_tau:.4f}", flush=True)
    print(f"      full-horizon winner: {fid.full_winner} "
          f"(screen ranks it {fid.full_winner_screen_rank} of {fid.n_candidates})", flush=True)
    print(f"      screen pick: {fid.screen_winner} "
          f"(full-horizon rank {fid.screen_winner_full_rank})", flush=True)
    print(f"      top-{fid.top_k} caught: {fid.caught}, missed: {list(fid.missed)}", flush=True)
    for n in fid.notes:
        print(f"      note: {n}", flush=True)


def run_search() -> None:
    payload = _load() or _base()
    payload.update(_base())
    geom = make_gbm_phantom(size=24, seed=12345)
    sim = TumorEvolutionSimulator(geom)
    print("[search] grid search ...", flush=True)
    t = time.time()
    res = search_schedules(
        sim,
        seeds=range(30),
        days=365,
        on_days=(3, 5, 7, 10),
        cycle_days=(28, 42),
        dose_levels=(0.4, 0.6, 0.8),
        dose_mg_m2=150.0,
        screen_seeds=2,
        confirm_top=2,
    )
    payload["search"] = res.to_dict()
    payload.setdefault("wall_seconds", {})
    payload["wall_seconds"]["search"] = round(time.time() - t, 1)
    _write(payload)
    print(
        f"      evaluated {len(res.scored)} confirmed calendars over "
        f"{res.engine_runs} engine runs; beats_baselines={res.beats_baselines}",
        flush=True,
    )
    if res.best:
        print(f"      best: {res.best.name} AUBC {res.best.median_aubc:.4f}", flush=True)
    for n in res.notes:
        print(f"      note: {n}", flush=True)


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "cr"
    if stage == "cr":
        run_cr()
    elif stage == "search":
        run_search()
    elif stage == "fidelity":
        run_fidelity()
    elif stage == "combination":
        run_combination()
    elif stage == "workers":
        run_workers()
    else:
        print(__doc__)
        sys.exit(2)
    else:
        print(__doc__)
        sys.exit(2)