"""Chunked builder for the emulator dataset (resumable; ~6 min per chunk).

Usage: python _build_emulator_chunk.py START END
Computes engine outcomes for configs [START, END) and merges them into
data/synthetic_emulator_partial.json. Run twice (0 24, then 24 48), then
finalise with --finalise.
"""

import json
import sys
import time

import numpy as np

from danon.high_ceiling import (
    _phantom_from_config,
    _sample_configs,
)
from danon.evosteer import CollateralSwitchPolicy, MTDPolicy, TumorEvolutionSimulator

N_CONFIGS = 48
N_REPLICATES = 3
DAYS = 365
SEED = 11
PARTIAL = "data/synthetic_emulator_partial.json"
FINAL = "data/synthetic_emulator_dataset.json"


def compute_chunk(start: int, end: int) -> dict:
    cfgs = _sample_configs(N_CONFIGS, SEED)
    out = {"rows": [], "policy_pfs": [], "mtd_pfs": [], "policy_tox": [], "mtd_tox": [],
           "burden_180": [], "final_resisted_frac": [], "final_burden": []}
    for i in range(start, end):
        cfg = cfgs[i]
        geom = _phantom_from_config(cfg, seed=SEED + 1000 + i)
        sim = TumorEvolutionSimulator(geom)
        p_pfs, m_pfs, p_tox, m_tox = [], [], [], []
        b180, rfrac, fburden = [], [], []
        for rep in range(N_REPLICATES):
            s = SEED + 10_000 + i * 100 + rep
            tr_p = sim.run(
                CollateralSwitchPolicy(), days=DAYS,
                initial_resistant_fraction=cfg["initial_resistant_fraction"],
                fill_fraction=cfg["fill_fraction"], seed=s,
            )
            tr_m = sim.run(
                MTDPolicy(), days=DAYS,
                initial_resistant_fraction=cfg["initial_resistant_fraction"],
                fill_fraction=cfg["fill_fraction"], seed=s,
            )
            p_pfs.append(tr_p.pfs_value(DAYS))
            m_pfs.append(tr_m.pfs_value(DAYS))
            p_tox.append(float(tr_p.toxicity))
            m_tox.append(float(tr_m.toxicity))
            # Continuous in-horizon targets from the POLICY arm (no censoring):
            b180.append(float(tr_p.burden_S[179] + tr_p.burden_R[179]))
            rfrac.append(float(tr_p.final_resistant_fraction))
            fburden.append(float(tr_p.burden_S[-1] + tr_p.burden_R[-1]))
        out["rows"].append(cfg)
        out["policy_pfs"].append(float(np.median(p_pfs)))
        out["mtd_pfs"].append(float(np.median(m_pfs)))
        out["policy_tox"].append(float(np.median(p_tox)))
        out["mtd_tox"].append(float(np.median(m_tox)))
        out["burden_180"].append(float(np.median(b180)))
        out["final_resisted_frac"].append(float(np.median(rfrac)))
        out["final_burden"].append(float(np.median(fburden)))
        print(f"cfg {i}: policy PFS {out['policy_pfs'][-1]:.0f} d, "
              f"MTD PFS {out['mtd_pfs'][-1]:.0f} d, burden180 {out['burden_180'][-1]:.0f}", flush=True)
    return out


def merge(partial: dict, chunk: dict) -> dict:
    for key in ("rows", "policy_pfs", "mtd_pfs", "policy_tox", "mtd_tox",
                "burden_180", "final_resisted_frac", "final_burden"):
        partial[key] = partial.get(key, []) + chunk[key]
    return partial


def finalise(partial: dict) -> None:
    assert len(partial["rows"]) == N_CONFIGS, f"expected {N_CONFIGS} rows, got {len(partial['rows'])}"
    pfs_m = np.array(partial["mtd_pfs"])
    pol_pfs = np.array(partial["policy_pfs"])
    ext = (pol_pfs - pfs_m) / np.where(pfs_m > 0, pfs_m, 1.0)
    dataset = {
        "key": {"n_configs": N_CONFIGS, "n_replicates": N_REPLICATES, "days": DAYS, "seed": SEED},
        "dataset": {
            "rows": partial["rows"],
            "policy_pfs": partial["policy_pfs"],
            "mtd_pfs": partial["mtd_pfs"],
            "pfs_extension": ext.tolist(),
            "policy_toxicity": partial["policy_tox"],
            "mtd_toxicity": partial["mtd_tox"],
            "burden_180": partial["burden_180"],
            "final_resisted_frac": partial["final_resisted_frac"],
            "final_burden": partial["final_burden"],
            "n_replicates": N_REPLICATES,
            "days": DAYS,
            "seed": SEED,
        },
    }
    with open(FINAL, "w") as fh:
        json.dump(dataset, fh)
    print(f"finalised {len(partial['rows'])} configs -> {FINAL}")


if __name__ == "__main__":
    if sys.argv[1] == "--finalise":
        with open(PARTIAL) as fh:
            finalise(json.load(fh))
        sys.exit(0)
    start, end = int(sys.argv[1]), int(sys.argv[2])
    t0 = time.time()
    chunk = compute_chunk(start, end)
    try:
        with open(PARTIAL) as fh:
            partial = json.load(fh)
    except FileNotFoundError:
        partial = {}
    partial = merge(partial, chunk)
    with open(PARTIAL, "w") as fh:
        json.dump(partial, fh)
    print(f"chunk [{start},{end}) done in {(time.time()-t0)/60:.1f} min; "
          f"{len(partial['rows'])} rows total")
