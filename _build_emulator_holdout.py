"""Build configs 48..96 of the emulator dataset (holdout expansion, ~12 min).

The first 48 configs trained the feature selection (locked subset below).
This batch is the untouched holdout the locked pipeline is evaluated on.
Usage: python _build_emulator_holdout.py 48 72   then  72 96   then --finalise
"""

import json
import sys
import time

import numpy as np

from danon.high_ceiling import _phantom_from_config, _sample_configs
from danon.evosteer import CollateralSwitchPolicy, MTDPolicy, TumorEvolutionSimulator

N_TOTAL = 96
N_REPLICATES = 3
DAYS = 365
SEED = 11
PARTIAL = "data/synthetic_emulator_holdout_partial.json"
FINAL = "data/synthetic_emulator_holdout.json"


def compute_chunk(start: int, end: int) -> dict:
    cfgs_all = _sample_configs(N_TOTAL, SEED)  # same stream as the first 48
    out = {"rows": [], "policy_pfs": [], "mtd_pfs": [], "policy_tox": [], "mtd_tox": [],
           "burden_180": [], "final_resisted_frac": [], "final_burden": []}
    for i in range(start, end):
        cfg = cfgs_all[i]
        geom = _phantom_from_config(cfg, seed=SEED + 1000 + i)
        sim = TumorEvolutionSimulator(geom)
        p_pfs, m_pfs, p_tox, m_tox, b180, rfrac, fburden = [], [], [], [], [], [], []
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
        print(f"cfg {i}: burden180 {out['burden_180'][-1]:.0f}", flush=True)
    return out


def merge(partial: dict, chunk: dict) -> dict:
    for key in chunk:
        partial[key] = partial.get(key, []) + chunk[key]
    return partial


if __name__ == "__main__":
    if sys.argv[1] == "--finalise":
        with open(PARTIAL) as fh:
            partial = json.load(fh)
        assert len(partial["rows"]) == 48, f"expected 48 holdout rows, got {len(partial['rows'])}"
        with open(FINAL, "w") as fh:
            json.dump(partial, fh)
        print(f"finalised {len(partial['rows'])} holdout configs -> {FINAL}")
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
          f"{len(partial['rows'])} holdout rows total")
