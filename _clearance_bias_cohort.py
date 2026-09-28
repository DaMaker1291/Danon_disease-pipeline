"""Does the clearance sub-step bias (bounded by test_benchmarks) move anything?

The verification suite measures the engine's forward-Euler clearance as
integrating at rate lambda*(1 + 0.000842) instead of lambda -- a +0.0842%/day
bias whose direction is fixed (forward Euler cannot undershoot exp(-lambda)).

A bias is only interesting if it changes an outcome. So: apply the EQUIVALENT
parameter perturbation (lambda0 -> lambda0 * (1 + bias)) to a paired cohort on
the same geometry with the same seeds, and compare the endpoint change against
the cohort's own seed-to-seed spread. If the bias is smaller than the spread by
orders of magnitude, it is below the model's resolution -- measured, not argued.
"""

import json
import math
import os

import numpy as np

from danon.evosteer import (
    DEFAULT_DRUGS,
    CollateralSwitchPolicy,
    MTDPolicy,
    TumorEvolutionSimulator,
)
from danon.imaging import from_nifti_mask

ROOT = os.path.dirname(os.path.abspath(__file__))
CASE = "UPENN_GBM_00576"
MASK = f"data/tcia/engine_ready/{CASE}_roi.nii.gz"
VESSELS = f"data/tcia/engine_ready/{CASE}_rcbv.nii.gz"
DAYS = 365
SEEDS = (101, 102, 103)

# The same rule the benchmark test pins: 6*D*dts <= 0.4 from the constants.
D = DEFAULT_DRUGS.diffusion[0]
LAM = DEFAULT_DRUGS.clearance[0]
sub_dt = min(0.4 / (6.0 * D), 1.0)
n_sub = max(1, int(math.ceil(1.0 / sub_dt)))
dts = 1.0 / n_sub
BIAS = math.exp(-LAM) / (1.0 - LAM * dts) ** n_sub - 1.0

print(f"constants: D={D} lambda0={LAM}")
print(f"sub-steps/day={n_sub} dts={dts:.6f}")
print(f"realized bias on the clearance RATE = {BIAS * 100:+.6f}%/day")


def summarize(lam_scale):
    drugs = DEFAULT_DRUGS.__class__(
        **{**DEFAULT_DRUGS.__dict__, "clearance": (LAM * lam_scale, LAM * lam_scale)}
    )
    geom = from_nifti_mask(MASK, vessels_path=VESSELS).geometry
    out = {}
    for name, make in (("mtd", MTDPolicy), ("steering", CollateralSwitchPolicy)):
        final, pfs = [], []
        for s in SEEDS:
            sim = TumorEvolutionSimulator(geom, drugs=drugs)
            traj = sim.run(make(), days=DAYS, seed=s)
            final.append(traj.burden_S[-1] + traj.burden_R[-1])
            pfs.append(traj.pfs_value(DAYS))
        out[name] = {"final": final, "pfs": pfs}
    return out


base = summarize(1.0)
biased = summarize(1.0 + BIAS)

report = {"case": CASE, "bias_rate_per_day": BIAS, "seeds": list(SEEDS), "arms": {}}
print()
print(f"{'arm':9s} {'metric':7s} {'base(seed 101/102/103)':32s} {'spread%':>9s} {'bias delta%':>12s}")
for arm in ("mtd", "steering"):
    for metric in ("final", "pfs"):
        b = np.array(base[arm][metric], dtype=float)
        x = np.array(biased[arm][metric], dtype=float)
        spread = (b.max() - b.min()) / b.mean() * 100 if b.mean() else 0.0
        delta = np.abs(x - b) / np.maximum(np.abs(b), 1e-300) * 100
        report["arms"].setdefault(arm, {})[metric] = {
            "base": b.tolist(),
            "perturbed": x.tolist(),
            "max_rel_delta": float(delta.max()),
            "seed_spread_rel": float(spread),
        }
        print(
            f"{arm:9s} {metric:7s} {str(np.round(b, 4).tolist()):32s} "
            f"{spread:9.4f} {delta.max():12.8f}"
        )

worst_delta = max(
    a[m]["max_rel_delta"] for a in report["arms"].values() for m in a
)
worst_spread = max(
    a[m]["seed_spread_rel"] for a in report["arms"].values() for m in a
)
report["worst_max_rel_delta_percent"] = worst_delta
report["worst_seed_spread_percent"] = worst_spread
report["delta_over_spread"] = (
    worst_delta / worst_spread if worst_spread else None
)
print()
print(f"worst endpoint change from the bias : {worst_delta:.8f}%")
print(f"worst seed-to-seed spread (same arm): {worst_spread:.6f}%")
print(f"ratio delta/spread = {report['delta_over_spread']:.3e}")

os.makedirs("outputs_real", exist_ok=True)
with open("outputs_real/clearance_bias_cohort.json", "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=1)
print("\nwrote outputs_real/clearance_bias_cohort.json")
