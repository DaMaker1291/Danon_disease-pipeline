import json
from pathlib import Path

from danon.assimilation import assimilation_twin_validation

rec = assimilation_twin_validation()
out = Path("outputs_real/assimilation_twin_holdout.json")
out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(rec["reading"])
print()
for row in rec["runs"]:
    print(
        f"  seed {row['geom_seed']:2d}: blind {row['blind_burden_accuracy_pct']:6.2f}%"
        f" -> assimilated {row['assimilated_burden_accuracy_pct']:6.2f}%"
        f" ({row['burden_gain_points']:+6.2f} pts)"
    )
print()
for k, v in rec["summary"].items():
    print(f"  {k}: {v}")
print(f"\nwrote {out}")