import json
from pathlib import Path

from danon.assimilation import assimilation_twin_audit

rec = assimilation_twin_audit()
out = Path("outputs_real/assimilation_twin.json")
out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
print(rec["reading"])
print()
m = rec["metrics"]
for k, v in m.items():
    if not isinstance(v, list):
        print(f"  {k}: {v}")
print(f"\nwrote {out}")