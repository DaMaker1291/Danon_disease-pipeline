"""Build outputs_real/tumor_agnostic.json -- the tissue-portability record.

Run: python _tumor_agnostic_cohort.py
"""

import json
import os

from danon.tumor_agnostic import PORTABILITY_RECORD, tumor_agnostic_report


def main() -> None:
    report = tumor_agnostic_report()
    os.makedirs(os.path.dirname(PORTABILITY_RECORD) or ".", exist_ok=True)
    with open(PORTABILITY_RECORD, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {PORTABILITY_RECORD}")
    print(f"swap surface: {report['swap_surface_fields']} fields")
    for r in report["runs"]:
        print(
            f"  {r['days']:>3} d seed {r['seed']}: final {r['final_burden']:.4f} "
            f"max {r['max_daily_burden']:.4f} / cap {r['capacity']:.0f} "
            f"pfs {r['pfs_day']} finite={r['finite']} bounded={r['bounded']}"
        )
    hits = report["inventory"]["data_organ_hits"]
    print("organ/drug path hits:", {k: len(v) for k, v in hits.items()})
    print("tissue_d_keys:", report["inventory"]["tissue_d_keys"])


if __name__ == "__main__":
    main()
