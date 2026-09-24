"""Rebuild outputs_real/mgmt_resistance.txt from live measurement.

The artifact is the sweep's measurement record: it cites engine PFS numbers, so
a change to the engine's dynamics makes it stale. Regenerate rather than assert
an unrecorded number.

Usage: PYTHONPATH=. python _rebuild_mgmt_resistance.py
"""

from __future__ import annotations

from danon.mgmt import (
    load_mgmt_beta_panel,
    load_published_mgmt_calls,
    mgmt_resistance_sweep,
    render_mgmt_report,
    resistance_audit,
)

OUT = "outputs_real/mgmt_resistance.txt"


def main() -> None:
    panel = load_mgmt_beta_panel()
    betas = panel.quantiles()
    print(f"panel: {panel.n_patients} patient(s), betas {[round(b, 4) for b in betas]}")

    sweep = mgmt_resistance_sweep(
        betas,
        beta_source="cbioportal TCGA-GBM beta quantiles "
        f"({panel.n_patients} patients, 2 probes)",
        n_replicates=3,
        days=365,
    )
    print(f"sweep arms: {len(sweep.by_policy.get('mtd', []))} mtd cells", flush=True)

    text = render_mgmt_report(
        panel=panel,
        calls=load_published_mgmt_calls(),
        sweep=sweep,
        audit=resistance_audit(),
    )
    with open(OUT, "w") as fh:
        fh.write(text)
    print("MGMT_DONE", flush=True)


if __name__ == "__main__":
    main()
