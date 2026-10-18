"""The TCGA-fitted covariate model applied unchanged to CGGA, on real files.

Reads the two cached CGGA clinical releases (downloaded from www.cgga.org.cn),
fits the repository's Cox model on the cached TCGA-GBM export once, and scores
the CGGA rows with those frozen coefficients. Records what the transport gives
-- whatever it gives -- alongside the pitch corrections, so the numbers in
`danon/external_cohorts.py`'s refusals move with the data.

Writes outputs_real/cgga_ood.json
Run:  python _cgga_ood_cohort.py

Requires network ONCE, to seed data/cgga/ (this script does not download; it
refuses if the cache is absent, so the claim check can never fetch).
"""

import datetime
import json
import os

from danon.external_cohorts import (
    CGGA_693_ZIP,
    CGGA_325_ZIP,
    CGGA_CITATION,
    CGGA_OOD_RECORD,
    CGGA_SOURCE_325,
    CGGA_SOURCE_693,
    ENGINE_CAVEAT,
    PITCH_FIGURES,
    cgga_ood_report,
    external_cohorts_provenance,
    file_sha256,
    pitch_corrections,
    what_external_data_cannot_show,
)
from danon.predictive import GDC_CLINICAL_PATH

CGG_A_PATHS = (CGGA_325_ZIP, CGGA_693_ZIP)


def main() -> None:
    for path in CGG_A_PATHS + (GDC_CLINICAL_PATH,):
        if not os.path.exists(path):
            raise SystemExit(
                f"{path} is not cached. CGGA files come from "
                f"www.cgga.org.cn (see danon/external_cohorts.py for the exact "
                f"URLs); this script does not download."
            )

    report = cgga_ood_report(CGG_A_PATHS)
    generated = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    payload = {
        "generated_utc": generated,
        "source": {
            "cgga": {
                "files": {
                    os.path.basename(p): {
                        "url": CGGA_SOURCE_325 if p == CGGA_325_ZIP else CGGA_SOURCE_693,
                        "sha256": file_sha256(p),
                        "bytes": os.path.getsize(p),
                    }
                    for p in CGG_A_PATHS
                },
                "citation": CGGA_CITATION,
                "downloaded_for": "out-of-distribution transport test",
            },
            "tcga": {
                "path": GDC_CLINICAL_PATH,
                "sha256": file_sha256(GDC_CLINICAL_PATH),
                "role": "training cohort for the frozen coefficients",
            },
        },
        "tcga_reference": {
            "n": report.reference.n,
            "n_events": report.reference.n_events,
            "names": list(report.reference.names),
            "beta": list(report.reference.beta),
            "hazard_ratios": report.reference.hazard_ratios,
            "internal_c": report.reference.internal_c,
            "internal_c_sd": report.reference.internal_c_sd,
            "notes": report.reference.notes,
        },
        "cohorts": {
            key: {
                "path": c.path,
                "table_rows": c.n_table_rows,
                "usable": c.n,
                "events": c.n_events,
                "dropped_no_age": c.dropped_no_age,
                "dropped_no_endpoint": c.dropped_no_endpoint,
                "dropped_no_sex": c.dropped_no_sex,
                "sex_values_normalised": c.sex_values_normalised,
                "histology_counts": c.histology_counts(),
                "prs_counts": c.prs_counts(),
                "notes": c.notes,
            }
            for key, c in report.cohorts.items()
        },
        "tests": {
            key: [
                {
                    "label": t.label,
                    "n": t.n,
                    "n_events": t.n_events,
                    "c_index": t.c_index,
                    "age_only_c_index": t.age_only_c_index,
                    "note": t.note,
                }
                for t in tests
            ]
            for key, tests in report.tests.items()
        },
        "provenance": {
            k: v.value for k, v in external_cohorts_provenance().items()
        },
        "pitch_figures": dict(PITCH_FIGURES),
        "corrections": pitch_corrections(),
        "notes": report.notes,
        "engine_caveat": ENGINE_CAVEAT,
        "what_external_data_cannot_show": what_external_data_cannot_show(),
    }

    os.makedirs(os.path.dirname(CGGA_OOD_RECORD), exist_ok=True)
    with open(CGGA_OOD_RECORD, "w") as fh:
        json.dump(payload, fh, indent=1)

    ref = report.reference
    print(
        f"TCGA reference: n={ref.n}, {ref.n_events} events, "
        f"internal C={ref.internal_c:.4f} +/- {ref.internal_c_sd:.4f}, "
        f"betas={dict(zip(ref.names, (round(b, 5) for b in ref.beta)))}",
        flush=True,
    )
    for key, tests in report.tests.items():
        for t in tests:
            print(
                f"  {key:<52s} {t.label:<34s} C={t.c_index:.4f} "
                f"(age alone {t.age_only_c_index:.4f}) n={t.n} events={t.n_events}",
                flush=True,
            )
    print(f"wrote {CGGA_OOD_RECORD}", flush=True)


if __name__ == "__main__":
    main()
