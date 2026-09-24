"""Re-score the UPENN-GBM evidence entry over the whole cohort.

The committed record said ``patients_checked_for_timepoints = 2`` and
``patients_with_multi_timepoint = 0``, i.e. two named patients were queried and
the collection was declared single-timepoint. Scoring the cached collection-wide
series listing instead shows 41 of 630 patients carry a follow-up interval of at
least 28 days, and all 41 also carry DSC perfusion series.

This script rewrites one entry's detail from the listing. It preserves the
original sample under ``sampled_*`` names so the correction is auditable, and it
keeps every other entry untouched. Run from the repo root:

    python _rebuild_multimodal_evidence.py
"""

from __future__ import annotations

import json

from danon.multimodal import (
    EVIDENCE_PATH,
    MIN_FOLLOWUP_INTERVAL_DAYS,
    ModalityEvidence,
    load_evidence,
    measure_collection_timepoints,
    save_evidence,
)

SERIES_LISTING = "data/tcia_probe/upenn_series.json"
SCOPED_KEY = "tcia_upenn_gbm"
SCOPE_NOTE = (
    "timepoints scored over every patient in the collection-wide series "
    "listing, not a sample"
)


def main() -> None:
    measured = measure_collection_timepoints(
        SERIES_LISTING, min_interval_days=MIN_FOLLOWUP_INTERVAL_DAYS
    )
    items = load_evidence()
    out = []
    replaced = 0
    for m in items:
        if m.key != SCOPED_KEY:
            out.append(m)
            continue
        detail = dict(m.detail)
        # Keep the sample that produced the old (wrong) answer, under names that
        # say what they are.
        if "patients_checked_for_timepoints" in detail:
            detail["sampled_patients_checked"] = detail["patients_checked_for_timepoints"]
        if "patients_with_multi_timepoint" in detail:
            detail["sampled_patients_with_multi_timepoint"] = detail[
                "patients_with_multi_timepoint"
            ]
        # The cohort-scored numbers are what the verdicts read.
        detail["n_patients"] = measured["n_patients"]
        detail["n_series"] = measured["n_series"]
        detail["patients_checked_for_timepoints"] = measured["n_patients"]
        detail["patients_with_multi_timepoint"] = measured["n_patients_longitudinal"]
        detail["patients_multi_date_any_interval"] = measured["n_patients_multi_date"]
        detail["min_followup_interval_days"] = measured["min_interval_days"]
        detail["timepoint_scope"] = SCOPE_NOTE
        detail["timepoint_source_file"] = measured["source_file"]
        detail["n_patients_dsc_perfusion"] = measured["n_patients_dsc_perfusion"]
        detail["n_dsc_series"] = measured["n_dsc_series"]
        detail["n_patients_processed_perfusion"] = measured[
            "n_patients_processed_perfusion"
        ]
        detail["n_processed_series"] = measured["n_processed_series"]
        detail["n_longitudinal_with_perfusion"] = measured[
            "n_longitudinal_with_perfusion"
        ]
        # Recompute the refusal grounds from the numbers rather than carrying
        # the prose forward: the second-timepoint ground no longer holds.
        lacks = []
        if int(detail.get("n_seg_series") or 0) == 0:
            lacks.append("an expert tumour mask to score a predicted field against")
        if int(detail.get("patients_with_multi_timepoint") or 0) == 0:
            lacks.append("a second timepoint, so there is no follow-up scan to compare to")
        supports = [
            "anonymised glioma MRI with a documented collection page",
            f"a follow-up timepoint for {detail['patients_with_multi_timepoint']} of "
            f"{detail['n_patients']} patients (>={detail['min_followup_interval_days']} "
            "day interval), each with DSC perfusion series",
            f"DSC perfusion acquisition for {detail['n_patients_dsc_perfusion']} patients "
            f"and CaPTk-processed maps for {detail['n_patients_processed_perfusion']} "
            "(rCBV/Ktrans proxies for stream 3)",
        ]
        out.append(
            ModalityEvidence(
                key=m.key,
                name=m.name,
                status=m.status,
                source=m.source,
                license=m.license,
                detail=detail,
                supports=tuple(supports),
                cannot_support=tuple(lacks),
            )
        )
        replaced += 1

    if replaced != 1:
        raise SystemExit(f"expected exactly one {SCOPED_KEY} entry, found {replaced}")
    save_evidence(out, EVIDENCE_PATH)
    print(f"rewrote {EVIDENCE_PATH}: {SCOPED_KEY}")
    for k in (
        "n_patients",
        "patients_with_multi_timepoint",
        "patients_multi_date_any_interval",
        "min_followup_interval_days",
        "n_patients_dsc_perfusion",
        "n_patients_processed_perfusion",
        "n_longitudinal_with_perfusion",
    ):
        print(f"  {k} = {detail[k]}")
    print(f"  cannot_support = {list(lacks)}")


if __name__ == "__main__":
    main()
