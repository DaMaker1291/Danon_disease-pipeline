"""Verify the cached GLASS release against the LIVE cBioPortal source, then re-score.

Three questions, each answered by measurement rather than assertion:

1. Does the cached release still match what the online source returns right now?
   Same panel, same sample list: record counts per gene, and the per-record
   identity (sample, gene, protein change, tumour alt/ref counts). A cache that
   has silently drifted from the source cannot support any number built on it.
2. Do the two published tasks still land where they landed when the record was
   written, if they are re-scored on the LIVE payload through the shipping
   loader (same rules, same seeds)?
3. Is the detection result a fluke of one seed? The cross-validated estimate is
   re-run across several seeds; a signal that only exists for seed 13 is not a
   signal.

Run:  python _online_glass_probe.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

from _glass_variant_cohort import API, CACHE, MUTATIONS, SEQUENCED_LIST, STUDY, _get
from danon.evolution_forecast import (
    change_examples,
    cross_validated_change,
    cross_validated_detection,
    detection_examples,
    paired_spectra,
    records_from_cache,
    timepoint_census,
)
from danon.variants import VARIANT_PANEL

SEEDS = (13, 17, 23, 31, 41, 53, 67, 89)


def fetch_live() -> dict:
    """The same payload _glass_variant_cohort.py would cache, right now."""
    status, samples = _get(f"{API}/studies/{STUDY}/samples?pageSize=10000")
    if status != 200 or not samples:
        raise SystemExit(f"live samples unreadable: status={status}")
    records = []
    per_gene = {}
    for gene, entrez in VARIANT_PANEL:
        url = (
            f"{MUTATIONS}?sampleListId={SEQUENCED_LIST}"
            f"&entrezGeneId={entrez}&pageSize=100000"
        )
        code, payload = _get(url)
        rows = payload if isinstance(payload, list) else []
        per_gene[gene] = (code, len(rows))
        for rec in rows:
            rec = dict(rec)
            rec["_gene"] = gene
            records.append(rec)
        print(f"  live {gene:8s} status={code} records={len(rows)}")
    return {
        "source": f"{API}/molecular-profiles/{STUDY}_mutations/mutations",
        "study_id": STUDY,
        "sample_list_id": SEQUENCED_LIST,
        "panel": [list(p) for p in VARIANT_PANEL],
        "samples": samples,
        "records": records,
        "_per_gene": per_gene,
    }


def identity(rec: dict) -> tuple:
    """What makes two records the same observation, counts included."""
    return (
        rec.get("sampleId", ""),
        rec.get("_gene", ""),
        rec.get("proteinChange") or "",
        rec.get("tumorAltCount"),
        rec.get("tumorRefCount"),
        rec.get("chr") or "",
        rec.get("startPosition") or 0,
        str(rec.get("referenceAllele") or ""),
        str(rec.get("variantAllele") or ""),
    )


def compare(cached: dict, live: dict) -> dict:
    cache_ids = {s["sampleId"] for s in cached.get("samples", [])}
    live_ids = {s["sampleId"] for s in live["samples"]}
    a = Counter(identity(r) for r in cached.get("records", []))
    b = Counter(identity(r) for r in live["records"])
    only_cache = a - b
    only_live = b - a
    return {
        "samples_cached": len(cache_ids),
        "samples_live": len(live_ids),
        "samples_only_in_cache": sorted(cache_ids - live_ids),
        "samples_only_live": sorted(live_ids - cache_ids),
        "records_cached": sum(a.values()),
        "records_live": sum(b.values()),
        "records_identical": sum((a & b).values()),
        "records_only_in_cache": sum(only_cache.values()),
        "records_only_live": sum(only_live.values()),
        "example_only_in_cache": [list(k) for k in list(only_cache)[:3]],
        "example_only_live": [list(k) for k in list(only_live)[:3]],
        "per_gene_live": {g: s for g, (c, s) in live["_per_gene"].items()},
    }


def score(payload: dict) -> None:
    """Both tasks on a payload, through the shipping loader and rules."""
    records, dropped = records_from_cache(payload)
    spectra, pairs = paired_spectra(records)
    det = detection_examples(pairs)
    chg = change_examples(pairs)
    census = timepoint_census(payload)
    print(
        f"  rows={len(records)} dropped={dropped} patients="
        f"{len({r.patient_id for r in records})} spectra={len(spectra)} "
        f"pairs={len(pairs)} scored_rows={len(det)}/{len(chg)} "
        f"declared_types={census['declared_types_distinct']}"
    )
    cv_a = cross_validated_detection(det)
    cv_b = cross_validated_change(chg)
    m = cv_a.metrics
    ci = (m["oof_auc_ci_low"], m["oof_auc_ci_high"])
    print(
        f"  TASK A pooled OOF AUC {m['oof_auc']:.3f} "
        f"CI [{ci[0]:.3f}, {ci[1]:.3f}] "
        f"folds_above_chance={m['folds_above_chance']}/{m['n_folds_scored']} "
        f"resolves={m['beats_prevalence_resolvably']} "
        f"calibration_brier_skill="
        f"{(cv_a.calibration or {}).get('summary', {}).get('brier_skill_score')}"
    )
    op = (cv_a.clinical_use or {}).get("operating_point") or {}
    print(
        f"  clinical call @declared 0.5: sens={op.get('sensitivity')} "
        f"spec={op.get('specificity')} ppv={op.get('ppv')} npv={op.get('npv')} "
        f"tp/fp/tn/fn={op.get('tp')}/{op.get('fp')}/{op.get('tn')}/{op.get('fn')}"
    )
    dc = (cv_a.clinical_use or {}).get("decision_curve") or {}
    print(f"  decision curve: {dc.get('reading', '')[:240]}")
    bm = cv_b.metrics
    print(
        f"  TASK B pooled advantage {bm['oof_mae_advantage']:.5f} "
        f"CI [{bm['oof_mae_advantage_ci_low']:.5f}, "
        f"{bm['oof_mae_advantage_ci_high']:.5f}] "
        f"beats_persistence={bm['beats_persistence_resolvably']} "
        f"folds={bm['folds_beating_persistence']}/{bm['n_folds_scored']}"
    )


def seed_sweep(payload: dict) -> None:
    records, _ = records_from_cache(payload)
    spectra, pairs = paired_spectra(records)
    det = detection_examples(pairs)
    print("\n--- seed sweep of task A (5x5 patient-level CV per seed) ---")
    rows = []
    for seed in SEEDS:
        cv = cross_validated_detection(det, seed=seed)
        m = cv.metrics
        op = (cv.clinical_use or {}).get("operating_point") or {}
        resolved = bool(m["beats_prevalence_resolvably"])
        rows.append((seed, m["oof_auc"], m["oof_auc_ci_low"], resolved))
        print(
            f"  seed {seed:3d}: AUC {m['oof_auc']:.3f} "
            f"CI [{m['oof_auc_ci_low']:.3f}, {m['oof_auc_ci_high']:.3f}] "
            f"folds_above={m['folds_above_chance']}/{m['n_folds_scored']} "
            f"resolves={resolved} sens@0.5={op.get('sensitivity'):.3f} "
            f"spec@0.5={op.get('specificity'):.3f}"
        )
    aucs = [r[1] for r in rows]
    n_res = sum(1 for r in rows if r[3])
    print(
        f"  across {len(rows)} seeds: AUC min {min(aucs):.3f} / "
        f"median {sorted(aucs)[len(aucs) // 2]:.3f} / max {max(aucs):.3f}; "
        f"{n_res} of {len(rows)} seeds resolve above chance with an interval "
        f"clear of 0.5"
    )


def main() -> int:
    print(f"=== 1. live fetch from {API} (study {STUDY}) ===")
    live = fetch_live()
    with open(CACHE, encoding="utf-8") as fh:
        cached = json.load(fh)
    print("\n=== 2. live vs cached ===")
    diff = compare(cached, live)
    for k, v in diff.items():
        if not k.startswith("per_gene"):
            print(f"  {k}: {v}")
    print(f"  cache_fetched_utc: {cached.get('fetched_utc')}")

    print("\n=== 3. re-score the LIVE payload with the shipping rules ===")
    score(live)

    print("\n=== 4. re-score the CACHED payload (should reproduce the record) ===")
    score(cached)

    print("\n=== 5. seed sweep ===")
    seed_sweep(live)

    verdict = (
        "cache matches the live source" if diff["records_only_in_cache"] == 0
        and diff["records_only_live"] == 0
        else "cache differs from the live source (see counts above)"
    )
    print(f"\nVERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
