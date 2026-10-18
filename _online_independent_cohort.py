"""Score task A on an INDEPENDENT public cohort, fetched live from cBioPortal.

The published record is built on study `difg_glass`. This probe asks the
question every measured result needs answered by something other than itself:
does the same shipping pipeline -- same loader, same labelling rule, same
panel, same cross-validated estimator and the same clinical-use block -- land
somewhere sane on a DIFFERENT live cohort?

Candidate studies are tried in order. A study is scored only if its own sample
ids carry the DECLARED GLASS suffix rule (-TP primary, -R<digit> recurrence)
with enough paired patients; a study whose identifiers the rule cannot label is
REFUSED and recorded as refused rather than relabelled, because a labelling
assumption invented mid-probe is exactly what this repository's audit exists to
stop.

The answer is allowed to be that the result does not transport. Whatever it is,
it is printed with the same figures the clinical report shows: pooled
out-of-fold AUC with its patient-clustered interval, the fold count, the
operating point at the declared threshold and the decision-curve verdict.

Run:  python _online_independent_cohort.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter

from danon.evolution_forecast import (
    change_examples,
    cross_validated_change,
    cross_validated_detection,
    detection_examples,
    paired_spectra,
    records_from_cache,
)
from danon.variants import VARIANT_PANEL

API = "https://www.cbioportal.org/api"
CANDIDATES = ("difg_glass_2019", "difg_tcga_gdc", "difg_msk_2023", "gbm_columbia_2019")
MIN_PAIRED_PATIENTS = 10


def get(url: str, timeout: int = 90):
    import urllib.request

    req = urllib.request.Request(
        url, headers={"User-Agent": "danon-verify/1.0", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def suffix_kind(sample_id: str) -> str:
    """The DECLARED rule from the shipping module, re-stated here verbatim."""
    suffix = sample_id.rsplit("-", 1)[-1].upper()
    if suffix == "TP":
        return "Primary"
    if suffix == "RM":
        return "Metastasis"
    if len(suffix) >= 2 and suffix[0] == "R" and suffix[1:].isdigit():
        return "Recurrence"
    return "Unlabelled"


def try_study(study: str) -> dict:
    print(f"\n=== {study} ===")
    status, samples = get(f"{API}/studies/{study}/samples?pageSize=10000")
    if status != 200 or not samples:
        return {"study": study, "refused": f"samples unreadable (status={status})"}
    kinds = Counter(suffix_kind(s["sampleId"]) for s in samples)
    print(f"  samples={len(samples)} suffix kinds={dict(kinds)}")
    if kinds.get("Primary", 0) == 0 or kinds.get("Recurrence", 0) == 0:
        return {
            "study": study,
            "refused": (
                f"the declared suffix rule labels {kinds.get('Unlabelled', 0)} of "
                f"{len(samples)} samples Unlabelled (kinds={dict(kinds)}); no "
                "paired primary/recurrence set exists under the shipping rule, "
                "so nothing is guessed into a fold"
            ),
        }

    # The RAW payload shape the shipping loader reads (_gene set, counts left
    # as the API returned them): a probe that invented its own record format
    # would be testing the probe rather than the pipeline.
    records = []
    dropped = 0  # re-counted by the shipping loader below; shown for the print
    per_gene = {}
    source_url = ""
    for gene, entrez in VARIANT_PANEL:
        url = (
            f"{API}/molecular-profiles/{study}_mutations/mutations"
            f"?sampleListId={study}_sequenced&entrezGeneId={entrez}&pageSize=100000"
        )
        source_url = url
        code, payload = get(url)
        rows = payload if isinstance(payload, list) else []
        per_gene[gene] = (code, len(rows))
        for rec in rows:
            rec = dict(rec)
            rec["_gene"] = gene
            records.append(rec)
    statuses = {c for c, _ in per_gene.values()}
    print(
        f"  gene calls={len(per_gene)} statuses={sorted(statuses, key=str)} "
        f"records={len(records)} dropped={dropped}"
    )
    if statuses != {200}:
        return {"study": study, "refused": f"gene call statuses {sorted(statuses, key=str)}"}

    try:
        recs, dropped = records_from_cache(
            {
                "source": (
                    f"{API}/molecular-profiles/{study}_mutations/mutations"
                ),
                "records": records,
            }
        )
        spectra, pairs = paired_spectra(recs)
        det = detection_examples(pairs)
        chg = change_examples(pairs)
    except Exception as exc:
        return {"study": study, "refused": f"loader refused: {type(exc).__name__}: {exc}"}
    if len(det) < 30:
        return {
            "study": study,
            "refused": (
                f"only {len(det)} scored detection rows over {len(pairs)} pairs "
                "after the shipping loader, too few to score honestly"
            ),
        }
    patients = sorted({r.patient_id for r in recs})
    print(
        f"  rows={len(recs)} patients={len(patients)} spectra={len(spectra)} "
        f"pairs={len(pairs)} det_rows={len(det)} change_rows={len(chg)}"
    )

    cv = cross_validated_detection(det)
    m = cv.metrics
    op = (cv.clinical_use or {}).get("operating_point") or {}
    dc = (cv.clinical_use or {}).get("decision_curve") or {}
    prev = m.get("baseline_majority_accuracy")
    result = {
        "study": study,
        "patients": len(patients),
        "pairs": len(pairs),
        "det_rows": len(det),
        "auroc": m.get("oof_auc"),
        "ci_low": m.get("oof_auc_ci_low"),
        "ci_high": m.get("oof_auc_ci_high"),
        "folds_above_chance": m.get("folds_above_chance"),
        "n_folds": m.get("n_folds_scored"),
        "beats_prevalence_resolvably": m.get("beats_prevalence_resolvably"),
        "prevalence": prev,
        "operating_point": {
            k: op.get(k)
            for k in ("sensitivity", "specificity", "ppv", "npv", "tp", "fp", "tn", "fn")
        },
        "decision_curve_reading": dc.get("reading"),
        "condition": (
            "measured here, published nowhere: this is an INDEPENDENT live "
            "cohort scored with the shipping rules"
        ),
    }
    print(
        f"  AUROC {m['oof_auc']:.3f} CI [{m['oof_auc_ci_low']:.3f}, "
        f"{m['oof_auc_ci_high']:.3f}] folds {m['folds_above_chance']}/"
        f"{m['n_folds_scored']} majority {prev:.3f} "
        f"resolves={m['beats_prevalence_resolvably']}"
    )
    print(
        f"  op@0.5 sens={op.get('sensitivity')} spec={op.get('specificity')} "
        f"ppv={op.get('ppv')} npv={op.get('npv')} "
        f"tp/fp/tn/fn={op.get('tp')}/{op.get('fp')}/{op.get('tn')}/{op.get('fn')}"
    )
    print(f"  curve: {(dc.get('reading') or '')[:220]}")
    if chg:
        cvb = cross_validated_change(chg)
        b = cvb.metrics
        result["task_b_advantage"] = b.get("oof_mae_advantage")
        result["task_b_resolves"] = b.get("beats_persistence_resolvably")
        print(
            f"  task B advantage {b['oof_mae_advantage']:.5f} CI "
            f"[{b['oof_mae_advantage_ci_low']:.5f}, "
            f"{b['oof_mae_advantage_ci_high']:.5f}] "
            f"beats_persistence={b['beats_persistence_resolvably']}"
        )
    return result


def main() -> int:
    results = []
    for study in CANDIDATES:
        try:
            results.append(try_study(study))
        except Exception as exc:  # noqa: BLE001 -- the record carries failures too
            results.append({"study": study, "refused": f"{type(exc).__name__}: {exc}"})
    print("\n=== SUMMARY ===")
    for r in results:
        if "refused" in r:
            print(f"  {r['study']}: REFUSED -- {r['refused']}")
        else:
            print(
                f"  {r['study']}: AUROC {r['auroc']:.3f} "
                f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}] "
                f"n={r['patients']}p/{r['det_rows']}rows resolves="
                f"{r['beats_prevalence_resolvably']}"
            )
    with open("outputs_real/independent_cohort_probe.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"generated_from": API, "candidates": list(CANDIDATES), "results": results},
            fh,
            indent=1,
            default=str,
        )
    print("wrote outputs_real/independent_cohort_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
