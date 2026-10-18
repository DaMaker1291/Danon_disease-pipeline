"""Repeat the published task-A estimate across 41 FRESH seeds (100-140).

None of these seeds is the published seed (13), so this is a stability check on
the shipped estimator: a result that only holds for the recorded seed would show
up here as a spread that crosses chance. Prints every run, then the summary.
"""
import json
import statistics

from danon.evolution_forecast import (
    records_from_cache,
    paired_spectra,
    detection_examples,
    cross_validated_detection,
)

CACHE = "data/glass_paired/glass_mutations.json"
raw = json.load(open(CACHE, encoding="utf-8"))
records, _ = records_from_cache(raw)
spectra, pairs = paired_spectra(records)
det = detection_examples(pairs)
seeds = list(range(100, 141))
aucs, sens, spec, ci_lo, folds = [], [], [], [], []
resolved = 0
for s in seeds:
    cv = cross_validated_detection(det, seed=s)
    m = cv.metrics
    op = (cv.clinical_use or {}).get("operating_point") or {}
    aucs.append(m["oof_auc"]); ci_lo.append(m["oof_auc_ci_low"])
    sens.append(op.get("sensitivity")); spec.append(op.get("specificity"))
    folds.append(m["folds_above_chance"])
    if m["beats_prevalence_resolvably"]:
        resolved += 1
    print(
        f"seed {s}: AUC {m['oof_auc']:.4f} CI [{m['oof_auc_ci_low']:.4f},"
        f"{m['oof_auc_ci_high']:.4f}] folds {m['folds_above_chance']}/25 "
        f"sens {op.get('sensitivity'):.3f} spec {op.get('specificity'):.3f} "
        f"resolves {m['beats_prevalence_resolvably']}",
        flush=True,
    )
print(f"\nSUMMARY over {len(seeds)} fresh seeds", flush=True)
print(
    f"  AUC mean {statistics.mean(aucs):.4f} sd {statistics.pstdev(aucs):.4f} "
    f"min {min(aucs):.4f} max {max(aucs):.4f}",
    flush=True,
)
print(
    f"  lowest CI low bound {min(ci_lo):.4f} (all above 0.5: "
    f"{all(x > 0.5 for x in ci_lo)})",
    flush=True,
)
print(f"  resolving: {resolved}/{len(seeds)}", flush=True)
print(
    f"  folds above chance: min {min(folds)} max {max(folds)} "
    f"mean {statistics.mean(folds):.1f}",
    flush=True,
)
print(
    f"  sens@0.5 mean {statistics.mean(sens):.4f} "
    f"[{min(sens):.4f}, {max(sens):.4f}]; spec@0.5 mean "
    f"{statistics.mean(spec):.4f} [{min(spec):.4f}, {max(spec):.4f}]",
    flush=True,
)
