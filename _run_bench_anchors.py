"""Persist the Step-1 bench anchors: measured potency vs authored constants.

WHY THIS EXISTS. `danon/claims.py` already checks, live, that the potency
figures come from real measured records (`potency_is_measured`) and that the
engine's EC50 has no declared concentration unit (`ec50_unit_undeclared`). Both
checks compute their numbers on the spot and throw them away. That is fine for
an audit and useless for a dossier: a reviewer reading a Phase-1b filing needs
to see the measured distribution, the assumed constants it has to be reconciled
against, and the reason the reconciliation is REFUSED -- as a dated artifact
with a provenance chain, not as a live expression in a test.

WHAT IT WRITES. `outputs_real/bench_anchors.json`, containing:

  * the molecule record (ChEMBL, measured/curated descriptors only),
  * the glioma-line temozolomide potency distribution, with right-censored
    records COUNTED and excluded from the point estimate rather than silently
    deleted or silently treated as exact,
  * the GDSC2 fitted-release glioma audit, including the extrapolation count
    that makes most of those IC50s lower bounds,
  * the engine's stated constants (`DEFAULT_DRUGS`, `DEFAULT_CLONES`), tagged
    as authored assumptions,
  * the reconciliation verdict -- REFUSED, with the blockers named -- and the
    exact bench protocol that would close the gap.

WHAT IT REFUSES. It does not fit anything, does not convert an IC50 into a
kill rate, and does not touch `DEFAULT_DRUGS`. The honest end state of Step 1
without a wet-lab partner is a measured anchor and a named gap; writing a
plausible-looking calibration on top of an incommensurable estimand is exactly
the failure this repository is built to avoid.

USAGE
-----
python _run_bench_anchors.py
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from danon.dosing import engine_identity, provenance  # noqa: E402
from danon.eval_targets import (  # noqa: E402
    GDSC2_FITTED_PATH,
    GDSC2_FITTED_SOURCE,
    load_gdsc_glioma,
    what_targets_cannot_show,
)
from danon.pharmacology import (  # noqa: E402
    compare_engine_ec50,
    engine_ec50_unit,
    load_glioma_potency,
    pharmacology_provenance,
    what_pharmacology_cannot_prove,
)

OUT = os.path.join(ROOT, "outputs_real", "bench_anchors.json")


def _engine_constants() -> dict:
    """The engine's stated constants, read from the engine itself.

    Read rather than transcribed: a constant set copied into this file would
    drift away from the one the simulator actually runs.
    """
    import dataclasses

    from danon.evosteer import DEFAULT_CLONES, DEFAULT_DRUGS

    return {
        "source": "danon/evosteer.py::DEFAULT_DRUGS / DEFAULT_CLONES",
        "drugs": dataclasses.asdict(DEFAULT_DRUGS),
        "clones": dataclasses.asdict(DEFAULT_CLONES),
        "engine_ec50_unit": engine_ec50_unit(),
        "evidence_tier": (
            "AUTHORED ASSUMPTION -- not fitted to any measured kill rate. The "
            "engine's EC50 carries no concentration unit, so it cannot be "
            "compared to a molar IC50 at all; that refusal is the finding, not "
            "an omission to be papered over."
        ),
    }


def _potency_summary(summary) -> dict:
    return {
        "standard_type": summary.standard_type,
        "n_records": summary.n_records,
        "n_exact": summary.n_exact,
        "n_right_censored": summary.n_right_censored,
        "n_left_censored": summary.n_left_censored,
        "n_unconvertible_units": summary.n_unconvertible,
        "median_exact_nM": summary.median_exact_nM,
        "most_potent_exact_nM": summary.most_potent_exact_nM,
        "least_potent_exact_nM": summary.least_potent_exact_nM,
        "fold_spread_exact": summary.fold_spread,
        "censored_fraction": summary.censored_fraction,
        "median_lower_bound_nM": summary.median_lower_bound_nM,
        "exact_values_nM": list(summary.exact_nM),
        "targets": list(summary.targets),
        "publication_years": sorted(set(summary.years)),
        "notes": list(summary.notes),
        "describe": summary.describe(),
        "unit_rule": (
            "mass-concentration records convert to molar ONLY through the "
            "molecule's molecular weight; anything unconvertible is counted in "
            "n_unconvertible_units and excluded from the point estimate"
        ),
    }


def _gdsc_audit(audit) -> dict:
    return {
        "path": audit.path,
        "source": GDSC2_FITTED_SOURCE,
        "n_rows_scanned": audit.n_rows,
        "n_gbm_rows": audit.n_gbm_rows,
        "n_gbm_drugs": audit.n_gbm_drugs,
        "n_gbm_cell_lines": audit.n_gbm_lines,
        "n_inside_tested_range": audit.n_inside_range,
        "n_above_max_tested": audit.n_above_max_tested,
        "n_below_min_tested": audit.n_below_min_tested,
        "extrapolated_fraction": audit.extrapolated_fraction,
        "temozolomide_rows": audit.temozolomide_rows,
        "temozolomide_above_max_tested": audit.temozolomide_above_max_tested,
        "verdict": audit.verdict,
        "why_it_matters": (
            "a fitted IC50 above the plate's maximum tested concentration is a "
            "model extrapolation, not a measurement; those values are lower "
            "bounds and cannot anchor a kill constant"
        ),
    }


def main() -> int:
    started = time.time()
    print("=" * 78)
    print("STEP 1: BENCH ANCHORS vs THE ENGINE'S AUTHORED CONSTANTS")
    print("=" * 78, flush=True)

    mol, summary = load_glioma_potency()
    print(f"[chembl] {mol.describe()}", flush=True)
    print(f"[chembl] {summary.describe()}", flush=True)
    for note in summary.notes:
        print(f"         note: {note}", flush=True)

    unit = engine_ec50_unit()
    comparison = compare_engine_ec50(summary, 0.30, unit)
    print(f"[ec50]   {comparison.describe()}", flush=True)

    try:
        audit = load_gdsc_glioma()
        gdsc = _gdsc_audit(audit)
        print(f"[gdsc]   {audit.verdict}", flush=True)
    except FileNotFoundError as exc:
        gdsc = {"unavailable": str(exc)}
        print(f"[gdsc]   unavailable: {exc}", flush=True)

    engine_constants = _engine_constants()
    print(f"[engine] {engine_constants['evidence_tier']}", flush=True)

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "what_this_is": (
            "The measured bench data that the scheduling engine's pharmacology "
            "constants must be reconciled against, the constants themselves, and "
            "the reason the reconciliation is refused. Written by "
            "_run_bench_anchors.py; every measured number is engine-independent."
        ),
        "molecule": {
            "name": mol.name,
            "chembl_id": mol.chembl_id,
            "mw_g_per_mol": mol.mw,
            "alogp": mol.alogp,
            "psa": mol.psa,
            "hbd": mol.hbd,
            "hba": mol.hba,
            "max_phase": mol.max_phase,
            "source": mol.source,
            "payload_sha256": mol.payload_sha256,
        },
        "glioma_line_potency": _potency_summary(summary),
        "gdsc2_glioma_audit": gdsc,
        "engine_constants": engine_constants,
        "ec50_comparison": {
            "comparable": comparison.comparable,
            "reason": comparison.reason,
            "describe": comparison.describe(),
            "engine_ec50_compared": 0.30,
            "unit_declared": unit,
        },
        "reconciliation": {
            "calibrated": False,
            "verdict": (
                "REFUSED. The engine's kill constants remain authored "
                "assumptions. Public potency data exist and are measured, but "
                "they cannot be substituted into the engine as if they were "
                "kill rates."
            ),
            "blockers": [
                "TYPE: an IC50 is a concentration at 50% effect in a 2D "
                "monolayer at a fixed exposure time; the engine's kill_max is a "
                "per-day fractional kill rate in a 3D substrate with a "
                "diffusion gradient. Converting between them requires a "
                "rate-reporting (time-kill or clonogenic) assay, which is not "
                "in this tree.",
                "UNIT: the engine's EC50 has no declared concentration unit, "
                "so any ratio between it and a molar IC50 is a type error "
                "rather than a fit (danon/pharmacology.py::compare_engine_ec50 "
                "refuses it).",
                "CENSORING: 60 of temozolomide's 166 ChEMBL records are "
                "right-censored (the assay saturated below 50% effect), and "
                "GDSC2's glioma IC50s are largely extrapolated beyond the "
                "highest tested concentration -- the measured values are "
                "therefore bounded, and treating the bounds as exact points "
                "would import the assay's detection limit as pharmacology.",
            ],
            "what_would_close_the_gap": [
                "A time-kill (or clonogenic surviving-fraction) assay on "
                "paired TMZ-sensitive and TMZ-resistant glioma lines reporting "
                "the net per-day kill rate at >= 4 concentrations.",
                "The untreated growth rate of the same lines on the same "
                "substrate, so growth_S/growth_R are measured rather than "
                "assumed.",
                "A tmz concentration axis in molar units, so a declared EC50 "
                "unit can be attached to the engine's parameters.",
                "The same experiment for the second drug on the same line "
                "pair, which is what would make the drug-B marrow and potency "
                "assumptions measured instead of swept.",
            ],
        },
        "provenance": {
            "pharmacology": {k: v.value for k, v in pharmacology_provenance().items()},
            "note": (
                "measured/curated inputs are ESTABLISHED at source; every "
                "quantity the engine actually consumes from them is UNKNOWN"
            ),
        },
        "cannot_prove": list(what_pharmacology_cannot_prove())
        + list(what_targets_cannot_show()),
        "engine_identity": engine_identity(),
        "artifact_provenance": provenance(),
        "wall_seconds": round(time.time() - started, 1),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, OUT)
    print(f"\nwrote {OUT}", flush=True)
    print(f"  median exact IC50 {summary.median_exact_nM:,.0f} nM "
          f"({summary.median_exact_nM / 1000:,.1f} uM), most potent "
          f"{summary.most_potent_exact_nM:,.0f} nM "
          f"({summary.most_potent_exact_nM / 1000:,.2f} uM), "
          f"{summary.n_right_censored} right-censored", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
