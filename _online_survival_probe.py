"""Does the re-detection score associate with SURVIVAL? Live, on GLASS 2019.

The banner the app carries says the software has not been shown to improve
survival. The closest question the available data can actually answer is a
PROGNOSTIC one: among patients with paired primary/recurrence specimens, does
the out-of-fold re-detection score (and the measured re-detection rate itself)
separate longer from shorter overall survival? That is an ASSOCIATION, measured
on right-censored data with the repository's own Harrell's C -- it is not a
treatment effect, not a cure, and nothing in this probe was given to a patient.

Design, fixed before the clinical data is read:

* cohort        difg_glass_2019 (the independent study already scored by
                _online_independent_cohort.py), fetched live from cBioPortal
* exposure      (1) the patient's mean OUT-OF-FOLD task-A score, from the same
                repeated patient-level splits the record publishes, and
                (2) the measured fraction of that patient's primary variants
                that were re-detected -- a pure measurement, no model
* endpoint      OS_MONTHS + OS_STATUS, right-censored
* statistic     Harrell's C (danon.validation.concordance_index via
                danon.predictive.harrell_c_index), permutation null with 1000
                shuffles, two-sided around 0.5
* refusals      fewer than 30 patients with both score and survival -> no
                number is printed; unparseable OS -> counted, not imputed

Run:  python -u _online_survival_probe.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict

from _online_independent_cohort import API, VARIANT_PANEL, get
import numpy as np
from danon.evolution_forecast import (
    _oof_detection_aucs,
    detection_examples,
    paired_spectra,
    records_from_cache,
    repeated_folds,
)
from danon.predictive import _breslow_objective, cox_fit, harrell_c_index

STUDY = "difg_glass_2019"
MIN_PATIENTS = 30
N_PERMUTATIONS = 1000
SEED = 13


def fetch_mutations() -> list:
    records = []
    for gene, entrez in VARIANT_PANEL:
        url = (
            f"{API}/molecular-profiles/{STUDY}_mutations/mutations"
            f"?sampleListId={STUDY}_sequenced&entrezGeneId={entrez}&pageSize=100000"
        )
        code, payload = get(url)
        rows = payload if isinstance(payload, list) else []
        if code != 200:
            raise SystemExit(f"{gene}: status {code}")
        for rec in rows:
            rec = dict(rec)
            rec["_gene"] = gene
            records.append(rec)
    return records


def fetch_os() -> dict:
    status, data = get(
        f"{API}/studies/{STUDY}/clinical-data?clinicalDataType=PATIENT"
        "&projection=SUMMARY&pageSize=1000000"
    )
    if status != 200 or not data:
        raise SystemExit(f"clinical data unreadable: status={status}")
    out = defaultdict(dict)
    for row in data:
        out[row.get("patientId", "")][row.get("clinicalAttributeId", "")] = row.get(
            "value", ""
        )
    return out


def parse_time(value) -> float | None:
    if value is None:
        return None
    try:
        t = float(str(value).strip())
    except ValueError:
        return None
    return t if t >= 0 else None


def parse_event(value) -> int | None:
    """cBioPortal writes OS_STATUS as a coded label: `1:DECEASED`, `0:LIVING`.

    Both halves are accepted, and an exact-label fallback covers the uncoded
    form; anything else stays unparsed and is COUNTED as dropped rather than
    guessed, because a survival endpoint inferred from a string nobody wrote is
    the same fabrication this repository refuses everywhere else.
    """
    s = str(value or "").strip().upper()
    if not s or s in {"N/A", "NA", "NAN", "NONE"}:
        return None
    head = s.split(":", 1)[0].strip()
    if head in {"DECEASED", "1", "TRUE", "YES", "DIED", "EVENT"} or "DECEASED" in s:
        return 1
    if head in {"LIVING", "ALIVE", "0", "FALSE", "NO", "CENSORED"} or (
        "LIVING" in s or "ALIVE" in s
    ):
        return 0
    return None


def permutation_pvalue(risk, time, event, n_perm=N_PERMUTATIONS) -> dict:
    observed = harrell_c_index(risk, time, event).c_index
    if observed != observed:  # NaN
        return {"observed_c_index": None, "n_permutations": 0, "p_two_sided": None}
    rng = random.Random(SEED)
    risk_list = list(risk)
    as_far = 0
    for _ in range(n_perm):
        shuffled = risk_list[:]
        rng.shuffle(shuffled)
        c = harrell_c_index(shuffled, time, event).c_index
        if c == c and abs(c - 0.5) >= abs(observed - 0.5):
            as_far += 1
    return {
        "observed_c_index": observed,
        "n_permutations": n_perm,
        "p_two_sided": (as_far + 1) / (n_perm + 1),
    }


def analyse(name: str, risk_by_patient: dict) -> dict:
    clinical = fetch_os()
    times, events, risk = [], [], []
    dropped = {"no_survival_attr": 0, "unparseable_time": 0, "unparseable_event": 0}
    for pid, score in risk_by_patient.items():
        attrs = clinical.get(pid) or {}
        if "OS_MONTHS" not in attrs and "OS_STATUS" not in attrs:
            dropped["no_survival_attr"] += 1
            continue
        t = parse_time(attrs.get("OS_MONTHS"))
        e = parse_event(attrs.get("OS_STATUS"))
        if t is None:
            dropped["unparseable_time"] += 1
            continue
        if e is None:
            dropped["unparseable_event"] += 1
            continue
        times.append(t)
        events.append(e)
        risk.append(float(score))
    print(f"\n--- {name} ---")
    print(
        f"  patients with exposure: {len(risk_by_patient)}; with usable OS: "
        f"{len(times)} (dropped: {dropped})"
    )
    if len(times) < MIN_PATIENTS:
        return {
            "refused": (
                f"only {len(times)} patients carry both an exposure and a usable "
                f"OS_MONTHS/OS_STATUS pair (minimum {MIN_PATIENTS})"
            ),
            "dropped": dropped,
        }
    res = permutation_pvalue(risk, times, events)
    n_events = int(sum(events))
    print(
        f"  n={len(times)} events={n_events} censored={len(times) - n_events} "
        f"follow-up {min(times):.1f}-{max(times):.1f} months"
    )
    print(
        f"  Harrell's C = {res['observed_c_index']:.4f} "
        f"(permutation two-sided p = {res['p_two_sided']:.4f} over "
        f"{res['n_permutations']} shuffles)"
    )
    print(
        "  reading: C > 0.5 means the HIGHER score died sooner; this is a "
        "prognostic association on censored data, not a treatment effect"
    )
    return {**res, "n": len(times), "n_events": n_events, "dropped": dropped}


def adjusted_cox(mean_score: dict) -> dict:
    """Univariable and age/IDH-adjusted Cox for the score, with Wald CIs.

    The confounders are chosen BEFORE the fit because this cohort's survival is
    famously driven by them: age at diagnosis and IDH status. A score that
    loses its coefficient once those two are in the model was carrying their
    signal, which is the difference between an independent association and a
    proxy -- reported either way.
    """
    clinical = fetch_os()
    scores, ages, idh_flags, times, events = [], [], [], [], []
    skipped = {"incomplete_survival": 0, "missing_age": 0, "unparsed_idh": 0}
    idh_seen = set()
    for pid, score in mean_score.items():
        attrs = clinical.get(pid) or {}
        t = parse_time(attrs.get("OS_MONTHS"))
        e = parse_event(attrs.get("OS_STATUS"))
        if t is None or e is None:
            skipped["incomplete_survival"] += 1
            continue
        raw_age = attrs.get("AGE_AT_DX")
        try:
            age = float(str(raw_age).strip())
        except (TypeError, ValueError):
            skipped["missing_age"] += 1
            continue
        idh_raw = str(attrs.get("IDH_STATUS") or "").strip().upper()
        idh_seen.add(idh_raw)
        # cBioPortal codes this cohort as IDHMUT / IDHWT; the longer labels are
        # kept working too, and anything else stays UNPARSED (counted above)
        # rather than folded into one arm.
        if "MUTANT" in idh_raw or idh_raw.startswith("MUT") or "MUT" in idh_raw or idh_raw in {"M", "1"}:
            idh = 1.0
        elif (
            "WILD" in idh_raw
            or idh_raw.endswith("WT")
            or idh_raw in {"W", "0"}
        ):
            idh = 0.0
        else:
            skipped["unparsed_idh"] += 1
            continue
        scores.append(float(score))
        ages.append(age)
        idh_flags.append(idh)
        times.append(t)
        events.append(e)
    print("\n--- Cox: score alone, then adjusted for age and IDH ---")
    print(
        f"  usable patients={len(times)} events={int(sum(events))} "
        f"skipped={skipped} IDH_STATUS values seen={sorted(idh_seen)}"
    )
    if len(times) < MIN_PATIENTS or sum(events) < 20:
        return {"refused": "too few patients/events for a Cox fit", "skipped": skipped}

    def z(values):
        a = np.asarray(values, dtype=float)
        sd = float(a.std(ddof=1)) or 1.0
        return (a - a.mean()) / sd

    def fit(X, names):
        f = cox_fit(X, times, events, names)
        _, _, info = _breslow_objective(f.coefficients, np.asarray(X, float),
                                        np.asarray(times, float),
                                        np.asarray(events, int))
        se = np.sqrt(np.diag(np.linalg.inv(info + 1e-6 * np.eye(info.shape[0]))))
        out = {}
        for name, b, s in zip(f.names, f.coefficients, se):
            out[name] = {
                "coef": float(b),
                "hr": float(np.exp(b)),
                "ci_low": float(np.exp(b - 1.96 * s)),
                "ci_high": float(np.exp(b + 1.96 * s)),
            }
        print(f"  converged={f.converged} iterations={f.iterations}")
        for name in f.names:
            r = out[name]
            print(
                f"    {name:14s} HR {r['hr']:.3f} "
                f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
            )
        return out

    uni = fit(z(scores)[:, None], ["score_per_sd"])
    adj = fit(
        np.column_stack([z(scores), z(ages), np.asarray(idh_flags, float)]),
        ["score_per_sd", "age_per_sd", "idh_mutant"],
    )
    verdict = (
        "the score's coefficient survives adjustment"
        if adj["score_per_sd"]["ci_high"] < 1.0 or adj["score_per_sd"]["ci_low"] > 1.0
        else "the score's interval crosses 1 once age and IDH are in the model"
    )
    print(f"  reading: {verdict}")
    return {"univariable": uni, "adjusted_age_idh": adj, "reading": verdict,
            "skipped": skipped, "n": len(times), "events": int(sum(events))}


def main() -> int:
    print(f"=== live fetch: {STUDY} ===")
    records = fetch_mutations()
    recs, dropped = records_from_cache({"records": records, "source": API})
    spectra, pairs = paired_spectra(recs)
    det = detection_examples(pairs)
    print(
        f"  rows={len(recs)} dropped={dropped} patients="
        f"{len({r.patient_id for r in recs})} pairs={len(pairs)} "
        f"det_rows={len(det)}"
    )

    splits = repeated_folds([e.patient_id for e in det], seed=SEED)
    (
        _per_fold,
        row_scores,
        row_labels,
        row_patients,
        patient_scores,
        _patient_labels,
        patients,
    ) = _oof_detection_aucs(det, splits)

    mean_score = dict(zip(patients, patient_scores))
    rows_by_patient = defaultdict(list)
    for pid, label in zip(row_patients, row_labels):
        rows_by_patient[pid].append(bool(label))
    measured_rate = {
        pid: sum(v) / len(v) for pid, v in rows_by_patient.items() if v
    }

    results = {
        "study": STUDY,
        "endpoint": "OS_MONTHS + OS_STATUS (right-censored, Harrell's C)",
        "model_score": analyse("mean out-of-fold task-A score per patient", mean_score),
        "measured_re-detection_rate": analyse(
            "measured fraction of primary variants re-detected", measured_rate
        ),
        "cox_adjusted": adjusted_cox(mean_score),
        "interpretation_guard": (
            "a prognostic ASSOCIATION measured on one cohort; it is not a "
            "treatment effect, not evidence that anything here changes the "
            "course of anyone's disease, and not grounds for the banner's "
            "sentence to change"
        ),
    }
    with open("outputs_real/survival_association_probe.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, default=str)
    print("\nwrote outputs_real/survival_association_probe.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
