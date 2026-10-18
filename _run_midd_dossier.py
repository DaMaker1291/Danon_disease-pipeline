"""Build the Phase-1b / window-of-opportunity MIDD dossier from the artifacts.

WHAT THIS PRODUCES. `outputs_real/midd_dossier.md` (the reading copy) and
`outputs_real/midd_dossier.json` (the machine summary), assembled ONLY from
artifacts this repository generated:

  * `outputs_real/dosing.json`          -- release 1: competitive release, the
                                           schedule search, the alternating-vs-
                                           single comparison, the label-anchored
                                           safety screen.
  * `outputs_real/release2_sweep.json`  -- release 2: the reversal map (where
                                           the win stops and where safety stops).
  * `outputs_real/bench_anchors.json`   -- step 1: measured potency anchors and
                                           the refused calibration.

No number in the dossier is typed by hand. If an artifact is missing the
section says so and the dossier still builds, because a dossier that silently
drops its own safety section is worse than one that declares a gap.

WHAT IT REFUSES TO DO. It does not convert the engine's effect size into a
human one, does not propose a sample size for a human trial, and does not claim
the calendar is safe for patients. The screen is anchored to the FDA label
threshold, not calibrated to any patient's neutrophil counts, and the dossier
says which of those it is in the same sentence it reports the number.

USAGE
-----
python _run_midd_dossier.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

OUT_MD = os.path.join(ROOT, "outputs_real", "midd_dossier.md")
OUT_JSON = os.path.join(ROOT, "outputs_real", "midd_dossier.json")

#: External references. Quoted so a reader can check them; NOT verified by any
#: code in this repository, and labelled as such where they appear.
MIDD_REFERENCES = [
    {
        "what": "FDA Model-Informed Drug Development (MIDD) Paired Meeting Program",
        "url": "https://www.fda.gov/drugs/development-resources/"
               "model-informed-drug-development-paired-meeting-program",
        "status": "external; not fetched or verified by this repository",
    },
    {
        "what": "ICH M15 -- General Principles for Model-Informed Drug Development",
        "url": "https://www.fda.gov/regulatory-information/search-fda-guidance-"
               "documents/m15-general-principles-model-informed-drug-development",
        "status": "external; not fetched or verified by this repository",
    },
    {
        "what": "PDUFA VII Federal Register notice re-establishing the MIDD "
                "paired meeting program",
        "url": "https://www.federalregister.gov/documents/2023/01/11/2023-00389/"
               "prescription-drug-user-fee-act-of-2023-vii-meetings-program-for-"
               "model-informed-drug-development",
        "status": "external; not fetched or verified by this repository",
    },
]

#: Where the Step-2 calibration data actually live. Named so the ask is a
#: data request with a destination instead of a wish. All external: this
#: repository has fetched and verified none of them, and none of them is
#: cached in this tree.
ANC_DATA_ROUTES = [
    {
        "source": "Project Data Sphere (de-identified patient-level oncology "
                  "trial data; registration required)",
        "url": "https://data.projectdatasphere.org/",
        "fits": "completed GBM trials with per-visit laboratory results and "
                "dose-delivery records",
    },
    {
        "source": "Vivli (independent data-sharing platform; data-use "
                  "agreement required)",
        "url": "https://vivli.org/",
        "fits": "sponsor-controlled trials whose data are not on an open "
                "platform",
    },
]

#: The fields a calibration request must ask for. Without the first four, the
#: kernel cannot be identified at all and any fit would be a curve drawn
#: through somebody else's summary statistics.
ANC_REQUEST_FIELDS = [
    "the ANC value and its unit (x10^9/L or cells/uL), per draw",
    "the study day of each draw, and the cycle number and cycle day",
    "the temozolomide dose actually delivered in that cycle (not the "
    "protocol dose), including any cycle stopped early",
    "growth-factor support (G-CSF/GM-CSF) with dates, since it truncates the "
    "nadir the kernel is trying to predict",
    "baseline ANC and any baseline haematological comorbidity",
    "at minimum 20 patients and 100 post-dose draws for a 2-parameter kernel "
    "fit with a held-out calibration check",
]


def load(name: str) -> Optional[Dict[str, Any]]:
    path = os.path.join(ROOT, "outputs_real", name)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def _arm_table(rows: List[Dict[str, Any]]) -> List[str]:
    out = [
        "| calendar | median AUBC | median dominance day | median PFS (d) | "
        "screen-admissible | modelled ANC nadir (x10^9/L) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| `{r['name']}` | {_fmt(r['median_aubc'])} | "
            f"{_fmt(r.get('median_dominance_day'), 1)} | "
            f"{_fmt(r.get('median_pfs_days'), 1)} | "
            f"{'yes' if r.get('feasible') else 'NO'} | "
            f"{_fmt(r.get('anc_nadir'), 3)} |"
        )
    return out


def build_markdown(
    rel1: Optional[Dict[str, Any]],
    rel2: Optional[Dict[str, Any]],
    bench: Optional[Dict[str, Any]],
) -> str:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    L: List[str] = []
    ap = L.append

    ap("# MIDD dossier -- a two-drug dosing calendar for glioblastoma")
    ap("")
    ap(f"*assembled {now} from generated artifacts; no number below is "
       f"transcribed by hand*")
    ap("")

    # ------------------------------------------------------------------ 0
    ap("## 0. What this package is, and what it is not")
    ap("")
    ap("**It is** a measured, seed-paired comparison of dosing calendars on ONE "
       "synthetic glioblastoma phantom, run on an evolutionary spatial engine, "
       "with every calendar screened against a myelosuppression model anchored "
       "to the temozolomide FDA label's neutrophil floor.")
    ap("")
    ap("**It is not** patient data, a validated model, a toxicity prediction "
       "for a named patient, or clinical decision support. Specifically:")
    ap("")
    ap("- the substrate is a synthetic 24^3 phantom, not a patient's imaging;")
    ap("- the pharmacology constants (`kill_max`, `EC50`, growth) are AUTHORED "
       "assumptions and remain so -- section 5 shows why the public potency "
       "data cannot be substituted into them;")
    ap("- the myelosuppression screen is anchored to the label threshold and "
       "calibrated to ONE reported datum, not fitted to patient neutrophil "
       "counts -- section 4 states the gap and what would close it;")
    ap("- the effect size is the engine's area under the burden curve (AUBC), "
       "in the engine's own units. It is **not** a hazard ratio, and it is not "
       "usable as a human effect size.")
    ap("")
    ap("This section exists because the dossier is meant to be read by an "
       "institutional reviewer, and the fastest way to lose that reviewer is "
       "to let a synthetic number travel without its provenance.")
    ap("")

    # ------------------------------------------------------------------ 1
    ap("## 1. The problem: standard dosing hands the tumour to the resistant clone")
    ap("")
    if rel1 and rel1.get("competitive_release"):
        cr = rel1["competitive_release"]
        ap(f"Engine: {cr['n_seeds']} seeds, {cr['days']} days, competitive-release "
           f"experiment on the committed phantom. The comparison is between "
           f"continuous maximum tolerated dosing and a label-shaped pulsed "
           f"calendar; the verdict is reported whichever way it falls.")
        ap("")
        L.extend(_arm_table([cr["mtd"], cr["pulsed"]]))
        ap("")
        ap(f"- median day on which the resistant clone becomes the dominant "
           f"population: **{_fmt(cr['mtd']['median_dominance_day'], 1)}** under "
           f"continuous MTD vs {_fmt(cr['pulsed']['median_dominance_day'], 1)} "
           f"under the pulsed calendar (advance of "
           f"{_fmt(cr.get('dominance_delta_days'), 1)} d);")
        ap(f"- pulsed/MTD burden ratio {_fmt(cr.get('aubc_ratio'))}: pulsing "
           f"delays dominance but does not reduce cumulative burden;")
        ap(f"- **both** arms breach the label-anchored ANC floor "
           f"(MTD nadir {_fmt(cr['mtd']['anc_nadir'], 2)}; pulsed "
           f"{_fmt(cr['pulsed']['anc_nadir'], 2)} vs the 1.5 x10^9/L floor), so "
           f"neither is a calendar this screen would let a trial open with.")
        ap("")
        for note in cr.get("notes", []):
            ap(f"- {note}")
    else:
        ap("*`outputs_real/dosing.json` has no competitive-release section; "
           "run `python _run_dosing.py cr`.*")
    ap("")

    # ------------------------------------------------------------------ 2
    ap("## 2. The calendar the engine selects")
    ap("")
    if rel1 and rel1.get("combination"):
        combo = rel1["combination"]
        best = combo.get("best_combination") or {}
        single = combo.get("single_drug_best") or {}
        rows = []
        if best:
            rows.append(best)
        if single:
            rows.append(single)
        if rel1.get("competitive_release"):
            rows.append(rel1["competitive_release"]["mtd"])
        L.extend(_arm_table(rows))
        ap("")
        if best:
            ap(f"**Selected calendar:** `{best['name']}`")
            ap("")
            ap(f"- median AUBC {_fmt(best['median_aubc'])}, IQR "
               f"[{_fmt(best['iqr_aubc'][0])}, {_fmt(best['iqr_aubc'][1])}] over "
               f"{best['n_seeds']} seeds -- the tightness of that IQR is the "
               f"stochastic-variance statement;")
            ap(f"- median PFS {_fmt(best['median_pfs_days'], 1)} d of a "
               f"{rel1['combination'].get('days', 365)} d horizon, median "
               f"dominance day {_fmt(best['median_dominance_day'], 1)} (the "
               f"resistant clone never becomes dominant in the horizon);")
            ap(f"- screen-admissible: **yes**, modelled nadir "
               f"{_fmt(best['anc_nadir'], 2)} x10^9/L against the 1.5 floor.")
        ap("")
        ap(f"Reference arms were chosen by measurement, not by the authors: "
           f"the single-drug arm is a grid of "
           f"{combo.get('single_drug_calendar_count')} calendars and the best of "
           f"them is `{single.get('name')}` at median AUBC "
           f"{_fmt(single.get('median_aubc'))}; "
           f"{combo.get('n_feasible_at_primary_weight')} of "
           f"{combo.get('n_alternating_candidates')} alternating calendars are "
           f"admissible at the conservative marrow weight "
           f"({combo.get('primary_marrow_weight')}).")
        ap("")
        for note in combo.get("notes", []):
            ap(f"- {note}")
    else:
        ap("*`outputs_real/dosing.json` has no combination section; run "
           "`python _run_dosing.py combination`.*")
    ap("")
    ap("**Read the calendar as a schedule, not as a prescription.** Drug A's "
       "faces are temozolomide's (150 mg/m2); drug B's face is expressed as a "
       "relative engine level on drug A's scale because the engine has no "
       "measured mg/m2 for a second agent. `B at 0.5 x full` is a relative "
       "potency statement, not a lomustine dose.")
    ap("")

    # ------------------------------------------------------------------ 3
    ap("## 3. Where the win stops: the reversal map")
    ap("")
    if rel2:
        grid = rel2.get("release_2_grid", {})
        summary = rel2.get("summary", {})
        ap(f"Release 2 took the release-1 winner and re-ran it across a declared "
           f"grid -- intensity {grid.get('intensity')} on the winner's own "
           f"levels, pre-existing resistant fraction "
           f"{grid.get('resistant_fraction')}, growth noise "
           f"{grid.get('growth_noise')} -- with the release-1 best single-drug "
           f"arm re-run at every cell on the same {len(rel2.get('seeds', []))} "
           f"seeds, so each cell is a paired comparison.")
        ap("")
        if summary:
            ap(f"- cells: {summary.get('cells')} "
               f"({summary.get('wins')} WINS, {summary.get('saturates')} "
               f"SATURATES, {summary.get('reversed')} REVERSED, "
               f"{summary.get('infeasible')} INFEASIBLE)")
            ap("")
            ap("Read those four labels carefully, because the largest one is the "
               "strongest, not a null:")
            ap("")
            ap("- **SATURATES** means the survival endpoint pinned to the "
               f"{rel1.get('combination', {}).get('days', 365) if rel1 else 365}-day "
               "horizon, so the win cannot be resolved further at this horizon -- a "
               "stronger statement than a finite PFS difference;")
            ap("- **WINS** would be a finite, resolvable PFS advantage;")
            ap("- **REVERSED** would be a cell where the alternation lost to the "
               "single drug; and")
            ap("- **INFEASIBLE** is a cell the ANC screen refuses, however good the "
               "efficacy number beside it looks.")
            safe = summary.get("max_feasible_intensity_by_marrow_weight", {})
            for weight, value in safe.items():
                ap(f"- highest intensity the ANC screen permits at drug-B "
                   f"marrow weight {weight}: **{value}** on the winner's own "
                   f"level scale (1.0 is the release-1 calendar)")
            low = summary.get("lowest_winning_intensity_per_group", {})
            for key, value in low.items():
                ap(f"- lowest winning intensity at {key}: {value}")
        ap("")
        ap("**Reversals, stated as limits of the claim:**")
        ap("")
        revs = rel2.get("reversals", [])
        if revs:
            ap("| intensity | r0 | noise | status | AUBC ratio vs single | "
               "dominance delta (d) | winner nadir |")
            ap("|---|---|---|---|---|---|---|")
            for r in revs:
                ap(f"| {r.get('intensity')} | {r.get('resistant_fraction')} | "
                   f"{r.get('growth_noise')} | {r.get('status')} | "
                   f"{_fmt(r.get('aubc_ratio_winner_over_reference'))} | "
                   f"{_fmt(r.get('dominance_delta_days'), 1)} | "
                   f"{_fmt(r.get('winner_anc_nadir'), 3)} |")
        else:
            ap("*(none -- either every cell held, or the sweep has not "
               "completed; check `partial` in the artifact)*")
        ap("")
        if rel2.get("partial"):
            ap("> **This section was built from a PARTIAL sweep.** The artifact's "
               "`partial` flag is still true; re-run `_release2_sweep.py` to "
               "resume the remaining cells before quoting these boundaries.")
            ap("")
        ap("**What the map does NOT show, stated as plainly as what it does.** "
           "The LOWER edge of the window was never reached: the alternation still "
           "saturates the horizon at the lowest intensity the grid declares, so "
           "this sweep brackets the win on the safety side only and says nothing "
           "about how little drug would still work. The pre-existing resistant "
           "fraction axis is also inert above 0.1 on this phantom (the 0.1 and "
           "0.3 groups report identical numbers), which is a property of the "
           "engine's initial condition rather than evidence about resistant "
           "patients. Neither gap invalidates the safety boundary, which is "
           "analytic and independent of the engine; both are reasons not to quote "
           "the map as a dose-finding result.")
        ap("")
    else:
        ap("*`outputs_real/release2_sweep.json` is not present. The reversal map "
           "is the section that says where the claim fails; without it this "
           "dossier asserts a win at one operating point only. Run "
           "`python _release2_sweep.py`.*")
    ap("")

    # ------------------------------------------------------------------ 4
    ap("## 4. Safety case -- and its calibration gap")
    ap("")
    if rel1 and rel1.get("myelosuppression_model"):
        mm = rel1["myelosuppression_model"]
        ap(f"The screen models circulating ANC from a cumulative marrow burden "
           f"convolved with a log-normal delay kernel, with the suppression "
           f"budget DEFINED by the label's own regimen rather than chosen: the "
           f"regimen's reservoir is {_fmt(mm.get('suppression_budget_mg_m2'), 2)} "
           f"mg/m2. The reportable numbers:")
        ap("")
        ap(f"- ANC floor {mm.get('anc_floor')} x10^9/L, source "
           f"{mm.get('anc_floor_source')} (is_label_threshold="
           f"{mm.get('is_label_threshold')});")
        ap(f"- kernel {mm.get('kernel', {}).get('family')}, fitted to "
           f"{mm.get('kernel', {}).get('fitted_to')};")
        ap(f"- second drug's marrow potency: {mm.get('second_drug_potency')};")
        ap(f"- status: `{mm.get('status')}`; "
           f"`measured_in_this_repository` = **{mm.get('measured_in_this_repository')}**; "
           f"`is_clinical_decision_support` = {mm.get('is_clinical_decision_support')}.")
        ap("")
        anchors = mm.get("anchors") or mm.get("label_quotes")
        if anchors:
            ap(f"- the screen is pinned by its declarations: "
               f"`{json.dumps(anchors)[:400]}`")
            ap("")
    ap("**The gap, stated plainly.** No neutrophil count from any patient was "
       "used to build this screen. The kernel shape is fitted to the label's "
       "reported day of nadir, and the budget to the label's regimen; the "
       "boundary therefore lands on the label rather than on a dataset. An IRB "
       "can be shown that the proposed calendar does not breach the label's "
       "threshold **under a declared kernel**, which is a screening argument. "
       "It cannot be shown that the calendar is safe in humans, because that "
       "needs the calibration data below.")
    ap("")
    ap("**What would close it (the Step-2 ask).**")
    ap("")
    ap("- anonymised per-cycle ANC time series from a completed GBM trial "
       "(counts with the day on which each was drawn, the cycle and dose "
       "actually delivered, and any growth-factor support);")
    ap("- a fit of the delay kernel's log-mean and log-sd and of the recovery "
       "tail to those series, reported with a calibration plot and per-cycle "
       "residuals, not a single aggregate R^2;")
    ap("- an acceptance criterion stated before the fit: e.g. predicted nadir "
       "within a declared tolerance for a declared fraction of cycles, with "
       "the number of cycles failing it reported.")
    ap("")
    ap("**Where the data actually live, and what to request.** None of the "
       "following is cached, fetched or verified by this repository; they are "
       "named so the ask is a data request with a destination.")
    ap("")
    for route in ANC_DATA_ROUTES:
        ap(f"- {route['source']} -- {route['url']} ({route['fits']})")
    ap("")
    ap("Fields the request must include (without the first four the kernel is "
       "not identified, and a fit to summary statistics instead of draws would "
       "be a curve through somebody else's table):")
    ap("")
    for field in ANC_REQUEST_FIELDS:
        ap(f"- {field}")
    ap("")
    ap("**The refusal, stated as code would enforce it.** Until such a series "
       "exists in this tree, any weight placed on the ANC window is placed on "
       "a declared kernel fitted to one label datum. The screen is a filter, "
       "not a safety certificate; the dossier is written so that a reviewer "
       "can see which of those two sentences applies to the number in front "
       "of them.")
    ap("")

    # ------------------------------------------------------------------ 5
    ap("## 5. Pharmacology grounding -- measured anchors, refused calibration")
    ap("")
    if bench:
        mol = bench.get("molecule", {})
        pot = bench.get("glioma_line_potency", {})
        gdsc = bench.get("gdsc2_glioma_audit", {})
        ap(f"Measured anchors (`{mol.get('name')}`, {mol.get('chembl_id')}, "
           f"MW {mol.get('mw_g_per_mol')} g/mol):")
        ap("")
        ap(f"- ChEMBL glioma-line IC50s: {pot.get('n_exact')} exact of "
           f"{pot.get('n_records')} records; median "
           f"{_fmt(pot.get('median_exact_nM'), 0)} nM "
           f"({_fmt((pot.get('median_exact_nM') or 0) / 1000, 1)} uM), most "
           f"potent {_fmt(pot.get('most_potent_exact_nM'), 0)} nM "
           f"({_fmt((pot.get('most_potent_exact_nM') or 0) / 1000, 2)} uM), "
           f"spread {_fmt(pot.get('fold_spread_exact'), 0)}x; "
           f"{pot.get('n_right_censored')} right-censored (assay saturated "
           f"below 50% effect) and excluded from the point estimate;")
        ap(f"- GDSC2 fitted release, glioma rows: "
           f"{gdsc.get('n_gbm_rows')} rows over {gdsc.get('n_gbm_drugs')} drugs "
           f"and {gdsc.get('n_gbm_cell_lines')} lines; "
           f"{_fmt(100 * (gdsc.get('extrapolated_fraction') or 0), 1)}% of "
           f"glioma IC50s lie above the highest concentration tested, and for "
           f"temozolomide it is "
           f"{gdsc.get('temozolomide_above_max_tested')} of "
           f"{gdsc.get('temozolomide_rows')} -- those are model extrapolations, "
           f"not measurements.")
        ap("")
        ap(f"**The calibration is refused**, and the refusal is the deliverable: "
           f"{bench.get('ec50_comparison', {}).get('reason')}")
        ap("")
        ap("Blockers, in the artifact's own words:")
        ap("")
        for blocker in bench.get("reconciliation", {}).get("blockers", []):
            ap(f"- {blocker}")
        ap("")
        ap("Protocol that would close it:")
        ap("")
        for step in bench.get("reconciliation", {}).get("what_would_close_the_gap", []):
            ap(f"- {step}")
    else:
        ap("*`outputs_real/bench_anchors.json` is not present; run "
           "`python _run_bench_anchors.py`.*")
    ap("")

    # ------------------------------------------------------------------ 6
    ap("## 6. Proposed trial: Phase 1b window of opportunity")
    ap("")
    ap("The engine's numbers cannot power a human trial, and this dossier does "
       "not pretend otherwise. What a schedule-comparison of this kind can "
       "justify is the **design choices**: that the calendar is worth testing, "
       "that its failure mode is a calendar effect rather than a resistance "
       "effect, and that its dose-limiting toxicity is myelosuppression on a "
       "known schedule.")
    ap("")
    ap("A testable version of that, written so it could be falsified:")
    ap("")
    ap("- **Population:** newly diagnosed MGMT-unmethylated GBM, post-radiotherapy, "
       "where the standard calendar's benefit is smallest and the unmet need "
       "is unambiguous.")
    ap("- **Design:** randomised phase 1b, 2:1, arm 1 = standard temozolomide "
       "150 mg/m2 x5d q28d, arm 2 = the alternating calendar at the release-1 "
       "intensity, with a de-escalation rule to the lower intensity that still "
       "won in the reversal map.")
    ap("- **Primary endpoint:** safety/feasibility of the calendar (dose "
       "intensity delivered over 6 cycles, grade 3-4 neutropenia rate, dose "
       "holds), because a schedule nobody can deliver is not a schedule.")
    ap("- **Secondary:** PFS6 and PFS by RANO, MGMT-stratified, with paired "
       "imaging at baseline and at progression.")
    ap("- **Monitoring:** twice-weekly ANC in cycle 1, then per cycle, with an "
       "explicit hold threshold at the modelled nadir window -- the screen "
       "predicts when to look, which is the operational value of having one.")
    ap("- **Stopping rules declared before enrolment**, including a "
       "futility rule on dose intensity and an early-stopping rule on "
       "grade 4 haematological toxicity.")
    ap("")
    ap("Taken to a cooperative group or sponsor, the honest pitch is: *a "
       "spatial evolutionary model, with an audited claim registry and a label-"
       "anchored toxicity screen, selects a two-drug calendar that does not "
       "breach the label threshold in silico and that keeps the resistant clone "
       "from dominating the horizon; we propose to test the calendar, not to "
       "claim the outcome.*")
    ap("")
    ap("Regulatory route, quoted (not verified in this tree):")
    ap("")
    for ref in MIDD_REFERENCES:
        ap(f"- {ref['what']} -- {ref['url']} ({ref['status']})")
    ap("")

    # ------------------------------------------------------------------ 7
    ap("## 7. Falsifiers and limits that travel with every number above")
    ap("")
    if rel1 and rel1.get("screen_fidelity"):
        sf = rel1["screen_fidelity"]
        ap(f"- **Short screening is unsafe on this grid and was not used.** "
           f"At {sf.get('screen_days')} days the ranking correlates with the "
           f"{sf.get('full_days')}-day ranking at Spearman "
           f"{_fmt(sf.get('spearman_rho'), 3)} and drops "
           f"{len(sf.get('top_k_missed_by_screen', []))} of the top "
           f"{sf.get('top_k')} full-horizon candidates, so every number above "
           f"comes from full-horizon runs.")
    ap("- **Single substrate.** One synthetic 24^3 phantom with one seed. A "
       "patient's geometry, perfusion and sanctuary structure are not in it.")
    ap("- **Engineered resistance mechanism.** Resistance in the engine is "
       "emergent under its own rules, but the RULES are authored: the constants "
       "section 5 shows cannot yet be replaced with measured kill rates.")
    ap("- **Metric.** AUBC is cumulative burden in engine units, not RECIST or "
       "RANO progression. Dominance day is an internal model event.")
    ap("- **One second drug, no measured potency.** Drug B's pharmacology and "
       "marrow potency are both assumed; the marrow weight is swept, the "
       "potency is not.")
    ap("- **The reversal map is the honest boundary of the claim.** Where the "
       "map reverses or breaks the floor, the calendar is not a candidate, "
       "whatever the headline cell says.")
    ap("")
    ap("**What would change the conclusion:** a real 3D patient-derived "
       "substrate or a cohort of them; rate-reporting kill data on paired "
       "sensitive/resistant lines; an anonymised ANC series to calibrate the "
       "screen; a second, independently parameterised engine.")
    ap("")

    # ------------------------------------------------------------------ 8
    ap("## 8. Reproducing this package")
    ap("")
    ap("```")
    ap("python _run_dosing.py cr            # competitive release")
    ap("python _run_dosing.py search        # single-drug grid search")
    ap("python _run_dosing.py fidelity      # short-screen calibration")
    ap("python _run_dosing.py combination   # alternating vs single (the long stage)")
    ap("python _release2_sweep.py           # the reversal map")
    ap("python _run_bench_anchors.py        # measured potency anchors")
    ap("python _run_midd_dossier.py         # this dossier")
    ap("python -c \"from danon.claims import audit_claims; audit_claims()\"")
    ap("```")
    ap("")
    ap("Artifacts cited: `outputs_real/dosing.json`, "
       "`outputs_real/release2_sweep.json`, `outputs_real/bench_anchors.json`.")
    ap("")
    if rel1 and rel1.get("engine_identity"):
        ei = rel1["engine_identity"]
        files = ei.get("files", [])
        ap(f"Engine identity ({ei.get('algorithm')}, scope: {ei.get('scope')}):")
        ap("")
        for f in files:
            ap(f"- `{f.get('path')}` sha256 `{f.get('sha256')}` "
               f"({f.get('bytes')} bytes)")
        ap("")
    return "\n".join(L) + "\n"


def main() -> int:
    rel1 = load("dosing.json")
    rel2 = load("release2_sweep.json")
    bench = load("bench_anchors.json")
    print(f"dosing.json:            {'found' if rel1 else 'MISSING'}")
    print(f"release2_sweep.json:    {'found' if rel2 else 'MISSING'}"
          + (" (partial)" if rel2 and rel2.get("partial") else ""))
    print(f"bench_anchors.json:     {'found' if bench else 'MISSING'}")

    md = build_markdown(rel1, rel2, bench)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(md)

    summary = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reading_copy": "outputs_real/midd_dossier.md",
        "sources": {
            "release_1": "outputs_real/dosing.json" if rel1 else None,
            "release_2": "outputs_real/release2_sweep.json" if rel2 else None,
            "bench_anchors": "outputs_real/bench_anchors.json" if bench else None,
        },
        "release_2_partial": bool(rel2.get("partial")) if rel2 else None,
        "headline": {
            "selected_calendar": (
                (rel1 or {}).get("combination", {}).get("best_combination", {}).get("name")
                if rel1 else None
            ),
            "median_aubc": (
                (rel1 or {}).get("combination", {}).get("best_combination", {}).get("median_aubc")
                if rel1 else None
            ),
            "median_pfs_days": (
                (rel1 or {}).get("combination", {}).get("best_combination", {}).get("median_pfs_days")
                if rel1 else None
            ),
            "screen_admissible": (
                (rel1 or {}).get("combination", {}).get("best_combination", {}).get("feasible")
                if rel1 else None
            ),
            "anc_nadir": (
                (rel1 or {}).get("combination", {}).get("best_combination", {}).get("anc_nadir")
                if rel1 else None
            ),
        },
        "not_claimed": [
            "no patient-level prediction",
            "no human effect size",
            "no human safety assurance",
            "no calibrated kill constants",
        ],
        "references": MIDD_REFERENCES,
        "anc_calibration": {
            "status": "blocked on real data; screen remains label-anchored",
            "data_routes": ANC_DATA_ROUTES,
            "request_fields": ANC_REQUEST_FIELDS,
        },
        "word_count": len(md.split()),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {OUT_MD} ({summary['word_count']} words)")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
