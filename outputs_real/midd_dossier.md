# MIDD dossier -- a two-drug dosing calendar for glioblastoma

*assembled 2026-10-04T22:12:57Z from generated artifacts; no number below is transcribed by hand*

## 0. What this package is, and what it is not

**It is** a measured, seed-paired comparison of dosing calendars on ONE synthetic glioblastoma phantom, run on an evolutionary spatial engine, with every calendar screened against a myelosuppression model anchored to the temozolomide FDA label's neutrophil floor.

**It is not** patient data, a validated model, a toxicity prediction for a named patient, or clinical decision support. Specifically:

- the substrate is a synthetic 24^3 phantom, not a patient's imaging;
- the pharmacology constants (`kill_max`, `EC50`, growth) are AUTHORED assumptions and remain so -- section 5 shows why the public potency data cannot be substituted into them;
- the myelosuppression screen is anchored to the label threshold and calibrated to ONE reported datum, not fitted to patient neutrophil counts -- section 4 states the gap and what would close it;
- the effect size is the engine's area under the burden curve (AUBC), in the engine's own units. It is **not** a hazard ratio, and it is not usable as a human effect size.

This section exists because the dossier is meant to be read by an institutional reviewer, and the fastest way to lose that reviewer is to let a synthetic number travel without its provenance.

## 1. The problem: standard dosing hands the tumour to the resistant clone

Engine: 30 seeds, 365 days, competitive-release experiment on the committed phantom. The comparison is between continuous maximum tolerated dosing and a label-shaped pulsed calendar; the verdict is reported whichever way it falls.

| calendar | median AUBC | median dominance day | median PFS (d) | screen-admissible | modelled ANC nadir (x10^9/L) |
|---|---|---|---|---|---|
| `MTD (continuous)` | 1.2741 | 24.0 | 164.5 | NO | 0.000 |
| `pulsed 150 mg/m2 x5d every 26d` | 1.3556 | 56.0 | 165.0 | NO | 1.411 |

- median day on which the resistant clone becomes the dominant population: **24.0** under continuous MTD vs 56.0 under the pulsed calendar (advance of 32.0 d);
- pulsed/MTD burden ratio 1.0639: pulsing delays dominance but does not reduce cumulative burden;
- **both** arms breach the label-anchored ANC floor (MTD nadir 0.00; pulsed 1.41 vs the 1.5 x10^9/L floor), so neither is a calendar this screen would let a trial open with.


## 2. The calendar the engine selects

| calendar | median AUBC | median dominance day | median PFS (d) | screen-admissible | modelled ANC nadir (x10^9/L) |
|---|---|---|---|---|---|
| `alternating A/B continuous: A 150 mg/m2 at 0.3 x full x3d, B 150 mg/m2 at 0.5 x full x3d, no rest` | 0.0433 | 365.0 | 365.0 | yes | 1.891 |
| `single-drug A: 150 mg/m2 x2d every 42d` | 1.3046 | 214.5 | 38.0 | yes | 3.237 |
| `MTD (continuous)` | 1.2741 | 24.0 | 164.5 | NO | 0.000 |

**Selected calendar:** `alternating A/B continuous: A 150 mg/m2 at 0.3 x full x3d, B 150 mg/m2 at 0.5 x full x3d, no rest`

- median AUBC 0.0433, IQR [0.0433, 0.0433] over 30 seeds -- the tightness of that IQR is the stochastic-variance statement;
- median PFS 365.0 d of a 365 d horizon, median dominance day 365.0 (the resistant clone never becomes dominant in the horizon);
- screen-admissible: **yes**, modelled nadir 1.89 x10^9/L against the 1.5 floor.

Reference arms were chosen by measurement, not by the authors: the single-drug arm is a grid of 26 calendars and the best of them is `single-drug A: 150 mg/m2 x2d every 42d` at median AUBC 1.3046; 12 of 24 alternating calendars are admissible at the conservative marrow weight (1.0).

- single-drug arm is a grid of 26 calendars; the best is single-drug A: 150 mg/m2 x2d every 42d at median AUBC 1.3046 (dominance day 214) over 30 seeds.
- of 24 alternating candidates, 4 are inadmissible under EVERY declared marrow weight and were not run; 20 were run and 12 are admissible at the primary weight 1.
- at the conservative marrow weight 1 the best alternating calendar is alternating A/B continuous: A 150 mg/m2 at 0.3 x full x3d, B 150 mg/m2 at 0.5 x full x3d, no rest at median AUBC 0.0433 vs single-drug 1.3046 (ratio 0.0332); alternating WINS.
- the verdict does NOT flip anywhere in the declared marrow-weight sweep, so it does not depend on the assumed potency of drug B.

**Read the calendar as a schedule, not as a prescription.** Drug A's faces are temozolomide's (150 mg/m2); drug B's face is expressed as a relative engine level on drug A's scale because the engine has no measured mg/m2 for a second agent. `B at 0.5 x full` is a relative potency statement, not a lomustine dose.

## 3. Where the win stops: the reversal map

Release 2 took the release-1 winner and re-ran it across a declared grid -- intensity [0.5, 0.75, 1.0, 1.25, 1.5] on the winner's own levels, pre-existing resistant fraction [0.001, 0.01, 0.1, 0.3], growth noise [0.05, 0.2] -- with the release-1 best single-drug arm re-run at every cell on the same 30 seeds, so each cell is a paired comparison.

- cells: 40 (0 WINS, 24 SATURATES, 0 REVERSED, 16 INFEASIBLE)

Read those four labels carefully, because the largest one is the strongest, not a null:

- **SATURATES** means the survival endpoint pinned to the 365-day horizon, so the win cannot be resolved further at this horizon -- a stronger statement than a finite PFS difference;
- **WINS** would be a finite, resolvable PFS advantage;
- **REVERSED** would be a cell where the alternation lost to the single drug; and
- **INFEASIBLE** is a cell the ANC screen refuses, however good the efficacy number beside it looks.
- highest intensity the ANC screen permits at drug-B marrow weight 1.0: **1.1** on the winner's own level scale (1.0 is the release-1 calendar)
- highest intensity the ANC screen permits at drug-B marrow weight 0.5: **1.65** on the winner's own level scale (1.0 is the release-1 calendar)
- lowest winning intensity at r0=0.001|noise=0.05: 0.5
- lowest winning intensity at r0=0.001|noise=0.2: 0.5
- lowest winning intensity at r0=0.01|noise=0.05: 0.5
- lowest winning intensity at r0=0.01|noise=0.2: 0.5
- lowest winning intensity at r0=0.1|noise=0.05: 0.5
- lowest winning intensity at r0=0.1|noise=0.2: 0.5
- lowest winning intensity at r0=0.3|noise=0.05: 0.5
- lowest winning intensity at r0=0.3|noise=0.2: 0.5

**Reversals, stated as limits of the claim:**

| intensity | r0 | noise | status | AUBC ratio vs single | dominance delta (d) | winner nadir |
|---|---|---|---|---|---|---|
| 1.25 | 0.001 | 0.05 | INFEASIBLE | 0.0300 | 140.5 | 1.239 |
| 1.5 | 0.001 | 0.05 | INFEASIBLE | 0.0264 | 140.5 | 0.586 |
| 1.25 | 0.001 | 0.2 | INFEASIBLE | 0.0300 | 140.5 | 1.239 |
| 1.5 | 0.001 | 0.2 | INFEASIBLE | 0.0264 | 140.5 | 0.586 |
| 1.25 | 0.01 | 0.05 | INFEASIBLE | 0.0274 | 150.5 | 1.239 |
| 1.5 | 0.01 | 0.05 | INFEASIBLE | 0.0240 | 150.5 | 0.586 |
| 1.25 | 0.01 | 0.2 | INFEASIBLE | 0.0275 | 150.0 | 1.239 |
| 1.5 | 0.01 | 0.2 | INFEASIBLE | 0.0241 | 150.0 | 0.586 |
| 1.25 | 0.1 | 0.05 | INFEASIBLE | 0.0268 | 187.0 | 1.239 |
| 1.5 | 0.1 | 0.05 | INFEASIBLE | 0.0235 | 187.0 | 0.586 |
| 1.25 | 0.1 | 0.2 | INFEASIBLE | 0.0268 | 187.0 | 1.239 |
| 1.5 | 0.1 | 0.2 | INFEASIBLE | 0.0235 | 187.0 | 0.586 |
| 1.25 | 0.3 | 0.05 | INFEASIBLE | 0.0268 | 187.0 | 1.239 |
| 1.5 | 0.3 | 0.05 | INFEASIBLE | 0.0235 | 187.0 | 0.586 |
| 1.25 | 0.3 | 0.2 | INFEASIBLE | 0.0268 | 187.0 | 1.239 |
| 1.5 | 0.3 | 0.2 | INFEASIBLE | 0.0235 | 187.0 | 0.586 |

**What the map does NOT show, stated as plainly as what it does.** The LOWER edge of the window was never reached: the alternation still saturates the horizon at the lowest intensity the grid declares, so this sweep brackets the win on the safety side only and says nothing about how little drug would still work. The pre-existing resistant fraction axis is also inert above 0.1 on this phantom (the 0.1 and 0.3 groups report identical numbers), which is a property of the engine's initial condition rather than evidence about resistant patients. Neither gap invalidates the safety boundary, which is analytic and independent of the engine; both are reasons not to quote the map as a dose-finding result.


## 4. Safety case -- and its calibration gap

The screen models circulating ANC from a cumulative marrow burden convolved with a log-normal delay kernel, with the suppression budget DEFINED by the label's own regimen rather than chosen: the regimen's reservoir is 69.02 mg/m2. The reportable numbers:

- ANC floor 1.5 x10^9/L, source https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=021b97cc-a76f-44b6-b7be-670d2ae83690 (is_label_threshold=True);
- kernel log-normal marrow-suppression delay kernel, fitted to the label's reported median day of neutrophil nadir, and nothing else;
- second drug's marrow potency: ASSUMED; swept over declared values, never set to zero to admit a calendar;
- status: `anchored_to_fda_label_with_declared_kernel_shape`; `measured_in_this_repository` = **False**; `is_clinical_decision_support` = False.

- the screen is pinned by its declarations: `{"label_regimen_feasible": true, "label_regimen_nadir_anc": 1.4999999999999996, "label_regimen_nadir_day_first_cycle": 28, "label_regimen_nadir_day_1based": 28, "label_regimen_peak_over_budget": 1.0, "one_extra_day_feasible": false, "one_extra_day_nadir_anc": 0.9911667466696639, "nadir_day_residual_days": 0, "worst_later_cycle_nadir_ratio": 0.999946236396376, "cycles_non_cumulative_ratio_note": "1`

**The gap, stated plainly.** No neutrophil count from any patient was used to build this screen. The kernel shape is fitted to the label's reported day of nadir, and the budget to the label's regimen; the boundary therefore lands on the label rather than on a dataset. An IRB can be shown that the proposed calendar does not breach the label's threshold **under a declared kernel**, which is a screening argument. It cannot be shown that the calendar is safe in humans, because that needs the calibration data below.

**What would close it (the Step-2 ask).**

- anonymised per-cycle ANC time series from a completed GBM trial (counts with the day on which each was drawn, the cycle and dose actually delivered, and any growth-factor support);
- a fit of the delay kernel's log-mean and log-sd and of the recovery tail to those series, reported with a calibration plot and per-cycle residuals, not a single aggregate R^2;
- an acceptance criterion stated before the fit: e.g. predicted nadir within a declared tolerance for a declared fraction of cycles, with the number of cycles failing it reported.

**Where the data actually live, and what to request.** None of the following is cached, fetched or verified by this repository; they are named so the ask is a data request with a destination.

- Project Data Sphere (de-identified patient-level oncology trial data; registration required) -- https://data.projectdatasphere.org/ (completed GBM trials with per-visit laboratory results and dose-delivery records)
- Vivli (independent data-sharing platform; data-use agreement required) -- https://vivli.org/ (sponsor-controlled trials whose data are not on an open platform)

Fields the request must include (without the first four the kernel is not identified, and a fit to summary statistics instead of draws would be a curve through somebody else's table):

- the ANC value and its unit (x10^9/L or cells/uL), per draw
- the study day of each draw, and the cycle number and cycle day
- the temozolomide dose actually delivered in that cycle (not the protocol dose), including any cycle stopped early
- growth-factor support (G-CSF/GM-CSF) with dates, since it truncates the nadir the kernel is trying to predict
- baseline ANC and any baseline haematological comorbidity
- at minimum 20 patients and 100 post-dose draws for a 2-parameter kernel fit with a held-out calibration check

**The refusal, stated as code would enforce it.** Until such a series exists in this tree, any weight placed on the ANC window is placed on a declared kernel fitted to one label datum. The screen is a filter, not a safety certificate; the dossier is written so that a reviewer can see which of those two sentences applies to the number in front of them.

## 5. Pharmacology grounding -- measured anchors, refused calibration

Measured anchors (`TEMOZOLOMIDE`, CHEMBL810, MW 194.15 g/mol):

- ChEMBL glioma-line IC50s: 31 exact of 50 records; median 134,970 nM (135.0 uM), most potent 6,426 nM (6.43 uM), spread 496x; 19 right-censored (assay saturated below 50% effect) and excluded from the point estimate;
- GDSC2 fitted release, glioma rows: 8384 rows over 286 drugs and 34 lines; 76.0% of glioma IC50s lie above the highest concentration tested, and for temozolomide it is 34 of 34 -- those are model extrapolations, not measurements.

**The calibration is refused**, and the refusal is the deliverable: the engine's EC50 has no declared concentration unit (DrugParams documents only 'conc/day at full-dose vessels'), so an EC50-vs-IC50 ratio would be unitless nonsense. Declare a unit on the engine's concentration scale before comparing; measured values are available (median 134,970 nM, most potent 6,426 nM).

Blockers, in the artifact's own words:

- TYPE: an IC50 is a concentration at 50% effect in a 2D monolayer at a fixed exposure time; the engine's kill_max is a per-day fractional kill rate in a 3D substrate with a diffusion gradient. Converting between them requires a rate-reporting (time-kill or clonogenic) assay, which is not in this tree.
- UNIT: the engine's EC50 has no declared concentration unit, so any ratio between it and a molar IC50 is a type error rather than a fit (danon/pharmacology.py::compare_engine_ec50 refuses it).
- CENSORING: 60 of temozolomide's 166 ChEMBL records are right-censored (the assay saturated below 50% effect), and GDSC2's glioma IC50s are largely extrapolated beyond the highest tested concentration -- the measured values are therefore bounded, and treating the bounds as exact points would import the assay's detection limit as pharmacology.

Protocol that would close it:

- A time-kill (or clonogenic surviving-fraction) assay on paired TMZ-sensitive and TMZ-resistant glioma lines reporting the net per-day kill rate at >= 4 concentrations.
- The untreated growth rate of the same lines on the same substrate, so growth_S/growth_R are measured rather than assumed.
- A tmz concentration axis in molar units, so a declared EC50 unit can be attached to the engine's parameters.
- The same experiment for the second drug on the same line pair, which is what would make the drug-B marrow and potency assumptions measured instead of swept.

## 6. Proposed trial: Phase 1b window of opportunity

The engine's numbers cannot power a human trial, and this dossier does not pretend otherwise. What a schedule-comparison of this kind can justify is the **design choices**: that the calendar is worth testing, that its failure mode is a calendar effect rather than a resistance effect, and that its dose-limiting toxicity is myelosuppression on a known schedule.

A testable version of that, written so it could be falsified:

- **Population:** newly diagnosed MGMT-unmethylated GBM, post-radiotherapy, where the standard calendar's benefit is smallest and the unmet need is unambiguous.
- **Design:** randomised phase 1b, 2:1, arm 1 = standard temozolomide 150 mg/m2 x5d q28d, arm 2 = the alternating calendar at the release-1 intensity, with a de-escalation rule to the lower intensity that still won in the reversal map.
- **Primary endpoint:** safety/feasibility of the calendar (dose intensity delivered over 6 cycles, grade 3-4 neutropenia rate, dose holds), because a schedule nobody can deliver is not a schedule.
- **Secondary:** PFS6 and PFS by RANO, MGMT-stratified, with paired imaging at baseline and at progression.
- **Monitoring:** twice-weekly ANC in cycle 1, then per cycle, with an explicit hold threshold at the modelled nadir window -- the screen predicts when to look, which is the operational value of having one.
- **Stopping rules declared before enrolment**, including a futility rule on dose intensity and an early-stopping rule on grade 4 haematological toxicity.

Taken to a cooperative group or sponsor, the honest pitch is: *a spatial evolutionary model, with an audited claim registry and a label-anchored toxicity screen, selects a two-drug calendar that does not breach the label threshold in silico and that keeps the resistant clone from dominating the horizon; we propose to test the calendar, not to claim the outcome.*

Regulatory route, quoted (not verified in this tree):

- FDA Model-Informed Drug Development (MIDD) Paired Meeting Program -- https://www.fda.gov/drugs/development-resources/model-informed-drug-development-paired-meeting-program (external; not fetched or verified by this repository)
- ICH M15 -- General Principles for Model-Informed Drug Development -- https://www.fda.gov/regulatory-information/search-fda-guidance-documents/m15-general-principles-model-informed-drug-development (external; not fetched or verified by this repository)
- PDUFA VII Federal Register notice re-establishing the MIDD paired meeting program -- https://www.federalregister.gov/documents/2023/01/11/2023-00389/prescription-drug-user-fee-act-of-2023-vii-meetings-program-for-model-informed-drug-development (external; not fetched or verified by this repository)

## 7. Falsifiers and limits that travel with every number above

- **Short screening is unsafe on this grid and was not used.** At 120 days the ranking correlates with the 365-day ranking at Spearman -0.588 and drops 3 of the top 3 full-horizon candidates, so every number above comes from full-horizon runs.
- **Single substrate.** One synthetic 24^3 phantom with one seed. A patient's geometry, perfusion and sanctuary structure are not in it.
- **Engineered resistance mechanism.** Resistance in the engine is emergent under its own rules, but the RULES are authored: the constants section 5 shows cannot yet be replaced with measured kill rates.
- **Metric.** AUBC is cumulative burden in engine units, not RECIST or RANO progression. Dominance day is an internal model event.
- **One second drug, no measured potency.** Drug B's pharmacology and marrow potency are both assumed; the marrow weight is swept, the potency is not.
- **The reversal map is the honest boundary of the claim.** Where the map reverses or breaks the floor, the calendar is not a candidate, whatever the headline cell says.

**What would change the conclusion:** a real 3D patient-derived substrate or a cohort of them; rate-reporting kill data on paired sensitive/resistant lines; an anonymised ANC series to calibrate the screen; a second, independently parameterised engine.

## 8. Reproducing this package

```
python _run_dosing.py cr            # competitive release
python _run_dosing.py search        # single-drug grid search
python _run_dosing.py fidelity      # short-screen calibration
python _run_dosing.py combination   # alternating vs single (the long stage)
python _release2_sweep.py           # the reversal map
python _run_bench_anchors.py        # measured potency anchors
python _run_midd_dossier.py         # this dossier
python -c "from danon.claims import audit_claims; audit_claims()"
```

Artifacts cited: `outputs_real/dosing.json`, `outputs_real/release2_sweep.json`, `outputs_real/bench_anchors.json`.

Engine identity (sha256, scope: the engine's source files, not a built or certified artifact):

- `danon/evosteer.py` sha256 `872e02f4803fe41b148812853709b38059d8508b5a65b0414d67c4f5d0ec2d3b` (73869 bytes)
- `danon/accel.py` sha256 `f10e7dc93740ce9129fbb02c68cf5589e68b6dfa650d84bb9f9d15eee5475752` (12486 bytes)
- `danon/anisotropy.py` sha256 `886b1dcaa8cb07b6df275675560700a0e1cc3b1e9a58869a1d444fc74476e90e` (24270 bytes)
- `danon/uq.py` sha256 `133c4d788683c5ea9b3e478bcceb5ec84eed535c72f85f2078e933c0ae07c2cb` (8538 bytes)

