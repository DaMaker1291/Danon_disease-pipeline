# Validation Layers — what has been tested, and what that does and does not prove

This platform is a **predictive decision-support engine**, not a drug. So "testing" it
cannot mean curing a patient: it means establishing predictive fidelity, mathematical
robustness, and clinical plausibility, layer by layer, and stating exactly where each
layer stops.

Every number below is produced by the code in this repository and printed by the
artifacts named beside it. Nothing here is transcribed by hand.

Regenerate everything:

```bash
python -m pytest tests/ danon/multiagent/ -q          # the code layer
python -m pytest tests/test_retrospective.py -q        # the real-data layer
# artifacts (plots, packet, campaign log) live in outputs_real/
```

---

## Layer 0 — Implementation integrity (does the code do what it claims?)

`danon/campaign.py` accumulates millions of elementary property checks across paired
runs: non-negativity of every voxel of every field, dose bounds on every day,
concentration ceilings from the diffusion maximum principle, progression-bookkeeping
consistency, bit-exact determinism of repeated seeds.

* **Proves:** the implementation obeys its own specification; a seeded run reproduces
  exactly; a documented failure mode is a *counted* violation, not silence.
* **Does not prove:** anything about biology. Every run shares one model, one parameter
  set, one geometry class.

The test suite is the same idea at finer grain: **582 tests**, of which the ones that
matter scientifically are the integration tests (paired MTD controls, cohort KM
behaviour, refusal paths).

---

## Layer 1 — Ingestion anchor (is the REAL data handled correctly?)

`danon/retrospective.py` reconstructs two observed cohorts from the GDC open-access
clinical export for project **TCGA-GBM** (617 cases, fetched by API, cached in
`data/gdc_clinical/`), using this repository's own Kaplan-Meier estimator:

| arm | n | events | reconstructed median |
|---|---|---|---|
| recorded progression (`days_to_progression` with `progression_or_recurrence == "Yes"`) | 595 | 267 | 450 d (KM, censoring-inflated) |
| overall survival (`days_to_death`) | 593 | 491 | **427 d** |

`anchor_check()` requires that reconstructed median OS to land inside the published
glioblastoma standard-of-care band **330–550 d** (Stupp 2005: 14.6 months; TCGA-GBM
cohort reports 12–15 months). It does: 427 d ≈ 14.0 months.

* **Proves:** the ingestion path is correct. A wrong reconstruction cannot reproduce a
  textbook number, so this is a real test of the data handling.
* **Does not prove:** anything about the model. It is a test of *us reading the data*,
  not of the simulation.

---

## Layer 2 — Control-arm timescale plausibility (the weakest layer, labelled as such)

The model's **MTD arm** (its standard-of-care proxy) is compared against the observed
progression curve at fixed time points:

| day | observed S(t) | model S(t) | gap |
|---|---|---|---|
| 100 | 0.878 | 1.000 | +0.122 |
| 200 | 0.721 | 1.000 | +0.279 |
| 365 | 0.548 | 0.074 | **−0.473** |

Model MTD-arm median progression **268 d** (n = 54, 50 events) sits **inside** the
observed dated-event IQR (101–321 d) — consistent with the real timescale. The late
discrepancy is expected and explainable: the real cohort received maximal resection and
radiotherapy, and the model's control arm has **no resection or radiotherapy term**, so
it over-progresses at one year. That is a stated model limitation, not a hidden one.

* **Proves:** the model's control arm is not describing a different disease timescale.
* **Does not prove:** validation of any kind. No patient matching is possible (BraTS and
  GDC are anonymised differently, so no imaging case can be tied to a clinical case),
  treatment is heterogeneous, the endpoints are not matched (RANO-fraction burden
  threshold vs recorded clinical progression), and the model is not fitted to this
  cohort — so agreement is not calibration and disagreement would be a re-calibration
  target.

---

## Layer 3 — Parametric robustness (Monte Carlo over every assumed constant)

`danon/sensitivity.py` sweeps each load-bearing constant across a scale grid and re-runs
the **paired** experiment (policy vs MTD, identical seeds, identical geometry, identical
perturbed world) — 84 cells over 6 targets × 7 scales × 2 real BraTS geometries, 365-day
horizon. Primary criterion is mean tumour burden (defined for every replicate, so
nothing is dropped); PFS is secondary and reported only where *both* arms progressed.

Policy better on mean burden in **80/84** cells; materially better (ratio ≤ 0.75) in
**70/84**. Median burden ratio (policy / MTD) per target and scale:

| target scaled | 0.02 | 0.05 | 0.10 | 0.25 | 0.50 | 1.00 | 2.00 |
|---|---|---|---|---|---|---|---|
| `collateral_b_kill` (the mechanism) | 0.91 | 0.55 | 0.23 | 0.09 | 0.08 | 0.07 | 0.06 |
| `drug_a_kill` (selective pressure) | **1.00** | **1.00** | 0.84 | 0.27 | 0.10 | 0.07 | 0.08 |
| `tumour_growth` | 0.85 | 0.83 | 0.81 | 0.67 | 0.31 | 0.07 | 0.03 |
| `resistance_cost` | 0.04 | 0.04 | 0.04 | 0.05 | 0.05 | 0.07 | 0.18 |
| `drug_transport` | 0.07 | 0.07 | 0.06 | 0.07 | 0.07 | 0.07 | 0.07 |
| `intake_r0` | 0.16 | 0.13 | 0.12 | 0.10 | 0.08 | 0.07 | 0.06 |

Three results worth stating plainly:

1. **The binding constraint is not the collateral mechanism — it is the primary drug.**
   At or below 5% of the assumed Drug-A kill rate the schedule's advantage collapses to
   parity (ratio 1.00): holidays only work if the on-treatment phase applies real
   selective pressure. This is a *testable prediction for bench work*: measure the
   temozolomide kill rate first, not the collateral effect.
2. **The conclusion is now measurably sensitive to `intake_r0`** — the pre-existing
   resistant fraction — moving the ratio from 0.16 to 0.06 across the 0.02×–2× grid.
   This REVERSES the earlier reading of this same sweep (a flat 0.030 → 0.034, i.e.
   "insensitive to intake uncertainty"), and the reversal is a direct consequence of
   the extinction/detection floors: with mutation unable to supply resistance at the
   calibrated rate, the seeded fraction is the only source of resistance, so it is
   load-bearing. Direction: *more* pre-existing resistance makes the policy look
   better, because MTD then fails harder on its own. This is the correct place for
   the model's sensitivity to sit, and it is uncomfortable — `intake_r0` is an
   ASSUMED input (a calibrated transcriptome mapping, not a measured clonal
   fraction), so the headline contrast is now known to depend on a number this
   project does not measure.
3. **The engine's verdict, not just its magnitude, depends on that input.** With no
   pre-existing resistant clone, the same dose eradicates the tumour under MTD and the
   adaptive policy buys nothing at all (see the CORRECTION section in
   `danon/evosteer.py`). The sensitivity grid above perturbs r0 but never sets it to
   zero, so it cannot see that regime; the emergence tests in
   `tests/test_resistance_emergence.py` cover it instead.

*Re-measured after the resistance floors landed (extinction floor at half a cell per
voxel, detection floor at ~1e8 cells). The previous table read 81/84 and 77/84 with a
flat `intake_r0` row; the load-bearing conclusions above are unchanged, the magnitudes
are not, and the `intake_r0` row changed sign of finding.*

* **Proves:** which constants the conclusion depends on, and how far each can be wrong
  before the result changes.
* **Does not prove:** that the assumed constants are the true ones. Stability under a
  wrong number is insensitivity, not validity. The constants remain `UNKNOWN` in the
  provenance registry until measured (tier-3 bench campaign).

---

## Layer 4 — Paired baseline control (does it beat the status quo?)

Every cell above is a paired comparison against **MTD at the same seed in the same
world**; `danon/observer.py` additionally closes the loop with a controller that sees
only noisy CSF-derived observations and its own dose log, and reproduces the
perfect-information reference (**100% PFS retention vs a policy that can read the hidden
clonal state; 2.64× MTD PFS on paired seeds**).

The real-geometry campaign (6 BraTS 2021 cases × 3 cohort-derived r0s, 3 replicates
each, `outputs_real/campaign_log.json`) gives median PFS extension ≈ **+38.8%** (range
+9.0% to +51.5%, censored lower bounds) and toxicity reduction **0.9–2.3%**.

The toxicity figure moved from 11.7–17.7% to 0.9–2.3% when the resistance floors
landed, and the reason matters more than the number: the collateral switch used to take
drug-B holidays on the strength of a resistant fraction computed for a sub-detectable
tumour. R/(S+R) reads 1.0 as soon as the sensitive population is extinct, so the policy
was switching — and banking the drug-B toxicity saving — in response to a number that
described nothing. It no longer does, so it stays on drug A, and its toxicity advantage
largely disappears. The PFS advantage does not. Read together: **the survival contrast
in this model comes from the collateral mechanism, while the previously advertised
toxicity benefit was substantially an artifact of the degenerate ratio.**

* **Proves:** the policy beats standard dosing in this model, on real patient geometry,
  at equal information.
* **Does not prove:** that a real patient benefits. Nobody has received this schedule.

---

## In-vitro cross-validation with public drug-response data (GDSC / CTRP): NOT DONE

This layer is buildable and remains open. It would map publicly measured cell-line
kill curves (temozolomide and a metabolic inhibitor) onto the engine's kill-rate
constants, replacing `ASSUMED` with `EMERGING_HUMAN`-class evidence for at least the
two rate constants layer 3 identifies as load-bearing. Until then, say so out loud:
the kill-rate constants are assumptions that have passed a robustness sweep, nothing
more.

---

## The one-sentence summary

> The engine is verified as code (layer 0), the real-data ingestion is validated against
> published survival (layer 1), the control arm is on the right timescale (layer 2), the
> policy is favoured in 80/84 perturbed cells and collapses to parity only when the
> primary drug's kill rate is perturbed below ~5% (layer 3), and it beats the standard
> of care at equal information in paired runs (layer 4) — and none of that is evidence
> that a patient benefits. The contrast demonstrably depends on the ASSUMED pre-existing
> resistant fraction (layer 3, item 2), so the honest summary is that this model is
> policy-favourable across a perturbed parameter space, not that the policy works.

`outputs_real/validation_packet.txt` is generated by `danon/reporting.py:validation_packet()`
from the artifacts themselves, so no number in this document can drift from the code
that produced it.

---

## The day-365 outcome artifact and the UI

`danon/outcome_export.py` runs both arms on real geometry and exports **exact per-day
engine state** — sensitive burden, resistant burden, resistant fraction, per-day dose of
each drug — plus an **exact day-365 readout** per replicate and as medians. Nothing is
interpolated: every number is read from the engine at that day.

```bash
# regenerate the data and both views (engine run, ~50 s for 9 cells)
python -c "
from danon.realdata import prepare_brats_case
from danon.outcome_export import export_outcome_cards, render_dashboard_html
p = export_outcome_cards(cases, 'outputs_real/outcome_cards.json',
                         mirror_path='frontend/public/evotwin/outcome_cards.json')
render_dashboard_html(p, 'outputs_real/day365_dashboard.html')"

# re-render either view from the artifact alone (no engine run)
python -c "
from danon.outcome_export import load_cards, render_dashboard_html
render_dashboard_html(load_cards('outputs_real/outcome_cards.json'),
                      'outputs_real/day365_dashboard.html')"
```

Two views over the same artifact: the React app's **365-Day Outcome** tab
(`frontend/src/components/Day365Outcome.tsx`, reads `public/evotwin/outcome_cards.json`)
and `outputs_real/day365_dashboard.html`, a self-contained file with the data embedded
so it can be opened or attached anywhere.

Two flags are rendered rather than buried, because each would otherwise read as a
result. Across the nine real-geometry cells, both fire:

* **the median-PFS extension is a LOWER BOUND** when no replicate in the adaptive arm
  progressed inside the horizon (it eradicates the modelled tumour on several cells, so
  the extension cannot be measured — only bounded below);
* **the day-365 resistant fraction is not interpretable** when the median burden is
  below 0.1% of the seeded tumour, because at that magnitude the fraction is a ratio of
  two near-zero numbers (the post-eradication artifact this repo already refuses to
  report as a schedule step).
