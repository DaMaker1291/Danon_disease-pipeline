# EvoTwin-10X — Spatial Digital-Twin Oncology Platform

**Upload an MRI segmentation mask → get a simulated 365-day adaptive treatment plan, with clonal-evolution dynamics, a paired MTD control, and the final 3D fields exported for ParaView.**

> **Status:** research/engineering platform. Every number the app renders is a recorded output of the simulation engine in `danon/` or an explicitly labelled refusal — never a placeholder, never a fabricated patient prediction. The engine's constants are ASSUMED pending bench measurement, and the schedules it produces are simulation artifacts, not clinical advice.

---

## What this app does

1. **MRI Twin Planner** (front page) — upload a tumour segmentation mask (`.nii` / `.nii.gz`):
   - the mask is parsed by the repo's own NIfTI-1 reader (`danon/imaging.py`),
   - oversize grids are block-max-downsampled (tumour-preserving) to the 4M-voxel engine budget,
   - the daily-step spatial simulator (`danon/evosteer.py`) runs a **paired** trial: adaptive steering policy vs standard maximum-tolerated-dose control on identical geometry and seeds,
   - you get: geometry summary measured from your file, the executed dose schedule, daily burden curves, regime-change events, a bound toxicity reduction, the compiled schedule text, and a provenance table (measured vs assumed vs refused).
2. **3D evolution viewer** — the final simulated fields (sensitive, resistant, drug A, drug B, plus the derived tissue-class and per-voxel diffusion D(x) maps) rendered in-browser as an instanced voxel volume, plus a one-click **ParaView `.vti` export** of the same file. A dedicated **ParaView Gallery** tab renders the committed BraTS-2021 artifact with a color legend, live mm scale bar, PNG capture, and a byte-identical `.vti` download. See [`PARAVIEW.md`](PARAVIEW.md) for the corrected render walkthrough and the artifact in `outputs_real/brats_case01017_mtd_day120.vti`.
3. **365-day evolution & resistance player** — the whole paired trial as a movie: the **Evolution Player** tab scrubs/plays the engine's validated per-day spatial frames for both arms (adaptive switch vs MTD) on the same tumour and seed, with the resistant-fraction trajectories synced to the timeline. Frames export as a ParaView **`.pvd` time collection** (`danon.evolution_export.write_pvd_series`) so ParaView's play controls become the day axis.
4. **Data intake** — inspect a file the way the intake does: NIfTI masks, FASTA/FASTQ, or a TCIA availability manifest. Nothing is uploaded or stored.

The app routes **one study**. The earlier AAV9-LAMP2B vector-design section (24 design gates, capsid, LNP, microfluidics, cardiomyocyte dashboards) is preserved but no longer routed — see *Repository layout* below for where it lives and why.

## Running locally (full engine + UI)

```bash
# backend (FastAPI bridge over the danon engine)
pip install fastapi uvicorn python-multipart
uvicorn api_server:app --port 8000

# frontend
cd frontend && npm install && npm run dev
# open http://localhost:6006
```

With the backend up, the Twin Planner runs live planning on your mask (~4 s for a small mask, paired 5-replicate trial).

## Running the test suite

```bash
python -m pytest tests/ danon/multiagent/ -q    # full suite ~13 min; the count is derived, never quoted here
python -m pytest tests/test_recovery.py -q      # Standard 1 recovery suite, 46 tests ~37 s
python -m pytest tests/test_recovery.py::TestTransportRecovery -q   # ~2 s stencil check
python -c "from danon.claims import audit_claims; audit_claims()"   # 42/42 bound claims
python -m danon.reproduce            # the whole paired trial on the committed sample, no downloads
```

## Verification matrix (live-measured 2026-09-23; artifacts in `outputs_real/`)

| layer | measured value | source |
|---|---|---|
| Full integration suite | **0 failing** across `tests/` + `danon/multiagent/`; the count is computed live by `danon/claims.py::check_test_suite_is_counted_live` and not quoted here | `pytest tests/ danon/multiagent/ -q` |
| Standard 1 · transport operator | **R² = 0.999951** over 210,912 equations (D err 3.56%, λ err 1.25%) | `outputs_real/recovery_report.txt` |
| Standard 1 · planted Cox hazard | oracle C 0.6522 vs fitted 0.6519, shortfall **+0.0003** | same |
| Standard 2 · real TCGA-GBM cohort | OS **C = 0.656 ± 0.025** (n=595, 491 events, 104 censored); progression C = 0.534 | same |
| Standard 2 · permutation leakage audit | observed 0.6559 vs null band **0.468–0.536** (20 draws, null centre 0.506) | `outputs_real/audit_trace.txt` |
| High-ceiling · engine emulator | **holdout R² = 0.9211** (CI 0.8615–0.9536, n_test=48 untouched configs) — synthetic, never a patient accuracy | `outputs_real/high_ceiling.txt` |
| High-ceiling · MGMT status (real patients) | **AUC = 0.5387** (CI 0.485–0.596, n=396) — indistinguishable from chance, reported at its measured value | same |
| Covariate upgrades · KPS beside age/sex | HR=**0.9810**/point (3.3 SE, n=247) — real protective direction, delta_C=+0.0264 vs fold sd 0.0476: **no measurable held-out gain** | `outputs_real/covariate_upgrades.txt` |
| Covariate upgrades · MGMT binary / continuous beta | HR=0.8326 (delta_C=−0.0006, n=409) / HR=0.6627 (delta_C=+0.0006, n=137) — directionally consistent with trial literature, unresolvable in this cohort | same |
| Covariate upgrades · RT+temozolomide restriction | keeps **356/595**; cohort-membership delta −0.0112 inside fold sd 0.0247 — selection reported separately from signal | same |
| Covariate upgrades · all covariates stacked | pooled C **0.5785** (n=68) vs age-alone 0.6203 on the same kept subset — stacking unavailable-for-most covariates *costs* discrimination | same |
| MGMT β → resistance bridge | r0 = 0.75·(1−β); across the real 140-patient beta range MTD median PFS **144 → 168 days**; steering policy saturates at 365 d at every β (zero dynamic range) | `outputs_real/mgmt_resistance.txt` |
| Solver vs analytic solutions | Neumann eigenvector residual **3.43e-14** relative (float noise); closed-domain mass identity **3.47e-18**; the two transport branches agree at constant D to **2.75e-16**; observed spatial convergence order **1.9944** (second); forward Euler's signed clearance bias **+0.0843%/day** moves no endpoint by more than **0.7505%**, i.e. **7.53e-03** of the paired cohort's own seed spread (99.66%) | `danon/claims.py::check_solver_verified_against_analytic_solutions`, `tests/test_benchmarks.py`, `_clearance_bias_cohort.py` → `outputs_real/clearance_bias_cohort.json` |
| Runnable from a fresh clone | `data/sample_bids/` commits a **14 KB** synthetic phantom (3 NIfTI volumes + config + description, all byte-identical to `make_gbm_phantom`'s output); `python -m danon.reproduce` runs the real paired 365-day trial on it — **16 s** wall clock measured on this machine, one process: MTD median PFS **166 d** (progresses in 3/3, median burden 4,640), switching arm **365 d** (0/3 progress, median burden 16.8), paired median delta **+199 d**, 3/3 pairs differ — a phantom, so a pipeline demonstration and never a patient result | `danon/reproduce.py`, `tests/test_reproduce.py`, `outputs_real/sample_twin_report.json` |
| Bound-claims audit | **42/42 verified** | `danon/claims.py::audit_claims` |
| Stream 3 · measured DSC perfusion | 3 real UPENN-GBM DSC series downloaded and read (JPEG-2000 DICOM, 700–900 files each, TR 1.83–2.0 s): first-pass rCBV with the head-to-tail drift line removed (30–51% of the raw peak is drift); planted 3:1 CBV recovered to **0.9%**; transport field **15.3–21.6×**; final-burden effect **0.18–8.81%** (steering more sensitive); PFS censored in every arm | `danon/perfusion.py`, `outputs_real/perfusion_cohort.json` |
| Clinical datastreams (4 tests) | streams wired with identity guarantees; stream 3 measured on real t1ce (MTD PFS paired delta −1.17 d, 18 runs) and on real DSC series (rCBV field contrast 15.3–21.6×); stream 2 measures real whole-slide cellularity (636 nuclei/mm² on one TCGA-GBM slide), feeds it to the engine's density ceiling, and is measured there paired on real geometry — a scale under continuous MTD, a lever only where an absolute floor gates the adaptive switch — while refusing the Ki-67 ratio and any absolute cells/mm³; streams 1/2/4 refuse as inputs with live counts | `danon/clinical_twin.py::clinical_stream_census`, `danon/histology.py`, `_stream2_capacity_cohort.py`, `outputs_real/clinical_twin.txt` |
| Stream 4 as a comparator | `stream4_velocity_agreement` refuses without a measured pair (cites its own census: 0 cached / 41 retrievable) and scores r=1.0 for identical fields, lower when the growth voxel differs — a validation target, never an engine input | `danon/clinical_twin.py::stream4_velocity_agreement` |
| Stream-4 cohort resolution (design calc) | the CFB-GBM manifest's **168** usable-pair patients resolve `|ΔC| > 0.0219` (1 SE over 24 draws; fold yardstick 0.0610) versus 0.0247 on the full n=595 cohort — a floor to clear, not a measured gain | `outputs_real/stream4_resolution.json` |

Standard 1 validates the *solver* on synthetic planted ground truth — no patients involved, and it must never be blended with the Standard 2 patient numbers. The dosing "policies" (`MTDPolicy`, `AdaptiveDosePolicy`, `CollateralSwitchPolicy`) are closed-form rules; **no reinforcement-learning agent exists or is claimed** (bound claim `no_rl_agent`). The PDE operators live in `danon/anisotropy.py` and `danon/evosteer.py`; `danon/imaging.py` is NIfTI intake and geometry only.

## Repository layout

This repository holds one product and three studies. The split is now explicit in the tree, because it was not before:

| path | what it is |
|---|---|
| `danon/` | the **glioblastoma tumour twin**: spatial engine, policy comparison, evaluation against real TCGA cohorts, claims registry. This is what the app runs. |
| `danon/vector_design/` | the **AAV9-LAMP2B vector-design study** (capsid, payload, LNP, microfluidics rig) plus the vascular-delivery biology it reuses. A different disease, reachable only by explicit import — `danon/__init__.py` does not pull it in. |
| `danon/multiagent/` | the cardiovascular/ischaemic-heart-disease deployment layer. Not routed by this app. |
| `frontend/src/` | the tumour-twin UI: MRI → plan, the 365-day evolution player, the day-365 state, data intake, the ParaView gallery, the claim registry. |
| `frontend/src/vector_design/` | the vector-design UI, kept runnable and documented but **deliberately not routed**; its README states what re-routing a view would require. |
| `frontend/public/artifacts/` · `frontend/public/vector_design/` | the recorded engine outputs each view renders, split by the same boundary (see `danon/app_artifacts.py::MIRROR_SUBDIR`). |

`import danon` loads no vector-design module. Names that used to be re-exported from the top level (the AAV9 structure, the rig telemetry, the blood-to-cure pipeline) are still reachable as `danon.<Name>` through a lazy resolver, but new code should import `danon.vector_design` directly.

## Deployment

The site deploys to GitHub Pages automatically on push to `main` via `.github/workflows/deploy-pages.yml`. The deployed site renders all precomputed artifacts; **live MRI planning requires the local API** (the UI states this when not on localhost).

## Scientific honesty boundaries (enforced in code)

- The switch rule acts on the **simulated** resistant fraction, which is not clinically observable in real time.
- All growth/kill/diffusion constants are **ASSUMED** pending bench measurement; every derived number inherits that uncertainty.
- Headline metrics are **per-patient paired simulation differences**, not predictions for a real patient.
- Day precision is the simulation's daily resolution.
- The `measurement` layer refuses to report numbers it cannot derive from data — refusals are first-class, labelled outputs.
