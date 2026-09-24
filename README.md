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
4. **24-phase AAV pipeline, LNP assembly, microfluidics, clinical dashboards** — the earlier Danon-engine views, each rendering precomputed engine artifacts committed in `frontend/public/artifacts/`.

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
python -m pytest tests/ danon/multiagent/ -q    # 955 tests, ~6 min full suite
python -m pytest tests/test_recovery.py -q      # Standard 1 recovery suite, 46 tests ~37 s
python -m pytest tests/test_recovery.py::TestTransportRecovery -q   # ~2 s stencil check
python -c "from danon.claims import audit_claims; audit_claims()"   # 31/31 bound claims
```

## Verification matrix (live-measured 2026-09-23; artifacts in `outputs_real/`)

| layer | measured value | source |
|---|---|---|
| Full integration suite | **941 passed** | `pytest tests/ danon/multiagent/ -q` |
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
| Bound-claims audit | **31/31 verified** | `danon/claims.py::audit_claims` |
| Clinical datastreams (4 tests) | streams wired with identity guarantees; stream 3 measured on real t1ce: MTD PFS paired delta −1.17 d (18 runs); streams 1/2/4 refuse with live zero counts | `danon/clinical_twin.py::stream3_report`, `outputs_real/clinical_twin.txt` |

Standard 1 validates the *solver* on synthetic planted ground truth — no patients involved, and it must never be blended with the Standard 2 patient numbers. The dosing "policies" (`MTDPolicy`, `AdaptiveDosePolicy`, `CollateralSwitchPolicy`) are closed-form rules; **no reinforcement-learning agent exists or is claimed** (bound claim `no_rl_agent`). The PDE operators live in `danon/anisotropy.py` and `danon/evosteer.py`; `danon/imaging.py` is NIfTI intake and geometry only.
```

## Deployment

The site deploys to GitHub Pages automatically on push to `main` via `.github/workflows/deploy-pages.yml`. The deployed site renders all precomputed artifacts; **live MRI planning requires the local API** (the UI states this when not on localhost).

## Scientific honesty boundaries (enforced in code)

- The switch rule acts on the **simulated** resistant fraction, which is not clinically observable in real time.
- All growth/kill/diffusion constants are **ASSUMED** pending bench measurement; every derived number inherits that uncertainty.
- Headline metrics are **per-patient paired simulation differences**, not predictions for a real patient.
- Day precision is the simulation's daily resolution.
- The `measurement` layer refuses to report numbers it cannot derive from data — refusals are first-class, labelled outputs.
