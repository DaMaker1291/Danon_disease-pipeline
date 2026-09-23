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
2. **3D evolution viewer** — the final simulated fields (sensitive, resistant, drug A, drug B) rendered in-browser as an instanced voxel volume, plus a one-click **ParaView `.vti` export** of the same file.
3. **24-phase AAV pipeline, LNP assembly, microfluidics, clinical dashboards** — the earlier Danon-engine views, each rendering precomputed engine artifacts committed in `frontend/public/artifacts/`.

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
python -m pytest tests/ danon/multiagent/ -q    # 941 tests
python -m pytest tests/test_twin_plan.py -q     # twin planner tests
```

## Deployment

The site deploys to GitHub Pages automatically on push to `main` via `.github/workflows/deploy-pages.yml`. The deployed site renders all precomputed artifacts; **live MRI planning requires the local API** (the UI states this when not on localhost).

## Scientific honesty boundaries (enforced in code)

- The switch rule acts on the **simulated** resistant fraction, which is not clinically observable in real time.
- All growth/kill/diffusion constants are **ASSUMED** pending bench measurement; every derived number inherits that uncertainty.
- Headline metrics are **per-patient paired simulation differences**, not predictions for a real patient.
- Day precision is the simulation's daily resolution.
- The `measurement` layer refuses to report numbers it cannot derive from data — refusals are first-class, labelled outputs.
