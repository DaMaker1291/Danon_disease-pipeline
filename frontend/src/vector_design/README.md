# Vector design (AAV9-LAMP2B) — preserved, not routed

This folder is the **AAV9-LAMP2B vector-design study**: an earlier project in
this repository that designed a capsid, a competing lipid-nanoparticle
formulation, and the microfluidic rig that would make either. It shares a
codebase with the glioblastoma tumour twin, but it is a different disease and a
different product.

It is **not part of the app's navigation**. `src/nav.ts` routes only the tumour
twin. The components here are kept because they run — each renders a recorded
engine artifact and states its own provenance — but mixing a cardiac
gene-therapy design pipeline into an oncology tool made both harder to read, so
the section was moved out of the product rather than deleted.

If you re-route one of these views, it needs three things:

1. **An entry in `src/nav.ts`** (`TabId`, `GROUPS`, `TABS`) — the nav, the page
   header and the landing page all read that one file.
2. **The artifact loader's folder argument.** These views load from
   `public/vector_design/<key>.json`, not `public/artifacts/<key>.json`, so
   their `useArtifact` calls pass `'vector_design'` as the second argument.
   The mapping is `danon/app_artifacts.py::MIRROR_SUBDIR`, which raises if an
   artifact key has no folder assigned — the split is explicit, not implied.
3. **A provenance chip** in its `PageHeader` entry. Every one of these screens
   shows design-engine scores, which are the project's own scorers, not assays.

## What is in here

| file | renders |
|---|---|
| `CapsidViewer.tsx` | the deposited AAV9 capsid (PDB 3UX1) assembled from its own symmetry operators |
| `SequenceOutputPanel.tsx` | the VP1 design's substitutions and their scores |
| `PhaseMatrix.tsx` | the 24 screening gates and their pass/warn/fail composition |
| `GeneViewer.tsx` | the single-vector LAMP2B construct map |
| `ParetoFrontChart.tsx` | the NSGA-II front over six competing objectives |
| `ClinicalDashboard.tsx` | the modelled cardiomyocyte rescue (an ODE sweep, not a cohort) |
| `TranslationalGateway.tsx` | the regulatory gates that are still unmet |
| `LNPViewer.tsx` | the scored lipid-nanoparticle formulation |
| `MicrofluidicViewer.tsx`, `fluidics.ts` | Y-junction mixing hydraulics |
| `TelemetryPanel.tsx` | that flow telemetry as readouts |
| `StructureViewer.tsx` | ESM-based fitness/structure scoring over a sequence |
| `capsidGeometry.ts` | the shared geometry helpers for the three capsid views |

Python counterparts live in `danon/vector_design/`. Nothing in either folder
has been synthesised, expressed, or dosed; every gate is a scorer in this
repository, not an assay.
