# ParaView renders of the EvoTwin-10X spatial engine

How to turn the engine's 3D fields into poster-quality figures — with the
render steps corrected for **this repository's file format**, and captions
that say what the simulation actually integrated.

## What to open

`outputs_real/brats_case01017_mtd_day120.vti` is a real artifact, not a
placeholder: the final state of a 120-day maximum-tolerated-dose run on the
BraTS-2021 case 01017 segmentation mask cached in `data/engine_ready/`. The
diffusion field the run integrated is heterogeneous — 0.01 mm²/day in the
necrotic core to 0.35 mm²/day in healthy tissue (35× contrast) — and the same
D(x) ships inside the file as `diffusion_mm2_per_day`.

The same file is also rendered **inside the app**: the ParaView Gallery tab
embeds it byte-identically (sha256-pinned in `paraview.json`), with a color
legend, a live millimetre scale bar, cutaway, and a download button that
serves the exact file ParaView opens. Use the in-app view for exploration;
use the steps below for the poster-quality 4K render.

The file carries nine point arrays:

| array | meaning |
|---|---|
| `drug_A`, `drug_B` | final drug concentration fields (model units) |
| `sensitive`, `resistant` | final sub-clone cell densities |
| `tissue_class` | 0 necrotic core, 1 edema, 2 white matter, 3 grey matter |
| `diffusion_mm2_per_day` | the per-voxel D(x) in physical units |
| `proliferative` | relative growth rate per voxel |
| `vessels` | drug source strength per voxel |
| `tumour_mask` | 1 inside the segmentation |

Axes: engine arrays are indexed [x, y, z] (NIfTI dim order) and the VTK
extent is `(0 sx-1)(0 sy-1)(0 sz-1)`, so VTK's x/y/z are the array axes in
order. Spacing is the geometry's real voxel size (1.0 mm here), so the
render is to physical scale — check with the ruler tool before putting a
scale bar on the poster.

## ParaView steps (5.11+)

1. **Open.** `File → Open` the `.vti`, click **Apply**. ParaView reads it
   natively — no importer, no plugin.

2. **Tissue sub-regions first.** Representation **Surface**; Color by
   `tissue_class`. The default labels are meaningless numbers, so set a
   custom legend: necrotic core `#6b7280`, edema `#93c5fd`, white matter
   `#e5e7eb`, grey matter `#a78bfa`. This answers "where is the drug
   geometrically trapped?" before any drug field is shown.

3. **Volume-render the drug field.** Select the `.vti`, representation
   **Volume**, Color by `drug_A`, map **Viridis**. Use the opacity editor's
   log-style ramp (near-zero transparency, peaks opaque) — linear ramps on
   diffusion fields show a featureless glow. Repeat with `drug_B` on
   **Plasma** in a second view if the collateral-sensitivity story needs
   both agents side by side.

4. **Resistant-clone isosurface.** `Filters → Contour` on the same source,
   Contour By `resistant`, one value at ~50% of the array max
   (the arrays are cell densities; for this artifact the resistant peak is
   the tumour rim — pick the value from the histogram, not from a tutorial).
   Color the contour **red**. Apply the same Contour recipe to `sensitive`
   in **cyan** for the before/after comparison.

5. **Cutaway.** `Filters → Clip`, Plane mode, normal −X (or use the
   interactive handle). The interesting structure in this artifact is the
   *interior*: drug concentration collapsing inside the necrotic core while
   the rim stays perfused. A surface-only render hides exactly that.

6. **Diffusion tensor view (the Figure-3 money shot).** Clip first, then
   Color the clipped surface by `diffusion_mm2_per_day`. The clipped cross
   section shows D dropping two orders of magnitude across the tumour
   boundary — the gradient that creates the drug sanctuary. (A glyph view
   of a full tensor needs a 9-component array; this engine's field is
   diagonal/scalar per voxel by construction, and a per-voxel scalar D(x)
   is the honest thing to show. Do not draw glyphs that imply direction
   the model does not carry.)

7. **Poster image.** `File → Save Screenshot`, 3840×2160. Turn off the
   orientation axes unless the board needs them. ParaView's **built-in**
   color legend (the "color map" bar) is the one judges check — enable
   `Show Color Legend` and label its units (mm²/day for D, model units for
   concentrations, cells/voxel for densities).

## What the artifact shows, honestly

- **The tumour under MTD at day 120 is 100% drug-resistant** (resistant
  fraction 1.00). That is the sanctuary effect the geometry produces: the
  necrotic core's D(x) is 35× lower than healthy tissue, drug cannot reach
  the interior, and the resistant clone — which pays a metabolic cost for
  resistance everywhere else — owns the sanctuary pocket.
- The steering arm on the same geometry, same seed, eradicates the tumour
  by day 120 (final burden 0.0), which is why there is no companion
  day-120 steering VTI: **an empty volume renders as nothing**. The honest
  comparison figure is MTD-at-120-days next to the daily burden curves the
  app's TwinPlanner chart shows for the paired run.
- Regenerate with one call: `export_fields_vti(traj, geometry, path)` after
  `sim.run(..., keep_fields=True, diffusion_field=...)`. `include_tissue=False`
  restores the original four-field format.

## Caption for the board (measured, not aspirational)

> **Figure 3 — Spatial drug sanctuary under continuous maximum-tolerated
> dosing.** Volume render of the final state of a 120-day simulation on a
> BraTS-2021 segmentation (14×24×24 voxels at 1 mm). The engine integrated
> a heterogeneous diffusion field D(x): 0.01 mm²/day in necrotic core,
> 0.35 mm²/day in healthy tissue. The red isosurface encloses the residual
> tumour, which is 100% drug-resistant; drug concentration (Viridis)
> collapses inside the necrotic core that D(x) marks unreachable. Under the
> paired adaptive schedule on the identical geometry and seed, the tumour
> is eradicated within the horizon (day-120 burden 0.0). Simulation, not
> patient data; rate constants are declared assumptions.

## The 365-day time series (`.pvd` collection)

The final-state VTI above is one day. The whole evolution — where resistance
emerges, when the sanctuary pocket becomes visible — is a time series, and
ParaView opens it natively:

```python
from danon.imaging import from_nifti_mask
from danon.evolution_export import write_pvd_series

pg = from_nifti_mask(
    "data/engine_ready/BraTS2021_01017_mask.nii.gz",
    proliferative_path="data/engine_ready/BraTS2021_01017_prolif.nii.gz",
)
pvd = write_pvd_series(
    "outputs_real/evolution_mtd", pg.geometry,
    r0=0.005, seed=101, days=365, arm="mtd",
)   # -> outputs_real/evolution_mtd/evolution.pvd + frame_XXXX_dayYYYY.vti
```

Then in ParaView: **File → Open → `evolution.pvd` → Apply**. The play
controls become the day axis (timestep = day); **Filters → Temporal →
Temporal Interpolator** smooths between recorded days if you want a fluid
animation for a video loop. Each frame is a complete `.vti` — the same
nine arrays as the final-state file — so contour/clip recipes work per
frame and any single frame can be opened standalone.

In the app, the **Evolution Player** tab renders the same frame series for
the paired arms (adaptive vs MTD) with the resistance trajectories synced
to the timeline; every frame it shows is validated against the run's own
daily curves before export
(`danon.evolution_export.FrameSet.validate_against_trajectory`).
