"""Digital-twin research simulation: MRI mask in, paired 365-day run out.

WHY THIS MODULE EXISTS
----------------------
The engine pieces already existed and are tested individually: NIfTI mask
intake (danon.imaging), the daily-step spatial simulator (danon.evosteer),
the schedule compiler with its paired MTD control (danon.schedule), and the
ParaView VTI exporter (danon.schedule.export_fields_vti). What did not exist
is the one function a clinical-facing UI needs: take an uploaded segmentation
mask and return the complete planning package in one call, with the honesty
labels attached at every field.

WHAT IS MEASURED VS ASSUMED (and kept visible in every payload)
---------------------------------------------------------------
- measured-from-your-file : the mask geometry itself (tumour voxel count,
  bounding box, volume in mm^3 using the file's own voxel sizes), the
  proliferation proxy derived from the mask's boundary distance transform,
  and every simulated trajectory (burden curves, PFS days, dose exposure).
- assumed                 : all growth/kill/diffusion constants (the engine's
  calibrated defaults -- ASSUMED pending bench measurement), the initial
  resistant fraction r0 unless the caller supplies provenance, and the drug
  identity (generic "A"/"B" with saturating kinetics, NOT temozolomide).
- refused                 : any claim that this schedule predicts the
  outcome of a real patient. The r0 the planner uses is NOT clinically
  observable (danon.schedule's own disclaimer), and every number here is a
  property of the simulation, labelled as such.

The r0 default (1%) is the engine's historical default. A caller with a real
r0 source must pass it; the report echoes whatever source string it was
given, so an unprovenanced default can never masquerade as a measurement.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.evosteer import (
    CollateralSwitchPolicy,
    MTDPolicy,
    TumorEvolutionSimulator,
    TumorGeometry,
)
from danon.imaging import PatientGeometry, from_nifti_mask
from danon.schedule import (
    DosingScheduleReport,
    PatientIntake,
    compile_dosing_schedule,
    export_fields_vti,
    format_schedule_text,
)

# The engine budget from danon.imaging; kept as an import so the two modules
# cannot drift apart on the one number that gates uploads.
from danon.imaging import MAX_TOTAL_VOXELS

DEFAULT_R0 = 0.01
DEFAULT_R0_SOURCE = (
    "engine default 1% (ASSUMED; no assay provided with the mask)"
)


@dataclass
class TwinPlanResult:
    """Everything the UI renders for one uploaded mask."""

    geometry_summary: Dict[str, Any]
    schedule_text: str
    daily_curves: Dict[str, List[Optional[float]]]
    paired_metrics: Dict[str, Any]
    events: List[Dict[str, Any]]
    notes: List[str]
    provenance: Dict[str, str]
    vti_path: Optional[str] = None
    vti_bytes: Optional[bytes] = None

    def to_json_dict(self) -> Dict[str, Any]:
        """JSON-safe payload (VTI binary is transported separately)."""
        return {
            "geometry": self.geometry_summary,
            "schedule_text": self.schedule_text,
            "daily_curves": self.daily_curves,
            "paired_metrics": self.paired_metrics,
            "events": self.events,
            "notes": self.notes,
            "provenance": self.provenance,
        }


def plan_from_mask(
    mask_path: str,
    *,
    r0: float = DEFAULT_R0,
    r0_source: str = DEFAULT_R0_SOURCE,
    days: int = 365,
    n_replicates: int = 5,
    seed0: int = 7,
    export_vti: bool = True,
    geometry_note: Optional[str] = None,
) -> TwinPlanResult:
    """One call: segmentation mask -> complete treatment-planning package.

    Runs the steering policy and the MTD control PAIRED (same geometry, same
    seeds) via compile_dosing_schedule, extracts the reference run's daily
    curves for plotting, and exports the final spatial fields as a ParaView
    VTI file. `days` and `n_replicates` are exposed but bounded: a UI must
    not be able to request a 30-year horizon or a 100-replicate average and
    wait for it.
    """
    if not 1 <= days <= 365:
        raise ValueError(f"days must be in [1, 365], got {days}")
    if not 1 <= n_replicates <= 10:
        raise ValueError(
            f"n_replicates must be in [1, 10] (UI budget), got {n_replicates}"
        )

    patient_geom: PatientGeometry = from_nifti_mask(mask_path)
    geom = patient_geom.geometry

    note = geometry_note or (
        f"uploaded segmentation mask {Path(mask_path).name} "
        f"(voxel {patient_geom.voxel_mm:.2f} mm; vasculature: "
        f"{patient_geom.vasculature_source})"
    )
    intake = PatientIntake(r0=r0, r0_source=r0_source, geometry_note=note)

    report: DosingScheduleReport = compile_dosing_schedule(
        geom,
        intake,
        days=days,
        n_replicates=n_replicates,
        seed0=seed0,
    )

    # Re-run the reference seed with keep_fields for (a) the daily curves at
    # the exact parameters the schedule text describes and (b) the VTI export.
    sim = TumorEvolutionSimulator(geom)
    ref = sim.run(
        CollateralSwitchPolicy(),
        days=days,
        initial_resistant_fraction=r0,
        seed=seed0,
        keep_fields=True,
    )

    mtd = TumorEvolutionSimulator(geom).run(
        MTDPolicy(),
        days=days,
        initial_resistant_fraction=r0,
        seed=seed0,
    )

    daily_curves = {
        "days": list(range(days)),
        "burden_S": [float(v) for v in ref.burden_S],
        "burden_R": [float(v) for v in ref.burden_R],
        "dose_A": [float(v) for v in ref.dose_A],
        "dose_B": [float(v) for v in ref.dose_B],
        "mtd_burden_total": [
            float(s + r) for s, r in zip(mtd.burden_S, mtd.burden_R)
        ],
    }

    paired = {
        "policy_median_pfs_days": report.policy_median_pfs,
        "mtd_median_pfs_days": report.paired_mtd_median_pfs,
        "pfs_extension_median": report.pfs_extension_median,
        "toxicity_reduction_median": report.toxicity_reduction_median,
        "n_replicates": report.n_replicates,
        "policy_pfs_replicates": list(report.pfs_policy_replicates),
        "mtd_pfs_replicates": list(report.pfs_mtd_replicates),
        "ref_pfs_day": ref.pfs_day,
        "ref_progressed": ref.progressed,
        "ref_toxicity": ref.toxicity,
        "mtd_toxicity": mtd.toxicity,
        "ref_final_resistant_fraction": ref.final_resistant_fraction,
    }

    events = [
        {
            "day": ev.day,
            "action": ev.action,
            "resistant_fraction": ev.resistant_fraction,
            "burden": ev.burden,
        }
        for ev in report.events
    ]

    mask = geom.tumor_mask
    vox_idx = np.argwhere(mask)
    bbox = {
        "min": vox_idx.min(axis=0).tolist(),
        "max": vox_idx.max(axis=0).tolist(),
    }
    voxel_vol = patient_geom.voxel_mm**3
    geometry_summary = {
        "mask_path": patient_geom.mask_path,
        "grid_shape_xyz": [int(s) for s in mask.shape],
        "total_voxels": int(mask.size),
        "tumour_voxels": int(patient_geom.n_tumour_voxels),
        "tumour_volume_mm3": float(patient_geom.n_tumour_voxels * voxel_vol),
        "voxel_mm": patient_geom.voxel_mm,
        "bounding_box_xyz": bbox,
        "vasculature_source": patient_geom.vasculature_source,
        "proliferative_source": patient_geom.proliferative_source,
        "engine_budget_voxels": MAX_TOTAL_VOXELS,
        "budget_fraction": float(mask.size / MAX_TOTAL_VOXELS),
    }

    provenance = {
        "r0": r0_source,
        "r0_value": f"{r0:.4f}",
        "geometry": note,
        "constants": "engine calibrated defaults -- ASSUMED pending bench measurement",
        "drugs": "generic A/B saturating kinetics, NOT named clinical agents",
        "predictions": "simulation outputs on the uploaded mask; NOT patient outcome predictions",
    }

    result = TwinPlanResult(
        geometry_summary=geometry_summary,
        schedule_text=format_schedule_text(report),
        daily_curves=daily_curves,
        paired_metrics=paired,
        events=events,
        notes=list(report.notes),
        provenance=provenance,
    )

    if export_vti:
        with tempfile.TemporaryDirectory(prefix="twinplan_") as td:
            vti = Path(td) / "twin_fields.vti"
            export_fields_vti(ref, geom, str(vti))
            result.vti_bytes = vti.read_bytes()
    return result


def downsample_mask_to_budget(mask: np.ndarray, budget: int = MAX_TOTAL_VOXELS):
    """Factor-2 block-downsample until the grid fits the engine budget.

    A 3D NIfTI from a clinical series is routinely 512x512x~150 -- two orders
    of magnitude above the engine budget. Max-pooling by 2 per axis keeps
    every tumour-positive voxel (a max over a block containing tumour is
    tumour), which is the conservative choice for a planning geometry: the
    downsampled tumour can only be equal-or-larger, never smaller.
    Returns (downsampled_mask, factors) so the caller can report the exact
    transform applied to the patient's file.
    """
    import math

    factors = []
    work = mask.astype(bool)
    while work.size > budget:
        nz, ny, nx = work.shape
        crop = (
            nz - (nz % 2),
            ny - (ny % 2),
            nx - (nx % 2),
        )
        trimmed = work[: crop[0], : crop[1], : crop[2]]
        work = (
            trimmed[0::2, 0::2, 0::2]
            | trimmed[1::2, 0::2, 0::2]
            | trimmed[0::2, 1::2, 0::2]
            | trimmed[0::2, 0::2, 1::2]
            | trimmed[1::2, 1::2, 0::2]
            | trimmed[1::2, 0::2, 1::2]
            | trimmed[0::2, 1::2, 1::2]
            | trimmed[1::2, 1::2, 1::2]
        )
        factors.append(2)
        if math.prod(work.shape) == work.size and work.size <= budget:
            break
        if len(factors) > 6:
            raise ValueError(
                "mask still exceeds the engine budget after six halvings; "
                "refusing to degrade it further"
            )
    return work, factors


def derived_voxel_sizes(
    voxel_sizes: Sequence[float], factors: Sequence[int]
) -> Tuple[float, ...]:
    """The spacing a downsampled volume must carry in its OWN header.

    One factor-2 halving means each derived voxel spans two source voxels per
    axis, so the derived header's spacing is the source spacing times two per
    halving. Writing the source spacing back into the derived file instead
    understates the derived voxel volume -- and every physical volume taken
    from it -- by 8x per halving: the exact defect `danon.grid_artifact`
    records against the committed derived volumes (a 1 mm header over 48 mm^3
    voxels), and the reason `danon.realdata` builds its decimated spacing as
    `source_vox[i] * factors[i]`. The engine's own constants are voxel-unit,
    so this changes what the geometry REPORT says, not what the engine solves.
    """
    per_axis = 1
    for f in factors:
        if int(f) <= 0:
            raise ValueError(f"downsample factors must be positive, got {f}")
        per_axis *= int(f)
    return tuple(float(v) * per_axis for v in voxel_sizes)
