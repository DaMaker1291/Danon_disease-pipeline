"""
Adaptive steering schedule compiler: the artifact layer between the engine
and a clinic-shaped deliverable.

WHAT THIS ADDS OVER danon.evosteer / danon.evotwin
--------------------------------------------------
The engine runs policies and returns trajectories. This module compiles a
run into the two outputs the EvoTwin pipeline promises:

Output A (spatial fields): export_fields_vti() writes the final drug
    concentrations and clonal densities as a ParaView-loadable .vti
    ImageData file. No VR, no OpenXR claims -- a 3D drug gradient renders
    identically in any viewer, and ParaView is free.

Output B (the schedule): compile_dosing_schedule() turns a policy run
    into a dosing report with the ACTUAL switch days extracted from the
    simulated dose logs (never asserted), the resistant fraction at each
    switch, and paired quantitative metrics (PFS extension and toxicity
    reduction vs MTD on the SAME geometry and seeds). Headline numbers are
    computed here per-patient; they are never quoted from an abstract.

poster_summary() renders the pitch-claims-vs-measured-artifact table for
    the poster: the abstract's claims are kept VERBATIM as the comparison
    column (POSTER_PITCH_CLAIMS), and the measured column is computed from
    the report -- PFS is rendered as a censored lower bound wherever the
    policy arm reaches the horizon without progressing, never as a bare
    point estimate.

THE OBSERVABILITY LIMIT THAT SHAPES THIS MODULE
-----------------------------------------------
A schedule that says "switch at day 42 when the resistant fraction hits
15%" contains a quantity NO CLINIC CAN MEASURE in real time: sub-clone
density inside a tumour. The switch rule is a function of the simulation
state, not of any observable. Operationalising this schedule requires a
validated surrogate biomarker (serial biopsy-derived clone fractions,
ctDNA branch, or imaging proxies) -- and the day-42 precision is an
artifact of the simulation's daily resolution, not a claim about when a
real switch should occur. The report states this in its notes and the
formatted text, because a schedule that hides this limit is the kind of
over-precision that gets projects dismantled in judging.

INPUTS -- what is required, what is refused
-------------------------------------------
PatientIntake demands the initial resistant fraction WITH a provenance
string (where the number came from: scRNA-seq clone fractions, a measured
batch, etc.). An intake without provenance is refused: a schedule computed
for a fabricated patient is exactly the failure mode this repo exists to
prevent. Geometry comes from a TumorGeometry; the synthetic phantom is the
default and is labelled as NOT patient MRI (segmentation is an upstream
stage, see danon.evosteer SCOPE).

All pharmacology/growth constants remain ASSUMED pending the tier-3 bench
campaign (danon.validation); every metric here inherits that uncertainty.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.anisotropy import (
    DiffusionField,
    tissue_classes_from_geometry,
)
from danon.cure import EvidenceStatus
from danon.evosteer import (
    ADAPTIVE_TTP_RATIO_ANCHOR,
    AdaptiveDosePolicy,
    BasePolicy,
    CloneParams,
    CollateralSwitchPolicy,
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    DrugParams,
    ImmuneParams,
    MTDPolicy,
    SimulationTrajectory,
    TumorEvolutionSimulator,
    TumorGeometry,
    threshold_sweep,
)

# ---------------------------------------------------------------------------
# Patient intake: provenance-required inputs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatientIntake:
    """The measured inputs a schedule may be computed for.

    r0        : initial drug-resistant fraction, in (0, 1). REQUIRED with
                source provenance -- this is the one quantity single-cell
                genomics would actually provide.
    r0_source : non-empty provenance string (assay, batch, date). A schedule
                for an unprovenanced number is refused.
    geometry_note : what the geometry represents. The phantom default is
                labelled honestly; a real segmentation mask would carry its
                model/batch here.
    """

    r0: float
    r0_source: str
    geometry_note: str = "synthetic GBM phantom (NOT patient MRI)"

    def __post_init__(self) -> None:
        if not 0.0 < self.r0 < 1.0:
            raise ValueError(f"r0 must be in (0, 1), got {self.r0}")
        if not self.r0_source.strip():
            raise ValueError(
                "r0_source is required: a schedule computed for an "
                "unprovenanced resistant fraction is a fabricated patient"
            )
        if not self.geometry_note.strip():
            raise ValueError("geometry_note must describe the geometry's origin")


# ---------------------------------------------------------------------------
# Schedule-event extraction from a trajectory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScheduleEvent:
    """One regime change in the executed dose schedule."""

    day: int
    action: str                 # e.g. "switch A->B", "switch B->A", "dose on", "withdraw"
    dose_A: float
    dose_B: float
    resistant_fraction: float
    burden: float


def extract_schedule_events(traj: SimulationTrajectory) -> List[ScheduleEvent]:
    """Extract the ACTUAL regime changes from a run's dose logs.

    The switch days are read off the executed (dose_A, dose_B) sequence --
    they are measured outputs of the simulation, never asserted targets.
    The resistant fraction at each event comes from the daily burden
    arrays, which is the same quantity the policy's threshold acts on.
    """
    if len(traj.dose_A) == 0:
        return []
    events: List[ScheduleEvent] = []
    prev = (traj.dose_A[0], traj.dose_B[0])
    for day in range(1, len(traj.dose_A)):
        cur = (traj.dose_A[day], traj.dose_B[day])
        if cur != prev:
            burden = float(traj.burden_S[day] + traj.burden_R[day])
            r_frac = (
                float(traj.burden_R[day] / burden) if burden > 0 else 0.0
            )
            if prev == (0.0, 0.0):
                action = "dose on"
            elif cur == (0.0, 0.0):
                action = "withdraw"
            elif cur[1] > prev[1]:
                action = "switch A->B"
            elif cur[0] > prev[0]:
                action = "switch B->A"
            else:
                action = "dose change"
            events.append(
                ScheduleEvent(
                    day=day,
                    action=action,
                    dose_A=float(cur[0]),
                    dose_B=float(cur[1]),
                    resistant_fraction=r_frac,
                    burden=burden,
                )
            )
            prev = cur
    return events


# ---------------------------------------------------------------------------
# The schedule report
# ---------------------------------------------------------------------------


@dataclass
class DosingScheduleReport:
    patient: PatientIntake
    policy: str
    horizon_days: int
    events: List[ScheduleEvent]
    dose_A_days: int
    dose_B_days: int
    off_days: int
    progressed: bool
    pfs_day: Optional[int]
    paired_mtd_median_pfs: float
    policy_median_pfs: float
    pfs_extension_median: Optional[float]      # fraction, across paired replicates
    toxicity_reduction_median: Optional[float]  # fraction vs MTD
    n_replicates: int
    threshold_band: Optional[Tuple[float, float]] = None
    notes: List[str] = field(default_factory=list)
    # Per-replicate paired series, so KM/statistical plots can be drawn
    # without re-running the simulation.
    pfs_policy_replicates: List[float] = field(default_factory=list)
    pfs_mtd_replicates: List[float] = field(default_factory=list)
    toxicity_reduction_replicates: List[float] = field(default_factory=list)


def compile_dosing_schedule(
    geometry: TumorGeometry,
    intake: PatientIntake,
    policy: Optional[BasePolicy] = None,
    *,
    days: int = 365,
    n_replicates: int = 5,
    seed0: int = 7,
    immune: Optional[ImmuneParams] = None,
    clones: CloneParams = DEFAULT_CLONES,
    drugs: DrugParams = DEFAULT_DRUGS,
    sweep_thetas: Optional[Sequence[float]] = None,
) -> DosingScheduleReport:
    """Compile a policy run into the dosing-schedule deliverable.

    The policy arm and an MTD control are run PAIRED (same geometry, same
    seeds) so the PFS-extension and toxicity numbers are per-patient
    computed differences, not literature quotations. Optionally runs a
    small switch-threshold sweep and reports the best BAND (a single-number
    optimum from a simulation sweep is over-precision).
    """
    policy = policy if policy is not None else CollateralSwitchPolicy()
    sim = TumorEvolutionSimulator(geometry, clones=clones, drugs=drugs)

    def _one(pol: BasePolicy, seed: int) -> SimulationTrajectory:
        return sim.run(
            pol,
            days=days,
            initial_resistant_fraction=intake.r0,
            seed=seed,
            immune=immune,
        )

    # Reference (full-detail) run for the event extraction.
    ref = _one(policy, seed0)
    events = extract_schedule_events(ref)
    eradication_day = next(
        (
            d
            for d in range(len(ref.burden_S))
            if float(ref.burden_S[d] + ref.burden_R[d]) < 1e-6
        ),
        None,
    )
    dose_A_days = int(np.sum(ref.dose_A > 0))
    dose_B_days = int(np.sum(ref.dose_B > 0))
    off_days = int(days - dose_A_days - dose_B_days)

    # Paired replicates for the headline metrics.
    policy_pfs: List[float] = []
    mtd_pfs: List[float] = []
    ext: List[float] = []
    tox_red: List[float] = []
    for rep in range(n_replicates):
        seed = seed0 + rep
        tr_p = _one(policy, seed)
        tr_m = _one(MTDPolicy(), seed)
        policy_pfs.append(tr_p.pfs_value(days))
        mtd_pfs.append(tr_m.pfs_value(days))
        if tr_m.toxicity > 0:
            tox_red.append(1.0 - tr_p.toxicity / tr_m.toxicity)
        if tr_m.pfs_day is not None:
            ext.append((tr_p.pfs_value(days) - tr_m.pfs_value(days)) / tr_m.pfs_value(days))

    policy_median = float(np.median(policy_pfs))
    mtd_median = float(np.median(mtd_pfs))
    ext_median = float(np.median(ext)) if ext else None
    tox_median = float(np.median(tox_red)) if tox_red else None

    threshold_band: Optional[Tuple[float, float]] = None
    notes = [
        "the switch rule acts on the simulated resistant fraction, which is "
        "NOT clinically observable in real time; operationalising this "
        "schedule requires a validated surrogate biomarker (serial biopsy "
        "clone fractions, ctDNA, or imaging), not the simulation itself",
        f"day precision is the simulation's daily resolution, not a claim "
        f"about when a real switch should occur",
        "PFS is the RANO-fraction burden endpoint of the simulator, not a "
        "clinical measurement",
        "all growth/kill/diffusion constants are ASSUMED pending the tier-3 "
        "bench campaign (danon.validation); every number here inherits that "
        "uncertainty",
        f"headline metrics are computed per-patient against a paired MTD "
        f"control; the mCRPC anchor ratio is {ADAPTIVE_TTP_RATIO_ANCHOR:.2f} "
        f"and is a dynamics-class context, not this patient's prediction",
    ]
    if intake.geometry_note.startswith("synthetic"):
        notes.append(
            f"geometry: {intake.geometry_note} -- schedule metrics are "
            "properties of the phantom, not of any patient"
        )
    if eradication_day is not None:
        chatter = sum(1 for ev in events if ev.day > eradication_day)
        if chatter:
            notes.append(
                f"{chatter} regime changes occur AFTER eradication (day "
                f"{eradication_day}): at near-zero burden the resistant "
                "fraction is noise-dominated and threshold policies "
                "chatter between drugs -- events past this point are a "
                "numeric artifact, not interpretable schedule steps"
            )
    if sweep_thetas is not None and len(sweep_thetas) >= 2:
        sweep = threshold_sweep(
            geometry,
            thetas=tuple(sweep_thetas),
            days=days,
            n_replicates=max(2, min(3, n_replicates)),
            seed0=seed0 + 100,
            clones=clones,
            drugs=drugs,
            include_no_collateral=False,
        )
        threshold_band = sweep.best_band
        notes.append(
            f"switch-threshold band {sweep.best_band[0]:.2f}-{sweep.best_band[1]:.2f} "
            "reported as a BAND (within 5% of best median PFS): a single-number "
            "optimum from a simulation sweep is over-precision"
        )

    return DosingScheduleReport(
        patient=intake,
        policy=policy.describe(),
        horizon_days=days,
        events=events,
        dose_A_days=dose_A_days,
        dose_B_days=dose_B_days,
        off_days=off_days,
        progressed=ref.progressed,
        pfs_day=ref.pfs_day,
        paired_mtd_median_pfs=mtd_median,
        policy_median_pfs=policy_median,
        pfs_extension_median=ext_median,
        toxicity_reduction_median=tox_median,
        n_replicates=n_replicates,
        threshold_band=threshold_band,
        notes=notes,
        pfs_policy_replicates=list(policy_pfs),
        pfs_mtd_replicates=list(mtd_pfs),
        toxicity_reduction_replicates=list(tox_red),
    )


def format_schedule_text(report: DosingScheduleReport) -> str:
    """Render the clinic-shaped report. Every number printed is a measured
    output of the simulation; no target values are ever inserted."""
    lines = []
    bar = "=" * 74
    lines.append(bar)
    lines.append("ADAPTIVE STEERING SCHEDULE (simulation-derived)")
    lines.append(bar)
    lines.append(f"patient intake : r0 = {report.patient.r0:.4f} "
                 f"(source: {report.patient.r0_source})")
    lines.append(f"geometry       : {report.patient.geometry_note}")
    lines.append(f"policy         : {report.policy}")
    lines.append(f"horizon        : {report.horizon_days} d")
    lines.append(
        f"dose exposure  : A on {report.dose_A_days} d, B on {report.dose_B_days} d, "
        f"off {report.off_days} d"
    )
    lines.append("")
    lines.append("Regime changes (extracted from the executed dose logs):")
    if report.events:
        for ev in report.events:
            lines.append(
                f"  day {ev.day:>3}: {ev.action:<12} at R = "
                f"{ev.resistant_fraction:6.1%}, burden = {ev.burden:8.1f}"
            )
    else:
        lines.append("  none -- the policy never changed regime")
    lines.append("")
    lines.append(
        f"paired metrics ({report.n_replicates} replicates, same seeds as MTD control):"
    )
    lines.append(
        f"  median PFS  : policy {report.policy_median_pfs:.0f} d vs MTD "
        f"{report.paired_mtd_median_pfs:.0f} d"
    )
    if report.pfs_extension_median is not None:
        lines.append(
            f"  PFS extension: {report.pfs_extension_median:+.1%} (capped at the "
            f"{report.horizon_days} d horizon where arms are censored)"
        )
    if report.toxicity_reduction_median is not None:
        lines.append(
            f"  toxicity reduction vs MTD: {report.toxicity_reduction_median:.1%}"
        )
    if report.threshold_band is not None:
        lines.append(
            f"  switch-threshold band: {report.threshold_band[0]:.2f}-"
            f"{report.threshold_band[1]:.2f}"
        )
    lines.append("")
    lines.append("Limits (load-bearing; a schedule without these is over-precision):")
    for n in report.notes:
        lines.append(f"  - {n}")
    lines.append(bar)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Poster summary: pitch claims vs measured artifact
# ---------------------------------------------------------------------------


# The original project abstract's headline claims, kept VERBATIM so the
# comparison is always against what was actually promised, never a softened
# retelling. These are CLAIMS, not baselines -- the measured column is the
# only column computed from data.
POSTER_PITCH_CLAIMS: Dict[str, str] = {
    "pfs_extension": "+38% progression-free survival extension",
    "toxicity_reduction": "~60% reduction in overall drug toxicity",
    "output_a": "interactive 3D / VR spatial simulation (OpenXR)",
    "real_data": "full automated patient imaging + genomics pipeline",
}


def poster_summary(report: DosingScheduleReport) -> str:
    """Render the pitch-claims-vs-measured-artifact table for the poster.

    The pitch column is POSTER_PITCH_CLAIMS (verbatim abstract claims);
    the measured column is computed from THIS report. When the policy arm
    reaches the horizon without progressing, the PFS figure is rendered
    as a censored lower bound ("x ..."), never as a point estimate --
    hand-transcribing '+196.7%' without the censoring qualifier is exactly
    the over-precision this module exists to prevent.
    """
    horizon = report.horizon_days

    # PFS row: censoring is determined by the report, not asserted.
    if report.pfs_extension_median is None:
        pfs_measured = "not estimable (MTD control never progressed within "
        pfs_measured += f"the {horizon} d horizon)"
    else:
        pct = f"{report.pfs_extension_median:+.1%}"
        censored = report.policy_median_pfs >= horizon
        pfs_measured = (
            f">= {pct} (censored at the {horizon} d horizon)"
            if censored
            else f"{pct} (uncensored)"
        )

    # Toxicity row.
    if report.toxicity_reduction_median is None:
        tox_measured = "not estimable"
    else:
        tox_measured = (
            f"{report.toxicity_reduction_median:.1%} (measured under policy, "
            "paired MTD control)"
        )

    output_a_measured = (
        "ParaView volumetric .vti export (drug gradients + clone "
        "densities; no VR claims)"
    )
    data_measured = (
        f"{report.patient.geometry_note}; resistant fraction from intake "
        "provenance; explicit input bounds"
    )

    bar = "=" * 76
    lines = [bar, "MEASURED PIPELINE PERFORMANCE vs ORIGINAL PITCH CLAIMS", bar]
    rows = [
        ("Progression-free survival", "pfs_extension", pfs_measured),
        ("Toxicity reduction", "toxicity_reduction", tox_measured),
        ("Output A visualizer", "output_a", output_a_measured),
        ("Real-data integration", "real_data", data_measured),
    ]
    for metric, key, measured in rows:
        lines.append(metric)
        lines.append(f"  pitch claim (verbatim) : {POSTER_PITCH_CLAIMS[key]}")
        lines.append(f"  measured artifact      : {measured}")
        lines.append("")
    lines.append(
        "pitch column: verbatim claims from the original abstract, kept "
        "unsoftened."
    )
    lines.append(
        "measured column: computed by compile_dosing_schedule under "
        "ASSUMED constants"
    )
    lines.append(
        "-- see schedule_provenance() and report.notes for the full "
        "uncertainty record."
    )
    lines.append(bar)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output A: ParaView-loadable spatial fields (.vti ImageData)
# ---------------------------------------------------------------------------


def export_fields_vti(
    traj: SimulationTrajectory,
    geometry: TumorGeometry,
    path: str,
    *,
    include_tissue: bool = True,
) -> str:
    """Write the final spatial fields as a VTK ImageData (.vti) XML file.

    Exports drug_A, drug_B, sensitive and resistant densities on the
    simulation grid, loadable directly in ParaView (Output A of the
    pipeline, minus the VR claims -- ParaView's own 3D view is the
    honest version of 'interactive 3D analytics').

    With ``include_tissue=True`` (default) three derived arrays are added
    so a render can actually SHOW the heterogeneity the model integrates:

    * ``tissue_class``  -- 0 necrotic core, 1 edema, 2 white matter, 3 grey
      matter (labels from `danon.anisotropy.tissue_classes_from_geometry`,
      derived from the geometry itself, not asserted from imaging).
    * ``diffusion_mm2_per_day`` -- the per-voxel physical drug diffusivity
      D(x) those classes imply, via `DiffusionField.from_tissue_classes`.
    * ``proliferative`` / ``vessels`` -- the geometry's own growth and
      drug-source maps.

    Axis convention (pinned by test on a non-cubic grid): engine arrays are
    indexed [x, y, z] (NIfTI dim order, axis 0 = x), and VTK ImageData is
    x-fastest, so the bytes are written in Fortran order and the extent is
    ``0 sx-1 0 sy-1 0 sz-1`` with sx = shape[0]. The tumour mask is PointData
    (one value per point, same as every other array) rather than CellData,
    which previously carried point-count values -- malformed VTK.

    Requires run(..., keep_fields=True). Returns the written path.
    """
    fields: Dict[str, Optional[np.ndarray]] = {
        "drug_A": traj.final_conc_A,
        "drug_B": traj.final_conc_B,
        "sensitive": traj.final_N_S,
        "resistant": traj.final_N_R,
    }
    missing = [k for k, v in fields.items() if v is None]
    if missing:
        raise ValueError(
            f"run the simulation with keep_fields=True to export {missing}"
        )

    # Derived tissue/diffusion arrays, from the geometry the run consumed.
    extra: Dict[str, np.ndarray] = {}
    if include_tissue:
        classes = tissue_classes_from_geometry(geometry)
        extra["tissue_class"] = classes.astype(np.float32)
        extra["diffusion_mm2_per_day"] = (
            DiffusionField.from_tissue_classes(
                classes, voxel_mm=float(geometry.voxel_mm)
            ).d_mm2_per_day.astype(np.float32)
        )
        extra["proliferative"] = np.asarray(
            geometry.proliferative, dtype=np.float32
        )
        extra["vessels"] = np.asarray(geometry.vessels, dtype=np.float32)

    shape = geometry.tumor_mask.shape
    sx, sy, sz = (int(s) for s in shape)  # axis0 = x (NIfTI dim order)
    spacing = float(geometry.voxel_mm)
    header = (
        '<?xml version="1.0"?>\n'
        f'<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="0 {sx - 1} 0 {sy - 1} 0 {sz - 1}" '
        f'Origin="0 0 0" Spacing="{spacing} {spacing} {spacing}">\n'
        f'    <Piece Extent="0 {sx - 1} 0 {sy - 1} 0 {sz - 1}">\n'
        f'      <PointData Scalars="drug_A">\n'
    )
    body = []
    offset = 0
    for name, arr in fields.items():
        assert arr is not None
        # VTK expects x-fastest ordering; our arrays are (x, y, z) with x
        # first, so Fortran order gives the required x-fastest layout.
        raw = np.asfortranarray(arr).astype("<f4").tobytes(order="F")
        b64 = base64.b64encode(raw).decode("ascii")
        body.append(
            f'        <DataArray type="Float32" Name="{name}" '
            f'NumberOfComponents="1" format="binary" offset="{offset}">'
            f"{b64}"
            f"</DataArray>\n"
        )
        offset += len(raw)
    for name, arr in extra.items():
        raw = np.asfortranarray(arr).astype("<f4").tobytes(order="F")
        b64 = base64.b64encode(raw).decode("ascii")
        body.append(
            f'        <DataArray type="Float32" Name="{name}" '
            f'NumberOfComponents="1" format="binary" offset="{offset}">'
            f"{b64}"
            f"</DataArray>\n"
        )
        offset += len(raw)
    # The mask rides with the point data (one value per point, like the
    # fields) -- previously it was declared CellData with point-count
    # values, which is not a well-formed VTK ImageData.
    raw_mask = geometry.tumor_mask.astype("u1").tobytes(order="F")
    b64_mask = base64.b64encode(raw_mask).decode("ascii")
    body.append(
        f'        <DataArray type="UInt8" Name="tumour_mask" '
        f'NumberOfComponents="1" format="binary" offset="{offset}">'
        f"{b64_mask}"
        f"</DataArray>\n"
    )
    offset += len(raw_mask)
    footer = (
        f"      </PointData>\n"
        f"    </Piece>\n"
        f"  </ImageData>\n"
        f"</VTKFile>\n"
    )
    with open(path, "w", encoding="ascii") as fh:
        fh.write(header + "".join(body) + footer)
    return path


# ---------------------------------------------------------------------------
# Provenance and honest limits
# ---------------------------------------------------------------------------


def schedule_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "adaptive_vs_standard_ttp_ratio_mcrpc": EvidenceStatus.ESTABLISHED,
        "rano_progression_threshold": EvidenceStatus.ESTABLISHED,
        "switch_rule_observability": EvidenceStatus.PROVEN_FAILED,  # no real-time clone assay exists
        "schedule_day_precision": EvidenceStatus.PROVEN_FAILED,  # simulation resolution artifact
        "growth_kill_constants": EvidenceStatus.UNKNOWN,  # tier-3 bench
        "threshold_band_optimality": EvidenceStatus.UNKNOWN,  # sweep-derived, over-precision risk
    }


def what_this_model_cannot_know() -> List[str]:
    return [
        "a real-time measure of intratumoural resistant fraction: the "
        "switch trigger is not clinically observable, so the schedule is "
        "operationalisable only through a validated surrogate biomarker",
        "patient-specific geometry: the phantom is not an MRI "
        "reconstruction; segmentation is upstream and unclaimed here",
        "pharmacokinetic reality (metabolism, clearance organs, "
        "blood-brain barrier): diffusion/clearance constants are toy "
        "values pending bench measurement",
        "when a real switch should occur: day stamps carry the "
        "simulation's daily resolution and the phantom's dynamics only",
        "whether steering helps THIS patient: the paired metrics are "
        "per-phantom simulation outputs, not predictions",
    ]


def print_schedule_disclaimer() -> None:
    print(
        "=" * 72 + "\n"
        "SCHEDULE COMPILER -- simulation-derived artifact only\n"
        "Switch days are read off simulated dose logs; the resistant\n"
        "fraction trigger is NOT clinically observable. Constants are\n"
        "ASSUMED pending bench measurement. No patient data, no claim.\n"
        + "=" * 72
    )
