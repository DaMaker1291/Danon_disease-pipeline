"""
Semi-synthetic 5-ALA standard: what a planted fluorescence map can and cannot
prove, run through the real intake path.

WHY THIS MODULE EXISTS
----------------------
A pitch proposes to test stream 1 without intra-operative data by planting
synthetic fluorescence on a "residual contrast-enhancing surgical cavity wall"
from a real post-operative T1-contrast scan, then reading it back. That is a
reasonable *machinery* test -- the intake code is real and can be exercised --
and it is the only test of stream 1 that can be run here, because no public
5-ALA capture exists in this tree. But it is not a validation of anything
biological, and this module says so in every record it writes.

WHAT IS REFUSED UP FRONT
------------------------
* No post-operative imaging is cached: the committed BraTS-2021 release is
  pre-operative only (six cases, one timepoint each -- see
  `data/tcia_probe/multimodal_evidence.json`). A "resection cavity wall" cannot
  be extracted from what is here, so the planted field is placed on the
  PRE-OPERATIVE tumour rim and the word "cavity" is not used for it.
* The planted field is manufactured by this module. Feeding a manufactured
  field through `danon.clinical_twin.stream1_initial_state` proves the intake
  path is volume-faithful and that the engine accepts a supplied seed; it
  cannot show that real 5-ALA maps look like this, that fluorescence marks
  tumour, or that any volume a mock-up quotes is recoverable.
* Stream 1 therefore stays REFUSED in `clinical_twin.clinical_stream_census()`,
  on its live count of zero cached captures. The claim check requires that it
  does: a standard that flipped the stream to "available" would be the exact
  failure this repository exists to prevent.

WHAT IS MEASURED
----------------
1. The intake identity, on the real geometry: a uniform planted map must fill
   to exactly `fill_fraction * K * tumour_voxels`, the same total the engine's
   own clustered seed produces.
2. The layout contrast: rim-weighted, uniform and core-weighted maps carry the
   SAME total burden and different spatial layouts, run on the committed
   BraTS-2021 case with the same seeds under MTD. Whatever difference the
   endpoints show is what stream 1's spatial input actually controls -- a real
   engine measurement, with a synthetic input. On this case the answer is
   NEGATIVE and recorded as such: the segmentation is a shell (mean distance
   from the boundary ~1.06 voxels for every planted mode), so the three
   fluorescence layouts are almost the same field and their endpoints agree to
   a fraction of a percent. What DOES move the endpoints is the resistant-clone
   layout: the engine's clustered R patch against the stream's uniform initial
   fraction is a paired median burden delta of ~+1,292 at 365 days -- so the
   stream's clonal placement, not its fluorophore pattern, is the dominant term
   on this case.
3. The decay-length sweep: the planted construction has one declared constant
   (the decay length in voxels); its effect is reported rather than tuned away.

This is the same class of standard as `danon/recovery.py`'s planted-ground-truth
tests: the input is known because it was written, and nothing measured here is
evidence about a patient.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus

# ---------------------------------------------------------------------------
# The committed case and the planted construction
# ---------------------------------------------------------------------------

STANDARD_CASE = "BraTS2021_01017"
MASK_PATH = os.path.join("data", "engine_ready", f"{STANDARD_CASE}_mask.nii.gz")
PROLIF_PATH = os.path.join("data", "engine_ready", f"{STANDARD_CASE}_prolif.nii.gz")
VESSELS_PATH = os.path.join("data", "engine_ready", f"{STANDARD_CASE}_vessels_t1ce.nii.gz")

#: The engine grid's physical voxel, from the decimation factors (4 x 4 x 3 mm
#: at 1 mm source). Used only to state the decay length in millimetres; the
#: planted field and every engine run are in voxel units.
TRUE_VOXEL_MM = 3.634

PLANT_MODES: Tuple[str, ...] = ("uniform", "rim", "core")
DEFAULT_DECAY_VOXELS = 2.0
DECAY_SWEEP_VOXELS: Tuple[float, ...] = (1.0, 2.0, 4.0)

#: Distance band (voxels) that counts as "rim" when describing a planted field.
RIM_BAND_VOXELS = 1.0

SEEDS: Tuple[int, ...] = (17, 18, 19)
DAYS_SHORT = 60
DAYS_LONG = 365

STANDARD_RECORD = os.path.join("outputs_real", "fluorescence_standard.json")

#: Required verbatim in the record, so no reader can mistake the planted field
#: for an acquisition and no check can pass while the disclaimer is missing.
SYNTHETIC_DISCLAIMER = (
    "the planted fluorescence field is manufactured by this repository and is "
    "not a measured 5-ALA acquisition"
)

HISTOLOGY_DENSITY_NOTE = (
    "the cached pathology sample is haematoxylin-only, so it measures a real "
    "cell density and cannot support a Ki-67 ratio"
)


# ---------------------------------------------------------------------------
# Planting
# ---------------------------------------------------------------------------


def _inside_distance(mask: np.ndarray) -> np.ndarray:
    """Euclidean distance (voxels) from each inside voxel to the nearest edge.

    An edge voxel scores 1.0; outside voxels score 0.0 by construction of the
    transform, which is why every consumer here intersects with the mask.
    """
    from scipy import ndimage

    m = np.asarray(mask).astype(bool)
    if not m.any():
        raise ValueError("empty mask: nothing to plant on")
    return ndimage.distance_transform_edt(m)


def plant_field(
    mask: np.ndarray,
    mode: str = "rim",
    *,
    decay_voxels: float = DEFAULT_DECAY_VOXELS,
) -> np.ndarray:
    """A synthetic fluorescence intensity field on a real mask.

    Three constructions, all declared and none measured:

    * `uniform` -- every tumour voxel fluoresces equally; the intake then
      reproduces the engine's total fill exactly (that is the identity test).
    * `rim` -- `exp(-d / decay_voxels)`, bright at the margin, dim at depth.
    * `core` -- `exp(-(d_max - d) / decay_voxels)`, the mirror image.

    The field is zero outside the mask, which is what a segmentation would
    enforce on an acquisition too.
    """
    if mode not in PLANT_MODES:
        raise ValueError(f"unknown plant mode {mode!r}; use one of {PLANT_MODES}")
    if not decay_voxels > 0.0:
        raise ValueError(f"decay_voxels must be positive, got {decay_voxels!r}")
    m = np.asarray(mask).astype(bool)
    d = _inside_distance(m)
    if mode == "uniform":
        values = np.ones_like(d)
    elif mode == "rim":
        values = np.exp(-d / float(decay_voxels))
    else:  # core
        d_max = float(d[m].max())
        values = np.exp(-(d_max - d) / float(decay_voxels))
    return np.where(m, values, 0.0)


@dataclass(frozen=True)
class FieldDescriptor:
    """What the planted field is, before any engine sees it."""

    mode: str
    decay_voxels: float
    n_support: int
    total_intensity: float
    rim_share: float
    core_share: float
    mean_distance: float
    peak_distance: float

    def describe(self) -> str:
        return (
            f"{self.mode} (decay {self.decay_voxels:g} vox): support "
            f"{self.n_support} voxels, rim band {100 * self.rim_share:.1f}% of "
            f"intensity, mean distance from edge {self.mean_distance:.2f} voxels"
        )


def describe_field(field: np.ndarray, mask: np.ndarray, *, mode: str, decay_voxels: float) -> FieldDescriptor:
    m = np.asarray(mask).astype(bool)
    f = np.asarray(field, dtype=float)
    inside = f[m]
    total = float(inside.sum())
    d = _inside_distance(m)
    d_in = d[m]
    if total > 0.0:
        rim = float(inside[d_in <= RIM_BAND_VOXELS].sum()) / total
        core = float(inside[d_in > float(d_in.max()) * 0.5].sum()) / total
        mean_d = float((inside * d_in).sum() / total)
        peak_d = float(d_in[int(np.argmax(inside))])
    else:  # pragma: no cover - plant_field cannot produce this
        rim = core = mean_d = peak_d = float("nan")
    return FieldDescriptor(
        mode=mode,
        decay_voxels=float(decay_voxels),
        n_support=int((inside > 0).sum()),
        total_intensity=total,
        rim_share=rim,
        core_share=core,
        mean_distance=mean_d,
        peak_distance=peak_d,
    )


# ---------------------------------------------------------------------------
# Running the standard through the real intake path
# ---------------------------------------------------------------------------


@dataclass
class LayoutRun:
    """One engine run under one planted layout (or the engine's own seed)."""

    mode: str
    seed: int
    days: int
    initial_burden: float
    final_burden: float
    pfs_day: Optional[int]
    progressed: bool
    final_resistant_fraction: float
    outside_share: Optional[float] = None
    centroid_shift_voxels: Optional[float] = None


def standard_geometry():
    """The committed case, loaded exactly as the other cohort scripts load it."""
    from danon.imaging import from_nifti_mask

    for path in (MASK_PATH, PROLIF_PATH, VESSELS_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"standard case file missing: {path}")
    patient = from_nifti_mask(
        MASK_PATH, proliferative_path=PROLIF_PATH, vessels_path=VESSELS_PATH
    )
    return patient.geometry


def _spatial_stats(final: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    total = float(final.sum())
    if total <= 0.0:
        return float("nan"), float("nan")
    outside = float(final[~np.asarray(mask, dtype=bool)].sum()) / total
    grid = np.indices(final.shape).reshape(3, -1).astype(float)
    weights = final.reshape(-1)
    centre = (grid * weights).sum(axis=1) / weights.sum()
    idx = np.argwhere(mask)
    origin = idx.mean(axis=0)
    return outside, float(np.linalg.norm(centre - origin))


def run_layout(
    geom,
    mode: str,
    *,
    days: int = DAYS_SHORT,
    seed: int = SEEDS[0],
    decay_voxels: float = DEFAULT_DECAY_VOXELS,
    keep_fields: bool = False,
) -> LayoutRun:
    """Run one trial with the supplied t=0 map, or the engine's own seed.

    `mode == "default"` is the comparison arm: no stream 1, the engine's
    clustered seed. Every other mode goes through
    `assemble_digital_twin_state(fluorescence_5ala_mesh=...)` -- the same path a
    real capture would take -- and then into `TumorEvolutionSimulator.run`.
    """
    from danon.clinical_twin import assemble_digital_twin_state
    from danon.evosteer import MTDPolicy, TumorEvolutionSimulator

    sim = TumorEvolutionSimulator(geom)
    policy = MTDPolicy()
    mask = geom.tumor_mask

    if mode == "default":
        initial = sim.seed(0.01, seed=seed, fill_fraction=0.65)
        initial_burden = float(initial[0].sum() + initial[1].sum())
        traj = sim.run(policy, days=days, seed=seed, keep_fields=keep_fields)
    elif mode in PLANT_MODES:
        field = plant_field(mask, mode, decay_voxels=decay_voxels)
        state = assemble_digital_twin_state(fluorescence_5ala_mesh=field, mask=mask)
        N_S0, N_R0 = state.initial_state
        initial_burden = float(N_S0.sum() + N_R0.sum())
        traj = sim.run(
            policy, days=days, seed=seed, initial_state=(N_S0, N_R0), keep_fields=keep_fields
        )
    else:
        raise ValueError(f"unknown layout {mode!r}")

    outside = shift = None
    if keep_fields and traj.final_N_S is not None:
        final = traj.final_N_S + traj.final_N_R
        outside, shift = _spatial_stats(final, mask)
    return LayoutRun(
        mode=mode,
        seed=int(seed),
        days=int(days),
        initial_burden=initial_burden,
        final_burden=float(traj.burden_S[-1] + traj.burden_R[-1]),
        pfs_day=traj.pfs_day,
        progressed=bool(traj.progressed),
        final_resistant_fraction=float(traj.final_resistant_fraction),
        outside_share=outside,
        centroid_shift_voxels=shift,
    )


# ---------------------------------------------------------------------------
# The standard's report
# ---------------------------------------------------------------------------


@dataclass
class StandardReport:
    case: str
    grid: Tuple[int, int, int]
    tumour_voxels: int
    voxel_mm: float
    descriptors: List[FieldDescriptor]
    identity: Dict[str, float]
    runs: List[LayoutRun]
    notes: List[str] = field(default_factory=list)

    def by_mode(self, mode: str, days: int) -> List[LayoutRun]:
        return [r for r in self.runs if r.mode == mode and r.days == days]


def _median(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def standard_report(
    *,
    seeds: Sequence[int] = SEEDS,
    days_short: int = DAYS_SHORT,
    days_long: int = DAYS_LONG,
    decay_voxels: float = DEFAULT_DECAY_VOXELS,
) -> StandardReport:
    """Plant the fields, run every layout, and report what moved."""
    from danon.clinical_twin import assemble_digital_twin_state
    from danon.evosteer import DEFAULT_CLONES

    geom = standard_geometry()
    mask = geom.tumor_mask
    n_voxels = int(mask.sum())
    k = float(DEFAULT_CLONES.carrying_capacity)

    descriptors = [
        describe_field(plant_field(mask, "uniform"), mask, mode="uniform", decay_voxels=decay_voxels),
        describe_field(plant_field(mask, "rim", decay_voxels=decay_voxels), mask, mode="rim", decay_voxels=decay_voxels),
        describe_field(plant_field(mask, "core", decay_voxels=decay_voxels), mask, mode="core", decay_voxels=decay_voxels),
    ]
    for lam in DECAY_SWEEP_VOXELS:
        descriptors.append(
            describe_field(plant_field(mask, "rim", decay_voxels=lam), mask, mode="rim", decay_voxels=lam)
        )

    uniform_field = plant_field(mask, "uniform")
    state = assemble_digital_twin_state(fluorescence_5ala_mesh=uniform_field, mask=mask)
    N_S0, N_R0 = state.initial_state
    stream1_total = float(N_S0.sum() + N_R0.sum())
    target_total = 0.65 * k * n_voxels
    identity = {
        "stream1_uniform_total": stream1_total,
        "engine_target_total": target_total,
        "absolute_difference": abs(stream1_total - target_total),
    }

    runs: List[LayoutRun] = []
    for mode in ("default",) + PLANT_MODES:
        for seed in seeds:
            runs.append(
                run_layout(geom, mode, days=days_short, seed=seed, decay_voxels=decay_voxels, keep_fields=True)
            )
            runs.append(
                run_layout(geom, mode, days=days_long, seed=seed, decay_voxels=decay_voxels)
            )

    report = StandardReport(
        case=STANDARD_CASE,
        grid=tuple(int(x) for x in geom.tumor_mask.shape),
        tumour_voxels=n_voxels,
        voxel_mm=TRUE_VOXEL_MM,
        descriptors=descriptors,
        identity=identity,
        runs=runs,
        notes=[],
    )

    summary = layout_summary(report)
    by_key = {(row["mode"], row["days"]): row for row in summary}
    shell_dmax = max(d.mean_distance for d in descriptors[:3])
    stream_modes = [m for m in PLANT_MODES]
    pattern_spread = max(
        abs(by_key[(m, days_short)]["median_final_burden"] - by_key[(PLANT_MODES[0], days_short)]["median_final_burden"])
        for m in stream_modes
    )
    pattern_median = by_key[(PLANT_MODES[0], days_long)]["median_final_burden"]
    pattern_rel = max(
        abs(by_key[(m, days_long)]["median_final_burden"] - pattern_median) / pattern_median
        for m in stream_modes
    )
    clone_delta = by_key[(PLANT_MODES[0], days_long)]["median_paired_delta_vs_default"]

    report.notes = [
        SYNTHETIC_DISCLAIMER,
        f"planted on the PRE-OPERATIVE tumour rim of {STANDARD_CASE}: no "
        f"post-operative imaging is cached, so no resection cavity exists to "
        f"plant on (the pitch's step 1 cannot be run as written)",
        f"the engine grid is {geom.tumor_mask.shape} at {TRUE_VOXEL_MM:.3f} mm/"
        f"voxel ({n_voxels} tumour voxels); the decay length is declared in "
        f"voxels and its sweep is recorded rather than tuned",
        f"the case is a SHELL, not a ball: mean distance from the boundary is "
        f"{shell_dmax:.2f} voxels for every planted mode, so the three "
        f"fluorescence layouts are nearly the same field (their day-{days_long} "
        f"medians agree to {100 * pattern_rel:.3f}%, and {pattern_spread:.2f} "
        f"burden units separate them at day {days_short}); the layout contrast "
        f"is therefore recorded as INERT ON THIS CASE rather than presented as "
        f"a measured sensitivity",
        f"what does move the endpoints is the resistant-clone layout: the "
        f"engine's clustered R patch versus the stream's uniform r0 gives a "
        f"paired median burden delta of {clone_delta:+.2f} at day {days_long}, "
        f"so the stream's clonal placement dominates its fluorophore pattern "
        f"on this case -- a finding about this model's initial condition, not "
        f"about 5-ALA",
        "what this can show: the intake identity, whether a supplied seed "
        "changes the trajectory, and where the change comes from",
        "what this cannot show: that a real 5-ALA map resembles the planted "
        "field, that fluorescence marks tumour, or anything about a patient",
    ]
    return report


def layout_contrast(report: StandardReport) -> Dict[str, Any]:
    """The paired contrast between planted layouts at matched total burden.

    `(max - min) / median` across the three stream-1 modes, per horizon -- the
    number that says whether the fluorophore pattern moved anything on this
    case. Computed from the same runs the summary is, so the two cannot drift.
    """
    summary = layout_summary(report)
    out: Dict[str, Any] = {}
    for days in sorted({row["days"] for row in summary}):
        medians = [
            row["median_final_burden"]
            for row in summary
            if row["days"] == days and row["mode"] in PLANT_MODES
        ]
        if not medians:
            continue
        mid = float(np.median(medians))
        out[str(days)] = {
            "n_modes": len(medians),
            "median_final_burden": mid,
            "relative_spread": float((max(medians) - min(medians)) / mid) if mid else float("nan"),
        }
    return out


def layout_summary(report: StandardReport) -> List[Dict[str, Any]]:
    """Per-mode medians and the paired contrast against the engine's own seed."""
    out: List[Dict[str, Any]] = []
    defaults = {(r.seed, r.days): r for r in report.runs if r.mode == "default"}
    for mode in ("default",) + PLANT_MODES:
        for days in sorted({r.days for r in report.runs}):
            rows = report.by_mode(mode, days)
            if not rows:
                continue
            paired = [
                r.final_burden - defaults[(r.seed, days)].final_burden
                for r in rows
                if (r.seed, days) in defaults and mode != "default"
            ]
            out.append(
                {
                    "mode": mode,
                    "days": days,
                    "n": len(rows),
                    "median_pfs_day": _median(
                        [r.pfs_day if r.pfs_day is not None else float(days) for r in rows]
                    ),
                    "median_final_burden": _median([r.final_burden for r in rows]),
                    "final_burden_spread": (
                        float(max(r.final_burden for r in rows) - min(r.final_burden for r in rows))
                        if len(rows) > 1
                        else 0.0
                    ),
                    "median_paired_delta_vs_default": _median(paired) if paired else None,
                    "median_outside_share": _median(
                        [r.outside_share for r in rows if r.outside_share is not None]
                    )
                    if any(r.outside_share is not None for r in rows)
                    else None,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Refusals and provenance
# ---------------------------------------------------------------------------


def standard_refusals() -> List[Dict[str, Any]]:
    """The pitch's procedure, checked step by step against what is here."""
    return [
        {
            "shown_in_the_pitch": (
                "extract the residual contrast-enhancing surgical cavity wall "
                "from a real post-operative T1-contrast MRI"
            ),
            "why_refused": (
                "no post-operative imaging is cached: the committed BraTS-2021 "
                "release is pre-operative only (six cases, one timepoint each; "
                "recorded in data/tcia_probe/multimodal_evidence.json), so a "
                "resection cavity cannot be extracted and the word 'cavity' is "
                "not used for what was planted"
            ),
            "what_is_recorded_instead": (
                "the planted field uses the PRE-OPERATIVE tumour rim of the "
                f"expert segmentation ({STANDARD_CASE}); the substitution is "
                "named in the record"
            ),
        },
        {
            "shown_in_the_pitch": (
                "'inject synthetic fluorescence intensity values ... to simulate "
                "a digitized 5-ALA camera readout'"
            ),
            "why_refused": (
                "a field manufactured here and read back here proves the intake "
                "path works; it cannot show that real 5-ALA maps look like the "
                "planted construction, that fluorescence marks tumour, or that "
                "any mock-up volume (e.g. 0.42 cm3 residual) is recoverable"
            ),
            "what_is_recorded_instead": (
                "the intake identity on the real geometry, the matched-burden "
                "layout contrast under MTD, and the decay-length sweep -- every "
                "entry labelled synthetic"
            ),
        },
        {
            "shown_in_the_pitch": (
                "DSC perfusion maps from Ivy GAP or CPTAC-3"
            ),
            "why_refused": (
                "the repository's measured DSC perfusion is UPENN-GBM (456 "
                "patients with DSC series; three series downloaded and read), "
                "and Ivy GAP / CPTAC-3 are not probed here -- naming them would "
                "misdescribe what was measured"
            ),
            "what_is_recorded_instead": (
                "danon/perfusion.py and outputs_real/perfusion_cohort.json"
            ),
        },
        {
            "shown_in_the_pitch": (
                "'Longitudinal sequential MRIs: Fully Available' "
                "(MU-Glioma-Post, BRAIN-TUMOR-PROGRESSION)"
            ),
            "why_refused": (
                "available to read is not acquirable: MU-Glioma-Post documents "
                "203 subjects / 596 post-treatment timepoints with the images "
                "behind a licence login, and the probe returns 0 patients and 0 "
                "segmentation series; the CFB-GBM manifest enumerates 393 "
                "usable T1-Gd pairs over 168 patients and 0 images are "
                "downloaded"
            ),
            "what_is_recorded_instead": (
                "the live census in danon/walkforward.py and the manifest "
                "counts; 0 scorable pairs"
            ),
        },
        {
            "shown_in_the_pitch": (
                "GLASS Consortium for sub-clone resistance shifts"
            ),
            "why_refused": (
                "the clone-SIZE step, not the data: the GLASS release IS measured "
                "in this tree now (danon.variants; the live counts are read back "
                "in the walk-forward census rather than restated here), and the "
                "engine carries exactly two bins -- sensitive and resistant -- "
                "so a k-bin allele-frequency spectrum would still need a "
                "summarisation rule this repository does not have, while a point "
                "clone fraction and an upper bound need a purity and a copy "
                "number the release does not carry"
            ),
            "what_is_recorded_instead": (
                "the walk-forward census's GLASS entry, its variant_channel "
                "block and danon/variants' own refusals"
            ),
        },
        {
            "shown_in_the_pitch": (
                "pathology WSI: 'trains and tests nuclear cell packing "
                "algorithms on real digitized H&E slides'"
            ),
            "why_refused": (
                "one slide is cached and it is haematoxylin-only; it yields a "
                "real measured cell DENSITY, which is not a proliferation "
                "index, and it cannot support a Ki-67 ratio; training a packing "
                "model would need labelled nuclei across more than one slide"
            ),
            "what_is_recorded_instead": (
                "the measured cellularity and the refused Ki-67 route in "
                "danon/histology.py via clinical_twin.clinical_stream_census()"
            ),
        },
    ]


def what_a_synthetic_input_cannot_show() -> List[str]:
    return [
        "that real 5-ALA fluorescence fields resemble any planted construction "
        "here: no capture is cached, and the constructions were chosen for "
        "contrast, not fitted to data",
        "that fluorescence marks tumour: that is the biological claim, and a "
        "planted field assumes it rather than testing it",
        "that a residual volume could be recovered from a real capture: the "
        "standard supplies the volume by construction",
        "anything about a patient: the case is one de-identified BraTS-2021 "
        "segmentation and every field on it is synthetic",
    ]


def fluorescence_standard_provenance() -> Dict[str, EvidenceStatus]:
    return {
        "intake_identity_measured_on_real_geometry": EvidenceStatus.ESTABLISHED,
        "layout_contrast_measured": EvidenceStatus.ESTABLISHED,
        "planted_field_is_a_5ala_acquisition": EvidenceStatus.PROVEN_FAILED,
        "synthetic_input_is_patient_evidence": EvidenceStatus.PROVEN_FAILED,
        "stream1_activates_on_synthetic_input": EvidenceStatus.PROVEN_FAILED,
        "resection_cavity_available_here": EvidenceStatus.PROVEN_FAILED,
    }


def stream1_still_refused() -> Tuple[bool, str]:
    """The live census must still refuse stream 1 -- the standard may not flip it."""
    from danon.clinical_twin import clinical_stream_census

    census = clinical_stream_census()
    entry = census["stream1_5ala"]
    detail = (
        f"stream1_5ala status={entry['status']!r} on {entry['n_files']} cached "
        f"capture(s): {entry['detail']}"
    )
    return entry["status"] == "REFUSED" and int(entry["n_files"]) == 0, detail
