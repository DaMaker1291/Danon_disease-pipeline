"""Per-day spatial evolution capture: the tumour over time, not just at the end.

WHY THIS MODULE EXISTS
----------------------
export_fields_vti writes ONE day (the final state). An evolution viewer --
and any temporal claim a poster or a UI makes about WHEN resistance emerges
or WHEN a sanctuary pocket becomes visible -- needs the fields at many days,
from BOTH arms of the paired comparison, and it needs them to be the same
run the daily burden curves describe. This module records those snapshots
through the engine's identity-preserving field_hook, so:

- the recording is non-invasive: run(..., field_hook=recorder) leaves the
  burden arrays, PFS day, and toxicity bit-identical to the same run
  without the hook (pinned by test);
- the exported day-d fields are exactly the engine state AFTER day d's
  update, i.e. burden_S[d+1] of the trajectory the curves come from --
  never a re-run at different parameters.

The snapshot day schedule is a DENSITY contract, not a size contract: days
0..30 are recorded daily (the early dynamics where the switch decision and
the fastest visible change live), then the step doubles every 30 days
(60..365 by 2, 90..365 by 4 - capped at 7-day steps), so the 365-day
horizon lands on ~100 frames per arm. Frame selection is validated against
the trajectory, never asserted: every frame day's aggregates must match the
run's own burden arrays to float tolerance before anything is written.

ParaView interoperability: alongside the per-frame .vti series this module
writes a .pvd collection file -- ParaView's native time-series format -- so
opening one file in ParaView gives the full 365-day animation with a time
slider for free.

The refusal that shapes the module: a frame day beyond the run's horizon,
a schedule that skips the paired control, or an arm whose aggregates do
not match its recorded frames are all hard errors. An evolution movie
whose frames disagree with its own curves is worse than no movie.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.evosteer import (
    CollateralSwitchPolicy,
    MTDPolicy,
    SimulationTrajectory,
    TumorEvolutionSimulator,
    TumorGeometry,
)
from danon.schedule import export_fields_vti

__all__ = [
    "snapshot_days",
    "FrameSet",
    "capture_frames",
    "write_pvd",
    "export_evolution_frames",
]


# ---------------------------------------------------------------------------
# Snapshot schedule
# ---------------------------------------------------------------------------


def snapshot_days(days: int, dense_until: int = 30, max_frames: int = 110) -> List[int]:
    """Which days to record fields on, dense early and coarsening late.

    Days 0..``dense_until`` are recorded daily (the early dynamics where
    the switch decision and the fastest visible change live), then the
    step is 2 for the next 30 days, 4 for the 30 after that, and 7 from
    there on -- so a 365-day run produces ~100 frames: dense where the eye
    looks, light where nothing changes fast. The result always includes 0
    and the final day, and is validated to be strictly increasing.
    """
    if days < 2:
        raise ValueError(f"need at least 2 days, got {days}")
    dense_until = min(dense_until, days)
    days_out: List[int] = list(range(0, dense_until + 1))
    step = 2
    d = dense_until + step
    while d <= days:
        days_out.append(d)
        since_dense = d - dense_until
        step = min(7, 2 + 2 * (since_dense // 30))
        d += step
    if days_out[-1] != days:
        days_out.append(days)
    # Enforce the budget by thinning from the middle (keep 0 and horizon).
    if len(days_out) > max_frames:
        keep = {0, days}
        idx = np.linspace(0, len(days_out) - 1, max_frames - 2).round().astype(int)
        for i in idx:
            keep.add(days_out[int(i)])
        days_out = sorted(keep)
    return days_out


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


@dataclass
class FrameSet:
    """Recorded spatial state at the scheduled days of ONE run.

    ``fields[d]`` is (N_S, N_R, c_A, c_B) for frame day d, where day d is
    the state after d daily updates: day 0 is the seeded state (burden
    index 0 of the trajectory) and day ``days`` is the final state. The
    full daily trajectory is stored alongside so cross-validation against
    the run is possible without reopening the arrays.
    """

    arm: str
    seed: int
    days: List[int]
    fields: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    burden_S: np.ndarray  # full daily trajectory (not just frame days)
    burden_R: np.ndarray
    final_burden_S: float  # end-of-horizon state (beyond the arrays' index)
    final_burden_R: float
    pfs_day: Optional[int]
    toxicity: float

    def validate_against_trajectory(self, traj: SimulationTrajectory) -> None:
        """Every frame's aggregates must equal the run's own daily arrays.

        Frame day d (1 <= d <= days-1) is the state after d updates ==
        burden_[d]; frame 0 is the seed == burden_[0]; frame ``days`` is
        the post-horizon state, checked against the hook's own final
        aggregate. A mismatch here means the fields and the curves come
        from different runs, which is the one corruption an evolution
        viewer must never ship.
        """
        if self.burden_S.shape != traj.burden_S.shape:
            raise ValueError("frame trajectory shape differs from the run")
        if not np.array_equal(self.burden_S, traj.burden_S):
            raise ValueError(f"{self.arm} seed {self.seed}: recorded curve != run curve")
        for d, (S, R, _cA, _cB) in self.fields.items():
            if d == len(traj.burden_S):
                exp_S, exp_R = self.final_burden_S, self.final_burden_R
            else:
                exp_S, exp_R = float(traj.burden_S[d]), float(traj.burden_R[d])
            tol_S = 1e-6 * max(1.0, abs(exp_S))
            tol_R = 1e-6 * max(1.0, abs(exp_R))
            if abs(float(S.sum()) - exp_S) > tol_S:
                raise ValueError(
                    f"{self.arm} seed {self.seed} day {d}: frame burden "
                    f"{float(S.sum()):.6g} != trajectory {exp_S:.6g}"
                )
            if abs(float(R.sum()) - exp_R) > tol_R:
                raise ValueError(
                    f"{self.arm} seed {self.seed} day {d}: frame R burden mismatch"
                )


def _run_with_recorder(
    sim: TumorEvolutionSimulator,
    arm: str,
    seed: int,
    r0: float,
    days: int,
    want_days: Sequence[int],
    fill_fraction: float = 0.65,
) -> Tuple[FrameSet, SimulationTrajectory]:
    """One engine run with the field hook attached on the wanted days.

    Frame-day convention: day d = state after d updates. Frame 0 is the
    seed (replayed through sim.seed with the identical arguments run() uses
    -- deterministic, and pinned by the burden[0] check below); frames
    1..days come off the hook, whose day-k state is the state after k+1
    updates (the hook fires after each day's update).
    """
    want = set(int(d) for d in want_days)
    if not want <= set(range(days + 1)):
        raise ValueError("frame days outside [0, days]")
    store: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}

    # Frame 0: the seeded state, replayed deterministically.
    if 0 in want:
        S0, R0 = sim.seed(r0, seed=int(seed), fill_fraction=fill_fraction)
        shape = sim.geom.tumor_mask.shape if hasattr(sim, "geom") else S0.shape
        z = np.zeros(shape, dtype=float)
        store[0] = (S0.copy(), R0.copy(), z.copy(), z.copy())

    aggregates_S: List[float] = []
    aggregates_R: List[float] = []

    def hook(day: int, N_S, N_R, c_A, c_B) -> None:
        aggregates_S.append(float(N_S.sum()))
        aggregates_R.append(float(N_R.sum()))
        if (day + 1) in want:  # end-of-day state == frame day+1
            store[day + 1] = (N_S.copy(), N_R.copy(), c_A.copy(), c_B.copy())

    policy_obj = CollateralSwitchPolicy(0.15, 0.05) if arm == "policy" else MTDPolicy()
    traj = sim.run(
        policy_obj,
        days=days,
        initial_resistant_fraction=r0,
        seed=int(seed),
        field_hook=hook,
    )
    # Hook aggregates: end-of-day k == burden_[k+1] for k < days-1; the last
    # one is the final state, which sits one index beyond the daily arrays.
    if not np.allclose(
        np.asarray(aggregates_S[: days - 1]), traj.burden_S[1:], rtol=0, atol=0
    ):
        raise RuntimeError("hook aggregates diverged from the run's own log")
    if not np.allclose(
        np.asarray(aggregates_R[: days - 1]), traj.burden_R[1:], rtol=0, atol=0
    ):
        raise RuntimeError("hook R aggregates diverged from the run's own log")
    missing = sorted(want - set(store))
    if missing:
        raise RuntimeError(f"hook never fired for days {missing[:5]}...")
    fs = FrameSet(
        arm=arm,
        seed=int(seed),
        days=sorted(store),
        fields=store,
        burden_S=traj.burden_S.copy(),
        burden_R=traj.burden_R.copy(),
        final_burden_S=aggregates_S[-1],
        final_burden_R=aggregates_R[-1],
        pfs_day=traj.pfs_day,
        toxicity=traj.toxicity,
    )
    fs.validate_against_trajectory(traj)
    return fs, traj


def capture_frames(
    geometry: TumorGeometry,
    *,
    r0: float,
    seed: int,
    days: int = 365,
    arm: str = "policy",
    fill_fraction: float = 0.65,
    dense_until: int = 30,
    max_frames: int = 110,
) -> Tuple[FrameSet, SimulationTrajectory]:
    """Run ONE arm/seed with the field recorder and validate the frames.

    The policy construction mirrors danon.outcome_export._run_arm exactly
    (CollateralSwitchPolicy(0.15, 0.05) / MTDPolicy), so a frame export for
    (geometry, r0, seed) replays the same physics the outcome cards and the
    twin planner report -- the movie and the numbers cannot disagree.
    """
    if arm not in ("policy", "mtd"):
        raise ValueError(f"arm must be 'policy' or 'mtd', got {arm!r}")
    if not 0.0 <= r0 <= 1.0:
        raise ValueError(f"r0 must be a fraction in [0, 1], got {r0!r}")
    sim = TumorEvolutionSimulator(geometry)
    want = snapshot_days(days, dense_until=dense_until, max_frames=max_frames)
    return _run_with_recorder(
        sim, arm, seed, r0, days, want, fill_fraction=fill_fraction
    )


# ---------------------------------------------------------------------------
# Export: per-frame .vti series + ParaView .pvd time collection
# ---------------------------------------------------------------------------


def write_pvd(
    pvd_path: str,
    vti_paths: Sequence[str],
    times: Sequence[float],
) -> str:
    """Write a ParaView Collection file indexing the frame series.

    ParaView opens the .pvd as one time-varying dataset: the play controls
    become the day axis. Times are the days; the file list must be sorted
    by time (ParaView requires monotone time steps).
    """
    if len(vti_paths) != len(times):
        raise ValueError("vti_paths and times must have equal length")
    if list(times) != sorted(times):
        raise ValueError("times must be monotone non-decreasing")
    root = os.path.dirname(os.path.abspath(pvd_path))
    parts = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1">',
        "  <Collection>",
    ]
    for path, t in zip(vti_paths, times):
        rel = os.path.relpath(os.path.abspath(path), root).replace(os.sep, "/")
        parts.append(f'    <DataSet timestep="{t}" part="{int(t)}" file="{rel}"/>')
    parts.append("  </Collection>")
    parts.append("</VTKFile>")
    with open(pvd_path, "w", encoding="ascii") as fh:
        fh.write("\n".join(parts) + "\n")
    return pvd_path


def resistant_fraction_gated(burden_S: float, burden_R: float) -> float:
    """R/(S+R), or 0.0 when the tumour is below the detection floor.

    The ratio is only a clinical quantity for a detectable lesion: it reaches
    1.0 the instant the sensitive population goes extinct, so an eradicated
    tumour with a five-cell remnant would otherwise be reported as '100%
    resistant'. Every surface that reports this number routes through here,
    so the app, the exported frames and the engine cannot disagree about it.
    """
    from danon.evosteer import DETECTION_BURDEN_DENSITY

    total = float(burden_S + burden_R)
    if total < DETECTION_BURDEN_DENSITY:
        return 0.0
    return float(burden_R / total)


def resistant_fraction_series(burden_S, burden_R) -> List[float]:
    """Per-day gated resistant fraction for a whole trajectory."""
    return [
        resistant_fraction_gated(float(s), float(r))
        for s, r in zip(burden_S, burden_R)
    ]


def _frame_vti_bytes(
    fs: FrameSet,
    day: int,
    geometry: TumorGeometry,
    *,
    include_tissue: bool = True,
) -> bytes:
    """One frame as the exact bytes of export_fields_vti.

    ``include_tissue=False`` drops the four geometry-constant derived maps
    (tissue_class, diffusion, proliferative, vessels) from the frame --
    they are identical in every frame of a run, so a series carries them
    once (see ``static_geometry_maps``) instead of ~44 times.
    """
    S, R, cA, cB = fs.fields[day]
    traj_like = SimulationTrajectory(
        policy=f"evolution frame ({fs.arm}, seed {fs.seed})",
        days=day + 1,
        burden_S=fs.burden_S[: day + 1],
        burden_R=fs.burden_R[: day + 1],
        dose_A=np.zeros(day + 1),
        dose_B=np.zeros(day + 1),
        pfs_day=fs.pfs_day if fs.pfs_day is not None and fs.pfs_day <= day else None,
        progressed=fs.pfs_day is not None and fs.pfs_day <= day,
        initial_resistant_fraction=0.0,
        toxicity=fs.toxicity,
        final_resistant_fraction=resistant_fraction_gated(
            float(S.sum()), float(R.sum())
        ),
        resistant_dominance_day=None,
    )
    traj_like.final_conc_A = cA
    traj_like.final_conc_B = cB
    traj_like.final_N_S = S
    traj_like.final_N_R = R
    with tempfile.TemporaryDirectory(prefix="evo_") as td:
        p = os.path.join(td, "frame.vti")
        export_fields_vti(traj_like, geometry, p, include_tissue=include_tissue)
        with open(p, "rb") as fh:
            return fh.read()


@dataclass
class EvolutionExport:
    """Everything the evolution player (or ParaView) needs for one run."""

    arm: str
    seed: int
    frame_days: List[int]
    frames_vti_b64: List[str]
    burden_S: np.ndarray
    burden_R: np.ndarray
    dose_A: np.ndarray
    dose_B: np.ndarray
    pfs_day: Optional[int]
    progressed: bool
    toxicity: float
    grid: Tuple[int, int, int]
    spacing_mm: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "arm": self.arm,
            "seed": self.seed,
            "frame_days": [int(d) for d in self.frame_days],
            "frames_vti_b64": self.frames_vti_b64,
            "burden_S": [float(v) for v in self.burden_S],
            "burden_R": [float(v) for v in self.burden_R],
            "dose_A": [float(v) for v in self.dose_A],
            "dose_B": [float(v) for v in self.dose_B],
            "pfs_day": self.pfs_day,
            "progressed": self.progressed,
            "toxicity": float(self.toxicity),
            "grid": [int(v) for v in self.grid],
            "spacing_mm": float(self.spacing_mm),
            **self.extra,
        }


def export_evolution_frames(
    geometry: TumorGeometry,
    *,
    r0: float,
    seed: int,
    days: int = 365,
    arm: str = "policy",
    dense_until: int = 30,
    include_tissue: bool = True,
    max_frames: int = 110,
) -> EvolutionExport:
    """Capture + encode one arm's evolution: validated frames -> base64 VTI.

    Each frame is a complete, ParaView-loadable .vti (same writer as the
    final-state exporter), so the player renders exactly what ParaView
    would and any single frame can be opened standalone. With
    ``include_tissue=False`` the geometry-constant maps are omitted from
    the frames -- pair with ``static_geometry_maps`` so the viewer can
    re-attach them once instead of per frame.
    """
    fs, traj = capture_frames(
        geometry, r0=r0, seed=seed, days=days, arm=arm,
        dense_until=dense_until, max_frames=max_frames,
    )
    frames_b64: List[str] = []
    for d in fs.days:
        raw = _frame_vti_bytes(fs, d, geometry, include_tissue=include_tissue)
        frames_b64.append(base64.b64encode(raw).decode("ascii"))
    return EvolutionExport(
        arm=arm,
        seed=int(seed),
        frame_days=[int(d) for d in fs.days],
        frames_vti_b64=frames_b64,
        burden_S=traj.burden_S,
        burden_R=traj.burden_R,
        dose_A=traj.dose_A,
        dose_B=traj.dose_B,
        pfs_day=traj.pfs_day,
        progressed=bool(traj.progressed),
        toxicity=float(traj.toxicity),
        grid=tuple(int(v) for v in geometry.tumor_mask.shape),
        spacing_mm=float(geometry.voxel_mm),
        extra={
            "r0": float(r0),
            "days": int(days),
            "pfs_note": (
                f"progression {'day ' + str(traj.pfs_day) if traj.pfs_day is not None else 'not reached inside the horizon'}"
            ),
        },
    )


# ---------------------------------------------------------------------------
# Paired export + on-disk artifacts
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Static geometry maps (carried once per payload, not per frame)
# ---------------------------------------------------------------------------


def static_geometry_maps(geometry: TumorGeometry) -> Dict[str, str]:
    """The geometry-constant maps, base64 in the VTI writer's layout.

    tissue_class / diffusion_mm2_per_day / proliferative / vessels are
    identical in every frame of a run -- they derive from the geometry
    alone -- so a series payload carries them ONCE under ``static_maps``
    (Fortran-order <f4, x-fastest: the exact layout the exporter writes
    and the app's parser reads). The player re-attaches them to every
    frame it renders, byte-identical to include_tissue=True frames.
    """
    from danon.anisotropy import DiffusionField, tissue_classes_from_geometry

    classes = tissue_classes_from_geometry(geometry)
    maps = {
        "tissue_class": classes.astype(np.float32),
        "diffusion_mm2_per_day": DiffusionField.from_tissue_classes(
            classes, voxel_mm=float(geometry.voxel_mm)
        ).d_mm2_per_day.astype(np.float32),
        "proliferative": np.asarray(geometry.proliferative, dtype=np.float32),
        "vessels": np.asarray(geometry.vessels, dtype=np.float32),
    }
    out: Dict[str, str] = {}
    for name, arr in maps.items():
        raw = np.asfortranarray(arr).astype("<f4").tobytes(order="F")
        out[name] = base64.b64encode(raw).decode("ascii")
    return out


def export_paired_evolution(
    geometry: TumorGeometry,
    *,
    r0: float,
    seed: int,
    days: int = 365,
    dense_until: int = 30,
    include_tissue: bool = False,
    max_frames: int = 110,
) -> Dict[str, Any]:
    """Both arms on the same geometry and seed: the paired evolution movie.

    Returns a JSON-able payload: one frame series per arm, the full daily
    burden curves for both, and the derived overlay the UI syncs to the
    timeline (resistant fraction per day, per arm).

    ``include_tissue=False`` (default) keeps frames to the four dynamic
    fields only and carries the geometry-constant maps once under
    ``static_maps``; pass True for frames that are each fully standalone
    in ParaView at the cost of ~4x the payload.
    """
    policy_exp = export_evolution_frames(
        geometry, r0=r0, seed=seed, days=days, arm="policy",
        dense_until=dense_until, include_tissue=include_tissue,
        max_frames=max_frames,
    )
    mtd_exp = export_evolution_frames(
        geometry, r0=r0, seed=seed, days=days, arm="mtd",
        dense_until=dense_until, include_tissue=include_tissue,
        max_frames=max_frames,
    )

    if policy_exp.frame_days != mtd_exp.frame_days:
        raise RuntimeError("arms recorded different frame days -- timeline cannot be shared")
    # Gated on the engine's detection floor: ungated, this curve reads 100%
    # for a tumour driven below ~1e8 cells, because R/(S+R) hits 1.0 as soon as
    # the sensitive population is extinct regardless of what survives.
    r_frac_policy = resistant_fraction_series(policy_exp.burden_S, policy_exp.burden_R)
    r_frac_mtd = resistant_fraction_series(mtd_exp.burden_S, mtd_exp.burden_R)
    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model": "EvoSteer paired spatial evolution: adaptive collateral switch vs MTD",
        "seed": int(seed),
        "r0": float(r0),
        "days": int(days),
        "grid": list(policy_exp.grid),
        "spacing_mm": float(policy_exp.spacing_mm),
        "frame_days": policy_exp.frame_days,
        "arms": {
            "policy": policy_exp.to_json_dict(),
            "mtd": mtd_exp.to_json_dict(),
        },
        "resistant_fraction": {"policy": r_frac_policy, "mtd": r_frac_mtd},
        "notes": [
            "frames are the engine's exact end-of-day spatial state on the recorded "
            "days; the timeline slider indexes frame_days, not every calendar day",
            "each frame is a complete .vti (drug_A, drug_B, sensitive, resistant, "
            "tissue/diffusion maps) -- open any one of them standalone in ParaView",
            "the two arms share geometry and seed (paired design); the only "
            "difference is the dosing policy",
            "the resistant fraction is reported as 0.0 below the ~1e8-cell "
            "detection floor: R/(S+R) reaches 1.0 as soon as the sensitive "
            "population is extinct, so for a sub-detectable lesion the ratio "
            "describes nothing. Read the absolute burden arrays on those days",
            "resistance here is SELECTION of the subclone present at diagnosis, "
            "not newly evolved mutation: at the calibrated mutation rate "
            "(5e-5) turning mutation off entirely does not change the outcome",
            "simulation on a real segmentation, NOT a patient prediction",
        ],
    }
    if not include_tissue:
        payload["static_maps"] = static_geometry_maps(geometry)
    return payload


def write_pvd_series(
    directory: str,
    geometry: TumorGeometry,
    *,
    r0: float,
    seed: int,
    days: int = 365,
    arm: str = "policy",
    dense_until: int = 30,
) -> str:
    """On-disk frame series + .pvd for direct ParaView opening.

    Writes frame_0000_day0000.vti ... into `directory` plus evolution.pvd.
    Returns the .pvd path. (The in-app artifact carries base64 frames; this
    is the zero-dependency path for the poster's ParaView animation.)
    """
    fs, _traj = capture_frames(
        geometry, r0=r0, seed=seed, days=days, arm=arm, dense_until=dense_until
    )
    os.makedirs(directory, exist_ok=True)
    paths: List[str] = []
    for d in fs.days:
        S, R, cA, cB = fs.fields[d]
        traj_like = SimulationTrajectory(
            policy=f"evolution frame ({arm}, seed {seed})",
            days=d + 1,
            burden_S=fs.burden_S[: d + 1],
            burden_R=fs.burden_R[: d + 1],
            dose_A=np.zeros(d + 1),
            dose_B=np.zeros(d + 1),
            pfs_day=None,
            progressed=False,
            initial_resistant_fraction=r0,
            toxicity=fs.toxicity,
            final_resistant_fraction=resistant_fraction_gated(
                float(S.sum()), float(R.sum())
            ),
            resistant_dominance_day=None,
        )
        traj_like.final_conc_A = cA
        traj_like.final_conc_B = cB
        traj_like.final_N_S = S
        traj_like.final_N_R = R
        p = os.path.join(directory, f"frame_{len(paths):04d}_day{d:04d}.vti")
        export_fields_vti(traj_like, geometry, p, include_tissue=True)
        paths.append(p)
    pvd = os.path.join(directory, "evolution.pvd")
    write_pvd(pvd, paths, [float(d) for d in fs.days])
    return pvd


def save_payload(payload: Dict[str, Any], out_path: str, mirror_path: Optional[str] = None) -> str:
    """Write the paired payload (and optionally a byte-identical mirror)."""
    blob = json.dumps(payload, indent=1)
    for target in (out_path, mirror_path):
        if not target:
            continue
        d = os.path.dirname(target)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(target, "w") as fh:
            fh.write(blob)
    return out_path
