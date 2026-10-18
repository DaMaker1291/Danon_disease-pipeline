"""Uncertainty quantification throughput, by parallelism that cannot change a number.

WHY THIS EXISTS
---------------
A 95% interval over the engine's stochastic parameters needs thousands of
replicates, and `outputs_real/perf.json` records what the two obvious answers
actually cost on the committed BraTS case:

  * BATCHING the day loop over a leading replicate axis (the `run_batch` path,
    the numpy version of "vectorise the Monte Carlo") is a measured SLOWDOWN:
    at 32 replicates x 60 days it ran at 0.69x the serial rate with
    per-replicate random streams and 0.67x with one batched stream draw, so it
    is not the RNG that prevents it. An 8,064-voxel field is ~64 KB; forty
    numpy calls a day on it are dominated by per-call cost, and a batch large
    enough to amortise them stops fitting in cache. The measurement is in the
    artifact; the conclusion is that this engine cannot be made faster by
    widening its arrays.
  * RUNNING THE REPLICATES IN PARALLEL is free of numerical consequence,
    because every worker runs the unmodified serial engine on its own seeds.
    That is what this module does.

WHAT IT COSTS AND WHAT IT CANNOT DO
-----------------------------------
It multiplies throughput by the number of cores actually idle -- measured, not
assumed, and recorded with the machine's core count beside it. On the shared
development machine that number was far below the core count, which is exactly
the kind of thing a claim about "10,000 draws in N minutes" has to carry. A
single trajectory is never accelerated by this: it is one core's work, and the
only thing that makes IT faster is a faster kernel (`danon.accel`).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.evosteer import (
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    CloneParams,
    DrugParams,
    SimulationTrajectory,
    TrajectorySpec,
    TumorEvolutionSimulator,
    TumorGeometry,
)


@dataclass(frozen=True)
class ReplicatePlan:
    """Everything a worker needs to run a block of replicates.

    Frozen and array-only on purpose: on Windows the default start method is
    'spawn', so this object is PICKLED to each worker. A closure, a local
    function or an unpicklable policy would fail there rather than here, so the
    plan carries plain data and rebuilds the simulator inside the worker.
    """

    geometry: TumorGeometry
    specs: Tuple[TrajectorySpec, ...]
    days: int
    growth_noise: float = 0.05
    clones: CloneParams = DEFAULT_CLONES
    drugs: DrugParams = DEFAULT_DRUGS
    cell_motility: Optional[float] = None
    #: The raw `d_engine` array, not the `DiffusionField` wrapper: an array is
    #: trivially picklable and the simulator accepts either.
    field_d_engine: Optional[np.ndarray] = None

    @classmethod
    def build(
        cls,
        geometry: TumorGeometry,
        specs: Sequence[TrajectorySpec],
        days: int,
        *,
        growth_noise: float = 0.05,
        clones: CloneParams = DEFAULT_CLONES,
        drugs: DrugParams = DEFAULT_DRUGS,
        cell_motility: Optional[float] = None,
        diffusion_field: Optional[object] = None,
    ) -> "ReplicatePlan":
        d_engine = None
        if diffusion_field is not None:
            d_engine = np.asarray(
                np.array(getattr(diffusion_field, "d_engine", diffusion_field)),
                dtype=float,
            )
        return cls(
            geometry=geometry,
            specs=tuple(specs),
            days=int(days),
            growth_noise=float(growth_noise),
            clones=clones,
            drugs=drugs,
            cell_motility=cell_motility,
            field_d_engine=d_engine,
        )


@dataclass
class UQResult:
    trajectories: List[SimulationTrajectory]
    seeds: List[int]
    policies: List[str]
    workers: int
    chunks: int

    def to_summary(self) -> Dict[str, Any]:
        """Per-arm distribution summaries -- the reason to run replicates at all."""
        by_arm: Dict[str, List[SimulationTrajectory]] = {}
        for traj in self.trajectories:
            by_arm.setdefault(traj.policy, []).append(traj)
        out: Dict[str, Any] = {}
        for arm, trajs in by_arm.items():
            pfs = np.array([t.pfs_value(t.days) for t in trajs], dtype=float)
            out[arm] = {
                "n": len(trajs),
                "median_pfs_days": float(np.median(pfs)),
                "pfs_ci95": [float(np.percentile(pfs, 2.5)), float(np.percentile(pfs, 97.5))],
                "median_toxicity": float(np.median([t.toxicity for t in trajs])),
                "notes": (
                    "a distribution over the engine's OWN stochastic terms at "
                    "fixed parameters; it is not a clinical confidence interval"
                ),
            }
        return out


def _run_block(payload: Tuple[ReplicatePlan, List[int]]) -> List[SimulationTrajectory]:
    """Run the replicate indices in `payload[1]`. Module level: must be picklable."""
    plan, indices = payload
    sim = TumorEvolutionSimulator(
        plan.geometry,
        clones=plan.clones,
        drugs=plan.drugs,
        diffusion_field=plan.field_d_engine,
        cell_motility=plan.cell_motility,
    )
    out: List[SimulationTrajectory] = []
    for i in indices:
        spec = plan.specs[i]
        out.append(
            sim.run(
                spec.policy,
                days=plan.days,
                initial_resistant_fraction=spec.initial_resistant_fraction,
                seed=int(spec.seed),
                growth_noise=plan.growth_noise,
                fill_fraction=spec.fill_fraction,
            )
        )
    return out


def _chunk(n: int, chunks: int) -> List[List[int]]:
    return [list(range(c, n, chunks)) for c in range(min(chunks, n))]


def run_replicates(
    plan: ReplicatePlan,
    *,
    workers: Optional[int] = None,
    chunks: Optional[int] = None,
    serial: bool = False,
) -> UQResult:
    """Run every replicate in `plan`, in parallel, IN ITS ORIGINAL ORDER.

    Each trajectory is produced by the unmodified `TumorEvolutionSimulator.run`
    with its own seed, so parallelism cannot change a result: the returned list
    is identical, element for element, to the serial loop over the same specs
    (`tests/test_uq.py` compares whole trajectories). What the chunking exists
    for is that a sparse index assignment keeps the output order stable while
    still letting the slowest block finish last.

    `workers=1` or `serial=True` runs in-process, which is what the speedup is
    measured against rather than assumed from a core count.
    """
    n = len(plan.specs)
    if n == 0:
        raise ValueError("plan has no specs")
    if serial or workers == 1:
        results = {0: _run_block((plan, list(range(n))))}
        return UQResult(
            trajectories=results[0],
            seeds=[int(s.seed) for s in plan.specs],
            policies=[s.policy.describe() for s in plan.specs],
            workers=1,
            chunks=1,
        )

    want = workers if workers is not None else (os.cpu_count() or 1)
    want = max(1, min(int(want), n))
    n_chunks = int(chunks) if chunks is not None else min(n, want * 2)
    blocks = _chunk(n, max(1, n_chunks))
    payloads = [(plan, block) for block in blocks]
    results: Dict[int, List[SimulationTrajectory]] = {}
    if n_chunks == 1:
        results[0] = _run_block(payloads[0])
    else:
        with ProcessPoolExecutor(max_workers=want) as pool:
            for block, trajs in zip(blocks, pool.map(_run_block, payloads)):
                results[block[0]] = trajs
    # `_owner` maps a replicate index to the block that holds it, so the output
    # keeps the plan's order rather than the order the workers happened to
    # finish in.
    out: List[SimulationTrajectory] = []
    cursor = {block[0]: 0 for block in blocks}
    for i in range(n):
        start = _owner(i, blocks)
        out.append(results[start][cursor[start]])
        cursor[start] += 1
    return UQResult(
        trajectories=out,
        seeds=[int(s.seed) for s in plan.specs],
        policies=[s.policy.describe() for s in plan.specs],
        workers=want,
        chunks=len(blocks),
    )


def _owner(index: int, blocks: List[List[int]]) -> int:
    for block in blocks:
        if index in block:
            return block[0]
    raise KeyError(index)
