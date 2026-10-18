"""The parallel replicate runner: same numbers, in a fraction of the wall clock.

Parallelism is the one speed-up in this work that CANNOT change a result, because
every worker runs the unmodified serial engine with its own seed. That is not a
reason to skip the test: it is the reason the test is an exact comparison rather
than a tolerance. A worker that picked up its neighbour's spec, a plan that
pickled the wrong field, or a chunk boundary that dropped a replicate all produce
plausible-looking output, and all three are caught here by comparing whole
trajectories element for element.

The runner is also asserted to preserve the plan's ORDER, and to refuse an empty
plan, because a UQ summary is a distribution over seeds and a silently reordered
or truncated list of seeds is a changed experiment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from danon.evosteer import (
    AdaptiveDosePolicy,
    MTDPolicy,
    TrajectorySpec,
    TumorEvolutionSimulator,
    make_gbm_phantom,
)
from danon.uq import ReplicatePlan, run_replicates

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "outputs_real" / "perf.json"


@pytest.fixture(scope="module")
def phantom():
    return make_gbm_phantom(size=16, seed=0)


def _specs(n, seed0=4000):
    return [
        TrajectorySpec(AdaptiveDosePolicy(), seed=seed0 + i, initial_resistant_fraction=0.01)
        for i in range(n)
    ]


def test_parallel_replicates_are_exactly_the_serial_ones(phantom):
    specs = _specs(8)
    plan = ReplicatePlan.build(phantom, specs, days=60)
    serial = run_replicates(plan, serial=True)
    parallel = run_replicates(plan, workers=2)
    assert parallel.seeds == serial.seeds
    assert parallel.policies == serial.policies
    for a, b in zip(parallel.trajectories, serial.trajectories):
        assert np.array_equal(a.burden_S, b.burden_S)
        assert np.array_equal(a.burden_R, b.burden_R)
        assert np.array_equal(a.dose_A, b.dose_A)
        assert a.pfs_day == b.pfs_day
        assert a.toxicity == b.toxicity
        assert a.resistant_dominance_day == b.resistant_dominance_day


def test_the_plan_order_survives_the_chunking(phantom):
    """Chunk sizes that do not divide the replicate count are the interesting case."""
    specs = _specs(7)
    plan = ReplicatePlan.build(phantom, specs, days=40)
    out = run_replicates(plan, workers=3, chunks=3)
    for i, spec in enumerate(specs):
        assert out.trajectories[i].initial_resistant_fraction == spec.initial_resistant_fraction
    serial = run_replicates(plan, serial=True)
    for a, b in zip(out.trajectories, serial.trajectories):
        assert np.array_equal(a.burden_S, b.burden_S), "order is part of the result"


def test_an_empty_plan_is_refused(phantom):
    plan = ReplicatePlan.build(phantom, [], days=30)
    with pytest.raises(ValueError, match="no specs"):
        run_replicates(plan, workers=2)


def test_a_mixed_arm_plan_keeps_each_arm_on_its_own_policy(phantom):
    specs = [
        TrajectorySpec(MTDPolicy(), seed=900),
        TrajectorySpec(AdaptiveDosePolicy(), seed=900),
        TrajectorySpec(MTDPolicy(), seed=901),
    ]
    plan = ReplicatePlan.build(phantom, specs, days=60)
    out = run_replicates(plan, workers=2)
    arms = [t.policy for t in out.trajectories]
    assert arms[0] == arms[2] != arms[1]
    assert out.policies == arms


def test_the_summary_reports_distributions_not_points(phantom):
    plan = ReplicatePlan.build(phantom, _specs(6), days=60)
    summary = run_replicates(plan, serial=True).to_summary()
    assert len(summary) == 1
    arm = next(iter(summary.values()))
    assert arm["n"] == 6
    lo, hi = arm["pfs_ci95"]
    assert lo <= arm["median_pfs_days"] <= hi
    assert "not a clinical confidence interval" in arm["notes"]


def test_a_field_is_carried_to_the_workers(phantom):
    from danon.anisotropy import DiffusionField

    field = DiffusionField.uniform(0.35, phantom.tumor_mask.shape)
    specs = [TrajectorySpec(MTDPolicy(), seed=700)]
    plan = ReplicatePlan.build(phantom, specs, days=40, diffusion_field=field)
    assert plan.field_d_engine is not None
    got = run_replicates(plan, workers=1).trajectories[0]
    one = TumorEvolutionSimulator(phantom, diffusion_field=field).run(
        MTDPolicy(), days=40, seed=700
    )
    assert np.array_equal(got.burden_S, one.burden_S)
    assert not np.array_equal(
        got.burden_S,
        TumorEvolutionSimulator(phantom).run(MTDPolicy(), days=40, seed=700).burden_S,
    ), "the field must have reached the worker, or this plan is a different model"


def test_the_recorded_process_speedup_is_positive_and_bounded_by_the_cores():
    if not RECORD.exists():
        pytest.skip(f"{RECORD} not built (run _perf_cohort.py)")
    perf = json.loads(RECORD.read_text(encoding="utf-8"))
    tp = perf["throughput"]
    cores = tp["machine_cores"]
    assert cores and cores > 1
    for workers, entry in tp["processes"]["by_workers"].items():
        assert entry["ratio"] > 1.0, f"{workers} workers were recorded as no faster"
        assert entry["ratio"] <= int(workers) + 0.5, (
            "a speed-up above the worker count is not a speed-up"
        )
