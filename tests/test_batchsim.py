"""The batched engine: identical per replicate, and NOT faster -- both pinned.

Two separate things are asserted here.

IDENTITY. `run_batch` exists to be `run` N times, so the gate is exact equality
against the serial loop, replicate by replicate and day by day: burden arrays,
doses, the final fields, progression day, dominance day, toxicity. This is what
lets a batch be used for a result that has to be reproducible, and it is what
makes the *default* stream mode the correct default even though it is the slow
one.

THE MEASUREMENT. The reason a batch is interesting is throughput, and on this
engine the throughput is NOT there. `outputs_real/perf.json` records it, and the
test that reads that record asserts the recorded direction -- that array
widening was slower than the serial loop, in both stream modes -- so that if the
engine's shape or the machine ever changes such that batching starts to win, the
recorded claim fails loudly instead of quietly rotting. A recorded negative
result that nothing checks is a rumour.
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

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "outputs_real" / "perf.json"

POLICIES = {
    "mtd": MTDPolicy,
    "adaptive": AdaptiveDosePolicy,
}


@pytest.fixture(scope="module")
def phantom():
    return make_gbm_phantom(size=16, seed=0)


def _specs(arm, seeds, r0=0.01):
    cls = POLICIES[arm]
    return [TrajectorySpec(cls(), seed=s, initial_resistant_fraction=r0) for s in seeds]


def _serial(geometry, arm, seeds, days, r0=0.01, keep_fields=False):
    out = []
    for spec in _specs(arm, seeds, r0):
        sim = TumorEvolutionSimulator(geometry)
        out.append(
            sim.run(
                spec.policy,
                days=days,
                initial_resistant_fraction=r0,
                seed=spec.seed,
                keep_fields=keep_fields,
            )
        )
    return out


@pytest.mark.parametrize("arm", ["mtd", "adaptive"])
def test_a_batch_reproduces_the_serial_loop_exactly(phantom, arm):
    seeds = (301, 302, 303)
    batch = TumorEvolutionSimulator(phantom).run_batch(
        _specs(arm, seeds), days=120, keep_fields=True
    )
    serial = _serial(phantom, arm, seeds, 120, keep_fields=True)
    assert [t.policy for t in batch] == [t.policy for t in serial]
    for batched, one in zip(batch, serial):
        assert np.array_equal(batched.burden_S, one.burden_S)
        assert np.array_equal(batched.burden_R, one.burden_R)
        assert np.array_equal(batched.dose_A, one.dose_A)
        assert np.array_equal(batched.dose_B, one.dose_B)
        assert batched.pfs_day == one.pfs_day
        assert batched.resistant_dominance_day == one.resistant_dominance_day
        assert batched.toxicity == one.toxicity
        assert batched.final_resistant_fraction == one.final_resistant_fraction
        assert np.array_equal(
            batched.final_N_S.view(np.uint64), one.final_N_S.view(np.uint64)
        )
        assert np.array_equal(
            batched.final_N_R.view(np.uint64), one.final_N_R.view(np.uint64)
        )
        assert np.array_equal(
            batched.final_conc_A.view(np.uint64), one.final_conc_A.view(np.uint64)
        )


def test_order_is_preserved_and_replicates_do_not_mix(phantom):
    seeds = (11, 12, 13, 14, 15, 16, 17)
    batch = TumorEvolutionSimulator(phantom).run_batch(_specs("adaptive", seeds), days=60)
    serial = _serial(phantom, "adaptive", seeds, 60)
    for i, (batched, one) in enumerate(zip(batch, serial)):
        assert np.array_equal(batched.burden_S, one.burden_S), f"replicate {i} moved"


def test_one_spec_still_works(phantom):
    batch = TumorEvolutionSimulator(phantom).run_batch(_specs("mtd", (99,)), days=30)
    serial = _serial(phantom, "mtd", (99,), 30)
    assert np.array_equal(batch[0].burden_S, serial[0].burden_S)


def test_a_stateful_policy_does_not_leak_between_replicates(phantom):
    """`reset()` must run per spec: a schedule carried across replicates would
    make the batch depend on its own ordering, which is a silent bug."""
    seeds = (21, 22)
    batch = TumorEvolutionSimulator(phantom).run_batch(_specs("adaptive", seeds), days=90)
    serial = _serial(phantom, "adaptive", seeds, 90)
    assert np.array_equal(batch[1].burden_S, serial[1].burden_S)


def test_the_batched_stream_mode_is_reproducible_but_not_the_serial_one(phantom):
    """The fast stream is allowed to be a different realisation -- not a different
    distribution, and not a different trajectory run to run."""
    a = TumorEvolutionSimulator(phantom).run_batch(
        _specs("adaptive", (31, 32)), days=90, stream="batched"
    )
    b = TumorEvolutionSimulator(phantom).run_batch(
        _specs("adaptive", (31, 32)), days=90, stream="batched"
    )
    serial = _serial(phantom, "adaptive", (31, 32), 90)
    for x, y in zip(a, b):
        assert np.array_equal(x.burden_S, y.burden_S), "must be reproducible"
    assert not np.array_equal(a[0].burden_S, serial[0].burden_S), (
        "if this ever becomes equal the two stream modes have silently merged"
    )


def test_an_unknown_stream_mode_is_refused(phantom):
    with pytest.raises(ValueError, match="stream must be"):
        TumorEvolutionSimulator(phantom).run_batch(
            _specs("mtd", (1,)), days=30, stream="whatever"
        )


def test_an_empty_batch_and_a_one_day_run_are_refused(phantom):
    sim = TumorEvolutionSimulator(phantom)
    with pytest.raises(ValueError, match="at least one spec"):
        sim.run_batch([], days=30)
    with pytest.raises(ValueError, match="at least 2 days"):
        sim.run_batch(_specs("mtd", (1,)), days=1)


def test_the_observation_hook_says_which_replicate_it_represents(phantom):
    seen = []
    TumorEvolutionSimulator(phantom).run_batch(
        _specs("mtd", (41, 42)),
        days=12,
        observation_hook=lambda i, day, bs, br: seen.append((i, day)),
    )
    assert sorted({i for i, _ in seen}) == [0, 1]
    assert len(seen) == 24


def test_heterogeneous_transport_batches_identically(phantom):
    from danon.anisotropy import DiffusionField

    field = DiffusionField.uniform(0.35, phantom.tumor_mask.shape)
    seeds = (51, 52)
    batch = TumorEvolutionSimulator(phantom, diffusion_field=field).run_batch(
        _specs("mtd", seeds), days=60, keep_fields=True
    )
    for spec in _specs("mtd", seeds):
        one = TumorEvolutionSimulator(phantom, diffusion_field=field).run(
            spec.policy, days=60, seed=spec.seed, keep_fields=True
        )
        idx = spec.seed - 51
        assert np.array_equal(batch[idx].burden_S, one.burden_S)
        assert np.array_equal(batch[idx].final_conc_A, one.final_conc_A)


def test_the_recorded_throughput_result_is_a_negative_one():
    if not RECORD.exists():
        pytest.skip(f"{RECORD} not built (run _perf_cohort.py)")
    perf = json.loads(RECORD.read_text(encoding="utf-8"))
    tp = perf["throughput"]
    assert tp["batch_per_replicate_streams"]["ratio"] < 1.0, (
        "the record says batching is faster; the module docstring of "
        "danon.evosteer.run_batch and this test both have to change with it"
    )
    assert tp["batch_one_stream"]["ratio"] < 1.0
    assert (
        tp["processes"]["by_workers"][max(tp["processes"]["by_workers"], key=int)]["ratio"]
        > 1.0
    ), "processes were recorded as doing nothing either"


def test_the_identity_claims_in_the_record_are_in_the_record():
    if not RECORD.exists():
        pytest.skip(f"{RECORD} not built (run _perf_cohort.py)")
    perf = json.loads(RECORD.read_text(encoding="utf-8"))
    ident = perf["bit_identity"]
    assert all(row["exact_equal"] for row in ident["fields"])
    for label in ("scalar", "heterogeneous"):
        row = ident["trajectory"][label]
        for key in (
            "burden_S_exact",
            "burden_R_exact",
            "dose_A_exact",
            "pfs_day_equal",
            "toxicity_equal",
            "final_resistant_fraction_equal",
            "dominance_day_equal",
            "final_N_S_bitwise",
            "final_conc_A_bitwise",
        ):
            assert row[key] is True, f"{label}: {key} is recorded as False"
