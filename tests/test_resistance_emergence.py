"""Falsification tests for the resistance mechanism.

These exist because the suite previously verified that resistance *happens*
without verifying that the mechanism *causes* it. It did not:

- The clustered seed always placed a resistant clone in at least nine voxels,
  so a "1e-9" initial fraction still seeded ~9 viable cells. Resistance was
  therefore present at t=0 in every run, and mutation_rate did no work: mu=0
  and mu=5e-5 produced byte-identical outcomes.
- Continuous densities have no extinction, so a sub-single-cell remnant
  (10^-9 of a voxel) grew deterministically back to dominance. Combined with
  a resistant fraction computed as R/(S+R), the model reported "resistant
  dominance on day 87" for a tumour whose entire remaining burden was 0.016
  of one cell -- and no policy acting on that number could tell the
  difference between a tumour that resisted and one that had been
  eradicated.

Each test below pins one of those failure modes shut. The load-bearing one is
`test_mutation_rate_controls_the_outcome`: if it fails, resistance in this
engine is an initial condition or an arithmetic artifact, not an evolved
property of the model.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from danon.evosteer import (
    CELLS_PER_VOXEL_AT_UNIT_DENSITY,
    DEFAULT_CLONES,
    DETECTION_BURDEN_DENSITY,
    MIN_VIABLE_DENSITY,
    MTDPolicy,
    TumorEvolutionSimulator,
    make_gbm_phantom,
)

NO_MUTATION = dataclasses.replace(DEFAULT_CLONES, mutation_rate=0.0)
HIGH_MUTATION = dataclasses.replace(DEFAULT_CLONES, mutation_rate=5e-3)


@pytest.fixture(scope="module")
def geom():
    return make_gbm_phantom()


def _mtd(geom, clones, *, ifr, days=365, seed=101):
    return TumorEvolutionSimulator(geom, clones=clones).run(
        MTDPolicy(), days=days, initial_resistant_fraction=ifr, seed=seed
    )


def test_no_seed_and_no_mutation_means_no_resistance(geom):
    """With no resistant clone seeded and mutation switched off, no resistant
    cell may appear. This is the test that failed before the extinction floor:
    resistance still reached 100% dominance."""
    traj = _mtd(geom, NO_MUTATION, ifr=0.0)
    assert np.all(traj.burden_R == 0.0), (
        "resistant cells appeared with mutation_rate=0 and no seeded clone: "
        "resistance is not coming from the mechanism"
    )
    assert traj.final_resistant_fraction == 0.0
    assert traj.resistant_dominance_day is None


def test_resistance_is_created_by_mutation_alone(geom):
    """Raised mutation rate, still no seeded clone: resistance must EMERGE."""
    traj = _mtd(geom, HIGH_MUTATION, ifr=0.0)
    assert traj.burden_R.max() > 0.0, "mutation never produced a resistant cell"
    assert traj.resistant_dominance_day is not None, (
        "mutation created resistant cells but they never reached dominance"
    )


def test_mutation_rate_controls_the_outcome(geom):
    """The load-bearing invariant: the mutation rate must change the result.

    Before the fix, mu=0 and mu=5e-5 gave identical PFS and dominance days,
    which is only possible if mutation is not the source of resistance."""
    zero = _mtd(geom, NO_MUTATION, ifr=0.0)
    high = _mtd(geom, HIGH_MUTATION, ifr=0.0)
    assert (zero.pfs_day, zero.final_resistant_fraction) != (
        high.pfs_day,
        high.final_resistant_fraction,
    ), "mutation_rate has no effect on the outcome"


def test_sub_cell_populations_are_extinguished(geom):
    """A voxel holding less than half a cell holds no cells, and a tumour
    pushed below one cell goes to exactly zero rather than lingering as a
    density that can regrow."""
    traj = _mtd(geom, NO_MUTATION, ifr=0.0, days=200)
    for burden in (traj.burden_S, traj.burden_R):
        live = burden[burden > 0.0]
        assert np.all(live >= MIN_VIABLE_DENSITY), (
            "a sub-cell population survived as a positive density"
        )
    # Eradication must be exact, not asymptotic: with no resistant clone and
    # no mutation the drug kills the whole sensitive population.
    assert traj.burden_S[-1] == 0.0, (
        f"sensitive population lingered at {traj.burden_S[-1]:.3e} instead of "
        "being extinguished"
    )


def test_dominance_is_never_declared_below_the_detection_floor(geom):
    """R/(S+R) is only a clinical quantity for a detectable lesion."""
    for ifr in (0.005, 1e-9):
        traj = _mtd(geom, DEFAULT_CLONES, ifr=ifr)
        if traj.resistant_dominance_day is None:
            continue
        d = traj.resistant_dominance_day
        burden = float(traj.burden_S[d] + traj.burden_R[d])
        assert burden >= DETECTION_BURDEN_DENSITY, (
            f"resistant dominance declared on day {d} at a burden of "
            f"{burden * CELLS_PER_VOXEL_AT_UNIT_DENSITY:.3e} cells, below the "
            "1e8-cell detection floor"
        )


def test_pre_existing_resistance_is_what_makes_mtd_fail(geom):
    """The engine's qualitative conclusion depends on an ASSUMED input.

    Same drug, same geometry, same seed; the only difference is whether a
    resistant subclone exists at diagnosis. This is the result the UI must
    present, because it is the result the model actually produces.
    """
    seeded = _mtd(geom, DEFAULT_CLONES, ifr=0.005)
    unseeded = _mtd(geom, DEFAULT_CLONES, ifr=0.0)
    assert seeded.pfs_day is not None, "a 0.5% pre-existing clone should resist MTD"
    assert unseeded.pfs_day is None, "with no resistance, MTD should control the tumour"
    seeded_total = float(seeded.burden_S[-1] + seeded.burden_R[-1])
    unseeded_total = float(unseeded.burden_S[-1] + unseeded.burden_R[-1])
    assert unseeded_total < seeded_total, (
        "the pre-existing clone should leave more tumour behind than the run "
        "in which no resistant clone exists at all"
    )
