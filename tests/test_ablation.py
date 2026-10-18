"""Tests for the ablation harness, including planted controls in both directions.

Two of these tests are the ones that make the harness trustworthy:

* PLANTED INERT. `MTDPolicy` doses `(1.0, 0.0)` -- drug B is never administered.
  Ablating `second_agent` under the MTD arm therefore cannot change a single
  trajectory, so every readout delta must be EXACTLY 0.0 and the verdict must be
  `inert` (or `degenerate` where the intact arm had no spread at all). A harness
  that reported an effect here would be reporting noise as mechanism.
* PLANTED LOAD-BEARING. Cell motility is the only term that moves cells outwards,
  so ablating it must move the day-N spatial burden, and the harness must say so
  with a ratio above 1.

The refusals are pinned as strictly: "turn off the heterogeneous transport field"
on a geometry that has no field is a different experiment, and it is refused
rather than run against a substituted homogeneous one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from danon.ablation import (
    INERT_RATIO,
    LOAD_BEARING_RATIO,
    MECHANISMS,
    READOUTS,
    AblationError,
    Mechanism,
    ablate_all,
    ablation_contrast,
    ablation_provenance,
    what_an_ablation_cannot_show,
)
from danon.anisotropy import DiffusionField
from danon.cure import EvidenceStatus
from danon.evosteer import make_gbm_phantom

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "outputs_real" / "ablation_cohort.json"


@pytest.fixture(scope="module")
def phantom():
    return make_gbm_phantom(size=16, seed=0)


def test_a_mechanism_the_policy_never_uses_is_measurably_inert(phantom):
    """MTD never doses drug B, so removing drug B must change nothing at all."""
    report = ablation_contrast(
        MECHANISMS["second_agent"],
        phantom,
        arms=("mtd",),
        seeds=(101, 102, 103),
        days=90,
    )
    entry = report["arms"]["mtd"]
    for readout, values in entry["readouts"].items():
        if values["delta"] is not None:
            assert values["delta"] == 0.0, readout
        assert values["verdict"] in ("inert", "degenerate"), (readout, values)
    assert report["any_load_bearing"] is False
    assert entry["load_bearing_on"] == []
    assert report["load_bearing_on"] == []


def test_a_mechanism_that_moves_the_tumour_outwards_is_measurably_load_bearing(phantom):
    report = ablation_contrast(
        MECHANISMS["cell_motility"],
        phantom,
        arms=("mtd",),
        seeds=(101, 102, 103),
        days=90,
    )
    density = report["arms"]["mtd"]["readouts"]["mean_density"]
    assert density["verdict"] == "load_bearing"
    assert density["delta_over_spread"] > LOAD_BEARING_RATIO
    # Freezing motility must LOSE burden outside the mask, so the ablated arm's
    # mean density is the lower one.
    assert density["delta"] < 0.0
    assert report["any_load_bearing"] is True
    assert "mtd/mean_density" in report["load_bearing_on"]


def test_the_recorded_arithmetic_is_the_arithmetic_it_reports(phantom):
    report = ablation_contrast(
        MECHANISMS["resistant_kill_ceiling"],
        phantom,
        seeds=(101, 102),
        days=90,
    )
    for arm, entry in report["arms"].items():
        assert set(entry["readouts"]) == set(READOUTS)
        for readout, values in entry["readouts"].items():
            intact = np.asarray(
                [v for v in entry["intact"]["readouts"][readout] if v is not None],
                dtype=float,
            )
            ablated = np.asarray(
                [v for v in entry["ablated"]["readouts"][readout] if v is not None],
                dtype=float,
            )
            if intact.size and ablated.size:
                assert values["intact_mean"] == pytest.approx(float(intact.mean()))
                assert values["ablated_mean"] == pytest.approx(float(ablated.mean()))
                assert values["delta"] == pytest.approx(
                    values["ablated_mean"] - values["intact_mean"]
                )
            if not values["intact_spread_degenerate"] and values["rel_delta"]:
                assert values["delta_over_spread"] == pytest.approx(
                    abs(values["rel_delta"]) / values["intact_spread_rel"]
                )
                ratio = values["delta_over_spread"]
                if ratio >= LOAD_BEARING_RATIO:
                    assert values["verdict"] == "load_bearing"
                elif ratio < INERT_RATIO:
                    assert values["verdict"] == "inert"
                else:
                    assert values["verdict"] == "unresolved"
            else:
                assert values["delta_over_spread"] is None
                assert values["verdict"] == "degenerate"
        wants = sorted(
            r for r, v in entry["readouts"].items() if v["verdict"] == "load_bearing"
        )
        assert entry["load_bearing_on"] == wants
    assert report["arms"]["mtd"]["intact"]["voxels"] == int(
        phantom.tumor_mask.sum()
    )


def test_a_transport_ablation_without_a_transport_field_is_refused(phantom):
    with pytest.raises(AblationError, match="never a field"):
        ablation_contrast(
            MECHANISMS["heterogeneous_transport"], phantom, days=20, seeds=(101, 102)
        )
    out = ablate_all(phantom, seeds=(101, 102), days=20)
    assert "heterogeneous_transport" in out["skipped"]
    assert "heterogeneous_transport" not in out["mechanisms"]
    assert "never a field" in out["skipped"]["heterogeneous_transport"]


def test_with_a_field_present_every_mechanism_runs(phantom):
    field = DiffusionField.uniform(0.35, phantom.tumor_mask.shape, voxel_mm=1.0)
    out = ablate_all(phantom, seeds=(101, 102), days=30, diffusion_field=field)
    assert out["skipped"] == {}
    assert set(out["mechanisms"]) == set(MECHANISMS)
    transport = out["mechanisms"]["heterogeneous_transport"]
    # A UNIFORM field is mean-one after the simulator's normalisation, so the
    # homogeneous operator is recovered and removing the field removes nothing: the
    # verdict must not be load-bearing. That is the control showing the contrast
    # measures spatial variation rather than merely the code path. The comparison
    # is made on the RATIO, not on bit equality, because the two transport branches
    # sum in a different order (~1e-16 relative, pinned by test_benchmarks).
    for values in transport["arms"]["mtd"]["readouts"].values():
        assert values["verdict"] in ("inert", "degenerate"), values
    assert transport["any_load_bearing"] is False


def test_a_field_of_zeros_is_refused_so_the_absent_case_is_explicit(phantom):
    """The engine will not accept a zero field, which is why `requires` exists.

    "There was never a field" and "the field was removed" are different
    experiments, and the engine's own mean-zero refusal is what keeps the second
    one from being reached by accident.
    """
    from danon.evosteer import TumorEvolutionSimulator

    flat = DiffusionField.uniform(0.0, phantom.tumor_mask.shape, voxel_mm=1.0)
    with pytest.raises(ValueError, match="zero mean"):
        TumorEvolutionSimulator(phantom, diffusion_field=flat)


def test_mechanism_construction_refuses_an_unknown_target():
    with pytest.raises(AblationError, match="target must be"):
        Mechanism(
            name="bogus",
            removes="nothing",
            target="telepathy",
            patch=lambda x: x,
            intact_note="nothing",
        )


def test_unknown_mechanism_names_are_refused_with_the_known_ones(phantom):
    with pytest.raises(AblationError, match="unknown mechanism"):
        ablate_all(phantom, mechanisms=["not_a_mechanism"], days=10, seeds=(101, 102))


def test_provenance_and_refusals_state_that_an_ablation_is_not_validation():
    prov = ablation_provenance()
    assert prov["mechanism_is_load_bearing"] is EvidenceStatus.ESTABLISHED
    assert prov["mechanism_is_correct"] is EvidenceStatus.PROVEN_FAILED
    assert prov["ablation_is_validation"] is EvidenceStatus.PROVEN_FAILED
    assert prov["white_matter_tract_anisotropy"] is EvidenceStatus.PROVEN_FAILED
    refusals = what_an_ablation_cannot_show()
    assert any("correctness" in r for r in refusals)
    assert any("tensor" in r for r in refusals)


def test_the_committed_cohort_record_recomputes_its_leaderboard():
    if not RECORD.exists():
        pytest.skip("run _ablation_cohort.py to build the real-case cohort")
    with RECORD.open(encoding="utf-8") as fh:
        record = json.load(fh)
    assert record["case"] == "BraTS2021_01017"
    assert record["skipped"] == {}, "the real case carries a transport field"
    assert set(record["mechanisms"]) == set(MECHANISMS)
    assert record["transport_field_contrast"] > 1.0
    counted = sorted(
        (
            (name, len(report["load_bearing_on"]))
            for name, report in record["mechanisms"].items()
        ),
        key=lambda kv: -kv[1],
    )
    assert record["leaderboard"] == [
        {"mechanism": name, "n_load_bearing_readouts": n} for name, n in counted
    ]
    for report in record["mechanisms"].values():
        for arm, entry in report["arms"].items():
            for readout, values in entry["readouts"].items():
                ratio = values["delta_over_spread"]
                if ratio is None:
                    assert values["verdict"] == "degenerate"
                elif ratio >= LOAD_BEARING_RATIO:
                    assert values["verdict"] == "load_bearing"
                elif ratio < INERT_RATIO:
                    assert values["verdict"] == "inert"
                else:
                    assert values["verdict"] == "unresolved"
