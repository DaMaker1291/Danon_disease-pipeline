"""Tests for the committed sample dataset and the config-driven paired trial.

These pin the two things a stranger has to be able to trust:

- **The committed dataset is the generator's own output.** Regenerating the
  sample into a temporary directory must produce bytes identical to what is in
  `data/sample_bids/`, so the committed files cannot drift from the phantom
  they claim to be, and a hand-edited volume fails here rather than quietly
  becoming the input every reviewer runs.
- **The round trip through uint8 NIfTI does not silently change the physics.**
  The mask and proliferation map are discrete so they must survive exactly; the
  vessel map is quantised to 1/255 and must not, so the trial outcome is
  MEASURED against the in-memory phantom and asserted at the measured bound
  rather than assumed equal.

The refusals are pinned too, because a config parser that accepts an empty seed
list hands back a median of nothing, which reads like a null result.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

from danon.evosteer import MTDPolicy, TumorEvolutionSimulator, make_gbm_phantom
from danon.imaging import from_nifti_mask
from danon.reproduce import (
    ARM_POLICIES,
    SAMPLE_ROOT,
    SAMPLE_SEEDS,
    SAMPLE_SUBJECT,
    build_sample_dataset,
    format_report,
    load_config,
    run_config,
    sample_config_path,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / SAMPLE_ROOT

#: The loader quantises a [0, 1] map to k/255, so the worst per-voxel error of
#: the round trip is half a step. Stated as the bound it is, not as a tolerance
#: chosen to make the test pass.
QUANTUM = 0.5 / 255.0


# ---------------------------------------------------------------------------
# The committed dataset is the generator's output
# ---------------------------------------------------------------------------


def test_committed_sample_is_byte_identical_to_a_fresh_build(tmp_path: Path):
    """Anti-drift pin: the dataset on disk is exactly what the builder writes."""
    rebuild = tmp_path / "sample"
    build_sample_dataset(str(rebuild))

    committed = sorted(p.relative_to(SAMPLE_DIR) for p in SAMPLE_DIR.rglob("*") if p.is_file())
    fresh = sorted(p.relative_to(rebuild) for p in rebuild.rglob("*") if p.is_file())
    assert committed == fresh, "the committed sample and a fresh build hold different files"

    for relative in committed:
        assert (SAMPLE_DIR / relative).read_bytes() == (rebuild / relative).read_bytes(), (
            f"{relative} differs from what build_sample_dataset produces; regenerate it "
            f"with `python -m danon.reproduce build-sample`"
        )


def test_sample_config_names_only_files_that_exist_and_one_subject():
    spec = load_config(str(SAMPLE_DIR / f"{SAMPLE_SUBJECT}" / f"{SAMPLE_SUBJECT}_twin.json"))
    for role, path in spec["inputs"].items():
        assert path is None or Path(path).exists(), f"{role} points at {path}"
    assert spec["subject"] == SAMPLE_SUBJECT
    assert spec["inputs"]["mask"] is not None
    assert spec["seeds"] == SAMPLE_SEEDS
    assert set(spec["arms"]) <= set(ARM_POLICIES)


def test_dataset_description_says_it_is_not_patient_data():
    """The honest label is part of the artifact, not only of the docstring."""
    description = json.loads((SAMPLE_DIR / "dataset_description.json").read_text(encoding="utf-8"))
    assert description["Synthetic"] is True
    caveat = description["Caveat"].lower()
    assert "not patient data" in caveat
    assert "not a bids-conformant" in caveat


# ---------------------------------------------------------------------------
# The round trip: what survives exactly, and what is measured
# ---------------------------------------------------------------------------


def test_round_trip_preserves_the_discrete_maps_exactly():
    """Mask and proliferation are discrete, so uint8 must lose nothing."""
    phantom = make_gbm_phantom(size=24, seed=0)
    page = from_nifti_mask(
        str(SAMPLE_DIR / SAMPLE_SUBJECT / f"{SAMPLE_SUBJECT}_mask.nii.gz"),
        proliferative_path=str(SAMPLE_DIR / SAMPLE_SUBJECT / "anat" / f"{SAMPLE_SUBJECT}_prolif.nii.gz"),
        vessels_path=str(SAMPLE_DIR / SAMPLE_SUBJECT / "perf" / f"{SAMPLE_SUBJECT}_vessels.nii.gz"),
    )
    assert np.array_equal(phantom.tumor_mask, page.geometry.tumor_mask)
    assert float(np.abs(phantom.proliferative - page.geometry.proliferative).max()) == 0.0
    # The vessel map is continuous, so it is the one that quantises -- and the
    # loader's own max-normalisation is what sets the step.
    assert float(np.abs(phantom.vessels - page.geometry.vessels).max()) <= QUANTUM
    assert page.proliferative_source == "file"
    assert page.vasculature_source == "file"


def test_quantisation_does_not_move_the_paired_trial_measurably():
    """The committed files' outcome is measured against the in-memory phantom.

    A single MTD replicate on each geometry, same seed: the progression day must
    be the same day, and the end-of-horizon burden must agree to well inside the
    run's own seed-to-seed spread -- otherwise the committed dataset would be a
    different experiment from the one the engine documents.
    """
    phantom = make_gbm_phantom(size=24, seed=0)
    spec = load_config(sample_config_path(str(SAMPLE_DIR)))
    page = from_nifti_mask(
        spec["inputs"]["mask"],
        proliferative_path=spec["inputs"]["proliferative"],
        vessels_path=spec["inputs"]["vessels"],
    )
    outcomes = []
    for geometry in (phantom, page.geometry):
        traj = TumorEvolutionSimulator(geometry).run(
            MTDPolicy(), days=365, initial_resistant_fraction=0.005, seed=101
        )
        outcomes.append(
            (traj.pfs_value(365), float(traj.burden_S[-1] + traj.burden_R[-1]))
        )
    assert outcomes[0][0] == outcomes[1][0], f"PFS moved: {outcomes}"
    assert math.isclose(outcomes[0][1], outcomes[1][1], rel_tol=0.005), outcomes


# ---------------------------------------------------------------------------
# The run: paired, labelled, and reproducible from the printed line
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sample_report():
    return run_config(sample_config_path(str(SAMPLE_DIR)))


def test_report_pairs_every_arm_by_seed_in_order(sample_report):
    report = sample_report
    arms = report["arms"]
    assert set(arms) == set(report["trial"]["arms"])
    for arm in arms:
        assert [r["seed"] for r in arms[arm]["runs"]] == list(SAMPLE_SEEDS)

    first, second = report["paired"]["arms"]
    expected = [
        b["pfs_days"] - a["pfs_days"]
        for a, b in zip(arms[first]["runs"], arms[second]["runs"])
    ]
    assert report["paired"]["pfs_delta_days"] == expected
    assert report["paired"]["n_pairs_in_disagreement"] == sum(d != 0 for d in expected)


def test_report_labels_the_run_synthetic_and_repeats_the_caveat(sample_report):
    assert sample_report["synthetic"] is True
    assert sample_report["caveat"]
    text = format_report(sample_report)
    assert "SYNTHETIC PHANTOM" in text and "not patient data" in text


def test_sample_trial_separates_the_two_policies(sample_report):
    """The result the sample exists to show, at the resolution of its own spread.

    Not a claim about a patient: the phantom is synthetic, so what is pinned is
    that the harness reproduces the engine's own qualitative contrast -- the
    continuous-dose arm reaches progression while the switching arm does not --
    and that the contrast is larger than the run's own seed-to-seed spread,
    which is the only thing that makes it readable at all.
    """
    mtd, steering = sample_report["arms"]["mtd"], sample_report["arms"]["steering"]
    assert mtd["median_pfs_days"] < 365.0
    assert all(r["progressed"] for r in mtd["runs"])
    assert all(not r["progressed"] for r in steering["runs"])
    assert sample_report["paired"]["n_pairs_in_disagreement"] == len(SAMPLE_SEEDS)

    delta_days = sample_report["paired"]["median_pfs_delta_days"]
    spread_days = max(r["pfs_days"] for r in mtd["runs"]) - min(
        r["pfs_days"] for r in mtd["runs"]
    )
    assert delta_days > spread_days, (
        f"the paired delta ({delta_days} d) does not clear the arm's own "
        f"seed-to-seed spread ({spread_days} d)"
    )


def test_report_json_on_disk_is_the_returned_report(tmp_path: Path):
    """Short horizon: this pins the writer, not the trial."""
    config = _valid(tmp_path)
    config["synthetic"] = True
    report = run_config(_write(tmp_path, config), out_file=str(tmp_path / "report.json"))
    assert json.loads((tmp_path / "report.json").read_text(encoding="utf-8")) == report
    assert report["how_to_reproduce"].endswith(report["config"])


def test_report_paths_are_repo_relative_not_one_machines(sample_report):
    """A committed report must be followable by whoever clones the repository."""
    assert sample_report["config"] == "data/sample_bids/sub-01/sub-01_twin.json"
    assert not os.path.isabs(sample_report["config"])
    for role in ("mask", "proliferative", "vessels"):
        assert not os.path.isabs(sample_report["inputs"][role]), role
    assert sample_report["how_to_reproduce"].endswith(sample_report["config"])


# ---------------------------------------------------------------------------
# Transport options and the proxy paths
# ---------------------------------------------------------------------------


def test_heterogeneous_transport_branch_runs_and_is_labelled(tmp_path: Path):
    """`diffusion_from_vessels` must reach the real D(x) field, not be ignored.

    Run short: the heterogeneous operator costs roughly 19x per day at this
    grid, which is exactly why the committed sample ships the homogeneous
    default.
    """
    config = {
        "subject": "sub-01",
        "synthetic": True,
        "inputs": {
            "mask": str(SAMPLE_DIR / "sub-01" / "sub-01_mask.nii.gz"),
            "vessels": str(SAMPLE_DIR / "sub-01" / "perf" / "sub-01_vessels.nii.gz"),
        },
        "trial": {
            "days": 15,
            "seeds": [1],
            "arms": ["mtd"],
            "diffusion_from_vessels": True,
        },
    }
    path = tmp_path / "twin.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    report = run_config(str(path))
    assert "heterogeneous" in report["trial"]["transport"]
    assert len(report["arms"]["mtd"]["runs"]) == 1

    from danon.clinical_twin import build_stream3_diffusion_field

    page = from_nifti_mask(config["inputs"]["mask"], vessels_path=config["inputs"]["vessels"])
    pattern = np.asarray(
        build_stream3_diffusion_field(page.geometry.vessels, voxel_mm=page.voxel_mm).d_engine,
        dtype=float,
    )
    assert pattern.max() / pattern.mean() > 1.0, "the field carries no spatial contrast"


def test_mask_only_config_uses_the_labelled_proxies(tmp_path: Path):
    """Omitting the maps is allowed and must be reported as a proxy, not hidden."""
    config = {
        "subject": "sub-proxy",
        "inputs": {"mask": str(SAMPLE_DIR / "sub-01" / "sub-01_mask.nii.gz")},
        "trial": {"days": 5, "seeds": [2], "arms": ["mtd"]},
    }
    path = tmp_path / "twin.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    report = run_config(str(path))
    assert report["inputs"]["proliferative_source"] == "distance-proxy"
    assert report["inputs"]["vasculature_source"] == "null-uniform"
    assert report["synthetic"] is False  # the config did not claim the phantom


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, config: dict) -> str:
    path = tmp_path / "twin.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return str(path)


def _valid(tmp_path: Path) -> dict:
    return {
        "subject": "sub-x",
        "inputs": {"mask": str(SAMPLE_DIR / "sub-01" / "sub-01_mask.nii.gz")},
        "trial": {"days": 5, "seeds": [1], "arms": ["mtd"]},
    }


def test_empty_seed_list_is_refused_as_no_measurement(tmp_path: Path):
    config = _valid(tmp_path)
    config["trial"]["seeds"] = []
    with pytest.raises(ValueError, match="empty cohort is not a null result"):
        load_config(_write(tmp_path, config))


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: c["trial"].update(arms=["mtd", "magic"]), "is not one of"),
        (lambda c: c["trial"].update(arms=[]), "nothing would be run"),
        (lambda c: c["trial"].update(days=0), "positive integer"),
        (lambda c: c["trial"].update(days=True), "positive integer"),
        (lambda c: c["trial"].update(seeds=[1.5]), "must be integers"),
        (lambda c: c["trial"].update(initial_resistant_fraction=1.5), "must be in"),
        (lambda c: c["inputs"].update(mask=None), "'inputs.mask' is required"),
        (lambda c: c["inputs"].update(mask="nope.nii.gz"), "which is not there"),
        (
            lambda c: c["trial"].update(diffusion_from_vessels=True),
            "names no map to build D\\(x\\) from",
        ),
    ],
)
def test_config_refusals_name_the_field(tmp_path: Path, mutate, match):
    config = _valid(tmp_path)
    mutate(config)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_config(_write(tmp_path, config))


def test_malformed_config_is_refused_with_the_path(tmp_path: Path):
    path = tmp_path / "bad.json"
    path.write_text("{ not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_config(str(path))
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a JSON object"):
        load_config(str(path))
    with pytest.raises(ValueError, match="'trial' must be an object"):
        load_config(_write(tmp_path, {"inputs": {"mask": "anything.nii.gz"}}))
    # Shape is reported before the filesystem, so a malformed config does not
    # come back as "file not found" for whichever volume it named first.
    with pytest.raises(ValueError, match="'trial' must be an object"):
        load_config(
            _write(
                tmp_path,
                {"inputs": {"mask": str(SAMPLE_DIR / "sub-01" / "sub-01_mask.nii.gz")}},
            )
        )
