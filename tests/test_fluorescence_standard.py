"""The semi-synthetic 5-ALA standard: identity exact, contrast measured, stream refused.

The standard plants fields, so most of these tests are about the parts that must
NOT be synthetic: the intake identity (a real invariant of the mapping), the
engine's acceptance of a supplied seed, the found inertness of the fluorophore
pattern on this shell-like case, and the requirement that stream 1 stays REFUSED
on its live count of zero cached captures. A standard that activated the stream
would be the failure this repository exists to prevent.
"""

import json
import os

import numpy as np
import pytest

from danon.cure import EvidenceStatus
from danon.fluorescence_standard import (
    DAYS_LONG,
    DAYS_SHORT,
    DECAY_SWEEP_VOXELS,
    MASK_PATH,
    PLANT_MODES,
    STANDARD_RECORD,
    SYNTHETIC_DISCLAIMER,
    describe_field,
    fluorescence_standard_provenance,
    plant_field,
    run_layout,
    standard_geometry,
    standard_refusals,
    stream1_still_refused,
    what_a_synthetic_input_cannot_show,
)

needs_record = pytest.mark.skipif(
    not os.path.exists(STANDARD_RECORD), reason="fluorescence standard record not built"
)


@pytest.fixture(scope="module")
def geom():
    if not os.path.exists(MASK_PATH):
        pytest.skip("standard case not cached")
    return standard_geometry()


# ---------------------------------------------------------------------------
# Planting
# ---------------------------------------------------------------------------


def test_plant_rejects_unknown_modes_and_bad_decay(geom):
    with pytest.raises(ValueError, match="unknown plant mode"):
        plant_field(geom.tumor_mask, "cavity")
    with pytest.raises(ValueError, match="decay_voxels must be positive"):
        plant_field(geom.tumor_mask, "rim", decay_voxels=0.0)


def test_planted_fields_live_inside_the_mask(geom):
    mask = geom.tumor_mask
    for mode in PLANT_MODES:
        field = plant_field(mask, mode, decay_voxels=2.0)
        assert field.shape == mask.shape
        assert (field[~mask] == 0.0).all()
        assert (field[mask] > 0.0).all()


def test_uniform_is_exactly_one_inside(geom):
    field = plant_field(geom.tumor_mask, "uniform")
    assert float(field[geom.tumor_mask].min()) == 1.0
    assert float(field[geom.tumor_mask].max()) == 1.0


@pytest.mark.parametrize("lam", DECAY_SWEEP_VOXELS)
def test_rim_is_boundary_weighted_and_core_is_not(geom, lam):
    mask = geom.tumor_mask
    rim = plant_field(mask, "rim", decay_voxels=lam)
    core = plant_field(mask, "core", decay_voxels=lam)
    d_rim = describe_field(rim, mask, mode="rim", decay_voxels=lam)
    d_core = describe_field(core, mask, mode="core", decay_voxels=lam)
    assert d_rim.rim_share > d_core.rim_share
    assert d_rim.mean_distance <= d_core.mean_distance


def test_the_case_is_a_shell_so_the_planted_layouts_are_nearly_the_same(geom):
    """Pins the measured reason the layout contrast is small on THIS case."""
    mask = geom.tumor_mask
    for mode in PLANT_MODES:
        d = describe_field(
            plant_field(mask, mode), mask, mode=mode, decay_voxels=2.0
        )
        assert d.mean_distance < 2.0  # measured ~1.06 voxels


# ---------------------------------------------------------------------------
# The intake identity and the engine
# ---------------------------------------------------------------------------


def test_intake_identity_fills_to_the_engine_total_exactly(geom):
    from danon.clinical_twin import assemble_digital_twin_state
    from danon.evosteer import DEFAULT_CLONES

    mask = geom.tumor_mask
    state = assemble_digital_twin_state(
        fluorescence_5ala_mesh=plant_field(mask, "uniform"), mask=mask
    )
    N_S0, N_R0 = state.initial_state
    total = float(N_S0.sum() + N_R0.sum())
    target = 0.65 * float(DEFAULT_CLONES.carrying_capacity) * int(mask.sum())
    assert total == pytest.approx(target, abs=1e-9)


def test_every_planted_layout_carries_the_same_total_burden(geom):
    from danon.clinical_twin import assemble_digital_twin_state

    mask = geom.tumor_mask
    totals = []
    for mode in PLANT_MODES:
        state = assemble_digital_twin_state(
            fluorescence_5ala_mesh=plant_field(mask, mode), mask=mask
        )
        N_S0, N_R0 = state.initial_state
        totals.append(float(N_S0.sum() + N_R0.sum()))
    assert max(totals) - min(totals) < 1e-9


def test_run_is_reproducible(geom):
    a = run_layout(geom, "uniform", days=30, seed=17)
    b = run_layout(geom, "uniform", days=30, seed=17)
    assert (a.final_burden, a.pfs_day, a.progressed) == (b.final_burden, b.pfs_day, b.progressed)


def test_a_supplied_seed_changes_the_trajectory_and_the_clonal_layout_dominates(geom):
    """The measured finding: uniform R placement vs the clustered default."""
    default = run_layout(geom, "default", days=DAYS_LONG, seed=17)
    uniform = run_layout(geom, "uniform", days=DAYS_LONG, seed=17)
    assert uniform.final_burden > default.final_burden + 500.0
    assert uniform.pfs_day is not None and default.pfs_day is not None
    assert uniform.pfs_day < default.pfs_day


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_stream1_is_still_refused_on_its_live_count():
    refused, detail = stream1_still_refused()
    assert refused
    assert "0 cached capture" in detail


def test_provenance_refuses_every_synthetic_claim():
    prov = fluorescence_standard_provenance()
    for key in (
        "planted_field_is_a_5ala_acquisition",
        "synthetic_input_is_patient_evidence",
        "stream1_activates_on_synthetic_input",
        "resection_cavity_available_here",
    ):
        assert prov[key] is EvidenceStatus.PROVEN_FAILED, key


def test_refusals_name_the_pitch_steps_that_cannot_run_here():
    text = json.dumps(standard_refusals())
    for must in ("post-operative", "0.42", "Ivy GAP", "MU-Glioma-Post", "GLASS", "haematoxylin"):
        assert must in text, must


def test_cannot_show_states_the_biological_limit():
    notes = what_a_synthetic_input_cannot_show()
    assert notes and any("marks tumour" in n for n in notes)


@needs_record
def test_record_carries_the_disclaimer_and_the_inertness_finding():
    with open(STANDARD_RECORD) as fh:
        record = json.load(fh)
    assert record["synthetic_disclaimer"] == SYNTHETIC_DISCLAIMER
    assert record["stream1_census"]["still_refused"] is True
    contrast = record["layout_contrast"]
    assert contrast[str(DAYS_LONG)]["relative_spread"] < 0.01
    # internal consistency: every horizon carries all four layouts
    rows = record["runs"]
    for days in (DAYS_SHORT, DAYS_LONG):
        modes = {r["mode"] for r in rows if r["days"] == days}
        assert modes == {"default", "uniform", "rim", "core"}


@needs_record
def test_record_matches_a_live_partial_recompute(geom):
    """The recorded engine runs must reproduce: a sample is re-run live."""
    with open(STANDARD_RECORD) as fh:
        record = json.load(fh)
    live = {
        (r.mode, r.seed, r.days): r
        for r in (
            run_layout(geom, mode, days=days, seed=17, keep_fields=(days == DAYS_SHORT))
            for mode, days in (("uniform", DAYS_SHORT), ("rim", DAYS_LONG), ("default", DAYS_LONG))
        )
    }
    for run in record["runs"]:
        key = (run["mode"], run["seed"], run["days"])
        if key not in live:
            continue
        assert run["final_burden"] == pytest.approx(live[key].final_burden, abs=1e-9)
        assert run["initial_burden"] == pytest.approx(live[key].initial_burden, abs=1e-9)
        assert run["pfs_day"] == live[key].pfs_day
        if run["outside_share"] is not None:
            assert run["outside_share"] == pytest.approx(live[key].outside_share, abs=1e-12)


@needs_record
def test_claim_check_passes():
    from danon.claims import check_synthetic_5ala_standard_is_labelled_and_leaves_stream1_refused

    result = check_synthetic_5ala_standard_is_labelled_and_leaves_stream1_refused()
    assert result.supported, result.detail
