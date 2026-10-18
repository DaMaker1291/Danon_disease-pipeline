"""Tissue portability: generic solver, glioma-only data, mocked figures refused.

The pitch this suite answers claims the core is tumor-agnostic and therefore
generalizes to breast, lung and pancreatic cancer. Half of that is a real,
measurable property of the code (the swap surface, the phantom run); the other
half is a prediction with no data behind it. These tests keep the two halves
apart: the portability run must reproduce, stay finite and bounded, the
inventory must still find zero non-glioma assets, and every mocked figure the
pitch offered must still be carried inside an explicit refusal -- a standard
that quietly started predicting other cancers is the failure this repository
exists to prevent.
"""

import json
import os

import pytest

from danon.cure import EvidenceStatus
from danon.tumor_agnostic import (
    BOUND_RATIO,
    BRAIN_TISSUE_CLASSES,
    DAYS_LONG,
    DAYS_SHORT,
    PHANTOM_SIZE,
    PORTABILITY_DISCLAIMER,
    PORTABILITY_RECORD,
    generic_solid_mass,
    portability_provenance,
    portability_refusals,
    portability_run,
    swap_surface,
    swap_surface_fields,
    tissue_inventory,
    what_portability_cannot_show,
)

needs_record = pytest.mark.skipif(
    not os.path.exists(PORTABILITY_RECORD),
    reason="tissue-portability record not built",
)


@pytest.fixture(scope="module")
def geom():
    return generic_solid_mass()


# ---------------------------------------------------------------------------
# The phantom and the swap surface
# ---------------------------------------------------------------------------


def test_phantom_satisfies_the_tumorgeometry_contract(geom):
    mask = geom.tumor_mask
    assert mask.ndim == 3 and mask.shape == (PHANTOM_SIZE,) * 3
    assert mask.sum() > 0
    assert geom.proliferative.shape == mask.shape
    assert geom.vessels.shape == mask.shape
    assert geom.vessels.max() > 0.0
    # uniform proliferation: no necrotic/hypoxic core by construction
    assert float(geom.proliferative[mask].min()) == 1.0
    assert float(geom.proliferative[~mask].max()) == 0.0


def test_phantom_is_not_the_gbm_phantom(geom):
    from danon.evosteer import make_gbm_phantom

    gbm = make_gbm_phantom()
    assert gbm.tumor_mask.shape == geom.tumor_mask.shape
    assert not (gbm.tumor_mask == geom.tumor_mask).all()
    assert not (gbm.proliferative == geom.proliferative).all()


def test_swap_surface_is_five_dataclasses_with_stable_fields():
    surface = swap_surface()
    assert set(surface) == {
        "TumorGeometry",
        "CloneParams",
        "DrugParams",
        "ImmuneParams",
        "DiffusionField",
    }
    assert swap_surface_fields() == sum(len(v) for v in surface.values())
    assert "tumor_mask" in surface["TumorGeometry"]
    assert "kill_max_S" in surface["DrugParams"]
    assert "competition" in surface["CloneParams"]


# ---------------------------------------------------------------------------
# The portability run
# ---------------------------------------------------------------------------


def test_run_is_deterministic_finite_and_bounded():
    a = portability_run(DAYS_SHORT)
    b = portability_run(DAYS_SHORT)
    assert (a.final_burden, a.pfs_day, a.progressed) == (
        b.final_burden,
        b.pfs_day,
        b.progressed,
    )
    assert a.finite and a.bounded
    assert 0.0 < a.capacity_ratio <= BOUND_RATIO


def test_long_run_stays_below_the_structural_bound():
    run = portability_run(DAYS_LONG)
    assert run.finite and run.bounded
    # the model's own growth equilibrium is 1.4 x K x V (asymmetric room
    # terms); noise and in-transit mass may lift the max a little above it,
    # but the enforced bound is BOUND_RATIO and the excess is recorded
    assert run.capacity_ratio <= BOUND_RATIO
    assert run.pfs_day is not None  # this phantom progresses within 365 d


# ---------------------------------------------------------------------------
# The glioma-only inventory
# ---------------------------------------------------------------------------


def test_inventory_is_brain_only_and_the_data_scan_is_empty():
    inv = tissue_inventory()
    assert inv["tissue_d_keys"] == sorted(BRAIN_TISSUE_CLASSES)
    assert inv["tissue_d_brain_only"] is True
    for group, hits in inv["data_organ_hits"].items():
        assert hits == [], (group, hits)
    assert inv["calibration_markers"]["clone_params_calibrated"] is True
    assert inv["calibration_markers"]["immune_params_assumed"] is True
    assert inv["calibration_markers"]["tissue_d_assumed"] is True


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


def test_refusals_carry_every_mocked_figure():
    text = json.dumps(portability_refusals())
    for must in (
        "Mammary / Fatty Parenchyma",
        "Porous Pulmonary Alveolar",
        "Dense Fibrotic Stromal",
        "90% of the codebase",
        "generalizes to predict spatial recurrence",
        "invisible residual cells after surgery",
        "Every cancer type benefits",
    ):
        assert must in text, must


def test_provenance_refuses_every_cross_cancer_claim():
    prov = portability_provenance()
    for key in (
        "predicts_breast_lung_pancreas_recurrence",
        "non_brain_mesh_exists_here",
        "non_brain_drug_parameter_set_exists",
        "postoperative_residual_input_available_here",
    ):
        assert prov[key] is EvidenceStatus.PROVEN_FAILED, key


def test_cannot_show_names_the_phantom_limit():
    notes = what_portability_cannot_show()
    assert notes and any("not lung" in n for n in notes)
    assert any("denominator" in n for n in notes)


# ---------------------------------------------------------------------------
# The record and the claim check
# ---------------------------------------------------------------------------


@needs_record
def test_record_carries_the_disclaimer_and_the_measured_numbers():
    with open(PORTABILITY_RECORD) as fh:
        record = json.load(fh)
    assert record["disclaimer"] == PORTABILITY_DISCLAIMER
    assert record["swap_surface_fields"] == swap_surface_fields()
    ratios = {r["days"]: r["capacity_ratio"] for r in record["runs"]}
    assert set(ratios) == {DAYS_SHORT, DAYS_LONG}
    assert all(r["finite"] and r["bounded"] for r in record["runs"])
    assert record["inventory"]["data_organ_hits"] == {
        "breast": [],
        "lung": [],
        "pancreas": [],
        "non_gbm_drugs": [],
    }
    assert len(record["refusals"]) == 7


@needs_record
def test_claim_check_passes():
    from danon.claims import (
        check_solver_portability_is_measured_and_generalization_is_refused,
    )

    result = check_solver_portability_is_measured_and_generalization_is_refused()
    assert result.supported, result.detail
