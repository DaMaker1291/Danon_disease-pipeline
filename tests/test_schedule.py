"""Tests for the schedule compiler module.

Pins: provenance-required intake (refusals), event extraction from dose
logs (synthetic trajectory + real runs), paired per-patient metrics, the
observability limits in the report, and well-formed .vti field export.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from danon.cure import EvidenceStatus
from danon.evosteer import (
    AdaptiveDosePolicy,
    CollateralSwitchPolicy,
    MTDPolicy,
    SimulationTrajectory,
    make_gbm_phantom,
)
from danon.schedule import (
    PatientIntake,
    compile_dosing_schedule,
    export_fields_vti,
    extract_schedule_events,
    format_schedule_text,
    schedule_provenance,
    what_this_model_cannot_know,
)

MTD_NAME = MTDPolicy().describe()
SWITCH_NAME = CollateralSwitchPolicy().describe()


@pytest.fixture(scope="module")
def gbm():
    return make_gbm_phantom(size=16, seed=3)


# ---------------------------------------------------------------------------
# Intake refusals
# ---------------------------------------------------------------------------


def test_intake_requires_provenance():
    PatientIntake(r0=0.01, r0_source="scRNA-seq clone fractions, batch 2024-06")
    with pytest.raises(ValueError):
        PatientIntake(r0=0.0, r0_source="x")
    with pytest.raises(ValueError):
        PatientIntake(r0=1.0, r0_source="x")
    with pytest.raises(ValueError):
        PatientIntake(r0=0.01, r0_source="   ")
    with pytest.raises(ValueError):
        PatientIntake(r0=0.01, r0_source="x", geometry_note="")


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------


def _synthetic_traj(dose_A, dose_B, burden_S, burden_R):
    n = len(dose_A)
    return SimulationTrajectory(
        policy="synthetic",
        days=n,
        burden_S=np.asarray(burden_S, dtype=float),
        burden_R=np.asarray(burden_R, dtype=float),
        dose_A=np.asarray(dose_A, dtype=float),
        dose_B=np.asarray(dose_B, dtype=float),
        pfs_day=None,
        progressed=False,
        initial_resistant_fraction=0.01,
        toxicity=float(sum(dose_A)),
        final_resistant_fraction=0.0,
        resistant_dominance_day=None,
    )


def test_extract_events_labels_and_fractions():
    traj = _synthetic_traj(
        dose_A=[1, 1, 0, 0, 1],
        dose_B=[0, 0, 1, 1, 0],
        burden_S=[90, 80, 70, 70, 60],
        burden_R=[10, 20, 30, 30, 40],
    )
    events = extract_schedule_events(traj)
    assert [(e.day, e.action) for e in events] == [
        (2, "switch A->B"),
        (4, "switch B->A"),
    ]
    assert abs(events[0].resistant_fraction - 0.30) < 1e-12
    assert abs(events[1].resistant_fraction - 0.40) < 1e-12


def test_extract_events_withdraw_and_dose_on():
    traj = _synthetic_traj(
        dose_A=[1, 0, 0, 1],
        dose_B=[0, 0, 0, 0],
        burden_S=[90, 80, 80, 85],
        burden_R=[10, 10, 10, 10],
    )
    events = extract_schedule_events(traj)
    assert [(e.day, e.action) for e in events] == [
        (1, "withdraw"),
        (3, "dose on"),
    ]


def test_extract_events_mtd_has_none(gbm):
    from danon.evosteer import TumorEvolutionSimulator

    sim = TumorEvolutionSimulator(gbm)
    traj = sim.run(MTDPolicy(), days=60, initial_resistant_fraction=0.01, seed=1)
    assert extract_schedule_events(traj) == []


# ---------------------------------------------------------------------------
# Compiled schedule (paired metrics)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def compiled(gbm):
    return compile_dosing_schedule(
        gbm,
        PatientIntake(r0=0.01, r0_source="phantom default seed"),
        policy=CollateralSwitchPolicy(),
        days=250,
        n_replicates=3,
        seed0=7,
    )


def test_compiled_schedule_pairs_and_metrics(compiled):
    rep = compiled
    assert rep.policy == SWITCH_NAME
    # The switch policy actually switched: events were extracted, not asserted.
    assert rep.events, "switch policy should produce regime-change events"
    for ev in rep.events:
        assert 0.0 <= ev.resistant_fraction <= 1.0
    # Bookkeeping adds up.
    assert (
        rep.dose_A_days + rep.dose_B_days + rep.off_days == rep.horizon_days
    )
    # Paired metrics exist and are sane.
    assert rep.n_replicates == 3
    assert rep.policy_median_pfs > 0
    assert rep.paired_mtd_median_pfs > 0
    assert rep.toxicity_reduction_median is not None
    assert rep.toxicity_reduction_median > 0.0  # pulsed dosing exposes less drug
    # The observability limit is stated in the notes, every time.
    joined = " ".join(rep.notes)
    assert "NOT clinically observable" in joined
    assert "phantom" in joined


def test_compiled_schedule_threshold_band(gbm):
    rep = compile_dosing_schedule(
        gbm,
        PatientIntake(r0=0.01, r0_source="phantom default seed"),
        policy=CollateralSwitchPolicy(theta_on=0.15),
        days=200,
        n_replicates=2,
        seed0=7,
        sweep_thetas=(0.10, 0.30),
    )
    assert rep.threshold_band is not None
    lo, hi = rep.threshold_band
    assert lo <= hi and lo in (0.10, 0.30)
    assert any("over-precision" in n for n in rep.notes)


def test_formatted_report_contains_measured_values(compiled):
    text = format_schedule_text(compiled)
    assert "NOT patient MRI" in text
    assert "Regime changes" in text
    assert "paired metrics" in text
    assert "Limits" in text
    # No fabricated precision: the abstract's +38%/60% must not appear
    # unless measured here.
    assert "38.4" not in text and "38%" not in text


# ---------------------------------------------------------------------------
# .vti export (Output A without VR claims)
# ---------------------------------------------------------------------------


def _probe_geometry(shape):
    """An asymmetric geometry so any axis-order confusion shows as a transpose."""
    from danon.evosteer import TumorGeometry

    mask = np.zeros(shape, dtype=bool)
    mask[1, 2, 3] = True
    return TumorGeometry(
        tumor_mask=mask,
        proliferative=np.where(mask, 0.9, 0.5).astype(float),
        vessels=np.where(mask, 1.0, 0.1).astype(float),
        voxel_mm=2.0,
    )


def _probe_trajectory(shape, mask):
    class _T:
        pass

    t = _T()
    t.final_conc_A = np.full(shape, 0.25)
    t.final_conc_B = np.full(shape, 0.5)
    t.final_N_S = np.where(mask, 100.0, 0.0)
    t.final_N_R = np.where(mask, 7.0, 0.0)
    return t


def test_vti_extent_matches_engine_axes_on_non_cubic_grid(tmp_path):
    """The extent must be (0 sx-1)(0 sy-1)(0 sz-1) for engine shape (sx, sy, sz).

    The previous writer destructured the shape reversed (nz, ny, nx), which
    silently transposed every non-cubic -- i.e. every real patient -- export
    while passing every cubic phantom test.
    """
    import base64

    shape = (5, 6, 7)
    geom = _probe_geometry(shape)
    t = _probe_trajectory(shape, geom.tumor_mask)
    out = tmp_path / "probe.vti"
    export_fields_vti(t, geom, str(out))
    root = ET.parse(str(out)).getroot()
    img = root.find("ImageData")
    assert img is not None
    assert img.attrib["WholeExtent"] == "0 4 0 5 0 6"
    assert img.attrib["Spacing"] == "2.0 2.0 2.0"

    # Independent F-order reshape recovers the array exactly: the on-disk
    # layout, not just the metadata, is in the declared axis order.
    piece = img.find("Piece")
    arrs = {da.attrib["Name"]: da.text for da in piece.findall("PointData/DataArray")}
    back = np.frombuffer(base64.b64decode(arrs["resistant"]), dtype="<f4").reshape(
        shape, order="F"
    )
    assert np.array_equal(back, np.where(geom.tumor_mask, 7.0, 0.0).astype("<f4").astype(float))


def test_vti_tissue_and_diffusion_maps_are_derived_from_the_geometry(tmp_path):
    """The derived maps must equal what anisotropy derives from the same geometry."""
    import base64

    from danon.anisotropy import DiffusionField, tissue_classes_from_geometry

    shape = (5, 6, 7)
    geom = _probe_geometry(shape)
    t = _probe_trajectory(shape, geom.tumor_mask)
    out = tmp_path / "probe.vti"
    export_fields_vti(t, geom, str(out))
    piece = ET.parse(str(out)).getroot().find("ImageData/Piece")
    arrs = {da.attrib["Name"]: da.text for da in piece.findall("PointData/DataArray")}

    def decoded(name):
        return np.frombuffer(base64.b64decode(arrs[name]), dtype="<f4").reshape(
            shape, order="F"
        )

    classes = tissue_classes_from_geometry(geom)
    assert np.array_equal(decoded("tissue_class"), classes.astype("<f4").astype(float))
    expected_d = DiffusionField.from_tissue_classes(
        classes, voxel_mm=float(geom.voxel_mm)
    ).d_mm2_per_day
    np.testing.assert_allclose(decoded("diffusion_mm2_per_day"), expected_d, rtol=1e-6)
    # A heterogeneous probe must actually carry heterogeneous D.
    assert decoded("diffusion_mm2_per_day").max() > decoded("diffusion_mm2_per_day").min()


def test_vti_include_tissue_false_keeps_the_original_four():
    shape = (5, 6, 7)
    geom = _probe_geometry(shape)
    t = _probe_trajectory(shape, geom.tumor_mask)
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "probe.vti")
        export_fields_vti(t, geom, out, include_tissue=False)
        piece = ET.parse(out).getroot().find("ImageData/Piece")
        names = [da.attrib["Name"] for da in piece.findall("PointData/DataArray")]
        assert names == ["drug_A", "drug_B", "sensitive", "resistant", "tumour_mask"]


def test_read_fields_vti_round_trips_export(tmp_path):
    """read_fields_vti recovers exactly what export_fields_vti wrote."""
    from danon.schedule import read_fields_vti

    shape = (5, 6, 7)
    geom = _probe_geometry(shape)
    t = _probe_trajectory(shape, geom.tumor_mask)
    out = tmp_path / "probe.vti"
    export_fields_vti(t, geom, str(out))

    vti = read_fields_vti(str(out))
    assert vti.dims == shape
    assert vti.spacing_mm == 2.0
    assert vti.n_points == 5 * 6 * 7
    assert {"drug_A", "drug_B", "sensitive", "resistant", "tissue_class"} <= set(vti.fields)
    np.testing.assert_array_equal(vti.fields["sensitive"], np.where(geom.tumor_mask, 100.0, 0.0))
    np.testing.assert_array_equal(vti.fields["resistant"], np.where(geom.tumor_mask, 7.0, 0.0))
    np.testing.assert_array_equal(vti.mask.astype(bool), geom.tumor_mask)


def test_read_fields_vti_rejects_malformed_files(tmp_path):
    """The reader raises on garbage instead of returning a plausible volume."""
    from danon.schedule import read_fields_vti

    bad = tmp_path / "bad.vti"
    bad.write_text("<VTKFile type='PolyData'/>")
    with pytest.raises(ValueError, match="not a VTK ImageData"):
        read_fields_vti(str(bad))

    empty = tmp_path / "empty.vti"
    empty.write_text("")
    with pytest.raises(Exception):
        read_fields_vti(str(empty))


def test_read_fields_vti_reads_committed_brats_artifact():
    """The committed BraTS artifact decodes with its documented properties."""
    import os

    from danon.schedule import read_fields_vti

    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "outputs_real", "brats_case01017_mtd_day120.vti",
    )
    if not os.path.exists(path):
        pytest.skip("committed BraTS VTI artifact not present")
    vti = read_fields_vti(path)
    assert vti.dims == (14, 24, 24)
    assert vti.spacing_mm == 1.0
    assert int(vti.mask.sum()) == 1774
    d = vti.fields["diffusion_mm2_per_day"]
    assert abs(d.max() / d.min() - 35.0) < 0.01  # 0.35 / 0.01 sanctuary contrast
    assert vti.fields["sensitive"].shape == vti.dims


# ---------------------------------------------------------------------------
# .vti export (Output A without VR claims)
# ---------------------------------------------------------------------------


def test_vti_export_wellformed(gbm, tmp_path):
    from danon.evosteer import TumorEvolutionSimulator

    sim = TumorEvolutionSimulator(gbm)
    traj = sim.run(
        CollateralSwitchPolicy(),
        days=80,
        initial_resistant_fraction=0.01,
        seed=3,
        keep_fields=True,
    )
    out = tmp_path / "fields.vti"
    export_fields_vti(traj, gbm, str(out))
    tree = ET.parse(str(out))
    root = tree.getroot()
    assert root.tag == "VTKFile" and root.attrib["type"] == "ImageData"
    img = root.find("ImageData")
    assert img is not None
    piece = img.find("Piece")
    assert piece is not None
    names = [
        da.attrib["Name"]
        for da in piece.findall("PointData/DataArray")
    ]
    assert names[:4] == ["drug_A", "drug_B", "sensitive", "resistant"]
    extent = piece.attrib["Extent"]
    assert extent == "0 15 0 15 0 15"
    # Non-empty base64 payload for the first array.
    first = piece.find("PointData/DataArray")
    assert first is not None and len(first.text or "") > 1000
    # The derived tissue/diffusion maps ride along by default, and the mask
    # is point data (one value per point) rather than malformed CellData.
    point_names = [da.attrib["Name"] for da in piece.findall("PointData/DataArray")]
    for derived in ("tissue_class", "diffusion_mm2_per_day", "proliferative", "vessels"):
        assert derived in point_names
    assert piece.find("CellData/DataArray") is None
    # The four simulation fields lead; the derived maps follow; the mask is last.
    assert point_names[:4] == ["drug_A", "drug_B", "sensitive", "resistant"]
    assert point_names[-1] == "tumour_mask"
    with pytest.raises(ValueError):
        export_fields_vti(
            sim.run(MTDPolicy(), days=20, initial_resistant_fraction=0.01, seed=1),
            gbm,
            str(tmp_path / "no_fields.vti"),
        )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_schedule_provenance_pins_observability():
    prov = schedule_provenance()
    assert prov["switch_rule_observability"] is EvidenceStatus.PROVEN_FAILED
    assert prov["schedule_day_precision"] is EvidenceStatus.PROVEN_FAILED
    assert prov["growth_kill_constants"] is EvidenceStatus.UNKNOWN


def test_schedule_limits_listed():
    limits = what_this_model_cannot_know()
    joined = " ".join(limits)
    assert "surrogate biomarker" in joined
    assert "daily resolution" in joined


# ---------------------------------------------------------------------------
# Poster summary: pitch claims vs measured artifact
# ---------------------------------------------------------------------------


def test_poster_summary_verbatim_claims_and_censoring(compiled):
    from danon.schedule import POSTER_PITCH_CLAIMS, poster_summary

    text = poster_summary(compiled)
    # The claim column is the abstract's wording, unsoftened.
    assert POSTER_PITCH_CLAIMS["pfs_extension"] in text
    assert POSTER_PITCH_CLAIMS["toxicity_reduction"] in text
    # The measured toxicity number is the report's, not a literal.
    assert compiled.toxicity_reduction_median is not None
    assert f"{compiled.toxicity_reduction_median:.1%}" in text
    # Censoring is computed from the report, and the fixture's policy arm
    # reaches the horizon -> the PFS figure must carry the lower-bound form.
    if compiled.pfs_extension_median is not None:
        if compiled.policy_median_pfs >= compiled.horizon_days:
            assert f">= {compiled.pfs_extension_median:+.1%}" in text
            assert "censored" in text
        else:
            assert "uncensored" in text
    # The column semantics are stated so the table cannot be misread.
    assert "verbatim claims" in text


def test_poster_summary_uncensored_branch(gbm):
    # MTD as the "policy" forces progression inside the horizon -> the
    # uncensored PFS branch and a non-censored rendering.
    rep = compile_dosing_schedule(
        gbm,
        PatientIntake(r0=0.01, r0_source="phantom default seed"),
        policy=MTDPolicy(),
        days=200,
        n_replicates=2,
        seed0=7,
    )
    from danon.schedule import poster_summary

    text = poster_summary(rep)
    if rep.pfs_extension_median is not None:
        if rep.policy_median_pfs < rep.horizon_days:
            assert f"{rep.pfs_extension_median:+.1%} (uncensored)" in text
        else:
            assert "censored" in text
    else:
        assert "not estimable" in text
