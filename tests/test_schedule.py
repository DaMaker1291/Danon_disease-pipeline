"""Tests for the schedule compiler module.

Pins: provenance-required intake (refusals), event extraction from dose
logs (synthetic trajectory + real runs), paired per-patient metrics, the
observability limits in the report, and well-formed .vti field export.
"""

from __future__ import annotations

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
    assert names == ["drug_A", "drug_B", "sensitive", "resistant"]
    extent = piece.attrib["Extent"]
    assert extent == "0 15 0 15 0 15"
    # Non-empty base64 payload for the first array.
    first = piece.find("PointData/DataArray")
    assert first is not None and len(first.text or "") > 1000
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
