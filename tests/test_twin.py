"""Tests for the organ-level cardiovascular digital twin.

The theme of this module is refusal: the twin must return NOT_MODELLED
rather than invent numbers for treatments with no validated human effect
size. Several tests attack exactly that property.
"""

from __future__ import annotations

import pytest

from danon.twin import (
    POST_STENT_RESIDUAL_STENOSIS_PCT,
    ScanInputs,
    TwinPatient,
    TwinTreatment,
    TwinVerdict,
    CardiovascularTwin,
    risk_category_of,
    simulate_treatment,
)
from danon.prevention import Sex, risk_category


def _demo_patient(**overrides) -> TwinPatient:
    patient = TwinPatient(
        patient_id="twin-test",
        age_years=60,
        sex=Sex.MALE,
        scan=ScanInputs(
            lvef_pct=48,
            scar_fraction=0.08,
            ischaemic_fraction=0.12,
            worst_stenosis_pct=70,
            cac_score=180,
            coronary_diameter_mm=3.4,
            systolic_bp_mmhg=128,
        ),
        chol_total_mgdl=232,
        chol_hdl_mgdl=44,
        ldl_mg_dl=158,
        bp_sys_mmhg=134,
        on_bp_medication=True,
        established_ascvd=True,
    )
    for key, value in overrides.items():
        setattr(patient, key, value)
    return patient


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_initialization_counts_usable_and_missing_scan_fields():
    twin = CardiovascularTwin(_demo_patient())
    ini = twin.initialization
    assert len(ini.usable_scan_fields) == 7
    assert "lv_mass_g" in ini.missing_scan_fields
    assert "lv_edv_ml" in ini.missing_scan_fields


def test_initialization_flags_cross_modality_conflicts():
    """A twin that silently accepts contradictory inputs is worse than no
    twin: the imaging BP and the chart BP disagree by >=15 mmHg here."""
    patient = _demo_patient()
    patient.scan.systolic_bp_mmhg = 155
    twin = CardiovascularTwin(patient)
    assert any("mmHg" in c for c in twin.initialization.conflicts)


def test_initialization_flags_cac_zero_with_significant_stenosis():
    patient = _demo_patient()
    patient.scan.cac_score = 0
    patient.scan.worst_stenosis_pct = 70
    twin = CardiovascularTwin(patient)
    assert any("non-calcified" in c for c in twin.initialization.conflicts)


def test_initialization_flags_ascvd_history_without_scar():
    patient = _demo_patient()
    patient.scan.scar_fraction = 0.0
    twin = CardiovascularTwin(patient)
    assert any("secondary-prevention" in c for c in twin.initialization.conflicts)


def test_clean_patient_has_no_conflicts():
    twin = CardiovascularTwin(_demo_patient())
    assert twin.initialization.conflicts == []


def test_baseline_risk_matches_prevention_engine():
    """The twin must not have its own private risk math: its baseline equals
    what triage() computes for the same inputs."""
    patient = _demo_patient()
    twin = CardiovascularTwin(patient)
    from danon.prevention import triage as prevention_triage
    direct = prevention_triage(
        patient.to_risk_profile(), patient.to_enhancers(),
        established_ascvd=patient.established_ascvd,
    )
    assert twin.initialization.baseline_risk_10y == pytest.approx(direct.risk_10y)


# ---------------------------------------------------------------------------
# LDL-anchored counterfactuals
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_results():
    twin = CardiovascularTwin(_demo_patient())
    return {r.treatment: r for r in twin.run_all_treatments(10)}


def test_statin_counterfactuals_are_modelled_with_finite_nnt(all_results):
    for t in (
        TwinTreatment.MODERATE_STATIN,
        TwinTreatment.HIGH_INTENSITY_STATIN,
        TwinTreatment.STATIN_EZETIMIBE,
        TwinTreatment.PCSK9_INHIBITOR,
    ):
        r = all_results[t]
        assert r.verdict is TwinVerdict.MODELLED
        assert r.absolute_risk_reduction > 0
        assert 1.0 < r.number_needed_to_treat < 200


def test_benefit_is_monotone_in_treatment_intensity(all_results):
    """Stronger LDL lowering must give larger absolute benefit and deeper
    plaque regression. A twin that ranks a moderate statin above a PCSK9
    inhibitor on benefit is broken."""
    arrs = [
        all_results[TwinTreatment.MODERATE_STATIN].absolute_risk_reduction,
        all_results[TwinTreatment.HIGH_INTENSITY_STATIN].absolute_risk_reduction,
        all_results[TwinTreatment.STATIN_EZETIMIBE].absolute_risk_reduction,
        all_results[TwinTreatment.PCSK9_INHIBITOR].absolute_risk_reduction,
    ]
    assert arrs == sorted(arrs)

    pavs = [
        all_results[TwinTreatment.MODERATE_STATIN].pav_change_pct,
        all_results[TwinTreatment.HIGH_INTENSITY_STATIN].pav_change_pct,
        all_results[TwinTreatment.STATIN_EZETIMIBE].pav_change_pct,
        all_results[TwinTreatment.PCSK9_INHIBITOR].pav_change_pct,
    ]
    # More regression = more negative PAV change, so the sequence descends.
    assert pavs == sorted(pavs, reverse=True)
    assert all_results[TwinTreatment.PCSK9_INHIBITOR].pav_change_pct < 0
    assert all_results[TwinTreatment.MODERATE_STATIN].pav_change_pct < 0


def test_treated_risk_is_lower_than_baseline(all_results):
    for t, r in all_results.items():
        if r.verdict in (TwinVerdict.MODELLED, TwinVerdict.MODELLED_APPROXIMATE) \
                and r.absolute_risk_reduction > 0:
            assert r.treated_event_risk < r.baseline_event_risk_10y


def test_base_editing_is_flagged_approximate_not_established(all_results):
    r = all_results[TwinTreatment.ONE_SHOT_BASE_EDITING]
    assert r.verdict is TwinVerdict.MODELLED_APPROXIMATE
    assert any("VERVE" in a or "not an approved therapy" in a for a in r.approximations)


def test_lifestyle_is_flagged_as_upper_bound(all_results):
    r = all_results[TwinTreatment.LIFESTYLE]
    assert r.verdict is TwinVerdict.MODELLED_APPROXIMATE
    assert any("upper bound" in a for a in r.approximations)


# ---------------------------------------------------------------------------
# The refusal layer — the most important behaviour in the module
# ---------------------------------------------------------------------------

def test_unknown_treatment_returns_not_modelled_without_crashing():
    """A treatment outside the catalogue must produce a clean refusal, not an
    AttributeError on .value and not a fabricated number."""
    r = simulate_treatment(_demo_patient(), "CAR-T_like_scar_regeneration")
    assert r.verdict is TwinVerdict.NOT_MODELLED
    assert r.not_modelled_reason is not None
    assert "no validated human effect-size model" in r.not_modelled_reason
    assert r.absolute_risk_reduction == 0.0
    assert r.number_needed_to_treat == float("inf")


def test_stent_without_stenosis_measurement_is_refused():
    """The twin cannot model a structural intervention on a lumen it has not
    seen; refusing beats imputing a stenosis."""
    patient = _demo_patient()
    patient.scan.worst_stenosis_pct = None
    r = simulate_treatment(patient, TwinTreatment.STENT)
    assert r.verdict is TwinVerdict.NOT_MODELLED
    assert "stenosis" in r.not_modelled_reason


def test_stent_reduces_pressure_drop_without_claiming_event_benefit():
    """Stenting changes the haemodynamics; it does NOT change the event-risk
    math, because ISCHEMIA showed no mortality advantage over optimal medical
    therapy in stable disease. The result must reflect both facts."""
    patient = _demo_patient()
    r = simulate_treatment(patient, TwinTreatment.STENT)

    assert r.verdict is TwinVerdict.MODELLED_APPROXIMATE
    assert r.pressure_drop_change_pct is not None
    assert r.pressure_drop_change_pct < -50.0     # 70% stenosis -> big drop
    assert r.absolute_risk_reduction == 0.0       # no fabricated event benefit
    assert any("ISCHEMIA" in n for n in r.evidence_notes)


def test_regenerative_therapies_are_absent_from_the_catalogue():
    """cure.py documents why these have no validated human effect size; the
    catalogue must not quietly include them."""
    names = {t.value for t in TwinTreatment}
    for banned in ("scar_conversion", "anti_fibrotic", "angiogenesis", "mirna_regeneration"):
        assert banned not in names


# ---------------------------------------------------------------------------
# Horizon handling and observability
# ---------------------------------------------------------------------------

def test_horizon_is_clamped_to_a_sane_range():
    short = simulate_treatment(_demo_patient(), TwinTreatment.HIGH_INTENSITY_STATIN, 0)
    long = simulate_treatment(_demo_patient(), TwinTreatment.HIGH_INTENSITY_STATIN, 100)
    assert short.horizon_years == 1
    assert long.horizon_years == 30


def test_longer_horizon_gives_larger_absolute_benefit():
    five = simulate_treatment(_demo_patient(), TwinTreatment.HIGH_INTENSITY_STATIN, 5)
    ten = simulate_treatment(_demo_patient(), TwinTreatment.HIGH_INTENSITY_STATIN, 10)
    assert ten.absolute_risk_reduction > five.absolute_risk_reduction
    # Plaque trajectory is computed over the requested horizon in months.
    assert ten.pav_change_pct < five.pav_change_pct


def test_what_the_twin_cannot_know_states_observability_limits():
    twin = CardiovascularTwin(_demo_patient())
    limits = twin.what_the_twin_cannot_know()
    assert any("millimetres" in s for s in limits)          # imaging resolution
    assert any("variant" in s for s in limits)              # mutation map
    assert any("arrhythmia" in s for s in limits)           # sub-organ processes
    # And the patient-specific gaps are named.
    assert any("lv_mass_g" in s for s in limits)


def test_risk_category_of_matches_published_thresholds():
    assert risk_category_of(0.02) == risk_category(0.02)
    assert risk_category_of(0.06) == "borderline"
    assert risk_category_of(0.30) == "high"
