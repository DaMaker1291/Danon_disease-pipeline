"""Tests for the cure-roadmap module.

The anchor tests reproduce the GLAGOV and ASTEROID IVUS trials from the
derived PAV/LDL model. The regression tests pin the two bugs found during
development: a clamp that collapsed every LDL to the same value, and a
proliferation-risk model that saturated at 1.0 for every delivery regime.
"""

from __future__ import annotations

import pytest

from danon.cure import (
    ASTEROID_LDL_ON_TREATMENT_MG_DL,
    ASTEROID_PAV_CHANGE_PCT,
    GLAGOV_LDL_ON_TREATMENT_MG_DL,
    GLAGOV_PAV_CHANGE_PCT,
    ANGIOGENESIS_OPTIMAL_LOG_DOSE,
    EvidenceStatus,
    assess_cure_roadmap,
    angiogenesis_response,
    cure_provenance,
    delivery_feasibility,
    plaque_regression,
    regeneration_regime_comparison,
    regeneration_safety,
    scar_remodeling,
)


# ---------------------------------------------------------------------------
# Stage 1: plaque regression — anchor trials
# ---------------------------------------------------------------------------

def test_pav_model_reproduces_glagov_anchor():
    """The derived PAV/LDL/time model must reproduce GLAGOV (-0.95% over
    76 weeks at LDL 36.6) to within measurement noise."""
    r = plaque_regression(
        on_treatment_ldl_mg_dl=GLAGOV_LDL_ON_TREATMENT_MG_DL, duration_months=17.5
    )
    assert r.pav_change_pct == pytest.approx(GLAGOV_PAV_CHANGE_PCT, abs=0.02)
    assert r.regression_likely
    assert not r.extrapolated


def test_pav_model_reproduces_asteroid_anchor():
    """ASTEROID: -0.98% over 24 months at LDL 60.8."""
    r = plaque_regression(
        on_treatment_ldl_mg_dl=ASTEROID_LDL_ON_TREATMENT_MG_DL, duration_months=24
    )
    assert r.pav_change_pct == pytest.approx(ASTEROID_PAV_CHANGE_PCT, abs=0.02)
    assert r.regression_likely


def test_lower_ldl_gives_more_regression():
    at_55 = plaque_regression(on_treatment_ldl_mg_dl=55, duration_months=24)
    at_90 = plaque_regression(on_treatment_ldl_mg_dl=90, duration_months=24)
    assert at_55.pav_change_pct < at_90.pav_change_pct < 0


def test_pav_domain_clamp_does_not_collapse_all_inputs():
    """Regression test.

    The original clamp was written as max(ldl, lo, hi) — a three-argument max
    that returns the domain ceiling for EVERY input, making an LDL of 36.6 and
    an LDL of 155 produce identical plaque change. A model where the input has
    no effect is not a model.
    """
    # Both values are INSIDE the 30-160 mg/dL domain and must give different
    # answers.
    low = plaque_regression(on_treatment_ldl_mg_dl=36.6, duration_months=24)
    high = plaque_regression(on_treatment_ldl_mg_dl=155.0, duration_months=24)
    assert low.pav_change_pct < high.pav_change_pct
    assert low.extrapolated is False
    assert high.extrapolated is False

    # A genuinely out-of-domain value is clamped to the domain edge and
    # flagged, and behaves identically to the edge value itself.
    below = plaque_regression(on_treatment_ldl_mg_dl=20.0, duration_months=24)
    edge = plaque_regression(on_treatment_ldl_mg_dl=30.0, duration_months=24)
    assert below.extrapolated is True
    assert below.pav_change_pct == pytest.approx(edge.pav_change_pct)
    assert any("clamped" in n for n in below.notes)


def test_pav_model_never_predicts_regression_above_the_ldl_reference():
    """Above the fitted reference LDL the model must flip to progression,
    which is what the IVUS literature shows for patients left at high LDL."""
    r = plaque_regression(on_treatment_ldl_mg_dl=160, duration_months=24)
    assert r.pav_change_pct > 0
    assert not r.regression_likely
    assert r.event_hr > 1.0  # worsening trajectory carries higher event risk


def test_longer_duration_gives_more_change_and_flags_extrapolation():
    """Within the anchor horizon the rate is linear (the two trials imply the
    same per-month constant to within 0.2%); past it the model must flag that
    the linear rate is unvalidated, because atheroma cannot fall forever."""
    six = plaque_regression(on_treatment_ldl_mg_dl=50, duration_months=6)
    twentyfour = plaque_regression(on_treatment_ldl_mg_dl=50, duration_months=24)
    ninety = plaque_regression(on_treatment_ldl_mg_dl=50, duration_months=90)

    assert twentyfour.pav_change_pct < six.pav_change_pct < 0
    assert ninety.pav_change_pct < twentyfour.pav_change_pct

    # Linear within the anchors: per-month rate is constant to within 5%.
    rate_6 = abs(six.pav_change_pct) / 6
    rate_24 = abs(twentyfour.pav_change_pct) / 24
    assert rate_6 == pytest.approx(rate_24, rel=0.05)

    # Beyond the anchor horizon the extrapolation must be flagged, not hidden.
    assert ninety.extrapolated
    assert any("36" in n or "saturate" in n for n in ninety.notes)


# ---------------------------------------------------------------------------
# Stage 2: scar remodelling
# ---------------------------------------------------------------------------

def test_scar_remodeling_is_flagged_animal_only():
    r = scar_remodeling(scar_fraction=0.20, hibernating_fraction=0.05)
    assert r.evidence_status is EvidenceStatus.ANIMAL_ONLY
    assert r.recoverable_fraction == pytest.approx(0.05)
    assert any("hibernating" in n.lower() for n in r.notes)


def test_no_therapy_recovers_transmural_scar():
    """The honest headline: established scar fraction contributes zero
    recoverable myocardium in this model."""
    r = scar_remodeling(scar_fraction=0.35, hibernating_fraction=0.0)
    assert r.recoverable_fraction == 0.0


# ---------------------------------------------------------------------------
# Stage 3: regeneration safety wall
# ---------------------------------------------------------------------------

def test_constitutive_expression_lands_in_danger_band():
    """The constitutive-AAV regime must land in the high-danger band that
    matches the miR-199a pig outcome (uncontrolled growth)."""
    s = regeneration_safety(fraction_cms_in_cycle=0.06, treatment_weeks=52.0)
    assert s["uncontrolled_proliferation_risk"] > 0.5
    assert s["expected_uncontrolled_foci"] > 1.0


def test_regimes_discriminate_rather_than_saturate():
    """Regression test.

    The original risk model clipped the exponent, so EVERY regime — including
    a 2-week transient course — saturated at risk 1.0. A safety model that
    returns 'certain disaster' for every option cannot rank options, which was
    its entire purpose. Transient courses must come out meaningfully safer.
    """
    regimes = regeneration_regime_comparison()
    by_name = {r["regime"]: r for r in regimes}

    constitutive = by_name["constitutive_AAV"]["uncontrolled_risk"]
    transient = by_name["transient_mRNA_repeat_dose"]["uncontrolled_risk"]
    short = by_name["transient_short_window"]["uncontrolled_risk"]

    assert constitutive > transient > short
    assert transient < 0.5
    assert short < 0.1
    # Risk must scale with exposure duration (roughly linearly at low hazard).
    assert constitutive / short > 10


def test_shorter_exposure_monotonically_reduces_risk():
    risks = [
        regeneration_safety(treatment_weeks=w)["uncontrolled_proliferation_risk"]
        for w in (1.0, 2.0, 4.0, 8.0, 52.0)
    ]
    assert risks == sorted(risks)


def test_risk_scales_with_fraction_in_cycle():
    low = regeneration_safety(fraction_cms_in_cycle=0.01, treatment_weeks=4.0)
    high = regeneration_safety(fraction_cms_in_cycle=0.10, treatment_weeks=4.0)
    assert high["uncontrolled_proliferation_risk"] > low["uncontrolled_proliferation_risk"]
    assert high["expected_new_contractile_mass_fraction"] > low["expected_new_contractile_mass_fraction"]


def test_no_input_produces_negative_risk_or_mass():
    for f, w in ((0.0, 0.0), (0.06, 0.0), (0.0, 52.0)):
        s = regeneration_safety(fraction_cms_in_cycle=f, treatment_weeks=w)
        assert s["uncontrolled_proliferation_risk"] >= 0.0
        assert s["expected_new_contractile_mass_fraction"] >= 0.0


# ---------------------------------------------------------------------------
# Stage 4: angiogenesis window
# ---------------------------------------------------------------------------

def test_angiogenesis_window_is_at_the_modelled_optimum():
    in_window = angiogenesis_response(ANGIOGENESIS_OPTIMAL_LOG_DOSE)
    assert in_window["inside_therapeutic_window"]
    assert in_window["perfusion_gain_fraction"] > in_window["edema_vascular_toxicity_risk"]


def test_high_dose_is_toxic_low_dose_is_futile():
    """The V-shaped response the human VEGF trials actually showed."""
    low = angiogenesis_response(-3.0)
    high = angiogenesis_response(0.5)

    assert low["perfusion_gain_fraction"] < 0.1          # nothing at low dose
    assert high["edema_vascular_toxicity_risk"] > 0.8    # oedema at high dose
    assert not high["inside_therapeutic_window"]
    assert not low["inside_therapeutic_window"]


def test_perfusion_gain_is_bounded_by_ischaemic_fraction():
    """Revascularising tissue that is already perfused does nothing."""
    tiny = angiogenesis_response(ANGIOGENESIS_OPTIMAL_LOG_DOSE, ischaemic_fraction=0.0)
    large = angiogenesis_response(ANGIOGENESIS_OPTIMAL_LOG_DOSE, ischaemic_fraction=0.4)
    assert tiny["perfusion_gain_fraction"] == pytest.approx(0.0)
    assert large["perfusion_gain_fraction"] > tiny["perfusion_gain_fraction"]


def test_toxicity_is_monotone_in_dose():
    tox = [angiogenesis_response(d)["edema_vascular_toxicity_risk"] for d in (-3, -2, -1, 0, 1)]
    assert tox == sorted(tox)


# ---------------------------------------------------------------------------
# Delivery feasibility
# ---------------------------------------------------------------------------

def test_aav9_human_transduction_is_not_the_mouse_number():
    """Near-unity transduction is a mouse result. The human number is 5-35%
    and the model must not pretend otherwise."""
    d = delivery_feasibility("aav9")
    assert 0.0 < d.cm_transduction_fraction < 0.5
    assert d.transduction_range[1] < 0.5
    assert any("CUPID 2" in issue for issue in d.blocking_issues)


def test_lnp_is_liver_tropic_not_cardiac_tropic():
    d = delivery_feasibility("lnp")
    assert d.cm_transduction_fraction < delivery_feasibility("aav9").cm_transduction_fraction
    assert any("liver" in issue.lower() for issue in d.blocking_issues)


def test_unknown_modality_rejected():
    with pytest.raises(ValueError):
        delivery_feasibility("quantum_tunnelling")


# ---------------------------------------------------------------------------
# Roadmap assembly
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def roadmap():
    return assess_cure_roadmap("pt-1", on_treatment_ldl_mg_dl=55.0)


def test_all_four_stages_are_assessed(roadmap):
    assert len(roadmap.stages) == 4


def test_plaque_reversal_is_the_only_achievable_stage(roadmap):
    achievable = [s.stage.value for s in roadmap.stages if s.achievable_today]
    assert achievable == ["plaque_reversal"]


def test_regeneration_is_labelled_proven_failed(roadmap):
    reg = next(s for s in roadmap.stages if s.stage.value == "cardiomyocyte_regeneration")
    assert reg.evidence_status is EvidenceStatus.PROVEN_FAILED
    # The failed programmes must be named, not glossed over.
    text = " ".join(reg.blocked_by)
    assert "miR-199a" in text
    assert "CUPID 2" in text


def test_verdict_reports_achievable_count(roadmap):
    assert "1 of 4" in roadmap.overall_verdict


def test_roadmap_renders(roadmap):
    text = roadmap.render()
    assert "CURE ROADMAP" in text
    assert "VERDICT" in text
    for stage in ("plaque_reversal", "scar_remodeling",
                  "cardiomyocyte_regeneration", "microvascular_angiogenesis"):
        assert stage in text


def test_failed_mechanisms_are_never_labelled_established():
    prov = cure_provenance()
    assert prov["constitutive_cardiomyocyte_proliferation"] is EvidenceStatus.PROVEN_FAILED
    assert prov["serca2a_cardiac_gene_transfer"] is EvidenceStatus.PROVEN_FAILED
    assert prov["mechanical_plaque_dissolution"] is EvidenceStatus.PROVEN_FAILED
    assert prov["ldl_lowering_plaque_regression_glagov_asteroid"] is EvidenceStatus.ESTABLISHED
