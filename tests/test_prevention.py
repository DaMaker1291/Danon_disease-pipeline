"""Tests for the pre-event prevention engine.

The single most important test in this file is
``test_pooled_cohort_reproduces_published_example``: it pins the risk
equations to a published worked example. Everything else in the module is
downstream of that number being right, and the coefficients are transcribed
by hand, so this is the test that catches a transcription error.

Several tests are explicit regression tests for bugs found while building the
module, and are labelled as such.
"""

from __future__ import annotations

import math

import pytest

from danon.prevention import (
    CTT_MORTALITY_RR_PER_MMOL_L,
    CTT_MVE_RR_CI95,
    CTT_MVE_RR_PER_MMOL_L,
    LPA_HIGH_NMOL_L,
    LPA_VERY_HIGH_MG_DL,
    MG_DL_PER_MMOL_L_CHOLESTEROL,
    PCR_DOMAIN,
    FHClassification,
    FHInputs,
    PopulationPreventionCampaign,
    PreventionAction,
    RaceStratum,
    RiskEnhancers,
    RiskEquation,
    RiskFactorProfile,
    Sex,
    TherapyRung,
    ascvd_risk_10y,
    assess,
    dutch_lipid_clinic_score,
    ldl_exposure_years,
    prevention_constant_audit,
    prevention_provenance,
    project_benefit,
    risk_category,
    routine_ldl_reduction,
    triage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _goff_reference_profile() -> RiskFactorProfile:
    """The worked example published with the Pooled Cohort Equations:
    55 years old, total cholesterol 213 mg/dL, HDL 50 mg/dL, untreated
    systolic BP 120 mmHg, non-smoker, non-diabetic."""
    return RiskFactorProfile(
        patient_id="reference",
        age_years=55,
        sex=Sex.MALE,
        race=RaceStratum.WHITE,
        chol_total_mgdl=213,
        chol_hdl_mgdl=50,
        bp_sys_mmhg=120,
        on_bp_medication=False,
        current_smoker=False,
        diabetes=False,
    )


# ---------------------------------------------------------------------------
# Risk equations: the anchor tests
# ---------------------------------------------------------------------------

def test_pooled_cohort_reproduces_published_example():
    """Goff 2013 worked example, white male: 5.38%.

    This value is what the reference implementation returns for the unrounded
    coefficients, and it is the check that validates the coefficient
    transcription, the treated/untreated SBP split, and the
    `S0 ** exp(sum - mean)` form all at once.
    """
    risk = ascvd_risk_10y(_goff_reference_profile(), RiskEquation.GOFF_2013)
    assert risk == pytest.approx(0.0538, abs=0.0002)


def test_all_four_strata_produce_plausible_risk():
    """The published example is 4 strata of the same patient. All four must
    land in a low single-digit risk band, with men above women and Black
    above White, as the published table shows."""
    risks = {}
    for sex in (Sex.FEMALE, Sex.MALE):
        for race in (RaceStratum.BLACK, RaceStratum.WHITE):
            p = RiskFactorProfile("x", 55, sex, race, 213, 50, 120, False, False, False)
            risks[(sex, race)] = ascvd_risk_10y(p, RiskEquation.GOFF_2013)

    assert all(0.005 < r < 0.12 for r in risks.values())
    assert risks[(Sex.MALE, RaceStratum.WHITE)] > risks[(Sex.FEMALE, RaceStratum.WHITE)]
    assert risks[(Sex.MALE, RaceStratum.BLACK)] > risks[(Sex.MALE, RaceStratum.WHITE)]
    assert risks[(Sex.FEMALE, RaceStratum.BLACK)] > risks[(Sex.FEMALE, RaceStratum.WHITE)]


@pytest.mark.parametrize(
    "field,value",
    [
        ("chol_total_mgdl", 400.0),
        ("bp_sys_mmhg", 240.0),
    ],
)
def test_risk_rises_when_a_risk_factor_worsens(field, value):
    """Raising total cholesterol or blood pressure, or getting older, must
    raise risk. A model that fails this is not a risk model."""
    base = _goff_reference_profile()
    worse = _goff_reference_profile()
    setattr(worse, field, value)

    assert ascvd_risk_10y(worse, RiskEquation.GOFF_2013) > ascvd_risk_10y(
        base, RiskEquation.GOFF_2013
    )


def test_risk_rises_with_age():
    base = _goff_reference_profile()
    older = _goff_reference_profile()
    older.age_years = 75
    assert ascvd_risk_10y(older, RiskEquation.GOFF_2013) > ascvd_risk_10y(
        base, RiskEquation.GOFF_2013
    )


def test_risk_falls_when_hdl_rises():
    """HDL is protective, so it moves in the opposite direction to every
    other input in the equation. A sign error here would be invisible
    without this test."""
    base = _goff_reference_profile()
    protective = _goff_reference_profile()
    protective.chol_hdl_mgdl = 75
    assert ascvd_risk_10y(protective, RiskEquation.GOFF_2013) < ascvd_risk_10y(
        base, RiskEquation.GOFF_2013
    )

    harmful = _goff_reference_profile()
    harmful.chol_hdl_mgdl = 30
    assert ascvd_risk_10y(harmful, RiskEquation.GOFF_2013) > ascvd_risk_10y(
        base, RiskEquation.GOFF_2013
    )


def test_smoking_and_diabetes_raise_risk():
    base = ascvd_risk_10y(_goff_reference_profile(), RiskEquation.GOFF_2013)

    smoker = _goff_reference_profile()
    smoker.current_smoker = True
    assert ascvd_risk_10y(smoker, RiskEquation.GOFF_2013) > base

    diabetic = _goff_reference_profile()
    diabetic.diabetes = True
    assert ascvd_risk_10y(diabetic, RiskEquation.GOFF_2013) > base


def test_treated_sbp_is_not_double_counted():
    """On a treated SBP, the untreated coefficient must switch off entirely.

    Regression test: a naive implementation adds both the treated and
    untreated log-SBP terms, which roughly doubles the blood-pressure
    contribution and inflates every hypertensive patient's risk.
    """
    treated = _goff_reference_profile()
    treated.bp_sys_mmhg = 140
    treated.on_bp_medication = True

    untreated = _goff_reference_profile()
    untreated.bp_sys_mmhg = 140
    untreated.on_bp_medication = False

    r_treated = ascvd_risk_10y(treated, RiskEquation.GOFF_2013)
    r_untreated = ascvd_risk_10y(untreated, RiskEquation.GOFF_2013)

    # On medication carries a slightly higher coefficient, but the two must be
    # the same order of magnitude. If both terms were summed, the treated
    # value would be far larger.
    assert 0.5 < r_treated / r_untreated < 2.0


def test_yadlowsky_is_lower_than_goff_for_midlife_profile():
    """The 2018 refit exists because the original equations over-predict in
    contemporary populations. For a midlife profile it must predict less."""
    p = _goff_reference_profile()
    goff = ascvd_risk_10y(p, RiskEquation.GOFF_2013)
    yadlowsky = ascvd_risk_10y(p, RiskEquation.YADLOWSKY_2018)
    assert yadlowsky < goff


def test_yadlowsky_risk_is_in_unit_interval():
    for age in (40, 50, 60, 70, 79):
        for female in (True, False):
            p = RiskFactorProfile(
                "x", age, Sex.FEMALE if female else Sex.MALE, RaceStratum.WHITE,
                260, 35, 160, True, True, True,
            )
            r = ascvd_risk_10y(p, RiskEquation.YADLOWSKY_2018)
            assert 0.0 <= r <= 1.0


def test_unknown_equation_is_rejected():
    with pytest.raises(ValueError):
        ascvd_risk_10y(_goff_reference_profile(), "not-an-equation")  # type: ignore[arg-type]


def test_risk_categories_match_acc_aha_thresholds():
    assert risk_category(0.02) == "low"
    assert risk_category(0.06) == "borderline"
    assert risk_category(0.12) == "intermediate"
    assert risk_category(0.25) == "high"
    # Exact boundaries.
    assert risk_category(0.05) == "borderline"
    assert risk_category(0.075) == "intermediate"
    assert risk_category(0.20) == "high"


# ---------------------------------------------------------------------------
# Input-domain honesty
# ---------------------------------------------------------------------------

def test_out_of_domain_inputs_are_flagged_not_silently_accepted():
    p = _goff_reference_profile()
    p.age_years = 95  # outside the validated 40-79 range
    a = assess(p)
    assert a.outside_validated_domain
    assert any("age_years" in v for v in a.outside_validated_domain)
    assert any("extrapolation" in n for n in a.notes)


def test_domain_bounds_are_the_published_ones():
    assert PCR_DOMAIN["age_years"] == (40.0, 79.0)
    assert PCR_DOMAIN["chol_total_mgdl"] == (130.0, 320.0)


# ---------------------------------------------------------------------------
# Risk enhancers
# ---------------------------------------------------------------------------

def test_enhancers_do_not_reclassify_low_risk_patients():
    """Applying enhancers to low-risk patients would reclassify almost
    everyone upward and destroy the model's discrimination."""
    young = RiskFactorProfile("y", 45, Sex.FEMALE, RaceStratum.WHITE, 170, 65, 110,
                              False, False, False)
    a = assess(young, RiskEnhancers(lipoprotein_a_nmol_l=300, hscrp_mg_l=5,
                                    family_history_premature_ascvd=True))
    assert a.risk_category == "low"
    assert a.reclassified_risk_10y == pytest.approx(a.risk_10y)
    assert a.effective_risk_10y == pytest.approx(a.risk_10y)


def test_enhancers_reclassify_borderline_risk_upward():
    """Find a profile in the borderline band, then verify that a very high
    Lp(a) pushes it into the treatment range."""
    p = RiskFactorProfile("b", 60, Sex.MALE, RaceStratum.WHITE, 215, 45, 138,
                          True, False, False)
    base = assess(p)
    assert base.risk_category in ("borderline", "intermediate")

    boosted = assess(p, RiskEnhancers(lipoprotein_a_nmol_l=LPA_VERY_HIGH_MG_DL))
    assert boosted.reclassified_risk_10y > base.risk_10y
    assert boosted.effective_risk_10y > base.risk_10y


def test_zero_cac_reclassifies_downward():
    """A CAC score of zero moves intermediate-risk patients DOWN, which is
    the entire reason the guideline recommends measuring it before
    committing someone to lifelong therapy.

    Regression test: an earlier version took max(equation, reclassified) for
    the effective risk, which silently discarded every downward
    reclassification and made the CAC score meaningless.
    """
    p = RiskFactorProfile("c", 60, Sex.MALE, RaceStratum.WHITE, 215, 45, 138,
                          True, False, False)
    base = assess(p)
    assert base.risk_category in ("borderline", "intermediate")

    down = assess(p, RiskEnhancers(cac_score=0.0))
    assert down.reclassified_risk_10y < base.risk_10y
    assert down.effective_risk_10y < base.risk_10y


def test_unmeasured_enhancers_are_reported_as_missing():
    """Absence of a measurement must never be presented as absence of risk."""
    a = assess(_goff_reference_profile(), RiskEnhancers())
    assert "lipoprotein_a" in a.unmeasured_enhancers
    assert "apob" in a.unmeasured_enhancers


def test_absent_family_history_is_a_negative_not_a_missing_value():
    """Regression test: family history is a question you always ask, so
    reporting it as 'unmeasured' misrepresents a clean history as ignorance."""
    a = assess(_goff_reference_profile(), RiskEnhancers(family_history_premature_ascvd=False))
    assert "family_history" not in a.unmeasured_enhancers


def test_apob_discordance_requires_high_apob_and_low_ldl():
    p = RiskFactorProfile("d", 55, Sex.MALE, RaceStratum.WHITE, 170, 55, 120,
                          False, False, False)
    discordant = assess(p, RiskEnhancers(apob_mg_dl=140, ldl_mg_dl=85))
    assert "apob_discordant" in discordant.enhancer_flags

    concordant = assess(p, RiskEnhancers(apob_mg_dl=140, ldl_mg_dl=150))
    assert "apob_discordant" not in concordant.enhancer_flags


# ---------------------------------------------------------------------------
# Familial hypercholesterolaemia (Dutch Lipid Clinic Network)
# ---------------------------------------------------------------------------

def test_dlcn_ldl_bands_match_published_cutpoints():
    """The bands are defined in mmol/L in the source criteria."""
    def score_for(ldl_mg_dl: float) -> int:
        return dutch_lipid_clinic_score(FHInputs(untreated_ldl_mg_dl=ldl_mg_dl)).score

    mm = MG_DL_PER_MMOL_L_CHOLESTEROL
    assert score_for(3.0 * mm) == 0      # below 4.0 mmol/L
    assert score_for(4.5 * mm) == 1      # 4.0-4.9
    assert score_for(5.5 * mm) == 3      # 5.0-6.4
    assert score_for(7.0 * mm) == 5      # 6.5-8.4
    assert score_for(9.0 * mm) == 8      # >= 8.5


def test_dlcn_classification_thresholds():
    assert dutch_lipid_clinic_score(
        FHInputs(untreated_ldl_mg_dl=4.5 * MG_DL_PER_MMOL_L_CHOLESTEROL)
    ).classification is FHClassification.UNLIKELY

    possible = dutch_lipid_clinic_score(
        FHInputs(untreated_ldl_mg_dl=5.5 * MG_DL_PER_MMOL_L_CHOLESTEROL)
    )
    assert possible.score == 3
    assert possible.classification is FHClassification.POSSIBLE

    definite = dutch_lipid_clinic_score(FHInputs(
        untreated_ldl_mg_dl=9.0 * MG_DL_PER_MMOL_L_CHOLESTEROL,
        tendon_xanthomata=True,
    ))
    assert definite.score == 14
    assert definite.classification is FHClassification.DEFINITE
    assert definite.cascade_screening_recommended


def test_cascade_screening_only_for_probable_or_definite():
    unlikely = dutch_lipid_clinic_score(FHInputs(untreated_ldl_mg_dl=4.2 * MG_DL_PER_MMOL_L_CHOLESTEROL))
    assert not unlikely.cascade_screening_recommended

    probable = dutch_lipid_clinic_score(FHInputs(
        untreated_ldl_mg_dl=7.0 * MG_DL_PER_MMOL_L_CHOLESTEROL,
        family_history_premature_cad=True,
    ))
    assert probable.score == 6
    assert probable.classification is FHClassification.PROBABLE
    assert probable.cascade_screening_recommended


def test_genetic_confirmation_alone_reaches_definite():
    r = dutch_lipid_clinic_score(FHInputs(
        untreated_ldl_mg_dl=5.5 * MG_DL_PER_MMOL_L_CHOLESTEROL,
        genetic_test_confirmed=True,
    ))
    assert r.score == 11
    assert r.classification is FHClassification.DEFINITE


# ---------------------------------------------------------------------------
# Cumulative exposure
# ---------------------------------------------------------------------------

def test_ldl_exposure_accumulates_with_age_and_level():
    young = ldl_exposure_years(160, 30)
    old = ldl_exposure_years(160, 60)
    assert old > young

    moderate = ldl_exposure_years(160, 60)
    high = ldl_exposure_years(250, 60)
    assert high > moderate


def test_ldl_exposure_is_zero_at_reference_level():
    """At exactly the 1.8 mmol/L reference there is no excess exposure, no
    matter how long the patient has lived."""
    reference_mg_dl = 1.8 * MG_DL_PER_MMOL_L_CHOLESTEROL
    assert ldl_exposure_years(reference_mg_dl, 60) == pytest.approx(0.0, abs=1e-9)
    assert ldl_exposure_years(reference_mg_dl, 80) == pytest.approx(0.0, abs=1e-9)

    # And just below the reference it floors at zero rather than going negative.
    assert ldl_exposure_years(reference_mg_dl - 20.0, 60) == 0.0


# ---------------------------------------------------------------------------
# Treatment effect: the CTT engine
# ---------------------------------------------------------------------------

def test_ldl_reduction_increases_with_rung():
    ldl = 180.0
    ladder = [
        TherapyRung.LIFESTYLE,
        TherapyRung.MODERATE_STATIN,
        TherapyRung.HIGH_INTENSITY_STATIN,
        TherapyRung.STATIN_EZETIMIBE,
        TherapyRung.PCSK9_OR_INCLISIRAN,
    ]
    reductions = [routine_ldl_reduction(r, ldl) for r in ladder]
    assert reductions == sorted(reductions)
    assert all(r > 0 for r in reductions)


def test_ldl_reduction_cannot_remove_cholesterol_that_is_not_there():
    """A drug cannot remove circulating cholesterol that does not exist.
    Low baseline LDL must self-limit."""
    tiny = routine_ldl_reduction(TherapyRung.ONESHOT_BASE_EDITING, 30.0)
    assert tiny <= 30.0 / MG_DL_PER_MMOL_L_CHOLESTEROL
    assert tiny >= 0.0


def test_nnt_falls_as_baseline_risk_rises():
    """Higher baseline risk means more events to prevent and a smaller NNT.
    This is the entire argument for risk-stratified treatment."""
    nnts = [
        project_benefit(r, 160, TherapyRung.HIGH_INTENSITY_STATIN, age_years=60).number_needed_to_treat
        for r in (0.05, 0.10, 0.20, 0.40)
    ]
    assert nnts == sorted(nnts, reverse=True)


def test_nnt_lands_in_the_published_statin_range():
    """Published 10-year NNTs for statin primary prevention are roughly
    10-50 depending on baseline risk. A result outside this range means the
    hazard model has been mis-applied."""
    for risk in (0.075, 0.15, 0.25):
        nnt = project_benefit(
            risk, 160, TherapyRung.HIGH_INTENSITY_STATIN, age_years=60
        ).number_needed_to_treat
        assert 4.0 < nnt < 80.0


def test_arr_is_not_compounded_across_years():
    """Regression test for the single easiest way to get this wrong.

    The CTT proportional reduction applies to the ANNUAL event rate. Applying
    it to the cumulative 10-year risk instead -- i.e. 0.78 ** (delta * 10) --
    predicts a ~92% event reduction for a 1 mmol/L drop, which no trial has
    ever shown and which would make every therapy look like a cure.
    """
    b = project_benefit(0.20, 160, TherapyRung.HIGH_INTENSITY_STATIN, years=10,
                        age_years=60)
    # A roughly 2 mmol/L reduction must not eliminate most of the risk.
    assert b.ldl_reduction_mmol_l > 1.5
    assert b.absolute_risk_reduction < 0.15
    assert b.relative_risk_reduction < 0.60

    # And the treated risk must remain strictly positive.
    assert b.treated_risk_10y > 0.0


def test_more_ldl_reduction_gives_more_benefit():
    arrs = [
        project_benefit(0.20, 200, rung, age_years=60).absolute_risk_reduction
        for rung in (
            TherapyRung.LIFESTYLE,
            TherapyRung.MODERATE_STATIN,
            TherapyRung.HIGH_INTENSITY_STATIN,
            TherapyRung.PCSK9_OR_INCLISIRAN,
        )
    ]
    assert arrs == sorted(arrs)


def test_mortality_benefit_is_smaller_than_event_benefit():
    """CTT found a ~22% reduction in vascular events per mmol/L but only ~10%
    for all-cause mortality, because most deaths are not vascular. A model
    where the mortality benefit matches the event benefit is wrong."""
    b = project_benefit(0.20, 180, TherapyRung.HIGH_INTENSITY_STATIN, age_years=65)
    assert 0.0 < b.mortality_benefit_10y < b.absolute_risk_reduction


def test_lifecycle_effect_sizes_match_published_values():
    assert CTT_MVE_RR_PER_MMOL_L == 0.78
    assert CTT_MVE_RR_CI95 == (0.76, 0.80)
    assert CTT_MORTALITY_RR_PER_MMOL_L == 0.90


def test_low_baseline_risk_yields_no_negative_benefit():
    b = project_benefit(0.001, 90, TherapyRung.LIFESTYLE, age_years=45)
    assert b.absolute_risk_reduction >= 0.0
    assert b.mortality_benefit_10y >= 0.0


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------

def test_fh_index_case_is_flagged_for_statin_first_not_the_experimental_option():
    """Regression test.

    A 42-year-old with an untreated LDL of 265 mg/dL scores 'low' on a
    10-year horizon because 10-year equations are dominated by age. An earlier
    version of this code then made the long-horizon investigational option the
    primary action, which is clinically backwards.
    """
    profile = RiskFactorProfile(
        "fh", 42, Sex.MALE, RaceStratum.WHITE, 330, 42, 128, False, False, False,
        ldl_mg_dl=265,
    )
    enh = RiskEnhancers(ldl_mg_dl=265, family_history_premature_ascvd=True)
    fh = dutch_lipid_clinic_score(FHInputs(
        untreated_ldl_mg_dl=265, family_history_premature_cad=True
    ))
    d = triage(profile, enh, fh=fh)

    assert d.primary_action is PreventionAction.HIGH_INTENSITY_STATIN
    assert d.recommended_rung is TherapyRung.HIGH_INTENSITY_STATIN
    assert PreventionAction.CASCADE_SCREEN_FAMILY in d.actions
    assert PreventionAction.ONESHOT_THERAPY_CANDIDATE in d.actions


def test_primary_action_is_never_a_long_horizon_option_ahead_of_therapy():
    """In any indicated patient the primary action must be an immediate
    therapy step, not a screening or investigational one."""
    profile = RiskFactorProfile(
        "x", 50, Sex.MALE, RaceStratum.WHITE, 330, 40, 130, False, False, False,
        ldl_mg_dl=270,
    )
    fh = dutch_lipid_clinic_score(FHInputs(untreated_ldl_mg_dl=270))
    d = triage(profile, RiskEnhancers(ldl_mg_dl=270), fh=fh)
    assert d.primary_action not in (
        PreventionAction.ONESHOT_THERAPY_CANDIDATE,
        PreventionAction.CASCADE_SCREEN_FAMILY,
    )


def test_young_high_exposure_patient_gets_a_lifetime_projection():
    profile = RiskFactorProfile(
        "fh", 42, Sex.MALE, RaceStratum.WHITE, 330, 42, 128, False, False, False,
        ldl_mg_dl=265,
    )
    fh = dutch_lipid_clinic_score(FHInputs(untreated_ldl_mg_dl=265))
    d = triage(profile, RiskEnhancers(ldl_mg_dl=265), fh=fh)

    assert not d.risk_score_reliable
    assert d.lifetime_benefit is not None
    assert d.lifetime_benefit.years > 10
    # Over a lifetime the case is far stronger than the 10-year view suggests.
    assert d.lifetime_benefit.number_needed_to_treat < d.benefit.number_needed_to_treat
    assert any("not the right quantity" in r for r in d.rationale)


def test_low_risk_patient_is_not_offered_pharmacotherapy():
    profile = RiskFactorProfile("low", 42, Sex.FEMALE, RaceStratum.WHITE,
                                165, 68, 108, False, False, False, ldl_mg_dl=85)
    d = triage(profile)
    assert d.primary_action in (
        PreventionAction.LIFESTYLE_ONLY, PreventionAction.MEASURE_ENHANCERS
    )
    assert d.benefit is None
    assert any("No treatment indicated" in w for w in d.warnings)


def test_borderline_patient_is_sent_for_enhancer_measurement():
    """The guideline answer at borderline risk is a measurement, not a pill."""
    profile = RiskFactorProfile("b", 60, Sex.MALE, RaceStratum.WHITE, 215, 45,
                                138, True, False, False, ldl_mg_dl=140)
    d = triage(profile, RiskEnhancers(ldl_mg_dl=140))
    if 0.05 <= d.risk_10y < 0.075:
        assert d.primary_action is PreventionAction.MEASURE_ENHANCERS
        assert d.benefit is None


def test_very_high_risk_patient_gets_treatment_and_a_projection():
    profile = RiskFactorProfile(
        "h", 68, Sex.MALE, RaceStratum.WHITE, 240, 38, 152, True, True, True,
        ldl_mg_dl=165,
    )
    d = triage(profile, RiskEnhancers(ldl_mg_dl=165))
    assert d.risk_10y > 0.20
    assert d.recommended_rung is TherapyRung.HIGH_INTENSITY_STATIN
    assert d.benefit is not None
    assert d.benefit.absolute_risk_reduction > 0.05
    assert d.nnt < 15


def test_secondary_prevention_is_recognised():
    profile = RiskFactorProfile("s", 62, Sex.MALE, RaceStratum.WHITE, 180, 40,
                                128, True, False, False, ldl_mg_dl=105)
    d = triage(profile, RiskEnhancers(ldl_mg_dl=105), established_ascvd=True)
    assert PreventionAction.SECONDARY_PREVENTION in d.actions


def test_high_ldl_is_an_indication_regardless_of_calculated_risk():
    """LDL >= 190 mg/dL is a statin indication on its own. A young patient
    whose 10-year score is low must still be treated."""
    profile = RiskFactorProfile("y", 38, Sex.FEMALE, RaceStratum.WHITE, 300, 60,
                                115, False, False, False, ldl_mg_dl=210)
    d = triage(profile, RiskEnhancers(ldl_mg_dl=210))
    assert d.primary_action is PreventionAction.HIGH_INTENSITY_STATIN
    assert d.benefit is not None


def test_triage_report_renders_without_error():
    d = triage(_goff_reference_profile())
    text = d.render()
    assert "PREVENTION TRIAGE" in text
    assert "10-year ASCVD risk" in text


# ---------------------------------------------------------------------------
# Population campaign
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def campaign_report():
    c = PopulationPreventionCampaign(baseline_statin_coverage=0.55)
    return c.run(n_patients=40_000, bootstrap_resamples=100, seed=7)


def test_population_nnt_confidence_interval_contains_the_point_estimate(campaign_report):
    """Regression test.

    An earlier version averaged the CTT interval over the whole population
    instead of the eligible subset, which diluted it with patients who
    contribute no benefit and produced a CI (45-54) that excluded the point
    estimate (26). A confidence interval that excludes its own point estimate
    is always a bug.
    """
    r = campaign_report
    lo, hi = r.mean_nnt_ci95
    assert lo <= r.mean_nnt <= hi
    assert lo < hi


def test_population_sampling_ci_brackets_the_estimate(campaign_report):
    r = campaign_report
    for point, (lo, hi) in (
        (r.events_prevented, r.events_prevented_ci95),
        (r.deaths_prevented, r.deaths_prevented_ci95),
    ):
        assert lo <= point <= hi


def test_population_eligibility_fraction_is_physiological(campaign_report):
    """Roughly 30-55% of adults aged 40-79 are statin-eligible in US data.
    A wildly different number means the indication logic is broken."""
    fraction = campaign_report.statin_eligible / campaign_report.n_patients
    assert 0.20 < fraction < 0.65


def test_population_deaths_prevented_is_smaller_than_events_prevented(campaign_report):
    assert 0 < campaign_report.deaths_prevented < campaign_report.events_prevented


def test_treatment_gap_scales_with_coverage():
    model_low = PopulationPreventionCampaign(baseline_statin_coverage=0.10).run(
        n_patients=20_000, bootstrap_resamples=0, seed=3
    )
    model_high = PopulationPreventionCampaign(baseline_statin_coverage=0.90).run(
        n_patients=20_000, bootstrap_resamples=0, seed=3
    )
    # Same population, same seed: only the coverage differs.
    assert model_low.treatment_gap_count > model_high.treatment_gap_count
    assert model_low.events_prevented > model_high.events_prevented


def test_vectorised_and_scalar_risk_agree():
    """The batch path duplicates the scalar equations for speed. If they ever
    disagree, the campaign is answering a different question than the
    single-patient API."""
    campaign = PopulationPreventionCampaign()
    pop = campaign._generate(n=2000, seed=99)
    vector = campaign._risk_vector(pop)

    goff = PopulationPreventionCampaign(equation=RiskEquation.GOFF_2013)
    goff_vector = goff._risk_vector(pop)

    for i in range(0, 2000, 137):
        profile = RiskFactorProfile(
            patient_id=f"p{i}",
            age_years=float(pop["age"][i]),
            sex=Sex.FEMALE if pop["female"][i] else Sex.MALE,
            race=RaceStratum.BLACK if pop["black"][i] else RaceStratum.WHITE,
            chol_total_mgdl=float(pop["tc"][i]),
            chol_hdl_mgdl=float(pop["hdl"][i]),
            bp_sys_mmhg=float(pop["sbp"][i]),
            on_bp_medication=bool(pop["bp_meds"][i]),
            current_smoker=bool(pop["smoker"][i]),
            diabetes=bool(pop["diabetes"][i]),
        )
        assert vector[i] == pytest.approx(
            ascvd_risk_10y(profile, RiskEquation.YADLOWSKY_2018), abs=1e-9
        )
        assert goff_vector[i] == pytest.approx(
            ascvd_risk_10y(profile, RiskEquation.GOFF_2013), abs=1e-9
        )


def test_campaign_is_deterministic_for_a_fixed_seed():
    a = PopulationPreventionCampaign().run(n_patients=10_000, bootstrap_resamples=0, seed=42)
    b = PopulationPreventionCampaign().run(n_patients=10_000, bootstrap_resamples=0, seed=42)
    assert a.events_prevented == pytest.approx(b.events_prevented)
    assert a.statin_eligible == b.statin_eligible


# ---------------------------------------------------------------------------
# Real cohort intake
# ---------------------------------------------------------------------------

def _cohort_table(n: int = 400):
    """A cohort export shaped the way real ones are: aliased column names and
    string categoricals rather than booleans."""
    import numpy as np

    rng = np.random.default_rng(11)
    return {
        "age_years": rng.integers(45, 76, n).astype(float).tolist(),
        "sex": ["female" if x else "male" for x in rng.random(n) < 0.5],
        "race": ["black" if x else "white" for x in rng.random(n) < 0.14],
        "total_cholesterol": rng.normal(205, 40, n).clip(130, 320).tolist(),
        "hdl_cholesterol": rng.normal(50, 14, n).clip(20, 100).tolist(),
        "systolic_bp": rng.normal(132, 19, n).clip(90, 200).tolist(),
        "antihypertensive": ["yes" if x else "no" for x in rng.random(n) < 0.35],
        "current_smoker": ["no"] * n,
        "diabetes": ["yes" if x else "no" for x in rng.random(n) < 0.15],
        "ldl_mg_dl": rng.normal(130, 35, n).clip(40, 300).tolist(),
    }


def test_from_table_accepts_aliased_columns_and_string_categoricals():
    import numpy as np

    campaign = PopulationPreventionCampaign.from_table(_cohort_table(300))
    report = campaign.run(bootstrap_resamples=0)
    assert report.n_patients == 300
    assert report.dropped_rows == 0
    assert report.from_real_cohort
    assert 0.0 < report.mean_risk_10y < 1.0
    assert 0.20 < report.statin_eligible / report.n_patients < 0.70


def test_from_table_drops_invalid_rows_and_says_so():
    """Dropping rows changes the denominator of every rate, so the count must
    be reported rather than swallowed."""
    table = _cohort_table(200)
    table["systolic_bp"][5] = float("nan")
    table["ldl_mg_dl"][9] = float("nan")

    report = PopulationPreventionCampaign.from_table(table).run(bootstrap_resamples=0)
    assert report.n_patients == 198
    assert report.dropped_rows == 2


def test_from_table_rejects_missing_required_columns_with_a_useful_message():
    with pytest.raises(ValueError) as excinfo:
        PopulationPreventionCampaign.from_table({"age": [50, 60]})
    message = str(excinfo.value)
    assert "tc" in message
    # The error must name the accepted aliases, or the user cannot fix it.
    assert "chol_total_mgdl" in message


def test_from_table_accepts_iterable_of_row_dicts():
    rows = [
        {
            "age": 58, "sex": "male", "tc": 210, "hdl": 44, "sbp": 138,
            "bp_meds": False, "smoker": False, "diabetes": False, "ldl": 141,
        },
        {
            "age": 66, "sex": "female", "tc": 245, "hdl": 52, "sbp": 152,
            "bp_meds": True, "smoker": False, "diabetes": True, "ldl": 168,
        },
    ]
    campaign = PopulationPreventionCampaign.from_table(rows)
    pop = campaign._generate()
    risk = campaign._risk_vector(pop)
    indicated = campaign._statin_indicated(pop, risk)

    # Patient 1 is 58 with a 6.8% risk: borderline, no other indication, and
    # deliberately NOT treated. Patient 2 is diabetic at 18%, which is a
    # statin indication independent of the calculated risk.
    assert risk[0] < 0.075 < risk[1]
    assert list(indicated) == [False, True]

    report = campaign.run(bootstrap_resamples=0)
    assert report.n_patients == 2
    assert report.statin_eligible == 1


def test_from_table_rejects_mismatched_column_lengths():
    table = _cohort_table(50)
    table["hdl_cholesterol"] = table["hdl_cholesterol"][:40]
    with pytest.raises(ValueError):
        PopulationPreventionCampaign.from_table(table)


def test_table_campaign_nnt_interval_also_brackets_its_estimate():
    report = PopulationPreventionCampaign.from_table(_cohort_table(1500)).run(
        bootstrap_resamples=60
    )
    lo, hi = report.mean_nnt_ci95
    assert lo <= report.mean_nnt <= hi


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def test_provenance_is_declared_for_every_load_bearing_number():
    prov = prevention_provenance()
    assert prov["pooled_cohort_goff_2013_coefficients"].value == "literature"
    assert prov["ctt_mve_rr_per_mmol_l"].value == "literature"
    assert prov["dutch_lipid_clinic_point_system"].value == "literature"
    # The things that are genuinely assumptions must be labelled as such.
    assert prov["population_risk_factor_distributions"].value == "assumed"
    assert prov["baseline_statin_coverage"].value == "assumed"


def test_constant_audit_matches_literature_values():
    audit = {c.name: c for c in prevention_constant_audit()}
    assert audit["ctt_mve_rr_per_mmol_l"].value == 0.78
    assert audit["ctt_mve_rr_per_mmol_l"].provenance.value == "literature"
    assert audit["ctt_mortality_rr_per_mmol_l"].value == 0.90
    assert audit["baseline_statin_coverage"].provenance.value == "assumed"
    assert all(c.unit for c in audit.values())
