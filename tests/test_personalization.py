"""Tests for the patient personalization layer.

These assert orchestration behaviour (tiering, payload selection, eligibility,
order emission). They do NOT assert that the underlying biological models are
accurate — those are unvalidated simulations.
"""

from __future__ import annotations

import pytest

from danon.personalization import (
    IschemicIndication,
    Modality,
    NAB_EXCLUSION_TITER,
    PatientBloodPanel,
    PatientPersonalizationEngine,
    PersonalizationTier,
    Transgene,
    TRANSGENE_CATALOG,
)


@pytest.fixture
def engine() -> PatientPersonalizationEngine:
    return PatientPersonalizationEngine()


def panel(**kwargs) -> PatientBloodPanel:
    base = dict(patient_id="PT-TEST", age_years=60, weight_kg=80.0, lvef_pct=45.0)
    base.update(kwargs)
    return PatientBloodPanel(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Assessment
# ---------------------------------------------------------------------------


class TestAssessment:
    def test_seronegative_patient_is_eligible(self, engine: PatientPersonalizationEngine) -> None:
        a = engine.assess(panel(nab_titer=2.0))
        assert a.eligible
        assert not a.exclusion_reasons
        assert a.titer_classification == "negative"

    def test_titer_above_exclusion_threshold_is_ineligible(
        self, engine: PatientPersonalizationEngine
    ) -> None:
        a = engine.assess(panel(nab_titer=NAB_EXCLUSION_TITER + 1))
        assert not a.eligible
        assert any("exclusion" in r.lower() for r in a.exclusion_reasons)

    def test_low_lvef_is_excluded(self, engine: PatientPersonalizationEngine) -> None:
        a = engine.assess(panel(nab_titer=5.0, lvef_pct=20.0))
        assert not a.eligible
        assert any("LVEF" in r for r in a.exclusion_reasons)

    def test_pediatric_patient_excluded(self, engine: PatientPersonalizationEngine) -> None:
        a = engine.assess(panel(nab_titer=5.0, age_years=12))
        assert not a.eligible

    def test_elevated_c3a_raises_complement_risk(self, engine: PatientPersonalizationEngine) -> None:
        low = engine.assess(panel(nab_titer=5.0, complement_c3a_ng_ml=0.0))
        high = engine.assess(panel(nab_titer=5.0, complement_c3a_ng_ml=150.0))
        assert high.complement_risk > low.complement_risk

    def test_hla_note_surfaced_when_alleles_supplied(
        self, engine: PatientPersonalizationEngine
    ) -> None:
        a = engine.assess(panel(nab_titer=5.0, hla_alleles=("HLA-A*02:01",)))
        assert any("HLA" in n for n in a.notes)


# ---------------------------------------------------------------------------
# Transgene selection
# ---------------------------------------------------------------------------


class TestTransgeneSelection:
    def test_ldlr_pathogenic_selects_ldlr(self, engine: PatientPersonalizationEngine) -> None:
        spec = engine.select_transgene(
            panel(genotype_flags=("LDLR_pathogenic",), ldl_mg_dl=250.0),
            IschemicIndication.FAMILIAL_HYPERCHOLESTEROLEMIA,
        )
        assert spec.transgene is Transgene.LDLR

    def test_high_ldl_selects_pcsk9_knockdown(self, engine: PatientPersonalizationEngine) -> None:
        spec = engine.select_transgene(
            panel(ldl_mg_dl=210.0), IschemicIndication.STABLE_ANGINA
        )
        assert spec.transgene is Transgene.PCSK9_KD

    def test_ischemic_cardiomyopathy_selects_serca2a(
        self, engine: PatientPersonalizationEngine
    ) -> None:
        spec = engine.select_transgene(
            panel(ldl_mg_dl=90.0), IschemicIndication.CHRONIC_ISCHEMIC_CARDIOMYOPATHY
        )
        assert spec.transgene is Transgene.SERCA2A

    def test_post_mi_selects_angiogenic_payload(
        self, engine: PatientPersonalizationEngine
    ) -> None:
        spec = engine.select_transgene(panel(ldl_mg_dl=90.0), IschemicIndication.POST_MI)
        assert spec.transgene in (Transgene.VEGF_A165, Transgene.SDF1A, Transgene.SERCA2A)

    def test_every_catalog_entry_is_reachable(self) -> None:
        for spec in TRANSGENE_CATALOG:
            assert spec.indications
            assert spec.epitope_segments
            assert spec.payload_kb > 0


# ---------------------------------------------------------------------------
# Regimen assembly
# ---------------------------------------------------------------------------


class TestPersonalize:
    def test_typical_patient_gets_off_the_shelf(
        self, engine: PatientPersonalizationEngine
    ) -> None:
        r = engine.personalize(
            panel(nab_titer=4.0, ldl_mg_dl=95.0),
            IschemicIndication.CHRONIC_ISCHEMIC_CARDIOMYOPATHY,
        )
        assert r.tier is PersonalizationTier.OFF_THE_SHELF
        assert not r.epitope_edit_required

    def test_ineligible_patient_flagged(self, engine: PatientPersonalizationEngine) -> None:
        r = engine.personalize(
            panel(nab_titer=5000.0), IschemicIndication.STABLE_ANGINA
        )
        assert r.tier is PersonalizationTier.NOT_ELIGIBLE

    def test_very_high_titer_switches_modality_to_mrna(
        self, engine: PatientPersonalizationEngine
    ) -> None:
        r = engine.personalize(
            panel(nab_titer=NAB_EXCLUSION_TITER + 500), IschemicIndication.STABLE_ANGINA
        )
        assert r.modality is Modality.LNP_MRNA

    def test_promoter_is_never_ubiquitous(self, engine: PatientPersonalizationEngine) -> None:
        """A ubiquitous promoter is the documented hepatic-toxicity failure mode."""
        r = engine.personalize(panel(nab_titer=5.0), IschemicIndication.STABLE_ANGINA)
        assert "CMV" not in r.promoter_strategy
        assert "miR-122" in r.promoter_strategy

    def test_every_regimen_carries_warnings(self, engine: PatientPersonalizationEngine) -> None:
        r = engine.personalize(panel(nab_titer=5.0), IschemicIndication.STABLE_ANGINA)
        assert r.warnings
        assert any("not" in w.lower() and "cure" in w.lower() for w in r.warnings)

    def test_ratio_range_brackets_optimal_ratio(
        self, engine: PatientPersonalizationEngine
    ) -> None:
        r = engine.personalize(panel(nab_titer=300.0), IschemicIndication.STABLE_ANGINA)
        lo, hi = r.ratio_range
        assert lo <= r.empty_full_ratio <= hi

    def test_manufacturing_order_is_flat_serializable(
        self, engine: PatientPersonalizationEngine
    ) -> None:
        import json

        r = engine.personalize(panel(nab_titer=120.0), IschemicIndication.STABLE_ANGINA)
        order = r.to_manufacturing_order()
        # must survive a JSON round-trip for the LIMS / robot compiler
        assert json.loads(json.dumps(order)) == order
        assert order["patient_id"] == "PT-TEST"
        assert order["tier"] == r.tier.value
        assert order["transgene"] == r.transgene.value

    def test_render_report_contains_key_fields(
        self, engine: PatientPersonalizationEngine
    ) -> None:
        text = engine.personalize(panel(nab_titer=5.0), IschemicIndication.STABLE_ANGINA).render_report()
        for token in ("PERSONALIZED REGIMEN", "Tier", "Empty:Full", "Rationale"):
            assert token in text


# ---------------------------------------------------------------------------
# Cohort view
# ---------------------------------------------------------------------------


class TestCohortSummary:
    def test_cohort_summary_accounts_for_every_patient(
        self, engine: PatientPersonalizationEngine
    ) -> None:
        panels = [
            panel(patient_id=f"P{i:03d}", nab_titer=t)
            for i, t in enumerate([2, 20, 120, 700, 2400])
        ]
        s = engine.cohort_summary(panels, IschemicIndication.STABLE_ANGINA)
        assert s["n_patients"] == 5
        assert sum(s["tier_distribution"].values()) == 5  # type: ignore[union-attr]
        assert 0.0 <= s["ineligible_fraction"] <= 1.0

    def test_empty_cohort_does_not_divide_by_zero(
        self, engine: PatientPersonalizationEngine
    ) -> None:
        s = engine.cohort_summary([], IschemicIndication.STABLE_ANGINA)
        assert s["n_patients"] == 0
        assert s["mean_empty_full_ratio"] == 0.0
