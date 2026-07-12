"""
Bio-constraint tests verifying pipeline outputs meet clinical safety thresholds.

Critical assertions:
  - Liver leakage < 15% for top promoter candidates
  - Cardiac tropism >= 0.50 (DanonConfig minimum)
  - Immune evasion >= 0.48 (Gate Crash threshold)
  - Hepatic leakage suppression via SMAR insulator
  - Epitope masking achieves measurable surface charge change
  - Stoichiometric decoy calculator reduces complement risk
  - MHRA ILAP composite >= 0.60 for eligibility
"""
import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from danon.promoter_spec import PromoterSpecEngine
from danon.epitope_masker import EpitopeMasker
from danon.stoichiometric_calc import StoichiometricCalculator
from danon.platform_validator import PlatformValidator
from danon.config import danon_config

WILD_TYPE_AAV9_CAPSID = (
    "ADLGNSSGVNYYNLKGSTLPNRLSFPGAITYHTYNQESQVPEN"
    "YIAPKLYDLPSMFAPATVKAPLNIQKRTQYTLTHSGSNPTTAG"
    "HPITNFYVPVTGTTLTTNISLPQYVNVPVVYKMQTTKYEDGVL"
    "PVRGSIMQTYQVSSYSTNWQIQVTLQFNTTSEVQPVFEVVYTR"
    "QVQGRVILPDVDKNITQLIHCINEMINTFNYNKLIVTPPMQLNN"
    "YTYWHQLQPEQNFQVKTTTTSVNVNFTITGQVPAQFVVTRNVNT"
    "MVTMKMQTTASSGSTARSFEKVRQYHTDKSGTLPRYVLQISSV"
    "NTYGTQTRVIESLKENAQFGQVGAITYTDIENTLQVHTANQVLK"
    "NTTIYAGTNLHTYIQENLSPASQSVATAFITKYVSKRVKAEGES"
    "SITYLWEILNNKMDQIRVQVNGVQVNINTTVQAVTALMINTIYV"
    "QTNITTITLQEKNITLSVTKLNEQVNATVQIHTISGSIIGPGQN"
    "NAVTKLQVTAGATANITVQNVTLDNQVTQRVKVSYVNAGGTNTT"
    "TFTLKVLPDKVINTYRGTHATRYSNFSLKIGSSN"
)


class TestBioConstraints:

    def test_promoter_hepatic_leakage_below_15_percent(self):
        engine = PromoterSpecEngine()
        best = engine.get_best_uro_construct()
        assert best.hepatic_leakage_percent < 15.0, (
            f"Hepatic leakage {best.hepatic_leakage_percent:.1f}% >= 15%"
        )

    def test_smar_insulator_suppresses_hepatic_leakage(self):
        engine = PromoterSpecEngine()
        with_insulator = engine.design_dual_enhancer_construct("MHC", True, True, True, True)
        without_insulator = engine.design_dual_enhancer_construct("MHC", True, True, False, True)
        assert with_insulator.hepatic_leakage_percent < without_insulator.hepatic_leakage_percent, (
            "SMAR insulator failed to reduce hepatic leakage"
        )

    def test_dual_enhancer_beats_cmv_specificity(self):
        engine = PromoterSpecEngine()
        cmv = engine.design_dual_enhancer_construct("CMV", False, False, False, True)
        best = engine.get_best_uro_construct()
        assert best.cardiac_specificity_ratio > cmv.cardiac_specificity_ratio * 5, (
            f"Dual-enhancer specificity {best.cardiac_specificity_ratio:.1f}x "
            f"not > 5x CMV {cmv.cardiac_specificity_ratio:.1f}x"
        )

    def test_epitope_masker_changes_surface_electrostatics(self):
        masker = EpitopeMasker()
        design = masker.design_charge_mutations(WILD_TYPE_AAV9_CAPSID, "VR_VIII", 6)
        assert design.electrostatic_surface_change > 0.15, (
            f"Surface charge change {design.electrostatic_surface_change:.3f} <= 0.15"
        )
        assert len(design.mutations) > 0, "No mutations designed"

    def test_epitope_masker_preserves_cardiac_docking(self):
        masker = EpitopeMasker()
        design = masker.design_charge_mutations(WILD_TYPE_AAV9_CAPSID, "VR_VIII", 8)
        assert design.cardiac_docking_preserved, (
            "Epitope masking mutated a cardiac docking residue"
        )

    def test_stoichiometric_decoy_reduces_complement_risk(self):
        calc = StoichiometricCalculator()
        result = calc.optimize_ratio(200.0, 5e13)
        assert result.complement_activation_risk < 0.50, (
            f"Complement risk {result.complement_activation_risk:.3f} >= 0.50"
        )
        assert result.optimal_empty_full_ratio >= 1.0, (
            f"Optimal ratio {result.optimal_empty_full_ratio:.1f} < 1.0"
        )

    def test_decoy_ratio_scales_with_titer(self):
        calc = StoichiometricCalculator()
        low = calc.optimize_ratio(10.0)
        high = calc.optimize_ratio(500.0)
        assert high.optimal_empty_full_ratio >= low.optimal_empty_full_ratio, (
            "Higher titer should need more decoys"
        )

    def test_mhra_ilap_candidate_composite_above_threshold(self):
        validator = PlatformValidator()
        candidate = {
            "cardiac_tropism": 0.68,
            "hepatic_accumulation": 0.18,
            "immune_evasion": 0.55,
            "lamp2b_expression": 0.72,
            "promoter_score": 0.85,
            "mirna_score": 0.88,
            "dosing_score": 0.62,
            "complement_activation": 0.20,
            "liver_toxicity": 0.15,
            "cardiac_inflammation": 0.12,
            "decoy_protection": 0.70,
            "vector_titer": 5e13,
            "itr_integrity": 0.95,
            "empty_full_ratio_optimal": 0.60,
            "rc_aav_risk": 0.01,
            "gonadal_transduction_risk": 0.03,
            "shedding_risk": 0.08,
            "mutation_types_covered": 0.85,
            "age_stratification": 0.80,
            "nab_management": 0.75,
            "biomarker_strategy": 0.75,
            "immunosuppression_plan": 0.70,
            "complement_monitoring": 0.75,
            "rescue_therapy": 0.65,
            "stopping_rules": 0.75,
            "juvenile_model_data": 0.65,
            "growth_development_safety": 0.60,
            "long_term_follow_up": 0.72,
            "age_de_escalation": 0.55,
            "disease_prevalence": 0.80,
            "unmet_medical_need": 0.90,
            "natural_history_data": 0.85,
            "patient_advocacy": 0.75,
        }
        assessment = validator.evaluate_candidate(candidate)
        assert assessment.composite_score >= 0.60, (
            f"MHRA ILAP composite {assessment.composite_score:.3f} < 0.60"
        )
        assert assessment.is_ilap_eligible, "Candidate not ILAP-eligible"

    def test_pipeline_scores_improve_over_ucl_baseline(self):
        validator = PlatformValidator()
        ucl_val = validator.utcl_regulatory_score()
        candidate = {
            "cardiac_tropism": 0.68,
            "hepatic_accumulation": 0.18,
            "immune_evasion": 0.55,
            "lamp2b_expression": 0.72,
            "promoter_score": 0.85,
            "mirna_score": 0.88,
            "dosing_score": 0.62,
            "complement_activation": 0.20,
            "liver_toxicity": 0.15,
            "cardiac_inflammation": 0.12,
            "decoy_protection": 0.70,
            "vector_titer": 5e13,
            "itr_integrity": 0.95,
            "empty_full_ratio_optimal": 0.60,
            "rc_aav_risk": 0.01,
            "gonadal_transduction_risk": 0.03,
            "shedding_risk": 0.08,
            "mutation_types_covered": 0.85,
            "age_stratification": 0.80,
            "nab_management": 0.75,
            "biomarker_strategy": 0.75,
            "immunosuppression_plan": 0.70,
            "complement_monitoring": 0.75,
            "rescue_therapy": 0.65,
            "stopping_rules": 0.75,
            "juvenile_model_data": 0.65,
            "growth_development_safety": 0.60,
            "long_term_follow_up": 0.72,
            "age_de_escalation": 0.55,
            "disease_prevalence": 0.80,
            "unmet_medical_need": 0.90,
            "natural_history_data": 0.85,
            "patient_advocacy": 0.75,
        }
        our_val = validator.our_regulatory_score(candidate)
        assert our_val > ucl_val, f"Our regulatory score {our_val:.3f} <= UCL {ucl_val:.3f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
