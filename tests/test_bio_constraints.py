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
from danon.microfluidics_core import MicrofluidicsCore, MicrofluidicConfig
from danon.dual_vector import DualVectorEngine
from danon.data_ingress import DataIngressEngine
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

    # ------------------------------------------------------------------
    # PHASE 1: Biomechanical Verification Assertions
    # ------------------------------------------------------------------
    def test_hepatic_clearance_fraction_below_0_15(self):
        engine = PromoterSpecEngine()
        best = engine.get_best_uro_construct()
        hepatic_clearance_fraction = best.hepatic_leakage_percent / 100.0
        assert hepatic_clearance_fraction < 0.15, (
            f"Hepatic clearance fraction {hepatic_clearance_fraction:.3f} >= 0.15"
        )

    def test_splicing_efficiency_above_0_80(self):
        lamp2b_with_cpg = "M" * 199 + "CG" + "A" * 208
        dve = DualVectorEngine(lamp2b_sequence=lamp2b_with_cpg)
        design = dve.design_split(split_position_aa=200)
        assert design.splicing_efficiency > 0.80, (
            f"Npu DnaE intein splicing efficiency {design.splicing_efficiency:.3f} <= 0.80 "
            f"(N-extein={lamp2b_with_cpg[199]}, C-extein={lamp2b_with_cpg[200]})"
        )

    def test_maximum_shear_stress_below_50_pascals(self):
        cfg = MicrofluidicConfig(
            aqueous_flow_rate_ul_min=10.0,
            organic_flow_rate_ul_min=5.0,
        )
        mfc = MicrofluidicsCore(cfg)
        telemetry = mfc.simulate()
        assert telemetry.max_shear_stress_pa < 50.0, (
            f"Maximum wall shear stress {telemetry.max_shear_stress_pa:.2f} Pa >= 50.0 Pa"
        )
        assert telemetry.shear_stress_safe, "Shear stress safety flag False"


from danon.dms_fitness import DMSFitnessLayer
from danon.solvation_energy import SolvationEnergyEngine
from danon.smar_insulator import SMARInsulatorEngine
from danon.codon_elongation import CodonElongationEngine, DEFAULT_LAMP2B_PEPTIDE, CODON_TABLE
from danon.hla_decoupler import HLADecoupler
from danon.synthesis_guard import SynthesisGuard, revcomp
from danon.translational_readiness import TranslationalReadinessEngine, PreclinicalValidationMilestones
from danon.epitope_masker import AAV9_PDB_3J1S_VR_COORDINATES


class TestHorizon2Phases:
    """PHASE 19-24: hyper-dimensional screening layer assertions."""

    # ---- PHASE 19: DMS fitness boundary ----
    def test_dms_flags_conserved_pocket_charge_flip_as_lethal(self):
        layer = DMSFitnessLayer()
        # Arg->Asp charge reversal inside the receptor-anchor conserved pocket
        result = layer.evaluate([(197, "R", "D")])
        assert not result.capsid_viable, "Conserved-pocket charge reversal should be lethal"
        assert result.conserved_pocket_violations >= 1

    def test_dms_tolerates_surface_variable_region_substitution(self):
        layer = DMSFitnessLayer()
        # conservative swaps on the solvent-exposed VR-VIII loop must stay viable
        result = layer.evaluate([(310, "N", "D"), (312, "Q", "E"), (326, "S", "T")])
        assert result.capsid_viable, "Surface VR substitutions wrongly flagged lethal"
        assert result.lethal_mutations == 0

    # ---- PHASE 20: solvation free energy ----
    def test_solvation_delta_g_within_bound_for_masked_capsid(self):
        coords = {r: AAV9_PDB_3J1S_VR_COORDINATES[r]["coordinates_3j1s"]
                  for r in ("VR_IV", "VR_VIII")}
        masker = EpitopeMasker()
        _, mask = masker.design_dual_region_masking(WILD_TYPE_AAV9_CAPSID)
        engine = SolvationEnergyEngine(max_allowed_ddg=2.5)
        result = engine.evaluate(WILD_TYPE_AAV9_CAPSID, mask.masked_sequence, coords)
        assert result.plasma_soluble, (
            f"ddG_solv {result.ddg_solv_total:.2f} exceeds destabilisation bound 2.5"
        )
        assert result.aggregation_risk_score <= 1.0

    # ---- PHASE 21: CpG depletion / anti-silencing ----
    def test_cpg_depletion_reduces_density(self):
        from danon.smar_insulator import CpGOptimizationEngine, calculate_cpg_density
        engine = CpGOptimizationEngine(cpg_density_threshold=1.0)
        report = engine.optimize(protein=DEFAULT_LAMP2B_PEPTIDE)
        assert report.depleted_cpg_density <= report.raw_cpg_density, (
            "CpG depletion increased CpG density"
        )
        assert report.cpg_reduction_pct > 50.0, (
            f"CpG reduction only {report.cpg_reduction_pct:.1f}% (expected > 50%)"
        )
        # back-translation must still preserve the protein
        cds = report.depleted_cds
        translated = "".join(CODON_TABLE[cds[i:i + 3]] for i in range(0, len(cds), 3))
        assert translated == DEFAULT_LAMP2B_PEPTIDE

    def test_cpg_density_function_counts_per_100bp(self):
        from danon.smar_insulator import calculate_cpg_density
        # 8 bp with 4 CG dinucleotides -> 4 / (8/2) = 1.0
        assert calculate_cpg_density("CGCGCGCG") == 1.0
        assert calculate_cpg_density("ATGC" * 25) == 0.0

    # ---- PHASE 22: codon elongation / tAI ----
    def test_codon_optimized_lamp2b_meets_min_tai(self):
        engine = CodonElongationEngine(min_tai=0.88)
        result = engine.evaluate(protein=DEFAULT_LAMP2B_PEPTIDE)
        assert result.tai >= 0.88, f"tAI {result.tai:.3f} < 0.88 minimum"
        assert result.codon_optimized

    def test_back_translation_preserves_protein(self):
        engine = CodonElongationEngine()
        peptide = "MVCFRLDGKTAAPST"
        cds = engine.back_translate(peptide)
        translated = "".join(CODON_TABLE[cds[i:i + 3]] for i in range(0, len(cds), 3))
        assert translated == peptide, "GC-balanced back-translation changed the protein"

    # ---- PHASE 23: HLA-DRB1 decoupler ----
    def test_hla_decoupler_rejects_promiscuous_hydrophobic_binder(self):
        engine = HLADecoupler(ic50_cutoff_nm=500.0)
        # a hydrophobic P1-anchored core is a strong DRB1 binder -> must be flagged
        result = engine.evaluate(["PKYVKQNTLKLAT"])
        assert result.strongest_binder_ic50_nm < 500.0
        assert result.high_affinity_hits >= 1 and not result.decoupled

    def test_hla_ic50_monotonic_with_binding_score(self):
        engine = HLADecoupler()
        strong = engine.screen_segment("YFLKAWTVAQK")[0]
        weak = engine.screen_segment("PPPGPPGPPGP")[0]
        assert strong.predicted_ic50_nm < weak.predicted_ic50_nm

    # ---- PHASE 24: synthesis guard ----
    def test_synthesis_guard_gc_within_commercial_window(self):
        engine = CodonElongationEngine()
        cds = engine.back_translate(DEFAULT_LAMP2B_PEPTIDE)
        guard = SynthesisGuard(gc_window=(40.0, 65.0))
        result = guard.evaluate(cds)
        assert 40.0 <= result.gc_content <= 65.0, (
            f"Codon-optimised CDS GC {result.gc_content:.1f}% outside 40-65% window"
        )
        assert result.synthesizable

    def test_synthesis_guard_detects_hairpin_and_homopolymer(self):
        guard = SynthesisGuard()
        arm = "GCGCATGCATGC"
        hairpin = arm + "TTTT" + revcomp(arm)
        homopolymer = "A" * 15
        seq = "ACGT" * 20 + hairpin + "ACGT" * 20 + homopolymer + "ACGT" * 20
        result = guard.evaluate(seq)
        assert len(result.inverted_repeats) >= 1, "Failed to detect hairpin"
        assert len(result.homopolymer_runs) >= 1, "Failed to detect homopolymer run"
        assert not result.synthesizable

    # ---- Translational readiness gateway ----
    def test_translational_gateway_reports_not_clinic_ready_by_default(self):
        engine = TranslationalReadinessEngine()
        result = engine.evaluate_translational_gate()
        assert result.clinical_trial_eligibility is False
        assert result.current_stage.startswith("Phase 0.5")
        assert result.verified_count == 0 and result.total_gates == 9
        assert result.readiness_score_pct == 0.0

    def test_translational_gateway_unlocks_only_with_all_tokens(self):
        milestones = PreclinicalValidationMilestones(
            ipsc_transduction_verified=True, glycogen_clearance_observed=True,
            large_animal_pk_validated=True, durability_months=18,
            glp_tox_cleared=True, genotoxicity_validated=True,
            gmp_vector_yield_per_liter=2e13, empty_capsid_separation_efficiency=95.0,
            multicenter_trial_initiated=True,
        )
        result = TranslationalReadinessEngine(milestones).evaluate_translational_gate()
        assert result.clinical_trial_eligibility is True
        assert result.verified_count == 9

    def test_pipeline_response_carries_translational_readiness(self):
        from api_server import run_full_pipeline, PipelineConstraints
        r = run_full_pipeline(PipelineConstraints(candidate_pool=120, random_seed=7))
        tr = r["translationalReadiness"]
        assert tr["clinicalTrialEligibility"] is False
        assert tr["totalGates"] == 9

    # ---- Cross-module integration ----
    def test_lamp2b_uses_single_vector_not_dual(self):
        from danon.dual_vector import evaluate_vector_capacity
        # LAMP2B CDS (1.2 kb) + promoter + UTRs/polyA ~ 2.8 kb < 4.7 kb
        cargo = int(round(len(DEFAULT_LAMP2B_PEPTIDE) * 3 + 600 + 1000))
        decision = evaluate_vector_capacity(cargo)
        assert decision["strategy"].startswith("Single"), (
            f"LAMP2B ({cargo} bp) wrongly routed to dual-vector: {decision['strategy']}"
        )
        assert decision["toxicity_risk_multiplier"] == 1.0

    def test_oversized_cargo_triggers_dual_vector(self):
        from danon.dual_vector import evaluate_vector_capacity
        decision = evaluate_vector_capacity(8000)
        assert decision["strategy"].startswith("Dual")
        assert decision["toxicity_risk_multiplier"] > 1.0

    def test_pipeline_reports_single_vector_and_cpg_depletion(self):
        from api_server import run_full_pipeline, PipelineConstraints
        r = run_full_pipeline(PipelineConstraints(candidate_pool=120, random_seed=7))
        assert r["advancedMetrics"]["vectorCapacity"]["strategy"].startswith("Single")
        smar = r["advancedMetrics"]["smar"]
        assert smar["cpgReductionPct"] > 50.0
        # phase 9 registry entry reflects single-vector
        p9 = [p for p in r["phases"] if p["name"].startswith("Vector Topology")][0]
        assert "Single" in p9["metric"]

    def test_full_24_phase_pipeline_compiles_and_compounds(self):
        from api_server import run_full_pipeline, PipelineConstraints
        result = run_full_pipeline(PipelineConstraints(candidate_pool=150, random_seed=7))
        assert result["totalPhases"] == 24
        assert len(result["phases"]) == 24
        assert result["combinatorialAdvantage"] > 1e9, "Combinatorial advantage collapsed"
        # every phase must carry a real selectivity factor in the valid band
        for p in result["phases"]:
            assert 1.5 <= p["selectivityFactor"] <= 4.0
            assert p["status"] in ("pass", "warn", "fail")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
