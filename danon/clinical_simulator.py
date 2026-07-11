"""
Clinical Simulator: Projects patient outcomes from candidate expression profiles.

Uses published Danon Disease natural history data and AAV gene therapy
dose-response relationships to predict:
  - LAMP2 protein expression in cardiomyocytes
  - Reduction in left ventricular mass index (LVMI)
  - Improvement in ejection fraction (EF)
  - Survival probability over 10 years
  - Optimal patient stratification by mutation type
"""
import logging
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DANON_NATURAL_HISTORY = {
    "mean_age_diagnosis_years": 17.0,
    "mean_age_death_untreated_years": 25.0,
    "male_female_ratio": 0.75,
    "cardiac_penetrance": 0.95,
    "skeletal_penetrance": 0.85,
    "intellectual_disability_rate": 0.30,
    "wolff_parkinson_white_rate": 0.60,
    "lvmi_baseline_g_m2": 120.0,
    "lvmi_normal_g_m2": 65.0,
    "ef_baseline_percent": 55.0,
    "ef_normal_percent": 65.0,
    "ck_elevation_baseline": 850.0,
    "annual_lvmi_increase_g_m2": 3.5,
    "annual_ef_decline_percent": 1.2,
    "ref": "Boucek et al. 2011, Circ; Cenacchi et al. 2020, Acta Neuropathol",
}

AAV_GENE_THERAPY_PARAMS = {
    "zolgensma_dose_response": {"dose_vg_kg": 1.1e14, "expression_level": 0.65, "ref": "Mendell et al. 2017, NEJM"},
    "luxterna_dose_response": {"dose_vg": 1.5e11, "expression_level": 0.55, "ref": "Russell et al. 2017, Lancet"},
    "aav9_cardiac_expression_ratio": 0.35,
    "promoter_boost_mhc_vs_cmv": 2.5,
    "mirna_detarget_liver_reduction": 0.85,
    "immune_stealth_evasion_boost": 0.35,
    "dual_vector_splicing_efficiency": 0.82,
    "empty_capsid_decoy_effect": 0.40,
    "refs": ["Zsebo et al. 2014, Mol Ther", "Katz et al. 2019, JCI"],
}

MUTATION_CLASSIFICATION = {
    "null": {
        "description": "Complete loss of LAMP2 protein",
        "examples": ["frameshift", "nonsense", "large deletion"],
        "frequency": 0.40,
        "severity_score": 1.0,
        "projected_response": "Best — no pre-existing LAMP2 to trigger immune response against",
    },
    "missense_catalytic": {
        "description": "Single amino acid change affecting function",
        "examples": ["G93R", "Q155P", "G232R"],
        "frequency": 0.35,
        "severity_score": 0.70,
        "projected_response": "Good — residual protein may reduce immunogenicity",
    },
    "splice_site": {
        "description": "Splice junction mutation → truncated protein",
        "examples": ["c.64+1G>A", "c.507-1G>C"],
        "frequency": 0.15,
        "severity_score": 0.85,
        "projected_response": "Good — depending on residual wild-type splicing",
    },
    "partial_gene_deletion": {
        "description": "Exon-level deletion with residual reading frame",
        "examples": ["exon 2-3 deletion", "exon 6 deletion"],
        "frequency": 0.10,
        "severity_score": 0.60,
        "projected_response": "Fair — need to confirm residual LAMP2 expression",
    },
}


@dataclass
class PatientOutcome:
    mutation_type: str
    mutation_example: str
    age_at_treatment_years: float
    baseline_lvmi: float
    baseline_ef: float
    predicted_lamp2_expression: float
    predicted_lvmi_reduction_at_1y: float
    predicted_lvmi_reduction_at_5y: float
    predicted_ef_improvement_at_1y: float
    predicted_ef_improvement_at_5y: float
    survival_at_5y_untreated: float
    survival_at_5y_treated: float
    survival_at_10y_untreated: float
    survival_at_10y_treated: float
    number_needed_to_treat: float
    clinical_benefit_score: float
    is_approved_for_trial: bool


class ClinicalSimulator:
    def __init__(self):
        self.nh = DANON_NATURAL_HISTORY
        self.p = AAV_GENE_THERAPY_PARAMS

    def simulate_outcome(self, candidate_fitness: float,
                         cardiac_tropism: float,
                         hepatic_avoidance: float,
                         immune_evasion: float,
                         lamp2b_expression: float,
                         promoter_score: float,
                         mirna_score: float,
                         dosing_score: float,
                         mutation_type: str = "null",
                         age_at_treatment: float = 10.0) -> PatientOutcome:
        p = self.p
        base_expression = p["aav9_cardiac_expression_ratio"]
        expression = base_expression * (
            1.0 + (cardiac_tropism - 0.5) * 2.0 +
            (promoter_score - 0.5) * 1.5 +
            (mirna_score - 0.5) * 0.8 +
            (immune_evasion - 0.5) * 0.6 +
            (dosing_score - 0.5) * 0.8
        ) * min(candidate_fitness * 2, 1.5)
        expression = float(np.clip(expression, 0.05, 1.0))

        mutation = MUTATION_CLASSIFICATION.get(mutation_type, MUTATION_CLASSIFICATION["null"])
        mutation_effect = mutation["severity_score"]

        years_until_death_untreated = max(self.nh["mean_age_death_untreated_years"] - age_at_treatment, 1.0)

        lvmi_annual_change = self.nh["annual_lvmi_increase_g_m2"]
        lvmi_baseline = self.nh["lvmi_baseline_g_m2"] + age_at_treatment * lvmi_annual_change
        ef_baseline = self.nh["ef_baseline_percent"] - age_at_treatment * self.nh["annual_ef_decline_percent"]

        expr_effect = expression / 0.5
        lvmi_reduction_1y = float(np.clip(15 * expr_effect * mutation_effect, 0, 30))
        lvmi_reduction_5y = float(np.clip(40 * expr_effect * mutation_effect, 0, 60))
        ef_improvement_1y = float(np.clip(5 * expr_effect * mutation_effect, 0, 12))
        ef_improvement_5y = float(np.clip(10 * expr_effect * mutation_effect, 0, 20))

        hazard_untreated = 1.0 - 1.0 / years_until_death_untreated
        surv_5y_untreated = float(np.clip((1.0 - hazard_untreated) ** 5, 0.1, 0.99))
        surv_5y_treated = float(np.clip(surv_5y_untreated + 0.15 * expr_effect * mutation_effect, 0.1, 0.99))
        surv_10y_untreated = float(np.clip((1.0 - hazard_untreated) ** 10, 0.01, 0.95))
        surv_10y_treated = float(np.clip(surv_10y_untreated + 0.25 * expr_effect * mutation_effect, 0.01, 0.99))

        nnt = float(np.clip(1.0 / max((surv_5y_treated - surv_5y_untreated) * 10, 0.05), 1, 20))

        benefit = (
            0.25 * expression +
            0.20 * (lvmi_reduction_1y / 30) +
            0.15 * (ef_improvement_1y / 12) +
            0.20 * ((surv_5y_treated - 0.5) / 0.5) +
            0.10 * (1.0 - (nnt - 1) / 19) +
            0.10 * (1.0 - hepatic_avoidance * 0.5)
        )
        clinical_benefit = float(np.clip(benefit, 0, 1))

        approved = (
            mutation_type != "null" or
            (expression >= 0.30 and immune_evasion >= 0.45)
        )

        return PatientOutcome(
            mutation_type=mutation_type,
            mutation_example=mutation["examples"][0] if mutation["examples"] else "unknown",
            age_at_treatment_years=age_at_treatment,
            baseline_lvmi=float(lvmi_baseline),
            baseline_ef=float(ef_baseline),
            predicted_lamp2_expression=expression,
            predicted_lvmi_reduction_at_1y=lvmi_reduction_1y,
            predicted_lvmi_reduction_at_5y=lvmi_reduction_5y,
            predicted_ef_improvement_at_1y=ef_improvement_1y,
            predicted_ef_improvement_at_5y=ef_improvement_5y,
            survival_at_5y_untreated=surv_5y_untreated,
            survival_at_5y_treated=surv_5y_treated,
            survival_at_10y_untreated=surv_10y_untreated,
            survival_at_10y_treated=surv_10y_treated,
            number_needed_to_treat=nnt,
            clinical_benefit_score=clinical_benefit,
            is_approved_for_trial=approved,
        )

    def stratify_patients(self, candidate_fitness: float,
                          cardiac_tropism: float,
                          hepatic_avoidance: float,
                          immune_evasion: float,
                          lamp2b: float,
                          promoter: float,
                          mirna: float,
                          dosing: float) -> List[PatientOutcome]:
        results = []
        for mut_type in MUTATION_CLASSIFICATION:
            age = 10.0 if mut_type in ["null", "splice_site"] else 15.0
            outcome = self.simulate_outcome(
                candidate_fitness, cardiac_tropism, hepatic_avoidance,
                immune_evasion, lamp2b, promoter, mirna, dosing,
                mutation_type=mut_type, age_at_treatment=age
            )
            results.append(outcome)
        results.sort(key=lambda x: x.clinical_benefit_score, reverse=True)
        return results

    def print_trial_design(self, outcomes: List[PatientOutcome]):
        logger.info("  CLINICAL TRIAL DESIGN — Recommended Entry Criteria")
        logger.info("  ---------------------------------------------------")
        for o in outcomes:
            logger.info("  %-25s age>=%2d  LVMI>=%.0f  EF>=%.0f%%  NNT=%.1f  benefit=%.2f  %s",
                         o.mutation_type.replace("_", " ").title(),
                         int(o.age_at_treatment_years),
                         o.baseline_lvmi, o.baseline_ef,
                         o.number_needed_to_treat,
                         o.clinical_benefit_score,
                         "APPROVED" if o.is_approved_for_trial else "EXCLUDED")

    def survival_curve_data(self, outcome: PatientOutcome) -> Dict:
        y = list(range(11))
        surv_u = [1.0]
        surv_t = [1.0]
        hazard_u = 1.0 / max(self.nh["mean_age_death_untreated_years"] - outcome.age_at_treatment_years, 1)
        hazard_t = 1.0 - (outcome.survival_at_5y_treated - 0.5) / 5
        for yi in range(1, 11):
            surv_u.append(max(surv_u[-1] * (1.0 - hazard_u), 0))
            surv_t.append(np.clip(surv_t[-1] * hazard_t, 0, 1))
        return {"years": y, "untreated": surv_u, "treated": surv_t}
