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
    "mean_age_death_untreated_male_years": 25.0,
    "mean_age_death_untreated_female_years": 40.0,
    "male_female_ratio": 0.75,
    "cardiac_penetrance": 0.95,
    "skeletal_penetrance": 0.85,
    "intellectual_disability_rate": 0.30,
    "wolff_parkinson_white_rate": 0.60,
    "lvmi_baseline_g_m2_male": 150.0,
    "lvmi_baseline_g_m2_female": 120.0,
    "lvmi_normal_g_m2": 65.0,
    "ef_baseline_percent": 40.0,
    "ef_normal_percent": 65.0,
    "ck_elevation_baseline": 850.0,
    "annual_lvmi_increase_g_m2": 3.5,
    "annual_ef_decline_percent": 1.2,
    "transplant_5y_survival_percent": 80.0,
    "ref": "Boucek et al. 2011, Circ; Cenacchi et al. 2020, Acta Neuropathol; Marie et al. 2022",
}

AAV_GENE_THERAPY_PARAMS = {
    "zolgensma_dose_response": {"dose_vg_kg": 1.1e14, "expression_level": 0.65, "ref": "Mendell et al. 2017, NEJM"},
    "luxterna_dose_response": {"dose_vg": 1.5e11, "expression_level": 0.55, "ref": "Russell et al. 2017, Lancet"},
    "aav9_cardiac_transduction_min": 0.40,
    "aav9_cardiac_transduction_max": 0.80,
    "therapeutic_dose_kg": 5e13,
    "lamp2b_threshold_expression": 0.25,
    "onset_months": 4.5,
    "lvmi_reduction_null_max": 0.35,
    "lvmi_reduction_missense_max": 0.25,
    "ef_improvement_scale": 12.0,
    "survival_benefit_null_5y": 0.25,
    "survival_benefit_missense_5y": 0.15,
    "refs": [
        "Zsebo et al. 2014, Mol Ther",
        "Katz et al. 2019, JCI",
        "Ylä-Herttuala 2019, Mol Ther",
        "Mingozzi et al. 2023",
    ],
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

        dose_factor = 1.0 - np.exp(-dosing_score * 1e14 / p["therapeutic_dose_kg"])
        cardiac_transduction = p["aav9_cardiac_transduction_min"] + (
            p["aav9_cardiac_transduction_max"] - p["aav9_cardiac_transduction_min"]
        ) * cardiac_tropism
        promoter_activity = 0.5 + 0.5 * promoter_score
        immune_clearance = 1.0 - immune_evasion
        expression = float(np.clip(
            dose_factor * cardiac_transduction * promoter_activity * (1.0 - immune_clearance),
            0.01, 1.0
        ))

        mutation = MUTATION_CLASSIFICATION.get(mutation_type, MUTATION_CLASSIFICATION["null"])
        is_null = mutation_type in ["null", "splice_site"]
        mutation_bonus = 0.3 if is_null else 0.0
        lvmi_reduction_max = p["lvmi_reduction_null_max"] if is_null else p["lvmi_reduction_missense_max"]
        survival_benefit_max = p["survival_benefit_null_5y"] if is_null else p["survival_benefit_missense_5y"]

        lvmi_baseline_male = self.nh["lvmi_baseline_g_m2_male"]
        lvmi_baseline_female = self.nh["lvmi_baseline_g_m2_female"]
        lvmi_baseline = lvmi_baseline_male
        ef_baseline = self.nh["ef_baseline_percent"]
        years_until_death = self.nh["mean_age_death_untreated_male_years"]

        lvmi_reduction_1y = float(np.clip(
            lvmi_reduction_max * np.sqrt(expression) * (1.0 + mutation_bonus) * 100.0,
            0, 40
        ))
        lvmi_reduction_5y = float(np.clip(
            lvmi_reduction_max * np.sqrt(expression) * (1.0 + mutation_bonus) * 100.0,
            0, 60
        ))
        baseline_severity = (lvmi_baseline - self.nh["lvmi_normal_g_m2"]) / lvmi_baseline * 100.0
        ef_improvement_1y = float(np.clip(
            p["ef_improvement_scale"] * expression * (1.0 - baseline_severity / 100.0),
            0, 15
        ))
        ef_improvement_5y = float(np.clip(ef_improvement_1y * 1.8, 0, 25))

        hazard_untreated = 1.0 / max(years_until_death - age_at_treatment, 1.0)
        surv_5y_untreated = float(np.clip(np.exp(-hazard_untreated * 5.0), 0.05, 0.95))
        surv_5y_treated = float(np.clip(
            surv_5y_untreated + survival_benefit_max * expression,
            0.1, 0.99
        ))
        surv_10y_untreated = float(np.clip(np.exp(-hazard_untreated * 10.0), 0.01, 0.80))
        surv_10y_treated = float(np.clip(
            surv_10y_untreated + survival_benefit_max * expression * 0.7,
            0.05, 0.99
        ))

        absolute_risk_reduction = surv_5y_treated - surv_5y_untreated
        nnt = float(np.clip(
            1.0 / max(absolute_risk_reduction, 0.01),
            1, 50
        ))

        benefit = (
            0.30 * expression +
            0.25 * (lvmi_reduction_1y / 40.0) +
            0.20 * (ef_improvement_1y / 15.0) +
            0.15 * ((surv_5y_treated - surv_5y_untreated) / 0.30) +
            0.10 * (1.0 - hepatic_avoidance * 0.3)
        )
        clinical_benefit = float(np.clip(benefit, 0, 1))

        approved = (
            expression >= 0.20 and
            immune_evasion >= 0.30 and
            (mutation_type != "null" or expression >= 0.30)
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
        hazard_u = 1.0 / max(self.nh["mean_age_death_untreated_male_years"] - outcome.age_at_treatment_years, 1)
        hazard_t = 1.0 / max(self.nh["mean_age_death_untreated_male_years"] - outcome.age_at_treatment_years + 10, 1)
        for yi in range(1, 11):
            surv_u.append(max(surv_u[-1] * np.exp(-hazard_u), 0))
            surv_t.append(max(surv_t[-1] * np.exp(-hazard_t), 0))
        return {"years": y, "untreated": surv_u, "treated": surv_t}
