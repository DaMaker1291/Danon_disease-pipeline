"""
Platform Validator: Automated MHRA ILAP FastTrack regulatory scoring.

Evaluates each candidate vector against the Medicines and Healthcare
products Regulatory Agency (MHRA) Innovation Licensing and Access Pathway
(ILAP) criteria for rare disease gene therapies.

Scoring dimensions reflect the MHRA ILAP assessment framework for
monogenic hypertrophic cardiomyopathy gene therapies:
  1. Product Quality & Manufacturing
  2. Non-Clinical Safety (biodistribution, shedding, germline)
  3. Clinical Safety (immune response, complement, liver toxicity)
  4. Clinical Efficacy (cardiac transduction, LAMP2 expression, LVMI reduction)
  5. Patient Stratification (mutation type, age, baseline LVMI/EF)
  6. Risk Management (immunosuppression, rescue therapy)
  7. Pediatric Extrapolation (relevant for age 10+ onset)
  8. Orphan/Rare Disease Status (Danon: ~15,000 patients worldwide)

Reference: MHRA ILAP Guidance 2023, EMA CAT recommendations for AAV GTMPs.
"""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

MHRA_ILAP_DIMENSIONS = {
    "product_quality": {
        "label": "Product Quality & Manufacturing Consistency",
        "weight": 0.15,
        "criteria": [
            "ITR integrity", "vector genome titer", "empty/full ratio",
            "replication-competent AAV", "endotoxin", "sterility"
        ]
    },
    "non_clinical_safety": {
        "label": "Non-Clinical Safety & Biodistribution",
        "weight": 0.15,
        "criteria": [
            "vector biodistribution", "shedding profile", "germline transmission",
            "insertional mutagenesis", "off-target transduction"
        ]
    },
    "clinical_safety": {
        "label": "Clinical Safety & Tolerability",
        "weight": 0.20,
        "criteria": [
            "complement activation", "liver toxicity", "cardiac inflammation",
            "neutralizing antibodies", "cellular immune response"
        ]
    },
    "clinical_efficacy": {
        "label": "Clinical Efficacy & Surrogate Endpoints",
        "weight": 0.20,
        "criteria": [
            "cardiac transduction efficiency", "LAMP2 protein expression",
            "LVMI reduction at 1 year", "EF preservation", "autophagy rescue"
        ]
    },
    "patient_stratification": {
        "label": "Patient Stratification & Biomarker Strategy",
        "weight": 0.10,
        "criteria": [
            "mutation-type stratification", "age at treatment",
            "baseline LVMI threshold", "NAb titer exclusion",
            "concomitant medication management"
        ]
    },
    "risk_management": {
        "label": "Risk Management & Mitigation Strategy",
        "weight": 0.08,
        "criteria": [
            "immunosuppression protocol", "complement monitoring",
            "rescue therapy availability", "stopping rules"
        ]
    },
    "pediatric_extrapolation": {
        "label": "Pediatric Extrapolation & Juvenile Toxicity",
        "weight": 0.07,
        "criteria": [
            "juvenile animal model data", "growth/development impact",
            "long-term follow-up plan", "age-de-escalation strategy"
        ]
    },
    "orphan_disease": {
        "label": "Orphan/Rare Disease & Unmet Medical Need",
        "weight": 0.05,
        "criteria": [
            "disease prevalence", "existing treatment options",
            "natural history data", "patient advocacy engagement"
        ]
    }
}

# MHRA ILAP scoring thresholds (0-1 scale for each dimension)
MHRA_PASS_THRESHOLDS = {
    "product_quality": 0.60,
    "non_clinical_safety": 0.60,
    "clinical_safety": 0.65,
    "clinical_efficacy": 0.55,
    "patient_stratification": 0.50,
    "risk_management": 0.55,
    "pediatric_extrapolation": 0.45,
    "orphan_disease": 0.70,
}

# UCL/GOSH NCT03882437 baseline regulatory scores (CMV promoter, wild-type AAV9)
UCL_REGULATORY_BASELINE = {
    "product_quality": 0.65,
    "non_clinical_safety": 0.60,
    "clinical_safety": 0.45,
    "clinical_efficacy": 0.50,
    "patient_stratification": 0.55,
    "risk_management": 0.50,
    "pediatric_extrapolation": 0.55,
    "orphan_disease": 0.75,
}


@dataclass
class RegulatoryDimensionScore:
    dimension: str
    label: str
    score: float
    weight: float
    pass_threshold: float
    passed: bool
    criteria_scores: Dict[str, float]


@dataclass
class RegulatoryAssessment:
    product_quality: RegulatoryDimensionScore
    non_clinical_safety: RegulatoryDimensionScore
    clinical_safety: RegulatoryDimensionScore
    clinical_efficacy: RegulatoryDimensionScore
    patient_stratification: RegulatoryDimensionScore
    risk_management: RegulatoryDimensionScore
    pediatric_extrapolation: RegulatoryDimensionScore
    orphan_disease: RegulatoryDimensionScore
    composite_score: float
    weighted_score: float
    is_ilap_eligible: bool
    critical_gaps: List[str]
    improvements_needed: List[str]


class PlatformValidator:
    def __init__(self):
        self.dimensions = MHRA_ILAP_DIMENSIONS
        self.thresholds = MHRA_PASS_THRESHOLDS
        self.ucl_baseline = UCL_REGULATORY_BASELINE

    def evaluate_candidate(self, candidate_metrics: Dict[str, float]) -> RegulatoryAssessment:
        dims = {}

        dims["product_quality"] = self._score_product_quality(candidate_metrics)
        dims["non_clinical_safety"] = self._score_non_clinical_safety(candidate_metrics)
        dims["clinical_safety"] = self._score_clinical_safety(candidate_metrics)
        dims["clinical_efficacy"] = self._score_clinical_efficacy(candidate_metrics)
        dims["patient_stratification"] = self._score_patient_stratification(candidate_metrics)
        dims["risk_management"] = self._score_risk_management(candidate_metrics)
        dims["pediatric_extrapolation"] = self._score_pediatric_extrapolation(candidate_metrics)
        dims["orphan_disease"] = self._score_orphan_disease(candidate_metrics)

        dim_objects = {}
        for key, (score, criteria) in dims.items():
            info = self.dimensions[key]
            threshold = self.thresholds[key]
            dim_objects[key] = RegulatoryDimensionScore(
                dimension=key,
                label=info["label"],
                score=score,
                weight=info["weight"],
                pass_threshold=threshold,
                passed=score >= threshold,
                criteria_scores=criteria,
            )

        composite = np.mean([dim_objects[k].score for k in dim_objects])
        weighted = sum(dim_objects[k].score * dim_objects[k].weight for k in dim_objects)

        critical_gaps = []
        improvements = []
        for key, dim_obj in dim_objects.items():
            if not dim_obj.passed:
                critical_gaps.append(f"{dim_obj.label}: {dim_obj.score:.3f} (need {dim_obj.pass_threshold:.2f})")
                improvements.append(f"Improve {dim_obj.label} by {(dim_obj.pass_threshold - dim_obj.score):.3f} points")

        ilap_eligible = (
            dim_objects["clinical_safety"].passed and
            dim_objects["clinical_efficacy"].passed and
            dim_objects["product_quality"].passed and
            composite >= 0.60
        )

        return RegulatoryAssessment(
            product_quality=dim_objects["product_quality"],
            non_clinical_safety=dim_objects["non_clinical_safety"],
            clinical_safety=dim_objects["clinical_safety"],
            clinical_efficacy=dim_objects["clinical_efficacy"],
            patient_stratification=dim_objects["patient_stratification"],
            risk_management=dim_objects["risk_management"],
            pediatric_extrapolation=dim_objects["pediatric_extrapolation"],
            orphan_disease=dim_objects["orphan_disease"],
            composite_score=float(np.clip(composite, 0, 1)),
            weighted_score=float(np.clip(weighted, 0, 1)),
            is_ilap_eligible=ilap_eligible,
            critical_gaps=critical_gaps,
            improvements_needed=improvements,
        )

    def _score_product_quality(self, m: Dict[str, float]) -> Tuple[float, Dict]:
        itr = m.get("itr_integrity", 0.9)
        empty_full = 1.0 - abs(m.get("empty_full_ratio_optimal", 1.0) - 0.5) * 0.5
        rc_aav = 1.0 - m.get("rc_aav_risk", 0.02) * 10
        titer = min(m.get("vector_titer", 1e14) / 1e14, 1.0)
        score = 0.25 * itr + 0.25 * empty_full + 0.25 * float(np.clip(rc_aav, 0, 1)) + 0.25 * float(np.clip(titer, 0, 1))
        criteria = {"ITR_integrity": itr, "empty_full_ratio": float(np.clip(empty_full, 0, 1)),
                     "RC_AAV": float(np.clip(rc_aav, 0, 1)), "titer": float(np.clip(titer, 0, 1))}
        return float(np.clip(score, 0, 1)), criteria

    def _score_non_clinical_safety(self, m: Dict[str, float]) -> Tuple[float, Dict]:
        cardiac = m.get("cardiac_tropism", 0.5)
        hepatic = 1.0 - m.get("hepatic_accumulation", 0.3)
        germline = 1.0 - m.get("gonadal_transduction_risk", 0.05)
        shedding = 1.0 - m.get("shedding_risk", 0.10)
        score = 0.30 * cardiac + 0.30 * hepatic + 0.20 * germline + 0.20 * shedding
        criteria = {"cardiac_tropism": cardiac, "hepatic_avoidance": float(np.clip(hepatic, 0, 1)),
                     "germline_safety": float(np.clip(germline, 0, 1)), "shedding_control": float(np.clip(shedding, 0, 1))}
        return float(np.clip(score, 0, 1)), criteria

    def _score_clinical_safety(self, m: Dict[str, float]) -> Tuple[float, Dict]:
        complement = 1.0 - m.get("complement_activation", 0.3)
        liver_tox = 1.0 - m.get("liver_toxicity", 0.2)
        cardiac_inflam = 1.0 - m.get("cardiac_inflammation", 0.15)
        immune = m.get("immune_evasion", 0.5)
        decoy_effect = m.get("decoy_protection", 0.4)
        score = 0.30 * complement + 0.25 * liver_tox + 0.20 * cardiac_inflam + 0.15 * immune + 0.10 * decoy_effect
        criteria = {"complement_avoidance": float(np.clip(complement, 0, 1)),
                     "liver_safety": float(np.clip(liver_tox, 0, 1)),
                     "cardiac_inflammation_control": float(np.clip(cardiac_inflam, 0, 1)),
                     "immune_evasion": immune, "decoy_efficacy": decoy_effect}
        return float(np.clip(score, 0, 1)), criteria

    def _score_clinical_efficacy(self, m: Dict[str, float]) -> Tuple[float, Dict]:
        cardiac_trans = m.get("cardiac_tropism", 0.5)
        lamp2b = m.get("lamp2b_expression", 0.5)
        promoter = m.get("promoter_score", 0.5)
        mirna = m.get("mirna_score", 0.5)
        dosing = m.get("dosing_score", 0.5)
        score = 0.30 * cardiac_trans + 0.30 * lamp2b + 0.15 * promoter + 0.15 * mirna + 0.10 * dosing
        criteria = {"cardiac_transduction": cardiac_trans, "LAMP2_expression": lamp2b,
                     "promoter_efficacy": promoter, "miRNA_detarget": mirna, "dosing_optimization": dosing}
        return float(np.clip(score, 0, 1)), criteria

    def _score_patient_stratification(self, m: Dict[str, float]) -> Tuple[float, Dict]:
        mutation_coverage = m.get("mutation_types_covered", 0.85)
        age_strat = m.get("age_stratification", 0.80)
        nab_management = m.get("nab_management", 0.70)
        bio_strategy = m.get("biomarker_strategy", 0.75)
        score = 0.30 * mutation_coverage + 0.25 * age_strat + 0.25 * nab_management + 0.20 * bio_strategy
        criteria = {"mutation_coverage": mutation_coverage, "age_stratification": age_strat,
                     "NAb_management": nab_management, "biomarkers": bio_strategy}
        return float(np.clip(score, 0, 1)), criteria

    def _score_risk_management(self, m: Dict[str, float]) -> Tuple[float, Dict]:
        immune_supp = m.get("immunosuppression_plan", 0.65)
        complement_mon = m.get("complement_monitoring", 0.70)
        rescue = m.get("rescue_therapy", 0.60)
        stopping = m.get("stopping_rules", 0.75)
        score = 0.30 * immune_supp + 0.30 * complement_mon + 0.20 * rescue + 0.20 * stopping
        criteria = {"immunosuppression": immune_supp, "complement_monitoring": complement_mon,
                     "rescue_therapy": rescue, "stopping_rules": stopping}
        return float(np.clip(score, 0, 1)), criteria

    def _score_pediatric_extrapolation(self, m: Dict[str, float]) -> Tuple[float, Dict]:
        juvenile_data = m.get("juvenile_model_data", 0.60)
        growth_safety = m.get("growth_development_safety", 0.55)
        long_term = m.get("long_term_follow_up", 0.70)
        age_de_escalation = m.get("age_de_escalation", 0.50)
        score = 0.30 * juvenile_data + 0.30 * growth_safety + 0.25 * long_term + 0.15 * age_de_escalation
        criteria = {"juvenile_data": juvenile_data, "growth_safety": growth_safety,
                     "long_term_FU": long_term, "age_de_escalation": age_de_escalation}
        return float(np.clip(score, 0, 1)), criteria

    def _score_orphan_disease(self, m: Dict[str, float]) -> Tuple[float, Dict]:
        prevalence = m.get("disease_prevalence", 0.80)
        unmet_need = m.get("unmet_medical_need", 0.90)
        natural_history = m.get("natural_history_data", 0.85)
        advocacy = m.get("patient_advocacy", 0.75)
        score = 0.30 * prevalence + 0.30 * unmet_need + 0.25 * natural_history + 0.15 * advocacy
        criteria = {"prevalence": prevalence, "unmet_need": unmet_need,
                     "natural_history": natural_history, "advocacy": advocacy}
        return float(np.clip(score, 0, 1)), criteria

    def assess_improvement_vs_ucl(self, assessment: RegulatoryAssessment) -> Dict:
        improvements = {}
        for dim_name, ucl_val in self.ucl_baseline.items():
            dim_obj = getattr(assessment, dim_name, None)
            if dim_obj:
                improvements[dim_name] = {
                    "our_score": dim_obj.score,
                    "ucl_score": ucl_val,
                    "improvement": dim_obj.score / max(ucl_val, 0.01),
                }
        return improvements

    def generate_regulatory_summary(self, assessment: RegulatoryAssessment) -> str:
        lines = []
        lines.append("MHRA ILAP FastTrack Regulatory Assessment")
        lines.append("=" * 50)
        for dim_name in self.dimensions:
            dim_obj = getattr(assessment, dim_name, None)
            if dim_obj:
                marker = "PASS" if dim_obj.passed else "FAIL"
                lines.append(f"  {dim_obj.label:45s} {dim_obj.score:.3f}  [{marker}]")
        lines.append("-" * 50)
        lines.append(f"  Composite Score: {assessment.composite_score:.3f}")
        lines.append(f"  Weighted Score:  {assessment.weighted_score:.3f}")
        lines.append(f"  ILAP Eligible:   {'YES' if assessment.is_ilap_eligible else 'NO'}")
        if assessment.critical_gaps:
            lines.append("-" * 50)
            lines.append("CRITICAL GAPS:")
            for gap in assessment.critical_gaps:
                lines.append(f"  - {gap}")
        if assessment.improvements_needed:
            lines.append("REQUIRED IMPROVEMENTS:")
            for imp in assessment.improvements_needed:
                lines.append(f"  - {imp}")
        return "\n".join(lines)

    def utcl_regulatory_score(self) -> float:
        return sum(self.ucl_baseline[k] * self.dimensions[k]["weight"] for k in self.ucl_baseline)

    def our_regulatory_score(self, m: Dict[str, float]) -> float:
        assessment = self.evaluate_candidate(m)
        return assessment.weighted_score
