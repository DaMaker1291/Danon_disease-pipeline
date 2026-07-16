import logging, numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

MOUSE_PHYSIOLOGY = {
    "weight_g": 25.0,
    "blood_volume_ml": 1.5,
    "heart_weight_g": 0.15,
    "liver_weight_g": 1.2,
    "spleen_weight_g": 0.08,
    "brain_weight_g": 0.4,
    "gonad_weight_g": 0.05,
    "cardiac_myocytes_per_mg": 5000,
    "hepatocytes_per_mg": 10000,
    "injection_volume_ul_max": 200,
}

LAMP2_KO_MOUSE_HISTORY = {
    "strain": "B6;129S-Lamp2<tm1Ytf>/J",
    "phenotype": "Cardiac hypertrophy, LV dilation, reduced EF by 12 weeks",
    "mean_survival_weeks": 36,
    "lvmi_baseline_g_m2": 85.0,
    "ef_baseline_pct": 55.0,
    "annual_lvmi_increase_pct": 12.0,
    "annual_ef_decline_pct": 8.0,
    "ref": "Stypmann et al. 2006, Cardiovasc Res; Tanaka et al. 2020",
}

AAV9_MOUSE_BIODISTRIBUTION = {
    "cardiac_transduction_efficiency": 0.45,
    "liver_transduction_efficiency": 0.85,
    "spleen_transduction_efficiency": 0.30,
    "brain_transduction_efficiency": 0.02,
    "gonad_transduction_efficiency": 0.01,
    "vg_per_cell_per_dose": 0.5,
    "dose_response_exponent": 0.7,
}

TISSUE_WEIGHT_MULTIPLIERS = {
    "heart": 1.0, "liver": -0.5, "spleen": -0.3, "brain": -0.1, "gonads": -1.0,
}

@dataclass
class MouseBiodistribution:
    heart_vg_per_cell: float
    liver_vg_per_cell: float
    spleen_vg_per_cell: float
    brain_vg_per_cell: float
    gonad_vg_per_cell: float
    heart_liver_ratio: float
    cardiac_selectivity_index: float
    total_vector_genomes: float

@dataclass
class MouseCardiacOutcome:
    lvmi_at_12w: float
    ef_at_12w: float
    lvmi_reduction_pct: float
    ef_improvement_pct: float
    heart_failure_free: bool

@dataclass
class MouseStudyResult:
    candidate_id: int
    dose_vg: float
    biodistribution: MouseBiodistribution
    cardiac_6m: MouseCardiacOutcome
    survival_6m: float
    survival_12m: float
    treatment_effect_pvalue: float
    study_cost_usd: float
    go_decision: str

class MouseStudySimulator:
    def __init__(self):
        self.phys = MOUSE_PHYSIOLOGY
        self.history = LAMP2_KO_MOUSE_HISTORY
        self.biodist = AAV9_MOUSE_BIODISTRIBUTION
        self.rng = np.random.RandomState(42)

    def simulate_biodistribution(self, cardiac_tropism: float, hepatic_avoidance: float,
                                 immune_evasion: float, dose_vg: float = 5e11,
                                 capsid_redesign_factor: float = 1.0) -> MouseBiodistribution:
        dose_factor = (dose_vg / 5e11) ** self.biodist["dose_response_exponent"]
        base_cardiac = self.biodist["cardiac_transduction_efficiency"] * dose_factor
        base_liver = self.biodist["liver_transduction_efficiency"] * dose_factor

        cardiac_eff = base_cardiac * (0.3 + 0.7 * cardiac_tropism) * capsid_redesign_factor
        liver_eff = base_liver * (1.0 - 0.8 * hepatic_avoidance) * max(0.3, 1.0 - capsid_redesign_factor * 0.3)

        noise = lambda: 1.0 + 0.1 * self.rng.randn()
        heart_vg = cardiac_eff * self.biodist["vg_per_cell_per_dose"] * noise()
        liver_vg = liver_eff * self.biodist["vg_per_cell_per_dose"] * noise()
        spleen_vg = self.biodist["spleen_transduction_efficiency"] * dose_factor * 0.5 * noise()
        brain_vg = self.biodist["brain_transduction_efficiency"] * dose_factor * 0.01 * noise()
        gonad_vg = self.biodist["gonad_transduction_efficiency"] * dose_factor * 0.1 * noise()

        heart_liver_ratio = heart_vg / max(liver_vg, 0.001)
        cardiac_sel = heart_liver_ratio / max(heart_liver_ratio + 0.5, 0.01)

        total_vg = (heart_vg * self.phys["heart_weight_g"] * self.phys["cardiac_myocytes_per_mg"] * 1000 +
                    liver_vg * self.phys["liver_weight_g"] * self.phys["hepatocytes_per_mg"] * 1000)

        return MouseBiodistribution(
            heart_vg_per_cell=float(heart_vg), liver_vg_per_cell=float(liver_vg),
            spleen_vg_per_cell=float(spleen_vg), brain_vg_per_cell=float(brain_vg),
            gonad_vg_per_cell=float(gonad_vg),
            heart_liver_ratio=float(heart_liver_ratio),
            cardiac_selectivity_index=float(cardiac_sel),
            total_vector_genomes=float(total_vg),
        )

    def simulate_cardiac_outcome(self, biodist: MouseBiodistribution,
                                 lamp2b_expression: float, dosing_score: float,
                                 promoter_score: float, weeks: int = 24) -> MouseCardiacOutcome:
        lvmi_baseline = self.history["lvmi_baseline_g_m2"]
        ef_baseline = self.history["ef_baseline_pct"]

        lvmi_decline_rate = self.history["annual_lvmi_increase_pct"] / 52.0
        ef_decline_rate = self.history["annual_ef_decline_pct"] / 52.0

        therapeutic_factor = (biodist.heart_vg_per_cell / max(biodist.liver_vg_per_cell, 0.01))
        therapeutic_factor = np.clip(therapeutic_factor, 0, 3)
        expression_efficacy = lamp2b_expression * therapeutic_factor * (0.5 + 0.5 * dosing_score)

        lvmi_change = lvmi_decline_rate * weeks * (1.0 - 1.5 * expression_efficacy * promoter_score)
        lvmi_final = lvmi_baseline * (1.0 + lvmi_change / 100.0)

        ef_change = ef_decline_rate * weeks * (1.0 - 2.0 * expression_efficacy * promoter_score)
        ef_final = ef_baseline * (1.0 - ef_change / 100.0)

        lvmi_reduction = max(0, lvmi_baseline - lvmi_final)
        ef_improvement = max(0, ef_final - ef_baseline)

        hf_free = ef_final > 40.0 and lvmi_final < 120.0

        return MouseCardiacOutcome(
            lvmi_at_12w=float(lvmi_final), ef_at_12w=float(ef_final),
            lvmi_reduction_pct=float(lvmi_reduction / lvmi_baseline * 100),
            ef_improvement_pct=float(ef_improvement / ef_baseline * 100),
            heart_failure_free=hf_free,
        )

    def run_study(self, candidate_id: int, cardiac_tropism: float, hepatic_avoidance: float,
                  immune_evasion: float, lamp2b_expression: float, promoter_score: float,
                  dosing_score: float, dose_vg: float = 5e11,
                  capsid_redesign_factor: float = 1.0,
                  n_mice_per_group: int = 12, weeks: int = 24) -> MouseStudyResult:
        biodist = self.simulate_biodistribution(
            cardiac_tropism, hepatic_avoidance, immune_evasion, dose_vg, capsid_redesign_factor
        )
        cardiac = self.simulate_cardiac_outcome(biodist, lamp2b_expression, dosing_score, promoter_score, weeks)

        untreated_survival_6m = np.exp(-6.0 / (self.history["mean_survival_weeks"] / 4.33))
        hazard_reduction = 0.3 + 0.5 * cardiac.ef_improvement_pct / 20.0
        treated_survival_6m = 1.0 - (1.0 - untreated_survival_6m) * (1.0 - hazard_reduction)

        untreated_survival_12m = np.exp(-12.0 / (self.history["mean_survival_weeks"] / 4.33))
        treated_survival_12m = 1.0 - (1.0 - untreated_survival_12m) * (1.0 - hazard_reduction * 0.7)

        effect_size = (treated_survival_6m - untreated_survival_6m)
        t_stat = effect_size * np.sqrt(n_mice_per_group) / 0.15
        p_value = float(2.0 * (1.0 - 0.5 * (1.0 + np.tanh(t_stat / np.sqrt(2 * n_mice_per_group)))))

        study_cost = 12000 + n_mice_per_group * 850 + weeks * 2500 + 8000 * (1 if biodist.gonad_vg_per_cell > 0.001 else 0)

        exit_gates = [
            biodist.heart_liver_ratio >= 0.3,
            cardiac.heart_failure_free,
            treated_survival_6m > 0.80,
            biodist.gonad_vg_per_cell < 0.1,
            cardiac.lvmi_reduction_pct > 0,
        ]
        n_passed = sum(exit_gates)
        if n_passed >= 4:
            go = "ADVANCE_TO_NHP"
        elif n_passed >= 2:
            go = "CONDITIONAL (iterate capsid)"
        else:
            go = "HALT"

        return MouseStudyResult(
            candidate_id=candidate_id,
            dose_vg=dose_vg,
            biodistribution=biodist,
            cardiac_6m=cardiac,
            survival_6m=float(treated_survival_6m),
            survival_12m=float(treated_survival_12m),
            treatment_effect_pvalue=p_value,
            study_cost_usd=study_cost,
            go_decision=go,
        )

    def panel_of_candidates(self, candidates: List[Dict]) -> List[Dict]:
        results = []
        for c in candidates:
            r = self.run_study(
                c["candidate_id"], c.get("cardiac_tropism", 0.5),
                c.get("hepatic_avoidance", 0.5), c.get("immune_evasion", 0.5),
                c.get("lamp2b_expression", 0.5), c.get("promoter_score", 0.5),
                c.get("dosing_score", 0.5),
            )
            results.append({
                "candidate_id": c["candidate_id"],
                "heart_liver_ratio": r.biodistribution.heart_liver_ratio,
                "cardiac_selectivity": r.biodistribution.cardiac_selectivity_index,
                "lvmi_reduction_pct": r.cardiac_6m.lvmi_reduction_pct,
                "ef_improvement_pct": r.cardiac_6m.ef_improvement_pct,
                "survival_6m": r.survival_6m,
                "p_value": r.treatment_effect_pvalue,
                "go": r.go_decision,
                "cost": r.study_cost_usd,
            })
        return results

    def utcl_score(self) -> float:
        return 0.25

    def our_best_score(self, candidates: List[Dict]) -> float:
        if not candidates:
            return 0.0
        results = self.panel_of_candidates(candidates)
        return float(np.mean([r["cardiac_selectivity"] for r in results]))
