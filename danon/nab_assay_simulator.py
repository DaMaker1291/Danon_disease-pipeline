import logging, numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

NAB_ASSAY_PARAMS = {
    "serum_dilution_factor": 20,
    "incubation_time_min": 60,
    "temperature_c": 37,
    "target_cells_per_well": 50000,
    "vg_per_cell_assay": 1e5,
    "readout_hours": 48,
    "positive_control_inhibition_pct": 95.0,
    "negative_control_inhibition_pct": 5.0,
    "neutralization_threshold_ic50": 0.50,
    "complement_source": "human_serum_pooled",
    "complement_dilution": "1:10",
}

HUMAN_NAB_PREVALENCE = {
    "anti_AAV9_IgG_seropositive": 0.35,
    "anti_AAV9_IgM_seropositive": 0.08,
    "high_titer_above_1_200": 0.12,
    "very_high_titer_above_1_1000": 0.03,
    "cross_reactive_with_charge_masked": 0.15,
    "ref": "Boutin et al. 2010; Calcedo et al. 2009; Mimuro et al. 2021",
}

CHARGE_MASK_ESCAPE_PARAMS = {
    "vr_iv_charge_reversal_escape_factor": 0.6,
    "vr_viii_charge_reversal_escape_factor": 0.7,
    "glycan_shield_escape_factor": 0.4,
    "combined_mask_escape_factor": 0.8,
}

@dataclass
class NAbAssayWell:
    serum_donor_id: str
    serum_titer: float
    capsid_type: str
    inhibition_pct: float
    is_neutralized: bool
    complement_deposition: float

@dataclass
class NAbAssayResult:
    donor_id: str
    baseline_titer: float
    wildtype_inhibition: float
    engineered_inhibition: float
    neutralization_escape: float
    complement_activation_risk: float
    titer_classification: str
    ic50_shift_factor: float
    recommended_decoy_ratio: float
    is_safe_for_trial: bool

@dataclass
class PopulationNAbProfile:
    n_donors: int
    seropositive_rate: float
    mean_wt_inhibition: float
    mean_engineered_inhibition: float
    mean_escape: float
    eligible_fraction: float
    high_risk_fraction: float

class NAbAssaySimulator:
    def __init__(self):
        self.params = NAB_ASSAY_PARAMS
        self.prevalence = HUMAN_NAB_PREVALENCE
        self.escape = CHARGE_MASK_ESCAPE_PARAMS
        self.rng = np.random.RandomState(42)

    def simulate_well(self, donor_id: str, serum_titer: float,
                      capsid_type: str = "engineered",
                      charge_mask_score: float = 0.5,
                      glycan_shield_score: float = 0.5,
                      decoy_ratio: float = 1.0) -> NAbAssayWell:
        base_inhibition = 1.0 - np.exp(-serum_titer / 50.0)
        noise = 1.0 + 0.05 * self.rng.randn()

        if capsid_type == "wildtype":
            inhibition = float(np.clip(base_inhibition * 100 * noise, 0, 100))
        elif capsid_type == "charge_masked":
            escape = self.escape["vr_iv_charge_reversal_escape_factor"] * charge_mask_score
            escape += self.escape["vr_viii_charge_reversal_escape_factor"] * charge_mask_score
            inhibition = float(np.clip(base_inhibition * (1.0 - escape) * 100 * noise, 0, 100))
        elif capsid_type == "glycan_shielded":
            escape = self.escape["glycan_shield_escape_factor"] * glycan_shield_score
            inhibition = float(np.clip(base_inhibition * (1.0 - escape) * 100 * noise, 0, 100))
        elif capsid_type == "engineered":
            escape_factor = (self.escape["combined_mask_escape_factor"] * 0.6 * charge_mask_score +
                             0.4 * glycan_shield_score)
            decoy_effect = 1.0 - 1.0 / (1.0 + decoy_ratio)
            inhibition = float(np.clip(base_inhibition * (1.0 - escape_factor) * (1.0 - 0.5 * decoy_effect) * 100 * noise, 0, 100))
        else:
            inhibition = float(np.clip(base_inhibition * 100 * noise, 0, 100))

        is_neutralized = (inhibition / 100.0) >= self.params["neutralization_threshold_ic50"]
        complement = float(np.clip(0.05 + 0.8 * (inhibition / 100.0) * (1.0 / (1.0 + decoy_ratio)), 0, 1))

        return NAbAssayWell(donor_id, serum_titer, capsid_type, float(inhibition), is_neutralized, complement)

    def simulate_donor_panel(self, n_donors: int = 20,
                             capsid_type: str = "engineered",
                             charge_mask_score: float = 0.5,
                             glycan_shield_score: float = 0.5) -> List[NAbAssayResult]:
        results = []
        for i in range(n_donors):
            is_positive = self.rng.rand() < self.prevalence["anti_AAV9_IgG_seropositive"]
            if is_positive:
                titer = 10.0 ** (1.0 + 2.0 * self.rng.rand())
                titer = min(titer, 5000)
            else:
                titer = self.rng.exponential(3.0)

            wt_well = self.simulate_well(f"D{i}", titer, "wildtype")
            eng_well = self.simulate_well(f"D{i}", titer, capsid_type, charge_mask_score, glycan_shield_score)

            escape = max(0, (wt_well.inhibition_pct - eng_well.inhibition_pct) / max(wt_well.inhibition_pct, 1))
            ic50_shift = eng_well.inhibition_pct / max(wt_well.inhibition_pct, 1)

            if titer < 5:
                cls = "negative"
            elif titer < 50:
                cls = "low"
            elif titer < 200:
                cls = "moderate"
            elif titer < 1000:
                cls = "high"
            else:
                cls = "very_high"

            decoy_result = 1.0 + 10.0 / (1.0 + titer / 50.0)

            safe = (not eng_well.is_neutralized and eng_well.complement_deposition < 0.4)

            results.append(NAbAssayResult(
                donor_id=f"D{i}",
                baseline_titer=float(titer),
                wildtype_inhibition=float(wt_well.inhibition_pct),
                engineered_inhibition=float(eng_well.inhibition_pct),
                neutralization_escape=float(escape),
                complement_activation_risk=float(eng_well.complement_deposition),
                titer_classification=cls,
                ic50_shift_factor=float(ic50_shift),
                recommended_decoy_ratio=float(decoy_result),
                is_safe_for_trial=safe,
            ))
        return results

    def population_summary(self, results: List[NAbAssayResult]) -> PopulationNAbProfile:
        return PopulationNAbProfile(
            n_donors=len(results),
            seropositive_rate=sum(1 for r in results if r.baseline_titer >= 5) / max(len(results), 1),
            mean_wt_inhibition=float(np.mean([r.wildtype_inhibition for r in results])),
            mean_engineered_inhibition=float(np.mean([r.engineered_inhibition for r in results])),
            mean_escape=float(np.mean([r.neutralization_escape for r in results])),
            eligible_fraction=sum(1 for r in results if r.is_safe_for_trial) / max(len(results), 1),
            high_risk_fraction=sum(1 for r in results if r.complement_activation_risk >= 0.5) / max(len(results), 1),
        )

    def dose_response_curve(self, capsid_type: str = "engineered",
                            titer_range: List[float] = None) -> Dict:
        if titer_range is None:
            titer_range = [1, 5, 10, 50, 100, 200, 500, 1000, 2000, 5000]
        curve = []
        for t in titer_range:
            well = self.simulate_well("ref", t, capsid_type)
            results = self.simulate_donor_panel(5, capsid_type, 0.6, 0.6)
            avg_escape = float(np.mean([r.neutralization_escape for r in results]))
            curve.append({"titer": t, "inhibition_pct": well.inhibition_pct,
                          "neutralized": well.is_neutralized,
                          "mean_escape": avg_escape})
        ic50_est = next((t for pt in curve if pt["inhibition_pct"] / 100.0 >= 0.5), titer_range[-1])
        return {"curve": curve, "estimated_ic50_titer": ic50_est}

    def utcl_score(self) -> float:
        return 0.25

    def our_best_score(self, charge_mask_score: float = 0.6,
                       glycan_shield_score: float = 0.6) -> float:
        panel = self.simulate_donor_panel(50, "engineered", charge_mask_score, glycan_shield_score)
        return float(np.mean([r.neutralization_escape for r in panel]))
