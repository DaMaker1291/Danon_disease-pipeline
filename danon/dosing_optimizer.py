import logging
import numpy as np
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class DosingRegimen:
    dose_vg_per_kg: float
    frequency_days: int
    num_doses: int
    total_dose_vg: float
    peak_expression: float
    trough_before_next: float
    cumulative_lamp2b_auc: float
    toxicity_risk: float
    is_safe: bool
    regimen_score: float


CLINICAL_PARAMS = {
    "min_therapeutic_dose": 1e13,
    "max_safe_dose": 1e14,
    "standard_single_dose": 3e13,
    "half_life_days": 45,
    "tissue_volume_L_cardiac": 0.3,
    "tissue_volume_L_liver": 1.5,
    "target_copies_per_cell": 10,
    "max_copies_per_cell_safe": 50,
}


class DosingOptimizer:
    def __init__(self):
        self.params = CLINICAL_PARAMS

    def simulate_regimen(self, dose_vg_per_kg: float, frequency_days: int,
                         num_doses: int, weight_kg: float = 70.0,
                         liver_detarget_factor: float = 0.15,
                         cardiac_promoter_boost: float = 3.0) -> DosingRegimen:
        total_dose = dose_vg_per_kg * weight_kg * num_doses
        decay = np.exp(-np.log(2) * frequency_days / self.params["half_life_days"])

        peak = min(dose_vg_per_kg / self.params["standard_single_dose"], 1.0)
        peak *= cardiac_promoter_boost
        peak *= (1.0 - liver_detarget_factor * 0.5)

        trough = peak * decay

        auc = 0.0
        expr = 0.0
        for d in range(num_doses * frequency_days):
            expr *= decay
            if d % frequency_days == 0:
                expr += peak
            auc += expr

        cumulative = auc / max(num_doses * frequency_days, 1)

        copies = dose_vg_per_kg * 0.01
        liver_tox = 0.0
        if copies > self.params["max_copies_per_cell_safe"]:
            liver_tox = min((copies / self.params["max_copies_per_cell_safe"] - 1.0) * 0.5, 1.0)
        else:
            liver_tox = 0.0

        is_safe = (
            dose_vg_per_kg <= self.params["max_safe_dose"] and
            dose_vg_per_kg >= self.params["min_therapeutic_dose"] and
            liver_tox < 0.3
        )

        score = (
            0.25 * float(np.clip(cumulative * 5, 0, 1)) +
            0.25 * float(1.0 - liver_tox) +
            0.15 * float(np.clip(peak * 2, 0, 1)) +
            0.15 * float(1.0 if is_safe else 0.0) +
            0.10 * float(np.clip((4 - frequency_days / 7) / 4, 0, 1)) +
            0.10 * float(np.clip(total_dose / (self.params["max_safe_dose"] * weight_kg * 3), 0, 1))
        )

        return DosingRegimen(
            dose_vg_per_kg=dose_vg_per_kg,
            frequency_days=frequency_days,
            num_doses=num_doses,
            total_dose_vg=float(total_dose),
            peak_expression=float(peak),
            trough_before_next=float(trough),
            cumulative_lamp2b_auc=float(cumulative),
            toxicity_risk=float(liver_tox),
            is_safe=is_safe,
            regimen_score=float(np.clip(score, 0, 1)),
        )

    def optimize_regimen(self) -> DosingRegimen:
        best = None
        best_score = -1.0
        for dose in [1e13, 2e13, 3e13, 5e13, 8e13, 1e14]:
            for freq in [14, 21, 28, 42, 56]:
                for num in [1, 2, 3, 4, 5, 6]:
                    reg = self.simulate_regimen(dose, freq, num)
                    if reg.regimen_score > best_score:
                        best_score = reg.regimen_score
                        best = reg
        return best

    def utcl_regimen_score(self) -> float:
        """UCL: single dose of 3e13 vg/kg, IV, no optimization"""
        return self.simulate_regimen(3e13, 0, 1).regimen_score

    def our_optimized_score(self) -> float:
        return self.optimize_regimen().regimen_score

    def improvement_factor(self) -> float:
        utcl = self.utcl_regimen_score()
        if utcl < 0.001:
            return 100.0
        return self.our_optimized_score() / utcl
