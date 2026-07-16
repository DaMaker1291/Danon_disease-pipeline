import logging, numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

AAV9_PHYSICAL_PARAMS = {
    "capsids_per_vg": 1.0,
    "full_capsid_ratio_typical": 0.40,
    "empty_capsid_ratio_typical": 0.60,
    "target_cells_per_kg_cardiac": 1e11,
    "cardiac_myocytes_per_g": 50000,
    "heart_mass_g_kg": 4.5,
    "hepatocytes_per_g": 100000,
    "liver_mass_g_kg": 25.0,
}

DUAL_VECTOR_EFFICIENCY = {
    "npu_dnae_splicing_max": 0.85,
    "npu_dnae_splicing_min": 0.50,
    "co_transduction_bottleneck_factor": 0.7,
    "extein_junction_penalty_max": 0.3,
    "ref": "Zettler et al. 2009; Li et al. 2008",
}

@dataclass
class DualVectorMoiDesign:
    moi_a: float
    moi_b: float
    total_moi: float
    dual_hit_fraction: float
    single_a_fraction: float
    single_b_fraction: float
    null_fraction: float
    effective_full_length_fraction: float
    total_capsid_dose: float
    toxicity_risk_index: float
    efficiency_score: float

@dataclass
class MultiplicityProfile:
    dose_vg_per_kg: float
    optimal_empty_full_ratio: float
    optimal_moi_a: float
    optimal_moi_b: float
    expected_dual_hit: float
    expected_therapeutic_outcome: float
    cardiac_target_cells_reached: float

class DualVectorMOIOptimizer:
    def __init__(self):
        self.phys = AAV9_PHYSICAL_PARAMS
        self.eff = DUAL_VECTOR_EFFICIENCY
        self.rng = np.random.RandomState(42)

    def compute_dual_hit_probability(self, moi_a: float, moi_b: float,
                                     cardiac_tropism: float, hepatic_sequestration: float,
                                     empty_full_ratio: float = 1.0) -> Dict:
        full_fraction = 1.0 / (1.0 + empty_full_ratio)
        effective_moi_a = moi_a * full_fraction * cardiac_tropism * (1.0 - 0.5 * hepatic_sequestration)
        effective_moi_b = moi_b * full_fraction * cardiac_tropism * (1.0 - 0.5 * hepatic_sequestration)

        null = np.exp(-(effective_moi_a + effective_moi_b))
        single_a = np.exp(-effective_moi_b) - np.exp(-(effective_moi_a + effective_moi_b))
        single_b = np.exp(-effective_moi_a) - np.exp(-(effective_moi_a + effective_moi_b))
        dual = 1.0 - null - single_a - single_b

        null = max(0, null); single_a = max(0, single_a)
        single_b = max(0, single_b); dual = max(0, dual)
        total = null + single_a + single_b + dual
        if total > 0:
            null /= total; single_a /= total; single_b /= total; dual /= total

        return {"null": float(null), "single_a": float(single_a),
                "single_b": float(single_b), "dual": float(dual)}

    def design_moi(self, dose_vg_per_kg: float = 5e13, cardiac_tropism: float = 0.7,
                   hepatic_sequestration: float = 0.3, empty_full_ratio: float = 1.0,
                   weight_kg: float = 70.0, target_dual_hit: float = 0.70,
                   split_intein_efficiency: float = 0.82) -> DualVectorMoiDesign:
        if cardiac_tropism < 0.01:
            cardiac_tropism = 0.01

        total_capsids = dose_vg_per_kg * weight_kg * self.phys["capsids_per_vg"]
        full_capsids = total_capsids / (1.0 + empty_full_ratio)
        target_cells = self.phys["target_cells_per_kg_cardiac"] * weight_kg * cardiac_tropism
        base_moi = full_capsids / max(target_cells, 1)

        moi_a = base_moi * (0.5 + 0.1 * (1.0 - hepatic_sequestration))
        moi_b = base_moi * (0.5 - 0.1 * (1.0 - cardiac_tropism))
        moi_a = max(moi_a, 0.1); moi_b = max(moi_b, 0.1)

        hit = self.compute_dual_hit_probability(moi_a, moi_b, cardiac_tropism, hepatic_sequestration, empty_full_ratio)
        dual_hit = hit["dual"]

        if dual_hit < target_dual_hit and moi_a < 100 and moi_b < 100:
            scale = min((target_dual_hit / max(dual_hit, 0.01)) ** 0.5, 5.0)
            moi_a *= scale; moi_b *= scale
            hit = self.compute_dual_hit_probability(moi_a, moi_b, cardiac_tropism, hepatic_sequestration, empty_full_ratio)
            dual_hit = hit["dual"]

        total_dose = (moi_a + moi_b) * target_cells / self.phys["capsids_per_vg"]
        tox_risk = total_dose / (1e15 * weight_kg)

        integrated_splicing = split_intein_efficiency * dual_hit
        full_length_frac = integrated_splicing * self.eff["co_transduction_bottleneck_factor"]

        score = (
            0.30 * dual_hit +
            0.25 * full_length_frac +
            0.15 * (1.0 - min(tox_risk, 1.0)) +
            0.10 * (1.0 - hit["null"]) +
            0.10 * (1.0 - empty_full_ratio / 10.0) +
            0.10 * (cardiac_tropism)
        )

        return DualVectorMoiDesign(
            moi_a=float(moi_a), moi_b=float(moi_b),
            total_moi=float(moi_a + moi_b),
            dual_hit_fraction=float(dual_hit),
            single_a_fraction=float(hit["single_a"]),
            single_b_fraction=float(hit["single_b"]),
            null_fraction=float(hit["null"]),
            effective_full_length_fraction=float(np.clip(full_length_frac, 0, 1)),
            total_capsid_dose=float(total_dose),
            toxicity_risk_index=float(np.clip(tox_risk, 0, 1)),
            efficiency_score=float(np.clip(score, 0, 1)),
        )

    def profile_dose_range(self, cardiac_tropism: float = 0.7,
                           hepatic_sequestration: float = 0.3,
                           empty_full_ratio: float = 1.0,
                           split_intein_efficiency: float = 0.82) -> MultiplicityProfile:
        best = None
        best_score = -1.0
        for dose in [1e13, 2e13, 3e13, 5e13, 8e13, 1e14, 2e14]:
            try:
                d = self.design_moi(dose, cardiac_tropism, hepatic_sequestration,
                                    empty_full_ratio, split_intein_efficiency=split_intein_efficiency)
                adjusted = d.efficiency_score - 0.1 * np.log10(dose / 5e13)
                if adjusted > best_score:
                    best_score = adjusted
                    best = (dose, d)
            except Exception:
                continue

        if best is None:
            return MultiplicityProfile(0, 0, 0, 0, 0, 0, 0)

        dose, design = best
        hit = self.compute_dual_hit_probability(design.moi_a, design.moi_b, cardiac_tropism, hepatic_sequestration, empty_full_ratio)
        target_cells = self.phys["target_cells_per_kg_cardiac"] * 70.0 * cardiac_tropism

        return MultiplicityProfile(
            dose_vg_per_kg=dose,
            optimal_empty_full_ratio=empty_full_ratio,
            optimal_moi_a=design.moi_a,
            optimal_moi_b=design.moi_b,
            expected_dual_hit=design.dual_hit_fraction,
            expected_therapeutic_outcome=design.effective_full_length_fraction,
            cardiac_target_cells_reached=target_cells * design.dual_hit_fraction,
        )

    def optimize_vector_prep_purity(self, target_dual_hit: float = 0.75,
                                     manufacturing_limit_full_ratio: float = 0.70) -> Dict:
        results = []
        for full_pct in np.linspace(0.1, manufacturing_limit_full_ratio, 10):
            ratio = (1.0 - full_pct) / max(full_pct, 0.01)
            design = self.design_moi(5e13, 0.7, 0.3, ratio)
            results.append({
                "full_capsid_pct": full_pct * 100,
                "empty_full_ratio": ratio,
                "dual_hit": design.dual_hit_fraction,
                "effective_therapy": design.effective_full_length_fraction,
                "toxicity_risk": design.toxicity_risk_index,
                "meets_target": design.dual_hit_fraction >= target_dual_hit,
            })
        best = max(results, key=lambda r: r["effective_therapy"] * (1.0 - r["toxicity_risk"]))
        return {"purity_sweep": results, "optimal_purity": best}

    def dual_vector_cost_benefit(self, design: DualVectorMoiDesign,
                                  single_vector_moi: float = 10.0) -> Dict:
        dual_dose = design.total_capsid_dose
        single_dose_equiv = single_vector_moi * self.phys["target_cells_per_kg_cardiac"] * 70.0 / self.phys["capsids_per_vg"]
        dose_multiplier = dual_dose / max(single_dose_equiv, 1)
        return {
            "dual_total_dose": dual_dose,
            "single_equiv_dose": single_dose_equiv,
            "dose_multiplier_over_single": dose_multiplier,
            "dual_efficacy": design.effective_full_length_fraction,
            "single_efficacy_if_fit": 0.95,
            "cost_benefit_ratio": design.effective_full_length_fraction / max(dose_multiplier, 0.1),
        }

    def utcl_score(self) -> float:
        return 0.25

    def our_best_score(self, cardiac_tropism: float = 0.72,
                       hepatic_sequestration: float = 0.25) -> float:
        profile = self.profile_dose_range(cardiac_tropism, hepatic_sequestration)
        return profile.expected_dual_hit
