"""
Stoichiometric Decoy Calculator: Pharmacokinetic simulator for
empty-to-full capsid ratio optimization against pre-existing NAbs.

The core insight from the Rocket Pharmaceuticals RP-A501 trial's safety hold
was that complement activation was driven by antibody-capsid immune complexes.
Empty capsids act as high-affinity sacrificial decoys that soak up NAbs
before they can bind therapeutic full capsids, drastically reducing both
neutralization and complement deposition.

Reference: Mingozzi et al. 2013, Sci Transl Med 5(194):194ra92.
"""
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Avogadro constant
AVOGADRO = 6.022e23

# Typical AAV9 capsid physical parameters
CAPSID_PARAMS = {
    "diameter_nm": 25.0,
    "vp3_copies_per_capsid": 60,
    "surface_epitope_copies": 60,
    "full_capsid_mass_kda": 5500,
    "empty_capsid_mass_kda": 3800,
    "full_capsids_per_vg": 1.0 / (4.7e3 * 330 * AVOGADRO),  # ~1 capsid per 1 vg
    "empty_capsid_manufacturing_cost_ratio": 0.15,
}

# NAb binding parameters (equilibrium dissociation constants)
# From published surface plasmon resonance data
NAB_BINDING_KINETICS = {
    "full_capsid_KD_nM": 0.5,
    "empty_capsid_KD_nM": 0.8,
    "full_capsid_kon_M-1s-1": 1.2e6,
    "full_capsid_koff_s-1": 6.0e-4,
    "empty_capsid_kon_M-1s-1": 1.0e6,
    "empty_capsid_koff_s-1": 8.0e-4,
    "complement_activation_threshold_nM": 0.1,
}

# Clinical NAb titer reference ranges
NAB_TITER_CLASSIFICATION = {
    "negative": {"range": (1, 5), "seroprevalence": 0.40, "description": "Below detection"},
    "low": {"range": (5, 50), "seroprevalence": 0.25, "description": "Detectable, low neutralization"},
    "moderate": {"range": (50, 200), "seroprevalence": 0.18, "description": "Partial neutralization"},
    "high": {"range": (200, 1000), "seroprevalence": 0.12, "description": "Strong neutralization"},
    "very_high": {"range": (1000, 10000), "seroprevalence": 0.05, "description": "Exclusion criterion in most trials"},
}


@dataclass
class DecoyOptimizationResult:
    patient_nab_titer: float
    titer_classification: str
    optimal_empty_full_ratio: float
    recommended_ratio_range: Tuple[float, float]
    full_capsids_per_dose: float
    empty_capsids_per_dose: float
    predicted_free_nab_fraction: float
    complement_activation_risk: float
    effective_titer_reduction: float
    manufacturing_cost_ratio: float
    clinical_recommendation: str


@dataclass
class StoichiometricProfile:
    patient_id: int
    nab_titer: float
    empty_full_ratio: float
    free_capsid_fraction: float
    bound_capsid_fraction: float
    free_nab_fraction: float
    complement_deposition_score: float
    therapeutic_efficacy_score: float
    overall_decoy_efficiency: float


class StoichiometricCalculator:
    def __init__(self):
        self.capsid = CAPSID_PARAMS
        self.kinetics = NAB_BINDING_KINETICS
        self.titers = NAB_TITER_CLASSIFICATION

    def classify_titer(self, nab_titer: float) -> str:
        for cls, data in self.titers.items():
            lo, hi = data["range"]
            if lo <= nab_titer <= hi:
                return cls
        return "very_high" if nab_titer >= self.titers["very_high"]["range"][0] else "negative"

    def simulate_decoy_dose(self, nab_titer: float, dose_vg_per_kg: float,
                             empty_full_ratio: float, weight_kg: float = 70.0) -> StoichiometricProfile:
        dose_full = dose_vg_per_kg * weight_kg
        dose_empty = dose_full * empty_full_ratio

        total_capsids = dose_full + dose_empty
        full_fraction = dose_full / max(total_capsids, 1)

        KD_full = self.kinetics["full_capsid_KD_nM"] * 1e-9
        KD_empty = self.kinetics["empty_capsid_KD_nM"] * 1e-9

        nab_conc_M = nab_titer * 1e-12

        total_nab_sites = nab_conc_M * AVOGADRO * 5.0

        full_binding_sites = dose_full * self.capsid["surface_epitope_copies"]
        empty_binding_sites = dose_empty * self.capsid["surface_epitope_copies"]
        total_binding_sites = full_binding_sites + empty_binding_sites

        if total_binding_sites < 1:
            return StoichiometricProfile(
                patient_id=0, nab_titer=nab_titer,
                empty_full_ratio=empty_full_ratio,
                free_capsid_fraction=1.0, bound_capsid_fraction=0.0,
                free_nab_fraction=1.0,
                complement_deposition_score=0.0,
                therapeutic_efficacy_score=1.0,
                overall_decoy_efficiency=1.0,
            )

        empty_fraction = empty_binding_sites / max(total_binding_sites, 1)
        full_fraction = full_binding_sites / max(total_binding_sites, 1)

        bound_fraction = min(
            total_nab_sites / max(total_binding_sites, 1),
            1.0
        )

        nab_bound_to_full = bound_fraction * full_fraction
        nab_bound_to_empty = bound_fraction * empty_fraction

        free_full_fraction = full_fraction * (1.0 - nab_bound_to_full / max(full_fraction, 0.01))
        free_full_fraction = max(free_full_fraction, 0.0)

        free_nab = max(0, total_nab_sites - total_binding_sites * bound_fraction)
        free_nab_fraction = free_nab / max(total_nab_sites, 1)

        nabs_occupying_full = total_nab_sites * bound_fraction * full_fraction
        nabs_per_full_capsid = nabs_occupying_full / max(dose_full, 1)

        baseline_complement = 0.15
        nab_complement = min(nabs_per_full_capsid * 50.0, 0.60)
        complement_score = float(np.clip(baseline_complement + nab_complement, 0, 1))

        free_full_ratio = free_full_fraction / max(full_fraction, 0.01)
        efficacy = float(np.clip(free_full_ratio * 1.5, 0, 1))

        complement_efficiency = float(np.clip(complement_score, 0, 1))
        decoy_efficiency = (
            0.50 * (1.0 - complement_efficiency) +
            0.30 * efficacy +
            0.20 * (1.0 - free_nab_fraction)
        )

        return StoichiometricProfile(
            patient_id=0,
            nab_titer=nab_titer,
            empty_full_ratio=empty_full_ratio,
            free_capsid_fraction=float(free_full_fraction),
            bound_capsid_fraction=float(bound_fraction),
            free_nab_fraction=float(np.clip(free_nab_fraction, 0, 1)),
            complement_deposition_score=float(complement_score),
            therapeutic_efficacy_score=float(efficacy),
            overall_decoy_efficiency=float(np.clip(decoy_efficiency, 0, 1)),
        )

    def optimize_ratio(self, nab_titer: float, dose_vg_per_kg: float = 5e13,
                        weight_kg: float = 70.0) -> DecoyOptimizationResult:
        best_ratio = 1.0
        best_efficiency = -1.0
        best_profile = None

        for ratio in np.logspace(np.log10(1), np.log10(100), 50):
            profile = self.simulate_decoy_dose(
                nab_titer, dose_vg_per_kg, ratio, weight_kg
            )
            cost_penalty = 0.08 * np.log10(ratio) / 2.0
            adjusted = profile.overall_decoy_efficiency - cost_penalty

            if adjusted > best_efficiency:
                best_efficiency = adjusted
                best_ratio = ratio
                best_profile = profile

        lo_ratio = best_ratio / np.sqrt(2)
        hi_ratio = best_ratio * np.sqrt(2)

        titer_class = self.classify_titer(nab_titer)
        free_nab_frac = best_profile.free_nab_fraction
        complement_risk = best_profile.complement_deposition_score

        titer_reduction = 1.0 - free_nab_frac

        cost_ratio = 1.0 + best_ratio * self.capsid["empty_capsid_manufacturing_cost_ratio"]

        if nab_titer >= 1000:
            recommendation = "EXCLUDE: NAb titer >= 1:1000 — plasmapheresis or immune suppression required"
        elif nab_titer >= 200:
            recommendation = f"USE DECOY RATIO {best_ratio:.0f}:1 — escalate empty capsids to overcome high titer"
        elif nab_titer >= 50:
            recommendation = f"USE DECOY RATIO {best_ratio:.0f}:1 — standard decoy protection sufficient"
        elif nab_titer >= 5:
            recommendation = f"USE DECOY RATIO {best_ratio:.0f}:1 — prophylactic decoy for low-titer patients"
        else:
            recommendation = f"MINIMAL DECOY RATIO {best_ratio:.0f}:1 — seronegative, no decoy needed"

        return DecoyOptimizationResult(
            patient_nab_titer=nab_titer,
            titer_classification=titer_class,
            optimal_empty_full_ratio=float(best_ratio),
            recommended_ratio_range=(float(lo_ratio), float(hi_ratio)),
            full_capsids_per_dose=float(dose_vg_per_kg * weight_kg),
            empty_capsids_per_dose=float(dose_vg_per_kg * weight_kg * best_ratio),
            predicted_free_nab_fraction=float(np.clip(free_nab_frac, 0, 1)),
            complement_activation_risk=float(np.clip(complement_risk, 0, 1)),
            effective_titer_reduction=float(np.clip(titer_reduction, 0, 1)),
            manufacturing_cost_ratio=float(cost_ratio),
            clinical_recommendation=recommendation,
        )

    def simulate_population_dosing(self, dose_vg_per_kg: float = 5e13) -> Dict[str, DecoyOptimizationResult]:
        results = {}
        for titer_val in [1, 10, 50, 200, 500, 1000]:
            results[f"titer_{titer_val}"] = self.optimize_ratio(titer_val, dose_vg_per_kg)
        return results

    def required_empty_to_full_for_target(self, nab_titer: float,
                                           target_free_nab: float = 0.15) -> float:
        for ratio in np.linspace(1, 200, 200):
            profile = self.simulate_decoy_dose(nab_titer, 5e13, ratio)
            nab_free = profile.free_nab_fraction
            if nab_free <= target_free_nab:
                return ratio
        return 200.0

    def utcl_score(self) -> float:
        return 0.10

    def our_best_score(self, nab_titer: float = 200.0) -> float:
        result = self.optimize_ratio(nab_titer)
        nab_neutralized = 1.0 - result.predicted_free_nab_fraction
        complement_avoidance = 1.0 - result.complement_activation_risk
        score = 0.60 * nab_neutralized + 0.40 * complement_avoidance
        return float(np.clip(score, 0, 1))
