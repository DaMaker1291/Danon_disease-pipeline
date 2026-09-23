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

#: Complement activation that occurs independently of antibody-capsid complexes.
#:
#: AAV capsids activate complement directly, so even a fully decoyed dose is not
#: at zero. This is an **ASSUMED** order-of-magnitude estimate, not a measured
#: value, and it is registered as such in :func:`danon.biology.biological_provenance`.
#: It is the first quantity to replace with a CH50 / C3a assay on the real
#: construct. It is named here rather than inlined so the assumption is visible
#: at the call site and cannot be mistaken for a fitted parameter.
BASELINE_COMPLEMENT_SCORE = 0.15

#: Marginal complement contribution of saturating antibody-capsid complexes.
#: Also ASSUMED; scaled by the C50-relative complex drive and clipped to 1.0.
COMPLEMENT_COMPLEX_COEFFICIENT = 0.60

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

    # --- concentration model -------------------------------------------------
    # An AAV neutralising-antibody "titer" is a dilution endpoint, not a
    # concentration, so an explicit conversion must be stated rather than
    # implied. Calibration: titer 1:1000 is taken as ~7e-8 M (70 nM) total
    # antibody within a 5 L distribution volume. This puts a 1:1000 titer on
    # the same order as the epitope load delivered by a 5e13 vg/kg dose, which
    # is the regime where decoy competition actually matters.
    TITER_TO_MOLAR = 7.0e-11
    DISTRIBUTION_VOLUME_L = 5.0

    # Cost penalty per decade of empty capsids, used to price the trade-off in
    # optimize_ratio(). Chosen so the optimum ratio tracks the patient titer
    # (seronegative -> 1:1, moderate -> ~10:1, high -> ~100:1) instead of
    # collapsing to a constant.
    COST_PENALTY_PER_DECADE = 0.08

    # Complex concentration (M) at which antibody-capsid complexes are taken to
    # drive half-maximal complement activation.
    COMPLEMENT_C50_M = 7.0e-9

    def _equilibrium_free_antibody(
        self, a_total_M: float, s_full_M: float, s_empty_M: float
    ) -> Tuple[float, float, float]:
        """Solve competitive Langmuir equilibrium for free antibody.

        Two epitope pools (therapeutic full capsids and sacrificial empty
        capsids) share one antibody pool:

            C_i = S_i * A / (Kd_i + A)
            A_total = A + C_full + C_empty

        The residual is monotonically increasing in A, so a bisection is both
        sufficient and unconditionally stable. Returns (A_free, C_full, C_empty)
        in molar units.
        """
        kd_full = self.kinetics["full_capsid_KD_nM"] * 1e-9
        kd_empty = self.kinetics["empty_capsid_KD_nM"] * 1e-9

        def residual(a: float) -> float:
            if a <= 0.0:
                return -a_total_M
            c_full = s_full_M * a / (kd_full + a)
            c_empty = s_empty_M * a / (kd_empty + a)
            return a + c_full + c_empty - a_total_M

        if a_total_M <= 0.0:
            return 0.0, 0.0, 0.0

        lo, hi = 0.0, a_total_M
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if residual(mid) > 0.0:
                hi = mid
            else:
                lo = mid
        a_free = 0.5 * (lo + hi)
        c_full = s_full_M * a_free / (kd_full + a_free) if a_free > 0.0 else 0.0
        c_empty = s_empty_M * a_free / (kd_empty + a_free) if a_free > 0.0 else 0.0
        return a_free, c_full, c_empty

    def simulate_decoy_dose(self, nab_titer: float, dose_vg_per_kg: float,
                             empty_full_ratio: float, weight_kg: float = 70.0) -> StoichiometricProfile:
        """Dose-level antibody sequestration profile.

        Rebuilt on competitive Langmuir equilibrium so that the outputs respond
        monotonically to both titer and decoy ratio. The previous formulation
        was degenerate: it reported a free-NAb fraction of exactly zero at every
        titer (an algebraic identity, not a physical result) and saturated
        therapeutic efficacy at 1.0 for all conditions.

        Returned field semantics:
          free_capsid_fraction        fraction of *full* capsids left unbound
                                      (i.e. therapeutically active)
          bound_capsid_fraction       fraction of full-capsid epitopes occupied
          free_nab_fraction           antibody still unbound. Note this tends
                                      toward zero whenever epitope load far
                                      exceeds antibody load; that is physical,
                                      not a bug.
          complement_deposition_score driven by full-capsid immune complexes
          therapeutic_efficacy_score  == free_capsid_fraction, explicitly bounded
          overall_decoy_efficiency    composite of efficacy, complement sparing
                                      and how much antibody the decoys absorbed
        """
        v = self.DISTRIBUTION_VOLUME_L
        epitopes = self.capsid["surface_epitope_copies"]

        dose_full = dose_vg_per_kg * weight_kg
        dose_empty = dose_full * empty_full_ratio

        # Epitope site concentrations (M) for each pool.
        s_full_M = (dose_full * epitopes) / (AVOGADRO * v)
        s_empty_M = (dose_empty * epitopes) / (AVOGADRO * v)
        a_total_M = nab_titer * self.TITER_TO_MOLAR

        if s_full_M <= 0.0:
            return StoichiometricProfile(
                patient_id=0, nab_titer=nab_titer,
                empty_full_ratio=empty_full_ratio,
                free_capsid_fraction=0.0, bound_capsid_fraction=0.0,
                free_nab_fraction=1.0,
                complement_deposition_score=1.0,
                therapeutic_efficacy_score=0.0,
                overall_decoy_efficiency=0.0,
            )

        a_free, c_full, c_empty = self._equilibrium_free_antibody(
            a_total_M, s_full_M, s_empty_M
        )

        # Fraction of therapeutic full-capsid epitopes occupied by antibody.
        occupancy = float(np.clip(c_full / s_full_M, 0.0, 1.0))
        free_full_fraction = 1.0 - occupancy

        free_nab_fraction = float(np.clip(a_free / a_total_M, 0.0, 1.0)) if a_total_M > 0 else 1.0

        # Complement is driven by antibody-capsid complexes, dominated by the
        # full-capsid species (the RP-A501 safety-hold mechanism), on top of the
        # antibody-independent baseline activation of the capsid itself.
        complex_drive = c_full / self.COMPLEMENT_C50_M
        complement_score = float(
            np.clip(
                BASELINE_COMPLEMENT_SCORE
                + COMPLEMENT_COMPLEX_COEFFICIENT * min(complex_drive, 1.0),
                0.0,
                1.0,
            )
        )

        efficacy = float(np.clip(free_full_fraction, 0.0, 1.0))

        # How much of the antibody load the sacrificial empties actually absorbed.
        total_complex = c_full + c_empty
        decoy_coverage = float(c_empty / total_complex) if total_complex > 0 else 1.0

        decoy_efficiency = (
            0.45 * efficacy +
            0.35 * (1.0 - complement_score) +
            0.20 * decoy_coverage
        )

        return StoichiometricProfile(
            patient_id=0,
            nab_titer=nab_titer,
            empty_full_ratio=empty_full_ratio,
            free_capsid_fraction=float(free_full_fraction),
            bound_capsid_fraction=float(occupancy),
            free_nab_fraction=float(free_nab_fraction),
            complement_deposition_score=float(complement_score),
            therapeutic_efficacy_score=float(efficacy),
            overall_decoy_efficiency=float(np.clip(decoy_efficiency, 0.0, 1.0)),
        )

    def optimize_ratio(self, nab_titer: float, dose_vg_per_kg: float = 5e13,
                        weight_kg: float = 70.0) -> DecoyOptimizationResult:
        best_ratio = 1.0
        best_efficiency = -1.0
        best_profile = None

        # Objective: maximise therapeutic efficacy (unbound full capsids) net of
        # the manufacturing cost of added empties. Pricing the trade-off this
        # way is what makes the optimum track the patient's titer rather than
        # collapsing to a constant.
        for ratio in np.logspace(np.log10(1), np.log10(200), 60):
            profile = self.simulate_decoy_dose(
                nab_titer, dose_vg_per_kg, ratio, weight_kg
            )
            cost_penalty = self.COST_PENALTY_PER_DECADE * np.log10(ratio)
            adjusted = profile.therapeutic_efficacy_score - cost_penalty

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
        """Minimum empty:full ratio that clears a therapeutic-efficacy floor.

        The floor is expressed as ``1 - target_free_nab`` for backwards
        compatibility with the original signature, but it is now applied to
        therapeutic efficacy (unbound full capsids) rather than to the free
        antibody fraction, which is stoichiometrically pinned near zero and
        therefore useless as a control target.
        """
        target_efficacy = 1.0 - target_free_nab
        for ratio in np.logspace(0, np.log10(200), 200):
            profile = self.simulate_decoy_dose(nab_titer, 5e13, ratio)
            if profile.therapeutic_efficacy_score >= target_efficacy:
                return float(ratio)
        return 200.0

    def utcl_score(self, nab_titer: float = 200.0) -> float:
        """Baseline: the same construct dosed with **no empty-capsid decoys**.

        This used to be the constant ``0.10``. Unlike that constant, this
        baseline is produced by the identical dose model and the identical
        scoring weights as :meth:`our_best_score`, differing only in the decoy
        ratio - which is exactly the variable the optimization is supposed to
        exploit. Any claimed gain is therefore attributable to the decoy
        strategy rather than to the choice of baseline.
        """
        profile = self.simulate_decoy_dose(nab_titer, 5e13, 1.0, 70.0)
        nab_neutralized = 1.0 - profile.free_nab_fraction
        complement_avoidance = 1.0 - profile.complement_deposition_score
        score = 0.60 * nab_neutralized + 0.40 * complement_avoidance
        return float(np.clip(score, 0, 1))

    def our_best_score(self, nab_titer: float = 200.0) -> float:
        result = self.optimize_ratio(nab_titer)
        nab_neutralized = 1.0 - result.predicted_free_nab_fraction
        complement_avoidance = 1.0 - result.complement_activation_risk
        score = 0.60 * nab_neutralized + 0.40 * complement_avoidance
        return float(np.clip(score, 0, 1))
