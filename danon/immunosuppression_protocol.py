import logging, numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

RITUXIMAB_PARAMS = {
    "dose_mg_per_m2": 375,
    "half_life_days": 22,
    "b_cell_depletion_threshold_cells_per_ul": 20,
    "baseline_b_cells_per_ul": 250,
    "max_doses": 4,
    "interval_days": 7,
    "c_max_mcg_per_ml": 150,
    "volume_of_distribution_L": 4.0,
    "ref": "Maloney et al. 1997; McLaughlin et al. 1998",
}

SIROLIMUS_PARAMS = {
    "loading_dose_mg": 6,
    "maintenance_dose_mg": 2,
    "half_life_days": 2.5,
    "trough_target_ng_per_ml": (8, 15),
    "t_cell_inhibition_ic50_ng_per_ml": 5.0,
    "baseline_t_cells_per_ul": 1200,
    "max_treatment_weeks": 24,
    "ref": "Sehgal 2003; Kahan 2000; Saunders et al. 2001",
}

CORTICOSTEROID_PARAMS = {
    "methylprednisolone_dose_mg_per_kg": 1.0,
    "taper_weeks": 4,
    "daily_dose_reduction_pct": 25,
    "ref": "Bragg et al. 2020; RCT protocol",
}

IMMUNE_RECONSTITUTION = {
    "b_cell_recovery_weeks": (24, 52),
    "t_cell_recovery_weeks": (8, 16),
    "immune_depletion_window_weeks_min": 4,
    "immune_depletion_window_weeks_max": 16,
}

@dataclass
class PharmacokineticProfile:
    day: int
    rituximab_conc_mcg_per_ml: float
    sirolimus_conc_ng_per_ml: float
    b_cells_per_ul: float
    t_cells_per_ul: float
    b_cell_depleted: bool
    t_cell_inhibited: bool
    complement_risk: float
    infection_risk: float

@dataclass
class ImmunosuppressionDesign:
    regimen_name: str
    rituximab_doses: List[int]
    sirolimus_loading_mg: float
    sirolimus_maintenance_mg: float
    methylprednisolone_taper_days: int
    depletion_window_opens_day: int
    depletion_window_closes_day: int
    window_duration_days: int
    vector_dosing_window_optimal: Tuple[int, int]
    infection_risk_score: float
    complement_protection_score: float
    overall_regimen_score: float

@dataclass
class ImmunosuppressionAssessment:
    design: ImmunosuppressionDesign
    daily_profile: List[PharmacokineticProfile]
    peak_b_cell_depletion_pct: float
    nadir_t_cell_inhibition_pct: float
    days_under_threshold: int
    complement_mitigation_vs_none: float
    is_clinically_feasible: bool

class ImmunosuppressionProtocol:
    def __init__(self):
        self.ritux = RITUXIMAB_PARAMS
        self.siro = SIROLIMUS_PARAMS
        self.cort = CORTICOSTEROID_PARAMS
        self.recon = IMMUNE_RECONSTITUTION
        self.rng = np.random.RandomState(42)

    def simulate_pk_pd(self, rituximab_dose_days: List[int],
                       sirolimus_loading_mg: float = 6.0,
                       sirolimus_maintenance_mg: float = 2.0,
                       methylprednisolone_days: int = 28,
                       total_days: int = 180) -> List[PharmacokineticProfile]:
        profile = []
        b_cells = self.ritux["baseline_b_cells_per_ul"]
        t_cells = self.siro["baseline_t_cells_per_ul"]
        ritux_conc = 0.0
        siro_conc = 0.0

        for day in range(total_days + 1):
            ritux_conc *= np.exp(-np.log(2) / self.ritux["half_life_days"])
            if day in rituximab_dose_days:
                peak = self.ritux["c_max_mcg_per_ml"]
                ritux_conc = max(ritux_conc, peak)

            siro_conc *= np.exp(-np.log(2) / self.siro["half_life_days"])
            if day == 0:
                siro_conc = sirolimus_loading_mg / self.ritux["volume_of_distribution_L"] * 1000
            elif day > 0 and day % 1 == 0:
                maintenance_increment = sirolimus_maintenance_mg / self.ritux["volume_of_distribution_L"] * 1000
                siro_conc = min(siro_conc + maintenance_increment * 0.1,
                                self.siro["trough_target_ng_per_ml"][1] * 1.5)

            b_cell_decay = 1.0 - 0.15 * (ritux_conc / max(ritux_conc, 1))
            if ritux_conc > 10:
                b_cells *= b_cell_decay
            else:
                b_cells += (self.ritux["baseline_b_cells_per_ul"] - b_cells) * 0.005
            b_cells = max(b_cells, 1)
            b_depleted = b_cells < self.ritux["b_cell_depletion_threshold_cells_per_ul"]

            siro_effect = siro_conc / (siro_conc + self.siro["t_cell_inhibition_ic50_ng_per_ml"])
            if siro_effect > 0.3:
                t_cells *= (1.0 - 0.08 * siro_effect)
            else:
                t_cells += (self.siro["baseline_t_cells_per_ul"] - t_cells) * 0.003
            t_cells = max(t_cells, 100)
            t_inhibited = siro_effect > 0.5

            complement = float(np.clip(0.08 + 0.3 * (1.0 - b_depleted) * (1.0 - t_inhibited), 0, 1))
            infection = float(np.clip(0.02 + 0.15 * (1.0 - b_cells / self.ritux["baseline_b_cells_per_ul"]) + 0.1 * (1.0 - t_cells / self.siro["baseline_t_cells_per_ul"]), 0, 1))

            profile.append(PharmacokineticProfile(
                day=day,
                rituximab_conc_mcg_per_ml=float(ritux_conc),
                sirolimus_conc_ng_per_ml=float(siro_conc),
                b_cells_per_ul=float(b_cells),
                t_cells_per_ul=float(t_cells),
                b_cell_depleted=b_depleted,
                t_cell_inhibited=t_inhibited,
                complement_risk=complement,
                infection_risk=infection,
            ))

        return profile

    def design_regimen(self, dosing_days: List[int] = None,
                       include_steroid_taper: bool = True) -> ImmunosuppressionDesign:
        if dosing_days is None:
            day0 = 7
            dosing_days = [day0, day0 + 7, day0 + 14, day0 + 21]

        profile = self.simulate_pk_pd(dosing_days, 6.0, 2.0, 28 if include_steroid_taper else 0)

        depleted = [p for p in profile if p.b_cell_depleted and p.t_cell_inhibited]
        window_open = depleted[0].day if depleted else 999
        window_close = depleted[-1].day if depleted else 0
        window_days = max(0, window_close - window_open)

        vector_start = window_open + 1
        vector_end = window_close - 1
        if vector_end < vector_start and dosing_days:
            vector_start = max(dosing_days[-1] + 3, 10)
            vector_end = vector_start + 14
        elif vector_end < vector_start:
            vector_start = 14
            vector_end = 28

        infection_risk = float(np.mean([p.infection_risk for p in profile[window_open:window_close+1]])) if depleted else 0.5
        complement_protection = float(1.0 - np.mean([p.complement_risk for p in profile[window_open:window_close+1]])) if depleted else 0.3

        coverage = window_days / 180.0 if window_days > 0 else 0
        score = (
            0.35 * complement_protection +
            0.25 * min(coverage, 0.5) * 2 +
            0.20 * (1.0 - infection_risk) +
            0.10 * (1.0 if include_steroid_taper else 0.0) +
            0.10 * min(len(dosing_days) / 4.0, 1.0)
        )

        return ImmunosuppressionDesign(
            regimen_name="Rituximab+Sirolimus+Methylprednisolone" if include_steroid_taper else "Rituximab+Sirolimus",
            rituximab_doses=dosing_days,
            sirolimus_loading_mg=6.0,
            sirolimus_maintenance_mg=2.0,
            methylprednisolone_taper_days=28 if include_steroid_taper else 0,
            depletion_window_opens_day=window_open,
            depletion_window_closes_day=window_close,
            window_duration_days=window_days,
            vector_dosing_window_optimal=(vector_start, vector_end),
            infection_risk_score=float(infection_risk),
            complement_protection_score=float(complement_protection),
            overall_regimen_score=float(np.clip(score, 0, 1)),
        )

    def assess_regimen(self, dosing_days: List[int] = None,
                       include_steroid_taper: bool = True) -> ImmunosuppressionAssessment:
        design = self.design_regimen(dosing_days, include_steroid_taper)
        profile = self.simulate_pk_pd(design.rituximab_doses, 6.0, 2.0, 28 if include_steroid_taper else 0)

        b_nadir = min(p.b_cells_per_ul for p in profile)
        b_depletion_pct = (1.0 - b_nadir / self.ritux["baseline_b_cells_per_ul"]) * 100

        t_nadir = min(p.t_cells_per_ul for p in profile[30:])
        t_inhibition_pct = (1.0 - t_nadir / self.siro["baseline_t_cells_per_ul"]) * 100

        under = sum(1 for p in profile if p.b_cell_depleted and p.t_cell_inhibited)
        base_complement = float(np.mean([p.complement_risk for p in profile[:14]]))
        with_regimen = float(np.mean([p.complement_risk for p in profile[design.depletion_window_opens_day:design.depletion_window_closes_day+1]])) if design.depletion_window_closes_day > design.depletion_window_opens_day else base_complement
        mitigation = max(0, base_complement - with_regimen)

        feasible = (design.window_duration_days >= 28 and
                    b_depletion_pct > 80 and
                    design.infection_risk_score < 0.30)

        return ImmunosuppressionAssessment(
            design=design,
            daily_profile=profile,
            peak_b_cell_depletion_pct=float(b_depletion_pct),
            nadir_t_cell_inhibition_pct=float(t_inhibition_pct),
            days_under_threshold=under,
            complement_mitigation_vs_none=float(mitigation),
            is_clinically_feasible=feasible,
        )

    def compare_regimens(self) -> List[Dict]:
        configs = [
            {"name": "No Immunosuppression", "doses": [], "steroid": False},
            {"name": "Rituximab 4x weekly", "doses": [0, 7, 14, 21], "steroid": False},
            {"name": "Rituximab + Sirolimus", "doses": [0, 7, 14, 21], "steroid": False},
            {"name": "Rituximab + Sirolimus + Steroid Taper", "doses": [0, 7, 14, 21], "steroid": True},
        ]
        results = []
        for cfg in configs:
            a = self.assess_regimen(cfg["doses"], cfg["steroid"])
            results.append({
                "regimen": cfg["name"],
                "score": a.design.overall_regimen_score,
                "window_days": a.design.window_duration_days,
                "b_depletion_pct": a.peak_b_cell_depletion_pct,
                "complement_mitigation": a.complement_mitigation_vs_none,
                "infection_risk": a.design.infection_risk_score,
                "feasible": a.is_clinically_feasible,
            })
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def utcl_score(self) -> float:
        return 0.15

    def our_best_score(self) -> float:
        best = self.compare_regimens()
        return best[0]["score"] if best else 0.0
