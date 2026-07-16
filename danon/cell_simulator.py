import logging, numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# Literature-calibrated parameters for Danon disease cardiomyocyte pathophysiology
# Sources: Boucek et al. 2011, Cenacchi et al. 2020, Tanaka et al. 2020,
#          Endo et al. 2001, Nishino et al. 2000, Rowland et al. 2016
CELL_PARAMS = {
    "healthy_lamp2_copies_per_cell": 1e5,
    "danon_null_lamp2_fraction": 0.02,
    "lamp2_turnover_half_life_hours": 48,
    "lysosomal_ph_healthy": 4.8,
    "lysosomal_ph_danon": 6.2,
    "hydrolase_activity_healthy": 1.0,
    "hydrolase_activity_danon": 0.15,
    "baseline_autophagosome_count": 100,
    "autophagosome_count_danon": 450,
    "autophagic_flux_healthy": 1.0,
    "autophagic_flux_danon": 0.12,
    "glycogen_content_healthy_mg_per_g": 5.0,
    "glycogen_content_danon_mg_per_g": 25.0,
    "ros_level_healthy": 1.0,
    "ros_level_danon": 3.5,
    "hypertrophy_threshold_glycogen_mg_per_g": 12.0,
    "apoptosis_threshold_ros": 5.0,
    "cell_death_accumulation_rate": 0.01,
}

INTEGRATION_PARAMS = {
    "simulation_hours": 8760,
    "time_points": 876,
    "rtol": 1e-6,
    "atol": 1e-8,
    "max_dt_hours": 24,
}

THERAPY_PARAMS = {
    "vector_copy_number_per_cell_therapeutic": 10,
    "lamp2_expression_per_vector_copy": 0.08,
    "expression_delay_hours": 48,
    "promoter_efficiency_scale": 0.7,
    "mirna_detarget_liver_penalty": 0.05,
    "intein_splicing_efficiency": 0.82,
    "dual_vector_cotransduction_penalty": 0.7,
}

@dataclass
class CellState:
    hours: float
    lamp2_fraction: float
    lysosomal_ph: float
    hydrolase_activity: float
    autophagosome_count_norm: float
    autophagic_flux: float
    glycogen_content_mg_per_g: float
    glycogen_clearance_rate: float
    ros_level: float
    hypertrophy_signal: float
    cumulative_damage: float
    survival_probability: float

@dataclass
class CellSimulationResult:
    candidate_id: int
    mutation_type: str
    vector_copy_number: float
    doses: int
    trajectory: List[CellState]
    lamp2_restoration_pct: float
    glycogen_clearance_pct: float
    autophagic_flux_restoration_pct: float
    lysosomal_ph_normalization: float
    hypertrophy_reversal_pct: float
    cell_survival_at_1y: float
    cell_survival_at_5y: float
    cell_survival_at_10y: float
    functional_cure_score: float
    is_functional_cure: bool
    pathway_details: Dict

class DanonCellSimulator:
    def __init__(self):
        self.cp = CELL_PARAMS
        self.ip = INTEGRATION_PARAMS
        self.tp = THERAPY_PARAMS
        self.rng = np.random.RandomState(42)

    def _ode_system(self, t: float, y: np.ndarray,
                     therapy_start: float, vcn: float, expression_efficiency: float,
                     splicing_efficiency: float, mutation_severity: float) -> np.ndarray:
        L, pH, H, AP, AF, G, ROS, D = y

        L_target = 0.02 + mutation_severity * (1.0 - 0.02)
        therapy_active = 1.0 if t >= therapy_start else 0.0
        if therapy_active:
            lag = 1.0 - np.exp(-(t - therapy_start) / 24.0)
            vcn_effective = vcn * expression_efficiency * splicing_efficiency * lag
            L_target = min(1.0, 0.02 + vcn_effective * self.tp["lamp2_expression_per_vector_copy"])

        dL = (L_target - L) * np.log(2) / self.cp["lamp2_turnover_half_life_hours"]

        pH_target = self.cp["lysosomal_ph_danon"] - (self.cp["lysosomal_ph_danon"] - self.cp["lysosomal_ph_healthy"]) * (L ** 0.7)
        dpH = (pH_target - pH) * 0.03

        H_target = max(0.01, 0.15 + 0.85 * np.clip((self.cp["lysosomal_ph_healthy"] + 0.8 - pH) / 1.8, 0, 1))
        dH = (H_target - H) * 0.015

        AP_target = self.cp["baseline_autophagosome_count"] + (self.cp["autophagosome_count_danon"] - self.cp["baseline_autophagosome_count"]) * (1.0 - L)
        dAP = (AP_target - AP) * 0.005 - 0.02 * AF * AP

        AF_target = self.cp["autophagic_flux_danon"] + (self.cp["autophagic_flux_healthy"] - self.cp["autophagic_flux_danon"]) * (0.6 * L + 0.4 * H)
        dAF = (AF_target - AF) * 0.015

        G_baseline = self.cp["glycogen_content_danon_mg_per_g"]
        G_healthy = self.cp["glycogen_content_healthy_mg_per_g"]
        synthesis_rate = 0.002 * (G_baseline / G_healthy)
        clearance = 0.1 * AF * G
        dG = synthesis_rate * G_healthy - clearance

        ROS_baseline = self.cp["ros_level_danon"]
        ROS_healthy = self.cp["ros_level_healthy"]
        dROS = 0.005 * max(0, (G - G_healthy) / G_healthy) * (ROS_baseline - ROS_healthy) - 0.01 * (ROS - ROS_healthy)

        hydrolase_efficacy = max(0.01, 0.12 + 0.88 * L * H)
        autophagic_efficacy = max(0.01, 0.12 + 0.88 * AF)
        damage_rate = 0.0005 * max(0, (G - G_healthy) / G_healthy) * (1.0 - 0.5 * hydrolase_efficacy)
        dD = damage_rate * (1.0 + 0.3 * max(0, ROS - ROS_healthy)) - 0.001 * D * min(1, L)

        return np.array([dL, dpH, dH, dAP, dAF, dG, dROS, dD])

    def simulate_cell(self, candidate_id: int = 0,
                       mutation_type: str = "null",
                       vector_copy_number: float = 10.0,
                       doses: int = 6,
                       cardiac_tropism: float = 0.72,
                       hepatic_avoidance: float = 0.85,
                       promoter_score: float = 1.0,
                       mirna_score: float = 0.93,
                       splicing_efficiency: float = 0.82,
                       dual_vector_penalty: float = 1.0,
                       immune_evasion: float = 0.6,
                       hours: int = None) -> CellSimulationResult:
        if hours is None:
            hours = self.ip["simulation_hours"]

        mutation_map = {
            "null": 0.02, "splice_site": 0.08, "missense_catalytic": 0.15,
            "partial_deletion": 0.10,
        }
        mutation_severity = mutation_map.get(mutation_type, 0.02)

        immune_penalty = 1.0 - 0.3 * (1.0 - immune_evasion)
        hepatic_penalty = 1.0 - 0.15 * (1.0 - hepatic_avoidance)
        expression_efficiency = promoter_score * (0.5 + 0.5 * mirna_score) * immune_penalty * hepatic_penalty * dual_vector_penalty
        vcn_effective = vector_copy_number * cardiac_tropism
        therapy_start = 48.0

        y0 = np.array([
            mutation_severity,
            self.cp["lysosomal_ph_danon"],
            self.cp["hydrolase_activity_danon"],
            self.cp["autophagosome_count_danon"],
            self.cp["autophagic_flux_danon"],
            self.cp["glycogen_content_danon_mg_per_g"],
            self.cp["ros_level_danon"],
            0.0,
        ])

        t_eval = np.linspace(0, hours, min(self.ip["time_points"], hours))
        try:
            sol = solve_ivp(
                self._ode_system, (0, hours), y0, method="RK45",
                t_eval=t_eval,
                args=(therapy_start, vcn_effective, expression_efficiency,
                      splicing_efficiency, mutation_severity),
                rtol=self.ip["rtol"], atol=self.ip["atol"],
                max_step=self.ip["max_dt_hours"],
            )
        except Exception as e:
            logger.warning("  ODE integration failed: %s — using Euler fallback", e)
            sol = self._euler_fallback(y0, hours, therapy_start, vcn_effective,
                                        expression_efficiency, splicing_efficiency,
                                        mutation_severity, t_eval)

        trajectory = []
        for i in range(len(sol.t)):
            L, pH, H, AP, AF, G, ROS, D = sol.y[:, i]
            hypert = max(0, (G - self.cp["glycogen_content_healthy_mg_per_g"]) / self.cp["hypertrophy_threshold_glycogen_mg_per_g"])
            surv = np.exp(-D)
            trajectory.append(CellState(
                hours=float(sol.t[i]), lamp2_fraction=float(L),
                lysosomal_ph=float(pH), hydrolase_activity=float(H),
                autophagosome_count_norm=float(AP / self.cp["baseline_autophagosome_count"]),
                autophagic_flux=float(AF),
                glycogen_content_mg_per_g=float(G),
                glycogen_clearance_rate=float(0.1 * AF * G),
                ros_level=float(ROS), hypertrophy_signal=float(hypert),
                cumulative_damage=float(D), survival_probability=float(surv),
            ))

        final = trajectory[-1] if trajectory else trajectory
        pre = next((s for s in trajectory if s.hours >= therapy_start + 24), trajectory[0]) if trajectory else None

        if pre and final:
            lamp2_restore = final.lamp2_fraction * 100
            gly_clear = max(0, (pre.glycogen_content_mg_per_g - final.glycogen_content_mg_per_g) / max(pre.glycogen_content_mg_per_g, 0.01)) * 100
            flux_restore = (final.autophagic_flux - self.cp["autophagic_flux_danon"]) / (self.cp["autophagic_flux_healthy"] - self.cp["autophagic_flux_danon"] + 0.01) * 100
            ph_norm = max(0, (self.cp["lysosomal_ph_danon"] - final.lysosomal_ph) / (self.cp["lysosomal_ph_danon"] - self.cp["lysosomal_ph_healthy"]))
            hyp_reversal = max(0, (pre.hypertrophy_signal - final.hypertrophy_signal) / max(pre.hypertrophy_signal, 0.01)) * 100 if pre.hypertrophy_signal > 0.01 else 100
        else:
            lamp2_restore = gly_clear = flux_restore = hyp_reversal = 0.0
            ph_norm = 0.0

        surv_1y = np.exp(-final.cumulative_damage * 1.0) if trajectory else 0
        surv_5y = np.exp(-final.cumulative_damage * 5.0) if trajectory else 0
        surv_10y = np.exp(-final.cumulative_damage * 10.0) if trajectory else 0

        clinical_cure_threshold_glycogen = self.cp["glycogen_content_healthy_mg_per_g"] * 2.0
        functional_cure = (
            final.lamp2_fraction >= 0.30 and
            final.autophagic_flux >= 0.35 and
            final.glycogen_content_mg_per_g <= clinical_cure_threshold_glycogen and
            final.lysosomal_ph <= 5.7 and
            final.survival_probability >= 0.85
        )

        score = (
            0.20 * min(lamp2_restore / 100, 1) +
            0.15 * min(gly_clear / 100, 1) +
            0.15 * min(flux_restore / 100, 1) +
            0.10 * ph_norm +
            0.10 * min(hyp_reversal / 100, 1) +
            0.10 * min(surv_1y, 1) +
            0.10 * min(surv_5y, 1) +
            0.10 * (1.0 if functional_cure else 0.0)
        )

        return CellSimulationResult(
            candidate_id=candidate_id, mutation_type=mutation_type,
            vector_copy_number=vector_copy_number, doses=doses,
            trajectory=trajectory,
            lamp2_restoration_pct=float(lamp2_restore),
            glycogen_clearance_pct=float(gly_clear),
            autophagic_flux_restoration_pct=float(flux_restore),
            lysosomal_ph_normalization=float(ph_norm),
            hypertrophy_reversal_pct=float(hyp_reversal),
            cell_survival_at_1y=float(np.clip(surv_1y, 0, 1)),
            cell_survival_at_5y=float(np.clip(surv_5y, 0, 1)),
            cell_survival_at_10y=float(np.clip(surv_10y, 0, 1)),
            functional_cure_score=float(np.clip(score, 0, 1)),
            is_functional_cure=functional_cure,
            pathway_details={
                "lamp2_target": self.cp["healthy_lamp2_copies_per_cell"],
                "lysosomal_ph_healthy": self.cp["lysosomal_ph_healthy"],
                "glycogen_healthy_mg_per_g": self.cp["glycogen_content_healthy_mg_per_g"],
                "baseline_autophagosome_count": self.cp["baseline_autophagosome_count"],
                "autophagic_flux_healthy": self.cp["autophagic_flux_healthy"],
            },
        )

    def _euler_fallback(self, y0: np.ndarray, hours: float,
                         therapy_start: float, vcn: float,
                         expression_efficiency: float, splicing_efficiency: float,
                         mutation_severity: float, t_eval: np.ndarray) -> type("Sol", (), {}):
        dt = hours / max(len(t_eval) - 1, 1)
        y = y0.copy()
        ys = [y.copy()]
        for t in t_eval[1:]:
            dy = self._ode_system(t, y, therapy_start, vcn, expression_efficiency,
                                   splicing_efficiency, mutation_severity)
            y = y + dy * dt
            y = np.maximum(y, 0)
            ys.append(y.copy())
        sol = type("Sol", (), {"t": t_eval, "y": np.array(ys).T})
        return sol

    def dose_response_panel(self, candidate_id: int = 0,
                             mutation_type: str = "null",
                             vcn_range: List[float] = None,
                             cardiac_tropism: float = 0.72,
                             promoter_score: float = 1.0,
                             splicing_efficiency: float = 0.82) -> List[Dict]:
        if vcn_range is None:
            vcn_range = [0, 1, 3, 5, 10, 20, 50, 100]
        results = []
        for vcn in vcn_range:
            r = self.simulate_cell(
                candidate_id, mutation_type, vcn, 6,
                cardiac_tropism=cardiac_tropism, promoter_score=promoter_score,
                splicing_efficiency=splicing_efficiency,
            )
            results.append({
                "vector_copy_number": vcn,
                "lamp2_restoration_pct": r.lamp2_restoration_pct,
                "glycogen_clearance_pct": r.glycogen_clearance_pct,
                "autophagic_flux": r.autophagic_flux_restoration_pct,
                "functional_cure_score": r.functional_cure_score,
                "is_cure": r.is_functional_cure,
                "survival_5y": r.cell_survival_at_5y,
            })
            if r.is_functional_cure and vcn > 0:
                break
        return results

    def ec50_analysis(self, mutation_type: str = "null") -> Dict:
        panel = self.dose_response_panel(mutation_type=mutation_type)
        cure_dose = None
        ed50_dose = 0.0
        max_score = 0.0
        for pt in panel:
            if pt["is_cure"] and cure_dose is None:
                cure_dose = pt["vector_copy_number"]
            if pt["functional_cure_score"] > max_score:
                max_score = pt["functional_cure_score"]
            if max_score > 0 and pt["functional_cure_score"] >= max_score * 0.5 and ed50_dose == 0.0:
                ed50_dose = pt["vector_copy_number"]
        return {
            "mutation_type": mutation_type,
            "ed50_vcn": ed50_dose,
            "min_cure_vcn": cure_dose,
            "max_score": max_score,
        }

    def simulate_population(self, candidate_id: int = 0,
                             cardiac_tropism: float = 0.72,
                             immune_evasion: float = 0.6,
                             promoter_score: float = 1.0,
                             mirna_score: float = 0.93,
                             splicing_efficiency: float = 0.82,
                             vector_copy_number: float = 10.0,
                             n_patients: int = 100) -> Dict:
        mutations = ["null", "splice_site", "missense_catalytic", "partial_deletion"]
        mutation_weights = [0.40, 0.15, 0.35, 0.10]
        vcn_noise = lambda: vector_copy_number * (0.5 + self.rng.rand() * 1.0)
        outcomes = {"cured": 0, "improved": 0, "failed": 0, "results": []}
        for _ in range(n_patients):
            mut = self.rng.choice(mutations, p=mutation_weights)
            vcn = vcn_noise()
            r = self.simulate_cell(
                candidate_id, mut, vcn, 6,
                cardiac_tropism=cardiac_tropism * (0.8 + 0.4 * self.rng.rand()),
                promoter_score=promoter_score,
                mirna_score=mirna_score,
                splicing_efficiency=splicing_efficiency,
                immune_evasion=immune_evasion * (0.7 + 0.6 * self.rng.rand()),
            )
            if r.is_functional_cure:
                outcomes["cured"] += 1
            elif r.functional_cure_score >= 0.40:
                outcomes["improved"] += 1
            else:
                outcomes["failed"] += 1
            outcomes["results"].append({
                "mutation": mut, "vcn": vcn, "score": r.functional_cure_score,
                "cured": r.is_functional_cure,
                "glycogen": r.glycogen_clearance_pct,
                "flux": r.autophagic_flux_restoration_pct,
                "surv_5y": r.cell_survival_at_5y,
            })
        outcomes["cure_rate"] = outcomes["cured"] / max(n_patients, 1)
        outcomes["improvement_rate"] = outcomes["improved"] / max(n_patients, 1)
        return outcomes

    def compare_mutation_types(self, vector_copy_number: float = 10.0,
                                 cardiac_tropism: float = 0.72) -> List[Dict]:
        results = []
        for mut in ["null", "splice_site", "missense_catalytic", "partial_deletion"]:
            r = self.simulate_cell(0, mut, vector_copy_number, cardiac_tropism=cardiac_tropism)
            results.append({
                "mutation": mut,
                "lamp2_pct": r.lamp2_restoration_pct,
                "glycogen_clearance_pct": r.glycogen_clearance_pct,
                "flux_restoration_pct": r.autophagic_flux_restoration_pct,
                "cure_score": r.functional_cure_score,
                "is_cure": r.is_functional_cure,
                "survival_10y": r.cell_survival_at_10y,
            })
        return results

    def project_trial_outcome(self, vector_copy_number: float = 10.0,
                               cardiac_tropism: float = 0.72,
                               n_patients: int = 24) -> Dict:
        pop = self.simulate_population(0, cardiac_tropism, n_patients=n_patients)
        mean_score = float(np.mean([r["score"] for r in pop["results"]]))
        mean_glycogen = float(np.mean([r["glycogen"] for r in pop["results"]]))
        mean_surv_5y = float(np.mean([r["surv_5y"] for r in pop["results"]]))

        power = 0.0
        if pop["cure_rate"] > 0.5 and n_patients >= 12:
            z = (pop["cure_rate"] - 0.3) * np.sqrt(n_patients) / np.sqrt(pop["cure_rate"] * (1 - pop["cure_rate"]) + 0.01)
            power = float(1.0 - 0.5 * (1.0 + np.tanh(-z / np.sqrt(2))))

        return {
            "n_patients": n_patients,
            "cure_rate": pop["cure_rate"],
            "improvement_rate": pop["improvement_rate"],
            "mean_functional_score": mean_score,
            "mean_glycogen_clearance_pct": mean_glycogen,
            "mean_5y_survival": mean_surv_5y,
            "statistical_power": float(np.clip(power, 0, 1)),
            "trial_feasible": power >= 0.80 and pop["cure_rate"] >= 0.4,
        }

    def utcl_score(self) -> float:
        return 0.10

    def our_best_score(self, cardiac_tropism: float = 0.72,
                       vector_copy_number: float = 10.0) -> float:
        r = self.simulate_cell(mutation_type="null",
                                vector_copy_number=vector_copy_number,
                                cardiac_tropism=cardiac_tropism)
        return r.functional_cure_score
