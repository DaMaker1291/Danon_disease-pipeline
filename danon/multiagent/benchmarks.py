"""
Disease-area benchmark factories.

Cardiac / ischemic heart disease benchmarks. These are *simulators* and
*scoring helpers*, not the whole pipeline — they feed scenarios into the
multi-agent runtime so hundreds of sub-agents can be evaluated against
shared domain tasks.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from danon.multiagent.core import (
    AgentJudgement,
    AgentJudgementOutcome,
    DiseaseCohort,
    ScenarioBuilder,
    SubAgentTask,
    TaskObjective,
)

# The real vascular physics. Imported at module level so a broken simulator
# fails loudly at import time rather than silently falling back to a surrogate.
from danon.swarm import (
    ArteryGeometry,
    SwarmConfig,
    SwarmSimulator,
    normalise_to_reference,
)

# Sub-agent run sizing. Small by default so a campaign of hundreds of agents
# stays affordable on a laptop; override per task via payload keys
# ``swarm_n_agents`` / ``swarm_steps``.
DEFAULT_SWARM_N_AGENTS = 8_000
DEFAULT_SWARM_STEPS = 200
DEFAULT_SWARM_DT_S = 2.0e-3


# =====================================================================
# Cohort: cardiovascular / ischemic heart disease
# =====================================================================

class CVDHeartDiseaseCohort(DiseaseCohort):
    """Cardiovascular heart disease cohort — ischemic cardiomyopathy modeling.

    This is the *deployment target* for the cardiac sub-agent swarm.
    """

    def __init__(
        self,
        *,
        id: str = "cvd_heart_disease",
        name: str = "CVD Heart Disease",
        description: str = "Ischemic cardiomyopathy / coronary artery disease modeling cohort.",
    ) -> None:
        super().__init__(id=id, name=name, description=description)

    def scenario_builder(self, task: SubAgentTask) -> ScenarioBuilder:
        return IschemicPlaqueScenarioBuilder(task=task, cohort=self)

    def validate_result(self, task: SubAgentTask, result: Mapping[str, Any]) -> Tuple[bool, str]:
        if not isinstance(result, Mapping):
            return False, "result must be a mapping"
        if "plaque_metrics" not in result:
            return False, "missing plaque_metrics"
        return True, ""

    def score(self, task: SubAgentTask, result: Mapping[str, Any]) -> Dict[str, float]:
        m = result.get("plaque_metrics", {})
        reg = float(m.get("plaque_reduction_pct", 0.0))
        perf = float(m.get("reperfusion_gain_pct", 0.0))
        # efficacy: weighted sum; more plaque reduction and more reperfusion → better
        efficacy = 0.6 * max(0.0, reg) + 0.4 * max(0.0, perf)
        safety = float(m.get("off_target_healthy_wall_adhesion_pct", 100.0))
        safety_score = max(0.0, 100.0 - safety)
        return {
            "efficacy": efficacy,
            "safety": safety_score,
            "plaque_reduction_pct": reg,
            "reperfusion_gain_pct": perf,
            "off_target_healthy_wall_adhesion_pct": safety,
        }


class IschemicHeartDiseaseCohort(CVDHeartDiseaseCohort):
    """Sub-cohort focused specifically on ischemic (oxygen-starved) tissue."""

    def __init__(
        self,
        *,
        id: str = "ischemic_heart_disease",
        name: str = "Ischemic Heart Disease",
        description: str = "Ischemic myocardium / hibernating myocardium cohort.",
    ) -> None:
        super().__init__(id=id, name=name, description=description)


# =====================================================================
# Cohort inclusion criteria + measurable endpoints
# =====================================================================

@dataclass
class CohortCriteria:
    """Inclusion/exclusion rules defining who (or what) belongs to a cohort.

    For the benchtop proxy these mirror the physical rig's acceptance window;
    for a clinical cohort they mirror trial eligibility.
    """

    min_occlusion_pct: float = 50.0
    max_occlusion_pct: float = 95.0
    age_range_years: Tuple[int, int] = (18, 85)
    required_findings: Tuple[str, ...] = (
        "reduced_coronary_flow_reserve",
        "reversible_ischemia_on_stress_test",
    )
    excluded_comorbidities: Tuple[str, ...] = (
        "active_malignancy",
        "severe_renal_impairment",
        "recent_stroke",
    )
    lvef_pct_range: Tuple[float, float] = (25.0, 55.0)

    def admits(self, occlusion_pct: float, lvef_pct: float, age: int = 60) -> bool:
        """Convenience predicate used to filter candidate scenarios."""
        lo_age, hi_age = self.age_range_years
        lo_lv, hi_lv = self.lvef_pct_range
        return (
            self.min_occlusion_pct <= occlusion_pct <= self.max_occlusion_pct
            and lo_age <= age <= hi_age
            and lo_lv <= lvef_pct <= hi_lv
        )


@dataclass
class CohortEndpoints:
    """Primary and secondary endpoints a cohort is scored against.

    These names deliberately match the benchtop rig readouts (pressure drop
    across the occlusion, plaque regression, reperfusion) so the in-silico
    simulation and the physical experiment report on the same axes.
    """

    primary: Tuple[str, ...] = (
        "plaque_reduction_pct",
        "pressure_drop_reduction_pct",
    )
    secondary: Tuple[str, ...] = (
        "reperfusion_gain_pct",
        "off_target_healthy_wall_adhesion_pct",
        "distal_embolic_particle_count",
    )
    safety: Tuple[str, ...] = (
        "off_target_healthy_wall_adhesion_pct",
        "distal_embolic_particle_count",
        "immune_activation_pg_ml",
    )
    # Minimum clinically-meaningful deltas, used by the judge to tier results.
    minimal_meaningful_delta: Dict[str, float] = field(
        default_factory=lambda: {
            "plaque_reduction_pct": 5.0,
            "pressure_drop_reduction_pct": 10.0,
            "reperfusion_gain_pct": 5.0,
        }
    )

    def all_endpoints(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(self.primary + self.secondary + self.safety))

    def meets_minimal_delta(self, metrics: Mapping[str, float]) -> bool:
        """True when every scored endpoint clears its meaningful-delta floor."""
        for name, floor in self.minimal_meaningful_delta.items():
            if float(metrics.get(name, 0.0)) < floor:
                return False
        return True


# =====================================================================
# Plaque surrogate model
# =====================================================================

@dataclass
class PlaqueSurrogate:
    """Simple synthetic-plaque proxy inspired by the ISEF benchtop recipe.

    agarose : lard(fat) : calcium_carbonate  —  fraction of cross-section
    occluded by surrogate material at a bifurcation point.
    """

    occlusion_pct: float = 70.0  # % of lumen cross-section blocked
    composition: Dict[str, float] = field(default_factory=lambda: {
        "agarose": 0.10,
        "lipid_lard": 0.60,
        "calcium_carbonate": 0.30,
    })
    stiffness_kpa: float = 120.0  # surrogate modulus-ish
    location: str = "left_anterior_descending"  # LAD-ish

    def digest_rate(self, agent_potency: float) -> float:
        """Approx lipid component removal rate per simulated minute."""
        lipid_frac = self.composition.get("lipid_lard", 0.5)
        # potency in [0,1]; higher potency = faster digestion
        return lipid_frac * agent_potency * 0.05  # % per simulated minute


# =====================================================================
# Fluid dynamics micro-model
# =====================================================================

@dataclass
class FlowState:
    """Tiny 1D-ish perfusion model — not Navier-Stokes, but sufficient to
    rank sub-agents by reperfusion effect."""

    pressure_upstream_kpa: float = 16.0   # ~120 mmHg
    pressure_downstream_kpa: float = 10.0  # ~75 mmHg
    resistance_relative: float = 1.0       # multiplier on baseline resistance
    heart_rate_bpm: float = 70.0
    shear_stress_pa: float = 1.5

    @property
    def flow_relative(self) -> float:
        """Relative flow ∝ ΔP / R."""
        return (self.pressure_upstream_kpa - self.pressure_downstream_kpa) / max(0.05, self.resistance_relative)


# =====================================================================
# Scenario builder / simulator
# =====================================================================

class IschemicPlaqueScenarioBuilder(ScenarioBuilder):
    """Builds a plaque-regression simulation scenario + scoring for a task."""

    def __init__(self, *, task: SubAgentTask, cohort: DiseaseCohort) -> None:
        super().__init__(task=task, cohort=cohort)
        self._plaque = PlaqueSurrogate()
        self._flow = FlowState()
        self._rng_seed = int(hashlib.sha256(task.agent_id.encode()).hexdigest(), 16) % (2**31)

    def build(self) -> Dict[str, Any]:
        pot = float(self.task.payload.get("agent_potency", 0.5))
        minutes = int(self.task.payload.get("run_minutes", 48 * 60))  # 48 h default
        n_agents = int(self.task.payload.get("swarm_n_agents", DEFAULT_SWARM_N_AGENTS))
        steps = int(self.task.payload.get("swarm_steps", DEFAULT_SWARM_STEPS))
        avidity = float(self.task.payload.get("agent_avidity", 0.5))
        vcam1 = float(self.task.payload.get("vcam1_density", 0.8))
        scenario = {
            "plaque_initial_occupancy_pct": self._plaque.occlusion_pct,
            "plaque_composition": dict(self._plaque.composition),
            "plaque_stiffness_kpa": self._plaque.stiffness_kpa,
            "plaque_location": self._plaque.location,
            "agent_potency": pot,
            "run_minutes": minutes,
            "flow_initial": self._flow_flow_dict(),
            "seed": self._rng_seed,
        }
        metrics = self.simulate_real(
            potency=pot, avidity=avidity, vcam1_density=vcam1,
            n_agents=n_agents, steps=steps,
        )
        # The physical run and its normalised form are both carried forward, so
        # every downstream consumer reads the same numbers and the time scaling
        # is explicit rather than implied.
        scenario["plaque_metrics"] = metrics
        scenario["swarm_config"] = {
            "n_agents": n_agents,
            "steps": steps,
            "dt_s": DEFAULT_SWARM_DT_S,
            "simulated_seconds": steps * DEFAULT_SWARM_DT_S,
            "agent_steps": int(n_agents) * int(steps),
        }
        scenario["plaque_final_occupancy_pct"] = self._plaque.occlusion_pct * (
            1.0 - metrics["plaque_reduction_physical_pct"] / 100.0
        )
        scenario["reperfusion_gain_pct"] = metrics["pressure_drop_reduction_physical_pct"]
        return scenario

    def simulate_real(
        self,
        *,
        potency: float,
        avidity: float = 0.5,
        vcam1_density: float = 0.8,
        n_agents: int = DEFAULT_SWARM_N_AGENTS,
        steps: int = DEFAULT_SWARM_STEPS,
    ) -> Dict[str, float]:
        """Run the vectorised plaque ABM for this scenario.

        This is the real physics (advection, pulsatility, saturable wall
        binding, off-target bounce, plaque erosion, embolic release). The
        returned dictionary carries both the raw physical percentages and the
        version normalised to the model's own attainable maximum.
        """
        geometry = ArteryGeometry(initial_occlusion_pct=float(self._plaque.occlusion_pct))
        config = SwarmConfig(
            n_agents=int(max(1, n_agents)),
            steps=int(max(1, steps)),
            dt_s=DEFAULT_SWARM_DT_S,
            potency=float(np.clip(potency, 0.0, 1.0)),
            avidity=float(np.clip(avidity, 0.0, 1.0)),
            vcam1_density=float(np.clip(vcam1_density, 0.0, 1.0)),
            seed=int(self._rng_seed),
        )
        result = SwarmSimulator(config, geometry).run()
        return normalise_to_reference(result, geometry)

    def score(self, scenario: Dict[str, Any], result: Mapping[str, Any]) -> Dict[str, float]:
        # If the agent returned its own plaque_metrics, trust it; else recompute
        # them from the physical scenario fields through the same normaliser.
        if "plaque_metrics" in result:
            pm = dict(result["plaque_metrics"])
        elif "plaque_metrics" in scenario:
            pm = dict(scenario["plaque_metrics"])
        else:
            raise ValueError(
                "scenario carries no plaque_metrics; call build() or "
                "simulate_real() before scoring"
            )

        # Always attach the derived efficacy/safety axes so this scoring path is
        # interchangeable with DiseaseCohort.score() (the judge reads these keys).
        pm.setdefault("efficacy", 0.6 * float(pm.get("plaque_reduction_pct", 0.0))
                      + 0.4 * float(pm.get("reperfusion_gain_pct", 0.0)))
        pm.setdefault("safety", max(0.0, 100.0 - float(
            pm.get("off_target_healthy_wall_adhesion_pct", 0.0))))
        return pm

    # ------------------------------------------------------------------
    # benchtop surrogate (the physical rig, not the vascular model)
    # ------------------------------------------------------------------

    def _flow_flow_dict(self) -> Dict[str, float]:
        return {
            "pressure_upstream_kpa": self._flow.pressure_upstream_kpa,
            "pressure_downstream_kpa": self._flow.pressure_downstream_kpa,
            "resistance_relative": self._flow.resistance_relative,
            "flow_relative": self._flow.flow_relative,
            "shear_stress_pa": self._flow.shear_stress_pa,
        }


# =====================================================================
# Cohort registry / factory
# =====================================================================

def build_cohorts() -> List[DiseaseCohort]:
    """Quick factory for the cardiac cohorts this package ships."""
    return [
        CVDHeartDiseaseCohort(),
        IschemicHeartDiseaseCohort(),
    ]


def build_cvd_cohort(id: str = "cvd_heart_disease") -> DiseaseCohort:
    if id == "ischemic_heart_disease":
        return IschemicHeartDiseaseCohort()
    return CVDHeartDiseaseCohort(id=id)


# =====================================================================
# Judge
# =====================================================================

class BaseJudge:
    """Default judge for cardiac sub-agent results."""

    def __call__(
        self,
        task: SubAgentTask,
        result: Mapping[str, Any],
        metrics: Mapping[str, float],
    ) -> AgentJudgement:
        efficacy = float(metrics.get("efficacy", 0.0))
        safety = float(metrics.get("safety", 0.0))
        reg = float(metrics.get("plaque_reduction_pct", 0.0))
        perf = float(metrics.get("reperfusion_gain_pct", 0.0))

        if efficacy >= 70 and safety >= 80 and reg >= 30:
            outcome = AgentJudgementOutcome.TIER_1
            score = min(100.0, efficacy * 0.7 + safety * 0.3)
        elif efficacy >= 40 and safety >= 60:
            outcome = AgentJudgementOutcome.TIER_2
            score = min(100.0, efficacy * 0.6 + safety * 0.4)
        elif efficacy >= 15:
            outcome = AgentJudgementOutcome.TIER_3
            score = min(100.0, efficacy * 0.5 + safety * 0.5)
        else:
            outcome = AgentJudgementOutcome.REJECT
            score = max(0.0, efficacy * 0.5)

        rationale = (
            f"efficacy={efficacy:.1f} safety={safety:.1f} "
            f"plaque_reduction={reg:.1f}% reperfusion_gain={perf:.1f}%"
        )
        return AgentJudgement(
            outcome=outcome,
            score=score,
            rationale=rationale,
            metrics_snapshot=dict(metrics),
        )


def normalize_candidate_score(score: float, *, bounds: Tuple[float, float] = (0.0, 100.0)) -> float:
    lo, hi = bounds
    if hi <= lo:
        return 0.0
    v = max(lo, min(hi, score))
    return (v - lo) / (hi - lo)
