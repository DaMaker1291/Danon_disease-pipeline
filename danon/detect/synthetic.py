"""Synthetic longitudinal observations with known ground truth.

Everything the mission's Phase 1 asked the generator to do:

* known changes planted on a known day with a known direction;
* measurement noise with a DECLARED sigma that follows each reading;
* missing observations (irregular sampling gaps, plus whole-day dropouts);
* negative controls: cases with no change at all.

Each generated case returns a validated :class:`ObservationStream` for the
detector and a separate :class:`GroundTruth` book for the evaluator. That
split is the anti-leakage measure: the stream cannot carry its own answer.

The generative model is deliberately simple -- exp(growth) in radius with a
carried exponential molecule floor and an optional rate change mid-study.
Its purpose is to be knowable, not to be an oncology model; the mission
asks for a controlled experiment first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from danon.detect.schema import (
    ETMeasurement,
    GroundTruth,
    MeasurementKind,
    ObservationStream,
    Provenance,
    Scenario,
    ScenarioCase,
    ScenarioKind,
    Units,
)

#: The generative growth model, stated in the record's own vocabulary.
ET_GROWTH_MODEL = "radius(t) = r0 * exp(k * t), with an optional rate change at the planted day"

GROWTH_CHANGE_DAY_DEFAULT = 120
#: multiplicative factor applied to the growth rate AT the planted day
GROWTH_CHANGE_DELTA_DEFAULT = 2.0


@dataclass(frozen=True)
class NoiseModel:
    """Declared per-reading noise: sigma is known to every consumer."""

    sigma: float = 2.0
    Cadence: int = 7  # default sampling cadence (days)

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.Cadence < 1:
            raise ValueError("Cadence must be >= 1 day")


@dataclass
class ScenarioConfig:
    """Generator settings for one scenario batch."""

    n_cases: int = 20
    fraction_negatives: float = 0.25
    duration_days: int = 365
    cadence_days: int = 7
    baseline_rate: float = 0.003       # per-day exponential growth rate
    changed_rate_multiplier: float = GROWTH_CHANGE_DELTA_DEFAULT
    change_day_range: Tuple[int, int] = (90, 180)
    r0_range: Tuple[float, float] = (8.0, 15.0)
    missing_fraction: float = 0.1
    noise: NoiseModel = field(default_factory=NoiseModel)
    seed: int = 20261010

    def __post_init__(self) -> None:
        if self.n_cases < 2:
            raise ValueError("need at least 2 cases for a scenario")
        if not 0.0 <= self.fraction_negatives <= 1.0:
            raise ValueError("fraction_negatives must be in [0, 1]")
        if self.duration_days < 30:
            raise ValueError("duration_days must be >= 30")
        if self.change_day_range[0] < 2 or self.change_day_range[1] >= self.duration_days:
            raise ValueError("change_day_range must leave unchangeable margins")
        if self.changed_rate_multiplier <= 0:
            raise ValueError("changed_rate_multiplier must be positive")
        if not 0 <= self.missing_fraction < 0.5:
            raise ValueError("missing_fraction must be in [0, 0.5)")


@dataclass
class SyntheticPopulation:
    """A generated scenario, sealed: streams handed out, truth kept apart."""

    scenario: Scenario
    config: ScenarioConfig

    @property
    def streams(self) -> List[ObservationStream]:
        return [c.stream for c in self.scenario.cases]

    @property
    def truths(self) -> List[GroundTruth]:
        return [c.truth for c in self.scenario.cases]


def generate_scenario(config: Optional[ScenarioConfig] = None) -> SyntheticPopulation:
    """Generate one scenario batch: observe, plant, hide, return."""
    cfg = config or ScenarioConfig()
    rng = np.random.default_rng(cfg.seed)

    n_change = int(round(cfg.n_cases * (1.0 - cfg.fraction_negatives)))
    n_neg = cfg.n_cases - n_change
    prov = Provenance(
        source="danon.detect.synthetic.generate_scenario",
        seed=cfg.seed,
        scenario_kind="mixed",
        config_fingerprint=f"seed={cfg.seed},n={cfg.n_cases}",
    )

    cases: List[ScenarioCase] = []
    for i in range(cfg.n_cases):
        is_negative = i >= n_change
        case_seed = int(rng.integers(0, 2**31 - 1))
        case_rng = np.random.default_rng(case_seed)
        r0 = float(case_rng.uniform(*cfg.r0_range))
        if is_negative:
            change_day = None
            rate_after = cfg.baseline_rate
        else:
            change_day = int(case_rng.integers(*cfg.change_day_range))
            rate_after = cfg.baseline_rate * cfg.changed_rate_multiplier

        obs_day_grid = np.arange(0, cfg.duration_days + 1, cfg.cadence_days)
        # missing observations: drop a fraction of scheduled days (never day 0)
        drop_mask = case_rng.random(len(obs_day_grid)) < cfg.missing_fraction
        drop_mask[0] = False
        kept = obs_day_grid[~drop_mask]
        values = []
        sigmas = []
        for d in kept:
            rate = cfg.baseline_rate if (change_day is None or d < change_day) else rate_after
            true_radius = r0 * np.exp(rate * d)
            measured = true_radius + case_rng.normal(0.0, cfg.noise.sigma)
            values.append(float(measured))
            sigmas.append(float(cfg.noise.sigma))

        measurements = tuple(
            ETMeasurement(
                day=int(d),
                kind=MeasurementKind.RADIUS,
                units=Units.DIAMETER_MM,
                value=v,
                sigma=s,
                provenance=prov,
            )
            for d, v, s in zip(kept, values, sigmas)
        )
        stream = ObservationStream(
            case_id=f"synthetic-case-{i:03d}",
            observations=measurements,
            provenance=prov,
        )
        truth = GroundTruth(
            case_id=stream.case_id,
            planted_change_day=change_day,
            planted_direction="growth" if change_day is not None else "",
            planted_magnitude=float(cfg.changed_rate_multiplier) if change_day is not None else 0.0,
            is_negative_control=is_negative,
            missing_days=tuple(int(d) for d in obs_day_grid[drop_mask]),
            noise_sigma_used=cfg.noise.sigma,
            scenario_kind=ScenarioKind.SINGLE_CHANGE.value if not is_negative else ScenarioKind.NEGATIVE_CONTROL.value,
        )
        cases.append(ScenarioCase(stream=stream, truth=truth))

    scenario = Scenario(
        name="phase1-synthetic-mixed",
        seed=cfg.seed,
        cases=tuple(cases),
    )
    return SyntheticPopulation(scenario=scenario, config=cfg)
