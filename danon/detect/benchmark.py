"""Repeated experiments, recorded.

The benchmark runs the detector over MULTIPLE scenarios with different
seeds, scores each against its ground-truth book, and writes a
machine-readable record: JSON for the report and Parquet for the
run-level table. Seeds and configurations are saved; every scenario the
detector fails -- every missed change, every false alarm -- is named in
the record rather than averaged away.

Held-out discipline: nothing in this module tunes the detector's
thresholds on the scenarios it scores. The threshold set is the shipped
default, fixed before any scenario is generated. (A sweep parameter in the
config is recorded but only for multi-seed repetition, never used to pick
the shipped thresholds.)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from danon.detect.detector import DetectorConfig, detect
from danon.detect.evaluate import evaluate_detections, CALIBRATION_WINDOW_DAYS
from danon.detect.schema import ObservationStream, Provenance
from danon.detect.synthetic import ScenarioConfig, SyntheticPopulation, generate_scenario


@dataclass
class RunRecord:
    """One scenario run: config, evaluation, and failures, on the record."""

    seed: int
    n_cases: int
    detector_config: Dict[str, float]
    evaluation: Dict[str, Optional[float]]
    missed_cases: List[str]
    false_alarm_cases: List[str]
    delayed_cases: List[Dict[str, int]]     # [{case, planted, detector}]
    n_insufficient: int                     # refusals, counted not averaged away
    insufficient_cases: List[str]
    scenario_fingerprint: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class BenchmarkResults:
    """Aggregate over run records, plus the records themselves."""

    runs: List[RunRecord] = field(default_factory=list)
    mean_sensitivity: float = float("nan")
    mean_specificity: float = float("nan")
    mean_precision: Optional[float] = None
    mean_false_alarms_per_1000_days: float = float("nan")
    mean_delay_days: Optional[float] = None
    calibration_rate: Optional[float] = None
    total_cases: int = 0
    total_flagged: int = 0
    total_missed: int = 0
    total_false_alarms: int = 0

    def summary_line(self) -> str:
        prec = f"{self.mean_precision:.3f}" if self.mean_precision is not None else "n/a"
        delay = f"{self.mean_delay_days:.1f}" if self.mean_delay_days is not None else "n/a"
        calib = f"{self.calibration_rate:.3f}" if self.calibration_rate is not None else "n/a"
        return (
            f"mean sensitivity {self.mean_sensitivity:.3f} | "
            f"mean specificity {self.mean_specificity:.3f} | "
            f"mean precision {prec} | "
            f"false alarms/1000d {self.mean_false_alarms_per_1000_days:.2f} | "
            f"mean detection delay {delay} d | calibration {calib} | "
            f"{self.total_flagged} flagged / {self.total_missed} missed / "
            f"{self.total_false_alarms} false alarms over {self.total_cases} cases"
        )


def run_benchmark(
    seeds: Sequence[int],
    config: Optional[ScenarioConfig] = None,
    detector_config: Optional[DetectorConfig] = None,
    *,
    calibration_window_days: int = CALIBRATION_WINDOW_DAYS,
) -> BenchmarkResults:
    """Run, score, and record a multi-seed benchmark sweep."""
    cfg = config or ScenarioConfig()
    dcfg = detector_config or DetectorConfig()
    runs: List[RunRecord] = []

    for seed in seeds:
        population: SyntheticPopulation = generate_scenario(
            ScenarioConfig(
                n_cases=cfg.n_cases,
                fraction_negatives=cfg.fraction_negatives,
                duration_days=cfg.duration_days,
                cadence_days=cfg.cadence_days,
                baseline_rate=cfg.baseline_rate,
                changed_rate_multiplier=cfg.changed_rate_multiplier,
                change_day_range=cfg.change_day_range,
                r0_range=cfg.r0_range,
                missing_fraction=cfg.missing_fraction,
                noise=cfg.noise,
                seed=seed,
            )
        )
        results = [detect(c.stream, dcfg) for c in population.scenario.cases]
        evaluation = evaluate_detections(
            results,
            population.truths,
            horizon_days_per_negative=cfg.duration_days,
            calibration_window_days=calibration_window_days,
        )
        insufficient = [
            r.case_id
            for r in results
            if r.status.value == "insufficient_evidence"
        ]
        runs.append(
            RunRecord(
                seed=seed,
                n_cases=population.scenario.n_cases,
                detector_config={
                    "confirmation_threshold": dcfg.confirmation_threshold,
                    "suspicion_threshold": dcfg.suspicion_threshold,
                    "min_points_for_call": dcfg.min_points_for_call,
                    "min_left_segment": dcfg.min_left_segment,
                    "min_right_segment": dcfg.min_right_segment,
                },
                evaluation={
                    "sensitivity": evaluation.sensitivity,
                    "specificity": evaluation.specificity,
                    "precision": evaluation.precision,
                    "false_alarms_per_1000_days": evaluation.false_alarms_per_1000_days,
                    "median_delay_days": evaluation.median_delay_days,
                    "calibration_rate": evaluation.calibration_rate,
                },
                missed_cases=list(evaluation.missed_cases),
                false_alarm_cases=list(evaluation.false_alarm_cases),
                delayed_cases=[
                    {"case_id": c, "planted_day": p, "detector_day": d}
                    for c, p, d in evaluation.delayed_cases
                ],
                n_insufficient=len(insufficient),
                insufficient_cases=insufficient,
                scenario_fingerprint=(
                    f"n={cfg.n_cases},neg={cfg.fraction_negatives},"
                    f"mult={cfg.changed_rate_multiplier},missing={cfg.missing_fraction},"
                    f"noise={cfg.noise.sigma}"
                ),
            )
        )

    # Aggregate with explicit nans-suppressed means -- a mean over an empty
    # list is a lie; "not measured" is the honest report for that slot.
    def mean_or_none(vals: List[Optional[float]]) -> Optional[float]:
        present = [v for v in vals if v is not None]
        return float(np.mean(present)) if present else None

    return BenchmarkResults(
        runs=runs,
        mean_sensitivity=float(np.mean([r.evaluation["sensitivity"] for r in runs])),
        mean_specificity=float(np.mean([r.evaluation["specificity"] for r in runs])),
        mean_precision=mean_or_none([r.evaluation["precision"] for r in runs]),
        mean_false_alarms_per_1000_days=float(
            np.mean([r.evaluation["false_alarms_per_1000_days"] for r in runs])
        ),
        mean_delay_days=mean_or_none([r.evaluation["median_delay_days"] for r in runs]),
        calibration_rate=mean_or_none([r.evaluation["calibration_rate"] for r in runs]),
        total_cases=sum(r.n_cases for r in runs),
        total_flagged=sum(
            1
            for r in runs
            for e in [r.evaluation]
            if e["sensitivity"] is not None
        ),
        total_missed=sum(len(r.missed_cases) for r in runs),
        total_false_alarms=sum(len(r.false_alarm_cases) for r in runs),
    )


def record_to_dict(results: BenchmarkResults) -> Dict[str, object]:
    """Full machine-readable record: aggregates + per-run rows."""
    return {
        "benchmark_name": "evotwin.phase1.change_detection",
        "aggregates": {
            "mean_sensitivity": results.mean_sensitivity,
            "mean_specificity": results.mean_specificity,
            "mean_precision": results.mean_precision,
            "mean_false_alarms_per_1000_days": results.mean_false_alarms_per_1000_days,
            "mean_delay_days": results.mean_delay_days,
            "calibration_rate": results.calibration_rate,
            "total_cases": results.total_cases,
            "total_missed": results.total_missed,
            "total_false_alarms": results.total_false_alarms,
        },
        "runs": [r.to_dict() for r in results.runs],
        "limitations": _limitations_text(),
    }


def _limitations_text() -> List[str]:
    """The limitations the record carries verbatim -- measured, not aspirational."""
    return [
        "Synthetic only: the generator is exp-growth in radius with a rate "
        "change, not a tumour model; sensitivity on this data says nothing "
        "about real imaging or liquid-biopsy series.",
        "Single change per case: no multiple-change, no gradual-drift, and no "
        "regression-to-baseline scenarios are in Phase 1.",
        "Call-level, not event-level metrics: precision counts flagged CASES, "
        "and a case can only flag once in this version.",
        "The detector knows the generative family (exp-growth, log-linear); "
        "a model violation is out of scope and would need a new baseline.",
        "Calibration is scored only for confirmed calls with a planted day; "
        "suspected-only calls lower sensitivity but are not delay-scored.",
        "No clinical claim of any kind: the report describes a controlled "
        "synthetic experiment; no patient benefit is implied or measured.",
    ]


def write_parquet_record(results: BenchmarkResults, path: str) -> None:
    """Persist run-level rows to Parquet for cross-experiment comparison."""
    import pandas as pd

    rows = []
    for r in results.runs:
        rows.append(
            {
                "seed": r.seed,
                "n_cases": r.n_cases,
                **{f"eval_{k}": v for k, v in r.evaluation.items()},
                "n_missed": len(r.missed_cases),
                "n_false_alarms": len(r.false_alarm_cases),
                "n_insufficient": r.n_insufficient,
                "scenario_fingerprint": r.scenario_fingerprint,
                "confirmation_threshold": r.detector_config["confirmation_threshold"],
                "suspicion_threshold": r.detector_config["suspicion_threshold"],
                "min_left_segment": r.detector_config["min_left_segment"],
            }
        )
    df = pd.DataFrame(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def write_json_record(results: BenchmarkResults, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record_to_dict(results), fh, indent=2, default=str)


def read_json_record(path: str) -> Dict[str, object]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
