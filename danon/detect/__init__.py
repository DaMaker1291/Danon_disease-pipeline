"""Phase 1 of EvoTwin: a detection engine for synthetic longitudinal data.

This package receives observations over time, estimates the tumour's changing
state, detects unexpected changes in its dynamics, and records how accurate
those detections are against known ground truth.

RESEARCH-ONLY. The data is synthetic; nothing here touches a patient, supports
a treatment decision, or claims clinical benefit. The story this package can
establish is narrow and stated in its own record: on controlled synthetic
scenarios, a detector with known answers.

Layout
------
``schema``      Pydantic contract every observation must satisfy (timestamp,
                measurement type, units, provenance) and the strict
                ground-truth separation the leakage tests lean on.
``synthetic``   The scenario generator: known changes, measurement noise,
                missing observations, negative controls, and a ``ground_truth``
                book that is returned SEPARATELY from the observations.
``state``       The uncertainty-aware state estimate the detector reports
                alongside each change decision, and the insufficient-evidence
                path that refuses to force a confident answer.
``detector``    The statistical change-point baseline (log-ratio CUSUM on the
                observed growth rate, Weibull-light reporting instead of
                local p-values) and its threshold semantics.
``evaluate``    Independent evaluation: sensitivity, specificity, precision,
                false alarms per observation period, detection delay,
                calibration, and a hold-out score the detector never tuned on.
``benchmark``   Repeated experiments over held-out scenarios, recorded into a
                machine-readable experiment record (JSON + Parquet), a report
                with plots, and a limitations section written by measurement
                rather than by prose.
"""

from danon.detect.schema import (
    ETMeasurement,
    ETValues,
    GroundTruth,
    GenomicsSchema,
    ObservationStream,
    Provenance,
    Scenario,
    ScenarioCase,
    ScenarioKind,
    Units,
    ground_truth_summary,
)
from danon.detect.synthetic import (
    ET_GROWTH_MODEL,
    GROWTH_CHANGE_DAY_DEFAULT,
    GROWTH_CHANGE_DELTA_DEFAULT,
    NoiseModel,
    ScenarioConfig,
    SyntheticPopulation,
    generate_scenario,
)
from danon.detect.state import StateEstimate, estimate_state
from danon.detect.detector import (
    DetectionResult,
    DetectionStatus,
    DetectorConfig,
    detect,
)
from danon.detect.evaluate import (
    DetectionEvaluation,
    evaluate_detections,
    FORMAT_RATES,
)
from danon.detect.benchmark import (
    BenchmarkResults,
    RunRecord,
    run_benchmark,
)
from danon.detect.report import (
    BenchmarkReport,
    ReportBuilder,
    write_report,
)

__all__ = [
    "ETMeasurement",
    "ETValues",
    "GroundTruth",
    "GenomicsSchema",
    "ObservationStream",
    "Provenance",
    "Scenario",
    "ScenarioCase",
    "ScenarioKind",
    "Units",
    "ground_truth_summary",
    "ET_GROWTH_MODEL",
    "GROWTH_CHANGE_DAY_DEFAULT",
    "GROWTH_CHANGE_DELTA_DEFAULT",
    "NoiseModel",
    "ScenarioConfig",
    "SyntheticPopulation",
    "generate_scenario",
    "StateEstimate",
    "estimate_state",
    "DetectionResult",
    "DetectionStatus",
    "DetectorConfig",
    "detect",
    "DetectionEvaluation",
    "evaluate_detections",
    "FORMAT_RATES",
    "BenchmarkResults",
    "RunRecord",
    "run_benchmark",
    "BenchmarkReport",
    "ReportBuilder",
    "write_report",
]
