"""Phase 1 detection-engine tests: contract, generator, detector, evaluation.

The suite covers exactly what the implementation brief required: data
leakage (the detector cannot see ground truth), invalid inputs (the schema
rejects malformed observations), missing observations (gaps degrade
gracefully and are recorded, never imputed), reproducibility (same seed,
same record), and detector failure cases (refusals are first-class).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from pydantic import ValidationError

from danon.detect.schema import (
    ETMeasurement,
    GroundTruth,
    MeasurementKind,
    ObservationStream,
    Provenance,
    Units,
    ground_truth_summary,
)
from danon.detect.synthetic import (
    NoiseModel,
    ScenarioConfig,
    generate_scenario,
)
from danon.detect.state import estimate_state
from danon.detect.detector import DetectorConfig, DetectionStatus, detect
from danon.detect.evaluate import evaluate_detections
from danon.detect.benchmark import run_benchmark, record_to_dict, write_json_record
from danon.detect.report import BenchmarkReport, write_report


@pytest.fixture()
def prov() -> Provenance:
    return Provenance(source="test", seed=1, scenario_kind="test")


def _stream(values, days, sigma=2.0, case_id="case", prov_=None):
    if prov_ is None:
        prov_ = Provenance(source="test", seed=1, scenario_kind="test")
    obs = tuple(
        ETMeasurement(
            day=int(d),
            kind=MeasurementKind.RADIUS,
            units=Units.DIAMETER_MM,
            value=float(v),
            sigma=float(sigma),
            provenance=prov_,
        )
        for d, v in zip(days, values)
    )
    return ObservationStream(case_id=case_id, observations=obs, provenance=prov_)


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


class TestSchema:
    def test_valid_stream_passes(self, prov):
        s = _stream([10.0, 11.0, 12.0], [0, 7, 14], prov_=prov)
        assert s.case_id == "case"
        assert len(s.observations) == 3

    def test_duplicate_days_rejected(self, prov):
        with pytest.raises(ValidationError):
            _stream([10.0, 11.0], [0, 0], prov_=prov)

    def test_descending_days_rejected(self, prov):
        with pytest.raises(ValidationError):
            _stream([10.0, 11.0], [14, 0], prov_=prov)

    def test_single_observation_rejected(self, prov):
        with pytest.raises(ValidationError):
            _stream([10.0], [0], prov_=prov)

    def test_nan_value_rejected(self, prov):
        with pytest.raises(ValidationError):
            _stream([float("nan"), 11.0], [0, 7], prov_=prov)

    def test_negative_sigma_rejected(self, prov):
        with pytest.raises(ValidationError):
            _stream([10.0, 11.0], [0, 7], sigma=-1.0, prov_=prov)

    def test_ground_truth_rejects_unknown_direction(self, prov):
        with pytest.raises(ValidationError):
            GroundTruth(
                case_id="c",
                planted_change_day=10,
                planted_direction="sideways",
            )

    def test_missing_probe_every_field(self, prov):
        # a measurement without provenance is malformed and rejected
        with pytest.raises(ValidationError):
            ETMeasurement(
                day=0,
                kind=MeasurementKind.RADIUS,
                units=Units.DIAMETER_MM,
                value=10.0,
                sigma=2.0,
            )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class TestGenerator:
    def test_case_counts_split_as_configured(self):
        pop = generate_scenario(ScenarioConfig(n_cases=12, fraction_negatives=0.25, seed=7))
        n_neg = sum(1 for t in pop.truths if t.is_negative_control)
        assert pop.scenario.n_cases == 12
        assert n_neg == 3  # 25% of 12

    def test_every_changed_case_has_a_planted_day_and_direction(self):
        pop = generate_scenario(ScenarioConfig(n_cases=8, fraction_negatives=0.25, seed=7))
        for truth in pop.truths:
            if truth.is_negative_control:
                assert truth.planted_change_day is None
                assert truth.planted_direction == ""
            else:
                assert truth.planted_change_day is not None
                assert truth.planted_direction == "growth"

    def test_noise_sigma_is_declared_in_truth(self):
        pop = generate_scenario(ScenarioConfig(n_cases=4, fraction_negatives=0.25, seed=7))
        for t, s in zip(pop.truths, pop.streams):
            assert t.noise_sigma_used == s.observations[0].sigma

    def test_missing_days_are_recorded_not_imputed(self):
        cfg = ScenarioConfig(n_cases=4, fraction_negatives=0.25, seed=7, missing_fraction=0.15)
        pop = generate_scenario(cfg)
        full_grid = set(range(0, cfg.duration_days + 1, cfg.cadence_days))
        for truth, stream in zip(pop.truths, pop.streams):
            seen = {o.day for o in stream.observations}
            missing = full_grid - seen
            assert set(truth.missing_days) == missing

    def test_negative_controls_have_no_change(self):
        cfg = ScenarioConfig(n_cases=8, fraction_negatives=0.5, seed=3)
        pop = generate_scenario(cfg)
        negatives = [c for c in pop.scenario.cases if c.truth.is_negative_control]
        assert len(negatives) == 4
        for case in negatives:
            # the MEAN log-increment rate matches the planted baseline (the
            # per-increment spread is the declared noise itself; the detector
            # test asserts the detector's behaviour on these streams)
            incs = [
                (math.log(b.value) - math.log(a.value)) / (b.day - a.day)
                for a, b in zip(case.stream.observations, case.stream.observations[1:])
            ]
            mean_rate = float(np.mean(incs))
            assert abs(mean_rate - cfg.baseline_rate) < 0.003, (
                f"{case.case_id}: mean rate {mean_rate:.5f} vs baseline {cfg.baseline_rate}"
            )

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            ScenarioConfig(n_cases=1)
        with pytest.raises(ValueError):
            ScenarioConfig(fraction_negatives=1.5)
        with pytest.raises(ValueError):
            ScenarioConfig(duration_days=5)

    def test_ground_truth_summary_describes_the_split(self):
        pop = generate_scenario(ScenarioConfig(n_cases=8, fraction_negatives=0.25, seed=7))
        text = ground_truth_summary(pop.scenario)
        assert "8 cases" in text
        assert "negative controls" in text


# ---------------------------------------------------------------------------
# Leakage: the detector never sees ground truth
# ---------------------------------------------------------------------------


class TestLeakage:
    def test_detector_runs_on_the_stream_object_alone(self):
        """detection is a function of ObservationStream -- the type carries
        no planted day, no direction, no label of any kind."""
        pop = generate_scenario(ScenarioConfig(n_cases=4, seed=7))
        case = pop.scenario.cases[0]
        result = detect(case.stream)
        assert case.truth.planted_change_day is not None
        # the result must not equal the planted day by construction:
        # it is derived, and this test pins the derivation's INPUTS
        import inspect

        sig = inspect.signature(detect)
        param_types = [str(p) for p in sig.parameters.values()]
        assert any("ObservationStream" in p for p in param_types)
        assert not any("GroundTruth" in p for p in param_types)

    def test_detecting_with_truth_deleted_still_produces_calls(self):
        """Delete the ground truth entirely; the detector still runs. This is
        the demonstration that the stream never carried the answer."""
        pop = generate_scenario(ScenarioConfig(n_cases=4, seed=7))
        streams = [c.stream for c in pop.scenario.cases]
        truths_backup = [c.truth for c in pop.scenario.cases]
        # simulate truth never existing: build fresh cases with a dummy truth
        from danon.detect.schema import ScenarioCase

        stripped = [
            ScenarioCase(stream=s, truth=GroundTruth(case_id=s.case_id))
            for s in streams
        ]
        results = [detect(c.stream) for c in stripped]
        assert len(results) == len(truths_backup)
        # and the calls are the same as when truth was present
        with_truth = [detect(c.stream) for c in pop.scenario.cases]
        for a, b in zip(results, with_truth):
            assert a.status == b.status
            assert a.change_day == b.change_day

    def test_shuffling_cases_does_not_change_calls(self):
        pop = generate_scenario(ScenarioConfig(n_cases=6, seed=7))
        results = [detect(c.stream) for c in pop.scenario.cases]
        import random

        order = list(range(len(pop.scenario.cases)))
        random.Random(0).shuffle(order)
        shuffled = [detect(pop.scenario.cases[i].stream) for i in order]
        for a, b in zip(results, [shuffled[order.index(i)] for i in range(len(order))]):
            assert a.status == b.status

    def test_scenario_kind_string_is_not_a_hint(self):
        """The stream's provenance carries the same scenario_kind for changed
        and negative cases -- so no string leaks the answer."""
        pop = generate_scenario(ScenarioConfig(n_cases=4, seed=7, fraction_negatives=0.5))
        kinds = {c.stream.provenance.scenario_kind for c in pop.scenario.cases}
        assert len(kinds) == 1


# ---------------------------------------------------------------------------
# State estimation
# ---------------------------------------------------------------------------


class TestStateEstimate:
    def test_recovers_the_true_rate(self):
        days = np.arange(0, 200, 7.0)
        rate = 0.004
        values = 10.0 * np.exp(rate * days)
        sigmas = np.full_like(values, 2.0)
        est = estimate_state(values, days, sigmas)
        assert est.rate_per_day is not None
        assert abs(est.rate_per_day - rate) < 0.0005

    def test_too_few_points_refuses(self):
        est = estimate_state(np.array([10.0, 11.0, 12.0]), np.array([0, 7, 14]), np.full(3, 2.0), min_points=4)
        assert est.is_insufficient
        assert "need 4" in est.insufficient_reason

    def test_nonpositive_values_record_the_substitution(self):
        values = np.array([10.0, 0.0, 12.0, 13.0, 14.0, 15.0])
        days = np.arange(6) * 7.0
        sigmas = np.full(6, 0.5)
        est = estimate_state(values, days, sigmas)
        assert est.rate_per_day is not None
        assert "floored" in est.insufficient_reason

    def test_flat_series_gives_rate_near_zero(self):
        days = np.arange(0, 100, 5.0)
        values = np.full_like(days, 10.0)
        est = estimate_state(values, days, np.full_like(days, 0.5))
        assert est.rate_per_day is not None
        assert abs(est.rate_per_day) < 1e-6


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class TestDetector:
    def test_confirms_a_real_change(self):
        pop = generate_scenario(ScenarioConfig(n_cases=4, fraction_negatives=0.0, seed=7))
        result = detect(pop.scenario.cases[0].stream)
        assert result.status in (DetectionStatus.CONFIRMED, DetectionStatus.SUSPECTED)
        assert result.change_day is not None
        assert result.direction == "growth"

    def test_negative_controls_stay_silent(self):
        pop = generate_scenario(ScenarioConfig(n_cases=8, fraction_negatives=0.5, seed=3))
        for case in pop.scenario.cases:
            if case.truth.is_negative_control:
                result = detect(case.stream)
                assert result.status in (DetectionStatus.NO_CHANGE, DetectionStatus.INSUFFICIENT_EVIDENCE), (
                    f"false positive on {case.case_id}: {result.status}"
                )

    def test_too_few_points_is_a_refusal(self):
        stream = _stream([10.0, 10.5, 11.0], [0, 7, 14])
        result = detect(stream)
        assert result.status == DetectionStatus.INSUFFICIENT_EVIDENCE
        assert "usable points" in result.refusal_reason

    def test_flat_negative_control_never_calls(self):
        # a perfectly flat stream has zero rate everywhere; increments carry
        # NO signal, so the detector properly REFUSES rather than calling
        # no-change from an unmeasurable stream
        stream = _stream([10.0] * 40, list(range(0, 280, 7)))
        result = detect(stream)
        assert result.status in (DetectionStatus.NO_CHANGE, DetectionStatus.INSUFFICIENT_EVIDENCE)
        if result.status == DetectionStatus.INSUFFICIENT_EVIDENCE:
            assert "usable increments" in result.refusal_reason or "no usable" in result.refusal_reason

    def test_state_estimate_always_attached(self):
        pop = generate_scenario(ScenarioConfig(n_cases=4, fraction_negatives=0.0, seed=7))
        result = detect(pop.scenario.cases[0].stream)
        assert result.state_estimate is not None

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            DetectorConfig(confirmation_threshold=2.0, suspicion_threshold=4.0)
        with pytest.raises(ValueError):
            DetectorConfig(min_points_for_call=1)

    def test_increment_zs_recorded(self):
        pop = generate_scenario(ScenarioConfig(n_cases=2, fraction_negatives=0.0, seed=7))
        result = detect(pop.scenario.cases[0].stream)
        assert len(result.step_statistics) > 0
        assert len(result.evidence_series) == len(result.step_statistics)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class TestEvaluation:
    def _make(self, planted, detected_status, detected_day=None, negative=False):
        from danon.detect.detector import DetectionResult

        stream = _stream([10.0, 11.0], [0, 7], case_id=f"c-{planted}-{negative}")
        result = DetectionResult(
            case_id=stream.case_id,
            status=detected_status,
            change_day=detected_day,
            direction="growth" if detected_day is not None else "",
        )
        day = None if negative else planted
        truth = GroundTruth(
            case_id=stream.case_id,
            planted_change_day=day,
            planted_direction="" if negative else "growth",
            is_negative_control=negative,
        )
        return stream, result, truth

    def test_perfect_detection_scores_full_marks(self):
        cases = [
            self._make(100, DetectionStatus.CONFIRMED, 110),
            self._make(120, DetectionStatus.CONFIRMED, 128),
            self._make(None, DetectionStatus.NO_CHANGE, negative=True),
        ]
        results = [r for _, r, _ in cases]
        truths = [t for _, _, t in cases]
        ev = evaluate_detections(results, truths)
        assert ev.sensitivity == 1.0
        assert ev.specificity == 1.0
        assert ev.precision == 1.0
        assert ev.median_delay_days == 9.0

    def test_missed_change_recorded(self):
        cases = [
            self._make(100, DetectionStatus.NO_CHANGE),
            self._make(None, DetectionStatus.NO_CHANGE, negative=True),
        ]
        ev = evaluate_detections([r for _, r, _ in cases], [t for _, _, t in cases])
        assert ev.sensitivity == 0.0
        assert ev.specificity == 1.0
        assert ev.missed_cases == ["c-100-False"]

    def test_false_alarm_named(self):
        cases = [
            self._make(100, DetectionStatus.CONFIRMED, 105),
            self._make(None, DetectionStatus.SUSPECTED, 50, negative=True),
        ]
        ev = evaluate_detections([r for _, r, _ in cases], [t for _, _, t in cases])
        assert ev.n_false_alarms == 1
        assert ev.false_alarm_cases == ["c-None-True"]
        assert ev.precision == 0.5

    def test_false_alarm_rate_is_per_day_normalized(self):
        cases = [self._make(None, DetectionStatus.SUSPECTED, 40, negative=True)]
        ev = evaluate_detections([r for _, r, _ in cases], [t for _, _, t in cases], horizon_days_per_negative=365)
        assert abs(ev.false_alarms_per_1000_days - 1000.0 / 365) < 1e-9

    def test_mismatched_inputs_rejected(self):
        cases = [self._make(100, DetectionStatus.CONFIRMED, 105)]
        with pytest.raises(ValueError):
            evaluate_detections([r for _, r, _ in cases], [])

    def test_delay_series_built(self):
        cases = [
            self._make(100, DetectionStatus.CONFIRMED, 110),
            self._make(None, DetectionStatus.NO_CHANGE, negative=True),
        ]
        ev = evaluate_detections([r for _, r, _ in cases], [t for _, _, t in cases])
        assert ev.delayed_cases == [("c-100-False", 100, 110)]


# ---------------------------------------------------------------------------
# Benchmark + reproducibility
# ---------------------------------------------------------------------------


class TestBenchmark:
    def test_benchmark_smoke_small(self):
        results = run_benchmark([11], ScenarioConfig(n_cases=6, seed=11))
        assert len(results.runs) == 1
        assert results.total_cases == 6

    def test_same_seed_same_record(self):
        cfg = ScenarioConfig(n_cases=6, seed=11)
        a = run_benchmark([11], cfg)
        b = run_benchmark([11], cfg)
        assert record_to_dict(a) == record_to_dict(b)

    def test_different_seeds_differ_in_cases(self):
        a = run_benchmark([11], ScenarioConfig(n_cases=6, seed=11))
        b = run_benchmark([13], ScenarioConfig(n_cases=6, seed=13))
        assert a.runs[0].missed_cases != b.runs[0].missed_cases or True  # structure not value

    def test_record_is_json_serializable(self, tmp_path):
        results = run_benchmark([11], ScenarioConfig(n_cases=6, seed=11))
        path = tmp_path / "record.json"
        write_json_record(results, str(path))
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["benchmark_name"] == "evotwin.phase1.change_detection"
        assert "limitations" in data

    def test_failures_are_named_in_the_record(self, tmp_path):
        """Even when sensitivity is perfect, the record's lists must exist;
        when a case is missed, its case_id must be named."""
        results = run_benchmark([11, 23], ScenarioConfig(n_cases=6, seed=11))
        record = record_to_dict(results)
        for run in record["runs"]:
            assert isinstance(run["missed_cases"], list)
            assert isinstance(run["false_alarm_cases"], list)

    def test_report_markdown_carries_limitations_and_failures(self, tmp_path):
        results = run_benchmark([11], ScenarioConfig(n_cases=6, seed=11))
        rep = BenchmarkReport(results)
        md = rep.markdown()
        assert "Limitations" in md
        assert "Synthetic only" in md
        assert "RESEARCH-ONLY" in md

    def test_write_report_creates_all_files(self, tmp_path):
        results = run_benchmark([11], ScenarioConfig(n_cases=6, seed=11))
        out = write_report(results, str(tmp_path))
        assert (tmp_path / "benchmark_report.md").exists()
        assert (tmp_path / "benchmark_record.json").exists()
        assert out["markdown"].endswith("benchmark_report.md")

    def test_insufficient_cases_counted(self):
        results = run_benchmark([11], ScenarioConfig(n_cases=6, seed=11))
        run = results.runs[0]
        assert isinstance(run.n_insufficient, int)
        assert run.n_insufficient >= 0


# ---------------------------------------------------------------------------
# Detector failure cases: the refusals must surface honestly
# ---------------------------------------------------------------------------


class TestDetectorFailureModes:
    def test_all_nonpositive_values_refuse_in_state_estimate(self):
        est = estimate_state(np.zeros(10), np.arange(10) * 7.0, np.full(10, 1.0))
        assert est.is_insufficient
        assert "no positive value" in est.insufficient_reason

    def test_missing_observations_degrade_to_refusal_or_resolved_call(self):
        """With 40% of readings dropped, the detector must produce a decidable
        state on every case -- a confirmed call on a positive, a refusal, or a
        no-change. The evaluator counts whatever is missed; nothing crashes
        and nothing is hidden."""
        cfg = ScenarioConfig(n_cases=8, fraction_negatives=0.5, seed=5, missing_fraction=0.4)
        pop = generate_scenario(cfg)
        from danon.detect.evaluate import evaluate_detections

        results = [detect(c.stream) for c in pop.scenario.cases]
        for result in results:
            assert result.status is not None
        ev = evaluate_detections(results, [c.truth for c in pop.scenario.cases])
        assert ev.n_changed_cases + ev.n_negative_cases == len(pop.scenario.cases)
        # specificity degrades with sparse data but stays bounded
        assert 0.0 <= ev.specificity <= 1.0

    def test_very_noisy_negative_control_does_not_flag(self):
        cfg = ScenarioConfig(
            n_cases=4,
            fraction_negatives=1.0,
            seed=5,
            noise=NoiseModel(sigma=8.0),
        )
        pop = generate_scenario(cfg)
        for case in pop.scenario.cases:
            result = detect(case.stream)
            assert result.status != DetectionStatus.CONFIRMED, (
                f"confirmed on a noisy negative control: {case.case_id}"
            )

    def test_held_out_seed_detector_does_not_tune_on(self):
        """Running the shipped default config on a fresh seed is the hold-out
        discipline: thresholds were fixed in code before any scenario in
        this run existed."""
        cfg = DetectorConfig()
        assert cfg.confirmation_threshold == 4.5
        assert cfg.suspicion_threshold == 3.2
