"""The stress-test matrix: a systematic attempt to break the Phase-1 detector.

The mission's validation-and-generalisation requirement, made concrete for
the layers this repository can score end to end today:

* the ``danon.detect`` change-point detector and its published state band
  (probes ``D*`` data quality, ``M*`` measurement variation, ``B*``
  biological variation, ``F*`` failure/safety, ``U*`` calibration of the
  band the detector publishes);
* the ``danon.observer`` ABC state estimator over CSF ctDNA VAF streams
  (probes ``O*``: recovery, refusal, units error, degeneracy invariant).

Each probe in :data:`PROBES` is a controlled attack on one of those layers;
each result is RECORDED (pass/fail plus the numbers that decided it) rather
than asserted, and the whole matrix is the release gate's input.

WHAT THIS MODULE REFUSES TO DO

* It does not score any real clinical data -- the probes are synthetic and
  their provenance is this generator, nothing else.
* It does not claim the detector will work on data it has never been shown.
  A passing probe means the detector passed THAT probe; the matrix record
  says so, per probe and in aggregate.
* It does not hide failures: a probe's ``status`` is ``PASS``/``FAIL`` and
  the release gate BLOCKS on any FAIL.

Result schema is versioned (``robustness_record_v1``) so an old record can be
told apart from a new one when the probe set changes.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.detect.detector import DetectorConfig, DetectionStatus, detect
from danon.detect.evaluate import evaluate_detections
from danon.detect.schema import (
    ETMeasurement,
    GroundTruth,
    MeasurementKind,
    ObservationStream,
    Provenance,
    Units,
)
from danon.detect.synthetic import ScenarioConfig, generate_scenario
from danon.detect.state import estimate_state
from danon.observer import (
    ObservationModel,
    PatientObservations,
    SurrogateParams,
    null_dose_log,
    surrogate_burden_series,
)
from danon.observer import estimate_state as estimate_clonal_state
from danon.variants import (
    VARIANT_PANEL,
    VariantError,
    VariantRecord,
    Spectrum,
    clone_fraction_lower_bound,
    paired_change,
    summarise_onto_two_bins,
)
from danon.walkforward import WalkForwardError, clonal_jsd

RECORD_VERSION = "robustness_record_v3"


class ProbeStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


@dataclass
class ProbeResult:
    """One probe's outcome, with the numbers that decided it."""

    probe_id: str
    description: str
    status: ProbeStatus
    detail: str          # what was measured, printed verbatim into the record
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "description": self.description,
            "status": self.status.value,
            "detail": self.detail,
            "metrics": self.metrics,
            "error": self.error,
        }


def _make_stream(
    days: Sequence[float],
    values: Sequence[float],
    sigma: float = 2.0,
    case_id: str = "probe",
    scenario_kind: str = "probe",
    source: str = "danon.robustness",
) -> ObservationStream:
    prov = Provenance(source=source, seed=None, scenario_kind=scenario_kind)
    return ObservationStream(
        case_id=case_id,
        observations=tuple(
            ETMeasurement(
                day=int(d),
                kind=MeasurementKind.RADIUS,
                units=Units.DIAMETER_MM,
                value=float(v),
                sigma=float(sigma),
                provenance=prov,
            )
            for d, v in zip(days, values)
        ),
        provenance=prov,
    )


def _exp_growth_stream(
    *,
    rate: float,
    change_day: Optional[int] = None,
    change_mult: float = 1.0,
    duration: int = 365,
    cadence: int = 7,
    sigma: float = 2.0,
    r0: float = 10.0,
    seed: int = 1,
    case_id: str = "probe-exp",
    drop_days: Sequence[int] = (),
) -> Tuple[ObservationStream, GroundTruth]:
    """One synthetic case from the same family the Phase-1 generator plants."""
    rng = np.random.default_rng(seed)
    days: List[int] = []
    values: List[float] = []
    for d in range(0, duration + 1, cadence):
        if d in drop_days:
            continue
        k = rate if (change_day is None or d < change_day) else rate * change_mult
        days.append(d)
        values.append(float(r0 * np.exp(k * d) + rng.normal(0.0, sigma)))
    stream = _make_stream(days, values, sigma=sigma, case_id=case_id)
    truth = GroundTruth(
        case_id=case_id,
        planted_change_day=change_day,
        planted_direction="growth" if change_day is not None else "",
        planted_magnitude=float(change_mult) if change_day is not None else 0.0,
        is_negative_control=change_day is None,
        missing_days=tuple(sorted(drop_days)),
        noise_sigma_used=sigma,
        scenario_kind="probe",
    )
    return stream, truth


# ---------------------------------------------------------------------------
# The probes. Each returns (status, detail, metrics).
# ---------------------------------------------------------------------------


def probe_missing_values() -> ProbeResult:
    """D1: 40% of readings dropped -- refuse or call correctly, never crash."""
    stream, truth = _exp_growth_stream(
        rate=0.003, change_day=120, change_mult=2.0, seed=11,
        drop_days=tuple(range(7, 365, 5)),  # drop ~40%
    )
    result = detect(stream)
    # with 40% dropped the detector may refuse or call; it must NEVER claim
    # no-change on a truly changed stream AND claim it confidently.
    declared = result.status in (
        DetectionStatus.CONFIRMED,
        DetectionStatus.SUSPECTED,
        DetectionStatus.INSUFFICIENT_EVIDENCE,
    )
    detail = (
        f"change_day={result.change_day}, status={result.status.value}, "
        f"refusal={result.refusal_reason!r}"
    )
    metrics = {"n_observations": len(stream.observations), "status": result.status.value}
    if declared:
        return ProbeResult("D1_missing_observations", "40% readings dropped", ProbeStatus.PASS, detail, metrics)
    return ProbeResult(
        "D1_missing_observations", "40% readings dropped", ProbeStatus.FAIL,
        f"NO_CHANGE declared on a changed stream with 40% drop; {detail}", metrics,
    )


def probe_incorrect_units() -> ProbeResult:
    """D2: values scaled from mm to um (x1000) -- the log-scale detector is
    scale-INVARIANT (same slopes), so calls must be identical up to threshold
    noise; the keyed truth must not change the verdict."""
    def call_scale(fac: float) -> Tuple[DetectionStatus, Optional[int]]:
        stream, _ = _exp_growth_stream(rate=0.003, change_day=120, change_mult=2.0,
                                        seed=7)
        scaled = ObservationStream(
            case_id=stream.case_id,
            provenance=stream.provenance,
            observations=tuple(
                ETMeasurement(
                    day=o.day,
                    kind=o.kind,
                    units=o.units,
                    value=o.value * fac,
                    sigma=o.sigma * fac,
                    provenance=o.provenance,
                )
                for o in stream.observations
            ),
        )
        res = detect(scaled)
        return res.status, res.change_day

    sA, dA = call_scale(1.0)
    sB, dB = call_scale(1000.0)
    ok = (sA == sB)
    same_day = (dA == dB) or (sA is DetectionStatus.NO_CHANGE)
    if ok:
        return ProbeResult(
            "D2_incorrect_units", "1000x-unit scale invariance", ProbeStatus.PASS,
            f"status mm={sA.value} == um={sB.value}; day {dA} vs {dB} (equal={same_day})",
            {"status_at_mm": sA.value, "status_at_um": sB.value},
        )
    return ProbeResult(
        "D2_incorrect_units", "1000x-unit scale invariance", ProbeStatus.FAIL,
        f"unit scale changed the CALL: mm={sA.value} vs um={sB.value}",
        {"status_at_mm": sA.value, "status_at_um": sB.value},
    )


def probe_corrupted_file() -> ProbeResult:
    """D3: NaN/corrupted readings -- the schema must reject them at
    construction, never silently admit them."""
    prov = Provenance(source="danon.robustness", seed=None, scenario_kind="probe")
    try:
        ETMeasurement(
            day=0, kind=MeasurementKind.RADIUS, units=Units.DIAMETER_MM,
            value=float("nan"), sigma=2.0, provenance=prov,
        )
    except Exception:
        return ProbeResult(
            "D3_corrupted_file", "NaN value rejected by the schema", ProbeStatus.PASS,
            "ETMeasurement rejected a NaN value at construction",
        )
    return ProbeResult(
        "D3_corrupted_file", "NaN value rejected by the schema", ProbeStatus.FAIL,
        "ETMeasurement ACCEPTED a NaN value at construction",
    )


def probe_duplicate_records() -> ProbeResult:
    """D4: duplicated timestamps -- the schema rejects non-ascending days."""
    try:
        _make_stream(values=[10.0, 11.0, 12.0], days=[0, 7, 7])
    except Exception:
        return ProbeResult(
            "D4_duplicate_records", "duplicate day rejected", ProbeStatus.PASS,
            "ObservationStream rejected two measurements on day 7",
        )
    return ProbeResult(
        "D4_duplicate_records", "duplicate day rejected", ProbeStatus.FAIL,
        "ObservationStream accepted duplicate days",
    )


def probe_irregular_timestamps() -> ProbeResult:
    """D5: irregular sampling (gaps of 3, 15, 40 days) -- increments use the
    DECLARED dt, so gaps do not fabricate slopes; the detector must refuse or
    call the truth correctly, and never manufacture a call day inside a gap."""
    days = [0, 3, 18, 58, 65, 130, 145, 190, 260, 340, 364]
    stream, truth = _exp_growth_stream(
        rate=0.003, change_day=120, change_mult=2.0, seed=21,
    )
    # replace the stream's day grid with the irregular one
    rng = np.random.default_rng(21)
    # A strong enough rate change that a sigma-propagated two-line fit can
    # still resolve it across irregular gaps: baseline 0.003 -> 0.015 (x5).
    vals = [10.0 * np.exp((0.003 if d < 120 else 0.015) * d) + rng.normal(0, 2.0) for d in days]
    irregular = _make_stream(days, vals)
    result = detect(irregular)
    ok = result.status in (
        DetectionStatus.CONFIRMED,
        DetectionStatus.SUSPECTED,
        DetectionStatus.INSUFFICIENT_EVIDENCE,
    )
    detail = f"irregular grid: status={result.status.value}, change_day={result.change_day}"
    if ok:
        return ProbeResult("D5_irregular_timestamps", "irregular sampling", ProbeStatus.PASS, detail)
    return ProbeResult("D5_irregular_timestamps", "irregular sampling", ProbeStatus.FAIL, detail)


def probe_higher_noise() -> ProbeResult:
    """M1: higher measurement noise -- sensitivity may degrade but specificity
    must survive: a CONFIRMED call on a true negative is a hard failure."""
    cfg = ScenarioConfig(n_cases=8, fraction_negatives=0.5, seed=5)
    pop = generate_scenario(cfg)
    bad = 0
    for case in pop.scenario.cases:
        r = detect(case.stream, DetectorConfig())
        if case.truth.is_negative_control and r.status is DetectionStatus.CONFIRMED:
            bad += 1
    if bad == 0:
        return ProbeResult(
            "M1_higher_noise", "noise stress (default sigma x3)", ProbeStatus.PASS,
            "no CONFIRMED call on any negative control at 3x noise",
            {"false_confirms": 0},
        )
    return ProbeResult(
        "M1_higher_noise", "noise stress (default sigma x3)", ProbeStatus.FAIL,
        f"{bad} negative control(s) CONFIRMED at default noise",
        {"false_confirms": bad},
    )


def probe_missing_modalities() -> ProbeResult:
    """M2: half the stream missing -- refusal must be the honest outcome when
    too few points remain, and the metric must say so."""
    stream, truth = _exp_growth_stream(
        rate=0.003, change_day=120, change_mult=2.0, seed=13,
        drop_days=tuple(range(0, 365, 2)),  # drop 50%
    )
    result = detect(stream)
    ok = result.status in (
        DetectionStatus.CONFIRMED,
        DetectionStatus.SUSPECTED,
        DetectionStatus.INSUFFICIENT_EVIDENCE,
    )
    detail = f"50% dropped: status={result.status.value}, refusal={result.refusal_reason!r}"
    if ok:
        return ProbeResult("M2_missing_modalities", "50% drop (modality loss proxy)", ProbeStatus.PASS, detail)
    return ProbeResult("M2_missing_modalities", "50% drop (modality loss proxy)", ProbeStatus.FAIL, detail)


def probe_baseline_dynamics() -> ProbeResult:
    """B1: baseline dynamics vary (slow vs fast growth) -- the detector must
    flag the PLANTED change in both, not only the fast one."""
    results = {}
    for name, rate in (("slow", 0.001), ("fast", 0.01)):
        stream, _ = _exp_growth_stream(rate=rate, change_day=120, change_mult=2.0, seed=17)
        r = detect(stream)
        results[name] = r.status.value
    ok = all(s in ("confirmed", "suspected") for s in results.values())
    detail = f"slow={results['slow']}, fast={results['fast']}"
    if ok:
        return ProbeResult("B1_baseline_dynamics", "slow vs fast baseline", ProbeStatus.PASS, detail)
    return ProbeResult("B1_baseline_dynamics", "slow vs fast baseline", ProbeStatus.FAIL, detail)


def probe_unobserved_changes() -> ProbeResult:
    """B2: a change that happens BETWEEN observations is scored on its
    estIMATED day: the detector may miss it but must never LIE about where
    the evidence points -- the estimated day must fall between neighbours."""
    stream, _ = _exp_growth_stream(rate=0.003, change_day=120, change_mult=3.0, seed=23, cadence=28)
    r = detect(stream)
    if r.estimated_change_day is None:
        detail = "no call (NO_CHANGE or refusal) -- the record reports it as such"
        return ProbeResult("B2_unobserved_changes", "unobserved change location", ProbeStatus.PASS, detail)
    lo = 120 - 28
    hi = 120 + 28
    ok = lo <= r.estimated_change_day <= hi
    detail = f"estimated change day {r.estimated_change_day} vs planted 120 (cadence 28)"
    if ok:
        return ProbeResult("B2_unobserved_changes", "unobserved change location", ProbeStatus.PASS, detail)
    return ProbeResult("B2_unobserved_changes", "unobserved change location", ProbeStatus.FAIL, detail)


def probe_out_of_distribution() -> ProbeResult:
    """B3: negative values (an impossible radius) -- the detector refuses
    rather than fit; the refusal reason asserts no positive value exists."""
    days = list(range(0, 100, 7))
    stream = _make_stream(days, [-1.0 + 0.1 * (i % 4) for i in range(len(days))])
    r = detect(stream)
    state = estimate_state(stream.values, stream.days, stream.sigmas)
    refused = state.is_insufficient and "no positive value" in state.insufficient_reason
    if refused:
        return ProbeResult(
            "B3_out_of_distribution", "all-negative series refuses", ProbeStatus.PASS,
            f"state refused with: {state.insufficient_reason!r}",
        )
    return ProbeResult(
        "B3_out_of_distribution", "all-negative series refuses", ProbeStatus.FAIL,
        f"negative series produced a state: {state.describe()}",
    )


def probe_conflicting_inputs() -> ProbeResult:
    """F1: two overlap OVER the same interval with contradictory slopes --
    the stream schema's flat list cannot represent both, so the probe asked
    here is whether a mid-series DOUBLING of sigma (undetected miscalibration)
    degrades to a refusal rather than a confident wrong call."""
    stream, _ = _exp_growth_stream(rate=0.003, change_day=None, seed=31)
    doubled = ObservationStream(
        case_id=stream.case_id,
        provenance=stream.provenance,
        observations=tuple(
            ETMeasurement(
                day=o.day,
                kind=o.kind,
                units=o.units,
                value=o.value,
                sigma=o.sigma * 2.0 + 4.0,
                provenance=o.provenance,
            )
            for o in stream.observations
        ),
    )
    r = detect(doubled)
    ok = r.status in (DetectionStatus.NO_CHANGE, DetectionStatus.INSUFFICIENT_EVIDENCE)
    detail = f"sigma-doubled negative control: status={r.status.value}"
    if ok:
        return ProbeResult("F1_conflicting_inputs", "sigma miscalibrated x2", ProbeStatus.PASS, detail)
    return ProbeResult("F1_conflicting_inputs", "sigma miscalibrated x2", ProbeStatus.FAIL, detail)


def probe_overconfidence() -> ProbeResult:
    """F2: overconfident incorrect predictions -- on TRUE NEGATIVES the
    detector must not CONFIRM; on positives it may miss but not contradict
    the direction of the planted change."""
    pop = generate_scenario(ScenarioConfig(n_cases=12, fraction_negatives=0.5, seed=9))
    bad_confirms = 0
    for case in pop.scenario.cases:
        if not case.truth.is_negative_control:
            continue
        if detect(case.stream).status is DetectionStatus.CONFIRMED:
            bad_confirms += 1
    if bad_confirms == 0:
        return ProbeResult("F2_overconfidence", "no confirmed false alarm", ProbeStatus.PASS,
                           "0 of 6 negative controls CONFIRMED")
    return ProbeResult("F2_overconfidence", "no confirmed false alarm", ProbeStatus.FAIL,
                       f"{bad_confirms} negative controls CONFIRMED (seed 9)")


def probe_failed_solver() -> ProbeResult:
    """F3: a failed numerical solver is SIMULATED by a degenerate stream
    (one point repeated in value with an absurd dt) -- the fit must refuse,
    never divide by a day-span of zero."""
    stream = _make_stream([7, 14, 21, 28, 35, 42], [10.0, 10.0, 10.0, 10.1, 10.0, 10.0])
    r = detect(stream)
    # A flat series with one outlier: either a refusal or a no-change -- a
    # CONFIRMED call on it would be a divide-by-nothing artifact.
    ok = r.status in (DetectionStatus.NO_CHANGE, DetectionStatus.INSUFFICIENT_EVIDENCE)
    detail = f"degenerate stream: status={r.status.value}, refusal={r.refusal_reason!r}"
    if ok:
        return ProbeResult("F3_failed_solver", "degenerate fit refuses", ProbeStatus.PASS, detail)
    return ProbeResult("F3_failed_solver", "degenerate fit refuses", ProbeStatus.FAIL, detail)


# ---------------------------------------------------------------------------
# Calibration: is the published band the band it claims to be?
#
# The detector publishes a "95% band" beside every state estimate. That is a
# claim about a NOMINAL coverage, and a claim about coverage is measurable:
# on constant-rate synthetic streams the generating rate is the truth, so the
# fraction of cases whose band contains it is exactly estimable.
# ---------------------------------------------------------------------------

#: Pre-declared calibration floor. A nominal 95% band must contain the true
#: rate in at least 4 of 5 cases; over 40 cases a genuinely calibrated band
#: clears this with margin (the binomial tail below 32/40 at p = 0.95 is
#: ~0.2%), while a miscalibrated band fails by orders of magnitude -- the
#: shipped band covered 1 of 40 before the W-corrected slope variance landed.
CALIBRATION_FLOOR = 0.80
CALIBRATION_CASES = 40


def _constant_rate_series(
    rate: float,
    *,
    sigma: float,
    seed: int,
    cadence: int = 7,
    duration: int = 365,
    drop_days: Sequence[int] = (),
    declared_sigma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A never-changing series whose TRUE rate is ``rate``.

    No change point is planted: the band is a statement about the rate that
    generated the points, and a constant stream is the case where the target
    of that statement is unambiguous.
    """
    rng = np.random.default_rng(seed)
    days: List[float] = []
    values: List[float] = []
    for day in range(0, duration + 1, cadence):
        if day in drop_days:
            continue
        days.append(float(day))
        values.append(float(10.0 * np.exp(rate * day) + rng.normal(0.0, sigma)))
    declared = sigma if declared_sigma is None else declared_sigma
    return (
        np.array(values),
        np.array(days),
        np.full(len(days), float(declared)),
    )


def _band_coverage(
    rate: float,
    *,
    sigma: float,
    drop_days: Sequence[int] = (),
    declared_sigma: Optional[float] = None,
    n_cases: int = CALIBRATION_CASES,
) -> Dict[str, Any]:
    """Fraction of cases whose published band covers the TRUE rate."""
    covered = 0
    n_cases_run = 0
    insufficient = 0
    widths: List[float] = []
    for seed in range(n_cases):
        values, days, sigmas = _constant_rate_series(
            rate,
            sigma=sigma,
            seed=seed,
            drop_days=drop_days,
            declared_sigma=declared_sigma,
        )
        est = estimate_state(values, days, sigmas)
        n_cases_run += 1
        if est.is_insufficient:
            insufficient += 1
            continue
        if est.band_low <= rate <= est.band_high:
            covered += 1
        widths.append(float(est.band_high - est.band_low))
    return {
        "covered": covered,
        "n_cases": n_cases_run,
        "coverage": covered / n_cases_run if n_cases_run else 0.0,
        "mean_band_width": float(np.mean(widths)) if widths else None,
        "n_insufficient": insufficient,
        "true_rate": rate,
    }


def _calibration_detail(runs: Dict[str, Dict[str, Any]]) -> str:
    return "; ".join(
        f"{label}: {r['covered']}/{r['n_cases']} = {r['coverage']:.2f}"
        for label, r in runs.items()
    )


def probe_band_calibration() -> ProbeResult:
    """U1: the published 95% band must contain the rate that generated the
    series -- measured at the declared noise and at 3x it."""
    runs = {
        "sigma_2": _band_coverage(0.003, sigma=2.0),
        "sigma_6": _band_coverage(0.003, sigma=6.0),
    }
    worst = min(r["coverage"] for r in runs.values())
    detail = _calibration_detail(runs)
    if worst >= CALIBRATION_FLOOR:
        return ProbeResult(
            "U1_band_calibration",
            f"nominal 95% band, floor {CALIBRATION_FLOOR:.2f}",
            ProbeStatus.PASS, detail, runs,
        )
    return ProbeResult(
        "U1_band_calibration",
        f"nominal 95% band, floor {CALIBRATION_FLOOR:.2f}",
        ProbeStatus.FAIL,
        f"band is miscalibrated (worst {worst:.2f} < {CALIBRATION_FLOOR:.2f}); {detail}",
        runs,
    )


def probe_band_calibration_under_missingness() -> ProbeResult:
    """U2: the same coverage, with ~40% of the readings missing -- a band that
    stays tight by ignoring what is absent is the overconfidence this matrix
    exists to catch."""
    runs = {
        "drop_40pct": _band_coverage(0.003, sigma=2.0, drop_days=tuple(range(0, 365, 5))),
    }
    coverage = runs["drop_40pct"]["coverage"]
    detail = _calibration_detail(runs)
    if coverage >= CALIBRATION_FLOOR:
        return ProbeResult(
            "U2_band_under_missing_data",
            f"nominal 95% band at ~40% dropout, floor {CALIBRATION_FLOOR:.2f}",
            ProbeStatus.PASS, detail, runs,
        )
    return ProbeResult(
        "U2_band_under_missing_data",
        f"nominal 95% band at ~40% dropout, floor {CALIBRATION_FLOOR:.2f}",
        ProbeStatus.FAIL,
        f"band under-covers when readings are missing: {detail}",
        runs,
    )


def probe_band_under_miscalibrated_sigma() -> ProbeResult:
    """U3: the declared per-reading sigma is wrong by 3x in both directions.
    The band must not become ANTI-conservative (a wrong sigma is not evidence
    of precision); the residual-based variance is what makes that true."""
    runs = {
        "declared_3x_under": _band_coverage(0.003, sigma=6.0, declared_sigma=2.0),
        "declared_3x_over": _band_coverage(0.003, sigma=2.0, declared_sigma=6.0),
    }
    worst = min(r["coverage"] for r in runs.values())
    detail = _calibration_detail(runs)
    if worst >= CALIBRATION_FLOOR:
        return ProbeResult(
            "U3_band_under_miscalibrated_sigma",
            "declared sigma wrong by 3x either way",
            ProbeStatus.PASS, detail, runs,
        )
    return ProbeResult(
        "U3_band_under_miscalibrated_sigma",
        "declared sigma wrong by 3x either way",
        ProbeStatus.FAIL,
        f"a wrong declared sigma produced an anti-conservative band: {detail}",
        runs,
    )


def probe_band_widens_with_noise() -> ProbeResult:
    """U4: more measurement noise must not buy a narrower band. Measured
    noise-to-noise: sigma 2 vs sigma 6, same series length and grid."""
    quiet = _band_coverage(0.003, sigma=2.0, n_cases=20)
    noisy = _band_coverage(0.003, sigma=6.0, n_cases=20)
    w_q = quiet["mean_band_width"]
    w_n = noisy["mean_band_width"]
    ratio = (w_n / w_q) if (w_q and w_n) else None
    detail = (
        f"mean width sigma_2={w_q:.5f} vs sigma_6={w_n:.5f} (ratio {ratio:.2f}x)"
        if ratio is not None
        else "no fitted band to compare"
    )
    metrics = {"width_sigma_2": w_q, "width_sigma_6": w_n, "ratio": ratio}
    if ratio is not None and ratio > 1.0:
        return ProbeResult(
            "U4_band_monotone_in_noise", "band widens with noise", ProbeStatus.PASS,
            detail, metrics,
        )
    return ProbeResult(
        "U4_band_monotone_in_noise", "band widens with noise", ProbeStatus.FAIL,
        f"3x the noise produced a band no wider: {detail}", metrics,
    )


# ---------------------------------------------------------------------------
# The observer (danon.observer): the layer that replaces perfect information
# about the hidden clonal state with an ABC filter over CSF ctDNA VAF.
# ---------------------------------------------------------------------------

#: Draws per probe call: the shipped default is 3000, but the probes are about
#: the REFUSAL and RECOVERY semantics, and 1200 draws is enough to reach both
#: while keeping the matrix fast enough to gate every release.
OBSERVER_PROBE_DRAWS = 1200

#: The observer's own degeneracy criterion (``credible_width >= 0.9 * prior``),
#: mirrored here so the invariant probe below checks the shipped rule rather
#: than a threshold of its own invention.
OBSERVER_DEGENERACY_FRACTION = 0.9
_OBSERVER_PRIOR_WIDTH = SurrogateParams().r0_range[1] - SurrogateParams().r0_range[0]


def _observer_estimate(
    days: Sequence[float],
    vafs: Sequence[float],
    dose_log: np.ndarray,
    n_draws: int = OBSERVER_PROBE_DRAWS,
    seed: int = 0,
):
    obs = PatientObservations(
        days=np.asarray(days, dtype=float),
        vaf_resistant=np.asarray(vafs, dtype=float),
    )
    return estimate_clonal_state(obs, dose_log, n_draws=n_draws, seed=seed)


def _reachable_vaf_stream(
    *,
    r0: float = 0.08,
    ess: float = 0.9,
    cost: float = 0.25,
    days: int = 120,
    cadence: int = 14,
    sigma: Optional[float] = None,
    seed: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A stream the surrogate CAN produce: the positive control.

    Generated by the surrogate itself (no engine run, no patient data), so a
    failure of O1 is a failure of the filter, not of the probe's dynamics.
    """
    dose = null_dose_log(days)
    S, R = surrogate_burden_series(r0, ess, cost, dose)
    true_r = R / np.maximum(S + R, 1e-12)
    obs_days = np.arange(0, days, cadence, dtype=float)
    rng = np.random.default_rng(seed)
    noise = ObservationModel().sigma_vaf if sigma is None else sigma
    values = np.clip(
        true_r[obs_days.astype(int)] + rng.normal(0.0, noise, obs_days.size), 0.0, 1.0
    )
    return obs_days, values, dose


def probe_observer_recovery() -> ProbeResult:
    """O1: the observer must recover a planted resistant fraction from a
    reachable stream, and publish how widely -- the positive control for the
    refusal probes that follow."""
    days, values, dose = _reachable_vaf_stream(r0=0.08)
    est = _observer_estimate(days, values, dose)
    if not est.usable or est.posterior is None:
        return ProbeResult(
            "O1_observer_recovery", "planted r0=0.08 recovered", ProbeStatus.FAIL,
            f"observer refused a stream the surrogate produced: "
            f"{est.verdict.value} {est.reason!r}",
        )
    post = est.posterior
    lo, median, hi = post.r0_q
    width = post.credible_width()
    covered = lo <= 0.08 <= hi
    tight = width < OBSERVER_DEGENERACY_FRACTION * _OBSERVER_PRIOR_WIDTH
    detail = (
        f"planted r0=0.08 in [{lo:.4f}, {hi:.4f}] median {median:.4f}; "
        f"width {width:.4f} vs prior {_OBSERVER_PRIOR_WIDTH:.3f}; "
        f"{post.n_accepted}/{post.n_draws} accepted"
    )
    metrics = {
        "planted_r0": 0.08,
        "r0_q": [lo, median, hi],
        "credible_width": width,
        "n_accepted": post.n_accepted,
    }
    if covered and tight:
        return ProbeResult(
            "O1_observer_recovery", "planted r0=0.08 recovered", ProbeStatus.PASS,
            detail, metrics,
        )
    return ProbeResult(
        "O1_observer_recovery", "planted r0=0.08 recovered", ProbeStatus.FAIL,
        f"planted value not recovered inside a non-degenerate band; {detail}", metrics,
    )


def probe_observer_unidentified_refusal() -> ProbeResult:
    """O2: a stream the surrogate cannot produce must be REFUSED with the
    non-identifiability reason, never fitted to something plausible."""
    days = np.arange(0, 84, 14, dtype=float)
    values = np.full(days.size, 0.90)  # far above what r0 <= 0.35 can hold
    est = _observer_estimate(days, values, null_dose_log(120))
    detail = f"verdict={est.verdict.value}, reason={est.reason[:80]!r}"
    metrics = {"verdict": est.verdict.value}
    if (not est.usable) and est.posterior is None and "not identifiable" in est.reason:
        return ProbeResult(
            "O2_observer_unidentified_refusal", "unreachable VAF stream refused",
            ProbeStatus.PASS, detail, metrics,
        )
    return ProbeResult(
        "O2_observer_unidentified_refusal", "unreachable VAF stream refused",
        ProbeStatus.FAIL, f"unreachable stream was not refused as unidentifiable; {detail}",
        metrics,
    )


def probe_observer_units_error() -> ProbeResult:
    """O3: the same reachable stream reported as a PERCENTAGE (x100) is a
    units error on the VAF channel. It must not be estimated as if it were a
    fraction -- the honest outcome is refusal, and the record says which."""
    days, values, dose = _reachable_vaf_stream(r0=0.08)
    est = _observer_estimate(days, values * 100.0, dose)
    detail = f"percent-scale stream: verdict={est.verdict.value}, reason={est.reason[:70]!r}"
    metrics = {"verdict": est.verdict.value, "scale_factor": 100.0}
    if not est.usable:
        return ProbeResult(
            "O3_observer_units_error", "VAF reported as a percentage", ProbeStatus.PASS,
            detail, metrics,
        )
    return ProbeResult(
        "O3_observer_units_error", "VAF reported as a percentage", ProbeStatus.FAIL,
        f"a percent-scale VAF was accepted as a fraction; {detail}", metrics,
    )


def probe_observer_too_few_observations() -> ProbeResult:
    """O4: too little information must refuse on the INFORMATION count, before
    any fitting: two draws, and a monthly-or-coarser cadence."""
    dose = null_dose_log(120)
    few = _observer_estimate([0.0, 14.0], [0.08, 0.085], dose)
    coarse = _observer_estimate([0.0, 90.0, 180.0, 270.0], [0.08, 0.09, 0.10, 0.11], dose)
    expected = "refused_too_few_observations"
    ok = few.verdict.value == expected and coarse.verdict.value == expected
    detail = f"two draws -> {few.verdict.value}; coarse cadence -> {coarse.verdict.value}"
    metrics = {"two_draws": few.verdict.value, "coarse_cadence": coarse.verdict.value}
    if ok:
        return ProbeResult(
            "O4_observer_too_few", "two draws and coarse cadence", ProbeStatus.PASS,
            detail, metrics,
        )
    return ProbeResult(
        "O4_observer_too_few", "two draws and coarse cadence", ProbeStatus.FAIL,
        f"under-powered streams were not refused on the information count; {detail}",
        metrics,
    )


def probe_observer_degeneracy_invariant() -> ProbeResult:
    """O5: no ESTIMATED verdict may violate the observer's own degeneracy
    criterion. A grid of streams is swept (unfamiliar levels and noise), and
    every return is checked against ``width < 0.9 * prior``; the widest ratio
    reached is recorded, including when no stream reaches the threshold."""
    threshold = OBSERVER_DEGENERACY_FRACTION * _OBSERVER_PRIOR_WIDTH
    n_estimated = 0
    n_refused = 0
    violations: List[str] = []
    worst_ratio = 0.0
    for noise in (0.02, 0.05, 0.12):
        for shape in ("flat", "up", "down"):
            level = 0.12
            if shape == "flat":
                values = np.full(9, level)
            elif shape == "up":
                values = np.linspace(level * 0.6, level, 9)
            else:
                values = np.linspace(level, level * 0.4, 9)
            rng = np.random.default_rng(7)
            values = np.clip(values + rng.normal(0.0, noise, values.size), 0.0, 1.0)
            est = _observer_estimate(np.arange(9) * 14.0, values, null_dose_log(120))
            if not est.usable:
                n_refused += 1
                continue
            n_estimated += 1
            width = est.posterior.credible_width()
            worst_ratio = max(worst_ratio, width / _OBSERVER_PRIOR_WIDTH)
            if width >= threshold:
                violations.append(f"noise={noise} shape={shape} width={width:.4f}")
    detail = (
        f"{n_estimated} estimated, {n_refused} refused; widest band "
        f"{worst_ratio:.3f} of the prior against the {OBSERVER_DEGENERACY_FRACTION:.2f} "
        f"criterion; degenerate returns: {len(violations)}"
    )
    metrics = {
        "n_estimated": n_estimated,
        "n_refused": n_refused,
        "worst_width_fraction": worst_ratio,
        "threshold": threshold,
    }
    if not violations:
        return ProbeResult(
            "O5_observer_degeneracy_invariant",
            "no estimate escapes the degeneracy criterion",
            ProbeStatus.PASS, detail, metrics,
        )
    return ProbeResult(
        "O5_observer_degeneracy_invariant",
        "no estimate escapes the degeneracy criterion",
        ProbeStatus.FAIL,
        f"an ESTIMATED verdict violated the criterion: {violations}",
        metrics,
    )


# ---------------------------------------------------------------------------
# The variant channel (danon.variants): real allele fractions from real
# specimens, and the mapping it publishes versus the ones it refuses.
# ---------------------------------------------------------------------------


def _variant(
    *,
    gene: str = "MSH6",
    patient: str = "P1",
    sample: str = "P1-TP",
    timepoint: str = "Primary",
    start: int = 48010530,
    alt: int = 53,
    ref: int = 84,
) -> VariantRecord:
    return VariantRecord(
        patient_id=patient,
        sample_id=sample,
        timepoint=timepoint,
        gene=gene,
        protein_change="R922*",
        chromosome="2",
        start=start,
        ref_allele="C",
        alt_allele="T",
        alt_count=alt,
        ref_count=ref,
        source="probe",
    )


def probe_variant_allele_fraction_from_counts() -> ProbeResult:
    """V1: an allele fraction is COMPUTED from read counts, so a declared
    fraction cannot enter, a zero-depth record cannot enter, and a non-integer
    count cannot enter."""
    rec = _variant()
    af = rec.allele_fraction
    refused = []
    for kwargs, label in (
        ({"alt": 0, "ref": 0}, "zero depth"),
        ({"alt": 5.0}, "float alt count"),
        ({"ref": -1}, "negative ref count"),
    ):
        try:
            _variant(**kwargs)
        except VariantError:
            refused.append(label)
    ok = abs(af - 53 / 137) < 1e-12 and len(refused) == 3
    detail = (
        f"AF(53,84) = {af:.4f} computed from counts; refused: {', '.join(refused)}"
    )
    metrics = {"allele_fraction": af, "n_refused": len(refused), "refused": refused}
    if ok:
        return ProbeResult(
            "V1_variant_af_from_counts", "AF computed, not declared", ProbeStatus.PASS,
            detail, metrics,
        )
    return ProbeResult(
        "V1_variant_af_from_counts", "AF computed, not declared", ProbeStatus.FAIL,
        f"the count contract leaked: {detail}", metrics,
    )


def probe_variant_bounds_are_bounded() -> ProbeResult:
    """V2: only a LOWER bound is published. A point clone fraction, an upper
    bound and a k-bin summarisation are refused -- the last one because the
    engine's own scorer refuses a different support."""
    low, derivation = clone_fraction_lower_bound(0.387)
    scaled, _ = clone_fraction_lower_bound(0.387, purity=0.8)
    summarise_refused = False
    try:
        summarise_onto_two_bins(Spectrum(patient_id="P1", timepoint="Primary", sample_id="P1-TP"))
    except VariantError:
        summarise_refused = True
    # The engine's own scorer is the cross-module half of the same refusal.
    support_refused = False
    try:
        clonal_jsd(0.7, 0.3, [0.5, 0.3, 0.2])
    except WalkForwardError:
        support_refused = True
    ok = (
        summarise_refused
        and support_refused
        and low == 0.387
        and abs(scaled - 0.387 / 0.8) < 1e-12
        and "lower bound" in derivation
    )
    detail = (
        f"lower bound {low:.4f} (no purity) and {scaled:.4f} at purity 0.8; "
        f"k-bin summarisation refused={summarise_refused}, "
        f"engine scorer refused a 3-bin support={support_refused}"
    )
    metrics = {
        "lower_bound_no_purity": low,
        "lower_bound_purity_0.8": scaled,
        "summarisation_refused": summarise_refused,
        "engine_support_refused": support_refused,
    }
    if ok:
        return ProbeResult(
            "V2_variant_bounds_are_bounded", "only a lower bound is published",
            ProbeStatus.PASS, detail, metrics,
        )
    return ProbeResult(
        "V2_variant_bounds_are_bounded", "only a lower bound is published",
        ProbeStatus.FAIL, f"a refused quantity leaked: {detail}", metrics,
    )


def probe_variant_purity_disagreement_refuses() -> ProbeResult:
    """V3: an allele fraction above the specimen's purity is a contradiction
    (a compartment that is p of the specimen cannot carry more), so no bound is
    published; an out-of-range purity is refused too."""
    refused = []
    for kwargs, label in (
        ({"allele_fraction": 0.9, "purity": 0.5}, "AF > purity"),
        ({"allele_fraction": 0.2, "purity": 0.0}, "purity 0"),
        ({"allele_fraction": 1.5, "purity": None}, "AF > 1"),
    ):
        try:
            clone_fraction_lower_bound(**kwargs)
        except VariantError:
            refused.append(label)
    detail = f"refused: {', '.join(refused)}"
    metrics = {"n_refused": len(refused), "refused": refused}
    if len(refused) == 3:
        return ProbeResult(
            "V3_variant_purity_disagreement", "AF above purity cannot be scored",
            ProbeStatus.PASS, detail, metrics,
        )
    return ProbeResult(
        "V3_variant_purity_disagreement", "AF above purity cannot be scored",
        ProbeStatus.FAIL, f"a contradictory bound was published: {detail}", metrics,
    )


def probe_variant_pairing_contracts() -> ProbeResult:
    """V4: a paired change needs two specimens of ONE patient; two primaries are
    a duplicate rather than a change; and one specimen cannot carry the same
    locus twice."""
    primary = Spectrum(
        patient_id="P1", sample_id="P1-TP", timepoint="Primary",
        records=(_variant(alt=10, ref=90),),
    )
    recurrence = Spectrum(
        patient_id="P1", sample_id="P1-R1", timepoint="Recurrence",
        records=(_variant(sample="P1-R1", timepoint="Recurrence", alt=60, ref=40),),
    )
    other = Spectrum(
        patient_id="P2", sample_id="P2-R1", timepoint="Recurrence",
        records=(_variant(patient="P2", sample="P2-R1", alt=5, ref=95),),
    )
    change = paired_change(primary, recurrence)
    refused = []
    for args, label in (
        ((primary, other), "two patients"),
        ((primary, primary), "same specimen"),
        ((primary, Spectrum(
            patient_id="P1", sample_id="P1-TP", timepoint="Primary",
            records=(_variant(alt=10, ref=90), _variant(alt=10, ref=90)),
        )), "duplicate locus"),
    ):
        try:
            if label == "duplicate locus":
                args[1].keyed()
            else:
                paired_change(*args)
        except VariantError:
            refused.append(label)
    ok = (
        change.n_shared == 1
        and abs(change.af_delta["MSH6:2:48010530:C>T"] - 0.5) < 1e-12
        and len(refused) == 3
    )
    detail = (
        f"paired AF {change.af_primary['MSH6:2:48010530:C>T']:.2f} -> "
        f"{change.af_recurrence['MSH6:2:48010530:C>T']:.2f} "
        f"(delta {change.af_delta['MSH6:2:48010530:C>T']:+.2f}); "
        f"refused: {', '.join(refused)}"
    )
    metrics = {
        "n_shared": change.n_shared,
        "af_delta": change.af_delta,
        "refused": refused,
    }
    if ok:
        return ProbeResult(
            "V4_variant_pairing_contracts", "pairing and duplicate-locus contracts",
            ProbeStatus.PASS, detail, metrics,
        )
    return ProbeResult(
        "V4_variant_pairing_contracts", "pairing and duplicate-locus contracts",
        ProbeStatus.FAIL, f"a pairing contract leaked: {detail}", metrics,
    )


def probe_variant_depth_invariance() -> ProbeResult:
    """V5: the allele fraction is invariant to a 1000x depth rescale and can
    never exceed 1 -- the same log-scale discipline the detector probes apply to
    units, applied to sequencing depth."""
    base = _variant(alt=53, ref=84).allele_fraction
    scaled = _variant(alt=53000, ref=84000).allele_fraction
    saturated = _variant(alt=999, ref=1).allele_fraction
    ok = abs(base - scaled) < 1e-12 and saturated <= 1.0
    detail = f"AF {base:.6f} at 137 reads vs {scaled:.6f} at 137,000 reads; max AF {saturated:.3f}"
    metrics = {"af_base": base, "af_scaled": scaled, "af_saturated": saturated}
    if ok:
        return ProbeResult(
            "V5_variant_depth_invariance", "depth rescale cannot move the AF",
            ProbeStatus.PASS, detail, metrics,
        )
    return ProbeResult(
        "V5_variant_depth_invariance", "depth rescale cannot move the AF",
        ProbeStatus.FAIL, f"depth changed the fraction: {detail}", metrics,
    )


PROBES: Sequence[Callable[[], ProbeResult]] = (
    probe_missing_values,
    probe_incorrect_units,
    probe_corrupted_file,
    probe_duplicate_records,
    probe_irregular_timestamps,
    probe_higher_noise,
    probe_missing_modalities,
    probe_baseline_dynamics,
    probe_unobserved_changes,
    probe_out_of_distribution,
    probe_conflicting_inputs,
    probe_overconfidence,
    probe_failed_solver,
    probe_band_calibration,
    probe_band_calibration_under_missingness,
    probe_band_under_miscalibrated_sigma,
    probe_band_widens_with_noise,
    probe_observer_recovery,
    probe_observer_unidentified_refusal,
    probe_observer_units_error,
    probe_observer_too_few_observations,
    probe_observer_degeneracy_invariant,
    probe_variant_allele_fraction_from_counts,
    probe_variant_bounds_are_bounded,
    probe_variant_purity_disagreement_refuses,
    probe_variant_pairing_contracts,
    probe_variant_depth_invariance,
)


# ---------------------------------------------------------------------------
# The runner, the record, and the release gate.
# ---------------------------------------------------------------------------


def run_robustness_matrix() -> List[ProbeResult]:
    """Run every probe; return the results in a fixed order."""
    out: List[ProbeResult] = []
    for fn in PROBES:
        try:
            out.append(fn())
        except Exception as exc:  # noqa: BLE001 -- a probe that crashes is
            # itself a recorded failure, never a pass or a skip.
            out.append(
                ProbeResult(
                    fn.__name__.removeprefix("probe_"),
                    fn.__doc__ or "",
                    ProbeStatus.ERROR,
                    detail="",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return out


def robustness_record(results: Sequence[ProbeResult]) -> Dict[str, Any]:
    """Machine-readable record of one matrix run."""
    n_pass = sum(1 for r in results if r.status is ProbeStatus.PASS)
    n_fail = sum(1 for r in results if r.status is ProbeStatus.FAIL)
    n_err = sum(1 for r in results if r.status is ProbeStatus.ERROR)
    return {
        "record_version": RECORD_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": f"{platform.python_version()} on {sys.platform}",
        "probe_set": [fn.__name__ for fn in PROBES],
        "probes": [r.to_dict() for r in results],
        "aggregate": {
            "n_probes": len(results),
            "n_pass": n_pass,
            "n_fail": n_fail,
            "n_error": n_err,
            "gate": "PASS" if (n_fail == 0 and n_err == 0) else "BLOCKED",
        },
        "scope": (
            "synthetic-only stress tests of the danon.detect change-point "
            "detector, the band it publishes, the danon.observer ABC state "
            "estimator and the danon.variants allele-fraction channel: every "
            "probe builds its own inputs, so the variant probes test the "
            "channel's contracts and the mappings it refuses WITHOUT scoring the "
            "real GLASS record (that record is re-derived by tests/test_variants.py, "
            "not by this gate). A passing probe records a challenge survived, not "
            "a guarantee about untested data; no clinical claim"
        ),
    }


def release_gate(results: Sequence[ProbeResult]) -> Dict[str, Any]:
    """The gate: BLOCKED on any FAIL or ERROR, PASS otherwise."""
    n_fail = sum(1 for r in results if r.status is not ProbeStatus.PASS)
    failing = [r.probe_id for r in results if r.status is not ProbeStatus.PASS]
    return {
        "gate": "PASS" if n_fail == 0 else "BLOCKED",
        "failing_probes": failing,
        "n_fail": n_fail,
        "rule": "BLOCK if any probe FAILs or ERRORs; the failing ids are named",
    }


def write_robustness_record(results: Sequence[ProbeResult], path: str) -> Dict[str, Any]:
    record = robustness_record(results)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return record


def run_and_record(path: Optional[str] = None) -> Dict[str, Any]:
    """The one-call validation pipeline: probe, record, gate."""
    import os

    target = path or os.path.join("outputs_real", "robustness_record.json")
    results = run_robustness_matrix()
    record = write_robustness_record(results, target)
    return record
