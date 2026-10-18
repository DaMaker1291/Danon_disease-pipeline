"""Independent evaluation of the detector against ground truth.

The evaluator is the ONLY component that sees both the detector's outputs
and the ground-truth book; the detector never does. It measures, over a
scenario:

* **Sensitivity** -- of cases with a planted change, the fraction the
  detector flagged (confirmed or suspected),
* **Specificity** -- of negative controls with no planted change, the
  fraction the detector left unflagged,
* **Precision** -- the number of flagged cases that were real, divided by
  the number of flagged cases.
* **False alarms per observation period** -- among negative controls, the
  rate of false flags per horizon day (an FR is a case-level flag, not the
  count of increment-level exceedances),
* **Detection delay** -- detector change day minus planted day among
  correctly caught positive cases, measured both as a median and as a
  detection-day distribution,
* **Calibration** -- of confirmed calls, the fraction whose day falls
  inside the ?? 15-day window around the planted day, and how confirmed /
  suspected status predicted the divergence event (direction / magnitude
  checks are not in Phase 1's metrics),
* **Failure accounting** -- every missed change and every false alarm is
  named and counted in the record, per the mission's documentation
  requirement.

The evaluation is independent: it consumes a list of `DetectionResult`
objects and the corresponding `GroundTruth` items. Nothing imports the
detector here, so this file can later evaluate ANY detector -- a Layer-2
requirement handled here and not there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.detect.detector import DetectionResult, DetectionStatus
from danon.detect.schema import GroundTruth


#: The calibration tolerance the report publishes beside every delay figure:
#: a confirmed call is counted calibrated when its detector day lies within
#: this many days of the planted day, and the raw median delay is counted
#: separately so the summary never hides the actual bias.
CALIBRATION_WINDOW_DAYS = 15

#: How many distinct planted-change day-groups the record accounts for.
FORMAT_RATES: Tuple[str, ...] = (
    "sensitivity",
    "specificity",
    "precision",
    "false_alarms_per_1000_days",
    "median_delay_days",
    "calibration_rate",
)


@dataclass
class DetectionEvaluation:
    """Aggregate detection performance against known ground truth."""

    sensitivity: float
    specificity: float
    precision: Optional[float]      # None when no flagged case exists
    false_alarms_per_1000_days: float
    median_delay_days: Optional[float]
    calibration_rate: Optional[float]
    n_changed_cases: int
    n_flagged_changed: int
    n_negative_cases: int
    n_false_alarms: int
    delayed_cases: List[Tuple[str, int, int]] = field(default_factory=list)  # case, planted, detector
    missed_cases: List[str] = field(default_factory=list)
    false_alarm_cases: List[str] = field(default_factory=list)
    detected_changed_cases: List[str] = field(default_factory=list)

    @property
    def flagged_changed_frac(self) -> float:
        if self.n_changed_cases == 0:
            return float("nan")
        return self.n_flagged_changed / self.n_changed_cases

    @property
    def left_unflagged_frac(self) -> float:
        if self.n_negative_cases == 0:
            return float("nan")
        return 1.0 - self.n_false_alarms / self.n_negative_cases

    def summary_line(self) -> str:
        prec = f"{self.precision:.3f}" if self.precision is not None else "n/a (no flags)"
        delay = f"{self.median_delay_days:.1f}" if self.median_delay_days is not None else "n/a"
        calib = f"{self.calibration_rate:.3f}" if self.calibration_rate is not None else "n/a"
        return (
            f"sensitivity {self.sensitivity:.3f} | specificity {self.specificity:.3f} | "
            f"precision {prec} | false alarms/1000d {self.false_alarms_per_1000_days:.2f} | "
            f"median delay {delay} d | calibration {calib}"
        )


def _is_flagged(result: DetectionResult) -> bool:
    return result.status in (DetectionStatus.CONFIRMED, DetectionStatus.SUSPECTED)


def evaluate_detections(
    results: Sequence[DetectionResult],
    truths: Sequence[GroundTruth],
    *,
    horizon_days_per_negative: int = 365,
    calibration_window_days: int = CALIBRATION_WINDOW_DAYS,
) -> DetectionEvaluation:
    """Score a detector's results against the ground-truth book.

    ``horizon_days_per_negative`` is the observation-period length used to
    normalise false alarms among negative controls.
    """
    if len(results) != len(truths):
        raise ValueError("results and truths must be parallel")
    by_case: Dict[str, DetectionResult] = {r.case_id: r for r in results}
    missing = [t.case_id for t in truths if t.case_id not in by_case]
    extra = [r.case_id for r in results if r.case_id not in {t.case_id for t in truths}]
    if missing or extra:
        raise ValueError(f"case-id mismatch: missing={missing} extra={extra}")

    n_changed = sum(1 for t in truths if not t.is_negative_control)
    n_negative = len(truths) - n_changed

    n_flagged_changed = 0
    n_false_alarms = 0
    delays: List[float] = []
    calibrated: List[bool] = []
    flagged_changed_ids: List[str] = []
    missed_ids: List[str] = []
    false_alarm_ids: List[str] = []
    delayed: List[Tuple[str, int, int]] = []

    for truth in truths:
        result = by_case[truth.case_id]
        flagged = _is_flagged(result)
        if not truth.is_negative_control:
            if flagged:
                n_flagged_changed += 1
                flagged_changed_ids.append(truth.case_id)
                if result.change_day is not None and truth.planted_change_day is not None:
                    delay = int(result.change_day) - int(truth.planted_change_day)
                    delays.append(float(delay))
                    delayed.append((truth.case_id, int(truth.planted_change_day), int(result.change_day)))
                    calibrated.append(abs(delay) <= calibration_window_days)
            else:
                missed_ids.append(truth.case_id)
        else:
            if flagged:
                n_false_alarms += 1
                false_alarm_ids.append(truth.case_id)

    sensitivity = n_flagged_changed / n_changed if n_changed else 0.0
    specificity = 1.0 - (n_false_alarms / n_negative) if n_negative else 1.0
    precision = n_flagged_changed / (n_flagged_changed + n_false_alarms) if (n_flagged_changed + n_false_alarms) else None
    far = 1000.0 * n_false_alarms / (max(1, n_negative) * horizon_days_per_negative)

    return DetectionEvaluation(
        sensitivity=sensitivity,
        specificity=specificity,
        precision=precision,
        false_alarms_per_1000_days=far,
        median_delay_days=float(np.median(delays)) if delays else None,
        calibration_rate=(sum(calibrated) / len(calibrated)) if calibrated else None,
        n_changed_cases=n_changed,
        n_flagged_changed=n_flagged_changed,
        n_negative_cases=n_negative,
        n_false_alarms=n_false_alarms,
        delayed_cases=delayed,
        missed_cases=missed_ids,
        false_alarm_cases=false_alarm_ids,
        detected_changed_cases=flagged_changed_ids,
    )
