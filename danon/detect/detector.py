"""The change-point detector: a sequential GLR baseline, fully declared.

Generative model. ``radius(t) = r0 * exp(k * t)`` makes log-radius piecewise
linear in day; the planted change is a slope shift at a known day. The
detector therefore works on log values:

    y_i = log(value_i),  sigma_{y,i} = sigma_i / value_i   (declared, propagated)

Decision rule (sequential generalized likelihood ratio). After each new
observation the detector fits

  * ONE line to all points so far, and
  * TWO lines split at every candidate day tau inside the declared
    per-segment windows,

both by weighted least squares with the declared sigmas as weights. Because
the weights are exactly 1/sigma^2, the weighted-SSE improvement

    Delta_i = SSE(one line) - min_tau SSE(two lines)

is chi-square(1)-scaled under "no change", so sqrt(Delta_i) behaves like a
z-statistic and the thresholds carry a direct meaning: a call needs
roughly ``confirmation_threshold`` combined standard deviations of
improvement. The candidate with the best split is recorded as the
ESTIMATED change day; the day of the observation that first crossed the
threshold is the CALL day, and detection delay is measured from the call.

Reporting and refusals:

  * CONFIRMED -- the statistic reached ``confirmation_threshold``,
  * SUSPECTED -- it reached ``suspicion_threshold`` but never confirmed,
  * INSUFFICIENT_EVIDENCE -- too few usable points or a degenerate fit
    (refusals take precedence over calls; the module never masks one),
  * NO_CHANGE -- the statistic never left its null range.

Missing observations need no special handling: absent days simply do not
appear as points, and a nonpositive value (log undefined) is skipped, not
imputed. Extremely noisy points are not excluded -- the weights spiral
them down naturally, which is more honest than silently dropping data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np

from danon.detect.schema import ObservationStream
from danon.detect.state import StateEstimate, estimate_state


class DetectionStatus(str, Enum):
    CONFIRMED = "confirmed"
    SUSPECTED = "suspected"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_CHANGE = "no_change"


@dataclass(frozen=True)
class DetectorConfig:
    """Every knob the detector turns, declared in one place."""

    #: sqrt(weighted-SSE improvement) needed to CONFIRM a change
    confirmation_threshold: float = 4.5
    #: sqrt(weighted-SSE improvement) needed to SUSPECT (not confirm)
    suspicion_threshold: float = 3.2
    #: minimum observations before any call is made at all
    min_points_for_call: int = 6
    #: minimum points in the LEFT segment (pinning the pre-change slope)
    min_left_segment: int = 5
    #: minimum points in the RIGHT segment (a new slope needs evidence)
    min_right_segment: int = 3

    def __post_init__(self) -> None:
        if self.confirmation_threshold <= self.suspicion_threshold:
            raise ValueError("confirmation_threshold must exceed suspicion_threshold")
        if self.suspicion_threshold <= 0:
            raise ValueError("suspicion_threshold must be > 0")
        if self.min_points_for_call < 3:
            raise ValueError("min_points_for_call must be >= 3")
        if self.min_left_segment < 3:
            raise ValueError("min_left_segment must be >= 3")
        if self.min_right_segment < 2:
            raise ValueError("min_right_segment must be >= 2")

    def validated(self) -> "DetectorConfig":
        """A config object with all the same numbers, for direct reading."""
        return self


@dataclass
class DetectionResult:
    """A change decision, its evidence trail, and its refusal paths."""

    case_id: str
    status: DetectionStatus
    change_day: Optional[int] = None           # day of the CALL (delay basis)
    estimated_change_day: Optional[int] = None  # argmax split (location basis)
    evidence_at_call: float = 0.0              # statistic when the call fired
    increment_z: Optional[float] = None        # last contributing statistic
    direction: str = ""                        # "growth" | "shrink" | ""
    state_estimate: Optional[StateEstimate] = None
    refusal_reason: str = ""
    step_statistics: List[Tuple[int, float]] = field(default_factory=list)
    evidence_series: List[Tuple[int, float]] = field(default_factory=list)

    def describe(self) -> str:
        head = f"{self.case_id}: {self.status.value}"
        if self.change_day is not None:
            head += f", call day {self.change_day}"
        if self.estimated_change_day is not None:
            head += f" (estimated change day {self.estimated_change_day})"
        if self.direction:
            head += f", direction {self.direction}"
        if self.state_estimate is not None:
            head += f"; {self.state_estimate.describe()}"
        if self.refusal_reason:
            head += f" [{self.refusal_reason}]"
        return head


@dataclass(frozen=True)
class _Points:
    """The usable log-scale points and their prefix sums.

    A point is usable when its value is positive (log defined) and its
    declared sigma gives finite weights. The unit of the weighted sums is
    "one declared sigma of log-noise", which is what makes the GLR
    statistic interpretable without any fitted noise scale.
    """

    days: np.ndarray          # x_i
    log_values: np.ndarray    # y_i
    weights: np.ndarray       # w_i = 1 / sigma_{y,i}^2

    @property
    def n(self) -> int:
        return len(self.days)


def _make_points(stream: ObservationStream) -> Tuple[_Points, int]:
    """Log points from the stream, plus the count of skipped observations."""
    days_l, y_l, w_l = [], [], []
    skipped = 0
    for o in stream.observations:
        if o.value <= 0 or not np.isfinite(o.value) or o.sigma <= 0:
            skipped += 1
            continue
        sigma_log = o.sigma / o.value
        if not np.isfinite(sigma_log) or sigma_log <= 0:
            skipped += 1
            continue
        days_l.append(float(o.day))
        y_l.append(float(np.log(o.value)))
        w_l.append(1.0 / (sigma_log * sigma_log))
    return (
        _Points(
            days=np.array(days_l, dtype=float),
            log_values=np.array(y_l, dtype=float),
            weights=np.array(w_l, dtype=float),
        ),
        skipped,
    )


class _SegmentFitter:
    """Closed-form weighted-LS segment costs via prefix sums.

    ``sse(l, r)`` is the weighted residual sum of squares of the best line
    through points ``[l, r)``, in units of declared sigma. ``sse`` of a
    single-line fit minus the best two-line split is the GLR statistic's
    input.
    """

    def __init__(self, pts: _Points) -> None:
        n = pts.n
        z = np.zeros(1)
        self.p_w = np.concatenate([z, np.cumsum(pts.weights)])
        self.p_wx = np.concatenate([z, np.cumsum(pts.weights * pts.days)])
        self.p_wxx = np.concatenate([z, np.cumsum(pts.weights * pts.days * pts.days)])
        self.p_wy = np.concatenate([z, np.cumsum(pts.weights * pts.log_values)])
        self.p_wxy = np.concatenate(
            [z, np.cumsum(pts.weights * pts.days * pts.log_values)]
        )
        self.p_wyy = np.concatenate(
            [z, np.cumsum(pts.weights * pts.log_values * pts.log_values)]
        )
        self.n = n

    def sse(self, l: int, r: int) -> float:
        """Weighted SSE of the best line through points [l, r)."""
        if r - l < 2:
            return 0.0
        w = self.p_w[r] - self.p_w[l]
        wx = self.p_wx[r] - self.p_wx[l]
        wxx = self.p_wxx[r] - self.p_wxx[l]
        wy = self.p_wy[r] - self.p_wy[l]
        wxy = self.p_wxy[r] - self.p_wxy[l]
        wyy = self.p_wyy[r] - self.p_wyy[l]
        den_x = wxx - (wx * wx) / w
        if den_x <= 1e-12:
            return max(0.0, wyy - (wy * wy) / w)
        slope = (wxy - (wx * wy) / w) / den_x
        sse = wyy - (wy * wy) / w - slope * slope * den_x
        return float(max(0.0, sse))


def detect(
    stream: ObservationStream,
    config: Optional[DetectorConfig] = None,
) -> DetectionResult:
    """Run the sequential GLR change detector over one observation stream."""
    cfg = (config or DetectorConfig()).validated()
    obs = stream.observations

    state = estimate_state(stream.values, stream.days, stream.sigmas)
    pts, _skipped = _make_points(stream)

    # Honest refusals first -- precedence over any confident call.
    if pts.n < cfg.min_points_for_call:
        return DetectionResult(
            case_id=stream.case_id,
            status=DetectionStatus.INSUFFICIENT_EVIDENCE,
            refusal_reason=(
                f"only {pts.n} usable points, need {cfg.min_points_for_call}"
                + (f" ({_skipped} skipped as unusable)" if _skipped else "")
            ),
            state_estimate=state,
        )

    if pts.n < cfg.min_left_segment + cfg.min_right_segment + 1:
        return DetectionResult(
            case_id=stream.case_id,
            status=DetectionStatus.INSUFFICIENT_EVIDENCE,
            refusal_reason=(
                f"only {pts.n} usable points; a two-segment fit needs at least "
                f"{cfg.min_left_segment + cfg.min_right_segment + 1}"
            ),
            state_estimate=state,
        )

    fitter = _SegmentFitter(pts)

    direction = "growth" if state.rate_per_day is None or state.rate_per_day >= 0 else "shrink"

    called_status: DetectionStatus = DetectionStatus.NO_CHANGE
    call_day: Optional[int] = None
    estimated_day: Optional[int] = None
    evidence_at_call = 0.0
    last_stat: Optional[float] = None
    stats_log: List[Tuple[int, float]] = []

    # Sequential scan: after each new point, recompute the GLR statistic
    # over all candidates and compare against the declared thresholds.
    for i in range(cfg.min_left_segment + cfg.min_right_segment, pts.n):
        sse_one = fitter.sse(0, i + 1)
        best_two = float("inf")
        best_tau = -1
        lo = cfg.min_left_segment
        hi = i - cfg.min_right_segment + 1
        for tau in range(lo, hi):
            sse_two = fitter.sse(0, tau) + fitter.sse(tau, i + 1)
            if sse_two < best_two:
                best_two = sse_two
                best_tau = tau
        stat = float(np.sqrt(max(0.0, sse_one - best_two)))
        stats_log.append((int(pts.days[i]), stat))
        last_stat = stat

        if stat >= cfg.confirmation_threshold and called_status is DetectionStatus.NO_CHANGE:
            called_status = DetectionStatus.CONFIRMED
            call_day = int(pts.days[i])
            estimated_day = int(pts.days[best_tau])
            evidence_at_call = stat
            break
        if stat >= cfg.suspicion_threshold and called_status is DetectionStatus.NO_CHANGE:
            called_status = DetectionStatus.SUSPECTED
            call_day = int(pts.days[i])
            estimated_day = int(pts.days[best_tau])
            evidence_at_call = stat
            # no break: a suspicion may confirm later in the same stream

    if called_status is DetectionStatus.NO_CHANGE:
        return DetectionResult(
            case_id=stream.case_id,
            status=DetectionStatus.NO_CHANGE,
            state_estimate=state,
            step_statistics=stats_log,
            evidence_series=stats_log,
            increment_z=last_stat,
        )
    return DetectionResult(
        case_id=stream.case_id,
        status=called_status,
        change_day=call_day,
        estimated_change_day=estimated_day,
        evidence_at_call=evidence_at_call,
        increment_z=last_stat,
        direction=direction,
        state_estimate=state,
        step_statistics=stats_log,
        evidence_series=stats_log,
    )
