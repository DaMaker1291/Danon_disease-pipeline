"""The state estimate: what the tumour's state looks like, and how sure.

The detector never forces a confident answer. Alongside every change-point
decision it publishes an estimate of the CURRENT growth rate with an
uncertainty band, and when the observations cannot support even that (too
few points, or a rate so uncertain the band is unusable), the estimate
itself reports INSUFFICIENT rather than a number.

This is deliberately frequentist -- a rolling weighted fit with declared
per-point sigmas -- because Phase 1's contract is that every quantity it
publishes is reproducible from declared inputs and nohidden state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class StateEstimate:
    """The state at the latest observation, with honest uncertainty."""

    rate_per_day: Optional[float]      # None == insufficient evidence
    rate_sigma: Optional[float]
    band_low: Optional[float]          # rate +- 2 sigma ("95% band")
    band_high: Optional[float]
    n_used: int
    days_spanned: Tuple[float, float]
    insufficient_reason: str = ""

    @property
    def is_insufficient(self) -> bool:
        return self.rate_per_day is None

    def describe(self) -> str:
        if self.is_insufficient:
            return f"insufficient evidence ({self.insufficient_reason or 'no fit'})"
        return (
            f"rate {self.rate_per_day:.5f}/d (sigma {self.rate_sigma:.5f}, "
            f"95% band [{self.band_low:.5f}, {self.band_high:.5f}]), "
            f"from {self.n_used} points over days "
            f"{self.days_spanned[0]:.0f}-{self.days_spanned[1]:.0f}"
        )


def estimate_state(
    values: np.ndarray,
    days: np.ndarray,
    sigmas: np.ndarray,
    window: int = 10,
    min_points: int = 4,
) -> StateEstimate:
    """Weighted linear fit of log(value) vs day over the last ``window`` points.

    ``value`` must be positive (a radius, an area, a count). Zero or negative
    readings are replaced by the smallest positive value in the series; if
    none exists, the estimate refuses outright. That substitution is itself
    recorded in the insufficient-reason string when it fires.
    """
    n = len(values)
    if n < min_points:
        return StateEstimate(
            None, None, None, None, n,
            (float(days[0]), float(days[-1])),
            insufficient_reason=f"only {n} points, need {min_points}",
        )

    positive_floor = float(np.min(values[values > 0])) if np.any(values > 0) else 0.0
    if positive_floor <= 0.0:
        return StateEstimate(
            None, None, None, None, n,
            (float(days[0]), float(days[-1])),
            insufficient_reason="no positive value in the series",
        )
    substituted = bool(np.any(values <= 0))
    safe_values = np.where(values <= 0, positive_floor, values)
    if substituted:
        note = f" (nonpositive readings floored at {positive_floor:.4g})"
    else:
        note = ""

    last = slice(max(0, n - window), n)
    y = np.log(safe_values[last])
    x = days[last]
    w = 1.0 / np.maximum(sigmas[last] / safe_values[last], 1e-9) ** 2

    # weighted least squares for y = slope * x + intercept
    W = np.sum(w)
    if W <= 0 or not np.isfinite(W):
        return StateEstimate(
            None, None, None, None, int(w.size),
            (float(x[0]), float(x[-1])),
            insufficient_reason="degenerate weights" + note,
        )
    sx = np.sum(w * x) / W
    sxx = np.sum(w * x * x) / W
    sy = np.sum(w * y) / W
    sxy = np.sum(w * x * y) / W
    denom = sxx - sx * sx
    if denom <= 1e-12 or not np.all(np.isfinite([sx, sxx, sy, sxy])):
        return StateEstimate(
            None, None, None, None, int(w.size),
            (float(x[0]), float(x[-1])),
            insufficient_reason="days do not span enough range to fit a rate" + note,
        )
    slope = (sxy - sx * sy) / denom
    # residual variance (weighted), then sigma of the slope
    intercept = sy - slope * sx
    resid = y - (slope * x + intercept)
    dof = max(1, w.size - 2)
    # Weighted least squares: Var(slope) = s2 / sum_i w_i (x_i - xbar)^2, with
    # sum_i w_i (x_i - xbar)^2 = W * denom and the residual variance s2
    # ESTIMATED from the residuals themselves (sum_i w_i resid_i^2 / dof) --
    # which is what makes the band honest when a declared sigma is wrong: a
    # 3x-under-declared sigma inflates the residuals and the band with them.
    # An earlier form divided s2 by W as well, shrinking the published sigma by
    # sqrt(W) (~40x over this window) and turning the "95% band" into one that
    # contained the true rate in 1 of 40 synthetic cases. Measured, and pinned
    # by the U-probes in danon/robustness.py.
    s2 = float(np.sum(w * resid * resid) / dof)
    slope_sigma = float(np.sqrt(max(s2, 0.0) / (W * denom)))

    return StateEstimate(
        rate_per_day=float(slope),
        rate_sigma=slope_sigma,
        band_low=float(slope - 2 * slope_sigma),
        band_high=float(slope + 2 * slope_sigma),
        n_used=int(w.size),
        days_spanned=(float(x[0]), float(x[-1])),
        insufficient_reason=note.lstrip(),
    )
