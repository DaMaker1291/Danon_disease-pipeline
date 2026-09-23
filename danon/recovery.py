"""
Planted ground-truth recovery: proving the machinery against a known answer.

WHY THIS MODULE EXISTS
----------------------
Every measurement in this repository is made on data whose true answer is
unknown: real patient cohorts tell you whether a model is *useful*, never
whether the estimation machinery is *correct*. A pipeline can recover C = 0.64
on real data and still be computing the statistic wrong -- the repo has two
pinned regressions of exactly that kind (a concordance index that returned 2.92
by counting subjects in the denominator while accumulating pairs in the
numerator, and a Cox hazard ratio that omitted beta).

The fix is a known answer. Plant a physical constant, run it through the
pipeline, and check what comes back. This module does that for the two
estimators the rest of the project rests on: the spatial transport operator and
the survival/hazard machinery.

THE TWO RECOVERY BENCHMARKS
---------------------------
1. **Transport (diffusion + clearance).** A 3D field is generated from the
   *analytic* point-source solution

       C(r, t) = M / (4*pi*D*t)^(3/2) * exp(-r^2 / (4*D*t)) * exp(-lambda*t)

   which is an independent solver: it knows nothing about this codebase's grid,
   stencil or time-stepping. That field is then sampled on the engine grid and
   the engine's own discrete residual

       (C^{n+1} - C^n) / dt  =  D * Lap(C^n)  -  lambda * C^n

   is fitted for D and lambda by linear least squares.

   **This is deliberately non-circular.** Generating the ground truth by
   advancing the same discrete operator and then inverting it would recover the
   planted constants to machine precision and prove nothing -- it would only
   show that a linear solve can invert a linear operator it was handed. Because
   the ground truth comes from a different solver, the recovery error *is* the
   engine's discretisation error, and the fitted R^2 measures how well the
   engine's discrete PDE explains a field that obeys the continuous one.

2. **Hazard estimation.** A cohort is simulated from a Cox model with *planted*
   coefficients and exponential survival, so the true linear predictor is known.
   The default holdout pipeline then has to (a) recover the coefficients and
   (b) reach a C-index at the **oracle ceiling** -- the value obtained by scoring
   with the true predictor. When the fitted value sits at the oracle, the
   real-data C-index becomes interpretable; when it does not, the estimator is
   biased and no real-data number from it can be trusted.

WHAT THESE BENCHMARKS DO NOT PROVE
----------------------------------
Recovering a constant you planted is evidence about the *estimator*, never about
the *biology*. R^2 here is not predictive accuracy on cancer and must not be
reported as such: a pipeline that inverts a known PDE perfectly has demonstrated
that it can invert a known PDE perfectly. The real-cohort numbers stay what they
are, and the two measurements answer different questions.

Citations:
  Point-source diffusion with first-order clearance: Crank J. The Mathematics of
    Diffusion, 2nd ed., Oxford (1975), ch. 4.
  Forward-Euler stability for the 3D 7-point Laplacian: D*dt <= 1/6.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from danon.cure import EvidenceStatus

#: Stability limit for explicit forward-Euler diffusion on the engine's 7-point
#: 3D stencil: D*dt must not exceed 1/6, or the scheme amplifies grid-scale modes.
FORWARD_EULER_STABILITY_LIMIT = 1.0 / 6.0


class RecoveryFailed(RuntimeError):
    """Raised when a planted constant was not recovered inside its tolerance.

    A recovery benchmark that cannot fail is a print statement. This exception is
    the mechanism by which the check has teeth.
    """


# ---------------------------------------------------------------------------
# Benchmark 1: transport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IdentifiabilityDiagnostic:
    """Whether a planted (D, lambda) pair is recoverable on this grid at all.

    This exists because the first version of this benchmark reported
    ``R^2 = 1.000000`` next to a 70% error on D and a 2400% error on lambda, and
    the pair was **not identifiable** rather than the estimator being broken.

    The physics: a point source with clearance has a diffusion length
    ``L = sqrt(2*D/lambda)`` -- how far the drug spreads before it is cleared.
    If ``L`` is below the voxel pitch, the drug is gone before it moves one
    voxel, the spatial curvature term is negligible next to the clearance term,
    and the least-squares design is effectively rank-1: it can fit perfectly
    while placing the coefficients anywhere along a line. The planted pair in
    the original sketch (D=0.15, lambda=0.80 per day on a 1-voxel grid) gives
    L = 0.61 voxels, which is exactly that degenerate case.
    """

    planted_d: float
    planted_lambda: float
    voxel_mm: float
    diffusion_length_voxels: float
    diffusion_length_mm: float
    grid_points_per_diffusion_length: float
    #: ``||D * Lap(C)|| / ||lambda * C||`` over the fitted equations. Order 1
    #: means both terms are visible; << 1 means only clearance is measurable.
    diffusion_to_clearance_signal: float
    identifiable: bool
    reason: str

    def describe(self) -> str:
        return (
            f"diffusion length {self.diffusion_length_voxels:.2f} voxels "
            f"({self.diffusion_length_mm:.2f} mm); signal ratio "
            f"D*Lap/(lam*C) = {self.diffusion_to_clearance_signal:.3g}; "
            f"identifiable = {self.identifiable} ({self.reason})"
        )


@dataclass(frozen=True)
class TransportRecovery:
    """Planted versus recovered transport constants, with the fit's own quality."""

    planted_d: float
    planted_lambda: float
    recovered_d: float
    recovered_lambda: float
    r_squared: float
    n_equations: int
    n_times: int
    n_steps_used: int
    grid_shape: Tuple[int, ...]
    source_sigma_voxels: float
    dt: float
    time_range: Tuple[float, float]
    condition_number: float
    peak_concentration: float
    diagnostics: IdentifiabilityDiagnostic

    @property
    def d_relative_error(self) -> float:
        return abs(self.recovered_d - self.planted_d) / abs(self.planted_d)

    @property
    def lambda_relative_error(self) -> float:
        return abs(self.recovered_lambda - self.planted_lambda) / abs(self.planted_lambda)

    @property
    def max_relative_error(self) -> float:
        return max(self.d_relative_error, self.lambda_relative_error)

    def describe(self) -> str:
        return (
            f"D planted {self.planted_d:.4f} -> recovered {self.recovered_d:.4f} "
            f"({self.d_relative_error * 100:.2f}% error); lambda planted "
            f"{self.planted_lambda:.4f} -> recovered {self.recovered_lambda:.4f} "
            f"({self.lambda_relative_error * 100:.2f}% error); fit R^2 = "
            f"{self.r_squared:.6f} over {self.n_equations} equations from "
            f"{self.n_steps_used} step(s); {self.diagnostics.describe()}"
        )


def analytic_point_source_field(
    *,
    planted_d: float,
    planted_lambda: float,
    times: Sequence[float],
    grid_size: int = 24,
    mass: float = 1.0,
    sigma_voxels: float = 3.0,
    min_sigma_voxels: float = 1.0,
) -> np.ndarray:
    """Sample the analytic 3D point-source solution on a cubic grid.

    Independent of this codebase's numerics by construction: the closed-form
    solution for an instantaneous point source of mass ``mass`` in an infinite
    medium with first-order clearance.

    Two different widths, deliberately separated:

    * ``min_sigma_voxels`` is the **singularity guard**. The solution is a delta
      at t=0, so the blob must be at least a voxel wide before it is sampled or
      the amplitude is meaningless. This is a numerical floor.
    * ``sigma_voxels`` is the **design width** the caller wants its time series to
      start from. It is enforced by :func:`infer_times`, not here -- an earlier
      version conflated the two, and the central-difference samples at ``t - h``
      then tripped the design floor on a series that was correctly designed.
    """
    if planted_d <= 0:
        raise ValueError("planted_d must be positive")
    if planted_lambda < 0:
        raise ValueError("planted_lambda must be non-negative")
    g = int(grid_size)
    centre = (g - 1) / 2.0
    coords = np.arange(g, dtype=float) - centre
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
    r2 = xx ** 2 + yy ** 2 + zz ** 2

    out = np.empty((len(times), g, g, g), dtype=float)
    for i, t in enumerate(times):
        if t <= 0:
            raise ValueError("times must be strictly positive")
        sigma = math.sqrt(2.0 * planted_d * t)
        if sigma < min_sigma_voxels:
            raise ValueError(
                f"t={t} gives sigma={sigma:.3f} voxels, below the singularity "
                f"guard {min_sigma_voxels}; the point-source delta is not "
                f"resolved at this time"
            )
        amplitude = mass / (4.0 * math.pi * planted_d * t) ** 1.5
        out[i] = amplitude * np.exp(-r2 / (4.0 * planted_d * t)) * math.exp(
            -planted_lambda * t
        )
    return out


def _engine_laplacian(f: np.ndarray) -> np.ndarray:
    """The engine's stencil, imported so a change there is felt here.

    Zero-flux edge padding, 6-neighbour, as in :func:`danon.evosteer._laplacian`.
    """
    from danon.evosteer import _laplacian as engine_laplacian

    return engine_laplacian(f)


@dataclass(frozen=True)
class TransportFit:
    """Raw output of the least-squares transport fit."""

    d_hat: float
    lambda_hat: float
    r_squared: float
    n_equations: int
    n_steps_used: int
    condition_number: float
    signal_ratio: float
    peak_concentration: float
    #: Largest |C| on the grid boundary as a fraction of the peak. The analytic
    #: solution is for an infinite domain while the engine stencil assumes
    #: zero flux, so a non-negligible edge value contaminates the fit.
    edge_amplitude_ratio: float


def recover_transport_parameters(
    fields: np.ndarray,
    dfields_dt: np.ndarray,
    *,
    interior_margin: int = 3,
    min_relative_peak: float = 1e-6,
) -> TransportFit:
    """Fit ``dC/dt = D * Lap(C) - lambda * C`` for (D, lambda).

    The regression is exact in form, so a residual means the field does not
    satisfy the engine's discrete operator -- which is precisely the error this
    benchmark is looking for.

    Three guards, each learned from a failure of the previous version:

    * ``interior_margin`` excludes the boundary, where the zero-flux padding
      differs from the infinite-domain analytic solution.
    * ``min_relative_peak`` drops time samples whose peak has fallen below this
      fraction of the series maximum. A frame at 1e-39 is floating-point noise,
      and including it lets rounding error dominate the normal equations.
    * ``dfields_dt`` is supplied by the caller and must be a **second-order**
      estimate of the time derivative. The first version regressed the
      forward-Euler difference ``(C^{n+1}-C^n)/dt`` against ``Lap(C^n)``, so the
      leading ``O(dt)*d2C/dt2`` truncation term was unmodeled and the fitted
      coefficients absorbed the misfit -- producing R^2 = 0.9997 alongside
      coefficients an order of magnitude wrong.

    ``signal_ratio`` is ``||D*Lap(C)|| / ||lambda*C||``: the fit is only
    informative about D when this is of order 1.
    """
    if fields.ndim != 4 or dfields_dt.shape != fields.shape:
        raise ValueError("fields and dfields_dt must both be (n_times, nx, ny, nz)")
    if fields.shape[0] < 2:
        raise ValueError("need at least two time samples")
    m = int(interior_margin)
    sl = tuple(slice(m, -m) for _ in range(3))

    peak0 = float(np.max(np.abs(fields[0]))) or 1.0
    lhs_parts: List[np.ndarray] = []
    lap_parts: List[np.ndarray] = []
    conc_parts: List[np.ndarray] = []
    n_steps = 0
    for n in range(fields.shape[0]):
        c_now = fields[n]
        if float(np.max(np.abs(c_now))) < min_relative_peak * peak0:
            continue
        n_steps += 1
        lhs_parts.append(dfields_dt[n][sl].ravel())
        lap_parts.append(_engine_laplacian(c_now)[sl].ravel())
        conc_parts.append(c_now[sl].ravel())

    if n_steps == 0:
        raise RecoveryFailed(
            f"every time sample fell below the {min_relative_peak:.0e} "
            f"relative-peak floor; the series decays too fast to fit"
        )

    lhs_vec = np.concatenate(lhs_parts)
    lap_vec = np.concatenate(lap_parts)
    conc_vec = np.concatenate(conc_parts)

    # Columns are scaled before the solve because the Laplacian and the
    # concentration differ by orders of magnitude across the series; an
    # unscaled solve is dominated by whichever is numerically larger.
    scale_lap = float(np.sqrt(np.mean(lap_vec ** 2))) or 1.0
    scale_conc = float(np.sqrt(np.mean(conc_vec ** 2))) or 1.0
    A = np.column_stack([lap_vec / scale_lap, -conc_vec / scale_conc])
    coef, *_ = np.linalg.lstsq(A, lhs_vec, rcond=None)
    d_hat = float(coef[0] / scale_lap)
    lambda_hat = float(coef[1] / scale_conc)

    pred = A @ coef
    ss_res = float(np.sum((lhs_vec - pred) ** 2))
    ss_tot = float(np.sum((lhs_vec - lhs_vec.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    num = float(np.linalg.norm(d_hat * lap_vec))
    den = float(np.linalg.norm(lambda_hat * conc_vec)) or 1.0
    # Largest magnitude on the outermost shell of the grid, relative to the
    # series peak: how much of the field is still alive at the boundary the
    # zero-flux operator assumes is sealed.
    edge = max(
        float(np.max(np.abs(fields[0][idx]))) for idx in _shell_indices(fields.shape[1:])
    )
    return TransportFit(
        d_hat=d_hat, lambda_hat=lambda_hat, r_squared=float(r2),
        n_equations=int(lhs_vec.size), n_steps_used=n_steps,
        condition_number=float(np.linalg.cond(A)),
        signal_ratio=num / den,
        peak_concentration=peak0,
        edge_amplitude_ratio=edge / peak0,
    )


def _shell_indices(shape: Sequence[int]) -> List[Tuple[object, ...]]:
    """Slices selecting the six faces of a 3D array."""
    nx, ny, nz = (int(s) for s in shape)
    ax = [slice(None), slice(None), slice(None)]
    out: List[Tuple[object, ...]] = []
    for axis in range(3):
        for which in (0, -1):
            idx = list(ax)
            idx[axis] = which
            out.append(tuple(idx))
    return out


def assess_transport_identifiability(
    planted_d: float,
    planted_lambda: float,
    *,
    voxel_mm: float = 1.0,
    diffusion_length_min_voxels: float = 1.5,
    min_signal_ratio: float = 1e-2,
) -> IdentifiabilityDiagnostic:
    """Decide whether a planted (D, lambda) pair is recoverable on this grid.

    A pair is treated as identifiable when the diffusion length
    ``sqrt(2*D/lambda)`` spans at least ``diffusion_length_min_voxels``. Below
    that the spatial term is invisible relative to clearance and the inverse
    problem has no unique solution, whatever the fit quality says.
    """
    if planted_d <= 0 or planted_lambda <= 0:
        raise ValueError("planted_d and planted_lambda must be positive")
    length_vox = math.sqrt(2.0 * planted_d / planted_lambda) / voxel_mm
    length_mm = length_vox * voxel_mm
    # The criterion is the diffusion length, not a signal ratio: the ratio is a
    # *measured* property of the fitted field (TransportFit.signal_ratio) and is
    # filled in by the caller. Predicting it here from the planted constants
    # would be circular -- it would assume the answer the fit is meant to test.
    identifiable = length_vox >= diffusion_length_min_voxels
    if identifiable:
        reason = (
            f"diffusion length spans {length_vox:.2f} voxels, so both the "
            f"curvature and the clearance term are visible"
        )
    else:
        reason = (
            f"diffusion length is only {length_vox:.2f} voxels (needs >= "
            f"{diffusion_length_min_voxels}): the drug is cleared before it "
            f"crosses a voxel, so D is not recoverable from this experiment "
            f"regardless of fit quality"
        )
    return IdentifiabilityDiagnostic(
        planted_d=float(planted_d),
        planted_lambda=float(planted_lambda),
        voxel_mm=float(voxel_mm),
        diffusion_length_voxels=float(length_vox),
        diffusion_length_mm=float(length_mm),
        grid_points_per_diffusion_length=float(length_vox),
        diffusion_to_clearance_signal=float("nan"),
        identifiable=bool(identifiable),
        reason=reason,
    )


def infer_times(
    planted_d: float,
    planted_lambda: float,
    *,
    sigma_voxels: float,
    n_times: int,
    span_in_clearance_times: float = 2.5,
) -> List[float]:
    """Time points that make **both** transport terms visible.

    Two constraints bracket the series, and both are physical:

    * **Lower bound -- resolvability.** A point source is a delta at t=0; at
      early times the blob is narrower than the grid and cannot be represented.
      The series may not start before ``t = sigma^2 / (2D)``, when the analytic
      width reaches ``sigma_voxels``.
    * **Upper bound -- clearance.** Past ``t ~ span/lambda`` the field has been
      attenuated by ``exp(-lambda*t)`` until only clearance is measurable, and
      the last frames become floating-point noise.

    A pair with no valid window between the two is not identifiable on this
    grid, and :func:`assess_transport_identifiability` says so.
    """
    t_min = sigma_voxels ** 2 / (2.0 * planted_d)
    t_max = span_in_clearance_times / planted_lambda
    if t_max <= t_min:
        raise ValueError(
            f"no usable window: resolvability requires t >= {t_min:.3f} days but "
            f"clearance kills the field by t = {t_max:.3f} days. The planted "
            "pair is not identifiable on this grid."
        )
    return [t_min + (t_max - t_min) * i / max(n_times - 1, 1) for i in range(n_times)]


def transport_recovery(
    *,
    planted_d: float = 0.35,
    planted_lambda: float = 0.10,
    grid_size: int = 32,
    n_times: int = 12,
    derivative_step: float = 0.01,

    mass: float = 1.0,
    sigma_voxels: float = 2.5,
    interior_margin: int = 3,
    voxel_mm: float = 1.0,
    engine_dt: float = 0.05,
) -> TransportRecovery:
    """Plant (D, lambda), generate analytically, recover with the engine stencil.

    Defaults are the engine's own transport regime (``DEFAULT_DRUGS``:
    D = 0.35 voxel^2/day, clearance = 0.10/day), because that is the pair whose
    recovery actually matters. Its diffusion length is ``sqrt(2*0.35/0.10) =
    2.65`` voxels, which makes both terms visible. The sketch's original
    (0.15, 0.80) gives 0.61 voxels and is refused outright.

    Two numerical choices are load-bearing and both were found by failure:

    * The time derivative is a **central difference with a small step**
      (``derivative_step``, default 0.01 days) taken on the analytic solution,
      not the forward-Euler difference the engine would step with. Regressing
      the forward difference leaves ``O(dt)*d2C/dt2`` unmodeled and biases the
      coefficients by an order of magnitude even at R^2 = 0.9997.
    * ``grid_size`` is 32 so the field decays before the boundary. The analytic
      solution is for an infinite domain while the engine stencil assumes zero
      flux, so a boundary at only ~2.4 widths contaminates the fit.

    The explicit-Euler stability condition is checked against ``engine_dt``
    first: an unstable scheme would produce a field no operator can fit, and
    that failure would be misattributed to the estimator.
    """
    if planted_d * engine_dt > FORWARD_EULER_STABILITY_LIMIT:
        raise ValueError(
            f"D*dt = {planted_d * engine_dt:.4f} exceeds the forward-Euler "
            f"stability limit {FORWARD_EULER_STABILITY_LIMIT:.4f} for the 3D "
            f"7-point Laplacian; the engine could not step this field at dt = "
            f"{engine_dt}"
        )
    diagnostic = assess_transport_identifiability(
        planted_d, planted_lambda, voxel_mm=voxel_mm
    )
    if not diagnostic.identifiable:
        raise RecoveryFailed(
            "refusing to report a recovery figure for a non-identifiable "
            "planted pair: " + diagnostic.reason
        )

    times = infer_times(
        planted_d, planted_lambda, sigma_voxels=sigma_voxels, n_times=n_times
    )
    h = float(derivative_step)
    # The series itself must honour the requested design width, and h must be
    # small enough that the derivative samples cannot approach the singularity.
    if h >= times[0]:
        raise ValueError(
            f"derivative_step {h} is not small against the first sample "
            f"{times[0]:.3f}; the central difference would sample t <= 0"
        )
    kw = dict(
        planted_d=planted_d, planted_lambda=planted_lambda, grid_size=grid_size,
        mass=mass, sigma_voxels=sigma_voxels,
    )
    fields = analytic_point_source_field(times=times, **kw)
    minus = analytic_point_source_field(times=[t - h for t in times], **kw)
    plus = analytic_point_source_field(times=[t + h for t in times], **kw)
    dfields_dt = (plus - minus) / (2.0 * h)

    fit = recover_transport_parameters(
        fields, dfields_dt, interior_margin=interior_margin
    )
    # Replace the placeholder signal ratio with the MEASURED one. The
    # identifiability flag stays a prediction from the planted pair (it is a
    # property of the experiment design); the ratio is an outcome.
    diagnostic = replace(
        diagnostic, diffusion_to_clearance_signal=fit.signal_ratio
    )
    return TransportRecovery(
        planted_d=float(planted_d),
        planted_lambda=float(planted_lambda),
        recovered_d=fit.d_hat,
        recovered_lambda=fit.lambda_hat,
        r_squared=fit.r_squared,
        n_equations=fit.n_equations,
        n_times=n_times,
        n_steps_used=fit.n_steps_used,
        grid_shape=tuple(int(s) for s in fields.shape[1:]),
        source_sigma_voxels=float(sigma_voxels),
        dt=h,
        time_range=(float(times[0]), float(times[-1])),
        condition_number=fit.condition_number,
        peak_concentration=fit.peak_concentration,
        diagnostics=diagnostic,
    )


@dataclass(frozen=True)
class IdentifiabilitySweepRow:
    """Recovery outcome across the identifiability boundary.

    The sweep axis is the clearance rate, because that is the axis that matters
    clinically: it moves the diffusion length ``sqrt(2*D/lambda)``, and below the
    grid pitch the experiment stops being informative about D no matter what
    estimator is used.
    """

    planted_lambda: float
    diffusion_length_voxels: float
    identifiable: bool
    recovered_d: Optional[float]
    recovered_lambda: Optional[float]
    d_relative_error: Optional[float]
    lambda_relative_error: Optional[float]
    r_squared: Optional[float]
    signal_ratio: Optional[float]
    note: str

    def describe(self) -> str:
        if not self.identifiable:
            return (
                f"lambda={self.planted_lambda:.3f}: diffusion length "
                f"{self.diffusion_length_voxels:.2f} voxels -- NOT IDENTIFIABLE "
                f"({self.note})"
            )
        return (
            f"lambda={self.planted_lambda:.3f}: diffusion length "
            f"{self.diffusion_length_voxels:.2f} voxels -- D error "
            f"{self.d_relative_error * 100:.2f}%, lambda error "
            f"{self.lambda_relative_error * 100:.2f}%, R^2 "
            f"{self.r_squared:.5f}, signal {self.signal_ratio:.3f}"
        )


def transport_identifiability_sweep(
    lambdas: Sequence[float] = (0.05, 0.10, 0.20, 0.40, 0.80, 1.60),
    *,
    planted_d: float = 0.35,
    grid_size: int = 32,
    n_times: int = 10,
    sigma_voxels: float = 2.5,
    diffusion_length_min_voxels: float = 1.5,
) -> List[IdentifiabilitySweepRow]:
    """Recovery against clearance, straddling the identifiability boundary.

    Produces exactly the figure a calibration campaign needs: for which drugs is
    this experiment capable of measuring D at all? Above the boundary the
    recovered constants and their errors are reported; below it the row is marked
    not identifiable and carries no error figure, because there is no error to
    report -- only an undefined answer.
    """
    from danon.recovery import (  # local import keeps the module self-referential
        assess_transport_identifiability as _assess,
    )

    out: List[IdentifiabilitySweepRow] = []
    for lam in lambdas:
        diag = _assess(
            planted_d, lam, diffusion_length_min_voxels=diffusion_length_min_voxels
        )
        if not diag.identifiable:
            out.append(
                IdentifiabilitySweepRow(
                    planted_lambda=float(lam),
                    diffusion_length_voxels=diag.diffusion_length_voxels,
                    identifiable=False,
                    recovered_d=None, recovered_lambda=None,
                    d_relative_error=None, lambda_relative_error=None,
                    r_squared=None, signal_ratio=None,
                    note="clearance outruns diffusion: reported as undefined "
                         "rather than fitted",
                )
            )
            continue
        try:
            rec = transport_recovery(
                planted_d=planted_d, planted_lambda=float(lam),
                grid_size=grid_size, n_times=n_times, sigma_voxels=sigma_voxels,
            )
        except (RecoveryFailed, ValueError) as exc:
            out.append(
                IdentifiabilitySweepRow(
                    planted_lambda=float(lam),
                    diffusion_length_voxels=diag.diffusion_length_voxels,
                    identifiable=True,
                    recovered_d=None, recovered_lambda=None,
                    d_relative_error=None, lambda_relative_error=None,
                    r_squared=None, signal_ratio=None,
                    note=f"above the length criterion but not fittable: {exc}",
                )
            )
            continue
        out.append(
            IdentifiabilitySweepRow(
                planted_lambda=float(lam),
                diffusion_length_voxels=rec.diagnostics.diffusion_length_voxels,
                identifiable=True,
                recovered_d=rec.recovered_d,
                recovered_lambda=rec.recovered_lambda,
                d_relative_error=rec.d_relative_error,
                lambda_relative_error=rec.lambda_relative_error,
                r_squared=rec.r_squared,
                signal_ratio=rec.diagnostics.diffusion_to_clearance_signal,
                note="recovered",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Benchmark 2: hazard estimation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoxRecovery:
    """Planted Cox coefficients versus what the holdout pipeline gets back.

    Coefficients are averaged over ``n_cohorts`` independent planted cohorts and
    reported with the Monte Carlo standard error of that mean.

    This is not decoration. A single planted cohort cannot separate *estimator
    bias* from *sampling noise*: one draw of 1,200 patients recovered the planted
    glioblastoma hazard ratio as 4.13 against a planted 3.32 -- a 24% miss that
    looked like a failure and was in fact about 1.8 standard errors on a
    single-cohort fit. Averaging turns that into a measurable bias with a known
    precision, which is the quantity a recovery benchmark is supposed to test.
    """

    planted_betas: Tuple[float, ...]
    planted_hazard_ratios: Tuple[float, ...]
    recovered_hazard_ratios: Tuple[float, ...]
    #: Standard error of each recovered hazard ratio, across cohorts.
    hr_standard_errors: Tuple[float, ...]
    covariate_names: Tuple[str, ...]
    n_patients: int
    n_cohorts: int
    n_events: int
    oracle_c_index: float
    fitted_c_index: float
    fitted_c_index_sd: float
    n_folds: int
    seed: int

    @property
    def hr_relative_errors(self) -> Tuple[float, ...]:
        return tuple(
            abs(r - p) / abs(p)
            for r, p in zip(self.recovered_hazard_ratios, self.planted_hazard_ratios)
        )

    @property
    def max_hr_relative_error(self) -> float:
        return max(self.hr_relative_errors)

    @property
    def hr_sigma_deviations(self) -> Tuple[float, ...]:
        """Recovered-minus-planted in units of the Monte Carlo standard error.

        The quantity the tolerance should be expressed in: a 24% miss is
        meaningless without knowing whether the standard error is 5% or 25%.
        """
        return tuple(
            abs(r - p) / se if se > 0 else float("inf")
            for r, p, se in zip(
                self.recovered_hazard_ratios, self.planted_hazard_ratios,
                self.hr_standard_errors,
            )
        )

    @property
    def max_sigma_deviation(self) -> float:
        return max(self.hr_sigma_deviations)

    @property
    def c_index_loss_vs_oracle(self) -> float:
        """How far the fitted pipeline falls short of the true predictor."""
        return self.oracle_c_index - self.fitted_c_index

    def describe(self) -> str:
        hrs = ", ".join(
            f"{n}: planted {p:.3f} -> recovered {r:.3f} +/- {s:.3f} "
            f"({d:.1f} SE)"
            for n, p, r, s, d in zip(
                self.covariate_names, self.planted_hazard_ratios,
                self.recovered_hazard_ratios, self.hr_standard_errors,
                self.hr_sigma_deviations,
            )
        )
        return (
            f"hazard ratios over {self.n_cohorts} planted cohort(s) ({hrs}); "
            f"C-index oracle {self.oracle_c_index:.4f} vs fitted "
            f"{self.fitted_c_index:.4f} +/- {self.fitted_c_index_sd:.4f} over "
            f"{self.n_folds} folds on {self.n_patients * self.n_cohorts} "
            f"patient-draws ({self.n_events} events in the last cohort); "
            f"shortfall vs oracle {self.c_index_loss_vs_oracle:+.4f}"
        )


#: Stupp-era glioblastoma median overall survival, 14.6 months (NEJM 352:987).
#: The planted cohort is calibrated to this so its structure resembles the real
#: export the pipeline is run against.
REFERENCE_GBM_MEDIAN_OS_DAYS = 444.0


def _mixture_survival(
    baseline_hazard: float, multipliers: np.ndarray, t: float
) -> float:
    """Marginal survival of the heterogeneous cohort at time ``t``.

    Exponential hazards with patient-specific rates ``baseline * exp(X beta)``
    mix to ``E[exp(-h t)]``; the expectation is taken over the sampled
    covariates, so it is exact for this cohort rather than for a density.
    """
    return float(np.mean(np.exp(-baseline_hazard * multipliers * t)))


def _solve_baseline_for_median(
    multipliers: np.ndarray, target_median_days: float
) -> float:
    """Baseline hazard putting the cohort's marginal median at the target."""
    lo, hi = 1e-9, 1e-1
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _mixture_survival(mid, multipliers, target_median_days) > 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _solve_censor_for_fraction(
    event_hazards: np.ndarray, target_fraction: float
) -> float:
    """Censor hazard yielding the target fraction of censored patients.

    For independent exponential censoring, ``P(censored) = E[h / (h + c)]`` with
    ``h`` the event hazard -- monotone decreasing in ``c``, so bisection is exact.
    """
    if not 0.0 < target_fraction < 1.0:
        raise ValueError("target_censoring_fraction must lie strictly in (0, 1)")
    lo, hi = 1e-12, 1e2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        frac = float(np.mean(event_hazards / (event_hazards + mid)))
        if frac > target_fraction:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def planted_cox_cohort(
    *,
    betas: Sequence[float] = (0.030, 0.500, 1.200),
    n_patients: int = 1500,
    target_median_survival_days: float = REFERENCE_GBM_MEDIAN_OS_DAYS,
    target_censoring_fraction: float = 0.45,
    age_mean: float = 60.0,
    age_sd: float = 12.0,
    male_fraction: float = 0.6,
    gbm_fraction: float = 0.85,
    seed: int = 0,
) -> Tuple[List[object], np.ndarray, np.ndarray, np.ndarray]:
    """Simulate survival from a Cox model with known coefficients.

    Returns ``(rows, X, oracle_risk, beta)`` where ``rows`` are the repo's own
    :class:`~danon.predictive.PatientRow` objects, so the *default* pipeline runs
    unmodified against planted truth.

    Covariates come from realistic glioma distributions (age, sex, diagnosis) and
    hazard is ``baseline * exp(X @ beta)``. The rates are **solved, not
    hand-picked**: ``baseline`` is bisected so the cohort's marginal median lands
    on ``target_median_survival_days``, and the censor hazard is bisected so the
    censored fraction lands on ``target_censoring_fraction``.

    Deriving them matters. The first version chose the two rates directly
    (1/600 and 1/1400 per day), which gave a median survival of ~26 days -- six
    weeks, not the glioblastoma it was standing in for -- and censored **nobody**,
    so the entire censoring path of the survival machinery went unexercised by
    the benchmark meant to validate it. The real export is 267 progression events
    among 617 cases, so heavy censoring is a structural feature of the data, not
    an edge case.
    """
    from danon.predictive import PatientRow

    if not 0.0 < target_median_survival_days:
        raise ValueError("target_median_survival_days must be positive")
    b = np.asarray(betas, dtype=float)
    rng = np.random.RandomState(seed)
    age = rng.normal(age_mean, age_sd, n_patients).clip(18.0, 95.0)
    male = (rng.rand(n_patients) < male_fraction).astype(float)
    gbm = (rng.rand(n_patients) < gbm_fraction).astype(float)
    X = np.column_stack([age, male, gbm])
    risk = X @ b
    multipliers = np.exp(risk)

    baseline = _solve_baseline_for_median(multipliers, target_median_survival_days)
    event_hazards = baseline * multipliers
    censor_hazard = _solve_censor_for_fraction(
        event_hazards, target_censoring_fraction
    )

    t_event = rng.exponential(1.0 / event_hazards)
    t_censor = rng.exponential(1.0 / censor_hazard, n_patients)
    observed = np.minimum(t_event, t_censor)
    event = (t_event <= t_censor).astype(int)
    rows = [
        PatientRow(
            submitter_id=f"planted-{i:05d}",
            age_years=float(age[i]),
            male=int(male[i]),
            is_glioblastoma=int(gbm[i]),
            time_days=float(max(observed[i], 1.0)),
            event=int(event[i]),
        )
        for i in range(n_patients)
    ]
    return rows, X, risk, b


def oracle_c_index(
    risk: np.ndarray, times: np.ndarray, events: np.ndarray
) -> float:
    """C-index of the *true* linear predictor -- the ceiling for any estimator.

    Computed with the repo's own concordance implementation, so the oracle and
    the fitted value are the same statistic on the same data and the comparison
    is apples to apples.
    """
    from danon.predictive import harrell_c_index

    c = harrell_c_index(np.asarray(risk, dtype=float), np.asarray(times, dtype=float),
                        np.asarray(events, dtype=int))
    return float(c.c_index) if c.estimable else float("nan")


def cox_recovery(
    *,
    betas: Sequence[float] = (0.030, 0.500, 1.200),
    n_patients: int = 1200,
    n_folds: int = 5,
    n_cohorts: int = 5,
    seed: int = 0,
) -> CoxRecovery:
    """Plant coefficients in ``n_cohorts`` independent cohorts and average.

    Coefficients are averaged in **log** space (the natural scale for a hazard
    ratio, and the scale the fit is linear on), then exponentiated; the standard
    error of the mean is carried through by the delta method. The held-out
    C-index and its oracle are averaged over the same cohorts.
    """
    from danon.predictive import cox_fit, kfold_holdout_c_index

    if n_cohorts < 1:
        raise ValueError("n_cohorts must be at least 1")
    names = ("age_years", "male", "is_glioblastoma")
    log_hrs: List[np.ndarray] = []
    oracle_cs: List[float] = []
    fitted_cs: List[float] = []
    fold_sds: List[float] = []
    n_events = 0
    n_rows = 0
    planted_betas: Tuple[float, ...] = tuple(float(x) for x in betas)

    for k in range(n_cohorts):
        rows, X, risk, b = planted_cox_cohort(
            betas=betas, n_patients=n_patients, seed=seed + k
        )
        planted_betas = tuple(float(x) for x in b)
        times = np.array([r.time_days for r in rows], dtype=float)
        events = np.array([r.event for r in rows], dtype=int)
        n_events = int(events.sum())
        n_rows = len(rows)

        # Full-cohort fit for the coefficients: the recovery question is whether
        # the estimator converges on the planted value; the held-out C-index
        # separately asks whether that estimate generalises.
        fit = cox_fit(X, times, events, names)
        if not fit.converged:
            raise RecoveryFailed(
                f"Cox fit did not converge on planted cohort {k}, so no "
                f"coefficient recovery can be reported"
            )
        # ``CoxFit`` stores log-hazard-ratio coefficients in ``coefficients``.
        beta_arr = np.asarray(fit.coefficients, dtype=float)
        if beta_arr.shape != (len(names),):
            raise RecoveryFailed(
                f"fit returned {beta_arr.shape} coefficients for {len(names)} "
                f"covariates; recovery cannot be compared"
            )
        log_hrs.append(beta_arr)
        oracle_cs.append(oracle_c_index(risk, times, events))
        rep = kfold_holdout_c_index(
            rows, n_folds=n_folds, seed=seed + k, endpoint="os"
        )
        fitted_cs.append(float(rep.mean_multivariable))
        fold_sds.append(float(rep.sd_multivariable))

    stacked = np.vstack(log_hrs)                       # (n_cohorts, n_covariates)
    mean_log_hr = stacked.mean(axis=0)
    recovered_hr = np.exp(mean_log_hr)
    if stacked.shape[0] > 1:
        se_log_hr = stacked.std(axis=0, ddof=1) / math.sqrt(stacked.shape[0])
    else:
        se_log_hr = np.zeros_like(mean_log_hr)
    # Delta method: SE(exp(x)) ~ exp(x) * SE(x).
    se_hr = recovered_hr * se_log_hr

    return CoxRecovery(
        planted_betas=planted_betas,
        planted_hazard_ratios=tuple(float(np.exp(x)) for x in planted_betas),
        recovered_hazard_ratios=tuple(float(x) for x in recovered_hr),
        hr_standard_errors=tuple(float(x) for x in se_hr),
        covariate_names=names,
        n_patients=n_rows,
        n_cohorts=n_cohorts,
        n_events=n_events,
        oracle_c_index=float(np.mean(oracle_cs)),
        fitted_c_index=float(np.mean(fitted_cs)),
        fitted_c_index_sd=float(np.mean(fold_sds)),
        n_folds=n_folds,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Assertions -- the part that makes it a test rather than a report
# ---------------------------------------------------------------------------


def assert_transport_recovery(
    recovery: TransportRecovery,
    *,
    max_relative_error: float = 0.15,
    min_r_squared: float = 0.90,
    min_signal_ratio: float = 1e-2,
) -> None:
    """Raise :class:`RecoveryFailed` if the planted constants did not come back.

    The signal-ratio gate comes first and is the important one. A high R^2 with
    a degenerate design is the failure this benchmark was first written with: it
    reported R^2 = 1.000000 beside a 2400% error on clearance and would have
    looked like a pass to anyone reading the fit quality alone.
    """
    if not recovery.diagnostics.identifiable:
        raise RecoveryFailed(
            f"planted pair is not identifiable, so no recovery can be claimed: "
            f"{recovery.diagnostics.reason}"
        )
    if not math.isfinite(recovery.diagnostics.diffusion_to_clearance_signal) or (
        recovery.diagnostics.diffusion_to_clearance_signal < min_signal_ratio
    ):
        raise RecoveryFailed(
            f"D*Lap carries only "
            f"{recovery.diagnostics.diffusion_to_clearance_signal:.3g} of the "
            f"clearance term's magnitude (need >= {min_signal_ratio:g}): the fit "
            f"cannot constrain D, whatever its R^2 says. "
            f"({recovery.describe()})"
        )
    if not math.isfinite(recovery.r_squared) or recovery.r_squared < min_r_squared:
        raise RecoveryFailed(
            f"transport fit R^2 = {recovery.r_squared:.5f} < {min_r_squared}: the "
            f"engine's discrete operator does not explain the planted field. "
            f"({recovery.describe()})"
        )
    if recovery.max_relative_error > max_relative_error:
        raise RecoveryFailed(
            f"planted constants not recovered within {max_relative_error:.0%}: "
            f"{recovery.describe()}"
        )


def assert_cox_recovery(
    recovery: CoxRecovery,
    *,
    max_sigma_deviation: float = 3.0,
    max_hr_relative_error: Optional[float] = None,
    max_c_index_shortfall: float = 0.05,
) -> None:
    """Raise if the pipeline misses planted coefficients or the oracle ceiling.

    The coefficient tolerance is expressed in **standard errors of the Monte
    Carlo mean**, not as a flat percentage. A 24% miss is not evidence of bias if
    the standard error is 13%, and a flat percentage cannot tell the two apart --
    which is why an earlier version of this benchmark reported a failure that was
    ordinary sampling noise.

    ``max_hr_relative_error`` remains available to force a flat-percentage check,
    and is applied in addition when given.
    """
    # At least 5 cohorts, for a statistical reason rather than a taste one. The
    # standard error is a sample SD over ``n_cohorts`` draws, so with 3 cohorts it
    # has 2 degrees of freedom and is itself extremely noisy: one measurement on
    # 3 cohorts put the male hazard ratio 22 standard errors from its planted
    # value, which reads as decisive bias and was an artefact of a two-df error
    # bar. At 4 cohorts the 95% two-sided t-multiplier is 3.18, so the flat 3.0
    # limit is wrong in the other direction. At 5 cohorts (4 df) the multiplier
    # is 2.78 and 3.0 is conservative.
    if recovery.n_cohorts < 5:
        raise RecoveryFailed(
            f"coefficient recovery needs at least 5 planted cohorts to test bias "
            f"from a {recovery.n_cohorts}-cohort draw: the standard error would "
            f"have {max(recovery.n_cohorts - 1, 0)} degree(s) of freedom and bias "
            f"is indistinguishable from sampling noise"
        )
    if recovery.max_sigma_deviation > max_sigma_deviation:
        raise RecoveryFailed(
            f"a hazard ratio sits {recovery.max_sigma_deviation:.1f} standard "
            f"errors from its planted value (limit {max_sigma_deviation:.1f}): "
            f"this is bias, not sampling noise. ({recovery.describe()})"
        )
    if (
        max_hr_relative_error is not None
        and recovery.max_hr_relative_error > max_hr_relative_error
    ):
        raise RecoveryFailed(
            f"hazard ratios not recovered within {max_hr_relative_error:.0%}: "
            f"{recovery.describe()}"
        )
    if not math.isfinite(recovery.oracle_c_index):
        raise RecoveryFailed("oracle C-index not estimable on planted data")
    if recovery.c_index_loss_vs_oracle > max_c_index_shortfall:
        raise RecoveryFailed(
            f"fitted C-index falls {recovery.c_index_loss_vs_oracle:.4f} below the "
            f"oracle ceiling (limit {max_c_index_shortfall:.4f}): the holdout "
            f"pipeline is losing signal the true predictor has. "
            f"({recovery.describe()})"
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def recovery_provenance() -> Dict[str, EvidenceStatus]:
    """What each recovered quantity is, in the repo's provenance vocabulary."""
    return {
        "transport_operator_recovery": EvidenceStatus.ESTABLISHED,
        "hazard_estimator_recovery": EvidenceStatus.ESTABLISHED,
        "discretisation_error_curve": EvidenceStatus.ESTABLISHED,
        "planted_biology": EvidenceStatus.PROVEN_FAILED,
        "predictive_accuracy_on_cancer": EvidenceStatus.PROVEN_FAILED,
    }


def what_recovery_cannot_prove() -> List[str]:
    """Limits that must travel with any recovery figure."""
    return [
        "Recovering a planted constant is evidence about the estimator, not the "
        "biology. An R^2 of 0.99 here means the pipeline inverts a PDE it was "
        "handed; it is not predictive accuracy on cancer, and presenting it as an "
        "accuracy figure invites a question the benchmark cannot answer.",
        "The planted transport field is a smooth analytic solution on an infinite "
        "domain sampled away from the boundary. Real tissue has heterogeneous "
        "diffusivity, binding, cellular uptake and convection, none of which the "
        "benchmark contains, so a small recovery error says nothing about "
        "transport accuracy in vivo.",
        "The planted Cox cohort has exponential survival, proportional hazards and "
        "no unmeasured confounding. Real progression data has none of those "
        "properties, so recovering the planted coefficients to 1% is not evidence "
        "that the same estimator recovers real coefficients.",
        "A recovery benchmark cannot fail on a bug it does not exercise. It covers "
        "the transport operator and the hazard estimator; it does not cover the "
        "clonal dynamics, the switch rule, or the endpoint translation, each of "
        "which carries its own assumptions.",
        "Success here does not raise the real-cohort numbers. The held-out "
        "discrimination on real patients stays whatever it is, and the two "
        "measurements must be reported separately rather than blended.",
    ]


def render_recovery_report(
    transport: Optional[TransportRecovery] = None,
    cox: Optional[CoxRecovery] = None,
    sweep: Optional[Sequence[IdentifiabilitySweepRow]] = None,
) -> str:
    """Two-baseline summary, with every number measured at call time.

    The measured block and the real-cohort block are kept in separate sections
    on purpose. The real-cohort figures are the ones reported by
    ``python -m danon.audit``; they are restated here so a reader sees both
    standards together, but they are not recomputed by this function and the
    provenance line says where to get them.
    """
    transport = transport or transport_recovery()
    cox = cox or cox_recovery()
    sweep = list(sweep) if sweep is not None else transport_identifiability_sweep()

    def _pct(x: Optional[float]) -> str:
        return "n/a" if x is None else f"{x * 100:.2f}%"

    lines = [
        "RECOVERY AND DISCRIMINATION: TWO STANDARDS, TWO QUESTIONS",
        "=" * 76,
        "",
        "[ STANDARD 1 -- MECHANISTIC GROUND TRUTH ]",
        "  Question: does the pipeline recover constants it planted?",
        "",
        "  Transport operator (planted analytically, recovered by the engine stencil)",
        f"    D       planted {transport.planted_d:.4f}  -> recovered "
        f"{transport.recovered_d:.4f}   ({_pct(transport.d_relative_error)} error)",
        f"    lambda  planted {transport.planted_lambda:.4f}  -> recovered "
        f"{transport.recovered_lambda:.4f}   "
        f"({_pct(transport.lambda_relative_error)} error)",
        f"    fit R^2 {transport.r_squared:.6f} over "
        f"{transport.n_equations:,} equations, cond(A) = "
        f"{transport.condition_number:.1f}",
        f"    grid {transport.grid_shape}, derivative step {transport.dt}, "
        f"series {transport.time_range[0]:.2f}-{transport.time_range[1]:.2f} d, "
        f"blob sigma {transport.source_sigma_voxels:.1f} voxels",
        "",
        "  Identifiability limit (the axis that decides which drugs are",
        "  calibratable at all: diffusion length sqrt(2*D/lambda) vs grid pitch)",
    ]
    lines += [f"    {r.describe()}" for r in sweep]
    lines += [
        "",
        "  Hazard estimator (planted Cox coefficients, exponential survival)",
        f"    {cox.describe()}",
        f"    max hazard-ratio error {_pct(cox.max_hr_relative_error)}",
        f"    C-index shortfall against the oracle ceiling: "
        f"{cox.c_index_loss_vs_oracle:+.4f} -- this is the number that makes the",
        f"    real-cohort C below interpretable: the estimator is not losing signal.",
        "",
        "[ STANDARD 2 -- REAL COHORT ]",
        "  Question: is the model useful on patients? Source: python -m danon.audit",
    ]
    # Computed live rather than transcribed: these numbers moved when the
    # censoring defect in danon.predictive.load_cohort_rows was fixed (the
    # progression C went 0.590 -> 0.534), and a hardcoded copy would have kept
    # printing the stale, flattering value.
    from danon.multimodal import measured_discrimination

    lines += ["    " + ln for ln in measured_discrimination().splitlines()]
    lines += [
        "",
        "  These two blocks answer different questions and must never be blended.",
        "  A high R^2 in block 1 does not raise any number in block 2.",
        "",
        "[ WHAT THE R^2 IS NOT ]",
    ]
    lines += [f"  - {c}" for c in what_recovery_cannot_prove()]
    return "\n".join(lines)
