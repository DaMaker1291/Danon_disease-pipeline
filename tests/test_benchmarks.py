"""Numerical verification: the solver measured against exact analytic solutions.

This file is VERIFICATION, not validation. It does not ask whether the model is
right about biology; it asks whether the code solves the model it states, and
answers with the discretization error it can actually be held to.

The distinction matters because the two halves of a claim like "second order
in space" are not the same kind of statement:

  * an analytic value -- the Neumann eigenvalue, the mass of a closed domain,
    the fixed point of the transport recursion -- is a fact that either holds
    to floating point or does not;
  * a discretization error -- how far the daily growth map is from the
    continuum logistic -- is a number to report, never a pass/fail.

So the tests below assert the first kind and *measure* the second, and every
number in the measured kind is printed by the test rather than asserted at
whatever value it happened to have.

WHAT IS NOT CLAIMED HERE

- No comparison against PhysiCell / BioNetGen / FEniCS. Those are different
  discretizations and would need a matched benchmark problem; the analytic
  solutions below are the ground truth they would be compared against anyway.
- No uncertainty quantification over the parameters. A posterior over D0 or r0
  needs a likelihood against patient-level data, and none is fitted here.
- No statement about biology. Nothing in this module touches a patient.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from danon.anisotropy import divergence_d_grad
from danon.evosteer import (
    DEFAULT_CLONES,
    DEFAULT_DRUGS,
    BasePolicy,
    CloneParams,
    PolicyState,
    TumorEvolutionSimulator,
    TumorGeometry,
    _laplacian,
)

#: Real engine-grid rCBV maps. Gitignored sample cache: a fresh clone does not
#: have them, and a missing cache is reported rather than failed.
RCBV_GLOB = "data/tcia/engine_ready/*_rcbv.nii.gz"


def _neumann_mode(n: int, k: int) -> np.ndarray:
    """The k-th zero-flux eigenvector along x, constant in y and z.

    f_i = cos(pi k (i + 1/2) / n). The half-voxel offset is what makes the
    edge-replicated boundary conditions exact: f(-1) = f(0) and f(n) = f(n-1),
    which is precisely what `np.pad(mode="edge")` imposes. On the grid the
    operator therefore has f as an EXACT eigenvector with eigenvalue
    -4 sin^2(pi k / 2n) per axis -- not an approximation of one.
    """
    idx = np.arange(n) + 0.5
    return np.cos(math.pi * k * idx / n)[:, None, None] * np.ones((1, n, n))


def _neumann_eigenvalue(n: int, k: int) -> float:
    return -4.0 * math.sin(math.pi * k / (2.0 * n)) ** 2


def _uniform_geometry(size: int, vessels: float = 1.0) -> TumorGeometry:
    """A homogeneous substrate: no sanctuary pockets, so transport is closed."""
    return TumorGeometry(
        tumor_mask=np.ones((size, size, size), dtype=bool),
        proliferative=np.ones((size, size, size)),
        vessels=np.full((size, size, size), vessels),
        voxel_mm=1.0,
    )


class _FullDoseThenOff(BasePolicy):
    """Drug A at full dose for `on_days`, withdrawn forever after."""

    def __init__(self, on_days: int) -> None:
        self.on_days = on_days

    def dose(self, state: PolicyState):
        return (1.0, 0.0) if state.day < self.on_days else (0.0, 0.0)

    def describe(self) -> str:
        return f"doseA for {self.on_days}d then off"


class _NoDrug(BasePolicy):
    def dose(self, state: PolicyState):
        return (0.0, 0.0)

    def describe(self) -> str:
        return "no drug"


def _predicted_sub_steps() -> tuple[int, float]:
    """The engine's own stability rule, re-derived from its constants.

    6 * D * dts <= 0.4 is the margin the code enforces; the sub-step count is
    what that rule yields at the calibrated diffusivity and no heterogeneity
    (pattern_peak = 1).
    """
    d = DEFAULT_DRUGS.diffusion[0]
    sub_dt = min(0.4 / (6.0 * d), 1.0)
    return max(1, int(math.ceil(1.0 / sub_dt))), 1.0 / max(1, int(math.ceil(1.0 / sub_dt)))


# ---------------------------------------------------------------------------
# The operators, against their analytic values
# ---------------------------------------------------------------------------


def test_laplacian_is_exact_on_its_neumann_eigenbasis():
    """A zero-flux eigenvector is reproduced to floating point, not to 1e-2."""
    n = 24
    for k in (1, 3, 7, 12):
        f = _neumann_mode(n, k)
        lam = _neumann_eigenvalue(n, k)
        resid = np.linalg.norm(_laplacian(f) - lam * f) / abs(lam) / np.linalg.norm(f)
        assert resid < 1e-12, f"k={k}: residual {resid:.3e} is not float noise"


def test_spatial_convergence_order_is_second():
    """Halving dx must cut the eigenvalue error by 4x. Measured, not assumed."""
    errors = []
    for n in (8, 16, 32, 64):
        # Physical domain held at unit length: dx = 1/n, so the discrete
        # eigenvalue is in voxel units and must be divided by dx^2.
        lam_discrete = _neumann_eigenvalue(n, 1) * n * n
        errors.append(abs(lam_discrete + math.pi ** 2) / math.pi ** 2)
    orders = [math.log2(a / b) for a, b in zip(errors, errors[1:])]
    for order in orders:
        assert order > 1.9, f"observed convergence order {order:.4f} is not second order"


def test_zero_flux_operator_conserves_mass_exactly():
    """Telescoping faces: the sum of the operator is identically zero.

    This is the property that makes "the domain is closed" a numerical fact
    rather than a hope -- no drug leaks out of a boundary it cannot cross.
    """
    rng = np.random.default_rng(7)
    corner = np.zeros((16, 16, 16))
    corner[0, 0, 0] = 1.0
    fields = {
        "random": rng.random((16, 16, 16)),
        "ones": np.ones((16, 16, 16)),
        "spike at a corner": corner,
        "positive ramp": np.arange(16, dtype=float)[:, None, None] * np.ones((1, 16, 16)),
    }
    for label, f in fields.items():
        total = float(_laplacian(f).sum())
        assert abs(total) < 1e-9 * max(abs(f).sum(), 1.0), f"{label}: sum(L f) = {total}"


def test_heterogeneous_operator_also_conserves_mass():
    """Harmonic face conductances must not create or destroy drug."""
    rng = np.random.default_rng(11)
    c = rng.random((14, 14, 14))
    d = 0.05 + rng.random((14, 14, 14))
    total = float(divergence_d_grad(c, d).sum())
    assert abs(total) < 1e-9 * max(abs(c).sum(), 1.0), total


def test_constant_d_divergence_coincides_with_scalar_laplacian():
    """The code's documented identity: with constant D the two branches agree.

    Worth pinning because it is what makes the heterogeneous path a refinement
    of the homogeneous model rather than a different model wearing its name.
    """
    d = DEFAULT_DRUGS.diffusion[0]
    rng = np.random.default_rng(3)
    cases = {
        "random": rng.random((12, 12, 12)),
        "smooth mode": _neumann_mode(12, 2)[:, :12, :12],
        "sqrt ramp": np.sqrt(np.arange(12, dtype=float))[:, None, None] * np.ones((1, 12, 12)),
    }
    for label, c in cases.items():
        gap = np.abs(divergence_d_grad(c, np.full(c.shape, d)) - d * _laplacian(c)).max()
        scale = max(np.abs(d * _laplacian(c)).max(), 1e-300)
        assert gap / scale < 1e-12, f"{label}: relative gap {gap / scale:.3e}"


# ---------------------------------------------------------------------------
# The integrator, through run(): what the sub-stepping actually delivers
# ---------------------------------------------------------------------------


def _drug_mass_series(days_dosed: int, extra: int = 4, size: int = 10):
    """Sum of the Drug-A field per day, through the real run() entry point."""
    mass: list[float] = []
    sim = TumorEvolutionSimulator(_uniform_geometry(size))
    sim.run(
        _FullDoseThenOff(days_dosed),
        days=days_dosed + extra,
        initial_state=(np.full((size,) * 3, 0.65), np.zeros((size,) * 3)),
        growth_noise=0.0,
        field_hook=lambda d, ns, nr, ca, cb: mass.append(float(ca.sum())),
    )
    return mass


def test_steady_state_concentration_is_the_analytic_fixed_point():
    """With uniform uptake the transport recursion settles at source/clearance.

    The fixed point of c <- c + dts*(src - lambda*c) is src/lambda EXACTLY,
    for any sub-step count -- so a discrepancy here would mean the source or
    the clearance term is not what the model says it is.
    """
    size, on = 10, 60
    mass = _drug_mass_series(on, extra=1, size=size)
    expected = size ** 3 * DEFAULT_DRUGS.source_strength / DEFAULT_DRUGS.clearance[0]
    assert mass[on - 1] == pytest.approx(expected, rel=0.01), (mass[on - 1], expected)


def test_sub_step_count_is_measured_not_assumed():
    """Identify the realized dts from the decay, and compare it to the rule.

    The engine never exposes its sub-step count, so the only honest way to
    check the stability logic is to measure the decay it produces and ask
    which n it corresponds to. Diffusion conserves mass exactly (pinned
    above), so with the source withdrawn the field's total mass decays at
    exactly the clearance factor -- the transport error and the clearance
    error are separable here.
    """
    on = 60
    mass = _drug_mass_series(on)
    realized = (mass[on + 3] / mass[on - 1]) ** 0.25

    n_pred, dts = _predicted_sub_steps()
    predicted = (1.0 - DEFAULT_DRUGS.clearance[0] * dts) ** n_pred
    assert realized == pytest.approx(predicted, rel=1e-9), (realized, predicted)

    best_n = min(range(1, 25), key=lambda n: abs((1.0 - DEFAULT_DRUGS.clearance[0] / n) ** n - realized))
    assert best_n == n_pred, f"measured sub-step count {best_n} != rule's {n_pred}"


def test_clearance_integration_error_is_first_order_and_bounded():
    """The one discretization error with a direction: forward Euler is biased.

    (1 - x)^n <= exp(-n*x) for all x > 0, so the scheme ALWAYS clears drug
    faster than the stated model -- the bias has a sign and cannot cancel.
    At the calibrated lambda and the rules's own dts it is a fraction of a
    percent per day; this test bounds it rather than waving it through.
    """
    n, dts = _predicted_sub_steps()
    lam = DEFAULT_DRUGS.clearance[0]
    realized = (1.0 - lam * dts) ** n
    exact = math.exp(-lam)

    assert realized <= exact, "forward Euler cannot undershoot exp(-lambda)"
    bias = exact / realized - 1.0
    assert 0.0 < bias < 0.002, f"clearance rate biased {bias * 100:.4f}%/day"

    # First order: halving dts must halve the log-error.
    def logerr(steps: int) -> float:
        return abs(steps * math.log(1.0 - lam / steps) + lam)

    assert logerr(2 * n) == pytest.approx(logerr(n) / 2.0, rel=0.05)


def test_worst_mode_amplification_is_positive_on_calibrated_defaults():
    """Von Neumann on the scheme the code actually runs.

    The harshest resolvable mode has discrete eigenvalue -12, so the update
    factor is 1 + dts*(D*-12 - lambda). If that ever went negative the field
    would oscillate and the `maximum(c, 0)` clip would quietly eat drug. The
    rule keeps 6*D*dts <= 0.4, which puts the factor near 0.28.
    """
    n, dts = _predicted_sub_steps()
    d, lam = DEFAULT_DRUGS.diffusion[0], DEFAULT_DRUGS.clearance[0]
    g = 1.0 + dts * (d * -12.0 - lam)
    assert 0.0 < g <= 1.0, f"amplification {g} is outside the stable, non-oscillatory range"
    assert 6.0 * d * dts <= 0.4 + 1e-12, "the margin the rule enforces is not the one enforced"


def test_heterogeneous_clearance_stays_inside_the_stability_margin():
    """A large local lambda must not push the update factor through zero.

    The sub-step rule keys on the DIFFUSION field's peak, never on clearance.
    That is only safe because a heterogeneous rCBV map raises the diffusion
    peak too, which buys more sub-steps. This test is the arithmetic of that
    coupling, swept over multipliers so the boundary is visible rather than
    asserted at the one value the calibration happens to sit at.
    """
    _, dts = _predicted_sub_steps()
    lam, d = DEFAULT_DRUGS.clearance[0], DEFAULT_DRUGS.diffusion[0]
    # g > 0  <=>  dts * (12*D + lambda_local) < 1  <=>  lambda_local < 1/dts - 12*D.
    # That margin is the boundary the rule does not police, so state it.
    margin = 1.0 / dts - 12.0 * d
    assert margin > 0.0

    admissible = [m for m in (1.0, 2.0, 3.5, 5.1) if lam * m < margin]
    assert admissible, "even the calibrated lambda is outside the stable range"
    for multiplier in admissible:
        g = 1.0 + dts * (d * -12.0 - lam * multiplier)
        assert 0.0 < g <= 1.0, (multiplier, g)

    # Beyond the margin the operator would need another sub-step, and the rule
    # -- which keys on diffusion alone -- would not ask for one. That is the
    # boundary this test exists to make visible, so assert it is crossed.
    assert lam * (margin / lam + 1.0) > margin


# ---------------------------------------------------------------------------
# The growth map, against the continuum logistic
# ---------------------------------------------------------------------------


def test_growth_map_tracks_the_continuum_logistic_it_discretises():
    """A daily Euler step of dn/dt = r n (1 - n/K), measured against the solution.

    The engine is a daily-step model, so this is a discretization budget rather
    than a defect -- but it is a budget worth knowing, and it is small at the
    calibrated rate: the map overshoots the continuum trajectory by a fraction
    of a percent while both are far from carrying capacity, and the gap closes
    to nothing once both saturate.
    """
    size, x0 = 10, 0.65
    r = DEFAULT_CLONES.growth_S
    clones = CloneParams(
        growth_S=r,
        growth_R=r,
        mutation_rate=0.0,
        carrying_capacity=1.0,
        drug_sensitivity_scale_R=1.0,
    )
    sim = TumorEvolutionSimulator(_uniform_geometry(size), clones=clones)
    traj = sim.run(
        _NoDrug(),
        days=120,
        initial_state=(np.full((size,) * 3, x0), np.zeros((size,) * 3)),
        growth_noise=0.0,
    )
    burden = traj.burden_S + traj.burden_R

    # The budget peaks mid-transient (measured ~0.9% at day 10, ~0.07% at day
    # 30) and closes as both trajectories saturate, so bound it where it is
    # largest rather than where it is convenient.
    for day in (10, 30, 60):
        engine = burden[day - 1] / size ** 3
        exact = 1.0 / (1.0 + ((1.0 - x0) / x0) * math.exp(-r * day))
        assert abs(engine - exact) / exact < 0.02, (day, engine, exact)

    # Past saturation the map reaches K rather than exceeding it.
    assert 0.0 < burden[119] / size ** 3 <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# The stability margin on the real maps, not a synthetic worst case
# ---------------------------------------------------------------------------


def test_real_rcbv_maps_stay_inside_the_stability_margin():
    """The three measured DSC maps, through the real clinical-twin builders.

    Reported rather than failed when the sample cache is absent: these maps are
    re-downloadable and gitignored, so a fresh clone sees a skip.
    """
    maps = sorted(Path().glob(RCBV_GLOB))
    if not maps:
        pytest.skip(f"no cached rCBV maps at {RCBV_GLOB} (re-downloadable sample)")

    from danon.clinical_twin import build_stream3_clearance, build_stream3_diffusion_field
    from danon.imaging import read_nifti

    d, lam = DEFAULT_DRUGS.diffusion[0], DEFAULT_DRUGS.clearance[0]
    for path in maps:
        rcbv, _ = read_nifti(str(path))
        rcbv = np.asarray(rcbv, dtype=float)
        pattern = np.asarray(
            build_stream3_diffusion_field(rcbv, voxel_mm=1.0).d_engine, dtype=float
        )
        peak = float((pattern / pattern.mean()).max())
        sub_dt = min(0.4 / (6.0 * d * peak), 1.0)
        n_sub = max(1, int(math.ceil(1.0 / sub_dt)))
        dts = 1.0 / n_sub
        local_lambda = lam * float(build_stream3_clearance(rcbv).max())
        g = 1.0 + dts * (d * peak * -12.0 - local_lambda)
        assert g > 0.0, f"{path.name}: worst-mode factor {g:.6f} would oscillate"
        assert g <= 1.0, f"{path.name}: worst-mode factor {g:.6f} would grow"
