"""Measure the solver against exact analytic solutions. Nothing here asserts.

Every number printed is either an exact analytic value or a measurement through
the engine's real entry points.
"""

import math

import numpy as np

from danon.anisotropy import divergence_d_grad
from danon.evosteer import (
    DEFAULT_DRUGS,
    BasePolicy,
    PolicyState,
    TumorEvolutionSimulator,
    TumorGeometry,
    _laplacian,
)

print("=" * 78)
print("1. NEUMANN-MODE EIGENVALUE: implementation vs its analytic value")
print("=" * 78)
# f_i = cos(pi k (i+0.5)/N) satisfies the edge-replicated (zero-flux) BCs
# exactly, and is an eigenvector of the 7-point stencil with eigenvalue
# -4 sin^2(pi k / 2N) per axis.  So the residual must be float noise.
N = 24
for k in (1, 3, 7):
    idx = np.arange(N) + 0.5
    f = np.cos(math.pi * k * idx / N)[:, None, None] * np.ones((1, N, N))
    lam = -4.0 * math.sin(math.pi * k / (2.0 * N)) ** 2
    r = _laplacian(f)
    resid = np.linalg.norm(r - lam * f) / max(np.linalg.norm(lam * f), 1e-300)
    print(f"  k={k:2d}  lambda_analytic={lam:+.10f}  residual={resid:.3e}")

print()
print("=" * 78)
print("2. SPATIAL CONVERGENCE ORDER (fixed physical mode, dx = 1/N)")
print("=" * 78)
# Hold the physical domain at length 1 (dx = 1/N, mode k=1 for all N) and
# compare the discrete eigenvalue against the continuum -kappa^2 = -pi^2.
prev = None
for N in (8, 16, 32, 64):
    lam_voxel = -4.0 * math.sin(math.pi / (2.0 * N)) ** 2
    lam_phys = lam_voxel * N * N          # divide by dx^2 = 1/N^2
    rel = abs(lam_phys + math.pi ** 2) / math.pi ** 2
    order = "" if prev is None else f"  observed_order={math.log2(prev / rel):.4f}"
    print(f"  N={N:3d}  lambda_discrete={lam_phys:+.12f}  rel_err={rel:.6e}{order}")
    prev = rel

print()
print("=" * 78)
print("3. MASS CONSERVATION (closed domain, zero-flux)")
print("=" * 78)
rng = np.random.default_rng(0)
for label, f in (
    ("random uniform [0,1)", rng.random((16, 16, 16))),
    ("all ones", np.ones((16, 16, 16))),
    ("single spike", np.eye(16)[None, :, :] * 0 + (np.arange(16)[:, None, None] == 8).astype(float)),
    ("peak at a corner", np.zeros((16, 16, 16)) ),
):
    if label == "peak at a corner":
        f[0, 0, 0] = 1.0
    total = float(_laplacian(f).sum())
    print(f"  {label:24s} sum(L f)={total:+.3e}   scale={abs(f).sum():.1f}")

print()
print("=" * 78)
print("4. THE 'COINCIDES EXACTLY' CLAIM: div(D grad C) vs D * laplacian(C)")
print("=" * 78)
D = 0.35
for label, c in (
    ("random", rng.random((12, 12, 12))),
    ("smooth mode", np.cos(math.pi * (np.arange(12) + 0.5) / 12)[:, None, None]
        * np.ones((1, 12, 12))),
    ("square-root ramp", np.sqrt(np.arange(12))[:, None, None] * np.ones((1, 12, 12))),
):
    a = divergence_d_grad(c, np.full(c.shape, D))
    b = D * _laplacian(c)
    denom = max(np.abs(b).max(), 1e-300)
    print(f"  {label:18s} max|D*div(grad) - D*lap| / scale = {np.abs(a - b).max() / denom:.3e}")

print()
print("=" * 78)
print("5. TRANSPORT THROUGH run(): steady state and the clearance sub-step")
print("=" * 78)


class DosedThenOff(BasePolicy):
    """Full-dose Drug A for `on_days`, then nothing. No tumour dependence."""

    def __init__(self, on_days):
        self.on_days = on_days

    def dose(self, state: PolicyState):
        return (1.0, 0.0) if state.day < self.on_days else (0.0, 0.0)

    def describe(self):
        return f"doseA for {self.on_days}d then off"


size = 10
geom = TumorGeometry(
    tumor_mask=np.ones((size, size, size), dtype=bool),
    proliferative=np.ones((size, size, size)),
    vessels=np.ones((size, size, size)),
    voxel_mm=1.0,
)
ON = 60
mass = []
sim = TumorEvolutionSimulator(geom)
sim.run(
    DosedThenOff(ON),
    days=ON + 6,
    initial_state=(np.full((size,) * 3, 0.65), np.zeros((size,) * 3)),
    growth_noise=0.0,
    field_hook=lambda d, ns, nr, ca, cb: mass.append(float(ca.sum())),
)
lam, src = DEFAULT_DRUGS.clearance[0], DEFAULT_DRUGS.source_strength
print(f"  engine constants: diffusion={DEFAULT_DRUGS.diffusion[0]}  lambda={lam}  src={src}")
n_sub_pred = max(1, math.ceil(1.0 / min(0.4 / (6.0 * DEFAULT_DRUGS.diffusion[0]), 1.0)))
dts_pred = 1.0 / n_sub_pred
print(f"  predicted sub-steps/day = {n_sub_pred}  (dts = {dts_pred:.6f})")

m_off0, m_off3 = mass[ON - 1], mass[ON + 3]
obs = (m_off3 / m_off0) ** (1.0 / 4.0)
best_n = min(range(1, 25), key=lambda n: abs((1.0 - lam / n) ** n - obs))
print(f"  observed per-day decay factor g = {obs:.10f}")
print(f"  engine formula (1-lam/n)^n = {(1.0 - lam / n_sub_pred) ** n_sub_pred:.10f}  "
      f"(n={n_sub_pred})")
print(f"  exact continuum      exp(-lam) = {math.exp(-lam):.10f}")
print(f"  best-fitting n      = {best_n}")
print(f"  realized clearance rate = {-math.log(obs):.8f}/day  "
      f"vs stated {lam}  -> {(math.exp(-lam) - obs) / math.exp(-lam) * 100:+.4f}% per day")

print()
print("=" * 78)
print("6. GROWTH MAP vs THE CONTINUUM LOGISTIC WITH THE SAME r")
print("=" * 78)


class NoDrug(BasePolicy):
    def dose(self, state: PolicyState):
        return (0.0, 0.0)

    def describe(self):
        return "no drug"


r = DEFAULT_DRUGS  # placeholder to keep linters quiet about unused import
from danon.evosteer import DEFAULT_CLONES, CloneParams

r0 = DEFAULT_CLONES.growth_S
x0 = 0.65
clones = CloneParams(growth_S=r0, growth_R=r0, mutation_rate=0.0,
                     carrying_capacity=1.0, drug_sensitivity_scale_R=1.0)
sim2 = TumorEvolutionSimulator(geom, clones=clones)
traj = sim2.run(
    NoDrug(),
    days=365,
    initial_state=(np.full((size,) * 3, x0), np.zeros((size,) * 3)),
    growth_noise=0.0,
    immune=None,
)
burden = traj.burden_S + traj.burden_R
n_vox = size ** 3
print(f"  r={r0}  x0={x0}  voxels={n_vox}  K/voxel=1")
for day in (30, 90, 180, 365):
    x_euler = burden[day - 1] / n_vox
    x_exact = 1.0 / (1.0 + ((1.0 - x0) / x0) * math.exp(-r0 * day))
    print(f"  day {day:3d}  engine={x_euler:.8f}  continuum={x_exact:.8f}  "
          f"rel_err={(x_euler - x_exact) / x_exact * 100:+.4f}%")
