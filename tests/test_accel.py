"""The accelerated kernels must be a speed change and nothing else.

The gate here is EXACT equality, not a tolerance. The day loop is a nonlinear
difference equation iterated for hundreds of days: a one-ulp change in a
diffusion flux is amplified by clonal competition and drug kill, so two runs that
agree to 1e-12 on day one can disagree on the progression day. A tolerance would
let that through, and every number downstream of the engine would then be a
number from a slightly different model.

So the tests are:

* every accelerated operator equals its reference entry for entry, on random
  fields, on degenerate fields (all-zero, all-equal, a field with a zero block),
  and on a real patient transport field;
* a 365-day trajectory run through the accelerated engine equals, day by day and
  field by field, the same run with the reference kernels put back -- on the
  scalar path AND on the heterogeneous-transport path;
* the workspace REUSES its buffers, which is what makes it fast, and does not
  leak one caller's data into the next.

A test that cannot fail is not a test: the equality assertions are checked
against a deliberately different operator as a control, so a stencil that
silently became a no-op would be caught rather than passing trivially.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import danon.evosteer as evosteer
from danon import accel
from danon.anisotropy import (
    _divergence_d_grad_reference,
    _harmonic_mean_reference,
)
from danon.clinical_twin import build_stream3_diffusion_field
from danon.evosteer import (
    AdaptiveDosePolicy,
    MTDPolicy,
    TumorEvolutionSimulator,
    _laplacian_reference,
    make_gbm_phantom,
)

TRUE_VOXEL_MM = 3.634
ENGINE_GRID = (14, 24, 24)
CASE = "BraTS2021_01017"
ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# The operator is the operator it replaces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(14, 24, 24), (7, 9, 11), (3, 5, 5), (2, 6, 7)])
def test_laplacian_is_bit_identical_to_the_padded_reference(shape):
    rng = np.random.default_rng(7)
    f = rng.random(shape) * 0.7
    want = _laplacian_reference(f)
    got = np.empty_like(f)
    accel.laplacian_into(f, got, accel.KernelWorkspace())
    assert np.array_equal(want, got)
    # Not merely equal under `==`: equal as bit patterns, which also pins the
    # sign of a zero and would catch a reassociated sum that happens to agree.
    assert np.array_equal(want.view(np.uint64), got.view(np.uint64))


def test_laplacian_on_a_batch_equals_it_element_by_element():
    rng = np.random.default_rng(11)
    f = rng.random((3,) + ENGINE_GRID)
    want = np.stack([_laplacian_reference(x) for x in f])
    got = accel.laplacian(f, accel.KernelWorkspace())
    assert np.array_equal(want, got)


@pytest.mark.parametrize(
    "name,field",
    [
        ("zeros", lambda s: np.zeros(s)),
        ("constant", lambda s: np.full(s, 0.35)),
        ("zero-block", None),
        ("spike", None),
    ],
)
def test_laplacian_agrees_on_degenerate_fields(name, field):
    rng = np.random.default_rng(3)
    s = (6, 7, 8)
    if name == "zero-block":
        f = rng.random(s)
        f[:2, :2, :2] = 0.0
    elif name == "spike":
        f = np.zeros(s)
        f[3, 3, 3] = 1.0
    else:
        f = field(s)
    assert np.array_equal(
        _laplacian_reference(f), accel.laplacian(f, accel.KernelWorkspace())
    )


def test_a_zero_flux_laplacian_conserves_the_sum():
    """The control: a different operator would NOT pass the equality tests above."""
    f = np.random.default_rng(5).random(ENGINE_GRID)
    lap = accel.laplacian(f, accel.KernelWorkspace())
    assert abs(float(lap.sum())) < 1e-9
    # And the reference's own identity still holds under the accelerated path.
    assert np.array_equal(_laplacian_reference(f), lap)


@pytest.mark.parametrize("shape", [(14, 24, 24), (6, 7, 9)])
def test_divergence_matches_the_masked_reference(shape):
    rng = np.random.default_rng(13)
    c = rng.random(shape)
    d = np.abs(rng.random(shape)) * 0.1
    want = _divergence_d_grad_reference(c, d)
    got = accel.divergence_d_grad(c, d, accel.KernelWorkspace())
    assert np.array_equal(want, got)
    assert np.array_equal(want.view(np.uint64), got.view(np.uint64))


def test_divergence_agrees_where_the_field_has_a_dead_region():
    """A zero block is the case the mask existed for: `0/0`, not a small number."""
    rng = np.random.default_rng(17)
    c = rng.random(ENGINE_GRID)
    d = np.abs(rng.random(ENGINE_GRID)) * 0.1
    d[:4, :, :] = 0.0
    want = _divergence_d_grad_reference(c, d)
    got = accel.divergence_d_grad(c, d, accel.KernelWorkspace())
    assert np.array_equal(want, got)
    assert np.isfinite(got).all()


def test_divergence_with_a_constant_field_is_d_times_the_laplacian():
    """The property the module docstring of `anisotropy` leans on, re-derived.

    `allclose`, not equality: on a constant field the harmonic-mean formulation
    and `d * laplacian` differ by 1-2 ulp of rounding ORDER, and the reference
    fails an exact comparison here too (measured, max relative 2.6e-16). The
    exact gate belongs on accelerated-versus-reference, which the tests above
    apply; this test is about the model, not about the kernel.
    """
    rng = np.random.default_rng(19)
    c = rng.random((8, 8, 8))
    d = 0.35
    got = accel.divergence_d_grad(c, np.full_like(c, d), accel.KernelWorkspace())
    assert np.allclose(got, d * _laplacian_reference(c), atol=1e-12)
    # The kernel itself is still exact against the reference it replaces.
    assert np.array_equal(got, _divergence_d_grad_reference(c, np.full_like(c, d)))


def test_divergence_accepts_one_field_shared_by_a_batch():
    rng = np.random.default_rng(23)
    c = rng.random((2, 8, 9, 10))
    d = np.abs(rng.random((8, 9, 10))) * 0.1
    got = accel.divergence_d_grad(c, d, accel.KernelWorkspace())
    for i in range(2):
        assert np.array_equal(
            got[i], _divergence_d_grad_reference(c[i], d)
        ), "a shared field must be applied identically to every batched volume"


def test_harmonic_mean_is_the_masked_one_without_the_mask():
    rng = np.random.default_rng(29)
    a = np.abs(rng.random((5, 6, 7)))
    b = np.abs(rng.random((5, 6, 7)))
    a[0, 0, 0] = 0.0
    b[1, 2, 3] = 0.0
    a[4, 5, 6] = b[4, 5, 6] = 0.0
    want = _harmonic_mean_reference(a, b)
    got = accel.harmonic_mean_into(a, b, np.empty_like(a))
    assert np.array_equal(want, got)
    assert got[0, 0, 0] == 0.0 and got[1, 2, 3] == 0.0 and got[4, 5, 6] == 0.0


def test_a_non_physical_input_is_still_refused():
    with pytest.raises(ValueError):
        accel.laplacian(np.zeros((4, 4)), accel.KernelWorkspace())  # 2D has no volume
    with pytest.raises(ValueError):
        accel.divergence_d_grad(
            np.zeros((4, 4, 4)), np.zeros((4, 4)), accel.KernelWorkspace()
        )
    with pytest.raises(ValueError):
        accel.laplacian_into(np.zeros((4, 4, 4)), np.zeros((5, 5, 5)))


# ---------------------------------------------------------------------------
# The workspace: reuse is the point, and it must not leak
# ---------------------------------------------------------------------------


def test_a_warm_workspace_stops_allocating():
    ws = accel.KernelWorkspace()
    f = np.random.default_rng(2).random(ENGINE_GRID)
    accel.laplacian(f, ws)
    after_first = ws.allocations
    assert after_first > 0
    for _ in range(20):
        accel.laplacian(f, ws)
    assert ws.allocations == after_first, "a warm workspace must allocate nothing"
    assert ws.hits >= 20


def test_the_workspace_does_not_leak_one_call_into_the_next():
    """Reuse is only safe if every buffer is fully written before it is read."""
    ws = accel.KernelWorkspace()
    rng = np.random.default_rng(31)
    for _ in range(5):
        f = rng.random(ENGINE_GRID)
        fresh = accel.laplacian(f, accel.KernelWorkspace())
        reused = accel.laplacian(f, ws)
        assert np.array_equal(fresh, reused)
        c = rng.random(ENGINE_GRID)
        d = np.abs(rng.random(ENGINE_GRID)) * 0.2
        d[:5] = 0.0
        assert np.array_equal(
            _divergence_d_grad_reference(c, d), accel.divergence_d_grad(c, d, ws)
        )


def test_two_operands_are_not_given_the_same_buffer():
    """The failure this guards is silent: a shared pad buffer overwrites `c` with
    `d` and the operator then returns a plausible, wrong field."""
    ws = accel.KernelWorkspace()
    rng = np.random.default_rng(37)
    c = rng.random((5, 6, 7))
    d = np.full((5, 6, 7), 0.3)
    assert np.array_equal(
        accel.divergence_d_grad(c, d, ws), _divergence_d_grad_reference(c, d)
    )


# ---------------------------------------------------------------------------
# A whole trajectory, against the same trajectory on the reference kernels
# ---------------------------------------------------------------------------

DATA = ROOT / "data" / "engine_ready" / f"{CASE}_mask.nii.gz"
pytestmark_data = pytest.mark.skipif(
    not DATA.exists(), reason=f"{DATA} not present"
)


def _real_geometry():
    from danon.imaging import from_nifti_mask

    return from_nifti_mask(
        str(DATA),
        proliferative_path=str(DATA.with_name(f"{CASE}_prolif.nii.gz")),
        vessels_path=str(DATA.with_name(f"{CASE}_vessels_t1ce.nii.gz")),
    ).geometry


@pytestmark_data
@pytest.mark.parametrize("heterogeneous", [False, True])
def test_a_365_day_trajectory_is_unchanged_by_the_accelerated_kernels(heterogeneous):
    geometry = _real_geometry()
    field = None
    if heterogeneous:
        field = build_stream3_diffusion_field(geometry.vessels, voxel_mm=TRUE_VOXEL_MM)

    def run_once():
        sim = TumorEvolutionSimulator(geometry, diffusion_field=field)
        return sim.run(
            AdaptiveDosePolicy(),
            days=365,
            seed=11,
            initial_resistant_fraction=0.01,
            keep_fields=True,
        )

    fast = run_once()
    keep_lap, keep_div = evosteer._laplacian, evosteer.divergence_d_grad
    evosteer._laplacian = lambda f, out=None, workspace=None: _laplacian_reference(f)
    evosteer.divergence_d_grad = (
        lambda c, d, workspace=None, out=None: _divergence_d_grad_reference(c, d)
    )
    try:
        slow = run_once()
    finally:
        evosteer._laplacian, evosteer.divergence_d_grad = keep_lap, keep_div

    assert np.array_equal(fast.burden_S, slow.burden_S)
    assert np.array_equal(fast.burden_R, slow.burden_R)
    assert np.array_equal(fast.dose_A, slow.dose_A)
    assert fast.pfs_day == slow.pfs_day
    assert fast.toxicity == slow.toxicity
    assert fast.final_resistant_fraction == slow.final_resistant_fraction
    assert fast.resistant_dominance_day == slow.resistant_dominance_day
    assert np.array_equal(
        fast.final_N_S.view(np.uint64), slow.final_N_S.view(np.uint64)
    )
    assert np.array_equal(
        fast.final_conc_A.view(np.uint64), slow.final_conc_A.view(np.uint64)
    )


def test_a_phantom_run_is_unchanged_too():
    """Same gate at the synthetic geometry every other test uses."""
    geom = make_gbm_phantom(size=16, seed=0)

    def run_once():
        return TumorEvolutionSimulator(geom).run(
            MTDPolicy(), days=180, seed=5, initial_resistant_fraction=0.01
        )

    fast = run_once()
    keep = evosteer._laplacian
    evosteer._laplacian = lambda f, out=None, workspace=None: _laplacian_reference(f)
    try:
        slow = run_once()
    finally:
        evosteer._laplacian = keep
    assert np.array_equal(fast.burden_S, slow.burden_S)
    assert np.array_equal(fast.burden_R, slow.burden_R)
    assert fast.pfs_day == slow.pfs_day
