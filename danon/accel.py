"""The same operators, computed without the padding copies or the boolean masks.

WHY THIS EXISTS
---------------
The engine's two hot kernels were the two slowest things in the pipeline, for
reasons that have nothing to do with mathematics:

  * `evosteer._laplacian` called `np.pad(f, 1, mode="edge")` on every one of the
    14 operator calls a simulated day makes. `np.pad` is Python-level: it builds
    a padding spec, slices it into per-axis pairs and copies. Measured on this
    machine that copy is 37.7% of the whole Laplacian at the engine's own grid
    and 13.1% at 256^3 -- paid per call, per day.
  * `anisotropy._harmonic_mean` built two boolean masks, used them to
    fancy-index, and scattered back. A face conductance needs no mask:
    `2ab/(a+b)` is already `0` when either side is `0`, and where both are `0`
    the numerator is `0` too, so `np.divide(..., where=)` leaving those entries
    alone leaves them at exactly zero. Measured at 256^3, `divergence_d_grad`
    cost 10.36 s per call, and the masks were the bulk of it.

WHAT THIS MODULE IS NOT
-----------------------
It is not a new discretisation. Every function here computes the SAME floating
point numbers as the reference implementation it replaces, and that is a claim
the test suite pins with `np.array_equal` (exact equality, not `allclose`) on
random fields, on a real patient field, and across a full 365-day trajectory
compared day by day against the unmodified engine.

WHY BIT-IDENTITY IS THE RIGHT GATE HERE
---------------------------------------
A tolerance would not be enough. The day loop is a nonlinear difference equation
iterated 365 times; a one-ulp difference in a diffusion flux is amplified by
clonal competition and drug kill, and two runs agreeing to 1e-12 on day 1 can
disagree on the progression day. The gate that matters is exact equality of the
trajectory, so that is the gate implemented.

The references this module must match are kept live in their own modules:

  * `danon.evosteer._laplacian_reference`           -- pad + six-term chain
  * `danon.anisotropy._harmonic_mean_reference`     -- masked harmonic mean
  * `danon.anisotropy._divergence_d_grad_reference` -- pad + masked faces
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

#: The last three axes are the volume; any leading axes are batched volumes.
#: Every kernel here is written against the last three axes so that a batch of
#: trajectories is one call rather than a loop (see `evosteer.run_batch`).
_VOL = 3


class KernelWorkspace:
    """Reusable buffers for the kernels, keyed by (name, shape).

    Allocation is not free at a clinical grid: one float64 volume at 256^3 is
    134 MB and a day makes 14 operator calls, so a workspace turns the pad copy
    and the `6 * f` temporary from per-call allocations into per-run ones. It
    also lets a stencil's output buffer be handed back instead of allocated.

    Buffers are returned BY REFERENCE and are invalidated by the next call that
    asks for the same key. Every use inside this module consumes its buffer in
    the call that produced it; a caller that wants to keep a result must copy.
    """

    __slots__ = ("_bufs", "allocations", "hits")

    def __init__(self) -> None:
        self._bufs: Dict[Tuple[str, Tuple[int, ...]], np.ndarray] = {}
        #: Counters, so a test can assert that a warm workspace stops
        #: allocating rather than trusting that it does.
        self.allocations = 0
        self.hits = 0

    def get(self, name: str, shape: Tuple[int, ...]) -> np.ndarray:
        shape = tuple(shape)
        if len(shape) < _VOL:
            raise ValueError(f"need at least 3 axes, got {shape}")
        key = (name, shape)
        buf = self._bufs.get(key)
        if buf is None:
            buf = np.empty(shape, dtype=float)
            self._bufs[key] = buf
            self.allocations += 1
        else:
            self.hits += 1
        return buf

    def padded(self, shape: Tuple[int, ...], name: str = "pad") -> np.ndarray:
        """A padded working buffer. `name` must differ for every operand that has
        to be ALIVE at the same time: two operands sharing a name share a
        buffer, and the second fill silently overwrites the first."""
        shape = tuple(shape)
        return self.get(
            name, tuple(shape[:-3]) + tuple(int(n) + 2 for n in shape[-3:])
        )

    def clear(self) -> None:
        """Drop every buffer, so a long-lived workspace can release memory."""
        self._bufs.clear()


def pad_edge_into(f: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Write the edge-padded `f` into `out`, without `np.pad`.

    One interior copy plus six face copies. `np.pad` also writes the twelve
    edges and eight corners, but no caller in this module reads them: all six
    stencil slices are interior in the two axes they do not shift, so they only
    ever touch faces. Leaving those locations undefined is what makes this cheap
    and why `out` is not a drop-in for code that wants a complete padded array.

    IDENTITY: the interior and all six faces of `out` equal
    `np.pad(f, 1, mode="edge")` exactly.
    """
    if tuple(f.shape[:-3]) != tuple(out.shape[:-3]) or any(
        int(o) != int(n) + 2 for o, n in zip(out.shape[-3:], f.shape[-3:])
    ):
        raise ValueError(f"out {out.shape} is not the padded shape of f {f.shape}")
    out[..., 1:-1, 1:-1, 1:-1] = f
    for axis in range(3):
        for first in (True, False):
            dst = [slice(None, None)] * out.ndim
            src = [slice(None, None)] * f.ndim
            for k in range(3):
                d = out.ndim - 3 + k
                s = f.ndim - 3 + k
                if k == axis:
                    dst[d] = 0 if first else -1
                    src[s] = 0 if first else -1
                else:
                    # A face of the padded array is only the interior of the
                    # other two axes; the padded edges there are never read.
                    dst[d] = slice(1, -1)
            out[tuple(dst)] = f[tuple(src)]
    return out


def laplacian_into(
    f: np.ndarray, out: np.ndarray, workspace: Optional[KernelWorkspace] = None
) -> np.ndarray:
    """Zero-flux (edge-padded) 3D Laplacian, in place, in the reference's order.

    The reference is `sum of six edge-padded neighbours - 6 * f`, evaluated left
    to right. Floating point addition is not associative, so the order is
    reproduced exactly: `(((((a+b)+c)+d)+e)+f) - 6f`. Adding in any other order
    -- for instance the more natural per-axis second differences, which is three
    subtractions instead of five adds -- produces a different last bit and is
    therefore NOT this function. Measured, that alternative is ~2.4x faster and
    is not used for that reason.
    """
    f = np.asarray(f, dtype=float)
    if out.shape != f.shape:
        raise ValueError(f"out shape {out.shape} != f shape {f.shape}")
    ws = workspace if workspace is not None else KernelWorkspace()
    p = pad_edge_into(f, ws.padded(f.shape))
    z0 = (..., slice(None, -2), slice(1, -1), slice(1, -1))
    z1 = (..., slice(2, None), slice(1, -1), slice(1, -1))
    y0 = (..., slice(1, -1), slice(None, -2), slice(1, -1))
    y1 = (..., slice(1, -1), slice(2, None), slice(1, -1))
    x0 = (..., slice(1, -1), slice(1, -1), slice(None, -2))
    x1 = (..., slice(1, -1), slice(1, -1), slice(2, None))
    np.add(p[z0], p[z1], out=out)
    np.add(out, p[y0], out=out)
    np.add(out, p[y1], out=out)
    np.add(out, p[x0], out=out)
    np.add(out, p[x1], out=out)
    np.multiply(f, 6.0, out=ws.get("six", f.shape))
    np.subtract(out, ws.get("six", f.shape), out=out)
    return out


def laplacian(f: np.ndarray, workspace: Optional[KernelWorkspace] = None) -> np.ndarray:
    """Allocating wrapper for `laplacian_into`, matching `evosteer._laplacian`."""
    f = np.asarray(f, dtype=float)
    return laplacian_into(f, np.empty_like(f), workspace)


def harmonic_mean_into(
    a: np.ndarray,
    b: np.ndarray,
    out: np.ndarray,
    denom: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Face conductance `2ab/(a+b)`, zero where either side is zero.

    The reference builds the both-positive mask and fancy-indexes with it, which
    at 256^3 is a gather-scatter of 16.7M doubles per face, six faces per call.
    The same value is reachable arithmetically: `2ab` is `0` whenever either
    input is `0`, and when both are `0` the quotient would be `0/0`, so
    `np.divide(..., where=)` skips exactly those entries and leaves them at the
    zero the multiply already wrote.

    IDENTITY: `fl(fl(2a) * b) / fl(a + b)` per entry, which is the reference's
    order, since Python evaluates `2.0 * a * b` as `(2.0 * a) * b`.

    `a`, `b` and `out` must be distinct arrays; `out` is fully overwritten.
    """
    np.multiply(a, 2.0, out=out)
    np.multiply(out, b, out=out)
    if denom is None:
        denom = np.empty_like(out)
    np.add(a, b, out=denom)
    np.divide(out, denom, out=out, where=denom != 0.0)
    return out


def harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Allocating wrapper for `harmonic_mean_into`, matching the reference."""
    a = np.asarray(a, dtype=float)
    return harmonic_mean_into(a, np.asarray(b, dtype=float), np.empty_like(a))


def divergence_d_grad_into(
    c: np.ndarray,
    d: np.ndarray,
    out: np.ndarray,
    workspace: Optional[KernelWorkspace] = None,
) -> np.ndarray:
    """`div(D(x) grad C)` on a uniform grid with zero-flux boundaries, in place.

    Same finite-volume face formulation as `anisotropy.divergence_d_grad`: per
    axis, harmonic face conductances on the edge-padded field, accumulated as
    `out += f_hi*(c_hi - c_centre) - f_lo*(c_centre - c_lo)` in axis order. What
    changes is only how the pieces are materialised: the padding is one
    reuse-buffer fill instead of two `np.pad` calls, the harmonic means are
    mask-free, and every accumulation is in place.

    `c`, `d` and `out` must be distinct arrays. `out` is fully overwritten.
    """
    c = np.asarray(c, dtype=float)
    d = np.asarray(d, dtype=float)
    if d.shape == c.shape[-3:] and d.ndim == 3 and c.ndim == 4:
        # One coefficient field shared by a batch of volumes: a zero-stride
        # view, so the padding stays the size of the batch and no copy of the
        # field is made per element.
        d = np.broadcast_to(d, c.shape)
    if c.shape != d.shape:
        raise ValueError("c and d must have matching shape")
    if c.ndim not in (3, 4):
        raise ValueError(f"expected a 3D volume or a batch of them, got ndim={c.ndim}")
    if out.shape != c.shape:
        raise ValueError(f"out shape {out.shape} != c shape {c.shape}")
    ws = workspace if workspace is not None else KernelWorkspace()
    cp = pad_edge_into(c, ws.padded(c.shape, "pad_c"))
    dp = pad_edge_into(d, ws.padded(d.shape, "pad_d"))
    centre = (..., slice(1, -1), slice(1, -1), slice(1, -1))
    c_mid = cp[centre]
    d_mid = dp[centre]
    shape = c.shape
    f_lo = ws.get("f_lo", shape)
    f_hi = ws.get("f_hi", shape)
    denom = ws.get("denom", shape)
    t1 = ws.get("t1", shape)
    t2 = ws.get("t2", shape)
    out.fill(0.0)
    for axis in range(3):
        # Face shifts: the shifted axis steps by one, the other two stay in the
        # padded interior -- exactly the slices `_divergence_d_grad_reference`
        # builds on its two padded arrays.
        lo = [slice(None, None)] * cp.ndim
        hi = [slice(None, None)] * cp.ndim
        for k in range(3):
            lo[cp.ndim - 3 + k] = slice(1, -1)
            hi[cp.ndim - 3 + k] = slice(1, -1)
        lo[-3 + axis] = slice(0, -2)
        hi[-3 + axis] = slice(2, None)
        lo = tuple(lo)
        hi = tuple(hi)
        harmonic_mean_into(dp[lo], d_mid, f_lo, denom)
        harmonic_mean_into(d_mid, dp[hi], f_hi, denom)
        np.subtract(cp[hi], c_mid, out=t1)
        np.multiply(t1, f_hi, out=t1)
        np.subtract(c_mid, cp[lo], out=t2)
        np.multiply(t2, f_lo, out=t2)
        np.subtract(t1, t2, out=t1)
        np.add(out, t1, out=out)
    return out


def divergence_d_grad(
    c: np.ndarray,
    d: np.ndarray,
    workspace: Optional[KernelWorkspace] = None,
) -> np.ndarray:
    """Allocating wrapper, matching `anisotropy.divergence_d_grad`."""
    c = np.asarray(c, dtype=float)
    return divergence_d_grad_into(c, d, np.empty(c.shape), workspace)
