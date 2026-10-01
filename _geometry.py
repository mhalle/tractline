"""The predictive streamline encoding, as one statement: quantize, predict, store residuals.

Positions are rounded to `quantum` mm about `origin`. Within each streamline the stored residual
is the first vertex absolute, the second as a first-order delta, and every later vertex as
p[i] - (2 p[i-1] - p[i-2]) (constant-velocity prediction). Each streamline decodes on its own
with two running sums. Residuals go through blosc (bitshuffle, zstd) as int8 when they fit, else
int16; the absolute and first-delta rows as int16 (int32 when the grid is too fine for int16).

Lossless relative to the grid: decode(encode(P)) == round((P - origin) / quantum) exactly.
"""
import numpy as np
from numcodecs import Blosc


def line_index(off):
    """Each vertex's index within its streamline, from (N + 1,) offsets."""
    return np.arange(off[-1]) - np.repeat(off[:-1], np.diff(off))


def residuals(Q, k, order=2):
    """Integer positions (V, 3) -> residuals of the given prediction order (1 or 2)."""
    R = np.empty_like(Q)
    R[0] = Q[0]
    R[1:] = Q[1:] - Q[:-1]
    if order == 2:
        R2 = R.copy()
        R2[2:] = Q[2:] - 2 * Q[1:-1] + Q[:-2]
        R2[k == 1] = R[k == 1]
        R = R2
    R[k == 0] = Q[k == 0]
    return R


def _segment_cumsum(x, off):
    c = np.cumsum(x, axis=0)
    before = np.vstack([np.zeros((1, x.shape[1]), x.dtype), c[off[:-1] - 1][1:]])
    return c - np.repeat(before, np.diff(off), axis=0)


def decode(R, k, off, order=2):
    """Residuals -> integer positions: the reader's running sums, per streamline."""
    first = np.repeat(R[k == 0], np.diff(off), axis=0)
    D = R.copy()
    D[k == 0] = 0
    if order == 2:
        D = _segment_cumsum(D, off)          # the steps: D[1] is stored, later D[i] = D[i-1] + r
        D[k == 0] = 0
    return first + _segment_cumsum(D, off)


def encode(P, off, quantum, origin, order=2, clevel=9):
    """Positions (V, 3) float, offsets (N + 1,) -> (compressed bytes, integer grid positions).
    The byte count covers the residuals only; streamline lengths are counted separately."""
    k = line_index(off)
    Q = np.round((P - origin) / quantum).astype(np.int64)
    R = residuals(Q, k, order)
    assert (decode(R, k, off, order) == Q).all(), "round trip failed"
    body, head = R[k >= order], R[k < order]
    bdt = "<i1" if np.abs(body).max() < 128 else "<i2"
    hdt = "<i2" if np.abs(head).max() < 32768 else "<i4"
    z = lambda a, s: len(Blosc(cname="zstd", clevel=clevel, shuffle=s).encode(np.ascontiguousarray(a)))
    return z(body.astype(bdt), Blosc.BITSHUFFLE) + z(head.astype(hdt), Blosc.SHUFFLE), Q


def lengths_bytes(off, clevel=9):
    """Streamline lengths as uint16 under the same coder: what a reader needs besides residuals."""
    return len(Blosc(cname="zstd", clevel=clevel, shuffle=Blosc.SHUFFLE).encode(np.diff(off).astype("<u2")))
