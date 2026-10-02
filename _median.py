"""DIPY's median_otsu, exactly and fast: the same four passes of scipy.ndimage.median_filter (a 9^3
box, mode "reflect": d c b a | a b c d | d c b a) and the same Otsu threshold, the median found by
Huang's sliding-window histogram instead of a full sort.

  - values replaced by their ranks among the volume's distinct values (np.unique): the median of
    ranks is the rank of the median, and a median is always one of the window's values, so every
    pass stays in rank space and the result maps back exactly;
  - per (y, z) row, the window slides along x: the 81 values leaving and the 81 entering update a
    histogram over ranks, the median is walked from its last position (Huang, Yang & Tang 1979);
    rows in parallel (numba prange);
  - dipy.segment.threshold.otsu on the filtered values, as median_otsu.
Checked voxel for voxel against dipy.segment.mask.median_otsu (median_check.py).
"""
import numpy as np
from numba import njit, prange
from dipy.segment.threshold import otsu


def _reflect(n, r):
    i = np.arange(-r, n + r)
    i = np.where(i < 0, -i - 1, i)
    return np.where(i > n - 1, 2 * n - 1 - i, i).astype(np.int64)


@njit(parallel=True, cache=True)
def _median_ranks(R, r, nb, ix, iy, iz):
    X, Y, Z = R.shape
    w = 2 * r + 1
    k = (w * w * w) // 2                                                 # 0-based rank of the median in the window
    out = np.empty_like(R)
    for yz in prange(Y * Z):
        y, z = yz // Z, yz % Z
        h = np.zeros(nb, np.int32)
        for a in range(w):
            for b in range(w):
                for c in range(w):
                    h[R[ix[a], iy[y + b], iz[z + c]]] += 1
        m = 0; lt = 0
        while lt + h[m] <= k:
            lt += h[m]; m += 1
        out[0, y, z] = m
        for x in range(1, X):
            xo, xn = ix[x - 1], ix[x - 1 + w]                            # the plane leaving, the plane entering
            for b in range(w):
                for c in range(w):
                    v = R[xo, iy[y + b], iz[z + c]]
                    h[v] -= 1
                    if v < m:
                        lt -= 1
                    v = R[xn, iy[y + b], iz[z + c]]
                    h[v] += 1
                    if v < m:
                        lt += 1
            while lt > k:
                m -= 1; lt -= h[m]
            while lt + h[m] <= k:
                lt += h[m]; m += 1
            out[x, y, z] = m
    return out


def multi_median(vol, radius=4, passes=4):
    """scipy.ndimage.median_filter(vol, size=2 radius + 1, mode="reflect"), `passes` times, exactly."""
    vals, R = np.unique(vol, return_inverse=True)
    R = R.reshape(vol.shape).astype(np.int32)
    ix, iy, iz = (_reflect(n, radius) for n in vol.shape)
    for _ in range(passes):
        R = _median_ranks(R, radius, len(vals), ix, iy, iz)
    return vals[R]


def median_otsu(b0, median_radius=4, numpass=4):
    """(masked b0, mask) as dipy.segment.mask.median_otsu(b0, median_radius=, numpass=) for a 3D volume."""
    m = multi_median(b0, median_radius, numpass)
    mask = m > otsu(m)
    return b0 * mask, mask
