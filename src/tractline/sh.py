"""Real spherical harmonics, even orders, in the descoteaux07 convention (Descoteaux et al. 2007) as DIPY defines it - the
basis DIPY's CSD writes and its PTT tracker reads, so FODs made here can be handed to the step 6 oracle as they are
(docs/sota-pipeline-design.md, section 7.1). numpy, float64: the matrices are small and made once per sphere or gradient
table.

    m, l = indices(8)                          # 45 coefficients: l = 0, 2, ..., 8; m = -l..l within each l
    B = basis(8, dirs)                         # (n, 45): FOD amplitude along each unit vector = B @ coeffs
    B = basis(8, dirs, legacy=False)           # the paper's convention (see below)

The complex harmonic is Y_l^m(theta, phi) = N_lm P_l^m(cos theta) e^{i m phi}, theta the polar angle, phi the azimuth,
N_lm = sqrt((2l + 1) / (4 pi) (l - m)! / (l + m)!), and P_l^m the associated Legendre function with the Condon-Shortley
phase (-1)^m (scipy.special's convention, which DIPY uses). The real basis, per coefficient (m, l):
  m > 0   sqrt(2) Im Y_l^|m|
  m = 0   Y_l^0
  m < 0   sqrt(2) Re Y_l^|m|  (legacy=True, DIPY's default and its CSD's: the |m| there is DIPY's "legacy" descoteaux07)
          sqrt(2) Re Y_l^m    (legacy=False, as the paper writes it: (-1)^m times the legacy value)
Written from those definitions; tests/test_sh.py checks every coefficient against dipy.reconst.shm."""
from __future__ import annotations

import math
import numpy as np


def n_coeffs(order: int) -> int:
    return (order + 1) * (order + 2) // 2


def indices(order: int):
    """(m, l) per coefficient, DIPY's order (sph_harm_ind_list): l = 0, 2, ..., order, and within each l, m = -l..l."""
    if order < 0 or order % 2:
        raise ValueError(f"sh order {order}: an even integer >= 0")
    ls = np.arange(0, order + 1, 2)
    l = np.repeat(ls, 2 * ls + 1)
    m = np.concatenate([np.arange(-k, k + 1) for k in ls])
    return m, l


def order_of(n: int) -> int:
    """The even order with n coefficients."""
    order = int(round((math.sqrt(8 * n + 1) - 3) / 2))
    if n_coeffs(order) != n or order % 2:
        raise ValueError(f"{n} coefficients is not an even order's count")
    return order


def angles(dirs):
    """Unit vectors (n, 3) -> (theta polar, phi azimuth), as dipy.core.geometry.cart2sphere gives them."""
    d = np.asarray(dirs, np.float64)
    r = np.linalg.norm(d, axis=-1)
    theta = np.arccos(np.clip(np.divide(d[..., 2], r, out=np.zeros_like(r), where=r > 0), -1.0, 1.0))
    return theta, np.arctan2(d[..., 1], d[..., 0])


def legendre(lmax: int, x):
    """P_l^m(x) for 0 <= m <= l <= lmax with the Condon-Shortley phase: (lmax + 1, lmax + 1, n), [l, m] (zero where m > l).
    By the standard recurrences: P_m^m = (-1)^m (2m - 1)!! (1 - x^2)^(m/2), P_{m+1}^m = x (2m + 1) P_m^m, then
    (l - m) P_l^m = x (2l - 1) P_{l-1}^m - (l + m - 1) P_{l-2}^m."""
    x = np.asarray(x, np.float64)
    s = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    P = np.zeros((lmax + 1, lmax + 1) + x.shape)
    pmm = np.ones_like(x)
    for m in range(lmax + 1):
        if m > 0:
            pmm = -pmm * (2 * m - 1) * s
        P[m, m] = pmm
        if m + 1 <= lmax:
            P[m + 1, m] = x * (2 * m + 1) * pmm
        for l in range(m + 2, lmax + 1):
            P[l, m] = (x * (2 * l - 1) * P[l - 1, m] - (l + m - 1) * P[l - 2, m]) / (l - m)
    return P


def basis(order: int, dirs, legacy=True) -> np.ndarray:
    """(n, n_coeffs(order)) float64: the real harmonics along each direction (rows), DIPY's coefficient order."""
    theta, phi = angles(dirs)
    m, l = indices(order)
    P = legendre(order, np.cos(theta))
    out = np.empty(theta.shape + (len(m),))
    for j, (mj, lj) in enumerate(zip(m, l)):
        a = abs(int(mj))
        norm = math.sqrt((2 * lj + 1) / (4 * math.pi) * math.factorial(lj - a) / math.factorial(lj + a))
        y = norm * P[lj, a]
        if mj == 0:
            out[..., j] = y
        elif mj > 0:
            out[..., j] = math.sqrt(2) * y * np.sin(a * phi)
        else:
            sign = 1.0 if legacy or a % 2 == 0 else -1.0                   # Re Y_l^{-a} = (-1)^a Re Y_l^a
            out[..., j] = sign * math.sqrt(2) * y * np.cos(a * phi)
    return out


def zonal(order: int, l_coeffs) -> np.ndarray:
    """Rotational harmonics (one value per even l: an axially symmetric function along z) spread to every (m, l):
    the per-coefficient diagonal of a convolution with that kernel, as a spherical deconvolution needs it."""
    _, l = indices(order)
    rh = np.asarray(l_coeffs, np.float64)
    if len(rh) != order // 2 + 1:
        raise ValueError(f"{len(rh)} rotational harmonics for order {order}: {order // 2 + 1} expected")
    return rh[l // 2]
