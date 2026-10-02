"""Susceptibility-distortion correction from reversed phase-encoding images: FSL topup's model in torch
(any device; float32 on MPS), to be judged against topup itself (topup_ref.py, susc_check.py).

The model (Andersson, Skare & Ashburner 2003; topup's b02b0.cnf schedule):
  - an off-resonance field h (Hz), a tensor-product cubic B-spline on the image grid;
  - volume v, acquired with phase-encoding vector pe_v and total readout time T_v, is displaced along
    its phase-encoding axis by d_v = (pe_v . axis) T_v h voxels, its intensity scaled by the Jacobian
    1 + dd_v/dy, and moved by a rigid transform M_v (volume 0 fixed);
  - unwarping volume v at reference point x samples it at M_v(x) + d_v(x) e_pe and multiplies by the
    Jacobian; the cost is the sum of squared differences between the unwarped volumes and their mean,
    plus lambda * ssq * bending energy of h (ssq the current mean squared difference: ssqlambda = 1);
  - levels as b02b0.cnf: knot spacing (mm), subsampling, Gaussian smoothing (FWHM mm), iterations,
    lambda, motion estimated or not; the field carried from level to level by least-squares
    projection onto the next level's basis (exact for tensor-product bases);
  - volumes scaled to a common mean intensity first (--scale=1);
  - no translation along the phase-encoding axis for the first volume of each acquisition (topup's
    convention, which fixes the trade between a field offset and such translations).
The fit: Levenberg-Marquardt Gauss-Newton (gauss_newton(); topup's --minmet 0), matrix-free, at
topup's iteration counts, stopping earlier on the cost's relative decrease.
Departures from topup, measured not hidden: trilinear interpolation throughout the estimate by default
(interp="cubic_pe": cubic B-splines along the phase-encoding axis where motion is held, as topup's
splines; measured no better, slower); HySCO's anti-folding penalty (fold_phi(), Ruthotto et al. 2012)
added to the cost - without it Gauss-Newton folds the field outside the brain; motion composed with the
displacement to first order (M_v(x) + d_v(x)). Measured: NOTES 2026-10-02 "by Gauss-Newton" and
"Repeatability from independent b0s" (the held-out prediction on 12 patients picks topup's lambda).
"""
from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as Fnn

B02B0 = dict(warpres=(20, 16, 14, 12, 10, 6, 4, 4, 4), subsamp=(2, 2, 2, 2, 2, 1, 1, 1, 1), fwhm=(8, 6, 4, 3, 3, 2, 1, 0, 0),
             miter=(5, 5, 5, 5, 5, 10, 10, 20, 20),
             lam=(0.005, 0.001, 0.0001, 0.000015, 0.000005, 0.0000005, 0.00000005, 0.0000000005, 0.00000000001),
             estmov=(1, 1, 1, 1, 1, 0, 0, 0, 0))


# ------------------------------------------------------------------ cubic B-spline bases

def _b3(t):
    a = t.abs()
    return torch.where(a < 1, 2 / 3 - a ** 2 + a ** 3 / 2, torch.where(a < 2, (2 - a) ** 3 / 6, torch.zeros_like(a)))


def _db3(t):
    a, s = t.abs(), torch.sign(t)
    return s * torch.where(a < 1, -2 * a + 1.5 * a ** 2, torch.where(a < 2, -0.5 * (2 - a) ** 2, torch.zeros_like(a)))


def _d2b3(t):
    a = t.abs()
    return torch.where(a < 1, -2 + 3 * a, torch.where(a < 2, 2 - a, torch.zeros_like(a)))


def basis(n: int, spacing: float, device, dtype):
    """(B, dB, d2B): (n, k) matrices of a cubic B-spline basis with knots every `spacing` voxels,
    covering voxels 0..n-1 (one knot beyond each end); derivatives per voxel."""
    k = int(math.ceil((n - 1) / spacing)) + 3
    centers = (torch.arange(k, dtype=dtype, device=device) - 1) * spacing
    t = (torch.arange(n, dtype=dtype, device=device)[:, None] - centers[None, :]) / spacing
    return _b3(t), _db3(t) / spacing, _d2b3(t) / spacing ** 2


def sep(Bx, By, Bz, c):
    """Bx (x) By (x) Bz . c, contracted one axis at a time (never an (i, j, k, a, b, c) intermediate)."""
    t = torch.tensordot(c, Bz, dims=([2], [1]))                    # (a, b, k)
    t = torch.tensordot(t, By, dims=([1], [1]))                    # (a, k, j)
    t = torch.tensordot(t, Bx, dims=([0], [1]))                    # (k, j, i)
    return t.permute(2, 1, 0)


def field(c, Bs):
    """h = Bx (x) By (x) Bz . c for coefficients c (kx, ky, kz) and per-axis matrices Bs = (Bx, By, Bz)."""
    return sep(Bs[0], Bs[1], Bs[2], c)


def project(h, Bs):
    """The least-squares coefficients of h on the tensor-product basis (separable pseudo-inverses)."""
    P = [torch.linalg.pinv(B.cpu()).to(B.device) for B in Bs]            # (k, n) each
    return sep(P[0], P[1], P[2], h)


# ------------------------------------------------------------------ images

def smooth(img, fwhm_mm, vox):
    """Separable Gaussian smoothing of (V, X, Y, Z) images."""
    if fwhm_mm <= 0:
        return img
    out = img
    for ax in range(3):
        sig = fwhm_mm / 2.3548 / vox[ax]
        r = int(math.ceil(3 * sig))
        x = torch.arange(-r, r + 1, dtype=img.dtype, device=img.device)
        w = torch.exp(-0.5 * (x / sig) ** 2); w = w / w.sum()
        shape = [1, 1, 1, 1, 1]; shape[2 + ax] = len(w)
        pad = [0, 0, 0, 0, 0, 0]; pad[2 * (2 - ax)] = r; pad[2 * (2 - ax) + 1] = r
        out = Fnn.conv3d(Fnn.pad(out[:, None], pad, mode="replicate"), w.view(shape))[:, 0]
    return out


def subsample(img, f):
    return img if f == 1 else Fnn.avg_pool3d(img[:, None], f, f, ceil_mode=True)[:, 0]


def rigid(p, center_mm):
    """4x4 (mm) of 6 parameters: translations (mm) and rotations (rad) about the image center."""
    tx, ty, tz, rx, ry, rz = p
    one, zero = torch.ones_like(rx), torch.zeros_like(rx)
    Rx = torch.stack([torch.stack([one, zero, zero]), torch.stack([zero, torch.cos(rx), -torch.sin(rx)]), torch.stack([zero, torch.sin(rx), torch.cos(rx)])])
    Ry = torch.stack([torch.stack([torch.cos(ry), zero, torch.sin(ry)]), torch.stack([zero, one, zero]), torch.stack([-torch.sin(ry), zero, torch.cos(ry)])])
    Rz = torch.stack([torch.stack([torch.cos(rz), -torch.sin(rz), zero]), torch.stack([torch.sin(rz), torch.cos(rz), zero]), torch.stack([zero, zero, one])])
    R = Rz @ Ry @ Rx
    t = torch.stack([tx, ty, tz]) + center_mm - R @ center_mm
    return R, t


def unwarp(img, h, dh, pe_ax, pe_scale, R, t, vox, jac=True):
    """Volumes (V, X, Y, Z) unwarped by the field h (Hz) and its derivative along the phase-encoding
    axis dh (Hz/voxel): sample volume v at R_v x + t_v (mm) plus pe_scale_v h voxels along pe_ax, times
    the Jacobian 1 + pe_scale_v dh. Trilinear."""
    V, X, Y, Z = img.shape
    dev, dt = img.device, img.dtype
    g = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=dt, device=dev) for n in (X, Y, Z)], indexing="ij"), -1)  # (X, Y, Z, 3) voxels
    mm = g * vox
    out = []
    for v in range(V):
        p = (mm @ R[v].T + t[v]) / vox                                      # moved, in voxels
        p = p.clone(); p[..., pe_ax] = p[..., pe_ax] + pe_scale[v] * h
        # grid_sample wants (x, y, z) -> (W, H, D) order in [-1, 1]: input (1, 1, X, Y, Z) is (D, H, W) = (X, Y, Z)
        sizes = torch.tensor([X, Y, Z], dtype=dt, device=dev)
        norm = 2 * p / (sizes - 1) - 1
        grid = norm[..., [2, 1, 0]][None]
        s = Fnn.grid_sample(img[v][None, None], grid, mode="bilinear", padding_mode="zeros", align_corners=True)[0, 0]
        out.append(s * (1 + pe_scale[v] * dh) if jac else s)
    return torch.stack(out)


def prefilter(img, ax):
    """Cubic B-spline coefficients of images (V, X, Y, Z) along axis ax (0-2 of the image), mirrored
    edges: solves c[i-1]/6 + 4 c[i]/6 + c[i+1]/6 = f[i] with a dense (n, n) inverse (n <= a few hundred)."""
    n = img.shape[1 + ax]
    M = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        M[i, i] = 4 / 6
        for j in (i - 1, i + 1):
            jj = -j if j < 0 else (2 * (n - 1) - j if j > n - 1 else j)        # mirror (whole-sample)
            M[i, jj] += 1 / 6
    Minv = torch.linalg.inv(M).to(img.device, img.dtype)
    return torch.movedim(torch.tensordot(torch.movedim(img, 1 + ax, -1), Minv, dims=([3], [1])), -1, 1 + ax)


def sample_pe(coef, pos, ax):
    """Cubic B-spline samples of coef (V, X, Y, Z) at positions pos (V, X, Y, Z) along axis ax (voxels),
    the other coordinates being the voxel's own; mirrored edges. pos (1, X, Y, Z): the same positions
    for every volume, the weights and indices computed once."""
    n = coef.shape[1 + ax]
    i0 = torch.floor(pos)
    t = pos - i0
    w = [(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6]
    out = None
    for k, wk in zip(range(-1, 3), w):
        idx = (i0 + k).long()
        idx = torch.where(idx < 0, -idx, idx)
        idx = torch.where(idx > n - 1, 2 * (n - 1) - idx, idx).clamp(0, n - 1)
        s = wk * torch.gather(coef, 1 + ax, idx.expand(coef.shape[0], *idx.shape[1:]))
        out = s if out is None else out + s
    return out


def unwarp_pe_cubic(coef, h, dh, pe_ax, pe_scale, jac=True):
    """unwarp() without motion, cubic B-spline along the phase-encoding axis (coef: prefilter() of the
    already moved images): volume v sampled at x + pe_scale_v h along pe_ax, times 1 + pe_scale_v dh."""
    V = coef.shape[0]
    n = coef.shape[1 + pe_ax]
    shape = [1, 1, 1, 1]; shape[1 + pe_ax] = n
    base = torch.arange(n, dtype=coef.dtype, device=coef.device).view(shape)
    if V > 1 and bool((pe_scale == pe_scale[0]).all()):                      # one acquisition: one set of positions
        pe_scale = pe_scale[:1]; V = 1
    pos = base + pe_scale.view(V, 1, 1, 1) * h[None]
    s = sample_pe(coef, pos, pe_ax)
    return s * (1 + pe_scale.view(V, 1, 1, 1) * dh[None]) if jac else s


def unwarp_pe_linear(moved, h, dh, pe_ax, pe_scale, jac=True):
    """unwarp() without motion, linear along the phase-encoding axis (moved: the already moved images,
    (V, X, Y, Z)): volume v sampled at x + pe_scale_v h along pe_ax, zero outside (grid_sample's
    padding), times 1 + pe_scale_v dh. Two gathers a voxel instead of a 3D trilinear sample."""
    V, n = moved.shape[0], moved.shape[1 + pe_ax]
    shape = [1, 1, 1, 1]; shape[1 + pe_ax] = n
    pos = torch.arange(n, dtype=moved.dtype, device=moved.device).view(shape) + pe_scale.view(V, 1, 1, 1) * h[None]
    out = sample_pe_linear(moved, pos, pe_ax)
    return out * (1 + pe_scale.view(V, 1, 1, 1) * dh[None]) if jac else out


def sample_pe_linear(moved, pos, ax):
    """Linear samples of moved (V, X, Y, Z) at positions pos (V, X, Y, Z) along axis ax (voxels), zero
    outside."""
    n = moved.shape[1 + ax]
    i0 = torch.floor(pos)
    f = pos - i0
    out = 0
    for k, w in ((0, 1 - f), (1, f)):
        idx = (i0 + k).long()
        ok = (idx >= 0) & (idx <= n - 1)
        v = torch.gather(moved, 1 + ax, idx.clamp(0, n - 1))
        out = out + torch.where(ok, w * v, torch.zeros_like(v))
    return out


def bending(c, Bs, dBs, d2Bs, vox):
    """Bending energy of h, mean over voxels, in (Hz / mm^2)^2."""
    hxx = sep(d2Bs[0], Bs[1], Bs[2], c) / vox[0] ** 2
    hyy = sep(Bs[0], d2Bs[1], Bs[2], c) / vox[1] ** 2
    hzz = sep(Bs[0], Bs[1], d2Bs[2], c) / vox[2] ** 2
    hxy = sep(dBs[0], dBs[1], Bs[2], c) / (vox[0] * vox[1])
    hxz = sep(dBs[0], Bs[1], dBs[2], c) / (vox[0] * vox[2])
    hyz = sep(Bs[0], dBs[1], dBs[2], c) / (vox[1] * vox[2])
    return (hxx ** 2 + hyy ** 2 + hzz ** 2 + 2 * (hxy ** 2 + hxz ** 2 + hyz ** 2)).mean()


def sep_t(mats, y):
    """The transpose of sep(*mats, .): y (X, Y, Z) back onto the coefficients."""
    return sep(mats[0].T, mats[1].T, mats[2].T, y)


def bending_terms(Bs, dBs, d2Bs, vox):
    """bending() as sum_t w_t * mean(sep(*mats_t, c) ** 2): [(mats_t, w_t)], the cross terms twice."""
    v = [float(x) for x in vox]
    T = [([d2Bs[i] if i == a else Bs[i] for i in range(3)], 1 / v[a] ** 4) for a in range(3)]
    T += [([dBs[i] if i in (a, b) else Bs[i] for i in range(3)], 2 / (v[a] * v[b]) ** 2) for a, b in ((0, 1), (0, 2), (1, 2))]
    return T


FOLD_XC = 0.95


def fold_phi(x, order=0):
    """HySCO's intensity-modulation penalty phi(x) = x^4 / (1 - x^2) (Ruthotto et al. 2012), x the
    displacement's derivative along the phase-encoding axis (voxels per voxel): convex, ~x^4 near 0, and
    unbounded as either polarity's Jacobian 1 +/- x approaches 0 - folding. Beyond |x| = FOLD_XC its
    quadratic Taylor extension, so a trial step that folds costs much but finitely. order 1, 2: the
    derivatives."""
    a = x.abs().clamp_max(FOLD_XC)
    a2 = a * a; q = 1 - a2
    p0 = a2 * a2 / q
    p1 = (4 * a2 * a - 2 * a2 * a2 * a) / q ** 2
    p2 = (12 * a2 - 6 * a2 * a2 + 2 * a2 * a2 * a2) / q ** 3
    e = (x.abs() - FOLD_XC).clamp_min(0)
    if order == 0:
        return p0 + p1 * e + 0.5 * p2 * e * e
    if order == 1:
        return torch.sign(x) * (p1 + p2 * e)
    return p2


def _pcg(A, b, Minv, iters, rtol, check=1):
    """Preconditioned conjugate gradients for A x = b (A symmetric positive definite, as a function);
    returns (x, iterations). The residual is tested every `check` iterations (each test a device sync)."""
    x = torch.zeros_like(b); r = b.clone(); z = Minv * r; p = z.clone(); rz = (r * z).sum()
    stop = rtol * float(b.norm())
    for k in range(1, iters + 1):
        Ap = A(p)
        alpha = rz / (p * Ap).sum()
        x = x + alpha * p; r = r - alpha * Ap
        if k % check == 0 and float(r.norm()) <= stop:
            break
        z = Minv * r; rz_new = (r * z).sum(); p = z + (rz_new / rz) * p; rz = rz_new
    return x, k


def gauss_newton(c, m, *, sample, Bs, dBs, d2Bs, vox, pe_ax, pe_scale, lam, mmask, est, center, max_iter, fold=0.0,
                 xtol=0.005, ftol=1e-4, cg_iters=30, cg_rtol=0.1, cg_check=5):
    """Levenberg-Marquardt Gauss-Newton on one level's cost (estimate()'s: the unwarped volumes' mean
    squared difference from their mean, plus lam * ssd * bending energy, ssd held within an iteration as
    topup's ssqlambda; with fold > 0, plus fold * ssd * the mean of fold_phi() over the voxels, x the
    largest-readout volume's displacement derivative along pe - HySCO's anti-folding term).
    Matrix-free: the residual's Jacobian is the sampled images' derivative along the sampling position
    (autograd, pointwise) times the field's B-spline basis, the Jacobian factor's
    derivative, and - when motion is estimated - the position's derivative by the six rigid parameters;
    each step solves (J'J + lam s L + mu D) d = -g by conjugate gradients, D = diag of the Gauss-Newton
    matrix (computed separably), mu adapted as Marquardt's. Stops when the step moves no voxel by more
    than xtol voxels (at this level) or the cost falls by less than ftol relatively, or at max_iter.

    c (kx, ky, kz) Hz, m (V-1, 6) mm / rad (masked by mmask) on entry; sample(h, m, grad) returns the
    volumes sampled at the reference grid moved by m and shifted pe_scale_v h along pe_ax (V, X, Y, Z),
    and with grad, their derivative by the sampling position (V, X, Y, Z, 3) and the position's by the
    motion parameters (V-1, X, Y, Z, 3, 6), or None. Returns (c, m, info)."""
    V = pe_scale.shape[0]
    sv = pe_scale.view(V, 1, 1, 1)
    dmats = [dBs[a] if a == pe_ax else Bs[a] for a in range(3)]
    terms = bending_terms(Bs, dBs, d2Bs, vox)
    nvox = Bs[0].shape[0] * Bs[1].shape[0] * Bs[2].shape[0]
    kshape = c.shape; nc = c.numel()
    mm_ = mmask.reshape(-1)
    kx = float(pe_scale.abs().max())                                         # x = kx * dh: displacement voxels per voxel

    # each term's sep_t(M, sep(M, v)) is one contraction by its per-axis Gram matrices M_a' M_a (k x k)
    grams = [([Mi.T @ Mi for Mi in M], 2 * w / nvox) for M, w in terms]

    def grad_b(v):                                                           # gradient of bending(v): linear in v
        return sum(w2 * sep(*G, v) for G, w2 in grams)

    def bend(c):                                                             # bending(c) = c . grad_b(c) / 2
        return 0.5 * (c * grad_b(c)).sum()

    def cost(c, m, s):
        h = field(c, Bs); dh = sep(*dmats, c)
        u = sample(h, m, False)[0] * (1 + sv * dh[None])
        ssd = ((u - u.mean(0, keepdim=True)) ** 2).mean()
        b = bend(c)
        pf = fold_phi(kx * dh).mean() if fold else 0.0
        return float(ssd + lam * s * b + fold * s * pf), float(ssd)

    reg_diag = sum(2 * w / nvox * torch.einsum("i,j,k->ijk", *[(Mi ** 2).sum(0) for Mi in M]) for M, w in terms)
    mu, it, evals, cg_total, converged, trace = 1e-3, 0, 0, 0, False, []
    while it < max_iter:
        h = field(c, Bs); dh = sep(*dmats, c)
        S, G, dpdm = sample(h, m, True)
        jf = 1 + sv * dh[None]
        u = S * jf
        r = u - u.mean(0, keepdim=True)
        Mtot = u.numel()
        s = float((r ** 2).mean())
        f0 = s + lam * s * float(bend(c))
        xf = kx * dh
        if fold:
            f0 += fold * s * float(fold_phi(xf).mean())
            ff1 = fold * s * kx / nvox * fold_phi(xf, 1)                     # d/d(dh) of the penalty, per voxel
            ff2 = fold * s * kx * kx / nvox * fold_phi(xf, 2)                # its second derivative
        A = sv * G[..., pe_ax] * jf                                          # d u / d h
        Bj = sv * S                                                          # d u / d (dh/dpe)
        Mv = (jf[1:, ..., None] * torch.einsum("vxyzi,vxyzik->vxyzk", G[1:], dpdm)) * mmask[:, None, None, None, :] if est else None

        def JtP(w):                                                          # J' w for w already projected (mean over volumes removed)
            gc = sep_t(Bs, (A * w).sum(0)) + sep_t(dmats, (Bj * w).sum(0))
            gm = (Mv * w[1:, ..., None]).sum((1, 2, 3)) if est else None
            return gc, gm

        gc, gm = JtP(r)
        gcc = 2 / Mtot * gc + lam * s * grad_b(c) + (sep_t(dmats, ff1) if fold else 0)
        g = torch.cat([gcc.reshape(-1)] + ([(2 / Mtot * gm).reshape(-1)] if est else []))
        # the Gauss-Newton matrix's diagonal, separably: sum over volumes of (A B_i + Bj dB_i)^2 minus V times its mean squared
        W1 = (A ** 2).sum(0) - A.sum(0) ** 2 / V
        W2 = 2 * ((A * Bj).sum(0) - A.sum(0) * Bj.sum(0) / V)
        W3 = (Bj ** 2).sum(0) - Bj.sum(0) ** 2 / V
        sq = lambda mats: [Mi ** 2 for Mi in mats]
        cross = [Bs[a] * dBs[a] if a == pe_ax else Bs[a] ** 2 for a in range(3)]
        dc_diag = 2 / Mtot * (sep_t(sq(Bs), W1) + sep_t(cross, W2) + sep_t(sq(dmats), W3)) + lam * s * reg_diag
        if fold:
            dc_diag = dc_diag + sep_t(sq(dmats), ff2)
        D = torch.cat([dc_diag.reshape(-1)] + ([(2 / Mtot * (1 - 1 / V) * (Mv ** 2).sum((1, 2, 3))).reshape(-1)] if est else []))
        free = torch.cat([torch.ones(nc, dtype=c.dtype, device=c.device)] + ([mm_] if est else []))
        D = torch.where(free > 0, D.clamp_min(float(D[:nc].max()) * 1e-8), torch.ones_like(D))

        def H(v, mu):
            dc = v[:nc].reshape(kshape)
            du = A * field(dc, Bs)[None] + Bj * sep(*dmats, dc)[None]
            if est:
                dm = v[nc:].reshape(-1, 6) * mmask
                du = torch.cat([du[:1], du[1:] + (Mv * dm[:, None, None, None, :]).sum(-1)])
            hc, hm = JtP(du - du.mean(0, keepdim=True))
            hcc = 2 / Mtot * hc + lam * s * grad_b(dc) + (sep_t(dmats, ff2 * sep(*dmats, dc)) if fold else 0)
            out = [hcc.reshape(-1)] + ([(2 / Mtot * hm).reshape(-1)] if est else [])
            return torch.cat(out) * free + mu * D * v + (1 - free) * v

        ks = []
        while True:                                                          # Marquardt: shrink the step until the cost falls
            d, k = _pcg(lambda v: H(v, mu), -g, 1 / ((1 + mu) * D), cg_iters, cg_rtol, cg_check)
            cg_total += k; ks.append(k)
            c_new = c + d[:nc].reshape(kshape)
            m_new = m + d[nc:].reshape(-1, 6) * mmask if est else m
            f1, _ = cost(c_new, m_new, s); evals += 1
            if f1 < f0:
                mu = max(mu / 3, 1e-7)
                break
            mu *= 8
            if mu > 1e6:
                break
        it += 1
        if f1 >= f0:                                                         # no descent left at any damping
            converged = True
            break
        step = (pe_scale.abs().max() * field(c_new - c, Bs)).abs()
        step_vox = float(step.max())
        trace.append([round(f0, 4), round(f1, 4), round(step_vox, 4), round(float(step.quantile(0.99) if step.numel() < 2 ** 24 else 0), 4), ks])
        c, m = c_new, m_new
        if step_vox < xtol or (f0 - f1) < ftol * f0:
            converged = True
            break
    return c, m, {"n_iter": it, "func_evals": evals, "cg_iters": cg_total, "converged": converged, "mu": mu, "trace": trace}


# ------------------------------------------------------------------ the fit

def estimate(b0s: np.ndarray, vox, pe_vectors: np.ndarray, readout_s, *, device="cpu", dtype=None, progress=None,
             schedule=B02B0, iter_scale=None, lam_scale=1.0, fixed_motion=None, interp="trilinear", optimizer="gn",
             fold=10.0, gn=None, coarse_device="auto", init=None, diagnostics=False):
    """The susceptibility field from b0s with at least two phase-encoding directions.

    b0s (X, Y, Z, V); vox (3,) mm; pe_vectors (V, 3), each b0's phase-encoding vector; readout_s the
    total readout time (s), one for all or (V,). device "mps" runs in float32, "cpu" in float64 unless
    dtype says otherwise;
    progress(level_log) is called after each level. Returns (field Hz (X, Y, Z) numpy float64,
    motion (V, 6): translations mm, rotations rad, volume 0 fixed; the per-level log).

    The rest are experiment options, their defaults the pipeline's: schedule (topup's b02b0.cnf),
    iter_scale (the schedule's iterations per level times this; default 1, 3 for "lbfgs"), lam_scale (topup's lambda, rescaled: its units are not
    ours), fixed_motion ((V, 6) held instead of estimated), interp ("cubic_pe": at levels whose motion
    is held, cubic B-splines along the phase-encoding axis, topup's --interp=spline; measured no
    better, 70 % slower; "linear_pe": the same, linear along the axis - the images moved once per level,
    then two gathers a voxel instead of a 3D trilinear sample and its gradient), optimizer ("gn":
    gauss_newton(); "lbfgs": torch's L-BFGS, the earlier default - never converges, its result where
    the cap falls), fold (with "gn": the weight of HySCO's anti-folding penalty, fold_phi()), gn
    (gauss_newton()'s tolerances), coarse_device (the subsampled levels there, in this dtype, the rest
    on device: small grids are launch-bound on a GPU, 3x slower than the CPU on the M2; "auto": the CPU
    under "mps", else none), init ((field Hz at full resolution, motion (V, 6)) to start from
    instead of zero). diagnostics: each level's log also holds the optimizer's iterations and evaluations
    and the field at full resolution."""
    if iter_scale is None:
        iter_scale = 3 if optimizer == "lbfgs" else 1
    if coarse_device == "auto":
        coarse_device = "cpu" if torch.device(device).type == "mps" else None
    if coarse_device is not None and torch.device(coarse_device) != torch.device(device):
        k = next(i for i, f in enumerate(schedule["subsamp"]) if f == 1)
        dt = dtype or (torch.float32 if torch.device(device).type == "mps" else torch.float64)
        opts = dict(dtype=dt, progress=progress, iter_scale=iter_scale, lam_scale=lam_scale, fixed_motion=fixed_motion,
                    interp=interp, optimizer=optimizer, fold=fold, gn=gn, coarse_device=None, diagnostics=diagnostics)
        h0, m0, log0 = estimate(b0s, vox, pe_vectors, readout_s, device=coarse_device, init=init,
                                schedule={key: v[:k] for key, v in schedule.items()}, **opts)
        renumber = lambda L: L.update(level=L["level"] + k) or (progress(L) if progress else None)
        h, m, log1 = estimate(b0s, vox, pe_vectors, readout_s, device=device, init=(h0, m0),
                              schedule={key: v[k:] for key, v in schedule.items()}, **{**opts, "progress": renumber})
        return h, m, log0 + log1
    pe = np.asarray(pe_vectors, float)
    trt = np.broadcast_to(np.asarray(readout_s, float), (len(pe),))
    dev = torch.device(device)
    dt = dtype or (torch.float32 if dev.type == "mps" else torch.float64)
    img = torch.as_tensor(np.moveaxis(b0s, -1, 0), dtype=dt, device=dev)    # (V, X, Y, Z)
    img = img * (img.mean() / img.mean(dim=(1, 2, 3), keepdim=True))         # --scale=1
    V = img.shape[0]
    pe_ax = int(np.argmax(np.abs(pe).sum(0)))
    pe_scale_full = torch.as_tensor(pe[:, pe_ax] * trt, dtype=dt, device=dev)  # voxels per Hz (pe sign included)
    vox_t = torch.as_tensor(np.asarray(vox, float), dtype=dt, device=dev)
    CS = 10.0                                                                # Hz per optimizer unit
    MS = torch.tensor([1.0, 1.0, 1.0, 0.01, 0.01, 0.01], dtype=dt, device=dev)   # mm, mm, mm, rad, rad, rad per unit
    # topup's convention: the first volume of each acquisition (distinct phase encoding and readout)
    # has no translation along the phase-encoding axis - a field offset and such translations
    # are otherwise interchangeable (topup_movpar.txt shows it: the first PA volume's is exactly 0)
    first_of = {}
    for v in range(V):
        first_of.setdefault((tuple(np.round(pe[v], 6)), round(float(trt[v]), 9)), v)
    mmask = torch.ones(V - 1, 6, dtype=dt, device=dev)
    for v in first_of.values():
        if v > 0:
            mmask[v - 1, pe_ax] = 0.0
    mov = torch.zeros(V - 1, 6, dtype=dt, device=dev)
    if fixed_motion is not None:                                             # (V, 6), volume 0's row ignored: not estimated
        mov = torch.as_tensor(np.asarray(fixed_motion)[1:], dtype=dt, device=dev)
    h_prev, log = None, []
    if init is not None:
        h_prev = torch.as_tensor(np.asarray(init[0]), dtype=dt, device=dev)
        if fixed_motion is None:
            mov = torch.as_tensor(np.asarray(init[1])[1:], dtype=dt, device=dev)
    for lev in range(len(schedule["warpres"])):
        f = schedule["subsamp"][lev]
        im = subsample(smooth(img, schedule["fwhm"][lev], vox_t), f)
        vl = vox_t * f
        X, Y, Z = im.shape[1:]
        Bs, dBs, d2Bs = zip(*[basis(n, max(1.0, round(schedule["warpres"][lev] / float(vl[a]))), dev, dt) for a, n in enumerate((X, Y, Z))])
        if h_prev is None:
            c = torch.zeros(*(B.shape[1] for B in Bs), dtype=dt, device=dev)
        else:
            hp = Fnn.interpolate(h_prev[None, None], size=(X, Y, Z), mode="trilinear", align_corners=True)[0, 0]
            c = project(hp, Bs)
        # the optimizer works on scaled parameters, so field coefficients (tens of Hz) and rotations
        # (hundredths of a radian) take steps of similar size: c = cs * CS, motion = ms * MS
        cs = (c / CS).contiguous().clone().requires_grad_(True)
        est = bool(schedule["estmov"][lev]) and fixed_motion is None
        ms = (mov / MS).clone().requires_grad_(est)
        center = (torch.tensor([X, Y, Z], dtype=dt, device=dev) - 1) / 2 * vl
        pe_scale = pe_scale_full / f                                           # displacement in this level's voxels
        lam = schedule["lam"][lev] * lam_scale                                   # topup's lambda units are not ours: lam_scale calibrates

        cubic = interp == "cubic_pe" and not est
        linear = interp == "linear_pe" and not est
        if cubic or linear:                                                      # motion held: move once, then sample along pe
            with torch.no_grad():
                Rs0, ts0 = [torch.eye(3, dtype=dt, device=dev)], [torch.zeros(3, dtype=dt, device=dev)]
                for v in range(V - 1):
                    R, t = rigid(mov[v], center); Rs0.append(R); ts0.append(t)
                moved = unwarp(im, torch.zeros_like(im[0]), torch.zeros_like(im[0]), pe_ax, pe_scale, Rs0, ts0, vl, jac=False)
                coef = prefilter(moved, pe_ax) if cubic else moved

        def cost():
            c = cs * CS
            m = ms * MS * mmask
            h = field(c, Bs)
            dh = sep(*[(dBs[a] if a == pe_ax else Bs[a]) for a in range(3)], c)
            if cubic or linear:
                u = (unwarp_pe_cubic if cubic else unwarp_pe_linear)(coef, h, dh, pe_ax, pe_scale)
                ssd = ((u - u.mean(0, keepdim=True)) ** 2).mean()
                return ssd + lam * ssd.detach() * bending(c, Bs, dBs, d2Bs, vl), ssd
            Rs, ts = [torch.eye(3, dtype=dt, device=dev)], [torch.zeros(3, dtype=dt, device=dev)]
            for v in range(V - 1):
                R, t = rigid(m[v], center); Rs.append(R); ts.append(t)
            u = unwarp(im, h, dh, pe_ax, pe_scale, Rs, ts, vl)
            ssd = ((u - u.mean(0, keepdim=True)) ** 2).mean()
            return ssd + lam * ssd.detach() * bending(c, Bs, dBs, d2Bs, vl), ssd

        with torch.no_grad():
            before = float(cost()[1])
        if optimizer == "gn":
            gmm = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=dt, device=dev) for n in (X, Y, Z)], indexing="ij"), -1) * vl
            sizes = torch.tensor([X, Y, Z], dtype=dt, device=dev)
            e_pe = torch.zeros(3, dtype=dt, device=dev); e_pe[pe_ax] = 1
            pshape = [1, 1, 1, 1]; pshape[1 + pe_ax] = (X, Y, Z)[pe_ax]
            base = torch.arange((X, Y, Z)[pe_ax], dtype=dt, device=dev).view(pshape)

            def sample(h, m, grad):
                """The volumes at the moved grid plus pe_scale h along pe (V, X, Y, Z); with grad, their
                derivative by the position (V, X, Y, Z, 3) and the position's by the motion (V-1, X, Y, Z, 3, 6)."""
                if cubic or linear:
                    pos = (base + pe_scale.view(V, 1, 1, 1) * h[None]).detach().requires_grad_(grad)
                    with torch.set_grad_enabled(grad):
                        Sv = (sample_pe if cubic else sample_pe_linear)(coef, pos, pe_ax)
                    if not grad:
                        return Sv, None, None
                    G = torch.autograd.grad(Sv.sum(), pos)[0][..., None] * e_pe
                    return Sv.detach(), G, None
                with torch.no_grad():
                    p = [gmm / vl] + [(gmm @ R.T + t) / vl for R, t in (rigid(m[v], center) for v in range(V - 1))]
                    p = torch.stack(p) + (pe_scale.view(V, 1, 1, 1) * h[None])[..., None] * e_pe
                p = p.requires_grad_(grad)
                with torch.set_grad_enabled(grad):
                    Sv = Fnn.grid_sample(im[:, None], (2 * p / (sizes - 1) - 1)[..., [2, 1, 0]], mode="bilinear",
                                         padding_mode="zeros", align_corners=True)[:, 0]
                if not grad:
                    return Sv, None, None
                G = torch.autograd.grad(Sv.sum(), p)[0]
                dpdm = None
                if est:
                    dpdm = []
                    for v in range(V - 1):
                        dR, dtt = torch.autograd.functional.jacobian(lambda q: rigid(q, center), m[v])
                        dpdm.append((torch.einsum("ijk,xyzj->xyzik", dR, gmm) + dtt) / vl[:, None])
                    dpdm = torch.stack(dpdm)
                return Sv.detach(), G, dpdm

            c_gn, m_gn, info = gauss_newton((cs * CS).detach(), (ms * MS * mmask).detach(), sample=sample, Bs=Bs, dBs=dBs,
                                            d2Bs=d2Bs, vox=vl, pe_ax=pe_ax, pe_scale=pe_scale, lam=lam, mmask=mmask, est=est,
                                            center=center, max_iter=schedule["miter"][lev] * iter_scale, fold=fold, **(gn or {}))
            with torch.no_grad():
                cs.copy_(c_gn / CS); ms.copy_(m_gn / MS)
        else:
            params = [cs] + ([ms] if est else [])
            opt = torch.optim.LBFGS(params, lr=1, max_iter=schedule["miter"][lev] * iter_scale, history_size=20,
                                    line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                total, _ = cost()
                total.backward()
                return total
            opt.step(closure)
            st = opt.state[opt._params[0]]
            info = {"n_iter": int(st.get("n_iter", 0)), "func_evals": int(st.get("func_evals", 0))}
        with torch.no_grad():
            after = float(cost()[1])
            h_prev = field(cs * CS, Bs).detach()
            mov = (ms * MS * mmask).detach()
        log.append({"level": lev + 1, "grid": [X, Y, Z], "knots": [B.shape[1] for B in Bs], "ssd_before": before, "ssd_after": after,
                    "motion": est, "interp": "cubic along pe" if cubic else "linear along pe" if linear else "trilinear"})
        if diagnostics:
            log[-1].update(**info, max_iter=schedule["miter"][lev] * iter_scale,
                           field=Fnn.interpolate(h_prev[None, None], size=tuple(b0s.shape[:3]), mode="trilinear", align_corners=True)[0, 0].cpu().double().numpy(),
                           motion_now=torch.cat([torch.zeros(1, 6, dtype=dt, device=dev), mov]).cpu().numpy())
        if progress:
            progress(log[-1])
    full = Fnn.interpolate(h_prev[None, None], size=tuple(b0s.shape[:3]), mode="trilinear", align_corners=True)[0, 0]
    motion = torch.cat([torch.zeros(1, 6, dtype=dt, device=dev), mov]).cpu().numpy()
    return full.cpu().double().numpy(), motion, log


# ------------------------------------------------------------------ applying a field

def displacement_mm(field_hz, readout_s, pe_sign, vox_pe):
    """The displacement (mm, along the phase-encoding axis, signed) that a field (Hz) causes in a
    scan with that readout time (s), phase-encoding sign (+1 / -1) and voxel size along the axis."""
    return pe_sign * readout_s * np.asarray(field_hz, dtype=np.float64) * vox_pe


def apply(dwi, field_hz, pe_axis, pe_sign, readout_s):
    """A DWI (X, Y, Z, G) corrected by a field (Hz, on its grid): each volume sampled at x + the
    displacement along the phase-encoding axis, cubic B-splines along that axis (applytopup's
    splines), times the Jacobian (applytopup --method=jac), clipped at 0. CPU, float64; returns float32
    (X, Y, Z, G). One field and one readout for every volume: the sampling positions are shared."""
    h = torch.as_tensor(np.asarray(field_hz, dtype=np.float64))
    vols = torch.as_tensor(np.moveaxis(dwi, -1, 0).astype(np.float64))
    out = unwarp_pe_cubic(prefilter(vols, pe_axis), h, torch.gradient(h, dim=pe_axis)[0], pe_axis,
                          torch.full((vols.shape[0],), pe_sign * readout_s, dtype=torch.float64)).numpy()
    return np.moveaxis(np.clip(out, 0, None), 0, -1).astype(np.float32)
