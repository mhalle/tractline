"""Rigid mutual-information registration in torch (any device), for the shell anchor: dipy's approach on the GPU.

Mattes-style MI: a joint histogram with cubic B-spline Parzen windows for both images, built as W_fixed' W_moving (dense
(N, bins) weights, so it is differentiable through the moving intensities); three levels (Gaussian smoothing 2, 1, 1.5
voxels, subsampling 4, 2, 1). The full-resolution level keeps sigma 1.5 voxels: unsmoothed, trilinear sampling's
shift-dependent blur biases MI's optimum (a known 0.5, -0.3, 0.4 mm shift came back as 0.40, -0.23, 0.26; at 1.5
voxels, to 0.008 mm); six rigid parameters in proto.py's frame (tx, ty, tz mm, rx, ry, rz in units of 0.01 rad,
R = Rz Ry Rx about the grid center c, world = (x - c) vox), mov(phi_d(x)) ~ fixed(x); maximized by BFGS with an Armijo
backtracking line search on autograd gradients.

    from mi_torch import register
    d = register(mov, fixed, mask, vox)      # (6,) tensor: mm, mm, mm, 0.01 rad x 3
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _rot(w):
    cx, cy, cz = torch.cos(w); sx, sy, sz = torch.sin(w)
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx]).view(3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy]).view(3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o]).view(3, 3)
    return Rz @ Ry @ Rx


def _gauss(v, sd):
    if sd <= 0:
        return v
    r = int(math.ceil(3 * sd))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=v.dtype, device=v.device) / sd) ** 2); k = k / k.sum()
    out = v[None, None]
    for ax in range(3):
        shape = [1, 1, 1, 1, 1]; shape[2 + ax] = len(k)
        pad = [0] * 6; pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = r
        out = F.conv3d(F.pad(out, pad, mode="replicate"), k.view(shape))
    return out[0, 0]


def _parzen(v, lo, hi, bins):
    """Cubic B-spline Parzen weights (N, bins) of intensities v scaled to [1, bins - 3] (2 bins of margin each side)."""
    u = (v - lo) / (hi - lo) * (bins - 5) + 2
    u = u.clamp(1.0, bins - 3.0)
    i0 = torch.floor(u); t = u - i0
    w = [(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6]
    W = torch.zeros(v.shape[0], bins, dtype=v.dtype, device=v.device)
    for k in range(4):
        W = W.scatter_add(1, (i0.long() + k - 1)[:, None], w[k][:, None])
    return W


def _mi(Wf, Wm):
    P = Wf.T @ Wm
    P = P / P.sum()
    pf, pm = P.sum(1, keepdim=True), P.sum(0, keepdim=True)
    nz = P > 1e-12
    return (P[nz] * torch.log(P[nz] / (pf @ pm)[nz])).sum()


def _mono(u, nmon):
    a, b, c = u.unbind(-1)
    cols = [torch.ones_like(a), a, b, c] + ([a * a, b * b, c * c, a * b, a * c, b * c] if nmon == 10 else [])
    return torch.stack(cols, -1)


def register(mov, fixed, mask, vox, bins=32, levels=((4, 2.0, 200), (2, 1.0, 100), (1, 1.5, 100)), d0=None, tol=1e-5,
             pe_axis=None, nmon=0):
    """pe_axis, nmon > 0: also an eddy-like displacement along pe_axis, a polynomial (nmon 4 or 10 terms) of the
    coordinates scaled to [-1, 1] on the full grid, in voxels of the full grid, as proto.py's eddy terms: d then has
    6 + nmon entries, mov(phi_d(x) + e(x) e_pe) ~ fixed(x)."""
    dev, dt = fixed.device, torch.float32
    mov, fixed = mov.to(dt), fixed.to(dt)
    vox = torch.as_tensor(vox, dtype=dt, device=dev)
    P = 6 + nmon
    d = torch.zeros(P, dtype=dt, device=dev) if d0 is None else d0.clone().to(dt)
    full_cen = (torch.tensor(mov.shape, dtype=dt, device=dev) - 1) / 2
    q = torch.tensor([0.02, 0.98], dtype=dt, device=dev)
    for f, sd, iters in levels:
        m_l = _gauss(mov, sd); f_l = _gauss(fixed, sd)
        if f > 1:
            m_l = F.avg_pool3d(m_l[None, None], f, f, ceil_mode=True)[0, 0]
            f_l = F.avg_pool3d(f_l[None, None], f, f, ceil_mode=True)[0, 0]
            mk = F.max_pool3d(mask.to(dt)[None, None], f, f, ceil_mode=True)[0, 0] > 0
        else:
            mk = mask.bool()
        X, Y, Z = f_l.shape
        vl = vox * f
        cen = (torch.tensor([X, Y, Z], dtype=dt, device=dev) - 1) / 2
        sizes = torch.tensor([X, Y, Z], dtype=dt, device=dev)
        pts = torch.nonzero(mk).to(dt)                                                    # (N, 3) the fixed image's mask
        fv = f_l[mk]
        flo, fhi = torch.quantile(fv, q); mv_all = m_l[m_l > 0] if (m_l > 0).any() else m_l.reshape(-1)
        mlo, mhi = torch.quantile(mv_all.reshape(-1)[:2 ** 24], q)
        Wf = _parzen(fv, flo, fhi, bins)
        xm = (pts - cen) * vl
        if nmon:
            xfull = pts * f + (f - 1) / 2                                                 # these points on the full grid
            mon = _mono((xfull - full_cen) / full_cen, nmon)                              # (N, nmon)

        def cost(d):
            R = _rot(d[3:6] * 0.01)
            p = (xm @ R.T + d[:3]) / vl + cen
            if nmon:
                e = (mon @ d[6:]) / f                                                     # full-grid voxels -> this level's
                p = p + e[:, None] * torch.eye(3, dtype=dt, device=dev)[pe_axis]
            s = F.grid_sample(m_l[None, None], (2 * p / (sizes - 1) - 1)[:, [2, 1, 0]][None, None, None], mode="bilinear",
                              padding_mode="border", align_corners=True).reshape(-1)
            return -_mi(Wf, _parzen(s, mlo, mhi, bins))

        def fg(d):
            d = d.detach().requires_grad_(True)
            with torch.enable_grad():
                c = cost(d)
                g, = torch.autograd.grad(c, d)
            return float(c), g.detach()

        c, g = fg(d)
        Hinv = torch.eye(P, dtype=dt, device=dev) * 0.1 / max(float(g.abs().max()), 1e-12)
        for _ in range(iters):
            p_ = -Hinv @ g
            if float(p_ @ g) >= 0:                                                        # not a descent direction: reset
                Hinv = torch.eye(P, dtype=dt, device=dev) * 0.1 / max(float(g.abs().max()), 1e-12); p_ = -Hinv @ g
            a = 1.0
            while True:
                c1, g1 = fg(d + a * p_)
                if c1 <= c + 1e-4 * a * float(p_ @ g) or a < 1e-6:
                    break
                a *= 0.5
            s_ = a * p_; y_ = g1 - g
            d, c_prev, c, g = d + s_, c, c1, g1
            sy = float(s_ @ y_)
            if sy > 1e-12:                                                                # BFGS inverse-Hessian update
                rho = 1.0 / sy
                I = torch.eye(P, dtype=dt, device=dev)
                Hinv = (I - rho * s_[:, None] @ y_[None]) @ Hinv @ (I - rho * y_[:, None] @ s_[None]) + rho * s_[:, None] @ s_[None]
            if float(s_.abs().max()) < tol:
                break
    return d
