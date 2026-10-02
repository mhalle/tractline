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
  - volumes scaled to a common mean intensity first (--scale=1).
Departures from topup, measured not hidden: trilinear interpolation (topup: cubic spline), L-BFGS
(topup: Levenberg-Marquardt, then scaled conjugate gradients), motion composed with the displacement
to first order (M_v(x) + d_v(x)).
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


def bending(c, Bs, dBs, d2Bs, vox):
    """Bending energy of h, mean over voxels, in (Hz / mm^2)^2."""
    hxx = sep(d2Bs[0], Bs[1], Bs[2], c) / vox[0] ** 2
    hyy = sep(Bs[0], d2Bs[1], Bs[2], c) / vox[1] ** 2
    hzz = sep(Bs[0], Bs[1], d2Bs[2], c) / vox[2] ** 2
    hxy = sep(dBs[0], dBs[1], Bs[2], c) / (vox[0] * vox[1])
    hxz = sep(dBs[0], Bs[1], dBs[2], c) / (vox[0] * vox[2])
    hyz = sep(Bs[0], dBs[1], dBs[2], c) / (vox[1] * vox[2])
    return (hxx ** 2 + hyy ** 2 + hzz ** 2 + 2 * (hxy ** 2 + hxz ** 2 + hyz ** 2)).mean()


# ------------------------------------------------------------------ the fit

def estimate(b0s: np.ndarray, vox, pe: np.ndarray, trt: np.ndarray, device="cpu", schedule=B02B0, iter_scale=3,
             lam_scale=1.0, fixed_motion=None, progress=None):
    """b0s (X, Y, Z, V), voxel sizes (mm), pe (V, 3) phase-encoding vectors, trt (V,) total readout
    times (s). Returns (field Hz on the full grid as numpy, motion (V, 6), per-level log)."""
    dev = torch.device(device)
    dt = torch.float32 if dev.type == "mps" else torch.float64
    img = torch.as_tensor(np.moveaxis(b0s, -1, 0), dtype=dt, device=dev)    # (V, X, Y, Z)
    img = img * (img.mean() / img.mean(dim=(1, 2, 3), keepdim=True))         # --scale=1
    V = img.shape[0]
    pe_ax = int(np.argmax(np.abs(pe).sum(0)))
    pe_scale_full = torch.as_tensor(pe[:, pe_ax] * trt, dtype=dt, device=dev)  # voxels per Hz (pe sign included)
    vox_t = torch.as_tensor(np.asarray(vox, float), dtype=dt, device=dev)
    CS = 10.0                                                                # Hz per optimizer unit
    MS = torch.tensor([1.0, 1.0, 1.0, 0.01, 0.01, 0.01], dtype=dt, device=dev)   # mm, mm, mm, rad, rad, rad per unit
    mov = torch.zeros(V - 1, 6, dtype=dt, device=dev)
    if fixed_motion is not None:                                             # (V, 6), volume 0's row ignored: not estimated
        mov = torch.as_tensor(np.asarray(fixed_motion)[1:], dtype=dt, device=dev)
    h_prev, log = None, []
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

        def cost():
            c = cs * CS
            m = ms * MS
            h = field(c, Bs)
            dh = sep(*[(dBs[a] if a == pe_ax else Bs[a]) for a in range(3)], c)
            Rs, ts = [torch.eye(3, dtype=dt, device=dev)], [torch.zeros(3, dtype=dt, device=dev)]
            for v in range(V - 1):
                R, t = rigid(m[v], center); Rs.append(R); ts.append(t)
            u = unwarp(im, h, dh, pe_ax, pe_scale, Rs, ts, vl)
            ssd = ((u - u.mean(0, keepdim=True)) ** 2).mean()
            return ssd + lam * ssd.detach() * bending(c, Bs, dBs, d2Bs, vl), ssd

        params = [cs] + ([ms] if est else [])
        opt = torch.optim.LBFGS(params, lr=1, max_iter=schedule["miter"][lev] * iter_scale, history_size=20,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            total, _ = cost()
            total.backward()
            return total
        with torch.no_grad():
            before = float(cost()[1])
        opt.step(closure)
        with torch.no_grad():
            after = float(cost()[1])
            h_prev = field(cs * CS, Bs).detach()
            mov = (ms * MS).detach()
        log.append({"level": lev + 1, "grid": [X, Y, Z], "knots": [B.shape[1] for B in Bs], "ssd_before": before, "ssd_after": after,
                    "motion": est})
        if progress:
            progress(log[-1])
    full = Fnn.interpolate(h_prev[None, None], size=tuple(b0s.shape[:3]), mode="trilinear", align_corners=True)[0, 0]
    motion = torch.cat([torch.zeros(1, 6, dtype=dt, device=dev), mov]).cpu().numpy()
    return full.cpu().double().numpy(), motion, log
