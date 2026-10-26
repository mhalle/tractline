"""Head motion and eddy-current correction of a diffusion series, with the susceptibility field composed into the same
single resample (docs/gpu-motion-eddy-design.md). The method follows FSL eddy's (Andersson & Sotiropoulos 2016):
each volume is registered to a leave-one-out prediction from the others, with a rigid transform and an eddy-current
displacement along the phase-encoding axis that is a polynomial of position, and the b-vectors are rotated with the
head. The implementation is ours: one batched Levenberg-Marquardt Gauss-Newton fit over all volumes at once, in
PyTorch, with fused Metal kernels on Apple GPUs (tractline.motion_eddy_metal) and plain torch on the CPU or CUDA.

    r = estimate(dwi, bval, bvec, affine, pe_axis, field_hz=h, pe_sign=+1, readout_s=0.069, device="mps")
    corrected, valid = apply_volumes(dwi, r, field_hz=h)        # one cubic resample per volume: motion, eddy and field
    r.bvec                                                      # (3, G) FSL b-vectors, rotated with the head
    r.qc                                                        # displacements, framewise displacement, residuals

The model (design doc, section 1). For a reference point x (voxels) and volume v, the acquired volume is sampled at

    phi_v(x) = c + (R_v (x - c) vox + t_v) / vox + [d_h(x) + e_v(x)] e_pe

R_v, t_v the volume's rigid motion about the grid center c (mm, rad); e_v the eddy-current displacement along the
phase-encoding axis e_pe (voxels), a polynomial of the coordinates scaled to [-1, 1] (4 terms, linear, or 10,
quadratic: Haselgrove & Moore 1996; eddy's --flm); d_h = pe_sign readout_s h the static field's displacement (voxels,
as susceptibility.apply). Both displacements are evaluated at the reference point (first order, as eddy without its
susceptibility-by-movement option). The corrected volume is the acquired one sampled at phi_v(x) times the Jacobian
1 + d/dx_pe (d_h + e_v), clipped at 0.

The fit (design doc, section 2):
  - Each volume's target is a leave-one-out prediction from the other volumes of its shell: Laplace-Beltrami regularized
    spherical harmonics per DW shell (order 6, lambda 0.06), the mean of the other b0s for a b0. The predictions are
    built in the reference frame from the current estimate and rebuilt before every step.
  - The residual is taken in acquisition space: the prediction (a cubic B-spline, so its gradient is continuous) moved
    into each volume's acquired grid by phi_v^-1 (Newton steps along PE: a field's displacement can be steep) and
    compared with the data as acquired, so noisy data are never interpolated. Its Jacobian is the chain rule through
    phi_v^-1 at the reference point, the map's full derivative included (the prototype's columns were taken at the
    acquired point without it: 3-15 % off without a field, up to 39 % with a steep one).
  - Gauges, each because a leave-one-out predictor cannot see the quantity: the eddy constant is held out of the solve
    (it is the translation along PE by another name) and afterwards the PE offset is split as eddy splits it; each DW
    shell's eddy field common to all its directions is removed after each step (eddy's second-level model, linear in
    the gradient with no intercept, makes the same assumption); each group's mean pose is held at 0, the b0s' being
    the location reference. With a field, the reference is the field's own frame instead: the first b0 is held fixed
    (the other b0s free). Either way each DW shell's mean pose is held at the b0s' mean pose.
  - Optionally, each DW shell's corrected mean is registered to the corrected mean b0 by mutual information afterwards
    (eddy's post-eddy alignment of shells; tractline.mi), off by default (it moved one scan's b2000 shell 1.27 mm).

Choices that follow eddy's implementation, not only its paper (FSL's source was read for method; no code is taken from
it): the PE-offset split by a demeaned linear model in (b / 1000) g; the output masked to the voxels every volume
covers (eddy's default output mask); 256 histogram bins for the shell alignment's mutual information.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import torch
import torch.nn.functional as F

ORDERS = {"rigid": 0, "linear": 4, "quadratic": 10}


@dataclass(frozen=True)
class Options:
    """The step's settings; every default is the validated configuration (design doc, sections 2.10-2.12)."""
    model: str = "quadratic"         # "rigid", "linear" or "quadratic": the eddy polynomial's terms (0, 4, 10)
    steps: int = 40                  # Gauss-Newton steps (the cap when tol > 0)
    tol: float = 0.0                 # stop when no volume's step moves an edge point more than this (voxels), twice
    smooth_steps: int = 2            # the first steps on smoothed volumes (FWHM 2, then 1 voxel)
    coarse: int = 0                  # the first steps on every other voxel per axis
    trial_vox: float = 0.3           # a Levenberg-Marquardt trial only for steps that move an edge point more
    sh_order: int = 6                # the per-shell predictor's SH order (lowered for shells with few directions)
    sh_lambda: float = 0.06          # its Laplace-Beltrami weight (section 2.11)
    b0_ec: bool = False              # the b0s fit the PE polynomial too (motion-by-susceptibility; section 2.10)
    anchor: str = "none"             # "none" or "mi": each DW shell's corrected mean aligned to the mean b0 at the end
    anchor_eddy: bool = False        # the anchor also fits the shell's common eddy polynomial
    anchor_bins: int = 256           # the anchor's histogram bins (eddy's PEAS: 256)
    mask_output: bool = True         # apply_volumes zeroes voxels not inside every volume's field of view (eddy's)
    reference: str = "auto"          # "b0-mean", "first-b0", or "auto": first-b0 with a field, else b0-mean
    npass: int = 0                   # Newton steps of the inverse map (0: 2 without a field, 3 with one)
    backend: str = "auto"            # "metal" (MPS), "torch", or "auto": metal on MPS, else torch

    def __post_init__(self):
        if self.model not in ORDERS:
            raise ValueError(f"motion_eddy: model {self.model!r} is not one of {sorted(ORDERS)}")
        if self.anchor not in ("none", "mi"):
            raise ValueError(f"motion_eddy: anchor {self.anchor!r} is not 'none' or 'mi'")
        if self.reference not in ("auto", "b0-mean", "first-b0"):
            raise ValueError(f"motion_eddy: reference {self.reference!r}")
        if self.backend not in ("auto", "metal", "torch"):
            raise ValueError(f"motion_eddy: backend {self.backend!r}")
        if self.steps < 1 or self.sh_order < 0 or self.sh_order % 2 or self.sh_lambda < 0:
            raise ValueError("motion_eddy: steps >= 1, sh_order an even integer >= 0, sh_lambda >= 0")

    @classmethod
    def make(cls, options=None):
        if options is None:
            return cls()
        if isinstance(options, cls):
            return options
        unknown = set(options) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"motion_eddy: unknown options {sorted(unknown)}")
        return cls(**options)


@dataclass
class MotionEddy:
    """estimate()'s result. theta: (G, 6 + k) per volume - translations (mm), rotations (rad, R = Rz Ry Rx about the
    grid center), eddy coefficients (voxels along PE) - as the sampling map uses them, the whole PE offset in the
    translation (theta_fit, as fitted) or split as eddy splits it (theta: the offset a demeaned linear model in
    (b / 1000) g explains is the eddy constant, the rest movement; the same map). bvec: (3, G) rotated FSL b-vectors."""
    theta: np.ndarray
    theta_fit: np.ndarray
    bvec: np.ndarray
    model: str
    pe_axis: int
    shape: tuple
    vox: np.ndarray
    groups: list                     # volume indices per group: the b0s first (if any), then each shell by b
    group_b: list                    # each group's nominal b-value (0 for the b0s)
    reference: str
    field_used: bool
    pe_scale: float | None           # pe_sign * readout_s (s), when a field was used
    anchors: dict
    qc: dict
    options: dict
    seconds: dict = field(default_factory=dict)
    backend: str = ""                # "metal" or "torch": what ran


# ------------------------------------------------------------------------------------------------- geometry

class Geometry:
    """The grid and the parameterization: center c, the polynomial's scaling (half = c), voxel sizes, PE axis, terms."""

    def __init__(self, shape, vox, pe_axis, nmon, device, dtype):
        self.shape = tuple(int(n) for n in shape)
        self.device, self.dtype = torch.device(device), dtype
        self.cen = (torch.tensor(self.shape, dtype=dtype, device=device) - 1) / 2
        self.half = self.cen.clone()
        self.vox = torch.as_tensor(np.asarray(vox, np.float64), dtype=dtype, device=device)
        self.pe_axis, self.nmon = int(pe_axis), int(nmon)
        self.P = 6 + self.nmon
        self.scale = torch.cat([torch.ones(3), torch.full((3,), 0.01), torch.ones(self.nmon)]).to(device, dtype)
        self.e_pe = torch.eye(3, dtype=dtype, device=device)[self.pe_axis]


def rot(w):
    """w (G, 3) rad -> (G, 3, 3), R = Rz Ry Rx (susceptibility.rigid's)."""
    cx, cy, cz = torch.cos(w).unbind(-1)
    sx, sy, sz = torch.sin(w).unbind(-1)
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], -1).view(-1, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).view(-1, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], -1).view(-1, 3, 3)
    return Rz @ Ry @ Rx


def pack(theta, eps=1e-4):
    """Per-volume parameters (G, 49) as the kernels read them: R (9, row-major), t (3, mm), eddy (10, voxels; zero-padded),
    dR/dw_j (3 x 9, central differences)."""
    G = theta.shape[0]
    w = theta[:, 3:6]
    R = rot(w)
    dR = []
    for j in range(3):
        d = torch.zeros_like(w)
        d[:, j] = eps
        dR.append(((rot(w + d) - rot(w - d)) / (2 * eps)).reshape(G, 9))
    te = torch.zeros(G, 10, dtype=theta.dtype, device=theta.device)
    te[:, :theta.shape[1] - 6] = theta[:, 6:]
    return torch.cat([R.reshape(G, 9), theta[:, :3], te] + dR, 1).contiguous()


def eddy_poly(geo, te, x):
    """The eddy displacement (voxels along PE) of coefficients te (G, k) at points x ((N, 3) shared or (G, N, 3))."""
    if geo.nmon == 0:
        return torch.zeros((te.shape[0],) + x.shape[-2:-1], dtype=x.dtype, device=x.device)
    a, b, c = ((x - geo.cen) / geo.half).unbind(-1)
    k = lambda i: te[:, i, None]                                                          # noqa: E731
    e = k(0) + k(1) * a + k(2) * b + k(3) * c
    if geo.nmon == 10:
        e = e + k(4) * a * a + k(5) * b * b + k(6) * c * c + k(7) * a * b + k(8) * a * c + k(9) * b * c
    return e


def eddy_dpe(geo, te, x):
    """d e / d x_pe (voxels per voxel) at points x."""
    if geo.nmon == 0:
        return torch.zeros((te.shape[0],) + x.shape[-2:-1], dtype=x.dtype, device=x.device)
    u = (x - geo.cen) / geo.half
    a, b, c = u.unbind(-1)
    k = lambda i: te[:, i, None]                                                          # noqa: E731
    pe = geo.pe_axis
    d = k(1 + pe) + torch.zeros_like(a)
    if geo.nmon == 10:
        d = d + (2 * k(4) * a + k(7) * b + k(8) * c if pe == 0 else
                 2 * k(5) * b + k(7) * a + k(9) * c if pe == 1 else 2 * k(6) * c + k(8) * a + k(9) * b)
    return d / geo.half[pe]


def field_at(dfield, x):
    """dfield (X, Y, Z) trilinear at x ((N, 3) or (G, N, 3) voxels), the position clamped to the grid -> (G?, N)."""
    sizes = torch.tensor(dfield.shape, dtype=x.dtype, device=x.device)
    q = (2 * x / (sizes - 1) - 1)[..., [2, 1, 0]].reshape(1, 1, 1, -1, 3)
    s = F.grid_sample(dfield[None, None].to(x.dtype), q, mode="bilinear", padding_mode="border", align_corners=True)
    return s.reshape(x.shape[:-1])


def phi(geo, theta, x, dfield=None):
    """The sampling map: reference points x ((N, 3) or (G, N, 3), voxels) -> acquired positions (G, N, 3), voxels."""
    R = rot(theta[:, 3:6])
    xm = (x - geo.cen) * geo.vox
    eq = "gij,nj->gni" if x.dim() == 2 else "gij,gnj->gni"
    p = (torch.einsum(eq, R, xm) + theta[:, None, :3]) / geo.vox + geo.cen
    d = eddy_poly(geo, theta[:, 6:], x)
    if dfield is not None:
        d = d + field_at(dfield, x)
    return p + d[..., None] * geo.e_pe


def phi_inv(geo, theta, y, dfield=None, npass=2, ddfield=None):
    """x with phi(x) = y (y (N, 3) voxels -> (G, N, 3)). A first estimate inverts the rigid part in closed form with the
    displacement taken at y: x0 = c + R^T((y - c) vox - d(y) e_pe vox - t) / vox. Then npass Newton steps, the map's
    derivative being A + e_pe s e_pe' (A the rotation in voxel units, s = d d / d x_pe, the displacement's slope along
    PE; its other slopes left out), inverted by Sherman-Morrison: x <- x - (u - a s u_pe / (1 + s a_pe)), u = A^-1 r,
    a = A^-1 e_pe, r = phi(x) - y. A field's displacement can be steep along PE (several voxels over a few), where
    fixed-point passes converge slowly. ddfield: the field displacement's derivative along PE (computed if None)."""
    R = rot(theta[:, 3:6])
    te, t = theta[:, 6:], theta[:, None, :3]
    pe = geo.pe_axis
    if dfield is not None and ddfield is None:
        ddfield = torch.gradient(dfield, dim=pe)[0]

    def disp(x):
        d = eddy_poly(geo, te, x)
        return d + field_at(dfield, x) if dfield is not None else d
    q = (y - geo.cen) * geo.vox - disp(y)[..., None] * (geo.e_pe * geo.vox) - t
    x = torch.einsum("gji,gnj->gni", R, q) / geo.vox + geo.cen
    a = torch.einsum("gji,j->gi", R, geo.e_pe * geo.vox) / geo.vox                      # A^-1 e_pe (G, 3)
    for _ in range(npass):
        r = phi(geo, theta, x, dfield) - y
        u = torch.einsum("gji,gnj->gni", R, r * geo.vox) / geo.vox                       # A^-1 r
        s = eddy_dpe(geo, te, x)
        if dfield is not None:
            s = s + field_at(ddfield, x)
        den = (1 + s * a[:, None, pe]).clamp_min(0.1)                                     # a fold (s <= -1): damped
        x = x - (u - a[:, None, :] * (s * u[..., pe] / den)[..., None])
    return x


# ------------------------------------------------------------------------------------------------- plain torch backend
# The same pieces as tractline.motion_eddy_metal's kernels, in torch on any device and dtype (the CPU and CUDA path, and
# in float64 the reference the kernels are held to).

def _bspline(t):
    w = [(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6]
    dw = [-(1 - t) ** 2 / 2, (3 * t ** 2 - 4 * t) / 2, (-3 * t ** 2 + 2 * t + 1) / 2, t ** 2 / 2]
    return w, dw


def cubic_value_grad(coef, p, want_grad=True):
    """The cubic B-spline of coef (G, X, Y, Z) at p (G, N, 3) voxels and its exact gradient: the 64 coefficients
    gathered once each (indices clamped), accumulated into the value and the three derivatives."""
    G, X, Y, Z = coef.shape
    i0 = torch.floor(p)
    w, dw = _bspline(p - i0)
    i0 = i0.long()
    flat = coef.reshape(G, -1)
    val = torch.zeros(p.shape[:2], dtype=coef.dtype, device=coef.device)
    gx, gy, gz = (torch.zeros_like(val) for _ in range(3))
    for a in range(4):
        ia = (i0[..., 0] + a - 1).clamp(0, X - 1)
        for b in range(4):
            ib = (i0[..., 1] + b - 1).clamp(0, Y - 1)
            wab = w[a][..., 0] * w[b][..., 1]
            base = (ia * Y + ib) * Z
            for c in range(4):
                v = torch.gather(flat, 1, base + (i0[..., 2] + c - 1).clamp(0, Z - 1))
                val = val + wab * w[c][..., 2] * v
                if want_grad:
                    gx = gx + dw[a][..., 0] * w[b][..., 1] * w[c][..., 2] * v
                    gy = gy + w[a][..., 0] * dw[b][..., 1] * w[c][..., 2] * v
                    gz = gz + wab * dw[c][..., 2] * v
    return val, (torch.stack([gx, gy, gz], -1) if want_grad else None)


def _prefilter_matrix(n, dtype, device):
    M = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        M[i, i] = 4 / 6
        for j in (i - 1, i + 1):
            jj = -j if j < 0 else (2 * (n - 1) - j if j > n - 1 else j)            # whole-sample mirror
            M[i, jj] += 1 / 6
    return torch.linalg.inv(M).to(device, dtype)


def t_prefilter(vols):
    """Cubic B-spline coefficients of vols (G, X, Y, Z), whole-sample mirrored edges (a dense solve per axis)."""
    for ax in range(3):
        Minv = _prefilter_matrix(vols.shape[1 + ax], vols.dtype, vols.device)
        vols = torch.movedim(torch.movedim(vols, 1 + ax, -1) @ Minv.T, -1, 1 + ax)
    return vols.contiguous()


def t_smooth(vols, fwhm_vox):
    if fwhm_vox <= 0:
        return vols
    sd = fwhm_vox / 2.3548
    r = int(math.ceil(3 * sd))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=vols.dtype, device=vols.device) / sd) ** 2)
    k = k / k.sum()
    out = vols[:, None]
    for ax in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + ax] = len(k)
        pad = [0] * 6
        pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = r
        out = F.conv3d(F.pad(out, pad, mode="replicate"), k.view(shape))
    return out[:, 0]


def t_resample(geo, vols, pts, theta, dfield=None, chunk=16):
    """vols (G, X, Y, Z) sampled trilinearly at phi_g(pts), zero outside -> (G, N)."""
    G, X, Y, Z = vols.shape
    sizes = torch.tensor([X, Y, Z], dtype=vols.dtype, device=vols.device)
    out = []
    for a in range(0, G, chunk):
        p = phi(geo, theta[a:a + chunk], pts, dfield)
        q = (2 * p / (sizes - 1) - 1)[..., [2, 1, 0]][:, None, None]
        out.append(F.grid_sample(vols[a:a + chunk, None], q, mode="bilinear", padding_mode="zeros",
                                 align_corners=True)[:, 0, 0, 0])
    return torch.cat(out)


def eddy_grad(geo, te, x):
    """d e / d x (G, N, 3), voxels per voxel, at points x (G, N, 3)."""
    if geo.nmon == 0:
        return torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    a, b, c = ((x - geo.cen) / geo.half).unbind(-1)
    k = lambda i: te[:, i, None]                                                          # noqa: E731
    g = [k(1) + 0 * a, k(2) + 0 * a, k(3) + 0 * a]
    if geo.nmon == 10:
        g = [g[0] + 2 * k(4) * a + k(7) * b + k(8) * c, g[1] + 2 * k(5) * b + k(7) * a + k(9) * c,
             g[2] + 2 * k(6) * c + k(8) * a + k(9) * b]
    return torch.stack(g, -1) / geo.half


def field_grad(dfield, x):
    """The gradient (G, N, 3) of field_at's trilinear interpolant at x: the derivative of the map as sampled."""
    with torch.enable_grad():
        xq = x.detach().requires_grad_(True)
        g, = torch.autograd.grad(field_at(dfield, xq).sum(), xq)
    return g


def _jacobian(geo, theta, x, gp, wgrad):
    """J (G, N, P) of the prediction sampled at x = phi^-1(y) by the scaled parameters: d/dtheta P(phi^-1(y)) =
    -grad P' (dphi/dx)^-1 dphi/dtheta, all at x. dphi/dx = A + e_pe w' (A the rotation in voxel units, w the gradient of
    the displacement along PE: the eddy polynomial's and the field's), its inverse transpose by Sherman-Morrison:
    g~ = A^-T gp - A^-T w (A^-T gp)_pe / (1 + (A^-T w)_pe). gp: the cubic's gradient at x (voxels); wgrad: w at x."""
    R = rot(theta[:, 3:6])
    vox, pe = geo.vox, geo.pe_axis
    g0 = torch.einsum("gij,gnj->gni", R, gp / vox) * vox                                 # A^-T gp
    aw = torch.einsum("gij,gnj->gni", R, wgrad / vox) * vox                              # A^-T w
    den = (1 + aw[..., pe]).clamp_min(0.1)
    gt = g0 - aw * (g0[..., pe] / den)[..., None]
    xm = (x - geo.cen) * vox
    w = theta[:, 3:6]
    cols = [gt / vox]
    rc = []
    for j in range(3):
        d = torch.zeros_like(w)
        d[:, j] = 1e-4
        dR = (rot(w + d) - rot(w - d)) / 2e-4
        rc.append((torch.einsum("gij,gnj->gni", dR, xm) / vox * gt).sum(-1))
    cols.append(torch.stack(rc, -1))
    if geo.nmon:
        u = (x - geo.cen) / geo.half
        a_, b_, c_ = u.unbind(-1)
        mon = [torch.ones_like(a_), a_, b_, c_] + ([a_ * a_, b_ * b_, c_ * c_, a_ * b_, a_ * c_, b_ * c_]
                                                    if geo.nmon == 10 else [])
        cols.append(gt[..., pe:pe + 1] * torch.stack(mon, -1))
    return -torch.cat(cols, -1) * geo.scale


def _wgrad(geo, theta, x, dfield):
    w = eddy_grad(geo, theta[:, 6:], x)
    return w + field_grad(dfield, x) if dfield is not None else w


def t_step_moments(geo, coef, obs, ys, theta, offset, dfield=None, npass=2, chunk=8, ddfield=None):
    """H (G, P, P), b (G, P), f (G,) of the residual prediction(phi^-1(y)) - obs over the points ys (N, 3)."""
    off = torch.as_tensor(offset, dtype=coef.dtype, device=coef.device)
    Hs, bs, fs = [], [], []
    for a in range(0, coef.shape[0], chunk):
        th = theta[a:a + chunk]
        if dfield is not None and ddfield is None:
            ddfield = torch.gradient(dfield, dim=geo.pe_axis)[0]
        x = phi_inv(geo, th, ys, dfield, npass, ddfield)
        v, gp = cubic_value_grad(coef[a:a + chunk], x - off)
        r = v - obs[a:a + chunk]
        J = _jacobian(geo, th, x, gp, _wgrad(geo, th, x, dfield))
        Hs.append(torch.bmm(J.transpose(1, 2), J))
        bs.append(torch.bmm(J.transpose(1, 2), r[..., None])[..., 0])
        fs.append((r * r).sum(1))
    return torch.cat(Hs), torch.cat(bs), torch.cat(fs)


def t_step_cost(geo, coef, obs, ys, theta_sub, vols, offset, dfield=None, npass=2, ddfield=None):
    off = torch.as_tensor(offset, dtype=coef.dtype, device=coef.device)
    x = phi_inv(geo, theta_sub, ys, dfield, npass, ddfield)
    v, _ = cubic_value_grad(coef[vols], x - off, want_grad=False)
    return ((v - obs[vols]) ** 2).sum(1)


def t_apply_volumes(geo, vols, theta, dfield=None, ddfield=None, chunk=4):
    """The corrected volumes and their validity (G, X, Y, Z): sampled once at phi(x), times the Jacobian along PE."""
    G, X, Y, Z = vols.shape
    full = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=vols.dtype, device=vols.device) for n in (X, Y, Z)],
                                      indexing="ij"), -1).reshape(-1, 3)
    hi = torch.tensor([X - 1, Y - 1, Z - 1], dtype=vols.dtype, device=vols.device)
    jf = field_at(ddfield, full) if ddfield is not None else 0.0
    outs, valids = [], []
    for a in range(0, G, chunk):
        th = theta[a:a + chunk]
        coef = t_prefilter(vols[a:a + chunk])
        outs_c, valid_c = [], []
        for s in range(0, full.shape[0], 1 << 20):
            xs = full[s:s + (1 << 20)]
            p = phi(geo, th, xs, dfield)
            val, _ = cubic_value_grad(coef, p, want_grad=False)
            jac = 1 + eddy_dpe(geo, th[:, 6:], xs) + (jf[s:s + (1 << 20)] if ddfield is not None else 0.0)
            inside = ((p >= 0) & (p <= hi)).all(-1)
            outs_c.append(torch.where(inside, (val * jac).clamp_min(0), 0.0))
            valid_c.append(inside.to(vols.dtype))
        outs.append(torch.cat(outs_c, 1))
        valids.append(torch.cat(valid_c, 1))
    return torch.cat(outs).reshape(G, X, Y, Z), torch.cat(valids).reshape(G, X, Y, Z)


# ------------------------------------------------------------------------------------------------- the predictor

def sh_basis(order, dirs):
    """A real, orthonormal even-order SH basis (n, (order + 1)(order + 2) / 2) and each column's degree l. The
    leave-one-out smoother below is the same for any orthonormal basis of each degree, so no convention is needed."""
    from scipy.special import sph_harm_y
    d = np.asarray(dirs, np.float64)
    theta = np.arccos(np.clip(d[:, 2], -1, 1))
    ph = np.arctan2(d[:, 1], d[:, 0])
    cols, ls = [], []
    for l in range(0, order + 1, 2):
        for m in range(-l, l + 1):
            y = sph_harm_y(l, abs(m), theta, ph)
            cols.append(np.sqrt(2) * y.imag if m < 0 else (np.sqrt(2) * y.real if m > 0 else y.real))
            ls.append(l)
    return np.stack(cols, 1), np.array(ls)


def loo_smoother(groups, isb0, g, sh_order, sh_lambda):
    """W (G, G): row v the leave-one-out prediction of volume v from the others of its group. A DW shell: the regularized
    SH smoother H = B (B'B + lambda L)^-1 B' (L = diag(l(l+1))^2), W = (H - diag H) / (1 - diag H). The b0s: the mean of
    the others. A group of one: a zero row (nothing to predict it from)."""
    G = len(isb0)
    W = np.zeros((G, G))
    orders = {}
    for idx in groups:
        n = len(idx)
        if n < 2:
            continue
        if isb0[idx[0]]:
            H = np.full((n, n), 1.0 / n)
        else:
            order = sh_order
            while order > 0 and (order + 1) * (order + 2) // 2 > n:
                order -= 2
            orders[int(idx[0])] = order
            B, ls = sh_basis(order, g[idx])
            L = np.diag((ls * (ls + 1.0)) ** 2)
            H = B @ np.linalg.solve(B.T @ B + sh_lambda * L, B.T)
        dh = np.diag(H)
        W[np.ix_(idx, idx)] = (H - np.diag(dh)) / (1 - dh)[:, None]
    return W, orders


def shell_groups(bval):
    """The b0s (b < 50) first, then each DW shell, volumes grouped by b rounded to 100: (groups, nominal b-values)."""
    bval = np.asarray(bval, np.float64)
    isb0 = bval < 50
    groups, bs = [], []
    if isb0.any():
        groups.append(np.flatnonzero(isb0))
        bs.append(0)
    key = np.round(bval / 100) * 100
    for b in sorted(set(key[~isb0])):
        groups.append(np.flatnonzero(~isb0 & (key == b)))
        bs.append(int(b))
    return groups, bs


# ------------------------------------------------------------------------------------------------- masks

def _masks(dwi, bval, dfield_np, pe_axis, mask):
    """(mask in the reference frame, points mask for the acquired data, wider mask for the predictions). Without a field
    the reference and acquired frames coincide to motion; with one, the brain's acquired position is shifted by the
    field, so the acquired-data points come from the acquired mean b0's mask and the predictions cover both."""
    from scipy.ndimage import binary_dilation
    from .mask import median_otsu
    b0 = np.asarray(dwi, np.float32)[..., np.asarray(bval) < 50]
    mean_b0 = b0.mean(-1) if b0.shape[-1] else np.asarray(dwi, np.float32).mean(-1)
    if mask is None:
        _, m_acq = median_otsu(mean_b0, median_radius=4, numpass=4)
        m_acq = np.asarray(m_acq, bool)
    else:
        m_acq = np.asarray(mask, bool)
    m_ref = m_acq
    if dfield_np is not None:                                             # the field's unwarp (no motion), cubic along PE
        from . import susceptibility as S
        h = torch.as_tensor(dfield_np)

        def unwarp(img):
            return S.unwarp_pe_cubic(S.prefilter(torch.as_tensor(img[None], dtype=torch.float64), pe_axis), h,
                                     torch.zeros_like(h), pe_axis, torch.ones(1, dtype=torch.float64), jac=False)[0].numpy()
        if mask is None:
            _, m_ref = median_otsu(unwarp(mean_b0).astype(np.float32), median_radius=4, numpass=4)
            m_ref = np.asarray(m_ref, bool)
        else:
            m_ref = unwarp(m_acq.astype(np.float64)) > 0.5
    pts = binary_dilation(m_acq, iterations=2)
    pred = binary_dilation(m_acq | m_ref, iterations=4)
    return m_ref, pts, pred


# ------------------------------------------------------------------------------------------------- the estimate

def estimate(dwi, bval, bvec, affine, pe_axis, field_hz=None, pe_sign=None, readout_s=None, mask=None, options=None,
             device="mps", dtype=torch.float32, progress=None) -> MotionEddy:
    """Motion and eddy-current parameters for every volume of dwi (X, Y, Z, G), as acquired (module docstring).
    bval (G,), bvec (3, G) FSL; affine the voxel -> RAS mm map (its determinant sets FSL's x flip for the b-vectors);
    pe_axis the phase-encoding axis (0, 1, 2). field_hz (X, Y, Z): the susceptibility field (Hz, on the DWI's grid, in
    the frame of the DWI's first b0, as susceptibility.estimate returns it), with pe_sign (+1 / -1) and readout_s; None
    fits without one (as eddy without topup). mask: a brain mask of the data as acquired (with a field, its unwarped
    copy is the reference frame's), else median_otsu of the mean b0 (and of its unwarped copy).
    options: an Options or a dict of its fields. progress: called with each step's record."""
    opt = Options.make(options)
    t_all = time.perf_counter()
    dev = torch.device(device)
    backend = opt.backend if opt.backend != "auto" else ("metal" if dev.type == "mps" else "torch")
    if backend == "metal" and (dev.type != "mps" or dtype != torch.float32):
        raise ValueError("motion_eddy: the metal backend runs on MPS in float32")
    dwi = np.asarray(dwi)
    if dwi.ndim != 4:
        raise ValueError("motion_eddy: dwi (X, Y, Z, G)")
    X, Y, Z, G = dwi.shape
    bval = np.asarray(bval, np.float64)
    bvec = np.asarray(bvec, np.float64)
    if bval.shape != (G,) or bvec.shape != (3, G):
        raise ValueError(f"motion_eddy: bval (G,), bvec (3, G) for G = {G}")
    vox = np.sqrt((np.asarray(affine, np.float64)[:3, :3] ** 2).sum(0))
    has_field = field_hz is not None
    if has_field and (pe_sign is None or readout_s is None):
        raise ValueError("motion_eddy: a field needs pe_sign and readout_s")
    reference = opt.reference if opt.reference != "auto" else ("first-b0" if has_field else "b0-mean")
    npass = opt.npass or (3 if has_field else 2)
    geo = Geometry((X, Y, Z), vox, pe_axis, ORDERS[opt.model], dev, dtype)
    P, NMON = geo.P, geo.nmon

    isb0 = bval < 50
    groups, group_b = shell_groups(bval)
    g = np.zeros((G, 3))
    nrm = np.linalg.norm(bvec.T, axis=1)
    g[~isb0] = bvec.T[~isb0] / np.clip(nrm[~isb0, None], 1e-9, None)
    W_np, sh_orders = loo_smoother(groups, isb0, g, opt.sh_order, opt.sh_lambda)
    W = torch.tensor(W_np, dtype=dtype, device=dev)

    pe_scale = float(pe_sign) * float(readout_s) if has_field else None
    dfield_np = (pe_scale * np.asarray(field_hz, np.float64)) if has_field else None
    t_m = time.perf_counter()
    m_ref, m_pts, m_pred = _masks(dwi, bval, dfield_np, pe_axis, mask)
    t_mask = time.perf_counter() - t_m
    from scipy.ndimage import binary_erosion
    edge = m_ref & ~binary_erosion(m_ref, iterations=1)

    acq = torch.tensor(np.ascontiguousarray(np.moveaxis(dwi, -1, 0)), dtype=dtype, device=dev)
    dfield = torch.tensor(dfield_np, dtype=dtype, device=dev) if has_field else None
    ddfield = (torch.gradient(torch.as_tensor(dfield_np), dim=pe_axis)[0].to(dev, dtype) if has_field else None)
    full = lambda m: torch.tensor(np.argwhere(m), dtype=dtype, device=dev).contiguous()     # noqa: E731
    ye, xp, xedge = full(m_pts), full(m_pred), full(edge)
    lin_e = torch.tensor(np.flatnonzero(m_pts.reshape(-1)), device=dev)
    lin_p = torch.tensor(np.flatnonzero(m_pred.reshape(-1)), device=dev)
    sub = torch.tensor(np.flatnonzero((np.argwhere(m_pts) % 2 == 0).all(1)), device=dev)
    probe = xedge[torch.randperm(len(xedge), generator=torch.Generator(device="cpu").manual_seed(1))[:2000].to(dev)]

    # held out of the solve: the b0s' eddy terms (zero at b = 0) unless b0_ec; every volume's eddy constant (the PE
    # translation by another name); a volume alone in its group (nothing predicts it); the first b0's rigid 6 when it is
    # the reference
    frozen = torch.zeros(G, P, dtype=torch.bool, device=dev)
    b0_t = torch.tensor(isb0, device=dev)
    if NMON:
        if isb0.any() and not opt.b0_ec:
            frozen[b0_t, 6:] = True
        frozen[:, 6] = True
    for idx in groups:
        if len(idx) < 2:
            frozen[int(idx[0])] = True
    if reference == "first-b0" and not isb0.any():
        raise ValueError("motion_eddy: reference 'first-b0' needs a b0 (b < 50)")
    if has_field and reference == "b0-mean":
        import warnings
        warnings.warn("motion_eddy: reference 'b0-mean' with a field: the field is in the first b0's frame, and the "
                      "b0s' mean pose may differ from it", stacklevel=2)
    ref_vol = int(np.flatnonzero(isb0)[0]) if isb0.any() else 0
    if reference == "first-b0":
        frozen[ref_vol, :6] = True
    keep = (~frozen).to(dtype)
    # each group's mean pose is invisible to its own predictor. The b0s: with reference "b0-mean" their mean pose is
    # held at 0 (the location reference); with "first-b0" the first b0 is held instead and the others are free. Each DW
    # shell's mean pose is held at the b0s' mean pose (interleaved b0s sample the session's mean position; a single b0
    # does not), whichever the reference; without b0s, at 0
    b0_idx = torch.tensor(np.flatnonzero(isb0), device=dev) if isb0.any() else None
    b0_pose = b0_idx if (b0_idx is not None and reference == "b0-mean") else None
    shell_poses = [torch.tensor(idx, device=dev) for idx in groups if not isb0[idx].all() and len(idx) > 1]
    ec_gauge = []
    if NMON > 1:
        for idx in groups:
            if isb0[idx].all() or len(idx) < 5:
                continue
            Xg = np.c_[np.ones(len(idx)), g[idx]]
            ec_gauge.append((torch.tensor(idx, device=dev), torch.tensor(np.linalg.pinv(Xg)[0], dtype=dtype, device=dev)))

    def gauge(th):
        for idx, w0 in ec_gauge:
            th[idx, 7:] -= w0 @ th[idx, 7:]
        if b0_pose is not None:
            th[b0_pose, :6] -= th[b0_pose, :6].mean(0)
        target = th[b0_idx, :6].mean(0) if b0_idx is not None else 0.0
        for idx in shell_poses:
            th[idx, :6] -= th[idx, :6].mean(0) - target
        if opt.b0_ec and NMON and int(b0_t.sum()) > 1:
            th[b0_t, 7:] -= th[b0_t, 7:].mean(0)
        return th

    nz = np.argwhere(m_pred)
    lo = np.maximum(nz.min(0) - 4, 0)
    hi = np.minimum(nz.max(0) + 5, [X, Y, Z])
    crop = (slice(None),) + tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    offset = tuple(float(a) for a in lo)
    if backend == "metal":
        from . import motion_eddy_metal as M
        dummy = torch.zeros(1, dtype=torch.float32, device=dev)
        dfk = dfield.contiguous() if has_field else dummy
        ddfk = ddfield.contiguous() if has_field else dummy
        rs = M.consts(geo, G, len(xp), (X, Y, Z), npass=npass, has_field=has_field)
        mcache = {}
        smooth_fn, prefilter_fn = M.smooth, M.prefilter
    else:
        smooth_fn, prefilter_fn = t_smooth, t_prefilter

    th = torch.zeros(G, P, dtype=dtype, device=dev)
    mu = torch.full((G,), 1e-3, dtype=dtype, device=dev)
    fw_steps = ([2.0, 1.0][:opt.smooth_steps] + [0.0] * opt.steps)[:opt.steps]
    smooth_cache = {}
    history, quiet, steps_run, ntrials = [], 0, 0, 0
    f_last = None
    with torch.no_grad():
        for it, fw in enumerate(fw_steps):
            if fw not in smooth_cache:
                smooth_cache[fw] = smooth_fn(acq, fw)
            vols = smooth_cache[fw]
            obs_full = vols.reshape(G, -1)[:, lin_e]
            if backend == "metal":
                u = M.resample(vols.contiguous(), xp, pack(th), rs[1], rs[2], dfk)
            else:
                u = t_resample(geo, vols, xp, th, dfield)
            pv = torch.zeros(G, X * Y * Z, dtype=dtype, device=dev)
            pv[:, lin_p] = W @ u
            coef = prefilter_fn(pv.reshape(G, X, Y, Z)[crop])
            c_ = it < opt.coarse
            ys, obs = (ye[sub], obs_full[:, sub]) if c_ else (ye, obs_full)
            if backend == "metal":
                if c_ not in mcache:
                    mcache[c_] = M.consts(geo, G, len(ys), tuple(coef.shape[1:]), offset, npass, has_field)
                nch, ipk, fpk = mcache[c_]
                ys_c, obs_c, coef_c = ys.contiguous(), obs.contiguous(), coef.contiguous()
                H, b, f0 = M.step_moments(coef_c, obs_c, ys_c, pack(th), nch, ipk, fpk, dfk, ddfk, P)
            else:
                H, b, f0 = t_step_moments(geo, coef, obs, ys, th, offset, dfield, npass, ddfield=ddfield)
            Hd = torch.diagonal(H, dim1=1, dim2=2)
            accept = torch.zeros(G, dtype=torch.bool, device=dev)
            th_new = th.clone()
            for _ls in range(6):
                todo = ~accept
                if not bool(todo.any()):
                    break
                A_ = H + torch.diag_embed(mu[:, None] * Hd)
                A_ = A_ * keep[:, :, None] * keep[:, None, :] + torch.diag_embed(frozen.to(dtype))
                step = -torch.linalg.solve(A_.cpu(), (b * keep).cpu()[..., None])[..., 0].to(dev) * geo.scale
                if NMON and isb0.any() and not opt.b0_ec:
                    step[b0_t, 6:] = 0
                step = step * keep
                cand = th + step
                big = (phi(geo, cand, probe) - phi(geo, th, probe)).norm(dim=-1).amax(1) > opt.trial_vox
                need = todo & (big | (it < 2))
                free = todo & ~need
                th_new = torch.where(free[:, None], cand, th_new)
                accept |= free
                mu = torch.where(free, (mu / 3).clamp_min(1e-7), mu)
                if not bool(need.any()):
                    break
                k = torch.nonzero(need)[:, 0]
                if backend == "metal":
                    f1 = M.step_cost(coef_c, obs_c, ys_c, pack(cand[k]), k, nch, ipk, fpk, dfk, ddfk)
                else:
                    f1 = t_step_cost(geo, coef, obs, ys, cand[k], k, offset, dfield, npass, ddfield)
                ntrials += 1
                ok = torch.zeros(G, dtype=torch.bool, device=dev)
                ok[k] = f1 < f0[k]
                th_new = torch.where(ok[:, None], cand, th_new)
                accept |= ok
                bad = need & ~ok
                mu = torch.where(ok, (mu / 3).clamp_min(1e-7), torch.where(bad, mu * 8, mu))
                if not bool(bad.any()):
                    break
            moved = float((phi(geo, th_new, probe) - phi(geo, th, probe)).norm(dim=-1).amax())
            th = gauge(th_new)
            if not c_:
                f_last = (f0 / len(ys)).cpu().numpy()
            quiet = quiet + 1 if (opt.tol and fw == 0 and moved < opt.tol) else 0
            steps_run = it + 1
            rec = {"step": it, "fwhm": fw, "coarse": bool(c_), "accepted": int(accept.sum()),
                   "largest move (vox)": round(moved, 4)}
            history.append(rec)
            if progress:
                progress(rec)
            if quiet >= 2:
                break
    if backend == "metal":
        torch.mps.synchronize()
    t_fit = time.perf_counter() - t_all - t_mask

    anchors = {}
    if opt.anchor == "mi" and isb0.any():
        th, anchors = _shell_anchor(geo, acq, th, groups, isb0, group_b, m_ref, dfield, ddfield, opt, backend)
    seconds = {"masks": round(t_mask, 2), "fit": round(t_fit, 2), "total": round(time.perf_counter() - t_all, 2)}

    thn = th.cpu().double().numpy()
    th_fit = thn.copy()
    dw = ~isb0
    if NMON and dw.sum() > 3:                                                             # eddy's PE-offset split
        off = thn[dw, pe_axis] / vox[pe_axis] + thn[dw, 6]
        Xd = (bval[dw] / 1000.0)[:, None] * bvec[:, dw].T
        Xd = Xd - Xd.mean(0)
        off_hat = Xd @ np.linalg.lstsq(Xd, off, rcond=None)[0]
        thn[dw, 6] = off_hat
        thn[dw, pe_axis] = (off - off_hat) * vox[pe_axis]
    bv_out = rotate_bvecs(bvec, thn[:, 3:6], affine)
    bv_out[:, isb0] = bvec[:, isb0]
    qc = _qc(geo, thn, th_fit, xedge, groups, group_b, isb0, f_last, steps_run, opt, history, sh_orders, ntrials)
    return MotionEddy(theta=thn, theta_fit=th_fit, bvec=bv_out, model=opt.model, pe_axis=int(pe_axis), shape=(X, Y, Z),
                      vox=vox, groups=[np.asarray(i) for i in groups], group_b=group_b, reference=reference,
                      field_used=has_field, pe_scale=pe_scale, anchors=anchors, qc=qc, options=asdict(opt),
                      seconds=seconds, backend=backend)


def rotate_bvecs(bvec, w, affine):
    """FSL b-vectors (3, G) rotated with the head: bvec' = F R^T F bvec, F the x flip when det(affine) > 0 (FSL's
    radiological convention), R the volume's rotation (reference -> acquisition, so its transpose; design doc 1.3)."""
    Fl = np.diag([-1.0, 1.0, 1.0]) if np.linalg.det(np.asarray(affine, np.float64)[:3, :3]) > 0 else np.eye(3)
    R = rot(torch.as_tensor(np.asarray(w, np.float64))).numpy()
    return np.einsum("ij,gkj,kl,lg->ig", Fl, R, Fl, np.asarray(bvec, np.float64))


def _qc(geo, thn, th_fit, xedge, groups, group_b, isb0, f_last, steps_run, opt, history, sh_orders, ntrials):
    """The QC summary: per volume, the displacement at the brain's edge from motion (rigid) and from eddy currents
    (mm; RMS and max over the edge), framewise displacement (Power et al. 2012: translations plus rotations on a 50 mm
    sphere, between consecutive volumes, movement only: the eddy-split parameters), and the residual against the
    leave-one-out prediction (RMS) at the start of the last full-resolution step, with a z-score within each group (a
    robust one: median, MAD)."""
    th = torch.as_tensor(thn, dtype=torch.float64)
    gd = Geometry(geo.shape, geo.vox.cpu().double().numpy(), geo.pe_axis, geo.nmon, "cpu", torch.float64)
    xe = xedge.cpu().double()
    rig = th.clone()
    rig[:, 6:] = 0
    d_rigid = ((phi(gd, rig, xe) - xe) * gd.vox).norm(dim=-1)                                # (G, N) mm
    d_eddy = (eddy_poly(gd, th[:, 6:], xe).abs() * gd.vox[gd.pe_axis]) if gd.nmon else torch.zeros_like(d_rigid)
    fd = np.r_[0.0, np.abs(np.diff(thn[:, :3], axis=0)).sum(1) + 50.0 * np.abs(np.diff(thn[:, 3:6], axis=0)).sum(1)]
    q = {"volumes": int(len(thn)), "steps run": int(steps_run), "LM trials": int(ntrials),
         "converged (tol)": bool(opt.tol and steps_run < opt.steps),
         "rigid displacement at edge, mm (rms, max) per volume": np.stack(
             [d_rigid.pow(2).mean(1).sqrt().numpy(), d_rigid.amax(1).numpy()], 1).round(4).tolist(),
         "eddy displacement at edge, mm (rms, max) per volume": np.stack(
             [d_eddy.pow(2).mean(1).sqrt().numpy(), d_eddy.amax(1).numpy()], 1).round(4).tolist(),
         "framewise displacement, mm per volume": fd.round(4).tolist(),
         "framewise displacement, mm (mean, max)": [round(float(fd.mean()), 4), round(float(fd.max()), 4)],
         "groups": [{"b": int(b), "volumes": int(len(i)), "sh order": sh_orders.get(int(i[0]))}
                    for b, i in zip(group_b, groups)],
         "volumes not corrected (alone in their group: nothing predicts them)": sorted(
             int(i[0]) for i in groups if len(i) < 2)}
    if f_last is not None:
        rms = np.sqrt(np.clip(np.asarray(f_last, np.float64), 0, None))
        z = np.zeros_like(rms)
        for idx in groups:
            if len(idx) > 2:
                med = np.median(rms[idx])
                mad = 1.4826 * np.median(np.abs(rms[idx] - med)) or 1.0
                z[idx] = (rms[idx] - med) / mad
        q["residual vs prediction, rms per volume"] = rms.round(4).tolist()
        q["residual z within group per volume"] = z.round(3).tolist()
        q["volumes with residual z > 4"] = np.flatnonzero(z > 4).tolist()
    return q


# ------------------------------------------------------------------------------------------------- the shell anchor

def _shell_anchor(geo, acq, th, groups, isb0, group_b, m_ref, dfield, ddfield, opt, backend):
    """Each DW shell's corrected mean registered to the corrected mean b0 by mutual information (tractline.mi), the
    shift added to the shell's volumes (first order: small shifts compose additively). (theta, {shell: values})."""
    from scipy.ndimage import binary_dilation
    from . import mi
    corr, _ = _apply(geo, acq, th, dfield, backend, ddfield)
    b0g = [i for i in groups if isb0[i[0]]]
    if not b0g:
        return th, {}
    sm = (lambda v: t_smooth(v, 1.0))                                                     # noqa: E731
    dev = acq.device
    fixed = sm(corr[torch.tensor(b0g[0], device=dev)].mean(0, keepdim=True))[0]
    mk = torch.tensor(binary_dilation(m_ref, iterations=2), device=dev)
    th = th.clone()
    out = {}
    for idx, b in zip(groups, group_b):
        if isb0[idx[0]]:
            continue
        mov = sm(corr[torch.tensor(idx, device=dev)].mean(0, keepdim=True))[0]
        dd = mi.register(mov, fixed, mk, geo.vox.cpu().numpy(), pe_axis=geo.pe_axis,
                         nmon=geo.nmon if opt.anchor_eddy else 0, bins=opt.anchor_bins).to(th.dtype)
        ii = torch.tensor(idx, device=dev)
        th[ii, :3] += dd[:3]
        th[ii, 3:6] += dd[3:6] * 0.01
        if dd.shape[0] > 6:
            th[ii, 6:] += dd[6:]
        out[f"b{b}"] = [round(float(v), 4) for v in dd[:3]] + [round(math.degrees(float(v) * 0.01), 4) for v in dd[3:6]] \
            + [round(float(v), 4) for v in dd[6:]]
    return th, out


# ------------------------------------------------------------------------------------------------- applying

def _apply(geo, vols, theta, dfield, backend, ddfield=None):
    if backend == "metal":
        from . import motion_eddy_metal as M
        dummy = torch.zeros(1, dtype=torch.float32, device=vols.device)
        nch, ip, fp = M.consts(geo, vols.shape[0], 1, tuple(vols.shape[1:]), has_field=dfield is not None)
        return M.apply_volumes(vols.contiguous(), pack(theta), ip, fp,
                               dfield.contiguous() if dfield is not None else dummy,
                               ddfield.contiguous() if ddfield is not None else dummy)
    return t_apply_volumes(geo, vols, theta, dfield, ddfield)


def apply_volumes(dwi, result: MotionEddy, field_hz=None, device="mps", dtype=torch.float32, mask_output=None,
                  backend="auto"):
    """The corrected series (X, Y, Z, G) float32 and its validity (X, Y, Z) bool: each volume resampled once (3D cubic
    B-spline) at phi_v(x), composing its motion, its eddy displacement and the field's (field_hz, Hz, the one
    estimate() was given), times the displacement's Jacobian along PE, clipped at 0. Voxels some volume's map leaves
    the field of view for are not valid; mask_output (default: the estimate's option) sets them to 0 in every volume,
    as eddy's output mask."""
    if (field_hz is not None) != result.field_used:
        raise ValueError("motion_eddy.apply_volumes: give the field estimate() was given (or none, if none)")
    dev = torch.device(device)
    backend = backend if backend != "auto" else ("metal" if dev.type == "mps" else "torch")
    geo = Geometry(result.shape, result.vox, result.pe_axis, ORDERS[result.model], dev, dtype)
    vols = torch.tensor(np.ascontiguousarray(np.moveaxis(np.asarray(dwi), -1, 0)), dtype=dtype, device=dev)
    th = torch.tensor(result.theta_fit, dtype=dtype, device=dev)
    dfield = ddfield = None
    if field_hz is not None:
        h = torch.as_tensor(np.asarray(field_hz, np.float64))
        dfield = (result.pe_scale * h).to(dev, dtype)
        ddfield = (result.pe_scale * torch.gradient(h, dim=result.pe_axis)[0]).to(dev, dtype)
    with torch.no_grad():
        out, valid = _apply(geo, vols, th, dfield, backend, ddfield)
    common = (valid.amin(0) > 0)
    mo = result.options["mask_output"] if mask_output is None else mask_output
    if mo:
        out = out * common
    return np.moveaxis(out.cpu().numpy(), 0, -1).astype(np.float32), common.cpu().numpy()


# ------------------------------------------------------------------------------------------------- all three, one resample

@dataclass
class Corrected:
    """correct()'s result: the series corrected for the susceptibility field, head motion and eddy currents in one
    resample, its rotated b-vectors, and what was estimated."""
    dwi: np.ndarray                  # (X, Y, Z, G) float32
    bvec: np.ndarray                 # (3, G) FSL, rotated with the head
    valid: np.ndarray                # (X, Y, Z) bool: inside every volume's field of view
    motion_eddy: MotionEddy
    field_hz: np.ndarray | None      # (X, Y, Z) Hz; None without a reversed pair (motion and eddy only, as eddy alone)
    b0_motion: np.ndarray | None     # (V, 6) the field estimate's own motion of the b0s it used
    note: str = ""                   # what was corrected, or why the field was not
    seconds: dict = field(default_factory=dict)


def correct(s, device="mps", options=None, mask=None) -> Corrected:
    """The three corrections that FSL makes with topup, eddy and their resampling, in one call and one resample: the
    susceptibility field from the b0s of both polarities (susceptibility.estimate, topup's model; as pipeline.correct
    estimates it), the motion and eddy currents of every volume given that field (estimate()), and each volume
    resampled once with all three composed (apply_volumes()). s: a subject as pipeline.run takes it (dwi, bval, bvec,
    affine, vox, pe_axis, pe_sign, readout_s, b0s, pe_vectors, and b0_readout_s when bids.load made it). Without a pair
    to estimate the field from (pipeline.can_correct), motion and eddy currents only."""
    import os
    from . import pipeline as P
    from . import susceptibility as S
    t0 = time.perf_counter()
    h = b0_motion = None
    if P.can_correct(s):
        f32 = torch.device(device).type in ("cpu", "cuda")
        ro = getattr(s, "b0_readout_s", None)
        with P.threads(min(P.ESTIMATE_THREADS, os.cpu_count() or 1)):
            h, b0_motion, _ = S.estimate(s.b0s, s.vox, s.pe_vectors, s.readout_s if ro is None else ro, device=device,
                                         **(dict(dtype=torch.float32) if f32 else {}))
    t1 = time.perf_counter()
    with P.exact_float32(device):                                       # no TF32 on CUDA (pipeline.correct's rule)
        r = estimate(s.dwi, s.bval, s.bvec, s.affine, s.pe_axis, field_hz=h, pe_sign=s.pe_sign, readout_s=s.readout_s,
                     mask=mask, options=options, device=device)
        t2 = time.perf_counter()
        dwi, valid = apply_volumes(s.dwi, r, field_hz=h, device=device)
    pairing = getattr(s, "pairing", None)
    note = ("field, motion and eddy currents" if h is not None else
            "motion and eddy currents only: " + (pairing.message if pairing is not None and not pairing.ok
                                                 else "no b0s from both phase-encoding polarities"))
    return Corrected(dwi=dwi, bvec=r.bvec, valid=valid, motion_eddy=r, field_hz=h, b0_motion=b0_motion, note=note,
                     seconds={"field": round(t1 - t0, 2), "motion_eddy": round(t2 - t1, 2),
                              "apply": round(time.perf_counter() - t2, 2)})


# ------------------------------------------------------------------------------------------------- the record and QC files

def record(r: MotionEddy) -> dict:
    """What estimate() found, JSON-ready (pipeline's Correction.motion_eddy): the model and frame, each group's volumes,
    the per-volume parameters (theta: translations mm, rotations rad, eddy coefficients voxels along PE, the PE offset
    split as eddy splits it; theta_fit as fitted), the shell anchors, the QC summary, the options and the times."""
    return {"model": r.model, "reference": r.reference, "field used": r.field_used, "pe_axis": r.pe_axis,
            "groups (b, volumes)": [[int(b), [int(i) for i in idx]] for b, idx in zip(r.group_b, r.groups)],
            "theta": np.round(r.theta, 6).tolist(), "theta_fit": np.round(r.theta_fit, 6).tolist(),
            "shell anchors": r.anchors, "qc": r.qc, "options": r.options, "seconds": r.seconds, "backend": r.backend}


QC_COLUMNS = ("volume", "b", "tx_mm", "ty_mm", "tz_mm", "rx_deg", "ry_deg", "rz_deg", "rigid_edge_rms_mm",
              "rigid_edge_max_mm", "eddy_edge_rms_mm", "eddy_edge_max_mm", "framewise_displacement_mm",
              "residual_rms", "residual_z")


def qc_rows(rec: dict) -> list:
    """One row per volume (QC_COLUMNS, then the eddy coefficients e0.. in voxels along PE) from a record()."""
    th = np.asarray(rec["theta"], float)
    q = rec["qc"]
    b = np.zeros(len(th))
    for bv, idx in rec["groups (b, volumes)"]:
        b[idx] = bv
    rig, edd = np.asarray(q["rigid displacement at edge, mm (rms, max) per volume"]), \
        np.asarray(q["eddy displacement at edge, mm (rms, max) per volume"])
    fd = q["framewise displacement, mm per volume"]
    res = q.get("residual vs prediction, rms per volume", [None] * len(th))
    z = q.get("residual z within group per volume", [None] * len(th))
    rows = []
    for v in range(len(th)):
        row = [v, int(b[v]), *np.round(th[v, :3], 4), *np.round(np.degrees(th[v, 3:6]), 4), *np.round(rig[v], 4),
               *np.round(edd[v], 4), round(fd[v], 4), res[v], z[v], *np.round(th[v, 6:], 4)]
        rows.append([float(x) if isinstance(x, (np.floating, float)) else x for x in row])
    return rows


def write_qc(rec: dict, folder) -> dict:
    """The QC files for a run with the motion_eddy slot, in `folder`: motion_eddy_qc.tsv (one row per volume:
    QC_COLUMNS and the eddy coefficients) and motion_eddy_qc.json (the summary: model, frame, field, framewise
    displacement, the largest displacements, the volumes that fit their prediction worst, the times, the options).
    Returns the summary. The translation along PE is each volume's PE offset less the part linear in its gradient
    (eddy's split, given to the eddy currents): its volume-to-volume jitter, and so part of the framewise
    displacement, can be that split's residue rather than head motion."""
    import json
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = qc_rows(rec)
    ncoef = len(rec["theta"][0]) - 6
    header = list(QC_COLUMNS) + [f"eddy_e{k}_vox" for k in range(ncoef)]
    with open(folder / "motion_eddy_qc.tsv", "w") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            f.write("\t".join("" if x is None else str(x) for x in row) + "\n")
    q = rec["qc"]
    rig = np.asarray(q["rigid displacement at edge, mm (rms, max) per volume"])
    edd = np.asarray(q["eddy displacement at edge, mm (rms, max) per volume"])
    summary = {"model": rec["model"], "reference frame": rec["reference"], "field used": rec["field used"],
               "volumes": q["volumes"], "steps run": q["steps run"],
               "framewise displacement mm (mean, max)": q["framewise displacement, mm (mean, max)"],
               "rigid displacement at the brain's edge, mm: median and largest RMS (volume, mm)": [
                   round(float(np.median(rig[:, 0])), 3), int(rig[:, 0].argmax()), round(float(rig[:, 0].max()), 3)],
               "rigid displacement at the brain's edge, mm: the largest at any edge point (volume, mm)": [
                   int(rig[:, 1].argmax()), round(float(rig[:, 1].max()), 3)],
               "eddy displacement at the brain's edge, mm: median and largest RMS (volume, mm)": [
                   round(float(np.median(edd[:, 0])), 3), int(edd[:, 0].argmax()), round(float(edd[:, 0].max()), 3)],
               "eddy displacement at the brain's edge, mm: the largest at any edge point (volume, mm; the quadratic's "
               "corners)": [int(edd[:, 1].argmax()), round(float(edd[:, 1].max()), 3)],
               "volumes with residual z > 4 (worst fit to their prediction)": q.get("volumes with residual z > 4"),
               "volumes not corrected (alone in their group)":
                   q.get("volumes not corrected (alone in their group: nothing predicts them)"),
               "groups": q["groups"], "shell anchors": rec["shell anchors"], "seconds": rec["seconds"],
               "options": rec["options"], "table": "motion_eddy_qc.tsv"}
    (folder / "motion_eddy_qc.json").write_text(json.dumps(summary, indent=1))
    return summary
