"""tractline.motion_eddy's inner loop and final resample as Triton kernels, for CUDA: the computations of
tractline.motion_eddy_metal's Metal kernels (read there for the method), in Triton's block form. The plain-torch versions
of the same pieces are in tractline.motion_eddy (the CPU path, and the float64 reference the tests hold these to).

A program takes one volume and a block of points (or voxels). Per point:
  - x = phi^-1(y): the rigid part in closed form with the displacement along PE taken at y, then Newton steps along PE;
  - the prediction's cubic B-spline value and exact gradient at x, the 64 coefficients gathered once each, edges clamped;
  - the residual and the Jacobian by the chain rule through phi^-1 at x (motion_eddy._jacobian).
step_moments forms the block's J (BLOCK x 16) and writes J'J (16 x 16, one tl.dot in IEEE float32), J'r and r^2 per
block; torch sums the blocks in a fixed order, so a run is deterministic on a device. step_cost: r^2 only, for the
volumes listed (the Levenberg-Marquardt trials); step_values: the prediction at each point; resample: the forward map
and a trilinear sample; apply_volumes: the final cubic resample times the displacement's Jacobian along PE. The cubic
prefilter and the Gaussian smoothing stay in torch (dense products and a convolution, already fast on CUDA).

The PE axis, the number of eddy terms, the Newton passes and whether a field is given are compile-time constants:
Triton compiles one copy per combination met (a few), cached on disk across processes (TRITON_CACHE_DIR, by default
~/.triton/cache). The sizes are not specialized on, so a scan of other dimensions reuses the copy.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:                                       # no Triton (macOS, a CPU-only torch): the torch path runs
    triton = None

BLOCK = 256                                               # points (or voxels) per program
NPV = 49                                                  # per volume: R (9, row-major), t (3), eddy (10), dR_j (27)
NOUT = 16 * 16 + 16 + 1                                   # per block: J'J, J'r, r^2


def cpus() -> int:
    """The CPUs this process may run on (a container's or Slurm's allocation, where the platform says), at least 1."""
    import os
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:                                # macOS, Windows
        return max(1, os.cpu_count() or 1)


def available(device=None) -> bool:
    """Triton is installed and a CUDA GPU it compiles for (compute capability 7.0 or newer) is there."""
    if triton is None or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(device) >= (7, 0)


class Consts:
    """What the kernels need beyond the arrays: sizes, the compile-time switches, and fp (cen, half, vox, scale (16),
    the sampled grid's offset in the full grid), the same layout as motion_eddy_metal's."""

    def __init__(self, geo, G, N, grid, offset, npass, has_field):
        self.G, self.N = int(G), int(N)
        self.X, self.Y, self.Z = (int(n) for n in grid)
        self.FX, self.FY, self.FZ = geo.shape
        self.pe, self.nmon, self.npass, self.hf = geo.pe_axis, geo.nmon, int(npass), int(bool(has_field))
        self.nch = (self.N + BLOCK - 1) // BLOCK
        sc = torch.zeros(16, dtype=torch.float32)
        sc[:geo.P] = geo.scale.float().cpu()
        self.fp = torch.cat([geo.cen.float().cpu(), geo.half.float().cpu(), geo.vox.float().cpu(), sc,
                             torch.tensor(offset, dtype=torch.float32)]).to(geo.device)


def consts(geo, G, N, grid, offset=(0, 0, 0), npass=2, has_field=False):
    """(nch, Consts, fp), as motion_eddy_metal.consts's (nch, ip, fp): the estimator passes [1] and [2] back."""
    c = Consts(geo, G, N, grid, offset, npass, has_field)
    return c.nch, c, c.fp


def prefilter(vols):
    from .motion_eddy import t_prefilter
    return t_prefilter(vols)


def smooth(vols, fwhm_vox):
    from .motion_eddy import t_smooth
    return t_smooth(vols, fwhm_vox)


#: each kernel's arguments that are not compile-time constants: none specialized on (Triton would otherwise compile a
#: copy per size divisible by 16 or equal to 1, and per pointer alignment), so one copy serves every scan and warm()'s
#: copies are the ones a run uses
_UNSPEC = {'_step_moments': ['coef', 'obs', 'pts', 'pvol', 'part', 'fp', 'dfield', 'ddfield', 'N', 'NCH', 'X', 'Y', 'Z', 'FX', 'FY', 'FZ'],
           '_step_cost': ['coef', 'obs', 'pts', 'pvol', 'vols', 'part', 'fp', 'dfield', 'ddfield', 'N', 'NCH', 'X', 'Y', 'Z', 'FX', 'FY', 'FZ'],
           '_step_values': ['coef', 'pts', 'pvol', 'vols', 'out', 'fp', 'dfield', 'ddfield', 'N', 'X', 'Y', 'Z', 'FX', 'FY', 'FZ'],
           '_resample': ['vols', 'pts', 'pvol', 'out', 'fp', 'dfield', 'N', 'X', 'Y', 'Z', 'FX', 'FY', 'FZ'],
           '_apply_volumes': ['coef', 'pvol', 'out', 'valid', 'fp', 'dfield', 'ddfield', 'X', 'Y', 'Z', 'FX', 'FY', 'FZ']}

if triton is not None:
    _NPV = tl.constexpr(NPV)                              # a module global a kernel may read

    # ------------------------------------------------------------------------------------------------- helpers

    @triton.jit
    def _pick(v0, v1, v2, PE: tl.constexpr):
        if PE == 0:
            r = v0
        elif PE == 1:
            r = v1
        else:
            r = v2
        return r

    @triton.jit
    def _eddy(te, u0, u1, u2, NMON: tl.constexpr):
        e = tl.zeros_like(u0)
        if NMON > 0:
            e = tl.load(te) + tl.load(te + 1) * u0 + tl.load(te + 2) * u1 + tl.load(te + 3) * u2
        if NMON == 10:
            e += (tl.load(te + 4) * u0 * u0 + tl.load(te + 5) * u1 * u1 + tl.load(te + 6) * u2 * u2
                  + tl.load(te + 7) * u0 * u1 + tl.load(te + 8) * u0 * u2 + tl.load(te + 9) * u1 * u2)
        return e

    @triton.jit
    def _eddy_grad(te, u0, u1, u2, h0, h1, h2, NMON: tl.constexpr):
        """d e / d x (voxels per voxel) at u = (x - cen) / half."""
        g0 = tl.zeros_like(u0)
        g1 = tl.zeros_like(u0)
        g2 = tl.zeros_like(u0)
        if NMON > 0:
            g0 += tl.load(te + 1)
            g1 += tl.load(te + 2)
            g2 += tl.load(te + 3)
        if NMON == 10:
            g0 += 2 * tl.load(te + 4) * u0 + tl.load(te + 7) * u1 + tl.load(te + 8) * u2
            g1 += 2 * tl.load(te + 5) * u1 + tl.load(te + 7) * u0 + tl.load(te + 9) * u2
            g2 += 2 * tl.load(te + 6) * u2 + tl.load(te + 8) * u0 + tl.load(te + 9) * u1
        return g0 / h0, g1 / h1, g2 / h2

    @triton.jit
    def _corners(d, x0, x1, x2, FX, FY, FZ):
        """A full-grid volume's 8 trilinear corners around x (clamped to the grid), the fractions, and whether each
        axis's pair collapsed or the position was clamped (the gradient is zero along such an axis)."""
        p0 = tl.minimum(tl.maximum(x0, 0.0), (FX - 1).to(tl.float32))
        p1 = tl.minimum(tl.maximum(x1, 0.0), (FY - 1).to(tl.float32))
        p2 = tl.minimum(tl.maximum(x2, 0.0), (FZ - 1).to(tl.float32))
        f0 = tl.floor(p0)
        f1 = tl.floor(p1)
        f2 = tl.floor(p2)
        t0, t1, t2 = p0 - f0, p1 - f1, p2 - f2
        i0, i1, i2 = f0.to(tl.int32), f1.to(tl.int32), f2.to(tl.int32)
        j0, j1, j2 = tl.minimum(i0 + 1, FX - 1), tl.minimum(i1 + 1, FY - 1), tl.minimum(i2 + 1, FZ - 1)
        c000 = tl.load(d + (i0 * FY + i1) * FZ + i2)
        c001 = tl.load(d + (i0 * FY + i1) * FZ + j2)
        c010 = tl.load(d + (i0 * FY + j1) * FZ + i2)
        c011 = tl.load(d + (i0 * FY + j1) * FZ + j2)
        c100 = tl.load(d + (j0 * FY + i1) * FZ + i2)
        c101 = tl.load(d + (j0 * FY + i1) * FZ + j2)
        c110 = tl.load(d + (j0 * FY + j1) * FZ + i2)
        c111 = tl.load(d + (j0 * FY + j1) * FZ + j2)
        z0 = (x0 < 0.0) | (x0 > (FX - 1).to(tl.float32)) | (i0 == j0)
        z1 = (x1 < 0.0) | (x1 > (FY - 1).to(tl.float32)) | (i1 == j1)
        z2 = (x2 < 0.0) | (x2 > (FZ - 1).to(tl.float32)) | (i2 == j2)
        return c000, c001, c010, c011, c100, c101, c110, c111, t0, t1, t2, z0, z1, z2

    @triton.jit
    def _field(d, x0, x1, x2, FX, FY, FZ):
        c000, c001, c010, c011, c100, c101, c110, c111, t0, t1, t2, z0, z1, z2 = _corners(d, x0, x1, x2, FX, FY, FZ)
        c00 = c000 + (c001 - c000) * t2
        c01 = c010 + (c011 - c010) * t2
        c10 = c100 + (c101 - c100) * t2
        c11 = c110 + (c111 - c110) * t2
        c0 = c00 + (c01 - c00) * t1
        c1 = c10 + (c11 - c10) * t1
        return c0 + (c1 - c0) * t0

    @triton.jit
    def _field_grad(d, x0, x1, x2, FX, FY, FZ):
        c000, c001, c010, c011, c100, c101, c110, c111, t0, t1, t2, z0, z1, z2 = _corners(d, x0, x1, x2, FX, FY, FZ)
        g0 = ((1 - t1) * (1 - t2) * (c100 - c000) + t1 * (1 - t2) * (c110 - c010)
              + (1 - t1) * t2 * (c101 - c001) + t1 * t2 * (c111 - c011))
        g1 = ((1 - t0) * (1 - t2) * (c010 - c000) + t0 * (1 - t2) * (c110 - c100)
              + (1 - t0) * t2 * (c011 - c001) + t0 * t2 * (c111 - c101))
        g2 = ((1 - t0) * (1 - t1) * (c001 - c000) + t0 * (1 - t1) * (c101 - c100)
              + (1 - t0) * t1 * (c011 - c010) + t0 * t1 * (c111 - c110))
        return tl.where(z0, 0.0, g0), tl.where(z1, 0.0, g1), tl.where(z2, 0.0, g2)

    @triton.jit
    def _disp(te, dfield, x0, x1, x2, c0, c1, c2, h0, h1, h2, FX, FY, FZ, NMON: tl.constexpr, HF: tl.constexpr):
        e = _eddy(te, (x0 - c0) / h0, (x1 - c1) / h1, (x2 - c2) / h2, NMON)
        if HF:
            e += _field(dfield, x0, x1, x2, FX, FY, FZ)
        return e

    @triton.jit
    def _dpe(te, ddfield, x0, x1, x2, c0, c1, c2, h0, h1, h2, FX, FY, FZ, PE: tl.constexpr, NMON: tl.constexpr,
             HF: tl.constexpr):
        g0, g1, g2 = _eddy_grad(te, (x0 - c0) / h0, (x1 - c1) / h1, (x2 - c2) / h2, h0, h1, h2, NMON)
        d = _pick(g0, g1, g2, PE)
        if HF:
            d += _field(ddfield, x0, x1, x2, FX, FY, FZ)
        return d

    @triton.jit
    def _phi_fwd(pv, fp, dfield, x0, x1, x2, FX, FY, FZ, PE: tl.constexpr, NMON: tl.constexpr, HF: tl.constexpr):
        c0, c1, c2 = tl.load(fp), tl.load(fp + 1), tl.load(fp + 2)
        h0, h1, h2 = tl.load(fp + 3), tl.load(fp + 4), tl.load(fp + 5)
        v0, v1, v2 = tl.load(fp + 6), tl.load(fp + 7), tl.load(fp + 8)
        m0, m1, m2 = (x0 - c0) * v0, (x1 - c1) * v1, (x2 - c2) * v2
        p0 = (tl.load(pv) * m0 + tl.load(pv + 1) * m1 + tl.load(pv + 2) * m2 + tl.load(pv + 9)) / v0 + c0
        p1 = (tl.load(pv + 3) * m0 + tl.load(pv + 4) * m1 + tl.load(pv + 5) * m2 + tl.load(pv + 10)) / v1 + c1
        p2 = (tl.load(pv + 6) * m0 + tl.load(pv + 7) * m1 + tl.load(pv + 8) * m2 + tl.load(pv + 11)) / v2 + c2
        e = _disp(pv + 12, dfield, x0, x1, x2, c0, c1, c2, h0, h1, h2, FX, FY, FZ, NMON, HF)
        if PE == 0:
            p0 += e
        elif PE == 1:
            p1 += e
        else:
            p2 += e
        return p0, p1, p2

    @triton.jit
    def _phi_inv(pv, fp, dfield, ddfield, y0, y1, y2, FX, FY, FZ, PE: tl.constexpr, NMON: tl.constexpr,
                 NPASS: tl.constexpr, HF: tl.constexpr):
        c0, c1, c2 = tl.load(fp), tl.load(fp + 1), tl.load(fp + 2)
        h0, h1, h2 = tl.load(fp + 3), tl.load(fp + 4), tl.load(fp + 5)
        v0, v1, v2 = tl.load(fp + 6), tl.load(fp + 7), tl.load(fp + 8)
        r00, r01, r02 = tl.load(pv), tl.load(pv + 1), tl.load(pv + 2)
        r10, r11, r12 = tl.load(pv + 3), tl.load(pv + 4), tl.load(pv + 5)
        r20, r21, r22 = tl.load(pv + 6), tl.load(pv + 7), tl.load(pv + 8)
        e = _disp(pv + 12, dfield, y0, y1, y2, c0, c1, c2, h0, h1, h2, FX, FY, FZ, NMON, HF)
        q0 = (y0 - c0) * v0 - tl.load(pv + 9)
        q1 = (y1 - c1) * v1 - tl.load(pv + 10)
        q2 = (y2 - c2) * v2 - tl.load(pv + 11)
        if PE == 0:
            q0 -= e * v0
        elif PE == 1:
            q1 -= e * v1
        else:
            q2 -= e * v2
        x0 = (r00 * q0 + r10 * q1 + r20 * q2) / v0 + c0                          # R^T q
        x1 = (r01 * q0 + r11 * q1 + r21 * q2) / v1 + c1
        x2 = (r02 * q0 + r12 * q1 + r22 * q2) / v2 + c2
        vp = _pick(v0, v1, v2, PE)                                                 # A^-1 e_pe
        a0 = _pick(r00, r10, r20, PE) * vp / v0
        a1 = _pick(r01, r11, r21, PE) * vp / v1
        a2 = _pick(r02, r12, r22, PE) * vp / v2
        ape = _pick(a0, a1, a2, PE)
        for _ in tl.static_range(NPASS):
            p0, p1, p2 = _phi_fwd(pv, fp, dfield, x0, x1, x2, FX, FY, FZ, PE, NMON, HF)
            s0, s1, s2 = (p0 - y0) * v0, (p1 - y1) * v1, (p2 - y2) * v2
            u0 = (r00 * s0 + r10 * s1 + r20 * s2) / v0                             # A^-1 r
            u1 = (r01 * s0 + r11 * s1 + r21 * s2) / v1
            u2 = (r02 * s0 + r12 * s1 + r22 * s2) / v2
            s = _dpe(pv + 12, ddfield, x0, x1, x2, c0, c1, c2, h0, h1, h2, FX, FY, FZ, PE, NMON, HF)
            den = tl.maximum(1.0 + s * ape, 0.1)
            k = s * _pick(u0, u1, u2, PE) / den
            x0 = x0 - (u0 - a0 * k)
            x1 = x1 - (u1 - a1 * k)
            x2 = x2 - (u2 - a2 * k)
        return x0, x1, x2

    @triton.jit
    def _bw(t, K: tl.constexpr):
        """The cubic B-spline's weight K (0-3) at fraction t, and its derivative."""
        s = 1.0 - t
        if K == 0:
            w = s * s * s / 6.0
            dw = -s * s / 2.0
        elif K == 1:
            w = (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0
            dw = (3.0 * t * t - 4.0 * t) / 2.0
        elif K == 2:
            w = (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0
            dw = (-3.0 * t * t + 2.0 * t + 1.0) / 2.0
        else:
            w = t * t * t / 6.0
            dw = t * t / 2.0
        return w, dw

    @triton.jit
    def _cubic(c, p0, p1, p2, X, Y, Z, GRAD: tl.constexpr):
        """The cubic B-spline of one volume's coefficients c (X, Y, Z) at p (voxels), indices clamped; with GRAD its
        gradient too."""
        f0, f1, f2 = tl.floor(p0), tl.floor(p1), tl.floor(p2)
        t0, t1, t2 = p0 - f0, p1 - f1, p2 - f2
        i0, i1, i2 = f0.to(tl.int32), f1.to(tl.int32), f2.to(tl.int32)
        v = tl.zeros_like(p0)
        g0 = tl.zeros_like(p0)
        g1 = tl.zeros_like(p0)
        g2 = tl.zeros_like(p0)
        for a in tl.static_range(4):
            wa, da = _bw(t0, a)
            ia = tl.minimum(tl.maximum(i0 + (a - 1), 0), X - 1)
            for b in tl.static_range(4):
                wb, db = _bw(t1, b)
                ib = tl.minimum(tl.maximum(i1 + (b - 1), 0), Y - 1)
                row = c + (ia * Y + ib) * Z
                wab = wa * wb
                for k in tl.static_range(4):
                    wk, dk = _bw(t2, k)
                    cv = tl.load(row + tl.minimum(tl.maximum(i2 + (k - 1), 0), Z - 1))
                    v += wab * wk * cv
                    if GRAD:
                        g0 += da * wb * wk * cv
                        g1 += wa * db * wk * cv
                        g2 += wab * dk * cv
        return v, g0, g1, g2

    @triton.jit
    def _rot_col(d, m0, m1, m2, v0, v1, v2, gt0, gt1, gt2):
        """(dR m / vox) . g~ for one dR (row-major at d)."""
        a0 = (tl.load(d) * m0 + tl.load(d + 1) * m1 + tl.load(d + 2) * m2) / v0
        a1 = (tl.load(d + 3) * m0 + tl.load(d + 4) * m1 + tl.load(d + 5) * m2) / v1
        a2 = (tl.load(d + 6) * m0 + tl.load(d + 7) * m1 + tl.load(d + 8) * m2) / v2
        return a0 * gt0 + a1 * gt1 + a2 * gt2

    @triton.jit
    def _residual(coef, obs, pts, pvol, fp, dfield, ddfield, g, li, n, m, N, X, Y, Z, FX, FY, FZ,
                  PE: tl.constexpr, NMON: tl.constexpr, NPASS: tl.constexpr, HF: tl.constexpr, GRAD: tl.constexpr):
        """At the block's points n of volume g (parameters at row li of pvol): x = phi^-1(y), the prediction's value
        (and gradient) there, and the residual against obs."""
        y0 = tl.load(pts + 3 * n, mask=m, other=0.0)
        y1 = tl.load(pts + 3 * n + 1, mask=m, other=0.0)
        y2 = tl.load(pts + 3 * n + 2, mask=m, other=0.0)
        pv = pvol + li * _NPV
        x0, x1, x2 = _phi_inv(pv, fp, dfield, ddfield, y0, y1, y2, FX, FY, FZ, PE, NMON, NPASS, HF)
        cg = coef + g.to(tl.int64) * X * Y * Z
        v, gr0, gr1, gr2 = _cubic(cg, x0 - tl.load(fp + 25), x1 - tl.load(fp + 26), x2 - tl.load(fp + 27), X, Y, Z,
                                  GRAD)
        r = v - tl.load(obs + g.to(tl.int64) * N + n, mask=m, other=0.0)
        return tl.where(m, r, 0.0), v, x0, x1, x2, gr0, gr1, gr2

    # ------------------------------------------------------------------------------------------------- kernels

    @triton.jit(do_not_specialize=_UNSPEC["_step_moments"])
    def _step_moments(coef, obs, pts, pvol, part, fp, dfield, ddfield, N, NCH, X, Y, Z, FX, FY, FZ,
                      PE: tl.constexpr, NMON: tl.constexpr, NPASS: tl.constexpr, HF: tl.constexpr,
                      BLOCK: tl.constexpr):
        ch = tl.program_id(0)
        g = tl.program_id(1)
        n = ch * BLOCK + tl.arange(0, BLOCK)
        m = n < N
        r, v, x0, x1, x2, gr0, gr1, gr2 = _residual(coef, obs, pts, pvol, fp, dfield, ddfield, g, g, n, m, N, X, Y, Z,
                                                    FX, FY, FZ, PE, NMON, NPASS, HF, True)
        c0, c1, c2 = tl.load(fp), tl.load(fp + 1), tl.load(fp + 2)
        h0, h1, h2 = tl.load(fp + 3), tl.load(fp + 4), tl.load(fp + 5)
        v0, v1, v2 = tl.load(fp + 6), tl.load(fp + 7), tl.load(fp + 8)
        pv = pvol + g * _NPV
        r00, r01, r02 = tl.load(pv), tl.load(pv + 1), tl.load(pv + 2)
        r10, r11, r12 = tl.load(pv + 3), tl.load(pv + 4), tl.load(pv + 5)
        r20, r21, r22 = tl.load(pv + 6), tl.load(pv + 7), tl.load(pv + 8)
        # g~ = (dphi/dx)^-T grad, dphi/dx = A + e_pe w' by Sherman-Morrison; A^-T q = vox * R (q / vox)
        q0, q1, q2 = gr0 / v0, gr1 / v1, gr2 / v2
        g00 = (r00 * q0 + r01 * q1 + r02 * q2) * v0
        g01 = (r10 * q0 + r11 * q1 + r12 * q2) * v1
        g02 = (r20 * q0 + r21 * q1 + r22 * q2) * v2
        u0, u1, u2 = (x0 - c0) / h0, (x1 - c1) / h1, (x2 - c2) / h2
        w0, w1, w2 = _eddy_grad(pv + 12, u0, u1, u2, h0, h1, h2, NMON)
        if HF:
            f0, f1, f2 = _field_grad(dfield, x0, x1, x2, FX, FY, FZ)
            w0 += f0
            w1 += f1
            w2 += f2
        w0, w1, w2 = w0 / v0, w1 / v1, w2 / v2
        aw0 = (r00 * w0 + r01 * w1 + r02 * w2) * v0
        aw1 = (r10 * w0 + r11 * w1 + r12 * w2) * v1
        aw2 = (r20 * w0 + r21 * w1 + r22 * w2) * v2
        den = tl.maximum(1.0 + _pick(aw0, aw1, aw2, PE), 0.1)
        k = _pick(g00, g01, g02, PE) / den
        gt0, gt1, gt2 = g00 - aw0 * k, g01 - aw1 * k, g02 - aw2 * k
        m0, m1, m2 = (x0 - c0) * v0, (x1 - c1) * v1, (x2 - c2) * v2
        sc = fp + 9
        cols = tl.arange(0, 16)[None, :]
        J = tl.where(cols == 0, (-gt0 / v0 * tl.load(sc))[:, None], 0.0)
        J += tl.where(cols == 1, (-gt1 / v1 * tl.load(sc + 1))[:, None], 0.0)
        J += tl.where(cols == 2, (-gt2 / v2 * tl.load(sc + 2))[:, None], 0.0)
        J += tl.where(cols == 3, (-_rot_col(pv + 22, m0, m1, m2, v0, v1, v2, gt0, gt1, gt2) * tl.load(sc + 3))[:, None],
                      0.0)
        J += tl.where(cols == 4, (-_rot_col(pv + 31, m0, m1, m2, v0, v1, v2, gt0, gt1, gt2) * tl.load(sc + 4))[:, None],
                      0.0)
        J += tl.where(cols == 5, (-_rot_col(pv + 40, m0, m1, m2, v0, v1, v2, gt0, gt1, gt2) * tl.load(sc + 5))[:, None],
                      0.0)
        gpe = -_pick(gt0, gt1, gt2, PE)
        if NMON > 0:
            J += tl.where(cols == 6, (gpe * tl.load(sc + 6))[:, None], 0.0)
            J += tl.where(cols == 7, (gpe * u0 * tl.load(sc + 7))[:, None], 0.0)
            J += tl.where(cols == 8, (gpe * u1 * tl.load(sc + 8))[:, None], 0.0)
            J += tl.where(cols == 9, (gpe * u2 * tl.load(sc + 9))[:, None], 0.0)
        if NMON == 10:
            J += tl.where(cols == 10, (gpe * u0 * u0 * tl.load(sc + 10))[:, None], 0.0)
            J += tl.where(cols == 11, (gpe * u1 * u1 * tl.load(sc + 11))[:, None], 0.0)
            J += tl.where(cols == 12, (gpe * u2 * u2 * tl.load(sc + 12))[:, None], 0.0)
            J += tl.where(cols == 13, (gpe * u0 * u1 * tl.load(sc + 13))[:, None], 0.0)
            J += tl.where(cols == 14, (gpe * u0 * u2 * tl.load(sc + 14))[:, None], 0.0)
            J += tl.where(cols == 15, (gpe * u1 * u2 * tl.load(sc + 15))[:, None], 0.0)
        J = tl.where(m[:, None], J, 0.0)
        H = tl.dot(tl.trans(J), J, input_precision="ieee")                       # (16, 16)
        b = tl.sum(J * r[:, None], axis=0)
        f = tl.sum(r * r, axis=0)
        base = part + (g.to(tl.int64) * NCH + ch) * (16 * 16 + 16 + 1)
        i16 = tl.arange(0, 16)
        tl.store(base + i16[:, None] * 16 + i16[None, :], H)
        tl.store(base + 256 + i16, b)
        tl.store(base + 272, f)

    @triton.jit(do_not_specialize=_UNSPEC["_step_cost"])
    def _step_cost(coef, obs, pts, pvol, vols, part, fp, dfield, ddfield, N, NCH, X, Y, Z, FX, FY, FZ,
                   PE: tl.constexpr, NMON: tl.constexpr, NPASS: tl.constexpr, HF: tl.constexpr, BLOCK: tl.constexpr):
        ch = tl.program_id(0)
        li = tl.program_id(1)
        g = tl.load(vols + li)
        n = ch * BLOCK + tl.arange(0, BLOCK)
        m = n < N
        r, v, x0, x1, x2, gr0, gr1, gr2 = _residual(coef, obs, pts, pvol, fp, dfield, ddfield, g, li, n, m, N, X, Y, Z,
                                                    FX, FY, FZ, PE, NMON, NPASS, HF, False)
        tl.store(part + li.to(tl.int64) * NCH + ch, tl.sum(r * r, axis=0))

    @triton.jit(do_not_specialize=_UNSPEC["_step_values"])
    def _step_values(coef, pts, pvol, vols, out, fp, dfield, ddfield, N, X, Y, Z, FX, FY, FZ,
                     PE: tl.constexpr, NMON: tl.constexpr, NPASS: tl.constexpr, HF: tl.constexpr,
                     BLOCK: tl.constexpr):
        ch = tl.program_id(0)
        li = tl.program_id(1)
        g = tl.load(vols + li)
        n = ch * BLOCK + tl.arange(0, BLOCK)
        m = n < N
        r, v, x0, x1, x2, gr0, gr1, gr2 = _residual(coef, pts, pts, pvol, fp, dfield, ddfield, g, li, n, m, 0, X, Y, Z,
                                                    FX, FY, FZ, PE, NMON, NPASS, HF, False)
        tl.store(out + li.to(tl.int64) * N + n, v, mask=m)

    @triton.jit(do_not_specialize=_UNSPEC["_resample"])
    def _resample(vols, pts, pvol, out, fp, dfield, N, X, Y, Z, FX, FY, FZ,
                  PE: tl.constexpr, NMON: tl.constexpr, HF: tl.constexpr, BLOCK: tl.constexpr):
        ch = tl.program_id(0)
        g = tl.program_id(1)
        n = ch * BLOCK + tl.arange(0, BLOCK)
        m = n < N
        x0 = tl.load(pts + 3 * n, mask=m, other=0.0)
        x1 = tl.load(pts + 3 * n + 1, mask=m, other=0.0)
        x2 = tl.load(pts + 3 * n + 2, mask=m, other=0.0)
        p0, p1, p2 = _phi_fwd(pvol + g * _NPV, fp, dfield, x0, x1, x2, FX, FY, FZ, PE, NMON, HF)
        f0, f1, f2 = tl.floor(p0), tl.floor(p1), tl.floor(p2)
        t0, t1, t2 = p0 - f0, p1 - f1, p2 - f2
        i0, i1, i2 = f0.to(tl.int32), f1.to(tl.int32), f2.to(tl.int32)
        c = vols + g.to(tl.int64) * X * Y * Z
        v = tl.zeros_like(p0)
        for a in tl.static_range(2):
            ia = i0 + a
            wa = t0 if a == 1 else 1.0 - t0
            for b in tl.static_range(2):
                ib = i1 + b
                wb = t1 if b == 1 else 1.0 - t1
                for k in tl.static_range(2):
                    ik = i2 + k
                    wk = t2 if k == 1 else 1.0 - t2
                    ok = m & (ia >= 0) & (ia <= X - 1) & (ib >= 0) & (ib <= Y - 1) & (ik >= 0) & (ik <= Z - 1)
                    v += wa * wb * wk * tl.load(c + (ia * Y + ib) * Z + ik, mask=ok, other=0.0)
        tl.store(out + g.to(tl.int64) * N + n, v, mask=m)

    @triton.jit(do_not_specialize=_UNSPEC["_apply_volumes"])
    def _apply_volumes(coef, pvol, out, valid, fp, dfield, ddfield, X, Y, Z, FX, FY, FZ,
                       PE: tl.constexpr, NMON: tl.constexpr, HF: tl.constexpr, BLOCK: tl.constexpr):
        ch = tl.program_id(0)
        g = tl.program_id(1)
        nv = X * Y * Z
        v = ch * BLOCK + tl.arange(0, BLOCK)
        m = v < nv
        x0 = (v // (Y * Z)).to(tl.float32)
        x1 = ((v // Z) % Y).to(tl.float32)
        x2 = (v % Z).to(tl.float32)
        pv = pvol + g * _NPV
        p0, p1, p2 = _phi_fwd(pv, fp, dfield, x0, x1, x2, FX, FY, FZ, PE, NMON, HF)
        c0, c1, c2 = tl.load(fp), tl.load(fp + 1), tl.load(fp + 2)
        h0, h1, h2 = tl.load(fp + 3), tl.load(fp + 4), tl.load(fp + 5)
        jac = 1.0 + _dpe(pv + 12, ddfield, x0, x1, x2, c0, c1, c2, h0, h1, h2, FX, FY, FZ, PE, NMON, HF)
        val, d0, d1, d2 = _cubic(coef + g.to(tl.int64) * nv, p0, p1, p2, X, Y, Z, False)
        inside = ((p0 >= 0.0) & (p0 <= (X - 1).to(tl.float32)) & (p1 >= 0.0) & (p1 <= (Y - 1).to(tl.float32))
                  & (p2 >= 0.0) & (p2 <= (Z - 1).to(tl.float32)))
        o = g.to(tl.int64) * nv + v
        tl.store(valid + o, tl.where(inside, 1.0, 0.0), mask=m)
        tl.store(out + o, tl.where(inside, tl.maximum(val * jac, 0.0), 0.0), mask=m)


# ------------------------------------------------------------------------------------------------- wrappers

def _switches(c):
    return dict(PE=c.pe, NMON=c.nmon, HF=c.hf, BLOCK=BLOCK)


def unpack(part, P):
    """(G, NCH, NOUT) partials -> H (G, P, P), b (G, P), f (G,), summed over the blocks in a fixed order."""
    s = part.sum(1)
    return s[:, :256].reshape(-1, 16, 16)[:, :P, :P], s[:, 256:256 + P], s[:, 272]


def step_moments(coef, obs, pts, pvol, nch, c, fp, dfield, ddfield, P):
    G = coef.shape[0]
    part = torch.empty(G, nch, NOUT, dtype=torch.float32, device=coef.device)
    with torch.cuda.device(coef.device):                                 # Triton launches on the current device
        _step_moments[(nch, G)](coef, obs, pts, pvol, part, fp, dfield, ddfield, c.N, nch, c.X, c.Y, c.Z, c.FX, c.FY, c.FZ,
                                NPASS=c.npass, **_switches(c))
    return unpack(part, P)


def step_cost(coef, obs, pts, pvol_sub, vols, nch, c, fp, dfield, ddfield):
    L = vols.shape[0]
    part = torch.empty(L, nch, dtype=torch.float32, device=coef.device)
    with torch.cuda.device(coef.device):                                 # Triton launches on the current device
        _step_cost[(nch, L)](coef, obs, pts, pvol_sub, vols.int().contiguous(), part, fp, dfield, ddfield, c.N, nch,
                             c.X, c.Y, c.Z, c.FX, c.FY, c.FZ, NPASS=c.npass, **_switches(c))
    return part.sum(1)


def step_values(coef, pts, pvol_sub, vols, c, fp, dfield, ddfield):
    """The prediction moved into acquisition space at each point, for the listed volumes -> (len(vols), N)."""
    L, N = vols.shape[0], pts.shape[0]
    out = torch.empty(L, N, dtype=torch.float32, device=coef.device)
    with torch.cuda.device(coef.device):                                 # Triton launches on the current device
        _step_values[((N + BLOCK - 1) // BLOCK, L)](coef, pts, pvol_sub, vols.int().contiguous(), out, fp, dfield, ddfield,
                                                    N, c.X, c.Y, c.Z, c.FX, c.FY, c.FZ, NPASS=c.npass, **_switches(c))
    return out


def resample(vols, pts, pvol, c, fp, dfield):
    """vols (G, X, Y, Z) sampled trilinearly at phi_g(pts), zero outside -> (G, N)."""
    G, N = vols.shape[0], pts.shape[0]
    out = torch.empty(G, N, dtype=torch.float32, device=vols.device)
    with torch.cuda.device(vols.device):                                 # Triton launches on the current device
        _resample[((N + BLOCK - 1) // BLOCK, G)](vols, pts, pvol, out, fp, dfield, N, c.X, c.Y, c.Z, c.FX, c.FY, c.FZ,
                                                 **_switches(c))
    return out


def apply_volumes(vols, pvol, c, fp, dfield, ddfield):
    """The corrected volumes and their validity (each (G, X, Y, Z)), as motion_eddy_metal.apply_volumes."""
    coef = prefilter(vols).contiguous()
    G, X, Y, Z = vols.shape
    out = torch.empty_like(coef)
    valid = torch.empty_like(coef)
    with torch.cuda.device(coef.device):                                 # Triton launches on the current device
        _apply_volumes[((X * Y * Z + BLOCK - 1) // BLOCK, G)](coef, pvol, out, valid, fp, dfield, ddfield, X, Y, Z,
                                                              c.FX, c.FY, c.FZ, **_switches(c))
    return out, valid


# ------------------------------------------------------------------------------------------------- warming the cache

MODELS = {"rigid": 0, "linear": 4, "quadratic": 10}


def combos(pe_axes=(1,), models=("quadratic",), fields=(True, False)) -> list:
    """The compile-time combinations (PE axis, eddy terms, with a field, Newton passes) for these choices; the passes
    are the estimator's default for each (3 with a field, 2 without)."""
    return [(int(pe), MODELS[m], int(bool(hf)), 3 if hf else 2) for pe in pe_axes for m in models for hf in fields]


def _warm_combo(combo) -> float:
    import time
    pe, nmon, hf, npass = combo
    dev = torch.device("cuda")
    f = lambda: torch.zeros(1, dtype=torch.float32, device=dev)                       # noqa: E731
    i = torch.zeros(1, dtype=torch.int32, device=dev)
    sw = dict(PE=pe, NMON=nmon, HF=hf, BLOCK=BLOCK)
    t0 = time.time()
    _step_moments.warmup(f(), f(), f(), f(), f(), f(), f(), f(), 1, 1, 1, 1, 1, 1, 1, 1, NPASS=npass, grid=(1, 1), **sw)
    _step_cost.warmup(f(), f(), f(), f(), i, f(), f(), f(), f(), 1, 1, 1, 1, 1, 1, 1, 1, NPASS=npass, grid=(1, 1), **sw)
    _step_values.warmup(f(), f(), f(), i, f(), f(), f(), f(), 1, 1, 1, 1, 1, 1, 1, NPASS=npass, grid=(1, 1), **sw)
    _resample.warmup(f(), f(), f(), f(), f(), f(), 1, 1, 1, 1, 1, 1, 1, grid=(1, 1), **sw)
    _apply_volumes.warmup(f(), f(), f(), f(), f(), f(), f(), 1, 1, 1, 1, 1, 1, grid=(1, 1), **sw)
    return round(time.time() - t0, 1)


def warm(which=None, workers=None) -> dict:
    """Compile the five kernels for each combination in `which` (combos(); default: AP/PA phase encoding, the quadratic
    model, with and without a field) without running them, into Triton's cache (TRITON_CACHE_DIR, by default
    ~/.triton/cache): a run with those settings then compiles nothing. Compiling is single-threaded CPU work, so the
    combinations compile in parallel, `workers` spawned processes (default one per combination, up to the CPUs; 1: in
    this process). A cached combination is loaded, not compiled. Returns the seconds per combination. A script calling
    it with workers needs an `if __name__ == "__main__":` guard."""
    if not available():
        raise RuntimeError("motion_eddy_triton.warm: needs CUDA and Triton")
    which = [tuple(c) for c in (which or combos())]
    workers = min(len(which), cpus()) if workers is None else int(workers)
    if workers > 1 and len(which) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from .ukf import require_main_guard
        require_main_guard()
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            return dict(zip(which, ex.map(_warm_combo, which)))
    return {c: _warm_combo(c) for c in which}
