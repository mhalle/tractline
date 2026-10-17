"""UKF tractography's step as a compact Triton kernel: ukf_triton's arithmetic with the algebra on
blocks, so it compiles in a reasonable time (the unrolled kernel took 32 minutes).

A program takes F half-fibers. The 10-state is an [F, 16] row (padded), the 10x10 matrices are
[F, 16, 16] blocks (padded with the identity, so inverses and Cholesky stay defined), per-gradient
quantities are [F, NPAD] tiles and the cross-covariance an [F, 16, NPAD] block. Products of blocks
are batched tl.dot in IEEE float32; Cholesky and the triangular inverse are 10-step loops over block
rows and columns; the sigma points are a runtime loop. The rules and this kernel's own choices are
ukf_triton's (N measurements with the sums doubled; the cross-covariance rearranged so H is computed
once per sigma point; Ht never formed; both inverses through Cholesky; IEEE divide and sqrt, libdevice
exp and acos; no fused multiply-add).
"""
from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _pick(V, i16, k):
    """Component k of [F, 16] rows -> [F]."""
    return tl.sum(tl.where(i16 == k, V, 0.0), axis=1)


@triton.jit
def _chol(A, ii, jj, i16):
    """Lower Cholesky factor of [F, 16, 16] blocks (the first 10 rows and columns; identity beyond)."""
    L = tl.zeros_like(A)
    for j in tl.static_range(10):
        rowj = tl.sum(tl.where(ii == j, L, 0.0), axis=1)                 # L[j, :]
        v = tl.sum(L * rowj[:, None, :], axis=2)                         # sum_k L[i, k] L[j, k]
        acol = tl.sum(tl.where(jj == j, A, 0.0), axis=2)                 # A[:, j]
        ljj = tl.sqrt_rn(_pick(acol - v, i16, j))
        col = tl.where(i16 > j, tl.div_rn(acol - v, ljj[:, None]), tl.where(i16 == j, ljj[:, None], 0.0))
        L = L + tl.where(jj == j, col[:, :, None], 0.0)
    return L + tl.where((ii == jj) & (ii >= 10), 1.0, 0.0)


@triton.jit
def _spd_inv(A, ii, jj, i16):
    """A^-1 = L^-T L^-1 for symmetric positive definite blocks."""
    L = _chol(A, ii, jj, i16)
    M = tl.zeros_like(A)
    for i in tl.static_range(10):
        Lrow = tl.sum(tl.where(ii == i, L, 0.0), axis=1)                 # L[i, :]
        acc = tl.sum(Lrow[:, :, None] * M, axis=1)                       # sum_k L[i, k] M[k, :]
        row = tl.div_rn(tl.where(i16 == i, 1.0, 0.0) - acc, _pick(Lrow, i16, i)[:, None])
        M = M + tl.where(ii == i, row[:, None, :], 0.0)
    M = M + tl.where((ii == jj) & (ii >= 10), 1.0, 0.0)
    return tl.dot(tl.permute(M, (0, 2, 1)), M, input_precision="ieee")


@triton.jit
def _expq(a0, a1, a2, a3, a4, gx, gy, gz, bb):
    """exp(-b u'Du) at every gradient for one tensor of a state ([F] components), filter_Simple2T::H."""
    n = tl.sqrt_rn(a0 * a0 + a1 * a1 + a2 * a2)
    m0 = tl.div_rn(a0, n); m1 = tl.div_rn(a1, n); m2 = tl.div_rn(a2, n)
    neg = m0 < 0.0
    m0 = tl.where(neg, -m0, m0); m1 = tl.where(neg, -m1, m1); m2 = tl.where(neg, -m2, m2)
    l1 = tl.maximum(a3, 100.0); l2 = tl.maximum(a4, 100.0)
    r11 = tl.div_rn(m1 * m1, 1.0 + m0) - 1.0
    r12 = tl.div_rn(m1 * m2, 1.0 + m0)
    r22 = tl.div_rn(m2 * m2, 1.0 + m0) - 1.0
    d00 = (((0.0 + m0 * l1 * m0) + m1 * l2 * m1) + m2 * l2 * m2) * 1e-6
    d01 = (((0.0 + m0 * l1 * m1) + m1 * l2 * r11) + m2 * l2 * r12) * 1e-6
    d02 = (((0.0 + m0 * l1 * m2) + m1 * l2 * r12) + m2 * l2 * r22) * 1e-6
    d11 = (((0.0 + m1 * l1 * m1) + r11 * l2 * r11) + r12 * l2 * r12) * 1e-6
    d12 = (((0.0 + m1 * l1 * m2) + r11 * l2 * r12) + r12 * l2 * r22) * 1e-6
    d22 = (((0.0 + m2 * l1 * m2) + r12 * l2 * r12) + r22 * l2 * r22) * 1e-6
    q = 0.0 + gx * (d00[:, None] * gx + d01[:, None] * gy + d02[:, None] * gz)
    q = q + gy * (d01[:, None] * gx + d11[:, None] * gy + d12[:, None] * gz)
    q = q + gz * (d02[:, None] * gx + d12[:, None] * gy + d22[:, None] * gz)
    return libdevice.exp(-bb * q)


@triton.jit
def _H(V, i16, gx, gy, gz, bb, nm2):
    """H(state) at every gradient [F, NPAD] for [F, 16] states, masked to the real gradients."""
    h = tl.zeros(nm2.shape, dtype=tl.float32)
    h = h + _expq(_pick(V, i16, 0), _pick(V, i16, 1), _pick(V, i16, 2), _pick(V, i16, 3), _pick(V, i16, 4),
                  gx, gy, gz, bb) * 0.5
    h = h + _expq(_pick(V, i16, 5), _pick(V, i16, 6), _pick(V, i16, 7), _pick(V, i16, 8), _pick(V, i16, 9),
                  gx, gy, gz, bb) * 0.5
    return tl.where(nm2, h, 0.0)


@triton.jit
def _l2fa(a, b):
    return tl.div_rn(tl.abs(a - b), tl.sqrt_rn(a * a + 2.0 * b * b))


@triton.jit
def _cround(x):
    return tl.where(x < 0.0, -tl.floor(-x + 0.5), tl.floor(x + 0.5)).to(tl.int32)


#: the integer arguments are not specialized: by default Triton compiles a copy for each that equals 1 or is
#: divisible by 16, and B (active fibers) and step pass through all three, the grid adds its own - 6-8 copies
#: per scan, 33-50 s each (ds005123: 4.4-6.7 min on its first run; NOTES 2026-10-06). One copy per NPAD now.
#: cos_tie, TIE and TIE_SW2 are ukf.advance's opt-in tie_angle and tie_sw2 (TIE=False: the code is not compiled in,
#: cos_tie unused). They
#: change the default kernel's signature, not its arithmetic, but bit-for-bit equality with the earlier kernel on CUDA
#: is unverified (no CUDA where they were added), as is the tie-break itself on this kernel.
@triton.jit(do_not_specialize=["B", "N", "nk", "nj", "ni", "step", "max_steps"])
def ukf_step_block(A, MASK, G, BV, X, ST, P, OLD, DIR, FLAGS, FAO, MSIG, B, N, nk, nj, ni, step, max_steps,
                   vox0, vox1, vox2, sigma, step_length, stop_fa, stop_thr, rr, Qm, Ql, c, W0, Wi, cos_tie,
                   F: tl.constexpr, NPAD: tl.constexpr, TIE: tl.constexpr, TIE_SW2: tl.constexpr):
    f = tl.program_id(0) * F + tl.arange(0, F)
    fm = f < B
    n = tl.arange(0, NPAD)
    nm2 = fm[:, None] & (n < N)[None, :]
    i16 = tl.arange(0, 16)[None, :]                                      # [1, 16]: a row's components
    ii = tl.arange(0, 16)[None, :, None]                                 # [1, 16, 1]: a block's rows
    jj = tl.arange(0, 16)[None, None, :]                                 # [1, 1, 16]: a block's columns
    eye_pad = tl.where((ii == jj) & (ii >= 10), 1.0, 0.0)

    x0 = tl.load(X + f * 3 + 0, mask=fm, other=0.0)
    x1 = tl.load(X + f * 3 + 1, mask=fm, other=0.0)
    x2 = tl.load(X + f * 3 + 2, mask=fm, other=0.0)
    o0 = tl.load(OLD + f * 3 + 0, mask=fm, other=1.0)
    o1 = tl.load(OLD + f * 3 + 1, mask=fm, other=0.0)
    o2 = tl.load(OLD + f * 3 + 2, mask=fm, other=0.0)
    sm = fm[:, None] & (i16 < 10)
    s = tl.load(ST + f[:, None] * 10 + i16, mask=sm, other=0.0)
    s = tl.where(fm[:, None], s, tl.where((i16 % 5) < 3, 0.5, 1000.0))  # padded fibers: a harmless state
    pm_ = fm[:, None, None] & (ii < 10) & (jj < 10)
    Pc = tl.load(P + f[:, None, None] * 100 + ii * 10 + jj, mask=pm_, other=0.0)
    Pc = Pc + eye_pad + tl.where((~fm[:, None, None]) & (ii == jj) & (ii < 10), 1.0, 0.0)
    gx = tl.load(G + n * 3 + 0, mask=n < N, other=0.0)[None, :]
    gy = tl.load(G + n * 3 + 1, mask=n < N, other=0.0)[None, :]
    gz = tl.load(G + n * 3 + 2, mask=n < N, other=0.0)[None, :]
    bb = tl.load(BV + n, mask=n < N, other=0.0)[None, :]

    # ---- interp (Interp3Signal), the binary's x, y, z order
    r0 = _cround(x0); r1 = _cround(x1); r2 = _cround(x2)
    z = tl.zeros([F, NPAD], dtype=tl.float32)
    wsum = tl.zeros([F], dtype=tl.float32) + 1e-16
    for xx in tl.static_range(3):
        for yy in tl.static_range(3):
            for zz in tl.static_range(3):
                Xi = r0 + (xx - 1); Yi = r1 + (yy - 1); Zi = r2 + (zz - 1)
                ok = (Xi >= 0) & (Xi < nk) & (Yi >= 0) & (Yi < nj) & (Zi >= 0) & (Zi < ni)
                dx = (Xi.to(tl.float32) - x0) * vox0
                dy = (Yi.to(tl.float32) - x1) * vox1
                dz = (Zi.to(tl.float32) - x2) * vox2
                w = libdevice.exp(tl.div_rn(-(dx * dx + dy * dy + dz * dz), sigma))
                w = tl.where(ok, w, 0.0)
                v = ((tl.minimum(tl.maximum(Xi, 0), nk - 1) * nj + tl.minimum(tl.maximum(Yi, 0), nj - 1)) * ni
                     + tl.minimum(tl.maximum(Zi, 0), ni - 1)).to(tl.int64) * N
                a = tl.load(A + v[:, None] + n[None, :], mask=nm2, other=0.0)
                z = z + w[:, None] * a
                wsum = wsum + w
    z = tl.div_rn(z, wsum[:, None])

    # ---- sigma points: x, x + c L[:, j], x - c L[:, j], then F
    L = _chol(Pc, ii, jj, i16)
    lam = ((i16 % 5) >= 3) & (i16 < 10)
    xh = tl.zeros([F, 16], dtype=tl.float32)
    for si in range(21):
        col = si - 1 - 10 * (si > 10).to(tl.int32)
        sg = tl.where(si == 0, 0.0, tl.where(si <= 10, 1.0, -1.0))
        W = tl.where(si == 0, W0, Wi)
        q = s + sg * (c * tl.sum(tl.where(jj == col, L, 0.0), axis=2))
        r1v = tl.div_rn(1.0, tl.sqrt_rn(tl.sum(tl.where(i16 < 3, q * q, 0.0), axis=1)))
        r2v = tl.div_rn(1.0, tl.sqrt_rn(tl.sum(tl.where((i16 >= 5) & (i16 < 8), q * q, 0.0), axis=1)))
        q = tl.where(i16 < 3, q * r1v[:, None], tl.where((i16 >= 5) & (i16 < 8), q * r2v[:, None],
                                                         tl.where(lam, tl.maximum(q, 100.0), q)))
        xh = xh + W * q
    Pm = tl.zeros([F, 16, 16], dtype=tl.float32)
    sxt = tl.zeros([F, 16], dtype=tl.float32)
    zh = tl.zeros([F, NPAD], dtype=tl.float32)
    pxz = tl.zeros([F, 16, NPAD], dtype=tl.float32)
    for si in range(21):
        col = si - 1 - 10 * (si > 10).to(tl.int32)
        sg = tl.where(si == 0, 0.0, tl.where(si <= 10, 1.0, -1.0))
        W = tl.where(si == 0, W0, Wi)
        q = s + sg * (c * tl.sum(tl.where(jj == col, L, 0.0), axis=2))
        r1v = tl.div_rn(1.0, tl.sqrt_rn(tl.sum(tl.where(i16 < 3, q * q, 0.0), axis=1)))
        r2v = tl.div_rn(1.0, tl.sqrt_rn(tl.sum(tl.where((i16 >= 5) & (i16 < 8), q * q, 0.0), axis=1)))
        q = tl.where(i16 < 3, q * r1v[:, None], tl.where((i16 >= 5) & (i16 < 8), q * r2v[:, None],
                                                         tl.where(lam, tl.maximum(q, 100.0), q)))
        t = tl.where(i16 < 10, q - xh, 0.0)
        tw = t * W
        Pm = Pm + tw[:, :, None] * t[:, None, :]
        sxt = sxt + W * t
        h = _H(q, i16, gx, gy, gz, bb, nm2)
        zh = zh + W * h
        pxz = pxz + tw[:, :, None] * h[:, None, :]
    pxz = pxz - sxt[:, :, None] * zh[:, None, :]
    qd = tl.where(i16 < 10, tl.where((i16 % 5) < 3, Qm, Ql), 0.0)
    Pm = Pm + tl.where(ii == jj, qd[:, :, None], 0.0) + eye_pad

    # ---- the information-form update
    Yk = _spd_inv(Pm, ii, jj, i16)
    yh = tl.sum(Yk * xh[:, None, :], axis=2)
    term = (z - zh) + tl.sum(pxz * yh[:, :, None], axis=1)
    S = tl.dot(pxz, tl.permute(pxz, (0, 2, 1)), input_precision="ieee")
    v = tl.sum(pxz * term[:, None, :], axis=2)
    T = tl.dot(Yk, S, input_precision="ieee")
    M = Yk + 2.0 * (rr * tl.dot(T, Yk, input_precision="ieee"))
    iv = 2.0 * (rr * tl.sum(Yk * v[:, None, :], axis=2))
    Pn = _spd_inv(M, ii, jj, i16)
    sn = tl.sum(Pn * (iv + yh)[:, None, :], axis=2)

    # ---- State2Tensor2T, the two swaps, FA, the Euler step (tractography.cc)
    s0 = _pick(sn, i16, 0); s1 = _pick(sn, i16, 1); s2 = _pick(sn, i16, 2); s3 = _pick(sn, i16, 3); s4 = _pick(sn, i16, 4)
    s5 = _pick(sn, i16, 5); s6 = _pick(sn, i16, 6); s7 = _pick(sn, i16, 7); s8 = _pick(sn, i16, 8); s9 = _pick(sn, i16, 9)
    n1 = tl.sqrt_rn(s0 * s0 + s1 * s1 + s2 * s2); n2 = tl.sqrt_rn(s5 * s5 + s6 * s6 + s7 * s7)
    m1x = tl.div_rn(s0, n1); m1y = tl.div_rn(s1, n1); m1z = tl.div_rn(s2, n1)
    m2x = tl.div_rn(s5, n2); m2y = tl.div_rn(s6, n2); m2z = tl.div_rn(s7, n2)
    L1a = tl.maximum(s3, 100.0); L1b = tl.maximum(s4, 100.0); L2a = tl.maximum(s8, 100.0); L2b = tl.maximum(s9, 100.0)
    fl = (m1x * o0 + m1y * o1 + m1z * o2) < 0.0
    m1x = tl.where(fl, -m1x, m1x); m1y = tl.where(fl, -m1y, m1y); m1z = tl.where(fl, -m1z, m1z)
    fl = (m2x * o0 + m2y * o1 + m2z * o2) < 0.0
    m2x = tl.where(fl, -m2x, m2x); m2y = tl.where(fl, -m2y, m2y); m2z = tl.where(fl, -m2z, m2z)
    fat1 = _l2fa(L1a, L1b); fat2 = _l2fa(L2a, L2b)
    angle = libdevice.acos(m1x * m2x + m1y * m2y + m1z * m2z) * 57.29577951308232
    sw = (m1x * o0 + m1y * o1 + m1z * o2) < (m2x * o0 + m2y * o1 + m2z * o2)
    if TIE:                                                              # ukf.advance's opt-in tie_angle: at a near tie sw keeps the order
        tie = tl.abs(m1x * m2x + m1y * m2y + m1z * m2z) > cos_tie
        sw = sw & ~tie
    # the permutation that swaps the two tensors: Pi[i, k] = (k == (i + 5) % 10), identity past 10
    pidx = tl.where(ii < 10, (ii + 5) % 10, ii)
    Pi = tl.where(jj == pidx, 1.0, 0.0) + tl.zeros([F, 16, 16], dtype=tl.float32)
    PiT = tl.permute(Pi, (0, 2, 1))
    Psw = tl.dot(tl.dot(Pi, Pn, input_precision="ieee"), PiT, input_precision="ieee")   # exact: one term per entry
    Pn = tl.where(sw[:, None, None], Psw, Pn)
    t = m1x; m1x = tl.where(sw, m2x, m1x); m2x = tl.where(sw, t, m2x)
    t = m1y; m1y = tl.where(sw, m2y, m1y); m2y = tl.where(sw, t, m2y)
    t = m1z; m1z = tl.where(sw, m2z, m1z); m2z = tl.where(sw, t, m2z)
    t = L1a; L1a = tl.where(sw, L2a, L1a); L2a = tl.where(sw, t, L2a)
    t = L1b; L1b = tl.where(sw, L2b, L1b); L2b = tl.where(sw, t, L2b)
    t = fat1; fat1 = tl.where(sw, fat2, fat1); fat2 = tl.where(sw, t, fat2)
    t = s0; s0 = tl.where(sw, s5, s0); s5 = tl.where(sw, t, s5)
    t = s1; s1 = tl.where(sw, s6, s1); s6 = tl.where(sw, t, s6)
    t = s2; s2 = tl.where(sw, s7, s2); s7 = tl.where(sw, t, s7)
    t = s3; s3 = tl.where(sw, s8, s3); s8 = tl.where(sw, t, s8)
    t = s4; s4 = tl.where(sw, s9, s4); s9 = tl.where(sw, t, s9)
    sw2 = (angle <= 20.0) & (tl.minimum(fat1, fat2) <= 0.2) & ~(fat1 > 0.2)
    if TIE and TIE_SW2:                                                  # tie_sw2: and so does sw2
        sw2 = sw2 & ~tie
    Psw = tl.dot(tl.dot(Pi, Pn, input_precision="ieee"), PiT, input_precision="ieee")
    Pn = tl.where(sw2[:, None, None], Psw, Pn)
    t = m1x; m1x = tl.where(sw2, m2x, m1x); m2x = tl.where(sw2, t, m2x)
    t = m1y; m1y = tl.where(sw2, m2y, m1y); m2y = tl.where(sw2, t, m2y)
    t = m1z; m1z = tl.where(sw2, m2z, m1z); m2z = tl.where(sw2, t, m2z)
    t = L1a; L1a = tl.where(sw2, L2a, L1a); L2a = tl.where(sw2, t, L2a)
    t = L1b; L1b = tl.where(sw2, L2b, L1b); L2b = tl.where(sw2, t, L2b)
    t = s0; s0 = tl.where(sw2, s5, s0); s5 = tl.where(sw2, t, s5)
    t = s1; s1 = tl.where(sw2, s6, s1); s6 = tl.where(sw2, t, s6)
    t = s2; s2 = tl.where(sw2, s7, s2); s7 = tl.where(sw2, t, s7)
    t = s3; s3 = tl.where(sw2, s8, s3); s8 = tl.where(sw2, t, s8)
    t = s4; s4 = tl.where(sw2, s9, s4); s9 = tl.where(sw2, t, s9)
    fa = tl.where(L1a < L1b, 0.0, _l2fa(L1a, L1b))
    x0 = x0 + tl.div_rn(m1z, vox0) * step_length
    x1 = x1 + tl.div_rn(m1y, vox1) * step_length
    x2 = x2 + tl.div_rn(m1x, vox2) * step_length

    # ---- the checks at the new position
    hs = tl.zeros([F, NPAD], dtype=tl.float32)
    hs = hs + _expq(s0, s1, s2, s3, s4, gx, gy, gz, bb) * 0.5
    hs = hs + _expq(s5, s6, s7, s8, s9, gx, gy, gz, bb) * 0.5
    hs = tl.where(nm2, hs, 0.0)
    mean_sig = tl.div_rn(tl.sum(hs, axis=1), N.to(tl.float32))
    q0 = _cround(x0); q1 = _cround(x1); q2 = _cround(x2)
    inb = (q0 >= 0) & (q0 < nk) & (q1 >= 0) & (q1 < nj) & (q2 >= 0) & (q2 < ni)
    mi = ((tl.minimum(tl.maximum(q0, 0), nk - 1) * nj + tl.minimum(tl.maximum(q1, 0), nj - 1)) * ni
          + tl.minimum(tl.maximum(q2, 0), ni - 1)).to(tl.int64)
    inside = inb & (tl.load(MASK + mi, mask=fm, other=0) > 0)
    stop = (~inside) | (mean_sig < stop_thr) | (fa < stop_fa) | (step > max_steps)

    tl.store(X + f * 3 + 0, x0, mask=fm); tl.store(X + f * 3 + 1, x1, mask=fm); tl.store(X + f * 3 + 2, x2, mask=fm)
    tl.store(DIR + f * 3 + 0, m1x, mask=fm); tl.store(DIR + f * 3 + 1, m1y, mask=fm); tl.store(DIR + f * 3 + 2, m1z, mask=fm)
    tl.store(ST + f * 10 + 0, s0, mask=fm); tl.store(ST + f * 10 + 1, s1, mask=fm); tl.store(ST + f * 10 + 2, s2, mask=fm)
    tl.store(ST + f * 10 + 3, s3, mask=fm); tl.store(ST + f * 10 + 4, s4, mask=fm); tl.store(ST + f * 10 + 5, s5, mask=fm)
    tl.store(ST + f * 10 + 6, s6, mask=fm); tl.store(ST + f * 10 + 7, s7, mask=fm); tl.store(ST + f * 10 + 8, s8, mask=fm)
    tl.store(ST + f * 10 + 9, s9, mask=fm)
    tl.store(P + f[:, None, None] * 100 + ii * 10 + jj, Pn, mask=pm_)
    tl.store(FLAGS + f, stop.to(tl.int32) | (sw.to(tl.int32) << 1) | (sw2.to(tl.int32) << 2) | (inside.to(tl.int32) << 3), mask=fm)
    tl.store(FAO + f, fa, mask=fm)
    tl.store(MSIG + f, mean_sig, mask=fm)


#: launch defaults (tuned on an A10G by modal_ukf_triton.py)
F_DEFAULT, WARPS_DEFAULT = 2, 4


def npad(n: int) -> int:
    """The per-gradient tiles' width for n gradients: the next power of two, at least 16 - the kernel's one
    compile-time size, so one compiled copy per value (16-512 covers up to 512 gradients)."""
    return max(16, 1 << max(0, (int(n) - 1).bit_length()))


NPADS = (16, 32, 64, 128, 256, 512)


def _warm_one(n, F, num_warps):
    return warm((n,), F, num_warps, workers=1)[int(n)]


def warm(npads=NPADS, F=None, num_warps=None, workers=None) -> dict:
    """Compile the kernel for each NPAD on this GPU without running it (into Triton's cache, TRITON_CACHE_DIR):
    after this no scan compiles, whatever its gradient count or grid. Compiling is single-threaded CPU work,
    so the NPADs compile in parallel, `workers` spawned processes (default one per NPAD, up to the CPUs; 1:
    in this process, one after another). A cached NPAD is loaded, not compiled: with all six cached, 3.0 s one
    after another, 11.6 s in parallel (each process starts torch and the GPU) on an L4 - so call it once per
    machine or image, not at every start (a run loads its kernel from the cache by itself). Returns the seconds
    per NPAD. A script calling it with workers needs an `if __name__ == "__main__":` guard."""
    import os, time
    npads = tuple(int(n) for n in npads)
    workers = min(len(npads), os.cpu_count() or 1) if workers is None else int(workers)
    if workers > 1 and len(npads) > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        from .ukf import require_main_guard
        require_main_guard()
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            return dict(zip(npads, ex.map(_warm_one, npads, [F] * len(npads), [num_warps] * len(npads))))
    F, num_warps = F or F_DEFAULT, num_warps or WARPS_DEFAULT
    dev = torch.device("cuda")
    f32 = lambda *n: torch.empty(n, dtype=torch.float32, device=dev)
    out = {}
    for n in npads:
        t0 = time.time()
        ukf_step_block.warmup(f32(1), torch.empty(1, dtype=torch.int8, device=dev), f32(1), f32(1), f32(1), f32(1),
                              f32(1), f32(1), f32(1), torch.empty(1, dtype=torch.int32, device=dev), f32(1), f32(1),
                              1, 1, 1, 1, 1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                              F=F, NPAD=n, TIE=False, TIE_SW2=False, num_warps=num_warps, enable_fp_fusion=False, grid=(1,))
        out[n] = round(time.time() - t0, 1)
    return out


@torch.no_grad()
def advance(D: dict, xa, sa, Pa, oa, Q, Rs, step, max_steps, step_length=0.3, stopping_fa=0.08,
            stopping_threshold=0.06, tie_angle=0.0, tie_sw2=False, F=None, num_warps=None):
    """ukf.advance's contract on CUDA, float32. D is ukf.at_dtype(D, torch.float32, 'cuda'). tie_angle > 0: the
    kernel's TIE copy (compiled on first use; warm() compiles only today's)."""
    F = F or F_DEFAULT
    num_warps = num_warps or WARPS_DEFAULT
    dev = xa.device
    N = int(D["N"])
    B = xa.shape[0]
    nk, nj, ni = (int(v) for v in D["dim"])
    x = xa.contiguous().clone(); s = sa.contiguous().clone(); P = Pa.reshape(B, -1).contiguous().clone()
    o = oa.contiguous()
    d = torch.empty((B, 3), dtype=torch.float32, device=dev)
    flags = torch.empty(B, dtype=torch.int32, device=dev)
    fa = torch.empty(B, dtype=torch.float32, device=dev); ms = torch.empty(B, dtype=torch.float32, device=dev)
    # the step's constants, read back from the device once per tractography (cached on D) - not every step:
    # each .cpu() of a device tensor waits for the device
    key = (id(Q), float(Rs), N)
    cache = D.get("_triton_consts")
    if cache is None or cache[0] != key:
        q = torch.diagonal(Q).cpu()
        vox = D["voxel"].cpu()
        f32 = np.float32
        c = float(np.sqrt(f32(10) + f32(0.01), dtype=f32))
        W0 = float(f32(0.01) / (f32(10) + f32(0.01))); Wi = float(f32(0.5) / (f32(10) + f32(0.01)))
        rr = float(f32(1.0) / f32(Rs))
        cache = D["_triton_consts"] = (key, (float(vox[0]), float(vox[1]), float(vox[2]), float(vox.min())), rr,
                                       (float(q[0]), float(q[3])), (c, W0, Wi), D["g"][:N].contiguous(), D["b"][:N].contiguous())
    _, (v0, v1, v2, vmin), rr, (q0, q3), (c, W0, Wi), g, b = cache
    NPAD = npad(N)
    grid = ((B + F - 1) // F,)
    ukf_step_block[grid](D["A"], D["mask"], g, b, x, s, P, o, d, flags, fa, ms, B, N, nk, nj, ni, int(step), int(max_steps),
                         v0, v1, v2, vmin, float(step_length),
                         float(stopping_fa), float(stopping_threshold), rr, q0, q3, c, W0, Wi,
                         float(np.float32(np.cos(np.radians(tie_angle)))) if tie_angle else 1.0,
                         F=F, NPAD=NPAD, TIE=bool(tie_angle), TIE_SW2=bool(tie_angle and tie_sw2), num_warps=num_warps, enable_fp_fusion=False)
    stop = (flags & 1).bool()
    info = {"swap": (flags & 2).bool(), "swap2": (flags & 4).bool(), "fa": fa, "mean_signal": ms, "inside": (flags & 8).bool()}
    return x, s, P.reshape(B, 10, 10), d, stop, info
