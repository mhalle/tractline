"""Gibbs ringing removal by local subvoxel shifts (Kellner et al. 2016), batched over every slice of every volume in torch:
the opt-in `gibbs` slot (docs/sota-pipeline-design.md, section 4). The oracle is DIPY 1.12's gibbs_removal, whose code was
read to match its behavior (as the project's policy asks: read to understand; this is our own implementation, batched);
tests/test_gibbs.py checks it against DIPY.

    out = gibbs_removal(dwi, slice_axis=2, n_points=3, device="mps")    # numpy in, numpy out (float64 stays float64
                                                                         # but on MPS; anything else float32)

The method, per 2D slice (Kellner 2016, as DIPY implements it):
  along one axis at a time: the slice's Fourier transform along it, shifted by s voxels for s in SHIFTS (0.02 .. 0.9, 45
  values), positive and negative, each back as a magnitude; a voxel's oscillation is the smaller of its total variation
  over the n_points steps to its right and to its left (the line taken as periodic); for each voxel the positive shift
  and the negative shift that lower it most are kept (the first of equals; no shift when none lowers it), and the voxel's
  value is interpolated between their two images at zero shift;
  the slice corrected along each axis, the two corrected slices combined in 2D Fourier space with weights that favor
  each axis's correction where the other axis's frequency is high (G0, G1: (1 + cos k) ratios on a -pi..pi grid, DIPY's),
  and taken back as a magnitude.
Behaviors taken from DIPY's code beyond the paper: the shift grid (linspace(0.02, 0.9, 45)), the weights' grid
(linspace(-pi, pi, n), not the FFT's own frequencies) and boundary values, the magnitude after each inverse transform, and
the periodic total variation. The partial-Fourier caveat applies (the method assumes a full, symmetric k-space).
"""
from __future__ import annotations

import numpy as np
import torch

SHIFTS = np.linspace(0.02, 0.9, 45)
CHUNK_VOXELS = 8_000_000            # slice voxels per batch (the shifted images are kept per batch)


def weights(shape):
    """(G0, G1) for a slice of `shape`: G1 weights the correction along axis 1, G0 along axis 0. With a_i = 1 + cos k_i
    on a -pi..pi grid of each axis's length (DIPY's grid), G1 = a_0 / (a_0 + a_1) and G0 = a_1 / (a_0 + a_1): each
    axis's correction is trusted where the other axis's frequency is high. At the grid's ends a_i is 0, so the formula
    gives 0 or 1 on the edges; at the four corners (0 / 0) each takes half."""
    a0 = 1.0 + np.cos(np.linspace(-np.pi, np.pi, shape[0]))[:, None]
    a1 = 1.0 + np.cos(np.linspace(-np.pi, np.pi, shape[1]))[None, :]
    total = a0 + a1
    corner = total == 0
    total = np.where(corner, 1.0, total)
    G1 = np.where(corner, 0.5, a0 / total)
    G0 = np.where(corner, 0.5, a1 / total)
    return G0, G1


def _tv(x, n_points):
    """The oscillation of each sample of x (..., N) along its last axis, the line periodic: the smaller of the total
    variation over the n_points steps to the right and to the left. With d[j] = |x[j] - x[j + 1]|: right[i] = sum of
    d[i + n], n = 0 .. n_points - 1 (a box sum of one difference), and left[i] = right[i - n_points]. The line is padded
    periodically once and read by slices (torch.roll copies, and on MPS costs several times its memory traffic)."""
    N, p = x.shape[-1], n_points
    xp = torch.cat([x[..., N - p:], x, x[..., :p + 1]], -1)                # x[-p .. N + p]
    d = (xp[..., :-1] - xp[..., 1:]).abs()                                  # d[j - p], j = 0 .. N + 2p - 1
    box = d[..., :N + p]
    for n in range(1, p):
        box = box + d[..., n:N + p + n]                                     # right[i - p], i = 0 .. N + p - 1
    return torch.minimum(box[..., p:], box[..., :N])                        # right[i], left[i] = right[i - p]


_SHIFT_MATS = {}


def _shift_matrices(N, dtype, device):
    """For each shift s in SHIFTS and each sign: the real and imaginary parts of the N x N matrix that takes a line to its
    inverse transform after the phase ramp exp(+-2 pi i s m / N) (m the FFT's integer frequencies), transposed for x @ W:
    (W_re, W_im), each (N, 2 len(SHIFTS) N) with the shifts in order and, within each, the positive sign first. Made
    in float64 on the host once per (N, dtype, device). A line of N samples is short, so the transform as a product
    with these is faster on the GPU than an FFT per shift."""
    key = (N, dtype, str(device))
    if key not in _SHIFT_MATS:
        F = np.fft.fft(np.eye(N), axis=0)                                  # F @ x = fft(x)
        Fi = np.fft.ifft(np.eye(N), axis=0)
        m = np.fft.fftfreq(N) * N
        blocks = []
        for s in SHIFTS:
            for sign in (1.0, -1.0):
                M = Fi @ (np.exp(sign * 2j * np.pi * s * m / N)[:, None] * F)   # y = M @ x
                blocks.append(M.T)
        W = np.concatenate(blocks, 1)
        _SHIFT_MATS[key] = (torch.as_tensor(W.real, dtype=dtype, device=device),
                            torch.as_tensor(W.imag, dtype=dtype, device=device))
    return _SHIFT_MATS[key]


GROUP = 9                           # shifts (each with both signs) per matrix product and per minimum


def _removal_1d(x, n_points, per_shift=None):
    """x (..., N) real: corrected along the last axis. per_shift: pick the best shift one shift at a time (None: on the
    CPU) or by a minimum per group of shifts; the two choose alike, bit for bit (tests/test_gibbs.py)."""
    shape, N = x.shape, x.shape[-1]
    x2 = x.reshape(-1, N)
    Wr, Wi = _shift_matrices(N, x.dtype, x.device)
    tv_pos = _tv(x2, n_points)
    tv_neg = tv_pos.clone()
    img_pos, img_neg = x2.clone(), x2.clone()
    s_pos, s_neg = torch.zeros_like(x2), torch.zeros_like(x2)
    S = len(SHIFTS)
    for g in range(0, S, GROUP):
        h = min(GROUP, S - g)
        cols = slice(2 * g * N, 2 * (g + h) * N)
        img = torch.hypot(x2 @ Wr[:, cols], x2 @ Wi[:, cols]).reshape(-1, 2 * h, N)  # (L, 2h, N)
        tv = _tv(img, n_points)
        if per_shift or (per_shift is None and x.device.type == "cpu"):
            # One shift at a time, in order: torch's CPU argmin across the shifts (a strided middle axis) costs several
            # times the matrix products; the same choice (strict, the first of equals), bit for bit.
            for j in range(2 * h):
                tv_best, img_best, s_best = (tv_pos, img_pos, s_pos) if j % 2 == 0 else (tv_neg, img_neg, s_neg)
                t = tv[:, j]
                better = tv_best > t
                tv_best.copy_(torch.where(better, t, tv_best))
                img_best.copy_(torch.where(better, img[:, j], img_best))
                s_best.masked_fill_(better, float(SHIFTS[g + j // 2]))
            continue
        svals = torch.as_tensor(SHIFTS[g:g + h], dtype=x.dtype, device=x.device)
        for sign, (tv_best, img_best, s_best) in enumerate(((tv_pos, img_pos, s_pos), (tv_neg, img_neg, s_neg))):
            tg, ig = tv[:, sign::2], img[:, sign::2]                        # (L, h, N): this sign's shifts, in order
            k = torch.argmin(tg, dim=1, keepdim=True)                       # the first minimum within the group
            t = tg.gather(1, k).squeeze(1)
            better = tv_best > t                                            # strict: an earlier group's equal stays
            img_best.copy_(torch.where(better, ig.gather(1, k).squeeze(1), img_best))   # where: no sync on MPS
            s_best.copy_(torch.where(better, svals[k.squeeze(1)], s_best))
            tv_best.copy_(torch.where(better, t, tv_best))
    # the value at zero shift, linear between the image at +s_pos and the one at -s_neg (a voxel no shift improved: as is)
    span = s_pos + s_neg
    moved = span != 0
    span = torch.where(moved, span, torch.ones_like(span))
    return torch.where(moved, (img_pos * s_neg + img_neg * s_pos) / span, x2).reshape(shape)


def _removal_2d(slices, G0, G1, n_points):
    """slices (B, n0, n1) real: each corrected."""
    c1 = _removal_1d(slices, n_points)                                          # along axis 1 (the last)
    c0 = _removal_1d(slices.transpose(1, 2), n_points).transpose(1, 2)          # along axis 0
    C1 = torch.fft.fftshift(torch.fft.fft2(c1), dim=(-2, -1))
    C0 = torch.fft.fftshift(torch.fft.fft2(c0), dim=(-2, -1))
    return torch.fft.ifft2(C1 * G1 + C0 * G0).abs()


def gibbs_removal(vol, slice_axis=2, n_points=3, device="cpu"):
    """vol (X, Y, Z) or (X, Y, Z, V), numpy: every 2D slice across slice_axis corrected (module docstring). Float64 in,
    float64 out (on the CPU or CUDA; MPS has no float64 and runs float32); anything else float32."""
    v = np.asarray(vol)
    if v.ndim not in (3, 4) or not 0 <= slice_axis <= 2:
        raise ValueError("gibbs_removal: a 3D or 4D volume, slice_axis 0, 1 or 2")
    dev = torch.device(device)
    dt = torch.float64 if v.dtype == np.float64 and dev.type != "mps" else torch.float32
    four = v.ndim == 4
    w = v if four else v[..., None]
    w = np.moveaxis(w, slice_axis, 0)                                           # (S, n0, n1, V): slices first
    S, n0, n1, V = w.shape
    stack = np.ascontiguousarray(np.moveaxis(w, 3, 1)).reshape(S * V, n0, n1)    # every slice of every volume
    G0, G1 = (torch.as_tensor(g, dtype=dt, device=dev) for g in weights((n0, n1)))
    out = np.empty(stack.shape, np.float64 if dt == torch.float64 else np.float32)
    per = max(1, CHUNK_VOXELS // (n0 * n1))
    for a in range(0, len(stack), per):
        x = torch.as_tensor(stack[a:a + per], dtype=dt, device=dev)
        out[a:a + per] = _removal_2d(x, G0, G1, n_points).cpu().numpy()
    res = np.moveaxis(out.reshape(S, V, n0, n1), 1, 3)
    res = np.moveaxis(res, 0, slice_axis)
    return res if four else res[..., 0]
