"""UKF two-tensor tractography (the "2T simple" model), batched in torch over every half-fiber.

A re-implementation of pnlbwh/ukftractography 2d2b661 as configured by the ORG atlas
(numTensor 2, everything else default but seedingThreshold, stoppingFA, stoppingThreshold and
recordLength), written from its source to reproduce its fibers, not its idea. Line references are
to that commit. Every rule below is the binary's, including the ones that look like accidents:

  data      dwi_normalize.cc: short -> float32; b0 = bmax*|g|^2 <= 50 (float32); baseline = float32
            running sum of the b0s / count; signal = S/b0 in float32 (1e-10 where b0 == 0). NrrdData.cc:
            gradients in the voxel-axis frame after the RAS flip (inv(R/voxel) . MF), each appended
            again negated (2N rows); b = |g|^2 b when ||g| - 1| > 1e-4. Positions are (k, j, i) voxel
            indices; voxel spacings reversed to match.
  interp    NrrdData.cc:52 Interp3Signal: a Gaussian over the 3x3x3 voxels around round(pos), weight
            exp(-|d|^2 / min spacing) with d in mm, out-of-volume voxels skipped, divided by the weight
            sum (started at 1e-16), per gradient, in the loop order x, y, z.
  mask      NrrdData.cc:137: nearest voxel by C round(), read as SIGNED char, inside when > 0.
  seeds     tractography.cc:399-556: every mask voxel, k slowest; ONE offset for all of them,
            normalize(rand()%10001 - 5000 x3) * 0.5 voxel after srand(0) (libc-specific; passed in);
            rejected when any signal < 0 or non-finite, or mean signal < seedingThreshold, or the
            seed tensor's FA <= seedingThreshold. Seed tensor: least squares of log(S) (S <= 0 -> 1e-7)
            = -b g'Dg over the 2N rows, SVD, det(U) < 0 -> -U, lambda * 1e6, l2 = l3 = their mean.
            Two half-fibers per seed: [e1, l1, l2, e1, l1, l2] and [-e1, l1, l2, e1, l1, l2], P0 = 0.01 I.
  filter    unscented_kalman_filter.cc: kappa = 0.01, 21 sigma points from the Cholesky of P,
            F = renormalize m and clamp lambda >= 100, Q added after, the INFORMATION-form update.
  H         filter_Simple2T.cc:42: m normalized and flipped to m[0] >= 0, lambda clamped, D built as
            R diag(l1, l2, l2) R' 1e-6 with linalg.h's R, y = 1/2 exp(-b u'Du) + 1/2 exp(...) per row.
  step      tractography.cc:1641-1776: State2Tensor (normalize, clamp, flip to the previous direction),
            swap the tensors when m2 is closer to the previous direction, then when they are within 20
            degrees and the smaller FA <= 0.2 swap unless FA1 > 0.2; fa = 0 when l1 is not the largest;
            Euler step of 0.3 mm along m1 (index units: m1 reversed over the reversed spacings).
  stop      tractography.cc:1341-1382, at the new position: mask <= 0, mean of H(state) < stoppingThreshold,
            fa < stoppingFA, or step > ceil(maxHalfFiberLength / stepLength). The stopped step is not
            recorded. A point is recorded after the checks when (step + 1) % 6 == 0, the seed first.
  output    ukffiber.cc:91: the first half reversed without its seed, then the second half with it;
            fibers under 10 points dropped; seed order; points i2r (i, j, k, 1) in RAS mm.

Arithmetic is float64 where the binary's is double. Floating-point order inside Eigen's small
products is not reproduced, so fibers agree to rounding until a threshold decision flips.
track(dtype=torch.float32) runs the steps in float32 instead (an Apple GPU has no float64), from
the same float64 seeds; ukf32_compare.py measures what that changes.
"""
from __future__ import annotations

import math
import re
import numpy as np
import torch

KAPPA, P0, LMIN = 0.01, 0.01, 100.0

#: The binary's one seed offset, added to every mask voxel (tractography.cc:399-556): the vector of
#: rand() % 10001 - 5000 three times after srand(0), normalized to 0.5 voxel, (k, j, i). These are
#: macOS libc's draws (-4158, -2201, -2855); another libc draws others.
SRAND0_DRAWS = (-4158, -2201, -2855)
SRAND0_OFFSET = np.array(SRAND0_DRAWS, float) / np.linalg.norm(np.array(SRAND0_DRAWS, float)) * 0.5


# ------------------------------------------------------------------------------- data

def load(nhdr: str, mask_path: str, device="cpu", data=None) -> dict:
    """The DWI as the binary normalizes it. `data`, when given, replaces the file's voxels (same
    shape, (i, j, k, G)): a bootstrap replicate keeps the header, gradients and mask."""
    import nrrd
    raw, h = nrrd.read(nhdr)                                  # (i, j, k, G), Fortran index order
    if data is not None:
        assert data.shape == raw.shape, (data.shape, raw.shape)
        raw = data
    m, _ = nrrd.read(mask_path)
    return from_arrays(raw, h, m, device)


def from_arrays(dwi, header: dict, mask, device="cpu") -> dict:
    """load() without files: the DWI's voxels (i, j, k, G), its NRRD header as a dict (as pynrrd reads
    it, or as prep.prepare builds it) and the mask (i, j, k)."""
    keys = sorted(k for k in header if k.startswith("DWMRI_gradient_"))
    gtxt = [header[k] for k in keys]
    bmax_int = int(re.match(r"\s*(-?\d+)", header["DWMRI_b-value"]).group(1))
    b_hdr = float(header["DWMRI_b-value"])
    g32 = np.array([[np.float32(v) for v in s.split()] for s in gtxt], np.float32)
    mag = np.sqrt(g32[:, 0] * g32[:, 0] + g32[:, 1] * g32[:, 1] + g32[:, 2] * g32[:, 2], dtype=np.float32)
    nonzero = (np.float32(bmax_int) * mag * mag) > 50
    data = dwi.astype(np.float32)
    acc = np.zeros(data.shape[:3], np.float32)
    for gi in np.flatnonzero(~nonzero):                       # the float32 running sum, in gradient order
        acc = acc + data[..., gi]
    base = acc / np.float32((~nonzero).sum())
    nz = np.flatnonzero(nonzero)
    sig = np.empty(data.shape[:3] + (len(nz),), np.float32)
    for o, gi in enumerate(nz):
        sig[..., o] = np.where(base != 0, data[..., gi] / np.where(base != 0, base, 1), np.float32(1e-10))
    A = np.ascontiguousarray(sig.transpose(2, 1, 0, 3))      # (k, j, i, N): the binary's memory order

    g = np.array([[float(v) for v in gtxt[i].split()] for i in nz], np.float64)
    gn = np.linalg.norm(g, axis=1)
    b = np.where(np.abs(gn - 1) > 1e-4, gn * gn * b_hdr, b_hdr)
    g = g / gn[:, None]
    dirs = np.asarray(header["space directions"], dtype=object)
    sd = np.array([np.asarray(dirs[a], float) for a in range(3)])        # rows: axes i, j, k
    origin = np.asarray(header["space origin"], float).copy()
    mf = np.asarray(header.get("measurement frame", np.eye(3)), float).copy()  # mf[vector][component]
    space = header.get("space", "")
    if space in ("left-posterior-superior", "left-anterior-superior"):
        sd[:, 0] *= -1; mf[:, 0] *= -1; origin[0] *= -1
    if space == "left-posterior-superior":
        sd[:, 1] *= -1; mf[:, 1] *= -1; origin[1] *= -1
    spacing = np.linalg.norm(sd, axis=1)                       # (i, j, k)
    i2r = np.eye(4); i2r[:3, :3] = sd.T; i2r[:3, 3] = origin
    Rn = sd.T / spacing[None, :]
    MF = mf.T                                                  # measurement_frame(i, j) = mf[j][i]
    tm = np.linalg.inv(Rn) @ MF
    g = (tm @ g.T).T
    g = g / np.linalg.norm(g, axis=1)[:, None]
    g2 = np.concatenate([g, -g]); b2 = np.concatenate([b, b])

    mask = np.ascontiguousarray(np.asarray(mask).transpose(2, 1, 0)).astype(np.uint8).view(np.int8)  # signed char
    voxel = spacing[::-1].copy()                               # (k, j, i)
    t = lambda a, dt=torch.float64: torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=device)
    return {"A": torch.as_tensor(A, device=device), "g": t(g2), "b": t(b2), "voxel": t(voxel), "i2r": i2r,
            "mask": torch.as_tensor(mask, device=device), "dim": A.shape[:3], "N": len(nz)}


def cround(x: torch.Tensor) -> torch.Tensor:
    """C round(): half away from zero."""
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def interp(D: dict, pos: torch.Tensor) -> torch.Tensor:
    """Interp3Signal for a batch: pos (B, 3) in (k, j, i) -> (B, 2N) normalized signal."""
    A, vox = D["A"], D["voxel"]
    nk, nj, ni = D["dim"]
    N = D["N"]
    sigma = float(vox.min())
    r = cround(pos).long()
    out = torch.zeros(pos.shape[0], N, dtype=pos.dtype, device=pos.device)
    wsum = torch.full((pos.shape[0],), 1e-16, dtype=pos.dtype, device=pos.device)
    for xx in (-1, 0, 1):
        x = r[:, 0] + xx
        okx = (x >= 0) & (x < nk)
        dx = (x - pos[:, 0]) * vox[0]
        for yy in (-1, 0, 1):
            y = r[:, 1] + yy
            oky = (y >= 0) & (y < nj)
            dy = (y - pos[:, 1]) * vox[1]
            for zz in (-1, 0, 1):
                z = r[:, 2] + zz
                ok = okx & oky & (z >= 0) & (z < ni)
                dz = (z - pos[:, 2]) * vox[2]
                w = torch.exp(-(dx * dx + dy * dy + dz * dz) / sigma)
                w = torch.where(ok, w, torch.zeros_like(w))
                vals = A[x.clamp(0, nk - 1), y.clamp(0, nj - 1), z.clamp(0, ni - 1)].to(pos.dtype)
                out += w[:, None] * vals
                wsum += w
    out = out / wsum[:, None]
    return torch.cat([out, out], 1)


def _neighbors(D: dict) -> dict:
    """What _interp_half needs beside D, made once per track() (not cached in a module global: under
    torch.compile a global filled on first use is a guard that fails, and every worker recompiled twice):
    the 27 offsets in interp's loop order, the grid's dims and strides."""
    nk, nj, ni = D["dim"]
    dev = D["A"].device
    o = torch.tensor([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], device=dev)
    return {"nbr_off": o, "nbr_dims": torch.tensor([nk, nj, ni], device=dev), "nbr_strides": torch.tensor([nj * ni, ni, 1], device=dev)}


def _interp_half(D: dict, pos: torch.Tensor) -> torch.Tensor:
    """interp for the fast step: the N gradients once (interp's other N columns repeat them), the 27 voxels
    read in one gather and summed as one weighted product (B, 27) x (B, 27, N) - ~10 operations instead of 27
    rounds of ~15. The same weights; the sums in another order. D carries _neighbors(D)."""
    A, vox = D["A"], D["voxel"]
    N = D["N"]
    o, dims, strides = D["nbr_off"], D["nbr_dims"], D["nbr_strides"]
    idx = cround(pos).long()[:, None, :] + o[None]                       # (B, 27, 3), interp's loop order
    ok = ((idx >= 0) & (idx < dims)).all(-1)
    d = (idx - pos[:, None, :]) * vox
    w = torch.exp(-(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]) / vox.min())
    w = torch.where(ok, w, torch.zeros_like(w))
    lin = (torch.minimum(idx.clamp_min(0), dims - 1) * strides).sum(-1)
    vals = A.reshape(-1, N)[lin].to(pos.dtype)                           # (B, 27, N)
    return (w[:, None, :] @ vals)[:, 0] / (w.sum(1) + 1e-16)[:, None]


def mask_at(D: dict, pos: torch.Tensor) -> torch.Tensor:
    nk, nj, ni = D["dim"]
    r = cround(pos).long()
    ok = (r[:, 0] >= 0) & (r[:, 0] < nk) & (r[:, 1] >= 0) & (r[:, 1] < nj) & (r[:, 2] >= 0) & (r[:, 2] < ni)
    v = D["mask"][r[:, 0].clamp(0, nk - 1), r[:, 1].clamp(0, nj - 1), r[:, 2].clamp(0, ni - 1)]
    return torch.where(ok, v.to(torch.int32), torch.zeros_like(v, dtype=torch.int32))


# ------------------------------------------------------------------------------- model

def _normalize(v):
    return v / torch.sqrt((v * v).sum(-1, keepdim=True))


def _H_half(D: dict, X: torch.Tensor) -> torch.Tensor:
    """H for the fast step, (..., N): the N gradients once (H's other N columns repeat them), u'Du as one
    matrix product of the tensor's six components and the gradients' six products (u0^2, u1^2, u2^2,
    2 u0 u1, 2 u0 u2, 2 u1 u2), -b folded in. Another rounding than the binary's order."""
    N = D["N"]
    g, b = D["g"][:N], D["b"][:N]
    G = -b[None] * torch.stack([g[:, 0] * g[:, 0], g[:, 1] * g[:, 1], g[:, 2] * g[:, 2],
                                2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2], 2 * g[:, 1] * g[:, 2]])   # (6, N)
    out = 0
    for o in (0, 5):
        m = _normalize(X[..., o:o + 3])
        m = torch.where(m[..., :1] < 0, -m, m)
        l1 = X[..., o + 3].clamp_min(LMIN)
        l2 = X[..., o + 4].clamp_min(LMIN)
        m0, m1, m2 = m[..., 0], m[..., 1], m[..., 2]
        R = torch.stack([torch.stack([m0, m1, m2], -1),
                         torch.stack([m1, m1 * m1 / (1 + m0) - 1, m1 * m2 / (1 + m0)], -1),
                         torch.stack([m2, m1 * m2 / (1 + m0), m2 * m2 / (1 + m0) - 1], -1)], -2)
        L = torch.stack([l1, l2, l2], -1)
        Dm = (R * L[..., None, :]) @ R.transpose(-1, -2) * 1e-6
        d6 = torch.stack([Dm[..., 0, 0], Dm[..., 1, 1], Dm[..., 2, 2], Dm[..., 0, 1], Dm[..., 0, 2], Dm[..., 1, 2]], -1)
        out = out + torch.exp(d6 @ G) * 0.5
    return out


def H(D: dict, X: torch.Tensor) -> torch.Tensor:
    """filter_Simple2T::H: X (..., 10) -> (..., 2N). Computed for the N gradients and repeated: the
    other N are the same gradients negated (load), and u'Du is unchanged by u -> -u, bit for bit
    (every product of two negated factors is the product of the factors, exactly)."""
    N = D["N"]
    g, b = D["g"][:N], D["b"][:N]
    out = 0
    for o in (0, 5):
        m = _normalize(X[..., o:o + 3])
        m = torch.where(m[..., :1] < 0, -m, m)
        l1 = X[..., o + 3].clamp_min(LMIN)
        l2 = X[..., o + 4].clamp_min(LMIN)
        m0, m1, m2 = m[..., 0], m[..., 1], m[..., 2]
        # linalg.h diffusion(): R rows (m0, m1, m2), (m1, m1^2/(1+m0) - 1, m1 m2/(1+m0)), (m2, m1 m2/(1+m0), m2^2/(1+m0) - 1)
        R = torch.stack([torch.stack([m0, m1, m2], -1),
                         torch.stack([m1, m1 * m1 / (1 + m0) - 1, m1 * m2 / (1 + m0)], -1),
                         torch.stack([m2, m1 * m2 / (1 + m0), m2 * m2 / (1 + m0) - 1], -1)], -2)
        L = torch.stack([l1, l2, l2], -1)
        Dm = (R * L[..., None, :]) @ R.transpose(-1, -2) * 1e-6          # (..., 3, 3)
        # u.(D u) in the binary's order - (D u)_r = D_r0 u0 + D_r1 u1 + D_r2 u2, then the dot -
        # one component at a time, so nothing (..., 2N, 3)-sized is ever held
        u0, u1, u2 = g[:, 0], g[:, 1], g[:, 2]
        q = 0
        for r, ur in enumerate((u0, u1, u2)):
            Du_r = Dm[..., r, 0, None] * u0 + Dm[..., r, 1, None] * u1 + Dm[..., r, 2, None] * u2
            q = q + ur * Du_r
        out = out + torch.exp(-b * q) * 0.5
    return torch.cat([out, out], -1)


def F(X: torch.Tensor) -> torch.Tensor:
    X = X.clone()
    for o in (0, 5):
        m = X[..., o:o + 3]
        X[..., o:o + 3] = m * (1.0 / torch.sqrt((m * m).sum(-1, keepdim=True)))
    X[..., [3, 4, 8, 9]] = X[..., [3, 4, 8, 9]].clamp_min(LMIN)
    return X


def ukf_step(D: dict, x, P, z, Q, Rs):
    n = 10
    W = torch.full((2 * n + 1,), 0.5 / (n + KAPPA), dtype=x.dtype, device=x.device)
    W[0] = KAPPA / (n + KAPPA)
    L, _ = torch.linalg.cholesky_ex(P)
    dX = math.sqrt(n + KAPPA) * L.transpose(1, 2)
    X = F(torch.cat([x[:, None], x[:, None] + dX, x[:, None] - dX], 1))
    xh = torch.einsum("s,bsk->bk", W, X)
    Xt = X - xh[:, None]
    Pm = torch.einsum("bsi,s,bsj->bij", Xt, W, Xt) + Q
    Yk = torch.linalg.inv(Pm)
    yh = torch.einsum("bij,bj->bi", Yk, xh)
    Z = H(D, X)
    zh = torch.einsum("s,bsn->bn", W, Z)
    Pxz = torch.einsum("bsi,s,bsn->bin", Xt, W, Z - zh[:, None])
    Ht = Yk @ Pxz
    r = 1.0 / Rs
    I = (Ht * r) @ Ht.transpose(1, 2)
    i = torch.einsum("bin,bn->bi", Ht * r, (z - zh) + torch.einsum("bin,bi->bn", Pxz, yh))
    Pn = torch.linalg.inv(Yk + I)
    return torch.einsum("bij,bj->bi", Pn, i + yh), Pn


def _spd_inv(A):
    return torch.cholesky_inverse(torch.linalg.cholesky_ex(A)[0])


def _ukf_step_fast(D: dict, x, P, z, Q, Rs):
    """ukf_step for track(fast=True), z (B, N): the N measurements once with every sum over the 2N doubled (the
    other N repeat them; the Metal and Triton kernels do the same), the weights applied first and each
    contraction one batched product, both inverses from Cholesky factors (both matrices SPD), H as _H_half.
    The same filter in another rounding: ukf_cpu_tune.py measures it against float64."""
    n = 10
    W = torch.full((2 * n + 1,), 0.5 / (n + KAPPA), dtype=x.dtype, device=x.device)
    W[0] = KAPPA / (n + KAPPA)
    L, _ = torch.linalg.cholesky_ex(P)
    dX = math.sqrt(n + KAPPA) * L.transpose(1, 2)
    X = F(torch.cat([x[:, None], x[:, None] + dX, x[:, None] - dX], 1))
    Wc = W[None, :, None]
    xh = (Wc * X).sum(1)
    Xt = X - xh[:, None]
    XwT = (Xt * Wc).transpose(1, 2)                                      # (B, 10, 21)
    Yk = _spd_inv(XwT @ Xt + Q)
    yh = (Yk @ xh[..., None])[..., 0]
    Z = _H_half(D, X)                                                    # (B, 21, N)
    zh = (Wc * Z).sum(1)
    Pxz = XwT @ (Z - zh[:, None])                                        # (B, 10, N)
    Ht = Yk @ Pxz
    Htr = Ht * (1.0 / Rs)
    I = 2 * (Htr @ Ht.transpose(1, 2))
    i = 2 * (Htr @ ((z - zh) + (Pxz.transpose(1, 2) @ yh[..., None])[..., 0])[..., None])[..., 0]
    Pn = _spd_inv(Yk + I)
    return (Pn @ (i + yh)[..., None])[..., 0], Pn


def l2fa(l1, l2, l3):
    eq = l2 == l3
    a = (l1 - l2).abs() / torch.sqrt(l1 * l1 + 2.0 * l2 * l2)
    c = torch.sqrt(0.5 * ((l1 - l2) ** 2 + (l2 - l3) ** 2 + (l3 - l1) ** 2) / (l1 * l1 + l2 * l2 + l3 * l3))
    return torch.where(eq, a, c)


def _swap(state, P, sel):
    """SwapState2T on the rows in sel."""
    s2 = torch.cat([state[:, 5:], state[:, :5]], 1)
    P2 = torch.cat([torch.cat([P[:, 5:, 5:], P[:, 5:, :5]], 2), torch.cat([P[:, :5, 5:], P[:, :5, :5]], 2)], 1)
    return torch.where(sel[:, None], s2, state), torch.where(sel[:, None, None], P2, P)


# ------------------------------------------------------------------------------- seeds

def seeds(D: dict, offset_kji=SRAND0_OFFSET, seeding_threshold=0.1):
    """Seed points (k, j, i) and their two initial states, in the binary's seed order."""
    dev = D["A"].device
    kji = torch.nonzero(D["mask"] > 0)                               # k slowest, i fastest: the binary's loop
    off = torch.as_tensor(offset_kji, dtype=torch.float64, device=dev)
    pts = kji.double() + off
    S = torch.cat([interp(D, pts[s:s + 65536]) for s in range(0, len(pts), 65536)])
    N = D["N"]
    keep = (S[:, :N] >= 0).all(1) & torch.isfinite(S[:, :N]).all(1) & (S.mean(1) >= seeding_threshold)
    pts, S = pts[keep], S[keep]
    e1, l1, l2, fa = _seed_tensor(D, S)
    ok = fa > seeding_threshold
    pts, e1, l1, l2, fa = pts[ok], e1[ok], l1[ok], l2[ok], fa[ok]
    lam = torch.stack([l1, l2], 1)
    fwd = torch.cat([e1, lam, e1, lam], 1)
    inv = torch.cat([-e1, lam, e1, lam], 1)
    return pts, fwd, inv, e1, fa


def seed_states(D: dict, pts: torch.Tensor, seeding_threshold=0.1):
    """The initial states at given seed points (k, j, i), none rejected: (fwd, inv, e1, fa, accepted),
    `accepted` saying whether seeds() would have kept each point on this data. For tracking the same
    points through different data (a bootstrap replicate)."""
    N = D["N"]
    S = torch.cat([interp(D, pts[s:s + 65536]) for s in range(0, len(pts), 65536)])
    e1, l1, l2, fa = _seed_tensor(D, S)
    accepted = ((S[:, :N] >= 0).all(1) & torch.isfinite(S[:, :N]).all(1) & (S.mean(1) >= seeding_threshold)
                & (fa > seeding_threshold))
    lam = torch.stack([l1, l2], 1)
    return torch.cat([e1, lam, e1, lam], 1), torch.cat([-e1, lam, e1, lam], 1), e1, fa, accepted


def _seed_tensor(D: dict, S: torch.Tensor):
    """The seed tensor from interpolated signals S (B, 2N): (e1, l1, l2, fa), tractography.cc's rules."""
    g, b = D["g"], D["b"]
    dev = S.device
    B = -b[:, None] * torch.stack([g[:, 0] ** 2, 2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2], g[:, 1] ** 2,
                                   2 * g[:, 1] * g[:, 2], g[:, 2] ** 2], 1)
    logS = torch.log(torch.where(S <= 0, torch.full_like(S, 10e-8), S))
    # every seed's least squares has the same design matrix: its pseudo-inverse once (SVD, as gelsd), applied to all -
    # a batched per-seed gelsd took 1-4 s on the M2 and far more on a slow host CPU, ahead of every GPU tracking
    d = (logS.double() @ torch.linalg.pinv(B.cpu().double()).T.to(dev)).to(S.dtype)
    Dt = torch.stack([torch.stack([d[:, 0], d[:, 1], d[:, 2]], -1), torch.stack([d[:, 1], d[:, 3], d[:, 4]], -1),
                      torch.stack([d[:, 2], d[:, 4], d[:, 5]], -1)], -2)
    U, sv, _ = torch.linalg.svd(Dt)
    U = torch.where((torch.linalg.det(U) < 0)[:, None, None], -U, U)
    e1 = U[:, :, 0]
    l1 = sv[:, 0] * 1e6
    l2 = (sv[:, 1] * 1e6 + sv[:, 2] * 1e6) / 2.0
    return e1, l1, l2, l2fa(l1, l2, l2)


# ------------------------------------------------------------------------------- tracking

def at_dtype(D: dict, dtype, device=None) -> dict:
    """D with its float64 tensors (gradients, b-values, spacings) in `dtype`, and everything on
    `device` (default: where D is); the signal stays float32, as dwi_normalize.cc makes it, and is
    cast where it is read (interp)."""
    device = D["A"].device if device is None else torch.device(device)
    if dtype == torch.float64 and device == D["A"].device:
        return D
    return {**D, "A": D["A"].to(device), "mask": D["mask"].to(device),
            **{k: D[k].to(device=device, dtype=dtype) for k in ("g", "b", "voxel")}}


def advance(D: dict, xa, sa, Pa, oa, Q, Rs, step, max_steps, step_length=0.3, stopping_fa=0.08,
            stopping_threshold=0.06):
    """One step of every live half-fiber: the filter at xa, the tensors, the swaps, the Euler step
    and the stopping checks at the new position (tractography.cc:1641-1776, 1341-1382). Every
    tensor in `dtype` of the state. Returns (x, state, P, direction, stop, info), info holding the
    intermediate decisions the one-step fixtures compare: both swaps, fa, mean signal, in-mask."""
    vox = D["voxel"]
    fast = D.get("fast", False)
    z = _interp_half(D, xa) if fast else interp(D, xa)
    sa, Pa = (_ukf_step_fast if fast else ukf_step)(D, sa, Pa, z, Q, Rs)
    # State2Tensor2T
    m1, m2 = _normalize(sa[:, 0:3]), _normalize(sa[:, 5:8])
    L1 = torch.stack([sa[:, 3].clamp_min(LMIN), sa[:, 4].clamp_min(LMIN)], 1)
    L2 = torch.stack([sa[:, 8].clamp_min(LMIN), sa[:, 9].clamp_min(LMIN)], 1)
    m1 = torch.where(((m1 * oa).sum(1) < 0)[:, None], -m1, m1)
    m2 = torch.where(((m2 * oa).sum(1) < 0)[:, None], -m2, m2)
    fat1, fat2 = l2fa(L1[:, 0], L1[:, 1], L1[:, 1]), l2fa(L2[:, 0], L2[:, 1], L2[:, 1])
    angle = torch.rad2deg(torch.acos((m1 * m2).sum(1)))
    sw = (m1 * oa).sum(1) < (m2 * oa).sum(1)
    m1, m2 = torch.where(sw[:, None], m2, m1), torch.where(sw[:, None], m1, m2)
    L1, L2 = torch.where(sw[:, None], L2, L1), torch.where(sw[:, None], L1, L2)
    fat1, fat2 = torch.where(sw, fat2, fat1), torch.where(sw, fat1, fat2)
    sa, Pa = _swap(sa, Pa, sw)
    sw2 = (angle <= 20) & (torch.minimum(fat1, fat2) <= 0.2) & ~(fat1 > 0.2)
    m1, m2 = torch.where(sw2[:, None], m2, m1), torch.where(sw2[:, None], m1, m2)
    L1, L2 = torch.where(sw2[:, None], L2, L1), torch.where(sw2[:, None], L1, L2)
    sa, Pa = _swap(sa, Pa, sw2)
    fa = torch.where(L1[:, 0] < L1[:, 1], torch.zeros_like(fat1), l2fa(L1[:, 0], L1[:, 1], L1[:, 1]))
    xa = xa + torch.stack([m1[:, 2] / vox[0], m1[:, 1] / vox[1], m1[:, 0] / vox[2]], 1) * step_length
    # the checks, at the new position
    mean_sig = (_H_half if fast else H)(D, sa[:, None])[:, 0].mean(1)
    inside = mask_at(D, xa) > 0
    stop = ~inside | (mean_sig < stopping_threshold) | (fa < stopping_fa) | (step > max_steps)
    return xa, sa, Pa, m1, stop, {"swap": sw, "swap2": sw2, "fa": fa, "mean_signal": mean_sig, "inside": inside}


def _track_block(D: dict, x, state, old, step_fn, params, s0=0, capture=None, progress=None):
    """One batch of half-fibers, from their seed states to their recorded points: (list of (n_rec, 3)
    (k, j, i) arrays, steps taken). A batch's results depend on its composition at the rounding level
    (vectorized math rounds by position), so blocks are fixed by `batch`, never by worker count."""
    dtype, dev = x.dtype, x.device
    Q, Rs, max_steps, spr = params["Q"], params["Rs"], params["max_steps"], params["spr"]
    nb = len(x)
    P = (P0 * torch.eye(10, dtype=dtype, device=dev)).expand(nb, 10, 10).clone()
    rec = torch.full((nb, max_steps // spr + 2, 3), float("nan"), dtype=dtype, device=dev)
    rec[:, 0] = x
    nrec = torch.ones(nb, dtype=torch.long, device=dev)
    alive = torch.arange(nb, device=dev)
    xa, sa, Pa, oa = x.clone(), state.clone(), P, old.clone()            # the live half-fibers, compacted as they stop
    step, steps = 0, 0
    while len(alive):
        step += 1
        steps += len(alive)
        if capture:
            capture(step, s0 + alive, xa, sa, Pa, oa)
        xa, sa, Pa, m1, stop, _ = step_fn(D, xa, sa, Pa, oa, Q, Rs, step, max_steps, params["step_length"],
                                          params["stopping_fa"], params["stopping_threshold"])
        go = torch.nonzero(~stop).squeeze(1)                              # one synchronization; then plain gathers
        if (step + 1) % spr == 0:
            ia = alive[go]
            rec[ia, nrec[ia]] = xa[go]
            nrec[ia] += 1
        alive = alive[go]
        xa, sa, Pa, oa = xa[go], sa[go], Pa[go], m1[go]                     # a stopped half-fiber's state is never read again
        if progress and step % 50 == 0:
            progress(s0, step, len(alive))
    r = rec.cpu().numpy(); nr = nrec.cpu().numpy()
    return [r[k, :nr[k]] for k in range(nb)], steps


def _track_stream(D: dict, x0, st0, dir0, step_fn, params, capacity, progress=None):
    """All half-fibers through one stream of at most `capacity` live, topped up in seed order as they stop - the GPU
    kernels only: a fiber's arithmetic there does not depend on its batch (gpu_track_profile.py: blocks of 50,000 and
    200,000 bit for bit), so this tracks every fiber exactly as _track_block does, but keeps the device full until the
    last fibers instead of letting every block run down to its longest one. Topped up only on steps that are multiples
    of the recording interval, when a quarter of the places are free: every half-fiber's age then has the same
    remainder, so the survivors record together on the block's steps, and the step limit (the kernels get step 0)
    is checked only on the step a cohort reaches it, known on the host - the per-step work stays the block's.
    Returns _track_block's (halves in seed order, steps)."""
    dtype, dev = x0.dtype, x0.device
    Q, Rs, max_steps, spr = params["Q"], params["Rs"], params["max_steps"], params["spr"]
    sl, sfa, sth = params["step_length"], params["stopping_fa"], params["stopping_threshold"]
    n = len(x0)
    rec = torch.full((n, max_steps // spr + 2, 3), float("nan"), dtype=dtype, device=dev)
    rec[:, 0] = x0
    nrec = torch.ones(n, dtype=torch.long, device=dev)
    P0m = P0 * torch.eye(10, dtype=dtype, device=dev)
    empty = lambda *shape: torch.empty(0, *shape, dtype=dtype, device=dev)
    alive, born = torch.empty(0, dtype=torch.long, device=dev), torch.empty(0, dtype=torch.long, device=dev)
    xa, sa, Pa, oa = empty(3), empty(10), empty(10, 10), empty(3)
    births = set()
    nxt, steps, it = 0, 0, 0
    while True:
        if nxt < n and it % spr == 0 and len(alive) <= capacity * 3 // 4:      # top up, in seed order
            k = min(capacity - len(alive), n - nxt)
            alive = torch.cat([alive, torch.arange(nxt, nxt + k, device=dev)])
            born = torch.cat([born, torch.full((k,), it, dtype=torch.long, device=dev)])
            xa, sa, oa = torch.cat([xa, x0[nxt:nxt + k]]), torch.cat([sa, st0[nxt:nxt + k]]), torch.cat([oa, dir0[nxt:nxt + k]])
            Pa = torch.cat([Pa, P0m.expand(k, 10, 10)])
            births.add(it)
            nxt += k
        if not len(alive):
            break
        it += 1
        steps += len(alive)
        xa, sa, Pa, m1, stop, _ = step_fn(D, xa, sa, Pa, oa, Q, Rs, 0, max_steps, sl, sfa, sth)
        if it - max_steps - 1 in births:                                    # a cohort reaches the limit (age > max_steps)
            stop = stop | (it - born > max_steps)
        go = torch.nonzero(~stop).squeeze(1)                                  # the one synchronization a step
        if (it + 1) % spr == 0:                                               # every age's remainder is it's: all record
            ia = alive[go]
            rec[ia, nrec[ia]] = xa[go]
            nrec[ia] += 1
        alive, born, xa, sa, Pa, oa = alive[go], born[go], xa[go], sa[go], Pa[go], m1[go]
        if progress and it % 50 == 0:
            progress(nxt, it, len(alive))
    r = rec.cpu().numpy(); nr = nrec.cpu().numpy()
    return [r[k, :nr[k]] for k in range(n)], steps


_COMPILED = None


def _compiled_advance():
    """advance through torch.compile (Inductor's C++ backend), dynamic batch size - one per process; its code
    is cached on disk (TORCHINDUCTOR_CACHE_DIR), so a second process, or run, loads it instead of compiling.
    Single-threaded kernels (cpp.threads 1): Inductor otherwise bakes the compiling process's thread count into
    each kernel's OpenMP region - compiled in a parent at 4 threads, 8 workers ran 32 threads on 8 cores (load
    35) and gained nothing; the CPU tracker's parallelism is its processes, one thread each.
    One graph for every step: the step limit (step > max_steps stops every half-fiber, the same rule) is applied
    outside it, and a lone last half-fiber is tracked as a pair with itself (sizes 1 and 0 are specialized),
    so no guard fails mid-run - each failure was a recompile, 2-4 s with the disk cache, 10-20 s without.
    The padded pair's arithmetic is a batch of two's, the same in every process."""
    global _COMPILED
    if _COMPILED is None:
        core = torch.compile(lambda D, xa, sa, Pa, oa, Q, Rs, sl, sfa, sth: advance(D, xa, sa, Pa, oa, Q, Rs, 0, 1, sl, sfa, sth),
                             dynamic=True, options={"cpp.threads": 1})

        def step_fn(D, xa, sa, Pa, oa, Q, Rs, step, max_steps, step_length=0.3, stopping_fa=0.08, stopping_threshold=0.06):
            one = len(xa) == 1
            if one:
                xa, sa, Pa, oa = (torch.cat([t, t]) for t in (xa, sa, Pa, oa))
            xa, sa, Pa, m1, stop, info = core(D, xa, sa, Pa, oa, Q, Rs, step_length, stopping_fa, stopping_threshold)
            if one:
                xa, sa, Pa, m1, stop = xa[:1], sa[:1], Pa[:1], m1[:1], stop[:1]
                info = {k: v[:1] for k, v in info.items()}
            if step > max_steps:
                stop = torch.ones_like(stop)
            return xa, sa, Pa, m1, stop, info

        _COMPILED = step_fn
    return _COMPILED


def can_compile() -> bool:
    """Whether the pipeline compiles the CPU tracker's step (track(compile=True)): only when TRACTLINE_COMPILE=1
    and a C++ compiler is on the PATH (CXX, c++, g++ or clang++; Inductor's CPU backend needs one). Off by
    default: one process gains 1.7-2.9x, but every spawned worker pays torch.compile's start (tracing, loading
    the cached code: ~8 s on the M2, ~100 s on 32 x86 cores), and with 8 or more workers the compiled tracker
    was slower than the uncompiled one (NOTES 2026-10-06)."""
    import os, shutil
    if os.environ.get("TRACTLINE_COMPILE") != "1":
        return False
    return any(shutil.which(c) for c in (os.environ.get("CXX"), "c++", "g++", "clang++") if c)


def _warm_compiled(D, blocks, params):
    """Compile the fast step here, on 16 half-fibers of the first block tracked to their ends (every batch size
    down to 1, so no later recompile), before workers load it from the disk cache. Returns (step function,
    seconds, None) or, when compiling fails, (advance, seconds, the error) - the caller falls back."""
    import time
    t0 = time.perf_counter()
    try:
        f = _compiled_advance()
        _, x, st, d = blocks[0]
        _track_block(D, x[:16], st[:16], d[:16], f, params)
        return f, time.perf_counter() - t0, None
    except Exception as e:                                               # noqa: BLE001 - no compiler, an Inductor failure
        return advance, time.perf_counter() - t0, e


_WORKER_D = None


def _worker_init(D, threads):
    global _WORKER_D
    torch.set_num_threads(threads)
    _WORKER_D = {k: torch.from_numpy(v) if k in _SHARED_KEYS and isinstance(v, np.ndarray) else v for k, v in D.items()}


_SHARED_KEYS = ("A", "g", "b", "voxel", "mask")


def _worker_block(args):
    x, st, d, params = (torch.from_numpy(a) if isinstance(a, np.ndarray) else a for a in args)
    return _track_block(_WORKER_D, x, st, d, _compiled_advance() if params.get("compile") else advance, params)


def require_main_guard():
    """Spawned workers re-import the main script: without an `if __name__ == "__main__":` guard, each
    worker re-runs it from the top. A comparison script did - 8 workers each re-ran its GPU pipeline,
    1.9 GB apiece, and the M2 swapped. Raises when the running script has no module-level `if` testing
    `__name__` against "__main__" (read from its syntax, so a comment or string does not count; either
    order, `==` or `in`). It cannot tell whether work also runs above the guard. A session without a
    script file (REPL, notebook, -c) passes. pipeline.run and pipeline.track call it before any work on
    the CPU path."""
    import ast, sys
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if not path or not str(path).endswith(".py"):
        return
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except (OSError, SyntaxError, ValueError):
        return

    def is_guard(test):
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            return False
        sides = [test.left, test.comparators[0]]
        names = [s for s in sides if isinstance(s, ast.Name) and s.id == "__name__"]
        consts = [s for s in sides if isinstance(s, ast.Constant) and s.value == "__main__"]
        seqs = [s for s in sides if isinstance(s, (ast.Tuple, ast.List, ast.Set))
                and any(isinstance(e, ast.Constant) and e.value == "__main__" for e in s.elts)]
        return bool(names) and (bool(consts) and isinstance(test.ops[0], ast.Eq) or bool(seqs) and isinstance(test.ops[0], ast.In))

    if not any(isinstance(node, ast.If) and is_guard(node.test) for node in tree.body):
        raise RuntimeError(f"{path}: the CPU tracker spawns worker processes, which re-import the main script - "
                           f"put its work under `if __name__ == \"__main__\":` (or pass workers=1)")


def _parallel(D, blocks, params, workers, threads_per_worker):
    """The blocks on `workers` processes (spawned: macOS and Windows cannot fork safely), D's tensors
    shared through shared memory - or copied to each worker when TRACTOGRAPHY_SHARE=0 (a container's
    /dev/shm can be smaller than the signal); results in block order."""
    import os, torch.multiprocessing as tmp
    require_main_guard()
    if os.environ.get("TRACTOGRAPHY_SHARE", "1") == "1":
        for v in D.values():
            if torch.is_tensor(v):
                v.share_memory_()
        tasks = [(x, st, d, params) for _, x, st, d in blocks]
    else:                                                            # a small /dev/shm (containers): each worker gets a copy,
        D = {k: v.numpy() if torch.is_tensor(v) and k in _SHARED_KEYS else v for k, v in D.items()}
        # and the blocks go as numpy: torch's reducers would move a tensor (a view: its whole base) to /dev/shm
        tasks = [(x.numpy().copy(), st.numpy().copy(), d.numpy().copy(), params) for _, x, st, d in blocks]
    with tmp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(D, threads_per_worker)) as pool:
        return pool.map(_worker_block, tasks, chunksize=1)


def track(D: dict, offset_kji=SRAND0_OFFSET, seeding_threshold=0.1, stopping_fa=0.08, stopping_threshold=0.06,
          step_length=0.3, record_length=1.8, max_half_length=250.0, Qm=0.001, Ql=50.0, Rs=0.02,
          batch=50_000, progress=None, select=None, dtype=torch.float64, device=None, capture=None,
          seed_points=None, backend="torch", workers=1, threads_per_worker=1, fast=False, compile=False, stream=True):
    """All fibers. Returns (points list of (n, 3) RAS arrays in seed order, stats). With `select`
    (indices into the seed list), only those seeds are tracked; stats["seed_index"] then says which
    seed each returned fiber came from, stats["seed_voxel"] its mask voxel (k, j, i) - the key to
    match fibers across runs whose masks, and so seed lists, differ.

    `dtype` is the tracking arithmetic: float64 is the binary's; float32 is what an Apple GPU can do.
    Seeds and their initial states are computed in float64 either way, so two dtypes start from the
    same half-fibers and differ only in the steps. `device` is where the steps run (default: where D
    is); seeds are made where D is, so a float64 D on the CPU can seed an MPS run, which has no
    float64. `capture(step, index, xa, sa, Pa, oa)`, when given, sees every step's inputs before the
    step, `index` being the half-fibers' (for one-step fixtures). `seed_points` (k, j, i), when given,
    replaces the seeds: every point is tracked from its state on this data (seed_states), none
    rejected, and `select` indexes them. `backend` "metal" takes each step with ukf_metal's kernel
    (float32, device "mps"), "triton" with ukf_triton's and "triton_block" with ukf_triton_block's
    (float32, "cuda"); the loop, the recording and the joining stay these. `workers` > 1 (the CPU
    torch tracker) tracks the batches on that many processes of `threads_per_worker` threads each;
    the result is the serial one at the same `batch`, bit for bit. `fast` (the torch tracker): the N gradients
    once with the filter's sums doubled, H as a matrix product, the 27 voxels in one gather, both inverses from
    Cholesky factors - another rounding, not the binary's (_ukf_step_fast). `compile` (with fast, the CPU torch
    tracker): that step through torch.compile, compiled here first and loaded by the workers from Inductor's
    disk cache; when compiling fails, a warning and the uncompiled step. stats["compile_s"]: that time. `stream` (the
    GPU kernels, "metal" and "triton_block"; on by default): one stream of at most `batch` live half-fibers, topped up
    as they stop, instead of blocks of `batch` (_track_stream: the same fibers, bit for bit)."""
    step_fn = advance
    if backend == "metal":
        from . import ukf_metal as ukf_metal
        dtype, device, step_fn = torch.float32, "mps", ukf_metal.advance
    elif backend == "triton":
        from . import ukf_triton as ukf_triton
        dtype, device, step_fn = torch.float32, "cuda", ukf_triton.advance
    elif backend == "triton_block":
        from . import ukf_triton_block as ukf_triton_block
        dtype, device, step_fn = torch.float32, "cuda", ukf_triton_block.advance
    if seed_points is None:
        pts, fwd, inv, e1, fa0 = seeds(D, offset_kji, seeding_threshold)
    else:
        pts = torch.as_tensor(seed_points, dtype=torch.float64, device=D["A"].device)
        fwd, inv, e1, fa0, _ = seed_states(D, pts, seeding_threshold)
    sel = torch.arange(len(pts), device=pts.device) if select is None else torch.as_tensor(select, device=pts.device)
    vox_kji = np.rint(pts[sel].cpu().numpy() - np.asarray(offset_kji, float)).astype(np.int64)   # each seed's mask voxel (float64 seeds)
    D = at_dtype(D, dtype, device)
    if fast:
        D = {**D, "fast": True}
        D.update(_neighbors(D))
    dev = D["A"].device
    pts, fwd, inv, e1, fa0 = (t[sel].to(dtype).to(dev) for t in (pts, fwd, inv, e1, fa0))
    sel = sel.cpu()
    S = len(pts)
    x0 = torch.stack([pts, pts], 1).reshape(-1, 3)                    # half-fibers 2s (fwd), 2s+1 (inv)
    st0 = torch.stack([fwd, inv], 1).reshape(-1, 10)
    dir0 = torch.stack([e1, -e1], 1).reshape(-1, 3)
    Q = torch.diag(torch.tensor([Qm] * 3 + [Ql] * 2 + [Qm] * 3 + [Ql] * 2, dtype=dtype, device=dev))
    max_steps = math.ceil(max_half_length / step_length)
    spr = int(record_length / step_length)
    params = dict(Q=Q, Rs=Rs, max_steps=max_steps, spr=spr, step_length=step_length, stopping_fa=stopping_fa,
                  stopping_threshold=stopping_threshold)
    blocks = [(s0, x0[s0:s0 + batch], st0[s0:s0 + batch], dir0[s0:s0 + batch]) for s0 in range(0, len(x0), batch)]
    compile_s = 0.0
    if compile and blocks:
        assert backend == "torch" and fast and dev.type == "cpu" and capture is None, "compile: the CPU torch tracker with fast"
        step_fn, compile_s, err = _warm_compiled(D, blocks, params)
        if err is not None:
            import warnings
            warnings.warn(f"ukf.track(compile=True): compiling failed ({type(err).__name__}: {str(err)[:200]}); "
                          "tracking with the uncompiled fast step")
        params = {**params, "compile": err is None}
    if workers > 1:
        assert backend == "torch" and dev.type == "cpu" and capture is None, "workers: the CPU torch tracker only"
        done = _parallel(D, blocks, params, workers, threads_per_worker)
    elif stream and backend in ("metal", "triton_block") and capture is None:
        done = [_track_stream(D, x0, st0, dir0, step_fn, params, batch, progress)]
    else:
        done = [_track_block(D, x, st, d, step_fn, params, s0, capture, progress) for s0, x, st, d in blocks]
    halves = [h for hs, _ in done for h in hs]                       # per half-fiber: (n_rec, 3) in (k, j, i)
    steps_total = sum(n for _, n in done)
    # join: first half reversed without its seed, then the second half with it; drop < 10 points
    i2r = D["i2r"]
    fibers, kept = [], []
    for s in range(S):
        a, c = halves[2 * s], halves[2 * s + 1]
        if len(a) + len(c) - 1 < 10:
            continue
        kept.append(int(sel[s]))
        kji = np.concatenate([a[:0:-1], c])
        ijk = kji[:, ::-1]
        fibers.append(ijk @ i2r[:3, :3].T + i2r[:3, 3])
    row = {int(k): r for r, k in enumerate(sel.tolist())}                 # seed index -> its row here
    seed_voxel = vox_kji[[row[k] for k in kept]] if kept else np.zeros((0, 3), np.int64)
    return fibers, {"seeds": S, "half_fibers": 2 * S, "fiber_steps": steps_total, "fibers": len(fibers), "seed_index": kept,
                    "seed_voxel": seed_voxel, "compile_s": round(compile_s, 1)}
