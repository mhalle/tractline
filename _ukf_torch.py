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


# ------------------------------------------------------------------------------- data

def load(nhdr: str, mask_path: str, device="cpu", data=None) -> dict:
    """The DWI as the binary normalizes it. `data`, when given, replaces the file's voxels (same
    shape, (i, j, k, G)): a bootstrap replicate keeps the header, gradients and mask."""
    import nrrd
    raw, h = nrrd.read(nhdr)                                  # (i, j, k, G), Fortran index order
    if data is not None:
        assert data.shape == raw.shape, (data.shape, raw.shape)
        raw = data
    keys = sorted(k for k in h if k.startswith("DWMRI_gradient_"))
    gtxt = [h[k] for k in keys]
    bmax_int = int(re.match(r"\s*(-?\d+)", h["DWMRI_b-value"]).group(1))
    b_hdr = float(h["DWMRI_b-value"])
    g32 = np.array([[np.float32(v) for v in s.split()] for s in gtxt], np.float32)
    mag = np.sqrt(g32[:, 0] * g32[:, 0] + g32[:, 1] * g32[:, 1] + g32[:, 2] * g32[:, 2], dtype=np.float32)
    nonzero = (np.float32(bmax_int) * mag * mag) > 50
    data = raw.astype(np.float32)
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
    dirs = np.asarray(h["space directions"], dtype=object)
    sd = np.array([np.asarray(dirs[a], float) for a in range(3)])        # rows: axes i, j, k
    origin = np.asarray(h["space origin"], float).copy()
    mf = np.asarray(h.get("measurement frame", np.eye(3)), float).copy()  # mf[vector][component]
    space = h.get("space", "")
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

    m, _ = nrrd.read(mask_path)
    mask = np.ascontiguousarray(np.asarray(m).transpose(2, 1, 0)).astype(np.uint8).view(np.int8)  # signed char
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


def mask_at(D: dict, pos: torch.Tensor) -> torch.Tensor:
    nk, nj, ni = D["dim"]
    r = cround(pos).long()
    ok = (r[:, 0] >= 0) & (r[:, 0] < nk) & (r[:, 1] >= 0) & (r[:, 1] < nj) & (r[:, 2] >= 0) & (r[:, 2] < ni)
    v = D["mask"][r[:, 0].clamp(0, nk - 1), r[:, 1].clamp(0, nj - 1), r[:, 2].clamp(0, ni - 1)]
    return torch.where(ok, v.to(torch.int32), torch.zeros_like(v, dtype=torch.int32))


# ------------------------------------------------------------------------------- model

def _normalize(v):
    return v / torch.sqrt((v * v).sum(-1, keepdim=True))


def H(D: dict, X: torch.Tensor) -> torch.Tensor:
    """filter_Simple2T::H: X (..., 10) -> (..., 2N)."""
    g, b = D["g"], D["b"]
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
    return out


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

def seeds(D: dict, offset_kji, seeding_threshold=0.1):
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
    d = torch.linalg.lstsq(B.expand(len(S), -1, -1).cpu(), logS[..., None].cpu(), driver="gelsd").solution[..., 0].to(dev)
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
    z = interp(D, xa)
    sa, Pa = ukf_step(D, sa, Pa, z, Q, Rs)
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
    mean_sig = H(D, sa[:, None])[:, 0].mean(1)
    inside = mask_at(D, xa) > 0
    stop = ~inside | (mean_sig < stopping_threshold) | (fa < stopping_fa) | (step > max_steps)
    return xa, sa, Pa, m1, stop, {"swap": sw, "swap2": sw2, "fa": fa, "mean_signal": mean_sig, "inside": inside}


def track(D: dict, offset_kji, seeding_threshold=0.1, stopping_fa=0.08, stopping_threshold=0.06,
          step_length=0.3, record_length=1.8, max_half_length=250.0, Qm=0.001, Ql=50.0, Rs=0.02,
          batch=50_000, progress=None, select=None, dtype=torch.float64, device=None, capture=None,
          seed_points=None, backend="torch"):
    """All fibers. Returns (points list of (n, 3) RAS arrays in seed order, stats). With `select`
    (indices into the seed list), only those seeds are tracked; stats["seed_index"] then says which
    seed each returned fiber came from.

    `dtype` is the tracking arithmetic: float64 is the binary's; float32 is what an Apple GPU can do.
    Seeds and their initial states are computed in float64 either way, so two dtypes start from the
    same half-fibers and differ only in the steps. `device` is where the steps run (default: where D
    is); seeds are made where D is, so a float64 D on the CPU can seed an MPS run, which has no
    float64. `capture(step, index, xa, sa, Pa, oa)`, when given, sees every step's inputs before the
    step, `index` being the half-fibers' (for one-step fixtures). `seed_points` (k, j, i), when given,
    replaces the seeds: every point is tracked from its state on this data (seed_states), none
    rejected, and `select` indexes them. `backend` "metal" takes each step with _ukf_metal's kernel
    (float32, device "mps"), "triton" with _ukf_triton's and "triton_block" with _ukf_triton_block's
    (float32, "cuda"); the loop, the recording and the joining stay these."""
    step_fn = advance
    if backend == "metal":
        import _ukf_metal
        dtype, device, step_fn = torch.float32, "mps", _ukf_metal.advance
    elif backend == "triton":
        import _ukf_triton
        dtype, device, step_fn = torch.float32, "cuda", _ukf_triton.advance
    elif backend == "triton_block":
        import _ukf_triton_block
        dtype, device, step_fn = torch.float32, "cuda", _ukf_triton_block.advance
    if seed_points is None:
        pts, fwd, inv, e1, fa0 = seeds(D, offset_kji, seeding_threshold)
    else:
        pts = torch.as_tensor(seed_points, dtype=torch.float64, device=D["A"].device)
        fwd, inv, e1, fa0, _ = seed_states(D, pts, seeding_threshold)
    sel = torch.arange(len(pts), device=pts.device) if select is None else torch.as_tensor(select, device=pts.device)
    D = at_dtype(D, dtype, device)
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
    halves = []                                                      # per half-fiber: (n_rec, 3) in (k, j, i)
    steps_total = 0
    for s0 in range(0, len(x0), batch):
        x = x0[s0:s0 + batch].clone(); state = st0[s0:s0 + batch].clone(); old = dir0[s0:s0 + batch].clone()
        nb = len(x)
        P = (P0 * torch.eye(10, dtype=dtype, device=dev)).expand(nb, 10, 10).clone()
        rec = torch.full((nb, max_steps // spr + 2, 3), float("nan"), dtype=dtype, device=dev)
        rec[:, 0] = x
        nrec = torch.ones(nb, dtype=torch.long, device=dev)
        alive = torch.arange(nb, device=dev)
        step = 0
        while len(alive):
            step += 1
            steps_total += len(alive)
            xa, sa, Pa, oa = x[alive], state[alive], P[alive], old[alive]
            if capture:
                capture(step, s0 + alive, xa, sa, Pa, oa)
            xa, sa, Pa, m1, stop, _ = step_fn(D, xa, sa, Pa, oa, Q, Rs, step, max_steps, step_length,
                                              stopping_fa, stopping_threshold)
            go = ~stop
            if (step + 1) % spr == 0:
                ia = alive[go]
                rec[ia, nrec[ia]] = xa[go]
                nrec[ia] += 1
            x[alive], state[alive], P[alive], old[alive] = xa, sa, Pa, m1
            alive = alive[go]
            if progress and step % 50 == 0:
                progress(s0, step, len(alive))
        r = rec.cpu().numpy(); nr = nrec.cpu().numpy()
        halves += [r[k, :nr[k]] for k in range(nb)]
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
    return fibers, {"seeds": S, "half_fibers": 2 * S, "fiber_steps": steps_total, "fibers": len(fibers), "seed_index": kept}
