"""Pure-torch variants of the CPU tracker's step, against the current fast path: speed (one process, one thread,
float32, batch 1,024, every 24th seed of PAT16 as acquired: 4 blocks) and agreement with float64 (every 20th
seed, ukf_cpu_check.py's measures: same point count, all points within 0.1 mm, ends within 0.06 mm, density r).

    uv run bench/ukf_cpu_tune.py [--variants fast,half,half_bmm] [--no-check]

Variants (installed for each step by wrapping advance(), on the exact path - track(fast=False) - whose functions they
replace; spawned workers would not see them):
  fast      ukf.track(fast=True). Since 2026-10-06 that is half_bmm_gather, moved into the package (bit for bit);
            the 2026-10-06 results' "fast" is the step before it (H as a product, Cholesky inverses, 2N columns).
  half      the N gradients once instead of 2N: interp and H stop duplicating their N columns (the other N are the
            same gradients negated - identical values, bit for bit), and the filter's sums over measurements
            are doubled - as the Metal and Triton kernels do. Another rounding: sums over N doubled, not over 2N.
  half_bmm  half, with the three-operand einsums as batched matrix products of weights applied first
  half_bmm_gather   half_bmm, interp's 27 neighbors in one gather and one weighted product
  compiled  half_bmm_gather's advance() through torch.compile (Inductor's C++ backend: needs a C++ compiler),
            dynamic batch size; its compile in "warm_s"
  compiled_split    the same, split into two graphs at interp (sigma as a Python float)
Writes results/ukf_cpu_tune.json.
"""
import argparse, json, math, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np, torch
from tractline import ukf as U
from ukf_cpu_check import compare
from cpu_profile import tracker_input, machine

HERE = Path(__file__).resolve().parent


def interp_half(D, pos):
    """ukf.interp without repeating its N columns."""
    A, vox = D["A"], D["voxel"]
    nk, nj, ni = D["dim"]
    N = D["N"]
    sigma = float(vox.min())
    r = U.cround(pos).long()
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
    return out / wsum[:, None]


_OFFS = {}


def interp_gather(D, pos):
    return _interp_gather(D, pos, D["voxel"].min())


def _interp_gather(D, pos, sigma):
    """interp_half with the 27 neighbors read in one gather and summed as one weighted product: ~10 operations
    instead of 27 rounds of ~15. The weights' sum and the weighted sum in another order (a product, not a loop)."""
    A, vox = D["A"], D["voxel"]
    nk, nj, ni = D["dim"]
    N = D["N"]
    key = (pos.device, pos.dtype)
    if key not in _OFFS:
        o = torch.tensor([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], device=pos.device)
        _OFFS[key] = (o, torch.tensor([nk, nj, ni], device=pos.device), torch.tensor([nj * ni, ni, 1], device=pos.device))
    o, dims, strides = _OFFS[key]
    idx = U.cround(pos).long()[:, None, :] + o[None]                   # (B, 27, 3) in the loop's order
    ok = ((idx >= 0) & (idx < dims)).all(-1)
    d = (idx - pos[:, None, :]) * vox
    w = torch.exp(-(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]) / sigma)
    w = torch.where(ok, w, torch.zeros_like(w))                          # (B, 27)
    lin = (torch.minimum(idx.clamp_min(0), dims - 1) * strides).sum(-1)
    vals = A.reshape(-1, N)[lin].to(pos.dtype)                           # (B, 27, N): one gather
    return (w[:, None, :] @ vals)[:, 0] / (w.sum(1) + 1e-16)[:, None]


def interp_gather_split(D, pos):
    """interp_gather with sigma taken out as a Python float: under torch.compile a graph break there (Tensor.item),
    splitting the step into two compiled graphs."""
    D2 = {**D, "voxel": D["voxel"]}
    sigma = float(D["voxel"].min())
    return _interp_gather(D2, pos, sigma)


def H_half(D, X):
    """ukf._H_gemm without repeating its N columns."""
    N = D["N"]
    g, b = D["g"][:N], D["b"][:N]
    G = -b[None] * torch.stack([g[:, 0] * g[:, 0], g[:, 1] * g[:, 1], g[:, 2] * g[:, 2],
                                2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2], 2 * g[:, 1] * g[:, 2]])
    out = 0
    for o in (0, 5):
        m = U._normalize(X[..., o:o + 3])
        m = torch.where(m[..., :1] < 0, -m, m)
        l1 = X[..., o + 3].clamp_min(U.LMIN)
        l2 = X[..., o + 4].clamp_min(U.LMIN)
        m0, m1, m2 = m[..., 0], m[..., 1], m[..., 2]
        R = torch.stack([torch.stack([m0, m1, m2], -1),
                         torch.stack([m1, m1 * m1 / (1 + m0) - 1, m1 * m2 / (1 + m0)], -1),
                         torch.stack([m2, m1 * m2 / (1 + m0), m2 * m2 / (1 + m0) - 1], -1)], -2)
        L = torch.stack([l1, l2, l2], -1)
        Dm = (R * L[..., None, :]) @ R.transpose(-1, -2) * 1e-6
        d6 = torch.stack([Dm[..., 0, 0], Dm[..., 1, 1], Dm[..., 2, 2], Dm[..., 0, 1], Dm[..., 0, 2], Dm[..., 1, 2]], -1)
        out = out + torch.exp(d6 @ G) * 0.5
    return out


def _sigma(x, P):
    n = 10
    W = torch.full((2 * n + 1,), 0.5 / (n + U.KAPPA), dtype=x.dtype, device=x.device)
    W[0] = U.KAPPA / (n + U.KAPPA)
    L, _ = torch.linalg.cholesky_ex(P)
    dX = math.sqrt(n + U.KAPPA) * L.transpose(1, 2)
    return W, U.F(torch.cat([x[:, None], x[:, None] + dX, x[:, None] - dX], 1))


def _spd_inv(A):
    return torch.cholesky_inverse(torch.linalg.cholesky_ex(A)[0])


def ukf_step_half(D, x, P, z, Q, Rs):
    """ukf.ukf_step (fast) on N measurements: z (B, N); every sum over the 2N measurements doubled."""
    W, X = _sigma(x, P)
    xh = torch.einsum("s,bsk->bk", W, X)
    Xt = X - xh[:, None]
    Pm = torch.einsum("bsi,s,bsj->bij", Xt, W, Xt) + Q
    Yk = _spd_inv(Pm)
    yh = torch.einsum("bij,bj->bi", Yk, xh)
    Z = H_half(D, X)
    zh = torch.einsum("s,bsn->bn", W, Z)
    Pxz = torch.einsum("bsi,s,bsn->bin", Xt, W, Z - zh[:, None])
    Ht = Yk @ Pxz
    r = 1.0 / Rs
    I = 2 * ((Ht * r) @ Ht.transpose(1, 2))
    i = 2 * torch.einsum("bin,bn->bi", Ht * r, (z - zh) + torch.einsum("bin,bi->bn", Pxz, yh))
    Pn = _spd_inv(Yk + I)
    return torch.einsum("bij,bj->bi", Pn, i + yh), Pn


def ukf_step_half_bmm(D, x, P, z, Q, Rs):
    """ukf_step_half with the weights applied first and every contraction a batched matrix product."""
    W, X = _sigma(x, P)
    Wc = W[None, :, None]
    xh = (Wc * X).sum(1)
    Xt = X - xh[:, None]
    XwT = (Xt * Wc).transpose(1, 2)                                     # (B, 10, 21)
    Pm = XwT @ Xt + Q
    Yk = _spd_inv(Pm)
    yh = (Yk @ xh[..., None])[..., 0]
    Z = H_half(D, X)
    zh = (Wc * Z).sum(1)
    Pxz = XwT @ (Z - zh[:, None])                                       # (B, 10, N)
    Ht = Yk @ Pxz
    r = 1.0 / Rs
    Htr = Ht * r
    I = 2 * (Htr @ Ht.transpose(1, 2))
    v = (z - zh) + (Pxz.transpose(1, 2) @ yh[..., None])[..., 0]
    i = 2 * (Htr @ v[..., None])[..., 0]
    Pn = _spd_inv(Yk + I)
    return (Pn @ (i + yh)[..., None])[..., 0], Pn


VARIANTS = {"fast": {}, "half": dict(interp=interp_half, H=H_half, ukf_step=ukf_step_half),
            "half_bmm": dict(interp=interp_half, H=H_half, ukf_step=ukf_step_half_bmm),
            "half_bmm_gather": dict(interp=interp_gather, H=H_half, ukf_step=ukf_step_half_bmm),
            "compiled": dict(interp=interp_gather, H=H_half, ukf_step=ukf_step_half_bmm, _compile=True),
            "compiled_split": dict(interp=interp_gather_split, H=H_half, ukf_step=ukf_step_half_bmm, _compile=True)}


@contextmanager
def variant(name):
    """The variant's functions in place for each step only (advance wrapped): the seeds' fit also calls interp,
    and needs its 2N columns."""
    swap = {k: v for k, v in VARIANTS[name].items() if not k.startswith("_")}
    orig = {k: getattr(U, k) for k in swap}
    advance = U.advance
    run = torch.compile(advance, dynamic=True) if VARIANTS[name].get("_compile") else advance

    def stepped(*a, **k):
        for n, f in swap.items():
            setattr(U, n, f)
        try:
            return run(*a, **k)
        finally:
            for n, f in orig.items():
                setattr(U, n, f)

    U.advance = stepped
    try:
        yield
    finally:
        U.advance = advance


def main(names, check=True):
    _, D = tracker_input()
    nseeds = len(U.seeds(D)[0])
    out = {"machine": machine(), "speed": {}, "against_float64": {}}
    torch.set_num_threads(1)
    sel = np.arange(0, nseeds, 24)[:2048]
    for name in names:
        with variant(name):
            t0 = time.perf_counter()
            U.track(D, select=sel[:64], dtype=torch.float32, fast=name == "fast", batch=1024)  # warm (and compile)
            warm = time.perf_counter() - t0
            t0 = time.perf_counter()
            _, st = U.track(D, select=sel, dtype=torch.float32, fast=name == "fast", batch=1024)
            wall = time.perf_counter() - t0
        out["speed"][name] = {"warm_s": round(warm, 1), "wall_s": round(wall, 2), "fiber_steps": int(st["fiber_steps"]), "steps_per_s": round(st["fiber_steps"] / wall)}
        print(name, out["speed"][name], flush=True)
    if check:
        torch.set_num_threads(min(8, torch.get_num_threads() or 8))
        csel = np.arange(0, nseeds, 20)
        ref, refst = U.track(D, select=csel)                                # float64, the binary's arithmetic
        torch.set_num_threads(1)
        for name in names:
            with variant(name):
                f, st = U.track(D, select=csel, dtype=torch.float32, fast=name == "fast", batch=1024)
            out["against_float64"][name] = compare(ref, refst, f, st)
            print(name, out["against_float64"][name], flush=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--variants", default="fast,half,half_bmm"); ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--tag", default="m2")
    a = ap.parse_args()
    res = main(a.variants.split(","), not a.no_check)
    (HERE / f"results/ukf_cpu_tune_{a.tag}.json").write_text(json.dumps(res, indent=1))
