"""Timing of one batched Gauss-Newton iteration's pieces for the motion + eddy design (docs/gpu-motion-eddy-design.md,
section 3), on a device: the moved sampling grid, grid_sample with its position gradient over the mask's voxels, the
residual, the moments sum g_a g_b m_i m_j (6 x 55 per volume) and sum g_a r m_i (3 x 10), the LOO prediction matmul,
and the final 3D cubic B-spline resample of every volume over the whole box (prefilter + 64-tap gather, chunked).

    python bench_pieces.py --device mps|cpu [--reps 5] [--mask 0.3]
"""
import argparse, json, math, time
import torch
import torch.nn.functional as F
from pathlib import Path
from tractline.data import DATA                                   # $TRACTOGRAPHY_DATA
OUT = Path(__file__).resolve().parent.parent / "results/motion_eddy"
SYNTH = DATA / "derived/motion_eddy_synth"                     # proto.py --save-synth; eddy runs beside it

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="mps")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--mask", type=float, default=0.3)
ap.add_argument("--threads", type=int, default=8)
args = ap.parse_args()
torch.set_num_threads(args.threads)
dev = torch.device(args.device)
dt = torch.float32


def sync():
    if dev.type == "mps":
        torch.mps.synchronize()
    elif dev.type == "cuda":
        torch.cuda.synchronize()


def timeit(f, reps):
    f(); sync()                                   # warm-up (kernel compile, allocator)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); sync(); ts.append(time.perf_counter() - t0)
    ts.sort()
    return round(ts[len(ts) // 2] * 1000, 1)      # median ms


def monomials(x):                                 # (N, 3) centered, scaled coordinates -> (N, 10): 1, x, y, z, x2, y2, z2, xy, xz, yz
    a, b, c = x.unbind(-1)
    one = torch.ones_like(a)
    return torch.stack([one, a, b, c, a * a, b * b, c * c, a * b, a * c, b * c], -1)


def prefilter_axis(img, ax):                      # cubic B-spline coefficients along axis ax of (G, X, Y, Z), mirrored edges
    n = img.shape[1 + ax]
    M = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        M[i, i] = 4 / 6
        for j in (i - 1, i + 1):
            jj = -j if j < 0 else (2 * (n - 1) - j if j > n - 1 else j)
            M[i, jj] += 1 / 6
    Minv = torch.linalg.inv(M).to(img.device, img.dtype)
    return torch.movedim(torch.movedim(img, 1 + ax, -1) @ Minv.T, -1, 1 + ax)


def cubic_sample(coef, p):                        # coef (X, Y, Z) one volume, p (N, 3) voxel positions -> (N,)
    X, Y, Z = coef.shape
    i0 = torch.floor(p); t = p - i0; i0 = i0.long()
    w = [((1 - t) ** 3 / 6), ((3 * t ** 3 - 6 * t ** 2 + 4) / 6), ((-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6), (t ** 3 / 6)]
    flat = coef.reshape(-1)
    out = torch.zeros(p.shape[0], dtype=coef.dtype, device=coef.device)
    for a in range(4):
        ia = (i0[:, 0] + a - 1).clamp(0, X - 1)
        for b in range(4):
            ib = (i0[:, 1] + b - 1).clamp(0, Y - 1)
            wab = w[a][:, 0] * w[b][:, 1]
            base = (ia * Y + ib) * Z
            for c in range(4):
                ic = (i0[:, 2] + c - 1).clamp(0, Z - 1)
                out = out + wab * w[c][:, 2] * flat[base + ic]
    return out


results = {"device": args.device, "mask_fraction": args.mask, "reps": args.reps, "sizes": {}}
for (X, Y, Z, G) in [(96, 96, 50, 100), (104, 104, 72, 105)]:
    torch.manual_seed(0)
    img = F.avg_pool3d(torch.rand(G, 1, X, Y, Z, device=dev, dtype=dt), 5, 1, 2)[:, 0]          # smooth-ish volumes
    N = X * Y * Z
    Nm = int(args.mask * N)
    idx = torch.randperm(N, device=dev)[:Nm].sort().values
    grid = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev, dtype=dt) for n in (X, Y, Z)], indexing="ij"), -1).reshape(-1, 3)[idx]
    sizes = torch.tensor([X, Y, Z], device=dev, dtype=dt)
    center = (sizes - 1) / 2
    xs = (grid - center) / center                                                                # (Nm, 3) in [-1, 1]
    m = monomials(xs)                                                                            # (Nm, 10)
    mm = (m[:, :, None] * m[:, None, :])[:, *torch.triu_indices(10, 10)]                         # (Nm, 55)
    R = torch.eye(3, device=dev, dtype=dt).expand(G, 3, 3) + 0.01 * torch.randn(G, 3, 3, device=dev, dtype=dt)
    t = 0.5 * torch.randn(G, 3, device=dev, dtype=dt)
    ec = 0.1 * torch.randn(G, 10, device=dev, dtype=dt)                                          # eddy displacement, voxels, per monomial
    pred = torch.rand(G, Nm, device=dev, dtype=dt)
    W = torch.rand(G, G, device=dev, dtype=dt)
    vol = img[:, None]

    def positions():
        p = torch.einsum("gij,nj->gni", R, grid - center) + center + t[:, None]
        disp = ec @ m.T                                                                          # (G, Nm) along PE (axis 1)
        return torch.cat([p[..., :1], (p[..., 1] + disp)[..., None], p[..., 2:]], -1)

    def sample_and_grad():
        p = positions().requires_grad_(True)
        g = (2 * p / (sizes - 1) - 1)[..., [2, 1, 0]][:, None, None]                           # (G, 1, 1, Nm, 3)
        S = F.grid_sample(vol, g, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0, 0, 0]
        Gp, = torch.autograd.grad(S.sum(), p)
        return S.detach(), Gp

    def iteration():
        S, Gp = sample_and_grad()
        r = S - pred
        gg = torch.stack([Gp[..., a] * Gp[..., b] for a, b in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))], -1)  # (G, Nm, 6)
        H = torch.bmm(gg.transpose(1, 2), mm.expand(G, -1, -1))                                 # (G, 6, 55)
        gr = Gp * r[..., None]
        b = torch.bmm(gr.transpose(1, 2), m.expand(G, -1, -1))                                   # (G, 3, 10)
        return H, b

    def cost_only():
        with torch.no_grad():
            p = positions()
            g = (2 * p / (sizes - 1) - 1)[..., [2, 1, 0]][:, None, None]
            S = F.grid_sample(vol, g, mode="bilinear", padding_mode="zeros", align_corners=True)[:, 0, 0, 0]
            return ((S - pred) ** 2).sum(1)

    def predict():
        return pred.T @ W                                                                       # (Nm, G) @ (G, G)

    full = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev, dtype=dt) for n in (X, Y, Z)], indexing="ij"), -1).reshape(-1, 3)

    def final_resample():
        out = torch.empty(G, N, device=dev, dtype=dt)
        for g0 in range(0, G, 8):
            c = img[g0:g0 + 8]
            for ax in range(3):
                c = prefilter_axis(c, ax)
            for k in range(c.shape[0]):
                p = full @ R[g0 + k].T + t[g0 + k]
                out[g0 + k] = cubic_sample(c[k], p)
        return out

    with torch.no_grad():
        r_iter = None
    row = {"G": G, "box_voxels": N, "mask_voxels": Nm,
           "gn_iteration_ms": timeit(iteration, args.reps),
           "sample_and_grad_ms": timeit(sample_and_grad, args.reps),
           "cost_eval_ms": timeit(cost_only, args.reps),
           "loo_predict_ms": timeit(predict, args.reps),
           "final_cubic_resample_ms": timeit(final_resample, max(1, args.reps // 2))}
    if dev.type == "mps":
        row["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 2 ** 20)
    results["sizes"][f"{X}x{Y}x{Z}x{G}"] = row
    print(json.dumps({f"{X}x{Y}x{Z}x{G}": row}), flush=True)
    del img, vol
    if dev.type == "mps":
        torch.mps.empty_cache()
json.dump(results, open(OUT / f"bench_pieces_{args.device}.json", "w"), indent=1)
