"""Faster variants of bench_pieces.py's two slow pieces, checked against the plain versions:
  - the moments: g g' products built as (G, 6, Nm) contiguous, one matmul against (Nm, 55) (no expand, no transpose);
  - the final cubic B-spline resample as 8 trilinear fetches of the prefiltered volume (Sigg & Hadwiger 2005, GPU Gems 2
    ch. 20: a cubic B-spline is a sum of 2 linear fetches per axis at shifted positions with folded weights), batched over
    volumes through grid_sample.

    python bench_fast.py --device mps|cpu [--check]
"""
import argparse, json, time
import torch
import torch.nn.functional as F
from pathlib import Path
from tractline.data import DATA                                   # $TRACTOGRAPHY_DATA
OUT = Path(__file__).resolve().parent.parent / "results/motion_eddy"
SYNTH = DATA / "derived/motion_eddy_synth"                     # proto.py --save-synth; eddy runs beside it

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="mps")
ap.add_argument("--reps", type=int, default=5)
ap.add_argument("--check", action="store_true")
args = ap.parse_args()
dev = torch.device(args.device)
dt = torch.float32
torch.set_num_threads(8)


def sync():
    if dev.type == "mps":
        torch.mps.synchronize()


def timeit(f, reps):
    f(); sync(); ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); sync(); ts.append(time.perf_counter() - t0)
    return round(sorted(ts)[len(ts) // 2] * 1000, 1)


def prefilter_axis(img, ax):
    n = img.shape[1 + ax]
    M = torch.zeros(n, n, dtype=torch.float64)
    for i in range(n):
        M[i, i] = 4 / 6
        for j in (i - 1, i + 1):
            jj = -j if j < 0 else (2 * (n - 1) - j if j > n - 1 else j)
            M[i, jj] += 1 / 6
    Minv = torch.linalg.inv(M).to(img.device, img.dtype)
    return torch.movedim(torch.movedim(img, 1 + ax, -1) @ Minv.T, -1, 1 + ax)


def prefilter(img):
    for ax in range(3):
        img = prefilter_axis(img, ax)
    return img


def cubic_gather(coef, p):                     # reference: 64 taps, edge-clamped; coef (X, Y, Z), p (N, 3)
    X, Y, Z = coef.shape
    i0 = torch.floor(p); t = p - i0; i0 = i0.long()
    w = [((1 - t) ** 3 / 6), ((3 * t ** 3 - 6 * t ** 2 + 4) / 6), ((-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6), (t ** 3 / 6)]
    flat = coef.reshape(-1); out = 0
    for a in range(4):
        ia = (i0[:, 0] + a - 1).clamp(0, X - 1)
        for b in range(4):
            ib = (i0[:, 1] + b - 1).clamp(0, Y - 1)
            for c in range(4):
                ic = (i0[:, 2] + c - 1).clamp(0, Z - 1)
                out = out + w[a][:, 0] * w[b][:, 1] * w[c][:, 2] * flat[(ia * Y + ib) * Z + ic]
    return out


def cubic_8fetch(coef, p):
    """coef (G, X, Y, Z), p (G, N, 3) voxel positions -> (G, N): per axis, g0 = w0 + w1, g1 = w2 + w3, fetched at
    i - 1 + w1 / g0 and i + 1 + w3 / g1 (linear interpolation does the rest); border padding = edge clamp."""
    G, X, Y, Z = coef.shape
    sizes = torch.tensor([X, Y, Z], dtype=p.dtype, device=p.device)
    i0 = torch.floor(p); t = p - i0
    w0, w1, w2, w3 = (1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6
    g0, g1 = w0 + w1, w2 + w3
    h0, h1 = i0 - 1 + w1 / g0, i0 + 1 + w3 / g1                     # (G, N, 3) each
    out = 0
    vol = coef[:, None]
    for sx in (0, 1):
        for sy in (0, 1):
            for sz in (0, 1):
                q = torch.stack([(h0, h1)[sx][..., 0], (h0, h1)[sy][..., 1], (h0, h1)[sz][..., 2]], -1)
                wq = (g0, g1)[sx][..., 0] * (g0, g1)[sy][..., 1] * (g0, g1)[sz][..., 2]
                grid = (2 * q / (sizes - 1) - 1)[..., [2, 1, 0]][:, None, None]
                s = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border", align_corners=True)[:, 0, 0, 0]
                out = out + wq * s
    return out


if args.check:
    torch.manual_seed(1)
    c = torch.rand(1, 20, 22, 18, dtype=torch.float64)
    p = torch.rand(5000, 3, dtype=torch.float64) * torch.tensor([19, 21, 17.0]) * 0.9 + 0.5
    a = cubic_gather(c[0], p); b = cubic_8fetch(c, p[None])[0]
    print("8-fetch vs 64-tap cubic, interior points, max |diff|:", float((a - b).abs().max()))

out = {"device": args.device, "sizes": {}}
for (X, Y, Z, G) in [(96, 96, 50, 100), (104, 104, 72, 105)]:
    torch.manual_seed(0)
    img = F.avg_pool3d(torch.rand(G, 1, X, Y, Z, device=dev, dtype=dt), 5, 1, 2)[:, 0]
    N = X * Y * Z; Nm = int(0.3 * N)
    idx = torch.randperm(N, device=dev)[:Nm].sort().values
    full = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev, dtype=dt) for n in (X, Y, Z)], indexing="ij"), -1).reshape(-1, 3)
    center = (torch.tensor([X, Y, Z], device=dev, dtype=dt) - 1) / 2
    m = torch.rand(Nm, 10, device=dev, dtype=dt)
    iu = torch.triu_indices(10, 10, device=dev)
    mm = (m[:, iu[0]] * m[:, iu[1]]).contiguous()                  # (Nm, 55)
    Gp = torch.randn(G, Nm, 3, device=dev, dtype=dt); r = torch.randn(G, Nm, device=dev, dtype=dt)
    pairs = torch.tensor([[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]], device=dev)

    def moments_v2():
        Gt = Gp.permute(0, 2, 1)                                        # (G, 3, Nm)
        gg = (Gt[:, pairs[:, 0]] * Gt[:, pairs[:, 1]]).contiguous()     # (G, 6, Nm)
        H = gg @ mm                                                     # (G, 6, 55)
        b = (Gt * r[:, None]) @ m                                       # (G, 3, 10)
        return H, b

    R = torch.eye(3, device=dev, dtype=dt).expand(G, 3, 3) + 0.01 * torch.randn(G, 3, 3, device=dev, dtype=dt)
    t = 0.5 * torch.randn(G, 3, device=dev, dtype=dt)

    def resample_fast(chunk=16):
        res = torch.empty(G, N, device=dev, dtype=dt)
        for g0 in range(0, G, chunk):
            c = prefilter(img[g0:g0 + chunk])
            p = torch.einsum("gij,nj->gni", R[g0:g0 + chunk], full - center) + center + t[g0:g0 + chunk, None]
            res[g0:g0 + chunk] = cubic_8fetch(c, p)
        return res

    row = {"moments_v2_ms": timeit(moments_v2, args.reps), "final_cubic_8fetch_ms": timeit(resample_fast, 3)}
    if dev.type == "mps":
        row["mps_driver_mb"] = round(torch.mps.driver_allocated_memory() / 2 ** 20)
    out["sizes"][f"{X}x{Y}x{Z}x{G}"] = row
    print(json.dumps({f"{X}x{Y}x{Z}x{G}": row}), flush=True)
    img = Gp = None                                                 # freed before the next size
    if dev.type == "mps":
        torch.mps.empty_cache()
json.dump(out, open(OUT / f"bench_fast_{args.device}.json", "w"), indent=1)
