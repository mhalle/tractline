"""What float32 does to UKF tractography, against the float64 tracker that reproduces the Slicer
binary (ukf_compare.json). An Apple GPU has no float64, so float32 is the only GPU arithmetic on a
Mac; this measures its cost before any Metal kernel is written.

    DATA/.venv/bin/python bench/tractography/ukf32_compare.py [--fixture-every 197] [--every 49]

1. One-step fixtures. A float64 run over every `fixture-every`-th seed keeps the inputs of a sample
   of steps (position, state, covariance, previous direction, step number): DATA/ukf32/steps.npz,
   the fixture a Metal kernel will be tested against. Each step is then taken once more in float64
   (CPU), float32 (CPU) and float32 (MPS) from identical inputs, and the outputs compared: state,
   covariance, new position, FA, and the decisions (both swaps, inside the mask, stop).
2. Whole fibers. Every `every`-th seed tracked in float64 (CPU), float32 (CPU) and float32 (MPS),
   the same seeds made in float64 for all three. Per seed: kept by both or one, the same point count,
   the largest point distance, the ends' distances (albula-diffusion's GPU-vs-CPU measure: "fiber
   ends within 0.06 mm"), and streamline-density maps on the DWI grid (Pearson r).
3. The floor for scale: the same float64 fibers against the full float64 run on CUDA
   (torch_fibers.npz, modal_ukf_track.py), matched through each fiber's seed point: two float64
   implementations whose only differences are rounding order.

Writes results/ukf32.json.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from scipy.spatial import cKDTree
import _ukf_torch as U
import _fibercmp as C

ap = argparse.ArgumentParser()
ap.add_argument("--fixture-every", type=int, default=197)
ap.add_argument("--every", type=int, default=49)
ap.add_argument("--per-step", type=int, default=4, help="fixture rows kept per step")
args = ap.parse_args()
HERE = Path(__file__).resolve().parent
DATA = Path.home() / "tmp/data/tractography"
H, OUT = DATA / "ukf/hardi", DATA / "ukf32"
OUT.mkdir(exist_ok=True)
F64, F32 = torch.float64, torch.float32
DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])

D = U.load(str(H / "dwi.nhdr"), str(H / "mask.nrrd"))
off = np.array([-4158, -2201, -2855], float); off = off / np.linalg.norm(off) * 0.5    # macOS srand(0)
pts, *_ = U.seeds(D, off)
vox = D["voxel"].numpy()
q = C.q
res = {"data": "Stanford HARDI, ORG settings (ukf_bench.py)", "torch": torch.__version__, "devices": DEVICES}

# ---------------------------------------------------------------- 1. one-step fixtures
fix = OUT / "steps.npz"
if not fix.exists():
    rng = np.random.default_rng(20261001)
    rows = []
    def keep(step, idx, xa, sa, Pa, oa):
        k = rng.choice(len(xa), min(args.per_step, len(xa)), replace=False)
        rows.append((np.full(len(k), step), xa[k].numpy(), sa[k].numpy(), Pa[k].numpy(), oa[k].numpy()))
    U.track(D, off, select=np.arange(0, len(pts), args.fixture_every), capture=keep)
    st, x, s, P, o = (np.concatenate(c) for c in zip(*rows))
    np.savez(fix, step=st, x=x, state=s, P=P, old=o, seed_every=args.fixture_every)
Z = np.load(fix)
res["fixture"] = {"file": str(fix), "rows": int(len(Z["step"])), "seed_every": int(Z["seed_every"]),
                  "steps_covered": [int(Z["step"].min()), int(Z["step"].max())]}

def one_step(dtype, device):
    Dd = U.at_dtype(D, dtype, device)
    t = lambda a: torch.as_tensor(a).to(dtype).to(device)
    Q = torch.diag(torch.tensor([0.001] * 3 + [50.0] * 2 + [0.001] * 3 + [50.0] * 2, dtype=dtype)).to(device)
    out = {"x": [], "state": [], "P": [], "dir": [], "stop": [], "swap": [], "swap2": [], "fa": [], "mean_signal": [], "inside": []}
    for s in np.unique(Z["step"]):                                    # the step number enters only through max_steps
        r = Z["step"] == s
        x, sa, Pa, m1, stop, info = U.advance(Dd, t(Z["x"][r]), t(Z["state"][r]), t(Z["P"][r]), t(Z["old"][r]), Q, 0.02,
                                              int(s), 834)
        for k, v in (("x", x), ("state", sa), ("P", Pa), ("dir", m1), ("stop", stop), *info.items()):
            out[k].append(v.cpu().double().numpy() if v.dtype != torch.bool else v.cpu().numpy())
    order = np.argsort(np.concatenate([np.flatnonzero(Z["step"] == s) for s in np.unique(Z["step"])]))  # back to row order
    return {k: np.concatenate(v)[order] for k, v in out.items()}

ref = one_step(F64, "cpu")
again = one_step(F64, "cpu")
res["one_step"] = {"f64_repeat_identical": bool(all(np.array_equal(ref[k], again[k]) for k in ref))}

steps = {}
for dev in DEVICES:
    steps[dev] = one_step(F32, dev)
    res["one_step"][f"f32_{dev}_vs_f64"] = C.step_diff(steps[dev], ref, vox)
if "mps" in steps:
    res["one_step"]["f32_mps_vs_f32_cpu"] = C.step_diff(steps["mps"], steps["cpu"], vox)

# ---------------------------------------------------------------- 2. whole fibers
sel = np.arange(0, len(pts), args.every)
runs = {}
for name, dtype, dev in [("f64_cpu", F64, "cpu")] + [(f"f32_{d}", F32, d) for d in DEVICES]:
    t0 = time.time()
    f, st = U.track(D, off, select=sel, dtype=dtype, device=dev)
    s = time.time() - t0
    runs[name] = {int(k): p for k, p in zip(st["seed_index"], f)}
    res.setdefault("runs", {})[name] = {"seeds": int(len(sel)), "fibers": st["fibers"], "fiber_steps": st["fiber_steps"],
                                        "seconds": round(s, 1), "steps_per_s": round(st["fiber_steps"] / s)}

i2r = D["i2r"]
dims_ijk = tuple(int(v) for v in D["dim"][::-1])
compare = lambda a, b: C.compare(a, b, i2r, dims_ijk)

res["fibers"] = {f"{k}_vs_f64_cpu": compare(runs[k], runs["f64_cpu"]) for k in runs if k != "f64_cpu"}
if "f32_mps" in runs:
    res["fibers"]["f32_mps_vs_f32_cpu"] = compare(runs["f32_mps"], runs["f32_cpu"])

# ---------------------------------------------------------------- 3. the float64 floor: CPU vs CUDA
T = np.load(H / "torch_fibers.npz")
tp, to = T["points"], T["offsets"]
seed_ras = pts[sel].numpy()[:, ::-1] @ i2r[:3, :3].T + i2r[:3, 3]
dd, ii = cKDTree(tp).query(seed_ras)
fi = np.searchsorted(to, ii, side="right") - 1
cuda = {int(sel[k]): tp[to[f]:to[f + 1]] for k, (d, f) in enumerate(zip(dd, fi)) if d < 1e-6}
res["fibers"]["f64_cuda_vs_f64_cpu"] = compare({k: v for k, v in cuda.items()}, {k: v for k, v in runs["f64_cpu"].items()})
res["fibers"]["f64_cuda_vs_f64_cpu"]["note"] = "CUDA fibers matched by seed point; seeds whose CUDA fiber was dropped (< 10 points) count as only_second"

print(json.dumps(res, indent=1))
(HERE / "results" / "ukf32.json").write_text(json.dumps(res, indent=1))
