"""The Metal UKF step (_ukf_metal.py) against the torch tracker, in labelfield's pattern: one-step
fixtures first, then whole fibers, then speed.

    DATA/.venv/bin/python bench/tractography/ukf_metal_check.py [--every 49] [--full]

1. One step, every fixture in DATA/ukf32/steps.npz (ukf32_compare.py made them): the kernel against
   float64 (the reference) and against torch's float32 step on the CPU; and the kernel against itself
   (two launches, bit for bit).
2. Whole fibers, every `every`-th seed: the kernel against float64 on the CPU, and against itself.
   The bar is float32's, which ukf_labels.json showed sits below the noise floor: the kernel should
   look like float32 (about 88 % of fiber ends within 0.06 mm, density r 0.97), not better or worse.
3. Speed: steps per second on this machine, and with --full the whole brain (98,491 seeds),
   compared fiber by fiber with the float64 CUDA run (torch_fibers.npz) through seed points.

Writes results/ukf_metal.json.
"""
import argparse, json, platform, subprocess, time
from pathlib import Path
import numpy as np, torch
from scipy.spatial import cKDTree
import _ukf_torch as U
import _ukf_metal as M
import _fibercmp as C

ap = argparse.ArgumentParser()
ap.add_argument("--every", type=int, default=49)
ap.add_argument("--full", action="store_true")
args = ap.parse_args()
HERE = Path(__file__).resolve().parent
DATA = Path.home() / "tmp/data/tractography"
H = DATA / "ukf/hardi"
D = U.load(str(H / "dwi.nhdr"), str(H / "mask.nrrd"))
off = U.SRAND0_OFFSET                                                    # the binary's seed offset (macOS srand(0))
pts, *_ = U.seeds(D, off)
vox = D["voxel"].numpy()
i2r, dims_ijk = D["i2r"], tuple(int(v) for v in D["dim"][::-1])
chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
res = {"machine": chip or platform.machine(), "torch": torch.__version__, "kernel": "_ukf_metal.ukf_step, one SIMD group per half-fiber"}

# ---------------------------------------------------------------- 1. one step
Z = np.load(DATA / "ukf32/steps.npz")
Q = lambda dt, dev: torch.diag(torch.tensor([0.001] * 3 + [50.0] * 2 + [0.001] * 3 + [50.0] * 2, dtype=dt)).to(dev)

def one_step(fn, Dd, dtype, device):
    t = lambda a: torch.as_tensor(a).to(dtype).to(device)
    out = {k: [] for k in ("x", "state", "P", "dir", "stop", "swap", "swap2", "fa", "mean_signal", "inside")}
    steps = np.unique(Z["step"])
    for s in steps:
        r = Z["step"] == s
        x, sa, Pa, m1, stop, info = fn(Dd, t(Z["x"][r]), t(Z["state"][r]), t(Z["P"][r]), t(Z["old"][r]), Q(dtype, device),
                                       0.02, int(s), 834)
        for k, v in (("x", x), ("state", sa), ("P", Pa), ("dir", m1), ("stop", stop), *info.items()):
            out[k].append(v.cpu().numpy() if v.dtype == torch.bool else v.cpu().double().numpy())
    order = np.argsort(np.concatenate([np.flatnonzero(Z["step"] == s) for s in steps]))
    return {k: np.concatenate(v)[order] for k, v in out.items()}

ref = one_step(U.advance, D, torch.float64, "cpu")
t32 = one_step(U.advance, U.at_dtype(D, torch.float32, "cpu"), torch.float32, "cpu")
Dm = U.at_dtype(D, torch.float32, "mps")
km = one_step(M.advance, Dm, torch.float32, "mps")
km2 = one_step(M.advance, Dm, torch.float32, "mps")
res["one_step"] = {"rows": int(len(Z["step"])), "metal_vs_f64": C.step_diff(km, ref, vox),
                   "torch32_cpu_vs_f64": C.step_diff(t32, ref, vox), "metal_vs_torch32_cpu": C.step_diff(km, t32, vox),
                   "metal_repeat_identical": bool(all(np.array_equal(km[k], km2[k]) for k in km))}
print(json.dumps(res["one_step"]["metal_vs_f64"]), flush=True)

# ---------------------------------------------------------------- 2. whole fibers, a seed subset
sel = np.arange(0, len(pts), args.every)
def run(**kw):
    t0 = time.time()
    f, st = U.track(D, off, select=sel, **kw)
    if kw.get("backend") == "metal":
        torch.mps.synchronize()
    return {int(k): p for k, p in zip(st["seed_index"], f)}, st, time.time() - t0
f64, st64, s64 = run()
fm, stm, sm = run(backend="metal")
fm2, _, sm2 = run(backend="metal")
res["subset"] = {"seeds": int(len(sel)),
                 "f64_cpu": {"fibers": st64["fibers"], "fiber_steps": st64["fiber_steps"], "seconds": round(s64, 1),
                             "steps_per_s": round(st64["fiber_steps"] / s64)},
                 "metal": {"fibers": stm["fibers"], "fiber_steps": stm["fiber_steps"], "seconds": round(sm, 1),
                           "steps_per_s": round(stm["fiber_steps"] / sm), "second_run_seconds": round(sm2, 1)},
                 "metal_vs_f64": C.compare(fm, f64, i2r, dims_ijk),
                 "metal_repeat_identical": bool(fm.keys() == fm2.keys() and all(np.array_equal(fm[k], fm2[k]) for k in fm))}
f32 = json.loads((HERE / "results" / "ukf32.json").read_text())["fibers"]
res["subset"]["for_scale_torch32_vs_f64"] = {k: f32["f32_mps_vs_f64_cpu"][k] for k in
                                            ("within_0.1_mm", "ends_within_0.06_mm_fraction_same_count", "density_r")}
print(json.dumps(res["subset"], indent=1), flush=True)

# ---------------------------------------------------------------- 3. the whole brain
if args.full:
    t0 = time.time()
    f, st = U.track(D, off, backend="metal")
    torch.mps.synchronize()
    s = time.time() - t0
    res["full"] = {"seeds": st["seeds"], "fibers": st["fibers"], "fiber_steps": st["fiber_steps"], "seconds": round(s, 1),
                   "steps_per_s": round(st["fiber_steps"] / s)}
    T = np.load(H / "torch_fibers.npz")
    tp, to = T["points"], T["offsets"]
    seed_ras = pts.numpy()[:, ::-1] @ i2r[:3, :3].T + i2r[:3, 3]
    dd, ii = cKDTree(tp).query(seed_ras)
    fi = np.searchsorted(to, ii, side="right") - 1
    cuda = {k: tp[to[j]:to[j + 1]] for k, (d, j) in enumerate(zip(dd, fi)) if d < 1e-6}
    res["full"]["metal_vs_f64_cuda"] = C.compare({int(k): p for k, p in zip(st["seed_index"], f)}, cuda, i2r, dims_ijk)
    np.savez(DATA / "ukf32/metal_full.npz", points=np.concatenate(f).astype(np.float32),
             offsets=np.r_[0, np.cumsum([len(p) for p in f])], seed_index=np.array(st["seed_index"]))
    print(json.dumps(res["full"], indent=1), flush=True)

(HERE / "results" / "ukf_metal.json").write_text(json.dumps(res, indent=1))
