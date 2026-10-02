"""Ron Kikinis's UKF port (rkikinis/albula-diffusion ukf.ts, ac1644e) against the C++ program, through
our torch re-implementation, which reproduces the C++ program's fibers (ukf_compare.json).

    DATA/.venv/bin/python bench/tractography/albula_compare.py [--every 49]

Inputs identical on both sides: UKF's own normalized signal (dwi_normalize.cc's float32 baseline),
gradients in the voxel frame, the NRRD mask, and the C++ program's seed points (its voxels plus the
srand(0) offset) for every `every`-th accepted seed. The port runs with freeWater false and the
ORG settings (seeding 0.1, stop FA 0.08, stop signal 0.06, record 1.8 mm); the torch tracker on the
same seeds stands in for the C++ program, seed for seed. Per seed: whether each side kept a fiber,
the point counts, and the largest distance between corresponding points (either orientation).

Writes DATA/albula/ (inputs and the port's output) and results/albula_ukf.json.
"""
import argparse, json, subprocess, time
from pathlib import Path
import numpy as np, torch
import _ukf_torch as U

ap = argparse.ArgumentParser(); ap.add_argument("--every", type=int, default=49)
ap.add_argument("--module", default=None, help="a diagnostic copy of ukf.ts to run instead of the published one")
ap.add_argument("--tag", default="albula_ukf"); args = ap.parse_args()
HERE = Path(__file__).resolve().parent
DATA = Path.home() / "tmp/data/tractography"
H, OUT = DATA / "ukf/hardi", DATA / "albula"
OUT.mkdir(exist_ok=True)

D = U.load(str(H / "dwi.nhdr"), str(H / "mask.nrrd"))
off = U.SRAND0_OFFSET                                                    # the binary's seed offset (macOS srand(0))
pts, *_ = U.seeds(D, off)
sel = np.arange(0, len(pts), args.every)
N = D["N"]
nk, nj, ni = D["dim"]
meta = {"dims": [ni, nj, nk], "voxel": D["voxel"].numpy()[::-1].tolist(), "ijkToRAS": D["i2r"].reshape(-1).tolist(), "G": N,
        "opts": {"freeWater": False, "seedingThreshold": 0.1, "stoppingFA": 0.08, "stoppingThreshold": 0.06,
                 "recordLength": 1.8, "stepLength": 0.3}}
(OUT / "meta.json").write_text(json.dumps(meta))
D["A"].numpy().astype("<f4").tofile(OUT / "signal.f32")              # (k, j, i, N): the port's voxel-major order
D["g"][:N].numpy().astype("<f8").tofile(OUT / "g.f64")
D["b"][:N].numpy().astype("<f8").tofile(OUT / "b.f64")
(D["mask"].numpy().view(np.uint8) > 0).astype(np.uint8).tofile(OUT / "mask.u8")
pts[sel].numpy()[:, ::-1].astype("<f8").tofile(OUT / "seeds.f64")     # (k, j, i) -> the port's (i, j, k)

t0 = time.time()
r = subprocess.run(["deno", "run", "-A", "--config", str(HERE / "albula/deno.json"), str(HERE / "albula/ukf_port_run.ts"), str(OUT)] + ([args.module] if args.module else []),
                   capture_output=True, text=True)
if r.returncode:
    raise SystemExit(r.stderr[-3000:])
port_info = json.loads(r.stdout.strip().splitlines()[-1]); port_info["wall_s"] = round(time.time() - t0, 1)
po = np.fromfile(OUT / "port_offsets.u32", "<u4").astype(np.int64)
pp = np.fromfile(OUT / "port_points.f32", "<f4").reshape(-1, 3).astype(np.float64)
ps = np.fromfile(OUT / "port_seed.i32", "<i4")
port = {int(s): pp[po[i]:po[i + 1]] for i, s in enumerate(ps)}          # keyed by position in the subset

t0 = time.time()
tf, st = U.track(D, off, select=sel)
torch_s = time.time() - t0
ref = {int(np.searchsorted(sel, s)): f for s, f in zip(st["seed_index"], tf)}

def dist(a, b):
    if len(a) != len(b):
        return None
    return float(min(np.linalg.norm(a - b, axis=1).max(), np.linalg.norm(a - b[::-1], axis=1).max()))

rows = {"both": 0, "ref_only": 0, "port_only": 0, "port_short_kept": 0, "same_len": 0, "d": []}
for k in range(len(sel)):
    a, b = ref.get(k), port.get(k)
    if b is not None and len(b) < 10:
        rows["port_short_kept"] += 1
        b = None                                    # the C++ program drops fibers under 10 points
    if a is None and b is None:
        continue
    if a is None:
        rows["port_only"] += 1; continue
    if b is None:
        rows["ref_only"] += 1; continue
    rows["both"] += 1
    d = dist(a, b)
    if d is not None:
        rows["same_len"] += 1; rows["d"].append(d)
d = np.array(rows.pop("d"))
q = lambda x: [float(f"{v:.3g}") for v in np.quantile(x, [0.5, 0.9, 0.99, 1.0])] if len(x) else []
res = {"seeds_tested": int(len(sel)), "every": args.every, "port": port_info, "torch_cpu_s": round(torch_s, 1),
       "reference_fibers": len(ref), **rows,
       "same_len_max_point_distance_mm_quantiles_50_90_99_100": q(d),
       "same_len_within_1e-3_mm": int((d < 1e-3).sum()), "same_len_within_0.1_mm": int((d < 0.1).sum())}
print(json.dumps(res, indent=1))
(HERE / "results" / f"{args.tag}.json").write_text(json.dumps(res, indent=1))
