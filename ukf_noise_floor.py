"""The noise floor for UKF: how far the scan's own measurement noise moves the float64 tracker, to judge
float32 (ukf32.json) against something the original pipeline already has.

    DATA/.venv/bin/python bench/tractography/ukf_noise_floor.py [--every 49] [--replicates 4]

Stanford HARDI, ORG settings. Wild-bootstrap replicates of the scan (_bootstrap.py, SH order 6),
each tracked in float64 on the CPU from the SAME seed points as the scan as acquired: every
`every`-th seed the binary accepts on the original (its voxels plus the srand(0) offset), each
started from its state on the replicate, none rejected (how many the replicate would have rejected
is reported). Fibers are compared seed by seed with the measures of ukf32_compare.py
(_fibercmp.py): replicate against original, and replicate against replicate. The float32-vs-float64
row from ukf32.json is carried beside them.

Writes results/ukf_noise_floor.json.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
import _ukf_torch as U
import _fibercmp as C
from _bootstrap import WildBootstrap, read

ap = argparse.ArgumentParser()
ap.add_argument("--every", type=int, default=49)
ap.add_argument("--replicates", type=int, default=4)
ap.add_argument("--sh-order", type=int, default=6)
args = ap.parse_args()
HERE = Path(__file__).resolve().parent
H = Path.home() / "tmp/data/tractography/ukf/hardi"
NHDR, MASK = str(H / "dwi.nhdr"), str(H / "mask.nrrd")

D = U.load(NHDR, MASK)
off = U.SRAND0_OFFSET                                                    # the binary's seed offset (macOS srand(0))
pts, *_ = U.seeds(D, off)
P = pts[np.arange(0, len(pts), args.every)]
i2r, dims_ijk = D["i2r"], tuple(int(v) for v in D["dim"][::-1])
cmp = lambda a, b: C.compare(a, b, i2r, dims_ijk)

def run(Dx):
    t0 = time.time()
    f, st = U.track(Dx, off, seed_points=P)
    return {int(k): p for k, p in zip(st["seed_index"], f)}, st, time.time() - t0

import nrrd
raw, g, is_b0 = read(NHDR)
wb = WildBootstrap(raw, g, is_b0, sh_order=args.sh_order)
res = {"data": "Stanford HARDI, ORG settings", "seeds": int(len(P)), "every": args.every,
       "bootstrap": wb.noise_check(nrrd.read(MASK)[0])}
orig, st, s = run(D)
res["original"] = {"fibers": st["fibers"], "fiber_steps": st["fiber_steps"], "seconds": round(s, 1)}
reps = []
for r in range(args.replicates):
    Dr = U.load(NHDR, MASK, data=wb.replicate(20261001 + r))
    acc = U.seed_states(Dr, P.clone())[4]
    f, st, s = run(Dr)
    reps.append(f)
    res.setdefault("replicates", []).append({"seed": 20261001 + r, "fibers": st["fibers"], "seconds": round(s, 1),
                                             "seeds_the_replicate_would_reject": int((~acc).sum()),
                                             "vs_original": cmp(f, orig)})
    print(json.dumps(res["replicates"][-1]), flush=True)
res["replicate_pairs"] = [cmp(reps[i], reps[j]) for i in range(len(reps)) for j in range(i + 1, len(reps))][:3]
f32 = json.loads((HERE / "results" / "ukf32.json").read_text())["fibers"]
res["float32_for_scale"] = {"f32_cpu_vs_f64_cpu": f32["f32_cpu_vs_f64_cpu"], "f32_mps_vs_f64_cpu": f32["f32_mps_vs_f64_cpu"]}
print(json.dumps(res, indent=1))
(HERE / "results" / "ukf_noise_floor.json").write_text(json.dumps(res, indent=1))
