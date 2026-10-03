"""The CPU tracker's speed-ups against float64, on PAT16 (as acquired): what `fast` changes, and what
the batch size changes, next to what float32 alone changes.

    uv run bench/ukf_cpu_check.py

Every 20th seed (~2,400), tracked by: float64 (the binary's arithmetic; the reference), float32 at the
default batch, float32 at batch 1024 (the worker setting), float32 `fast` at batch 1024. Against the
reference, fiber by fiber (matched by seed voxel): the share with the same number of points, the share
whose points are all within 0.1 mm, the share of fiber ends within 0.06 mm, and the correlation of
point densities on a 2.5 mm grid (ukf32_compare.py's measures). Then speed on a quarter of the seeds:
8 workers x 1 thread, exact and fast. Writes results/ukf_cpu_check.json.
"""
import json, time
from pathlib import Path
import numpy as np, torch
from tractline import ukf as U
from _ds001226 import load
from tractline.prep import prepare

HERE = Path(__file__).resolve().parent


def compare(ref, refst, f, st):
    key = lambda v: tuple(int(c) for c in v)
    R = {key(v): a for v, a in zip(refst["seed_voxel"], ref)}
    F = {key(v): a for v, a in zip(st["seed_voxel"], f)}
    both = [k for k in R if k in F]
    same_n = [k for k in both if len(R[k]) == len(F[k])]
    within = sum(float(np.abs(R[k] - F[k]).max()) < 0.1 for k in same_n)
    ends = [np.linalg.norm(R[k][e] - F[k][e]) < 0.06 for k in both for e in (0, -1)]
    lo = np.concatenate(ref).min(0) - 5
    def dens(fs):
        P = np.concatenate(fs); i = np.floor((P - lo) / 2.5).astype(int)
        return np.bincount(np.ravel_multi_index(i.T, (100, 100, 100)), minlength=10 ** 6)
    return {"fibers": [len(ref), len(f)], "matched": len(both), "same_point_count": round(len(same_n) / len(both), 4),
            "all_points_within_0.1mm": round(within / len(both), 4), "ends_within_0.06mm": round(float(np.mean(ends)), 4),
            "density_r": round(float(np.corrcoef(dens(ref), dens(f))[0, 1]), 5)}


if __name__ == "__main__":
    torch.set_num_threads(8)
    s = load("PAT16"); t = prepare(s.dwi, s.affine, s.bval, s.bvec); D = U.from_arrays(t.dwi, t.header, t.mask)
    pts, *_ = U.seeds(D)
    sel = np.arange(0, len(pts), 20)
    ref, refst = U.track(D, select=sel, dtype=torch.float64)
    res = {"seeds": len(sel), "against_float64": {}}
    for name, kw in (("float32, batch 50000", {}), ("float32, batch 1024", dict(batch=1024)),
                     ("float32 fast, batch 1024", dict(batch=1024, fast=True))):
        f, st = U.track(D, select=sel, dtype=torch.float32, device="cpu", **kw)
        res["against_float64"][name] = compare(ref, refst, f, st)
        print(name, res["against_float64"][name], flush=True)
    quarter = np.arange(0, len(pts), 4)
    res["speed_quarter_of_seeds"] = {}
    for name, kw in (("8 workers, exact", {}), ("8 workers, fast", dict(fast=True))):
        t0 = time.time(); f, st = U.track(D, select=quarter, dtype=torch.float32, device="cpu", batch=1024, workers=8, **kw); tt = time.time() - t0
        res["speed_quarter_of_seeds"][name] = {"seconds": round(tt, 1), "k_steps_per_s": round(st["fiber_steps"] / tt / 1e3, 1)}
        print(name, res["speed_quarter_of_seeds"][name], flush=True)
    (HERE / "results/ukf_cpu_check.json").write_text(json.dumps(res, indent=1))
