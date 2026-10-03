"""The pipeline on one subject entirely on the CPU, timed by stage, against the same pipeline on the
GPU: what a machine without a GPU would pay, and what changes.

    python bench/cpu_timing.py [--sub PAT16]

CPU: the field estimate in float32 (pipeline.correct's CPU dtype), the tracking in float32 (ukf's own
steps, the GPU kernel's arithmetic), the default labeler (RapidParc) on the CPU; torch on 8 threads. Then the GPU run.
Compared: the field (displacement difference), the tractograms (fibers, steps), the labels (tract
mix r, label agreement where both tracked from the same seed). Writes results/cpu_timing.json.
"""
import argparse, json, platform, subprocess
from pathlib import Path
import numpy as np, torch
from tractline import pipeline as P, susceptibility as S
from _ds001226 import load
from tractline.labelers.base import OTHER


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sub", default="PAT16"); args = ap.parse_args()
    HERE = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    s = load(args.sub)
    runs = {}
    for device in ("cpu", "mps"):
        timer = P.Timer(echo=f"{args.sub} {device}")
        corr, tg, labels = P.run(s, P.default_labeler(device), timer, device=device)
        runs[device] = (timer, corr, tg, labels)

    def shares(v):
        c = np.bincount(v, minlength=43)[:OTHER].astype(float)
        return c / c.sum()
    (tc, cc, gc, lc), (tg_, cg, gg, lg) = runs["cpu"], runs["mps"]
    dd = np.abs(cc.displacement_mm - cg.displacement_mm)
    brain = gg.mask.astype(bool)
    # labels where both arms tracked from the same seed voxel and both kept the streamline (seed indices
    # are not comparable: each run's seeds come from its own mask)
    key = lambda v: tuple(int(c) for c in v)
    kc = {key(v): i for i, v in enumerate(np.asarray(gc.stats["seed_voxel"])[lc.keep])}
    pairs = [(kc[key(v)], j) for j, v in enumerate(np.asarray(gg.stats["seed_voxel"])[lg.keep]) if key(v) in kc]
    a, b = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    res = {"subject": args.sub, "machine": chip, "torch_threads": torch.get_num_threads(),
           "seconds": {d: r[0].seconds for d, r in runs.items()},
           "scan_to_labels_s": {d: r[0].total(*P.pipeline_stages()) for d, r in runs.items()},
           "ukf_steps_per_s": {d: round(r[2].stats["fiber_steps"] / r[0].seconds["ukf"]) for d, r in runs.items()},
           "cpu_vs_gpu": {"field_displacement_diff_mm_median_99th": [round(float(np.median(dd[brain])), 3), round(float(np.quantile(dd[brain], 0.99)), 3)],
                          "fibers": [gc.stats["fibers"], gg.stats["fibers"]], "fiber_steps": [gc.stats["fiber_steps"], gg.stats["fiber_steps"]],
                          "tract_mix_r": round(float(np.corrcoef(shares(lc.tract), shares(lg.tract))[0, 1]), 4),
                          "same_seed_streamlines": len(pairs), "label_agreement": round(float((lc.tract[a] == lg.tract[b]).mean()), 4)}}
    print(json.dumps(res, indent=1))
    (HERE / "results/cpu_timing.json").write_text(json.dumps(res, indent=1))
