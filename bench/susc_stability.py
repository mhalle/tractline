"""The field estimate's stability and accuracy, current settings against candidates, on one subject.

    python bench/susc_stability.py --sub PAT13 [--configs '[[3, 1], [5, 1e5]]']

Per configuration ([iter_scale, lam_scale] of L-BFGS, as recorded before Gauss-Newton, or a dict of
estimate options), on the GPU: the field from the b0s as they are and plus noise
of SD 0.01 - their displacement difference deep in the brain (the cohort's mask, eroded 3 voxels) and at
its edge (median / 99th / max) - and the scan corrected by each field against the subject's T1
(t1check: brain, tumor, 10 mm margin; median / 90th / 99th); the first field's folds (voxels where
a polarity's Jacobian 1 -/+ readout dh/dpe is negative) in the brain and outside, and its largest
displacement outside (mm). Writes results/susc_stability/<sub>[_tag].json.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, nibabel as nib, torch
from scipy.ndimage import binary_dilation, binary_erosion
from tractline import susceptibility as S
from _ds001226 import load, ROOT
from tractline.t1check import T1Check, tumor_regions, stats
from tractline.prep import prepare

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--sub", required=True)
ap.add_argument("--configs", default='[[3, 1], {}]', help='[[iter_scale, lam_scale], {"optimizer": "gn"}, ...]')
ap.add_argument("--tag", default="")
args = ap.parse_args()
torch.set_num_threads(8)
s = load(args.sub)
brain = np.load(ROOT / f"derived/{args.sub}/cohort_fields.npz")["brain"]
deep = binary_erosion(brain, iterations=3); edge = brain & ~deep
mm = lambda h: S.displacement_mm(h, s.readout_s, s.pe_sign, s.vox[s.pe_axis])
q = lambda d, m: [round(float(np.median(d[m])), 3), round(float(np.quantile(d[m], 0.99)), 3), round(float(d[m].max()), 3)]
fit_mask = binary_dilation(brain, iterations=2) | binary_dilation(prepare(s.dwi, s.affine, s.bval, s.bvec).mask, iterations=2)
t1img = nib.load(s.t1); tumor_t1, margin_t1 = tumor_regions(t1img, nib.load(s.tumor_mask))
C = T1Check(t1img, s.affine, s.dwi.shape[:3], fit_mask, s.pe_axis, s.vox)


def t1(h):
    corr = S.apply(s.dwi, h, s.pe_axis, s.pe_sign, s.readout_s)
    b0 = corr[..., s.b0_index].astype(np.float64).mean(-1)
    T0 = C.rigid_start(b0); d, Tm, _ = C.fit(b0, T0)
    tumor = (C.on_grid(Tm, tumor_t1.astype(float)) > 0.5).numpy()
    margin = (C.on_grid(Tm, margin_t1.astype(float)) > 0.5).numpy() & brain & ~tumor
    return {"brain": stats(d, brain), "tumor": stats(d, tumor), "margin": stats(d, margin)}


res = {"subject": args.sub, "configs": {}}
for cfg in json.loads(args.configs):
    opts = cfg if isinstance(cfg, dict) else {"optimizer": "lbfgs", "iter_scale": cfg[0], "lam_scale": cfg[1]}
    name = f"iter_scale {cfg[0]}, lam_scale {cfg[1]:g}" if isinstance(cfg, list) else ", ".join(f"{k} {v}" for k, v in cfg.items())
    fields, secs = [], []
    for b0s in (s.b0s, s.b0s + np.random.default_rng(0).normal(0, 0.01, s.b0s.shape)):
        t0 = time.time(); h, *_ = S.estimate(b0s, s.vox, s.pe_vectors, s.readout_s, device="mps", **opts)
        secs.append(round(time.time() - t0, 1)); fields.append(h)
    d = np.abs(mm(fields[0]) - mm(fields[1]))
    jmin = 1 - np.abs(s.readout_s * np.gradient(fields[0], axis=s.pe_axis))
    r = {"seconds": secs, "stability_deep": q(d, deep), "stability_edge": q(d, edge), "t1": t1(fields[0]),
         "folds_brain": int((jmin[brain] < 0).sum()), "folds_outside": int((jmin[~brain] < 0).sum()),
         "outside_max_mm": round(float(np.abs(mm(fields[0])[~brain]).max()), 1)}
    res["configs"][name] = r
    print(args.sub, name, json.dumps(r), flush=True)
out = HERE / "results/susc_stability"; out.mkdir(exist_ok=True)
(out / f"{args.sub}{('_' + args.tag) if args.tag else ''}.json").write_text(json.dumps(res, indent=1))
