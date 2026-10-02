"""Our susceptibility-field estimate (_susc.py) against FSL topup's (topup_ref.py) on PAT16's 8 b0s.

    DATA/.venv/bin/python bench/tractography/susc_check.py [--device mps|cpu] [--iter-scale 3]

Compared inside the brain (median_otsu of topup's mean unwarped b0): the field (Hz) - correlation,
median and 99th-percentile absolute difference, the same as displacement along the phase-encoding
axis (mm, at PAT16's 0.0266 s readout) - and the motion estimates against topup_movpar.txt. A field
of the opposite sign to topup's would correlate negatively: that is reported, not hidden.

Writes DATA/ds001226/derived/PAT16/susc/field_hz.nii.gz and results/susc_check.json.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, nibabel as nib
from dipy.segment.mask import median_otsu
import _susc as S

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="mps"); ap.add_argument("--iter-scale", type=int, default=3)
ap.add_argument("--lam-scale", type=float, default=1.0); ap.add_argument("--tag", default="")
ap.add_argument("--interp", choices=("cubic_pe", "trilinear"), default="cubic_pe")
ap.add_argument("--motion", choices=("estimate", "zero", "topup", "topup_neg"), default="estimate",
                help="diagnostic: hold the motion fixed (none, topup's movpar as given, or with its sign reversed)")
args = ap.parse_args()
HERE = Path(__file__).resolve().parent
TP = Path.home() / "tmp/data/tractography/ds001226/derived/PAT16/topup"
OUT = TP.parent / "susc"; OUT.mkdir(exist_ok=True)

img = nib.load(TP / "b0s.nii.gz")
b0s = np.asarray(img.dataobj, dtype=np.float64)
vox = img.header.get_zooms()[:3]
acq = np.loadtxt(TP / "acqparams.txt")
pe, trt = acq[:, :3], acq[:, 3]

mp = np.loadtxt(TP / "topup_movpar.txt")
fixed = {"estimate": None, "zero": np.zeros_like(mp), "topup": mp, "topup_neg": -mp}[args.motion]
t0 = time.time()
h, motion, log = S.estimate(b0s, vox, pe, trt, device=args.device, iter_scale=args.iter_scale, lam_scale=args.lam_scale, fixed_motion=fixed, interp=args.interp,
                            progress=lambda r: print(f"level {r['level']}: grid {r['grid']}, ssd {r['ssd_before']:.4g} -> {r['ssd_after']:.4g}, "
                                                     f"{time.time() - t0:.0f} s", flush=True))
secs = time.time() - t0
nib.save(nib.Nifti1Image(h.astype(np.float32), img.affine), OUT / f"field_hz{args.tag}.nii.gz")

ref = np.asarray(nib.load(TP / "field_hz.nii.gz").dataobj, dtype=np.float64)
mean_unwarped = np.asarray(nib.load(TP / "b0_unwarped_mean.nii.gz").dataobj, dtype=np.float64) if (TP / "b0_unwarped_mean.nii.gz").exists() \
    else np.asarray(nib.load(TP / "b0_unwarped.nii.gz").dataobj, dtype=np.float64).mean(-1)
_, brain = median_otsu(mean_unwarped, median_radius=4, numpass=4)
a, b = h[brain], ref[brain]
r = float(np.corrcoef(a, b)[0, 1])
d = np.abs(a - b)
mm = d * trt[0] * vox[int(np.argmax(np.abs(pe).sum(0)))]
mov_ref = np.loadtxt(TP / "topup_movpar.txt")
res = {"interp": args.interp, "motion_mode": args.motion, "device": args.device, "iter_scale": args.iter_scale, "lam_scale": args.lam_scale, "seconds": round(secs, 1), "brain_voxels": int(brain.sum()),
       "field_r_vs_topup": round(r, 4),
       "field_abs_diff_hz_median_99th": [round(float(np.median(d)), 2), round(float(np.quantile(d, 0.99)), 2)],
       "displacement_diff_mm_median_99th_max": [round(float(np.median(mm)), 3), round(float(np.quantile(mm, 0.99)), 3), round(float(mm.max()), 3)],
       "field_hz_ours_1_50_99": [round(float(v), 1) for v in np.quantile(a, [0.01, 0.5, 0.99])],
       "field_hz_topup_1_50_99": [round(float(v), 1) for v in np.quantile(b, [0.01, 0.5, 0.99])],
       "motion_ours_translation_mm_max": round(float(np.abs(motion[:, :3]).max()), 3),
       "motion_topup_translation_mm_max": round(float(np.abs(mov_ref[:, :3]).max()), 3),
       "motion_abs_diff_translation_mm_max": round(float(np.abs(motion[:, :3] - mov_ref[:, :3]).max()), 3),
       "levels": log}
# where the field differs: deep inside the brain or at its edge, and by signal level
from scipy.ndimage import binary_erosion
deep = binary_erosion(brain, iterations=3)
sig = mean_unwarped[brain]
q = np.quantile(sig, [0.25, 0.5, 0.75])
res["displacement_diff_mm_99th_deep_vs_edge"] = [round(float(np.quantile(np.abs(h - ref)[deep] * trt[0] * 2.5, 0.99)), 3),
                                                 round(float(np.quantile(np.abs(h - ref)[brain & ~deep] * trt[0] * 2.5, 0.99)), 3)]
res["displacement_diff_mm_median_by_signal_quartile"] = [round(float(np.median(mm[(sig >= lo) & (sig < hi)])), 3)
                                                         for lo, hi in zip([-np.inf, *q], [*q, np.inf])]

# the corrected images: our unwarp (field + motion) against topup's iout, the uncorrected images for scale
import torch
dt = torch.float64
img4 = torch.as_tensor(np.moveaxis(b0s, -1, 0), dtype=dt)
img4 = img4 * (img4.mean() / img4.mean(dim=(1, 2, 3), keepdim=True))
pe_ax = int(np.argmax(np.abs(pe).sum(0)))
ht = torch.as_tensor(h, dtype=dt)
dh = torch.gradient(ht, dim=pe_ax)[0]
vt = torch.as_tensor(np.asarray(vox, float), dtype=dt)
center = (torch.tensor(b0s.shape[:3], dtype=dt) - 1) / 2 * vt
Rs, ts = zip(*[S.rigid(torch.as_tensor(motion[v], dtype=dt), center) for v in range(b0s.shape[-1])])
ours = S.unwarp(img4, ht, dh, pe_ax, torch.as_tensor(pe[:, pe_ax] * trt, dtype=dt), list(Rs), list(ts), vt).numpy()
theirs = np.moveaxis(np.asarray(nib.load(TP / "b0_unwarped.nii.gz").dataobj, dtype=np.float64), -1, 0)
raw = img4.numpy()
cc = lambda A, B: float(np.mean([np.corrcoef(A[v][brain], B[v][brain])[0, 1] for v in range(len(A))]))
rr = lambda A, B: float(np.mean([np.sqrt(np.mean((A[v][brain] / A[v][brain].mean() - B[v][brain] / B[v][brain].mean()) ** 2)) for v in range(len(A))]))
res["unwarped_b0_vs_topup"] = {"corr_ours": round(cc(ours, theirs), 4), "corr_uncorrected": round(cc(raw, theirs), 4),
                               "rel_rms_ours": round(rr(ours, theirs), 4), "rel_rms_uncorrected": round(rr(raw, theirs), 4),
                               "ap_pa_rel_rms_ours": round(float(np.sqrt(np.mean((ours[:6].mean(0)[brain] - ours[6:].mean(0)[brain]) ** 2)) / ours[:, brain].mean()), 4),
                               "ap_pa_rel_rms_topup": round(float(np.sqrt(np.mean((theirs[:6].mean(0)[brain] - theirs[6:].mean(0)[brain]) ** 2)) / theirs[:, brain].mean()), 4),
                               "ap_pa_rel_rms_uncorrected": round(float(np.sqrt(np.mean((raw[:6].mean(0)[brain] - raw[6:].mean(0)[brain]) ** 2)) / raw[:, brain].mean()), 4)}
print(json.dumps({k: v for k, v in res.items() if k != "levels"}, indent=1))
(HERE / "results" / f"susc_check{args.tag}.json").write_text(json.dumps(res, indent=1))
