"""One ds001226 patient end to end on this Mac: our susceptibility correction, the faithful pipeline on
the scan as acquired and as corrected, and both against the patient's T1 and tumor mask.

    DATA/.venv/bin/python bench/tractography/cohort.py --sub PAT13 [--keep]

  1. correction (_susc.py): the field from the AP b0s and the PA pair (estimate, MPS, trilinear),
     applied to the 102 AP volumes (cubic B-splines along phase encoding, Jacobian, CPU float64);
  2. per arm (uncorrected, ours): _prep.py (b0 + b = 2800, median_otsu mask), UKF (Metal kernel, the
     binary's seeds, ORG settings), TractCloud (_tractcloud.py: 40 mm cut, 5-draw majority vote, MPS);
     timed by stage, the one-draw network time reported apart (the pipeline's own);
  3. between the arms: the tract-mix correlation, per-tract relative changes, tract centers moved
     (pat16_topup_compare.py's measures; TractCloud's own redraw floor r 0.995-0.999 on PAT16);
  1b. the estimate's sensitivity: the field again from the b0s plus noise of SD 0.01, the displacement
     difference reported (the optimizer's path, not the scan's noise);
  4. against the T1 (_t1check.py): per arm the residual displacement a rigid alignment cannot remove,
     in the brain (ours-arm mask), where our field displaces > 3 mm, the tumor and its 10 mm margin
     (derivatives/tumor_masks, read by its header); validation: the uncorrected arm's residual
     against our own displacement map (r, slope); precision: the ours-arm fit on b0s 1-3 and 4-6.

Writes results/cohort/<sub>.json and DATA/ds001226/derived/<sub>/cohort_fields.npz (residual maps,
masks, our displacement); the prepared NRRDs and the corrected NIfTI are deleted unless --keep.
"""
import argparse, json, shutil, time
from pathlib import Path
import numpy as np, nibabel as nib, torch
from scipy.ndimage import binary_dilation
import _susc as S
import _ukf_torch as U
from _prep import prep
from _t1check import T1Check, tumor_regions, stats
from _tractcloud import Labeler, TRACT_NAMES
from _subject import load

ap = argparse.ArgumentParser(); ap.add_argument("--sub", required=True); ap.add_argument("--keep", action="store_true")
args = ap.parse_args()
HERE = Path(__file__).resolve().parent
TD = Path.home() / "tmp/data/tractography/ds001226"
SUB = TD / f"sub-{args.sub}/ses-preop"
OUT = TD / "derived" / args.sub / "cohort"
OUT.mkdir(parents=True, exist_ok=True)
(HERE / "results/cohort").mkdir(exist_ok=True)
torch.set_num_threads(8)
T = {}
def stage(name, t0):
    if torch.backends.mps.is_available():
        torch.mps.synchronize()
    T[name] = round(time.time() - t0, 2); print(f"{args.sub} {name} {T[name]} s", flush=True)

# ------------------------------------------------------------------ inputs
X = load(args.sub)
side, PE, sign, trt, A, vox = X.side, X.PE, X.sign, X.trt, X.A, X.vox
bval, bvec, raw, b0i, b0s, pe_rows, pa_offset = X.bval, X.bvec, X.raw, X.b0i, X.b0s, X.pe_rows, X.pa_offset

# ------------------------------------------------------------------ 1. correction
t0 = time.time()
h, motion, _ = S.estimate(b0s, vox, pe_rows, np.full(len(pe_rows), trt), device="mps")
stage("field_estimate", t0)
# the estimate's own sensitivity: the same b0s plus noise of SD 0.01 (signals in the hundreds; the scan's
# noise is ~3 orders larger). PAT16 showed the field move by 0.08 mm (median) / 0.68 mm (99th) when
# topup_ref.py's int16-quantized copy of the b0s (errors <= 0.03) was the input instead.
t0 = time.time()
h_p, _, _ = S.estimate(b0s + np.random.default_rng(0).normal(0, 0.01, b0s.shape), vox, pe_rows, np.full(len(pe_rows), trt), device="mps")
stage("field_estimate_perturbed", t0)
t0 = time.time()
ht = torch.as_tensor(np.asarray(h, dtype=np.float64))
vols = torch.as_tensor(np.moveaxis(raw, -1, 0).astype(np.float64))
corr = S.unwarp_pe_cubic(S.prefilter(vols, PE), ht, torch.gradient(ht, dim=PE)[0], PE,
                         torch.full((vols.shape[0],), sign * trt, dtype=torch.float64)).numpy()
corr = np.moveaxis(np.clip(corr, 0, None), 0, -1).astype(np.float32)
stage("field_apply", t0)
d_ours = sign * trt * np.asarray(h, dtype=np.float64) * vox[PE]           # mm: the AP scan's displacement, by our field

# ------------------------------------------------------------------ 2. the pipeline, per arm
off = np.array([-4158, -2201, -2855], float); off = off / np.linalg.norm(off) * 0.5
lab = Labeler()
arms, masks = {}, {}
for name, data in (("uncorrected", raw), ("ours", corr)):
    t0 = time.time(); info, masks[name] = prep(data, A, bval, bvec, OUT / name); stage(f"{name}_prep", t0)
    t0 = time.time(); D = U.load(str(OUT / name / "dwi.nhdr"), str(OUT / name / "mask.nrrd")); stage(f"{name}_load", t0)
    t0 = time.time(); pts, *_ = U.seeds(D, off); stage(f"{name}_seeds", t0)
    t0 = time.time(); fib, st = U.track(D, off, backend="metal"); stage(f"{name}_ukf", t0)
    t0 = time.time(); lab(fib, draws=[0]); stage(f"{name}_tractcloud_one_draw", t0)
    t0 = time.time(); vote, kept, length = lab(fib); stage(f"{name}_tractcloud_five_draws", t0)
    arms[name] = {"seeds": int(len(pts)), "fibers": st["fibers"], "fiber_steps": st["fiber_steps"], "lab": vote, "kept": kept, "length": length,
                  "ukf_steps_per_s": round(st["fiber_steps"] / T[f"{name}_ukf"]), "mask_voxels": info["mask_voxels"]}

def shares(v):
    c = np.bincount(v, minlength=43)[:42].astype(float)
    return c / c.sum()
def centers(r):
    return {t: np.concatenate([r["kept"][i] for i in np.flatnonzero(r["lab"] == t)]).mean(0)
            for t in range(42) if (r["lab"] == t).sum() >= 20}
a, b = arms["uncorrected"], arms["ours"]
sa, sb = shares(a["lab"]), shares(b["lab"])
named = (sa > 0.002) | (sb > 0.002)
rel = np.abs(sb[named] - sa[named]) / np.maximum(sa[named], 1e-9)
ca, cb = centers(a), centers(b)
moved = {TRACT_NAMES[t]: round(float(np.linalg.norm(ca[t] - cb[t])), 1) for t in ca if t in cb}
mv = np.array(list(moved.values()))

# ------------------------------------------------------------------ 3. against the T1
t0 = time.time()
t1img = nib.load(SUB / f"anat/sub-{args.sub}_ses-preop_T1w.nii.gz")
tumor_t1, margin_t1 = tumor_regions(t1img, nib.load(TD / f"derivatives/tumor_masks/sub-{args.sub}/anat/sub-{args.sub}_space_T1_label-tumor.nii"))
brain = masks["ours"].astype(bool)
fit_mask = binary_dilation(brain, iterations=2) | binary_dilation(masks["uncorrected"].astype(bool), iterations=2)
C = T1Check(t1img, A, raw.shape[:3], fit_mask, PE, vox)
big = brain & (np.abs(d_ours - np.median(d_ours[brain])) > 3)
t1res, fields = {}, {"d_ours": d_ours.astype(np.float32), "brain": brain}
for name, data in (("uncorrected", raw), ("ours", corr)):
    b0 = data[..., b0i].astype(np.float64)
    T0 = C.rigid_start(b0.mean(-1))
    d, Tm, log = C.fit(b0.mean(-1), T0)
    tumor = (C.on_grid(Tm, tumor_t1.astype(float)) > 0.5).numpy()
    margin = (C.on_grid(Tm, margin_t1.astype(float)) > 0.5).numpy() & brain & ~tumor
    fields[name] = d.astype(np.float32); fields[name + "_tumor"] = tumor; fields[name + "_margin"] = margin
    t1res[name] = {"residual_brain": stats(d, brain), "residual_where_ours_displaces_gt_3mm": stats(d, big),
                   "residual_tumor": stats(d, tumor), "residual_tumor_margin_10mm": stats(d, margin),
                   "tumor_voxels": int(tumor.sum()), "margin_voxels": int(margin.sum()),
                   "ngf_after": log[-1]["ngf"], **C.rigid_summary(T0)}
    if name == "ours":
        da, _, _ = C.fit(b0[..., :3].mean(-1), T0)
        db, _, _ = C.fit(b0[..., 3:].mean(-1), T0)
        err = np.abs(da - db) / 2
        t1res[name]["half_split_error_brain"] = stats(err, brain)
        t1res[name]["half_split_error_tumor_margin"] = stats(err, margin)
dc = d_ours - np.median(d_ours[brain])
u = fields["uncorrected"]
t1res["validation"] = {"our_displacement_brain": stats(dc, brain), "our_displacement_tumor": stats(dc, fields["ours_tumor"]),
                       "our_displacement_tumor_margin_10mm": stats(dc, fields["ours_margin"]),
                       "uncorrected_residual_r_vs_our_displacement": round(float(np.corrcoef(u[brain], d_ours[brain])[0, 1]), 3),
                       "uncorrected_residual_slope_vs_our_displacement": round(float(np.polyfit(d_ours[brain], u[brain], 1)[0]), 3),
                       "ours_residual_r_vs_our_displacement": round(float(np.corrcoef(fields["ours"][brain], d_ours[brain])[0, 1]), 3)}
stage("t1_check", t0)
dp = np.abs(np.asarray(h_p, dtype=np.float64) - np.asarray(h, dtype=np.float64)) * trt * vox[PE]
t1res["estimate_sensitivity_mm"] = {"brain": stats(dp, brain), "tumor_margin_10mm": stats(dp, fields["ours_margin"])}

tumor_c = t1img.affine[:3, :3] @ np.argwhere(tumor_t1).mean(0) + t1img.affine[:3, 3]
res = {"subject": args.sub, "data": f"ds001226 sub-{args.sub} ses-preop (CC0): AP {side['AP']['PhaseEncodingDirection']} + PA {side['PA']['PhaseEncodingDirection']}, readout {trt} s",
       "tumor": {"volume_cm3_on_T1": round(float(tumor_t1.sum()) / 1000, 1), "centroid_ras_mm": [round(float(v), 1) for v in tumor_c],
                 "t1_mean_inside_vs_mirror": [round(float(C.t1[tumor_t1].mean()), 1), round(float(C.t1[tumor_t1[::-1]].mean()), 1)]},
       "pa_slab_offset": pa_offset,
       "seconds": T,
       "pipeline_seconds_corrected": round(T["field_estimate"] + T["field_apply"] + T["ours_prep"] + T["ours_load"] + T["ours_seeds"] + T["ours_ukf"] + T["ours_tractcloud_one_draw"], 1),
       "field_hz_1_50_99": [round(float(v), 1) for v in np.quantile(np.asarray(h)[brain], [0.01, 0.5, 0.99])],
       "motion_translation_mm_max": round(float(np.abs(motion[:, :3]).max()), 2),
       "arms": {n: {k: v for k, v in r.items() if k not in ("lab", "kept", "length")} | {
                "streamlines_ge_40mm": len(r["kept"]), "median_length_mm": round(float(np.median(r["length"])), 1),
                "other_fraction": round(float((r["lab"] == 42).mean()), 4)} for n, r in arms.items()},
       "correction_changes": {"tract_mix_r": round(float(np.corrcoef(sa, sb)[0, 1]), 4),
                              "per_tract_abs_rel_change_median_90th": [round(float(np.median(rel)), 3), round(float(np.quantile(rel, 0.9)), 3)],
                              "tract_center_moved_mm_median_90th_max": [round(float(np.median(mv)), 1), round(float(np.quantile(mv, 0.9)), 1), round(float(mv.max()), 1)],
                              "tract_center_moved_mm": dict(sorted(moved.items(), key=lambda x: -x[1]))},
       "t1": t1res, "columns": "|displacement| mm: median / 90th / 99th percentile"}
(HERE / f"results/cohort/{args.sub}.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(TD / "derived" / args.sub / "cohort_fields.npz", **fields)
if not args.keep:
    shutil.rmtree(OUT)
print(json.dumps({k: res[k] for k in ("tumor", "pipeline_seconds_corrected", "correction_changes")} | {"t1": {k: {kk: vv for kk, vv in v.items() if kk.startswith("residual") or kk.startswith("uncorr")} for k, v in t1res.items()}}, indent=1))
