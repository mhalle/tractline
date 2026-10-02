"""FSL topup on a cohort subject, from the very b0 stack our estimate used (_subject.py), as a second
reference where the T1 check is in doubt: the two fields compared, and topup's corrected scan put
through the same T1 check as cohort.py's arms (same brain and fit masks, same tumor regions).

    DATA/.venv/bin/python bench/tractography/cohort_topup.py --sub PAT23

  - topup b02b0.cnf on the b0 stack written as float32 with its own header (topup_ref.py wrote PAT16's
    through the int16 DWI's header, which quantized it by up to 0.03), applytopup --method=jac on the
    AP DWI (inindex 1);
  - our field: _susc.estimate on the same stack (deterministic: cohort.py's field);
  - regions: the brain (cohort.py's ours-arm mask), the tumor's 10 mm margin, and "spot": the margin
    voxels where cohort.py found our corrected scan > 3 mm off the T1.

Writes results/cohort/<sub>_topup.json.
"""
import argparse, json, os, subprocess, time
from pathlib import Path
import numpy as np, nibabel as nib, torch
from scipy.ndimage import binary_dilation
from dipy.segment.mask import median_otsu
import _susc as S
from _subject import load, TD
from _t1check import T1Check, tumor_regions, stats

ap = argparse.ArgumentParser(); ap.add_argument("--sub", required=True); args = ap.parse_args()
HERE = Path(__file__).resolve().parent
X = load(args.sub)
OUT = TD / "derived" / args.sub / "topup"; OUT.mkdir(parents=True, exist_ok=True)
FSL = TD.parent / "fsl-env"
env = {**os.environ, "FSLDIR": str(FSL), "FSLOUTPUTTYPE": "NIFTI_GZ", "PATH": f"{FSL / 'bin'}:{os.environ['PATH']}"}
CNF = FSL / "src/fsl-topup/flirtsch/b02b0.cnf"
torch.set_num_threads(8)

def run(cmd):
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=OUT)
    if r.returncode:
        raise SystemExit(f"{cmd[0]} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return round(time.time() - t0, 1)

hdr = nib.Nifti1Header(); hdr.set_data_dtype(np.float32)
nib.save(nib.Nifti1Image(X.b0s.astype(np.float32), X.A, hdr), OUT / "b0s.nii.gz")
fmt = lambda v: " ".join(f"{int(x)}" for x in v)
(OUT / "acqparams.txt").write_text("".join(f"{fmt(v)} {X.trt}\n" for v in X.pe_rows))
t_topup = run([str(FSL / "bin/topup"), "--imain=b0s", "--datain=acqparams.txt", f"--config={CNF}", "--out=topup", "--fout=field_hz", "--iout=b0_unwarped"])
src = X.SUB / f"dwi/sub-{args.sub}_ses-preop_acq-AP_dwi.nii.gz"
t_apply = run([str(FSL / "bin/applytopup"), f"--imain={src}", "--inindex=1", "--datain=acqparams.txt", "--topup=topup", "--method=jac", "--out=dwi_AP_topup"])
h_top = np.asarray(nib.load(OUT / "field_hz.nii.gz").dataobj, dtype=np.float64)
top = np.asarray(nib.load(OUT / "dwi_AP_topup.nii.gz").dataobj, dtype=np.float64)

t0 = time.time()
h_ours, _, _ = S.estimate(X.b0s, X.vox, X.pe_rows, np.full(len(X.pe_rows), X.trt), device="mps")
t_ours = round(time.time() - t0, 1)
h_ours = np.asarray(h_ours, dtype=np.float64)

F = np.load(TD / "derived" / args.sub / "cohort_fields.npz")
brain, margin = F["brain"], F["ours_margin"]
spot = margin & (np.abs(F["ours"]) > 3)
mm = X.sign * X.trt * X.vox[X.PE]                                         # Hz -> mm along PE for the AP scan
d_top, d_ours = h_top * mm, h_ours * mm
dd = np.abs(d_ours - d_top)
dc = lambda d: d - np.median(d[brain])

# topup's corrected scan through cohort.py's T1 check (the same masks: brain, and the fit mask from both arms' prep masks)
_, unc_mask = median_otsu(X.raw[..., X.bval < 50].astype(np.float32).mean(-1), median_radius=4, numpass=4)
fit_mask = binary_dilation(brain, iterations=2) | binary_dilation(unc_mask, iterations=2)
t1img = nib.load(X.SUB / f"anat/sub-{args.sub}_ses-preop_T1w.nii.gz")
tumor_t1, margin_t1 = tumor_regions(t1img, nib.load(TD / f"derivatives/tumor_masks/sub-{args.sub}/anat/sub-{args.sub}_space_T1_label-tumor.nii"))
C = T1Check(t1img, X.A, X.raw.shape[:3], fit_mask, X.PE, X.vox)
b0 = top[..., X.b0i].mean(-1)
T0 = C.rigid_start(b0)
d_res, Tm, _ = C.fit(b0, T0)
tm = (C.on_grid(Tm, tumor_t1.astype(float)) > 0.5).numpy()
mg = (C.on_grid(Tm, margin_t1.astype(float)) > 0.5).numpy() & brain & ~tm

res = {"subject": args.sub, "seconds": {"topup": t_topup, "applytopup": t_apply, "ours_estimate": t_ours},
       "fields": {"r_brain": round(float(np.corrcoef(h_ours[brain], h_top[brain])[0, 1]), 4),
                  "displacement_diff_mm_brain": stats(dd, brain), "displacement_diff_mm_margin": stats(dd, margin),
                  "spot_voxels": int(spot.sum()), "displacement_diff_mm_spot": stats(dd, spot),
                  "spot_median_displacement_mm_ours_topup": [round(float(np.median(dc(d_ours)[spot])), 2), round(float(np.median(dc(d_top)[spot])), 2)]},
       "t1_residual": {"topup": {"brain": stats(d_res, brain), "tumor": stats(d_res, tm), "margin": stats(d_res, mg), "spot": stats(d_res, spot)},
                       "ours": {"brain": stats(F["ours"], brain), "tumor": stats(F["ours"], F["ours_tumor"]), "margin": stats(F["ours"], margin), "spot": stats(F["ours"], spot)},
                       "uncorrected": {"brain": stats(F["uncorrected"], brain), "margin": stats(F["uncorrected"], F["uncorrected_margin"]), "spot": stats(F["uncorrected"], spot)}},
       "columns": "|x| mm: median / 90th / 99th percentile"}
(HERE / f"results/cohort/{args.sub}_topup.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(TD / "derived" / args.sub / "topup_fields.npz", d_top=d_top.astype(np.float32), topup_residual=d_res.astype(np.float32))
print(json.dumps(res, indent=1))
