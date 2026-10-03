"""Does distortion correction put PAT16's diffusion data where the T1 says the anatomy is? The T1
(MPRAGE, 1 mm, essentially undistorted) is the independent arbiter between the scan as acquired,
FSL topup's correction and ours (susceptibility.py) - none of which saw it.

    uv run bench/t1_alignment.py

Per arm, from the mean of the AP DWI's 6 b0s, t1check.py's measure (rigid start, then rigid + a smooth
residual displacement along phase encoding against the T1, normalized gradient fields).
d is the distortion a rigid alignment to the T1 cannot remove: what an overlay of tracts on the T1
would be off by.
  - validation: the uncorrected arm's d against topup's own displacement (field x readout x voxel):
    if the T1 recovers topup's map independently, the measure measures distortion;
  - precision: the fit repeated on b0s 1-3 and 4-6 separately, |d_a - d_b| / 2 the error of one;
  - regions: inside the topup arm's brain mask, where topup displaces > 3 mm and elsewhere; the
    tumor (ds001226 derivatives/tumor_masks, manual + disconnectome, on the T1; its array is stored
    left-right flipped against the T1's with a header to match: read by its header it covers the
    hypointense lesion, by voxel order the healthy mirror) and a 10 mm margin around it, each
    carried onto the b0 grid by the arm's own rigid fit.

Writes results/t1_alignment.json and results/t1_alignment.png.
"""
import json, time
from pathlib import Path
from tractline.data import DATA as _DATA                          # $TRACTOGRAPHY_DATA
import numpy as np, nibabel as nib, nrrd
from scipy.ndimage import binary_dilation
import torch
from tractline.t1check import T1Check, tumor_regions, stats

HERE = Path(__file__).resolve().parent
TD = _DATA / "ds001226"
SUB = TD / "sub-PAT16/ses-preop"
DER = TD / "derived"
ARMS = {"uncorrected": SUB / "dwi/sub-PAT16_ses-preop_acq-AP_dwi.nii.gz",
        "topup": DER / "PAT16/topup/dwi_AP_topup.nii.gz",
        "ours": DER / "PAT16/susc/dwi_AP_ours2.nii.gz",
        "ours_fast": DER / "PAT16/susc/dwi_AP_ours_fast.nii.gz"}                 # the field estimated with trilinear sampling (29 s)
PE = 1                                                                    # j: AP acquisition, "j-"
dt = torch.float64
torch.set_num_threads(8)

t1img = nib.load(SUB / "anat/sub-PAT16_ses-preop_T1w.nii.gz")
bval = np.loadtxt(SUB / "dwi/sub-PAT16_ses-preop_acq-AP_dwi.bval")
b0i = np.flatnonzero(bval < 50)
ref = nib.load(ARMS["uncorrected"])
A, shape = ref.affine, ref.shape[:3]
vox = np.asarray(ref.header.get_zooms()[:3], float)

topup_mask = nrrd.read(str(DER / "PAT16_topup/mask.nrrd"))[0].astype(bool)
fit_mask = binary_dilation(topup_mask, iterations=2)
for a in ("PAT16", "PAT16_ours2"):
    fit_mask |= binary_dilation(nrrd.read(str(DER / a / "mask.nrrd"))[0].astype(bool), iterations=2)
side = json.loads((SUB / "dwi/sub-PAT16_ses-preop_acq-AP_dwi.json").read_text())
h_topup = np.asarray(nib.load(DER / "PAT16/topup/field_hz.nii.gz").dataobj, dtype=np.float64)
d_topup = -1 * side["TotalReadoutTime"] * h_topup * vox[PE]               # mm, j-: the AP scan's displacement
big = topup_mask & (np.abs(d_topup) > 3)
tumor_t1, margin_t1 = tumor_regions(t1img, nib.load(TD / "derivatives/tumor_masks/sub-PAT16/anat/sub-PAT16_space_T1_label-tumor.nii"))
C = T1Check(t1img, A, shape, fit_mask, PE, vox)
t1 = C.t1

res = {"data": "ds001226 PAT16: mean AP b0 of each arm against the T1w (MPRAGE 1 mm)",
       "tumor": {"volume_cm3_on_T1": round(float(tumor_t1.sum()) / 1000, 1), "centroid_ras_mm": [round(float(v), 1) for v in t1img.affine[:3, :3] @ np.argwhere(tumor_t1).mean(0) + t1img.affine[:3, 3]],
                 "t1_mean_inside_vs_mirror": [round(float(t1[tumor_t1].mean()), 1), round(float(t1[tumor_t1[::-1]].mean()), 1)]},
       "regions": {"brain_voxels": int(topup_mask.sum()), "topup_displaces_gt_3mm_voxels": int(big.sum())},
       "columns": "|residual displacement| along PE, mm: median / 90th / 99th percentile", "arms": {}}
fields = {}
for name, path in ARMS.items():
    t0 = time.time()
    dwi = nib.load(path)
    b0s = np.asarray(dwi.dataobj, dtype=np.float64)[..., b0i]
    assert np.allclose(dwi.affine, A, atol=1e-3)
    b0 = b0s.mean(-1)
    T0 = C.rigid_start(b0)
    d, T, log = C.fit(b0, T0)
    da, _, _ = C.fit(b0s[..., :3].mean(-1), T0)
    db, _, _ = C.fit(b0s[..., 3:].mean(-1), T0)
    err = np.abs(da - db) / 2
    fields[name] = d
    tumor = (C.on_grid(T, tumor_t1.astype(float)) > 0.5).numpy()
    margin = (C.on_grid(T, margin_t1.astype(float)) > 0.5).numpy() & topup_mask & ~tumor
    fields[name + "_T"] = T.numpy(); fields[name + "_tumor"] = tumor; fields[name + "_margin"] = margin
    res["arms"][name] = {
        "residual_tumor": stats(d, tumor), "residual_tumor_margin_10mm": stats(d, margin),
        "topup_displacement_tumor": stats(d_topup - np.median(d_topup[topup_mask]), tumor),
        "topup_displacement_tumor_margin_10mm": stats(d_topup - np.median(d_topup[topup_mask]), margin),
        "tumor_voxels": int(tumor.sum()), "margin_voxels": int(margin.sum()),
        "residual_brain": stats(d, topup_mask), "residual_where_topup_gt_3mm": stats(d, big), "residual_elsewhere": stats(d, topup_mask & ~big),
        "half_split_error_brain": stats(err, topup_mask), "half_split_error_where_topup_gt_3mm": stats(err, big),
        "ngf_rigid_only": round(C.rigid_only_ngf(b0, T0), 5), "ngf_after": log[-1]["ngf"],
        "r_residual_vs_topup_displacement": round(float(np.corrcoef(d[topup_mask], d_topup[topup_mask])[0, 1]), 3),
        **C.rigid_summary(T0),
        "seconds": round(time.time() - t0, 1), "levels": log}
    print(name, json.dumps({k: v for k, v in res["arms"][name].items() if k != "levels"}), flush=True)

u = fields["uncorrected"]
res["validation"] = {"topup_displacement_brain": stats(d_topup, topup_mask),
                     "uncorrected_residual_r_vs_topup": res["arms"]["uncorrected"]["r_residual_vs_topup_displacement"],
                     "uncorrected_residual_slope_vs_topup": round(float(np.polyfit(d_topup[topup_mask], u[topup_mask], 1)[0]), 3)}
(HERE / "results/t1_alignment.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(DER / "PAT16/t1_alignment_fields.npz", d_topup=d_topup, mask=topup_mask, **fields)
print(json.dumps(res["validation"]))
