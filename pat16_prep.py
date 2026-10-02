"""OpenNeuro ds001226 sub-PAT16 (BTC_preop, CC0) into the tracker's input: a NRRD DWI and a NRRD mask.

    DATA/.venv/bin/python bench/tractography/pat16_prep.py [--shell 2800]

PAT16 is albula-diffusion's reference case: Siemens TrioTim 3 T, 2.5 mm isotropic, b = 700 / 1200 /
2800 (16 / 30 / 50 directions) + 6 b0, and a reversed-phase-encoding b0 pair (not used here: both
arms of every comparison skip distortion correction alike).
  - gradients: FSL bvec (image axes; x flipped when the affine's determinant is positive, as dwi.ts
    fromFsl and dcm2niix do) turned into RAS world by the affine's rotation; written with
    space right-anterior-superior and the identity measurement frame.
  - shell: the b0s and one shell (default 2800, the nearest to the b = 3000 of TractCloud's
    training tractography), as the ORG pipeline tracked one shell.
  - mask: DIPY median_otsu on the mean b0 (median_radius 4, numpass 4), as ukf_bench.py made
    HARDI's; the same mask for every arm.
Writes DATA/ds001226/derived/PAT16/{dwi.nhdr, dwi.raw, mask.nrrd, prep.json}.
"""
import argparse, json
from pathlib import Path
import numpy as np, nibabel as nib, nrrd
from dipy.segment.mask import median_otsu

ap = argparse.ArgumentParser(); ap.add_argument("--shell", type=float, default=2800)
ap.add_argument("--dwi", default=None, help="a corrected AP DWI to use instead (e.g. topup_ref.py's dwi_AP_topup.nii.gz)")
ap.add_argument("--out", default="PAT16", help="the output folder under DATA/ds001226/derived/")
args = ap.parse_args()
DATA = Path.home() / "tmp/data/tractography/ds001226"
SRC = DATA / "sub-PAT16/ses-preop/dwi/sub-PAT16_ses-preop_acq-AP_dwi"
OUT = DATA / "derived" / args.out
OUT.mkdir(parents=True, exist_ok=True)

img = nib.load(args.dwi or str(SRC) + ".nii.gz")
data = np.asarray(img.dataobj)                                            # (i, j, k, G): int16, or float after a correction
A = img.affine
bval = np.loadtxt(str(SRC) + ".bval")
bvec = np.loadtxt(str(SRC) + ".bvec")                                     # (3, G), image axes
keep = (bval < 50) | (np.abs(bval - args.shell) < 50)
data, bval, bvec = data[..., keep], bval[keep], bvec[:, keep]

M = A[:3, :3]
spacing = np.linalg.norm(M, axis=0)
R = M / spacing                                                           # columns: the axes' directions
g = bvec.copy()
if np.linalg.det(M) > 0:                                                  # FSL's convention
    g[0] *= -1
g = (R @ g).T                                                             # (G, 3) RAS
bmax = float(bval.max())
g = np.where((bval > 50)[:, None], g / np.linalg.norm(g, axis=1, keepdims=True) * np.sqrt(bval / bmax)[:, None], 0.0)

floating = not np.issubdtype(data.dtype, np.integer)
hdr = {"type": "float" if floating else "short", "dimension": 4, "space": "right-anterior-superior", "sizes": list(data.shape),
       "space directions": [M[:, 0].tolist(), M[:, 1].tolist(), M[:, 2].tolist(), [np.nan] * 3],
       "kinds": ["space", "space", "space", "list"], "endian": "little", "encoding": "raw",
       "space origin": A[:3, 3].tolist(), "measurement frame": np.eye(3).tolist(),
       "modality": "DWMRI", "DWMRI_b-value": f"{bmax:g}"}
for n, v in enumerate(g):
    hdr[f"DWMRI_gradient_{n:04d}"] = f"{v[0]:.8f} {v[1]:.8f} {v[2]:.8f}"
nrrd.write(str(OUT / "dwi.nhdr"), data.astype(np.float32 if floating else np.int16), hdr, detached_header=True)

b0 = data[..., bval < 50].astype(np.float32).mean(-1)
_, mask = median_otsu(b0, median_radius=4, numpass=4)
nrrd.write(str(OUT / "mask.nrrd"), mask.astype(np.uint8),
           {"type": "unsigned char", "dimension": 3, "space": "right-anterior-superior", "sizes": list(mask.shape),
            "space directions": [M[:, 0].tolist(), M[:, 1].tolist(), M[:, 2].tolist()], "kinds": ["space"] * 3,
            "space origin": A[:3, 3].tolist(), "encoding": "gzip"})
info = {"source": "OpenNeuro ds001226 v5.0.1 sub-PAT16 ses-preop acq-AP (CC0)" + (f", corrected: {args.dwi}" if args.dwi else ""), "shape": list(data.shape),
        "voxel_mm": [round(float(v), 3) for v in spacing], "b0": int((bval < 50).sum()),
        "shell": args.shell, "directions": int((bval > 50).sum()), "affine_det_positive": bool(np.linalg.det(M) > 0),
        "mask_voxels": int(mask.sum())}
(OUT / "prep.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
