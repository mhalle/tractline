"""OpenNeuro ds001226 sub-PAT16 (BTC_preop, CC0) into the tracker's input: a NRRD DWI and a NRRD mask.

    python bench/pat16_prep.py [--shell 2800]

PAT16 is albula-diffusion's reference case: Siemens TrioTim 3 T, 2.5 mm isotropic, b = 700 / 1200 /
2800 (16 / 30 / 50 directions) + 6 b0, and a reversed-phase-encoding b0 pair (not used here: both
arms of every comparison skip distortion correction alike).
  - gradients: FSL bvec (image axes; x flipped when the affine's determinant is positive, as dwi.ts
    fromFsl and dcm2niix do) turned into RAS world by the affine's rotation; written with
    space right-anterior-superior and the identity measurement frame.
  - shell: the b0s and one shell (default 2800, the nearest to the b = 3000 of TractCloud's
    training tractography), as the ORG pipeline tracked one shell.
  - mask: DIPY median_otsu on the mean b0 (median_radius 4, numpass 4), as ukf_bench.py made
    HARDI's.
The rules live in prep.py (cohort.py uses them for every subject).
Writes DATA/ds001226/derived/PAT16/{dwi.nhdr, dwi.raw, mask.nrrd, prep.json}.
"""
import argparse, json
from pathlib import Path
import numpy as np, nibabel as nib
from tractline.prep import prep

ap = argparse.ArgumentParser(); ap.add_argument("--shell", type=float, default=2800)
ap.add_argument("--dwi", default=None, help="a corrected AP DWI to use instead (e.g. topup_ref.py's dwi_AP_topup.nii.gz)")
ap.add_argument("--out", default="PAT16", help="the output folder under DATA/ds001226/derived/")
args = ap.parse_args()
DATA = Path.home() / "tmp/data/tractography/ds001226"
SRC = DATA / "sub-PAT16/ses-preop/dwi/sub-PAT16_ses-preop_acq-AP_dwi"
OUT = DATA / "derived" / args.out

img = nib.load(args.dwi or str(SRC) + ".nii.gz")
info, _ = prep(np.asarray(img.dataobj), img.affine, np.loadtxt(str(SRC) + ".bval"), np.loadtxt(str(SRC) + ".bvec"), OUT, args.shell)
info = {"source": "OpenNeuro ds001226 v5.0.1 sub-PAT16 ses-preop acq-AP (CC0)" + (f", corrected: {args.dwi}" if args.dwi else ""), **info}
(OUT / "prep.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
