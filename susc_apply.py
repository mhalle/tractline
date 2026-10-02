"""Apply our susceptibility field to PAT16's AP DWI: applytopup's equivalent (--inindex=1 --method=jac:
the field, the AP phase-encoding direction and readout time, the Jacobian; volume 1's motion, which is
zero), with _susc.unwarp (trilinear; applytopup uses splines).

    DATA/.venv/bin/python bench/tractography/susc_apply.py [--field field_hz_scaled.nii.gz]

Reads DATA/ds001226/derived/PAT16/susc/<field> (susc_check.py) and the AP DWI; writes
DATA/ds001226/derived/PAT16/susc/dwi_AP_ours.nii.gz, to be prepared by pat16_prep.py --dwi ... --out PAT16_ours.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, nibabel as nib, torch
import _susc as S

ap = argparse.ArgumentParser(); ap.add_argument("--field", default="field_hz_scaled.nii.gz")
ap.add_argument("--out", default="dwi_AP_ours.nii.gz"); args = ap.parse_args()
TD = Path.home() / "tmp/data/tractography/ds001226"
SRC = TD / "sub-PAT16/ses-preop/dwi/sub-PAT16_ses-preop_acq-AP_dwi"
SU = TD / "derived/PAT16/susc"
side = json.loads(Path(str(SRC) + ".json").read_text())
pe = {"j": (1, 1), "j-": (1, -1), "i": (0, 1), "i-": (0, -1), "k": (2, 1), "k-": (2, -1)}[side["PhaseEncodingDirection"]]
img = nib.load(str(SRC) + ".nii.gz")
dwi = np.asarray(img.dataobj, dtype=np.float64)
h = torch.as_tensor(np.asarray(nib.load(SU / args.field).dataobj, dtype=np.float64))
vox = torch.as_tensor(np.asarray(img.header.get_zooms()[:3], float), dtype=torch.float64)
t0 = time.time()
dh = torch.gradient(h, dim=pe[0])[0]
V = dwi.shape[-1]
eye = [torch.eye(3, dtype=torch.float64)] * V
zero = [torch.zeros(3, dtype=torch.float64)] * V
scale = torch.full((V,), pe[1] * side["TotalReadoutTime"], dtype=torch.float64)
out = S.unwarp(torch.as_tensor(np.moveaxis(dwi, -1, 0)), h, dh, pe[0], scale, eye, zero, vox).numpy()
out = np.moveaxis(np.clip(out, 0, None), 0, -1).astype(np.float32)
nib.save(nib.Nifti1Image(out, img.affine), SU / args.out)
print(json.dumps({"volumes": V, "seconds": round(time.time() - t0, 1), "out": str(SU / args.out)}))
