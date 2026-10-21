"""Which of eddy's PEAS choices moves the cross-contrast z offset? The raw b2000 mean registered to the raw b0 mean (as
z_ends.py) by mi_torch.register_fsl, each choice switched alone and all together. FSL: z = +0.03 to +0.08 mm.

    uv run --group dipy bench/motion_eddy/fsl_mi_test.py <dir>
"""
import json, math, sys
from pathlib import Path
import numpy as np, nibabel as nib, torch
from scipy.ndimage import binary_dilation
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mi_torch import register, register_fsl

D = Path(sys.argv[1]); dev = torch.device("mps")
img = nib.load(D / "imain.nii.gz"); S = np.asarray(img.dataobj, np.float32); b = np.loadtxt(D / "bvals")
vox = [float(v) for v in img.header.get_zooms()[:3]]
mask = torch.as_tensor(binary_dilation(np.asarray(nib.load(D / "mask.nii.gz").dataobj) > 0, iterations=2), device=dev)
b0 = torch.as_tensor(S[..., b < 50].mean(-1), device=dev); dw = torch.as_tensor(S[..., np.abs(b - 2000) < 100].mean(-1), device=dev)
fmt = lambda r: [round(float(v), 3) for v in r[:3]] + [round(math.degrees(float(v) * 0.01), 3) for v in r[3:]]
cases = {
    "ours (register: brain mask, 32 bins, cubic Parzen, one-way)": lambda: register(dw, b0, mask, vox),
    "fsl-like, brain mask only changed back": lambda: register_fsl(dw, b0, mask, vox, mask_mode="brain"),
    "fsl-like, 32 bins": lambda: register_fsl(dw, b0, mask, vox, bins=32),
    "fsl-like, cubic Parzen": lambda: register_fsl(dw, b0, mask, vox, parzen="cubic"),
    "fsl-like, one-way": lambda: register_fsl(dw, b0, mask, vox, symmetric=False),
    "fsl-like, full range (no robust)": lambda: register_fsl(dw, b0, mask, vox, robust=(0.0, 1.0)),
    "fsl-like (FOV mask, 256 bins, linear Parzen, symmetric, robust range)": lambda: register_fsl(dw, b0, mask, vox),
    "ours with only the FOV mask": lambda: register_fsl(dw, b0, mask, vox, bins=32, parzen="cubic", symmetric=False, robust=(0.0, 1.0)),
}
out = {}
for k, f in cases.items():
    r = f().cpu().tolist(); out[k] = fmt(r)
    print(k.ljust(72), out[k], flush=True)
json.dump(out, open(HERE.parent / "results/motion_eddy/fsl_mi_test_16974.json", "w"), indent=1)
