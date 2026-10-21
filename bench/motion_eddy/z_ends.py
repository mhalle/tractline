"""Is z special? On ON-Harmony sub-16974 the brain mask spans every slice (0-71; the head is cut at both ends of the
slab) and the DW/b0 ratio drops at the slab's ends (about 0.13 and 0.17 in the end slices against 0.34 inside). A
cross-contrast registration then sees a different contrast relation at the two ends of z only. Here: the raw b2000 mean
registered to the raw b0 mean (as is: what the anchor and eddy's PEAS measure), by the GPU MI (mi_torch, which matches
dipy) and by dipy, with the mask's first and last k slices excluded, k = 0, 3, 6, 10. FSL (eddy) puts this offset at
z = +0.03 to +0.08 mm (its MI and its interspersed b0s).

    uv run --group dipy bench/motion_eddy/z_ends.py <dir>
"""
import json, math, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, nibabel as nib, torch
from scipy.ndimage import binary_dilation, gaussian_filter
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mi_torch import register

D = Path(sys.argv[1]); dev = torch.device("mps")
img = nib.load(D / "imain.nii.gz"); S = np.asarray(img.dataobj, np.float32); b = np.loadtxt(D / "bvals")
vox = [float(v) for v in img.header.get_zooms()[:3]]
mask = np.asarray(nib.load(D / "mask.nii.gz").dataobj) > 0
b0 = gaussian_filter(S[..., b < 50].mean(-1), 0.42); dw = gaussian_filter(S[..., np.abs(b - 2000) < 100].mean(-1), 0.42)
cen = [(n - 1) / 2 for n in b0.shape]
T = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)
fmt = lambda r: [round(float(v), 3) for v in r[:3]] + [round(math.degrees(float(v)), 3) for v in r[3:]]


def dipy_reg(m):
    with tempfile.TemporaryDirectory() as td:
        np.savez(os.path.join(td, "in.npz"), mov=dw.astype(np.float64), fixed=b0.astype(np.float64), mask=m, vox=np.asarray(vox), cen=np.asarray(cen))
        subprocess.run([sys.executable, str(HERE / "mi_dipy.py"), "register", os.path.join(td, "in.npz"), os.path.join(td, "out.json")], check=True, capture_output=True)
        return json.load(open(os.path.join(td, "out.json")))


if "--flatten" in sys.argv:
    # remove each image's slowly varying intensity (a z-dependent DW/b0 relation would bias a cross-contrast MI along z):
    # divide by a heavily smoothed copy (sigma in voxels), within the head
    sd = float(sys.argv[sys.argv.index("--flatten") + 1])
    head = binary_dilation(mask, iterations=4)
    def flat(v):
        num = gaussian_filter(np.where(head, v, 0.0), sd); den = gaussian_filter(head.astype(np.float32), sd)
        return np.where(head, v / np.clip(num / np.clip(den, 1e-3, None), 1e-3, None), 0.0)
    b0, dw = flat(b0), flat(dw)
out = {}
for k in ((0, 6) if "--flatten" in sys.argv else (0, 3, 6, 10)):
    m = binary_dilation(mask, iterations=2)
    if k:
        m[..., :k] = False; m[..., -k:] = False
    r = register(T(dw), T(b0), torch.as_tensor(m, device=dev), vox).cpu().tolist()
    r = r[:3] + [v * 0.01 for v in r[3:]]
    out[f"exclude {k} end slices"] = {"GPU MI (mm, deg)": fmt(r), "dipy MI (mm, deg)": fmt(dipy_reg(m))}
    print(k, json.dumps(out[f"exclude {k} end slices"]), flush=True)
json.dump(out, open(HERE.parent / ("results/motion_eddy/z_ends_16974" + (f"_flatten{sys.argv[sys.argv.index('--flatten') + 1]}" if "--flatten" in sys.argv else "") + ".json"), "w"), indent=1)
