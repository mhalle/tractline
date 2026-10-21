"""Where each method puts the b0s against the DW shells, measured across methods at the same contrast (accurate:
mi_torch_check): each method's corrected b0 mean registered to eddy's corrected b0 mean, and its corrected DW mean
(b1000 and b2000 together) to eddy's DW mean, rigid, mi_torch on the device. A method's b0 placement relative to its DW
shells, against eddy's, is the difference of the two (DW shift - b0 shift), reported as the displacement it makes at
the brain mask's edge (RMS, p95, voxels) and as rigid parameters.

    uv run --group dipy bench/motion_eddy/placement.py <dir>
"""
import json, math, sys
from pathlib import Path
import numpy as np, nibabel as nib, torch
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mi_torch import register

D = Path(sys.argv[1]); dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
bval = np.loadtxt(D / "bvals"); mask = np.asarray(nib.load(D / "mask.nii.gz").dataobj) > 0
edge = mask & ~binary_erosion(mask, iterations=1)
mk = torch.as_tensor(binary_dilation(mask, iterations=2), device=dev)
vox = [float(v) for v in nib.load(D / "imain.nii.gz").header.get_zooms()[:3]]
X, Y, Z = mask.shape; cen = np.array([(X - 1) / 2, (Y - 1) / 2, (Z - 1) / 2]); xe = np.argwhere(edge).astype(float)


def rot(w):
    cx, cy, cz = np.cos(w); sx, sy, sz = np.sin(w)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def disp(d):
    p = (((xe - cen) * vox) @ rot(np.asarray(d[3:6]) * 0.01).T + np.asarray(d[:3])) / vox + cen
    return p - xe


def means(f):
    S = np.asarray(nib.load(f).dataobj, np.float32)
    return gaussian_filter(S[..., bval < 50].mean(-1), 0.42), gaussian_filter(S[..., bval >= 50].mean(-1), 0.42)


T = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)
ref_b0, ref_dw = means(D / "eddy.nii.gz")
sets = {"ours, GPU MI rigid anchor": D / "v_gpumi2/ours.nii.gz", "ours, no anchor": D / "v_none/ours.nii.gz",
        "ours, dipy MI anchor": D / "v_dipy/ours.nii.gz", "TORTOISE": D / "tortoise_out2/tortoise.nii", "as acquired": D / "imain.nii.gz"}
out = {}
for name, f in sets.items():
    if not Path(f).exists():
        continue
    b0, dw = means(f)
    rb = register(T(b0), T(ref_b0), mk, vox).cpu().numpy(); rd = register(T(dw), T(ref_dw), mk, vox).cpu().numpy()
    rel = disp(rd) - disp(rb)                                                           # DW placement minus b0 placement, vs eddy's
    n = np.linalg.norm(rel, axis=-1)
    out[name] = {"b0 placement vs eddy's, relative to the DW shells: edge rms, p95 (voxels)": [round(float(np.sqrt((n ** 2).mean())), 3), round(float(np.quantile(n, 0.95)), 3)],
                 "b0 mean vs eddy's (mm, deg)": [round(float(v), 3) for v in rb[:3]] + [round(math.degrees(float(v) * 0.01), 3) for v in rb[3:]],
                 "DW mean vs eddy's (mm, deg)": [round(float(v), 3) for v in rd[:3]] + [round(math.degrees(float(v) * 0.01), 3) for v in rd[3:]]}
    print(name, json.dumps(out[name]), flush=True)
json.dump(out, open(Path(__file__).resolve().parent.parent / "results/motion_eddy/placement_16974.json", "w"), indent=1)
