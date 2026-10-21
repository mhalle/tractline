"""The b0s' fit to each other, per method (real scan): each corrected b0's leave-one-out residual against the mean of
the other corrected b0s (RMS over the brain mask), and the misalignment left between them, measured at the same
contrast: each corrected b0 registered rigidly to the mean of the others (mi_torch), as the displacement it makes at
the mask's edge (voxels). Also intensity: each b0's mean in the mask against the b0s' average (drift, or a scale).

    uv run --group dipy bench/motion_eddy/b0_check.py <dir>
"""
import json, math, sys
from pathlib import Path
import numpy as np, nibabel as nib, torch
from scipy.ndimage import binary_dilation, binary_erosion
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from mi_torch import register

D = Path(sys.argv[1]); dev = torch.device("mps")
bval = np.loadtxt(D / "bvals"); b0i = np.flatnonzero(bval < 50)
mask = np.asarray(nib.load(D / "mask.nii.gz").dataobj) > 0
edge = mask & ~binary_erosion(mask, iterations=1)
mk = torch.as_tensor(binary_dilation(mask, iterations=2), device=dev)
vox = [float(v) for v in nib.load(D / "imain.nii.gz").header.get_zooms()[:3]]
X, Y, Z = mask.shape; cen = np.array([(X - 1) / 2, (Y - 1) / 2, (Z - 1) / 2]); xe = np.argwhere(edge).astype(float)


def rot(w):
    cx, cy, cz = np.cos(w); sx, sy, sz = np.sin(w)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def edge_rms(d):
    p = (((xe - cen) * vox) @ rot(np.asarray(d[3:6]) * 0.01).T + np.asarray(d[:3])) / vox + cen
    return float(np.sqrt((np.linalg.norm(p - xe, axis=1) ** 2).mean()))


sets = {"as acquired": D / "imain.nii.gz", "eddy": D / "eddy.nii.gz", "TORTOISE": D / "tortoise_out2/tortoise.nii",
        "ours": D / "v_bins256/ours.nii.gz", "ours, gain": D / "v_gain/ours.nii.gz", "ours, b0 fix": D / "v_b0fix/ours.nii.gz", "ours, masked output": D / "v_maskout/ours.nii.gz"}
if "--only" in sys.argv:
    sets = {k: v for k, v in sets.items() if k in sys.argv[sys.argv.index("--only") + 1].split(";")}
out = {}
for name, f in sets.items():
    S = np.asarray(nib.load(f).dataobj, np.float32)[..., b0i]
    rows = []
    for k in range(len(b0i)):
        others = np.delete(S, k, axis=-1).mean(-1)
        r = (S[..., k] - others)[mask]
        d = register(torch.as_tensor(S[..., k], device=dev), torch.as_tensor(others, device=dev), mk, vox).cpu().numpy()
        rows.append({"volume": int(b0i[k]), "LOO residual rms": round(float(np.sqrt((r ** 2).mean())), 1),
                     "misalignment at the edge (voxels)": round(edge_rms(d), 3),
                     "rigid (mm, deg)": [round(float(v), 3) for v in d[:3]] + [round(math.degrees(float(v) * 0.01), 3) for v in d[3:]],
                     "mean in mask / b0s' mean": round(float(S[..., k][mask].mean() / S[mask].mean()), 4)})
    out[name] = rows
    print(name)
    for r in rows:
        print("   ", r, flush=True)
json.dump(out, open(HERE.parent / "results/motion_eddy/b0_check_16974.json", "w"), indent=1)
