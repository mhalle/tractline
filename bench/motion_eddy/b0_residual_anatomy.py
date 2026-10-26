"""Where and why our corrected b0s fit each other less well than eddy's and TORTOISE's (real scan), with alignment at
their level (b0_check.py): per method,
  - intensity scale: the b0s' mean in the brain mask (an output rescaled by a few % rescales its residual);
  - smoothness: the high-frequency share, |v - G_1voxel(v)| RMS over v's RMS in the mask (an interpolation that
    smooths lowers residuals without aligning better), for the b0s and, as a reference, the raw scan's;
  - each b0's leave-one-out residual (against the mean of the other corrected b0s), relative to the mean intensity;
  - for the worst volume, the residual by slice (z) and in the mask's edge band against its interior.

    uv run --group dipy bench/motion_eddy/b0_residual_anatomy.py <dir>
"""
import json, sys
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import gaussian_filter, binary_erosion

D = Path(sys.argv[1])
bval = np.loadtxt(D / "bvals"); b0i = np.flatnonzero(bval < 50)
mask = np.asarray(nib.load(D / "mask.nii.gz").dataobj) > 0
inner = binary_erosion(mask, iterations=3); band = mask & ~inner
sets = {"as acquired": D / "imain.nii.gz", "eddy": D / "eddy.nii.gz", "TORTOISE": D / "tortoise_out2/tortoise.nii",
        "ours": D / "v_b0fix/ours.nii.gz"}
out, R = {}, {}
for name, f in sets.items():
    S = np.asarray(nib.load(f).dataobj, np.float32)[..., b0i]
    mu = float(S[mask].mean())
    hf = [float(np.sqrt(((S[..., k] - gaussian_filter(S[..., k], 1.0))[mask] ** 2).mean()) / np.sqrt((S[..., k][mask] ** 2).mean())) for k in range(len(b0i))]
    res = []
    for k in range(len(b0i)):
        r = S[..., k] - np.delete(S, k, -1).mean(-1)
        R[(name, k)] = r
        res.append(round(float(np.sqrt((r[mask] ** 2).mean())), 1))
    out[name] = {"b0 mean in mask": round(mu, 1), "high-frequency share (mean over b0s)": round(float(np.mean(hf)), 4),
                 "LOO residual by b0": res, "LOO residual / mean intensity (mean over b0s, %)": round(100 * float(np.mean(res)) / mu, 3)}
    print(name, json.dumps(out[name]), flush=True)
k = 2                                                                              # volume 42, the largest gap to eddy
for name in ("ours", "eddy", "TORTOISE"):
    r = R[(name, k)]
    out[f"{name}: volume {b0i[k]} residual"] = {"edge band": round(float(np.sqrt((r[band] ** 2).mean())), 1),
                                               "interior": round(float(np.sqrt((r[inner] ** 2).mean())), 1),
                                               "by slice (every 6th)": [round(float(np.sqrt((r[..., z][mask[..., z]] ** 2).mean())), 1) if mask[..., z].any() else None
                                                                        for z in range(0, mask.shape[2], 6)]}
    print(name, f"volume {b0i[k]}", json.dumps(out[f"{name}: volume {b0i[k]} residual"]), flush=True)
d = np.abs(R[("ours", k)]) - np.abs(R[("eddy", k)])
zs = [round(float(d[..., z][mask[..., z]].mean()), 2) if mask[..., z].any() else None for z in range(mask.shape[2])]
out["ours - eddy, |residual| by slice, volume 42"] = zs
print("ours - eddy |residual| by slice:", zs, flush=True)
json.dump(out, open(Path(__file__).resolve().parent.parent / "results/motion_eddy/b0_residual_anatomy_16974.json", "w"), indent=1)
