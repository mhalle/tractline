"""eddy's displacement fields on the full synthetic scan (proto.py --all-shells --save-synth), scored as proto.py
--all-shells scores itself: the edge displacement error after the transform common to all volumes is removed, overall,
within each group (b0s, each shell) and between the groups' means; voxels.

    uv run --group dipy bench/motion_eddy/compare_eddy_all.py <synth dir> <eddy out basename>
"""
import glob, json, sys, math
import numpy as np, nibabel as nib

D, base = sys.argv[1], sys.argv[2]
T = np.load(f"{D}/truth.npz")
th, grp, PE, cen, vox, nmon, edge = T["th"], T["groups"], int(T["pe_axis"]), T["cen"], T["vox"], int(T["nmon"]), T["edge"]
xe = np.argwhere(edge).astype(float); idx = tuple(np.argwhere(edge).T)


def rot(w):
    cx, cy, cz = np.cos(w); sx, sy, sz = np.sin(w)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def mono(x):
    a, b, c = ((x - cen) / cen).T
    return np.stack([np.ones_like(a), a, b, c] + ([a * a, b * b, c * c, a * b, a * c, b * c] if nmon == 10 else []), -1)


def disp(t):
    p = (((xe - cen) * vox) @ rot(t[3:6]).T + t[:3]) / vox + cen
    p[:, PE] += mono(xe) @ t[6:]
    return p - xe


truth = np.stack([disp(t) for t in th])
files = sorted(glob.glob(f"{base}.eddy_displacement_fields*.nii*"))
est = np.stack([np.asarray(nib.load(f).dataobj)[idx] / vox for f in files])
d = est - truth
d = d - d.mean(0, keepdims=True)
within, means = [], []
for k in sorted(set(grp.tolist())):
    g = np.flatnonzero(grp == k)
    m = d[g].mean(0, keepdims=True); means.append(m)
    within.append(np.linalg.norm(d[g] - m, axis=-1).ravel())
mm = np.concatenate(means)
n = np.linalg.norm(d, axis=-1)
out = {"volumes": len(files), "edge_error_rms_p95": [round(float(np.sqrt((n ** 2).mean())), 3), round(float(np.quantile(n, 0.95)), 3)],
       "within_shell_rms": round(float(np.sqrt((np.concatenate(within) ** 2).mean())), 3),
       "between_shell_rms": round(float(np.sqrt((np.linalg.norm(mm - mm.mean(0, keepdims=True), axis=-1) ** 2).mean())), 3),
       "groups": [int((grp == k).sum()) for k in sorted(set(grp.tolist()))]}
print(json.dumps(out))
