"""Between-shell alignment on a real scan, without a truth, for each image set in a --real folder (as acquired
imain.nii.gz, eddy's eddy.nii.gz, ours.nii.gz):

  1. b2000 mean registered to the b1000 mean (similar contrasts; mi_torch, rigid + the eddy polynomial along PE): what
     is left between the DW shells, as the displacement at the brain mask's edge (RMS, p95, voxels);
  2. an extrapolated b0 (Nilsson et al. 2015), S0 ~ S1000^2 / S2000 per voxel from the shells' means (mono-exponential,
     b2000 = 2 b1000), registered to the b0 mean (nearly the same contrast): what is left between the DW shells as a
     group and the b0s;
  3. ours only: each shell's mean eddy coefficients after the anchor (the b0s' are fixed at 0), as the displacement
     they make at the edge: the common eddy field the anchor attributes to each shell.
Registrations on the device (MPS), each image set's means smoothed by 1 voxel.

    uv run --group dipy bench/motion_eddy/between_shells.py <dir>
"""
import json, math, sys
from pathlib import Path
import numpy as np, nibabel as nib, torch
from scipy.ndimage import binary_dilation, binary_erosion

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mi_torch import register

D = Path(sys.argv[1])
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
bval = np.loadtxt(D / "bvals")
mask = np.asarray(nib.load(D / "mask.nii.gz").dataobj) > 0
edge = mask & ~binary_erosion(mask, iterations=1)
mk = torch.as_tensor(binary_dilation(mask, iterations=2), device=dev)
img = nib.load(D / "imain.nii.gz")
vox = [float(v) for v in img.header.get_zooms()[:3]]
X, Y, Z = mask.shape
cen = np.array([(X - 1) / 2, (Y - 1) / 2, (Z - 1) / 2])
PE, NMON = 1, 10
xe = np.argwhere(edge).astype(float)


def rot(w):
    cx, cy, cz = np.cos(w); sx, sy, sz = np.sin(w)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def edge_disp(d):
    """d: mi_torch's (tx, ty, tz mm, 3 x 0.01 rad, eddy 10 voxels) -> the displacement at the edge (N, 3) voxels."""
    p = (((xe - cen) * vox) @ rot(np.asarray(d[3:6]) * 0.01).T + np.asarray(d[:3])) / vox + cen
    a, b, c = ((xe - cen) / cen).T
    mon = np.stack([np.ones_like(a), a, b, c, a * a, b * b, c * c, a * b, a * c, b * c], -1)
    if len(d) > 6:
        p[:, PE] += mon @ np.asarray(d[6:])
    return p - xe


def summarize(d):
    n = np.linalg.norm(edge_disp(d), axis=-1)
    return {"edge rms": round(float(np.sqrt((n ** 2).mean())), 3), "edge p95": round(float(np.quantile(n, 0.95)), 3),
            "rigid (mm, deg)": [round(float(v), 3) for v in d[:3]] + [round(math.degrees(float(v) * 0.01), 3) for v in d[3:6]]}


def smooth(v, sd=1.0 / 2.3548):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(v, sd)


out = {}
for name, f in (("as acquired", "imain.nii.gz"), ("eddy", "eddy.nii.gz"), ("ours", "ours.nii.gz")):
    if not (D / f).exists():
        continue
    S = np.asarray(nib.load(D / f).dataobj, np.float32)
    m0 = smooth(S[..., bval < 50].mean(-1)); m1 = smooth(S[..., np.abs(bval - 1000) < 100].mean(-1))
    m2 = smooth(S[..., np.abs(bval - 2000) < 100].mean(-1))
    T = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)
    r12 = register(T(m2), T(m1), mk, vox, pe_axis=PE, nmon=NMON).cpu().tolist()
    s0 = np.where(mask, m1 ** 2 / np.clip(m2, 1.0, None), 0.0)
    s0 = np.clip(s0, 0, np.quantile(m0[mask], 0.999) * 2)
    r0 = register(T(s0), T(m0), mk, vox, pe_axis=PE, nmon=NMON).cpu().tolist()
    out[name] = {"b2000 mean -> b1000 mean": summarize(r12), "extrapolated b0 -> b0 mean": summarize(r0)}
    print(name, json.dumps(out[name]), flush=True)
th_f = D / "ours_theta.npz"
if th_f.exists():
    Tn = np.load(th_f); th, grp = Tn["th"], Tn["groups"]
    common = {}
    for k in sorted(set(grp.tolist())):
        gi = np.flatnonzero(grp == k)
        e = th[gi, 6:].mean(0)
        dd = [0.0] * 6 + e.tolist()
        n = np.linalg.norm(edge_disp(dd), axis=-1)
        common[f"group {k} (b={int(round(bval[gi[0]]))})"] = {"edge rms": round(float(np.sqrt((n ** 2).mean())), 3),
                                                              "edge p95": round(float(np.quantile(n, 0.95)), 3),
                                                              "coefficients": [round(float(v), 3) for v in e]}
    out["ours: each shell's mean eddy field after the anchor (voxels at the edge)"] = common
    print(json.dumps(common), flush=True)
json.dump(out, open(Path(__file__).resolve().parent.parent / "results/motion_eddy/between_shells_16974.json", "w"), indent=1)
