"""Model fit of corrected image sets on one grid (compare_real.py's measures), for sets made on other grids: TORTOISE
writes an axis-aligned grid of its own, so its output is resampled onto the scan's grid (cubic, by world coordinates).
That adds an interpolation, which smooths noise and lowers residuals; eddy's output is sent through the same round trip
(scan grid -> TORTOISE's grid -> scan grid) so the bias can be read off and TORTOISE compared at equal smoothing.

    uv run --group dipy bench/motion_eddy/score_sets.py <dir> <tortoise.nii>
"""
import json, sys
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import map_coordinates

D = Path(sys.argv[1]); tort = nib.load(sys.argv[2])
ref = nib.load(D / "imain.nii.gz")
mask = np.asarray(nib.load(D / "mask.nii.gz").dataobj) > 0
bval = np.loadtxt(D / "bvals"); bvec = np.loadtxt(D / "bvecs"); G = len(bval)


def resample(data, src_aff, dst_aff, dst_shape):
    """data (X, Y, Z, G) on src_aff's grid, sampled at dst's voxel centers (cubic spline, 0 outside)."""
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in dst_shape], indexing="ij"), 0).reshape(3, -1)
    w = dst_aff @ np.vstack([ijk, np.ones(ijk.shape[1])])
    v = np.linalg.inv(src_aff) @ w
    out = np.empty(tuple(dst_shape) + (data.shape[-1],), np.float32)
    for k in range(data.shape[-1]):
        out[..., k] = map_coordinates(data[..., k], v[:3], order=3, mode="constant", cval=0.0).reshape(dst_shape)
    return out


# the measures (compare_real.py's)
from dipy.reconst.shm import real_sh_descoteaux
isb0 = bval < 50
gd = np.zeros((G, 3)); gd[~isb0] = (bvec[:, ~isb0] / np.linalg.norm(bvec[:, ~isb0], axis=0)).T
Hh = np.zeros((G, G)); groups = [np.flatnonzero(isb0)]
for b in sorted(set(np.round(bval[~isb0] / 100) * 100)):
    groups.append(np.flatnonzero((~isb0) & (np.abs(bval - b) < 100)))
for idx in groups:
    if isb0[idx[0]]:
        Hh[np.ix_(idx, idx)] = 1 / len(idx); continue
    gg = gd[idx]
    B, m_, l_ = real_sh_descoteaux(6, np.arccos(np.clip(gg[:, 2], -1, 1)), np.arctan2(gg[:, 1], gg[:, 0]), legacy=False)
    Hh[np.ix_(idx, idx)] = B @ np.linalg.solve(B.T @ B + 0.006 * np.diag((l_ * (l_ + 1)) ** 2), B.T)
dH = np.diag(Hh); Wl = (Hh - np.diag(dH)) / (1 - dH)[:, None]
bs = bval / 1000.0
Bm = np.stack([np.ones(G), -bs * gd[:, 0] ** 2, -bs * gd[:, 1] ** 2, -bs * gd[:, 2] ** 2,
               -2 * bs * gd[:, 0] * gd[:, 1], -2 * bs * gd[:, 0] * gd[:, 2], -2 * bs * gd[:, 1] * gd[:, 2]], 1)
Bp = np.linalg.pinv(Bm)


def fa_of(Dc):
    Dm = np.stack([Dc[:, 0], Dc[:, 3], Dc[:, 4], Dc[:, 3], Dc[:, 1], Dc[:, 5], Dc[:, 4], Dc[:, 5], Dc[:, 2]], 1).reshape(-1, 3, 3)
    ev = np.linalg.eigvalsh(Dm); md = ev.mean(1, keepdims=True)
    return np.sqrt(1.5 * ((ev - md) ** 2).sum(1) / np.clip((ev ** 2).sum(1), 1e-12, None))


def score(S4, fa_mask):
    S = S4[mask].astype(np.float64)
    r_loo = S - S @ Wl.T
    coef = np.log(np.clip(S, 1.0, None)) @ Bp.T
    r_dti = S - np.exp(coef @ Bm.T)
    fa = fa_of(coef[:, 1:])
    return {"LOO residual rms, by group": {f"b{int(round(bval[i[0]]))}": round(float(np.sqrt((r_loo[:, i] ** 2).mean())), 2) for i in groups},
            "tensor fit residual rms (DW)": round(float(np.sqrt((r_dti[:, ~isb0] ** 2).mean())), 2),
            "mean FA where FA > 0.25 as acquired": round(float(fa[fa_mask].mean()), 4)}


acq = np.asarray(ref.dataobj, np.float32)
coef_a = np.log(np.clip(acq[mask].astype(np.float64), 1.0, None)) @ Bp.T
fa_mask = fa_of(coef_a[:, 1:]) > 0.25
out = {"as acquired": score(acq, fa_mask)}
eddy = np.asarray(nib.load(D / "eddy.nii.gz").dataobj, np.float32)
out["eddy"] = score(eddy, fa_mask)
tshape = tort.shape[:3]
out["eddy, round trip through TORTOISE's grid"] = score(resample(resample(eddy, ref.affine, tort.affine, tshape), tort.affine, ref.affine, ref.shape[:3]), fa_mask)
T = np.asarray(tort.dataobj, np.float32)
out["TORTOISE (quadratic), resampled to the scan's grid"] = score(resample(T, tort.affine, ref.affine, ref.shape[:3]), fa_mask)
for name, f in (("ours (GPU MI rigid anchor)", D / "v_gpumi2/ours.nii.gz"), ("ours, no anchor", D / "v_none/ours.nii.gz")):
    if f.exists():
        o = np.asarray(nib.load(f).dataobj, np.float32)
        out[name] = score(o, fa_mask)
        out[name + ", round trip"] = score(resample(resample(o, ref.affine, tort.affine, tshape), tort.affine, ref.affine, ref.shape[:3]), fa_mask)
print(json.dumps(out, indent=1))
json.dump(out, open(Path(__file__).resolve().parent.parent / "results/motion_eddy/score_sets_16974.json", "w"), indent=1)
