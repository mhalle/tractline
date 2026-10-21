"""The prototype against eddy on a real scan (no truth): proto.py --all-shells --fast --metal --real <dir> and eddy run in
the same folder (imain, mask, bvals, bvecs; eddy's output basename "eddy", with --dfields).

  - agreement: each volume's displacement at the brain mask's edge, ours (from ours_theta.npz) against eddy's
    (eddy.eddy_displacement_fields.*), the transform common to all volumes removed from each; RMS and p95, voxels;
    split by group (b0s, shells), and the rotated b-vectors' angles between the two;
  - model fit, per image set (as acquired, eddy's, ours): each shell's leave-one-out SH prediction residual (order 6,
    Laplace-Beltrami l = 0.006, the b0s against the mean of the others), and a single-tensor log-linear fit's residual,
    RMS over the brain mask; and mean FA where FA > 0.25 as acquired.
Both corrected sets are interpolated once (eddy: its splines; ours: cubic B-splines), which smooths noise a little; the
as-acquired set is not.

    uv run --group dipy bench/motion_eddy/compare_real.py <dir>
"""
import glob, json, sys
from pathlib import Path
import numpy as np, nibabel as nib

D = Path(sys.argv[1])
T = np.load(D / "ours_theta.npz")
th, PE, cen, vox, nmon, grp = T["th"], int(T["pe_axis"]), T["cen"], T["vox"], int(T["nmon"]), T["groups"]
mask = np.asarray(nib.load(D / "mask.nii.gz").dataobj) > 0
from scipy.ndimage import binary_erosion
edge = mask & ~binary_erosion(mask, iterations=1)
bval = np.loadtxt(D / "bvals"); bvec = np.loadtxt(D / "bvecs")
G = len(bval)


def rot(w):
    cx, cy, cz = np.cos(w); sx, sy, sz = np.sin(w)
    return (np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]) @ np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
            @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]))


def mono(x):
    a, b, c = ((x - cen) / cen).T
    return np.stack([np.ones_like(a), a, b, c] + ([a * a, b * b, c * c, a * b, a * c, b * c] if nmon == 10 else []), -1)


def disp(t, x):
    p = (((x - cen) * vox) @ rot(t[3:6]).T + t[:3]) / vox + cen
    p[:, PE] += mono(x) @ t[6:]
    return p - x


out = {}
xe = np.argwhere(edge).astype(float); ie = tuple(np.argwhere(edge).T)
ours = np.stack([disp(t, xe) for t in th])
files = sorted(glob.glob(str(D / "eddy.eddy_displacement_fields.*.nii*")))
if len(files) == G:
    eddy = np.stack([np.asarray(nib.load(f).dataobj)[ie] / vox for f in files])
    a, b = ours - ours.mean(0, keepdims=True), eddy - eddy.mean(0, keepdims=True)
    n = np.linalg.norm(a - b, axis=-1)
    out["agreement at the edge (voxels)"] = {"rms": round(float(np.sqrt((n ** 2).mean())), 3), "p95": round(float(np.quantile(n, 0.95)), 3),
                                             "ours motion rms": round(float(np.sqrt((np.linalg.norm(a, axis=-1) ** 2).mean())), 3),
                                             "eddy motion rms": round(float(np.sqrt((np.linalg.norm(b, axis=-1) ** 2).mean())), 3)}
    by = {}
    for k in sorted(set(grp.tolist())):
        gi = np.flatnonzero(grp == k)
        ng = np.linalg.norm((a - b)[gi], axis=-1); mg = np.linalg.norm((a - b)[gi].mean(0), axis=-1)
        by[f"group {k} (n={len(gi)}, b={int(round(bval[gi[0]]))})"] = {"rms": round(float(np.sqrt((ng ** 2).mean())), 3),
                                                                      "group mean difference rms": round(float(np.sqrt((mg ** 2).mean())), 3)}
    out["agreement by group"] = by
try:
    re_, ro = np.loadtxt(D / "eddy.eddy_rotated_bvecs"), np.loadtxt(D / "ours_rotated_bvecs")
    dw = bval > 50
    c = np.abs((re_[:, dw] * ro[:, dw]).sum(0) / np.linalg.norm(re_[:, dw], axis=0) / np.linalg.norm(ro[:, dw], axis=0))
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    c0 = np.abs((re_[:, dw] * bvec[:, dw]).sum(0) / np.linalg.norm(re_[:, dw], axis=0) / np.linalg.norm(bvec[:, dw], axis=0))
    out["rotated b-vectors, ours against eddy's (deg)"] = {"rms": round(float(np.sqrt((ang ** 2).mean())), 3), "max": round(float(ang.max()), 3),
                                                           "eddy's rotation from the input, rms": round(float(np.sqrt((np.degrees(np.arccos(np.clip(c0, -1, 1))) ** 2).mean())), 3)}
except OSError as e:
    out["bvecs"] = str(e)

# model fit per image set
from dipy.reconst.shm import real_sh_descoteaux
isb0 = bval < 50
gdir = np.zeros((G, 3)); gdir[~isb0] = (bvec[:, ~isb0] / np.linalg.norm(bvec[:, ~isb0], axis=0)).T
Hh = np.zeros((G, G)); groups = [np.flatnonzero(isb0)]
for b in sorted(set(np.round(bval[~isb0] / 100) * 100)):
    groups.append(np.flatnonzero((~isb0) & (np.abs(bval - b) < 100)))
for idx in groups:
    if isb0[idx[0]]:
        Hh[np.ix_(idx, idx)] = 1 / len(idx); continue
    gg = gdir[idx]
    B, m_, l_ = real_sh_descoteaux(6, np.arccos(np.clip(gg[:, 2], -1, 1)), np.arctan2(gg[:, 1], gg[:, 0]), legacy=False)
    Hh[np.ix_(idx, idx)] = B @ np.linalg.solve(B.T @ B + 0.006 * np.diag((l_ * (l_ + 1)) ** 2), B.T)
dH = np.diag(Hh); Wl = (Hh - np.diag(dH)) / (1 - dH)[:, None]
bs = bval / 1000.0
Bm = np.stack([np.ones(G), -bs * gdir[:, 0] ** 2, -bs * gdir[:, 1] ** 2, -bs * gdir[:, 2] ** 2,
               -2 * bs * gdir[:, 0] * gdir[:, 1], -2 * bs * gdir[:, 0] * gdir[:, 2], -2 * bs * gdir[:, 1] * gdir[:, 2]], 1)
Bp = np.linalg.pinv(Bm)


def fa_of(Dc):
    Dm = np.stack([Dc[:, 0], Dc[:, 3], Dc[:, 4], Dc[:, 3], Dc[:, 1], Dc[:, 5], Dc[:, 4], Dc[:, 5], Dc[:, 2]], 1).reshape(-1, 3, 3)
    ev = np.linalg.eigvalsh(Dm); md = ev.mean(1, keepdims=True)
    return np.sqrt(1.5 * ((ev - md) ** 2).sum(1) / np.clip((ev ** 2).sum(1), 1e-12, None))


sets = {"as acquired": D / "imain.nii.gz", "eddy": D / "eddy.nii.gz", "ours": D / "ours.nii.gz"}
fa_mask = None
fit = {}
for name, f in sets.items():
    if not f.exists():
        continue
    S = np.asarray(nib.load(f).dataobj, np.float64)[mask]                             # (voxels, G)
    r_loo = S - S @ Wl.T
    logS = np.log(np.clip(S, 1.0, None)); coef = logS @ Bp.T
    r_dti = S - np.exp(coef @ Bm.T)
    fa = fa_of(coef[:, 1:])
    if fa_mask is None:
        fa_mask = fa > 0.25                                                               # from the scan as acquired
    fit[name] = {"LOO prediction residual rms, by group": {f"b{int(round(bval[i[0]]))}": round(float(np.sqrt((r_loo[:, i] ** 2).mean())), 2) for i in groups},
                 "tensor fit residual rms (DW)": round(float(np.sqrt((r_dti[:, ~isb0] ** 2).mean())), 2),
                 "mean FA where FA > 0.25 as acquired": round(float(fa[fa_mask].mean()), 4)}
out["model fit"] = fit
print(json.dumps(out, indent=1))
json.dump(out, open(Path(__file__).resolve().parent.parent / "results/motion_eddy/compare_real_16974.json", "w"), indent=1)
