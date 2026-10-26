"""eddy's displacement fields on the synthetic scan (proto.py --save-synth) against the known truth, scored as proto.py
scores itself: per DW volume, the displacement at the brain mask's edge, the transform common to all DW volumes removed
(the gauge), RMS and p95 over volumes and edge voxels, in voxels. Also the rotation error of eddy's rotated b-vectors.

    uv run --group dipy bench/motion_eddy/compare_eddy.py $TRACTOGRAPHY_DATA/derived/motion_eddy_synth/eddy_slm_none [...]
"""
import glob, json, sys
import numpy as np, nibabel as nib
from pathlib import Path
from tractline.data import DATA                                   # $TRACTOGRAPHY_DATA
OUT = Path(__file__).resolve().parent.parent / "results/motion_eddy"
SYNTH = DATA / "derived/motion_eddy_synth"                     # proto.py --save-synth; eddy runs beside it

D = str(SYNTH)
T = np.load(f"{D}/truth.npz")
order, shell, PE, cen, vox, nmon, edge = T["order"], T["shell"], int(T["pe_axis"]), T["cen"], T["vox"], int(T["nmon"]), T["edge"]
th = T["th_dw"]
X, Y, Z = edge.shape
pos = np.flatnonzero(np.isin(order, shell))                       # DW volumes' positions in imain
xe = np.argwhere(edge).astype(float)                               # (N, 3) voxels
half = cen


def rot(w):
    cx, cy, cz = np.cos(w); sx, sy, sz = np.sin(w)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def monomials(x):
    u = (x - cen) / half
    a, b, c = u.T
    cols = [np.ones_like(a), a, b, c] + ([a * a, b * b, c * c, a * b, a * c, b * c] if nmon == 10 else [])
    return np.stack(cols, -1)


def true_disp(v):                                                  # phi(x) - x, voxels, (N, 3)
    t = th[v]
    p = ((xe - cen) * vox) @ rot(t[3:6]).T + t[:3]
    p = p / vox + cen
    p[:, PE] += monomials(xe) @ t[6:]
    return p - xe


def score(d):                                                      # d (G, N, 3) voxels
    d = d - d.mean(0, keepdims=True)
    n = np.linalg.norm(d, axis=-1)
    return round(float(np.sqrt((n ** 2).mean())), 3), round(float(np.quantile(n, 0.95)), 3)


truth = np.stack([true_disp(v) for v in range(len(shell))])
out = {"truth_displacement_rms_p95_vox": score(truth)}
idx = tuple(np.argwhere(edge).T)
for base in sys.argv[1:]:
    files = sorted(glob.glob(f"{base}.eddy_displacement_fields*.nii*"))
    if len(files) == 1:
        F = np.asarray(nib.load(files[0]).dataobj)                 # (X, Y, Z, 3 * V) or (X, Y, Z, 3, V)
        F = F.reshape(X, Y, Z, -1)
        V = F.shape[-1] // 3
        fields = [F[..., 3 * k:3 * k + 3] for k in range(V)]
    else:
        fields = [np.asarray(nib.load(f).dataobj) for f in files]  # one (X, Y, Z, 3) per volume
    est = np.stack([fields[k][idx] / vox for k in pos])            # (G, N, 3) voxels
    best = None
    for sx in (1, -1):                                             # FSL's x convention, and the field's direction
        for sg in (1, -1):
            e = sg * est * np.array([sx, 1, 1])
            sc = score(e - truth)
            if best is None or sc[0] < best[1][0]:
                best = ((sx, sg), sc)
    res = {"edge_error_rms_p95_vox": best[1], "sign_convention (x, field)": best[0], "fields_files": len(files)}
    # the b0s too: the error by group (b0s, DW shell) after the common transform of all volumes is removed, as
    # proto.py --all-shells scores itself; the b0s' truth from th_b0 (rigid only)
    sx, sg = best[0]
    b0pos = np.flatnonzero(~np.isin(order, shell))
    th_b0 = T["th_b0"]

    def disp_b0(k):
        t = th_b0[k]
        p_ = ((xe - cen) * vox) @ rot(t[3:6]).T + t[:3]
        return p_ / vox + cen - xe
    truth_all = np.concatenate([np.stack([disp_b0(k) for k in range(len(b0pos))]), truth])
    est_b0 = np.stack([fields[k][idx] / vox for k in b0pos]) * sg * np.array([sx, 1, 1])
    est_all = np.concatenate([est_b0, sg * est * np.array([sx, 1, 1])])
    d = est_all - truth_all
    d = d - d.mean(0, keepdims=True)
    groups = [np.arange(len(b0pos)), np.arange(len(b0pos), len(b0pos) + len(pos))]
    within, means = [], []
    for gi in groups:
        m = d[gi].mean(0, keepdims=True); means.append(m)
        within.append(np.linalg.norm(d[gi] - m, axis=-1).ravel())
    mm_ = np.concatenate(means)
    n_all = np.linalg.norm(d, axis=-1)
    res["with_b0s"] = {"edge_error_rms": round(float(np.sqrt((n_all ** 2).mean())), 3),
                       "within_group_rms": round(float(np.sqrt((np.concatenate(within) ** 2).mean())), 3),
                       "between_group_rms": round(float(np.sqrt((np.linalg.norm(mm_ - mm_.mean(0, keepdims=True), axis=-1) ** 2).mean())), 3),
                       "groups": [len(g_) for g_ in groups]}
    try:
        rb = np.loadtxt(f"{base}.eddy_rotated_bvecs")[:, pos]
        b0v = np.loadtxt(f"{D}/bvecs")[:, pos]
        ang = []
        for conv in ("RT", "R", "FRTF", "FRF"):
            F_ = np.diag([-1, 1, 1])
            a = []
            for k in range(len(shell)):
                R = rot(th[k, 3:6])
                M = {"RT": R.T, "R": R, "FRTF": F_ @ R.T @ F_, "FRF": F_ @ R @ F_}[conv]
                g = M @ b0v[:, k]; g /= np.linalg.norm(g)
                c = abs(float(np.clip(g @ rb[:, k] / np.linalg.norm(rb[:, k]), -1, 1)))
                a.append(np.degrees(np.arccos(c)))
            ang.append((conv, round(float(np.sqrt(np.mean(np.square(a)))), 3), round(float(np.median(a)), 3)))
        conv = min(ang, key=lambda z: z[1])
        res["bvec_rotation_error_deg (convention, rms, median)"] = conv
        res["bvec_rotation_true_deg_rms"] = round(float(np.sqrt(np.mean([np.degrees(np.linalg.norm(th[k, 3:6])) ** 2 for k in range(len(shell))]))), 3)
    except OSError as e:
        res["bvecs"] = str(e)
    out[base.split("/")[-1]] = res
print(json.dumps(out, indent=1))
json.dump(out, open(OUT / "compare_eddy.json", "w"), indent=1)
