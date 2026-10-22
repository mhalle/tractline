"""Motion estimates on the SHORELine benchmark phantoms (Cieslak et al.; Fiberfox; OSF 38vce, BSD-3) against their
ground-truth motion, scored as proto.py scores itself: the displacement at the brain's edge, after the transform
common to all volumes (the output frame) is removed, overall, within each group (b0s, each shell) and between the
groups' means; voxels.

The ground truth is rigid (Fiberfox moved the fibers before simulating). Its convention, found by registering each
noise-free moved volume to its no-motion copy (`center.py` in the scratch notes; 0.11 deg RMS): the rotation is
Rx Ry Rz of the listed angles (degrees), in the voxel-axis frame, about a fixed center; the translation is added in
the same frame. The center is fitted once from all noise-free high-motion volumes (--center, voxel-index mm).

Ours: the rigid part of ours_theta.npz (the PE offset split as eddy splits it) and, separately, the whole map with the
eddy terms. eddy: its displacement fields (motion and eddy together), and its rigid parameters in the convention
that fits best (--eddy-rigid).

    python score_phantom.py <phantom data dir> <scheme> <noise> <motion> --center cx,cy,cz [--ours DIR] [--eddy DIR] [--tag T]
"""
import glob, itertools, json, sys
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import binary_erosion

R, S, NOISE, MOT = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
arg = lambda k: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else None
c = np.array([float(v) for v in arg("--center").split(",")])
base = R / f"{NOISE}/{MOT}/sub-{S}/dwi/sub-{S}_acq-{NOISE}X{MOT}_dwi"
img = nib.load(str(base) + ".nii.gz")
vox = np.array(img.header.get_zooms()[:3], float); shape = img.shape[:3]; cen = (np.array(shape) - 1) / 2
bval = np.loadtxt(str(base) + ".bval"); G = len(bval)
gt = np.loadtxt(R / f"ground_truth_motion/sub-{S}_acq-{NOISE}_run-{MOT}_dwi_motion.txt")
ref = np.asarray(nib.load(str(R / f"{NOISE}/nomotion/sub-{S}/dwi/sub-{S}_acq-{NOISE}Xnomotion_dwi.nii.gz")).dataobj, np.float32)[..., 0]
mask = ref > 0.1 * np.percentile(ref[ref > 0], 95)
edge = mask & ~binary_erosion(mask)
xe = np.argwhere(edge).astype(float); ie = tuple(np.argwhere(edge).T)
groups = [np.flatnonzero(bval < 50)] + [np.flatnonzero((bval >= 50) & (np.abs(bval - b) < 100)) for b in sorted(set(np.round(bval[bval >= 50] / 100) * 100))]


def Rx(a): return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
def Ry(a): return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
def Rz(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
def euler(order, w): return np.linalg.multi_dot([{"x": Rx, "y": Ry, "z": Rz}[a](w["xyz".index(a)]) for a in order])


def truth_disp(k):
    Rw = euler("xyz", np.radians(gt[k, 3:6]))
    v = xe * vox
    return ((v - c) @ Rw.T + c + gt[k, :3]) / vox - xe


def rigid_disp(R_, t):                                                                    # about the grid center, mm
    return (((xe - cen) * vox) @ R_.T + t) / vox + cen - xe


def score(est, truth):
    d = est - truth
    d = d - d.mean(0, keepdims=True)
    within, means = [], []
    for g in groups:
        if len(g) == 0:
            continue
        m = d[g].mean(0, keepdims=True); means.append(m)
        within.append(np.linalg.norm(d[g] - m, axis=-1).ravel())
    mm = np.concatenate(means)
    n = np.linalg.norm(d, axis=-1)
    return {"edge_error_rms_p95": [round(float(np.sqrt((n ** 2).mean())), 3), round(float(np.quantile(n, 0.95)), 3)],
            "within_shell_rms": round(float(np.sqrt((np.concatenate(within) ** 2).mean())), 3),
            "between_shell_rms": round(float(np.sqrt((np.linalg.norm(mm - mm.mean(0, keepdims=True), axis=-1) ** 2).mean())), 3),
            "worst_volume_rms": round(float(np.sqrt((n ** 2).mean(1)).max()), 3)}


truth = np.stack([truth_disp(k) for k in range(G)])
out = {"scan": f"{S} {NOISE} {MOT}", "volumes": G, "groups": [len(g) for g in groups],
       "true motion at the edge, rms (voxels)": round(float(np.sqrt((np.linalg.norm(truth - truth.mean(0), axis=-1) ** 2).mean())), 3),
       "as acquired": score(np.zeros_like(truth), truth)}

if arg("--ours"):
    T = np.load(Path(arg("--ours")) / "ours_theta.npz")
    th, PE, nmon = T["th"], int(T["pe_axis"]), int(T["nmon"])
    ours_rigid = np.stack([rigid_disp(euler("zyx", th[k, 3:6]), th[k, :3]) for k in range(G)])
    u = (xe - cen) / cen
    a, b, cc = u.T
    mono = np.stack([np.ones_like(a), a, b, cc] + ([a * a, b * b, cc * cc, a * b, a * cc, b * cc] if nmon == 10 else []), -1)
    ours_all = ours_rigid.copy()
    for k in range(G):
        ours_all[k][:, PE] += mono @ th[k, 6:]
    out["ours, rigid"] = score(ours_rigid, truth)
    thf = T["th_fit"]
    out["ours, rigid, as fitted (PE offset all in the translation)"] = score(np.stack([rigid_disp(euler("zyx", thf[k, 3:6]), thf[k, :3]) for k in range(G)]), truth)
    Rt = [euler("xyz", np.radians(gt[k, 3:6])) for k in range(G)]
    Ro = [euler("zyx", th[k, 3:6]) for k in range(G)]
    rel = [Ro[k] @ Rt[k].T for k in range(G)]
    U, _, Vt = np.linalg.svd(sum(rel)); Rm = U @ Vt                                     # the mean rotation (frame)
    ang = [np.degrees(np.arccos(np.clip((np.trace(Rm.T @ r) - 1) / 2, -1, 1))) for r in rel]
    out["ours, rotation error after the frame (deg), rms / max"] = [round(float(np.sqrt(np.mean(np.square(ang)))), 3), round(float(np.max(ang)), 3)]
    out["ours, with eddy terms"] = score(ours_all, truth)

if arg("--eddy"):
    E = Path(arg("--eddy"))
    files = sorted(glob.glob(str(E / "eddy.eddy_displacement_fields.*.nii*")))
    if len(files) == G:
        raw = np.stack([np.asarray(nib.load(f).dataobj)[ie] / vox for f in files])
        best = None                                                                       # FSL's frame: x flipped when det > 0
        for sx, sa in itertools.product((1, -1), (1, -1)):
            est = raw * np.array([sx, 1, 1]) * sa
            sc = score(est, truth)
            if best is None or sc["edge_error_rms_p95"][0] < best[0]["edge_error_rms_p95"][0]:
                best = (sc, f"x sign {sx}, overall sign {sa}")
        out["eddy, displacement fields (" + best[1] + ")"] = best[0]
    p = np.loadtxt(E / "eddy.eddy_parameters")
    # FSL's movement matrix (TOPUP::MovePar2Matrix, MISCMATHS::construct_rotmat_euler; read for method): R = Rx Ry Rz of
    # the angles (radians) about the grid center ((n - 1) * dim / 2), then the translation, in FSL's coordinates, whose x
    # axis is flipped when the affine's determinant is positive. Which way eddy applies it (forward or inverse) and the
    # elementary rotations' sign are taken from the candidate that fits.
    cmm = cen * vox
    Fl = np.eye(4); Fl[0, 0] = -1; Fl[0, 3] = 2 * cmm[0]                                   # x -> 2 c_x - x, its own inverse
    def fsl_mat(k, sr):
        Rr = euler("xyz", sr * p[k, 3:6]); M = np.eye(4); M[:3, :3] = Rr; M[:3, 3] = cmm - Rr @ cmm + p[k, :3]
        return M
    best = None
    v4 = np.c_[xe * vox, np.ones(len(xe))]
    for sr, inv, flip in itertools.product((1, -1), (False, True), (False, True)):
        est = []
        for k in range(G):
            M = fsl_mat(k, sr)
            M = np.linalg.inv(M) if inv else M
            M = Fl @ M @ Fl if flip else M
            est.append((v4 @ M.T)[:, :3] / vox - xe)
        sc = score(np.stack(est), truth)
        if best is None or sc["edge_error_rms_p95"][0] < best[0]["edge_error_rms_p95"][0]:
            best = (sc, f"rotation sign {sr}, inverse {inv}, x-flip {flip}")
    out["eddy, rigid parameters (best-fitting convention: " + best[1] + ")"] = best[0]

print(json.dumps(out, indent=1))
tag = f"{S}_{NOISE}_{MOT}" + (f"_{arg('--tag')}" if arg("--tag") else "")
json.dump(out, open(Path(__file__).resolve().parent.parent / f"results/motion_eddy/phantom_{tag}.json", "w"), indent=1)
