"""dipy's rigid mutual-information registration for the shell anchor, in a process of its own (dipy's and torch's OpenMP
runtimes clash in one process), in proto.py's frame: world = (x - cen) * vox, voxel x; the rigid parameters as phi's,
d = (tx, ty, tz mm, rx, ry, rz rad), R = Rz Ry Rx, with mov(phi_d(x)) ~ fixed(x).

    uv run --group dipy bench/motion_eddy/mi_dipy.py selftest                     # known shifts of a real b0 mean, recovered?
    uv run --group dipy bench/motion_eddy/mi_dipy.py register <in.npz> <out.json> # mov, fixed, mask, vox, cen -> d

The self-test moves the mean b0 of ON-Harmony sub-16974 by known rigid transforms (scipy, cubic) and registers it back,
both as the same image and against the b2000 mean (cross contrast, the anchor's case).
"""
import json, math, sys
import numpy as np


def rot(a, b, c):
    ca, sa, cb, sb, cc, sc = math.cos(a), math.sin(a), math.cos(b), math.sin(b), math.cos(c), math.sin(c)
    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]]); Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def register(mov, fixed, mask, vox, cen, pipeline=("translation", "rigid"), nbins=32, sigmas=(2.0, 1.0, 0.0),
             factors=(4, 2, 1), level_iters=(1000, 500, 100)):
    from dipy.align import affine_registration
    A = np.diag(list(vox) + [1.0]); A[:3, 3] = -np.asarray(cen) * np.asarray(vox)
    _, M = affine_registration(np.asarray(mov, np.float64), np.asarray(fixed, np.float64), moving_affine=A, static_affine=A,
                               pipeline=list(pipeline), nbins=nbins, level_iters=list(level_iters), sigmas=list(sigmas),
                               factors=list(factors), static_mask=None if mask is None else np.asarray(mask, np.int32))
    R, t = M[:3, :3], M[:3, 3]                                           # moving world = R static world + t
    beta = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    alpha, gamma = math.atan2(R[2, 1], R[2, 2]), math.atan2(R[1, 0], R[0, 0])
    return [float(t[0]), float(t[1]), float(t[2]), alpha, beta, gamma]


def moved(img, d, vox, cen):
    """img sampled at phi_d(x) for every voxel x (cubic, scipy), phi_d(x) = (R (x - c) vox + t) / vox + c."""
    from scipy.ndimage import affine_transform
    R = rot(*d[3:]); v = np.asarray(vox, float); c = np.asarray(cen, float)
    Mv = np.diag(1 / v) @ R @ np.diag(v)                                 # voxel-frame matrix
    off = c - Mv @ c + np.asarray(d[:3]) / v
    return affine_transform(img, Mv, offset=off, order=3, mode="nearest")


def selftest():
    import nibabel as nib
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
    from tractline.data import DATA
    from scipy.ndimage import binary_dilation
    D = str(DATA / "ds004712/sub-16974/ses-OXF1PRI001/dwi/sub-16974_ses-OXF1PRI001_dir-AP_dwi")
    img = nib.load(D + ".nii.gz"); data = np.asarray(img.dataobj, np.float32); bval = np.loadtxt(D + ".bval")
    vox = [float(v) for v in img.header.get_zooms()[:3]]
    b0 = data[..., bval < 50].mean(-1); dw = data[..., np.abs(bval - 2000) < 100].mean(-1)
    cen = [(n - 1) / 2 for n in b0.shape]
    mask = binary_dilation(b0 > np.quantile(b0, 0.6), iterations=2)
    tests = [[0.0, 0.7, 0.0, 0.005, 0.0, 0.0], [0.5, -0.3, 0.4, 0.0, 0.004, -0.006], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    out = []
    tests = [[0.0, 0.7, 0.0, 0.005, 0.0, 0.0], [0.5, -0.3, 0.4, 0.0, 0.004, -0.006], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    out = []
    # cross contrast: no truth for the b2000 mean's own offset from the b0 mean in this scan, so consistency: register it
    # as it is (r0) and moved by d (r1); r1 should be r0 - d to first order
    r0 = register(dw, b0, mask, vox, cen)
    for d in tests[:2]:
        r1 = register(moved(dw, d, vox, cen), b0, mask, vox, cen)
        err = [r1[i] - r0[i] + d[i] for i in range(3)] + [math.degrees(r1[i] - r0[i] + d[i]) for i in range(3, 6)]
        out.append({"shift": d, "case": "cross contrast (b2000 mean -> b0 mean), consistency", "r0 (mm, deg)":
                    [round(v, 3) for v in r0[:3]] + [round(math.degrees(v), 3) for v in r0[3:]], "error (mm, deg)": [round(v, 3) for v in err]})
        print(json.dumps(out[-1]), flush=True)
    for d in tests:
        for name, fixed in (("same contrast (b0 -> b0)", b0), ("same contrast (b2000 mean -> b2000 mean)", dw)):
            mov = moved(b0 if fixed is b0 else dw, d, vox, cen)              # the moving image, moved by d
            for msk in (None, mask):
                r = register(mov, fixed, msk, vox, cen)
                # mov(phi_r(x)) ~ fixed(x) and mov = fixed o phi_d: phi_d o phi_r = id, so r ~ -d to first order
                err = [r[i] + d[i] for i in range(3)] + [math.degrees(r[i] + d[i]) for i in range(3, 6)]
                out.append({"shift": d, "case": name, "mask": msk is not None, "recovered (mm, deg)": [round(v, 3) for v in r[:3]] +
                            [round(math.degrees(v), 3) for v in r[3:]], "error (mm, deg)": [round(v, 3) for v in err]})
                print(json.dumps(out[-1]), flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    else:
        z = np.load(sys.argv[2])
        r = register(z["mov"], z["fixed"], z["mask"] if "mask" in z else None, z["vox"], z["cen"])
        json.dump(r, open(sys.argv[3], "w"))
