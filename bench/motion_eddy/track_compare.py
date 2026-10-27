"""What the corrections change downstream, on a real scan: tractline's tracker (UKF, the b = 2000 shell and the b0s,
ORG settings) and its labeler (RapidParc, one draw) on each corrected set, each with its own rotated b-vectors:
  as acquired, eddy, TORTOISE, ours with the GPU-MI anchor, ours with no anchor.
Ours with and without the anchor differ only by where the DW shells sit against the b0s (about 0.2 voxel at the edge),
so their difference is the effect of that placement alone. Per pair: the streamline-density maps' correlation and the
Dice of their supports (voxels visited by at least 5 streamlines), the per-tract count difference (sum |a - b| / sum a
over the 42 tracts), and the Other share of each.

    uv run --group dipy bench/motion_eddy/track_compare.py <dir> [--floor]      # MPS; --floor: the scan against itself moved 0.05 voxel
"""
import json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np, nibabel as nib

from tractline import pipeline as P

D = Path(sys.argv[1])


def sets_arg(D, default):
    """--sets "name=relpath;name=relpath" (relative to the scan folder; "img,bvecs" pairs for track_compare) replaces the
    default sets; the output file is tagged with the scan folder's number"""
    if "--sets" not in sys.argv:
        return default
    out = {}
    for item in sys.argv[sys.argv.index("--sets") + 1].split(";"):
        k, v = item.split("=", 1)
        out[k] = tuple(D / p for p in v.split(",")) if "," in v else D / v
    return out


TAG = Path(sys.argv[1]).name.replace("motion_eddy_real", "")
ref = nib.load(D / "imain.nii.gz")
bval = np.loadtxt(D / "bvals")
sets = {"as acquired": (D / "imain.nii.gz", D / "bvecs"),
        "eddy": (D / "eddy.nii.gz", D / "eddy.eddy_rotated_bvecs"),
        "TORTOISE": (D / "tortoise_out2/tortoise.nii", D / "tortoise_out2/tortoise.bvecs"),
        "ours, GPU MI anchor": (D / "v_gpumi2/ours.nii.gz", D / "v_gpumi2/ours_rotated_bvecs"),
        "ours, no anchor": (D / "v_none/ours.nii.gz", D / "v_none/ours_rotated_bvecs")}
sets = sets_arg(D, sets)
if "--floor" in sys.argv:                                                                 # the floor: a trivial change, the whole scan moved
    from scipy.ndimage import shift as _shift                                             # by 0.05 voxel (b0s and shells together)
    acq = np.asarray(ref.dataobj, np.float32)
    moved = np.stack([_shift(acq[..., k], (0.05, 0.05, 0.05), order=3, mode="nearest") for k in range(acq.shape[-1])], -1)
    nib.save(nib.Nifti1Image(moved, ref.affine), str(D / "floor_shift005.nii.gz"))
    sets = {"as acquired": sets["as acquired"], "as acquired, moved 0.05 voxel": (D / "floor_shift005.nii.gz", D / "bvecs")}
labeler = P.default_labeler("mps")
X, Y, Z = ref.shape[:3]
inv = np.linalg.inv(ref.affine)
res = {}
for name, (img_f, bvec_f) in sets.items():
    dwi = np.asarray(nib.load(img_f).dataobj, np.float32)
    bvec = np.loadtxt(bvec_f)
    s = SimpleNamespace(affine=ref.affine, bval=bval, bvec=bvec, vox=np.array(ref.header.get_zooms()[:3], float))
    timer = P.Timer()
    with P.exact_float32("mps"):
        tg = P.track(s, dwi, timer, device="mps", shell=2000.0)
        lab = P.label(tg, labeler, timer)
    P.release_memory("mps")
    dens = np.zeros((X, Y, Z), np.int32)
    for f in lab.kept(tg.fibers):
        v = np.rint((np.c_[f, np.ones(len(f))] @ inv.T)[:, :3]).astype(int)
        v = v[(v >= 0).all(1) & (v < [X, Y, Z]).all(1)]
        np.add.at(dens, tuple(np.unique(v, axis=0).T), 1)
    counts = np.bincount(lab.tract, minlength=43)
    res[name] = {"dens": dens, "counts": counts, "fibers": len(tg.fibers), "labeled": int(lab.keep.sum()),
                 "other_share": round(float(counts[42] / max(1, counts.sum())), 4), "seconds": timer.seconds}
    print(name, {k: v for k, v in res[name].items() if k not in ("dens", "counts")}, flush=True)


def compare(a, b):
    da, db = res[a]["dens"].ravel().astype(float), res[b]["dens"].ravel().astype(float)
    sa, sb = da >= 5, db >= 5
    ca, cb = res[a]["counts"][:42].astype(float), res[b]["counts"][:42].astype(float)
    return {"density correlation": round(float(np.corrcoef(da, db)[0, 1]), 4),
            "support Dice (>= 5 streamlines)": round(float(2 * (sa & sb).sum() / (sa.sum() + sb.sum())), 4),
            "tract count difference": round(float(np.abs(ca - cb).sum() / ca.sum()), 4)}


import itertools
pairs = [("as acquired", "as acquired, moved 0.05 voxel")] if "--floor" in sys.argv else list(itertools.combinations(sets, 2)) if "--sets" in sys.argv else [("ours, GPU MI anchor", "ours, no anchor"), ("eddy", "TORTOISE"), ("eddy", "ours, no anchor"),
         ("eddy", "ours, GPU MI anchor"), ("TORTOISE", "ours, no anchor"), ("as acquired", "eddy"), ("as acquired", "ours, no anchor")]
out = {"per set": {k: {kk: vv for kk, vv in v.items() if kk not in ("dens", "counts")} for k, v in res.items()},
       "pairs": {f"{a} vs {b}": compare(a, b) for a, b in pairs}}
for k, v in out["pairs"].items():
    print(k.ljust(44), v)
json.dump(out, open(Path(__file__).resolve().parent.parent / (f"results/motion_eddy/track_compare_{TAG}" + ("_floor" if "--floor" in sys.argv else "") + ".json"), "w"), indent=1)
