"""Does track:ptt's stopping "fa" end paths in gray matter on real brains, and does its threshold hold across shells?
(NOTES 2026-10-29, "Stopping at low FA, checked on real brains"). On the two scans with FastSurfer labels and a rigid
T1-to-DWI fit (cst_thalamus_outside.subjects: ds001226 PAT16, MASiVar cIIIs04).

    uv run python bench/ptt_fa_anatomy.py        # -> results/ptt_fa_anatomy.json
    uv run python bench/ptt_fa_anatomy.py --scan <dwi.nii.gz> <labels.mgz> <t1_to_dwi.npz> <tag> <out folder>
                                                 # another scan: FastSurfer's label volume (aparc.DKTatlas+aseg.deep.mgz,
                                                 # on its conformed grid) and the rigid fit (t1_to_dwi: T1 world -> DWI
                                                 # world); -> <out folder>/ptt_fa_anatomy_<tag>.json (a private scan: an
                                                 # out folder outside the repository)

1. Ends. Each arm through pipeline.run (field-corrected, the shell nearest b = 3000, median_otsu): today's UKF, and PTT
   (seeding fa, 100,000 seeds) with stopping mask and fa at 0.10, 0.15, 0.20 and auto (its value recorded). Each kept streamline's two ends, carried
   into the T1's world, are classed by the FastSurfer label there (CLASSES), after one rule: an end within 2 voxels of
   the DWI grid's faces is "at the field of view's face" (cIIIs04's ends at the pons). A white-matter end more than
   2 mm from any gray label (cortex, cerebellar cortex, deep gray) is "deep in white matter": where a path should not
   stop, and where stopping too early at low FA would show. Reported as shares of all ends, overall and for the
   streamlines named CST, AF and UF.
2. FA by tissue and shell. DTI on each shell with the b0s (the corrected series, over the mask), each DWI voxel's
   center classed by its FastSurfer label: the median FA of cortex, deep gray and white matter deep (more than 2 mm from
   gray), and the shares of deep white matter below 0.10, 0.15 and 0.20 (where fa would stop inside white matter) and
   of cortex at or above them (where it would not stop at gray).
"""
import json, sys, time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from tractline import csd as C, pipeline as P
from tractline.labelers.base import TRACT_NAMES
from tractline.motion_eddy import shell_groups

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cst_thalamus_outside import nearest_3000, subjects  # noqa: E402
from ensemble_endpoints import Anatomy  # noqa: E402

DEEP_GM = (10, 11, 12, 13, 17, 18, 26, 49, 50, 51, 52, 53, 54, 58)
WM = (2, 41, 7, 46, 77, 85, 251, 252, 253, 254, 255)
CSF = (4, 5, 14, 15, 24, 31, 43, 44, 63, 72)
CLASSES = ("cortex", "cerebellar cortex", "deep gray", "brainstem", "ventral DC", "white matter, near gray",
           "white matter, deep", "CSF", "outside the brain", "other", "at the field of view's face")
DEEP_MM, FACE_VOX = 2.0, 2
PTT = {"impl": "ptt", "seeding": "fa", "seed_count": 100_000}
ARMS = {"ukf": None, "ptt, mask": {**PTT, "stopping": "mask"},
        **{f"ptt, fa {x:.2f}": {**PTT, "stopping": "fa", "stopping_fa": x} for x in (0.10, 0.15, 0.20)},
        "ptt, fa auto": {**PTT, "stopping": "fa", "stopping_fa": "auto"}}
FOCUS = ("CST", "AF", "UF")
THRESHOLDS = (0.10, 0.15, 0.20)


class MgzAnatomy:
    """A FastSurfer label volume read from its .mgz, with Anatomy's interface (lab, affine, voxels)."""
    def __init__(self, path):
        import nibabel as nib
        img = nib.load(str(path))
        self.lab = np.asarray(img.dataobj).astype(np.int32)
        self.affine = img.affine
        self.inv = np.linalg.inv(self.affine)

    def voxels(self, world):
        ijk = np.rint(world @ self.inv[:3, :3].T + self.inv[:3, 3]).astype(int)
        return ijk, np.all((ijk >= 0) & (ijk < self.lab.shape), 1)


class Tissue:
    """On the T1 grid: each voxel's class index (CLASSES, without the face rule) and the distance to gray (mm)."""
    def __init__(self, anat):
        lab = anat.lab
        sp = np.sqrt((anat.affine[:3, :3] ** 2).sum(0))
        cortex = (lab >= 1000) | np.isin(lab, (3, 42))
        cbl = np.isin(lab, (8, 47))
        deep = np.isin(lab, DEEP_GM)
        gray = cortex | cbl | deep
        self.to_gray = distance_transform_edt(~gray, sampling=sp)
        cls = np.full(lab.shape, CLASSES.index("other"), np.int8)
        cls[lab == 0] = CLASSES.index("outside the brain")
        cls[np.isin(lab, CSF)] = CLASSES.index("CSF")
        cls[cortex] = CLASSES.index("cortex")
        cls[cbl] = CLASSES.index("cerebellar cortex")
        cls[deep] = CLASSES.index("deep gray")
        cls[lab == 16] = CLASSES.index("brainstem")
        cls[np.isin(lab, (28, 60))] = CLASSES.index("ventral DC")
        wm = np.isin(lab, WM)
        cls[wm] = CLASSES.index("white matter, near gray")
        cls[wm & (self.to_gray > DEEP_MM)] = CLASSES.index("white matter, deep")
        self.cls, self.anat = cls, anat

    def classes(self, world_t1):
        ijk, ok = self.anat.voxels(world_t1)
        out = np.full(len(world_t1), CLASSES.index("outside the brain"), np.int8)
        out[ok] = self.cls[tuple(ijk[ok].T)]
        return out


def shares(c):
    return {k: round(float((c == i).mean()), 4) for i, k in enumerate(CLASSES)} if len(c) else {}


def ends(fibers, tissue, to_t1, affine, shape):
    """Each streamline's two ends: their classes (n, 2)."""
    e = np.stack([np.stack([f[0], f[-1]]) for f in fibers])                    # (n, 2, 3) DWI world
    inv = np.linalg.inv(affine)
    vox = e @ inv[:3, :3].T + inv[:3, 3]
    face = np.any((vox < FACE_VOX) | (vox > np.asarray(shape) - 1 - FACE_VOX), -1)
    t1 = e.reshape(-1, 3) @ to_t1[:3, :3].T + to_t1[:3, 3]
    c = tissue.classes(t1).reshape(-1, 2)
    c[face] = CLASSES.index("at the field of view's face")
    return c


def fa_by_tissue(corr, s, mask, tissue, to_t1):
    """Per shell: FA medians by class and the shares below and above each threshold (module docstring, 2)."""
    b = np.asarray(s.bval, float)
    groups, nominal = shell_groups(b)
    ijk = np.argwhere(mask)
    world = ijk @ s.affine[:3, :3].T + s.affine[:3, 3]
    cls = tissue.classes(world @ to_t1[:3, :3].T + to_t1[:3, 3])
    out = {}
    for g, nb in zip(groups, nominal):
        if nb <= 0 or len(g) < 6:
            continue
        shell = float(np.median(b[g]))
        d, bval, bvec = P._shell_volumes(corr.dwi, s.bval, s.bvec, shell)
        ev, _ = C.tensor_fit(d[mask].astype(np.float64), bval, P._image_axes(bvec, s.affine))
        fa = C.fa(ev)
        row = {"directions": int((bval > 50).sum())}
        for name in ("cortex", "deep gray", "white matter, deep", "white matter, near gray"):
            v = fa[cls == CLASSES.index(name)]
            row[f"{name}: voxels"] = int(len(v))
            row[f"{name}: median FA"] = round(float(np.median(v)), 3) if len(v) else None
        wd, cx = fa[cls == CLASSES.index("white matter, deep")], fa[cls == CLASSES.index("cortex")]
        for x in THRESHOLDS:
            row[f"deep white matter below {x:.2f}"] = round(float((wd < x).mean()), 4)
            row[f"cortex at or above {x:.2f}"] = round(float((cx >= x).mean()), 4)
        out[f"b = {shell:.0f}"] = row
    return out


def main(scan=None):
    res = {"classes": CLASSES, "deep white matter: mm from gray": DEEP_MM, "face rule: voxels": FACE_VOX}
    labeler = P.default_labeler("mps")
    if scan:
        from tractline import bids
        dwi, labels_mgz, fit, tag, out = scan
        items = [(tag, bids.load(dwi), np.linalg.inv(np.load(fit)["t1_to_dwi"]), MgzAnatomy(labels_mgz))]
    else:
        items = [(name, s, to_t1, Anatomy(name)) for name, s, to_t1 in subjects()]
    for name, s, to_t1, anat in items:
        tissue = Tissue(anat)
        shell = nearest_3000(s.bval)
        res[name] = {"shell tracked": shell, "arms": {}}
        for arm, track in ARMS.items():
            t0 = time.perf_counter()
            base = P.config_for(shell=shell)
            cfg = base if track is None else base.with_(model={"impl": "csd", "shell": shell}, track=track)
            corr, tg, labels = P.run(s, labeler, P.Timer(), device="mps", config=cfg)
            keep = np.asarray(labels.keep)
            tract = np.asarray(labels.tract)
            c = ends(tg.fibers, tissue, to_t1, s.affine, s.dwi.shape[:3])
            row = {"streamlines": len(tg.fibers), "labeled": int(keep.sum()), "all ends": shares(c.ravel()),
                   **{k: tg.stats[k] for k in ("stopping_fa (auto)", "cortex FA estimate", "auto shell") if k in tg.stats}}
            ck = c[keep]
            for t in FOCUS:
                sel = tract == TRACT_NAMES.index(t)
                row[f"{t} ends ({int(sel.sum())} streamlines)"] = shares(ck[sel].ravel())
            row["streamlines with an end deep in white matter"] = round(
                float((c == CLASSES.index("white matter, deep")).any(1).mean()), 4)
            row["seconds"] = round(time.perf_counter() - t0, 1)
            res[name]["arms"][arm] = row
            print(name, arm, json.dumps({k: row[k] for k in ("streamlines", "labeled",
                                                             "streamlines with an end deep in white matter")}),
                  json.dumps(row["all ends"]), flush=True)
            if arm == "ukf":
                res[name]["FA by tissue and shell"] = fa_by_tissue(corr, s, tg.mask, tissue, to_t1)
                print(name, json.dumps(res[name]["FA by tissue and shell"]), flush=True)
    path = Path(scan[4]) / f"ptt_fa_anatomy_{scan[3]}.json" if scan else HERE / "results" / "ptt_fa_anatomy.json"
    path.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    if "--scan" in sys.argv:
        a = sys.argv[sys.argv.index("--scan") + 1:]
        if len(a) != 5:
            sys.exit("usage: ptt_fa_anatomy.py --scan <dwi.nii.gz> <labels.mgz> <t1_to_dwi.npz> <tag> <out folder>")
        main(a)
    else:
        main()
