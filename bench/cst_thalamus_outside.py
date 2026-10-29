"""Two claims from Albula's pipeline description (Ron Kikinis, 2026-10-28), checked on tractline (NOTES 2026-10-28,
"Ron's claims checked; the masks with and without CSF"):
  1. "About four in five streamlines named corticospinal, in our set and in tractline's, run through a thalamus."
  2. Tracking inside a mask that includes CSF can follow a cranial nerve through a cistern, and the labeler gives it a
     brain tract's name (a trigeminal nerve named uncinate, ds001226 PAT29).

    uv run python bench/cst_thalamus_outside.py        # -> results/cst_thalamus_outside.json

Subjects with FastSurfer labels (VINN asegdkt via haversack, DATA/fastsurfer) and a rigid T1-to-DWI fit: ds001226 PAT16
(the T1 to its corrected mean b0, DATA/ensemble/PAT16/tumor_dwi.npz) and MASiVar cIIIs04 (s1Bx1 scan 1; its field of
view ends at the pons; DATA/fastsurfer/cIIIs04_t1_to_dwi.npz). Each tracked now (tractline as checked out: field
corrected, the shell nearest b = 3000) with each mask in ARMS: the default (median_otsu), SynthStrip's (border 1 mm,
the opt-in), and, put in at the pipeline's mask step in place of brain_mask, the tighter variants: SynthStrip's main
model at border 0, its no-CSF model (fetch "synthstrip-nocsf") at borders 1 and 2, and the no-CSF mask (border 1) joined
with median_otsu's.

1. Of the streamlines named CST: the share with 2 or more points in a thalamus (FastSurfer labels 10/49), counting the
   whole label and its core (points deeper than 1, 2, 3 mm inside it), so an alignment error at the thalamus's edge
   shows as a share that falls with depth; and the share that fits the CST endpoint rule (ensemble_endpoints.RULES:
   one end in motor cortex, the other in the brainstem, ventral DC or the field of view's lower face). The thalamic
   classes (TF, TP, TO, TT) and PLIC beside them, for scale.
2. Of the labeled streamlines: those with a run of 3 or more consecutive points (1.8 mm apart) more than 2 mm outside
   FastSurfer's brain (any label but 0 and 24, CSF), so in CSF, a cistern or beyond, not explained by a 1-2 mm
   alignment error. By name; and those whose outside run lies within 15 mm of the brainstem label (where cranial
   nerves leave it).
"""
import json, sys, time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from tractline import pipeline as P
from tractline.data import DATA
from tractline.labelers.base import TRACT_NAMES
from tractline.motion_eddy import shell_groups

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_endpoints import Anatomy, FS, consistent, ends_info  # noqa: E402

THAL = (10, 49)
BRAINSTEM = 16
NOT_BRAIN = (0, 24)
OUT_MM, RUN, NEAR_STEM_MM = 2.0, 3, 15.0
WATCH = ["CST", "PLIC", "TF", "TP", "TO", "TT"]
#: name -> (the run's mask= keyword, a replacement for pipeline.brain_mask or None)
ARMS = {}


def _sdt(dwi, bval, affine, device, weights_name):
    from tractline import skullstrip as K
    from tractline.fetch import weights
    b0 = dwi[..., np.asarray(bval) < 50].astype(np.float32).mean(-1)
    return K.signed_distance(b0, affine, weights(weights_name), device), b0


def _arm(fn):
    def brain_mask(dwi, bval, timer, prefix="", device="mps", impl="median-otsu", opts=None, affine=None):
        with timer(prefix + "mask"):
            return fn(dwi, bval, affine, device)
    return brain_mask


def _union_otsu(dwi, bval, affine, device):
    from tractline.mask import median_otsu
    sdt, b0 = _sdt(dwi, bval, affine, device, "synthstrip-nocsf")
    return (sdt < 1) | median_otsu(b0, 4, 4, device=device)[1]


ARMS.update({
    "median-otsu": (None, None), "synthstrip (border 1)": ("synthstrip", None),
    "synthstrip, border 0": (None, _arm(lambda d, b, a, dev: _sdt(d, b, a, dev, "synthstrip")[0] < 0)),
    "no-CSF, border 1": (None, _arm(lambda d, b, a, dev: _sdt(d, b, a, dev, "synthstrip-nocsf")[0] < 1)),
    "no-CSF, border 2": (None, _arm(lambda d, b, a, dev: _sdt(d, b, a, dev, "synthstrip-nocsf")[0] < 2)),
    "no-CSF (border 1) or median_otsu": (None, _arm(_union_otsu)),
})


def nearest_3000(bval):
    b = np.asarray(bval, float)
    groups, nominal = shell_groups(b)
    shells = [(abs(float(np.median(b[g])) - 3000), float(np.median(b[g]))) for g, n in zip(groups, nominal) if n > 0]
    return min(shells)[1]


def subjects():
    from _ds001226 import load as load_btc
    from _masivar import load as load_masivar
    z = np.load(DATA / "ensemble" / "PAT16" / "tumor_dwi.npz")
    yield "PAT16", load_btc("PAT16"), np.linalg.inv(z["t1_to_dwi"])
    yield "cIIIs04", load_masivar("cIIIs04", "s1Bx1", 1), np.linalg.inv(np.load(FS / "cIIIs04_t1_to_dwi.npz")["t1_to_dwi"])


class Maps:
    """On the T1 grid: depth inside the thalamus (mm), distance outside the brain (mm), distance to the brainstem (mm)."""
    def __init__(self, anat):
        lab = anat.lab
        sp = np.sqrt((anat.affine[:3, :3] ** 2).sum(0))
        self.anat = anat
        self.thal_depth = distance_transform_edt(np.isin(lab, THAL), sampling=sp)
        brain = ~np.isin(lab, NOT_BRAIN)
        self.outside = distance_transform_edt(~brain, sampling=sp)
        self.to_stem = distance_transform_edt(lab != BRAINSTEM, sampling=sp)

    def sample(self, field, world_t1):
        ijk, ok = self.anat.voxels(world_t1)
        out = np.zeros(len(world_t1))
        out[ok] = field[tuple(ijk[ok].T)]
        return out


def longest_run(mask):
    best = cur = 0
    for m in mask:
        cur = cur + 1 if m else 0
        best = max(best, cur)
    return best


def analyze(fibers, keep, tract, anat, maps, to_t1, aff):
    kept = [f for f, k in zip(fibers, keep) if k]
    t1 = [f @ to_t1[:3, :3].T + to_t1[:3, 3] for f in kept]
    out = {}
    # 1. thalamus passage
    ra, rb, sa, sb, enters = ends_info(kept, anat, to_t1, aff)
    ok, scored = consistent(tract, ra, rb, sa, sb, enters)
    thal = {}
    for name in WATCH:
        idx = np.flatnonzero(tract == TRACT_NAMES.index(name))
        if not len(idx):
            continue
        depth = [maps.sample(maps.thal_depth, t1[i]) for i in idx]
        thal[name] = {"n": int(len(idx)),
                      **{f"through thalamus (>= 2 points deeper than {d} mm)": round(float(np.mean([(x > d).sum() >= 2 for x in depth])), 3)
                         for d in (0, 1, 2, 3)}}
        if name == "CST":
            through = np.array([(x > 0).sum() >= 2 for x in depth])
            fits = ok[idx]
            thal[name]["fits the CST endpoint rule"] = round(float(fits.mean()), 3)
            thal[name]["fits the rule: through the thalamus / not"] = [round(float(fits[through].mean()), 3) if through.any() else None,
                                                                       round(float(fits[~through].mean()), 3) if (~through).any() else None]
    out["thalamus"] = thal
    # 2. outside the brain
    flagged, near_stem, by_name = [], [], {}
    for i, p in enumerate(t1):
        o = maps.sample(maps.outside, p) > OUT_MM
        if longest_run(o) >= RUN:
            flagged.append(i)
            n = TRACT_NAMES[int(tract[i])]
            by_name[n] = by_name.get(n, 0) + 1
            if (maps.sample(maps.to_stem, p[o]) <= NEAR_STEM_MM).any():
                near_stem.append(n)
    named = tract != TRACT_NAMES.index("Other")
    fl = np.zeros(len(t1), bool)
    fl[flagged] = True
    out["outside the brain"] = {
        "labeled": int(len(t1)), "flagged (a run of 3+ points more than 2 mm outside FastSurfer's brain)": int(fl.sum()),
        "flagged and named (not Other)": int((fl & named).sum()),
        "named, by tract (top 12)": dict(sorted(((k, v) for k, v in by_name.items() if k != "Other"), key=lambda kv: -kv[1])[:12]),
        "Other": by_name.get("Other", 0),
        "outside run within 15 mm of the brainstem: named, by tract": {k: near_stem.count(k) for k in sorted(set(near_stem)) if k != "Other"},
        "outside run within 15 mm of the brainstem: Other": near_stem.count("Other")}
    return out


def main():
    res = {}
    labeler = P.default_labeler("mps")
    for name, s, to_t1 in subjects():
        anat = Anatomy(name)
        maps = Maps(anat)
        shell = nearest_3000(s.bval)
        res[name] = {"shell": shell}
        for key, (mask, replace) in ARMS.items():
            t0 = time.perf_counter()
            original = P.brain_mask
            if replace is not None:
                P.brain_mask = replace
            try:
                r = P.run(s, labeler, P.Timer(), device="mps", shell=shell, mask=mask)
            finally:
                P.brain_mask = original
            corr, tg, labels = r
            tract = np.asarray(labels.tract)
            ra, rb, sa, sb, enters = ends_info([f for f, k in zip(tg.fibers, labels.keep) if k], anat, to_t1, s.affine)
            ok, _ = consistent(tract, ra, rb, sa, sb, enters)
            cst = tract == TRACT_NAMES.index("CST")
            res[name][key] = {"field applied": bool(getattr(corr, "applied", False)), "streamlines": len(tg.fibers),
                              "mask voxels": int(tg.mask.sum()), "CST fitting the endpoint rule": int((ok & cst).sum()),
                              "seconds": round(time.perf_counter() - t0, 1),
                              **analyze(tg.fibers, np.asarray(labels.keep), tract, anat, maps, to_t1, s.affine)}
            print(name, key, json.dumps(res[name][key])[:1500], flush=True)
    (HERE / "results" / "cst_thalamus_outside.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
