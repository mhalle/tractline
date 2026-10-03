"""TractCloud at the trained context (k_global 500, 10,000 local candidates; tractline's Labeler) against
upstream's inference context (80, 10 %), beyond PAT16's one tractogram:

  - across trackers, PAT16 (the M2's field): the Metal and the float64 tractograms, one draw each,
    seed-matched label agreement and Other fractions, at each context;
  - another scan, the Stanford HARDI brain (healthy, 2 mm, 150 directions, b = 2000; Metal): the Other
    fraction over draws 0-4 and per-streamline agreement between draws 0 and 1, at each context.

    uv run bench/label_trained_check.py

Writes results/label_trained_check.json.
"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np, nibabel as nib, torch
from tractline import pipeline as P, ukf as U
from tractline.prep import prepare
from tractline.data import DATA
from tractline.labelers.tractcloud import Labeler
from _ds001226 import load

HERE = Path(__file__).resolve().parent


def other(t):
    return round(float((t == 42).mean()), 4)


if __name__ == "__main__":
    labs = {"upstream (80, 10 %)": Labeler("mps", upstream=True), "trained (500, 10,000)": Labeler("mps")}
    res = {"pat16_trackers": {}, "hardi": {}}

    s = load("PAT16"); timer = P.Timer()
    corr = P.correct(s, timer, device="mps")
    ti = prepare(corr.dwi, s.affine, s.bval, s.bvec, device="mps")
    D = U.from_arrays(ti.dwi, ti.header, ti.mask)
    trk = {"metal": U.track(D, backend="metal"), "f64": U.track(D, dtype=torch.float64, device="cpu", batch=P.CPU_BATCH, workers=8)}
    for name, lab in labs.items():
        L = {k: lab(f) for k, (f, _) in trk.items()}
        rows = {}
        for k, (f, st) in trk.items():
            r = np.full(len(f), -1); r[np.flatnonzero(L[k].keep)] = np.arange(int(L[k].keep.sum()))
            rows[k] = {int(s_): r[i] for i, s_ in enumerate(np.asarray(st["seed_index"])) if r[i] >= 0}
        both = sorted(set(rows["metal"]) & set(rows["f64"]))
        a = L["metal"].tract[[rows["metal"][k] for k in both]]; b = L["f64"].tract[[rows["f64"][k] for k in both]]
        res["pat16_trackers"][name] = {"other_metal_f64": [other(L["metal"].tract), other(L["f64"].tract)],
                                       "matched": len(both), "agreement": round(float((a == b).mean()), 4)}
        print("PAT16", name, json.dumps(res["pat16_trackers"][name]), flush=True)

    H = DATA / "ukf/hardi"
    img = nib.load(H / "HARDI150.nii.gz")
    h = SimpleNamespace(affine=img.affine, bval=np.loadtxt(H / "HARDI150.bval"), bvec=np.loadtxt(H / "HARDI150.bvec"))
    tg = P.track(h, np.asarray(img.dataobj), timer, device="mps", shell=2000.0)
    for name, lab in labs.items():
        per = [lab(tg.fibers, draws=(d,)).tract for d in range(5)]
        res["hardi"][name] = {"fibers": len(tg.fibers), "other_per_draw": [other(t) for t in per],
                              "other_sd": round(float(np.std([other(t) for t in per], ddof=1)), 4),
                              "agreement_draw0_draw1": round(float((per[0] == per[1]).mean()), 4)}
        print("HARDI", name, json.dumps(res["hardi"][name]), flush=True)
    (HERE / "results/label_trained_check.json").write_text(json.dumps(res, indent=1))
