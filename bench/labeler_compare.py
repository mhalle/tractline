"""The labelers side by side on the 12 ds001226 patients: each patient tracked once (the default
pipeline: correction, the Metal tracker), the same tractogram labeled by TractCloud at upstream's
inference context (80) and at the trained one (500), RapidParc and RapidParc's hemiaug model, each at
seeds (draws) 0 and 1.

Per patient and labeler: the Other fraction (seed 0, seed 1), per-streamline agreement between the
two seeds (stability), the seconds a draw; between labelers (seed 0): per-streamline agreement and
the tract mix (Pearson r of the 42 tracts' counts).

    python bench/labeler_compare.py [--subs PAT16,PAT13]

Writes results/labeler_compare.json, merged patient by patient: run one process per patient (the
labelers' GPU memory and the Metal allocator's cache grow across patients in one process - 13 GB on
the 16 GB M2 after five, swapping).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from tractline import pipeline as P
from tractline.labelers import tractcloud, rapidparc
from tractline.labelers.base import OTHER
from _ds001226 import load

HERE = Path(__file__).resolve().parent
SUBS = "PAT05,PAT07,PAT08,PAT13,PAT14,PAT16,PAT19,PAT20,PAT23,PAT25,PAT26,PAT29"
OUT = HERE / "results/labeler_compare.json"


def other(t):
    return round(float((t == OTHER).mean()), 4)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--subs", default=SUBS); args = ap.parse_args()
    labelers = {"tractcloud_80": tractcloud.Labeler("mps", upstream=True), "tractcloud_500": tractcloud.Labeler("mps"),
                "rapidparc": rapidparc.Labeler("mps"), "rapidparc_hemiaug": rapidparc.Labeler("mps", model="hemiaug")}
    res = json.loads(OUT.read_text()) if OUT.exists() else {"labelers": list(labelers), "patients": {}}
    for sub in args.subs.split(","):
        s = load(sub); timer = P.Timer()
        corr = P.correct(s, timer, device="mps")
        tg = P.track(s, corr.dwi, timer, device="mps")
        row = {"fibers": len(tg.fibers), "labelers": {}, "between": {}}
        lab0 = {}
        for name, lab in labelers.items():
            t0 = time.time(); a = lab(tg.fibers, draws=(0,)).tract; secs = round(time.time() - t0, 1)
            b = lab(tg.fibers, draws=(1,)).tract
            lab0[name] = a
            row["labelers"][name] = {"labeled": len(a), "other": [other(a), other(b)], "seed_agreement": round(float((a == b).mean()), 4), "seconds": secs}
        names = list(lab0)
        for i, x in enumerate(names):
            for y in names[i + 1:]:
                cx, cy = np.bincount(lab0[x], minlength=43)[:OTHER], np.bincount(lab0[y], minlength=43)[:OTHER]
                row["between"][f"{x} | {y}"] = {"agreement": round(float((lab0[x] == lab0[y]).mean()), 4),
                                                "tract_mix_r": round(float(np.corrcoef(cx, cy)[0, 1]), 4)}
        del corr, tg; P.release_memory("mps")                               # see pipeline.release_memory
        res["patients"][sub] = row
        print(sub, json.dumps({n: (v["other"], v["seed_agreement"], v["seconds"]) for n, v in row["labelers"].items()}), flush=True)
        OUT.write_text(json.dumps(res, indent=1))
