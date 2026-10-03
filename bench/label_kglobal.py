"""TractCloud's global context at inference: upstream's packaged inference uses k_global = 80, the released
model was trained with 500 (TrainedModel/cli_args.txt; the paper's w = 500). PAT16's Metal tractogram (the
M2's field), labeled with each: the Other fraction over draws 0-9, per-streamline agreement between single
draws (0 vs 1, 2 vs 3, ...), between disjoint 5-draw ensembles (cluster probabilities averaged), and
against each other; the time per draw.

    uv run bench/label_kglobal.py

Writes results/label_kglobal.json.
"""
import json, time
from pathlib import Path
import numpy as np, torch
from tractline import pipeline as P, ukf as U
from tractline.prep import prepare
from tractline.labelers.tractcloud import Labeler, inf, MODEL
from label_draws import Draws, other
from _ds001226 import load

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    s = load("PAT16"); timer = P.Timer()
    corr = P.correct(s, timer, device="mps")
    ti = prepare(corr.dwi, s.affine, s.bval, s.bvec, device="mps")
    fibers, _ = U.track(U.from_arrays(ti.dwi, ti.header, ti.mask), backend="metal")
    res = {"subject": "PAT16", "fibers": len(fibers), "k_global": {}}
    ens = {}
    for kg in (80, 500):
        lab = Labeler("mps")
        lab.model, _ = inf.load_model(str(MODEL / "best_tract_f1_model.pth"), str(MODEL / "cli_args.txt"), lab.device,
                                      k_override=20, k_global_override=kg)
        d = Draws(lab, fibers, k_global=kg)
        t0 = time.time(); lps = []
        for i in range(10):
            lps.append(d.logp(i))
        secs = round((time.time() - t0) / 10, 2)
        tr = [lab.lut[lp.argmax(1).numpy()] for lp in lps]
        pairs = [round(float((tr[i] == tr[i + 1]).mean()), 4) for i in range(0, 10, 2)]
        A = lab.lut[sum(lp.exp() for lp in lps[:5]).argmax(1).numpy()]
        B = lab.lut[sum(lp.exp() for lp in lps[5:]).argmax(1).numpy()]
        ens[kg] = (A, B)
        res["k_global"][kg] = {"seconds_per_draw": secs, "other_per_draw": [other(t) for t in tr],
                               "other_sd": round(float(np.std([other(t) for t in tr], ddof=1)), 4),
                               "single_draw_agreement": pairs, "ensemble5_agreement": round(float((A == B).mean()), 4),
                               "ensemble5_other": [other(A), other(B)]}
        print(kg, json.dumps(res["k_global"][kg]), flush=True)
    res["ensemble5_80_vs_500"] = round(float((ens[80][0] == ens[500][0]).mean()), 4)
    print(json.dumps({"ensemble5_80_vs_500": res["ensemble5_80_vs_500"]}))
    (HERE / "results/label_kglobal.json").write_text(json.dumps(res, indent=1))
