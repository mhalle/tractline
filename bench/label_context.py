"""TractCloud's context settings at inference, on PAT16's Metal tractogram (the M2's field): global
streamlines k_global and the local subsample k_ds_rate. The released model was trained with k_global =
500 and local neighbors among all of a 10,000-streamline brain (k_ds_rate 1.0); upstream's packaged
inference uses 80 and 0.1. Per setting, draws 0-9: the Other fraction (and its SD), per-streamline
agreement between single draws (0 vs 1, 2 vs 3, ...), between disjoint 5-draw ensembles (cluster
probabilities averaged), each setting's 5-draw ensemble against the others', and the time per draw.

    python bench/label_context.py

Writes results/label_context.json.
"""
import json, time
from pathlib import Path
import numpy as np
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
    res = {"subject": "PAT16", "fibers": len(fibers), "settings": {}}
    ens = {}
    n_kept = None
    for name, kg, ds in (("80, 0.1 (upstream's inference)", 80, 0.1), ("500, 0.1", 500, 0.1),
                         ("500, training density (10,000 candidates)", 500, None), ("500, 1.0 (all)", 500, 1.0)):
        lab = Labeler("mps")
        lab.model, _ = inf.load_model(str(MODEL / "best_tract_f1_model.pth"), str(MODEL / "cli_args.txt"), lab.device,
                                      k_override=20, k_global_override=kg)
        d = Draws(lab, fibers, k_global=kg)
        n_kept = int(d.keep.sum())
        d.k_ds_rate = ds if ds is not None else min(1.0, 10000 / n_kept)
        t0 = time.time(); lps = [d.logp(i) for i in range(10)]
        secs = round((time.time() - t0) / 10, 2)
        tr = [lab.lut[lp.argmax(1).numpy()] for lp in lps]
        A = lab.lut[sum(lp.exp() for lp in lps[:5]).argmax(1).numpy()]
        B = lab.lut[sum(lp.exp() for lp in lps[5:]).argmax(1).numpy()]
        ens[name] = A
        res["settings"][name] = {"k_global": kg, "k_ds_rate": round(d.k_ds_rate, 4), "seconds_per_draw": secs,
                                 "other_per_draw": [other(t) for t in tr], "other_sd": round(float(np.std([other(t) for t in tr], ddof=1)), 4),
                                 "single_draw_agreement": [round(float((tr[i] == tr[i + 1]).mean()), 4) for i in range(0, 10, 2)],
                                 "ensemble5_agreement": round(float((A == B).mean()), 4), "ensemble5_other": [other(A), other(B)]}
        print(name, json.dumps(res["settings"][name]), flush=True)
    names = list(ens)
    res["ensemble5_between_settings"] = {f"{a} | {b}": round(float((ens[a] == ens[b]).mean()), 4) for i, a in enumerate(names) for b in names[i + 1:]}
    print(json.dumps(res["ensemble5_between_settings"], indent=1))
    (HERE / "results/label_context.json").write_text(json.dumps(res, indent=1))
