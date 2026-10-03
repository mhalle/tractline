"""How much does TractCloud's Other fraction move on PAT16, and is the Metal tracker's lower Other
fraction a bias or noise? The M2's field held fixed (pipeline.correct, the default estimate), then:

  - TractCloud's own spread: the Other fraction of one tractogram (Metal, the scan as acquired) over
    context draws 0-9;
  - rounding alone: the float64 tracker at batch 1,024 and 4,096 (only the summation order differs);
  - the scan's noise: wild-bootstrap replicates of the corrected tracker input (_bootstrap.py, SH order
    6, as ukf_noise_floor.py), each tracked by the Metal kernel and by the float64 tracker (CPU) and
    labeled (draw 0) - Metal minus float64 per replicate says whether the gap keeps its sign;
  - per streamline, the scan as acquired: Metal's and float64's fibers matched by seed, their labels
    compared, and where they disagree, which way (tract -> Other or Other -> tract) and how the two
    fibers' lengths differ.

    python bench/label_noise_floor.py [--replicates 4]

TractCloud at upstream's inference context (k_global 80, 10 %), as when it ran (Labeler(upstream=True)).

Writes results/label_noise_floor.json as it goes.
"""
import argparse, json, re, time
from pathlib import Path
import numpy as np, torch
from tractline import pipeline as P, ukf as U
from tractline.prep import prepare
from tractline.labelers.tractcloud import Labeler
from _ds001226 import load
from _bootstrap import WildBootstrap

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/label_noise_floor.json"


def other(t):
    return round(float((t == 42).mean()), 4)


def length_mm(f):
    return float(np.linalg.norm(np.diff(f, axis=0), axis=1).sum())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--replicates", type=int, default=4); args = ap.parse_args()
    s = load("PAT16")
    timer = P.Timer()
    corr = P.correct(s, timer, device="mps")
    t = prepare(corr.dwi, s.affine, s.bval, s.bvec, device="mps")
    keys = sorted(k for k in t.header if k.startswith("DWMRI_gradient_"))
    g = np.array([[float(v) for v in t.header[k].split()] for k in keys])
    bmax = int(re.match(r"\s*(-?\d+)", t.header["DWMRI_b-value"]).group(1))
    wb = WildBootstrap(np.asarray(t.dwi), g, (bmax * (g * g).sum(1)) <= 50, sh_order=6)
    lab = Labeler("mps", upstream=True)                                   # as the committed result: upstream's context
    res = {"subject": "PAT16", "labeler": "TractCloud, upstream's context (k_global 80, 10 %)", "field": "pipeline.correct (mps)", "bootstrap": wb.noise_check(np.asarray(t.mask)), "replicates": []}

    def track(data, kind, batch=P.CPU_BATCH):
        D = U.from_arrays(data, t.header, t.mask)
        t0 = time.time()
        if kind == "metal":
            f, st = U.track(D, backend="metal")
        else:
            f, st = U.track(D, dtype=torch.float64, device="cpu", batch=batch, workers=8)
        return f, st, round(time.time() - t0, 1)

    def save():
        OUT.write_text(json.dumps(res, indent=1))

    # the scan as acquired, both trackers; TractCloud's draws; float64 rounding; per-streamline
    fm, sm, tm = track(np.asarray(t.dwi), "metal")
    ff, sf, tf = track(np.asarray(t.dwi), "f64")
    Lm, Lf = lab(fm, draws=(0,)), lab(ff, draws=(0,))
    res["draws_metal"] = [other(lab(fm, draws=(d,)).tract) for d in range(10)]
    save()
    ff2, _, _ = track(np.asarray(t.dwi), "f64", batch=4096)
    Lf2 = lab(ff2, draws=(0,))
    res["rounding_f64"] = {"batch_1024": other(Lf.tract), "batch_4096": other(Lf2.tract),
                           "fibers": [len(ff), len(ff2)], "identical_fibers": len(ff) == len(ff2) and all(np.array_equal(a, b) for a, b in zip(ff, ff2))}
    # match by seed: the tracker's seed_index per fiber; labels for kept (>= 40 mm) fibers only
    def by_seed(f, st, L):
        rows = np.full(len(f), -1); rows[np.flatnonzero(L.keep)] = np.arange(int(L.keep.sum()))
        return {int(k): (f[i], int(L.tract[rows[i]]) if rows[i] >= 0 else None) for i, k in enumerate(np.asarray(st["seed_index"]))}
    A, B = by_seed(fm, sm, Lm), by_seed(ff, sf, Lf)
    both = [k for k in A if k in B and A[k][1] is not None and B[k][1] is not None]
    same = sum(A[k][1] == B[k][1] for k in both)
    to_other = [k for k in both if A[k][1] == 42 and B[k][1] != 42]          # Metal Other, float64 a tract
    from_other = [k for k in both if A[k][1] != 42 and B[k][1] == 42]
    dl = lambda ks: round(float(np.median([length_mm(A[k][0]) - length_mm(B[k][0]) for k in ks])), 1) if ks else None
    res["scan_as_acquired"] = {
        "metal": {"fibers": len(fm), "labeled": int(Lm.keep.sum()), "other": other(Lm.tract), "seconds": tm},
        "f64": {"fibers": len(ff), "labeled": int(Lf.keep.sum()), "other": other(Lf.tract), "seconds": tf},
        "matched_labeled": len(both), "label_agreement": round(same / len(both), 4),
        "metal_other_f64_tract": len(to_other), "metal_tract_f64_other": len(from_other),
        "median_length_diff_mm_metal_minus_f64": {"metal_other_f64_tract": dl(to_other), "metal_tract_f64_other": dl(from_other),
                                                  "all_matched": dl(both)},
        "only_in_metal_labeled": sum(1 for k in A if A[k][1] is not None and (k not in B or B[k][1] is None)),
        "only_in_f64_labeled": sum(1 for k in B if B[k][1] is not None and (k not in A or A[k][1] is None))}
    save()
    print(json.dumps({k: res[k] for k in ("draws_metal", "rounding_f64", "scan_as_acquired")}, indent=1), flush=True)
    for r in range(1, args.replicates + 1):
        data = wb.replicate(r)
        row = {"seed": r}
        for kind in ("metal", "f64"):
            f, st, secs = track(data, kind)
            L = lab(f, draws=(0,))
            row[kind] = {"fibers": len(f), "labeled": int(L.keep.sum()), "other": other(L.tract), "seconds": secs}
        row["metal_minus_f64"] = round(row["metal"]["other"] - row["f64"]["other"], 4)
        res["replicates"].append(row); save()
        print(json.dumps(row), flush=True)
