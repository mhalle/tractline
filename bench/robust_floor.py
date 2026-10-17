"""The combined floor: step 1's seed-keyed grouping with step 2b's tie-break (docs/robustness-plan.md, steps 1 and 2b).

    uv run --group dipy --group rankfield bench/robust_floor.py [--out results/robust_floor.json]

Relabels existing compact records (no tracking) with RapidParc at draw 0, grouping "index" (today's) and "hash-sorted"
(step 1), for two trackers: today's (cIIIs04 A, DATA/ensemble/cIIIs04/retest/cIIIs04-2.1, and ensemble_stages'
epsilon and fieldonly r001..r008) and tie_angle 0.5 (robust_ukf_score.py's records,
DATA/ensemble/cIIIs04/robustness/ukf/score/tie_angle=0.5/). Per tracker, arm and grouping: each run against A relabeled
the same way, over the seeds both label: renamed (mean over runs), and pooled by how far the path moved
(ensemble_stages.by_distance's MDF bins). The stored labels' renames (robust_ukf_score's figure) are given beside the
index row; the relabel differs from them only through the records' 0.05 mm quantization.
Writes --out (refuses to overwrite); nothing under DATA.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from tractline import ensemble as E
from tractline.labelers import rapidparc as RP

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_stages import where, by_distance                            # noqa: E402
from robust_labeler import relabel, on_A, mdf_on_A, changed, L, DEVICE     # noqa: E402
from robust_ukf_score import SCORE, PERTURBED                              # noqa: E402

MODES = ("index", "hash-sorted")
VARIANT = "tie_angle=0.5"


def main(args):
    t0 = time.time()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists: refusing to overwrite (pass another --out)")
    d, sd = where()
    trackers = {"today": (d / "A.npz", lambda arm, r: sd / arm / f"r{r:03d}.npz"),
                VARIANT: (SCORE / VARIANT / "A.npz", lambda arm, r: SCORE / VARIANT / arm / f"r{r:03d}.npz")}
    lab = RP.Labeler(DEVICE)
    load = lambda p: E.RunRecord.load(p, logp=False, fibers=True)
    res = {"runs per arm": PERTURBED, "draw": 0, "modes": list(MODES), "trackers": {}}
    for name, (a_path, run_path) in trackers.items():
        A = load(a_path)
        SA, n = E.StreamlineEnsemble(A), len(A.keys)
        tA = {m: on_A(SA, n, *relabel(lab, A, m)) for m in MODES}
        row = {"A seeds labeled": n, "A: index relabel against the stored labels, renamed": round(changed(tA["index"], L[A.cluster]), 4)}
        for arm in ("epsilon", "fieldonly"):
            T, M, stored = {m: [] for m in MODES}, [], []
            for r in range(1, PERTURBED + 1):
                rec = load(run_path(arm, r))
                M.append(mdf_on_A(SA, A, rec))
                ri, qi = SA.match(rec)
                stored.append(float((L[A.cluster[ri]] != L[rec.cluster[qi]]).mean()))
                for m in MODES:
                    T[m].append(on_A(SA, n, *relabel(lab, rec, m)))
                print(name, arm, r, f"{time.time() - t0:.0f}s", flush=True)
            M = np.stack(M)
            row[arm] = {"stored labels (index), renamed (mean)": round(float(np.mean(stored)), 4),
                        "paths more than 5 mm apart (mean)": round(float(np.nanmean([(m[~np.isnan(m)] > 5).mean() for m in M])), 4)}
            for m in MODES:
                X = np.stack(T[m])
                per = [changed(x, tA[m]) for x in X]
                row[arm][m] = {"renamed: mean, SD over runs": [round(float(np.mean(per)), 4), round(float(np.std(per)), 4)],
                               "by path distance": by_distance(X, M, tA[m])}
            print(name, arm, json.dumps(row[arm]), flush=True)
        res["trackers"][name] = row
    res["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results/robust_floor.json"))
    main(ap.parse_args())
