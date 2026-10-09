"""Do compact run records (tractline.compact, format 2) give the ensemble's results? (NOTES 2026-10-09, "Compact
run records")

    uv run --group dipy --group rankfield bench/compact_check.py [--pair cIIIs01-2.5] [--tmp DIR]

From ensemble_retest.py's records of one pair: each run of the reference's and the rescan's ensembles (smooth signs)
is written again as a compact record - its log-probabilities as the depth-12 rank field; its 20-point paths stand in
for the streamlines (format 1 files kept no more) - and read back. The pair is then scored both ways, as
ensemble_coverage.py does, from the dense and from the compact records: change, AUROC and ECE of the stated
probabilities, voxel calibration and envelope coverage, and the bytes. Writes results/compact_check.json.
"""
import argparse, json, sys, tempfile, time
from pathlib import Path
import numpy as np
from tractline import ensemble as E
from tractline.data import DATA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS  # noqa: E402
from ensemble_coverage import Ensemble, score, winners  # noqa: E402


def recompact(src: Path, dst: Path):
    r = E.RunRecord.load(src)
    r.fibers = list(r.paths)
    r.save(dst, compact=True)
    return src.stat().st_size, dst.stat().st_size


def main(args):
    subject, _, rescans, _, _ = PAIRS[args.pair]
    d = DATA / "ensemble" / subject / "retest" / args.pair
    tag = next(iter(rescans))
    groups = {"A": [d / "A.npz"] + [d / "smooth" / f"r{r:03d}.npz" for r in range(1, 9)],
              tag: [d / f"{tag}_onA.npz"] + [d / "smooth" / f"{tag}_r{r:03d}.npz" for r in range(1, 9)]}
    tmp = Path(args.tmp or tempfile.mkdtemp())
    sizes, compact = [], {}
    t0 = time.time()
    for g, paths in groups.items():
        compact[g] = []
        for p in paths:
            q = tmp / f"{g}_{p.stem}.npz"
            sizes.append(recompact(p, q))
            compact[g].append(q)
    res = {"pair": args.pair, "rescan": tag, "runs": sum(len(v) for v in groups.values()), "recompact_s": round(time.time() - t0, 1),
           "bytes per run: dense, compact (MB)": [round(float(np.mean([s[0] for s in sizes])) / 1e6, 2), round(float(np.mean([s[1] for s in sizes])) / 1e6, 2)]}
    for name, src in (("dense", groups), ("compact", compact)):
        A, B = Ensemble(src["A"]), Ensemble(src[tag])
        sa, sb = score(A, *winners(src[tag][0])), score(B, *winners(src["A"][0]))
        pick = lambda s: {"change": s["change_rate"], "auroc mixture": s["auroc predicting change"]["ensemble: mixture"],  # noqa: E731
                          "auroc one run": s["auroc predicting change"]["single run: labeler probability"],
                          "ece mixture": s["calibration"]["ensemble: mixture"]["ece"], "mean P(keep) mixture": s["mean stated P(keep)"]["ensemble: mixture"],
                          "voxel ece": s["voxels"]["calibration: P(v, k) against the other run's map (P > 0)"]["ece"],
                          "coverage 0.9": s["voxels"]["envelope coverage (ensemble)"]["0.9"]}
        res[name] = {"A scored on the rescan": pick(sa), "the rescan scored on A": pick(sb)}
        print(name, json.dumps(res[name]), flush=True)
    (HERE / "results/compact_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("dense", "compact")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="cIIIs01-2.5", choices=sorted(PAIRS))
    ap.add_argument("--tmp", default=None)
    main(ap.parse_args())
