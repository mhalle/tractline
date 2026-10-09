"""A one-page viewer of PAT16's tracts at its tumor, with the ensemble's envelopes (NOTES 2026-10-08, "PAT16 at the
tumor").

    uv run --group dipy bench/ensemble_pat16_viewer.py OUT.html [--flip smooth]

Reads ensemble_pat16_report.py's DATA/ensemble/PAT16/report_<flip>.npz and results/ensemble_pat16_report_<flip>.json,
ensemble_coverage.py's results/ensemble_coverage_<flip>.json (the envelope levels calibrated on MASiVar's rescans)
and ensemble_where_which.py's results/ensemble_where_which_<flip>.json. Axial slices through the tumor (radiological:
the patient's right on the left, anterior up) over the corrected mean b0. For each shown tract:
  - today: the voxels r000 (the pipeline as it runs) gives it;
  - envelope: the voxels whose share of runs is at least the threshold that holds the chosen level of its mass over
    the whole volume (ties at a share included). Calibrated: the level that covered the chosen one on real rescans.
Self-contained HTML (the data inline, base64).
"""
import argparse, base64, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
NAMES = {"AF": "arcuate fasciculus", "ILF": "inferior longitudinal fasciculus", "IOFF": "inferior occipito-frontal fasciculus",
         "MdLF": "middle longitudinal fasciculus", "UF": "uncinate fasciculus", "CST": "corticospinal tract",
         "SF": "striato-frontal", "SO": "striato-occipital", "SP": "striato-parietal", "TF": "thalamo-frontal",
         "Sup-F": "superficial frontal", "Sup-T": "superficial temporal", "Sup-FP": "superficial fronto-parietal",
         "CC1": "corpus callosum 1", "CC6": "corpus callosum 6", "CB": "cingulum bundle", "SLF-III": "superior longitudinal fasciculus III",
         "EC": "external capsule", "EmC": "extreme capsule", "PLIC": "posterior limb of internal capsule", "CR-F": "corona radiata, frontal"}
COLORS = ["#f2c14e", "#5fd0c5", "#e48ad6", "#8fb8ff", "#b6e36b", "#ff9b54", "#c9a7ff", "#7fe0a0", "#ff7a8a"]
SHOWN = ("ILF", "Sup-F", "SF", "MdLF", "CC1", "TF", "CB", "AF", "CST")   # the tracts that surely enter, those that enter in some
                                                                       # runs, and the arcuate and corticospinal tracts nearby
LEVELS = [round(x, 2) for x in np.arange(0.5, 0.951, 0.05)]


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def mean_b0():
    from tractline import pipeline as P
    from _ds001226 import load
    s = load("PAT16")
    with P.exact_float32("mps"):
        corr = P.correct(s, P.Timer(), "mps")
    return np.asarray(corr.dwi, np.float32)[..., s.b0_index].mean(-1)


def threshold(p, level):
    """The smallest share t whose voxels {p >= t} hold `level` of p's mass (p in multiples of 1/R)."""
    vals = np.unique(p[p > 0])[::-1]
    mass = np.array([p[p >= v].sum() for v in vals])
    i = int(np.searchsorted(mass, level * p.sum()))
    return float(vals[min(i, len(vals) - 1)])


def main(args):
    from tractline.data import DATA
    from tractline.labelers.base import TRACT_NAMES
    z = np.load(DATA / "ensemble" / "PAT16" / f"report_{args.flip}.npz")
    P, dist, tumor, today = z["P"].astype(np.float32), z["dist"], z["tumor"], z["today"]
    rep = json.loads((HERE / f"results/ensemble_pat16_report_{args.flip}.json").read_text())
    cov = json.loads((HERE / f"results/ensemble_coverage_{args.flip}.json").read_text())
    ww = json.loads((HERE / f"results/ensemble_where_which_{args.flip}.json").read_text())
    runs = rep["runs"]
    shown = [t for t in SHOWN if t in rep["tracts"]]
    zs = np.flatnonzero(tumor.any((0, 1)))
    z0, z1 = max(int(zs.min()) - 3, 0), min(int(zs.max()) + 4, tumor.shape[2])
    curve = cov["recalibration on real rescans (leave one pair out)"]["coverage curve, all rescan scorings (level: coverage)"]
    cl = np.array([float(k) for k in curve]); cc = np.array(list(curve.values()))
    fine = cov["summary"]["rescan"]["envelope coverage"]

    def calibrated(L):
        ok = np.flatnonzero(cc >= L - 1e-9)
        return (float(cl[ok[0]]), True) if len(ok) else (float(cl[-1]), False)

    b0 = mean_b0()[:, :, z0:z1]
    hi = np.quantile(b0[b0 > 0], 0.995)
    img = np.clip(b0 / hi * 255, 0, 255).astype(np.uint8)
    region = np.where(tumor, 0, np.where(dist <= 5, 1, 2)).astype(np.uint8)[:, :, z0:z1]
    tracts = []
    for i, name in enumerate(shown):
        k = TRACT_NAMES.index(name)
        votes = np.rint(P[k] * runs).astype(np.uint8)
        thr = {"stated": [round(threshold(P[k], L) * runs) for L in LEVELS], "calibrated": []}
        for L in LEVELS:
            use, _ = calibrated(L)
            thr["calibrated"].append(round(threshold(P[k], use) * runs))
        row = rep["tracts"][name]
        tracts.append({"name": name, "full": NAMES.get(name, name), "color": COLORS[i], "votes": b64(votes[:, :, z0:z1]),
                       "k": k, "thr": thr,
                       "enter": row["runs: P(enters the tumor)"], "within5": row["runs: P(within 5 mm)"],
                       "ml_in": row["runs: mL inside the tumor, median [min, max]"], "ml_in_today": row["today (r000): mL inside the tumor"],
                       "ml_near": row["runs: mL within 5 mm, median [min, max]"], "ml_near_today": row["today (r000): mL within 5 mm"]})
    s = cov["summary"]["rescan"]
    pat = ww["PAT16"][f"replicate vs replicate ({args.flip}), tumor margin"]
    data = {"shape": [int(v) for v in img.shape], "z0": z0, "runs": runs, "img": b64(img), "region": b64(region), "tracts": tracts,
            "today": b64(today[:, :, z0:z1].astype(np.uint8)),
            "levels": LEVELS, "calib": [{"target": L, "use": calibrated(L)[0], "reached": calibrated(L)[1]} for L in LEVELS],
            "curve": {"levels": cl.tolist(), "coverage": cc.tolist()},
            "fine": fine,
            "stats": {"auroc_single": s["auroc"]["single run: labeler probability"], "auroc_mix": s["auroc"]["ensemble: mixture"],
                      "ece_single": s["ece"]["single run: labeler probability"], "ece_mix": s["ece"]["ensemble: mixture"],
                      "change": s["change_rate"], "scorings": s["scorings"]},
            "margin": {"density": pat["spheres: where: density intersection (quartiles)"][1], "direction": pat["spheres: where: direction |cos| (quartiles)"][1],
                       "dominant": pat["spheres: which: same dominant tract"], "mix_tv": pat["spheres: which: composition total variation (quartiles)"][1],
                       "streamline_keep": round(1 - s["change_rate"], 3)},
            "mix": rep["margin 0-10 mm: tract mix (mean, min-max over runs)"]}
    html = (HERE / "pat16_viewer.html").read_text().replace("/*DATA*/null", json.dumps(data))
    Path(args.out).write_text(html)
    print(f"{args.out}: {len(html) / 1e6:.2f} MB, slices {z0}..{z1 - 1}, tracts {shown}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--flip", default="smooth")
    main(ap.parse_args())
