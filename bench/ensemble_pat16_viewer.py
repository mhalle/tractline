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
NAMES = {"AF": "arcuate fasciculus", "CB": "cingulum bundle", "EC": "external capsule", "EmC": "extreme capsule",
         "ILF": "inferior longitudinal fasciculus", "IOFF": "inferior occipito-frontal fasciculus", "MdLF": "middle longitudinal fasciculus",
         "SLF-I": "superior longitudinal fasciculus I", "SLF-II": "superior longitudinal fasciculus II",
         "SLF-III": "superior longitudinal fasciculus III", "UF": "uncinate fasciculus", "CST": "corticospinal tract",
         "CR-F": "corona radiata, frontal", "CR-P": "corona radiata, parietal", "SF": "striato-frontal", "SO": "striato-occipital",
         "SP": "striato-parietal", "TF": "thalamo-frontal", "TO": "thalamo-occipital", "TT": "thalamo-temporal", "TP": "thalamo-parietal",
         "PLIC": "posterior limb of the internal capsule", **{f"CC{i}": f"corpus callosum, segment {i}" for i in range(1, 8)},
         "CPC": "cortico-ponto-cerebellar", "ICP": "inferior cerebellar peduncle", "MCP": "middle cerebellar peduncle",
         "Sup-F": "superficial frontal", "Sup-FP": "superficial fronto-parietal", "Sup-O": "superficial occipital",
         "Sup-OT": "superficial occipito-temporal", "Sup-P": "superficial parietal", "Sup-PO": "superficial parieto-occipital",
         "Sup-PT": "superficial parieto-temporal", "Sup-T": "superficial temporal"}
COLORS = ["#f2c14e", "#5fd0c5", "#e48ad6", "#8fb8ff", "#b6e36b", "#ff9b54", "#c9a7ff", "#7fe0a0", "#ff7a8a"]
SHOWN = ("ILF", "Sup-F", "SF", "MdLF", "CC1", "TF", "CB", "AF", "CST")   # drawn: tracts that enter in every run, some that enter in
                                                                       # some runs, and the arcuate and corticospinal tracts nearby
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

    def calibrated(L):
        """The stated level whose envelope covered L of the rescans' tracts, interpolated (as ensemble_pat16_report.py);
        beyond what the widest measured level covered, that level, flagged."""
        if L > cc.max():
            return float(cl[-1]), False
        return float(np.interp(L, cc, cl)), True

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
        tracts.append({"name": name, "full": NAMES.get(name, name), "color": COLORS[i], "votes": b64(votes[:, :, z0:z1]), "k": k, "thr": thr})
    s = cov["summary"]["rescan"]
    pat = ww["PAT16"][f"replicate vs replicate ({args.flip}), tumor margin"]
    sp = lambda key: pat[f"spheres: {key}"]  # noqa: E731
    table = [{"name": n, "full": NAMES.get(n, n), "color": COLORS[shown.index(n)] if n in shown else None,
              "enter": r["runs: P(enters the tumor)"], "within5": r["runs: P(within 5 mm)"],
              "ml_in": r["runs: mL inside the tumor, median [min, max]"], "ml_in_single": r["today (r000): mL inside the tumor"]}
             for n, r in rep["tracts"].items() if r["runs: P(enters the tumor)"] > 0 or r["runs: P(within 5 mm)"] >= 0.5]
    table.sort(key=lambda r: (-r["enter"], -r["within5"]))
    data = {"shape": [int(v) for v in img.shape], "z0": z0, "runs": runs, "img": b64(img), "region": b64(region), "tracts": tracts,
            "single": b64(today[:, :, z0:z1].astype(np.uint8)), "table": table,
            "levels": LEVELS, "calib": [{"target": L, "use": calibrated(L)[0], "reached": calibrated(L)[1]} for L in LEVELS],
            "curve": {"levels": cl[::4].tolist() + [float(cl[-1])], "coverage": cc[::4].tolist() + [float(cc[-1])]},
            "rescans": {"auroc_single": s["auroc"]["single run: labeler probability"], "auroc_mix": s["auroc"]["ensemble: mixture"],
                        "ece_single": s["ece"]["single run: labeler probability"], "ece_mix": s["ece"]["ensemble: mixture"],
                        "keep": round(1 - s["change_rate"], 3), "scorings": s["scorings"]},
            "margin": {"density": sp("where: density intersection (quartiles)")[1],
                       "angle_deg": round(float(np.degrees(np.arccos(sp("where: direction |cos| (quartiles)")[1]))), 1),
                       "dominant": sp("which, named tracts (Other excluded): same dominant tract"),
                       "mix_shared": round(1 - sp("which, named tracts (Other excluded): composition total variation (quartiles)")[1], 4),
                       "other_share": sp("Other: share of a sphere's streamlines (quartiles)")[1], "spheres": pat["pairs"]},
            "mix": [[n, NAMES.get(n, "assigned to none of the 42 tracts" if n == "Other" else n), *v]
                    for n, v in rep["margin 0-10 mm: tract mix (mean, min-max over runs)"].items()]}
    html = (HERE / "pat16_viewer.html").read_text().replace("/*DATA*/null", json.dumps(data))
    Path(args.out).write_text(html)
    print(f"{args.out}: {len(html) / 1e6:.2f} MB, slices {z0}..{z1 - 1}, tracts {shown}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--flip", default="smooth")
    main(ap.parse_args())
