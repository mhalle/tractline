"""M1b: is the FIELD worth keeping - does a margin decoded at a grouping chosen after inference
beat every scalar that could have been stored at inference?

    DATA/.venv/bin/python bench/tractography/regroup.py

M1 showed run 0's tract margin predicts tract instability, but one byte of that margin,
computed at inference for TractCloud's fixed 42 tracts, would do as well. The field's case is
regrouping later. For each grouping G below, the label of a streamline is the group of its
argmax cluster, the target is that label changing in any of runs 1..R-1 relative to run 0, and
the predictors are all from run 0:

    field   G mass margin, G best-class margin    decoded at G from the stored field
    fixed   p_max, cluster margin                  one byte each, no grouping
    fixed   tract mass margin                      one byte, TractCloud's 42-tract grouping

Headline per grouping: the G mass margin's AUROC minus the best fixed scalar's, with a paired
bootstrap 95 % CI. The margins here come from the full captured fp16 field (the numpy
reference); M2 checks that the rankfield encoding reproduces them.

Groupings:
    category          Association / Projection / Commissural / Cerebellar / Superficial / Other
    merged tracts     the 42 tracts with SLF-I/II/III -> SLF and CC1-7 -> CC
    tract, no outlier each outlier twin c+800 counted in its cluster's tract instead of Other
    cluster, no outlier  the 800 clusters, each with its outlier twin folded in

Writes results/m1b.json and results/m1b_regroup.png.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from _data import HCP
from tractcloud.tract_mapping import TRACT_NAMES, TRACT_CATEGORIES, _CLUSTER_TO_TRACT_LUT as TRACT

OUT = Path(__file__).resolve().parent / "results"
ref = dict(np.load(HCP / "reference.npz"))
cluster = ref["cluster"].astype(np.int64)            # (R, N)
R, N = cluster.shape
TRACT = TRACT.astype(np.int64)


def relabel(lut, names):
    """Compact a class -> group table to 0..G-1 (G = groups actually used)."""
    used, compact = np.unique(lut, return_inverse=True)
    return compact.astype(np.int64), [names[u] for u in used]


cat_of_tract = {t: c for c, ts in TRACT_CATEGORIES.items() for t in ts}
CATS = list(TRACT_CATEGORIES)
merge = {**{t: "SLF" for t in ("SLF-I", "SLF-II", "SLF-III")}, **{f"CC{i}": "CC" for i in range(1, 8)}}
MERGED = sorted({merge.get(t, t) for t in TRACT_NAMES})
no_outlier_tract = TRACT.copy()
no_outlier_tract[800:] = TRACT[:800]
GROUPINGS = {
    "tract (TractCloud's)": relabel(TRACT, TRACT_NAMES),
    "category": relabel(np.array([CATS.index(cat_of_tract[TRACT_NAMES[t]]) for t in TRACT]), CATS),
    "merged tracts (SLF, CC)": relabel(np.array([MERGED.index(merge.get(TRACT_NAMES[t], TRACT_NAMES[t]))
                                                 for t in TRACT]), MERGED),
    "tract, outliers folded in": relabel(no_outlier_tract, TRACT_NAMES),
    "cluster, outliers folded in": relabel(np.arange(1600) % 800, [str(c) for c in range(800)]),
}


def lse(x, axis):
    m = x.max(axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.log(np.exp(x - m).sum(axis, keepdims=True)) + m).squeeze(axis)


def margins(l, lut):
    """Mass and best-class margin of each streamline's own group (the group of its argmax)."""
    order = np.argsort(lut, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(lut[order]) != 0])
    lo = l[:, order]
    top = l.max(1, keepdims=True)
    with np.errstate(divide="ignore"):
        glse = np.log(np.add.reduceat(np.exp(lo - top), starts, axis=1)) + top
    gmax = np.maximum.reduceat(lo, starts, axis=1)
    own = lut[l.argmax(1)]
    rows = np.arange(len(l))
    out = {}
    for name, g, agg in (("mass", glse, lambda o: lse(o, 1)), ("best", gmax, lambda o: o.max(1))):
        other = g.copy()
        other[rows, own] = -np.inf
        out[name] = g[rows, own] - agg(other)
    return out


l0 = np.load(HCP / "logp_run0.npy").astype(np.float32)
l0 -= lse(l0, 1)[:, None]
assert (l0.argmax(1) == cluster[0]).all()
FIXED = {"p_max": ref["p_max"][0], "cluster margin": ref["cluster_margin"][0],
         "tract mass margin (42, fixed)": ref["m_mass"][0]}
auroc = lambda x, t: float(roc_auc_score(t, -x))

rng = np.random.default_rng(0)
B = 200
boots = [rng.integers(0, N, N) for _ in range(B)]
results = {}
for gname, (lut, names) in GROUPINGS.items():
    lab = lut[cluster]
    y = (lab[1:] != lab[0]).any(0)
    m = margins(l0, lut)
    preds = {"field mass margin": m["mass"], "field best-class margin": m["best"], **FIXED}
    au = {k: auroc(x, y) for k, x in preds.items()}
    best_fixed = max(FIXED, key=au.get)
    gain = np.array([auroc(m["mass"][i], y[i]) - auroc(FIXED[best_fixed][i], y[i]) for i in boots])
    results[gname] = {
        "groups": len(names), "changed_fraction": round(float(y.mean()), 4),
        "auroc_run0": {k: round(v, 4) for k, v in au.items()},
        "best_fixed": best_fixed,
        "gain_over_best_fixed": round(au["field mass margin"] - au[best_fixed], 4),
        "gain_ci95": [round(float(v), 4) for v in np.quantile(gain, [0.025, 0.975])],
    }
    print(gname, json.dumps(results[gname]), flush=True)

summary = {"streamlines": N, "runs": R, "bootstrap": B,
           "target": "the group of the argmax cluster differs from run 0 in any of runs 1..R-1",
           "groupings": results}
(OUT / "m1b.json").write_text(json.dumps(summary, indent=1))

# ---- figure: dumbbell, field margin at G vs the best scalar fixed at inference ----
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FIELD_C, FIXED_C = "#2a78d6", "#eb6834"                 # palette slots 1, 2
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=150)
rows = list(results)[::-1]
for i, g in enumerate(rows):
    a, b = results[g]["auroc_run0"]["field mass margin"], results[g]["auroc_run0"][results[g]["best_fixed"]]
    ax.plot([b, a], [i, i], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    ax.scatter([b], [i], s=64, color=FIXED_C, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="best scalar stored at inference" if i == 0 else None)
    ax.scatter([a], [i], s=64, color=FIELD_C, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="field margin at this grouping" if i == 0 else None)
    ax.text(a + 0.006, i, f"{a:.3f}", va="center", fontsize=8, color=INK2)
    ax.text(b, i - 0.22, f"{b:.3f}  {results[g]['best_fixed']}", va="top", ha="right",
            fontsize=7, color=INK2)
ax.set_yticks(range(len(rows)))
ax.set_ylim(-0.7, len(rows) - 0.5)
ax.set_yticklabels([f"{g}  ({results[g]['changed_fraction'] * 100:.0f} % change)" for g in rows])
ax.set_xlim(0.45, 0.95)
ax.set_xlabel("AUROC for a label change at this grouping, from run 0 alone")
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
ax.tick_params(length=0)
ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, labelcolor=INK)
fig.tight_layout()
fig.savefig(OUT / "m1b_regroup.png")
