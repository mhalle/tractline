"""M1: does a single run's tract margin predict tract-label instability, and better than one byte?

    DATA/.venv/bin/python bench/tractography/analyze.py

Target: the tract label of a streamline changes in at least one of runs 1..R-1 relative to
run 0 (reference.npz `tract_changed`). Predictors, ALL FROM RUN 0 ALONE - TractCloud is run
once in practice:

    tract best-class margin  m_hat     (what rankfield's decode_groups gives)
    tract mass margin        m_mass    (log-odds of the tract as a whole)
    baseline p_max           the winning cluster's probability    (one byte, no field)
    baseline cluster margin  top-1 minus top-2 log-probability    (one byte, no field)

Low predictor = predicted unstable, so each AUROC scores the negated predictor. The headline
is the best tract margin's AUROC minus the best baseline's, with a paired bootstrap 95 % CI.
Secondary: the margins averaged over all R runs (uses the ensemble a user would not have),
and the plausibility margin against a change of outlier status (cluster >= 800 or not).

Writes results/m1.json, results/m1_instability.png, results/m1_pairs.png (beside this script:
small, tracked) from DATA/hcp/reference.npz.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from _data import HCP
from tractcloud.tract_mapping import TRACT_NAMES

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
ref = dict(np.load(HCP / "reference.npz"))
y = ref["tract_changed"]
tract = ref["tract"]
R, N = tract.shape

PRED = {                                   # name -> (run-0 values, kind)
    "tract best-class margin": (ref["m_hat"][0], "field"),
    "tract mass margin": (ref["m_mass"][0], "field"),
    "p_max (winning cluster)": (ref["p_max"][0], "baseline"),
    "cluster margin": (ref["cluster_margin"][0], "baseline"),
}
auroc = lambda x, t=y: float(roc_auc_score(t, -x))

# ---- AUROC and the headline gain, paired bootstrap ----
rng = np.random.default_rng(0)
B = 200
boot = {k: [] for k in PRED}
for _ in range(B):
    i = rng.integers(0, N, N)
    for k, (x, _) in PRED.items():
        boot[k].append(auroc(x[i], y[i]))
boot = {k: np.array(v) for k, v in boot.items()}
au = {k: auroc(x) for k, (x, _) in PRED.items()}
best_field = max((k for k in PRED if PRED[k][1] == "field"), key=au.get)
best_base = max((k for k in PRED if PRED[k][1] == "baseline"), key=au.get)
gain = boot[best_field] - boot[best_base]

# ---- fraction changed by predictor percentile (20 bins, least confident first) ----
BINS = 20
curves = {}
for k, (x, _) in PRED.items():
    order = np.argsort(x, kind="stable")
    curves[k] = [float(y[part].mean()) for part in np.array_split(order, BINS)]
dec = np.array([y[p].mean() for p in np.array_split(np.argsort(PRED[best_field][0], kind="stable"), 10)])

# ---- fraction changed by best-class margin in logits ----
edges = [0, 0.25, 0.5, 1, 2, 4, 8, np.inf]
m = ref["m_hat"][0]
by_logit = []
for lo, hi in zip(edges[:-1], edges[1:]):
    sel = (m >= lo) & (m < hi)
    by_logit.append({"m_hat": f"[{lo}, {hi})", "n": int(sel.sum()),
                     "changed": round(float(y[sel].mean()), 4) if sel.any() else None})

# ---- which tract pairs change ----
pairs = {}
for r in range(1, R):
    d = tract[r] != tract[0]
    for a, b in zip(tract[0][d], tract[r][d]):
        key = tuple(sorted((TRACT_NAMES[a], TRACT_NAMES[b])))
        pairs[key] = pairs.get(key, 0) + 1
top_pairs = sorted(pairs.items(), key=lambda kv: -kv[1])[:15]
n_events = sum(pairs.values())

# ---- secondary ----
outlier = ref["cluster"] >= 800
y_out = (outlier[1:] != outlier[0]).any(0)
secondary = {
    "ensemble_mean_m_hat_auroc": round(auroc(ref["m_hat"].mean(0)), 4),
    "ensemble_mean_m_mass_auroc": round(auroc(ref["m_mass"].mean(0)), 4),
    "outlier_status_changed_fraction": round(float(y_out.mean()), 4),
    "plausibility_auroc_for_outlier_change": {
        "best-class |plaus|": round(auroc(np.abs(ref["plaus_hat"][0]), y_out), 4),
        "mass |plaus|": round(auroc(np.abs(ref["plaus_mass"][0]), y_out), 4),
        "baseline p_max": round(auroc(ref["p_max"][0], y_out), 4)},
}

summary = {
    "streamlines": N, "runs": R, "target": "tract label differs from run 0 in any of runs 1..R-1",
    "tract_changed_fraction": round(float(y.mean()), 4),
    "auroc_run0": {k: round(v, 4) for k, v in au.items()},
    "auroc_run0_ci95": {k: [round(float(v), 4) for v in np.quantile(boot[k], [0.025, 0.975])] for k in PRED},
    "headline": {"best_field": best_field, "best_baseline": best_base,
                 "auroc_gain": round(au[best_field] - au[best_base], 4),
                 "gain_ci95": [round(float(v), 4) for v in np.quantile(gain, [0.025, 0.975])],
                 "bootstrap": B},
    "monotone": {"best_field_decile_change_rates": [round(float(v), 4) for v in dec],
                 "non_increasing": bool((np.diff(dec) <= 0).all())},
    "changed_by_m_hat_logits": by_logit,
    "tract_pair_changes": {"events": n_events,
                           "top": [{"pair": f"{a} / {b}", "events": c, "share": round(c / n_events, 4)}
                                   for (a, b), c in top_pairs]},
    "secondary": secondary,
}
(OUT / "m1.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))

# ---- figures: reference palette, light mode (dataviz skill references/palette.md) ----
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # slots 1-4, fixed order
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
xs = (np.arange(BINS) + 0.5) * 100 / BINS
for (k, (_, kind)), c in zip(PRED.items(), SERIES):
    ax.plot(xs, np.array(curves[k]) * 100, color=c, lw=2, ls="-" if kind == "field" else "--",
            marker="o", ms=4, label=f"{k}  (AUROC {au[k]:.3f})", zorder=3)
ax.set_xlabel("predictor percentile in run 0 (least confident first)")
ax.set_ylabel("tract label changes in runs 1-4 (%)")
ax.set_xlim(0, 100)
ax.set_ylim(bottom=0)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.tick_params(length=0)
ax.legend(frameon=False, loc="upper right", labelcolor=INK)
ax.set_title(f"Run-0 confidence vs tract instability, HCP 101006, {N:,} streamlines, {R} seeded runs",
             loc="left", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(OUT / "m1_instability.png")

fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=150)
labels = [f"{a} / {b}" for (a, b), _ in top_pairs][::-1]
counts = np.array([c for _, c in top_pairs][::-1])
ax.barh(labels, counts, color=SERIES[0], height=0.7, zorder=3)
for i, c in enumerate(counts):
    ax.text(c, i, f" {c}", va="center", fontsize=8, color=INK2)
ax.set_xlabel(f"label-change events, runs 1-4 vs run 0 (of {n_events:,})")
ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
ax.tick_params(length=0)
ax.set_title("Tract pairs that account for most changes", loc="left", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(OUT / "m1_pairs.png")
