"""M0 step 2: the numpy reference - per-streamline labels, margins and instability across runs.

    DATA/.venv/bin/python bench/tractography/reference.py

Computed from the stored fp16 log-softmax (logp_run{r}.npy), the same values rankfield will
encode, so the reference and the encoding see identical inputs. Two group margins (plan.md,
"Two margins, defined once"), for the winning tract T = the tract of the argmax cluster,
TractCloud's own label:

    best-class  m_hat = max_{c in T} l_c - max_{c not in T} l_c     (> 0 always: T holds the argmax)
    mass        m     = LSE_{c in T} l_c - LSE_{c not in T} l_c     (< 0 where another tract has more mass)

and the plausibility margin in both forms (clusters 0-799 against their outlier twins
800-1599). Baselines that need no field: p_max (the winning cluster's probability) and the
cluster margin (top-1 minus top-2 log-probability).

"Other" (tract 42) is every outlier class AND 289 plausible clusters the atlas leaves
unannotated (tract_mapping.py), so it is a group of 1089 classes, not just the outliers.

Writes DATA/hcp/reference.npz (every array (R, T) in run order, plus the run-0-vs-rest
instability) and DATA/hcp/reference.json (summary numbers).
"""
import json
import numpy as np
from _data import HCP, SEEDS
from tractcloud.tract_mapping import TRACT_NAMES, _CLUSTER_TO_TRACT_LUT as LUT

R = len(SEEDS)
G = len(TRACT_NAMES)
ORDER = np.argsort(LUT, kind="stable")                  # classes grouped by tract
STARTS = np.flatnonzero(np.r_[True, np.diff(LUT[ORDER]) != 0])
assert len(STARTS) == G and (LUT[ORDER][STARTS] == np.arange(G)).all()


def lse(x, axis):
    m = x.max(axis, keepdims=True)
    return (np.log(np.exp(x - m).sum(axis, keepdims=True)) + m).squeeze(axis)


def group_reduce(l):
    """(N, 1600) -> per-tract max and per-tract LSE, each (N, 43)."""
    lo = l[:, ORDER]
    gmax = np.maximum.reduceat(lo, STARTS, axis=1)
    top = l.max(1, keepdims=True)
    with np.errstate(divide="ignore"):                  # a tract whose mass underflows: LSE = -inf
        glse = np.log(np.add.reduceat(np.exp(lo - top), STARTS, axis=1)) + top
    return gmax, glse


def lead(g, win):
    """Margin of column `win` over the rest of g (N, G), by max and the best other column."""
    rows = np.arange(len(g))
    mine = g[rows, win]
    other = g.copy()
    other[rows, win] = -np.inf
    return mine, other, other.argmax(1)


fields = ("cluster", "tract", "m_hat", "runner_hat", "m_mass", "runner_mass", "mass_winner",
          "plaus_hat", "plaus_mass", "p_max", "cluster_margin")
out = {k: [] for k in fields}
for r in range(R):
    l = np.load(HCP / f"logp_run{r}.npy").astype(np.float32)
    l = l - lse(l, 1)[:, None]                          # renormalize away fp16 rounding of the constant
    cluster = l.argmax(1)
    tract = LUT[cluster]
    gmax, glse = group_reduce(l)
    mine, other, runner_hat = lead(gmax, tract)
    m_hat = mine - other.max(1)
    mine, other, runner_mass = lead(glse, tract)
    m_mass = mine - lse(other, 1)
    top2 = np.partition(l, -2, axis=1)[:, -2:]
    for k, v in (("cluster", cluster), ("tract", tract), ("m_hat", m_hat), ("runner_hat", runner_hat),
                 ("m_mass", m_mass), ("runner_mass", runner_mass), ("mass_winner", glse.argmax(1)),
                 ("plaus_hat", l[:, :800].max(1) - l[:, 800:].max(1)),
                 ("plaus_mass", lse(l[:, :800], 1) - lse(l[:, 800:], 1)),
                 ("p_max", np.exp(l.max(1))), ("cluster_margin", top2[:, 1] - top2[:, 0])):
        out[k].append(v)
    print(f"run {r}: done", flush=True)

arr = {k: np.stack(v) for k, v in out.items()}
for k in ("cluster", "tract", "runner_hat", "runner_mass", "mass_winner"):
    arr[k] = arr[k].astype(np.int16)
for k in ("m_hat", "m_mass", "plaus_hat", "plaus_mass", "p_max", "cluster_margin"):
    arr[k] = arr[k].astype(np.float32)
tract, cluster = arr["tract"], arr["cluster"]
arr["tract_changed"] = (tract[1:] != tract[0]).any(0)
arr["cluster_changed"] = (cluster[1:] != cluster[0]).any(0)
arr["tract_distinct"] = np.array([len(set(c)) for c in tract.T], np.int8)
np.savez(HCP / "reference.npz", **arr)

# the labels the capture wrote from the fp32 output, against the fp16 argmax used here
fp32 = np.stack([np.load(HCP / f"label_run{r}.npy") for r in range(R)])
q = lambda x: [round(float(v), 3) for v in np.quantile(x, [0.01, 0.1, 0.5, 0.9, 0.99])]
summary = {
    "runs": R, "seeds": list(SEEDS), "targets": int(tract.shape[1]),
    "fp16_vs_fp32_cluster_differs_per_run": [int(v) for v in (fp32 != cluster).sum(1)],
    "tract_changed_fraction": round(float(arr["tract_changed"].mean()), 4),
    "cluster_changed_fraction": round(float(arr["cluster_changed"].mean()), 4),
    "tract_distinct_counts": {int(k): int(v) for k, v in zip(*np.unique(arr["tract_distinct"], return_counts=True))},
    "run0_other_fraction": round(float((tract[0] == G - 1).mean()), 4),
    "run0_outlier_class_fraction": round(float((cluster[0] >= 800).mean()), 4),
    "run0_mass_winner_not_label_fraction": round(float((arr["mass_winner"][0] != tract[0]).mean()), 4),
    "run0_quantiles_1_10_50_90_99": {k: q(arr[k][0]) for k in
                                     ("m_hat", "m_mass", "plaus_hat", "plaus_mass", "p_max", "cluster_margin")},
}
(HCP / "reference.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
