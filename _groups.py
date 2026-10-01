"""The groupings M1b and M2 score, and the reference (full-field) margins of a streamline's own
group. One definition, so the regroup test and the rankfield fidelity check measure the same
thing.

A grouping is a 1600-entry class -> group table, compacted to 0..G-1, with the group names.
"""
import numpy as np
from tractcloud.tract_mapping import TRACT_NAMES, TRACT_CATEGORIES, _CLUSTER_TO_TRACT_LUT

TRACT = _CLUSTER_TO_TRACT_LUT.astype(np.int64)


def relabel(lut, names):
    """Compact a class -> group table to 0..G-1 (G = groups actually used)."""
    used, compact = np.unique(lut, return_inverse=True)
    return compact.astype(np.int64), [names[u] for u in used]


_cat_of_tract = {t: c for c, ts in TRACT_CATEGORIES.items() for t in ts}
_CATS = list(TRACT_CATEGORIES)
_merge = {**{t: "SLF" for t in ("SLF-I", "SLF-II", "SLF-III")}, **{f"CC{i}": "CC" for i in range(1, 8)}}
_MERGED = sorted({_merge.get(t, t) for t in TRACT_NAMES})
_no_outlier_tract = TRACT.copy()
_no_outlier_tract[800:] = TRACT[:800]

GROUPINGS = {
    "tract (TractCloud's)": relabel(TRACT, TRACT_NAMES),
    "category": relabel(np.array([_CATS.index(_cat_of_tract[TRACT_NAMES[t]]) for t in TRACT]), _CATS),
    "merged tracts (SLF, CC)": relabel(np.array([_MERGED.index(_merge.get(TRACT_NAMES[t], TRACT_NAMES[t]))
                                                 for t in TRACT]), _MERGED),
    "tract, outliers folded in": relabel(_no_outlier_tract, TRACT_NAMES),
    "cluster, outliers folded in": relabel(np.arange(1600) % 800, [str(c) for c in range(800)]),
}
# plausible clusters (0-799) against their outlier twins (800-1599)
PLAUSIBILITY = (np.arange(1600) >= 800).astype(np.int64)


def lse(x, axis):
    m = x.max(axis, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.log(np.exp(x - m).sum(axis, keepdims=True)) + m).squeeze(axis)


def margins(l, lut):
    """Mass and best-class margin of each streamline's own group (the group of its argmax),
    from the full (N, 1600) log-probabilities."""
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
