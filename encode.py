"""M2: encode run 0 with rankfield and check what the 19-odd bytes give back against the full field.

    DATA/.venv/bin/python bench/tractography/encode.py

rankfield v0.3.10, keep="clip" (the winner, then the non-winners within the clip, closest
first: no grid neighbors), clip 8 logits, the default log byte curve, a T = 1 tail, at depths
4, 6 and 8. The (T, 1600) fp16 log-softmax goes in as a (1600, T, 1, 1) view - streamlines on
Z, which is the axis the encoder slabs along.

Checked against the full-field reference (_groups.margins on the same fp16 values):
  - labels: ranks[0] - 1 must equal the reference argmax exactly
  - best-class margin of the streamline's own group, from rankfield.decode_groups, against the
    reference capped at the clip (decode_groups floors levels at -clip, so a lead reads at most
    `clip`). Overestimates beyond the byte step come from the depth cut: the nearest non-member
    was within the clip but did not get a plane.
  - mass margin of its own group from rankfield.probabilities with the tail counted against
    the group: a lower bound on the true mass margin, up to byte quantization. Where the stored
    outside mass is below half a tail quantum the margin is floored there.
  - the M1/M1b AUROCs recomputed from the decoded margins, against those from the reference
  - plausibility: the winner's side (clusters 0-799 or outlier twins) as a two-group margin
Sizes: raw planes, and each depth's planes compressed (blosc zstd, byte shuffle), against
dense fp16, a naive top-6 (fp16 value + uint16 index), argmax, argmax + p_max.

Writes DATA/hcp/rf_run0_d{depth}.npz (planes + meta, for the export) and results/m2.json.
"""
import json, time
from pathlib import Path
import numpy as np, torch
import rankfield as rf
from numcodecs import Blosc
from sklearn.metrics import roc_auc_score
from _data import HCP
from _groups import GROUPINGS, PLAUSIBILITY, lse, margins

OUT = Path(__file__).resolve().parent / "results"
DEPTHS = (4, 6, 8)
CLIP = 8.0
TAIL_MAX = rf.TAIL_MAX
codec = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)
csize = lambda a: len(codec.encode(np.ascontiguousarray(a)))
auroc = lambda x, t: float(roc_auc_score(t, -x))
q = lambda x, ps=(0.001, 0.01, 0.5, 0.99, 0.999): [round(float(v), 4) for v in np.quantile(x, ps)]

ref = dict(np.load(HCP / "reference.npz"))
cluster = ref["cluster"].astype(np.int64)
R, N = cluster.shape
l16 = np.load(HCP / "logp_run0.npy")
l = l16.astype(np.float32)
l -= lse(l, 1)[:, None]
assert (l.argmax(1) == cluster[0]).all()

ALL = {**GROUPINGS, "plausibility": (PLAUSIBILITY, ["plausible", "outlier"])}
reference = {g: margins(l, lut) for g, (lut, _) in ALL.items()}
targets = {}
for g, (lut, _) in GROUPINGS.items():
    lab = lut[cluster]
    targets[g] = (lab[1:] != lab[0]).any(0)
outlier = cluster >= 800
targets["plausibility"] = (outlier[1:] != outlier[0]).any(0)


def signed_plaus(m):
    """own-group margin -> margin of the plausible side (|.| is what M1 scored)"""
    return np.where(cluster[0] < 800, m, -m)


def decode_own(code, lut):
    """Best-class (decode_groups) and mass (probabilities + tail) margin of the own group."""
    G = int(lut.max()) + 1
    ranks = np.asarray(code.ranks).reshape(code.depth, N).astype(np.int64)
    own = lut[ranks[0] - 1]
    groups = [np.flatnonzero(lut == g) for g in range(G)]
    dg = rf.decode_groups(code, groups).reshape(G, N)
    best = dg[torch.from_numpy(own), torch.arange(N)].numpy()
    del dg
    ids, p = rf.probabilities(code)
    ids, p = ids.reshape(code.depth, N), p.reshape(code.depth, N).astype(np.float64)
    kept = ids >= 0
    in_s = kept & (lut[np.clip(ids, 0, None)] == own)
    p_in = (p * in_s).sum(0)
    tail = np.asarray(code.tail).reshape(N).astype(np.float64) / TAIL_MAX
    p_out = (p * (kept & ~in_s)).sum(0) + tail
    floored = p_out < 0.5 / TAIL_MAX
    mass = np.log(p_in) - np.log(np.maximum(p_out, 0.5 / TAIL_MAX))
    return {"best": best, "mass": mass, "mass_floored": floored}


lg = torch.from_numpy(l16).T[:, :, None, None]          # (1600, T, 1, 1), a view
sizes_fixed = {"dense fp16": N * 1600 * 2, "dense fp16, blosc zstd": csize(l16),
               "naive top-6 (fp16 + uint16 index)": N * 6 * 4,
               "argmax only": N * 2, "argmax + p_max byte": N * 3}
out = {"streamlines": N, "rankfield": rf.__version__, "keep": "clip", "clip": CLIP,
       "sizes_bytes_per_streamline_fixed": {k: round(v / N, 2) for k, v in sizes_fixed.items()},
       "depths": {}}
for depth in DEPTHS:
    t0 = time.time()
    code = rf.encode(lg, keep="clip", depth=depth, clip=CLIP, tail_temperatures=(1.0,))
    t_enc = time.time() - t0
    np.savez(HCP / f"rf_run0_d{depth}.npz", ranks=code.ranks, support=code.support, tail=code.tail,
             meta=json.dumps(code.meta))
    ranks = np.asarray(code.ranks).reshape(depth, N)
    tail = np.asarray(code.tail).reshape(N) / TAIL_MAX
    planes_kept = (ranks != 0).sum(0)
    d = {"encode_s": round(t_enc, 1),
         "labels_differ": int((ranks[0].astype(np.int64) - 1 != cluster[0]).sum()),
         "kept_planes_mean": round(float(planes_kept.mean()), 3),
         "kept_planes_hist": {int(k): int(v) for k, v in zip(*np.unique(planes_kept, return_counts=True))},
         "full_depth_fraction": round(float((planes_kept == depth).mean()), 4),
         "tail_quantiles_0.1_1_50_99_99.9pct": q(tail),
         "tail_over_1pct_fraction": round(float((tail > 0.01).mean()), 4),
         "bytes_raw_per_streamline": round(code.nbytes / N, 2),
         "bytes_blosc_per_streamline": round(sum(csize(a) for a in (code.ranks, code.support, code.tail)) / N, 2),
         "groupings": {}}
    for g, (lut, _) in ALL.items():
        dec = decode_own(code, lut)
        rb, rm = reference[g]["best"], reference[g]["mass"]
        eb = dec["best"] - np.minimum(rb, CLIP)
        em = dec["mass"] - rm
        if g == "plausibility":
            pred = {"best": np.abs(signed_plaus(dec["best"])), "mass": np.abs(signed_plaus(dec["mass"]))}
            pref = {"best": np.abs(signed_plaus(rb)), "mass": np.abs(signed_plaus(rm))}
        else:
            pred, pref = {"best": dec["best"], "mass": dec["mass"]}, {"best": rb, "mass": rm}
        y = targets[g]
        d["groupings"][g] = {
            "best_class": {"err_vs_ref_capped_quantiles": q(eb),
                           "within_0.05_logit": round(float((np.abs(eb) <= 0.05).mean()), 4),
                           "over_by_more_than_0.05": round(float((eb > 0.05).mean()), 4),
                           "ref_at_or_over_clip": round(float((rb >= CLIP).mean()), 4)},
            "mass": {"err_quantiles": q(em),
                     "over_by_more_than_0.05": round(float((em > 0.05).mean()), 4),
                     "floored_fraction": round(float(dec["mass_floored"].mean()), 4),
                     "sign_flips": int(((dec["mass"] < 0) != (rm < 0)).sum())},
            "auroc": {k: {"decoded": round(auroc(pred[k], y), 4), "reference": round(auroc(pref[k], y), 4)}
                      for k in ("best", "mass")},
        }
    out["depths"][depth] = d
    print(json.dumps({"depth": depth, **{k: v for k, v in d.items() if k != "groupings"}}), flush=True)
    for g, v in d["groupings"].items():
        print(f"  {g}: auroc {v['auroc']}  best within .05: {v['best_class']['within_0.05_logit']}  "
              f"mass err q: {v['mass']['err_quantiles']}", flush=True)
    del code

(OUT / "m2.json").write_text(json.dumps(out, indent=1))
