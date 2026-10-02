"""Does distortion correction change what the faithful pipeline finds on PAT16? The scan as acquired
against FSL topup's correction of it (topup_ref.py), both prepared by pat16_prep.py (b0 + b = 2800,
each with a mask from its own b0), tracked with UKF's own seeds and the ORG settings (Metal kernel)
and labeled by TractCloud (5-draw majority vote, MPS), on this Mac.

    DATA/.venv/bin/python bench/tractography/pat16_topup_compare.py

Per scan: seeds, streamlines >= 40 mm, median length, the share labeled Other, each tract's share of
the named streamlines. Between them: the tract-mix correlation and per-tract relative changes (against
the floors measured before: TractCloud's own redraws r 0.995-0.999, pat16_seeding_floor.json), and
each named tract's center (mean of its streamline points, RAS mm) moved by the correction.

Writes results/pat16_topup_compare.json.
"""
import json, sys, time, types
from pathlib import Path
import numpy as np, torch
import _ukf_torch as U
from _resample import resample
from _data import DATA as TD, MODEL, MASS_CENTER

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(TD / "TractCloud/src")); sys.modules.setdefault("vtk", types.ModuleType("vtk"))
from tractcloud import inference as inf
from tractcloud.tract_mapping import TRACT_NAMES, _CLUSTER_TO_TRACT_LUT as LUT
lut = LUT.astype(np.int64)
dev = torch.device("mps")
model, _ = inf.load_model(str(MODEL / "best_tract_f1_model.pth"), str(MODEL / "cli_args.txt"), dev, k_override=20, k_global_override=80)
center = np.load(MASS_CENTER)
off = U.SRAND0_OFFSET                                                    # the binary's seed offset (macOS srand(0))

def labels(fibers, draws=range(5)):
    lens = np.array([len(f) for f in fibers])
    P = np.concatenate(fibers).astype(np.float32).astype(np.float64)
    o = np.r_[0, np.cumsum(lens)]
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    length = seg[o[1:] - 1] - seg[o[:-1]]
    keep = length >= 40
    feat = resample(torch.from_numpy(P), torch.from_numpy(o)).numpy()[keep]
    out = []
    for s in draws:
        np.random.seed(s)
        ds = inf.RealDataDataset(inf.center_tractography(feat, center), k=20, k_global=80, k_ds_rate=0.1)
        Pf = torch.from_numpy(ds.feat).float().to(dev).transpose(2, 1).contiguous()
        L = torch.from_numpy(ds.local_feat).float().to(dev).transpose(2, 1).contiguous()
        G = torch.from_numpy(ds.global_feat).float().to(dev).transpose(2, 1).contiguous()
        cl = []
        with torch.no_grad():
            for a in range(0, len(ds), 1024):
                b = min(len(ds), a + 1024)
                cl.append(model(Pf[a:b], torch.cat((L[a:b], G.expand(b - a, -1, -1, -1)), 3)).view(-1, 1600).argmax(1).cpu())
        out.append(lut[torch.cat(cl).numpy()])
    vote = np.array([np.bincount(c, minlength=43).argmax() for c in np.stack(out).T])
    return vote, [f for f, k in zip(fibers, keep) if k], length[keep]

import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--a", default="PAT16"); ap.add_argument("--b", default="PAT16_topup")
ap.add_argument("--tag", default=""); args = ap.parse_args()
runs = {}
for name in (args.a, args.b):
    O = TD / "ds001226/derived" / name
    D = U.load(str(O / "dwi.nhdr"), str(O / "mask.nrrd"))
    pts, *_ = U.seeds(D, off)
    t0 = time.time()
    fib, st = U.track(D, off, backend="metal"); torch.mps.synchronize()
    t = time.time() - t0
    lab, kept, length = labels(fib)
    runs[name] = {"seeds": int(len(pts)), "track_s": round(t, 1), "lab": lab, "kept": kept, "length": length}

def shares(lab):
    c = np.bincount(lab, minlength=43)[:42].astype(float)
    return c / c.sum()
a, b = runs[args.a], runs[args.b]
sa, sb = shares(a["lab"]), shares(b["lab"])
named = (sa > 0.002) | (sb > 0.002)
rel = np.abs(sb[named] - sa[named]) / np.maximum(sa[named], 1e-9)

def centers(r):
    out = {}
    for t in range(42):
        idx = np.flatnonzero(r["lab"] == t)
        if len(idx) >= 20:
            out[t] = np.concatenate([r["kept"][i] for i in idx]).mean(0)
    return out
ca, cb = centers(a), centers(b)
moved = {TRACT_NAMES[t]: round(float(np.linalg.norm(ca[t] - cb[t])), 1) for t in ca if t in cb}
mv = np.array(list(moved.values()))
res = {"data": f"ds001226 PAT16, b0 + b=2800: {args.a} against {args.b}",
       args.a: {"seeds": a["seeds"], "streamlines_ge_40mm": len(a["kept"]), "median_length_mm": round(float(np.median(a["length"])), 1),
                       "other_fraction": round(float((a["lab"] == 42).mean()), 4), "track_s": a["track_s"]},
       args.b: {"seeds": b["seeds"], "streamlines_ge_40mm": len(b["kept"]), "median_length_mm": round(float(np.median(b["length"])), 1),
                           "other_fraction": round(float((b["lab"] == 42).mean()), 4), "track_s": b["track_s"]},
       "tract_mix_r": round(float(np.corrcoef(sa, sb)[0, 1]), 4),
       "per_tract_abs_rel_change_median_90th": [round(float(np.median(rel)), 3), round(float(np.quantile(rel, 0.9)), 3)],
       "largest_share_changes": sorted(((TRACT_NAMES[t], round(float(sb[t] / max(sa[t], 1e-9)), 2)) for t in np.flatnonzero(named)),
                                       key=lambda x: -abs(np.log(max(x[1], 1e-3))))[:8],
       "tract_center_moved_mm_median_90th_max": [round(float(np.median(mv)), 1), round(float(np.quantile(mv, 0.9)), 1), round(float(mv.max()), 1)],
       "tract_center_moved_mm": dict(sorted(moved.items(), key=lambda x: -x[1])),
       "for_scale": "TractCloud's own redraws on one tractogram: tract mix r 0.995-0.999 (pat16_seeding_floor.json)"}
print(json.dumps(res, indent=1))
(HERE / "results" / f"pat16_topup_compare{args.tag}.json").write_text(json.dumps(res, indent=1))
