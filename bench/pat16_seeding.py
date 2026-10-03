"""Is albula-diffusion's sparse seeding reasonable? Its whole-brain naming seeds 25,000 points in voxels
with FA > 0.2 (planning.ts wholeBrainSeeds), where UKF seeds every mask voxel it accepts. On its own
reference case, OpenNeuro ds001226 PAT16 (pat16_prep.py), on this Mac.

    uv run bench/pat16_seeding.py [--draws 5]

Both arms: the same scan (b0 + the b = 2800 shell), mask, tracker (ukf with the Metal kernel,
the ORG settings: two tensors, stop FA 0.08, stop signal 0.06, a point every 1.8 mm) and TractCloud
(40 mm cut, upstream context, MPS). Only the seeds differ:
  faithful  UKF's own seeds: every mask voxel plus the srand(0) offset, kept by its rules (0.1)
  sparse    albula's rule: 25,000 mask voxels with single-tensor FA > 0.2 at the voxel center, drawn
            by a seeded generator and placed uniformly inside their voxel; `draws` different draws
TractCloud labels: faithful, a 5-draw majority vote of the context (the best estimate); sparse, one
context draw (albula's default, draws = 1).

Measured: streamlines, Other share, each tract's share of the named streamlines (faithful against
each sparse draw, and the spread across draws), and albula's planning rule - a tract is "near" a
structure when at least 5 of its streamlines come within 8 mm (MIN_NEAR_STREAMLINES, withinMm) -
around 20 test spheres (radius 10 mm, centers at random voxels with FA > 0.3; PAT16's tumor outline
is not public). Per sphere: the near-tract sets of each sparse draw against faithful (recall,
precision) and against each other (Jaccard). Faithful also with the threshold scaled by its streamline
count, separating sampling density from sampling noise.

Writes results/pat16_seeding.json.
"""
import argparse, json, sys, time, types
from pathlib import Path
import numpy as np, torch
from tractline import ukf as U
from tractline.resample import resample
from tractline.data import DATA as TDATA, MODEL, MASS_CENTER

ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=5)
ap.add_argument("--floor", action="store_true", help="only the faithful tractogram, single context draws: TractCloud's own floor")
args = ap.parse_args()
HERE = Path(__file__).resolve().parent
O = TDATA / "ds001226/derived/PAT16"
sys.path.insert(0, str(TDATA / "TractCloud/src")); sys.modules.setdefault("vtk", types.ModuleType("vtk"))
from tractcloud import inference as inf
from tractcloud.tract_mapping import TRACT_NAMES, _CLUSTER_TO_TRACT_LUT as LUT

N_SEEDS, SEED_FA, NEAR_N, NEAR_MM, R_MM, N_ROI = 25000, 0.2, 5, 8.0, 10.0, 20
lut = LUT.astype(np.int64)
dev = torch.device("mps")
model, _ = inf.load_model(str(MODEL / "best_tract_f1_model.pth"), str(MODEL / "cli_args.txt"), dev, k_override=20, k_global_override=80)
center = np.load(MASS_CENTER)

D = U.load(str(O / "dwi.nhdr"), str(O / "mask.nrrd"))
i2r = D["i2r"]
off = U.SRAND0_OFFSET                                                    # the binary's seed offset (macOS srand(0))
T = {}

def track(points):
    t0 = time.time()
    f, st = U.track(D, off, seed_points=points, backend="metal")
    torch.mps.synchronize()
    return f, time.time() - t0, st["fiber_steps"]

def labels(fibers, draws):
    lens = np.array([len(f) for f in fibers])
    P = np.concatenate(fibers).astype(np.float32).astype(np.float64)
    o = np.r_[0, np.cumsum(lens)]
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    keep = (seg[o[1:] - 1] - seg[o[:-1]]) >= 40
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
    T_ = np.stack(out)
    vote = np.array([np.bincount(c, minlength=43).argmax() for c in T_.T])
    return vote, [f for f, k in zip(fibers, keep) if k]

# ---- the faithful arm
pts_f, *_ = U.seeds(D, off)
fib_f, t, steps = track(pts_f)
T["faithful_track"] = (round(t, 1), steps)
lab_f, kept_f = labels(fib_f, range(5))

if args.floor:
    lens = np.array([len(f) for f in fib_f]); P = np.concatenate(fib_f).astype(np.float32).astype(np.float64)
    o = np.r_[0, np.cumsum(lens)]; seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    keepf = (seg[o[1:] - 1] - seg[o[:-1]]) >= 40
    singles = [labels(fib_f, [d])[0] for d in range(args.draws)]
    mask_kji = torch.nonzero(D["mask"] > 0).double()
    rng = np.random.default_rng(7)
    _, _, _, fa_all, _ = U.seed_states(D, mask_kji)
    hi = mask_kji[fa_all > 0.3].numpy()
    roi = hi[rng.choice(len(hi), N_ROI, replace=False)][:, ::-1] @ i2r[:3, :3].T + i2r[:3, 3]
    lens = np.array([len(f) for f in kept_f]); Pk = np.concatenate(kept_f)
    ok = np.r_[0, np.cumsum(lens)][:-1]
    close = np.stack([np.minimum.reduceat(np.linalg.norm(Pk - c, axis=1), ok) for c in roi], 1) <= R_MM + NEAR_MM
    sets = lambda lab: [set(np.flatnonzero(np.bincount(lab[close[:, r]], minlength=43)[:42] >= NEAR_N).tolist()) for r in range(N_ROI)]
    V, Sd = sets(lab_f), [sets(l) for l in singles]
    jac = lambda A, B: len(A & B) / len(A | B) if (A | B) else 1.0
    out = {"faithful_single_context_draws": args.draws,
           "draw_to_draw_jaccard": round(float(np.mean([jac(Sd[a][r], Sd[b][r]) for r in range(N_ROI) for a in range(args.draws) for b in range(a + 1, args.draws)])), 3),
           "recall_vs_vote": round(float(np.mean([len(Sd[d][r] & V[r]) / len(V[r]) for d in range(args.draws) for r in range(N_ROI) if V[r]])), 3),
           "rois_where_draws_disagree": int(sum(len({frozenset(Sd[d][r]) for d in range(args.draws)}) > 1 for r in range(N_ROI))),
           "tract_share_r_single_vs_vote": [round(float(np.corrcoef(np.bincount(l, minlength=43)[:42], np.bincount(lab_f, minlength=43)[:42])[0, 1]), 4) for l in singles]}
    print(json.dumps(out, indent=1))
    (HERE / "results" / "pat16_seeding_floor.json").write_text(json.dumps(out, indent=1))
    raise SystemExit(0)

# ---- the sparse arm: albula's candidate voxels, then `draws` draws
mask_kji = torch.nonzero(D["mask"] > 0).double()
_, _, _, fa_c, _ = U.seed_states(D, mask_kji)
cand = mask_kji[fa_c > SEED_FA].numpy()
sparse = []
for d in range(args.draws):
    rng = np.random.default_rng(20260930 + d)                       # albula seeds its generator with 20260930
    pick = cand[rng.choice(len(cand), min(N_SEEDS, len(cand)), replace=False)]
    pts = torch.from_numpy(pick + rng.uniform(-0.5, 0.5, pick.shape))
    fib, t, steps = track(pts)
    T[f"sparse_track_{d}"] = (round(t, 1), steps)
    lab, kept = labels(fib, [d])
    sparse.append((lab, kept))

# ---- tract shares of the named streamlines
def shares(lab):
    c = np.bincount(lab, minlength=43)[:42].astype(float)
    return c / max(c.sum(), 1)
sf = shares(lab_f)
ss = np.stack([shares(l) for l, _ in sparse])
named = sf > 0.002                                                    # tracts with at least 0.2 % of faithful's named streamlines
res = {"data": "OpenNeuro ds001226 PAT16 (CC0), b0 + b=2800 shell, 2.5 mm", "tracker": "Metal kernel, ORG settings",
       "faithful": {"seeds": int(len(pts_f)), "streamlines_ge_40mm": int(len(kept_f)), "other_fraction": round(float((lab_f == 42).mean()), 4)},
       "sparse": {"candidate_voxels_fa_gt_0.2": int(len(cand)), "seeds_per_draw": int(min(N_SEEDS, len(cand))), "draws": args.draws,
                  "streamlines_ge_40mm": [int(len(k)) for _, k in sparse],
                  "other_fraction": [round(float((l == 42).mean()), 4) for l, _ in sparse]},
       "timing_s_and_steps": T,
       "tract_shares": {"r_faithful_vs_each_draw": [round(float(np.corrcoef(sf, s)[0, 1]), 4) for s in ss],
                        "median_abs_rel_error_vs_faithful": [round(float(np.median(np.abs(s[named] - sf[named]) / sf[named])), 3) for s in ss],
                        "median_cv_across_draws": round(float(np.median(ss[:, named].std(0) / np.maximum(ss[:, named].mean(0), 1e-9))), 3),
                        "tracts_compared": int(named.sum()),
                        "largest_relative_shortfalls_draw0": sorted(((TRACT_NAMES[t], round(float(ss[0, t] / sf[t]), 2)) for t in np.flatnonzero(named)), key=lambda x: x[1])[:6]}}

# ---- albula's near-structure rule around test spheres
rng = np.random.default_rng(7)
_, _, _, fa_all, _ = U.seed_states(D, mask_kji)
hi = mask_kji[fa_all > 0.3].numpy()
roi_kji = hi[rng.choice(len(hi), N_ROI, replace=False)]
roi = roi_kji[:, ::-1] @ i2r[:3, :3].T + i2r[:3, 3]                  # RAS mm

def near_sets(fibers, lab, threshold):
    lens = np.array([len(f) for f in fibers]); P = np.concatenate(fibers)
    o = np.r_[0, np.cumsum(lens)][:-1]
    dmin = np.stack([np.minimum.reduceat(np.linalg.norm(P - c, axis=1), o) for c in roi], 1)   # (fibers, rois)
    close = dmin <= R_MM + NEAR_MM
    sets = []
    for r in range(N_ROI):
        c = np.bincount(lab[close[:, r]], minlength=43)[:42]
        sets.append(set(np.flatnonzero(c >= threshold).tolist()))
    return sets

F5 = near_sets(kept_f, lab_f, NEAR_N)
scale = len(kept_f) / np.mean([len(k) for _, k in sparse])
Fs = near_sets(kept_f, lab_f, NEAR_N * scale)
S = [near_sets(k, l, NEAR_N) for l, k in sparse]
pr = lambda A, B: (len(A & B) / len(B) if B else 1.0, len(A & B) / len(A) if A else 1.0)   # recall vs B, precision
jac = lambda A, B: len(A & B) / len(A | B) if (A | B) else 1.0
rec = [[pr(S[d][r], F5[r])[0] for r in range(N_ROI)] for d in range(args.draws)]
recS = [[pr(S[d][r], Fs[r])[0] for r in range(N_ROI)] for d in range(args.draws)]
prc = [[pr(S[d][r], Fs[r])[1] for r in range(N_ROI)] for d in range(args.draws)]
jd = [jac(S[a][r], S[b][r]) for r in range(N_ROI) for a in range(args.draws) for b in range(a + 1, args.draws)]
res["near_rule"] = {"rois": N_ROI, "roi_radius_mm": R_MM, "within_mm": NEAR_MM, "min_streamlines": NEAR_N,
                    "near_tracts_per_roi_mean": {"faithful_rule_5": round(float(np.mean([len(s) for s in F5])), 2),
                                                 f"faithful_rule_scaled_{NEAR_N * scale:.1f}": round(float(np.mean([len(s) for s in Fs])), 2),
                                                 "sparse": round(float(np.mean([len(s) for d in S for s in d])), 2)},
                    "sparse_recall_vs_faithful_rule_5": round(float(np.mean(rec)), 3),
                    "sparse_recall_vs_faithful_scaled": round(float(np.mean(recS)), 3),
                    "sparse_precision_vs_faithful_scaled": round(float(np.mean(prc)), 3),
                    "sparse_draw_to_draw_jaccard": round(float(np.mean(jd)), 3),
                    "rois_where_draws_disagree": int(sum(len({frozenset(S[d][r]) for d in range(args.draws)}) > 1 for r in range(N_ROI)))}
print(json.dumps(res, indent=1))
(HERE / "results" / "pat16_seeding.json").write_text(json.dumps(res, indent=1))
