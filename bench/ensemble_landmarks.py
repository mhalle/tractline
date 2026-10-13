"""Landmark sites on the healthy subject cIIIs04: what the ensemble says there, and what a real rescan did (NOTES
2026-10-09, "Landmark sites on cIIIs04").

    uv run --group rankfield bench/ensemble_landmarks.py [--pair cIIIs04-2.1] [--fans 16]

From ensemble_retest.py's compact records of the pair (no pipeline runs): A's 17-run ensemble (A, smooth/r001..r016)
and the three rescans' single runs on A's grid (<tag>_onA). Most sites are found in the ensemble itself,
on its presence shares P(v, k) (the share of runs with at least MIN_COUNT streamlines of tract k in voxel v,
smoothed by a Gaussian of sigma 1 voxel; the genu and splenium on mean counts):
  - centrum semiovale, left and right: corticospinal, callosal (CC3-CC5) and superior longitudinal (SLF-II, SLF-III,
    AF) fibers all present (the largest of the three's minimum);
  - temporal stem, left and right: the most posterior voxel where the uncinate and inferior occipito-frontal
    fasciculi are both present (smoothed P at least 0.6): their shared course runs forward from where they leave the
    temporal lobe (the best overlap alone falls in the frontal lobe);
  - the posterior limb of the internal capsule, left and right, from FastSurfer's anatomy (ensemble_endpoints.py; the
    T1 registered to the DWI): the cerebral white matter within PLIC_MM of both the thalamus and the lentiform nucleus
    (putamen, pallidum), and of that set the voxel nearest its centroid, carried onto the DWI grid. Where the
    corticospinal tract should pass, found without the tracts;
  - the genu (CC1) and splenium (CC7) of the corpus callosum: the tract's densest voxel (mean streamline count)
    within 3 mm of the midline.
At each site:
  - the voxel rank field of the ensemble (the 43 classes by their mean streamline count over the 17 runs, those of at
    least MIN_COUNT; the rank field's NONE below that) at the site's voxel. For the viewer's runner-up shells, every
    voxel's first and second named tract (Other, which leads most voxels, left out);
  - its seeds: the streamlines of A whose full path passes within NEAR_MM of the site, labeled in all 17 runs; per
    seed its 17 tracts. Agreement (every run the same tract; the mode's share); the replicate baseline, leaving each
    run out in turn (does it give the other 16 runs' mode, against that mode's share among them); and against each
    rescan, among the seeds the rescan also labeled, the share whose rescan tract is the ensemble's mode (stated: the
    mode's mean share) or any tract the 17 runs gave;
  - runner-up: per streamline the labeler's second-choice tract (tract probabilities summed over its clusters). When
    the rescan's tract differs from A's, how often it is A's runner-up; when A's runs split, how often the second
    most frequent tract is A's runner-up (also over all seeds of the brain).
Writes results/ensemble_landmarks_smooth.json and DATA/ensemble/<subject>/retest/<pair>/landmarks.npz (the voxel
data and the fans' full streamlines, for ensemble_landmarks_3d.py).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import TRACT_NAMES, LUT

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS  # noqa: E402
from ensemble_pat16_viewer import threshold  # noqa: E402

RUNS = 17
MIN_COUNT = 2                     # presence: a tract with at least 2 streamlines in a voxel (ensemble_coverage.py --min-count 2)
NAMED = E.N_TRACTS - 1            # 0-41; 42 is Other
OTHER = NAMED
NEAR_MM = 4.0
ONEHOT = np.eye(E.N_TRACTS, dtype=np.float32)[LUT]                  # (1600, 43)
T = TRACT_NAMES.index
FOCUS = ("CST", "AF", "UF")       # the clinical focus (the collaborator's suggestion: the most reliable for planning)
SHOWN = FOCUS + ("SLF-II", "IOFF", "ILF", "CC4", "CC1", "CC7")   # the tracts drawn; the focus first
PLIC_MM = 5.0
ENVELOPE_80 = 0.925               # the stated level that held 0.8 of cIIIs04's rescan tracts, fitted on the other pairs
                                  # (ensemble_coverage_smooth_presence2_with_cIIIs04.json, leave one pair out)


def runner_up(logp):
    """(n, 1600) cluster log-probabilities -> each streamline's (first, second) tract and the second's probability."""
    p = np.exp(logp.astype(np.float32)) @ ONEHOT
    o = np.argsort(-p, 1)
    return o[:, 0], o[:, 1], p[np.arange(len(p)), o[:, 1]]


def leave_one_out(t):
    """(m, R) tracts -> per seed, over each run left out: whether it gives the other runs' mode, and that mode's share
    among them (the baseline for the in-sample statement: a replicate the mode was not chosen on)."""
    m, R = t.shape
    cnt = np.zeros((m, E.N_TRACTS), np.int32)
    np.add.at(cnt, (np.repeat(np.arange(m), R), t.ravel()), 1)
    hit, stated = np.zeros(m), np.zeros(m)
    for r in range(R):
        c = cnt.copy()
        c[np.arange(m), t[:, r]] -= 1
        mode = c.argmax(1)
        hit += mode == t[:, r]
        stated += c.max(1) / (R - 1)
    return hit / R, stated / R


def sites(Ps, mean, aff, shape, mid_x):
    """The landmark sites: name, what defines it, its voxel (i, j, k), its world position (mm), the tracts it is about."""
    w = (np.indices(shape).reshape(3, -1).T @ aff[:3, :3].T + aff[:3, 3]).T.reshape(3, *shape)
    xw, yw = w[0], w[1]
    side = {"left": xw < mid_x - 6, "right": xw > mid_x + 6, "midline": np.abs(xw - mid_x) <= 3}
    dens = lambda k: gaussian_filter(mean[k].reshape(shape), 1.0)  # noqa: E731
    cc_body = Ps[[T("CC3"), T("CC4"), T("CC5")]].max(0)
    slf = Ps[[T("SLF-II"), T("SLF-III"), T("AF")]].max(0)
    defs = [("centrum semiovale", "corticospinal, callosal (CC3-CC5) and superior longitudinal fibers all present",
             np.minimum(np.minimum(Ps[T("CST")], cc_body), slf), ("left", "right"), ("CST", "CC4", "SLF-II", "AF")),
            ("temporal stem", "the most posterior point where the uncinate and inferior occipito-frontal fasciculi are both present",
             np.where(np.minimum(Ps[T("UF")], Ps[T("IOFF")]) >= 0.6, -yw, -np.inf), ("left", "right"), ("UF", "IOFF", "ILF")),
            ("genu of the corpus callosum", "CC1 densest at the midline", dens(T("CC1")), ("midline",), ("CC1",)),
            ("splenium of the corpus callosum", "CC7 densest at the midline", dens(T("CC7")), ("midline",), ("CC7",))]
    out = []
    for name, what, score, sides, tracts in defs:
        for s in sides:
            v = np.where(side[s], score, -np.inf)
            ijk = np.unravel_index(int(np.argmax(v)), shape)
            assert np.isfinite(v[ijk]), f"no voxel qualifies for {name}, {s}"
            label = name if s == "midline" else f"{name}, {s}"
            out.append({"name": label, "defined by": what, "score": round(float(v[ijk]), 3), "voxel": [int(c) for c in ijk],
                        "mm": [round(float(c), 1) for c in aff[:3, :3] @ np.array(ijk) + aff[:3, 3]], "tracts": list(tracts)})
    return out


def plic_sites(aff):
    """The posterior limb of the internal capsule, left and right, from FastSurfer's labels on the T1."""
    from scipy.ndimage import distance_transform_edt
    from ensemble_endpoints import Anatomy, FS
    anat = Anatomy("cIIIs04")
    T1 = np.load(FS / "cIIIs04_t1_to_dwi.npz")["t1_to_dwi"]
    vox = np.sqrt((anat.affine[:3, :3] ** 2).sum(0))
    out = []
    for side, (wm, thal, lent) in {"left": (2, 10, (12, 13)), "right": (41, 49, (51, 52))}.items():
        dt = distance_transform_edt(anat.lab != thal, sampling=vox)
        dl = distance_transform_edt(~np.isin(anat.lab, lent), sampling=vox)
        ijk = np.argwhere((anat.lab == wm) & (dt <= PLIC_MM) & (dl <= PLIC_MM))
        w = ijk @ anat.affine[:3, :3].T + anat.affine[:3, 3]
        pick = w[np.argmin(((w - w.mean(0)) ** 2).sum(1))]
        mm = T1[:3, :3] @ pick + T1[:3, 3]
        v = np.rint(np.linalg.inv(aff)[:3, :3] @ mm + np.linalg.inv(aff)[:3, 3]).astype(int)
        out.append({"name": f"posterior limb of the internal capsule, {side}",
                    "defined by": f"FastSurfer: white matter within {PLIC_MM:g} mm of both the thalamus and the lentiform nucleus "
                                  f"({len(ijk)} voxels at 1 mm), the one nearest their centroid",
                    "score": None, "voxel": [int(c) for c in v], "mm": [round(float(c), 1) for c in aff[:3, :3] @ v + aff[:3, 3]],
                    "tracts": ["CST"]})
    return out


def rank_field_at(mean, P, flat):
    """The voxel rank field at one voxel: classes by mean count over the runs, those of at least MIN_COUNT."""
    m = mean[:, flat]
    order = np.argsort(-m)
    return [{"tract": TRACT_NAMES[c], "mean streamlines": round(float(m[c]), 1),
             "share of runs present": None if c == OTHER else round(float(P[c, flat]), 2)}
            for c in order if m[c] >= MIN_COUNT][:5]


def main(args):
    from _masivar import scan_files
    t0 = time.time()
    subject, (ses, scan), rescans, shells, _ = PAIRS[args.pair]
    d = DATA / "ensemble" / subject / "retest" / args.pair
    img = nib.load(scan_files(subject, ses, scan, shells)["APP"][0][1])
    aff, shape = img.affine, img.shape[:3]
    nvox = int(np.prod(shape))
    paths = [d / "A.npz"] + [d / "smooth" / f"r{r:03d}.npz" for r in range(1, RUNS)]

    # pass 1: voxel presence and mean counts; each run's seeds and tracts
    votes = np.zeros((NAMED, nvox), np.uint8)
    total = np.zeros((E.N_TRACTS, nvox), np.float32)
    keys, tracts = [], []
    for p in paths:
        r = E.RunRecord.load(p, logp=False)
        votes += r.counts[:NAMED] >= MIN_COUNT
        total += r.counts
        keys.append(r.keys); tracts.append(LUT[r.cluster])
        del r
    P = votes / RUNS
    mean = total / RUNS
    del total
    print(f"pass 1 {time.time() - t0:.0f}s", flush=True)

    # the rank field from all runs: each voxel's first and second named tract by mean count (255: none of MIN_COUNT)
    o = np.argsort(-mean[:NAMED], 0)[:2]                                     # named tracts only: Other leads most voxels
    m2 = np.take_along_axis(mean[:NAMED], o, 0)
    rank1 = np.where(m2[0] >= MIN_COUNT, o[0], 255).astype(np.uint8)
    rank2 = np.where(m2[1] >= MIN_COUNT, o[1], 255).astype(np.uint8)

    mask = np.load(d / "A_mask.npy")
    mid_x = float((np.argwhere(mask) @ aff[:3, :3].T + aff[:3, 3])[:, 0].mean())
    Ps = np.stack([gaussian_filter(P[k].reshape(shape), 1.0) for k in range(NAMED)])
    S = plic_sites(aff) + sites(Ps, mean, aff, shape, mid_x)
    del Ps

    # seeds labeled in all 17 runs; their tracts (m, 17)
    common = keys[0]
    for k in keys[1:]:
        common = np.intersect1d(common, k)
    idx = [np.searchsorted(k, common) if np.all(np.diff(k) > 0) else None for k in keys]
    for i, k in enumerate(keys):
        if idx[i] is None:
            s = np.argsort(k)
            idx[i] = s[np.searchsorted(k[s], common)]
        assert (k[idx[i]] == common).all()
    tr = np.stack([tracts[i][idx[i]] for i in range(RUNS)], 1)                   # (m, 17)

    # A with its streamlines and probabilities: which seeds pass near each site; A's runner-up for every seed
    A = E.RunRecord.load(paths[0], fibers=True)
    first, second, p2 = runner_up(A.logp)
    del A.logp
    rowA = idx[0]
    lens = np.array([len(f) for f in A.fibers])
    verts = np.concatenate(A.fibers)
    owner = np.repeat(np.arange(len(lens)), lens)
    common_row = np.full(len(A.keys), -1)
    common_row[rowA] = np.arange(len(common))
    near = []
    for s in S:
        dd = np.linalg.norm(verts - np.array(s["mm"]), axis=1)
        rows = np.unique(owner[dd <= NEAR_MM])
        near.append(common_row[rows][common_row[rows] >= 0])
    del verts, owner

    # the rescans: presence maps, and each common seed's tract on the rescan (-1: not labeled there)
    resc = {}
    for t in rescans:
        r = E.RunRecord.load(d / f"{t}_onA.npz", logp=False)
        pos = np.searchsorted(r.keys, common) if np.all(np.diff(r.keys) > 0) else None
        if pos is None:
            s = np.argsort(r.keys); pos = s[np.minimum(np.searchsorted(r.keys[s], common), len(s) - 1)]
        pos = np.minimum(pos, len(r.keys) - 1)
        hit = r.keys[pos] == common
        resc[t] = {"tract": np.where(hit, LUT[r.cluster[pos]], -1), "row": np.where(hit, pos, -1),
                   "presence": r.counts[:NAMED] >= MIN_COUNT}
        del r

    # whole brain: does a changed label go to A's runner-up? Each seed's mode over the 17 runs and its share
    cnt_all = np.zeros((len(tr), E.N_TRACTS), np.int32)
    np.add.at(cnt_all, (np.repeat(np.arange(len(tr)), RUNS), tr.ravel()), 1)
    mode_all, share_all = cnt_all.argmax(1), cnt_all.max(1) / RUNS
    a_tr, a_up = tr[:, 0], second[rowA]
    brain = {"seeds labeled in all 17 runs": int(len(common))}
    for t, R in resc.items():
        ok = R["tract"] >= 0
        ch = ok & (R["tract"] != a_tr)
        brain[f"{t}: changed"] = round(float(ch.sum() / ok.sum()), 3)
        brain[f"{t}: changed, to A's runner-up"] = round(float((R["tract"][ch] == a_up[ch]).mean()), 3)
        brain[f"{t}: changed, to a tract A's 17 runs gave"] = round(float((tr[ch] == R["tract"][ch, None]).any(1).mean()), 3)
        brain[f"{t}: all seeds, the rescan's tract among the 17 runs'"] = round(float((tr[ok] == R["tract"][ok, None]).any(1).mean()), 3)
        brain[f"{t}: gives the mode / stated"] = [round(float((R["tract"][ok] == mode_all[ok]).mean()), 3), round(float(share_all[ok].mean()), 3)]
    loo = leave_one_out(tr)
    brain["replicate (leave one run out): gives the mode / stated"] = [round(float(loo[0].mean()), 3), round(float(loo[1].mean()), 3)]
    split = (tr != tr[:, :1]).any(1)
    alt = np.array([np.bincount(np.where(row == row[0], E.N_TRACTS, row), minlength=E.N_TRACTS + 1)[:E.N_TRACTS].argmax() for row in tr[split]])
    brain["runs split: share of seeds"] = round(float(split.mean()), 3)
    brain["runs split: the most frequent other tract is A's runner-up"] = round(float((alt == a_up[split]).mean()), 3)
    brain["A's runner-up probability (median, all seeds)"] = round(float(np.median(p2[rowA])), 3)
    print(json.dumps(brain), flush=True)

    # per site
    rng = np.random.default_rng(0)
    fan_rows = []
    for s, nr in zip(S, near):
        flat = int(np.ravel_multi_index(s["voxel"], shape))
        s["rank field at the site (mean over 17 runs)"] = rank_field_at(mean, P, flat)
        t = tr[nr]
        counts = np.stack([np.bincount(row, minlength=E.N_TRACTS) for row in t]) if len(nr) else np.zeros((0, E.N_TRACTS), int)
        mode, share = counts.argmax(1), counts.max(1) / RUNS
        named = mode != OTHER
        s["seeds near (labeled in all 17 runs)"] = int(len(nr))
        s["share whose mode is a named tract"] = round(float(named.mean()), 3)
        s["modes (top 5)"] = {TRACT_NAMES[c]: int(n) for c, n in sorted(zip(*np.unique(mode, return_counts=True)), key=lambda x: -x[1])[:5]}
        s["all 17 runs agree"] = round(float((share == 1).mean()), 3)
        s["mode's mean share (stated P that another run gives the mode)"] = round(float(share.mean()), 3)
        lo = leave_one_out(t)
        s["replicate (leave one run out): gives the mode / stated"] = [round(float(lo[0].mean()), 3), round(float(lo[1].mean()), 3)]
        s["rescans"] = {}
        for tag, R in resc.items():
            rt = R["tract"][nr]
            ok = rt >= 0
            ch = ok & (rt != t[:, 0])
            s["rescans"][tag] = {"seeds labeled": int(ok.sum()), "rescan gives the mode": round(float((rt[ok] == mode[ok]).mean()), 3),
                                 "stated (mode's mean share, these seeds)": round(float(share[ok].mean()), 3),
                                 "rescan's tract among the 17 runs' tracts": round(float((t[ok] == rt[ok, None]).any(1).mean()), 3),
                                 "changed from A": round(float(ch.sum() / max(ok.sum(), 1)), 3),
                                 "changed, to A's runner-up": round(float((rt[ch] == a_up[nr][ch]).mean()), 3) if ch.any() else None}
        dis = np.flatnonzero((share <= 0.6) & (t != OTHER).any(1))
        agr = np.flatnonzero((share == 1) & named)
        s["fan seeds"] = {"split (mode in at most 60 % of runs, some named)": int(len(dis)), "all agree on a named tract": int(len(agr))}
        pick = {"split": rng.choice(dis, min(args.fans, len(dis)), replace=False),
                "agree": rng.choice(agr, min(args.fans, len(agr)), replace=False)}
        fan_rows.append({k: nr[v] for k, v in pick.items()})

    # pass 2: the fans' streamlines from every run and rescan, with each version's tract and runner-up
    want = np.unique(np.concatenate([np.concatenate(list(f.values())) for f in fan_rows]))
    lines = {}                                                            # (common row, source) -> (fiber, tract, runner-up)
    for i, p in enumerate(paths):
        r = A if i == 0 else E.RunRecord.load(p, fibers=True)
        if i == 0:
            fs, sc = first[idx[0][want]], second[idx[0][want]]
        else:
            fs, sc, _ = runner_up(r.logp[idx[i][want]])
        for w, f_, s_ in zip(want, fs, sc):
            lines[(int(w), i)] = (r.fibers[idx[i][w]], int(f_), int(s_))
        del r
    for j, (tag, R) in enumerate(resc.items()):
        r = E.RunRecord.load(d / f"{tag}_onA.npz", fibers=True)
        rows = R["row"][want]
        ok = rows >= 0
        fs, sc, _ = runner_up(r.logp[rows[ok]])
        for w, rw, f_, s_ in zip(want[ok], rows[ok], fs, sc):
            lines[(int(w), RUNS + j)] = (r.fibers[rw], int(f_), int(s_))
        del r
    print(f"pass 2 {time.time() - t0:.0f}s", flush=True)

    # the fans as flat arrays: per (site, kind, seed, version) a streamline
    fan_site, fan_kind, fan_src, fan_tract, fan_up, fan_len, fan_pts = [], [], [], [], [], [], []
    for si, f in enumerate(fan_rows):
        for ki, kind in enumerate(("split", "agree")):
            for w in f[kind]:
                for src in range(RUNS + len(resc)):
                    if (int(w), src) not in lines:
                        continue
                    fib, ft, up = lines[(int(w), src)]
                    fan_site.append(si); fan_kind.append(ki); fan_src.append(src); fan_tract.append(ft); fan_up.append(up)
                    fan_len.append(len(fib)); fan_pts.append(fib)
            fan_seed_count = len(f[kind])
            S[si]["fan seeds"][f"drawn ({kind})"] = int(fan_seed_count)

    # envelopes and the rescans against them, for the shown tracts
    env = {}
    for name in SHOWN:
        k = T(name)
        t80 = threshold(P[k], ENVELOPE_80)
        inside = P[k] >= t80 - 1e-6
        row = {"core votes": int(np.ceil(RUNS / 2)), "80 % envelope votes": int(round(t80 * RUNS)), "voxels: core / 80 % / widest":
               [int((votes[k] >= np.ceil(RUNS / 2)).sum()), int(inside.sum()), int((votes[k] > 0).sum())]}
        for tag, R in resc.items():
            pr = R["presence"][k]
            row[f"{tag}: rescan voxels inside the 80 % envelope"] = round(float((pr & inside).sum() / max(pr.sum(), 1)), 3)
            row[f"{tag}: inside the widest"] = round(float((pr & (votes[k] > 0)).sum() / max(pr.sum(), 1)), 3)
        env[name] = row

    # the focus tracts: how A's streamlines of each fare on a fresh run and on each rescan, and where changed ones go
    focus = {}
    for name in FOCUS:
        k = T(name)
        m = tr[:, 0] == k
        row = {"A's streamlines (labeled in all 17 runs)": int(m.sum()),
               "stated: the runs' share giving it": round(float((tr[m] == k).mean()), 3),
               "a fresh run keeps it (the 16 others)": round(float((tr[m, 1:] == k).mean()), 3)}
        for tag, R in resc.items():
            rt = R["tract"][m]
            ok = rt >= 0
            row[f"{tag}: keeps it"] = round(float((rt[ok] == k).mean()), 3)
            gone = rt[ok & (rt != k)]
            row[f"{tag}: changed to (top 3)"] = {TRACT_NAMES[c]: round(float(n / max(len(gone), 1)), 2)
                                                for c, n in sorted(zip(*np.unique(gone, return_counts=True)), key=lambda x: -x[1])[:3]}
        focus[name] = row
    print(json.dumps(focus), flush=True)

    res = {"pair": args.pair, "runs": RUNS, "min_count": MIN_COUNT, "near_mm": NEAR_MM, "midline x (mm)": round(mid_x, 1),
           "whole brain": brain, "focus tracts": focus, "sites": S, f"envelopes (80 % = stated {ENVELOPE_80})": env}
    (HERE / "results/ensemble_landmarks_smooth.json").write_text(json.dumps(res, indent=1))
    sh = [T(n) for n in SHOWN]
    np.savez_compressed(d / "landmarks.npz", votes=votes[sh].reshape(len(sh), *shape), shown=np.array(SHOWN),
                        rank1=rank1.reshape(shape), rank2=rank2.reshape(shape),
                        rescan_presence=np.stack([R["presence"][sh].reshape(len(sh), *shape) for R in resc.values()]),
                        rescans=np.array(list(resc)), affine=aff, meanb0=np.load(d / "A_meanb0.npy"), mask=mask,
                        fan_site=np.array(fan_site, np.int16), fan_kind=np.array(fan_kind, np.int8), fan_src=np.array(fan_src, np.int8),
                        fan_tract=np.array(fan_tract, np.int8), fan_up=np.array(fan_up, np.int8), fan_len=np.array(fan_len, np.int32),
                        fan_pts=np.concatenate(fan_pts).astype(np.float32))
    print(json.dumps({s["name"]: [s["seeds near (labeled in all 17 runs)"], s["all 17 runs agree"], s["rescans"]] for s in S}, indent=0)[:4000])
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="cIIIs04-2.1", choices=sorted(PAIRS))
    ap.add_argument("--fans", type=int, default=16)
    main(ap.parse_args())
