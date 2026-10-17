"""RapidParc's context groups keyed by seed, against its index shuffle (docs/robustness-plan.md, step 1).

    uv run --group rankfield --group sklearn bench/robust_labeler.py [--draws 16] [--out results/robust_labeler.json]
    uv run --group rankfield --group sklearn bench/robust_labeler.py --smoke --out <scratch>.json

RapidParc labels a streamline in the company of its 2,000 group-mates, and its groups are cuts of a shuffle of all
streamlines (labelers/rapidparc.py, grouping "index"). A run whose seed set differs from another's by a few
streamlines puts every streamline in new company: on cIIIs04 a 1e-6 nudge to the DWI (the epsilon arm, about 82 seeds
lost a run) renames 3.4 % of the streamlines whose path did not move, about a labeler draw's worth (NOTES 2026-10-17).
Two opt-in groupings key the company to the seed instead (a hash of seed key and draw):
  hash-sorted  the hash order cut every 2,000: removing a streamline shifts each later group by one member;
  hash-bucket  group floor(u G), G = ceil(n / 2,000): a group's members are its seeds whatever else exists (while G
               holds), groups of about n / G (1,000-2,000) +- binomial.
Relabels the existing compact records (cIIIs04 A's ensemble, DATA/ensemble/cIIIs04/retest/cIIIs04-2.1; no tracking)
with each grouping, draw 0 unless stated, every run against A relabeled the same way:
  (a) epsilon r001..r016 against A: renames split by how far the path moved (ensemble_stages' MDF bins);
  (b) draws: A relabeled with draws 0..k, each draw against draw 0 and against the next;
  (c) the full arm (A + smooth r001..r016) against the three rescans tracked on A's grid (relabeled the same way):
      stated mode share, how often a rescan gives the mode, AUROC (ensemble_stages.against_rescans, over A's seeds
      every run labels), and the rescan's renames against A over those seeds; their spread over draws 0..RESCAN_DRAWS
      (A and the rescan relabeled at the same draw);
  (d) accuracy on TractCloud's test split (as accuracy_tractcloud_test.py loads it; 20 subjects x 10,000
      streamlines), draws 0 and 1. The split has no seeds: the key is the streamline's index in its subject;
  (e) the Other share and the tract mix of the full arm's 17 runs pooled, against the index grouping's;
  (f) why (c) does not move: per hash-bucket group of each full run and rescan, the share of members whose seed A
      lacks or whose path moved more than 5 mm; and their renames against A split by path distance.
The relabeled index grouping differs from the stored labels only through the records' 0.05 mm quantization (A's
share is reported). --smoke: epsilon r001, draws 0..2, the full arm A + r001, accuracy on 2 subjects x 5,000.
Writes --out (refuses to overwrite); nothing under DATA.
"""
import argparse, json, pickle, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np, torch
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers import rapidparc as RP
from tractline.labelers.base import TRACT_NAMES, OTHER

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_stages import PAIR, FLIP, where, by_distance, against_rescans  # noqa: E402
from ensemble_retest import PAIRS  # noqa: E402

DEVICE = "mps"
RESCAN_DRAWS = 4                                        # (c): the rescans' renames at draws 0..4 (A's from (b))
MODES = RP.GROUPINGS                                    # index (RapidParc's), hash-sorted, hash-bucket
L = np.asarray(RP.LUT)


def relabel(lab, rec, mode, draw=0):
    """rec's streamlines relabeled: (keys, tract) of the ones labeled (all of them, unless quantization moved one
    below 40 mm)"""
    lb = lab(rec.fibers, draws=(draw,), keys=rec.keys, grouping=mode)
    return rec.keys[lb.keep], lb.tract


def on_A(SA, n, keys, tract):
    """A tract per A seed (-1 where the run has none)"""
    ri, qi = SA.match(SimpleNamespace(keys=keys))
    t = np.full(n, -1, np.int16)
    t[ri] = tract[qi]
    return t


def mdf_on_A(SA, A, rec):
    n = len(A.keys)
    ri, qi = SA.match(rec)
    m = np.full(n, np.nan, np.float32)
    m[ri] = E.mdf(A.paths[ri], rec.paths[qi])
    return m


def changed(a, b):
    ok = (a >= 0) & (b >= 0)
    return float((a[ok] != b[ok]).mean())


FAR_BINS = (("MDF <= 1 mm", 0.0, 1.0), ("MDF (1, 5] mm", 1.0, 5.0), ("MDF > 5 mm", 5.0, np.inf))


def by_far(X, M, tA):
    """(f): renames against A by path distance, in bins for runs whose paths all moved (MDF 0 counted in the first)"""
    ok = X >= 0
    ch = (X != tA) & ok
    out = {}
    for name, lo, hi in FAR_BINS:
        b = ok & (M <= hi) & ((M > lo) if lo > 0 else True)
        out[name] = {"share of streamlines": round(float(b.sum() / ok.sum()), 4), "renamed": round(float(ch[b].sum() / max(b.sum(), 1)), 4)}
    return out


def company(SA, A, rec):
    """(f): the run's hash-bucket groups (draw 0): G, and per group the share of members whose seed A lacks or whose
    path is more than 5 mm from A's (mean, min over groups)"""
    ri, qi = SA.match(rec)
    m = np.full(len(rec.keys), np.inf); m[qi] = E.mdf(A.paths[ri], rec.paths[qi])
    far = [float((m[g] > 5).mean()) for g in RP.keyed_groups(rec.keys, 0, "hash-bucket")]
    return {"G": len(far), "new or > 5 mm: mean, min over groups": [round(float(np.mean(far)), 3), round(float(np.min(far)), 3)]}


def mix(T):
    """the share of each of the 43 classes over every labeled (run, seed)"""
    c = np.bincount(T[T >= 0].astype(np.int64), minlength=E.N_TRACTS)
    return c / c.sum()


def accuracy(lab, subjects, draws=(0, 1)):
    """TractCloud's test split per mode: tract accuracy, macro F1, cluster accuracy, Other share, draw agreement"""
    from sklearn.metrics import f1_score
    out = {}
    for mode in MODES:
        pred, t0 = {s: [] for s in draws}, time.time()
        truth = []
        for feat, label in subjects:
            keys = np.arange(len(feat), dtype=np.int64)                  # no seeds here: the index stands in
            truth.append(label)
            for s in draws:
                pred[s].append(lab.logits(feat, s, keys, mode).argmax(1).numpy())
        truth = np.concatenate(truth); tt = L[truth]
        row = {"seconds per subject and draw": round((time.time() - t0) / (len(subjects) * len(draws)), 2)}
        for s in draws:
            p = np.concatenate(pred[s]); pt = L[p]
            row[f"draw {s}"] = {"tract accuracy %": round(float((pt == tt).mean()) * 100, 2),
                                "tract macro F1 %": round(float(f1_score(tt, pt, average="macro")) * 100, 2),
                                "cluster accuracy %": round(float((p == truth).mean()) * 100, 2),
                                "Other share": round(float((pt == OTHER).mean()), 4)}
        row["draws agree (tract)"] = round(float((L[np.concatenate(pred[draws[0]])] == L[np.concatenate(pred[draws[1]])]).mean()), 4)
        out[mode] = row
        print("accuracy", mode, json.dumps(row), flush=True)
    return out


def main(args):
    t0 = time.time()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} exists: refusing to overwrite (pass another --out)")
    d, sd = where()
    eps_runs = 1 if args.smoke else args.runs
    full_runs = 1 if args.smoke else args.runs
    draws = 2 if args.smoke else args.draws
    lab = RP.Labeler(DEVICE)
    load = lambda p: E.RunRecord.load(p, logp=False, fibers=True)
    A = load(d / "A.npz")
    SA, n = E.StreamlineEnsemble(A), len(A.keys)
    sizes = [len(g) for g in RP.keyed_groups(A.keys, 0, "hash-bucket")]
    res = {"pair": PAIR, "modes": list(MODES), "A seeds": n, "smoke": bool(args.smoke), "context": RP.CONTEXT,
           "hash-bucket groups on A (draw 0): count, min, max size": [len(sizes), min(sizes), max(sizes)]}
    tA = {}
    for mode in MODES:
        tA[mode] = on_A(SA, n, *relabel(lab, A, mode))
    res["A: index relabel vs stored labels (quantization), renamed"] = round(changed(tA["index"], L[A.cluster]), 4)
    res["A: renamed against the index grouping"] = {m: round(changed(tA[m], tA["index"]), 4) for m in MODES[1:]}
    print(json.dumps(res), f"{time.time() - t0:.0f}s", flush=True)

    # (a) epsilon against A, by how far the path moved
    T, M, nrec = {m: [] for m in MODES}, [], []
    for r in range(1, eps_runs + 1):
        rec = load(sd / "epsilon" / f"r{r:03d}.npz")
        nrec.append(len(rec.keys))
        M.append(mdf_on_A(SA, A, rec))
        for mode in MODES:
            T[mode].append(on_A(SA, n, *relabel(lab, rec, mode)))
        print("epsilon", r, f"{time.time() - t0:.0f}s", flush=True)
    M = np.stack(M)
    res["(a) epsilon vs A"] = {"runs": eps_runs, "streamlines per run (mean)": int(np.mean(nrec)),
                               "hash-bucket G per run (A's: %d)" % -(-n // RP.CONTEXT): sorted(set(-(-k // RP.CONTEXT) for k in nrec)),
                               "A's seeds unmatched (mean per run)": round(float(np.isnan(M).sum(1).mean()), 1)}
    for mode in MODES:
        X = np.stack(T[mode])
        per_run = [changed(x, tA[mode]) for x in X]
        res["(a) epsilon vs A"][mode] = {"renamed: mean, SD over runs": [round(float(np.mean(per_run)), 4), round(float(np.std(per_run)), 4)],
                                         "by path distance": by_distance(X, M, tA[mode])}
    del T, M
    print(json.dumps(res["(a) epsilon vs A"]), f"{time.time() - t0:.0f}s", flush=True)

    # (b) draws on A's own streamlines
    res["(b) draws on A"] = {"draws": f"0..{draws}"}
    rd = min(RESCAN_DRAWS, draws)
    DA = {}
    for mode in MODES:
        D = [tA[mode]] + [on_A(SA, n, *relabel(lab, A, mode, k)) for k in range(1, draws + 1)]
        DA[mode] = D[:rd + 1]
        vs0 = [changed(D[k], D[0]) for k in range(1, draws + 1)]
        nxt = [changed(D[k], D[k + 1]) for k in range(draws)]
        res["(b) draws on A"][mode] = {"renamed against draw 0: mean, SD": [round(float(np.mean(vs0)), 4), round(float(np.std(vs0)), 4)],
                                       "renamed, draw k against k + 1 (mean)": round(float(np.mean(nxt)), 4)}
    print(json.dumps(res["(b) draws on A"]), f"{time.time() - t0:.0f}s", flush=True)

    # (c) the full arm against the rescans; (e) its tract mix
    full, Mf, comp = {m: [tA[m]] for m in MODES}, [], {"full": []}
    for r in range(1, full_runs + 1):
        rec = load(d / FLIP / f"r{r:03d}.npz")
        Mf.append(mdf_on_A(SA, A, rec)); comp["full"].append(company(SA, A, rec))
        for mode in MODES:
            full[mode].append(on_A(SA, n, *relabel(lab, rec, mode)))
        print("full", r, f"{time.time() - t0:.0f}s", flush=True)
    rescans, rdraws, Mr = {m: {} for m in MODES}, {m: {} for m in MODES}, {}
    for tag in PAIRS[PAIR][2]:
        rec = load(d / f"{tag}_onA.npz")
        Mr[tag] = mdf_on_A(SA, A, rec); comp[tag] = company(SA, A, rec)
        for mode in MODES:
            rescans[mode][tag] = on_A(SA, n, *relabel(lab, rec, mode))
            rdraws[mode][tag] = [rescans[mode][tag]] + [on_A(SA, n, *relabel(lab, rec, mode, k)) for k in range(1, rd + 1)]
    res["(c) full arm vs rescans"] = {"runs": full_runs + 1}
    common = np.logical_and.reduce([(np.stack(full[m]) >= 0).all(0) for m in MODES])
    for mode in MODES:
        Tm = np.stack(full[mode])
        res["(c) full arm vs rescans"][mode] = against_rescans(Tm, tA[mode], rescans[mode], common)
        c = res["(c) full arm vs rescans"][mode]
        c["rescan vs A, renamed (every A seed both label)"] = {t: round(changed(x, tA[mode]), 4) for t, x in rescans[mode].items()}
        c["rescan vs A, renamed (the seeds above)"] = {t: round(changed(x[common], tA[mode][common]), 4) for t, x in rescans[mode].items()}
        c[f"rescan vs A, renamed (the seeds above), draws 0..{rd}: mean, SD, min, max"] = {
            t: [round(float(f(v)), 4) for f in (np.mean, np.std, np.min, np.max)]
            for t, X in rdraws[mode].items() for v in [[changed(X[k][common], DA[mode][k][common]) for k in range(rd + 1)]]}
    mixes = {m: mix(np.stack(full[m])) for m in MODES}
    res["(e) tract mix, full arm pooled"] = {}
    for mode in MODES:
        diff = mixes[mode] - mixes["index"]
        top = np.argsort(-np.abs(diff))[:5]
        res["(e) tract mix, full arm pooled"][mode] = {
            "Other share": round(float(mixes[mode][OTHER]), 4),
            "total variation against index": round(float(np.abs(diff).sum() / 2), 4),
            "largest share changes against index (points)": {TRACT_NAMES[k]: round(float(diff[k]) * 100, 3) for k in top}}
    # (f) why the rescans do not move: every group's company changes anyway; renames by path distance
    G = [c["G"] for c in comp["full"]]
    fr = np.array([c["new or > 5 mm: mean, min over groups"] for c in comp["full"]])
    res["(f) full arm and rescans: company and renames by path distance"] = {
        "hash-bucket company": {"full runs: G (min, max)": [min(G), max(G)],
                                "full runs: new or > 5 mm, mean over groups (mean over runs), min over groups and runs":
                                    [round(float(fr[:, 0].mean()), 3), round(float(fr[:, 1].min()), 3)],
                                **{tag: comp[tag] for tag in Mr}},
        **{mode: {"full runs": by_far(np.stack(full[mode][1:]), np.stack(Mf), tA[mode]),
                  **{tag: by_far(rescans[mode][tag][None], Mr[tag][None], tA[mode]) for tag in Mr}} for mode in MODES}}
    print(json.dumps(res["(f) full arm and rescans: company and renames by path distance"]), f"{time.time() - t0:.0f}s", flush=True)
    del full, rescans, rdraws, Mf, Mr
    print(json.dumps(res["(c) full arm vs rescans"]), json.dumps(res["(e) tract mix, full arm pooled"]), f"{time.time() - t0:.0f}s", flush=True)

    # (d) accuracy on TractCloud's test split
    z = pickle.load(open(DATA / "TrainData_800clu800ol/test.pickle", "rb"))
    feat, label, subj = z["feat"].astype(np.float32), z["label"], z["subject_id"]
    ids = list(dict.fromkeys(subj.tolist()))[: 2 if args.smoke else None]
    subjects = []
    for sid in ids:
        idx = np.flatnonzero(subj == sid)[: 5000 if args.smoke else None]
        subjects.append((torch.from_numpy(feat[idx]), label[idx]))
    del z, feat
    res["(d) TractCloud test split"] = {"subjects": len(subjects), "streamlines": int(sum(len(f) for f, _ in subjects)),
                                        "keys": "the streamline's index in its subject (the split has no seeds)",
                                        **accuracy(lab, subjects)}
    res["seconds"] = round(time.time() - t0, 1)
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out, f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="one epsilon run, draws 0..2, A + one full run, 2 x 5,000 test streamlines")
    ap.add_argument("--runs", type=int, default=16, help="epsilon and full-arm runs")
    ap.add_argument("--draws", type=int, default=16, help="(b): draws 0..draws")
    ap.add_argument("--out", default=str(HERE / "results/robust_labeler.json"))
    main(ap.parse_args())
