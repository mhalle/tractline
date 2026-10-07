"""The pipeline's ensemble on one patient: does uncertainty kept from every stage say more than one run's
labeler probabilities, and can a rank field carry it? (tractline.ensemble; NOTES 2026-10-07)

    uv run --group dipy --group rankfield bench/ensemble_pilot.py run [--subject PAT16] [--runs 16] [--draws 8]
    uv run --group dipy --group rankfield bench/ensemble_pilot.py analyze [--subject PAT16] [--build 10]

run: the reference - the pipeline as it runs today (the scan as acquired, RapidParc draw 0) - then
`runs` runs with every stage resampled: a wild-bootstrap replicate of the scan as acquired, both
series' b0s included (_bootstrap.ShellWildBootstrap: each shell fitted at its own SH order), so the
field is re-estimated from it; the replicate corrected, tracked (Metal) and labeled with draw r. And
`draws` draw-only runs: the reference's tractogram labeled with draws 1..draws, the labeler's share
alone. Each run's RunRecord goes to DATA/ensemble/<subject>/ (the reference r000, replicates r001..,
draw-only d001..), skipped when there; a summary per run in runs.json there.

analyze: the ensemble from the reference and replicates 1..build; replicates build+1.. held out as
fresh runs of the pipeline. Per streamline of the reference (matched by seed voxel): which confidence
predicts a held-out run changing the reference's tract (AUROC), and how well each is calibrated (the
held-out agreement against the stated probability); the ensemble encoded as a rank field (rankfield,
depth 6, keep "clip", as the web export) with its vote planes; votes regrouped after the fact (SLF
I-III, CC 1-7 merged) from the planes alone against the truth; and path distances between runs. Per
voxel (the DWI grid): the tract density field encoded (depth 6, keep "shell"), and its votes'
calibration against the held-out runs' winners. Writes results/ensemble_pilot_<b0 noise>.json.
"""
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np, torch
from scipy.stats import rankdata
from tractline import pipeline as P
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT, TRACT_NAMES, OTHER

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results/ensemble_pilot_{}.json"            # .format(b0_noise)
DEVICE = "mps"


def record_dir(subject, b0_noise):
    d = DATA / "ensemble" / subject / b0_noise
    d.mkdir(parents=True, exist_ok=True)
    return d


def run(args):
    from _ds001226 import load
    from _bootstrap import ShellWildBootstrap
    s = load(args.subject)
    d = record_dir(args.subject, args.b0_noise)
    log_path = d / "runs.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    labeler = P.default_labeler(DEVICE)
    grid = (s.affine, s.dwi.shape[:3])
    n_ap = len(s.b0_index)

    def one(name, subj, draw):
        if (d / f"{name}.npz").exists():
            return None
        timer = P.Timer()
        t0 = time.time()
        with P.exact_float32(DEVICE):
            corr = P.correct(subj, timer, DEVICE)
            tg = P.track(subj, corr.dwi, timer, device=DEVICE)
            with timer("label"):
                labels = labeler(tg.fibers, draws=(draw,), logp=True)
        rec = E.RunRecord.of(tg, labels, *grid)
        rec.save(d / f"{name}.npz")
        log[name] = {"draw": draw, "fibers": len(tg.fibers), "labeled": int(labels.keep.sum()),
                     "other": round(float((labels.tract == OTHER).mean()), 4), "residual_left": corr.residual_left,
                     "field_hz_p99_abs": round(float(np.quantile(np.abs(corr.field_hz), 0.99)), 2) if corr.field_hz is not None else None,
                     "seconds": timer.seconds, "total_s": round(time.time() - t0, 1)}
        log_path.write_text(json.dumps(log, indent=1))
        print(name, json.dumps(log[name]), flush=True)
        P.release_memory(DEVICE)
        return tg, labels

    ref = one("r000", s, 0)
    if ref is None:
        with P.exact_float32(DEVICE):
            from tractline.prep import prepare
            mask = prepare(P.correct(s, P.Timer(), DEVICE).dwi, s.affine, s.bval, s.bvec, device=DEVICE).mask
    else:
        mask = ref[0].mask
    wb = ShellWildBootstrap(s.dwi, s.bval, s.bvec, extra_b0s=s.b0s[..., n_ap:], b0_noise=args.b0_noise, mask=mask)
    log["bootstrap"] = wb.noise_check(mask)
    log_path.write_text(json.dumps(log, indent=1))
    for r in range(1, args.runs + 1):
        if (d / f"r{r:03d}.npz").exists():
            continue
        dwi, pa = wb.replicate(r)
        subj = replace(s, dwi=dwi, b0s=np.concatenate([dwi[..., s.b0_index].astype(np.float64), pa], -1))
        one(f"r{r:03d}", subj, r)
    # the labeler's share alone: the reference tractogram, other draws
    need = [k for k in range(1, args.draws + 1) if not (d / f"d{k:03d}.npz").exists()]
    if need:
        timer = P.Timer()
        with P.exact_float32(DEVICE):
            corr = P.correct(s, timer, DEVICE)
            tg = P.track(s, corr.dwi, timer, device=DEVICE)
        for k in need:
            labels = labeler(tg.fibers, draws=(k,), logp=True)
            E.RunRecord.of(tg, labels, *grid).save(d / f"d{k:03d}.npz")
            log[f"d{k:03d}"] = {"draw": k, "other": round(float((labels.tract == OTHER).mean()), 4)}
            log_path.write_text(json.dumps(log, indent=1))
            print(f"d{k:03d}", log[f"d{k:03d}"], flush=True)


# ----------------------------------------------------------------------------------------------- analysis

def auroc(score, positive):
    """P(score of a positive > score of a negative), ties half."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n1, n0 = positive.sum(), (~positive).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(score)
    return round(float((r[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)


def calibration(p, hit, bins=(0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001)):
    """Stated probability p against the observed rate of hit, per bin; Brier score and the expected calibration
    error (bin-weighted |mean p - rate|)."""
    p, hit = np.asarray(p, float), np.asarray(hit, float)
    rows, ece = [], 0.0
    for a, b in zip(bins[:-1], bins[1:]):
        m = (p >= a) & (p < b)
        if m.sum() == 0:
            continue
        rows.append({"bin": [a, min(b, 1.0)], "n": int(m.sum()), "stated": round(float(p[m].mean()), 4),
                     "observed": round(float(hit[m].mean()), 4)})
        ece += m.mean() * abs(p[m].mean() - hit[m].mean())
    return {"brier": round(float(((p - hit) ** 2).mean()), 4), "ece": round(float(ece), 4), "bins": rows}


def tract_mass(logp):
    """(n, 1600) log-probabilities -> (n, 43) tract probabilities (float64)."""
    p = np.exp(np.asarray(logp, np.float64))
    out = np.zeros((len(p), E.N_TRACTS))
    np.add.at(out.T, LUT, p.T)
    return out


def logodds(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _winners_only(rec):
    """A record whose log-probabilities were not loaded, for an ensemble that only counts winners and paths: a
    one-hot stand-in (its mixture is not read)."""
    lp = np.full((len(rec.keys), 1600), -30.0, np.float16)
    lp[np.arange(len(rec.keys)), rec.cluster] = 0.0
    return replace(rec, logp=lp)


def analyze(args):
    import rankfield as rf
    d = record_dir(args.subject, args.b0_noise)
    names = sorted(p.stem for p in d.glob("r*.npz"))
    ref = E.RunRecord.load(d / "r000.npz")
    build = [n for n in names if n != "r000" and int(n[1:]) <= args.build]
    held = [n for n in names if int(n[1:]) > args.build]
    draws = sorted(p.stem for p in d.glob("d*.npz"))
    assert build and held, f"need runs above and below --build {args.build}: have {names}"
    res = {"subject": args.subject, "b0_noise": args.b0_noise, "reference": "r000: the scan as acquired, RapidParc draw 0 (the pipeline today)",
           "ensemble_runs": ["r000"] + build, "held_out_runs": held, "draw_only_runs": draws,
           "runs_log": json.loads((d / "runs.json").read_text())}

    S = E.StreamlineEnsemble(ref)
    V = E.VoxelEnsemble(ref.shape, tau=args.tau)
    for n in ["r000"] + build:
        rec = E.RunRecord.load(d / f"{n}.npz")
        S.add(rec); V.add(rec)
    R = S.runs
    ref_tract = LUT[ref.cluster]
    # confidences per reference streamline (high = confident in the reference's tract)
    Pref = tract_mass(ref.logp)
    p_single = Pref[np.arange(S.n), ref_tract]
    other = Pref.copy(); other[np.arange(S.n), ref_tract] = -1
    margin_single = p_single - other.max(1)                                  # TRX's tract_margin
    mix = S.mixture().numpy()
    Pmix = tract_mass(mix)
    p_mix = Pmix[np.arange(S.n), ref_tract]
    votes = S.votes()                                                       # (n, 43)
    vfrac = votes[np.arange(S.n), ref_tract] / np.maximum(S.present, 1)
    consensus = votes.argmax(1)
    cfrac = votes.max(1) / np.maximum(S.present, 1)

    # held-out runs: per (streamline, run) where the run labeled the seed
    H = E.StreamlineEnsemble(ref)
    for n in held:
        H.add(_winners_only(E.RunRecord.load(d / f"{n}.npz", logp=False)))
    W = np.stack(H.winners)                                                 # (h, n)
    have = W >= 0
    held_tract = np.where(have, LUT[np.maximum(W, 0)], -1)
    rows = np.broadcast_to(np.arange(S.n), W.shape)[have]
    changed = (held_tract != ref_tract[None])[have]
    agrees_consensus = (held_tract == consensus[None])[have]
    res["streamlines"] = {
        "reference_labeled": int(S.n), "ensemble_runs": R,
        "present_in_ensemble_runs": {"all": int((S.present == R).sum()), "median": float(np.median(S.present))},
        "held_out_pairs": int(have.sum()),
        "held_out_change_rate": round(float(changed.mean()), 4),
        "auroc_predicting_change (lower confidence = change)": {
            "single run: tract probability (TRX)": auroc(-p_single[rows], changed),
            "single run: tract margin (TRX)": auroc(-margin_single[rows], changed),
            "ensemble: mixture tract probability": auroc(-p_mix[rows], changed),
            "ensemble: vote fraction for the reference's tract": auroc(-vfrac[rows], changed)},
        "calibration: P(held-out run keeps the reference's tract)": {
            "single run: tract probability": calibration(p_single[rows], ~changed),
            "ensemble: mixture tract probability": calibration(p_mix[rows], ~changed),
            "ensemble: vote fraction": calibration(vfrac[rows], ~changed)},
        "calibration: P(held-out run gives the ensemble's consensus)": {
            "vote fraction of the consensus": calibration(cfrac[rows], agrees_consensus),
            "consensus agreement": round(float(agrees_consensus.mean()), 4)},
        "consensus differs from the reference": round(float((consensus != ref_tract).mean()), 4)}

    # the labeler's share: draw-only runs on the reference tractogram
    if draws:
        D = E.StreamlineEnsemble(ref)
        for n in draws:
            D.add(_winners_only(E.RunRecord.load(d / f"{n}.npz", logp=False)))
        Wd = np.stack(D.winners)
        res["streamlines"]["change_rate_by_source"] = {
            "labeler draw only (the reference's tractogram)": round(float((LUT[Wd] != ref_tract[None]).mean()), 4),
            "every stage (held-out runs)": round(float(changed.mean()), 4)}

    # positions: path distance to the reference's path (MDF, mm), and whether it goes with label change
    dist = H.distances()[have]
    res["streamlines"]["paths"] = {
        "mdf_mm_quantiles (0.5, 0.9, 0.99)": [round(float(v), 2) for v in np.quantile(dist, [0.5, 0.9, 0.99])],
        "auroc: mdf predicting change": auroc(dist, changed),
        "auroc: ensemble median mdf predicting change": auroc(np.nanmedian(S.distances(), 0)[rows], changed)}

    # the rank field of the mixture (as the web export: depth 6, keep clip) and its vote planes
    lg = torch.from_numpy(np.maximum(mix, -1e4).T.copy()).reshape(1600, S.n, 1, 1)
    code = rf.encode(lg, depth=6, keep="clip")
    cv = S.cluster_votes(code.ranks.reshape(6, -1).astype(np.int64) - 1)       # (6, n)
    allv = np.zeros((1600, S.n), np.int32)
    Wb = np.stack(S.winners)
    for w in Wb:
        ok = w >= 0
        np.add.at(allv, (w[ok], np.flatnonzero(ok)), 1)
    vp, vtail = E.vote_planes(code.ranks.reshape(6, -1), allv, S.present)
    assert (vp.astype(np.int64) == cv).all()
    ids, pr = rf.probabilities(code)
    ids, pr = ids.reshape(6, -1), pr.reshape(6, -1)
    p_dec = np.where(ids >= 0, pr, 0) * (LUT[np.maximum(ids, 0)] == ref_tract[None])
    p_dec = p_dec.sum(0)
    merged = LUT.copy()
    slf = [TRACT_NAMES.index(t) for t in ("SLF-I", "SLF-II", "SLF-III")]
    cc = [TRACT_NAMES.index(f"CC{k}") for k in range(1, 8)]
    remap = np.arange(E.N_TRACTS); remap[slf] = slf[0]; remap[cc] = cc[0]
    _, remap = np.unique(remap, return_inverse=True)
    merged = remap[LUT]
    G = int(merged.max()) + 1
    gv_planes = E.group_votes(code.ranks.reshape(6, -1), vp, merged, G)       # (G, n)
    gv_true = S.votes(merged, G).T
    res["streamline_rank_field"] = {
        "meta": {k: code.meta[k] for k in ("depth", "keep", "clip", "classes")},
        "bytes_per_streamline": {"ranks+support+tail": round(code.nbytes / S.n, 1), "vote planes + vote tail": 7},
        "vote_tail_zero": round(float((vtail == 0).mean()), 4),
        "vote_tail_quantiles (0.99, 1.0)": [int(np.quantile(vtail, 0.99)), int(vtail.max())],
        "regrouped votes from the planes exact (SLF I-III, CC 1-7 merged)": round(float((gv_planes == gv_true).all(0).mean()), 4),
        "auroc: decoded mixture tract probability predicting change": auroc(-p_dec[rows], changed),
        "decoded mixture probability vs full: max abs": round(float(np.abs(p_dec - p_mix).max()), 4)}

    # voxels: held-out winners against the ensemble's votes; a single run's map has no confidence at all
    nvox = V.sum.shape[1]
    votesV = V.vote.astype(np.float64)
    cons = V.winner(V.sum / V.runs)                                       # the field's winner: pooled density, NONE at tau
    cfV = votesV[cons, np.arange(nvox)] / V.runs
    any_tract = (V.sum.max(0) > 0)
    hits, refhit, pc = [], [], []
    ref_win = V.winner(ref.counts)
    for n in held:
        hw = V.winner(E.RunRecord.load(d / f"{n}.npz", logp=False).counts)
        hits.append((hw == cons)[any_tract]); refhit.append((hw == ref_win)[any_tract]); pc.append(cfV[any_tract])
    hits, refhit, pc = np.concatenate(hits), np.concatenate(refhit), np.concatenate(pc)
    lgV = torch.from_numpy(V.logits().transpose(0, 3, 2, 1).copy())          # (44, Z, Y, X)
    codeV = rf.encode(lgV, depth=6, keep="shell")
    rV = codeV.ranks                                                         # (6, Z, Y, X)
    vV, tV = E.vote_planes(rV, V.votes().transpose(0, 3, 2, 1), V.runs)
    win_dec = rV[0].astype(np.int64) - 1
    win_true = lgV.numpy().argmax(0)
    inside = any_tract.reshape(V.shape).transpose(2, 1, 0)
    res["voxels"] = {
        "grid": list(V.shape), "tau": V.tau, "voxels_any_tract": int(any_tract.sum()),
        "agreement with a held-out run's winner": {"ensemble consensus": round(float(hits.mean()), 4),
                                                   "the reference run's map": round(float(refhit.mean()), 4)},
        "calibration: P(held-out winner = consensus)": calibration(pc, hits),
        "rank_field": {"meta": {k: codeV.meta[k] for k in ("depth", "keep", "clip", "classes")},
                       "bytes_per_voxel": round(codeV.nbytes / nvox, 1),
                       "labels exact (winner)": round(float((win_dec == win_true)[inside].mean()), 6),
                       "vote_tail_zero (voxels with a tract)": round(float((tV == 0)[inside].mean()), 4)},
        "classes_winning (consensus, voxels with a tract)": {
            ("NONE" if c == E.NONE else TRACT_NAMES[c]): int(k) for c, k in
            zip(*np.unique(cons[any_tract], return_counts=True)) if k >= 0.01 * any_tract.sum()}}
    np.savez(d / "fields.npz", s_ranks=code.ranks, s_support=code.support, s_tail=code.tail, s_votes=vp, s_vote_tail=vtail,
             v_ranks=codeV.ranks, v_support=codeV.support, v_tail=codeV.tail if codeV.tail is not None else np.zeros(0),
             v_votes=vV, v_vote_tail=tV, present=S.present)
    (d / "fields_meta.json").write_text(json.dumps({"streamline": code.meta, "voxel": codeV.meta, "voxel_axes": "(Z, Y, X) of the DWI's (X, Y, Z)",
                                                    "runs": V.runs, "tau": V.tau}, indent=1))
    Path(str(OUT).format(args.b0_noise)).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("streamlines", "streamline_rank_field", "voxels")}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "analyze"))
    ap.add_argument("--subject", default="PAT16")
    ap.add_argument("--runs", type=int, default=16)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--build", type=int, default=10)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--b0-noise", default="thermal", choices=("thermal", "residual"),
                    help="the b0s' bootstrap residuals: scaled to the DW volumes' noise (default), or as they are (motion included)")
    args = ap.parse_args()
    run(args) if args.what == "run" else analyze(args)
