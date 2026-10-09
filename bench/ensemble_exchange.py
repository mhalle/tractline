"""Why is the reference closer to a replicate than two replicates are to each other? (NOTES 2026-10-08)

    uv run --group dipy --group rankfield bench/ensemble_exchange.py run [--subject PAT16] [--runs 8]
    uv run --group dipy --group rankfield bench/ensemble_exchange.py center [--subject PAT16]
    uv run --group dipy --group rankfield bench/ensemble_exchange.py analyze [--subject PAT16] [--runs 8]

In ensemble_pilot's "added" bootstrap a held-out replicate changes 33 % of the reference's labels, two replicates
38 % of each other's. Two candidates:
- noise level: the DW residuals' leverage correction and the b0s' added noise put a replicate farther from another
  replicate than from the scan (squared signal distance 1.31x in the DW volumes, 2x in the b0s);
- whitening: per-voxel sign flips break up residuals that are spatially structured in the scan.

center: two runs of the scan's DW fit alone (the raw-residual fit, b0s as acquired): "fit" (no residual) and "anti"
(fit - residual: every sign flipped, the replicate farthest from the scan). A least-squares residual is orthogonal to
the fit's space in every voxel; if the pipeline reads the signal through a similar model, both track like the scan.

run: replicates 1..runs of each variant below (ShellWildBootstrap options), as ensemble_pilot runs them, records in
DATA/ensemble/<subject>/exchange/<variant>/; the reference is ensemble_pilot's added/r000 (the scan as acquired, draw
0). analyze: for each variant and for ensemble_pilot's added r001..runs, the reference against each replicate and
every pair of replicates: labels changed, path distance (MDF), and the signal distance in the brain. Writes
results/ensemble_exchange.json.
"""
import argparse, itertools, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tractline import pipeline as P
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT, OTHER

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_pilot import DEVICE, _winners_only  # noqa: E402

OUT = HERE / "results/ensemble_exchange.json"
VARIANTS = {
    "member": dict(b0_noise="flipped", leverage=False),               # the scan is one of the replicates
    "lev0": dict(b0_noise="added", leverage=False),                   # DW symmetric, b0s still 2x
    "member-block": dict(b0_noise="flipped", leverage=False, flip="volume"),   # member, spatial structure kept
}


def bootstrap(s, mask, variant):
    from _bootstrap import ShellWildBootstrap
    kw = dict(b0_noise="added") if variant == "added" else VARIANTS[variant]
    return ShellWildBootstrap(s.dwi, s.bval, s.bvec, extra_b0s=s.b0s[..., len(s.b0_index):], mask=mask, **kw)


def reference_mask(s):
    from tractline.prep import prepare
    with P.exact_float32(DEVICE):
        return prepare(P.correct(s, P.Timer(), DEVICE).dwi, s.affine, s.bval, s.bvec, device=DEVICE).mask


def run(args):
    from _ds001226 import load
    s = load(args.subject)
    labeler = P.default_labeler(DEVICE)
    grid = (s.affine, s.dwi.shape[:3])
    mask = reference_mask(s)
    for v in VARIANTS:
        d = DATA / "ensemble" / args.subject / "exchange" / v
        d.mkdir(parents=True, exist_ok=True)
        log_path = d / "runs.json"
        log = json.loads(log_path.read_text()) if log_path.exists() else {}
        wb = bootstrap(s, mask, v)
        log["bootstrap"] = wb.noise_check(mask)
        for r in range(1, args.runs + 1):
            name = f"r{r:03d}"
            if (d / f"{name}.npz").exists():
                continue
            dwi, pa = wb.replicate(r)
            subj = replace(s, dwi=dwi, b0s=np.concatenate([dwi[..., s.b0_index].astype(np.float64), pa], -1))
            timer, t0 = P.Timer(), time.time()
            with P.exact_float32(DEVICE):
                corr = P.correct(subj, timer, DEVICE)
                tg = P.track(subj, corr.dwi, timer, device=DEVICE)
                with timer("label"):
                    labels = labeler(tg.fibers, draws=(r,), logp=True)
            E.RunRecord.of(tg, labels, *grid).save(d / f"{name}.npz")
            log[name] = {"draw": r, "fibers": len(tg.fibers), "labeled": int(labels.keep.sum()),
                         "other": round(float((labels.tract == OTHER).mean()), 4), "residual_left": corr.residual_left,
                         "field_hz_p99_abs": round(float(np.quantile(np.abs(corr.field_hz), 0.99)), 2),
                         "total_s": round(time.time() - t0, 1)}
            log_path.write_text(json.dumps(log, indent=1))
            print(v, name, json.dumps(log[name]), flush=True)
            P.release_memory(DEVICE)


def center(args):
    from _ds001226 import load
    s = load(args.subject)
    labeler = P.default_labeler(DEVICE)
    grid = (s.affine, s.dwi.shape[:3])
    wb = bootstrap(s, reference_mask(s), "lev0")
    d = DATA / "ensemble" / args.subject / "exchange" / "center"
    d.mkdir(parents=True, exist_ok=True)
    log_path = d / "runs.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    dw = np.setdiff1d(np.arange(s.dwi.shape[-1]), wb.b0)
    S = np.asarray(s.dwi, np.float32).reshape(wb.fit.shape)
    info = np.iinfo(s.dwi.dtype)
    for name, draw, sign in (("fit", 101, 0), ("anti", 102, -1)):
        if (d / f"{name}.npz").exists():
            continue
        x = S.copy()
        x[:, dw] = np.clip(np.rint(wb.fit[:, dw] + sign * wb.res[:, dw]), 0, info.max)
        subj = replace(s, dwi=x.astype(s.dwi.dtype).reshape(s.dwi.shape))
        timer = P.Timer()
        with P.exact_float32(DEVICE):
            corr = P.correct(subj, timer, DEVICE)
            tg = P.track(subj, corr.dwi, timer, device=DEVICE)
            labels = labeler(tg.fibers, draws=(draw,), logp=True)
        E.RunRecord.of(tg, labels, *grid).save(d / f"{name}.npz")
        log[name] = {"draw": draw, "labeled": int(labels.keep.sum()), "residual_left": corr.residual_left,
                     "field_hz_p99_abs": round(float(np.quantile(np.abs(corr.field_hz), 0.99)), 2)}
        log_path.write_text(json.dumps(log, indent=1))
        print(name, json.dumps(log[name]), flush=True)
        P.release_memory(DEVICE)


def compare(base, other):
    """Labels of base's streamlines that other changes (where other labels the seed), and their MDF (mm)."""
    S = E.StreamlineEnsemble(base)
    S.add(other)
    w, dist = S.winners[0], S.distances()[0]
    ok = w >= 0
    return float((LUT[w[ok]] != LUT[base.cluster[ok]]).mean()), float(np.nanmedian(dist))


def signal(s, mask, variant, runs):
    """Mean squared difference in the mask, DW volumes and b0s (both series): scan vs replicates 1, 2; 1 vs 2."""
    wb = bootstrap(s, mask, variant)
    m = np.asarray(mask).reshape(-1) > 0
    X = np.concatenate([s.dwi, s.b0s[..., len(s.b0_index):]], -1).reshape(-1, s.dwi.shape[-1] + s.b0s.shape[-1] - len(s.b0_index))[m]
    R = [np.concatenate(wb.replicate(r), -1).reshape(len(m), -1)[m] for r in (1, 2)]
    dw = np.setdiff1d(np.arange(s.dwi.shape[-1]), wb.b0)
    b0 = np.r_[wb.b0, np.arange(s.dwi.shape[-1], X.shape[1])]
    out = {}
    for name, idx in (("dw", dw), ("b0", b0)):
        sq = lambda a, b: float(((a[:, idx].astype(np.float64) - b[:, idx]) ** 2).mean())  # noqa: E731
        out[name] = {"scan-replicate": round((sq(X, R[0]) + sq(X, R[1])) / 2, 1), "replicate-replicate": round(sq(R[0], R[1]), 1)}
    return out


def analyze(args):
    from _ds001226 import load
    s = load(args.subject)
    mask = reference_mask(s)
    root = DATA / "ensemble" / args.subject
    ref = _winners_only(E.RunRecord.load(root / "added" / "r000.npz", logp=False))
    res = {"subject": args.subject, "runs": args.runs, "variants": {"added": {"b0_noise": "added"}, **VARIANTS}, "results": {}}
    for v in ["added", *VARIANTS]:
        d = root / "added" if v == "added" else root / "exchange" / v
        reps = [_winners_only(E.RunRecord.load(d / f"r{r:03d}.npz", logp=False)) for r in range(1, args.runs + 1)]
        rr = [compare(ref, x) for x in reps]
        pp = [compare(a, b) for a, b in itertools.permutations(reps, 2)]
        log = json.loads((d / "runs.json").read_text())
        p99 = [log[f"r{r:03d}"]["field_hz_p99_abs"] for r in range(1, args.runs + 1)]
        rl = [log[f"r{r:03d}"]["residual_left"] for r in range(1, args.runs + 1)]
        res["results"][v] = {
            "labels changed: reference vs replicate": round(float(np.mean([c for c, _ in rr])), 4),
            "labels changed: replicate vs replicate": round(float(np.mean([c for c, _ in pp])), 4),
            "median mdf mm: reference vs replicate": round(float(np.mean([m for _, m in rr])), 2),
            "median mdf mm: replicate vs replicate": round(float(np.mean([m for _, m in pp])), 2),
            "field p99 |Hz| (min, mean, max)": [min(p99), round(float(np.mean(p99)), 2), max(p99)],
            "residual_left (min, max)": [min(rl), max(rl)],
            "signal squared difference in the mask": signal(s, mask, v, args.runs)}
        print(v, json.dumps(res["results"][v]), flush=True)
    c = root / "exchange" / "center"
    if (c / "fit.npz").exists():
        fit, anti = (_winners_only(E.RunRecord.load(c / f"{n}.npz", logp=False)) for n in ("fit", "anti"))
        reps = [_winners_only(E.RunRecord.load(root / "exchange" / "lev0" / f"r{r:03d}.npz", logp=False))
                for r in range(1, args.runs + 1)]
        row = lambda pairs: [round(float(np.mean([x[0] for x in pairs])), 4), round(float(np.mean([x[1] for x in pairs])), 2)]  # noqa: E731
        res["center"] = {"(labels changed, median mdf mm); replicates are lev0's": {
            "reference vs fit": row([compare(ref, fit)]), "reference vs anti": row([compare(ref, anti)]),
            "fit vs anti": row([compare(fit, anti)]),
            "fit vs replicate": row([compare(fit, x) for x in reps]),
            "anti vs replicate": row([compare(anti, x) for x in reps]),
            "reference vs replicate": row([compare(ref, x) for x in reps])}}
        print("center", json.dumps(res["center"]), flush=True)
    ref_log = json.loads((root / "added" / "runs.json").read_text())["r000"]
    res["reference"] = {"field p99 |Hz|": ref_log["field_hz_p99_abs"], "residual_left": ref_log["residual_left"]}
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "center", "analyze"))
    ap.add_argument("--subject", default="PAT16")
    ap.add_argument("--runs", type=int, default=8)
    args = ap.parse_args()
    {"run": run, "center": center, "analyze": analyze}[args.what](args)
