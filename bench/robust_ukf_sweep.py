"""Tracker settings that could damp the amplification of small input changes, on step 2's diagnostic sample
(docs/robustness-plan.md, step 2; NOTES 2026-10-17, "Robustness step 2").

    uv run --group dipy --group rankfield bench/robust_ukf_sweep.py run [--variants base ..] [--runs 1 2 3 4 5 6 7 8]
    uv run --group dipy --group rankfield bench/robust_ukf_sweep.py analyze

robust_ukf_diagnose.py found no single hard decision behind divergence: about three quarters of epsilon's diverging
pairs part by drift, the halves' distance growing from the float32 quantum near the seed. Each variant here is one
opt-in setting of ukf.track (Metal, as the records), every other setting the default:
  seeding_threshold  0.1 today: 0.15, 0.2 (fewer low-FA seeds, where both tensors start as one weak tensor);
  Qm, Ql             the process noise of the directions (0.001) and of the eigenvalues (50): lower trusts the model's
                     constancy more;
  Rs                 the measurement noise (0.02): higher trusts each step's signal less;
  step_length        0.3 mm: 0.6 (half the steps; record_length 1.8 mm is still a whole number of steps);
  stopping_fa        0.08: 0.06. Hysteresis is not offered: the Metal kernel decides each step's stop on its own, so a
                     stop that needs several low-FA steps would need a change to the kernel or the tracking loop.
run, per variant:
  - A's whole tractography (A's corrected DWI, its own mask): seeds, fibers and lengths for bias, and the A side of
    every pair; stored compact (0.05 mm), the sampled seeds' fibers also in float32;
  - per input (epsilon, fieldonly) and run r (default r001-r008): the perturbed input's tractography of the seeds
    robust_ukf_diagnose sampled for that run (its records' keys, strata and populations), rebuilt as there.
Records in DATA/ensemble/cIIIs04/robustness/ukf/sweep/<variant>/ (never overwritten; existing files are skipped).
analyze, per variant:
  - robustness: over the sampled pairs, the share that diverge (MDF > 5 mm, or one side's fiber under 10 points and
    the other's not), weighted back to each run's population by stratum (robust_ukf_diagnose's strata, which
    today's tracker defines), the weighted median MDF, and the diverging share in the "> 5 mm" and "<= 0.1 mm"
    strata (fragile seeds steadied, steady seeds made fragile); seeds a variant does not seed are left out and counted;
  - cost and bias, on A's whole tractography: seeds, fibers, streamlines >= 40 mm (the labeled ones), their length
    quartiles, and against today's A (the base variant) the median MDF over seeds both keep at >= 40 mm, and the
    share of today's >= 40 mm seeds the variant loses.
Writes results/robust_ukf_sweep.json.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
from tractline import pipeline as P
from tractline import ensemble as E
from tractline import susceptibility as S
from tractline import ukf as U
from tractline import compact as C
from tractline.labelers.base import lengths, MIN_LENGTH_MM
from tractline.prep import prepare

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS                                          # noqa: E402
from ensemble_stages import PAIR, EPS, where as stages                     # noqa: E402
from robust_ukf_diagnose import OUT, DEVICE, MIN_FREE_GB, STRATA, DIVERGED_MM, free_gb   # noqa: E402

SWEEP = OUT / "sweep"
VARIANTS = {
    "base": {},
    "seeding_threshold=0.15": {"seeding_threshold": 0.15},
    "seeding_threshold=0.2": {"seeding_threshold": 0.2},
    "Qm=0.0003": {"Qm": 0.0003},
    "Qm=0.0001": {"Qm": 0.0001},
    "Ql=10": {"Ql": 10.0},
    "Rs=0.05": {"Rs": 0.05},
    "Rs=0.1": {"Rs": 0.1},
    "step_length=0.6": {"step_length": 0.6},
    "stopping_fa=0.06": {"stopping_fa": 0.06},
}
INPUTS = ("epsilon", "fieldonly")


def tracked(D, params, keys=None):
    """ukf.track on Metal with these settings: (seed keys of the fibers' seeds, fibers by seed in D's seed order, seeds),
    every seed or (keys) only those of D's seeds with these keys. Fibers under 10 points are dropped, as always."""
    th = params.get("seeding_threshold", 0.1)
    sel = None
    if keys is not None:
        pts = U.seeds(D, seeding_threshold=th)[0]
        allk = E.seed_keys(np.rint(pts.cpu().numpy() - U.SRAND0_OFFSET).astype(np.int64))
        want = set(int(k) for k in keys)
        sel = np.flatnonzero([int(k) in want for k in allk])
    fibers, st = U.track(D, backend="metal", batch=P.GPU_BATCH, select=sel, **params)
    return E.seed_keys(st["seed_voxel"]), fibers, int(st["seeds"])


def raw(fibers):
    """Fibers as float32 points and lengths (any may be empty)."""
    return {"flen": np.array([len(f) for f in fibers], np.int32),
            "fib": np.concatenate([np.asarray(f, np.float32).reshape(-1, 3) for f in fibers] + [np.zeros((0, 3), np.float32)])}


def unraw(z, prefix=""):
    flen, fib = z[prefix + "flen"], z[prefix + "fib"]
    ends = np.cumsum(flen)
    return [fib[e - n:e] for n, e in zip(flen, ends)]


def save(path, **payload):
    if free_gb(SWEEP) < MIN_FREE_GB:
        raise SystemExit(f"stopping: {free_gb(SWEEP):.1f} GB free, below {MIN_FREE_GB} GB")
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    if path.exists():
        tmp.unlink()
        raise SystemExit(f"{path} appeared meanwhile; not overwritten")
    os.replace(tmp, path)


def diag(inp, r):
    z = np.load(OUT / inp / f"r{r:03d}.npz")
    return z["keys"], z["strata"], z["population"]


def run(args):
    from _masivar import load
    subject, (ses, scan), _, shells, shell = PAIRS[PAIR]
    _, sd = stages()
    SWEEP.mkdir(parents=True, exist_ok=True)
    log_path = SWEEP / "runs.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    A = load(subject, ses, scan, shells)
    apply = lambda dwi, h: S.apply(dwi, h, A.pe_axis, A.pe_sign, A.readout_s)
    dA = apply(A.dwi, np.load(sd / "A_field_hz.npy").astype(np.float64))
    tA = prepare(dA, A.affine, A.bval, A.bvec, shell=shell, device=DEVICE)
    assert np.array_equal(np.load(OUT / "A_mask.npy"), tA.mask)
    DA = U.from_arrays(tA.dwi, tA.header, tA.mask)
    sample = np.unique(np.concatenate([diag(i, r)[0] for i in INPUTS for r in args.runs]))
    for v in args.variants:
        (SWEEP / v).mkdir(exist_ok=True)
        path = SWEEP / v / "A.npz"
        if path.exists():
            continue
        t = time.time()
        keys, fibers, seeds = tracked(DA, VARIANTS[v])
        pos = {int(k): i for i, k in enumerate(keys)}
        sk = np.array([k for k in sample.tolist() if k in pos], np.int64)
        save(path, keys=keys, seeds=seeds, sample_keys=sk, **C.encode_geometry(fibers), **raw([fibers[pos[k]] for k in sk.tolist()]))
        log[f"{v}/A"] = {"seeds": seeds, "fibers": len(fibers), "seconds": round(time.time() - t, 1)}
        log_path.write_text(json.dumps(log, indent=1))
        print(v, "A", json.dumps(log[f"{v}/A"]), flush=True)
        P.release_memory(DEVICE)
    for inp in INPUTS:
        for r in args.runs:
            todo = [v for v in args.variants if not (SWEEP / v / f"{inp}_r{r:03d}.npz").exists()]
            if not todo:
                continue
            t = time.time()
            keys = diag(inp, r)[0]
            if inp == "epsilon":
                u = np.random.default_rng(10_000 + r).uniform(-1, 1, dA.shape).astype(np.float32)
                dP = (dA * (1 + np.float32(EPS) * u)).astype(np.float32)
                del u
            else:
                dP = apply(A.dwi, np.load(sd / "fieldonly" / f"r{r:03d}_field_hz.npy").astype(np.float64))
            tP = prepare(dP, A.affine, A.bval, A.bvec, shell=shell, device=DEVICE)
            del dP
            DP = U.from_arrays(tP.dwi, tP.header, tP.mask)
            t_in = round(time.time() - t, 1)
            for v in todo:
                t = time.time()
                k, fibers, seeds = tracked(DP, VARIANTS[v], keys)
                save(SWEEP / v / f"{inp}_r{r:03d}.npz", keys=k, seeds=seeds, **raw(fibers))
                log[f"{v}/{inp}_r{r:03d}"] = {"fibers": len(fibers), "input_s": t_in, "seconds": round(time.time() - t, 1)}
                log_path.write_text(json.dumps(log, indent=1))
                print(v, inp, f"r{r:03d}", json.dumps(log[f"{v}/{inp}_r{r:03d}"]), flush=True)
            del DP, tP
            P.release_memory(DEVICE)


# ----------------------------------------------------------------------------------------------- analysis

def wmedian(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])


def analyze(args):
    t0 = time.time()
    res = {"pair": PAIR, "diverged mm": DIVERGED_MM, "settings": VARIANTS, "variants": {}}
    base = None
    for v in [v for v in VARIANTS if (SWEEP / v / "A.npz").exists()]:
        z = np.load(SWEEP / v / "A.npz")
        fibers = C.decode_geometry(z)
        _, _, length = lengths(fibers)
        long_ = length >= MIN_LENGTH_MM
        keys = z["keys"]
        Afib = dict(zip(z["sample_keys"].tolist(), unraw(z)))
        paths = E.resample([f for f, l in zip(fibers, long_) if l])
        lk = keys[long_]
        row = {"cost and bias (A's whole tractography)": {
            "seeds": int(z["seeds"]), "fibers (>= 10 points)": len(fibers), "streamlines >= 40 mm": int(long_.sum()),
            "length mm of those (quartiles)": [round(float(x), 1) for x in np.quantile(length[long_], [0.25, 0.5, 0.75])]}}
        if base is None:
            assert v == "base", "the base variant first"
            base = (lk, paths)
        else:
            bk, bp = base
            pos = {int(k): i for i, k in enumerate(lk.tolist())}
            both = [(i, pos[int(k)]) for i, k in enumerate(bk.tolist()) if int(k) in pos]
            m = E.mdf(bp[[a for a, _ in both]], paths[[b for _, b in both]])
            row["cost and bias (A's whole tractography)"].update({
                "against today's A: median MDF mm over seeds both keep at >= 40 mm": round(float(np.median(m)), 3),
                "against today's A: share moved more than 5 mm": round(float((m > DIVERGED_MM).mean()), 4),
                "today's >= 40 mm seeds this variant loses": round(1 - len(both) / len(bk), 4)})
        for inp in INPUTS:
            files = sorted((SWEEP / v).glob(f"{inp}_r???.npz"))
            if not files:
                continue
            mdf, div, w, strat, left_out = [], [], [], [], 0
            for f in files:
                r = int(f.stem[-3:])
                dk, ds, pop = diag(inp, r)
                n_s = np.bincount(ds, minlength=len(STRATA))
                zp = np.load(f)
                Pfib = dict(zip(zp["keys"].tolist(), unraw(zp)))
                for k, s in zip(dk.tolist(), ds.tolist()):
                    a, b = Afib.get(k), Pfib.get(k)
                    if a is None and b is None:
                        left_out += 1
                        continue
                    if a is not None and b is not None and len(a) and len(b):
                        m = float(E.mdf(E.resample([a]), E.resample([b]))[0])
                        mdf.append(m); div.append(m > DIVERGED_MM)
                    else:
                        mdf.append(np.inf); div.append(True)
                    w.append(pop[s] / n_s[s]); strat.append(s)
            mdf, div, w, strat = np.array(mdf), np.array(div), np.array(w), np.array(strat)
            row[inp] = {"runs": len(files), "pairs": len(div), "neither side tracked (not seeded, or under 10 points)": left_out,
                        "diverged (weighted)": round(float((w * div).sum() / w.sum()), 4),
                        "median MDF mm (weighted)": round(wmedian(mdf, w), 4),
                        "diverged in the '> 5 mm' stratum": round(float(div[strat == 0].mean()), 4),
                        "diverged in the '<= 0.1 mm' stratum": round(float(div[strat == 2].mean()), 4)}
        res["variants"][v] = row
        print(v, json.dumps(row), flush=True)
    (HERE / "results/robust_ukf_sweep.json").write_text(json.dumps(res, indent=1))
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "analyze"))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--runs", type=int, nargs="+", default=list(range(1, 9)))
    args = ap.parse_args()
    {"run": run, "analyze": analyze}[args.what](args)
