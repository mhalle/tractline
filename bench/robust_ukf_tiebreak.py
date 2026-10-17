"""A tie-break for near-equal tensors, on step 2's diagnostic sample (docs/robustness-plan.md, step 2b; NOTES 2026-10-17,
"Robustness step 2").

    uv run --group dipy --group rankfield bench/robust_ukf_tiebreak.py run [--variants base ..] [--runs 1 2 .. 8]
    uv run --group dipy --group rankfield bench/robust_ukf_tiebreak.py analyze [--variants ..] [--out <results>.json]

Each step the tracker may swap its two tensors (sw: the second is closer to the previous direction; sw2: within 20
degrees and the first's FA <= 0.2), and a swap permutes the state and its covariance, which changes the next step's
sigma points. In epsilon's pairs that part with no >= 5-degree event, the switch flags differ beyond 3 mm in 60 % of
pairs against 30 % of pairs that stay together, at steps where A's tensors are a median 0.02 degrees apart: rounding
may break a near tie either way. Each variant is ukf.track's opt-in tie_angle (degrees; 0 today): sw does not fire when
the two directions are within it as axes, and sw2 is the binary's. (The first rule, sw2 suppressed too, is superseded;
its records and results carry the suffix _sw-and-sw2.) On Metal, as the records:
  - A's whole tractography (A's corrected DWI, its own mask), stored compact (0.05 mm): seeds, fibers, lengths, and
    labeled here (RapidParc, draw 0, index grouping) for the tract mix and the focus tracts' volumes;
  - A's sampled seeds (robust_ukf_diagnose's, runs --runs of both inputs) and each perturbed run's (epsilon,
    fieldonly; rebuilt as robust_ukf_sweep.py does) tracked with every step observed: positions, both switch flags,
    the cosine between the two tensors after the swaps and both tensors' FA (A_fa.npz holds the FA for an A_obs.npz
    written before they were recorded: the same steps, observed again).
Records in DATA/ensemble/cIIIs04/robustness/ukf/sweep_tiebreak/<variant>/ (never overwritten; a file that exists is
skipped; stops below MIN_FREE_GB free).
analyze, per variant (robust_ukf_sweep.py's weights: each run's strata back to its population):
  - robustness: the share of pairs that diverge (MDF > 5 mm, or one side's fiber under 10 points and not the other's),
    the median MDF, the diverging share in the "> 5 mm" and "<= 0.1 mm" strata, and the switch flags differing
    beyond FLAG_FROM_MM (either half, up to where the halves separate: SEP_MM apart or one stops) in pairs that diverge
    and in pairs that stay together;
  - cost and bias on A's whole tractography, against today's (the base variant): streamlines >= 40 mm, their lengths,
    today's >= 40 mm seeds lost, paths moved more than 5 mm (change, not bias), the tract mix (total variation), the
    Other share and the focus tracts' volumes. Rs 0.1 and step_length 0.6 from robust_ukf_sweep's records the same way,
    for scale. And how often the rule could act: the share of A's sampled steps whose tensors lie within each angle,
    and of those, the share that straddle FA 0.2 (one tensor's FA above, the other's at or below: there sw2 follows
    the higher-FA tensor whatever sw did, so a tie there is not a coin flip of the order).
The base variant must match robust_ukf_sweep's base records bit for bit (the observed path tracks as the stream does).
Writes results/robust_ukf_tiebreak_sweep.json.
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, torch
from tractline import pipeline as P
from tractline import ensemble as E
from tractline import susceptibility as S
from tractline import ukf as U
from tractline import compact as C
from tractline.labelers.base import TRACT_NAMES, OTHER, lengths, MIN_LENGTH_MM
from tractline.prep import prepare

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS                                          # noqa: E402
from ensemble_stages import PAIR, EPS, FOCUS, where as stages              # noqa: E402
from robust_ukf_diagnose import (OUT, DEVICE, MIN_FREE_GB, STRATA, DIVERGED_MM, SEP_MM, FLAG_FROM_MM, STEP_MM, STOP, SW,  # noqa: E402
                                 SW2, free_gb)
from robust_ukf_sweep import SWEEP, INPUTS, tracked, raw, unraw, diag, wmedian   # noqa: E402

TIE = OUT / "sweep_tiebreak"
ANGLES_TRIED = (0.1, 0.5, 2.0, 5.0)
#: the sw-only rule (run and analyzed by default), and the first rule, sw2 held at a tie too (tie_sw2; its records were
#: written before tie_sw2 existed and carry the suffix; results/robust_ukf_tiebreak_sweep_sw-and-sw2.json)
VARIANTS = {"base": {}, **{f"tie_angle={a:g}": {"tie_angle": a} for a in ANGLES_TRIED},
            **{f"tie_angle={a:g}_sw-and-sw2": {"tie_angle": a, "tie_sw2": True} for a in ANGLES_TRIED}}
DEFAULT = ["base"] + [f"tie_angle={a:g}" for a in ANGLES_TRIED]
FOR_SCALE = ("Rs=0.1", "step_length=0.6")                                  # robust_ukf_sweep's records, labeled here too
ANGLES = (0.1, 0.5, 2.0, 5.0, 20.0)


def save(path, **payload):
    if free_gb(TIE) < MIN_FREE_GB:
        raise SystemExit(f"stopping: {free_gb(TIE):.1f} GB free, below {MIN_FREE_GB} GB")
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **payload)
    if path.exists():
        tmp.unlink()
        raise SystemExit(f"{path} appeared meanwhile; not overwritten")
    os.replace(tmp, path)


def observed(D, keys, params):
    """The seeds with these keys (those in D's seed list), on Metal with every step observed: keys (D's seed order),
    per step of every half-fiber (2 j forward, 2 j + 1 backward) its half, step, position (k, j, i), flags (stop, sw,
    sw2), |cos| between the two tensors after the swaps and their FA; the fibers (empty when under 10 points)."""
    pts = U.seeds(D)[0]
    allk = E.seed_keys(np.rint(pts.cpu().numpy() - U.SRAND0_OFFSET).astype(np.int64))
    pos = {int(k): i for i, k in enumerate(allk)}
    sel = np.array(sorted(pos[int(k)] for k in keys if int(k) in pos), np.int64)
    rows = []

    def observe(step, index, x, state, direction, stop, info):
        fl = stop.int() * STOP | info["swap"].int() * SW | info["swap2"].int() * SW2
        m1, m2 = state[:, 0:3], state[:, 5:8]
        c = (m1 * m2).sum(1).abs() / (m1.norm(dim=1) * m2.norm(dim=1))
        fa = [U.l2fa(*(state[:, o + i].clamp_min(U.LMIN) for i in (3, 4, 4)))[:, None] for o in (0, 5)]
        rows.append((step, index.cpu().numpy(), torch.cat([x, fl.float()[:, None], c[:, None], *fa], 1).cpu().numpy()))

    fibers, st = U.track(D, backend="metal", batch=P.GPU_BATCH, select=sel, observe=observe, **params)
    half = np.concatenate([i for _, i, _ in rows]).astype(np.int32)
    step = np.concatenate([np.full(len(i), s, np.int16) for s, i, _ in rows])
    V = np.concatenate([v for _, _, v in rows])
    o = np.lexsort((step, half))
    fib = dict(zip(st["seed_index"], fibers))
    F = [np.asarray(fib.get(int(k), np.zeros((0, 3))), np.float32) for k in sel]
    return {"keys": allk[sel], "half": half[o], "step": step[o], "pos": V[o, :3], "flags": V[o, 3].astype(np.uint8),
            "cos": V[o, 4].astype(np.float32), "fa1": V[o, 5].astype(np.float32), "fa2": V[o, 6].astype(np.float32), **raw(F)}


def run(args):
    from _masivar import load
    subject, (ses, scan), _, shells, shell = PAIRS[PAIR]
    _, sd = stages()
    TIE.mkdir(parents=True, exist_ok=True)
    log_path = TIE / "runs.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    A = load(subject, ses, scan, shells)
    apply = lambda dwi, h: S.apply(dwi, h, A.pe_axis, A.pe_sign, A.readout_s)
    dA = apply(A.dwi, np.load(sd / "A_field_hz.npy").astype(np.float64))
    tA = prepare(dA, A.affine, A.bval, A.bvec, shell=shell, device=DEVICE)
    assert np.array_equal(np.load(OUT / "A_mask.npy"), tA.mask)
    DA = U.from_arrays(tA.dwi, tA.header, tA.mask)
    sample = np.unique(np.concatenate([diag(i, r)[0] for i in INPUTS for r in args.runs]))
    for v in args.variants:
        (TIE / v).mkdir(exist_ok=True)
        path = TIE / v / "A.npz"
        if not path.exists():
            t = time.time()
            keys, fibers, seeds = tracked(DA, VARIANTS[v])
            save(path, keys=keys, seeds=seeds, voxel=DA["voxel"].cpu().numpy(), **C.encode_geometry(fibers))
            log[f"{v}/A"] = {"seeds": seeds, "fibers": len(fibers), "seconds": round(time.time() - t, 1)}
            log_path.write_text(json.dumps(log, indent=1))
            print(v, "A", json.dumps(log[f"{v}/A"]), flush=True)
            P.release_memory(DEVICE)
        path = TIE / v / "A_obs.npz"
        if not path.exists():
            t = time.time()
            save(path, **observed(DA, sample, VARIANTS[v]))
            log[f"{v}/A_obs"] = {"seeds": int(len(sample)), "seconds": round(time.time() - t, 1), "MB": round(path.stat().st_size / 1e6, 1)}
            log_path.write_text(json.dumps(log, indent=1))
            print(v, "A_obs", json.dumps(log[f"{v}/A_obs"]), flush=True)
            P.release_memory(DEVICE)
        fa_path = TIE / v / "A_fa.npz"
        with np.load(path) as z:
            old = "fa1" not in z.files
        if old and not fa_path.exists():                                      # an A_obs from before the FA was recorded
            t = time.time()
            o = observed(DA, sample, VARIANTS[v])
            with np.load(path) as z:
                assert all(np.array_equal(z[k], o[k]) for k in ("keys", "half", "step", "pos", "flags")), "not the same steps"
            save(fa_path, fa1=o["fa1"], fa2=o["fa2"])
            log[f"{v}/A_fa"] = {"seconds": round(time.time() - t, 1)}
            log_path.write_text(json.dumps(log, indent=1))
            print(v, "A_fa", json.dumps(log[f"{v}/A_fa"]), flush=True)
            P.release_memory(DEVICE)
    for inp in INPUTS:
        for r in args.runs:
            todo = [v for v in args.variants if not (TIE / v / f"{inp}_r{r:03d}.npz").exists()]
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
                path = TIE / v / f"{inp}_r{r:03d}.npz"
                save(path, **observed(DP, keys, VARIANTS[v]))
                log[f"{v}/{inp}_r{r:03d}"] = {"input_s": t_in, "seconds": round(time.time() - t, 1), "MB": round(path.stat().st_size / 1e6, 2)}
                log_path.write_text(json.dumps(log, indent=1))
                print(v, inp, f"r{r:03d}", json.dumps(log[f"{v}/{inp}_r{r:03d}"]), flush=True)
            del DP, tP
            P.release_memory(DEVICE)


# ----------------------------------------------------------------------------------------------- analysis

class Obs:
    """One observed tractography: per seed key its fiber and each half's steps."""
    def __init__(self, z):
        self.keys, self.half, self.step = z["keys"], z["half"], z["step"]
        self.pos, self.flags, self.cos = z["pos"], z["flags"], z["cos"]
        self.row = {int(k): i for i, k in enumerate(self.keys)}
        self.fib = dict(zip(self.keys.tolist(), unraw(z)))
        self.off = np.searchsorted(self.half, np.arange(2 * len(self.keys) + 1))

    def steps(self, k, h):
        j = self.row[int(k)]
        return slice(self.off[2 * j + h], self.off[2 * j + h + 1])


def flags_differ(A, B, k, vox):
    """Whether the switch flags (sw, sw2) differ beyond FLAG_FROM_MM in either half of seed k, up to the step where
    the halves separate (SEP_MM apart, or one stops). A half is paired with the other side's half whose first step lies
    nearer (the seed tensor's e1 can change sign between inputs, which swaps the halves)."""
    first = lambda T, h: T.pos[T.steps(k, h)][0] * vox
    hb = (0, 1) if (np.linalg.norm(first(A, 0) - first(B, 0)) <= np.linalg.norm(first(A, 0) - first(B, 1))) else (1, 0)
    for h in (0, 1):
        sa, sb = A.steps(k, h), B.steps(k, hb[h])
        pa, pb = A.pos[sa] * vox, B.pos[sb] * vox
        K = min(len(pa), len(pb))
        far = np.flatnonzero(np.linalg.norm(pa[:K] - pb[:K], axis=1) > SEP_MM)
        ks = int(far[0]) if len(far) else K - 1
        e = np.arange(K)
        d = ((A.flags[sa][:K] ^ B.flags[sb][:K]) & (SW | SW2)) != 0
        if (d & ((e + 1) * STEP_MM > FLAG_FROM_MM) & (e <= ks)).any():
            return True
    return False


class Mix:
    """RapidParc's tract mix (draw 0, index grouping) and the focus tracts' density presence on A's grid
    (robust_ukf_score.Presence's measure), of a whole tractography."""
    def __init__(self, affine, shape):
        from ensemble_grid_check import Grid, sample, RHO
        self.grid, self.sample, self.rho = Grid("dwi", affine, shape), sample, RHO
        self.vox_ml = float(np.prod(self.grid.h)) / 1000
        self.labeler = P.default_labeler(DEVICE)

    def __call__(self, fibers):
        with P.exact_float32(DEVICE):
            lab = self.labeler(fibers, draws=(0,))
        kept = [f for f, k in zip(fibers, lab.keep) if k]
        Pt, Ot = self.sample(kept)
        t = np.asarray(lab.tract)[Ot]
        vol = {}
        for nm in FOCUS:
            sel = t == TRACT_NAMES.index(nm)
            vol[nm] = float((self.grid.density(self.grid.counts(Pt[sel], Ot[sel])).ravel() >= self.rho).sum()) * self.vox_ml
        return np.bincount(lab.tract, minlength=len(TRACT_NAMES)) / len(lab.tract), vol


def analyze(args):
    from _masivar import load
    t0 = time.time()
    subject, (ses, scan), _, shells, _ = PAIRS[PAIR]
    A0 = load(subject, ses, scan, shells)
    mix = Mix(A0.affine, A0.dwi.shape[:3])
    res = {"pair": PAIR, "diverged mm": DIVERGED_MM, "separated mm": SEP_MM, "flags from mm": FLAG_FROM_MM, "settings": {v: VARIANTS[v] for v in args.variants},
           "variants": {}, "for scale (robust_ukf_sweep's records)": {}}
    base = None

    def whole(z):
        fibers = C.decode_geometry(z)
        _, _, length = lengths(fibers)
        long_ = length >= MIN_LENGTH_MM
        return z["keys"][long_], E.resample([f for f, l in zip(fibers, long_) if l]), length[long_], fibers

    def bias(z, row):
        lk, paths, length, fibers = whole(z)
        m, vol = mix(fibers)
        row.update({"seeds": int(z["seeds"]), "fibers (>= 10 points)": len(fibers), "streamlines >= 40 mm": int(len(lk)),
                    "length mm of those (quartiles)": [round(float(x), 1) for x in np.quantile(length, [0.25, 0.5, 0.75])],
                    "Other share (RapidParc, draw 0)": round(float(m[OTHER]), 4),
                    "focus volume mL": {k: round(v, 1) for k, v in vol.items()}})
        if base is not None:
            bk, bp, bm, bvol = base
            pos = {int(k): i for i, k in enumerate(lk.tolist())}
            both = [(i, pos[int(k)]) for i, k in enumerate(bk.tolist()) if int(k) in pos]
            d = E.mdf(bp[[a for a, _ in both]], paths[[b for _, b in both]])
            row.update({"against today's A: share moved more than 5 mm (change, not bias)": round(float((d > DIVERGED_MM).mean()), 4),
                        "against today's A: median MDF mm": round(float(np.median(d)), 3),
                        "today's >= 40 mm seeds lost": round(1 - len(both) / len(bk), 4),
                        "streamlines >= 40 mm against today's": round(len(lk) / len(bk), 4),
                        "tract mix: total variation against today's": round(0.5 * float(np.abs(m - bm).sum()), 4),
                        "focus volume against today's": {k: round(vol[k] / bvol[k], 3) for k in vol}})
        return lk, paths, m, vol

    for v in [v for v in args.variants if (TIE / v / "A.npz").exists()]:
        z = np.load(TIE / v / "A.npz")
        vox = z["voxel"]
        row = {"cost and bias (A's whole tractography)": {}}
        out = bias(z, row["cost and bias (A's whole tractography)"])
        if base is None:
            assert v == "base", "the base variant first"
            base = out
            zs = np.load(SWEEP / "base" / "A.npz")                         # today's, as robust_ukf_sweep tracked it
            assert np.array_equal(zs["keys"], z["keys"]) and all(np.array_equal(a, b) for a, b in zip(C.decode_geometry(zs), C.decode_geometry(z)))
        zo = np.load(TIE / v / "A_obs.npz")
        Ao = Obs(zo)
        ang = np.degrees(np.arccos(np.clip(Ao.cos.astype(np.float64), 0, 1)))
        row["A's sampled steps: share with the two tensors within (deg)"] = {f"{a:g}": round(float((ang < a).mean()), 4) for a in ANGLES}
        zf = zo if "fa1" in zo.files else np.load(TIE / v / "A_fa.npz")
        straddle = (zf["fa1"] > 0.2) != (zf["fa2"] > 0.2)
        row["A's sampled steps within (deg): share straddling FA 0.2 (one tensor's FA above, the other's at or below)"] = {
            f"{a:g}": round(float(straddle[ang < a].mean()), 4) for a in ANGLES}
        if v == "base":                                                     # the observed path against robust_ukf_sweep's base
            zs = np.load(SWEEP / "base" / "A.npz")
            ref = dict(zip(zs["sample_keys"].tolist(), unraw(zs)))
            row["reproduction: A's sampled fibers equal robust_ukf_sweep's base"] = all(
                np.array_equal(ref[k], Ao.fib[k]) for k in ref if k in Ao.fib) and set(ref) <= set(Ao.fib)
        for inp in INPUTS:
            files = sorted((TIE / v).glob(f"{inp}_r???.npz"))
            if not files:
                continue
            mdf, div, fl, w, strat, left_out, same = [], [], [], [], [], 0, True
            for f in files:
                r = int(f.stem[-3:])
                dk, ds, pop = diag(inp, r)
                n_s = np.bincount(ds, minlength=len(STRATA))
                Po = Obs(np.load(f))
                if v == "base":
                    zs = np.load(SWEEP / "base" / f.name)
                    same &= bool(np.array_equal(zs["keys"], Po.keys[[len(Po.fib[k]) > 0 for k in Po.keys.tolist()]]) and all(
                        np.array_equal(a, Po.fib[k]) for k, a in zip(zs["keys"].tolist(), unraw(zs))))
                for k, s in zip(dk.tolist(), ds.tolist()):
                    a, b = Ao.fib.get(k), Po.fib.get(k)
                    a = a if a is not None and len(a) else None
                    b = b if b is not None and len(b) else None
                    if a is None and b is None:
                        left_out += 1
                        continue
                    if a is not None and b is not None:
                        m = float(E.mdf(E.resample([a]), E.resample([b]))[0])
                        mdf.append(m); div.append(m > DIVERGED_MM)
                    else:
                        mdf.append(np.inf); div.append(True)
                    fl.append(k in Ao.row and k in Po.row and flags_differ(Ao, Po, k, vox))
                    w.append(pop[s] / n_s[s]); strat.append(s)
            mdf, div, fl, w, strat = (np.array(x) for x in (mdf, div, fl, w, strat))
            wm = lambda sel: round(float((w * fl)[sel].sum() / w[sel].sum()), 4) if sel.any() else None
            row[inp] = {"runs": len(files), "pairs": len(div), "neither side tracked (not seeded, or under 10 points)": left_out,
                        "diverged (weighted)": round(float((w * div).sum() / w.sum()), 4),
                        "median MDF mm (weighted)": round(wmedian(mdf, w), 4),
                        "diverged in the '> 5 mm' stratum": round(float(div[strat == 0].mean()), 4),
                        "diverged in the '<= 0.1 mm' stratum": round(float(div[strat == 2].mean()), 4),
                        f"switch flags differ beyond {FLAG_FROM_MM:g} mm (either half, to the separation): diverged / together (weighted)":
                            [wm(div), wm(~div)],
                        "the same, unweighted": [round(float(fl[div].mean()), 4), round(float(fl[~div].mean()), 4)]}
            if v == "base":
                row[inp]["reproduction: fibers equal robust_ukf_sweep's base"] = same
        res["variants"][v] = row
        print(v, json.dumps(row), flush=True)
    for v in FOR_SCALE:
        row = {}
        bias(np.load(SWEEP / v / "A.npz"), row)
        res["for scale (robust_ukf_sweep's records)"][v] = row
        print(v, json.dumps(row), flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "analyze"))
    ap.add_argument("--variants", nargs="+", default=DEFAULT, choices=list(VARIANTS))
    ap.add_argument("--out", default=str(HERE / "results/robust_ukf_tiebreak_sweep.json"), help="analyze: the results file")
    ap.add_argument("--runs", type=int, nargs="+", default=list(range(1, 9)))
    args = ap.parse_args()
    {"run": run, "analyze": analyze}[args.what](args)
