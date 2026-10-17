"""Which stage the ensemble's spread comes from: cIIIs04 A's ensemble rerun with stages held fixed (NOTES 2026-10-17,
"Where the ensemble's spread comes from").

    uv run --group dipy --group rankfield bench/ensemble_stages.py run [--arm frozen|fieldonly|draw|epsilon|check] [--runs 16]
    uv run --group dipy --group rankfield bench/ensemble_stages.py analyze

A full run resamples every stage at once (bench/ensemble_retest.py, smooth signs, FWHM 4): a replicate of the scan,
the field re-estimated from its b0s and applied to it, tracked, labeled with draw 0. The tracker is deterministic
given its input. The labeler is not, in effect, even at a fixed draw: RapidParc shuffles streamlines into context
groups by their index (labelers/rapidparc.py, batches), so a run whose seed set differs from another's by a few
streamlines labels every streamline in new company - about a draw's worth of change (NOTES 2026-10-17). Arms, each A
plus `runs` replicates:
  full       the existing records: A.npz and smooth/r001..r016;
  frozen     replicate r's DWI (seeded by r, as full run r: the same DWI noise) corrected with A's own field
             (estimated once, from A as acquired): the noise the tracker sees, with the field fixed;
  fieldonly  the field estimated from replicate r's b0s (both series'; seeded by r, the same b0 noise as full run r),
             applied to A's DWI as acquired: the field's noise alone;
  draw       A's corrected DWI tracked once (the fibers are A's), labeled with draws 1..runs: the labeler alone;
  epsilon    A's corrected DWI (A's field) times 1 + EPS u, u uniform in [-1, 1] per value, seeded by 10,000 + r: every
             value moved by up to EPS (1e-6: about 4-8 float32 units in the last place on average, up to about 17), no
             noise model at all. It also moves the mask and so the seed set (about 82 of A's seeds per run);
  check      full replicate 1 rebuilt here, compared with smooth/r001.npz: the replicates are the full arm's.
Records in DATA/ensemble/cIIIs04/retest/cIIIs04-2.1/stages/<arm>/r001.. (compact), the fields beside them.

analyze: per arm,
  - streamlines, over A's seeds: the share whose tract a run changes (mean, SD and range over runs), split by how far
    the path moved (MDF to A's path: identical, up to 0.1 mm, up to 5 mm, beyond); run against run; the overlap of
    each arm's renames with full run r's, for run r and, as the unpaired baseline, run r - 1; and each arm's 17 runs
    against the three real rescans (tracked on A's grid): the stated share of the mode against how often a rescan
    gives it, and AUROC for the rescan's change from A's tract, over each arm's own seeds and over the seeds common
    to all arms;
  - presence (named tracts; density presence as in ensemble_grid_check, A's grid): each voxel's spread
    U = sum over tracts of p (1 - p), p the share of the arm's 17 runs with the tract present; summed over regions
    (FastSurfer's labels carried to A's grid; bands of A's field strength; 8 mm spheres at the landmark sites), and
    which arm's U is largest voxel by voxel where the full arm's U is at least U_MIN; and where presence flips, A's
    density against the presence threshold;
  - focus tracts (CST, AF, UF): Dice of each run's presence with A's, and the uncertain band (voxels some but not
    all runs give the tract), mL.
Writes results/ensemble_stages.json and stages/spread.npz (the arms' U on A's grid, for a viewer).
"""
import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tractline import pipeline as P
from tractline import ensemble as E
from tractline import susceptibility as S
from tractline.data import DATA
from tractline.labelers.base import LUT, TRACT_NAMES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS, key_change, voxel_agree  # noqa: E402

PAIR, FLIP, FWHM = "cIIIs04-2.1", "smooth", 4.0
DEVICE = "mps"
ARMS = ("full", "frozen", "fieldonly", "draw", "epsilon")
FOCUS = ("CST", "AF", "UF")
U_MIN = 0.5                     # analyze: voxels where the full arm's spread is at least this enter the dominance count
SITE_MM = 8.0
EPS = 1e-6                      # the epsilon arm's relative perturbation
L = np.asarray(LUT)


def where():
    subject = PAIRS[PAIR][0]
    d = DATA / "ensemble" / subject / "retest" / PAIR
    return d, d / "stages"


def run(args):
    from _masivar import load
    from _bootstrap import ShellWildBootstrap
    subject, (ses, scan), _, shells, shell = PAIRS[PAIR]
    d, sd = where()
    out = sd / args.arm
    out.mkdir(parents=True, exist_ok=True)
    log_path = sd / "runs.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    labeler = P.default_labeler(DEVICE)
    A = load(subject, ses, scan, shells)
    mask = np.load(d / "A_mask.npy")
    wb = ShellWildBootstrap(A.dwi, A.bval, A.bvec, extra_b0s=A.b0s[..., len(A.b0_index):], b0_noise="added", mask=mask,
                            flip=FLIP, fwhm=FWHM)
    apply = lambda dwi, h: S.apply(dwi, h, A.pe_axis, A.pe_sign, A.readout_s)

    def replicate(r):
        dwi, apa = wb.replicate(r)
        return replace(A, dwi=dwi, b0s=np.concatenate([dwi[..., A.b0_index].astype(np.float64), apa], -1))

    def finish(name, dwi, timer, t0, draws=(0,), extra=None, tg=None):
        with P.exact_float32(DEVICE):
            tg = tg or P.track(A, dwi, timer, device=DEVICE, shell=shell)
            with timer("label"):
                labels = labeler(tg.fibers, draws=draws, logp=True)
        E.RunRecord.of(tg, labels, A.affine, A.dwi.shape[:3]).save(out / f"{name}.npz", compact=True)
        log[f"{args.arm}/{name}"] = {"fibers": len(tg.fibers), "labeled": int(labels.keep.sum()), **(extra or {}),
                                     "seconds": timer.seconds, "total_s": round(time.time() - t0, 1)}
        log_path.write_text(json.dumps(log, indent=1))
        print(args.arm, name, json.dumps(log[f"{args.arm}/{name}"]), flush=True)
        P.release_memory(DEVICE)
        return tg

    log["A geometry"] = {"readout_s": float(A.readout_s), "pe_axis": int(A.pe_axis), "pe_voxel_mm": float(A.vox[A.pe_axis])}
    hA_path = sd / "A_field_hz.npy"
    if not hA_path.exists():
        with P.exact_float32(DEVICE):
            cA = P.correct(A, P.Timer(), DEVICE)
        np.save(hA_path, cA.field_hz.astype(np.float32))
        P.release_memory(DEVICE)
    hA = np.load(hA_path).astype(np.float64)
    p99 = lambda h: round(float(np.quantile(np.abs(h), 0.99)), 2)

    if args.arm == "check":
        timer, t0 = P.Timer(), time.time()
        with P.exact_float32(DEVICE):
            corr = P.correct(replicate(1), timer, DEVICE)
        finish("r001", corr.dwi, timer, t0, extra={"field_hz_p99_abs": p99(corr.field_hz)})
        a, b = E.RunRecord.load(out / "r001.npz", logp=False), E.RunRecord.load(d / FLIP / "r001.npz", logp=False)
        same_keys = np.array_equal(a.keys, b.keys)
        print(json.dumps({"same seeds": same_keys, "same clusters": same_keys and np.array_equal(a.cluster, b.cluster),
                          "same paths": same_keys and np.array_equal(a.paths, b.paths), "against the stored r001": key_change(b, a)}))
        return
    if args.arm == "epsilon":
        dA = apply(A.dwi, hA)
        for r in range(1, args.runs + 1):
            if (out / f"r{r:03d}.npz").exists():
                continue
            u = np.random.default_rng(10_000 + r).uniform(-1, 1, dA.shape).astype(np.float32)
            finish(f"r{r:03d}", (dA * (1 + np.float32(EPS) * u)).astype(np.float32), P.Timer(), time.time())
        return
    if args.arm == "draw":
        timer, t0 = P.Timer(), time.time()
        with P.exact_float32(DEVICE):
            tg = P.track(A, apply(A.dwi, hA), timer, device=DEVICE, shell=shell)
        for r in range(1, args.runs + 1):
            if not (out / f"r{r:03d}.npz").exists():
                finish(f"r{r:03d}", None, P.Timer(), time.time(), draws=(r,), tg=tg)
        return
    for r in range(1, args.runs + 1):
        name = f"r{r:03d}"
        if (out / f"{name}.npz").exists():
            continue
        timer, t0 = P.Timer(), time.time()
        s = replicate(r)
        if args.arm == "frozen":
            with timer("field_apply"):
                dwi = apply(s.dwi, hA)
            finish(name, dwi, timer, t0)
        else:                                                             # fieldonly
            with P.exact_float32(DEVICE):
                corr = P.correct(s, timer, DEVICE)
            np.save(out / f"{name}_field_hz.npy", corr.field_hz.astype(np.float16))
            with timer("field_apply_A"):
                dwi = apply(A.dwi, corr.field_hz)
            finish(name, dwi, timer, t0, extra={"field_hz_p99_abs": p99(corr.field_hz), "residual_left": corr.residual_left})


# ----------------------------------------------------------------------------------------------- analysis

MDF_BINS = (("identical (MDF 0)", 0.0, 0.0), ("MDF (0, 0.1] mm", 0.0, 0.1), ("MDF (0.1, 5] mm", 0.1, 5.0), ("MDF > 5 mm", 5.0, np.inf))


def arm_records(arm, runs):
    """A and the arm's runs 1..runs, all present: the paired statistics index runs by position"""
    d, sd = where()
    paths = [d / "A.npz"] + [(d / FLIP if arm == "full" else sd / arm) / f"r{r:03d}.npz" for r in range(1, runs + 1)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{arm}: {len(missing)} of {len(paths)} records missing, the first {missing[0]}")
    return paths


def matched(SA, A, rec):
    """Over A's seeds: rec's tract (-1 where rec has no streamline from the seed) and the MDF of its path to A's
    (nan there)"""
    n = len(A.keys)
    ri, qi = SA.match(rec)
    t = np.full(n, -1, np.int16)
    t[ri] = L[rec.cluster[qi]]
    m = np.full(n, np.nan, np.float32)
    m[ri] = E.mdf(A.paths[ri], rec.paths[qi])
    return t, m


def by_distance(T, M, tA):
    """Renames against A pooled over the runs, split by how far the seed's path moved (MDF to A's path)"""
    ok = T >= 0
    ch = (T != tA) & ok
    out = {}
    for name, lo, hi in MDF_BINS:
        b = ok & ((M == 0) if hi == 0 else (M > lo) & (M <= hi))
        out[name] = {"share of streamlines": round(float(b.sum() / ok.sum()), 4),
                     "renamed": round(float(ch[b].sum() / max(b.sum(), 1)), 4),
                     "share of all renames": round(float(ch[b].sum() / max(ch.sum(), 1)), 4)}
    return out


def overlap(F, X):
    """F, X (runs, seeds): -1 unmatched, 0 kept A's tract, else the new tract + 1. Of the seeds F's run renames, the
    share X's run of the same index renames too, and to the same tract; and X's renames among the seeds F's run keeps."""
    both = (F > 0) & (X >= 0)
    kept = (F == 0) & (X >= 0)
    return {"renames too": round(float((X[both] > 0).mean()), 4), "to the same tract": round(float((X[both] == F[both]).mean()), 4),
            "renames seeds full's run keeps": round(float((X[kept] > 0).mean()), 4)}


def auroc(score, event):
    """P(score of a random event > score of a random non-event), ties counted half"""
    from scipy.stats import rankdata
    r = rankdata(score)
    n1 = int(event.sum()); n0 = len(event) - n1
    return float((r[event].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else None


def against_rescans(T, tA, rescans, ok=None):
    """T (R, n): the arm's runs' tracts over A's seeds (row 0 is A). Over the seeds every run labels (or `ok`): the
    mode and its share (the stated P that another run gives it); per rescan, how often it gives the mode (observed),
    and how well 1 - (share of runs giving A's tract) ranks the seeds whose tract the rescan changes from A's (AUROC)."""
    ok = (T >= 0).all(0) if ok is None else ok
    X = T[:, ok]
    counts = np.stack([(X == k).sum(0) for k in range(E.N_TRACTS)])
    mode, share = counts.argmax(0), counts.max(0) / len(T)
    keepA = counts[tA[ok], np.arange(ok.sum())] / len(T)
    out = {"seeds": int(ok.sum()), "stated: mode's mean share": round(float(share.mean()), 4)}
    for tag, rt in rescans.items():
        m = rt[ok] >= 0
        out[tag] = {"gives the mode": round(float((rt[ok][m] == mode[m]).mean()), 4),
                    "AUROC, change from A's tract": round(auroc(1 - keepA[m], rt[ok][m] != tA[ok][m]), 4)}
    return out


def regions(grid_affine, shape, d):
    """Boolean masks on A's grid: FastSurfer's tissue classes (each voxel's label at its center, through the
    T1-to-DWI transform), A's field-strength bands, and spheres at the landmark sites."""
    from ensemble_endpoints import Anatomy, FS
    anat = Anatomy("cIIIs04")
    dwi_to_t1 = np.linalg.inv(np.load(FS / "cIIIs04_t1_to_dwi.npz")["t1_to_dwi"])
    ijk = np.indices(shape).reshape(3, -1).T
    world = ijk @ grid_affine[:3, :3].T + grid_affine[:3, 3]
    lab = anat.raw(world @ dwi_to_t1[:3, :3].T + dwi_to_t1[:3, 3]).reshape(shape)
    cortex = (lab >= 1000) & (lab < 3000)
    out = {
        "cerebral white matter": np.isin(lab, [2, 41]),
        "cortex": cortex,
        "deep gray (thalamus, basal ganglia)": np.isin(lab, [10, 11, 12, 13, 49, 50, 51, 52]),
        "brainstem": lab == 16,
        "cerebellum": np.isin(lab, [7, 8, 46, 47]),
        # DKT: lateral and medial orbitofrontal (12, 14); entorhinal, fusiform, parahippocampal (6, 7, 16)
        "orbitofrontal and ventral temporal cortex": np.isin(lab % 1000, [12, 14, 6, 7, 16]) & cortex,
    }
    _, sd = where()
    h = np.abs(np.load(sd / "A_field_hz.npy"))
    brain = lab > 0
    qs = np.quantile(h[brain], [0.5, 0.9])
    out["field: weakest half (|field| < %.0f Hz)" % qs[0]] = brain & (h < qs[0])
    out["field: 50th-90th percentile"] = brain & (h >= qs[0]) & (h < qs[1])
    out["field: strongest tenth (|field| >= %.0f Hz)" % qs[1]] = brain & (h >= qs[1])
    sites = json.loads((HERE / "results/ensemble_landmarks_smooth.json").read_text())["sites"]
    for s in sites:
        out["site: " + s["name"]] = np.linalg.norm(world - np.asarray(s["mm"]), axis=1).reshape(shape) <= SITE_MM
    return out, brain


def analyze(args):
    from ensemble_grid_check import Grid, sample, RHO
    t0 = time.time()
    d, sd = where()
    A = E.RunRecord.load(d / "A.npz", logp=False)
    affine = np.load(d / "landmarks.npz")["affine"]                       # A's grid (ensemble_landmarks.py wrote it)
    grid = Grid("dwi", affine, A.shape)
    named = [k for k, n in enumerate(TRACT_NAMES) if n != "Other"]
    focus = [TRACT_NAMES.index(n) for n in FOCUS]
    vox_ml = float(np.prod(grid.h)) / 1000
    nvox = int(np.prod(grid.shape))

    def presence(path, keep_density=False):
        """(named tracts, voxels) density presence of one run, and (keep_density) the densities themselves"""
        rec = E.RunRecord.load(path, logp=False, fibers=True)
        Pt, Ot = sample(rec.fibers)
        pt_tract = L[rec.cluster][Ot]
        pres = np.zeros((len(TRACT_NAMES), nvox), bool)
        dens = np.zeros((len(TRACT_NAMES), nvox), np.float32) if keep_density else None
        for k in named:
            sel = pt_tract == k
            if sel.any():
                dk = grid.density(grid.counts(Pt[sel], Ot[sel])).ravel()
                pres[k] = dk >= RHO
                if keep_density:
                    dens[k] = dk
        return pres, dens

    presA, densA = presence(d / "A.npz", keep_density=True)
    DA = densA[named] / RHO                                                 # A's density against the presence threshold
    del densA
    near = lambda sel: round(float(((DA[sel] >= 0.5) & (DA[sel] < 2)).mean()), 4)
    res = {"pair": PAIR, "flip": FLIP, "fwhm": FWHM, "u_min": U_MIN, "site radius mm": SITE_MM, "eps": EPS, "arms": {}}
    spread, focus_rows, changed, Ts = {}, {}, {}, {}
    SA, tA = E.StreamlineEnsemble(A), L[A.cluster]
    rescan_t = {tag: matched(SA, A, E.RunRecord.load(d / f"{tag}_onA.npz", logp=False))[0] for tag in PAIRS[PAIR][2]}
    for arm in args.arms:
        paths = arm_records(arm, args.runs)
        recs = [E.RunRecord.load(p, logp=False) for p in paths]
        R = len(recs)
        TM = [matched(SA, A, r) for r in recs[1:]]
        T = np.stack([tA] + [t for t, _ in TM])
        M = np.stack([m for _, m in TM])
        Ts[arm] = T
        changed[arm] = np.where(T[1:] < 0, -1, np.where(T[1:] == tA, 0, T[1:] + 1)).astype(np.int16)
        per_run = [float((T[r][T[r] >= 0] != tA[T[r] >= 0]).mean()) for r in range(1, R)]
        rr = [key_change(recs[i], recs[i + 1]) for i in range(1, R - 1, 2)]
        vox = [voxel_agree(recs[0], r)["agree (either has a tract)"] for r in recs[1:]]
        res.setdefault("against rescans, each arm's own seeds", {})[arm] = against_rescans(T, tA, rescan_t)
        votes = presA.astype(np.uint8)                                      # run 0 is A in every arm
        dice = {n: [] for n in FOCUS}
        for path in paths[1:]:
            pres, _ = presence(path)
            votes += pres
            for n, k in zip(FOCUS, focus):
                a, b = presA[k], pres[k]
                dice[n].append(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
        p = votes[named] / R
        U = (p * (1 - p)).sum(0).reshape(grid.shape)
        spread[arm] = U
        # where presence flips: A's density there, against RHO (the presence threshold), beside every voxel any run has
        V = votes[named]
        flip, anyrun = (V > 0) & (V < R), V > 0
        focus_rows[arm] = {n: {"Dice with A (mean)": round(float(np.mean(dice[n])), 3),
                               "uncertain band, mL": round(float(((votes[k] > 0) & (votes[k] < R)).sum()) * vox_ml, 1),
                               "any run, mL": round(float((votes[k] > 0).sum()) * vox_ml, 1)} for n, k in zip(FOCUS, focus)}
        res["arms"][arm] = {
            "runs": R, "fibers (mean)": int(np.mean([len(r.keys) for r in recs])),
            "A's seeds unmatched (mean per run)": round(float((T[1:] < 0).sum(1).mean()), 1),
            "A vs run: tract changed (mean, SD, min, max over runs)": [round(float(f(per_run)), 4) for f in (np.mean, np.std, np.min, np.max)],
            "A vs run: median MDF mm (mean over runs)": round(float(np.mean([np.nanmedian(m) for m in M])), 4),
            "A vs run: renames by path distance": by_distance(T[1:], M, tA),
            "run vs run (disjoint pairs): tract changed": round(float(np.mean([x["changed"] for x in rr])), 4),
            "run vs run: median MDF mm": round(float(np.mean([x["median_mdf_mm"] for x in rr])), 4),
            "A vs run: voxel winner agrees": round(float(np.mean(vox)), 4),
            "presence spread U, whole brain (sum)": round(float(U.sum()), 1),
            "presence flips (tract, voxel) with A's density in [0.5, 2) x RHO": near(flip),
            "every (tract, voxel) any run has, the same": near(anyrun),
            "A's density / RHO at flips: quartiles": [round(float(q), 3) for q in np.quantile(DA[flip], [0.25, 0.5, 0.75])] if flip.any() else None}
        print(arm, json.dumps(res["arms"][arm]), f"{time.time() - t0:.0f}s", flush=True)
        del recs, V, flip, anyrun
    res["focus tracts"] = focus_rows
    # the same comparison over one seed set, the seeds every run of every arm labels
    common = np.logical_and.reduce([(T >= 0).all(0) for T in Ts.values()])
    res["against rescans, common seeds"] = {arm: against_rescans(T, tA, rescan_t, common) for arm, T in Ts.items()}
    # paired by run: run r of frozen and fieldonly shares its noise with full run r (the DWI's, the b0s'); epsilon
    # (seeds 10,000 + r) and draw (no data noise) share none, so their rows are unpaired by construction. Unpaired
    # baselines: the arm's run r - 1 (cyclic) against full run r, and full against itself the same way.
    F = changed["full"]
    parts = [a for a in ARMS[1:] if a in changed]
    res["renames of full run r found in another run"] = {"full, run r - 1 (baseline)": overlap(F, np.roll(F, 1, 0))}
    for arm in parts:
        X = changed[arm]
        res["renames of full run r found in another run"][f"{arm}, run r"] = overlap(F, X)
        res["renames of full run r found in another run"][f"{arm}, run r - 1"] = overlap(F, np.roll(X, 1, 0))
    regs, brain = regions(affine, grid.shape, d)
    full = spread["full"]
    res["presence spread by region"] = {}
    for name, m in regs.items():
        row = {"voxels": int(m.sum()), "full U (sum)": round(float(full[m].sum()), 1)}
        for arm in parts:
            row[f"{arm} / full"] = round(float(spread[arm][m].sum() / max(full[m].sum(), 1e-9)), 3)
        hot = m & (full >= U_MIN)
        if hot.any():
            # argmax: a tie goes to the arm listed first (frozen); the share of ties is reported beside it
            S_ = np.stack([spread[a][hot] for a in parts])
            dom = np.argmax(S_, 0)
            row[f"voxels with full U >= {U_MIN}"] = int(hot.sum())
            row["largest there: " + " / ".join(parts)] = [round(float((dom == i).mean()), 3) for i in range(len(parts))]
            row["ties for the largest"] = round(float(((S_ == S_.max(0)).sum(0) > 1).mean()), 4)
        res["presence spread by region"][name] = row
        print(name, json.dumps(row), flush=True)
    res["whole brain"] = {f"{a} / full": round(float(spread[a][brain].sum() / full[brain].sum()), 3) for a in parts}
    fields = sorted((sd / "fieldonly").glob("r???_field_hz.npy"))
    if fields:
        m = brain
        H = np.stack([np.load(f).astype(np.float32) for f in fields])
        sdv = H.std(0)
        hA = np.load(sd / "A_field_hz.npy")
        geo = json.loads((sd / "runs.json").read_text())["A geometry"]
        mm = geo["readout_s"] * geo["pe_voxel_mm"]                        # a field of 1 Hz shifts the image by readout x voxel
        log = json.loads((sd / "runs.json").read_text())
        res["field across fieldonly replicates"] = {
            "runs": len(fields), "SD Hz: median / 90th / 99th in the brain": [round(float(np.quantile(sdv[m], q)), 2) for q in (0.5, 0.9, 0.99)],
            "SD mm (Hz x readout x voxel): median / 90th / 99th": [round(float(np.quantile(sdv[m], q)) * mm, 2) for q in (0.5, 0.9, 0.99)],
            "SD Hz: 99th, strongest tenth of A's field": round(float(np.quantile(sdv[m & (np.abs(hA) >= np.quantile(np.abs(hA[m]), 0.9))], 0.99)), 2),
            "p99 |field| Hz: A / replicates": [round(float(np.quantile(np.abs(hA), 0.99)), 2), [round(float(np.quantile(np.abs(h), 0.99)), 2) for h in H]],
            "residual_left: replicates (min, max)": [min(v["residual_left"] for k, v in log.items() if k.startswith("fieldonly/")),
                                                     max(v["residual_left"] for k, v in log.items() if k.startswith("fieldonly/"))]}
    np.savez_compressed(sd / "spread.npz", affine=affine, **{a: spread[a].astype(np.float32) for a in spread})
    (HERE / "results/ensemble_stages.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["whole brain"]), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "analyze"))
    ap.add_argument("--arm", default="frozen", choices=("frozen", "fieldonly", "draw", "epsilon", "check"))
    ap.add_argument("--runs", type=int, default=16)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS, help="analyze: the arms; full is required")
    args = ap.parse_args()
    {"run": run, "analyze": analyze}[args.what](args)
