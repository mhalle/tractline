"""Tracker variants scored on what matters: real rescans, honesty, robustness and bias (docs/robustness-plan.md, step 2;
NOTES 2026-10-17, "Robustness step 2").

    uv run --group dipy --group rankfield bench/robust_ukf_score.py run --variant Rs=0.1 [--only A within smooth ..]
    uv run --group dipy --group rankfield bench/robust_ukf_score.py run --variant seeding_threshold=0.2
    uv run --group dipy --group rankfield bench/robust_ukf_score.py analyze
    uv run --group dipy --group rankfield bench/robust_ukf_score.py reweight     (after analyze)
    uv run --group dipy --group rankfield bench/robust_ukf_score.py paths        (after analyze)

The sweep (robust_ukf_sweep.py) found no setting that damps divergence without changing the tractography; two that
damp it most for their cost, Rs 0.1 and step_length 0.6, are scored here on cIIIs04 (pair cIIIs04-2.1) like today's
tracker. A variant is ukf.track settings passed through pipeline.track's opt-in `tracker=`; every other stage is
today's (the field, the mask, RapidParc at draw 0 with its default "index" grouping).

run --variant <setting>=<value>[,<setting>=<value>]: the runs today's records hold, again with the variant, into
DATA/ensemble/cIIIs04/robustness/ukf/score/<variant>/ (never overwritten; a run whose record exists is skipped; stops
below MIN_FREE_GB free):
  A            scan A corrected and tracked (ensemble_retest.py's A);
  <tag>_onA    each rescan (within, session3, session5) corrected on its own grid, resampled onto A's
               (ensemble_retest.onto_reference, with the stored registration) and tracked there: A's seed keys;
  smooth/r..   16 bootstrap replicates of A (ShellWildBootstrap, smooth signs, FWHM 4, b0 noise "added", A's mask,
               seeded by r), each corrected and tracked: with A, the 17-run ensemble (ensemble_retest's smooth/r001..);
  epsilon/r..  8 runs: A's corrected DWI (A's field) x (1 + 1e-6 u), seeded by 10,000 + r (ensemble_stages' arm);
  fieldonly/r.. 8 runs: the field estimated from replicate r's b0s, applied to A's DWI as acquired (ensemble_stages').
run --variant seeding_threshold=0.2 tracks nothing. A higher seeding threshold only drops seeds: every seed it keeps
has today's path (the sweep: median MDF 0, none moved more than 5 mm). Its seeds are those ukf.seeds accepts at 0.2
on A's corrected DWI (A's field, stages/A_field_hz.npy): mean signal >= 0.2 and the seed tensor's FA > 0.2. Their
keys are written to score/seeding_threshold=0.2/seed_keys.npy, and every record (A, the rescans, the replicates,
epsilon, fieldonly) is restricted to them. The labels stay as stored, labeled in the company of all of today's
streamlines: a pipeline seeding at 0.2 would label in a smaller company (a draw's worth of renames). A rescan or a
replicate seeding at 0.2 on its own data would keep a slightly different set; A's set is used for all.

analyze: per configuration (today from the existing records; seeding_threshold=0.2; each variant with records):
  (a) rescan agreement, A against each rescan on A's grid: the share of A's streamlines the rescan renames over the
      seeds both label (also with RapidParc's hash-sorted grouping, A and the rescan relabeled); presence Dice of the
      focus tracts (density presence as ensemble_grid_check, A's grid); at the eight landmark sites
      (results/ensemble_landmarks_smooth.json), the main named tract among the streamlines passing within SITE_MM,
      the same in A and the rescan;
  (b) honesty: the 17-run ensemble's stated share of the mode against how often each rescan gives it, and AUROC for
      the rescan's change from A's tract (ensemble_stages.against_rescans, over the seeds every run labels);
  (c) robustness: epsilon and fieldonly against A, whole brain: renamed, paths more than 5 mm apart, A's seeds lost;
  (d) bias and the labeler's fit: streamlines labeled (>= 40 mm), their lengths, the Other share, the tract mix
      (total variation against today's), RapidParc's mean top tract probability, and the focus tracts' volumes (A's
      presence, and the mean over the 17 runs) against today's.
Writes results/robust_ukf_score.json, then runs reweight and paths on it, so a rerun of analyze rebuilds every key.
reweight adds, per rescan, the rename rate by A's tract and reweighted to today's tract mix: whether a lower rate is
only fewer streamlines in tracts that rename often. paths adds the rescans' path agreement and where the CST's renames
go and its new streamlines come from (paths()). Each can also be run on its own on an existing results file.
"""
import argparse, json, os, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tractline import pipeline as P
from tractline import ensemble as E
from tractline import susceptibility as S
from tractline import ukf as U
from tractline.data import DATA
from tractline.labelers.base import LUT, TRACT_NAMES, OTHER, lengths
from tractline.prep import prepare

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS, onto_reference                                         # noqa: E402
from ensemble_stages import PAIR, FLIP, FWHM, EPS, FOCUS, SITE_MM, where as stages, against_rescans   # noqa: E402
from robust_ukf_diagnose import MIN_FREE_GB, DIVERGED_MM, free_gb                          # noqa: E402

DEVICE = "mps"
SCORE = DATA / "ensemble" / "cIIIs04" / "robustness" / "ukf" / "score"
REPLICATES, PERTURBED = 16, 8
SEEDING = "seeding_threshold=0.2"           # scored from today's records
L = np.asarray(LUT)
ONEHOT = np.eye(E.N_TRACTS, dtype=np.float32)[LUT]                                         # (1600, 43)


#: a variant name's suffix for the first tie-break rule, sw and sw2 both held at a tie: "tie_angle=0.5_sw-and-sw2" is
#: {tie_angle: 0.5, tie_sw2: True} (its records were written under that name before tie_sw2 existed)
SW2_SUFFIX = "_sw-and-sw2"


def parse(variant):
    """'Rs=0.1' or 'Rs=0.1,step_length=0.6' -> {setting: value}, settings pipeline.track passes on; a boolean setting
    (tie_sw2) as 0 or 1. A name ending in SW2_SUFFIX adds tie_sw2."""
    out = {}
    if variant.endswith(SW2_SUFFIX):
        variant, out["tie_sw2"] = variant[:-len(SW2_SUFFIX)], True
    for part in variant.split(","):
        k, v = part.split("=")
        if k not in P.TRACKER_SETTINGS:
            raise SystemExit(f"{k}: not a tracker setting ({sorted(P.TRACKER_SETTINGS)})")
        out[k] = bool(float(v)) if k == "tie_sw2" else float(v)
    return out


def names(rescans):
    return (["A"] + [f"{t}_onA" for t in rescans] + [f"{FLIP}/r{r:03d}" for r in range(1, REPLICATES + 1)]
            + [f"{arm}/r{r:03d}" for arm in ("epsilon", "fieldonly") for r in range(1, PERTURBED + 1)])


def today(name):
    """Today's record of a run (ensemble_retest's and ensemble_stages')."""
    d, sd = stages()
    return sd / f"{name}.npz" if name.split("/")[0] in ("epsilon", "fieldonly") else d / f"{name}.npz"


def run(args):
    from _masivar import load
    from _bootstrap import ShellWildBootstrap
    subject, (ses, scan), rescans, shells, shell = PAIRS[PAIR]
    d, sd = stages()
    out = SCORE / args.variant
    out.mkdir(parents=True, exist_ok=True)
    A = load(subject, ses, scan, shells)
    apply = lambda dwi, h: S.apply(dwi, h, A.pe_axis, A.pe_sign, A.readout_s)
    if args.variant == SEEDING:
        path = out / "seed_keys.npy"
        if path.exists():
            print(path, "exists, skipped")
            return
        t = prepare(apply(A.dwi, np.load(sd / "A_field_hz.npy").astype(np.float64)), A.affine, A.bval, A.bvec, shell=shell, device=DEVICE)
        pts = U.seeds(U.from_arrays(t.dwi, t.header, t.mask), seeding_threshold=0.2)[0]
        keys = np.sort(E.seed_keys(np.rint(pts.cpu().numpy() - U.SRAND0_OFFSET).astype(np.int64)))
        np.save(path, keys)
        print(SEEDING, "seeds", len(keys))
        return
    tracker = parse(args.variant)
    log_path = out / "runs.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"tracker": tracker}
    retest_log = json.loads((d / "runs.json").read_text())
    labeler = P.default_labeler(DEVICE)
    grid = (A.affine, A.dwi.shape[:3])
    group = lambda n: n.split("/")[0].replace("_onA", "")
    todo = [n for n in names(rescans) if (not args.only or group(n) in args.only) and not (out / f"{n}.npz").exists()]

    def finish(name, s, dwi, t0, timer, extra=None):
        if free_gb(SCORE) < MIN_FREE_GB:
            raise SystemExit(f"stopping: {free_gb(SCORE):.1f} GB free, below {MIN_FREE_GB} GB")
        path = out / f"{name}.npz"
        path.parent.mkdir(exist_ok=True)
        with P.exact_float32(DEVICE):
            tg = P.track(s, dwi, timer, device=DEVICE, shell=shell, tracker=tracker)
            with timer("label"):
                labels = labeler(tg.fibers, draws=(0,), logp=True)
        tmp = path.with_suffix(".tmp.npz")
        E.RunRecord.of(tg, labels, *grid).save(tmp, compact=True)
        if path.exists():
            tmp.unlink()
            raise SystemExit(f"{path} appeared meanwhile; not overwritten")
        os.replace(tmp, path)
        log[name] = {"fibers": len(tg.fibers), "labeled": int(labels.keep.sum()), **(extra or {}), "seconds": timer.seconds,
                     "total_s": round(time.time() - t0, 1)}
        log_path.write_text(json.dumps(log, indent=1))
        print(args.variant, name, json.dumps(log[name]), flush=True)
        P.release_memory(DEVICE)

    def corrected(s, timer):
        with P.exact_float32(DEVICE):
            c = P.correct(s, timer, DEVICE)
        P.release_memory(DEVICE)
        return c

    wb = None
    if any(group(n) in (FLIP, "fieldonly") for n in todo):
        wb = ShellWildBootstrap(A.dwi, A.bval, A.bvec, extra_b0s=A.b0s[..., len(A.b0_index):], b0_noise="added",
                                mask=np.load(d / "A_mask.npy"), flip=FLIP, fwhm=FWHM)

    def replicate(r):
        dwi, apa = wb.replicate(r)
        return replace(A, dwi=dwi, b0s=np.concatenate([dwi[..., A.b0_index].astype(np.float64), apa], -1))

    dA = None
    for name in todo:
        t0, timer = time.time(), P.Timer()
        g, r = group(name), int(name[-3:]) if name[-3:].isdigit() else None
        if name == "A":
            finish(name, A, corrected(A, timer).dwi, t0, timer)
        elif g in rescans:
            ses_b, scan_b = rescans[g]
            B = load(subject, ses_b, scan_b, shells)
            cb = corrected(B, timer)
            with timer("onto_A"):
                dwi, bvec = onto_reference(cb.dwi, B.bvec, B.affine, np.array(retest_log[f"{g}_scan"]["to_A"]), A.affine, A.dwi.shape[:3])
            finish(name, replace(A, dwi=dwi, bval=B.bval, bvec=bvec, b0s=None), dwi, t0, timer)
        elif g == FLIP:
            s = replicate(r)
            finish(name, s, corrected(s, timer).dwi, t0, timer)
        elif g == "epsilon":
            if dA is None:
                dA = apply(A.dwi, np.load(sd / "A_field_hz.npy").astype(np.float64))
            u = np.random.default_rng(10_000 + r).uniform(-1, 1, dA.shape).astype(np.float32)
            dP = (dA * (1 + np.float32(EPS) * u)).astype(np.float32)
            del u
            finish(name, A, dP, t0, timer)
        else:                                                                   # fieldonly
            c = corrected(replicate(r), timer)
            with timer("field_apply_A"):
                dwi = apply(A.dwi, c.field_hz)
            finish(name, A, dwi, t0, timer, {"residual_left": c.residual_left})


# ----------------------------------------------------------------------------------------------- analysis

def restrict(rec, keyset):
    """A record restricted to the seeds in keyset (sorted keys); the counts are left as they are (unused here)."""
    if keyset is None:
        return rec
    m = np.isin(rec.keys, keyset)
    idx = np.flatnonzero(m)
    return E.RunRecord(rec.keys[m], rec.cluster[m], None if rec.logp is None else rec.logp[m], rec.paths[m], rec.counts, rec.shape,
                       None if rec.fibers is None else [rec.fibers[i] for i in idx])


class Presence:
    """Density presence of the named tracts on A's grid (ensemble_stages.analyze's)."""
    def __init__(self, affine, shape):
        from ensemble_grid_check import Grid, sample, RHO
        self.grid, self.sample, self.rho = Grid("dwi", affine, shape), sample, RHO
        self.named = [k for k, n in enumerate(TRACT_NAMES) if n != "Other"]
        self.vox_ml = float(np.prod(self.grid.h)) / 1000

    def __call__(self, rec):
        Pt, Ot = self.sample(rec.fibers)
        pt_tract = L[rec.cluster][Ot]
        pres = {}
        for k in self.named:
            sel = pt_tract == k
            if sel.any():
                pres[k] = self.grid.density(self.grid.counts(Pt[sel], Ot[sel])).ravel() >= self.rho
        return pres


def site_main(rec, sites):
    """Per landmark site, the named tract with the most streamlines passing within SITE_MM (-1: none)."""
    lens = np.array([len(f) for f in rec.fibers])
    pts = np.concatenate(rec.fibers)
    owner = np.repeat(np.arange(len(rec.fibers)), lens)
    t = L[rec.cluster]
    out = []
    for s in sites:
        near = np.unique(owner[np.linalg.norm(pts - np.asarray(s["mm"]), axis=1) <= SITE_MM])
        c = np.bincount(t[near], minlength=E.N_TRACTS)[:OTHER]
        out.append(int(c.argmax()) if c.max() > 0 else -1)
    return out


def top_probability(rec):
    """RapidParc's mean top tract probability (each tract's clusters' probabilities summed)."""
    out = []
    for s in range(0, len(rec.keys), 8192):
        p = np.exp(rec.logp[s:s + 8192].astype(np.float32))
        out.append((p @ ONEHOT).max(1))
    return float(np.concatenate(out).mean())


def score(get, keyset, presence, sites, labeler, log):
    """One configuration: get(name) -> its record path or None."""
    load = lambda n, **kw: restrict(E.RunRecord.load(get(n), **kw), keyset)
    rescans = PAIRS[PAIR][2]
    focus = [TRACT_NAMES.index(n) for n in FOCUS]
    A = load("A", logp=True, fibers=True)
    SA, tA, n = E.StreamlineEnsemble(A), L[A.cluster], len(A.keys)
    on_A = SA.match                                                      # (A's rows, the run's rows) of the seeds both label
    _, _, lenA = lengths(A.fibers)
    pA = presence(A)
    vol = lambda p, k: float(p[k].sum()) * presence.vox_ml if k in p else 0.0

    def dice(a, b, k):
        if k not in a or k not in b:
            return 0.0 if (k in a or k in b) else None
        return 2 * float((a[k] & b[k]).sum()) / max(float(a[k].sum() + b[k].sum()), 1)
    siteA = site_main(A, sites)
    hashA = labeler(A.fibers, draws=(0,), keys=A.keys, grouping="hash-sorted")
    res = {"A": {"streamlines labeled (>= 40 mm)": n, "fibers (>= 10 points), from the run's log": log("A"),
                 "length mm (quartiles)": [round(float(x), 1) for x in np.quantile(lenA, [0.25, 0.5, 0.75])],
                 "Other share": round(float((tA == OTHER).mean()), 4), "mean top tract probability": round(top_probability(A), 4),
                 "focus volume mL": {nm: round(vol(pA, k), 1) for nm, k in zip(FOCUS, focus)}},
           "tract mix, A": (np.bincount(tA, minlength=E.N_TRACTS) / n).tolist()}
    res["(a) rescan agreement"], rescan_t = {}, {}
    for tag in rescans:
        if get(f"{tag}_onA") is None:
            continue
        R = load(f"{tag}_onA", fibers=True)
        ri, qi = on_A(R)
        t = np.full(n, -1, np.int16); t[ri] = L[R.cluster[qi]]
        rescan_t[tag] = t
        hashR = labeler(R.fibers, draws=(0,), keys=R.keys, grouping="hash-sorted")
        kA, kR = A.keys[hashA.keep], R.keys[hashR.keep]
        ia = {int(k): i for i, k in enumerate(kA)}
        both = [(ia[int(k)], j) for j, k in enumerate(kR) if int(k) in ia]
        pR, sR = presence(R), site_main(R, sites)
        res["(a) rescan agreement"][tag] = {
            "common seeds": len(ri), "renamed": round(float((tA[ri] != L[R.cluster[qi]]).mean()), 4),
            "renamed, hash-sorted grouping": round(float(np.mean([hashA.tract[a] != hashR.tract[b] for a, b in both])), 4),
            "presence Dice": {nm: dice(pA, pR, k) and round(dice(pA, pR, k), 4) for nm, k in zip(FOCUS, focus)},
            f"landmark sites: the same main named tract within {SITE_MM:g} mm": int(sum(a == b for a, b in zip(siteA, sR))),
            "landmark sites": len(sites)}
    reps = [f"{FLIP}/r{r:03d}" for r in range(1, REPLICATES + 1)]
    if all(get(nm) is not None for nm in reps):
        T, vols, mix = [tA], [[vol(pA, k) for k in focus]], np.bincount(tA, minlength=E.N_TRACTS).astype(float)
        for nm in reps:
            R = load(nm, fibers=True)
            ri, qi = on_A(R)
            t = np.full(n, -1, np.int16); t[ri] = L[R.cluster[qi]]
            T.append(t)
            p = presence(R)
            vols.append([vol(p, k) for k in focus])
            mix += np.bincount(L[R.cluster], minlength=E.N_TRACTS)
        res["(b) honesty (17 runs)"] = against_rescans(np.stack(T), tA, rescan_t) if rescan_t else None
        res["17 runs: Other share"] = round(float(mix[OTHER] / mix.sum()), 4)
        res["17 runs: focus volume mL (mean)"] = {nm: round(float(v), 1) for nm, v in zip(FOCUS, np.mean(vols, 0))}
        res["tract mix, 17 runs"] = (mix / mix.sum()).tolist()
    res["(c) robustness"] = {}
    for arm in ("epsilon", "fieldonly"):
        nms = [f"{arm}/r{r:03d}" for r in range(1, PERTURBED + 1) if get(f"{arm}/r{r:03d}") is not None]
        if not nms:
            continue
        ren, far, lost = [], [], []
        for nm in nms:
            R = load(nm)
            ri, qi = on_A(R)
            ren.append(float((tA[ri] != L[R.cluster[qi]]).mean()))
            far.append(float((E.mdf(A.paths[ri], R.paths[qi]) > DIVERGED_MM).mean()))
            lost.append(n - len(ri))
        res["(c) robustness"][arm] = {"runs": len(nms), "renamed (mean)": round(float(np.mean(ren)), 4),
                                      "paths more than 5 mm apart (mean)": round(float(np.mean(far)), 4),
                                      "A's seeds the run lacks (mean)": round(float(np.mean(lost)), 1)}
    return res


def analyze(args):
    from tractline.labelers import rapidparc as RP
    t0 = time.time()
    d, sd = stages()
    affine = np.load(d / "landmarks.npz")["affine"]
    A0 = E.RunRecord.load(d / "A.npz", logp=False)
    presence = Presence(affine, A0.shape)
    sites = json.loads((HERE / "results/ensemble_landmarks_smooth.json").read_text())["sites"]
    labeler = RP.Labeler(DEVICE)
    logs = {"retest": json.loads((d / "runs.json").read_text()), "stages": json.loads((sd / "runs.json").read_text())}
    configs = {"today": (lambda nm: today(nm) if today(nm).exists() else None, None, lambda nm: logs["retest"].get(nm, {}).get("fibers"))}
    if (SCORE / SEEDING / "seed_keys.npy").exists():
        configs[SEEDING] = (configs["today"][0], np.load(SCORE / SEEDING / "seed_keys.npy"), lambda nm: None)
    for v in sorted(p.name for p in SCORE.iterdir() if p.is_dir() and p.name != SEEDING) if SCORE.exists() else []:
        vl = json.loads((SCORE / v / "runs.json").read_text()) if (SCORE / v / "runs.json").exists() else {}
        configs[v] = ((lambda vv: lambda nm: SCORE / vv / f"{nm}.npz" if (SCORE / vv / f"{nm}.npz").exists() else None)(v), None,
                      (lambda ll: lambda nm: ll.get(nm, {}).get("fibers"))(vl))
    res = {"pair": PAIR, "focus": FOCUS, "site radius mm": SITE_MM, "configurations": {}}
    for name, (get, keyset, log) in configs.items():
        if get("A") is None:
            continue
        res["configurations"][name] = score(get, keyset, presence, sites, labeler, log)
        print(name, f"{time.time() - t0:.0f}s", flush=True)
    base = res["configurations"]["today"]
    tv = lambda a, b: round(0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum()), 4)
    for name, r in res["configurations"].items():
        r["(d) against today"] = {"tract mix, A: total variation": tv(r["tract mix, A"], base["tract mix, A"]),
                                  "focus volume, A: ratio": {nm: round(r["A"]["focus volume mL"][nm] / max(base["A"]["focus volume mL"][nm], 1e-9), 3) for nm in FOCUS}}
        if "tract mix, 17 runs" in r and "tract mix, 17 runs" in base:
            r["(d) against today"]["tract mix, 17 runs: total variation"] = tv(r["tract mix, 17 runs"], base["tract mix, 17 runs"])
            r["(d) against today"]["focus volume, 17 runs: ratio"] = {
                nm: round(r["17 runs: focus volume mL (mean)"][nm] / max(base["17 runs: focus volume mL (mean)"][nm], 1e-9), 3) for nm in FOCUS}
        for k in ("tract mix, A", "tract mix, 17 runs"):
            r[k] = [round(x, 5) for x in r.get(k, [])] or None
        print(name, json.dumps({k: v for k, v in r.items() if not k.startswith("tract mix")}), flush=True)
    (HERE / "results/robust_ukf_score.json").write_text(json.dumps(res, indent=1))
    reweight(args)                                                       # the keys these add, so analyze is complete on its own
    paths(args)
    print(f"{time.time() - t0:.0f}s")


def reweight(args):
    """Is a lower rename rate only a different tract mix? Per configuration and rescan, over the seeds both label: the
    rename rate per A tract, and their mean weighted by today's A tract mix over today's common seeds (tracts a
    configuration lacks left out, the weights renormalized). Added to results/robust_ukf_score.json in place."""
    path = HERE / "results/robust_ukf_score.json"
    res = json.loads(path.read_text())
    d, _ = stages()
    get = {"today": lambda nm: today(nm), SEEDING: lambda nm: today(nm)}
    keysets = {SEEDING: np.load(SCORE / SEEDING / "seed_keys.npy")} if (SCORE / SEEDING / "seed_keys.npy").exists() else {}
    focus = {n: TRACT_NAMES.index(n) for n in FOCUS + ("Other",)}

    def rates(name, tag):
        g = get.get(name, lambda nm: SCORE / name / f"{nm}.npz")
        A = restrict(E.RunRecord.load(g("A"), logp=False), keysets.get(name))
        R = restrict(E.RunRecord.load(g(f"{tag}_onA"), logp=False), keysets.get(name))
        ri, qi = E.StreamlineEnsemble(A).match(R)
        tA, ch = L[A.cluster[ri]], L[A.cluster[ri]] != L[R.cluster[qi]]
        n = np.bincount(tA, minlength=E.N_TRACTS)
        return n, np.bincount(tA, weights=ch, minlength=E.N_TRACTS) / np.maximum(n, 1)

    for tag in PAIRS[PAIR][2]:
        n0, _ = rates("today", tag)
        w0 = n0 / n0.sum()
        for name, r in res["configurations"].items():
            n, rate = rates(name, tag)
            has = n > 0
            r["(a) rescan agreement"][tag]["renamed, reweighted to today's tract mix"] = round(float((w0[has] * rate[has]).sum() / w0[has].sum()), 4)
            r["(a) rescan agreement"][tag]["renamed by A's tract"] = {k: round(float(rate[i]), 4) for k, i in focus.items()}
            print(name, tag, r["(a) rescan agreement"][tag]["renamed"], r["(a) rescan agreement"][tag]["renamed, reweighted to today's tract mix"],
                  r["(a) rescan agreement"][tag]["renamed by A's tract"], flush=True)
    path.write_text(json.dumps(res, indent=1))


def paths(args):
    """Two checks of whether the variants' rescan gains are smoothing, added to results/robust_ukf_score.json in place:
      (a) path agreement: per configuration and rescan, over the seeds both label, the share of A's paths more than
          5 mm (MDF) from the rescan's path from the same seed, and the median MDF;
      (b) the CST: where the rescans' renames of A's CST streamlines go (the top destinations, the three rescans
          pooled), and, against today's A over the seeds both label, which of today's tracts the variant's CST
          streamlines had, and how many of today's CST streamlines the variant keeps CST."""
    path = HERE / "results/robust_ukf_score.json"
    res = json.loads(path.read_text())
    keysets = {SEEDING: np.load(SCORE / SEEDING / "seed_keys.npy")} if (SCORE / SEEDING / "seed_keys.npy").exists() else {}
    get = lambda name, nm: today(nm) if name in ("today", SEEDING) else SCORE / name / f"{nm}.npz"
    load = lambda name, nm: restrict(E.RunRecord.load(get(name, nm), logp=False), keysets.get(name))
    cst = TRACT_NAMES.index("CST")
    top = lambda t, n=5: {TRACT_NAMES[k]: round(float((t == k).mean()), 4) for k in np.argsort(-np.bincount(t, minlength=E.N_TRACTS))[:n] if (t == k).any()}
    A0 = load("today", "A")
    for name, r in res["configurations"].items():
        A = load(name, "A")
        SA, tA = E.StreamlineEnsemble(A), L[A.cluster]
        dest = []
        for tag in rescan_tags():
            R = load(name, f"{tag}_onA")
            ri, qi = SA.match(R)
            m = E.mdf(A.paths[ri], R.paths[qi])
            r["(a) rescan agreement"][tag]["paths more than 5 mm apart"] = round(float((m > DIVERGED_MM).mean()), 4)
            r["(a) rescan agreement"][tag]["median MDF mm"] = round(float(np.median(m)), 3)
            tR = L[R.cluster[qi]]
            sel = (tA[ri] == cst) & (tR != cst)
            dest.append(tR[sel])
        dest = np.concatenate(dest)
        cst_row = {"A's CST streamlines a rescan renames, pooled: their new tract (top 5)": top(dest)}
        if name != "today":
            ri, qi = E.StreamlineEnsemble(A0).match(A)
            t0, t1 = L[A0.cluster[ri]], L[A.cluster[qi]]
            cst_row["against today's A, the seeds both label"] = int(len(ri))
            cst_row["the variant's CST streamlines: today's tract (top 5)"] = top(t0[t1 == cst])
            cst_row["today's CST streamlines the variant keeps CST"] = round(float((t1[t0 == cst] == cst).mean()), 4)
        r["(b) CST: where renames go and where CST comes from"] = cst_row
        print(name, json.dumps({t: {k: v for k, v in r["(a) rescan agreement"][t].items() if k in ("paths more than 5 mm apart", "median MDF mm")}
                                for t in rescan_tags()}), json.dumps(cst_row), flush=True)
    path.write_text(json.dumps(res, indent=1))


def rescan_tags():
    return list(PAIRS[PAIR][2])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "analyze", "reweight", "paths"))
    ap.add_argument("--variant", help="run: <setting>=<value>[,...] (pipeline.track's tracker=), or seeding_threshold=0.2")
    ap.add_argument("--only", nargs="+", help="run: only these groups: A, within, session3, session5, smooth, epsilon, fieldonly")
    args = ap.parse_args()
    if args.what == "run" and not args.variant:
        ap.error("run needs --variant")
    {"run": run, "analyze": analyze, "reweight": reweight, "paths": paths}[args.what](args)
