"""Tracker variants scored on what matters: real rescans, honesty, robustness and bias (docs/robustness-plan.md, step 2;
NOTES 2026-10-17, "Robustness step 2").

    uv run --group dipy --group rankfield bench/robust_ukf_score.py run --variant Rs=0.1 [--only A within smooth ..]
    uv run --group dipy --group rankfield bench/robust_ukf_score.py run --variant seeding_threshold=0.2
    uv run --group dipy --group rankfield bench/robust_ukf_score.py run --variant denoise=patch2self [--only A within smooth]
    uv run --group dipy --group rankfield bench/robust_ukf_score.py analyze
    uv run --group dipy --group rankfield bench/robust_ukf_score.py reweight     (after analyze)
    uv run --group dipy --group rankfield bench/robust_ukf_score.py paths        (after analyze)
    uv run --group dipy --group rankfield bench/robust_ukf_score.py records      (after analyze)

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
run --variant denoise=<mppca|patch2self>[,denoise_order=corrected][,<tracker setting>=<value>...]: the DWI denoised
(pipeline.denoise_subject; docs/robustness-plan.md, step 3), with or without tracker settings, into
DATA/ensemble/cIIIs04/robustness/denoise/<variant>/ (the same guards). Every scan is denoised as acquired, on its own
grid, before its field is estimated and it is corrected (denoise_order=corrected: after, for comparison):
  A            A denoised, corrected, tracked; its denoising's stats also over A's tracking mask (retest's A_mask),
               beside the bootstrap's noise check (retest's runs.json) for MP-PCA's sigma;
  <tag>_onA    the rescan denoised on its own grid, corrected there, then resampled onto A's;
  smooth/r..   the replicate made from the raw scan's fit and residuals, as today, then denoised and corrected as a
               scan would be: it carries the scan's noise into the denoiser;
  epsilon/r..  A's tracker input as the variant makes it (denoised and corrected), x (1 + 1e-6 u);
  fieldonly/r.. the field estimated from replicate r denoised (as smooth/r's; reused when smooth/r ran in the same
               process), applied to A's DWI denoised as acquired (order corrected: the raw replicate's field, applied to
               A's raw DWI, then denoised).
  denoise=<m>,bootstrap=denoised: the honesty ensemble of a denoised scan, its own records only the replicates
               (smooth/r..), drawn from A denoised as acquired (its own fit and residuals) and corrected with no second
               denoising; A and the rescans are denoise=<m>'s (record());
  fieldonly_rawfield/.. (only when named in --only): today's raw-replicate fields (ensemble_stages', float16), and
               A's own raw field (A_field), applied to A denoised as acquired: fieldonly's field and DWI apart.
--first N runs only the first N of each arm (a smoke test).
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
Writes results/robust_ukf_score.json (--out: elsewhere; --configs: today and only these), then runs reweight, paths,
records, by_length, cst and checks on it, so a rerun of analyze rebuilds every key. records marks each configuration's scope: "full" when it
has every run (A, the three rescans, 16 replicates, 8 epsilon, 8 fieldonly), else "partial" with the runs it has (a
smoke test, e.g. denoise=mppca: A and the within rescan only); a partial configuration's figures are not comparable
with the full ones.
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
DENOISED = DATA / "ensemble" / "cIIIs04" / "robustness" / "denoise"   # step 3's variants (a name with denoise=)
DENOISE_KEYS = ("denoise", "denoise_order", "bootstrap")
BOOTSTRAP_DENOISED = ",bootstrap=denoised"  # a denoise variant's honesty ensemble bootstrapped from its denoised A (run)
RAWFIELD, RAWFIELD_RUNS = "fieldonly_rawfield", 4   # a denoise variant's fieldonly, but with today's raw-replicate fields
RAWFIELD_A = f"{RAWFIELD}/A_field"                 # and with A's own raw field: the offset between the two fields alone
REPLICATES, PERTURBED = 16, 8
SEEDING = "seeding_threshold=0.2"           # scored from today's records
L = np.asarray(LUT)
ONEHOT = np.eye(E.N_TRACTS, dtype=np.float32)[LUT]                                         # (1600, 43)


#: a variant name's suffix for the first tie-break rule, sw and sw2 both held at a tie: "tie_angle=0.5_sw-and-sw2" is
#: {tie_angle: 0.5, tie_sw2: True} (its records were written under that name before tie_sw2 existed)
SW2_SUFFIX = "_sw-and-sw2"


def parse(variant):
    """'Rs=0.1' or 'Rs=0.1,step_length=0.6' -> {setting: value}, settings pipeline.track passes on; a boolean setting
    (tie_sw2) as 0 or 1. A name ending in SW2_SUFFIX adds tie_sw2. The denoising's keys (denoising) are skipped."""
    out = {}
    if variant.endswith(SW2_SUFFIX):
        variant, out["tie_sw2"] = variant[:-len(SW2_SUFFIX)], True
    for part in variant.split(","):
        k, v = part.split("=")
        if k in DENOISE_KEYS:
            continue
        if k not in P.TRACKER_SETTINGS:
            raise SystemExit(f"{k}: not a tracker setting ({sorted(P.TRACKER_SETTINGS)})")
        out[k] = bool(float(v)) if k == "tie_sw2" else float(v)
    return out


def denoising(variant):
    """'denoise=mppca[:r3][,denoise_order=corrected][,bootstrap=denoised][,...]' -> {"method", "order", "bootstrap"};
    None without denoise=. The method may carry its setting (tractline.denoise.parse)."""
    from tractline import denoise as D
    kv = dict(part.split("=", 1) for part in variant.removesuffix(SW2_SUFFIX).split(",") if "=" in part)
    if "denoise" not in kv:
        if "denoise_order" in kv or "bootstrap" in kv:
            raise SystemExit("denoise_order or bootstrap without denoise")
        return None
    out = {"method": kv["denoise"], "order": kv.get("denoise_order", "acquired"), "bootstrap": kv.get("bootstrap", "raw")}
    try:
        D.parse(out["method"])
    except ValueError as e:
        raise SystemExit(f"{variant}: {e}")
    if out["order"] not in P.DENOISE_ORDERS or out["bootstrap"] not in ("raw", "denoised"):
        raise SystemExit(f"{variant}: denoise_order one of {P.DENOISE_ORDERS}, bootstrap raw or denoised")
    if out["bootstrap"] == "denoised" and out["order"] != "acquired":
        raise SystemExit(f"{variant}: bootstrap=denoised needs the order acquired")
    return out


def parent(variant):
    """The variant whose A and rescans a bootstrap=denoised configuration uses (its own records are the replicates)."""
    return variant.replace(BOOTSTRAP_DENOISED, "") if BOOTSTRAP_DENOISED in variant else variant


def vdir(variant):
    """A variant's records: step 3's (denoise=) under DENOISED, the others under SCORE."""
    return (DENOISED if "denoise=" in variant else SCORE) / variant


def record(variant, nm):
    """A configuration's record of run nm: today's for today (and the seeding subset); else the variant's own, or, for
    a bootstrap=denoised configuration, its parent's (A and the rescans). None when there is none."""
    if variant in ("today", SEEDING):
        return today(nm) if today(nm).exists() else None
    shared = nm == "A" or nm.endswith("_onA")                          # what a bootstrap=denoised configuration borrows
    for v in dict.fromkeys((variant, parent(variant) if shared else variant)):
        if (vdir(v) / f"{nm}.npz").exists():
            return vdir(v) / f"{nm}.npz"
    return None


def variants():
    return sorted(p.name for root in (SCORE, DENOISED) if root.exists() for p in root.iterdir() if p.is_dir() and p.name != SEEDING)


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
    out = vdir(args.variant)
    if free_gb(out) < MIN_FREE_GB:
        raise SystemExit(f"stopping: {free_gb(out):.1f} GB free, below {MIN_FREE_GB} GB")
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
    tracker, dn = parse(args.variant), denoising(args.variant)
    log_path = out / "runs.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {"tracker": tracker, **({"denoise": dn} if dn else {})}
    retest_log = json.loads((d / "runs.json").read_text())
    labeler = P.default_labeler(DEVICE)
    grid = (A.affine, A.dwi.shape[:3])
    group = lambda n: n.split("/")[0].replace("_onA", "")
    rawfield = [RAWFIELD_A] + [f"{RAWFIELD}/r{r:03d}" for r in range(1, RAWFIELD_RUNS + 1)]
    every = names(rescans) + (rawfield if dn and dn["order"] == "acquired" and args.only and RAWFIELD in args.only else [])
    if dn and dn["bootstrap"] == "denoised":                           # only the replicates are its own
        every = [n for n in every if group(n) == FLIP]
    first = lambda n: args.first is None or not n[-3:].isdigit() or int(n[-3:]) <= args.first
    todo = [n for n in every if (not args.only or group(n) in args.only) and first(n) and not (out / f"{n}.npz").exists()]

    def finish(name, s, dwi, t0, timer, extra=None):
        if free_gb(out) < MIN_FREE_GB:
            raise SystemExit(f"stopping: {free_gb(out):.1f} GB free, below {MIN_FREE_GB} GB")
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

    mask_A = np.load(d / "A_mask.npy")                                 # A's grid: A's and the replicates' denoising stats
    wb, base = None, A                                                 # base: the scan the replicates are drawn from
    if any(group(n) in (FLIP, "fieldonly") for n in todo):
        if dn and dn["bootstrap"] == "denoised":
            # the honesty ensemble of a denoised scan: today's bootstrap applied to A denoised as acquired - its own
            # shell-wise fit, its own residuals about that fit (the noise the denoised scan keeps, and model error, as
            # today's residuals of the raw scan), b0 noise "added" - and each replicate corrected and tracked with no
            # second denoising: a plausible denoised scan. A's reversed b0 (not denoised) is the extra b0, as acquired.
            t = P.Timer()
            base = P.denoise_subject(A, t, dn["method"])[0]
            log["bootstrap"] = {"from": "A denoised as acquired", "denoise_s": t.seconds["denoise"]}
        wb = ShellWildBootstrap(base.dwi, A.bval, A.bvec, extra_b0s=A.b0s[..., len(A.b0_index):], b0_noise="added",
                                mask=mask_A, flip=FLIP, fwhm=FWHM)
        if base is not A:
            log["bootstrap"]["noise check, A's mask"] = wb.noise_check(mask_A)
            log_path.write_text(json.dumps(log, indent=1))

    def replicate(r):
        dwi, apa = wb.replicate(r)
        return replace(A, dwi=dwi, b0s=np.concatenate([dwi[..., A.b0_index].astype(np.float64), apa], -1))

    def prepared(s, timer, mask=None):
        """What the variant tracks: s denoised as acquired (step 3), corrected, or corrected then denoised (order
        corrected); without denoise=, today's corrected DWI. (the DWI tracked, the Correction, s as corrected, the
        denoising's stats or None)."""
        st = None
        if dn and dn["order"] == "acquired":
            s, st = P.denoise_subject(s, timer, dn["method"], mask=mask)
        c = corrected(s, timer)
        dwi = c.dwi
        if dn and dn["order"] == "corrected":
            dwi, st = P.denoise_corrected(c.dwi, s.bval, timer, dn["method"], mask=mask)
        return dwi, c, s, st

    cache, fields = {}, {}                                             # A's tracker input and state; replicates' fields

    def state_A(timer):
        """A as the variant makes it (computed once a process): its tracker input, the Correction, A as corrected
        (denoised as acquired, when the order is), the denoising's stats."""
        if "A" not in cache:
            cache["A"] = prepared(A, timer, mask_A)
        return cache["A"]

    dA = None
    for name in todo:
        t0, timer = time.time(), P.Timer()
        g, r = group(name), int(name[-3:]) if name[-3:].isdigit() else None
        if name == "A":
            if dn is None:
                finish(name, A, corrected(A, timer).dwi, t0, timer)
            else:
                dwi, c, _, st = state_A(timer)
                bs = retest_log.get(f"bootstrap_{FLIP}", {})
                check = {k: bs.get(k) for k in ("dw_residual_rms", "b0_residual_sd", "b0_scale")}
                finish(name, A, dwi, t0, timer, {"residual_left": c.residual_left, "denoise": st,
                                                 "bootstrap noise check (retest's runs.json, A's mask)": check})
        elif g in rescans:
            ses_b, scan_b = rescans[g]
            B = load(subject, ses_b, scan_b, shells)
            dwi_b, cb, _, st = prepared(B, timer)                      # denoised on its own grid, before resampling
            with timer("onto_A"):
                dwi, bvec = onto_reference(dwi_b, B.bvec, B.affine, np.array(retest_log[f"{g}_scan"]["to_A"]), A.affine, A.dwi.shape[:3])
            del dwi_b
            finish(name, replace(A, dwi=dwi, bval=B.bval, bvec=bvec, b0s=None), dwi, t0, timer, {"denoise": st} if dn else None)
        elif g == FLIP:
            s = replicate(r)
            if dn and dn["bootstrap"] == "denoised":                   # a replicate of the denoised scan: no second denoising
                c = corrected(s, timer)
                finish(name, s, c.dwi, t0, timer, {"residual_left": c.residual_left, "bootstrap": "from A denoised"})
            elif dn is None:
                finish(name, s, corrected(s, timer).dwi, t0, timer)
            else:
                dwi, c, _, st = prepared(s, timer, mask_A)
                fields[r] = c.field_hz
                finish(name, s, dwi, t0, timer, {"residual_left": c.residual_left, "denoise": st})
        elif g == "epsilon":
            if dA is None:
                dA = apply(A.dwi, np.load(sd / "A_field_hz.npy").astype(np.float64)) if dn is None else state_A(timer)[0]
            u = np.random.default_rng(10_000 + r).uniform(-1, 1, dA.shape).astype(np.float32)
            dP = (dA * (1 + np.float32(EPS) * u)).astype(np.float32)
            del u
            finish(name, A, dP, t0, timer)
        elif g == RAWFIELD:
            # fieldonly's two changes apart: today's field from raw replicate r (stored as float16 by ensemble_stages),
            # applied to A denoised as acquired. Against the variant's fieldonly (the denoised replicate's field), this
            # separates a calmer field from a denoised DWI that amplifies a field's wobble less.
            # RAWFIELD_A: A's own field from its raw b0s (stages/A_field_hz.npy, float32), the offset between the raw and
            # the denoised b0s' fields with no replicate noise
            src = sd / "A_field_hz.npy" if name == RAWFIELD_A else sd / "fieldonly" / f"r{r:03d}_field_hz.npy"
            with timer("field_apply_A"):
                dwi = apply(state_A(timer)[2].dwi, np.load(src).astype(np.float64))
            finish(name, A, dwi, t0, timer, {"field": str(src.relative_to(d))})
        elif dn is None:                                                        # fieldonly
            c = corrected(replicate(r), timer)
            with timer("field_apply_A"):
                dwi = apply(A.dwi, c.field_hz)
            finish(name, A, dwi, t0, timer, {"residual_left": c.residual_left})
        else:                                                                   # fieldonly, denoised
            extra = {"field": "smooth/r's, this process" if r in fields else "estimated here"}
            if r not in fields:
                fields[r] = (prepared(replicate(r), timer)[1] if dn["order"] == "acquired" else corrected(replicate(r), timer)).field_hz
            with timer("field_apply_A"):
                dwi = apply((state_A(timer)[2] if dn["order"] == "acquired" else A).dwi, fields[r])
            if dn["order"] == "corrected":
                dwi, extra["denoise"] = P.denoise_corrected(dwi, A.bval, timer, dn["method"], mask=mask_A)
            finish(name, A, dwi, t0, timer, extra)


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
    for arm in ("epsilon", "fieldonly", RAWFIELD):
        nms = [f"{arm}/r{r:03d}" for r in range(1, PERTURBED + 1) if get(f"{arm}/r{r:03d}") is not None]
        if arm == RAWFIELD and get(RAWFIELD_A) is not None:
            R = load(RAWFIELD_A)
            ri, qi = on_A(R)
            res["(c) robustness"][f"{RAWFIELD}: A's own raw field"] = {
                "renamed": round(float((tA[ri] != L[R.cluster[qi]]).mean()), 4),
                "paths more than 5 mm apart": round(float((E.mdf(A.paths[ri], R.paths[qi]) > DIVERGED_MM).mean()), 4),
                "A's seeds the run lacks": int(n - len(ri))}
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
    if (SCORE / SEEDING / "seed_keys.npy").exists() and (not args.configs or SEEDING in args.configs):
        configs[SEEDING] = (configs["today"][0], np.load(SCORE / SEEDING / "seed_keys.npy"), lambda nm: None)
    for v in variants():
        if args.configs and v not in args.configs:
            continue
        logs_v = [json.loads((vdir(x) / "runs.json").read_text()) for x in dict.fromkeys((v, parent(v))) if (vdir(x) / "runs.json").exists()]
        configs[v] = ((lambda vv: lambda nm: record(vv, nm))(v), None,
                      (lambda ll: lambda nm: next((x[nm].get("fibers") for x in ll if nm in x), None))(logs_v))
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
    out(args).write_text(json.dumps(res, indent=1))
    reweight(args)                                                       # the keys these add, so analyze is complete on its own
    paths(args)
    records(args)                                                        # before the steps that read each scope
    by_length(args)
    cst(args)
    checks(args)
    print(f"{time.time() - t0:.0f}s")


def reweight(args):
    """Is a lower rename rate only a different tract mix? Per configuration and rescan, over the seeds both label: the
    rename rate per A tract, and their mean weighted by today's A tract mix over today's common seeds (tracts a
    configuration lacks left out, the weights renormalized). Added to results/robust_ukf_score.json in place."""
    path = out(args)
    res = json.loads(path.read_text())
    d, _ = stages()
    get = {"today": lambda nm: today(nm), SEEDING: lambda nm: today(nm)}
    keysets = {SEEDING: np.load(SCORE / SEEDING / "seed_keys.npy")} if (SCORE / SEEDING / "seed_keys.npy").exists() else {}
    focus = {n: TRACT_NAMES.index(n) for n in FOCUS + ("Other",)}

    def rates(name, tag):
        g = get.get(name, lambda nm: record(name, nm))
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
            if tag not in r["(a) rescan agreement"]:
                continue
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
    path = out(args)
    res = json.loads(path.read_text())
    keysets = {SEEDING: np.load(SCORE / SEEDING / "seed_keys.npy")} if (SCORE / SEEDING / "seed_keys.npy").exists() else {}
    get = record
    load = lambda name, nm: restrict(E.RunRecord.load(get(name, nm), logp=False), keysets.get(name))
    cst = TRACT_NAMES.index("CST")
    top = lambda t, n=5: {TRACT_NAMES[k]: round(float((t == k).mean()), 4) for k in np.argsort(-np.bincount(t, minlength=E.N_TRACTS))[:n] if (t == k).any()}
    A0 = load("today", "A")
    for name, r in res["configurations"].items():
        A = load(name, "A")
        SA, tA = E.StreamlineEnsemble(A), L[A.cluster]
        dest = []
        for tag in [t for t in rescan_tags() if t in r["(a) rescan agreement"]]:
            R = load(name, f"{tag}_onA")
            ri, qi = SA.match(R)
            m = E.mdf(A.paths[ri], R.paths[qi])
            r["(a) rescan agreement"][tag]["paths more than 5 mm apart"] = round(float((m > DIVERGED_MM).mean()), 4)
            r["(a) rescan agreement"][tag]["median MDF mm"] = round(float(np.median(m)), 3)
            tR = L[R.cluster[qi]]
            sel = (tA[ri] == cst) & (tR != cst)
            dest.append(tR[sel])
        dest = np.concatenate(dest) if dest else np.zeros(0, np.int64)
        cst_row = {"A's CST streamlines a rescan renames, pooled: their new tract (top 5)": top(dest)}
        if name != "today":
            ri, qi = E.StreamlineEnsemble(A0).match(A)
            t0, t1 = L[A0.cluster[ri]], L[A.cluster[qi]]
            cst_row["against today's A, the seeds both label"] = int(len(ri))
            cst_row["the variant's CST streamlines: today's tract (top 5)"] = top(t0[t1 == cst])
            cst_row["today's CST streamlines the variant keeps CST"] = round(float((t1[t0 == cst] == cst).mean()), 4)
        r["(b) CST: where renames go and where CST comes from"] = cst_row
        print(name, json.dumps({t: {k: v for k, v in r["(a) rescan agreement"][t].items() if k in ("paths more than 5 mm apart", "median MDF mm")}
                                for t in rescan_tags() if t in r["(a) rescan agreement"]}), json.dumps(cst_row), flush=True)
    path.write_text(json.dumps(res, indent=1))


def by_length(args):
    """Do shorter paths flatter the rescan agreement? Per full configuration and rescan, over the seeds both label, by
    the tertile of A's streamline length (the cut points today's A's tertiles, the same for every configuration): the
    share of the common seeds in the bin, the share the rescan renames, and A's paths more than 5 mm (MDF) from the
    rescan's. Added to the results file in place, under each rescan's "(a) rescan agreement"."""
    path = out(args)
    res = json.loads(path.read_text())
    keysets = {SEEDING: np.load(SCORE / SEEDING / "seed_keys.npy")} if (SCORE / SEEDING / "seed_keys.npy").exists() else {}
    load = lambda name, nm: restrict(E.RunRecord.load(record(name, nm), logp=False, fibers=nm == "A"), keysets.get(name))
    cuts = np.quantile(lengths(load("today", "A").fibers)[2], [1 / 3, 2 / 3])
    res["length tertiles of today's A (mm)"] = [round(float(c), 1) for c in cuts]
    for name, r in res["configurations"].items():
        if not r.get("records", {}).get("scope", "full").startswith("full"):
            continue
        A = load(name, "A")
        b = np.digitize(lengths(A.fibers)[2], cuts)                      # 0, 1, 2: short, middle, long
        SA, tA = E.StreamlineEnsemble(A), L[A.cluster]
        for tag in [t for t in rescan_tags() if t in r["(a) rescan agreement"]]:
            R = load(name, f"{tag}_onA")
            ri, qi = SA.match(R)
            ren, far, bi = tA[ri] != L[R.cluster[qi]], E.mdf(A.paths[ri], R.paths[qi]) > DIVERGED_MM, b[ri]
            r["(a) rescan agreement"][tag]["by A's length (today's tertiles): share / renamed / paths more than 5 mm apart"] = {
                k: [round(float((bi == i).mean()), 4), round(float(ren[bi == i].mean()), 4), round(float(far[bi == i].mean()), 4)]
                for i, k in enumerate(("short", "middle", "long"))}
        print(name, json.dumps({t: r["(a) rescan agreement"][t].get("by A's length (today's tertiles): share / renamed / paths more than 5 mm apart")
                                for t in rescan_tags() if t in r["(a) rescan agreement"]}), flush=True)
    path.write_text(json.dumps(res, indent=1))


def cst(args):
    """Is a variant's CST today's bundle, reached from other seeds, or another bundle? Per configuration with a
    changed tractography (each full one but today and the seeding subset), from A's records against today's A: the
    presence Dice of the focus tracts; over the seeds both label, the CST streamlines of each (counts), and of the
    variant's CST streamlines by today's tract of the same seed (CST, Other, any other): their share, the median MDF
    to today's path, and the mean share of their points inside today's CST presence (also for today's path from those
    seeds); the same for today's CST streamlines the variant does not call CST. Added to the results file in place."""
    from ensemble_grid_check import sample
    path = out(args)
    res = json.loads(path.read_text())
    d, _ = stages()
    presence = Presence(np.load(d / "landmarks.npz")["affine"], E.RunRecord.load(d / "A.npz", logp=False).shape)
    grid, k = presence.grid, {n: TRACT_NAMES.index(n) for n in FOCUS + ("Other",)}
    A0 = E.RunRecord.load(record("today", "A"), logp=False, fibers=True)
    p0, t0 = presence(A0), L[A0.cluster]

    def inside(fibers, mask):
        """per streamline, the share of its sampled points inside mask (A's grid, flat)"""
        Pt, Ot = sample(fibers)
        ijk = np.rint(Pt @ grid.inv[:3, :3].T + grid.inv[:3, 3]).astype(int)
        ok = np.all((ijk >= 0) & (ijk < grid.shape), 1)
        hit = np.zeros(len(Pt), bool)
        hit[ok] = mask[np.ravel_multi_index(ijk[ok].T, grid.shape)]
        return np.bincount(Ot, weights=hit, minlength=len(fibers)) / np.maximum(np.bincount(Ot, minlength=len(fibers)), 1)

    in0 = inside(A0.fibers, p0[k["CST"]])
    res["CST: today's A, mean share of points inside today's CST presence: its CST / its Other streamlines"] = [
        round(float(in0[t0 == k["CST"]].mean()), 3), round(float(in0[t0 == k["Other"]].mean()), 3)]
    for name, r in res["configurations"].items():
        if name in ("today", SEEDING) or BOOTSTRAP_DENOISED in name or r.get("records", {}).get("scope") != "full":
            continue
        V = E.RunRecord.load(record(name, "A"), logp=False, fibers=True)
        pv, tv = presence(V), L[V.cluster]
        ri, qi = E.StreamlineEnsemble(A0).match(V)
        a, b = t0[ri], tv[qi]
        m = E.mdf(A0.paths[ri], V.paths[qi])
        inv = inside([V.fibers[i] for i in qi], p0[k["CST"]])
        vc = b == k["CST"]
        dice = lambda n: round(2 * float((p0[k[n]] & pv[k[n]]).sum()) / float(p0[k[n]].sum() + pv[k[n]].sum()), 4)
        row = {"presence Dice, A against today's A": {n: dice(n) for n in FOCUS},
               "seeds both label": int(len(ri)), "CST streamlines over those seeds: today / the variant": [int((a == k["CST"]).sum()), int(vc.sum())]}
        for lab, sel in (("today's CST", vc & (a == k["CST"])), ("today's Other", vc & (a == k["Other"])),
                         ("another tract", vc & (a != k["CST"]) & (a != k["Other"]))):
            row[f"the variant's CST from {lab}: share / median MDF mm / points in today's CST: the variant's path, today's"] = [
                round(float(sel.sum() / max(vc.sum(), 1)), 4), round(float(np.median(m[sel])), 2) if sel.any() else None,
                round(float(inv[sel].mean()), 3) if sel.any() else None, round(float(in0[ri][sel].mean()), 3) if sel.any() else None]
        lost = (a == k["CST"]) & ~vc
        row["today's CST the variant does not call CST: share / median MDF mm / points in today's CST"] = [
            round(float(lost.sum() / max((a == k["CST"]).sum(), 1)), 4), round(float(np.median(m[lost])), 2) if lost.any() else None,
            round(float(inv[lost].mean()), 3) if lost.any() else None]
        r["(b) CST against today's A"] = row
        print(name, json.dumps(row), flush=True)
    path.write_text(json.dumps(res, indent=1))


def checks(args):
    """Added under "checks" in the results file, each over the seeds both runs label (renamed, paths more than 5 mm
    apart, median MDF):
      - Patch2Self's sketch seed: a variant "denoise=patch2self:s<k>" with A alone (partial) against
        denoise=patch2self's A (seed 0);
      - fieldonly's wobble alone on the denoised A: each fieldonly_rawfield/r.. (today's raw-replicate field) against
        fieldonly_rawfield/A_field (A's own raw field), both on A denoised as acquired, so the raw-against-denoised
        field offset is in both and only the replicate's wobble differs (comparable with today's fieldonly);
      - what the run logs hold (denoise=patch2self's and its bootstrap=denoised's runs.json, the retest's runs.json):
        the RMS Patch2Self removed, its seconds, the replicates' labeled counts, the denoised bootstrap's noise
        check, and each design's first replicate against its A."""
    path = out(args)
    res = json.loads(path.read_text())
    base = "denoise=patch2self"
    res["checks"] = {}

    def compare(a_path, b_path):
        A, V = (E.RunRecord.load(x, logp=False) for x in (a_path, b_path))
        ri, qi = E.StreamlineEnsemble(A).match(V)
        m = E.mdf(A.paths[ri], V.paths[qi])
        return {"seeds both label": int(len(ri)), "renamed": round(float((L[A.cluster[ri]] != L[V.cluster[qi]]).mean()), 4),
                "paths more than 5 mm apart": round(float((m > DIVERGED_MM).mean()), 4), "median MDF mm": round(float(np.median(m)), 3)}

    ref = record(base, RAWFIELD_A)
    runs = [record(base, f"{RAWFIELD}/r{r:03d}") for r in range(1, RAWFIELD_RUNS + 1)]
    if ref is not None and all(x is not None for x in runs):
        rows = [compare(ref, x) for x in runs]
        res["checks"]["fieldonly's wobble alone on the denoised A: raw-replicate fields against A's own raw field (mean of runs)"] = {
            "runs": len(rows), **{k: round(float(np.mean([x[k] for x in rows])), 4) for k in ("renamed", "paths more than 5 mm apart", "median MDF mm")},
            "per run, paths more than 5 mm apart": [x["paths more than 5 mm apart"] for x in rows]}
    d, _ = stages()
    lg = json.loads((vdir(base) / "runs.json").read_text()) if (vdir(base) / "runs.json").exists() else {}
    if lg:
        dn = lambda nm: lg[nm].get("denoise") or {}
        rng = lambda xs: [round(float(min(xs)), 3), round(float(max(xs)), 3)]
        groups = {"A": ["A"], "rescans": [f"{t}_onA" for t in rescan_tags()], "replicates": [f"{FLIP}/r{r:03d}" for r in range(1, REPLICATES + 1)]}
        key = "bright voxels: removed RMS, all / b0s / DW volumes"
        res["checks"][f"{base}: the run log (runs.json)"] = {
            "removed RMS, bright voxels, all / b0s / DW volumes: range": {
                g: [rng([dn(nm)[key][i] for nm in nms if nm in lg]) for i in range(3)] for g, nms in groups.items()},
            "removed mean, bright voxels: range": {g: rng([dn(nm)["bright voxels: removed mean (a bias)"] for nm in nms if nm in lg]) for g, nms in groups.items()},
            "A, removed RMS in A's tracking mask, all / b0s / DW volumes": dn("A").get("the mask: removed RMS, all / b0s / DW volumes"),
            "denoising seconds (the worker), range": rng([dn(nm)["seconds"] for nms in groups.values() for nm in nms if nm in lg]),
            "replicates: streamlines labeled (mean)": round(float(np.mean([lg[nm]["labeled"] for nm in groups["replicates"] if nm in lg])), 1),
            "today's replicates: streamlines labeled (mean)": round(float(np.mean([json.loads((d / "runs.json").read_text())[nm]["labeled"] for nm in groups["replicates"]])), 1),
            "run seconds (total_s) by group": {g: round(float(sum(lg[nm]["total_s"] for nm in lg if isinstance(lg[nm], dict) and "total_s" in lg[nm]
                                                                 and nm.split("/")[0].replace("_onA", "") == g))) for g in ["A", *rescan_tags(), FLIP, "epsilon", "fieldonly"]}}
    boot = base + BOOTSTRAP_DENOISED
    if (vdir(boot) / "runs.json").exists():
        bl = json.loads((vdir(boot) / "runs.json").read_text())
        res["checks"][f"{boot}: the bootstrap's noise check (runs.json)"] = bl.get("bootstrap", {}).get("noise check, A's mask")
    for v in ("today", base, boot):
        if record(v, f"{FLIP}/r001") is not None:
            res["checks"][f"{v}: the first replicate against its A"] = compare(record(v, "A"), record(v, f"{FLIP}/r001"))
    for v in [x for x in variants() if x.startswith(base + ":s") and "," not in x]:
        if record(v, "A") is None or record(base, "A") is None:
            continue
        A, V = (E.RunRecord.load(record(x, "A"), logp=False) for x in (base, v))
        res["checks"][f"Patch2Self's sketch seed: {v}'s A against {base}'s (seed 0); A only (partial)"] = {
            "streamlines labeled: seed 0 / this": [int(len(A.keys)), int(len(V.keys))], **compare(record(base, "A"), record(v, "A"))}
    print(json.dumps(res["checks"]), flush=True)
    path.write_text(json.dumps(res, indent=1))


def records(args):
    """Each configuration's runs and scope ("full" or "partial"), added to the results file in place."""
    path = out(args)
    res = json.loads(path.read_text())
    want = names(rescan_tags())
    for name, r in res["configurations"].items():
        got = [nm for nm in want if record(name, nm) is not None]
        own = [nm for nm in got if name in ("today", SEEDING) or (vdir(name) / f"{nm}.npz").exists()]
        if BOOTSTRAP_DENOISED in name:
            scope = (f"honesty only: its own 16 replicates (bootstrapped from A denoised), A and the rescans {parent(name)}'s"
                     if sum(nm.startswith(FLIP + "/") for nm in own) == REPLICATES else "partial: not comparable with the full configurations")
        else:
            scope = "full" if len(got) == len(want) else "partial: not comparable with the full configurations"
        r["records"] = {"scope": scope,
                        "runs": {g: sum(nm.split("/")[0].replace("_onA", "") == g for nm in got)
                                 for g in ["A", *rescan_tags(), FLIP, "epsilon", "fieldonly"]}}
        print(name, json.dumps(r["records"]), flush=True)
    path.write_text(json.dumps(res, indent=1))


def out(args):
    return Path(args.out) if args.out else HERE / "results/robust_ukf_score.json"


def rescan_tags():
    return list(PAIRS[PAIR][2])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "analyze", "reweight", "paths", "by_length", "cst", "checks", "records"))
    ap.add_argument("--variant", help="run: <setting>=<value>[,...] (pipeline.track's tracker=; denoise=mppca|patch2self, "
                                      "denoise_order=acquired|corrected), or seeding_threshold=0.2")
    ap.add_argument("--only", nargs="+", help="run: only these groups: A, within, session3, session5, smooth, epsilon, fieldonly, "
                                              f"{RAWFIELD} (denoise variants; only when named)")
    ap.add_argument("--first", type=int, help="run: only the first N runs of each arm (a smoke test)")
    ap.add_argument("--configs", nargs="+", help="analyze: today and only these configurations (default: every one with records)")
    ap.add_argument("--out", help="analyze, reweight, paths, records: the results file (default results/robust_ukf_score.json)")
    args = ap.parse_args()
    if args.what == "run" and not args.variant:
        ap.error("run needs --variant")
    {"run": run, "analyze": analyze, "reweight": reweight, "paths": paths, "by_length": by_length, "cst": cst, "checks": checks,
     "records": records}[args.what](args)
