"""A calibration card for single scans (docs/robustness-plan.md, step 5): learned once per protocol from test-retest
subjects, it turns one scan's 17-run ensemble statistics into calibrated statements about what a rescan shows.

    uv run --group dipy --group rankfield bench/calibration_card.py collect --subject cIIIs03 [--runs 16]
    uv run --group dipy --group rankfield bench/calibration_card.py fit [--subjects ...] [--out results/calibration_card_masivar_cIII.json]
    uv run --group dipy --group rankfield bench/calibration_card.py test          # the unit tests, on synthetic data (or: pytest bench/calibration_card.py)

The records are bench/calibration_runs.py's (CARD: eight MASiVar cohort III subjects, scanner B): per subject the
reference scan A's ensemble (A and smooth/r001..r016, R = 17) and its rescans tracked on A's grid (<tag>_onA), each a
same-session or an other-session rescan.

collect (per subject; no pipeline runs): every run's labeled streamlines, density presence and the main named tract
near a set of points, reduced to what the card needs, in DATA/ensemble/<subject>/card/<pair>/statements_R<R>.npz
(refused if it exists, unless --force: it is derived, not a record). The statements:
  - streamline: over the seeds A's R runs all label, the ensemble's mode (the most frequent tract) and its share
    (votes / R): the stated P(a rescan gives the mode). Observed: the rescan's tract at the seed (seed keys, A's grid).
    The pipeline has never presented a single streamline's name as a finding; this level is kept for diagnosis and as
    the comparison with earlier entries, not as a statement the data product makes;
  - envelope: per named tract, density presence (ensemble_grid_check: streamlines sampled every 0.4 mm, density over
    the voxel's mean cross-section, smoothed 1 mm, at least 0.3 per mm^2) of each run on A's grid; the votes (0..R)
    per voxel. The envelope at stated level L keeps the highest-vote voxels holding L of the tract's summed votes
    (ensemble_pooled.thresholds); its coverage is the share of the rescan's presence voxels inside it, pooled over the
    42 named tracts (weighted by the rescan's voxels). Beside it the core (at least half the runs), the Vorob'ev shape
    (the cut whose volume is closest to a run's mean, ensemble_vorobev.shapes) and the widest (any run): coverage,
    precision and size against the rescan. Stored as vote histograms per tract: cnt[k, v] (voxels with v votes) and
    hit[k, v] (of those, in the rescan's presence), which give every {votes >= t} shape exactly;
  - point: the main named tract within SITE_MM (8 mm) of a point: per run, the named tract with the most streamlines
    passing within 8 mm (ensemble_where_which's composition; Other excluded; none if no named streamline). Points: a
    lattice every LATTICE voxels (about 8.6 x 8.6 x 8.8 mm) in A's tracking mask where A's run has at least
    MIN_STREAMLINES streamlines. Stated: the mode's share over the R runs (where the mode is a named tract). Observed:
    the rescan's main tract is the mode.
  - pooled (the upper reference, NOTES 2026-10-13 "Pooling sessions"): where rescans have their own 17-run ensembles
    (cIIIs04; cIIIs02 if `calibration_runs.py rescans` is run), each rescan's run predicted by three scans' ensembles
    pooled (51 runs, A first), as stated.

fit: per statement and term (same session; other session), a monotone map from stated to observed, fitted on the
subjects' rescans (each subject weighted equally; a subject's rescans of one term pooled): streamline and point, the
observed rate per vote count k = 1..R, made nondecreasing by weighted pool-adjacent-violators (isotonic regression on
the R + 1 stated levels); envelope, the observed coverage at each stated level on GRID (nondecreasing by
construction: the envelopes nest), and its inverse (the stated level whose training coverage first reaches a target);
shapes, the training coverage, precision and size. Evaluated leaving one subject out: fitted on the others, scored on
the held-out subject's rescans, against no card (as stated), PAT16's double-bootstrap vote table (ensemble_double,
R = 11, applied to the vote fraction) and, where available, pooled sessions. Writes the card.

The card (JSON; schema "tractline.calibration_card/2"):
  card id, card version, created (UTC);
  protocol      scanner, shells and directions, b0s (forward, reversed), readout assumption, series handling
                (_masivar), and "recognized by": calibration_runs.PROTOCOL, every item load_checked checks, with its
                tolerances (machine-readable);
  pipeline      the fit's git state (dirty files among src/ and the card's bench modules), the bench modules' sha256,
                the settings (from every subject's card folder, asserted equal) and their sha256, library versions,
                the labeler weights' sha256, R, the statement files' sha256, and per subject the provenance of each
                record group (the card's folder with its git state at creation, or a reused folder with its file
                dates and no git state where none was recorded);
  validity      "match": the machine-checkable fields (PROTOCOL, R, the settings' sha256, the weights' sha256, the
                terms); and the domain in words: healthy adults, this protocol, these settings and R, rescans within a
                session or in another session on the same scanner; outside it the product says "run-to-run only";
  terms         {"same session" | "other session": {
                   "streamline": {"stated": [k/R], "observed": [P(rescan gives the mode)], "n": [...], "use": ...},
                   "point": {as streamline, for the main named tract within 8 mm},
                   "envelope": {"stated levels": GRID, "observed coverage": [...],
                                "stated level for target coverage": {"0.5": {level, reached, at floor}, ...}},
                   "shapes": {"core" | "vorobev" | "widest" | "A alone" | "envelope L": {coverage, precision, size}},
                   "subjects": [...], "rescans": n}};
  accuracy (provisional)  step 4's phantom term (results/robust_phantom.json): envelope coverage of the ground truth's
                own density presence, against a bias-free ensemble's; F1 under the 2023 ROI scoring; label agreement.
                One simulated brain at b = 1000 (32 directions), not this protocol: provisional, protocol-mismatched,
                "apply": false (its gaps are at uncalibrated stated levels and must not be stacked on the card's);
  held-out      leave-one-subject-out: per subject and term, the fold's own map ("fold map"; fold stability in the
                summary is computed from them), ECE and mean stated / observed (streamline, point) as stated, with the
                card, with PAT16's map; envelope coverage at targets as stated and with the card; shapes' predicted
                against observed; pooled sessions where available, and the pooled ensemble against the card like for
                like (common seeds, each against its own mode; envelope size at matched held-out coverage); the
                same-session card on other-session rescans; per rescan, its mode rates and coverage;
  subjects      per subject: its pair, R, rescans and terms.
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT, TRACT_NAMES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from calibration_runs import CARD, Records, FLIP, PROTOCOL, git_state, module_hashes, sha256  # noqa: E402

OUT = HERE / "results/calibration_card_masivar_cIII.json"
L = np.asarray(LUT)
NAMED = E.N_TRACTS - 1                          # the 42 named tracts; Other excluded from regional statements
FOCUS = ("CST", "AF", "UF")
TERMS = ("same session", "other session")
GRID = np.round(np.arange(0.2, 0.9951, 0.005), 3)    # from 0.2: stated 0.5 covers about 0.53 of a rescan
LEVELS = (0.5, 0.8, 0.9, 0.925, 0.95)
TARGETS = (0.5, 0.8, 0.9, 0.95)
SITE_MM, LATTICE, MIN_STREAMLINES = 8.0, 4, 5
POOL_SCANS = 3
MATCHED = (0.8, 0.85, 0.9, 0.95)                # held-out coverages at which card and pooled envelope sizes are compared
CARD_VERSION = 3                 # 3 (2026-10-24): validity.match gains the pipeline config hash (tractline.config) when the
                                 # subjects' settings carry a config (folders made from 2026-10-24); older folders give none
ECE_BINS = (0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001)


# ----------------------------------------------------------------------------------------------- collect

def centers(mask, counts, shape):
    """Lattice points (flat voxel indices) in A's mask where A's run has at least MIN_STREAMLINES streamlines."""
    ijk = np.indices(shape)
    on = (ijk % LATTICE == LATTICE // 2).all(0) & mask.reshape(shape) & (counts.sum(0).reshape(shape) >= MIN_STREAMLINES)
    return np.flatnonzero(on.ravel())


def sphere_matrix(cent, shape, vox):
    """(nvox, ncenters) sparse: voxel v lies within SITE_MM of center c."""
    from scipy.sparse import csr_matrix
    r = np.ceil(SITE_MM / vox).astype(int)
    off = np.stack(np.meshgrid(*[np.arange(-a, a + 1) for a in r], indexing="ij"), -1).reshape(-1, 3)
    off = off[np.linalg.norm(off * vox, axis=1) <= SITE_MM]
    c = np.stack(np.unravel_index(cent, shape), 1)
    v = (c[:, None] + off[None]).reshape(-1, 3)
    col = np.repeat(np.arange(len(cent)), len(off))
    ok = ((v >= 0) & (v < shape)).all(1)
    return csr_matrix((np.ones(ok.sum(), np.float32), (np.ravel_multi_index(v[ok].T, shape), col[ok])), shape=(int(np.prod(shape)), len(cent)))


def run_stats(path, grid, M):
    """One run on A's grid: (seed keys, tracts, density presence (NAMED, nvox) bool, main named tract per point)."""
    from scipy.sparse import coo_matrix
    from ensemble_grid_check import sample, RHO
    rec = E.RunRecord.load(path, logp=False, fibers=True)
    Pt, Ot = sample(rec.fibers)
    tract = L[rec.cluster]
    pt_tract = tract[Ot]
    nvox = int(np.prod(grid.shape))
    pres = np.zeros((NAMED, nvox), bool)
    for k in range(NAMED):
        sel = pt_tract == k
        if sel.any():
            pres[k] = grid.density(grid.counts(Pt[sel], Ot[sel])).ravel() >= RHO
    ijk = np.rint(Pt @ grid.inv[:3, :3].T + grid.inv[:3, 3]).astype(np.int64)
    ok = np.all((ijk >= 0) & (ijk < grid.shape), 1)
    pair = np.unique(Ot[ok].astype(np.int64) * nvox + np.ravel_multi_index(ijk[ok].T, grid.shape))
    s, v = np.divmod(pair, nvox)
    C = (coo_matrix((np.ones(len(s), np.float32), (s, v)), shape=(len(tract), nvox)).tocsr() @ M).tocoo()   # streamline x point
    n = M.shape[1]
    cnt = np.bincount(tract[C.row].astype(np.int64) * n + C.col, minlength=E.N_TRACTS * n).reshape(E.N_TRACTS, n)[:NAMED]
    main = np.where(cnt.max(0) > 0, cnt.argmax(0), -1).astype(np.int16)
    return rec.keys, tract.astype(np.int16), pres, main


def on_keys(keys_ref, keys, tract):
    """keys' tracts placed on keys_ref (-1 where the run has no streamline from the seed)."""
    out = np.full(len(keys_ref), -1, np.int16)
    order = np.argsort(keys)
    pos = np.searchsorted(keys[order], keys_ref).clip(0, len(keys) - 1)
    hit = keys[order][pos] == keys_ref
    out[hit] = tract[order[pos[hit]]]
    return out


def mode_votes(T, values):
    """T (R, n) with values in `values` (-1 allowed): per column the most frequent value and its count."""
    counts = np.stack([(T == x).sum(0) for x in values])
    return np.asarray(values)[counts.argmax(0)].astype(np.int16), counts.max(0).astype(np.int16)


def vote_hist(V, target, R):
    """V (NAMED, nvox) votes 0..R, target (NAMED, nvox) bool -> cnt, hit (NAMED, R + 1): voxels with v votes, and
    those in the target."""
    cnt = np.stack([np.bincount(V[k], minlength=R + 1) for k in range(NAMED)])
    hit = np.stack([np.bincount(V[k][target[k]], minlength=R + 1) for k in range(NAMED)])
    return cnt.astype(np.int64), hit.astype(np.int64)


class Scan:
    """One scan's runs on A's grid, reduced: tracts on A's seeds (R, n), presence votes (NAMED, nvox) and its own
    run's presence, main tract per point (R, ncenters)."""
    def __init__(self, paths, keys_ref, grid, M, t0):
        self.R = len(paths)
        T, P = [], []
        self.votes = None
        for i, p in enumerate(paths):
            k, t, pres, main = run_stats(p, grid, M)
            T.append(on_keys(keys_ref, k, t)); P.append(main)
            self.votes = pres.astype(np.uint8) if self.votes is None else self.votes + pres
            if i == 0:
                self.own = pres
            print(f"  {p.parent.name}/{p.name} {time.time() - t0:.0f}s", flush=True)
        self.T, self.P = np.stack(T), np.stack(P)


def collect(args):
    import nibabel as nib
    from _masivar import scan_files
    from ensemble_grid_check import Grid
    t0 = time.time()
    subject = args.subject
    R = Records(subject)
    pair = R.pair
    R.dir.mkdir(parents=True, exist_ok=True)
    out = R.dir / f"statements_R{args.runs + 1}.npz"
    if out.exists() and not args.force:
        raise FileExistsError(f"{out} exists (--force to rebuild it)")
    paths = R.ensemble("", args.runs)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{subject}: {len(missing)} of A's {len(paths)} runs missing, the first {missing[0]}")
    tags = [t for t in pair.rescans if R.have(f"{t}_onA")]
    img = nib.load(scan_files(subject, *pair.ref)["APP"][0][1])
    A0 = E.RunRecord.load(paths[0], logp=False)
    grid = Grid("dwi", img.affine, A0.shape)
    mask = np.load(R.path("A_mask", ".npy"))
    cent = centers(mask, A0.counts, A0.shape)
    M = sphere_matrix(cent, A0.shape, grid.h)
    keys = A0.keys
    print(f"{subject}: R = {len(paths)}, rescans {tags}, {len(cent)} points", flush=True)
    A = Scan(paths, keys, grid, M, t0)
    Rn = A.R
    tA = A.T[0]
    s_ok = (A.T >= 0).all(0)
    s_mode, s_votes = mode_votes(A.T[:, s_ok], range(E.N_TRACTS))
    s_Avotes = (A.T[:, s_ok] == tA[s_ok]).sum(0).astype(np.int16)
    p_mode, p_votes = mode_votes(A.P, range(-1, NAMED))
    cnt, _ = vote_hist(A.votes, A.own, Rn)
    arrays = {"keys": keys[s_ok], "s_A": tA[s_ok], "s_mode": s_mode, "s_votes": s_votes, "s_Avotes": s_Avotes,
              "centers": cent, "p_mode": p_mode, "p_votes": p_votes, "e_cnt": cnt,
              "e_Asize": A.own.sum(1).astype(np.int64)}
    targets, target_keys = {}, {}
    for tag in tags:
        k, t, pres, main = run_stats(R.path(f"{tag}_onA"), grid, M)
        targets[tag], target_keys[tag] = pres, (k, t)
        arrays[f"s_tract_{tag}"] = on_keys(keys, k, t)[s_ok]
        arrays[f"p_main_{tag}"] = main
        arrays[f"e_hit_{tag}"] = vote_hist(A.votes, pres, Rn)[1]
        arrays[f"e_Ahit_{tag}"] = (A.own & pres).sum(1).astype(np.int64)
        print(f"  {tag}_onA {time.time() - t0:.0f}s", flush=True)
    # pooled sessions: rescans with their own complete ensembles
    full = [t for t in tags if all(p.exists() for p in R.ensemble(t, args.runs))]
    pooled = {}
    if full:
        scans = {"A": A}
        for t in full:
            scans[t] = Scan(R.ensemble(t, args.runs), keys, grid, M, t0)
        for tag in tags:
            pool = [s for s in scans if s != tag][:POOL_SCANS]
            if len(pool) < POOL_SCANS:
                continue
            Rp = sum(scans[s].R for s in pool)
            V = sum(scans[s].votes.astype(np.int16) for s in pool)
            pc, ph = vote_hist(V, targets[tag], Rp)
            T = np.concatenate([scans[s].T for s in pool])
            ok = (T >= 0).all(0)
            m, v = mode_votes(T[:, ok], range(E.N_TRACTS))
            rt = on_keys(keys, *target_keys[tag])[ok]
            arrays.update({f"pool_e_cnt_{tag}": pc, f"pool_e_hit_{tag}": ph, f"pool_s_votes_{tag}": v, f"pool_s_keys_{tag}": keys[ok],
                           f"pool_s_hit_{tag}": np.where(rt >= 0, (rt == m).astype(np.int8), -1).astype(np.int8)})
            pooled[tag] = {"scans": pool, "runs": Rp}
    log = json.loads((R.dir / "runs.json").read_text()) if (R.dir / "runs.json").exists() else {}
    reuse_log = json.loads((R.reuse / "runs.json").read_text()) if R.reuse is not None else {}
    scan_log = lambda t: log.get(f"{t}_scan") or reuse_log.get(f"{t}_scan", {})  # noqa: E731
    meta = {"subject": subject, "pair": pair.name, "R": Rn, "runs": [str(p.relative_to(DATA)) for p in paths],
            "rescans": {t: {"term": "same session" if t in pair.same else "other session", "scan": scan_log(t).get("name"),
                            "rotation_deg": scan_log(t).get("rotation_deg"), "translation_mm": scan_log(t).get("translation_mm"),
                            "ncc_after": scan_log(t).get("ncc_after")} for t in tags},
            "pooled": pooled, "points": int(len(cent)), "seeds labeled in all R runs": int(s_ok.sum()),
            "provenance": provenance(subject, tags, args.runs), "collect git": git_state(), "seconds": round(time.time() - t0, 1)}
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)
    print(json.dumps(meta, indent=1))


def provenance(subject, tags, runs=16):
    """Where each group of a subject's records came from: the card's folder (made by calibration_runs.py, its runs.json
    holding the git state at creation) or a reused folder (read only; its records' file times, and no git state where
    none was recorded)."""
    from datetime import datetime, timezone
    R = Records(subject)
    log = json.loads((R.dir / "runs.json").read_text()) if (R.dir / "runs.json").exists() else {}
    groups = {"A and its replicates": R.ensemble("", runs), **{f"{t}_onA": [R.path(f"{t}_onA")] for t in tags}}
    for t in tags:
        reps = R.ensemble(t, runs)[1:]
        if all(p.exists() for p in reps):
            groups[f"{t}'s replicates"] = reps
    day = lambda p: datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime("%Y-%m-%d")  # noqa: E731
    out = {}
    for name, paths in groups.items():
        rows = {}
        for q in paths:
            mine = R.dir in q.parents
            key = "card folder" if mine else "reused folder"
            rows.setdefault(key, []).append(q)
        out[name] = {k: {"folder": str((R.dir if k == "card folder" else R.reuse).relative_to(DATA)), "records": len(v),
                         "made (file dates, UTC)": sorted({day(q) for q in v}),
                         "git at creation": (log.get("created") or {}).get("git") if k == "card folder" else
                         "none recorded (made by ensemble_retest.py before the card; calibration_runs.py verify reruns A at HEAD)"}
                     for k, v in rows.items()}
    return out


# ----------------------------------------------------------------------------------------------- the maps

def pav(y, w):
    """Weighted pool-adjacent-violators: the nondecreasing sequence nearest y in weighted least squares."""
    blocks = []                                                  # [mean, weight, count]
    for yi, wi in zip(np.asarray(y, float), np.asarray(w, float)):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2, n2 = blocks.pop()
            m1, w1, n1 = blocks[-1]
            blocks[-1] = [(m1 * w1 + m2 * w2) / max(w1 + w2, 1e-300), w1 + w2, n1 + n2]
    return np.concatenate([np.full(n, m) for m, _, n in blocks])


def rate_table(groups, R):
    """groups: per subject (votes 1..R, hit 0/1) -> per vote count k = 0..R the observed rate, each subject weighted
    equally (a subject's rows normalized to 1), and the counts. Rates of absent counts are nan."""
    num, den, n = np.zeros(R + 1), np.zeros(R + 1), np.zeros(R + 1, np.int64)
    for v, h in groups:
        if not len(v):
            continue
        c = np.bincount(v, minlength=R + 1)
        w = c / c.sum()
        rate = np.bincount(v, h.astype(float), minlength=R + 1) / np.maximum(c, 1)
        num += w * rate; den += w; n += c
    return np.where(den > 0, num / np.maximum(den, 1e-300), np.nan), den, n


def fit_votes(groups, R):
    """The monotone map stated (k / R) -> observed: isotonic over the vote counts present, interpolated between."""
    rate, w, n = rate_table(groups, R)
    have = np.flatnonzero(w > 0)
    k = np.arange(R + 1)
    fitted = np.full(R + 1, np.nan)
    if len(have):
        fitted[have] = pav(rate[have], w[have])
        fitted = np.interp(k, have, fitted[have])
    return {"stated": (k / R).round(4).tolist(), "observed": [None if np.isnan(x) else round(float(x), 4) for x in fitted],
            "n": n.tolist()}


def apply_votes(card_map, v, R):
    obs = np.array([np.nan if x is None else x for x in card_map["observed"]], float)
    return obs[v] if len(obs) == R + 1 else np.interp(v / R, card_map["stated"], obs)


def thresholds_from_counts(c, levels):
    """ensemble_pooled.thresholds from a vote histogram c (R + 1,): per level, the lowest vote count its envelope keeps."""
    R = len(c) - 1
    mass = c * np.arange(R + 1, dtype=float)
    cum = np.cumsum(mass[::-1])[::-1]
    total = cum[1] if R >= 1 else 0
    out = []
    for lv in levels:
        ok = np.flatnonzero(cum >= lv * total)
        out.append(int(ok.max()) if len(ok) else 1)
    return out


def vorobev_cut(c):
    """ensemble_vorobev.shapes: the vote count whose set's volume is closest to a run's mean volume."""
    R = len(c) - 1
    vol = np.cumsum(c[::-1])[::-1]
    mean_run = (c * np.arange(R + 1)).sum() / R
    return int(np.argmin(np.abs(vol[1:] - mean_run))) + 1


def shape_cuts(cnt, levels=GRID):
    """Per tract the lowest vote count kept: (NAMED, len(levels)) envelopes, and core, Vorob'ev, widest (NAMED,)."""
    R = cnt.shape[1] - 1
    env = np.array([thresholds_from_counts(c, levels) for c in cnt])
    return env, {"core": np.full(len(cnt), int(np.ceil(R / 2))), "vorobev": np.array([vorobev_cut(c) for c in cnt]),
                 "widest": np.ones(len(cnt), int)}


def kept(h, cut):
    """h (NAMED, R + 1) -> per tract the sum at or above its cut (NAMED,)"""
    cum = np.cumsum(h[:, ::-1], 1)[:, ::-1]
    return cum[np.arange(len(h)), cut]


def coverage_curve(cnt, hit, levels=GRID, tracts=None):
    """Envelope coverage of the target at each stated level, pooled over tracts (weights: the target's voxels)."""
    env, _ = shape_cuts(cnt, levels)
    sel = np.arange(len(cnt)) if tracts is None else np.asarray(tracts)
    den = hit[sel].sum()
    return np.array([kept(hit[sel], env[sel, j]).sum() / max(den, 1) for j in range(len(levels))])


def shape_scores(cnt, hit, tracts=None):
    """Coverage, precision and size (against the target) of the core, Vorob'ev, widest and envelope shapes."""
    env, cuts = shape_cuts(cnt, np.array(LEVELS))
    sel = np.arange(len(cnt)) if tracts is None else np.asarray(tracts)
    n_t = hit[sel].sum()
    out = {}
    for name, cut in list(cuts.items()) + [(f"envelope {lv:g}", env[:, j]) for j, lv in enumerate(LEVELS)]:
        size, both = kept(cnt[sel], cut[sel]).sum(), kept(hit[sel], cut[sel]).sum()
        out[name] = {"coverage": both / max(n_t, 1), "precision": both / max(size, 1), "size": size / max(n_t, 1)}
    return out


def level_for(curve, target):
    """(level, reached, at floor): the lowest stated level on GRID whose coverage reaches the target; the widest
    (1.0) if none does; at floor when GRID's first level already reaches it (the level then over-meets the target,
    and a lower one is not known)."""
    ok = np.flatnonzero(curve >= target)
    return (float(GRID[ok[0]]), True, bool(ok[0] == 0)) if len(ok) else (1.0, False, False)


def curves(cnt, hit, levels=GRID):
    """Coverage and size (against the target) of the envelopes at each level, then the widest, pooled over tracts."""
    env, cuts = shape_cuts(cnt, levels)
    n = max(hit.sum(), 1)
    cov = [kept(hit, env[:, j]).sum() / n for j in range(len(levels))] + [kept(hit, cuts["widest"]).sum() / n]
    size = [kept(cnt, env[:, j]).sum() / n for j in range(len(levels))] + [kept(cnt, cuts["widest"]).sum() / n]
    return np.array(cov), np.array(size)


def size_at(cov, size, q):
    """The envelope size at held-out coverage q, interpolated along the (nondecreasing) coverage curve; None beyond it."""
    c = np.maximum.accumulate(cov)
    return round(float(np.interp(q, c, size)), 4) if c[0] <= q <= c[-1] else None


def ece(p, hit):
    from ensemble_pilot import calibration
    c = calibration(p, hit, ECE_BINS)
    return {"ece": c["ece"], "stated": round(float(np.mean(p)), 4), "observed": round(float(np.mean(hit)), 4), "n": int(len(p)),
            "bins": c["bins"]}


def pat16_table():
    """PAT16's double-bootstrap vote table (ensemble_double, R = 11, smooth signs): P(keep | votes of 11), as a
    function of the vote fraction (interpolated between 1/11 and 11/11)."""
    t = json.loads((HERE / "results/ensemble_double_smooth.json").read_text())["recalibration for rescans"]["votes: P(keep | votes of R)"]
    R = t["R"]
    return lambda f: np.interp(f, np.arange(1, R + 1) / R, np.array(t["table"][1:], float))


# ----------------------------------------------------------------------------------------------- fitting

class Subject:
    """A subject's statements (collect's npz), grouped by term."""
    def __init__(self, path):
        z = np.load(path)
        self.meta = json.loads(str(z["meta"]))
        self.z = {k: z[k] for k in z.files if k != "meta"}
        self.name, self.R = self.meta["subject"], self.meta["R"]

    def tags(self, term):
        return [t for t, r in self.meta["rescans"].items() if r["term"] == term]

    def streamline(self, term, event="mode"):
        """(votes, hit) pooled over the term's rescans; event "mode" (rescan gives the mode) or "A" (keeps A's tract)."""
        z, vs, hs = self.z, [], []
        for t in self.tags(term):
            rt = z[f"s_tract_{t}"]
            ok = rt >= 0
            ref, votes = (z["s_mode"], z["s_votes"]) if event == "mode" else (z["s_A"], z["s_Avotes"])
            vs.append(votes[ok]); hs.append(rt[ok] == ref[ok])
        return (np.concatenate(vs), np.concatenate(hs)) if vs else (np.zeros(0, int), np.zeros(0, bool))

    def point(self, term):
        z, vs, hs = self.z, [], []
        named = z["p_mode"] >= 0
        for t in self.tags(term):
            vs.append(z["p_votes"][named]); hs.append(z[f"p_main_{t}"][named] == z["p_mode"][named])
        return (np.concatenate(vs), np.concatenate(hs)) if vs else (np.zeros(0, int), np.zeros(0, bool))

    def envelopes(self, term):
        """[(cnt, hit, A's |A n T|, |A|)] per rescan of the term"""
        z = self.z
        return [(z["e_cnt"], z[f"e_hit_{t}"], z[f"e_Ahit_{t}"], z["e_Asize"]) for t in self.tags(term)]

    def pooled(self, term):
        return [t for t in self.tags(term) if f"pool_e_cnt_{t}" in self.z]


def term_curve(subj, term, tracts=None):
    """The subject's mean coverage curve over its rescans of the term (None if it has none)."""
    rows = subj.envelopes(term)
    return np.mean([coverage_curve(c, h, tracts=tracts) for c, h, _, _ in rows], 0) if rows else None


def term_shapes(subj, term, tracts=None):
    rows = subj.envelopes(term)
    if not rows:
        return None
    per = [shape_scores(c, h, tracts) for c, h, _, _ in rows]
    sel = slice(None) if tracts is None else np.asarray(tracts)
    alone = [{"coverage": ah[sel].sum() / max(h[sel].sum(), 1), "precision": ah[sel].sum() / max(a[sel].sum(), 1),
              "size": a[sel].sum() / max(h[sel].sum(), 1)} for _, h, ah, a in rows]
    out = {k: {m: float(np.mean([p[k][m] for p in per])) for m in per[0][k]} for k in per[0]}
    out["A alone"] = {m: float(np.mean([x[m] for x in alone])) for m in alone[0]}
    return out


def fit_term(subjects, term):
    """The card's maps for one term, from the subjects that have rescans of it (each weighted equally)."""
    have = [s for s in subjects if s.tags(term)]
    if not have:
        return None
    R = have[0].R
    assert all(s.R == R for s in have), "subjects at different R"
    curves = [term_curve(s, term) for s in have]
    curve = np.mean(curves, 0)
    focus = [TRACT_NAMES.index(n) for n in FOCUS]
    shapes = [term_shapes(s, term) for s in have]
    shapes_f = [term_shapes(s, term, focus) for s in have]
    avg = lambda rows: {k: {m: round(float(np.mean([r[k][m] for r in rows])), 4) for m in rows[0][k]} for k in rows[0]}  # noqa: E731
    return {
        "subjects": [s.name for s in have], "rescans": int(sum(len(s.tags(term)) for s in have)), "R": R,
        "streamline": {**fit_votes([s.streamline(term) for s in have], R),
                       "statement": "P(a rescan's streamline from this seed gets the ensemble's mode tract), stated as the mode's share of the R runs",
                       "use": "diagnostic only: the data product never presents a single streamline's name as a finding"},
        "point": {**fit_votes([s.point(term) for s in have], R),
                  "statement": f"P(a rescan's main named tract within {SITE_MM:g} mm of the point is the ensemble's mode), stated as the mode's share"},
        "envelope": {"stated levels": GRID.tolist(), "observed coverage": [round(float(x), 4) for x in curve],
                     "stated level for target coverage": {f"{q:g}": dict(zip(("level", "reached", "at floor"), level_for(curve, q))) for q in TARGETS},
                     "presence": "density >= 0.3 per mm^2 (ensemble_grid_check), the 42 named tracts pooled, weighted by the rescan's voxels"},
        "shapes": {"all named tracts": avg(shapes), "focus tracts (CST, AF, UF)": avg(shapes_f)},
        "_curve": curve}


def evaluate(train, test, term, pat):
    """The held-out subject's rescans of `term`, against the map fitted on `train`."""
    card = fit_term(train, term)
    if card is None or not test.tags(term):
        return None
    R = test.R
    out = {"rescans": test.tags(term)}
    out["fold map"] = {"subjects": card["subjects"], "streamline": card["streamline"]["observed"], "point": card["point"]["observed"],
                       "envelope: stated level for target": {q: x["level"] for q, x in card["envelope"]["stated level for target coverage"].items()},
                       "envelope: coverage at stated level": {f"{lv:g}": round(float(card["_curve"][np.flatnonzero(GRID == lv)[0]]), 4) for lv in LEVELS}}
    v, h = test.streamline(term)
    out["streamline"] = {"as stated": ece(v / R, h), "card": ece(apply_votes(card["streamline"], v, R), h),
                         "PAT16 map": ece(pat(v / R), h)}
    va, ha = test.streamline(term, "A")
    out["streamline, keeps A's tract (PAT16's event)"] = {"as stated": ece(va / R, ha), "PAT16 map": ece(pat(va / R), ha)}
    v, h = test.point(term)
    out["point"] = {"as stated": ece(v / R, h), "card": ece(apply_votes(card["point"], v, R), h)}
    curve = term_curve(test, term)
    env = {}
    for q in TARGETS:
        lv, reached, floor = level_for(card["_curve"], q)
        env[f"{q:g}"] = {"as stated: coverage at stated {:g}".format(q): round(float(curve[np.flatnonzero(GRID == round(q, 3))[0]]), 4),
                         "card: stated level used": lv, "card: reached in training": reached, "card: at floor": floor,
                         "card: coverage": round(float(curve[np.flatnonzero(GRID == lv)[0]]) if lv < 1 else float(term_shapes(test, term)["widest"]["coverage"]), 4)}
    out["envelope"] = env
    out["envelope: card's predicted coverage at each stated level, |error| (mean over GRID)"] = round(float(np.abs(card["_curve"] - curve).mean()), 4)
    sh = term_shapes(test, term)
    pred = card["shapes"]["all named tracts"]
    out["shapes: predicted (card) / observed coverage, size"] = {
        k: {"coverage": [pred[k]["coverage"], round(sh[k]["coverage"], 4)], "size": [pred[k]["size"], round(sh[k]["size"], 4)]}
        for k in ("core", "vorobev", "widest", "A alone")}
    pooled = test.pooled(term)
    if pooled:
        z = test.z
        pv = [(z[f"pool_s_votes_{t}"], z[f"pool_s_hit_{t}"]) for t in pooled]
        Rp = test.meta["pooled"][pooled[0]]["runs"]
        vv = np.concatenate([a[b >= 0] for a, b in pv]); hh = np.concatenate([b[b >= 0] for _, b in pv]).astype(bool)
        pc = np.mean([coverage_curve(z[f"pool_e_cnt_{t}"], z[f"pool_e_hit_{t}"]) for t in pooled], 0)
        out["pooled sessions (upper reference, as stated)"] = {
            "rescans": pooled, "runs": Rp, "streamline": ece(vv / Rp, hh),
            "envelope coverage at stated": {f"{q:g}": round(float(pc[np.flatnonzero(GRID == round(q, 3))[0]]), 4) for q in TARGETS}}
        out["pooled against the card, like for like"] = like_for_like(test, pooled, card, Rp)
    return out


def like_for_like(test, tags, card, Rp):
    """The card (on A's 17 runs) and the pooled ensemble (as stated) on the same seeds, A's seed set intersected with the
    pool's, each against its own mode; and envelope size (and precision) at matched held-out coverage, each curve
    interpolated to the same coverage rather than compared at the same target."""
    z, R = test.z, test.R
    if not all(f"pool_s_keys_{t}" in z for t in tags):
        return "the statements lack the pooled seed keys: collect again (--force)"
    cs, ch, ps, ph = [], [], [], []
    for t in tags:
        common, ia, ip = np.intersect1d(z["keys"], z[f"pool_s_keys_{t}"], return_indices=True)
        hp = z[f"pool_s_hit_{t}"][ip]
        rt = z[f"s_tract_{t}"][ia]
        ok = (hp >= 0) & (rt >= 0)
        cs.append(apply_votes(card["streamline"], z["s_votes"][ia][ok], R)); ch.append(rt[ok] == z["s_mode"][ia][ok])
        ps.append(z[f"pool_s_votes_{t}"][ip][ok] / Rp); ph.append(hp[ok] == 1)
    seeds = {"seeds (A's set and the pool's, the rescan labeling)": int(sum(len(x) for x in cs)),
             "card": ece(np.concatenate(cs), np.concatenate(ch)), "pooled, as stated": ece(np.concatenate(ps), np.concatenate(ph))}
    sizes = {}
    for name, cnt_key, hit_key in (("A's ensemble (the card's envelopes)", "e_cnt", "e_hit_{}"), ("pooled", "pool_e_cnt_{}", "pool_e_hit_{}")):
        rows = []
        for t in tags:
            cv, sz = curves(z[cnt_key.format(t)], z[hit_key.format(t)])
            rows.append({q: size_at(cv, sz, q) for q in MATCHED})
        sizes[name] = {f"{q:g}": (round(float(np.mean([r[q] for r in rows])), 4) if all(r[q] is not None for r in rows) else None) for q in MATCHED}
    return {"streamline, common seeds, each against its own mode": seeds,
            "envelope size (over the rescan's tract) at matched held-out coverage": sizes}


def cross_term(train, test, pat):
    """The same-session card (fitted on train) applied to the test subject's other-session rescans."""
    card = fit_term(train, "same session")
    if card is None or not test.tags("other session"):
        return None
    R = test.R
    v, h = test.streamline("other session")
    vp, hp = test.point("other session")
    curve = term_curve(test, "other session")
    return {"rescans": test.tags("other session"),
            "streamline": {"as stated": ece(v / R, h)["ece"], "same-session card": ece(apply_votes(card["streamline"], v, R), h)["ece"]},
            "point": {"as stated": ece(vp / R, hp)["ece"], "same-session card": ece(apply_votes(card["point"], vp, R), hp)["ece"]},
            "envelope coverage at target, with the same-session card's level": {
                f"{q:g}": round(float(curve[np.flatnonzero(GRID == level_for(card["_curve"], q)[0])[0]]), 4) for q in TARGETS}}


def per_rescan(subj):
    """Per rescan: the mean stated share and how often it gives the mode (streamline, point), coverage at stated levels."""
    z, R, out = subj.z, subj.R, {}
    named = z["p_mode"] >= 0
    for t, r in subj.meta["rescans"].items():
        rt = z[f"s_tract_{t}"]
        ok = rt >= 0
        cv = coverage_curve(z["e_cnt"], z[f"e_hit_{t}"], np.array([0.8, 0.9]))
        out[t] = {"term": r["term"], "registration (rotation deg, translation mm, NCC)": [r["rotation_deg"], r["translation_mm"], r["ncc_after"]],
                  "streamline: stated, observed (gives the mode)": [round(float((z["s_votes"][ok] / R).mean()), 4), round(float((rt[ok] == z["s_mode"][ok]).mean()), 4)],
                  "streamline: rescan labels the seed": round(float(ok.mean()), 4),
                  "point: observed": round(float((z[f"p_main_{t}"][named] == z["p_mode"][named]).mean()), 4),
                  "envelope coverage at stated 0.8, 0.9": [round(float(x), 4) for x in cv]}
    return out


def fold_stability(held, term):
    """From the folds' stored maps: the range over folds of the streamline and point maps (k = 9, 13, 17), the levels
    for the targets and the coverage at stated levels."""
    maps = [h[term]["fold map"] for h in held.values() if h and h.get(term)]
    if not maps:
        return None
    rng = lambda xs: [round(float(min(xs)), 4), round(float(max(xs)), 4)]  # noqa: E731
    return {"folds": len(maps),
            "streamline map at k (min, max)": {k: rng([m["streamline"][k] for m in maps]) for k in (9, 13, 17)},
            "point map at k (min, max)": {k: rng([m["point"][k] for m in maps]) for k in (9, 13, 17)},
            "envelope level for target (min, max)": {q: rng([m["envelope: stated level for target"][q] for m in maps]) for q in maps[0]["envelope: stated level for target"]},
            "envelope coverage at stated level (min, max)": {lv: rng([m["envelope: coverage at stated level"][lv] for m in maps]) for lv in maps[0]["envelope: coverage at stated level"]}}


def phantom_term():
    """Step 4's provisional accuracy term, from results/robust_phantom.json."""
    r = json.loads((HERE / "results/robust_phantom.json").read_text())
    em = r["envelopes, count-matched"]
    pool = em["pooled"]["labeled"]
    keys = ["A alone", "core (at least half)", "widest"] + [f"envelope {lv:g}" for lv in em["levels"]]
    reps = [v["mean_f1_all21"] for v in r["rescore"]["replicates"]["runs"].values()]
    return {
        "apply": False,
        "how to read": "not applied to any statement. Its gaps are between the pipeline's envelopes and a bias-free ensemble at the "
                       "same uncalibrated stated levels (0.8-0.95), not at the card's levels; it must not be added to or stacked on the "
                       "card's maps, which are about rescans",
        "status": "provisional and protocol-mismatched: one simulated brain (ISMRM 2015, Fiberfox), b = 1000 with 32 directions "
                  "(this card's protocol: b1000 x40 + b2000 x56, tracked at 2000), its own readout and noise (DW SNR about 7.6)",
        "event": "the ground truth's own density presence (count-matched to each run's streamlines), not a rescan",
        "envelope coverage of the truth (labeled streamlines, pooled over bundles)": {
            k: {"pipeline's 17 runs": pool["runs"][k]["coverage"],
                **({"bias-free ensemble (ground-truth subsamples)": pool["perfect ensemble"][k]["coverage"],
                    "gap": round(pool["perfect ensemble"][k]["coverage"] - pool["runs"][k]["coverage"], 4)} if k in pool["perfect ensemble"] else {})}
            for k in keys},
        "2023 ROI scoring, today": {"mean F1 (21 bundles)": r["configs"]["today"]["all streamlines"]["mean_f1_all21"],
                                    "16 replicates' range": [min(reps), max(reps)],
                                    "valid bundles": f"{r['configs']['today']['all streamlines']['VB']} of {r['configs']['today']['all streamlines']['of']}",
                                    "valid streamlines": r["configs"]["today"]["all streamlines"]["VS_ratio"]},
        "labels against the truth": {"valid streamlines with a counterpart labeled as it": r["configs"]["today"]["labels against the truth"]["valid with a counterpart: labeled as it"]},
        "reading": "the envelopes state reproducibility, not accuracy: on the phantom they cover 17-18 points less of the truth than a "
                   "bias-free ensemble does at stated 0.8-0.95 (the gaps above). A statement about the anatomy needs this bias term; "
                   "without it the card's statements are about rescans only",
        "source": "bench/results/robust_phantom.json (NOTES 2026-10-17, Robustness step 4)"}


def protocol():
    return {
        "dataset": "MASiVar (OpenNeuro ds003416, CC0), cohort III, scanner B: Philips Achieva 3T (Cai et al., MRM 2021)",
        "shells": {"b1000": {"directions": 40, "series": "acq-b1000n40, APP"}, "b2000": {"directions": 56, "series": "acq-b2000n56, APP"}},
        "b0s": {"forward": "16 (interleaved, stated b = 0.1; each series' last volume, stated b = 0, is a real b0 at 0.98-1.0 of the others and kept)",
                "reversed": "1 (the APA series, named b1000 n3: its last volume, 0.89-0.95 of the forward b0s; its three b1000s, 0.38-0.45, dropped)"},
        "name in BIDS": "r21x21x22",
        "readout": "not in the sidecars: nominal 0.05 s (_masivar.NOMINAL_READOUT_S); the field in Hz is on a nominal scale, safe within this protocol",
        "tracked shell": 2000,
        "recognized by": {"checked by": "calibration_runs.load_checked (protocol_check), on the scan as _masivar loads it", **PROTOCOL}}


def fit(args):
    names = args.subjects or list(CARD)
    subjects = []
    for n in names:
        p = Records(n).dir / f"statements_R{args.runs + 1}.npz"
        if p.exists():
            subjects.append(Subject(p))
        else:
            print(f"{n}: no {p.name}, left out", flush=True)
    if not subjects:
        sys.exit("no statements")
    pat = pat16_table()
    terms = {t: fit_term(subjects, t) for t in TERMS}
    held = {}
    for s in subjects:
        rest = [x for x in subjects if x is not s]
        held[s.name] = {t: evaluate(rest, s, t, pat) for t in TERMS} if rest else None
    summary = {}
    for t in TERMS:
        rows = [h[t] for h in held.values() if h and h.get(t)]
        if not rows:
            continue
        mean = lambda f: round(float(np.mean([f(r) for r in rows])), 4)  # noqa: E731
        summary[t] = {"subjects held out": len(rows),
                      "streamline ECE: as stated / card / PAT16 map": [mean(lambda r: r["streamline"][k]["ece"]) for k in ("as stated", "card", "PAT16 map")],
                      "point ECE: as stated / card": [mean(lambda r: r["point"][k]["ece"]) for k in ("as stated", "card")],
                      "envelope coverage at target, as stated / card": {
                          q: [mean(lambda r: r["envelope"][q][f"as stated: coverage at stated {q}"]), mean(lambda r: r["envelope"][q]["card: coverage"])]
                          for q in (f"{x:g}" for x in TARGETS)}}
    for t in TERMS:
        if t in summary:
            summary[t]["fold stability"] = fold_stability(held, t)
    cross = {s.name: cross_term([x for x in subjects if x is not s], s, pat) for s in subjects if s.tags("other session")}
    cross = {k: v for k, v in cross.items() if v}
    if cross:
        cm = lambda f: round(float(np.mean([f(v) for v in cross.values()])), 4)  # noqa: E731
        summary["the same-session card on other-session rescans"] = {
            "subjects": len(cross), "streamline ECE": cm(lambda v: v["streamline"]["same-session card"]),
            "point ECE": cm(lambda v: v["point"]["same-session card"]),
            "envelope coverage at target": {q: cm(lambda v: v["envelope coverage at target, with the same-session card's level"][q]) for q in (f"{x:g}" for x in TARGETS)}}
    # provenance and settings, from every subject
    import importlib.metadata as md
    from datetime import datetime, timezone
    from tractline.labelers.rapidparc import HASHES
    settings = {}
    for s in subjects:
        log = Records(s.name).dir / "runs.json"
        if log.exists():
            settings[s.name] = json.loads(log.read_text())["created"]["pipeline"]
    canon = {json.dumps(v, sort_keys=True) for v in settings.values()}
    assert len(canon) == 1, f"the subjects' pipeline settings differ: {sorted(settings)}"
    settings_json = canon.pop()
    if "config" in json.loads(settings_json) and json.loads(settings_json)["config"]["sha256"] is None:
        raise ValueError("the pipeline's config is not hashable (a caller's object in a slot): a card cannot key on it")
    settings_hash = hashlib.sha256(settings_json.encode()).hexdigest()
    R = subjects[0].R
    assert all(s.R == R for s in subjects)
    versions = {}
    for lib in ("numpy", "scipy", "torch", "dipy", "nibabel"):
        try:
            versions[lib] = md.version(lib)
        except md.PackageNotFoundError:
            versions[lib] = None
    stmt = {s.name: Records(s.name).dir / f"statements_R{args.runs + 1}.npz" for s in subjects}
    card = {
        "schema": f"tractline.calibration_card/{CARD_VERSION}",
        "card id": f"masivar-cIII-scannerB_b1000x40-b2000x56_2.1mm_R{R}_{settings_hash[:12]}",
        "card version": CARD_VERSION,
        "created (UTC)": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": protocol(),
        "pipeline": {"fit": git_state(), "bench module sha256": module_hashes(),
                     "settings": json.loads(settings_json), "settings sha256": settings_hash,
                     "settings recorded by (subjects whose card folder has a runs.json)": sorted(settings),
                     "libraries": versions,
                     "labeler weights": {"model": "rapidparc", "sha256": HASHES["rapidparc"], "check": "verified on every load (labelers/rapidparc._verified)"},
                     "R": R, "ensemble": f"A and {R - 1} bootstrap replicates ({FLIP} signs, FWHM 4, b0s added), labeler draw 0",
                     "statement files sha256": {n: sha256(p) for n, p in stmt.items()},
                     "provenance": {s.name: provenance(s.name, list(s.meta["rescans"]), args.runs) for s in subjects}},
        "validity": {"match": {"protocol": PROTOCOL, "R": R, "pipeline settings sha256": settings_hash,
                               **({"pipeline config sha256": json.loads(settings_json)["config"]["sha256"]}
                                  if "config" in json.loads(settings_json) else {}),
                               "labeler weights sha256": HASHES["rapidparc"], "terms": list(TERMS)},
                     "population": "healthy adults (MASiVar cohort III)", "protocol": "this protocol only (see protocol)",
                     "pipeline": "these settings and R only; any option that changes the pipeline (denoising, tracker settings, labeler grouping) needs its own card",
                     "events": {"same session": "a rescan in the same session, minutes later (MASiVar scan 2)",
                                "other session": "a rescan in another session (a separate visit) on the same scanner; the interval between sessions was not checked"},
                     "outside": "without a matching card the product says \"run-to-run only\"",
                     "not covered": "accuracy (what the anatomy is): see accuracy (provisional)"},
        "terms": {t: ({k: v for k, v in c.items() if not k.startswith("_")} if c else None) for t, c in terms.items()},
        "accuracy (provisional)": phantom_term(),
        "held-out evaluation (leave one subject out)": {"summary": summary, "per subject": held, "the same-session card on other-session rescans": cross,
                                                         "per rescan": {s.name: per_rescan(s) for s in subjects}},
        "subjects": {s.name: {k: s.meta[k] for k in ("pair", "R", "rescans", "pooled", "points", "seeds labeled in all R runs")} for s in subjects}}
    out = Path(args.out)
    out.write_text(json.dumps(card, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({"summary": summary, "terms": {t: (c and {k: c[k] for k in ("subjects", "rescans")}) for t, c in terms.items()}}, indent=1))
    print(f"wrote {out}")


# ----------------------------------------------------------------------------------------------- tests (synthetic)

def test_pav():
    assert np.allclose(pav([0.1, 0.2, 0.3], [1, 1, 1]), [0.1, 0.2, 0.3])
    assert np.allclose(pav([0.3, 0.1, 0.5], [1, 1, 2]), [0.2, 0.2, 0.5])
    y = pav(np.random.default_rng(0).random(50), np.ones(50))
    assert (np.diff(y) >= -1e-12).all()


def synthetic_subject(rng, R, g, offset, n=20000):
    """votes (1..R) and hits drawn with P(hit | k) = g(k / R) + offset"""
    v = rng.integers(max(1, R // 4), R + 1, n)
    h = rng.random(n) < np.clip(g(v / R) + offset, 0, 1)
    return v, h


def test_streamline_card():
    """Eight synthetic subjects whose rescans give the mode less often than stated: the card, fitted on seven, is
    calibrated on the eighth where the stated share is not, and it recovers the true map."""
    rng = np.random.default_rng(1)
    R = 17
    g = lambda f: 0.1 + 0.75 * f ** 1.5  # noqa: E731
    subs = [synthetic_subject(rng, R, g, rng.normal(0, 0.01)) for _ in range(8)]
    for i, (v, h) in enumerate(subs):
        m = fit_votes([s for j, s in enumerate(subs) if j != i], R)
        p = apply_votes(m, v, R)
        assert ece(p, h)["ece"] < 0.02 < 0.1 < ece(v / R, h)["ece"], (ece(p, h)["ece"], ece(v / R, h)["ece"])
    m = fit_votes(subs, R)
    k = np.arange(R // 4, R + 1)
    assert np.abs(np.array(m["observed"])[k] - g(k / R)).max() < 0.03
    assert (np.diff([x for x in m["observed"] if x is not None]) >= -1e-12).all()


def test_equal_subject_weights():
    """A subject with ten times the rows does not outweigh another."""
    R = 4
    a = (np.full(1000, 4), np.ones(1000, bool))
    b = (np.full(100, 4), np.zeros(100, bool))
    assert abs(fit_votes([a, b], R)["observed"][4] - 0.5) < 1e-9


def test_envelopes_match_existing():
    """The histogram route gives ensemble_pooled's envelope thresholds and ensemble_vorobev's cuts on explicit voxels,
    and the coverage curve nests; the card's level for a target reaches it on the training curve."""
    from ensemble_pooled import thresholds
    from ensemble_vorobev import shapes
    rng = np.random.default_rng(2)
    R = 17
    V = np.zeros((NAMED, 3000), np.uint8)
    T = np.zeros_like(V, bool)
    for k in range(NAMED):
        q = np.clip(rng.normal(rng.uniform(2, 15), 5, 3000), 0, R).astype(np.uint8)
        q[rng.random(3000) < 0.5] = 0
        V[k] = q
        T[k] = rng.random(3000) < (q / R) ** 0.7 * 0.9
    cnt, hit = vote_hist(V, T, R)
    for k in range(NAMED):
        assert thresholds_from_counts(cnt[k], LEVELS) == thresholds(V[k], R, LEVELS)
        sh = shapes(V[k], R)
        assert vorobev_cut(cnt[k]) == sh["vorobev"]
    curve = coverage_curve(cnt, hit)
    assert (np.diff(curve) >= -1e-12).all()
    env, _ = shape_cuts(cnt, np.array([0.8]))
    direct = sum((T[k] & (V[k] >= env[k, 0])).sum() for k in range(NAMED)) / T.sum()
    assert abs(coverage_curve(cnt, hit, np.array([0.8]))[0] - direct) < 1e-12
    lv, ok, floor = level_for(curve, 0.8)
    assert ok and not floor and curve[np.flatnonzero(GRID == lv)[0]] >= 0.8 and (lv == GRID[0] or curve[np.flatnonzero(GRID == lv)[0] - 1] < 0.8)
    s = shape_scores(cnt, hit)
    w = s["widest"]
    assert abs(w["coverage"] - (T & (V > 0)).sum() / T.sum()) < 1e-12 and abs(w["size"] - (V > 0).sum() / T.sum()) < 1e-12


def test_level_for_floor():
    """A target the first level already reaches is flagged at floor, not silently reached."""
    curve = np.linspace(0.6, 0.99, len(GRID))
    assert level_for(curve, 0.5) == (float(GRID[0]), True, True)
    lv, ok, floor = level_for(curve, 0.9)
    assert ok and not floor and lv > GRID[0]
    assert level_for(curve, 0.999) == (1.0, False, False)


def test_pat16_table():
    """PAT16's vote table as a function of the vote fraction: its values at k / 11, linear between, clamped outside."""
    t = json.loads((HERE / "results/ensemble_double_smooth.json").read_text())["recalibration for rescans"]["votes: P(keep | votes of R)"]
    tab, R = np.array(t["table"], float), t["R"]
    pat = pat16_table()
    for k in range(1, R + 1):
        assert abs(pat(k / R) - tab[k]) < 1e-12
    assert abs(pat(6.5 / R) - (tab[6] + tab[7]) / 2) < 1e-12
    assert pat(0.0) == tab[1] and pat(0.5 / R) == tab[1] and pat(1.2) == tab[R]


def synthetic_statements(rng, name, offset, R=17, n=4000, npts=300):
    """A Subject with collect's arrays, synthetic: a same-session and an other-session rescan."""
    s = object.__new__(Subject)
    s.name, s.R = name, R
    s.meta = {"subject": name, "R": R, "rescans": {"within": {"term": "same session"}, "session2": {"term": "other session"}}, "pooled": {}}
    votes = rng.integers(5, R + 1, n)
    mode = rng.integers(0, NAMED, n).astype(np.int16)
    z = {"keys": np.arange(n), "s_mode": mode, "s_votes": votes, "s_A": mode, "s_Avotes": votes,
         "p_mode": rng.integers(-1, NAMED, npts).astype(np.int16), "p_votes": rng.integers(5, R + 1, npts)}
    V = np.clip(rng.normal(8, 6, (NAMED, 2000)), 0, R).astype(np.uint8)
    z["e_cnt"] = vote_hist(V, V > 0, R)[0]
    z["e_Asize"] = (V > 8).sum(1)
    for t in ("within", "session2"):
        hit = rng.random(n) < np.clip(0.2 + 0.7 * votes / R + offset, 0, 1)
        z[f"s_tract_{t}"] = np.where(hit, mode, (mode + 1) % NAMED).astype(np.int16)
        z[f"p_main_{t}"] = np.where(rng.random(npts) < 0.8, z["p_mode"], -1).astype(np.int16)
        T = rng.random(V.shape) < (V / R) ** 0.7
        z[f"e_hit_{t}"] = vote_hist(V, T, R)[1]
        z[f"e_Ahit_{t}"] = ((V > 8) & T).sum(1)
    s.z = z
    return s


def test_evaluate_excludes_test_subject():
    """The held-out fold's map is fitted on the other subjects only: changing the test subject leaves it unchanged,
    changing a training subject does not."""
    rng = np.random.default_rng(3)
    subs = [synthetic_statements(rng, f"s{i}", rng.normal(0, 0.02)) for i in range(4)]
    pat = pat16_table()
    for term in TERMS:
        ref = evaluate(subs[1:], subs[0], term, pat)["fold map"]
        changed = synthetic_statements(np.random.default_rng(9), "s0", -0.3)
        assert evaluate(subs[1:], changed, term, pat)["fold map"] == ref
        other = synthetic_statements(np.random.default_rng(10), "s1", 0.3)
        assert evaluate([other] + subs[2:], subs[0], term, pat)["fold map"] != ref


def run_tests(args):
    for f in (test_pav, test_streamline_card, test_equal_subject_weights, test_envelopes_match_existing, test_level_for_floor,
              test_pat16_table, test_evaluate_excludes_test_subject):
        f()
        print(f.__name__, "ok", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("collect", "fit", "test"))
    ap.add_argument("--subject", choices=sorted(CARD), help="collect: the subject")
    ap.add_argument("--subjects", nargs="+", choices=sorted(CARD), help="fit: the subjects (default: all with statements)")
    ap.add_argument("--runs", type=int, default=16, help="A's replicates used (R = runs + 1)")
    ap.add_argument("--force", action="store_true", help="collect: rebuild an existing statements file (derived, not a record)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.what == "collect" and not args.subject:
        ap.error("collect needs --subject")
    {"collect": collect, "fit": fit, "test": run_tests}[args.what](args)
