"""MASiVar cohort II, the same-site cross-scanner rung: one Philips Achieva's 17-run ensemble against the same person's
scans on the OTHER Philips Achieva at the same site (site 1, scanners A and B). How much does coverage fall beyond the
same-scanner other-session gap when only the scanner unit changes? (The same kind of measurement as the site 3
Siemens rung, calibration_crosssite.py, and the cohort III card's other-session term; docs/ismrm2027-abstract.md.)

    uv run --group dipy --group rankfield bench/calibration_crossscanner.py run --source B32 [--subject cIIs01 ...] [--targets xA1] [--smoke]
    uv run --group dipy --group rankfield bench/calibration_crossscanner.py collect --source B32 [--subject ...] [--smoke] [--force]
    uv run --group dipy --group rankfield bench/calibration_crossscanner.py evaluate [--dry-run --out <elsewhere>]

The directions (DIRECTIONS; the ensemble's protocol -> the target scans' protocol, both sessions of the other scanner):
  primary, a fixed protocol (b1000n32 at 2.5 mm): B32 -> A32 scans (s1Ax1, s1Ax2) and A32 -> B32 scans (s1Bx1, s1Bx2);
  secondary, CONFOUNDED: B2000 -> A2000r19 scans and A2000r19 -> B2000 scans. b2000n96 on both, but 2.5 mm on B and
    1.89 x 1.89 x 2.2 mm on A, and A2000r19's runs have about twice the streamlines (every mask voxel a seed); a target
    is resampled onto the ensemble scan's grid (up or down) and tracked there.
The ensemble (A, its companions and its 16 replicates) is calibration_cohortII's record folder of the source protocol,
read, never written. Each target scan is loaded and checked by calibration_cohortII (load_checked, its own protocol's
descriptor), corrected on its grid with its own reversed b0, registered rigidly onto the ensemble scan by registration
mode 3 (calibration_cohortII.register_multi: starts (a) mode 1's pipeline, (b) mode 2's FA stage, (c) where the reverse
pair exists (the target protocol's ensemble against the source's first-session scan: the x?1 pairs), its transform
inverted; the best final b0 NCC; QC flag below calibration_cohortII.NCC_QC), resampled onto the ensemble's grid
(onto_reference, b-vectors turned) and tracked once at the source protocol's shell (calibration_cohortII.Runner.one).
Records: DATA/ensemble/<subject>/card/<subject>-<source>-x<target scanner>/ (x<T>1reg3_onA.npz, x<T>2reg3_onA.npz,
runs.json, statements_R17_reg3.npz); --smoke: ...-smoke/. The records refuse to be written anywhere else.

collect: calibration_cohortII.collect's statements for the target scans (term TERM_X, both sessions), A's arrays checked
equal to the source protocol's own statements (mode 3, calibration_cohortII.statements_file).

evaluate (no pipeline runs) -> results/calibration_crossscanner.json: per direction and subject, at stated levels
0.5-0.95 (calibration_cohortII.SCHEMA_LEVELS): the cross-scanner targets as stated (both sessions pooled, each subject
weighted equally, and per session); the same ensemble's same-scanner other-session rescan as stated (calibration_
cohortII's mode 3 statements, checked equal to calibration_cohortII.json); the difference (the extra gap from changing
the scanner unit); the same-scanner LOSO map (fitted without the target person) applied to the cross-scanner targets; a
cross-scanner LOSO map; ECE beside them; the registration QC; and "fig6 (COHORT2_SCHEMA shape)" (protocol and transfer
"<source> -> <target> scans (same site)", as calibration_crosssite.json's block).

Imports calibration_cohortII and calibration_crosssite (definitions only on import) and writes neither.
"""
import argparse, copy, json, os, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tractline import pipeline as P
from tractline import ensemble as E
from tractline.data import DATA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_cohortII as C2   # noqa: E402
import calibration_crosssite as XS   # noqa: E402  (map_on, stated)
import calibration_runs as CR   # noqa: E402
import calibration_card as CC   # noqa: E402
from ensemble_retest import onto_reference, mean_b0  # noqa: E402

SUBJECTS = ("cIIs01", "cIIs02", "cIIs03", "cIIs04")
DIRECTIONS = {"B32": "A32", "A32": "B32", "B2000": "A2000r19", "A2000r19": "B2000"}
PRIMARY, SECONDARY = ("B32", "A32"), ("B2000", "A2000r19")
CONFOUNDED = ("resolution (2.5 mm on scanner B against 1.89 x 1.89 x 2.2 mm on A, the target resampled onto the ensemble's "
              "grid) and streamline count (A2000r19's runs have about twice as many: every mask voxel a seed)")
TERM_X = "cross scanner (same site)"
REG = 3
RUNS = CR.RUNS
LEVELS = C2.SCHEMA_LEVELS
OUT = HERE / "results/calibration_crossscanner.json"


def scanner(key):
    return C2.PROTOCOLS[key]["scanner"][-1]


def name_of(source):
    """The fig6 name: "B32 -> A32 scans (same site)" (A2000r19 is "A2000", as COHORT2_SCHEMA names it)."""
    return f"{C2.SCHEMA_NAME[source]} -> {C2.SCHEMA_NAME[DIRECTIONS[source]]} scans (same site)"


def sha(path):
    return CR.sha256(Path(path).resolve())


def module_hashes():
    return {"calibration_crossscanner.py sha256": sha(__file__), "calibration_cohortII.py sha256": C2.module_sha256(),
            "calibration_crosssite.py sha256": sha(HERE / "calibration_crosssite.py"), "_register_fa.py sha256": sha(HERE / "_register_fa.py")}


class Records(CR.Records):
    """<subject>-<source>-x<T>[-smoke]/ (written) on top of calibration_cohortII's source folder (read, never written)."""
    def __init__(self, subject, source, smoke=False):
        self.subject, self.source, self.target = subject, source, DIRECTIONS[source]
        T = scanner(self.target)
        src = C2.pairs(source)[subject]
        self.pair = CR.Pair(f"{subject}-{source}-x{T}", src.ref, {}, {f"x{T}{i + 1}": (ses, 1) for i, ses in enumerate(C2.SESSIONS[T])},
                            notes=(f"the {source} ensemble (A: {src.ref[0]}) against the same person's {self.target} scans on scanner {T}",
                                   "same site, the other Philips Achieva" + ("" if source in PRIMARY else f"; CONFOUNDED: {CONFOUNDED}")))
        self.dir = DATA / "ensemble" / subject / "card" / (self.pair.name + ("-smoke" if smoke else ""))
        self.reuse = C2.Records(subject, source).dir

    def new(self, name, ext=".npz"):
        assert f"-{self.source}-x" in self.dir.name, self.dir
        return super().new(name, ext)


def input_hashes(subject, target):
    out = {}
    for ses in C2.SESSIONS[scanner(target)]:
        f = C2.series_files(subject, ses, target)
        for p in [p for _, p in f["APP"]] + [f["APA"]]:
            for q in (p, C2.M._side(p, ".bval"), C2.M._side(p, ".bvec")):
                out[str(Path(q).relative_to(C2.M.ROOT))] = sha(q)
    return out


class Runner(C2.Runner):
    """calibration_cohortII.Runner's one() (tracked at the source protocol's shell), records in this script's folder."""
    def __init__(self, subject, source, smoke=False):
        self.key, self.shell = source, C2.PROTOCOLS[source]["tracked shell"]
        self.R = Records(subject, source, smoke)
        self.subject, self.pair = subject, self.R.pair
        self.R.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.R.dir / "runs.json"
        self.log = json.loads(self.log_path.read_text()) if self.log_path.exists() else {}
        self.log.setdefault("created", {"git": CR.git_state(), **module_hashes(), "pipeline": C2.pipeline_settings(source),
                                        "source protocol": {"key": source, **C2.PROTOCOLS[source]},
                                        "target protocol": {"key": self.R.target, **C2.PROTOCOLS[self.R.target]},
                                        "confounded": None if source in PRIMARY else CONFOUNDED, "smoke": smoke,
                                        "ensemble (A)": {"session": self.pair.ref, "records read from": str(self.R.reuse.relative_to(DATA))},
                                        "targets": {t: {"session": s, "term": TERM_X} for t, s in self.pair.rescans.items()},
                                        "inputs sha256 (target scans)": input_hashes(subject, self.R.target), "registration mode": REG})
        self.labeler = None

    def reverse_start(self, tag):
        """Start (c): the reverse pair's transform (the target protocol's ensemble against this source's first-session
        scan), for the x?1 targets, if that pair has been registered."""
        if not tag.endswith("1"):
            return None, None
        rev = Records(self.subject, self.R.target).dir / "runs.json"
        key = f"{C2.rescan_tag('x' + scanner(self.key) + '1', REG)}_scan"
        if rev.exists() and key in (log := json.loads(rev.read_text())):
            return np.array(log[key]["to_A"]), f"{rev.parent.name} {key}"
        return None, None

    def rescan_x(self, A, ref_b0, tag):
        name = C2.rescan_tag(tag, REG)
        if self.R.have(f"{name}_onA"):
            print(f"{self.subject} {self.key} {name}_onA exists", flush=True)
            return
        ses = self.pair.rescans[tag][0]
        B = C2.load_checked(self.subject, ses, self.R.target)
        timer, t0 = P.Timer(), time.time()
        with P.exact_float32(CR.DEVICE):
            cb = P.correct(B, timer, CR.DEVICE)
        P.release_memory(CR.DEVICE)
        other, other_from = self.reverse_start(tag)
        r = C2.register_multi(B, mean_b0(cb.dwi, B.bval), A, ref_b0, other)
        pts = C2.mask_world(np.load(self.R.path("A_mask", ".npy")), A.affine)
        e = C2.reg3_entry(r, None, tag, pts, other_from)
        e.update({"decision": "mode 3 (no earlier registration of this pair)", "name": B.name, "term": TERM_X, "target session": ses,
                  "target protocol": self.R.target, "series": B.series, "shape": list(B.dwi.shape), "residual_left": cb.residual_left,
                  "field_hz_p99_abs": round(float(np.quantile(np.abs(cb.field_hz), 0.99)), 2), "correct_seconds": timer.seconds,
                  "correct_register_s": round(time.time() - t0, 1)})
        dwi, bvec = onto_reference(cb.dwi, B.bvec, B.affine, np.array(e["to_A"]), A.affine, A.dwi.shape[:3])
        self.log[f"{name}_scan"] = e
        self.save_log()
        print(self.subject, self.key, "->", self.R.target, ses, name, "best", e["best"], "b0 ncc", e["best b0 ncc"], "QC flag", e["QC"]["flag"],
              {x: c["b0 ncc"] for x, c in e["candidates"].items()}, f"{e['rotation_deg']} deg {e['translation_mm']} mm", flush=True)
        self.one(replace(A, dwi=dwi, bval=B.bval, bvec=bvec, b0s=None), f"{name}_onA", corrected=dwi)


def source_ready(subject, source):
    return C2.statements_file(C2.Records(subject, source).dir, RUNS + 1, REG).exists()


def run(args):
    for sub in args.subject:
        if not source_ready(sub, args.source):
            print(f"{sub} {args.source}: the source's mode 3 statements are missing: skipped", flush=True)
            continue
        t0 = time.time()
        rn = Runner(sub, args.source, args.smoke)
        A = C2.load_checked(sub, rn.pair.ref[0], args.source)
        ref_b0 = np.load(rn.R.path("A_meanb0", ".npy"))
        for tag in (args.targets or list(rn.pair.rescans)):
            rn.rescan_x(A, ref_b0, tag)
        print(f"{sub} {args.source} -> {DIRECTIONS[args.source]} done {time.time() - t0:.0f}s, {CR.free_gb():.1f} GB free", flush=True)


# ----------------------------------------------------------------------------------------------- the statements

def collect(args):
    """calibration_cohortII.collect's statements for the cross-scanner targets against the source protocol's A ensemble."""
    import nibabel as nib
    from ensemble_grid_check import Grid
    for sub in args.subject:
        t0 = time.time()
        R = Records(sub, args.source, args.smoke)
        out = R.dir / C2.statements_name(args.runs + 1, REG)
        if out.exists() and not args.force:
            raise FileExistsError(f"{out} exists (--force to rebuild it)")
        if not (R.dir / "runs.json").exists():
            print(f"{sub} {args.source}: no records: skipped", flush=True)
            continue
        log = json.loads((R.dir / "runs.json").read_text())
        tags = [C2.rescan_tag(t, REG) for t in R.pair.rescans if R.have(f"{C2.rescan_tag(t, REG)}_onA")]
        if len(tags) < len(R.pair.rescans) and not args.smoke:
            raise FileNotFoundError(f"{sub} {args.source}: targets {tags} of {list(R.pair.rescans)} only (a partial collect only with --smoke)")
        paths = R.ensemble("", args.runs)
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{sub}: {len(missing)} of A's {len(paths)} runs missing, the first {missing[0]}")
        img = nib.load(C2.series_files(sub, R.pair.ref[0], args.source)["APP"][0][1])
        A0 = E.RunRecord.load(paths[0], logp=False)
        grid = Grid("dwi", img.affine, A0.shape)
        mask = np.load(R.path("A_mask", ".npy"))
        cent = CC.centers(mask, A0.counts, A0.shape)
        Mx = CC.sphere_matrix(cent, A0.shape, grid.h)
        keys = A0.keys
        print(f"{sub} {args.source}: R = {len(paths)}, targets {tags}, {len(cent)} points", flush=True)
        A = CC.Scan(paths, keys, grid, Mx, t0)
        Rn = A.R
        tA = A.T[0]
        s_ok = (A.T >= 0).all(0)
        s_mode, s_votes = CC.mode_votes(A.T[:, s_ok], range(E.N_TRACTS))
        s_Avotes = (A.T[:, s_ok] == tA[s_ok]).sum(0).astype(np.int16)
        p_mode, p_votes = CC.mode_votes(A.P, range(-1, CC.NAMED))
        cnt, _ = CC.vote_hist(A.votes, A.own, Rn)
        arrays = {"keys": keys[s_ok], "s_A": tA[s_ok], "s_mode": s_mode, "s_votes": s_votes, "s_Avotes": s_Avotes,
                  "centers": cent, "p_mode": p_mode, "p_votes": p_votes, "e_cnt": cnt, "e_Asize": A.own.sum(1).astype(np.int64)}
        for tag in tags:
            k, t, pres, main = CC.run_stats(R.path(f"{tag}_onA"), grid, Mx)
            arrays[f"s_tract_{tag}"] = CC.on_keys(keys, k, t)[s_ok]
            arrays[f"p_main_{tag}"] = main
            arrays[f"e_hit_{tag}"] = CC.vote_hist(A.votes, pres, Rn)[1]
            arrays[f"e_Ahit_{tag}"] = (A.own & pres).sum(1).astype(np.int64)
            print(f"  {tag}_onA {time.time() - t0:.0f}s", flush=True)
        src = C2.statements_file(R.reuse, Rn, REG)
        same = None
        if src.exists():
            z = np.load(src)
            same = {n: bool(np.array_equal(z[n], arrays[n])) for n in ("keys", "s_A", "s_mode", "s_votes", "s_Avotes", "centers", "p_mode", "p_votes", "e_cnt", "e_Asize")}
            assert all(same.values()), f"{sub}: A's arrays differ from {src}: {same}"
        meta = {"subject": sub, "protocol": args.source, "target protocol": R.target, "pair": R.dir.name, "R": Rn,
                "runs": [str(p.relative_to(DATA)) for p in paths], "registration mode": REG,
                "rescans": {t: {"term": TERM_X, "scan": log[f"{t}_scan"]["name"], "target session": log[f"{t}_scan"]["target session"],
                                "rotation_deg": log[f"{t}_scan"]["rotation_deg"], "translation_mm": log[f"{t}_scan"]["translation_mm"],
                                "ncc_after": log[f"{t}_scan"]["ncc_after"], "registration mode": REG, "QC flag": log[f"{t}_scan"]["QC"]["flag"],
                                "record": f"{t}_onA"} for t in tags},
                "pooled": {}, "points": int(len(cent)), "seeds labeled in all R runs": int(s_ok.sum()),
                "A's arrays equal to the source's statements": {"file": str(src.relative_to(DATA)), "equal": same} if same else None,
                "records git at creation": log["created"]["git"], "collect git": CR.git_state(), **module_hashes(),
                "seconds": round(time.time() - t0, 1)}
        tmp = out.with_name(out.stem + ".partial.npz")
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
        os.replace(tmp, out)
        print(json.dumps({k: meta[k] for k in ("pair", "R", "rescans", "points")}, indent=1), flush=True)


# ----------------------------------------------------------------------------------------------- evaluation

def view(s, tag):
    """A subject's statements restricted to one target (per-session numbers)."""
    v = copy.copy(s)
    v.meta = {**s.meta, "rescans": {tag: s.meta["rescans"][tag]}}
    return v


def mean_curve(rows):
    return [round(float(x), 4) for x in np.mean(rows, 0)] if rows else None


def pick(xs, qs=(0.8, 0.9, 0.95)):
    return {f"{q:g}": xs[int(np.flatnonzero(np.isclose(LEVELS, q))[0])] for q in qs}


def evaluate_direction(source, pat, dry_run, cohort2):
    tgt = DIRECTIONS[source]
    xf = {sub: Records(sub, source).dir / C2.statements_name(RUNS + 1, REG) for sub in SUBJECTS}
    X = {sub: CC.Subject(f) for sub, f in xf.items() if f.exists()}
    sf = {sub: C2.statements_file(C2.Records(sub, source).dir, RUNS + 1, REG) for sub in X}
    S = {sub: CC.Subject(sf[sub]) for sub in X}
    if not dry_run and len(X) < len(SUBJECTS):
        sys.exit(f"{source} -> {tgt}: statements for {list(X)} only (a partial evaluation only as --dry-run)")
    if len(X) < 2:
        return {"skipped": f"cross-scanner statements for {list(X)} only (need >= 2)"}
    assert len({s.R for s in list(X.values()) + list(S.values())}) == 1
    stored = (cohort2 or {}).get("protocols", {}).get(C2.SCHEMA_NAME[source], {}).get("as stated", {}).get("per subject", {})
    per = {}
    for sub in X:
        rest = [m for m in X if m != sub]
        xs, ss = XS.stated(X[sub], TERM_X), XS.stated(S[sub], C2.TERM)
        if sub in stored:
            assert np.allclose(stored[sub], ss["coverage at stated level"], atol=1e-4), (source, sub)
        sessions = {t: XS.stated(view(X[sub], t), TERM_X)["coverage at stated level"] for t in X[sub].tags(TERM_X)}
        same_map = XS.map_on([S[m] for m in rest], C2.TERM, X[sub], TERM_X, pat)
        x_map = XS.map_on([X[m] for m in rest], TERM_X, X[sub], TERM_X, pat)
        s_map = XS.map_on([S[m] for m in rest], C2.TERM, S[sub], C2.TERM, pat)
        log = json.loads((Records(sub, source).dir / "runs.json").read_text())
        reg = {t: {k: log[f"{t}_scan"][k] for k in ("target session", "rotation_deg", "translation_mm", "ncc_after", "best", "QC")}
               | {"candidates b0 ncc": {x: c["b0 ncc"] for x, c in log[f"{t}_scan"]["candidates"].items()}} for t in X[sub].tags(TERM_X)}
        diff = [round(a - b, 4) for a, b in zip(xs["coverage at stated level"], ss["coverage at stated level"])]
        per[sub] = {"cross scanner, as stated (both sessions)": xs, "cross scanner, as stated, per target session": sessions,
                    "same scanner (other session), as stated": ss,
                    "extra gap: cross scanner - same scanner, as stated": diff,
                    "same-scanner LOSO map (without this subject) on the cross-scanner targets": same_map,
                    "cross-scanner LOSO map": x_map, "same-scanner LOSO map on its own rescan": s_map,
                    "table (0.8, 0.9, 0.95)": {"same scanner, as stated": pick(ss["coverage at stated level"]), "same scanner, map": pick(s_map["coverage at target"]),
                                               "cross scanner, as stated": pick(xs["coverage at stated level"]), "cross scanner, map": pick(x_map["coverage at target"]),
                                               "cross scanner, same-scanner map": pick(same_map["coverage at target"]), "extra gap": pick(diff)},
                    "registration (mode 3)": reg}
    names = list(per)
    tab = {k: {q: round(float(np.mean([per[s]["table (0.8, 0.9, 0.95)"][k][q] for s in names])), 4) for q in ("0.8", "0.9", "0.95")}
           for k in per[names[0]]["table (0.8, 0.9, 0.95)"]}
    summary = {"subjects": names, "mean (0.8, 0.9, 0.95)": tab,
               "mean curves at levels": {k: mean_curve([per[s][f][g] for s in names]) for k, f, g in (
                   ("cross scanner, as stated", "cross scanner, as stated (both sessions)", "coverage at stated level"),
                   ("same scanner, as stated", "same scanner (other session), as stated", "coverage at stated level"),
                   ("cross scanner, same-scanner map", "same-scanner LOSO map (without this subject) on the cross-scanner targets", "coverage at target"),
                   ("cross scanner, map", "cross-scanner LOSO map", "coverage at target"))},
               "extra gap at 0.9, per subject": {s: per[s]["table (0.8, 0.9, 0.95)"]["extra gap"]["0.9"] for s in names},
               "extra gap at 0.9, per target session": {s: {t: round(per[s]["cross scanner, as stated, per target session"][t][int(np.flatnonzero(np.isclose(LEVELS, 0.9))[0])]
                                                               - per[s]["table (0.8, 0.9, 0.95)"]["same scanner, as stated"]["0.9"], 4)
                                                           for t in per[s]["cross scanner, as stated, per target session"]} for s in names},
               "ECE (mean): streamline / point": {
                   "cross scanner, as stated": [round(float(np.mean([per[s]["cross scanner, as stated (both sessions)"][m] for s in names])), 4) for m in ("streamline ECE", "point ECE")],
                   "cross scanner, same-scanner map": [round(float(np.mean([per[s]["same-scanner LOSO map (without this subject) on the cross-scanner targets"][m] for s in names])), 4) for m in ("streamline ECE (map)", "point ECE (map)")],
                   "cross scanner, map": [round(float(np.mean([per[s]["cross-scanner LOSO map"][m] for s in names])), 4) for m in ("streamline ECE (map)", "point ECE (map)")],
                   "same scanner, as stated": [round(float(np.mean([per[s]["same scanner (other session), as stated"][m] for s in names])), 4) for m in ("streamline ECE", "point ECE")]},
               "registration QC flags": {f"{s} {t}": r["QC"]["flag"] for s in names for t, r in per[s]["registration (mode 3)"].items()},
               "same-scanner numbers equal to calibration_cohortII.json": bool(stored)}
    pr = C2.PROTOCOLS[source]
    nm = name_of(source)
    xs_c = {s: per[s]["cross scanner, as stated (both sessions)"]["coverage at stated level"] for s in names}
    xm = {s: per[s]["cross-scanner LOSO map"] for s in names}
    sm = {s: per[s]["same-scanner LOSO map (without this subject) on the cross-scanner targets"] for s in names}
    fig6 = {"protocols": {nm: {
                "scanner": scanner(source), "voxel_mm": pr["check"]["voxel_mm"], "tracked shell": int(pr["tracked shell"]),
                "directions": pr["check"]["volumes per shell"][str(int(pr["tracked shell"]))], "subjects": names,
                "rescan": f"the same person's {tgt} scans (both sessions) on scanner {scanner(tgt)}, the other Philips Achieva at the same site",
                "confounded": None if source in PRIMARY else CONFOUNDED,
                "as stated": {"per subject": xs_c, "mean": mean_curve(list(xs_c.values()))},
                "map (leave one subject out)": {"per subject": {s: v["coverage at target"] for s, v in xm.items()},
                                                "mean": mean_curve([v["coverage at target"] for v in xm.values()]),
                                                "reached": {s: v["reached"] for s, v in xm.items()}}}},
            "transfer (fixed protocol, other scanner)": {nm: {
                "map from": C2.SCHEMA_NAME[source], "applied to": nm, "map fitted on": names, "target person excluded": True,
                "confounded": None if source in PRIMARY else CONFOUNDED,
                "per subject": {s: v["coverage at target"] for s, v in sm.items()}, "mean": mean_curve([v["coverage at target"] for v in sm.values()]),
                "reached": {s: v["reached"] for s, v in sm.items()},
                "note": f"the {C2.SCHEMA_NAME[source]} same-scanner (other-session) map fitted without the target subject, applied to the "
                        f"{C2.SCHEMA_NAME[source]} envelope against that subject's {tgt} scans"}}}
    prov = {}
    for sub in names:
        R = Records(sub, source)
        prov[sub] = {"records": str(R.dir.relative_to(DATA)), "statements sha256": sha(xf[sub]), "runs.json sha256": sha(R.dir / "runs.json"),
                     "created": json.loads((R.dir / "runs.json").read_text())["created"]["git"],
                     "target records sha256": {t: sha(R.path(f"{t}_onA")) for t in X[sub].tags(TERM_X)},
                     "source records": str(R.reuse.relative_to(DATA)), "source statements": sf[sub].name, "source statements sha256": sha(sf[sub]),
                     "source A.npz sha256": sha(R.path("A"))}
    return {"target protocol": tgt, "primary (fixed protocol)": source in PRIMARY, "confounded": None if source in PRIMARY else CONFOUNDED,
            "per subject": per, "summary": summary, "fig6": fig6, "provenance": prov}


def inverse_consistency():
    """The x?1 pairs both ways (the B ensemble against A's first-session scan, and the A ensemble against B's): one
    composed with the other should be the identity."""
    out = {}
    for a, b in (("B32", "A32"), ("B2000", "A2000r19")):
        for sub in SUBJECTS:
            fa, fb = Records(sub, a).dir / "runs.json", Records(sub, b).dir / "runs.json"
            ka, kb = f"{C2.rescan_tag('x' + scanner(b) + '1', REG)}_scan", f"{C2.rescan_tag('x' + scanner(a) + '1', REG)}_scan"
            if fa.exists() and fb.exists() and ka in (la := json.loads(fa.read_text())) and kb in (lb := json.loads(fb.read_text())):
                c = np.array(la[ka]["to_A"]) @ np.array(lb[kb]["to_A"])
                out[f"{a} <-> {b} {sub}"] = {"rotation_deg": C2.angle(c), "translation_mm": round(float(np.linalg.norm(c[:3, 3])), 3)}
    return out


def evaluate(args):
    out_path = Path(args.out)
    if args.dry_run and out_path.resolve() == OUT.resolve():
        sys.exit("a dry run writes elsewhere: pass --out")
    pat = CC.pat16_table()
    c2p = HERE / "results/calibration_cohortII.json"
    cohort2 = json.loads(c2p.read_text()) if c2p.exists() else None
    if cohort2 is not None and cohort2.get("registration mode") != REG:
        sys.exit(f"{c2p} is registration mode {cohort2.get('registration mode')}, not {REG}")
    res = {"what": "MASiVar cohort II, the same-site cross-scanner rung: one Philips Achieva's 17-run ensemble against the same "
                   "person's scans on the other Philips Achieva at the same site, beside the same-scanner other-session rescan",
           "term": TERM_X, "levels": LEVELS.tolist(), "registration mode": REG, "registration": C2.REGISTRATION[REG], "dry run": bool(args.dry_run),
           "coverage": "calibration_card.coverage_curve: the share of the rescan's density presence inside the envelope, the 42 named "
                       "tracts pooled, weighted by the rescan's voxels; a subject's two target sessions averaged",
           "directions": {}}
    fig6 = {"schema": C2.SCHEMA, "mock": False, "levels": LEVELS.tolist(), "protocols": {}, "transfer (fixed protocol, other scanner)": {},
            "note": "COHORT2_SCHEMA's shape; fig6_protocol_gap draws protocols named B2000, B32, A2000, A32 only, so these rows need "
                    "their names added there to be drawn (ismrm2027_figures.py is not changed by this script)"}
    for source in PRIMARY + SECONDARY:
        r = evaluate_direction(source, pat, args.dry_run, cohort2)
        res["directions"][name_of(source)] = {k: v for k, v in r.items() if k != "fig6"}
        if "fig6" in r:
            fig6["protocols"].update(r["fig6"]["protocols"])
            fig6["transfer (fixed protocol, other scanner)"].update(r["fig6"]["transfer (fixed protocol, other scanner)"])
    res["inverse consistency (x?1 pairs both ways)"] = inverse_consistency()
    res["fig6 (COHORT2_SCHEMA shape)"] = fig6
    res["provenance"] = {"git": CR.git_state(), **module_hashes(), "calibration_cohortII.json sha256": sha(c2p) if c2p.exists() else None,
                         "created (UTC)": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    res["caveats"] = [
        "four subjects; two target sessions per subject and direction (averaged), one same-scanner other-session rescan",
        "B2000 <-> A2000r19 is CONFOUNDED: " + CONFOUNDED,
        "B32 <-> A32 is a fixed protocol, but the reversed series differ (2.5 mm on B; 0.625 x 0.625 x 2.5 mm, box-averaged "
        "and rescaled, on A), so the field estimate is not acquired alike",
        "the same people on both scanners: the LOSO maps leave the target person out",
        "a target session's scan is a different day as well as a different scanner unit, as the same-scanner rescan is a different day"]
    out_path.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    short = {k: (v.get("summary", {}).get("mean (0.8, 0.9, 0.95)") or v.get("skipped")) for k, v in res["directions"].items()}
    print(json.dumps(short, indent=1))
    print(json.dumps(res["inverse consistency (x?1 pairs both ways)"]))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "collect", "evaluate"))
    ap.add_argument("--source", choices=sorted(DIRECTIONS), default="B32", help="run/collect: the ensemble's protocol")
    ap.add_argument("--subject", nargs="+", default=list(SUBJECTS), choices=SUBJECTS)
    ap.add_argument("--targets", nargs="+", help="run: only these targets (x<T>1, x<T>2)")
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--smoke", action="store_true", help="run/collect: the <pair>-smoke folder")
    ap.add_argument("--force", action="store_true", help="collect: rebuild an existing statements file (derived, not a record)")
    ap.add_argument("--dry-run", action="store_true", help="evaluate: whatever statements exist; needs --out elsewhere")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    {"run": run, "collect": collect, "evaluate": evaluate}[args.what](args)
