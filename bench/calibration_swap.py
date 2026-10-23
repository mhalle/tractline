"""MASiVar cohort II, B2000 with the sessions' roles swapped: session 2 (s1Bx2) is the "A" scan and its 17-run ensemble,
session 1 (s1Bx1) the other-session rescan ("session1"), registered onto session 2 and tracked there. The forward
direction is calibration_cohortII's B2000 (A = s1Bx1, rescan session2); this gives a second pair per person, 8 in all.

    uv run --group dipy --group rankfield bench/calibration_swap.py run [--subject cIIs01 ...] [--runs 16] [--smoke] [--registration 2]
    uv run --group dipy --group rankfield bench/calibration_swap.py collect [--subject ...] [--runs 16] [--smoke] [--force] [--registration 2]
    uv run --group dipy --group rankfield bench/calibration_swap.py evaluate [--runs 16] [--registration 2] [--dry-run --out <elsewhere>]

Everything but the roles is calibration_cohortII's (imported, not edited: its load, check, Runner, collect and
evaluation functions): the protocol's descriptor and loader, the field from the A scan's own reversed b0 (APA 107 of
session 2), the pipeline settings, the replicates (smooth signs, FWHM 4, b0 noise "added", A's mask, replicate r's
seed r) and R = 17, and the rescan's registration modes (calibration_cohortII.REGISTRATION; default 2, FA onto FA then
the mean b0s: session1reg2_onA.npz, statements_R17_reg2.npz; 3, multi-start, with the forward direction's transform
as start (c): session1reg3_scan, and session1reg3_onA only where it differs from mode 2; evaluate reads both
directions at the same mode). Records: DATA/ensemble/<subject>/card/<subject>-B2000-swap/ (--smoke: ...-swap-smoke/); SwapRecords
refuses to write anywhere else, so the forward records are never touched. runs.json's "created" carries the git state,
both modules' SHA-256 and the input files' SHA-256; collect appends its own provenance there ("collect (swap)").

evaluate (no pipeline runs) writes results/calibration_swap_B2000.json, never results/calibration_cohortII.json (whose
B2000 rows stay forward only):
  forward     calibration_cohortII's B2000 analysis again (as stated, leave one subject out, learning curve), for reference;
  swapped     the same on the swapped pairs;
  combined    the 8 pairs (each a "subject" of equal weight in the map's fit): as stated over 8; the map leave one PERSON
              out (fitted on the other three people's 6 pairs, scored on the held-out person's 2); the learning curve
              over people (k = 1-3 people, both directions each); and the same without cIIs04 (sensitivity: its session
              2 is under question);
  per person  the as-stated gap forward against swapped (the direction asymmetry);
  registration per person the forward (session 2 -> 1; modes 1 and 2) and swapped (session 1 -> 2) rigid transforms and
              NCC, and each forward one composed with the swapped one: its distance from the identity (inverse
              consistency), cIIs04 called out.
"""
import argparse, itertools, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_cohortII as C2   # noqa: E402  (no side effects at import: definitions only)
import calibration_runs as CR   # noqa: E402
import calibration_card as CC   # noqa: E402
from tractline.data import DATA   # noqa: E402

KEY = "B2000"
TAG = "session1"
TERM = C2.TERM
OUT = HERE / "results/calibration_swap_B2000.json"
QUESTIONED = "cIIs04"                 # its session 2 is under question (misregistered or a bad scan?)


def sha(path):
    return CR.sha256(Path(path).resolve())


def module_hashes():
    return {"calibration_swap.py sha256": sha(__file__), "calibration_cohortII.py sha256": C2.module_sha256(),
            "_register_fa.py sha256": sha(HERE / "_register_fa.py")}


def swap_pair(subject):
    fwd = C2.pairs(KEY)[subject]
    s1, s2 = fwd.ref[0], fwd.other["session2"][0]
    return CR.Pair(f"{subject}-{KEY}-swap", (s2, 1), {}, {TAG: (s1, 1)},
                   notes=(C2.PROTOCOLS[KEY]["note"], "swapped: A is session 2, the rescan session 1",
                          "one scan per session: no same-session rescan"))


def subjects():
    return list(C2.pairs(KEY))


class SwapRecords(CR.Records):
    """The swapped pair's record folder (<subject>-B2000-swap[-smoke]); writes anywhere else are refused."""
    def __init__(self, subject, key=KEY, smoke=False):
        assert key == KEY, key
        self.subject, self.pair = subject, swap_pair(subject)
        self.dir = DATA / "ensemble" / subject / "card" / (self.pair.name + ("-smoke" if smoke else ""))
        self.reuse = None

    def new(self, name, ext=".npz"):
        assert self.dir.name.endswith(("-swap", "-swap-smoke")), self.dir
        return super().new(name, ext)


def input_hashes(subject):
    """SHA-256 of every input file (APP and the chosen APA series, with their .bval/.bvec) of both sessions."""
    out = {}
    pair = swap_pair(subject)
    for role, ses in (("A (session 2)", pair.ref[0]), ("rescan (session 1)", pair.rescans[TAG][0])):
        f = C2.series_files(subject, ses, KEY)
        for p in [p for _, p in f["APP"]] + [f["APA"]]:
            for q in (p, C2.M._side(p, ".bval"), C2.M._side(p, ".bvec")):
                out[str(Path(q).relative_to(C2.M.ROOT))] = {"role": role, "sha256": sha(q)}
    return out


class SwapRunner(C2.Runner):
    """calibration_cohortII.Runner with the swapped pair's records (its reference, rescan and replicates unchanged)."""
    def __init__(self, subject, key=KEY, smoke=False):
        self.key, self.shell = key, C2.PROTOCOLS[key]["tracked shell"]
        self.R = SwapRecords(subject, key, smoke)
        self.subject, self.pair = subject, self.R.pair
        self.R.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.R.dir / "runs.json"
        self.log = json.loads(self.log_path.read_text()) if self.log_path.exists() else {}
        self.log.setdefault("created", {"git": CR.git_state(), **module_hashes(), "inputs sha256": input_hashes(subject),
                                        "pipeline": C2.pipeline_settings(key), "protocol": {"key": key, **C2.PROTOCOLS[key]},
                                        "smoke": smoke, "swapped": True, "forward records": str(C2.Records(subject, key).dir.relative_to(DATA)),
                                        "pair": {"reference": self.pair.ref, "same session": self.pair.same, "other sessions": self.pair.other,
                                                 "reuse": None, "notes": list(self.pair.notes)}})
        self.labeler = None

    def reg3_inputs(self, tag):
        """Mode 3's shortcuts for session1 (calibration_cohortII.Runner.reg3_inputs): the forward folder's A_meanb0
        (session 1 corrected as here) and the forward direction's best transform (mode 3's, else mode 2's)."""
        d = C2.Records(self.subject, KEY).dir
        T, src = C2.best_to_A(json.loads((d / "runs.json").read_text()), "session2")
        return np.load(d / "A_meanb0.npy"), T, (f"{d.name} {src}" if src else None)


def run(args):
    for subject in args.subject:
        t0 = time.time()
        rn = SwapRunner(subject, KEY, args.smoke)
        A, ref_b0, mask = rn.reference()
        if args.register_only:
            assert args.registration == 3, "--register-only is mode 3's"
            rn.register3(A, ref_b0, TAG)
            continue
        rn.rescan(A, ref_b0, TAG, args.registration)
        rn.replicates(A, mask, args.runs)
        tag = C2.rescan_tag(TAG, args.registration)
        s = rn.log[f"{tag}_scan"]
        print(f"{subject} {KEY} swap: {tag} -> A (registration {args.registration}) rotation {s['rotation_deg']} deg, translation {s['translation_mm']} mm, NCC {s['ncc_after']}", flush=True)
        print(f"{subject} {KEY} swap done {time.time() - t0:.0f}s, {CR.free_gb():.1f} GB free", flush=True)


def collect(args):
    """calibration_cohortII.collect on the swapped records (its Records rebound to SwapRecords for this call only),
    then this module's provenance appended to runs.json."""
    orig = C2.Records
    C2.Records = SwapRecords
    try:
        C2.collect(argparse.Namespace(subject=args.subject, protocol=KEY, runs=args.runs, smoke=args.smoke, force=args.force,
                                      registration=args.registration))
    finally:
        C2.Records = orig
    for subject in args.subject:
        R = SwapRecords(subject, KEY, args.smoke)
        log_path = R.dir / "runs.json"
        log = json.loads(log_path.read_text())
        st = R.dir / C2.statements_name(args.runs + 1, args.registration)
        log.setdefault("collect (swap)", []).append({"git": CR.git_state(), **module_hashes(), "statements": st.name, "statements sha256": sha(st),
                                                     "when (UTC)": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        tmp = log_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(log, indent=1))
        os.replace(tmp, log_path)


# ----------------------------------------------------------------------------------------------- evaluation

def load_statements(direction, runs, dry_run, registration):
    """{subject: calibration_card.Subject} of one direction ("forward": calibration_cohortII's B2000 records) at a
    registration mode."""
    rec = (lambda s: C2.Records(s, KEY)) if direction == "forward" else (lambda s: SwapRecords(s, KEY))
    out = {}
    for sub in subjects():
        d = rec(sub).dir
        p = C2.statements_file(d, runs + 1, registration)
        if not p.exists() and dry_run:
            found = sorted(x for x in d.glob("statements_R*.npz") if ("_reg" in x.name) == (registration != 1))
            p = found[-1] if found else p
        if p.exists():
            out[sub] = CC.Subject(p)
    return out


def curves(S):
    """As calibration_cohortII.schema_part for one protocol: the as-stated curve and the leave-one-out map's, at
    schema levels."""
    mean = lambda d: [round(float(x), 4) for x in np.mean(list(d.values()), 0)] if d else None  # noqa: E731
    stated = {sub: [C2.at(CC.term_curve(s, TERM), float(q)) for q in C2.SCHEMA_LEVELS] for sub, s in S.items()}
    per, reached = {}, {}
    if len(S) >= 2:
        for sub, s in S.items():
            per[sub], reached[sub] = C2.coverage_at_targets([x for m, x in S.items() if m != sub], s)
    return {"levels": C2.SCHEMA_LEVELS.tolist(), "as stated": {"per subject": stated, "mean": mean(stated)},
            "map (leave one subject out)": {"per subject": per, "mean": mean(per), "reached": reached} if per else None}


def one_direction(S, pat):
    per = {sub: C2.as_stated(s) for sub, s in S.items()}
    return {"subjects": list(S), "as stated": {"summary": C2.summarize_as_stated(per), "per subject": per},
            "map, leave one subject out": C2.loso(S, pat), "learning curve": C2.learning_curve(S, pat), "curves": curves(S)}


def combined(F, W, pat, people):
    """The 8 pairs (forward F and swapped W of `people`): as stated, the map leave one person out, the learning curve
    over people, and the curves at schema levels."""
    import calibration_learning_curve as LC
    units = [(p, "forward", F[p]) for p in people if p in F] + [(p, "swapped", W[p]) for p in people if p in W]
    if not units or len({u[2].R for u in units}) != 1:
        return {"skipped": "no pairs, or pairs at different R"}
    label = lambda u: f"{u[0]} {u[1]}"  # noqa: E731
    stated = {label(u): C2.as_stated(u[2]) for u in units}
    held, cov, reached = {}, {}, {}
    for u in units:
        train = [v[2] for v in units if v[0] != u[0]]
        held[label(u)] = CC.evaluate(train, u[2], TERM, pat) if train else None
        if train:
            cov[label(u)], reached[label(u)] = C2.coverage_at_targets(train, u[2])
    data = {label(u): LC.Data(u[2], TERM) for u in units}
    widest = {k: d.widest() for k, d in data.items()}
    by_person = {p: [label(u) for u in units if u[0] == p] for p in people}
    lc = {"as stated (k = 0)": LC.summarize([LC.score("as stated", d, widest[k], pat) for k, d in data.items()]), "k (people)": {}}
    ps = [p for p in people if by_person[p]]
    for k in range(1, len(ps)):
        rows = []
        for p in ps:
            for c in itertools.combinations([q for q in ps if q != p], k):
                card = LC.fit([data[x] for q in c for x in by_person[q]])
                rows += [LC.score(card, data[x], widest[x], pat) for x in by_person[p]]
        lc["k (people)"][str(k)] = {**LC.summarize(rows), "training sets per held-out person": len(list(itertools.combinations(range(len(ps) - 1), k)))}
    mean = lambda d: [round(float(x), 4) for x in np.mean(list(d.values()), 0)] if d else None  # noqa: E731
    st_curves = {label(u): [C2.at(CC.term_curve(u[2], TERM), float(q)) for q in C2.SCHEMA_LEVELS] for u in units}
    return {"pairs": [label(u) for u in units], "people": ps,
            "as stated": {"summary": C2.summarize_as_stated(stated), "per pair": stated},
            "map, leave one person out": {"per pair": held, "summary": C2.summarize_mapped(list(held.values()))},
            "learning curve (people)": lc,
            "curves": {"levels": C2.SCHEMA_LEVELS.tolist(), "as stated": {"per pair": st_curves, "mean": mean(st_curves)},
                       "map (leave one person out)": {"per pair": cov, "mean": mean(cov), "reached": reached}}}


def registration(people, mode):
    """Per person: forward (session 2 -> session 1 world; registration modes 1 and 2) and swapped (session 1 ->
    session 2, the given mode) transforms and NCC, and each forward one composed with the swapped one (session 1 world
    -> itself): the identity if the two directions agree."""
    out = {}
    for p in people:
        row = {}
        for name, d, tag in [(f"forward, registration {m} (session2 onto session 1)", C2.Records(p, KEY).dir, C2.rescan_tag("session2", m)) for m in (1, 2, 3)] + \
                            [(f"swapped, registration {m} (session1 onto session 2)", SwapRecords(p, KEY).dir, C2.rescan_tag(TAG, m)) for m in (2, 3) if m <= mode]:
            f = d / "runs.json"
            if f.exists() and f"{tag}_scan" in (log := json.loads(f.read_text())):
                s = log[f"{tag}_scan"]
                row[name] = {k: s[k] for k in ("to_A", "rotation_deg", "translation_mm", "ncc_after", "residual_left", "field_hz_p99_abs") if k in s}
                row[name]["registration mode"] = s.get("registration mode", 1)
                for k in ("record", "best", "best b0 ncc", "QC", "decision"):
                    if k in s:
                        row[name][k] = s[k]
        sw = row.get(f"swapped, registration {mode} (session1 onto session 2)")
        for k in [k for k in row if k.startswith("forward")]:
            if sw is not None:
                c = np.array(row[k]["to_A"]) @ np.array(sw["to_A"])
                row[f"{k.split(' (')[0]} @ swapped (identity if consistent)"] = {
                    "rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(c[:3, :3]) - 1) / 2, -1, 1)))), 3),
                    "translation_mm": round(float(np.linalg.norm(c[:3, 3])), 3)}
        out[p] = row
    return out


def evaluate(args):
    out_path = Path(args.out)
    if out_path.resolve() == (HERE / "results/calibration_cohortII.json").resolve():
        sys.exit("calibration_cohortII.json is calibration_cohortII's (forward only): write elsewhere")
    if args.dry_run and out_path.resolve() == OUT.resolve():
        sys.exit("a dry run writes elsewhere: pass --out")
    F, W = (load_statements(d, args.runs, args.dry_run, args.registration) for d in ("forward", "swapped"))
    people = subjects()
    if not args.dry_run:
        missing = {d: [p for p in people if p not in S] for d, S in (("forward", F), ("swapped", W))}
        if any(missing.values()):
            sys.exit(f"{C2.statements_name(args.runs + 1, args.registration)} missing: {missing} (a partial evaluation only as --dry-run)")
    pat = CC.pat16_table()
    gap = lambda S, p, q: C2.as_stated(S[p])["envelope gap (coverage - stated)"][f"{q:g}"]  # noqa: E731
    res = {"what": "MASiVar cohort II B2000, both directions: forward (A = session 1) and swapped (A = session 2), 8 pairs",
           "dry run": bool(args.dry_run), "term": TERM, "registration mode (both directions)": args.registration,
           "registration": C2.REGISTRATION[args.registration], "R": sorted({s.R for s in list(F.values()) + list(W.values())}),
           "forward": one_direction(F, pat), "swapped": one_direction(W, pat),
           "combined (8 pairs)": combined(F, W, pat, people),
           f"combined without {QUESTIONED} (sensitivity)": combined(F, W, pat, [p for p in people if p != QUESTIONED]),
           "per person: as-stated gap, forward / swapped": {p: {f"{q:g}": [gap(F, p, q) if p in F else None, gap(W, p, q) if p in W else None]
                                                               for q in CC.TARGETS} for p in people},
           "registration (per person)": registration(people, args.registration),
           "provenance": {"evaluate git": CR.git_state(), **module_hashes(),
                          "statements": {d: {p: {"path": str(Path(s.meta["runs"][0]).parent), "R": s.R,
                                                 "calibration_cohortII.py sha256 at collect": s.meta.get("calibration_cohortII.py sha256"),
                                                 "registration mode": s.meta.get("registration mode", 1)}
                                             for p, s in S.items()} for d, S in (("forward", F), ("swapped", W))}},
           "created (UTC)": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "caveats": ["the two directions of a person share both scans: they are not independent pairs, so the map is left out "
                       "by person and the learning curve counts people",
                       "results/calibration_cohortII.json's B2000 rows are the forward direction only; this file adds the swapped one"]}
    res["registration (per person)"][QUESTIONED] = {"note": "under question (its session 2): both directions' registrations, their NCC "
                                                            "and their consistency", **res["registration (per person)"].get(QUESTIONED, {})}
    out_path.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    short = {k: (res[k]["as stated"]["summary"] if "as stated" in res[k] else res[k]) for k in ("forward", "swapped", "combined (8 pairs)")}
    short = {k: {m: v.get(m) for m in ("subjects", "streamline ECE", "envelope gap (coverage - stated)")} if isinstance(v, dict) else v for k, v in short.items()}
    short["map: |coverage - target|, mean over targets"] = {
        "forward": (res["forward"]["map, leave one subject out"]["summary"] or {}).get("envelope |coverage - target| with the map, mean over targets"),
        "swapped": (res["swapped"]["map, leave one subject out"]["summary"] or {}).get("envelope |coverage - target| with the map, mean over targets"),
        "combined (leave one person out)": ((res["combined (8 pairs)"].get("map, leave one person out") or {}).get("summary") or {}).get(
            "envelope |coverage - target| with the map, mean over targets")}
    short["registration"] = {p: {k: ({m: v[m] for m in ("rotation_deg", "translation_mm", "ncc_after") if m in v}) for k, v in r.items() if isinstance(v, dict)}
                             for p, r in res["registration (per person)"].items()}
    print(json.dumps(short, indent=1, default=str))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "collect", "evaluate"))
    ap.add_argument("--subject", nargs="+")
    ap.add_argument("--runs", type=int, default=CR.RUNS, help="A's replicates (R = runs + 1)")
    ap.add_argument("--smoke", action="store_true", help="run/collect: the <subject>-B2000-swap-smoke folder")
    ap.add_argument("--force", action="store_true", help="collect: rebuild an existing statements file (derived, not a record)")
    ap.add_argument("--dry-run", action="store_true", help="evaluate: whatever statements exist, at any R; needs --out elsewhere")
    ap.add_argument("--out", default=str(OUT), help="evaluate: the output (default results/calibration_swap_B2000.json)")
    ap.add_argument("--registration", type=int, choices=(1, 2, 3), default=C2.DEFAULT_REGISTRATION,
                    help="run: the rescan's registration mode; collect, evaluate: the records' (default calibration_cohortII.DEFAULT_REGISTRATION)")
    ap.add_argument("--register-only", action="store_true", help="run: mode 3's transform and decision only (no tracking)")
    args = ap.parse_args()
    if args.what != "evaluate":
        args.subject = args.subject or subjects()
        bad = [s for s in args.subject if s not in subjects()]
        if bad:
            ap.error(f"{bad}: no B2000 pair (pairs: {subjects()})")
    {"run": run, "collect": collect, "evaluate": evaluate}[args.what](args)
