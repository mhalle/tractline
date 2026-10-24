"""Does the cohort III calibration card transfer to another protocol? An out-of-protocol check on MASiVar cIs1.

    uv run --group dipy --group rankfield bench/calibration_transfer.py plan       # the check against both protocols, what is on disk
    uv run --group dipy --group rankfield bench/calibration_transfer.py run [--runs 16]
    uv run --group dipy --group rankfield bench/calibration_transfer.py collect [--force]
    uv run --group dipy --group rankfield bench/calibration_transfer.py evaluate [--out results/calibration_transfer_cIs1.json]

The card (results/calibration_card_masivar_cIII.json; bench/calibration_runs.py, bench/calibration_card.py) was fitted
on MASiVar cohort III, scanner B: b1000 x40 + b2000 x56, 2.1 x 2.1 x 2.2 mm, 16 forward b0s, tracked at b = 2000.
cIs1 is scanner A: 2.5 mm iso (96 x 96 x 38), five shells of 96 directions of which b1000, b2000 and b3000 are on disk
and joined, 3 forward b0s (each series' last volume), one reversed b0, tracked at b = 3000 (as ensemble_retest.py's
cIs1 pair, NOTES 2026-10-08).

This module reuses calibration_runs (Records, Runner: the no-overwrite records, the disk guard, the bootstrap
replicates, the rescans' registration and resampling onto A's grid) and calibration_card (the statements, the maps'
application, ECE, the envelope curves) through a protocol definition of its own, PROTOCOL_CIS1. calibration_runs'
PROTOCOL, CARD and SHELL are not touched: the cohort III check still refuses cIs1, and `evaluate` shows it does
(the card's own `validity` -> `match`, applied as written).

The statements, as for the card's subjects (calibration_card.collect): A = s1Ax1 scan 1 and its 17-run ensemble (A and
16 replicates: smooth signs, FWHM 4, b0 noise "added", A's mask, draw 0, today's default pipeline at the 3000 shell);
the same-session rescan s1Ax1 scan 2 ("within") and the other-session rescan s1Ax2 scan 1 ("nextday", the next day),
each corrected on its grid, registered rigidly to A, resampled onto A's grid and tracked there.

Records: compact (format 2) in DATA/ensemble/cIs1/card/cIs1/ with runs.json beside them. The 2026-10-08 records in
DATA/ensemble/cIs1/retest/cIs1/ are format 1 (20-point paths, no streamlines, so no density presence; 8 replicates;
the pipeline before the robustness steps) and are not reused.

evaluate (no pipeline runs), per term (same session; other session), on cIs1's rescans:
  (a) as stated (no card): streamline and point ECE, envelope coverage at stated 0.5 / 0.8 / 0.9 / 0.95;
  (b) the cohort III card applied anyway, deliberately out of protocol: its maps on the stated shares, its stated
      level for each target coverage, its predicted shape coverage;
  (c) the card's `match`: every machine-checkable item (protocol, R, settings sha256, labeler weights sha256, terms)
      against cIs1's scans and settings; and, as the control, against a cohort III scan (cIIIs03 A, which it should
      accept), with this module's general check asserted equal to calibration_runs.protocol_check there.
Writes results/calibration_transfer_cIs1.json.
"""
import argparse, hashlib, json, os, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tractline import pipeline as P
from tractline import ensemble as E
from tractline.data import DATA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_runs as CR   # noqa: E402
import calibration_card as CC   # noqa: E402
from ensemble_retest import register, onto_reference, mean_b0  # noqa: E402

SUBJECT = "cIs1"
SHELLS, SHELL = (1000, 2000, 3000), 3000.0     # the shells on disk, joined; the tracker's shell (as ensemble_retest's cIs1)
PROTOCOL_CIS1 = {               # scanner A, as found in cIs1's three scans (plan): a scan is this protocol only if every item holds
    "volumes": 291,
    "volumes per shell": {"0": 3, "1000": 96, "2000": 96, "3000": 96},
    "shell_round": 100,
    "matrix": [96, 96, 38],
    "voxel_mm": [2.5, 2.5, 2.5], "voxel_tolerance_mm": 0.01,
    "phase_encoding": ["j", "j-"],
    "reversed_b0s": 1,
    "reversed_b0_over_forward": [0.8, 1.05],    # seen 0.847-0.901 (cohort III's 0.85 floor would refuse scan 2's 0.847)
}
PAIR = CR.Pair("cIs1", ("s1Ax1", 1), {"within": ("s1Ax1", 2)}, {"nextday": ("s1Ax2", 1)},
               notes=("scanner A; shells b1000, b2000, b3000 of five joined, tracked at 3000",
                      "retest/cIs1 (2026-10-08) is format 1 with 8 replicates: not reused"))
CARD_PATH = HERE / "results/calibration_card_masivar_cIII.json"
OUT = HERE / "results/calibration_transfer_cIs1.json"
CONTROL = ("cIIIs03", ("s1Bx1", 1))            # a cohort III scan the card's match should accept


# ----------------------------------------------------------------------------------------------- the protocol

def check(s, protocol):
    """calibration_runs.protocol_check against a given protocol dict: {item: (found, ok)}."""
    rnd = protocol["shell_round"]
    b = (np.round(np.asarray(s.bval) / rnd) * rnd).astype(int)
    per = {str(k): int(n) for k, n in zip(*np.unique(b, return_counts=True))}
    apa = next(x for x in s.series if "APA" in x["file"])
    rev = [apa["over_app_b0s"][i] for i in apa["taken_as_b0s"]]
    lo, hi = protocol["reversed_b0_over_forward"]
    found = {"volumes": s.dwi.shape[-1], "volumes per shell": per, "matrix": list(s.dwi.shape[:3]),
             "voxel_mm": np.round(s.vox, 4).tolist(), "phase_encoding": list(s.pe_directions), "reversed_b0s": len(rev),
             "reversed_b0_over_forward": rev}
    ok = {"volumes": found["volumes"] == protocol["volumes"], "volumes per shell": per == protocol["volumes per shell"],
          "matrix": found["matrix"] == protocol["matrix"],
          "voxel_mm": bool(np.allclose(s.vox, protocol["voxel_mm"], atol=protocol["voxel_tolerance_mm"])),
          "phase_encoding": found["phase_encoding"] == protocol["phase_encoding"],
          "reversed_b0s": len(rev) == protocol["reversed_b0s"], "reversed_b0_over_forward": all(lo <= r <= hi for r in rev)}
    return {k: (found[k], ok[k]) for k in found}


def load_checked(session, scan):
    from _masivar import load
    s = load(SUBJECT, session, scan, SHELLS)
    bad = {k: v for k, (v, ok) in check(s, PROTOCOL_CIS1).items() if not ok}
    if bad:
        raise ValueError(f"{s.name}: not cIs1's protocol: {bad} (expected {({k: PROTOCOL_CIS1[k] for k in bad})})")
    return s


def pipeline_settings():
    """calibration_runs.pipeline_settings, at cIs1's shells: the tracked shell 3000, b1000-3000 joined."""
    s = CR.pipeline_settings(SHELL)
    s["shells joined"] = list(SHELLS)
    return s


def module_sha256():
    return CR.sha256(Path(__file__).resolve())


# ----------------------------------------------------------------------------------------------- the runs

class Records(CR.Records):
    """cIs1's card folder (written; nothing reused)."""
    def __init__(self):
        self.subject, self.pair = SUBJECT, PAIR
        self.dir = DATA / "ensemble" / SUBJECT / "card" / PAIR.name
        self.reuse = None


class Runner(CR.Runner):
    """calibration_runs.Runner at cIs1's protocol: its own records, check and tracked shell; replicates as is."""
    def __init__(self):
        self.R = Records()
        self.subject, self.pair, self.shell = SUBJECT, PAIR, SHELL
        self.R.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.R.dir / "runs.json"
        self.log = json.loads(self.log_path.read_text()) if self.log_path.exists() else {}
        self.log.setdefault("created", {"git": CR.git_state(), "calibration_transfer.py sha256": module_sha256(),
                                        "pipeline": pipeline_settings(), "protocol": PROTOCOL_CIS1, "pair": {
            "reference": self.pair.ref, "same session": self.pair.same, "other sessions": self.pair.other,
            "reuse": None, "notes": list(self.pair.notes)}})
        self.labeler = None

    def reference(self):
        """calibration_runs.Runner.reference with cIs1's check: A and its companions, all three or none."""
        A = load_checked(*self.pair.ref)
        parts = {"A": self.R.have("A"), "A_meanb0": self.R.have("A_meanb0", ".npy"), "A_mask": self.R.have("A_mask", ".npy")}
        if not any(parts.values()):
            self.log["A_scan"] = {"name": A.name, "series": A.series, "notes": A.notes, "shape": list(A.dwi.shape)}

            def companions(corr, tg):
                for name, arr in (("A_meanb0", mean_b0(corr.dwi, A.bval)), ("A_mask", tg.mask)):
                    p = self.R.new(name, ".npy")
                    tmp = p.with_name(p.stem + ".partial.npy")
                    np.save(tmp, arr)
                    os.replace(tmp, p)
            self.one(A, "A", before_record=companions)
        elif not all(parts.values()):
            raise RuntimeError(f"{self.subject}: A's records are incomplete {parts}: remove the companions of a missing A.npz by hand")
        return A, np.load(self.R.path("A_meanb0", ".npy")), np.load(self.R.path("A_mask", ".npy"))

    def rescan(self, A, ref_b0, tag):
        """calibration_runs.Runner.rescan with cIs1's check: <tag>_onA."""
        if self.R.have(f"{tag}_onA"):
            return
        B = load_checked(*self.pair.rescans[tag])
        timer, t0 = P.Timer(), time.time()
        with P.exact_float32(CR.DEVICE):
            cb = P.correct(B, timer, CR.DEVICE)
        P.release_memory(CR.DEVICE)
        to_ref, ncc = register(mean_b0(cb.dwi, B.bval), B.affine, ref_b0, A.affine)
        rot = np.degrees(np.arccos(np.clip((np.trace(to_ref[:3, :3]) - 1) / 2, -1, 1)))
        dwi, bvec = onto_reference(cb.dwi, B.bvec, B.affine, to_ref, A.affine, A.dwi.shape[:3])
        self.log[f"{tag}_scan"] = {"name": B.name, "term": "same session" if tag in self.pair.same else "other session",
                                   "series": B.series, "shape": list(B.dwi.shape), "residual_left": cb.residual_left,
                                   "field_hz_p99_abs": round(float(np.quantile(np.abs(cb.field_hz), 0.99)), 2),
                                   "correct_seconds": timer.seconds, "correct_register_resample_s": round(time.time() - t0, 1),
                                   "to_A": np.round(to_ref, 5).tolist(), "rotation_deg": round(float(rot), 2),
                                   "translation_mm": round(float(np.linalg.norm(to_ref[:3, 3])), 2), "ncc_after": round(ncc, 4)}
        self.save_log()
        self.one(replace(A, dwi=dwi, bval=B.bval, bvec=bvec, b0s=None), f"{tag}_onA", corrected=dwi)


def run(args):
    t0 = time.time()
    rn = Runner()
    A, ref_b0, mask = rn.reference()
    for tag in rn.pair.rescans:
        rn.rescan(A, ref_b0, tag)
    rn.replicates(A, mask, args.runs)
    print(f"{SUBJECT} done {time.time() - t0:.0f}s, {CR.free_gb():.1f} GB free", flush=True)


def plan(args):
    from _masivar import load
    R = Records()
    for ses, scan in [PAIR.ref] + list(PAIR.rescans.values()):
        s = load(SUBJECT, ses, scan, SHELLS)
        mine = [k for k, (_, ok) in check(s, PROTOCOL_CIS1).items() if not ok]
        theirs = [k for k, (_, ok) in CR.protocol_check(s).items() if not ok]
        print(f"{s.name}: cIs1's protocol: {'yes' if not mine else mine};  cohort III's: {'yes' if not theirs else 'no ' + str(theirs)}")
    need = {"A": int(not R.have("A")), "rescan": sum(not R.have(f"{t}_onA") for t in PAIR.rescans),
            "replicate": sum(not R.have(f"{CR.FLIP}/r{r:03d}") for r in range(1, args.runs + 1))}
    print(f"to run: {need} in {R.dir.relative_to(DATA)}; {CR.free_gb():.1f} GB free")


# ----------------------------------------------------------------------------------------------- the statements

def collect(args):
    """calibration_card.collect for cIs1 (no pooled reference: the rescans have no ensembles of their own)."""
    import nibabel as nib
    from _masivar import scan_files
    from ensemble_grid_check import Grid
    t0 = time.time()
    R = Records()
    out = R.dir / f"statements_R{args.runs + 1}.npz"
    if out.exists() and not args.force:
        raise FileExistsError(f"{out} exists (--force to rebuild it)")
    paths = R.ensemble("", args.runs)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} of A's {len(paths)} runs missing, the first {missing[0]}")
    tags = [t for t in PAIR.rescans if R.have(f"{t}_onA")]
    img = nib.load(scan_files(SUBJECT, *PAIR.ref, SHELLS)["APP"][0][1])
    A0 = E.RunRecord.load(paths[0], logp=False)
    grid = Grid("dwi", img.affine, A0.shape)
    mask = np.load(R.path("A_mask", ".npy"))
    cent = CC.centers(mask, A0.counts, A0.shape)
    M = CC.sphere_matrix(cent, A0.shape, grid.h)
    keys = A0.keys
    print(f"{SUBJECT}: R = {len(paths)}, rescans {tags}, {len(cent)} points", flush=True)
    A = CC.Scan(paths, keys, grid, M, t0)
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
        k, t, pres, main = CC.run_stats(R.path(f"{tag}_onA"), grid, M)
        arrays[f"s_tract_{tag}"] = CC.on_keys(keys, k, t)[s_ok]
        arrays[f"p_main_{tag}"] = main
        arrays[f"e_hit_{tag}"] = CC.vote_hist(A.votes, pres, Rn)[1]
        arrays[f"e_Ahit_{tag}"] = (A.own & pres).sum(1).astype(np.int64)
        print(f"  {tag}_onA {time.time() - t0:.0f}s", flush=True)
    log = json.loads((R.dir / "runs.json").read_text())
    meta = {"subject": SUBJECT, "pair": PAIR.name, "R": Rn, "runs": [str(p.relative_to(DATA)) for p in paths],
            "rescans": {t: {"term": "same session" if t in PAIR.same else "other session", "scan": log[f"{t}_scan"]["name"],
                            "rotation_deg": log[f"{t}_scan"]["rotation_deg"], "translation_mm": log[f"{t}_scan"]["translation_mm"],
                            "ncc_after": log[f"{t}_scan"]["ncc_after"]} for t in tags},
            "pooled": {}, "points": int(len(cent)), "seeds labeled in all R runs": int(s_ok.sum()),
            "records git at creation": log["created"]["git"], "collect git": CR.git_state(),
            "calibration_transfer.py sha256": module_sha256(), "seconds": round(time.time() - t0, 1)}
    np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)
    print(json.dumps(meta, indent=1))


# ----------------------------------------------------------------------------------------------- evaluation

def settings_sha256(settings):
    """As calibration_card.fit hashes the subjects' settings."""
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def match(card, s, settings, R, weights, terms):
    """The card's validity -> match, applied as written, to scan s with these settings, R, weights and terms."""
    m = card["validity"]["match"]
    proto = {k: {"found": found, "expected": m["protocol"][k], "ok": bool(ok)} for k, (found, ok) in check(s, m["protocol"]).items()}
    items = {"protocol": {"ok": all(v["ok"] for v in proto.values()), "items": proto},
             "R": {"found": R, "expected": m["R"], "ok": R == m["R"]},
             "pipeline settings sha256": {"found": settings_sha256(settings), "expected": m["pipeline settings sha256"],
                                          "ok": settings_sha256(settings) == m["pipeline settings sha256"]},
             **({"pipeline config sha256": {"found": settings.get("config", {}).get("sha256"), "expected": m["pipeline config sha256"],
                                            "ok": m["pipeline config sha256"] is not None
                                                  and settings.get("config", {}).get("sha256") == m["pipeline config sha256"]}}
                if "pipeline config sha256" in m else {}),       # cards from 2026-10-24 (schema 3) key on the config hash
             "labeler weights sha256": {"found": weights, "expected": m["labeler weights sha256"], "ok": weights == m["labeler weights sha256"]},
             "terms": {"found": terms, "expected": m["terms"], "ok": all(t in m["terms"] for t in terms)}}
    failed = [k for k, v in items.items() if not v["ok"]]
    return {"scan": s.name, "verdict": "accepted" if not failed else "refused", "failed": failed, "items": items}


def stated_at(curve, q):
    return round(float(curve[np.flatnonzero(CC.GRID == round(q, 3))[0]]), 4)


def evaluate_term(subj, term, card):
    """(a) as stated and (b) with the cohort III card, on cIs1's rescans of `term`."""
    R = subj.R
    t = card["terms"][term]
    v, h = subj.streamline(term)
    vp, hp = subj.point(term)
    curve = CC.term_curve(subj, term)
    shapes = CC.term_shapes(subj, term)
    env_b = {}
    for q in CC.TARGETS:
        x = t["envelope"]["stated level for target coverage"][f"{q:g}"]
        lv = x["level"]
        env_b[f"{q:g}"] = {"card: stated level used": lv, "card: predicted coverage (its training curve there)":
                           round(float(np.array(t["envelope"]["observed coverage"])[np.flatnonzero(CC.GRID == lv)[0]]), 4) if lv < 1 else None,
                           "cIs1: coverage": stated_at(curve, lv) if lv < 1 else round(float(shapes["widest"]["coverage"]), 4)}
    pred = t["shapes"]["all named tracts"]
    return {
        "rescans": subj.tags(term),
        "(a) as stated": {
            "streamline": CC.ece(v / R, h), "point": CC.ece(vp / R, hp),
            "envelope coverage at stated level": {f"{q:g}": stated_at(curve, q) for q in CC.TARGETS},
            "shapes (coverage, precision, size against the rescan)": {k: {m: round(x, 4) for m, x in shapes[k].items()}
                                                                       for k in ("A alone", "core", "vorobev", "widest")}},
        "(b) cohort III card applied (out of protocol)": {
            "streamline": CC.ece(CC.apply_votes(t["streamline"], v, R), h),
            "point": CC.ece(CC.apply_votes(t["point"], vp, R), hp),
            "envelope at target coverage": env_b,
            "envelope: |card's predicted coverage - cIs1's| over GRID, mean (max)": [
                round(float(np.abs(np.array(t["envelope"]["observed coverage"]) - curve).mean()), 4),
                round(float(np.abs(np.array(t["envelope"]["observed coverage"]) - curve).max()), 4)],
            "shapes: card's predicted / cIs1's observed coverage": {k: [pred[k]["coverage"], round(shapes[k]["coverage"], 4)]
                                                                   for k in ("A alone", "core", "vorobev", "widest")}},
        "streamline: the rescan labels the seed": round(float(np.mean(subj.z[f"s_tract_{subj.tags(term)[0]}"] >= 0)), 4),
    }


def evaluate(args):
    from _masivar import load
    from tractline.labelers.rapidparc import HASHES
    card = json.loads(CARD_PATH.read_text())
    R = Records()
    stmt = R.dir / f"statements_R{args.runs + 1}.npz"
    subj = CC.Subject(stmt)
    log = json.loads((R.dir / "runs.json").read_text())
    settings = log["created"]["pipeline"]
    terms = {t: evaluate_term(subj, t, card) for t in CC.TERMS}
    # (c) the card's match: cIs1's scans, and a cohort III control
    verdicts = {}
    for tag, (ses, scan) in [("A", PAIR.ref)] + list(PAIR.rescans.items()):
        s = load(SUBJECT, ses, scan, SHELLS)
        verdicts[tag] = match(card, s, settings, subj.R, HASHES["rapidparc"], [subj.meta["rescans"][tag]["term"]] if tag != "A" else list(CC.TERMS))
    cs = load(CONTROL[0], *CONTROL[1], CR.SHELLS)
    assert check(cs, CR.PROTOCOL) == CR.protocol_check(cs), "the general check differs from calibration_runs.protocol_check"
    control_settings = json.loads((CR.Records(CONTROL[0]).dir / "runs.json").read_text())["created"]["pipeline"]
    control = match(card, cs, control_settings, card["validity"]["match"]["R"], HASHES["rapidparc"], list(CC.TERMS))
    # cohort III's held-out reference, from the card
    summ = card["held-out evaluation (leave one subject out)"]["summary"]
    ref = {t: {"streamline ECE: as stated / card": summ[t]["streamline ECE: as stated / card / PAT16 map"][:2],
               "point ECE: as stated / card": summ[t]["point ECE: as stated / card"],
               "envelope coverage at target, as stated / card": summ[t]["envelope coverage at target, as stated / card"],
               "subjects held out": summ[t]["subjects held out"]} for t in CC.TERMS}
    out = {
        "what": "the cohort III calibration card (scanner B) on MASiVar cIs1 (scanner A), deliberately out of protocol",
        "card": {"path": str(CARD_PATH.relative_to(HERE.parent)), "card id": card["card id"], "sha256": CR.sha256(CARD_PATH)},
        "cIs1": {"protocol": PROTOCOL_CIS1, "shells joined": list(SHELLS), "tracked shell": SHELL, "pair": {
                     "A": PAIR.ref, "same session": PAIR.same, "other session": PAIR.other},
                 "R": subj.R, "rescans": subj.meta["rescans"], "points": subj.meta["points"],
                 "seeds labeled in all R runs": subj.meta["seeds labeled in all R runs"],
                 "records": str(R.dir.relative_to(DATA)), "records git at creation": log["created"]["git"],
                 "statements sha256": CR.sha256(stmt),
                 "not reused": "ensemble/cIs1/retest/cIs1 (2026-10-08): format 1 records (no streamlines, so no density presence), 8 replicates, the pipeline before the robustness steps"},
        "terms": terms,
        "(c) match": {"cIs1": verdicts, "control (cohort III scan)": control,
                      "check": "calibration_transfer.check against the card's validity -> match -> protocol; asserted equal to calibration_runs.protocol_check on the control"},
        "cohort III held out (from the card, for comparison)": ref,
        "evaluate git": CR.git_state(), "calibration_transfer.py sha256": module_sha256(),
        "created (UTC)": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "caveats": ["one subject, one same-session and one other-session rescan: no interval, only a point estimate",
                    "cIs1's ensemble and rescans differ from the card's in protocol and in the tracked shell (3000, not 2000), so the settings hash differs too; the two cannot be separated here",
                    "the statements are as the card's (calibration_card.collect); envelope coverage is pooled over the 42 named tracts"],
    }
    Path(args.out).write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    short = {t: {"(a) streamline / point ECE": [terms[t]["(a) as stated"]["streamline"]["ece"], terms[t]["(a) as stated"]["point"]["ece"]],
                 "(b) streamline / point ECE": [terms[t]["(b) cohort III card applied (out of protocol)"]["streamline"]["ece"],
                                                terms[t]["(b) cohort III card applied (out of protocol)"]["point"]["ece"]],
                 "(a) coverage at stated": terms[t]["(a) as stated"]["envelope coverage at stated level"],
                 "(b) coverage at target": {q: x["cIs1: coverage"] for q, x in terms[t]["(b) cohort III card applied (out of protocol)"]["envelope at target coverage"].items()}}
             for t in CC.TERMS}
    print(json.dumps({"terms": short, "match": {k: (v["verdict"], v["failed"]) for k, v in verdicts.items()},
                      "control": (control["verdict"], control["failed"])}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("plan", "run", "collect", "evaluate"))
    ap.add_argument("--runs", type=int, default=CR.RUNS, help="A's replicates (R = runs + 1)")
    ap.add_argument("--force", action="store_true", help="collect: rebuild an existing statements file (derived, not a record)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    {"plan": plan, "run": run, "collect": collect, "evaluate": evaluate}[args.what](args)
