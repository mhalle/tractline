"""What recalibration costs: the size and precision of the envelopes a recalibration map picks, against the as-stated
envelopes at the same target. Read-only on the records; CPU only (no pipeline runs).

    uv run --group dipy --group rankfield bench/calibration_precision.py [--out results/calibration_precision.json] [--no-floor]

Per condition and subject (a subject's rescans of the condition's term averaged, then the subjects averaged), at
targets 0.8, 0.9 and 0.95, from the statements' vote histograms (calibration_card.curves: the envelope at each stated
level on calibration_card.GRID, and the widest outline last), pooled over the 42 named tracts:
  coverage   the share of the rescan's presence voxels inside the envelope;
  precision  the share of the envelope's voxels inside the rescan's presence;
  size       the envelope's voxels over the rescan's presence voxels;
  (a) as stated: the envelope at stated level p = target;
  (b) the leave-one-subject-out map: fitted (calibration_card.fit_term) on the condition's other subjects, its stated
      level for the target (calibration_card.level_for; the widest outline where the map never reaches it, "not
      reached"); its coverage is asserted equal to calibration_crosssite.map_on's;
  (c) (a) -> (b): size as a relative change (of the mean sizes, and the mean of the subjects' ratios), precision as a
      difference; and "A alone" (A's own run's presence as the outline, calibration_card.term_shapes) beside them.
Conditions and their statements (each read as its own script reads them; cohort II and the cross-site, cross-scanner
and Siemens pairs at registration mode 3 through calibration_cohortII.statements_file or the scripts' own file names):
  cohort III other session      calibration_card: DATA/ensemble/<subject>/card/<pair>/statements_R17.npz, CARD's subjects
                                with other-session rescans (cohort III predates the registration modes);
  cohort II same scanner        calibration_cohortII, per protocol B2000, A2000r19, B32, A32 (cIIs01-04);
  ON-Harmony same scanner       calibration_onharmony.statements: the four six-session subjects, A on their scanner, the
                                five same-scanner rescans each; ON-Harmony cross site: A on OXF1PRI, OXF2PRI001 (20);
  MASiVar cross scanner         calibration_crossscanner: B32 -> A32 scans and A32 -> B32 scans (both sessions pooled);
  Siemens (site 3)              calibration_crosssite: B2000's ensemble against the site 3 scan.
The floor ("best possible" envelope precision; --no-floor skips it): one real rescan's presence used as the outline for
another rescan of the same person on the same grid, every ordered pair (presence recomputed from the records as
calibration_card.run_stats does: density >= RHO per named tract): ON-Harmony's same-scanner rescans (5 per subject);
cohort II's two sessions of one scanner placed on the other scanner's A (calibration_crossscanner's x?1 and x?2:
B32 -> A32, A32 -> B32, B2000 -> A2000r19, A2000r19 -> B2000 scans); cohort III's rescans of subjects with two or
more (same session against other session, and other against other, kept apart).
"""
import argparse, itertools, json, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_card as CC   # noqa: E402
import calibration_cohortII as CII   # noqa: E402  (definitions only on import)
import calibration_crosssite as XS   # noqa: E402
import calibration_crossscanner as XC   # noqa: E402
import calibration_onharmony as OH   # noqa: E402
import calibration_runs as CR   # noqa: E402
from tractline.data import DATA   # noqa: E402

OUT = HERE / "results/calibration_precision.json"
TARGETS = (0.8, 0.9, 0.95)
RUNS, REG = CR.RUNS, 3
SUBS2 = ("cIIs01", "cIIs02", "cIIs03", "cIIs04")
COHORT3_TERM = "other session"


def load(path):
    s = CC.Subject(path)
    s.path = Path(path)
    return s


def conditions():
    """{condition: (term, {subject: calibration_card.Subject})}"""
    out = {}
    c3 = {}
    for n in CR.CARD:
        p = CR.Records(n).dir / f"statements_R{RUNS + 1}.npz"
        if p.exists():
            s = load(p)
            if s.tags(COHORT3_TERM):
                c3[n] = s
    out["cohort III other session"] = (COHORT3_TERM, c3)
    for key in ("B2000", "A2000r19", "B32", "A32"):
        out[f"cohort II same scanner {key}"] = (CII.TERM, {n: load(CII.statements_file(CII.Records(n, key).dir, RUNS + 1, REG)) for n in SUBS2})
    S = {sc: OH.statements(sc, RUNS) for sc in OH.SCANNERS}
    out["ON-Harmony same scanner"] = (OH.TERM_SAME, {n: S[sc][n] for n, sc in OH.RESCAN.items() if n in S[sc] and S[sc][n].tags(OH.TERM_SAME)})
    out["ON-Harmony cross site (OXF1 -> OXF2)"] = (OH.TERM_X, {n: s for n, s in S["OXF1"].items() if s.tags(OH.TERM_X)})
    for src, tgt in (("B32", "A32"), ("A32", "B32")):
        out[f"MASiVar cross scanner {src} -> {tgt}"] = (XC.TERM_X, {n: load(XC.Records(n, src).dir / CII.statements_name(RUNS + 1, REG)) for n in SUBS2})
    out["Siemens (site 3, B2000 ensemble)"] = (XS.TERM_X, {n: load(CII.statements_file(DATA / f"ensemble/{n}/card/{n}-B2000-site3", RUNS + 1, REG)) for n in SUBS2})
    return out


def subject_curves(s, term):
    """The subject's mean over its rescans of term: coverage, size, precision at every GRID level and the widest last."""
    rows = []
    for cnt, hit, _, _ in s.envelopes(term):
        cv, sz = CC.curves(cnt, hit)
        rows.append((cv, sz, cv / np.maximum(sz, 1e-12)))
    return [np.mean([r[i] for r in rows], 0) for i in range(3)]


def index_of(level):
    return -1 if level >= 1 else int(np.flatnonzero(CC.GRID == round(level, 3))[0])


def condition_rows(term, S, pat):
    per = {}
    for n, s in S.items():
        cov, size, prec = subject_curves(s, term)
        train = [x for m, x in S.items() if m != n]
        fit = CC.fit_term(train, term)["_curve"]
        mapped = XS.map_on(train, term, s, term, pat, levels=np.array(TARGETS))
        alone = CC.term_shapes(s, term)["A alone"]
        row = {"rescans": s.tags(term), "statements": s.path.name, "A alone (coverage, precision, size)": [round(float(alone[m]), 4) for m in ("coverage", "precision", "size")]}
        for j, q in enumerate(TARGETS):
            ia = index_of(q)
            lv, reached, _ = CC.level_for(fit, q)
            ib = index_of(lv)
            assert abs(float(cov[ib]) - mapped["coverage at target"][j]) < 1e-4, (n, q, float(cov[ib]), mapped["coverage at target"][j])
            row[f"{q:g}"] = {"as stated": {"level": q, "coverage": round(float(cov[ia]), 4), "precision": round(float(prec[ia]), 4), "size": round(float(size[ia]), 4)},
                             "map": {"level": lv if lv < 1 else "widest", "reached": bool(reached), "coverage": round(float(cov[ib]), 4),
                                     "precision": round(float(prec[ib]), 4), "size": round(float(size[ib]), 4)}}
        per[n] = row
    names = list(per)
    summ = {}
    for q in (f"{x:g}" for x in TARGETS):
        m = lambda w, k: float(np.mean([per[n][q][w][k] for n in names]))  # noqa: E731
        summ[q] = {"as stated (coverage, precision, size)": [round(m("as stated", k), 4) for k in ("coverage", "precision", "size")],
                   "map (coverage, precision, size)": [round(m("map", k), 4) for k in ("coverage", "precision", "size")],
                   "map reached the target (subjects)": sum(per[n][q]["map"]["reached"] for n in names),
                   "size change % (of the means)": round((m("map", "size") / m("as stated", "size") - 1) * 100, 1),
                   "size change % (mean of the subjects' ratios)": round(float(np.mean([per[n][q]["map"]["size"] / per[n][q]["as stated"]["size"] - 1 for n in names])) * 100, 1),
                   "precision change (map - as stated)": round(m("map", "precision") - m("as stated", "precision"), 4)}
    summ["A alone (coverage, precision, size)"] = [round(float(np.mean([per[n]["A alone (coverage, precision, size)"][i] for n in names])), 4) for i in range(3)]
    return {"term": term, "subjects": names, "per subject": per, "summary": summ}


# ----------------------------------------------------------------------------------------------- the floor

def presence(path, grid):
    from ensemble_grid_check import sample, RHO
    from tractline import ensemble as E
    from tractline.labelers.base import LUT
    rec = E.RunRecord.load(path, logp=False, fibers=True)
    Pt, Ot = sample(rec.fibers)
    pt = np.asarray(LUT)[rec.cluster][Ot]
    out = np.zeros((CC.NAMED, int(np.prod(grid.shape))), bool)
    for k in range(CC.NAMED):
        s = pt == k
        if s.any():
            out[k] = grid.density(grid.counts(Pt[s], Ot[s])).ravel() >= RHO
    return out


def overlap(E_, T):
    both = (E_ & T).sum()
    return [both / T.sum(), both / max(E_.sum(), 1), E_.sum() / T.sum()]


def pairs_score(pres, kinds, hashes):
    """pres {tag: presence}; kinds(t1, t2) -> label or None. Every ordered pair: (coverage, precision, size)."""
    out = {}
    for a, b in itertools.permutations(pres, 2):
        k = kinds(a, b)
        if k:
            out.setdefault(k, []).append(overlap(pres[a], pres[b]))
    return {k: [round(float(x), 4) for x in np.mean(v, 0)] + [len(v)] for k, v in out.items()}


def floor(hashes):
    import nibabel as nib
    from ensemble_grid_check import Grid
    from tractline import ensemble as E
    from _masivar import scan_files
    res = {}
    # ON-Harmony: the same-scanner rescans of the four six-session subjects
    for n, sc in OH.RESCAN.items():
        R = OH.Records(n, sc)
        img = nib.load(OH.files(n, R.pair.ref[0])["AP"])
        A0 = E.RunRecord.load(R.path("A"), logp=False)
        grid = Grid("dwi", img.affine, A0.shape)
        tags = [CII.rescan_tag(t, REG) for t in R.pair.other if OH.term_of(sc, t) == OH.TERM_SAME and R.have(f"{CII.rescan_tag(t, REG)}_onA")]
        pres = {t: presence(R.path(f"{t}_onA"), grid) for t in tags}
        hashes.update({f"{R.dir.relative_to(DATA)}/{t}_onA.npz": CR.sha256(R.path(f"{t}_onA")) for t in tags})
        res.setdefault("ON-Harmony same scanner", {})[n] = pairs_score(pres, lambda a, b: "same scanner, other session", hashes)
        print("floor ON-Harmony", n, res["ON-Harmony same scanner"][n], flush=True)
    # cohort II: the two sessions of one scanner, both placed on the other scanner's A
    for src in ("B32", "A32", "B2000", "A2000r19"):
        name = f"cohort II {XC.DIRECTIONS[src]} sessions 1 and 2 (on {src}'s A)"
        for n in SUBS2:
            R = XC.Records(n, src)
            img = nib.load(CII.series_files(n, CII.pairs(src)[n].ref[0], src)["APP"][0][1])
            A0 = E.RunRecord.load(R.path("A"), logp=False)
            grid = Grid("dwi", img.affine, A0.shape)
            tags = [t for t in (f"x{XC.scanner(XC.DIRECTIONS[src])}1reg3", f"x{XC.scanner(XC.DIRECTIONS[src])}2reg3") if R.have(f"{t}_onA")]
            if len(tags) < 2:
                continue
            pres = {t: presence(R.path(f"{t}_onA"), grid) for t in tags}
            hashes.update({f"{R.dir.relative_to(DATA)}/{t}_onA.npz": CR.sha256(R.path(f"{t}_onA")) for t in tags})
            res.setdefault(name, {})[n] = pairs_score(pres, lambda a, b: "same scanner, other session", hashes)
            print("floor", name, n, res[name][n], flush=True)
    # cohort III: subjects with two or more rescans
    for n, pair in CR.CARD.items():
        if len(pair.rescans) < 2:
            continue
        R = CR.Records(n)
        img = nib.load(scan_files(n, *pair.ref)["APP"][0][1])
        A0 = E.RunRecord.load(R.path("A"), logp=False)
        grid = Grid("dwi", img.affine, A0.shape)
        tags = [t for t in pair.rescans if R.have(f"{t}_onA")]
        pres = {t: presence(R.path(f"{t}_onA"), grid) for t in tags}
        hashes.update({f"{n}/{t}_onA.npz": CR.sha256(R.path(f"{t}_onA")) for t in tags})
        kind = lambda a, b: ("same-session rescan against an other-session one" if (a in pair.same) != (b in pair.same)  # noqa: E731
                             else "two other-session rescans" if a not in pair.same else None)
        res.setdefault("cohort III", {})[n] = pairs_score(pres, kind, hashes)
        print("floor cohort III", n, res["cohort III"][n], flush=True)
    summ = {}
    for g, rows in res.items():
        kinds = sorted({k for r in rows.values() for k in r})
        summ[g] = {k: {"subjects": [n for n, r in rows.items() if k in r], "ordered pairs": int(sum(r[k][3] for r in rows.values() if k in r)),
                       "coverage, precision, size (mean over subjects)": [round(float(np.mean([r[k][i] for r in rows.values() if k in r])), 4) for i in range(3)]}
                   for k in kinds}
    return {"per subject (coverage, precision, size, ordered pairs)": res, "summary": summ}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--no-floor", action="store_true")
    args = ap.parse_args()
    pat = CC.pat16_table()
    conds = conditions()
    res = {"what": "the size and precision of recalibrated envelopes against as-stated ones at the same target",
           "targets": list(TARGETS), "definitions": {"coverage": "rescan presence voxels inside the envelope / rescan presence voxels",
                                                    "precision": "envelope voxels inside the rescan's presence / envelope voxels",
                                                    "size": "envelope voxels / rescan presence voxels",
                                                    "pooling": "the 42 named tracts pooled; a subject's rescans of the term averaged; subjects averaged"},
           "conditions": {}}
    inputs = {}
    for name, (term, S) in conds.items():
        res["conditions"][name] = condition_rows(term, S, pat)
        inputs[name] = {n: {"statements": str(s.path.relative_to(DATA)), "sha256": CR.sha256(s.path)} for n, s in S.items()}
        sm = res["conditions"][name]["summary"]
        print(name, len(S), {q: (sm[q]["as stated (coverage, precision, size)"], sm[q]["map (coverage, precision, size)"], sm[q]["size change % (of the means)"],
                                 sm[q]["precision change (map - as stated)"]) for q in ("0.8", "0.9", "0.95")}, sm["A alone (coverage, precision, size)"], flush=True)
    hashes = {}
    if not args.no_floor:
        res["floor: one rescan's presence as the outline for another"] = floor(hashes)
    run = lambda *a: subprocess.run(["git", *a], cwd=HERE.parent, capture_output=True, text=True).stdout.rstrip()  # noqa: E731
    res["provenance"] = {"git": CR.git_state(), "git status": [x for x in run("status", "--porcelain", "--", "bench/calibration_precision.py").splitlines() if x],
                         "calibration_precision.py sha256": CR.sha256(Path(__file__).resolve()),
                         "module sha256": {m: CR.sha256(HERE / m) for m in ("calibration_cohortII.py", "calibration_card.py", "calibration_crosssite.py",
                                                                            "calibration_crossscanner.py", "calibration_onharmony.py")},
                         "registration mode": REG, "statements": inputs, "floor records sha256": hashes,
                         "created (UTC)": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(args.out).write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
