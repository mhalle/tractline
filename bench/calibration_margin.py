"""The ceiling of an ensemble's outlines, on the cohort II records at one registration mode (default
calibration_cohortII.DEFAULT_REGISTRATION, 3): a margin step past the widest outline, and the widest outline's coverage
against the number of runs. Read-only on the records; CPU only (no pipeline runs).

    uv run --group dipy --group rankfield bench/calibration_margin.py [--registration 3] [--out results/calibration_margin.json]
                                                                      [--compare <earlier output>]

Pairs: per protocol (B2000, B32, A32, A2000r19) and subject (cIIs01-04), A's 17-run ensemble against its same-scanner
other-session rescan; and B2000's ensemble against the site 3 (Siemens) rescan (calibration_crosssite.py's records).
Each pair's statements and rescan record are the ones calibration_cohortII.statements_file resolves at the mode (mode 3:
its own statements_R17_reg3.npz where the pair was rerun, else mode 2's, whose transform it confirmed); the rescan record
is the <tag>_onA.npz those statements name.
Presence is recomputed from the records as calibration_card.run_stats does (density >= RHO per named tract), and the
union of the runs' density >= RHO / 2 besides. Per pair, against the rescan's presence (pooled over the 42 named tracts):
  - steps past the widest outline (U, the union of the 17 runs), each restricted to A's mask: "half step" U plus the
    6-neighbors of U where some run has density >= RHO / 2; "1 voxel (6-neighbors)" U dilated by the 6-neighborhood
    (2.5 mm on a 2.5 mm grid); "1 voxel (26-neighbors)" U dilated by the 26-neighborhood; coverage, precision and size
    (the step's voxels over the rescan's). A distance-transform half voxel (1.25 mm) adds nothing on these grids;
  - the envelope ladder (calibration_card.curves on the pair's statements: coverage and precision at each stated level
    on calibration_card.GRID) extended by the four steps, which summary's LOSO map reads;
  - the widest outline's coverage over R' runs (A always included, the rest drawn without replacement from r001-r016;
    R' = 1, 2, 4, 8, 12, 17; DRAWS draws each, one at R' = 1 and 17), mean and SD over draws.
The widest coverage recomputed here is checked against the statements' ("statements: widest coverage (check)").
summary: per group the means of the steps, the leave-one-subject-out map on the ladder without and with the steps at
targets 0.8, 0.9, 0.95 (the first ladder step whose training mean coverage reaches the target; the last step if none
does: "not reached"), and the R' means. No asymptote is fitted: the curves still rise at R' = 17 and saturating fits are
not stable (a hyperbola's asymptote moves by 0.4-1.4 points between fitting R' = 1-17 and 4-17; a power law exceeds 1).
--compare: the per-pair rows must equal those of an earlier output (summary and provenance may differ).
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_card as CC   # noqa: E402
import calibration_cohortII as CII   # noqa: E402  (no side effects at import)
import calibration_runs as CR   # noqa: E402

OUT = HERE / "results/calibration_margin.json"
PROTOCOLS = ("B2000", "B32", "A32", "A2000r19")
SUBJECTS = ("cIIs01", "cIIs02", "cIIs03", "cIIs04")
RS, DRAWS, SEED, RUNS = (1, 2, 4, 8, 12, 17), 20, 0, 16
STEPS = ("widest", "half step", "1 voxel (6-neighbors)", "1 voxel (26-neighbors)")
TARGETS = (0.8, 0.9, 0.95)
SITE3 = "site 3 (B2000 ensemble)"


def presence(path, grid):
    """(density >= RHO, density >= RHO / 2), each (NAMED, nvox) bool, of one run record on the grid."""
    from ensemble_grid_check import sample, RHO
    from tractline import ensemble as E
    from tractline.labelers.base import LUT
    rec = E.RunRecord.load(path, logp=False, fibers=True)
    Pt, Ot = sample(rec.fibers)
    pt = np.asarray(LUT)[rec.cluster][Ot]
    n = int(np.prod(grid.shape))
    full, half = np.zeros((CC.NAMED, n), bool), np.zeros((CC.NAMED, n), bool)
    for k in range(CC.NAMED):
        s = pt == k
        if s.any():
            d = grid.density(grid.counts(Pt[s], Ot[s])).ravel()
            full[k], half[k] = d >= RHO, d >= RHO / 2
    return full, half


def scores(E_, T):
    both, size, t = (E_ & T).sum(), E_.sum(), T.sum()
    return {"coverage": both / t, "precision": both / max(size, 1), "size": size / t}


def dilate(U, shape, connectivity, mask):
    from scipy.ndimage import binary_dilation, generate_binary_structure
    st = generate_binary_structure(3, connectivity)
    return np.stack([binary_dilation(u.reshape(shape), st).ravel() & mask if u.any() else u for u in U])


def pairs_rows(registration, draws, seed, inputs):
    """The per-pair rows, {group: {subject: row}}; inputs collects each pair's files and their SHA-256."""
    import nibabel as nib
    from ensemble_grid_check import Grid
    from tractline import ensemble as E
    from tractline.data import DATA
    rng = np.random.default_rng(seed)
    out = {}
    for key in PROTOCOLS:
        for sub in SUBJECTS:
            R = CII.Records(sub, key)
            paths = R.ensemble("", RUNS)
            img = nib.load(CII.series_files(sub, R.pair.ref[0], key)["APP"][0][1])
            A0 = E.RunRecord.load(paths[0], logp=False)
            grid = Grid("dwi", img.affine, A0.shape)
            mask = np.load(R.path("A_mask", ".npy")).ravel()
            runs = [presence(p, grid) for p in paths]
            P = np.stack([r[0] for r in runs])
            U, Uh = P.any(0), np.stack([r[1] for r in runs]).any(0)
            del runs
            d6, d26 = dilate(U, A0.shape, 1, mask), dilate(U, A0.shape, 3, mask)
            steps = dict(zip(STEPS, (U, U | (d6 & Uh), U | d6, U | d26)))
            targets = [(f"{key} same scanner", R.dir)]
            if key == "B2000":
                targets.append((SITE3, DATA / f"ensemble/{sub}/card/{sub}-B2000-site3"))
            for gname, d in targets:
                sf = CII.statements_file(d, RUNS + 1, registration)
                z = np.load(sf)
                tag = list(json.loads(str(z["meta"]))["rescans"])[0]
                rescan = d / f"{tag}_onA.npz"
                T = presence(rescan, grid)[0]
                row = {k: {m: round(float(v), 4) for m, v in scores(st, T).items()} for k, st in steps.items()}
                cv, sz = CC.curves(z["e_cnt"], z[f"e_hit_{tag}"])
                row["statements"] = sf.name
                row["rescan record"] = rescan.name
                row["statements: widest coverage (check)"] = round(float(cv[-1]), 4)
                row["_ladder"] = {"coverage": cv[:-1].tolist() + [row[k]["coverage"] for k in steps],
                                  "precision": (cv[:-1] / np.maximum(sz[:-1], 1e-9)).tolist() + [row[k]["precision"] for k in steps]}
                sub_r = {}
                for r in RS:
                    cov = []
                    for _ in range(1 if r in (1, RUNS + 1) else draws):
                        idx = np.r_[0, rng.choice(np.arange(1, RUNS + 1), r - 1, replace=False)] if r > 1 else np.array([0])
                        cov.append((P[idx].any(0) & T).sum() / T.sum())
                    sub_r[str(r)] = [round(float(np.mean(cov)), 4), round(float(np.std(cov)), 4)]
                row["widest coverage by R' (mean, sd over draws)"] = sub_r
                out.setdefault(gname, {})[sub] = row
                inputs[f"{gname} {sub}"] = {"folder": str(d.relative_to(DATA)), "statements": sf.name, "statements sha256": CR.sha256(sf),
                                            "rescan record": rescan.name, "rescan record sha256": CR.sha256(rescan)}
                print(gname, sub, sf.name, {k: (row[k]["coverage"], row[k]["precision"], row[k]["size"]) for k in STEPS},
                      "check", row["statements: widest coverage (check)"], {k: v[0] for k, v in sub_r.items()}, flush=True)
    return out


def loso(rows, n_steps):
    """Leave one subject out on the ladder (envelope levels, then the first n_steps steps): per target the mean held-out
    coverage and precision, and the steps used."""
    names = [f"stated {g:g}" for g in CC.GRID] + list(STEPS)
    cut = len(CC.GRID) + n_steps
    subs = list(rows)
    out = {}
    for q in TARGETS:
        cov, prec, used = [], [], []
        for s in subs:
            train = np.mean([np.array(rows[m]["_ladder"]["coverage"])[:cut] for m in subs if m != s], 0)
            ok = np.flatnonzero(train >= q)
            i = int(ok[0]) if len(ok) else cut - 1
            cov.append(rows[s]["_ladder"]["coverage"][i])
            prec.append(rows[s]["_ladder"]["precision"][i])
            used.append(names[i] + ("" if len(ok) else " (not reached)"))
        out[f"{q:g}"] = {"coverage": round(float(np.mean(cov)), 4), "precision": round(float(np.mean(prec)), 4), "steps used": sorted(set(used))}
    return out


def summary(res):
    out = {}
    for g, rows in res.items():
        subs = list(rows)
        out[g] = {"subjects": subs, "statements": {s: rows[s]["statements"] for s in subs},
                  "widest equals the statements' (all pairs)": all(abs(rows[s]["widest"]["coverage"] - rows[s]["statements: widest coverage (check)"]) < 1e-4 for s in subs),
                  "steps, mean (coverage, precision, size)": {k: [round(float(np.mean([rows[s][k][m] for s in subs])), 4) for m in ("coverage", "precision", "size")] for k in STEPS},
                  "LOSO map, envelopes and the widest outline": loso(rows, 1),
                  "LOSO map, with the margin steps": loso(rows, len(STEPS)),
                  "widest coverage by R', mean over subjects": {str(r): round(float(np.mean([rows[s]["widest coverage by R' (mean, sd over draws)"][str(r)][0] for s in subs])), 4) for r in RS}}
    return out


def provenance(inputs, registration, draws, seed):
    run = lambda *a: subprocess.run(["git", *a], cwd=HERE.parent, capture_output=True, text=True).stdout.rstrip()  # noqa: E731
    return {"git": CR.git_state(), "git status of this script and calibration_cohortII.py": [
                x for x in run("status", "--porcelain", "--", "bench/calibration_margin.py", "bench/calibration_cohortII.py").splitlines() if x],
            "calibration_margin.py sha256": CR.sha256(Path(__file__).resolve()), "calibration_cohortII.py sha256": CII.module_sha256(),
            "registration mode": registration, "draws": draws, "seed": seed, "inputs": inputs,
            "created (UTC)": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def compare(new, path):
    """The per-pair rows of `new` against an earlier output's: (equal, differences)."""
    old = json.loads(Path(path).read_text())
    diffs = []
    for g in sorted(set(old) | set(new)):
        if g in ("summary", "provenance"):
            continue
        if g not in old or g not in new:
            diffs.append(f"group {g} only in one")
            continue
        for s in sorted(set(old[g]) | set(new[g])):
            if old[g].get(s) != new[g].get(s):
                keys = sorted(k for k in set(old[g].get(s, {})) | set(new[g].get(s, {})) if old[g].get(s, {}).get(k) != new[g].get(s, {}).get(k))
                diffs.append(f"{g} {s}: {keys}")
    return not diffs, diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registration", type=int, default=CII.DEFAULT_REGISTRATION)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--compare", help="an earlier output whose per-pair rows must equal this run's")
    args = ap.parse_args()
    inputs = {}
    res = pairs_rows(args.registration, args.draws, args.seed, inputs)
    out = {**res, "summary": summary(res), "provenance": provenance(inputs, args.registration, args.draws, args.seed)}
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    print(f"wrote {args.out}")
    if args.compare:
        ok, diffs = compare(out, args.compare)
        print(f"per-pair rows equal to {args.compare}: {ok}" + ("" if ok else f" {diffs}"))
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
