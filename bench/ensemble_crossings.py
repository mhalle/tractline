"""Where do a seed's versions part, and is it at crossing fibres? (NOTES 2026-10-08, "Where the paths part")

    uv run --group dipy bench/ensemble_crossings.py [--flip smooth] [--seeds 4000]

From existing records; no pipeline runs. Ensembles: MASiVar's references (A with <flip>/r001..r008, ensemble_retest.py)
and PAT16's (added-<flip>/r000..r016, ensemble_pilot.py). For seeds every run labeled (a random `seeds` of them):
each run's path, densified, is followed outward from the seed voxel's centre in both directions by arc length, every
2 mm to 80 mm. At each distance the spread is the median, over the other runs, of their point's distance to the
reference's. A side's divergence is the first distance where the spread exceeds DIVERGE_MM; its place is the
reference's point there.

Crossings: on the reference's corrected DWI, the tracked shell's constant-solid-angle ODF (dipy, SH order 6) and its
peaks (relative threshold 0.5, 25 degrees apart, in the tracking mask); a voxel with two or more peaks is a crossing.
dipy runs in a process of its own (OpenMP, as _register.py). Reported: how far versions stay together; the crossing
share at divergence points against along the reference paths before them (the baseline), and against the brain.
Writes results/ensemble_crossings_<flip>.json.
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/ensemble_crossings_{}.json"
DIVERGE_MM, STEP_MM, MAX_MM, DENSE = 4.0, 2.0, 80.0, 100


def peaks_main(src, dst):
    """In its own process: dipy's CSA ODF peaks -> the number of peaks per voxel."""
    from dipy.core.gradients import gradient_table
    from dipy.reconst.shm import CsaOdfModel
    from dipy.direction import peaks_from_model
    from dipy.data import default_sphere
    z = np.load(src)
    gtab = gradient_table(z["bval"], bvecs=z["bvec"].T, b0_threshold=50)
    pk = peaks_from_model(CsaOdfModel(gtab, sh_order_max=6), z["dwi"], default_sphere, relative_peak_threshold=0.5,
                          min_separation_angle=25, mask=z["mask"], npeaks=3)
    np.savez(dst, npeaks=(pk.peak_values > 0).sum(-1).astype(np.int8))


def crossing_map(s, shell):
    """(X, Y, Z) number of fibre peaks, from the subject's corrected DWI at the tracked shell (and its b0s)."""
    from tractline import pipeline as P
    from tractline.prep import prepare
    with P.exact_float32("mps"):
        corr = P.correct(s, P.Timer(), "mps")
        mask = prepare(corr.dwi, s.affine, s.bval, s.bvec, shell=shell, device="mps").mask
    keep = (s.bval < 50) | (np.abs(s.bval - shell) < 50)
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(a, dwi=np.asarray(corr.dwi, np.float32)[..., keep], bval=s.bval[keep], bvec=s.bvec[:, keep], mask=mask)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "peaks", str(a), str(b)], check=True)
        return np.load(b)["npeaks"], mask


def seed_world(keys, affine):
    k, j, i = keys >> 24, (keys >> 12) & 4095, keys & 4095                 # seed_keys: (k, j, i), the binary's order
    ijk = np.stack([i, j, k], 1).astype(float)
    return ijk @ affine[:3, :3].T + affine[:3, 3]


def walk(path, seed):
    """A densified path's points at arc distances 0, STEP_MM, .. MAX_MM from its point nearest the seed, both ways:
    (2, n_steps, 3), NaN beyond its end."""
    i0 = int(np.argmin(np.linalg.norm(path - seed, axis=1)))
    steps = np.arange(0, MAX_MM + 1e-9, STEP_MM)
    out = np.full((2, len(steps), 3), np.nan)
    for side, seg in enumerate((path[i0:], path[: i0 + 1][::-1])):
        if len(seg) < 2:
            continue
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(seg, axis=0), axis=1))]
        ok = steps <= s[-1]
        out[side, ok] = np.stack([np.interp(steps[ok], s, seg[:, a]) for a in range(3)], 1)
    return out


def analyze(paths_by_run, keys, affine, npeaks, rng, n_seeds):
    from tractline import ensemble as E
    common = keys[0]
    for k in keys[1:]:
        common = np.intersect1d(common, k)
    pick = rng.choice(common, min(n_seeds, len(common)), replace=False)
    seeds = seed_world(pick, affine)
    t, s20 = np.linspace(0, 1, DENSE), np.linspace(0, 1, E.POINTS)
    dense = lambda p: np.stack([np.interp(t, s20, p[:, a]) for a in range(3)], 1)  # noqa: E731
    rows = [{int(k): i for i, k in enumerate(kk)} for kk in keys]
    inv = np.linalg.inv(affine)
    to_vox = lambda w: np.rint(w @ inv[:3, :3].T + inv[:3, 3]).astype(int)  # noqa: E731
    shape = npeaks.shape
    div_d, div_cross, base_cross, never = [], [], [], 0
    for key, seed in zip(pick, seeds):
        W = [walk(dense(paths_by_run[r][rows[r][int(key)]]), seed) for r in range(len(keys))]
        ref = W[0]
        for side in (0, 1):
            others = []
            for w in W[1:]:                                             # the other run's side nearer the reference's first 10 mm
                a, b = np.nanmean(np.linalg.norm(w[0, 1:6] - ref[side, 1:6], axis=1)), np.nanmean(np.linalg.norm(w[1, 1:6] - ref[side, 1:6], axis=1))
                others.append(w[0] if not (b < a) else w[1])
            spread = np.nanmedian(np.stack([np.linalg.norm(o - ref[side], axis=1) for o in others]), 0)
            reach = np.isfinite(ref[side, :, 0]) & np.isfinite(spread)
            over = np.flatnonzero(reach & (spread > DIVERGE_MM))
            last = np.flatnonzero(reach)
            if len(last) < 3:
                continue
            j = over[0] if len(over) else None
            stop = j if j is not None else last[-1] + 1
            v = to_vox(ref[side, 1:stop]); ok = (v >= 0).all(1) & (v < shape).all(1)
            base_cross += (npeaks[tuple(v[ok].T)] >= 2).tolist()
            if j is None:
                never += 1
                continue
            div_d.append(j * STEP_MM)
            vj = to_vox(ref[side, j])
            if (vj >= 0).all() and (vj < shape).all():
                div_cross.append(npeaks[tuple(vj)] >= 2)
    div_d = np.array(div_d)
    return {"seeds": len(pick), "sides": 2 * len(pick),
            "sides diverging within 80 mm": round(len(div_d) / max(len(div_d) + never, 1), 4),
            "divergence distance mm (quartiles)": [float(v) for v in np.quantile(div_d, [0.25, 0.5, 0.75])] if len(div_d) else None,
            "crossing share: at divergence": round(float(np.mean(div_cross)), 4),
            "crossing share: along paths before divergence": round(float(np.mean(base_cross)), 4)}


def main(args):
    sys.path.insert(0, str(HERE))
    from tractline import ensemble as E
    from tractline.data import DATA
    from ensemble_retest import PAIRS
    from _masivar import load as load_m
    from _ds001226 import load as load_d
    rng = np.random.default_rng(0)
    res = {"flip": args.flip, "diverge_mm": DIVERGE_MM}
    cases = []
    for pair, (subject, (ses, scan), _, shells, shell) in PAIRS.items():
        d = DATA / "ensemble" / subject / "retest" / pair
        cases.append((pair, lambda subject=subject, ses=ses, scan=scan, shells=shells: load_m(subject, ses, scan, shells), shell,
                      [d / "A.npz"] + [d / args.flip / f"r{r:03d}.npz" for r in range(1, 9)]))
    first = DATA / "ensemble" / "PAT16" / (f"added-{args.flip}" if args.flip != "voxel" else "added")
    cases.append(("PAT16", lambda: load_d("PAT16"), 2800.0, [first / f"r{r:03d}.npz" for r in range(0, 17)]))
    for name, loader, shell, paths in cases:
        s = loader()
        npeaks, mask = crossing_map(s, shell)
        recs = [E.RunRecord.load(p, logp=False) for p in paths]
        r = analyze([x.paths for x in recs], [x.keys for x in recs], s.affine, npeaks, rng, args.seeds)
        r["crossing share: brain (tracking mask)"] = round(float((npeaks[mask] >= 2).mean()), 4)
        r["runs"] = len(recs)
        res[name] = r
        print(name, json.dumps(r), flush=True)
        del recs, s
    Path(str(OUT).format(args.flip)).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "peaks":
        peaks_main(sys.argv[2], sys.argv[3])
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--flip", default="smooth")
        ap.add_argument("--seeds", type=int, default=4000)
        main(ap.parse_args())
