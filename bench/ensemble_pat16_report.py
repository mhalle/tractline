"""PAT16's tracts at its tumour, with the ensemble's uncertainty, calibrated on real rescans (NOTES 2026-10-08, "PAT16 at
the tumour").

    uv run --group dipy bench/ensemble_pat16_report.py [--flip smooth]

The ensemble: ensemble_pilot.py's added-<flip>/r000..r016 (r000 the scan as acquired, what the pipeline outputs
today). Each run's tract per voxel: its winner (VoxelEnsemble.winner) where the winner has at least TAU streamlines
(1, as calibrated in ensemble_coverage.py; stray single streamlines then count, so distances also use TAU_DIST). The
tumour: ensemble_where_which.py's DATA/ensemble/PAT16/tumour_dwi.npz.

Per named tract within 20 mm of the tumour in any run (ordered by the ensemble's mass within 5 mm); a tract's voxels
are those it wins with at least TAU_DIST streamlines:
  today       r000's distance from the tumour (mm, nearest voxel centre; 0 inside it), and its volume inside the
              tumour and within 5 mm of it (mL);
  runs        the same in each of the 17 runs: median and range; P(enters the tumour) - the share of runs with
              ENTER_VOXELS or more voxels inside it - and P(within 5 mm);
  envelope    P(v, k) over the runs; the highest-probability voxels holding L of the tract's mass, for the stated L
              and for the level that covered L on real rescans (results/ensemble_coverage_<flip>.json, "rescan":
              interpolated; beyond its largest measured level, flagged): its volume inside the tumour and within 5 mm.
The margin (0-10 mm): the tract mix of the streamlines crossing it, per run, and its range. Writes
results/ensemble_pat16_report_<flip>.json and DATA/ensemble/PAT16/report_<flip>.npz (P maps, distance map, tumour)
for the viewer.
"""
import argparse, json
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import distance_transform_edt
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import TRACT_NAMES

HERE = Path(__file__).resolve().parent
OUT = HERE / "results/ensemble_pat16_report_{}.json"
TAU, TAU_DIST, NEAR_MM, ENTER_VOXELS = 1.0, 2.0, 20.0, 3
STATED = (0.5, 0.9)


def hdr(w, level):
    """The highest-w voxels holding `level` of w's mass (bool)."""
    order = np.argsort(-w, kind="stable")
    inside = np.zeros(len(w), bool)
    inside[order[: int(np.searchsorted(np.cumsum(w[order]), level * w.sum())) + 1]] = True
    return inside & (w > 0)


def corrected_level(level, flip):
    """The stated level whose envelope covered `level` on real rescans (the coverage curve, inverted by interpolation):
    (level to use, flag)."""
    f = HERE / f"results/ensemble_coverage_{flip}.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    curve = res.get("recalibration on real rescans (leave one pair out)", {}).get("coverage curve, all rescan scorings (level: coverage)")
    if not curve:
        return level, "no coverage measured"
    xs = np.array([float(k) for k in curve]); ys = np.array(list(curve.values()))
    if level > ys.max():
        return float(xs.max()), f"rescans covered at most {ys.max():.2f}: the widest measured envelope used"
    return float(np.interp(level, ys, xs)), "interpolated"


def main(args):
    first = DATA / "ensemble" / "PAT16" / (f"added-{args.flip}" if args.flip != "voxel" else "added")
    names = [f"r{r:03d}" for r in range(17)]
    img = nib.load(DATA / "ds001226/sub-PAT16/ses-preop/dwi/sub-PAT16_ses-preop_acq-AP_dwi.nii.gz")
    aff, shape = img.affine, img.shape[:3]
    vox = np.sqrt((aff[:3, :3] ** 2).sum(0))
    tumour = np.load(DATA / "ensemble" / "PAT16" / "tumour_dwi.npz")["tumour"]
    dist = distance_transform_edt(~tumour, sampling=vox).ravel()
    V = E.VoxelEnsemble(shape, TAU)
    wins, wins_d, comps = [], [], []
    margin = (dist > 0) & (dist <= 10)
    for n in names:
        rec = E.RunRecord.load(first / f"{n}.npz", logp=False)
        V.add(rec)
        wins.append(V.winner(rec.counts))
        w2 = rec.counts.argmax(0)
        wins_d.append(np.where(rec.counts.max(0) >= TAU_DIST, w2, E.NONE))
        h = rec.counts[:, margin].sum(1).astype(float)
        comps.append(h / max(h.sum(), 1))
    P = V.vote.astype(np.float32) / V.runs
    comps = np.array(comps)
    levels = {L: corrected_level(L, args.flip) for L in STATED}
    res = {"flip": args.flip, "runs": V.runs, "tumour_voxels": int(tumour.sum()), "tau": TAU, "tau_distance": TAU_DIST,
           "envelope levels": {f"{L:g}": {"use": round(u, 3), "how": how} for L, (u, how) in levels.items()}, "tracts": {}}
    inside, near = tumour.ravel(), dist <= 5
    ml = float(np.prod(vox)) / 1000
    for k in range(E.N_TRACTS - 1):
        d_runs = np.array([dist[w == k].min() if (w == k).any() else np.inf for w in wins_d])
        if not np.isfinite(d_runs).any() or d_runs.min() > NEAR_MM:
            continue
        in_runs = np.array([((w == k) & inside).sum() for w in wins_d])
        near_runs = np.array([((w == k) & near).sum() for w in wins_d])
        row = {"today (r000): distance mm": round(float(d_runs[0]), 1) if np.isfinite(d_runs[0]) else None,
               "today (r000): mL inside the tumour": round(in_runs[0] * ml, 2),
               "today (r000): mL within 5 mm": round(near_runs[0] * ml, 2),
               "runs: distance mm, median [min, max]": [round(float(np.median(d_runs)), 1), round(float(d_runs.min()), 1),
                                                        round(float(d_runs.max()), 1) if np.isfinite(d_runs.max()) else None],
               "runs: P(enters the tumour)": round(float((in_runs >= ENTER_VOXELS).mean()), 3),
               "runs: P(within 5 mm)": round(float((d_runs <= 5).mean()), 3),
               "runs: mL inside the tumour, median [min, max]": [round(float(np.median(in_runs)) * ml, 2), round(in_runs.min() * ml, 2), round(in_runs.max() * ml, 2)],
               "runs: mL within 5 mm, median [min, max]": [round(float(np.median(near_runs)) * ml, 2), round(near_runs.min() * ml, 2), round(near_runs.max() * ml, 2)],
               "_mass_near": float(P[k][near].sum())}
        for L, (u, how) in levels.items():
            for tag, lev in (("stated", L), ("calibrated", u)):
                env = hdr(P[k], lev)
                row[f"envelope {L:g} ({tag}): mL inside the tumour / within 5 mm"] = [round(float((env & inside).sum()) * ml, 2), round(float((env & near).sum()) * ml, 2)]
        res["tracts"][TRACT_NAMES[k]] = row
    res["tracts"] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                     for k, v in sorted(res["tracts"].items(), key=lambda kv: -kv[1]["_mass_near"])}
    top = np.argsort(-comps.mean(0))[:8]
    res["margin 0-10 mm: tract mix (mean, min-max over runs)"] = {
        TRACT_NAMES[k]: [round(float(comps[:, k].mean()), 3), round(float(comps[:, k].min()), 3), round(float(comps[:, k].max()), 3)] for k in top}
    np.savez_compressed(DATA / "ensemble" / "PAT16" / f"report_{args.flip}.npz", P=P.reshape(-1, *shape).astype(np.float16), dist=dist.reshape(shape).astype(np.float32),
                        tumour=tumour, today=wins[0].reshape(shape).astype(np.int8), affine=aff)
    Path(str(OUT).format(args.flip)).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip", default="smooth")
    main(ap.parse_args())
