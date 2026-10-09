"""Labels held as probabilities, not winners: are the pipeline's outputs steadier across real rescans when the
labeler's tract probabilities are carried forward instead of its argmax? (NOTES 2026-10-08, "Soft labels against
rescans")

    uv run --group dipy bench/ensemble_soft.py [--pairs cIIIs01-2.5 cIIIs01-2.1 cIs1]

From records ensemble_retest.py left (DATA/ensemble/<subject>/retest/<pair>/); no pipeline runs. For each pair of runs
on one grid - the reference A against each rescan resampled onto A's grid (<tag>_onA), and replicate against
replicate (disjoint pairs, per-voxel and smooth signs) - two ways to read the same records:

  hard  each streamline its winning tract (what the pipeline outputs today);
  soft  each streamline its tract probabilities (the 1600 clusters' summed into 43 tracts, "Other" one of them).

Per streamline (matched by seed key): hard, whether the winner changed; soft, the total variation distance between
the two probability vectors (half the summed absolute difference: 0 the same belief, 1 disjoint). Per voxel: each
tract's density - streamlines visiting the voxel, each counted as 1 for its winner (hard) or as its probability for
every tract (soft); paths are the records' 20 points, densified to 100 - and the voxel's composition (the densities
normalized). Compared between runs: the composition's total variation distance, weighted by the voxels' density,
and per tract the histogram intersection of its density map (sum of minima over sum of maxima; 1 identical), weighted
by the tract's mass. Sharpness, so that steadiness is not bought by hedging: the mean probability of the top tract
(per streamline) and the mean share of the top tract (per voxel). Writes results/ensemble_soft.json.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, nibabel as nib
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS  # noqa: E402

OUT = HERE / "results/ensemble_soft.json"
ONEHOT = np.eye(E.N_TRACTS)[LUT]                    # (1600, 43): cluster -> tract
DENSE = 100


def tract_probs(rec):
    """(n, 43) float32 tract probabilities from a record's cluster log-probabilities."""
    return (np.exp(rec.logp.astype(np.float32)) @ ONEHOT.astype(np.float32)).astype(np.float32)


def visits(paths, affine, shape):
    """(streamline index, voxel index) once per streamline and voxel its path, densified to DENSE points, visits."""
    t = np.linspace(0, 1, DENSE)
    s = np.linspace(0, 1, paths.shape[1])
    dense = np.stack([np.stack([np.interp(t, s, p[:, a]) for a in range(3)], 1) for p in paths]) if len(paths) else np.zeros((0, DENSE, 3))
    inv = np.linalg.inv(affine)
    ijk = np.rint(dense @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64).reshape(-1, 3)
    sid = np.repeat(np.arange(len(paths)), DENSE)
    ok = (ijk >= 0).all(1) & (ijk < np.array(shape)).all(1) & np.isfinite(dense.reshape(-1, 3)).all(1)
    nvox = int(np.prod(shape))
    key = np.unique(sid[ok] * nvox + np.ravel_multi_index(tuple(ijk[ok].T), shape))
    return np.divmod(key, nvox)


def density(P, sv, nvox):
    """(43, nvox) float64: per tract, the summed weight P[s, k] of the streamlines visiting each voxel."""
    s, v = sv
    return np.stack([np.bincount(v, weights=P[s, k].astype(np.float64), minlength=nvox) for k in range(P.shape[1])])


class Run:
    def __init__(self, path, affine, shape):
        rec = E.RunRecord.load(path)
        self.keys = rec.keys
        self.soft = tract_probs(rec)
        self.hard = np.eye(E.N_TRACTS, dtype=np.float32)[LUT[rec.cluster]]
        sv = visits(rec.paths, affine, shape)
        nvox = int(np.prod(shape))
        self.maps = {"hard": density(self.hard, sv, nvox), "soft": density(self.soft, sv, nvox)}
        del rec


def compare(a: Run, b: Run):
    order = np.argsort(a.keys)
    pos = np.searchsorted(a.keys[order], b.keys).clip(0, len(a.keys) - 1)
    hit = a.keys[order][pos] == b.keys
    ia, ib = order[pos[hit]], np.flatnonzero(hit)
    out = {"matched": round(len(ia) / len(a.keys), 4)}
    for kind in ("hard", "soft"):
        pa, pb = getattr(a, kind)[ia], getattr(b, kind)[ib]
        A, B = a.maps[kind], b.maps[kind]
        ta, tb = A.sum(0), B.sum(0)
        either = (ta > 0) | (tb > 0)
        qa = np.where(ta > 0, A / np.maximum(ta, 1e-12), 0)[:, either]
        qb = np.where(tb > 0, B / np.maximum(tb, 1e-12), 0)[:, either]
        tv_vox = 0.5 * np.abs(qa - qb).sum(0)
        tv_vox = np.where((ta[either] > 0) & (tb[either] > 0), tv_vox, 1.0)       # a voxel only one run reaches: disjoint
        w = ta[either] + tb[either]
        mass = A.sum(1) + B.sum(1)
        inter = np.minimum(A, B).sum(1) / np.maximum(np.maximum(A, B).sum(1), 1e-12)
        out[kind] = {
            "streamline: changed (hard) / total variation (soft)": round(float((0.5 * np.abs(pa - pb).sum(1)).mean()), 4),
            "voxel composition: total variation, density-weighted": round(float((tv_vox * w).sum() / w.sum()), 4),
            "tract maps: histogram intersection, mass-weighted": round(float((inter * mass).sum() / mass.sum()), 4),
            "tract maps: histogram intersection, named tracts only": round(float((inter[:-1] * mass[:-1]).sum() / mass[:-1].sum()), 4)}
    return out


def sharpness(r: Run):
    t = r.maps["soft"].sum(0)
    m = t > 0
    return {"streamline: mean top-tract probability": round(float(r.soft.max(1).mean()), 4),
            "streamline: share with top probability < 0.5": round(float((r.soft.max(1) < 0.5).mean()), 4),
            "voxel: mean top-tract share (soft), density-weighted": round(float((r.maps["soft"][:, m].max(0)).sum() / t[m].sum()), 4),
            "voxel: mean top-tract share (hard), density-weighted": round(float((r.maps["hard"][:, m].max(0)).sum() / r.maps["hard"].sum(0)[m].sum()), 4)}


def mean_rows(rows):
    if not rows:
        return None
    out = {"pairs": len(rows), "matched": round(float(np.mean([r["matched"] for r in rows])), 4)}
    for kind in ("hard", "soft"):
        out[kind] = {k: round(float(np.mean([r[kind][k] for r in rows])), 4) for k in rows[0][kind]}
    return out


def main(args):
    sys.path.insert(0, str(HERE))
    from _masivar import scan_files
    res = {}
    for pair in args.pairs:
        subject, (ses, scan), rescans, shells, _ = PAIRS[pair]
        d = DATA / "ensemble" / subject / "retest" / pair
        img = nib.load(scan_files(subject, ses, scan, shells)["APP"][0][1])
        aff, shape = img.affine, img.shape[:3]
        load = lambda name: Run(d / f"{name}.npz", aff, shape)  # noqa: E731
        A = load("A")
        row = {"sharpness (A)": sharpness(A), "rescan": {}}
        for t in rescans:
            if (d / f"{t}_onA.npz").exists():
                row["rescan"][t] = compare(A, load(f"{t}_onA"))
        for flip in ("voxel", "smooth"):
            runs = [load(f"{flip}/r{i:03d}") for i in range(1, 9) if (d / flip / f"r{i:03d}.npz").exists()]
            row[f"replicate vs replicate ({flip} signs)"] = mean_rows([compare(runs[i], runs[i + 1]) for i in range(0, len(runs) - 1, 2)])
        res[pair] = row
        print(pair, json.dumps(row, indent=1), flush=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=list(PAIRS))
    main(ap.parse_args())
