"""Where against which: on a rescan, is the position of the white matter steadier than the name of the tract it
belongs to? (NOTES 2026-10-08, "Where and which")

    uv run --group dipy bench/ensemble_where_which.py [--flip smooth]

From existing records; no pipeline runs. Pairs of runs on one grid:
  - MASiVar (ensemble_retest.py's pairs): the reference A against each rescan resampled onto its grid (<tag>_onA),
    and replicate against replicate (<flip>/r001-r002, ..), the bootstrap's rescan-like pairs;
  - PAT16 (ensemble_pilot.py's added-<flip>/): replicate against replicate, its tumour's margin as the regions. There
    is no rescan of PAT16; with smooth signs a replicate pair changes about as much as a same-session rescan
    (NOTES 2026-10-08, "The bootstrap against real rescans").

Per pair, over the brain and in spheres (radius 8 mm): MASiVar, 300 centred on dense white matter; PAT16, every
margin voxel (0-10 mm from the tumour) at least 4 mm apart.
  where  the streamlines' density, all tracts together (histogram intersection: sum of minima over sum of maxima,
         1 identical), and their local direction (density-weighted |cos| between the two runs' principal directions
         per voxel, from each voxel's orientation tensor of path segments; voxels with >= 5 streamlines in both);
  which  per tract, the same intersection of its density (named tracts, mass-weighted), and the tract composition
         of the streamlines in a region (total variation: 0 the same mix, 1 disjoint) and whether its dominant
         tract is the same.
Paths are the records' 20 points densified to 100. PAT16's tumour (ds001226's mask, on the T1) is put on the DWI
grid by a rigid registration of the T1 to the corrected mean b0 (_register.py), kept in
DATA/ensemble/PAT16/tumour_dwi.npz. Writes results/ensemble_where_which_<flip>.json.
"""
import argparse, json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import affine_transform, distance_transform_edt
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS  # noqa: E402

OUT = HERE / "results/ensemble_where_which_{}.json"
DENSE, RADIUS_MM, N_SPHERES = 100, 8.0, 300


class Run:
    def __init__(self, path, affine, shape):
        rec = E.RunRecord.load(path, logp=False)
        self.shape, nvox = tuple(shape), int(np.prod(shape))
        self.counts = rec.counts.astype(np.float64)                         # (43, nvox)
        self.dens = self.counts.sum(0)
        tract = LUT[rec.cluster]
        t = np.linspace(0, 1, DENSE)
        s = np.linspace(0, 1, rec.paths.shape[1])
        P = np.stack([np.stack([np.interp(t, s, p[:, a]) for a in range(3)], 1) for p in rec.paths])     # (n, DENSE, 3)
        inv = np.linalg.inv(affine)
        mid = 0.5 * (P[:, 1:] + P[:, :-1])
        seg = np.diff(P, axis=1)
        seg /= np.maximum(np.linalg.norm(seg, axis=2, keepdims=True), 1e-9)
        ijk = np.rint(mid.reshape(-1, 3) @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
        ok = (ijk >= 0).all(1) & (ijk < np.array(shape)).all(1)
        v = np.ravel_multi_index(tuple(ijk[ok].T), shape)
        g = seg.reshape(-1, 3)[ok]
        self.tensor = np.stack([np.bincount(v, weights=g[:, i] * g[:, j], minlength=nvox) for i, j in
                                ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))])          # (6, nvox)
        sid = np.repeat(np.arange(len(P)), DENSE - 1)[ok]
        key = np.unique(sid * nvox + v)
        self.vs, self.vv = np.divmod(key, nvox)                              # (streamline, voxel) once
        self.tract = tract

    def direction(self, idx):
        T = self.tensor[:, idx]
        M = np.stack([np.stack([T[0], T[1], T[2]], 1), np.stack([T[1], T[3], T[4]], 1), np.stack([T[2], T[4], T[5]], 1)], 1)
        return np.linalg.eigh(M)[1][:, :, -1]                                # (m, 3) principal direction

    def composition(self, mask):
        s = np.unique(self.vs[mask[self.vv]])
        h = np.bincount(self.tract[s], minlength=E.N_TRACTS).astype(float)
        return h / max(h.sum(), 1)


def inter(a, b):
    return float(np.minimum(a, b).sum() / max(np.maximum(a, b).sum(), 1e-12))


def compare(a: Run, b: Run, spheres):
    named = slice(0, E.N_TRACTS - 1)
    both = (a.dens >= 5) & (b.dens >= 5)
    idx = np.flatnonzero(both)
    cos = np.abs((a.direction(idx) * b.direction(idx)).sum(1))
    w = a.dens[idx] + b.dens[idx]
    ma, mb = a.counts[named].sum(1), b.counts[named].sum(1)
    per_tract = [inter(a.counts[k], b.counts[k]) for k in range(E.N_TRACTS - 1)]
    out = {"brain": {"where: density intersection (all tracts)": round(inter(a.dens, b.dens), 4),
                     "where: direction agreement |cos|": round(float((cos * w).sum() / w.sum()), 4),
                     "which: density intersection per tract (named, mass-weighted)": round(float(np.dot(per_tract, ma + mb) / (ma + mb).sum()), 4)}}
    rows = []
    for m in spheres:
        i = np.flatnonzero(m.ravel() & both)
        ca, cb = a.composition(m.ravel()), b.composition(m.ravel())
        dcos = np.abs((a.direction(i) * b.direction(i)).sum(1)) if len(i) else np.array([np.nan])
        rows.append((inter(a.dens[m.ravel()], b.dens[m.ravel()]), float(np.nanmean(dcos)), 0.5 * np.abs(ca - cb).sum(), ca.argmax() == cb.argmax()))
    r = np.array(rows, float)
    q = lambda x: [round(float(v), 4) for v in np.nanquantile(x, [0.25, 0.5, 0.75])]  # noqa: E731
    out["spheres"] = {"n": len(rows), "where: density intersection (quartiles)": q(r[:, 0]),
                      "where: direction |cos| (quartiles)": q(r[:, 1]),
                      "which: composition total variation (quartiles)": q(r[:, 2]),
                      "which: same dominant tract": round(float(r[:, 3].mean()), 4)}
    return out


def spheres_at(centres, affine, shape):
    vox = np.sqrt((affine[:3, :3] ** 2).sum(0))
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    out = []
    for c in centres:
        d = np.sqrt((((grid - c) * vox) ** 2).sum(1))
        out.append((d <= RADIUS_MM).reshape(shape))
    return out


def tumour_on_dwi():
    """PAT16's tumour mask (ds001226, on the T1) on the DWI grid: the T1 registered rigidly to the corrected mean b0."""
    cache = DATA / "ensemble" / "PAT16" / "tumour_dwi.npz"
    if cache.exists():
        return np.load(cache)["tumour"]
    from tractline import pipeline as P
    from tractline.t1check import tumor_regions
    from _ds001226 import load
    s = load("PAT16")
    with P.exact_float32("mps"):
        corr = P.correct(s, P.Timer(), "mps")
    b0 = np.asarray(corr.dwi, np.float32)[..., s.b0_index].mean(-1)
    t1img = nib.load(s.t1)
    tumour_t1, _ = tumor_regions(t1img, nib.load(s.tumor_mask))
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(a, moving=np.asarray(t1img.dataobj, np.float32), moving_affine=t1img.affine, static=b0, static_affine=s.affine)
        subprocess.run([sys.executable, str(HERE / "_register.py"), str(a), str(b)], check=True)
        z = np.load(b)
        to_dwi, ncc = z["to_static"], float(z["ncc"])                   # T1 world -> DWI world
    M = np.linalg.inv(t1img.affine) @ np.linalg.inv(to_dwi) @ s.affine     # DWI voxel -> T1 voxel
    tumour = affine_transform(tumour_t1.astype(np.float32), M[:3, :3], M[:3, 3], output_shape=s.dwi.shape[:3], order=1) > 0.5
    np.savez(cache, tumour=tumour, t1_to_dwi=to_dwi, ncc=ncc)
    print(f"tumour on the DWI grid: {int(tumour.sum())} voxels, registration NCC {ncc:.3f}", flush=True)
    return tumour


def summarize(rows):
    def mean(path):
        vals = []
        for r in rows:
            v = r
            for k in path:
                v = v[k]
            vals.append(v)
        return np.round(np.mean(np.array(vals, float), 0), 4).tolist()
    keys = [("brain", k) for k in rows[0]["brain"]] + [("spheres", k) for k in rows[0]["spheres"] if k != "n"]
    return {"pairs": len(rows), **{f"{a}: {b}": mean((a, b)) for a, b in keys}}


def main(args):
    from _masivar import scan_files
    rng = np.random.default_rng(0)
    res = {"flip": args.flip, "radius_mm": RADIUS_MM}
    for pair, (subject, (ses, scan), rescans, shells, _) in PAIRS.items():
        d = DATA / "ensemble" / subject / "retest" / pair
        img = nib.load(scan_files(subject, ses, scan, shells)["APP"][0][1])
        aff, shape = img.affine, img.shape[:3]
        A = Run(d / "A.npz", aff, shape)
        dense = np.argwhere((A.dens >= 20).reshape(shape))
        spheres = spheres_at(dense[rng.choice(len(dense), N_SPHERES, replace=False)], aff, shape)
        row = {"rescan": {}}
        for t in rescans:
            row["rescan"][t] = compare(A, Run(d / f"{t}_onA.npz", aff, shape), spheres)
        R = [Run(d / args.flip / f"r{i:03d}.npz", aff, shape) for i in range(1, 9)]
        row[f"replicate vs replicate ({args.flip})"] = summarize([compare(R[i], R[i + 1], spheres) for i in range(0, 8, 2)])
        res[pair] = row
        print(pair, json.dumps(row), flush=True)
    # PAT16: the tumour's margin
    tumour = tumour_on_dwi()
    first = DATA / "ensemble" / "PAT16" / (f"added-{args.flip}" if args.flip != "voxel" else "added")
    img = nib.load(DATA / "ds001226/sub-PAT16/ses-preop/dwi/sub-PAT16_ses-preop_acq-AP_dwi.nii.gz")
    aff, shape = img.affine, img.shape[:3]
    vox = np.sqrt((aff[:3, :3] ** 2).sum(0))
    dist = distance_transform_edt(~tumour, sampling=vox)
    R = [Run(first / f"r{i:03d}.npz", aff, shape) for i in range(1, 9)]
    margin = (dist > 0) & (dist <= 10) & (R[0].dens.reshape(shape) > 0)
    cand = np.argwhere(margin)
    keep = []
    for c in cand[rng.permutation(len(cand))]:
        if all(np.sqrt((((c - k) * vox) ** 2).sum()) >= 4 for k in keep):
            keep.append(c)
    spheres = spheres_at(np.array(keep), aff, shape)
    res["PAT16"] = {"tumour_voxels": int(tumour.sum()), "margin spheres": len(spheres),
                    f"replicate vs replicate ({args.flip}), tumour margin": summarize([compare(R[i], R[i + 1], spheres) for i in range(0, 8, 2)])}
    print("PAT16", json.dumps(res["PAT16"]), flush=True)
    Path(str(OUT).format(args.flip)).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--flip", default="smooth")
    main(ap.parse_args())
