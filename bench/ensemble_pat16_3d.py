"""A 3D view of PAT16's tracts at its tumor over 17 runs: tract shells and streamline fans (NOTES 2026-10-08, "PAT16 at the
tumor").

    uv run --group dipy --with scikit-image bench/ensemble_pat16_3d.py OUT.html [--flip smooth] [--fans 40]

Reads ensemble_pat16_report.py's DATA/ensemble/PAT16/report_<flip>.npz, its JSON, ensemble_coverage.py's results and the
17 run records (ensemble_pilot.py's added-<flip>/r000..r016). In world millimeters (the DWI's RAS frame):
  - the tumor (the dataset's mask on the DWI grid) and a faint brain outline (the corrected mean b0, thresholded);
  - per shown tract, within CROP_MM of the tumor, nested shells of its vote count over the 17 runs (P(v, k) as in the report, winners of at least
    1 streamline): the core, voxels it holds in at least 9 of 17 runs; the level that held 50 % of a healthy rescan's
    tract (stated 0.764); and the widest, every voxel any run gave it (stated 0.995, which held 0.68 on MASiVar);
  - fans: for seeds near the tumor (the first run's path within 10 mm of the mask) that all 17 runs labeled, the 17
    versions of the seed's streamline (the records' 20 points), colored by the tract each run gave it; `fans` seeds
    where the runs disagree on the tract and `fans` where all 17 agree on one named tract;
  - the mean b0 volume, for a movable axial plane.
Surfaces: marching cubes on the vote counts smoothed by a Gaussian (sigma 0.8 voxels), at the threshold minus 0.5.
Self-contained HTML (three.js from jsDelivr, the data inline).
"""
import argparse, base64, json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, binary_fill_holes, map_coordinates

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_pat16_viewer import NAMES, COLORS, SHOWN, threshold, mean_b0  # noqa: E402

RUNS = 17
CROP_MM = 25.0                    # shells are drawn within this distance of the tumor (tapered over 5 mm more)


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def mesh(vol, level, affine, sigma=0.8, step=1):
    """Marching cubes of a smoothed volume at `level`, in world mm: (vertices int16 in 0.1 mm, faces uint32)."""
    from skimage.measure import marching_cubes
    v = gaussian_filter(vol.astype(np.float32), sigma)
    if v.max() <= level:
        return None
    verts, faces, _, _ = marching_cubes(v, level, step_size=step)
    w = verts @ affine[:3, :3].T + affine[:3, 3]
    return np.rint(w * 10).astype(np.int16), faces.astype(np.uint32)


def pack(name, kind, color, m):
    if m is None:
        return None
    v, f = m
    return {"name": name, "kind": kind, "color": color, "nv": len(v), "nf": len(f), "v": b64(v),
            "f": b64(f.astype(np.uint16) if len(v) < 65536 else f), "f16": len(v) < 65536}


def main(args):
    from tractline import ensemble as E
    from tractline.data import DATA
    from tractline.labelers.base import TRACT_NAMES, LUT
    z = np.load(DATA / "ensemble" / "PAT16" / f"report_{args.flip}.npz")
    P, dist, tumor, aff = z["P"].astype(np.float32), z["dist"], z["tumor"], z["affine"]
    rep = json.loads((HERE / f"results/ensemble_pat16_report_{args.flip}.json").read_text())
    use50 = rep["envelope levels"]["0.5"]["use"]
    meshes = [pack("tumor", "tumor", "#d9572f", mesh(tumor, 0.5, aff, sigma=0.7))]
    b0 = mean_b0()
    brain = binary_fill_holes(gaussian_filter(b0, 1.5) > np.quantile(b0[b0 > 0], 0.99) * 0.12)
    meshes.append(pack("brain", "brain", "#9fb3c4", mesh(brain, 0.5, aff, sigma=1.5, step=2)))
    shown = [t for t in SHOWN if t in rep["tracts"]]
    near_w = np.clip((CROP_MM + 5 - dist) / 5, 0, 1)                             # 1 within CROP_MM of the tumor, 0 beyond +5 mm
    shells = {}
    for i, name in enumerate(shown):
        k = TRACT_NAMES.index(name)
        votes = np.rint(P[k] * RUNS) * near_w
        t50 = max(round(threshold(P[k], use50) * RUNS), 1)
        th = {"core": int(np.ceil(RUNS / 2)), "mid": t50, "outer": 1}
        shells[name] = th
        for kind, t in th.items():
            meshes.append(pack(name, kind, COLORS[i], mesh(votes, t - 0.5, aff)))
    meshes = [m for m in meshes if m]
    # fans
    first = DATA / "ensemble" / "PAT16" / (f"added-{args.flip}" if args.flip != "voxel" else "added")
    recs = [E.RunRecord.load(first / f"r{r:03d}.npz", logp=False) for r in range(RUNS)]
    common = recs[0].keys
    for r in recs[1:]:
        common = np.intersect1d(common, r.keys)
    rows = [dict(zip(r.keys.tolist(), range(len(r.keys)))) for r in recs]
    inv = np.linalg.inv(aff)
    p0 = np.stack([recs[0].paths[rows[0][int(k)]] for k in common])               # (n, 20, 3)
    ijk = (p0.reshape(-1, 3) @ inv[:3, :3].T + inv[:3, 3]).T
    dmin = map_coordinates(dist, ijk, order=0, mode="nearest").reshape(len(common), -1).min(1)
    near = common[dmin <= 10]
    tracts = np.stack([[LUT[recs[r].cluster[rows[r][int(k)]]] for r in range(RUNS)] for k in near])   # (m, 17)
    other = len(TRACT_NAMES) - 1
    top = np.array([np.bincount(t, minlength=len(TRACT_NAMES)).max() for t in tracts])
    named_any = (tracts != other).any(1)
    agree = np.flatnonzero((top == RUNS) & (tracts[:, 0] != other))
    disagree = np.flatnonzero((top <= RUNS * 0.6) & named_any)
    rng = np.random.default_rng(0)
    pick = {"disagree": rng.choice(disagree, min(args.fans, len(disagree)), replace=False),
            "agree": rng.choice(agree, min(args.fans, len(agree)), replace=False)}
    fans = {}
    for kind, idx in pick.items():
        paths = np.stack([[recs[r].paths[rows[r][int(near[i])]] for r in range(RUNS)] for i in idx])   # (s, 17, 20, 3)
        fans[kind] = {"n": len(idx), "paths": b64(np.rint(paths * 10).astype(np.int16)), "tracts": tracts[idx].astype(np.uint8).ravel().tolist()}
    palette = {TRACT_NAMES.index(n): COLORS[i] for i, n in enumerate(shown)}
    cov = json.loads((HERE / f"results/ensemble_coverage_{args.flip}.json").read_text())
    curve = cov["recalibration on real rescans (leave one pair out)"]["coverage curve, all rescan scorings (level: coverage)"]
    data = {"meshes": meshes, "fans": fans, "palette": {str(k): v for k, v in palette.items()}, "other": other,
            "tract_names": list(TRACT_NAMES), "full": {n: NAMES.get(n, n) for n in TRACT_NAMES},
            "shown": [{"name": n, "color": COLORS[i], "full": NAMES.get(n, n), "thr": shells[n]} for i, n in enumerate(shown)],
            "b0": b64(np.clip(b0 / np.quantile(b0[b0 > 0], 0.995) * 255, 0, 255).astype(np.uint8)), "shape": list(b0.shape),
            "affine": np.asarray(aff).tolist(), "tumor_z": int(np.argmax(tumor.sum((0, 1)))),
            "levels": {"mid_stated": use50, "widest_held": curve["0.995"], "near_seeds": int(len(near)),
                       "disagree_seeds": int(len(disagree)), "agree_seeds": int(len(agree))}}
    html = (HERE / "pat16_3d.html").read_text().replace("/*DATA*/null", json.dumps(data))
    Path(args.out).write_text(html)
    print(f"{args.out}: {len(html) / 1e6:.2f} MB; meshes {len(meshes)}, triangles {sum(m['nf'] for m in meshes)}; "
          f"seeds near {len(near)}, disagree {len(disagree)}, agree {len(agree)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--flip", default="smooth")
    ap.add_argument("--fans", type=int, default=40)
    main(ap.parse_args())
