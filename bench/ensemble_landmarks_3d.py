"""A 3D view of cIIIs04's tracts at landmark sites: the ensemble's shells, the real rescans against them, and
full-resolution streamline fans (NOTES 2026-10-09, "Landmark sites on cIIIs04").

    uv run --group dipy --with scikit-image bench/ensemble_landmarks_3d.py OUT.html [--pair cIIIs04-2.1]

Reads ensemble_landmarks.py's DATA/ensemble/<subject>/retest/<pair>/landmarks.npz and results/ensemble_landmarks_smooth.json.
In world millimeters (the scanner's RAS frame), whole brain:
  - a faint brain outline (A's mask) and the mean b0 on a movable axial plane;
  - per shown tract, nested shells of its presence votes over the 17 runs (a run's tract present in a voxel: at least
    2 streamlines): the core (at least 9 of 17 runs), the 80 % envelope (its highest-share voxels holding stated 0.925
    of its mass, the level that held 0.8 of this subject's rescan tracts, fitted on other subjects), the widest (any
    run); and, as a fourth choice, where the tract is the runner-up of the voxel rank field (second by mean count);
  - for a chosen rescan, its voxels of each tract outside the 80 % envelope, as dots;
  - the sites, and at the chosen site its fans: seeds near the site, each drawn from all 17 runs (full streamlines),
    colored by the tract each run gave it, by that run's runner-up, or by its endpoint check (ensemble_endpoints.py's
    rules against FastSurfer's labels: consistent, inconsistent, not scored), with the rescans' versions in white;
  - FastSurfer's anatomy (DATA/fastsurfer/cIIIs04_asegdkt.seg.nrrd, carried onto the DWI by the T1-to-DWI transform):
    surfaces of the structures the focus tracts (CST, AF, UF) run between - sensorimotor cortex, the language cortex
    (inferior frontal gyrus, lateral temporal, supramarginal), orbitofrontal cortex, thalamus, lentiform nucleus,
    brainstem, lateral ventricles;
  - the focus tracts' figures: name kept on each rescan against the runs' statement, where changed names go, 80 %
    outline coverage, endpoint consistency (ensemble_endpoints.json).
Surfaces: marching cubes on the votes smoothed by a Gaussian (sigma 0.8 voxels), at the threshold minus 0.5.
Self-contained HTML (three.js from jsDelivr, the data inline).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_pat16_viewer import NAMES, COLORS  # noqa: E402
from ensemble_pat16_3d import b64, mesh, pack  # noqa: E402
from ensemble_retest import PAIRS  # noqa: E402

RUNS = 17
CTX = lambda *codes: [b + c for b in (1000, 2000) for c in codes]            # noqa: E731  DKT, both hemispheres
ANATOMY = [   # name, labels, color, default on, which focus tract it bears on
    ("Sensorimotor cortex", CTX(24, 22, 17), "#c98b6b", True, "CST: precentral, postcentral, paracentral"),
    ("Language cortex", CTX(18, 20, 30, 15, 31), "#b07aa1", True, "AF: inferior frontal gyrus, superior and middle temporal, supramarginal"),
    ("Orbitofrontal cortex", CTX(12, 14, 19), "#a39a5c", False, "UF: lateral and medial orbitofrontal, pars orbitalis"),
    ("Thalamus", [10, 49], "#5f8fb0", True, "CST passes lateral to it, in the internal capsule"),
    ("Lentiform nucleus", [12, 13, 51, 52], "#6aa17a", True, "putamen and pallidum; the internal capsule lies medial to it"),
    ("Brainstem", [16], "#8a8f96", False, "CST's lower end (the scans cut it off)"),
    ("Lateral ventricles", [4, 43], "#4d6b8a", False, "orientation"),
]


def main(args):
    from tractline.data import DATA
    from tractline.labelers.base import TRACT_NAMES
    d = DATA / "ensemble" / PAIRS[args.pair][0] / "retest" / args.pair
    z = np.load(d / "landmarks.npz")
    res = json.loads((HERE / "results/ensemble_landmarks_smooth.json").read_text())
    aff, votes, shown = z["affine"], z["votes"], [str(s) for s in z["shown"]]
    envkey = next(k for k in res if k.startswith("envelopes"))
    env = res[envkey]
    meshes = [pack("brain", "brain", "#9fb3c4", mesh(z["mask"].astype(np.float32), 0.5, aff, sigma=1.5, step=2))]
    color = {n: COLORS[i] for i, n in enumerate(shown)}
    for i, name in enumerate(shown):
        v = votes[i].astype(np.float32)
        th = {"core": int(np.ceil(RUNS / 2)), "mid": env[name]["80 % envelope votes"], "outer": 1}
        for kind, t in th.items():
            meshes.append(pack(name, kind, color[name], mesh(v, t - 0.5, aff, step=1 if kind == "core" else 2)))
        second = (z["rank2"] == TRACT_NAMES.index(name)).astype(np.float32)
        meshes.append(pack(name, "second", color[name], mesh(second, 0.5, aff, sigma=0.8, step=2)))
    meshes = [m for m in meshes if m]
    # the rescans' voxels outside the 80 % envelope, per tract
    ijk2w = lambda ijk: np.rint((ijk @ aff[:3, :3].T + aff[:3, 3]) * 10).astype(np.int16)   # noqa: E731
    misses = {}
    for j, tag in enumerate(str(t) for t in z["rescans"]):
        misses[tag] = {}
        for i, name in enumerate(shown):
            inside = votes[i] >= env[name]["80 % envelope votes"]
            out = np.argwhere(z["rescan_presence"][j, i] & ~inside)
            misses[tag][name] = {"n": int(len(out)), "p": b64(ijk2w(out))}
    # fans per site
    fans = []
    pts = np.rint(z["fan_pts"] * 10).astype(np.int16)
    off = np.concatenate([[0], np.cumsum(z["fan_len"])])
    for si in range(len(res["sites"])):
        site = {}
        for ki, kind in enumerate(("split", "agree")):
            sel = np.flatnonzero((z["fan_site"] == si) & (z["fan_kind"] == ki))
            site[kind] = {"n": int(len(sel)), "len": z["fan_len"][sel].tolist(), "src": z["fan_src"][sel].tolist(),
                          "tract": z["fan_tract"][sel].tolist(), "up": z["fan_up"][sel].tolist(),
                          "p": b64(np.concatenate([pts[off[s]:off[s + 1]] for s in sel]) if len(sel) else np.zeros((0, 3), np.int16))}
        fans.append(site)
    # FastSurfer's anatomy onto the DWI's world, and the fans' endpoint check
    from ensemble_endpoints import Anatomy, FS, ends_info, consistent
    anat = Anatomy("cIIIs04")
    t1_to_dwi = np.load(FS / "cIIIs04_t1_to_dwi.npz")["t1_to_dwi"]
    anat_world = t1_to_dwi @ anat.affine
    anatomy = []
    for name, labels, col, on, note in ANATOMY:
        m = pack(name, "anatomy", col, mesh(np.isin(anat.lab, labels).astype(np.float32), 0.5, anat_world, sigma=0.7, step=2))
        if m:
            m.update(on=on, note=note)
            anatomy.append(m)
    fibers = [z["fan_pts"][off[i]:off[i + 1]] for i in range(len(z["fan_len"]))]
    ra, rb, sa, sb, enters = ends_info(fibers, anat, np.linalg.inv(t1_to_dwi), aff)
    ok, scored = consistent(z["fan_tract"].astype(int), ra, rb, sa, sb, enters)
    check = np.where(scored, ok.astype(int), 2)                                # 1 consistent, 0 not, 2 not scored
    for si, site in enumerate(fans):
        for ki, kind in enumerate(("split", "agree")):
            site[kind]["ok"] = check[(z["fan_site"] == si) & (z["fan_kind"] == ki)].tolist()
    endpoints = json.loads((HERE / "results/ensemble_endpoints.json").read_text())["cIIIs04"]["17 runs pooled: per bundle"]
    focus = {n: {**res["focus tracts"][n], "outline": env[n], "endpoints consistent": endpoints.get(n, [0, None])[1]}
             for n in res["focus tracts"]}
    b0 = z["meanb0"]
    data = {"meshes": meshes, "anatomy": anatomy, "focus": focus, "fans": fans, "misses": misses, "sites": res["sites"], "brain": res["whole brain"],
            "env": env, "runs": RUNS, "rescans": [str(t) for t in z["rescans"]], "other": len(TRACT_NAMES) - 1,
            "tract_names": list(TRACT_NAMES), "full": {n: NAMES.get(n, n) for n in TRACT_NAMES},
            "shown": [{"name": n, "color": color[n], "full": NAMES.get(n, n)} for n in shown],
            "b0": b64(np.clip(b0 / np.quantile(b0[b0 > 0], 0.995) * 255, 0, 255).astype(np.uint8)), "shape": list(b0.shape),
            "affine": np.asarray(aff).tolist(), "voxel_mm": [round(float(x), 2) for x in np.sqrt((aff[:3, :3] ** 2).sum(0))]}
    html = (HERE / "landmarks_3d.html").read_text().replace("/*DATA*/null", json.dumps(data))
    Path(args.out).write_text(html)
    print(f"{args.out}: {len(html) / 1e6:.2f} MB; meshes {len(meshes)}, triangles {sum(m['nf'] for m in meshes)}; "
          f"fan streamlines {len(z['fan_len'])}, vertices {int(z['fan_len'].sum())}; anatomy {len(anatomy)} surfaces, "
          f"{sum(m['nf'] for m in anatomy)} triangles; fan versions consistent / not / unscored "
          f"{int((check == 1).sum())} / {int((check == 0).sum())} / {int((check == 2).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--pair", default="cIIIs04-2.1", choices=sorted(PAIRS))
    main(ap.parse_args())
