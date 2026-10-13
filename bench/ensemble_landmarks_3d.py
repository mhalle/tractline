"""A 3D view of cIIIs04's tracts at landmark sites: the ensemble's presence volume-rendered from the duckn stores, the
real rescans against it, and full-resolution streamline fans (NOTES 2026-10-09, "Landmark sites on cIIIs04"; 2026-10-13,
"Volume rendering from the stores").

    uv run --group dipy --with scikit-image bench/ensemble_landmarks_3d.py OUTDIR [--pair cIIIs04-2.1]

Writes OUTDIR/index.html and, beside it, two subsets of the duckn stores that the page reads in the browser (entries
copied byte for byte by ensemble_store.subset):
  - cIIIs04_ensemble.view.duckn.zip: each scan's 17-run votes and its own run (ensembles/<scan>/votes, /run), on
    FastSurfer's conformed 1 mm grid cropped to the brain, presence by streamline density (ensemble_store.py);
  - cIIIs04_asegdkt.view.duckn.zip: FastSurfer's ranked anatomy, only its ranks (the top class per voxel is plane 0).
The page unzips them, decodes the sharded zarr chunks (blosc zstd with bitshuffle, or zstd; fzstd from jsDelivr), and
ray-marches the shown tracts and the anatomy groups together in one pass, so surfaces composite in depth order:
  - shells per tract: the core (at least half the runs), the envelope at a stated level (votes from the envelope rule,
    computed in the page as ensemble_pooled.thresholds does), the widest (any run); or a cloud of the runs' share;
  - the ensemble: this scan's 17 runs, or the other three scans' 51 runs pooled (leaving out the scan compared with);
  - a compared scan's own run where it falls outside the envelope, with the share of its tract the envelope holds;
  - the held-out coverage of each stated level (the store's validation block, from ensemble_grid_check.py).
Inline, from ensemble_landmarks.py's DATA/ensemble/<subject>/retest/<pair>/landmarks.npz and
results/ensemble_landmarks_smooth.json (A's DWI world, millimeters, the scanner's RAS frame): the sites, their fans
(each seed drawn from all 17 runs, colored by tract, runner-up or endpoint check against FastSurfer's labels;
ensemble_endpoints.py) with the rescans' versions in white, A's mean b0 on a movable axial plane, and the panels'
figures. The stores sit in the T1 world; the page carries them onto A's DWI world by the store's T1-to-DWI transform.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_pat16_viewer import NAMES, COLORS  # noqa: E402
from ensemble_pat16_3d import b64  # noqa: E402
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


ENSEMBLE_VIEW, ANATOMY_VIEW = "cIIIs04_ensemble.view.duckn.zip", "cIIIs04_asegdkt.view.duckn.zip"


def main(args):
    import zipfile
    from tractline.data import DATA
    from tractline.labelers.base import TRACT_NAMES
    from ensemble_store import subset
    subject = PAIRS[args.pair][0]
    d = DATA / "ensemble" / subject / "retest" / args.pair
    z = np.load(d / "landmarks.npz")
    res = json.loads((HERE / "results/ensemble_landmarks_smooth.json").read_text())
    aff, shown = z["affine"], [str(s) for s in z["shown"]]
    color = {n: COLORS[i] for i, n in enumerate(shown)}
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
    # the fans' endpoint check against FastSurfer's labels
    from ensemble_endpoints import Anatomy, FS, ends_info, consistent
    anat = Anatomy("cIIIs04")
    t1_to_dwi = np.load(FS / "cIIIs04_t1_to_dwi.npz")["t1_to_dwi"]
    fibers = [z["fan_pts"][off[i]:off[i + 1]] for i in range(len(z["fan_len"]))]
    ra, rb, sa, sb, enters = ends_info(fibers, anat, np.linalg.inv(t1_to_dwi), aff)
    ok, scored = consistent(z["fan_tract"].astype(int), ra, rb, sa, sb, enters)
    check = np.where(scored, ok.astype(int), 2)                                # 1 consistent, 0 not, 2 not scored
    for si, site in enumerate(fans):
        for ki, kind in enumerate(("split", "agree")):
            site[kind]["ok"] = check[(z["fan_site"] == si) & (z["fan_kind"] == ki)].tolist()
    endpoints = json.loads((HERE / "results/ensemble_endpoints.json").read_text())["cIIIs04"]["17 runs pooled: per bundle"]
    focus = {n: {**res["focus tracts"][n], "endpoints consistent": endpoints.get(n, [0, None])[1]} for n in res["focus tracts"]}
    # the stores, as subsets the page reads whole
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full = DATA / "ensemble" / subject / f"{subject}_ensemble.duckn.zip"
    with zipfile.ZipFile(full) as zf:
        scans = json.loads(zf.read("ensembles/zarr.json"))["attributes"]["duckn"]["extensions"]["tractline"]["scans"]
    subset(full, out / ENSEMBLE_VIEW, [f"ensembles/{s}/{a}" for s in scans for a in ("votes", "run")],
           "Built for the landmarks viewer (bench/ensemble_landmarks_3d.py); the pooled votes and rank fields are left out.")
    subset(FS / "cIIIs04_asegdkt.duckn.zip", out / ANATOMY_VIEW, ["parts/0/ranks"],
           "Built for the landmarks viewer (bench/ensemble_landmarks_3d.py): the top class per voxel is ranks plane 0.")
    stores = {"ensemble": {"path": ENSEMBLE_VIEW, "bytes": (out / ENSEMBLE_VIEW).stat().st_size, "scans": scans},
              "anatomy": {"path": ANATOMY_VIEW, "bytes": (out / ANATOMY_VIEW).stat().st_size}}
    anatomy = [{"name": n, "labels": lab, "color": c, "on": on, "note": note} for n, lab, c, on, note in ANATOMY]
    b0 = z["meanb0"]
    data = {"stores": stores, "anatomy": anatomy, "focus": focus, "fans": fans, "sites": res["sites"], "brain": res["whole brain"],
            "runs": RUNS, "rescans": [str(t) for t in z["rescans"]], "other": len(TRACT_NAMES) - 1,
            "tract_names": list(TRACT_NAMES), "full": {n: NAMES.get(n, n) for n in TRACT_NAMES},
            "shown": [{"name": n, "color": color[n], "full": NAMES.get(n, n)} for n in shown],
            "b0": b64(np.clip(b0 / np.quantile(b0[b0 > 0], 0.995) * 255, 0, 255).astype(np.uint8)), "shape": list(b0.shape),
            "affine": np.asarray(aff).tolist(), "voxel_mm": [round(float(x), 2) for x in np.sqrt((aff[:3, :3] ** 2).sum(0))]}
    html = (HERE / "landmarks_3d.html").read_text().replace("/*DATA*/null", json.dumps(data))
    (out / "index.html").write_text(html)
    print(f"{out}/index.html: {len(html) / 1e6:.2f} MB; stores {ENSEMBLE_VIEW} {stores['ensemble']['bytes'] / 1e6:.2f} MB, "
          f"{ANATOMY_VIEW} {stores['anatomy']['bytes'] / 1e6:.2f} MB; fan streamlines {len(z['fan_len'])}, vertices "
          f"{int(z['fan_len'].sum())}; fan versions consistent / not / unscored "
          f"{int((check == 1).sum())} / {int((check == 0).sum())} / {int((check == 2).sum())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="a directory: index.html and the store subsets")
    ap.add_argument("--pair", default="cIIIs04-2.1", choices=sorted(PAIRS))
    main(ap.parse_args())
