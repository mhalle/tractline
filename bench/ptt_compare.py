"""Phase D, level 1 for the tracker as a whole (docs/sota-pipeline-design.md, section 7.2): tractline.ptt against DIPY's
PTT as step 6 ran it (robust_benchmark.py's records), on the ISMRM 2015 phantom and on cIIIs04's reference scan A, from the
same work files, tissue maps, ACT rule and seed points.

    uv run --group dipy --group rankfield bench/ptt_compare.py run --scan phantom --variant compat [--seeds N]
    uv run --group dipy --group rankfield bench/ptt_compare.py run --scan phantom --variant mm --probe-length 6 --max-angle 20
    uv run --group dipy --group rankfield bench/ptt_compare.py subsample --n 40000 --sources dipy ukf compat mm
    uv run --group dipy --group rankfield bench/ptt_compare.py summary          # results/ptt_compare.json

Variants (the FODs and the tracker; everything else as step 6):
  compat   our CSD on DIPY's constraint sphere (DIPY's FODs, to rounding: csd_check.py) and our tracker with the validation
           switch on (_dipy_compat: DIPY 1.12's units): the oracle as it ran, but for the random numbers. Validates the
           implementation: its scores should fall within DIPY's own run-to-run spread.
  mm       the same FODs, our tracker in real units: what the units alone change.
  mm-own   our CSD on our own sphere and real units: the pipeline's default.
The PMF is read on DIPY's default_sphere in all three (as step 6). Seeds: DIPY's random_seeds_from_mask on wm >= 0.5 with
step 6's count and generator (1), made in a DIPY process and cached, so every variant starts where DIPY's run started.
Without a T1 (NOTES 2026-10-29): --stopping mask|fa (pipeline.ptt_stopping; --stopping-fa) and --seeding fa (the
pipeline's: uniform in the tracking mask's FA >= 0.2, step 6's count, generator 1; FA by DTI on the work file's shell)
replace the ACT maps and the WM seeds; the run's name says so.
Each run: RapidParc labels; on the phantom, the 2023 ROI scoring (robust_phantom.score), as step 6's.
Records: DATA/ensemble/benchmark/ptt_ours/<variant>/<subject>/; the summary sets them beside DIPY's step 6 records and, on
cIIIs04, beside DIPY's reruns with other seeds (its own spread).
"""
import argparse, json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("TRACTLINE_MIN_FREE_GB", "5")                    # the runs write ~0.5 GB; robust_benchmark's 20 is for its queue
import robust_benchmark as RB  # noqa: E402
import _benchmark_dipy as BD  # noqa: E402
from tractline import pipeline as P, csd, fields as F, ptt  # noqa: E402
from tractline.labelers.base import OTHER, lengths  # noqa: E402

VARIANTS = ("compat", "mm", "mm-own")
OUT = RB.OUT / "ptt_ours"
DEVICE = "mps"
SEEDS_PY = """
import sys, warnings
import numpy as np
warnings.simplefilter("ignore")
from dipy.data import default_sphere, small_sphere
from dipy.tracking.utils import random_seeds_from_mask
z = np.load(sys.argv[1])
seeds = random_seeds_from_mask(z["wm"] >= 0.5, z["affine"], seeds_count=int(z["n"]), seed_count_per_voxel=False,
                               random_seed=1)
np.savez(sys.argv[2], seeds=seeds, default_sphere=default_sphere.vertices, small_sphere=small_sphere.vertices)
"""


def inputs(scan, n):
    """The work file's arrays, the tissue maps, step 6's seeds and DIPY's spheres (cached under OUT/inputs)."""
    work, winfo = RB.work_file(scan)
    tis = RB.tissue(scan, work)
    w = np.load(work)
    pve = np.load(tis)["pve"].astype(np.float64)
    d = OUT / "inputs"
    d.mkdir(parents=True, exist_ok=True)
    cache = d / f"{RB.subject_of(scan)}_{scan}_seeds{n}.npz"
    if not cache.exists():
        src = d / "seed_request.npz"
        np.savez(src, wm=pve[..., 0], affine=w["affine"], n=n)
        subprocess.run([sys.executable, "-I", "-c", SEEDS_PY, str(src), str(cache)], check=True)
        src.unlink()
    s = np.load(cache)
    return w, winfo, pve, s["seeds"], s["default_sphere"], s["small_sphere"]


def act_maps(pve, data):
    """_benchmark_dipy.py's ACT (step 6, without its opt-in no-data rules): include gm + background + the field of
    view's faces, exclude csf."""
    gm, csf = pve[..., 1], pve[..., 3]
    include = gm.copy()
    include[pve.sum(-1) < 1e-6] = 1.0
    for ax in range(3):
        for idx in (0, -1):
            sl = [slice(None)] * 3
            sl[ax] = idx
            include[tuple(sl)] = 1.0
    return include, csf.copy()


def run_name(variant, probe_length, max_angle):
    """The run's folder name: the variant, and the tracker's settings when not the defaults (1.5 mm, 10 degrees)."""
    default = ptt.Params()
    if probe_length == default.probe_length and max_angle == default.max_angle:
        return variant
    return f"{variant}_probe{probe_length:g}_angle{max_angle:g}"


def run(args):
    scan, variant = args.scan, args.variant
    subject = RB.subject_of(scan)
    n = args.seeds or RB.SEEDS["phantom" if scan == "phantom" else "cIIIs04"]
    name = run_name(variant, args.probe_length, args.max_angle)
    if args.stopping != "act":
        name += f"_{args.stopping}" + (f"{args.stopping_fa:g}" if args.stopping == "fa" else "")
    if args.seeding == "fa":
        name += '_faseeds'
    d = OUT / name / subject
    stem = scan if args.track_seed == 1 else f"{scan}_track{args.track_seed}"   # another tracking seed: run to run
    out = d / ("record.npz" if scan == "phantom" else f"{stem}.npz")
    if out.exists() and not args.pilot:
        raise SystemExit(f"{out} exists; not overwritten")
    RB.guard(RB.OUT)
    d.mkdir(parents=True, exist_ok=True)
    timer, t0 = P.Timer(), time.time()
    w, winfo, pve, seeds, default_sphere, small_sphere = inputs(scan, RB.SEEDS["phantom" if scan == "phantom" else "cIIIs04"])
    seeds = seeds[:n]                                                    # a pilot: step 6's first seeds
    dwi, bval, affine, mask = w["dwi"], w["bval"], w["affine"], w["mask"].astype(bool)
    bvec = BD.flip(affine) @ w["bvec"]                                   # image axes, as step 6 gave DIPY
    order = 8 if int((bval > 50).sum()) >= 45 else 6
    with timer("response"):
        resp = csd.response_ssst(dwi, bval, bvec)
    with timer("csd"):
        model = csd.CSD(bval, bvec, resp, order=order, sphere=None if variant == "mm-own" else small_sphere)
        coeffs, cinfo = model.fit(dwi, mask, device=DEVICE)
    include, exclude = act_maps(pve, dwi)
    if args.stopping != "act" or args.seeding == "fa":
        ev, _ = csd.tensor_fit(dwi[mask].astype(np.float64), bval, bvec)
        fa = np.zeros(mask.shape)
        fa[mask] = csd.fa(ev)
        if args.stopping != "act":
            include, exclude = P.ptt_stopping({"stopping": args.stopping, "stopping_fa": args.stopping_fa}, mask, fa_map=fa)
        if args.seeding == "fa":
            seeds = P.ptt_seeds(mask & (fa >= 0.2), affine, n, 1)
    vox = np.linalg.norm(affine[:3, :3], axis=0)
    with timer("track"):
        pmf = F.PmfField(coeffs, mask, default_sphere, DEVICE)
        act = F.Act(include, exclude, DEVICE)
        tracker = ptt.Tracker(pmf, act, affine, vox, ptt.Params(min_len=20.0, max_len=250.0, probe_length=args.probe_length,
                                                                  max_angle=args.max_angle,
                                                                  keep_open_ends=args.stopping == "mask"), DEVICE,
                              _dipy_compat=variant == "compat", backend=args.backend)
        fibers, tinfo = tracker.track(seeds, seed=args.track_seed)
    if args.pilot:                                                       # a timing and sanity run: nothing written
        _, _, lens = lengths(fibers)
        print(json.dumps({"variant": variant, "scan": scan, "backend": args.backend, "seeds": len(seeds), "tracker": tinfo,
                          "streamlines per seed": round(len(fibers) / len(seeds), 3),
                          "length quartiles": np.quantile(lens, [0.25, 0.5, 0.75]).round(1).tolist() if len(fibers) else None,
                          "seconds": timer.seconds}, indent=1), flush=True)
        return
    labeler = P.default_labeler(DEVICE)
    with timer("label"):
        labels = labeler(fibers, draws=(0,), logp=True)
    P.release_memory(DEVICE)
    _, _, lens = lengths(fibers)
    log = {"variant": variant, "name": name, "work": winfo, "seeds": int(len(seeds)), "tracker": tinfo, "streamlines": len(fibers),
           "labeled": int(labels.keep.sum()), "Other share": round(float((labels.tract == OTHER).mean()), 4),
           "length mm (quartiles), all": [round(float(x), 1) for x in np.quantile(lens, [0.25, 0.5, 0.75])],
           "csd": {"order": order, "response evals": resp.evals.tolist(), "response S0": resp.S0, **cinfo},
           "settings": {**{k: getattr(tracker.p, k) for k in ("step_size", "max_angle", "probe_length", "probe_quality",
                                                               "probe_count", "pmf_threshold", "min_len", "max_len")},
                        "units": "DIPY 1.12's (validation switch)" if variant == "compat" else "real",
                        "constraint sphere": "ours (hemisphere 181)" if variant == "mm-own" else "DIPY's small_sphere",
                        "pmf sphere": "DIPY's default_sphere",
                        "seeds": ("uniform in the mask's FA >= 0.2, generator 1" if args.seeding == "fa"
                                  else "DIPY random_seeds_from_mask, wm >= 0.5, random_seed 1"),
                        "stopping": args.stopping + (f" {args.stopping_fa:g}" if args.stopping == "fa" else ""),
                        "tracking random numbers": ("Philox, counter-based" if args.backend == "metal"
                                                    else "torch generator") + f", seed {args.track_seed}",
                        "backend": args.backend, "device": DEVICE},
           "seconds": timer.seconds}
    rec = RB.record_of(fibers, labels, affine, mask.shape)
    if scan == "phantom":
        import robust_phantom as RP
        np.savez_compressed(d / "labels.npz", keys=np.arange(len(fibers)), keep=labels.keep, tract=labels.tract.astype(np.int16),
                            length_mm=labels.length_mm.astype(np.float32))
        log["trk"] = RP.write_trk(d / "tractogram.trk", fibers)
    tmp = out.with_suffix(".tmp.npz")
    rec.save(tmp, compact=True)
    RB.place(tmp, out)
    log["total_s"] = round(time.time() - t0, 1)
    RB.log_update(d / "runs.json", stem if scan != "phantom" else scan, log)
    print(name, scan, json.dumps({k: log[k] for k in ("streamlines", "labeled", "Other share", "seconds")}), flush=True)
    if scan == "phantom":
        import robust_phantom as RP
        RB.log_update(d / "runs.json", "scoring_seconds", RP.score(d))


def phantom_summary(d):
    import robust_phantom as RP
    log = json.loads((d / "runs.json").read_text())["phantom"]
    lab = np.load(d / "labels.npz")
    vd, pair = RP.verdicts(d)
    sc = d / "score"
    seg = json.loads((sc / "segmentation" / "results.json").read_text())
    return {"run": {k: log.get(k) for k in ("seeds", "streamlines", "labeled", "Other share", "length mm (quartiles), all",
                                            "seconds", "settings")},
            "all streamlines": RP.headline(json.loads((sc / "final" / "results.json").read_text()), vd, np.ones(len(vd), bool))
            | {"IB": seg.get("IB"), "IC": seg.get("IC"), "NC": seg.get("NC")},
            "labeled streamlines": RP.headline(json.loads((sc / "labeled" / "results.json").read_text()), vd, lab["keep"])}


def record_compare(a, b):
    """Two cIIIs04 records on A's grid: the tract mix's total variation, and the per-tract voxel counts' correlation."""
    from tractline import ensemble as E
    A, B = (E.RunRecord.load(p, logp=False) for p in (a, b))
    ma = np.bincount(np.asarray(TRACT_LUT)[A.cluster], minlength=43) / len(A.cluster)
    mb = np.bincount(np.asarray(TRACT_LUT)[B.cluster], minlength=43) / len(B.cluster)
    ca, cb = A.counts.astype(np.float64), B.counts.astype(np.float64)
    corr = [float(np.corrcoef(ca[t], cb[t])[0, 1]) for t in range(42) if ca[t].sum() and cb[t].sum()]
    return {"labeled (a, b)": [int(len(A.cluster)), int(len(B.cluster))], "tract mix total variation": round(float(0.5 * np.abs(ma - mb).sum()), 4),
            "per-tract voxel count correlation: min, median": [round(min(corr), 4), round(float(np.median(corr)), 4)]}


def summary(args):
    from tractline.labelers.base import LUT
    global TRACT_LUT
    TRACT_LUT = LUT
    res = {"phantom": {}, "cIIIs04 A": {}}
    dipy_ph = RB.OUT / "ptt" / "phantom"
    if (dipy_ph / "score" / "labeled" / "results.json").exists():
        res["phantom"]["DIPY (step 6)"] = phantom_summary(dipy_ph)
    for d in sorted(OUT.glob("*/phantom")):
        if d.parent.name != "inputs" and not d.parent.name.startswith("matched") and (d / "score" / "labeled" / "results.json").exists():
            res["phantom"][d.parent.name] = phantom_summary(d)
    for m in sorted(OUT.glob("matched_*")):
        res[f"phantom, count-matched ({m.name})"] = {d.name: phantom_summary(d) for d in sorted(m.iterdir())
                                                     if (d / "score" / "labeled" / "results.json").exists()}
    dA = RB.OUT / "ptt" / RB.SUBJECT / "A.npz"
    reps = sorted((RB.OUT / "ptt" / RB.SUBJECT / "replicates").glob("A_seed*_track*.npz"))
    reps = [p for p in reps if not p.stem.endswith(("_edge", "_nodata"))]
    res["cIIIs04 A"]["DIPY against its reruns (other seeds): the spread"] = {p.stem: record_compare(dA, p) for p in reps}
    for v in VARIANTS:
        p = OUT / v / RB.SUBJECT / "A.npz"
        if p.exists():
            res["cIIIs04 A"][f"DIPY against {v}"] = record_compare(dA, p)
            res["cIIIs04 A"][f"{v}: run"] = {k: json.loads((p.parent / "runs.json").read_text())["A"].get(k)
                                             for k in ("streamlines", "labeled", "Other share", "length mm (quartiles), all", "seconds")}
    dlog = json.loads((RB.OUT / "ptt" / RB.SUBJECT / "runs.json").read_text()).get("A", {})
    res["cIIIs04 A"]["DIPY: run"] = {k: dlog.get(k) for k in ("streamlines", "labeled", "Other share", "length mm (quartiles), all", "seconds")}
    (HERE / "results" / "ptt_compare.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


def source_dir(name):
    """A phantom run's folder: "dipy" (step 6), "ukf" (today's UKF, robust_phantom), or one of ours (OUT/<name>)."""
    import robust_phantom as RP
    if name == "dipy":
        return RB.OUT / "ptt" / "phantom"
    if name == "ukf":
        return RP.OUT / "today"
    return OUT / name / "phantom"


def subsample(args):
    """Count-matched phantom runs: each source's streamlines, n drawn at random (generator 0) with their labels, into
    OUT/matched_<n>/<source>/, scored with the 2023 scoring (as step 6's count controls, robust_benchmark.subsample_control:
    F1 depends on the streamline count)."""
    import nibabel as nib
    import robust_phantom as RP
    if not args.n or not args.sources:
        raise SystemExit("subsample needs --n and --sources")
    n = args.n
    for name in args.sources:
        src = source_dir(name)
        d = OUT / f"matched_{n}" / name
        if (d / "score" / "labeled" / "results.json").exists():
            print(name, "scored, skipped")
            continue
        RB.guard(RB.OUT)
        d.mkdir(parents=True, exist_ok=True)
        fibers = list(nib.streamlines.load(str(src / "tractogram.trk")).streamlines)
        if len(fibers) < n:
            raise SystemExit(f"{name}: {len(fibers)} streamlines, fewer than {n}")
        lab = np.load(src / "labels.npz")
        idx = np.sort(np.random.default_rng(0).choice(len(fibers), n, replace=False))
        kept_rank = np.cumsum(lab["keep"]) - 1
        keep = lab["keep"][idx]
        rows = kept_rank[idx[keep]]
        log = {"source": str(src), "streamlines": int(n), "of": len(fibers), "labeled": int(keep.sum()),
               "Other share": round(float((lab["tract"][rows] == OTHER).mean()), 4)}
        done = (d / "labels.npz").exists() and (d / "runs.json").exists() and \
            "phantom" in json.loads((d / "runs.json").read_text())
        if done and json.loads((d / "runs.json").read_text())["phantom"]["source"] != str(src):
            raise SystemExit(f"{d} holds a subsample of another source; remove it")
        if not done:                                                     # (re)written whole: trk, labels, then the log
            log["trk"] = RP.write_trk(d / "tractogram.trk", [np.asarray(fibers[i], np.float32) for i in idx])
            np.savez_compressed(d / "labels.npz", keys=lab["keys"][idx], keep=keep, tract=lab["tract"][rows],
                                length_mm=lab["length_mm"][rows])
            RB.log_update(d / "runs.json", "phantom", log)
        RB.log_update(d / "runs.json", "scoring_seconds", RP.score(d))
        print(name, json.dumps({k: log[k] for k in ("streamlines", "of", "labeled")}), flush=True)


def rescan(args):
    """cIIIs04's rescan agreement, as step 6 measured it (robust_benchmark.analyze): per method, A against each rescan
    (within, session3, session5) on A's grid - presence Dice of CST, AF and UF and the landmark sites (tracker_measures),
    raw and with each scan's labeled streamlines subsampled to today's UKF count on that scan; the count-matched per-tract
    Dice against UKF (matched_rescan_dice); and, where a run with another tracking seed exists, run to run on A. Methods:
    DIPY's PTT (step 6's records), today's UKF, and every run of ours that has all four scans. results/ptt_rescan.json."""
    from robust_ukf_score import Presence
    from tractline import ensemble as E
    d = RB.DATA / "ensemble" / RB.SUBJECT / "retest" / RB.PAIR
    affine = np.load(d / "landmarks.npz")["affine"]
    A0 = E.RunRecord.load(d / "A.npz", logp=False)
    presence = Presence(affine, A0.shape)
    sites = json.loads((HERE / "results/ensemble_landmarks_smooth.json").read_text())["sites"]
    load = lambda p: E.RunRecord.load(p, logp=False, fibers=True)  # noqa: E731
    ukf = {s: load(d / ("A.npz" if s == "A" else f"{s}_onA.npz")) for s in RB.CIII_SCANS}
    counts = {s: len(r.keys) for s, r in ukf.items()}
    methods = {"DIPY (step 6)": RB.OUT / "ptt" / RB.SUBJECT}
    methods.update({p.parent.name: p.parent for p in sorted(OUT.glob(f"*/{RB.SUBJECT}"))
                    if all((p.parent / RB.SUBJECT / f"{s}.npz").exists() for s in RB.CIII_SCANS)})
    out = {"pair": RB.PAIR, "focus": list(RB.FOCUS), "site radius mm": RB.SITE_MM, "landmark sites": len(sites),
           "ukf (today)": RB.tracker_measures(ukf, presence, sites), "methods": {}}
    for name, md in methods.items():
        base = md if name == "DIPY (step 6)" else md / RB.SUBJECT
        recs = {s: load(base / f"{s}.npz") for s in RB.CIII_SCANS}
        m = {"raw": RB.tracker_measures(recs, presence, sites),
             "count-matched to today's UKF labeled count per scan": RB.tracker_measures(
                 {s: RB.subsample(r, counts[s]) for s, r in recs.items()}, presence, sites),
             "count-matched per tract against UKF": RB.matched_rescan_dice(presence, ukf, recs)}
        reps = sorted(p for p in base.glob("A_track*.npz") if re.fullmatch(r"A_track\d+", p.stem)) \
            if name != "DIPY (step 6)" else \
            [base / "replicates" / "A_seed1_track2.npz"]
        reps = [p for p in reps if p.exists()]
        if reps:
            pa = presence(recs["A"])
            m["run to run on A (the same seeds, another tracking seed)"] = {}
            for p in reps:
                pb = presence(load(p))
                m["run to run on A (the same seeds, another tracking seed)"][p.stem] = {
                    n: round(2 * float((pa[k] & pb[k]).sum()) / max(float(pa[k].sum() + pb[k].sum()), 1), 4)
                    for n in RB.FOCUS for k in [RB.TRACT_NAMES.index(n)] if k in pa and k in pb}
        out["methods"][name] = m
        del recs
    (HERE / "results" / args.rescan_out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({n: {s: v["presence Dice"] for s, v in m["raw"]["rescans"].items()} for n, m in out["methods"].items()},
                     indent=1))


TRACT_LUT = None

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "summary", "subsample", "rescan"))
    ap.add_argument("--scan", choices=("phantom",) + RB.CIII_SCANS)
    ap.add_argument("--track-seed", type=int, default=1, help="the tracking's random numbers (seeds stay step 6's)")
    ap.add_argument("--variant", choices=VARIANTS)
    ap.add_argument("--seeds", type=int, help="the first N of step 6's seeds (a pilot); default: all of them")
    ap.add_argument("--pilot", action="store_true", help="track and report; write nothing")
    ap.add_argument("--backend", choices=("torch", "metal"), default="torch")
    ap.add_argument("--probe-length", type=float, default=1.5, help="mm (the study; default step 6's)")
    ap.add_argument("--max-angle", type=float, default=10.0, help="degrees per step (the study; default step 6's)")
    ap.add_argument("--stopping", choices=("act", "mask", "fa"), default="act", help="run: the stopping (default step 6's)")
    ap.add_argument("--stopping-fa", type=float, default=0.15, help="run, --stopping fa: the FA below which a path ends")
    ap.add_argument("--seeding", choices=("wm", "fa"), default="wm", help="run: step 6's WM seeds, or the mask's FA >= 0.2")
    ap.add_argument("--rescan-out", default="ptt_rescan.json",
                    help="rescan: the results file (another name keeps ptt_rescan.json's methods whose records are gone)")
    ap.add_argument("--n", type=int, help="subsample: the common streamline count")
    ap.add_argument("--sources", nargs="+", help="subsample: dipy, ukf, or our run names")
    args = ap.parse_args()
    {"run": run, "summary": summary, "subsample": subsample, "rescan": rescan}[args.what](args)
