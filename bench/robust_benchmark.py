"""Today's UKF + RapidParc against other methods: a probabilistic CSD tracker with anatomical stopping, and TractSeg's
bundle masks (docs/robustness-plan.md, step 6).

    uv run --group dipy --group rankfield bench/robust_benchmark.py pilot --scan A --seeds 20000   (the PTT yield; A or phantom)
    uv run --group dipy --group rankfield bench/robust_benchmark.py run --method ptt --scans A [within session3 session5 phantom]
    uv run --group dipy --group rankfield bench/robust_benchmark.py run --method tractseg --scans A [...]
    uv run --group dipy --group rankfield bench/robust_benchmark.py replicate A:2:2 A:1:2 within:1:1:edge ...
                                              (PTT with other RNGs, or with the scan's own face: _benchmark_dipy.py)
    uv run --group dipy --group rankfield bench/robust_benchmark.py controls ukf_ptt_wm_seeds ptt_ukf_seeds ptt_return_all \
                                              ptt_at_ukf_seed_count ukf_at_wm_seed_count   (the phantom's seeding controls)
    uv run --group dipy --group rankfield bench/robust_benchmark.py diagnose                       (figures the NOTES quote)
    uv run --group dipy --group rankfield bench/robust_benchmark.py check                          (TractSeg's masks on A)
    uv run --group dipy --group rankfield bench/robust_benchmark.py analyze

The scans: cIIIs04-2.1's four (ensemble_retest.PAIRS: A, and the within, session 3 and session 5 rescans), each corrected
on its own grid as today (P.correct) and the rescans resampled onto A's (ensemble_retest.onto_reference with the stored
registration, the b-vectors turned), exactly as robust_ukf_score's <tag>_onA; and the ISMRM 2015 phantom (_ismrm2015,
corrected as today). The corrected DWI (the b0s and the shell tracked: 2000 on cIIIs04, 1000 on the phantom), its
b-values and b-vectors and the pipeline's tracking mask go to a work file that bench/_benchmark_dipy.py reads in a
subprocess (DIPY's OpenMP clashes with torch); the work file is deleted when the scan is done.

Methods:
  ptt       DIPY's PTT on a single-shell CSD (the response from the data), ACT stopping from tissue maps
            (_benchmark_dipy.py's docstring has every setting). Tissue maps: cIIIs04 from FastSurfer's asegdkt labels
            (DATA/fastsurfer/cIIIs04_asegdkt.seg.nrrd) carried into A's DWI world by cIIIs04_t1_to_dwi.npz, as partial
            volumes on A's grid (used for all four scans, which are on A's grid); the phantom from its T1.nii.gz
            (three intensity classes inside the tracking mask), never from the ground-truth bundles or the scoring ROIs.
            SEEDS seeds drawn uniformly in white matter (wm >= 0.5), a number chosen from the pilot so that A has about
            as many labeled streamlines as today's UKF (64,378); the phantom's chosen so that it has about as many
            streamlines as today's UKF run there (61,637). The streamlines are labeled by RapidParc (P.default_labeler,
            draw 0, "index" grouping), which was trained on UKF tractograms: its Other share and tract mix are the
            domain-shift check. A run's record is an ensemble.RunRecord (compact) whose keys are the streamlines'
            indices (probabilistic streamlines have no seed-voxel identity: several per voxel, none matched across scans).
  tractseg  TractSeg 2.10 (isolated environment DATA/tools/tractseg-2.10, its weights best_weights_ep220.npz, Zenodo
            3518348) on fODF peaks from DIPY's CSD (the same fit as ptt's), so that no MRtrix is needed: 3 peaks per
            voxel, world (RAS) directions scaled by amplitude (MRtrix's sh2peaks' format), on TractSeg's MNI FA
            template's grid (1.25 mm, MNI's axis signs) by a rigid registration of A's FA (cIIIs04: A's transform for
            all four scans, which are on A's grid) or the phantom's FA to the template. The 72 masks come back to A's grid
            (cIIIs04) or the scorer's 1 mm t1 grid (phantom) by trilinear interpolation of each binary mask, >= 0.5. On the
            CPU. TractSeg was trained on HCP (b = 1000/2000/3000, 270 directions, 1.25 mm, MRtrix's CSD peaks): cIIIs04
            (b = 2000, 56 directions, 2.1 mm) and above all the phantom (b = 1000, 32 directions, 2 mm, simulated) are off
            its domain, and DIPY's peaks are not MRtrix's.

Records (never overwritten: a scan whose record exists is skipped; stops below MIN_FREE_GB free):
  OUT/ptt/cIIIs04/<scan>.npz, runs.json, tissue.npz
  OUT/ptt/phantom/record.npz, labels.npz, tractogram.trk, score/ (robust_phantom.score: the 2023 ROI scoring), runs.json,
                  tissue.npz
  OUT/tractseg/cIIIs04/<scan>.npz (masks on A's grid), <scan>_mni.npz (TractSeg's output), to_mni.npz, runs.json
  OUT/tractseg/phantom/masks.npz (on the t1 grid), masks_mni.npz, to_mni.npz, runs.json

analyze -> results/robust_benchmark.json, regional measures only (probabilistic streamlines cannot be matched by seed):
  cIIIs04, per method (ukf: today's records, ensemble_retest's A and <tag>_onA; ptt: raw and count-matched, each scan's
  labeled streamlines subsampled to today's UKF count on that scan; tractseg):
    presence Dice of CST, AF and UF between A and each rescan (trackers: robust_ukf_score.Presence, the density presence
    of ensemble_grid_check on A's grid; TractSeg: its masks, left and right together);
    the landmark sites (results/ensemble_landmarks_smooth.json): the same main named tract within SITE_MM in A and the
    rescan (trackers: robust_ukf_score.site_main; TractSeg: the mapped tract with the most mask voxels within SITE_MM);
    the focus tracts' volumes per scan; the run times; for ptt the Other share, the tract mix (total variation against
    today's A), RapidParc's mean top probability and the lengths.
  phantom: ptt under the 2023 scoring (robust_phantom.headline, labels_against_truth), beside today's UKF figures from
    results/robust_phantom.json; Dice against the ground-truth bundles' masks (the scorer's traversal masks: every voxel
    a ground-truth streamline crosses) of TractSeg's masks (TS_PHANTOM), and of the labeled presence of UKF and of ptt
    (robust_phantom.STRICT's counterparts, by side) for the bundles that map; the same count-matched to UKF per bundle
    (DRAWS draws), since a tracker's density presence against a traversal mask rewards streamline count.
After the first reading and the review (NOTES, step 6):
  replicate  PTT on cIIIs04 with other seed and tracking RNGs (scan:seed_rng:track_rng), and with the scan's own face as
             a valid end (:edge, _benchmark_dipy.py --edge-from-data; post hoc: chosen after the CST result, where a
             rescan resampled onto A's grid leaves A's lowest slice empty), into OUT/ptt/cIIIs04/replicates/;
  controls   on the phantom, each scored with the 2023 scoring, into OUT/controls/phantom/<name>/:
               ukf_ptt_wm_seeds       today's UKF tracked only from its seeds in PTT's white-matter voxels (select);
               ptt_ukf_seeds          PTT + ACT seeded at UKF's own seed points (one per voxel, FA > 0.1 in the mask);
               ptt_return_all         PTT + ACT keeping every streamline (the invalid ends too), PTT's seeds;
               ptt_at_ukf_seed_count  PTT's run subsampled (seed 0) to ptt_ukf_seeds' count, and
               ukf_at_wm_seed_count   today's UKF run subsampled to ukf_ptt_wm_seeds' count: F1 depends on the count;
  diagnose   figures the NOTES quote, kept in the JSON ("diagnostics");
  analyze    also: count-matched per-tract rescan Dice (UKF against PTT as run and with the own-face rule: per scan and
             tract both subsampled to the smaller count), and PTT's Dice for pairs that do or do not share the RNGs.
What each method reuses across cIIIs04's scans differs: PTT A's tissue maps and seed points; TractSeg A's registration
(and it is deterministic); UKF nothing (its mask and seeds come from each scan).
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, nibabel as nib
from tractline import pipeline as P
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT, TRACT_NAMES, OTHER, lengths
from tractline.prep import prepare

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS, onto_reference                                         # noqa: E402
from ensemble_stages import PAIR, FOCUS, SITE_MM                                          # noqa: E402
import _ismrm2015 as I                                                                    # noqa: E402

DEVICE = "mps"
OUT = DATA / "ensemble" / "benchmark"
WORK = OUT / "_work"
RESULTS = HERE / "results" / "robust_benchmark.json"
MIN_FREE_GB = float(os.environ.get("TRACTLINE_MIN_FREE_GB", 20.0))
SUBJECT = PAIRS[PAIR][0]
CIII_SCANS = ("A", "within", "session3", "session5")
SCANS = CIII_SCANS + ("phantom",)
METHODS = ("ptt", "tractseg")
#: PTT seeds per scan, from the pilots (20,000 seeds; OUT/ptt/<subject>/pilot_*.json): A gave 0.50 labeled streamlines per
#: seed, so 130,000 for about today's UKF 64,378 labeled; the phantom 0.62 streamlines per seed, so 100,000 for about
#: today's UKF 61,637 streamlines there (the scorer scores every streamline)
SEEDS = {"cIIIs04": 130_000, "phantom": 100_000}
DIPY_HELPER = HERE / "_benchmark_dipy.py"
TS_HELPER = HERE / "_tractseg_run.py"
TS_ENV = DATA / "tools" / "tractseg-2.10"
TS_WEIGHTS = DATA / "tools" / "tractseg-weights"
TS_WEIGHTS_FILE = "best_weights_ep220.npz"                       # Zenodo 3518348
TS_TEMPLATE = TS_ENV / "lib" / "python3.11" / "site-packages" / "tractseg" / "resources" / "MNI_FA_template.nii.gz"
FS = DATA / "fastsurfer"
L = np.asarray(LUT)
#: TractSeg's bundles behind each RapidParc (ORG) tract they map to, left and right together; the rest have none
TS_MAP = {"AF": ("AF_left", "AF_right"), "CST": ("CST_left", "CST_right"), "UF": ("UF_left", "UF_right"),
          "ILF": ("ILF_left", "ILF_right"), "IOFF": ("IFO_left", "IFO_right"), "SLF-I": ("SLF_I_left", "SLF_I_right"),
          "SLF-II": ("SLF_II_left", "SLF_II_right"), "SLF-III": ("SLF_III_left", "SLF_III_right"),
          "CB": ("CG_left", "CG_right"), "ICP": ("ICP_left", "ICP_right"), "MCP": ("MCP",),
          **{f"CC{k}": (f"CC_{k}",) for k in range(1, 8)}}
#: the phantom's 2023 bundles -> TractSeg's (BPS: brainstem projections, CST + FPT + POPT; CP has none)
TS_PHANTOM = {"CA": ("CA",), "CC": ("CC",), "MCP": ("MCP",), "Fornix": ("FX_left", "FX_right"),
              **{f"{b}_{s}": tuple(f"{t}_{s}" for t in ts) for s in ("left", "right") for b, ts in
                 (("Cingulum", ("CG",)), ("BPS", ("CST", "FPT", "POPT")), ("ICP", ("ICP",)), ("ILF", ("ILF",)),
                  ("OR", ("OR",)), ("SCP", ("SCP",)), ("SLF", ("SLF_I", "SLF_II", "SLF_III")), ("UF", ("UF",)))}}


def free_gb(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def guard(path):
    if free_gb(path) < MIN_FREE_GB:
        raise SystemExit(f"stopping: {free_gb(path):.1f} GB free, below {MIN_FREE_GB} GB")


def place(tmp, path):
    """tmp -> path, refusing to overwrite."""
    if Path(path).exists():
        Path(tmp).unlink()
        raise SystemExit(f"{path} appeared meanwhile; not overwritten")
    os.replace(tmp, path)


def subject_of(scan):
    return "phantom" if scan == "phantom" else SUBJECT


def log_update(path, key, value):
    log = json.loads(path.read_text()) if path.exists() else {}
    log[key] = value
    path.write_text(json.dumps(log, indent=1))


# ----------------------------------------------------------------------------------------------- the scans

def work_file(scan):
    """The corrected DWI as the methods take it (module docstring), written once per process to WORK; (path, info)."""
    WORK.mkdir(parents=True, exist_ok=True)
    path = WORK / f"{subject_of(scan)}_{scan}.npz"
    if path.exists():
        return path, json.loads(str(np.load(path)["info"]))
    guard(WORK)
    timer, t0 = P.Timer(), time.time()
    if scan == "phantom":
        s, shell = I.load(), I.SHELL
        with P.exact_float32(DEVICE):
            dwi = P.correct(s, timer, DEVICE).dwi
        affine, bval, bvec = s.affine, s.bval, s.bvec
    else:
        from _masivar import load
        subject, (ses, sc), rescans, shells, shell = PAIRS[PAIR]
        A = load(subject, ses, sc, shells)
        if scan == "A":
            with P.exact_float32(DEVICE):
                dwi = P.correct(A, timer, DEVICE).dwi
            bval, bvec = A.bval, A.bvec
        else:
            retest_log = json.loads((DATA / "ensemble" / subject / "retest" / PAIR / "runs.json").read_text())
            ses_b, scan_b = rescans[scan]
            B = load(subject, ses_b, scan_b, shells)
            with P.exact_float32(DEVICE):
                dwi_b = P.correct(B, timer, DEVICE).dwi
            with timer("onto_A"):
                dwi, bvec = onto_reference(dwi_b, B.bvec, B.affine, np.array(retest_log[f"{scan}_scan"]["to_A"]), A.affine,
                                           A.dwi.shape[:3])
            bval = B.bval
            del dwi_b
        affine = A.affine
    P.release_memory(DEVICE)
    with timer("mask"):
        mask = prepare(dwi, affine, bval, bvec, shell=shell, device=DEVICE).mask
    keep = (bval < 50) | (np.abs(bval - shell) < 50)
    info = {"scan": scan, "shell": shell, "volumes": int(keep.sum()), "directions": int((np.abs(bval - shell) < 50).sum()),
            "mask_voxels": int(mask.sum()), "seconds": timer.seconds, "total_s": round(time.time() - t0, 1)}
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, dwi=np.ascontiguousarray(dwi[..., keep], np.float32), bval=bval[keep], bvec=bvec[:, keep], affine=affine,
             mask=mask, info=np.array(json.dumps(info)))
    os.replace(tmp, path)
    print(scan, "work file", json.dumps(info), flush=True)
    return path, info


def dipy(*argv):
    """_benchmark_dipy.py in a subprocess of this environment (no torch there); its last printed line, parsed."""
    t0 = time.time()
    r = subprocess.run([sys.executable, str(DIPY_HELPER), *map(str, argv)], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"_benchmark_dipy.py {argv[0]} failed ({r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-5000:]}")
    out = json.loads(r.stdout.strip().splitlines()[-1])
    out["subprocess_s"] = round(time.time() - t0, 1)
    return out


def tissue(scan, work):
    """The tissue maps on the scan's grid (cIIIs04: A's grid, from FastSurfer; the phantom: from its T1), cached."""
    d = OUT / "ptt" / subject_of(scan)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "tissue.npz"
    if not path.exists():
        if scan == "phantom":
            dipy("tissue_t1", "--work", work, "--t1", I.CHALLENGE / "T1.nii.gz", "--out", path)
        else:
            dipy("tissue_fs", "--work", work, "--labels", FS / f"{SUBJECT}_asegdkt.seg.nrrd", "--t1-to-dwi",
                 FS / f"{SUBJECT}_t1_to_dwi.npz", "--out", path)
    z = np.load(path)
    assert np.allclose(z["affine"], np.load(work)["affine"]), "tissue maps on another grid"
    return path


def record_of(fibers, labels, affine, shape):
    """ensemble.RunRecord.of for streamlines without seed voxels: the keys are the labeled streamlines' indices."""
    kept = np.flatnonzero(labels.keep)
    f = [np.asarray(fibers[i], np.float32) for i in kept]
    logp = labels.logp.cpu().numpy().astype(np.float16)
    lp32 = logp.astype(np.float32)
    cluster = lp32.argmax(1)
    bad = np.flatnonzero(L[cluster] != labels.tract)
    if len(bad):                                                         # float16 ties: as RunRecord.of
        tied = (lp32[bad] == lp32[bad].max(1, keepdims=True)) & (L[None] == labels.tract[bad, None])
        assert tied.any(1).all()
        cluster[bad] = tied.argmax(1)
    return E.RunRecord(kept.astype(np.int64), cluster.astype(np.int16), logp, E.resample(f),
                       E.voxel_counts(f, labels.tract, affine, shape), tuple(int(v) for v in shape), f)


def streamlines(path):
    z = np.load(path)
    o = np.r_[0, np.cumsum(z["lengths"])]
    pts = z["points"]
    return [pts[o[i]:o[i + 1]] for i in range(len(o) - 1)], json.loads(str(z["info"]))


NODATA_ARGS = {"nodata": ("--include-nodata",), "edge": ("--edge-from-data",), None: ()}


def run_ptt(scan, labeler, seeds=None, pilot=False, rng=None, nodata=None, control=None, extra=()):
    """rng: (seed points' RNG, tracking's RNG) for a replicate (replicate()), into OUT/ptt/<subject>/replicates/;
    nodata: "nodata" (_benchmark_dipy.py --include-nodata: changed nothing) or "edge" (--edge-from-data: the scan's own
    face, where a rescan on A's grid leaves A's edge slice empty), the replicate's name suffix. control: a phantom
    seeding control's name (controls()), into OUT/controls/phantom/<control>/, with extra arguments for the helper."""
    subject = subject_of(scan)
    d = OUT / "ptt" / subject
    out = d / ("record.npz" if scan == "phantom" else f"{scan}.npz")
    if control is not None:
        d = OUT / "controls" / "phantom" / control
        out = d / "record.npz"
    if rng is not None:
        d = d / "replicates"
        out = d / f"{scan}_seed{rng[0]}_track{rng[1]}{'_' + nodata if nodata else ''}.npz"
    if out.exists() and not pilot:
        print("ptt", scan, "exists, skipped")
        return
    guard(OUT)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    work, winfo = work_file(scan)
    tis = tissue(scan, work)
    n = seeds or SEEDS["phantom" if scan == "phantom" else "cIIIs04"]
    tracked = WORK / f"{subject}_{scan}_ptt.npz"
    if tracked.exists():
        tracked.unlink()
    info = dipy("csd", "--work", work, "--tissue", tis, "--seeds", n, "--track", tracked,
                *(() if rng is None else ("--seed-rng", rng[0], "--track-rng", rng[1])), *NODATA_ARGS[nodata], *extra)
    fibers, _ = streamlines(tracked)
    tracked.unlink()
    timer = P.Timer()
    with timer("label"):
        labels = labeler(fibers, draws=(0,), logp=True)
    P.release_memory(DEVICE)
    w = np.load(work)
    affine, shape = w["affine"], w["mask"].shape
    _, _, lens = lengths(fibers)
    log = {"work": winfo, "dipy": info, "seeds": info.get("seeds", n), "streamlines": len(fibers), "labeled": int(labels.keep.sum()),
           "Other share": round(float((labels.tract == OTHER).mean()), 4),
           "length mm (quartiles), all": [round(float(x), 1) for x in np.quantile(lens, [0.25, 0.5, 0.75])],
           "seconds": {"correct_and_mask (work file)": winfo["total_s"], "dipy subprocess": info["subprocess_s"],
                       **{f"dipy {k}": v for k, v in info["seconds"].items()}, **timer.seconds}}
    if pilot:
        log["labeled per seed"] = round(log["labeled"] / n, 4)
        print("pilot", json.dumps(log, indent=1), flush=True)
        return log
    rec = record_of(fibers, labels, affine, shape)
    guard(OUT)
    if scan == "phantom":
        import robust_phantom as RP
        for name in ("labels.npz", "tractogram.trk"):
            if (d / name).exists():
                raise SystemExit(f"{d / name} exists; not overwritten")
        np.savez_compressed(d / "labels.npz", keys=np.arange(len(fibers)), keep=labels.keep, tract=labels.tract.astype(np.int16),
                            length_mm=labels.length_mm.astype(np.float32))
        log["trk"] = RP.write_trk(d / "tractogram.trk", fibers)
    tmp = out.with_suffix(".tmp.npz")
    rec.save(tmp, compact=True)
    place(tmp, out)
    log["total_s"] = round(time.time() - t0, 1)
    log_update(d / "runs.json", out.stem if rng is not None else scan, log)
    print("ptt", out.stem, json.dumps({k: log[k] for k in ("streamlines", "labeled", "Other share", "total_s")}), flush=True)
    if scan == "phantom":
        import robust_phantom as RP
        secs = RP.score(d)
        log_update(d / "runs.json", "scoring_seconds", secs)
        print("ptt phantom scored", json.dumps(secs), flush=True)


CONTROLS = ("ukf_ptt_wm_seeds", "ptt_ukf_seeds", "ptt_return_all", "ptt_at_ukf_seed_count", "ukf_at_wm_seed_count")
#: the count controls: a run's streamlines subsampled (seed 0) to another control's count, then scored: (source, the
#: control whose streamline count it takes)
SUBSAMPLED = {"ptt_at_ukf_seed_count": (("ptt", "phantom"), "ptt_ukf_seeds"), "ukf_at_wm_seed_count": (None, "ukf_ptt_wm_seeds")}


def ukf_seeds(work):
    """UKF's seeds on a work file's DWI as P.track makes them (prepare, ukf.seeds at today's threshold): (D, the seed
    points (k, j, i) as ukf.track takes them, each seed's voxel (i, j, k), its point in world RAS mm)."""
    from tractline import ukf as U
    w = np.load(work)
    shell = float(json.loads(str(w["info"]))["shell"])
    t = prepare(w["dwi"], w["affine"], w["bval"], w["bvec"], shell=shell, device=DEVICE)
    D = U.from_arrays(t.dwi, t.header, t.mask)
    kji = U.seeds(D)[0].cpu().numpy()
    ijk_f = kji[:, ::-1]
    vox = np.rint(ijk_f - np.asarray(U.SRAND0_OFFSET, float)[::-1]).astype(np.int64)
    world = ijk_f @ w["affine"][:3, :3].T + w["affine"][:3, 3]
    return D, t, kji, vox, world


def seed_share_in_truth(vox, affine):
    """The share of seed voxels (i, j, k on the DWI grid) whose center falls in the union of the 2023 bundles'
    ground-truth traversal masks (the 21 of config_file_tractometry.json, on the t1 grid)."""
    ref = nib.load(I.REFERENCE)
    union = np.zeros(ref.shape[:3], bool)
    for b in json.loads((I.SCORING / "config_file_tractometry.json").read_text()):
        union |= gt_mask(b).reshape(ref.shape[:3])
    M = np.linalg.inv(ref.affine) @ affine
    t = np.rint(vox @ M[:3, :3].T + M[:3, 3]).astype(int)
    ok = np.all((t >= 0) & (t < union.shape), 1)
    return round(float(union[tuple(t[ok].T)].mean()), 4)


def gt_mask(b):
    """A 2023 bundle's ground-truth traversal mask on the t1 grid (flat bool): step 4's cache, else this step's."""
    import robust_phantom as RP
    for p in (RP.OUT / "gt_masks" / f"{b}.nii.gz", OUT / "gt_masks" / f"{b}.nii.gz"):
        if p.exists():
            return np.asarray(nib.load(p).dataobj).ravel() > 0
    cfg = json.loads((I.SCORING / "config_file_tractometry.json").read_text())
    (OUT / "gt_masks").mkdir(parents=True, exist_ok=True)
    RP.scil("scil_compute_streamlines_density_map.py", I.SCORING / cfg[b]["gt_mask"], OUT / "gt_masks" / f"{b}.nii.gz",
            "--binary", cwd=OUT / "gt_masks")
    return np.asarray(nib.load(OUT / "gt_masks" / f"{b}.nii.gz").dataobj).ravel() > 0


def run_ukf_control(labeler):
    """ukf_ptt_wm_seeds: today's UKF on the phantom (today's settings, Metal, RapidParc as today), tracked only from the
    seeds whose voxel PTT seeds in (wm >= 0.5 of OUT/ptt/phantom/tissue.npz), scored with the 2023 scoring."""
    import robust_phantom as RP
    from tractline import ukf as U
    d = OUT / "controls" / "phantom" / "ukf_ptt_wm_seeds"
    if (d / "record.npz").exists():
        print("ukf_ptt_wm_seeds exists, skipped")
        return
    guard(OUT)
    d.mkdir(parents=True, exist_ok=True)
    t0, timer = time.time(), P.Timer()
    work, winfo = work_file("phantom")
    wm = np.load(tissue("phantom", work))["pve"][..., 0].astype(np.float32) >= 0.5
    D, t, kji, vox, _ = ukf_seeds(work)
    select = np.flatnonzero(wm[tuple(vox.T)])
    with P.exact_float32(DEVICE), timer("ukf"):
        fibers, stats = U.track(D, backend="metal", batch=P.GPU_BATCH, select=select)
    tg = P.Tractogram(fibers, stats, t.mask, t.info)
    with timer("label"):
        labels = labeler(fibers, draws=(0,), logp=True)
    P.release_memory(DEVICE)
    for name in ("labels.npz", "tractogram.trk"):
        if (d / name).exists():
            raise SystemExit(f"{d / name} exists; not overwritten")
    keys = E.seed_keys(np.asarray(stats["seed_voxel"]))
    np.savez_compressed(d / "labels.npz", keys=keys, keep=labels.keep, tract=labels.tract.astype(np.int16),
                        length_mm=labels.length_mm.astype(np.float32))
    log = {"work": winfo, "seeds of today's": [int(len(select)), int(len(vox))], "streamlines": len(fibers),
           "labeled": int(labels.keep.sum()), "Other share": round(float((labels.tract == OTHER).mean()), 4),
           "trk": RP.write_trk(d / "tractogram.trk", fibers), "seconds": timer.seconds}
    w = np.load(work)
    tmp = d / "record.tmp.npz"
    E.RunRecord.of(tg, labels, w["affine"], w["mask"].shape).save(tmp, compact=True)
    place(tmp, d / "record.npz")
    log["total_s"] = round(time.time() - t0, 1)
    log_update(d / "runs.json", "phantom", log)
    print("ukf_ptt_wm_seeds", json.dumps({k: log[k] for k in ("seeds of today's", "streamlines", "labeled")}), flush=True)
    log_update(d / "runs.json", "scoring_seconds", RP.score(d))


def subsample_control(name):
    """A count control (SUBSAMPLED): the source run's streamlines, n of them drawn at random (seed 0), with their labels,
    into OUT/controls/phantom/<name>/ (labels.npz, tractogram.trk), scored with the 2023 scoring."""
    import robust_phantom as RP
    src, other = SUBSAMPLED[name]
    src = RP.OUT / "today" if src is None else OUT.joinpath(*src)
    d = OUT / "controls" / "phantom" / name
    if (d / "labels.npz").exists():
        print(name, "exists, skipped")
        return
    n = json.loads((OUT / "controls" / "phantom" / other / "runs.json").read_text())["phantom"]["streamlines"]
    guard(OUT)
    d.mkdir(parents=True, exist_ok=True)
    fibers = list(nib.streamlines.load(str(src / "tractogram.trk")).streamlines)
    lab = np.load(src / "labels.npz")
    idx = np.sort(np.random.default_rng(0).choice(len(fibers), n, replace=False))
    kept_rank = np.cumsum(lab["keep"]) - 1                                # a kept streamline's row in tract / length_mm
    keep = lab["keep"][idx]
    rows = kept_rank[idx[keep]]
    log = {"source": str(src), "streamlines": int(n), "of": len(fibers), "labeled": int(keep.sum()),
           "Other share": round(float((lab["tract"][rows] == OTHER).mean()), 4)}
    log["trk"] = RP.write_trk(d / "tractogram.trk", [np.asarray(fibers[i], np.float32) for i in idx])
    np.savez_compressed(d / "labels.npz", keys=lab["keys"][idx], keep=keep, tract=lab["tract"][rows], length_mm=lab["length_mm"][rows])
    log_update(d / "runs.json", "phantom", log)
    log_update(d / "runs.json", "scoring_seconds", RP.score(d))
    print(name, json.dumps({k: log[k] for k in ("streamlines", "of", "labeled")}), flush=True)


def controls(args):
    """The phantom's seeding controls (module docstring): which part of PTT's gain is its seeds."""
    for c in args.names:
        if c not in CONTROLS:
            raise SystemExit(f"{c}: one of {CONTROLS}")
    labeler = P.default_labeler(DEVICE)
    for c in args.names:
        if c in SUBSAMPLED:
            subsample_control(c)
        elif c == "ukf_ptt_wm_seeds":
            run_ukf_control(labeler)
        elif c == "ptt_ukf_seeds":
            work, _ = work_file("phantom")
            _, _, _, vox, world = ukf_seeds(work)
            P.release_memory(DEVICE)
            pts = WORK / "phantom_ukf_seeds.npz"
            np.savez(pts, world=world, vox=vox)
            run_ptt("phantom", labeler, control=c, extra=("--seeds-file", pts))
        else:
            run_ptt("phantom", labeler, control=c, extra=("--return-all",))
    for p in WORK.glob("phantom_*.npz") if WORK.exists() else []:
        p.unlink()


def logged_runs():
    """Counts the NOTES quote from the runs' own logs (runs.json under the records), copied into "diagnostics": PTT's
    streamlines and labeled counts on cIIIs04 (the main records and the replicates), and the edge rule's voxels (the
    brain's voxels with no data, and those plus the data's edge next to them, which the rule makes a valid end)."""
    pd = OUT / "ptt" / SUBJECT
    out = {"source": "runs.json under DATA/ensemble/benchmark/ptt/cIIIs04/ (main records) and its replicates/"}
    main = json.loads((pd / "runs.json").read_text())
    out["main records"] = {s: {"streamlines": main[s]["streamlines"], "labeled": main[s]["labeled"]} for s in CIII_SCANS if s in main}
    rp = pd / "replicates" / "runs.json"
    if rp.exists():
        reps = json.loads(rp.read_text())
        out["replicates"] = {}
        for k, v in reps.items():
            row = {"streamlines": v["streamlines"], "labeled": v["labeled"]}
            edge = v["dipy"].get("brain voxels with no data / the data's edge next to them")
            if edge is not None:
                row["edge rule: brain voxels with no data"] = edge[0]
                row["edge rule: voxels made a valid end (those and the data's edge next to them)"] = edge[0] + edge[1]
            out["replicates"][k] = row
    return out


def diagnose(args):
    """Figures the NOTES quote, recomputed and kept in the JSON ("diagnostics"): per cIIIs04 scan on A's grid, the
    brainstem voxels (FastSurfer's label 16 at the voxel centers) of A's lowest slice with no data (every volume 0);
    the rescans' registration to A; where the labeled paths that reach above SPAN_Z[1] and below 20 mm end; the
    phantom's T1-only classes against FA (_benchmark_dipy.py t1_fa_check); the share of each method's seed voxels in
    the union of the ground-truth bundles."""
    from ensemble_endpoints import Anatomy
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    dg = res.setdefault("diagnostics", {})
    anat = Anatomy(SUBJECT)
    to_t1 = np.linalg.inv(np.load(FS / f"{SUBJECT}_t1_to_dwi.npz")["t1_to_dwi"])
    retest_log = json.loads((DATA / "ensemble" / SUBJECT / "retest" / PAIR / "runs.json").read_text())
    empty = {}
    for scan in CIII_SCANS:
        work, _ = work_file(scan)
        w = np.load(work)
        shape, aff = w["mask"].shape, w["affine"]
        ijk = np.argwhere(np.ones(shape[:2] + (1,), bool))                 # slice 0
        world = ijk @ aff[:3, :3].T + aff[:3, 3]
        stem = (anat.raw(world @ to_t1[:3, :3].T + to_t1[:3, 3]) == 16).reshape(shape[:2])
        nodata = ~np.any(w["dwi"][:, :, 0] != 0, axis=-1)
        empty[scan] = {"brainstem voxels in A's lowest slice": int(stem.sum()), "of them with no data": int((stem & nodata).sum()),
                       "voxels of that slice with no data": int(nodata.sum())}
        if scan != "A":
            T = np.array(retest_log[f"{scan}_scan"]["to_A"])
            empty[scan]["registration to A: translation mm"] = round(float(np.linalg.norm(T[:3, 3])), 2)
        w.close()
        work.unlink()
    dg["A's lowest slice on A's grid"] = empty
    ends = {}
    for name, p in (("ptt A", OUT / "ptt" / SUBJECT / "A.npz"), ("ptt within", OUT / "ptt" / SUBJECT / "within.npz"),
                    ("ptt within, edge rule", OUT / "ptt" / SUBJECT / "replicates" / "within_seed1_track1_edge.npz")):
        if not p.exists():
            continue
        r = E.RunRecord.load(p, logp=False, fibers=True)
        zmin = np.array([f[:, 2].min() for f in r.fibers]); zmax = np.array([f[:, 2].max() for f in r.fibers])
        m = (zmax > SPAN_Z[1]) & (zmin < 20)
        ends[name] = {"labeled paths above 60 mm and below 20 mm": int(m.sum()), "of them ending below z = -10 mm": int((m & (zmin < -10)).sum())}
    dg["descending paths"] = ends
    work, _ = work_file("phantom")
    dg["phantom: T1-only classes against FA"] = dipy("t1_fa_check", "--work", work, "--t1", I.CHALLENGE / "T1.nii.gz")
    _, _, _, vox, _ = ukf_seeds(work)
    w = np.load(work)
    wm = np.load(tissue("phantom", work))["pve"][..., 0].astype(np.float32) >= 0.5
    dg["phantom: seed voxels in the ground-truth union"] = {
        "UKF's seeds (one per voxel)": [seed_share_in_truth(vox, w["affine"]), int(len(vox))],
        "PTT's white-matter seed voxels (wm >= 0.5)": [seed_share_in_truth(np.argwhere(wm), w["affine"]), int(wm.sum())],
        "UKF's seeds in PTT's white matter": int(wm[tuple(vox.T)].sum()),
        "the tracking mask": [seed_share_in_truth(np.argwhere(w["mask"]), w["affine"]), int(w["mask"].sum())]}
    work.unlink()
    dg["logged runs"] = logged_runs()
    RESULTS.write_text(json.dumps(res, indent=1))
    print(json.dumps(dg, indent=1))


def ensure_weights():
    """TractSeg looks for pretrained_weights_tract_segmentation_v3.npz in TRACTSEG_WEIGHTS_DIR: a link to the Zenodo file."""
    link = TS_WEIGHTS / "pretrained_weights_tract_segmentation_v3.npz"
    if not (TS_WEIGHTS / TS_WEIGHTS_FILE).exists():
        raise SystemExit(f"{TS_WEIGHTS / TS_WEIGHTS_FILE} missing")
    if not link.exists():
        link.symlink_to(TS_WEIGHTS_FILE)
    return link


def tractseg_env():
    keep = ("PATH", "HOME", "USER", "TMPDIR", "LANG")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    dead = "http://127.0.0.1:9"                                          # any download attempt fails
    env.update(TRACTSEG_WEIGHTS_DIR=str(TS_WEIGHTS), HTTP_PROXY=dead, HTTPS_PROXY=dead, http_proxy=dead, https_proxy=dead,
               ALL_PROXY=dead, KMP_DUPLICATE_LIB_OK="FALSE")
    return env


def unpack(z):
    return np.unpackbits(z["masks"])[:int(np.prod(z["shape"]))].reshape(tuple(z["shape"])).astype(bool)


def masks_onto(masks, mni_affine, mni_to_scan, affine, shape):
    """(72, mx, my, mz) masks on the MNI grid -> (72, X, Y, Z) on a scan grid (affine, shape): each mask trilinearly
    interpolated at the grid's voxel centers carried into the MNI grid, >= 0.5."""
    from scipy.ndimage import map_coordinates
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3).astype(np.float64)
    M = np.linalg.inv(mni_affine) @ np.linalg.inv(mni_to_scan) @ affine
    c = (ijk @ M[:3, :3].T + M[:3, 3]).T
    out = np.zeros((len(masks), *shape), bool)
    for k, m in enumerate(masks):
        if m.any():
            out[k] = (map_coordinates(m.astype(np.float32), c, order=1, mode="constant", cval=0.0) >= 0.5).reshape(shape)
    return out


def run_tractseg(scan):
    subject = subject_of(scan)
    d = OUT / "tractseg" / subject
    out = d / ("masks.npz" if scan == "phantom" else f"{scan}.npz")
    if out.exists():
        print("tractseg", scan, "exists, skipped")
        return
    guard(OUT)
    d.mkdir(parents=True, exist_ok=True)
    to_mni = d / "to_mni.npz"
    if scan not in ("A", "phantom") and not to_mni.exists():
        raise SystemExit(f"{to_mni} missing: run A first (the rescans, on A's grid, use A's registration)")
    ensure_weights()
    t0 = time.time()
    work, winfo = work_file(scan)
    peaks = WORK / f"{subject}_{scan}_peaks.npz"
    if peaks.exists():
        peaks.unlink()
    info = dipy("csd", "--work", work, "--peaks-mni", peaks, "--template", TS_TEMPLATE, "--to-mni", to_mni)
    mni = d / ("masks_mni.npz" if scan == "phantom" else f"{scan}_mni.npz")
    if mni.exists():
        raise SystemExit(f"{mni} exists; not overwritten")
    t1 = time.time()
    r = subprocess.run([str(TS_ENV / "bin" / "python"), "-I", str(TS_HELPER), str(peaks), str(mni)], env=tractseg_env(),
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"TractSeg failed ({r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-5000:]}")
    ts_s = round(time.time() - t1, 1)
    zp = np.load(peaks)
    mni_affine, mni_to_scan = zp["mni_affine"], zp["mni_to_scan"]
    peaks.unlink()
    zm = np.load(mni)
    names = [str(x) for x in zm["names"]]
    if scan == "phantom":
        ref = nib.load(I.REFERENCE)
        affine, shape = ref.affine, ref.shape[:3]
    else:
        w = np.load(work)
        affine, shape = w["affine"], w["mask"].shape
    t2 = time.time()
    masks = masks_onto(unpack(zm), mni_affine, mni_to_scan, affine, shape)
    guard(OUT)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, masks=np.packbits(masks.ravel()), shape=np.array(masks.shape), names=np.array(names),
                        affine=affine)
    place(tmp, out)
    vox_ml = float(abs(np.linalg.det(affine[:3, :3]))) / 1000
    log = {"work": winfo, "dipy": info, "tractseg": json.loads(str(zm["info"])), "tractseg subprocess_s": ts_s,
           "volume mL": {n: round(float(m.sum()) * vox_ml, 1) for n, m in zip(names, masks)},
           "seconds": {"correct_and_mask (work file)": winfo["total_s"], "dipy subprocess": info["subprocess_s"],
                       "tractseg subprocess": ts_s, "masks onto the scan grid": round(time.time() - t2, 1)},
           "total_s": round(time.time() - t0, 1)}
    log_update(d / "runs.json", scan, log)
    print("tractseg", scan, json.dumps({k: log[k] for k in ("tractseg subprocess_s", "total_s")}), flush=True)


def run(args):
    for m in args.method:
        if m not in METHODS:
            raise SystemExit(f"--method one of {METHODS}")
    for s in args.scans:
        if s not in SCANS:
            raise SystemExit(f"--scans among {SCANS}")
    labeler = P.default_labeler(DEVICE) if "ptt" in args.method else None
    for scan in args.scans:
        for m in args.method:
            run_ptt(scan, labeler) if m == "ptt" else run_tractseg(scan)
        for p in WORK.glob(f"{subject_of(scan)}_{scan}*.npz") if WORK.exists() else []:
            p.unlink()                                                     # the large intermediates
    if WORK.exists() and not any(WORK.iterdir()):
        WORK.rmdir()


def replicate(args):
    """PTT run again on cIIIs04 scans with other RNGs (seed points, tracking draws): its run-to-run spread on one scan,
    against the spread between scans (OUT/ptt/cIIIs04/replicates/<scan>_seed<s>_track<t>.npz)."""
    labeler = P.default_labeler(DEVICE)
    for spec in args.runs:                                               # scan:seed_rng:track_rng[:nodata|:edge]
        scan, a, b, *flag = spec.split(":")
        if scan not in CIII_SCANS or flag not in ([], ["nodata"], ["edge"]):
            raise SystemExit(f"{spec}: scan:seed_rng:track_rng[:nodata|:edge], a cIIIs04 scan")
        run_ptt(scan, labeler, rng=(int(a), int(b)), nodata=flag[0] if flag else None)
    for p in WORK.glob(f"{SUBJECT}_*.npz") if WORK.exists() else []:
        p.unlink()


def pilot(args):
    """PTT on a scan (A or the phantom) with --seeds seeds, labeled, nothing kept but the yield and the tissue maps (to
    choose SEEDS); the log in OUT/ptt/<subject>/pilot_<scan>_<seeds>.json."""
    log = run_ptt(args.scan, P.default_labeler(DEVICE), seeds=args.seeds, pilot=True)
    (OUT / "ptt" / subject_of(args.scan) / f"pilot_{args.scan}_{args.seeds}.json").write_text(json.dumps(log, indent=1))
    if not args.keep_work:
        for p in WORK.glob(f"{subject_of(args.scan)}_{args.scan}*.npz"):
            p.unlink()


# ----------------------------------------------------------------------------------------------- checks and analysis

def load_ts(path):
    z = np.load(path)
    return dict(zip([str(x) for x in z["names"]], unpack(z))), z["affine"]


def check(args):
    """TractSeg's CST, AF and UF on A against FastSurfer's anatomy: where each mask's lowest and highest 5 % of voxels
    (by world z) and its most anterior and posterior 5 % (by world y) fall in the asegdkt labels (the nearest cerebral
    gray label within NEAR_MM, "outside" when none), the z and y ranges, and the side (mean x against the midsagittal
    x). The UF's frontal end is anterior, not superior: its highest 5 % lie in white matter."""
    from ensemble_endpoints import Anatomy
    masks, affine = load_ts(OUT / "tractseg" / SUBJECT / "A.npz")
    anat = Anatomy(SUBJECT)
    to_t1 = np.linalg.inv(np.load(FS / f"{SUBJECT}_t1_to_dwi.npz")["t1_to_dwi"])
    from ensemble_endpoints import CTX, ASEG, BRAINSTEM
    names = {BRAINSTEM: "brainstem", 2: "cerebral WM L", 41: "cerebral WM R", 7: "cerebellar WM L", 46: "cerebellar WM R",
             0: "outside", 4: "lateral ventricle L", 43: "lateral ventricle R", 8: "cerebellar cortex L", 47: "cerebellar cortex R"}
    names.update({v + base: f"ctx {k} {'L' if base == 1000 else 'R'}" for k, v in CTX.items() for base in (1000, 2000)})
    names.update({x: f"{k} {'LR'[i]}" for k, p in ASEG.items() for i, x in enumerate(p)})
    res = {}
    for b in ("CST_left", "CST_right", "AF_left", "AF_right", "UF_left", "UF_right"):
        ijk = np.argwhere(masks[b])
        if not len(ijk):
            res[b] = {"voxels": 0}
            continue
        world = ijk @ affine[:3, :3].T + affine[:3, 3]
        t1 = world @ to_t1[:3, :3].T + to_t1[:3, 3]
        lab = anat.raw(t1)
        near = anat.nearest(t1)
        z, y = world[:, 2], world[:, 1]
        lo, hi = z <= np.quantile(z, 0.05), z >= np.quantile(z, 0.95)
        front, back = y >= np.quantile(y, 0.95), y <= np.quantile(y, 0.05)
        top = lambda m, labs: {names.get(int(k), str(int(k))): round(float(c / m.sum()), 3)
                               for k, c in zip(*np.unique(labs[m], return_counts=True)) if c / m.sum() >= 0.05}
        res[b] = {"voxels": int(len(ijk)), "z mm range": [round(float(z.min()), 1), round(float(z.max()), 1)],
                  "side (mean x - midline mm, RAS: < 0 left)": round(float(world[:, 0].mean() - anat.mid_x), 1),
                  "lowest 5 %: labels": top(lo, lab), "highest 5 %: labels": top(hi, lab),
                  "highest 5 %: nearest gray label": top(hi, near),
                  "y mm range": [round(float(y.min()), 1), round(float(y.max()), 1)],
                  "most anterior 5 %: nearest gray label": top(front, near),
                  "most posterior 5 %: nearest gray label": top(back, near)}
    w_affine = affine
    shape = masks["CST_left"].shape
    corners = np.array([[0, 0, 0], [0, 0, shape[2] - 1]]) @ w_affine[:3, :3].T + w_affine[:3, 3]
    res["A's grid: z of the lowest and highest slices' corner (mm)"] = [round(float(c[2]), 1) for c in corners]
    print(json.dumps(res, indent=1))
    res_all = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res_all["tractseg anatomy check, A"] = res
    RESULTS.write_text(json.dumps(res_all, indent=1))


def subsample(rec, n, seed=0):
    """A record's labeled streamlines subsampled to n (all when it has fewer): the count-matched comparison."""
    if len(rec.keys) <= n:
        return rec
    idx = np.sort(np.random.default_rng(seed).choice(len(rec.keys), n, replace=False))
    return E.RunRecord(rec.keys[idx], rec.cluster[idx], None if rec.logp is None else rec.logp[idx], rec.paths[idx], rec.counts,
                       rec.shape, [rec.fibers[i] for i in idx])


def tracker_measures(recs, presence, sites):
    """Per scan's record (A first): focus volumes; per rescan: presence Dice and the landmark sites against A."""
    from robust_ukf_score import site_main
    focus = [TRACT_NAMES.index(n) for n in FOCUS]
    pres = {s: presence(r) for s, r in recs.items()}
    vol = lambda p, k: round(float(p[k].sum()) * presence.vox_ml, 1) if k in p else 0.0
    site = {s: site_main(r, sites) for s, r in recs.items()}
    out = {"streamlines labeled": {s: len(r.keys) for s, r in recs.items()},
           "focus volume mL": {s: {n: vol(p, k) for n, k in zip(FOCUS, focus)} for s, p in pres.items()}, "rescans": {}}
    for s in recs:
        if s == "A":
            continue
        dice = {}
        for n, k in zip(FOCUS, focus):
            a, b = pres["A"].get(k), pres[s].get(k)
            dice[n] = None if a is None and b is None else (0.0 if a is None or b is None else
                                                           round(2 * float((a & b).sum()) / max(float(a.sum() + b.sum()), 1), 4))
        out["rescans"][s] = {"presence Dice": dice,
                             f"landmark sites: the same main named tract within {SITE_MM:g} mm": int(sum(x == y for x, y in zip(site["A"], site[s]))),
                             "landmark sites": len(sites)}
    out["landmark sites: main named tract"] = {s: [TRACT_NAMES[t] if t >= 0 else None for t in v] for s, v in site.items()}
    return out


SPAN_Z = (0.0, 60.0)        # ptt_spread: a labeled streamline reaching below the first z and above the second (A's world mm)


def spanning(rec):
    """Labeled streamlines that reach below SPAN_Z[0] and above SPAN_Z[1] (from the brainstem's level to the upper
    hemisphere), and how many of them are labeled CST."""
    lo = np.array([f[:, 2].min() for f in rec.fibers]) < SPAN_Z[0]
    hi = np.array([f[:, 2].max() for f in rec.fibers]) > SPAN_Z[1]
    m = lo & hi
    return {"spanning": int(m.sum()), "spanning, labeled CST": int((L[rec.cluster][m] == TRACT_NAMES.index("CST")).sum())}


def ptt_spread(presence, ukf_recs):
    """PTT's run-to-run spread on one scan (replicates with other RNGs) against its spread between scans: per record, the
    labeled count, the focus volumes, the CST count and the brainstem-to-hemisphere spanning count; presence Dice of each
    replicate against the scan's main record (seed and tracking RNG 1); the same spanning counts for today's UKF."""
    pd = OUT / "ptt" / SUBJECT
    reps = sorted((pd / "replicates").glob("*.npz")) if (pd / "replicates").exists() else []
    if not reps:
        return None
    focus = [TRACT_NAMES.index(n) for n in FOCUS]
    vol = lambda p, k: round(float(p[k].sum()) * presence.vox_ml, 1) if k in p else 0.0
    dice = lambda a, b, k: (0.0 if (k in a) != (k in b) else None if k not in a else
                            round(2 * float((a[k] & b[k]).sum()) / max(float(a[k].sum() + b[k].sum()), 1), 4))

    def row(rec, p):
        t = L[rec.cluster]
        return {"labeled": len(rec.keys), "CST streamlines": int((t == TRACT_NAMES.index("CST")).sum()),
                "focus volume mL": {n: vol(p, k) for n, k in zip(FOCUS, focus)}, **spanning(rec)}
    out = {"spanning": f"labeled streamlines reaching below z = {SPAN_Z[0]:g} mm and above z = {SPAN_Z[1]:g} mm (A's world)",
           "main records (seed and tracking RNG 1)": {}, "replicates": {}, "ukf (today)": {}}
    main = {}
    for s in CIII_SCANS:
        if (pd / f"{s}.npz").exists():
            r = E.RunRecord.load(pd / f"{s}.npz", logp=False, fibers=True)
            main[s] = presence(r)
            out["main records (seed and tracking RNG 1)"][s] = row(r, main[s])
    for s, r in ukf_recs.items():
        out["ukf (today)"][s] = spanning(r) | {"CST streamlines": int((L[r.cluster] == TRACT_NAMES.index("CST")).sum())}
    pres_nd = {}
    for path in reps:
        scan = path.stem.split("_seed")[0]
        r = E.RunRecord.load(path, logp=False, fibers=True)
        p = presence(r)
        out["replicates"][path.stem] = row(r, p) | {
            f"presence Dice against {scan} (RNG 1)": {n: dice(main[scan], p, k) for n, k in zip(FOCUS, focus)},
            **({"presence Dice against A (RNG 1)": {n: dice(main["A"], p, k) for n, k in zip(FOCUS, focus)}} if scan != "A" else {})}
        for rule in ("nodata", "edge"):
            if path.stem.endswith(f"_seed1_track1_{rule}"):
                pres_nd[rule, scan] = (path.stem, p)
    for (rule, scan), (stem, p) in pres_nd.items():                     # the same rule on both sides
        if scan != "A" and (rule, "A") in pres_nd:
            a_stem, a_p = pres_nd[rule, "A"]
            out["replicates"][stem][f"presence Dice against {a_stem}"] = {n: dice(a_p, p, k) for n, k in zip(FOCUS, focus)}
    return out


DRAWS = 5                   # count-matched comparisons: subsampling draws


def tract_subset(rec, k, n, seed):
    """A record's streamlines of tract k, n of them drawn without replacement (all when it has n or fewer)."""
    idx = np.flatnonzero(L[rec.cluster] == k)
    if len(idx) > n:
        idx = np.sort(np.random.default_rng(seed).choice(idx, n, replace=False))
    return E.RunRecord(rec.keys[idx], rec.cluster[idx], None, rec.paths[idx], rec.counts, rec.shape, [rec.fibers[i] for i in idx])


def matched_rescan_dice(presence, ukf, ptt):
    """Per focus tract, A against each rescan with both methods' tract streamlines subsampled, per scan, to the smaller
    of the two methods' counts of that tract on that scan (DRAWS draws): {"ukf"|"ptt": {scan: {tract: [mean, min,
    max]}}, "counts": {scan: {tract: n}}}."""
    out = {"ukf": {}, "ptt": {}, "counts": {}}
    for s in ptt:
        out["counts"][s] = {}
    for name in FOCUS:
        k = TRACT_NAMES.index(name)
        n = {s: int(min((L[ukf[s].cluster] == k).sum(), (L[ptt[s].cluster] == k).sum())) for s in ptt}
        for s in ptt:
            out["counts"][s][name] = n[s]
        for who, recs in (("ukf", ukf), ("ptt", ptt)):
            for s in recs:
                if s == "A":
                    continue
                v = []
                for dr in range(DRAWS):
                    a, b = presence(tract_subset(recs["A"], k, n["A"], dr)).get(k), presence(tract_subset(recs[s], k, n[s], 100 + dr)).get(k)
                    v.append(2 * float((a & b).sum()) / max(float(a.sum() + b.sum()), 1) if a is not None and b is not None else 0.0)
                out[who].setdefault(s, {})[name] = [round(float(np.mean(v)), 4), round(float(min(v)), 4), round(float(max(v)), 4)]
    return out


def rng_pairs(presence):
    """PTT's presence Dice (focus tracts) for pairs that share or do not share the seed and tracking RNGs: run-to-run on
    A, and A against the rescans (the edge rule), so that "run-to-run" and "rescan" compare like with like."""
    pd = OUT / "ptt" / SUBJECT
    rp = pd / "replicates"
    path = lambda nm: pd / f"{nm}.npz" if "_seed" not in nm else rp / f"{nm}.npz"
    pairs = {
        "run to run on A, other seed and tracking RNGs": [("A", "A_seed2_track2"), ("A", "A_seed3_track3"), ("A_seed2_track2", "A_seed3_track3")],
        "run to run on A, the same seed points, another tracking RNG": [("A", "A_seed1_track2")],
        "rescan (edge rule) against A, the same RNGs": [("A", "within_seed1_track1_edge"), ("A", "session3"), ("A", "session5_seed1_track1_edge"),
                                                        ("A_seed2_track2", "within_seed2_track2_edge"),
                                                        ("A_seed2_track2", "session3_seed2_track2_edge"),
                                                        ("A_seed2_track2", "session5_seed2_track2_edge")],
        "rescan (edge rule) against A, other RNGs": [("A", "within_seed2_track2_edge"), ("A", "session3_seed2_track2_edge"),
                                                     ("A", "session5_seed2_track2_edge"),
                                                     ("A_seed2_track2", "within_seed1_track1_edge"), ("A_seed2_track2", "session3"),
                                                     ("A_seed2_track2", "session5_seed1_track1_edge")],
        "rescan (as run) against A, other RNGs": [("A", "within_seed2_track2"), ("A", "within_seed1_track2")]}
    focus = [TRACT_NAMES.index(n) for n in FOCUS]
    cache = {}

    def pres(nm):
        if nm not in cache:
            cache[nm] = presence(E.RunRecord.load(path(nm), logp=False, fibers=True))
        return cache[nm]
    out = {}
    for label, ps in pairs.items():
        ps = [(a, b) for a, b in ps if path(a).exists() and path(b).exists()]
        if not ps:
            continue
        rows = {f"{a} : {b}": {n: round(2 * float((pres(a)[k] & pres(b)[k]).sum()) / max(float(pres(a)[k].sum() + pres(b)[k].sum()), 1), 4)
                               for n, k in zip(FOCUS, focus)} for a, b in ps}
        out[label] = {"pairs": rows, "range": {n: [min(r[n] for r in rows.values()), max(r[n] for r in rows.values())] for n in FOCUS}}
    return out


def tractseg_measures(masks, affine, sites):
    vox_ml = float(abs(np.linalg.det(affine[:3, :3]))) / 1000
    shape = next(iter(masks["A"].values())).shape
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1).reshape(-1, 3)
    world = ijk @ affine[:3, :3].T + affine[:3, 3]
    near = [np.linalg.norm(world - np.asarray(s["mm"]), axis=1) <= SITE_MM for s in sites]
    tract = lambda m, n: np.any([m[b] for b in TS_MAP[n]], 0)
    mapped = list(TS_MAP)

    def main_tract(m):
        out = []
        for nr in near:
            c = [int(tract(m, n).ravel()[nr].sum()) for n in mapped]
            out.append(mapped[int(np.argmax(c))] if max(c) > 0 else None)
        return out
    site = {s: main_tract(m) for s, m in masks.items()}
    out = {"focus volume mL": {s: {n: round(float(tract(m, n).sum()) * vox_ml, 1) for n in FOCUS} for s, m in masks.items()},
           "rescans": {}, "landmark sites: main mapped tract": site, "mapped tracts (RapidParc <- TractSeg)": TS_MAP}
    for s, m in masks.items():
        if s == "A":
            continue
        dice = {}
        for n in FOCUS:
            a, b = tract(masks["A"], n), tract(m, n)
            dice[n] = round(2 * float((a & b).sum()) / max(float(a.sum() + b.sum()), 1), 4)
        out["rescans"][s] = {"presence Dice": dice,
                             f"landmark sites: the same main mapped tract within {SITE_MM:g} mm":
                                 int(sum(x == y for x, y in zip(site["A"], site[s]))),
                             "landmark sites": len(sites)}
    return out


PHANTOM_RUNS = (("ukf (today)", None), ("ptt", ("ptt", "phantom")), *((c, ("controls", "phantom", c)) for c in CONTROLS))


def phantom_dir(d):
    import robust_phantom as RP
    return RP.OUT / "today" if d is None else OUT.joinpath(*d)


def strict_streamlines(d, x_mid):
    """A phantom run's labeled streamlines of each STRICT bundle's counterparts, on its side: {bundle: [fibers]}."""
    import robust_phantom as RP
    fibers = list(nib.streamlines.load(str(d / "tractogram.trk")).streamlines)
    lab = np.load(d / "labels.npz")
    lt = np.full(len(fibers), "", dtype=object)
    lt[np.flatnonzero(lab["keep"])] = np.array(TRACT_NAMES)[lab["tract"]]
    left = np.array([f[:, 0].mean() for f in fibers]) < x_mid
    out = {}
    for b, ts in RP.STRICT.items():
        side = np.ones(len(fibers), bool) if b in ("CC", "MCP") else left == b.endswith("_left")
        out[b] = [fibers[i] for i in np.flatnonzero(np.isin(lt, ts) & side)]
    return out


def phantom_dice(res):
    """Dice against the ground-truth bundles' traversal masks on the t1 grid: TractSeg's masks (TS_PHANTOM; binary
    masks, the kind of object the target is), and the labeled density presence (STRICT counterparts, by side) of
    today's UKF run, of ptt's and of the seeding controls (density presence: another kind of object, which rewards
    streamline count). Count-matched: each run's and UKF's streamlines of a bundle subsampled to the smaller of the two
    counts (DRAWS draws), so that the count does not decide."""
    import robust_phantom as RP
    from ensemble_grid_check import Grid
    ref = nib.load(I.REFERENCE)
    grid = Grid("t1", ref.affine, ref.shape[:3])
    dice = lambda a, b: round(2 * float((a & b).sum()) / max(float(a.sum() + b.sum()), 1), 4)
    out = {"target": "the ground-truth bundle's traversal mask (scil_compute_streamlines_density_map --binary), t1 grid 1 mm"}
    s = I.load()
    x_mid = float((s.affine @ [(s.dwi.shape[0] - 1) / 2, 0, 0, 1])[0])
    sl = {who: strict_streamlines(phantom_dir(d), x_mid) for who, d in PHANTOM_RUNS
          if (phantom_dir(d) / "tractogram.trk").exists() and (phantom_dir(d) / "labels.npz").exists()}
    for who, by in sl.items():
        r = {b: {"labeled streamlines": len(f), "Dice": dice(RP.density_presence(grid, f), gt_mask(b))} for b, f in by.items()}
        r["mean Dice"] = round(float(np.mean([v["Dice"] for v in r.values()])), 4)
        out[who + ": labeled presence (strict counterparts)"] = r
    cm = {"rule": f"per bundle, the run and UKF subsampled to the smaller count; {DRAWS} draws, mean"}
    U = sl.get("ukf (today)")
    for who, by in sl.items():
        if who == "ukf (today)" or U is None:
            continue
        r, du, dm = {}, [], []
        for b in RP.STRICT:
            n = min(len(U[b]), len(by[b]))
            g = gt_mask(b)
            pick = lambda f, seed: [f[i] for i in np.random.default_rng(seed).choice(len(f), n, replace=False)] if n else []
            u = np.mean([dice(RP.density_presence(grid, pick(U[b], dr)), g) for dr in range(DRAWS)])
            m = np.mean([dice(RP.density_presence(grid, pick(by[b], 100 + dr)), g) for dr in range(DRAWS)])
            r[b] = {"n": n, "ukf": round(float(u), 4), who: round(float(m), 4)}
            du.append(u); dm.append(m)
        r["mean"] = {"ukf": round(float(np.mean(du)), 4), who: round(float(np.mean(dm)), 4)}
        cm[f"{who} against ukf"] = r
    out["count-matched labeled presence"] = cm
    p = OUT / "tractseg" / "phantom" / "masks.npz"
    if p.exists():
        masks, _ = load_ts(p)
        r = {}
        for b, ts in TS_PHANTOM.items():
            m = np.any([masks[t] for t in ts], 0).ravel()
            r[b] = {"tractseg bundles": list(ts), "Dice": dice(m, gt_mask(b))}
        r["mean Dice, all mapped"] = round(float(np.mean([v["Dice"] for v in r.values()])), 4)
        r["mean Dice, STRICT's bundles"] = round(float(np.mean([r[b]["Dice"] for b in RP.STRICT])), 4)
        out["tractseg (binary masks)"] = r
    res["phantom: Dice against the ground truth"] = out


def phantom_scores(ph):
    """The 2023 scoring of ptt and of the seeding controls, as robust_phantom.analyze gives today's."""
    import robust_phantom as RP
    for who, d in PHANTOM_RUNS[1:]:
        pp = phantom_dir(d)
        if not (pp / "score" / "labeled" / "results.json").exists():
            continue
        log = json.loads((pp / "runs.json").read_text())
        lab = np.load(pp / "labels.npz")
        vd, pair = RP.verdicts(pp)
        sc = pp / "score"
        seg_res = json.loads((sc / "segmentation" / "results.json").read_text())
        run_keys = ("seeds", "seeds of today's", "streamlines", "labeled", "Other share", "seconds", "total_s")
        run = {k: log["phantom"][k] for k in run_keys if k in log["phantom"]}
        if "dipy" in log["phantom"]:                                       # the seeds DIPY tracked from (a seeds file's
            run["seeds"] = log["phantom"]["dipy"].get("seeds", run.get("seeds"))     # count; logs before the fix had n)
            run["settings"] = log["phantom"]["dipy"]["settings"]
        ph[who] = {"run": run,
                   "all streamlines": RP.headline(json.loads((sc / "final" / "results.json").read_text()), vd, np.ones(len(vd), bool))
                   | {"IB": seg_res.get("IB"), "IC": seg_res.get("IC"), "NC": seg_res.get("NC")},
                   "labeled streamlines": RP.headline(json.loads((sc / "labeled" / "results.json").read_text()), vd, lab["keep"]),
                   "labels against the truth": RP.labels_against_truth(vd, pair, lab)}


def analyze(args):
    from robust_ukf_score import Presence, top_probability
    import robust_phantom as RP
    t0 = time.time()
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    d = DATA / "ensemble" / SUBJECT / "retest" / PAIR
    affine = np.load(d / "landmarks.npz")["affine"]
    A0 = E.RunRecord.load(d / "A.npz", logp=False)
    presence = Presence(affine, A0.shape)
    sites = json.loads((HERE / "results/ensemble_landmarks_smooth.json").read_text())["sites"]
    today = json.loads((HERE / "results/robust_ukf_score.json").read_text())["configurations"]["today"]
    retest_log = json.loads((d / "runs.json").read_text())
    res.update({"pair": PAIR, "focus": FOCUS, "site radius mm": SITE_MM, "seeds": SEEDS})
    ciii = res.setdefault("cIIIs04", {})
    # ukf: today's records
    ukf_recs = {s: E.RunRecord.load(d / ("A.npz" if s == "A" else f"{s}_onA.npz"), logp=False, fibers=True) for s in CIII_SCANS}
    ukf = tracker_measures(ukf_recs, presence, sites)
    ukf["reproduces robust_ukf_score.json's today (presence Dice, sites)"] = all(
        ukf["rescans"][s]["presence Dice"] == today["(a) rescan agreement"][s]["presence Dice"]
        and ukf["rescans"][s][f"landmark sites: the same main named tract within {SITE_MM:g} mm"]
        == today["(a) rescan agreement"][s][f"landmark sites: the same main named tract within {SITE_MM:g} mm"] for s in ukf["rescans"])
    ukf["Other share, A"] = today["A"]["Other share"]
    ukf["mean top tract probability, A"] = today["A"]["mean top tract probability"]
    ukf["seconds"] = {s: retest_log[s if s == "A" else f"{s}_onA"]["seconds"] | {"total_s": retest_log[s if s == "A" else f"{s}_onA"]["total_s"]}
                      for s in CIII_SCANS}
    ukf["device"] = "Metal (the M2's GPU); the field estimate partly on the CPU"
    ciii["ukf (today)"] = ukf
    counts = {s: len(r.keys) for s, r in ukf_recs.items()}
    spread = ptt_spread(presence, ukf_recs)
    if spread:
        ciii["ptt: run to run (replicates with other RNGs)"] = spread
    pd = OUT / "ptt" / SUBJECT
    if all((pd / f"{s}.npz").exists() for s in CIII_SCANS):
        ptt_main = {s: E.RunRecord.load(pd / f"{s}.npz", logp=False, fibers=True) for s in CIII_SCANS}
        cm = {"rule": f"per scan and tract, both methods subsampled to the smaller count; {DRAWS} draws; [mean, min, max]",
              "ptt as run": matched_rescan_dice(presence, ukf_recs, ptt_main)}
        edge = {s: pd / "replicates" / f"{s}_seed1_track1_edge.npz" for s in ("within", "session5")}
        if all(p.exists() for p in edge.values()):
            cm["ptt, edge rule"] = matched_rescan_dice(presence, ukf_recs, {**ptt_main, **{s: E.RunRecord.load(p, logp=False, fibers=True)
                                                                                          for s, p in edge.items()}})
        ciii["count-matched per-tract rescan Dice (ukf against ptt)"] = cm
        del ptt_main
        ciii["ptt: rescan against run to run, by RNG"] = rng_pairs(presence)
    del ukf_recs
    # ptt
    pd = OUT / "ptt" / SUBJECT
    have = [s for s in CIII_SCANS if (pd / f"{s}.npz").exists()]
    if "A" in have:
        recs = {s: E.RunRecord.load(pd / f"{s}.npz", logp=True, fibers=True) for s in have}
        log = json.loads((pd / "runs.json").read_text())
        tA = L[recs["A"].cluster]
        mixA = np.bincount(tA, minlength=E.N_TRACTS) / len(tA)
        tv = 0.5 * float(np.abs(mixA - np.asarray(today["tract mix, A"])).sum())
        _, _, lenA = lengths(recs["A"].fibers)
        ptt = {"scans": have, "raw": tracker_measures(recs, presence, sites),
               "count-matched to today's UKF labeled count per scan": tracker_measures(
                   {s: subsample(r, counts[s]) for s, r in recs.items()}, presence, sites),
               "A": {"seeds": log["A"]["seeds"], "streamlines (valid, 20-250 mm)": log["A"]["streamlines"],
                     "labeled (>= 40 mm)": log["A"]["labeled"], "Other share": log["A"]["Other share"],
                     "tract mix: total variation against today's UKF A": round(tv, 4),
                     "mean top tract probability": round(top_probability(recs["A"]), 4),
                     "labeled length mm (quartiles)": [round(float(x), 1) for x in np.quantile(lenA, [0.25, 0.5, 0.75])],
                     "focus tracts' share of labeled": {n: round(float(mixA[TRACT_NAMES.index(n)]), 4) for n in FOCUS},
                     "tract mix": mixA.tolist()},
               "Other share": {s: log[s]["Other share"] for s in have},
               "seconds": {s: log[s]["seconds"] | {"total_s": log[s]["total_s"]} for s in have},
               "device": "the CPU (DIPY, OpenMP threads: os.cpu_count()); RapidParc on Metal"}
        ciii["ptt"] = ptt
        del recs
    # tractseg
    td = OUT / "tractseg" / SUBJECT
    have = [s for s in CIII_SCANS if (td / f"{s}.npz").exists()]
    if "A" in have:
        masks, ts_aff = {}, None
        for s in have:
            masks[s], ts_aff = load_ts(td / f"{s}.npz")
        log = json.loads((td / "runs.json").read_text())
        ts = tractseg_measures(masks, ts_aff, sites)
        ts["scans"] = have
        ts["seconds"] = {s: log[s]["seconds"] | {"total_s": log[s]["total_s"]} for s in have}
        ts["device"] = "the CPU (TractSeg's torch; DIPY for the peaks)"
        ciii["tractseg"] = ts
    # phantom
    ph = res.setdefault("phantom", {})
    ukf_ph = json.loads(RP.RESULTS.read_text())["configs"].get("today") if RP.RESULTS.exists() else None
    if ukf_ph:
        ph["ukf (today), from robust_phantom.json"] = {k: ukf_ph[k] for k in ("all streamlines", "labeled streamlines") if k in ukf_ph}
    phantom_scores(ph)
    pp = OUT / "ptt" / "phantom"
    if (OUT / "tractseg" / "phantom" / "masks.npz").exists() or (pp / "tractogram.trk").exists():
        phantom_dice(ph)
    if (OUT / "tractseg" / "phantom" / "runs.json").exists():
        ph["tractseg run"] = json.loads((OUT / "tractseg" / "phantom" / "runs.json").read_text()).get("phantom")
    res["domain"] = ("RapidParc was trained on UKF tractograms (ORG atlas): ptt's Other share and tract mix are the "
                     "domain-shift check. TractSeg was trained on HCP (multi-shell, 270 directions, 1.25 mm, MRtrix CSD "
                     "peaks): cIIIs04 (b = 2000, 56 directions, 2.1 mm, DIPY's peaks) and the phantom (b = 1000, 32 "
                     "directions, 2 mm, simulated) are off its domain.")
    RESULTS.write_text(json.dumps(res, indent=1))
    print(f"analyze {time.time() - t0:.0f}s ->", RESULTS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run"); a.add_argument("--method", nargs="+", required=True); a.add_argument("--scans", nargs="+", required=True)
    a = sub.add_parser("pilot"); a.add_argument("--scan", default="A", choices=("A", "phantom"))
    a.add_argument("--seeds", type=int, default=20000); a.add_argument("--keep-work", action="store_true")
    sub.add_parser("diagnose")
    a = sub.add_parser("controls"); a.add_argument("names", nargs="+", help=f"among {CONTROLS}")
    a = sub.add_parser("replicate"); a.add_argument("runs", nargs="+", help="scan:seed_rng:track_rng, e.g. A:2:2 A:1:2")
    sub.add_parser("check")
    sub.add_parser("analyze")
    args = ap.parse_args()
    cmds = {"run": run, "pilot": pilot, "replicate": replicate, "controls": controls, "diagnose": diagnose, "check": check, "analyze": analyze}
    cmds[args.cmd](args)


if __name__ == "__main__":
    main()
