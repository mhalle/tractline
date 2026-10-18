"""Accuracy on the ISMRM 2015 phantom: today's pipeline and the robustness options, scored against a known ground truth
with the 2023 ROI scoring (docs/robustness-plan.md, step 4; docs/validation-datasets.md, "ISMRM 2015 Tractography
Challenge"; the phantom as a subject: bench/_ismrm2015.py).

    uv run --group dipy --group rankfield bench/robust_phantom.py check
    uv run --group dipy --group rankfield bench/robust_phantom.py run --config today [--config Rs=0.1 ...]   (or --config all)
    uv run --group dipy --group rankfield bench/robust_phantom.py analyze
    uv run --group dipy --group rankfield bench/robust_phantom.py patch2self
    uv run --group dipy --group rankfield bench/robust_phantom.py ensemble [--first N]
    uv run --group dipy --group rankfield bench/robust_phantom.py rescore          (the review: WPC, --unique, F1 bands)
    uv run --group dipy --group rankfield bench/robust_phantom.py envelopes        (after ensemble: a count-matched truth)
    uv run --group dipy --group rankfield bench/robust_phantom.py review_checks    (the readout's scale, Patch2Self robustly)

check: the loader's three decisions, from the data, into results/robust_phantom.json ("loader"):
  - the gradients' x sign: the corrected DWI's tensor (a log-linear fit) against the ground-truth bundles' own
    direction, in voxels one bundle holds (> 95 % of the segments, coherence > 0.9, FA > 0.3), with x as FSL's
    convention reads the .bvecs and flipped;
  - the phase-encoding sign: the field estimated with each, against fmap_Hz (its sign follows the convention);
  - the readout time: the estimated field's scale against fmap_Hz (slope through 0, and the SD ratio) over the brain
    (the Otsu threshold of the mean b0) and the tracking mask: if fmap_Hz is the simulated field, the readout time
    that matches it is READOUT_S times the slope.
run --config <name>: the configuration on the phantom, into DATA/ensemble/phantom_ismrm2015/<name>/ (never overwritten:
a run whose record exists is not tracked again, a scoring whose results exist is not rerun; stops below MIN_FREE_GB):
  record.npz       the labeled streamlines (>= 40 mm), compact (ensemble.RunRecord), on the DWI's grid
  labels.npz       per streamline in seed order: the seed key, labeled (>= 40 mm), and the labeled ones' tract
  field_hz.npy     the estimated field (float32), and the run's log in runs.json (times, fibers, the field against fmap_Hz)
  tractogram.trk   every streamline, in the scorer's reference space (t1.nii.gz: its affine, dimensions, voxel sizes and
                   LPS voxel order in the header; the points are the same RAS mm), with each streamline's seed-order
                   index as data_per_streamline "index", so the scorer's outputs map back to the labels
  score/           the 2023 scoring (score()): segmentation/ (scil_score_tractogram), final/ (the sub-bundles merged,
                   scil_score_bundles), labeled/ (the same on the labeled streamlines only)
analyze: per configuration, into results/robust_phantom.json ("configs"): the scorer's numbers, for all streamlines and
for the labeled ones, and RapidParc's labels against the ground-truth bundles (labels_against_truth()).

Configurations (CONFIGS; a name is read by parse(): tracker settings, "denoise=", "grouping="):
  today                                         the pipeline's defaults
  tie_angle=0.5_sw-and-sw2,grouping=hash-sorted the tie-break holding both swaps (step 2b) and RapidParc's hash-sorted
                                                grouping (step 1)
  denoise=patch2self                            Patch2Self before the field (step 3)
  step_length=0.6, Rs=0.1                       the tracker settings step 2 scored
The phantom is b = 1000 with 32 directions: shell=1000 (the pipeline's default is 2800).

The 2023 scoring (Renauld et al. 2023), from scilpy 1.5.0's scripts (an isolated environment, SCILPY) and the scoring
archive's two config files, as the tractometer's scil_score_ismrm_Renauld2023.sh is described: segment with
scil_score_tractogram (config_file_segmentation.json: 26 bundles, the CC in four sub-bundles and each ICP in two), merge
the sub-bundles with scil_tractogram_math union, score the 21 bundles with scil_score_bundles
(config_file_tractometry.json). The wrapper itself was not downloaded; its options are not known here. Chosen:
  --compute_ic   invalid connections (IC: an invalid pair of endpoint regions) apart from no connection (NC);
  --unique       each streamline in at most one bundle (the first it fits, in the config's order), so VS, IC and NC
                 partition the tractogram and each streamline has one verdict;
  no endpoint dilation, WPC (right endpoints, outside the bundle's mask) counted as invalid, as by default.
scil_score_bundles in 1.5.0 counts a segmented_IB/ file as the reference tractogram's size (ib_sft_list.append(ref_sft)),
so final/ holds segmented_VB/ and an IS.trk (NC and every IC concatenated) instead, and IB, IC and NC come from the
segmentation's results. An invalid bundle is any non-valid pair of endpoint regions with at least one streamline (the
2015 scoring clustered them); scilpy's mean OL, OR and F1 are over the bundles found (mean_*_found here), so the means
over all 21 (a missing bundle 0) are given too.

patch2self: why Patch2Self halves the streamlines here, into results/robust_phantom.json ("patch2self"). The phantom's
noise-free DWI is not on disk (Zenodo's ground-truth DWI was not downloaded), so the noise and the signal are read from
the data: the noise SD from the background (outside the tracking mask dilated by BACKGROUND_DILATE voxels; Rayleigh, from
the mean and from the median, which the N/2 ghosts inflate less), and the bootstrap's residual RMS about the shell's
spherical-harmonic fit in the mask; in the voxels one ground-truth bundle holds (check's selection, without the FA cut),
raw and denoised (both corrected with today's field): the DW volumes' mean, their spread across directions (the angular
contrast plus noise; with white noise of SD sigma, the noise-free spread is about sqrt(raw^2 - sigma^2)), FA, the angle
to the bundle's direction, and what Patch2Self removed (RMS; its correlation with the raw volumes' deviation from their
own mean: what varies with direction); the tracker's seeds (ukf.seeds) on each; and the labeled streamlines' lengths.

ensemble: today's 17-run ensemble on the phantom (A, today's run, and ENSEMBLE_REPLICATES bootstrap replicates as
robust_ukf_score makes them: ShellWildBootstrap, smooth signs, FWHM 4, b0 noise "added", A's tracking mask, seeded by r;
the phantom's one b0 a polarity borrows a DW volume's residual, _bootstrap.py), each corrected, tracked and labeled as
today, into DATA/ensemble/phantom_ismrm2015/today/ensemble/r###/ (record.npz, labels.npz, tractogram.trk, verdicts.npz,
runs.json; never overwritten; MIN_FREE_GB). Each replicate's streamlines are assigned to the 2023 bundles by
scil_score_tractogram alone (--unique, without --compute_ic: the valid streamlines are extracted before the invalid
connections, so their assignment is the same; 16 s against 166 s; checked on A), its directory removed once the
verdicts are saved. Then, per bundle with a strict label counterpart, presence per run on the t1 grid (density, as
ensemble_grid_check: streamlines sampled every 0.4 mm, at least 0.3 per mm^2 after a 1 mm smoothing), from
  - scored: the streamlines the scorer assigns to the bundle (every length);
  - labeled: the labeled streamlines (>= 40 mm) whose tract is the bundle's strict counterpart, on the bundle's side
    (the streamline's mean x against the DWI grid's center) for a lateral bundle;
and the votes (0-17) against the bundle's ground-truth mask (scil_compute_streamlines_density_map --binary of its .trk:
the scorer's own mask, every voxel a ground-truth streamline crosses): for A alone, the widest outline (any run), the
core (at least half) and the envelopes at ensemble_pooled's stated levels, coverage of the truth |S n T| / |T|,
precision |S n T| / |S| and size |S| / |T|, per bundle and pooled over the bundles (weighted by |T|). Into
results/robust_phantom.json ("envelopes"). The traversal mask is not a like-for-like target for a run's density
presence (it marks every voxel any ground-truth streamline crosses); envelopes() replaces it with the ground truth's own
density presence at each run's count, and rescore(), review_checks() hold the review's other checks.
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np, nibabel as nib
from tractline import pipeline as P
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import TRACT_NAMES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _ismrm2015 as I                                                                    # noqa: E402

DEVICE = "mps"
OUT = DATA / "ensemble" / "phantom_ismrm2015"
RESULTS = HERE / "results" / "robust_phantom.json"
SCILPY = DATA / "tools" / "scilpy-1.5.0" / "bin"
MIN_FREE_GB = float(os.environ.get("TRACTLINE_MIN_FREE_GB", 20.0))   # as robust_ukf_diagnose's
BACKGROUND_DILATE = 4                        # patch2self: the background is outside the tracking mask dilated this much
ENSEMBLE_REPLICATES = 16
CONFIGS = ("today", "tie_angle=0.5_sw-and-sw2,grouping=hash-sorted", "denoise=patch2self", "step_length=0.6", "Rs=0.1")
SW2_SUFFIX = "_sw-and-sw2"                   # robust_ukf_score's: tie_sw2 with the tie angle
GROUPINGS = ("index", "hash-sorted", "hash-bucket")
#: the sub-bundles the segmentation config splits and the tractometry config scores whole (the wrapper's merge)
MERGED = {"CC": ("CC_temporal", "CC_u_shaped", "CC_ventro_striatal1", "CC_ventro_striatal2"),
          "ICP_left": ("ICP_left_part1", "ICP_left_part2"), "ICP_right": ("ICP_right_part1", "ICP_right_part2")}

#: the 2023 bundles with an ORG counterpart, conservatively (docs/validation-datasets.md's table): strict, and loose
#: (projection tracts the BPS's frontopontine and parieto-occipito-pontine parts may be labeled; the optic radiation as
#: thalamo-occipital). The BPS is CST + FPT + POPT merged; CST alone is its strict counterpart. No counterpart: CA, CP,
#: Fornix, SCP; OR only loosely.
STRICT = {"CC": ("CC1", "CC2", "CC3", "CC4", "CC5", "CC6", "CC7"), "Cingulum_left": ("CB",), "Cingulum_right": ("CB",),
          "BPS_left": ("CST",), "BPS_right": ("CST",), "ICP_left": ("ICP",), "ICP_right": ("ICP",), "MCP": ("MCP",),
          "ILF_left": ("ILF",), "ILF_right": ("ILF",), "SLF_left": ("SLF-I", "SLF-II", "SLF-III"),
          "SLF_right": ("SLF-I", "SLF-II", "SLF-III"), "UF_left": ("UF",), "UF_right": ("UF",)}
LOOSE = {"BPS_left": ("CR-F", "CR-P", "CPC"), "BPS_right": ("CR-F", "CR-P", "CPC"), "OR_left": ("TO",), "OR_right": ("TO",)}
NO_COUNTERPART = ("CA", "CP", "Fornix", "SCP_left", "SCP_right", "OR_left", "OR_right")
#: tracts the phantom cannot hold: the 2015 paper's invalid bundles with an ORG name, and the superficial ones
FALSE_TRACTS = ("AF", "EC", "EmC", "IOFF", "MdLF") + tuple(t for t in TRACT_NAMES if t.startswith("Sup-"))
#: what the challenge reports, for orientation only (docs/validation-datasets.md): the 2015 submissions under the 2015
#: scoring (RecoBundles segmentation, 25 bundles), and Renauld 2023's best F1 under the ROI scoring
PUBLISHED = {"2015 submissions, 2015 scoring (Maier-Hein 2017)": {
                 "submissions": 96, "valid bundles of 25": "mean 21, median 23", "invalid bundles": "88 +- 58",
                 "valid streamlines": "54 +- 23 %", "overlap": "36 +- 16 %", "overreach": "29 +- 26 %"},
             "2023 ROI scoring (Renauld 2023)": {
                 "ground truth against itself, F1": "92 %", "TractoFlow with topup, F1": "up to 58 % (the site's erratum; "
                 "the paper: 64-65 %)", "the same run without the reversed b0, F1": "57 % (as docs/validation-datasets.md "
                 "states it; against the erratum's 58 % it reads oddly, unchecked)"}}


def free_gb(path):
    p = Path(path)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free / 1e9


def parse(config):
    """A configuration's name -> {"tracker": ukf settings, "denoise": method or None, "grouping": RapidParc's}."""
    out = {"tracker": {}, "denoise": None, "grouping": "index"}
    if config == "today":
        return out
    name = config
    for part in name.split(","):
        k, v = part.split("=", 1)
        if v.endswith(SW2_SUFFIX):
            v, out["tracker"]["tie_sw2"] = v[:-len(SW2_SUFFIX)], True
        if k == "denoise":
            from tractline import denoise as D
            D.parse(v)
            out["denoise"] = v
        elif k == "grouping":
            if v not in GROUPINGS:
                raise SystemExit(f"{config}: grouping one of {GROUPINGS}")
            out["grouping"] = v
        elif k in P.TRACKER_SETTINGS:
            out["tracker"][k] = bool(float(v)) if k == "tie_sw2" else float(v)
        else:
            raise SystemExit(f"{config}: {k} is not a tracker setting ({sorted(P.TRACKER_SETTINGS)}), denoise or grouping")
    return out


# ----------------------------------------------------------------------------------------------- the field and the tensor

def brain_mask(s):
    from tractline.mask import otsu
    b0 = s.b0s.mean(-1)
    return b0 > otsu(b0)


def against_fmap(h, s, masks):
    """The estimated field (Hz) against fmap_Hz, per mask: Pearson r, the slope through 0 (estimated on fmap), the SD
    ratio, the RMS difference in Hz and as displacement (mm, at the subject's readout time), and the readout time that
    would make the slope 1 (if fmap_Hz is the simulated field)."""
    f = I.fmap_hz()
    out = {}
    for name, m in masks.items():
        x, y = f[m], np.asarray(h, np.float64)[m]
        slope = float(x @ y / (x @ x))
        rms = float(np.sqrt(((y - x) ** 2).mean()))
        out[name] = {"voxels": int(m.sum()), "r": round(float(np.corrcoef(x, y)[0, 1]), 4), "slope": round(slope, 4),
                     "sd_ratio": round(float(y.std() / x.std()), 4), "fmap_sd_hz": round(float(x.std()), 2),
                     "rms_diff_hz": round(rms, 2), "rms_diff_mm": round(rms * s.readout_s * float(s.vox[s.pe_axis]), 3),
                     "abs_fmap_p99_hz": round(float(np.percentile(np.abs(x), 99)), 1),
                     "readout_s_matching": round(s.readout_s * slope, 4)}
    return out


def gt_directions(affine, shape):
    """The ground-truth bundles' (bundles/*.trk) segment directions on the DWI's grid: (voxel indices (m, 3), the main
    direction (m, 3) RAS, coherence (m,): the mean outer product's largest eigenvalue, the dominant bundle's share of
    segments (m,), segments (m,))."""
    inv = np.linalg.inv(affine)
    paths = sorted((I.SCORING / "bundles").glob("*.trk"))
    T = np.zeros((len(paths),) + tuple(shape) + (3, 3), np.float32)
    for b, p in enumerate(paths):
        for sl in nib.streamlines.load(p).streamlines:
            d = np.diff(sl, axis=0)
            n = np.linalg.norm(d, axis=1)
            ok = n > 0
            d, mid = d[ok] / n[ok, None], (sl[:-1][ok] + sl[1:][ok]) / 2
            ijk = np.rint(mid @ inv[:3, :3].T + inv[:3, 3]).astype(int)
            inb = (ijk >= 0).all(1) & (ijk < shape).all(1)
            np.add.at(T[b], tuple(ijk[inb].T), np.einsum("ni,nj->nij", d[inb], d[inb]))
    seg = np.trace(T, axis1=-2, axis2=-1)                                  # (B, X, Y, Z)
    tot = seg.sum(0)
    idx = np.argwhere(tot > 0)
    at = tuple(idx.T)
    w, v = np.linalg.eigh(T.sum(0)[at] / tot[at][:, None, None])
    return idx, v[:, :, -1], w[:, -1], seg.max(0)[at] / tot[at], tot[at]


def tensor_e1(dwi, s, idx):
    """The tensor's principal direction (RAS) and FA at voxels idx, a log-linear least-squares fit of the shell, with
    s's gradients turned into RAS as prep.prepare turns them (FSL's convention)."""
    M = s.affine[:3, :3]
    g = s.bvec[:, s.bval > 50].copy()
    if np.linalg.det(M) > 0:
        g[0] *= -1
    g = (M / np.linalg.norm(M, axis=0) @ g).T
    b = s.bval[s.bval > 50]
    X = -b[:, None] * np.stack([g[:, 0] ** 2, g[:, 1] ** 2, g[:, 2] ** 2, 2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2],
                                2 * g[:, 1] * g[:, 2]], 1)
    at = tuple(idx.T)
    S0 = np.clip(dwi[..., s.bval < 50].mean(-1)[at], 1, None)
    y = np.log(np.clip(dwi[..., s.bval > 50][at], 1, None) / S0[:, None])
    c = np.linalg.lstsq(X, y.T, rcond=None)[0].T
    D = np.zeros((len(idx), 3, 3))
    for k, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))):
        D[:, i, j] = D[:, j, i] = c[:, k]
    ev, evec = np.linalg.eigh(D)
    md = ev.mean(1, keepdims=True)
    fa = np.sqrt(1.5 * ((ev - md) ** 2).sum(1) / np.maximum((ev ** 2).sum(1), 1e-12))
    return evec[:, :, -1], fa


def check(args):
    t0 = time.time()
    loader = {"constants": {"GRADIENT_X": I.GRADIENT_X, "PE_SIGN": I.PE_SIGN, "READOUT_S": I.READOUT_S,
                            "readout derivation": f"({I.PE_LINES} phase-encoding lines - 1) x {I.DWELL_S * 1e3:g} ms dwell "
                                                  "(Fiberfox's tLine: one k-space line); TE 108 ms = 108 lines x 1 ms",
                            "shell": I.SHELL}}
    fields = {}
    for sign in (I.PE_SIGN, -I.PE_SIGN):
        s = I.load(pe_sign=sign)
        timer = P.Timer()
        with P.exact_float32(DEVICE):
            c = P.correct(s, timer, DEVICE)
        P.release_memory(DEVICE)
        fields[sign] = (s, c)
        loader[f"field, pe_sign {sign:+g}"] = {"seconds": timer.seconds, "residual_left": c.residual_left,
                                               "against fmap_Hz": against_fmap(c.field_hz, s, {"brain": brain_mask(s)})}
    s, c = fields[I.PE_SIGN]
    t = P.prepare(c.dwi, s.affine, s.bval, s.bvec, shell=I.SHELL, device=DEVICE)
    loader[f"field, pe_sign {I.PE_SIGN:+g}"]["against fmap_Hz"] = against_fmap(c.field_hz, s, {"brain": brain_mask(s),
                                                                                                  "tracking mask": t.mask})
    idx, gdir, coh, share, nseg = gt_directions(s.affine, s.dwi.shape[:3])
    xs = {}
    for gx in (1.0, -1.0):                                             # the .bvecs' x as FSL's convention reads it, or flipped
        e1, fa = tensor_e1(c.dwi.astype(np.float64), I.load(gradient_x=gx, pe_sign=I.PE_SIGN), idx)
        ang = np.degrees(np.arccos(np.clip(np.abs((e1 * gdir).sum(1)), 0, 1)))
        sel = (share > 0.95) & (coh > 0.9) & (nseg > 20) & (fa > 0.3)
        obl = sel & (np.abs(gdir[:, 0]) > 0.3) & (np.abs(gdir[:, 0]) < 0.9)
        key = "as FSL's convention reads the .bvecs" if gx == 1 else "x flipped against FSL's convention"
        xs[key] = {"voxels": int(sel.sum()), "median_angle_deg": round(float(np.median(ang[sel])), 1),
                   "oblique_in_x_voxels": int(obl.sum()), "oblique_median_angle_deg": round(float(np.median(ang[obl])), 1),
                   "oblique_within_20_deg": round(float(np.mean(ang[obl] < 20)), 3)}
    loader["gradient x sign: the tensor against the ground-truth bundles' directions"] = xs
    loader["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(loader, indent=1))
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res["loader"] = loader
    RESULTS.write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------------------------- tracking

def write_trk(path, fibers):
    """Every streamline (RAS mm, the DWI's world, which t1.nii.gz shares: the archive's README) into a .trk whose header
    is the reference's (affine, dimensions, voxel sizes, voxel order), each with its seed-order index; read back and
    checked: the same points, every point inside the reference's grid."""
    ref = nib.load(I.REFERENCE)
    F = nib.streamlines.Field
    hdr = {F.VOXEL_TO_RASMM: ref.affine.astype(np.float32), F.DIMENSIONS: np.array(ref.shape[:3], np.int16),
           F.VOXEL_SIZES: np.array(ref.header.get_zooms()[:3], np.float32), F.VOXEL_ORDER: "".join(nib.aff2axcodes(ref.affine))}
    tg = nib.streamlines.Tractogram([np.asarray(f, np.float32) for f in fibers], affine_to_rasmm=np.eye(4),
                                    data_per_streamline={"index": np.arange(len(fibers), dtype=np.float32)[:, None]})
    tmp = path.with_suffix(".tmp.trk")
    nib.streamlines.save(tg, str(tmp), header=hdr)
    back = nib.streamlines.load(str(tmp))
    assert np.allclose(back.header[F.VOXEL_TO_RASMM], ref.affine) and back.header[F.VOXEL_ORDER] == b"LPS", back.header
    pts = back.streamlines.get_data()
    assert len(back.streamlines) == len(fibers) and np.abs(pts - np.concatenate(fibers)).max() < 1e-3
    assert (back.tractogram.data_per_streamline["index"][:, 0] == np.arange(len(fibers))).all()
    vox = nib.affines.apply_affine(np.linalg.inv(ref.affine), pts)
    inside = float(((vox >= -0.5) & (vox < np.array(ref.shape[:3]) - 0.5)).all(1).mean())
    assert inside == 1.0, f"{1 - inside:.2%} of the points outside the reference grid"
    if path.exists():
        tmp.unlink()
        raise SystemExit(f"{path} appeared meanwhile; not overwritten")
    os.replace(tmp, path)
    return {"streamlines": len(fibers), "points": int(len(pts)), "points inside the reference grid": inside,
            "header": {"voxel_order": "LPS", "dimensions": list(map(int, ref.shape[:3])), "affine": ref.affine.tolist()}}


def track(config, out):
    """The configuration on the phantom: (denoise), correct, track at b = 1000, label; the record, the labels, the field and
    the .trk under out (runs.json logs the run)."""
    cfg = parse(config)
    if free_gb(out) < MIN_FREE_GB:
        raise SystemExit(f"stopping: {free_gb(out):.1f} GB free, below {MIN_FREE_GB} GB")
    out.mkdir(parents=True, exist_ok=True)
    t0, timer = time.time(), P.Timer(echo=config)
    s = I.load()
    log = {"config": cfg, "loader": s.notes}
    if cfg["denoise"]:
        s, log["denoise"] = P.denoise_subject(s, timer, cfg["denoise"])
    labeler = P.default_labeler(DEVICE)
    with P.exact_float32(DEVICE):
        c = P.correct(s, timer, DEVICE)
        tg = P.track(s, c.dwi, timer, device=DEVICE, shell=I.SHELL, tracker=cfg["tracker"])
        keys = E.seed_keys(np.asarray(tg.stats["seed_voxel"]))
        with timer("label"):
            labels = labeler(tg.fibers, draws=(0,), logp=True,
                             **({} if cfg["grouping"] == "index" else {"keys": keys, "grouping": cfg["grouping"]}))
    P.release_memory(DEVICE)
    if free_gb(out) < MIN_FREE_GB:
        raise SystemExit(f"stopping: {free_gb(out):.1f} GB free, below {MIN_FREE_GB} GB")
    for name in ("record.npz", "labels.npz", "field_hz.npy", "tractogram.trk"):
        if (out / name).exists():
            raise SystemExit(f"{out / name} exists; not overwritten")
    with timer("write"):
        np.save(out / "field_hz.npy", c.field_hz.astype(np.float32))
        np.savez_compressed(out / "labels.npz", keys=keys, keep=labels.keep, tract=labels.tract.astype(np.int16),
                            length_mm=labels.length_mm.astype(np.float32))
        log["trk"] = write_trk(out / "tractogram.trk", tg.fibers)
        tmp = out / "record.tmp.npz"
        E.RunRecord.of(tg, labels, s.affine, s.dwi.shape[:3]).save(tmp, compact=True)
        os.replace(tmp, out / "record.npz")                               # last: its presence marks a finished run
    log.update({"fibers": len(tg.fibers), "labeled": int(labels.keep.sum()), "residual_left": c.residual_left,
                "warnings": c.warnings, "mask_voxels": int(tg.mask.sum()),
                "field against fmap_Hz": against_fmap(c.field_hz, s, {"brain": brain_mask(s), "tracking mask": tg.mask}),
                "seconds": timer.seconds, "total_s": round(time.time() - t0, 1)})
    (out / "runs.json").write_text(json.dumps(log, indent=1))
    print(config, json.dumps({k: log[k] for k in ("fibers", "labeled", "residual_left", "total_s")}), flush=True)


# ----------------------------------------------------------------------------------------------- scoring

def scil(script, *argv, cwd):
    """A scilpy 1.5.0 script in its own environment (its shebang's interpreter), with none of this one's Python settings."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSAFEPATH")}
    t0 = time.time()
    r = subprocess.run([str(SCILPY / script), *map(str, argv)], cwd=cwd, env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{script} failed ({r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return round(time.time() - t0, 1)


def trk_index(path):
    return nib.streamlines.load(str(path)).tractogram.data_per_streamline["index"][:, 0].astype(np.int64)


def final(seg, tmp, subset=None, cwd=None):
    """From a segmentation directory, the wrapper's last steps into tmp (made afresh): the sub-bundles merged (union), the
    other valid bundles copied, IS.trk (NC and every IC concatenated, or the segmentation's own IS.trk without
    --compute_ic), optionally only the streamlines in subset (a bool per seed-order index), then scil_score_bundles
    (results.json in tmp)."""
    cwd = cwd or seg.parent
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "segmented_VB").mkdir(parents=True)
    vb = {p.name[:-len("_VS.trk")]: p for p in (seg / "segmented_VB").glob("*_VS.trk")}
    parts = {p for ps in MERGED.values() for p in ps}
    for name, p in vb.items():                                             # the bundles scored whole: as segmented
        if name not in parts:
            shutil.copy(p, tmp / "segmented_VB" / p.name)
    for name, ps in MERGED.items():                                        # the sub-bundles merged (the wrapper's union)
        got = [vb[q] for q in ps if q in vb]
        if got:
            scil("scil_tractogram_math.py", "union", *got, tmp / "segmented_VB" / f"{name}_VS.trk", "--reference",
                 I.REFERENCE, "--no_bbox_check", cwd=cwd)
    invalid = [seg / f for f in ("NC.trk", "IS.trk") if (seg / f).exists()]
    invalid += sorted((seg / "segmented_IB").glob("*.trk")) if (seg / "segmented_IB").exists() else []
    if invalid:
        scil("scil_tractogram_math.py", "concatenate", *invalid, tmp / "IS.trk", "--reference", I.REFERENCE,
             "--no_bbox_check", cwd=cwd)
    if subset is not None:                                                  # only the streamlines in subset
        for p in [*(tmp / "segmented_VB").glob("*.trk"), tmp / "IS.trk"]:
            if not p.exists():
                continue
            t = nib.streamlines.load(str(p))
            m = subset[t.tractogram.data_per_streamline["index"][:, 0].astype(np.int64)]
            p.unlink()
            if m.any():
                nib.streamlines.save(t.tractogram[np.flatnonzero(m)], str(p), header=t.header)
    scil("scil_score_bundles.py", I.SCORING / "config_file_tractometry.json", tmp, "--json_prefix", "", "--indent", "1",
         "--gt_dir", I.SCORING, "--reference", I.REFERENCE, "--no_bbox_check", cwd=cwd)
    return json.loads((tmp / "results.json").read_text())


def score(out):
    """The 2023 ROI scoring of out/tractogram.trk into out/score/ (module docstring): segmentation/, final/ and labeled/,
    each with its results.json; the seconds in out/score/seconds.json. A step whose results exist is not rerun; a step
    begun and not finished is refused (remove its directory)."""
    sc = out / "score"
    sc.mkdir(exist_ok=True)
    seg, fin, lab = sc / "segmentation", sc / "final", sc / "labeled"
    secs_path = sc / "seconds.json"
    secs = json.loads(secs_path.read_text()) if secs_path.exists() else {}
    common = ["--gt_dir", I.SCORING, "--reference", I.REFERENCE, "--no_bbox_check"]
    for d in (seg, fin, lab):
        if d.exists() and not (d / "results.json").exists():
            raise SystemExit(f"{d} exists without results.json: an unfinished scoring; remove it to rerun")
    if not (seg / "results.json").exists():
        secs["segmentation"] = scil("scil_score_tractogram.py", out / "tractogram.trk", I.SCORING / "config_file_segmentation.json",
                                    seg, "--compute_ic", "--unique", "--no_empty", "--json_prefix", "", *common, cwd=sc)
        secs_path.write_text(json.dumps(secs, indent=1))
    keep = np.load(out / "labels.npz")["keep"]
    for d, subset in ((fin, None), (lab, keep)):
        if (d / "results.json").exists():
            continue
        t0 = time.time()
        tmp = sc / (d.name + ".tmp")
        final(seg, tmp, subset)
        os.replace(tmp, d)
        secs[d.name] = round(time.time() - t0, 1)
        secs_path.write_text(json.dumps(secs, indent=1))
    return secs


def verdicts(out):
    """Per streamline (seed order): its verdict, a 2023 bundle's name (the sub-bundles merged), "IC" or "NC"; and the
    invalid connection's endpoint pair for each IC."""
    seg = out / "score" / "segmentation"
    n = len(np.load(out / "labels.npz")["keep"])
    v = np.full(n, "", dtype=object)
    pair = {}
    part_of = {q: name for name, ps in MERGED.items() for q in ps}
    for p in (seg / "segmented_VB").glob("*_VS.trk"):
        b = p.name[:-len("_VS.trk")]
        v[trk_index(p)] = part_of.get(b, b)
    for p in (seg / "segmented_IB").glob("*.trk") if (seg / "segmented_IB").exists() else []:
        i = trk_index(p)
        assert (v[i] == "").all(), "a streamline in two verdicts (--unique should partition)"
        v[i] = "IC"
        pair.update(dict.fromkeys(i.tolist(), p.name[:-len("_IC.trk")]))
    if (seg / "NC.trk").exists():
        i = trk_index(seg / "NC.trk")
        assert (v[i] == "").all()
        v[i] = "NC"
    assert (v != "").all(), f"{(v == '').sum()} streamlines without a verdict"
    return v, pair


# ----------------------------------------------------------------------------------------------- analysis

def top(values, k=6):
    """The k commonest values and their shares, commonest first."""
    if len(values) == 0:
        return {}
    u, c = np.unique(values, return_counts=True)
    o = np.argsort(-c, kind="stable")[:k]
    return {str(u[i]): round(float(c[i] / len(values)), 4) for i in o}


def headline(results, vd, subset):
    """The scorer's numbers: VB of 21, VS / IS shares (results: scil_score_bundles on the merged bundles); IB, IC and NC
    (the segmentation's, over the streamlines in subset); OL, OR and F1, per bundle and mean (over the bundles found, as
    scilpy gives them, and over all 21)."""
    bw = results["bundle_wise"]
    names = list(json.loads((I.SCORING / "config_file_tractometry.json").read_text()))
    per = {b: {k: (round(bw[b][k], 4) if isinstance(bw[b].get(k), float) else bw[b].get(k))
               for k in ("VS", "OL", "OR_pct_vs", "OR_pct_gt", "f1")} if b in bw else {"VS": 0} for b in names}
    full = lambda k: round(float(np.mean([per[b].get(k, 0.0) for b in names])), 4)
    v = vd[subset]
    n = len(v)
    ic = v == "IC"
    out = {"streamlines": n, "VB": results["VB"], "of": len(names), "VS": results["VS"], "VS_ratio": round(results["VS_ratio"], 4),
           "IS_ratio": round(results["IS_ratio"], 4), "IC_ratio": round(float(ic.mean()), 4),
           "NC_ratio": round(float((v == "NC").mean()), 4),
           "mean_OL_found": round(results.get("mean_OL", 0), 4), "mean_OR_vs_found": round(results.get("mean_OR_vs", 0), 4),
           "mean_OR_gt_found": round(results.get("mean_OR_gt", 0), 4), "mean_f1_found": round(results.get("mean_f1", 0), 4),
           "mean_OL_all21": full("OL"), "mean_OR_vs_all21": full("OR_pct_vs"), "mean_f1_all21": full("f1"),
           "missing_bundles": [b for b in names if not per[b]["VS"]], "bundles": per}
    assert results["VS"] == int(np.isin(v, names).sum()) and results["total_streamlines"] == n, (results["VS"], n)
    return out


def labels_against_truth(vd, pair, lab):
    """RapidParc's labels (the labeled streamlines) against the scorer's verdicts: per 2023 bundle with a counterpart, the
    share of its valid streamlines labeled as the counterpart (strict; strict or loose) and its top labels; per ORG tract
    of the mapping and the false ones, where its streamlines go (the counterpart, another bundle, IC, NC); the invalid
    streamlines' labels; and the AF's in full."""
    keep = lab["keep"]
    v = vd[keep]
    tract = np.array(TRACT_NAMES)[lab["tract"]]
    by_bundle = {}
    for b in [*STRICT, *NO_COUNTERPART]:
        t = tract[v == b]
        r = {"valid_labeled": int(len(t)), "top_labels": top(t)}
        if b in STRICT and len(t):
            r["as_counterpart"] = round(float(np.isin(t, STRICT[b]).mean()), 4)
            r["as_counterpart_or_loose"] = round(float(np.isin(t, STRICT[b] + LOOSE.get(b, ())).mean()), 4)
        elif b in LOOSE and len(t):
            r["as_loose"] = round(float(np.isin(t, LOOSE[b]).mean()), 4)
        by_bundle[b] = r
    mapped = {}
    for b, ts in STRICT.items():
        for t in ts:
            mapped.setdefault(t, set()).add(b)
    by_tract = {}
    for t in [*mapped, *FALSE_TRACTS]:
        m = tract == t
        if not m.any():
            by_tract[t] = {"labeled": 0}
            continue
        vt = v[m]
        own = np.isin(vt, sorted(mapped.get(t, ())))
        other = (vt != "IC") & (vt != "NC") & ~own
        by_tract[t] = {"labeled": int(m.sum()), "valid_counterpart": round(float(own.mean()), 4),
                       "valid_other_bundle": round(float(other.mean()), 4), "IC": round(float((vt == "IC").mean()), 4),
                       "NC": round(float((vt == "NC").mean()), 4), "valid_bundles": top(vt[other | own], 4)}
    inv = (v == "IC") | (v == "NC")
    agree = np.isin(v, list(STRICT))
    strict_hit = np.array([t in STRICT[b] for b, t in zip(v[agree], tract[agree])], bool)
    af = tract == "AF"
    idx = np.flatnonzero(keep)
    af_pairs = [pair[i] for i in idx[af & (v == "IC")]]
    return {"labeled": int(keep.sum()),
            "valid with a counterpart: labeled as it": round(float(strict_hit.mean()), 4) if agree.any() else None,
            "valid with a counterpart": int(agree.sum()),
            "invalid (IC + NC), labeled": int(inv.sum()), "invalid: top labels": top(tract[inv], 10),
            "false tracts (AF, EC, EmC, IOFF, MdLF, Sup-*): share of labeled": round(float(np.isin(tract, FALSE_TRACTS).mean()), 4),
            "AF": {"labeled": int(af.sum()), "share of labeled": round(float(af.mean()), 4),
                   "share of invalid": round(float((af & inv).sum() / max(inv.sum(), 1)), 4),
                   "verdicts": {k: int(c) for k, c in zip(*np.unique(v[af], return_counts=True))},
                   "IC endpoint pairs (top)": dict(sorted({p: af_pairs.count(p) for p in set(af_pairs)}.items(),
                                                          key=lambda kv: -kv[1])[:8])},
            "by bundle": by_bundle, "by tract": by_tract}


def analyze(args):
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res["published"] = PUBLISHED
    res["scoring"] = {"scilpy": "1.5.0", "segmentation": "scil_score_tractogram --compute_ic --unique --no_empty "
                      "(config_file_segmentation.json, 26 bundles)", "merge": {k: list(v) for k, v in MERGED.items()},
                      "final": "scil_score_bundles (config_file_tractometry.json, 21 bundles) on segmented_VB/ and IS.trk",
                      "labeled": "the same on the labeled streamlines (>= 40 mm) only",
                      "mapping": {"strict": {k: list(v) for k, v in STRICT.items()}, "loose": {k: list(v) for k, v in LOOSE.items()},
                                  "no counterpart": list(NO_COUNTERPART), "false by construction": list(FALSE_TRACTS)}}
    configs = res.setdefault("configs", {})
    for config in (args.config or CONFIGS):
        out = OUT / config
        if not (out / "score" / "labeled" / "results.json").exists():
            print(config, "not scored, skipped")
            continue
        log = json.loads((out / "runs.json").read_text())
        lab = np.load(out / "labels.npz")
        sc = out / "score"
        seg_res = json.loads((sc / "segmentation" / "results.json").read_text())
        vd, pair = verdicts(out)
        everything = np.ones(len(vd), bool)
        configs[config] = {
            "run": {k: log[k] for k in ("config", "fibers", "labeled", "mask_voxels", "residual_left", "warnings",
                                        "field against fmap_Hz", "seconds", "total_s") if k in log}
                   | ({"denoise": log["denoise"]} if "denoise" in log else {}),
            "scoring_seconds": json.loads((sc / "seconds.json").read_text()),
            "all streamlines": headline(json.loads((sc / "final" / "results.json").read_text()), vd, everything)
                               | {"IB": seg_res.get("IB"), "IC": seg_res.get("IC"), "NC": seg_res.get("NC")},
            "labeled streamlines": headline(json.loads((sc / "labeled" / "results.json").read_text()), vd, lab["keep"])
                                   | {"IB": len({pair[i] for i in np.flatnonzero(lab["keep"] & (vd == "IC"))})},
            "labels against the truth": labels_against_truth(vd, pair, lab)}
        print(config, json.dumps({k: configs[config]["all streamlines"][k] for k in
                                  ("VB", "VS_ratio", "IC_ratio", "NC_ratio", "IB", "mean_f1_all21")}), flush=True)
    RESULTS.write_text(json.dumps(res, indent=1))


def run(args):
    configs = CONFIGS if args.config == ["all"] else args.config
    for config in configs:
        parse(config)                                                      # every name checked before any work
    for config in configs:
        out = OUT / config
        if (out / "record.npz").exists():
            print(config, "tracked already: record.npz exists, not tracked again")
        else:
            track(config, out)
        if free_gb(out) < MIN_FREE_GB:
            raise SystemExit(f"stopping: {free_gb(out):.1f} GB free, below {MIN_FREE_GB} GB")
        print(config, "scoring", json.dumps(score(out)), flush=True)


# ----------------------------------------------------------------------------------------------- Patch2Self here

def patch2self(args):
    from scipy.ndimage import binary_dilation
    from tractline import susceptibility as S
    from tractline import ukf as U
    from _bootstrap import ShellWildBootstrap
    t0 = time.time()
    s = I.load()
    d, st = P.denoise_subject(s, P.Timer(), "patch2self")
    raw, den = s.dwi.astype(np.float64), d.dwi.astype(np.float64)
    h = np.load(OUT / "today" / "field_hz.npy").astype(np.float64)
    cr, cd = (S.apply(x, h, s.pe_axis, s.pe_sign, s.readout_s).astype(np.float64) for x in (raw, den))
    t = P.prepare(cr, s.affine, s.bval, s.bvec, shell=I.SHELL, device=DEVICE)
    bg = ~binary_dilation(t.mask, iterations=BACKGROUND_DILATE)
    dw = s.bval > 50
    rayleigh = lambda x: {"from the mean": round(float(x.mean() / np.sqrt(np.pi / 2)), 2),
                          "from the median": round(float(np.median(x) / np.sqrt(2 * np.log(2))), 2)}
    wb = ShellWildBootstrap(s.dwi, s.bval, s.bvec, extra_b0s=s.b0s[..., 1:], b0_noise="added", mask=t.mask, flip="smooth", fwhm=4)
    out = {"noise": {"background voxels": int(bg.sum()), "sigma, background DW volumes (Rayleigh)": rayleigh(raw[bg][:, dw]),
                     "sigma, background b0 (Rayleigh)": rayleigh(raw[bg][:, 0]),
                     "bootstrap noise check, tracking mask": wb.noise_check(t.mask)},
           "denoise": st}
    idx, gdir, coh, share, nseg = gt_directions(s.affine, s.dwi.shape[:3])
    sel = (share > 0.95) & (coh > 0.9) & (nseg > 20)
    at = tuple(idx[sel].T)
    sigma = out["noise"]["sigma, background DW volumes (Rayleigh)"]["from the median"]
    wm = {"voxels": int(sel.sum())}
    for name, x in (("raw", cr), ("Patch2Self", cd)):
        e1, fa = tensor_e1(x, s, idx[sel])
        ang = np.degrees(np.arccos(np.clip(np.abs((e1 * gdir[sel]).sum(1)), 0, 1)))
        v = x[at][:, dw]
        spread = np.median(v.std(1))
        wm[name] = {"b0 median": round(float(np.median(x[at][:, 0])), 1), "DW mean": round(float(v.mean()), 2),
                    "DW SNR (DW mean / sigma)": round(float(v.mean() / sigma), 2),
                    "spread across directions, median SD": round(float(spread), 2), "FA median": round(float(np.median(fa)), 3),
                    "FA > 0.18 (the seeding threshold's FA)": round(float((fa > 0.18).mean()), 3),
                    "angle to the bundle, median deg": round(float(np.median(ang)), 1)}
    wm["noise-free spread estimate, sqrt(raw^2 - sigma^2)"] = round(float(np.sqrt(max(wm["raw"]["spread across directions, median SD"] ** 2
                                                                                     - sigma ** 2, 0))), 2)
    rem = (raw - den)[at][:, dw]
    dev = raw[at][:, dw] - raw[at][:, dw].mean(1, keepdims=True)
    wm["removed"] = {"RMS": round(float(np.sqrt((rem ** 2).mean())), 2), "mean": round(float(rem.mean()), 3),
                     "correlation with the raw deviation from the voxel's DW mean": round(float(np.corrcoef(rem.ravel(), dev.ravel())[0, 1]), 3)}
    out["ground-truth bundle voxels"] = wm
    seeds = {}
    for name, x in (("raw", cr), ("Patch2Self", cd)):
        tt = P.prepare(x, s.affine, s.bval, s.bvec, shell=I.SHELL, device=DEVICE)
        seeds[name] = int(len(U.seeds(U.from_arrays(tt.dwi, tt.header, tt.mask))[0]))
    out["seeds"] = seeds
    for c in ("today", "denoise=patch2self"):
        lab = np.load(OUT / c / "labels.npz")
        out.setdefault("streamlines", {})[c] = {"fibers": int(len(lab["keep"])), "labeled": int(lab["keep"].sum()),
                                                "labeled length, median mm": round(float(np.median(lab["length_mm"])), 1)}
    out["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=1))
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res["patch2self"] = out
    RESULTS.write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------------------------- the ensemble's envelopes

def gt_masks(names):
    """The ground-truth bundles' masks on the t1 grid, as the scorer makes them (scil_compute_streamlines_density_map
    --binary of config_file_tractometry.json's .trk), cached under OUT/gt_masks/: {name: flat bool}."""
    cfg = json.loads((I.SCORING / "config_file_tractometry.json").read_text())
    d = OUT / "gt_masks"
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for b in names:
        p = d / f"{b}.nii.gz"
        if not p.exists():
            scil("scil_compute_streamlines_density_map.py", I.SCORING / cfg[b]["gt_mask"], p, "--binary", cwd=d)
        out[b] = np.asarray(nib.load(p).dataobj).ravel() > 0
    return out


def segment_vb(trk, seg):
    """scil_score_tractogram's valid streamlines only (--unique, no --compute_ic): per streamline, its bundle (the
    sub-bundles merged) or ""."""
    scil("scil_score_tractogram.py", trk, I.SCORING / "config_file_segmentation.json", seg, "--unique", "--no_empty",
         "--json_prefix", "", "--gt_dir", I.SCORING, "--reference", I.REFERENCE, "--no_bbox_check", cwd=seg.parent)
    n = int(nib.streamlines.load(str(trk), lazy_load=True).header["nb_streamlines"])
    v = np.full(n, "", dtype=object)
    part_of = {q: name for name, ps in MERGED.items() for q in ps}
    for p in (seg / "segmented_VB").glob("*_VS.trk"):
        b = p.name[:-len("_VS.trk")]
        v[trk_index(p)] = part_of.get(b, b)
    return v


def replicate_run(s, wb, r, out, labeler):
    """Replicate r: corrected, tracked and labeled as today; record, labels, .trk, verdicts (segment_vb) under out."""
    from dataclasses import replace
    if free_gb(out) < MIN_FREE_GB:
        raise SystemExit(f"stopping: {free_gb(out):.1f} GB free, below {MIN_FREE_GB} GB")
    out.mkdir(parents=True, exist_ok=True)
    t0, timer = time.time(), P.Timer()
    dwi, apa = wb.replicate(r)
    rs = replace(s, dwi=dwi, b0s=np.concatenate([dwi[..., s.b0_index].astype(np.float64), apa], -1))
    with P.exact_float32(DEVICE):
        c = P.correct(rs, timer, DEVICE)
        tg = P.track(rs, c.dwi, timer, device=DEVICE, shell=I.SHELL)
        with timer("label"):
            labels = labeler(tg.fibers, draws=(0,), logp=True)
    P.release_memory(DEVICE)
    for name in ("record.npz", "labels.npz", "tractogram.trk", "verdicts.npz"):
        if (out / name).exists():
            raise SystemExit(f"{out / name} exists; not overwritten")
    log = {"trk": write_trk(out / "tractogram.trk", tg.fibers)}
    np.savez_compressed(out / "labels.npz", keys=E.seed_keys(np.asarray(tg.stats["seed_voxel"])), keep=labels.keep,
                        tract=labels.tract.astype(np.int16), length_mm=labels.length_mm.astype(np.float32))
    seg = out / "segmentation"
    if seg.exists():
        raise SystemExit(f"{seg} exists: an unfinished segmentation; remove it to rerun")
    with timer("segment"):
        v = segment_vb(out / "tractogram.trk", seg)
    np.savez_compressed(out / "verdicts.npz", bundle=v.astype(str))
    shutil.rmtree(seg)                                                   # the verdicts hold what it said
    tmp = out / "record.tmp.npz"
    E.RunRecord.of(tg, labels, s.affine, s.dwi.shape[:3]).save(tmp, compact=True)
    os.replace(tmp, out / "record.npz")                                  # last: its presence marks a finished run
    log.update({"fibers": len(tg.fibers), "labeled": int(labels.keep.sum()), "valid": int((v != "").sum()),
                "residual_left": c.residual_left, "seconds": timer.seconds, "total_s": round(time.time() - t0, 1)})
    (out / "runs.json").write_text(json.dumps(log, indent=1))
    print(f"r{r:03d}", json.dumps({k: log[k] for k in ("fibers", "labeled", "valid", "residual_left", "total_s")}), flush=True)


def presences(fibers, verdict, keep, tract, grid, x_mid):
    """Per strict bundle: (scored presence, labeled presence), flat bool on the grid (module docstring)."""
    from ensemble_grid_check import sample
    names = np.array(TRACT_NAMES)
    side = np.array([np.asarray(f)[:, 0].mean() for f in fibers]) < x_mid                    # RAS: left is lower x
    lab_tract = np.full(len(fibers), "", dtype=object)
    lab_tract[np.flatnonzero(keep)] = names[tract]
    out = {}
    for b, ts in STRICT.items():
        want_left = None if b in ("CC", "MCP") else b.endswith("_left")
        pres = []
        for m in (verdict == b, np.isin(lab_tract, ts) & (True if want_left is None else side == want_left)):
            f = [fibers[i] for i in np.flatnonzero(m)]
            pres.append(grid.density(grid.counts(*sample(f))).ravel() >= RHO_PRESENCE if f else np.zeros(int(np.prod(grid.shape)), bool))
        out[b] = pres
    return out


def shape_scores(votes, R, target, single):
    from ensemble_pooled import thresholds, LEVELS
    n_t = int(target.sum())
    def sc(S):
        n_s, both = int(S.sum()), int((S & target).sum())
        return {"coverage": round(both / n_t, 4), "precision": round(both / max(n_s, 1), 4), "size": round(n_s / n_t, 4),
                "_n": [both, n_s, n_t]}
    out = {"A alone": sc(single), "widest": sc(votes >= 1), "core (at least half)": sc(votes >= int(np.ceil(R / 2)))}
    for L, t in zip(LEVELS, thresholds(votes.astype(np.int64)[votes > 0], R, LEVELS) if (votes > 0).any() else [1] * len(LEVELS)):
        out[f"envelope {L}"] = sc(votes >= t) | {"votes at least": int(t)}
    return out


RHO_PRESENCE = 0.3                           # ensemble_grid_check's RHO


def ensemble(args):
    from _bootstrap import ShellWildBootstrap
    from ensemble_grid_check import Grid
    s = I.load()
    base = OUT / "today"
    if not (base / "record.npz").exists() or not (base / "score" / "segmentation" / "results.json").exists():
        raise SystemExit("today's run and its scoring first: run --config today")
    ens = base / "ensemble"
    reps = list(range(1, (args.first or ENSEMBLE_REPLICATES) + 1))
    todo = [r for r in reps if not (ens / f"r{r:03d}" / "record.npz").exists()]
    if todo:
        if free_gb(ens) < MIN_FREE_GB:
            raise SystemExit(f"stopping: {free_gb(ens):.1f} GB free, below {MIN_FREE_GB} GB")
        with P.exact_float32(DEVICE):
            mask = P.prepare(P.correct(s, P.Timer(), DEVICE).dwi, s.affine, s.bval, s.bvec, shell=I.SHELL, device=DEVICE).mask
        wb = ShellWildBootstrap(s.dwi, s.bval, s.bvec, extra_b0s=s.b0s[..., 1:], b0_noise="added", mask=mask, flip="smooth", fwhm=4)
        ens.mkdir(parents=True, exist_ok=True)
        (ens / "bootstrap.json").write_text(json.dumps({"noise check, tracking mask": wb.noise_check(mask), "flip": "smooth",
                                                        "fwhm": 4, "b0_noise": "added", "one b0 a polarity": "borrowed DW residuals"}, indent=1))
        labeler = P.default_labeler(DEVICE)
        for r in todo:
            replicate_run(s, wb, r, ens / f"r{r:03d}", labeler)
    # A's verdicts: today's segmentation (with --compute_ic; the valid streamlines are the same without it)
    vA, _ = verdicts(base)
    runs = [("A", base / "tractogram.trk", np.where(np.isin(vA, ["IC", "NC"]), "", vA), np.load(base / "labels.npz"))]
    runs += [(f"r{r:03d}", ens / f"r{r:03d}" / "tractogram.trk", np.load(ens / f"r{r:03d}" / "verdicts.npz")["bundle"].astype(object),
              np.load(ens / f"r{r:03d}" / "labels.npz")) for r in reps]
    ref = nib.load(I.REFERENCE)
    grid = Grid("t1", ref.affine, ref.shape[:3])
    x_mid = float((s.affine @ [(s.dwi.shape[0] - 1) / 2, 0, 0, 1])[0])
    truth = gt_masks(list(STRICT))
    R = len(runs)
    votes = {b: [np.zeros(int(np.prod(grid.shape)), np.uint8) for _ in range(2)] for b in STRICT}
    single = {}
    t0 = time.time()
    for name, trk, v, lab in runs:
        fibers = list(nib.streamlines.load(str(trk)).streamlines)
        for b, pres in presences(fibers, v, lab["keep"], lab["tract"], grid, x_mid).items():
            for k in range(2):
                votes[b][k] += pres[k]
            if name == "A":
                single[b] = pres
    res_e = {"runs": R, "replicates": reps[-1], "x_mid": x_mid, "presence": f"density >= {RHO_PRESENCE} per mm^2 (ensemble_grid_check)",
             "per bundle": {}, "pooled": {}}
    for k, kind in enumerate(("scored", "labeled")):
        pooled = {}
        for b in STRICT:
            sc = shape_scores(votes[b][k], R, truth[b], single[b][k])
            res_e["per bundle"].setdefault(b, {})[kind] = {sh: {m: x for m, x in d.items() if m != "_n"} for sh, d in sc.items()}
            for sh, d in sc.items():
                pooled[sh] = [a + c for a, c in zip(pooled.get(sh, [0, 0, 0]), d["_n"])]
        res_e["pooled"][kind] = {sh: {"coverage": round(both / n_t, 4), "precision": round(both / max(n_s, 1), 4),
                                      "size": round(n_s / n_t, 4)} for sh, (both, n_s, n_t) in pooled.items()}
    res_e["seconds, presence"] = round(time.time() - t0, 1)
    print(json.dumps(res_e["pooled"], indent=1))
    res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    res["envelopes"] = res_e
    RESULTS.write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------------------------- rescoring (the review)

SUBSAMPLE_DRAWS = 3


def f1_all21(results):
    names = list(json.loads((I.SCORING / "config_file_tractometry.json").read_text()))
    bw = results["bundle_wise"]
    return float(np.mean([bw.get(b, {}).get("f1", 0.0) or 0.0 for b in names]))


def vb_score(trk, d, unique=True):
    """trk's valid streamlines only (scil_score_tractogram without --compute_ic; with --unique unless unique=False), then
    final(): d/segmentation/ and d/final/results.json; not rerun when that exists. (results, seconds)."""
    if (d / "final" / "results.json").exists():
        return json.loads((d / "final" / "results.json").read_text()), None
    for x in (d / "segmentation", d / "final"):
        if x.exists():
            raise SystemExit(f"{x} exists without final results: an unfinished scoring; remove {d} to rerun")
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scil("scil_score_tractogram.py", trk, I.SCORING / "config_file_segmentation.json", d / "segmentation",
         *(["--unique"] if unique else []), "--no_empty", "--json_prefix", "", "--gt_dir", I.SCORING, "--reference",
         I.REFERENCE, "--no_bbox_check", cwd=d)
    tmp = d / "final.tmp"
    res = final(d / "segmentation", tmp, cwd=d)
    os.replace(tmp, d / "final")
    return res, round(time.time() - t0, 1)


def summary(res):
    names = list(json.loads((I.SCORING / "config_file_tractometry.json").read_text()))
    return {"VB": res["VB"], "VS": res["VS"], "VS_ratio": round(res["VS_ratio"], 4), "mean_f1_all21": round(f1_all21(res), 4),
            "f1": {b: round(res["bundle_wise"].get(b, {}).get("f1", 0.0) or 0.0, 4) for b in names}}


def rescore(args):
    """The review's rescorings (results/robust_phantom.json, "rescore"); each step skipped when its results exist:
      wpc         every configuration segmented again with --compute_ic --unique --save_wpc_separately
                  (score/segmentation_wpc/): wrong-path connections (WPC: a valid bundle's endpoint regions, outside its
                  all_mask) apart from NC, the overlaps between the sets, and whether the valid streamlines change;
      no_unique   today without --unique (and without --compute_ic; score/no_unique/): VB, VS and F1, and how many
                  streamlines fit more than one bundle;
      subsample   today's tractogram subsampled to Patch2Self's streamline count (SUBSAMPLE_DRAWS draws, seeds 0..):
                  F1 at the same count, to tell fewer streamlines from misplaced ones (subsample_<n>/d<k>/);
      replicates  today's 16 bootstrap replicates scored (ensemble/r###/score_vb/): a band for today's F1."""
    res = json.loads(RESULTS.read_text())
    out = res.setdefault("rescore", {})
    seg_args = ["--gt_dir", I.SCORING, "--reference", I.REFERENCE, "--no_bbox_check"]
    wpc = out.setdefault("wpc", {})
    for c in CONFIGS:
        base = OUT / c
        d = base / "score" / "segmentation_wpc"
        if not (d / "results.json").exists():
            if d.exists():
                raise SystemExit(f"{d} exists without results.json; remove it to rerun")
            if free_gb(base) < MIN_FREE_GB:
                raise SystemExit(f"stopping: {free_gb(base):.1f} GB free, below {MIN_FREE_GB} GB")
            scil("scil_score_tractogram.py", base / "tractogram.trk", I.SCORING / "config_file_segmentation.json", d,
                 "--compute_ic", "--unique", "--save_wpc_separately", "--no_empty", "--json_prefix", "", *seg_args, cwd=base / "score")
        n = len(np.load(base / "labels.npz")["keep"])
        ids = lambda paths: np.unique(np.concatenate([trk_index(p) for p in paths])) if paths else np.zeros(0, np.int64)
        vs = ids(sorted((d / "segmented_VB").glob("*.trk")))
        wp = ids(sorted((d / "segmented_WPC").glob("*.trk")))
        ic = ids(sorted((d / "segmented_IB").glob("*.trk")))
        nc = ids([d / "NC.trk"] if (d / "NC.trk").exists() else [])
        old = json.loads((base / "score" / "segmentation" / "results.json").read_text())
        new = json.loads((d / "results.json").read_text())
        same_vs = all(old["bundle_wise"].get(b, {}).get("VS") == new["bundle_wise"].get(b, {}).get("VS")
                      for b in set(old["bundle_wise"]) | set(new["bundle_wise"]) if b != "IB")
        sh = lambda x: round(len(x) / n, 4)
        wpc[c] = {"streamlines": n, "VS": sh(vs), "WPC": sh(wp), "IC": sh(ic), "NC (as written)": sh(nc),
                  "WPC also IC": sh(np.intersect1d(wp, ic)), "WPC also NC": sh(np.intersect1d(wp, nc)),
                  "NC without WPC": sh(np.setdiff1d(nc, wp)), "IC without WPC": sh(np.setdiff1d(ic, wp)),
                  "VS also WPC": sh(np.intersect1d(vs, wp)), "unassigned": sh(np.setdiff1d(np.arange(n), np.concatenate([vs, wp, ic, nc]))),
                  "WPC per bundle": {k: v["WPC"]["Count"] for k, v in new["bundle_wise"].items() if k != "IB" and v.get("WPC")} if any(
                      isinstance(v, dict) and v.get("WPC") for k, v in new["bundle_wise"].items() if k != "IB") else
                      {p.name[:-len("_wpc.trk")]: len(trk_index(p)) for p in sorted((d / "segmented_WPC").glob("*.trk"))},
                  "valid streamlines per bundle unchanged": bool(same_vs), "scilpy's own": {k: new.get(k) for k in
                      ("VS", "WPC", "WPC_bundle", "IC", "NC", "IB", "total_streamlines")}}
        print("wpc", c, json.dumps({k: v for k, v in wpc[c].items() if k not in ("WPC per bundle",)}), flush=True)
        RESULTS.write_text(json.dumps(res, indent=1))
    today = OUT / "today"
    r_nu, secs = vb_score(today / "tractogram.trk", today / "score" / "no_unique", unique=False)
    multi = np.concatenate([trk_index(p) for p in (today / "score" / "no_unique" / "segmentation" / "segmented_VB").glob("*.trk")])
    part_of = {q: name for name, ps in MERGED.items() for q in ps}
    owner = {}
    for p in (today / "score" / "no_unique" / "segmentation" / "segmented_VB").glob("*.trk"):
        b = part_of.get(p.name[:-len("_VS.trk")], p.name[:-len("_VS.trk")])
        for i in trk_index(p).tolist():
            owner.setdefault(i, set()).add(b)
    unique_res = json.loads((today / "score" / "final" / "results.json").read_text())
    out["no_unique"] = {"without --unique": summary(r_nu), "with --unique (final/)": summary(unique_res),
                        "streamlines in more than one sub-bundle": int(len(multi) - len(np.unique(multi))),
                        "streamlines in more than one bundle (merged)": int(sum(len(v) > 1 for v in owner.values()))}
    for k in ("without --unique", "with --unique (final/)"):
        out["no_unique"][k].pop("f1")
    out["no_unique"]["max |F1 difference| over the 21"] = round(max(abs(summary(r_nu)["f1"][b] - summary(unique_res)["f1"][b])
                                                                   for b in summary(r_nu)["f1"]), 4)
    print("no_unique", json.dumps(out["no_unique"]), flush=True)
    RESULTS.write_text(json.dumps(res, indent=1))
    target = len(np.load(OUT / "denoise=patch2self" / "labels.npz")["keep"])
    sub = out.setdefault("subsample", {"to": target, "from": "today", "draws": {}})
    full = None
    for k in range(SUBSAMPLE_DRAWS):
        d = today / f"subsample_{target}" / f"d{k}"
        if not (d / "final" / "results.json").exists():
            if full is None:
                full = nib.streamlines.load(str(today / "tractogram.trk"))
            pick = np.sort(np.random.default_rng(k).choice(len(full.streamlines), target, replace=False))
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "tractogram.trk").exists():
                nib.streamlines.save(full.tractogram[pick], str(d / "tractogram.trk"), header=full.header)
        sub["draws"][f"d{k}"] = {kk: v for kk, v in summary(vb_score(d / "tractogram.trk", d)[0]).items() if kk != "f1"}
        print("subsample", k, json.dumps(sub["draws"][f"d{k}"]), flush=True)
        RESULTS.write_text(json.dumps(res, indent=1))
    f1s = [v["mean_f1_all21"] for v in sub["draws"].values()]
    sub["mean_f1_all21: mean, min, max"] = [round(float(np.mean(f1s)), 4), min(f1s), max(f1s)]
    reps = out.setdefault("replicates", {"runs": {}})
    for r in range(1, ENSEMBLE_REPLICATES + 1):
        d = today / "ensemble" / f"r{r:03d}"
        reps["runs"][f"r{r:03d}"] = {k: v for k, v in summary(vb_score(d / "tractogram.trk", d / "score_vb")[0]).items() if k != "f1"}
        print("replicate", r, json.dumps(reps["runs"][f"r{r:03d}"]), flush=True)
        RESULTS.write_text(json.dumps(res, indent=1))
    f = np.array([v["mean_f1_all21"] for v in reps["runs"].values()])
    vb = [v["VB"] for v in reps["runs"].values()]
    reps["mean_f1_all21"] = {"mean": round(float(f.mean()), 4), "sd": round(float(f.std(ddof=1)), 4), "min": round(float(f.min()), 4),
                             "max": round(float(f.max()), 4), "A (today's run)": round(f1_all21(unique_res), 4)}
    reps["VB: min, max"] = [min(vb), max(vb)]
    reps["VS_ratio: min, max"] = [min(v["VS_ratio"] for v in reps["runs"].values()), max(v["VS_ratio"] for v in reps["runs"].values())]
    print("replicates", json.dumps({k: v for k, v in reps.items() if k != "runs"}), flush=True)
    RESULTS.write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------------------------- envelopes, count-matched

TARGET_DRAWS = 3                             # ground-truth subsamples per run and bundle
KINDS = ("labeled", "scored, >= 40 mm", "scored")


def density_presence(grid, fibers):
    """ensemble_grid_check's density presence (flat bool), the smoothing done on the counts' bounding box padded by the
    Gaussian's reach (scipy truncates at 4 SD): the same as on the whole grid, far faster for one bundle."""
    from scipy.ndimage import gaussian_filter
    from ensemble_grid_check import sample, SMOOTH_MM
    out = np.zeros(grid.shape, bool)
    if not len(fibers):
        return out.ravel()
    c = grid.counts(*sample(fibers)).reshape(grid.shape)
    nz = np.argwhere(c)
    sd = SMOOTH_MM / grid.h
    pad = np.ceil(4 * sd).astype(int) + 1
    lo, hi = np.maximum(nz.min(0) - pad, 0), np.minimum(nz.max(0) + pad + 1, grid.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    out[box] = gaussian_filter((c[box] / grid.area).astype(np.float32), sd) >= RHO_PRESENCE
    return out.ravel()


def envelopes(args):
    """The ensemble's envelopes against a like-for-like truth (the review; results/robust_phantom.json,
    "envelopes, count-matched"). ensemble's target, the ground-truth bundle's traversal mask, is not like for like: a
    run's density presence of a few hundred streamlines cannot fill a mask every ground-truth streamline marks. Here the
    target is the ground truth's own density presence at the run's streamline count: for each run and bundle, the
    ground-truth bundle's streamlines (bundles/*.trk) subsampled to the run's count for that kind (TARGET_DRAWS draws),
    presence by the same density rule. Coverage and precision of each shape are pooled over runs, draws and bundles
    (weighted by the target's voxels). Kinds of presence:
      labeled            the labeled streamlines with the bundle's strict counterpart, on its side (what the product uses);
      scored, >= 40 mm   the scorer's valid streamlines of the bundle, only the labeled lengths (the scorer accepts a
                         valid streamline only inside the bundle's all_mask, drawn from the truth: truth-assisted);
      scored             the same, every length.
    Baselines, the same rule on the truth alone: self, one ground-truth subsample against another at the same count;
    perfect ensemble, 17 ground-truth subsamples (one per run, at its count) in place of the runs, its shapes against
    the targets: what the shapes give with no bias, only sparsity. Also each shape against the traversal mask."""
    from ensemble_grid_check import Grid
    from ensemble_pooled import thresholds, LEVELS
    t0 = time.time()
    s = I.load()
    base, ens = OUT / "today", OUT / "today" / "ensemble"
    reps = sorted(int(p.name[1:]) for p in ens.glob("r[0-9][0-9][0-9]") if (p / "verdicts.npz").exists())
    vA, _ = verdicts(base)
    runs = [(base / "tractogram.trk", np.where(np.isin(vA, ["IC", "NC"]), "", vA), np.load(base / "labels.npz"))]
    runs += [(ens / f"r{r:03d}" / "tractogram.trk", np.load(ens / f"r{r:03d}" / "verdicts.npz")["bundle"].astype(object),
              np.load(ens / f"r{r:03d}" / "labels.npz")) for r in reps]
    R = len(runs)
    ref = nib.load(I.REFERENCE)
    grid = Grid("t1", ref.affine, ref.shape[:3])
    x_mid = float((s.affine @ [(s.dwi.shape[0] - 1) / 2, 0, 0, 1])[0])
    names = np.array(TRACT_NAMES)
    bundles = list(STRICT)
    nvox = int(np.prod(grid.shape))
    votes = {(b, k): np.zeros(nvox, np.uint8) for b in bundles for k in KINDS}
    single, count = {}, {}
    for r, (trk, v, lab) in enumerate(runs):
        fibers = list(nib.streamlines.load(str(trk)).streamlines)
        keep = lab["keep"]
        left = np.array([f[:, 0].mean() for f in fibers]) < x_mid
        lt = np.full(len(fibers), "", dtype=object)
        lt[np.flatnonzero(keep)] = names[lab["tract"]]
        for b in bundles:
            side = np.ones(len(fibers), bool) if b in ("CC", "MCP") else left == b.endswith("_left")
            for k, m in zip(KINDS, (np.isin(lt, STRICT[b]) & side, (v == b) & keep, v == b)):
                pres = density_presence(grid, [fibers[i] for i in np.flatnonzero(m)])
                votes[b, k] += pres
                count[r, b, k] = int(m.sum())
                if r == 0:
                    single[b, k] = pres
        print("run", r, round(time.time() - t0, 1), flush=True)
    from ensemble_grid_check import sample
    cc = [f for f, ok in zip(nib.streamlines.load(str(runs[0][0])).streamlines, runs[0][1] == "CC") if ok]
    check = grid.density(grid.counts(*sample(cc))).ravel() >= RHO_PRESENCE
    assert (check == single["CC", "scored"]).all(), "the boxed density differs from the whole grid's"
    gt = {b: list(nib.streamlines.load(str(I.SCORING / "bundles" / f"{b}.trk")).streamlines) for b in bundles}
    trav = gt_masks(bundles)
    draw = lambda b, k, r, d, n: density_presence(grid, [gt[b][i] for i in np.random.default_rng(
        [bundles.index(b), KINDS.index(k), r, d]).choice(len(gt[b]), min(n, len(gt[b])), replace=False)])
    shape_names = ["A alone", "core (at least half)", "widest"] + [f"envelope {L}" for L in LEVELS]
    acc = {(k, sh, w): np.zeros(3) for k in KINDS for sh in shape_names for w in ("runs", "perfect ensemble", "traversal mask")}
    self_acc = {k: np.zeros(3) for k in KINDS}
    per_bundle = {}
    level_votes = {}
    for b in bundles:
        for k in KINDS:
            targets = [[draw(b, k, r, d, count[r, b, k]) for d in range(TARGET_DRAWS)] for r in range(R)]
            gv = np.zeros(nvox, np.uint8)
            for r in range(R):
                gv += draw(b, k, r, 100, count[r, b, k])
            for T in targets:                                              # self: draw 0 against draws 1.. at one count
                for T1 in T[1:]:
                    self_acc[k] += [(T[0] & T1).sum(), T[0].sum(), T1.sum()]
            for who, vv, one in (("runs", votes[b, k], single[b, k]), ("perfect ensemble", gv, None)):
                q = vv.astype(np.int64)
                th = thresholds(q[q > 0], R, LEVELS) if (q > 0).any() else [1] * len(LEVELS)
                if who == "runs":
                    level_votes[b, k] = dict(zip(LEVELS, th))
                shapes = {"core (at least half)": q >= int(np.ceil(R / 2)), "widest": q >= 1,
                          **{f"envelope {L}": q >= t for L, t in zip(LEVELS, th)}}
                if one is not None:
                    shapes["A alone"] = one
                for sh, S in shapes.items():
                    nS = int(S.sum())
                    for T in (x for row in targets for x in row):
                        a = acc[k, sh, who]
                        a += [(S & T).sum(), nS, T.sum()]
                        pb = per_bundle.setdefault(b, {}).setdefault(k, {}).setdefault(sh, {}).setdefault(who, np.zeros(3))
                        pb += [(S & T).sum(), nS, T.sum()]
                    if who == "runs":
                        acc[k, sh, "traversal mask"] += [(S & trav[b]).sum(), nS, trav[b].sum()]
            print(b, k, round(time.time() - t0, 1), flush=True)
    fmt = lambda a: {"coverage": round(float(a[0] / max(a[2], 1)), 4), "precision": round(float(a[0] / max(a[1], 1)), 4),
                     "size": round(float(a[1] / max(a[2], 1)), 4)}
    out = {"runs": R, "target draws": TARGET_DRAWS, "kinds": list(KINDS), "levels": list(LEVELS),
           "counts, mean over runs": {b: {k: round(float(np.mean([count[r, b, k] for r in range(R)])), 1) for k in KINDS} for b in bundles},
           "ground-truth streamlines": {b: len(gt[b]) for b in bundles},
           "self (one subsample against another)": {k: fmt(self_acc[k]) for k in KINDS},
           "pooled": {k: {who: {sh: fmt(acc[k, sh, who]) for sh in shape_names if acc[k, sh, who][2] > 0}
                          for who in ("runs", "perfect ensemble", "traversal mask")} for k in KINDS},
           "per bundle": {b: {k: {sh: {who: fmt(a) for who, a in d.items()} for sh, d in v.items()} for k, v in kk.items()}
                          for b, kk in per_bundle.items()},
           "votes kept at each level": {b: {k: {str(L): int(t) for L, t in level_votes[b, k].items()} for k in KINDS} for b in bundles},
           "seconds": round(time.time() - t0, 1)}
    print(json.dumps(out["pooled"], indent=1))
    print(json.dumps(out["self (one subsample against another)"]))
    res = json.loads(RESULTS.read_text())
    res["envelopes, count-matched"] = out
    RESULTS.write_text(json.dumps(res, indent=1))


# ----------------------------------------------------------------------------------------------- the review's checks

#: the volumes the challenge lists as moved ("motion in volumes 6, 12 and 24"), both readings (0- and 1-based) of the
#: 33-volume series, which is the DWI here (volume 0 the b0)
MOTION_VOLUMES = (5, 6, 11, 12, 23, 24)


def review_checks(args):
    """results/robust_phantom.json, "review checks":
      readout scale  today's field against fmap_Hz smoothed by Gaussians of 0-3 voxels (SD), in the tracking mask: if the
                     estimate is a smoothed field at the right scale, r peaks at the estimator's smoothing and the slope
                     there is the scale;
      patch2self     patch2self()'s bundle-voxel measures without the moved volumes (MOTION_VOLUMES, both readings) and
                     with robust statistics (1.4826 x the median absolute deviation): what Patch2Self removed, against the
                     in-brain noise (the bootstrap's leverage-scaled residuals about the shell's fit, same volumes and
                     voxels) and the background's; the spread across directions; FA from the kept volumes; and each
                     volume's removed robust SD, to see whether the moved volumes stand out."""
    from dataclasses import replace
    from scipy.ndimage import gaussian_filter
    from tractline import susceptibility as S
    from _bootstrap import ShellWildBootstrap
    s = I.load()
    out = {}
    h = np.load(OUT / "today" / "field_hz.npy").astype(np.float64)
    f = I.fmap_hz()
    with P.exact_float32(DEVICE):
        cr = S.apply(s.dwi.astype(np.float64), h, s.pe_axis, s.pe_sign, s.readout_s).astype(np.float64)
    mask = P.prepare(cr, s.affine, s.bval, s.bvec, shell=I.SHELL, device=DEVICE).mask
    rows = {}
    for sd in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        g = gaussian_filter(f, sd) if sd else f
        x, y = g[mask], h[mask]
        rows[f"{sd:g} voxels"] = {"r": round(float(np.corrcoef(x, y)[0, 1]), 4), "slope": round(float(x @ y / (x @ x)), 4),
                                  "sd_ratio": round(float(y.std() / x.std()), 4)}
    out["readout scale: the field against fmap_Hz smoothed (Gaussian SD), tracking mask"] = rows
    d, _ = P.denoise_subject(s, P.Timer(), "patch2self")
    raw, den = s.dwi.astype(np.float64), d.dwi.astype(np.float64)
    cd = S.apply(den, h, s.pe_axis, s.pe_sign, s.readout_s).astype(np.float64)
    idx, gdir, coh, share, nseg = gt_directions(s.affine, s.dwi.shape[:3])
    sel = (share > 0.95) & (coh > 0.9) & (nseg > 20)
    at = tuple(idx[sel].T)
    dw = np.flatnonzero(s.bval > 50)
    kept = np.setdiff1d(dw, MOTION_VOLUMES)
    mad = lambda x, axis=None: 1.4826 * np.median(np.abs(x - np.median(x, axis=axis, keepdims=axis is not None)), axis=axis)
    rem = (raw - den)[at]
    wb = ShellWildBootstrap(s.dwi, s.bval, s.bvec, extra_b0s=s.b0s[..., 1:], b0_noise="added", mask=mask, flip="smooth", fwhm=4)
    res = wb.res.reshape(*s.dwi.shape)[at]
    bg_sigma = json.loads(RESULTS.read_text())["patch2self"]["noise"]["sigma, background DW volumes (Rayleigh)"]["from the median"]
    per_vol = {int(v): round(float(mad(rem[:, v])), 2) for v in dw}
    pv = {"removed robust SD per DW volume": per_vol,
          "moved volumes' removed robust SD": {int(v): per_vol[int(v)] for v in MOTION_VOLUMES},
          "the others' median": round(float(np.median([per_vol[int(v)] for v in kept])), 2)}
    m = {}
    for name, vols in (("all 32 DW volumes", dw), ("without the moved volumes", kept)):
        sub = replace(s, bval=s.bval[np.r_[0, vols]], bvec=s.bvec[:, np.r_[0, vols]])
        r = {"volumes": int(len(vols)), "removed RMS": round(float(np.sqrt((rem[:, vols] ** 2).mean())), 2),
             "removed robust SD (MAD)": round(float(mad(rem[:, vols])), 2),
             "in-brain noise: bootstrap residual RMS": round(float(np.sqrt((res[:, vols] ** 2).mean())), 2),
             "in-brain noise: bootstrap residual robust SD (MAD)": round(float(mad(res[:, vols])), 2),
             "background noise sigma (Rayleigh, median)": bg_sigma}
        for nm, x in (("raw", cr), ("Patch2Self", cd)):
            v = x[at][:, vols]
            _, fa = tensor_e1(x[..., np.r_[0, vols]], sub, idx[sel])
            r[nm] = {"spread across directions, median SD": round(float(np.median(v.std(1))), 2),
                     "spread across directions, median robust SD (MAD)": round(float(np.median(mad(v, axis=1))), 2),
                     "FA median": round(float(np.median(fa)), 3), "FA > 0.18": round(float((fa > 0.18).mean()), 3)}
        sig = r["in-brain noise: bootstrap residual robust SD (MAD)"]
        r["noise-free spread estimate, sqrt(raw SD^2 - in-brain robust noise^2)"] = round(float(np.sqrt(max(
            r["raw"]["spread across directions, median SD"] ** 2 - sig ** 2, 0))), 2)
        rr = (raw - den)[at][:, vols]
        dev = raw[at][:, vols] - raw[at][:, vols].mean(1, keepdims=True)
        r["removed: correlation with the raw deviation from the voxel's DW mean"] = round(float(np.corrcoef(rr.ravel(), dev.ravel())[0, 1]), 3)
        m[name] = r
    out["patch2self, bundle voxels"] = {"voxels": int(sel.sum()), **m, "per volume": pv,
                                        "dipy": "each DW volume regressed on the other 31 DW volumes; the single b0 is not denoised"}
    print(json.dumps(out, indent=1))
    res_all = json.loads(RESULTS.read_text())
    res_all["review checks"] = out
    RESULTS.write_text(json.dumps(res_all, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("check", "run", "analyze", "patch2self", "ensemble", "rescore", "envelopes", "review_checks"))
    ap.add_argument("--config", action="append", help=f"run (required; 'all': {', '.join(CONFIGS)}), analyze (default: all)")
    ap.add_argument("--first", type=int, help=f"ensemble: only replicates 1..N (a smoke test; default {ENSEMBLE_REPLICATES})")
    args = ap.parse_args()
    if args.what == "run" and not args.config:
        ap.error("run needs --config")
    {"check": check, "run": run, "analyze": analyze, "patch2self": patch2self, "ensemble": ensemble,
     "rescore": rescore, "envelopes": envelopes,
     "review_checks": review_checks}[args.what](args)
