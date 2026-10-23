"""MASiVar cohort II across vendors: does one Philips scan's 17-run ensemble envelope cover the same person's scan on a
Siemens scanner, as stated, and with a recalibration map for that term (fitted leaving one subject out)? And does
the same-scanner (other-session) map carry over to the other vendor? (bench/calibration_cohortII.py: the scanner A
and B records this reads; docs/ismrm2027-abstract.md.)

    uv run --group dipy --group rankfield bench/calibration_crosssite.py check
    uv run --group dipy --group rankfield bench/calibration_crosssite.py run [--source B2000] [--subject cIIs01 ...] [--registration 2]
    uv run --group dipy --group rankfield bench/calibration_crosssite.py collect [--source B2000] [--subject ...] [--force] [--registration 2]
    uv run --group dipy --group rankfield bench/calibration_crosssite.py evaluate [--sources B2000 [A2000r19]] [--registration 2] [--out ...]

The rescans' registration mode (calibration_cohortII.REGISTRATION; --registration, default 2): 1 is
ensemble_retest.register (center of mass, translation, rigid on the mean b0s; every record before 2026-10-19 evening:
site3_onA.npz, "site3_scan", statements_R17.npz); 2 is FA onto FA, then a rigid refinement of the mean b0s
(calibration_cohortII.register_fa): site3reg2_onA.npz, "site3reg2_scan", statements_R17_reg2.npz, beside mode 1's,
never over them; 3 is multi-start (calibration_cohortII.register_multi; site 3 has no other direction, so starts (a) and
(b)): site3reg3_scan, and site3reg3_onA (statements_R17_reg3.npz) only where it differs from mode 2, else mode 2's
record and statements are used, the transform confirmed (calibration_cohortII.reg3_entry, statements_file). A mode
reads the source protocol's records of the same mode (its session2reg2 rescan and
statements_R17_reg2.npz), so the cross-vendor and same-scanner pairs are registered alike.

The data (bench/fetch_ds003416.py, SITE3): cIIs01-04's site 3 session (s3x1; Siemens Skyra 3T), the b2000n96r25x25x25
APP series and the b1000n3r25x25x25 APA series nearest it in acquisition number (ties: the later): APA 108 for
cIIs01 (b2000 106; 104 and 108 tie), 105 for cIIs02 (b2000 106), 108 for cIIs03 and cIIs04 (b2000 107). cIIs00 has
no site 3 APA series and is not used. Cohort II has no sidecars; what the NIfTI headers say (dcm2niix's descrip):
  - the APP series: 96 x 96 x 50 at 2.5 mm, 98 volumes, two b0s FIRST (volumes 0 and 1, stated b = 0, gradient 0),
    then 96 directions at stated b 1980-2005 (Siemens writes each direction's own b; the tracker's shell selection,
    |b - 2000| < 50, takes them all, and prep.prepare scales each gradient by sqrt(b / b_max)); uint16, slope 1;
    TE 95 ms ("TE=95"), "phase=1"; the 4th pixdim 6.35 (TR in s, as dcm2niix writes it);
  - the APA series: 4 volumes, the b0 FIRST (volume 0), then three b1000 volumes whose b-vectors are zero (only the
    b0 is used); TE 95 ms, "phase=0": the other phase polarity, as the names say (APP j, APA j-, calibration_cohortII.PE);
  - scanner B (s1B, Philips Achieva) for comparison: 96 x 96 x 48, 97 volumes, its one b0 LAST, b exactly 2000,
    int16 with a scale per series, TE 90 ms, pixdim 5.9; its reversed b0 0.71-0.81 of the forward b0s'.
The protocol S3_2000 (PROTOCOL_S3, a strict descriptor as calibration_cohortII.PROTOCOLS): its `check` items are
calibration_cohortII.check's (volumes, per shell, matrix, voxel, phase encoding, one reversed b0 and its brightness,
the reversed series' voxel, the site and scanner by the BIDS session label: "s3x") plus the forward b0s' positions,
the stated b range of the shell, the echo time and the phase polarity from the NIfTI descrip. It refuses every
cohort II protocol's scans, cohort III's and cIs1's, and they refuse it (`check`; evaluate's "refusals").
load_s3: the b0s by stated b (< 50, zero gradient), each checked to be a b0 (B0_OVER_DW times the series' DW
volumes' brain median) and no DW volume b0-like (below B0_LIKE of the b0s): a mislabeled or derived volume fails;
the DW b-vectors unit length; the reversed b0 by its stated b and B0_OVER_DW over its own DW volumes.

The rescan (run; per subject and source protocol): the site 3 scan corrected on its own grid (the field from its own
reversed b0), its mean b0 registered rigidly onto the source protocol's A (A_meanb0.npy; ensemble_retest.register,
dipy mutual information), resampled onto A's grid with its b-vectors turned (onto_reference), and tracked there at
the source protocol's shell with today's default pipeline: calibration_cohortII.Runner.rescan's steps, as the module
makes session2_onA. Before tracking, the b-vector check (bvec_check): a tensor fit (log-linear least squares) of the
data the tracker gets (prep.prepare's gradients, RAS), on A's grid:
  - the principal direction in the corpus callosum body (A's CC4 + CC5 streamline counts >= 5, within 5 mm of their
    median world x: the midline) should run left-right, and in the CST below the callosum superior-inferior;
  - its agreement (|cos|) with A's streamline tangents (A's run: the Philips scan's tracking), voxels where those are
    coherent, for the b-vectors as stated and with each axis flipped (a flipped axis should agree worse);
  - the same for A's own scan (the Philips reference) and FA's correlation between the rescan and A in A's mask;
  - both again for the same-scanner rescan (session2, corrected again and put on A's grid by the transform its
    records hold), the baseline for the cross-vendor numbers (logged here; the source's records are not written).
Records: DATA/ensemble/<subject>/card/<subject>-<source>-site3/ (site3_onA.npz, runs.json); A, its companions and its
replicates are read from the source protocol's folder (calibration_cohortII.Records: reused, never written).

collect: calibration_cohortII.collect's statements (A's 17 runs against the site 3 rescan, term "cross vendor (site
3)"), statements_R17.npz in the same folder; A's arrays are checked equal to the source's own statements_R17.npz.

evaluate (no pipeline runs): per source protocol (B2000; A2000r19 once the scanner A queue has made its records) and
subject, envelope coverage as calibration_card.coverage_curve (the share of the rescan's density presence inside the
envelope, the 42 named tracts pooled, weighted by the rescan's voxels), at stated levels 0.5-0.95:
  (i)   the cross-vendor rescan as stated;
  (ii)  a cross-vendor map, leave one subject out: fitted (calibration_card.fit_term's envelope curve) on the other
        subjects' cross-vendor pairs, applied to the held-out subject: its coverage at the map's level for each target
        (the widest outline where the map never reaches it);
  (iii) the same subject's same-scanner other-session rescan as stated and with its LOSO map (calibration_cohortII's
        statements, as its evaluate reads them: coverage_at_targets);
  (iv)  the same-scanner LOSO map (fitted on the other subjects' other-session rescans) applied to the cross-vendor pair.
Streamline and point ECE beside them (diagnostic). Writes results/calibration_crosssite.json with provenance (git, this
script's and calibration_cohortII.py's SHA-256, every input record's), and "fig6 (COHORT2_SCHEMA shape)": the rows
ismrm2027_figures.fig6_protocol_gap reads, under protocol name S3_2000 ("B2000 -> S3_2000" a transfer entry).

Imports calibration_cohortII (no side effects on import: constants, imports and definitions only) and never writes it.
"""
import argparse, json, re, subprocess, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tractline import pipeline as P
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import TRACT_NAMES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import calibration_cohortII as CII   # noqa: E402
import calibration_runs as CR   # noqa: E402
import calibration_card as CC   # noqa: E402
import calibration_transfer as CT   # noqa: E402
import _masivar as M   # noqa: E402
from ensemble_retest import register, onto_reference, mean_b0  # noqa: E402

SUBJECTS = ("cIIs01", "cIIs02", "cIIs03", "cIIs04")
SESSION = "s3x1"
KEY = "S3_2000"
TAG = "site3"
TERM_X = "cross vendor (site 3)"
SOURCES = ("B2000", "A2000r19")
RUNS = CR.RUNS
LEVELS = CII.SCHEMA_LEVELS
TABLE = (0.8, 0.9, 0.95)
OUT = HERE / "results/calibration_crosssite.json"
B0_LIKE = M.B0_LIKE                     # a DW volume at least this bright against the b0s would be b0-like: refused

PROTOCOL_S3 = {
    "scanner": "s3x", "vendor": "Siemens Skyra 3T (MASiVar site 3)", "series": ["b2000n96"],
    "reversed series rescaled to the forward b0s": False,
    "series resolution (BIDS acq r...)": "25x25x25", "reversed series resolution": "25x25x25",
    "resolution": "2.5 x 2.5 x 2.5 mm (r25x25x25); reversed series 2.5 x 2.5 x 2.5 mm (r25x25x25)",
    "reversed series": "the r25x25x25 APA series (b1000n3) nearest in acquisition number to b2000n96 (ties: the later)",
    "tracked shell": 2000.0,
    "note": "site 3 (Siemens Skyra), b2000n96 alone, 2.5 mm, tracked at 2000: B2000's nominal protocol on another vendor",
    "check": {"site and scanner (BIDS session label)": "s3x", "volumes": 98, "volumes per shell": {"0": 2, "2000": 96},
              "shell_round": 100, "matrix": [96, 96, 50], "voxel_mm": [2.5, 2.5, 2.5], "voxel_tolerance_mm": 0.01,
              "phase_encoding": ["j", "j-"], "reversed_b0s": 1,
              "reversed_b0_over_forward": [0.85, 1.15],          # seen 0.935-0.967 (4 scans; same TE, 95 ms, both)
              "reversed series voxel_mm": [2.5, 2.5, 2.5],
              "forward b0s at (volume index)": [0, 1],
              "stated b of the 2000 shell (min, max)": [1975, 2010],
              "echo time ms (NIfTI descrip), forward and reversed": [95, 95],
              "phase polarity (NIfTI descrip), forward and reversed": [1, 0]}}


# ----------------------------------------------------------------------------------------------- the scans

def descrip(img):
    """dcm2niix's NIfTI descrip ("TE=95;Time=125442.435;phase=1") as a dict of numbers."""
    d = img.header["descrip"].tobytes().rstrip(b"\0").decode(errors="replace")
    out = {}
    for part in d.split(";"):
        k, _, v = part.partition("=")
        try:
            out[k.strip()] = float(v)
        except ValueError:
            pass
    return out


def series_files(subject):
    """{"APP": path, "APA": path, ...}: site 3's b2000n96 APP series and the r25x25x25 APA series nearest it."""
    d = M.ROOT / f"sub-{subject}/ses-{SESSION}/dwi"
    app, apa = [], []
    for p in sorted(d.glob(f"sub-{subject}_ses-{SESSION}_acq-*_dwi.nii.gz")):
        m = CII.NAME.search(p.name)
        if not m or m.group(3) != PROTOCOL_S3["series resolution (BIDS acq r...)"]:
            continue
        run = int(m.group(5) + m.group(6))
        if m.group(4) == "APA":
            apa.append((run, p))
        elif f"b{m.group(1)}n{m.group(2)}" in PROTOCOL_S3["series"]:
            app.append((run, p))
    if len(app) != 1 or not apa:
        raise FileNotFoundError(f"{subject} {SESSION}: {len(app)} b2000n96 APP series, {len(apa)} APA series on disk")
    anchor = app[0][0]
    run, path = min(apa, key=lambda x: (abs(x[0] - anchor), -x[0]))
    return {"APP": app[0][1], "APA": path, "APA run": run, "APA candidates on disk": [r for r, _ in apa], "tracked series run": anchor}


def load_s3(subject) -> M.Subject:
    """A site 3 scan as calibration_cohortII.load loads scanner B's (the APP series' stored units, the reversed b0s for
    the field, the phase encoding from the names), with Siemens' b0s found by their stated b and checked."""
    f = series_files(subject)
    img, raw, bval, bvec, slope = M._read(f["APP"])
    unit = slope
    vol = raw.astype(np.float32)
    b0 = np.flatnonzero(bval < 50)
    dw = np.flatnonzero(bval >= 50)
    assert len(b0) and np.abs(bvec[:, b0]).sum() == 0, f"{f['APP'].name}: b0s {b0.tolist()} with a gradient"
    norms = np.linalg.norm(bvec[:, dw], axis=0)
    assert np.allclose(norms, 1, atol=0.01), f"{f['APP'].name}: DW b-vectors not unit length ({norms.min():.3f}-{norms.max():.3f})"
    ref = vol[..., b0].astype(np.float64).mean(-1)
    b0_med = M.brain_median(vol[..., b0].astype(np.float64), ref)
    dw_med = M.brain_median(vol[..., dw].astype(np.float64), ref)
    over = b0_med / np.median(dw_med)
    assert (over >= CII.B0_OVER_DW).all(), f"{f['APP'].name}: stated b0s {over.round(3).tolist()} x the DW volumes: not all b0s"
    dw_like = dw_med / np.median(b0_med)
    assert dw_like.max() < B0_LIKE, f"{f['APP'].name}: a DW volume at {dw_like.max():.2f} of the b0s (b0-like)"
    dapp = descrip(img)
    series = [{"file": f["APP"].name, "volumes": len(bval), "scale": 1.0, "b0s": int(len(b0)), "b0s at": b0.tolist(),
               "b0s over dw": over.round(3).tolist(), "dw over b0s (max)": round(float(dw_like.max()), 3),
               "stated b of the shell (min, max)": [int(bval[dw].min()), int(bval[dw].max())],
               "descrip": dapp, "TR s (pixdim[4])": round(float(img.header.get_zooms()[3]), 4),
               "dtype": str(img.get_data_dtype())}]
    A = img.affine
    bval = np.where(bval < 50, 0.0, bval)
    b0_level = np.median(b0_med)
    rimg, rraw, rb, rv, rslope = M._read(f["APA"])
    rvol, how = CII.onto_coarser(rraw.astype(np.float64) * (rslope / unit), rimg, A, img.shape[:3], np.asarray(img.header.get_zooms()[:3], float))
    ratio = M.brain_median(rvol, ref) / b0_level
    rdw = np.flatnonzero(rb >= 50)
    own = ratio / np.median(ratio[rdw]) if len(rdw) else np.full(len(rb), np.inf)
    take = np.flatnonzero((rb < 50) & (own >= CII.B0_OVER_DW))
    series.append({"file": f["APA"].name, "volumes": len(rb), "scale": round(rslope / unit, 4), "stated_b": rb.tolist(),
                   "over_app_b0s": np.round(ratio, 3).tolist(), "over_own_dw": np.round(own, 3).tolist(), "taken_as_b0s": take.tolist(),
                   "b-vectors of its DW volumes": "zero" if np.abs(rv[:, rdw]).sum() == 0 else "nonzero",
                   "chosen of reversed runs": f["APA candidates on disk"], "voxel_mm": [round(float(v), 5) for v in rimg.header.get_zooms()[:3]],
                   "onto the APP grid": how, "descrip": descrip(rimg), "TR s (pixdim[4])": round(float(rimg.header.get_zooms()[3]), 4),
                   "why": f"nearest in acquisition number to the tracked series (run {f['tracked series run']})"})
    assert len(take) == (rb < 50).sum(), f"{f['APA'].name}: stated b0s {np.flatnonzero(rb < 50).tolist()}, of which {take.tolist()} are b0s ({own.round(3).tolist()})"
    pe_app, pe_apa = (np.array(M.VEC[CII.PE[d]], float) for d in ("APP", "APA"))
    pe_axis = int(np.argmax(np.abs(pe_app)))
    notes = [f"readout time nominal ({M.NOMINAL_READOUT_S} s): no sidecars", "phase encoding from the names (APP j, APA j-): no sidecars; "
             "the NIfTI descrip's phase polarity differs between the two (check item)",
             "Siemens: the b0s are the series' first volumes; each direction's stated b is its own (1980-2005)"]
    return M.Subject(name=f"{subject}_{SESSION}_{KEY}", dwi=vol, affine=A, vox=np.asarray(img.header.get_zooms()[:3], float),
                     bval=bval, bvec=bvec, pe_axis=pe_axis, pe_sign=float(pe_app[pe_axis]), pe_directions=(CII.PE["APP"], CII.PE["APA"]),
                     readout_s=M.NOMINAL_READOUT_S, b0s=np.concatenate([vol[..., b0].astype(np.float64), rvol[..., take]], -1),
                     pe_vectors=np.array([pe_app] * len(b0) + [pe_apa] * len(take)), series=series, notes=notes)


def check(s, pr=None):
    """calibration_cohortII.check against PROTOCOL_S3 (or any cohort II descriptor dict), and S3's own items (found
    from the scan: the b0 positions, the shell's stated b range; from the descrip, where a series has one)."""
    pr = PROTOCOL_S3["check"] if pr is None else pr
    out = CII.check(s, pr)
    extra = {k: v for k, v in pr.items() if k in PROTOCOL_S3["check"] and k not in out and k not in
             ("shell_round", "voxel_tolerance_mm")}
    if not extra:
        return out
    app = next(x for x in s.series if "APA" not in x["file"])
    apa = next(x for x in s.series if "APA" in x["file"])
    b = np.asarray(s.bval)
    shell = (b >= 50) & (np.abs(b - 2000) < 100)
    found = {"forward b0s at (volume index)": np.flatnonzero(b < 50).tolist(),
             "stated b of the 2000 shell (min, max)": [int(b[shell].min()), int(b[shell].max())] if shell.any() else None,
             "echo time ms (NIfTI descrip), forward and reversed": [x.get("descrip", {}).get("TE") for x in (app, apa)],
             "phase polarity (NIfTI descrip), forward and reversed": [x.get("descrip", {}).get("phase") for x in (app, apa)]}
    for k, want in extra.items():
        v = found[k]
        if k == "stated b of the 2000 shell (min, max)":
            ok = v is not None and want[0] <= v[0] and v[1] <= want[1]
        else:
            ok = v is not None and None not in v and [round(float(x), 3) for x in v] == [round(float(x), 3) for x in want]
        out[k] = (v, bool(ok))
    return out


def load_checked(subject):
    s = load_s3(subject)
    bad = {k: v for k, (v, ok) in check(s).items() if not ok}
    if bad:
        raise ValueError(f"{s.name}: not protocol {KEY}: {bad} (expected { {k: PROTOCOL_S3['check'][k] for k in bad} })")
    return s


def refusals():
    """S3_2000 against one scan of every cohort II protocol (the first loadable pair's A), cohort III's (cIIIs03 A) and
    cIs1's; and every site 3 scan against every cohort II descriptor, cohort III's, cIs1's and the cohort III map's
    match (protocol items)."""
    from _masivar import load as load_iii
    from tractline.labelers.rapidparc import HASHES
    card = json.loads(CII.CARD_PATH.read_text())
    others = {}
    for key in CII.PROTOCOLS:
        for sub, pair in CII.pairs(key).items():
            try:
                others[key] = CII.load(sub, pair.ref[0], key)
                break
            except (FileNotFoundError, AssertionError):
                continue
    if "A2000r25" not in others:
        try:
            others["A2000r25"] = CII.load("cIIs00", "s1Ax1", "A2000r25")
        except (FileNotFoundError, AssertionError):
            pass
    others["cohort III (cIIIs03 A)"] = load_iii(CT.CONTROL[0], *CT.CONTROL[1], CR.SHELLS)
    others["cIs1 A"] = load_iii("cIs1", *CT.PAIR.ref, CT.SHELLS)
    theirs = {f"{k}: {s.name}": {"S3_2000 accepts": not [i for i, (_, ok) in check(s).items() if not ok],
                                 "S3_2000 items failed": [i for i, (_, ok) in check(s).items() if not ok]} for k, s in others.items()}
    mine = {}
    for sub in SUBJECTS:
        try:
            s = load_s3(sub)
        except FileNotFoundError as e:
            mine[sub] = {"not on disk": str(e)}
            continue
        row = {k: not [i for i, (_, ok) in CII.check(s, k).items() if not ok] for k in CII.PROTOCOLS}
        row.update({"cohort III": not [i for i, (_, ok) in CT.check(s, CR.PROTOCOL).items() if not ok],
                    "cIs1": not [i for i, (_, ok) in CT.check(s, CT.PROTOCOL_CIS1).items() if not ok],
                    "cohort III map's match (protocol items only)": CT.match(card, s, {}, card["validity"]["match"]["R"], HASHES["rapidparc"],
                                                                              [CII.TERM])["items"]["protocol"]["ok"],
                    KEY: not [i for i, (_, ok) in check(s).items() if not ok]})
        mine[s.name] = {"accepted by": [k for k, ok in row.items() if ok], "refused by": [k for k, ok in row.items() if not ok],
                        "S3_2000 items": {k: v for k, (v, _) in check(s).items()}}
    return {"other protocols' scans against S3_2000": theirs, "site 3 scans against every descriptor": mine,
            "S3_2000 refuses every other protocol's scan": all(not v["S3_2000 accepts"] for v in theirs.values()),
            "each site 3 scan accepted by S3_2000 only": all(v.get("accepted by") == [KEY] for v in mine.values())}


# ----------------------------------------------------------------------------------------------- the b-vector check

def tensor_fit(dwi, affine, bval, bvec, shell, voxels=None):
    """A log-linear least-squares tensor fit of the data as the tracker gets it (prep.prepare: the b0s and the shell,
    the gradients in RAS): (flat voxel indices, principal direction (V, 3) RAS, FA (V,)), on `voxels` (flat indices;
    default prepare's mask)."""
    from tractline.prep import prepare
    t = prepare(dwi, affine, bval, bvec, shell=shell, device="cpu")
    G = np.array([[float(x) for x in t.header[f"DWMRI_gradient_{n:04d}"].split()] for n in range(t.dwi.shape[-1])])
    bmax = float(t.header["DWMRI_b-value"])
    b = bmax * (G ** 2).sum(1)
    w = b > 50
    g = G[w] / np.linalg.norm(G[w], axis=1, keepdims=True)
    idx = np.flatnonzero(t.mask.ravel()) if voxels is None else np.asarray(voxels)
    S = t.dwi.reshape(-1, t.dwi.shape[-1])[idx].astype(np.float64)
    s0 = np.clip(S[:, ~w].mean(1), 1, None)
    y = np.log(np.clip(S[:, w], 1, None) / s0[:, None])
    X = -b[w][:, None] * np.stack([g[:, 0] ** 2, g[:, 1] ** 2, g[:, 2] ** 2, 2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2], 2 * g[:, 1] * g[:, 2]], 1)
    d = np.linalg.lstsq(X, y.T, rcond=None)[0].T
    T = np.stack([np.stack([d[:, 0], d[:, 3], d[:, 4]], -1), np.stack([d[:, 3], d[:, 1], d[:, 5]], -1),
                  np.stack([d[:, 4], d[:, 5], d[:, 2]], -1)], 1)
    ev, vec = np.linalg.eigh(T)
    ev = np.clip(ev, 0, None)
    fa = np.sqrt(1.5 * ((ev - ev.mean(1, keepdims=True)) ** 2).sum(1) / np.maximum((ev ** 2).sum(1), 1e-30))
    return idx, vec[:, :, -1], fa, t.mask


def tangents(fibers, affine, shape, min_segments=20, min_linearity=0.9):
    """Per voxel of the grid, the streamlines' principal tangent direction (RAS) where they are coherent: (flat indices,
    directions (V, 3))."""
    lens = np.array([len(f) for f in fibers])
    F = np.concatenate(fibers).astype(np.float64)
    start = np.concatenate([[0], np.cumsum(lens)[:-1]])
    a = np.setdiff1d(np.arange(len(F) - 1), start[1:] - 1)
    d = F[a + 1] - F[a]
    n = np.linalg.norm(d, axis=1)
    ok = n > 0
    u, mid = d[ok] / n[ok, None], (F[a + 1] + F[a])[ok] / 2
    inv = np.linalg.inv(affine)
    ijk = np.rint(mid @ inv[:3, :3].T + inv[:3, 3]).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < shape), 1)
    flat, u = np.ravel_multi_index(ijk[inside].T, shape), u[inside]
    nvox = int(np.prod(shape))
    cnt = np.bincount(flat, minlength=nvox)
    sel = np.flatnonzero(cnt >= min_segments)
    pos = np.full(nvox, -1)
    pos[sel] = np.arange(len(sel))
    T = np.zeros((len(sel), 3, 3))
    keep = pos[flat] >= 0
    for i in range(3):
        for j in range(i, 3):
            c = np.bincount(pos[flat[keep]], u[keep, i] * u[keep, j], minlength=len(sel))
            T[:, i, j] = T[:, j, i] = c
    ev, vec = np.linalg.eigh(T)
    lin = ev[:, -1] / np.maximum(ev.sum(1), 1e-30)
    good = lin >= min_linearity
    return sel[good], vec[good, :, -1]


def rois(rec, affine, shape):
    """A's corpus callosum body at the midline (CC4 + CC5 counts >= 5, within 5 mm of their median world x) and its CST
    below the callosum (CST counts >= 5, world z below the callosal body's median): flat indices."""
    cnt = rec.counts
    ijk = np.indices(shape).reshape(3, -1).T
    world = ijk @ affine[:3, :3].T + affine[:3, 3]
    ccb = cnt[[TRACT_NAMES.index("CC4"), TRACT_NAMES.index("CC5")]].sum(0) >= 5
    xm = np.median(world[ccb, 0])
    mid = ccb & (np.abs(world[:, 0] - xm) <= 5)
    zcc = np.median(world[mid, 2])
    cst = (cnt[TRACT_NAMES.index("CST")] >= 5) & (world[:, 2] < zcc)
    return {"CC body, midline": np.flatnonzero(mid), "CST below the callosum": np.flatnonzero(cst)}


def axis_stats(e1, fa, axis, frame):
    """|component| of the principal direction along `axis` (0 x/left-right, 2 z/superior-inferior), in `frame` (3x3
    columns: world, or the grid's axes)."""
    c = np.abs(e1 @ frame)
    return {"voxels": int(len(e1)), "FA median": round(float(np.median(fa)), 3) if len(fa) else None,
            "median |e1 along the axis|": round(float(np.median(c[:, axis])), 3) if len(c) else None,
            "share where it is the largest component": round(float(np.mean(c.argmax(1) == axis)), 3) if len(c) else None}


def bvec_check(dwi, affine, bval, bvec, shell, rec, flips_in=None):
    """The principal direction in A's CC body and CST (should be left-right and superior-inferior), and its agreement
    with A's streamline tangents for the b-vectors as stated and with each axis flipped. flips_in: a 3x3 T with
    bvec = T @ native, so that a flip is made in the scan's native b-vector axes (T @ diag(flip) @ native); default
    flips in bvec's own axes."""
    shape = dwi.shape[:3]
    r = rois(rec, affine, shape)
    tv, tdir = tangents(rec.fibers, affine, shape)
    grid_axes = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)
    native = np.linalg.pinv(flips_in) @ bvec if flips_in is not None else bvec
    T = flips_in if flips_in is not None else np.eye(3)
    out = {"tangent voxels (A's streamlines coherent, >= 20 segments, linearity >= 0.9)": int(len(tv))}
    allv = np.unique(np.concatenate([tv, *r.values()]))
    for name, flip in (("as stated", None), ("x flipped", 0), ("y flipped", 1), ("z flipped", 2)):
        nb = native.copy()
        if flip is not None:
            nb[flip] *= -1
        idx, e1, fa, _ = tensor_fit(dwi, affine, bval, T @ nb, shell, allv)
        at = {v: i for i, v in enumerate(idx)}
        row = {}
        if flip is None:
            for rn, vox in r.items():
                ii = np.array([at[v] for v in vox], int)
                axis = 0 if rn.startswith("CC") else 2
                row[rn] = {"world axes": axis_stats(e1[ii], fa[ii], axis, np.eye(3)), "the grid's axes": axis_stats(e1[ii], fa[ii], axis, grid_axes)}
        ii = np.array([at[v] for v in tv], int)
        sel = fa[ii] >= 0.3
        cos = np.abs((e1[ii][sel] * tdir[sel]).sum(1))
        row["agreement with A's streamline tangents (FA >= 0.3)"] = {"voxels": int(sel.sum()), "mean |cos|": round(float(cos.mean()), 4),
                                                                     "median |cos|": round(float(np.median(cos)), 4)}
        out[name] = row
    agree = {k: out[k]["agreement with A's streamline tangents (FA >= 0.3)"]["mean |cos|"] for k in ("as stated", "x flipped", "y flipped", "z flipped")}
    out["as stated agrees best"] = bool(agree["as stated"] > max(v for k, v in agree.items() if k != "as stated"))
    return out


def fa_correlation(dwi_a, bval_a, bvec_a, dwi_b, bval_b, bvec_b, affine, shell, mask):
    """FA's Pearson correlation between two scans on one grid, in a mask (contrast-free registration quality)."""
    vox = np.flatnonzero(mask.ravel())
    _, _, fa_a, _ = tensor_fit(dwi_a, affine, bval_a, bvec_a, shell, vox)
    _, _, fa_b, _ = tensor_fit(dwi_b, affine, bval_b, bvec_b, shell, vox)
    ok = np.isfinite(fa_a) & np.isfinite(fa_b)
    return round(float(np.corrcoef(fa_a[ok], fa_b[ok])[0, 1]), 4)


# ----------------------------------------------------------------------------------------------- the runs

class Records(CR.Records):
    """<subject>-<source>-site3/ (written) on top of the source protocol's record folder (read, never written)."""
    def __init__(self, subject, source="B2000"):
        self.subject, self.source = subject, source
        self.pair = CII.pairs(source)[subject]
        self.dir = DATA / "ensemble" / subject / "card" / f"{self.pair.name}-{TAG}"
        self.reuse = CII.Records(subject, source).dir


def module_sha256():
    return CR.sha256(Path(__file__).resolve())


class Runner(CII.Runner):
    """calibration_cohortII.Runner's one() (tracked at the source protocol's shell), records in this script's folder."""
    def __init__(self, subject, source="B2000"):
        self.key, self.shell = source, CII.PROTOCOLS[source]["tracked shell"]
        self.R = Records(subject, source)
        self.subject, self.pair = subject, self.R.pair
        self.R.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.R.dir / "runs.json"
        self.log = json.loads(self.log_path.read_text()) if self.log_path.exists() else {}
        self.log.setdefault("created", {"git": CR.git_state(), "calibration_crosssite.py sha256": module_sha256(),
                                        "calibration_cohortII.py sha256": CII.module_sha256(), "pipeline": CII.pipeline_settings(source),
                                        "source protocol": {"key": source, **CII.PROTOCOLS[source]}, "rescan protocol": {"key": KEY, **PROTOCOL_S3},
                                        "reference (A)": {"session": self.pair.ref, "records read from": str(self.R.reuse.relative_to(DATA))},
                                        "rescan": {"tag": TAG, "session": SESSION, "term": TERM_X}})
        self.labeler = None

    def rescan_site3(self, registration=1, register_only=False):
        """calibration_cohortII.Runner.rescan's steps for the site 3 scan: site3_onA (registration 1) or site3reg2_onA
        (registration 2), with the b-vector check."""
        tag = CII.rescan_tag(TAG, registration)
        if self.R.have(f"{tag}_onA"):
            print(f"{self.subject} {self.key}: {tag}_onA exists", flush=True)
            return
        if registration == 3 and f"{tag}_scan" in self.log and (self.log[f"{tag}_scan"]["record"] != f"{tag}_onA" or register_only):
            print(f"{self.subject} {self.key}: {tag}_scan exists ({self.log[f'{tag}_scan']['decision']})", flush=True)
            return
        A = CII.load_checked(self.subject, self.pair.ref[0], self.key)
        ref_b0 = np.load(self.R.path("A_meanb0", ".npy"))
        B = load_checked(self.subject)
        timer, t0 = P.Timer(), time.time()
        with P.exact_float32(CR.DEVICE):
            cb = P.correct(B, timer, CR.DEVICE)
        P.release_memory(CR.DEVICE)
        reg = {"registration mode": registration, "registration": CII.REGISTRATION[registration]}
        if registration == 3:
            if f"{tag}_scan" in self.log:                         # computed by an earlier --register-only: track it now
                reg = {k: v for k, v in self.log[f"{tag}_scan"].items() if k not in ("to_A", "rotation_deg", "translation_mm", "ncc_after")}
                to_ref, ncc = np.array(self.log[f"{tag}_scan"]["to_A"]), self.log[f"{tag}_scan"]["ncc_after"]
            else:
                r = CII.register_multi(B, mean_b0(cb.dwi, B.bval), A, ref_b0, None)
                pts = CII.mask_world(np.load(self.R.path("A_mask", ".npy")), A.affine)
                e = CII.reg3_entry(r, self.log.get(f"{CII.rescan_tag(TAG, 2)}_scan"), TAG, pts)
                e.update({"name": B.name, "term": TERM_X, "moving mean b0": "corrected here"})
                print(self.subject, self.key, tag, e["decision"], "best", e["best"], e["best b0 ncc"], "QC flag", e["QC"]["flag"],
                      json.dumps(e.get("against registration 2")), flush=True)
                if e["record"] != f"{tag}_onA" or register_only:
                    self.log[f"{tag}_scan"] = e
                    self.save_log()
                    return
                reg = {k: v for k, v in e.items() if k not in ("to_A", "rotation_deg", "translation_mm", "ncc_after")}
                to_ref, ncc = np.array(e["to_A"]), e["ncc_after"]
        elif registration == 1:
            to_ref, ncc = register(mean_b0(cb.dwi, B.bval), B.affine, ref_b0, A.affine)
        else:
            r = CII.register_fa(B, mean_b0(cb.dwi, B.bval), A, ref_b0)
            to_ref, ncc, init = r["to_static"], r["ncc"], r["init_to_static"]
            angle = lambda T: round(float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))), 2)  # noqa: E731
            reg.update({"_register_fa.py sha256": CR.sha256(HERE / "_register_fa.py"),
                        "FA stage": {"to_A": np.round(init, 5).tolist(), "rotation_deg": angle(init), "translation_mm": round(float(np.linalg.norm(init[:3, 3])), 2),
                                     "b0 ncc": round(r["init_ncc"], 4), "FA ncc": round(r["fa_ncc_init"], 4)},
                        "FA ncc after": round(r["fa_ncc"], 4)})
            if f"{TAG}_scan" in self.log:
                d = to_ref @ np.linalg.inv(np.array(self.log[f"{TAG}_scan"]["to_A"]))
                reg["against registration 1"] = {"rotation_deg": angle(d), "translation_mm (of the difference)": round(float(np.linalg.norm(d[:3, 3])), 2),
                                                 "registration 1 ncc_after": self.log[f"{TAG}_scan"]["ncc_after"]}
        rot = np.degrees(np.arccos(np.clip((np.trace(to_ref[:3, :3]) - 1) / 2, -1, 1)))
        dwi, bvec = onto_reference(cb.dwi, B.bvec, B.affine, to_ref, A.affine, A.dwi.shape[:3])
        secs = round(time.time() - t0, 1)
        rec = E.RunRecord.load(self.R.path("A"), logp=False, fibers=True)
        T = bvec @ np.linalg.pinv(B.bvec)                       # onto_reference's turn of the b-vectors (linear)
        t1 = time.time()
        check_rescan = bvec_check(dwi, A.affine, B.bval, bvec, self.shell, rec, flips_in=T)
        check_A = bvec_check(A.dwi, A.affine, A.bval, A.bvec, self.shell, rec)
        mask = np.load(self.R.path("A_mask", ".npy"))
        fa_r = fa_correlation(A.dwi, A.bval, A.bvec, dwi, B.bval, bvec, A.affine, self.shell, mask)
        self.log[f"{tag}_scan"] = {"name": B.name, "term": TERM_X, **reg, "series": B.series, "notes": B.notes, "shape": list(B.dwi.shape),
                                   "residual_left": cb.residual_left, "field_hz_p99_abs": round(float(np.quantile(np.abs(cb.field_hz), 0.99)), 2),
                                   "correction warnings": cb.warnings, "correct_seconds": timer.seconds, "correct_register_resample_s": secs,
                                   "to_A": np.round(to_ref, 5).tolist(), "rotation_deg": round(float(rot), 2),
                                   "translation_mm": round(float(np.linalg.norm(to_ref[:3, 3])), 2), "ncc_after": round(ncc, 4),
                                   "FA correlation with A, in A's mask (A as acquired; the rescan corrected, on A's grid)": fa_r,
                                   "b-vector check": {"rescan on A's grid (flips in its native axes)": check_rescan, "A (the Philips reference)": check_A},
                                   "checks_seconds": round(time.time() - t1, 1)}
        self.save_log()
        print(self.subject, self.key, tag, json.dumps({k: self.log[f"{tag}_scan"][k] for k in ("rotation_deg", "translation_mm", "ncc_after", "residual_left")}),
              "FA r", fa_r, "bvec as stated best:", check_rescan["as stated agrees best"], check_A["as stated agrees best"], flush=True)
        self.one(replace(A, dwi=dwi, bval=B.bval, bvec=bvec, b0s=None), f"{tag}_onA", corrected=dwi)

    def same_scanner_reference(self, registration=1):
        """The same diagnostics for the same-scanner rescan (session2): corrected again and put on A's grid by the
        transform the source's runs.json holds (as calibration_cohortII made session2_onA), for the FA correlation with A
        and the b-vector check. Logged here; nothing of the source's records is written."""
        key = same_key(registration)
        if key in self.log:
            return
        tag = CII.rescan_tag("session2", registration)
        src = json.loads((self.R.reuse / "runs.json").read_text())
        if registration == 3 and src.get(f"{tag}_scan", {}).get("record") == f"session2reg2_onA" and same_key(2) in self.log:
            self.log[key] = {"same as": same_key(2), "why": "the source's mode 3 entry confirmed mode 2's transform"}
            self.save_log()
            return
        A = CII.load_checked(self.subject, self.pair.ref[0], self.key)
        if f"{tag}_scan" not in src:
            print(f"{self.subject} {self.key}: the source has no {tag}_scan: the same-scanner reference skipped", flush=True)
            return
        slog = src[f"{tag}_scan"]
        B = CII.load_checked(self.subject, self.pair.rescans["session2"][0], self.key)
        t0 = time.time()
        with P.exact_float32(CR.DEVICE):
            cb = P.correct(B, P.Timer(), CR.DEVICE)
        P.release_memory(CR.DEVICE)
        dwi, bvec = onto_reference(cb.dwi, B.bvec, B.affine, np.array(slog["to_A"]), A.affine, A.dwi.shape[:3])
        rec = E.RunRecord.load(self.R.path("A"), logp=False, fibers=True)
        chk = bvec_check(dwi, A.affine, B.bval, bvec, self.shell, rec, flips_in=bvec @ np.linalg.pinv(B.bvec))
        mask = np.load(self.R.path("A_mask", ".npy"))
        fa_r = fa_correlation(A.dwi, A.bval, A.bvec, dwi, B.bval, bvec, A.affine, self.shell, mask)
        self.log[key] = {"scan": B.name, "transform": f"to_A from the source's runs.json ({tag}_scan)", "registration mode": registration,
                         "residual_left (again; the source's)": [cb.residual_left, slog["residual_left"]],
                         "FA correlation with A, in A's mask (A as acquired; the rescan corrected, on A's grid)": fa_r,
                         "b-vector check": chk, "seconds": round(time.time() - t0, 1)}
        self.save_log()
        print(self.subject, self.key, tag, "FA r", fa_r, "tangent agreement", chk["as stated"]["agreement with A's streamline tangents (FA >= 0.3)"]["mean |cos|"], flush=True)


def same_key(registration):
    return "same-scanner reference (session2 rebuilt on A's grid)" + ("" if registration == 1 else f", registration {registration}")


def source_ready(subject, source, runs=RUNS, registration=1):
    """The source protocol's records complete for the subject: its statements at R = runs + 1 and this registration
    mode exist (collect ran last)."""
    return CII.statements_file(CII.Records(subject, source).dir, runs + 1, registration).exists()


def run(args):
    for sub in args.subject:
        if not source_ready(sub, args.source, args.runs, args.registration):
            print(f"{sub} {args.source}: the source's records are not complete (no {CII.statements_name(args.runs + 1, args.registration)}): skipped", flush=True)
            continue
        t0 = time.time()
        rn = Runner(sub, args.source)
        rn.rescan_site3(args.registration, getattr(args, "register_only", False))
        if not getattr(args, "register_only", False):
            rn.same_scanner_reference(args.registration)
        print(f"{sub} {args.source} {TAG} done {time.time() - t0:.0f}s, {CR.free_gb():.1f} GB free", flush=True)


# ----------------------------------------------------------------------------------------------- the statements

def collect(args):
    """calibration_cohortII.collect for the site 3 rescan against the source protocol's A ensemble."""
    import nibabel as nib
    from ensemble_grid_check import Grid
    for sub in args.subject:
        t0 = time.time()
        R = Records(sub, args.source)
        reg = args.registration
        tag = CII.rescan_tag(TAG, reg)
        out = R.dir / CII.statements_name(args.runs + 1, reg)
        if out.exists() and not args.force:
            raise FileExistsError(f"{out} exists (--force to rebuild it)")
        rlog = json.loads((R.dir / "runs.json").read_text())
        if not R.have(CII.record_of(rlog, tag)):
            print(f"{sub} {args.source}: no {CII.record_of(rlog, tag)}: skipped", flush=True)
            continue
        paths = R.ensemble("", args.runs)
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{sub}: {len(missing)} of A's {len(paths)} runs missing, the first {missing[0]}")
        img = nib.load(CII.series_files(sub, R.pair.ref[0], args.source)["APP"][0][1])
        A0 = E.RunRecord.load(paths[0], logp=False)
        grid = Grid("dwi", img.affine, A0.shape)
        mask = np.load(R.path("A_mask", ".npy"))
        cent = CC.centers(mask, A0.counts, A0.shape)
        Mx = CC.sphere_matrix(cent, A0.shape, grid.h)
        keys = A0.keys
        print(f"{sub} {args.source}: R = {len(paths)}, rescan {tag}, {len(cent)} points", flush=True)
        A = CC.Scan(paths, keys, grid, Mx, t0)
        Rn = A.R
        tA = A.T[0]
        s_ok = (A.T >= 0).all(0)
        s_mode, s_votes = CC.mode_votes(A.T[:, s_ok], range(E.N_TRACTS))
        s_Avotes = (A.T[:, s_ok] == tA[s_ok]).sum(0).astype(np.int16)
        p_mode, p_votes = CC.mode_votes(A.P, range(-1, CC.NAMED))
        cnt, _ = CC.vote_hist(A.votes, A.own, Rn)
        arrays = {"keys": keys[s_ok], "s_A": tA[s_ok], "s_mode": s_mode, "s_votes": s_votes, "s_Avotes": s_Avotes,
                  "centers": cent, "p_mode": p_mode, "p_votes": p_votes, "e_cnt": cnt, "e_Asize": A.own.sum(1).astype(np.int64)}
        k, t, pres, main = CC.run_stats(R.path(CII.record_of(rlog, tag)), grid, Mx)
        arrays[f"s_tract_{tag}"] = CC.on_keys(keys, k, t)[s_ok]
        arrays[f"p_main_{tag}"] = main
        arrays[f"e_hit_{tag}"] = CC.vote_hist(A.votes, pres, Rn)[1]
        arrays[f"e_Ahit_{tag}"] = (A.own & pres).sum(1).astype(np.int64)
        src = CII.statements_file(CII.Records(sub, args.source).dir, Rn, reg)
        same = None
        if src.exists():
            z = np.load(src)
            same = {n: bool(np.array_equal(z[n], arrays[n])) for n in ("keys", "s_A", "s_mode", "s_votes", "s_Avotes", "centers", "p_mode", "p_votes", "e_cnt", "e_Asize")}
            assert all(same.values()), f"{sub}: A's arrays differ from {src}: {same}"
        log = json.loads((R.dir / "runs.json").read_text())
        sl = log[f"{tag}_scan"]
        meta = {"subject": sub, "protocol": args.source, "rescan protocol": KEY, "pair": R.dir.name, "R": Rn,
                "runs": [str(p.relative_to(DATA)) for p in paths], "registration mode": reg,
                "rescans": {tag: {"term": TERM_X, "scan": sl["name"], "rotation_deg": sl["rotation_deg"], "translation_mm": sl["translation_mm"],
                                  "ncc_after": sl["ncc_after"], "registration mode": sl.get("registration mode", 1), "record": CII.record_of(log, tag)}},
                "pooled": {}, "points": int(len(cent)), "seeds labeled in all R runs": int(s_ok.sum()),
                "A's arrays equal to the source's statements": {"file": str(src.relative_to(DATA)), "equal": same} if same else None,
                "records git at creation": log["created"]["git"], "collect git": CR.git_state(),
                "calibration_crosssite.py sha256": module_sha256(), "calibration_cohortII.py sha256": CII.module_sha256(),
                "seconds": round(time.time() - t0, 1)}
        tmp = out.with_name(out.stem + ".partial.npz")
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
        tmp.replace(out)
        print(json.dumps(meta, indent=1), flush=True)


# ----------------------------------------------------------------------------------------------- evaluation

def at(curve, q):
    return CII.at(curve, q)


def map_on(train, train_term, test, test_term, pat=None, levels=LEVELS):
    """A map fitted on `train`'s rescans of train_term (calibration_card.fit_term), applied to `test`'s rescans of
    test_term: per target the coverage at the map's level (the widest outline where the map never reaches it), as
    calibration_cohortII.coverage_at_targets; and the streamline and point ECE with the map's vote tables."""
    card = CC.fit_term(train, train_term)
    curve = CC.term_curve(test, test_term)
    widest = float(CC.term_shapes(test, test_term)["widest"]["coverage"])
    cov, reached, used = [], [], []
    for q in levels:
        lv, ok, _ = CC.level_for(card["_curve"], float(q))
        cov.append(round(float(curve[np.flatnonzero(CC.GRID == lv)[0]]) if lv < 1 else widest, 4))
        reached.append(bool(ok))
        used.append(lv)
    R = test.R
    v, h = test.streamline(test_term)
    vp, hp = test.point(test_term)
    out = {"map fitted on": card["subjects"], "map term": train_term, "coverage at target": cov, "reached": reached, "stated level used": used,
           "streamline ECE (map)": CC.ece(CC.apply_votes(card["streamline"], v, R), h)["ece"],
           "point ECE (map)": CC.ece(CC.apply_votes(card["point"], vp, R), hp)["ece"]}
    if pat is not None:
        out["streamline ECE (PAT16 map)"] = CC.ece(pat(v / R), h)["ece"]
    return out


def stated(s, term):
    curve = CC.term_curve(s, term)
    R = s.R
    v, h = s.streamline(term)
    vp, hp = s.point(term)
    return {"coverage at stated level": [at(curve, float(q)) for q in LEVELS], "streamline ECE": CC.ece(v / R, h)["ece"],
            "point ECE": CC.ece(vp / R, hp)["ece"], "streamline: the rescan labels the seed": round(float(np.mean(s.z[f"s_tract_{s.tags(term)[0]}"] >= 0)), 4)}


def pick(xs, qs=TABLE):
    return {f"{q:g}": xs[int(np.flatnonzero(np.isclose(LEVELS, q))[0])] for q in qs}


def mean_curve(rows):
    return [round(float(x), 4) for x in np.mean(rows, 0)] if rows else None


def tang(chk):
    return chk["as stated"]["agreement with A's streamline tangents (FA >= 0.3)"]["mean |cos|"] if chk else None


def brief(chk):
    """The b-vector check in short: CC body |e1 x| and CST |e1 z| (the grid's axes, median), tangent agreement per flip."""
    if not chk:
        return None
    st = chk["as stated"]
    return {"CC body midline: median |e1 left-right|, share largest": [st["CC body, midline"]["the grid's axes"][k] for k in ("median |e1 along the axis|", "share where it is the largest component")],
            "CST below the callosum: median |e1 superior-inferior|, share largest": [st["CST below the callosum"]["the grid's axes"][k] for k in ("median |e1 along the axis|", "share where it is the largest component")],
            "tangent agreement, mean |cos|: as stated, x, y, z flipped": [chk[k]["agreement with A's streamline tangents (FA >= 0.3)"]["mean |cos|"] for k in ("as stated", "x flipped", "y flipped", "z flipped")],
            "as stated agrees best": chk["as stated agrees best"]}


def evaluate_source(source, pat, registration=1):
    """(i)-(iv) for one source protocol, both pairs at one registration mode: {per subject, summary, fig6 rows}."""
    stn = CII.statements_name(RUNS + 1, registration)
    tag, stag = CII.rescan_tag(TAG, registration), CII.rescan_tag("session2", registration)
    xf = {sub: CII.statements_file(Records(sub, source).dir, RUNS + 1, registration) for sub in SUBJECTS}
    X = {sub: CC.Subject(f) for sub, f in xf.items() if f.exists()}
    sf = {sub: CII.statements_file(CII.Records(sub, source).dir, RUNS + 1, registration) for sub in X}
    S = {sub: CC.Subject(sf[sub]) for sub in X}
    if len(X) < 2:
        return {"skipped": f"cross-vendor statements for {list(X)} only (need >= 2)"}
    assert len({s.R for s in list(X.values()) + list(S.values())}) == 1
    per = {}
    for sub in X:
        rest = [m for m in X if m != sub]
        xs, ss = stated(X[sub], TERM_X), stated(S[sub], CII.TERM)
        x_map = map_on([X[m] for m in rest], TERM_X, X[sub], TERM_X, pat)
        s_map = map_on([S[m] for m in rest], CII.TERM, S[sub], CII.TERM, pat)
        mod_cov, mod_reached = CII.coverage_at_targets([S[m] for m in rest], S[sub])   # calibration_cohortII's evaluate, as written
        assert mod_cov == s_map["coverage at target"] and mod_reached == s_map["reached"], (sub, mod_cov, s_map["coverage at target"])
        carry = map_on([S[m] for m in rest], CII.TERM, X[sub], TERM_X, pat)
        log = json.loads((Records(sub, source).dir / "runs.json").read_text())
        slog = json.loads((CII.Records(sub, source).dir / "runs.json").read_text())
        sl = log[f"{tag}_scan"]
        ref = log.get(same_key(registration), {})
        ref = log.get(ref["same as"], {}) if "same as" in ref else ref
        per[sub] = {
            "(i) cross vendor, as stated": xs,
            "(ii) cross vendor, map (leave one subject out; fitted on the others' cross-vendor pairs)": x_map,
            "(iii) same scanner (other session), as stated": ss,
            "(iii) same scanner (other session), map (leave one subject out)": s_map,
            "(iv) same-scanner map (without this subject) applied cross vendor": carry,
            "table (0.8, 0.9, 0.95)": {
                "same scanner, as stated": pick(ss["coverage at stated level"]), "same scanner, map": pick(s_map["coverage at target"]),
                "cross vendor, as stated": pick(xs["coverage at stated level"]), "cross vendor, map": pick(x_map["coverage at target"]),
                "cross vendor, same-scanner map": pick(carry["coverage at target"])},
            "registration": {"site 3 (rotation deg, translation mm, NCC)": [sl["rotation_deg"], sl["translation_mm"], sl["ncc_after"]],
                             "same-scanner session 2 (rotation deg, translation mm, NCC)": [slog[f"{stag}_scan"][k] for k in ("rotation_deg", "translation_mm", "ncc_after")],
                             "registration mode (site 3, session 2)": [sl.get("registration mode", 1), slog[f"{stag}_scan"].get("registration mode", 1)],
                             "FA correlation with A (site 3, session 2)": [
                                 sl.get("FA correlation with A, in A's mask (A as acquired; the rescan corrected, on A's grid)"),
                                 ref.get("FA correlation with A, in A's mask (A as acquired; the rescan corrected, on A's grid)")],
                             "tensor direction against A's streamline tangents, mean |cos| (site 3, session 2, A itself)": [
                                 tang(sl.get("b-vector check", {}).get("rescan on A's grid (flips in its native axes)")), tang(ref.get("b-vector check")),
                                 tang(sl.get("b-vector check", {}).get("A (the Philips reference)"))]},
            "b-vector check (site 3 on A's grid)": brief(sl.get("b-vector check", {}).get("rescan on A's grid (flips in its native axes)")),
            "b-vector check (A, the Philips reference)": brief(sl.get("b-vector check", {}).get("A (the Philips reference)")),
            "correction": {"site 3 residual_left": sl["residual_left"], "site 3 field_hz_p99_abs": sl["field_hz_p99_abs"],
                           "session 2 residual_left": slog[f"{stag}_scan"]["residual_left"]},
            "registration 3 (site 3, session 2)": [{k: x.get(k) for k in ("decision", "record", "best", "best b0 ncc", "QC")} if x.get("registration mode") == 3 else None
                                                  for x in (sl, slog[f"{stag}_scan"])],
            "streamlines": {"A": slog["A"]["fibers"], "session2_onA": slog.get(CII.record_of(slog, stag), {}).get("fibers"),
                            "site3_onA": log.get(CII.record_of(log, tag), {}).get("fibers")}}
    names = list(per)
    col = lambda f: {q: round(float(np.mean([f(per[s])[q] for s in names])), 4) for q in (f"{x:g}" for x in TABLE)}  # noqa: E731
    tab = {k: col(lambda r, k=k: r["table (0.8, 0.9, 0.95)"][k]) for k in per[names[0]]["table (0.8, 0.9, 0.95)"]}
    gap = {k: {q: round(v - float(q), 4) for q, v in d.items()} for k, d in tab.items()}
    diff = lambda a, b: {q: [round(per[s]["table (0.8, 0.9, 0.95)"][a][q] - per[s]["table (0.8, 0.9, 0.95)"][b][q], 4) for s in names] for q in tab[a]}  # noqa: E731
    summary = {"subjects": names, "mean coverage (0.8, 0.9, 0.95)": tab, "mean gap, coverage - target": gap,
               "per subject: cross vendor as stated - same scanner as stated": diff("cross vendor, as stated", "same scanner, as stated"),
               "per subject: cross-vendor map - cross vendor as stated": diff("cross vendor, map", "cross vendor, as stated"),
               "per subject: same-scanner map on cross vendor - cross vendor as stated": diff("cross vendor, same-scanner map", "cross vendor, as stated"),
               "per subject: |coverage - target| mean over 0.8, 0.9, 0.95": {
                   k: {s: round(float(np.mean([abs(per[s]["table (0.8, 0.9, 0.95)"][k][f"{q:g}"] - q) for q in TABLE])), 4) for s in names}
                   for k in tab},
               "ECE (mean over subjects): streamline / point": {
                   "same scanner, as stated": [round(float(np.mean([per[s]["(iii) same scanner (other session), as stated"][m] for s in names])), 4) for m in ("streamline ECE", "point ECE")],
                   "same scanner, map": [round(float(np.mean([per[s]["(iii) same scanner (other session), map (leave one subject out)"][m] for s in names])), 4) for m in ("streamline ECE (map)", "point ECE (map)")],
                   "cross vendor, as stated": [round(float(np.mean([per[s]["(i) cross vendor, as stated"][m] for s in names])), 4) for m in ("streamline ECE", "point ECE")],
                   "cross vendor, map": [round(float(np.mean([per[s]["(ii) cross vendor, map (leave one subject out; fitted on the others' cross-vendor pairs)"][m] for s in names])), 4) for m in ("streamline ECE (map)", "point ECE (map)")],
                   "cross vendor, same-scanner map": [round(float(np.mean([per[s]["(iv) same-scanner map (without this subject) applied cross vendor"][m] for s in names])), 4) for m in ("streamline ECE (map)", "point ECE (map)")]}}
    pr = CII.PROTOCOLS[source]
    name = CII.SCHEMA_NAME[source]
    xs_c = {s: per[s]["(i) cross vendor, as stated"]["coverage at stated level"] for s in names}
    xm_c = {s: per[s]["(ii) cross vendor, map (leave one subject out; fitted on the others' cross-vendor pairs)"]["coverage at target"] for s in names}
    xm_r = {s: per[s]["(ii) cross vendor, map (leave one subject out; fitted on the others' cross-vendor pairs)"]["reached"] for s in names}
    cr_c = {s: per[s]["(iv) same-scanner map (without this subject) applied cross vendor"]["coverage at target"] for s in names}
    fig6 = {"protocols": {f"{name} -> {KEY}": {
                "scanner": pr["scanner"][-1], "voxel_mm": pr["check"]["voxel_mm"], "tracked shell": int(pr["tracked shell"]),
                "directions": pr["check"]["volumes per shell"][str(int(pr["tracked shell"]))], "subjects": names,
                "rescan": f"site 3 ({PROTOCOL_S3['vendor']}), {KEY}: the rescan is the same person on another vendor",
                "as stated": {"per subject": xs_c, "mean": mean_curve(list(xs_c.values()))},
                "map (leave one subject out)": {"per subject": xm_c, "mean": mean_curve(list(xm_c.values())), "reached": xm_r}}},
            "transfer (fixed protocol, other scanner)": {f"{name} -> {KEY}": {
                "map from": name, "applied to": f"{name} -> {KEY}", "map fitted on": names,
                "confounded": "vendor (Philips Achieva against Siemens Skyra; TE 90 against 95 ms; one session)" + (
                    "; and resolution (1.9 x 1.9 x 2.2 mm A against the 2.5 mm site 3 scan resampled onto it)" if source == "A2000r19" else ""),
                "per subject": cr_c, "mean": mean_curve(list(cr_c.values())),
                "note": f"the {name} same-scanner (other-session) map fitted without the target subject, applied to the {name} envelope "
                        f"against that subject's site 3 rescan"}}}
    prov = {}
    for sub in names:
        R = Records(sub, source)
        rl = json.loads((R.dir / "runs.json").read_text())
        prov[sub] = {"cross-vendor records": str(R.dir.relative_to(DATA)), "cross-vendor statements": xf[sub].name, "cross-vendor statements sha256": CR.sha256(xf[sub]),
                     "registration mode": registration, "rescan record": CII.record_of(rl, tag), "rescan record sha256": CR.sha256(R.path(CII.record_of(rl, tag))), "cross-vendor runs.json sha256": CR.sha256(R.dir / "runs.json"),
                     "cross-vendor records created": json.loads((R.dir / "runs.json").read_text())["created"]["git"],
                     "source records": str(CII.Records(sub, source).dir.relative_to(DATA)),
                     "source statements": sf[sub].name, "source statements sha256": CR.sha256(sf[sub]),
                     "source runs.json sha256": CR.sha256(CII.Records(sub, source).dir / "runs.json"),
                     "source A.npz sha256": CR.sha256(R.path("A")),
                     "site 3 files sha256": {p.name: CR.sha256(p) for p in (series_files(sub)["APP"], series_files(sub)["APA"])}}
    return {"per subject": per, "summary": summary, "fig6": fig6, "provenance": prov}


def git_extra():
    run = lambda *a: subprocess.run(["git", *a], cwd=HERE.parent, capture_output=True, text=True).stdout.rstrip()  # noqa: E731
    files = ["bench/calibration_crosssite.py", "bench/calibration_cohortII.py", "bench/fetch_ds003416.py"]
    return {"status of this script, calibration_cohortII.py and fetch_ds003416.py": [x for x in run("status", "--porcelain", "--", *files).splitlines() if x]}


def evaluate(args):
    pat = CC.pat16_table()
    res = {"what": "MASiVar cohort II across vendors: a Philips scan's 17-run ensemble envelope against the same person's Siemens "
                   "(site 3) scan, as stated and with recalibration maps; beside the same-scanner other-session rescan",
           "term": TERM_X, "levels": LEVELS.tolist(), "registration mode": args.registration, "registration": CII.REGISTRATION[args.registration],
           "coverage": "calibration_card.coverage_curve: the share of the rescan's density presence inside the envelope, the 42 named "
                       "tracts pooled, weighted by the rescan's voxels",
           "protocol descriptor S3_2000": PROTOCOL_S3, "sources": {}}
    fig6 = {"schema": CII.SCHEMA, "mock": False, "levels": LEVELS.tolist(), "protocols": {}, "transfer (fixed protocol, other scanner)": {},
            "note": "COHORT2_SCHEMA's shape; fig6_protocol_gap draws protocols named B2000, B32, A2000, A32 only, so these rows need "
                    "their names added there to be drawn (ismrm2027_figures.py is not changed by this script)"}
    for source in args.sources:
        r = evaluate_source(source, pat, args.registration)
        res["sources"][source] = {k: v for k, v in r.items() if k != "fig6"}
        if "fig6" in r:
            fig6["protocols"].update(r["fig6"]["protocols"])
            fig6["transfer (fixed protocol, other scanner)"].update(r["fig6"]["transfer (fixed protocol, other scanner)"])
    res["fig6 (COHORT2_SCHEMA shape)"] = fig6
    res["refusals"] = refusals()
    res["provenance"] = {"git": CR.git_state(), **git_extra(), "calibration_crosssite.py sha256": module_sha256(),
                         "calibration_cohortII.py sha256": CII.module_sha256(), "created (UTC)": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                         "downloads": "bench/fetch_ds003416.py (SITE3), every size equal to OpenNeuro's S3 listing (ds003416/fetched.json)"}
    res["caveats"] = [
        "four subjects, one site 3 session each: means and per-subject signs, no intervals; the cross-vendor map is fitted on 3 subjects",
        "vendor differs in more than the vendor: TE 95 against 90 ms, TR (pixdim) 6.35 against 5.9 s, 50 against 48 slices, two forward "
        "b0s (first) against one (last), stated b per direction 1980-2005 against 2000, uint16 against scaled int16, other gradient "
        "hardware, coil and reconstruction; the site 3 session is another day as well as another scanner",
        "the readout time is nominal on both (no sidecars): the field is estimated and applied with the same value, so the "
        "correction is consistent within a scan but the Hz scale is not comparable across vendors",
        "the site 3 rescan is resampled once onto A's grid (cubic), as the same-scanner rescan is",
        "the same people on both scanners: the cross-vendor LOSO map never sees the target person; the same-scanner map applied "
        "cross vendor is fitted without the target person too"]
    Path(args.out).write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    short = {src: (v.get("summary", {}).get("mean coverage (0.8, 0.9, 0.95)") or v) for src, v in res["sources"].items()}
    print(json.dumps(short, indent=1))
    print("refusals strict:", res["refusals"]["S3_2000 refuses every other protocol's scan"], res["refusals"]["each site 3 scan accepted by S3_2000 only"])
    print(f"wrote {args.out}")


def check_cmd(args):
    """Load every site 3 scan, print the S3 items, the b0s, the reversed b0's contrast and the refusals table."""
    for sub in SUBJECTS:
        s = load_s3(sub)
        app, apa = s.series
        print(f"{s.name}: b0s at {app['b0s at']} ({app['b0s over dw']} x DW; DW max {app['dw over b0s (max)']} of b0s), shell b {app['stated b of the shell (min, max)']}, "
              f"TE {app['descrip'].get('TE')}/{apa['descrip'].get('TE')}, phase {app['descrip'].get('phase')}/{apa['descrip'].get('phase')}, "
              f"TR {app['TR s (pixdim[4])']}/{apa['TR s (pixdim[4])']}; reversed {apa['file'].split('_acq-')[1]}: over APP b0s {apa['over_app_b0s']}, "
              f"own {apa['over_own_dw']}, taken {apa['taken_as_b0s']}, DW b-vectors {apa['b-vectors of its DW volumes']}", flush=True)
        bad = {k: v for k, (v, ok) in check(s).items() if not ok}
        print("   S3_2000:", "accepted" if not bad else f"refused {bad}", flush=True)
    print(json.dumps(refusals(), indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("check", "run", "collect", "evaluate"))
    ap.add_argument("--source", choices=SOURCES, default="B2000", help="run/collect: the protocol whose A ensemble is tested")
    ap.add_argument("--sources", nargs="+", choices=SOURCES, default=["B2000"], help="evaluate: the source protocols")
    ap.add_argument("--subject", nargs="+", default=list(SUBJECTS), choices=SUBJECTS)
    ap.add_argument("--runs", type=int, default=RUNS)
    ap.add_argument("--force", action="store_true", help="collect: rebuild an existing statements file (derived, not a record)")
    ap.add_argument("--out", default=None, help="evaluate: default results/calibration_crosssite.json (registration 1) or "
                                                 "results/calibration_crosssite_reg<mode>.json")
    ap.add_argument("--registration", type=int, choices=(1, 2, 3), default=CII.DEFAULT_REGISTRATION,
                    help="run: the site 3 rescan's registration mode; collect, evaluate: the records' (default calibration_cohortII.DEFAULT_REGISTRATION)")
    ap.add_argument("--register-only", action="store_true", help="run: mode 3's transform and decision only (no tracking)")
    args = ap.parse_args()
    args.out = args.out or str(OUT if args.registration == 1 else OUT.with_name(f"{OUT.stem}_reg{args.registration}.json"))
    {"check": check_cmd, "run": run, "collect": collect, "evaluate": evaluate}[args.what](args)
