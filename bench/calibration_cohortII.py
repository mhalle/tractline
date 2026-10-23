"""MASiVar cohort II: is the as-stated gap the same on another protocol and scanner, and does a recalibration map (the
"card") transfer between scanners when the protocol is fixed? (docs/ismrm2027-abstract.md; the cohort III card,
results/calibration_card_masivar_cIII.json; its out-of-protocol check on cIs1, results/calibration_transfer_cIs1.json.)

    uv run --group dipy --group rankfield bench/calibration_cohortII.py plan [--check]     # inventory, pairs, what is on disk, the time
    uv run --group dipy --group rankfield bench/calibration_cohortII.py run --protocol B2000 [--subject cIIs01 ...] [--runs 16] [--smoke] [--registration 2]
    uv run --group dipy --group rankfield bench/calibration_cohortII.py collect --protocol B2000 [--subject ...] [--runs 16] [--smoke] [--force] [--registration 2]
    uv run --group dipy --group rankfield bench/calibration_cohortII.py evaluate [--runs 16] [--registration 2] [--dry-run --out <elsewhere> [--smoke]]
    uv run --group dipy --group rankfield bench/calibration_cohortII.py compare [--protocols B2000 B32 ...] [--modes 2 3] [--out x.json]
    uv run --group dipy --group rankfield bench/calibration_cohortII.py run --protocol B2000 --registration 3 --register-only   # mode 3's transforms only

The data (bench/fetch_ds003416.py): cohort II, site 1, subjects cIIs00-cIIs04, scanners A and B, sessions s1Ax1,
s1Ax2, s1Bx1 and s1Bx2 (cIIs00: s1Ax1 and s1Bx1): the 2.5 mm series, and on scanner A b2000n96 at 1.9 mm and the
reversed series nearest b2000n96 and b1000n32. What a session holds (plan prints it):
  - one scan per session: every series is run-1<acquisition>, so there is no same-session rescan. The term is the
    other session only (s1Bx1 against s1Bx2; s1Ax1 against s1Ax2);
  - scanner B: b2500n96, b2000n96, b1500n96, b1000n96 and b1000n32 (APP, 96 x 96 x 48 at 2.5 mm), and two (cIIs00) or
    three reversed series (APA, "b1000n3": b1000 x3 and a b0), interleaved: APA 104, b2500 105, b2000 106, APA 107,
    b1500 108, b1000 109, APA 110, b1000n32 112;
  - scanner A: only cIIs00's series are all 2.5 mm (96 x 96 x 42). cIIs01-04's 96-direction shells there are
    1.89286 x 1.89286 x 2.2 mm (112 x 112 x 48) and their reversed series r06x06x25: 384 x 384 x 48 at 0.625 x 0.625 x
    2.5 mm, the same 240 mm field of view and slices as b1000n32 (96 x 96 x 48 at 2.5 mm), probably an interpolated
    reconstruction. On disk: b1000n32, b2000n96 (1.9 mm) and the two reversed series nearest them;
  - no sidecars (.json) in cohort II: the phase encoding is taken from the names, APP = j and APA = j-, as every
    cohort I and III sidecar states it (116 of 116);
  - each APP series has one b0, its last volume (stated b = 0, gradient 0), no interleaved b0s. load() checks it is a
    real b0 and not a derived isotropic image: brighter than the series' diffusion-weighted volumes by B0_OVER_DW.

The protocols (PROTOCOLS; each a descriptor its `check` holds a scan to, item by item, as calibration_runs.PROTOCOL;
each names its resolution, and its items include the reversed series' voxel size):
  B2000     scanner B, the b2000n96 series alone (2.5 mm), tracked at b = 2000: 97 volumes, one forward b0 (its last
            volume) and the reversed series nearest it (APA 107) for the field. The other shells are not joined: each run
            and each replicate's wild bootstrap then works on that one shell and its b0 (the bootstrap's single-b0 path:
            its b0 borrows a DW residual pattern, scaled to the DW noise);
  B2000J    scanner B, the four 96-direction shells joined (as _masivar.load joins a scan's shells; 388 volumes, 4
            forward b0s), tracked at 2000: kept only to measure what joining costs (the smoke test), not for the queue;
  B32       scanner B, b1000n32 alone (2.5 mm), tracked at b = 1000: 33 volumes, one forward b0, the nearest reversed
            series (110);
  A2000r19  scanner A, b2000n96 alone at 1.89 x 1.89 x 2.2 mm, tracked at 2000; the reversed b0 from the nearest
            r06x06x25 series (onto_coarser: box-averaged 3 x 3 in-plane, then cubic onto the APP grid);
  A2000r25  scanner A at 2.5 mm, as B2000: on disk for cIIs00 s1Ax1 only (96 x 96 x 42), so it has no pair;
  A32       scanner A, b1000n32 alone (2.5 mm), tracked at 1000; the reversed b0 from the nearest r06x06x25 series
            (box-averaged 4 x 4 in-plane, then cubic onto the APP grid).
Scanner A's reversed series is stored on another scale (its b0 0.02 (A32) or 0.09 (A2000r19) of the forward b0s'): the
descriptors bound that ratio as found, and load() rescales the series to the forward b0s' level, so the bootstrap's added
noise (in APP units) is at the right relative level; the field estimate scales its b0s to a common mean first.
The scanner is a descriptor item (the BIDS session label, s1A or s1B): the sidecars name no scanner, and B32 and A32
differ otherwise only in the reversed series' voxel. Each descriptor refuses the others' scans, cohort III's and cIs1's;
cohort III's PROTOCOL, its map's match and cIs1's refuse every cohort II scan (`plan --check`; evaluate's "refusals").
Tracking at 1.9 mm needs no setting changed: the tracker's step (0.3 mm), record spacing (1.8 mm), length limits and
the labeler work in mm, and the mask (median_otsu, radius 4 voxels) follows the grid. Every mask voxel is a seed, so
A2000r19 has about twice the seeds and streamlines of a 2.5 mm scan (and twice the tracking time), and collect's point
lattice (every 4 voxels) is 7.6 x 7.6 x 8.8 mm there against 10 mm at 2.5 mm (cohort III: 8.6 x 8.6 x 8.8).

Per subject and protocol (PAIRS): A = the first session's scan, its 17-run ensemble (A and 16 replicates: smooth
signs, FWHM 4, b0 noise "added", A's mask, draw 0, today's default pipeline at the protocol's shell); the rescan = the
second session's scan ("session2"), corrected on its grid, registered rigidly to A, resampled onto A's grid and tracked
there. calibration_runs' Records and Runner are reused (no overwrites, the disk guard, the replicates, the rescan's
registration and resampling), with this module's loader, check and shell. Records: compact (format 2) in
DATA/ensemble/<subject>/card/<subject>-<protocol>/ with runs.json beside them; --smoke writes to
<subject>-<protocol>-smoke/ instead, so a smoke run never becomes part of the full one.

collect: calibration_card.collect's statements (as calibration_transfer.collect: no pooled reference), in the record
folder as statements_R<runs + 1>.npz (registration 1) or statements_R<runs + 1>_reg2.npz (registration 2); refused if it
exists, unless --force (derived, not a record).

The rescan's registration (REGISTRATION; --registration, default 3):
  1  ensemble_retest.register (bench/_register.py): center of mass, translation, rigid (dipy, mutual information) on the
     corrected mean b0s. Every record before 2026-10-19 evening: <tag>_onA.npz, runs.json "<tag>_scan" (no mode item);
  2  bench/_register_fa.py: FA onto FA (a log-linear tensor fit of each scan as loaded, before correction; center of
     mass, translation, rigid), then a rigid refinement of the corrected mean b0s started from it. Records beside mode
     1's, never over them: <tag>reg2_onA.npz, runs.json "<tag>reg2_scan" (with "registration mode": 2, the FA stage's
     transform and NCCs, and _register_fa.py's SHA-256).
  3  multi-start (bench/_register_fa.py mode 3): (a) mode 1's pipeline, (b) mode 2's FA stage, (c) the other
     direction's transform inverted where it exists (B2000: the swapped folder's), each refined by rigid MI on the
     corrected mean b0s; the best final b0 NCC kept, every candidate recorded, a QC flag below NCC_QC (0.8). register3
     computes it first (runs.json "<tag>reg3_scan"); the rescan is tracked again (<tag>reg3_onA, statements_R17_reg3.npz)
     only where it differs from mode 2 (SAME_DEG, SAME_MM, SAME_NCC); elsewhere the entry's "record" is mode 2's
     (<tag>reg2_onA, its transform confirmed) and statements_file reads mode 2's statements.
Mode 1 stopped in a local minimum on cIIs04's scanner B session 2 (5.4 deg / 2.8 mm against 8.3 deg / 4.2 mm; NCC 0.51
against 0.86); mode 2 is used for every pair, not that one alone. Mode 2 in turn stopped in one on cIIs04's swapped pair
(calibration_swap: 6.0 deg / 2.6 mm, NCC 0.63): FA onto FA is not symmetric; mode 3 is, again, for every pair. A, its companions and the replicates do not depend on
the mode: run --registration 2 on a finished folder runs the mode 2 rescan only.

evaluate (no pipeline runs; other-session rescans): writes results/calibration_cohortII.json, schema
tractline.calibration_cohortII/1 (ismrm2027_figures.COHORT2_SCHEMA, read by fig6_protocol_gap; there A2000r19 is
"A2000"; its "transfer (fixed protocol, other scanner)" maps are fitted without the target person, as (c)'s "source
map, without the target person", the all-subject fit an extra key) and, in further keys:
  (a) as stated, per protocol: streamline and point ECE, envelope coverage and its gap at the stated levels, per subject
      and their mean, beside cohort III's held-out as-stated numbers;
  (b) the cohort III map on B2000 (its scanner B with another protocol): its match refuses the scans (the failed
      items, per subject), and it is applied anyway; also on the other three protocols;
  (c) across scanners: B32 <-> A32 (a fixed protocol) and B2000 <-> A2000r19 (scanner and resolution confounded), the
      source map from all source subjects and from those without the target person, against the target scanner's
      own leave-one-out map; the streamline counts per run beside them (A2000r19 seeds every mask voxel of a finer grid:
      about twice a 2.5 mm scan's streamlines, a confound of its own);
  (d) a map per protocol (B2000, A2000r19, B32, A32: each scanner's two), leave one subject out (calibration_card.evaluate);
  (e) their learning curves (calibration_learning_curve's scores, every training set of k = 1 .. n - 1: 1-3 subjects);
  (f) the 32-direction protocols as stated against the 96-direction ones, paired by subject (A: resolution confounded);
  refusals: every descriptor, cohort III's and cIs1's against a scan of every protocol.
Every evaluated protocol needs all its pairs' statements at R = runs + 1; --dry-run takes whatever exists (and must
write elsewhere).

A copy of this module outside bench/ (a scratchpad patch) finds bench/ through TRACTLINE_BENCH.
"""
import argparse, json, os, re, sys, time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tractline import pipeline as P
from tractline import ensemble as E
from tractline.data import DATA

HERE = Path(__file__).resolve().parent
BENCH = HERE if (HERE / "calibration_runs.py").exists() else Path(os.environ["TRACTLINE_BENCH"])   # a copy outside bench/: set it
sys.path.insert(0, str(BENCH))
import calibration_runs as CR   # noqa: E402
import calibration_card as CC   # noqa: E402
import calibration_transfer as CT   # noqa: E402
import _masivar as M   # noqa: E402
from ensemble_retest import register, onto_reference, mean_b0  # noqa: E402

SUBJECTS = ("cIIs00", "cIIs01", "cIIs02", "cIIs03", "cIIs04")
SESSIONS = {"A": ("s1Ax1", "s1Ax2"), "B": ("s1Bx1", "s1Bx2")}
NAME = re.compile(r"acq-b(\d+)n(\d+)r(\w+?)pe(APP|APA)_run-(\d)(\d\d)_dwi\.nii\.gz$")
PE = {"APP": "j", "APA": "j-"}          # no sidecars in cohort II: as every cohort I and III sidecar states it
RES = "25x25x25"                        # the default series resolution (BIDS acq ...r<res>...)
B0_OVER_DW = 1.5                        # a series' last volume is a b0 only if its brain median is this many times the
                                        # series' DW volumes' (a derived isotropic image would be about 1)


REGISTRATION = {1: "center of mass, translation, rigid (MI) on the corrected mean b0s (ensemble_retest.register, bench/_register.py)",
                2: "FA onto FA (center of mass, translation, rigid; MI), then rigid MI refinement of the corrected mean b0s (bench/_register_fa.py)",
                3: "multi-start: (a) mode 1's pipeline on the mean b0s, (b) mode 2's FA stage, (c) the other direction's transform "
                   "inverted, where it exists; each refined by rigid MI on the corrected mean b0s, the best final b0 NCC kept "
                   "(bench/_register_fa.py, mode 3)"}
DEFAULT_REGISTRATION = 3
NCC_QC = 0.8                            # mode 3: a rescan whose best final b0 NCC is below this is flagged (kept, not dropped)
SAME_DEG, SAME_MM, SAME_NCC = 0.5, 0.5, 0.02   # mode 3 reruns a rescan only if its transform differs from mode 2's by more
                                        # than SAME_DEG or moves a voxel of A's mask by more than SAME_MM, or its b0 NCC
                                        # is more than SAME_NCC higher; otherwise mode 2's record is used, its transform confirmed


def rescan_tag(tag, registration):
    """The record tag of a rescan registered by a mode: "session2" (1), "session2reg2" (2)."""
    return tag if registration == 1 else f"{tag}reg{registration}"


def statements_name(R, registration):
    return f"statements_R{R}.npz" if registration == 1 else f"statements_R{R}_reg{registration}.npz"


def statements_file(d, R, registration):
    """A record folder's statements at a registration mode. Mode 3: its own file if collect made one; else, when every
    mode 3 rescan entry in runs.json confirmed mode 2's transform (its "record" is mode 2's), mode 2's file."""
    p = d / statements_name(R, registration)
    if registration == 3 and not p.exists() and (d / "runs.json").exists():
        log = json.loads((d / "runs.json").read_text())
        e = [v for k, v in log.items() if k.endswith("reg3_scan")]
        if e and all(v.get("record", "").endswith("reg2_onA") for v in e):
            return d / statements_name(R, 2)
    return p


def record_of(log, tag):
    """The _onA record a rescan entry uses: its "record" (mode 3 may point at mode 2's), else <tag>_onA."""
    return log.get(f"{tag}_scan", {}).get("record", f"{tag}_onA")


def angle(T):
    """A rigid transform's rotation angle (degrees), by atan2 of its skew and symmetric parts: arccos of the trace alone
    reads about 0.1 deg of noise from 5-decimal matrices near the identity."""
    R = np.asarray(T, float)[:3, :3]
    return round(float(np.degrees(np.arctan2(np.linalg.norm(R - R.T) / (2 * np.sqrt(2)), (np.trace(R) - 1) / 2))), 3)


def register_multi(B, moving_b0, A, static_b0, other_to_A=None):
    """Mode 3 (bench/_register_fa.py, in a process of its own): the starts (a) mode 1's pipeline, (b) mode 2's FA stage,
    (c) the other direction's to_A (A's world -> B's, as that direction registered A onto B) inverted, refined alike; the
    best by final b0 NCC. Returns {to_static, ncc, best, and per candidate x: start_to_static_x, to_static_x, ncc_x,
    fa_ncc_x}."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        extra = {} if other_to_A is None else {"other_start": np.asarray(other_to_A, float)}   # dipy's start: static -> moving world
        np.savez(a, moving_fa=fa_map(B), static_fa=fa_map(A), moving_b0=np.asarray(moving_b0, np.float32),
                 static_b0=np.asarray(static_b0, np.float32), moving_affine=B.affine, static_affine=A.affine, mode=3, **extra)
        subprocess.run([sys.executable, str(BENCH / "_register_fa.py"), str(a), str(b)], check=True)
        z = np.load(b)
        return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}


def mask_world(mask, affine):
    ijk = np.argwhere(mask)
    return (affine[:3, :3] @ ijk.T).T + affine[:3, 3]


def max_shift(T1, T2, pts):
    """The largest distance, over A's mask points (A's world), between where two to_A transforms sample the rescan."""
    d = (np.linalg.inv(T1) - np.linalg.inv(T2))
    return round(float(np.linalg.norm(pts @ d[:3, :3].T + d[:3, 3], axis=1).max()), 3)


def reg3_entry(r, prev2, tag, pts, other_from=None):
    """A mode 3 rescan entry: every candidate, the best, the QC flag, and the decision against mode 2's entry prev2:
    "record" is mode 2's (<tag>reg2_onA, transform confirmed) or mode 3's own (<tag>reg3_onA, to be tracked)."""
    cands = {}
    for x in ("a", "b", "c"):
        if f"to_static_{x}" in r:
            T = r[f"to_static_{x}"]
            cands[x] = {"start": {"a": "mode 1's pipeline", "b": "mode 2's FA stage", "c": f"the other direction's transform inverted ({other_from})"}[x],
                        "start_to_A": np.round(r[f"start_to_static_{x}"], 5).tolist(), "to_A": np.round(T, 5).tolist(),
                        "rotation_deg": angle(T), "translation_mm": round(float(np.linalg.norm(T[:3, 3])), 2),
                        "b0 ncc": round(float(r[f"ncc_{x}"]), 4), "FA ncc": round(float(r[f"fa_ncc_{x}"]), 4)}
    best = r["best"]
    T3 = np.asarray(r["to_static"], float)
    entry = {"registration mode": 3, "registration": REGISTRATION[3], "_register_fa.py sha256": CR.sha256(BENCH / "_register_fa.py"),
             "candidates": cands, "best": best, "best b0 ncc": round(float(r["ncc"]), 4),
             "QC": {"threshold (b0 ncc)": NCC_QC, "flag": bool(r["ncc"] < NCC_QC)}}
    if prev2 is not None:
        T2 = np.array(prev2["to_A"])
        dif = {"rotation_deg": angle(T3 @ np.linalg.inv(T2)), "max shift over A's mask (mm)": max_shift(T3, T2, pts),
               "b0 ncc gain": round(float(r["ncc"]) - prev2["ncc_after"], 4), "mode 2 ncc_after": prev2["ncc_after"]}
        same = dif["rotation_deg"] <= SAME_DEG and dif["max shift over A's mask (mm)"] <= SAME_MM and dif["b0 ncc gain"] <= SAME_NCC
        entry["against registration 2"] = {**dif, "same (within the tolerances)": same,
                                           "tolerances": {"deg": SAME_DEG, "mm": SAME_MM, "ncc gain": SAME_NCC}}
    else:
        same = False
    if same:
        entry.update({"decision": "confirmed: mode 2's transform and record are used", "record": f"{tag}reg2_onA",
                      **{k: prev2[k] for k in ("name", "term", "to_A", "rotation_deg", "translation_mm", "ncc_after", "residual_left", "field_hz_p99_abs") if k in prev2}})
    else:
        entry.update({"decision": "rerun: mode 3's transform differs from mode 2's (or mode 2 has none)", "record": f"{tag}reg3_onA",
                      "to_A": np.round(T3, 5).tolist(), "rotation_deg": round(angle(T3), 2), "translation_mm": round(float(np.linalg.norm(T3[:3, 3])), 2),
                      "ncc_after": round(float(r["ncc"]), 4)})
        if prev2 is not None:
            entry.update({k: prev2[k] for k in ("name", "term", "residual_left", "field_hz_p99_abs") if k in prev2})
    return entry


def swap_dir(subject, key):
    """The swapped direction's record folder (calibration_swap: A = the second session), if made."""
    return DATA / "ensemble" / subject / "card" / f"{subject}-{key}-swap"


def best_to_A(log, tag):
    """The best transform a folder holds for a rescan: mode 3's entry, else mode 2's (None if neither)."""
    for m in (3, 2):
        e = log.get(f"{rescan_tag(tag, m)}_scan")
        if e:
            return np.array(e["to_A"]), f"{rescan_tag(tag, m)}_scan"
    return None, None


def fa_map(s):
    """FA of a loaded scan (as loaded, before correction): a log-linear tensor fit in the voxels where its mean b0 is
    above a third of its 99th percentile (zero elsewhere); the b-vectors' frame does not change FA."""
    d = np.asarray(s.dwi, np.float64)
    b0i, dw = np.flatnonzero(s.bval < 50), np.flatnonzero(s.bval >= 50)
    b0 = d[..., b0i].mean(-1)
    m = b0 > np.quantile(b0, 0.99) / 3
    g = np.asarray(s.bvec, float)[:, dw].T
    g = g / np.linalg.norm(g, axis=1, keepdims=True)
    X = -s.bval[dw][:, None] * np.stack([g[:, 0] ** 2, g[:, 1] ** 2, g[:, 2] ** 2, 2 * g[:, 0] * g[:, 1], 2 * g[:, 0] * g[:, 2], 2 * g[:, 1] * g[:, 2]], 1)
    y = np.log(np.clip(d[m][:, dw], 1, None) / np.clip(b0[m], 1, None)[:, None])
    D = np.linalg.lstsq(X, y.T, rcond=None)[0].T
    T = np.stack([np.stack([D[:, 0], D[:, 3], D[:, 4]], -1), np.stack([D[:, 3], D[:, 1], D[:, 5]], -1), np.stack([D[:, 4], D[:, 5], D[:, 2]], -1)], 1)
    ev = np.clip(np.linalg.eigvalsh(T), 0, None)
    fa = np.zeros(m.shape, np.float32)
    fa[m] = np.sqrt(1.5 * ((ev - ev.mean(1, keepdims=True)) ** 2).sum(1) / np.maximum((ev ** 2).sum(1), 1e-30))
    return fa


def register_fa(B, moving_b0, A, static_b0):
    """Mode 2 (bench/_register_fa.py, in a process of its own): {to_static, ncc, init_to_static, init_ncc, fa_ncc_init,
    fa_ncc}, moving = the rescan B (its FA as loaded, its corrected mean b0), static = A."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(a, moving_fa=fa_map(B), static_fa=fa_map(A), moving_b0=np.asarray(moving_b0, np.float32),
                 static_b0=np.asarray(static_b0, np.float32), moving_affine=B.affine, static_affine=A.affine)
        subprocess.run([sys.executable, str(BENCH / "_register_fa.py"), str(a), str(b)], check=True)
        z = np.load(b)
        return {k: (z[k] if z[k].ndim else float(z[k])) for k in z.files}


def tracked_series(series, shell):
    """The series of the tracked shell ("b2000n96" of FOUR at 2000)."""
    return next(s for s in series if int(s[1:].split("n")[0]) == int(shell))


def _proto(scanner, series, shell, volumes, per_shell, matrix, note, reversed_over, res=RES, apa_res=RES,
           voxel=(2.5, 2.5, 2.5), apa_voxel=(2.5, 2.5, 2.5), rescale=False):
    return {"scanner": f"s1{scanner}", "series": series, "reversed series rescaled to the forward b0s": rescale, "series resolution (BIDS acq r...)": res, "reversed series resolution": apa_res,
            "resolution": f"{' x '.join(f'{v:g}' for v in voxel)} mm (r{res}); reversed series {' x '.join(f'{v:g}' for v in apa_voxel)} mm (r{apa_res})",
            "reversed series": f"the r{apa_res} APA series nearest in acquisition number to {tracked_series(series, shell)} (ties: the later)",
            "tracked shell": shell, "note": note,
            "check": {"site and scanner (BIDS session label)": f"s1{scanner}", "volumes": volumes, "volumes per shell": per_shell,
                      "shell_round": 100, "matrix": matrix, "voxel_mm": list(voxel), "voxel_tolerance_mm": 0.01,
                      "phase_encoding": ["j", "j-"], "reversed_b0s": 1, "reversed_b0_over_forward": reversed_over,
                      "reversed series voxel_mm": list(apa_voxel)}}


FOUR = ["b1000n96", "b1500n96", "b2000n96", "b2500n96"]
PER4 = {"0": 4, "1000": 96, "1500": 96, "2000": 96, "2500": 96}
PROTOCOLS = {
    "B2000": _proto("B", ["b2000n96"], 2000.0, 97, {"0": 1, "2000": 96}, [96, 96, 48], "scanner B, b2000n96 alone, tracked at 2000",
                    [0.65, 0.9]),      # seen 0.710-0.808 (cohort III 0.89-0.96)
    "A2000r25": _proto("A", ["b2000n96"], 2000.0, 97, {"0": 1, "2000": 96}, [96, 96, 42],
                       "scanner A at 2.5 mm, as B2000; on disk for cIIs00 s1Ax1 only: no pair", [0.65, 0.9]),   # seen 0.807 (its one scan)
    "A2000r19": _proto("A", ["b2000n96"], 2000.0, 97, {"0": 1, "2000": 96}, [112, 112, 48],
                       "scanner A at 1.9 x 1.9 x 2.2 mm, b2000n96 alone, tracked at 2000; field from the nearest r06x06x25 reversed b0",
                       [0.06, 0.14], res="19x19x22", apa_res="06x06x25", voxel=(1.89286, 1.89286, 2.2), apa_voxel=(0.625, 0.625, 2.5),
                       rescale=True),   # stored scale: seen 0.086-0.102 (8 scans)
    "B2000J": _proto("B", FOUR, 2000.0, 388, PER4, [96, 96, 48], "scanner B, the four 96-direction shells joined (as _masivar.load "
                     "joins a scan's shells), tracked at 2000: for the cost comparison only", [0.7, 0.9]),   # seen 0.739-0.842 (cohort III 0.89-0.96)
    "B32": _proto("B", ["b1000n32"], 1000.0, 33, {"0": 1, "1000": 32}, [96, 96, 48], "scanner B, b1000n32 alone (clinical-like), tracked at 1000",
                  [0.3, 0.45]),        # seen 0.348-0.376: the b1000n32 b0 is about twice the 96-direction series' (see load)
    "A32": _proto("A", ["b1000n32"], 1000.0, 33, {"0": 1, "1000": 32}, [96, 96, 48],
                  "scanner A, b1000n32 alone at 2.5 mm, tracked at 1000; field from the nearest r06x06x25 reversed b0",
                  [0.015, 0.035], apa_res="06x06x25", apa_voxel=(0.625, 0.625, 2.5), rescale=True),   # stored scale: seen 0.021-0.024 (8 scans)
}


def pairs(key):
    """{subject: Pair} for a protocol: A = the scanner's first session, the rescan its second (scan 1 each)."""
    s1, s2 = SESSIONS[PROTOCOLS[key]["scanner"][-1]]
    out = {}
    for sub in SUBJECTS:
        if all((M.ROOT / f"sub-{sub}/ses-{ses}/dwi").exists() for ses in (s1, s2)):
            out[sub] = CR.Pair(f"{sub}-{key}", (s1, 1), {}, {"session2": (s2, 1)},
                               notes=(PROTOCOLS[key]["note"], "one scan per session: no same-session rescan"))
    return out


# ----------------------------------------------------------------------------------------------- the scans

def series_files(subject, session, key):
    """{"APP": [(b, path)] ascending, "APA": path, "APA candidates": [...]} for a protocol's series in a session."""
    proto = PROTOCOLS[key]
    res, apa_res = proto["series resolution (BIDS acq r...)"], proto["reversed series resolution"]
    d = M.ROOT / f"sub-{subject}/ses-{session}/dwi"
    app, apa = {}, []
    for p in sorted(d.glob(f"sub-{subject}_ses-{session}_acq-*_dwi.nii.gz")):
        m = NAME.search(p.name)
        if not m:
            continue
        run = int(m.group(5) + m.group(6))
        if m.group(4) == "APA" and m.group(3) == apa_res:
            apa.append((run, p))
        elif m.group(4) == "APP" and m.group(3) == res:
            app.setdefault(f"b{m.group(1)}n{m.group(2)}", []).append((run, p))
    missing = [s for s in proto["series"] if len(app.get(s, [])) != 1]
    if missing:
        raise FileNotFoundError(f"{subject} {session}: not exactly one r{res} APP series of {missing} ({ {s: len(app.get(s, [])) for s in missing} })")
    if not apa:
        raise FileNotFoundError(f"{subject} {session}: no r{apa_res} reversed (APA) series")
    anchor = app[tracked_series(proto["series"], proto["tracked shell"])][0][0]
    run, path = min(apa, key=lambda x: (abs(x[0] - anchor), -x[0]))
    return {"APP": sorted((int(s.split("n")[0][1:]), app[s][0][1]) for s in proto["series"]), "APA": path,
            "APA candidates": [r for r, _ in apa], "APA run": run, "tracked series run": anchor}


def load(subject, session, key) -> M.Subject:
    """_masivar.load for cohort II: the protocol's APP series joined (shells ascending, onto the first's grid and stored
    units), each series' last volume checked to be a b0 (B0_OVER_DW), the chosen reversed series' stated b0s (each
    checked: B0_OVER_DW of that series' DW volumes) for the field. The phase encoding from the names (PE)."""
    files = series_files(subject, session, key)
    first = unit = None
    dws, bvals, bvecs, series = [], [], [], []
    for b, p in files["APP"]:
        img, raw, bval, bvec, slope = M._read(p)
        if first is None:
            first, unit = img, slope
        vol = M._onto(raw.astype(np.float32) * np.float32(slope / unit), img.affine, first.affine, first.shape[:3]).astype(np.float32)
        b0 = np.flatnonzero(bval < 50)
        dw = np.flatnonzero(bval >= 50)
        last = len(bval) - 1
        assert list(b0) == [last] and np.abs(bvec[:, last]).sum() == 0, f"{p.name}: b0s at {b0.tolist()}, expected the last volume only"
        ref = vol[..., last].astype(np.float64)
        over = M.brain_median(vol[..., [last]].astype(np.float64), ref)[0] / np.median(M.brain_median(vol[..., dw].astype(np.float64), ref))
        assert over >= B0_OVER_DW, f"{p.name}: the last volume is {over:.2f} x the DW volumes: not a b0 (a derived image?)"
        series.append({"file": p.name, "volumes": len(bval), "scale": round(slope / unit, 4), "b0s": 1,
                       "last_volume_over_dw": round(float(over), 3), "last_volume": "kept (a b0)"})
        dws.append(vol); bvals.append(np.where(bval < 50, 0.0, bval)); bvecs.append(bvec)
    dwi = np.concatenate(dws, -1)
    bval, bvec = np.concatenate(bvals), np.concatenate(bvecs, 1)
    A = first.affine
    b0i = np.flatnonzero(bval < 50)
    ref = dwi[..., b0i].astype(np.float64).mean(-1)
    b0_level = np.median(M.brain_median(dwi[..., b0i].astype(np.float64), ref))
    p = files["APA"]
    img, raw, rb, rv, slope = M._read(p)
    rvol, how = onto_coarser(raw.astype(np.float64) * (slope / unit), img, A, first.shape[:3], np.asarray(first.header.get_zooms()[:3], float))
    ratio = M.brain_median(rvol, ref) / b0_level
    # the reversed b0s: its stated b0s, each B0_OVER_DW brighter than the series' own DW volumes. Not by brightness
    # against the APP b0s (_masivar's B0_LIKE): b1000n32's b0 is about twice as bright as the 96-direction series' and
    # the reversed b0 is 0.35-0.38 of it (another TE or TR, the sidecars do not say); descriptors bound the ratio
    rdw = np.flatnonzero(rb >= 50)
    own = ratio / np.median(ratio[rdw]) if len(rdw) else np.full(len(rb), np.inf)
    take = np.flatnonzero((rb < 50) & (own >= B0_OVER_DW))
    series.append({"file": p.name, "volumes": len(rb), "scale": round(slope / unit, 4), "stated_b": rb.tolist(), "over_app_b0s": np.round(ratio, 3).tolist(),
                   "over_own_dw": np.round(own, 3).tolist(), "taken_as_b0s": take.tolist(), "chosen of reversed runs": files["APA candidates"],
                   "voxel_mm": [round(float(v), 5) for v in img.header.get_zooms()[:3]], "onto the APP grid": how,
                   "why": f"nearest in acquisition number to the tracked series (run {files['tracked series run']})"})
    assert len(take) == (rb < 50).sum(), f"{p.name}: stated b0s {np.flatnonzero(rb < 50).tolist()}, of which {take.tolist()} are {B0_OVER_DW} x its DW volumes ({np.round(own, 3).tolist()})"
    pe_app, pe_apa = (np.array(M.VEC[PE[d]], float) for d in ("APP", "APA"))
    pe_axis = int(np.argmax(np.abs(pe_app)))
    notes = [f"readout time nominal ({M.NOMINAL_READOUT_S} s): no sidecars", "phase encoding from the names (APP j, APA j-): no sidecars"]
    if PROTOCOLS[key].get("reversed series rescaled to the forward b0s"):
        # scanner A's r06x06x25 series is stored on another scale (its b0 0.02-0.10 of the forward b0s'): put on the
        # forward b0s' level, so the bootstrap's added b0 noise (in the APP's units) is at the right relative level.
        # The field estimate scales every b0 to a common mean intensity first, so the field does not depend on it
        f = float(1 / ratio[take].mean())
        rvol = rvol * f
        series[-1]["rescaled to the forward b0s' level by"] = round(f, 4)
        notes.append(f"the reversed series rescaled by {f:.4g} to the forward b0s' brain median (its stored scale is not comparable)")
    if not np.allclose(img.affine, A, atol=1e-3):
        notes.append(f"the reversed series was put onto the APP grid by the scanner's coordinates: {how}")
    return M.Subject(name=f"{subject}_{session}_{key}", dwi=dwi, affine=A, vox=np.asarray(first.header.get_zooms()[:3], float),
                     bval=bval, bvec=bvec, pe_axis=pe_axis, pe_sign=float(pe_app[pe_axis]), pe_directions=(PE["APP"], PE["APA"]),
                     readout_s=M.NOMINAL_READOUT_S, b0s=np.concatenate([dwi[..., b0i].astype(np.float64), rvol[..., take]], -1),
                     pe_vectors=np.array([pe_app] * len(b0i) + [pe_apa] * len(take)), series=series, notes=notes)


def onto_coarser(vol, img, dst_affine, dst_shape, dst_vox):
    """A reversed series onto the APP grid (_masivar._onto: cubic, by the scanner's coordinates), box-averaged first
    along any axis where its voxel is at least twice finer than the APP's: a box of round(APP voxel / its voxel)
    voxels (scanner A's r06x06x25: 4 x 4 in-plane onto 2.5 mm, 3 x 3 onto 1.89 mm; the slices, 2.5 mm, are not
    averaged), so the cubic sampling does not alias the finer grid. An even box (scipy's uniform_filter: voxels i - b/2
    .. i + b/2 - 1) is centered half a voxel below i, so the source affine is shifted by that half voxel along such an
    axis (A32's 4 x 4: 0.3125 mm in-plane; A2000r19's 3 x 3 is centered). Returns (volume, how)."""
    from scipy.ndimage import uniform_filter
    src = np.asarray(img.header.get_zooms()[:3], float)
    box = [int(round(d / s)) if d / s >= 2 else 1 for s, d in zip(src, dst_vox)]
    affine = img.affine
    if max(box) > 1:
        vol = uniform_filter(vol, size=box + [1] * (vol.ndim - 3), mode="nearest")
        shift = np.eye(4)
        shift[:3, 3] = [-0.5 if b % 2 == 0 else 0.0 for b in box]
        affine = img.affine @ shift
        how = (f"box average {' x '.join(map(str, box))} voxels ({' x '.join(f'{b * s:g}' for b, s in zip(box, src))} mm; an even box's "
               "half-voxel offset taken into the affine), then cubic onto the APP grid by the scanner's coordinates")
    else:
        how = "cubic onto the APP grid by the scanner's coordinates" if not np.allclose(img.affine, dst_affine, atol=1e-3) else "same grid"
    return M._onto(vol, affine, dst_affine, dst_shape), how


def scanner_of(s):
    """The site and scanner from a loaded scan's name (<subject>_<session>_...): the session label's first three
    characters (s1A, s1B)."""
    return s.name.split("_")[1][:3]


def check(s, key_or_protocol):
    """calibration_transfer.check against a cohort II descriptor, and its scanner item: {item: (found, ok)}."""
    pr = PROTOCOLS[key_or_protocol]["check"] if isinstance(key_or_protocol, str) else key_or_protocol
    out = CT.check(s, pr)
    want = pr.get("site and scanner (BIDS session label)")
    if want is not None:
        out["site and scanner (BIDS session label)"] = (scanner_of(s), scanner_of(s) == want)
    want = pr.get("reversed series voxel_mm")
    if want is not None:
        apa = next(x for x in s.series if "APA" in x["file"])
        found = apa.get("voxel_mm")
        out["reversed series voxel_mm"] = (found, found is not None and bool(np.allclose(found, want, atol=pr["voxel_tolerance_mm"])))
    return out


def load_checked(subject, session, key):
    s = load(subject, session, key)
    bad = {k: v for k, (v, ok) in check(s, key).items() if not ok}
    if bad:
        raise ValueError(f"{s.name}: not protocol {key}: {bad} (expected { {k: PROTOCOLS[key]['check'][k] for k in bad} })")
    return s


def pipeline_settings(key):
    s = CR.pipeline_settings()
    s["track"] = {**s["track"], "shell": PROTOCOLS[key]["tracked shell"]}
    s["series joined"] = PROTOCOLS[key]["series"]
    return s


def module_sha256():
    return CR.sha256(Path(__file__).resolve())


# ----------------------------------------------------------------------------------------------- the runs

class Records(CR.Records):
    def __init__(self, subject, key, smoke=False):
        self.subject, self.pair = subject, pairs(key)[subject]
        self.dir = DATA / "ensemble" / subject / "card" / (self.pair.name + ("-smoke" if smoke else ""))
        self.reuse = None


class Runner(CR.Runner):
    """calibration_runs.Runner at a cohort II protocol: its own records, loader, check and tracked shell."""
    def __init__(self, subject, key, smoke=False):
        self.key, self.shell = key, PROTOCOLS[key]["tracked shell"]
        self.R = Records(subject, key, smoke)
        self.subject, self.pair = subject, self.R.pair
        self.R.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.R.dir / "runs.json"
        self.log = json.loads(self.log_path.read_text()) if self.log_path.exists() else {}
        self.log.setdefault("created", {"git": CR.git_state(), "calibration_cohortII.py sha256": module_sha256(),
                                        "pipeline": pipeline_settings(key), "protocol": {"key": key, **PROTOCOLS[key]}, "smoke": smoke, "pair": {
            "reference": self.pair.ref, "same session": self.pair.same, "other sessions": self.pair.other,
            "reuse": None, "notes": list(self.pair.notes)}})
        self.labeler = None

    def one(self, s, name, corrected=None, extra=None, before_record=None):
        """calibration_runs.Runner.one, tracked at the protocol's shell."""
        CR.guard_disk()
        path = self.R.new(name)
        if self.labeler is None:
            self.labeler = P.default_labeler(CR.DEVICE)
        timer, t0 = P.Timer(), time.time()
        with P.exact_float32(CR.DEVICE):
            corr = P.correct(s, timer, CR.DEVICE) if corrected is None else P.Correction(corrected, None, None, None, applied=True, note="given")
            tg = P.track(s, corr.dwi, timer, device=CR.DEVICE, shell=self.shell)
            with timer("label"):
                labels = self.labeler(tg.fibers, draws=(0,), logp=True)
        if before_record is not None:
            before_record(corr, tg)
        tmp = path.with_name(path.stem + ".partial.npz")
        E.RunRecord.of(tg, labels, s.affine, s.dwi.shape[:3]).save(tmp, compact=True)
        os.replace(tmp, path)
        self.log[name] = {"fibers": len(tg.fibers), "labeled": int(labels.keep.sum()), "residual_left": corr.residual_left,
                          "field_hz_p99_abs": round(float(np.quantile(np.abs(corr.field_hz), 0.99)), 2) if corr.field_hz is not None else None,
                          "seconds": timer.seconds, "total_s": round(time.time() - t0, 1), **(extra or {})}
        self.save_log()
        print(self.subject, self.key, name, json.dumps(self.log[name]), flush=True)
        P.release_memory(CR.DEVICE)
        return corr, tg

    def reference(self):
        """calibration_runs.Runner.reference with this protocol's loader: A and its companions, all three or none."""
        A = load_checked(self.subject, self.pair.ref[0], self.key)
        parts = {"A": self.R.have("A"), "A_meanb0": self.R.have("A_meanb0", ".npy"), "A_mask": self.R.have("A_mask", ".npy")}
        if not any(parts.values()):
            self.log["A_scan"] = {"name": A.name, "series": A.series, "notes": A.notes, "shape": list(A.dwi.shape)}

            def companions(corr, tg):
                for name, arr in (("A_meanb0", mean_b0(corr.dwi, A.bval)), ("A_mask", tg.mask)):
                    p = self.R.new(name, ".npy")
                    tmp = p.with_name(p.stem + ".partial.npy")
                    np.save(tmp, arr)
                    os.replace(tmp, p)
            self.one(A, "A", before_record=companions)
        elif not all(parts.values()):
            raise RuntimeError(f"{self.subject}: A's records are incomplete {parts}: remove the companions of a missing A.npz by hand")
        return A, np.load(self.R.path("A_meanb0", ".npy")), np.load(self.R.path("A_mask", ".npy"))

    def reg3_inputs(self, tag):
        """Mode 3's shortcuts: (the rescan's corrected mean b0 if a record already holds it, else None; the other
        direction's to_A and where it is from, or (None, None)). For B2000's session2: the swapped folder's A_meanb0
        (session 2 corrected as here) and its session1 transform."""
        d = swap_dir(self.subject, self.key)
        if self.key == "B2000" and tag == "session2" and (d / "runs.json").exists() and (d / "A_meanb0.npy").exists():
            T, src = best_to_A(json.loads((d / "runs.json").read_text()), "session1")
            return np.load(d / "A_meanb0.npy"), T, (f"{d.name} {src}" if src else None)
        return None, None, None

    def register3(self, A, ref_b0, tag):
        """Mode 3's transform for a rescan (no tracking): the entry <tag>reg3_scan, its decision against mode 2."""
        name = rescan_tag(tag, 3)
        if f"{name}_scan" in self.log:
            return self.log[f"{name}_scan"]
        t0 = time.time()
        B = load_checked(self.subject, self.pair.rescans[tag][0], self.key)
        moving_b0, other, other_from = self.reg3_inputs(tag)
        if moving_b0 is None:
            with P.exact_float32(CR.DEVICE):
                cb = P.correct(B, P.Timer(), CR.DEVICE)
            P.release_memory(CR.DEVICE)
            moving_b0 = mean_b0(cb.dwi, B.bval)
            how = "corrected here"
        else:
            how = f"the corrected mean b0 from {swap_dir(self.subject, self.key).name}/A_meanb0.npy (the same scan and correction)"
        r = register_multi(B, moving_b0, A, ref_b0, other)
        pts = mask_world(np.load(self.R.path("A_mask", ".npy")), A.affine)
        e = reg3_entry(r, self.log.get(f"{rescan_tag(tag, 2)}_scan"), tag, pts, other_from)
        e.update({"name": B.name, "moving mean b0": how, "register3_s": round(time.time() - t0, 1)})
        self.log[f"{name}_scan"] = e
        self.save_log()
        print(self.subject, self.key, name, e["decision"], "best", e["best"], e["best b0 ncc"], "QC flag", e["QC"]["flag"],
              json.dumps(e.get("against registration 2")), flush=True)
        return e

    def rescan(self, A, ref_b0, tag, registration=1):
        """calibration_runs.Runner.rescan with this protocol's loader: <tag>_onA (registration 1, as before), or
        <tag>reg2_onA (registration 2: FA onto FA, then the mean b0s), or registration 3 (register3's entry; tracked as
        <tag>reg3_onA only if it differs from mode 2's; see REGISTRATION)."""
        name = rescan_tag(tag, registration)
        if registration == 3:
            e = self.register3(A, ref_b0, tag)
            if e["record"] != f"{name}_onA" or self.R.have(f"{name}_onA"):
                return
            B = load_checked(self.subject, self.pair.rescans[tag][0], self.key)
            timer, t0 = P.Timer(), time.time()
            with P.exact_float32(CR.DEVICE):
                cb = P.correct(B, timer, CR.DEVICE)
            P.release_memory(CR.DEVICE)
            dwi, bvec = onto_reference(cb.dwi, B.bvec, B.affine, np.array(e["to_A"]), A.affine, A.dwi.shape[:3])
            e.update({"series": B.series, "shape": list(B.dwi.shape), "residual_left": cb.residual_left,
                      "field_hz_p99_abs": round(float(np.quantile(np.abs(cb.field_hz), 0.99)), 2), "correct_seconds": timer.seconds,
                      "correct_resample_s": round(time.time() - t0, 1), "term": "other session"})
            self.save_log()
            self.one(replace(A, dwi=dwi, bval=B.bval, bvec=bvec, b0s=None), f"{name}_onA", corrected=dwi)
            return
        if self.R.have(f"{name}_onA"):
            return
        B = load_checked(self.subject, self.pair.rescans[tag][0], self.key)
        timer, t0 = P.Timer(), time.time()
        with P.exact_float32(CR.DEVICE):
            cb = P.correct(B, timer, CR.DEVICE)
        P.release_memory(CR.DEVICE)
        extra = {"registration mode": registration, "registration": REGISTRATION[registration]}
        if registration == 1:
            to_ref, ncc = register(mean_b0(cb.dwi, B.bval), B.affine, ref_b0, A.affine)
        else:
            r = register_fa(B, mean_b0(cb.dwi, B.bval), A, ref_b0)
            to_ref, ncc = r["to_static"], r["ncc"]
            init = r["init_to_static"]
            extra.update({"_register_fa.py sha256": CR.sha256(BENCH / "_register_fa.py"),
                     "FA stage": {"to_A": np.round(init, 5).tolist(),
                                  "rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(init[:3, :3]) - 1) / 2, -1, 1)))), 2),
                                  "translation_mm": round(float(np.linalg.norm(init[:3, 3])), 2), "b0 ncc": round(r["init_ncc"], 4),
                                  "FA ncc": round(r["fa_ncc_init"], 4)},
                     "FA ncc after": round(r["fa_ncc"], 4)})
            if f"{tag}_scan" in self.log:                     # mode 1's transform, for the record
                old = np.array(self.log[f"{tag}_scan"]["to_A"])
                d = to_ref @ np.linalg.inv(old)
                extra["against registration 1"] = {"rotation_deg": round(float(np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))), 2),
                                                   "translation_mm (of the difference)": round(float(np.linalg.norm(d[:3, 3])), 2),
                                                   "registration 1 ncc_after": self.log[f"{tag}_scan"]["ncc_after"]}
        rot = np.degrees(np.arccos(np.clip((np.trace(to_ref[:3, :3]) - 1) / 2, -1, 1)))
        dwi, bvec = onto_reference(cb.dwi, B.bvec, B.affine, to_ref, A.affine, A.dwi.shape[:3])
        tag = name
        self.log[f"{tag}_scan"] = {"name": B.name, "term": "same session" if tag in self.pair.same else "other session", **extra,
                                   "series": B.series, "shape": list(B.dwi.shape), "residual_left": cb.residual_left,
                                   "field_hz_p99_abs": round(float(np.quantile(np.abs(cb.field_hz), 0.99)), 2),
                                   "correct_seconds": timer.seconds, "correct_register_resample_s": round(time.time() - t0, 1),
                                   "to_A": np.round(to_ref, 5).tolist(), "rotation_deg": round(float(rot), 2),
                                   "translation_mm": round(float(np.linalg.norm(to_ref[:3, 3])), 2), "ncc_after": round(ncc, 4)}
        self.save_log()
        self.one(replace(A, dwi=dwi, bval=B.bval, bvec=bvec, b0s=None), f"{tag}_onA", corrected=dwi)


def run(args):
    for subject in args.subject:
        t0 = time.time()
        rn = Runner(subject, args.protocol, args.smoke)
        A, ref_b0, mask = rn.reference()
        if getattr(args, "register_only", False):
            assert args.registration == 3, "--register-only is mode 3's"
            for tag in rn.pair.rescans:
                rn.register3(A, ref_b0, tag)
            continue
        for tag in rn.pair.rescans:
            rn.rescan(A, ref_b0, tag, args.registration)
        rn.replicates(A, mask, args.runs)
        print(f"{subject} {args.protocol} done {time.time() - t0:.0f}s, {CR.free_gb():.1f} GB free", flush=True)


# ----------------------------------------------------------------------------------------------- the statements

def collect(args):
    """calibration_transfer.collect for a cohort II protocol and subject."""
    import nibabel as nib
    from ensemble_grid_check import Grid
    for subject in args.subject:
        t0 = time.time()
        R = Records(subject, args.protocol, args.smoke)
        pair = R.pair
        reg = getattr(args, "registration", 1)
        out = R.dir / statements_name(args.runs + 1, reg)
        if out.exists() and not args.force:
            raise FileExistsError(f"{out} exists (--force to rebuild it)")
        paths = R.ensemble("", args.runs)
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{subject}: {len(missing)} of A's {len(paths)} runs missing, the first {missing[0]}")
        log = json.loads((R.dir / "runs.json").read_text())
        tags = [rescan_tag(t, reg) for t in pair.rescans if R.have(record_of(log, rescan_tag(t, reg)))]
        img = nib.load(series_files(subject, pair.ref[0], args.protocol)["APP"][0][1])
        A0 = E.RunRecord.load(paths[0], logp=False)
        grid = Grid("dwi", img.affine, A0.shape)
        mask = np.load(R.path("A_mask", ".npy"))
        cent = CC.centers(mask, A0.counts, A0.shape)
        Mx = CC.sphere_matrix(cent, A0.shape, grid.h)
        keys = A0.keys
        print(f"{subject} {args.protocol}: R = {len(paths)}, rescans {tags}, {len(cent)} points", flush=True)
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
        for tag in tags:
            k, t, pres, main = CC.run_stats(R.path(record_of(log, tag)), grid, Mx)
            arrays[f"s_tract_{tag}"] = CC.on_keys(keys, k, t)[s_ok]
            arrays[f"p_main_{tag}"] = main
            arrays[f"e_hit_{tag}"] = CC.vote_hist(A.votes, pres, Rn)[1]
            arrays[f"e_Ahit_{tag}"] = (A.own & pres).sum(1).astype(np.int64)
            print(f"  {tag}_onA {time.time() - t0:.0f}s", flush=True)
        log = json.loads((R.dir / "runs.json").read_text())
        meta = {"subject": subject, "protocol": args.protocol, "pair": pair.name, "R": Rn, "runs": [str(p.relative_to(DATA)) for p in paths],
                "rescans": {t: {"term": "other session", "scan": log[f"{t}_scan"]["name"],
                                "rotation_deg": log[f"{t}_scan"]["rotation_deg"], "translation_mm": log[f"{t}_scan"]["translation_mm"],
                                "ncc_after": log[f"{t}_scan"]["ncc_after"], "registration mode": log[f"{t}_scan"].get("registration mode", 1),
                                "record": record_of(log, t)}
                            for t in tags},
                "registration mode": reg,
                "pooled": {}, "points": int(len(cent)), "seeds labeled in all R runs": int(s_ok.sum()),
                "records git at creation": log["created"]["git"], "collect git": CR.git_state(),
                "calibration_cohortII.py sha256": module_sha256(), "seconds": round(time.time() - t0, 1)}
        np.savez_compressed(out, meta=np.array(json.dumps(meta)), **arrays)
        print(json.dumps(meta, indent=1), flush=True)


# ----------------------------------------------------------------------------------------------- plan

SECONDS = {   # per run on the M2, cIIs01's smoke runs (2026-10-19; the machine not idle, UKF 93-131 s a run): a rescan is its
             # correction and registration (25 s) and its _onA run (145 s); a replicate the mean of r001 and r002
    "B2000": {"A": 152, "rescan": 170, "replicate": 126},
    "B2000J": {"A": 117, "rescan": 197, "replicate": 151},
    "B32": {"A": 66, "rescan": 94, "replicate": 96},
    # scanner A, cIIs01's R = 2 smoke runs (2026-10-19), sharing the GPU with the scanner B queue
    "A32": {"A": 87, "rescan": 100, "replicate": 83},
    "A2000r19": {"A": 213, "rescan": 221, "replicate": 207},   # about 2x the streamlines: every mask voxel a seed, 2x the voxels
}


def inventory():
    """Per subject and session on disk: the 2.5 mm series (run, acquisition, volumes, b0s), and the scan numbers."""
    import nibabel as nib
    out = {}
    for d in sorted(M.ROOT.glob("sub-cIIs*/ses-*/dwi")):
        rows, scans = [], set()
        for p in sorted(d.glob("*_dwi.nii.gz"), key=lambda p: NAME.search(p.name).group(5) + NAME.search(p.name).group(6)):
            m = NAME.search(p.name)
            bval = np.loadtxt(M._side(p, ".bval"), ndmin=1)
            scans.add(int(m.group(5)))
            rows.append({"run": int(m.group(5) + m.group(6)), "series": f"b{m.group(1)}n{m.group(2)} {m.group(4)}", "res": m.group(3),
                         "shape": list(nib.load(p).shape), "b0s (stated b < 50) at": np.flatnonzero(bval < 50).tolist()})
        out[f"{d.parent.parent.name[4:]} {d.parent.name[4:]}"] = {"scans": sorted(scans), "series": rows}
    return out


def plan(args):
    inv = inventory()
    for k, v in inv.items():
        print(f"{k}: scans {v['scans']}; " + "; ".join(f"{r['run']} {r['series']} {r['shape'][-1]} vol, b0 at {r['b0s (stated b < 50) at']}" for r in v["series"]))
    for key, proto in PROTOCOLS.items():
        ps = pairs(key)
        secs = 0
        print(f"\n{key}: {proto['note']}")
        for sub, pair in ps.items():
            R = Records(sub, key)
            rlog = json.loads((R.dir / "runs.json").read_text()) if (R.dir / "runs.json").exists() else {}
            need = {"A": int(not R.have("A")), "rescan": sum(not R.have(record_of(rlog, rescan_tag(t, DEFAULT_REGISTRATION))) for t in pair.rescans),
                    "replicate": sum(not R.have(f"{CR.FLIP}/r{r:03d}") for r in range(1, args.runs + 1))}
            try:
                f = {ses: series_files(sub, ses, key) for ses in (pair.ref[0], pair.rescans["session2"][0])}
                files = "; ".join(f"{ses}: APP runs {[int(NAME.search(p.name).group(5) + NAME.search(p.name).group(6)) for _, p in x['APP']]}, APA {x['APA run']} of {x['APA candidates']}" for ses, x in f.items())
            except FileNotFoundError as e:
                files = f"cannot run: {e}"
            print(f"  {sub}: A {pair.ref[0]}, rescan {pair.rescans['session2'][0]}; to run {need}; {files}")
            if not files.startswith("cannot") and key in SECONDS:
                secs += sum(SECONDS[key][k] * v for k, v in need.items())
        if secs:
            print(f"  about {secs / 3600:.1f} h on the M2 (SECONDS)")
    if args.check:
        from _masivar import load as load_iii
        print("\nchecks (each scan against every descriptor; cohort III's is calibration_runs.PROTOCOL, cIs1's calibration_transfer.PROTOCOL_CIS1):")
        others = {"cohort III": CR.PROTOCOL, "cIs1": CT.PROTOCOL_CIS1}
        scans = []
        for key in PROTOCOLS:
            for sub in SUBJECTS:
                for ses in SESSIONS[PROTOCOLS[key]["scanner"][-1]]:
                    try:
                        scans.append(load(sub, ses, key))
                    except (FileNotFoundError, AssertionError) as e:
                        print(f"  {sub} {ses} {key}: not loadable: {e}")
        scans += [load_iii("cIIIs03", "s1Bx1", 1, CR.SHELLS), load_iii("cIs1", "s1Ax1", 1, CT.SHELLS)]
        for s in scans:
            verdict = {k: not [i for i, (_, ok) in check(s, k).items() if not ok] for k in PROTOCOLS}
            verdict.update({k: not [i for i, (_, ok) in CT.check(s, pr).items() if not ok] for k, pr in others.items()})
            apa = next(x for x in s.series if "APA" in x["file"])
            print(f"  {s.name}: accepted by {[k for k, ok in verdict.items() if ok]}; forward b0s {len(s.b0_index)}, reversed b0 over forward "
                  f"{[apa['over_app_b0s'][i] for i in apa['taken_as_b0s']]}" + (f", last volume over DW {[x['last_volume_over_dw'] for x in s.series if 'last_volume_over_dw' in x]}" if "last_volume_over_dw" in s.series[0] else ""), flush=True)
    print(f"\n{CR.free_gb():.1f} GB free")


# ----------------------------------------------------------------------------------------------- evaluation

OUT = BENCH / "results/calibration_cohortII.json"
CARD_PATH = BENCH / "results/calibration_card_masivar_cIII.json"
TERM = "other session"                        # cohort II has one scan per session
EVALUATED = ("B2000", "A2000r19", "B32", "A32")
MAPPED = EVALUATED                            # (d) and (e): a map per protocol (each scanner's two), leave one subject out,
                                              # and its learning curve
CROSS = (("B32", "A32", "fixed protocol (b1000n32, 2.5 mm; the reversed series' resolution differs: 2.5 mm on B, 0.625 x 0.625 x 2.5 on A)"),
         ("B2000", "A2000r19", "NOT a fixed protocol: b2000n96 on both, but 2.5 mm on B and 1.89 x 1.89 x 2.2 mm on A, so scanner "
                               "and resolution are confounded (and with them the seeds per mm^3, every mask voxel being a seed)"))
SCHEMA = "tractline.calibration_cohortII/1"    # ismrm2027_figures.COHORT2_SCHEMA (fig6_protocol_gap reads it)
SCHEMA_NAME = {"B2000": "B2000", "A2000r19": "A2000", "B32": "B32", "A32": "A32"}   # the protocol names the figure uses
SCHEMA_LEVELS = np.round(np.arange(0.5, 0.951, 0.05), 3)


def coverage_at_targets(train, test, levels=SCHEMA_LEVELS):
    """A map fitted on `train` (calibration_card.fit_term's envelope curve) applied to `test`: per target, the test
    subject's coverage at the map's stated level for it, or its widest outline's where the map never reaches the
    target (reached false)."""
    curve = CC.fit_term(train, TERM)["_curve"]
    own = CC.term_curve(test, TERM)
    widest = float(CC.term_shapes(test, TERM)["widest"]["coverage"])
    cov, reached = [], []
    for q in levels:
        lv, ok, _ = CC.level_for(curve, float(q))
        cov.append(round(float(own[np.flatnonzero(CC.GRID == lv)[0]]) if lv < 1 else widest, 4))
        reached.append(bool(ok))
    return cov, reached


def schema_part(S):
    """The keys COHORT2_SCHEMA defines: per protocol the as-stated curve and the leave-one-out map's, and the
    transfers: per target subject, a map fitted on the other protocol's subjects WITHOUT the target person (the same
    people are on both scanners), applied to the target's pair; the numbers of (c)'s "source map, without the target
    person". The map fitted on all source subjects (the target person included) is kept beside it as an extra key."""
    mean = lambda d: [round(float(x), 4) for x in np.mean(list(d.values()), 0)] if d else None  # noqa: E731
    prot = {}
    for k in EVALUATED:
        pr = PROTOCOLS[k]
        stated = {sub: [at(CC.term_curve(s, TERM), float(q)) for q in SCHEMA_LEVELS] for sub, s in S[k].items()}
        mapped = None
        if k in MAPPED and len(S[k]) >= 2:
            per, reached = {}, {}
            for sub, s in S[k].items():
                per[sub], reached[sub] = coverage_at_targets([x for m, x in S[k].items() if m != sub], s)
            mapped = {"per subject": per, "mean": mean(per), "reached": reached}
        directions = pr["check"]["volumes per shell"][str(int(pr["tracked shell"]))]
        prot[SCHEMA_NAME[k]] = {"scanner": pr["scanner"][-1], "voxel_mm": pr["check"]["voxel_mm"], "tracked shell": int(pr["tracked shell"]),
                                "directions": directions, "subjects": list(S[k]), "as stated": {"per subject": stated, "mean": mean(stated)},
                                "map (leave one subject out)": mapped, "protocol key (calibration_cohortII.PROTOCOLS)": k}
    trans = {}
    for a, b, what in CROSS:
        conf = None if a == "B32" else "resolution: 2.5 mm against 1.9 mm (and about twice the streamlines on A2000: every mask voxel a seed)"
        for src, dst in ((a, b), (b, a)):
            if not S[src] or not S[dst] or len({s.R for s in list(S[src].values()) + list(S[dst].values())}) != 1:
                continue
            fitted_on = {sub: [m for m in S[src] if m != sub] for sub in S[dst]}
            per = {sub: coverage_at_targets([S[src][m] for m in fitted_on[sub]], s)[0] for sub, s in S[dst].items() if fitted_on[sub]}
            per_all = {sub: coverage_at_targets(list(S[src].values()), s)[0] for sub, s in S[dst].items()}
            trans[f"{SCHEMA_NAME[src]} -> {SCHEMA_NAME[dst]}"] = {
                "map from": SCHEMA_NAME[src], "applied to": SCHEMA_NAME[dst], "map fitted on": list(S[src]),
                "target person excluded": True, "map fitted on, per target subject": fitted_on,
                "confounded": conf, "per subject": per, "mean": mean(per),
                "map fitted on all source subjects, target person included (extra; not drawn)": {"per subject": per_all, "mean": mean(per_all)},
                "note": "per target subject, the source map is fitted on the source protocol's other subjects: the same people are on both "
                        "scanners, so the target person is left out (as (c)'s \"source map, without the target person\"); the all-subject "
                        "fit, which has seen the target person on the other scanner, is the extra key beside it"}
    return {"schema": SCHEMA, "mock": False, "levels": SCHEMA_LEVELS.tolist(), "protocols": prot, "transfer (fixed protocol, other scanner)": trans}


def statements(key, runs, smoke, registration=DEFAULT_REGISTRATION):
    """{subject: calibration_card.Subject} for the protocol's pairs with a statements file of that registration mode."""
    out = {}
    for sub in pairs(key):
        p = statements_file(Records(sub, key, smoke).dir, runs + 1, registration)
        if p.exists():
            out[sub] = CC.Subject(p)
            out[sub].path = p
    return out


def at(curve, q):
    return round(float(curve[np.flatnonzero(CC.GRID == round(q, 3))[0]]), 4)


def as_stated(s):
    """One subject's other-session rescan, no map: ECE (streamline, point) and envelope coverage at stated = target."""
    R = s.R
    v, h = s.streamline(TERM)
    vp, hp = s.point(TERM)
    curve = CC.term_curve(s, TERM)
    cov = {f"{q:g}": at(curve, q) for q in CC.TARGETS}
    return {"streamline": CC.ece(v / R, h), "point": CC.ece(vp / R, hp), "envelope coverage at stated level": cov,
            "envelope gap (coverage - stated)": {q: round(c - float(q), 4) for q, c in cov.items()},
            "streamline: the rescan labels the seed": round(float(np.mean(s.z[f"s_tract_{s.tags(TERM)[0]}"] >= 0)), 4)}


def mean_of(rows, f):
    x = [f(r) for r in rows]
    return round(float(np.mean(x)), 4) if x else None


def summarize_as_stated(per):
    rows = list(per.values())
    return {"subjects": len(rows),
            "streamline ECE": mean_of(rows, lambda r: r["streamline"]["ece"]),
            "streamline stated / observed": [mean_of(rows, lambda r: r["streamline"]["stated"]), mean_of(rows, lambda r: r["streamline"]["observed"])],
            "point ECE": mean_of(rows, lambda r: r["point"]["ece"]),
            "point stated / observed": [mean_of(rows, lambda r: r["point"]["stated"]), mean_of(rows, lambda r: r["point"]["observed"])],
            "envelope coverage at stated level": {f"{q:g}": mean_of(rows, lambda r: r["envelope coverage at stated level"][f"{q:g}"]) for q in CC.TARGETS},
            "envelope gap (coverage - stated)": {f"{q:g}": mean_of(rows, lambda r: r["envelope gap (coverage - stated)"][f"{q:g}"]) for q in CC.TARGETS},
            "between-subject range of the envelope gap at 0.9": [min(r["envelope gap (coverage - stated)"]["0.9"] for r in rows),
                                                                  max(r["envelope gap (coverage - stated)"]["0.9"] for r in rows)] if rows else None}


def card_applied(s, card):
    """The cohort III map ("other session" term) applied to a subject's statements (b)."""
    t = card["terms"][TERM]
    R = s.R
    if R != card["validity"]["match"]["R"]:
        return {"skipped": f"R = {R}, the map's R is {card['validity']['match']['R']}"}
    v, h = s.streamline(TERM)
    vp, hp = s.point(TERM)
    curve = CC.term_curve(s, TERM)
    env = {}
    for q in CC.TARGETS:
        lv = t["envelope"]["stated level for target coverage"][f"{q:g}"]["level"]
        env[f"{q:g}"] = {"map: stated level used": lv,
                         "coverage": at(curve, lv) if lv < 1 else round(float(CC.term_shapes(s, TERM)["widest"]["coverage"]), 4)}
        env[f"{q:g}"]["gap (coverage - target)"] = round(env[f"{q:g}"]["coverage"] - q, 4)
    return {"streamline": CC.ece(CC.apply_votes(t["streamline"], v, R), h), "point": CC.ece(CC.apply_votes(t["point"], vp, R), hp),
            "envelope at target": env}


def card_match(key, sub, smoke, card, R):
    """(b): the cohort III map's match (calibration_transfer.match, as written) on a subject's A scan of a protocol, with
    the settings its records were made with (runs.json), its statements' R, the labeler's weights and the term."""
    from tractline.labelers.rapidparc import HASHES
    rec = Records(sub, key, smoke)
    settings = json.loads((rec.dir / "runs.json").read_text())["created"]["pipeline"]
    m = CT.match(card, load(sub, rec.pair.ref[0], key), settings, R, HASHES["rapidparc"], [TERM])
    bad = {k: {x: v[x] for x in ("found", "expected")} for k, v in m["items"]["protocol"]["items"].items() if not v["ok"]}
    return {"scan": m["scan"], "verdict": m["verdict"], "failed": m["failed"], "protocol items failed": bad,
            "other items": {k: v["ok"] for k, v in m["items"].items() if k != "protocol"}}


def streamline_counts(key, subjects, smoke):
    """Streamlines per run (runs.json): A's, the replicates' mean and the rescan's, per subject and their means."""
    per = {}
    for sub in subjects:
        log = json.loads((Records(sub, key, smoke).dir / "runs.json").read_text())
        reps = [v["fibers"] for k, v in log.items() if k.startswith(f"{CR.FLIP}/r") and isinstance(v, dict) and "fibers" in v]
        per[sub] = {"A": log["A"]["fibers"], "replicates (mean)": round(float(np.mean(reps)), 1) if reps else None,
                    "replicates": len(reps), "session2_onA": log.get("session2_onA", {}).get("fibers")}
    mean = lambda f: round(float(np.mean([f(x) for x in per.values() if f(x) is not None])), 1) if per else None  # noqa: E731
    return {"per subject": per, "mean": {"A": mean(lambda x: x["A"]), "replicates": mean(lambda x: x["replicates (mean)"]),
                                         "session2_onA": mean(lambda x: x["session2_onA"])}}


def summarize_mapped(rows, key="card"):
    """Mean over held-out subjects of calibration_card.evaluate's rows: ECE as stated / with the map, coverage at target."""
    rows = [r for r in rows if r]
    if not rows:
        return None
    return {"subjects": len(rows),
            "streamline ECE: as stated / map / PAT16 map": [mean_of(rows, lambda r: r["streamline"][k]["ece"]) for k in ("as stated", "card", "PAT16 map")],
            "point ECE: as stated / map": [mean_of(rows, lambda r: r["point"][k]["ece"]) for k in ("as stated", "card")],
            "envelope coverage at target: as stated / map": {
                f"{q:g}": [mean_of(rows, lambda r: r["envelope"][f"{q:g}"][f"as stated: coverage at stated {q:g}"]),
                           mean_of(rows, lambda r: r["envelope"][f"{q:g}"]["card: coverage"])] for q in CC.TARGETS},
            "envelope |coverage - target| with the map, mean over targets": mean_of(
                rows, lambda r: np.mean([abs(r["envelope"][f"{q:g}"]["card: coverage"] - q) for q in CC.TARGETS]))}


def loso(subjects, pat):
    """(d): per held-out subject, the map fitted on the others (calibration_card.evaluate)."""
    names = list(subjects)
    held = {n: CC.evaluate([subjects[m] for m in names if m != n], subjects[n], TERM, pat) if len(names) > 1 else None for n in names}
    return {"per subject": held, "summary": summarize_mapped(list(held.values()))}


def learning_curve(subjects, pat):
    """(e): calibration_learning_curve's scores for training sets of k = 1 .. n - 1 of the other subjects (every
    combination), on the held-out subject's rescans; k = 0 as stated, and PAT16's map."""
    import itertools
    import calibration_learning_curve as LC
    data = [LC.Data(s, TERM) for s in subjects.values()]
    widest = [d.widest() for d in data]
    n = len(data)
    out = {"subjects": [d.name for d in data],
           "as stated (k = 0)": LC.summarize([LC.score("as stated", d, w, pat) for d, w in zip(data, widest)]) if data else None,
           "PAT16 map": LC.summarize([LC.score("PAT16 map", d, w, pat) for d, w in zip(data, widest)]) if data else None, "k": {}}
    for k in range(1, n):
        rows = []
        for i, test in enumerate(data):
            others = [j for j in range(n) if j != i]
            for c in itertools.combinations(others, k):
                rows.append(LC.score(LC.fit([data[j] for j in c]), test, widest[i], pat))
        out["k"][str(k)] = {**LC.summarize(rows), "training sets per held-out subject": len(list(itertools.combinations(range(n - 1), k)))}
    return out


def cross(src, dst, pat):
    """(c): maps fitted on scanner `src`'s subjects applied to scanner `dst`'s, against dst's own leave-one-out map.
    Per target subject: the source map from all source subjects (the same person included: the map has seen this
    person on the other scanner), from the source subjects without that person, and dst's own map without that
    person (as (d)); all scored on the target's rescans by calibration_card.evaluate."""
    names = list(dst)
    Rs = {s.R for s in list(src.values()) + list(dst.values())}
    if len(Rs) != 1:
        return {"skipped": f"R differs between the statements ({sorted(Rs)})"}
    rows = {"source map, all source subjects": {}, "source map, without the target person": {}, "target scanner's own map, without the target person": {}}
    for n in names:
        rows["source map, all source subjects"][n] = CC.evaluate(list(src.values()), dst[n], TERM, pat) if src else None
        rest = [s for m, s in src.items() if m != n]
        rows["source map, without the target person"][n] = CC.evaluate(rest, dst[n], TERM, pat) if rest else None
        own = [dst[m] for m in names if m != n]
        rows["target scanner's own map, without the target person"][n] = CC.evaluate(own, dst[n], TERM, pat) if own else None
    return {"per target subject": rows, "summary": {k: summarize_mapped(list(v.values())) for k, v in rows.items()},
            "source subjects": list(src), "target subjects": names}


def refusals(card):
    """Each descriptor (and the cohort III map's match, and cIs1's protocol) against one scan of every protocol, the
    first pair's A, as that protocol loads it; and cohort III's and cIs1's scans against the cohort II descriptors."""
    from _masivar import load as load_iii
    from tractline.labelers.rapidparc import HASHES
    scans = {}
    for key in PROTOCOLS:
        for sub, pair in pairs(key).items():
            try:
                scans[key] = load(sub, pair.ref[0], key)
                break
            except (FileNotFoundError, AssertionError):
                continue
    if "A2000r25" not in scans:
        try:
            scans["A2000r25"] = load("cIIs00", "s1Ax1", "A2000r25")
        except (FileNotFoundError, AssertionError):
            pass
    scans["cohort III (cIIIs03 A)"] = load_iii(*CT.CONTROL[0:1], *CT.CONTROL[1], CR.SHELLS)
    scans["cIs1 A"] = load_iii("cIs1", *CT.PAIR.ref, CT.SHELLS)
    others = {"cohort III": CR.PROTOCOL, "cIs1": CT.PROTOCOL_CIS1}
    table = {}
    for name, s in scans.items():
        row = {k: not [i for i, (_, ok) in check(s, k).items() if not ok] for k in PROTOCOLS}
        row.update({k: not [i for i, (_, ok) in CT.check(s, pr).items() if not ok] for k, pr in others.items()})
        m = CT.match(card, s, {}, card["validity"]["match"]["R"], HASHES["rapidparc"], [TERM])
        row["cohort III map's match (protocol items only)"] = m["items"]["protocol"]["ok"]
        table[f"{name}: {s.name}"] = {"accepted by": [k for k, ok in row.items() if ok], "refused by": [k for k, ok in row.items() if not ok]}
    own = all(v["accepted by"] == [k.split(":")[0]] for k, v in table.items() if k.split(":")[0] in PROTOCOLS)
    foreign = {"cohort III (cIIIs03 A)": "cohort III", "cIs1 A": "cIs1"}
    theirs = all(set(v["accepted by"]) <= {foreign[k.split(":")[0]], "cohort III map's match (protocol items only)"}
                 for k, v in table.items() if k.split(":")[0] in foreign)
    return {"table": table, "each cohort II scan accepted by its own descriptor only": own,
            "cohort III's and cIs1's scans refused by every cohort II descriptor": theirs}


def provenance(key, subjects, smoke):
    out = {}
    for sub, s in subjects.items():
        R = Records(sub, key, smoke)
        log = json.loads((R.dir / "runs.json").read_text())
        out[sub] = {"records": str(R.dir.relative_to(DATA)), "records git at creation": log["created"]["git"],
                    "calibration_cohortII.py sha256 at creation": log["created"].get("calibration_cohortII.py sha256"),
                    "statements": getattr(s, "path", R.dir / statements_name(s.R, s.meta.get("registration mode", 1))).name,
                    "statements sha256": CR.sha256(getattr(s, "path", R.dir / statements_name(s.R, s.meta.get("registration mode", 1)))), "R": s.R,
                    "registration mode (of the statements)": s.meta.get("registration mode", 1),
                    "registration 3": {k: log.get("session2reg3_scan", {}).get(k) for k in ("decision", "record", "best", "best b0 ncc", "QC", "against registration 2")}
                                      if "session2reg3_scan" in log else None,
                    "rescan registration (rotation deg, translation mm, NCC)": [next(iter(s.meta["rescans"].values()))[k] for k in ("rotation_deg", "translation_mm", "ncc_after")]}
    return out


def evaluate(args):
    """(a)-(f) on the protocols' statements; writes results/calibration_cohortII.json. Every evaluated protocol must
    have all its pairs' statements at R = runs + 1, unless --dry-run (which must write elsewhere: --out)."""
    out_path = Path(args.out)
    if args.dry_run and out_path.resolve() == OUT.resolve():
        sys.exit("a dry run writes elsewhere: pass --out")
    card = json.loads(CARD_PATH.read_text())
    pat = CC.pat16_table()
    S = {k: statements(k, args.runs, args.smoke, args.registration) for k in EVALUATED}
    if args.dry_run:
        for k in EVALUATED:                           # a dry run takes whatever statements exist, at any R
            if not S[k]:
                for sub in pairs(k):
                    found = sorted(x for x in Records(sub, k, args.smoke).dir.glob("statements_R*.npz")
                                   if x.name.endswith(".npz" if args.registration == 1 else f"_reg{args.registration}.npz")
                                   and (args.registration != 1 or "_reg" not in x.name))
                    if found:
                        S[k][sub] = CC.Subject(found[-1])
    else:
        missing = {k: [s for s in pairs(k) if s not in S[k]] for k in EVALUATED}
        missing = {k: v for k, v in missing.items() if v}
        if missing:
            sys.exit(f"{statements_name(args.runs + 1, args.registration)} missing: {missing} (a partial evaluation only as --dry-run)")
    res = {**schema_part(S),
           "what": "MASiVar cohort II (site 1, scanners A and B): the as-stated gap on other protocols and scanners, the cohort III "
                   "map out of protocol, maps across scanners, per-scanner maps left one subject out and their learning curve",
           "dry run": bool(args.dry_run), "smoke records": bool(args.smoke), "term": TERM,
           "registration mode": args.registration, "registration": REGISTRATION[args.registration],
           "protocol descriptors": {k: {x: PROTOCOLS[k][x] for x in ("note", "resolution", "series", "tracked shell", "check")} for k in EVALUATED},
           "subjects": {k: list(v) for k, v in S.items()},
           "provenance": {k: provenance(k, v, args.smoke) for k, v in S.items()},
           "cohort III map": {"path": str(CARD_PATH.relative_to(BENCH.parent)), "card id": card["card id"], "sha256": CR.sha256(CARD_PATH)}}
    # (a) as stated, per protocol, beside cohort III's held-out as-stated numbers (other session)
    c3 = card["held-out evaluation (leave one subject out)"]["summary"][TERM]
    per = {k: {sub: as_stated(s) for sub, s in v.items()} for k, v in S.items()}
    res["(a) as stated"] = {"per protocol": {k: {"summary": summarize_as_stated(p), "per subject": p} for k, p in per.items()},
                            "cohort III, held out, as stated (from the map)": {
                                "streamline ECE": c3["streamline ECE: as stated / card / PAT16 map"][0], "point ECE": c3["point ECE: as stated / card"][0],
                                "envelope coverage at stated level": {q: x[0] for q, x in c3["envelope coverage at target, as stated / card"].items()},
                                "subjects": c3["subjects held out"]}}
    # (b) the cohort III map applied anyway; B2000 is the same scanner (B) with another protocol
    b = {}
    for k in EVALUATED:
        rows = {sub: card_applied(s, card) for sub, s in S[k].items()}
        ok = [r for r in rows.values() if "skipped" not in r]
        b[k] = {"per subject": rows, "summary": None if not ok else {
            "streamline ECE (map)": mean_of(ok, lambda r: r["streamline"]["ece"]), "point ECE (map)": mean_of(ok, lambda r: r["point"]["ece"]),
            "envelope coverage at target (map)": {f"{q:g}": mean_of(ok, lambda r: r["envelope at target"][f"{q:g}"]["coverage"]) for q in CC.TARGETS}}}
        if k == "B2000":
            b[k]["the map's match, per subject (A's scan)"] = {sub: card_match(k, sub, args.smoke, card, s.R) for sub, s in S[k].items()}
            b[k]["refused by the match: every subject"] = all(v["verdict"] == "refused" for v in b[k]["the map's match, per subject (A's scan)"].values())
    b["note"] = ("deliberately out of protocol: the map's match refuses B2000's scans (above; and every cohort II scan's protocol "
                 "items: see refusals), and the map is applied anyway. B2000 is cohort III's scanner B with another protocol; "
                 "A2000r19, B32 and A32 differ further (scanner and resolution; b-value and directions)")
    res["(b) cohort III map applied"] = b
    # (c) across scanners
    counts = {k: streamline_counts(k, S[k], args.smoke) for k in EVALUATED}
    res["(c) across scanners"] = {f"{s} -> {d}": {"what": what, **cross(S[s], S[d], pat)} for s, d, what in CROSS}
    res["(c) across scanners"].update({f"{d} -> {s}": {"what": what, **cross(S[d], S[s], pat)} for s, d, what in CROSS})
    ratio = lambda x, y: round(counts[x]["mean"]["A"] / counts[y]["mean"]["A"], 3) if counts[x]["per subject"] and counts[y]["per subject"] else None  # noqa: E731
    res["(c) across scanners"]["streamlines per run"] = {
        "per protocol": counts, "A2000r19 / B2000 (A's runs, mean)": ratio("A2000r19", "B2000"), "A32 / B32 (A's runs, mean)": ratio("A32", "B32"),
        "confound": ("B2000 <-> A2000r19: every mask voxel is a seed, and A2000r19's grid (1.89 x 1.89 x 2.2 mm) has about 1.9 x the "
                     "voxels of a 2.5 mm grid in the same brain, so its runs have about twice the streamlines (and seeds per mm^3): "
                     "the envelopes, the votes and the rescan's presence differ in sampling density as well as in scanner and "
                     "resolution. B32 <-> A32 track the same 2.5 mm grid (counts about equal).")}
    # (d), (e) per-scanner maps, leave one subject out, and the learning curve
    res["(d) map per protocol, leave one subject out"] = {k: loso(S[k], pat) for k in MAPPED}
    res["(e) learning curve"] = {k: learning_curve(S[k], pat) for k in MAPPED}
    # (f) 32 directions against 96, as stated, paired by subject
    f = {}
    for k32, k96, what in (("B32", "B2000", "scanner B, both 2.5 mm: b1000 x32 against b2000 x96"),
                           ("A32", "A2000r19", "scanner A: b1000 x32 at 2.5 mm against b2000 x96 at 1.89 x 1.89 x 2.2 mm (resolution confounded)")):
        common = [s for s in per[k32] if s in per[k96]]
        d = lambda g: round(float(np.mean([g(per[k32][s]) - g(per[k96][s]) for s in common])), 4) if common else None  # noqa: E731
        f[f"{k32} - {k96}"] = {"what": what, "subjects": common,
                               "streamline ECE difference": d(lambda r: r["streamline"]["ece"]), "point ECE difference": d(lambda r: r["point"]["ece"]),
                               "envelope gap difference": {f"{q:g}": d(lambda r: r["envelope gap (coverage - stated)"][f"{q:g}"]) for q in CC.TARGETS}}
    res["(f) 32 directions against 96, as stated"] = f
    res["refusals"] = refusals(card)
    res["evaluate git"] = CR.git_state()
    res["calibration_cohortII.py sha256"] = module_sha256()
    res["created (UTC)"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["caveats"] = [
        "one other-session rescan per subject and scanner (no same-session repeats in cohort II); 4 subjects a scanner: means, no intervals",
        "the same four people on both scanners: (c) reports maps with and without the target person",
        "B2000 -> A2000r19 and back confound scanner with resolution (2.5 against 1.89 x 1.89 x 2.2 mm) and with the streamline count "
        "(about twice on A2000r19: every mask voxel a seed; see (c) streamlines per run); B32 <-> A32 is the fixed-protocol pair",
        "scanner A's reversed series is r06x06x25 (box-averaged onto the APP grid) and stored on another scale (rescaled to the forward b0s)",
        "every APP series has one b0; the reversed b0's contrast differs from the forward b0's (cohort II has no sidecars to say why)",
        "the point lattice is every 4 voxels: 10 mm at 2.5 mm, 7.6 x 7.6 x 8.8 mm at 1.89 x 1.89 x 2.2 (cohort III 8.6 x 8.6 x 8.8)"]
    out_path.write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    short = {"(a)": {k: {m: v["summary"][m] for m in ("streamline ECE", "point ECE", "envelope gap (coverage - stated)")} for k, v in res["(a) as stated"]["per protocol"].items()},
             "(b)": {k: v["summary"] for k, v in b.items() if k != "note"},
             "(c)": {k: (v.get("summary") or v.get("skipped")) for k, v in res["(c) across scanners"].items() if k != "streamlines per run"},
             "(c) streamlines": {k: v["mean"] for k, v in counts.items()},
             "(d)": {k: v["summary"] for k, v in res["(d) map per protocol, leave one subject out"].items()},
             "(e) k": {k: {n: {m: x[m]["mean"] if isinstance(x.get(m), dict) and "mean" in x[m] else x.get(m) for m in ("streamline ECE", "coverage error at 0.9")}
                           for n, x in v["k"].items()} for k, v in res["(e) learning curve"].items()},
             "(b) B2000 refused by the match": b["B2000"].get("refused by the match: every subject"),
             "refusals strict": [res["refusals"]["each cohort II scan accepted by its own descriptor only"],
                                 res["refusals"]["cohort III's and cIs1's scans refused by every cohort II descriptor"]]}
    print(json.dumps(short, indent=1, default=str))
    print(f"wrote {out_path}")


def compare(args):
    """Registration modes side by side (--modes; default 2 and 3) on the protocols' full statements (R = runs + 1): as stated, the map left one subject
    out at 0.8, 0.9 and 0.95, the widest outline's coverage, and each subject's registration; printed, and written to
    --out if given."""
    pat = CC.pat16_table()
    res = {}
    for key in args.protocol_list:
        row = {}
        for reg in args.modes:
            S = statements(key, args.runs, False, reg)
            if not S:
                row[f"registration {reg}"] = None
                continue
            per = {sub: as_stated(s) for sub, s in S.items()}
            st = summarize_as_stated(per)
            lo = loso(S, pat)["summary"] or {}
            row[f"registration {reg}"] = {
                "subjects": list(S),
                "as stated: coverage at 0.8 / 0.9 / 0.95": [st["envelope coverage at stated level"][q] for q in ("0.8", "0.9", "0.95")],
                "as stated: streamline ECE, point ECE": [st["streamline ECE"], st["point ECE"]],
                "map, leave one subject out: coverage at target 0.8 / 0.9 / 0.95": [
                    (lo.get("envelope coverage at target: as stated / map") or {}).get(q, [None, None])[1] for q in ("0.8", "0.9", "0.95")],
                "map, leave one subject out: streamline ECE, point ECE": [(lo.get("streamline ECE: as stated / map / PAT16 map") or [None] * 2)[1],
                                                                          (lo.get("point ECE: as stated / map") or [None] * 2)[1]],
                "widest outline coverage (mean)": mean_of(list(S.values()), lambda s: float(CC.term_shapes(s, TERM)["widest"]["coverage"])),
                "per subject: gap at 0.9 as stated, widest coverage, registration (rotation deg, translation mm, NCC)": {
                    sub: [per[sub]["envelope gap (coverage - stated)"]["0.9"], round(float(CC.term_shapes(s, TERM)["widest"]["coverage"]), 4),
                          [next(iter(s.meta["rescans"].values()))[k] for k in ("rotation_deg", "translation_mm", "ncc_after")]] for sub, s in S.items()}}
        res[key] = row
    print(json.dumps(res, indent=1, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps({"what": f"registration modes {args.modes} (calibration_cohortII compare)", "R": args.runs + 1,
                                              "calibration_cohortII.py sha256": module_sha256(), "git": CR.git_state(), **res}, indent=1, default=str))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("plan", "run", "collect", "evaluate", "compare"))
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="B2000")
    ap.add_argument("--subject", nargs="+")
    ap.add_argument("--runs", type=int, default=CR.RUNS, help="A's replicates (R = runs + 1)")
    ap.add_argument("--smoke", action="store_true", help="run/collect: the <pair>-smoke record folder")
    ap.add_argument("--force", action="store_true", help="collect: rebuild an existing statements file (derived, not a record)")
    ap.add_argument("--check", action="store_true", help="plan: load every scan and check it against every descriptor")
    ap.add_argument("--dry-run", action="store_true", help="evaluate: whatever statements exist, at any R; needs --out elsewhere")
    ap.add_argument("--out", default=None, help="evaluate: the output (default results/calibration_cohortII.json); compare: a JSON copy")
    ap.add_argument("--registration", type=int, choices=sorted(REGISTRATION), default=DEFAULT_REGISTRATION,
                    help="run: the rescan's registration mode; collect, evaluate: the records' (default 2)")
    ap.add_argument("--protocols", nargs="+", dest="protocol_list", default=["B2000", "B32"], help="compare: the protocols")
    ap.add_argument("--modes", nargs="+", type=int, default=[2, 3], help="compare: the registration modes")
    ap.add_argument("--register-only", action="store_true", help="run: mode 3's transforms and decisions only (register3; no tracking)")
    args = ap.parse_args()
    if args.what == "evaluate":
        args.out = args.out or str(OUT)
    elif args.what not in ("plan", "compare"):
        ok = pairs(args.protocol)
        args.subject = args.subject or list(ok)
        bad = [s for s in args.subject if s not in ok]
        if bad:
            ap.error(f"{bad}: no pair for protocol {args.protocol} (pairs: {list(ok)})")
    {"plan": plan, "run": run, "collect": collect, "evaluate": evaluate, "compare": compare}[args.what](args)
