"""A cheap sanity check: does uncorrected head motion, or uncorrected eddy-current distortion, explain part of the
as-stated gap (how far a single scan's 17-run ensemble envelopes under-cover a real rescan)?

    uv run --group dipy --group rankfield bench/calibration_motion_check.py scans [--only <scan name> ...]   # per-scan proxies (cached)
    uv run --group dipy --group rankfield bench/calibration_motion_check.py validate [--scans <name> ...] [--volumes 12]
    uv run --group dipy --group rankfield bench/calibration_motion_check.py evaluate [--onharmony <eval json>]

The pipeline corrects only the static susceptibility field (from the reversed-phase b0s); it does not register the
diffusion volumes, model eddy currents or rotate b-vectors within a scan (docs/pipeline.md, "Open"). If scans or pairs
with more motion or eddy distortion have larger gaps, part of the gap is preprocessing. Everything here is CPU only
(numpy, scipy, dipy), reads the raw data through each study's own loader and the records' runs.json, and writes only
results/calibration_motion_check.json (and its per-scan cache, outside the repository: CACHE).

The pairs and their as-stated gaps (coverage - target at stated 0.8, 0.9, 0.95; per rescan, recomputed from the
statements with calibration_card.coverage_curve and checked against each study's results file):
  - cohort II, registration mode 3: B2000, A2000r19, B32, A32 (calibration_cohortII.statements_file), the swapped
    B2000 pairs (calibration_swap_B2000.json's statements), the Siemens site 3 pairs (calibration_crosssite_reg3.json's);
  - cohort III: 8 subjects, the same-session and other-session rescans (calibration_runs.CARD; statements_R17.npz);
  - ON-Harmony: the cross-site pairs (A on OXF1PRI, forward; and the reverse folders) and the same-scanner rescans
    (calibration_onharmony; statements_R17_reg3.npz), checked against calibration_onharmony.json or a dry run's file.

The per-scan proxies (PROXIES; each scan as its study loads it):
  residual_left        the field fit's residual (runs.json; pipeline.residual_left): motion between the b0 pair counts
                       as left;
  b0 motion            where a scan has 2 or more forward b0s (cohort III 16, ON-Harmony 5, site 3 2; cohort II's
                       scanner A and B series 1: none): each b0 registered rigidly to the first (dipy, mutual
                       information), the displacement at the brain mask's edge (max and mean over edge voxels; the max
                       over b0s of the max, the mean over b0s of the mean) and the rotation;
  DW volumes           each volume of the tracked shell aligned to its own prediction (a log-linear tensor fit of the
                       smoothed scan, b0s and the shell: the prediction has the volume's diffusion contrast and the
                       scan's average position) by Gauss-Newton on a model of 6 rigid parameters plus a phase-encoding
                       (PE) affine (the PE displacement linear in x, y, z: scale along PE and the two shears) with an
                       intensity gain and offset, in the brain mask dilated 2 voxels, smoothed 1 voxel. Per scan, after
                       each parameter's median over volumes is removed: the rigid displacement at the edge (p95, max
                       over volumes), the PE translation's SD and p95 |shift|, the PE displacement at the edge (p95),
                       and the eddy signature: the PE parameters regressed on the gradient direction (linear in g, as
                       eddy currents are; motion is not direction-locked): R^2 of the PE edge displacement explained by
                       g, and the g-explained PE displacement at the edge (RMS and p95 over volumes, mm).
Per pair: A's value, the rescan's, their sum and max, and |A - rescan|.

Analysis: Spearman's rho of each proxy with the gap at 0.9 and the mean gap over 0.8/0.9/0.95 within each stratum (a
permutation p, exact when n <= 8); the ON-Harmony same-scanner rescans and the cohort III other-session rescans also
within subject (A fixed); pooled over strata by within-stratum ranks (a stratified permutation p) and a fixed-effects
slope (gap per mm, and per within-stratum SD) with a bootstrap interval. Correlation only, small n, proxies not full
eddy correction (module docstring of the output: "caveats").
"""
import argparse, hashlib, json, logging, os, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CACHE = Path(os.environ.get("MOTION_CHECK_CACHE",
                            "/private/tmp/claude-501/-Users-halazar-Dropbox-development-tractline/05f52605-7344-4a49-9d29-015cd95ba481/scratchpad/motion"))
OUT = HERE / "results/calibration_motion_check.json"
ONH_FINAL = HERE / "results/calibration_onharmony.json"
ONH_PARTIAL = Path("/private/tmp/claude-501/-Users-halazar-Dropbox-development-tractline/05f52605-7344-4a49-9d29-015cd95ba481/scratchpad/onharmony/eval_partial.json")
TARGETS = (0.8, 0.9, 0.95)
PROXY_VERSION = 4                      # the per-scan cache is used only at this version
SMOOTH_VOX = 1.0                       # Gaussian sigma (voxels) before the tensor fit and the alignment
DILATE = 2                             # the alignment's mask: the brain mask dilated this many voxels (the edge carries
                                       # most of the information about a shift)
GN_ITERS = 6
SEED = 0

logging.getLogger("dipy").setLevel(logging.WARNING)


# ----------------------------------------------------------------------------------------------- the studies

def _mods():
    """The study modules (imported lazily: they import the pipeline)."""
    import calibration_cohortII as CII
    import calibration_crosssite as CS
    import calibration_runs as CR
    import calibration_card as CC
    import calibration_onharmony as OH
    import _masivar as M
    return CII, CS, CR, CC, OH, M


def study_of(name):
    parts = name.split("_")
    return ("ON-Harmony" if name.endswith("_PRISMA_UKB") else "cohort II site 3" if name.endswith("_S3_2000")
            else "cohort III" if parts[2].startswith("scan") else "cohort II")


def load_scan(name):
    """A scan by its records' name, through its study's loader: (Subject, tracked shell, study)."""
    CII, CS, CR, CC, OH, M = _mods()
    parts = name.split("_")
    if name.endswith("_PRISMA_UKB"):
        return OH.load(parts[0], parts[1]), OH.SHELL, "ON-Harmony"
    if name.endswith("_S3_2000"):
        return CS.load_s3(parts[0]), 2000.0, "cohort II site 3"
    if parts[2].startswith("scan"):
        return M.load(parts[0], parts[1], int(parts[2][4:]), CR.SHELLS), CR.SHELL, "cohort III"
    key = "_".join(parts[2:])
    return CII.load(parts[0], parts[1], key), CII.PROTOCOLS[key]["tracked shell"], "cohort II"


def gaps_of(cnt, hit):
    from calibration_card import coverage_curve
    cov = coverage_curve(cnt, hit, np.array(TARGETS))
    g = {f"{q:g}": round(float(c - q), 4) for q, c in zip(TARGETS, cov)}
    g["mean"] = round(float(np.mean([g[f"{q:g}"] for q in TARGETS])), 4)
    return g


def runs_logs(*dirs):
    out = {}
    for d in dirs:
        p = Path(d) / "runs.json"
        if p.exists():
            out.update({k: v for k, v in json.loads(p.read_text()).items() if k not in out})
    return out


def residual_of(log, scan):
    """A scan's residual_left from a runs.json: A's run if it is A, else any rescan entry of that scan that has one."""
    if log.get("A_scan", {}).get("name") == scan:
        return log.get("A", {}).get("residual_left")
    for k, v in log.items():
        if k.endswith("_scan") and isinstance(v, dict) and v.get("name") == scan:
            r = v.get("residual_left")
            if r is None:                      # older records: the rescan's own-grid run under its tag
                r = (log.get(k[:-5]) or {}).get("residual_left")
            if r is not None:
                return r
    return None


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_pairs(onh_json):
    """Every pair: stratum, subject, A and rescan scan names, term, gaps, residual_left (A, rescan), registration,
    and the check against the study's results file."""
    CII, CS, CR, CC, OH, M = _mods()
    from tractline.data import DATA
    res = HERE / "results"
    pairs, checks, sources = [], [], {}

    def add(stratum, study, sub, d_log, s, tag, a_name, extra=None):
        z = s.z
        r = s.meta["rescans"][tag]
        log = d_log
        g = gaps_of(z["e_cnt"], z[f"e_hit_{tag}"])
        pairs.append({"stratum": stratum, "study": study, "subject": sub, "A": a_name, "rescan": r["scan"], "tag": tag, "term": r["term"],
                      "gap": g, "residual_left": {"A": residual_of(log, a_name), "rescan": residual_of(log, r["scan"])},
                      "registration (rotation deg, translation mm, NCC)": [r.get("rotation_deg"), r.get("translation_mm"), r.get("ncc_after")],
                      "QC flag": r.get("QC flag"), "statements": str(s.path.relative_to(DATA)), **(extra or {})})

    # cohort II, mode 3, per protocol
    c2 = json.loads((res / "calibration_cohortII.json").read_text())
    sources["calibration_cohortII.json"] = sha(res / "calibration_cohortII.json")
    for key in CII.EVALUATED:
        S = CII.statements(key, CR.RUNS, False, 3)
        for sub, s in S.items():
            d = CII.Records(sub, key).dir
            log = runs_logs(d)
            a = log["A_scan"]["name"]
            for t in s.tags(CII.TERM):
                add(f"cohort II {key}", "cohort II", sub, log, s, t, a)
                want = c2["(a) as stated"]["per protocol"][key]["per subject"][sub]["envelope gap (coverage - stated)"]
                checks.append((f"cohort II {key} {sub}", {q: pairs[-1]["gap"][q] for q in ("0.8", "0.9", "0.95")}, {q: want[q] for q in ("0.8", "0.9", "0.95")}))
    # the swapped B2000 pairs
    sw = json.loads((res / "calibration_swap_B2000.json").read_text())
    sources["calibration_swap_B2000.json"] = sha(res / "calibration_swap_B2000.json")
    for sub, pv in sw["provenance"]["statements"]["swapped"].items():
        d = DATA / pv["path"]
        p = d / CII.statements_name(pv["R"], pv["registration mode"])
        s = CC.Subject(p); s.path = p
        log = runs_logs(d)
        a = log["A_scan"]["name"]
        for t in s.tags(CII.TERM):
            add("cohort II B2000 swapped", "cohort II", sub, log, s, t, a)
            want = sw["swapped"]["as stated"]["per subject"][sub]["envelope gap (coverage - stated)"]
            checks.append((f"cohort II B2000 swapped {sub}", {q: pairs[-1]["gap"][q] for q in ("0.8", "0.9", "0.95")}, {q: want[q] for q in ("0.8", "0.9", "0.95")}))
    # the Siemens site 3 pairs (A: B2000's session 1 scan)
    xs = json.loads((res / "calibration_crosssite_reg3.json").read_text())
    sources["calibration_crosssite_reg3.json"] = sha(res / "calibration_crosssite_reg3.json")
    for sub, pv in xs["sources"]["B2000"]["provenance"].items():
        d = DATA / "ensemble" / pv["cross-vendor records"].removeprefix("ensemble/")
        p = d / pv["cross-vendor statements"]
        s = CC.Subject(p); s.path = p
        src = DATA / "ensemble" / pv["source records"].removeprefix("ensemble/")
        log = runs_logs(d, src)
        a = log["A_scan"]["name"]
        for t in s.tags(CS.TERM_X):
            add("cohort II site 3 (Siemens)", "cohort II", sub, log, s, t, a)
            cov = xs["sources"]["B2000"]["per subject"][sub]["(i) cross vendor, as stated"]["coverage at stated level"]
            lv = list(np.round(CII.SCHEMA_LEVELS, 3))
            want = {f"{q:g}": round(cov[lv.index(q)] - q, 4) for q in TARGETS}
            checks.append((f"cohort II site 3 {sub}", {q: pairs[-1]["gap"][q] for q in ("0.8", "0.9", "0.95")}, want))
    # cohort III
    card = json.loads((res / "calibration_card_masivar_cIII.json").read_text())
    sources["calibration_card_masivar_cIII.json"] = sha(res / "calibration_card_masivar_cIII.json")
    for sub, pr in CR.CARD.items():
        R = CR.Records(sub)
        p = R.dir / f"statements_R{CR.RUNS + 1}.npz"
        if not p.exists():
            continue
        s = CC.Subject(p); s.path = p
        log = runs_logs(R.dir, *([R.reuse] if R.reuse else []))
        a = log["A_scan"]["name"]
        for t, r in s.meta["rescans"].items():
            term = r["term"]
            add(f"cohort III {term}", "cohort III", sub, log, s, t, a)
            want = card["held-out evaluation (leave one subject out)"]["per rescan"][sub][t]["envelope coverage at stated 0.8, 0.9"]
            checks.append((f"cohort III {sub} {t}", {q: pairs[-1]["gap"][q] for q in ("0.8", "0.9")}, {"0.8": round(want[0] - 0.8, 4), "0.9": round(want[1] - 0.9, 4)}))
    # ON-Harmony
    onh = json.loads(Path(onh_json).read_text()) if onh_json and Path(onh_json).exists() else None
    sources["ON-Harmony results"] = {"path": str(onh_json), "sha256": sha(onh_json) if onh else None, "dry run": onh.get("dry run") if onh else None}
    for sc in OH.SCANNERS:
        for sub in OH.SUBJECTS:
            try:
                Rr = OH.Records(sub, sc)
            except AssertionError:
                continue
            p = Rr.dir / OH.statements_name(CR.RUNS + 1)
            if not p.exists():
                continue
            s = CC.Subject(p); s.path = p
            log = runs_logs(Rr.dir)
            a = log["A_scan"]["name"]
            for t, r in s.meta["rescans"].items():
                cross = r["term"] == OH.TERM_X
                stratum = ("ON-Harmony cross site " + ("forward (A on OXF1)" if sc == "OXF1" else "reverse (A on OXF2)")) if cross else "ON-Harmony same scanner"
                add(stratum, "ON-Harmony", sub, log, s, t, a, {"A's scanner": sc})
                if onh is None:
                    continue
                a_ = onh["(a) as stated"]
                if cross:
                    blk = a_["cross site, forward (OXF1 -> OXF2)" if sc == "OXF1" else "cross site, reverse (OXF2 -> OXF1)"]["per subject"].get(sub)
                    if blk:
                        lv = list(np.round(onh["levels"], 3))
                        want = {f"{q:g}": round(blk["coverage at stated level"][lv.index(q)] - q, 4) for q in TARGETS}
                        checks.append((f"ON-Harmony {sc} {sub} {t}", {q: pairs[-1]["gap"][q] for q in ("0.8", "0.9", "0.95")}, want))
                else:
                    blk = a_["per rescan (same-scanner subjects)"].get(sub, {}).get(t)
                    if blk:
                        w = blk["envelope coverage at stated 0.8, 0.9"]
                        checks.append((f"ON-Harmony {sc} {sub} {t}", {q: pairs[-1]["gap"][q] for q in ("0.8", "0.9")}, {"0.8": round(w[0] - 0.8, 4), "0.9": round(w[1] - 0.9, 4)}))
    bad = [(n, got, want) for n, got, want in checks if any(abs(got[q] - want[q]) > 1.5e-4 for q in want)]
    return pairs, {"compared": len(checks), "mismatched": [{"pair": n, "recomputed": g, "results file": w} for n, g, w in bad]}, sources


# ----------------------------------------------------------------------------------------------- the proxies

def brain_mask(mean_b0):
    from dipy.segment.mask import median_otsu
    return median_otsu(mean_b0.astype(np.float32), median_radius=4, numpass=4)[1].astype(bool)


def edge_points(mask):
    from scipy.ndimage import binary_erosion
    return np.argwhere(mask & ~binary_erosion(mask))


def rigid_b0(static, moving, affine):
    """dipy rigid registration (mutual information) of one b0 onto another of the same scan: the 4x4 world transform
    (static world -> moving world)."""
    from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
    from dipy.align.transforms import RigidTransform3D
    np.random.seed(SEED)
    reg = AffineRegistration(metric=MutualInformationMetric(nbins=32, sampling_proportion=0.5),
                             level_iters=[200, 100, 50], sigmas=[2.0, 1.0, 0.0], factors=[4, 2, 1], verbosity=0)
    m = reg.optimize(static.astype(np.float64), moving.astype(np.float64), RigidTransform3D(), None, static_grid2world=affine,
                     moving_grid2world=affine, starting_affine=np.eye(4))
    return np.asarray(m.affine, float)


def angle(T):
    R = np.asarray(T, float)[:3, :3]
    return float(np.degrees(np.arctan2(np.linalg.norm(R - R.T) / (2 * np.sqrt(2)), (np.trace(R) - 1) / 2)))


def b0_motion(s, mask):
    b0i = s.b0_index
    if len(b0i) < 2:
        return None
    pts = edge_points(mask)
    w = (s.affine[:3, :3] @ pts.T).T + s.affine[:3, 3]
    static = s.dwi[..., b0i[0]]
    # the cross-check: Gauss-Newton rigid (least squares with a gain and offset; smoothed 2, then 1 voxel)
    from scipy.ndimage import gaussian_filter, binary_dilation
    vox = np.asarray(s.vox, float)
    idx = np.argwhere(binary_dilation(mask, iterations=DILATE))
    cen = idx.mean(0)
    rc, ec = (idx - cen) * vox, (pts - cen) * vox
    sm = {sg: gaussian_filter(static.astype(np.float32), sg) for sg in (2.0, 1.0)}
    rows = []
    for k in b0i[1:]:
        T = rigid_b0(static, s.dwi[..., k], s.affine)
        d = np.linalg.norm((T[:3, :3] @ w.T).T + T[:3, 3] - w, axis=1)
        init = None
        for sg in (2.0, 1.0):
            init = align_volume(gaussian_filter(s.dwi[..., k].astype(np.float32), sg), sm[sg], idx, rc, vox, int(s.pe_axis), iters=10, pe_affine=False, init=init)
        dg = np.linalg.norm(init[0] + np.cross(init[1], ec), axis=1)
        rows.append({"volume": int(k), "edge max mm": round(float(d.max()), 3), "edge mean mm": round(float(d.mean()), 3),
                     "rotation deg": round(angle(T), 3),
                     "Gauss-Newton cross-check: edge max, mean mm": [round(float(dg.max()), 3), round(float(dg.mean()), 3)]})
    return {"b0s": int(len(b0i)), "reference": "the first forward b0", "per b0": rows,
            "max edge displacement mm (max over b0s)": max(r["edge max mm"] for r in rows),
            "mean edge displacement mm (mean over b0s)": round(float(np.mean([r["edge mean mm"] for r in rows])), 3),
            "max rotation deg": max(r["rotation deg"] for r in rows),
            "Gauss-Newton cross-check: max edge (max over b0s), mean edge (mean over b0s) mm": [
                max(r["Gauss-Newton cross-check: edge max, mean mm"][0] for r in rows),
                round(float(np.mean([r["Gauss-Newton cross-check: edge max, mean mm"][1] for r in rows])), 3)]}


def tensor_prediction(dwi, bval, bvec, sel, mask):
    """A log-linear least-squares tensor fit (S0 and D) in the mask, on volumes sel; the predicted signal (X, Y, Z, len(sel))."""
    S = np.clip(dwi[mask][:, sel].astype(np.float64), 1.0, None)
    b, g = bval[sel], bvec[:, sel]
    X = np.stack([np.ones_like(b), -b * g[0] ** 2, -b * g[1] ** 2, -b * g[2] ** 2, -2 * b * g[0] * g[1], -2 * b * g[0] * g[2], -2 * b * g[1] * g[2]], 1)
    beta, *_ = np.linalg.lstsq(X, np.log(S).T, rcond=None)
    pred = np.zeros(mask.shape + (len(sel),), np.float32)
    pred[mask] = np.exp(X @ beta).T.astype(np.float32)
    return pred


def align_volume(I, P, mask_idx, rc, vox, pe, iters=GN_ITERS, pe_affine=True, init=None):
    """Gauss-Newton: params u (12: t (3), omega (3), PE affine a (3: dPE per mm of x, y, z), unused) such that I sampled
    at r + u(r) matches P (with a gain and an offset), u(r) = t + omega x r + e_pe (a . r), r centered (mm, voxel axes).
    Returns (t, omega, a), the final normalized residual."""
    from scipy.ndimage import map_coordinates
    gP = np.stack([np.gradient(P, axis=k)[tuple(mask_idx.T)] / vox[k] for k in range(3)], 1)       # (N, 3) per mm
    Pm = P[tuple(mask_idx.T)].astype(np.float64)
    cross = np.cross(rc, gP)                                                                       # (r x grad P)
    Jpe = gP[:, [pe]] * rc                                                                         # dPE linear in r
    base = np.concatenate([gP, cross, Jpe] if pe_affine else [gP, cross], 1)
    t, om, a = (np.zeros(3), np.zeros(3), np.zeros(3)) if init is None else (init[0].copy(), init[1].copy(), init[2].copy())
    for _ in range(iters):
        u = t + np.cross(om, rc)
        u[:, pe] += rc @ a
        coords = (mask_idx + u / vox).T
        Iw = map_coordinates(I, coords, order=1, mode="nearest")
        J = np.concatenate([base, Iw[:, None], np.ones((len(Iw), 1))], 1)
        delta, *_ = np.linalg.lstsq(J, Pm - Iw, rcond=None)
        t += delta[:3]; om += delta[3:6]
        if pe_affine:
            a += delta[6:9]
        if np.abs(delta[:base.shape[1]]).max() < 1e-4:
            break
    return t, om, a


def r2(y, G):
    X = np.column_stack([np.ones(len(G)), G])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ beta
    ss = ((y - y.mean()) ** 2).sum()
    return (1 - ((y - fit) ** 2).sum() / ss if ss > 0 else 0.0), fit - beta[0]


def dw_proxies(s, shell, mask, keep_params=False):
    from scipy.ndimage import gaussian_filter, binary_dilation
    vox = np.asarray(s.vox, float)
    pe = int(s.pe_axis)
    b0i = s.b0_index
    sh = np.flatnonzero(np.abs(s.bval - shell) < 50)
    sel = np.concatenate([b0i, sh])
    sm = np.stack([gaussian_filter(s.dwi[..., k].astype(np.float32), SMOOTH_VOX) for k in sel], -1)
    m2 = binary_dilation(mask, iterations=DILATE)
    pred = tensor_prediction(sm, s.bval[sel], s.bvec[:, sel], np.arange(len(sel)), m2)
    idx = np.argwhere(m2)
    cen = idx.mean(0)
    rc = (idx - cen) * vox
    epts = (edge_points(mask) - cen) * vox
    P = []
    for j in range(len(b0i), len(sel)):
        t, om, a = align_volume(sm[..., j], pred[..., j], idx, rc, vox, pe)
        P.append(np.concatenate([t, om, a]))
    P = np.array(P)                                    # (n, 9): t (mm), omega (rad), a (mm / mm)
    Pc = P - np.median(P, 0)                           # relative to the scan's typical volume
    G = s.bvec[:, sh].T
    # rigid displacement at the edge per volume (magnitude), and the PE displacement (t_pe + a . r) at the edge
    rig = np.array([np.linalg.norm(p[:3] + np.cross(p[3:6], epts), axis=1).max() for p in Pc])
    pe_edge = np.array([p[pe] + epts @ p[6:9] for p in Pc])            # (n, E)
    pe_max = np.abs(pe_edge).max(1)
    # the eddy signature: the PE parameters (t_pe, a) linear in g
    cols = [pe, 6, 7, 8]
    fits = {}
    expl = np.zeros((len(Pc), 4))
    for c in cols:
        rr, f = r2(Pc[:, c], G)
        fits[c] = round(float(rr), 3)
        expl[:, cols.index(c)] = f
    eddy_edge = expl[:, [0]] + expl[:, 1:] @ epts.T                      # (n, E): the g-explained PE displacement
    eddy_max = np.abs(eddy_edge).max(1)
    tot = ((pe_edge - pe_edge.mean(0)) ** 2).sum()
    r2_edge = float(1 - ((pe_edge - eddy_edge - (pe_edge - eddy_edge).mean(0)) ** 2).sum() / tot) if tot > 0 else 0.0
    rig_r2 = float(np.mean([r2(Pc[:, c], G)[0] for c in [k for k in range(3) if k != pe] + [3, 4, 5]]))
    out = {"volumes": int(len(sh)), "shell": float(shell), "pe axis": pe,
           "rigid edge displacement mm: p95, max": [round(float(np.quantile(rig, 0.95)), 3), round(float(rig.max()), 3)],
           "PE translation mm: SD, p95 |shift|": [round(float(Pc[:, pe].std()), 3), round(float(np.quantile(np.abs(Pc[:, pe]), 0.95)), 3)],
           "PE scale (dPE/dPE, x 1e3): SD": round(float(Pc[:, 6 + pe].std() * 1e3), 3),
           "PE shears (x 1e3): SD": [round(float(Pc[:, 6 + k].std() * 1e3), 3) for k in range(3) if k != pe],
           "PE edge displacement mm: p95 over volumes (of the max over edge voxels)": round(float(np.quantile(pe_max, 0.95)), 3),
           "eddy: R^2 on g, per PE parameter (translation, d/dx, d/dy, d/dz)": [fits[c] for c in cols],
           "eddy: R^2 on g of the PE edge displacement": round(r2_edge, 3),
           "eddy: g-explained PE edge displacement mm: RMS, p95 over volumes": [round(float(np.sqrt((eddy_max ** 2).mean())), 3), round(float(np.quantile(eddy_max, 0.95)), 3)],
           "motion control: mean R^2 on g of the non-PE rigid parameters": round(rig_r2, 3),
           "edge voxels": int(len(epts)), "mask voxels (dilated)": int(len(idx))}
    if keep_params:
        out["_params"] = P
        out["_sel"] = sel
        out["_pred"] = pred
        out["_sm"] = sm
        out["_idx"] = idx
    return out


def scan_proxies(name):
    t0 = time.time()
    s, shell, study = load_scan(name)
    mean_b0 = s.dwi[..., s.b0_index].astype(np.float32).mean(-1)
    mask = brain_mask(mean_b0)
    t1 = time.time()
    dw = dw_proxies(s, shell, mask)
    t2 = time.time()
    b0 = b0_motion(s, mask)
    t3 = time.time()
    return {"scan": name, "study": study, "version": PROXY_VERSION, "shape": list(s.dwi.shape), "voxel mm": [round(float(v), 4) for v in s.vox],
            "pe axis": int(s.pe_axis), "forward b0s": int(len(s.b0_index)), "dw": dw, "b0 motion": b0,
            "seconds": {"load and mask": round(t1 - t0, 1), "dw": round(t2 - t1, 1), "b0 motion": round(t3 - t2, 1)}}


def cached(name, force=False):
    p = CACHE / "scans" / f"{name}.json"
    if p.exists() and not force:
        r = json.loads(p.read_text())
        if r.get("version") == PROXY_VERSION:
            return r
    r = scan_proxies(name)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(r, indent=1))
    os.replace(tmp, p)
    return r


# the scalar proxies the analysis uses: name -> (getter, unit, what)
PROXIES = {
    "residual_left": (None, "", "the field fit's residual (runs.json)"),
    "b0 motion: max edge mm": (lambda r: (r["b0 motion"] or {}).get("max edge displacement mm (max over b0s)"), "mm",
                               "b0s onto the first, rigid: max over b0s of the max edge displacement"),
    "b0 motion: mean edge mm": (lambda r: (r["b0 motion"] or {}).get("mean edge displacement mm (mean over b0s)"), "mm",
                                "b0s onto the first, rigid: mean over b0s of the mean edge displacement"),
    "DW rigid: p95 edge mm": (lambda r: r["dw"]["rigid edge displacement mm: p95, max"][0], "mm",
                              "DW volumes onto their prediction: rigid displacement at the edge, p95 over volumes"),
    "DW PE translation SD mm": (lambda r: r["dw"]["PE translation mm: SD, p95 |shift|"][0], "mm", "SD over volumes of the PE translation"),
    "DW PE edge p95 mm": (lambda r: r["dw"]["PE edge displacement mm: p95 over volumes (of the max over edge voxels)"], "mm",
                          "PE affine displacement at the edge, p95 over volumes"),
    "DW eddy (g-explained PE) RMS mm": (lambda r: r["dw"]["eddy: g-explained PE edge displacement mm: RMS, p95 over volumes"][0], "mm",
                                        "the part of the PE edge displacement linear in the gradient direction, RMS over volumes"),
    "DW eddy R^2": (lambda r: r["dw"]["eddy: R^2 on g of the PE edge displacement"], "", "share of the PE edge displacement's variance linear in g"),
}


def scan_names(pairs):
    seen = []
    for p in pairs:
        for k in ("A", "rescan"):
            if p[k] not in seen:
                seen.append(p[k])
    return seen


# ----------------------------------------------------------------------------------------------- statistics

def rankdata(x):
    from scipy.stats import rankdata as rd
    return rd(x)


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def perm_p(x, y, groups=None, n_perm=10000, rng=None):
    """Two-sided permutation p of Spearman's rho, or (groups) of the pooled correlation of within-group ranks, x
    permuted within groups; exact enumeration when there are no groups and n <= 8. Vectorized over permutations."""
    import itertools
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = stat_grouped(x, y, groups)
    if obs is None:
        return None, None
    if groups is None:
        rx, ry = rankdata(x), rankdata(y)
    else:
        rx, ry = within_ranks(x, groups), within_ranks(y, groups)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    scale = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if groups is None and len(x) <= 8:
        perms = np.array(list(itertools.permutations(range(len(x)))))
    else:
        rng = rng or np.random.default_rng(SEED)
        g = np.zeros(len(x), int) if groups is None else np.unique(np.asarray(groups), return_inverse=True)[1]
        base = np.argsort(g, kind="stable")                         # positions grouped
        key = g[base][None, :] * 10.0 + rng.random((n_perm, len(x)))
        order = base[np.argsort(key, axis=1)]                       # a within-group shuffle of the grouped positions
        perms = np.empty_like(order)
        perms[:, base] = order
    vals = (rx[perms] * ry[None, :]).sum(1) / scale
    p = float(np.mean(np.abs(vals) >= abs(obs) - 1e-12)) if groups is None and len(x) <= 8 else float((np.sum(np.abs(vals) >= abs(obs) - 1e-12) + 1) / (len(vals) + 1))
    return obs, p


def within_ranks(x, groups):
    """Ranks within each group, centered and scaled to (-0.5, 0.5); a group of one gives 0."""
    x = np.asarray(x, float)
    out = np.zeros(len(x))
    for k in np.unique(groups):
        i = np.flatnonzero(np.asarray(groups) == k)
        if len(i) > 1:
            out[i] = (rankdata(x[i]) - 0.5) / len(i) - 0.5
    return out


def stat_grouped(x, y, groups):
    """Spearman's rho (no groups) or the pooled correlation of within-group ranks."""
    if groups is None:
        return spearman(x, y)
    rx, ry = within_ranks(x, groups), within_ranks(y, groups)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def fe_slope(x, y, groups, n_boot=1000, rng=None):
    """Fixed-effects slope (group means removed) of y on x, and on x in within-group SD units; bootstrap 95 % interval
    (pairs resampled within groups)."""
    x, y, g = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)

    def fit(x, y, g):
        xd, yd, xz = x.copy(), y.copy(), x.copy()
        for k in np.unique(g):
            i = g == k
            xd[i] -= x[i].mean(); yd[i] -= y[i].mean()
            sd = x[i].std()
            xz[i] = xd[i] / sd if sd > 0 else 0.0
        b = (xd * yd).sum() / (xd ** 2).sum() if (xd ** 2).sum() > 0 else np.nan
        bz = (xz * yd).sum() / (xz ** 2).sum() if (xz ** 2).sum() > 0 else np.nan
        r2_ = 1 - ((yd - b * xd) ** 2).sum() / (yd ** 2).sum() if (yd ** 2).sum() > 0 else np.nan
        return b, bz, r2_
    b, bz, r2_ = fit(x, y, g)
    rng = rng or np.random.default_rng(SEED)
    bs, bzs = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(np.flatnonzero(g == k), (g == k).sum()) for k in np.unique(g)])
        bb, bbz, _ = fit(x[idx], y[idx], g[idx])
        bs.append(bb); bzs.append(bbz)
    q = lambda v: [round(float(np.nanquantile(v, 0.025)), 5), round(float(np.nanquantile(v, 0.975)), 5)]  # noqa: E731
    return {"slope (gap per unit)": round(float(b), 5), "95 % interval": q(bs), "slope (gap per within-stratum SD)": round(float(bz), 5),
            "95 % interval (per SD)": q(bzs), "R^2 within strata": round(float(r2_), 4)}


# ----------------------------------------------------------------------------------------------- the analysis

COMBOS = ("A", "rescan", "sum", "max", "|A - rescan|")
GAPS = ("0.9", "mean")


def pair_values(p, scans, proxy):
    get = PROXIES[proxy][0]
    if get is None:
        a, b = p["residual_left"]["A"], p["residual_left"]["rescan"]
    else:
        a, b = get(scans[p["A"]]), get(scans[p["rescan"]])
    if a is None or b is None:
        return None
    return {"A": a, "rescan": b, "sum": a + b, "max": max(a, b), "|A - rescan|": abs(a - b)}


def table(rows, scans, groups_of=None, name=""):
    """Per proxy and combination: n, rho and p against the gap at 0.9 and the mean gap."""
    out = {}
    for proxy in PROXIES:
        vals = [(p, pair_values(p, scans, proxy)) for p in rows]
        vals = [(p, v) for p, v in vals if v is not None]
        if len(vals) < 4:
            continue
        out[proxy] = {}
        groups = None if groups_of is None else [groups_of(p) for p, _ in vals]
        for c in COMBOS:
            x = [v[c] for _, v in vals]
            e = {"n": len(vals)}
            for gk in GAPS:
                y = [p["gap"][gk] for p, _ in vals]
                rho, pv = perm_p(x, y, groups)
                e[f"rho (gap {gk})"] = None if rho is None else round(rho, 3)
                e[f"p (gap {gk})"] = None if pv is None else round(pv, 4)
            out[proxy][c] = e
    return out


def ranges(scans_by_study):
    out = {}
    for study, rs in scans_by_study.items():
        out[study] = {"scans": len(rs)}
        for proxy, (get, unit, _) in PROXIES.items():
            if get is None:
                continue
            v = [get(r) for r in rs]
            v = [x for x in v if x is not None]
            if v:
                out[study][proxy] = {"min": round(float(np.min(v)), 3), "median": round(float(np.median(v)), 3), "max": round(float(np.max(v)), 3)}
    return out


def evaluate(args):
    onh = args.onharmony or (ONH_FINAL if ONH_FINAL.exists() else ONH_PARTIAL)
    pairs, check, sources = collect_pairs(onh)
    print(f"{len(pairs)} pairs; gaps checked against the results files: {check['compared']} compared, {len(check['mismatched'])} mismatched", flush=True)
    for m in check["mismatched"]:
        print("  MISMATCH", m, flush=True)
    if args.cached_only:                      # testing: only the pairs whose scans are cached
        have = lambda n: (CACHE / "scans" / f"{n}.json").exists() and json.loads((CACHE / "scans" / f"{n}.json").read_text()).get("version") == PROXY_VERSION  # noqa: E731
        pairs = [p for p in pairs if have(p["A"]) and have(p["rescan"])]
        print(f"cached only: {len(pairs)} pairs", flush=True)
    names = scan_names(pairs)
    scans = {}
    for i, n in enumerate(names):
        t0 = time.time()
        scans[n] = cached(n)
        print(f"[{i + 1}/{len(names)}] {n} {time.time() - t0:.1f} s", flush=True)
    # residual_left per scan (from the records), for the ranges
    resid = {}
    for p in pairs:
        for k in ("A", "rescan"):
            v = p["residual_left"][k]
            if v is not None:
                resid.setdefault(p[k], set()).add(v)
    by_study = {}
    for n, r in scans.items():
        by_study.setdefault(r["study"], []).append(r)
    rng_ = ranges(by_study)
    for study, rs in by_study.items():
        v = [min(resid[r["scan"]]) for r in rs if r["scan"] in resid]
        if v:
            rng_[study]["residual_left"] = {"min": round(float(np.min(v)), 3), "median": round(float(np.median(v)), 3), "max": round(float(np.max(v)), 3)}
    strata = {}
    for p in pairs:
        strata.setdefault(p["stratum"], []).append(p)
    # within each stratum
    corr = {k: {"n pairs": len(v), "table": table(v, scans)} for k, v in strata.items()}
    # within subject (A fixed): ON-Harmony same scanner, cohort III other session, and all ON-Harmony pairs by subject
    within = {}
    for k in ("ON-Harmony same scanner", "cohort III other session"):
        if k in strata:
            within[f"{k}, within subject (ranks within subject, permuted within subject)"] = {
                "n pairs": len(strata[k]), "subjects": sorted({p["subject"] for p in strata[k]}),
                "table": table(strata[k], scans, groups_of=lambda p: p["subject"])}
    # cohort III, both terms, a subject's mean (term-centered gap)
    c3 = [p for p in pairs if p["study"] == "cohort III"]
    # pooled: within-stratum ranks; strata of independent-ish pairs (the swapped and site 3 pairs share scans with B2000)
    pooled = {}
    sets = {"all strata": list(strata),
            "without the swapped B2000 and site 3 (no scan in two strata)": [k for k in strata if "swapped" not in k and "site 3" not in k]}
    for label, keys in sets.items():
        rows = [p for k in keys for p in strata[k]]
        # ON-Harmony same scanner: ranked within subject (A fixed)
        grp = lambda p: p["stratum"] + (" " + p["subject"] if p["stratum"] == "ON-Harmony same scanner" else "")  # noqa: E731
        pooled[label] = {"strata": keys, "n pairs": len(rows), "table": table(rows, scans, groups_of=grp)}
        eff = {}
        for proxy in PROXIES:
            vals = [(p, pair_values(p, scans, proxy)) for p in rows]
            vals = [(p, v) for p, v in vals if v is not None]
            if len(vals) < 6:
                continue
            g = [grp(p) for p, _ in vals]
            eff[proxy] = {"n": len(vals), "unit": PROXIES[proxy][1]}
            y = np.array([p["gap"]["0.9"] for p, _ in vals]); ga = np.array(g)
            for c in ("sum", "max", "|A - rescan|"):
                x = [v[c] for _, v in vals]
                for gk in GAPS:
                    eff[proxy][f"{c}, gap {gk}"] = fe_slope(x, [p["gap"][gk] for p, _ in vals], g)
                # how much of the gap: the slope times the proxy's mean excess over its stratum's lowest value
                xa = np.array(x)
                excess = float(np.mean([xa[ga == k].mean() - xa[ga == k].min() for k in np.unique(ga)]))
                eff[proxy][f"gap at 0.9 removed if every pair had its stratum's lowest proxy ({c}): slope x mean excess"] = round(eff[proxy][f"{c}, gap 0.9"]["slope (gap per unit)"] * excess, 4)
            eff[proxy]["mean gap at 0.9 of these pairs"] = round(float(y.mean()), 4)
        pooled[label]["effect sizes (fixed effects, stratum means removed)"] = eff
    # the gaps themselves per stratum
    gaps = {k: {"n": len(v), "gap 0.9: mean, min, max": [round(float(np.mean([p["gap"]["0.9"] for p in v])), 4),
                                                         min(p["gap"]["0.9"] for p in v), max(p["gap"]["0.9"] for p in v)],
                "mean gap: mean": round(float(np.mean([p["gap"]["mean"] for p in v])), 4)} for k, v in strata.items()}
    # checks on the proxies themselves
    from scipy.stats import mannwhitneyu, spearmanr as _sp
    b0 = np.array([(r["b0 motion"]["max edge displacement mm (max over b0s)"],
                    r["b0 motion"]["Gauss-Newton cross-check: max edge (max over b0s), mean edge (mean over b0s) mm"][0])
                   for r in scans.values() if r["b0 motion"]])
    proxy_checks = {
        "b0 motion, dipy MI against the Gauss-Newton least-squares cross-check (max edge, per scan)": {
            "scans": int(len(b0)), "Pearson r": round(float(np.corrcoef(b0.T)[0, 1]), 3), "Spearman rho": round(float(_sp(b0[:, 0], b0[:, 1])[0]), 3),
            "median (Gauss-Newton - dipy) mm": round(float(np.median(b0[:, 1] - b0[:, 0])), 3)} if len(b0) > 2 else None,
        "eddy signature per study (medians over scans)": {
            st: {"R^2 on g per PE parameter (translation, d/dx, d/dy, d/dz)": np.round(np.median([r["dw"]["eddy: R^2 on g, per PE parameter (translation, d/dx, d/dy, d/dz)"] for r in rs], 0), 3).tolist(),
                 "control: R^2 on g of the non-PE rigid parameters (null about 3 / (volumes - 1))": round(float(np.median([r["dw"]["motion control: mean R^2 on g of the non-PE rigid parameters"] for r in rs])), 3)}
            for st, rs in by_study.items()}}
    # a natural experiment: ON-Harmony's OXF2PRI001 rescans fall in two eddy groups
    xs = strata.get("ON-Harmony cross site forward (A on OXF1)", [])
    if xs:
        e = np.array([scans[p["rescan"]]["dw"]["eddy: g-explained PE edge displacement mm: RMS, p95 over volumes"][0] for p in xs])
        g9 = np.array([p["gap"]["0.9"] for p in xs])
        hi = e > 1.2
        if hi.any() and (~hi).any():
            proxy_checks["ON-Harmony cross site: the rescans (OXF2PRI001) split by eddy (g-explained PE RMS above / below 1.2 mm)"] = {
                "high: n, eddy range mm, mean gap 0.9": [int(hi.sum()), [round(float(e[hi].min()), 3), round(float(e[hi].max()), 3)], round(float(g9[hi].mean()), 4)],
                "low: n, eddy range mm, mean gap 0.9": [int((~hi).sum()), [round(float(e[~hi].min()), 3), round(float(e[~hi].max()), 3)], round(float(g9[~hi].mean()), 4)],
                "difference (high - low)": round(float(g9[hi].mean() - g9[~hi].mean()), 4),
                "Mann-Whitney p (two-sided)": round(float(mannwhitneyu(g9[hi], g9[~hi]).pvalue), 4),
                "subjects (high)": [p["subject"] for p, h in zip(xs, hi) if h]}
    from calibration_runs import git_state
    val = json.loads((CACHE / "validation.json").read_text()) if (CACHE / "validation.json").exists() else None
    out = {"what": "a sanity check: do uncorrected head motion or eddy-current proxies track the as-stated envelope gap (coverage - target at stated level) across pairs?",
           "created (UTC)": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": git_state(),
           "calibration_motion_check.py sha256": sha(Path(__file__).resolve()),
           "sources": sources, "ON-Harmony results file used": str(onh),
           "gap": "coverage - target at stated 0.8, 0.9, 0.95 (negative: the envelope under-covers the rescan); 'mean' over the three. "
                  "A negative rho: more motion or eddy, a larger shortfall",
           "gap check against the results files": check,
           "proxies": {k: {"unit": u, "what": w} for k, (_, u, w) in PROXIES.items()},
           "proxy settings": {"smoothing sigma (voxels)": SMOOTH_VOX, "mask": "dipy median_otsu (radius 4, 4 passes) on the mean forward b0",
                              "alignment mask": f"the brain mask dilated {DILATE} voxels", "Gauss-Newton iterations (max)": GN_ITERS,
                              "prediction": "log-linear least-squares tensor fit of the smoothed b0s and tracked shell",
                              "b0 registration": "dipy AffineRegistration, rigid, MI (32 bins, sampling 0.5), levels 200/100/50, sigmas 2/1/0, factors 4/2/1, from the identity"},
           "typical ranges per study (per scan)": rng_,
           "gaps per stratum": gaps,
           "correlations within strata (Spearman, permutation p; exact when n <= 8)": corr,
           "within subject": within,
           "pooled (within-stratum ranks; ON-Harmony same scanner ranked within subject; stratified permutation p)": pooled,
           "proxy validation (per-volume dipy affine onto the prediction, a subsample)": val,
           "proxy checks": proxy_checks,
           "pairs": [{**p, "proxies": {k: pair_values(p, scans, k) for k in PROXIES}} for p in pairs],
           "scans": scans,
           "caveats": [
               "correlation only, across pairs that differ in more than motion; small n (cohort II 4 per protocol: the smallest exact two-sided p is 0.083; cohort III 8 and 9; ON-Harmony 20)",
               "many proxies x combinations x strata are tested: read single p-values below 0.05 with that multiplicity in mind",
               "the proxies are not an eddy correction: the DW alignment is to a tensor prediction of the scan itself (the contrast for each direction, the scan's average position); "
               "contrast the tensor misses can be read as shifts, and on a half-sphere scheme part of it can be linear in g",
               "the b0 motion is the b0s' only (cohort II's scanner A and B series have one forward b0: no b0 motion there); motion between b0s is not the motion of every DW volume",
               "residual_left mixes motion between the b0 pair with field-model error and noise; cohort II's single-pair scans have one b0 each way",
               "the swapped B2000 and site 3 pairs reuse B2000's scans (the same proxies, other roles): the pooled set without them is also given",
               "the rescan's registration onto A (rigid, mode 3) and the resampling are not part of these proxies; a pair's registration quality (NCC) is listed per pair"]}
    dest = Path(args.out) if args.out else OUT
    if args.cached_only and dest == OUT:
        sys.exit("--cached-only writes elsewhere (--out)")
    tmp = dest.with_suffix(".tmp")
    tmp.write_text(json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, dest)
    print("wrote", dest, flush=True)


# ----------------------------------------------------------------------------------------------- validation

def validate(args):
    """For a few scans: a dipy affine registration (MI) of a subsample of DW volumes onto their prediction; its PE
    displacement (mean over the mask and at the edge) against the Gauss-Newton proxy's, per volume."""
    from dipy.align.imaffine import AffineRegistration, MutualInformationMetric
    from dipy.align.transforms import AffineTransform3D
    out = {}
    for name in args.scans:
        s, shell, study = load_scan(name)
        mask = brain_mask(s.dwi[..., s.b0_index].astype(np.float32).mean(-1))
        d = dw_proxies(s, shell, mask, keep_params=True)
        P, sel, pred, sm, idx = d.pop("_params"), d.pop("_sel"), d.pop("_pred"), d.pop("_sm"), d.pop("_idx")
        nb = len(s.b0_index)
        vox, pe = np.asarray(s.vox, float), int(s.pe_axis)
        cen = idx.mean(0)
        epts = edge_points(mask)
        ec = (epts - cen) * vox
        mc = (np.argwhere(mask) - cen) * vox
        n = P.shape[0]
        pick = np.linspace(0, n - 1, min(args.volumes, n)).round().astype(int)
        rows = []
        A = s.affine
        Rdir = A[:3, :3] / np.linalg.norm(A[:3, :3], axis=0)          # voxel axes -> world directions
        for j in pick:
            np.random.seed(SEED)
            reg = AffineRegistration(metric=MutualInformationMetric(nbins=32, sampling_proportion=0.5),
                                     level_iters=[200, 100, 50], sigmas=[2.0, 1.0, 0.0], factors=[4, 2, 1], verbosity=0)
            static = pred[..., nb + j].astype(np.float64)
            moving = sm[..., nb + j].astype(np.float64)
            t0 = time.time()
            m = reg.optimize(static, moving, AffineTransform3D(), None, static_grid2world=A, moving_grid2world=A, starting_affine=np.eye(4))
            T = np.asarray(m.affine, float)

            def pe_disp(ijk_c):          # dipy's displacement along the PE voxel axis, mm, at centered voxel-axis points
                w = (A[:3, :3] @ (ijk_c / vox + cen).T).T + A[:3, 3]
                dw_ = (T[:3, :3] @ w.T).T + T[:3, 3] - w
                return dw_ @ Rdir[:, pe]
            p = P[j]

            def gn_disp(rc):
                return p[pe] + np.cross(p[3:6], rc)[:, pe] + rc @ p[6:9]
            rows.append({"volume (of the shell)": int(j), "seconds": round(time.time() - t0, 1),
                         "PE displacement, mean over the mask (dipy affine, proxy)": [round(float(pe_disp(mc).mean()), 3), round(float(gn_disp(mc).mean()), 3)],
                         "PE displacement at the edge, SD over edge voxels (dipy affine, proxy)": [round(float(pe_disp(ec).std()), 3), round(float(gn_disp(ec).std()), 3)],
                         "_edge": [pe_disp(ec), gn_disp(ec)]})
            print(name, rows[-1]["volume (of the shell)"], rows[-1]["PE displacement, mean over the mask (dipy affine, proxy)"],
                  rows[-1]["PE displacement at the edge, SD over edge voxels (dipy affine, proxy)"], rows[-1]["seconds"], "s", flush=True)
        a = np.array([r["PE displacement, mean over the mask (dipy affine, proxy)"] for r in rows])
        # per volume edge displacement, both methods, relative to the subsample's median volume
        Ed = np.array([r["_edge"][0] for r in rows]); Eg = np.array([r["_edge"][1] for r in rows])
        Ed -= np.median(Ed, 0); Eg -= np.median(Eg, 0)
        for r in rows:
            r.pop("_edge")
        out[name] = {"study": study, "volumes compared": len(rows), "per volume": rows,
                     "correlation over volumes, PE displacement mean over the mask": round(float(np.corrcoef(a[:, 0], a[:, 1])[0, 1]), 3),
                     "correlation, PE edge displacement (volumes x edge voxels, each relative to the subsample median)": round(float(np.corrcoef(Ed.ravel(), Eg.ravel())[0, 1]), 3),
                     "RMS difference, PE edge displacement mm": round(float(np.sqrt(((Ed - Eg) ** 2).mean())), 3),
                     "RMS PE edge displacement mm (dipy affine, proxy)": [round(float(np.sqrt((Ed ** 2).mean())), 3), round(float(np.sqrt((Eg ** 2).mean())), 3)],
                     "proxy summary (all volumes)": d}
    p = CACHE / "validation.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(old, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("per volume", "proxy summary (all volumes)")} for k, v in out.items()}, indent=1))


def scans_cmd(args):
    names = args.only
    if not names:
        onh = args.onharmony or (ONH_FINAL if ONH_FINAL.exists() else ONH_PARTIAL)
        pairs, check, _ = collect_pairs(onh)
        print(f"{len(pairs)} pairs; gap check: {check['compared']} compared, {len(check['mismatched'])} mismatched", flush=True)
        names = scan_names(pairs)
        if args.list:
            for m in check["mismatched"]:
                print("  MISMATCH", m)
            for p in pairs:
                print(p["stratum"], p["subject"], p["A"], p["rescan"], p["tag"], p["gap"], p["residual_left"])
            print(len(names), "scans")
            return
    if args.study:
        names = [n for n in names if study_of(n) in args.study]
    for i, n in enumerate(names):
        t0 = time.time()
        r = cached(n, force=args.force)
        print(f"[{i + 1}/{len(names)}] {n} {time.time() - t0:.1f} s", json.dumps({k: (PROXIES[k][0](r) if PROXIES[k][0] else None) for k in PROXIES if PROXIES[k][0]}),
              json.dumps(r["seconds"]), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("scans"); a.add_argument("--only", nargs="*"); a.add_argument("--force", action="store_true"); a.add_argument("--onharmony")
    a.add_argument("--list", action="store_true"); a.add_argument("--study", nargs="*")
    a = sub.add_parser("validate"); a.add_argument("--scans", nargs="+", required=True); a.add_argument("--volumes", type=int, default=12)
    a = sub.add_parser("evaluate"); a.add_argument("--onharmony"); a.add_argument("--cached-only", action="store_true")
    a.add_argument("--out")
    args = ap.parse_args()
    {"scans": scans_cmd, "validate": validate, "evaluate": evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
