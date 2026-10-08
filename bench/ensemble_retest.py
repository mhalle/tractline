"""The bootstrap against real rescans: does a wild-bootstrap replicate of a scan change the pipeline's labels as much
as rescanning the subject does? (NOTES 2026-10-08, "The bootstrap against real rescans")

    uv run --group dipy --group rankfield bench/ensemble_retest.py run [--pair cIIIs01-2.5] [--runs 8] [--flip voxel]
    uv run --group dipy --group rankfield bench/ensemble_retest.py analyze [--pair cIIIs01-2.5]

Data: MASiVar (OpenNeuro ds003416; bench/fetch_ds003416.py, bench/_masivar.py), scans repeated on one Philips scanner.
A pair is a reference scan A and its rescans (within the session; or another session, another day).

run: every scan through the pipeline (P.correct, P.track, the default labeler) and `runs` "added" bootstrap
replicates of A (_bootstrap.ShellWildBootstrap, A's mask; flip as given), as bench/ensemble_pilot.py runs them.
Every run is labeled with draw 0: the labeler's draw is the same for all, so only the data's share is counted. A
rescan's corrected mean b0 is registered rigidly to A's (bench/_register.py: dipy, mutual information, in a process
of its own). Each rescan is then recorded twice on A's grid:
  <tag>      tracked on its own grid, its streamlines moved into A's space by the transform (no data interpolated;
             its seeds are not A's voxels);
  <tag>_onA  its corrected DWI resampled onto A's grid (cubic; b-vectors turned by the registration) and tracked
             there: its seeds are A's voxels, so seed keys match.
Resampling smooths the noise, so two controls: A_fb, A's corrected DWI sent onto the first rescan's grid and back
(its last step is <tag>_onA's), and the first `fb` replicates sent the same way (<flip>/r001_fb..). Records in
DATA/ensemble/<subject>/retest/<pair>/; runs.json beside them.

analyze: label change and voxel winner agreement, measured alike for rescans and replicates. Streamlines: by seed key
(the share of matched seeds whose tract differs), rescan <tag>_onA against A and against A_fb, beside replicate
against replicate (native, and there and back). Voxels: each run's winner per voxel of A's grid
(VoxelEnsemble.winner, tau 1), agreement over the voxels where either run has a tract; <tag> needs no interpolation.
--nearest adds nearest-path matching (MDF), which cannot tell rescans from replicates: a run against itself, each
streamline matched to its nearest other one, changes as many labels (NOTES). Also the residuals' spatial correlation
(lag-1, per axis, in the mask), which per-voxel sign flips remove. Writes results/ensemble_retest_<pair>.json.
"""
import argparse, json, sys, time
from dataclasses import dataclass, replace
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from tractline import pipeline as P
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results/ensemble_retest_{}.json"
DEVICE = "mps"
PAIRS = {
    # name: subject, the reference scan (session, scan), the rescans {tag: (session, scan)}, shells used, the shell tracked
    "cIIIs01-2.5": ("cIIIs01", ("s1Bx1", 2), {"within": ("s1Bx1", 4)}, None, 2000.0),
    "cIIIs01-2.1": ("cIIIs01", ("s1Bx3", 1), {"within": ("s1Bx3", 2), "between": ("s1Bx1", 1)}, None, 2000.0),
    "cIs1": ("cIs1", ("s1Ax1", 1), {"within": ("s1Ax1", 2), "between": ("s1Ax2", 1)}, (1000, 2000, 3000), 3000.0),
}


@dataclass
class Moved:
    """A Tractogram's fibers and stats, the fibers moved by a rigid transform (for RunRecord.of)."""
    fibers: list
    stats: dict


def record_dir(pair):
    d = DATA / "ensemble" / PAIRS[pair][0] / "retest" / pair
    d.mkdir(parents=True, exist_ok=True)
    return d


def mean_b0(dwi, bval):
    return np.asarray(dwi, np.float32)[..., np.asarray(bval) < 50].mean(-1)


def register(moving, moving_affine, static, static_affine):
    """Rigid, moving onto static (bench/_register.py, in a process of its own): (4x4 taking moving world -> static
    world, normalized cross-correlation after)."""
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        a, b = Path(tmp) / "in.npz", Path(tmp) / "out.npz"
        np.savez(a, moving=moving, moving_affine=moving_affine, static=static, static_affine=static_affine)
        subprocess.run([sys.executable, str(HERE / "_register.py"), str(a), str(b)], check=True)
        z = np.load(b)
        return z["to_static"], float(z["ncc"])


def run(args):
    from _masivar import load
    from _bootstrap import ShellWildBootstrap
    subject, (ses, scan), rescans, shells, shell = PAIRS[args.pair]
    d = record_dir(args.pair)
    log_path = d / "runs.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}
    labeler = P.default_labeler(DEVICE)

    def save_log():
        log_path.write_text(json.dumps(log, indent=1))

    def one(s, name, to_ref=None, grid=None, corrected=None):
        """Correct (unless given the corrected DWI), track, label (draw 0); the record on the reference's grid (moved
        there by to_ref)."""
        timer = P.Timer()
        t0 = time.time()
        with P.exact_float32(DEVICE):
            corr = P.correct(s, timer, DEVICE) if corrected is None else P.Correction(corrected, None, None, None, applied=True, note="given")
            tg = P.track(s, corr.dwi, timer, device=DEVICE, shell=shell)
            with timer("label"):
                labels = labeler(tg.fibers, draws=(0,), logp=True)
        moved = tg if to_ref is None else Moved([f @ to_ref[:3, :3].T + to_ref[:3, 3] for f in tg.fibers], tg.stats)
        E.RunRecord.of(moved, labels, *(grid or (s.affine, s.dwi.shape[:3]))).save(d / f"{name}.npz")
        log[name] = {"fibers": len(tg.fibers), "labeled": int(labels.keep.sum()), "residual_left": corr.residual_left,
                     "field_hz_p99_abs": round(float(np.quantile(np.abs(corr.field_hz), 0.99)), 2) if corr.field_hz is not None else None,
                     "seconds": timer.seconds, "total_s": round(time.time() - t0, 1)}
        print(name, json.dumps(log[name]), flush=True)
        P.release_memory(DEVICE)
        return corr, tg

    A = load(subject, ses, scan, shells)
    grid = (A.affine, A.dwi.shape[:3])
    if not (d / "A.npz").exists() or not (d / "A_meanb0.npy").exists():
        log["A_scan"] = {"name": A.name, "series": A.series, "notes": A.notes, "shape": list(A.dwi.shape)}
        corr, tg = one(A, "A")
        np.save(d / "A_meanb0.npy", mean_b0(corr.dwi, A.bval)); np.save(d / "A_mask.npy", tg.mask)
        save_log()
    ref_b0, mask = np.load(d / "A_meanb0.npy"), np.load(d / "A_mask.npy")
    for tag, (ses_b, scan_b) in rescans.items():
        if (d / f"{tag}.npz").exists():
            continue
        B = load(subject, ses_b, scan_b, shells)
        timer = P.Timer()
        with P.exact_float32(DEVICE):
            cb = P.correct(B, timer, DEVICE)
        to_ref, ncc = register(mean_b0(cb.dwi, B.bval), B.affine, ref_b0, A.affine)
        P.release_memory(DEVICE)
        rot = np.degrees(np.arccos(np.clip((np.trace(to_ref[:3, :3]) - 1) / 2, -1, 1)))
        log[f"{tag}_scan"] = {"name": B.name, "series": B.series, "shape": list(B.dwi.shape),
                              "to_A": np.round(to_ref, 5).tolist(), "rotation_deg": round(float(rot), 2),
                              "translation_mm": round(float(np.linalg.norm(to_ref[:3, 3])), 2), "ncc_after": round(ncc, 4)}
        one(B, tag, to_ref, grid)
        save_log()
    # each rescan's corrected DWI resampled onto A's grid and tracked there: its seeds are A's voxels, so seed keys
    # match; and A sent through the first rescan's resampling and back (two interpolations), the control
    for tag, (ses_b, scan_b) in rescans.items():
        if (d / f"{tag}_onA.npz").exists():
            continue
        B = load(subject, ses_b, scan_b, shells)
        to_ref = np.array(log[f"{tag}_scan"]["to_A"])
        with P.exact_float32(DEVICE):
            cb = P.correct(B, P.Timer(), DEVICE)
        P.release_memory(DEVICE)
        dwi, bvec = onto_reference(cb.dwi, B.bvec, B.affine, to_ref, A.affine, A.dwi.shape[:3])
        one(replace(A, dwi=dwi, bval=B.bval, bvec=bvec, b0s=None), f"{tag}_onA", corrected=dwi)
        save_log()
    tag0 = next(iter(rescans))
    B0 = load(subject, *rescans[tag0], shells)
    to_ref0 = np.array(log[f"{tag0}_scan"]["to_A"])

    def there_and_back(s, name):
        """s corrected, resampled onto the first rescan's grid and back onto A's (B_onA's last step exactly), tracked."""
        with P.exact_float32(DEVICE):
            cs = P.correct(s, P.Timer(), DEVICE)
        P.release_memory(DEVICE)
        there = resample(cs.dwi, np.linalg.inv(A.affine) @ to_ref0 @ B0.affine, B0.dwi.shape[:3])
        back, _ = onto_reference(there, A.bvec, B0.affine, to_ref0, A.affine, A.dwi.shape[:3], rotate=False)
        one(replace(s, dwi=back, b0s=None), name, corrected=back)
        log[name]["through"] = tag0
        save_log()

    if not (d / "A_fb.npz").exists():
        there_and_back(A, "A_fb")
    rd = d / args.flip
    rd.mkdir(exist_ok=True)
    wb = ShellWildBootstrap(A.dwi, A.bval, A.bvec, extra_b0s=A.b0s[..., len(A.b0_index):], b0_noise="added", mask=mask,
                            flip=args.flip, fwhm=args.fwhm)
    log[f"bootstrap_{args.flip}"] = wb.noise_check(mask)
    save_log()
    for r in range(1, args.runs + 1):
        if (rd / f"r{r:03d}.npz").exists():
            continue
        dwi, apa = wb.replicate(r)
        subj = replace(A, dwi=dwi, b0s=np.concatenate([dwi[..., A.b0_index].astype(np.float64), apa], -1))
        one(subj, f"{args.flip}/r{r:03d}")
        save_log()
    # the first replicates sent there and back too: the bootstrap's pairs with the interpolation A_fb and B_onA have
    for r in range(1, args.fb + 1):
        if (rd / f"r{r:03d}_fb.npz").exists():
            continue
        dwi, apa = wb.replicate(r)
        there_and_back(replace(A, dwi=dwi, b0s=np.concatenate([dwi[..., A.b0_index].astype(np.float64), apa], -1)), f"{args.flip}/r{r:03d}_fb")


def resample(dwi, M, shape):
    """(X, Y, Z, G) sampled at M @ (output voxel) (cubic, outside 0, clipped at 0) -> shape + (G,) float32."""
    from scipy.ndimage import affine_transform
    d = np.asarray(dwi, np.float32)
    return np.stack([affine_transform(d[..., g], M[:3, :3], M[:3, 3], output_shape=tuple(shape), order=3, mode="constant", cval=0.0)
                     for g in range(d.shape[-1])], -1).clip(0, None).astype(np.float32)


def rotation(M):
    """The nearest rotation to a 3x3 matrix (its polar factor): an affine's directions without voxel sizes or shear."""
    u, _, vt = np.linalg.svd(M)
    return u @ vt


def onto_reference(dwi, bvec, affine_b, to_ref, affine_a, shape_a, rotate=True):
    """A rescan's DWI (on its grid, affine_b) onto the reference's grid (affine_a, shape_a), given to_ref (rescan world ->
    reference world): (dwi, bvec). FSL b-vectors are in voxel axes (x flipped when the affine's determinant is
    positive), so they are taken to the world, turned by the registration and back into the reference's axes."""
    out = resample(dwi, np.linalg.inv(affine_b) @ np.linalg.inv(to_ref) @ affine_a, shape_a)
    if not rotate:
        return out, bvec
    F = lambda a: np.diag([-1.0, 1.0, 1.0]) if np.linalg.det(a[:3, :3]) > 0 else np.eye(3)
    T = F(affine_a) @ rotation(affine_a[:3, :3]).T @ rotation(to_ref[:3, :3]) @ rotation(affine_b[:3, :3]) @ F(affine_b)
    return out, T @ np.asarray(bvec, float)


# ----------------------------------------------------------------------------------------------- analysis

def nearest(X: np.ndarray, Y: np.ndarray, self_match=False, k=8):
    """For each path of X (n, P, 3), the nearest path of Y (m, P, 3) by MDF: (index in Y, MDF mm). Candidates are
    the k nearest in the flattened coordinates, Y in both orientations; self_match: X is Y, its own row excluded."""
    n, m = len(X), len(Y)
    tree = cKDTree(np.concatenate([Y.reshape(m, -1), Y[:, ::-1].reshape(m, -1)]).astype(np.float64))
    kk = k + 2 if self_match else k
    _, idx = tree.query(X.reshape(n, -1).astype(np.float64), k=kk, workers=-1)
    cand = idx % m
    if self_match:
        cand = np.where(cand == np.arange(n)[:, None], -1, cand)
    best_i, best_d = np.full(n, -1), np.full(n, np.inf)
    for c in range(kk):
        j = cand[:, c]
        ok = j >= 0
        dd = np.full(n, np.inf)
        dd[ok] = E.mdf(X[ok], Y[j[ok]])
        better = dd < best_d
        best_i[better], best_d[better] = j[better], dd[better]
    return best_i, best_d


def path_change(a: E.RunRecord, b: E.RunRecord, limits=(2.0, 5.0)):
    """Label change on nearest-path matches, a -> b and b -> a pooled (b is a: against itself)."""
    same = a is b
    out = {}
    tracts = []
    for x, y in ((a, b),) if same else ((a, b), (b, a)):
        j, dd = nearest(x.paths, y.paths, self_match=same)
        tracts.append((LUT[x.cluster], LUT[y.cluster[j]], dd))
    for lim in limits:
        ch = np.concatenate([(tx != ty)[dd <= lim] for tx, ty, dd in tracts])
        n = sum(int((dd <= lim).sum()) for _, _, dd in tracts)
        tot = sum(len(dd) for _, _, dd in tracts)
        out[f"mdf<={lim:g}mm"] = {"changed": round(float(ch.mean()), 4), "matched": round(n / tot, 4)}
    dd = np.concatenate([t[2] for t in tracts])
    out["mdf_mm_quantiles (0.5, 0.9)"] = [round(float(v), 2) for v in np.quantile(dd, [0.5, 0.9])]
    return out


def key_change(a: E.RunRecord, b: E.RunRecord):
    S = E.StreamlineEnsemble(a)
    ri, qi = S.match(b)
    return {"changed": round(float((LUT[a.cluster[ri]] != LUT[b.cluster[qi]]).mean()), 4), "matched": round(len(ri) / len(a.keys), 4),
            "median_mdf_mm": round(float(np.median(E.mdf(a.paths[ri], b.paths[qi]))), 2)}


def voxel_agree(a: E.RunRecord, b: E.RunRecord, tau=1.0):
    V = E.VoxelEnsemble(a.shape, tau)
    wa, wb = V.winner(a.counts), V.winner(b.counts)
    m = (wa != E.NONE) | (wb != E.NONE)
    both = (wa != E.NONE) & (wb != E.NONE)
    return {"agree (either has a tract)": round(float((wa == wb)[m].mean()), 4), "agree (both have a tract)": round(float((wa == wb)[both].mean()), 4),
            "voxels": int(m.sum())}


def tract_shares(r: E.RunRecord):
    return np.bincount(LUT[r.cluster], minlength=E.N_TRACTS) / len(r.cluster)


def mean_of(rows, key_path):
    vals = []
    for r in rows:
        v = r
        for k in key_path:
            v = v[k]
        vals.append(v)
    return {"mean": round(float(np.mean(vals)), 4), "min": round(float(np.min(vals)), 4), "max": round(float(np.max(vals)), 4), "n": len(vals)}


def residual_correlation(wb, mask):
    """Lag-1 correlation of the bootstrap's residuals between neighbouring voxels along each axis, in the mask (DW
    volumes and b0s apart): what the scan's noise shares with its neighbours, which per-voxel flips remove."""
    X, Y, Z, G = wb.shape
    R = wb.res.reshape(X, Y, Z, G)
    dw = np.setdiff1d(np.arange(G), wb.b0)
    out = {}
    for name, vols in (("dw", dw), ("b0", wb.b0)):
        row = []
        for ax in range(3):
            a = np.moveaxis(R[..., vols], ax, 0)
            mk = np.moveaxis(mask, ax, 0)
            both = mk[1:] & mk[:-1]
            u, v = a[1:][both], a[:-1][both]
            row.append(round(float((u * v).sum() / np.sqrt((u * u).sum() * (v * v).sum())), 3))
        out[name] = row
    return out


def analyze(args):
    from _masivar import load
    from _bootstrap import ShellWildBootstrap
    subject, (ses, scan), rescans, shells, shell = PAIRS[args.pair]
    d = record_dir(args.pair)
    log = json.loads((d / "runs.json").read_text())
    load_rec = lambda name: E.RunRecord.load(d / f"{name}.npz", logp=False)
    A = load_rec("A")
    have = lambda name: (d / f"{name}.npz").exists()

    def compare(a, b, keys=True):
        row = {"voxels": voxel_agree(a, b), "tract_share_L1": round(float(np.abs(tract_shares(a) - tract_shares(b)).sum()), 4)}
        if keys:
            row["seed_keys"] = key_change(a, b)
        if args.nearest:
            row["paths"] = path_change(a, b)
        return row

    res = {"pair": args.pair, "subject": subject, "reference": log.get("A_scan", {}).get("name"),
           "rescans": {t: {k: log[f"{t}_scan"][k] for k in ("name", "rotation_deg", "translation_mm", "ncc_after")} for t in rescans
                       if f"{t}_scan" in log},
           "runs_log": {k: v for k, v in log.items() if not k.endswith("_scan")}}
    if args.nearest:
        res["floor: a run against itself (nearest other path)"] = path_change(A, A)
    # rescans: tracked on their own grid (streamlines moved into A's space; no seed keys), and on A's grid (keys match)
    res["rescan"] = {}
    for t in rescans:
        row = {}
        if have(t):
            row["own grid, moved"] = compare(A, load_rec(t), keys=False)
        if have(f"{t}_onA"):
            row["on A's grid"] = compare(A, load_rec(f"{t}_onA"))
        res["rescan"][t] = row
    if have("A_fb"):
        Afb = load_rec("A_fb")
        res["control: A resampled there and back"] = {"through": log["A_fb"].get("through"), **compare(A, Afb)}
        for t in rescans:
            if have(f"{t}_onA"):
                res["rescan"][t]["on A's grid, against A there and back"] = compare(Afb, load_rec(f"{t}_onA"))
    res["bootstrap"] = {}
    for flip in sorted(p.name for p in d.iterdir() if p.is_dir()):
        names = sorted(p.stem for p in (d / flip).glob("r???.npz"))
        fb = sorted(p.stem for p in (d / flip).glob("r???_fb.npz"))
        if len(names) < 2:
            continue
        R = [E.RunRecord.load(d / flip / f"{n}.npz", logp=False) for n in names]
        rr = [(i, i + 1) for i in range(0, len(R) - 1, 2)] + [(i, i + 2) for i in range(0, len(R) - 2, 2)]   # disjoint and overlapping pairs
        rep_rep = [compare(R[i], R[j]) for i, j in rr]
        ref_rep = [compare(A, r) for r in R]
        keys = [("seed keys changed", ("seed_keys", "changed")), ("seed keys median mdf", ("seed_keys", "median_mdf_mm")),
                ("seed keys matched", ("seed_keys", "matched")),
                ("voxels agree (either)", ("voxels", "agree (either has a tract)")), ("voxels agree (both)", ("voxels", "agree (both have a tract)")),
                ("tract share L1", ("tract_share_L1",))]
        if args.nearest:
            keys += [("paths mdf<=5mm changed", ("paths", "mdf<=5mm", "changed")), ("paths median mdf", ("paths", "mdf_mm_quantiles (0.5, 0.9)", 0))]
        summ = lambda rows: {name: mean_of(rows, path) for name, path in keys}
        fields = [log[f"{flip}/{n}"]["field_hz_p99_abs"] for n in names if f"{flip}/{n}" in log]
        if len(fb) >= 2:
            F = [E.RunRecord.load(d / flip / f"{n}.npz", logp=False) for n in fb]
            fpairs = [(i, i + 1) for i in range(0, len(F) - 1, 2)] + [(i, i + 2) for i in range(0, len(F) - 2, 2)]
            res.setdefault("bootstrap there and back", {})[flip] = {"runs": fb, "replicate vs replicate": summ([compare(F[i], F[j]) for i, j in fpairs])}
        res["bootstrap"][flip] = {"runs": names, "replicate vs replicate": summ(rep_rep), "A vs replicate": summ(ref_rep),
                                  "field_hz_p99_abs": [min(fields), max(fields)] if fields else None,
                                  "noise_check": log.get(f"bootstrap_{flip}")}
    # the scan's noise: its residuals' spatial correlation
    S = load(subject, ses, scan, shells)
    mask = np.load(d / "A_mask.npy")
    wb = ShellWildBootstrap(S.dwi, S.bval, S.bvec, extra_b0s=S.b0s[..., len(S.b0_index):], b0_noise="added", mask=mask)
    res["residual lag-1 correlation (x, y, z)"] = residual_correlation(wb, mask)
    OUT.with_name(OUT.name.format(args.pair)).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in res if k != "runs_log"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "analyze"))
    ap.add_argument("--pair", default="cIIIs01-2.5", choices=sorted(PAIRS))
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--flip", default="voxel", choices=("voxel", "volume", "smooth"))
    ap.add_argument("--fwhm", type=float, default=2.0, help="flip smooth: the sign field's smoothing, voxels FWHM")
    ap.add_argument("--fb", type=int, default=4, help="run: replicates also sent through the rescan's resampling and back")
    ap.add_argument("--nearest", action="store_true", help="analyze: also nearest-path matching (slow; at its floor, NOTES)")
    args = ap.parse_args()
    run(args) if args.what == "run" else analyze(args)
