"""One series, concatenate and pair-average on a real pair of full series (AP and PA): what each does to the misalignment
between the polarities, the signal, FA and the tracts. Run with a BIDS AP series beside its PA series:

    PYTHONPATH=src:bench python bench/reversed_series_compare.py <AP series> [--shell 2000] [--gibbs] [--name x]
                                                                 [--retest <the AP series of a second pair>]

The runs, each through pipeline.run with motion_eddy on: AP alone and PA alone (each with the other's b0s for the field,
as a run does today), concatenate, pair-average. Each alone is in the frame of its own first b0; concatenate and
pair-average are in AP's.
  misalignment  the same gradient's corrected AP and PA volumes, per pair: RMS of their difference over the brain, as a
                share of the mean signal, as acquired, corrected separately, and corrected together (concatenate's two
                halves); and the same on the mean of each shell, where noise averages away and what stays is geometry
  signal        the residual of a tensor fit to the tracked shell, over FA > 0.3
  FA            the two halves' agreement: AP alone against PA alone; AP's and PA's parts of the joint correction, each
                fit on its own (same frame). And each result against AP alone
  tracts        streamlines, labeled, and per tract the Dice of the voxels each run's streamlines cross, for the same pairs
  cost          seconds per stage, and the largest GPU memory seen
  test-retest   with --retest, a second pair from the same session: each mode against itself across the two pairs,
                run 2 put onto run 1 by a rigid fit of the mean b0s (tractline.mi): FA and tract agreement. This needs no
                ground truth: a mode that helps makes two independent acquisitions agree better

Writes results/reversed_series_compare_<name>.json (per-run outputs stay out of git until a NOTES entry cites one).
"""
import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
from tractline import bids, csd as C, pipeline as P
from tractline.labelers.rapidparc import Labeler

HERE = Path(__file__).parent


class Peak:
    """The largest GPU memory seen while the block runs, sampled every 0.1 s."""

    def __enter__(self):
        self.peak, self.go = 0, True

        def watch():
            while self.go:
                self.peak = max(self.peak, torch.mps.driver_allocated_memory())
                time.sleep(0.1)
        self.t = threading.Thread(target=watch, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.go = False
        self.t.join()


def tensor(dwi, bval, bvec, affine, shell, mask):
    """FA in the mask, and the residual of a log-linear tensor fit (RMS over the shell's volumes, about a share of the
    signal), on the b0s and the shell."""
    keep = (bval < 50) | (np.abs(bval - shell) < 50)
    g, b = P._image_axes(bvec[:, keep], affine).T, bval[keep]
    y = dwi[..., keep][mask].astype(np.float64)
    ev, _ = C.tensor_fit(y, b, g)
    fa = np.zeros(mask.shape)
    fa[mask] = C.fa(ev)
    u = g / np.clip(np.linalg.norm(g, axis=1, keepdims=True), 1e-9, None)
    X = np.c_[-b[:, None] * np.c_[u[:, 0] ** 2, 2 * u[:, 0] * u[:, 1], u[:, 1] ** 2, 2 * u[:, 0] * u[:, 2],
                                  2 * u[:, 1] * u[:, 2], u[:, 2] ** 2], np.ones(len(b))]
    ly = np.log(np.maximum(y, 1.0))
    res = np.zeros(mask.shape)
    res[mask] = np.sqrt(((ly - ly @ np.linalg.pinv(X).T @ X.T)[:, b >= 50] ** 2).mean(1))
    return fa, res


def density(tg, labels, affine, shape):
    """tract -> the boolean voxels its labeled streamlines cross."""
    inv = np.linalg.inv(affine)
    out = {}
    idx = np.flatnonzero(labels.keep)
    for row, f in enumerate(idx):
        t = int(labels.tract[row])
        v = np.rint(tg.fibers[f] @ inv[:3, :3].T + inv[:3, 3]).astype(int)
        v = v[((v >= 0) & (v < np.array(shape))).all(1)]
        m = out.setdefault(t, np.zeros(shape, bool))
        m[tuple(v.T)] = True
    return out


def dice(a, b, tract_other=42):
    keys = (set(a) | set(b)) - {tract_other}
    vals = [2 * (a[k] & b[k]).sum() / max(1, a[k].sum() + b[k].sum()) if k in a and k in b else 0.0 for k in sorted(keys)]
    return {"median over tracts": round(float(np.median(vals)), 3), "lowest": round(float(np.min(vals)), 3),
            "tracts": len(vals)}


def diff_share(a, b, mask, mean):
    """RMS over the brain, and the median and 90th percentile of the absolute difference: a gap can be mostly a few
    voxels of pile-up, so RMS alone reads a tail as misalignment."""
    d = np.abs((a - b)[mask]) / mean
    return {"rms": round(float(np.sqrt((d ** 2).mean())), 4), "median": round(float(np.median(d)), 4),
            "p90": round(float(np.percentile(d, 90)), 4)}


def run_one(s, lab, shell, gibbs, reversed_series=None):
    timer = P.Timer()
    gb = {"slice_axis": P.slice_axis_of(s)} if gibbs else None
    with Peak() as pk:
        r = P.run(s, lab, timer, device="mps", shell=shell, motion_eddy={"model": "quadratic"}, gibbs=gb,
                  reversed_series=reversed_series)
    c = r.correction
    bval = np.asarray(s.bval if c.bval is None else c.bval, float)
    keep = np.setdiff1d(np.arange(c.dwi.shape[-1]), [int(v) for v in c.motion_eddy["left out"]] if c.bval is None else [])
    return dict(dwi=c.dwi, bval=bval, bvec=c.bvec, run=r, seconds=dict(timer.seconds), peak=pk.peak / 2 ** 30, keep=keep)


def run_pair(path, a):
    """The four runs on one pair: {name: run}, and the subject with the reversed series."""
    both = bids.load(path, keep_reversed=True)
    print(both.pairing.message, flush=True)
    shell = a.shell or P.default_shell(both.bval)
    lab = Labeler("mps")
    runs = {}
    for name, s_, mode in (("AP", bids.load(path), None), ("PA", bids.load(both.reversed.path), None),
                           ("concatenate", both, "concatenate"), ("pair-average", both, "pair-average")):
        runs[name] = run_one(s_, lab, shell, a.gibbs, mode)
        print(name, runs[name]["seconds"], flush=True)
    return runs, both, shell


def rigid_to(fixed, moving, mask, vox):
    """T (4 x 4, voxels): run 1's voxels -> run 2's, from mi.register of the mean b0s (the moving image sampled at
    T x matches the fixed one)."""
    from tractline import mi
    from tractline.motion_eddy import rot
    dev = "mps"
    dd = mi.register(torch.tensor(moving, dtype=torch.float32, device=dev), torch.tensor(fixed, dtype=torch.float32, device=dev),
                     torch.tensor(mask, device=dev), np.asarray(vox), pe_axis=1, nmon=0).detach().cpu().double()
    R = rot(dd[3:6][None] * 0.01)[0].numpy()
    c = (np.array(fixed.shape, float) - 1) / 2
    v = np.asarray(vox, float)
    A = np.diag(1 / v) @ R @ np.diag(v)
    T = np.eye(4)
    T[:3, :3] = A
    T[:3, 3] = c - A @ c + dd[:3].numpy() / v
    return T, dd.numpy()


def retest(r1, r2, both1, both2, shell):
    """Each mode's agreement with itself across two pairs."""
    from scipy.ndimage import affine_transform
    b01 = r1["AP"]["dwi"][..., r1["AP"]["bval"] < 50].mean(-1)
    b02 = r2["AP"]["dwi"][..., r2["AP"]["bval"] < 50].mean(-1)
    mask = np.asarray(r1["AP"]["run"].tractogram.mask, bool)
    T, dd = rigid_to(b01, b02, mask, both1.vox)
    out = {"run 2 onto run 1, mm and 0.01 rad": np.round(dd, 3).tolist()}
    A1 = np.asarray(both1.affine, float)
    to1 = A1 @ np.linalg.inv(T) @ np.linalg.inv(np.asarray(both2.affine, float))       # run 2's RAS -> run 1's RAS
    shape = both1.dwi.shape[:3]
    for name in ("AP", "PA", "concatenate", "pair-average"):
        a, b = r1[name], r2[name]
        fa1, _ = tensor(a["dwi"], a["bval"], a["bvec"], both1.affine, shell, mask)
        m2 = np.asarray(b["run"].tractogram.mask, bool)
        fa2, _ = tensor(b["dwi"], b["bval"], b["bvec"], both2.affine, shell, m2)
        fa2 = affine_transform(fa2, T[:3, :3], T[:3, 3], order=1)
        wm = (fa1 > 0.2) & (fa2 > 0.2)
        fib = [f @ to1[:3, :3].T + to1[:3, 3] for f in b["run"].tractogram.fibers]
        d1 = density(a["run"].tractogram, a["run"].labels, A1, shape)
        d2 = density(type("T", (), {"fibers": fib}), b["run"].labels, A1, shape)
        out[name] = {"mean |dFA| in the mask": round(float(np.abs(fa1 - fa2)[mask].mean()), 4),
                     "FA correlation in the mask": round(float(np.corrcoef(fa1[mask], fa2[mask])[0, 1]), 4),
                     "mean |dFA| where FA > 0.2": round(float(np.abs(fa1 - fa2)[wm].mean()), 4),
                     "tract Dice": dice(d1, d2),
                     "labeled": [int(a["run"].labels.keep.sum()), int(b["run"].labels.keep.sum())]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series")
    ap.add_argument("--shell", type=float, default=None)
    ap.add_argument("--gibbs", action="store_true")
    ap.add_argument("--name", default=None)
    ap.add_argument("--retest", default=None)
    a = ap.parse_args()
    path = Path(a.series)
    name = a.name or path.name.split("_dir-")[0]
    runs, both, shell = run_pair(path, a)
    pa_path = both.reversed.path

    A, B, J, Q = (runs[k] for k in ("AP", "PA", "concatenate", "pair-average"))
    G = both.dwi.shape[-1]
    mask = np.asarray(J["run"].tractogram.mask, bool)
    from tractline.series import match
    m = match(both.bval, both.bvec, both.reversed.bval, both.reversed.bvec)
    dw = [(i, j) for i, j in m.pairs if abs(both.bval[i] - shell) < 50]
    mean = float(np.mean(J["dwi"][mask][:, [i for i, _ in dw]]))
    k = float(J["run"].correction.reversed_series["intensity scale applied to the reversed series"])

    def pairwise(x, y):
        return round(float(np.median([np.sqrt(((np.asarray(x[..., i], float) - y[..., j]) [mask] ** 2).mean()) / mean
                                      for i, j in dw])), 4)
    raw_a, raw_b = both.dwi, np.asarray(both.reversed.dwi, np.float32) * np.float32(k)
    out = {"series": [path.name, pa_path.name], "volumes": [int(both.dwi.shape[-1]), int(both.reversed.dwi.shape[-1])],
           "shell": shell, "pairs in the shell": len(dw), "brain voxels": int(mask.sum())}
    joint_b = J["dwi"][..., G:]
    means = lambda d, idx: d[..., idx].mean(-1)                             # noqa: E731
    ia, ib = [i for i, _ in dw], [j for _, j in dw]
    out["misalignment (share of the mean signal)"] = {
        "per pair, median": {"as acquired": pairwise(raw_a, raw_b), "corrected separately": pairwise(A["dwi"], B["dwi"]),
                             "corrected together": pairwise(J["dwi"], joint_b)},
        "shell means": {"as acquired": diff_share(means(raw_a, ia), means(raw_b, ib), mask, mean),
                        "corrected separately": diff_share(means(A["dwi"], ia), means(B["dwi"], ib), mask, mean),
                        "corrected together": diff_share(means(J["dwi"], ia), means(joint_b, ib), mask, mean)}}
    b0a, b0b = np.flatnonzero(both.bval < 50), np.flatnonzero(both.reversed.bval < 50)
    m0 = float(J["dwi"][mask][:, b0a].mean())
    out["misalignment (share of the mean signal)"]["mean b0s"] = {
        "as acquired": diff_share(means(raw_a, b0a), means(raw_b, b0b), mask, m0),
        "corrected separately": diff_share(means(A["dwi"], b0a), means(B["dwi"], b0b), mask, m0),
        "corrected together": diff_share(means(J["dwi"], b0a), means(joint_b, b0b), mask, m0)}
    fits = {n: tensor(r["dwi"], r["bval"], r["bvec"], both.affine, shell, mask) for n, r in runs.items()}
    fj_a = tensor(J["dwi"][..., :G], J["bval"][:G], J["bvec"][:, :G], both.affine, shell, mask)
    fj_b = tensor(joint_b, J["bval"][G:], J["bvec"][:, G:], both.affine, shell, mask)
    wm = fits["pair-average"][0] > 0.3
    out["tensor residual over FA > 0.3 (log signal, RMS)"] = {n: round(float(f[1][wm].mean()), 4) for n, f in fits.items()}
    fa_d = lambda x, y: {"mean |dFA|": round(float(np.abs(x - y)[mask].mean()), 4),   # noqa: E731
                         "correlation": round(float(np.corrcoef(x[mask], y[mask])[0, 1]), 4)}
    out["FA agreement"] = {"AP alone vs PA alone": fa_d(fits["AP"][0], fits["PA"][0]),
                           "AP part vs PA part of the joint correction": fa_d(fj_a[0], fj_b[0]),
                           "concatenate vs AP alone": fa_d(fits["concatenate"][0], fits["AP"][0]),
                           "pair-average vs AP alone": fa_d(fits["pair-average"][0], fits["AP"][0]),
                           "concatenate vs pair-average": fa_d(fits["concatenate"][0], fits["pair-average"][0])}
    dens = {n: density(r["run"].tractogram, r["run"].labels, both.affine, both.dwi.shape[:3]) for n, r in runs.items()}
    out["tracts"] = {"streamlines, labeled": {n: [len(r["run"].tractogram.fibers), int(r["run"].labels.keep.sum())]
                                              for n, r in runs.items()},
                     "Dice AP alone vs PA alone": dice(dens["AP"], dens["PA"]),
                     "Dice concatenate vs AP alone": dice(dens["concatenate"], dens["AP"]),
                     "Dice pair-average vs AP alone": dice(dens["pair-average"], dens["AP"]),
                     "Dice concatenate vs pair-average": dice(dens["concatenate"], dens["pair-average"])}
    out["record"] = {n: r["run"].correction.reversed_series for n, r in runs.items() if r["run"].correction.reversed_series}
    out["warnings"] = {n: r["run"].correction.warnings for n, r in runs.items()}
    out["seconds"] = {n: r["seconds"] for n, r in runs.items()}
    out["peak GPU driver memory, GB"] = {n: round(r["peak"], 2) for n, r in runs.items()}
    (HERE / f"results/reversed_series_compare_{name}.json").write_text(json.dumps(out, indent=1, default=str))
    if a.retest:
        runs2, both2, _ = run_pair(Path(a.retest), a)
        out["test-retest"] = retest(runs, runs2, both, both2, shell)
    (HERE / f"results/reversed_series_compare_{name}.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("warnings", "seconds")}, indent=1, default=str))


if __name__ == "__main__":
    main()
