"""Phase C, level 1 (docs/sota-pipeline-design.md, section 7.1): tractline.csd against DIPY's CSD on real scans, the step 6
benchmark's settings. Per scan, the corrected DWI's b0s and tracked shell, its mask (pipeline.brain_mask), and:
  - the response: ours (response_ssst) against DIPY's auto_response_ssst;
  - the FODs in the voxels with a fiber (FA >= 0.3 by our tensor fit, inside the mask): correlation of the amplitudes on
    300 x 2 directions and their relative L2 difference, for ours on DIPY's constraint sphere (the method), ours on our own
    sphere (the default), and ours on MPS (float32) against ours on the CPU (float64); and, for scale, DIPY's own sphere
    rotated at random against itself (where the constraint samples the sphere changes the FOD: 2026-10-24, cIIIs04, as
    much as our sphere does);
  - peaks, both FODs' found by the same DIPY procedure (_csd_dipy.py peaks): the share of voxels with the same number of
    peaks and the angle between first peaks;
  - times (ours on MPS and on the CPU; DIPY's fit and peaks, parallel).

    uv run bench/csd_check.py [--scans cIIIs04 PAT16]       # results/csd_check.json
"""
import argparse, json, os, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tractline import pipeline as P, csd, sh  # noqa: E402

DEVICE = "mps"


def scan_input(name):
    """(dwi b0s + shell float32, bval, bvec (3, G), affine, shell) of the corrected scan."""
    if name == "PAT16":
        import _ds001226 as L
        s, shell = L.load("PAT16"), 2800.0
    else:
        import calibration_runs as CR
        s, shell = CR.load_checked(name, *CR.CARD[name].ref), CR.SHELL
    with P.exact_float32(DEVICE):
        dwi = P.correct(s, P.Timer(), DEVICE).dwi
    P.release_memory(DEVICE)
    bval = np.asarray(s.bval, float)
    keep = (bval < 50) | (np.abs(bval - shell) < 50)
    return np.ascontiguousarray(dwi[..., keep], np.float32), bval[keep], np.asarray(s.bvec, float)[:, keep], shell


def dipy(what, src, out):
    t0 = time.time()
    subprocess.run([sys.executable, "-I", str(HERE / "_csd_dipy.py"), what, str(src), str(out)], check=True)
    return round(time.time() - t0, 1)


def agreement(a, b, B):
    """FOD amplitudes on B's directions: per-voxel correlation and relative L2 difference, summarized."""
    x, y = a @ B.T, b @ B.T
    xc, yc = x - x.mean(1, keepdims=True), y - y.mean(1, keepdims=True)
    corr = (xc * yc).sum(1) / np.sqrt((xc * xc).sum(1) * (yc * yc).sum(1))
    rel = np.linalg.norm(x - y, axis=1) / np.linalg.norm(y, axis=1)
    q = lambda v, p: round(float(np.quantile(v, p)), 5)  # noqa: E731
    return {"correlation: 1st percentile, median": [q(corr, 0.01), q(corr, 0.5)],
            "relative L2 difference: median, 99th percentile": [q(rel, 0.5), q(rel, 0.99)]}


def peaks_agree(pa, va, pb, vb, sel):
    na, nb = (va[sel] > 0).sum(-1), (vb[sel] > 0).sum(-1)
    both = (na > 0) & (nb > 0)
    cos = np.abs((pa[sel][both, 0] * pb[sel][both, 0]).sum(-1))
    ang = np.degrees(np.arccos(np.clip(cos, 0, 1)))
    return {"same number of peaks": round(float((na == nb).mean()), 4),
            "first peak angle, degrees: median, 95th percentile": [round(float(np.median(ang)), 2), round(float(np.quantile(ang, 0.95)), 2)],
            "first peaks within 5 degrees": round(float((ang <= 5).mean()), 4)}


def check(name, tmp):
    dwi, bval, bvec, shell = scan_input(name)
    mask = P.brain_mask(dwi, bval, P.Timer(), device=DEVICE)
    src, ref = tmp / f"{name}_in.npz", tmp / f"{name}_dipy.npz"
    np.savez(src, dwi=dwi, bval=bval, bvec=bvec, mask=mask)
    dipy_wall = dipy("fit", src, ref)
    z = np.load(ref)
    order, small = int(z["order"]), z["small_sphere"]
    t0 = time.time()
    resp = csd.response_ssst(dwi, bval, bvec)
    t_resp = time.time() - t0
    fits, times = {}, {}
    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))       # a random rotation, fixed
    for key, dev, sphere in (("cpu", "cpu", None), ("cpu, DIPY's sphere", "cpu", small), ("mps", "mps", None),
                             ("mps, DIPY's sphere rotated", "mps", small @ q.T)):
        model = csd.CSD(bval, bvec, resp, order=order, sphere=sphere)
        model.fit(dwi[:8, :8, :8], device=dev)                              # warm-up (kernels, allocator)
        t0 = time.time()
        fits[key], info = model.fit(dwi, mask, device=dev)
        times[key] = {"fit s": round(time.time() - t0, 2), **info}
    ev, _ = csd.tensor_fit(dwi[mask].astype(np.float64), bval, bvec)
    fiber = np.zeros(mask.shape, bool)
    fiber[mask] = csd.fa(ev) >= 0.3
    B = sh.basis(order, np.r_[csd.hemisphere(300), -csd.hemisphere(300)])
    ours = tmp / f"{name}_ours.npz"
    np.savez(ours, coeffs=fits["cpu"], mask=mask, order=order)
    dipy("peaks", ours, tmp / f"{name}_ours_peaks.npz")
    np.savez(tmp / f"{name}_ref_c.npz", coeffs=z["coeffs"], mask=mask, order=order)
    dipy("peaks", tmp / f"{name}_ref_c.npz", tmp / f"{name}_ref_peaks.npz")
    po, pr = np.load(tmp / f"{name}_ours_peaks.npz"), np.load(tmp / f"{name}_ref_peaks.npz")
    ref_c = z["coeffs"][fiber].astype(np.float64)
    return {
        "shell": shell, "directions": int((bval > 50).sum()), "order": order, "mask voxels": int(mask.sum()),
        "fiber voxels (FA >= 0.3)": int(fiber.sum()),
        "response": {"ours": {"evals": resp.evals.tolist(), "S0": resp.S0, "voxels": resp.voxels},
                     "DIPY": {"evals": z["evals"].tolist(), "S0": float(z["S0"])},
                     "relative difference, evals (max) and S0": [float(np.abs(resp.evals / z["evals"] - 1).max()), float(resp.S0 / z["S0"] - 1)]},
        "FOD against DIPY, fiber voxels": {"ours on DIPY's sphere (the method)": agreement(fits["cpu, DIPY's sphere"][fiber].astype(np.float64), ref_c, B),
                                           "ours on our sphere (the default)": agreement(fits["cpu"][fiber].astype(np.float64), ref_c, B)},
        "the constraint's own sensitivity: DIPY's sphere rotated at random against DIPY's, ours, fiber voxels":
            agreement(fits["mps, DIPY's sphere rotated"][fiber].astype(np.float64), fits["cpu, DIPY's sphere"][fiber].astype(np.float64), B),
        "MPS (float32) against CPU (float64), fiber voxels": agreement(fits["mps"][fiber].astype(np.float64), fits["cpu"][fiber].astype(np.float64), B),
        "peaks, ours (our sphere) against DIPY's, fiber voxels": peaks_agree(po["peak_dirs"], po["peak_values"], pr["peak_dirs"], pr["peak_values"], fiber),
        "seconds": {"ours: response": round(t_resp, 2), **{f"ours: {k}": v for k, v in times.items()},
                    "DIPY (subprocess wall, incl. loading)": dipy_wall, "DIPY": json.loads(str(z["seconds"]))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scans", nargs="+", default=["cIIIs04", "PAT16"])
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    out = {"note": args.note, "cpu_count": os.cpu_count(), "scans": {}}
    with tempfile.TemporaryDirectory() as d:
        for name in args.scans:
            out["scans"][name] = check(name, Path(d))
            print(name, json.dumps(out["scans"][name], indent=1), flush=True)
    (HERE / "results" / "csd_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
