"""tractline.motion_eddy (the library) on real scans, two checks (MPS):

  proto   the library against the prototype it came from (proto.py with phase 0's settings: 40 steps, no coarse steps,
          no shell anchor, output masking, lambda 0.06) on a whole scan as acquired, no field: how far apart the two
          sampling maps are at the brain's edge (voxels; the library's inverse map takes Newton steps where the
          prototype took two fixed-point passes, and its predictor is built from another, equivalent SH basis), the
          rotated b-vectors' angle, the corrected series' difference, and the times.
  field   the field composed into the single resample, against phase 0's order (the field applied first, then the
          step on the field-corrected series: two interpolations), on an ON-Harmony scan with its reversed b0s (b0s and
          b2000, as the ensemble keeps them): each corrected series' leave-one-out residual in the brain (the same
          predictor for both; lower is better aligned and less blurred), the two maps' difference at the edge, the
          times.

    PYTHONPATH=src:bench python bench/motion_eddy/library_check.py proto [--scan 16974/OXF1PRI001]
    PYTHONPATH=src:bench python bench/motion_eddy/library_check.py field [--subject 16974 --session OXF1PRI001]
Writes results/motion_eddy/library_check_<what>_<subject>.json.
"""
import argparse, json, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tractline import motion_eddy as ME                                          # noqa: E402
from tractline.data import DATA                                                  # noqa: E402

OUT = HERE.parent / "results" / "motion_eddy"
PROTO_ARGS = ["--device", "mps", "--all-shells", "--fast", "--metal", "--coarse", "0", "--outer", "40", "--mask-output"]


def edge_points(mask):
    from scipy.ndimage import binary_erosion
    return torch.tensor(np.argwhere(mask & ~binary_erosion(mask)), dtype=torch.float64)


def map_difference(a, b, shape, vox, pe, nmon, mask):
    """RMS and p95 over volumes and the mask's edge of |phi_a - phi_b| (voxels; no field: it is the same in both)."""
    geo = ME.Geometry(shape, vox, pe, nmon, "cpu", torch.float64)
    xe = edge_points(mask)
    d = (ME.phi(geo, torch.tensor(a), xe) - ME.phi(geo, torch.tensor(b), xe)).norm(dim=-1)
    return [round(float(d.pow(2).mean().sqrt()), 4), round(float(torch.quantile(d.reshape(-1)[:2 ** 24], 0.95)), 4)]


def angles(a, b, dw):
    c = np.abs((a[:, dw] * b[:, dw]).sum(0) / np.linalg.norm(a[:, dw], axis=0) / np.linalg.norm(b[:, dw], axis=0))
    ang = np.degrees(np.arccos(c.clip(0, 1)))
    return [round(float(np.median(ang)), 4), round(float(ang.max()), 4)]


def loo_residual(corr, bval, bvec, mask):
    """RMS over the brain (eroded 2) of each corrected volume minus its leave-one-out prediction (the library's
    predictor), per group: lower is more consistent across volumes."""
    from scipy.ndimage import binary_erosion
    groups, bs = ME.shell_groups(bval)
    isb0 = np.asarray(bval) < 50
    g = np.zeros((len(bval), 3))
    g[~isb0] = (bvec[:, ~isb0] / np.linalg.norm(bvec[:, ~isb0], axis=0)).T
    W, _ = ME.loo_smoother(groups, isb0, g, 6, 0.06)
    m = binary_erosion(mask, iterations=2)
    S = corr[m].astype(np.float64)                                                 # (N, G)
    r = S - S @ W.T
    return {f"b{b}": round(float(np.sqrt((r[:, idx] ** 2).mean())), 3) for b, idx in zip(bs, groups) if len(idx) > 1}


def proto_check(args):
    import nibabel as nib
    sub, ses = args.scan.split("/")
    base = str(DATA / f"ds004712/sub-{sub}/ses-{ses}/dwi/sub-{sub}_ses-{ses}_dir-AP_dwi")
    img = nib.load(base + ".nii.gz")
    dwi = np.asarray(img.dataobj, np.float32)
    bval, bvec = np.loadtxt(base + ".bval"), np.loadtxt(base + ".bvec")
    res = {"scan": f"{sub} {ses} (all {len(bval)} volumes, as acquired)", "shape": list(dwi.shape)}
    with tempfile.TemporaryDirectory() as tmp:
        t0 = time.time()
        r = subprocess.run([sys.executable, str(HERE / "proto.py"), "--dwi", base, "--pe", "1", *PROTO_ARGS, "--real", tmp],
                           capture_output=True, text=True, cwd=str(HERE.parent.parent))
        if r.returncode:
            raise RuntimeError(r.stderr[-1500:])
        info = json.loads([l for l in r.stdout.splitlines() if l.startswith('{"data"')][-1])
        res["prototype seconds"] = {"estimate": info["estimate_s"], "final resample": info["final_resample_s"],
                                    "subprocess wall": round(time.time() - t0, 1)}
        pz = np.load(Path(tmp) / "ours_theta.npz")
        p_fit, p_split = pz["th_fit"], pz["th"]
        p_bvec = np.loadtxt(Path(tmp) / "ours_rotated_bvecs")
        p_corr = np.asarray(nib.load(Path(tmp) / "ours.nii.gz").dataobj, np.float32)
    ME.estimate(dwi[..., :8], bval[:8], bvec[:, :8], img.affine, 1, device="mps", options={"steps": 2})   # warm-up
    t0 = time.time()
    lib = ME.estimate(dwi, bval, bvec, img.affine, 1, device="mps")
    t1 = time.time()
    corr, valid = ME.apply_volumes(dwi, lib, device="mps")
    torch.mps.synchronize()
    t2 = time.time()
    res["library seconds"] = {"estimate": round(t1 - t0, 2), "apply": round(t2 - t1, 2), "inside": lib.seconds}
    from tractline.mask import median_otsu
    _, mask = median_otsu(dwi[..., bval < 50].mean(-1), median_radius=4, numpass=4)
    mask = np.asarray(mask, bool)
    vox = np.sqrt((img.affine[:3, :3] ** 2).sum(0))
    disp = map_difference(p_fit, np.zeros_like(p_fit), dwi.shape[:3], vox, 1, 10, mask)
    res["motion and eddy found (prototype), voxels at the edge (rms, p95)"] = disp
    res["library against prototype (exact Jacobian, Newton inverse, gauges as the prototype without a field): the maps, voxels at the edge (rms, p95)"] = map_difference(
        lib.theta_fit, p_fit, dwi.shape[:3], vox, 1, 10, mask)
    res["library against prototype: eddy-split parameters, max |difference| (mm, rad, vox)"] = [
        round(float(np.abs(lib.theta[:, :3] - p_split[:, :3]).max()), 4),
        round(float(np.abs(lib.theta[:, 3:6] - p_split[:, 3:6]).max()), 6),
        round(float(np.abs(lib.theta[:, 6:] - p_split[:, 6:]).max()), 4)]
    dw = bval >= 50
    res["library against prototype: rotated b-vectors, deg (median, max)"] = angles(lib.bvec, p_bvec, dw)
    m = mask & valid
    scale = float(np.quantile(p_corr[m], 0.99))
    d = np.abs(corr - p_corr)[m]
    res["library against prototype: corrected series in the mask, |difference| / 99th pct (mean, p99, max)"] = [
        round(float(d.mean() / scale), 6), round(float(np.quantile(d, 0.99) / scale), 6), round(float(d.max() / scale), 4)]
    res["leave-one-out residual in the brain (library, prototype)"] = {
        "library": loo_residual(corr, bval, bvec, mask), "prototype": loo_residual(p_corr, bval, bvec, mask)}
    res["qc (library): framewise displacement mm (mean, max)"] = lib.qc["framewise displacement, mm (mean, max)"]
    res["qc (library): volumes with residual z > 4"] = lib.qc["volumes with residual z > 4"]
    return res, sub


def field_check(args):
    import calibration_onharmony as CO
    from tractline import pipeline as P
    s = CO.load_checked(args.subject, args.session)
    timer = P.Timer()
    with P.exact_float32("mps"):
        c = P.correct(s, timer, "mps")
    P.release_memory("mps")
    h = c.field_hz
    res = {"scan": s.name, "shape": list(s.dwi.shape), "field seconds": timer.seconds,
           "field: displacement mm (max |.|)": round(float(np.abs(c.displacement_mm).max()), 2)}
    ME.estimate(s.dwi[..., :8], s.bval[:8], s.bvec[:, :8], s.affine, s.pe_axis, device="mps", options={"steps": 2})
    # phase 0's order: the field applied, then the step on the corrected series (two interpolations)
    t0 = time.time()
    a = ME.estimate(c.dwi, s.bval, s.bvec, s.affine, s.pe_axis, device="mps")
    a_corr, a_valid = ME.apply_volumes(c.dwi, a, device="mps")
    torch.mps.synchronize()
    ta = time.time() - t0
    # composed: the step given the field, one resample of the data as acquired
    t0 = time.time()
    b = ME.estimate(s.dwi, s.bval, s.bvec, s.affine, s.pe_axis, field_hz=h, pe_sign=s.pe_sign, readout_s=s.readout_s,
                    device="mps")
    b_corr, b_valid = ME.apply_volumes(s.dwi, b, field_hz=h, device="mps")
    torch.mps.synchronize()
    tb = time.time() - t0
    res["seconds (step + apply)"] = {"field first, then the step": round(ta, 2), "composed": round(tb, 2)}
    from tractline.mask import median_otsu
    _, mask = median_otsu(c.dwi[..., s.bval < 50].mean(-1), median_radius=4, numpass=4)
    mask = np.asarray(mask, bool) & a_valid & b_valid
    res["leave-one-out residual in the brain"] = {
        "field only (no motion or eddy step)": loo_residual(c.dwi, s.bval, s.bvec, mask),
        "field first, then the step (two interpolations)": loo_residual(a_corr, s.bval, s.bvec, mask),
        "composed (one interpolation)": loo_residual(b_corr, s.bval, s.bvec, mask)}
    ref = int(np.flatnonzero(s.bval < 50)[0])
    a_rel = a.theta_fit.copy()
    a_rel[:, :6] -= a.theta_fit[ref, :6]                     # into the first b0's frame (first order: small poses)
    res["the two maps' difference, voxels at the edge (rms, p95), in the first b0's frame"] = map_difference(
        a_rel, b.theta_fit, s.dwi.shape[:3], s.vox, s.pe_axis, 10, mask)
    res["reference frames"] = {"field first": a.reference, "composed": b.reference}
    res["framewise displacement mm (mean, max)"] = {"field first": a.qc["framewise displacement, mm (mean, max)"],
                                                    "composed": b.qc["framewise displacement, mm (mean, max)"]}
    res["note"] = ("field first: the step's location reference is the b0s' mean pose; composed: the first b0's (the "
                   "field's frame); the maps are compared after moving the first into the second to first order, and "
                   "the residuals do not depend on the frame")
    return res, args.subject


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["proto", "field"])
    ap.add_argument("--scan", default="16974/OXF1PRI001")
    ap.add_argument("--subject", default="16974")
    ap.add_argument("--session", default="OXF1PRI001")
    args = ap.parse_args()
    res, sub = proto_check(args) if args.what == "proto" else field_check(args)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"library_check_{args.what}_{sub}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
