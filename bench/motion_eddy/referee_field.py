"""The field composed into the single resample, against phase 0's order (the field applied first, then the step: two
interpolations), refereed by FSL eddy fed the same field (bench/motion_eddy/phase0.py's inputs and eddy run: eddy_cpu
--flm=quadratic --slm=none --field=<tractline's field>, no --repol). On one prepared phase 0 scan folder:

  - images: each corrected volume against eddy's, the RMS difference over the brain (eroded 2) relative to the brain's
    mean signal, per volume, for both of our orders; and the share of volumes where the composed order is the nearer;
  - b-vectors: our rotated b-vectors' angles to eddy's (eddy_rotated_bvecs);
  - model fit, per set (as acquired, eddy, ours composed, ours two-step): each group's leave-one-out residual (the
    library's predictor) and a log-linear tensor fit's residual over the brain.
Ours: tractline.motion_eddy with its defaults but outliers "off" (eddy ran without --repol). Frames: eddy and the
composed order hold the first b0 fixed (the field's frame); the two-step order holds the b0s' mean pose, which differs
by the b0s' own spread (reported).

    PYTHONPATH=src:bench python bench/motion_eddy/referee_field.py $TRACTOGRAPHY_DATA/derived/phase0/16974_OXF1PRI001
Writes results/motion_eddy/referee_field_<folder>.json.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import nibabel as nib
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from library_check import loo_residual                                            # noqa: E402
from tractline import motion_eddy as ME, susceptibility as S                        # noqa: E402
from scipy.ndimage import binary_erosion                                           # noqa: E402


def tensor_residual(corr, bval, bvec, m):
    """RMS over the brain of the DW signal minus a log-linear single-tensor fit (all volumes)."""
    isb0 = bval < 50
    g = np.zeros((len(bval), 3))
    g[~isb0] = (bvec[:, ~isb0] / np.linalg.norm(bvec[:, ~isb0], axis=0)).T
    b = bval / 1000.0
    Bm = np.stack([np.ones(len(b)), -b * g[:, 0] ** 2, -b * g[:, 1] ** 2, -b * g[:, 2] ** 2,
                   -2 * b * g[:, 0] * g[:, 1], -2 * b * g[:, 0] * g[:, 2], -2 * b * g[:, 1] * g[:, 2]], 1)
    Sv = corr[m].astype(np.float64)
    coef = np.log(np.clip(Sv, 1.0, None)) @ np.linalg.pinv(Bm).T
    return round(float(np.sqrt(((Sv - np.exp(coef @ Bm.T))[:, ~isb0] ** 2).mean())), 3)


def main():
    d = Path(sys.argv[1])
    img = nib.load(d / "imain.nii.gz")
    dwi = np.asarray(img.dataobj, np.float32)
    h = np.asarray(nib.load(d / "field_hz.nii.gz").dataobj, np.float64)
    bval, bvec = np.loadtxt(d / "bvals"), np.loadtxt(d / "bvecs")
    acqp = np.loadtxt(d / "acqp.txt")
    pe_axis = int(np.argmax(np.abs(acqp[:3])))
    pe_sign, ro = float(np.sign(acqp[pe_axis])), float(acqp[3])
    m = binary_erosion(np.asarray(nib.load(d / "mask.nii.gz").dataobj) > 0, iterations=2)
    eddy = np.asarray(nib.load(d / "eddy.nii.gz").dataobj, np.float32)
    ebv = np.loadtxt(d / "eddy.eddy_rotated_bvecs")
    opts = {"outliers": "off"}
    res = {"folder": d.name, "shape": list(dwi.shape), "eddy": "eddy_cpu --flm=quadratic --slm=none --field (no --repol)"}
    ME.estimate(dwi[..., :8], bval[:8], bvec[:, :8], img.affine, pe_axis, device="mps", options={"steps": 2})
    t0 = time.time()
    a = ME.estimate(dwi, bval, bvec, img.affine, pe_axis, field_hz=h, pe_sign=pe_sign, readout_s=ro, device="mps",
                    options=opts)
    comp, _ = ME.apply_volumes(dwi, a, field_hz=h, device="mps")
    t1 = time.time()
    fdwi = S.apply(dwi, h, pe_axis, pe_sign, ro)
    b = ME.estimate(fdwi, bval, bvec, img.affine, pe_axis, device="mps", options=opts)
    two, _ = ME.apply_volumes(fdwi, b, device="mps")
    torch.mps.synchronize()
    res["seconds"] = {"composed": round(t1 - t0, 1), "two-step": round(time.time() - t1, 1)}
    scale = float(eddy[m].mean())

    def to_eddy(x):
        return np.sqrt(((x - eddy)[m] ** 2).mean(0)) / scale                            # per volume
    rc, rt = to_eddy(comp), to_eddy(two)
    res["image difference to eddy, RMS / brain mean (median over volumes)"] = {
        "composed": round(float(np.median(rc)), 4), "two-step": round(float(np.median(rt)), 4),
        "as acquired": round(float(np.median(to_eddy(dwi))), 4)}
    res["volumes where the composed order is nearer eddy"] = f"{int((rc < rt).sum())} of {len(rc)}"
    dw = bval >= 50

    def ang(x):
        c = np.abs((x[:, dw] * ebv[:, dw]).sum(0) / np.linalg.norm(x[:, dw], axis=0) / np.linalg.norm(ebv[:, dw], axis=0))
        v = np.degrees(np.arccos(np.clip(c, 0, 1)))
        return [round(float(np.median(v)), 3), round(float(v.max()), 3)]
    res["rotated b-vectors to eddy's, deg (median, max)"] = {"composed": ang(a.bvec), "two-step": ang(b.bvec),
                                                            "as given": ang(bvec)}
    sets = {"as acquired": dwi, "eddy": eddy, "ours composed": comp, "ours two-step": two}
    res["model fit"] = {k: {"leave-one-out residual": loo_residual(v, bval, bvec, m),
                            "tensor fit residual (DW)": tensor_residual(v, bval, bvec, m)} for k, v in sets.items()}
    b0 = bval < 50
    res["the b0s' mean pose against the first b0 (composed run), mm and deg"] = [
        np.round(a.theta[b0, :3].mean(0), 3).tolist(), np.round(np.degrees(a.theta[b0, 3:6].mean(0)), 3).tolist()]
    out = HERE.parent / "results" / "motion_eddy" / f"referee_field_{d.name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
