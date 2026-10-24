"""DIPY's side of csd_check.py, in a process of its own (DIPY's OpenMP and torch's clash in one process). Imports no torch
and nothing of tractline.

    python -I bench/_csd_dipy.py fit   IN.npz OUT.npz     # DIPY's response, CSD coefficients and peaks (step 6's settings)
    python -I bench/_csd_dipy.py peaks IN.npz OUT.npz     # DIPY's peaks of the coefficients in IN.npz ("coeffs")

IN.npz for fit: dwi (X, Y, Z, G) float32 (b0s and one shell), bval, bvec (3, G) FSL, mask. Settings as
_benchmark_dipy.py csd: auto_response_ssst (radius 10, FA 0.7), sh_order_max 8 (6 below 45 directions), lambda 1, tau
0.1, peaks on default_sphere (3 peaks, relative threshold 0.1, separation 25 degrees). OUT: response evals and S0,
coeffs (X, Y, Z, K) float32, peak_dirs (X, Y, Z, 3, 3), peak_values (X, Y, Z, 3), small_sphere's vertices, seconds.
"""
import json, sys, time, warnings
import numpy as np

PEAKS = dict(relative_peak_threshold=0.1, min_separation_angle=25, npeaks=3)


def main():
    what, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    warnings.simplefilter("ignore")
    from dipy.core.gradients import gradient_table
    from dipy.data import default_sphere, small_sphere
    from dipy.direction import peaks_from_model
    from dipy.direction.peaks import peak_directions
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
    from dipy.reconst.shm import sh_to_sf
    z = np.load(src)
    if what == "fit":
        data, bval, bvec, mask = z["dwi"], z["bval"], z["bvec"], z["mask"].astype(bool)
        gtab = gradient_table(bval, bvecs=bvec.T)
        ndir = int((bval > 50).sum())
        order = 8 if ndir >= 45 else 6
        t0 = time.time()
        response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
        t1 = time.time()
        model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=order)
        pam = peaks_from_model(model, data, default_sphere, PEAKS["relative_peak_threshold"], PEAKS["min_separation_angle"],
                               mask=mask, return_sh=True, npeaks=PEAKS["npeaks"], sh_order_max=order, parallel=True)
        t2 = time.time()
        np.savez(out, evals=np.asarray(response[0], float), S0=float(response[1]), order=order,
                 coeffs=pam.shm_coeff.astype(np.float32), peak_dirs=pam.peak_dirs.astype(np.float32),
                 peak_values=pam.peak_values.astype(np.float32), small_sphere=small_sphere.vertices,
                 seconds=np.array(json.dumps({"response": round(t1 - t0, 2), "fit and peaks (parallel)": round(t2 - t1, 2)})))
    else:
        coeffs, mask = z["coeffs"].astype(np.float64), z["mask"].astype(bool)
        order = int(z["order"])
        dirs = np.zeros(coeffs.shape[:3] + (3, 3), np.float32)
        vals = np.zeros(coeffs.shape[:3] + (3,), np.float32)
        sf = sh_to_sf(coeffs[mask], default_sphere, sh_order_max=order)
        for n, (i, j, k) in enumerate(zip(*np.nonzero(mask))):
            d, v, _ = peak_directions(sf[n], default_sphere, relative_peak_threshold=PEAKS["relative_peak_threshold"],
                                      min_separation_angle=PEAKS["min_separation_angle"])
            d, v = d[:3], v[:3]
            dirs[i, j, k, :len(d)] = d
            vals[i, j, k, :len(v)] = v
        np.savez(out, peak_dirs=dirs, peak_values=vals)


if __name__ == "__main__":
    main()
