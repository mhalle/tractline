"""Do real eddy currents turn with the phase encoding, as motion_eddy's mixed model says? ON-Harmony's 16974 (ds004712)
acquires its 105-volume series AP and, in the same session, a short PA series of 3 b0s and 3 b = 2000 volumes, so the PA
volumes' eddy fields can be foretold: fit the AP series alone, regress each eddy coefficient on the gradient, and read
what the 3 PA gradients should have. Then fit the two series together, with each volume's own polarity. In motion_eddy's
model a coefficient means the same field in both polarities, so a fitted PA coefficient should equal its forecast, and if
the field did not turn (a displacement fixed in the scanner's sense), it would equal minus its forecast. The coefficients
are free per volume, so the fit does not assume the answer. Also: each PA volume's fit to the prediction, with its true
polarity and with the polarity ignored.

    PYTHONPATH=src:bench python bench/reversed_eddy_sign.py [session ...]

Writes results/reversed_eddy_sign.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
from tractline import bids, motion_eddy as ME, pipeline as P, series as SR
from tractline.data import DATA

OUT = Path(__file__).parent / "results/reversed_eddy_sign.json"
SESSIONS = [f"ses-OXF1PRI00{i}" for i in range(1, 7)] + ["ses-OXF2PRI001"]


def run(ses):
    ap = DATA / f"ds004712/sub-16974/{ses}/dwi/sub-16974_{ses}_dir-AP_dwi.nii.gz"
    s = bids.load(ap, keep_reversed=True)
    h, _ = P._estimate_field(s, P.Timer(), "mps")
    r0 = ME.estimate(s.dwi, s.bval, s.bvec, s.affine, s.pe_axis, field_hz=h, pe_sign=s.pe_sign, readout_s=s.readout_s,
                     device="mps", slice_axis=P.slice_axis_of(s))
    bvec = np.asarray(s.bvec, float)
    dw = np.asarray(s.bval) > 1500                                          # the b = 2000 shell, as the PA volumes'
    X = np.c_[np.ones(int(dw.sum())), bvec.T[dw]]
    E = r0.theta_fit[dw][:, 7:]
    A = np.linalg.lstsq(X, E, rcond=None)[0]
    m = SR.merge(s)
    G = s.dwi.shape[-1]
    pa = np.flatnonzero(m.bval[G:] > 50) + G
    forecast = np.c_[np.ones(len(pa)), m.bvec.T[pa]] @ A
    r = ME.estimate(m.dwi, m.bval, m.bvec, m.affine, m.pe_axis, field_hz=h, pe_sign=m.pe_signs, readout_s=m.readouts_s,
                    device="mps", slice_axis=P.slice_axis_of(s))
    wrong = ME.estimate(m.dwi, m.bval, m.bvec, m.affine, m.pe_axis, field_hz=h, pe_sign=s.pe_sign, readout_s=s.readout_s,
                        device="mps", slice_axis=P.slice_axis_of(s))
    fitted = r.theta_fit[pa][:, 7:]
    rms = lambda x: round(float(np.sqrt(np.mean(x ** 2))), 4)                # noqa: E731
    res = lambda q: np.asarray(q.qc["residual vs prediction, rms per volume"], float)  # noqa: E731
    shell = np.flatnonzero(np.asarray(s.bval) > 1500)
    return {"series": ap.name, "AP volumes in the shell": int(dw.sum()), "PA volumes": len(pa),
            "scale": round(float(m.reversed_scale), 4),
            "share of the AP coefficients' variance the gradient explains": round(
                float(1 - ((E - X @ A) ** 2).sum() / (E ** 2).sum()), 3),
            "rms of the forecast (voxels)": rms(forecast),
            "fitted - forecast, rms (the field turns)": rms(fitted - forecast),
            "fitted + forecast, rms (it does not)": rms(fitted + forecast),
            "correlation, fitted and forecast": round(float(np.corrcoef(fitted.ravel(), forecast.ravel())[0, 1]), 3),
            "residual of the PA volumes, true polarity": np.round(res(r)[pa], 2).tolist(),
            "residual of the PA volumes, polarity ignored": np.round(res(wrong)[pa], 2).tolist(),
            "residual of the AP volumes of the shell, median": round(float(np.median(res(r)[shell])), 2),
            "left out": {str(k): v for k, v in r.excluded.items()}}


if __name__ == "__main__":
    out = {}
    for ses in sys.argv[1:] or SESSIONS:
        out[ses] = run(ses)
        print(ses, json.dumps(out[ses]), flush=True)
    A = [v["fitted - forecast, rms (the field turns)"] for v in out.values()]
    B = [v["fitted + forecast, rms (it does not)"] for v in out.values()]
    C = [v["correlation, fitted and forecast"] for v in out.values()]
    out["summary"] = {"sessions": len(A), "median fitted - forecast": float(np.median(A)),
                      "median fitted + forecast": float(np.median(B)), "correlations": C}
    print(out["summary"])
    OUT.write_text(json.dumps(out, indent=1))
