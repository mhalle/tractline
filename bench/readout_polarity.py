"""What the correction needs to know about the acquisition: does the readout time's value matter, does the
phase encoding's absolute polarity, and can the relative polarity of two series be read from the images?
(For a DICOM front end: polarity and readout time are vendor-private and sometimes absent.)

Per ds001226 patient, the field estimate (susceptibility.estimate, the pipeline's defaults, on "mps") on:
  - base: the AP b0s and the PA b0s as acquired;
  - readout x0.8 and x1.25: the same, the readout time (both series) scaled - topup's model shifts by
    sign x readout x field, so the data term cannot tell; the smoothing weight, on the field in Hz, can;
  - flipped: every phase-encoding sign flipped - the model's displacements should not change at all;
  - control: the AP b0s alone, split in two halves labeled as opposite polarities - truly one polarity,
    so a field found there is what the fit makes of noise and motion.
Compared in the brain (median_otsu of the mean AP b0): the DWI's displacement (mm), each variant against
base - median / 99th percentile of |difference| - and its own 99th percentile; how much of the first
level's cost the fit removes from no field and no motion (ssd_after over ssd_before at that level's own
smoothing; levels differ in smoothing, so costs compare only within one), and every level's costs.

    uv run bench/readout_polarity.py --sub PAT16

Writes results/readout_polarity/<sub>.json.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
from tractline import susceptibility as S, pipeline as P
from tractline.mask import median_otsu
from _ds001226 import load

HERE = Path(__file__).resolve().parent


def q(x):
    return [round(float(np.median(x)), 3), round(float(np.quantile(x, 0.99)), 3)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--sub", required=True); ap.add_argument("--device", default="mps")
    args = ap.parse_args()
    s = load(args.sub)
    ap_b0 = [i for i, v in enumerate(s.pe_vectors) if np.allclose(v, s.pe_vectors[0])]
    _, brain = median_otsu(s.b0s[..., ap_b0].mean(-1).astype(np.float32), median_radius=4, numpass=4, device=args.device)
    vox_pe = float(s.vox[s.pe_axis])

    half = len(ap_b0) // 2
    ctrl_pe = np.array([s.pe_vectors[0]] * half + [-s.pe_vectors[0]] * (len(ap_b0) - half))
    runs = {
        "base": (s.b0s, s.pe_vectors, s.readout_s, s.pe_sign),
        "readout x0.8": (s.b0s, s.pe_vectors, 0.8 * s.readout_s, s.pe_sign),
        "readout x1.25": (s.b0s, s.pe_vectors, 1.25 * s.readout_s, s.pe_sign),
        "flipped": (s.b0s, -s.pe_vectors, s.readout_s, -s.pe_sign),
        "control (AP halves as opposite)": (s.b0s[..., ap_b0], ctrl_pe, s.readout_s, s.pe_sign),
    }
    res = {"subject": args.sub, "b0s": {"AP": len(ap_b0), "other": s.b0s.shape[-1] - len(ap_b0)}, "readout_s": s.readout_s,
           "brain_voxels": int(brain.sum()), "runs": {}}
    disp = {}
    for name, (b0s, pe, ro, sign) in runs.items():
        t0 = time.time()
        h, _, log = S.estimate(b0s, s.vox, pe, ro, device=args.device)
        d = S.displacement_mm(h, ro, sign, vox_pe)
        disp[name] = d
        res["runs"][name] = {"seconds": round(time.time() - t0, 1),
                             "displacement_abs_mm_median_99th": q(np.abs(d[brain])),
                             "level1_cost_after_over_before": round(float(log[0]["ssd_after"] / log[0]["ssd_before"]), 4),
                             "levels_ssd_before_after": [[float(f"{l['ssd_before']:.5g}"), float(f"{l['ssd_after']:.5g}")] for l in log]}
        if name != "base":
            res["runs"][name]["vs_base_abs_mm_median_99th"] = q(np.abs(d - disp["base"])[brain])
        print(args.sub, name, json.dumps(res["runs"][name]), flush=True)
        P.release_memory(args.device)
    out = HERE / "results/readout_polarity"; out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.sub}.json").write_text(json.dumps(res, indent=1))
