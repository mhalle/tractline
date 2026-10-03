"""Can the relative phase-encoding polarity of two series be read from the images, when the series are
acquired minutes apart? (readout_polarity.py's control was two halves of one series, seconds apart.)

OpenNeuro ds005123 (CC0, Siemens Prisma; fetch_ds005123.py): per subject the diffusion series' two leading
b0s (j-), and the spin-echo field maps acquired for it, dir-AP (j-, ~7 min after the diffusion series)
and dir-PA (j) - one grid, one echo time and readout time. The field estimate (the pipeline's defaults,
"mps"), each series first scaled to the diffusion b0s' mean in the brain (topup's --scale; the field maps
are ~10-15 % brighter), on:
  - pair: the diffusion b0s and the PA field map - opposite polarities;
  - control, separate series: the diffusion b0s and the AP field map labeled opposite - one polarity truly,
    minutes apart;
  - control, one series: the diffusion series' two b0s labeled opposite;
  - field maps: AP against PA, the pair they were acquired to be - the reference field.
The diffusion series' displacement (mm, along j) in the brain (median_otsu of its mean b0): median and
99th percentile of |d|, and the pair's against the field maps'.

    uv run bench/polarity_separate_series.py

    [--subjects N] [--scale on|off]

Writes results/polarity_separate_series_scale<on|off>.json.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, nibabel as nib
from tractline import susceptibility as S, pipeline as P
from tractline.data import DATA
from tractline.mask import median_otsu

HERE = Path(__file__).resolve().parent
D = DATA / "ds005123"
VEC = {"j": (0, 1, 0), "j-": (0, -1, 0), "i": (1, 0, 0), "i-": (-1, 0, 0)}


def q(x):
    return [round(float(np.median(x)), 3), round(float(np.quantile(x, 0.99)), 3)]


def series(path):
    img = nib.load(path)
    return np.asarray(img.dataobj, np.float64), img


if __name__ == "__main__":
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--subjects", type=int, default=0)
    ap_.add_argument("--scale", choices=["on", "off"], default="on"); args = ap_.parse_args()
    res = {"dataset": "OpenNeuro ds005123 v1.1.3 (CC0)", "scale": args.scale, "subjects": {}}
    subs = sorted(p.name for p in D.glob("sub-*"))
    for sub in subs[:args.subjects] if args.subjects else subs:
        dwi, img = series(D / sub / f"dwi/{sub}_desc-leadingb0_dwi.nii.gz")
        ap, _ = series(D / sub / f"fmap/{sub}_acq-dwi_dir-AP_epi.nii.gz")
        pa, _ = series(D / sub / f"fmap/{sub}_acq-dwi_dir-PA_epi.nii.gz")
        side = {k: json.loads((D / sub / f).read_text()) for k, f in (("dwi", f"dwi/{sub}_dwi.json"),
                ("ap", f"fmap/{sub}_acq-dwi_dir-AP_epi.json"), ("pa", f"fmap/{sub}_acq-dwi_dir-PA_epi.json"))}
        pe = {k: np.array(VEC[v["PhaseEncodingDirection"]], float) for k, v in side.items()}
        assert np.allclose(pe["dwi"], pe["ap"]) and np.allclose(pe["dwi"], -pe["pa"]), sub
        ro = side["dwi"]["TotalReadoutTime"]
        assert all(abs(v["TotalReadoutTime"] - ro) < 1e-6 for v in side.values()), sub
        _, brain = median_otsu(dwi.mean(-1).astype(np.float32), median_radius=4, numpass=4, device="mps")
        ref = dwi[brain].mean()
        if args.scale == "on":
            ap, pa = ap * ref / ap[brain].mean(), pa * ref / pa[brain].mean()  # one scale for all (topup --scale)
        vox = np.array(img.header.get_zooms()[:3], float)
        axis, sign = int(np.argmax(np.abs(pe["dwi"]))), float(pe["dwi"][np.argmax(np.abs(pe["dwi"]))])
        n = dwi.shape[-1]
        runs = {
            "pair (dwi b0s + PA)": (np.concatenate([dwi, pa], -1), [pe["dwi"]] * n + [pe["pa"]] * pa.shape[-1]),
            "control, separate series (dwi b0s + AP, labeled opposite)": (np.concatenate([dwi, ap], -1), [pe["dwi"]] * n + [-pe["ap"]] * ap.shape[-1]),
            "control, one series (dwi b0 0 + 1, labeled opposite)": (dwi, [pe["dwi"]] * (n // 2) + [-pe["dwi"]] * (n - n // 2)),
            "field maps (AP + PA)": (np.concatenate([ap, pa], -1), [pe["ap"]] * ap.shape[-1] + [pe["pa"]] * pa.shape[-1]),
        }
        rec, disp = {"acquisition_time": {k: v.get("AcquisitionTime") for k, v in side.items()}, "brain_voxels": int(brain.sum())}, {}
        for name, (b0s, pev) in runs.items():
            t0 = time.time()
            h, _, _ = S.estimate(b0s, vox, np.array(pev), ro, device="mps")
            disp[name] = S.displacement_mm(h, ro, sign, vox[axis])
            rec[name] = {"displacement_abs_mm_median_99th": q(np.abs(disp[name][brain])), "seconds": round(time.time() - t0, 1)}
            P.release_memory("mps")
        rec["pair vs field maps, |difference| mm median_99th"] = q(np.abs(disp["pair (dwi b0s + PA)"] - disp["field maps (AP + PA)"])[brain])
        res["subjects"][sub] = rec
        print(sub, json.dumps({k: (v["displacement_abs_mm_median_99th"] if isinstance(v, dict) and "displacement_abs_mm_median_99th" in v else v)
                               for k, v in rec.items() if k not in ("acquisition_time", "brain_voxels")}), flush=True)
    rows = list(res["subjects"].values())
    p99 = lambda name: [r[name]["displacement_abs_mm_median_99th"][1] for r in rows]
    pair, sep, one = p99("pair (dwi b0s + PA)"), p99("control, separate series (dwi b0s + AP, labeled opposite)"), p99("control, one series (dwi b0 0 + 1, labeled opposite)")
    ratio = [a / b for a, b in zip(pair, sep)]
    res["summary"] = {"pair_99th_mm_min_max": [min(pair), max(pair)], "separate_control_99th_mm_min_max": [min(sep), max(sep)],
                      "one_series_control_99th_mm_min_max": [min(one), max(one)],
                      "pair_over_separate_control_min_median_max": [round(min(ratio), 2), round(float(np.median(ratio)), 2), round(max(ratio), 2)]}
    print(json.dumps(res["summary"]))
    (HERE / f"results/polarity_separate_series_scale{args.scale}.json").write_text(json.dumps(res, indent=1))
