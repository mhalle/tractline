"""modal_gpu_pipeline.py's results against the M2's GPU path, PAT16: the field (displacement difference
in the brain, deep and at the edge: median / 99th / max mm), the fiber and label counts, and the
tract mix (Pearson r of the 42 tracts' counts; the Other fraction).

    python bench/gpu_pipeline_compare.py [--gpus a10,l40s,a10_tf32off]

The M2's field is the cohort's (derived/PAT16/cohort_fields.npz, the default estimate); its counts and
tract mix come from one run of the pipeline here (mps). Writes results/gpu_pipeline_compare.json.
"""
import argparse, base64, json, zlib
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from tractline import pipeline as P, susceptibility as S
from _ds001226 import load, ROOT

HERE = Path(__file__).resolve().parent
if __name__ == "__main__":                                                 # the CPU path spawns workers, which re-import this script
    ap = argparse.ArgumentParser(); ap.add_argument("--gpus", default="a10,l40s"); ap.add_argument("--m2-cpu", action="store_true")
    args = ap.parse_args()
    s = load("PAT16")
    F = np.load(ROOT / "derived/PAT16/cohort_fields.npz")
    brain = F["brain"]; deep = binary_erosion(brain, iterations=3); edge = brain & ~deep
    q = lambda d, m: [round(float(np.median(d[m])), 3), round(float(np.quantile(d[m], 0.99)), 3), round(float(d[m].max()), 3)]

    timer = P.Timer(echo="m2")
    corr, tg, labels = P.run(s, None, timer, device="mps")
    m2 = {"fibers": len(tg.fibers), "labeled": int(labels.keep.sum()), "scan_to_labels_s": timer.total(*P.pipeline_stages()),
          "tract_counts": np.bincount(labels.tract, minlength=43)}
    res = {"m2": {k: v for k, v in m2.items() if k != "tract_counts"}, "gpus": {}}
    if args.m2_cpu:                                                            # the M2's own CPU path, against its GPU path
        timer = P.Timer(echo="m2 cpu")
        corr_c, tg_c, lab_c = P.run(s, None, timer, device="cpu")
        tc = np.bincount(lab_c.tract, minlength=43); dc = np.abs(corr_c.displacement_mm - F["d_ours"])
        res["m2_cpu"] = {"fibers": len(tg_c.fibers), "labeled": int(lab_c.keep.sum()), "scan_to_labels_s": timer.total(*P.pipeline_stages()),
                         "field_vs_m2_mm": {"deep": q(dc, deep), "edge": q(dc, edge)},
                         "tract_mix_r_vs_m2": round(float(np.corrcoef(tc[:42], m2["tract_counts"][:42])[0, 1]), 5),
                         "other_fraction": [round(float(tc[42] / tc.sum()), 4), round(float(m2["tract_counts"][42] / m2["tract_counts"].sum()), 4)]}
    for g in args.gpus.split(","):
        r = json.loads((HERE / f"results/modal_gpu_pipeline_{g}.json").read_text())             # g: <gpu>[_tf32off]
        shape = tuple(r["field_shape"])
        h = np.frombuffer(zlib.decompress(base64.b64decode((ROOT / f"derived/PAT16/gpu_field_{g}.b64").read_text())), np.float32).reshape(shape)
        d = np.abs(S.displacement_mm(h, s.readout_s, s.pe_sign, s.vox[s.pe_axis]) - F["d_ours"])
        last = r["runs"][-1]; tc = np.asarray(last["tract_counts"])
        res["gpus"][g] = {"gpu": r["gpu"], "tf32": r.get("tf32"), "scan_to_labels_s": [x["scan_to_labels_s"] for x in r["runs"]],
                          "seconds_steady": last["seconds"], "fibers": last["fibers"], "labeled": last["labeled"],
                          "runs_identical_counts": len({(x["fibers"], x["labeled"], tuple(x["tract_counts"])) for x in r["runs"]}) == 1,
                          "field_vs_m2_mm": {"deep": q(d, deep), "edge": q(d, edge)},
                          "tract_mix_r_vs_m2": round(float(np.corrcoef(tc[:42], m2["tract_counts"][:42])[0, 1]), 5),
                          "other_fraction": [round(float(tc[42] / tc.sum()), 4), round(float(m2["tract_counts"][42] / m2["tract_counts"].sum()), 4)]}
    print(json.dumps(res, indent=1))
    (HERE / "results/gpu_pipeline_compare.json").write_text(json.dumps(res, indent=1))
