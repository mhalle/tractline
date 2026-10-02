"""How stable is the field estimate, and where does it diverge? Two runs on PAT16's b0s - as they are,
and plus noise of SD 0.01 (signals in the hundreds; the scan's own noise ~3 orders larger) - level by
level: the optimizer's iterations against its cap, the cost, and the two fields' displacement difference
inside the brain (the cohort's mask), at its edge, and outside.

    python bench/susc_convergence.py [--device mps] [--tag NAME] [estimate options as JSON]

Writes results/susc_convergence[_tag].json.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, nibabel as nib, torch
from scipy.ndimage import binary_erosion
from tractline import susceptibility as S
from _ds001226 import load, ROOT

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--device", default="mps"); ap.add_argument("--tag", default="")
ap.add_argument("--options", default="{}", help='estimate keyword arguments, e.g. {"iter_scale": 10}')
ap.add_argument("--sub", default="PAT16")
args = ap.parse_args()
opts = json.loads(args.options)
s = load(args.sub)
brain = np.load(ROOT / f"derived/{args.sub}/cohort_fields.npz")["brain"]
deep = binary_erosion(brain, iterations=3); edge = brain & ~deep
mm = lambda h: S.displacement_mm(h, s.readout_s, s.pe_sign, s.vox[s.pe_axis])
q = lambda d, m: [round(float(np.median(d[m])), 3), round(float(np.quantile(d[m], 0.99)), 3), round(float(d[m].max()), 3)]
runs = {}
for name, b0s in (("as is", s.b0s), ("plus noise 0.01", s.b0s + np.random.default_rng(0).normal(0, 0.01, s.b0s.shape))):
    t0 = time.time()
    h, motion, log = S.estimate(b0s, s.vox, s.pe_vectors, s.readout_s, device=args.device, diagnostics=True, **opts)
    runs[name] = (h, motion, log, round(time.time() - t0, 1))
(ha, ma, la, ta), (hb, mb, lb, tb) = runs["as is"], runs["plus noise 0.01"]
res = {"subject": args.sub, "device": args.device, "options": opts, "seconds": [ta, tb], "levels": [],
       "columns": "|displacement difference| mm: median / 99th / max"}
for a, b in zip(la, lb):
    d = np.abs(mm(a["field"]) - mm(b["field"]))
    res["levels"].append({"level": a["level"], "knots": a["knots"], "motion": a["motion"],
                          "iterations": [a["n_iter"], b["n_iter"]], "max_iter": a["max_iter"], "evals": [a["func_evals"], b["func_evals"]],
                          "ssd_after": [a["ssd_after"], b["ssd_after"]],
                          "diff_deep": q(d, deep), "diff_edge": q(d, edge), "diff_outside": q(d, ~brain),
                          "motion_diff_max": round(float(np.abs(a["motion_now"] - b["motion_now"]).max()), 4)})
    L = res["levels"][-1]
    print(f"level {L['level']}: iter {L['iterations']}/{L['max_iter']}, ssd {L['ssd_after'][0]:.5g}/{L['ssd_after'][1]:.5g}, "
          f"diff deep {L['diff_deep']}, edge {L['diff_edge']}, outside {L['diff_outside']}, motion {L['motion_diff_max']}", flush=True)
topup = mm(np.asarray(nib.load(ROOT / f"derived/{args.sub}/topup/field_hz.nii.gz").dataobj, float)) if (ROOT / f"derived/{args.sub}/topup/field_hz.nii.gz").exists() else None
if topup is not None:
    res["vs_topup_brain"] = {n: q(np.abs(mm(r[0]) - topup), brain) for n, r in runs.items()}
print(json.dumps({k: v for k, v in res.items() if k != "levels"}, indent=1))
(HERE / f"results/susc_convergence{('_' + args.tag) if args.tag else ''}.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(ROOT / f"derived/{args.sub}/susc_convergence{('_' + args.tag) if args.tag else ''}.npz", a=ha, b=hb)
