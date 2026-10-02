"""results/susc_held_out/*.json as one table: per configuration, over the patients, the median of each
measure, and per patient the held-out residual (mean of the two directions) against the first
configuration's.

    python bench/susc_held_out_summary.py [--tag T]

Writes results/susc_held_out/summary[_tag].json.
"""
import argparse, json
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parent / "results/susc_held_out"
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default=""); args = ap.parse_args()
runs = [json.loads(p.read_text()) for p in sorted(D.glob(f"PAT??{('_' + args.tag) if args.tag else ''}.json"))]
same = lambda x: (x.get("same_polarity_AP0_AP1") or x["noise_floor_AP0_AP1"])["brain"]
names = list(runs[0]["configs"])
held = lambda r, part="brain": float(np.mean([v[part] for v in r["held_out"].values()]))
insample = lambda r: float(np.mean([v["in_sample_brain"] for v in r["held_out"].values()]))
rows = {}


def _ok(f, r):
    try:
        f(r); return True
    except KeyError:
        return False


for n in names:
    rs = [x["configs"][n] for x in runs if n in x["configs"]]
    med = lambda f: round(float(np.median([f(r) for r in rs])), 4) if all(_ok(f, r) for r in rs) else None
    rows[n] = {"patients": len(rs), "held_out_brain": med(held), "held_out_deep": med(lambda r: held(r, "deep")),
               "held_out_edge": med(lambda r: held(r, "edge")), "in_sample_brain": med(insample),
               "split_half_deep_99": med(lambda r: r["split_half_deep"][1]), "split_half_edge_99": med(lambda r: r["split_half_edge"][1]),
               "split_half_deep_max": med(lambda r: r["split_half_deep"][2]), "drift_deep_99": med(lambda r: r["drift_deep"][1]),
               "drift_edge_99": med(lambda r: r["drift_edge"][1]), "seconds_all": med(lambda r: r["seconds"]["all"]),
               "folds_brain_A_total": int(sum(r["folds_brain_A"] for r in rs))}
base = names[0]
per = {x["subject"]: {"same_polarity": same(x), "uncorrected": round(float(np.mean([v["uncorrected_brain"] for v in x["configs"][base]["held_out"].values()])), 4),
                      **{n: round(held(x["configs"][n]), 4) for n in names if n in x["configs"]}} for x in runs}
wins = {n: sum(per[p][n] < per[p][base] for p in per if n in per[p]) for n in names[1:]}
out = {"medians": rows, "held_out_per_patient": per, f"patients_better_than_{base}": wins}
print(json.dumps(out, indent=1))
(D / f"summary{('_' + args.tag) if args.tag else ''}.json").write_text(json.dumps(out, indent=1))
