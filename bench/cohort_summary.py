"""The cohort in one table and one figure, from results/cohort/<sub>.json (cohort.py).

    uv run bench/cohort_summary.py

Per subject: the tumor (participants.tsv: type, location; the mask's volume), how far our field
displaces the tumor's 10 mm margin (99th percentile), the residual against the T1 there for the scan
as acquired and as corrected (90th / 99th percentile), the validation (the uncorrected residual
against our displacement map: r, slope), what correction changes in the tracts (tract-mix r, tract
centers moved: median / max) and the pipeline's time from the scan to the labels.

Writes results/cohort_summary.md and results/cohort_summary.png.
"""
import csv, json
from pathlib import Path
from tractline.data import DATA as _DATA                          # $TRACTOGRAPHY_DATA
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
TD = _DATA / "ds001226"
P = {r["participant_id"].strip().replace("sub-", ""): r for r in csv.DictReader(open(TD / "participants.tsv"), delimiter="\t")}
key = lambda r, k: next(v for kk, v in r.items() if kk.strip() == k)
R = sorted((json.loads(p.read_text()) for p in (HERE / "results/cohort").glob("PAT??.json")), key=lambda r: r["subject"])

rows = []
for r in R:
    s, t1 = r["subject"], r["t1"]
    rows.append({"subject": s, "tumor": f"{key(P[s], 'tumor type & grade')}, {key(P[s], 'tumor location')}", "cm3": r["tumor"]["volume_cm3_on_T1"],
                 "ras": r["tumor"]["centroid_ras_mm"],
                 "disp_margin": t1["validation"]["our_displacement_tumor_margin_10mm"],
                 "unc_margin": t1["uncorrected"]["residual_tumor_margin_10mm"], "ours_margin": t1["ours"]["residual_tumor_margin_10mm"],
                 "unc_brain": t1["uncorrected"]["residual_brain"], "ours_brain": t1["ours"]["residual_brain"],
                 "precision": t1["ours"]["half_split_error_brain"], "sens": t1["estimate_sensitivity_mm"]["tumor_margin_10mm"],
                 "r": t1["validation"]["uncorrected_residual_r_vs_our_displacement"], "slope": t1["validation"]["uncorrected_residual_slope_vs_our_displacement"],
                 "mix_r": r["correction_changes"]["tract_mix_r"], "moved": r["correction_changes"]["tract_center_moved_mm_median_90th_max"],
                 "secs": r["pipeline_seconds"], "est": r["seconds"]["field_estimate"], "ukf": r["seconds"]["ours_ukf"],
                 "other": (r["arms"]["uncorrected"]["other_fraction"], r["arms"]["ours"]["other_fraction"])})

L = ["| subject | tumor | cm3 | side | our displacement at the margin, 99th | margin vs T1, uncorrected 90th / 99th | margin vs T1, ours 90th / 99th | validation r / slope | tract mix r | centers moved median / max | scan to labels |",
     "|---|---|---|---|---|---|---|---|---|---|---|"]
for w in rows:
    side = "R" if w["ras"][0] > 5 else ("L" if w["ras"][0] < -5 else "mid")
    L.append(f"| {w['subject']} | {w['tumor']} | {w['cm3']} | {side} | {w['disp_margin'][2]} | {w['unc_margin'][1]} / {w['unc_margin'][2]} | "
             f"{w['ours_margin'][1]} / {w['ours_margin'][2]} | {w['r']} / {w['slope']} | {w['mix_r']} | {w['moved'][0]} / {w['moved'][2]} | {w['secs']} s |")
med = lambda k, i=None: np.median([w[k][i] if i is not None else w[k] for w in rows])
L += ["", f"Medians over {len(rows)}: margin vs T1 99th, uncorrected {med('unc_margin', 2):.2f} mm, ours {med('ours_margin', 2):.2f} mm; "
      f"brain 99th {med('unc_brain', 2):.2f} / {med('ours_brain', 2):.2f} mm; precision (half-split, brain 99th) {med('precision', 2):.2f} mm; "
      f"estimate sensitivity at the margin 99th {med('sens', 2):.2f} mm; validation r {med('r'):.2f}, slope {med('slope'):.2f}; tract mix r {med('mix_r'):.4f}; scan to labels {med('secs'):.0f} s "
      f"(field estimate {med('est'):.0f} s, UKF {med('ukf'):.0f} s)."]
(HERE / "results/cohort_summary.md").write_text("\n".join(L) + "\n")
print("\n".join(L))

fig, ax = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
x = np.arange(len(rows))
for k, (lab, col, dx) in {"unc_margin": ("as acquired", "#c0504d", -0.2), "ours_margin": ("our correction", "#4f81bd", 0.2)}.items():
    ax.bar(x + dx, [w[k][2] for w in rows], 0.38, color=col, label=lab)
    ax.scatter(x + dx, [w[k][1] for w in rows], color="k", s=10, zorder=3)
ax.set_xticks(x, [f"{w['subject']}\n{w['cm3']} cm3" for w in rows], fontsize=8)
ax.set_ylabel("misplacement against the T1 (mm)")
ax.set_title("Tumor margin (10 mm) misplaced against the T1: bars 99th percentile, dots 90th", fontsize=10)
ax.legend(frameon=False)
fig.savefig(HERE / "results/cohort_summary.png", dpi=130)
