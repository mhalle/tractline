"""One ds001226 patient end to end on this Mac: the pipeline (pipeline.py) on the scan as corrected,
the same tracking and labeling on the scan as acquired, and both against the patient's T1 and tumor
mask. Nothing in between touches the disk.

    python bench/cohort.py --sub PAT13

Measured, beyond the pipeline's own stages:
  - the estimate's sensitivity: the field again from the b0s plus noise of SD 0.01 (signals in the
    hundreds; the scan's noise ~3 orders larger), the displacement difference (the optimizer's path,
    not the scan's noise). PAT16 showed the field move by 0.08 mm (median) / 0.68 mm (99th) when
    topup_ref.py's int16-quantized copy of the b0s (errors <= 0.03) was the input instead;
  - per arm, a 5-draw TractCloud vote; between the arms the tract-mix correlation, per-tract relative
    changes and tract centers moved (TractCloud's own redraw floor r 0.995-0.999 on PAT16);
  - against the T1 (t1check.py), per arm: the residual displacement a rigid alignment cannot remove,
    in the brain (the corrected arm's mask), where our field displaces > 3 mm, the tumor and its 10 mm
    margin; validation: the uncorrected arm's residual against our displacement map (r, slope);
    precision: the corrected arm's fit on b0s 1-3 and 4-6.

Writes results/cohort/<sub>.json and DATA/ds001226/derived/<sub>/cohort_fields.npz (residual maps,
masks, our displacement).
"""
import argparse, json
from pathlib import Path
import numpy as np, nibabel as nib, torch
from scipy.ndimage import binary_dilation
from tractline import susceptibility as S
from tractline import pipeline as P
from _ds001226 import load, ROOT
from tractline.t1check import T1Check, tumor_regions, stats
from tractline.labelers.base import TRACT_NAMES, OTHER

ap = argparse.ArgumentParser(); ap.add_argument("--sub", required=True); args = ap.parse_args()
HERE = Path(__file__).resolve().parent
OUT = ROOT / "derived" / args.sub
OUT.mkdir(parents=True, exist_ok=True); (HERE / "results/cohort").mkdir(exist_ok=True)
torch.set_num_threads(8)

s = load(args.sub)
timer = P.Timer(echo=args.sub)
labeler = P.default_labeler("mps")

# ------------------------------------------------------------------ the pipeline, and the scan as acquired
corr, tg, lab1 = P.run(s, labeler, timer, prefix="ours_")
with timer("field_estimate_perturbed"):
    h_p, _, _ = S.estimate(s.b0s + np.random.default_rng(0).normal(0, 0.01, s.b0s.shape), s.vox, s.pe_vectors, s.readout_s, device="mps")
arms = {"uncorrected": P.track(s, s.dwi, timer, prefix="uncorrected_"), "ours": tg}
P.label(arms["uncorrected"], labeler, timer, prefix="uncorrected_")             # the pipeline's one draw, timed
votes = {}
for name, t in arms.items():
    with timer(f"{name}_label_five_draws"):
        votes[name] = labeler(t.fibers, draws=range(5))

# ------------------------------------------------------------------ what correction changes in the tracts
def shares(v):
    c = np.bincount(v, minlength=43)[:OTHER].astype(float)
    return c / c.sum()
def centers(fibers, lab):
    kept = lab.kept(fibers)
    return {t: np.concatenate([kept[i] for i in np.flatnonzero(lab.tract == t)]).mean(0)
            for t in range(OTHER) if (lab.tract == t).sum() >= 20}
sa, sb = shares(votes["uncorrected"].tract), shares(votes["ours"].tract)
named = (sa > 0.002) | (sb > 0.002)
rel = np.abs(sb[named] - sa[named]) / np.maximum(sa[named], 1e-9)
ca, cb = (centers(arms[n].fibers, votes[n]) for n in ("uncorrected", "ours"))
moved = {TRACT_NAMES[t]: round(float(np.linalg.norm(ca[t] - cb[t])), 1) for t in ca if t in cb}
mv = np.array(list(moved.values()))

# ------------------------------------------------------------------ against the T1
d_ours = corr.displacement_mm
with timer("t1_check"):
    t1img = nib.load(s.t1)
    tumor_t1, margin_t1 = tumor_regions(t1img, nib.load(s.tumor_mask))
    brain = arms["ours"].mask.astype(bool)
    fit_mask = binary_dilation(brain, iterations=2) | binary_dilation(arms["uncorrected"].mask.astype(bool), iterations=2)
    C = T1Check(t1img, s.affine, s.dwi.shape[:3], fit_mask, s.pe_axis, s.vox)
    big = brain & (np.abs(d_ours - np.median(d_ours[brain])) > 3)
    t1res, fields = {}, {"d_ours": d_ours.astype(np.float32), "brain": brain}
    for name, data in (("uncorrected", s.dwi), ("ours", corr.dwi)):
        b0 = data[..., s.b0_index].astype(np.float64)
        T0 = C.rigid_start(b0.mean(-1))
        d, Tm, log = C.fit(b0.mean(-1), T0)
        tumor = (C.on_grid(Tm, tumor_t1.astype(float)) > 0.5).numpy()
        margin = (C.on_grid(Tm, margin_t1.astype(float)) > 0.5).numpy() & brain & ~tumor
        fields[name] = d.astype(np.float32); fields[name + "_tumor"] = tumor; fields[name + "_margin"] = margin
        t1res[name] = {"residual_brain": stats(d, brain), "residual_where_ours_displaces_gt_3mm": stats(d, big),
                       "residual_tumor": stats(d, tumor), "residual_tumor_margin_10mm": stats(d, margin),
                       "tumor_voxels": int(tumor.sum()), "margin_voxels": int(margin.sum()),
                       "ngf_after": log[-1]["ngf"], **C.rigid_summary(T0)}
        if name == "ours":
            da, _, _ = C.fit(b0[..., :3].mean(-1), T0)
            db, _, _ = C.fit(b0[..., 3:].mean(-1), T0)
            err = np.abs(da - db) / 2
            t1res[name]["half_split_error_brain"] = stats(err, brain)
            t1res[name]["half_split_error_tumor_margin"] = stats(err, margin)
    dc = d_ours - np.median(d_ours[brain])
    u = fields["uncorrected"]
    t1res["validation"] = {"our_displacement_brain": stats(dc, brain), "our_displacement_tumor": stats(dc, fields["ours_tumor"]),
                           "our_displacement_tumor_margin_10mm": stats(dc, fields["ours_margin"]),
                           "uncorrected_residual_r_vs_our_displacement": round(float(np.corrcoef(u[brain], d_ours[brain])[0, 1]), 3),
                           "uncorrected_residual_slope_vs_our_displacement": round(float(np.polyfit(d_ours[brain], u[brain], 1)[0]), 3),
                           "ours_residual_r_vs_our_displacement": round(float(np.corrcoef(fields["ours"][brain], d_ours[brain])[0, 1]), 3)}
dp = np.abs(np.asarray(h_p, dtype=np.float64) - np.asarray(corr.field_hz, dtype=np.float64)) * s.readout_s * s.vox[s.pe_axis]
t1res["estimate_sensitivity_mm"] = {"brain": stats(dp, brain), "tumor_margin_10mm": stats(dp, fields["ours_margin"])}

# ------------------------------------------------------------------ results
T = timer.seconds
tumor_c = t1img.affine[:3, :3] @ np.argwhere(tumor_t1).mean(0) + t1img.affine[:3, 3]
res = {"subject": args.sub,
       "data": f"ds001226 sub-{args.sub} ses-preop (CC0): AP {s.pe_directions[0]} + PA {s.pe_directions[1]}, readout {s.readout_s} s",
       "tumor": {"volume_cm3_on_T1": round(float(tumor_t1.sum()) / 1000, 1), "centroid_ras_mm": [round(float(v), 1) for v in tumor_c],
                 "t1_mean_inside_vs_mirror": [round(float(C.t1[tumor_t1].mean()), 1), round(float(C.t1[tumor_t1[::-1]].mean()), 1)]},
       "pa_slab_offset": s.pa_offset,
       "seconds": T,
       "pipeline_seconds": timer.total(*P.pipeline_stages("ours_")),
       "scan_to_labels_seconds": timer.total(*P.pipeline_stages("ours_")),
       "field_hz_1_50_99": [round(float(v), 1) for v in np.quantile(np.asarray(corr.field_hz)[brain], [0.01, 0.5, 0.99])],
       "motion_translation_mm_max": round(float(np.abs(corr.motion[:, :3]).max()), 2),
       "arms": {n: {"seeds": t.stats["seeds"], "fibers": t.stats["fibers"], "fiber_steps": t.stats["fiber_steps"],
                    "ukf_steps_per_s": round(t.stats["fiber_steps"] / T[f"{n}_ukf"]), "mask_voxels": t.info["mask_voxels"],
                    "streamlines_ge_40mm": int(votes[n].keep.sum()), "median_length_mm": round(float(np.median(votes[n].length_mm)), 1),
                    "other_fraction": round(float((votes[n].tract == OTHER).mean()), 4)} for n, t in arms.items()},
       "correction_changes": {"tract_mix_r": round(float(np.corrcoef(sa, sb)[0, 1]), 4),
                              "per_tract_abs_rel_change_median_90th": [round(float(np.median(rel)), 3), round(float(np.quantile(rel, 0.9)), 3)],
                              "tract_center_moved_mm_median_90th_max": [round(float(np.median(mv)), 1), round(float(np.quantile(mv, 0.9)), 1), round(float(mv.max()), 1)],
                              "tract_center_moved_mm": dict(sorted(moved.items(), key=lambda x: -x[1]))},
       "t1": t1res, "columns": "|displacement| mm: median / 90th / 99th percentile"}
(HERE / f"results/cohort/{args.sub}.json").write_text(json.dumps(res, indent=1))
np.savez_compressed(OUT / "cohort_fields.npz", **fields)
print(json.dumps({k: res[k] for k in ("tumor", "pipeline_seconds", "scan_to_labels_seconds", "correction_changes")}, indent=1))
