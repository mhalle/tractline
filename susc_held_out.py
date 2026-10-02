"""The field estimate's repeatability from independent data, from each scan's own b0s (the AP series'
spread over the ~15-minute scan, the PA's two back to back just before it).

    DATA/.venv/bin/python bench/tractography/susc_held_out.py --sub PAT16 [--configs '{"name": {estimate options}}'] [--quick --tag T]

Per configuration, four fits:
  A = {AP0, PA0}, B = {AP1, PA1}: one TR apart each, independent noise, barely any motion;
  C = {AP last, PA1}: the end of the scan against its start; all = every b0, for C's pose.
Measured, in the brain (the cohort's mask; deep = eroded 3 voxels, edge the rest), as |displacement
difference| mm, median / 99th / max:
  - split_half: A's field against B's;
  - drift: A's against C's, C's carried to A's frame by the full fit's motion of the last AP b0 (the
    field moves with the head), minus nothing: compare with split_half for what time adds;
  (--quick: A and B only - split_half and held_out)
  - held_out: each fit corrects the other pair - B's AP and PA unwarped by A's field, the PA moved by
    A's PA pose (PA1 follows PA0 by one TR), and the other way round - RMS of corrected AP minus
    corrected PA in the brain, relative to the brain's mean; beside it the same-polarity mismatch (AP0
    against AP1: noise, and any motion between them or between their slice groups - PAT08's is 4x the
    others') and the uncorrected mismatch (field 0, same poses).
Writes results/susc_held_out/<sub>[_tag].json.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import binary_erosion, map_coordinates
import _susc as S
from _ds001226 import load, ROOT

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument("--sub", required=True)
ap.add_argument("--configs", default=json.dumps({"lbfgs": {"optimizer": "lbfgs"}, "gn": {}}),
                help="name: estimate options; the results' 'gn' runs were the options now estimate's defaults")
ap.add_argument("--quick", action="store_true"); ap.add_argument("--tag", default="")
args = ap.parse_args()
torch.set_num_threads(8)
s = load(args.sub)
brain = np.load(ROOT / f"derived/{args.sub}/cohort_fields.npz")["brain"]
deep = binary_erosion(brain, iterations=3); edge = brain & ~deep
mm = lambda h: S.displacement_mm(h, s.readout_s, s.pe_sign, s.vox[s.pe_axis])
q = lambda d, m: [round(float(np.median(d[m])), 3), round(float(np.quantile(d[m], 0.99)), 3), round(float(d[m].max()), 3)]
ap_i = np.flatnonzero((s.pe_vectors == s.pe_vectors[0]).all(1)); pa_i = np.flatnonzero(~(s.pe_vectors == s.pe_vectors[0]).all(1))
assert len(ap_i) >= 2 and len(pa_i) >= 2
FITS = {"A": [ap_i[0], pa_i[0]], "B": [ap_i[1], pa_i[1]]}
if not args.quick:
    FITS.update(C=[ap_i[-1], pa_i[1]], all=list(range(s.b0s.shape[-1])))
pe_ax = s.pe_axis
X, Y, Z = s.b0s.shape[:3]
vox = np.asarray(s.vox, float)
center = torch.tensor((np.array([X, Y, Z]) - 1) / 2 * vox, dtype=torch.float64)


def corrected_pair(vols, h, pa_motion):
    """The AP and PA b0s (indices vols) unwarped by the field h (Hz), the PA moved by pa_motion (6,) -
    estimate()'s model at full resolution, float64 - after its --scale to a common mean."""
    img = torch.as_tensor(np.moveaxis(s.b0s[..., vols], -1, 0), dtype=torch.float64)
    img = img * (img.mean() / img.mean(dim=(1, 2, 3), keepdim=True))
    ht = torch.as_tensor(h, dtype=torch.float64)
    dh = torch.gradient(ht, dim=pe_ax)[0]
    pe_scale = torch.as_tensor(s.pe_vectors[vols, pe_ax] * s.readout_s, dtype=torch.float64)
    R, t = S.rigid(torch.as_tensor(pa_motion, dtype=torch.float64), center)
    u = S.unwarp(img, ht, dh, pe_ax, pe_scale, [torch.eye(3, dtype=torch.float64), R], [torch.zeros(3, dtype=torch.float64), t],
                 torch.as_tensor(vox))
    return u.numpy()


def rel_rms(a, b, m):
    return round(float(np.sqrt(np.mean((a[m] - b[m]) ** 2)) / np.mean(0.5 * (a[m] + b[m]))), 4)


img01 = s.b0s[..., ap_i[:2]]
img01 = img01 * (img01.mean() / img01.mean(axis=(0, 1, 2)))
res = {"subject": args.sub, "fits": {k: [int(i) for i in v] for k, v in FITS.items()},
       "same_polarity_AP0_AP1": {"brain": rel_rms(img01[..., 0], img01[..., 1], brain), "deep": rel_rms(img01[..., 0], img01[..., 1], deep)},
       "configs": {}}


def drift_of(fit):
    """|A's field - C's| (mm), C's field carried from the last AP b0's frame - which sits at R x + t of
    A's (the first AP b0's) frame, by the full fit's motion - to A's."""
    R, t = S.rigid(torch.as_tensor(fit["all"][1][ap_i[-1]], dtype=torch.float64), center)
    g = np.stack(np.meshgrid(*[np.arange(n) for n in (X, Y, Z)], indexing="ij"), -1) * vox
    p = (g @ R.numpy().T + t.numpy()) / vox
    hC = map_coordinates(fit["C"][0], np.moveaxis(p, -1, 0), order=1, mode="nearest")
    return np.abs(mm(fit["A"][0]) - mm(hC))


for name, opts in json.loads(args.configs).items():
    fit, secs = {}, {}
    for k, vols in FITS.items():
        t0 = time.time()
        h, motion, _ = S.estimate(s.b0s[..., vols], s.vox, s.pe_vectors[vols], s.readout_s, device="mps", **opts)
        secs[k] = round(time.time() - t0, 1); fit[k] = (h, motion)
    split = np.abs(mm(fit["A"][0]) - mm(fit["B"][0]))
    held = {}
    for f, other in (("A", "B"), ("B", "A")):
        u = corrected_pair(FITS[other], fit[f][0], fit[f][1][1])
        u0 = corrected_pair(FITS[other], np.zeros_like(fit[f][0]), fit[f][1][1])
        insample = corrected_pair(FITS[f], fit[f][0], fit[f][1][1])
        held[f"{f} corrects {other}"] = {"brain": rel_rms(u[0], u[1], brain), "deep": rel_rms(u[0], u[1], deep), "edge": rel_rms(u[0], u[1], edge),
                                         "uncorrected_brain": rel_rms(u0[0], u0[1], brain), "in_sample_brain": rel_rms(insample[0], insample[1], brain)}
    jmin = 1 - np.abs(s.readout_s * np.gradient(fit["A"][0], axis=pe_ax))
    r = {"seconds": secs, "split_half_deep": q(split, deep), "split_half_edge": q(split, edge), "held_out": held,
         "folds_brain_A": int((jmin[brain] < 0).sum())}
    if not args.quick:
        drift = drift_of(fit)
        r.update(drift_deep=q(drift, deep), drift_edge=q(drift, edge), last_ap_pose=[round(float(x), 3) for x in fit["all"][1][ap_i[-1]]])
    res["configs"][name] = r
    print(args.sub, name, json.dumps(r), flush=True)
out = HERE / "results/susc_held_out"; out.mkdir(exist_ok=True)
(out / f"{args.sub}{('_' + args.tag) if args.tag else ''}.json").write_text(json.dumps(res, indent=1))
print(args.sub, "same-polarity mismatch", res["same_polarity_AP0_AP1"])
