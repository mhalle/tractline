"""Out-of-sample test of track:ptt's stopping_fa "auto" on local ON-Harmony data (NOTES 2026-10-30, "Auto on ON-Harmony,
out of sample"): ds004712 (Siemens Prisma, Oxford; sessions OXF1PRI001, Head_32 coil, and OXF2PRI001, HeadNeck_64),
sub-16974 and the first four other subjects in sorted order (sub-10975, the known high-motion one, skipped), both
sessions each: ten scans none of which the rule was fit on.

    uv run python bench/ptt_auto_onharmony.py            # FastSurfer, fits, checks, then the summary
    uv run python bench/ptt_auto_onharmony.py --summary  # the summary only, from the per-scan results
    uv run python bench/ptt_auto_onharmony.py --force    # rerun every scan's checks (after a change to the rule)

Per scan: FastSurfer (FastSurferCNN's run_prediction in haversack's environment, VINN v2.0.0 checkpoints from its
cache, on the CPU) on the session's T1 into DATA/fastsurfer_onharmony/; the rigid fit of its conformed T1 to the
field-corrected mean b0 (bench/_register.py) with an overlay to check by eye; then ptt_fa_anatomy.py --scan and
ptt_fa_auto.py --scan into results/ptt_auto_onharmony/. The summary (results/ptt_auto_onharmony/summary.json): per
scan, auto's value and its cortex estimate against FastSurfer's cortex median FA, every arm's bad-end share (deep in
white matter, CSF, outside the brain), the best of the fixed thresholds and auto's difference from it.
"""
import json, subprocess, sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tractline import bids, pipeline as P
from tractline.data import DATA

HOME = Path.home()
HAVERSACK = HOME / "Dropbox/development/haversack/.venvs/fastsurfer/bin/python"
CKPT = HOME / ".cache/haversack/fastsurfer-checkpoints"
BENCH = Path(__file__).resolve().parent
D = DATA / "ds004712"
FS = DATA / "fastsurfer_onharmony"
RES = Path(__file__).resolve().parent / "results" / "ptt_auto_onharmony"
PY = sys.executable
SUBJECTS = ("sub-16974", "sub-03286", "sub-03997", "sub-12813", "sub-13192")
SESSIONS = ("OXF1PRI001", "OXF2PRI001")


def overlay(b0, A, labels_img, T, path, title):
    L = np.asarray(labels_img.dataobj)
    ijk = np.stack(np.meshgrid(*[np.arange(n) for n in b0.shape], indexing="ij"), -1).reshape(-1, 3)
    w = ijk @ A[:3, :3].T + A[:3, 3]
    Ti = np.linalg.inv(T)
    t = w @ Ti[:3, :3].T + Ti[:3, 3]
    Li = np.linalg.inv(labels_img.affine)
    v = np.rint(t @ Li[:3, :3].T + Li[:3, 3]).astype(int)
    ok = np.all((v >= 0) & (v < L.shape), 1)
    lab = np.zeros(len(v), L.dtype)
    lab[ok] = L[tuple(v[ok].T)]
    lab = lab.reshape(b0.shape)
    brain, vent, wm = (lab > 0) & (lab != 24), np.isin(lab, (4, 43)), np.isin(lab, (2, 41))
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    sl = [("z", b0.shape[2] // 2 + d) for d in (-12, 0, 10)] + [("x", b0.shape[0] // 2 + d) for d in (-18, 0, 18)]
    for a, (axn, k) in zip(ax.ravel(), sl):
        pick = (lambda m: m[:, :, k]) if axn == "z" else (lambda m: m[k])
        a.imshow(pick(b0).T, origin="lower", cmap="gray", vmax=np.quantile(b0, 0.995))
        for m, c, lw in ((brain, "lime", 0.8), (vent, "red", 0.8), (wm, "cyan", 0.5)):
            a.contour(pick(m).T, [0.5], colors=c, linewidths=lw)
        a.set_title(f"{axn} = {k}")
        a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=60)
    plt.close(fig)


def fastsurfer(t1, tag):
    """FastSurfer's asegdkt labels and conformed T1 for one T1, on the CPU (its MPS path fails), into FS/tag/."""
    FS.mkdir(parents=True, exist_ok=True)
    ck = {v: str(CKPT / f"aparc_vinn_{n}_v2.0.0.pkl") for v, n in (("ax", "axial"), ("cor", "coronal"), ("sag", "sagittal"))}
    with open(FS / f"{tag}.log", "w") as log:
        subprocess.run([str(HAVERSACK), "-I", "-m", "FastSurferCNN.run_prediction", "--t1", str(t1), "--sd", str(FS),
                        "--sid", tag, "--asegdkt_segfile", str(FS / tag / "aparc.DKTatlas+aseg.deep.mgz"),
                        "--conformed_name", str(FS / tag / "orig.mgz"), "--ckpt_ax", ck["ax"], "--ckpt_cor", ck["cor"],
                        "--ckpt_sag", ck["sag"], "--device", "cpu", "--viewagg_device", "cpu", "--batch_size", "4"],
                       cwd=FS, stdout=log, stderr=subprocess.STDOUT, check=True)


def summary():
    from tractline import __version__
    out = {"tractline": __version__, "auto rule (pipeline.AUTO_FA)": {k: list(v) if isinstance(v, tuple) else v
                                                                     for k, v in P.AUTO_FA.items()}, "scans": {}}
    for s in SUBJECTS:
        for ses in SESSIONS:
            tag = f"{s}_{ses}"
            a = json.loads((RES / f"ptt_fa_anatomy_{tag}.json").read_text())[tag]
            au = json.loads((RES / f"ptt_fa_auto_{tag}.json").read_text())
            bad = {arm: round(sum(x["all ends"][k] for k in ("white matter, deep", "CSF", "outside the brain")), 4)
                   for arm, x in a["arms"].items()}
            fixed = {k: v for k, v in bad.items() if k.startswith("ptt, fa 0.")}
            best = min(fixed, key=fixed.get)
            out["scans"][tag] = {
                "shell tracked": a["shell tracked"], "auto stopping_fa": a["arms"]["ptt, fa auto"].get("stopping_fa (auto)"),
                "auto, no-CSF mask": au["synthstrip-nocsf"]["auto (pipeline.auto_stopping_fa)"]["stopping_fa"],
                "cortex FA estimate": au["median-otsu"]["auto (pipeline.auto_stopping_fa)"]["cortex FA estimate"],
                "FastSurfer cortex median FA": au["median-otsu"]["FastSurfer cortex median FA"],
                "bad ends": bad, "best fixed": best, "auto minus best": round(bad["ptt, fa auto"] - fixed[best], 4)}
    (RES / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def main(force=False):
    """force: rerun the checks on every scan (their results otherwise kept: a change to the rule or the tracker
    needs --force, or the summary is rebuilt from the old results). FastSurfer and the fits are kept either way."""
    RES.mkdir(parents=True, exist_ok=True)
    for s in SUBJECTS:
        for ses in SESSIONS:
            tag = f"{s}_{ses}"
            labels = FS / tag / "aparc.DKTatlas+aseg.deep.mgz"
            if not labels.exists():
                fastsurfer(D / s / f"ses-{ses}" / "anat" / f"{s}_ses-{ses}_T1w.nii.gz", tag)
            dwi = D / s / f"ses-{ses}" / "dwi" / f"{s}_ses-{ses}_dir-AP_dwi.nii.gz"
            fit = FS / f"{tag}_t1_to_dwi.npz"
            if not fit.exists():
                sub = bids.load(dwi)
                corr = P.correct(sub, P.Timer(), "mps")
                b0 = np.asarray(corr.dwi)[..., np.asarray(sub.bval) < 50].mean(-1).astype(np.float32)
                t1 = nib.load(str(FS / tag / "orig.mgz"))
                np.savez(FS / f"{tag}_register_in.npz", moving=np.asarray(t1.dataobj, np.float32), moving_affine=t1.affine,
                         static=b0, static_affine=sub.affine)
                subprocess.run([PY, "-I", str(BENCH / "_register.py"), str(FS / f"{tag}_register_in.npz"),
                                str(FS / f"{tag}_raw.npz")], check=True)
                z = np.load(FS / f"{tag}_raw.npz")
                np.savez(fit, t1_to_dwi=z["to_static"], ncc=z["ncc"])
                overlay(b0, sub.affine, nib.load(str(labels)), z["to_static"], FS / f"{tag}_registration_check.png",
                        f"{tag}: FastSurfer (brain lime, ventricles red, WM cyan) on the corrected mean b0")
                print(tag, "fit ncc", round(float(z["ncc"]), 3), "translation", np.round(z["to_static"][:3, 3], 2).tolist(),
                      flush=True)
            for script in (str(BENCH / "ptt_fa_anatomy.py"), str(BENCH / "ptt_fa_auto.py")):
                outname = RES / (("ptt_fa_anatomy_" if "anatomy" in script else "ptt_fa_auto_") + tag + ".json")
                if outname.exists() and not force:
                    continue
                r = subprocess.run([PY, script, "--scan", str(dwi), str(labels), str(fit), tag, str(RES)],
                                   capture_output=True, text=True)
                if r.returncode:
                    raise SystemExit(f"{tag}: {Path(script).name} failed\n{r.stderr[-2000:]}")
                print(tag, Path(script).name, "done", flush=True)


if __name__ == "__main__":
    if "--summary" not in sys.argv:
        main(force="--force" in sys.argv)
    print(json.dumps(summary(), indent=1))
