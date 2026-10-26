"""The pipeline on one ds001226 subject, or any BIDS diffusion series, scan to labels, optionally written as TRX.

    uv run bench/run_pipeline.py --sub PAT16 [--trx PATH] [--float16] [--labeled-only] [--device cpu] [--labeler rapidparc|hemiaug|tractcloud]
    uv run bench/run_pipeline.py --bids path/to/sub-01_dwi.nii.gz [--partners PA.nii.gz ...] [--shell 1000] [...]

--bids: tractline.bids.load - the series that correct it found and checked; corrected only when they pass
(the reason is printed either way). --shell: the b-value tracked (default: ds001226's 2800 for --sub; for
--bids, the shell with the most volumes, b-values rounded to 100).

--trx PATH: a .trx zip, or a directory for any other name (default for --trx without a path:
DATA/ds001226/derived/<sub>/<sub>.trx). Prints the stage times and the streamline counts.

--motion-eddy [rigid|linear|quadratic]: head motion and eddy-current correction (the motion_eddy slot; quadratic by
default), composed with the field into one resample, the b-vectors rotated for the tracker. --qc DIR: its QC there -
motion_eddy_qc.tsv (per volume: translations, rotations, displacement at the brain's edge from motion and from eddy
currents, framewise displacement, the residual against the volume's prediction and its z within the shell, the
number of outlier slices (signal loss, replaced by their prediction), whether it was left out, the eddy coefficients), motion_eddy_qc.json (the summary) and motion_eddy_qc.png (the traces, with matplotlib). The translation
along PE is what is left of each volume's PE offset after the part linear in its gradient is given to the eddy currents
(eddy's split); its volume-to-volume jitter, and so part of the framewise displacement, can be that split's residue,
not head motion;
--write-corrected also writes the corrected series there (corrected.nii.gz, corrected.bval, corrected.bvec: rotated).

    uv run bench/run_pipeline.py --bids sub-01_dwi.nii.gz --motion-eddy --qc out/sub-01 --write-corrected
"""
import argparse, json
from pathlib import Path
from tractline import pipeline as P
from _ds001226 import load, ROOT
from tractline.labelers import rapidparc


def plot_qc(rec, path, name):
    """The motion_eddy QC traces per volume: translations, rotations, displacement at the brain's edge (motion, eddy),
    framewise displacement, the residual z within the shell, the outlier slices replaced and the volumes left out
    (matplotlib; skipped without it)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    import numpy as np
    from tractline import motion_eddy as ME
    rows = np.array([[np.nan if x is None else x for x in r] for r in ME.qc_rows(rec)], float)
    col = {c: i for i, c in enumerate(ME.QC_COLUMNS)}
    v = rows[:, 0]
    fig, ax = plt.subplots(6, 1, figsize=(9, 13), sharex=True)
    for c in ("tx_mm", "ty_mm", "tz_mm"):
        ax[0].plot(v, rows[:, col[c]], label=c[:2])
    ax[0].set_ylabel("translation (mm)")
    for c in ("rx_deg", "ry_deg", "rz_deg"):
        ax[1].plot(v, rows[:, col[c]], label=c[:2])
    ax[1].set_ylabel("rotation (deg)")
    ax[2].plot(v, rows[:, col["rigid_edge_rms_mm"]], label="motion (RMS)")
    ax[2].plot(v, rows[:, col["eddy_edge_rms_mm"]], label="eddy (RMS)")
    ax[2].set_ylabel("at the edge (mm)")
    ax[3].bar(v, rows[:, col["framewise_displacement_mm"]], color="0.5")
    ax[3].set_ylabel("framewise (mm)")
    ax[4].bar(v, rows[:, col["residual_z"]], color=np.where(rows[:, col["residual_z"]] > 4, "C3", "0.5"))
    ax[4].axhline(4, color="C3", lw=0.8, ls="--")
    ax[4].set_ylabel("residual z")
    ax[5].bar(v, rows[:, col["outlier_slices"]], color="C1")
    left = np.flatnonzero(rows[:, col["left_out"]] > 0)
    if len(left):
        ax[5].plot(v[left], np.zeros(len(left)), "kx", label="left out")
        ax[5].legend(loc="upper right", fontsize=8)
    ax[5].set_ylabel("outlier slices")
    ax[5].set_xlabel("volume (as acquired)")
    for a in ax[:3]:
        a.legend(loc="upper right", fontsize=8, ncol=3)
    for bv, idx in rec["groups (b, volumes)"]:
        if bv == 0:
            for a in ax:
                for i in idx:
                    a.axvline(i, color="C2", lw=0.4, alpha=0.5)
    ax[0].set_title(f"{name}: head motion and eddy currents ({rec['model']}; green lines: b0s)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sub", help="a ds001226 patient (bench/_ds001226.py)")
    src.add_argument("--bids", help="a BIDS *_dwi.nii[.gz] (tractline.bids)")
    ap.add_argument("--partners", nargs="+", default=None, help="with --bids: the series that correct it, named (else found)")
    ap.add_argument("--shell", type=float, default=None)
    ap.add_argument("--trx", nargs="?", const="", default=None, help="write TRX (a .trx zip, or a directory)")
    ap.add_argument("--float16", action="store_true", help="TRX positions in float16 (up to 0.03 mm off)")
    ap.add_argument("--labeled-only", action="store_true", help="TRX of the labeled (>= 40 mm) streamlines only")
    ap.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps", help="cpu: no GPU (one tracking process per core)")
    ap.add_argument("--labeler", choices=("rapidparc", "hemiaug", "tractcloud"), default="rapidparc",
                    help="rapidparc (the default), its hemiaug model, or TractCloud at its trained context")
    ap.add_argument("--motion-eddy", nargs="?", const="quadratic", default=None, choices=("rigid", "linear", "quadratic"),
                    help="head motion and eddy-current correction (the eddy model; quadratic without a value)")
    ap.add_argument("--qc", default=None, help="with --motion-eddy: write its QC files to this folder")
    ap.add_argument("--write-corrected", action="store_true", help="with --qc: also the corrected series and b-vectors")
    args = ap.parse_args()
    if (args.qc or args.write_corrected) and not args.motion_eddy:
        ap.error("--qc and --write-corrected need --motion-eddy")
    if args.write_corrected and not args.qc:
        ap.error("--write-corrected writes into the --qc folder")

    if args.bids:
        import numpy as np
        from tractline import bids
        s = bids.load(args.bids, partners=args.partners)
        name = s.name
        b = np.round(s.bval[s.bval >= 50] / 100) * 100
        shells, counts = np.unique(b, return_counts=True)
        shell = args.shell or float(shells[np.lexsort((shells, counts))[-1]])
        print("pairing:", s.pairing.message)
    else:
        s, name, shell = load(args.sub), args.sub, args.shell or 2800.0
    trx = None if args.trx is None else (args.trx or ROOT / "derived" / name / f"{name}.trx")
    timer = P.Timer(echo=name)
    if args.labeler == "tractcloud":                                        # optional: needs TractCloud's code
        from tractline.labelers import tractcloud
        lab = tractcloud.Labeler(args.device)
    else:
        lab = rapidparc.Labeler(args.device, model=args.labeler)
    corr, tg, labels = P.run(s, lab, timer, trx=trx, device=args.device, shell=shell, motion_eddy=args.motion_eddy,
                             positions="float16" if args.float16 else "float32", labeled_only=args.labeled_only)
    cfg = P.config_for(motion_eddy=args.motion_eddy)
    out = {"corrected": corr.applied, "note": corr.note, "residual_left": corr.residual_left, "warnings": corr.warnings,
           "shell": shell, "seconds": timer.seconds, "scan_to_labels_s": timer.total(*P.pipeline_stages("", cfg)),
           "streamlines": len(tg.fibers), "labeled": int(labels.keep.sum()), "trx": str(trx) if trx else None}
    if args.qc:
        from tractline import motion_eddy as ME
        out["motion_eddy"] = ME.write_qc(corr.motion_eddy, args.qc)
        plot_qc(corr.motion_eddy, Path(args.qc) / "motion_eddy_qc.png", name)
        if args.write_corrected:
            import nibabel as nib
            import numpy as np
            src_path = getattr(s, "dwi_path", None)
            hdr = nib.load(str(src_path)).header.copy() if src_path is not None else None   # the source's codes, units, TR
            img = nib.Nifti1Image(corr.dwi, s.affine, hdr)
            img.set_data_dtype(np.float32)
            nib.save(img, str(Path(args.qc) / "corrected.nii.gz"))
            np.savetxt(Path(args.qc) / "corrected.bval", np.asarray(s.bval)[None], fmt="%g")
            np.savetxt(Path(args.qc) / "corrected.bvec", corr.bvec, fmt="%.6f")
    print(json.dumps(out, indent=1))
