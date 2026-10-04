"""The pipeline on one ds001226 subject, or any BIDS diffusion series, scan to labels, optionally written as TRX.

    uv run bench/run_pipeline.py --sub PAT16 [--trx PATH] [--float16] [--labeled-only] [--device cpu] [--labeler rapidparc|hemiaug|tractcloud]
    uv run bench/run_pipeline.py --bids path/to/sub-01_dwi.nii.gz [--partners PA.nii.gz ...] [--shell 1000] [...]

--bids: tractline.bids.load - the series that correct it found and checked; corrected only when they pass
(the reason is printed either way). --shell: the b-value tracked (default: ds001226's 2800 for --sub; for
--bids, the shell with the most volumes, b-values rounded to 100).

--trx PATH: a .trx zip, or a directory for any other name (default for --trx without a path:
DATA/ds001226/derived/<sub>/<sub>.trx). Prints the stage times and the streamline counts.
"""
import argparse, json
from tractline import pipeline as P
from _ds001226 import load, ROOT
from tractline.labelers import rapidparc


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
    args = ap.parse_args()

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
    corr, tg, labels = P.run(s, lab, timer, trx=trx, device=args.device, shell=shell,
                             positions="float16" if args.float16 else "float32", labeled_only=args.labeled_only)
    print(json.dumps({"corrected": corr.applied, "note": corr.note, "residual_left": corr.residual_left, "warnings": corr.warnings,
                      "shell": shell, "seconds": timer.seconds, "scan_to_labels_s": timer.total(*P.pipeline_stages()),
                      "streamlines": len(tg.fibers), "labeled": int(labels.keep.sum()),
                      "trx": str(trx) if trx else None}, indent=1))
