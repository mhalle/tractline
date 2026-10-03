"""The pipeline on one ds001226 subject, scan to labels, optionally written as TRX.

    python bench/run_pipeline.py --sub PAT16 [--trx PATH] [--float16] [--labeled-only] [--device cpu]

--trx PATH: a .trx zip, or a directory for any other name (default for --trx without a path:
DATA/ds001226/derived/<sub>/<sub>.trx). Prints the stage times and the streamline counts.
"""
import argparse, json
from tractline import pipeline as P
from _ds001226 import load, ROOT
from tractline.labelers import rapidparc, tractcloud


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True)
    ap.add_argument("--trx", nargs="?", const="", default=None, help="write TRX (a .trx zip, or a directory)")
    ap.add_argument("--float16", action="store_true", help="TRX positions in float16 (up to 0.03 mm off)")
    ap.add_argument("--labeled-only", action="store_true", help="TRX of the labeled (>= 40 mm) streamlines only")
    ap.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps", help="cpu: no GPU (one tracking process per core)")
    ap.add_argument("--labeler", choices=("rapidparc", "hemiaug", "tractcloud"), default="rapidparc",
                    help="rapidparc (the default), its hemiaug model, or TractCloud at its trained context")
    args = ap.parse_args()

    s = load(args.sub)
    trx = None if args.trx is None else (args.trx or ROOT / "derived" / args.sub / f"{args.sub}.trx")
    timer = P.Timer(echo=args.sub)
    lab = tractcloud.Labeler(args.device) if args.labeler == "tractcloud" else rapidparc.Labeler(args.device, model=args.labeler)
    corr, tg, labels = P.run(s, lab, timer, trx=trx, device=args.device,
                             positions="float16" if args.float16 else "float32", labeled_only=args.labeled_only)
    print(json.dumps({"seconds": timer.seconds, "scan_to_labels_s": timer.total(*P.pipeline_stages()),
                      "streamlines": len(tg.fibers), "labeled": int(labels.keep.sum()),
                      "trx": str(trx) if trx else None}, indent=1))
