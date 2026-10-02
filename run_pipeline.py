"""The pipeline on one ds001226 subject, scan to payload, optionally written as TRX.

    DATA/.venv/bin/python bench/tractography/run_pipeline.py --sub PAT16 [--trx PATH] [--float16] [--labeled-only]

--trx PATH: a .trx zip, or a directory for any other name (default for --trx without a path:
DATA/ds001226/derived/<sub>/<sub>.trx). Prints the stage times and the payload's size.
"""
import argparse, json
import _pipeline as P
from _ds001226 import load, ROOT
from _tractcloud import Labeler

ap = argparse.ArgumentParser()
ap.add_argument("--sub", required=True)
ap.add_argument("--trx", nargs="?", const="", default=None, help="write TRX (a .trx zip, or a directory)")
ap.add_argument("--float16", action="store_true", help="TRX positions in float16 (up to 0.03 mm off)")
ap.add_argument("--labeled-only", action="store_true", help="TRX of the labeled (>= 40 mm) streamlines only")
args = ap.parse_args()

s = load(args.sub)
trx = None if args.trx is None else (args.trx or ROOT / "derived" / args.sub / f"{args.sub}.trx")
timer = P.Timer(echo=args.sub)
corr, tg, labels, payload = P.run(s, Labeler(), timer, trx=trx, positions="float16" if args.float16 else "float32",
                                  labeled_only=args.labeled_only)
print(json.dumps({"seconds": timer.seconds, "scan_to_payload_s": timer.total(*P.pipeline_stages()),
                  "streamlines": len(tg.fibers), "labeled": payload.streamlines, "payload_mb": payload.total_mb,
                  "trx": str(trx) if trx else None}, indent=1))
