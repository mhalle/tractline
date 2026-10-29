"""track:ptt's stopping without a T1 on one scan given by path, beside today's UKF (NOTES 2026-10-29, "PTT without a
T1: stopping at the mask's edge, or at low FA"): each arm through pipeline.run, distortion-corrected as tractline run
does it, the shell nearest b = 3000, with median_otsu and with the no-CSF SynthStrip mask.

    uv run python bench/ptt_stopping_scan.py <dwi.nii.gz> <tag> [<out folder>]
                                                         # -> <out folder>/ptt_stopping_<tag>.json (default results/)

Arms: "ukf" (the default tracker), "ptt, mask" (model csd, track ptt: the mask's edge, every path kept) and "ptt, fa
0.15" and "ptt, fa auto" (the mask's edge or FA below 0.15, or below the threshold set from the scan; valid ends only); PTT seeded in the mask's FA >= 0.2, 100,000 seeds. Recorded per
arm: the streamlines, labeled, the Other share, the labeled streamlines' median length, the stage times and every tract's
count. A scan that is not public keeps its outputs outside the repository: give the out folder.
"""
import json, sys, time
from pathlib import Path

import numpy as np

from tractline import bids, pipeline as P
from tractline.labelers.base import TRACT_NAMES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cst_thalamus_outside import nearest_3000  # noqa: E402

PTT = {"impl": "ptt", "seeding": "fa", "seed_count": 100_000}
ARMS = {"ukf": None, "ptt, mask": {**PTT, "stopping": "mask"},
        "ptt, fa 0.15": {**PTT, "stopping": "fa", "stopping_fa": 0.15},
        "ptt, fa auto": {**PTT, "stopping": "fa", "stopping_fa": "auto"}}


def main(dwi, tag, out=None):
    out = Path(out) if out else HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    s = bids.load(dwi)
    shell = nearest_3000(s.bval)
    labeler = P.default_labeler("mps")
    from tractline import config
    res = {"series": Path(dwi).name, "shell": shell, "track:ptt version": config.implementation("track", "ptt").version}
    for mask in ("median-otsu", "synthstrip-nocsf"):
        base = P.config_for(shell=shell, mask=None if mask == "median-otsu" else mask)
        for arm, track in ARMS.items():
            cfg = base if track is None else base.with_(model={"impl": "csd", "shell": shell}, track=track)
            timer = P.Timer()
            t = time.perf_counter()
            corr, tg, labels = P.run(s, labeler, timer, device="mps", config=cfg)
            tr = np.asarray(labels.tract)
            key = f"{arm}, {mask}"
            res[key] = {"config": cfg.to_dict(), **{k: tg.stats[k] for k in ("stopping_fa (auto)", "cortex FA estimate")
                                                    if k in tg.stats},
                        "field applied": bool(corr.applied), "streamlines": len(tg.fibers),
                        "labeled": int(np.asarray(labels.keep).sum()),
                        "Other share": round(float((tr == TRACT_NAMES.index("Other")).mean()), 3),
                        "median length mm (labeled)": round(float(np.median(labels.length_mm)), 1),
                        "seconds": round(time.perf_counter() - t, 1),
                        "stages": {k: round(v, 1) for k, v in timer.seconds.items()},
                        "tracts": {n: int((tr == i).sum()) for i, n in enumerate(TRACT_NAMES)}}
            print(key, json.dumps({k: v for k, v in res[key].items() if k != "tracts"}), flush=True)
    (out / f"ptt_stopping_{tag}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: ptt_stopping_scan.py <dwi.nii.gz> <tag> [<out folder>]   (a private scan: give an out folder "
                 "outside the repository)")
    main(*sys.argv[1:4])
