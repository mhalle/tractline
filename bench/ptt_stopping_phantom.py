"""track:ptt's stopping without a T1, on the ISMRM 2015 phantom (NOTES 2026-10-29, "PTT without a T1: stopping at the
mask's edge, or at low FA"): the stopping maps as the pipeline makes them (pipeline.ptt_stopping), every other setting the
slot's defaults (real units, 20 degrees), our CSD on our own sphere, seeds uniform in the tracking mask's FA >= 0.2
(seeding "fa", 100,000, generator 1), RapidParc labels and the 2023 ROI scoring; then each arm with 60,000 streamlines or
more subsampled to each of COUNTS (ptt_compare's subsample: F1 depends on the count) beside today's UKF and step 6's PTT with ACT
(ptt_compare's mm_probe1.5_angle20: DIPY's FODs and sphere, seeds in WM; not like for like). The like-for-like ACT arm
is "act": the same FODs, seeds and settings, stopping act with step 6's tissue maps from the phantom's T1.

    uv run --group dipy --group rankfield python bench/ptt_stopping_phantom.py     # -> results/ptt_stopping_phantom.json
    uv run --group dipy --group rankfield python bench/ptt_stopping_phantom.py --matched-only
                                    # the count-matched blocks again, from the arms already run (no tracking)
    uv run --group dipy --group rankfield python bench/ptt_stopping_phantom.py --spread
                                    # run to run: SPREAD_ARMS with tracking seeds 2 and 3 beside 1, at SPREAD_N

Arms: "mask, version 1" (the mask's edge, only paths leaving it at both ends: track:ptt 1's rule), "mask" (every path:
version 2's), "fa 0.10", "fa 0.15", "fa 0.20" (the mask's edge or FA below that, valid ends only), "act" (the T1's tissue
maps). Records under ptt_compare's OUT/stop_<arm>/phantom/, and their subsamples under OUT/matched_<n>/stop_<arm>/: both
removed first (--matched-only: the subsamples only) (the scoring and the subsample keep any result they find, so a rerun would report the old ones).
"""
import argparse, json, shutil, sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import robust_benchmark as RB  # noqa: E402
import robust_phantom as RP  # noqa: E402
import ptt_compare as PC  # noqa: E402
import _benchmark_dipy as BD  # noqa: E402
from tractline import config, csd as C, fields as F, pipeline as P, ptt as PT  # noqa: E402
from tractline.labelers.base import OTHER, lengths  # noqa: E402

ARMS = {"mask, version 1": ({"stopping": "mask"}, False), "mask": ({"stopping": "mask"}, True),
        "fa 0.10": ({"stopping": "fa", "stopping_fa": 0.10}, False),
        "fa 0.15": ({"stopping": "fa", "stopping_fa": 0.15}, False),
        "fa 0.20": ({"stopping": "fa", "stopping_fa": 0.20}, False), "act": ({"stopping": "act"}, False)}
COUNTS = (55_000, 60_000)                                               # 55,000: the act arm's count
SPREAD_ARMS, SPREAD_RNG, SPREAD_N = ("fa 0.15", "act"), (1, 2, 3), 50_000
REFERENCES = {"ukf (today)": "ukf", "ptt + ACT, step 6's FODs and seeds": "mm_probe1.5_angle20"}


def folder(arm):
    return "stop_" + arm.replace(", ", "_").replace(" ", "")


def headline(d):
    vd, _ = RP.verdicts(d)
    r = RP.headline(json.loads((d / "score" / "final" / "results.json").read_text()), vd, np.ones(len(vd), bool))
    return {k: v for k, v in r.items() if not isinstance(v, dict)}


def matched(res):
    """Each COUNTS block: the arms with that many streamlines or more, and the references, subsampled and scored."""
    names = {folder(a): a for a in ARMS} | {v: k for k, v in REFERENCES.items()}
    for n in COUNTS:
        for src in [folder(a) for a in ARMS] + list(REFERENCES.values()):   # every subsample made anew
            shutil.rmtree(PC.OUT / f"matched_{n}" / src, ignore_errors=True)
        sources = [folder(a) for a in ARMS if res["arms"][a]["run"]["streamlines"] >= n] + list(REFERENCES.values())
        PC.subsample(argparse.Namespace(n=n, sources=sources))
        res[f"count-matched ({n:,})"] = {names[s]: headline(PC.OUT / f"matched_{n}" / s) for s in sources}


def setup():
    """The work file's arrays, the FODs, FA, the tissue maps and the seeds, as every arm shares them."""
    work, winfo = RB.work_file("phantom")
    w = np.load(work)
    dwi, bval, affine, mask = w["dwi"], w["bval"], w["affine"], w["mask"].astype(bool)
    bvec = BD.flip(affine) @ w["bvec"]                                   # image axes
    order = 8 if int((bval > 50).sum()) >= 45 else 6
    resp = C.response_ssst(dwi, bval, bvec)
    coeffs, _ = C.CSD(bval, bvec, resp, order=order).fit(dwi, mask, device="mps")
    ev, _ = C.tensor_fit(dwi[mask].astype(np.float64), bval, bvec)
    fa = np.zeros(mask.shape)
    fa[mask] = C.fa(ev)
    pve = np.load(RB.tissue("phantom", work))["pve"].astype(np.float64)
    impl = config.implementation("track", "ptt")
    seeds = P.ptt_seeds(mask & (fa >= impl.resolve({})["fa_seed"]), affine, 100_000, 1)
    return dict(winfo=winfo, affine=affine, mask=mask, coeffs=coeffs, fa=fa, pve=pve, impl=impl, seeds=seeds,
                labeler=P.default_labeler("mps"))


def run_arm(ctx, arm, d, track_rng=None):
    """One arm tracked, labeled, written into d and scored: its runs.json log."""
    opts, keep_open = ARMS[arm]
    affine, mask, seeds, impl = ctx["affine"], ctx["mask"], ctx["seeds"], ctx["impl"]
    v = C.hemisphere(181)
    vox = np.linalg.norm(affine[:3, :3], axis=0)
    labeler = ctx["labeler"]
    coeffs, fa, pve = ctx["coeffs"], ctx["fa"], ctx["pve"]
    o = impl.resolve({**opts, "seeding": "fa", **({"track_rng": track_rng} if track_rng else {})})
    d.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    include, exclude = P.ptt_stopping(o, mask, tissue=pve, fa_map=fa)
    params = PT.Params(step_size=o["step_size"], max_angle=o["max_angle"], probe_length=o["probe_length"],
                       probe_quality=o["probe_quality"], pmf_threshold=o["pmf_threshold"], min_len=o["min_len"],
                       max_len=o["max_len"], keep_open_ends=keep_open)
    tracker = PT.Tracker(F.PmfField(coeffs, mask, np.r_[v, -v], "mps"), F.Act(include, exclude, "mps"), affine, vox,
                         params, "mps", backend="metal")
    fibers, tinfo = tracker.track(seeds, seed=o["track_rng"])
    labels = labeler(fibers, draws=(0,), logp=True)
    P.release_memory("mps")
    _, _, lens = lengths(fibers)
    log = {"seeds": len(seeds), "tracker": tinfo, "streamlines": len(fibers), "labeled": int(labels.keep.sum()),
           "Other share": round(float((labels.tract == OTHER).mean()), 4),
           "length mm (quartiles), all": [round(float(x), 1) for x in np.quantile(lens, [0.25, 0.5, 0.75])],
           "seconds": round(time.perf_counter() - t, 1),
           "settings": {**{k: o[k] for k in ("stopping", "stopping_fa", "max_angle", "step_size", "probe_length",
                                             "pmf_threshold", "min_len", "max_len", "fa_seed")},
                        "open ends kept": keep_open}}
    np.savez_compressed(d / "labels.npz", keys=np.arange(len(fibers)), keep=labels.keep,
                        tract=labels.tract.astype(np.int16), length_mm=labels.length_mm.astype(np.float32))
    log["trk"] = RP.write_trk(d / "tractogram.trk", fibers)
    RB.log_update(d / "runs.json", "phantom", log)
    print(arm, json.dumps({k: log[k] for k in ("streamlines", "labeled", "Other share", "seconds")}), flush=True)
    RB.log_update(d / "runs.json", "scoring_seconds", RP.score(d))
    return log


def main():
    ctx = setup()
    for arm in ARMS:                                                     # a rerun starts clean (module docstring)
        shutil.rmtree(PC.OUT / folder(arm), ignore_errors=True)
    res = {"work": ctx["winfo"], "seeds": len(ctx["seeds"]), "track:ptt version": ctx["impl"].version, "arms": {}}
    for arm in ARMS:
        d = PC.OUT / folder(arm) / "phantom"
        log = run_arm(ctx, arm, d)
        res["arms"][arm] = {"run": {k: log[k] for k in ("streamlines", "labeled", "Other share",
                                                        "length mm (quartiles), all", "settings")},
                            "all streamlines": headline(d)}
    matched(res)
    write(res)


def spread():
    """SPREAD_ARMS with other tracking seeds (the same seed points), each subsampled to SPREAD_N with seed 1's run."""
    path = HERE / "results" / "ptt_stopping_phantom.json"
    res = json.loads(path.read_text())
    ctx = setup()
    block = {}
    for arm in SPREAD_ARMS:
        for r in SPREAD_RNG:
            name = folder(arm) + ("" if r == 1 else f"_track{r}")
            if r != 1:
                shutil.rmtree(PC.OUT / name, ignore_errors=True)
                log = run_arm(ctx, arm, PC.OUT / name / "phantom", track_rng=r)
                print(arm, r, json.dumps({k: log[k] for k in ("streamlines", "labeled")}), flush=True)
            shutil.rmtree(PC.OUT / f"matched_{SPREAD_N}" / name, ignore_errors=True)
            PC.subsample(argparse.Namespace(n=SPREAD_N, sources=[name]))
            block[f"{arm}, tracking seed {r}"] = headline(PC.OUT / f"matched_{SPREAD_N}" / name)
    res[f"run to run ({SPREAD_N:,} each)"] = block
    path.write_text(json.dumps(res, indent=1, default=str) + "\n")
    for k, r in block.items():
        print(k, json.dumps({x: r[x] for x in ("VS_ratio", "mean_OL_all21", "mean_OR_vs_all21", "mean_f1_all21")}))


def write(res):
    (HERE / "results" / "ptt_stopping_phantom.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    for n in COUNTS:
        for k, r in res[f"count-matched ({n:,})"].items():
            print(n, k, json.dumps({x: r[x] for x in ("VB", "VS_ratio", "IC_ratio", "NC_ratio", "mean_OL_all21",
                                                      "mean_OR_vs_all21", "mean_f1_all21")}))


def matched_only():
    path = HERE / "results" / "ptt_stopping_phantom.json"
    res = json.loads(path.read_text())
    for k in [k for k in res if k.startswith("count-matched")]:
        del res[k]
    matched(res)
    write(res)


if __name__ == "__main__":
    matched_only() if "--matched-only" in sys.argv else spread() if "--spread" in sys.argv else main()
