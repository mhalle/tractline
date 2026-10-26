"""Are RapidParc's names right on PTT's PTT-only paths? (decision 5; NOTES 2026-10-26.) On cIIIs04 only 14 % of PTT's
streamlines had a UKF-like counterpart, and only those could be checked against UKF. The ISMRM 2015 phantom has a truth:
the 2023 ROI scoring's verdict per streamline (robust_phantom.verdicts: a 2023 bundle, "IC" or "NC"), and each bundle's
ORG counterparts (robust_phantom.STRICT; LOOSE adds the near ones). So, for the valid, labeled streamlines of the bundles
with a counterpart, is RapidParc's name a counterpart of the true bundle - and does that differ between PTT's UKF-like
and PTT-only streamlines?

    uv run --group dipy --group rankfield bench/naming_phantom.py         # results/naming_phantom.json

Runs (existing, scored): today's UKF (robust_phantom's "today") and our PTT in real units at 20 degrees on DIPY's FODs
(ptt_compare.py's mm_probe1.5_angle20; the phantom's closest run to the proposed defaults). A PTT streamline is UKF-like
when a UKF streamline lies within RADIUS mm of it (RMS between the 20-point paths, the closer orientation), else
PTT-only. Reported per group: the streamlines, the share named a counterpart (strict; strict or loose), the share named
Other, the share named a tract the phantom cannot hold (robust_phantom.FALSE_TRACTS), per bundle and the bundles' mean.
The baseline is UKF's streamlines, all of them and split the same way (a PTT path within RADIUS, or UKF-only): a bundle's
shared core is named better than its periphery by either tracker, so UKF-only against PTT-only is the like-for-like
comparison (the review's control, 2026-10-26). The false-tract share is also given over all labeled streamlines and over
those the scoring calls IC or NC.
"""
import json, sys, time
from pathlib import Path
import numpy as np
import nibabel as nib
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import robust_benchmark as RB  # noqa: E402
import robust_phantom as RP  # noqa: E402
from tractline import ensemble as E  # noqa: E402
from tractline.labelers.base import TRACT_NAMES  # noqa: E402

RADIUS = 4.0
OURS = "mm_probe1.5_angle20"


def nearest_rms(query, ref, device="mps"):
    """Per query path (n, 20, 3), the RMS distance (mm) to the nearest ref path, the closer orientation."""
    import torch
    dev = torch.device(device)
    Q = torch.as_tensor(query.reshape(len(query), -1), dtype=torch.float32, device=dev)
    R = torch.as_tensor(ref, dtype=torch.float32, device=dev)
    Rf, R = R.flip(1).reshape(len(ref), -1), R.reshape(len(ref), -1)
    out = np.empty(len(query))
    for s in range(0, len(Q), 2048):
        d = torch.minimum(torch.cdist(Q[s:s + 2048], R), torch.cdist(Q[s:s + 2048], Rf)).min(1).values
        out[s:s + 2048] = d.cpu().numpy() / query.shape[1] ** 0.5
    return out


def load(d):
    fibers = list(nib.streamlines.load(str(d / "tractogram.trk")).streamlines)
    lab = np.load(d / "labels.npz")
    vd, _ = RP.verdicts(d)
    name = np.full(len(fibers), "", dtype=object)                         # "" = not labeled (under 40 mm)
    name[np.flatnonzero(lab["keep"])] = np.array(TRACT_NAMES)[lab["tract"]]
    return fibers, name, vd


def judge(name, vd, sel):
    """Over the valid, labeled streamlines of the bundles with a counterpart, within `sel`."""
    m = sel & (name != "") & np.isin(vd, list(RP.STRICT))
    out = {"streamlines": int(m.sum())}
    if not m.any():
        return out
    strict = np.array([n in RP.STRICT[b] for n, b in zip(name[m], vd[m])])
    loose = np.array([n in RP.STRICT[b] + RP.LOOSE.get(b, ()) for n, b in zip(name[m], vd[m])])
    out.update({"named a counterpart": round(float(strict.mean()), 4),
                "named a counterpart or a near one": round(float(loose.mean()), 4),
                "named Other": round(float((name[m] == "Other").mean()), 4),
                "named a tract the phantom cannot hold": round(float(np.isin(name[m], RP.FALSE_TRACTS).mean()), 4)})
    per = {}
    for b in RP.STRICT:
        mb = m & (vd == b)
        if mb.sum() >= 20:
            per[b] = {"streamlines": int(mb.sum()),
                      "named a counterpart": round(float(np.isin(name[mb], RP.STRICT[b]).mean()), 4)}
    out["per bundle"] = per
    out["bundle mean"] = round(float(np.mean([v["named a counterpart"] for v in per.values()])), 4) if per else None
    return out


def false_names(name, vd, sel):
    """Of the labeled streamlines in sel: the share named a tract the phantom cannot hold - all of them, those in a true
    bundle, and those the scoring calls IC or NC."""
    lab = sel & (name != "")
    f = np.isin(name, RP.FALSE_TRACTS)
    bad = np.isin(vd, ["IC", "NC"])
    share = lambda m: round(float(f[m].mean()), 4) if m.any() else None  # noqa: E731
    return {"all labeled": share(lab), "in a true bundle": share(lab & ~bad), "IC or NC": share(lab & bad)}


def main():
    t0 = time.time()
    uf, un, uv = load(RP.OUT / "today")
    pf, pn, pv = load(RB.OUT / "ptt_ours" / OURS / "phantom")
    up, pp = E.resample(uf), E.resample(pf)
    like = nearest_rms(pp, up) <= RADIUS
    ulike = nearest_rms(up, pp) <= RADIUS                                # the control: UKF split the same way
    res = {"ukf (today), all streamlines": judge(un, uv, np.ones(len(uf), bool)),
           "ukf, PTT-like (a PTT path within 4 mm)": judge(un, uv, ulike),
           "ukf, UKF-only": judge(un, uv, ~ulike),
           f"ptt ({OURS}), all streamlines": judge(pn, pv, np.ones(len(pf), bool)),
           "ptt, UKF-like (a UKF path within 4 mm)": judge(pn, pv, like),
           "ptt, PTT-only": judge(pn, pv, ~like),
           "named a tract the phantom cannot hold": {"ukf": false_names(un, uv, np.ones(len(uf), bool)),
                                                     "ptt": false_names(pn, pv, np.ones(len(pf), bool))},
           "ukf streamlines that are PTT-like": round(float(ulike.mean()), 4),
           "ptt streamlines that are UKF-like": round(float(like.mean()), 4),
           "ptt valid streamlines (a bundle) that are UKF-like": round(float(like[pv != "IC"][pv[pv != "IC"] != "NC"].mean()), 4),
           "radius mm": RADIUS, "seconds": round(time.time() - t0, 1)}
    (HERE / "results" / "naming_phantom.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "per bundle"} if isinstance(v, dict) else v)
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
