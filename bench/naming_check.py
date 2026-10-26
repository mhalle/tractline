"""Do RapidParc's tract names hold on PTT's streamlines? (docs/sota-pipeline-design.md, decision 5.) RapidParc was
trained on UKF tractography; on PTT's it is off its domain. Without a ground truth on a real brain, this checks
consistency with UKF on cIIIs04's reference scan A, separating naming from coverage (where each tracker's streamlines go).

    uv run --group dipy --group rankfield bench/naming_check.py           # results/naming_check.json

Records, all on A's grid (no new tracking): today's UKF (A and the same-session rescan, ensemble_retest), our PTT in real
units at 20 degrees (ptt_compare.py's mm-own_probe1.5_angle20: A and within), DIPY's PTT (step 6: A and within).

Territory (coverage and naming together). Per named tract, every record's streamlines of the tract subsampled to one
count n (the smallest of the five records' counts; DRAWS draws), so that presence (density >= 0.3 per mm^2,
robust_ukf_score.Presence) does not reward a count: presence Dice and the distance between the presences' centers of
mass, across trackers on A (UKF / ours, UKF / DIPY, ours / DIPY) and within each tracker (A against its rescan: the scale,
for Dice and for center distance).

Naming where the paths overlap. Each streamline's RapidParc label against the majority label of the UKF-A streamlines
within RADIUS mm of it (RMS distance between the 20-point paths, the closer orientation; at most K voters, the nearest
first; ties to the smaller tract index); a streamline with no UKF-A path within RADIUS has no counterpart. For our PTT and
DIPY's, and as the baseline UKF's own rescan. Reported overall, by the nearest voter's distance (so that the baseline's
closer matches do not decide), and per tract with the number of overlapping streamlines and the labels the voters gave
instead.

Classes:
  naming suspect     at least MIN_N overlapping streamlines, and agreement more than 0.15 below the baseline's for the
                     tract;
  coverage           fewer than 5 % of the tract's PTT streamlines overlap UKF's, or one tracker labels four times as many
                     as the other, or the UKF / ours center distance exceeds UKF's own rescan center distance by 5 mm;
  UKF-unstable       UKF's own rescan Dice below 0.5: UKF is no reference there.
"""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import robust_benchmark as RB  # noqa: E402
from robust_ukf_score import Presence, ONEHOT  # noqa: E402
from tractline import ensemble as E  # noqa: E402
from tractline.labelers.base import TRACT_NAMES, LUT  # noqa: E402

DRAWS = 3
OURS = "mm-own_probe1.5_angle20"
RADIUS, K, MIN_N = 4.0, 5, 30
BANDS = ((0.0, 2.0), (2.0, 3.0), (3.0, 4.0))
L = np.asarray(LUT)
OTHER = TRACT_NAMES.index("Other")


def top_probability(rec):
    """Per streamline, RapidParc's top tract probability (the tract's clusters summed)."""
    out = []
    for s in range(0, len(rec.keys), 8192):
        out.append((np.exp(rec.logp[s:s + 8192].astype(np.float32)) @ ONEHOT).max(1))
    return np.concatenate(out)


def knn_labels(query, ref, device="mps"):
    """Per `query` streamline: the majority tract of the `ref` streamlines within RADIUS (at most K, nearest first; ties
    to the smaller index; -1 when none), the nearest one's distance, and the number of voters."""
    import torch
    dev = torch.device(device)
    Q = torch.as_tensor(np.asarray(query.paths, np.float32).reshape(len(query.keys), -1), device=dev)
    R = torch.as_tensor(np.asarray(ref.paths, np.float32), device=dev)
    Rf = R.flip(1).reshape(len(ref.keys), -1)
    R = R.reshape(len(ref.keys), -1)
    rt = L[ref.cluster]
    npts = query.paths.shape[1]
    vote = np.full(len(Q), -1, np.int64)
    nearest = np.zeros(len(Q))
    voters = np.zeros(len(Q), np.int64)
    for s in range(0, len(Q), 1024):
        dd = torch.minimum(torch.cdist(Q[s:s + 1024], R), torch.cdist(Q[s:s + 1024], Rf)) / npts ** 0.5  # RMS
        dist, idx = torch.topk(dd, K, dim=1, largest=False)
        dist, lab = dist.cpu().numpy(), rt[idx.cpu().numpy()]
        within = dist <= RADIUS
        nearest[s:s + len(dist)] = dist[:, 0]
        voters[s:s + len(dist)] = within.sum(1)
        for i in np.flatnonzero(within[:, 0]):
            v = np.bincount(lab[i][within[i]], minlength=len(TRACT_NAMES))
            vote[s + i] = int(np.argmax(v))
    return vote, nearest, voters


def knn_agreement(query, ref):
    vote, near, voters = knn_labels(query, ref)
    own = L[query.cluster]
    has = vote >= 0
    agree = lambda m: round(float((vote[m] == own[m]).mean()), 4) if m.sum() else None  # noqa: E731
    named = has & (own != OTHER)
    per = {}
    for k, name in enumerate(TRACT_NAMES):
        mine = own == k
        if mine.sum() < 20:
            continue
        sel = mine & has
        instead = np.bincount(vote[sel & (vote != k)], minlength=len(TRACT_NAMES))
        top = [(TRACT_NAMES[t], int(instead[t])) for t in np.argsort(-instead)[:3] if instead[t] > 0]
        per[name] = {"streamlines": int(mine.sum()), "overlapping": int(sel.sum()),
                     "with a counterpart": round(float(sel.mean() / max(mine.mean(), 1e-12)), 4) if mine.any() else 0.0,
                     "agreement": agree(sel), "voters gave instead": top}
    return {"with a counterpart": round(float(has.mean()), 4), "agreement": agree(has),
            "agreement, named tracts only": agree(named),
            "by the nearest voter's distance (mm)": {f"{a:g}-{b:g}": {"share of the overlapping": round(float(
                (has & (near > a) & (near <= b)).sum() / max(has.sum(), 1)), 4), "agreement, named": agree(
                named & (near > a) & (near <= b))} for a, b in BANDS},
            "voters, mean": round(float(voters[has].mean()), 2) if has.any() else 0.0, "per tract": per}


def main():
    t0 = time.time()
    d = RB.DATA / "ensemble" / RB.SUBJECT / "retest" / RB.PAIR
    affine = np.load(d / "landmarks.npz")["affine"]
    A0 = E.RunRecord.load(d / "A.npz", logp=False)
    presence = Presence(affine, A0.shape)
    load = lambda p: E.RunRecord.load(p, logp=True, fibers=True)  # noqa: E731
    ours_dir = RB.OUT / "ptt_ours" / OURS / RB.SUBJECT
    dipy_dir = RB.OUT / "ptt" / RB.SUBJECT
    recs = {"ukf A": load(d / "A.npz"), "ukf within": load(d / "within_onA.npz"),
            "ours A": load(ours_dir / "A.npz"), "ours within": load(ours_dir / "within.npz"),
            "dipy A": load(dipy_dir / "A.npz")}
    shape = A0.shape
    ijk = np.stack(np.unravel_index(np.arange(int(np.prod(shape))), shape), 1).astype(np.float64)
    world = ijk @ affine[:3, :3].T + affine[:3, 3]
    dice = lambda a, b: 2 * float((a & b).sum()) / max(float(a.sum() + b.sum()), 1)  # noqa: E731
    center = lambda a, b: float(np.linalg.norm(world[a].mean(0) - world[b].mean(0)))  # noqa: E731
    pairs = {"across: ukf / ours (A)": ("ukf A", "ours A"), "across: ukf / dipy (A)": ("ukf A", "dipy A"),
             "across: ours / dipy (A)": ("ours A", "dipy A"), "rescan: ukf (A / within)": ("ukf A", "ukf within"),
             "rescan: ours (A / within)": ("ours A", "ours within")}
    naming = {"ukf within (the baseline: the same tracker, another scan)": knn_agreement(recs["ukf within"], recs["ukf A"]),
              "ours A": knn_agreement(recs["ours A"], recs["ukf A"]),
              "dipy A": knn_agreement(recs["dipy A"], recs["ukf A"])}
    base = naming["ukf within (the baseline: the same tracker, another scan)"]["per tract"]
    mine = naming["ours A"]["per tract"]
    out = {"records": {k: int(len(r.keys)) for k, r in recs.items()}, "ours": OURS, "draws": DRAWS,
           "naming: radius mm, voters at most, min overlapping for a suspect": [RADIUS, K, MIN_N], "tracts": {}}
    for k, name in enumerate(TRACT_NAMES):
        if k == OTHER:
            continue
        counts = {r: int((L[rec.cluster] == k).sum()) for r, rec in recs.items()}
        n = min(counts.values())
        row = {"labeled on A": {r: counts[r] for r in recs if r.endswith("A")}, "matched count": n}
        if n < 20:
            row["note"] = "fewer than 20 streamlines in some record: territory not compared"
        else:
            vals = {p: [] for p in pairs}
            cd = {"ukf / ours (A)": [], "ukf rescan": [], "ours rescan": []}
            for dr in range(DRAWS):
                pres = {r: presence(RB.tract_subset(rec, k, n, 100 * dr + i)).get(k) for i, (r, rec) in enumerate(recs.items())}
                for p, (a, b) in pairs.items():
                    vals[p].append(dice(pres[a], pres[b]) if pres[a] is not None and pres[b] is not None else 0.0)
                for c, (a, b) in (("ukf / ours (A)", ("ukf A", "ours A")), ("ukf rescan", ("ukf A", "ukf within")),
                                  ("ours rescan", ("ours A", "ours within"))):
                    if pres[a] is not None and pres[b] is not None and pres[a].any() and pres[b].any():
                        cd[c].append(center(pres[a], pres[b]))
            row.update({p: round(float(np.mean(v)), 4) for p, v in vals.items()})
            row["center distance mm"] = {c: round(float(np.mean(v)), 2) if v else None for c, v in cd.items()}
        mn, bs = mine.get(name), base.get(name)
        if mn is not None:
            row["naming (ours)"] = mn
        if bs is not None:
            row["naming baseline (ukf rescan)"] = {x: bs[x] for x in ("overlapping", "agreement")}
        cls = []
        if mn and bs and mn["overlapping"] >= MIN_N and mn["agreement"] is not None and bs["agreement"] is not None \
                and mn["agreement"] < bs["agreement"] - 0.15:
            cls.append("naming suspect")
        ca, cb = counts["ukf A"], counts["ours A"]
        cdm = row.get("center distance mm", {})
        if (mn and mn["with a counterpart"] < 0.05) or max(ca, cb) > 4 * max(min(ca, cb), 1) or (
                cdm.get("ukf / ours (A)") is not None and cdm.get("ukf rescan") is not None
                and cdm["ukf / ours (A)"] > cdm["ukf rescan"] + 5.0):
            cls.append("coverage")
        if row.get("rescan: ukf (A / within)", 1.0) < 0.5:
            cls.append("UKF-unstable")
        row["classes"] = cls
        out["tracts"][name] = row
    out["naming where the paths overlap"] = naming
    rows = [r for r in out["tracts"].values() if "across: ukf / ours (A)" in r]
    out["summary"] = {
        "territory compared": len(rows),
        **{f"median {p}": round(float(np.median([r[p] for r in rows])), 4) for p in pairs},
        "median center distance mm (ukf / ours; ukf rescan; ours rescan)": [
            round(float(np.median([r["center distance mm"][c] for r in rows if r["center distance mm"][c] is not None])), 2)
            for c in ("ukf / ours (A)", "ukf rescan", "ours rescan")],
        "naming (agreement; named only; with a counterpart)": {
            k: [v["agreement"], v["agreement, named tracts only"], v["with a counterpart"]] for k, v in naming.items()},
        "naming by distance band (named only)": {
            k: {b: v["by the nearest voter's distance (mm)"][b]["agreement, named"] for b in v["by the nearest voter's distance (mm)"]}
            for k, v in naming.items()},
        "naming suspects": [n for n, r in out["tracts"].items() if "naming suspect" in r["classes"]],
        "coverage": [n for n, r in out["tracts"].items() if "coverage" in r["classes"]],
        "UKF-unstable": [n for n, r in out["tracts"].items() if "UKF-unstable" in r["classes"]],
        "mean top probability, all streamlines": {r: round(float(top_probability(recs[r]).mean()), 4)
                                                 for r in ("ukf A", "ours A", "dipy A")},
        "seconds": round(time.time() - t0, 1)}
    (HERE / "results" / "naming_check.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
