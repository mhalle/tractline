"""Is the Vorob'ev expectation a better "core" than the majority vote? (NOTES 2026-10-13, "A volume-matched core").

    uv run --isolated --python 3.12 --with "zarr>=3.3" --with numpy bench/ensemble_vorobev.py

The ensemble store (ensemble_store.py: cIIIs04's four scans, 17 runs each, presence by streamline density on
FastSurfer's conformed 1 mm grid). Per tract and voxel, p = the share of runs with the tract present. Shapes compared,
each a set {p >= t}:
  - core: at least half the runs (the viewer's core);
  - Vorob'ev expectation: t set so the shape's volume matches a run's mean tract volume (sum of p); among all sets of
    that volume it minimizes the expected symmetric difference with a run drawn from the ensemble (Molchanov, Theory
    of Random Sets; cited from memory). The vote count whose set's volume is closest to the mean is used;
  - the 80 % and 90 % envelopes (ensemble_pooled.thresholds), and the widest shell (any run).
Each scan is held out in turn and its own run is the target:
  - single: each other scan's 17-run ensemble predicts it (averaged over the three);
  - pooled: the other three scans' 51 runs.
Against the target T, for a shape S: coverage |S n T| / |T|, precision |S n T| / |S|, Dice, size |S| / |T|, and
symmetric difference |S xor T| / |T|. Reported for the focus tracts (CST, AF, UF) and as medians over the named
tracts whose target has at least MIN_VOXELS voxels. Writes results/ensemble_vorobev.json.
"""
import json, sys, zipfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = Path.home() / "tmp/data/tractography"
STORE = DATA / "ensemble" / "cIIIs04" / "cIIIs04_ensemble.duckn.zip"
FOCUS = ("CST", "AF", "UF")
LEVELS = (0.8, 0.9)
MIN_VOXELS = 2000
RUNS = 17


def thresholds(q, R, levels):
    """ensemble_pooled.thresholds: per stated level, the lowest vote count its envelope keeps."""
    counts = np.bincount(q, minlength=R + 1).astype(float)
    mass = counts * np.arange(R + 1)
    mass[0] = 0
    cum = np.cumsum(mass[::-1])[::-1]
    return [int(np.flatnonzero(cum >= L * cum[1]).max()) for L in levels]


def shapes(v, R):
    """vote counts (0..R) -> {shape: lowest vote count kept}"""
    counts = np.bincount(v, minlength=R + 1)
    vol = np.cumsum(counts[::-1])[::-1]                                     # voxels with at least t votes
    mean_run = (counts * np.arange(R + 1)).sum() / R
    vor = int(np.argmin(np.abs(vol[1:] - mean_run))) + 1
    env = thresholds(v, R, LEVELS)
    return {"core": int(np.ceil(R / 2)), "vorobev": vor, "envelope 80": env[0], "envelope 90": env[1], "widest": 1}


def score(v, t, target):
    S = v >= t
    n_t, n_s, both = int(target.sum()), int(S.sum()), int((S & target).sum())
    return {"coverage": both / n_t, "precision": both / max(n_s, 1), "dice": 2 * both / (n_s + n_t),
            "size": n_s / n_t, "symmetric difference": (n_s + n_t - 2 * both) / n_t}


def main():
    import zarr
    z = zarr.open_group(zarr.storage.ZipStore(str(STORE), mode="r"), mode="r")
    root = z.attrs["duckn"]["extensions"]["tractline"]
    classes = root["classes"]
    scans = z["ensembles"].attrs["duckn"]["extensions"]["tractline"]["scans"]
    named = [c for c in classes if c != "Other"]
    rows = {"single": {}, "pooled": {}}                                      # [mode][tract] -> list of per-held-out scores
    cuts = {"single": {}, "pooled": {}}
    for name in named:
        k = classes.index(name)
        votes = {s: z[f"ensembles/{s}/votes"][k].ravel() for s in scans}
        runs = {s: z[f"ensembles/{s}/run"][k].ravel().astype(bool) for s in scans}
        for h in scans:
            target = runs[h]
            if target.sum() < MIN_VOXELS:
                continue
            others = [s for s in scans if s != h]
            singles = []
            for s in others:
                sh = shapes(votes[s], RUNS)
                singles.append({k2: score(votes[s], t, target) for k2, t in sh.items()})
                cuts["single"].setdefault(name, []).append(sh)
            rows["single"].setdefault(name, []).append(
                {k2: {m: float(np.mean([x[k2][m] for x in singles])) for m in singles[0][k2]} for k2 in singles[0]})
            pv = sum(votes[s].astype(np.int32) for s in others)
            sh = shapes(pv, RUNS * len(others))
            cuts["pooled"].setdefault(name, []).append(sh)
            rows["pooled"].setdefault(name, []).append({k2: score(pv, t, target) for k2, t in sh.items()})
        print(name, flush=True)
    res = {"store": str(STORE.relative_to(DATA)), "scans": scans, "min target voxels": MIN_VOXELS, "focus": {}, "all named tracts (median)": {}}
    metrics = ("coverage", "precision", "dice", "size", "symmetric difference")
    for mode in ("single", "pooled"):
        R = RUNS if mode == "single" else RUNS * 3
        for name in FOCUS:
            r = rows[mode][name]
            res["focus"][f"{mode}, {name}"] = {
                "runs needed": {k2: float(np.median([c[k2] for c in cuts[mode][name]])) for k2 in cuts[mode][name][0]},
                **{k2: {m: round(float(np.mean([x[k2][m] for x in r])), 3) for m in metrics} for k2 in r[0]}}
        per_tract = {name: {k2: {m: np.mean([x[k2][m] for x in r]) for m in metrics} for k2 in r[0]} for name, r in rows[mode].items()}
        shape_names = list(next(iter(per_tract.values())))
        res["all named tracts (median)"][mode] = {
            "tracts": len(per_tract), "runs": R,
            **{k2: {m: round(float(np.median([t[k2][m] for t in per_tract.values()])), 3) for m in metrics} for k2 in shape_names},
            "Vorob'ev beats core on symmetric difference (tracts)": int(sum(t["vorobev"]["symmetric difference"] < t["core"]["symmetric difference"] for t in per_tract.values())),
            "Vorob'ev beats core on Dice (tracts)": int(sum(t["vorobev"]["dice"] > t["core"]["dice"] for t in per_tract.values()))}
    (HERE / "results/ensemble_vorobev.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
