"""_trx.py's output read back by trx-python (the TRX reference implementation), on PAT16.

    DATA/.venv/bin/python bench/tractography/trx_check.py

Runs the pipeline once, writes the TRX two ways - a .trx zip of every streamline in float32, a
directory of the labeled ones in float16 - and checks each against what the pipeline holds: the
header (affine, grid, counts), every streamline's points (exactly in float32; within float16's
spacing otherwise), the groups (one per tract, partitioning the labeled streamlines), and every dps
array (labels, probabilities, rank field) row for row. Writes results/trx_check.json.
"""
import json, time
from pathlib import Path
import numpy as np
from trx.trx_file_memmap import load as trx_load
import _pipeline as P, _trx
from _ds001226 import load, ROOT
from _tractcloud import Labeler, TRACT_NAMES

HERE = Path(__file__).resolve().parent
s = load("PAT16")
timer = P.Timer(echo="PAT16")
OUT = ROOT / "derived/PAT16"
corr, tg, labels, payload = P.run(s, Labeler(), timer, trx=OUT / "PAT16.trx", encode=True, rank_field=True)
with timer("write_trx_dir_float16_labeled"):
    _trx.write(OUT / "PAT16_labeled_f16_trx", s, tg, labels, positions="float16", labeled_only=True, rank_field=True)

res = {"seconds": timer.seconds, "variants": {}}
keep = labels.keep
for name, path, labeled_only, dt in (("zip, all streamlines, float32", OUT / "PAT16.trx", False, np.float32),
                                     ("directory, labeled only, float16", OUT / "PAT16_labeled_f16_trx", True, np.float16)):
    t0 = time.time(); T = trx_load(str(path)); t_load = time.time() - t0
    which = np.flatnonzero(keep) if labeled_only else np.arange(len(tg.fibers))
    fibers = [tg.fibers[i] for i in which]
    got = list(T.streamlines)
    pts_exact = all(np.array_equal(np.asarray(g), f.astype(dt)) for g, f in zip(got, fibers))
    tol = max(float(np.abs(np.asarray(g, float) - f).max()) for g, f in zip(got, fibers))
    dps = {k: np.asarray(v) for k, v in T.data_per_streamline.items()}
    L = keep[which]
    rows = np.cumsum(keep)[which] - 1
    groups = {k: np.asarray(v) for k, v in T.groups.items()}
    union = np.sort(np.concatenate(list(groups.values())))
    checks = {
        "header_affine": bool(np.allclose(T.header["VOXEL_TO_RASMM"], s.affine)),
        "header_dimensions": list(map(int, T.header["DIMENSIONS"])) == list(s.dwi.shape[:3]),
        "nb_streamlines": int(T.header["NB_STREAMLINES"]) == len(fibers) == len(got),
        "points_exact" if dt == np.float32 else "points_within_float16": pts_exact if dt == np.float32 else tol < 0.0625,
        "groups_partition_labeled": bool(np.array_equal(union, np.flatnonzero(L))) and len(union) == len(np.unique(union)),
        "groups_are_tracts": all(np.all(dps["tract"][v] == TRACT_NAMES.index(k)) for k, v in groups.items()),
        "tract": bool(np.array_equal(dps["tract"][L].ravel(), labels.tract[rows[L]])) and bool(np.all(dps["tract"][~L] == 255)),
        "rf_ranks": bool(np.array_equal(dps["rf_ranks"][L], np.asarray(payload.field[0]).reshape(6, -1)[:, rows[L]].T)),
        "rf_gaps": bool(np.array_equal(dps["rf_gaps"][L], np.asarray(payload.field[1]).reshape(5, -1)[:, rows[L]].T)),
        "rf_tail": bool(np.array_equal(dps["rf_tail"][L].ravel(), np.asarray(payload.field[2]).reshape(-1)[rows[L]])),
        "cluster_maps_to_tract": bool(np.array_equal(_trx.LUT[dps["cluster"][L].ravel().astype(int)], dps["tract"][L].ravel())),
        "probability_in_0_1": bool(np.all((dps["tract_probability"][L] >= 0) & (dps["tract_probability"][L] <= 1.001))),
        "rankfield_meta_in_header": T.header.get("RANKFIELD", {}).get("depth") == 6,
    }
    size = path.stat().st_size if path.is_file() else sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    res["variants"][name] = {"all_checks": all(checks.values()), "checks": checks, "streamlines": len(got), "mb": round(size / 1e6, 2),
                             "max_point_error_mm": round(tol, 4), "trx_python_load_s": round(t_load, 2),
                             "margin_median_labeled": round(float(np.nanmedian(dps["tract_margin"][L].astype(float))), 3)}
    T.close()
res["payload_mb"] = payload.total_mb
print(json.dumps(res, indent=1))
(HERE / "results/trx_check.json").write_text(json.dumps(res, indent=1))
