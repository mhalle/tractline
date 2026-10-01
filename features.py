"""Read the HCP subject once: TractCloud's 15-point RAS features for every streamline, and the
seeded 100 k subsample of target streamlines that M0 and M1 run on.

    DATA/.venv/bin/python bench/tractography/features.py

Writes DATA/hcp/feat.npy ((N, 15, 3) float64, upstream extract_ras_features unchanged) and
DATA/hcp/targets.npy (sorted int64 indices into it). Features are float64 because upstream
recenters in float64 and casts to float32 only inside RealDataDataset.
"""
import json, time
import numpy as np
from _data import HCP, HCP_VTP, SUBSAMPLE, SUBSAMPLE_SEED
from tractcloud.inference import extract_ras_features
from tractcloud.vtk_io import read_polydata

t0 = time.time()
pd = read_polydata(str(HCP_VTP))
n_lines, n_points = pd.GetNumberOfLines(), pd.GetNumberOfPoints()
t1 = time.time()
feat = extract_ras_features(pd, num_points=15)
t2 = time.time()
np.save(HCP / "feat.npy", feat)

targets = np.sort(np.random.default_rng(SUBSAMPLE_SEED).choice(len(feat), SUBSAMPLE, replace=False))
np.save(HCP / "targets.npy", targets)

info = {"streamlines": n_lines, "points": n_points, "feat_shape": list(feat.shape),
        "ras_min": feat.reshape(-1, 3).min(0).round(1).tolist(),
        "ras_max": feat.reshape(-1, 3).max(0).round(1).tolist(),
        "targets": len(targets), "subsample_seed": SUBSAMPLE_SEED,
        "read_s": round(t1 - t0, 1), "extract_s": round(t2 - t1, 1)}
(HCP / "features.json").write_text(json.dumps(info, indent=1))
print(json.dumps(info, indent=1))
