"""The pipeline's tractogram as TRX (github.com/tee-ar-ex/trx-spec), the output other tools read: a zip (stored, not deflated, so it can be memory-mapped) or a directory.

  header.json      VOXEL_TO_RASMM and DIMENSIONS of the DWI the tracts came from; NB_STREAMLINES,
                   NB_VERTICES
  positions        RAS mm, float32 (the tracker's points), or float16 (TRX's suggestion: error up to
                   0.03 mm at 100 mm from the origin)
  offsets.uint64   NB_STREAMLINES + 1, the last NB_VERTICES
  groups/<tract>   the streamlines the labeler names each tract (one draw: the pipeline's), "Other" too
  dps/             per streamline:
                     tract.uint8               0-41 a tract (TRACT_NAMES order), 42 Other, 255 not labeled
                     cluster.uint16            the labeler's most probable of the 1600 clusters; 65535 not labeled
                     tract_probability.float16 the named tract's probability: its clusters' summed
                     tract_margin.float16      that, less the most probable other tract's (negative when the
                                               summed probabilities favor another tract than the top cluster's)
                     length_mm.float32, seed.uint32 (the seed's index, the binary's order)

Every streamline the tracker kept is written, those under 40 mm unlabeled (the labelers were trained on
>= 40 mm): the 40 mm cut becomes a filter a reader applies, not a deletion. labeled_only=True writes
the labeled streamlines alone.
"""
from __future__ import annotations

import json, shutil, tempfile, zipfile
from pathlib import Path
import numpy as np, torch
from .labelers.base import TRACT_NAMES, LUT

UNLABELED_TRACT, UNLABELED_CLUSTER = 255, 65535


def tract_probabilities(logp):
    """(n, 1600) cluster log-probabilities -> (n, 43) tract probabilities (clusters summed by tract)."""
    p = torch.exp(torch.as_tensor(logp).float())
    onehot = torch.zeros(1600, 43); onehot[torch.arange(1600), torch.as_tensor(LUT.astype(np.int64))] = 1
    return (p @ onehot).numpy()


def write(path, s, tg, labels, positions="float32", labeled_only=False):
    """path ending in .trx: a zip; otherwise a directory. s: the Subject (the DWI's grid); tg: the
    Tractogram; labels: its Labels (with logp). Returns the path."""
    if labels.logp is None:
        raise ValueError("trx.write needs the labels' cluster log-probabilities: label with logp=True (pipeline.run does)")
    path = Path(path)
    keep = labels.keep
    which = np.flatnonzero(keep) if labeled_only else np.arange(len(tg.fibers))
    fibers = [tg.fibers[i] for i in which]
    n, lab_rows = len(fibers), np.full(len(tg.fibers), -1)
    lab_rows[np.flatnonzero(keep)] = np.arange(int(keep.sum()))            # fiber -> row among the labeled
    rows = lab_rows[which]; L = rows >= 0

    tract = np.full(n, UNLABELED_TRACT, np.uint8); tract[L] = labels.tract[rows[L]]
    P = tract_probabilities(labels.logp)
    cluster = np.full(n, UNLABELED_CLUSTER, np.uint16); cluster[L] = labels.logp.float().argmax(1).numpy()[rows[L]]
    prob = np.full(n, np.nan, np.float16); margin = np.full(n, np.nan, np.float16)
    pt = P[rows[L], labels.tract[rows[L]]]
    other = P[rows[L]].copy(); other[np.arange(L.sum()), labels.tract[rows[L]]] = -np.inf
    prob[L] = pt; margin[L] = pt - other.max(1)
    lens = np.array([len(f) for f in fibers])
    length = np.array([np.linalg.norm(np.diff(f, axis=0), axis=1).sum() for f in fibers], np.float32)
    dps = {"tract.uint8": tract, "cluster.uint16": cluster, "tract_probability.float16": prob, "tract_margin.float16": margin,
           "length_mm.float32": length, "seed.uint32": np.asarray(tg.stats["seed_index"], np.uint32)[which]}

    pos = np.concatenate(fibers).astype(positions)
    header = {"VOXEL_TO_RASMM": np.asarray(s.affine, float).tolist(), "DIMENSIONS": [int(v) for v in s.dwi.shape[:3]],
              "NB_STREAMLINES": int(n), "NB_VERTICES": int(len(pos))}
    with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
        d = Path(tmp) / "trx"
        (d / "dps").mkdir(parents=True); (d / "groups").mkdir()
        (d / "header.json").write_text(json.dumps(header))
        pos.tofile(d / f"positions.3.{pos.dtype.name}")
        np.r_[0, np.cumsum(lens)].astype(np.uint64).tofile(d / "offsets.uint64")
        for k, a in dps.items():
            np.ascontiguousarray(a).tofile(d / "dps" / k)
        for t, name in enumerate(TRACT_NAMES):
            idx = np.flatnonzero(tract == t).astype(np.uint32)
            if len(idx):
                idx.tofile(d / "groups" / f"{name}.uint32")
        if path.suffix == ".trx":
            with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
                for f in sorted(d.rglob("*")):
                    if f.is_file():
                        z.write(f, f.relative_to(d).as_posix())
        else:
            if path.exists() and (not path.is_dir() or (any(path.iterdir()) and not (path / "header.json").exists())):
                raise FileExistsError(f"{path} exists and is not a TRX directory: not overwriting it")
            shutil.rmtree(path, ignore_errors=True); shutil.copytree(d, path)
    return path
