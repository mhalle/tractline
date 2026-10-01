"""M3 step 2: the web export - static files a browser viewer loads progressively.

    modal volume get tractography-bench hcp_full/derived.npz DATA/hcp_full/derived.npz
    DATA/.venv/bin/python bench/tractography/export.py [--depth 6] [--points 20] [--chunks 50]
    node bench/tractography/export_check.mjs DATA/hcp/web

Writes DATA/hcp/web/ (outside the repo; HCP data, not for public hosting until the data-use
terms are checked). Geometry and field are separate files, so the viewer can regroup without
reloading geometry. All binaries are little-endian, every array starts 2- or 4-byte aligned.

Order. Streamlines are shuffled once (EXPORT_SEED) and split into `--chunks` chunks, so the
first n chunks are a uniform random sample of the brain. The same order runs through every
file of a chunk; extra/ carries each line's original index.

    geom/gNNN.bin   u32 lines, u32 pointsPerLine, i16 xyz[lines * pointsPerLine * 3]
                    TractCloud's arc-length resampler at `--points`, ORIGINAL RAS mm (not the
                    model's recentered coordinates): xyz = center + q * quantum
    field/fNNN.bin  u32 lines, u32 depth, u16 ranks[depth][lines] (plane-major, class + 1,
                    0 = absent), u16 tail[lines] (dropped mass / 65535), u8 support[depth-1][lines]
                    (byte -> gap through levels.bin; 0 = absent). rankfield keep="clip", run 0.
    extra/eNNN.bin  u32 lines, u32 0, u32 index[lines], u8 tractChanges[lines],
                    u8 foldChanges[lines], u8 outlierChanges[lines], u8 marginSpread[lines]
                    (changes = how many of runs 1-4 differ from run 0: tract, tract with outliers
                    folded in, outlier status; spread = std over the 5 runs of the best-class
                    tract margin, in 0.05-logit units, saturating at 255)
    levels.bin      256 float32: rankfield's level table, byte -> gap in logits, from the meta
    tables.json     class -> tract, tract names / full names / categories / colors, clip
    summary/        m1, m1b, m2 JSON and figures, and this export's own numbers
    manifest.json   the chunk list with byte counts and sha256, center, quantum, the rankfield
                    meta block, and the file names above
"""
import argparse, hashlib, json, shutil, struct, time
from pathlib import Path
import numpy as np
import rankfield as rf
from _data import DATA, HCP, HCP_VTP, SEEDS
from tractcloud.colors import get_tract_color
from tractcloud.inference import extract_ras_features
from tractcloud.tract_mapping import TRACT_CATEGORIES, TRACT_FULL_NAMES, TRACT_NAMES, _CLUSTER_TO_TRACT_LUT as TRACT
from tractcloud.vtk_io import read_polydata

EXPORT_SEED = 20261002
QUANTUM = 0.01                                        # mm per int16 step
RESULTS = Path(__file__).resolve().parent / "results"

ap = argparse.ArgumentParser()
ap.add_argument("--depth", type=int, default=6)
ap.add_argument("--points", type=int, default=20)
ap.add_argument("--chunks", type=int, default=50)
args = ap.parse_args()
WEB = HCP / "web"
CHECK = HCP / "web_check"                             # expected values for export_check.mjs
for d in (WEB, CHECK):
    shutil.rmtree(d, ignore_errors=True)
for sub in ("geom", "field", "extra", "summary"):
    (WEB / sub).mkdir(parents=True)
CHECK.mkdir()

t0 = time.time()
der = np.load(DATA / "hcp_full" / "derived.npz")
meta = json.loads(str(der[f"rf_d{args.depth}_meta"]))
ranks, support, tail = (der[f"rf_d{args.depth}_{k}"] for k in ("ranks", "support", "tail"))
cluster = der["cluster"].astype(np.int64)
R, N = cluster.shape
assert ranks.shape == (args.depth, N) and (ranks[0].astype(np.int64) - 1 == cluster[0]).all()

pd = read_polydata(str(HCP_VTP))
assert pd.GetNumberOfLines() == N
xyz = extract_ras_features(pd, num_points=args.points)          # (N, P, 3) float64, original RAS
lo, hi = xyz.reshape(-1, 3).min(0), xyz.reshape(-1, 3).max(0)
center = np.round((lo + hi) / 2, 2)
q = np.round((xyz - center) / QUANTUM)
assert np.abs(q).max() < 32767, "geometry does not fit int16 at this quantum"
q = q.astype("<i2")
t1 = time.time()

# per-streamline extras from the five runs
tract = TRACT[cluster]
fold_lut = TRACT.copy(); fold_lut[800:] = TRACT[:800]
fold = fold_lut[cluster]
outlier = cluster >= 800
changes = lambda a: (a[1:] != a[0]).sum(0).astype(np.uint8)
spread = np.clip(np.round(der["m_best_tract"].std(0) / 0.05), 0, 255).astype(np.uint8)
extras = {"tractChanges": changes(tract), "foldChanges": changes(fold),
          "outlierChanges": changes(outlier), "marginSpread": spread}

perm = np.random.default_rng(EXPORT_SEED).permutation(N)
manifest_chunks = []
sha = lambda b: hashlib.sha256(b).hexdigest()
written = {}


def write(rel, payload):
    (WEB / rel).write_bytes(payload)
    written[rel] = len(payload)
    return {"file": rel, "bytes": len(payload), "sha256": sha(payload)}


for c, idx in enumerate(np.array_split(perm, args.chunks)):
    n = len(idx)
    geom = struct.pack("<II", n, args.points) + q[idx].tobytes()
    field = (struct.pack("<II", n, args.depth) + np.ascontiguousarray(ranks[:, idx]).astype("<u2").tobytes()
             + tail[idx].astype("<u2").tobytes() + np.ascontiguousarray(support[:, idx]).astype("u1").tobytes())
    extra = (struct.pack("<II", n, 0) + idx.astype("<u4").tobytes()
             + b"".join(extras[k][idx].tobytes() for k in ("tractChanges", "foldChanges", "outlierChanges", "marginSpread")))
    manifest_chunks.append({"lines": n, "geom": write(f"geom/g{c:03d}.bin", geom),
                            "field": write(f"field/f{c:03d}.bin", field),
                            "extra": write(f"extra/e{c:03d}.bin", extra)})

# expected values for the JS check: the own-tract best-class margin, from rankfield itself,
# for chunk 0 in export order, as float32
idx0 = np.array_split(perm, args.chunks)[0]
code0 = rf.RankField(ranks=ranks[:, idx0].reshape(args.depth, -1, 1, 1), support=support[:, idx0].reshape(args.depth - 1, -1, 1, 1),
                     tail=tail[idx0].reshape(-1, 1, 1), meta={**meta, "shape": [len(idx0), 1, 1]})
groups = [np.flatnonzero(TRACT == t) for t in range(len(TRACT_NAMES))]
dg = rf.decode_groups(code0, groups).reshape(len(groups), -1).numpy()
own = TRACT[ranks[0, idx0].astype(np.int64) - 1]
(CHECK / "chunk0_tract_margin.f32").write_bytes(dg[own, np.arange(len(idx0))].astype("<f4").tobytes())
(CHECK / "chunk0_tract.u8").write_bytes(own.astype(np.uint8).tobytes())
# ...and its mass margin, as encode.py computes it: rankfield.probabilities, the tail counted
# against the tract, floored at half a tail quantum (float64)
ids, p = rf.probabilities(code0)
ids, p = ids.reshape(args.depth, -1), p.reshape(args.depth, -1).astype(np.float64)
kept = ids >= 0
in_s = kept & (TRACT[np.clip(ids, 0, None)] == own)
p_out = (p * (kept & ~in_s)).sum(0) + tail[idx0].astype(np.float64) / rf.TAIL_MAX
mass = np.log((p * in_s).sum(0)) - np.log(np.maximum(p_out, 0.5 / rf.TAIL_MAX))
(CHECK / "chunk0_tract_mass.f64").write_bytes(mass.astype("<f8").tobytes())

levels = rf.levels(meta).astype("<f4")
written["levels.bin"] = len(levels.tobytes())
(WEB / "levels.bin").write_bytes(levels.tobytes())
cat_of = {t: c for c, ts in TRACT_CATEGORIES.items() for t in ts}
tables = {"classes": 1600, "clip": meta["clip"],
          "clusterToTract": TRACT.astype(int).tolist(),
          "outlierFrom": 800, "otherTract": TRACT_NAMES.index("Other"),
          "tracts": [{"name": t, "fullName": TRACT_FULL_NAMES.get(t, t), "category": cat_of[t],
                      "color": [round(v, 3) for v in get_tract_color(i + 1)]} for i, t in enumerate(TRACT_NAMES)],
          "categories": list(TRACT_CATEGORIES)}
(WEB / "tables.json").write_text(json.dumps(tables))
for f in ("m1.json", "m1b.json", "m2.json", "m1_instability.png", "m1_pairs.png", "m1b_regroup.png"):
    shutil.copy(RESULTS / f, WEB / "summary" / f)

geom_b = sum(ch["geom"]["bytes"] for ch in manifest_chunks)
field_b = sum(ch["field"]["bytes"] for ch in manifest_chunks)
extra_b = sum(ch["extra"]["bytes"] for ch in manifest_chunks)
export_summary = {"streamlines": N, "chunks": args.chunks, "pointsPerLine": args.points, "depth": args.depth,
                  "bytes": {"geometry": geom_b, "field": field_b, "extras": extra_b},
                  "field_bytes_per_streamline": round(field_b / N, 2),
                  "dense_fp16_bytes": N * 1600 * 2,
                  "seconds": {"read_and_resample": round(t1 - t0, 1), "total": round(time.time() - t0, 1)}}
(WEB / "summary" / "export.json").write_text(json.dumps(export_summary, indent=1))
manifest = {
    "format": "tractcloud-rankfield-web", "version": 1,
    "subject": "HCP 101006 (TractCloud TestData v1.0.0), UKF tractography",
    "streamlines": N, "pointsPerLine": args.points, "chunkCount": args.chunks,
    "coordinates": {"space": "RAS", "unit": "mm", "note": "original subject RAS, not the model's recentered input",
                    "center": center.tolist(), "quantum": QUANTUM, "bounds": [lo.round(2).tolist(), hi.round(2).tolist()]},
    "field": {"run": 0, "seed": SEEDS[0], "rankfield": meta, "levels": "levels.bin",
              "layout": "u32 lines, u32 depth, u16 ranks[depth][lines], u16 tail[lines], u8 support[depth-1][lines]"},
    "geometryLayout": "u32 lines, u32 pointsPerLine, i16 xyz[lines][pointsPerLine][3]",
    "extraLayout": "u32 lines, u32 0, u32 index[lines], u8 tractChanges, u8 foldChanges, u8 outlierChanges, u8 marginSpread (each [lines])",
    "runs": {"count": R, "seeds": list(SEEDS), "spreadUnit": 0.05},
    "tables": "tables.json",
    "summary": {f: f"summary/{f}" for f in ("m1.json", "m1b.json", "m2.json", "export.json",
                                              "m1_instability.png", "m1_pairs.png", "m1b_regroup.png")},
    "chunks": manifest_chunks,
}
(WEB / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps(export_summary, indent=1))
