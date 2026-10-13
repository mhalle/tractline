"""cIIIs04's ensemble presence as a duckn store: zarr v3, sharded, few blocks, zipped (NOTES 2026-10-13, "The ensemble
as a duckn store").

    uv run --group dipy --group rankfield --group store bench/ensemble_store.py [--out PATH]

Presence on FastSurfer's conformed grid (its native 256^3, 1 mm lattice, where its ranked anatomy lives; the
canonical per-subject grid), cropped to the brain with the crop's offset recorded so the anatomy's arrays slice to
the same voxels, by streamline
density (ensemble_grid_check.py: streamlines per mm^2 of a voxel's mean cross-section, smoothed 1 mm, at least 0.3),
which keeps the calibration across grids. For each of the four scans (A and the same-session, session 3 and session 5
rescans), its 17-run ensemble's votes per class and voxel (how many runs have the class present), and its own run's
presence; and the four ensembles pooled (68 runs). Votes are computed once and cached beside the records
(presence_t1.npz).

The store, following haversack's ranked stores (haversack.ranked_store): a zarr v3 hierarchy built in a staging
directory and packed into a zip with every entry stored (the chunks are already compressed); arrays (class, z, y, x),
one chunk per class in one shard per array, blosc zstd with bitshuffle; geometry through duckn's own models (LPS mm,
cell centering, a list axis for the class); this project's facts in a versioned `tractline` extension block: classes,
runs, the presence definition, per class and array the vote counts at which the envelopes of the stated levels start,
the coverage those levels held on held-out scans (ensemble_grid_check.json), the T1-to-DWI transform, and the
FastSurfer store it shares a frame with.
"""
import argparse, json, os, shutil, sys, tempfile, time, zipfile
from pathlib import Path
import numpy as np
from tractline import ensemble as E
from tractline.data import DATA
from tractline.labelers.base import LUT, TRACT_NAMES

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ensemble_retest import PAIRS  # noqa: E402
from ensemble_grid_check import Grid, sample, STEP_MM, SMOOTH_MM, RHO  # noqa: E402
from ensemble_pooled import thresholds, LEVELS  # noqa: E402
from ensemble_endpoints import Anatomy, FS  # noqa: E402

RUNS = 17
DUCKN_VERSION = "1.2"                     # haversack.ranked_store's
EXTENSION_VERSION = "0.1"
LPS = np.diag([-1.0, -1.0, 1.0, 1.0])
L = np.asarray(LUT)
K = len(TRACT_NAMES)


MARGIN = 5


def grid_conformed():
    """FastSurfer's conformed lattice, cropped to the brain (its labels' extent plus MARGIN voxels): the grid, the crop's
    start and stop in the conformed array's (i, j, k), the full conformed shape, and the T1-to-DWI transform."""
    import nibabel as nib
    conf = nib.load(FS / "ranked_cIIIs04_brain" / "conformed_1mm.nii.gz")
    anat = Anatomy("cIIIs04")                                            # labels on the original T1 grid, the same world
    w = np.argwhere(anat.lab > 0)[::7] @ anat.affine[:3, :3].T + anat.affine[:3, 3]
    ijk = w @ np.linalg.inv(conf.affine)[:3, :3].T + np.linalg.inv(conf.affine)[:3, 3]
    lo = np.maximum(np.floor(ijk.min(0)).astype(int) - MARGIN, 0)
    hi = np.minimum(np.ceil(ijk.max(0)).astype(int) + MARGIN + 1, conf.shape[:3])
    crop = conf.affine.copy()
    crop[:3, 3] = conf.affine[:3, :3] @ lo + conf.affine[:3, 3]
    t1_to_dwi = np.load(FS / "cIIIs04_t1_to_dwi.npz")["t1_to_dwi"]
    return Grid("conformed", crop, hi - lo, np.linalg.inv(t1_to_dwi)), (lo, hi, conf.shape[:3]), t1_to_dwi


def compute(d, rescans, g, cache):
    """per scan: votes (class, i, j, k) and its own run's presence; and the mean density per class over A's runs and
    over all runs (float16, for the rank fields)"""
    if cache.exists():
        z = np.load(cache)
        if "dens_A" in z:
            return ({s: z[f"votes_{s}"] for s in z["scans"]}, {s: z[f"run_{s}"] for s in z["scans"]},
                    {"A": z["dens_A"], "pooled": z["dens_pooled"]})
    scans = {"A": [d / "A.npz"] + [d / "smooth" / f"r{r:03d}.npz" for r in range(1, RUNS)]}
    for tag in rescans:
        scans[tag] = [d / f"{tag}_onA.npz"] + [d / "smooth" / f"{tag}_r{r:03d}.npz" for r in range(1, RUNS)]
    votes, run0 = {}, {}
    dA = np.zeros((K,) + g.shape, np.float32)
    dP = np.zeros((K,) + g.shape, np.float32)
    t0 = time.time()
    for s, paths in scans.items():
        v = np.zeros((K,) + g.shape, np.uint8)
        for i, p in enumerate(paths):
            rec = E.RunRecord.load(p, logp=False, fibers=True)
            tract = L[rec.cluster]
            for k in range(K):
                idx = np.flatnonzero(tract == k)
                if len(idx):
                    dens = g.density(g.counts(*sample([rec.fibers[j] for j in idx])))
                    pres = dens >= RHO
                    v[k] += pres
                    dP[k] += dens
                    if s == "A":
                        dA[k] += dens
                    if i == 0:
                        run0.setdefault(s, np.zeros((K,) + g.shape, bool))[k] = pres
            del rec
        votes[s] = v
        print(f"{s}: {time.time() - t0:.0f}s", flush=True)
    dens = {"A": (dA / RUNS).astype(np.float16), "pooled": (dP / (RUNS * len(scans))).astype(np.float16)}
    np.savez_compressed(cache, scans=np.array(list(scans)), **{f"votes_{s}": v for s, v in votes.items()},
                        **{f"run_{s}": r for s, r in run0.items()}, dens_A=dens["A"], dens_pooled=dens["pooled"])
    return votes, run0, dens


def array_attrs(g, extension=None, list_axis=True):
    """duckn geometry for a (class, z, y, x) array on grid g, through duckn's models (as haversack.ranked_store)."""
    from duckn import DucknMetadata, AxisMetadata
    from duckn.models import duckn_attrs
    A = LPS @ g.affine
    spacing_xyz = np.sqrt((A[:3, :3] ** 2).sum(0))
    cols = [A[:3, 2], A[:3, 1], A[:3, 0]]                                 # array axes z, y, x
    axes = ([AxisMetadata(kind="list")] if list_axis else []) + [
        AxisMetadata(kind="space", centering="cell", unit="mm", space_direction=[round(float(v), 9) for v in c]) for c in cols]
    meta = DucknMetadata(version=DUCKN_VERSION, space="left-posterior-superior",
                         space_origin=[round(float(v), 6) for v in A[:3, 3]], axes=axes,
                         extensions={"tractline": {"version": EXTENSION_VERSION, **extension}} if extension else None)
    assert np.allclose(spacing_xyz, g.h, atol=1e-6)
    return duckn_attrs(meta)


def group_attrs(extension, **others):
    """a group's duckn object: our `tractline` block, and any other versioned extension blocks (rankfield's `ranked`)"""
    from duckn import DucknGroupMetadata
    meta = DucknGroupMetadata(version=DUCKN_VERSION,
                              extensions={"tractline": {"version": EXTENSION_VERSION, **extension}, **others})
    return {"duckn": meta.model_dump(exclude_none=True)}


def pack(src_dir: Path, archive: Path):
    partial = archive.with_name(archive.name + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        for f in sorted(p for p in src_dir.rglob("*") if p.is_file()):
            zf.write(f, f.relative_to(src_dir).as_posix())
    os.replace(partial, archive)


def subset(src: Path, dst: Path, keep, note=""):
    """A smaller store holding only the nodes under `keep` and every group above them, entries copied byte for byte
    (no recompression), with a README naming what was kept. For publishing beside a viewer, which reads it whole."""
    keep = [k.strip("/") for k in keep]

    def wanted(n):
        if any(n.startswith(k + "/") for k in keep):
            return True
        if n == "zarr.json" or n.endswith("/zarr.json"):
            g = n[: -len("zarr.json")].rstrip("/")
            return g == "" or any(k.startswith(g + "/") for k in keep)
        return False
    partial = dst.with_name(dst.name + ".partial")
    with zipfile.ZipFile(src) as zs, zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zd:
        names = [n for n in zs.namelist() if wanted(n)]
        for k in keep:
            assert f"{k}/zarr.json" in names, f"{src.name} has no node {k}"
        for n in names:
            zd.writestr(zs.getinfo(n), zs.read(n))
        zd.writestr("README.md", f"# A subset of {src.name}\n\nEntries copied byte for byte from the full store; only these "
                    f"nodes (and the groups above them) are here:\n\n" + "".join(f"- {k}\n" for k in keep) + (f"\n{note}\n" if note else ""))
    os.replace(partial, dst)
    return dst


def main(args):
    import zarr
    from zarr.codecs import BloscCodec
    from _masivar import scan_files
    import nibabel as nib
    pair = "cIIIs04-2.1"
    subject, (ses, scan), rescans, shells, _ = PAIRS[pair]
    d = DATA / "ensemble" / subject / "retest" / pair
    g, (lo, hi, conf_shape), t1_to_dwi = grid_conformed()
    votes, run0, dens = compute(d, rescans, g, d / "presence_conformed.npz")
    pooled = sum(v.astype(np.uint16) for v in votes.values()).astype(np.uint8)
    out = Path(args.out or DATA / "ensemble" / subject / f"{subject}_ensemble.duckn.zip")
    validation = json.loads((HERE / "results/ensemble_grid_check.json").read_text())["coverage [widest, *levels], mean over held-out scans"]
    names = list(TRACT_NAMES)
    codec = BloscCodec(cname="zstd", clevel=args.level, shuffle="bitshuffle")
    staging = Path(tempfile.mkdtemp(prefix="ensemble_store_"))
    try:
        root = zarr.open_group(zarr.storage.LocalStore(str(staging)), mode="w")
        root.attrs.update(group_attrs({
            "kind": "ensemble presence", "subject": subject, "pair": pair, "classes": names,
            "presence": {"definition": "streamline density", "rho_per_mm2": RHO, "smooth_mm": SMOOTH_MM, "sample_step_mm": STEP_MM,
                         "cross_section": "1.5 h^2 (the mean projected area of a voxel of side h)"},
            "grid": {"lattice": "FastSurfer's conformed grid (256^3, 1 mm), cropped to the brain",
                     "crop_ijk_start": [int(v) for v in lo], "crop_ijk_stop": [int(v) for v in hi],
                     "conformed_shape_ijk": [int(v) for v in conf_shape],
                     "note": "the crop's (i, j, k) are the conformed NIfTI's array indices; these arrays are stored (z, y, x) = (k, j, i)"},
            "t1_to_dwi_ras_mm": np.round(t1_to_dwi, 9).tolist(),
            "dwi_affine_ras": np.round(nib.load(scan_files(subject, ses, scan, shells)["APP"][0][1]).affine, 9).tolist(),
            "anatomy": {"store": os.path.relpath(FS / "cIIIs04_asegdkt.duckn.zip", out.parent), "frame": "the same T1 world"},
            "levels": list(LEVELS),
            "validation (coverage on held-out scans, [widest, *levels]; ensemble_grid_check.json)":
                {k.replace("t1 grid, density, ", ""): v for k, v in validation.items() if k.startswith("t1 grid, density")},
            "arrays": {"ensembles/<scan>/votes": "runs (of the scan's 17) with the class present",
                       "ensembles/<scan>/run": "the scan's own run (1: present)",
                       "pooled/votes": "runs (of all 68, four scans) with the class present"}}))
        shape = (K,) + tuple(int(s) for s in g.shape[::-1])

        def write(group, name, data_kxyz, runs, ext):
            data = np.ascontiguousarray(np.transpose(data_kxyz, (0, 3, 2, 1)))     # (class, z, y, x)
            th = {names[k]: thresholds(data[k].ravel().astype(int), runs, LEVELS) for k in range(K)} if runs else None
            a = group.create_array(name, shape=shape, dtype=np.uint8, chunks=(1,) + shape[1:], shards=shape,
                                   compressors=codec, fill_value=0,
                                   attributes=array_attrs(g, {**ext, **({"runs": runs, "envelope_votes": th} if runs else {})}))
            a[...] = data

        ens = root.create_group("ensembles")
        ens.attrs.update(group_attrs({"kind": "per-scan ensembles", "scans": list(votes)}))
        for s, v in votes.items():
            grp = ens.create_group(s)
            grp.attrs.update(group_attrs({"kind": "one scan's ensemble", "scan": s, "runs": RUNS}))
            write(grp, "votes", v, RUNS, {"scan": s, "values": "runs with the class present"})
            write(grp, "run", run0[s].astype(np.uint8), None, {"scan": s, "values": "the scan's own run: 1 present"})
        pg = root.create_group("pooled")
        pg.attrs.update(group_attrs({"kind": "all scans' runs pooled", "scans": list(votes), "runs": RUNS * len(votes)}))
        write(pg, "votes", pooled, RUNS * len(votes),
              {"scans": list(votes), "values": "runs with the class present, all scans"})
        # the rank fields: per voxel the classes ranked by mean streamline density, with "none" (a pseudo-class at the
        # presence threshold) - encoded by rankfield, as FastSurfer's ranked anatomy is
        import rankfield as rf
        import torch
        rg = root.create_group("rankfield")
        rg.attrs.update(group_attrs({"kind": "rank fields of mean streamline density", "classes": names + ["none"],
                                     "none": f"a pseudo-class of density {RHO} per mm^2 (the presence threshold)",
                                     "floor": f"densities below {args.floor} of the threshold are absent"}))
        checks = {}
        for which, m in dens.items():
            m = np.transpose(m.astype(np.float32), (0, 3, 2, 1))                  # (class, z, y, x)
            m = np.where(m >= args.floor * RHO, m, 0)                              # faint smoothing tails: absent
            logits = np.concatenate([np.log(m + 1e-12), np.full((1,) + m.shape[1:], np.log(RHO), np.float32)])
            code = rf.encode(torch.from_numpy(logits), depth=args.depth)
            ids, _ = rf.probabilities(code, 1.0)
            ids = np.asarray(ids)
            top = ids[0].reshape(m.shape[1:])
            want = logits.argmax(0)
            brain = want != K                                                       # voxels whose best class is not "none"
            checks[which] = {"rank 1 = the densest class (voxels)": round(float((top == want).mean()), 4),
                             "rank 1 = the densest class (voxels with a tract)": round(float((top[brain] == want[brain]).mean()), 4),
                             "voxels with a tract ranked first": int(brain.sum())}
            sub = rg.create_group(which)
            meta = {**code.meta, "labels": names + ["none"]}
            ext = {"ensemble": which, "runs": RUNS if which == "A" else RUNS * len(votes)}
            sub.attrs.update(group_attrs(ext, ranked=meta))
            for name, arr in (("ranks", code.ranks), ("support", code.support), ("tail", code.tail)):
                a = np.ascontiguousarray(arr)
                lst = a.ndim == 4
                shp = a.shape
                sub.create_array(name, shape=shp, dtype=a.dtype, chunks=((1,) + shp[1:]) if lst else shp, shards=shp,
                                 compressors=codec, fill_value=0,
                                 attributes=array_attrs(g, {**ext, "part": name}, list_axis=lst))[...] = a
            print(which, checks[which], flush=True)
        (staging / "README.md").write_text(README.format(subject=subject, classes=K, runs=RUNS, scans=", ".join(votes)))
        out.parent.mkdir(parents=True, exist_ok=True)
        pack(staging, out)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"{out}: {out.stat().st_size / 1e6:.2f} MB")
    # read back through a ZipStore and check
    from duckn import DucknMetadata, DucknGroupMetadata
    z = zarr.open_group(zarr.storage.ZipStore(str(out), mode="r"), mode="r")
    for name, node in [("", z)] + list(z.members(max_depth=None)):
        model = DucknGroupMetadata if isinstance(node, zarr.Group) else DucknMetadata
        model.model_validate(node.attrs["duckn"])
    print("every node's duckn metadata validates")
    a = z["ensembles/A/votes"]
    DucknMetadata.model_validate(a.attrs["duckn"])
    back = a[names.index("CST")]
    assert np.array_equal(back, np.transpose(votes["A"][names.index("CST")], (2, 1, 0)))
    print("read back: ensembles/A/votes", a.shape, a.dtype, "CST voxels with votes:", int((back > 0).sum()),
          "| envelope votes CST:", a.attrs["duckn"]["extensions"]["tractline"]["envelope_votes"]["CST"])


README = """# {subject}: tractography ensemble presence (tractline)

A duckn store (zarr v3). Per class (42 tracts and Other, the TractCloud/RapidParc scheme) and voxel of FastSurfer's
conformed 1 mm grid (cropped to the brain, voxel-aligned with the subject's FastSurfer store): how many runs of the tractography ensemble have the class present, presence being streamline
density of at least 0.3 per mm^2 of a voxel's cross-section (smoothed 1 mm).

- ensembles/<scan>/votes: each scan's {runs}-run ensemble (its own run and {runs_less} resampled copies); scans {scans}
- ensembles/<scan>/run: the scan's own run, 0/1
- pooled/votes: all scans' runs together
- rankfield/A, rankfield/pooled: per voxel the classes ranked by mean streamline density (with "none"), as rankfield
  encodes them (ranks, support, tail; the `ranked` block is rankfield's metadata), for A's ensemble and all runs

The root's and each array's `duckn.extensions.tractline` block holds the classes, the presence definition, the vote
counts at which the envelopes of stated levels start, the coverage those levels held on held-out scans, the T1-to-DWI
transform, and the FastSurfer anatomy store in the same frame. Research use only.
""".replace("{runs_less}", "16")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--level", type=int, default=9)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--floor", type=float, default=0.1, help="densities below this share of RHO are absent")
    main(ap.parse_args())
