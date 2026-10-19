"""DIPY's side of the tracker benchmark (bench/robust_benchmark.py; docs/robustness-plan.md, step 6), run in a subprocess
of its own: DIPY's OpenMP and torch's clash in one process. Imports no torch and nothing of tractline.

    python bench/_benchmark_dipy.py tissue_fs   --work W.npz --labels L.seg.nrrd --t1-to-dwi T.npz --out tissue.npz
    python bench/_benchmark_dipy.py tissue_t1   --work W.npz --t1 T1.nii.gz --out tissue.npz
    python bench/_benchmark_dipy.py csd         --work W.npz [--tissue tissue.npz --seeds N --track out.npz]
                                                [--peaks-mni out.npz --template MNI_FA.nii.gz --to-mni T.npz]

W.npz (written by robust_benchmark.py): the corrected DWI on its grid (float32, b0s and the shell tracked only), bval,
bvec (FSL's convention on that grid: image axes, x flipped when the affine's determinant is positive), affine (voxel ->
RAS mm), mask (the pipeline's tracking mask, prep.prepare's).

tissue_fs: partial-volume maps on W's grid from FastSurfer's asegdkt labels (the T1 grid, carried into the DWI's world by
t1_to_dwi): each DWI voxel sampled at SUB^3 points, each point's label looked up (nearest), the classes' shares:
  wm    cerebral and cerebellar white matter, brainstem, ventral DC, the CC, white-matter hypointensities, optic chiasm;
  gm    cortex (DKT, cerebral and cerebellar), hippocampus, amygdala;
  deep  thalamus, caudate, putamen, pallidum, accumbens: neither a valid end nor an invalid one (MRtrix's 5TT keeps
        sub-cortical gray apart for this reason: the CST passes between thalamus and lentiform);
  csf   ventricles, CSF, choroid plexus;
  the rest (label 0, outside the brain; extra-cerebral CSF is not labeled by FastSurfer) is background.
tissue_t1: the same maps from a T1 with no labels (the ISMRM 2015 phantom's T1.nii.gz, never its ground-truth bundles or
ROI masks): inside the pipeline's tracking mask carried to the T1's grid (nearest), three intensity classes by 1D k-means
(CSF darkest, then GM, then WM), deep = 0, then the same SUB^3 sampling onto W's grid. On the phantom the T1's classes
put 29.7 % of the tracking mask's voxels with FA >= 0.2 in gray matter (the median FA of its white class is 0.347, of
its gray 0.114: t1_fa_check, robust_benchmark.py diagnose), so in the tracking mask a voxel with FA >= FA_WM (DTI on W)
has its gray share moved to white: an FA/T1 classifier.
t1_fa_check: those T1-only classes against FA, printed as JSON.

csd: DIPY's single-shell CSD: the response from the data (auto_response_ssst: FA > 0.7 within 10 voxels of the center),
sh_order_max 8 (6 when the shell has fewer than 45 directions), peaks_from_model over the mask (3 peaks, relative
threshold 0.1, 25 degrees apart; the SH kept for tracking). Then, as asked:
  --track: PTT (dipy.tracking.tracker.ptt_tracking, DIPY's defaults for its own settings: step 0.5 mm, max angle 10,
           probe length 1.5 mm, quality 7, count 1, pmf threshold 0.1) with ACT (ActStoppingCriterion: include = gm +
           background, exclude = csf; the image's outermost voxels included, so a path leaving the field of view ends
           validly, as the endpoints analysis's "fov-edge"); only the valid streamlines (return_all=False), 20-250 mm;
           N seeds drawn uniformly in the voxels with wm >= 0.5 (random_seeds_from_mask, --seed-rng, default 1: every
           scan on A's grid gets the same seed points), the tracking's random_seed --track-rng (default 1), so that a
           seed's path is reproducible. --include-nodata (opt-in, after step 6's run): voxels where every volume is 0
           (outside the scan's own field of view: a rescan resampled onto A's grid can leave A's edge slice empty) are
           a valid end too, as the image's faces are. It changes nothing (step 6: the face already was): near an empty
           slice the SH fade to zero and paths stop without a direction a voxel before it, an invalid end, dropped.
           --edge-from-data (opt-in): the brain's voxels (the tissue maps) with no data, and the voxels with data next to
           them (6-neighbors), are a valid end: the scan's own face (post hoc: chosen after step 6's CST result).
           The phantom's seeding controls: --seeds-file (an npz's "world" points, RAS mm, in place of the random
           white-matter seeds) and --return-all (every streamline, the invalid ends too). Writes the streamlines (RAS mm)
           and the times;
  --peaks-mni: the peaks for TractSeg: each peak's direction (the gradient table's frame, image axes) into the world (RAS,
           as MRtrix's sh2peaks writes them) and scaled by its amplitude, then onto the MNI template's grid (1.25 mm, its
           LAS voxel order and MNI signs: TractSeg's flip_axis_to_match_MNI_space is then a no-op) by a rigid registration
           of the scan's FA (DTI on the same data) to TractSeg's MNI FA template (its resources; mutual information,
           center of mass, translation, rigid), nearest neighbor (peaks are not interpolated), the directions turned by
           the registration's rotation. --to-mni: reuse a stored transform (the rescans on A's grid use A's); else it is
           estimated and saved there.
"""
import argparse, json, os, time
from pathlib import Path
import numpy as np

SUB = 4                                           # sub-samples per axis per DWI voxel for the partial-volume maps
WM_LABELS = (2, 41, 7, 46, 16, 28, 60, 77, 85, 251, 252, 253, 254, 255)
GM_LABELS = (3, 42, 8, 47, 17, 53, 18, 54)        # plus every label >= 1000 (DKT cortex)
DEEP_LABELS = (10, 49, 11, 50, 12, 51, 13, 52, 26, 58)
CSF_LABELS = (4, 43, 5, 44, 14, 15, 24, 31, 63, 72)
CLASSES = ("wm", "gm", "deep", "csf")
FA_WM = 0.25                                      # tissue_t1: a voxel this anisotropic is white matter whatever the T1 says
PTT = dict(step_size=0.5, max_angle=10, probe_length=1.5, probe_quality=7, probe_count=1, pmf_threshold=0.1,
           min_len=20, max_len=250)
PEAKS = dict(relative_peak_threshold=0.1, min_separation_angle=25, npeaks=3)


def load_work(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def flip(affine):
    """FSL's b-vectors are in image axes with x flipped when the affine's determinant is positive: the flip that takes
    them into image axes."""
    return np.diag([-1.0, 1.0, 1.0]) if np.linalg.det(affine[:3, :3]) > 0 else np.eye(3)


def rotation(affine):
    M = affine[:3, :3]
    return M / np.linalg.norm(M, axis=0)


def supersample(class_of, affine, shape, world_to_label, label_shape, sub=SUB):
    """Per DWI voxel, the share of its sub^3 sample points in each class: class_of (label grid -> class index, -1 none),
    world_to_label: 4x4 DWI world -> label voxel. -> (X, Y, Z, len(CLASSES)) float32."""
    off = (np.arange(sub) + 0.5) / sub - 0.5
    o = np.stack(np.meshgrid(off, off, off, indexing="ij"), -1).reshape(-1, 3)
    out = np.zeros((*shape, len(CLASSES)), np.float32)
    M = world_to_label @ affine
    ii, jj = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    for k in range(shape[2]):
        v = np.stack([ii, jj, np.full_like(ii, k)], -1).reshape(-1, 1, 3) + o[None]      # (n, s, 3) voxel coords
        lab = np.rint(v @ M[:3, :3].T + M[:3, 3]).astype(np.int64).reshape(-1, 3)
        ok = np.all((lab >= 0) & (lab < label_shape), 1)
        c = np.full(len(lab), -1, np.int64)
        c[ok] = class_of[tuple(lab[ok].T)]
        c = c.reshape(-1, len(o))
        for ci in range(len(CLASSES)):
            out[:, :, k, ci] = (c == ci).mean(1).reshape(shape[:2])
    return out


def tissue_fs(args):
    import nrrd
    w = load_work(args.work)
    lab, h = nrrd.read(str(args.labels))
    lps = np.eye(4)
    lps[:3, :3], lps[:3, 3] = np.asarray(h["space directions"], float).T, h["space origin"]
    t1_aff = np.diag([-1.0, -1.0, 1.0, 1.0]) @ lps                      # RAS, as ensemble_endpoints.Anatomy
    t1_to_dwi = np.load(args.t1_to_dwi)["t1_to_dwi"]
    cls = np.full(lab.shape, -1, np.int8)
    cls[np.isin(lab, WM_LABELS)] = 0
    cls[np.isin(lab, GM_LABELS) | (lab >= 1000)] = 1
    cls[np.isin(lab, DEEP_LABELS)] = 2
    cls[np.isin(lab, CSF_LABELS)] = 3
    unknown = np.setdiff1d(np.unique(lab[(cls < 0) & (lab > 0)]), [])
    world_to_label = np.linalg.inv(t1_aff) @ np.linalg.inv(t1_to_dwi)       # DWI world -> T1 world -> T1 voxel
    t0 = time.time()
    pve = supersample(cls, w["affine"], w["mask"].shape, world_to_label, np.array(lab.shape))
    info = {"source": str(args.labels), "labels without a class (left as background)": unknown.tolist(),
            "seconds": round(time.time() - t0, 1)}
    save_tissue(args.out, pve, w, info)


def kmeans1d(x, k=3, iters=100):
    c = np.percentile(x, np.linspace(10, 90, k))
    for _ in range(iters):
        a = np.abs(x[:, None] - c[None]).argmin(1)
        new = np.array([x[a == j].mean() for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return a, c


def t1_classes(w, t1_path):
    """tissue_t1's T1-only classes on W's grid: (pve, info), before the FA rule."""
    import nibabel as nib
    img = nib.load(str(t1_path))
    t1 = np.asarray(img.dataobj, np.float64)
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in t1.shape], indexing="ij"), -1).reshape(-1, 3)
    M = np.linalg.inv(w["affine"]) @ img.affine
    d = np.rint(ijk @ M[:3, :3].T + M[:3, 3]).astype(np.int64)
    ok = np.all((d >= 0) & (d < w["mask"].shape), 1)
    brain = np.zeros(len(ijk), bool)
    brain[ok] = w["mask"][tuple(d[ok].T)]
    brain = brain.reshape(t1.shape)
    a, centers = kmeans1d(t1[brain])
    order = np.argsort(centers)
    rank = np.empty(3, np.int64); rank[order] = np.arange(3)
    cls = np.full(t1.shape, -1, np.int8)
    cls[brain] = np.array([3, 1, 0])[rank[a]]                              # CSF -> 3, GM -> 1, WM -> 0 (CLASSES)
    pve = supersample(cls, w["affine"], w["mask"].shape, np.linalg.inv(img.affine), np.array(t1.shape))
    return pve, {"k-means centers (CSF, GM, WM)": [round(float(c), 1) for c in np.sort(centers)],
                 "voxels per class at 1 mm (CSF, GM, WM)": [int((cls == c).sum()) for c in (3, 1, 0)]}


def t1_fa_check(args):
    """Why tissue_t1 has its FA rule: on W's grid and in its tracking mask, the T1-only classes against FA (DTI)."""
    from dipy.core.gradients import gradient_table
    w = load_work(args.work)
    pve, _ = t1_classes(w, args.t1)
    mask = w["mask"].astype(bool)
    gtab = gradient_table(w["bval"], bvecs=(flip(w["affine"]) @ w["bvec"]).T, b0_threshold=50)
    fa = fa_map(w["dwi"], gtab, mask)
    out = {}
    for thr in (0.2, FA_WM):
        hi = (fa >= thr) & mask
        out[f"FA >= {thr}: voxels in the mask"] = int(hi.sum())
        out[f"FA >= {thr}: share with T1-only gm >= 0.5"] = round(float((pve[..., 1][hi] >= 0.5).mean()), 4)
        out[f"FA >= {thr}: share with T1-only wm >= 0.5"] = round(float((pve[..., 0][hi] >= 0.5).mean()), 4)
    for c, i in (("wm", 0), ("gm", 1)):
        out[f"median FA, T1-only {c} >= 0.5 in the mask"] = round(float(np.median(fa[(pve[..., i] >= 0.5) & mask])), 4)
    print(json.dumps(out), flush=True)


def tissue_t1(args):
    import nibabel as nib
    w = load_work(args.work)
    img = nib.load(str(args.t1))
    t1 = np.asarray(img.dataobj, np.float64)
    # the tracking mask (DWI grid) carried to the T1's grid, nearest
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in t1.shape], indexing="ij"), -1).reshape(-1, 3)
    M = np.linalg.inv(w["affine"]) @ img.affine
    d = np.rint(ijk @ M[:3, :3].T + M[:3, 3]).astype(np.int64)
    ok = np.all((d >= 0) & (d < w["mask"].shape), 1)
    brain = np.zeros(len(ijk), bool)
    brain[ok] = w["mask"][tuple(d[ok].T)]
    brain = brain.reshape(t1.shape)
    a, centers = kmeans1d(t1[brain])
    order = np.argsort(centers)                                            # darkest first: CSF, GM, WM
    rank = np.empty(3, np.int64); rank[order] = np.arange(3)
    to_class = np.array([3, 1, 0])                                         # CSF -> 3, GM -> 1, WM -> 0 (CLASSES)
    cls = np.full(t1.shape, -1, np.int8)
    cls[brain] = to_class[rank[a]]
    t0 = time.time()
    pve = supersample(cls, w["affine"], w["mask"].shape, np.linalg.inv(img.affine), np.array(t1.shape))
    # the T1's classes put 29.7 % of the voxels with FA >= 0.2 in gray matter (t1_fa_check), where ACT would
    # end paths inside bundles: in the tracking mask, a voxel with FA >= FA_WM has its gray share moved to white
    from dipy.core.gradients import gradient_table
    gtab = gradient_table(w["bval"], bvecs=(flip(w["affine"]) @ w["bvec"]).T, b0_threshold=50)
    fa = fa_map(w["dwi"], gtab, w["mask"].astype(bool))
    hi = (fa >= FA_WM) & w["mask"].astype(bool)
    moved = float(pve[..., 1][hi].sum())
    pve[..., 0][hi] += pve[..., 1][hi]
    pve[..., 1][hi] = 0
    info = {"source": str(args.t1), "brain": "the pipeline's tracking mask on the T1's grid (nearest)",
            "k-means centers (CSF, GM, WM)": [round(float(c), 1) for c in np.sort(centers)],
            "voxels per class at 1 mm (CSF, GM, WM)": [int((cls == c).sum()) for c in (3, 1, 0)],
            f"FA >= {FA_WM}: voxels in the mask, and their gray share moved to white (voxel equivalents)":
                [int(hi.sum()), round(moved, 1)],
            "seconds": round(time.time() - t0, 1)}
    save_tissue(args.out, pve, w, info)


def save_tissue(out, pve, w, info):
    out = Path(out)
    if out.exists():
        raise SystemExit(f"{out} exists; not overwritten")
    info["mean share in the tracking mask"] = {c: round(float(pve[..., i][w["mask"]].mean()), 3) for i, c in enumerate(CLASSES)}
    info["wm >= 0.5 voxels"] = int((pve[..., 0] >= 0.5).sum())
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, pve=pve.astype(np.float16), classes=np.array(CLASSES), affine=w["affine"],
                        info=np.array(json.dumps(info)))
    os.replace(tmp, out)
    print(json.dumps(info), flush=True)


def fa_map(data, gtab, mask):
    from dipy.reconst.dti import TensorModel
    fa = TensorModel(gtab).fit(data, mask=mask).fa
    return np.nan_to_num(fa).astype(np.float32)


def register_to_mni(fa, affine, template):
    """Rigid registration of fa (on affine's grid) to the MNI FA template: 4x4 mapping MNI world -> the scan's world
    (DIPY's AffineMap.affine: static world -> moving world), and the template's affine and shape."""
    import nibabel as nib
    from dipy.align import affine_registration
    t = nib.load(str(template))
    static = np.asarray(t.dataobj, np.float32)
    _, T = affine_registration(fa, static, moving_affine=affine, static_affine=t.affine,
                               pipeline=["center_of_mass", "translation", "rigid"], metric="MI",
                               level_iters=[10000, 1000, 100], sigmas=[3.0, 1.0, 0.0], factors=[4, 2, 1])
    return np.asarray(T), t.affine, t.shape


def peaks_on_mni(peak_dirs, peak_values, affine, mni_to_scan, mni_affine, mni_shape):
    """(X, Y, Z, 3, 3) directions in image axes (the gradient table's frame) and (X, Y, Z, 3) amplitudes on the scan's
    grid -> (mx, my, mz, 9) peaks in MNI world (RAS), amplitude-scaled, nearest neighbor."""
    R = rotation(affine)
    world = np.einsum("ij,xyzpj->xyzpi", R, peak_dirs) * peak_values[..., None]   # scan world RAS
    Rm = rotation(mni_to_scan)                                                   # MNI world -> scan world, rotation part
    world = np.einsum("ji,xyzpj->xyzpi", Rm, world)                              # scan world -> MNI world: Rm^T
    ijk = np.stack(np.meshgrid(*[np.arange(s) for s in mni_shape], indexing="ij"), -1).reshape(-1, 3)
    M = np.linalg.inv(affine) @ mni_to_scan @ mni_affine
    d = np.rint(ijk @ M[:3, :3].T + M[:3, 3]).astype(np.int64)
    ok = np.all((d >= 0) & (d < peak_dirs.shape[:3]), 1)
    out = np.zeros((len(ijk), 9), np.float32)
    out[ok] = world[tuple(d[ok].T)].reshape(-1, 9)
    return out.reshape(*mni_shape, 9)


def csd(args):
    from dipy.core.gradients import gradient_table
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
    from dipy.direction import peaks_from_model
    from dipy.data import default_sphere
    times, info = {}, {}
    t0 = time.time()
    w = load_work(args.work)
    data, affine, mask = w["dwi"], w["affine"], w["mask"].astype(bool)
    bvec = flip(affine) @ w["bvec"]                                          # image axes: DIPY's frame for voxel tracking
    gtab = gradient_table(w["bval"], bvecs=bvec.T, b0_threshold=50)
    ndir = int((w["bval"] > 50).sum())
    order = 8 if ndir >= 45 else 6
    response, ratio = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
    info.update(directions=ndir, sh_order_max=order, response_evals=[float(x) for x in response[0]],
                response_S0=float(response[1]), response_ratio=round(float(ratio), 4))
    model = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=order)
    pam = peaks_from_model(model, data, default_sphere, PEAKS["relative_peak_threshold"], PEAKS["min_separation_angle"],
                           mask=mask, return_sh=True, npeaks=PEAKS["npeaks"], sh_order_max=order, parallel=True,
                           num_processes=os.cpu_count())
    times["csd_fit_and_peaks"] = round(time.time() - t0, 1)
    if args.track:
        from dipy.tracking.stopping_criterion import ActStoppingCriterion
        from dipy.tracking.tracker import ptt_tracking
        from dipy.tracking.utils import random_seeds_from_mask
        from nibabel.streamlines import ArraySequence
        t1 = time.time()
        z = np.load(args.tissue)
        pve = z["pve"].astype(np.float64)
        wm, gm, csf = pve[..., 0], pve[..., 1], pve[..., 3]
        background = pve.sum(-1) < 1e-6
        include = gm.copy()
        include[background] = 1.0
        for ax in range(3):                                                  # the field of view's faces: a valid end
            sl = [slice(None)] * 3
            for idx in (0, -1):
                sl[ax] = idx
                include[tuple(sl)] = 1.0
        nodata = ~np.any(data != 0, axis=-1)                                  # outside the scan's own field of view
        info["voxels with no data (every volume 0)"] = int(nodata.sum())
        if args.include_nodata:                                              # the empty voxels as the image's faces
            include[nodata] = 1.0
        if args.edge_from_data:                                              # the last voxels with data next to brain
            from scipy.ndimage import binary_dilation                        # without data: the scan's own face
            hole = nodata & (pve.sum(-1) > 0)
            edge = binary_dilation(hole) & ~nodata
            include[hole | edge] = 1.0
            info["brain voxels with no data / the data's edge next to them"] = [int(hole.sum()), int(edge.sum())]
        exclude = csf.copy()
        sc = ActStoppingCriterion(np.ascontiguousarray(include), np.ascontiguousarray(exclude))
        seed_mask = wm >= 0.5
        if args.seeds_file:                                                  # given points (RAS mm), e.g. UKF's seeds
            seeds = np.ascontiguousarray(np.load(args.seeds_file)["world"], np.float64)
        else:
            seeds = random_seeds_from_mask(seed_mask, affine, seeds_count=int(args.seeds), seed_count_per_voxel=False,
                                           random_seed=args.seed_rng)
        gen = ptt_tracking(seeds, sc, affine, sh=pam.shm_coeff, sphere=default_sphere, nbr_threads=os.cpu_count(),
                           random_seed=args.track_rng, return_all=bool(args.return_all), **PTT)
        sl = ArraySequence(gen)
        times["track"] = round(time.time() - t1, 1)
        info.update(seeds=int(len(seeds)), seed_voxels=int(seed_mask.sum()), streamlines=int(len(sl)),
                    settings={**PTT, "stopping": "ACT (include gm + background + the FOV's faces"
                              + (" + voxels with no data" if args.include_nodata else "")
                              + (" + brain voxels with no data and the data next to them" if args.edge_from_data else "")
                              + ", exclude csf)",
                              "return_all": bool(args.return_all), "random_seed (tracking)": args.track_rng,
                              "seeding": (f"the points in {args.seeds_file}" if args.seeds_file
                                          else f"uniform in wm >= 0.5, random_seed {args.seed_rng}")})
        lens = np.asarray(sl._lengths, np.int64)
        np.savez(args.track + ".tmp.npz", points=np.asarray(sl.get_data(), np.float32), lengths=lens,
                 info=np.array(json.dumps({**info, "seconds": times})))
        os.replace(args.track + ".tmp.npz", args.track)
    if args.peaks_mni:
        t2 = time.time()
        if args.to_mni and Path(args.to_mni).exists():
            zt = np.load(args.to_mni)
            T, mni_aff, mni_shape = zt["mni_to_scan"], zt["mni_affine"], tuple(zt["mni_shape"])
            info["registration"] = f"reused {args.to_mni}"
        else:
            fa = fa_map(data, gtab, mask)
            T, mni_aff, mni_shape = register_to_mni(fa, affine, args.template)
            info["registration"] = "estimated (rigid, FA to TractSeg's MNI FA template)"
            if args.to_mni:
                np.savez(args.to_mni, mni_to_scan=T, mni_affine=mni_aff, mni_shape=np.array(mni_shape))
        peaks = peaks_on_mni(pam.peak_dirs, pam.peak_values, affine, T, mni_aff, mni_shape)
        times["peaks_to_mni"] = round(time.time() - t2, 1)
        np.savez_compressed(args.peaks_mni + ".tmp.npz", peaks=peaks, mni_affine=mni_aff, mni_to_scan=T,
                            info=np.array(json.dumps({**info, "seconds": times})))
        os.replace(args.peaks_mni + ".tmp.npz", args.peaks_mni)
    times["total"] = round(time.time() - t0, 1)
    print(json.dumps({**info, "seconds": times}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("tissue_fs"); a.add_argument("--work"); a.add_argument("--labels"); a.add_argument("--t1-to-dwi"); a.add_argument("--out")
    a = sub.add_parser("tissue_t1"); a.add_argument("--work"); a.add_argument("--t1"); a.add_argument("--out")
    a = sub.add_parser("t1_fa_check"); a.add_argument("--work"); a.add_argument("--t1")
    a = sub.add_parser("csd"); a.add_argument("--work"); a.add_argument("--tissue"); a.add_argument("--seeds", type=int)
    a.add_argument("--seed-rng", type=int, default=1); a.add_argument("--track-rng", type=int, default=1)
    a.add_argument("--include-nodata", action="store_true"); a.add_argument("--edge-from-data", action="store_true")
    a.add_argument("--seeds-file"); a.add_argument("--return-all", action="store_true")
    a.add_argument("--track"); a.add_argument("--peaks-mni"); a.add_argument("--template"); a.add_argument("--to-mni")
    args = ap.parse_args()
    {"tissue_fs": tissue_fs, "tissue_t1": tissue_t1, "t1_fa_check": t1_fa_check, "csd": csd}[args.cmd](args)


if __name__ == "__main__":
    main()
