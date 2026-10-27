"""The pipeline, scan to labeled streamlines, in memory: correct -> track -> label [-> TRX].

    s = bids.load("sub-01/dwi/sub-01_dwi.nii.gz"); timer = Timer()   # any BIDS series (tractline.bids)
    corr, tg, labels = run(s, None, timer, device="mps")   # None: the default labeler (RapidParc)
    print(corr.note, corr.warnings)                         # what corrected it, or why nothing did

    cfg = PipelineConfig.preset("default").with_(model={"shell": 2000.0}, denoise="mppca")   # tractline.config
    r = execute(s, cfg, timer, device="mps")                # Run: r.correction, r.tractogram, r.labels, r.config

The subject s: any object with these fields (bids.Subject and the bench's _ds001226.Subject are two):
  dwi         (X, Y, Z, G) the diffusion series as acquired (one phase-encoding direction)
  affine      4x4 voxel -> RAS mm; vox (3,) voxel sizes mm
  bval (G,), bvec (3, G)    FSL convention
  b0s         (X, Y, Z, V) float64 on the DWI's grid: the DWI's own b0s first (volume 0 is the
              estimate's fixed reference), then the other series' b0s (the reversed ones, and any of the
              same polarity that measure the field with them)
  pe_vectors  (V, 3) each b0's phase-encoding vector; pe_axis (0-2) and pe_sign (+1/-1): the DWI's
              (they must agree with pe_vectors[0]); readout_s: total readout time, s
  optional    b0_readout_s (V,): each b0's own readout time (else readout_s for all); pairing
              (bids.Pairing): correct only when pairing.ok. b0s None, or a failed pairing: the DWI is
              tracked as acquired, Correction.applied False and the reason in Correction.note.
Defaults to know: device="mps" (pass "cuda" or "cpu" elsewhere); shell=2800.0 (ds001226's b-value -
prepare refuses a scan without that shell; pass the scan's own).

Stages, and where they run:
  denoise   opt-in (run's denoise=; none by default): the diffusion series as acquired, by dipy's MP-PCA or
            Patch2Self in a process of its own (tractline.denoise; denoise_subject), before the field is estimated
  correct   the susceptibility field from the b0s and their reversed pair (susceptibility.estimate: GPU,
            float32, ~18 s on the M2), then applied to the DWI (susceptibility.apply: CPU, float64, ~1 s);
            residual_left: how much of the pair's difference the field leaves - above RESIDUAL_WARN a
            warning (Correction.warnings), never a refusal. Skipped when there is no pair (can_correct).
  motion_eddy  opt-in (run's motion_eddy=; the motion_eddy slot): each volume's head motion and eddy currents
            (tractline.motion_eddy, the field in the map), then one resample of every volume composing field, motion
            and eddy in place of susceptibility.apply, and the b-vectors rotated for the tracker
            (correct_motion_eddy; ~20 s on the M2 for 105 volumes). Without a pair: motion and eddy only.
  mask      median_otsu of the mean b0 (brain_mask; exact in torch on the pipeline's device; ~3 s)
  track     the tracker's input (prep.prepare: b0s + the b = 2800 shell, gradients in RAS; ~0.1 s),
            UKF two-tensor with the
            ORG settings and the binary's seeds (ukf.track: on "mps" the Metal kernel, float32 steps
            from float64 seeds, ~27 s; on "cuda" the Triton block kernel; on the CPU one process per
            core, the fast step; compiled by torch.compile only on request, TRACTLINE_COMPILE=1 (ukf.can_compile:
            each worker pays its start, and with many workers it was slower))
  label     RapidParc, one draw (labelers.rapidparc: shuffled groups of 2,000 streamlines; GPU or CPU,
            float32, ~1 s); TractCloud (labelers.tractcloud, at its trained context) on request
  TRX       optional: the tractogram with labels and tract probabilities (trx.write)
Dependencies of the default path: numpy, torch and RapidParc's weights - no scipy, no nibabel (reading the
scan is the caller's), no TractCloud code (bench/dependency_check.py).

Configuration (tractline.config): the stages are slots - denoise, field, mask, model, track, bundles - each off or one
implementation with its options; a PipelineConfig names them all, checks them before any work, and hashes what changes the
outputs (calibration maps key on that hash). execute(s, cfg, ...) runs one; run()'s keywords build one from the "default"
preset (config_for) and call execute, so both paths are the same code. The run's config and runtime are in
Correction.provenance (execute: also Run.config and Run.runtime).

Conventions at every module boundary (_ds001226, susceptibility, prep, ukf, the labelers, t1check):
  volumes   numpy (X, Y, Z[, V]) in the NIfTI's voxel order; affine voxel -> RAS mm; voxel sizes as
            the header states them (the affine's column norms differ by ~1e-8)
  fields    Hz on that grid; displacements mm along the phase-encoding axis, signed, as the DWI's
            own phase encoding moves the anatomy (susceptibility.displacement_mm)
  fibers    lists of (n, 3) RAS mm arrays, in seed order; labels for those >= 40 mm (Labels.keep)
  devices   device="mps" | "cuda" | "cpu"; timings through Timer, which waits for the GPU
"""
from __future__ import annotations

import copy, os, time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np, torch
from . import susceptibility as S
from . import ukf as U
from .prep import prepare
from .mask import median_otsu
from .config import PipelineConfig
from .labelers.base import Labels

ESTIMATE_THREADS = 16            # the field estimate on the CPU: its problem is small (~5e5 voxels); on 48 vCPUs
                                 # 8 threads 48.9 s, 16 40.2 s, 32 44.6 s, 48 51.5 s (modal_cpu_field_timing.json)


def release_memory(device):
    """Return the GPU allocator's cached blocks to the system (and collect Python's garbage). PyTorch keeps
    freed GPU memory cached for reuse; across patients in one process the M2's GPU driver memory grew to
    7.8 GB with 0.03 GB live (bench/memory_batch.py), 0.1 GB after this, at ~4 % more time. run() calls it;
    a batch that calls the stages or the labelers itself should call it between patients."""
    import gc
    gc.collect()
    kind = torch.device(device).type
    if kind == "mps":
        torch.mps.empty_cache()
    elif kind == "cuda":
        torch.cuda.empty_cache()


@contextmanager
def exact_float32(device):
    """On CUDA, float32 as float32 for the pipeline's stages: no TF32 in cuDNN's convolutions or in
    matmul (PyTorch allows it in cuDNN by default on Ampere and newer, a 10-bit mantissa), restored
    after. With TF32, TractCloud (at 80) on an A10 labeled 62.6 % of PAT16's streamlines Other against
    61.5 % without, on an L40S 64.0 % against 62.8 %, and the field moved by up to 1 mm; without it,
    TractCloud on an A10 labels a tractogram as the M2 does, every streamline. It costs nothing on the
    A10, 2.6 s on the L40S (NOTES 2026-10-02, "The pipeline on CUDA")."""
    if torch.device(device).type != "cuda":
        yield
        return
    before = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = before


@contextmanager
def threads(n):
    """torch's CPU threads set to n for a stage, restored after. Set per stage, not once: on Modal's
    Linux the count read 48 after a tracking pool though 32 had been set (not seen on macOS)."""
    before = torch.get_num_threads()
    torch.set_num_threads(max(1, int(n)))
    try:
        yield
    finally:
        torch.set_num_threads(before)


class Timer:
    """Named stage times (s); with `echo`, each printed as it ends, prefixed by echo, to `stream` (standard output by
    default; the tractline command prints them to standard error, keeping standard output for --json -)."""
    def __init__(self, echo=None, stream=None):
        self.seconds, self.echo, self.stream = {}, echo, stream

    @contextmanager
    def __call__(self, name):
        t0 = time.time()
        yield
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.seconds[name] = round(time.time() - t0, 2)
        if self.echo is not None:
            print(f"{self.echo} {name} {self.seconds[name]} s", flush=True, file=self.stream)

    def skip(self, name):
        """A stage that did not run: recorded as 0 s, so totals over it stay exact and a misspelled name still fails."""
        self.seconds[name] = 0.0
        if self.echo is not None:
            print(f"{self.echo} {name} skipped", flush=True, file=self.stream)

    def total(self, *names):
        return round(sum(self.seconds[n] for n in names), 1)


@dataclass
class Correction:
    dwi: np.ndarray              # (X, Y, Z, G) float32, corrected (as acquired when neither the field nor motion_eddy ran)
    field_hz: np.ndarray | None  # (X, Y, Z); None when the field was not applied
    motion: np.ndarray | None    # (V, 6) per b0: translations mm, rotations rad; None when not applied
    displacement_mm: np.ndarray | None   # (X, Y, Z): what the field did to the DWI as acquired
    applied: bool = True         # the field was applied (motion_eddy without a field: False, and its note says so)
    note: str = ""               # what corrected it, or why nothing did (bids.Pairing.message)
    residual_left: float | None = None   # the pair's mean-b0 difference left after correction (residual_left)
    warnings: list = field(default_factory=list)
    denoise: dict | None = None  # the opt-in denoising's settings and what it estimated (denoise_subject); None: not denoised
    gibbs: dict | None = None    # the gibbs slot's b0 counts (gibbs_subject), when it ran on b0s; None otherwise
    bvec: np.ndarray | None = None   # (3, G) FSL, rotated with the head by the motion_eddy slot; None when it is off
    motion_eddy: dict | None = None  # the motion_eddy slot's parameters and QC (motion_eddy.record); None when off
    provenance: dict | None = None   # the run's config (PipelineConfig.to_dict) and runtime (execute); None: correct() alone


RESIDUAL_WARN = 0.5              # residual_left above this: a warning, never a refusal. On 23 judged good pairs
                                 # (ds001226, ds005123) 0.27-0.64, 3 above 0.5 - ones that moved; on 30 judged bad
                                 # ones (one polarity labeled two, a re-shimmed pair) 0.53-0.96, all above; not
                                 # judged: 7, all where a series' own volumes moved or the field was ~0
                                 # (bench/pair_residual_check.py; NOTES 2026-10-03, "Release review")


def can_correct(s) -> bool:
    """b0s from both polarities along the diffusion series' phase-encoding axis, and (a bids.Subject) a
    pairing that passed its checks."""
    pairing = getattr(s, "pairing", None)
    if getattr(s, "b0s", None) is None or (pairing is not None and not pairing.ok):
        return False
    signs = set(np.sign(np.asarray(s.pe_vectors, float)[:, s.pe_axis]).tolist()) - {0.0}
    return signs == {-1.0, 1.0}


def residual_left(s, h):
    """How much of the pair's difference the field leaves: ||mean corrected A - mean corrected B|| /
    ||mean A - mean B|| over the bright voxels (above the Otsu threshold of the two groups' mean), A the b0s
    phase-encoded as the diffusion series is, B the others. Every volume is scaled to the common mean (as
    the estimate does) and corrected on its own, with its own polarity and readout time, before the group
    means are taken. Motion is not applied, so what motion moved counts as left. None when the groups
    differ by less than twice what their noise alone would give (estimated from each group's volumes about
    its own mean; with one volume a group, not estimated): a ratio of noise to noise judges nothing."""
    from .mask import otsu
    b0s = np.asarray(s.b0s, np.float64)
    b0s = b0s * (b0s.mean() / b0s.mean(axis=(0, 1, 2), keepdims=True))
    ro = getattr(s, "b0_readout_s", None)
    ro = np.broadcast_to(np.asarray(s.readout_s if ro is None else ro, float), (b0s.shape[-1],))
    sign = np.sign(np.asarray(s.pe_vectors, float)[:, s.pe_axis])
    a = sign == np.sign(s.pe_sign)
    b = ~a
    ma, mb = b0s[..., a].mean(-1), b0s[..., b].mean(-1)
    mean = (ma + mb) / 2
    bright = mean > otsu(mean)
    d0 = np.linalg.norm((ma - mb)[bright])
    dof = int(a.sum() + b.sum()) - 2
    if dof > 0:
        within = sum(((b0s[..., g] - m[..., None])[bright] ** 2).sum() for g, m in ((a, ma), (b, mb)))
        if d0 ** 2 < 4 * within / dof * (1 / a.sum() + 1 / b.sum()):
            return None
    corrected = np.stack([S.apply(b0s[..., v:v + 1], h, s.pe_axis, float(sign[v]), float(ro[v]))[..., 0]
                          for v in range(b0s.shape[-1])], -1)
    left = np.linalg.norm((corrected[..., a].mean(-1) - corrected[..., b].mean(-1))[bright]) / d0
    return float(left)


@dataclass
class Tractogram:
    fibers: list                 # (n, 3) RAS mm arrays, seed order
    stats: dict                  # ukf.track's: seeds, half_fibers, fiber_steps, fibers, seed_index,
                                 # seed_voxel - (k, j, i), the binary's order, not the volumes' (i, j, k)
    mask: np.ndarray             # (X, Y, Z) bool, the tracking mask
    info: dict                   # prep's


def correct(s, timer: Timer, device="mps") -> Correction:
    """The field estimate is susceptibility.estimate's defaults (Gauss-Newton, NOTES 2026-10-02): on "mps" its
    subsampled levels on the CPU, on the CPU and on "cuda" in float32 (all levels on the GPU there) -
    the same model on each, 18 s / 22.6 s on the M2. The CPU's levels get ESTIMATE_THREADS. Each b0's readout
    time is s.b0_readout_s's when the subject has one (bids.load), else s.readout_s for all.
    Without a pair to correct with (can_correct), the DWI as acquired, applied=False and the reason in note."""
    if not can_correct(s):
        timer.skip("field_estimate"); timer.skip("field_apply")
        return Correction(np.array(s.dwi, np.float32), None, None, None, applied=False, note=_no_pair_note(s))
    h, motion = _estimate_field(s, timer, device)
    with timer("field_apply"):
        dwi = S.apply(s.dwi, h, s.pe_axis, s.pe_sign, s.readout_s)
    note, left, warnings = _field_note(s, h)
    return Correction(dwi, h, motion, S.displacement_mm(h, s.readout_s, s.pe_sign, s.vox[s.pe_axis]), applied=True,
                      note=note, residual_left=left, warnings=warnings)


def _no_pair_note(s):
    pairing = getattr(s, "pairing", None)
    return (pairing.message if pairing is not None and not pairing.ok
            else "not corrected: no b0s from both phase-encoding polarities")


def _estimate_field(s, timer: Timer, device):
    """(field Hz, the b0s' motion): susceptibility.estimate as correct() runs it."""
    f32 = torch.device(device).type in ("cpu", "cuda")
    ro = getattr(s, "b0_readout_s", None)
    with timer("field_estimate"), threads(min(ESTIMATE_THREADS, os.cpu_count() or 1)):
        h, motion, _ = S.estimate(s.b0s, s.vox, s.pe_vectors, s.readout_s if ro is None else ro, device=device,
                                  **(dict(dtype=torch.float32) if f32 else {}))
    return h, motion


def _field_note(s, h):
    """(note, residual_left rounded or None, warnings) for a field: the pairing's message and the residual's verdict."""
    pairing = getattr(s, "pairing", None)
    note, warnings = (pairing.message if pairing is not None else "corrected"), []
    try:                                                                     # a diagnostic: it never fails the run
        left = residual_left(s, h)
    except Exception as e:                                                   # noqa: BLE001
        left, warnings = None, [f"the residual could not be computed ({type(e).__name__}: {e})"]
    if left is None and not warnings:
        note += ("; residual not judged: the series differ by little more than their volumes vary within each "
                 "(noise, or motion within a series)")
    elif left is not None and not np.isfinite(left):
        left, warnings = None, ["the residual could not be computed (not finite)"]
    elif left is not None and left > RESIDUAL_WARN:
        warnings = [f"the pair's difference left after correction is {left:.2f} (above {RESIDUAL_WARN}): head motion "
                    "between the series, or series that do not share one field (a re-shim, another protocol) - check "
                    "before relying on it"]
    return note, None if left is None else round(left, 3), warnings


def correct_motion_eddy(s, timer: Timer, device="mps", opts=None, field=True) -> Correction:
    """The field slot (when `field` and the subject has a pair: the estimate as correct() makes it), then the motion_eddy
    slot (tractline.motion_eddy.estimate: each volume's head motion and eddy currents, the field in the map), then one
    resample of every volume composing all three (timed as "field_apply", the stage it replaces). Without a field,
    motion and eddy currents only (as eddy without topup); the note says which. Correction.bvec: the rotated b-vectors;
    Correction.motion_eddy: tractline.motion_eddy.record()'s. opts: the slot's options."""
    from . import motion_eddy as ME
    h = motion = None
    note, left, warnings = "", None, []
    if field and can_correct(s):
        h, motion = _estimate_field(s, timer, device)
        note, left, warnings = _field_note(s, h)
        note += "; motion and eddy currents corrected with the field, one resample"
    else:
        timer.skip("field_estimate")
        note = ("not corrected for the field: the field slot is off" if not field else _no_pair_note(s)) \
            + "; motion and eddy currents corrected (no field)"
    with timer("motion_eddy"):
        # the brain mask of the data as acquired, by the pipeline's own median_otsu on the device (dipy's, exactly); with a
        # field the library takes its unwarped copy as the reference frame's (its own default would rerun median_otsu on
        # the unwarped mean b0, on the CPU: a few seconds, nearly the same mask)
        b0 = np.asarray(s.dwi)[..., np.asarray(s.bval) < 50].astype(np.float32).mean(-1)
        mask = median_otsu(b0, median_radius=4, numpass=4, device=device)[1]
        r = ME.estimate(s.dwi, s.bval, s.bvec, s.affine, s.pe_axis, field_hz=h, pe_sign=s.pe_sign,
                        readout_s=s.readout_s, mask=mask, options=dict(opts or {}), device=device,
                        slice_axis=slice_axis_of(s))
    with timer("field_apply"):
        dwi, _valid = ME.apply_volumes(s.dwi, r, field_hz=h, device=device)
    if r.excluded:
        warnings = warnings + ["motion_eddy: volumes left out of the fit and of tracking (in the corrected series with "
                               "identity motion, b-vector as acquired): "
                               + "; ".join(f"{v}: {why}" for v, why in r.excluded.items())]
    if r.outlier_slices:
        n = sum(len(v) for v in r.outlier_slices.values())
        how = "replaced by their prediction" if r.options["outliers"] == "replace" else "flagged, left as acquired"
        warnings = warnings + [f"motion_eddy: {n} outlier slices (signal loss) in {len(r.outlier_slices)} volumes, {how}: "
                               + "; ".join(f"{v}: {ks}" for v, ks in r.outlier_slices.items())]
    if r.qc.get("non-finite values set to 0"):
        warnings = warnings + [f"motion_eddy: {r.qc['non-finite values set to 0']} non-finite values in the series, set "
                               "to 0"]
    if r.qc.get("volumes with residual z > 4"):
        warnings = warnings + [f"motion_eddy: volumes {r.qc['volumes with residual z > 4']} fit their prediction worst "
                               "(residual z > 4 within their shell): signal dropout or motion within the volume?"]
    disp = S.displacement_mm(h, s.readout_s, s.pe_sign, s.vox[s.pe_axis]) if h is not None else None
    return Correction(dwi, h, motion, disp, applied=h is not None, note=note, residual_left=left, warnings=warnings,
                      bvec=r.bvec, motion_eddy=ME.record(r))


#: the opt-in denoisers (tractline.denoise; a name may carry a setting, "mppca:r3"), and where they run: "acquired", the
#: series as acquired, before the field
DENOISE = ("mppca", "patch2self")
DENOISE_ORDERS = ("acquired", "corrected")


def denoise_subject(s, timer: Timer, method="mppca", workers=None, mask=None):
    """The diffusion series as acquired, denoised (tractline.denoise: dipy, in a process of its own), before the field
    is estimated: (a copy of s with dwi denoised, the run's stats). Both methods assume the noise of the volumes as
    acquired; the correction interpolates along the phase-encoding axis and scales by the Jacobian, which smooths
    the noise and makes its level vary. The b0s that feed the field estimate and are the DWI's own volumes (found by
    value: s.b0s's volumes equal to one of the DWI's b0s) are taken from the denoised series, so the estimate's
    forward b0s are the ones the DWI is corrected as. The other b0s (the reversed series) stay as acquired: one or a
    few volumes, too few to denoise on their own, and distorted the other way, so they cannot share the forward
    series' patches (MP-PCA) or voxelwise regressions (Patch2Self). When fewer b0s match than the b0s phase-encoded as
    the DWI is, a warning (the estimate then mixes denoised and raw forward b0s). method: a name tractline.denoise.parse
    takes ("mppca", "mppca:r3", "patch2self"); workers: MP-PCA's processes (default one per core); mask: a brain mask
    on the DWI's grid, if one is known, over which the stats are given too (denoise.summary)."""
    import warnings
    from . import denoise as D
    D.parse(method)                                                        # a ValueError before any work
    with timer("denoise"):
        d = D.denoise(s.dwi, s.bval, method, workers)
    out = copy.copy(s)
    out.dwi = d.dwi
    replaced = forward = 0
    if getattr(s, "b0s", None) is not None:
        b0s = np.array(s.b0s, np.float64)
        b0i = np.flatnonzero(np.asarray(s.bval) < 50)
        for v in range(b0s.shape[-1]):
            same = [i for i in b0i if np.array_equal(b0s[..., v], np.asarray(s.dwi[..., i], np.float64))]
            if same:
                b0s[..., v], replaced = d.dwi[..., same[0]], replaced + 1
        out.b0s = b0s
        forward = int((np.sign(np.asarray(s.pe_vectors, float)[:, s.pe_axis]) == np.sign(s.pe_sign)).sum())
        if replaced < forward:
            warnings.warn(f"denoise: {replaced} of the {forward} b0s phase-encoded as the DWI matched its own volumes and "
                          "were denoised; the others feed the field as acquired", stacklevel=2)
    n_b0s = None if getattr(s, "b0s", None) is None else int(s.b0s.shape[-1])
    stats = {**D.summary(s.dwi, d, s.bval, mask), "order": "acquired",
             "b0s denoised (the DWI's own), of those feeding the field": [replaced, n_b0s],
             "b0s phase-encoded as the DWI": forward}
    return out, stats


def slice_axis_of(s) -> int:
    """The acquisition's slice axis (0, 1 or 2) from the BIDS sidecar's SliceEncodingDirection (i, j, k; k when the
    sidecar does not say): where the gibbs slot's 2D slices and motion_eddy's outlier slices lie."""
    sed = str((getattr(s, "sidecar", None) or {}).get("SliceEncodingDirection", "k"))[:1]
    return {"i": 0, "j": 1}.get(sed, 2)


def gibbs_subject(s, timer: Timer, opts: dict, device="cpu"):
    """The gibbs slot: Gibbs ringing removed from the diffusion series as acquired and from the b0s that feed the field
    (tractline.gibbs). The b0s that are the DWI's own volumes (found by value, as denoise_subject finds them) are taken
    from the corrected DWI, so the field is estimated from the b0s the DWI is corrected as. The other series' b0s are
    corrected on their own where s.b0_native says they were not resampled onto the DWI's grid (the ringing is a pattern
    on the acquired grid; an interpolated image no longer has it there), and left as they are otherwise, or when the
    loader does not say (or s.b0_native's length is not the b0s'). A warning when a b0 phase-encoded against the DWI is
    left: the field is then estimated from corrected forward b0s against uncorrected reversed ones. A copy of s, and
    the counts: {"b0s from the DWI": n, "b0s corrected on their own": n, "b0s left (resampled or not known)": n}."""
    import warnings
    from . import gibbs as GB
    out = copy.copy(s)
    run = lambda v: GB.gibbs_removal(np.asarray(v, np.float32), opts["slice_axis"], opts["n_points"], device)  # noqa: E731
    counts = None
    with timer("gibbs"):
        out.dwi = run(s.dwi)
        if getattr(s, "b0s", None) is not None:
            b0s = np.array(s.b0s, np.float64)
            b0i = np.flatnonzero(np.asarray(s.bval) < 50)
            native = getattr(s, "b0_native", None)
            if native is not None and len(native) != b0s.shape[-1]:
                native = None                                             # flags for other b0s (e.g. a replace): not known
            own, alone = [], []
            for v in range(b0s.shape[-1]):
                same = [i for i in b0i if np.array_equal(b0s[..., v], np.asarray(s.dwi[..., i], np.float64))]
                if same:
                    b0s[..., v] = out.dwi[..., same[0]]
                    own.append(v)
                elif native is not None and native[v]:
                    alone.append(v)
            if alone:
                b0s[..., alone] = run(b0s[..., alone])
            out.b0s = b0s
            left = sorted(set(range(b0s.shape[-1])) - set(own) - set(alone))
            pev = getattr(s, "pe_vectors", None)
            if left and pev is not None:
                against = [v for v in left if np.sign(np.asarray(pev, float)[v, s.pe_axis]) == -np.sign(s.pe_sign)]
                if against:
                    warnings.warn(f"gibbs: {len(against)} b0s phase-encoded against the DWI keep their ringing (resampled "
                                  "onto the DWI's grid, or the loader does not say they were not); the field is "
                                  "estimated from corrected forward b0s against them", stacklevel=2)
            counts = {"b0s from the DWI": len(own), "b0s corrected on their own": len(alone),
                      "b0s left (resampled or not known)": int(b0s.shape[-1] - len(own) - len(alone))}
    return out, counts


def denoise_corrected(dwi, bval, timer: Timer, method="mppca", workers=None, mask=None):
    """The other order, for comparison: the corrected DWI denoised (its noise interpolated and scaled by the field's
    Jacobian, against both methods' assumptions): (dwi, the run's stats)."""
    from . import denoise as D
    D.parse(method)
    with timer("denoise"):
        d = D.denoise(dwi, bval, method, workers)
    return d.dwi, {**D.summary(dwi, d, bval, mask), "order": "corrected"}


#: ukf.track's settings pipeline.track passes on when asked (tracker=); every other keyword is the pipeline's own
TRACKER_SETTINGS = frozenset({"seeding_threshold", "stopping_fa", "stopping_threshold", "step_length", "record_length",
                              "max_half_length", "Qm", "Ql", "Rs", "tie_angle", "tie_sw2"})
CPU_BATCH = 1024                 # half-fibers per batch on the CPU: cache-sized (ukf_cpu_check.py)
GPU_BATCH = 200_000              # on a GPU: each block steps until its longest fiber stops, so fewer, larger blocks keep it
                                 # fuller - 2-17 % faster than 50,000, the fibers bit for bit the same (a fiber's arithmetic
                                 # does not depend on its batch; gpu_track_profile.py); ~0.4 GB of device memory


def _check_tracker(tracker) -> dict:
    tracker = dict(tracker or {})
    unknown = set(tracker) - TRACKER_SETTINGS
    if unknown:
        raise ValueError(f"track: {sorted(unknown)} not among the tracker's settings {sorted(TRACKER_SETTINGS)}")
    return tracker


def _check_shell(bval, shell):
    """prepare's own check, made before any work: the shell tracked, and a b0."""
    bval = np.asarray(bval)
    if not (np.abs(bval - shell) < 50).any() or not (bval < 50).any():
        raise ValueError(f"prepare: no b = {shell:g} shell or no b0 in this scan (b-values: {sorted(set(np.round(bval).astype(int)))})")


def brain_mask(dwi, bval, timer: Timer, prefix="", device="mps") -> np.ndarray:
    """The mask slot's median-otsu: dipy's median_otsu (median radius 4, 4 passes; tractline.mask) of the mean b0. It is
    prep.prepare's own mask, voxel for voxel: prepare keeps every b0, in order, so the mean is of the same volumes."""
    bval = np.asarray(bval)
    with timer(prefix + "mask"):
        b0 = dwi[..., bval < 50].astype(np.float32).mean(-1)
        return median_otsu(b0, median_radius=4, numpass=4, device=device)[1]


def _ukf(t, timer: Timer, prefix, device, workers, tracker: dict) -> Tractogram:
    with timer(prefix + "load"):                                          # on the host: the seeds' fit in float64 there, the same
        D = U.from_arrays(t.dwi, t.header, t.mask)                         # seeds on every device (on CUDA's float64: +-2 fibers)
    with timer(prefix + "ukf"):
        kind = torch.device(device).type
        if kind == "mps":
            fibers, stats = U.track(D, backend="metal", batch=GPU_BATCH, **tracker)
        elif kind == "cuda":
            fibers, stats = U.track(D, backend="triton_block", batch=GPU_BATCH, **tracker)
        else:
            fibers, stats = U.track(D, dtype=torch.float32, device=device, fast=True, batch=CPU_BATCH,
                                    workers=workers or os.cpu_count() or 1, compile=U.can_compile(), **tracker)
    return Tractogram(fibers, stats, t.mask, t.info)


def track(s, dwi, timer: Timer, prefix="", device="mps", workers=None, shell=2800.0, tracker=None) -> Tractogram:
    """The mask (brain_mask), the tracker's input (prep.prepare) and the tracking. device "mps": the Metal kernel;
    "cuda": the Triton block kernel (ukf_triton_block); "cpu": ukf's steps in float32 with the fast step (ukf._ukf_step_fast: as close to float64 as the binary's operation
    order in float32, ukf_cpu_tune.py), through torch.compile only when TRACTLINE_COMPILE=1 (ukf.can_compile; the
    parent compiles, ~10-80 s, in stats["compile_s"], and each worker then loads it), batches of
    CPU_BATCH on `workers` processes of one thread (default: one per core). Workers are spawned, so a
    script running the CPU path needs an `if __name__ == "__main__":` guard. shell: the b-value tracked
    (prep.prepare; ds001226's 2800, the nearest to TractCloud's training b = 3000). tracker: opt-in settings passed
    on to ukf.track (e.g. {"Rs": 0.1} or {"step_length": 0.6}; its keywords: seeding_threshold, stopping_fa,
    stopping_threshold, step_length, record_length, max_half_length, Qm, Ql, Rs, tie_angle, tie_sw2); None or {} is today's, the ORG
    settings, and the labelers were trained on tractography made with those."""
    if torch.device(device).type == "cpu" and (workers or os.cpu_count() or 1) > 1:
        U.require_main_guard()                                             # before any work, not after it
    tracker = _check_tracker(tracker)
    _check_shell(s.bval, shell)
    mask = brain_mask(dwi, s.bval, timer, prefix, device)
    with timer(prefix + "prep"):
        t = prepare(dwi, s.affine, s.bval, s.bvec, shell=shell, device=device, mask=mask)
    return _ukf(t, timer, prefix, device, workers, tracker)


def default_labeler(device="mps"):
    """The pipeline's labeler: RapidParc (labelers.rapidparc; its released "rapidparc" model). On TractCloud's
    own test split 94.5 % tract accuracy against TractCloud's 92.0 % at its trained context, the steadiest
    across seeds and ~25x faster (NOTES 2026-10-02, "The labelers compared"). TractCloud stays available:
    labelers.tractcloud.Labeler. Needs no TractCloud code: torch and RapidParc's weights."""
    from .labelers.rapidparc import Labeler
    return Labeler(device)


def label(tg: Tractogram, labeler, timer: Timer, prefix="", draw=0, grouping="index") -> Labels:
    """One draw of the labeler on the tractogram, with the cluster log-probabilities. grouping other than "index"
    (RapidParc's hash groupings) keys each streamline by its seed voxel (ensemble.seed_keys)."""
    with timer(prefix + "label"):
        if grouping == "index":
            return labeler(tg.fibers, draws=(draw,), logp=True)
        from .ensemble import seed_keys
        return labeler(tg.fibers, draws=(draw,), logp=True, keys=seed_keys(tg.stats["seed_voxel"]), grouping=grouping)


STAGES = ("field_estimate", "field_apply", "mask", "prep", "load", "ukf", "label")


def _bundles_of(labeler) -> dict:
    """A labeler object as the bundles slot's entry: RapidParc's and TractCloud's with their settings (read from the
    object), anything else as "caller-object" (its settings unknown: the config is then not hashable). By module name, so
    that no labeler module is imported here. The object is trusted beyond what is read: weights loaded into it by hand
    would still hash as its model's; its batch size is not read (the labels do not depend on it)."""
    kind = type(labeler)
    where = f"{kind.__module__}.{kind.__qualname__}"
    if where == "tractline.labelers.rapidparc.Labeler":
        return {"impl": "rapidparc", "model": labeler.model_name}
    if where == "tractline.labelers.tractcloud.Labeler":
        return {"impl": "tractcloud", "upstream": bool(labeler.upstream),
                "exact": "no" if type(labeler.model).__name__ == "MatmulDGCNN" else "yes"}
    name = getattr(labeler, "__qualname__", None)
    return {"impl": "caller-object",
            "class": f"{getattr(labeler, '__module__', kind.__module__)}.{name}" if name else where}


def _labeler(cfg: PipelineConfig, labeler, device):
    """The labeler the bundles slot names: the caller's object when given (checked against the slot), else made here."""
    st = cfg["bundles"]
    if st is None:
        if labeler is not None:
            raise ValueError("execute: a labeler was given, but the bundles slot is off")
        return None
    o = st.opts()
    if labeler is not None:
        got = _bundles_of(labeler)
        if got["impl"] != st.impl or any(o.get(k) != v for k, v in got.items() if k not in ("impl", "class")):
            raise ValueError(f"execute: the labeler given ({got}) is not the bundles slot's ({st.impl}, {o})")
        return labeler
    if st.impl == "rapidparc":
        from .labelers.rapidparc import Labeler
        return Labeler(device, model=o["model"])
    if st.impl == "tractcloud":
        from .labelers.tractcloud import Labeler
        return Labeler(device, exact=o["exact"] == "yes", upstream=o["upstream"])
    raise ValueError(f"execute: bundles {st.impl!r} needs the labeler object itself (labeler=)")


def config_for(labeler=None, shell=2800.0, tracker=None, denoise=None, denoise_order="acquired",
               motion_eddy=None, gibbs=None) -> PipelineConfig:
    """run()'s keywords as a config: the "default" preset with them applied. labeler: None for the preset's (RapidParc),
    or an object (_bundles_of). motion_eddy: None (off), True (the slot's defaults), a model name ("rigid", "linear",
    "quadratic"), or a dict of the slot's options. gibbs: None (off), True (the slot's defaults: slice axis 2), or a
    dict of its options ({"slice_axis": slice_axis_of(s)} for a BIDS series)."""
    changes = {"model": {"shell": float(shell)}}
    tracker = _check_tracker(tracker)
    if tracker:
        changes["track"] = tracker
    if denoise is not None:
        from .denoise import parse
        method, settings = parse(denoise)
        changes["denoise"] = {"impl": method, **settings, "order": denoise_order}
    if labeler is not None:
        changes["bundles"] = _bundles_of(labeler)
    if motion_eddy is not None and motion_eddy is not False:
        opts = {} if motion_eddy is True else ({"model": motion_eddy} if isinstance(motion_eddy, str) else dict(motion_eddy))
        changes["motion_eddy"] = {"impl": "tractline", **opts}
    if gibbs is not None and gibbs is not False:
        changes["gibbs"] = {"impl": "subvoxel-shift", **({} if gibbs is True else dict(gibbs))}
    return PipelineConfig.preset("default").with_(**changes)


def _denoise_spec(st) -> str:
    """The denoise slot's entry as tractline.denoise's name ("mppca:r2", "patch2self:s0")."""
    o = st.opts()
    return f"mppca:r{o['patch_radius']}" if st.impl == "mppca" else f"patch2self:s{o['sketch_seed']}"


def runtime_record(device, workers, labeler) -> dict:
    """What a run's config leaves out because it does not change the outputs (or should not): where and how it ran."""
    import platform
    from . import __version__
    cpu = torch.device(device).type == "cpu"
    return {"tractline": __version__, "torch": torch.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "device": str(device),
            "workers": (workers or os.cpu_count() or 1) if cpu else None,       # the CPU tracker's processes, as used
            "torch_compile": bool(U.can_compile()) if cpu else False,          # TRACTLINE_COMPILE: the CPU tracker compiled
            "labeler": None if labeler is None else f"{type(labeler).__module__}.{type(labeler).__qualname__}"}


@dataclass
class Run:
    correction: Correction
    tractogram: Tractogram
    labels: Labels | None        # None: the bundles slot is off
    config: dict                 # PipelineConfig.to_dict(): the slots, their versions and options, origin, sha256
    runtime: dict                # runtime_record

    def __iter__(self):          # corr, tg, labels = execute(...), as run() returns them
        return iter((self.correction, self.tractogram, self.labels))


def _shell_volumes(dwi, bval, bvec, shell):
    """The b0s and one shell, as prepare keeps them: (dwi, bval, bvec (3, G))."""
    bval = np.asarray(bval, float)
    keep = (bval < 50) | (np.abs(bval - shell) < 50)
    return dwi[..., keep], bval[keep], np.asarray(bvec, float)[:, keep]


def _image_axes(bvec, affine):
    """FSL b-vectors in the image's axes: x flipped when the affine's determinant is positive (FSL's convention, as
    prep.prepare reads it)."""
    g = np.array(bvec, float)
    if np.linalg.det(np.asarray(affine)[:3, :3]) > 0:
        g[0] *= -1
    return g


def ptt_seeds(region, affine, count, rng):
    """`count` seed points (RAS mm), each in a voxel of `region` drawn uniformly (with replacement) and uniform within
    it (the voxel's center +- 0.5)."""
    vox = np.argwhere(region)
    if not len(vox):
        raise ValueError("ptt: no voxel to seed in")
    g = np.random.default_rng(rng)
    p = vox[g.integers(len(vox), size=count)] + g.uniform(-0.5, 0.5, size=(count, 3))
    A = np.asarray(affine, float)
    return p @ A[:3, :3].T + A[:3, 3]


def csd_ptt(s, dwi, mask, cfg: PipelineConfig, timer: Timer, prefix="", device="mps", tissue=None) -> Tractogram:
    """The model slot's csd and the track slot's ptt (docs/sota-pipeline-design.md, section 7.2): the b0s and the shell
    (b-vectors in image axes), the response and the FODs over the mask (tractline.csd), then PTT (tractline.ptt: the
    Metal kernel on mps, the torch prototype elsewhere) with ACT from `tissue` (stopping "act": (X, Y, Z, 4) partial
    volumes WM, GM, deep GM, CSF on the DWI's grid, as the bench's tissue maps; GM and the background end a path
    validly, CSF invalidly, the field of view's faces validly) or the mask's edge (stopping "mask"), seeded uniformly in
    WM >= 0.5 (seeding "wm") or in the mask where FA >= fa_seed ("fa"). The PMF is read on our own 362 directions."""
    from . import csd as C, fields as F, ptt as PT
    m, o = cfg.opts("model"), cfg.opts("track")
    d, bval, bvec = _shell_volumes(dwi, s.bval, s.bvec, m["shell"])
    g = _image_axes(bvec, s.affine)
    with timer(prefix + "csd"):
        resp = C.response_ssst(d, bval, g, roi_radius=m["roi_radius"], fa_threshold=m["fa_threshold"])
        order = int(m["order"]) if m["order"] != "auto" else (8 if int((bval > 50).sum()) >= 45 else 6)
        coeffs, cinfo = C.CSD(bval, g, resp, order=order, lambda_=m["lambda"], tau=m["tau"]).fit(d, mask, device=device)
    with timer(prefix + "ptt"):
        if o["stopping"] == "act":
            pve = np.asarray(tissue, float)
            include = pve[..., 1].copy()
            include[pve.sum(-1) < 1e-6] = 1.0
            for ax in range(3):                                            # the field of view's faces: a valid end
                for idx in (0, -1):
                    sl = [slice(None)] * 3
                    sl[ax] = idx
                    include[tuple(sl)] = 1.0
            exclude = pve[..., 3].copy()
        else:
            include, exclude = (~mask).astype(float), np.zeros(mask.shape)
        if o["seeding"] == "wm":
            region = np.asarray(tissue)[..., 0] >= 0.5
        else:
            ev, _ = C.tensor_fit(d[mask].astype(np.float64), bval, g)
            region = np.zeros(mask.shape, bool)
            region[mask] = C.fa(ev) >= o["fa_seed"]
        seeds = ptt_seeds(region, s.affine, o["seed_count"], o["seed_rng"])
        v = C.hemisphere(181)
        pmf = F.PmfField(coeffs, mask, np.r_[v, -v], device)
        act = F.Act(include, exclude, device)
        from . import ptt_metal
        backend = "metal" if torch.device(device).type == "mps" and ptt_metal.available() else "torch"
        if torch.device(device).type == "mps" and backend == "torch":
            import warnings
            warnings.warn("ptt: the Metal kernel is unavailable (torch.mps.compile_shader); the torch prototype runs, "
                          "slower and with another random stream", stacklevel=2)
        params = PT.Params(step_size=o["step_size"], max_angle=o["max_angle"], probe_length=o["probe_length"],
                           probe_quality=o["probe_quality"], pmf_threshold=o["pmf_threshold"], min_len=o["min_len"],
                           max_len=o["max_len"])
        tracker = PT.Tracker(pmf, act, s.affine, np.linalg.norm(np.asarray(s.affine)[:3, :3], axis=0), params, device,
                             backend=backend)
        by_seed, tinfo = tracker.track(seeds, seed=o["track_rng"], keep_index=True)
    order_ = sorted(by_seed)
    fibers = [by_seed[i] for i in order_]
    stats = {"seeds": int(len(seeds)), "seed_index": np.asarray(order_, np.int64), "backend": backend,
             "random numbers": ("Philox, counter-based, keyed by the seed's index" if backend == "metal"
                                else "a torch generator, its stream depending on the batch (20,000 seeds)"),
             "seed voxels in the seeding region": int(region.sum()), **tinfo}
    info = {"shell": m["shell"], "directions": int((bval > 50).sum()), "order": order,
            "response": {"evals": resp.evals.tolist(), "S0": resp.S0, "voxels": resp.voxels}, "csd": cinfo,
            "mask_voxels": int(mask.sum())}
    return Tractogram(fibers, stats, mask, info)


def execute(s, cfg: PipelineConfig, timer: Timer, prefix="", device="mps", workers=None, labeler=None, trx=None,
            release=True, tissue=None, **trx_options) -> Run:
    """The pipeline as cfg configures it (tractline.config), on subject s. Every check runs before any work: the
    config's own (at its construction), the shell tracked, the CPU's main guard, the labeler against the bundles slot.
    Then, in SLOTS' order: denoise (when on; order "acquired" before the field, "corrected" after it), field (when on
    and the subject has a pair: correct; off: the DWI as acquired, the reason in Correction.note), motion_eddy (when on:
    correct_motion_eddy in place of correct - the field estimate, then head motion and eddy currents, then one resample;
    the rotated b-vectors to the tracker), mask, model (the
    tracker's input), track, bundles (when on). gibbs (when on) runs before the field (after denoising, in denoise's
    order "acquired"), on the series as acquired and the b0s (gibbs_subject; its b0 counts in the Correction's gibbs);
    its time is timer.seconds["gibbs"] (like "denoise", not among pipeline_stages). labeler: an object for the bundles
    slot (needed for "caller-object"; otherwise made from the slot, or the caller's checked against it). device, workers, trx and release as run()'s:
    runtime, recorded beside the config, not in it. Times: timer.total(*pipeline_stages(prefix, cfg)), and "denoise".
    tissue: (X, Y, Z, 4) partial volumes (WM, GM, deep GM, CSF) on the DWI's grid, for track:ptt's stopping "act" and
    seeding "wm" (until the anatomy slots make them: section 7.2); an input like the DWI, recorded in the runtime record
    by its shape and sha256, not in the config."""
    if not isinstance(cfg, PipelineConfig):
        raise TypeError(f"execute: cfg is a {type(cfg).__name__}, not a PipelineConfig")
    if not cfg.is_on("track"):
        raise ValueError("execute: the track slot is off; there is nothing to run without it yet")
    shell, tracker = cfg.opts("model")["shell"], cfg.opts("track")
    _check_shell(s.bval, shell)
    ptt = cfg["track"].impl == "ptt"
    uses_tissue = ptt and (tracker["stopping"] == "act" or tracker["seeding"] == "wm")
    if uses_tissue:
        if tissue is None:
            raise ValueError("execute: track:ptt with stopping 'act' or seeding 'wm' needs tissue= (WM, GM, deep GM, CSF)")
        if np.shape(tissue)[:3] != np.shape(s.dwi)[:3] or np.ndim(tissue) != 4 or np.shape(tissue)[3] < 4:
            raise ValueError(f"execute: tissue of shape {np.shape(tissue)}: (X, Y, Z, 4) on the DWI's grid expected")
    elif tissue is not None:
        raise ValueError("execute: tissue= given, but this config does not use it (track:ptt with stopping 'act' or "
                         "seeding 'wm' does)")
    if ptt and cfg.is_on("bundles") and cfg.opts("bundles").get("grouping", "index") != "index":
        raise ValueError("execute: track:ptt has no seed voxels (a voxel may seed several streamlines), so the bundles "
                         "slot's grouping must be 'index'")
    if trx is not None and not cfg.is_on("bundles"):
        raise ValueError("execute: TRX output needs labels, and the bundles slot is off")
    if trx is not None:
        Path(trx).parent.mkdir(parents=True, exist_ok=True)               # before the work, not after it
    if torch.device(device).type == "cpu" and (workers or os.cpu_count() or 1) > 1:
        U.require_main_guard()                                             # before the 20 s field estimate
    labeler = _labeler(cfg, labeler, device)
    dn = cfg["denoise"]
    spec, order, dstats = (None, None, None) if dn is None else (_denoise_spec(dn), dn.opts()["order"], None)
    if order == "acquired":
        s, dstats = denoise_subject(s, timer, spec, workers)
    gibbs_b0s = None
    if cfg.is_on("gibbs"):                                     # before the field (after denoising "acquired": MRtrix's order)
        s, gibbs_b0s = gibbs_subject(s, timer, cfg.opts("gibbs"), device)
    with exact_float32(device):
        if cfg.is_on("motion_eddy"):
            corr = correct_motion_eddy(s, timer, device, cfg.opts("motion_eddy"), field=cfg.is_on("field"))
        elif cfg.is_on("field"):
            corr = correct(s, timer, device)
        else:
            timer.skip("field_estimate"); timer.skip("field_apply")
            corr = Correction(np.array(s.dwi, np.float32), None, None, None, applied=False,
                              note="not corrected: the field slot is off")
        if order == "corrected":
            corr.dwi, dstats = denoise_corrected(corr.dwi, s.bval, timer, spec, workers)
        corr.denoise, corr.gibbs = dstats, gibbs_b0s
        mask = brain_mask(corr.dwi, s.bval, timer, prefix, device)
        st, tdwi = s, corr.dwi
        if corr.bvec is not None:                                          # the b-vectors turned with the head
            st = copy.copy(s)
            st.bvec = corr.bvec
            left = [int(v) for v in (corr.motion_eddy or {}).get("left out", {})]
            if left:                                                       # not tracked: not registered, or corrupt
                use = np.setdiff1d(np.arange(len(s.bval)), left)
                st.bval, st.bvec, tdwi = np.asarray(s.bval)[use], corr.bvec[:, use], corr.dwi[..., use]
        if ptt:
            tg = csd_ptt(st, tdwi, mask, cfg, timer, prefix, device, tissue)
        else:
            with timer(prefix + "prep"):
                t = prepare(tdwi, s.affine, st.bval, st.bvec, shell=shell, device=device, mask=mask)
            tg = _ukf(t, timer, prefix, device, workers, tracker)
        if labeler is None:
            labels = None
            timer.skip(prefix + "label")
        else:
            b = cfg.opts("bundles")
            labels = label(tg, labeler, timer, prefix, draw=b.get("draw", 0), grouping=b.get("grouping", "index"))
    if trx is not None:
        from . import trx as _trx
        with timer(prefix + "write_trx"):
            _trx.write(trx, s, tg, labels, **trx_options)
    if release:
        release_memory(device)
    config, runtime = cfg.to_dict(), runtime_record(device, workers, labeler)
    if uses_tissue:
        import hashlib
        tz = np.ascontiguousarray(tissue, np.float32)
        runtime["tissue"] = {"shape": list(tz.shape), "sha256 (float32)": hashlib.sha256(tz.tobytes()).hexdigest()}
    if corr.motion_eddy is not None:
        runtime["motion_eddy backend"] = corr.motion_eddy["backend"]
    if ptt:                                                                # the CPU tracker's settings do not apply
        runtime.pop("workers", None)
        runtime.pop("torch_compile", None)
        runtime["ptt backend"] = tg.stats["backend"]
        runtime["ptt random numbers"] = tg.stats["random numbers"]
    corr.provenance = {"config": config, "runtime": runtime}
    return Run(corr, tg, labels, config, runtime)


def run(s, labeler, timer: Timer, prefix="", trx=None, device="mps", workers=None, release=True, shell=2800.0, tracker=None,
        denoise=None, denoise_order="acquired", config=None, motion_eddy=None, gibbs=None, **trx_options):
    """The whole pipeline on a subject: (Correction, Tractogram, Labels). labeler: None for the default
    (default_labeler: RapidParc), or any labeler (labelers.tractcloud.Labeler, ...). Its time is
    timer.total(*pipeline_stages(prefix)). trx: also write the tractogram there as TRX (trx.write;
    trx_options: positions= "float32" (default) or "float16", labeled_only= False (default) or True),
    timed apart as prefix + "write_trx". release: return
    the GPU allocator's cache after the run (release_memory: batches stay flat). device: where
    the estimate and the tracking run ("mps", "cuda" or "cpu"; the labeler has its own); shell: the b-value
    tracked (track); workers: the CPU
    tracker's processes (default one per core as os.cpu_count() sees them - in a container, pass the
    container's own); tracker: opt-in tracker settings (track), None for today's. denoise: opt-in, None for today's,
    or "mppca" / "patch2self" (denoise_subject; "mppca:r3" with a setting), its settings and estimates in Correction.denoise and its time in
    timer.seconds["denoise"] (not among pipeline_stages), and `workers` its processes too; denoise_order: "acquired" (before the field; the default)
    or "corrected" (denoise_corrected, for comparison). motion_eddy: opt-in head motion and eddy-current correction
    (correct_motion_eddy; config_for's values), its parameters and QC in Correction.motion_eddy, the rotated b-vectors
    in Correction.bvec. gibbs: opt-in Gibbs ringing removal (gibbs_subject; config_for's values), first, on the
    series as acquired; its time in timer.seconds["gibbs"] (not among pipeline_stages). config: a PipelineConfig
    instead of shell, tracker, denoise, denoise_order, motion_eddy and gibbs (which must then be left as they are).
    Either way the run goes through execute, the config and runtime recorded in Correction.provenance."""
    if motion_eddy is False:
        motion_eddy = None
    if gibbs is False:
        gibbs = None
    if config is None:
        if denoise is not None and denoise_order not in DENOISE_ORDERS:
            raise ValueError(f"denoise_order: {denoise_order!r} is not one of {DENOISE_ORDERS}")
        config = config_for(labeler, shell, tracker, denoise, denoise_order, motion_eddy, gibbs)
    elif (shell, tracker or None, denoise, denoise_order, motion_eddy, gibbs) != (2800.0, None, None, "acquired", None,
                                                                                    None):
        raise ValueError("run: with config=, set shell, tracker, denoise, motion_eddy and gibbs in the config, not as "
                         "keywords too")
    return tuple(execute(s, config, timer, prefix, device, workers, labeler, trx, release, **trx_options))


PTT_STAGES = ("field_estimate", "field_apply", "mask", "csd", "ptt", "label")


def pipeline_stages(prefix="", cfg: PipelineConfig = None):
    """The timer's stage names for a run: today's UKF path, or (cfg with track:ptt) the CSD and PTT path's."""
    names = PTT_STAGES if cfg is not None and cfg.is_on("track") and cfg["track"].impl == "ptt" else STAGES
    if cfg is not None and cfg.is_on("motion_eddy"):
        names = names[:1] + ("motion_eddy",) + names[1:]
    return [n if n.startswith("field_") or n == "motion_eddy" else prefix + n for n in names]
