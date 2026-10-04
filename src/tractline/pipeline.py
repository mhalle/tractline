"""The pipeline, scan to labeled streamlines, in memory: correct -> track -> label [-> TRX].

    s = bids.load("sub-01/dwi/sub-01_dwi.nii.gz"); timer = Timer()   # any BIDS series (tractline.bids)
    corr, tg, labels = run(s, None, timer, device="mps")   # None: the default labeler (RapidParc)
    print(corr.note, corr.warnings)                         # what corrected it, or why nothing did

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
  correct   the susceptibility field from the b0s and their reversed pair (susceptibility.estimate: GPU,
            float32, ~18 s on the M2), then applied to the DWI (susceptibility.apply: CPU, float64, ~1 s);
            residual_left: how much of the pair's difference the field leaves - above RESIDUAL_WARN a
            warning (Correction.warnings), never a refusal. Skipped when there is no pair (can_correct).
  track     the tracker's input (prep.prepare: b0s + the b = 2800 shell, gradients in RAS, the
            median_otsu mask, exact in torch on the pipeline's device; ~3 s), UKF two-tensor with the
            ORG settings and the binary's seeds (ukf.track: on "mps" the Metal kernel, float32 steps
            from float64 seeds, ~27 s; on "cuda" the Triton block kernel; on the CPU one process per
            core, fast algebra)
  label     RapidParc, one draw (labelers.rapidparc: shuffled groups of 2,000 streamlines; GPU or CPU,
            float32, ~1 s); TractCloud (labelers.tractcloud, at its trained context) on request
  TRX       optional: the tractogram with labels and tract probabilities (trx.write)
Dependencies of the default path: numpy, torch and RapidParc's weights - no scipy, no nibabel (reading the
scan is the caller's), no TractCloud code (bench/dependency_check.py).

Conventions at every module boundary (_ds001226, susceptibility, prep, ukf, the labelers, t1check):
  volumes   numpy (X, Y, Z[, V]) in the NIfTI's voxel order; affine voxel -> RAS mm; voxel sizes as
            the header states them (the affine's column norms differ by ~1e-8)
  fields    Hz on that grid; displacements mm along the phase-encoding axis, signed, as the DWI's
            own phase encoding moves the anatomy (susceptibility.displacement_mm)
  fibers    lists of (n, 3) RAS mm arrays, in seed order; labels for those >= 40 mm (Labels.keep)
  devices   device="mps" | "cuda" | "cpu"; timings through Timer, which waits for the GPU
"""
from __future__ import annotations

import os, time
from contextlib import contextmanager
from dataclasses import dataclass, field
import numpy as np, torch
from . import susceptibility as S
from . import ukf as U
from .prep import prepare
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
    """Named stage times (s); with `echo`, each printed as it ends, prefixed by echo."""
    def __init__(self, echo=None):
        self.seconds, self.echo = {}, echo

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
            print(f"{self.echo} {name} {self.seconds[name]} s", flush=True)

    def skip(self, name):
        """A stage that did not run: recorded as 0 s, so totals over it stay exact and a misspelled name still fails."""
        self.seconds[name] = 0.0
        if self.echo is not None:
            print(f"{self.echo} {name} skipped", flush=True)

    def total(self, *names):
        return round(sum(self.seconds[n] for n in names), 1)


@dataclass
class Correction:
    dwi: np.ndarray              # (X, Y, Z, G) float32, corrected (as acquired when not applied)
    field_hz: np.ndarray | None  # (X, Y, Z); None when not applied
    motion: np.ndarray | None    # (V, 6) per b0: translations mm, rotations rad; None when not applied
    displacement_mm: np.ndarray | None   # (X, Y, Z): what the field did to the DWI as acquired
    applied: bool = True
    note: str = ""               # what corrected it, or why nothing did (bids.Pairing.message)
    residual_left: float | None = None   # the pair's mean-b0 difference left after correction (residual_left)
    warnings: list = field(default_factory=list)


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
    pairing = getattr(s, "pairing", None)
    if not can_correct(s):
        timer.skip("field_estimate"); timer.skip("field_apply")
        note = (pairing.message if pairing is not None and not pairing.ok
                else "not corrected: no b0s from both phase-encoding polarities")
        return Correction(np.array(s.dwi, np.float32), None, None, None, applied=False, note=note)
    f32 = torch.device(device).type in ("cpu", "cuda")
    ro = getattr(s, "b0_readout_s", None)
    with timer("field_estimate"), threads(min(ESTIMATE_THREADS, os.cpu_count() or 1)):
        h, motion, _ = S.estimate(s.b0s, s.vox, s.pe_vectors, s.readout_s if ro is None else ro, device=device,
                                  **(dict(dtype=torch.float32) if f32 else {}))
    with timer("field_apply"):
        dwi = S.apply(s.dwi, h, s.pe_axis, s.pe_sign, s.readout_s)
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
    return Correction(dwi, h, motion, S.displacement_mm(h, s.readout_s, s.pe_sign, s.vox[s.pe_axis]), applied=True,
                      note=note, residual_left=None if left is None else round(left, 3), warnings=warnings)


CPU_BATCH = 1024                 # half-fibers per batch on the CPU: cache-sized (ukf_cpu_check.py)


def track(s, dwi, timer: Timer, prefix="", device="mps", workers=None, shell=2800.0) -> Tractogram:
    """device "mps": the Metal kernel; "cuda": the Triton block kernel (ukf_triton_block); "cpu": ukf's
    steps in float32 with the fast algebra (closer to float64 than the binary's operation order in
    float32: ukf_cpu_check.json), batches of
    CPU_BATCH on `workers` processes of one thread (default: one per core). Workers are spawned, so a
    script running the CPU path needs an `if __name__ == "__main__":` guard. shell: the b-value tracked
    (prep.prepare; ds001226's 2800, the nearest to TractCloud's training b = 3000)."""
    if torch.device(device).type == "cpu" and (workers or os.cpu_count() or 1) > 1:
        U.require_main_guard()                                             # before any work, not after it
    with timer(prefix + "prep"):
        t = prepare(dwi, s.affine, s.bval, s.bvec, shell=shell, device=device)
    with timer(prefix + "load"):
        D = U.from_arrays(t.dwi, t.header, t.mask)
    with timer(prefix + "ukf"):
        kind = torch.device(device).type
        if kind == "mps":
            fibers, stats = U.track(D, backend="metal")
        elif kind == "cuda":
            fibers, stats = U.track(D, backend="triton_block")
        else:
            fibers, stats = U.track(D, dtype=torch.float32, device=device, fast=True, batch=CPU_BATCH,
                                    workers=workers or os.cpu_count() or 1)
    return Tractogram(fibers, stats, t.mask, t.info)


def default_labeler(device="mps"):
    """The pipeline's labeler: RapidParc (labelers.rapidparc; its released "rapidparc" model). On TractCloud's
    own test split 94.5 % tract accuracy against TractCloud's 92.0 % at its trained context, the steadiest
    across seeds and ~25x faster (NOTES 2026-10-02, "The labelers compared"). TractCloud stays available:
    labelers.tractcloud.Labeler. Needs no TractCloud code: torch and RapidParc's weights."""
    from .labelers.rapidparc import Labeler
    return Labeler(device)


def label(tg: Tractogram, labeler, timer: Timer, prefix="") -> Labels:
    with timer(prefix + "label"):
        return labeler(tg.fibers, draws=(0,), logp=True)


STAGES = ("field_estimate", "field_apply", "prep", "load", "ukf", "label")


def run(s, labeler, timer: Timer, prefix="", trx=None, device="mps", workers=None, release=True, shell=2800.0, **trx_options):
    """The whole pipeline on a subject: (Correction, Tractogram, Labels). labeler: None for the default
    (default_labeler: RapidParc), or any labeler (labelers.tractcloud.Labeler, ...). Its time is
    timer.total(*pipeline_stages(prefix)). trx: also write the tractogram there as TRX (trx.write;
    trx_options: positions= "float32" (default) or "float16", labeled_only= False (default) or True),
    timed apart as prefix + "write_trx". release: return
    the GPU allocator's cache after the run (release_memory: batches stay flat). device: where
    the estimate and the tracking run ("mps", "cuda" or "cpu"; the labeler has its own); shell: the b-value
    tracked (track); workers: the CPU
    tracker's processes (default one per core as os.cpu_count() sees them - in a container, pass the
    container's own)."""
    if torch.device(device).type == "cpu" and (workers or os.cpu_count() or 1) > 1:
        U.require_main_guard()                                             # before the 20 s field estimate
    labeler = labeler if labeler is not None else default_labeler(device)
    with exact_float32(device):
        corr = correct(s, timer, device)
        tg = track(s, corr.dwi, timer, prefix, device, workers, shell)
        labels = label(tg, labeler, timer, prefix)
    if trx is not None:
        from . import trx as _trx
        with timer(prefix + "write_trx"):
            _trx.write(trx, s, tg, labels, **trx_options)
    if release:
        release_memory(device)
    return corr, tg, labels


def pipeline_stages(prefix=""):
    return [n if n.startswith("field_") else prefix + n for n in STAGES]
