"""The pipeline, scan to labeled streamlines, in memory: correct -> track -> label [-> TRX].

    s = _ds001226.load("PAT16"); timer = Timer()      # the bench's loader; any object with its fields
    corr, tg, labels = run(s, None, timer)                  # None: the default labeler (RapidParc)

Stages, and where they run:
  correct   the susceptibility field from the b0s and their reversed pair (susceptibility.estimate: GPU,
            float32, ~29 s), then applied to the DWI (susceptibility.apply: CPU, float64, ~1 s)
  track     the tracker's input (prep.prepare: b0s + the b = 2800 shell, gradients in RAS, the
            median_otsu mask, exact in torch on the pipeline's device; ~3 s), UKF two-tensor with the ORG settings and the binary's seeds
            (ukf.track, Metal kernel: float32 steps from float64 seeds, ~27 s; or on the CPU,
            one process per core, fast algebra)
  label     RapidParc, one draw (labelers.rapidparc: shuffled groups of 2,000 streamlines; GPU or CPU,
            float32, ~1 s); TractCloud (labelers.tractcloud, at its trained context) on request
  TRX       optional: the tractogram with labels and tract probabilities (trx.write)
Dependencies of the default path: numpy, scipy, torch, nibabel, RapidParc's weights (no TractCloud code).

Conventions at every module boundary (_ds001226, susceptibility, prep, ukf, the labelers, t1check):
  volumes   numpy (X, Y, Z[, V]) in the NIfTI's voxel order; affine voxel -> RAS mm; voxel sizes as
            the header states them (the affine's column norms differ by ~1e-8)
  fields    Hz on that grid; displacements mm along the phase-encoding axis, signed, as the DWI's
            own phase encoding moves the anatomy (susceptibility.displacement_mm)
  fibers    lists of (n, 3) RAS mm arrays, in seed order; labels for those >= 40 mm (Labels.keep)
  devices   device="mps" | "cuda" | "cpu"; timings through Timer, which waits for the GPU
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np, torch
from . import susceptibility as S
from . import ukf as U
from .prep import prepare
from .labelers.base import Labels

ESTIMATE_THREADS = 16            # the field estimate on the CPU: its problem is small (~5e5 voxels); on 48 vCPUs
                                 # 8 threads 43 s, 16 36.5 s, 32 41 s, 48 72.5 s (modal_cpu_field_timing.json)


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
    after. With TF32, TractCloud on an A10 labeled 62.6 % of PAT16's streamlines Other against 59.9 %
    without, and the field moved by up to 1 mm; without it, the A10 labels a tractogram as the M2 does,
    every streamline, at no cost in time (NOTES 2026-10-02, "The pipeline on CUDA")."""
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

    def total(self, *names):
        return round(sum(self.seconds[n] for n in names), 1)


@dataclass
class Correction:
    dwi: np.ndarray              # (X, Y, Z, G) float32, corrected
    field_hz: np.ndarray         # (X, Y, Z)
    motion: np.ndarray           # (V, 6) per b0: translations mm, rotations rad
    displacement_mm: np.ndarray  # (X, Y, Z): what the field did to the DWI as acquired


@dataclass
class Tractogram:
    fibers: list                 # (n, 3) RAS mm arrays, seed order
    stats: dict                  # ukf.track's: seeds, half_fibers, fiber_steps, fibers, seed_index
    mask: np.ndarray             # (X, Y, Z) bool, the tracking mask
    info: dict                   # prep's


def correct(s, timer: Timer, device="mps") -> Correction:
    """The field estimate is susceptibility.estimate's defaults (Gauss-Newton, NOTES 2026-10-02): on "mps" its
    subsampled levels on the CPU, on the CPU and on "cuda" in float32 (all levels on the GPU there) -
    the same model on each, 18 s / 22.5 s on the M2. The CPU's levels get ESTIMATE_THREADS."""
    import os
    f32 = torch.device(device).type in ("cpu", "cuda")
    with timer("field_estimate"), threads(min(ESTIMATE_THREADS, os.cpu_count() or 1)):
        h, motion, _ = S.estimate(s.b0s, s.vox, s.pe_vectors, s.readout_s, device=device, **(dict(dtype=torch.float32) if f32 else {}))
    with timer("field_apply"):
        dwi = S.apply(s.dwi, h, s.pe_axis, s.pe_sign, s.readout_s)
    return Correction(dwi, h, motion, S.displacement_mm(h, s.readout_s, s.pe_sign, s.vox[s.pe_axis]))


CPU_BATCH = 1024                 # half-fibers per batch on the CPU: cache-sized (ukf_cpu_check.py)


def track(s, dwi, timer: Timer, prefix="", device="mps", workers=None, shell=2800.0) -> Tractogram:
    """device "mps": the Metal kernel; "cuda": the Triton block kernel (ukf_triton_block); "cpu": ukf's
    steps in float32 with the fast algebra (closer to float64 than the binary's operation order in
    float32: ukf_cpu_check.json), batches of
    CPU_BATCH on `workers` processes of one thread (default: one per core). Workers are spawned, so a
    script running the CPU path needs an `if __name__ == "__main__":` guard. shell: the b-value tracked
    (prep.prepare; ds001226's 2800, the nearest to TractCloud's training b = 3000)."""
    with timer(prefix + "prep"):
        t = prepare(dwi, s.affine, s.bval, s.bvec, shell=shell, device=device)
    with timer(prefix + "load"):
        D = U.from_arrays(t.dwi, t.header, t.mask)
    with timer(prefix + "ukf"):
        if device == "mps":
            fibers, stats = U.track(D, backend="metal")
        elif torch.device(device).type == "cuda":
            fibers, stats = U.track(D, backend="triton_block")
        else:
            import os
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


def run(s, labeler, timer: Timer, prefix="", trx=None, device="mps", workers=None, release=True, **trx_options):
    """The whole pipeline on a subject: (Correction, Tractogram, Labels). labeler: None for the default
    (default_labeler: RapidParc), or any labeler (labelers.tractcloud.Labeler, ...). Its time is
    timer.total(*pipeline_stages(prefix)). trx: also write the tractogram there as TRX (trx.write;
    trx_options: positions="float16", labeled_only=True), timed apart as "write_trx". release: return
    the GPU allocator's cache after the run (release_memory: batches stay flat). device: where
    the estimate and the tracking run ("mps" or "cpu"; the labeler has its own); workers: the CPU
    tracker's processes (default one per core as os.cpu_count() sees them - in a container, pass the
    container's own)."""
    labeler = labeler if labeler is not None else default_labeler(device)
    with exact_float32(device):
        corr = correct(s, timer, device)
        tg = track(s, corr.dwi, timer, prefix, device, workers)
        labels = label(tg, labeler, timer, prefix)
    if trx is not None:
        from . import trx as _trx
        with timer("write_trx"):
            _trx.write(trx, s, tg, labels, **trx_options)
    if release:
        release_memory(device)
    return corr, tg, labels


def pipeline_stages(prefix=""):
    return [n if n.startswith("field_") else prefix + n for n in STAGES]
