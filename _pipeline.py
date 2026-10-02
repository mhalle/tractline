"""The pipeline, scan to payload, in memory: correct -> track -> label -> encode.

    s = _ds001226.load("PAT16"); timer = Timer()
    corr, tg, labels, payload = run(s, Labeler(), timer)

Stages, and where they run:
  correct   the susceptibility field from the b0s and their reversed pair (_susc.estimate: GPU,
            float32, ~29 s), then applied to the DWI (_susc.apply: CPU, float64, ~1 s)
  track     the tracker's input (_prep.prepare: b0s + the b = 2800 shell, gradients in RAS, the
            median_otsu mask; CPU, <1 s), UKF two-tensor with the ORG settings and the binary's seeds
            (_ukf_torch.track, Metal kernel: float32 steps from float64 seeds, ~27 s)
  label     TractCloud, one context draw (_tractcloud.Labeler: GPU, float32, ~4 s)
  encode    the rank field of the cluster log-probabilities (rankfield: depth 6, keep "clip", clip 8)
            and the geometry (a 0.05 mm grid, second-order prediction, int8 residuals; _geometry)

Conventions at every module boundary (_ds001226, _susc, _prep, _ukf_torch, _tractcloud, _t1check):
  volumes   numpy (X, Y, Z[, V]) in the NIfTI's voxel order; affine voxel -> RAS mm; voxel sizes as
            the header states them (the affine's column norms differ by ~1e-8)
  fields    Hz on that grid; displacements mm along the phase-encoding axis, signed, as the DWI's
            own phase encoding moves the anatomy (_susc.displacement_mm)
  fibers    lists of (n, 3) RAS mm arrays, in seed order; labels for those >= 40 mm (Labels.keep)
  devices   device="mps" | "cpu"; timings through Timer, which waits for the GPU
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
import numpy as np, torch
import _susc as S
import _ukf_torch as U
from _prep import prepare
from _tractcloud import Labeler, Labels

GEOMETRY_GRID_MM = 0.05
FIELD = dict(keep="clip", depth=6, clip=8.0, tail_temperatures=(1.0,))


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
    stats: dict                  # _ukf_torch.track's: seeds, half_fibers, fiber_steps, fibers, seed_index
    mask: np.ndarray             # (X, Y, Z) bool, the tracking mask
    info: dict                   # _prep's


@dataclass
class Payload:
    field_bytes: int             # the rank field's arrays, raw
    field_compressed_bytes: int  # the same, Blosc zstd 9
    geometry_bytes: int          # positions (0.05 mm grid, second order, compressed) + streamline lengths
    streamlines: int

    @property
    def total_mb(self):
        return round((self.field_compressed_bytes + self.geometry_bytes) / 1e6, 2)


def correct(s, timer: Timer, device="mps") -> Correction:
    with timer("field_estimate"):
        h, motion, _ = S.estimate(s.b0s, s.vox, s.pe_vectors, s.readout_s, device=device)
    with timer("field_apply"):
        dwi = S.apply(s.dwi, h, s.pe_axis, s.pe_sign, s.readout_s)
    return Correction(dwi, h, motion, S.displacement_mm(h, s.readout_s, s.pe_sign, s.vox[s.pe_axis]))


def track(s, dwi, timer: Timer, prefix="") -> Tractogram:
    with timer(prefix + "prep"):
        t = prepare(dwi, s.affine, s.bval, s.bvec)
    with timer(prefix + "load"):
        D = U.from_arrays(t.dwi, t.header, t.mask)
    with timer(prefix + "ukf"):
        fibers, stats = U.track(D, backend="metal")
    return Tractogram(fibers, stats, t.mask, t.info)


def label(tg: Tractogram, labeler: Labeler, timer: Timer, prefix="") -> Labels:
    with timer(prefix + "tractcloud"):
        return labeler(tg.fibers, draws=(0,), logp=True)


def encode(tg: Tractogram, labels: Labels, timer: Timer, prefix="") -> Payload:
    import rankfield as rf
    from numcodecs import Blosc
    from _geometry import encode as gencode, lengths_bytes
    with timer(prefix + "encode_field"):
        code = rf.encode(labels.logp.T[:, :, None, None], **FIELD)
        arrays = [np.ascontiguousarray(np.asarray(a)) for a in (code.ranks, code.support, code.tail)]
        raw = sum(a.nbytes for a in arrays)
        packed = sum(len(Blosc(cname="zstd", clevel=9, shuffle=Blosc.SHUFFLE).encode(a)) for a in arrays)
    with timer(prefix + "encode_geometry"):
        kept = labels.kept(tg.fibers)
        P = np.concatenate(kept); off = np.r_[0, np.cumsum([len(f) for f in kept])]
        nb, _ = gencode(P, off, GEOMETRY_GRID_MM, P.min(0))
        geometry = nb + lengths_bytes(off)
    return Payload(int(raw), int(packed), int(geometry), len(kept))


STAGES = ("field_estimate", "field_apply", "prep", "load", "ukf", "tractcloud", "encode_field", "encode_geometry")


def run(s, labeler: Labeler, timer: Timer, prefix=""):
    """The whole pipeline on a subject: (Correction, Tractogram, Labels, Payload). Its time is
    timer.total(*pipeline_stages(prefix))."""
    corr = correct(s, timer)
    tg = track(s, corr.dwi, timer, prefix)
    labels = label(tg, labeler, timer, prefix)
    return corr, tg, labels, encode(tg, labels, timer, prefix)


def pipeline_stages(prefix=""):
    return [n if n.startswith("field_") else prefix + n for n in STAGES]
