"""Parallel transport tractography (Aydogan & Shi 2021) with the anatomical constraint (ACT), vectorized in torch: phase D's
prototype (docs/sota-pipeline-design.md, section 7.2). Every half-streamline of a batch steps in lockstep on the device;
a Metal and a Triton kernel come after, on the same samplers (tractline.fields).

The oracle is DIPY 1.12's ptt_tracking, as the step 6 benchmark ran it. Its code (propspeed.pyx, tractogen.pyx,
tracker_parameters.pyx) was read to understand it; this is our own implementation. Reading it found three places where
DIPY 1.12 (and its master, 2026-10-24) mixes voxel and millimeter units, all of which matter at 2 mm voxels:
  1. the probe's steps are multiplied by the voxel size where the tracking step divides by it: the probe reaches
     voxel_size^2 times its stated length (1.5 mm stated: 6.6 mm at 2.1 mm voxels);
  2. the curvature bound is computed from the step in voxels, 2 sin(max_angle / 2) / (step / mean voxel size), and applied
     to steps in mm: the turn per step, k s, can reach 2 x mean voxel size x sin(max_angle / 2) radians (21 degrees at
     2.1 mm for 10 stated);
  3. each recorded point but the seed (and the stopping test there) is the tracker's position plus one step along the
     half's starting direction (the seed's; in the backward half, minus the first forward segment): a constant shift, so
     the two segments next to the seed are about two steps long.
This tracker works in real units: the probe and the curvature bound in mm, the points where the tracker is. For
validation only, Tracker(..., _dipy_compat=True) reproduces DIPY's three (the oracle as it ran, the step 6 benchmark's
PTT), so that the rest of the implementation can be checked against DIPY before the units differ. It is not an option of
the pipeline: no config, preset or calibration map uses it. An independent check (2026-10-24, a synthetic band at 2 mm,
400 seeds, DIPY's own ptt_tracking against this module): kept 382 / 382 (DIPY / compatibility), turn per step p99 16.8 /
16.9 degrees; real units 398 kept, p99 8.6 degrees; at 1 mm voxels DIPY's turns fall to 8.5 degrees (tests/test_ptt.py).

    params = Params()                                            # step 0.5 mm, max angle 10, probe 1.5 mm x 7, threshold 0.1
    tracker = Tracker(pmf_field, act, affine, vox, params, device)
    streamlines, info = tracker.track(seeds_world, seed=1)      # valid ones, RAS mm, min_len..max_len

The method (per half-streamline; voxel coordinates; a frame T, N1, N2 and curvatures k1, k2 in the normal plane):
  propagate by s: position += s T + s^2/2 (k1 N1 + k2 N2) (through 1/voxel size); T' = (1 - (k1^2 + k2^2) s^2/2) T +
    s (k1 N1 + k2 N2), normalized; N2' = -k2 s T - k1 k2 s^2/2 N1 + (1 - k2^2 s^2/2) N2; N1 = unit(N2' x T'), N2 = T' x N1.
    Curvatures below 1e-4 in magnitude become +1e-4 unless both are (then the step is straight);
  a candidate (k1, k2): uniform in the disk of radius max curvature;
  its likelihood: from the current point and frame, probe_quality - 1 sub-steps of probe_length / (probe_quality - 1)
    along the candidate's arc (the frame carried along, not re-orthonormalized), the PMF read along each new tangent
    (values at or below pmf_threshold count 0); L = (last value + the sum) / (quality x count), to the exponent;
  rejection sampling: 10 candidates estimate the maximum, M = (2 max)^exponent; up to 100 more, the first with u M < L
    (u M <= L at initialization) is taken: its curvatures, and its probe's last value as the next "last value";
    none taken: the half stops (an invalid end);
  initialization at the seed: the same, each candidate with T the seed direction and a random N2 perpendicular to it,
    its last value the PMF at the seed along T; none taken: the last candidate stands (as DIPY);
  each step: propagate with the current curvatures, then choose new ones; the point recorded, then ACT decides there:
    ENDPOINT or OUTSIDEIMAGE end validly, INVALIDPOINT invalidly, TRACKPOINT goes on; the deciding point is not kept;
  the backward half starts at the seed along minus the first forward segment, with fresh state; a streamline is kept
    when both halves end validly and its length is within [min_len, max_len].
Random numbers: one torch generator per call, seeded; a streamline's path depends on the seeds' order and the batch size
(the kernels will use counter-based numbers, independent of both).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
import torch
from . import fields as F

K_SMALL = 1e-4


@dataclass(frozen=True)
class Params:
    step_size: float = 0.5           # mm
    max_angle: float = 10.0          # degrees per step
    probe_length: float = 1.5        # mm
    probe_quality: int = 7           # points along the probe (sub-steps: quality - 1)
    probe_count: int = 1             # 1: the probe's center line only (a ring of probe_count at probe_radius: not yet)
    pmf_threshold: float = 0.1
    data_support_exponent: float = 1.0
    min_len: float = 20.0            # mm (the step 6 benchmark's)
    max_len: float = 250.0           # mm
    nbr_sample: int = 10             # candidates that estimate the maximum
    max_try: int = 100               # candidates tried for acceptance
    block: int = 10                  # tries scored at once (the rest in further blocks)


def _clamp_k(k):
    """DIPY's rule: both below K_SMALL in magnitude: straight (returned as zeros, flagged); else any below it becomes
    +K_SMALL."""
    small = k.abs() < K_SMALL
    straight = small.all(-1)
    k = torch.where(small, torch.full_like(k, K_SMALL), k)
    return k, straight


def _propagate(pos, T, N1, N2, k, s, scale):
    """One parallel-transport step of length s (mm) along each frame: (position, T, N2) after it. scale (3,): what the
    mm displacement is multiplied by to give voxels (1 / voxel size; DIPY's probe: the voxel size). Not re-orthonormalized."""
    k, straight = _clamp_k(k)
    k1, k2 = k[..., 0:1], k[..., 1:2]
    h = s * s / 2
    disp = s * T + h * (k1 * N1 + k2 * N2)
    Tn = (1 - (k1 * k1 + k2 * k2) * h) * T + s * (k1 * N1 + k2 * N2)
    N2n = -k2 * s * T - k1 * k2 * h * N1 + (1 - k2 * k2 * h) * N2
    st = straight[..., None]
    disp = torch.where(st, s * T, disp)
    Tn = torch.where(st, T, Tn)
    N2n = torch.where(st, N2, N2n)
    return pos + disp * scale, Tn, N2n


class Tracker:
    def __init__(self, pmf: F.PmfField, act: F.Act, affine, vox, params: Params = Params(), device="cpu",
                 _dipy_compat=False, backend="torch"):
        """_dipy_compat: validation only (module docstring): DIPY 1.12's units, to compare with it. backend: "torch" (the
        lockstep prototype, any device) or "metal" (tractline.ptt_metal: one thread per half; "mps" only)."""
        if params.probe_count != 1:
            raise NotImplementedError("probe_count > 1 (a ring of probes) is not implemented yet")
        if backend not in ("torch", "metal"):
            raise ValueError(f"backend {backend!r}: 'torch' or 'metal'")
        self.pmf, self.act, self.p, self.compat, self.backend = pmf, act, params, bool(_dipy_compat), backend
        self.dev = torch.device(device)
        if backend == "metal":
            from . import ptt_metal
            if self.dev.type != "mps" or not ptt_metal.available():
                raise ValueError("backend 'metal' needs device 'mps' and torch.mps.compile_shader")
            self.metal = ptt_metal.Halves(pmf, act, vox, params, compat=_dipy_compat)
        self.dt = F._dtype(device)
        self.affine = np.asarray(affine, np.float64)
        vox = np.asarray(vox, np.float64)
        self.inv_vox = torch.as_tensor(1.0 / vox, dtype=self.dt, device=self.dev)
        self.probe_scale = torch.as_tensor(vox if self.compat else 1.0 / vox, dtype=self.dt, device=self.dev)
        step_for_k = params.step_size / vox.mean() if self.compat else params.step_size
        theta = math.radians(params.max_angle)
        self.max_k = 2.0 * math.sin(theta / 2.0) / step_for_k               # 1 / (step / 2 / sin(theta / 2))
        self.probe_step = params.probe_length / (params.probe_quality - 1)
        self.norm = 1.0 / (params.probe_quality * params.probe_count)

    # -- random numbers
    def _disk(self, shape, g):
        r = self.max_k * torch.sqrt(torch.rand(shape, generator=g, device=self.dev, dtype=self.dt))
        a = 2 * math.pi * torch.rand(shape, generator=g, device=self.dev, dtype=self.dt)
        return torch.stack([r * torch.cos(a), r * torch.sin(a)], -1)

    def _pmf(self, pos, T):
        """Thresholded PMF along T at pos, for any leading shape."""
        sh = pos.shape[:-1]
        v, _ = self.pmf.value(pos.reshape(-1, 3), T.reshape(-1, 3))
        v = v.reshape(sh)
        return torch.where(v > self.p.pmf_threshold, v, torch.zeros_like(v))

    def _support(self, pos, T, N1, N2, last, k):
        """Likelihood of candidates k (B, C, 2) from frames (B, 3) at pos (B, 3): (L (B, C), the probe's last value)."""
        C = k.shape[1]
        P = pos[:, None].expand(-1, C, -1)
        t, n1, n2 = (x[:, None].expand(-1, C, -1) for x in (T, N1, N2))
        total = last[:, None].expand(-1, C).clone()
        val = torch.zeros_like(total)
        for q in range(1, self.p.probe_quality):
            P, tn, n2n = _propagate(P, t, n1, n2, k, self.probe_step, self.probe_scale)
            tn = tn / torch.linalg.norm(tn, dim=-1, keepdim=True)
            if q < self.p.probe_quality - 1:
                n1 = torch.cross(n2n, tn, dim=-1)
                n2 = n2n
            t = tn
            val = self._pmf(P, t)
            total = total + val
        L = total * self.norm
        if self.p.data_support_exponent != 1:
            L = L ** self.p.data_support_exponent
        return L, val

    def _sample(self, pos, T, N1, N2, last, g, init=False, seed_dir=None):
        """Rejection sampling for the halves given: (accepted (B,), k (B, 2), the new last value (B,), and at
        initialization the frames chosen (T, N1, N2))."""
        B = len(pos)
        p = self.p

        def candidates(C, r):
            """C candidates for the halves r: (k, their frames and last values, per-candidate frames or None)."""
            k = self._disk((len(r), C), g)
            if not init:
                return k, (T[r], N1[r], N2[r], last[r]), None
            # each candidate its own frame: T the seed direction, N2 a random unit vector perpendicular to it
            rr = torch.randn((len(r), C, 3), generator=g, device=self.dev, dtype=self.dt)
            t = seed_dir[r][:, None].expand(-1, C, -1)
            n2 = rr - (rr * t).sum(-1, keepdim=True) * t
            n2 = n2 / torch.linalg.norm(n2, dim=-1, keepdim=True)
            n1 = torch.cross(n2, t, dim=-1)
            lv = self._pmf(pos[r][:, None].expand(-1, C, -1), t)
            return k, (t, n1, n2, lv), True

        def support(k, fr, per, r):
            if per is None:
                return self._support(pos[r], *fr, k)
            t, n1, n2, lv = fr                                            # per-candidate frames: flatten (n C) as rows
            n, C = k.shape[:2]
            L, v = self._support(pos[r].repeat_interleave(C, 0), t.reshape(-1, 3), n1.reshape(-1, 3), n2.reshape(-1, 3),
                                 lv.reshape(-1), k.reshape(-1, 1, 2))
            return L.reshape(n, C), v.reshape(n, C)

        rows = torch.arange(B, device=self.dev)
        k, fr, per = candidates(p.nbr_sample, rows)
        L, _ = support(k, fr, per, rows)
        M = (2.0 * L.max(1).values) ** p.data_support_exponent
        done = torch.zeros(B, dtype=torch.bool, device=self.dev)
        out_k = torch.zeros((B, 2), dtype=self.dt, device=self.dev)
        out_v = torch.zeros(B, dtype=self.dt, device=self.dev)
        out_f = [torch.zeros((B, 3), dtype=self.dt, device=self.dev) for _ in range(3)] if init else None
        pending, tried = rows, 0                                           # later blocks score only the halves still waiting
        while tried < p.max_try and len(pending):
            C = min(p.block, p.max_try - tried)
            k, fr, per = candidates(C, pending)
            L, v = support(k, fr, per, pending)
            u = torch.rand((len(pending), C), generator=g, device=self.dev, dtype=self.dt)
            ok = (u * M[pending, None] <= L) if init else (u * M[pending, None] < L)
            first = torch.argmax(ok.to(torch.int8), 1)
            take = ok.any(1)
            got, sel = pending[take], first[take]
            out_k[got] = k[take, sel]
            out_v[got] = v[take, sel]
            if init:
                for j in range(3):
                    out_f[j][got] = fr[j][take, sel]
                nt = ~take                                                  # the last candidate tried stands (DIPY)
                out_k[pending[nt]] = k[nt, -1]
                out_v[pending[nt]] = fr[3][nt, -1]
                for j in range(3):
                    out_f[j][pending[nt]] = fr[j][nt, -1]
            done[got] = True
            pending = pending[~take]
            tried += C
        return done, out_k, out_v, out_f

    def _half(self, seed_vox, direction, g):
        """Track halves from seed_vox (B, 3) along direction (B, 3): (points (B, n, 3) voxel, lengths (B,), valid (B,))."""
        p = self.p
        B = len(seed_vox)
        max_pts = int(p.max_len / p.step_size)
        pos = seed_vox.clone()
        ok, k, last, frames = self._sample(pos, None, None, None, None, g, init=True, seed_dir=direction)
        T, N1, N2 = frames                                                  # initialization failures keep the last candidate
        alive = torch.ones(B, dtype=torch.bool, device=self.dev)
        status = torch.full((B,), F.TRACKPOINT, dtype=torch.int64, device=self.dev)
        pts = [seed_vox]
        n = torch.ones(B, dtype=torch.int64, device=self.dev)
        shift = direction * self.inv_vox * p.step_size if self.compat else torch.zeros_like(direction)
        for i in range(1, max_pts):
            idx = torch.nonzero(alive).flatten()
            if len(idx) == 0:
                break
            P, Tn, N2n = _propagate(pos[idx], T[idx], N1[idx], N2[idx], k[idx], p.step_size, self.inv_vox)
            Tn = Tn / torch.linalg.norm(Tn, dim=-1, keepdim=True)
            N1n = torch.cross(N2n, Tn, dim=-1)
            N1n = N1n / torch.linalg.norm(N1n, dim=-1, keepdim=True)
            N2n = torch.cross(Tn, N1n, dim=-1)
            pos[idx], T[idx], N1[idx], N2[idx] = P, Tn, N1n, N2n
            acc, kn, lv, _ = self._sample(P, Tn, N1n, N2n, last[idx], g)
            fail = idx[~acc]
            alive[fail] = False                                              # FAIL: no point, the half stops (TRACKPOINT)
            go = idx[acc]
            k[go], last[go] = kn[acc], lv[acc]
            point = torch.zeros_like(pos)
            point[go] = pos[go] + shift[go]
            st = self.act(point[go])
            status[go] = st
            n[go] += 1
            pts.append(point)
            stop = st != F.TRACKPOINT
            alive[go[stop]] = False
            n[go[stop]] -= 1                                                 # the deciding point is not kept
        stack = torch.stack(pts, 1)
        valid = (status == F.ENDPOINT) | (status == F.OUTSIDEIMAGE)
        return stack, n, valid

    @torch.no_grad()
    def track(self, seeds_world, seed=1, directions=None, batch=None, offset=0, keep_index=False):
        """seeds_world (n, 3) RAS mm -> (streamlines: list of (m, 3) RAS mm, kept: valid and within the lengths; info).
        Seeds go in batches of `batch` (each half's points are held for the whole batch: batch x max points x 3). offset:
        the first seed's index among all seeds (the Metal backend's random numbers are keyed by it: a seed's path is the
        same in any batch). keep_index: a dict {seed index: streamline} instead of the list."""
        g = torch.Generator(device=self.dev).manual_seed(int(seed))
        seeds_world = np.asarray(seeds_world, np.float64)
        batch = batch or (200_000 if self.backend == "metal" else 20_000)     # the prototype holds more per seed
        out, info = [], {"seeds": 0, "seeds without a peak": 0, "valid": 0, "kept": 0}
        for a in range(0, len(seeds_world), batch):
            o, i = self._track(seeds_world[a:a + batch], g, None if directions is None else directions[a:a + batch],
                               seed, offset + a)
            out += o
            for key in info:
                info[key] += i[key]
        lengths = [float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum()) for _, x in out]
        info["mean length mm"] = float(np.mean(lengths)) if lengths else 0.0
        return (dict(out) if keep_index else [x for _, x in out]), info

    def _halves_metal(self, sv, d, ids, seed, half):
        pts, n, st = self.metal.run(sv, d, seed=seed, half=half, ids=ids)
        valid = (st == F.ENDPOINT) | (st == F.OUTSIDEIMAGE)
        return pts.to(self.dt), n.to(torch.int64), valid

    def _track(self, seeds_world, g, directions, seed=1, offset=0):
        inv = np.linalg.inv(self.affine)
        sv = np.asarray(seeds_world, np.float64) @ inv[:3, :3].T + inv[:3, 3]
        sv = torch.as_tensor(sv, dtype=self.dt, device=self.dev)
        dropped = 0
        ids = torch.arange(offset, offset + len(sv), device=self.dev)
        if directions is None:                                              # the PMF's largest peak at the seed; seeds
            allv, _ = self.pmf.all(sv)                                      # with none (all <= 0) are dropped, as DIPY's
            peak = allv.max(1).values > 0                                   # seeds_directions_pairs does
            dropped = int((~peak).sum())
            sv, allv, ids = sv[peak], allv[peak], ids[peak]
            directions = self.pmf.vertices[torch.argmax(allv, 1)]
        d = torch.as_tensor(directions, dtype=self.dt, device=self.dev)
        if len(sv) == 0:
            return [], {"seeds": dropped, "seeds without a peak": dropped, "valid": 0, "kept": 0}
        half = (lambda s, dd, h: self._halves_metal(s, dd, ids, seed, h)) if self.backend == "metal" \
            else (lambda s, dd, h: self._half(s, dd, g))
        fw, nf, vf = half(sv, d, 0)
        back = torch.where((nf > 1)[:, None], fw[:, 1] - fw[:, 0], d)      # the first forward segment, else the seed's
        back = -back / torch.linalg.norm(back, dim=-1, keepdim=True)
        bw, nb, vb = half(sv, back, 1)
        mf, mb = int(nf.max()), int(nb.max())                              # only the points in use leave the device
        fw, bw = fw[:, :mf].cpu().double().numpy(), bw[:, :mb].cpu().double().numpy()
        nf, nb = nf.cpu().numpy(), nb.cpu().numpy()
        valid = (vf & vb).cpu().numpy()
        idn = ids.cpu().numpy()
        A, p = self.affine, self.p
        lo, hi = int(p.min_len / p.step_size), int(p.max_len / p.step_size)
        out = []
        for i in np.flatnonzero(valid):
            line = np.concatenate([bw[i, 1:nb[i]][::-1], fw[i, :nf[i]]])
            world = line @ A[:3, :3].T + A[:3, 3]
            if self.compat:                                                 # DIPY's filter: the point count
                keep = lo <= len(world) <= hi
            else:
                length = float(np.linalg.norm(np.diff(world, axis=0), axis=1).sum()) if len(world) > 1 else 0.0
                keep = p.min_len <= length <= p.max_len
            if keep:
                out.append((int(idn[i]), world.astype(np.float32)))
        return out, {"seeds": len(sv) + dropped, "seeds without a peak": dropped, "valid": int(valid.sum()), "kept": len(out)}
