"""Uncertainty from the whole pipeline: several runs of it, each with every stage resampled, summarized
per streamline and per voxel. A prototype (NOTES 2026-10-07, "The pipeline's ensemble").

One run's labels carry the labeler's probabilities and nothing else: not the scan's noise, the field
estimated from it, the tracker's path through it, or the labeler's draw. Run the pipeline R times with
each of those resampled (a bootstrap replicate of the scan, its own field, its own tracking, another
draw: bench/ensemble_pilot.py does it) and keep, from each run, a RunRecord: its labeled streamlines
keyed by seed voxel, their winning clusters and cluster log-probabilities, their 20-point paths, and
the streamlines of each tract per voxel.

Two summaries, both meant for a rank field (rankfield's format: per element, the kept classes and how
far each trails the winner) plus one plane rankfield does not have yet - votes:

  StreamlineEnsemble   per streamline of a reference run, matched across runs by seed voxel (a voxel
                       seeds one point; masks, and so seed lists, differ between runs). The MIXTURE (the
                       log of the mean cluster probability over the runs that labeled that seed) is what a
                       rank field encodes; VOTES are how many runs each cluster won. Votes compose under any
                       grouping of clusters made after the fact - a group's votes are its members' summed,
                       exactly, since a run's label is the group of its winning cluster - which margins
                       read from a depth-cut field do not. Also each run's path distance to the
                       reference's (MDF, mm): label uncertainty and position uncertainty, apart.
  VoxelEnsemble        per voxel: each tract's mean streamline count over the runs (a streamline counted
                       once per voxel it visits), as logits log(density), with one more class, NONE, at
                       log(tau). A tract wins a voxel where it is the densest and denser than tau, so its
                       margin against NONE is log(density / tau): the zero level set is the tract's
                       iso-density surface at tau, against another tract the boundary where they cross.
                       Position uncertainty (the scan, the field, the tracker) is in the pooled density;
                       VOTES are how many runs each class won the voxel in.

vote_planes() turns votes into per-rank planes aligned with an encoded field's ranks: votes[j] for the
class at rank j, and a vote tail - the runs whose winner the field did not keep.

numpy and torch only; rankfield is the caller's (bench/ensemble_pilot.py).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np, torch
from .labelers.base import LUT

N_TRACTS = 43                    # 0-41 tracts, 42 Other
NONE = 43                        # the voxel field's extra class: no tract dense enough
POINTS = 20                      # a path's points for the distance between runs (the web export's 20)
KEY_BITS = 12                    # seed voxel (k, j, i) -> one int64: 4,096 voxels an axis


def seed_keys(seed_voxel) -> np.ndarray:
    """(n, 3) seed voxels (k, j, i) -> (n,) int64 keys, the same voxel the same key in every run."""
    v = np.asarray(seed_voxel, np.int64)
    assert v.size == 0 or (v.min() >= 0 and v.max() < 1 << KEY_BITS), "seed voxel outside the key's range"
    return (v[:, 0] << 2 * KEY_BITS) | (v[:, 1] << KEY_BITS) | v[:, 2]


def resample(fibers, points=POINTS) -> np.ndarray:
    """(n, points, 3) float32: each streamline at `points` equally spaced arc lengths (linear between its
    points); an empty one is NaN, a one-point one that point."""
    out = np.full((len(fibers), points, 3), np.nan, np.float32)
    t = np.linspace(0.0, 1.0, points)
    for r, f in enumerate(fibers):
        f = np.asarray(f, np.float64)
        if len(f) == 0:
            continue
        s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(f, axis=0), axis=1))]
        if s[-1] <= 0:
            out[r] = f[0]
            continue
        u = t * s[-1]
        out[r] = np.stack([np.interp(u, s, f[:, a]) for a in range(3)], 1)
    return out


def mdf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, P, 3) x (n, P, 3) -> (n,) mean direct-flip distance (mm): the mean point distance, b taken in the
    orientation nearer a."""
    d = np.linalg.norm(a - b, axis=2).mean(1)
    f = np.linalg.norm(a - b[:, ::-1], axis=2).mean(1)
    return np.minimum(d, f)


def voxel_counts(fibers, tract, affine, shape) -> np.ndarray:
    """(43, X*Y*Z) uint16: per tract, the streamlines visiting each voxel of the grid (each streamline once
    per voxel it has a point in; points outside the grid dropped). fibers: (n, 3) RAS mm arrays; tract (n,)."""
    shape = tuple(int(v) for v in shape)
    nvox = int(np.prod(shape))
    if len(fibers) == 0:
        return np.zeros((N_TRACTS, nvox), np.uint16)
    lens = np.array([len(f) for f in fibers])
    P = np.concatenate([np.asarray(f, np.float64).reshape(-1, 3) for f in fibers])
    fid = np.repeat(np.arange(len(fibers)), lens)
    r2i = np.linalg.inv(np.asarray(affine, float))
    ijk = np.rint(P @ r2i[:3, :3].T + r2i[:3, 3]).astype(np.int64)
    ok = (ijk >= 0).all(1) & (ijk < np.array(shape)).all(1)
    vox = np.ravel_multi_index(tuple(ijk[ok].T), shape)
    key = np.unique(fid[ok].astype(np.int64) * nvox + vox)               # (streamline, voxel) once
    f, v = np.divmod(key, nvox)
    c = np.bincount(np.asarray(tract)[f].astype(np.int64) * nvox + v, minlength=N_TRACTS * nvox)
    assert c.max(initial=0) < 65536
    return c.reshape(N_TRACTS, nvox).astype(np.uint16)


@dataclass
class RunRecord:
    """What one run of the pipeline leaves for the ensemble: its labeled streamlines (>= 40 mm) only."""
    keys: np.ndarray             # (n,) int64 seed-voxel keys (seed_keys), unique
    cluster: np.ndarray          # (n,) int16 the winning cluster (0-1599)
    logp: np.ndarray             # (n, 1600) float16 cluster log-probabilities
    paths: np.ndarray            # (n, POINTS, 3) float32 (resample)
    counts: np.ndarray           # (43, X*Y*Z) uint16 (voxel_counts)
    shape: tuple                 # (X, Y, Z) the grid the counts are on

    @staticmethod
    def of(tg, labels, affine, shape) -> "RunRecord":
        """From a run's Tractogram and Labels (labeled with logp=True), counts on the grid (affine, shape)."""
        if labels.logp is None:
            raise ValueError("RunRecord needs the labels' cluster log-probabilities: label with logp=True")
        kept = np.flatnonzero(labels.keep)
        fibers = [tg.fibers[i] for i in kept]
        keys = seed_keys(np.asarray(tg.stats["seed_voxel"])[kept])
        assert len(np.unique(keys)) == len(keys), "two streamlines from one seed voxel"
        logp = labels.logp.cpu().numpy().astype(np.float16)
        lp32 = logp.astype(np.float32)
        cluster = lp32.argmax(1)
        bad = np.flatnonzero(LUT[cluster] != labels.tract)
        if len(bad):                                                       # ties after float16 rounding: the labeler took
            tied = (lp32[bad] == lp32[bad].max(1, keepdims=True)) & (LUT[None] == labels.tract[bad, None])   # its argmax in
            assert tied.any(1).all(), "labels.tract is not the winning cluster's tract"                       # float32
            cluster[bad] = tied.argmax(1)
        cluster = cluster.astype(np.int16)
        return RunRecord(keys, cluster, logp, resample(fibers), voxel_counts(fibers, labels.tract, affine, shape),
                         tuple(int(v) for v in shape))

    def save(self, path):
        """An .npz; the counts as their nonzero entries (a few percent of the grid has streamlines)."""
        nz = np.flatnonzero(self.counts)
        np.savez(path, keys=self.keys, cluster=self.cluster, logp=self.logp, paths=self.paths, counts_at=nz.astype(np.int64),
                 counts_n=self.counts.ravel()[nz], counts_shape=np.array(self.counts.shape), shape=np.array(self.shape))

    @staticmethod
    def load(path, logp=True) -> "RunRecord":
        """logp=False leaves the log-probabilities on disk (None): a held-out run needs its winners only."""
        z = np.load(path)
        counts = np.zeros(int(np.prod(z["counts_shape"])), np.uint16)
        counts[z["counts_at"]] = z["counts_n"]
        return RunRecord(z["keys"], z["cluster"], z["logp"] if logp else None, z["paths"], counts.reshape(z["counts_shape"]),
                         tuple(int(v) for v in z["shape"]))


class StreamlineEnsemble:
    """Per streamline of a reference run, over the runs added: the mixture, the votes, the path distances."""
    def __init__(self, ref: RunRecord):
        self.ref = ref
        self.order = np.argsort(ref.keys)
        self.n = len(ref.keys)
        self.acc = torch.full((self.n, 1600), -float("inf"))
        self.present = np.zeros(self.n, np.int32)
        self.winners, self.dist = [], []                          # per run: (n,) cluster (-1 absent), (n,) MDF mm (NaN)

    def match(self, run: RunRecord):
        """(reference rows, run rows) of the seeds both labeled."""
        sk = self.ref.keys[self.order]
        pos = np.searchsorted(sk, run.keys)
        pos[pos == len(sk)] = 0
        hit = sk[pos] == run.keys
        return self.order[pos[hit]], np.flatnonzero(hit)

    def add(self, run: RunRecord):
        ri, qi = self.match(run)
        rows = torch.from_numpy(ri)
        self.acc[rows] = torch.logaddexp(self.acc[rows], torch.from_numpy(run.logp[qi].astype(np.float32)))
        self.present[ri] += 1
        w = np.full(self.n, -1, np.int16); w[ri] = run.cluster[qi]
        d = np.full(self.n, np.nan, np.float32); d[ri] = mdf(self.ref.paths[ri], run.paths[qi])
        self.winners.append(w); self.dist.append(d)
        return len(ri)

    @property
    def runs(self):
        return len(self.winners)

    def mixture(self) -> torch.Tensor:
        """(n, 1600) float32 log of the mean cluster probability over the runs that labeled each seed (-inf rows
        where none did)."""
        return self.acc - torch.from_numpy(np.log(np.maximum(self.present, 1)).astype(np.float32))[:, None]

    def votes(self, lut=LUT, groups=None) -> np.ndarray:
        """(n, G) votes per group: how many runs each group's clusters won. lut: cluster -> group (default the 43
        tracts); groups: G (default lut.max() + 1)."""
        W = np.stack(self.winners)                                  # (R, n)
        G = int(lut.max()) + 1 if groups is None else groups
        g = np.where(W >= 0, np.asarray(lut)[np.maximum(W, 0)], G)  # absent -> an overflow column, dropped
        out = np.zeros((self.n, G + 1), np.int32)
        np.add.at(out, (np.broadcast_to(np.arange(self.n), W.shape), g), 1)
        return out[:, :G]

    def cluster_votes(self, classes) -> np.ndarray:
        """(len(classes), n) votes for given cluster ids per streamline ((K, n) int, -1 = none)."""
        W = np.stack(self.winners)                                  # (R, n)
        cls = np.asarray(classes)
        return ((W[None] == cls[:, None]) & (cls[:, None] >= 0)).sum(1)

    def distances(self) -> np.ndarray:
        """(R, n) MDF to the reference's path, mm (NaN where a run did not label the seed)."""
        return np.stack(self.dist)


class VoxelEnsemble:
    """Per voxel, over the runs added: each tract's summed streamline count, and the votes per class (43 + NONE)."""
    def __init__(self, shape, tau=1.0):
        self.shape, self.tau = tuple(int(v) for v in shape), float(tau)
        nvox = int(np.prod(self.shape))
        self.sum = np.zeros((N_TRACTS, nvox), np.float32)
        self.vote = np.zeros((N_TRACTS + 1, nvox), np.uint16)
        self.runs = 0

    def winner(self, counts) -> np.ndarray:
        """(nvox,) one run's (or a density's) class per voxel: the densest tract when denser than tau, else NONE."""
        c = np.asarray(counts, np.float32)
        w = c.argmax(0)
        return np.where(c.max(0) >= self.tau, w, NONE)

    def add(self, run: RunRecord):
        assert run.shape == self.shape
        self.sum += run.counts
        nvox = self.sum.shape[1]
        self.vote += np.bincount(self.winner(run.counts) * nvox + np.arange(nvox), minlength=(N_TRACTS + 1) * nvox
                                 ).reshape(N_TRACTS + 1, nvox).astype(np.uint16)
        self.runs += 1

    def logits(self, floor=None) -> np.ndarray:
        """(44, X, Y, Z) float32: log mean density per tract (floored at `floor`, default tau * e^-16, far past
        rankfield's clip), and NONE at log(tau)."""
        floor = self.tau * np.exp(-16.0) if floor is None else floor
        d = self.sum / max(self.runs, 1)
        lg = np.concatenate([np.log(np.maximum(d, floor)), np.full((1, d.shape[1]), np.log(self.tau), np.float32)])
        return lg.astype(np.float32).reshape(N_TRACTS + 1, *self.shape)

    def votes(self) -> np.ndarray:
        """(44, X, Y, Z) uint16 runs each class won the voxel in."""
        return self.vote.reshape(N_TRACTS + 1, *self.shape)


def vote_planes(ranks: np.ndarray, votes: np.ndarray, runs) -> tuple[np.ndarray, np.ndarray]:
    """An encoded field's ranks (N, ...) (class + 1, 0 absent) and votes per class (K, ...) -> (votes per rank
    (N, ...) uint8, vote tail (...) uint8): the runs the class at rank j won, and the runs whose winner is not
    among the kept classes. runs: the runs per element (scalar or (...)), at most 255."""
    runs = np.broadcast_to(np.asarray(runs), ranks.shape[1:])
    assert runs.max(initial=0) <= 255, "votes are bytes: at most 255 runs"
    cls = ranks.astype(np.int64) - 1
    v = np.take_along_axis(votes, np.maximum(cls, 0), axis=0)
    v = np.where(cls >= 0, v, 0)
    tail = runs - v.sum(0)
    assert (tail >= 0).all()
    return v.astype(np.uint8), tail.astype(np.uint8)


def group_votes(ranks: np.ndarray, vplanes: np.ndarray, lut, groups) -> np.ndarray:
    """Votes per group from the per-rank planes alone: (groups, ...) - each kept class's votes added to its
    group's. Exact when the vote tail is 0; otherwise short of the truth by at most the tail."""
    cls = ranks.astype(np.int64) - 1
    g = np.where(cls >= 0, np.asarray(lut)[np.maximum(cls, 0)], groups)
    out = np.zeros((groups + 1, *ranks.shape[1:]), np.int32)
    idx = np.indices(ranks.shape[1:])
    for j in range(len(ranks)):
        np.add.at(out, (g[j], *idx), vplanes[j])
    return out[:groups]
