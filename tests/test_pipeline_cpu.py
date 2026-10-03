"""The whole pipeline on the CPU path, on the synthetic phantom: it runs, repeats bit for bit, and
tracks the band. Labeled by a stand-in (no weights needed); test_labelers.py checks RapidParc."""
import numpy as np
import pytest
from phantom import subject
from tractline import pipeline as P
from tractline.labelers.base import Labels, MIN_LENGTH_MM, lengths


def stand_in(fibers, draws=(0,), logp=False):
    """A labeler that names every streamline of 40 mm or more tract 0."""
    _, _, length = lengths(fibers)
    keep = length >= MIN_LENGTH_MM
    return Labels(keep=keep, length_mm=length[keep], tract=np.zeros(int(keep.sum()), np.int64), logp=None)


@pytest.fixture(scope="module")
def runs():
    s = subject()
    # the default worker count (one per core), as callers run it, and two workers: the same fibers either way
    return s, [P.run(s, stand_in, P.Timer(), device="cpu", workers=w) for w in (None, 2)]


def test_runs_and_repeats(runs):
    _, ((c0, t0, l0), (c1, t1, l1)) = runs
    assert len(t0.fibers) > 500
    assert len(t0.fibers) == len(t1.fibers)
    assert all(np.array_equal(a, b) for a, b in zip(t0.fibers, t1.fibers))
    assert np.array_equal(c0.field_hz, c1.field_hz)
    assert l0.keep.sum() > 0.9 * len(t0.fibers)


def test_no_distortion_no_field(runs):
    s, ((corr, _, _), _) = runs
    assert np.quantile(np.abs(corr.field_hz[s.brain]), 0.99) < 2.0         # Hz: the pair has no distortion


def test_fibers_follow_the_band(runs):
    s, ((_, tg, _), _) = runs
    P_ = np.concatenate(tg.fibers)
    ijk = (np.c_[P_, np.ones(len(P_))] @ np.linalg.inv(s.affine).T)[:, :3]
    X, _, Z = s.dwi.shape[:3]
    off = np.abs(ijk[:, 0] - (X / 2 - 0.5)), np.abs(ijk[:, 2] - (Z / 2 - 0.5))
    assert np.mean((off[0] < 3) & (off[1] < 3)) > 0.95                     # within a voxel of the band
    _, _, length = lengths(tg.fibers)
    assert length.max() > 50                                               # along it: the band is ~70 mm


def test_one_worker_same_fibers(runs):
    s, ((_, t0, _), _) = runs
    _, t1, _ = P.run(s, stand_in, P.Timer(), device="cpu", workers=1)
    assert len(t0.fibers) == len(t1.fibers) and all(np.array_equal(a, b) for a, b in zip(t0.fibers, t1.fibers))
