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
    assert len(t0.fibers) > 700                                              # the phantom gives ~790
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
    assert np.mean((off[0] < 3) & (off[1] < 3)) > 0.95                     # the band is 2 voxels each side: within a voxel of its edge
    _, _, length = lengths(tg.fibers)
    assert length.max() > 50                                               # along it: the band is ~70 mm


def test_one_worker_same_fibers(runs):
    s, ((_, t0, _), _) = runs
    _, t1, _ = P.run(s, stand_in, P.Timer(), device="cpu", workers=1)
    assert len(t0.fibers) == len(t1.fibers) and all(np.array_equal(a, b) for a, b in zip(t0.fibers, t1.fibers))


def test_compiled_step(runs, monkeypatch):
    """The fast step through torch.compile: the same fibers on one worker and two (the workers load what the
    parent compiled), and the uncompiled step's tractogram to rounding (another rounding: fused arithmetic)."""
    from tractline import ukf as U
    monkeypatch.setenv("TRACTLINE_COMPILE", "1")
    if not U.can_compile():
        pytest.skip("no C++ compiler for torch.compile")
    s, ((_, t0, _), _) = runs
    tgs = [P.run(s, stand_in, P.Timer(), device="cpu", workers=w)[1] for w in (1, 2)]
    a, b = tgs
    assert len(a.fibers) == len(b.fibers) and all(np.array_equal(x, y) for x, y in zip(a.fibers, b.fibers))
    assert abs(len(a.fibers) - len(t0.fibers)) <= 0.02 * len(t0.fibers)
    ref = dict(zip(t0.stats["seed_index"], t0.fibers))                     # paired by seed
    same = [k for k, x in zip(a.stats["seed_index"], a.fibers) if k in ref and len(x) == len(ref[k]) and np.abs(x - ref[k]).max() < 0.1]
    assert len(same) > 0.8 * len(t0.fibers)


@pytest.mark.parametrize("backend", ["torch", "metal"])
def test_observe_changes_nothing(backend):
    """ukf.track's observe hook (a diagnostic, off by default) reads every step and changes no fiber: the same
    fibers with it as without, on a subset of seeds (select) as on all of them; each half-fiber stops exactly once, and
    its recorded points are one per observed step that went on at a multiple of record_length, after the seed."""
    import torch
    from tractline import ukf as U, ukf_metal
    from tractline.prep import prepare
    if backend == "metal" and not ukf_metal.available():
        pytest.skip("no Metal")
    s = subject()
    t = prepare(s.dwi, s.affine, s.bval, s.bvec, shell=2800.0)
    D = U.from_arrays(t.dwi, t.header, t.mask)
    kw = dict(dtype=torch.float32, fast=True) if backend == "torch" else dict(backend="metal")
    ref, st = U.track(D, **kw)
    sel = np.arange(0, st["seeds"], 7)
    rows = []
    def observe(step, index, x, state, direction, stop, info):
        rows.append((step, index.cpu().numpy(), stop.cpu().numpy()))
    sub, sst = U.track(D, select=sel, observe=observe, **kw)
    byseed = dict(zip(st["seed_index"], ref))
    assert len(sub) > 50 and all(np.array_equal(f, byseed[k]) for k, f in zip(sst["seed_index"], sub))
    stops = np.concatenate([i[s_] for _, i, s_ in rows])
    assert len(stops) == 2 * len(sel) and len(np.unique(stops)) == len(stops)
    nrec = np.ones(2 * len(sel), int)                                       # the seed, then every 6th step that went on
    for step, i, s_ in rows:
        if (step + 1) % 6 == 0:
            np.add.at(nrec, i[~s_], 1)
    row = {int(k): r for r, k in enumerate(sel)}
    assert all(len(f) == nrec[2 * row[k]] + nrec[2 * row[k] + 1] - 1 for k, f in zip(sst["seed_index"], sub))


def test_tracker_settings_opt_in(runs):
    """pipeline.track's tracker= (opt-in ukf.track settings): None, {} and today's values spelled out give the default
    fibers bit for bit; another value changes them; an unknown setting is refused before any work."""
    s, ((c0, t0, _), _) = runs
    same = lambda a, b: len(a.fibers) == len(b.fibers) and all(np.array_equal(x, y) for x, y in zip(a.fibers, b.fibers))
    today = {"Rs": 0.02, "Qm": 0.001, "Ql": 50.0, "step_length": 0.3, "seeding_threshold": 0.1, "stopping_fa": 0.08,
             "stopping_threshold": 0.06, "record_length": 1.8, "max_half_length": 250.0}
    assert set(today) == P.TRACKER_SETTINGS
    for tracker in (None, {}, today):
        assert same(P.track(s, c0.dwi, P.Timer(), device="cpu", workers=1, tracker=tracker), t0)
    assert not same(P.track(s, c0.dwi, P.Timer(), device="cpu", workers=1, tracker={"step_length": 0.6}), t0)
    with pytest.raises(ValueError, match="not among the tracker's settings"):
        P.track(s, c0.dwi, P.Timer(), device="cpu", workers=1, tracker={"batch": 10})
