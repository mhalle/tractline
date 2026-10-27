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
    """ukf.track's observe hook (a diagnostic, off by default) reads every step and changes no fiber: on a subset of
    seeds (select), the same fibers bit for bit with it as without; the subset's fibers are those of a run over all
    seeds, to rounding (float32 products on x86 can round differently with the batch's size and threads, and the
    tracker carries it along); each half-fiber stops exactly once, and its recorded points are one per observed step
    that went on at a multiple of record_length, after the seed."""
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
    plain, pst = U.track(D, select=sel, **kw)                              # the same seeds, the same batches, no hook
    assert len(sub) > 50 and list(sst["seed_index"]) == list(pst["seed_index"])
    assert all(np.array_equal(f, g) for f, g in zip(sub, plain))
    byseed = dict(zip(st["seed_index"], ref))
    close = [k for k, f in zip(sst["seed_index"], sub)
             if len(f) == len(byseed[k]) and np.abs(f - byseed[k]).max() < 0.1]
    assert len(close) > 0.8 * len(sub)
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
             "stopping_threshold": 0.06, "record_length": 1.8, "max_half_length": 250.0, "tie_angle": 0.0,
             "tie_sw2": False}
    assert set(today) == P.TRACKER_SETTINGS
    for tracker in (None, {}, today):
        assert same(P.track(s, c0.dwi, P.Timer(), device="cpu", workers=1, tracker=tracker), t0)
    for tracker in ({"step_length": 0.6}, {"tie_angle": 2.0}, {"tie_angle": 2.0, "tie_sw2": True}):
        assert not same(P.track(s, c0.dwi, P.Timer(), device="cpu", workers=1, tracker=tracker), t0)
    with pytest.raises(ValueError, match="needs tie_angle"):
        P.track(s, c0.dwi, P.Timer(), device="cpu", workers=1, tracker={"tie_sw2": True})
    with pytest.raises(ValueError, match="not among the tracker's settings"):
        P.track(s, c0.dwi, P.Timer(), device="cpu", workers=1, tracker={"batch": 10})


def _one_step(backend, tie_angle, tie_sw2=False):
    """One step from the phantom's seeds, every half-fiber's second tensor turned off the first: by 0.1 degree on
    even seeds (a near tie, as every seed's two tensors start equal), by 10 degrees on odd ones. Returns advance's
    outputs as numpy."""
    import torch
    from tractline import ukf as U, ukf_metal
    s = subject()
    from tractline.prep import prepare
    t = prepare(s.dwi, s.affine, s.bval, s.bvec, shell=2800.0)
    D = U.from_arrays(t.dwi, t.header, t.mask)
    pts, fwd, inv, e1, _ = U.seeds(D)
    x = torch.stack([pts, pts], 1).reshape(-1, 3)
    st = torch.stack([fwd, inv], 1).reshape(-1, 10).clone()
    o = torch.stack([e1, -e1], 1).reshape(-1, 3)
    m = st[:, 5:8]
    k = torch.linalg.cross(m, torch.tensor([0.3, 0.5, 0.8], dtype=m.dtype).expand_as(m))   # an axis normal to m
    k = k / k.norm(dim=1, keepdim=True)
    a = torch.deg2rad(torch.where(torch.arange(len(st)) // 2 % 2 == 0, 0.1, 10.0).to(m.dtype))[:, None]
    st[:, 5:8] = m * torch.cos(a) + torch.linalg.cross(k, m) * torch.sin(a)       # Rodrigues: m turned by a about k
    dev = "mps" if backend == "metal" else "cpu"
    Dd = U.at_dtype(D, torch.float32, dev)
    if backend == "torch":
        Dd = {**Dd, "fast": True}
        Dd.update(U._neighbors(Dd))
    f32 = lambda v: v.to(device=dev, dtype=torch.float32)
    P0 = (U.P0 * torch.eye(10)).expand(len(st), 10, 10)
    Q = f32(torch.diag(torch.tensor([0.001] * 3 + [50.0] * 2 + [0.001] * 3 + [50.0] * 2)))
    step = ukf_metal.advance if backend == "metal" else U.advance
    xo, so, Po, d, stop, info = step(Dd, f32(x), f32(st), f32(P0), f32(o), Q, 0.02, 1, 834, tie_angle=tie_angle,
                                  **({"tie_sw2": True} if tie_sw2 else {}))
    n = lambda v: v.cpu().numpy()
    return {"x": n(xo), "state": n(so), "P": n(Po), "dir": n(d), "stop": n(stop), "swap": n(info["swap"]),
            "swap2": n(info["swap2"]), "fa": n(info["fa"])}


def _swapped(state, P):
    return np.concatenate([state[:, 5:], state[:, :5]], 1), np.roll(np.roll(P, 5, 1), 5, 2)


@pytest.mark.parametrize("backend", ["torch", "metal"])
def test_tie_angle_keeps_the_order(backend):
    """The opt-in tie-break (ukf.track's tie_angle): one step on constructed near ties. Where the two tensors are
    within the angle, the first swap (sw) does not fire, and the second (sw2) is the default's rule on the state as it
    then stands - the filter's update, before both swaps, is the same bit for bit. So, at a tie:
      - where the default did not swap at sw, everything is the default's, bit for bit;
      - where it did, the tie-break's sw2 decides on the unswapped state S: it is the default's sw2 when S's two
        tensors are on the same side of FA 0.2 (sw2 reads only the angle, symmetric, and which side fat1 is on), and
        the state is S swapped exactly when sw2 fires.
    Elsewhere nothing changes. And the default does swap some of those near ties at sw."""
    from tractline import ukf as U, ukf_metal
    import torch
    if backend == "metal" and not ukf_metal.available():
        pytest.skip("no Metal")
    a, b = _one_step(backend, 0.0), _one_step(backend, 1.0)
    m1, m2 = b["state"][:, 0:3], b["state"][:, 5:8]
    c = np.abs((m1 * m2).sum(1)) / np.linalg.norm(m1, axis=1) / np.linalg.norm(m2, axis=1)
    tie, far = c > np.cos(np.radians(0.5)), c < np.cos(np.radians(2.0))       # clear of the 1-degree edge
    assert tie.sum() > 100 and far.sum() > 100
    assert not b["swap"][tie].any()
    assert a["swap"][tie].sum() > 10                                          # the default swaps near ties at sw
    kept = tie & ~a["swap"]
    for k in a:
        assert np.array_equal(a[k][kept], b[k][kept]), k
    swapped = lambda st, P, sel: [np.where(sel.reshape((-1,) + (1,) * (v.ndim - 1)), w, v) for v, w in zip((st, P), _swapped(st, P))]
    S, SP = swapped(a["state"], a["P"], a["swap"] ^ a["swap2"])               # the state after the update, before the swaps
    want = swapped(S, SP, b["swap2"])
    assert np.array_equal(b["state"][tie], want[0][tie]) and np.array_equal(b["P"][tie], want[1][tie])
    fa = lambda o: U.l2fa(*(torch.from_numpy(np.maximum(S[:, o + i], U.LMIN)).double() for i in (3, 4, 4))).numpy()
    f1, f2 = fa(0), fa(5)
    side = (np.abs(f1 - 0.2) > 1e-4) & (np.abs(f2 - 0.2) > 1e-4)
    same_side = tie & a["swap"] & side & ((f1 <= 0.2) == (f2 <= 0.2))
    assert same_side.sum() > 10 and np.array_equal(a["swap2"][same_side], b["swap2"][same_side])
    for k in a:                                                               # where no tie: the same, bit for bit
        assert np.array_equal(a[k][far], b[k][far]), k


@pytest.mark.parametrize("backend", ["torch", "metal"])
def test_tie_sw2_keeps_the_order(backend):
    """tie_sw2, the first rule: at a near tie neither swap fires and the state keeps its order. The default's state is
    the tie-break's, permuted where the default swapped an odd number of times, bit for bit (the filter's update
    comes before the swaps). Elsewhere nothing changes. And the default does swap some of those near ties."""
    from tractline import ukf_metal
    if backend == "metal" and not ukf_metal.available():
        pytest.skip("no Metal")
    a, b = _one_step(backend, 0.0), _one_step(backend, 1.0, tie_sw2=True)
    m1, m2 = b["state"][:, 0:3], b["state"][:, 5:8]
    c = np.abs((m1 * m2).sum(1)) / np.linalg.norm(m1, axis=1) / np.linalg.norm(m2, axis=1)
    tie, far = c > np.cos(np.radians(0.5)), c < np.cos(np.radians(2.0))
    assert tie.sum() > 100 and far.sum() > 100
    assert not b["swap"][tie].any() and not b["swap2"][tie].any()
    odd = a["swap"] ^ a["swap2"]
    assert odd[tie].sum() > 10 and a["swap2"][tie].sum() > 10                  # the default swaps near ties, sw2 too
    sw_state, sw_P = _swapped(b["state"], b["P"])
    for name, mine, other in (("state", b["state"], sw_state), ("P", b["P"], sw_P)):
        want = np.where(odd.reshape((-1,) + (1,) * (mine.ndim - 1)), other, mine)
        assert np.array_equal(a[name][tie], want[tie]), name
    for k in a:
        assert np.array_equal(a[k][far], b[k][far]), k


def test_tie_angle_torch_and_metal_agree():
    """Both tie-break rules (tie_angle; with tie_sw2) are the same on the torch step (float32, fast) and the Metal
    kernel: one step's flags agree on more than 99.5 % of half-fibers, and the phantom's tractographies agree as the
    compiled step's does with the uncompiled one (counts within 2 %, more than 80 % of fibers within 0.1 mm)."""
    import torch
    from tractline import ukf as U, ukf_metal
    from tractline.prep import prepare
    if not ukf_metal.available():
        pytest.skip("no Metal")
    for sw2 in (False, True):                                                 # both rules
        a, b = _one_step("torch", 1.0, sw2), _one_step("metal", 1.0, sw2)
        for k in ("swap", "swap2", "stop"):
            assert (a[k] == b[k]).mean() > 0.995, k
        same = (a["swap"] == b["swap"]) & (a["swap2"] == b["swap2"])
        assert np.abs(a["x"] - b["x"])[same].max() < 1e-4
        assert np.abs(a["state"] - b["state"])[same].max() < 1e-2 * np.abs(a["state"]).max()
    s = subject()
    t = prepare(s.dwi, s.affine, s.bval, s.bvec, shell=2800.0)
    D = U.from_arrays(t.dwi, t.header, t.mask)
    for kw in ({"tie_angle": 2.0}, {"tie_angle": 2.0, "tie_sw2": True}):
        ft, st = U.track(D, dtype=torch.float32, fast=True, **kw)
        fm, sm = U.track(D, backend="metal", **kw)
        assert abs(len(ft) - len(fm)) <= 0.02 * len(ft)
        ref = dict(zip(st["seed_index"], ft))
        close = [k for k, f in zip(sm["seed_index"], fm) if k in ref and len(f) == len(ref[k]) and np.abs(f - ref[k]).max() < 0.1]
        assert len(close) > 0.8 * len(ft)
