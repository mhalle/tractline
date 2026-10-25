"""tractline.ptt_metal against tractline.fields and tractline.ptt (MPS only): the same PMF values, ACT decisions and candidate
likelihoods to float32 rounding, and the whole tracker on the synthetic band (the same statistics as the torch
prototype's; the random numbers differ, so not the same paths). Determinism: the same seed gives the same paths, and a
streamline's path does not depend on the batch it is in."""
import numpy as np
import pytest
import torch
from tractline import fields as F, ptt

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")


@pytest.fixture(scope="module")
def setup():
    from tractline import ptt_metal as M
    from tractline.csd import hemisphere
    rng = np.random.default_rng(0)
    shape = (9, 8, 7)
    coeffs = rng.normal(size=shape + (45,))
    mask = np.ones(shape, bool)
    v = hemisphere(181)
    pmf = F.PmfField(coeffs, mask, np.r_[v, -v], "mps")
    inc, exc = rng.uniform(size=(2,) + shape)
    act = F.Act(inc, exc, "mps")
    return M, pmf, act, shape, rng


def points(shape, n, rng):
    return rng.uniform(-1.0, np.asarray(shape, float), size=(n, 3))


def test_pmf_and_act_match_fields(setup):
    M, pmf, act, shape, rng = setup
    h = M.Halves(pmf, act, np.ones(3), ptt.Params())
    p = points(shape, 4000, rng)
    d = rng.normal(size=(len(p), 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ref, _ = pmf.value(p, d)
    assert torch.allclose(h.pmf(p, d), ref, atol=1e-5, rtol=1e-5)
    assert torch.equal(h.act(p), act(p).to(torch.int32))


@pytest.mark.parametrize("compat", [False, True])
def test_support_matches_prototype(setup, compat):
    M, pmf, act, shape, rng = setup
    vox = np.array([2.0, 2.1, 1.9])
    params = ptt.Params()
    h = M.Halves(pmf, act, vox, params, compat=compat)
    tr = ptt.Tracker(pmf, act, np.diag([*vox, 1.0]), vox, params, "mps", _dipy_compat=compat)
    n = 2000
    P = torch.as_tensor(rng.uniform(1, 6, size=(n, 3)), dtype=torch.float32, device="mps")
    T = torch.nn.functional.normalize(torch.randn(n, 3, device="mps"), dim=1)
    N2 = torch.nn.functional.normalize(torch.cross(T, torch.randn(n, 3, device="mps"), dim=1), dim=1)
    N1 = torch.cross(N2, T, dim=1)
    k = (torch.rand(n, 2, device="mps") - 0.5) * 2 * tr.max_k / 1.5
    last = torch.rand(n, device="mps")
    L, v = h.support(P, T, N1, N2, k, last)
    Lr, vr = tr._support(P, T, N1, N2, last, k[:, None])
    assert torch.allclose(L, Lr[:, 0], atol=2e-4, rtol=2e-4) and torch.allclose(v, vr[:, 0], atol=2e-4, rtol=2e-4)


def band_tracker(compat):
    import test_ptt as TP
    from tractline.csd import hemisphere
    c, m, inc, exc, affine = TP.band()
    v = hemisphere(181)
    pmf = F.PmfField(c, m, np.r_[v, -v], "mps")
    act = F.Act(inc, exc, "mps")
    return ptt.Tracker(pmf, act, affine, np.full(3, 2.0), ptt.Params(min_len=10), "mps", _dipy_compat=compat,
                       backend="metal"), c


@pytest.mark.parametrize("compat", [False, True])
def test_band_metal(compat):
    tr, c = band_tracker(compat)
    rng = np.random.default_rng(0)
    seeds = np.c_[rng.uniform(8, 30, 400), rng.uniform(5, 7, 400), rng.uniform(5, 7, 400)] * 2.0
    lines, info = tr.track(seeds, seed=3)
    assert info["kept"] > (0.9 if compat else 0.95) * len(seeds)
    cy = c.shape[1] // 2 * 2.0
    span = np.array([x[:, 0].max() - x[:, 0].min() for x in lines])
    assert np.mean(span > 50) > 0.95                                          # most cover most of the band's 80 mm (a rare
    for x in lines:                                                           # one turns back: the FOD supports +x and -x)
        assert (x[:, 1:] > cy - 9.0).all() and (x[:, 1:] < cy + 7.0).all()   # the band, its fade, and a step past it
    seg = np.concatenate([np.diff(x, axis=0) for x in lines])
    ang = np.degrees(np.arccos(np.clip(np.abs(seg[:, 0]) / np.linalg.norm(seg, axis=1), 0, 1))).mean()
    assert ang < 15
    again, _ = tr.track(seeds, seed=3)
    assert all(np.array_equal(a, b) for a, b in zip(again, lines))
    # batch independence: the seeds' second half tracked alone, and in batches of 70, gives each seed the same path
    full, _ = tr.track(seeds, seed=3, keep_index=True)
    part, _ = tr.track(seeds[200:], seed=3, offset=200, keep_index=True)
    small, _ = tr.track(seeds, seed=3, batch=70, keep_index=True)
    assert part and set(part) == {i for i in full if i >= 200}
    assert all(np.array_equal(full[i], x) for i, x in part.items()) and all(np.array_equal(full[i], x) for i, x in small.items())


def test_metal_compat_matches_dipy(tmp_path):
    """DIPY's own ptt_tracking (its process) against the Metal kernel with the validation switch and in real units, on the
    band at 2 mm voxels: as tests/test_ptt.py's test for the torch prototype."""
    import subprocess, sys
    import test_ptt as TP
    pytest.importorskip("dipy.tracking.tracker")
    vox = 2.0
    shape = (40, 12, 12)
    c, m, inc, exc, affine = TP.band(shape, vox)
    rng = np.random.default_rng(0)
    n = 400
    sv = np.c_[rng.uniform(0.2, 0.75, n) * shape[0], rng.uniform(-0.5, 0.5, n) + shape[1] // 2 - 0.5,
               rng.uniform(-0.5, 0.5, n) + shape[2] // 2 - 0.5]
    seeds = sv * vox
    src, out = tmp_path / "band.npz", tmp_path / "dipy.npz"
    np.savez(src, c=c, inc=inc, exc=exc, affine=affine, seeds=seeds)
    subprocess.run([sys.executable, "-I", "-c", TP.DIPY_PTT, str(src), str(out)], check=True)
    z = np.load(out)
    ref = np.split(z["points"], np.cumsum(z["lengths"])[:-1])
    from dipy.data import default_sphere
    pmf = F.PmfField(c, m, default_sphere.vertices, "mps")
    act = F.Act(inc, exc, "mps")
    ours = {u: ptt.Tracker(pmf, act, affine, np.full(3, vox), ptt.Params(min_len=10), "mps", _dipy_compat=u == "dipy",
                           backend="metal").track(seeds, seed=3)[0] for u in ("dipy", "mm")}
    p99 = {k: np.percentile(TP._turns(v), 99) for k, v in {"DIPY": ref, **ours}.items()}
    assert abs(len(ours["dipy"]) - len(ref)) <= 0.03 * len(ref)
    assert abs(p99["dipy"] - p99["DIPY"]) < 1.5 and p99["mm"] < 10.5
    long_seed = lambda x: np.sort(np.linalg.norm(np.diff(x, axis=0), axis=1))[-2] > 0.9    # noqa: E731
    assert np.mean([long_seed(x) for x in ours["dipy"]]) > 0.9
    assert max(np.linalg.norm(np.diff(x, axis=0), axis=1).max() for x in ours["mm"]) < 0.55


@pytest.mark.parametrize("compat", [False, True])
def test_launch_size_does_not_matter(compat):
    """The halves' state survives the launch boundaries: steps per launch 1, 7, 25 and 10^4 give bit-identical points,
    counts and status, with max_len 12.5 mm (25 points) so that halves reach the point limit at a launch boundary."""
    import test_ptt as TP
    from tractline import ptt_metal as M
    from tractline.csd import hemisphere
    c, m, inc, exc, affine = TP.band()
    v = hemisphere(181)
    pmf = F.PmfField(c, m, np.r_[v, -v], "mps")
    act = F.Act(inc, exc, "mps")
    rng = np.random.default_rng(1)
    sv = np.c_[rng.uniform(8, 30, 300), rng.uniform(5, 7, 300), rng.uniform(5, 7, 300)]
    d = np.tile([1.0, 0.0, 0.0], (300, 1))
    for max_len in (12.5, 250.0):
        h = M.Halves(pmf, act, np.full(3, 2.0), ptt.Params(max_len=max_len), compat=compat)
        runs = [h.run(sv, d, seed=5, steps=s) for s in (1, 7, 25, 10_000)]
        for pts, n, st in runs[1:]:
            assert torch.equal(pts, runs[0][0]) and torch.equal(n, runs[0][1]) and torch.equal(st, runs[0][2])
        if max_len == 12.5:
            assert int((runs[0][1] == h.maxp).sum()) > 0                    # some reached the limit
