"""Compact run records (tractline.compact; the `rankfield` dependency group): the geometry decodes onto its grid,
the field keeps every label and nearly every probability, and a compact RunRecord reads back as the dense one does."""
from types import SimpleNamespace
import numpy as np
import pytest

pytest.importorskip("numcodecs")
pytest.importorskip("rankfield")
torch = pytest.importorskip("torch")

from tractline import compact as C, ensemble as E          # noqa: E402
from tractline.labelers.base import LUT                    # noqa: E402


def walks(n=300, seed=0):
    """Smooth random streamlines of 1 to 120 points, 1 mm steps, about a center far from the origin."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        m = 1 if i == 0 else 2 if i == 1 else int(rng.integers(3, 120))
        d = np.cumsum(rng.normal(0, 0.15, (m, 3)), 0) + rng.normal(0, 1, 3)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
        out.append((np.cumsum(d, 0) + rng.uniform(-60, 60, 3) + [10.0, -20.0, 30.0]).astype(np.float32))
    return out


def test_geometry_round_trip_is_on_the_grid():
    fibers = walks()
    back = C.decode_geometry(C.encode_geometry(fibers))
    assert [len(f) for f in back] == [len(f) for f in fibers]
    err = max(float(np.abs(a - b).max()) for a, b in zip(back, fibers))
    assert err <= C.QUANTUM_MM / 2 + 1e-4, err


def test_geometry_is_compact():
    fibers = walks(400, seed=1)
    g = C.encode_geometry(fibers)
    nbytes = sum(v.nbytes for k, v in g.items() if k != "geom_meta")
    assert nbytes / sum(len(f) for f in fibers) < 3.0             # float32 is 12 B per vertex


def logps(n=500, C_=1600, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, C_)).astype(np.float32)
    x[np.arange(n), rng.integers(0, C_, n)] += rng.uniform(2, 12, n)        # a winner, sometimes contested
    return torch.log_softmax(torch.from_numpy(x), 1).numpy()


def test_field_keeps_labels_and_probabilities():
    lp = logps()
    back = C.decode_field(C.encode_field(lp)).astype(np.float32)
    assert back.shape == lp.shape
    p, q = np.exp(lp), np.exp(back)
    assert np.allclose(q.sum(1), 1, atol=2e-3)
    assert (q.argmax(1) == p.argmax(1)).mean() > 0.995
    onehot = np.eye(E.N_TRACTS)[LUT]
    assert np.abs(q @ onehot - p @ onehot).max() < 0.05     # the tail is spread evenly; these synthetic tails are heavy
                                                              # (a real record of 41,595 streamlines: at most 0.005)


def test_compact_record_reads_back_as_the_dense_one(tmp_path):
    fibers = [f * 0.6 + [5, 5, 5] for f in walks(200, seed=3)]
    n = len(fibers)
    lp = torch.from_numpy(logps(n, seed=4)).to(torch.float16)
    tract = LUT[lp.float().numpy().argmax(1)]
    seeds = np.stack([np.arange(n) % 40, np.arange(n) // 40, np.full(n, 7)], 1)
    tg = SimpleNamespace(fibers=fibers, stats={"seed_voxel": seeds})
    labels = SimpleNamespace(logp=lp, keep=np.ones(n, bool), tract=tract)
    affine, shape = np.diag([2.0, 2.0, 2.0, 1.0]), (64, 64, 64)
    rec = E.RunRecord.of(tg, labels, affine, shape)
    rec.save(tmp_path / "dense.npz")
    rec.save(tmp_path / "compact.npz", compact=True)
    a, b = E.RunRecord.load(tmp_path / "dense.npz"), E.RunRecord.load(tmp_path / "compact.npz", fibers=True)
    assert (a.keys == b.keys).all() and (a.cluster == b.cluster).all() and (a.counts == b.counts).all() and a.shape == b.shape
    assert np.abs(a.paths - b.paths).max() <= C.QUANTUM_MM                           # rebuilt from the decoded streamlines
    assert max(float(np.abs(f - g).max()) for f, g in zip(rec.fibers, b.fibers)) <= C.QUANTUM_MM / 2 + 1e-4
    pa, pb = np.exp(a.logp.astype(np.float32)), np.exp(b.logp.astype(np.float32))
    assert np.abs(pa - pb).max() < 0.02
    assert (tmp_path / "compact.npz").stat().st_size < (tmp_path / "dense.npz").stat().st_size / 20
    assert E.RunRecord.load(tmp_path / "compact.npz", logp=False).logp is None
    with pytest.raises(ValueError):
        E.RunRecord.load(tmp_path / "dense.npz", fibers=True)
