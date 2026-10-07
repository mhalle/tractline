"""tractline.ensemble on synthetic runs: matching by seed voxel, the mixture, votes, their planes and regrouping,
voxel counts and the voxel field."""
import numpy as np, torch
from tractline import ensemble as E
from tractline.labelers.base import LUT


def record(keys, cluster, shape=(4, 4, 4), paths=None, counts=None, sharp=8.0):
    n = len(keys)
    lg = np.zeros((n, 1600), np.float32)
    lg[np.arange(n), cluster] = sharp
    logp = torch.log_softmax(torch.from_numpy(lg), 1).numpy().astype(np.float16)
    paths = np.zeros((n, E.POINTS, 3), np.float32) if paths is None else paths
    counts = np.zeros((E.N_TRACTS, int(np.prod(shape))), np.uint16) if counts is None else counts
    return E.RunRecord(np.asarray(keys, np.int64), np.asarray(cluster, np.int16), logp, paths, counts, shape)


def test_seed_keys_unique_and_stable():
    v = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [4095, 4095, 4095]])
    k = E.seed_keys(v)
    assert len(np.unique(k)) == len(k)
    assert (E.seed_keys(v[::-1]) == k[::-1]).all()


def test_streamline_ensemble_matches_votes_and_mixture():
    # clusters of two tracts, and one more of the first tract
    t0 = np.flatnonzero(LUT == 0)[:2]
    t1 = np.flatnonzero(LUT == 1)[0]
    ref = record([10, 20, 30], [t0[0], t0[0], t1])
    S = E.StreamlineEnsemble(ref)
    S.add(ref)
    S.add(record([30, 20, 99], [t1, t0[1], t0[0]]))          # key 10 absent, 99 not in the reference
    S.add(record([20, 10], [t1, t0[0]]))
    assert S.present.tolist() == [2, 3, 2]
    v = S.votes()
    assert v[0, 0] == 2 and v[1, 0] == 2 and v[1, 1] == 1 and v[2, 1] == 2
    assert (v.sum(1) == S.present).all()
    mix = S.mixture().exp().sum(1)
    assert torch.allclose(mix, torch.ones(3), atol=1e-2)       # each row a distribution
    # votes from per-rank planes regroup exactly when every winner is kept
    ranks = np.zeros((3, 3), np.int64)
    ranks[0] = [t0[0] + 1, t0[0] + 1, t1 + 1]
    ranks[1] = [0, t0[1] + 1, 0]
    ranks[2] = [0, t1 + 1, 0]
    allv = np.zeros((1600, 3), np.int64)
    for w in S.winners:
        ok = w >= 0
        np.add.at(allv, (w[ok], np.flatnonzero(ok)), 1)
    vp, tail = E.vote_planes(ranks, allv, S.present)
    assert tail.tolist() == [0, 0, 0]
    g = E.group_votes(ranks, vp, LUT, E.N_TRACTS)
    assert (g.T == v).all()
    # a winner the field dropped shows in the tail, and the regrouped votes fall short by it
    vp2, tail2 = E.vote_planes(ranks[:1], allv, S.present)
    assert tail2.tolist() == [0, 2, 0]


def test_mdf_orientation_free():
    a = np.random.default_rng(0).normal(size=(5, E.POINTS, 3)).astype(np.float32)
    assert np.allclose(E.mdf(a, a[:, ::-1]), 0)
    assert np.allclose(E.mdf(a, a + np.array([3.0, 4.0, 0.0], np.float32)), 5, atol=1e-5)


def test_resample_equal_arc_length():
    f = np.array([[0, 0, 0], [1, 0, 0], [10, 0, 0]], float)
    r = E.resample([f], points=11)[0]
    assert np.allclose(r[:, 0], np.arange(11)) and np.allclose(r[:, 1:], 0)


def test_voxel_counts_once_per_streamline_and_field():
    shape, aff = (4, 4, 4), np.eye(4)
    f1 = np.array([[0, 0, 0], [0.2, 0, 0], [1, 0, 0]], float)      # voxel (0,0,0) twice, then (1,0,0)
    f2 = np.array([[1, 0, 0], [2, 0, 0]], float)
    c = E.voxel_counts([f1, f2], np.array([3, 5]), aff, shape)
    flat = lambda *ijk: np.ravel_multi_index(ijk, shape)
    assert c[3, flat(0, 0, 0)] == 1 and c[3, flat(1, 0, 0)] == 1 and c[5, flat(1, 0, 0)] == 1 and c.sum() == 4
    V = E.VoxelEnsemble(shape, tau=1.0)
    V.add(record([1], [0], shape=shape, counts=c))
    V.add(record([1], [0], shape=shape, counts=np.zeros_like(c)))
    lg = V.logits()
    assert lg.shape == (44, *shape)
    w = lg.reshape(44, -1).argmax(0)
    assert w[flat(0, 0, 0)] == E.NONE                            # mean density 0.5 < tau
    votes = V.votes().reshape(44, -1)
    assert votes[3, flat(0, 0, 0)] == 1 and votes[E.NONE, flat(0, 0, 0)] == 1
    assert (votes.sum(0) == 2).all()
