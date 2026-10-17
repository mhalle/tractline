"""What the labelers share (lengths, LogMean, empty Labels) and RapidParc's edge cases. RapidParc needs
its released weights in $TRACTOGRAPHY_DATA/RapidParc (README); without them those tests skip."""
import numpy as np, torch
import pytest
from tractline.data import DATA
from tractline.labelers.base import LogMean, lengths

WEIGHTS = DATA / "RapidParc" / "rapidparc.safetensors"
needs_weights = pytest.mark.skipif(not WEIGHTS.exists(), reason=f"RapidParc's weights not in {WEIGHTS.parent}")


def line(n, step=1.0, start=0.0):
    return np.c_[np.arange(n) * step + start, np.zeros(n), np.zeros(n)].astype(np.float32)


def test_lengths_per_streamline():
    bad = line(5); bad[2, 0] = np.nan
    _, _, L = lengths([line(11), np.zeros((0, 3), np.float32), line(1), bad, line(41)])
    assert L[0] == 10 and L[1] == 0 and L[2] == 0 and np.isnan(L[3]) and L[4] == 40   # a NaN stays in its own


def test_lengths_empty_list():
    P, o, L = lengths([])
    assert len(P) == 0 and list(o) == [0] and len(L) == 0


def test_logmean_matches_the_mean_and_does_not_underflow():
    lp = torch.log_softmax(torch.randn(3, 5, 1600) * 4, -1)
    m = LogMean()
    for d in lp:
        m.add(d)
    assert torch.allclose(m.result(), lp.exp().mean(0).log(), atol=1e-4)
    tiny = LogMean(); tiny.add(torch.full((1, 3), -800.0)); tiny.add(torch.full((1, 3), -810.0))
    assert torch.isfinite(tiny.result()).all()                              # exp(-800) is 0 in float32


@pytest.fixture(scope="module")
def rapidparc():
    if not WEIGHTS.exists():
        pytest.skip(f"RapidParc's weights not in {WEIGHTS.parent}")
    from tractline.labelers.rapidparc import Labeler
    return Labeler("cpu")


def bundle(n, seed=0):
    rng = np.random.default_rng(seed)
    return [(line(60, 1.0, -30) + rng.normal(0, 3, 3)).astype(np.float32) for _ in range(n)]


@needs_weights
@pytest.mark.parametrize("n", [0, 1, 7, 999, 1000, 2500])
def test_rapidparc_any_count(rapidparc, n):
    fibers = bundle(n) + [line(10)]                                        # one too short to label
    L = rapidparc(fibers, logp=True)
    assert L.keep.sum() == n and len(L.tract) == n and L.logp.shape == (n, 1600)
    assert torch.isfinite(L.logp.float()).all()
    assert ((L.tract >= 0) & (L.tract <= 42)).all()


@needs_weights
def test_rapidparc_empty_and_draws(rapidparc):
    assert rapidparc([line(10)]).logp is None                              # logp only when asked
    a = rapidparc(bundle(300), draws=(0, 1, 2))
    b = rapidparc(bundle(300), draws=(d for d in range(3)))                # any iterable
    assert np.array_equal(a.tract, b.tract)
    assert np.array_equal(rapidparc(bundle(300), draws=(5,)).tract, rapidparc(bundle(300), draws=(5,)).tract)


def curves(n, seed=0, points=80):
    """n smooth random 79 mm curves across a brain-sized box: varied enough that labels depend on context."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    out, p = [], rng.uniform(-50, 50, (n, 3))
    for i in range(n):
        v, pts = d[i], [p[i]]
        for _ in range(points - 1):
            v = v + rng.normal(0, 0.08, 3); v /= np.linalg.norm(v)
            pts.append(pts[-1] + v)
        out.append(np.asarray(pts, np.float32))
    return out


def test_splitmix64_reference_value():
    from tractline.labelers.rapidparc import splitmix64
    assert int(splitmix64(np.uint64(0))) == 0xE220A8397B1DCDAF                # splitmix64's first output from state 0


def test_keyed_groups_partition_the_streamlines():
    from tractline.labelers.rapidparc import keyed_groups, GROUPINGS
    keys = np.random.default_rng(1).choice(1 << 36, 4500, replace=False)
    for g in GROUPINGS[1:]:
        groups = keyed_groups(keys, 3, g)
        assert [list(x) for x in groups] == [list(x) for x in keyed_groups(keys, 3, g)]
        idx = np.concatenate(groups)
        assert np.array_equal(np.sort(idx[:len(keys)] if g == "hash-sorted" else idx), np.arange(len(keys)))
    assert len(keyed_groups(keys, 3, "hash-bucket")) == 3                  # ceil(4500 / 2000)


@needs_weights
def test_rapidparc_default_grouping_is_rapidparcs_shuffle(rapidparc):
    # pins the default to the shuffle as written here; the check against RapidParc's package is bench/rapidparc_check.py
    from tractline.labelers.rapidparc import resample, normalize, CONTEXT, POINTS
    fibers = curves(2300)
    feat = resample(fibers)
    perm = torch.randperm(len(feat), generator=torch.Generator().manual_seed(4))
    data = normalize(feat[None])[0][perm]
    data = torch.cat([data, data])[:2 * CONTEXT].reshape(-1, CONTEXT, POINTS, 3)
    with torch.inference_mode():
        ref = torch.empty(len(feat), 1600)
        ref[perm] = rapidparc.model(data).reshape(-1, 1600)[:len(feat)]
    assert torch.equal(rapidparc.logits(feat, 4), ref)
    keys = np.arange(len(fibers)) * 7
    a, b = rapidparc(fibers, draws=(4,), logp=True), rapidparc(fibers, draws=(4,), logp=True, keys=keys, grouping="index")
    assert np.array_equal(a.tract, b.tract) and torch.equal(a.logp, b.logp)    # index ignores keys


@needs_weights
def test_rapidparc_keyed_groupings_need_keys(rapidparc):
    with pytest.raises(ValueError):
        rapidparc(curves(5), grouping="hash-bucket")
    with pytest.raises(ValueError):
        rapidparc(curves(5), keys=np.arange(4), grouping="hash-sorted")
    with pytest.raises(ValueError):
        rapidparc(curves(5), keys=np.arange(5), grouping="hash")
    with pytest.raises(ValueError, match="unique"):
        rapidparc(curves(5), keys=np.array([1, 2, 3, 3, 4]), grouping="hash-sorted")
    rapidparc(curves(4) + [line(10)], keys=np.array([1, 2, 3, 4, 4]), grouping="hash-sorted")   # repeated only among the unlabeled
    L = rapidparc(curves(5) + [line(10)], keys=np.arange(6), grouping="hash-bucket", draws=(0, 1), logp=True)
    assert L.keep.sum() == 5 and L.logp.shape == (5, 1600)


@needs_weights
def test_rapidparc_hash_bucket_other_buckets_bit_for_bit(rapidparc):
    from tractline.labelers.rapidparc import resample, keyed_groups
    fibers = curves(4500)
    keys = np.random.default_rng(2).choice(1 << 36, len(fibers), replace=False)
    feat = resample(fibers)
    groups = keyed_groups(keys, 0, "hash-bucket")
    drop = groups[1][::150]                                                # ~13 streamlines, all from bucket 1
    rest = np.setdiff1d(np.arange(len(fibers)), drop)
    full = rapidparc.logits(feat, 0, keys, "hash-bucket")
    less = torch.empty(len(fibers), 1600)
    less[rest] = rapidparc.logits(feat[rest], 0, keys[rest], "hash-bucket")
    for g in (groups[0], groups[2]):
        assert torch.equal(full[g], less[g])
    b1 = np.setdiff1d(groups[1], drop)
    assert not torch.equal(full[b1], less[b1])                             # bucket 1 itself has new company


@needs_weights
def test_rapidparc_hash_sorted_removal_renames_few(rapidparc):
    fibers = curves(4500, seed=3)
    keys = np.random.default_rng(3).choice(1 << 36, len(fibers), replace=False)
    from tractline.labelers.rapidparc import keyed_groups
    gone = keyed_groups(keys, 0, "hash-sorted")[0][5]                      # early in the order: every later group shifts
    rest = np.delete(np.arange(len(fibers)), gone)
    sub = [fibers[i] for i in rest]
    changed = {}
    for g in ("index", "hash-sorted"):
        a = rapidparc(fibers, keys=keys, grouping=g)
        b = rapidparc(sub, keys=keys[rest], grouping=g)
        changed[g] = int((a.tract[rest] != b.tract).sum())
    assert changed["index"] > 20 and changed["hash-sorted"] * 4 < changed["index"], changed


@needs_weights
@pytest.mark.parametrize("n", [1500, 4000])
def test_rapidparc_hash_sorted_groups_and_padding(rapidparc, n):
    """below one group (padded) and at an exact multiple of 2,000 (no padding): each streamline's logits are its
    hash-sorted group's"""
    from tractline.labelers.rapidparc import resample, keyed_groups
    feat = resample(curves(n, seed=5))
    keys = np.random.default_rng(5).choice(1 << 36, n, replace=False)
    groups = keyed_groups(keys, 1, "hash-sorted")
    assert len(groups) == -(-n // 2000) and all(len(g) == 2000 for g in groups)
    ref = torch.full((n, 1600), float("nan"))
    with torch.inference_mode():
        for a, g in enumerate(groups):
            real = min(2000, n - 2000 * a)                                 # rows past n are padding
            ref[torch.from_numpy(g[:real])] = rapidparc.model(feat[torch.from_numpy(g)][None])[0, :real]
    out = rapidparc.logits(feat, 1, keys, "hash-sorted")
    assert not ref.isnan().any()
    assert torch.allclose(out, ref, rtol=1e-4, atol=1e-3)                # batched groups: the CPU's rounding differs
