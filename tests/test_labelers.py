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
