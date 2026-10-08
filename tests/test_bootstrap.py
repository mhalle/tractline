"""bench/_bootstrap.py's ShellWildBootstrap: what each b0 noise mode does to the b0s (the field is estimated from
them) - centred on their mean (residual, thermal) or on each b0 as acquired (added), at which noise level."""
import sys
from pathlib import Path
import numpy as np
import pytest

pytest.importorskip("dipy.reconst.shm")                                          # the dipy group
sys.path.insert(0, str(Path(__file__).parents[1] / "bench"))
from _bootstrap import ShellWildBootstrap  # noqa: E402


def scan(seed=0):
    """6 b0s drifting by 30 a volume (motion stand-in), 30 directions at b 1000, 3 extra b0s; noise SD 20 / 40."""
    rng = np.random.default_rng(seed)
    bval = np.r_[np.zeros(6), np.full(30, 1000.0)]
    bvec = rng.normal(size=(3, 36))
    bvec /= np.linalg.norm(bvec, axis=0)
    dwi = rng.normal(500, 20, (4, 4, 4, 36)).astype(np.float32)
    dwi[..., :6] += np.arange(6) * 30
    extra = rng.normal(800, 40, (4, 4, 4, 3))
    return dwi, bval, bvec, extra, np.ones((4, 4, 4))


@pytest.mark.parametrize("mode", ["residual", "thermal", "added"])
def test_b0_modes_centre_and_noise(mode):
    dwi, bval, bvec, extra, mask = scan()
    wb = ShellWildBootstrap(dwi, bval, bvec, extra_b0s=extra, b0_noise=mode, mask=mask)
    reps = [wb.replicate(s) for s in range(300)]
    d = np.stack([r[0][..., :6] for r in reps]).astype(np.float64)
    e = np.stack([r[1] for r in reps])
    b0, b0_mean = dwi[..., :6].astype(np.float64), dwi[..., :6].mean(-1, keepdims=True)
    centre, centre_extra = (b0, extra) if mode == "added" else (b0_mean, extra.mean(-1, keepdims=True))
    assert np.abs(d.mean(0) - centre).mean() < 3 and np.abs(e.mean(0) - centre_extra).mean() < 3
    # the replicates' spread about that centre: the b0s' own (drift included), or the DW residual level
    sd = float(np.sqrt(((d - centre) ** 2).mean()))
    rms_dw = wb.noise_check(mask)["dw_residual_rms"]
    if mode == "residual":
        assert sd > 2 * rms_dw
    else:
        assert sd == pytest.approx(rms_dw, rel=0.1)
        assert wb.b0_scale < 1
