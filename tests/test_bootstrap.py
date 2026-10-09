"""bench/_bootstrap.py's ShellWildBootstrap: what each b0 noise mode does to the b0s (the field is estimated from
them) - centered on their mean (residual, thermal) or on each b0 as acquired (added), at which noise level."""
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
def test_b0_modes_center_and_noise(mode):
    dwi, bval, bvec, extra, mask = scan()
    wb = ShellWildBootstrap(dwi, bval, bvec, extra_b0s=extra, b0_noise=mode, mask=mask)
    reps = [wb.replicate(s) for s in range(300)]
    d = np.stack([r[0][..., :6] for r in reps]).astype(np.float64)
    e = np.stack([r[1] for r in reps])
    b0, b0_mean = dwi[..., :6].astype(np.float64), dwi[..., :6].mean(-1, keepdims=True)
    center, center_extra = (b0, extra) if mode == "added" else (b0_mean, extra.mean(-1, keepdims=True))
    assert np.abs(d.mean(0) - center).mean() < 3 and np.abs(e.mean(0) - center_extra).mean() < 3
    # the replicates' spread about that center: the b0s' own (drift included), or the DW residual level
    sd = float(np.sqrt(((d - center) ** 2).mean()))
    rms_dw = wb.noise_check(mask)["dw_residual_rms"]
    if mode == "residual":
        assert sd > 2 * rms_dw
    else:
        assert sd == pytest.approx(rms_dw, rel=0.1)
        assert wb.b0_scale < 1


def test_raw_residuals_and_flipped_b0s_make_the_scan_a_replicate():
    # leverage off, b0s flipped in place: every sign +1 gives the scan back
    dwi, bval, bvec, extra, mask = scan()
    wb = ShellWildBootstrap(dwi, bval, bvec, extra_b0s=extra, b0_noise="flipped", mask=mask, leverage=False)
    assert np.allclose(wb.fit + wb.res, dwi.reshape(-1, 36), atol=1e-3)
    fit, res, _ = wb.extra
    assert np.allclose(fit + res, extra.reshape(-1, 3))


def test_volume_flips_share_one_sign_across_voxels():
    dwi, bval, bvec, extra, mask = scan()
    wb = ShellWildBootstrap(dwi, bval, bvec, b0_noise="added", mask=mask, flip="volume")
    x = wb.replicate(3)[0].reshape(-1, 36).astype(np.float64)
    big = np.abs(wb.res) > 2                                                    # signs readable past rounding
    sign = np.where(big, np.sign((x - wb.fit) * wb.res), 0)
    assert all(len(set(sign[big[:, g], g].tolist())) == 1 for g in range(36) if big[:, g].any())


def test_double_bootstrap_recalibration_helpers():
    pytest.importorskip("torch")
    from ensemble_double import isotonic, platt, held_out, vote_table
    rng = np.random.default_rng(0)
    z = rng.normal(size=50000)
    y = (rng.random(z.size) < 1 / (1 + np.exp(-2 * z))).astype(float)
    a, b = platt(z, y)
    assert abs(a) < 0.05 and abs(b - 2) < 0.1
    x, p = isotonic(z, y)
    assert np.all(np.diff(x) > 0) and np.all(np.diff(p) >= 0)
    assert np.abs(np.interp(z, x, p) - 1 / (1 + np.exp(-2 * z))).mean() < 0.02
    # held-out scoring per group: a table fitted on the other groups, scored on the left-out one
    v = rng.integers(0, 4, z.size)
    yv = (rng.random(z.size) < v / 3).astype(float)
    g = np.repeat(np.arange(4), z.size // 4)
    out = held_out(lambda v, y: vote_table(v, y, 3), lambda t, v: t[v], g, v, yv)
    assert set(out) == {"r000", "r001", "r002", "r003", "mean"} and out["mean"]["ece"] < 0.02


def test_one_extra_b0_borrows_a_noise_pattern():
    dwi, bval, bvec, extra, mask = scan()
    one = extra[..., :1]
    for mode in ("thermal", "added", "flipped"):
        wb = ShellWildBootstrap(dwi, bval, bvec, extra_b0s=one, b0_noise=mode, mask=mask)
        e = np.stack([wb.replicate(s)[1] for s in range(200)])
        assert np.isfinite(e).all()
        sd = float(np.sqrt(((e - e.mean(0)) ** 2).mean()))
        assert sd == pytest.approx(wb.noise_check(mask)["dw_residual_rms"], rel=0.15)
    with pytest.raises(AssertionError):
        ShellWildBootstrap(dwi, bval, bvec, extra_b0s=one, b0_noise="residual")
