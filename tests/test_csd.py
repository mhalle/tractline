"""tractline.csd against DIPY's CSD (dipy only for the check), on a small synthetic volume: single fibers, crossings and
isotropic voxels at b = 2000, with Rician-like noise. With DIPY's own constraint sphere (small_sphere), the method itself is
compared: the same response and the same coefficients to rounding, voxel for voxel. With our own sphere, close."""
import warnings
import numpy as np
import pytest
from tractline import csd, sh

pytest.importorskip("dipy.reconst.csdeconv")


def volume(seed=0):
    from dipy.core.gradients import gradient_table
    from dipy.sims.voxel import multi_tensor
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(60, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    bval = np.r_[np.zeros(5), np.full(60, 2000.0)]
    bvec = np.r_[np.zeros((5, 3)), g]
    gtab = gradient_table(bval, bvecs=bvec)
    shape = (12, 12, 12)
    data = np.zeros(shape + (65,))
    ev = np.array([1.7e-3, 0.2e-3, 0.2e-3])
    for idx in np.ndindex(shape):
        kind = rng.integers(3) if np.abs(np.array(idx) - 6).max() > 2 else 0   # single fibers at the center (the response)
        a = rng.normal(size=3)
        a /= np.linalg.norm(a)
        if kind == 0:
            mevals, angles, fr = [ev], [(0, 0)], [100]
            a = np.array([1.0, 0.0, 0.0])
        elif kind == 1:
            p = np.cross(a, rng.normal(size=3)); p /= np.linalg.norm(p)
            ang = np.radians(rng.uniform(45, 90))                               # crossings at 45-90 degrees
            b = np.cos(ang) * a + np.sin(ang) * p
            mevals, fr = [ev, ev], [55, 45]
            angles = [tuple(np.degrees(np.arccos(np.clip(v[2], -1, 1))) for v in (a,)) + (np.degrees(np.arctan2(a[1], a[0])),),
                      tuple(np.degrees(np.arccos(np.clip(v[2], -1, 1))) for v in (b,)) + (np.degrees(np.arctan2(b[1], b[0])),)]
        else:
            mevals, angles, fr = [np.full(3, 0.8e-3)], [(0, 0)], [100]
        if kind == 0:
            angles = [(90.0, 0.0)]
        s, _ = multi_tensor(gtab, np.array(mevals), S0=1000.0, angles=angles, fractions=fr, snr=None)
        noise = rng.normal(scale=25.0, size=(2, 65))
        data[idx] = np.sqrt((s + noise[0]) ** 2 + noise[1] ** 2)
    return gtab, bval, bvec, data


@pytest.fixture(scope="module")
def case():
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel, auto_response_ssst
    from dipy.data import small_sphere
    gtab, bval, bvec, data = volume()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resp, _ = auto_response_ssst(gtab, data, roi_radii=10, fa_thr=0.7)
        model = ConstrainedSphericalDeconvModel(gtab, resp, sh_order_max=8)
        ref = model.fit(data).shm_coeff
    return gtab, bval, bvec, data, resp, ref, small_sphere.vertices


def test_response_is_dipys(case):
    _, bval, bvec, data, resp, _, _ = case
    r = csd.response_ssst(data, bval, bvec.T)
    assert np.allclose(r.evals, resp[0], rtol=1e-9) and np.isclose(r.S0, resp[1], rtol=1e-12)


def test_fit_is_dipys_on_its_sphere(case):
    _, bval, bvec, data, resp, ref, small = case
    r = csd.Response(np.asarray(resp[0]), float(resp[1]), 0, 0.0)
    f, info = csd.CSD(bval, bvec.T, r, order=8, sphere=small).fit(data, device="cpu")
    assert info["not converged"] == 0
    err = np.abs(f - ref).max() / np.abs(ref).max()
    assert err < 1e-5                                                         # float32 output; the fit is float64


def test_fit_close_on_our_sphere(case):
    """Our constraint sphere against DIPY's, where there is a fiber (FA >= 0.3): FOD correlation above 0.995 in 99 % of
    those voxels and relative L2 difference under 0.08 at the 99th percentile (measured on the 90-degree version of this
    volume, 2026-10-24: 0.999 and 0.05). In isotropic voxels
    the FOD is shaped by noise and by where the constraint samples it, and the two spheres differ there (median
    correlation 0.88): not compared. On real scans the spheres differ more (bench/csd_check.py, cIIIs04: relative L2
    0.26 at the 99th percentile), and DIPY's own sphere rotated at random differs from itself as much: the constraint's
    sampling, not our sphere."""
    _, bval, bvec, data, resp, ref, _ = case
    r = csd.Response(np.asarray(resp[0]), float(resp[1]), 0, 0.0)
    f, _ = csd.CSD(bval, bvec.T, r, order=8).fit(data, device="cpu")
    ev, _ = csd.tensor_fit(data.reshape(-1, data.shape[-1]), bval, bvec.T)
    fiber = csd.fa(ev) >= 0.3
    B = sh.basis(8, np.r_[csd.hemisphere(300), -csd.hemisphere(300)])
    a, b = (f.reshape(-1, 45) @ B.T)[fiber], (ref.reshape(-1, 45) @ B.T)[fiber]
    corr = np.array([np.corrcoef(x, y)[0, 1] for x, y in zip(a, b)])
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1)
    assert fiber.sum() > 500 and np.quantile(corr, 0.01) > 0.995 and np.quantile(rel, 0.99) < 0.08


def test_hemisphere():
    s = csd.hemisphere(181)
    assert s.shape == (181, 3) and np.allclose(np.linalg.norm(s, axis=1), 1) and (s[:, 2] > 0).all()
    axial = np.abs(s @ s.T)
    np.fill_diagonal(axial, 0)
    nn = np.degrees(np.arccos(axial.max(1)))
    assert nn.min() > 9.5 and nn.max() < 12.5                                # even, as axes (DIPY's: 10.6-11.7)


@pytest.mark.skipif(not __import__("torch").backends.mps.is_available(), reason="no MPS")
def test_mps_float32_close_to_cpu(case):
    """MPS has no float64: the iterations there are float32 (the first step is float64 on the host). In fiber voxels,
    within 1 % of the largest coefficient (measured 2026-10-24: 0.18 %)."""
    _, bval, bvec, data, _, _, _ = case
    r = csd.response_ssst(data, bval, bvec.T)
    m = csd.CSD(bval, bvec.T, r, order=8)
    a, _ = m.fit(data, device="cpu")
    b, info = m.fit(data, device="mps")
    ev, _ = csd.tensor_fit(data.reshape(-1, data.shape[-1]), bval, bvec.T)
    fiber = csd.fa(ev) >= 0.3
    x, y = a.reshape(-1, 45)[fiber], b.reshape(-1, 45)[fiber]
    assert info["not converged"] == 0 and np.abs(x - y).max() < 0.01 * np.abs(x).max()


def test_mask_and_chunks(case, monkeypatch):
    """A mask fits only its voxels (zero elsewhere), and chunk boundaries change nothing."""
    _, bval, bvec, data, _, _, _ = case
    r = csd.response_ssst(data, bval, bvec.T)
    m = csd.CSD(bval, bvec.T, r, order=8)
    whole, _ = m.fit(data)
    mask = np.zeros(data.shape[:3], bool)
    mask[2:9, 3:10, 1:7] = True
    monkeypatch.setattr(csd, "CHUNK", 37)
    part, info = m.fit(data, mask)
    assert info["voxels"] == mask.sum() and not part[~mask].any()
    assert np.array_equal(part[mask], whole[mask])


def test_fewer_directions_than_coefficients():
    """30 directions at order 8 (45 coefficients): X^T X is singular. The CPU fit (1e-5 I added) matches DIPY's, and MPS
    does not attempt it in float32: it runs on the CPU and says so."""
    import torch
    from dipy.core.gradients import gradient_table
    from dipy.data import small_sphere
    from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
    from dipy.sims.voxel import multi_tensor
    rng = np.random.default_rng(1)
    g = rng.normal(size=(30, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    bval, bvec = np.r_[0.0, np.full(30, 2000.0)], np.r_[np.zeros((1, 3)), g]
    gtab = gradient_table(bval, bvecs=bvec)
    ev = np.array([1.7e-3, 0.2e-3, 0.2e-3])
    data = np.stack([multi_tensor(gtab, np.array([ev, ev]), S0=1000.0, angles=[(90, 0), (90, a)], fractions=[50, 50],
                                  snr=40, rng=rng)[0] for a in rng.uniform(50, 90, 200)]).reshape(10, 20, 1, 31)
    resp = (ev, 1000.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = ConstrainedSphericalDeconvModel(gtab, resp, sh_order_max=8).fit(data).shm_coeff
    m = csd.CSD(bval, bvec, csd.Response(ev, 1000.0, 0, 0.0), order=8, sphere=small_sphere.vertices)
    assert m.cond > csd.COND_FLOAT32
    f, _ = m.fit(data)
    assert np.abs(f - ref).max() < 1e-4 * np.abs(ref).max()
    if torch.backends.mps.is_available():
        g2, info = m.fit(data, device="mps")
        assert info["device"] == "cpu" and np.array_equal(g2, f)


def test_bvec_layout():
    r = csd.Response(np.array([1.7e-3, 0.2e-3, 0.2e-3]), 1000.0, 0, 0.0)
    g = np.random.default_rng(0).normal(size=(50, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    bval = np.r_[0.0, np.full(50, 2000.0)]
    bvec = np.r_[np.zeros((1, 3)), g]
    assert np.array_equal(csd.CSD(bval, bvec, r).X, csd.CSD(bval, bvec.T, r).X)
    with pytest.raises(ValueError, match="cannot be told apart"):
        csd.CSD(np.r_[0.0, 2000.0, 2000.0], np.eye(3), r)
    with pytest.raises(ValueError, match="expected"):
        csd.CSD(bval, bvec[:-1], r)
