"""tractline.fields against DIPY's own trilinear interpolation, PMF lookup and ACT decision (dipy only for the check): the
same values at random points, including the half-voxel border, the edges and outside."""
import warnings
import numpy as np
import pytest
from tractline import fields as F, sh

pytest.importorskip("dipy.core.interpolation")


def points(shape, n=3000, seed=0):
    """Random voxel-coordinate points over the grid and a voxel beyond it, plus the exact border cases."""
    rng = np.random.default_rng(seed)
    s = np.asarray(shape, float)
    p = rng.uniform(-1.0, s, size=(n, 3))
    edge = np.array([[-0.5, 0, 0], [-0.5000001, 0, 0], [s[0] - 0.5, 1, 1], [s[0] - 0.5000001, 1, 1], [0, 0, 0],
                     [s[0] - 1, s[1] - 1, s[2] - 1], [s[0] - 0.75, s[1] - 0.75, s[2] - 0.75]])
    return np.r_[p, edge]


def test_trilinear_is_dipys():
    from dipy.core.interpolation import trilinear_interpolate4d
    rng = np.random.default_rng(1)
    vol = rng.normal(size=(7, 8, 6, 3))
    p = points(vol.shape[:3])
    ours, inside = F.Grid(vol)(p)
    for x, v, ok in zip(p, ours.numpy(), inside.numpy()):
        try:
            ref = trilinear_interpolate4d(vol, np.ascontiguousarray(x))
            assert ok and np.allclose(v, ref, atol=1e-12)
        except IndexError:
            assert not ok


def test_pmf_value_is_dipys():
    from dipy.data import default_sphere
    from dipy.direction.pmf import SHCoeffPmfGen
    rng = np.random.default_rng(2)
    coeffs = rng.normal(size=(6, 7, 5, 45))
    mask = np.ones(coeffs.shape[:3], bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gen = SHCoeffPmfGen(coeffs, default_sphere, basis_type="descoteaux07", legacy=True)
    field = F.PmfField(coeffs, mask, default_sphere.vertices)
    p = points(coeffs.shape[:3], n=1000)
    d = rng.normal(size=(len(p), 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ours, inside = field.value(p, d)
    for x, u, v, ok in zip(p, d, ours.numpy(), inside.numpy()):
        ref = gen.get_pmf_value(np.ascontiguousarray(x), np.ascontiguousarray(u))
        assert np.isclose(v, ref, atol=1e-9) if ok else ref == 0.0


def test_pmf_mask_rows_are_zero():
    rng = np.random.default_rng(3)
    coeffs = rng.normal(size=(4, 4, 4, 15))
    mask = np.zeros(coeffs.shape[:3], bool)
    mask[1:3, 1:3, 1:3] = True
    with pytest.raises(ValueError, match="outside the mask"):                # DIPY would read them: refused
        F.PmfField(coeffs, mask, np.eye(3))
    coeffs[~mask] = 0
    assert np.array_equal(F.PmfField(coeffs, None, np.eye(3)).table, F.PmfField(coeffs, mask, np.eye(3)).table)
    field = F.PmfField(coeffs, mask, np.eye(3))
    vals, _ = field.value(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), np.array([[1.0, 0, 0], [1.0, 0, 0]]))
    assert vals[0] == 0 and np.isclose(vals[1], coeffs[1, 1, 1] @ sh.basis(4, np.eye(3))[0])


def test_act_is_dipys():
    from dipy.tracking.stopping_criterion import ActStoppingCriterion, StreamlineStatus
    rng = np.random.default_rng(4)
    inc, exc = rng.uniform(size=(2, 9, 8, 7))
    sc = ActStoppingCriterion(inc, exc)
    names = {StreamlineStatus.TRACKPOINT: F.TRACKPOINT, StreamlineStatus.ENDPOINT: F.ENDPOINT,
             StreamlineStatus.INVALIDPOINT: F.INVALIDPOINT, StreamlineStatus.OUTSIDEIMAGE: F.OUTSIDEIMAGE}
    p = points(inc.shape)
    ours = F.Act(inc, exc)(p).numpy()
    ref = np.array([names[sc.check_point(np.ascontiguousarray(x))] for x in p])
    assert np.array_equal(ours, ref)
