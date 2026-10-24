"""tractline.sh against dipy.reconst.shm: the same real harmonics, coefficient for coefficient (dipy only for the check)."""
import warnings
import numpy as np
import pytest
from tractline import sh

dipy_shm = pytest.importorskip("dipy.reconst.shm")


def directions(n=500, seed=0):
    d = np.random.default_rng(seed).normal(size=(n, 3))
    d = np.r_[d, [[0, 0, 1], [0, 0, -1], [1, 0, 0], [0, 1, 0]]]               # the poles and the equator
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def test_indices_are_dipys():
    for order in (0, 2, 4, 6, 8, 10):
        m, l = sh.indices(order)
        dm, dl = dipy_shm.sph_harm_ind_list(order)
        assert np.array_equal(m, dm) and np.array_equal(l, dl)
        assert sh.n_coeffs(order) == len(m) and sh.order_of(len(m)) == order
    with pytest.raises(ValueError):
        sh.indices(3)


@pytest.mark.parametrize("legacy", [True, False])
def test_basis_is_dipys(legacy):
    from dipy.core.geometry import cart2sphere
    d = directions()
    _, theta, phi = cart2sphere(*d.T)
    for order in (2, 6, 8, 12):
        m, l = dipy_shm.sph_harm_ind_list(order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ref = dipy_shm.real_sh_descoteaux_from_index(m, l, theta[:, None], phi[:, None], legacy=legacy)
        assert np.abs(sh.basis(order, d, legacy=legacy) - ref).max() < 1e-11
