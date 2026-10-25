"""tractline.nearest.CubeMap: the same vertex as a full scan (to rounding at exact ties), on the directions where a cell lookup could go wrong: cube
edges and corners, cell edges and a hair either side, the vertices themselves, the midpoints between neighboring vertices
(near-ties), antipodes (exact ties), and random directions."""
import numpy as np
import pytest
from tractline.csd import hemisphere
from tractline.nearest import CubeMap, cell_of


def hard_directions(V, cells, rng):
    d = [rng.normal(size=(20000, 3))]
    e = np.linspace(-1, 1, cells + 1)
    for axis in range(3):
        for s in (1.0, -1.0):
            for eps in (0.0, 1e-7, -1e-7):
                u, w = np.meshgrid(e + eps, rng.uniform(-1, 1, 40))
                x = np.zeros(u.shape + (3,))
                x[..., axis] = s
                x[..., (axis + 1) % 3] = u
                x[..., (axis + 2) % 3] = w
                d.append(x.reshape(-1, 3))
                x2 = x.copy()
                x2[..., (axis + 1) % 3], x2[..., (axis + 2) % 3] = w, u
                d.append(x2.reshape(-1, 3))
    corners = np.array([[a, b, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)], float)
    d += [corners, np.eye(3), -np.eye(3), V, -V]
    dots = np.abs(V @ V.T)
    np.fill_diagonal(dots, 0)
    nb = np.argsort(-dots, 1)[:, :6]                                         # each vertex's nearest axes
    mids = V[:, None] + np.sign((V[:, None] * V[nb]).sum(-1, keepdims=True)) * V[nb]
    d.append(mids.reshape(-1, 3))
    d = np.concatenate(d)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


@pytest.mark.parametrize("which", ["ours", "dipy"])
@pytest.mark.parametrize("cells", [8, 32])
def test_cube_map_is_exact(which, cells):
    if which == "dipy":
        V = pytest.importorskip("dipy.data").default_sphere.vertices
    else:
        v = hemisphere(181)
        V = np.r_[v, -v]
    rng = np.random.default_rng(0)
    cm = CubeMap(V, cells)
    d = hard_directions(V, cells, rng)
    got = cm.nearest(d)
    dots = np.abs(d @ cm.vertices.T)
    best = dots.max(1)
    gv = dots[np.arange(len(d)), got]
    assert (best - gv).max() <= 1e-12                                        # always a maximum, to rounding (ties: the
    top = np.argmax(dots, 1)                                                 # dot products' order differs by 1 ulp)
    same_axis = np.abs(cm.vertices[top] @ cm.vertices.T) > 1 - 1e-12         # the top vertex and its antipode: one axis
    second = np.where(same_axis, -1.0, dots).max(1)                          # the best other axis
    clear = best - second > 1e-9                                             # where the maximum is clear: the same axis,
    assert clear.sum() > 0.5 * len(d)                                        # and the first of its two vertices
    assert np.array_equal(got[clear], top[clear])


def test_cells_cover_every_direction():
    rng = np.random.default_rng(2)
    d = rng.normal(size=(5000, 3))
    face, i, j = cell_of(d, 16)
    assert face.min() >= 0 and face.max() <= 5 and i.min() >= 0 and i.max() <= 15 and j.min() >= 0 and j.max() <= 15
