"""tractline.ptt: the propagator's geometry, and tracking on a synthetic band (one fiber direction, ACT ends at both
faces) on the CPU and, where present, MPS."""
import subprocess, sys
import numpy as np
import pytest
import torch
from tractline import fields as F, ptt, sh


def test_propagator_is_second_order():
    """Curvature k along N1 over an arc s: the exact tangent is cos(ks) T + sin(ks) N1; the step matches to O((ks)^3),
    plus the 1e-4 that DIPY's rule gives the other curvature when it is 0 (k_small s along N2)."""
    T = torch.tensor([[1.0, 0, 0]], dtype=torch.float64)
    N1 = torch.tensor([[0.0, 1, 0]], dtype=torch.float64)
    N2 = torch.tensor([[0.0, 0, 1]], dtype=torch.float64)
    for k1, s in ((0.2, 0.5), (0.05, 0.5), (0.4, 0.25)):
        pos, Tn, _ = ptt._propagate(torch.zeros((1, 3), dtype=torch.float64), T, N1, N2,
                                    torch.tensor([[k1, 0.0]], dtype=torch.float64), s, torch.ones(3, dtype=torch.float64))
        exact_T = np.array([np.cos(k1 * s), np.sin(k1 * s), 0])
        exact_p = np.array([np.sin(k1 * s), 1 - np.cos(k1 * s), 0]) / k1
        assert np.abs(Tn.numpy()[0] - exact_T).max() < (k1 * s) ** 3 + ptt.K_SMALL * s
        assert np.abs(pos.numpy()[0] - exact_p).max() < s * (k1 * s) ** 2 + ptt.K_SMALL * s * s
    _, Tn, N2n = ptt._propagate(torch.zeros((1, 3), dtype=torch.float64), T, N1, N2, torch.zeros((1, 2), dtype=torch.float64),
                                0.5, torch.ones(3, dtype=torch.float64))
    assert torch.equal(Tn, T) and torch.equal(N2n, N2)                       # both curvatures ~0: straight


def band(shape=(40, 12, 12), vox=2.0):
    """FODs of one fiber along x in a band (y, z within 3 voxels of the center), zero elsewhere; ACT: include at the x
    faces, exclude nowhere."""
    u = np.array([[1.0, 0, 0]])
    c = np.zeros(shape + (45,))
    m = np.zeros(shape, bool)
    cy, cz = shape[1] // 2, shape[2] // 2
    m[:, cy - 3:cy + 3, cz - 3:cz + 3] = True
    c[m] = sh.basis(8, u)[0] * 2.0
    include = np.zeros(shape)
    include[0], include[-1] = 1.0, 1.0
    affine = np.diag([vox, vox, vox, 1.0])
    return c, m, include, np.zeros(shape), affine


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
@pytest.mark.parametrize("compat", [False, True])
def test_band(device, compat):
    c, m, inc, exc, affine = band()
    from tractline.csd import hemisphere
    v = hemisphere(181)
    verts = np.r_[v, -v]
    pmf = F.PmfField(c, m, verts, device)
    act = F.Act(inc, exc, device)
    tr = ptt.Tracker(pmf, act, affine, np.full(3, 2.0), ptt.Params(min_len=10), device, _dipy_compat=compat)
    rng = np.random.default_rng(0)
    seeds_vox = np.c_[rng.uniform(8, 30, 200), rng.uniform(5, 7, 200), rng.uniform(5, 7, 200)]
    seeds = seeds_vox * 2.0
    lines, info = tr.track(seeds, seed=3)
    assert info["kept"] > 0.9 * len(seeds)
    # measured (2026-10-24, MPS): real units: all 200 valid, spans 75-75.5 mm, mean angle 8.4 degrees; DIPY's units: 195,
    # spans from 57 mm, 11.5 degrees (its curvature bound is twice the real one at 2 mm voxels)
    cy = c.shape[1] // 2 * 2.0                                               # mm: the band's voxels span cy - 6 .. cy + 4
    span = np.array([x[:, 0].max() - x[:, 0].min() for x in lines])
    assert np.mean(span > 50) > 0.95                                          # most cover most of the band's 80 mm (a rare
    for x in lines:                                                           # one turns back: the FOD supports +x and -x)
        assert (x[:, 1:] > cy - 8.0).all() and (x[:, 1:] < cy + 6.0).all()   # inside the band and its one-voxel fade
    seg = np.concatenate([np.diff(x, axis=0) for x in lines])
    cos = np.abs(seg[:, 0]) / np.linalg.norm(seg, axis=1)
    assert np.degrees(np.arccos(np.clip(cos, -1, 1))).mean() < 15            # along x
    again, _ = tr.track(seeds, seed=3)
    assert len(again) == len(lines) and all(np.array_equal(a, b) for a, b in zip(again, lines))


DIPY_PTT = """
import sys, warnings
import numpy as np
warnings.simplefilter("ignore")
from dipy.data import default_sphere
from dipy.tracking.stopping_criterion import ActStoppingCriterion
from dipy.tracking.tracker import ptt_tracking
z = np.load(sys.argv[1])
sc = ActStoppingCriterion(np.ascontiguousarray(z["inc"]), np.ascontiguousarray(z["exc"]))
lines = list(ptt_tracking(z["seeds"], sc, z["affine"], sh=z["c"], sphere=default_sphere, random_seed=1, return_all=False,
                          nbr_threads=1, step_size=0.5, max_angle=10, probe_length=1.5, probe_quality=7, probe_count=1,
                          pmf_threshold=0.1, min_len=10, max_len=250))
np.savez(sys.argv[2], points=np.concatenate(lines), lengths=np.array([len(x) for x in lines]))
"""


def _turns(lines):
    out = []
    for x in lines:
        d = np.diff(x, axis=0)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        out.append(np.degrees(np.arccos(np.clip((d[1:] * d[:-1]).sum(1), -1, 1))))
    return np.concatenate(out)


def test_dipy_units_match_dipy(tmp_path):
    """DIPY's own ptt_tracking against ours with DIPY's units (the validation switch) and in real units, on a band at 2 mm
    voxels (CPU). DIPY 1.12's three unit inconsistencies (module docstring of tractline.ptt) show as turns up to ~2x the
    stated 10 degrees and as two ~1 mm segments at the seed; the switch reproduces both, real units neither. Measured by the
    review (2026-10-24, 400 seeds): kept 382 / 382 / 398, turn p99 16.8 / 16.9 / 8.6 degrees."""
    pytest.importorskip("dipy.tracking.tracker")
    vox = 2.0
    shape = (40, 12, 12)
    c, m, inc, exc, affine = band(shape, vox)
    rng = np.random.default_rng(0)
    n = 400
    sv = np.c_[rng.uniform(0.2, 0.75, n) * shape[0], rng.uniform(-0.5, 0.5, n) + shape[1] // 2 - 0.5,
               rng.uniform(-0.5, 0.5, n) + shape[2] // 2 - 0.5]
    seeds = sv * vox
    # DIPY's tracking loads its own OpenMP runtime, which aborts beside torch's (macOS): it runs in a process of its own
    src, out = tmp_path / "band.npz", tmp_path / "dipy.npz"
    np.savez(src, c=c, inc=inc, exc=exc, affine=affine, seeds=seeds)
    subprocess.run([sys.executable, "-I", "-c", DIPY_PTT, str(src), str(out)], check=True)
    z = np.load(out)
    ref = np.split(z["points"], np.cumsum(z["lengths"])[:-1])
    from dipy.data import default_sphere
    pmf = F.PmfField(c, m, default_sphere.vertices)
    act = F.Act(inc, exc)
    ours = {u: ptt.Tracker(pmf, act, affine, np.full(3, vox), ptt.Params(min_len=10), _dipy_compat=u == "dipy")
            .track(seeds, seed=3)[0] for u in ("dipy", "mm")}
    p99 = {k: np.percentile(_turns(v), 99) for k, v in {"DIPY": ref, **ours}.items()}
    assert abs(len(ours["dipy"]) - len(ref)) <= 0.03 * len(ref)
    assert abs(p99["dipy"] - p99["DIPY"]) < 1.5 and p99["DIPY"] > 14 and p99["mm"] < 10.5
    long_seed = lambda x: np.sort(np.linalg.norm(np.diff(x, axis=0), axis=1))[-2] > 0.9    # noqa: E731
    assert np.mean([long_seed(x) for x in ref]) > 0.9 and np.mean([long_seed(x) for x in ours["dipy"]]) > 0.9
    assert max(np.linalg.norm(np.diff(x, axis=0), axis=1).max() for x in ours["mm"]) < 0.55
