"""The opt-in denoising (tractline.denoise, pipeline.denoise_subject; docs/robustness-plan.md, step 3) on the
synthetic phantom: dipy runs in a process of its own, MP-PCA's slabs give dipy's whole-volume result bit for bit,
the DWI's own b0s are denoised and the reversed ones are not, and the pipeline runs with each method and order."""
import importlib.util, sys
import numpy as np
import pytest
import torch                                                                     # loaded here, as in the pipeline: dipy must not be
from phantom import subject
from tractline import pipeline as P
from tractline import denoise as D
from test_pipeline_cpu import stand_in

if importlib.util.find_spec("dipy") is None:                                     # the dipy group; found without importing it
    pytest.skip("dipy not installed (uv run --group dipy)", allow_module_level=True)


@pytest.fixture(scope="module")
def s():
    return subject()


def test_subprocess_isolation(s):
    """dipy's denoising never loads in this process (it has torch); torch still works after."""
    before = {m for m in sys.modules if m.startswith("dipy.denoise")}
    d = D.denoise(s.dwi, s.bval, "mppca", workers=2)
    assert {m for m in sys.modules if m.startswith("dipy.denoise")} == before == set()
    assert d.dwi.dtype == np.float32 and d.dwi.shape == s.dwi.shape and d.sigma.shape == s.dwi.shape[:3]
    assert torch.ones(2).sum().item() == 2.0


def test_slabs_are_dipys_whole_volume(s):
    """MP-PCA in slabs (any count, along the longest axis) is dipy's mppca on the whole volume, bit for bit. dipy is
    called in a subprocess here too, so the test keeps the isolation."""
    import subprocess, tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as tmp:
        np.save(Path(tmp) / "dwi.npy", s.dwi)
        code = ("import sys, numpy as np; from dipy.denoise.localpca import mppca; a = np.load(sys.argv[1]); "
                "d, g = mppca(a, patch_radius=2, return_sigma=True, suppress_warning=True); "
                "np.save(sys.argv[2], d.astype(np.float32)); np.save(sys.argv[3], g.astype(np.float32))")
        subprocess.run([sys.executable, "-c", code, f"{tmp}/dwi.npy", f"{tmp}/d.npy", f"{tmp}/g.npy"], check=True)
        whole, sigma = np.load(f"{tmp}/d.npy"), np.load(f"{tmp}/g.npy")
    for workers in (1, 3, 7):
        d = D.denoise(s.dwi, s.bval, "mppca", workers=workers)
        assert d.info["slabs"] == workers and d.info["slab axis"] == 1
        assert np.array_equal(d.dwi, whole) and np.array_equal(d.sigma, sigma)


def test_mppca_finds_white_noise(s):
    """On the phantom's white noise (SD 8, independent between voxels) MP-PCA's sigma is the noise's, and it removes
    most of it."""
    d = D.denoise(s.dwi, s.bval, "mppca")
    st = D.summary(s.dwi, d, s.bval, mask=s.brain)
    assert 6.5 < st["the mask: sigma (quartiles)"][1] < 9.5
    assert 5.0 < st["the mask: removed RMS, all / b0s / DW volumes"][0] < 8.5
    assert abs(st["the mask: removed mean (a bias)"]) < 0.5


def test_denoise_subject_b0s(s):
    """The DWI's own b0s that feed the field are taken from the denoised series; the reversed b0s stay as acquired;
    the subject given is not changed."""
    dwi0, b0s0 = s.dwi.copy(), s.b0s.copy()
    for method in P.DENOISE:
        out, st = P.denoise_subject(s, P.Timer(), method)
        assert np.array_equal(s.dwi, dwi0) and np.array_equal(s.b0s, b0s0)
        assert st["method"] == method and st["order"] == "acquired"
        assert st["b0s denoised (the DWI's own), of those feeding the field"] == [2, 4]
        assert np.array_equal(out.b0s[..., :2], out.dwi[..., :2].astype(np.float64))
        assert not np.array_equal(out.b0s[..., :2], b0s0[..., :2])
        assert np.array_equal(out.b0s[..., 2:], b0s0[..., 2:])


def test_settings_in_the_name(s):
    """A denoiser's name may carry its setting: MP-PCA's patch radius, Patch2Self's sketch seed; others are refused."""
    assert D.parse("mppca") == ("mppca", {"patch_radius": 2}) and D.parse("mppca:r3") == ("mppca", {"patch_radius": 3})
    assert D.parse("patch2self:s1") == ("patch2self", {"sketch_seed": 1})
    for bad in ("nlmeans", "mppca:s1", "patch2self:r3", "mppca:r"):
        with pytest.raises(ValueError):
            D.parse(bad)
    d = D.denoise(s.dwi, s.bval, "mppca:r3", workers=2)
    assert d.info["patch_radius"] == 3 and not np.array_equal(d.dwi, D.denoise(s.dwi, s.bval, "mppca", workers=2).dwi)


def test_unmatched_forward_b0_warns(s):
    """A forward b0 that is not one of the DWI's own volumes stays as acquired, with a warning."""
    import copy
    t = copy.copy(s)
    t.b0s = s.b0s.copy()
    t.b0s[..., 1] += 1.0
    with pytest.warns(UserWarning, match="1 of the 2 b0s"):
        out, st = P.denoise_subject(t, P.Timer(), "patch2self")
    assert st["b0s denoised (the DWI's own), of those feeding the field"] == [1, 4] and st["b0s phase-encoded as the DWI"] == 2
    assert np.array_equal(out.b0s[..., 1], t.b0s[..., 1])


def test_patch2self_repeats(s):
    """Patch2Self's sketch is seeded: two runs agree bit for bit."""
    a, b = (D.denoise(s.dwi, s.bval, "patch2self") for _ in range(2))
    assert np.array_equal(a.dwi, b.dwi) and a.sigma is None and (a.dwi >= 0).all()


@pytest.mark.parametrize("method,order", [("mppca", "acquired"), ("patch2self", "acquired"), ("mppca", "corrected")])
def test_pipeline_runs_denoised(s, method, order):
    corr, tg, labels = P.run(s, stand_in, P.Timer(), device="cpu", denoise=method, denoise_order=order)
    assert corr.denoise["method"] == method and corr.denoise["order"] == order
    assert len(tg.fibers) > 700 and labels.keep.sum() > 0.9 * len(tg.fibers)


def test_default_has_no_denoising(s):
    """Without denoise= nothing is denoised: the fibers are run()'s as before, and Correction.denoise is None."""
    timer = P.Timer()
    corr, tg, _ = P.run(s, stand_in, timer, device="cpu")
    assert corr.denoise is None and "denoise" not in timer.seconds
    c = P.correct(s, P.Timer(), "cpu")
    t2 = P.track(s, c.dwi, P.Timer(), device="cpu")
    assert np.array_equal(corr.dwi, c.dwi) and all(np.array_equal(a, b) for a, b in zip(tg.fibers, t2.fibers))
    with pytest.raises(ValueError):
        P.run(s, stand_in, P.Timer(), device="cpu", denoise="nlmeans")
    with pytest.raises(ValueError):
        P.run(s, stand_in, P.Timer(), device="cpu", denoise="mppca", denoise_order="before")
