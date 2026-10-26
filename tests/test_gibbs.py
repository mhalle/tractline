"""tractline.gibbs against DIPY's gibbs_removal (dipy only for the check): the same weights and total variation, the same
corrected volume (float64 on the CPU to rounding; float32 but for near-ties between shifts), every slice axis, 3D or 4D
input, square or not; and the slot's handling of the b0s that feed the field."""
import importlib.util, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pytest
import torch
from tractline import gibbs as G

#: DIPY is called in a process of its own, as everywhere in the tests (test_denoise.test_subprocess_isolation checks
#: that dipy.denoise never loads in a process that has torch).
DIPY_PY = """
import sys, warnings, numpy as np
warnings.simplefilter("ignore")
from dipy.denoise.gibbs import gibbs_removal, _image_tv, _weights
args = np.load(sys.argv[1], allow_pickle=True).item()
if args["what"] == "removal":
    out = gibbs_removal(args["vol"], slice_axis=args["axis"], inplace=False)
elif args["what"] == "weights":
    out = np.stack(_weights(args["shape"]))
else:
    out = np.minimum(*_image_tv(args["x"], axis=1, n_points=args["p"]))
np.save(sys.argv[2], out)
"""


def dipy_call(**args):
    if importlib.util.find_spec("dipy") is None:                    # the dipy group; found without importing it
        pytest.skip("dipy not installed (uv run --group dipy)")
    with tempfile.TemporaryDirectory() as d:
        np.save(Path(d) / "a.npy", np.array(args, dtype=object), allow_pickle=True)
        subprocess.run([sys.executable, "-I", "-c", DIPY_PY, str(Path(d) / "a.npy"), str(Path(d) / "o.npy")], check=True)
        return np.load(Path(d) / "o.npy")


def ringing(seed=0, n=48):
    """A disk's truncated k-space (rings at its edge) plus noise: (n, n, 3 slices, 2 volumes)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:n, :n]
    disk = ((x - n / 2 + 0.3) ** 2 + (y - n / 2 - 1.7) ** 2 < (n / 3.5) ** 2).astype(float) * 100
    K = np.fft.fftshift(np.fft.fft2(disk))
    q = n // 4
    K[:, :q] = K[:, -q:] = 0
    K[:q] = K[-q:] = 0
    img = np.abs(np.fft.ifft2(np.fft.ifftshift(K)))
    return np.stack([np.stack([img + rng.normal(0, 1, img.shape) for _ in range(3)], -1) for _ in range(2)], -1)


def dipy_gibbs(v, axis):
    return dipy_call(what="removal", vol=np.asarray(v), axis=axis)


def test_pieces_are_dipys():
    for shape in ((48, 48), (40, 56)):
        for a, b in zip(G.weights(shape), dipy_call(what="weights", shape=shape)):
            assert np.allclose(a, b)
    x = np.random.default_rng(1).normal(size=(5, 40))
    for p in (1, 3):
        assert np.allclose(G._tv(torch.as_tensor(x), p).numpy(), dipy_call(what="tv", x=x, p=p))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_cpu_float64_is_dipys(axis):
    vol = np.moveaxis(ringing(), 2, axis)
    a = dipy_gibbs(vol, axis)
    b = G.gibbs_removal(vol, axis, device="cpu")
    assert b.dtype == np.float64 and b.shape == vol.shape and np.abs(b - a).max() < 1e-9
    b3 = G.gibbs_removal(vol[..., 0], axis, device="cpu")                    # 3D input
    assert np.abs(b3 - dipy_gibbs(vol[..., 0], axis)).max() < 1e-9


def test_odd_non_square_is_dipys():
    """An odd, non-square slice (the transform matrices, the weights' grid and the total variation's padding at odd N)."""
    vol = ringing(seed=3, n=48)[:45, :37]
    assert np.abs(G.gibbs_removal(vol, 2, device="cpu") - dipy_gibbs(vol, 2)).max() < 1e-9


@pytest.mark.parametrize("ties", [False, True])
def test_selection_paths_agree(ties, monkeypatch):
    """The CPU's one-shift-at-a-time choice and the GPU's minimum per group of shifts: bit for bit, on the CPU and (when
    there is one) on MPS. ties: the total variation rounded to whole units, so many shifts tie exactly (within a group
    and across groups) and the first of equals must win in both."""
    if ties:
        tv = G._tv
        monkeypatch.setattr(G, "_tv", lambda x, p: tv(x, p).round())
    x = torch.as_tensor(ringing(seed=5)[..., 0, 0][None])
    runs = [("cpu", torch.float64), ("cpu", torch.float32)]
    if torch.backends.mps.is_available():
        runs.append(("mps", torch.float32))
    for dev, dt in runs:
        a = G._removal_1d(x.to(dev, dt), 3, per_shift=True)
        b = G._removal_1d(x.to(dev, dt), 3, per_shift=False)
        assert torch.equal(a, b), (dev, dt)


def test_removes_ringing():
    vol = ringing()
    out = G.gibbs_removal(vol, 2)
    assert np.abs(np.diff(out, axis=0)).mean() < 0.85 * np.abs(np.diff(vol, axis=0)).mean()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
def test_mps_float32_close():
    """Float32: a few near-ties between shifts fall the other way (measured 2026-10-26: 0.1 % of voxels beyond 1e-3;
    on cIIIs04's full scan 9 voxels in a million beyond 0.1 % of its scale, against DIPY float64; NOTES)."""
    vol = ringing()
    a = dipy_gibbs(vol, 2)
    b = G.gibbs_removal(vol.astype(np.float32), 2, device="mps")
    d = np.abs(b - a)
    assert b.dtype == np.float32 and (d > 1e-3 * a.max()).mean() < 0.005 and d.max() < 0.05 * a.max()


def test_gibbs_slot_in_execute():
    """The slot runs in execute and records itself (time, config, its b0 counts); a config with it hashes otherwise,
    and the default config's hash is unchanged; denoising after correction is refused with it."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from phantom import subject
    from tractline import pipeline as P
    from tractline.config import PipelineConfig
    assert PipelineConfig.preset("default").hash() == "28d01a1d3947a3f2f2de450b9b472c61e228129a156fad14be37b252f5574678"

    def stand_in(fibers, draws=(0,), logp=False):
        from tractline.labelers.base import Labels, MIN_LENGTH_MM, lengths
        _, _, length = lengths(fibers)
        keep = length >= MIN_LENGTH_MM
        return Labels(keep=keep, length_mm=length[keep], tract=np.zeros(int(keep.sum()), np.int64), logp=None)
    s = subject()
    s.b0_native = np.ones(s.b0s.shape[-1], bool)                           # the phantom's b0s are all on its grid
    cfg = P.config_for(stand_in).with_(gibbs="subvoxel-shift")
    timer = P.Timer()
    r = P.execute(s, cfg, timer, device="cpu", workers=1, labeler=stand_in)
    assert "gibbs" in timer.seconds and r.config["slots"]["gibbs"]["impl"] == "subvoxel-shift"
    g = r.correction.gibbs
    assert g["b0s from the DWI"] >= 1 and g["b0s left (resampled or not known)"] == 0
    assert g["b0s from the DWI"] + g["b0s corrected on their own"] == s.b0s.shape[-1]
    assert r.config["sha256"] is None and len(r.tractogram.fibers) > 0          # a caller's labeler: not hashable
    assert P.config_for().with_(gibbs="subvoxel-shift").hash() != P.config_for().hash()
    with pytest.raises(ValueError, match="denoising must come before"):
        P.config_for(denoise="mppca", denoise_order="corrected").with_(gibbs="subvoxel-shift")


def test_gibbs_subject_b0s():
    """The DWI's own b0s come from the corrected DWI, bit for bit; another series' b0s are corrected on their own only
    when on their own grid, and left otherwise or when the loader does not say."""
    from types import SimpleNamespace
    from tractline import pipeline as P
    vol = ringing(seed=7)                                                  # (48, 48, 3, 2)
    dwi = np.concatenate([vol, vol[..., :1] * 0.5], -1).astype(np.float32) # volumes 0, 1 b0s; 2 a DW volume
    other = ringing(seed=8)[..., 0]                                        # (48, 48, 3): a reversed series' b0
    b0s = np.stack([dwi[..., 0], dwi[..., 1], other, other], -1).astype(np.float64)
    opts = {"slice_axis": 2, "n_points": 3}
    for native, n_alone in ((np.array([True, True, True, False]), 1), (None, 0)):
        s = SimpleNamespace(dwi=dwi, bval=np.array([0.0, 5.0, 1000.0]), b0s=b0s, b0_native=native)
        out, counts = P.gibbs_subject(s, P.Timer(), opts, "cpu")
        assert np.array_equal(out.b0s[..., 0], out.dwi[..., 0]) and np.array_equal(out.b0s[..., 1], out.dwi[..., 1])
        assert counts == {"b0s from the DWI": 2, "b0s corrected on their own": n_alone,
                          "b0s left (resampled or not known)": 2 - n_alone}
        assert np.array_equal(out.b0s[..., 3], other)                      # resampled, or not known: left
        assert np.array_equal(out.b0s[..., 2], other) == (n_alone == 0)
        assert not np.array_equal(out.dwi, dwi) and np.array_equal(s.b0s, b0s)   # the input is not changed
    # a reversed b0 left with its ringing: a warning (pe_vectors: the DWI's b0s forward, the others reversed)
    s = SimpleNamespace(dwi=dwi, bval=np.array([0.0, 5.0, 1000.0]), b0s=b0s, b0_native=None, pe_axis=1, pe_sign=1.0,
                        pe_vectors=np.array([[0, 1, 0], [0, 1, 0], [0, -1, 0], [0, -1, 0]], float))
    with pytest.warns(UserWarning, match="2 b0s phase-encoded against the DWI keep their ringing"):
        P.gibbs_subject(s, P.Timer(), opts, "cpu")
    s.b0_native = np.array([True])                                          # flags of the wrong length: not known
    with pytest.warns(UserWarning, match="2 b0s"):
        P.gibbs_subject(s, P.Timer(), opts, "cpu")
