"""tractline.skullstrip: the SynthStrip network and conform (an independent code base), and the mask slot that runs it."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tractline import config as C
from tractline import fetch
from tractline import skullstrip as K

#: the released weight file's tensors (synthstrip.1.pt's model_state_dict), name -> shape
SHAPES = {
    **{f"encoder.{i}.{j}.conv.weight": (f, c if j == 0 else f, 3, 3, 3)
       for i, (c, f) in enumerate(zip((1, 16, 32, 64, 64, 64), K.ENC)) for j in (0, 1)},
    **{f"decoder.{i}.{j}.conv.weight": (f, c if j == 0 else f, 3, 3, 3)
       for i, (c, f) in enumerate(zip((64, 128, 128, 128, 128, 96), K.DEC)) for j in (0, 1)},
    "remaining.0.conv.weight": (16, 48, 3, 3, 3), "remaining.1.conv.weight": (16, 16, 3, 3, 3),
    "remaining.2.conv.weight": (1, 16, 3, 3, 3)}


def _weights_or_skip():
    folder, name = fetch.WEIGHT_FILES["synthstrip"]
    for p in (fetch.cache_dir() / folder / name,):
        if p.is_file():
            return p
    from tractline.data import DATA
    p = DATA / folder / name
    if p.is_file():
        return p
    pytest.skip("SynthStrip's weights not downloaded (tractline fetch, or a run with --mask synthstrip)")


def _unit_net(seed):
    """The network with random weights that keep activations near unit scale (He initialization for leaky ReLU 0.2,
    zero biases), so a slab or halo mistake shows far above rounding (scaled-down weights hid some: the output fell to
    1e-3, under the tolerance)."""
    torch.manual_seed(seed)
    net = K.Net().eval()
    for m in net.modules():
        if isinstance(m, torch.nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            torch.nn.init.zeros_(m.bias)
    return net


def _close(a, b, rel=1e-3):
    """Equal up to rounding: the largest difference under rel x the reference's spread. Measured: correct code stays
    within 3e-5 of it in float32; a halo one slice short anywhere, or an upsampling one slice off, lands at 1.6-5.4."""
    return float((a - b).abs().max()) <= rel * float(b.std())


def test_architecture_matches_the_weight_file_shapes():
    net = K.Net()
    got = {k: tuple(v.shape) for k, v in net.state_dict().items() if k.endswith("weight")}
    assert got == SHAPES
    assert all(net.state_dict()[k.replace("weight", "bias")].shape == (s[0],) for k, s in SHAPES.items())


def test_weights_load_strictly():
    net = K.load(_weights_or_skip())
    assert sum(p.numel() for p in net.parameters()) > 2_000_000


@pytest.mark.parametrize("cin,cout,Z", [(1, 16, 5), (48, 16, 9), (32, 32, 1)])
def test_2d_decomposition_is_the_3d_convolution(cin, cout, Z):
    torch.manual_seed(0)
    conv = K._Conv(cin, cout, act=False)
    x = torch.randn(1, cin, 11, 7, Z)
    ref = F.conv3d(x, conv.conv.weight, conv.conv.bias, padding=1)
    assert torch.allclose(conv._as_2d(x), ref, atol=1e-5, rtol=1e-5)
    if Z > 2:                                                             # a strided view, as the slabs pass
        v = x[..., 1:]
        assert torch.allclose(conv._as_2d(v), F.conv3d(v, conv.conv.weight, conv.conv.bias, padding=1),
                              atol=1e-5, rtol=1e-5)


def test_slabs_do_not_change_the_result(monkeypatch):
    """The network with tiny slabs (every stage cut into many pieces, halos at every cut) against one slab."""
    net = _unit_net(1)
    x = torch.rand(1, 1, 64, 64, 128)
    with torch.inference_mode():
        monkeypatch.setattr(K, "SLAB", {"cpu": 2 ** 40})
        whole = net(x)
        monkeypatch.setattr(K, "SLAB", {"cpu": 2 ** 12})
        cut = net(x)
    assert whole.shape == (1, 1, 64, 64, 128) and float(whole.std()) > 0.1
    assert _close(cut, whole)


def _plain(net, x):
    """The U-Net run whole, by the paper's description: no slabs, torch's conv3d, F.interpolate."""
    def conv(c, h):
        y = F.conv3d(h, c.conv.weight, c.conv.bias, padding=1)
        return F.leaky_relu(y, 0.2) if c.act else y
    skips, h = [], x
    for level in net.encoder:
        h = conv(level[1], conv(level[0], h))
        skips.append(h)
        h = F.max_pool3d(h, 2)
    for i, level in enumerate(net.decoder):
        h = torch.cat([F.interpolate(conv(level[1], conv(level[0], h)), scale_factor=2, mode="nearest"), skips[-1 - i]], 1)
    for c in net.remaining:
        h = conv(c, h)
    return h


@pytest.mark.parametrize("shape,slab", [((64, 64, 128), 2 ** 40), ((64, 64, 192), 2 ** 13), ((128, 64, 64), 99081),
                                        ((64, 128, 64), 1)])
def test_slabbed_network_is_the_plain_one(monkeypatch, shape, slab):
    """Against the network run whole: slabs that do and do not divide the volume, one-slice slabs, batch 2."""
    net = _unit_net(3)
    x = torch.rand(2, 1, *shape)
    monkeypatch.setattr(K, "SLAB", {"cpu": slab})
    with torch.inference_mode():
        ref = _plain(net, x)
        assert float(ref.std()) > 0.1 and _close(net(x), ref)


def test_nan_voxels_do_not_poison_the_input():
    v = np.random.default_rng(0).random((40, 40, 30)).astype(np.float32) * 1000
    v[v > 999] = np.nan                                                   # about 0.1 %
    x, _ = K.conform(v, _affine(np.eye(3), (2, 2, 2)))
    assert torch.isfinite(x).all() and float(x.max()) == 1.0


def _affine(R, zooms, origin=(0.0, 0.0, 0.0)):
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = np.asarray(R, float) * np.asarray(zooms, float), origin
    return A


def test_lia_axes_for_plain_and_oblique_inputs():
    D, perm = K._lia_axes(_affine(np.eye(3), (2, 2, 2)))                  # RAS
    assert np.allclose(D, K.LIA) and list(perm) == [0, 2, 1]
    D, _ = K._lia_axes(_affine(np.diag([-1, 1, 1]), (2, 2, 2)))           # LAS
    assert np.allclose(D, K.LIA)
    t = np.deg2rad(24)                                                     # tilted about x, as ON-Harmony's series are
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    D, _ = K._lia_axes(_affine(Rx, (2, 2, 2)))
    assert np.allclose(D.T @ D, np.eye(3)) and np.all(np.diag(K.LIA.T @ D) > 0.9)   # oblique kept, nearest LIA


def test_conform_grid_and_intensities():
    rng = np.random.default_rng(0)
    v = np.zeros((40, 40, 30), np.float32)
    v[5:35, 5:35, 3:27] = rng.random((30, 30, 24)) * 1000
    A = _affine(np.eye(3), (2.0, 2.0, 2.4), (-40, -40, -36))
    x, Cm = K.conform(v, A)
    shape = np.array(x.shape[2:])
    assert np.all(shape % 64 == 0)
    box = np.array([30 * 2.0, 24 * 2.4, 30 * 2.0])                        # the nonzero box along L, I, A, in mm
    assert np.all(shape >= box + 2 * K.MARGIN)
    assert np.allclose(np.linalg.norm(Cm[:3, :3], axis=0), 1.0)           # 1 mm voxels
    assert float(x.min()) == 0.0 and float(x.max()) == 1.0
    assert abs(float((x == 1).float().mean()) - 0.01) < 0.002            # the top percent clipped to 1


def test_constant_field_is_refused(monkeypatch, tmp_path):
    class Flat(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)
    monkeypatch.setattr(K, "load", lambda path, device="cpu": Flat())
    v = np.random.default_rng(0).random((20, 20, 20)).astype(np.float32)
    with pytest.raises(RuntimeError, match="constant field"):
        K.signed_distance(v, np.eye(4) * 2 + np.diag([0, 0, 0, -1]), tmp_path / "w.pt")


def test_signed_distance_on_a_sphere():
    """With the released weights: a bright ball in a dark head-sized volume is found, inside negative."""
    w = _weights_or_skip()
    g = np.indices((60, 60, 50)).transpose(1, 2, 3, 0) - np.array([30, 30, 25])
    r = np.linalg.norm(g * [2, 2, 2.4], axis=-1)
    v = np.where(r < 70, 600.0, 0.0) + np.where(r < 60, 400.0, 0.0)
    s = K.signed_distance(v.astype(np.float32), _affine(np.eye(3), (2, 2, 2.4)), w)
    assert s.shape == v.shape and np.isfinite(s).all()
    assert s[30, 30, 25] < 0 and s[0, 0, 0] > 0


def test_mask_slot_registration_and_config():
    impl = C.implementation("mask", "synthstrip")
    assert impl.status == "experimental" and impl.options["border"].default == 1.0
    from tractline import pipeline as P
    a, b = P.config_for(), P.config_for(mask="synthstrip")
    assert b["mask"].impl == "synthstrip" and a["mask"].impl == "median-otsu"
    assert a.to_dict()["sha256"] != b.to_dict()["sha256"]
    assert P.config_for(mask=False)["mask"].impl == "median-otsu"
    c = P.config_for(mask={"impl": "synthstrip", "border": 3.0})
    assert c.opts("mask")["border"] == 3.0 and c.to_dict()["sha256"] != b.to_dict()["sha256"]


def test_fetch_pins_and_files_agree():
    assert set(fetch.WEIGHT_FILES) == set(fetch.WEIGHTS)
    url, sha, size = fetch.WEIGHTS["synthstrip"]
    assert url.startswith("https://") and len(sha) == 64 and size == 30851709


def test_provenance_records_the_mask_weights():
    from tractline import pipeline as P
    from tractline import provenance as PV
    assert PV.mask_weights(P.config_for().to_dict()) == []
    _weights_or_skip()
    (rec,) = PV.mask_weights(P.config_for(mask="synthstrip").to_dict())
    assert rec["pinned"] and rec["sha256"] == fetch.WEIGHTS["synthstrip"][1]


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_devices_agree(device):
    """The network on a GPU (its slabs and convolution path) against the CPU's, random weights."""
    if device == "mps" and not torch.backends.mps.is_available() or device == "cuda" and not torch.cuda.is_available():
        pytest.skip(f"no {device}")
    net = _unit_net(2)
    x = torch.rand(1, 1, 64, 128, 64)
    with torch.inference_mode():
        ref = net(x)
        got = net.to(device)(x.to(device)).cpu()
    assert _close(got, ref)                                                # float32 on another device: summation order
