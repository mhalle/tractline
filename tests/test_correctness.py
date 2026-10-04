"""Tests that check answers, not only that things run: a pair distorted by a known field (the correction
recovers it, the residual and the displacement agree), prepare's gradients in RAS for three storage
orientations, the tracked shell, a TRX round trip, RapidParc's rows returned to their own fibers (without
the weights), BIDS partners stored reversed or across protocols, and CPU-testable parts of the GPU paths.
Written by the 2026-10-03 test review: 23 of the 25 deliberate breaks the suite missed fail here."""
import json
from types import SimpleNamespace
import numpy as np, torch
import pytest
from phantom import subject
from tractline import pipeline as P, susceptibility as S, prep, trx
from tractline.labelers.base import Labels, LUT


# ---------------------------------------------------------------- a distorted pair, its field known
def distorted(ro_ap=0.05, ro_pa=0.025, amp=40.0):
    s = subject()
    X, Y, Z = s.brain.shape
    x, y, z = np.meshgrid(np.arange(X), np.arange(Y), np.arange(Z), indexing="ij")
    h = amp * np.exp(-(((x - X / 2) / 6) ** 2 + ((y - Y / 2) / 8) ** 2 + ((z - Z / 2) / 5) ** 2))    # Hz
    true = np.repeat(s.b0s[..., :1], 1, -1)
    # AP is phase-encoded -y (pe_sign -1): corrected = apply(obs, h, 1, -1, ro); so obs ~ apply(true, h, 1, +1, ro)
    ap = S.apply(true, h, 1, +1.0, ro_ap).astype(np.float64)
    pa = S.apply(true, h, 1, -1.0, ro_pa).astype(np.float64)
    s.b0s = np.concatenate([ap, ap, pa, pa], -1)
    s.b0_readout_s = np.array([ro_ap, ro_ap, ro_pa, ro_pa])
    s.readout_s = ro_ap
    s.dwi = np.concatenate([S.apply(s.dwi, h, 1, +1.0, ro_ap)], -1)
    return s, h


@pytest.fixture(scope="module")
def dcorr():
    s, h = distorted()
    return s, h, P.correct(s, P.Timer(), device="cpu")


def test_distorted_field_recovered(dcorr):
    s, h, c = dcorr
    m = s.brain
    err = np.abs(c.field_hz - h)[m]
    assert np.corrcoef(c.field_hz[m], h[m])[0, 1] > 0.95
    assert np.quantile(err, 0.9) < 0.05 * h.max()
    assert c.field_hz[m][np.argmax(h[m])] > 0.9 * h.max()                     # the sign and the size


def test_distorted_residual_small_no_warning(dcorr):
    s, h, c = dcorr
    assert c.residual_left < 0.1 and not c.warnings, (c.residual_left, c.warnings)
    # the uncorrected field would leave everything: the residual measures something
    assert P.residual_left(s, np.zeros_like(h)) > 0.9


def test_displacement_sign(dcorr):
    s, h, c = dcorr
    d = c.displacement_mm
    k = np.unravel_index(np.argmax(h), h.shape)
    assert np.sign(d[k]) == np.sign(s.pe_sign) and abs(d[k]) == pytest.approx(s.readout_s * h[k] * s.vox[1], rel=0.2)


def test_apply_conserves_signal():
    """The Jacobian: a displaced block's total signal is kept (it is not without it)."""
    s, h = distorted()
    vol = s.b0s[..., :1]
    out = S.apply(vol, h, 1, -1.0, 0.05)
    assert out.sum() == pytest.approx(vol.sum(), rel=2e-3)
    from tractline import susceptibility as SS
    hh = torch.as_tensor(h)
    nojac = SS.unwarp_pe_cubic(SS.prefilter(torch.as_tensor(np.moveaxis(vol, -1, 0)), 1), hh, torch.gradient(hh, dim=1)[0], 1,
                               torch.full((1,), -0.05, dtype=torch.float64), jac=False).numpy()
    assert abs(nojac.sum() - vol.sum()) > abs(out.sum() - vol.sum())


# ---------------------------------------------------------------- prepare's gradients
@pytest.mark.parametrize("affine, expect", [
    (np.diag([-2.0, 2, 2, 1]), [-1, 0, 0]),       # radiological storage: no flip, x image = -x RAS
    (np.diag([2.0, 2, 2, 1]), [-1, 0, 0]),        # neurological storage (det > 0): FSL's x flipped
    (np.array([[0, -2.0, 0, 0], [2, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 1]]), [0, -1, 0]),   # rotated 90 deg about z, det > 0
])
def test_prepare_gradients_in_ras(affine, expect):
    s = subject()
    bvec = np.zeros_like(s.bvec); bvec[0, 2:] = 1.0                      # every direction image x
    t = prep.prepare(s.dwi, affine, s.bval, bvec)
    g = np.array([float(v) for v in t.header["DWMRI_gradient_0002"].split()])
    assert np.allclose(g, expect, atol=1e-6), g


def test_shell_reaches_the_tracker():
    s = subject()
    s.bval = np.where(s.bval > 50, 1000.0, 0.0)
    s.b0s = None
    from test_pipeline_cpu import stand_in
    _, tg, _ = P.run(s, stand_in, P.Timer(), device="cpu", workers=1, shell=1000.0)
    assert tg.info["shell"] == 1000.0


# ---------------------------------------------------------------- TRX written, read back
def test_trx_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    fibers = [rng.normal(0, 30, (n, 3)).astype(np.float32) for n in (5, 30, 12, 40, 3)]
    keep = np.array([False, True, False, True, True])
    clusters = np.array([7, 1500, 300])
    logp = torch.full((3, 1600), -30.0); logp[np.arange(3), clusters] = 0.0
    labels = Labels(keep=keep, length_mm=np.zeros(3), tract=LUT[clusters], logp=logp.half())
    s = SimpleNamespace(affine=np.diag([-2.0, 2, 2, 1]), dwi=np.zeros((4, 5, 6, 1)))
    tg = SimpleNamespace(fibers=fibers, stats={"seed_index": [10, 11, 12, 13, 14]})
    d = trx.write(tmp_path / "t", s, tg, labels)
    off = np.fromfile(d / "offsets.uint64", np.uint64)
    pos = np.fromfile(d / "positions.3.float32", np.float32).reshape(-1, 3)
    assert off.tolist() == [0, 5, 35, 47, 87, 90]
    for i, f in enumerate(fibers):
        assert np.array_equal(pos[off[i]:off[i + 1]], f)
    tract = np.fromfile(d / "dps" / "tract.uint8", np.uint8)
    assert tract.tolist() == [255, LUT[7], 255, LUT[1500], LUT[300]]
    assert np.fromfile(d / "dps" / "cluster.uint16", np.uint16).tolist() == [65535, 7, 65535, 1500, 300]
    assert np.fromfile(d / "dps" / "seed.uint32", np.uint32).tolist() == [10, 11, 12, 13, 14]
    h = json.loads((d / "header.json").read_text())
    assert h["NB_STREAMLINES"] == 5 and h["NB_VERTICES"] == 90 and h["DIMENSIONS"] == [4, 5, 6]
    z = trx.write(tmp_path / "t.trx", s, tg, labels)
    import zipfile
    assert zipfile.ZipFile(z).read("offsets.uint64") == (d / "offsets.uint64").read_bytes()


# ---------------------------------------------------------------- RapidParc's plumbing, without weights
def test_rapidparc_rows_return_to_their_fibers():
    """A fake network whose answer is each streamline's own first coordinate: whatever the shuffle,
    groups and padding, fiber i must get its own answer."""
    from tractline.labelers import rapidparc as R
    lab = object.__new__(R.Labeler)
    lab.device, lab.batch, lab.lut = torch.device("cpu"), 2, LUT

    class Fake(torch.nn.Module):
        def forward(self, x):                                           # (bs, 2000, 15, 3) -> (bs*2000, 1600)
            key = x[..., 0, 1].reshape(-1)                              # the first point's y, normalized
            cl = ((key + 1) / 2 * 1599).round().long().clamp(0, 1599)
            return torch.nn.functional.one_hot(cl, 1600).float() * 10
    lab.model = Fake()
    n = 4321
    ys = np.linspace(-50, 50, n)
    fibers = [np.c_[np.linspace(0, 60, 20), np.full(20, y), np.linspace(0, 5, 20)].astype(np.float32) for y in ys]
    L = lab(fibers, draws=(3,), logp=True)
    want = np.round((ys - ys.min()) / (ys.max() - ys.min()) * 1599).astype(int)
    got = L.logp.float().argmax(1).numpy()
    assert (np.abs(got - want) <= 1).all()


# ---------------------------------------------------------------- BIDS
nib = pytest.importorskip("nibabel")


def test_bids_flipped_partner_resampled_and_aligned(tmp_path):
    """A partner stored with its j axis reversed (so its sidecar says j- too, physically opposite): found,
    its polarity right on the diffusion grid, its voxels where the diffusion series' are."""
    import test_bids as T
    s = subject()
    ses = tmp_path / "sub-01"
    dwi = ses / "dwi" / "sub-01_acq-AP_dwi.nii.gz"
    T.write(dwi, s.dwi, s.affine, T.SIDE, s.bval, s.bvec)
    A = s.affine.copy(); A[:3, 1] *= -1; A[:3, 3] = s.affine[:3, 3] + s.affine[:3, 1] * (s.dwi.shape[1] - 1)
    pa = s.b0s[..., 2:][:, ::-1]
    T.write(ses / "dwi" / "sub-01_acq-PA_dwi.nii.gz", pa, A, {**T.SIDE, "PhaseEncodingDirection": "j-"},
            bval=[0.0, 0.0], bvec=np.zeros((3, 2)))
    from tractline import bids
    # j- on both, but the partner's j is the other way in the world: same folder needs "the other way", so say it
    b = bids.load(dwi, partners=[ses / "dwi" / "sub-01_acq-PA_dwi.nii.gz"])
    assert b.pairing.ok, b.pairing.message
    assert [v[1] for v in b.pe_vectors] == [-1, -1, 1, 1]
    assert np.allclose(b.b0s[..., 2:][s.brain], s.b0s[..., 2:][s.brain], rtol=1e-3, atol=1e-2)


def test_bids_readout_not_assumed_across_protocols(tmp_path):
    import test_bids as T
    s = subject()
    ses = tmp_path / "sub-01"
    dwi = ses / "dwi" / "sub-01_acq-AP_dwi.nii.gz"
    T.write(dwi, s.dwi, s.affine, T.SIDE, s.bval, s.bvec)
    side = {k: v for k, v in T.SIDE.items() if k not in ("TotalReadoutTime", "EchoTime")}
    T.write(ses / "fmap" / "sub-01_dir-PA_epi.nii.gz", s.b0s[..., 2:], s.affine,
            {**side, "PhaseEncodingDirection": "j", "IntendedFor": "sub-01/dwi/sub-01_acq-AP_dwi.nii.gz"})
    from tractline import bids
    b = bids.load(dwi)
    assert not b.pairing.ok and "TotalReadoutTime" in b.pairing.message


@pytest.mark.parametrize("te", [0.085, 0.0815])
def test_bids_echo_time_tolerance(tmp_path, te):
    import test_bids as T
    s = subject()
    ses = tmp_path / "sub-01"
    dwi = ses / "dwi" / "sub-01_acq-AP_dwi.nii.gz"
    T.write(dwi, s.dwi, s.affine, T.SIDE, s.bval, s.bvec)
    T.write(ses / "dwi" / "sub-01_acq-PA_dwi.nii.gz", s.b0s[..., 2:], s.affine, {**T.SIDE, "PhaseEncodingDirection": "j", "EchoTime": te},
            bval=[0.0, 0.0], bvec=np.zeros((3, 2)))
    from tractline import bids
    assert not bids.load(dwi).pairing.ok                                 # 5 ms, 1.5 ms: past the 1 ms rule


# ---------------------------------------------------------------- the GPU paths' CPU-testable parts
def test_exact_float32_restores():
    before = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    with P.exact_float32("cuda"):
        assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32
    assert (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32) == before


def test_timer_records_skips_and_totals(capsys):
    t = P.Timer(echo="x")
    with t("a"):
        pass
    t.skip("b")
    assert set(t.seconds) == {"a", "b"} and t.seconds["b"] == 0.0 and t.total("a", "b") >= 0.0
    out = capsys.readouterr().out
    assert "x a" in out and "x b skipped" in out
    with pytest.raises(KeyError):                                            # a misspelled stage is not silently 0
        t.total("a", "missing")
