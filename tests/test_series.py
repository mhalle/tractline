"""tractline.series and the reversed_series slot: pairing two gradient tables, the average, the join's scale, bids.load
keeping a reversed series in full (the frame of its b-vectors, its grid, its refusals), the config and the run's checks."""
import json
import numpy as np
import pytest
from tractline import bids, series as SR, motion_eddy as ME
from tractline import pipeline as P
from tractline.config import PipelineConfig


def table(n=12, b=1000.0, nb0=2, seed=0):
    g = np.random.default_rng(seed).normal(size=(n, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    return np.r_[np.zeros(nb0), np.full(n, b)], np.c_[np.zeros((3, nb0)), g.T]


# ------------------------------------------------------------------------------------------------ pairing

def test_match_same_table():
    b, g = table()
    m = SR.match(b, g, b, g)
    assert m.pairs == [(i, i) for i in range(2, 14)] and m.b0_pairs == [(0, 0), (1, 1)]
    assert not m.unmatched_a and not m.unmatched_b and m.worst_angle_deg == 0


def test_match_is_by_direction_not_by_place():
    """A shuffled table, some vectors antipodal: found all the same, where pairing by place would average different
    directions."""
    b, g = table()
    order = np.r_[0, 1, np.random.default_rng(1).permutation(np.arange(2, 14))]
    g2 = g[:, order] * np.r_[1, 1, np.where(np.arange(12) % 2, -1.0, 1.0)]
    m = SR.match(b, g, b[order], g2)
    assert len(m.pairs) == 12
    assert all(order[j] == i for i, j in m.pairs)


def test_match_tolerances():
    b, g = table(n=2, nb0=1)
    v = g.copy()
    tilt = np.radians(4.0)                                                 # turn the first direction by 4 degrees
    u = np.cross(v[:, 1], [0.0, 0.0, 1.0])
    u /= np.linalg.norm(u)
    v[:, 1] = np.cos(tilt) * v[:, 1] + np.sin(tilt) * u
    assert len(SR.match(b, g, b, v, angle_tol=5.0).pairs) == 2
    tight = SR.match(b, g, b, v, angle_tol=3.0)
    assert len(tight.pairs) == 1 and tight.unmatched_a == [1] and tight.unmatched_b == [1]
    assert len(SR.match(b, g, b + 60 * (b > 50), g, b_tol=50.0).pairs) == 0
    assert len(SR.match(b, g, b + 40 * (b > 50), g, b_tol=50.0).pairs) == 2


def test_match_extra_b0s_stay():
    b, g = table(nb0=4)
    b2, g2 = table(nb0=1)
    m = SR.match(b, g, b2, g2)
    assert m.b0_pairs == [(0, 0)] and m.b0_alone_a == [1, 2, 3] and not m.b0_alone_b and len(m.pairs) == 12


def test_check_match_refuses_and_says_so():
    b, g = table(n=4)
    m = SR.match(b, g, b[:-1], g[:, :-1])
    with pytest.raises(ValueError, match="no partner within 50 s/mm"):
        SR.check_match(m, "refuse", 50.0, 5.0)
    SR.check_match(m, "keep", 50.0, 5.0)
    SR.check_match(m, "drop", 50.0, 5.0)


# ------------------------------------------------------------------------------------------------ the average and the join

def corrected(n=6, nb0=2):
    b, g = table(n=n, nb0=nb0)
    G = len(b)
    d = np.stack([np.full((2, 3, 4), float(i + 1)) for i in range(G)] + [np.full((2, 3, 4), 100.0 + i) for i in range(G)], -1)
    return d.astype(np.float32), np.r_[b, b], np.c_[g, -g], G


def test_average_means_each_pair_and_turns_the_vector():
    d, b, g, G = corrected()
    m = SR.match(b[:G], g[:, :G], b[G:], g[:, G:])
    out, bo, go, rec = SR.average(d, b, g, G, m)
    assert out.shape[-1] == G and (bo == b[:G]).all()
    assert np.allclose(out[0, 0, 0], [(i + 1 + 100 + i) / 2 for i in range(G)])
    assert np.allclose(np.abs((go[:, 2:] * g[:, 2:G]).sum(0)), 1) and np.allclose(np.linalg.norm(go[:, 2:], axis=0), 1)
    assert (go[:, :2] == 0).all()                                          # the b0s stay without a direction
    assert rec["pairs averaged (diffusion-weighted)"] == 6 and rec["b0 pairs averaged"] == 2


def test_average_a_left_out_volume():
    """A pair that lost a volume to the fit keeps the other, single, not an average with a volume never registered; a
    pair that lost both goes."""
    d, b, g, G = corrected()
    m = SR.match(b[:G], g[:, :G], b[G:], g[:, G:])
    out, bo, go, rec = SR.average(d, b, g, G, m, left_out=[3, G + 4, 5, G + 5])
    assert out.shape[-1] == G - 1
    assert out[0, 0, 0, 3] == 100.0 + 3 and out[0, 0, 0, 4] == 5.0
    assert rec["pairs that lost a volume to the fit (kept single)"] == 2 and rec["entries lost to the fit (gone)"] == 1


@pytest.mark.parametrize("how,n", [("keep", 8), ("drop", 7)])
def test_average_unmatched(how, n):
    d, b, g, G = corrected()
    keep = np.r_[np.arange(G), np.arange(G, 2 * G - 1)]                    # the reversed series lacks its last volume
    m = SR.match(b[:G], g[:, :G], b[G:2 * G - 1], g[:, G:2 * G - 1])
    assert m.unmatched_a == [G - 1]
    out, _, _, rec = SR.average(d[..., keep], b[keep], g[:, keep], G, m, how)
    assert out.shape[-1] == n and rec["unmatched diffusion-weighted volumes"]["what happened"] in ("kept, single", "left out")


def subject(k=1.0):
    b, g = table(n=4)
    rng = np.random.default_rng(0)
    d = rng.uniform(50, 100, (6, 7, 8, 6)).astype(np.float32)
    r = bids.ReversedSeries(d[..., ::-1] * k, b, g, -1.0, 0.05, None, {}, True)

    class S:
        pass
    s = S()
    s.dwi, s.bval, s.bvec, s.pe_sign, s.readout_s, s.reversed = d, b, g, 1.0, 0.05, r
    return s


def test_merge_scales_the_second_series_by_its_mean_b0():
    s = subject(k=3.0)
    m = SR.merge(s)
    assert m.dwi.shape[-1] == 12 and m.n_first == 6 and m.reversed is None
    assert (m.pe_signs == np.r_[np.ones(6), -np.ones(6)]).all()
    d0 = float(np.mean([s.dwi[..., v].mean() for v in (0, 1)]))
    r0 = float(np.mean([s.reversed.dwi[..., v].mean() for v in (0, 1)]))
    assert m.reversed_scale == pytest.approx(d0 / r0)
    assert float(m.dwi[..., 6:8].mean()) == pytest.approx(d0, rel=1e-5)
    assert (m.dwi[..., :6] == s.dwi).all()                                 # the diffusion series is left as it was


def test_axis_sets_counts_a_pair_once():
    _, g = table(n=8, nb0=0)
    u = g.T
    assert len(SR.axis_sets(np.r_[u, -u])) == 8


# ------------------------------------------------------------------------------------------------ the loader

def write(folder, stem, data, affine, b, g, pe, **side):
    import nibabel as nib
    folder.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(folder / f"{stem}.nii.gz"))
    np.savetxt(folder / f"{stem}.bval", b[None], fmt="%g")
    np.savetxt(folder / f"{stem}.bvec", g, fmt="%.8f")
    (folder / f"{stem}.json").write_text(json.dumps({"PhaseEncodingDirection": pe, "TotalReadoutTime": 0.05,
                                                     "EchoTime": 0.08, **side}))


def phantom(shape=(24, 26, 20)):
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    return ((x ** 2 + y ** 2 + z ** 2) < 0.6) * 100.0 + 1.0


def pair(tmp_path, pa_affine=None, pa_g=None, pa_b=None, nvol_pa=None):
    A = np.diag([2.0, 2.0, 2.0, 1.0])
    A[:3, 3] = -20, -22, -18
    b, g = table(n=6)
    base = phantom()
    d = np.stack([base] * len(b), -1).astype(np.float32)
    folder = tmp_path / "sub-01" / "dwi"
    write(folder, "sub-01_dir-AP_dwi", d, A, b, g, "j-")
    pb, pg = (b if pa_b is None else pa_b), (g if pa_g is None else pa_g)
    k = len(pb) if nvol_pa is None else nvol_pa
    write(folder, "sub-01_dir-PA_dwi", d[..., :k], A if pa_affine is None else pa_affine, pb[:k], pg[:, :k], "j")
    return folder / "sub-01_dir-AP_dwi.nii.gz", folder / "sub-01_dir-PA_dwi.nii.gz", b, g, A


def test_load_default_keeps_only_b0s(tmp_path):
    ap, _, b, _, _ = pair(tmp_path)
    s = bids.load(ap)
    assert s.pairing.ok and s.reversed is None and s.b0s.shape[-1] == 4


def test_load_keeps_the_reversed_series(tmp_path):
    ap, pa, b, g, A = pair(tmp_path)
    s = bids.load(ap, keep_reversed=True)
    r = s.reversed
    assert r.path == pa and r.pe_sign == -s.pe_sign and r.dwi.shape == s.dwi.shape and r.native
    assert (r.bvec == np.round(g, 8)).all() and (r.bval == b).all()
    assert "kept in full" in s.pairing.message


def test_load_turns_the_bvecs_of_a_partner_on_another_grid(tmp_path):
    """The same gradients in the scanner, the slab turned 3 degrees about z: the file's b-vectors differ, and after the
    frame change they are the diffusion series' again."""
    A = np.diag([2.0, 2.0, 2.0, 1.0])
    A[:3, 3] = -20, -22, -18
    t = np.radians(3.0)
    Rz = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])
    A2 = A.copy()
    A2[:3, :3] = Rz @ A[:3, :3]
    A2[:3, 3] = Rz @ (A[:3, 3] + np.array([20, 22, 18.0])) - np.array([20, 22, 18.0])
    b, g = table(n=6)
    Fl = np.diag([-1.0, 1.0, 1.0])                                         # det > 0: FSL flips x
    g2 = Fl @ Rz.T @ Fl @ g                                                # the same world direction, on the turned grid
    ap, _, _, _, _ = pair(tmp_path, pa_affine=A2, pa_g=g2)
    s = bids.load(ap, keep_reversed=True)
    assert not s.reversed.native and s.reversed.dwi.shape == s.dwi.shape
    assert np.allclose(s.reversed.bvec[:, 2:], g[:, 2:], atol=1e-6)
    assert "interpolated twice" in s.pairing.message


def test_load_refuses_a_partner_of_b0s_only(tmp_path):
    ap, *_ = pair(tmp_path, nvol_pa=2)
    assert bids.load(ap).pairing.ok                                        # as a field partner it is fine
    with pytest.raises(ValueError, match="diffusion-weighted volumes"):
        bids.load(ap, keep_reversed=True)


def test_load_refuses_two_candidates(tmp_path):
    ap, pa, b, g, A = pair(tmp_path)
    import nibabel as nib
    write(ap.parent, "sub-01_acq-x_dir-PA_dwi", np.asarray(nib.load(pa).dataobj), A, b, g, "j")
    with pytest.raises(ValueError, match="name the one to use"):
        bids.load(ap, keep_reversed=True)
    assert bids.load(ap, partners=[pa], keep_reversed=True).reversed.path == pa


def test_load_refuses_a_bvec_that_does_not_fit(tmp_path):
    ap, pa, *_ = pair(tmp_path)
    p = pa.with_name(pa.name.replace(".nii.gz", ".bvec"))
    np.savetxt(p, np.zeros((3, 4)))
    with pytest.raises(ValueError, match="does not fit"):
        bids.load(ap, keep_reversed=True)


# ------------------------------------------------------------------------------------------------ the config and the run

PINNED = {"default": "28d01a1d"}


def test_config_needs_motion_eddy_and_stays_out_of_the_hash():
    base = P.config_for(shell=2000.0, motion_eddy=True)
    with pytest.raises(ValueError, match="requires the motion_eddy slot"):
        P.config_for(shell=2000.0, reversed_series="concatenate")
    on = P.config_for(shell=2000.0, motion_eddy=True, reversed_series="pair-average")
    assert on.hash() != base.hash() != P.config_for(shell=2000.0, motion_eddy=True, reversed_series="concatenate").hash()
    assert "reversed_series" not in base.canonical()["slots"]
    assert on.opts("reversed_series") == {"angle_tol": 5.0, "b_tol": 50.0, "unmatched": "refuse"}
    assert PipelineConfig.from_dict(on.to_dict()) == on
    assert PipelineConfig.preset("default").hash().startswith(PINNED["default"])


def test_config_refuses_denoising_the_corrected_join():
    with pytest.raises(ValueError, match="denoise order"):
        P.config_for(shell=2000.0, motion_eddy=True, reversed_series="concatenate", denoise="mppca",
                     denoise_order="corrected")


def test_execute_refuses_a_subject_without_the_series(tmp_path):
    """execute itself, before the labeler or any work."""
    ap, *_ = pair(tmp_path)
    s = bids.load(ap)
    cfg = P.config_for(shell=1000.0, motion_eddy=True, reversed_series="concatenate")
    with pytest.raises(ValueError, match="keep_reversed"):
        P.execute(s, cfg, P.Timer(), device="cpu", workers=1)


def test_execute_refuses_unmatched_before_any_work(tmp_path):
    other = table(n=6, seed=9)[1]
    ap, *_ = pair(tmp_path, pa_g=other)
    s = bids.load(ap, keep_reversed=True)
    cfg = P.config_for(shell=1000.0, motion_eddy=True, reversed_series="pair-average")
    with pytest.raises(ValueError, match="no partner"):
        P.execute(s, cfg, P.Timer(), device="cpu", workers=1)
    assert P._check_reversed(s, P.config_for(shell=1000.0, motion_eddy=True, reversed_series="concatenate"), "cpu") is None


def test_scale_ignores_non_finite_values():
    s = subject()
    d = s.dwi.copy()
    d[0, 0, 0, 0] = np.nan
    assert SR.intensity_scale(d, s.bval, s.reversed.dwi, s.reversed.bval) > 0


def test_estimate_refuses_both_polarities_without_a_field():
    b, g = table(n=4)
    d = np.ones((8, 8, 8, 6), np.float32)
    with pytest.raises(ValueError, match="needs the field"):
        ME.estimate(d, b, g, np.eye(4), 1, pe_sign=np.r_[np.ones(3), -np.ones(3)], readout_s=0.05, device="cpu")


def test_memory_check_counts_both_series(monkeypatch):
    """One series fits the budget, both do not, and the refusal says why."""
    import torch
    shape = (100, 100, 70)
    one = 100 * np.prod(shape) * 4 * ME.MEMORY_FACTOR                       # the budget for 100 volumes, in bytes
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: int(one * 1.5), raising=False)

    class Dev:
        type = "mps"
    ME.check_memory(shape + (100,), Dev, "metal")
    with pytest.raises(ValueError, match="for the 200 volumes of both series"):
        ME.check_memory(shape + (200,), Dev, "metal", both_series=True)
