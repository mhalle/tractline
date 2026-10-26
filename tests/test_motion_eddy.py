"""tractline.motion_eddy: the Metal kernels against the float64 torch path, the inverse map, the field's resample
against susceptibility.apply, the b-vectors' rotation, and the whole estimate on a synthetic scan with known motion,
eddy currents and (optionally) a susceptibility field, on the CPU (torch) and on MPS (Metal), and its determinism."""
import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter
from tractline import motion_eddy as ME
from tractline import susceptibility as S

MPS = torch.backends.mps.is_available()


def geo64(shape=(30, 34, 26), vox=(2.2, 2.2, 2.5), pe=1, nmon=10):
    return ME.Geometry(shape, vox, pe, nmon, "cpu", torch.float64)


def smooth_field(shape, peak, seed=0):
    """A smooth field (voxels of displacement along PE): a frontal bump plus a gentle gradient, peak `peak`."""
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.normal(size=shape), 4.0)
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    f = f / np.abs(f).max() * 0.3 + np.exp(-((x - 0.1) ** 2 + (y - 0.5) ** 2 + (z + 0.2) ** 2) / 0.15)
    return f / np.abs(f).max() * peak


# ------------------------------------------------------------------------------------------------ pieces

def test_options():
    assert ME.Options.make(None) == ME.Options()
    assert ME.Options.make({"steps": 10}).steps == 10
    with pytest.raises(ValueError, match="unknown options"):
        ME.Options.make({"stepz": 10})
    with pytest.raises(ValueError):
        ME.Options(model="cubic")


def test_inverse_map():
    """phi(phi^-1(y)) = y by Newton steps, converging quadratically: on eddy terms stronger than real ones (up to 2
    voxels) in the default 2; with a field of up to 4 voxels (steep along PE) in the default 3."""
    geo = geo64()
    rng = np.random.default_rng(0)
    th = torch.tensor(np.c_[rng.normal(0, 1.5, (5, 3)), rng.normal(0, 0.03, (5, 3)), rng.normal(0, 0.4, (5, 10))])
    y = torch.tensor(rng.uniform(4, 22, (400, 3)))
    df = torch.tensor(smooth_field(geo.shape, 4.0))
    err = lambda n, f=None: float((ME.phi(geo, th, ME.phi_inv(geo, th, y, f, npass=n), f) - y).abs().max())  # noqa: E731
    assert err(2) < 1e-4 and err(3) < 1e-6
    assert err(3, df) < 1e-4 and err(2, df) < 0.1 * err(1, df), (err(1, df), err(2, df), err(3, df))


@pytest.mark.parametrize("peak", [0.0, 4.0])
def test_jacobian_is_the_chain_rule(peak):
    """The step's Jacobian, d/dtheta P(phi^-1(y)), against central differences (float64), with and without a field."""
    geo = geo64()
    rng = np.random.default_rng(0)
    th = torch.tensor(np.c_[rng.normal(0, 1.0, (3, 3)), rng.normal(0, 0.03, (3, 3)), rng.normal(0, 0.2, (3, 10))])
    coef = ME.t_prefilter(torch.tensor(rng.normal(size=(3,) + geo.shape)).cumsum(1).cumsum(2) / 10)
    y = torch.tensor(rng.uniform(8, 18, (300, 3)))
    df = torch.tensor(smooth_field(geo.shape, peak)) if peak else None
    x = ME.phi_inv(geo, th, y, df, 6)
    J = ME._jacobian(geo, th, x, ME.cubic_value_grad(coef, x)[1], ME._wgrad(geo, th, x, df))
    P = lambda t: ME.cubic_value_grad(coef, ME.phi_inv(geo, t, y, df, 8), want_grad=False)[0]   # noqa: E731
    for k in range(16):
        h = 1e-5
        tp, tm = th.clone(), th.clone()
        tp[:, k] += h
        tm[:, k] -= h
        fd = (P(tp) - P(tm)) / (2 * h) * geo.scale[k]
        assert float((J[..., k] - fd).norm() / fd.norm()) < 2e-3, k


def test_apply_field_is_susceptibility_apply():
    """No motion, no eddy, a field: the 3D cubic resample equals susceptibility.apply (cubic along PE, the Jacobian),
    away from the edges (they differ there: clamped against mirrored coefficients, zero outside the field of view)."""
    shape, pe = (24, 30, 20), 1
    rng = np.random.default_rng(1)
    dwi = gaussian_filter(rng.uniform(0, 100, shape + (3,)), (1.5, 1.5, 1.5, 0)) + 50
    h = smooth_field(shape, 40.0, seed=2)                                                  # Hz
    pe_sign, ro = -1.0, 0.05                                                               # up to 2 voxels
    ref = S.apply(dwi, h, pe, pe_sign, ro)
    r = ME.MotionEddy(theta=np.zeros((3, 16)), theta_fit=np.zeros((3, 16)), bvec=np.zeros((3, 3)), model="quadratic",
                      pe_axis=pe, shape=shape, vox=np.array([2.0, 2.0, 2.0]), groups=[], group_b=[], reference="first-b0",
                      field_used=True, pe_scale=pe_sign * ro, anchors={}, qc={}, options={"mask_output": False})
    out, _ = ME.apply_volumes(dwi, r, field_hz=h, device="cpu", dtype=torch.float64)
    core = (slice(3, -3), slice(5, -5), slice(3, -3))
    assert np.abs(out[core] - ref[core]).max() < 1e-3 * np.abs(ref[core]).max()


def test_bvec_rotation():
    """The b-vectors turn by R^T in the voxel frame (R: reference -> acquisition), FSL's x flip around it when
    det(affine) > 0: a volume rotated +10 deg about z measured gradient x; the tissue saw R^T x."""
    w = np.array([[0.0, 0.0, np.deg2rad(10.0)]])
    bvec = np.array([[1.0], [0.0], [0.0]])
    for aff, flip in ((np.diag([2.0, 2.0, 2.0, 1.0]), -1.0), (np.diag([-2.0, 2.0, 2.0, 1.0]), 1.0)):
        out = ME.rotate_bvecs(bvec, w, aff)[:, 0]
        c, s = np.cos(np.deg2rad(10)), np.sin(np.deg2rad(10))
        assert np.allclose(out, [c, -s * flip, 0.0], atol=1e-12)


def test_sh_smoother_is_basis_free():
    """The leave-one-out smoother: rows sum to 1 for the b0s' mean, zero diagonal; a constant signal is predicted
    exactly by an SH shell (degree 0 is not penalized)."""
    rng = np.random.default_rng(3)
    g = rng.normal(size=(40, 3))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    g[:4] = 0
    bval = np.r_[np.zeros(4), np.full(18, 1000.0), np.full(18, 2000.0)]
    groups, bs = ME.shell_groups(bval)
    assert bs == [0, 1000, 2000] and [len(i) for i in groups] == [4, 18, 18]
    W, orders = ME.loo_smoother(groups, bval < 50, g, 6, 0.06)
    assert np.allclose(np.diag(W), 0) and np.allclose(W.sum(1), 1, atol=1e-9)
    assert orders == {4: 4, 22: 4}                                                         # 18 directions: order 4


# ------------------------------------------------------------------------------------------------ the kernels

@pytest.mark.skipif(not MPS, reason="no MPS")
@pytest.mark.parametrize("with_field", [False, True])
def test_kernels_are_torch(with_field):
    """motion_eddy_metal's kernels (float32) against the torch path in float64: step_moments' H, b and cost, step_cost,
    resample, prefilter, smooth and apply_volumes."""
    from tractline import motion_eddy_metal as M
    torch.manual_seed(0)
    shape, G, N = (36, 40, 28), 6, 3000
    g64 = geo64(shape)
    g32 = ME.Geometry(shape, g64.vox.numpy(), 1, 10, "mps", torch.float32)
    coef = torch.randn(G, *shape, dtype=torch.float64)
    pts = torch.rand(N, 3, dtype=torch.float64) * (torch.tensor(shape) - 10) + 5
    obs = torch.randn(G, N, dtype=torch.float64)
    th = torch.cat([torch.randn(G, 3), torch.randn(G, 3) * 0.02, torch.randn(G, 10) * 0.3], 1).double()
    df = torch.tensor(smooth_field(shape, 2.5)) if with_field else None
    m = lambda t: t.float().to("mps").contiguous()                                       # noqa: E731
    dfk = m(df) if with_field else torch.zeros(1, device="mps")
    ddf = torch.gradient(df, dim=1)[0] if with_field else None
    ddfk = m(ddf) if with_field else dfk
    H, b, f = ME.t_step_moments(g64, coef, obs, pts, th, (0, 0, 0), df, npass=3, ddfield=ddf)
    nch, ip, fp = M.consts(g32, G, N, shape, npass=3, has_field=with_field)
    Hm, bm, fm = M.step_moments(m(coef), m(obs), m(pts), ME.pack(m(th)), nch, ip, fp, dfk, ddfk, 16)
    rel = lambda a, r: float((a.cpu().double() - r).abs().max() / r.abs().max())         # noqa: E731
    assert rel(Hm, H) < 1e-4 and rel(bm, b) < 1e-4 and rel(fm, f) < 1e-5
    k = torch.tensor([1, 4])
    fc = ME.t_step_cost(g64, coef, obs, pts, th[k], k, (0, 0, 0), df, npass=3, ddfield=ddf)
    fcm = M.step_cost(m(coef), m(obs), m(pts), ME.pack(m(th[k])), k.to("mps"), nch, ip, fp, dfk, ddfk)
    assert rel(fcm, fc) < 1e-5
    vols = torch.rand(G, *shape, dtype=torch.float64) * 100
    nch2, ip2, fp2 = M.consts(g32, G, N, shape, has_field=with_field)
    rs = ME.t_resample(g64, vols, pts, th, df)
    assert rel(M.resample(m(vols), m(pts), ME.pack(m(th)), ip2, fp2, dfk), rs) < 1e-5
    assert rel(M.prefilter(m(vols)), ME.t_prefilter(vols)) < 1e-5
    assert rel(M.smooth(m(vols), 2.0), ME.t_smooth(vols, 2.0)) < 1e-5
    out, valid = ME.t_apply_volumes(g64, vols, th, df, ddf)
    nch3, ip3, fp3 = M.consts(g32, G, 1, shape, has_field=with_field)
    outm, validm = M.apply_volumes(m(vols), ME.pack(m(th)), ip3, fp3, dfk, ddfk)
    inner = (valid > 0) & (validm.cpu().double() > 0)
    assert float((valid != validm.cpu().double()).double().mean()) < 1e-3
    assert float(((outm.cpu().double() - out) * inner).abs().max()) < 1e-3 * float(out.abs().max())


# ------------------------------------------------------------------------------------------------ a synthetic scan

def tensor_scan(seed=0, shape=(40, 44, 34), vox=2.5, nb0=4, shells=((1000, 30), (2000, 30)), motion=1.0, eddy=0.6,
                field_peak=0.0, pe=1, sigma=1.0, reference="b0-mean", lams=(0.8e-3, 0.8e-3), sh_order=6):
    """A scan with a known truth: tensors in an ellipsoid with texture; per volume a rigid motion (mm, deg ~ motion),
    a quadratic eddy displacement linear in the gradient (voxels ~ eddy, no intercept per shell, no constant), and a
    field's displacement (peak voxels); the tissue sees each gradient turned by its volume's R^T. Each group's mean pose
    is the b0s' mean pose, which is zero (the gauge the estimate holds; with reference "first-b0" the first b0's pose
    is zero instead, and the b0s' mean is not).
    lams: the tensors' eigenvalues (mm^2/s; the default isotropic). With anisotropy coherent across the whole brain the
    shells' leave-one-out prediction misfits in a spatially structured way and biases the fit, the prototype's alike:
    with (1.1e-3, 0.55e-3) and no motion at all, 0.25-0.34 voxel of spurious displacement for any lambda 0.002-0.2
    (NOTES 2026-10-26); isotropic, 0.02. So the regression cases here are isotropic."""
    rng = np.random.default_rng(seed)
    X, Y, Z = shape
    x, y, z = np.meshgrid(*[np.linspace(-1, 1, n) for n in shape], indexing="ij")
    brain = (x / 0.8) ** 2 + (y / 0.8) ** 2 + (z / 0.75) ** 2 < 1
    S0 = gaussian_filter(brain * (100 + 40 * gaussian_filter(rng.normal(size=shape), 1.5) / 0.15), 0.8)
    a = np.pi * (x + 0.5 * z)
    e1 = np.stack([np.cos(a), np.sin(a), 0.4 * np.ones_like(a)], -1)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    D = lams[1] * np.eye(3) + (lams[0] - lams[1]) * e1[..., :, None] * e1[..., None, :]      # mm^2/s
    D = np.where(brain[..., None, None], D, 2.5e-3 * np.eye(3))
    dirs, bval = [np.zeros((nb0, 3))], [np.zeros(nb0)]
    for b, n in shells:                                                                    # spread directions (Fibonacci)
        k = np.arange(n) + 0.5
        zc, ph = 1 - 2 * k / n, np.pi * (1 + 5 ** 0.5) * k
        v = np.c_[np.sqrt(1 - zc ** 2) * np.cos(ph), np.sqrt(1 - zc ** 2) * np.sin(ph), zc]
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        dirs.append(v @ Q.T)
        bval.append(np.full(n, float(b)))
    gv, bval = np.concatenate(dirs), np.concatenate(bval)                                 # voxel-frame directions
    G = len(bval)
    affine = np.diag([vox, vox, vox, 1.0])                                                 # det > 0: FSL flips x
    Fl = np.diag([-1.0, 1.0, 1.0])
    bvec = (Fl @ gv.T)
    th = np.zeros((G, 16))
    th[:, :3] = rng.normal(0, motion, (G, 3))
    th[:, 3:6] = np.deg2rad(rng.normal(0, motion, (G, 3)))
    groups, _ = ME.shell_groups(bval)
    A = rng.normal(0, 1, (10, 3)) * eddy
    A[0] = 0
    A[4:] *= 0.5
    th[:, 6:] = (gv * np.sqrt(bval / 2000.0)[:, None]) @ A.T
    b0g = groups[0]
    if reference == "first-b0":
        th[b0g, :6] -= th[b0g[0], :6]                                                     # first-order: small poses
    else:
        th[b0g, :6] -= th[b0g, :6].mean(0)
    for idx in groups:
        if bval[idx[0]] >= 50:
            th[idx, :6] -= th[idx, :6].mean(0) - th[b0g, :6].mean(0)                       # the shell at the b0s' mean pose
            Xg = np.c_[np.ones(len(idx)), gv[idx]]
            th[idx, 6:] -= np.outer(np.ones(len(idx)), np.linalg.pinv(Xg)[0] @ th[idx, 6:])
    geo = ME.Geometry(shape, [vox] * 3, pe, 10, "cpu", torch.float64)
    R = ME.rot(torch.tensor(th[:, 3:6])).numpy()
    q = np.einsum("gji,gj->gi", R, gv)                                                     # R^T g: what the tissue saw
    # the truth is each shell's tensor signal fitted by regularized SH (order 6, lambda 0.06; the predictor's own space,
    # as the prototype's truth was) and evaluated at the direction the tissue saw
    sig = np.repeat(S0[..., None], G, -1)
    for idx in groups:
        if bval[idx[0]] < 50:
            continue
        B, ls = ME.sh_basis(sh_order, gv[idx])
        nominal = S0[..., None] * np.exp(-bval[idx] * np.einsum("gi,xyzij,gj->xyzg", gv[idx], D, gv[idx]))
        C = nominal @ np.linalg.solve(B.T @ B + 0.06 * np.diag((ls * (ls + 1.0)) ** 2), B.T).T
        sig[..., idx] = C @ ME.sh_basis(sh_order, q[idx])[0].T
    dfield = smooth_field(shape, field_peak, seed + 7) if field_peak else None
    full = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float64) for n in shape], indexing="ij"), -1).reshape(-1, 3)
    tt = torch.tensor(th)
    df = torch.tensor(dfield) if dfield is not None else None
    xs = ME.phi_inv(geo, tt, full, df, npass=12)
    coef = ME.t_prefilter(torch.tensor(np.moveaxis(sig, -1, 0).copy()))
    val, _ = ME.cubic_value_grad(coef, xs, want_grad=False)
    jac = 1 + ME.eddy_dpe(geo, tt[:, 6:], xs)
    if df is not None:
        jac = jac + ME.field_at(torch.gradient(df, dim=pe)[0], xs)
    acq = (val / jac).reshape(G, *shape).numpy()
    acq = np.clip(acq + rng.normal(0, sigma, acq.shape), 0, None)
    return dict(dwi=np.moveaxis(acq, 0, -1).astype(np.float32), bval=bval, bvec=bvec, affine=affine, pe_axis=pe,
                theta=th, sig=sig, dfield=dfield, brain=brain, gv=gv, R=R)


def edge_error(t, r):
    """RMS over volumes and the brain's edge of |phi_est - phi_true| (voxels), the field excluded (the same in both)."""
    from scipy.ndimage import binary_erosion
    geo = ME.Geometry(t["brain"].shape, [t["affine"][0, 0]] * 3, t["pe_axis"], 10, "cpu", torch.float64)
    edge = t["brain"] & ~binary_erosion(t["brain"])
    xe = torch.tensor(np.argwhere(edge), dtype=torch.float64)
    est = torch.tensor(r.theta_fit)
    est = torch.nn.functional.pad(est, (0, 16 - est.shape[1]))
    d = ME.phi(geo, est, xe) - ME.phi(geo, torch.tensor(t["theta"]), xe)
    return float(d.norm(dim=-1).pow(2).mean().sqrt())


def true_disp(t):
    from scipy.ndimage import binary_erosion
    geo = ME.Geometry(t["brain"].shape, [t["affine"][0, 0]] * 3, t["pe_axis"], 10, "cpu", torch.float64)
    edge = t["brain"] & ~binary_erosion(t["brain"])
    xe = torch.tensor(np.argwhere(edge), dtype=torch.float64)
    return float((ME.phi(geo, torch.tensor(t["theta"]), xe) - xe).norm(dim=-1).pow(2).mean().sqrt())


def bvec_error_deg(t, r):
    """The angle (deg) between the estimate's rotated b-vectors and the directions the tissue saw (FSL frame)."""
    want = np.diag([-1.0, 1.0, 1.0]) @ np.einsum("gji,gj->gi", t["R"], t["gv"]).T
    dw = t["bval"] >= 50
    c = np.abs((want[:, dw] * r.bvec[:, dw]).sum(0)).clip(0, 1)
    return np.degrees(np.arccos(c))


@pytest.fixture(scope="module")
def scan():
    return tensor_scan()


def test_estimate_cpu(scan):
    """The torch path on the CPU recovers the known motion and eddy fields and turns the b-vectors with the head."""
    r = ME.estimate(scan["dwi"], scan["bval"], scan["bvec"], scan["affine"], scan["pe_axis"], device="cpu",
                    options={"steps": 25})
    before, after = true_disp(scan), edge_error(scan, r)
    assert after < 0.15 * before and after < 0.2, (before, after)
    err = bvec_error_deg(scan, r)
    unrotated = bvec_error_deg(scan, ME.MotionEddy(**{**r.__dict__, "bvec": scan["bvec"]}))
    assert np.median(err) < 0.3 and np.median(err) < 0.5 * np.median(unrotated)
    q = r.qc
    assert len(q["framewise displacement, mm per volume"]) == len(scan["bval"]) and q["steps run"] == 25
    corr, valid = ME.apply_volumes(scan["dwi"], r, device="cpu")
    assert corr.shape == scan["dwi"].shape and valid.any()


@pytest.mark.skipif(not MPS, reason="no MPS")
def test_estimate_mps_deterministic(scan):
    """The Metal path: as accurate as the CPU's, and the same parameters on a second run, bit for bit."""
    a = ME.estimate(scan["dwi"], scan["bval"], scan["bvec"], scan["affine"], scan["pe_axis"], device="mps",
                    options={"steps": 25})
    b = ME.estimate(scan["dwi"], scan["bval"], scan["bvec"], scan["affine"], scan["pe_axis"], device="mps",
                    options={"steps": 25})
    assert np.array_equal(a.theta_fit, b.theta_fit)
    assert edge_error(scan, a) < 0.15 * true_disp(scan)


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if MPS else []))
def test_estimate_with_field(device):
    """With a susceptibility field (up to 3 voxels) composed into the map: the motion and eddy fields are recovered
    with the field given, and apply_volumes' single resample brings back the truth better than the data as acquired."""
    t = tensor_scan(seed=4, field_peak=3.0, reference="first-b0")
    h_hz = t["dfield"] / (-1.0 * 0.05)                                                     # pe_sign -1, readout 0.05 s
    r = ME.estimate(t["dwi"], t["bval"], t["bvec"], t["affine"], t["pe_axis"], field_hz=h_hz, pe_sign=-1.0,
                    readout_s=0.05, device=device, options={"steps": 25})
    assert r.reference == "first-b0" and r.field_used
    before, after = true_disp(t), edge_error(t, r)
    assert after < 0.2 * before and after < 0.25, (before, after)
    corr, valid = ME.apply_volumes(t["dwi"], r, field_hz=h_hz, device=device, mask_output=False)
    from scipy.ndimage import binary_erosion
    core = binary_erosion(t["brain"], iterations=3) & valid
    truth = t["sig"]
    e_corr = np.sqrt(((corr - truth)[core] ** 2).mean())
    e_acq = np.sqrt(((t["dwi"] - truth)[core] ** 2).mean())
    assert e_corr < 0.5 * e_acq, (e_corr, e_acq)


def test_correct_one_call():
    """correct(): the field from the reversed pair, then motion and eddy given it, then one resample, on the test
    phantom; the plumbing only (its accuracy with a field: test_estimate_with_field). The phantom suits no accuracy
    check here: a uniform box crossed by a sharply anisotropic band (the misfit bias of tensor_scan's docstring; its DW
    volumes drift up to 1.4 voxels along the band, with or without the field). The first b0, the field's frame and the
    fit's reference, comes back as acquired but for the field's own resample."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from phantom import subject
    s = subject()
    c = ME.correct(s, device="cpu", options={"steps": 4, "model": "rigid"})
    assert c.field_hz is not None and c.motion_eddy.field_used and c.motion_eddy.reference == "first-b0"
    assert c.dwi.shape == s.dwi.shape and c.bvec.shape == s.bvec.shape and set(c.seconds) == {"field", "motion_eddy", "apply"}
    assert np.array_equal(c.motion_eddy.theta_fit[0], np.zeros(6))
    core = s.brain & c.valid
    assert np.abs(c.dwi[..., 0][core] - s.dwi[..., 0][core]).mean() < 0.01 * s.dwi[..., 0][core].mean()
    s.b0s = None                                                                           # no pair: motion and eddy only
    c2 = ME.correct(s, device="cpu", options={"steps": 2, "model": "rigid"})
    assert c2.field_hz is None and not c2.motion_eddy.field_used and c2.motion_eddy.reference == "b0-mean"
