"""tractline.motion_eddy: the Metal kernels against the float64 torch path, the inverse map, the field's resample
against susceptibility.apply, the b-vectors' rotation, and the whole estimate on a synthetic scan with known motion,
eddy currents and (optionally) a susceptibility field, on the CPU (torch) and on MPS (Metal), and its determinism."""
import json
import numpy as np
import pytest
import torch
from scipy.ndimage import binary_erosion, gaussian_filter
from tractline import motion_eddy as ME
from tractline import susceptibility as S

MPS = torch.backends.mps.is_available()
CUDA = torch.cuda.is_available()
from tractline import motion_eddy_triton as _T                             # noqa: E402
FUSED = [b for b, ok in (("metal", MPS), ("triton", _T.available())) if ok]   # the fused kernels this machine runs
FUSED_DEV = {"metal": "mps", "triton": "cuda"}


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

@pytest.mark.skipif(not FUSED, reason="no MPS or CUDA")
@pytest.mark.parametrize("backend", FUSED or ["none"])
@pytest.mark.parametrize("with_field", [False, True])
def test_kernels_are_torch(with_field, backend):
    """The fused kernels (motion_eddy_metal's on MPS, motion_eddy_triton's on CUDA; float32) against the torch path in
    float64: step_moments' H, b and cost, step_cost, step_values, resample, prefilter, smooth and apply_volumes."""
    M = ME._kernels(backend)
    dev = FUSED_DEV[backend]
    torch.manual_seed(0)
    shape, G, N = (36, 40, 28), 6, 3000
    g64 = geo64(shape)
    g32 = ME.Geometry(shape, g64.vox.numpy(), 1, 10, dev, torch.float32)
    coef = torch.randn(G, *shape, dtype=torch.float64)
    pts = torch.rand(N, 3, dtype=torch.float64) * (torch.tensor(shape) - 10) + 5
    obs = torch.randn(G, N, dtype=torch.float64)
    th = torch.cat([torch.randn(G, 3), torch.randn(G, 3) * 0.02, torch.randn(G, 10) * 0.3], 1).double()
    df = torch.tensor(smooth_field(shape, 2.5)) if with_field else None
    m = lambda t: t.float().to(dev).contiguous()                                         # noqa: E731
    dfk = m(df) if with_field else torch.zeros(1, device=dev)
    ddf = torch.gradient(df, dim=1)[0] if with_field else None
    ddfk = m(ddf) if with_field else dfk
    H, b, f = ME.t_step_moments(g64, coef, obs, pts, th, (0, 0, 0), df, npass=3, ddfield=ddf)
    nch, ip, fp = M.consts(g32, G, N, shape, npass=3, has_field=with_field)
    Hm, bm, fm = M.step_moments(m(coef), m(obs), m(pts), ME.pack(m(th)), nch, ip, fp, dfk, ddfk, 16)
    rel = lambda a, r: float((a.cpu().double() - r).abs().max() / r.abs().max())         # noqa: E731
    assert rel(Hm, H) < 1e-4 and rel(bm, b) < 1e-4 and rel(fm, f) < 1e-5
    k = torch.tensor([1, 4])
    fc = ME.t_step_cost(g64, coef, obs, pts, th[k], k, (0, 0, 0), df, npass=3, ddfield=ddf)
    fcm = M.step_cost(m(coef), m(obs), m(pts), ME.pack(m(th[k])), k.to(dev), nch, ip, fp, dfk, ddfk)
    assert rel(fcm, fc) < 1e-5
    sv = ME.t_step_values(g64, coef, pts, th[k], k, (0, 0, 0), df, npass=3, ddfield=ddf)
    assert rel(M.step_values(m(coef), m(pts), ME.pack(m(th[k])), k.to(dev), ip, fp, dfk, ddfk), sv) < 1e-5
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


@pytest.mark.skipif(not FUSED, reason="no MPS or CUDA")
@pytest.mark.parametrize("backend", FUSED or ["none"])
def test_estimate_fused_deterministic(scan, backend):
    """The fused path (Metal on MPS, Triton on CUDA): as accurate as the CPU's, and the same parameters on a second
    run, bit for bit."""
    dev = FUSED_DEV[backend]
    a = ME.estimate(scan["dwi"], scan["bval"], scan["bvec"], scan["affine"], scan["pe_axis"], device=dev,
                    options={"steps": 25})
    b = ME.estimate(scan["dwi"], scan["bval"], scan["bvec"], scan["affine"], scan["pe_axis"], device=dev,
                    options={"steps": 25})
    assert a.backend == backend and np.array_equal(a.theta_fit, b.theta_fit)
    assert edge_error(scan, a) < 0.15 * true_disp(scan)


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if MPS else []) + (["cuda"] if CUDA else []))
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


# ------------------------------------------------------------------------------------------------ robustness
# (the adversarial review of 2026-10-26: one bad volume or voxel had turned every volume's parameters NaN, or cropped
# every corrected volume, without a warning)

DEV = "mps" if MPS else "cpu"


def corrupt(t, how):
    d = t["dwi"].copy()
    if how == "nan voxel":
        d[20, 22, 17, 10] = np.nan
    elif how == "zero volume":
        d[..., 20] = 0
    elif how == "dropout":
        d[:, :, 15:18, 40] = 0
    return d


@pytest.mark.parametrize("how", ["nan voxel", "zero volume", "dropout", "dropout, outliers off"])
def test_one_bad_volume_stays_local(scan, how):
    """A NaN voxel is set to 0 and counted; an empty volume is left out of the fit (identity motion, b-vector as
    acquired, the reason recorded); a volume with dropout slices keeps its place, its slices flagged as outliers and
    imputed (or, with outliers off, it is left out). Every other volume is still recovered, and nothing crops the
    others' output."""
    d = corrupt(scan, how.split(",")[0])
    opts = {"steps": 25, **({"outliers": "off"} if "off" in how else {})}
    r = ME.estimate(d, scan["bval"], scan["bvec"], scan["affine"], 1, device=DEV, options=opts)
    assert np.isfinite(r.theta_fit).all()
    bad = {"nan voxel": None, "zero volume": 20, "dropout": None, "dropout, outliers off": 40}[how]
    if how == "nan voxel":
        assert r.qc["non-finite values set to 0"] == 1 and not r.excluded
    elif how == "dropout":
        assert not r.excluded and set(r.outlier_slices) == {40} and set(r.outlier_slices[40]) >= {15, 16, 17}
    else:
        assert list(r.excluded) == [bad] and not r.theta_fit[bad].any()
        assert np.array_equal(r.bvec[:, bad], scan["bvec"][:, bad])
    keep = np.ones(len(scan["bval"]), bool)
    if bad is not None:
        keep[bad] = False
    sub = ME.MotionEddy(**{**r.__dict__, "theta_fit": r.theta_fit[keep]})
    t2 = {**scan, "theta": scan["theta"][keep]}
    assert edge_error(t2, sub) < 0.15 * true_disp(t2)
    _, valid = ME.apply_volumes(d, r, device=DEV)
    assert valid[binary_erosion(scan["brain"], iterations=2)].mean() > 0.99
    json.dumps(ME.record(r), allow_nan=False)


def test_outlier_slices_replaced(scan):
    """Dropout slices are found and replaced: the corrected volume's slices come back near the truth, the clean data
    gives no outlier slices, and "flag" reports them but leaves the output as acquired there."""
    clean = ME.estimate(scan["dwi"], scan["bval"], scan["bvec"], scan["affine"], 1, device=DEV, options={"steps": 25})
    assert not clean.outlier_slices
    d = scan["dwi"].copy()
    d[:, :, 10:20, 50] *= 0.3                                               # a slab of signal loss in volume 50
    r = ME.estimate(d, scan["bval"], scan["bvec"], scan["affine"], 1, device=DEV, options={"steps": 25})
    assert set(r.outlier_slices) == {50} and set(range(11, 19)) <= set(r.outlier_slices[50])
    corr, valid = ME.apply_volumes(d, r, device=DEV, mask_output=False)
    core = binary_erosion(scan["brain"], iterations=3) & valid
    core[:, :, :11] = core[:, :, 19:] = False
    err = np.abs(corr[..., 50] - scan["sig"][..., 50])[core].mean()
    assert err < 0.15 * scan["sig"][..., 50][core].mean(), err
    f = ME.estimate(d, scan["bval"], scan["bvec"], scan["affine"], 1, device=DEV, options={"steps": 25, "outliers": "flag"})
    assert set(f.outlier_slices) == {50} and not f.replacements


def test_runaway_guard(scan):
    """A volume whose map moves the brain's edge more than max_motion_mm leaves the fit (here the limit is set below
    the real motion, so volumes that move are caught), and the fit stays finite."""
    r = ME.estimate(scan["dwi"], scan["bval"], scan["bvec"], scan["affine"], 1, device=DEV,
                    options={"steps": 6, "max_motion_mm": 2.0})
    assert r.excluded and all("moved the brain's edge" in why for why in r.excluded.values())
    assert np.isfinite(r.theta_fit).all() and not r.theta_fit[list(r.excluded)].any()


def test_small_shell_rigid_only_and_shell_grouping():
    """A shell of 2-4 volumes fits rigid motion only (its common eddy field has no gauge); b-values that straddle a
    multiple of 50 stay one shell."""
    groups, bs = ME.shell_groups(np.r_[0, 0, 1049, 1051, 1049, 2000, 2005, 3000])
    assert bs == [0, 1000, 2000, 3000] and [len(i) for i in groups] == [2, 3, 2, 1]
    t = tensor_scan(seed=6, shells=((500, 2), (1000, 30)), eddy=0.0)
    r = ME.estimate(t["dwi"], t["bval"], t["bvec"], t["affine"], 1, device=DEV, options={"steps": 15})
    small = [int(i) for i in np.flatnonzero(t["bval"] == 500)]
    assert r.qc["volumes fitted rigid only (a shell of 2-4: no gauge for its eddy field)"] == small
    assert not r.theta_fit[small, 6:].any()


@pytest.mark.skipif(not MPS, reason="no MPS")
def test_memory_check(scan, monkeypatch):
    monkeypatch.setattr(torch.mps, "recommended_max_memory", lambda: 1 << 20)
    with pytest.raises(ValueError, match="needed on MPS"):
        ME.estimate(scan["dwi"], scan["bval"], scan["bvec"], scan["affine"], 1, device="mps", options={"steps": 1})


def test_screen_motion_is_not_dropout():
    """screen(): a slice-specific dip is dropout; the brain's end moving through the slices (motion) is not; a dark
    volume is corrupt."""
    rng = np.random.default_rng(0)
    shape, G = (40, 40, 30), 12
    x, y, z = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    vols = []
    for v in range(G):
        shift = 3.0 if v == 3 else 0.0                                     # volume 3: the brain 3 slices lower
        brain = ((x - 20) ** 2 + (y - 20) ** 2) / 15 ** 2 + ((z - 15 + shift) / 12.0) ** 2 < 1
        vols.append(brain * 100.0 + rng.normal(0, 1, shape))
    d = np.stack(vols, -1).astype(np.float32)
    d[:, :, 14, 5] *= 0.3                                                   # volume 5: dropout in slice 14
    d[..., 8] *= 0.3                                                        # volume 8: dark throughout
    pts = np.mean(d[..., :3], -1) > 50
    out = ME.screen(d, np.full(G, 1000.0), [np.arange(G)], pts)
    assert set(out) == {5, 8} and "slices [14]" in out[5] and "mean signal" in out[8]


def test_tracking_leaves_out_left_out_volumes(monkeypatch):
    """The pipeline tracks without the volumes the step left out (they stay in Correction.dwi, uncorrected for
    motion)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from phantom import subject
    from tractline import pipeline as P
    s = subject()
    s.dwi = s.dwi.copy()
    s.dwi[..., 7] = 0                                                       # an empty DW volume
    seen = {}
    real = P.prepare

    def spy(dwi, affine, bval, bvec, **kw):
        seen.update(n=dwi.shape[-1], bval=np.array(bval))
        return real(dwi, affine, bval, bvec, **kw)
    monkeypatch.setattr(P, "prepare", spy)

    def stand_in(fibers, draws=(0,), logp=False):
        from tractline.labelers.base import Labels, MIN_LENGTH_MM, lengths
        _, _, length = lengths(fibers)
        keep = length >= MIN_LENGTH_MM
        return Labels(keep=keep, length_mm=length[keep], tract=np.zeros(int(keep.sum()), np.int64), logp=None)
    cfg = P.config_for(stand_in, motion_eddy={"model": "rigid", "steps": 2})
    r = P.execute(s, cfg, P.Timer(), device="cpu", workers=1, labeler=stand_in)
    assert "7" in r.correction.motion_eddy["left out"] and seen["n"] == s.dwi.shape[-1] - 1
    assert r.correction.dwi.shape == s.dwi.shape and any("left out of the fit and of tracking" in w
                                                          for w in r.correction.warnings)


def test_screen_finds_a_slab():
    """A slab of several dark slices in the middle of the brain is dropout too (a run with normal slices both sides)."""
    rng = np.random.default_rng(1)
    shape, G = (40, 40, 30), 10
    x, y, z = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    brain = ((x - 20) ** 2 + (y - 20) ** 2) / 15 ** 2 + ((z - 15) / 12.0) ** 2 < 1
    d = np.stack([brain * 100.0 + rng.normal(0, 1, shape) for _ in range(G)], -1).astype(np.float32)
    d[:, :, 10:18, 4] *= 0.3
    out = ME.screen(d, np.full(G, 1000.0), [np.arange(G)], np.mean(d[..., :3], -1) > 50)
    assert set(out) == {4} and "slices [10, 11, 12, 13, 14, 15, 16, 17]" in out[4]


def test_outliers_respect_gain_and_left_out(scan):
    """The review of the outlier slices: a volume dim throughout (its gain) has no outlier slices; a dropout volume's
    imputation is refreshed as the registration improves, so its own motion is recovered; a seeded volume that later
    leaves the fit gets no replacements."""
    d = scan["dwi"].copy()
    d[..., 30] *= 0.85                                                      # dim throughout: a gain, not slice loss
    d[:, :, 15:18, 40] = 0                                                  # dropout: seeded by the screen
    r = ME.estimate(d, scan["bval"], scan["bvec"], scan["affine"], 1, device=DEV, options={"steps": 25})
    assert 30 not in r.outlier_slices and 40 in r.outlier_slices
    one = ME.MotionEddy(**{**r.__dict__, "theta_fit": r.theta_fit[[40]]})
    assert edge_error({**scan, "theta": scan["theta"][[40]]}, one) < 0.15, "the dropout volume's own motion"
    r2 = ME.estimate(d, scan["bval"], scan["bvec"], scan["affine"], 1, device=DEV,
                     options={"steps": 6, "max_motion_mm": 2.0})
    for v in r2.excluded:
        assert v not in r2.replacements and v not in r2.outlier_slices


@pytest.mark.skipif(not CUDA, reason="no CUDA")
def test_memory_check_cuda(scan, monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (1 << 20, 1 << 30))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 0)
    with pytest.raises(ValueError, match="needed on CUDA"):
        ME.estimate(scan["dwi"], scan["bval"], scan["bvec"], scan["affine"], 1, device="cuda", options={"steps": 1})


def test_warm_combos():
    """The combinations warm() compiles: the estimator's default Newton passes for each field case."""
    assert _T.combos() == [(1, 10, 1, 3), (1, 10, 0, 2)]
    assert len(_T.combos((0, 1, 2), tuple(_T.MODELS), (True, False))) == 18


@pytest.mark.skipif(not _T.available(), reason="no CUDA with Triton")
def test_warm_then_run(scan):
    """warm() compiles a combination without running it; an estimate with those settings then runs as before."""
    secs = _T.warm(_T.combos((1,), ("quadratic",), (False,)), workers=1)
    assert list(secs) == [(1, 10, 0, 2)]
    r = ME.estimate(scan["dwi"], scan["bval"], scan["bvec"], scan["affine"], 1, device="cuda", options={"steps": 3})
    assert r.backend == "triton"


@pytest.mark.skipif(not FUSED, reason="no MPS or CUDA")
@pytest.mark.parametrize("backend", FUSED or ["none"])
@pytest.mark.parametrize("with_field", [False, True])
def test_devices_agree(backend, with_field):
    """The fused kernels (Metal, Triton) against the CPU's torch path on a scan with known motion, with and without a
    field: the parameters agree far inside the estimate's own error (measured on the M2: 8e-5 mm, 0.001 degrees,
    0.0002 voxels of eddy; the edge error against the truth 0.062 on both). The pipeline's phantom is not used here: a
    single straight band leaves the registration ill-posed, and both devices then return large, different motions."""
    kw = {"field_peak": 3.0, "reference": "first-b0", "seed": 4} if with_field else {}
    t = tensor_scan(**kw)
    extra = dict(field_hz=t["dfield"] / (-1.0 * 0.05), pe_sign=-1.0, readout_s=0.05) if with_field else {}
    est = {d: ME.estimate(t["dwi"], t["bval"], t["bvec"], t["affine"], t["pe_axis"], device=d, options={"steps": 25}, **extra)
           for d in ("cpu", FUSED_DEV[backend])}
    a, b = est["cpu"].theta_fit, est[FUSED_DEV[backend]].theta_fit
    assert est[FUSED_DEV[backend]].backend == backend
    assert np.abs(a[:, :3] - b[:, :3]).max() < 2e-3                                          # mm
    assert np.degrees(np.abs(a[:, 3:6] - b[:, 3:6]).max()) < 0.025                            # degrees
    assert np.abs(a[:, 6:] - b[:, 6:]).max() < 5e-3                                           # voxels along PE
    assert abs(edge_error(t, est["cpu"]) - edge_error(t, est[FUSED_DEV[backend]])) < 0.005
