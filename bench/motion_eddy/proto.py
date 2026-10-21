"""Prototype of the design's estimator (docs/gpu-motion-eddy-design.md, sections 1-2) on synthetic data with known motion
and eddy fields: batched Levenberg-Marquardt Gauss-Newton over all volumes at once, each volume registered to its
leave-one-out SH prediction from the others, rigid (6) + eddy polynomial along PE (linear 4 or quadratic 10).

Truth: a real ON-Harmony scan's b = 2000 shell (50 volumes), replaced by its SH fit (order 6, Laplace-Beltrami
regularized), so the truth has the scan's anatomy and diffusion contrast and no motion. Each volume is then moved by a
known rigid transform and distorted along PE by a known eddy polynomial, linear in the gradient direction, resampled
(cubic B-spline, 8 trilinear fetches) and given Gaussian noise at the scan's own residual level. No susceptibility field
(h = 0): the composition with h is the existing code's.

    uv run --group dipy bench/motion_eddy/proto.py --device mps|cpu [--acq-cost --cubic --outer 15 --inner 1]   # the final form
    uv run --group dipy bench/motion_eddy/proto.py --device mps --save-synth $TRACTOGRAPHY_DATA/derived/motion_eddy_synth   # the scan eddy ran on
Writes results/motion_eddy/proto_<device>[...].json.
"""
import argparse, json, math, time, sys
import numpy as np, torch, torch.nn.functional as F
import nibabel as nib
from pathlib import Path
from tractline.data import DATA                                   # $TRACTOGRAPHY_DATA
OUT = Path(__file__).resolve().parent.parent / "results/motion_eddy"
SYNTH = DATA / "derived/motion_eddy_synth"                     # proto.py --save-synth; eddy runs beside it

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="mps")
ap.add_argument("--order", type=int, default=2, help="eddy polynomial order, 1 or 2")
ap.add_argument("--no-loo", action="store_true", help="predict each volume from all volumes, itself included")
ap.add_argument("--scale", type=float, default=1.0, help="multiplies the simulated motion and eddy")
ap.add_argument("--outer", type=int, default=5)
ap.add_argument("--eddy-scale", type=float, default=1.0, help="multiplies the simulated eddy only")
ap.add_argument("--inner", type=int, default=3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--grad", default="autograd", choices=["autograd", "taps"], help="with --fast: the cubic gradient by autograd through the 8 fetches, or the one-pass 64 taps")
ap.add_argument("--selftest", action="store_true", help="check the fast path's pieces against the originals, then exit")
ap.add_argument("--fast", action="store_true", help="the final form (acquisition-space residual, cubic, one step per prediction) with the profile's fixes")
ap.add_argument("--coarse", type=int, default=12, help="with --fast: the first steps on every other voxel per axis")
ap.add_argument("--trial-vox", type=float, default=0.3, help="with --fast: an LM trial only when a step moves an edge point more than this (voxels)")
ap.add_argument("--profile", action="store_true", help="time the acquisition-cost loop's parts (device-synchronized)")
ap.add_argument("--cubic", action="store_true", help="with --acq-cost: the prediction sampled by cubic B-splines (8 fetches), its gradient exact")
ap.add_argument("--acq-cost", action="store_true", help="residual in acquisition space: the prediction moved into each volume's acquired grid and compared with the raw data there (no interpolation of noisy data)")
ap.add_argument("--save-synth", default=None, help="write the synthetic scan (5 b0s with rigid motion, b0 0 fixed, + the DW volumes) for eddy, then exit")
ap.add_argument("--slm", action="store_true", help="after each outer iteration, eddy coefficients projected onto [1, g] (eddy's second-level model, linear)")
ap.add_argument("--final-fwhm", type=float, default=0.0, help="smoothing (voxels) kept at the full-resolution levels")
args = ap.parse_args()
dev = torch.device(args.device)
dt = torch.float32
torch.set_num_threads(8)
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
sync = (lambda: torch.mps.synchronize()) if dev.type == "mps" else (lambda: None)

# ---------------------------------------------------------------- data -> truth
D = str(DATA / "ds004712/sub-16974/ses-OXF1PRI001/dwi/sub-16974_ses-OXF1PRI001_dir-AP_dwi")
img = nib.load(D + ".nii.gz")
data = np.asarray(img.dataobj, dtype=np.float32)
bval = np.loadtxt(D + ".bval"); bvec = np.loadtxt(D + ".bvec")
vox = np.array(img.header.get_zooms()[:3], float)
shell = np.flatnonzero(np.abs(bval - 2000) < 100)
b0 = data[..., bval < 50].mean(-1)
from tractline.mask import median_otsu
_, mask = median_otsu(b0, median_radius=4, numpass=4)
mask = np.asarray(mask, bool)
from scipy.ndimage import binary_dilation, binary_erosion
emask = binary_dilation(mask, iterations=2)
edge = mask & ~binary_erosion(mask, iterations=1)
X, Y, Z = mask.shape
PE = 1                                                                                   # j
g = bvec[:, shell].T; g = g / np.linalg.norm(g, axis=1, keepdims=True)
G = len(shell)
from dipy.reconst.shm import real_sh_descoteaux
theta, phi = np.arccos(np.clip(g[:, 2], -1, 1)), np.arctan2(g[:, 1], g[:, 0])
B, m_, l_ = real_sh_descoteaux(6, theta, phi, legacy=False)
lam = 0.006
L = np.diag((l_ * (l_ + 1)) ** 2)
Hhat = B @ np.linalg.solve(B.T @ B + lam * L, B.T)                                       # (G, G) hat matrix
dH = np.diag(Hhat)
W_loo = (Hhat - np.diag(dH)) / (1 - dH)[:, None]                                         # row v: LOO prediction of v
W_all = Hhat
S = data[..., shell].reshape(-1, G)
truth = (S @ Hhat.T).reshape(X, Y, Z, G)                                                 # the SH fit: the truth
sigma = float(np.sqrt(((S - S @ Hhat.T)[mask.reshape(-1)] ** 2).mean() / (1 - dH.mean())))
info = {"data": D.split("/")[-1], "grid": [X, Y, Z], "vox_mm": vox.tolist(), "volumes": int(G), "mask_voxels": int(mask.sum()),
        "estimate_voxels (mask dilated 2)": int(emask.sum()), "noise_sigma": round(sigma, 2),
        "mean_leverage": round(float(dH.mean()), 3), "eddy_order": args.order, "loo": not args.no_loo, "scale": args.scale}

# ---------------------------------------------------------------- the model's pieces
cen = torch.tensor([(X - 1) / 2, (Y - 1) / 2, (Z - 1) / 2], dtype=dt, device=dev)
half = cen.clone()
vox_t = torch.tensor(vox, dtype=dt, device=dev)
NMON = 4 if args.order == 1 else 10


def monomials(x):                                     # x (..., 3) voxels -> (..., NMON), coordinates scaled to [-1, 1]
    u = (x - cen) / half
    a, b, c = u.unbind(-1)
    cols = [torch.ones_like(a), a, b, c]
    if NMON == 10:
        cols += [a * a, b * b, c * c, a * b, a * c, b * c]
    return torch.stack(cols, -1)


def rot(w):                                           # w (G, 3) rad -> (G, 3, 3), Rz Ry Rx as susceptibility.rigid
    cx, cy, cz = torch.cos(w).unbind(-1); sx, sy, sz = torch.sin(w).unbind(-1)
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], -1).view(-1, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).view(-1, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], -1).view(-1, 3, 3)
    return Rz @ Ry @ Rx


def phi(theta, x):
    """The sampling map: reference points x (N, 3) voxels -> acquired-volume positions (G, N, 3) voxels.
    theta (G, 6 + NMON): translations (mm), rotations (rad), eddy coefficients (voxels along PE). Rigid in mm about
    the grid center; the eddy displacement evaluated at the reference point (first order, as section 1.1 notes)."""
    R = rot(theta[:, 3:6])
    xm = (x - cen) * vox_t
    if x.dim() == 2:
        p = (torch.einsum("gij,nj->gni", R, xm) + theta[:, None, :3]) / vox_t + cen
        e = theta[:, 6:] @ monomials(x).T                                                 # (G, N)
    else:                                                                                 # per-volume points (G, N, 3)
        p = (torch.einsum("gij,gnj->gni", R, xm) + theta[:, None, :3]) / vox_t + cen
        e = (monomials(x) * theta[:, None, 6:]).sum(-1)
    return p + e[..., None] * torch.eye(3, dtype=dt, device=dev)[PE]


def sample_lin(vols, p, grad=False):                  # vols (G, X, Y, Z), p (G, N, 3) -> (G, N) [, dS/dp (G, N, 3)]
    sizes = torch.tensor([X, Y, Z], dtype=dt, device=dev)
    if grad:
        p = p.detach().requires_grad_(True)
    with torch.set_grad_enabled(grad):
        s = F.grid_sample(vols[:, None], (2 * p / (sizes - 1) - 1)[..., [2, 1, 0]][:, None, None], mode="bilinear",
                          padding_mode="zeros", align_corners=True)[:, 0, 0, 0]
    if not grad:
        return s
    gp, = torch.autograd.grad(s.sum(), p)
    return s.detach(), gp


def prefilter(vols):
    for ax in range(3):
        n = vols.shape[1 + ax]
        M = torch.zeros(n, n, dtype=torch.float64)
        for i in range(n):
            M[i, i] = 4 / 6
            for j in (i - 1, i + 1):
                jj = -j if j < 0 else (2 * (n - 1) - j if j > n - 1 else j)
                M[i, jj] += 1 / 6
        Minv = torch.linalg.inv(M).to(vols.device, vols.dtype)
        vols = torch.movedim(torch.movedim(vols, 1 + ax, -1) @ Minv.T, -1, 1 + ax)
    return vols


def sample_cubic(coef, p):                            # 8 trilinear fetches (Sigg & Hadwiger 2005); checked exact in bench_fast.py
    sizes = torch.tensor([X, Y, Z], dtype=dt, device=dev)
    i0 = torch.floor(p); t = p - i0
    w0, w1, w2, w3 = (1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6
    g0, g1 = w0 + w1, w2 + w3
    h = (i0 - 1 + w1 / g0, i0 + 1 + w3 / g1); gw = (g0, g1)
    out = 0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                q = torch.stack([h[a][..., 0], h[b][..., 1], h[c][..., 2]], -1)
                s = F.grid_sample(coef[:, None], (2 * q / (sizes - 1) - 1)[..., [2, 1, 0]][:, None, None], mode="bilinear",
                                  padding_mode="border", align_corners=True)[:, 0, 0, 0]
                out = out + gw[a][..., 0] * gw[b][..., 1] * gw[c][..., 2] * s
    return out


def smooth(vols, fwhm_vox):
    if fwhm_vox <= 0:
        return vols
    sd = fwhm_vox / 2.3548; r = int(math.ceil(3 * sd))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=dt, device=dev) / sd) ** 2); k = k / k.sum()
    out = vols[:, None]
    for ax in range(3):
        shape = [1, 1, 1, 1, 1]; shape[2 + ax] = len(k)
        pad = [0] * 6; pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = r
        out = F.conv3d(F.pad(out, pad, mode="replicate"), k.view(shape))
    return out[:, 0]


# ---------------------------------------------------------------- simulate
P = 6 + NMON
th_true = np.zeros((G, P))
th_true[:, :3] = rng.normal(0, 1.0, (G, 3)) * args.scale                                  # mm
th_true[:, 3:6] = np.deg2rad(rng.normal(0, 1.0, (G, 3))) * args.scale                     # rad
A = rng.normal(0, 1, (NMON, 3))                                                           # eddy coefficients linear in g
A[0] *= 0.5; A[1:4] *= 0.8                                                                # constant, linear terms: voxels at the edge
if NMON == 10:
    A[4:] *= 0.3
th_true[:, 6:] = (g @ A.T) * args.scale * args.eddy_scale
th_true_t = torch.tensor(th_true, dtype=dt, device=dev)
full = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=dt, device=dev) for n in (X, Y, Z)], indexing="ij"), -1).reshape(-1, 3)
truth_t = torch.tensor(np.moveaxis(truth, -1, 0).copy(), dtype=dt, device=dev)
with torch.no_grad():
    # acquired f_v(y) = truth_v(phi_v^-1(y)): invert by fixed point, x <- x - (phi(x) - y), from x = y
    coef = prefilter(truth_t)
    acq = torch.empty(G, X * Y * Z, dtype=dt, device=dev)
    for g0 in range(0, G, 10):
        th = th_true_t[g0:g0 + 10]
        x = full.expand(len(th), -1, -1).clone()
        for _ in range(6):
            x = x - (torch.stack([phi(th[k:k + 1], x[k])[0] for k in range(len(th))]) - full)
        acq[g0:g0 + 10] = sample_cubic(coef[g0:g0 + 10], x)
    acq = acq.reshape(G, X, Y, Z)
    acq = acq + sigma * torch.randn_like(acq)
if args.save_synth:
    import os
    os.makedirs(args.save_synth, exist_ok=True)
    b0i = np.flatnonzero(bval < 50)
    order = np.sort(np.concatenate([b0i, shell]))                                       # the scan's own volume order
    b0truth = torch.tensor(b0, dtype=dt, device=dev)[None]
    th_b0 = np.zeros((len(b0i), P)); th_b0[1:, :3] = rng.normal(0, 1.0, (len(b0i) - 1, 3)) * args.scale
    th_b0[1:, 3:6] = np.deg2rad(rng.normal(0, 1.0, (len(b0i) - 1, 3))) * args.scale
    th_b0_t = torch.tensor(th_b0, dtype=dt, device=dev)
    with torch.no_grad():
        cb = prefilter(b0truth)
        b0acq = []
        for k in range(len(b0i)):
            x = full.clone()
            for _ in range(6):
                x = x - (phi(th_b0_t[k:k + 1], x)[0] - full)
            b0acq.append(sample_cubic(cb, x[None])[0].reshape(X, Y, Z) + sigma * torch.randn(X, Y, Z, device=dev))
    vols = {int(i): b0acq[k].cpu().numpy() for k, i in enumerate(b0i)}
    vols.update({int(i): acq[k].cpu().numpy() for k, i in enumerate(shell)})
    imain = np.stack([np.clip(vols[int(i)], 0, None) for i in order], -1).astype(np.float32)
    aff = img.affine
    nib.save(nib.Nifti1Image(imain, aff), f"{args.save_synth}/imain.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), aff), f"{args.save_synth}/mask.nii.gz")
    bv = bval[order].copy(); bv[bv < 50] = 0
    bvec_o = bvec[:, order].copy(); bvec_o[:, bv == 0] = 0
    np.savetxt(f"{args.save_synth}/bvals", bv[None], fmt="%g")
    np.savetxt(f"{args.save_synth}/bvecs", bvec_o, fmt="%.6f")
    open(f"{args.save_synth}/acqp.txt", "w").write("0 -1 0 0.0690115\n")
    np.savetxt(f"{args.save_synth}/index.txt", np.ones((1, len(order)), int), fmt="%d")
    np.savez(f"{args.save_synth}/truth.npz", order=order, shell=shell, b0i=b0i, th_dw=th_true, th_b0=th_b0, pe_axis=PE,
             cen=cen.cpu().numpy(), vox=vox, nmon=NMON, edge=edge, sigma=sigma)
    print("wrote", args.save_synth, imain.shape, flush=True)
    sys.exit(0)
emask_t = torch.tensor(emask, device=dev); edge_t = torch.tensor(edge, device=dev)
xe = full[emask_t.reshape(-1)]                                                            # estimate's voxels (N, 3)
xedge = full[edge_t.reshape(-1)]
W = torch.tensor(W_all if args.no_loo else W_loo, dtype=dt, device=dev)


def edge_err(th):
    """Displacement error at the mask's edge (voxels), after removing the common transform (the gauge: the LOO
    prediction cannot see a transform shared by all volumes): per volume, |phi_est(x) - phi_true(x) - mean_v(...)|."""
    d = phi(th, xedge) - phi(th_true_t, xedge)
    d = d - d.mean(0, keepdim=True)
    n = d.norm(dim=-1)
    return float(n.pow(2).mean().sqrt()), float(torch.quantile(n.reshape(-1)[:2 ** 24], 0.95))


def disp_true():
    d = phi(th_true_t, xedge) - xedge
    d = d - d.mean(0, keepdim=True)
    n = d.norm(dim=-1)
    return float(n.pow(2).mean().sqrt()), float(torch.quantile(n.reshape(-1)[:2 ** 24], 0.95))


# ---------------------------------------------------------------- estimate
def jacobian_cols(theta, x, gp):
    """J (G, N, P) of the sampled values by theta: gp (G, N, 3) dS/dp; dp/dt = I / vox; dp/dw_j = (dR/dw_j xm) / vox;
    dp/de_k = m_k e_pe."""
    xm = (x - cen) * vox_t
    w = theta[:, 3:6].detach()
    cols = [gp / vox_t]                                                                   # translations: (G, N, 3)
    eps = 1e-4
    R0 = rot(w)
    rc = []
    for j in range(3):                                                                    # dR/dw_j by central differences (tiny)
        dw = torch.zeros_like(w); dw[:, j] = eps
        dR = (rot(w + dw) - rot(w - dw)) / (2 * eps)
        rc.append(((torch.einsum("gij,nj->gni" if xm.dim() == 2 else "gij,gnj->gni", dR, xm) / vox_t) * gp).sum(-1))
    mon = monomials(x)
    J = torch.cat([cols[0], torch.stack(rc, -1), gp[..., PE:PE + 1] * (mon[None] if mon.dim() == 2 else mon)], -1)
    return J


timings = {}
sync(); t_start = time.perf_counter()
th = torch.zeros(G, P, dtype=dt, device=dev)
mu = torch.full((G,), 1e-3, dtype=dt, device=dev)
ff = args.final_fwhm
fwhms = [2.0, max(1.0, ff), ff, ff, ff][:args.outer] + [ff] * max(0, args.outer - 5)
history = []
scale = torch.cat([torch.ones(3), torch.full((3,), 0.01), torch.ones(NMON)]).to(dev, dt)  # mm, rad, voxels
emask4 = binary_dilation(mask, iterations=4)
xp = full[torch.tensor(emask4, device=dev).reshape(-1)]                                   # where predictions are built (wider)
lin_e = torch.tensor(np.flatnonzero(emask.reshape(-1)), device=dev)
lin_p = torch.tensor(np.flatnonzero(emask4.reshape(-1)), device=dev)


def phi_inv(theta, y, n=4):                                                               # x with phi(x) = y, fixed point
    x = y.expand(theta.shape[0], -1, -1).clone()
    for _ in range(n):
        x = x - (phi(theta, x) - y)
    return x


prof, _t = {}, [None]


def tick(name=None):                                                                      # --profile: time since the last tick, charged to name
    if not args.profile:
        return
    sync(); now = time.perf_counter()
    if name is not None and _t[0] is not None:
        prof[name] = prof.get(name, 0.0) + now - _t[0]
    _t[0] = now


ntrials = 0

# ---------------------------------------------------------------- the fast path (--fast): the final form with the profile's fixes
E_PE = torch.eye(3, dtype=dt, device=dev)[PE]


def eddy_poly(te, x):
    """The eddy displacement (voxels along PE) of coefficients te (G, NMON) at points x ((N, 3) shared or (G, N, 3)),
    written out: no (G, N, NMON) monomial array."""
    u = (x - cen) / half
    a, b, c = u.unbind(-1)
    k = lambda i: te[:, i, None]                                                          # (G, 1): broadcasts over (N,) or (G, N)
    e = k(0) + k(1) * a + k(2) * b + k(3) * c
    if NMON == 10:
        e = e + k(4) * a * a + k(5) * b * b + k(6) * c * c + k(7) * a * b + k(8) * a * c + k(9) * b * c
    return e                                                                              # (G, N)


def phi_fast(theta, x):
    R = rot(theta[:, 3:6])
    xm = (x - cen) * vox_t
    p = (torch.einsum("gij,nj->gni" if x.dim() == 2 else "gij,gnj->gni", R, xm) + theta[:, None, :3]) / vox_t + cen
    return p + eddy_poly(theta[:, 6:], x)[..., None] * E_PE


def phi_inv_fast(theta, y, n=2):
    """x with phi(x) = y: the rigid part inverted in closed form, x = c + R^T((y - e(x) e_pe - c) vox - t) / vox, with the
    eddy displacement e taken at the previous x (from e(y)); n passes."""
    R = rot(theta[:, 3:6])
    te, t = theta[:, 6:], theta[:, None, :3]
    e = eddy_poly(te, y)
    for _ in range(n):
        q = (y - cen) * vox_t - e[..., None] * (E_PE * vox_t) - t                         # (G, N, 3) mm
        x = torch.einsum("gji,gnj->gni", R, q) / vox_t + cen                              # R^T q
        e = eddy_poly(te, x)
    return x


def sample_vec(vols, p):                                                                  # vols (G, C, X, Y, Z), p (G, N, 3) -> (G, N, C) trilinear
    sizes = torch.tensor([X, Y, Z], dtype=dt, device=dev)
    s = F.grid_sample(vols, (2 * p / (sizes - 1) - 1)[..., [2, 1, 0]][:, None, None], mode="bilinear",
                      padding_mode="border", align_corners=True)                         # (G, C, 1, 1, N)
    return s[:, :, 0, 0].transpose(1, 2)


def cubic_value_grad(coef, p):
    """The cubic B-spline of coef (G, X, Y, Z) at p (G, N, 3) and its exact gradient (voxels): one pass over the 64 taps,
    each coefficient gathered once and accumulated into the value and the three derivatives (no autograd). Edges clamped."""
    Gn = coef.shape[0]
    i0 = torch.floor(p); t = p - i0; i0 = i0.long()
    w = [(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t ** 2 + 4) / 6, (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6, t ** 3 / 6]
    dw = [-(1 - t) ** 2 / 2, (3 * t ** 2 - 4 * t) / 2, (-3 * t ** 2 + 2 * t + 1) / 2, t ** 2 / 2]
    flat = coef.reshape(Gn, -1)
    val = torch.zeros(p.shape[:2], dtype=coef.dtype, device=coef.device)
    gx, gy, gz = torch.zeros_like(val), torch.zeros_like(val), torch.zeros_like(val)
    for a in range(4):
        ia = (i0[..., 0] + a - 1).clamp(0, X - 1)
        for b in range(4):
            ib = (i0[..., 1] + b - 1).clamp(0, Y - 1)
            wab, dab_x, dab_y = w[a][..., 0] * w[b][..., 1], dw[a][..., 0] * w[b][..., 1], w[a][..., 0] * dw[b][..., 1]
            base = (ia * Y + ib) * Z
            for c in range(4):
                ic = (i0[..., 2] + c - 1).clamp(0, Z - 1)
                v = torch.gather(flat, 1, base + ic)
                wc, dc = w[c][..., 2], dw[c][..., 2]
                val = val + wab * wc * v
                gx = gx + dab_x * wc * v
                gy = gy + dab_y * wc * v
                gz = gz + wab * dc * v
    return val, torch.stack([gx, gy, gz], -1)


if args.selftest:
    with torch.no_grad():
        tt = th_true_t.clone()
        y = xe[:5000]
        e1 = eddy_poly(tt[:, 6:], y); e2 = tt[:, 6:] @ monomials(y).T
        print("eddy_poly vs monomials:", float((e1 - e2).abs().max()))
        print("phi_fast vs phi:", float((phi_fast(tt, y) - phi(tt, y)).abs().max()))
        xi = phi_inv_fast(tt, y); xo = phi_inv(tt, y)
        print("phi(phi_inv_fast(y)) - y:", float((phi(tt, xi) - y).abs().max()), " vs phi_inv:", float((phi(tt, xo) - y).abs().max()))
        pv_ = truth_t[:, :, :, :]
        gv_ = torch.stack(torch.gradient(pv_, dim=(1, 2, 3)), 1)
        co_ = prefilter(pv_)
    xq = phi_inv(tt, xe[:20000]).detach().requires_grad_(True)
    v_ = sample_cubic(co_, xq); ga, = torch.autograd.grad(v_.sum(), xq)
    gb = sample_vec(gv_, xq.detach())
    vv, gc = cubic_value_grad(co_, xq.detach())
    print("64-tap value vs 8-fetch:", float((vv - v_.detach()).abs().max() / v_.detach().abs().max()),
          " gradient vs autograd:", float((gc - ga).abs().max() / ga.abs().max()))
    cc = [float(torch.corrcoef(torch.stack([ga[..., i].reshape(-1), gb[..., i].reshape(-1)]))[0, 1]) for i in range(3)]
    print("gradient: autograd cubic vs sampled gradient volume, corr per axis:", [round(c, 4) for c in cc],
          "ratio of norms:", round(float(gb.norm() / ga.norm()), 4))
    sys.exit(0)
if args.fast:
    fw_steps = [2.0, 1.0] + [0.0] * (args.outer - 2)                                      # the 15 x 1 schedule's smoothing
    coarse = args.coarse                                                                  # the first steps on 1/8 of the points
    sub = torch.tensor(np.flatnonzero((np.argwhere(emask) % 2 == 0).all(1)), device=dev)  # every other voxel per axis
    ye_full = xe
    mon_full = monomials(xe)                                                              # (N, NMON), the Jacobian's (at y, not x)
    ym_full = (xe - cen) * vox_t
    rng_t = torch.Generator(device="cpu").manual_seed(1)
    probe = xedge[torch.randperm(len(xedge), generator=rng_t)[:2000].to(dev)]            # edge points for the step size
    smooth_cache = {}
    for it, fw in enumerate(fw_steps):
        tick()
        if fw not in smooth_cache:
            smooth_cache[fw] = smooth(acq, fw)
        vols = smooth_cache[fw]
        obs_full = vols.reshape(G, -1)[:, lin_e]                                          # the raw data at the grid: never interpolated
        tick("smooth")
        with torch.no_grad():
            u = sample_lin(vols, phi_fast(th, xp))
            pv = torch.zeros(G, X * Y * Z, dtype=dt, device=dev)
            pv[:, lin_p] = W @ u
            pv = pv.reshape(G, X, Y, Z)
            coef = prefilter(pv)
        tick("predict (resample, matmul, prefilter)")
        c_ = it < coarse
        ys, obs = (ye_full[sub], obs_full[:, sub]) if c_ else (ye_full, obs_full)
        mon, ym = (mon_full[sub], ym_full[sub]) if c_ else (mon_full, ym_full)
        with torch.no_grad():
            x = phi_inv_fast(th, ys)
            tick("phi_inv (step)")
            if args.grad == "taps":
                Pm, gp = cubic_value_grad(coef, x)                                        # value and exact gradient, one 64-tap pass
            else:
                with torch.enable_grad():
                    xq = x.detach().requires_grad_(True)
                    Pm = sample_cubic(coef, xq)                                           # 8 fetches; gradient exact to ~1e-4
                    gp, = torch.autograd.grad(Pm.sum(), xq)
                Pm = Pm.detach()
            tick("sample + gradient")
            r = Pm - obs
            w = th[:, 3:6]
            rc = []
            for j in range(3):
                dw = torch.zeros_like(w); dw[:, j] = 1e-4
                dR = (rot(w + dw) - rot(w - dw)) / 2e-4
                rc.append(((torch.einsum("gij,nj->gni", dR, ym) / vox_t) * gp).sum(-1))
            J = -torch.cat([gp / vox_t, torch.stack(rc, -1), gp[..., PE:PE + 1] * mon[None]], -1) * scale
            H = torch.bmm(J.transpose(1, 2), J)
            b = torch.bmm(J.transpose(1, 2), r[..., None])[..., 0]
            f0 = (r ** 2).sum(1)
            Hd = torch.diagonal(H, dim1=1, dim2=2)
            tick("jacobian + H, b")
            accept = torch.zeros(G, dtype=torch.bool, device=dev)
            th_new = th.clone()
            for _ls in range(6):
                todo = ~accept
                if not bool(todo.any()):
                    break
                A_ = H + torch.diag_embed(mu[:, None] * Hd)
                step = -torch.linalg.solve(A_.cpu(), b.cpu()[..., None])[..., 0].to(dev) * scale
                cand = th + step
                big = (phi_fast(cand, probe) - phi_fast(th, probe)).norm(dim=-1).amax(1) > args.trial_vox
                need = todo & (big | (it < 2))                                            # a trial only for large steps (and the first two)
                free = todo & ~need
                th_new[free] = cand[free]; accept |= free
                mu = torch.where(free, (mu / 3).clamp_min(1e-7), mu)
                if bool(need.any()):
                    k = torch.nonzero(need)[:, 0]
                    f1 = ((sample_cubic(coef[k], phi_inv_fast(cand[k], ys)) - obs[k]) ** 2).sum(1)
                    ntrials += 1
                    ok = torch.zeros(G, dtype=torch.bool, device=dev); ok[k] = f1 < f0[k]
                    th_new[ok] = cand[ok]; accept |= ok
                    bad = need & ~ok
                    mu = torch.where(ok, (mu / 3).clamp_min(1e-7), torch.where(bad, mu * 8, mu))
                    if not bool(bad.any()):
                        break
                else:
                    break
            th = th_new
            tick("LM trials (solve, inverse, sample)")
        if args.profile or it == len(fw_steps) - 1 or it % 5 == 0:
            history.append({"step": it, "fwhm": fw, "coarse": bool(c_), "accepted": int(accept.sum()),
                            "edge_err_rms_p95": [round(v, 3) for v in edge_err(th)]})
            print(history[-1], flush=True)
        tick("diagnostics (edge error)")
    fwhms = []
    args.acq_cost = False                                                                 # skip the other paths below
if args.acq_cost:
    for it, fw in enumerate(fwhms):
        tick()
        vols = smooth(acq, fw)
        tick("smooth")
        obs = vols.reshape(G, -1)[:, lin_e]                                               # raw (smoothed) data at the grid: no interpolation
        with torch.no_grad():
            u = sample_lin(vols, phi(th, xp))                                             # corrected, on the wider mask
            pv = torch.zeros(G, X * Y * Z, dtype=dt, device=dev)
            pv[:, lin_p] = W @ u
            pv = pv.reshape(G, X, Y, Z)                                                   # LOO predictions as volumes, reference space
            if args.cubic:
                pv = prefilter(pv)
        tick("predict (resample, matmul, prefilter)")
        for _ in range(args.inner):
            with torch.no_grad():
                x = phi_inv(th, xe)
            tick("phi_inv (step)")
            if args.cubic:
                xq = x.detach().requires_grad_(True)
                Pm = sample_cubic(pv, xq)
                gp, = torch.autograd.grad(Pm.sum(), xq); Pm = Pm.detach()
            else:
                Pm, gp = sample_lin(pv, x, grad=True)                                     # prediction at phi^-1(y)
            tick("sample + gradient")
            r = Pm - obs
            J = -jacobian_cols(th, x, gp) * scale                                         # d/dtheta P(phi^-1(y)) ~ -grad P . dphi/dtheta
            H = torch.einsum("gnp,gnq->gpq", J, J)
            b = torch.einsum("gnp,gn->gp", J, r)
            f0 = (r ** 2).sum(1)
            Hd = torch.diagonal(H, dim1=1, dim2=2)
            tick("jacobian + H, b")
            accept = torch.zeros(G, dtype=torch.bool, device=dev)
            th_new = th.clone()
            for _ls in range(6):
                todo = ~accept
                if not bool(todo.any()):
                    break
                A_ = H + torch.diag_embed(mu[:, None] * Hd)
                step = -torch.linalg.solve(A_.cpu(), b.cpu()[..., None])[..., 0].to(dev) * scale
                cand = th + step
                with torch.no_grad():
                    f1 = (((sample_cubic(pv, phi_inv(cand, xe)) if args.cubic else sample_lin(pv, phi_inv(cand, xe))) - obs) ** 2).sum(1)
                ntrials += 1
                ok = (f1 < f0) & todo
                th_new[ok] = cand[ok]
                accept |= ok
                mu = torch.where(ok, (mu / 3).clamp_min(1e-7), torch.where(todo, mu * 8, mu))
            th = th_new
            tick("LM trials (solve, inverse, sample)")
            history.append({"outer": it, "fwhm": fw, "accepted": int(accept.sum()), "edge_err_rms_p95": [round(v, 3) for v in edge_err(th)]})
            print(history[-1], flush=True)
            tick("diagnostics (edge error)")
    fwhms = []                                                                            # skip the reference-space loop below
for it, fw in enumerate(fwhms):
    vols = smooth(acq, fw)
    with torch.no_grad():
        u = sample_lin(vols, phi(th, xe))                                                 # (G, N) corrected, current params
        pred = W @ u                                                                      # LOO predictions (G, N)
    for _ in range(args.inner):
        S, gp = sample_lin(vols, phi(th, xe), grad=True)
        r = S - pred
        J = jacobian_cols(th, xe, gp) * scale                                             # scaled parameters
        H = torch.einsum("gnp,gnq->gpq", J, J)
        b = torch.einsum("gnp,gn->gp", J, r)
        f0 = (r ** 2).sum(1)
        Hd = torch.diagonal(H, dim1=1, dim2=2)
        accept = torch.zeros(G, dtype=torch.bool, device=dev)
        th_new = th.clone()
        for _ls in range(6):                                                              # per-volume Marquardt
            todo = ~accept
            if not bool(todo.any()):
                break
            A_ = H + torch.diag_embed(mu[:, None] * Hd)
            step = -torch.linalg.solve(A_.cpu(), b.cpu()[..., None])[..., 0].to(dev) * scale
            cand = th + step
            with torch.no_grad():
                f1 = ((sample_lin(vols, phi(cand, xe)) - pred) ** 2).sum(1)
            ok = (f1 < f0) & todo
            th_new[ok] = cand[ok]
            accept |= ok
            mu = torch.where(ok, (mu / 3).clamp_min(1e-7), torch.where(todo, mu * 8, mu))
        th = th_new
        if args.slm:
            Xg = torch.cat([torch.ones(G, 1, dtype=dt, device=dev), torch.tensor(g, dtype=dt, device=dev)], 1)   # (G, 4)
            coefs = torch.linalg.lstsq(Xg.cpu(), th[:, 6:].cpu()).solution.to(dev)
            th = torch.cat([th[:, :6], Xg @ coefs], 1)
        history.append({"outer": it, "fwhm": fw, "accepted": int(accept.sum()), "edge_err_rms_p95": [round(v, 3) for v in edge_err(th)]})
        print(history[-1], flush=True)
sync(); timings["estimate_s"] = round(time.perf_counter() - t_start, 2)
def split_err(th):
    tr = th.clone(); tr[:, 6:] = th_true_t[:, 6:]                                         # estimated rigid, true eddy
    te = th.clone(); te[:, :6] = th_true_t[:, :6]                                         # true rigid, estimated eddy
    dw = th[:, 3:6] - th_true_t[:, 3:6]; dw = dw - dw.mean(0, keepdim=True)
    return {"rigid_part": [round(v, 3) for v in edge_err(tr)], "eddy_part": [round(v, 3) for v in edge_err(te)],
            "rotation_error_rms_deg": round(float(torch.rad2deg(dw).norm(dim=1).pow(2).mean().sqrt()), 3)}


gz = g.copy()
hemi = {"directions": int(G), "max |mean g|": round(float(np.linalg.norm(g.mean(0))), 3),
        "fraction with g_z > 0": round(float((g[:, 2] > 0).mean()), 2)}
info.update({"error_split_rms_p95_vox": split_err(th), "gradient_set": hemi})
info.update({"true_edge_displacement_rms_p95_vox": [round(v, 3) for v in disp_true()],
             "final_edge_error_rms_p95_vox": [round(v, 3) for v in edge_err(th)],
             "gn_iterations": args.outer * args.inner, **timings, "history": history,
             **({"profile_s": {k: round(v, 2) for k, v in prof.items()}, "lm_trials": ntrials} if args.profile else {}),
             "device": args.device})
tag = f"_{args.device}_o{args.order}" + ("_noloo" if args.no_loo else "") + (f"_s{args.scale}" if args.scale != 1 else "") + (f"_e{args.eddy_scale}" if args.eddy_scale != 1 else "") + f"_n{args.outer}" + (f"_f{args.final_fwhm}" if args.final_fwhm else "") + ("_slm" if args.slm else "") + ("_acq" if args.acq_cost else "") + ("_cubic" if args.cubic else "") + (f"_fast_c{args.coarse}_{args.grad}" if args.fast else "") + ("_prof" if args.profile else "")
json.dump(info, open(OUT / f"proto{tag}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "history"}))
