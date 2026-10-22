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
ap.add_argument("--anchor", action="store_true", help="with --all-shells --fast --metal: after the fit, each DW shell's corrected mean registered rigidly to the corrected mean b0 (normalized gradient fields), the shift added to the shell's volumes (decision 5)")
ap.add_argument("--anchor-selftest", action="store_true")
ap.add_argument("--anchor-method", default="ngf", choices=["ngf", "mi", "gpumi"], help="with --anchor: Gauss-Newton NGF on the GPU, dipy's rigid mutual information on the CPU (eddy's PEAS uses MI), or mi_torch's MI on the device")
ap.add_argument("--anchor-truthtest", action="store_true", help="a mechanical check: each group's corrected mean registered (NGF, same contrast) to the truth's mean of that group in the estimate's frame; found against needed, printed")
ap.add_argument("--anchor-eddy", action="store_true", help="with --anchor-method gpumi: the anchor also fits each shell's common eddy polynomial along PE (16 parameters, as the step's), not only its rigid position")
ap.add_argument("--anchor-flatten", type=float, default=0.0, help="with --anchor: divide each mean by a Gaussian-smoothed copy (sigma voxels) within the head before the registration: removes the slowly varying DW/b0 intensity relation that biases cross-contrast MI along z")
ap.add_argument("--anchor-no-pe", action="store_true", help="with --anchor: do not apply the anchor's translation along PE (eddy: its PE translations are calculated but not used)")
ap.add_argument("--anchor-bins", type=int, default=256, help="with --anchor-method gpumi: the MI histogram's bins (eddy's PEAS: 256; 32 biased the real scan's z by 0.3 mm)")
ap.add_argument("--gain", action="store_true", help="with --fast --metal: a per-volume intensity gain, each step: the observations divided by (their sum / the prediction's sum) over the estimate's voxels (signal drift; SSD otherwise trades it for position)")
ap.add_argument("--offset-in-solve", action="store_true", help="the earlier step: the eddy constant solved with the PE translation (an exact null direction; the pair walks)")
ap.add_argument("--no-ec-gauge", action="store_true", help="the earlier step: each shell's common eddy field left free (its leave-one-out predictor cannot see it; it drifts with the steps)")
ap.add_argument("--no-ref", action="store_true", help="the earlier step: no location reference (the scan's overall position unpinned; it drifts under large motion)")
ap.add_argument("--shell-pose-free", action="store_true", help="the earlier step: each DW shell's mean pose left free during the fit (its predictor cannot see it; it drifts slowly with the steps)")
ap.add_argument("--b0-ec", action="store_true", help="the b0s fit the PE polynomial too (not the constant), their mean shape held at 0: for motion-by-susceptibility, the distortion changing as the head moves in the field (TORTOISE fits its b0s with its full model)")
ap.add_argument("--ref-pin", action="store_true", help="the location reference as eddy's: the first b0 held fixed (default: the b0s' mean rigid parameters held at 0)")
ap.add_argument("--b0-bug", action="store_true", help="reproduce the earlier b0 step: the 16-parameter system solved, then the eddy part dropped")
ap.add_argument("--mask-output", action="store_true", help="with --metal: the final resample zeroes voxels not inside the field of view for every volume, as eddy's output mask")
ap.add_argument("--anchor-iters", type=int, default=30, help="with --anchor: Gauss-Newton iterations")
ap.add_argument("--anchor-every", type=int, default=0, help="with --anchor: also every this many steps (0: only at the end)")
ap.add_argument("--shells", default=None, help="with --all-shells: only these b-values, e.g. 0,2000 (b0s as 0)")
ap.add_argument("--tensor-truth", action="store_true", help="with --all-shells: the truth from one tensor model, S0 exp(-b g'Dg), S0 the real b0 mean and D a log-linear fit to the real data: every shell aligned with the b0s by construction (the real scan's own between-shell offsets, about 0.3 mm, otherwise sit in the truth)")
ap.add_argument("--real", default=None, help="with --all-shells --fast --metal: fit the scan as acquired (no simulation, no truth) and write its parameters, corrected volumes and rotated b-vectors to this folder (compare_real.py compares them with eddy's)")
ap.add_argument("--all-shells", action="store_true", help="every volume of the scan (5 b0s, b1000, b2000: 105 on ON-Harmony), each shell its own LOO-SH predictor, the b0s the mean of the others; eddy scaled by sqrt(b / 2000) and zero for b0s")
ap.add_argument("--tol", type=float, default=0.0, help="with --fast: stop when no volume's step moves an edge point more than this (voxels) two steps running; --outer is then the cap")
ap.add_argument("--smooth-steps", type=int, default=2, help="with --fast: the first steps smoothed (FWHM 2, then 1 voxel)")
ap.add_argument("--metal", action="store_true", help="with --fast on mps: the inverse, cubic value and gradient, Jacobian and H, b, cost in the fused Metal kernels (metal_step.py)")
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
ap.add_argument("--pred-rot", action="store_true", help="rebuild the predictor each step from the gradient directions rotated by the current motion (no gain on the SHORELine phantoms, worse on DW-POSSUM large motion; off by default)")
ap.add_argument("--pe", type=int, default=None, help="the PE axis (0, 1, 2), over the sidecar and the default (1)")
ap.add_argument("--dwi", default=None, help="any DWI instead of --scan: the path without .nii.gz (with .bval, .bvec, and .json for the PE axis)")
ap.add_argument("--scan", default="16974/OXF1PRI001", help="ON-Harmony subject/session of the dir-AP scan to use (synthetic truth or, with --real, the scan fitted)")
ap.add_argument("--final-fwhm", type=float, default=0.0, help="smoothing (voxels) kept at the full-resolution levels")
args = ap.parse_args()
dev = torch.device(args.device)
dt = torch.float32
torch.set_num_threads(8)
torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
sync = (lambda: torch.mps.synchronize()) if dev.type == "mps" else (lambda: None)

# ---------------------------------------------------------------- data -> truth
SUB, SES = args.scan.split("/")
D = str(DATA / f"ds004712/sub-{SUB}/ses-{SES}/dwi/sub-{SUB}_ses-{SES}_dir-AP_dwi")
if args.dwi:
    D = args.dwi[:-7] if args.dwi.endswith(".nii.gz") else args.dwi
    SUB = Path(D).name
img = nib.load(D + ".nii.gz")
data = np.asarray(img.dataobj, dtype=np.float32)
bval = np.loadtxt(D + ".bval"); bvec = np.loadtxt(D + ".bvec")
vox = np.array(img.header.get_zooms()[:3], float)
shell = np.arange(len(bval)) if args.all_shells else np.flatnonzero(np.abs(bval - 2000) < 100)
if args.all_shells and args.shells:
    want = [float(v) for v in args.shells.split(",")]
    shell = np.array([i for i in shell if any((bval[i] < 50) if w == 0 else abs(bval[i] - w) < 100 for w in want)])
b0 = data[..., bval < 50].mean(-1)
from tractline.mask import median_otsu
_, mask = median_otsu(b0, median_radius=4, numpass=4)
mask = np.asarray(mask, bool)
from scipy.ndimage import binary_dilation, binary_erosion
emask = binary_dilation(mask, iterations=2)
edge = mask & ~binary_erosion(mask, iterations=1)
X, Y, Z = mask.shape
PE = 1                                                                                   # j
if args.dwi and Path(D + ".json").exists():                                              # the PE axis from the sidecar
    PE = "ijk".index(json.load(open(D + ".json")).get("PhaseEncodingDirection", "j")[0])
if args.pe is not None:
    PE = args.pe
G = len(shell)
from dipy.reconst.shm import real_sh_descoteaux
bsel = bval[shell]
isb0 = bsel < 50
g = np.zeros((G, 3))
g[~isb0] = (bvec[:, shell].T / np.linalg.norm(bvec[:, shell].T, axis=1, keepdims=True).clip(1e-9))[~isb0]
lam = 0.006
Hhat = np.zeros((G, G))
groups = [np.flatnonzero(isb0)] if isb0.any() else []
for b in sorted(set(np.round(bsel[~isb0] / 100) * 100)):
    groups.append(np.flatnonzero((~isb0) & (np.abs(bsel - b) < 100)))


def build_hat(gd):
    """The per-shell regularized SH smoother (G, G) for gradient directions gd (G, 3); the b0s' block is their mean."""
    H = np.zeros((G, G))
    for idx in groups:                                                                    # block-diagonal: one predictor per shell
        if isb0[idx[0]]:
            H[np.ix_(idx, idx)] = 1.0 / len(idx)                                          # b0s: their mean
            continue
        gg = gd[idx]
        theta, phi = np.arccos(np.clip(gg[:, 2], -1, 1)), np.arctan2(gg[:, 1], gg[:, 0])
        B, m_, l_ = real_sh_descoteaux(6, theta, phi, legacy=False)
        L = np.diag((l_ * (l_ + 1)) ** 2)
        H[np.ix_(idx, idx)] = B @ np.linalg.solve(B.T @ B + lam * L, B.T)
    return H


def build_loo(gd):
    H = build_hat(gd); dh = np.diag(H)
    return (H - np.diag(dh)) / (1 - dh)[:, None]                                         # row v: LOO prediction of v


Hhat = build_hat(g)
dH = np.diag(Hhat)
W_loo = build_loo(g)
W_all = Hhat
S = data[..., shell].reshape(-1, G)
truth = (S @ Hhat.T).reshape(X, Y, Z, G)                                                 # the SH fit: the truth
if args.tensor_truth:
    # one model for all volumes: log S = log S0 - b g'Dg, fitted per voxel to the real data (b0s and DW), then the truth
    # rebuilt from the b0 mean S0 itself, so every volume's edges are S0's: the shells aligned with the b0s exactly
    bs = np.where(isb0, 0.0, bsel) / 1000.0
    Bm = np.stack([np.ones(G), -bs * g[:, 0] ** 2, -bs * g[:, 1] ** 2, -bs * g[:, 2] ** 2,
                   -2 * bs * g[:, 0] * g[:, 1], -2 * bs * g[:, 0] * g[:, 2], -2 * bs * g[:, 1] * g[:, 2]], 1)
    logS = np.log(np.clip(S, 1.0, None))
    coef = logS @ np.linalg.pinv(Bm).T                                                    # (voxels, 7): log S0, D (um^2/ms)
    Dtt = coef[:, 1:].copy()
    Dtt[~mask.reshape(-1)] = 0                                                            # outside the brain: no attenuation model
    S0 = data[..., bval < 50].mean(-1).reshape(-1)
    att = np.exp(-(Dtt @ Bm[:, 1:].T))                                                   # (voxels, G)
    truth = (S0[:, None] * np.clip(att, 0, 1)).reshape(X, Y, Z, G).astype(np.float32)
dwv = ~isb0                                                                               # the noise from the DW fits only
sigma = float(np.sqrt(((S - S @ Hhat.T)[mask.reshape(-1)][:, dwv] ** 2).mean() / (1 - dH[dwv].mean())))
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
th_true[:, 6:] = ((g * np.sqrt(np.clip(bsel, 0, None) / 2000.0)[:, None]) @ A.T) * args.scale * args.eddy_scale   # eddy ~ gradient amplitude
th_true_t = torch.tensor(th_true, dtype=dt, device=dev)
full = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=dt, device=dev) for n in (X, Y, Z)], indexing="ij"), -1).reshape(-1, 3)
truth_t = torch.tensor(np.moveaxis(truth, -1, 0).copy(), dtype=dt, device=dev)
if args.real:
    th_true[:] = 0; th_true_t.zero_()                                                    # no truth: the scores against it are not read
with torch.no_grad():
  if args.real:
    acq = torch.tensor(np.moveaxis(data[..., shell], -1, 0).copy(), dtype=dt, device=dev)   # the scan as acquired
  else:
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
if args.save_synth and args.all_shells:
    # the full scan as simulated (every volume, its motion and eddy), for eddy; scored by compare_eddy_all.py
    import os
    os.makedirs(args.save_synth, exist_ok=True)
    imain = np.moveaxis(acq.clamp_min(0).cpu().numpy(), 0, -1).astype(np.float32)
    nib.save(nib.Nifti1Image(imain, img.affine), f"{args.save_synth}/imain.nii.gz")
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), img.affine), f"{args.save_synth}/mask.nii.gz")
    bv = bsel.copy(); bv[isb0] = 0
    bvec_o = bvec[:, shell].copy(); bvec_o[:, isb0] = 0
    np.savetxt(f"{args.save_synth}/bvals", bv[None], fmt="%g")
    np.savetxt(f"{args.save_synth}/bvecs", bvec_o, fmt="%.6f")
    open(f"{args.save_synth}/acqp.txt", "w").write("0 -1 0 0.0690115\n")
    np.savetxt(f"{args.save_synth}/index.txt", np.ones((1, G), int), fmt="%d")
    np.savez(f"{args.save_synth}/truth.npz", th=th_true, groups=np.array([np.isin(np.arange(G), gi).astype(int) * (k + 1) for k, gi in enumerate(groups)]).sum(0),
             pe_axis=PE, cen=cen.cpu().numpy(), vox=vox, nmon=NMON, edge=edge, sigma=sigma)
    print("wrote", args.save_synth, imain.shape, flush=True)
    sys.exit(0)
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
def ngf_cost(a, b, mk, eps):
    """1 - (n_a . n_b)^2 averaged over mk, n the normalized gradient field |grad| -> grad / sqrt(|grad|^2 + eps^2)."""
    def n(v):
        gx, gy, gz = torch.gradient(v, dim=(0, 1, 2))
        nn = torch.sqrt(gx * gx + gy * gy + gz * gz + eps * eps)
        return gx / nn, gy / nn, gz / nn
    ax_, ay_, az_ = n(a); bx_, by_, bz_ = n(b)
    d = ax_ * bx_ + ay_ * by_ + az_ * bz_
    return (1 - d * d)[mk].mean()



def ngf_field(v, eps=1.0):
    """The normalized gradient field of v (X, Y, Z): grad / sqrt(|grad|^2 + eps^2), (3, X, Y, Z)."""
    g_ = torch.stack(torch.gradient(v, dim=(0, 1, 2)))
    return g_ / torch.sqrt((g_ * g_).sum(0, keepdim=True) + eps * eps)


def gn_ngf(mov, fixed, mk, iters=None):
    """The rigid d (3 mm, 3 x 0.01 rad, as phi's) with mov(phi_d(x)) aligned to fixed(x): Gauss-Newton on the residual
    n_mov(phi_d(x)) x n_fixed(x) (3 components; |a x b|^2 = |a|^2 |b|^2 - (a . b)^2, so it ignores the contrast's
    sign as NGF does), the Jacobian by central differences in d, Levenberg-Marquardt damping. Returns d (6,)."""
    iters = iters or args.anchor_iters
    nm, nf = ngf_field(mov), ngf_field(fixed)
    pts = full[mk.reshape(-1)]
    nf_m = nf.reshape(3, -1)[:, mk.reshape(-1)].T                                         # (M, 3)
    sizes = torch.tensor([X, Y, Z], dtype=dt, device=dev)

    def res(d):
        q = torch.cat([d[:3], d[3:] * 0.01, torch.zeros(NMON, dtype=dt, device=dev)])[None]
        pp = phi_fast(q, pts)[0]
        s_ = F.grid_sample(nm[None], (2 * pp / (sizes - 1) - 1)[..., [2, 1, 0]][None, None, None], mode="bilinear",
                           padding_mode="border", align_corners=True)[0, :, 0, 0].T     # (M, 3)
        return torch.linalg.cross(s_, nf_m).reshape(-1)
    d = torch.zeros(6, dtype=dt, device=dev)
    lam_ = 1e-3
    with torch.no_grad():
        r = res(d); f = float((r * r).sum())
        for _ in range(iters):
            J = torch.stack([(res(d + h) - res(d - h)) / (2 * 0.05) for h in torch.eye(6, dtype=dt, device=dev) * 0.05], 1)
            H_ = J.T @ J; b_ = J.T @ r
            while True:
                step = -torch.linalg.solve(H_ + lam_ * torch.diag(torch.diagonal(H_)), b_)
                r1 = res(d + step); f1 = float((r1 * r1).sum())
                if f1 < f:
                    d, r, f = d + step, r1, f1; lam_ = max(lam_ / 3, 1e-7); break
                lam_ *= 8
                if lam_ > 1e6:
                    break
            if float(step.abs().max()) < 1e-3 or lam_ > 1e6:
                break
    return d


def mi_rigid(mov, fixed, mk):
    """dipy's rigid mutual-information registration (mi_dipy.py, in a process of its own: dipy's and torch's OpenMP
    runtimes clash in one), mov onto fixed on the reference grid; in phi's parameterization: d (3 mm, 3 x 0.01 rad) with
    mov(phi_d(x)) ~ fixed(x)."""
    import subprocess, tempfile, os, json as _json
    with tempfile.TemporaryDirectory() as tdir:
        np.savez(os.path.join(tdir, "in.npz"), mov=mov.cpu().double().numpy(), fixed=fixed.cpu().double().numpy(),
                 mask=binary_dilation(mask, iterations=2), vox=np.asarray(vox), cen=np.asarray(cen.cpu()))
        subprocess.run([sys.executable, str(__import__("pathlib").Path(__file__).resolve().parent / "mi_dipy.py"), "register",
                        os.path.join(tdir, "in.npz"), os.path.join(tdir, "out.json")], check=True, capture_output=True)
        r = _json.load(open(os.path.join(tdir, "out.json")))
    return torch.tensor(r[:3] + [v / 0.01 for v in r[3:]], dtype=dt, device=dev)


def anchor_truthtest(th):
    """Same-contrast check of the anchor's machinery: per group, the corrected mean (apply_volumes with th) against the
    noise-free truth's mean, the truth moved into the estimate's frame by the transform common to all volumes. The shift
    gn_ngf finds should equal the group's mean error less the common one: need = mean_g(th_true - th) - mean(th_true - th)."""
    ipA, fpA = M.consts(G, 1, X, Y, Z, PE, NMON, 2, cen, half, vox_t, scale, dev)[1:]
    mk = torch.tensor(binary_dilation(mask, iterations=2), device=dev)
    with torch.no_grad():
        corr = M.apply_volumes(acq.contiguous(), M.pack(th, rot), ipA, fpA)
        thm = th_true_t.clone(); thm[:] = (th - th_true_t).mean(0, keepdim=True)
        tru = M.apply_volumes(truth_t.contiguous(), M.pack(thm, rot), ipA, fpA)
    for idx in groups:
        gi = torch.tensor(idx, device=dev)
        mov = smooth(corr[gi].mean(0, keepdim=True), 1.0)[0]
        fix = smooth(tru[gi].mean(0, keepdim=True), 1.0)[0]
        e = float(torch.stack(torch.gradient(fix)).norm(dim=0)[mk].mean()) * 0.5
        dd = gn_ngf(mov / e, fix / e, mk)
        need = (th_true_t[gi, :6] - th[gi, :6]).mean(0) - (th_true_t[:, :6] - th[:, :6]).mean(0)
        print(f"truthtest group b{int(round(float(bsel[idx[0]]) / 100) * 100)} (n={len(idx)}): found",
              [round(v, 3) for v in dd[:3].tolist()], [round(float(v) * 0.01 * 57.2958, 3) for v in dd[3:]],
              " needed", [round(v, 3) for v in need[:3].tolist()], [round(float(v) * 57.2958, 3) for v in need[3:6]], flush=True)

def shell_anchor(th):
    """Each DW shell's corrected mean, registered rigidly to the corrected mean b0 by NGF (L-BFGS through grid_sample,
    smoothed 1 voxel); the 6 rigid parameters added to the shell's volumes (first order: small shifts compose additively).
    Returns (theta, {shell: [mm, mm, mm, deg, deg, deg]})."""
    ipA, fpA = M.consts(G, 1, X, Y, Z, PE, NMON, 2, cen, half, vox_t, scale, dev)[1:]
    with torch.no_grad():
        corr = M.apply_volumes(acq.contiguous(), M.pack(th, rot), ipA, fpA)
    b0g = [i for i in groups if isb0[i[0]]]
    if not b0g:
        return th, {}
    fixed = smooth(corr[torch.tensor(b0g[0], device=dev)].mean(0, keepdim=True), 1.0)[0]
    if args.anchor_selftest:                                                              # the registration on a known shift of the b0 mean itself
        sizes_ = torch.tensor([X, Y, Z], dtype=dt, device=dev)
        q0 = torch.zeros(1, 6 + NMON, dtype=dt, device=dev); q0[0, 1] = 0.7; q0[0, 3] = 0.005
        pts0 = phi_fast(q0, full)[0]
        shifted = F.grid_sample(fixed[None, None], (2 * pts0 / (sizes_ - 1) - 1)[..., [2, 1, 0]][None, None, None],
                                mode="bilinear", padding_mode="border", align_corners=True).reshape(X, Y, Z)
        groups_test = [("self-test: b0 mean moved by ty 0.7 mm, rx 0.29 deg", shifted)]
    else:
        groups_test = []
    mk = torch.tensor(binary_dilation(mask, iterations=2), device=dev)
    th = th.clone(); out = {}
    for idx in groups_test + list(groups):
        if isinstance(idx, tuple):
            label, mov = idx
        else:
            if isb0[idx[0]]:
                continue
            label = None
            mov = smooth(corr[torch.tensor(idx, device=dev)].mean(0, keepdim=True), 1.0)[0]
        eps_m = float(torch.stack(torch.gradient(mov)).norm(dim=0)[mk].mean()) * 0.5
        eps_f = float(torch.stack(torch.gradient(fixed)).norm(dim=0)[mk].mean()) * 0.5
        fixed_n = fixed / eps_f; mov_n = mov / eps_m                                     # gradients scaled to ~1
        if args.anchor_flatten:
            from scipy.ndimage import gaussian_filter as _gf
            head_ = binary_dilation(mask, iterations=4)
            def _flat(v):
                v = v.cpu().numpy(); sdv = args.anchor_flatten
                num = _gf(np.where(head_, v, 0.0), sdv); den = _gf(head_.astype(np.float32), sdv)
                return torch.tensor(np.where(head_, v / np.clip(num / np.clip(den, 1e-3, None), 1e-3, None), 0.0), dtype=dt, device=dev)
            mov, fixed_use = _flat(mov), _flat(fixed)                                      # the shell's copies: fixed stays as it was
            mov_n, fixed_n = mov, fixed_use
        else:
            fixed_use = fixed
        if args.anchor_method == "gpumi":
            from mi_torch import register as mi_register
            dd = mi_register(mov, fixed_use, mk, vox, pe_axis=PE, nmon=NMON if args.anchor_eddy else 0, bins=args.anchor_bins)
        else:
            dd = mi_rigid(mov, fixed_use, mk) if args.anchor_method == "mi" else gn_ngf(mov_n, fixed_n, mk)
        if label is not None:
            print(label, "-> recovered", [round(v, 3) for v in dd[:3].tolist()], "mm", [round(float(v) * 0.01 * 180 / math.pi, 3) for v in dd[3:]], "deg", flush=True)
            continue
        if args.profile:
            # what the truth needs: the shell's mean rigid error against the b0 group's (the b0s are the frame)
            b0i_ = torch.tensor(b0g[0], device=dev); gi_ = torch.tensor(idx, device=dev)
            need = (th_true_t[gi_, :6] - th[gi_, :6]).mean(0) - (th_true_t[b0i_, :6] - th[b0i_, :6]).mean(0)
            print("anchor found:", [round(v, 3) for v in dd[:3].tolist()], [round(float(v) * 0.01 * 57.2958, 3) for v in dd[3:]],
                  " truth needs:", [round(v, 3) for v in need[:3].tolist()], [round(float(v) * 57.2958, 3) for v in need[3:6]], flush=True)
        if args.anchor_no_pe:
            dd = dd.clone(); dd[PE] = 0
        th[torch.tensor(idx, device=dev), :3] += dd[:3]
        th[torch.tensor(idx, device=dev), 3:6] += dd[3:6] * 0.01
        if dd.shape[0] > 6:
            th[torch.tensor(idx, device=dev), 6:] += dd[6:]                               # the shell's common eddy field, first order
        out[f"b{int(round(float(bsel[idx[0]]) / 100) * 100)}"] = [round(float(v), 3) for v in dd[:3]] + \
            [round(float(v) * 0.01 * 180 / math.pi, 3) for v in dd[3:6]] + [round(float(v), 3) for v in dd[6:]]
    return th, out


anchors = []
if args.fast:
    fw_steps = ([2.0, 1.0][:args.smooth_steps] + [0.0] * args.outer)[:args.outer]          # the 15 x 1 schedule's smoothing
    coarse = args.coarse                                                                  # the first steps on 1/8 of the points
    sub = torch.tensor(np.flatnonzero((np.argwhere(emask) % 2 == 0).all(1)), device=dev)  # every other voxel per axis
    ye_full = xe
    mon_full = monomials(xe)                                                              # (N, NMON), the Jacobian's (at y, not x)
    ym_full = (xe - cen) * vox_t
    rng_t = torch.Generator(device="cpu").manual_seed(1)
    probe = xedge[torch.randperm(len(xedge), generator=rng_t)[:2000].to(dev)]            # edge points for the step size
    smooth_cache, mcache = {}, {}
    quiet, steps_run = 0, 0
    gains_last = None
    no_eddy_b0 = torch.tensor(isb0, device=dev) if (args.all_shells and isb0.any()) else None
    # parameters held out of the solve, per volume: the b0s' eddy terms (zero at b = 0); every volume's eddy constant,
    # which is the PE translation by another name (the translation carries the whole PE offset during the fit, and the
    # two are separated afterwards, as eddy separates them); and with --ref-pin the first b0's rigid 6 (eddy's location
    # reference). By default the location reference is the b0s' mean pose instead (gauge() below): a pinned volume is
    # the one volume never interpolated, and the others settle around a consensus slightly off it
    frozen = torch.zeros(G, P, dtype=torch.bool, device=dev)
    if no_eddy_b0 is not None and not args.b0_bug and not args.b0_ec:
        frozen[no_eddy_b0, 6:] = True
    if not args.offset_in_solve:
        frozen[:, 6] = True
    ref_vol = int(np.flatnonzero(isb0)[0]) if isb0.any() else 0
    if args.ref_pin and not args.no_ref:
        frozen[ref_vol, :6] = True
    # each group's mean pose is invisible to its own predictor. The b0s' mean pose is the location reference; each DW
    # shell's is held at 0 during the fit too (it drifts slowly otherwise) and set at the end by the shell anchor
    pose_groups = []
    if isb0.any() and not args.no_ref and not args.ref_pin:
        pose_groups.append(torch.tensor(np.flatnonzero(isb0), device=dev))
    if not args.shell_pose_free:
        pose_groups += [torch.tensor(idx, device=dev) for idx in groups if not isb0[idx].all() and len(idx) > 1]
    keep = (~frozen).to(dt)
    # each DW shell's eddy field common to all its directions is invisible to the shell's own predictor and drifts with
    # the steps. Eddy fields scale with the gradient, so that common part is zero: after each step, each eddy term's
    # intercept in a per-shell linear model in g is removed (eddy's second-level model, linear in (b / 1000) g with
    # no intercept, makes the same assumption). The constant term is held out above, so terms 7.. only
    ec_gauge = []
    if not args.no_ec_gauge and NMON > 1:
        for idx in groups:
            if isb0[idx].all() or len(idx) < 5:
                continue
            Xg = np.c_[np.ones(len(idx)), g[idx]]
            ec_gauge.append((torch.tensor(idx, device=dev), torch.tensor(np.linalg.pinv(Xg)[0], dtype=dt, device=dev)))

    def gauge(th):
        for idx, w0 in ec_gauge:
            th[idx, 7:] -= w0 @ th[idx, 7:]
        for idx in pose_groups:                                                           # each group's mean pose
            th[idx, :6] -= th[idx, :6].mean(0)
        if args.b0_ec and no_eddy_b0 is not None and int(no_eddy_b0.sum()) > 1:          # the b0s' mean shape
            th[no_eddy_b0, 7:] -= th[no_eddy_b0, 7:].mean(0)
        return th
    if args.metal:
        import metal_step as M
        nz = np.argwhere(emask4)                                                          # the coefficients' crop: the predictions'
        lo = np.maximum(nz.min(0) - 4, 0); hi = np.minimum(nz.max(0) + 5, [X, Y, Z])     # support plus the cubic's reach
        crop = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
        Xc, Yc, Zc = (int(b - a) for a, b in zip(lo, hi))
        xp_c = xp.contiguous()
        rs_consts = M.consts(G, len(xp), X, Y, Z, PE, NMON, 2, cen, half, vox_t, scale, dev)
    for it, fw in enumerate(fw_steps):
        tick()
        if fw not in smooth_cache:
            smooth_cache[fw] = M.smooth(acq, fw) if args.metal else smooth(acq, fw)
        vols = smooth_cache[fw]
        obs_full = vols.reshape(G, -1)[:, lin_e]                                          # the raw data at the grid: never interpolated
        tick("smooth")
        with torch.no_grad():
            if args.metal:
                u = M.resample(vols.contiguous(), xp_c, M.pack(th, rot), rs_consts[1], rs_consts[2])
            else:
                u = sample_lin(vols, phi_fast(th, xp))
            tick("predict: resample")
            if args.pred_rot and not args.no_loo:
                # each volume measured its gradient in the scanner's frame, so in the corrected (anatomy's) frame its
                # direction is turned by the volume's rotation (as the output b-vectors are: F R^T F g); the predictor is
                # rebuilt from those directions, which matters when volumes rotate by several degrees
                Rk = rot(th[:, 3:6]).cpu().double().numpy()
                Fl = np.diag([-1.0, 1.0, 1.0]) if np.linalg.det(img.affine[:3, :3]) > 0 else np.eye(3)
                g_rot = np.einsum("ij,gkj,kl,gl->gi", Fl, Rk, Fl, g)
                W = torch.tensor(build_loo(g_rot), dtype=dt, device=dev)
            pv = torch.zeros(G, X * Y * Z, dtype=dt, device=dev)
            pv[:, lin_p] = W @ u
            pv = pv.reshape(G, X, Y, Z)
            tick("predict: matmul, scatter")
            coef = M.prefilter(pv[(slice(None),) + crop]) if args.metal else prefilter(pv)
        tick("predict: prefilter")
        c_ = it < coarse
        ys, obs = (ye_full[sub], obs_full[:, sub]) if c_ else (ye_full, obs_full)
        mon, ym = (mon_full[sub], ym_full[sub]) if c_ else (mon_full, ym_full)
        if args.metal:
            key = bool(c_)
            if key not in mcache:
                mcache[key] = M.consts(G, len(ys), Xc, Yc, Zc, PE, NMON, 2, cen, half, vox_t, scale, dev, offset=tuple(float(a) for a in lo))
            nch, ipk, fpk = mcache[key]
            ys_c, obs_c, coef_c = ys.contiguous(), obs.contiguous(), coef.contiguous()
            if args.gain:
                psum = pv.reshape(G, -1)[:, lin_e].sum(1) if not c_ else pv.reshape(G, -1)[:, lin_e][:, sub].sum(1)
                gain = (obs_c.sum(1) / psum.clamp_min(1e-6))                               # (G,): observed over predicted
                obs_c = (obs_c / gain[:, None]).contiguous()
                gains_last = gain
        with torch.no_grad():
          if args.metal:
            H, b, f0 = M.step_moments(coef_c, obs_c, ys_c, M.pack(th, rot), nch, ipk, fpk, P)
            Hd = torch.diagonal(H, dim1=1, dim2=2)
            tick("fused kernel: inverse, sample + gradient, jacobian, H, b")
          else:
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
              bb = b
              # held-out parameters leave the system (the b0s solve the rigid 6 alone, not the 16 solved and the eddy part
              # dropped: the PE translation and the eddy constant are confounded, so dropping the latter lost most of the
              # b0s' PE step)
              A_ = A_ * keep[:, :, None] * keep[:, None, :] + torch.diag_embed(frozen.to(dt))
              bb = b * keep
              step = -torch.linalg.solve(A_.cpu(), bb.cpu()[..., None])[..., 0].to(dev) * scale
              if no_eddy_b0 is not None and not args.b0_ec:
                  step[no_eddy_b0, 6:] = 0                                                 # eddy currents are zero at b = 0
              step = step * keep
              cand = th + step
              big = (phi_fast(cand, probe) - phi_fast(th, probe)).norm(dim=-1).amax(1) > args.trial_vox
              need = todo & (big | (it < 2))                                            # a trial only for large steps (and the first two)
              free = todo & ~need
              th_new = torch.where(free[:, None], cand, th_new); accept |= free
              mu = torch.where(free, (mu / 3).clamp_min(1e-7), mu)
              if bool(need.any()):
                  k = torch.nonzero(need)[:, 0]
                  if args.metal:
                      f1 = M.step_cost(coef_c, obs_c, ys_c, M.pack(cand[k], rot), k, nch, ipk, fpk)
                  else:
                      f1 = ((sample_cubic(coef[k], phi_inv_fast(cand[k], ys)) - obs[k]) ** 2).sum(1)
                  ntrials += 1
                  ok = torch.zeros(G, dtype=torch.bool, device=dev); ok[k] = f1 < f0[k]
                  th_new = torch.where(ok[:, None], cand, th_new); accept |= ok
                  bad = need & ~ok
                  mu = torch.where(ok, (mu / 3).clamp_min(1e-7), torch.where(bad, mu * 8, mu))
                  if not bool(bad.any()):
                      break
              else:
                  break
          moved = float((phi_fast(th_new, probe) - phi_fast(th, probe)).norm(dim=-1).amax()) if args.tol else 1.0
          th = gauge(th_new)
          tick("LM trials (solve, inverse, sample)")
        if args.anchor and args.anchor_every and (it + 1) % args.anchor_every == 0:
            th, a_ = shell_anchor(th); anchors.append({"step": it, **a_})
        quiet = quiet + 1 if (args.tol and fw == 0 and moved < args.tol) else 0
        done = quiet >= 2
        if args.profile or it == len(fw_steps) - 1 or it % 5 == 0 or done:
            history.append({"step": it, "fwhm": fw, "coarse": bool(c_), "accepted": int(accept.sum()),
                            "edge_err_rms_p95": [round(v, 3) for v in edge_err(th)]})
            print(history[-1], flush=True)
        tick("diagnostics (edge error)")
        steps_run = it + 1
        if done:
            break
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
                th_new = torch.where(ok[:, None], cand, th_new)
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
            th_new = torch.where(ok[:, None], cand, th_new)
            accept |= ok
            mu = torch.where(ok, (mu / 3).clamp_min(1e-7), torch.where(todo, mu * 8, mu))
        th = th_new
        if args.slm:
            Xg = torch.cat([torch.ones(G, 1, dtype=dt, device=dev), torch.tensor(g, dtype=dt, device=dev)], 1)   # (G, 4)
            coefs = torch.linalg.lstsq(Xg.cpu(), th[:, 6:].cpu()).solution.to(dev)
            th = torch.cat([th[:, :6], Xg @ coefs], 1)
        history.append({"outer": it, "fwhm": fw, "accepted": int(accept.sum()), "edge_err_rms_p95": [round(v, 3) for v in edge_err(th)]})
        print(history[-1], flush=True)
if args.fast and args.anchor_truthtest:
    anchor_truthtest(th)
if args.fast and args.anchor:
    th, a_ = shell_anchor(th); anchors.append({"step": "end", **a_})
sync(); timings["estimate_s"] = round(time.perf_counter() - t_start, 2)
if anchors:
    timings["shell_anchors (mm, deg)"] = anchors
if args.fast and args.metal:
    with torch.no_grad():
        t0 = time.perf_counter()
        ipA, fpA = M.consts(G, 1, X, Y, Z, PE, NMON, 2, cen, half, vox_t, scale, dev)[1:]
        corr = M.apply_volumes(acq.contiguous(), M.pack(th, rot), ipA, fpA, mask_output=args.mask_output)
        sync(); timings["final_resample_s"] = round(time.perf_counter() - t0, 2)
        timings["total_s"] = round(timings["estimate_s"] + timings["final_resample_s"], 2)
        mk = torch.tensor(mask, device=dev)
        # against the truth, after the gauge: the truth moved by the mean transform the estimate cannot see
        thm = th_true_t.clone(); thm[:] = (th - th_true_t).mean(0, keepdim=True)
        truth_g = M.apply_volumes(truth_t.contiguous(), M.pack(thm, rot), ipA, fpA)
        err_c = float(((corr - truth_g)[:, mk] ** 2).mean().sqrt())
        err_a = float(((acq - truth_t)[:, mk] ** 2).mean().sqrt())
        timings["image_rmse_in_mask"] = {"corrected": round(err_c, 2), "as_acquired": round(err_a, 2), "noise_sigma": round(sigma, 2)}
    if args.real:
        import os
        os.makedirs(args.real, exist_ok=True)
        nib.save(nib.Nifti1Image(np.moveaxis(corr.cpu().numpy(), 0, -1).astype(np.float32), img.affine), f"{args.real}/ours.nii.gz")
        # b-vectors rotated as section 1.3: F R^T F bvec (F the x flip when det(affine) > 0); b0s zero
        F_ = np.diag([-1.0, 1.0, 1.0]) if np.linalg.det(img.affine[:3, :3]) > 0 else np.eye(3)
        thn = th.cpu().double().numpy()
        Rs = rot(torch.tensor(thn[:, 3:6], dtype=torch.float64)).numpy()
        bv_in = bvec[:, shell].copy(); bv_in[:, isb0] = 0
        bv_out = np.stack([F_ @ Rs[k].T @ F_ @ bv_in[:, k] for k in range(G)], 1)
        np.savetxt(f"{args.real}/ours_rotated_bvecs", bv_out, fmt="%.6f")
        # reported parameters: each DW volume's PE offset (translation / voxel + eddy constant, voxels) split as eddy
        # splits it: the part a demeaned linear model in (b / 1000) g explains is the field offset, the rest movement.
        # The sampling map is unchanged; th_fit keeps the parameters as fitted (the offset all in the translation)
        th_fit = thn.copy()
        dw = ~isb0
        if dw.sum() > 3:
            off = thn[dw, PE] / vox[PE] + thn[dw, 6]
            Xd = (bsel[dw] / 1000.0)[:, None] * bvec[:, shell][:, dw].T
            Xd = Xd - Xd.mean(0)
            off_hat = Xd @ np.linalg.lstsq(Xd, off, rcond=None)[0]
            thn[dw, 6] = off_hat
            thn[dw, PE] = (off - off_hat) * vox[PE]
        np.savez(f"{args.real}/ours_theta.npz", th=thn, th_fit=th_fit, pe_axis=PE, cen=cen.cpu().numpy(), vox=vox, nmon=NMON, shell=shell,
                 groups=np.array([np.isin(np.arange(G), gi).astype(int) * (k + 1) for k, gi in enumerate(groups)]).sum(0))
        timings["real_outputs"] = args.real
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
if args.all_shells:
    def edge_err_by_group(th):
        """The edge error with each shell's own common transform removed (within-shell accuracy), and the shells' common
        transforms' spread against each other (between-shell drift), voxels RMS."""
        d = phi(th, xedge) - phi(th_true_t, xedge)
        d = d - d.mean(0, keepdim=True)
        within, means = [], []
        for idx in groups:
            dg = d[torch.tensor(idx, device=dev)]
            m = dg.mean(0, keepdim=True); means.append(m)
            within.append((dg - m).norm(dim=-1).reshape(-1))
        w = torch.cat(within)
        mm = torch.cat(means)
        return {"within_shell_rms": round(float(w.pow(2).mean().sqrt()), 3),
                "between_shell_rms": round(float((mm - mm.mean(0, keepdim=True)).norm(dim=-1).pow(2).mean().sqrt()), 3),
                "shells": [int(len(i)) for i in groups]}
    info["edge_error_by_shell"] = edge_err_by_group(th)
    dth = (th - th_true_t); dth = dth - dth.mean(0, keepdim=True)
    info["mean_param_error_by_shell (mm, deg, eddy vox)"] = {
        f"group{k} (n={len(idx)})": [round(float(v), 3) for v in torch.cat([dth[torch.tensor(idx, device=dev)].mean(0)[:3],
                                                                              torch.rad2deg(dth[torch.tensor(idx, device=dev)].mean(0)[3:6]),
                                                                              dth[torch.tensor(idx, device=dev)].mean(0)[6:]])]
        for k, idx in enumerate(groups)}
info.update({"true_edge_displacement_rms_p95_vox": [round(v, 3) for v in disp_true()],
             "final_edge_error_rms_p95_vox": [round(v, 3) for v in edge_err(th)],
             "gn_iterations": args.outer * args.inner, **timings, "history": history,
             **({"profile_s": {k: round(v, 2) for k, v in prof.items()}, "lm_trials": ntrials} if args.profile else {}),
             **({"steps_run": steps_run, "tol_vox": args.tol} if args.fast else {}),
             **({"gains (b0s first)": [round(float(v), 4) for v in torch.cat([gains_last[torch.tensor(np.flatnonzero(isb0), device=dev)], gains_last[torch.tensor(np.flatnonzero(~isb0)[:5], device=dev)]])]} if (args.fast and args.gain and gains_last is not None and args.all_shells) else {}),
             "device": args.device})
tag = f"_{args.device}_o{args.order}" + ("_noloo" if args.no_loo else "") + (f"_s{args.scale}" if args.scale != 1 else "") + (f"_e{args.eddy_scale}" if args.eddy_scale != 1 else "") + f"_n{args.outer}" + (f"_f{args.final_fwhm}" if args.final_fwhm else "") + ("_slm" if args.slm else "") + ("_acq" if args.acq_cost else "") + ("_cubic" if args.cubic else "") + ("_real" if args.real else "") + (f"_{SUB}" if (args.scan != "16974/OXF1PRI001" or args.dwi) else "") + ("_gain" if args.gain else "") + ("_all" if args.all_shells else "") + ("_tt" if args.tensor_truth else "") + (f"_b{args.shells.replace(',', '-')}" if args.shells else "") + (f"_anchor{args.anchor_every}{args.anchor_method}" + ("e" if args.anchor_eddy else "") + (f"f{args.anchor_flatten:g}" if args.anchor_flatten else "") + (f"b{args.anchor_bins}" if args.anchor_bins != 256 else "") + ("nope" if args.anchor_no_pe else "") if args.anchor else "") + (f"_fast_c{args.coarse}_" + ("metal" if args.metal else args.grad) + (f"_sm{args.smooth_steps}" if args.smooth_steps != 2 else "") + (f"_tol{args.tol}" if args.tol else "") if args.fast else "") + ("_prof" if args.profile else "")
json.dump(info, open(OUT / f"proto{tag}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in info.items() if k != "history"}))
