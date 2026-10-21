"""metal_step's kernels against float64 references on random data (MPS): step_moments (H, b, cost), step_cost, prefilter
(against proto.prefilter's dense mirrored solve), smooth (against proto.smooth's conv3d).

    uv run bench/motion_eddy/metal_check.py
"""
import sys, math, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import metal_step as M
dev = torch.device("mps"); dt = torch.float32
torch.manual_seed(0)
G, X, Y, Z, N, PE, NMON = 7, 40, 44, 30, 3000, 1, 10
cen = (torch.tensor([X, Y, Z], dtype=dt, device=dev) - 1) / 2; half = cen.clone()
vox = torch.tensor([2.02, 2.02, 2.0], dtype=dt, device=dev)
scale = torch.cat([torch.ones(3), torch.full((3,), 0.01), torch.ones(NMON)]).to(dev, dt)
def rot(w):
    cx, cy, cz = torch.cos(w).unbind(-1); sx, sy, sz = torch.sin(w).unbind(-1)
    o, z = torch.ones_like(cx), torch.zeros_like(cx)
    Rx = torch.stack([o, z, z, z, cx, -sx, z, sx, cx], -1).view(-1, 3, 3)
    Ry = torch.stack([cy, z, sy, z, o, z, -sy, z, cy], -1).view(-1, 3, 3)
    Rz = torch.stack([cz, -sz, z, sz, cz, z, z, z, o], -1).view(-1, 3, 3)
    return Rz @ Ry @ Rx
coef = torch.randn(G, X, Y, Z, dtype=dt, device=dev)
pts = (torch.rand(N, 3, device=dev) * (torch.tensor([X, Y, Z], device=dev) - 8) + 4).contiguous()
obs = torch.randn(G, N, dtype=dt, device=dev)
th = torch.cat([torch.randn(G, 3) * 1.0, torch.randn(G, 3) * 0.02, torch.randn(G, NMON) * 0.3], 1).to(dev, dt)
# torch reference
def poly(te, x):
    u = (x - cen) / half; a, b, c = u.unbind(-1); k = lambda i: te[:, i, None]
    return k(0) + k(1)*a + k(2)*b + k(3)*c + k(4)*a*a + k(5)*b*b + k(6)*c*c + k(7)*a*b + k(8)*a*c + k(9)*b*c
R = rot(th[:, 3:6]); t = th[:, None, :3]; e = poly(th[:, 6:], pts)
E = torch.eye(3, device=dev)[PE]
for _ in range(2):
    q = (pts - cen) * vox - e[..., None] * (E * vox) - t
    x = torch.einsum("gji,gnj->gni", R, q) / vox + cen
    e = poly(th[:, 6:], x)
# 64-tap cubic value/grad in float64 on CPU for the reference
cc, xx = coef.cpu().double(), x.cpu().double()
i0 = torch.floor(xx); tt = xx - i0; i0 = i0.long()
w = [(1-tt)**3/6, (3*tt**3-6*tt**2+4)/6, (-3*tt**3+3*tt**2+3*tt+1)/6, tt**3/6]
dw = [-(1-tt)**2/2, (3*tt**2-4*tt)/2, (-3*tt**2+2*tt+1)/2, tt**2/2]
flat = cc.reshape(G, -1); val = torch.zeros(G, N, dtype=torch.float64); gx = torch.zeros_like(val); gy = torch.zeros_like(val); gz = torch.zeros_like(val)
for a_ in range(4):
    ia = (i0[..., 0] + a_ - 1).clamp(0, X - 1)
    for b_ in range(4):
        ib = (i0[..., 1] + b_ - 1).clamp(0, Y - 1)
        for c_ in range(4):
            ic = (i0[..., 2] + c_ - 1).clamp(0, Z - 1)
            v = torch.gather(flat, 1, (ia * Y + ib) * Z + ic)
            val += w[a_][..., 0]*w[b_][..., 1]*w[c_][..., 2]*v; gx += dw[a_][..., 0]*w[b_][..., 1]*w[c_][..., 2]*v
            gy += w[a_][..., 0]*dw[b_][..., 1]*w[c_][..., 2]*v; gz += w[a_][..., 0]*w[b_][..., 1]*dw[c_][..., 2]*v
gp = torch.stack([gx, gy, gz], -1)
r = val - obs.cpu().double()
voxc, cenc, halfc = vox.cpu().double(), cen.cpu().double(), half.cpu().double()
ym = (pts.cpu().double() - cenc) * voxc
w_ = th[:, 3:6].cpu().double()
rc = []
for j in range(3):
    d = torch.zeros_like(w_); d[:, j] = 1e-4
    dR = ((rot(w_.float() + d.float()) - rot(w_.float() - d.float())) / 2e-4).double()
    rc.append(((torch.einsum("gij,nj->gni", dR, ym) / voxc) * gp).sum(-1))
u = (pts.cpu().double() - cenc) / halfc; a, b, c = u.unbind(-1)
mon = torch.stack([torch.ones_like(a), a, b, c, a*a, b*b, c*c, a*b, a*c, b*c], -1)
J = -torch.cat([gp / voxc, torch.stack(rc, -1), gp[..., PE:PE+1] * mon[None]], -1) * scale.cpu().double()
Href = torch.bmm(J.transpose(1, 2), J); bref = torch.bmm(J.transpose(1, 2), r[..., None])[..., 0]; fref = (r ** 2).sum(1)
# Metal
nch, ip, fp = M.consts(G, N, X, Y, Z, PE, NMON, 2, cen, half, vox, scale, dev)
pv = M.pack(th, rot)
H, bb, f = M.step_moments(coef.contiguous(), obs.contiguous(), pts, pv, nch, ip, fp, 16)
rel = lambda A, B: float((A.cpu().double() - B).abs().max() / B.abs().max())
print("H rel err", rel(H, Href), " b rel err", rel(bb, bref), " f rel err", rel(f, fref))
vols = torch.tensor([1, 4, 6], device=dev)
fc = M.step_cost(coef.contiguous(), obs.contiguous(), pts, pv[vols].contiguous(), vols, nch, ip, fp)
print("cost rel err", rel(fc, fref[[1, 4, 6]]))
# prefilter: recursive Metal against the dense mirrored solve (proto.prefilter's), float64 reference
def dense_prefilter(img):
    for ax in range(3):
        n = img.shape[1 + ax]
        Mx = torch.zeros(n, n, dtype=torch.float64)
        for i in range(n):
            Mx[i, i] = 4 / 6
            for j in (i - 1, i + 1):
                jj = -j if j < 0 else (2 * (n - 1) - j if j > n - 1 else j)
                Mx[i, jj] += 1 / 6
        Minv = torch.linalg.inv(Mx)
        img = torch.movedim(torch.movedim(img, 1 + ax, -1) @ Minv.T, -1, 1 + ax)
    return img
ref = dense_prefilter(coef.cpu().double())
got = M.prefilter(coef)
print("prefilter rel err", float((got.cpu().double() - ref).abs().max() / ref.abs().max()))
import torch.nn.functional as F
def smooth_ref(vols, fwhm_vox):
    sd = fwhm_vox / 2.3548; r = int(math.ceil(3 * sd))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=dt, device=dev) / sd) ** 2); k = k / k.sum()
    out = vols[:, None]
    for ax in range(3):
        shape = [1, 1, 1, 1, 1]; shape[2 + ax] = len(k)
        pad = [0] * 6; pad[2 * (2 - ax)] = pad[2 * (2 - ax) + 1] = r
        out = F.conv3d(F.pad(out, pad, mode="replicate"), k.view(shape))
    return out[:, 0]
for fw in (2.0, 1.0):
    a_, b_ = smooth_ref(coef, fw), M.smooth(coef, fw)
    print("smooth", fw, "rel err", float((a_ - b_).abs().max() / a_.abs().max()))
# apply_volumes: the final resample, against a float64 64-tap cubic of the dense-prefiltered volumes at phi(x), times
# the eddy Jacobian, clipped at 0
vols_ = coef.abs() + 0.5
full = torch.stack(torch.meshgrid(*[torch.arange(n, dtype=torch.float64) for n in (X, Y, Z)], indexing="ij"), -1).reshape(-1, 3)
c64 = dense_prefilter(vols_.cpu().double())
th64 = th.cpu().double(); cen64, half64, vox64 = cen.cpu().double(), half.cpu().double(), vox.cpu().double()
R64 = rot(th64[:, 3:6].float()).double()
p_ = (torch.einsum("gij,nj->gni", R64, (full - cen64) * vox64) + th64[:, None, :3]) / vox64 + cen64
u_ = (full - cen64) / half64; a, b, c = u_.unbind(-1)
mon64 = torch.stack([torch.ones_like(a), a, b, c, a*a, b*b, c*c, a*b, a*c, b*c], -1)
p_[..., PE] += th64[:, 6:] @ mon64.T
dmon = torch.stack([torch.zeros_like(a), torch.zeros_like(a), torch.ones_like(a), torch.zeros_like(a), 0*a, 2*b, 0*a, a, 0*a, c], -1)  # d/du_y (PE = 1)
jac = 1 + (th64[:, 6:] @ dmon.T) / half64[PE]
i0 = torch.floor(p_); tt = p_ - i0; i0 = i0.long()
w = [(1-tt)**3/6, (3*tt**3-6*tt**2+4)/6, (-3*tt**3+3*tt**2+3*tt+1)/6, tt**3/6]
flat = c64.reshape(G, -1); val = torch.zeros(G, full.shape[0], dtype=torch.float64)
for a_ in range(4):
    ia = (i0[..., 0] + a_ - 1).clamp(0, X - 1)
    for b_ in range(4):
        ib = (i0[..., 1] + b_ - 1).clamp(0, Y - 1)
        for c_ in range(4):
            ic = (i0[..., 2] + c_ - 1).clamp(0, Z - 1)
            val += w[a_][..., 0]*w[b_][..., 1]*w[c_][..., 2]*torch.gather(flat, 1, (ia * Y + ib) * Z + ic)
ref = (val * jac).clamp_min(0).reshape(G, X, Y, Z)
nchA, ipA, fpA = M.consts(G, 1, X, Y, Z, PE, NMON, 2, cen, half, vox, scale, dev)
got = M.apply_volumes(vols_.contiguous(), M.pack(th, rot), ipA, fpA)
print("apply_volumes rel err", float((got.cpu().double() - ref).abs().max() / ref.abs().max()))
