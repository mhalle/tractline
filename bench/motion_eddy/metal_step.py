"""The motion + eddy step's inner loop as two Metal kernels (torch.mps.compile_shader, as tractline.ukf_metal), for
proto.py --fast --metal. Per point y of each volume g, with no per-point array in memory:
  - x = phi_g^-1(y): the rigid part in closed form, the eddy polynomial (along PE, voxels) by fixed-point passes;
  - the prediction's cubic B-spline value and exact gradient at x: the 64 coefficients gathered once each, edges clamped
    (proto.cubic_value_grad's arithmetic);
  - the residual r = value - obs, and the 16 Jacobian entries J = -[grad / vox, (dR_j ym) . grad / vox,
    grad_pe m_k(y)] * scale, as proto.py's fast path (rotation and eddy columns at y);
  - step_moments: the upper triangle of J'J (136), J'r (16) and r^2, summed over a threadgroup (simd_sum, then
    threadgroup memory) and written per threadgroup; the threadgroups' partials are summed by torch in a fixed order,
    so a run is deterministic on a device (no atomics);
  - step_cost: r^2 only, for the volumes listed (the Levenberg-Marquardt trials).
float32 throughout (Apple GPUs have no float64).
"""
from __future__ import annotations

import torch

TG = 256                                                  # threads per threadgroup (8 SIMD groups of 32)
NH = 136                                                  # upper triangle of a 16 x 16
NA = NH + 16 + 1                                          # + J'r + r^2

_SRC = r"""
#include <metal_stdlib>
using namespace metal;

#define TG 256
#define NSG (TG / 32)
#define NH 136
#define NA (NH + 16 + 1)
#define NPV 49                                            // per volume: R (9), t (3), eddy (10), dR_j (27)

struct Geo { float cen[3]; float half_[3]; float vox[3]; };

static inline float eddy_at(device const float* te, float3 u, int nmon) {
    float e = te[0] + te[1] * u.x + te[2] * u.y + te[3] * u.z;
    if (nmon == 10)
        e += te[4] * u.x * u.x + te[5] * u.y * u.y + te[6] * u.z * u.z + te[7] * u.x * u.y + te[8] * u.x * u.z + te[9] * u.y * u.z;
    return e;
}

// x = phi^-1(y), voxels
static inline float3 phi_inv(device const float* pv, float3 y, device const float* fp, int pe, int nmon, int npass) {
    float3 cen = float3(fp[0], fp[1], fp[2]), hlf = float3(fp[3], fp[4], fp[5]), vox = float3(fp[6], fp[7], fp[8]);
    device const float* R = pv; device const float* t = pv + 9; device const float* te = pv + 12;
    float e = eddy_at(te, (y - cen) / hlf, nmon);
    float3 x = y;
    for (int k = 0; k < npass; k++) {
        float3 q = (y - cen) * vox - float3(t[0], t[1], t[2]);
        q[pe] -= e * vox[pe];
        float3 xm = float3(R[0] * q.x + R[3] * q.y + R[6] * q.z,          // R^T q (R row-major)
                           R[1] * q.x + R[4] * q.y + R[7] * q.z,
                           R[2] * q.x + R[5] * q.y + R[8] * q.z);
        x = xm / vox + cen;
        e = eddy_at(te, (x - cen) / hlf, nmon);
    }
    return x;
}

static inline void bweights(float t, thread float* w, thread float* dw) {
    float s = 1.0f - t;
    w[0] = s * s * s / 6.0f; w[1] = (3.0f * t * t * t - 6.0f * t * t + 4.0f) / 6.0f;
    w[2] = (-3.0f * t * t * t + 3.0f * t * t + 3.0f * t + 1.0f) / 6.0f; w[3] = t * t * t / 6.0f;
    dw[0] = -s * s / 2.0f; dw[1] = (3.0f * t * t - 4.0f * t) / 2.0f; dw[2] = (-3.0f * t * t + 2.0f * t + 1.0f) / 2.0f; dw[3] = t * t / 2.0f;
}

// the cubic B-spline of one volume's coefficients at x, and its gradient (voxels)
static inline float cubic_vg(device const float* c, float3 x, int X, int Y, int Z, thread float3& grad, bool want_grad) {
    float3 f = floor(x); float3 t = x - f;
    int ix = int(f.x), iy = int(f.y), iz = int(f.z);
    float wx[4], wy[4], wz[4], dx[4], dy[4], dz[4];
    bweights(t.x, wx, dx); bweights(t.y, wy, dy); bweights(t.z, wz, dz);
    float v = 0, gx = 0, gy = 0, gz = 0;
    for (int a = 0; a < 4; a++) {
        int ia = clamp(ix + a - 1, 0, X - 1);
        for (int b = 0; b < 4; b++) {
            int ib = clamp(iy + b - 1, 0, Y - 1);
            device const float* row = c + (long(ia) * Y + ib) * Z;
            float wab = wx[a] * wy[b], dab_x = dx[a] * wy[b], dab_y = wx[a] * dy[b];
            for (int k = 0; k < 4; k++) {
                float cv = row[clamp(iz + k - 1, 0, Z - 1)];
                v += wab * wz[k] * cv;
                if (want_grad) { gx += dab_x * wz[k] * cv; gy += dab_y * wz[k] * cv; gz += wab * dz[k] * cv; }
            }
        }
    }
    grad = float3(gx, gy, gz);
    return v;
}

kernel void step_moments(
    device const float* coef  [[buffer(0)]],    // (G, X, Y, Z) prefiltered prediction
    device const float* obs   [[buffer(1)]],    // (G, N) the data at the points
    device const float* pts   [[buffer(2)]],    // (N, 3) points y, voxels
    device const float* pvol  [[buffer(3)]],    // (G, NPV) per-volume parameters
    device float*       part  [[buffer(4)]],    // (G, NCH, NA) per-threadgroup partial sums
    device const int*       ip    [[buffer(5)]],    // G, N, NCH, X, Y, Z (the coefficients' grid), PE, NMON, NPASS
    device const float* fp    [[buffer(6)]],    // cen (3), half (3), vox (3), scale (16), the coefficients' offset (3)
    uint tg   [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    int N = ip[1], NCH = ip[2], X = ip[3], Y = ip[4], Z = ip[5], pe = ip[6], nmon = ip[7], npass = ip[8];
    int g = int(tg) / NCH, ch = int(tg) % NCH;
    int n = ch * TG + int(lid);
    threadgroup float sh[NSG * NA];
    float J[16]; float r = 0;
    for (int i = 0; i < 16; i++) J[i] = 0;
    if (n < N) {
        device const float* pv = pvol + long(g) * NPV;
        float3 y = float3(pts[3 * n], pts[3 * n + 1], pts[3 * n + 2]);
        float3 x = phi_inv(pv, y, fp, pe, nmon, npass);
        float3 gr;
        float v = cubic_vg(coef + long(g) * X * Y * Z, x - float3(fp[25], fp[26], fp[27]), X, Y, Z, gr, true);
        r = v - obs[long(g) * N + n];
        float3 cen = float3(fp[0], fp[1], fp[2]), hlf = float3(fp[3], fp[4], fp[5]), vox = float3(fp[6], fp[7], fp[8]);
        device const float* sc = fp + 9;
        float3 gv = gr / vox;
        J[0] = -gv.x * sc[0]; J[1] = -gv.y * sc[1]; J[2] = -gv.z * sc[2];
        float3 ym = (y - cen) * vox;
        for (int j = 0; j < 3; j++) {
            device const float* d = pv + 22 + 9 * j;                         // dR_j, row-major
            float3 dm = float3(d[0] * ym.x + d[1] * ym.y + d[2] * ym.z, d[3] * ym.x + d[4] * ym.y + d[5] * ym.z,
                               d[6] * ym.x + d[7] * ym.y + d[8] * ym.z);
            J[3 + j] = -dot(dm, gv) * sc[3 + j];
        }
        float3 u = (y - cen) / hlf;
        float m[10] = {1.0f, u.x, u.y, u.z, u.x * u.x, u.y * u.y, u.z * u.z, u.x * u.y, u.x * u.z, u.y * u.z};
        float gpe = gr[pe];
        for (int k = 0; k < nmon; k++) J[6 + k] = -gpe * m[k] * sc[6 + k];
    }
    // sums over the SIMD group, then over the threadgroup
    int q = 0;
    for (int i = 0; i < 16; i++)
        for (int j = i; j < 16; j++) {
            float s = simd_sum(J[i] * J[j]);
            if (lane == 0) sh[sg * NA + q] = s;
            q++;
        }
    for (int i = 0; i < 16; i++) {
        float s = simd_sum(J[i] * r);
        if (lane == 0) sh[sg * NA + NH + i] = s;
    }
    float s2 = simd_sum(r * r);
    if (lane == 0) sh[sg * NA + NH + 16] = s2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = int(lid); k < NA; k += TG) {
        float s = 0;
        for (int w = 0; w < NSG; w++) s += sh[w * NA + k];
        part[(long(g) * NCH + ch) * NA + k] = s;
    }
}

kernel void step_cost(
    device const float* coef  [[buffer(0)]],
    device const float* obs   [[buffer(1)]],
    device const float* pts   [[buffer(2)]],
    device const float* pvol  [[buffer(3)]],    // (len(vols), NPV): the trial parameters of the listed volumes
    device const int*   vols  [[buffer(4)]],    // the listed volumes' indices
    device float*       part  [[buffer(5)]],    // (len(vols), NCH)
    device const int*       ip    [[buffer(6)]],
    device const float* fp    [[buffer(7)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    int N = ip[1], NCH = ip[2], X = ip[3], Y = ip[4], Z = ip[5], pe = ip[6], nmon = ip[7], npass = ip[8];
    int li = int(tg) / NCH, ch = int(tg) % NCH;
    int g = vols[li];
    int n = ch * TG + int(lid);
    threadgroup float sh[NSG];
    float r2 = 0;
    if (n < N) {
        float3 y = float3(pts[3 * n], pts[3 * n + 1], pts[3 * n + 2]);
        float3 x = phi_inv(pvol + long(li) * NPV, y, fp, pe, nmon, npass);
        float3 gr;
        float v = cubic_vg(coef + long(g) * X * Y * Z, x - float3(fp[25], fp[26], fp[27]), X, Y, Z, gr, false);
        float r = v - obs[long(g) * N + n];
        r2 = r * r;
    }
    float s = simd_sum(r2);
    if (lane == 0) sh[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0) {
        float t = 0;
        for (int w = 0; w < NSG; w++) t += sh[w];
        part[long(li) * NCH + ch] = t;
    }
}

// the forward map phi(x) and a trilinear sample of the acquired volume there (zero outside: grid_sample's padding)
kernel void resample(
    device const float* vols  [[buffer(0)]],    // (G, X, Y, Z) the acquired (smoothed) volumes, full grid
    device const float* pts   [[buffer(1)]],    // (N, 3) reference points x, voxels
    device const float* pvol  [[buffer(2)]],    // (G, NPV)
    device float*       out   [[buffer(3)]],    // (G, N)
    device const int*   ip    [[buffer(4)]],    // G, N, -, X, Y, Z, PE, NMON
    device const float* fp    [[buffer(5)]],
    uint tid [[thread_position_in_grid]])
{
    int N = ip[1], X = ip[3], Y = ip[4], Z = ip[5], pe = ip[6], nmon = ip[7];
    int g = int(tid) / N, n = int(tid) % N;
    if (g >= ip[0]) return;
    device const float* pv = pvol + long(g) * NPV;
    float3 cen = float3(fp[0], fp[1], fp[2]), hlf = float3(fp[3], fp[4], fp[5]), vox = float3(fp[6], fp[7], fp[8]);
    float3 x = float3(pts[3 * n], pts[3 * n + 1], pts[3 * n + 2]);
    float3 xm = (x - cen) * vox;
    float3 p = float3(pv[0] * xm.x + pv[1] * xm.y + pv[2] * xm.z, pv[3] * xm.x + pv[4] * xm.y + pv[5] * xm.z,
                      pv[6] * xm.x + pv[7] * xm.y + pv[8] * xm.z);
    p = (p + float3(pv[9], pv[10], pv[11])) / vox + cen;
    p[pe] += eddy_at(pv + 12, (x - cen) / hlf, nmon);
    float3 f = floor(p); float3 t = p - f;
    int ix = int(f.x), iy = int(f.y), iz = int(f.z);
    device const float* c = vols + long(g) * X * Y * Z;
    float v = 0;
    for (int a = 0; a < 2; a++) {
        int ia = ix + a; if (ia < 0 || ia > X - 1) continue;
        float wa = a ? t.x : 1.0f - t.x;
        for (int b = 0; b < 2; b++) {
            int ib = iy + b; if (ib < 0 || ib > Y - 1) continue;
            float wb = b ? t.y : 1.0f - t.y;
            for (int k = 0; k < 2; k++) {
                int ik = iz + k; if (ik < 0 || ik > Z - 1) continue;
                v += wa * wb * (k ? t.z : 1.0f - t.z) * c[(long(ia) * Y + ib) * Z + ik];
            }
        }
    }
    out[long(g) * N + n] = v;
}

// cubic B-spline prefilter along one axis, in place: the recursive filter (pole z = sqrt(3) - 2, gain 6), whole-sample
// mirrored edges as the dense solve in proto.prefilter (Unser 1999); the causal start a truncated sum (|z|^K < 1e-9)
kernel void prefilter_axis(
    device float*     d   [[buffer(0)]],    // (G, X, Y, Z), filtered in place along `axis`
    device const int* ip  [[buffer(1)]],    // G, X, Y, Z, axis
    uint tid [[thread_position_in_grid]])
{
    int G = ip[0], X = ip[1], Y = ip[2], Z = ip[3], ax = ip[4];
    int n, stride; long base;
    if (ax == 2) {          // lines along z: (g, x, y)
        int nl = G * X * Y; if (int(tid) >= nl) return;
        n = Z; stride = 1; base = long(tid) * Z;
    } else if (ax == 1) {   // along y: (g, x, z)
        int nl = G * X * Z; if (int(tid) >= nl) return;
        int z = int(tid) % Z, gx = int(tid) / Z;
        n = Y; stride = Z; base = long(gx) * Y * Z + z;
    } else {                // along x: (g, y, z)
        int nl = G * Y * Z; if (int(tid) >= nl) return;
        int yz = int(tid) % (Y * Z), g = int(tid) / (Y * Z);
        n = X; stride = Y * Z; base = long(g) * X * Y * Z + yz;
    }
    if (n < 2) return;
    const float zp = -0.26794919243112270f;
    // causal: c+[0] = sum_k z^k f[k] (mirror: f[-k] = f[k], so the infinite sum is over the line and its reflection)
    float c0 = d[base];
    float zk = zp;
    int K = min(n - 1, 16);
    for (int k = 1; k <= K; k++) { c0 += zk * d[base + long(k) * stride]; zk *= zp; }
    d[base] = c0;
    for (int k = 1; k < n; k++) d[base + long(k) * stride] += zp * d[base + long(k - 1) * stride];
    // anticausal: c-[n-1] = z / (z^2 - 1) (c+[n-1] + z c+[n-2]); c-[k] = z (c-[k+1] - c+[k])
    long last = base + long(n - 1) * stride;
    d[last] = zp / (zp * zp - 1.0f) * (d[last] + zp * d[last - stride]);
    for (int k = n - 2; k >= 0; k--) d[base + long(k) * stride] = zp * (d[base + long(k + 1) * stride] - d[base + long(k) * stride]);
    for (int k = 0; k < n; k++) d[base + long(k) * stride] *= 6.0f;
}

// one axis of a separable Gaussian, replicate edges (proto.smooth's padding): dst = sum_k w[k] src[clamp(i + k - r)]
kernel void gauss_axis(
    device const float* src [[buffer(0)]],    // (G, X, Y, Z)
    device float*       dst [[buffer(1)]],
    device const float* w   [[buffer(2)]],    // (2r + 1) normalized weights
    device const int*   ip  [[buffer(3)]],    // G, X, Y, Z, axis, r
    uint tid [[thread_position_in_grid]])
{
    int G = ip[0], X = ip[1], Y = ip[2], Z = ip[3], ax = ip[4], r = ip[5];
    long nv = long(G) * X * Y * Z;
    if (long(tid) >= nv) return;
    int z = int(tid % Z), y = int((tid / Z) % Y), x = int((tid / (long(Y) * Z)) % X);
    int n = ax == 0 ? X : (ax == 1 ? Y : Z);
    int i = ax == 0 ? x : (ax == 1 ? y : z);
    long stride = ax == 0 ? long(Y) * Z : (ax == 1 ? long(Z) : 1);
    long base = long(tid) - long(i) * stride;
    float acc = 0;
    for (int k = -r; k <= r; k++) acc += w[k + r] * src[base + long(clamp(i + k, 0, n - 1)) * stride];
    dst[tid] = acc;
}

// the final resample: every voxel x of the reference grid, each volume sampled once at phi(x) by its cubic B-spline,
// times the eddy field's Jacobian along PE (1 + de/dx_pe), clipped at 0 (susceptibility.apply's clip); edges clamped
kernel void apply_volumes(
    device const float* coef  [[buffer(0)]],    // (G, X, Y, Z) prefiltered acquired volumes
    device const float* pvol  [[buffer(1)]],    // (G, NPV)
    device float*       out   [[buffer(2)]],    // (G, X, Y, Z)
    device const int*   ip    [[buffer(3)]],    // G, -, -, X, Y, Z, PE, NMON
    device const float* fp    [[buffer(4)]],
    device float*       valid [[buffer(5)]],    // (G, X, Y, Z): 1 where the sample is inside the field of view, else 0
    uint tid [[thread_position_in_grid]])
{
    int G = ip[0], X = ip[3], Y = ip[4], Z = ip[5], pe = ip[6], nmon = ip[7];
    long nv = long(X) * Y * Z;
    if (long(tid) >= long(G) * nv) return;
    int g = int(long(tid) / nv); long v = long(tid) % nv;
    float3 x = float3(float(v / (long(Y) * Z)), float((v / Z) % Y), float(v % Z));
    device const float* pv = pvol + long(g) * NPV;
    float3 cen = float3(fp[0], fp[1], fp[2]), hlf = float3(fp[3], fp[4], fp[5]), vox = float3(fp[6], fp[7], fp[8]);
    float3 xm = (x - cen) * vox;
    float3 p = float3(pv[0] * xm.x + pv[1] * xm.y + pv[2] * xm.z, pv[3] * xm.x + pv[4] * xm.y + pv[5] * xm.z,
                      pv[6] * xm.x + pv[7] * xm.y + pv[8] * xm.z);
    p = (p + float3(pv[9], pv[10], pv[11])) / vox + cen;
    float3 u = (x - cen) / hlf;
    device const float* te = pv + 12;
    p[pe] += eddy_at(te, u, nmon);
    // de/du along pe: the polynomial's derivative, then / half (u = (x - cen) / half)
    float d;
    if (pe == 0) d = te[1] + (nmon == 10 ? 2 * te[4] * u.x + te[7] * u.y + te[8] * u.z : 0);
    else if (pe == 1) d = te[2] + (nmon == 10 ? 2 * te[5] * u.y + te[7] * u.x + te[9] * u.z : 0);
    else d = te[3] + (nmon == 10 ? 2 * te[6] * u.z + te[8] * u.x + te[9] * u.y : 0);
    float jac = 1.0f + d / hlf[pe];
    float3 gr;
    float val = cubic_vg(coef + long(g) * nv, p, X, Y, Z, gr, false);
    bool in = p.x >= 0.0f && p.x <= float(X - 1) && p.y >= 0.0f && p.y <= float(Y - 1) && p.z >= 0.0f && p.z <= float(Z - 1);
    valid[tid] = in ? 1.0f : 0.0f;
    out[tid] = in ? max(val * jac, 0.0f) : 0.0f;
}
"""

_LIB = None


def library():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SRC)
    return _LIB


def pack(theta, rot, eps=1e-4):
    """Per-volume parameters (G, 49): R (9, row-major), t (3, mm), eddy (10, voxels; zero-padded), dR/dw_j (3 x 9)."""
    G = theta.shape[0]
    w = theta[:, 3:6]
    R = rot(w)
    dR = []
    for j in range(3):
        d = torch.zeros_like(w); d[:, j] = eps
        dR.append(((rot(w + d) - rot(w - d)) / (2 * eps)).reshape(G, 9))
    te = torch.zeros(G, 10, dtype=theta.dtype, device=theta.device)
    te[:, :theta.shape[1] - 6] = theta[:, 6:]
    return torch.cat([R.reshape(G, 9), theta[:, :3], te] + dR, 1).contiguous()


def unpack_H(part, P):
    """(G, NCH, NA) partials -> H (G, P, P), b (G, P), f (G,): summed over the threadgroups in a fixed order."""
    s = part.sum(1)
    G = s.shape[0]
    H = torch.zeros(G, 16, 16, dtype=s.dtype, device=s.device)
    iu = torch.triu_indices(16, 16, device=s.device)
    H[:, iu[0], iu[1]] = s[:, :NH]
    H = H + H.transpose(1, 2) - torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2))
    return H[:, :P, :P], s[:, NH:NH + P], s[:, NH + 16]


def consts(G, N, X, Y, Z, pe, nmon, npass, cen, half, vox, scale, device, offset=(0, 0, 0)):
    """X, Y, Z: the grid of the volumes the kernel samples (a crop, `offset` its first voxel in the full grid)."""
    nch = (N + TG - 1) // TG
    ip = torch.tensor([G, N, nch, X, Y, Z, pe, nmon, npass], dtype=torch.int32, device=device)
    sc = torch.zeros(16, dtype=torch.float32)
    sc[:len(scale)] = scale.float().cpu()
    fp = torch.cat([cen.float().cpu(), half.float().cpu(), vox.float().cpu(), sc,
                    torch.tensor(offset, dtype=torch.float32)]).to(device)
    return nch, ip, fp


def prefilter(vols):
    """Cubic B-spline coefficients of vols (G, X, Y, Z), mirrored edges, by the recursive filter on three axes (a copy)."""
    c = vols.contiguous().clone()
    G, X, Y, Z = c.shape
    lib = library()
    for ax, nl in ((2, G * X * Y), (1, G * X * Z), (0, G * Y * Z)):
        ip = torch.tensor([G, X, Y, Z, ax], dtype=torch.int32, device=c.device)
        lib.prefilter_axis(c, ip, threads=((nl + TG - 1) // TG) * TG, group_size=TG)
    return c


def smooth(vols, fwhm_vox):
    """vols (G, X, Y, Z) smoothed by a Gaussian of `fwhm_vox` voxels (FWHM) on each axis, replicate edges: proto.smooth's
    filter (radius ceil(3 sigma)), as three passes of gauss_axis."""
    if fwhm_vox <= 0:
        return vols
    import math
    sd = fwhm_vox / 2.3548; r = int(math.ceil(3 * sd))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / sd) ** 2); k = (k / k.sum()).to(vols.device)
    G, X, Y, Z = vols.shape
    a, b = vols.contiguous(), torch.empty_like(vols)
    lib = library()
    nv = G * X * Y * Z
    for ax in range(3):
        ip = torch.tensor([G, X, Y, Z, ax, r], dtype=torch.int32, device=vols.device)
        lib.gauss_axis(a, b, k, ip, threads=((nv + TG - 1) // TG) * TG, group_size=TG)
        a, b = b, (a if a is not vols else torch.empty_like(vols))
    return a


def apply_volumes(vols, pvol, ip, fp, mask_output=False):
    """The corrected volumes (G, X, Y, Z): each acquired volume prefiltered and sampled once at phi(x) for every voxel x,
    times the eddy Jacobian, clipped at 0; 0 where phi(x) leaves the field of view. ip, fp from consts() on vols' grid.
    mask_output: also 0 every voxel not valid in all volumes (eddy's default output mask: a voxel some volume's motion
    carried out of the slab holds no data for that volume, and edge-replicated values differ from volume to volume)."""
    coef = prefilter(vols)
    G, X, Y, Z = vols.shape
    out = torch.empty_like(coef); valid = torch.empty_like(coef)
    n = G * X * Y * Z
    library().apply_volumes(coef, pvol, out, ip, fp, valid, threads=((n + TG - 1) // TG) * TG, group_size=TG)
    if mask_output:
        out = out * (valid.amin(0, keepdim=True) > 0)
    return out


def resample(vols, pts, pvol, ip, fp):
    """vols (G, X, Y, Z) sampled trilinearly at phi_g(pts), zero outside -> (G, N); ip, fp from consts() on vols' grid."""
    G, N = vols.shape[0], pts.shape[0]
    out = torch.empty(G, N, dtype=torch.float32, device=vols.device)
    n = G * N
    library().resample(vols, pts, pvol, out, ip, fp, threads=((n + TG - 1) // TG) * TG, group_size=TG)
    return out


def step_moments(coef, obs, pts, pvol, nch, ip, fp, P):
    G = coef.shape[0]
    part = torch.empty(G, nch, NA, dtype=torch.float32, device=coef.device)
    library().step_moments(coef, obs, pts, pvol, part, ip, fp, threads=G * nch * TG, group_size=TG)
    return unpack_H(part, P)


def step_cost(coef, obs, pts, pvol_sub, vols, nch, ip, fp):
    L = vols.shape[0]
    part = torch.empty(L, nch, dtype=torch.float32, device=coef.device)
    library().step_cost(coef, obs, pts, pvol_sub, vols.int().contiguous(), part, ip, fp, threads=L * nch * TG, group_size=TG)
    return part.sum(1)
