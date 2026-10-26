"""tractline.motion_eddy's inner loop and final resample as Metal kernels (torch.mps.compile_shader, as
tractline.ukf_metal). The plain-torch versions of the same pieces are in tractline.motion_eddy (the CPU and CUDA path,
and the float64 reference the tests hold these to).

Per point y of each volume g, with no per-point array in memory:
  - x = phi_g^-1(y): the rigid part in closed form with the displacement along PE (the eddy polynomial plus, when a
    field is given, the field's displacement sampled at x) taken at y, then Newton steps along PE;
  - the prediction's cubic B-spline value and exact gradient at x: the 64 coefficients gathered once each, edges clamped;
  - the residual r = value - obs, and the Jacobian entries by the chain rule through phi^-1 at x:
    J = -[g~ / vox, (dR_j xm) / vox . g~, g~_pe m_k(x)] * scale, g~ = (dphi/dx)^-T grad (motion_eddy._jacobian);
  - step_moments: the upper triangle of J'J (136), J'r (16) and r^2, summed over a threadgroup (simd_sum, then
    threadgroup memory) and written per threadgroup; torch sums the threadgroups' partials in a fixed order, so a run is
    deterministic on a device (no atomics);
  - step_cost: r^2 only, for the volumes listed (the Levenberg-Marquardt trials).
Also: resample (the forward map and a trilinear sample, for the predictions), prefilter_axis (the recursive cubic
B-spline prefilter), gauss_axis (separable Gaussian smoothing) and apply_volumes (the final cubic resample, times the
displacement's Jacobian along PE). float32 throughout (Apple GPUs have no float64).

Moved here from the prototype (bench/motion_eddy/metal_step.py, 2026-10-21) and extended with the field's displacement.
"""
from __future__ import annotations

import math
import torch

TG = 256                                                  # threads per threadgroup (8 SIMD groups of 32)
NH = 136                                                  # upper triangle of a 16 x 16
NA = NH + 16 + 1                                          # + J'r + r^2
NPV = 49                                                  # per volume: R (9), t (3), eddy (10), dR_j (27)
NIP = 13                                                  # the int parameters (consts)

_SRC = r"""
#include <metal_stdlib>
using namespace metal;

#define TG 256
#define NSG (TG / 32)
#define NH 136
#define NA (NH + 16 + 1)
#define NPV 49

// ip: G, N, NCH, X, Y, Z (the sampled volumes' grid), PE, NMON, NPASS, HAS_FIELD, FX, FY, FZ (the field's grid: the full one)
// fp: cen (3), half (3), vox (3), scale (16), the sampled grid's offset in the full grid (3)

static inline float eddy_at(device const float* te, float3 u, int nmon) {
    if (nmon == 0) return 0.0f;
    float e = te[0] + te[1] * u.x + te[2] * u.y + te[3] * u.z;
    if (nmon == 10)
        e += te[4] * u.x * u.x + te[5] * u.y * u.y + te[6] * u.z * u.z + te[7] * u.x * u.y + te[8] * u.x * u.z + te[9] * u.y * u.z;
    return e;
}

// a volume on the full grid, trilinear at x (voxels), the position clamped to the grid
static inline float field_at(device const float* d, float3 x, int X, int Y, int Z) {
    float3 p = clamp(x, float3(0.0f), float3(float(X - 1), float(Y - 1), float(Z - 1)));
    float3 f = floor(p); float3 t = p - f;
    int ix = int(f.x), iy = int(f.y), iz = int(f.z);
    int jx = min(ix + 1, X - 1), jy = min(iy + 1, Y - 1), jz = min(iz + 1, Z - 1);
    float c000 = d[(long(ix) * Y + iy) * Z + iz], c001 = d[(long(ix) * Y + iy) * Z + jz];
    float c010 = d[(long(ix) * Y + jy) * Z + iz], c011 = d[(long(ix) * Y + jy) * Z + jz];
    float c100 = d[(long(jx) * Y + iy) * Z + iz], c101 = d[(long(jx) * Y + iy) * Z + jz];
    float c110 = d[(long(jx) * Y + jy) * Z + iz], c111 = d[(long(jx) * Y + jy) * Z + jz];
    float c00 = c000 + (c001 - c000) * t.z, c01 = c010 + (c011 - c010) * t.z;
    float c10 = c100 + (c101 - c100) * t.z, c11 = c110 + (c111 - c110) * t.z;
    float c0 = c00 + (c01 - c00) * t.y, c1 = c10 + (c11 - c10) * t.y;
    return c0 + (c1 - c0) * t.x;
}

// the gradient of field_at's trilinear interpolant at x (zero along an axis where the position is clamped)
static inline float3 field_grad_at(device const float* d, float3 x, int X, int Y, int Z) {
    float3 hi = float3(float(X - 1), float(Y - 1), float(Z - 1));
    float3 p = clamp(x, float3(0.0f), hi);
    float3 f = floor(p); float3 t = p - f;
    int ix = int(f.x), iy = int(f.y), iz = int(f.z);
    int jx = min(ix + 1, X - 1), jy = min(iy + 1, Y - 1), jz = min(iz + 1, Z - 1);
    float c000 = d[(long(ix) * Y + iy) * Z + iz], c001 = d[(long(ix) * Y + iy) * Z + jz];
    float c010 = d[(long(ix) * Y + jy) * Z + iz], c011 = d[(long(ix) * Y + jy) * Z + jz];
    float c100 = d[(long(jx) * Y + iy) * Z + iz], c101 = d[(long(jx) * Y + iy) * Z + jz];
    float c110 = d[(long(jx) * Y + jy) * Z + iz], c111 = d[(long(jx) * Y + jy) * Z + jz];
    float gx = (1 - t.y) * (1 - t.z) * (c100 - c000) + t.y * (1 - t.z) * (c110 - c010)
             + (1 - t.y) * t.z * (c101 - c001) + t.y * t.z * (c111 - c011);
    float gy = (1 - t.x) * (1 - t.z) * (c010 - c000) + t.x * (1 - t.z) * (c110 - c100)
             + (1 - t.x) * t.z * (c011 - c001) + t.x * t.z * (c111 - c101);
    float gz = (1 - t.x) * (1 - t.y) * (c001 - c000) + t.x * (1 - t.y) * (c101 - c100)
             + (1 - t.x) * t.y * (c011 - c010) + t.x * t.y * (c111 - c110);
    float3 g = float3(gx, gy, gz);
    if (x.x < 0.0f || x.x > hi.x || ix == jx) g.x = 0.0f;
    if (x.y < 0.0f || x.y > hi.y || iy == jy) g.y = 0.0f;
    if (x.z < 0.0f || x.z > hi.z || iz == jz) g.z = 0.0f;
    return g;
}

// d e / d x (voxels per voxel) of the eddy polynomial at x
static inline float3 eddy_grad_at(device const float* te, float3 x, float3 cen, float3 hlf, int nmon) {
    if (nmon == 0) return float3(0.0f);
    float3 u = (x - cen) / hlf;
    float3 g = float3(te[1], te[2], te[3]);
    if (nmon == 10) g += float3(2 * te[4] * u.x + te[7] * u.y + te[8] * u.z, 2 * te[5] * u.y + te[7] * u.x + te[9] * u.z,
                                2 * te[6] * u.z + te[8] * u.x + te[9] * u.y);
    return g / hlf;
}

// the displacement along PE at reference point x (voxels): the eddy polynomial plus the field's
static inline float disp_at(device const float* te, device const float* dfield, float3 x, float3 cen, float3 hlf,
                            int nmon, int has_field, int FX, int FY, int FZ) {
    float e = eddy_at(te, (x - cen) / hlf, nmon);
    if (has_field) e += field_at(dfield, x, FX, FY, FZ);
    return e;
}

// d(disp)/dx_pe at reference point x (voxels per voxel): the eddy polynomial's slope plus the field's
static inline float dpe_at(device const float* te, device const float* ddfield, float3 x, float3 cen, float3 hlf, int pe,
                           int nmon, int has_field, int FX, int FY, int FZ) {
    float3 u = (x - cen) / hlf;
    float d = 0.0f;
    if (nmon > 0) {
        if (pe == 0) d = te[1] + (nmon == 10 ? 2 * te[4] * u.x + te[7] * u.y + te[8] * u.z : 0);
        else if (pe == 1) d = te[2] + (nmon == 10 ? 2 * te[5] * u.y + te[7] * u.x + te[9] * u.z : 0);
        else d = te[3] + (nmon == 10 ? 2 * te[6] * u.z + te[8] * u.x + te[9] * u.y : 0);
        d /= hlf[pe];
    }
    if (has_field) d += field_at(ddfield, x, FX, FY, FZ);
    return d;
}

// x = phi^-1(y), voxels: the rigid part inverted with the displacement at y, then npass Newton steps with the map's
// derivative A + e_pe s e_pe' (s the displacement's slope along PE), by Sherman-Morrison (motion_eddy.phi_inv)
static inline float3 phi_inv(device const float* pv, float3 y, device const float* fp, device const float* dfield,
                             device const float* ddfield, int pe, int nmon, int npass, int has_field, int FX, int FY, int FZ) {
    float3 cen = float3(fp[0], fp[1], fp[2]), hlf = float3(fp[3], fp[4], fp[5]), vox = float3(fp[6], fp[7], fp[8]);
    device const float* R = pv; device const float* t = pv + 9; device const float* te = pv + 12;
    float e = disp_at(te, dfield, y, cen, hlf, nmon, has_field, FX, FY, FZ);
    float3 q = (y - cen) * vox - float3(t[0], t[1], t[2]);
    q[pe] -= e * vox[pe];
    float3 x = float3(R[0] * q.x + R[3] * q.y + R[6] * q.z,                // R^T q (R row-major)
                      R[1] * q.x + R[4] * q.y + R[7] * q.z,
                      R[2] * q.x + R[5] * q.y + R[8] * q.z) / vox + cen;
    float3 ev = float3(0.0f); ev[pe] = vox[pe];
    float3 a = float3(R[0] * ev.x + R[3] * ev.y + R[6] * ev.z, R[1] * ev.x + R[4] * ev.y + R[7] * ev.z,
                      R[2] * ev.x + R[5] * ev.y + R[8] * ev.z) / vox;   // A^-1 e_pe
    for (int k = 0; k < npass; k++) {
        float3 xm = (x - cen) * vox;
        float3 p = float3(R[0] * xm.x + R[1] * xm.y + R[2] * xm.z, R[3] * xm.x + R[4] * xm.y + R[5] * xm.z,
                          R[6] * xm.x + R[7] * xm.y + R[8] * xm.z);
        p = (p + float3(t[0], t[1], t[2])) / vox + cen;
        p[pe] += disp_at(te, dfield, x, cen, hlf, nmon, has_field, FX, FY, FZ);
        float3 r = (p - y) * vox;
        float3 u = float3(R[0] * r.x + R[3] * r.y + R[6] * r.z, R[1] * r.x + R[4] * r.y + R[7] * r.z,
                          R[2] * r.x + R[5] * r.y + R[8] * r.z) / vox;  // A^-1 r
        float s = dpe_at(te, ddfield, x, cen, hlf, pe, nmon, has_field, FX, FY, FZ);
        float den = max(1.0f + s * a[pe], 0.1f);
        x = x - (u - a * (s * u[pe] / den));
    }
    return x;
}

// phi(x), voxels
static inline float3 phi_fwd(device const float* pv, float3 x, device const float* fp, device const float* dfield,
                             int pe, int nmon, int has_field, int FX, int FY, int FZ) {
    float3 cen = float3(fp[0], fp[1], fp[2]), hlf = float3(fp[3], fp[4], fp[5]), vox = float3(fp[6], fp[7], fp[8]);
    float3 xm = (x - cen) * vox;
    float3 p = float3(pv[0] * xm.x + pv[1] * xm.y + pv[2] * xm.z, pv[3] * xm.x + pv[4] * xm.y + pv[5] * xm.z,
                      pv[6] * xm.x + pv[7] * xm.y + pv[8] * xm.z);
    p = (p + float3(pv[9], pv[10], pv[11])) / vox + cen;
    p[pe] += disp_at(pv + 12, dfield, x, cen, hlf, nmon, has_field, FX, FY, FZ);
    return p;
}

static inline void bweights(float t, thread float* w, thread float* dw) {
    float s = 1.0f - t;
    w[0] = s * s * s / 6.0f; w[1] = (3.0f * t * t * t - 6.0f * t * t + 4.0f) / 6.0f;
    w[2] = (-3.0f * t * t * t + 3.0f * t * t + 3.0f * t + 1.0f) / 6.0f; w[3] = t * t * t / 6.0f;
    dw[0] = -s * s / 2.0f; dw[1] = (3.0f * t * t - 4.0f * t) / 2.0f; dw[2] = (-3.0f * t * t + 2.0f * t + 1.0f) / 2.0f; dw[3] = t * t / 2.0f;
}

// the cubic B-spline of one volume's coefficients at x, and its gradient (voxels); coefficient indices clamped
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
    device const float* coef   [[buffer(0)]],    // (G, X, Y, Z) prefiltered prediction (a crop of the full grid)
    device const float* obs    [[buffer(1)]],    // (G, N) the data at the points
    device const float* pts    [[buffer(2)]],    // (N, 3) points y, voxels of the full grid
    device const float* pvol   [[buffer(3)]],    // (G, NPV) per-volume parameters
    device float*       part   [[buffer(4)]],    // (G, NCH, NA) per-threadgroup partial sums
    device const int*   ip     [[buffer(5)]],
    device const float* fp     [[buffer(6)]],
    device const float* dfield [[buffer(7)]],    // the field's displacement along PE (voxels) on the full grid, or a dummy
    device const float* ddfield [[buffer(8)]],   // its derivative along PE (voxels per voxel), or a dummy
    uint tg   [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    int N = ip[1], NCH = ip[2], X = ip[3], Y = ip[4], Z = ip[5], pe = ip[6], nmon = ip[7], npass = ip[8];
    int hf = ip[9], FX = ip[10], FY = ip[11], FZ = ip[12];
    int g = int(tg) / NCH, ch = int(tg) % NCH;
    int n = ch * TG + int(lid);
    threadgroup float sh[NSG * NA];
    float J[16]; float r = 0;
    for (int i = 0; i < 16; i++) J[i] = 0;
    if (n < N) {
        device const float* pv = pvol + long(g) * NPV;
        float3 y = float3(pts[3 * n], pts[3 * n + 1], pts[3 * n + 2]);
        float3 x = phi_inv(pv, y, fp, dfield, ddfield, pe, nmon, npass, hf, FX, FY, FZ);
        float3 gr;
        float v = cubic_vg(coef + long(g) * X * Y * Z, x - float3(fp[25], fp[26], fp[27]), X, Y, Z, gr, true);
        r = v - obs[long(g) * N + n];
        float3 cen = float3(fp[0], fp[1], fp[2]), hlf = float3(fp[3], fp[4], fp[5]), vox = float3(fp[6], fp[7], fp[8]);
        device const float* sc = fp + 9;
        device const float* R = pv;
        // the chain rule through phi^-1, at x: g~ = (dphi/dx)^-T grad, dphi/dx = A + e_pe w' (w the displacement's
        // gradient) by Sherman-Morrison (motion_eddy._jacobian); A^-T v = vox * R (v / vox)
        float3 gq = gr / vox;
        float3 g0 = float3(R[0] * gq.x + R[1] * gq.y + R[2] * gq.z, R[3] * gq.x + R[4] * gq.y + R[5] * gq.z,
                           R[6] * gq.x + R[7] * gq.y + R[8] * gq.z) * vox;
        float3 w = eddy_grad_at(pv + 12, x, cen, hlf, nmon);
        if (hf) w += field_grad_at(dfield, x, FX, FY, FZ);
        float3 wq = w / vox;
        float3 aw = float3(R[0] * wq.x + R[1] * wq.y + R[2] * wq.z, R[3] * wq.x + R[4] * wq.y + R[5] * wq.z,
                           R[6] * wq.x + R[7] * wq.y + R[8] * wq.z) * vox;           // A^-T w
        float den = max(1.0f + aw[pe], 0.1f);
        float3 gt = g0 - aw * (g0[pe] / den);
        float3 gv = gt / vox;
        J[0] = -gv.x * sc[0]; J[1] = -gv.y * sc[1]; J[2] = -gv.z * sc[2];
        float3 xm = (x - cen) * vox;
        for (int j = 0; j < 3; j++) {
            device const float* d = pv + 22 + 9 * j;                         // dR_j, row-major
            float3 dm = float3(d[0] * xm.x + d[1] * xm.y + d[2] * xm.z, d[3] * xm.x + d[4] * xm.y + d[5] * xm.z,
                               d[6] * xm.x + d[7] * xm.y + d[8] * xm.z);
            J[3 + j] = -dot(dm / vox, gt) * sc[3 + j];
        }
        float3 u = (x - cen) / hlf;
        float m[10] = {1.0f, u.x, u.y, u.z, u.x * u.x, u.y * u.y, u.z * u.z, u.x * u.y, u.x * u.z, u.y * u.z};
        float gpe = gt[pe];
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
    device const float* coef   [[buffer(0)]],
    device const float* obs    [[buffer(1)]],
    device const float* pts    [[buffer(2)]],
    device const float* pvol   [[buffer(3)]],    // (len(vols), NPV): the trial parameters of the listed volumes
    device const int*   vols   [[buffer(4)]],    // the listed volumes' indices
    device float*       part   [[buffer(5)]],    // (len(vols), NCH)
    device const int*   ip     [[buffer(6)]],
    device const float* fp     [[buffer(7)]],
    device const float* dfield [[buffer(8)]],
    device const float* ddfield [[buffer(9)]],
    uint tg   [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint sg   [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    int N = ip[1], NCH = ip[2], X = ip[3], Y = ip[4], Z = ip[5], pe = ip[6], nmon = ip[7], npass = ip[8];
    int hf = ip[9], FX = ip[10], FY = ip[11], FZ = ip[12];
    int li = int(tg) / NCH, ch = int(tg) % NCH;
    int g = vols[li];
    int n = ch * TG + int(lid);
    threadgroup float sh[NSG];
    float r2 = 0;
    if (n < N) {
        float3 y = float3(pts[3 * n], pts[3 * n + 1], pts[3 * n + 2]);
        float3 x = phi_inv(pvol + long(li) * NPV, y, fp, dfield, ddfield, pe, nmon, npass, hf, FX, FY, FZ);
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
    device const float* vols   [[buffer(0)]],    // (G, X, Y, Z) the acquired (smoothed) volumes, full grid
    device const float* pts    [[buffer(1)]],    // (N, 3) reference points x, voxels
    device const float* pvol   [[buffer(2)]],    // (G, NPV)
    device float*       out    [[buffer(3)]],    // (G, N)
    device const int*   ip     [[buffer(4)]],
    device const float* fp     [[buffer(5)]],
    device const float* dfield [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    int N = ip[1], X = ip[3], Y = ip[4], Z = ip[5], pe = ip[6], nmon = ip[7];
    int hf = ip[9], FX = ip[10], FY = ip[11], FZ = ip[12];
    int g = int(tid) / N, n = int(tid) % N;
    if (g >= ip[0]) return;
    device const float* pv = pvol + long(g) * NPV;
    float3 x = float3(pts[3 * n], pts[3 * n + 1], pts[3 * n + 2]);
    float3 p = phi_fwd(pv, x, fp, dfield, pe, nmon, hf, FX, FY, FZ);
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
// mirrored edges (Unser 1999); the causal start a truncated sum (|z|^K < 1e-9)
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
    float c0 = d[base];
    float zk = zp;
    int K = min(n - 1, 16);
    for (int k = 1; k <= K; k++) { c0 += zk * d[base + long(k) * stride]; zk *= zp; }
    d[base] = c0;
    for (int k = 1; k < n; k++) d[base + long(k) * stride] += zp * d[base + long(k - 1) * stride];
    long last = base + long(n - 1) * stride;
    d[last] = zp / (zp * zp - 1.0f) * (d[last] + zp * d[last - stride]);
    for (int k = n - 2; k >= 0; k--) d[base + long(k) * stride] = zp * (d[base + long(k + 1) * stride] - d[base + long(k) * stride]);
    for (int k = 0; k < n; k++) d[base + long(k) * stride] *= 6.0f;
}

// one axis of a separable Gaussian, replicate edges: dst = sum_k w[k] src[clamp(i + k - r)]
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
// times the displacement's Jacobian along PE (1 + the eddy polynomial's and the field's derivatives), clipped at 0
// (susceptibility.apply's clip); coefficient indices clamped; 0 where phi(x) leaves the field of view
kernel void apply_volumes(
    device const float* coef    [[buffer(0)]],    // (G, X, Y, Z) prefiltered acquired volumes
    device const float* pvol    [[buffer(1)]],    // (G, NPV)
    device float*       out     [[buffer(2)]],    // (G, X, Y, Z)
    device const int*   ip      [[buffer(3)]],
    device const float* fp      [[buffer(4)]],
    device float*       valid   [[buffer(5)]],    // (G, X, Y, Z): 1 where the sample is inside the field of view, else 0
    device const float* dfield  [[buffer(6)]],    // the field's displacement along PE (voxels), or a dummy
    device const float* ddfield [[buffer(7)]],    // its derivative along PE (voxels per voxel), or a dummy
    uint tid [[thread_position_in_grid]])
{
    int G = ip[0], X = ip[3], Y = ip[4], Z = ip[5], pe = ip[6], nmon = ip[7];
    int hf = ip[9], FX = ip[10], FY = ip[11], FZ = ip[12];
    long nv = long(X) * Y * Z;
    if (long(tid) >= long(G) * nv) return;
    int g = int(long(tid) / nv); long v = long(tid) % nv;
    float3 x = float3(float(v / (long(Y) * Z)), float((v / Z) % Y), float(v % Z));
    device const float* pv = pvol + long(g) * NPV;
    float3 cen = float3(fp[0], fp[1], fp[2]), hlf = float3(fp[3], fp[4], fp[5]);
    float3 p = phi_fwd(pv, x, fp, dfield, pe, nmon, hf, FX, FY, FZ);
    float jac = 1.0f + dpe_at(pv + 12, ddfield, x, cen, hlf, pe, nmon, hf, FX, FY, FZ);
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


def _threads(n):
    return ((n + TG - 1) // TG) * TG


def unpack_H(part, P):
    """(G, NCH, NA) partials -> H (G, P, P), b (G, P), f (G,): summed over the threadgroups in a fixed order."""
    s = part.sum(1)
    G = s.shape[0]
    H = torch.zeros(G, 16, 16, dtype=s.dtype, device=s.device)
    iu = torch.triu_indices(16, 16, device=s.device)
    H[:, iu[0], iu[1]] = s[:, :NH]
    H = H + H.transpose(1, 2) - torch.diag_embed(torch.diagonal(H, dim1=1, dim2=2))
    return H[:, :P, :P], s[:, NH:NH + P], s[:, NH + 16]


def consts(geo, G, N, grid, offset=(0, 0, 0), npass=2, has_field=False):
    """(nch, ip, fp) for the kernels. geo: the Geometry (motion_eddy.Geometry); grid: the shape of the volumes a kernel
    samples (a crop of the full grid, `offset` its first voxel). The kernels index threads in 32 bits: refused above."""
    nch = (N + TG - 1) // TG
    X, Y, Z = grid
    if max(G * N, G * X * Y * Z, G * nch * TG) >= 2 ** 31:
        raise ValueError(f"motion_eddy_metal: {G} volumes of {X} x {Y} x {Z} exceed the kernels' 32-bit indexing")
    FX, FY, FZ = geo.shape
    ip = torch.tensor([G, N, nch, X, Y, Z, geo.pe_axis, geo.nmon, npass, int(has_field), FX, FY, FZ],
                      dtype=torch.int32, device=geo.device)
    sc = torch.zeros(16, dtype=torch.float32)
    sc[:geo.P] = geo.scale.float().cpu()
    fp = torch.cat([geo.cen.float().cpu(), geo.half.float().cpu(), geo.vox.float().cpu(), sc,
                    torch.tensor(offset, dtype=torch.float32)]).to(geo.device)
    return nch, ip, fp


def prefilter(vols):
    """Cubic B-spline coefficients of vols (G, X, Y, Z), mirrored edges, by the recursive filter on three axes (a copy)."""
    c = vols.contiguous().clone()
    G, X, Y, Z = c.shape
    lib = library()
    for ax, nl in ((2, G * X * Y), (1, G * X * Z), (0, G * Y * Z)):
        ip = torch.tensor([G, X, Y, Z, ax], dtype=torch.int32, device=c.device)
        lib.prefilter_axis(c, ip, threads=_threads(nl), group_size=TG)
    return c


def smooth(vols, fwhm_vox):
    """vols (G, X, Y, Z) smoothed by a Gaussian of `fwhm_vox` voxels (FWHM) on each axis, replicate edges (radius
    ceil(3 sigma)), as three passes of gauss_axis."""
    if fwhm_vox <= 0:
        return vols
    sd = fwhm_vox / 2.3548
    r = int(math.ceil(3 * sd))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, dtype=torch.float32) / sd) ** 2)
    k = (k / k.sum()).to(vols.device)
    G, X, Y, Z = vols.shape
    a, b = vols.contiguous(), torch.empty_like(vols)
    lib = library()
    nv = G * X * Y * Z
    for ax in range(3):
        ip = torch.tensor([G, X, Y, Z, ax, r], dtype=torch.int32, device=vols.device)
        lib.gauss_axis(a, b, k, ip, threads=_threads(nv), group_size=TG)
        a, b = b, (a if a is not vols else torch.empty_like(vols))
    return a


def apply_volumes(vols, pvol, ip, fp, dfield, ddfield):
    """The corrected volumes and their validity (each (G, X, Y, Z)): each acquired volume prefiltered and sampled once at
    phi(x) for every voxel x, times the displacement's Jacobian along PE, clipped at 0; 0 (and not valid) where phi(x)
    leaves the field of view."""
    coef = prefilter(vols)
    G, X, Y, Z = vols.shape
    out = torch.empty_like(coef)
    valid = torch.empty_like(coef)
    library().apply_volumes(coef, pvol, out, ip, fp, valid, dfield, ddfield, threads=_threads(G * X * Y * Z),
                            group_size=TG)
    return out, valid


def resample(vols, pts, pvol, ip, fp, dfield):
    """vols (G, X, Y, Z) sampled trilinearly at phi_g(pts), zero outside -> (G, N)."""
    G, N = vols.shape[0], pts.shape[0]
    out = torch.empty(G, N, dtype=torch.float32, device=vols.device)
    library().resample(vols, pts, pvol, out, ip, fp, dfield, threads=_threads(G * N), group_size=TG)
    return out


def step_moments(coef, obs, pts, pvol, nch, ip, fp, dfield, ddfield, P):
    G = coef.shape[0]
    part = torch.empty(G, nch, NA, dtype=torch.float32, device=coef.device)
    library().step_moments(coef, obs, pts, pvol, part, ip, fp, dfield, ddfield, threads=G * nch * TG, group_size=TG)
    return unpack_H(part, P)


def step_cost(coef, obs, pts, pvol_sub, vols, nch, ip, fp, dfield, ddfield):
    L = vols.shape[0]
    part = torch.empty(L, nch, dtype=torch.float32, device=coef.device)
    library().step_cost(coef, obs, pts, pvol_sub, vols.int().contiguous(), part, ip, fp, dfield, ddfield,
                        threads=L * nch * TG, group_size=TG)
    return part.sum(1)
