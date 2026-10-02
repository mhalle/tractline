"""UKF tractography's step as one Metal kernel, compiled with torch.mps.compile_shader (torch >= 2.7):
a drop-in for _ukf_torch.advance on the 'mps' device, in float32 (Apple GPUs have no float64).

LPF lanes of a SIMD group per half-fiber (default 8: four half-fibers to a SIMD group). Every lane
carries the filter's small algebra (the 10-state, its 10x10 covariance, the 21 sigma points, two
10x10 inverses) redundantly; the gradients are split across the fiber's lanes (lane l takes
gradients l, l+LPF, ...), which is where the work is: the 3x3x3 Gaussian interpolation reads N values
per voxel, contiguous because the signal is stored volumes-fastest, and the predicted signal is
21 sigma points x N gradients of exp(). The two sums over gradients in the information-form update,
and the mean signal of the stopping check, are sums across the fiber's lanes (simd_sum at 32 lanes,
a fixed shuffle-xor butterfly below).

The rules are _ukf_torch's, which are the binary's; three choices are this kernel's own:
  - the binary measures 2N rows, each gradient and its negation; u'Du is the same for both, so the
    kernel measures N and doubles the two sums over rows (the mean is unchanged). Exact in arithmetic,
    different in rounding order.
  - Cholesky and the two inverses are written out (Cholesky-Banachiewicz; the inverses through
    Cholesky by default, Gauss-Jordan with partial pivoting without SPDINV), not LAPACK's.
  - sums run in a fixed order (sigma points 0..20; within a lane gradients in order; across lanes the
    fixed reduction); precise:: math and no FMA contraction, so the kernel is deterministic on a device.

Compile-time options (`defines`), measured on an M2 with HARDI (ukf_metal.json):
  LPF=32|16|8|4   lanes per half-fiber; 8 is fastest (at 4 the per-lane arrays spill)
  ZCACHE          keep the sigma points' predicted signal between the two passes instead of
                  recomputing it: same arithmetic, bit-identical output, +12 %
  MATHNS=fast     fast:: instead of precise:: exp, sqrt, divide, acos: +15-20 %, errors still at
                  float32's level, but not the same bits; off by default
  SIGHALF         the signal read as float16 (the caller passes it so): +1.4 % speed for 5x the
                  per-step error (1.7e-6); off
  SFORM           the update with Ht never formed (S = sum Pxz Pxz', I = 2/Rs Yk S Yk), as the Triton
                  kernel does: the same accuracy, no faster here (195 k against 207 k); off
  SPDINV          both 10x10 inverses (Pm, Yk + I: symmetric positive definite) through Cholesky,
                  L^-T L^-1 with packed triangles, instead of Gauss-Jordan: per-step state error
                  against float64 3.5e-7 instead of 1.3e-5, and faster (207 k against 183 k steps/s)
Tested against the float32 torch step and float64 on captured fixtures (ukf_metal_check.py).
"""
from __future__ import annotations

import torch

MAX_GRADIENTS = 256                       # the kernel's per-lane arrays hold N <= 256

_SRC = r"""
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#define NS 10
#define NSIG 21
#ifndef LPF
#define LPF 32
#endif
#define MAXG (256 / LPF)

// a sum over the LPF lanes of one fiber; a fixed butterfly, so the same on every launch
static inline float lanesum(float v) {
#if LPF == 32
    return simd_sum(v);
#else
    for (ushort o = LPF / 2; o > 0; o >>= 1) v = v + simd_shuffle_xor(v, o);
    return v;
#endif
}

#ifndef MATHNS
#define MATHNS precise
#endif
static inline float pdiv(float a, float b) { return MATHNS::divide(a, b); }

static inline float l2fa(float l1, float l2) {          // l2 == l3
    return pdiv(fabs(l1 - l2), MATHNS::sqrt(l1 * l1 + 2.0f * l2 * l2));
}

// F: normalize both directions (times the reciprocal, as _ukf_torch.F), clamp the four eigenvalues
static inline void Ffun(thread float* X) {
    for (int o = 0; o < 10; o += 5) {
        float r = pdiv(1.0f, MATHNS::sqrt(X[o] * X[o] + X[o + 1] * X[o + 1] + X[o + 2] * X[o + 2]));
        X[o] = X[o] * r; X[o + 1] = X[o + 1] * r; X[o + 2] = X[o + 2] * r;
        X[o + 3] = max(X[o + 3], 100.0f); X[o + 4] = max(X[o + 4], 100.0f);
    }
}

// the two tensors of a state, as 3x3 symmetric D (xx xy xz yy yz zz), each times 1e-6 (filter_Simple2T::H)
static inline void tensors(thread const float* X, thread float* Dm) {
    for (int t = 0; t < 2; t++) {
        int o = 5 * t;
        float n = MATHNS::sqrt(X[o] * X[o] + X[o + 1] * X[o + 1] + X[o + 2] * X[o + 2]);
        float m0 = pdiv(X[o], n), m1 = pdiv(X[o + 1], n), m2 = pdiv(X[o + 2], n);
        if (m0 < 0.0f) { m0 = -m0; m1 = -m1; m2 = -m2; }
        float l1 = max(X[o + 3], 100.0f), l2 = max(X[o + 4], 100.0f);
        float R[3][3] = {{m0, m1, m2},
                         {m1, pdiv(m1 * m1, 1.0f + m0) - 1.0f, pdiv(m1 * m2, 1.0f + m0)},
                         {m2, pdiv(m1 * m2, 1.0f + m0), pdiv(m2 * m2, 1.0f + m0) - 1.0f}};
        float L[3] = {l1, l2, l2};
        float D[3][3];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) {
            float s = 0.0f;
            for (int k = 0; k < 3; k++) s = s + R[i][k] * L[k] * R[j][k];
            D[i][j] = s * 1e-6f;
        }
        thread float* d = Dm + 9 * t;
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) d[3 * i + j] = D[i][j];
    }
}

// H for one gradient: 1/2 exp(-b u'D1u) + 1/2 exp(-b u'D2u), u'Du in _ukf_torch.H's order
static inline float Hn(thread const float* Dm, float u0, float u1, float u2, float b) {
    float out = 0.0f;
    for (int t = 0; t < 2; t++) {
        thread const float* d = Dm + 9 * t;
        float q = 0.0f;
        float u[3] = {u0, u1, u2};
        for (int r = 0; r < 3; r++) q = q + u[r] * (d[3 * r] * u0 + d[3 * r + 1] * u1 + d[3 * r + 2] * u2);
        out = out + MATHNS::exp(-b * q) * 0.5f;
    }
    return out;
}

// in-place inverse of a 10x10 (Gauss-Jordan, partial pivoting), A row-major
static inline void inv10(thread float* A) {
    float B[NS * NS];
    for (int i = 0; i < NS * NS; i++) B[i] = 0.0f;
    for (int i = 0; i < NS; i++) B[i * NS + i] = 1.0f;
    for (int c = 0; c < NS; c++) {
        int p = c; float best = fabs(A[c * NS + c]);
        for (int r = c + 1; r < NS; r++) { float v = fabs(A[r * NS + c]); if (v > best) { best = v; p = r; } }
        if (p != c) for (int k = 0; k < NS; k++) {
            float t = A[c * NS + k]; A[c * NS + k] = A[p * NS + k]; A[p * NS + k] = t;
            t = B[c * NS + k]; B[c * NS + k] = B[p * NS + k]; B[p * NS + k] = t;
        }
        float piv = A[c * NS + c];
        for (int k = 0; k < NS; k++) { A[c * NS + k] = pdiv(A[c * NS + k], piv); B[c * NS + k] = pdiv(B[c * NS + k], piv); }
        for (int r = 0; r < NS; r++) if (r != c) {
            float f = A[r * NS + c];
            for (int k = 0; k < NS; k++) { A[r * NS + k] = A[r * NS + k] - f * A[c * NS + k]; B[r * NS + k] = B[r * NS + k] - f * B[c * NS + k]; }
        }
    }
    for (int i = 0; i < NS * NS; i++) A[i] = B[i];
}

// in-place inverse of a symmetric positive definite 10x10 through Cholesky: A^-1 = L^-T L^-1
// (the lower triangle is read; the full inverse is written, symmetric by construction). L and
// M = L^-1 are kept as packed lower triangles (55 floats each), which keeps the kernel out of spills.
#define TRI(i, j) ((i) * ((i) + 1) / 2 + (j))
static inline void spd_inv10(thread float* A) {
    float L[55], M[55];
    for (int j = 0; j < NS; j++) {
        float d = A[j * NS + j];
        for (int k = 0; k < j; k++) d = d - L[TRI(j, k)] * L[TRI(j, k)];
        float ljj = MATHNS::sqrt(d);
        L[TRI(j, j)] = ljj;
        for (int i = j + 1; i < NS; i++) {
            float v = A[i * NS + j];
            for (int k = 0; k < j; k++) v = v - L[TRI(i, k)] * L[TRI(j, k)];
            L[TRI(i, j)] = pdiv(v, ljj);
        }
    }
    for (int i = 0; i < NS; i++) {                            // M = L^-1, forward substitution row by row
        M[TRI(i, i)] = pdiv(1.0f, L[TRI(i, i)]);
        for (int j = 0; j < i; j++) {
            float acc = 0.0f;
            for (int k = j; k < i; k++) acc = acc + L[TRI(i, k)] * M[TRI(k, j)];
            M[TRI(i, j)] = pdiv(-acc, L[TRI(i, i)]);
        }
    }
    for (int i = 0; i < NS; i++)                               // (M' M)_ij = sum_{k >= max(i, j)} M_ki M_kj
        for (int j = 0; j <= i; j++) {
            float acc = 0.0f;
            for (int k = i; k < NS; k++) acc = acc + M[TRI(k, i)] * M[TRI(k, j)];
            A[i * NS + j] = acc; A[j * NS + i] = acc;
        }
}

#ifdef SPDINV
#define INV10 spd_inv10
#else
#define INV10 inv10
#endif

static inline float cround(float x) { return x < 0.0f ? -floor(-x + 0.5f) : floor(x + 0.5f); }

kernel void ukf_step(
#ifdef SIGHALF
    device const half*  A       [[buffer(0)]],    // (nk, nj, ni, N) normalized signal, float16
#else
    device const float* A       [[buffer(0)]],    // (nk, nj, ni, N) normalized signal
#endif
    device const char*  mask    [[buffer(1)]],    // (nk, nj, ni) signed char
    device const float* g       [[buffer(2)]],    // (N, 3) gradients, voxel frame (k, j, i order as _ukf_torch)
    device const float* bval    [[buffer(3)]],    // (N,)
    device float*       x       [[buffer(4)]],    // (B, 3) position (k, j, i), updated
    device float*       st      [[buffer(5)]],    // (B, 10) state, updated
    device float*       P       [[buffer(6)]],    // (B, 100) covariance, updated
    device const float* old     [[buffer(7)]],    // (B, 3) previous direction
    device float*       dir     [[buffer(8)]],    // (B, 3) out: new direction m1
    device int*         flags   [[buffer(9)]],    // (B,) out: stop | swap<<1 | swap2<<2 | inside<<3
    device float*       faout   [[buffer(10)]],   // (B,) out
    device float*       msig    [[buffer(11)]],   // (B,) out
    device const int*   ip      [[buffer(12)]],   // B, N, nk, nj, ni, step, max_steps
    device const float* fp      [[buffer(13)]],   // vox0, vox1, vox2, sigma, step_length, stop_fa, stop_thr, Rs, Qm, Ql
    uint tid  [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int B = ip[0], N = ip[1], nk = ip[2], nj = ip[3], ni = ip[4], step = ip[5], max_steps = ip[6];
    const int f = int(tid / uint(LPF));
    if (f >= B) return;                                       // whole fibers only (threads = LPF B)
    const float vox0 = fp[0], vox1 = fp[1], vox2 = fp[2], sigma = fp[3], step_length = fp[4];
    const float stop_fa = fp[5], stop_thr = fp[6], Rs = fp[7], Qm = fp[8], Ql = fp[9];
    const int L = int(lane) % LPF;
    int ng = 0; int gidx[MAXG];
    for (int n = L; n < N; n += LPF) gidx[ng++] = n;

    float p0 = x[3 * f], p1 = x[3 * f + 1], p2 = x[3 * f + 2];
    float s[NS], Pc[NS * NS], o3[3];
    for (int i = 0; i < NS; i++) s[i] = st[NS * f + i];
    for (int i = 0; i < NS * NS; i++) Pc[i] = P[NS * NS * f + i];
    for (int i = 0; i < 3; i++) o3[i] = old[3 * f + i];

    // ---- interp (Interp3Signal): this lane's gradients
    float z[MAXG];
    for (int k = 0; k < MAXG; k++) z[k] = 0.0f;
    float wsum = 1e-16f;
    int r0 = int(cround(p0)), r1 = int(cround(p1)), r2 = int(cround(p2));
    for (int xx = -1; xx <= 1; xx++) {
        int X0 = r0 + xx; bool okx = X0 >= 0 && X0 < nk; float dx = (float(X0) - p0) * vox0;
        for (int yy = -1; yy <= 1; yy++) {
            int Y0 = r1 + yy; bool oky = Y0 >= 0 && Y0 < nj; float dy = (float(Y0) - p1) * vox1;
            for (int zz = -1; zz <= 1; zz++) {
                int Z0 = r2 + zz; bool ok = okx && oky && Z0 >= 0 && Z0 < ni; float dz = (float(Z0) - p2) * vox2;
                float w = MATHNS::exp(pdiv(-(dx * dx + dy * dy + dz * dz), sigma));
                if (!ok) w = 0.0f;
                long v = ((long(clamp(X0, 0, nk - 1)) * nj + clamp(Y0, 0, nj - 1)) * ni + clamp(Z0, 0, ni - 1)) * N;
                for (int k = 0; k < ng; k++) z[k] = z[k] + w * float(A[v + gidx[k]]);
                wsum = wsum + w;
            }
        }
    }
    for (int k = 0; k < ng; k++) z[k] = pdiv(z[k], wsum);

    // ---- the unscented step, information form
    const float kap = 0.01f, nk_ = float(NS) + kap;
    const float W0 = pdiv(kap, nk_), Wi = pdiv(0.5f, nk_);
    float Lc[NS * NS];                                         // Cholesky, lower: P = Lc Lc'
    for (int i = 0; i < NS * NS; i++) Lc[i] = 0.0f;
    for (int j = 0; j < NS; j++) {
        float d = Pc[j * NS + j];
        for (int k = 0; k < j; k++) d = d - Lc[j * NS + k] * Lc[j * NS + k];
        float ljj = MATHNS::sqrt(d);
        Lc[j * NS + j] = ljj;
        for (int i = j + 1; i < NS; i++) {
            float v = Pc[i * NS + j];
            for (int k = 0; k < j; k++) v = v - Lc[i * NS + k] * Lc[j * NS + k];
            Lc[i * NS + j] = pdiv(v, ljj);
        }
    }
    const float c = MATHNS::sqrt(nk_);
    // sigma point si: x, x + c Lc[:, j], x - c Lc[:, j]; F applied
    #define SIGMA(si, OUT) { for (int q_ = 0; q_ < NS; q_++) OUT[q_] = s[q_]; \
        if ((si) >= 1) { int j_ = ((si) - 1) % NS; float sg_ = (si) <= NS ? 1.0f : -1.0f; \
            for (int q_ = 0; q_ < NS; q_++) OUT[q_] = s[q_] + sg_ * (c * Lc[q_ * NS + j_]); } Ffun(OUT); }

    float xh[NS];
    for (int i = 0; i < NS; i++) xh[i] = 0.0f;
    for (int si = 0; si < NSIG; si++) {
        float Xs[NS]; SIGMA(si, Xs);
        float w = si == 0 ? W0 : Wi;
        for (int i = 0; i < NS; i++) xh[i] = xh[i] + w * Xs[i];
    }
    float Pm[NS * NS];
    for (int i = 0; i < NS * NS; i++) Pm[i] = 0.0f;
    for (int si = 0; si < NSIG; si++) {
        float Xs[NS]; SIGMA(si, Xs);
        float w = si == 0 ? W0 : Wi;
        for (int i = 0; i < NS; i++) Xs[i] = Xs[i] - xh[i];
        for (int i = 0; i < NS; i++) for (int j = 0; j < NS; j++) Pm[i * NS + j] = Pm[i * NS + j] + Xs[i] * w * Xs[j];
    }
    for (int i = 0; i < NS; i++) Pm[i * NS + i] = Pm[i * NS + i] + ((i % 5) < 3 ? Qm : Ql);
    INV10(Pm);                                                 // Pm is now Yk
    float yh[NS];
    for (int i = 0; i < NS; i++) { float v = 0.0f; for (int j = 0; j < NS; j++) v = v + Pm[i * NS + j] * xh[j]; yh[i] = v; }

    float zh[MAXG];
    for (int k = 0; k < MAXG; k++) zh[k] = 0.0f;
    float u[MAXG][3], bb[MAXG];
    for (int k = 0; k < ng; k++) { int n = gidx[k]; u[k][0] = g[3 * n]; u[k][1] = g[3 * n + 1]; u[k][2] = g[3 * n + 2]; bb[k] = bval[n]; }
#ifdef ZCACHE
    float Zc[NSIG][MAXG];
#endif
    for (int si = 0; si < NSIG; si++) {
        float Xs[NS]; SIGMA(si, Xs);
        float Dm[18]; tensors(Xs, Dm);
        float w = si == 0 ? W0 : Wi;
        for (int k = 0; k < ng; k++) {
            float h = Hn(Dm, u[k][0], u[k][1], u[k][2], bb[k]);
#ifdef ZCACHE
            Zc[si][k] = h;
#endif
            zh[k] = zh[k] + w * h;
        }
    }
    float Pxz[NS][MAXG];
    for (int i = 0; i < NS; i++) for (int k = 0; k < MAXG; k++) Pxz[i][k] = 0.0f;
    for (int si = 0; si < NSIG; si++) {
        float Xs[NS]; SIGMA(si, Xs);
        float w = si == 0 ? W0 : Wi;
#ifndef ZCACHE
        float Dm[18]; tensors(Xs, Dm);
#endif
        for (int i = 0; i < NS; i++) Xs[i] = Xs[i] - xh[i];
        for (int k = 0; k < ng; k++) {
#ifdef ZCACHE
            float dz = Zc[si][k] - zh[k];
#else
            float dz = Hn(Dm, u[k][0], u[k][1], u[k][2], bb[k]) - zh[k];
#endif
            for (int i = 0; i < NS; i++) Pxz[i][k] = Pxz[i][k] + Xs[i] * w * dz;
        }
    }
    const float rr = pdiv(1.0f, Rs);
    float Ip[NS * NS], iv[NS];
#ifdef SFORM
    // the same update with Ht never formed: S = sum_n Pxz_n Pxz_n' (packed), v = sum_n Pxz_n term_n,
    // then I = 2/Rs Yk S Yk and i = 2/Rs Yk v (Yk symmetric) - 65 terms per gradient instead of 210
    float Sp[55], vv[NS];
    for (int i = 0; i < 55; i++) Sp[i] = 0.0f;
    for (int i = 0; i < NS; i++) vv[i] = 0.0f;
    for (int k = 0; k < ng; k++) {
        float py = 0.0f;
        for (int i = 0; i < NS; i++) py = py + Pxz[i][k] * yh[i];
        float term = (z[k] - zh[k]) + py;
        for (int i = 0; i < NS; i++) {
            vv[i] = vv[i] + Pxz[i][k] * term;
            for (int j = 0; j <= i; j++) Sp[TRI(i, j)] = Sp[TRI(i, j)] + Pxz[i][k] * Pxz[j][k];
        }
    }
    for (int i = 0; i < 55; i++) Sp[i] = lanesum(Sp[i]);
    for (int i = 0; i < NS; i++) vv[i] = lanesum(vv[i]);
    for (int i = 0; i < NS; i++) {                                   // row i of T = Yk S, then of Yk + 2/Rs T Yk
        float Trow[NS];
        for (int j = 0; j < NS; j++) {
            float t = 0.0f;
            for (int k = 0; k < NS; k++) t = t + Pm[i * NS + k] * Sp[k >= j ? TRI(k, j) : TRI(j, k)];
            Trow[j] = t;
        }
        for (int j = 0; j < NS; j++) {
            float t = 0.0f;
            for (int k = 0; k < NS; k++) t = t + Trow[k] * Pm[k * NS + j];
            Ip[i * NS + j] = Pm[i * NS + j] + 2.0f * (rr * t);
        }
        float u = 0.0f;
        for (int j = 0; j < NS; j++) u = u + Pm[i * NS + j] * vv[j];
        iv[i] = 2.0f * (rr * u);
    }
#else
    // Ht = Yk Pxz; I = 2/Rs sum_n Ht Ht'; i = 2/Rs sum_n Ht ((z - zh) + Pxz' yh)
    for (int i = 0; i < NS * NS; i++) Ip[i] = 0.0f;
    for (int i = 0; i < NS; i++) iv[i] = 0.0f;
    for (int k = 0; k < ng; k++) {
        float Ht[NS];
        for (int i = 0; i < NS; i++) { float v = 0.0f; for (int j = 0; j < NS; j++) v = v + Pm[i * NS + j] * Pxz[j][k]; Ht[i] = v; }
        float py = 0.0f;
        for (int i = 0; i < NS; i++) py = py + Pxz[i][k] * yh[i];
        float term = (z[k] - zh[k]) + py;
        for (int i = 0; i < NS; i++) {
            iv[i] = iv[i] + Ht[i] * rr * term;
            for (int j = 0; j < NS; j++) Ip[i * NS + j] = Ip[i * NS + j] + Ht[i] * rr * Ht[j];
        }
    }
    for (int i = 0; i < NS * NS; i++) Ip[i] = 2.0f * lanesum(Ip[i]);
    for (int i = 0; i < NS; i++) iv[i] = 2.0f * lanesum(iv[i]);
    for (int i = 0; i < NS * NS; i++) Ip[i] = Pm[i] + Ip[i];       // Yk + I
#endif
    INV10(Ip);                                                      // Pn
    for (int i = 0; i < NS; i++) { float v = 0.0f; for (int j = 0; j < NS; j++) v = v + Ip[i * NS + j] * (iv[j] + yh[j]); s[i] = v; }
    for (int i = 0; i < NS * NS; i++) Pc[i] = Ip[i];

    // ---- State2Tensor2T, the two swaps, FA, the Euler step (tractography.cc)
    float m1[3], m2[3];
    { float n1 = MATHNS::sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2]), n2 = MATHNS::sqrt(s[5] * s[5] + s[6] * s[6] + s[7] * s[7]);
      for (int i = 0; i < 3; i++) { m1[i] = pdiv(s[i], n1); m2[i] = pdiv(s[5 + i], n2); } }
    float L1a = max(s[3], 100.0f), L1b = max(s[4], 100.0f), L2a = max(s[8], 100.0f), L2b = max(s[9], 100.0f);
    if (m1[0] * o3[0] + m1[1] * o3[1] + m1[2] * o3[2] < 0.0f) for (int i = 0; i < 3; i++) m1[i] = -m1[i];
    if (m2[0] * o3[0] + m2[1] * o3[1] + m2[2] * o3[2] < 0.0f) for (int i = 0; i < 3; i++) m2[i] = -m2[i];
    float fat1 = l2fa(L1a, L1b), fat2 = l2fa(L2a, L2b);
    float angle = MATHNS::acos(m1[0] * m2[0] + m1[1] * m2[1] + m1[2] * m2[2]) * (180.0f / M_PI_F);
    bool sw = (m1[0] * o3[0] + m1[1] * o3[1] + m1[2] * o3[2]) < (m2[0] * o3[0] + m2[1] * o3[1] + m2[2] * o3[2]);
    #define SWAPALL { for (int i = 0; i < 3; i++) { float t = m1[i]; m1[i] = m2[i]; m2[i] = t; } \
        float t = L1a; L1a = L2a; L2a = t; t = L1b; L1b = L2b; L2b = t; \
        for (int i = 0; i < 5; i++) { float t2 = s[i]; s[i] = s[5 + i]; s[5 + i] = t2; } \
        float Q2[NS * NS]; for (int i = 0; i < NS; i++) for (int j = 0; j < NS; j++) Q2[i * NS + j] = Pc[((i + 5) % NS) * NS + (j + 5) % NS]; \
        for (int i = 0; i < NS * NS; i++) Pc[i] = Q2[i]; }
    if (sw) { SWAPALL; float t = fat1; fat1 = fat2; fat2 = t; }
    bool sw2 = (angle <= 20.0f) && (min(fat1, fat2) <= 0.2f) && !(fat1 > 0.2f);
    if (sw2) { SWAPALL; }
    float fa = L1a < L1b ? 0.0f : l2fa(L1a, L1b);
    p0 = p0 + pdiv(m1[2], vox0) * step_length;
    p1 = p1 + pdiv(m1[1], vox1) * step_length;
    p2 = p2 + pdiv(m1[0], vox2) * step_length;

    // ---- the checks at the new position: the mean of H(state), the mask, FA, the step count
    float Dm[18]; tensors(s, Dm);
    float hs = 0.0f;
    for (int k = 0; k < ng; k++) hs = hs + Hn(Dm, u[k][0], u[k][1], u[k][2], bb[k]);
    float mean_sig = pdiv(lanesum(hs), float(N));
    int q0 = int(cround(p0)), q1 = int(cround(p1)), q2 = int(cround(p2));
    bool inside = q0 >= 0 && q0 < nk && q1 >= 0 && q1 < nj && q2 >= 0 && q2 < ni && mask[(long(q0) * nj + q1) * ni + q2] > 0;
    bool stop = !inside || mean_sig < stop_thr || fa < stop_fa || step > max_steps;

    if (L == 0) {
        x[3 * f] = p0; x[3 * f + 1] = p1; x[3 * f + 2] = p2;
        for (int i = 0; i < NS; i++) st[NS * f + i] = s[i];
        for (int i = 0; i < NS * NS; i++) P[NS * NS * f + i] = Pc[i];
        for (int i = 0; i < 3; i++) dir[3 * f + i] = m1[i];
        flags[f] = int(stop) | (int(sw) << 1) | (int(sw2) << 2) | (int(inside) << 3);
        faout[f] = fa; msig[f] = mean_sig;
    }
}
"""

_LIBS: dict = {}
#: the defaults: 8 lanes per half-fiber, H cached between passes, Cholesky inverses, precise math
DEFINES: tuple = ("ZCACHE", "LPF=8", "SPDINV")


def available() -> bool:
    return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


def library(defines: tuple | None = None):
    """The compiled kernel; `defines` (e.g. ("ZCACHE", "MATHNS=fast")) are prepended as #defines."""
    defines = DEFINES if defines is None else tuple(defines)
    if defines not in _LIBS:
        head = "".join(f"#define {d.replace('=', ' ', 1)}\n" for d in defines)
        _LIBS[defines] = torch.mps.compile_shader(head + _SRC)
    return _LIBS[defines]


@torch.no_grad()
def advance(D: dict, xa, sa, Pa, oa, Q, Rs, step, max_steps, step_length=0.3, stopping_fa=0.08,
            stopping_threshold=0.06, group_size=256, defines=None):
    """_ukf_torch.advance's contract on the 'mps' device, float32: returns (x, state, P, direction,
    stop, info). D is _ukf_torch.at_dtype(D, torch.float32, 'mps')."""
    dev = xa.device
    N = int(D["N"])
    if N > MAX_GRADIENTS:
        raise ValueError(f"_ukf_metal: {N} gradients; the kernel takes at most {MAX_GRADIENTS}")
    defs = DEFINES if defines is None else tuple(defines)
    lpf = next((int(d.split("=")[1]) for d in defs if d.startswith("LPF=")), 32)
    B = xa.shape[0]
    nk, nj, ni = (int(v) for v in D["dim"])
    x = xa.contiguous().clone(); s = sa.contiguous().clone(); P = Pa.reshape(B, -1).contiguous().clone()
    o = oa.contiguous()
    d = torch.empty((B, 3), dtype=torch.float32, device=dev)
    flags = torch.empty(B, dtype=torch.int32, device=dev)
    fa = torch.empty(B, dtype=torch.float32, device=dev); ms = torch.empty(B, dtype=torch.float32, device=dev)
    q = torch.diagonal(Q).cpu()
    vox = D["voxel"].cpu()
    ip = torch.tensor([B, N, nk, nj, ni, int(step), int(max_steps)], dtype=torch.int32, device=dev)
    fp = torch.tensor([float(vox[0]), float(vox[1]), float(vox[2]), float(vox.min()), step_length, stopping_fa,
                       stopping_threshold, float(Rs), float(q[0]), float(q[3])], dtype=torch.float32, device=dev)
    g = D["g"][:N].contiguous(); b = D["b"][:N].contiguous()
    A = D["A"]
    if "SIGHALF" in defs:
        A = D.get("A_half")
        if A is None:
            A = D["A_half"] = D["A"].half().contiguous()                 # cached on D: converted once per tractography
    library(defines).ukf_step(A, D["mask"], g, b, x, s, P, o, d, flags, fa, ms, ip, fp, threads=lpf * B, group_size=group_size)
    stop = (flags & 1).bool()
    info = {"swap": (flags & 2).bool(), "swap2": (flags & 4).bool(), "fa": fa, "mean_signal": ms, "inside": (flags & 8).bool()}
    return x, s, P.reshape(B, 10, 10), d, stop, info
