"""PTT with ACT as Metal kernels (torch.mps.compile_shader): one thread per half-streamline, in short launches of a few steps,
float32 (docs/sota-pipeline-design.md, section 7.2). The method and its rules are tractline.ptt's (that module's docstring
states them, with DIPY 1.12's three unit inconsistencies behind the validation switch); this is the same arithmetic in
one thread instead of lockstep tensors, so the per-step launches that bound the torch prototype are gone.

    halves = Halves(pmf_field, act, vox, params, compat=False)          # the fields' buffers, on "mps"
    pts, n, status = halves.run(seeds_vox, directions, seed=1, half=0, ids=None)

Random numbers are counter-based (Philox4x32-10, Salmon et al. 2011): key = (seed), counter = (the streamline's
global index, its draw number, the half), so a streamline's path depends on neither the batch it is in nor the order threads run.
The kernel's sampler matches tractline.fields (DIPY's conventions): the vertex nearest a direction by largest |cos|, the
first on a tie, then trilinear over the 8 corners with the edge voxel repeated; ACT include > 0.5 first, then exclude.
ptt_init and ptt_steps track; three debug kernels expose the pieces (ptt_pmf, ptt_act, ptt_support) for tests/test_ptt_metal.py.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from . import fields as F
from .nearest import CubeMap

_SRC = r"""
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#define TRACKPOINT 0
#define ENDPOINT 1
#define INVALIDPOINT 2
#define OUTSIDEIMAGE 3

// ---- Philox4x32-10: counter (c0..c3), key (k0, k1) -> 4 uniform uints
struct Rng { uint4 ctr; uint2 key; uint4 buf; int used; };

static inline uint4 philox(uint4 c, uint2 k) {
    for (int r = 0; r < 10; r++) {
        uint hi0 = mulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
        uint hi1 = mulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
        c = uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
        k = k + uint2(0x9E3779B9u, 0xBB67AE85u);
    }
    return c;
}

static inline float u01(thread Rng &g) {               // [0, 1), 24 bits
    if (g.used == 4) { g.buf = philox(g.ctr, g.key); g.ctr.y += 1u; g.used = 0; }
    uint v = g.used == 0 ? g.buf.x : (g.used == 1 ? g.buf.y : (g.used == 2 ? g.buf.z : g.buf.w));
    g.used++;
    return float(v >> 8) * (1.0f / 16777216.0f);
}

// ---- the fields: grid shape (X, Y, Z), voxel coordinates, DIPY's trilinear conventions
struct Grid { int X, Y, Z; };

static inline bool corners(float3 p, Grid g, thread int idx[8], thread float w[8]) {
    if (p.x < -0.5f || p.y < -0.5f || p.z < -0.5f || p.x >= g.X - 0.5f || p.y >= g.Y - 0.5f || p.z >= g.Z - 0.5f)
        return false;
    float3 f = floor(p);
    float3 r = p - f;
    int3 lo = int3(f), hi = lo + 1;
    int3 lim = int3(g.X - 1, g.Y - 1, g.Z - 1);
    lo = min(max(lo, int3(0)), lim);
    hi = min(max(hi, int3(0)), lim);
    for (int c = 0; c < 8; c++) {
        int ix = (c & 1) ? hi.x : lo.x, iy = (c & 2) ? hi.y : lo.y, iz = (c & 4) ? hi.z : lo.z;
        float wx = (c & 1) ? r.x : 1.0f - r.x, wy = (c & 2) ? r.y : 1.0f - r.y, wz = (c & 4) ? r.z : 1.0f - r.z;
        idx[c] = (ix * g.Y + iy) * g.Z + iz;
        w[c] = wx * wy * wz;
    }
    return true;
}

// the vertex nearest d (largest |cos|, the first on a tie), from the cube map's candidates (tractline.nearest: exact)
static inline int nearest(float3 d, device const float *verts, device const int *cmap) {
    int cells = cmap[0], nc = 6 * cells * cells;
    float3 a = fabs(d);
    int axis = (a.x >= a.y && a.x >= a.z) ? 0 : (a.y >= a.z ? 1 : 2);
    float m = axis == 0 ? d.x : (axis == 1 ? d.y : d.z);
    float o1 = axis == 0 ? d.y : (axis == 1 ? d.z : d.x);
    float o2 = axis == 0 ? d.z : (axis == 1 ? d.x : d.y);
    int face = axis * 2 + (m < 0.0f ? 1 : 0);
    int i = clamp((int)floor((o1 / fabs(m) + 1.0f) * 0.5f * cells), 0, cells - 1);
    int j = clamp((int)floor((o2 / fabs(m) + 1.0f) * 0.5f * cells), 0, cells - 1);
    int cell = (face * cells + i) * cells + j;
    device const int *cand = cmap + 1 + 2 * nc + cmap[1 + cell];
    int cnt = cmap[1 + nc + cell];
    int best = cand[0]; float bv = -1.0f;
    for (int q = 0; q < cnt; q++) {
        int v = cand[q];
        float c = fabs(d.x * verts[3 * v] + d.y * verts[3 * v + 1] + d.z * verts[3 * v + 2]);
        if (c > bv) { bv = c; best = v; }
    }
    return best;
}

static inline float pmf_at(float3 p, float3 d, device const float *table, device const int *rows,
                           device const float *verts, device const int *cmap, int V, Grid g) {
    int idx[8]; float w[8];
    if (!corners(p, g, idx, w)) return 0.0f;
    int k = nearest(d, verts, cmap);
    float s = 0.0f;
    for (int c = 0; c < 8; c++) s += table[rows[idx[c]] * V + k] * w[c];
    return s;
}

static inline int act_at(float3 p, device const float *inc, device const float *exc, Grid g) {
    int idx[8]; float w[8];
    if (!corners(p, g, idx, w)) return OUTSIDEIMAGE;
    float a = 0.0f, b = 0.0f;
    for (int c = 0; c < 8; c++) { a += inc[idx[c]] * w[c]; b += exc[idx[c]] * w[c]; }
    if (a > 0.5f) return ENDPOINT;
    if (b > 0.5f) return INVALIDPOINT;
    return TRACKPOINT;
}

// ---- the method (tractline.ptt's docstring)
struct Consts {
    float step, probe_step, max_k, thr, norm, expo, k_small;
    float3 inv_vox, probe_scale;
    int quality, nbr_sample, max_try, maxp, V, compat;
    Grid g;
};

static inline Consts consts(device const int *ip, device const float *fp) {
    Consts c;
    c.g.X = ip[0]; c.g.Y = ip[1]; c.g.Z = ip[2]; c.V = ip[3]; c.quality = ip[4]; c.nbr_sample = ip[5];
    c.max_try = ip[6]; c.maxp = ip[7]; c.compat = ip[8];
    c.step = fp[0]; c.probe_step = fp[1]; c.max_k = fp[2]; c.thr = fp[3]; c.norm = fp[4]; c.expo = fp[5];
    c.k_small = fp[6];
    c.inv_vox = float3(fp[7], fp[8], fp[9]); c.probe_scale = float3(fp[10], fp[11], fp[12]);
    return c;
}

// one parallel-transport step of length s: position (through scale), T' and N2' (not normalized); DIPY's clamp rule
static inline void propagate(thread float3 &P, float3 T, float3 N1, float3 N2, float k1, float k2, float s, float3 scale,
                             float k_small, thread float3 &Tn, thread float3 &N2n) {
    bool s1 = fabs(k1) < k_small, s2 = fabs(k2) < k_small;
    if (s1 && s2) { P = P + s * T * scale; Tn = T; N2n = N2; return; }
    if (s1) k1 = k_small;
    if (s2) k2 = k_small;
    float h = s * s * 0.5f;
    P = P + (s * T + h * (k1 * N1 + k2 * N2)) * scale;
    Tn = (1.0f - (k1 * k1 + k2 * k2) * h) * T + s * (k1 * N1 + k2 * N2);
    N2n = -k2 * s * T - k1 * k2 * h * N1 + (1.0f - k2 * k2 * h) * N2;
}

static inline float thresholded(float v, float thr) { return v > thr ? v : 0.0f; }

// a candidate's likelihood from (P, T, N1, N2) with last value `last`; the probe's last value in *val
static inline float support(float3 P, float3 T, float3 N1, float3 N2, float k1, float k2, float last, Consts c,
                            device const float *table, device const int *rows, device const float *verts,
                            device const int *cmap, thread float &val) {
    float total = last;
    val = 0.0f;
    float3 t = T, n1 = N1, n2 = N2;
    for (int q = 1; q < c.quality; q++) {
        float3 tn, n2n;
        propagate(P, t, n1, n2, k1, k2, c.probe_step, c.probe_scale, c.k_small, tn, n2n);
        tn = normalize(tn);
        if (q < c.quality - 1) { n1 = cross(n2n, tn); n2 = n2n; }
        t = tn;
        val = thresholded(pmf_at(P, t, table, rows, verts, cmap, c.V, c.g), c.thr);
        total += val;
    }
    float L = total * c.norm;
    return c.expo != 1.0f ? pow(L, c.expo) : L;
}

static inline float2 disk(thread Rng &g, float R) {
    float r = R * sqrt(u01(g)), a = 6.283185307179586f * u01(g);
    return float2(r * cos(a), r * sin(a));
}

// a random unit vector perpendicular to T (from a uniform direction)
static inline float3 perpendicular(float3 T, thread Rng &g) {
    for (int i = 0; i < 16; i++) {
        float z = 2.0f * u01(g) - 1.0f, a = 6.283185307179586f * u01(g);
        float rr = sqrt(max(0.0f, 1.0f - z * z));
        float3 r = float3(rr * cos(a), rr * sin(a), z);
        float3 n = r - dot(r, T) * T;
        float l = length(n);
        if (l > 1e-3f) return n / l;
    }
    return normalize(cross(T, fabs(T.x) < 0.9f ? float3(1, 0, 0) : float3(0, 1, 0)));
}

// The state of a half between launches: floats (B, 18) = pos 3, T 3, N1 3, N2 3, k1, k2, last, shift 3;
// ints (B, 5) = points so far, status, alive, the generator's block counter and its position in the block.
#define NF 18
#define NI 5

static inline Rng rng_of(uint id, device const int *ip, int y, int used) {
    Rng g; g.key = uint2((uint)ip[10], (uint)ip[11]); g.ctr = uint4(id, (uint)y, (uint)ip[13], 0x50545431u); g.used = used;
    if (used < 4) g.buf = philox(uint4(id, (uint)(y - 1), (uint)ip[13], 0x50545431u), g.key);
    return g;
}

kernel void ptt_init(device const float *table [[buffer(0)]], device const int *rows [[buffer(1)]],
                     device const float *verts [[buffer(2)]], device const float *seeds [[buffer(3)]],
                     device const float *dirs [[buffer(4)]], device float *pts [[buffer(5)]],
                     device float *fs [[buffer(6)]], device int *is [[buffer(7)]],
                     device const int *ip [[buffer(8)]], device const float *fp [[buffer(9)]],
                     device const int *ids [[buffer(10)]], device const int *cmap [[buffer(11)]],
                     uint b [[thread_position_in_grid]]) {
    // initialization at the seed: per candidate a frame (T the seed direction, N2 random, N1 = N2 x T)
    if ((int)b >= ip[9]) return;
    Consts c = consts(ip, fp);
    Rng g = rng_of((uint)ids[b], ip, 0, 4);
    float3 pos = float3(seeds[3 * b], seeds[3 * b + 1], seeds[3 * b + 2]);
    float3 d0 = float3(dirs[3 * b], dirs[3 * b + 1], dirs[3 * b + 2]);
    device float *out = pts + (size_t)b * (size_t)c.maxp * 3;
    out[0] = pos.x; out[1] = pos.y; out[2] = pos.z;
    float M = 0.0f;
    for (int t = 0; t < c.nbr_sample; t++) {
        float3 n2 = perpendicular(d0, g), n1 = cross(n2, d0);
        float2 k = disk(g, c.max_k);
        float lv = thresholded(pmf_at(pos, d0, table, rows, verts, cmap, c.V, c.g), c.thr);
        float v; float L = support(pos, d0, n1, n2, k.x, k.y, lv, c, table, rows, verts, cmap, v);
        M = max(M, L);
    }
    M = c.expo != 1.0f ? pow(2.0f * M, c.expo) : 2.0f * M;
    float3 N1 = float3(0), N2 = float3(0); float k1 = 0, k2 = 0, last = 0;
    for (int t = 0; t < c.max_try; t++) {
        float3 n2 = perpendicular(d0, g), n1 = cross(n2, d0);
        float2 k = disk(g, c.max_k);
        float lv = thresholded(pmf_at(pos, d0, table, rows, verts, cmap, c.V, c.g), c.thr);
        float v; float L = support(pos, d0, n1, n2, k.x, k.y, lv, c, table, rows, verts, cmap, v);
        N1 = n1; N2 = n2; k1 = k.x; k2 = k.y; last = lv;                 // none taken: the last candidate stands (DIPY)
        if (u01(g) * M <= L) { last = v; break; }
    }
    float3 shift = c.compat ? d0 * c.inv_vox * c.step : float3(0);
    device float *f = fs + (size_t)b * NF;
    f[0] = pos.x; f[1] = pos.y; f[2] = pos.z; f[3] = d0.x; f[4] = d0.y; f[5] = d0.z;
    f[6] = N1.x; f[7] = N1.y; f[8] = N1.z; f[9] = N2.x; f[10] = N2.y; f[11] = N2.z;
    f[12] = k1; f[13] = k2; f[14] = last; f[15] = shift.x; f[16] = shift.y; f[17] = shift.z;
    device int *s = is + (size_t)b * NI;
    s[0] = 1; s[1] = TRACKPOINT; s[2] = 1; s[3] = (int)g.ctr.y; s[4] = g.used;
}

kernel void ptt_steps(device const float *table [[buffer(0)]], device const int *rows [[buffer(1)]],
                      device const float *verts [[buffer(2)]], device const float *inc [[buffer(3)]],
                      device const float *exc [[buffer(4)]], device float *pts [[buffer(5)]],
                      device float *fs [[buffer(6)]], device int *is [[buffer(7)]],
                      device const int *ip [[buffer(8)]], device const float *fp [[buffer(9)]],
                      device const int *ids [[buffer(10)]], device const int *cmap [[buffer(11)]],
                      uint b [[thread_position_in_grid]]) {
    // up to ip[14] steps of a half, from and back into its state; each launch short (the GPU's time limit)
    if ((int)b >= ip[9]) return;
    device int *s = is + (size_t)b * NI;
    if (s[2] == 0) return;
    Consts c = consts(ip, fp);
    Rng g = rng_of((uint)ids[b], ip, s[3], s[4]);
    device float *f = fs + (size_t)b * NF;
    float3 pos = float3(f[0], f[1], f[2]), T = float3(f[3], f[4], f[5]), N1 = float3(f[6], f[7], f[8]);
    float3 N2 = float3(f[9], f[10], f[11]), shift = float3(f[15], f[16], f[17]);
    float k1 = f[12], k2 = f[13], last = f[14];
    device float *out = pts + (size_t)b * (size_t)c.maxp * 3;
    int n = s[0], st = TRACKPOINT, alive = 1;
    for (int it = 0; it < ip[14]; it++) {
        int i = n;
        if (i >= c.maxp) { alive = 0; break; }                              // the point limit: an invalid end
        float3 Tn, N2n;
        propagate(pos, T, N1, N2, k1, k2, c.step, c.inv_vox, c.k_small, Tn, N2n);
        T = normalize(Tn);
        N1 = normalize(cross(N2n, T));
        N2 = cross(T, N1);
        float Mx = 0.0f;
        for (int t = 0; t < c.nbr_sample; t++) {
            float2 k = disk(g, c.max_k);
            float v; Mx = max(Mx, support(pos, T, N1, N2, k.x, k.y, last, c, table, rows, verts, cmap, v));
        }
        Mx = c.expo != 1.0f ? pow(2.0f * Mx, c.expo) : 2.0f * Mx;
        bool took = false;
        for (int t = 0; t < c.max_try; t++) {
            float2 k = disk(g, c.max_k);
            float v; float L = support(pos, T, N1, N2, k.x, k.y, last, c, table, rows, verts, cmap, v);
            if (u01(g) * Mx < L) { k1 = k.x; k2 = k.y; last = v; took = true; break; }
        }
        if (!took) { st = TRACKPOINT; alive = 0; break; }                   // FAIL: no point, an invalid end
        float3 p = pos + shift;
        out[3 * i] = p.x; out[3 * i + 1] = p.y; out[3 * i + 2] = p.z;
        n = i + 1;
        st = act_at(p, inc, exc, c.g);
        if (st != TRACKPOINT) { n = i; alive = 0; break; }                  // the deciding point is not kept
    }
    f[0] = pos.x; f[1] = pos.y; f[2] = pos.z; f[3] = T.x; f[4] = T.y; f[5] = T.z;
    f[6] = N1.x; f[7] = N1.y; f[8] = N1.z; f[9] = N2.x; f[10] = N2.y; f[11] = N2.z;
    f[12] = k1; f[13] = k2; f[14] = last;
    s[0] = n; s[1] = st; s[2] = alive; s[3] = (int)g.ctr.y; s[4] = g.used;
}

// ---- the pieces, for the tests
kernel void ptt_pmf(device const float *table [[buffer(0)]], device const int *rows [[buffer(1)]],
                    device const float *verts [[buffer(2)]], device const float *p [[buffer(3)]],
                    device const float *d [[buffer(4)]], device float *out [[buffer(5)]],
                    device const int *ip [[buffer(6)]], device const int *cmap [[buffer(7)]],
                    uint b [[thread_position_in_grid]]) {
    if ((int)b >= ip[4]) return;
    Grid g; g.X = ip[0]; g.Y = ip[1]; g.Z = ip[2];
    out[b] = pmf_at(float3(p[3 * b], p[3 * b + 1], p[3 * b + 2]), float3(d[3 * b], d[3 * b + 1], d[3 * b + 2]),
                    table, rows, verts, cmap, ip[3], g);
}

kernel void ptt_act(device const float *inc [[buffer(0)]], device const float *exc [[buffer(1)]],
                    device const float *p [[buffer(2)]], device int *out [[buffer(3)]],
                    device const int *ip [[buffer(4)]], uint b [[thread_position_in_grid]]) {
    if ((int)b >= ip[3]) return;
    Grid g; g.X = ip[0]; g.Y = ip[1]; g.Z = ip[2];
    out[b] = act_at(float3(p[3 * b], p[3 * b + 1], p[3 * b + 2]), inc, exc, g);
}

kernel void ptt_support(device const float *table [[buffer(0)]], device const int *rows [[buffer(1)]],
                        device const float *verts [[buffer(2)]], device const float *frames [[buffer(3)]],
                        device float *out [[buffer(4)]], device const int *ip [[buffer(5)]],
                        device const float *fp [[buffer(6)]], device const int *cmap [[buffer(7)]],
                        uint b [[thread_position_in_grid]]) {
    // frames per row: P(3) T(3) N1(3) N2(3) k1 k2 last -> out: L, val
    if ((int)b >= ip[9]) return;
    Consts c = consts(ip, fp);
    device const float *f = frames + 15 * b;
    float v;
    float L = support(float3(f[0], f[1], f[2]), float3(f[3], f[4], f[5]), float3(f[6], f[7], f[8]),
                      float3(f[9], f[10], f[11]), f[12], f[13], f[14], c, table, rows, verts, cmap, v);
    out[2 * b] = L; out[2 * b + 1] = v;
}
"""

_LIB = None
CUBE_CELLS = 32                    # cube-map cells per face edge for the nearest vertex (2.5 candidates on average for
                                   # DIPY's 362-vertex default_sphere; 3.9 for our hemisphere(181) with its antipodes)
STEPS_PER_LAUNCH = 25              # steps per ptt_steps launch: each launch stays short (macOS's GPU time limit; class docstring)


def available() -> bool:
    return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


def library():
    global _LIB
    if _LIB is None:
        _LIB = torch.mps.compile_shader(_SRC)
    return _LIB


class Halves:
    """The fields' buffers and the method's constants on "mps", for ptt_half and the debug kernels.

    The tracking is split into short launches: ptt_init, then ptt_steps advancing every living half by
    STEPS_PER_LAUNCH steps, each thread's state (position, frame, curvatures, last value, its generator's position) kept
    in device buffers between launches, and every launch followed by torch.mps.synchronize(), so the temporaries it reads
    (constants, ids, float32 copies) outlive it. Why (2026-10-24): one launch per whole half, at 20,000 phantom seeds
    while another process used the GPU, first gave too many valid and too long streamlines (no synchronization), then
    0.15 % valid with most halves returned with no points (synchronized); at 2,000 seeds it agreed with the torch
    prototype. The cause was not isolated - freed temporaries read by a running kernel, a command buffer stopped for
    running too long, or both; no error was reported - and the short, synchronized launches fixed it (the same results at
    any launch size: tests/test_ptt_metal.py)."""
    def __init__(self, pmf: F.PmfField, act: F.Act, vox, params, compat=False):
        dev = torch.device("mps")
        if pmf.table.device.type != "mps":
            raise ValueError("ptt_metal: the PMF field must be on mps")
        self.dev, self.p, self.compat = dev, params, bool(compat)
        self.table = pmf.table.to(torch.float32).contiguous()
        self.rows = pmf.rows.to(torch.int32).contiguous()
        self.verts = pmf.vertices.to(torch.float32).contiguous()
        cm = CubeMap(pmf.vertices.cpu().double().numpy(), CUBE_CELLS)
        self.cmap = torch.as_tensor(np.concatenate([[cm.cells], cm.offsets, cm.counts, cm.candidates]).astype(np.int32),
                                    device=dev)
        self.inc = act.include.data[:, 0].to(torch.float32).contiguous()
        self.exc = act.exclude.data[:, 0].to(torch.float32).contiguous()
        self.shape = tuple(int(s) for s in pmf.shape)
        vox = np.asarray(vox, np.float64)
        step_for_k = params.step_size / vox.mean() if compat else params.step_size
        self.max_k = 2.0 * math.sin(math.radians(params.max_angle) / 2.0) / step_for_k
        probe_scale = vox if compat else 1.0 / vox
        self.maxp = int(params.max_len / params.step_size)
        self.fp = torch.tensor([params.step_size, params.probe_length / (params.probe_quality - 1), self.max_k,
                                params.pmf_threshold, 1.0 / (params.probe_quality * params.probe_count),
                                params.data_support_exponent, 1e-4, *(1.0 / vox), *probe_scale],
                               dtype=torch.float32, device=dev)

    def _ip(self, B, seed=0, half=0, steps=0):
        X, Y, Z = self.shape
        p = self.p
        return torch.tensor([X, Y, Z, len(self.verts), p.probe_quality, p.nbr_sample, p.max_try, self.maxp, int(self.compat),
                             B, seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, 0, half, steps], dtype=torch.int32, device=self.dev)

    def run(self, seeds_vox, directions, seed=1, half=0, ids=None, steps=STEPS_PER_LAUNCH, group_size=64):
        """seeds_vox, directions (B, 3) -> (points (B, maxp, 3) voxel coordinates, counts (B,), status (B,)). ids: each
        seed's global index (the random numbers' counter; default 0..B-1). One launch initializes every half, then
        launches of `steps` steps each advance the living ones until none is left."""
        s = torch.as_tensor(seeds_vox, dtype=torch.float32, device=self.dev).contiguous()
        d = torch.as_tensor(directions, dtype=torch.float32, device=self.dev).contiguous()
        B = len(s)
        ids = torch.arange(B, device=self.dev) if ids is None else ids
        ids = torch.as_tensor(ids, device=self.dev).to(torch.int32).contiguous()
        pts = torch.zeros((B, self.maxp, 3), dtype=torch.float32, device=self.dev)
        fs = torch.zeros((B, 18), dtype=torch.float32, device=self.dev)
        istate = torch.zeros((B, 5), dtype=torch.int32, device=self.dev)
        if B:
            ip = self._ip(B, seed, half, steps)
            lib = library()
            lib.ptt_init(self.table, self.rows, self.verts, s, d, pts, fs, istate, ip, self.fp, ids, self.cmap, threads=B,
                         group_size=group_size)
            torch.mps.synchronize()                                          # before the inputs are freed (class docstring)
            for _ in range(self.maxp // steps + 2):
                lib.ptt_steps(self.table, self.rows, self.verts, self.inc, self.exc, pts, fs, istate, ip, self.fp, ids,
                              self.cmap, threads=B, group_size=group_size)
                torch.mps.synchronize()
                if not bool(istate[:, 2].any()):
                    break
            if bool(istate[:, 2].any()):                                     # cannot happen unless a launch did not run
                raise RuntimeError(f"ptt_metal: {int(istate[:, 2].sum())} halves still alive after every launch")
        return pts, istate[:, 0].clone(), istate[:, 1].clone()

    def pmf(self, points, directions):
        p = torch.as_tensor(points, dtype=torch.float32, device=self.dev).contiguous()
        d = torch.as_tensor(directions, dtype=torch.float32, device=self.dev).contiguous()
        out = torch.zeros(len(p), dtype=torch.float32, device=self.dev)
        X, Y, Z = self.shape
        ip = torch.tensor([X, Y, Z, len(self.verts), len(p)], dtype=torch.int32, device=self.dev)
        library().ptt_pmf(self.table, self.rows, self.verts, p, d, out, ip, self.cmap, threads=len(p), group_size=64)
        torch.mps.synchronize()
        return out

    def act(self, points):
        p = torch.as_tensor(points, dtype=torch.float32, device=self.dev).contiguous()
        out = torch.zeros(len(p), dtype=torch.int32, device=self.dev)
        X, Y, Z = self.shape
        ip = torch.tensor([X, Y, Z, len(p)], dtype=torch.int32, device=self.dev)
        library().ptt_act(self.inc, self.exc, p, out, ip, threads=len(p), group_size=64)
        torch.mps.synchronize()
        return out

    def support(self, P, T, N1, N2, k, last):
        """(n, 3) x 4, k (n, 2), last (n,) -> (L, the probe's last value), each (n,)."""
        f = torch.cat([torch.as_tensor(x, dtype=torch.float32, device=self.dev).reshape(len(P), -1)
                       for x in (P, T, N1, N2, k, last)], 1).contiguous()
        out = torch.zeros((len(P), 2), dtype=torch.float32, device=self.dev)
        library().ptt_support(self.table, self.rows, self.verts, f, out, self._ip(len(P)), self.fp, self.cmap, threads=len(P),
                              group_size=64)
        torch.mps.synchronize()
        return out[:, 0], out[:, 1]
