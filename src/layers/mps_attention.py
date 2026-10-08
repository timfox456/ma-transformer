# SPDX-License-Identifier: Apache-2.0
"""
Metal (MPS) kernels for sparse attention on Apple silicon.

Supported patterns (query i, key j):
  * segments: inclusive ranges of key offsets relative to the query, so i
    attends to j when j - i falls in any segment [lo, hi]. A sliding window is
    one segment; the financial pattern is a causal local window plus one
    segment per dilated cluster. Overlapping segments count each key once.
  * block-sparse: |i // b - j // b| <= 1 for block size b.
  * Longformer: |i - j| <= w, or i or j is one of the first g (global) tokens.

Two kernel families:
  * Tiled (head_dim a multiple of 8, up to MAX_TILED_HEAD_DIM, and padded
    tensors below 2**31 elements, since indexing is 32-bit; all patterns).
    A threadgroup of 4 SIMD groups owns 32 rows and walks 32-wide tiles of the
    other side, so each loaded tile is reused by 32 rows. Products use 8x8
    simdgroup_matrix operations. Forward uses an online softmax and stores each
    row's log-sum-exp; backward is FlashAttention-2 style (dQ per query block,
    dK/dV per key block), with no atomics.
  * Per-row (any head_dim up to MAX_HEAD_DIM, any size; segment patterns only).
    One SIMD group per row, head_dim split across lanes.

Kernels are compiled at runtime with torch.mps.compile_shader, so no Xcode or
Objective-C++ build step is required. Inputs are [batch, seq, heads, dim] in
float32, float16 or bfloat16; arithmetic is float32 and results keep the input
dtype. Other dtypes are converted to float32 and back.
"""

import functools
import math
import os
from typing import List, Sequence, Tuple

import torch

MAX_HEAD_DIM = 256          # per-row kernels
MAX_TILED_HEAD_DIM = 128    # tiled kernels (also need head_dim % 8 == 0)
# Tiled kernels index with 32-bit integers (64-bit arithmetic is emulated on
# Apple GPUs and costs registers), so padded tensors must stay below 2**31 elements.
MAX_TILED_ELEMENTS = 2**31 - 1
_SIMD_WIDTH = 32
_ROWS_PER_THREADGROUP = 4
_TILE = 32

# Pattern kinds shared with the Metal source
KIND_SEGMENTS = 0
KIND_BLOCK_SPARSE = 1
KIND_LONGFORMER = 2

_METAL_TYPES = {torch.float32: "float", torch.float16: "half", torch.bfloat16: "bfloat"}

_ROW_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

typedef __T__ T;
constant int MAX_CHUNKS = 8;          // head_dim <= 8 * 32
constant float EMPTY_ROW_LSE = -1e30f;

// True if offset `delta` lies in one of the first `count` segments.
inline bool covered_by_earlier(device const int* seg, long count, long delta) {
    for (long m = 0; m < count; ++m) {
        if (delta >= seg[2 * m] && delta <= seg[2 * m + 1]) return true;
    }
    return false;
}

// Row r of the [B, S, H, D] layout is ordered (b, h, i) so neighbouring SIMD
// groups work on neighbouring sequence positions and share cached keys.
inline void decode_row(long row, long S, long H, thread long& b, thread long& h, thread long& i) {
    i = row % S;
    long bh = row / S;
    h = bh % H;
    b = bh / H;
}

kernel void segment_attention_forward(
    device const T* Q [[buffer(0)]],
    device const T* K [[buffer(1)]],
    device const T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    device float* LSE [[buffer(4)]],
    device const int* seg [[buffer(5)]],
    constant long& B [[buffer(6)]],
    constant long& S [[buffer(7)]],
    constant long& H [[buffer(8)]],
    constant long& D [[buffer(9)]],
    constant long& nseg [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    long row = long(tid) / 32;
    if (row >= B * H * S) return;
    long b, h, i;
    decode_row(row, S, H, b, h, i);

    long stride = H * D;
    long base = b * S * stride + h * D;
    device const T* q = Q + base + i * stride;

    float qr[MAX_CHUNKS];
    float acc[MAX_CHUNKS];
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        long d = long(lane) + 32 * c;
        qr[c] = d < D ? float(q[d]) * scale : 0.0f;
        acc[c] = 0.0f;
    }

    float m = 0.0f;   // running max (valid once l > 0)
    float l = 0.0f;   // running sum of exp
    for (long s = 0; s < nseg; ++s) {
        long j0 = max(0L, i + long(seg[2 * s]));
        long j1 = min(S - 1, i + long(seg[2 * s + 1]));
        for (long j = j0; j <= j1; ++j) {
            if (covered_by_earlier(seg, s, j - i)) continue;
            device const T* kr = K + base + j * stride;
            float part = 0.0f;
            for (int c = 0; c < MAX_CHUNKS; ++c) {
                long d = long(lane) + 32 * c;
                if (d < D) part += qr[c] * float(kr[d]);
            }
            float score = simd_sum(part);
            float m_new = l > 0.0f ? max(m, score) : score;
            float correction = l > 0.0f ? exp(m - m_new) : 0.0f;
            float p = exp(score - m_new);
            l = l * correction + p;
            device const T* vr = V + base + j * stride;
            for (int c = 0; c < MAX_CHUNKS; ++c) {
                long d = long(lane) + 32 * c;
                if (d < D) acc[c] = acc[c] * correction + p * float(vr[d]);
            }
            m = m_new;
        }
    }

    float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    device T* o = O + base + i * stride;
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        long d = long(lane) + 32 * c;
        if (d < D) o[d] = T(acc[c] * inv_l);
    }
    if (lane == 0) {
        LSE[(b * S + i) * H + h] = l > 0.0f ? m + log(l) : EMPTY_ROW_LSE;
    }
}

kernel void segment_attention_backward_dq(
    device const T* Q [[buffer(0)]],
    device const T* K [[buffer(1)]],
    device const T* V [[buffer(2)]],
    device const T* dO [[buffer(3)]],
    device const float* LSE [[buffer(4)]],
    device const float* Delta [[buffer(5)]],
    device T* dQ [[buffer(6)]],
    device const int* seg [[buffer(7)]],
    constant long& B [[buffer(8)]],
    constant long& S [[buffer(9)]],
    constant long& H [[buffer(10)]],
    constant long& D [[buffer(11)]],
    constant long& nseg [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    long row = long(tid) / 32;
    if (row >= B * H * S) return;
    long b, h, i;
    decode_row(row, S, H, b, h, i);

    long stride = H * D;
    long base = b * S * stride + h * D;
    long stat = (b * S + i) * H + h;
    float lse = LSE[stat];
    float delta_i = Delta[stat];

    float qr[MAX_CHUNKS];
    float gr[MAX_CHUNKS];
    float acc[MAX_CHUNKS];
    device const T* q = Q + base + i * stride;
    device const T* g = dO + base + i * stride;
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        long d = long(lane) + 32 * c;
        qr[c] = d < D ? float(q[d]) * scale : 0.0f;
        gr[c] = d < D ? float(g[d]) : 0.0f;
        acc[c] = 0.0f;
    }

    if (lse > EMPTY_ROW_LSE) {
        for (long s = 0; s < nseg; ++s) {
            long j0 = max(0L, i + long(seg[2 * s]));
            long j1 = min(S - 1, i + long(seg[2 * s + 1]));
            for (long j = j0; j <= j1; ++j) {
                if (covered_by_earlier(seg, s, j - i)) continue;
                device const T* kr = K + base + j * stride;
                device const T* vr = V + base + j * stride;
                float qk = 0.0f;
                float gv = 0.0f;
                for (int c = 0; c < MAX_CHUNKS; ++c) {
                    long d = long(lane) + 32 * c;
                    if (d < D) {
                        qk += qr[c] * float(kr[d]);
                        gv += gr[c] * float(vr[d]);
                    }
                }
                float p = exp(simd_sum(qk) - lse);
                float ds = p * (simd_sum(gv) - delta_i);
                for (int c = 0; c < MAX_CHUNKS; ++c) {
                    long d = long(lane) + 32 * c;
                    if (d < D) acc[c] += ds * float(kr[d]);
                }
            }
        }
    }

    device T* out = dQ + base + i * stride;
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        long d = long(lane) + 32 * c;
        if (d < D) out[d] = T(acc[c] * scale);
    }
}

kernel void segment_attention_backward_dkdv(
    device const T* Q [[buffer(0)]],
    device const T* K [[buffer(1)]],
    device const T* V [[buffer(2)]],
    device const T* dO [[buffer(3)]],
    device const float* LSE [[buffer(4)]],
    device const float* Delta [[buffer(5)]],
    device T* dK [[buffer(6)]],
    device T* dV [[buffer(7)]],
    device const int* seg [[buffer(8)]],
    constant long& B [[buffer(9)]],
    constant long& S [[buffer(10)]],
    constant long& H [[buffer(11)]],
    constant long& D [[buffer(12)]],
    constant long& nseg [[buffer(13)]],
    constant float& scale [[buffer(14)]],
    uint tid [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{
    long row = long(tid) / 32;
    if (row >= B * H * S) return;
    long b, h, j;
    decode_row(row, S, H, b, h, j);

    long stride = H * D;
    long base = b * S * stride + h * D;

    float kr[MAX_CHUNKS];
    float vr[MAX_CHUNKS];
    float dk[MAX_CHUNKS];
    float dv[MAX_CHUNKS];
    device const T* k = K + base + j * stride;
    device const T* v = V + base + j * stride;
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        long d = long(lane) + 32 * c;
        kr[c] = d < D ? float(k[d]) * scale : 0.0f;
        vr[c] = d < D ? float(v[d]) : 0.0f;
        dk[c] = 0.0f;
        dv[c] = 0.0f;
    }

    // Query i attends to key j when j - i lies in a segment [lo, hi],
    // i.e. i in [j - hi, j - lo].
    for (long s = 0; s < nseg; ++s) {
        long i0 = max(0L, j - long(seg[2 * s + 1]));
        long i1 = min(S - 1, j - long(seg[2 * s]));
        for (long i = i0; i <= i1; ++i) {
            if (covered_by_earlier(seg, s, j - i)) continue;
            long stat = (b * S + i) * H + h;
            device const T* q = Q + base + i * stride;
            device const T* g = dO + base + i * stride;
            float qk = 0.0f;
            float gv = 0.0f;
            for (int c = 0; c < MAX_CHUNKS; ++c) {
                long d = long(lane) + 32 * c;
                if (d < D) {
                    qk += float(q[d]) * kr[c];
                    gv += float(g[d]) * vr[c];
                }
            }
            float p = exp(simd_sum(qk) - LSE[stat]);
            float ds = p * (simd_sum(gv) - Delta[stat]);
            for (int c = 0; c < MAX_CHUNKS; ++c) {
                long d = long(lane) + 32 * c;
                if (d < D) {
                    dv[c] += p * float(g[d]);
                    dk[c] += ds * float(q[d]);
                }
            }
        }
    }

    device T* out_k = dK + base + j * stride;
    device T* out_v = dV + base + j * stride;
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        long d = long(lane) + 32 * c;
        if (d < D) {
            out_k[d] = T(dk[c] * scale);
            out_v[d] = T(dv[c]);
        }
    }
}
"""

_TILED_SOURCE = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define HEAD_DIM __HEAD_DIM__
typedef __T__ T;
typedef simdgroup_matrix<T, 8, 8> simdgroup_T8x8;

constant int DC = HEAD_DIM / 8;      // 8-wide chunks of head_dim
constant int BLOCK = 32;            // rows per threadgroup, columns per tile
constant float EMPTY_ROW_LSE = -1e30f;
constant int KIND_SEGMENTS = 0;
constant int KIND_BLOCK_SPARSE = 1;

// --- Loads and stores: inputs are T in device memory, arithmetic is float ---

inline simdgroup_float8x8 load8(device const T* src, int stride, bool transpose = false) {
    simdgroup_T8x8 m;
    simdgroup_load(m, src, stride, ulong2(0, 0), transpose);
    simdgroup_float8x8 f;
    f.thread_elements()[0] = float(m.thread_elements()[0]);
    f.thread_elements()[1] = float(m.thread_elements()[1]);
    return f;
}

inline simdgroup_float8x8 load8(threadgroup const float* src, int stride, bool transpose = false) {
    simdgroup_float8x8 f;
    simdgroup_load(f, src, stride, ulong2(0, 0), transpose);
    return f;
}

inline void store8(simdgroup_float8x8 f, device T* dst, int stride) {
    simdgroup_T8x8 m;
    m.thread_elements()[0] = T(f.thread_elements()[0]);
    m.thread_elements()[1] = T(f.thread_elements()[1]);
    simdgroup_store(m, dst, stride);
}

// --- Patterns ---------------------------------------------------------------
// `pat` holds the pattern parameters: (lo, hi) pairs for segments, [b] for
// block-sparse, [w, g] for Longformer. `np` is the number of segments.

inline int pattern_ranges(int kind, int np) {
    return kind == KIND_SEGMENTS ? np : (kind == KIND_BLOCK_SPARSE ? 1 : 2);
}

// Tiles [t0, t1] on the other side reached through range `idx` from the block
// of BLOCK rows starting at r0. Rows are queries, or keys when `transposed`
// (dK/dV), in which case segment offsets are negated.
inline bool pattern_tiles(int kind, device const int* pat, int idx, int r0, int ntiles,
                          bool transposed, thread int& t0, thread int& t1) {
    int n = ntiles * BLOCK;
    int first, last;
    if (kind == KIND_SEGMENTS) {
        int lo = transposed ? -int(pat[2 * idx + 1]) : int(pat[2 * idx]);
        int hi = transposed ? -int(pat[2 * idx]) : int(pat[2 * idx + 1]);
        first = r0 + lo;
        last = r0 + BLOCK - 1 + hi;
    } else if (kind == KIND_BLOCK_SPARSE) {
        int b = pat[0];
        first = (r0 / b - 1) * b;
        last = ((r0 + BLOCK - 1) / b + 2) * b - 1;
    } else {
        // Longformer is symmetric; global rows reach everything
        int w = pat[0];
        int g = pat[1];
        bool global_rows = r0 < g;
        if (idx == 0) {
            if (global_rows) {
                first = 0;
                last = n - 1;
            } else {
                if (g <= 0) return false;
                first = 0;
                last = g - 1;
            }
        } else {
            if (global_rows) return false;
            first = r0 - w;
            last = r0 + BLOCK - 1 + w;
        }
    }
    if (last < 0 || first > n - 1) return false;
    t0 = max(first, 0) / BLOCK;
    t1 = min(last, n - 1) / BLOCK;
    return true;
}

// True if tile t was already visited through an earlier range, so each
// (row, column) pair is processed exactly once.
inline bool tile_seen(int kind, device const int* pat, int idx, int r0, int ntiles,
                      bool transposed, int t) {
    for (int m = 0; m < idx; ++m) {
        int a, b;
        if (pattern_tiles(kind, pat, m, r0, ntiles, transposed, a, b) && t >= a && t <= b) return true;
    }
    return false;
}

// Mask for one SIMD group's strip: queries [qa, qb] x keys [ka, kb].
// `full` means every pair is valid, so the per-element test is skipped;
// `empty` means none is, so the SIMD group skips the strip. For segments, up
// to two segments overlapping the strip are kept so partial strips test only
// those (count < 0: more overlap, test all). They are separate scalars, not an
// array: an array written at a runtime index cannot live in registers, and
// spilling it to memory slowed every kernel.
struct TileMask {
    bool full;
    bool empty;
    int count;
    int lo0, hi0, lo1, hi1;
};

inline TileMask make_tile_mask(int kind, device const int* pat, int np,
                               int qa, int qb, int ka, int kb, int S) {
    TileMask m;
    m.full = false;
    m.empty = qa >= S || ka >= S;
    m.count = 0;
    if (m.empty) return m;
    bool in_bounds = qb < S && kb < S;
    if (kind == KIND_SEGMENTS) {
        int dmin = ka - qb;
        int dmax = kb - qa;
        for (int s = 0; s < np; ++s) {
            int lo = pat[2 * s];
            int hi = pat[2 * s + 1];
            if (hi < dmin || lo > dmax) continue;
            if (in_bounds && lo <= dmin && dmax <= hi) {
                m.full = true;
                return m;
            }
            if (m.count == 0) {
                m.lo0 = lo;
                m.hi0 = hi;
                m.count = 1;
            } else if (m.count == 1) {
                m.lo1 = lo;
                m.hi1 = hi;
                m.count = 2;
            } else {
                m.count = -1;
            }
        }
        m.empty = m.count == 0;
    } else if (kind == KIND_BLOCK_SPARSE) {
        int b = pat[0];
        m.full = in_bounds && max(qb / b - ka / b, kb / b - qa / b) <= 1;
        m.empty = qa / b - kb / b > 1 || ka / b - qb / b > 1;
    } else {
        int w = pat[0];
        int g = pat[1];
        m.full = in_bounds && (qb < g || kb < g || max(qb - ka, kb - qa) <= w);
        m.empty = qa >= g && ka >= g && (qa - kb > w || ka - qb > w);
    }
    return m;
}

inline bool mask_valid(thread const TileMask& m, int kind, device const int* pat, int np,
                       int i, int j, int S) {
    if (i >= S || j >= S) return false;
    if (m.full) return true;
    if (kind == KIND_SEGMENTS) {
        int d = j - i;
        if (m.count >= 0) {
            return (m.count >= 1 && d >= m.lo0 && d <= m.hi0) ||
                   (m.count == 2 && d >= m.lo1 && d <= m.hi1);
        }
        for (int s = 0; s < np; ++s) {
            if (d >= pat[2 * s] && d <= pat[2 * s + 1]) return true;
        }
        return false;
    }
    if (kind == KIND_BLOCK_SPARSE) {
        int b = pat[0];
        int d = i / b - j / b;
        return d >= -1 && d <= 1;
    }
    int w = pat[0];
    int g = pat[1];
    return (i - j <= w && j - i <= w) || i < g || j < g;
}

// --- Strip products ------------------------------------------------------------

// acc = diag @ acc, with the 8x8 diagonal already written to threadgroup memory
inline void scale_by_diag(thread simdgroup_float8x8* acc, threadgroup const float* diag) {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 d;
    simdgroup_load(d, diag, 8);
    for (int dc = 0; dc < DC; ++dc) simdgroup_multiply(acc[dc], d, acc[dc]);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// strip (8 x 32) = A_rows (8 x HEAD_DIM, in registers) @ X[col0 : col0 + 32]^T
// X is either device memory (T) or a tile staged in threadgroup memory (float).
template <typename P>
inline void strip_times_transpose(thread const simdgroup_float8x8* a, P X,
                                  int col0, int stride, threadgroup float* strip) {
    for (int cb = 0; cb < 4; ++cb) {
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        P x = X + (col0 + 8 * cb) * stride;
        for (int dc = 0; dc < DC; ++dc) {
            simdgroup_multiply_accumulate(acc, a[dc], load8(x + 8 * dc, stride, true), acc);
        }
        simdgroup_store(acc, strip + 8 * cb, BLOCK);
    }
}

// Same as above with the 8 x HEAD_DIM left operand held in threadgroup memory
template <typename P>
inline void tg_strip_times_transpose(threadgroup const float* a_rows, P X,
                                     int col0, int stride, threadgroup float* strip) {
    for (int cb = 0; cb < 4; ++cb) {
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        P x = X + (col0 + 8 * cb) * stride;
        for (int dc = 0; dc < DC; ++dc) {
            simdgroup_float8x8 a;
            simdgroup_load(a, a_rows + 8 * dc, HEAD_DIM);
            simdgroup_multiply_accumulate(acc, a, load8(x + 8 * dc, stride, true), acc);
        }
        simdgroup_store(acc, strip + 8 * cb, BLOCK);
    }
}

// acc (8 x HEAD_DIM) += strip (8 x 32) @ X[row0 : row0 + 32]
template <typename P>
inline void accumulate_strip_times(thread simdgroup_float8x8* acc, threadgroup const float* strip,
                                   P X, int row0, int stride) {
    for (int kb = 0; kb < 4; ++kb) {
        simdgroup_float8x8 p;
        simdgroup_load(p, strip + 8 * kb, BLOCK);
        P x = X + (row0 + 8 * kb) * stride;
        for (int dc = 0; dc < DC; ++dc) {
            simdgroup_multiply_accumulate(acc[dc], p, load8(x + 8 * dc, stride), acc[dc]);
        }
    }
}

// --- Kernels -----------------------------------------------------------------

kernel void tiled_forward(
    device const T* Q [[buffer(0)]],
    device const T* K [[buffer(1)]],
    device const T* V [[buffer(2)]],
    device T* O [[buffer(3)]],
    device float* LSE [[buffer(4)]],
    device const int* pat [[buffer(5)]],
    constant long& kind_arg [[buffer(6)]],
    constant long& np_arg [[buffer(7)]],
    constant long& B_arg [[buffer(8)]],
    constant long& S_arg [[buffer(9)]],
    constant long& S_pad_arg [[buffer(10)]],
    constant long& H_arg [[buffer(11)]],
    constant float& scale [[buffer(12)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int kind = int(kind_arg);
    const int np = int(np_arg);
    const int B = int(B_arg);
    const int S = int(S_arg);
    const int S_pad = int(S_pad_arg);
    const int H = int(H_arg);
    threadgroup float strips[4][8 * BLOCK];
    threadgroup float scratch[4][64];
    int ntiles = S_pad / BLOCK;
    int blk = tg % ntiles;
    int bh = tg / ntiles;
    int h = bh % H;
    int b = bh / H;
    if (b >= B) return;

    int stride = H * HEAD_DIM;
    int base = b * S_pad * stride + h * HEAD_DIM;
    int i0 = blk * BLOCK;
    int r0 = i0 + 8 * sg;
    threadgroup float* strip = strips[sg];

    simdgroup_float8x8 q[DC], o[DC];
    for (int dc = 0; dc < DC; ++dc) {
        q[dc] = load8(Q + base + r0 * stride + 8 * dc, stride);
        o[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    // Lane r (< 8) holds the running max and sum of row r; the rescale
    // factors go straight onto the diagonal of an 8x8 scratch matrix.
    float row_m = 0.0f;
    float row_l = 0.0f;
    threadgroup float* diag = scratch[sg];
    for (uint idx = lane; idx < 64; idx += 32) diag[idx] = 0.0f;

    int nranges = pattern_ranges(kind, np);
    for (int s = 0; s < nranges; ++s) {
        int t0, t1;
        if (!pattern_tiles(kind, pat, s, i0, ntiles, false, t0, t1)) continue;
        for (int t = t0; t <= t1; ++t) {
            if (tile_seen(kind, pat, s, i0, ntiles, false, t)) continue;
            int j0 = t * BLOCK;
            TileMask tm = make_tile_mask(kind, pat, np, r0, r0 + 7, j0, j0 + BLOCK - 1, S);
            if (tm.empty) continue;
            strip_times_transpose(q, K + base, j0, stride, strip);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            int j = j0 + lane;
            for (int r = 0; r < 8; ++r) {
                int i = r0 + r;
                float score = strip[r * BLOCK + lane] * scale;
                bool valid = mask_valid(tm, kind, pat, np, i, j, S);
                if (!simd_any(valid)) {
                    if (lane == 0) diag[r * 9] = 1.0f;
                    strip[r * BLOCK + lane] = 0.0f;
                    continue;
                }
                float m_r = simd_shuffle(row_m, ushort(r));
                float l_r = simd_shuffle(row_l, ushort(r));
                float tile_max = simd_max(valid ? score : -FLT_MAX);
                float m_new = l_r > 0.0f ? max(m_r, tile_max) : tile_max;
                float c = l_r > 0.0f ? exp(m_r - m_new) : 0.0f;
                float p = valid ? exp(score - m_new) : 0.0f;
                float l_new = l_r * c + simd_sum(p);
                if (lane == uint(r)) {
                    row_m = m_new;
                    row_l = l_new;
                }
                if (lane == 0) diag[r * 9] = c;
                strip[r * BLOCK + lane] = p;
            }
            scale_by_diag(o, diag);
            accumulate_strip_times(o, strip, V + base, j0, stride);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (lane < 8) diag[lane * 9] = row_l > 0.0f ? 1.0f / row_l : 0.0f;
    scale_by_diag(o, diag);
    for (int dc = 0; dc < DC; ++dc) store8(o[dc], O + base + r0 * stride + 8 * dc, stride);
    if (lane < 8) {
        LSE[(b * S_pad + r0 + lane) * H + h] = row_l > 0.0f ? row_m + log(row_l) : EMPTY_ROW_LSE;
    }
}

kernel void tiled_backward_dq(
    device const T* Q [[buffer(0)]],
    device const T* K [[buffer(1)]],
    device const T* V [[buffer(2)]],
    device const T* dO [[buffer(3)]],
    device const float* LSE [[buffer(4)]],
    device const float* Delta [[buffer(5)]],
    device T* dQ [[buffer(6)]],
    device const int* pat [[buffer(7)]],
    constant long& kind_arg [[buffer(8)]],
    constant long& np_arg [[buffer(9)]],
    constant long& B_arg [[buffer(10)]],
    constant long& S_arg [[buffer(11)]],
    constant long& S_pad_arg [[buffer(12)]],
    constant long& H_arg [[buffer(13)]],
    constant float& scale [[buffer(14)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int kind = int(kind_arg);
    const int np = int(np_arg);
    const int B = int(B_arg);
    const int S = int(S_arg);
    const int S_pad = int(S_pad_arg);
    const int H = int(H_arg);
    threadgroup float score_strips[4][8 * BLOCK];
    threadgroup float grad_strips[4][8 * BLOCK];
    int ntiles = S_pad / BLOCK;
    int blk = tg % ntiles;
    int bh = tg / ntiles;
    int h = bh % H;
    int b = bh / H;
    if (b >= B) return;

    int stride = H * HEAD_DIM;
    int base = b * S_pad * stride + h * HEAD_DIM;
    int i0 = blk * BLOCK;
    int r0 = i0 + 8 * sg;
    threadgroup float* sS = score_strips[sg];
    threadgroup float* sP = grad_strips[sg];

    simdgroup_float8x8 q[DC], g[DC], acc[DC];
    for (int dc = 0; dc < DC; ++dc) {
        q[dc] = load8(Q + base + r0 * stride + 8 * dc, stride);
        g[dc] = load8(dO + base + r0 * stride + 8 * dc, stride);
        acc[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    // Lane r (< 8) holds the log-sum-exp and delta of row r
    int own_stat = (b * S_pad + r0 + min(lane, 7u)) * H + h;
    float row_lse = LSE[own_stat];
    float row_delta = Delta[own_stat];

    int nranges = pattern_ranges(kind, np);
    for (int s = 0; s < nranges; ++s) {
        int t0, t1;
        if (!pattern_tiles(kind, pat, s, i0, ntiles, false, t0, t1)) continue;
        for (int t = t0; t <= t1; ++t) {
            if (tile_seen(kind, pat, s, i0, ntiles, false, t)) continue;
            int j0 = t * BLOCK;
            TileMask tm = make_tile_mask(kind, pat, np, r0, r0 + 7, j0, j0 + BLOCK - 1, S);
            if (tm.empty) continue;
            strip_times_transpose(q, K + base, j0, stride, sS);
            strip_times_transpose(g, V + base, j0, stride, sP);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            int j = j0 + lane;
            for (int r = 0; r < 8; ++r) {
                int i = r0 + r;
                bool valid = mask_valid(tm, kind, pat, np, i, j, S);
                float lse = simd_shuffle(row_lse, ushort(r));
                float delta = simd_shuffle(row_delta, ushort(r));
                float p = valid ? exp(sS[r * BLOCK + lane] * scale - lse) : 0.0f;
                sS[r * BLOCK + lane] = p * (sP[r * BLOCK + lane] - delta) * scale;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            accumulate_strip_times(acc, sS, K + base, j0, stride);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (int dc = 0; dc < DC; ++dc) store8(acc[dc], dQ + base + r0 * stride + 8 * dc, stride);
}

kernel void tiled_backward_dkdv(
    device const T* Q [[buffer(0)]],
    device const T* K [[buffer(1)]],
    device const T* V [[buffer(2)]],
    device const T* dO [[buffer(3)]],
    device const float* LSE [[buffer(4)]],
    device const float* Delta [[buffer(5)]],
    device T* dK [[buffer(6)]],
    device T* dV [[buffer(7)]],
    device const int* pat [[buffer(8)]],
    constant long& kind_arg [[buffer(9)]],
    constant long& np_arg [[buffer(10)]],
    constant long& B_arg [[buffer(11)]],
    constant long& S_arg [[buffer(12)]],
    constant long& S_pad_arg [[buffer(13)]],
    constant long& H_arg [[buffer(14)]],
    constant float& scale [[buffer(15)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    const int kind = int(kind_arg);
    const int np = int(np_arg);
    const int B = int(B_arg);
    const int S = int(S_arg);
    const int S_pad = int(S_pad_arg);
    const int H = int(H_arg);
    threadgroup float prob_strips[4][8 * BLOCK];
    threadgroup float grad_strips[4][8 * BLOCK];
    // V rows are staged in threadgroup memory and K rows stay in registers.
    // Staging K as well needs 24 KB, which leaves room for only one
    // threadgroup per core; keeping V in registers too raises register use
    // more than it saves (both measured slower on M1 Pro).
    threadgroup float value_rows[4][8 * HEAD_DIM];
    int ntiles = S_pad / BLOCK;
    int blk = tg % ntiles;
    int bh = tg / ntiles;
    int h = bh % H;
    int b = bh / H;
    if (b >= B) return;

    int stride = H * HEAD_DIM;
    int base = b * S_pad * stride + h * HEAD_DIM;
    int j0b = blk * BLOCK;
    int k0 = j0b + 8 * sg;
    threadgroup float* sS = prob_strips[sg];
    threadgroup float* sP = grad_strips[sg];

    threadgroup float* sV = value_rows[sg];
    simdgroup_float8x8 kk[DC];
    for (int dc = 0; dc < DC; ++dc) kk[dc] = load8(K + base + k0 * stride + 8 * dc, stride);
    for (uint idx = lane; idx < 8 * HEAD_DIM; idx += 32) {
        int r = idx / HEAD_DIM;
        int d = idx % HEAD_DIM;
        sV[idx] = float(V[base + (k0 + r) * stride + d]);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 dk[DC], dv[DC];
    for (int dc = 0; dc < DC; ++dc) {
        dk[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        dv[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    int nranges = pattern_ranges(kind, np);
    for (int s = 0; s < nranges; ++s) {
        int t0, t1;
        if (!pattern_tiles(kind, pat, s, j0b, ntiles, true, t0, t1)) continue;
        for (int t = t0; t <= t1; ++t) {
            if (tile_seen(kind, pat, s, j0b, ntiles, true, t)) continue;
            int i0 = t * BLOCK;
            TileMask tm = make_tile_mask(kind, pat, np, i0, i0 + BLOCK - 1, k0, k0 + 7, S);
            if (tm.empty) continue;
            // Transposed strips: rows are this SIMD group's 8 keys, columns are 32 queries
            strip_times_transpose(kk, Q + base, i0, stride, sS);
            tg_strip_times_transpose(sV, dO + base, i0, stride, sP);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            int i = i0 + lane;
            int stat = (b * S_pad + i) * H + h;
            float lse = LSE[stat];
            float delta = Delta[stat];
            for (int r = 0; r < 8; ++r) {
                int j = k0 + r;
                bool valid = mask_valid(tm, kind, pat, np, i, j, S);
                float p = valid ? exp(sS[r * BLOCK + lane] * scale - lse) : 0.0f;
                sS[r * BLOCK + lane] = p;
                sP[r * BLOCK + lane] = p * (sP[r * BLOCK + lane] - delta) * scale;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            accumulate_strip_times(dv, sS, dO + base, i0, stride);
            accumulate_strip_times(dk, sP, Q + base, i0, stride);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (int dc = 0; dc < DC; ++dc) {
        store8(dk[dc], dK + base + k0 * stride + 8 * dc, stride);
        store8(dv[dc], dV + base + k0 * stride + 8 * dc, stride);
    }
}
"""


def is_available() -> bool:
    """True if the Metal kernels can run (MPS present and not disabled)."""
    if os.environ.get("MA_DISABLE_MPS_KERNELS") == "1":
        return False
    return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


def supports(query: torch.Tensor) -> bool:
    """True if segment patterns (window, financial) can run on this input."""
    return (query.device.type == "mps" and query.dim() == 4
            and 0 < query.shape[-1] <= MAX_HEAD_DIM and is_available())


def supports_tiled(query: torch.Tensor) -> bool:
    """True if every pattern, including block-sparse and Longformer, can run on this input."""
    return supports(query) and _tiled_supported(query)


def _tiled_supported(query: torch.Tensor) -> bool:
    B, S, H, D = query.shape
    padded = B * (-(-S // _TILE) * _TILE) * H * D
    return D % 8 == 0 and D <= MAX_TILED_HEAD_DIM and padded <= MAX_TILED_ELEMENTS


@functools.lru_cache(maxsize=4)
def _row_library(metal_type: str):
    return torch.mps.compile_shader(_ROW_SOURCE.replace("__T__", metal_type))


@functools.lru_cache(maxsize=16)
def _tiled_library(head_dim: int, metal_type: str):
    source = _TILED_SOURCE.replace("__HEAD_DIM__", str(head_dim)).replace("__T__", metal_type)
    return torch.mps.compile_shader(source)


@functools.lru_cache(maxsize=64)
def _param_tensor(params: Tuple[int, ...], device: torch.device) -> torch.Tensor:
    return torch.tensor(params, dtype=torch.int32, device=device)


def _row_launch(rows: int):
    threads_per_group = _SIMD_WIDTH * _ROWS_PER_THREADGROUP
    groups = (rows + _ROWS_PER_THREADGROUP - 1) // _ROWS_PER_THREADGROUP
    return dict(threads=groups * threads_per_group, group_size=threads_per_group)


def _tiled_launch(batch: int, heads: int, padded_len: int):
    threads_per_group = _SIMD_WIDTH * 4
    groups = batch * heads * (padded_len // _TILE)
    return dict(threads=groups * threads_per_group, group_size=threads_per_group)


def _pad_seq(t: torch.Tensor, padded_len: int) -> torch.Tensor:
    extra = padded_len - t.shape[1]
    return torch.nn.functional.pad(t, (0, 0, 0, 0, 0, extra)) if extra else t


class _PatternAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, kind, params, num_segments, tiled):
        B, S, H, D = query.shape
        metal_type = _METAL_TYPES[query.dtype]
        pat = _param_tensor(params, query.device)
        scale = 1.0 / math.sqrt(D)
        if tiled:
            S_pad = -(-S // _TILE) * _TILE
            q, k, v = (_pad_seq(t.contiguous(), S_pad) for t in (query, key, value))
            out = torch.empty_like(q)
            lse = torch.empty(B, S_pad, H, dtype=torch.float32, device=q.device)
            _tiled_library(D, metal_type).tiled_forward(
                q, k, v, out, lse, pat, kind, num_segments, B, S, S_pad, H, scale,
                **_tiled_launch(B, H, S_pad))
        else:
            q, k, v = (t.contiguous() for t in (query, key, value))
            out = torch.empty_like(q)
            lse = torch.empty(B, S, H, dtype=torch.float32, device=q.device)
            _row_library(metal_type).segment_attention_forward(
                q, k, v, out, lse, pat, B, S, H, D, num_segments, scale, **_row_launch(B * H * S))
        ctx.save_for_backward(q, k, v, out, lse, pat)
        ctx.scale = scale
        ctx.kind = kind
        ctx.num_segments = num_segments
        ctx.tiled = tiled
        ctx.seq_len = S
        return out[:, :S]

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse, pat = ctx.saved_tensors
        B, S_stored, H, D = q.shape
        S = ctx.seq_len
        metal_type = _METAL_TYPES[q.dtype]
        grad_out = _pad_seq(grad_out.to(q.dtype).contiguous(), S_stored)
        delta = (grad_out.float() * out.float()).sum(-1).contiguous()
        need_q = ctx.needs_input_grad[0]
        need_kv = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        grad_q = torch.empty_like(q) if need_q else None
        grad_k = torch.empty_like(k) if need_kv else None
        grad_v = torch.empty_like(v) if need_kv else None

        if ctx.tiled:
            lib = _tiled_library(D, metal_type)
            launch = _tiled_launch(B, H, S_stored)
            common = (pat, ctx.kind, ctx.num_segments, B, S, S_stored, H, ctx.scale)
            if need_q:
                lib.tiled_backward_dq(q, k, v, grad_out, lse, delta, grad_q, *common, **launch)
            if need_kv:
                lib.tiled_backward_dkdv(q, k, v, grad_out, lse, delta, grad_k, grad_v, *common, **launch)
        else:
            lib = _row_library(metal_type)
            launch = _row_launch(B * H * S)
            common = (pat, B, S, H, D, ctx.num_segments, ctx.scale)
            if need_q:
                lib.segment_attention_backward_dq(q, k, v, grad_out, lse, delta, grad_q, *common, **launch)
            if need_kv:
                lib.segment_attention_backward_dkdv(q, k, v, grad_out, lse, delta, grad_k, grad_v,
                                                    *common, **launch)

        trim = (lambda g: g[:, :S] if g is not None else None)
        return trim(grad_q), trim(grad_k), trim(grad_v), None, None, None, None


def _check_inputs(query, key, value, needs_tiled: bool):
    if not supports(query):
        raise ValueError("MPS attention needs MPS tensors shaped [batch, seq, heads, dim] "
                         f"with head_dim <= {MAX_HEAD_DIM}")
    if needs_tiled and not _tiled_supported(query):
        raise ValueError(f"tiled kernels need head_dim a multiple of 8 and <= {MAX_TILED_HEAD_DIM}, "
                         f"and fewer than {MAX_TILED_ELEMENTS + 1} elements after padding")
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query, key and value must have identical shapes")
    if key.device != query.device or value.device != query.device:
        raise ValueError("query, key and value must be on the same device")


def _apply(query, key, value, kind, params, num_segments, tiled):
    dtype = query.dtype
    if dtype not in _METAL_TYPES:
        query, key, value = query.float(), key.float(), value.float()
    else:
        key, value = key.to(dtype), value.to(dtype)
    out = _PatternAttention.apply(query, key, value, kind, tuple(params), num_segments, tiled)
    return out.to(dtype)


def segment_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                      segments: Sequence[Tuple[int, int]], kernel: str = "auto") -> torch.Tensor:
    """
    Sparse attention where query i attends to key j if lo <= j - i <= hi for a
    segment (lo, hi). Inputs are MPS tensors shaped [batch, seq, heads, dim]
    with dim <= MAX_HEAD_DIM.

    kernel: "tiled" (head_dim a multiple of 8, at most MAX_TILED_HEAD_DIM),
    "row" (any head_dim up to MAX_HEAD_DIM), or "auto" to prefer tiled.
    """
    if kernel not in ("auto", "tiled", "row"):
        raise ValueError("kernel must be 'auto', 'tiled' or 'row'")
    _check_inputs(query, key, value, needs_tiled=kernel == "tiled")
    segments = tuple((int(lo), int(hi)) for lo, hi in segments)
    if not segments or any(lo > hi for lo, hi in segments):
        raise ValueError("segments must be a non-empty list of (lo, hi) with lo <= hi")
    tiled = kernel == "tiled" or (kernel == "auto" and _tiled_supported(query))
    params = [offset for segment in segments for offset in segment]
    return _apply(query, key, value, KIND_SEGMENTS, params, len(segments), tiled)


def window_segments(window_size: int, causal: bool = False) -> List[Tuple[int, int]]:
    return [(-window_size, 0 if causal else window_size)]


def financial_segments(local_window_size: int, dilation_stride: int,
                       dilation_cluster_size: int, dilation_num_clusters: int) -> List[Tuple[int, int]]:
    segments = [(-local_window_size + 1, 0)]
    for c in range(1, dilation_num_clusters + 1):
        end = -c * dilation_stride
        segments.append((end - dilation_cluster_size + 1, end))
    return segments


def sliding_window_attention(query, key, value, window_size: int, causal: bool = False,
                             kernel: str = "auto"):
    return segment_attention(query, key, value, window_segments(window_size, causal), kernel)


def financial_attention(query, key, value, local_window_size: int = 512, dilation_stride: int = 1000,
                        dilation_cluster_size: int = 8, dilation_num_clusters: int = 10,
                        kernel: str = "auto"):
    return segment_attention(query, key, value, financial_segments(
        local_window_size, dilation_stride, dilation_cluster_size, dilation_num_clusters), kernel)


def block_sparse_attention(query, key, value, block_size: int = 64):
    """Query i attends to key j when |i // block_size - j // block_size| <= 1 (tiled kernels only)."""
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    _check_inputs(query, key, value, needs_tiled=True)
    return _apply(query, key, value, KIND_BLOCK_SPARSE, [block_size], 0, True)


def longformer_attention(query, key, value, window_size: int = 64, num_global_tokens: int = 2):
    """
    Query i attends to key j when |i - j| <= window_size or either is one of
    the first num_global_tokens positions (tiled kernels only).
    """
    if window_size < 0 or num_global_tokens < 0:
        raise ValueError("window_size and num_global_tokens must be >= 0")
    _check_inputs(query, key, value, needs_tiled=True)
    return _apply(query, key, value, KIND_LONGFORMER, [window_size, num_global_tokens], 0, True)
