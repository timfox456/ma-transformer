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
import re
import subprocess
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

# Pattern logic shared by the simdgroup_matrix and Metal Performance Primitives
# tiled kernels: tile enumeration per pattern, and strip masks.
_PATTERN_SOURCE = r"""
constant int BLOCK = 32;            // rows per threadgroup, columns per tile
constant float EMPTY_ROW_LSE = -1e30f;
constant int KIND_SEGMENTS = 0;
constant int KIND_BLOCK_SPARSE = 1;

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

"""

_TILED_SOURCE = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define HEAD_DIM __HEAD_DIM__
typedef __T__ T;
typedef simdgroup_matrix<T, 8, 8> simdgroup_T8x8;

constant int DC = HEAD_DIM / 8;      // 8-wide chunks of head_dim

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

""" + _PATTERN_SOURCE + r"""// --- Strip products ------------------------------------------------------------

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


# Kernels built on Metal Performance Primitives tensor operations (Metal 4).
# Matrix products run through mpp::tensor_ops::matmul2d, which uses the GPU
# Neural Accelerators on M5 and later and the shader cores elsewhere. Each
# threadgroup (4 SIMD groups) owns a 32-row block and walks 32-wide tiles of the
# other side, like the simdgroup_matrix kernels; the pattern mask and online
# softmax reuse the same scalar code, one SIMD group per 8 rows, with scores
# passing through threadgroup memory. Accumulators (O, dQ, dK, dV) stay in
# cooperative tensors across tiles. HEAD_DIM must be a multiple of 32
# (static tensor slices) and at most 128.
_MPP_PRELUDE = r"""
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#define HEAD_DIM __HEAD_DIM__
typedef __T__ T;
"""

_MPP_KERNELS = r"""
using dev_tensor = tensor<device T, dextents<int32_t, 2>, tensor_inline>;
using tg_tensor = tensor<threadgroup T, dextents<int32_t, 2>, tensor_inline>;
using tg_float_tensor = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>;

// scores[32 x 32] = rows[32 x HEAD_DIM] @ other[32 x HEAD_DIM]^T
constexpr constant auto SCORES = matmul2d_descriptor(BLOCK, BLOCK, HEAD_DIM, false, true);
// acc[32 x HEAD_DIM] += weights[32 x 32] @ other[32 x HEAD_DIM]
constexpr constant auto ACCUMULATE = matmul2d_descriptor(BLOCK, HEAD_DIM, BLOCK, false, false, false,
                                                matmul2d_descriptor::mode::multiply_accumulate);
using scores_op_t = matmul2d<SCORES, execution_simdgroups<4>>;
using accumulate_op_t = matmul2d<ACCUMULATE, execution_simdgroups<4>>;

// [rows x HEAD_DIM] view of one (batch, head) slice; consecutive rows are `stride` elements apart.
// Tensor extents are (columns, rows).
inline dev_tensor head_view(device T* base, int rows, int stride) {
    return dev_tensor(base, dextents<int32_t, 2>(HEAD_DIM, rows), array<int32_t, 2>{1, stride});
}

template <typename CT>
inline void zero_fill(thread CT& t) {
    for (uint16_t e = 0; e < t.get_capacity(); ++e) {
        if (t.is_valid_element(e)) t[e] = 0.0f;
    }
}

// Write a [32 x HEAD_DIM] float accumulator to rows of device memory in the
// input type, element by element via each element's (column, row) index; a
// float cooperative tensor cannot store directly into half or bfloat tensors.
template <typename CT>
inline void store_rows(thread CT& t, device T* dst, int stride) {
    for (uint16_t e = 0; e < t.get_capacity(); ++e) {
        if (t.is_valid_element(e)) {
            auto ix = t.get_multidimensional_index(e);
            dst[ix[1] * stride + ix[0]] = T(t[e]);
        }
    }
}

// Multiply each row of a cooperative accumulator by factors[row].
template <typename CT>
inline void scale_accumulator_rows(thread CT& t, threadgroup const float* factors) {
    for (uint16_t e = 0; e < t.get_capacity(); ++e) {
        if (t.is_valid_element(e)) t[e] *= factors[t.get_multidimensional_index(e)[1]];
    }
}

kernel void mpp_forward(
    device T* Q [[buffer(0)]],
    device T* K [[buffer(1)]],
    device T* V [[buffer(2)]],
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
    threadgroup float s_tile[BLOCK * BLOCK];
    threadgroup T p_tile[BLOCK * BLOCK];
    threadgroup float row_scale[BLOCK];
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

    dev_tensor q_view = head_view(Q + base, S_pad, stride);
    dev_tensor k_view = head_view(K + base, S_pad, stride);
    dev_tensor v_view = head_view(V + base, S_pad, stride);
    tg_float_tensor s_t(s_tile, dextents<int32_t, 2>(BLOCK, BLOCK));
    tg_tensor p_t(p_tile, dextents<int32_t, 2>(BLOCK, BLOCK));
    scores_op_t scores_op;
    accumulate_op_t accumulate_op;

    auto q_rows = q_view.slice<HEAD_DIM, BLOCK>(0, i0);
    auto v_first = v_view.slice<HEAD_DIM, BLOCK>(0, 0);
    auto o_acc = accumulate_op.get_destination_cooperative_tensor<decltype(p_t), decltype(v_first), float>();
    zero_fill(o_acc);

    // Lane r (< 8) holds the running max and sum of this SIMD group's row r
    float row_m = 0.0f;
    float row_l = 0.0f;
    threadgroup float* strip = s_tile + 8 * sg * BLOCK;
    threadgroup T* p_strip = p_tile + 8 * sg * BLOCK;

    int nranges = pattern_ranges(kind, np);
    for (int s = 0; s < nranges; ++s) {
        int t0, t1;
        if (!pattern_tiles(kind, pat, s, i0, ntiles, false, t0, t1)) continue;
        for (int t = t0; t <= t1; ++t) {
            if (tile_seen(kind, pat, s, i0, ntiles, false, t)) continue;
            int j0 = t * BLOCK;
            if (make_tile_mask(kind, pat, np, i0, i0 + BLOCK - 1, j0, j0 + BLOCK - 1, S).empty) continue;
            TileMask tm = make_tile_mask(kind, pat, np, r0, r0 + 7, j0, j0 + BLOCK - 1, S);

            auto k_tile = k_view.slice<HEAD_DIM, BLOCK>(0, j0);
            auto scores = scores_op.get_destination_cooperative_tensor<decltype(q_rows), decltype(k_tile), float>();
            zero_fill(scores);
            scores_op.run(q_rows, k_tile, scores);
            scores.store(s_t);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            int j = j0 + lane;
            for (int r = 0; r < 8; ++r) {
                int i = r0 + r;
                float score = strip[r * BLOCK + lane] * scale;
                bool valid = !tm.empty && mask_valid(tm, kind, pat, np, i, j, S);
                if (!simd_any(valid)) {
                    if (lane == 0) row_scale[8 * sg + r] = 1.0f;
                    p_strip[r * BLOCK + lane] = T(0.0f);
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
                if (lane == 0) row_scale[8 * sg + r] = c;
                p_strip[r * BLOCK + lane] = T(p);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            scale_accumulator_rows(o_acc, row_scale);
            auto v_tile = v_view.slice<HEAD_DIM, BLOCK>(0, j0);
            accumulate_op.run(p_t, v_tile, o_acc);
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane < 8) row_scale[8 * sg + lane] = row_l > 0.0f ? 1.0f / row_l : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scale_accumulator_rows(o_acc, row_scale);
    store_rows(o_acc, O + base + i0 * stride, stride);
    if (lane < 8) {
        LSE[(b * S_pad + r0 + lane) * H + h] = row_l > 0.0f ? row_m + log(row_l) : EMPTY_ROW_LSE;
    }
}

kernel void mpp_backward_dq(
    device T* Q [[buffer(0)]],
    device T* K [[buffer(1)]],
    device T* V [[buffer(2)]],
    device T* dO [[buffer(3)]],
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
    threadgroup float s_tile[BLOCK * BLOCK];
    threadgroup float dp_tile[BLOCK * BLOCK];
    threadgroup T ds_tile[BLOCK * BLOCK];
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

    dev_tensor q_view = head_view(Q + base, S_pad, stride);
    dev_tensor k_view = head_view(K + base, S_pad, stride);
    dev_tensor v_view = head_view(V + base, S_pad, stride);
    dev_tensor g_view = head_view(dO + base, S_pad, stride);
    tg_float_tensor s_t(s_tile, dextents<int32_t, 2>(BLOCK, BLOCK));
    tg_float_tensor dp_t(dp_tile, dextents<int32_t, 2>(BLOCK, BLOCK));
    tg_tensor ds_t(ds_tile, dextents<int32_t, 2>(BLOCK, BLOCK));
    scores_op_t scores_op;
    accumulate_op_t accumulate_op;

    auto q_rows = q_view.slice<HEAD_DIM, BLOCK>(0, i0);
    auto g_rows = g_view.slice<HEAD_DIM, BLOCK>(0, i0);
    auto k_first = k_view.slice<HEAD_DIM, BLOCK>(0, 0);
    auto dq_acc = accumulate_op.get_destination_cooperative_tensor<decltype(ds_t), decltype(k_first), float>();
    zero_fill(dq_acc);

    // Lane r (< 8) holds the log-sum-exp and delta of this SIMD group's row r
    int own_stat = (b * S_pad + r0 + min(lane, 7u)) * H + h;
    float row_lse = LSE[own_stat];
    float row_delta = Delta[own_stat];
    threadgroup float* s_strip = s_tile + 8 * sg * BLOCK;
    threadgroup float* dp_strip = dp_tile + 8 * sg * BLOCK;
    threadgroup T* ds_strip = ds_tile + 8 * sg * BLOCK;

    int nranges = pattern_ranges(kind, np);
    for (int s = 0; s < nranges; ++s) {
        int t0, t1;
        if (!pattern_tiles(kind, pat, s, i0, ntiles, false, t0, t1)) continue;
        for (int t = t0; t <= t1; ++t) {
            if (tile_seen(kind, pat, s, i0, ntiles, false, t)) continue;
            int j0 = t * BLOCK;
            if (make_tile_mask(kind, pat, np, i0, i0 + BLOCK - 1, j0, j0 + BLOCK - 1, S).empty) continue;
            TileMask tm = make_tile_mask(kind, pat, np, r0, r0 + 7, j0, j0 + BLOCK - 1, S);

            auto k_tile = k_view.slice<HEAD_DIM, BLOCK>(0, j0);
            auto v_tile = v_view.slice<HEAD_DIM, BLOCK>(0, j0);
            auto scores = scores_op.get_destination_cooperative_tensor<decltype(q_rows), decltype(k_tile), float>();
            zero_fill(scores);
            scores_op.run(q_rows, k_tile, scores);
            scores.store(s_t);
            auto grads = scores_op.get_destination_cooperative_tensor<decltype(g_rows), decltype(v_tile), float>();
            zero_fill(grads);
            scores_op.run(g_rows, v_tile, grads);
            grads.store(dp_t);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            int j = j0 + lane;
            for (int r = 0; r < 8; ++r) {
                int i = r0 + r;
                bool valid = !tm.empty && mask_valid(tm, kind, pat, np, i, j, S);
                float lse = simd_shuffle(row_lse, ushort(r));
                float delta = simd_shuffle(row_delta, ushort(r));
                float p = valid ? exp(s_strip[r * BLOCK + lane] * scale - lse) : 0.0f;
                ds_strip[r * BLOCK + lane] = T(p * (dp_strip[r * BLOCK + lane] - delta) * scale);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            accumulate_op.run(ds_t, k_tile, dq_acc);
        }
    }
    store_rows(dq_acc, dQ + base + i0 * stride, stride);
}

kernel void mpp_backward_dkdv(
    device T* Q [[buffer(0)]],
    device T* K [[buffer(1)]],
    device T* V [[buffer(2)]],
    device T* dO [[buffer(3)]],
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
    threadgroup float s_tile[BLOCK * BLOCK];
    threadgroup float dp_tile[BLOCK * BLOCK];
    threadgroup T p_tile[BLOCK * BLOCK];
    threadgroup T ds_tile[BLOCK * BLOCK];
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

    dev_tensor q_view = head_view(Q + base, S_pad, stride);
    dev_tensor k_view = head_view(K + base, S_pad, stride);
    dev_tensor v_view = head_view(V + base, S_pad, stride);
    dev_tensor g_view = head_view(dO + base, S_pad, stride);
    tg_float_tensor s_t(s_tile, dextents<int32_t, 2>(BLOCK, BLOCK));
    tg_float_tensor dp_t(dp_tile, dextents<int32_t, 2>(BLOCK, BLOCK));
    tg_tensor p_t(p_tile, dextents<int32_t, 2>(BLOCK, BLOCK));
    tg_tensor ds_t(ds_tile, dextents<int32_t, 2>(BLOCK, BLOCK));
    scores_op_t scores_op;
    accumulate_op_t accumulate_op;

    auto k_rows = k_view.slice<HEAD_DIM, BLOCK>(0, j0b);
    auto v_rows = v_view.slice<HEAD_DIM, BLOCK>(0, j0b);
    auto q_first = q_view.slice<HEAD_DIM, BLOCK>(0, 0);
    auto dk_acc = accumulate_op.get_destination_cooperative_tensor<decltype(ds_t), decltype(q_first), float>();
    auto dv_acc = accumulate_op.get_destination_cooperative_tensor<decltype(p_t), decltype(q_first), float>();
    zero_fill(dk_acc);
    zero_fill(dv_acc);
    threadgroup float* s_strip = s_tile + 8 * sg * BLOCK;
    threadgroup float* dp_strip = dp_tile + 8 * sg * BLOCK;
    threadgroup T* p_strip = p_tile + 8 * sg * BLOCK;
    threadgroup T* ds_strip = ds_tile + 8 * sg * BLOCK;

    int nranges = pattern_ranges(kind, np);
    for (int s = 0; s < nranges; ++s) {
        int t0, t1;
        if (!pattern_tiles(kind, pat, s, j0b, ntiles, true, t0, t1)) continue;
        for (int t = t0; t <= t1; ++t) {
            if (tile_seen(kind, pat, s, j0b, ntiles, true, t)) continue;
            int i0 = t * BLOCK;
            if (make_tile_mask(kind, pat, np, i0, i0 + BLOCK - 1, j0b, j0b + BLOCK - 1, S).empty) continue;
            TileMask tm = make_tile_mask(kind, pat, np, i0, i0 + BLOCK - 1, k0, k0 + 7, S);

            // Transposed tiles: rows are this block's 32 keys, columns are 32 queries
            auto q_tile = q_view.slice<HEAD_DIM, BLOCK>(0, i0);
            auto g_tile = g_view.slice<HEAD_DIM, BLOCK>(0, i0);
            auto scores = scores_op.get_destination_cooperative_tensor<decltype(k_rows), decltype(q_tile), float>();
            zero_fill(scores);
            scores_op.run(k_rows, q_tile, scores);
            scores.store(s_t);
            auto grads = scores_op.get_destination_cooperative_tensor<decltype(v_rows), decltype(g_tile), float>();
            zero_fill(grads);
            scores_op.run(v_rows, g_tile, grads);
            grads.store(dp_t);
            threadgroup_barrier(mem_flags::mem_threadgroup);

            int i = i0 + lane;
            int stat = (b * S_pad + i) * H + h;
            float lse = LSE[stat];
            float delta = Delta[stat];
            for (int r = 0; r < 8; ++r) {
                int j = k0 + r;
                bool valid = !tm.empty && mask_valid(tm, kind, pat, np, i, j, S);
                float p = valid ? exp(s_strip[r * BLOCK + lane] * scale - lse) : 0.0f;
                p_strip[r * BLOCK + lane] = T(p);
                ds_strip[r * BLOCK + lane] = T(p * (dp_strip[r * BLOCK + lane] - delta) * scale);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            accumulate_op.run(p_t, g_tile, dv_acc);
            accumulate_op.run(ds_t, q_tile, dk_acc);
        }
    }
    store_rows(dk_acc, dK + base + j0b * stride, stride);
    store_rows(dv_acc, dV + base + j0b * stride, stride);
}
"""

_MPP_SOURCE = _MPP_PRELUDE + _PATTERN_SOURCE + _MPP_KERNELS


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


def _padded_elements(query: torch.Tensor) -> int:
    B, S, H, D = query.shape
    return B * (-(-S // _TILE) * _TILE) * H * D


def _tiled_supported(query: torch.Tensor) -> bool:
    D = query.shape[-1]
    return D % 8 == 0 and D <= MAX_TILED_HEAD_DIM and _padded_elements(query) <= MAX_TILED_ELEMENTS


@functools.lru_cache(maxsize=1)
def mpp_available() -> bool:
    """
    True if Metal Performance Primitives tensor kernels compile here. That needs
    a PyTorch whose shader compiler targets Metal 4 (2.14 does; 2.8 does not)
    and a macOS with Metal 4. Set MA_DISABLE_MPP=1 to turn the MPP path off.
    """
    if os.environ.get("MA_DISABLE_MPP") == "1" or not torch.backends.mps.is_available():
        return False
    try:
        _mpp_library(32, "float")
        return True
    except Exception:
        return False


def _mpp_supported(query: torch.Tensor) -> bool:
    D = query.shape[-1]
    return (D % 32 == 0 and D <= MAX_TILED_HEAD_DIM and _padded_elements(query) <= MAX_TILED_ELEMENTS
            and mpp_available())


@functools.lru_cache(maxsize=1)
def has_neural_accelerators() -> bool:
    """True on Apple M5 and later, whose GPU cores include Neural Accelerators."""
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    match = re.search(r"Apple M(\d+)", chip)
    return bool(match) and int(match.group(1)) >= 5


def _default_kernel(query: torch.Tensor, segments: bool) -> str:
    """
    Kernel used for kernel="auto". MPP only pays off with Neural Accelerators; on
    earlier chips matmul2d runs on the shader cores, where the simdgroup_matrix
    kernels are tuned. MA_MPS_KERNEL=mpp|tiled|row overrides when supported.
    """
    forced = os.environ.get("MA_MPS_KERNEL")
    if forced == "mpp" and _mpp_supported(query):
        return "mpp"
    if forced == "row" and segments:
        return "row"
    if forced != "tiled" and has_neural_accelerators() and _mpp_supported(query):
        return "mpp"
    if _tiled_supported(query):
        return "tiled"
    return "row"


@functools.lru_cache(maxsize=4)
def _row_library(metal_type: str):
    return torch.mps.compile_shader(_ROW_SOURCE.replace("__T__", metal_type))


@functools.lru_cache(maxsize=16)
def _tiled_library(head_dim: int, metal_type: str):
    source = _TILED_SOURCE.replace("__HEAD_DIM__", str(head_dim)).replace("__T__", metal_type)
    return torch.mps.compile_shader(source)


@functools.lru_cache(maxsize=16)
def _mpp_library(head_dim: int, metal_type: str):
    source = _MPP_SOURCE.replace("__HEAD_DIM__", str(head_dim)).replace("__T__", metal_type)
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


# Block kernels (tiled and MPP) share argument lists; these are their entry points.
_BLOCK_KERNELS = {
    "tiled": (_tiled_library, "tiled_forward", "tiled_backward_dq", "tiled_backward_dkdv"),
    "mpp": (_mpp_library, "mpp_forward", "mpp_backward_dq", "mpp_backward_dkdv"),
}


class _PatternAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, kind, params, num_segments, kernel):
        B, S, H, D = query.shape
        metal_type = _METAL_TYPES[query.dtype]
        pat = _param_tensor(params, query.device)
        scale = 1.0 / math.sqrt(D)
        if kernel in _BLOCK_KERNELS:
            library, forward_name, _, _ = _BLOCK_KERNELS[kernel]
            S_pad = -(-S // _TILE) * _TILE
            q, k, v = (_pad_seq(t.contiguous(), S_pad) for t in (query, key, value))
            out = torch.empty_like(q)
            lse = torch.empty(B, S_pad, H, dtype=torch.float32, device=q.device)
            getattr(library(D, metal_type), forward_name)(
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
        ctx.kernel = kernel
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

        if ctx.kernel in _BLOCK_KERNELS:
            library, _, dq_name, dkdv_name = _BLOCK_KERNELS[ctx.kernel]
            lib = library(D, metal_type)
            launch = _tiled_launch(B, H, S_stored)
            common = (pat, ctx.kind, ctx.num_segments, B, S, S_stored, H, ctx.scale)
            if need_q:
                getattr(lib, dq_name)(q, k, v, grad_out, lse, delta, grad_q, *common, **launch)
            if need_kv:
                getattr(lib, dkdv_name)(q, k, v, grad_out, lse, delta, grad_k, grad_v, *common, **launch)
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


def _check_inputs(query, key, value):
    if not supports(query):
        raise ValueError("MPS attention needs MPS tensors shaped [batch, seq, heads, dim] "
                         f"with head_dim <= {MAX_HEAD_DIM}")
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query, key and value must have identical shapes")
    if key.device != query.device or value.device != query.device:
        raise ValueError("query, key and value must be on the same device")


def _resolve_kernel(query, kernel: str, segments: bool) -> str:
    choices = ("auto", "mpp", "tiled", "row") if segments else ("auto", "mpp", "tiled")
    if kernel not in choices:
        raise ValueError(f"kernel must be one of {', '.join(repr(c) for c in choices)}")
    if kernel == "auto":
        kernel = _default_kernel(query, segments)
        if kernel == "row" and not segments:
            raise ValueError(f"block-sparse and Longformer need head_dim a multiple of 8 and <= "
                             f"{MAX_TILED_HEAD_DIM}, and fewer than {MAX_TILED_ELEMENTS + 1} elements after padding")
    elif kernel == "tiled" and not _tiled_supported(query):
        raise ValueError(f"tiled kernels need head_dim a multiple of 8 and <= {MAX_TILED_HEAD_DIM}, "
                         f"and fewer than {MAX_TILED_ELEMENTS + 1} elements after padding")
    elif kernel == "mpp" and not _mpp_supported(query):
        raise ValueError("MPP kernels need Metal 4 support in PyTorch's shader compiler (PyTorch 2.14 or "
                         f"later), head_dim a multiple of 32 and <= {MAX_TILED_HEAD_DIM}, and fewer than "
                         f"{MAX_TILED_ELEMENTS + 1} elements after padding")
    return kernel


def _apply(query, key, value, kind, params, num_segments, kernel):
    dtype = query.dtype
    if dtype not in _METAL_TYPES:
        query, key, value = query.float(), key.float(), value.float()
    else:
        key, value = key.to(dtype), value.to(dtype)
    out = _PatternAttention.apply(query, key, value, kind, tuple(params), num_segments, kernel)
    return out.to(dtype)


def segment_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                      segments: Sequence[Tuple[int, int]], kernel: str = "auto") -> torch.Tensor:
    """
    Sparse attention where query i attends to key j if lo <= j - i <= hi for a
    segment (lo, hi). Inputs are MPS tensors shaped [batch, seq, heads, dim]
    with dim <= MAX_HEAD_DIM.

    kernel: "mpp" (Metal Performance Primitives; head_dim a multiple of 32, at
    most MAX_TILED_HEAD_DIM), "tiled" (head_dim a multiple of 8, at most
    MAX_TILED_HEAD_DIM), "row" (any head_dim up to MAX_HEAD_DIM), or "auto":
    MPP on chips with Neural Accelerators (M5 and later), otherwise tiled,
    otherwise row.
    """
    _check_inputs(query, key, value)
    segments = tuple((int(lo), int(hi)) for lo, hi in segments)
    if not segments or any(lo > hi for lo, hi in segments):
        raise ValueError("segments must be a non-empty list of (lo, hi) with lo <= hi")
    kernel = _resolve_kernel(query, kernel, segments=True)
    params = [offset for segment in segments for offset in segment]
    return _apply(query, key, value, KIND_SEGMENTS, params, len(segments), kernel)


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


def block_sparse_attention(query, key, value, block_size: int = 64, kernel: str = "auto"):
    """Query i attends to key j when |i // block_size - j // block_size| <= 1 (tiled or MPP kernels)."""
    if block_size <= 0:
        raise ValueError("block_size must be > 0")
    _check_inputs(query, key, value)
    kernel = _resolve_kernel(query, kernel, segments=False)
    return _apply(query, key, value, KIND_BLOCK_SPARSE, [block_size], 0, kernel)


def longformer_attention(query, key, value, window_size: int = 64, num_global_tokens: int = 2,
                         kernel: str = "auto"):
    """
    Query i attends to key j when |i - j| <= window_size or either is one of
    the first num_global_tokens positions (tiled or MPP kernels).
    """
    if window_size < 0 or num_global_tokens < 0:
        raise ValueError("window_size and num_global_tokens must be >= 0")
    _check_inputs(query, key, value)
    kernel = _resolve_kernel(query, kernel, segments=False)
    return _apply(query, key, value, KIND_LONGFORMER, [window_size, num_global_tokens], 0, kernel)
