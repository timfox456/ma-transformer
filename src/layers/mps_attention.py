# SPDX-License-Identifier: Apache-2.0
"""
Metal (MPS) kernels for sparse attention on Apple silicon.

Patterns are described as "segments": inclusive ranges of key offsets
relative to the query, so query i attends to key j when j - i falls in any
segment [lo, hi]. A sliding window is one segment; the financial pattern is a
causal local window plus one segment per dilated cluster. Offsets covered by
an earlier segment are skipped, so overlapping segments count each key once.

Kernels (one SIMD group of 32 threads per row, head_dim split across lanes):
  * forward: online softmax over the row's keys; also stores the row's
    log-sum-exp for the backward pass.
  * backward dQ: one row per query.
  * backward dK/dV: one row per key, walking the transposed segments, so no
    atomics are needed and results are deterministic.

Kernels are compiled at runtime with torch.mps.compile_shader, so no Xcode
or Objective-C++ build step is required. Inputs are [batch, seq, heads, dim];
computation is in float32 (other dtypes are converted and converted back).
"""

import functools
import math
import os
from typing import List, Sequence, Tuple

import torch

MAX_HEAD_DIM = 256          # per-row kernels
MAX_TILED_HEAD_DIM = 128    # tiled kernels (also need head_dim % 8 == 0)
_SIMD_WIDTH = 32
_ROWS_PER_THREADGROUP = 4
_TILE = 32

_ROW_SOURCE = r"""
#include <metal_stdlib>
using namespace metal;

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
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
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
    device const float* q = Q + base + i * stride;

    float qr[MAX_CHUNKS];
    float acc[MAX_CHUNKS];
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        long d = long(lane) + 32 * c;
        qr[c] = d < D ? q[d] * scale : 0.0f;
        acc[c] = 0.0f;
    }

    float m = 0.0f;   // running max (valid once l > 0)
    float l = 0.0f;   // running sum of exp
    for (long s = 0; s < nseg; ++s) {
        long j0 = max(0L, i + long(seg[2 * s]));
        long j1 = min(S - 1, i + long(seg[2 * s + 1]));
        for (long j = j0; j <= j1; ++j) {
            if (covered_by_earlier(seg, s, j - i)) continue;
            device const float* kr = K + base + j * stride;
            float part = 0.0f;
            for (int c = 0; c < MAX_CHUNKS; ++c) {
                long d = long(lane) + 32 * c;
                if (d < D) part += qr[c] * kr[d];
            }
            float score = simd_sum(part);
            float m_new = l > 0.0f ? max(m, score) : score;
            float correction = l > 0.0f ? exp(m - m_new) : 0.0f;
            float p = exp(score - m_new);
            l = l * correction + p;
            device const float* vr = V + base + j * stride;
            for (int c = 0; c < MAX_CHUNKS; ++c) {
                long d = long(lane) + 32 * c;
                if (d < D) acc[c] = acc[c] * correction + p * vr[d];
            }
            m = m_new;
        }
    }

    float inv_l = l > 0.0f ? 1.0f / l : 0.0f;
    device float* o = O + base + i * stride;
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        long d = long(lane) + 32 * c;
        if (d < D) o[d] = acc[c] * inv_l;
    }
    if (lane == 0) {
        LSE[(b * S + i) * H + h] = l > 0.0f ? m + log(l) : EMPTY_ROW_LSE;
    }
}

kernel void segment_attention_backward_dq(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* dO [[buffer(3)]],
    device const float* LSE [[buffer(4)]],
    device const float* Delta [[buffer(5)]],
    device float* dQ [[buffer(6)]],
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
    device const float* q = Q + base + i * stride;
    device const float* g = dO + base + i * stride;
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        long d = long(lane) + 32 * c;
        qr[c] = d < D ? q[d] * scale : 0.0f;
        gr[c] = d < D ? g[d] : 0.0f;
        acc[c] = 0.0f;
    }

    if (lse > EMPTY_ROW_LSE) {
        for (long s = 0; s < nseg; ++s) {
            long j0 = max(0L, i + long(seg[2 * s]));
            long j1 = min(S - 1, i + long(seg[2 * s + 1]));
            for (long j = j0; j <= j1; ++j) {
                if (covered_by_earlier(seg, s, j - i)) continue;
                device const float* kr = K + base + j * stride;
                device const float* vr = V + base + j * stride;
                float qk = 0.0f;
                float gv = 0.0f;
                for (int c = 0; c < MAX_CHUNKS; ++c) {
                    long d = long(lane) + 32 * c;
                    if (d < D) {
                        qk += qr[c] * kr[d];
                        gv += gr[c] * vr[d];
                    }
                }
                float p = exp(simd_sum(qk) - lse);
                float ds = p * (simd_sum(gv) - delta_i);
                for (int c = 0; c < MAX_CHUNKS; ++c) {
                    long d = long(lane) + 32 * c;
                    if (d < D) acc[c] += ds * kr[d];
                }
            }
        }
    }

    device float* out = dQ + base + i * stride;
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        long d = long(lane) + 32 * c;
        if (d < D) out[d] = acc[c] * scale;
    }
}

kernel void segment_attention_backward_dkdv(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* dO [[buffer(3)]],
    device const float* LSE [[buffer(4)]],
    device const float* Delta [[buffer(5)]],
    device float* dK [[buffer(6)]],
    device float* dV [[buffer(7)]],
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
    device const float* k = K + base + j * stride;
    device const float* v = V + base + j * stride;
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        long d = long(lane) + 32 * c;
        kr[c] = d < D ? k[d] * scale : 0.0f;
        vr[c] = d < D ? v[d] : 0.0f;
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
            device const float* q = Q + base + i * stride;
            device const float* g = dO + base + i * stride;
            float qk = 0.0f;
            float gv = 0.0f;
            for (int c = 0; c < MAX_CHUNKS; ++c) {
                long d = long(lane) + 32 * c;
                if (d < D) {
                    qk += q[d] * kr[c];
                    gv += g[d] * vr[c];
                }
            }
            float p = exp(simd_sum(qk) - LSE[stat]);
            float ds = p * (simd_sum(gv) - Delta[stat]);
            for (int c = 0; c < MAX_CHUNKS; ++c) {
                long d = long(lane) + 32 * c;
                if (d < D) {
                    dv[c] += p * g[d];
                    dk[c] += ds * q[d];
                }
            }
        }
    }

    device float* out_k = dK + base + j * stride;
    device float* out_v = dV + base + j * stride;
    for (int c = 0; c < MAX_CHUNKS; ++c) {
        long d = long(lane) + 32 * c;
        if (d < D) {
            out_k[d] = dk[c] * scale;
            out_v[d] = dv[c];
        }
    }
}
"""


# Tiled kernels (FlashAttention-2 style). A threadgroup of 4 SIMD groups owns a
# block of 32 rows (8 per SIMD group) and walks 32-wide tiles of the other side,
# so each loaded key/value tile is reused by 32 queries. Matrix products use
# 8x8 simdgroup_matrix operations; scores pass through threadgroup memory so the
# pattern mask and softmax can be applied per element. Sequences are padded to
# a multiple of 32 by the caller. HEAD_DIM is substituted per compiled library.
_TILED_SOURCE = r"""
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define HEAD_DIM __HEAD_DIM__
constant int DC = HEAD_DIM / 8;      // 8-wide chunks of head_dim
constant long BLOCK = 32;            // rows per threadgroup, columns per tile
constant float EMPTY_ROW_LSE = -1e30f;

inline bool in_pattern(device const int* seg, long nseg, long delta) {
    for (long s = 0; s < nseg; ++s) {
        if (delta >= seg[2 * s] && delta <= seg[2 * s + 1]) return true;
    }
    return false;
}

// Tiles [t0, t1] on the other side reached by segment s from the block of
// rows starting at r0. Forward and dQ look up keys (offsets lo..hi); dK/dV
// look up queries, which sit at offsets -hi..-lo from a key.
inline bool segment_tiles(device const int* seg, long s, long r0, long ntiles, bool transposed,
                          thread long& t0, thread long& t1) {
    long lo = transposed ? -long(seg[2 * s + 1]) : long(seg[2 * s]);
    long hi = transposed ? -long(seg[2 * s]) : long(seg[2 * s + 1]);
    long first = r0 + lo;
    long last = r0 + BLOCK - 1 + hi;
    long n = ntiles * BLOCK;
    if (last < 0 || first > n - 1) return false;
    t0 = max(first, 0L) / BLOCK;
    t1 = min(last, n - 1) / BLOCK;
    return true;
}

// True if tile t was already visited through an earlier segment, so each
// (row, column) pair is processed exactly once.
inline bool tile_seen(device const int* seg, long s, long r0, long ntiles, bool transposed, long t) {
    for (long m = 0; m < s; ++m) {
        long a, b;
        if (segment_tiles(seg, m, r0, ntiles, transposed, a, b) && t >= a && t <= b) return true;
    }
    return false;
}

// Multiply each row of the 8 x HEAD_DIM accumulator by factors[row].
inline void scale_rows(thread simdgroup_float8x8* acc, thread const float* factors,
                       threadgroup float* scratch, uint lane) {
    for (uint idx = lane; idx < 64; idx += 32) {
        uint r = idx / 8;
        scratch[idx] = (r == idx % 8) ? factors[r] : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 diag;
    simdgroup_load(diag, scratch, 8);
    for (int dc = 0; dc < DC; ++dc) simdgroup_multiply(acc[dc], diag, acc[dc]);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// strip (8 x 32) = A_rows (8 x HEAD_DIM, in registers) @ X[col0 : col0 + 32]^T
inline void strip_times_transpose(thread const simdgroup_float8x8* a, device const float* X,
                                  long col0, long stride, threadgroup float* strip) {
    for (int cb = 0; cb < 4; ++cb) {
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        device const float* x = X + (col0 + 8 * cb) * stride;
        for (int dc = 0; dc < DC; ++dc) {
            simdgroup_float8x8 xt;
            simdgroup_load(xt, x + 8 * dc, stride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a[dc], xt, acc);
        }
        simdgroup_store(acc, strip + 8 * cb, BLOCK);
    }
}

// Same as above with the 8 x HEAD_DIM left operand held in threadgroup memory
inline void tg_strip_times_transpose(threadgroup const float* a_rows, device const float* X,
                                     long col0, long stride, threadgroup float* strip) {
    for (int cb = 0; cb < 4; ++cb) {
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        device const float* x = X + (col0 + 8 * cb) * stride;
        for (int dc = 0; dc < DC; ++dc) {
            simdgroup_float8x8 a, xt;
            simdgroup_load(a, a_rows + 8 * dc, HEAD_DIM);
            simdgroup_load(xt, x + 8 * dc, stride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a, xt, acc);
        }
        simdgroup_store(acc, strip + 8 * cb, BLOCK);
    }
}

// acc (8 x HEAD_DIM) += strip (8 x 32) @ X[row0 : row0 + 32]
inline void accumulate_strip_times(thread simdgroup_float8x8* acc, threadgroup const float* strip,
                                   device const float* X, long row0, long stride) {
    for (int kb = 0; kb < 4; ++kb) {
        simdgroup_float8x8 p;
        simdgroup_load(p, strip + 8 * kb, BLOCK);
        device const float* x = X + (row0 + 8 * kb) * stride;
        for (int dc = 0; dc < DC; ++dc) {
            simdgroup_float8x8 xv;
            simdgroup_load(xv, x + 8 * dc, stride);
            simdgroup_multiply_accumulate(acc[dc], p, xv, acc[dc]);
        }
    }
}

kernel void tiled_forward(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    device float* LSE [[buffer(4)]],
    device const int* seg [[buffer(5)]],
    constant long& B [[buffer(6)]],
    constant long& S [[buffer(7)]],
    constant long& S_pad [[buffer(8)]],
    constant long& H [[buffer(9)]],
    constant long& nseg [[buffer(10)]],
    constant float& scale [[buffer(11)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float strips[4][8 * BLOCK];
    threadgroup float scratch[4][64];
    long ntiles = S_pad / BLOCK;
    long blk = tg % ntiles;
    long bh = tg / ntiles;
    long h = bh % H;
    long b = bh / H;
    if (b >= B) return;

    long stride = H * HEAD_DIM;
    long base = b * S_pad * stride + h * HEAD_DIM;
    long i0 = blk * BLOCK;
    long r0 = i0 + 8 * sg;
    threadgroup float* strip = strips[sg];

    simdgroup_float8x8 q[DC], o[DC];
    for (int dc = 0; dc < DC; ++dc) {
        simdgroup_load(q[dc], Q + base + r0 * stride + 8 * dc, stride);
        o[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float m[8], l[8], corr[8];
    for (int r = 0; r < 8; ++r) { m[r] = 0.0f; l[r] = 0.0f; }

    for (long s = 0; s < nseg; ++s) {
        long t0, t1;
        if (!segment_tiles(seg, s, i0, ntiles, false, t0, t1)) continue;
        for (long t = t0; t <= t1; ++t) {
            if (tile_seen(seg, s, i0, ntiles, false, t)) continue;
            long j0 = t * BLOCK;
            strip_times_transpose(q, K + base, j0, stride, strip);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            long j = j0 + lane;
            for (int r = 0; r < 8; ++r) {
                long i = r0 + r;
                float score = strip[r * BLOCK + lane] * scale;
                bool valid = i < S && j < S && in_pattern(seg, nseg, j - i);
                if (!simd_any(valid)) {
                    corr[r] = 1.0f;
                    strip[r * BLOCK + lane] = 0.0f;
                    continue;
                }
                float tile_max = simd_max(valid ? score : -FLT_MAX);
                float m_new = l[r] > 0.0f ? max(m[r], tile_max) : tile_max;
                float c = l[r] > 0.0f ? exp(m[r] - m_new) : 0.0f;
                float p = valid ? exp(score - m_new) : 0.0f;
                l[r] = l[r] * c + simd_sum(p);
                m[r] = m_new;
                corr[r] = c;
                strip[r * BLOCK + lane] = p;
            }
            scale_rows(o, corr, scratch[sg], lane);
            accumulate_strip_times(o, strip, V + base, j0, stride);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (int r = 0; r < 8; ++r) corr[r] = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
    scale_rows(o, corr, scratch[sg], lane);
    for (int dc = 0; dc < DC; ++dc) simdgroup_store(o[dc], O + base + r0 * stride + 8 * dc, stride);
    if (lane < 8) {
        LSE[(b * S_pad + r0 + lane) * H + h] = l[lane] > 0.0f ? m[lane] + log(l[lane]) : EMPTY_ROW_LSE;
    }
}

kernel void tiled_backward_dq(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* dO [[buffer(3)]],
    device const float* LSE [[buffer(4)]],
    device const float* Delta [[buffer(5)]],
    device float* dQ [[buffer(6)]],
    device const int* seg [[buffer(7)]],
    constant long& B [[buffer(8)]],
    constant long& S [[buffer(9)]],
    constant long& S_pad [[buffer(10)]],
    constant long& H [[buffer(11)]],
    constant long& nseg [[buffer(12)]],
    constant float& scale [[buffer(13)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float score_strips[4][8 * BLOCK];
    threadgroup float grad_strips[4][8 * BLOCK];
    long ntiles = S_pad / BLOCK;
    long blk = tg % ntiles;
    long bh = tg / ntiles;
    long h = bh % H;
    long b = bh / H;
    if (b >= B) return;

    long stride = H * HEAD_DIM;
    long base = b * S_pad * stride + h * HEAD_DIM;
    long i0 = blk * BLOCK;
    long r0 = i0 + 8 * sg;
    threadgroup float* sS = score_strips[sg];
    threadgroup float* sP = grad_strips[sg];

    simdgroup_float8x8 q[DC], g[DC], acc[DC];
    for (int dc = 0; dc < DC; ++dc) {
        simdgroup_load(q[dc], Q + base + r0 * stride + 8 * dc, stride);
        simdgroup_load(g[dc], dO + base + r0 * stride + 8 * dc, stride);
        acc[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    float lse[8], delta[8];
    for (int r = 0; r < 8; ++r) {
        long stat = (b * S_pad + r0 + r) * H + h;
        lse[r] = LSE[stat];
        delta[r] = Delta[stat];
    }

    for (long s = 0; s < nseg; ++s) {
        long t0, t1;
        if (!segment_tiles(seg, s, i0, ntiles, false, t0, t1)) continue;
        for (long t = t0; t <= t1; ++t) {
            if (tile_seen(seg, s, i0, ntiles, false, t)) continue;
            long j0 = t * BLOCK;
            strip_times_transpose(q, K + base, j0, stride, sS);
            strip_times_transpose(g, V + base, j0, stride, sP);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            long j = j0 + lane;
            for (int r = 0; r < 8; ++r) {
                long i = r0 + r;
                bool valid = i < S && j < S && in_pattern(seg, nseg, j - i);
                float p = valid ? exp(sS[r * BLOCK + lane] * scale - lse[r]) : 0.0f;
                sS[r * BLOCK + lane] = p * (sP[r * BLOCK + lane] - delta[r]) * scale;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            accumulate_strip_times(acc, sS, K + base, j0, stride);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    for (int dc = 0; dc < DC; ++dc) simdgroup_store(acc[dc], dQ + base + r0 * stride + 8 * dc, stride);
}

kernel void tiled_backward_dkdv(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* dO [[buffer(3)]],
    device const float* LSE [[buffer(4)]],
    device const float* Delta [[buffer(5)]],
    device float* dK [[buffer(6)]],
    device float* dV [[buffer(7)]],
    device const int* seg [[buffer(8)]],
    constant long& B [[buffer(9)]],
    constant long& S [[buffer(10)]],
    constant long& S_pad [[buffer(11)]],
    constant long& H [[buffer(12)]],
    constant long& nseg [[buffer(13)]],
    constant float& scale [[buffer(14)]],
    uint tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]])
{
    threadgroup float prob_strips[4][8 * BLOCK];
    threadgroup float grad_strips[4][8 * BLOCK];
    // K and V rows live in threadgroup memory rather than registers: holding
    // them alongside the dK and dV accumulators spills registers. Above
    // head_dim 64 both would exceed 32 KB, so K stays in registers.
#if HEAD_DIM <= 64
    threadgroup float key_rows[4][8 * HEAD_DIM];
#endif
    threadgroup float value_rows[4][8 * HEAD_DIM];
    long ntiles = S_pad / BLOCK;
    long blk = tg % ntiles;
    long bh = tg / ntiles;
    long h = bh % H;
    long b = bh / H;
    if (b >= B) return;

    long stride = H * HEAD_DIM;
    long base = b * S_pad * stride + h * HEAD_DIM;
    long j0b = blk * BLOCK;
    long k0 = j0b + 8 * sg;
    threadgroup float* sS = prob_strips[sg];
    threadgroup float* sP = grad_strips[sg];

    threadgroup float* sV = value_rows[sg];
#if HEAD_DIM <= 64
    threadgroup float* sK = key_rows[sg];
#else
    simdgroup_float8x8 kk[DC];
    for (int dc = 0; dc < DC; ++dc) simdgroup_load(kk[dc], K + base + k0 * stride + 8 * dc, stride);
#endif
    for (uint idx = lane; idx < 8 * HEAD_DIM; idx += 32) {
        long r = idx / HEAD_DIM;
        long d = idx % HEAD_DIM;
        sV[idx] = V[base + (k0 + r) * stride + d];
#if HEAD_DIM <= 64
        sK[idx] = K[base + (k0 + r) * stride + d];
#endif
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    simdgroup_float8x8 dk[DC], dv[DC];
    for (int dc = 0; dc < DC; ++dc) {
        dk[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        dv[dc] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }

    for (long s = 0; s < nseg; ++s) {
        long t0, t1;
        if (!segment_tiles(seg, s, j0b, ntiles, true, t0, t1)) continue;
        for (long t = t0; t <= t1; ++t) {
            if (tile_seen(seg, s, j0b, ntiles, true, t)) continue;
            long i0 = t * BLOCK;
            // Transposed strips: rows are this SIMD group's 8 keys, columns are 32 queries
#if HEAD_DIM <= 64
            tg_strip_times_transpose(sK, Q + base, i0, stride, sS);
#else
            strip_times_transpose(kk, Q + base, i0, stride, sS);
#endif
            tg_strip_times_transpose(sV, dO + base, i0, stride, sP);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            long i = i0 + lane;
            long stat = (b * S_pad + i) * H + h;
            float lse = LSE[stat];
            float delta = Delta[stat];
            for (int r = 0; r < 8; ++r) {
                long j = k0 + r;
                bool valid = i < S && j < S && in_pattern(seg, nseg, j - i);
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
        simdgroup_store(dk[dc], dK + base + k0 * stride + 8 * dc, stride);
        simdgroup_store(dv[dc], dV + base + k0 * stride + 8 * dc, stride);
    }
}
"""


def is_available() -> bool:
    """True if the Metal kernels can run (MPS present and not disabled)."""
    if os.environ.get("MA_DISABLE_MPS_KERNELS") == "1":
        return False
    return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


def supports(query: torch.Tensor) -> bool:
    """True if the kernels can handle this input (device, rank, head_dim)."""
    return (query.device.type == "mps" and query.dim() == 4
            and 0 < query.shape[-1] <= MAX_HEAD_DIM and is_available())


@functools.lru_cache(maxsize=1)
def _row_library():
    return torch.mps.compile_shader(_ROW_SOURCE)


@functools.lru_cache(maxsize=8)
def _tiled_library(head_dim: int):
    return torch.mps.compile_shader(_TILED_SOURCE.replace("__HEAD_DIM__", str(head_dim)))


def _tiled_supported(head_dim: int) -> bool:
    return head_dim % 8 == 0 and head_dim <= MAX_TILED_HEAD_DIM


@functools.lru_cache(maxsize=64)
def _segment_tensor(segments: Tuple[Tuple[int, int], ...], device: torch.device) -> torch.Tensor:
    flat = [offset for segment in segments for offset in segment]
    return torch.tensor(flat, dtype=torch.int32, device=device)


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


class _SegmentAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, segments, tiled):
        B, S, H, D = query.shape
        seg = _segment_tensor(segments, query.device)
        scale = 1.0 / math.sqrt(D)
        nseg = len(segments)
        if tiled:
            S_pad = -(-S // _TILE) * _TILE
            q, k, v = (_pad_seq(t.contiguous(), S_pad) for t in (query, key, value))
            out = torch.empty_like(q)
            lse = torch.empty(B, S_pad, H, dtype=torch.float32, device=q.device)
            _tiled_library(D).tiled_forward(
                q, k, v, out, lse, seg, B, S, S_pad, H, nseg, scale, **_tiled_launch(B, H, S_pad))
        else:
            q, k, v = (t.contiguous() for t in (query, key, value))
            out = torch.empty_like(q)
            lse = torch.empty(B, S, H, dtype=torch.float32, device=q.device)
            _row_library().segment_attention_forward(
                q, k, v, out, lse, seg, B, S, H, D, nseg, scale, **_row_launch(B * H * S))
        ctx.save_for_backward(q, k, v, out, lse, seg)
        ctx.scale = scale
        ctx.tiled = tiled
        ctx.seq_len = S
        return out[:, :S]

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse, seg = ctx.saved_tensors
        B, S_stored, H, D = q.shape
        S = ctx.seq_len
        nseg = seg.numel() // 2
        grad_out = _pad_seq(grad_out.contiguous(), S_stored)
        delta = (grad_out * out).sum(-1).contiguous()
        need_q = ctx.needs_input_grad[0]
        need_kv = ctx.needs_input_grad[1] or ctx.needs_input_grad[2]
        grad_q = torch.empty_like(q) if need_q else None
        grad_k = torch.empty_like(k) if need_kv else None
        grad_v = torch.empty_like(v) if need_kv else None

        if ctx.tiled:
            lib = _tiled_library(D)
            launch = _tiled_launch(B, H, S_stored)
            if need_q:
                lib.tiled_backward_dq(q, k, v, grad_out, lse, delta, grad_q, seg,
                                      B, S, S_stored, H, nseg, ctx.scale, **launch)
            if need_kv:
                lib.tiled_backward_dkdv(q, k, v, grad_out, lse, delta, grad_k, grad_v, seg,
                                        B, S, S_stored, H, nseg, ctx.scale, **launch)
        else:
            lib = _row_library()
            launch = _row_launch(B * H * S)
            if need_q:
                lib.segment_attention_backward_dq(
                    q, k, v, grad_out, lse, delta, grad_q, seg, B, S, H, D, nseg, ctx.scale, **launch)
            if need_kv:
                lib.segment_attention_backward_dkdv(
                    q, k, v, grad_out, lse, delta, grad_k, grad_v, seg, B, S, H, D, nseg, ctx.scale, **launch)

        trim = (lambda g: g[:, :S] if g is not None else None)
        return trim(grad_q), trim(grad_k), trim(grad_v), None, None


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
    if not supports(query):
        raise ValueError("segment_attention needs MPS tensors shaped [batch, seq, heads, dim] "
                         f"with head_dim <= {MAX_HEAD_DIM}")
    if key.shape != query.shape or value.shape != query.shape:
        raise ValueError("query, key and value must have identical shapes")
    if key.device != query.device or value.device != query.device:
        raise ValueError("query, key and value must be on the same device")
    segments = tuple((int(lo), int(hi)) for lo, hi in segments)
    if not segments or any(lo > hi for lo, hi in segments):
        raise ValueError("segments must be a non-empty list of (lo, hi) with lo <= hi")

    head_dim = query.shape[-1]
    if kernel == "tiled" and not _tiled_supported(head_dim):
        raise ValueError(f"tiled kernel needs head_dim a multiple of 8 and <= {MAX_TILED_HEAD_DIM}")
    tiled = kernel == "tiled" or (kernel == "auto" and _tiled_supported(head_dim))

    dtype = query.dtype
    out = _SegmentAttention.apply(query.float(), key.float(), value.float(), segments, tiled)
    return out.to(dtype)


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
