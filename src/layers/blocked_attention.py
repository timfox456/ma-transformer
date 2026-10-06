# SPDX-License-Identifier: Apache-2.0
"""
Vectorized sparse attention in plain PyTorch.

Queries are processed in blocks. For each block we gather only the keys the
pattern can reach, compute a small dense score matrix, and mask the entries
outside the pattern. Memory is O(block * keys_per_block) instead of
O(seq_len^2), every operation is a batched matmul or gather, and autograd
works unchanged, so the same code serves CPU, CUDA and MPS.

All functions take and return tensors shaped [batch, seq, heads, dim] and use
the same pattern definitions as the ma_core C++ engine.
"""

import math
from typing import Callable, List, Tuple

import torch

# Each callable maps a query block [start, end) to the key ranges it may need,
# or builds the boolean mask for (query, key) index pairs.
RangeFn = Callable[[int, int], List[Tuple[int, int]]]
MaskFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

DEFAULT_BLOCK_SIZE = 256


def _blocked_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                       key_ranges: RangeFn, mask_fn: MaskFn,
                       block_size: int = DEFAULT_BLOCK_SIZE) -> torch.Tensor:
    if query.dim() != 4 or key.shape != query.shape or value.shape[:3] != query.shape[:3]:
        raise ValueError("Expected query, key and value shaped [batch, seq_len, num_heads, head_dim]")
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    seq_len = query.shape[1]
    scale = 1.0 / math.sqrt(query.shape[-1])
    device = query.device

    # [batch, heads, seq, dim] so the score matmuls batch over (batch, heads)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    outputs = []
    for start in range(0, seq_len, block_size):
        end = min(start + block_size, seq_len)
        ranges = [(max(lo, 0), min(hi, seq_len)) for lo, hi in key_ranges(start, end)]
        ranges = [(lo, hi) for lo, hi in ranges if lo < hi]
        key_idx = torch.unique(torch.cat([torch.arange(lo, hi) for lo, hi in ranges])).to(device)
        query_idx = torch.arange(start, end, device=device)

        scores = torch.matmul(q[:, :, start:end], k[:, :, key_idx].transpose(-2, -1)) * scale
        allowed = mask_fn(query_idx[:, None], key_idx[None, :])
        scores = scores.masked_fill(~allowed, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        outputs.append(torch.matmul(weights, v[:, :, key_idx]))

    return torch.cat(outputs, dim=2).transpose(1, 2).contiguous()


def sliding_window_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                             window_size: int, causal: bool = False,
                             block_size: int = DEFAULT_BLOCK_SIZE) -> torch.Tensor:
    """Each query i attends to keys j with |i - j| <= window_size (and j <= i if causal)."""
    if window_size < 0:
        raise ValueError("window_size must be >= 0")

    def key_ranges(start, end):
        return [(start - window_size, end if causal else end + window_size)]

    def mask_fn(i, j):
        allowed = (i - j).abs() <= window_size
        return allowed & (j <= i) if causal else allowed

    return _blocked_attention(query, key, value, key_ranges, mask_fn, block_size)


def financial_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        local_window_size: int = 512, dilation_stride: int = 1000,
                        dilation_cluster_size: int = 8, dilation_num_clusters: int = 10,
                        block_size: int = DEFAULT_BLOCK_SIZE) -> torch.Tensor:
    """
    Causal local window plus dilated clusters, matching ma_core's FINANCIAL pattern.

    Query i attends to j in [i - local_window_size + 1, i] and, for each
    k = 1..dilation_num_clusters, to j in
    [i - k*dilation_stride - dilation_cluster_size + 1, i - k*dilation_stride].
    """
    if local_window_size <= 0 or dilation_stride <= 0 or dilation_cluster_size <= 0 \
            or dilation_num_clusters < 0:
        raise ValueError("Financial attention parameters must be positive")

    def key_ranges(start, end):
        ranges = [(start - local_window_size + 1, end)]
        for c in range(1, dilation_num_clusters + 1):
            offset = c * dilation_stride
            ranges.append((start - offset - dilation_cluster_size + 1, end - offset))
        return ranges

    def mask_fn(i, j):
        allowed = (j <= i) & (j > i - local_window_size)
        for c in range(1, dilation_num_clusters + 1):
            cluster_end = i - c * dilation_stride
            allowed = allowed | ((j <= cluster_end) & (j > cluster_end - dilation_cluster_size))
        return allowed

    return _blocked_attention(query, key, value, key_ranges, mask_fn, block_size)


def block_sparse_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                           block_size: int = 64, block: int = DEFAULT_BLOCK_SIZE) -> torch.Tensor:
    """Query i attends to key j when |i // block_size - j // block_size| <= 1 (ma_core's BLOCK_SPARSE)."""
    if block_size <= 0:
        raise ValueError("block_size must be > 0")

    def key_ranges(start, end):
        return [((start // block_size - 1) * block_size, ((end - 1) // block_size + 2) * block_size)]

    def mask_fn(i, j):
        return (i // block_size - j // block_size).abs() <= 1

    return _blocked_attention(query, key, value, key_ranges, mask_fn, block)


def longformer_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                         window_size: int = 64, num_global_tokens: int = 2,
                         block: int = DEFAULT_BLOCK_SIZE) -> torch.Tensor:
    """
    Query i attends to key j when |i - j| <= window_size or either is one of
    the first num_global_tokens positions (ma_core's LONGFORMER).
    """
    if window_size < 0 or num_global_tokens < 0:
        raise ValueError("window_size and num_global_tokens must be >= 0")
    seq_len = query.shape[1]

    def key_ranges(start, end):
        if start < num_global_tokens:
            return [(0, seq_len)]
        return [(0, num_global_tokens), (start - window_size, end + window_size)]

    def mask_fn(i, j):
        return ((i - j).abs() <= window_size) | (i < num_global_tokens) | (j < num_global_tokens)

    return _blocked_attention(query, key, value, key_ranges, mask_fn, block)
