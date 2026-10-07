# SPDX-License-Identifier: Apache-2.0
"""
Device dispatch for the differentiable sparse attention patterns.

MPS tensors use the Metal kernels in mps_attention when they support the input
(block-sparse and Longformer need head_dim a multiple of 8, up to 128);
everything else uses the vectorized PyTorch implementation in blocked_attention. Set MA_DISABLE_MPS_KERNELS=1 to
force the PyTorch path on Apple silicon.

Tensors are [batch, seq, heads, dim].
"""

import torch

from . import blocked_attention, mps_attention


def sliding_window_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                             window_size: int, causal: bool = False) -> torch.Tensor:
    """Query i attends to keys j with |i - j| <= window_size (and j <= i if causal)."""
    if mps_attention.supports(query):
        return mps_attention.sliding_window_attention(query, key, value, window_size, causal)
    return blocked_attention.sliding_window_attention(query, key, value, window_size, causal)


def financial_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                        local_window_size: int = 512, dilation_stride: int = 1000,
                        dilation_cluster_size: int = 8, dilation_num_clusters: int = 10) -> torch.Tensor:
    """Causal local window plus dilated clusters (ma_core's FINANCIAL pattern)."""
    params = dict(local_window_size=local_window_size, dilation_stride=dilation_stride,
                  dilation_cluster_size=dilation_cluster_size, dilation_num_clusters=dilation_num_clusters)
    if mps_attention.supports(query):
        return mps_attention.financial_attention(query, key, value, **params)
    return blocked_attention.financial_attention(query, key, value, **params)


def block_sparse_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                           block_size: int = 64) -> torch.Tensor:
    """Query i attends to key j when |i // block_size - j // block_size| <= 1."""
    if mps_attention.supports_tiled(query):
        return mps_attention.block_sparse_attention(query, key, value, block_size)
    return blocked_attention.block_sparse_attention(query, key, value, block_size)


def longformer_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                         window_size: int = 64, num_global_tokens: int = 2) -> torch.Tensor:
    """Sliding window of radius window_size plus num_global_tokens global tokens at the start."""
    if mps_attention.supports_tiled(query):
        return mps_attention.longformer_attention(query, key, value, window_size, num_global_tokens)
    return blocked_attention.longformer_attention(query, key, value, window_size, num_global_tokens)
