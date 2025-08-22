#!/usr/bin/env python3
"""
PyTorch bridge for ma_core C++ attention engine.
Provides seamless integration between PyTorch tensors and ma_core tensors.
"""

import torch
import torch.nn as nn
import ma_core
import numpy as np
from typing import Optional, Tuple

# Optional CUDA sparse attention extension
import os
import sys
from pathlib import Path

def _try_import_sparse_ext():
    try:
        import sparse_attention_cuda  # type: ignore  # built by setup.py when CUDA available
        return sparse_attention_cuda
    except Exception:
        return None

# First attempt: normal import from site-packages
sparse_attention_cuda = _try_import_sparse_ext()
_HAS_SPARSE_CUDA = sparse_attention_cuda is not None

# Fallback: try to import from local build in src/ if present (editable dev)
if not _HAS_SPARSE_CUDA:
    try:
        this_dir = Path(__file__).resolve().parent  # .../src/layers
        src_dir = this_dir.parent  # .../src
        if str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))
        sparse_attention_cuda = _try_import_sparse_ext()
        _HAS_SPARSE_CUDA = sparse_attention_cuda is not None
    except Exception:
        sparse_attention_cuda = None
        _HAS_SPARSE_CUDA = False


class MACoreAttentionFunction(torch.autograd.Function):
    """
    Autograd function for ma_core attention computation.
    Handles forward pass through C++ engine and backward pass through PyTorch.
    """
    
    @staticmethod
    def forward(ctx, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                sparse: bool = False, window_size: int = 64, 
                use_causal_mask: bool = False) -> torch.Tensor:
        """
        Forward pass using ma_core C++ implementation.
        
        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]  
            value: Value tensor [batch, seq_len, num_heads, head_dim]
            sparse: Whether to use sparse attention
            window_size: Window size for sparse attention
            use_causal_mask: Whether to apply causal masking
            
        Returns:
            Output tensor [batch, seq_len, num_heads, head_dim]
        """
        # Save for backward pass
        ctx.save_for_backward(query, key, value)
        ctx.sparse = sparse
        ctx.window_size = window_size
        ctx.use_causal_mask = use_causal_mask
        
        # CUDA fast-paths when extension is available
        if _HAS_SPARSE_CUDA and query.is_cuda and key.is_cuda and value.is_cuda:
            if sparse:
                if query.dtype != torch.float32 or key.dtype != torch.float32 or value.dtype != torch.float32:
                    # fall back if dtype unsupported
                    output = pytorch_sparse_attention(query, key, value, window_size)
                else:
                    # Ensure contiguous for kernel access
                    cq = query.contiguous()
                    ck = key.contiguous()
                    cv = value.contiguous()
                    output = sparse_attention_cuda.forward(cq, ck, cv, int(window_size))
            else:
                if query.dtype != torch.float32 or key.dtype != torch.float32 or value.dtype != torch.float32:
                    output = pytorch_dense_attention(query, key, value, use_causal_mask)
                else:
                    cq = query.contiguous()
                    ck = key.contiguous()
                    cv = value.contiguous()
                    output = sparse_attention_cuda.dense_forward(cq, ck, cv, bool(use_causal_mask))
        else:
            # Prefer staying on-device if any tensor is non-CPU and we don't have a specialized path
            any_non_cpu = (query.device.type != 'cpu' or key.device.type != 'cpu' or value.device.type != 'cpu')
            if any_non_cpu:
                if sparse:
                    output = pytorch_sparse_attention(query, key, value, window_size)
                else:
                    output = pytorch_dense_attention(query, key, value, use_causal_mask)
            else:
                # CPU path through ma_core (pybind11 C++)
                batch_size, seq_len, num_heads, head_dim = query.shape
                mc_query = pytorch_to_ma_core(query)
                mc_key = pytorch_to_ma_core(key)
                mc_value = pytorch_to_ma_core(value)
                if sparse:
                    mc_output = ma_core.compute_sparse_attention(mc_query, mc_key, mc_value, window_size)
                else:
                    mc_output = ma_core.compute_dense_attention(mc_query, mc_key, mc_value, use_causal_mask)
                output = ma_core_to_pytorch(mc_output, query.device, query.dtype)
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Backward pass using PyTorch's autograd.
        For now, we fall back to PyTorch implementation for gradients.
        """
        query, key, value = ctx.saved_tensors
        
        # Use PyTorch implementation for backward pass
        # This ensures gradients work correctly for training
        if ctx.sparse:
            # Simple PyTorch sparse attention for gradients
            output = pytorch_sparse_attention(query, key, value, ctx.window_size)
        else:
            output = pytorch_dense_attention(query, key, value, ctx.use_causal_mask)
        
        # Compute gradients using PyTorch autograd
        query_grad = key_grad = value_grad = None
        if ctx.needs_input_grad[0]:
            query_grad = torch.autograd.grad(output, query, grad_output, retain_graph=True)[0]
        if ctx.needs_input_grad[1]:
            key_grad = torch.autograd.grad(output, key, grad_output, retain_graph=True)[0]
        if ctx.needs_input_grad[2]:
            value_grad = torch.autograd.grad(output, value, grad_output, retain_graph=True)[0]
        
        return query_grad, key_grad, value_grad, None, None, None


def pytorch_to_ma_core(tensor: torch.Tensor) -> 'ma_core.Tensor':
    """Convert PyTorch tensor to ma_core tensor."""
    # Ensure tensor is contiguous and on CPU for C++ processing
    cpu_tensor = tensor.detach().cpu().contiguous()
    # Shape in ma_core format: [batch, seq, heads, dim]
    batch_size, seq_len, num_heads, head_dim = cpu_tensor.shape
    shape = [batch_size, seq_len, num_heads, head_dim]
    # Create ma_core tensor and bulk copy from NumPy
    mc_tensor = ma_core.create_tensor(shape)
    np_view = cpu_tensor.numpy()  # shares memory with cpu_tensor
    mc_tensor.copy_from_numpy(np_view)
    return mc_tensor


def ma_core_to_pytorch(mc_tensor: 'ma_core.Tensor', device: torch.device, 
                      dtype: torch.dtype) -> torch.Tensor:
    """Convert ma_core tensor to PyTorch tensor."""
    # Pull data to NumPy (float32), then wrap as torch tensor and move/cast
    np_arr = mc_tensor.to_numpy()
    out_cpu = torch.from_numpy(np_arr)  # shares memory (CPU)
    if out_cpu.dtype != dtype:
        out_cpu = out_cpu.to(dtype)
    if device.type != 'cpu':
        return out_cpu.to(device)
    return out_cpu


def pytorch_dense_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                           use_causal_mask: bool = False) -> torch.Tensor:
    """PyTorch reference implementation of dense attention."""
    batch_size, seq_len, num_heads, head_dim = query.shape
    
    # Scale queries
    scale = 1.0 / (head_dim ** 0.5)
    query = query * scale
    
    # Compute attention scores: Q @ K^T
    # We need [batch, seq, num_heads, seq] from [batch, seq, num_heads, head_dim] @ [batch, seq, head_dim, num_heads]
    # Transpose key to [batch, seq, head_dim, num_heads], then transpose last 2 dims to [batch, head_dim, seq, num_heads]
    # Actually, let's reshape for proper matrix multiplication
    
    # Reshape to [batch * num_heads, seq, head_dim]
    q_reshaped = query.transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, head_dim)
    k_reshaped = key.transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, head_dim)
    v_reshaped = value.transpose(1, 2).contiguous().view(batch_size * num_heads, seq_len, head_dim)
    
    # Compute attention scores [batch * num_heads, seq, seq]
    scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1))
    
    # Apply causal mask if requested
    if use_causal_mask:
        # Create causal mask with shape [seq_len, seq_len]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, -1e9)
    
    # Apply softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Apply to values [batch * num_heads, seq, head_dim]
    output_reshaped = torch.matmul(attention_weights, v_reshaped)
    
    # Reshape back to [batch, seq, num_heads, head_dim]
    output = output_reshaped.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2).contiguous()
    
    return output


def pytorch_sparse_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                            window_size: int = 64) -> torch.Tensor:
    """PyTorch reference implementation of sparse attention."""
    batch_size, seq_len, num_heads, head_dim = query.shape
    scale = 1.0 / (head_dim ** 0.5)
    
    output = torch.zeros_like(value)
    
    for b in range(batch_size):
        for h in range(num_heads):
            for i in range(seq_len):
                # Sliding window range
                start_j = max(0, i - window_size)
                end_j = min(seq_len, i + window_size + 1)
                
                # Compute scores for window
                q_i = query[b, i, h, :] * scale
                k_window = key[b, start_j:end_j, h, :]
                
                scores = torch.matmul(k_window, q_i)
                attention_weights = torch.softmax(scores, dim=0)
                
                # Apply to values
                v_window = value[b, start_j:end_j, h, :]
                output[b, i, h, :] = torch.matmul(attention_weights, v_window)
    
    return output


class MACoreAttention(nn.Module):
    """
    PyTorch module wrapping ma_core attention computation.
    
    This module provides a drop-in replacement for attention layers,
    using the high-performance C++ ma_core engine for inference
    while maintaining PyTorch compatibility for training.
    """
    
    def __init__(self, sparse: bool = False, window_size: int = 64, 
                 use_causal_mask: bool = False, fallback_training: bool = True):
        """
        Initialize MACoreAttention module.
        
        Args:
            sparse: Whether to use sparse attention pattern
            window_size: Window size for sparse attention
            use_causal_mask: Whether to apply causal masking for dense attention
            fallback_training: Whether to fall back to PyTorch during training
        """
        super().__init__()
        self.sparse = sparse
        self.window_size = window_size
        self.use_causal_mask = use_causal_mask
        self.fallback_training = fallback_training
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through attention.
        
        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor [batch, seq_len, num_heads, head_dim]
            value: Value tensor [batch, seq_len, num_heads, head_dim]
            
        Returns:
            Output tensor [batch, seq_len, num_heads, head_dim]
        """
        
        # During training, optionally fall back to PyTorch for gradient compatibility
        if self.training and self.fallback_training:
            if self.sparse:
                return pytorch_sparse_attention(query, key, value, self.window_size)
            else:
                return pytorch_dense_attention(query, key, value, self.use_causal_mask)
        
        # Use ma_core C++ engine (with autograd support)
        return MACoreAttentionFunction.apply(
            query, key, value, self.sparse, self.window_size, self.use_causal_mask
        )


def create_attention_layer(attention_type: str = "dense", **kwargs) -> nn.Module:
    """
    Factory function to create attention layers.
    
    Args:
        attention_type: "dense", "sparse", or "auto"
        **kwargs: Additional arguments for attention configuration
        
    Returns:
        Attention layer module
    """
    if attention_type == "dense":
        return MACoreAttention(sparse=False, **kwargs)
    elif attention_type == "sparse":
        return MACoreAttention(sparse=True, **kwargs)
    elif attention_type == "auto":
        # Choose based on sequence length or other heuristics
        return MACoreAttention(sparse=True, **kwargs)  # Default to sparse for efficiency
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
