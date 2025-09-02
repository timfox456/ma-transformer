# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Check if ma_core C++ extension is available
try:
    import ma_core
    from .ma_core_bridge import MACoreAttention, pytorch_sparse_attention
    MA_CORE_AVAILABLE = True
    print("ma_core C++ engine available. Using high-performance attention implementation.")
except ImportError:
    MA_CORE_AVAILABLE = False
    print("ma_core extension not available. Falling back to PyTorch implementation.")

# Legacy CUDA check for backward compatibility
try:
    import ma_transformer_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


class SparseAttention(nn.Module):
    """
    Sparse Attention layer with multiple implementation backends.
    Priority: ma_core C++ > CUDA > PyTorch fallback.
    Uses a sliding window attention pattern.
    """
    def __init__(self, window_size=3, use_ma_core=True):
        super(SparseAttention, self).__init__()
        if not isinstance(window_size, int):
            raise TypeError("window_size must be an integer for SparseAttention")
        if window_size <= 0:
            raise ValueError("window_size must be > 0 for SparseAttention")
        self.window_size = window_size
        self.use_ma_core = use_ma_core
        
        # Initialize ma_core attention if available
        if MA_CORE_AVAILABLE and use_ma_core:
            self.ma_core_attention = MACoreAttention(
                sparse=True, 
                window_size=window_size,
                fallback_training=True  # Use PyTorch during training for gradients
            )

    def forward(self, q, k, v):
        """
        Forward pass for Sparse Attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, seq_len, model_dim)
            k (torch.Tensor): Key tensor of shape (batch_size, seq_len, model_dim)  
            v (torch.Tensor): Value tensor of shape (batch_size, seq_len, model_dim)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, model_dim)
        """
        
        # Reshape to multi-head format if needed
        batch_size, seq_len, model_dim = q.shape
        
        # For simplicity, treat as single head attention
        # In production, you'd want proper multi-head handling
        num_heads = 1
        head_dim = model_dim
        
        # Reshape: [batch, seq, model_dim] -> [batch, seq, heads, head_dim]
        q_reshaped = q.unsqueeze(2)  # [batch, seq, 1, model_dim]
        k_reshaped = k.unsqueeze(2)
        v_reshaped = v.unsqueeze(2)
        
        # Use ma_core C++ engine if available (highest priority)
        if MA_CORE_AVAILABLE and self.use_ma_core and hasattr(self, 'ma_core_attention'):
            output = self.ma_core_attention(q_reshaped, k_reshaped, v_reshaped)
            return output.squeeze(2)  # Remove head dimension
        
        # Fall back to CUDA implementation if available
        elif self.training is False and CUDA_AVAILABLE:
            return ma_transformer_cuda.forward(q, k, v)
        
        # Fall back to PyTorch implementation
        else:
            return self._pytorch_forward(q, k, v)

    def _pytorch_forward(self, q, k, v):
        batch_size, seq_len, model_dim = q.size()
        
        # Scale queries
        q = q / math.sqrt(model_dim)

        # Create a sliding window mask
        mask = torch.ones(seq_len, seq_len, device=q.device).tril(self.window_size - 1).triu(-self.window_size + 1)
        mask = mask.bool()

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1))

        # Apply the mask
        attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        return output
