# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from .blocked_attention import sliding_window_attention

# Check if ma_core C++ extension is available
try:
    import ma_core
    from .ma_core_bridge import MACoreAttention, pytorch_sparse_attention
    MA_CORE_AVAILABLE = True
    print("ma_core C++ engine available. Using high-performance attention implementation.")
except ImportError:
    MA_CORE_AVAILABLE = False
    print("ma_core extension not available. Falling back to PyTorch implementation.")

class SparseAttention(nn.Module):
    """
    Sparse Attention layer with multiple implementation backends.
    Priority: ma_core bridge (C++ on CPU, CUDA kernels on GPU) > PyTorch fallback.
    Uses a sliding window attention pattern: query i attends to keys j with
    |i - j| <= window_size.
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

        return self._pytorch_forward(q, k, v)

    def _pytorch_forward(self, q, k, v):
        out = sliding_window_attention(q.unsqueeze(2), k.unsqueeze(2), v.unsqueeze(2), self.window_size)
        return out.squeeze(2)
