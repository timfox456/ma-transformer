
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Check if the CUDA extension is available
try:
    import ma_transformer_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("CUDA extension not available. Falling back to PyTorch implementation.")


class SparseAttention(nn.Module):
    """
    Sparse Attention layer with a fallback to a PyTorch implementation.
    Uses a sliding window attention pattern.
    """
    def __init__(self, window_size=3):
        super(SparseAttention, self).__init__()
        self.window_size = window_size

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
        if self.training or not CUDA_AVAILABLE:
            # Use PyTorch implementation for training or if CUDA is not available
            return self._pytorch_forward(q, k, v)
        else:
            # Use CUDA implementation for inference
            return ma_transformer_cuda.forward(q, k, v)

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

