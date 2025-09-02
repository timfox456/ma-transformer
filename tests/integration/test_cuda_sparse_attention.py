 # SPDX-License-Identifier: Apache-2.0
import os
import pytest
import torch


def cuda_ext_available():
    try:
        import sparse_attention_cuda  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not torch.cuda.is_available() or not cuda_ext_available(), reason="CUDA or extension not available")
def test_sparse_attention_cuda_matches_pytorch():
    torch.manual_seed(0)
    from src.layers.ma_core_bridge import pytorch_sparse_attention
    import sparse_attention_cuda

    B, S, H, D = 2, 32, 4, 16
    W = 4
    q = torch.randn(B, S, H, D, device='cuda', dtype=torch.float32)
    k = torch.randn(B, S, H, D, device='cuda', dtype=torch.float32)
    v = torch.randn(B, S, H, D, device='cuda', dtype=torch.float32)

    out_cuda = sparse_attention_cuda.forward(q, k, v, W)
    out_ref = pytorch_sparse_attention(q, k, v, W)

    assert torch.allclose(out_cuda, out_ref, atol=1e-3, rtol=1e-3)
