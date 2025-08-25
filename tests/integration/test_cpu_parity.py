import pytest
import torch


@pytest.mark.parametrize("B,S,H,D", [
    (1, 8, 2, 4),
    (2, 16, 4, 8),
])
@pytest.mark.parametrize("causal", [False, True])
def test_dense_cpu_matches_pytorch(B, S, H, D, causal):
    from src.layers.ma_core_bridge import MACoreAttention, pytorch_dense_attention

    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, dtype=torch.float32)
    k = torch.randn(B, S, H, D, dtype=torch.float32)
    v = torch.randn(B, S, H, D, dtype=torch.float32)

    attn = MACoreAttention(sparse=False, use_causal_mask=causal, fallback_training=True).eval()
    y_cpu = attn(q, k, v)  # Should use ma_core CPU path in eval
    y_ref = pytorch_dense_attention(q, k, v, use_causal_mask=causal)

    torch.testing.assert_close(y_cpu, y_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("B,S,H,D", [
    (1, 8, 2, 4),
    (2, 20, 2, 8),
])
@pytest.mark.parametrize("W", [1, 2, 4])
def test_sparse_cpu_matches_pytorch(B, S, H, D, W):
    from src.layers.ma_core_bridge import MACoreAttention, pytorch_sparse_attention

    torch.manual_seed(42)
    q = torch.randn(B, S, H, D, dtype=torch.float32)
    k = torch.randn(B, S, H, D, dtype=torch.float32)
    v = torch.randn(B, S, H, D, dtype=torch.float32)

    attn = MACoreAttention(sparse=True, window_size=W, fallback_training=True).eval()
    y_cpu = attn(q, k, v)  # ma_core CPU path in eval
    y_ref = pytorch_sparse_attention(q, k, v, window_size=W)

    torch.testing.assert_close(y_cpu, y_ref, rtol=1e-5, atol=1e-6)

