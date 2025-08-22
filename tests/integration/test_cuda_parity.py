import pytest
import torch


def _has_cuda_ext():
    try:
        import sparse_attention_cuda  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not _has_cuda_ext(),
    reason="CUDA not available or sparse_attention_cuda extension not built",
)


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


@pytest.mark.parametrize("B,S,H,D", [(1, 32, 2, 16), (2, 64, 4, 32)])
@pytest.mark.parametrize("causal", [False, True])
def test_dense_cuda_matches_pytorch(B, S, H, D, causal):
    from src.layers.ma_core_bridge import MACoreAttention, pytorch_dense_attention

    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)

    attn = MACoreAttention(sparse=False, use_causal_mask=causal, fallback_training=False).eval()
    y_cuda = attn(q, k, v)
    y_ref = pytorch_dense_attention(q, k, v, use_causal_mask=causal)

    diff = max_abs_diff(y_cuda, y_ref)
    assert diff < 1e-4, f"dense CUDA parity failed (max diff {diff})"


@pytest.mark.parametrize("B,S,H,D", [(1, 32, 2, 16), (2, 64, 4, 32)])
@pytest.mark.parametrize("W", [1, 4, 8])
def test_sparse_cuda_matches_pytorch(B, S, H, D, W):
    from src.layers.ma_core_bridge import MACoreAttention, pytorch_sparse_attention

    torch.manual_seed(0)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)

    attn = MACoreAttention(sparse=True, window_size=W, fallback_training=False).eval()
    y_cuda = attn(q, k, v)
    y_ref = pytorch_sparse_attention(q, k, v, window_size=W)

    diff = max_abs_diff(y_cuda, y_ref)
    assert diff < 1e-4, f"sparse CUDA parity failed (max diff {diff})"

