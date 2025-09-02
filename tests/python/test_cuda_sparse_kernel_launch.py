import pytest
import torch


def _has_cuda_ext():
    try:
        import sparse_attention_cuda  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not torch.cuda.is_available() or not _has_cuda_ext(),
                    reason="CUDA or sparse_attention_cuda not available")
@pytest.mark.parametrize("S", [1, 17, 33, 64, 127])
@pytest.mark.parametrize("W", [0, 1, 4, 8])
def test_sparse_forward_cuda_launch_edgecases(S, W):
    from src.layers.ma_core_bridge import pytorch_sparse_attention
    import sparse_attention_cuda as ext

    torch.manual_seed(123)
    B, H, D = 2, 2, 16
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)

    y_cuda = ext.forward(q, k, v, int(W))
    y_ref = pytorch_sparse_attention(q, k, v, window_size=int(W))

    torch.testing.assert_close(y_cuda, y_ref, rtol=1e-4, atol=1e-4)

