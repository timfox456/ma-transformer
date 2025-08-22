import math
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


@pytest.mark.parametrize(
    "B,S,H,D",
    [
        (1, 1, 1, 1),
        (1, 7, 3, 5),  # odd sizes
        (2, 32, 2, 16),
        (3, 65, 1, 7),  # non power-of-two
        (2, 128, 4, 32),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_dense_cuda_extensive_parity(B, S, H, D, causal):
    from src.layers.ma_core_bridge import MACoreAttention, pytorch_dense_attention

    torch.manual_seed(123)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)

    attn = MACoreAttention(sparse=False, use_causal_mask=causal, fallback_training=False).eval()
    y_cuda = attn(q, k, v)
    y_ref = pytorch_dense_attention(q, k, v, use_causal_mask=causal)
    assert y_cuda.shape == y_ref.shape
    diff = max_abs_diff(y_cuda, y_ref)
    assert diff < 2e-4, f"dense CUDA parity failed (max diff {diff})"


@pytest.mark.parametrize(
    "B,S,H,D",
    [
        (1, 7, 2, 5),
        (2, 16, 4, 8),
        (2, 33, 1, 13),
    ],
)
@pytest.mark.parametrize("W", [0, 1, 2, 4, 8, 64])
def test_sparse_cuda_extensive_parity(B, S, H, D, W):
    from src.layers.ma_core_bridge import MACoreAttention, pytorch_sparse_attention

    torch.manual_seed(321)
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)

    attn = MACoreAttention(sparse=True, window_size=W, fallback_training=False).eval()
    y_cuda = attn(q, k, v)
    y_ref = pytorch_sparse_attention(q, k, v, window_size=W)
    assert y_cuda.shape == y_ref.shape
    diff = max_abs_diff(y_cuda, y_ref)
    assert diff < 2e-4, f"sparse CUDA parity failed (W={W}, max diff {diff})"


def test_numerical_stability_large_magnitudes():
    from src.layers.ma_core_bridge import MACoreAttention, pytorch_dense_attention
    torch.manual_seed(2024)
    B, S, H, D = 1, 33, 2, 16
    # Large magnitude inputs to stress softmax stability
    scale = 1e2
    q = (torch.randn(B, S, H, D, device="cuda") * scale).requires_grad_(True)
    k = (torch.randn(B, S, H, D, device="cuda") * scale).requires_grad_(True)
    v = torch.randn(B, S, H, D, device="cuda").requires_grad_(True)

    attn = MACoreAttention(sparse=False, use_causal_mask=False, fallback_training=False).eval()
    y_cuda = attn(q, k, v)
    y_ref = pytorch_dense_attention(q, k, v, use_causal_mask=False)
    assert torch.isfinite(y_cuda).all()
    assert torch.isfinite(y_ref).all()
    assert max_abs_diff(y_cuda, y_ref) < 5e-3


def test_noncontiguous_inputs_are_handled_by_bridge():
    from src.layers.ma_core_bridge import MACoreAttention, pytorch_dense_attention

    torch.manual_seed(0)
    B, S, H, D = 2, 17, 3, 9
    base = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    # Make tensors non-contiguous via slicing/transpose
    q = base[..., ::2].transpose(-1, -2).transpose(-1, -2).requires_grad_(True)
    k = base[..., ::2].transpose(1, 2).transpose(1, 2).requires_grad_(True)
    v = base[..., ::2].contiguous().requires_grad_(True)  # mix

    # Align shapes (S halved by ::2)
    S2 = q.size(1)
    q = q[:, :S2, :, :]
    k = k[:, :S2, :, :]
    v = v[:, :S2, :, :]

    attn = MACoreAttention(sparse=False, use_causal_mask=True, fallback_training=False).eval()
    y_cuda = attn(q, k, v)
    y_ref = pytorch_dense_attention(q, k, v, use_causal_mask=True)
    assert max_abs_diff(y_cuda, y_ref) < 2e-4


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dtype_fallback_to_pytorch(dtype):
    from src.layers.ma_core_bridge import MACoreAttention, pytorch_dense_attention

    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("BF16 not supported on this GPU")

    torch.manual_seed(42)
    B, S, H, D = 2, 33, 2, 8
    q = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    k = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
    v = torch.randn(B, S, H, D, device="cuda", dtype=dtype)

    attn = MACoreAttention(sparse=False, use_causal_mask=False, fallback_training=False).eval()
    # Should not throw; should produce reference result in same dtype
    y = attn(q, k, v)
    y_ref = pytorch_dense_attention(q, k, v, use_causal_mask=False)
    assert y.dtype == dtype
    assert max_abs_diff(y.float(), y_ref.float()) < 3e-3


def test_backward_grads_match_pytorch_dense():
    from src.layers.ma_core_bridge import MACoreAttention, pytorch_dense_attention

    torch.manual_seed(9)
    B, S, H, D = 1, 16, 2, 8
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)

    # CUDA forward, PyTorch backward under the hood in bridge
    attn = MACoreAttention(sparse=False, use_causal_mask=True, fallback_training=False)
    y = attn(q, k, v)
    loss = y.pow(2).mean()
    loss.backward()
    grads_cuda = (q.grad.clone(), k.grad.clone(), v.grad.clone())

    # Reset and compute PyTorch reference
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    y_ref = pytorch_dense_attention(q_ref, k_ref, v_ref, use_causal_mask=True)
    loss_ref = y_ref.pow(2).mean()
    loss_ref.backward()
    grads_ref = (q_ref.grad, k_ref.grad, v_ref.grad)

    for g_c, g_r in zip(grads_cuda, grads_ref):
        assert max_abs_diff(g_c, g_r) < 2e-3


def test_backward_grads_match_pytorch_sparse():
    from src.layers.ma_core_bridge import MACoreAttention, pytorch_sparse_attention

    torch.manual_seed(10)
    B, S, H, D, W = 2, 17, 2, 8, 3
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32, requires_grad=True)

    attn = MACoreAttention(sparse=True, window_size=W, fallback_training=False)
    y = attn(q, k, v)
    loss = y.sin().pow(2).mean()
    loss.backward()
    grads_cuda = (q.grad.clone(), k.grad.clone(), v.grad.clone())

    # Reference
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    y_ref = pytorch_sparse_attention(q_ref, k_ref, v_ref, window_size=W)
    loss_ref = y_ref.sin().pow(2).mean()
    loss_ref.backward()
    grads_ref = (q_ref.grad, k_ref.grad, v_ref.grad)

    for g_c, g_r in zip(grads_cuda, grads_ref):
        assert max_abs_diff(g_c, g_r) < 2e-3


def test_extension_error_messages_on_direct_call():
    import sparse_attention_cuda
    # Non-contiguous input should raise in direct CUDA ext (bridge makes it contiguous)
    B, S, H, D = 1, 8, 2, 4
    q = torch.randn(B, S, H, D, device="cuda").transpose(-1, -2)
    k = torch.randn(B, S, H, D, device="cuda")
    v = torch.randn(B, S, H, D, device="cuda")
    with pytest.raises(RuntimeError):
        _ = sparse_attention_cuda.forward(q, k, v, 2)
