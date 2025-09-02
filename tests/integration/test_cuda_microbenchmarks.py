# SPDX-License-Identifier: Apache-2.0
"""
Small CUDA microbenchmarks to catch performance regressions on tricky sizes.
Compares sparse_attention_cuda forward vs PyTorch reference for S in {17, 33, 65}.
"""

import pytest
import torch
import time


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


def _timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) * 1000.0 / iters


@pytest.mark.benchmark
@pytest.mark.parametrize("S", [17, 33, 65])
def test_cuda_sparse_microbenchmark_vs_pytorch(S):
    from src.layers.ma_core_bridge import pytorch_sparse_attention
    import sparse_attention_cuda as ext

    torch.manual_seed(0)
    B, H, D, W = 2, 4, 32, 8
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)

    def run_ext():
        ext.forward(q, k, v, W)

    def run_ref():
        pytorch_sparse_attention(q, k, v, window_size=W)

    t_ref = _timeit(run_ref)
    t_ext = _timeit(run_ext)

    # Print for visibility in CI logs
    print(f"Sparse CUDA vs PyTorch @ S={S}: ext={t_ext:.3f}ms, ref={t_ref:.3f}ms, speedup={t_ref/max(t_ext,1e-9):.2f}x")

    # Mild regression guard: extension should not be more than 2x slower
    assert t_ext < 2.0 * t_ref + 0.05, f"Sparse CUDA too slow at S={S}: {t_ext:.3f}ms vs {t_ref:.3f}ms"


@pytest.mark.benchmark
@pytest.mark.parametrize("S", [17, 33, 65])
@pytest.mark.parametrize("causal", [False, True])
def test_cuda_dense_microbenchmark_vs_pytorch(S, causal):
    from src.layers.ma_core_bridge import pytorch_dense_attention
    import sparse_attention_cuda as ext

    torch.manual_seed(0)
    B, H, D = 2, 4, 32
    q = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    k = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)
    v = torch.randn(B, S, H, D, device="cuda", dtype=torch.float32)

    def run_ext():
        ext.dense_forward(q, k, v, bool(causal))

    def run_ref():
        pytorch_dense_attention(q, k, v, use_causal_mask=bool(causal))

    t_ref = _timeit(run_ref, iters=30, warmup=8)
    t_ext = _timeit(run_ext, iters=30, warmup=8)

    print(f"Dense CUDA vs PyTorch @ S={S}, causal={causal}: ext={t_ext:.3f}ms, ref={t_ref:.3f}ms, speedup={t_ref/max(t_ext,1e-9):.2f}x")

    # Mild regression guard: extension should not be more than 2x slower
    assert t_ext < 2.0 * t_ref + 0.05, f"Dense CUDA too slow at S={S}: {t_ext:.3f}ms vs {t_ref:.3f}ms"

