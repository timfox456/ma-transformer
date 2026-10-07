# SPDX-License-Identifier: Apache-2.0
"""
Numerical parity tests for every attention backend.

Each sparsity pattern is defined here independently, as an explicit set of
allowed (query, key) pairs, and attention is computed densely in float64.
Every backend (ma_core C++ entry points, the vectorized PyTorch path on CPU and
MPS, and the PyTorch bridge) must match that reference on random inputs,
including shapes with several heads and sequence lengths that do not divide
evenly into blocks.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from layers import blocked_attention  # noqa: E402
from layers.blocked_attention import financial_attention, sliding_window_attention  # noqa: E402
from layers.sparse_attention import SparseAttention  # noqa: E402

try:
    import ma_core
    from layers.ma_core_bridge import MACoreAttention, ma_core_to_pytorch, pytorch_to_ma_core
    HAS_MA_CORE = True
except ImportError:
    HAS_MA_CORE = False

needs_ma_core = pytest.mark.skipif(not HAS_MA_CORE, reason="ma_core extension not built")

DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []) \
    + (["cuda"] if torch.cuda.is_available() else [])

SHAPES = [(1, 17, 1, 8), (2, 64, 3, 16), (2, 300, 2, 32)]

# Small financial parameters so clusters overlap the local window
FIN_SMALL = dict(local_window_size=8, dilation_stride=5, dilation_cluster_size=3, dilation_num_clusters=3)


# --- Reference patterns, written as plain sets of (query, key) pairs ---------

def window_pairs(S, w, causal=False):
    return {(i, j) for i in range(S) for j in range(S) if abs(i - j) <= w and (not causal or j <= i)}


def block_sparse_pairs(S, b):
    return {(i, j) for i in range(S) for j in range(S) if abs(i // b - j // b) <= 1}


def longformer_pairs(S, w, g):
    return {(i, j) for i in range(S) for j in range(S) if i < g or j < g or abs(i - j) <= w}


def financial_pairs(S, local_window_size, dilation_stride, dilation_cluster_size, dilation_num_clusters):
    pairs = set()
    for i in range(S):
        pairs.update((i, j) for j in range(max(0, i - local_window_size + 1), i + 1))
        for c in range(1, dilation_num_clusters + 1):
            end = i - c * dilation_stride
            pairs.update((i, j) for j in range(max(0, end - dilation_cluster_size + 1), end + 1))
    return pairs


def full_pairs(S, causal=False):
    return {(i, j) for i in range(S) for j in range(S) if not causal or j <= i}


def reference_attention(q, k, v, pairs):
    """Dense masked attention in float64. q, k, v are [batch, seq, heads, dim]."""
    S = q.shape[1]
    mask = torch.zeros(S, S, dtype=torch.bool)
    for i, j in pairs:
        mask[i, j] = True
    q, k, v = (t.cpu().double() for t in (q, k, v))
    scores = torch.einsum("bihd,bjhd->bhij", q, k) / q.shape[-1] ** 0.5
    weights = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    return torch.einsum("bhij,bjhd->bihd", weights, v)


def random_qkv(shape, dtype=torch.float32, device="cpu", seed=0):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randn(shape, generator=gen, dtype=dtype).to(device) for _ in range(3)]


def assert_close(actual, expected, atol=2e-5):
    actual = actual.detach().cpu().double()
    err = (actual - expected).abs().max().item()
    assert err <= atol, f"max abs error {err:.3g} exceeds {atol:.0e}"


# --- ma_core C++ engine --------------------------------------------------------

def run_ma_core(q, k, v, config):
    out = ma_core.compute_attention(pytorch_to_ma_core(q), pytorch_to_ma_core(k), pytorch_to_ma_core(v), config)
    return ma_core_to_pytorch(out, q.device, q.dtype)


def make_config(pattern, **fields):
    config = ma_core.AttentionConfig(getattr(ma_core.AttentionPattern, pattern))
    for name, value in fields.items():
        setattr(config, name, value)
    return config


@needs_ma_core
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("pattern, fields, pairs_fn", [
    ("FULL", {}, lambda S: full_pairs(S)),
    ("CAUSAL", {}, lambda S: full_pairs(S, causal=True)),
    ("SLIDING_WINDOW", {"window_size": 5}, lambda S: window_pairs(S, 5)),
    ("BLOCK_SPARSE", {"block_size": 16}, lambda S: block_sparse_pairs(S, 16)),
    ("LONGFORMER", {"window_size": 3, "num_global_tokens": 2}, lambda S: longformer_pairs(S, 3, 2)),
    ("FINANCIAL", FIN_SMALL, lambda S: financial_pairs(S, **FIN_SMALL)),
])
def test_compute_attention_matches_reference(shape, pattern, fields, pairs_fn):
    q, k, v = random_qkv(shape)
    out = run_ma_core(q, k, v, make_config(pattern, **fields))
    assert_close(out, reference_attention(q, k, v, pairs_fn(shape[1])))


@needs_ma_core
@pytest.mark.parametrize("seq_len", [1024, 2048])
def test_financial_default_config_long_sequence(seq_len):
    """Regression: these lengths used to crash the process with an out-of-bounds write."""
    shape = (1, seq_len, 2, 16)
    q, k, v = random_qkv(shape)
    out = run_ma_core(q, k, v, make_config("FINANCIAL"))
    defaults = dict(local_window_size=512, dilation_stride=1000, dilation_cluster_size=8, dilation_num_clusters=10)
    assert_close(out, reference_attention(q, k, v, financial_pairs(seq_len, **defaults)))


@needs_ma_core
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("causal", [False, True])
def test_compute_dense_attention_matches_reference(shape, causal):
    q, k, v = random_qkv(shape)
    mq, mk, mv = map(pytorch_to_ma_core, (q, k, v))
    out = ma_core_to_pytorch(ma_core.compute_dense_attention(mq, mk, mv, causal), q.device, q.dtype)
    assert_close(out, reference_attention(q, k, v, full_pairs(shape[1], causal)))


@needs_ma_core
@pytest.mark.parametrize("shape", SHAPES)
def test_compute_sparse_attention_matches_reference(shape):
    q, k, v = random_qkv(shape)
    mq, mk, mv = map(pytorch_to_ma_core, (q, k, v))
    out = ma_core_to_pytorch(ma_core.compute_sparse_attention(mq, mk, mv, 4), q.device, q.dtype)
    assert_close(out, reference_attention(q, k, v, window_pairs(shape[1], 4)))


# --- Vectorized PyTorch path ----------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("block_size", [1, 7, 64, 256])
@pytest.mark.parametrize("causal", [False, True])
def test_sliding_window_attention_matches_reference(device, shape, block_size, causal):
    q, k, v = random_qkv(shape, device=device)
    out = sliding_window_attention(q, k, v, 5, causal=causal, block_size=block_size)
    assert out.device.type == device
    assert_close(out, reference_attention(q, k, v, window_pairs(shape[1], 5, causal)), atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("block_size", [1, 7, 64])
def test_financial_attention_matches_reference(device, shape, block_size):
    q, k, v = random_qkv(shape, device=device)
    out = financial_attention(q, k, v, block_size=block_size, **FIN_SMALL)
    assert_close(out, reference_attention(q, k, v, financial_pairs(shape[1], **FIN_SMALL)), atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fn, pairs_fn", [
    (lambda q, k, v: blocked_attention.block_sparse_attention(q, k, v, block_size=7, block=16),
     lambda S: block_sparse_pairs(S, 7)),
    (lambda q, k, v: blocked_attention.block_sparse_attention(q, k, v, block_size=50, block=64),
     lambda S: block_sparse_pairs(S, 50)),
    (lambda q, k, v: blocked_attention.longformer_attention(q, k, v, window_size=3, num_global_tokens=2, block=16),
     lambda S: longformer_pairs(S, 3, 2)),
    (lambda q, k, v: blocked_attention.longformer_attention(q, k, v, window_size=5, num_global_tokens=40, block=16),
     lambda S: longformer_pairs(S, 5, 40)),
])
def test_block_sparse_and_longformer_match_reference(device, shape, fn, pairs_fn):
    q, k, v = random_qkv(shape, device=device)
    assert_close(fn(q, k, v), reference_attention(q, k, v, pairs_fn(shape[1])), atol=1e-4)


@pytest.mark.parametrize("fn, pairs_fn", [
    (lambda q, k, v: blocked_attention.block_sparse_attention(q, k, v, block_size=4, block=5),
     lambda S: block_sparse_pairs(S, 4)),
    (lambda q, k, v: blocked_attention.longformer_attention(q, k, v, window_size=2, num_global_tokens=3, block=5),
     lambda S: longformer_pairs(S, 2, 3)),
    (lambda q, k, v: sliding_window_attention(q, k, v, 3, block_size=5), lambda S: window_pairs(S, 3)),
    (lambda q, k, v: sliding_window_attention(q, k, v, 3, causal=True, block_size=5),
     lambda S: window_pairs(S, 3, causal=True)),
    (lambda q, k, v: financial_attention(q, k, v, block_size=5, **FIN_SMALL),
     lambda S: financial_pairs(S, **FIN_SMALL)),
])
def test_vectorized_gradients_match_reference(fn, pairs_fn):
    shape = (2, 23, 2, 8)
    q, k, v = (t.requires_grad_() for t in random_qkv(shape, dtype=torch.float64))
    grad_out = torch.randn(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
    actual = torch.autograd.grad(fn(q, k, v), (q, k, v), grad_out)

    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    expected = torch.autograd.grad(reference_attention(rq, rk, rv, pairs_fn(shape[1])), (rq, rk, rv), grad_out)
    for a, e in zip(actual, expected):
        assert_close(a, e, atol=1e-10)


def test_vectorized_gradcheck():
    q, k, v = (t.requires_grad_() for t in random_qkv((1, 9, 2, 4), dtype=torch.float64))
    assert torch.autograd.gradcheck(
        lambda q, k, v: sliding_window_attention(q, k, v, 2, block_size=4), (q, k, v))


# --- PyTorch bridge ---------------------------------------------------------------

@needs_ma_core
@pytest.mark.parametrize("sparse, causal", [(True, False), (False, False), (False, True)])
def test_bridge_backward_matches_reference(sparse, causal):
    """Gradients through MACoreAttentionFunction (eval mode) match the reference."""
    shape = (2, 20, 2, 8)
    q, k, v = (t.requires_grad_() for t in random_qkv(shape))
    attention = MACoreAttention(sparse=sparse, window_size=4, use_causal_mask=causal).eval()
    grad_out = torch.randn(shape, generator=torch.Generator().manual_seed(1))
    actual = torch.autograd.grad(attention(q, k, v), (q, k, v), grad_out)

    pairs = window_pairs(shape[1], 4) if sparse else full_pairs(shape[1], causal)
    rq, rk, rv = (t.detach().clone().requires_grad_() for t in (q, k, v))
    expected = torch.autograd.grad(reference_attention(rq, rk, rv, pairs), (rq, rk, rv), grad_out.double())
    for a, e in zip(actual, expected):
        assert_close(a, e, atol=1e-4)


@needs_ma_core
@pytest.mark.parametrize("sparse", [False, True])
def test_bridge_when_seq_len_equals_num_heads(sparse):
    """Regression: equal seq_len and num_heads used to trigger a silent transpose."""
    shape = (1, 8, 8, 16)
    q, k, v = random_qkv(shape)
    out = MACoreAttention(sparse=sparse, window_size=2).eval()(q, k, v)
    pairs = window_pairs(8, 2) if sparse else full_pairs(8)
    assert_close(out, reference_attention(q, k, v, pairs))


@needs_ma_core
def test_bridge_rejects_mismatched_shapes():
    q = torch.randn(1, 4, 2, 8)
    v = torch.randn(1, 2, 4, 8)
    with pytest.raises(ValueError):
        MACoreAttention(sparse=False).eval()(q, q, v)


@needs_ma_core
@pytest.mark.parametrize("training", [False, True])
def test_sparse_attention_backends_agree(training):
    """The ma_core path and the PyTorch fallback use the same window definition."""
    x = torch.randn(2, 40, 16, generator=torch.Generator().manual_seed(0))
    with_core = SparseAttention(window_size=3).train(training)
    fallback = SparseAttention(window_size=3, use_ma_core=False).train(training)
    expected = reference_attention(x[:, :, None], x[:, :, None], x[:, :, None], window_pairs(40, 3))[:, :, 0]
    assert_close(with_core(x, x, x), expected)
    assert_close(fallback(x, x, x), expected)


# --- Metal (MPS) kernels -------------------------------------------------------

from layers import attention_backends, mps_attention  # noqa: E402

needs_mps_kernels = pytest.mark.skipif(not mps_attention.is_available(), reason="MPS kernels unavailable")

MPS_CASES = [
    ("window", lambda S: window_pairs(S, 5), lambda: mps_attention.window_segments(5)),
    ("causal", lambda S: window_pairs(S, 7, causal=True), lambda: mps_attention.window_segments(7, causal=True)),
    ("financial", lambda S: financial_pairs(S, **FIN_SMALL), lambda: mps_attention.financial_segments(**FIN_SMALL)),
    # Strips are 8 x 32 and start at multiples of 8 and 32, so the boundaries
    # of the empty-strip test need w = 1 (mod 8) and of the full-strip test
    # need w = 6 (mod 8) with w >= 18; these windows reach both.
    ("window-w1", lambda S: window_pairs(S, 1), lambda: mps_attention.window_segments(1)),
    ("window-w9", lambda S: window_pairs(S, 9), lambda: mps_attention.window_segments(9)),
    ("window-w22", lambda S: window_pairs(S, 22), lambda: mps_attention.window_segments(22)),
]

# Patterns only the tiled kernels implement: (name, reference pairs, kernel call)
MPS_TILED_ONLY = [
    ("block7", lambda S: block_sparse_pairs(S, 7), lambda q, k, v: mps_attention.block_sparse_attention(q, k, v, 7)),
    ("block50", lambda S: block_sparse_pairs(S, 50), lambda q, k, v: mps_attention.block_sparse_attention(q, k, v, 50)),
    ("longformer", lambda S: longformer_pairs(S, 3, 2),
     lambda q, k, v: mps_attention.longformer_attention(q, k, v, 3, 2)),
    ("longformer-g40", lambda S: longformer_pairs(S, 5, 40),
     lambda q, k, v: mps_attention.longformer_attention(q, k, v, 5, 40)),
    ("longformer-g0", lambda S: longformer_pairs(S, 4, 0),
     lambda q, k, v: mps_attention.longformer_attention(q, k, v, 4, 0)),
    # Boundary cases for the empty/full strip tests (see MPS_CASES)
    ("block8", lambda S: block_sparse_pairs(S, 8), lambda q, k, v: mps_attention.block_sparse_attention(q, k, v, 8)),
    ("block32", lambda S: block_sparse_pairs(S, 32), lambda q, k, v: mps_attention.block_sparse_attention(q, k, v, 32)),
    ("longformer-w1", lambda S: longformer_pairs(S, 1, 2),
     lambda q, k, v: mps_attention.longformer_attention(q, k, v, 1, 2)),
    ("longformer-w9", lambda S: longformer_pairs(S, 9, 2),
     lambda q, k, v: mps_attention.longformer_attention(q, k, v, 9, 2)),
    ("longformer-w22", lambda S: longformer_pairs(S, 22, 2),
     lambda q, k, v: mps_attention.longformer_attention(q, k, v, 22, 2)),
]
# (shape, kernels to test); seq lengths include ones that are not multiples of the 32-row tile
MPS_SHAPES = [
    ((1, 1, 1, 8), ("tiled", "row")),
    ((2, 17, 3, 16), ("tiled", "row")),
    ((2, 300, 2, 64), ("tiled", "row")),
    ((1, 70, 2, 128), ("tiled", "row")),
    ((1, 45, 2, 20), ("row",)),
    ((1, 33, 1, 256), ("row",)),
]


def mps_params():
    for shape, kernels in MPS_SHAPES:
        for kernel in kernels:
            for name, pairs_fn, segments_fn in MPS_CASES:
                yield pytest.param(shape, kernel, pairs_fn, segments_fn, id=f"{name}-{kernel}-{'x'.join(map(str, shape))}")


@needs_mps_kernels
@pytest.mark.parametrize("shape, kernel, pairs_fn, segments_fn", list(mps_params()))
def test_mps_kernels_match_reference(shape, kernel, pairs_fn, segments_fn):
    q, k, v = (t.requires_grad_() for t in random_qkv(shape, device="mps"))
    grad_out = torch.randn(shape, generator=torch.Generator().manual_seed(1))
    out = mps_attention.segment_attention(q, k, v, segments_fn(), kernel=kernel)
    grads = torch.autograd.grad(out, (q, k, v), grad_out.to("mps"))

    rq, rk, rv = (t.detach().cpu().double().requires_grad_() for t in (q, k, v))
    expected = reference_attention(rq, rk, rv, pairs_fn(shape[1]))
    expected_grads = torch.autograd.grad(expected, (rq, rk, rv), grad_out.double())
    assert out.device.type == "mps"
    assert_close(out, expected.detach(), atol=1e-4)
    for actual, wanted in zip(grads, expected_grads):
        assert_close(actual, wanted, atol=1e-4)


def tiled_only_params():
    for shape, kernels in MPS_SHAPES:
        if "tiled" in kernels:
            for name, pairs_fn, call in MPS_TILED_ONLY:
                yield pytest.param(shape, pairs_fn, call, id=f"{name}-{'x'.join(map(str, shape))}")


@needs_mps_kernels
@pytest.mark.parametrize("shape, pairs_fn, call", list(tiled_only_params()))
def test_mps_block_sparse_and_longformer_match_reference(shape, pairs_fn, call):
    q, k, v = (t.requires_grad_() for t in random_qkv(shape, device="mps"))
    grad_out = torch.randn(shape, generator=torch.Generator().manual_seed(1))
    out = call(q, k, v)
    grads = torch.autograd.grad(out, (q, k, v), grad_out.to("mps"))

    rq, rk, rv = (t.detach().cpu().double().requires_grad_() for t in (q, k, v))
    expected = reference_attention(rq, rk, rv, pairs_fn(shape[1]))
    expected_grads = torch.autograd.grad(expected, (rq, rk, rv), grad_out.double())
    assert_close(out, expected.detach(), atol=1e-4)
    for actual, wanted in zip(grads, expected_grads):
        assert_close(actual, wanted, atol=1e-4)


LOW_PRECISION = [(torch.float16, 3e-3), (torch.bfloat16, 2e-2)]

LOW_PRECISION_CALLS = [
    ("window-tiled", lambda S: window_pairs(S, 5),
     lambda q, k, v: mps_attention.sliding_window_attention(q, k, v, 5, kernel="tiled")),
    ("window-row", lambda S: window_pairs(S, 5),
     lambda q, k, v: mps_attention.sliding_window_attention(q, k, v, 5, kernel="row")),
    ("financial", lambda S: financial_pairs(S, **FIN_SMALL),
     lambda q, k, v: mps_attention.financial_attention(q, k, v, **FIN_SMALL)),
    ("block", lambda S: block_sparse_pairs(S, 16),
     lambda q, k, v: mps_attention.block_sparse_attention(q, k, v, 16)),
    ("longformer", lambda S: longformer_pairs(S, 4, 3),
     lambda q, k, v: mps_attention.longformer_attention(q, k, v, 4, 3)),
]


@needs_mps_kernels
@pytest.mark.parametrize("dtype, tol", LOW_PRECISION, ids=["fp16", "bf16"])
@pytest.mark.parametrize("name, pairs_fn, call", LOW_PRECISION_CALLS, ids=[c[0] for c in LOW_PRECISION_CALLS])
def test_mps_kernels_native_low_precision(dtype, tol, name, pairs_fn, call):
    """fp16/bf16 run natively: outputs and gradients keep the dtype and match a
    reference computed from the same rounded inputs."""
    shape = (2, 70, 2, 32)
    q, k, v = (t.to(dtype).requires_grad_() for t in random_qkv(shape, device="mps"))
    grad_out = torch.randn(shape, generator=torch.Generator().manual_seed(1)).to(dtype)
    out = call(q, k, v)
    grads = torch.autograd.grad(out, (q, k, v), grad_out.to("mps"))
    assert out.dtype == dtype and all(g.dtype == dtype for g in grads)

    rq, rk, rv = (t.detach().cpu().double().requires_grad_() for t in (q, k, v))
    expected = reference_attention(rq, rk, rv, pairs_fn(shape[1]))
    expected_grads = torch.autograd.grad(expected, (rq, rk, rv), grad_out.double())
    for actual, wanted in [(out, expected.detach())] + list(zip(grads, expected_grads)):
        scale = wanted.abs().max().item()
        assert_close(actual, wanted, atol=tol * max(scale, 1.0))


@needs_mps_kernels
@pytest.mark.parametrize("kernel", ["tiled", "row"])
def test_mps_financial_default_config(kernel):
    shape = (1, 2100, 2, 64)
    q, k, v = random_qkv(shape, device="mps")
    out = mps_attention.financial_attention(q, k, v, kernel=kernel)
    defaults = dict(local_window_size=512, dilation_stride=1000, dilation_cluster_size=8, dilation_num_clusters=10)
    assert_close(out, reference_attention(q, k, v, financial_pairs(shape[1], **defaults)), atol=1e-4)


@needs_mps_kernels
def test_mps_kernels_reject_bad_input():
    q = torch.randn(1, 8, 1, 16, device="mps")
    with pytest.raises(ValueError):
        mps_attention.segment_attention(q.cpu(), q.cpu(), q.cpu(), [(-1, 1)])
    with pytest.raises(ValueError):
        mps_attention.segment_attention(q, q, q, [(2, 1)])
    with pytest.raises(ValueError):
        mps_attention.segment_attention(q[..., :12], q[..., :12], q[..., :12], [(-1, 1)], kernel="tiled")
    with pytest.raises(ValueError):
        mps_attention.block_sparse_attention(q[..., :12], q[..., :12], q[..., :12], 4)
    with pytest.raises(ValueError):
        mps_attention.block_sparse_attention(q, q, q, 0)
    with pytest.raises(ValueError):
        mps_attention.longformer_attention(q, q, q, -1, 2)


@needs_mps_kernels
def test_layers_dispatch_to_mps_kernels(monkeypatch):
    calls = []
    original = mps_attention.segment_attention
    monkeypatch.setattr(mps_attention, "segment_attention",
                        lambda *args, **kwargs: calls.append(1) or original(*args, **kwargs))
    x = torch.randn(2, 40, 16, device="mps", requires_grad=True)
    expected = reference_attention(x[:, :, None], x[:, :, None], x[:, :, None], window_pairs(40, 3))[:, :, 0]

    for use_ma_core in ([True, False] if HAS_MA_CORE else [False]):
        for training in (True, False):
            layer = SparseAttention(window_size=3, use_ma_core=use_ma_core).train(training)
            out = layer(x, x, x)
            out.sum().backward()
            assert out.device.type == "mps"
            assert_close(out, expected.detach(), atol=1e-4)
    assert calls, "SparseAttention on MPS did not use the Metal kernels"


@needs_mps_kernels
def test_mps_kernels_can_be_disabled(monkeypatch):
    monkeypatch.setenv("MA_DISABLE_MPS_KERNELS", "1")
    q, k, v = random_qkv((1, 30, 2, 16), device="mps")
    assert not mps_attention.supports(q)
    out = attention_backends.sliding_window_attention(q, k, v, 4)
    assert_close(out, reference_attention(q, k, v, window_pairs(30, 4)), atol=1e-4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("head_dim", [16, 20])
def test_backends_block_sparse_and_longformer(device, head_dim):
    """Dispatch picks Metal when it can (head_dim 16 on MPS) and PyTorch otherwise."""
    shape = (1, 50, 2, head_dim)
    q, k, v = random_qkv(shape, device=device)
    assert_close(attention_backends.block_sparse_attention(q, k, v, 8),
                 reference_attention(q, k, v, block_sparse_pairs(50, 8)), atol=1e-4)
    assert_close(attention_backends.longformer_attention(q, k, v, 3, 2),
                 reference_attention(q, k, v, longformer_pairs(50, 3, 2)), atol=1e-4)
