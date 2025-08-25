import os
import unittest
import torch


def pytorch_dense_attention(q, k, v, causal=False):
    B, S, H, D = q.shape
    scale = 1.0 / (D ** 0.5)
    q = q * scale
    qh = q.transpose(1, 2).contiguous().view(B * H, S, D)
    kh = k.transpose(1, 2).contiguous().view(B * H, S, D)
    vh = v.transpose(1, 2).contiguous().view(B * H, S, D)
    scores = torch.matmul(qh, kh.transpose(-2, -1))
    if causal:
        mask = torch.tril(torch.ones(S, S, device=scores.device, dtype=torch.bool))
        scores = scores.masked_fill(~mask, -1e9)
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, vh)
    return out.view(B, H, S, D).transpose(1, 2).contiguous()


def pytorch_sparse_attention(q, k, v, window):
    B, S, H, D = q.shape
    scale = 1.0 / (D ** 0.5)
    out = torch.zeros_like(v)
    for b in range(B):
        for h in range(H):
            for i in range(S):
                start_j = max(0, i - window)
                end_j = min(S, i + window + 1)
                qi = q[b, i, h] * scale
                kw = k[b, start_j:end_j, h]
                vw = v[b, start_j:end_j, h]
                scores = torch.matmul(kw, qi)
                weights = torch.softmax(scores, dim=0)
                out[b, i, h] = torch.matmul(weights, vw)
    return out


class TestCudaAttentionBackward(unittest.TestCase):
    def setUp(self):
        try:
            import sparse_attention_cuda  # noqa: F401
            self.has_ext = True
        except Exception:
            self.has_ext = False

    def test_dense_backward_cuda_matches_pytorch(self):
        if not torch.cuda.is_available() or not self.has_ext:
            self.skipTest("CUDA or extension not available")
        torch.manual_seed(0)
        B, S, H, D = 1, 8, 2, 4
        q = torch.randn(B, S, H, D, device='cuda', dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, S, H, D, device='cuda', dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, S, H, D, device='cuda', dtype=torch.float32, requires_grad=True)
        # Our CUDA kernels via extension
        import sparse_attention_cuda as ext
        out = ext.dense_forward(q, k, v, False)
        loss = out.sum()
        dO = torch.autograd.grad(loss, out, retain_graph=True)[0]
        dQ, dK, dV = ext.dense_backward(q, k, v, dO.contiguous(), False)

        # Reference PyTorch grads
        q2 = q.clone().detach().requires_grad_(True)
        k2 = k.clone().detach().requires_grad_(True)
        v2 = v.clone().detach().requires_grad_(True)
        out_ref = pytorch_dense_attention(q2, k2, v2, causal=False)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        self.assertTrue(torch.allclose(dQ, q2.grad, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(dK, k2.grad, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(dV, v2.grad, atol=1e-3, rtol=1e-3))

    def test_sparse_backward_cuda_matches_pytorch(self):
        if not torch.cuda.is_available() or not self.has_ext:
            self.skipTest("CUDA or extension not available")
        torch.manual_seed(0)
        B, S, H, D = 1, 10, 2, 4
        window = 2
        q = torch.randn(B, S, H, D, device='cuda', dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, S, H, D, device='cuda', dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, S, H, D, device='cuda', dtype=torch.float32, requires_grad=True)
        # Our CUDA kernels via extension
        import sparse_attention_cuda as ext
        out = ext.forward(q, k, v, window)
        loss = out.sum()
        dO = torch.autograd.grad(loss, out, retain_graph=True)[0]
        dQ, dK, dV = ext.backward(q, k, v, dO.contiguous(), window)

        # Reference PyTorch grads
        q2 = q.clone().detach().requires_grad_(True)
        k2 = k.clone().detach().requires_grad_(True)
        v2 = v.clone().detach().requires_grad_(True)
        out_ref = pytorch_sparse_attention(q2, k2, v2, window)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        self.assertTrue(torch.allclose(dQ, q2.grad, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(dK, k2.grad, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(dV, v2.grad, atol=1e-3, rtol=1e-3))


if __name__ == '__main__':
    unittest.main()

