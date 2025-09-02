 # SPDX-License-Identifier: Apache-2.0
import unittest
import torch
import sys
import os

# Ensure 'src' is on sys.path for bridge imports when not installed as a package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))


class TestCPUAttentionCPP(unittest.TestCase):
    def setUp(self):
        import ma_core  # Ensure extension imports
        self.ma_core = ma_core
        from layers.ma_core_bridge import (
            pytorch_dense_attention, pytorch_sparse_attention,
            pytorch_to_ma_core, ma_core_to_pytorch,
        )
        self.pytorch_dense_attention = pytorch_dense_attention
        self.pytorch_sparse_attention = pytorch_sparse_attention
        self.pytorch_to_ma_core = pytorch_to_ma_core
        self.ma_core_to_pytorch = ma_core_to_pytorch

    def test_dense_cpp_matches_pytorch_small(self):
        torch.manual_seed(0)
        B, S, H, D = 1, 6, 2, 4
        q = torch.randn(B, S, H, D, dtype=torch.float32)
        k = torch.randn(B, S, H, D, dtype=torch.float32)
        v = torch.randn(B, S, H, D, dtype=torch.float32)

        # Convert to ma_core tensors and compute in C++
        mq = self.pytorch_to_ma_core(q)
        mk = self.pytorch_to_ma_core(k)
        mv = self.pytorch_to_ma_core(v)
        mout = self.ma_core.compute_dense_attention(mq, mk, mv, False)
        y_cpp = self.ma_core_to_pytorch(mout, torch.device('cpu'), torch.float32)

        # Reference PyTorch
        y_ref = self.pytorch_dense_attention(q, k, v, use_causal_mask=False)
        torch.testing.assert_close(y_cpp, y_ref, rtol=1e-5, atol=1e-6)

    def test_sparse_cpp_matches_pytorch_small(self):
        torch.manual_seed(1)
        B, S, H, D, W = 1, 10, 2, 4, 2
        q = torch.randn(B, S, H, D, dtype=torch.float32)
        k = torch.randn(B, S, H, D, dtype=torch.float32)
        v = torch.randn(B, S, H, D, dtype=torch.float32)

        mq = self.pytorch_to_ma_core(q)
        mk = self.pytorch_to_ma_core(k)
        mv = self.pytorch_to_ma_core(v)
        mout = self.ma_core.compute_sparse_attention(mq, mk, mv, W)
        y_cpp = self.ma_core_to_pytorch(mout, torch.device('cpu'), torch.float32)

        y_ref = self.pytorch_sparse_attention(q, k, v, window_size=W)
        torch.testing.assert_close(y_cpp, y_ref, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
