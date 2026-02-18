# SPDX-License-Identifier: Apache-2.0
import unittest
import torch
import sys
import os

# Ensure 'src' is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

try:
    import ma_core
    from layers.ma_core_bridge import pytorch_to_ma_core, ma_core_to_pytorch
except ImportError:
    ma_core = None

@unittest.skipIf(ma_core is None, "ma_core extension not available")
class TestFinancialAttention(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.dtype = torch.float32

    def test_financial_config_and_compute(self):
        """Test that FinancialAttention can be configured and executed."""
        B, S, H, D = 1, 200, 2, 8
        
        # Configure financial attention
        config = ma_core.AttentionConfig(ma_core.AttentionPattern.FINANCIAL)
        config.local_window_size = 32
        config.dilation_stride = 50
        config.dilation_cluster_size = 4
        config.dilation_num_clusters = 2
        
        # Create input tensors
        q = torch.randn(B, S, H, D, dtype=self.dtype)
        k = torch.randn(B, S, H, D, dtype=self.dtype)
        v = torch.randn(B, S, H, D, dtype=self.dtype)
        
        mq = pytorch_to_ma_core(q)
        mk = pytorch_to_ma_core(k)
        mv = pytorch_to_ma_core(v)
        
        # Compute attention
        mout = ma_core.compute_attention(mq, mk, mv, config)
        out = ma_core_to_pytorch(mout, self.device, self.dtype)
        
        self.assertEqual(out.shape, (B, S, H, D))
        self.assertFalse(torch.isnan(out).any())

    def test_financial_pattern_causality(self):
        """Test that the financial attention pattern is strictly causal."""
        # This test ensures the pattern stays causal by checking dependencies
        B, S, H, D = 1, 100, 1, 4
        config = ma_core.AttentionConfig(ma_core.AttentionPattern.FINANCIAL)
        
        q = torch.ones(B, S, H, D)
        k = torch.zeros(B, S, H, D)
        v = torch.zeros(B, S, H, D)
        
        # Set a value in the future
        target_j = 50
        v[0, target_j, 0, 0] = 1.0
        k[0, target_j, 0, :] = 1.0
        
        mq = pytorch_to_ma_core(q)
        mk = pytorch_to_ma_core(k)
        mv = pytorch_to_ma_core(v)
        
        mout = ma_core.compute_attention(mq, mk, mv, config)
        out = ma_core_to_pytorch(mout, self.device, self.dtype)
        
        # No position i < target_j should see target_j
        for i in range(target_j):
            self.assertEqual(out[0, i, 0, 0], 0.0, f"Causality violation: Query {i} sees future target {target_j}")

    def test_financial_logic_verification(self):
        """Verify the logic of the financial pattern with a small example."""
        B, S, H, D = 1, 50, 1, 4
        
        config = ma_core.AttentionConfig(ma_core.AttentionPattern.FINANCIAL)
        config.local_window_size = 5
        config.dilation_stride = 10
        config.dilation_cluster_size = 2
        config.dilation_num_clusters = 2
        
        # Create inputs where we can track dependencies
        q = torch.ones(B, S, H, D)
        k = torch.zeros(B, S, H, D)
        v = torch.zeros(B, S, H, D)
        
        target_j = 10
        v[0, target_j, 0, 0] = 1.0
        k[0, target_j, 0, :] = 1.0
        
        mq = pytorch_to_ma_core(q)
        mk = pytorch_to_ma_core(k)
        mv = pytorch_to_ma_core(v)
        
        mout = ma_core.compute_attention(mq, mk, mv, config)
        out = ma_core_to_pytorch(mout, self.device, self.dtype)
        
        for i in range(S):
            sees_target = (out[0, i, 0, 0] > 0)
            expected = False
            # Local window check
            if i - 5 + 1 <= target_j <= i:
                expected = True
            # Dilated clusters check
            for k_idx in [1, 2]:
                cluster_end = i - k_idx * 10
                cluster_start = cluster_end - 2 + 1
                if cluster_start <= target_j <= cluster_end:
                    expected = True
            
            self.assertEqual(sees_target, expected, f"Query {i} seeing target {target_j} mismatch. Expected {expected}, got {sees_target}")

if __name__ == '__main__':
    unittest.main()
