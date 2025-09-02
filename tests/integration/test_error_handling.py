 # SPDX-License-Identifier: Apache-2.0
"""
Error handling and edge case tests for ma_core PyTorch integration.
Tests robustness and proper error reporting.
"""

import pytest
import torch
import numpy as np
from layers.ma_core_bridge import MACoreAttention, create_attention_layer
from layers.sparse_attention import SparseAttention


class TestInputValidation:
    """Test input validation and error handling."""
    
    @pytest.mark.error
    def test_mismatched_tensor_shapes(self, skip_if_ma_core_unavailable):
        """Test error handling for mismatched tensor shapes."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        attention.train()  # Use PyTorch for clearer error messages
        
        # Create mismatched tensors
        query = torch.randn(1, 8, 2, 16)
        key_seq_mismatch = torch.randn(1, 6, 2, 16)  # Different sequence length
        value = torch.randn(1, 8, 2, 16)
        
        with pytest.raises(Exception) as exc_info:
            attention(query, key_seq_mismatch, value)
        
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ['sequence', 'length', 'mismatch', 'size'])
    
    @pytest.mark.error
    def test_mismatched_head_dimensions(self, skip_if_ma_core_unavailable):
        """Test error handling for mismatched head dimensions."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        attention.train()
        
        query = torch.randn(1, 8, 2, 16)
        key = torch.randn(1, 8, 2, 16)
        value_head_mismatch = torch.randn(1, 8, 2, 32)  # Different head dimension
        
        with pytest.raises(Exception) as exc_info:
            attention(query, key, value_head_mismatch)
        
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ['dimension', 'head', 'mismatch', 'size'])
    
    @pytest.mark.error
    def test_mismatched_batch_sizes(self, skip_if_ma_core_unavailable):
        """Test error handling for mismatched batch sizes."""
        attention = MACoreAttention(sparse=True, window_size=4, fallback_training=True)
        attention.train()
        
        query = torch.randn(1, 8, 2, 16)
        key = torch.randn(2, 8, 2, 16)  # Different batch size
        value = torch.randn(1, 8, 2, 16)
        
        with pytest.raises(Exception):
            attention(query, key, value)
    
    @pytest.mark.error
    def test_zero_dimensions(self, skip_if_ma_core_unavailable):
        """Test handling of tensors with zero dimensions."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        
        # Zero batch size
        query_zero_batch = torch.randn(0, 8, 2, 16)
        key_zero_batch = torch.randn(0, 8, 2, 16)
        value_zero_batch = torch.randn(0, 8, 2, 16)
        
        with pytest.raises(Exception):
            attention(query_zero_batch, key_zero_batch, value_zero_batch)
        
        # Zero sequence length
        query_zero_seq = torch.randn(1, 0, 2, 16)
        key_zero_seq = torch.randn(1, 0, 2, 16)
        value_zero_seq = torch.randn(1, 0, 2, 16)
        
        with pytest.raises(Exception):
            attention(query_zero_seq, key_zero_seq, value_zero_seq)
    
    @pytest.mark.error
    def test_invalid_window_sizes(self, skip_if_ma_core_unavailable):
        """Test handling of invalid window sizes."""
        # Zero window size
        with pytest.raises(Exception):
            MACoreAttention(sparse=True, window_size=0, fallback_training=True)
        
        # Negative window size  
        with pytest.raises(Exception):
            MACoreAttention(sparse=True, window_size=-1, fallback_training=True)


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_extreme_values(self, skip_if_ma_core_unavailable):
        """Test handling of extreme tensor values."""
        attention = MACoreAttention(sparse=True, window_size=4, fallback_training=True)
        
        batch_size, seq_len, num_heads, head_dim = 1, 8, 2, 16
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        # Very large values
        query_large = torch.randn(*shape) * 100
        key_large = torch.randn(*shape) * 100
        value_large = torch.randn(*shape) * 100
        
        output_large = attention(query_large, key_large, value_large)
        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()
        
        # Very small values
        query_small = torch.randn(*shape) * 1e-6
        key_small = torch.randn(*shape) * 1e-6
        value_small = torch.randn(*shape) * 1e-6
        
        output_small = attention(query_small, key_small, value_small)
        assert not torch.isnan(output_small).any()
        assert not torch.isinf(output_small).any()
    
    def test_nan_input_handling(self, skip_if_ma_core_unavailable):
        """Test handling of NaN inputs."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        
        batch_size, seq_len, num_heads, head_dim = 1, 4, 1, 8
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        query = torch.randn(*shape)
        key = torch.randn(*shape)
        value = torch.randn(*shape)
        
        # Insert NaN values
        query[0, 0, 0, 0] = float('nan')
        
        try:
            output = attention(query, key, value)
            # If it doesn't raise an error, output should indicate the problem
            # (either NaN propagation or error handling)
        except Exception:
            # It's acceptable to raise an error for NaN inputs
            pass
    
    def test_inf_input_handling(self, skip_if_ma_core_unavailable):
        """Test handling of infinite inputs."""
        attention = MACoreAttention(sparse=True, window_size=2, fallback_training=True)
        
        batch_size, seq_len, num_heads, head_dim = 1, 4, 1, 8
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        query = torch.randn(*shape)
        key = torch.randn(*shape)
        value = torch.randn(*shape)
        
        # Insert infinite values
        key[0, 0, 0, 0] = float('inf')
        
        try:
            output = attention(query, key, value)
            # Should handle inf gracefully or raise appropriate error
        except Exception:
            # It's acceptable to raise an error for inf inputs
            pass
    
    def test_gradient_numerical_stability(self, skip_if_ma_core_unavailable):
        """Test numerical stability of gradients."""
        attention = MACoreAttention(sparse=True, window_size=3, fallback_training=True)
        attention.train()
        
        batch_size, seq_len, num_heads, head_dim = 1, 6, 1, 8
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        # Use values that might cause numerical issues
        query = torch.randn(*shape, requires_grad=True) * 10
        key = torch.randn(*shape, requires_grad=True) * 10
        value = torch.randn(*shape, requires_grad=True) * 0.1
        
        output = attention(query, key, value)
        loss = output.sum()
        loss.backward()
        
        # Check gradients are finite
        assert torch.isfinite(query.grad).all()
        assert torch.isfinite(key.grad).all()
        assert torch.isfinite(value.grad).all()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_single_element_tensors(self, skip_if_ma_core_unavailable):
        """Test with minimal tensor dimensions."""
        attention = MACoreAttention(sparse=True, window_size=1, fallback_training=True)
        
        # Minimal dimensions: 1 batch, 1 sequence, 1 head, 1 dimension
        query = torch.randn(1, 1, 1, 1)
        key = torch.randn(1, 1, 1, 1)
        value = torch.randn(1, 1, 1, 1)
        
        output = attention(query, key, value)
        
        assert output.shape == (1, 1, 1, 1)
        assert not torch.isnan(output).any()
    
    def test_very_long_sequences(self, skip_if_ma_core_unavailable):
        """Test with very long sequences (sparse attention should handle well)."""
        attention = MACoreAttention(sparse=True, window_size=8, fallback_training=True)
        
        batch_size, seq_len, num_heads, head_dim = 1, 1024, 2, 16
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        query = torch.randn(*shape)
        key = torch.randn(*shape)
        value = torch.randn(*shape)
        
        # This should complete without memory issues (sparse attention)
        output = attention(query, key, value)
        
        assert output.shape == shape
        assert not torch.isnan(output).any()
    
    def test_window_size_larger_than_sequence(self, skip_if_ma_core_unavailable):
        """Test sparse attention when window size exceeds sequence length."""
        seq_len = 4
        window_size = 10  # Larger than sequence
        
        attention = MACoreAttention(sparse=True, window_size=window_size, fallback_training=True)
        
        query = torch.randn(1, seq_len, 1, 8)
        key = torch.randn(1, seq_len, 1, 8)
        value = torch.randn(1, seq_len, 1, 8)
        
        output = attention(query, key, value)
        
        assert output.shape == (1, seq_len, 1, 8)
        assert not torch.isnan(output).any()
    
    def test_identical_qkv_tensors(self, skip_if_ma_core_unavailable):
        """Test attention when Q, K, V are identical."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        
        tensor = torch.randn(1, 4, 2, 8)
        
        output = attention(tensor, tensor, tensor)
        
        assert output.shape == tensor.shape
        assert not torch.isnan(output).any()
    
    def test_orthogonal_qk_tensors(self, skip_if_ma_core_unavailable):
        """Test attention with orthogonal Q and K tensors."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        
        batch_size, seq_len, num_heads, head_dim = 1, 4, 1, 4
        
        # Create orthogonal Q and K
        query = torch.tensor([[[[1., 0., 0., 0.],
                               [0., 1., 0., 0.],
                               [0., 0., 1., 0.],
                               [0., 0., 0., 1.]]]], dtype=torch.float32)
        
        key = torch.tensor([[[[0., 1., 0., 0.],
                             [1., 0., 0., 0.],
                             [0., 0., 0., 1.],
                             [0., 0., 1., 0.]]]], dtype=torch.float32)
        
        value = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        output = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        assert not torch.isnan(output).any()


class TestSparseAttentionErrorHandling:
    """Test error handling specific to SparseAttention class."""
    
    def test_sparse_attention_invalid_construction(self):
        """Test invalid SparseAttention construction parameters."""
        # Negative window size
        with pytest.raises(Exception):
            SparseAttention(window_size=-1)
        
        # Zero window size
        with pytest.raises(Exception):
            SparseAttention(window_size=0)
    
    def test_sparse_attention_mismatched_inputs(self, skip_if_ma_core_unavailable):
        """Test SparseAttention with mismatched inputs."""
        sparse_attn = SparseAttention(window_size=2, use_ma_core=True)
        sparse_attn.train()
        
        q = torch.randn(1, 8, 16)
        k = torch.randn(1, 6, 16)  # Different sequence length
        v = torch.randn(1, 8, 16)
        
        with pytest.raises(Exception):
            sparse_attn(q, k, v)
    
    def test_sparse_attention_wrong_dimensions(self, skip_if_ma_core_unavailable):
        """Test SparseAttention with wrong tensor dimensions."""
        sparse_attn = SparseAttention(window_size=3, use_ma_core=True)
        
        # Wrong number of dimensions
        q_wrong = torch.randn(8, 16)  # Missing batch dimension
        k = torch.randn(1, 8, 16)
        v = torch.randn(1, 8, 16)
        
        with pytest.raises(Exception):
            sparse_attn(q_wrong, k, v)


class TestFactoryFunctionErrors:
    """Test error handling in factory functions."""
    
    def test_invalid_attention_type(self, skip_if_ma_core_unavailable):
        """Test create_attention_layer with invalid type."""
        with pytest.raises(ValueError) as exc_info:
            create_attention_layer("nonexistent_type")
        
        assert "unknown attention type" in str(exc_info.value).lower()
    
    def test_invalid_factory_parameters(self, skip_if_ma_core_unavailable):
        """Test create_attention_layer with invalid parameters.""" 
        # This should work (valid parameters)
        attention = create_attention_layer("sparse", window_size=8)
        assert attention.window_size == 8
        
        # Invalid parameters should be caught by the underlying class
        with pytest.raises(Exception):
            create_attention_layer("sparse", window_size=-1)


class TestRecoveryAndRobustness:
    """Test recovery from errors and robustness."""
    
    def test_error_recovery(self, skip_if_ma_core_unavailable):
        """Test that valid operations work after errors."""
        attention = MACoreAttention(sparse=True, window_size=4, fallback_training=True)
        
        # First, cause an error
        try:
            query_bad = torch.randn(1, 4, 2, 8)
            key_bad = torch.randn(1, 6, 2, 8)  # Mismatched
            value_bad = torch.randn(1, 4, 2, 8)
            attention(query_bad, key_bad, value_bad)
        except Exception:
            pass  # Expected to fail
        
        # Then, use valid inputs - should work fine
        query_good = torch.randn(1, 4, 2, 8)
        key_good = torch.randn(1, 4, 2, 8)
        value_good = torch.randn(1, 4, 2, 8)
        
        output = attention(query_good, key_good, value_good)
        
        assert output.shape == (1, 4, 2, 8)
        assert not torch.isnan(output).any()
    
    def test_module_state_after_error(self, skip_if_ma_core_unavailable):
        """Test that module state remains consistent after errors."""
        attention = MACoreAttention(sparse=True, window_size=4, fallback_training=True)
        
        # Store initial state
        initial_sparse = attention.sparse
        initial_window_size = attention.window_size
        
        # Try to cause error
        try:
            bad_tensor = torch.randn(0, 4, 2, 8)  # Zero batch size
            attention(bad_tensor, bad_tensor, bad_tensor)
        except Exception:
            pass  # Expected
        
        # Check that module state is unchanged
        assert attention.sparse == initial_sparse
        assert attention.window_size == initial_window_size
        
        # Module should still work with valid inputs
        good_input = torch.randn(1, 4, 2, 8)
        output = attention(good_input, good_input, good_input)
        assert not torch.isnan(output).any()


class TestPerformanceUnderError:
    """Test performance characteristics under error conditions."""
    
    def test_error_performance_impact(self, skip_if_ma_core_unavailable):
        """Test that error handling doesn't significantly impact performance."""
        import time
        
        attention = MACoreAttention(sparse=True, window_size=4, fallback_training=True)
        
        valid_input = torch.randn(1, 8, 2, 16)
        
        # Benchmark normal operation
        iterations = 10
        start_time = time.time()
        
        for _ in range(iterations):
            output = attention(valid_input, valid_input, valid_input)
        
        normal_time = (time.time() - start_time) / iterations * 1000
        
        # Try some operations that might trigger error checking
        invalid_inputs = [
            (torch.randn(1, 6, 2, 16), valid_input, valid_input),  # seq mismatch
            (valid_input, torch.randn(1, 8, 2, 32), valid_input),  # dim mismatch
        ]
        
        error_count = 0
        for bad_q, bad_k, bad_v in invalid_inputs:
            try:
                attention(bad_q, bad_k, bad_v)
            except Exception:
                error_count += 1
        
        # Now benchmark again after error conditions
        start_time = time.time()
        
        for _ in range(iterations):
            output = attention(valid_input, valid_input, valid_input)
        
        after_error_time = (time.time() - start_time) / iterations * 1000
        
        # Performance should not be significantly impacted
        performance_ratio = after_error_time / normal_time
        assert performance_ratio < 2.0, f"Performance degraded {performance_ratio:.2f}x after errors"
        assert error_count > 0, "Error conditions should have been triggered"
