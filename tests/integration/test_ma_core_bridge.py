 # SPDX-License-Identifier: Apache-2.0
"""
Integration tests for ma_core PyTorch bridge functionality.
Tests the bridge between C++ ma_core engine and PyTorch tensors.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from layers.ma_core_bridge import (
    MACoreAttention, 
    MACoreAttentionFunction,
    pytorch_dense_attention,
    pytorch_sparse_attention,
    create_attention_layer
)


class TestMACoreAttentionBasic:
    """Test basic functionality of MACoreAttention module."""
    
    def test_dense_attention_forward(self, small_tensors, skip_if_ma_core_unavailable):
        """Test dense attention forward pass."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        
        query = small_tensors['query']
        key = small_tensors['key']
        value = small_tensors['value']
        expected_shape = small_tensors['shape']
        
        output = attention(query, key, value)
        
        assert output.shape == expected_shape
        assert output.dtype == query.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_sparse_attention_forward(self, small_tensors, window_sizes, skip_if_ma_core_unavailable):
        """Test sparse attention forward pass with different window sizes."""
        attention = MACoreAttention(sparse=True, window_size=window_sizes, fallback_training=True)
        
        query = small_tensors['query']
        key = small_tensors['key']
        value = small_tensors['value']
        expected_shape = small_tensors['shape']
        
        output = attention(query, key, value)
        
        assert output.shape == expected_shape
        assert output.dtype == query.dtype
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_causal_mask_attention(self, small_tensors, skip_if_ma_core_unavailable):
        """Test dense attention with causal masking."""
        attention = MACoreAttention(sparse=False, use_causal_mask=True, fallback_training=True)
        
        query = small_tensors['query']
        key = small_tensors['key']
        value = small_tensors['value']
        
        output = attention(query, key, value)
        
        assert output.shape == small_tensors['shape']
        assert not torch.isnan(output).any()
    
    def test_attention_module_modes(self, small_tensors, skip_if_ma_core_unavailable):
        """Test attention module in train vs eval modes."""
        attention = MACoreAttention(sparse=True, window_size=4, fallback_training=True)
        
        query = small_tensors['query'].clone()
        key = small_tensors['key'].clone()
        value = small_tensors['value'].clone()
        
        # Test training mode
        attention.train()
        train_output = attention(query, key, value)
        
        # Test eval mode  
        attention.eval()
        with torch.no_grad():
            eval_output = attention(query, key, value)
        
        # Outputs should have same shape (may have different values due to different implementations)
        assert train_output.shape == eval_output.shape
        assert not torch.isnan(train_output).any()
        assert not torch.isnan(eval_output).any()


class TestGradientFlow:
    """Test gradient flow through attention modules."""
    
    @pytest.mark.gradient
    def test_dense_attention_gradients(self, small_tensors, test_helper, skip_if_ma_core_unavailable):
        """Test gradient flow through dense attention."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        attention.train()  # Use PyTorch implementation for gradients
        
        query = small_tensors['query']
        key = small_tensors['key']  
        value = small_tensors['value']
        
        # Forward pass
        output = attention(query, key, value)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        test_helper.assert_gradients_exist({'query': query, 'key': key, 'value': value})
        
        # Check gradient shapes
        assert query.grad.shape == query.shape
        assert key.grad.shape == key.shape
        assert value.grad.shape == value.shape
    
    @pytest.mark.gradient
    def test_sparse_attention_gradients(self, small_tensors, test_helper, skip_if_ma_core_unavailable):
        """Test gradient flow through sparse attention."""
        attention = MACoreAttention(sparse=True, window_size=2, fallback_training=True)
        attention.train()  # Use PyTorch implementation for gradients
        
        query = small_tensors['query']
        key = small_tensors['key']
        value = small_tensors['value']
        
        # Forward pass
        output = attention(query, key, value)
        loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and have correct shape
        test_helper.assert_gradients_exist({'query': query, 'key': key, 'value': value})
        assert query.grad.shape == query.shape
        assert key.grad.shape == key.shape
        assert value.grad.shape == value.shape
    
    @pytest.mark.gradient
    def test_gradient_accumulation(self, small_tensors, skip_if_ma_core_unavailable):
        """Test gradient accumulation over multiple forward passes."""
        attention = MACoreAttention(sparse=True, window_size=2, fallback_training=True)
        attention.train()
        
        query = small_tensors['query']
        key = small_tensors['key'] 
        value = small_tensors['value']
        
        # First forward/backward pass
        output1 = attention(query, key, value)
        loss1 = output1.sum()
        loss1.backward()
        
        # Store gradients
        query_grad_1 = query.grad.clone()
        key_grad_1 = key.grad.clone()
        value_grad_1 = value.grad.clone()
        
        # Second forward/backward pass (accumulate gradients)
        output2 = attention(query, key, value)
        loss2 = output2.sum()
        loss2.backward()
        
        # Check gradients accumulated
        assert not torch.equal(query.grad, query_grad_1)
        assert not torch.equal(key.grad, key_grad_1)
        assert not torch.equal(value.grad, value_grad_1)


class TestNumericalAccuracy:
    """Test numerical accuracy and consistency."""
    
    def test_attention_self_consistency(self, identical_tensors, skip_if_ma_core_unavailable):
        """Test that attention produces consistent results with identical inputs."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        attention.eval()
        
        query = identical_tensors['query']
        key = identical_tensors['key'] 
        value = identical_tensors['value']
        
        with torch.no_grad():
            output1 = attention(query, key, value)
            output2 = attention(query, key, value)
        
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-7)
    
    def test_attention_symmetry(self, skip_if_ma_core_unavailable):
        """Test attention symmetry properties."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        attention.eval()
        
        # Create symmetric inputs
        batch_size, seq_len, num_heads, head_dim = 1, 4, 1, 8
        
        # Create identical Q and K for symmetry test
        query = torch.ones(batch_size, seq_len, num_heads, head_dim) 
        key = query.clone()
        value = torch.randn(batch_size, seq_len, num_heads, head_dim)
        
        with torch.no_grad():
            output = attention(query, key, value)
        
        # With identical Q and K, attention should be roughly uniform
        assert output.shape == (batch_size, seq_len, num_heads, head_dim)
        assert not torch.isnan(output).any()
    
    def test_attention_scale_invariance(self, small_tensors, skip_if_ma_core_unavailable):
        """Test that attention scaling works correctly.""" 
        attention = MACoreAttention(sparse=False, fallback_training=True)
        attention.eval()
        
        query = small_tensors['query'].detach()
        key = small_tensors['key'].detach()
        value = small_tensors['value'].detach()
        
        # Scale inputs
        scale_factor = 2.0
        query_scaled = query * scale_factor
        key_scaled = key * scale_factor
        
        with torch.no_grad():
            output_normal = attention(query, key, value)
            output_scaled = attention(query_scaled, key_scaled, value)
        
        # Attention should handle scaling properly (due to normalization)
        assert output_normal.shape == output_scaled.shape
        assert not torch.isnan(output_normal).any()
        assert not torch.isnan(output_scaled).any()


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.error
    def test_mismatched_sequence_lengths(self, test_helper, skip_if_ma_core_unavailable):
        """Test error handling for mismatched sequence lengths."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        attention.train()  # Use PyTorch for better error messages
        
        mismatched = test_helper.create_mismatched_tensors()
        
        with pytest.raises(Exception) as exc_info:
            attention(mismatched['query_normal'], mismatched['key_seq_mismatch'], mismatched['query_normal'])
        
        # Should mention sequence length in error
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ['sequence', 'length', 'mismatch', 'size'])
    
    @pytest.mark.error
    def test_mismatched_head_dimensions(self, test_helper, skip_if_ma_core_unavailable):
        """Test error handling for mismatched head dimensions."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        attention.train()
        
        mismatched = test_helper.create_mismatched_tensors()
        
        with pytest.raises(Exception) as exc_info:
            attention(mismatched['query_normal'], mismatched['query_normal'], mismatched['value_head_mismatch'])
        
        # Should mention dimension in error  
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ['dimension', 'head', 'mismatch', 'size'])
    
    @pytest.mark.error
    def test_empty_tensors(self, skip_if_ma_core_unavailable):
        """Test handling of empty tensors."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        
        # Create empty tensors
        empty_query = torch.randn(0, 4, 2, 8)
        empty_key = torch.randn(0, 4, 2, 8)
        empty_value = torch.randn(0, 4, 2, 8)
        
        with pytest.raises(Exception):
            attention(empty_query, empty_key, empty_value)
    
    def test_extreme_window_sizes(self, small_tensors, skip_if_ma_core_unavailable):
        """Test handling of extreme window sizes."""
        query = small_tensors['query'].detach()
        key = small_tensors['key'].detach()
        value = small_tensors['value'].detach()
        seq_len = query.shape[1]
        
        # Very small window
        attention_small = MACoreAttention(sparse=True, window_size=1, fallback_training=True)
        output_small = attention_small(query, key, value)
        assert output_small.shape == query.shape
        
        # Very large window (larger than sequence)
        attention_large = MACoreAttention(sparse=True, window_size=seq_len * 2, fallback_training=True)
        output_large = attention_large(query, key, value)
        assert output_large.shape == query.shape


class TestAttentionFactory:
    """Test attention factory functions."""
    
    def test_create_attention_layer(self, skip_if_ma_core_unavailable):
        """Test attention layer factory function."""
        # Test dense attention creation
        dense_attn = create_attention_layer("dense", use_causal_mask=True)
        assert isinstance(dense_attn, MACoreAttention)
        assert not dense_attn.sparse
        assert dense_attn.use_causal_mask
        
        # Test sparse attention creation  
        sparse_attn = create_attention_layer("sparse", window_size=16)
        assert isinstance(sparse_attn, MACoreAttention)
        assert sparse_attn.sparse
        assert sparse_attn.window_size == 16
        
        # Test auto attention creation
        auto_attn = create_attention_layer("auto", window_size=32)
        assert isinstance(auto_attn, MACoreAttention)
        assert auto_attn.sparse  # Should default to sparse
    
    def test_invalid_attention_type(self, skip_if_ma_core_unavailable):
        """Test error handling for invalid attention types."""
        with pytest.raises(ValueError) as exc_info:
            create_attention_layer("invalid_type")
        
        assert "unknown attention type" in str(exc_info.value).lower()


class TestParametrizedAttention:
    """Test attention with various parameter combinations."""
    
    @pytest.mark.parametrize("sparse", [True, False])
    @pytest.mark.parametrize("window_size", [2, 8, 16])
    @pytest.mark.parametrize("use_causal_mask", [True, False])
    def test_attention_configurations(self, tensor_shapes, sparse, window_size, use_causal_mask, skip_if_ma_core_unavailable):
        """Test various attention configurations."""
        if sparse and use_causal_mask:
            pytest.skip("Sparse attention doesn't use causal_mask parameter")
        
        # Create attention module
        if sparse:
            attention = MACoreAttention(sparse=True, window_size=window_size, fallback_training=True)
        else:
            attention = MACoreAttention(sparse=False, use_causal_mask=use_causal_mask, fallback_training=True)
        
        # Create test tensors
        shape = tensor_shapes['shape']
        query = torch.randn(*shape)
        key = torch.randn(*shape)
        value = torch.randn(*shape)
        
        # Test forward pass
        output = attention(query, key, value)
        
        assert output.shape == shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    @pytest.mark.parametrize("batch_size,seq_len,num_heads,head_dim", [
        (1, 8, 1, 16),   # Single head
        (1, 8, 4, 16),   # Multi-head
        (2, 16, 2, 32),  # Batch processing
        (1, 32, 8, 64),  # Larger dimensions
    ])
    def test_various_tensor_sizes(self, batch_size, seq_len, num_heads, head_dim, skip_if_ma_core_unavailable):
        """Test attention with various tensor sizes."""
        attention = MACoreAttention(sparse=True, window_size=4, fallback_training=True)
        
        shape = (batch_size, seq_len, num_heads, head_dim)
        query = torch.randn(*shape)
        key = torch.randn(*shape)
        value = torch.randn(*shape)
        
        output = attention(query, key, value)
        
        assert output.shape == shape
        assert not torch.isnan(output).any()
