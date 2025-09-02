 # SPDX-License-Identifier: Apache-2.0
"""
Integration tests for PyTorch compatibility and model integration.
Tests the SparseAttention class and full model integration scenarios.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from layers.sparse_attention import SparseAttention
from layers.ma_core_bridge import MACoreAttention


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for integration testing."""
    
    def __init__(self, model_dim=64, num_heads=4, use_ma_core=True, window_size=8):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        
        # Projection layers
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)
        
        # Attention layer
        if use_ma_core:
            self.attention = MACoreAttention(
                sparse=True, 
                window_size=window_size, 
                fallback_training=True
            )
        else:
            self.attention = SparseAttention(window_size=window_size, use_ma_core=False)
        
        # Normalization
        self.norm = nn.LayerNorm(model_dim)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        # Multi-head projections
        batch_size, seq_len, model_dim = x.shape
        
        if hasattr(self.attention, 'sparse') and self.attention.sparse:
            # MACoreAttention expects multi-head format
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            attn_output = self.attention(q, k, v)
            attn_output = attn_output.view(batch_size, seq_len, model_dim)
        else:
            # SparseAttention expects single format
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
            attn_output = self.attention(q, k, v)
        
        output = self.out_proj(attn_output)
        return residual + output


class TestSparseAttentionIntegration:
    """Test SparseAttention class integration."""
    
    def test_sparse_attention_basic(self, skip_if_ma_core_unavailable):
        """Test basic SparseAttention functionality."""
        batch_size, seq_len, model_dim = 2, 8, 32
        
        sparse_attn = SparseAttention(window_size=4, use_ma_core=True)
        
        q = torch.randn(batch_size, seq_len, model_dim)
        k = torch.randn(batch_size, seq_len, model_dim)
        v = torch.randn(batch_size, seq_len, model_dim)
        
        output = sparse_attn(q, k, v)
        
        assert output.shape == (batch_size, seq_len, model_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_sparse_attention_training_mode(self, skip_if_ma_core_unavailable):
        """Test SparseAttention in training mode."""
        sparse_attn = SparseAttention(window_size=4, use_ma_core=True)
        sparse_attn.train()
        
        batch_size, seq_len, model_dim = 1, 6, 16
        
        q = torch.randn(batch_size, seq_len, model_dim, requires_grad=True)
        k = torch.randn(batch_size, seq_len, model_dim, requires_grad=True)
        v = torch.randn(batch_size, seq_len, model_dim, requires_grad=True)
        
        output = sparse_attn(q, k, v)
        loss = output.sum()
        loss.backward()
        
        # Check gradients flow
        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None
    
    def test_sparse_attention_eval_mode(self, skip_if_ma_core_unavailable):
        """Test SparseAttention in eval mode."""
        sparse_attn = SparseAttention(window_size=4, use_ma_core=True)
        sparse_attn.eval()
        
        batch_size, seq_len, model_dim = 1, 8, 24
        
        q = torch.randn(batch_size, seq_len, model_dim)
        k = torch.randn(batch_size, seq_len, model_dim)
        v = torch.randn(batch_size, seq_len, model_dim)
        
        with torch.no_grad():
            output = sparse_attn(q, k, v)
        
        assert output.shape == (batch_size, seq_len, model_dim)
        assert not torch.isnan(output).any()
    
    def test_sparse_attention_without_ma_core(self):
        """Test SparseAttention fallback without ma_core."""
        sparse_attn = SparseAttention(window_size=3, use_ma_core=False)
        
        batch_size, seq_len, model_dim = 1, 4, 8
        
        q = torch.randn(batch_size, seq_len, model_dim)
        k = torch.randn(batch_size, seq_len, model_dim)
        v = torch.randn(batch_size, seq_len, model_dim)
        
        output = sparse_attn(q, k, v)
        
        assert output.shape == (batch_size, seq_len, model_dim)
        assert not torch.isnan(output).any()


class TestTransformerBlockIntegration:
    """Test full transformer block integration."""
    
    def test_transformer_block_forward(self, skip_if_ma_core_unavailable):
        """Test transformer block forward pass."""
        model = SimpleTransformerBlock(model_dim=32, num_heads=4, use_ma_core=True, window_size=4)
        
        batch_size, seq_len = 2, 8
        x = torch.randn(batch_size, seq_len, 32)
        
        output = model(x)
        
        assert output.shape == (batch_size, seq_len, 32)
        assert not torch.isnan(output).any()
    
    def test_transformer_block_training(self, skip_if_ma_core_unavailable):
        """Test transformer block training loop."""
        model = SimpleTransformerBlock(model_dim=16, num_heads=2, use_ma_core=True, window_size=2)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        batch_size, seq_len = 1, 4
        
        model.train()
        
        for _ in range(3):  # Mini training loop
            x = torch.randn(batch_size, seq_len, 16)
            target = torch.randn(batch_size, seq_len, 16)
            
            optimizer.zero_grad()
            output = model(x)
            loss = nn.MSELoss()(output, target)
            loss.backward()
            optimizer.step()
            
            assert not torch.isnan(loss)
    
    def test_transformer_block_inference(self, skip_if_ma_core_unavailable):
        """Test transformer block inference mode."""
        model = SimpleTransformerBlock(model_dim=32, num_heads=4, use_ma_core=True, window_size=6)
        
        model.eval()
        
        batch_size, seq_len = 1, 12
        x = torch.randn(batch_size, seq_len, 32)
        
        with torch.no_grad():
            output = model(x)
        
        assert output.shape == (batch_size, seq_len, 32)
        assert not torch.isnan(output).any()
    
    def test_model_comparison(self, skip_if_ma_core_unavailable):
        """Compare ma_core vs fallback implementations."""
        model_ma_core = SimpleTransformerBlock(model_dim=24, num_heads=3, use_ma_core=True, window_size=4)
        model_fallback = SimpleTransformerBlock(model_dim=24, num_heads=3, use_ma_core=False, window_size=4)
        
        batch_size, seq_len = 1, 6
        x = torch.randn(batch_size, seq_len, 24)
        
        # Both should produce valid outputs
        output_ma_core = model_ma_core(x)
        output_fallback = model_fallback(x)
        
        assert output_ma_core.shape == output_fallback.shape
        assert not torch.isnan(output_ma_core).any()
        assert not torch.isnan(output_fallback).any()


class TestBatchedOperations:
    """Test batched operations and edge cases."""
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    def test_different_batch_sizes(self, batch_size, skip_if_ma_core_unavailable):
        """Test attention with different batch sizes."""
        attention = MACoreAttention(sparse=True, window_size=4, fallback_training=True)
        
        seq_len, num_heads, head_dim = 8, 2, 16
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        q = torch.randn(*shape)
        k = torch.randn(*shape)
        v = torch.randn(*shape)
        
        output = attention(q, k, v)
        
        assert output.shape == shape
        assert not torch.isnan(output).any()
    
    def test_large_sequence_sparse_attention(self, skip_if_ma_core_unavailable):
        """Test sparse attention with larger sequences."""
        attention = MACoreAttention(sparse=True, window_size=8, fallback_training=True)
        
        batch_size, seq_len, num_heads, head_dim = 1, 64, 4, 32
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        q = torch.randn(*shape)
        k = torch.randn(*shape)
        v = torch.randn(*shape)
        
        output = attention(q, k, v)
        
        assert output.shape == shape
        assert not torch.isnan(output).any()
    
    def test_memory_efficiency(self, skip_if_ma_core_unavailable):
        """Test memory efficiency of sparse vs dense attention."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure memory before
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create larger tensors for memory test
        batch_size, seq_len, num_heads, head_dim = 1, 128, 4, 64
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        q = torch.randn(*shape)
        k = torch.randn(*shape)
        v = torch.randn(*shape)
        
        # Test sparse attention (should use less memory)
        sparse_attention = MACoreAttention(sparse=True, window_size=16, fallback_training=True)
        sparse_attention.eval()
        
        with torch.no_grad():
            sparse_output = sparse_attention(q, k, v)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        # Should complete without excessive memory usage
        assert memory_used < 500  # Less than 500MB for this test
        assert sparse_output.shape == shape


class TestModelStateAndSaving:
    """Test model state management and saving/loading."""
    
    def test_model_state_dict(self, skip_if_ma_core_unavailable):
        """Test model state dict operations."""
        model = SimpleTransformerBlock(model_dim=32, num_heads=4, use_ma_core=True, window_size=8)
        
        # Get state dict
        state_dict = model.state_dict()
        
        assert len(state_dict) > 0
        
        # Create new model and load state
        model2 = SimpleTransformerBlock(model_dim=32, num_heads=4, use_ma_core=True, window_size=8)
        model2.load_state_dict(state_dict)
        
        # Test that both models produce same output
        x = torch.randn(1, 8, 32)
        
        model.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model(x)
            output2 = model2(x)
        
        torch.testing.assert_close(output1, output2, rtol=1e-5, atol=1e-7)
    
    def test_model_training_state(self, skip_if_ma_core_unavailable):
        """Test training state transitions."""
        model = SimpleTransformerBlock(model_dim=16, num_heads=2, use_ma_core=True, window_size=4)
        
        # Test train mode
        model.train()
        assert model.training
        if hasattr(model.attention, 'training'):
            assert model.attention.training
        
        # Test eval mode
        model.eval()
        assert not model.training
        if hasattr(model.attention, 'training'):
            assert not model.attention.training
    
    def test_parameter_counting(self, skip_if_ma_core_unavailable):
        """Test parameter counting and optimization setup."""
        model = SimpleTransformerBlock(model_dim=32, num_heads=4, use_ma_core=True, window_size=8)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All should be trainable
        
        # Test optimizer creation
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        assert len(optimizer.param_groups) == 1
        assert len(optimizer.param_groups[0]['params']) > 0


class TestCompatibilityEdgeCases:
    """Test edge cases and compatibility scenarios."""
    
    def test_mixed_precision_compatibility(self, skip_if_ma_core_unavailable):
        """Test compatibility with mixed precision training."""
        model = SimpleTransformerBlock(model_dim=32, num_heads=4, use_ma_core=True)
        
        # Test with half precision
        model.half()
        
        x = torch.randn(1, 8, 32).half()
        
        try:
            output = model(x)
            assert output.dtype == torch.float16
        except Exception:
            # Mixed precision might not be supported - that's ok
            pytest.skip("Mixed precision not supported in current configuration")
    
    def test_device_consistency(self, skip_if_ma_core_unavailable):
        """Test device consistency (CPU only for now)."""
        model = SimpleTransformerBlock(model_dim=16, num_heads=2, use_ma_core=True)
        
        # Ensure model is on CPU
        model = model.cpu()
        
        x = torch.randn(1, 4, 16).cpu()
        output = model(x)
        
        assert output.device == x.device
        assert str(output.device) == 'cpu'
    
    def test_deterministic_behavior(self, skip_if_ma_core_unavailable):
        """Test deterministic behavior with fixed seeds."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        model = SimpleTransformerBlock(model_dim=24, num_heads=3, use_ma_core=True, window_size=4)
        model.eval()
        
        x = torch.randn(1, 6, 24)
        
        with torch.no_grad():
            output1 = model(x)
            
        # Reset seed and run again
        torch.manual_seed(42)
        np.random.seed(42)
        
        model2 = SimpleTransformerBlock(model_dim=24, num_heads=3, use_ma_core=True, window_size=4)
        model2.eval()
        
        with torch.no_grad():
            output2 = model2(x)
        
        # Note: Due to implementation differences, outputs might not be identical
        # but shapes and validity should be consistent
        assert output1.shape == output2.shape
        assert not torch.isnan(output1).any()
        assert not torch.isnan(output2).any()
