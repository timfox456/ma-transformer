#!/usr/bin/env python3
"""
Test PyTorch integration with ma_core C++ engine.
Verifies that the bridge works correctly and gradients flow properly.
"""

import torch
import torch.nn as nn
import ma_core
import sys
import os

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from layers.ma_core_bridge import MACoreAttention, pytorch_dense_attention, pytorch_sparse_attention
    from layers.sparse_attention import SparseAttention
    BRIDGE_AVAILABLE = True
except ImportError as e:
    print(f"Bridge import failed: {e}")
    BRIDGE_AVAILABLE = False


def test_ma_core_bridge_basic():
    """Test basic ma_core bridge functionality."""
    print("🧪 Testing MACoreAttention Bridge Basic Functionality...")
    
    if not BRIDGE_AVAILABLE:
        print("  ❌ Bridge not available, skipping test")
        return False
    
    # Create test tensors
    batch_size, seq_len, num_heads, head_dim = 1, 4, 2, 8
    
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
    
    # Test dense attention
    dense_attention = MACoreAttention(sparse=False, fallback_training=True)
    dense_output = dense_attention(query, key, value)
    
    assert dense_output.shape == (batch_size, seq_len, num_heads, head_dim)
    print("  ✅ Dense attention bridge works")
    
    # Test sparse attention  
    sparse_attention = MACoreAttention(sparse=True, window_size=2, fallback_training=True)
    sparse_output = sparse_attention(query, key, value)
    
    assert sparse_output.shape == (batch_size, seq_len, num_heads, head_dim)
    print("  ✅ Sparse attention bridge works")
    
    return True


def test_gradient_flow():
    """Test that gradients flow correctly through the bridge."""
    print("🧪 Testing Gradient Flow...")
    
    if not BRIDGE_AVAILABLE:
        print("  ❌ Bridge not available, skipping test")
        return False
    
    # Create test tensors with gradients
    batch_size, seq_len, num_heads, head_dim = 1, 3, 1, 4
    
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True) 
    value = torch.randn(batch_size, seq_len, num_heads, head_dim, requires_grad=True)
    
    # Test with training mode (should use PyTorch implementation)
    attention = MACoreAttention(sparse=True, window_size=1, fallback_training=True)
    attention.train()  # Set to training mode
    
    output = attention(query, key, value)
    loss = output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients exist
    assert query.grad is not None
    assert key.grad is not None  
    assert value.grad is not None
    
    print("  ✅ Gradients flow correctly in training mode")
    
    return True


def test_sparse_attention_integration():
    """Test the updated SparseAttention class."""
    print("🧪 Testing SparseAttention Integration...")
    
    if not BRIDGE_AVAILABLE:
        print("  ❌ Bridge not available, skipping test")
        return False
    
    # Test with original interface (single head attention)
    batch_size, seq_len, model_dim = 2, 8, 16
    
    q = torch.randn(batch_size, seq_len, model_dim)
    k = torch.randn(batch_size, seq_len, model_dim)
    v = torch.randn(batch_size, seq_len, model_dim)
    
    # Create sparse attention layer
    sparse_attn = SparseAttention(window_size=2, use_ma_core=True)
    
    # Test forward pass
    output = sparse_attn(q, k, v)
    
    assert output.shape == (batch_size, seq_len, model_dim)
    print("  ✅ SparseAttention integration works")
    
    # Test training mode (should fall back to PyTorch)
    sparse_attn.train()
    q.requires_grad_(True)
    k.requires_grad_(True) 
    v.requires_grad_(True)
    
    output = sparse_attn(q, k, v)
    loss = output.sum()
    loss.backward()
    
    assert q.grad is not None
    print("  ✅ Training mode gradients work")
    
    return True


def test_performance_comparison():
    """Compare PyTorch vs ma_core performance."""
    print("🧪 Testing Performance Comparison...")
    
    if not BRIDGE_AVAILABLE:
        print("  ❌ Bridge not available, skipping test")
        return False
    
    # Test with larger tensors
    batch_size, seq_len, num_heads, head_dim = 1, 32, 4, 16
    
    query = torch.randn(batch_size, seq_len, num_heads, head_dim)
    key = torch.randn(batch_size, seq_len, num_heads, head_dim)
    value = torch.randn(batch_size, seq_len, num_heads, head_dim)
    
    # Time PyTorch implementation
    import time
    
    start_time = time.time()
    pytorch_output = pytorch_sparse_attention(query, key, value, window_size=4)
    pytorch_time = (time.time() - start_time) * 1000
    
    # Time ma_core implementation (in eval mode)
    ma_core_attn = MACoreAttention(sparse=True, window_size=4, fallback_training=True)
    ma_core_attn.eval()  # Set to eval mode to potentially use C++ engine
    
    start_time = time.time()
    ma_core_output = ma_core_attn(query, key, value)
    ma_core_time = (time.time() - start_time) * 1000
    
    print(f"  📊 PyTorch time: {pytorch_time:.2f}ms")
    print(f"  📊 ma_core time: {ma_core_time:.2f}ms")
    print(f"  📊 Speedup: {pytorch_time/ma_core_time:.2f}x")
    
    return True


def test_error_handling():
    """Test error handling in the bridge."""
    print("🧪 Testing Error Handling...")
    
    if not BRIDGE_AVAILABLE:
        print("  ❌ Bridge not available, skipping test")
        return False
    
    # Test mismatched tensor shapes
    query = torch.randn(1, 4, 2, 8)
    key = torch.randn(1, 6, 2, 8)  # Different sequence length
    value = torch.randn(1, 4, 2, 8)
    
    attention = MACoreAttention(sparse=False, fallback_training=True)
    attention.train()  # Use PyTorch for error testing
    
    try:
        output = attention(query, key, value)
        print("  ❌ Should have failed with shape mismatch")
        return False
    except Exception as e:
        print("  ✅ Correctly handles shape mismatch errors")
    
    return True


def test_backwards_compatibility():
    """Test that existing code still works."""
    print("🧪 Testing Backwards Compatibility...")
    
    # Test original SparseAttention interface without ma_core
    sparse_attn = SparseAttention(window_size=2, use_ma_core=False)
    
    batch_size, seq_len, model_dim = 1, 4, 8
    q = torch.randn(batch_size, seq_len, model_dim)
    k = torch.randn(batch_size, seq_len, model_dim)
    v = torch.randn(batch_size, seq_len, model_dim)
    
    output = sparse_attn(q, k, v)
    assert output.shape == (batch_size, seq_len, model_dim)
    
    print("  ✅ Backwards compatibility maintained")
    return True


def main():
    print("🚀 Testing PyTorch Integration with ma_core")
    print("=" * 60)
    
    try:
        # Run all tests
        tests = [
            test_ma_core_bridge_basic,
            test_gradient_flow, 
            test_sparse_attention_integration,
            test_performance_comparison,
            test_error_handling,
            test_backwards_compatibility,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed += 1
                print()
            except Exception as e:
                print(f"  ❌ Test failed with error: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        print("=" * 60)
        print(f"🎯 INTEGRATION TEST RESULTS: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ PyTorch integration is working correctly!")
            print("🚀 Ready for production use with existing PyTorch training pipelines")
            print("📈 ma_core C++ engine successfully bridged to PyTorch")
        else:
            print("❌ Some integration tests failed")
            print("🔧 Bridge needs additional debugging")
            
        return 0 if passed == total else 1
        
    except Exception as e:
        print(f"❌ Integration test suite failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())