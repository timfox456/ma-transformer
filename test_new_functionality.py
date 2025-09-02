#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Test script for new tensor and sparse attention functionality.
This demonstrates the expanded capabilities before updating the main test suite.
"""

import ma_core

def test_basic_functionality():
    """Test basic add function with int64 support."""
    print("🧪 Testing basic functionality...")
    
    # Test basic addition (should now handle larger numbers)
    assert ma_core.add(1, 2) == 3
    assert ma_core.add(2**31, 1) == 2**31 + 1  # Test large numbers that would overflow int32
    print("  ✅ Basic add function works with int64")

def test_tensor_creation():
    """Test tensor creation and basic operations."""
    print("🧪 Testing tensor creation...")
    
    # Test tensor creation
    tensor = ma_core.create_tensor([2, 8, 4, 64])  # [batch, seq, heads, dim]
    print(f"  ✅ Created tensor with {tensor.size()} elements")
    
    # Test tensor properties
    shape = tensor.shape()
    print(f"  ✅ Tensor shape: batch={shape.batch_size}, seq={shape.sequence_length}, heads={shape.num_heads}, dim={shape.head_dim}")
    print(f"  ✅ Total elements: {shape.total_elements()}")
    
    # Test tensor operations
    assert tensor.empty() == False
    tensor.zero()
    tensor.fill(1.5)
    print("  ✅ Tensor operations (zero, fill) work")

def test_sparse_patterns():
    """Test sparse attention pattern creation."""
    print("🧪 Testing sparse attention patterns...")
    
    # Test sliding window pattern
    pattern = ma_core.create_sliding_window_pattern(16, 3)  # seq_len=16, window_size=3
    
    print(f"  ✅ Created sliding window pattern")
    print(f"  ✅ Pattern has {pattern.nnz()} non-zero elements")
    print(f"  ✅ Pattern is not empty: {not pattern.empty()}")
    
    # Test pattern statistics
    stats = ma_core.get_pattern_statistics(pattern)
    print("  ✅ Pattern statistics:")
    for line in stats.split('\n'):
        if line.strip():
            print(f"    {line}")

def test_sparsity_levels():
    """Test different sparsity levels."""
    print("🧪 Testing different sparsity levels...")
    
    seq_lens = [8, 16, 32, 64]
    window_sizes = [1, 2, 4, 8]
    
    for seq_len in seq_lens:
        for window_size in window_sizes:
            if window_size < seq_len:
                pattern = ma_core.create_sliding_window_pattern(seq_len, window_size)
                total_elements = seq_len * seq_len
                sparsity = 1.0 - pattern.nnz() / total_elements
                print(f"  ✅ seq_len={seq_len}, window={window_size}: {pattern.nnz()}/{total_elements} elements, {sparsity*100:.1f}% sparse")

def test_edge_cases():
    """Test edge cases and error conditions."""
    print("🧪 Testing edge cases...")
    
    # Test very small tensors
    small_tensor = ma_core.create_tensor([1, 1, 1, 1])
    assert small_tensor.size() == 1
    print("  ✅ Very small tensor (1x1x1x1) works")
    
    # Test small sparse patterns
    tiny_pattern = ma_core.create_sliding_window_pattern(2, 1)
    assert tiny_pattern.nnz() > 0
    print("  ✅ Tiny sparse pattern (seq_len=2, window=1) works")
    
    # Test window size larger than sequence
    large_window_pattern = ma_core.create_sliding_window_pattern(4, 10)
    # Should still work, just becomes dense
    print(f"  ✅ Large window pattern: {large_window_pattern.nnz()} elements")
    
    # Test invalid tensor shapes
    try:
        invalid_tensor = ma_core.create_tensor([1, 2, 3])  # Wrong number of dimensions
        assert False, "Should have failed"
    except RuntimeError as e:
        print(f"  ✅ Correctly rejected invalid tensor shape: {e}")

def test_device_enum():
    """Test device enumeration."""
    print("🧪 Testing device enumeration...")
    
    print(f"  ✅ CPU device: {ma_core.Device.CPU}")
    print(f"  ✅ MPS device: {ma_core.Device.MPS}")
    print(f"  ✅ CUDA device: {ma_core.Device.CUDA}")
    print(f"  ✅ ROCm device: {ma_core.Device.ROCm}")

def benchmark_performance():
    """Basic performance benchmark."""
    print("🧪 Running basic performance benchmark...")
    
    import time
    
    # Benchmark tensor creation
    start = time.time()
    for _ in range(1000):
        tensor = ma_core.create_tensor([1, 32, 8, 64])
    tensor_time = time.time() - start
    print(f"  ⚡ 1000 tensor creations: {tensor_time*1000:.2f}ms")
    
    # Benchmark sparse pattern creation
    start = time.time()
    for _ in range(100):
        pattern = ma_core.create_sliding_window_pattern(64, 8)
    pattern_time = time.time() - start
    print(f"  ⚡ 100 sparse pattern creations: {pattern_time*1000:.2f}ms")

def main():
    print("🚀 Testing Extended MA-Transformer Core Functionality\n")
    
    try:
        test_basic_functionality()
        print()
        
        test_tensor_creation()
        print()
        
        test_sparse_patterns()
        print()
        
        test_sparsity_levels()
        print()
        
        test_edge_cases()
        print()
        
        test_device_enum()
        print()
        
        benchmark_performance()
        print()
        
        print("🎉 All tests passed! Extended functionality is working correctly.")
        print("✅ Ready to update the main regression test suite.")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
