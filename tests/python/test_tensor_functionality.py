#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Comprehensive test suite for tensor and sparse attention functionality.
Tests the extended C++ core engine capabilities.
"""

import unittest
import sys
import os
import time
import gc
from typing import List, Tuple

# Add the project root to the path to import ma_core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    import ma_core
except ImportError as e:
    raise ImportError(f"Failed to import ma_core extension. Make sure it's built. Error: {e}")


class TestTensorFunctionality(unittest.TestCase):
    """Test tensor creation and basic operations."""
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Clean up after each test."""
        gc.collect()
    
    def test_tensor_creation_basic(self):
        """Test basic tensor creation."""
        shape = [1, 4, 2, 8]  # [batch, seq, heads, dim]
        tensor = ma_core.create_tensor(shape)
        
        self.assertEqual(tensor.size(), 64)  # 1*4*2*8
        self.assertFalse(tensor.empty())
        
        tensor_shape = tensor.shape()
        self.assertEqual(tensor_shape.batch_size, 1)
        self.assertEqual(tensor_shape.sequence_length, 4)
        self.assertEqual(tensor_shape.num_heads, 2)
        self.assertEqual(tensor_shape.head_dim, 8)
        self.assertEqual(tensor_shape.total_elements(), 64)
    
    def test_tensor_creation_various_shapes(self):
        """Test tensor creation with various shapes."""
        test_shapes = [
            [1, 1, 1, 1],      # Minimal tensor
            [2, 8, 4, 64],     # Small transformer
            [1, 512, 8, 64],   # Medium sequence
            [4, 128, 12, 64],  # Batch processing
        ]
        
        for shape in test_shapes:
            with self.subTest(shape=shape):
                tensor = ma_core.create_tensor(shape)
                expected_size = shape[0] * shape[1] * shape[2] * shape[3]
                self.assertEqual(tensor.size(), expected_size)
                
                tensor_shape = tensor.shape()
                self.assertEqual(tensor_shape.batch_size, shape[0])
                self.assertEqual(tensor_shape.sequence_length, shape[1])
                self.assertEqual(tensor_shape.num_heads, shape[2])
                self.assertEqual(tensor_shape.head_dim, shape[3])
    
    def test_tensor_operations(self):
        """Test basic tensor operations."""
        tensor = ma_core.create_tensor([2, 4, 2, 8])
        
        # Test zero operation
        tensor.zero()
        
        # Test fill operation
        tensor.fill(3.14)
        
        # Test multiple operations
        tensor.zero()
        tensor.fill(1.0)
        tensor.zero()
    
    def test_tensor_invalid_shapes(self):
        """Test error handling for invalid shapes."""
        invalid_shapes = [
            [],           # Empty shape
            [1],          # 1D
            [1, 2],       # 2D
            [1, 2, 3],    # 3D
            [1, 2, 3, 4, 5],  # 5D
        ]
        
        for shape in invalid_shapes:
            with self.subTest(shape=shape):
                with self.assertRaises(RuntimeError):
                    ma_core.create_tensor(shape)
    
    def test_tensor_device_parameter(self):
        """Test tensor creation with device parameter."""
        # CPU device should work
        tensor_cpu = ma_core.create_tensor([1, 4, 2, 8], "cpu")
        self.assertEqual(tensor_cpu.size(), 64)
        
        # Test case insensitive
        tensor_cpu2 = ma_core.create_tensor([1, 4, 2, 8], "CPU")
        self.assertEqual(tensor_cpu2.size(), 64)
    
    def test_large_tensors(self):
        """Test creation of larger tensors."""
        # Test a reasonably large tensor (but not too large for testing)
        large_shape = [2, 256, 8, 64]  # ~2M elements
        large_tensor = ma_core.create_tensor(large_shape)
        
        expected_size = 2 * 256 * 8 * 64
        self.assertEqual(large_tensor.size(), expected_size)
        
        # Test operations on large tensor
        large_tensor.zero()
        large_tensor.fill(0.5)


class TestSparsePatterns(unittest.TestCase):
    """Test sparse attention pattern functionality."""
    
    def test_sliding_window_basic(self):
        """Test basic sliding window pattern creation."""
        seq_len = 8
        window_size = 2
        pattern = ma_core.create_sliding_window_pattern(seq_len, window_size)
        
        self.assertFalse(pattern.empty())
        self.assertGreater(pattern.nnz(), 0)
        
        # Check pattern shape
        shape = pattern.shape
        self.assertEqual(shape.sequence_length, seq_len)
    
    def test_sliding_window_sparsity(self):
        """Test sliding window patterns with different sparsity levels."""
        test_cases = [
            (8, 1),    # Very sparse
            (16, 2),   # Sparse
            (32, 4),   # Medium sparse
            (16, 8),   # Less sparse
        ]
        
        for seq_len, window_size in test_cases:
            with self.subTest(seq_len=seq_len, window_size=window_size):
                pattern = ma_core.create_sliding_window_pattern(seq_len, window_size)
                
                # Calculate expected sparsity
                total_elements = seq_len * seq_len
                nonzero_elements = pattern.nnz()
                sparsity_ratio = 1.0 - nonzero_elements / total_elements
                
                # Sliding window should be sparse
                self.assertGreater(sparsity_ratio, 0.0)
                self.assertLess(sparsity_ratio, 1.0)
                
                # For small windows, should be quite sparse
                if window_size <= seq_len // 4:
                    self.assertGreater(sparsity_ratio, 0.3)  # At least 30% sparse
    
    def test_sliding_window_edge_cases(self):
        """Test sliding window patterns with edge cases."""
        # Very small sequence
        tiny_pattern = ma_core.create_sliding_window_pattern(2, 1)
        self.assertGreater(tiny_pattern.nnz(), 0)
        
        # Window size equals sequence length (should be dense)
        dense_pattern = ma_core.create_sliding_window_pattern(4, 4)
        self.assertEqual(dense_pattern.nnz(), 16)  # 4*4 = fully dense
        
        # Window size larger than sequence (should be dense)
        over_dense_pattern = ma_core.create_sliding_window_pattern(4, 10)
        self.assertEqual(over_dense_pattern.nnz(), 16)  # 4*4 = fully dense
    
    def test_pattern_statistics(self):
        """Test pattern statistics functionality."""
        pattern = ma_core.create_sliding_window_pattern(16, 3)
        stats = ma_core.get_pattern_statistics(pattern)
        
        self.assertIsInstance(stats, str)
        self.assertIn("Pattern Statistics", stats)
        self.assertIn("Total elements", stats)
        self.assertIn("Non-zero elements", stats)
        self.assertIn("Sparsity ratio", stats)
        
        # Check that numbers make sense
        lines = stats.split('\n')
        total_line = [l for l in lines if "Total elements" in l][0]
        nonzero_line = [l for l in lines if "Non-zero elements" in l][0]
        
        total = int(total_line.split(': ')[1])
        nonzero = int(nonzero_line.split(': ')[1])
        
        self.assertEqual(total, 256)  # 16*16
        self.assertGreater(nonzero, 0)
        self.assertLess(nonzero, total)
    
    def test_multiple_pattern_sizes(self):
        """Test patterns with various sequence lengths."""
        seq_lengths = [4, 8, 16, 32, 64, 128]
        window_size = 4
        
        for seq_len in seq_lengths:
            with self.subTest(seq_len=seq_len):
                pattern = ma_core.create_sliding_window_pattern(seq_len, window_size)
                
                # Pattern should exist and have reasonable sparsity
                self.assertGreater(pattern.nnz(), 0)
                
                # For larger sequences, should be more sparse
                total_elements = seq_len * seq_len
                sparsity = 1.0 - pattern.nnz() / total_elements
                
                if seq_len >= 16:
                    self.assertGreater(sparsity, 0.5)  # Should be at least 50% sparse


class TestDeviceEnumeration(unittest.TestCase):
    """Test device enumeration functionality."""
    
    def test_device_enum_values(self):
        """Test that device enum values are accessible."""
        # Test all device types exist
        self.assertIsNotNone(ma_core.Device.CPU)
        self.assertIsNotNone(ma_core.Device.MPS)
        self.assertIsNotNone(ma_core.Device.CUDA)
        self.assertIsNotNone(ma_core.Device.ROCm)
    
    def test_device_enum_comparison(self):
        """Test device enum comparisons."""
        cpu1 = ma_core.Device.CPU
        cpu2 = ma_core.Device.CPU
        mps = ma_core.Device.MPS
        
        self.assertEqual(cpu1, cpu2)
        self.assertNotEqual(cpu1, mps)


class TestPerformanceRegression(unittest.TestCase):
    """Performance regression tests."""
    
    def test_tensor_creation_performance(self):
        """Test tensor creation performance."""
        shape = [1, 64, 8, 64]  # Moderate size
        iterations = 1000
        
        start_time = time.time()
        for _ in range(iterations):
            tensor = ma_core.create_tensor(shape)
        end_time = time.time()
        
        elapsed_ms = (end_time - start_time) * 1000
        avg_time_per_creation = elapsed_ms / iterations
        
        # Should create tensors quickly (less than 0.1ms per tensor)
        self.assertLess(avg_time_per_creation, 0.1, 
                       f"Tensor creation took {avg_time_per_creation:.4f}ms per tensor")
    
    def test_sparse_pattern_performance(self):
        """Test sparse pattern creation performance."""
        seq_len = 64
        window_size = 8
        iterations = 100
        
        start_time = time.time()
        for _ in range(iterations):
            pattern = ma_core.create_sliding_window_pattern(seq_len, window_size)
        end_time = time.time()
        
        elapsed_ms = (end_time - start_time) * 1000
        avg_time_per_pattern = elapsed_ms / iterations
        
        # Should create patterns quickly (less than 1ms per pattern)
        self.assertLess(avg_time_per_pattern, 1.0,
                       f"Pattern creation took {avg_time_per_pattern:.4f}ms per pattern")
    
    def test_memory_usage_scaling(self):
        """Test that memory usage scales reasonably."""
        base_shape = [1, 16, 4, 32]
        base_tensor = ma_core.create_tensor(base_shape)
        base_size = base_tensor.size()
        
        # Double each dimension and check scaling
        double_shape = [2, 32, 8, 64]
        double_tensor = ma_core.create_tensor(double_shape)
        double_size = double_tensor.size()
        
        # Should be 16x larger (2*2*2*2)
        expected_ratio = 16
        actual_ratio = double_size / base_size
        
        self.assertEqual(actual_ratio, expected_ratio)


class TestIntegrationRegression(unittest.TestCase):
    """Integration tests for backward compatibility."""
    
    def test_original_add_function(self):
        """Test that original add function still works (regression test)."""
        # These are the exact tests from the original test suite
        self.assertEqual(ma_core.add(1, 2), 3)
        self.assertEqual(ma_core.add(0, 0), 0)
        self.assertEqual(ma_core.add(-1, 1), 0)
        self.assertEqual(ma_core.add(100, 200), 300)
    
    def test_large_number_support(self):
        """Test that large numbers now work (fixing original regression)."""
        # These were the failing tests in the original suite
        large_num = 2**31  # 2^30 was failing before
        self.assertEqual(ma_core.add(large_num, 1), large_num + 1)
        self.assertEqual(ma_core.add(large_num, large_num), 2 * large_num)
        
        # Test with very large numbers
        very_large = 2**60
        self.assertEqual(ma_core.add(very_large, 0), very_large)
    
    def test_mathematical_properties_still_hold(self):
        """Test that mathematical properties still hold with new implementation."""
        test_pairs = [(1, 2), (100, -50), (0, 100), (-7, -3)]
        
        for a, b in test_pairs:
            with self.subTest(a=a, b=b):
                # Commutative property
                self.assertEqual(ma_core.add(a, b), ma_core.add(b, a))
                
                # Identity property
                self.assertEqual(ma_core.add(a, 0), a)
                self.assertEqual(ma_core.add(0, b), b)


class TestExtendedErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def test_tensor_error_messages(self):
        """Test that error messages are informative."""
        try:
            ma_core.create_tensor([1, 2, 3])  # Wrong dimensions
            self.fail("Should have raised RuntimeError")
        except RuntimeError as e:
            error_msg = str(e)
            self.assertIn("4 dimensions", error_msg)
            self.assertIn("batch, seq, heads, dim", error_msg)
    
    def test_negative_values_handling(self):
        """Test handling of negative values where appropriate."""
        # Negative numbers in add should work
        self.assertEqual(ma_core.add(-5, -3), -8)
        self.assertEqual(ma_core.add(-100, 50), -50)
        
        # Test edge case: most negative int64
        min_int64 = -2**63
        self.assertEqual(ma_core.add(min_int64, 0), min_int64)
    
    def test_zero_size_handling(self):
        """Test handling of zero-size patterns and tensors."""
        # Zero in dimensions should be handled gracefully
        try:
            zero_tensor = ma_core.create_tensor([0, 4, 2, 8])
            # If this doesn't crash, that's good
        except RuntimeError:
            # If it does crash with clear error, that's also acceptable
            pass
        
        # Zero sequence length pattern
        try:
            zero_pattern = ma_core.create_sliding_window_pattern(0, 1)
        except (RuntimeError, ValueError):
            # Should either work or fail gracefully
            pass


if __name__ == '__main__':
    # Configure test runner for comprehensive output
    unittest.main(
        verbosity=2,
        buffer=True,
        failfast=False,
    )
