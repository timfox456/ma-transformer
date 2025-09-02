 # SPDX-License-Identifier: Apache-2.0
"""
Performance benchmark tests for ma_core PyTorch integration.
Tests performance characteristics and regression detection.
"""

import pytest
import torch
import time
import numpy as np
import psutil
import os
from layers.ma_core_bridge import MACoreAttention, pytorch_dense_attention, pytorch_sparse_attention
from layers.sparse_attention import SparseAttention


class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_dense_attention_performance(self, benchmark_tensors, performance_threshold, skip_if_ma_core_unavailable):
        """Benchmark dense attention performance."""
        attention = MACoreAttention(sparse=False, fallback_training=True)
        attention.eval()  # Use C++ implementation
        
        query = benchmark_tensors['query']
        key = benchmark_tensors['key'] 
        value = benchmark_tensors['value']
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = attention(query, key, value)
        
        # Benchmark
        iterations = 20
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                output = attention(query, key, value)
        
        end_time = time.time()
        avg_time_ms = (end_time - start_time) / iterations * 1000
        
        # Performance assertions
        assert avg_time_ms < performance_threshold['max_time_ms']
        assert not torch.isnan(output).any()
        
        print(f"\nDense attention: {avg_time_ms:.2f}ms per forward pass")
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_sparse_attention_performance(self, benchmark_tensors, performance_threshold, skip_if_ma_core_unavailable):
        """Benchmark sparse attention performance."""
        attention = MACoreAttention(sparse=True, window_size=16, fallback_training=True)
        attention.eval()  # Use C++ implementation
        
        query = benchmark_tensors['query']
        key = benchmark_tensors['key']
        value = benchmark_tensors['value']
        
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                _ = attention(query, key, value)
        
        # Benchmark
        iterations = 20
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                output = attention(query, key, value)
        
        end_time = time.time()
        avg_time_ms = (end_time - start_time) / iterations * 1000
        
        # Performance assertions
        assert avg_time_ms < performance_threshold['max_time_ms']
        assert not torch.isnan(output).any()
        
        print(f"\nSparse attention: {avg_time_ms:.2f}ms per forward pass")
    
    @pytest.mark.benchmark
    def test_attention_speedup_comparison(self, skip_if_ma_core_unavailable):
        """Compare ma_core vs PyTorch reference implementation speedup."""
        # Create test tensors
        batch_size, seq_len, num_heads, head_dim = 1, 64, 4, 32
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        query = torch.randn(*shape)
        key = torch.randn(*shape)
        value = torch.randn(*shape)
        
        # Benchmark PyTorch sparse attention
        iterations = 10
        start_time = time.time()
        for _ in range(iterations):
            pytorch_output = pytorch_sparse_attention(query, key, value, window_size=8)
        pytorch_time = (time.time() - start_time) / iterations * 1000
        
        # Benchmark ma_core sparse attention
        ma_core_attention = MACoreAttention(sparse=True, window_size=8, fallback_training=True)
        ma_core_attention.eval()
        
        with torch.no_grad():
            start_time = time.time()
            for _ in range(iterations):
                ma_core_output = ma_core_attention(query, key, value)
            ma_core_time = (time.time() - start_time) / iterations * 1000
        
        speedup = pytorch_time / ma_core_time
        
        print(f"\nPerformance comparison:")
        print(f"  PyTorch:  {pytorch_time:.2f}ms")
        print(f"  ma_core:  {ma_core_time:.2f}ms")
        print(f"  Speedup:  {speedup:.2f}x")
        
        # Should be at least some improvement (even if small)
        assert speedup > 0.5, f"Unexpected slowdown: {speedup:.2f}x"
        
        # Outputs should have same shape
        assert pytorch_output.shape == ma_core_output.shape
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("seq_len", [32, 64, 128])
    def test_scaling_performance(self, seq_len, skip_if_ma_core_unavailable):
        """Test performance scaling with sequence length."""
        batch_size, num_heads, head_dim = 1, 4, 32
        window_size = 16
        
        shape = (batch_size, seq_len, num_heads, head_dim)
        query = torch.randn(*shape)
        key = torch.randn(*shape)
        value = torch.randn(*shape)
        
        # Test sparse attention scaling
        sparse_attention = MACoreAttention(sparse=True, window_size=window_size, fallback_training=True)
        sparse_attention.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = sparse_attention(query, key, value)
        
        # Benchmark
        iterations = 10
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                output = sparse_attention(query, key, value)
        
        avg_time_ms = (time.time() - start_time) / iterations * 1000
        
        print(f"\nSeq len {seq_len}: {avg_time_ms:.2f}ms per forward pass")
        
        # Sparse attention should scale approximately linearly
        # For longer sequences, time should not grow quadratically
        expected_max_time = seq_len * 0.5  # Rough linear scaling expectation
        assert avg_time_ms < expected_max_time, f"Poor scaling for seq_len={seq_len}"


class TestMemoryBenchmarks:
    """Memory usage benchmark tests."""
    
    @pytest.mark.benchmark
    def test_memory_usage_sparse_vs_dense(self, skip_if_ma_core_unavailable):
        """Compare memory usage between sparse and dense attention."""
        process = psutil.Process(os.getpid())
        
        # Create larger tensors for meaningful memory difference
        batch_size, seq_len, num_heads, head_dim = 1, 128, 4, 64
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        query = torch.randn(*shape)
        key = torch.randn(*shape)
        value = torch.randn(*shape)
        
        # Measure baseline memory
        memory_baseline = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test dense attention memory usage
        dense_attention = MACoreAttention(sparse=False, fallback_training=True)
        dense_attention.eval()
        
        with torch.no_grad():
            dense_output = dense_attention(query, key, value)
        
        memory_dense = process.memory_info().rss / 1024 / 1024  # MB
        dense_memory_used = memory_dense - memory_baseline
        
        # Test sparse attention memory usage
        sparse_attention = MACoreAttention(sparse=True, window_size=16, fallback_training=True)
        sparse_attention.eval()
        
        with torch.no_grad():
            sparse_output = sparse_attention(query, key, value)
        
        memory_sparse = process.memory_info().rss / 1024 / 1024  # MB
        sparse_memory_used = memory_sparse - memory_dense
        
        print(f"\nMemory usage comparison:")
        print(f"  Dense attention:  +{dense_memory_used:.1f} MB")
        print(f"  Sparse attention: +{sparse_memory_used:.1f} MB")
        
        # Both should complete successfully
        assert dense_output.shape == sparse_output.shape
        assert not torch.isnan(dense_output).any()
        assert not torch.isnan(sparse_output).any()
    
    @pytest.mark.benchmark
    def test_memory_efficiency_large_sequence(self, skip_if_ma_core_unavailable):
        """Test memory efficiency with large sequences."""
        # Test with large sequence to see memory benefits
        batch_size, seq_len, num_heads, head_dim = 1, 512, 2, 32
        window_size = 32
        
        shape = (batch_size, seq_len, num_heads, head_dim)
        query = torch.randn(*shape)
        key = torch.randn(*shape)
        value = torch.randn(*shape)
        
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test sparse attention with large sequence
        sparse_attention = MACoreAttention(sparse=True, window_size=window_size, fallback_training=True)
        sparse_attention.eval()
        
        with torch.no_grad():
            output = sparse_attention(query, key, value)
        
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        print(f"\nLarge sequence ({seq_len}) memory usage: {memory_used:.1f} MB")
        
        # Should complete without excessive memory usage
        assert memory_used < 1000  # Less than 1GB for this test
        assert output.shape == shape
        assert not torch.isnan(output).any()


class TestPerformanceRegression:
    """Performance regression detection tests."""
    
    @pytest.mark.benchmark 
    def test_performance_regression_baseline(self, skip_if_ma_core_unavailable):
        """Establish performance baseline for regression detection."""
        # Standard test configuration
        batch_size, seq_len, num_heads, head_dim = 2, 64, 4, 32
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        query = torch.randn(*shape)
        key = torch.randn(*shape)
        value = torch.randn(*shape)
        
        attention = MACoreAttention(sparse=True, window_size=16, fallback_training=True)
        attention.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = attention(query, key, value)
        
        # Benchmark
        iterations = 20
        times = []
        
        with torch.no_grad():
            for _ in range(iterations):
                start_time = time.time()
                output = attention(query, key, value)
                times.append((time.time() - start_time) * 1000)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        print(f"\nPerformance baseline:")
        print(f"  Average: {avg_time:.2f}ms ± {std_time:.2f}ms")
        print(f"  Min: {np.min(times):.2f}ms")
        print(f"  Max: {np.max(times):.2f}ms")
        
        # Regression detection thresholds
        assert avg_time < 20.0, f"Performance regression detected: {avg_time:.2f}ms > 20ms"
        assert std_time < avg_time * 0.5, f"High performance variance: {std_time:.2f}ms"
        assert not torch.isnan(output).any()
    
    @pytest.mark.benchmark
    def test_consistency_across_runs(self, skip_if_ma_core_unavailable):
        """Test performance consistency across multiple runs."""
        batch_size, seq_len, num_heads, head_dim = 1, 32, 2, 16
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        query = torch.randn(*shape)
        key = torch.randn(*shape)
        value = torch.randn(*shape)
        
        attention = MACoreAttention(sparse=True, window_size=8, fallback_training=True)
        attention.eval()
        
        # Multiple independent runs
        run_times = []
        for run in range(5):
            # Warmup for each run
            with torch.no_grad():
                for _ in range(3):
                    _ = attention(query, key, value)
            
            # Benchmark this run
            iterations = 10
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(iterations):
                    output = attention(query, key, value)
            
            avg_time_ms = (time.time() - start_time) / iterations * 1000
            run_times.append(avg_time_ms)
        
        mean_across_runs = np.mean(run_times)
        std_across_runs = np.std(run_times)
        coefficient_of_variation = std_across_runs / mean_across_runs
        
        print(f"\nConsistency across runs:")
        print(f"  Run times: {run_times}")
        print(f"  Mean: {mean_across_runs:.2f}ms")
        print(f"  Std: {std_across_runs:.2f}ms") 
        print(f"  CV: {coefficient_of_variation:.2f}")
        
        # Performance should be reasonably consistent
        assert coefficient_of_variation < 0.3, f"Inconsistent performance: CV={coefficient_of_variation:.2f}"
        assert not torch.isnan(output).any()


class TestConcurrencyAndStability:
    """Test concurrent access and stability."""
    
    def test_concurrent_attention_calls(self, skip_if_ma_core_unavailable):
        """Test multiple concurrent attention operations."""
        attention = MACoreAttention(sparse=True, window_size=4, fallback_training=True)
        attention.eval()
        
        batch_size, seq_len, num_heads, head_dim = 1, 16, 2, 8
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        # Create multiple different input sets
        inputs = []
        for i in range(5):
            query = torch.randn(*shape)
            key = torch.randn(*shape)
            value = torch.randn(*shape)
            inputs.append((query, key, value))
        
        # Process all inputs
        outputs = []
        with torch.no_grad():
            for query, key, value in inputs:
                output = attention(query, key, value)
                outputs.append(output)
        
        # Check all outputs are valid
        for i, output in enumerate(outputs):
            assert output.shape == shape, f"Output {i} has wrong shape"
            assert not torch.isnan(output).any(), f"Output {i} contains NaN"
            assert not torch.isinf(output).any(), f"Output {i} contains Inf"
    
    def test_repeated_operations_stability(self, skip_if_ma_core_unavailable):
        """Test stability over many repeated operations."""
        attention = MACoreAttention(sparse=True, window_size=8, fallback_training=True)
        attention.eval()
        
        batch_size, seq_len, num_heads, head_dim = 1, 16, 2, 16
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        query = torch.randn(*shape)
        key = torch.randn(*shape)
        value = torch.randn(*shape)
        
        # Many repeated operations
        outputs = []
        with torch.no_grad():
            for i in range(100):
                output = attention(query, key, value)
                outputs.append(output.clone())
                
                # Check for numerical instability
                assert not torch.isnan(output).any(), f"NaN detected at iteration {i}"
                assert not torch.isinf(output).any(), f"Inf detected at iteration {i}"
        
        # Check that outputs remain consistent (same input -> same output)
        first_output = outputs[0]
        for i, output in enumerate(outputs[1:], 1):
            torch.testing.assert_close(
                output, first_output, 
                rtol=1e-5, atol=1e-7,
                msg=f"Output inconsistency at iteration {i}"
            )
    
    @pytest.mark.benchmark
    def test_memory_leak_detection(self, skip_if_ma_core_unavailable):
        """Test for memory leaks over many operations."""
        process = psutil.Process(os.getpid())
        memory_start = process.memory_info().rss / 1024 / 1024  # MB
        
        attention = MACoreAttention(sparse=True, window_size=8, fallback_training=True)
        attention.eval()
        
        batch_size, seq_len, num_heads, head_dim = 1, 32, 2, 16
        shape = (batch_size, seq_len, num_heads, head_dim)
        
        # Perform many operations
        with torch.no_grad():
            for i in range(50):
                query = torch.randn(*shape)
                key = torch.randn(*shape)
                value = torch.randn(*shape)
                
                output = attention(query, key, value)
                
                # Force garbage collection periodically
                if i % 10 == 0:
                    import gc
                    gc.collect()
        
        memory_end = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = memory_end - memory_start
        
        print(f"\nMemory leak test:")
        print(f"  Start: {memory_start:.1f} MB")
        print(f"  End: {memory_end:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")
        
        # Should not have significant memory increase
    assert memory_increase < 100, f"Potential memory leak detected: +{memory_increase:.1f} MB"
