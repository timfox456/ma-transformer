"""
pytest configuration and fixtures for integration tests.
Provides common test utilities for PyTorch bridge testing.
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    import ma_core
    from layers.ma_core_bridge import MACoreAttention, pytorch_dense_attention, pytorch_sparse_attention
    from layers.sparse_attention import SparseAttention
    MA_CORE_AVAILABLE = True
except ImportError:
    MA_CORE_AVAILABLE = False


@pytest.fixture
def skip_if_ma_core_unavailable():
    """Skip test if ma_core C++ extension is not available."""
    if not MA_CORE_AVAILABLE:
        pytest.skip("ma_core C++ extension not available")


@pytest.fixture(params=[(1, 4, 2, 8), (2, 8, 4, 16), (1, 16, 8, 32)])
def tensor_shapes(request):
    """Parametrized fixture for different tensor shapes."""
    batch_size, seq_len, num_heads, head_dim = request.param
    return {
        'batch_size': batch_size,
        'seq_len': seq_len, 
        'num_heads': num_heads,
        'head_dim': head_dim,
        'shape': (batch_size, seq_len, num_heads, head_dim)
    }


@pytest.fixture
def random_tensors(tensor_shapes):
    """Create random query, key, value tensors for testing."""
    shape = tensor_shapes['shape']
    return {
        'query': torch.randn(*shape, requires_grad=True),
        'key': torch.randn(*shape, requires_grad=True),
        'value': torch.randn(*shape, requires_grad=True),
        'shape_info': tensor_shapes
    }


@pytest.fixture
def small_tensors():
    """Create small tensors for quick tests."""
    batch_size, seq_len, num_heads, head_dim = 1, 4, 2, 8
    shape = (batch_size, seq_len, num_heads, head_dim)
    return {
        'query': torch.randn(*shape, requires_grad=True),
        'key': torch.randn(*shape, requires_grad=True), 
        'value': torch.randn(*shape, requires_grad=True),
        'shape': shape
    }


@pytest.fixture(params=[2, 4, 8, 16])
def window_sizes(request):
    """Parametrized fixture for different window sizes."""
    return request.param


@pytest.fixture(params=[True, False])
def causal_mask_options(request):
    """Parametrized fixture for causal mask testing."""
    return request.param


@pytest.fixture
def attention_modules():
    """Create different attention module configurations."""
    return {
        'dense': MACoreAttention(sparse=False, fallback_training=True),
        'sparse_small': MACoreAttention(sparse=True, window_size=4, fallback_training=True),
        'sparse_medium': MACoreAttention(sparse=True, window_size=8, fallback_training=True),
        'sparse_large': MACoreAttention(sparse=True, window_size=16, fallback_training=True),
        'dense_causal': MACoreAttention(sparse=False, use_causal_mask=True, fallback_training=True)
    }


@pytest.fixture
def benchmark_tensors():
    """Create larger tensors for performance benchmarking."""
    batch_size, seq_len, num_heads, head_dim = 2, 32, 8, 64
    shape = (batch_size, seq_len, num_heads, head_dim)
    return {
        'query': torch.randn(*shape),
        'key': torch.randn(*shape),
        'value': torch.randn(*shape),
        'shape': shape
    }


@pytest.fixture
def identical_tensors():
    """Create identical tensors for numerical testing."""
    torch.manual_seed(42)
    np.random.seed(42)
    
    batch_size, seq_len, num_heads, head_dim = 1, 8, 2, 16
    shape = (batch_size, seq_len, num_heads, head_dim)
    
    # Create identical tensors
    query = torch.randn(*shape)
    key = query.clone()
    value = torch.ones(*shape) * 2.0
    
    return {
        'query': query,
        'key': key, 
        'value': value,
        'shape': shape
    }


@pytest.fixture(scope="session")
def performance_threshold():
    """Performance thresholds for benchmarking tests."""
    return {
        'min_speedup': 0.8,  # Minimum expected speedup ratio
        'max_time_ms': 50.0,  # Maximum acceptable time per operation
        'memory_tolerance': 0.1  # Memory usage tolerance
    }


class TestHelper:
    """Helper class with common test utilities."""
    
    @staticmethod
    def assert_tensor_shape(tensor, expected_shape):
        """Assert tensor has expected shape."""
        assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, got {tensor.shape}"
    
    @staticmethod
    def assert_tensors_close(tensor1, tensor2, rtol=1e-4, atol=1e-6):
        """Assert two tensors are numerically close."""
        torch.testing.assert_close(tensor1, tensor2, rtol=rtol, atol=atol)
    
    @staticmethod
    def assert_gradients_exist(tensors):
        """Assert that gradients exist for all tensors."""
        for name, tensor in tensors.items():
            if tensor.requires_grad:
                assert tensor.grad is not None, f"Gradient not found for {name}"
    
    @staticmethod
    def benchmark_function(func, *args, iterations=10, warmup=3, **kwargs):
        """Benchmark a function call."""
        import time
        
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # Actual timing
        start_time = time.time()
        for _ in range(iterations):
            result = func(*args, **kwargs)
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) / iterations * 1000
        return result, avg_time_ms
    
    @staticmethod
    def create_mismatched_tensors():
        """Create tensors with mismatched dimensions for error testing."""
        return {
            'query_normal': torch.randn(1, 4, 2, 8),
            'key_seq_mismatch': torch.randn(1, 6, 2, 8),  # Different seq_len
            'value_head_mismatch': torch.randn(1, 4, 2, 16),  # Different head_dim
            'key_batch_mismatch': torch.randn(2, 4, 2, 8),  # Different batch_size
        }


@pytest.fixture
def test_helper():
    """Provide TestHelper instance."""
    return TestHelper()


# Performance markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "benchmark: marks tests as performance benchmarks")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gradient: marks tests that verify gradient flow")
    config.addinivalue_line("markers", "error: marks tests that verify error handling")


# Custom pytest collection hook
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add integration marker to all tests in integration/ directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add slow marker to benchmark tests
        if "benchmark" in item.name.lower():
            item.add_marker(pytest.mark.benchmark)
            item.add_marker(pytest.mark.slow)
        
        # Add gradient marker to gradient tests
        if "gradient" in item.name.lower():
            item.add_marker(pytest.mark.gradient)
        
        # Add error marker to error tests
        if "error" in item.name.lower():
            item.add_marker(pytest.mark.error)