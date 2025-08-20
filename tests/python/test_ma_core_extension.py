#!/usr/bin/env python3
"""
Comprehensive Python unit tests for the ma_core C++ extension.
Tests both functionality and integration between Python and C++.
"""

import unittest
import sys
import os
import time
import gc
from typing import Any, List, Tuple

# Add the project root to the path to import ma_core
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    import ma_core
except ImportError as e:
    raise ImportError(f"Failed to import ma_core extension. Make sure it's built. Error: {e}")


class TestMaCoreExtension(unittest.TestCase):
    """Test suite for the ma_core C++ extension."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        pass
    
    def tearDown(self):
        """Clean up after each test method."""
        gc.collect()  # Force garbage collection
    
    # Basic functionality tests
    def test_add_basic(self):
        """Test basic addition functionality."""
        self.assertEqual(ma_core.add(1, 2), 3)
        self.assertEqual(ma_core.add(0, 0), 0)
        self.assertEqual(ma_core.add(-1, 1), 0)
        self.assertEqual(ma_core.add(100, 200), 300)
    
    def test_add_negative_numbers(self):
        """Test addition with negative numbers."""
        self.assertEqual(ma_core.add(-5, -3), -8)
        self.assertEqual(ma_core.add(-10, 5), -5)
        self.assertEqual(ma_core.add(10, -15), -5)
    
    def test_add_large_numbers(self):
        """Test addition with large numbers."""
        large_num = 2**30
        self.assertEqual(ma_core.add(large_num, 1), large_num + 1)
        self.assertEqual(ma_core.add(large_num, large_num), 2 * large_num)
    
    # Type checking tests
    def test_add_type_validation(self):
        """Test that the function validates input types correctly."""
        with self.assertRaises(TypeError):
            ma_core.add("1", 2)
        
        with self.assertRaises(TypeError):
            ma_core.add(1, "2")
        
        with self.assertRaises(TypeError):
            ma_core.add(1.5, 2)
        
        with self.assertRaises(TypeError):
            ma_core.add(1, 2.5)
    
    def test_add_argument_count(self):
        """Test that the function requires exactly two arguments."""
        with self.assertRaises(TypeError):
            ma_core.add(1)
        
        with self.assertRaises(TypeError):
            ma_core.add(1, 2, 3)
        
        with self.assertRaises(TypeError):
            ma_core.add()
    
    # Mathematical property tests
    def test_add_commutative(self):
        """Test that addition is commutative (a + b = b + a)."""
        test_pairs = [(1, 2), (10, -5), (0, 100), (-7, -3)]
        for a, b in test_pairs:
            with self.subTest(a=a, b=b):
                self.assertEqual(ma_core.add(a, b), ma_core.add(b, a))
    
    def test_add_associative(self):
        """Test that addition is associative ((a + b) + c = a + (b + c))."""
        test_triples = [(1, 2, 3), (10, -5, 7), (0, 100, -50)]
        for a, b, c in test_triples:
            with self.subTest(a=a, b=b, c=c):
                result1 = ma_core.add(ma_core.add(a, b), c)
                result2 = ma_core.add(a, ma_core.add(b, c))
                self.assertEqual(result1, result2)
    
    def test_add_identity(self):
        """Test that 0 is the additive identity."""
        test_numbers = [1, -1, 100, -100, 0]
        for num in test_numbers:
            with self.subTest(num=num):
                self.assertEqual(ma_core.add(num, 0), num)
                self.assertEqual(ma_core.add(0, num), num)
    
    # Performance tests
    def test_add_performance(self):
        """Test that the C++ extension performs well."""
        iterations = 100000
        
        start_time = time.time()
        for i in range(iterations):
            result = ma_core.add(i, i + 1)
        end_time = time.time()
        
        elapsed = end_time - start_time
        # Should complete 100k operations in less than 1 second
        self.assertLess(elapsed, 1.0, f"Performance test took {elapsed:.4f}s for {iterations} operations")
        
        # Verify the last result is correct
        self.assertEqual(result, ma_core.add(iterations - 1, iterations))
    
    def test_add_memory_stability(self):
        """Test that repeated calls don't cause memory leaks."""
        # This test runs many operations to check for memory stability
        iterations = 10000
        test_values = [(1, 2), (100, -50), (0, 0), (-10, 10)]
        
        for _ in range(iterations):
            for a, b in test_values:
                result = ma_core.add(a, b)
                expected = a + b
                self.assertEqual(result, expected)
    
    # Edge case tests
    def test_add_integer_limits(self):
        """Test behavior near integer limits."""
        # Test with values near system limits
        max_int = sys.maxsize
        min_int = -sys.maxsize - 1
        
        # These should work without overflow (depending on implementation)
        self.assertEqual(ma_core.add(max_int, 0), max_int)
        self.assertEqual(ma_core.add(min_int, 0), min_int)
        self.assertEqual(ma_core.add(max_int, -1), max_int - 1)
        self.assertEqual(ma_core.add(min_int, 1), min_int + 1)
    
    # Module introspection tests
    def test_module_docstring(self):
        """Test that the module has proper documentation."""
        self.assertIsNotNone(ma_core.__doc__)
        self.assertIn("MA Transformer Core", ma_core.__doc__)
    
    def test_function_docstring(self):
        """Test that the add function has proper documentation."""
        self.assertIsNotNone(ma_core.add.__doc__)
        self.assertIn("add", ma_core.add.__doc__.lower())
    
    def test_module_attributes(self):
        """Test that the module has expected attributes."""
        self.assertTrue(hasattr(ma_core, 'add'))
        self.assertTrue(callable(ma_core.add))


class TestMaCoreIntegration(unittest.TestCase):
    """Integration tests for ma_core with other Python features."""
    
    def test_add_in_list_comprehension(self):
        """Test using ma_core.add in list comprehensions."""
        numbers = list(range(10))
        results = [ma_core.add(x, x) for x in numbers]
        expected = [2 * x for x in numbers]
        self.assertEqual(results, expected)
    
    def test_add_in_map(self):
        """Test using ma_core.add with map function."""
        pairs = [(1, 2), (3, 4), (5, 6)]
        results = list(map(lambda pair: ma_core.add(pair[0], pair[1]), pairs))
        expected = [3, 7, 11]
        self.assertEqual(results, expected)
    
    def test_add_with_unpacking(self):
        """Test using ma_core.add with argument unpacking."""
        pairs = [(1, 2), (10, 20), (-5, 5)]
        for a, b in pairs:
            with self.subTest(a=a, b=b):
                self.assertEqual(ma_core.add(*[a, b]), a + b)


class TestMaCoreRegression(unittest.TestCase):
    """Regression tests to ensure existing functionality doesn't break."""
    
    def test_basic_regression(self):
        """Test cases that should always work."""
        # These are the exact test cases from the original test
        self.assertEqual(ma_core.add(1, 2), 3)
        
        # Additional regression cases
        regression_cases = [
            (0, 0, 0),
            (1, -1, 0),
            (42, 58, 100),
            (-10, -20, -30),
            (1000000, 2000000, 3000000)
        ]
        
        for a, b, expected in regression_cases:
            with self.subTest(a=a, b=b, expected=expected):
                self.assertEqual(ma_core.add(a, b), expected)


if __name__ == '__main__':
    # Configure test runner
    unittest.main(
        verbosity=2,
        buffer=True,  # Capture stdout/stderr during tests
        failfast=False,  # Continue running tests after failures
    )