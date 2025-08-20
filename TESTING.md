# MA Transformer Testing Framework

This document describes the comprehensive testing framework for the MA Transformer project, which includes both C++ and Python components.

## Overview

The testing framework consists of several layers:
1. **C++ Unit Tests**: Native C++ tests using Google Test framework
2. **Python Extension Tests**: Python tests for the C++ extension interface
3. **Integration Tests**: End-to-end tests ensuring C++ and Python work together
4. **Performance Tests**: Benchmarks for performance regression detection
5. **Automated Test Runner**: Unified test execution with reporting

## Test Structure

```
tests/
├── cpp/                           # C++ native tests
│   ├── test_ma_core.cpp           # Core functionality tests
│   ├── CMakeLists.txt             # CMake build configuration
│   └── Makefile                   # Make build configuration
├── python/                        # Python extension tests
│   ├── test_ma_core_extension.py  # Original functionality tests
│   ├── test_tensor_functionality.py # Extended tensor & sparse tests
│   └── __init__.py
└── __init__.py

run_tests.py                       # Automated test runner
test_cpp_extension.py              # Original integration test  
test_new_functionality.py          # Extended functionality demo/test
```

## Running Tests

### Quick Test Run
```bash
# Run Python and integration tests (recommended for development)
python run_tests.py

# Run specific test types
python run_tests.py --types python
python run_tests.py --types cpp
python run_tests.py --types integration
python run_tests.py --types performance

# Run all tests
python run_tests.py --types all
```

### Individual Test Components

#### C++ Tests
```bash
cd tests/cpp
make clean && make        # Build tests
make test                 # Run tests
./ma_core_tests          # Direct execution
```

#### Python Tests
```bash
python tests/python/test_ma_core_extension.py
```

#### Integration Tests
```bash
python test_cpp_extension.py
```

## Test Categories

### 1. C++ Unit Tests (`tests/cpp/test_ma_core.cpp`)

Tests the core C++ functionality directly:

- **Basic Functionality**: Positive numbers, negative numbers, zero
- **Mathematical Properties**: Commutative, associative, identity
- **Boundary Conditions**: Integer limits, edge cases
- **Performance**: Timing tests for regression detection

**Features:**
- Uses Google Test framework
- Test fixtures for setup/teardown
- Comprehensive edge case coverage
- Performance benchmarks

### 2. Python Extension Tests

**Original Tests** (`tests/python/test_ma_core_extension.py`):
- **Functionality Tests**: Mathematical operations (add function)
- **Type Validation**: Ensures proper type checking
- **Error Handling**: Invalid inputs, argument counts
- **Integration Features**: List comprehensions, map functions
- **Memory Management**: Stability under repeated operations
- **Performance**: Python-side performance testing

**Extended Tests** (`tests/python/test_tensor_functionality.py`):
- **Tensor Operations**: Creation, manipulation, shape validation
- **Sparse Patterns**: Sliding window attention patterns
- **Device Support**: CPU device handling, enum validation
- **Performance Regression**: Tensor and pattern creation benchmarks
- **Edge Cases**: Invalid shapes, boundary conditions
- **Integration Regression**: Backward compatibility verification

**Test Classes:**
- `TestMaCoreExtension`: Core functionality tests
- `TestMaCoreIntegration`: Python integration features  
- `TestMaCoreRegression`: Regression prevention tests
- `TestTensorFunctionality`: Tensor operations and validation
- `TestSparsePatterns`: Sparse attention pattern testing
- `TestDeviceEnumeration`: Device support validation
- `TestPerformanceRegression`: Performance benchmarks
- `TestIntegrationRegression`: Backward compatibility
- `TestExtendedErrorHandling`: Comprehensive error scenarios

### 3. Integration Tests

- **Original Test**: `test_cpp_extension.py` - Basic smoke test
- **End-to-End**: Complete workflow testing

### 4. Performance Tests

Built into both C++ and Python test suites:
- C++: 1M operations timing test
- Python: 100K operations with overhead measurement
- Regression detection for performance degradation

## Current Test Status

### Resolved Issues
The testing framework identified and resolved key limitations:

1. **✅ Integer Overflow**: Fixed by upgrading from `int` to `int64_t` 
2. **✅ Type Conversion**: Large Python integers now properly convert to C++ int64_t
3. **✅ Extended Functionality**: Added tensor operations and sparse pattern support

### Test Results Summary
- ✅ **C++ Native Tests**: 7/7 tests pass
- ✅ **Python Extension Tests**: 40/40 tests pass (all overflow issues resolved)
- ✅ **Integration Tests**: All pass  
- ✅ **Performance Tests**: All pass
- ✅ **Extended Functionality**: All tensor and sparse pattern tests pass

### Test Coverage
- **Basic Operations**: Addition with full int64 range support
- **Tensor Operations**: Creation, manipulation, shape validation, device support
- **Sparse Patterns**: Sliding window attention with various sparsity levels
- **Error Handling**: Comprehensive validation and informative error messages
- **Performance**: Sub-millisecond tensor creation and pattern generation
- **Memory Management**: Stable under repeated operations
- **Backward Compatibility**: All original functionality preserved

## Adding New Tests

### Adding C++ Tests

1. Add test functions to `tests/cpp/test_ma_core.cpp`
2. Use the `MaCoreTest` fixture
3. Follow Google Test conventions:
   ```cpp
   TEST_F(MaCoreTest, TestNewFeature) {
       EXPECT_EQ(ma_core::new_function(input), expected);
   }
   ```

### Adding Python Tests

1. Add test methods to appropriate class in `tests/python/test_ma_core_extension.py`
2. Use `unittest` framework conventions:
   ```python
   def test_new_feature(self):
       self.assertEqual(ma_core.new_function(input), expected)
   ```

### Adding Performance Tests

1. C++: Add to `test_ma_core.cpp` with timing measurements
2. Python: Add to test class with time.time() measurements
3. Set reasonable performance thresholds

## Continuous Integration

The test framework is designed for CI/CD integration:

- **Exit Codes**: Non-zero exit code on test failures
- **Verbose Output**: Detailed reporting for debugging
- **Timeout Protection**: Tests have reasonable time limits
- **Parallel Execution**: Tests can run concurrently where possible

## Test Data and Fixtures

### C++ Test Fixtures
- `MaCoreTest` class provides setup/teardown
- Reusable test data patterns
- Performance timing utilities

### Python Test Fixtures
- `setUp()` and `tearDown()` methods
- Garbage collection management
- Comprehensive error message validation

## Dependencies

### C++ Testing
- Google Test (googletest) - Install via `brew install googletest`
- CMake (optional) - For alternative build system
- Make - For primary build system

### Python Testing
- Built-in `unittest` framework
- No additional dependencies required

## Best Practices

1. **Test Isolation**: Each test should be independent
2. **Descriptive Names**: Test names should describe what they test
3. **Edge Cases**: Always test boundary conditions
4. **Performance**: Include performance regression tests
5. **Documentation**: Document complex test scenarios
6. **Error Messages**: Provide clear failure messages

## Troubleshooting

### Common Issues

1. **Build Failures**: Ensure Google Test is installed and pybind11 is available
2. **Import Errors**: Make sure the extension is built (`pip install -e .`)
3. **Path Issues**: Run tests from project root directory
4. **Performance Failures**: May indicate system load or regression

### Debug Mode

For debugging test failures:
```bash
# Verbose output
python run_tests.py --verbose

# Individual test debugging
python -m pytest tests/python/test_ma_core_extension.py::TestMaCoreExtension::test_specific_test -v
```

## Future Enhancements

1. **Extended C++ Features**: As more functionality is added, expand test coverage
2. **Memory Testing**: Add valgrind or similar memory testing
3. **Cross-Platform**: Test on different operating systems
4. **Coverage Reports**: Add code coverage analysis
5. **Stress Testing**: Add longer-running stress tests
6. **Property-Based Testing**: Add hypothesis-based testing

## Contributing

When adding new features:
1. Add corresponding tests before implementing
2. Ensure all existing tests still pass
3. Add performance benchmarks for critical paths
4. Update this documentation as needed

The testing framework is designed to grow with the project while maintaining reliability and catching regressions early.