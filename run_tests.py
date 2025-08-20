#!/usr/bin/env python3
"""
Automated test runner for ma_transformer project.
Runs both C++ and Python tests with comprehensive reporting.
"""

import os
import sys
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional


class TestRunner:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.cpp_test_dir = project_root / "tests" / "cpp"
        self.python_test_dir = project_root / "tests" / "python"
        self.results: List[Tuple[str, bool, str]] = []
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, timeout: int = 300) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, f"Error running command: {e}"
    
    def build_cpp_tests(self) -> bool:
        """Build C++ tests using make."""
        print("Building C++ tests...")
        success, output = self.run_command(["make", "clean"], cwd=self.cpp_test_dir)
        success, output = self.run_command(["make"], cwd=self.cpp_test_dir)
        
        if success:
            print("✅ C++ tests built successfully")
        else:
            print("❌ Failed to build C++ tests")
            print(output)
        
        self.results.append(("C++ Build", success, output))
        return success
    
    def run_cpp_tests(self) -> bool:
        """Run C++ tests."""
        if not self.build_cpp_tests():
            return False
        
        print("Running C++ tests...")
        success, output = self.run_command(["make", "test"], cwd=self.cpp_test_dir)
        
        if success:
            print("✅ C++ tests passed")
        else:
            print("❌ C++ tests failed")
        
        print(output)
        self.results.append(("C++ Tests", success, output))
        return success
    
    def ensure_extension_built(self) -> bool:
        """Ensure the Python extension is built."""
        print("Ensuring Python extension is built...")
        
        # First, clean and rebuild
        success, output = self.run_command(["pip", "install", "-e", "."])
        
        if success:
            print("✅ Python extension built successfully")
        else:
            print("❌ Failed to build Python extension")
            print(output)
        
        self.results.append(("Python Extension Build", success, output))
        return success
    
    def run_python_tests(self) -> bool:
        """Run Python tests."""
        if not self.ensure_extension_built():
            return False
        
        print("Running Python tests...")
        
        # Run all Python test files
        test_files = [
            self.python_test_dir / "test_ma_core_extension.py",
            self.python_test_dir / "test_tensor_functionality.py"
        ]
        
        all_success = True
        combined_output = ""
        
        for test_file in test_files:
            if test_file.exists():
                print(f"  Running {test_file.name}...")
                success, output = self.run_command([sys.executable, str(test_file)])
                combined_output += f"\n=== {test_file.name} ===\n{output}\n"
                
                if success:
                    print(f"  ✅ {test_file.name} passed")
                else:
                    print(f"  ❌ {test_file.name} failed")
                    all_success = False
            else:
                print(f"  ⚠️ {test_file.name} not found, skipping")
        
        if all_success:
            print("✅ All Python tests passed")
        else:
            print("❌ Some Python tests failed")
        
        print(combined_output)
        self.results.append(("Python Tests", all_success, combined_output))
        return all_success
    
    def run_integration_tests(self) -> bool:
        """Run integration tests between C++ and Python."""
        print("Running integration tests...")
        
        # Test files for integration
        test_files = [
            self.project_root / "test_cpp_extension.py",
            self.project_root / "test_new_functionality.py"
        ]
        
        all_success = True
        combined_output = ""
        
        for test_file in test_files:
            if test_file.exists():
                print(f"  Running {test_file.name}...")
                success, output = self.run_command([sys.executable, str(test_file)])
                combined_output += f"\n=== {test_file.name} ===\n{output}\n"
                
                if success:
                    print(f"  ✅ {test_file.name} passed")
                else:
                    print(f"  ❌ {test_file.name} failed")
                    all_success = False
            else:
                print(f"  ⚠️ {test_file.name} not found, skipping")
        
        if all_success:
            print("✅ All integration tests passed")
        else:
            print("❌ Some integration tests failed")
        
        print(combined_output)
        self.results.append(("Integration Tests", all_success, combined_output))
        return all_success
    
    def run_performance_tests(self) -> bool:
        """Run performance benchmarks."""
        print("Running performance tests...")
        
        # Simple performance test
        perf_script = f"""
import ma_core
import time

# Warm up
for i in range(1000):
    ma_core.add(i, i+1)

# Benchmark
iterations = 1000000
start = time.time()
for i in range(iterations):
    result = ma_core.add(i, i+1)
end = time.time()

elapsed = end - start
ops_per_sec = iterations / elapsed
print(f"Performance: {{ops_per_sec:.0f}} ops/sec ({{elapsed:.4f}}s for {{iterations}} ops)")
print("✅ Performance test completed")
"""
        
        success, output = self.run_command([sys.executable, "-c", perf_script])
        
        if success:
            print("✅ Performance tests passed")
        else:
            print("❌ Performance tests failed")
        
        print(output)
        self.results.append(("Performance Tests", success, output))
        return success
    
    def print_summary(self):
        """Print a summary of all test results."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for _, success, _ in self.results if success)
        
        for test_name, success, _ in self.results:
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{test_name:<25} {status}")
        
        print("-"*60)
        print(f"Total: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed!")
            return False
    
    def run_all_tests(self, test_types: List[str]) -> bool:
        """Run all specified test types."""
        print("Starting ma_transformer test suite...")
        print(f"Project root: {self.project_root}")
        print(f"Test types: {', '.join(test_types)}")
        print("-"*60)
        
        all_passed = True
        
        if "cpp" in test_types:
            all_passed &= self.run_cpp_tests()
        
        if "python" in test_types:
            all_passed &= self.run_python_tests()
        
        if "integration" in test_types:
            all_passed &= self.run_integration_tests()
        
        if "performance" in test_types:
            all_passed &= self.run_performance_tests()
        
        return all_passed


def main():
    parser = argparse.ArgumentParser(description="Run ma_transformer tests")
    parser.add_argument(
        "--types",
        nargs="+",
        default=["python", "integration"],
        choices=["cpp", "python", "integration", "performance", "all"],
        help="Types of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    if "all" in args.types:
        test_types = ["cpp", "python", "integration", "performance"]
    else:
        test_types = args.types
    
    project_root = Path(__file__).parent.absolute()
    runner = TestRunner(project_root)
    
    start_time = time.time()
    all_passed = runner.run_all_tests(test_types)
    end_time = time.time()
    
    print(f"\nTotal test time: {end_time - start_time:.2f} seconds")
    runner.print_summary()
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()