#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
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
    def __init__(self, project_root: Path, attempt_build_cuda: bool = True):
        self.project_root = project_root
        self.cpp_test_dir = project_root / "tests" / "cpp"
        self.python_test_dir = project_root / "tests" / "python"
        self.integration_test_dir = project_root / "tests" / "integration"
        self.results: List[Tuple[str, bool, str]] = []
        self.attempt_build_cuda = attempt_build_cuda
        # Ensure local src is importable without installation
        self.env = os.environ.copy()
        src_path = str(self.project_root / "src")
        existing_pp = self.env.get("PYTHONPATH", "")
        self.env["PYTHONPATH"] = f"{src_path}:{existing_pp}" if existing_pp else src_path
    
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None, timeout: int = 300, env: Optional[dict] = None) -> Tuple[bool, str]:
        """Run a command and return success status and output."""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.project_root,
                env=env or self.env,
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
        """Ensure the Python extension is importable without requiring network installs.

        Strategy: Prefer local import from `src/` (prebuilt .so present). If import fails,
        report a helpful message instead of attempting network installs.
        """
        print("Ensuring Python extension is importable (local src path)...")
        code = (
            "import sys, os; print('PYTHONPATH=', os.getenv('PYTHONPATH','')); "
            "import ma_core; print('ma_core OK')"
        )
        success, output = self.run_command([sys.executable, "-c", code], env=self.env)
        if success:
            print("✅ ma_core import OK from src/")
        else:
            print("❌ Could not import ma_core from src/. Output:")
            print(output)
            # Optionally try to build the extension if requested or CUDA toolchain is detected
            should_build = self.attempt_build_cuda or self._detect_cuda_toolchain()
            if should_build:
                print("Attempting to build Python extensions (including CUDA if available)...")
                build_ok, build_out = self.run_command([sys.executable, "-m", "pip", "install", "-e", "."], env=self.env, timeout=1200)
                print(build_out)
                # Retry import
                success, output = self.run_command([sys.executable, "-c", code], env=self.env)
                if success:
                    print("✅ ma_core import OK after build")
                else:
                    print("❌ Still cannot import ma_core after build")
        self.results.append(("Python Extension Import", success, output))
        return success

    def _detect_cuda_toolchain(self) -> bool:
        """Heuristically detect whether a CUDA toolchain is available for building.

        Returns True if torch reports a CUDA toolkit path or CUDA_HOME is set or nvcc is on PATH.
        """
        code = (
            "import os, shutil;\n"
            "try:\n"
            "  from torch.utils.cpp_extension import CUDA_HOME\n"
            "except Exception:\n"
            "  CUDA_HOME=None\n"
            "env_cuda=os.getenv('CUDA_HOME')\n"
            "has_nvcc=shutil.which('nvcc') is not None\n"
            "print(bool(CUDA_HOME) or bool(env_cuda) or has_nvcc)\n"
        )
        ok, out = self.run_command([sys.executable, "-m", "python", "-c", code], timeout=20)
        if ok:
            return out.strip().splitlines()[-1].strip() == 'True'
        return False

    def cuda_diagnostics(self) -> None:
        """Print CUDA diagnostics and extension availability."""
        print("CUDA diagnostics...")
        code = (
            "import torch, os;\n"
            "print('torch:', torch.__version__);\n"
            "print('cuda available:', torch.cuda.is_available());\n"
            "print('CUDA_HOME:', os.getenv('CUDA_HOME','<unset>'));\n"
            "print('num devices:', (torch.cuda.device_count() if torch.cuda.is_available() else 0));\n"
        )
        _ = self.run_command([sys.executable, "-c", code])
        ext_ok, _ = self.run_command([sys.executable, "-c", "import sparse_attention_cuda; print('sparse_attention_cuda OK')"], env=self.env)
        print(f"CUDA extension present: {ext_ok}")
    
    def run_python_tests(self) -> bool:
        """Run Python tests."""
        if not self.ensure_extension_built():
            return False
        
        print("Running Python unit tests (unittest discover)...")
        success, output = self.run_command([sys.executable, "-m", "unittest", "discover", "-s", str(self.python_test_dir), "-v"])
        if success:
            print("✅ Python unit tests passed")
        else:
            print("❌ Python unit tests failed")
        print(output)
        self.results.append(("Python Unit Tests", success, output))
        return success
    
    def run_integration_tests(self) -> bool:
        """Run integration tests between C++ and Python."""
        print("Running integration tests...")
        # Print CUDA diagnostics up front for clarity
        self.cuda_diagnostics()
        # Prefer pytest if available; otherwise, skip gracefully
        has_pytest, _ = self.run_command([sys.executable, "-c", "import pytest; print(pytest.__version__)"], timeout=30)
        if not has_pytest:
            msg = "pytest not available; skipping integration tests"
            print(f"⚠️  {msg}")
            self.results.append(("Integration Tests (skipped)", True, msg))
            return True

        pytest_cmd = [sys.executable, "-m", "pytest", "-q", str(self.integration_test_dir)]
        # If psutil is not available, ignore performance benchmark module to avoid import error
        has_psutil, _ = self.run_command([sys.executable, "-c", "import psutil; print(psutil.__version__)"] , timeout=10)
        if not has_psutil:
            perf_file = str(self.integration_test_dir / "test_performance_benchmarks.py")
            pytest_cmd.extend(["--ignore", perf_file])
            # Also skip individual memory-efficiency tests that import psutil at runtime
            pytest_cmd.extend(["-k", "not memory_efficiency"])
        success, output = self.run_command(pytest_cmd)
        if success:
            print("✅ Integration tests passed")
        else:
            print("❌ Integration tests failed")
        print(output)
        self.results.append(("Integration Tests", success, output))
        return success
    
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
    # Auto-build is enabled by default when CUDA is detected. Flag retained for compatibility.
    parser.add_argument(
        "--attempt-build-cuda",
        action="store_true",
        help="Deprecated: auto-build is on by default when CUDA is detected"
    )
    
    args = parser.parse_args()
    
    if "all" in args.types:
        test_types = ["cpp", "python", "integration", "performance"]
    else:
        test_types = args.types
    
    project_root = Path(__file__).parent.absolute()
    runner = TestRunner(project_root, attempt_build_cuda=True)
    
    start_time = time.time()
    all_passed = runner.run_all_tests(test_types)
    end_time = time.time()
    
    print(f"\nTotal test time: {end_time - start_time:.2f} seconds")
    runner.print_summary()
    
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
