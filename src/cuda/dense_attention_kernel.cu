// SPDX-License-Identifier: Apache-2.0
// This file is intentionally minimal. The CUDA kernels and bindings
// are implemented in sparse_attention_kernel.cu to avoid multiple
// TORCH_EXTENSION_NAME module definitions during build.

// Keeping this file as a placeholder ensures setup.py can list it
// while we centralize the module in a single translation unit.

// Potential future split: move dense-specific kernels here and include
// a header that provides a single PYBIND11_MODULE in one .cu.
