# AGENTS.md: AI Agent Directives for `ma-transformer` Project

This document aligns AI agent contributions with the current state of the repository and the design notes in `.gemini/GEMINI.md`. It explains what exists today, the intended direction, and how to contribute safely and effectively.

### I. Project Context and Current Objectives

- **Project Name:** `ma-transformer`
- **Problem Domain:** High-frequency trading (HFT) on market microstructure data (tick-level sequences).
- **Core Goal (current):** Provide a high-performance attention engine (dense and sparse) via a custom C++ backend (`ma_core`) integrated with PyTorch, with correct training support and low-latency inference. CUDA acceleration is desired but currently partial.
- **Key Differentiator (today):** A custom C++17 attention engine compiled as a Python extension via `pybind11` (`ma_core`), offering dense and sliding-window sparse attention with an extensible factory. A PyTorch bridge (`MACoreAttention`) routes inference to C++ while using PyTorch for gradients.
- **Target Performance (aspirational):** Sub-millisecond end-to-end inference. Current bottlenecks include bridge tensor conversions and absence of a CUDA path in critical kernels.
- **Core Technologies:** C++17, `pybind11`, PyTorch; optional backends enumerated in code (`CPU`, `MPS`, `CUDA`, `ROCm`). CUDA kernels exist as stubs under `src/cuda/`.

### II. Repository Structure (authoritative)

- `src/csrc/`: C++ attention engine (ma_core)
  - `tensor.hpp/.cpp`: Lightweight tensor and shapes; device enum.
  - `dense_attention.cpp`, `sparse_attention.cpp`: Core attention implementations.
  - `attention_interface.hpp`, `attention_types.hpp`, `attention_factory.cpp`: Base classes, configs, and factory (`create_attention`).
  - `ma_core.cpp/.hpp`, `main.cpp`: Public API and `pybind11` bindings (module: `ma_core`).
- `src/layers/`:
  - `ma_core_bridge.py`: `MACoreAttentionFunction` and `MACoreAttention` bridging PyTorch tensors to `ma_core` and falling back to PyTorch for backward.
  - `sparse_attention.py`: High-level wrapper that prefers `ma_core`, otherwise PyTorch.
- `src/cuda/`: CUDA extension stubs (e.g., `sparse_attention_kernel.cu`), not yet wired into the training/inference path.
- `src/model.py`, `src/data_loader.py`, `src/train.py`: Example model, data pipeline, and training script.
- `tests/`:
  - `cpp/`: GoogleTest C++ tests for `ma_core` components.
  - `python/`: `unittest` for the Python bindings and tensor utilities.
  - `integration/`: `pytest` integration for PyTorch bridge and end-to-end behavior.
- Project tooling: `setup.py`/`pyproject.toml`, `run_tests.py`, `pytest.ini`, and example/benchmark scripts.

### III. Current Capabilities and Gaps

- **Inference path:** `MACoreAttention` forwards to `ma_core` for dense/sparse attention; correct shapes and device enums are validated via bindings.
- **Training path:** Backward currently falls back to PyTorch reference implementations in the bridge for gradient correctness.
- **Tensor conversion:** `pytorch_to_ma_core` and `ma_core_to_pytorch` use placeholder logic (no zero-copy; missing setters/getters on `ma_core::Tensor`). This is a known performance bottleneck.
- **CUDA:** `src/cuda/` contains initial stubs; the production path relies on the C++ engine (`pybind11`), not a `torch.utils.cpp_extension` module.
- **Device support:** Code enumerates `CPU`, `MPS`, `CUDA`, `ROCm`; real paths implemented focus on CPU/MPS-compatible behavior through `pybind11` and PyTorch.

### IV. High-Impact Focus Areas for AI Assistance

1. **Bridge Efficiency:**
   - Add data access methods to `ma_core::Tensor` (read/write, contiguous storage view) and implement efficient copy or zero-copy pathways in `ma_core_bridge.py`.
   - Avoid Python-side element loops; prefer bulk memory transfers and shape-aware views.

2. **Backward in C++:**
   - Design and implement backward kernels (dense and sparse) in `ma_core` with consistent numerics to PyTorch.
   - Expose via `pybind11` and update `MACoreAttentionFunction.backward` to call C++ when available.

3. **Sparse Patterns and Factory:**
   - Extend `sparse_attention` with additional patterns (block-sparse, dilated, causal windowing) via `attention_factory.cpp` while maintaining memory predictability.

4. **CUDA Acceleration Path:**
   - Flesh out `src/cuda/` kernels for attention paths and integrate them under a unified API (either through `ma_core` device dispatch or a separate Torch extension), with robust error checking and streams.

5. **Testing and Benchmarks:**
   - Expand integration tests to validate parity between PyTorch and `ma_core` outputs/gradients across shapes/devices.
   - Add microbenchmarks to detect regressions in copy/conversion and attention throughput.

### V. Directives for Contributions

- **Performance First:** Minimize data movement between Python and C++; coalesce memory accesses; consider cache locality. If implementing CUDA, ensure coalesced global memory, appropriate shared memory use, and occupancy-aware block sizing.
- **API Stability:** Extend `ma_core` via clear headers (`ma_core.hpp`, `attention_interface.hpp`) and bindings in `main.cpp`. Keep Python bridge signatures stable; add feature flags where needed.
- **Asynchrony:** Prefer async execution (`cudaStream_t`) and non-blocking transfers when adding CUDA paths. Keep Python-side synchronization minimal and explicit.
- **Error Handling:** Add robust checks in both C++ and Python (shape/device/dtype assertions; CUDA error checks). Surface actionable error messages through `pybind11` exceptions.
- **Modularity & Tests:** Keep new features modular. Provide or update tests under `tests/{cpp,python,integration}`. Targeted unit tests are preferred over broad rewrites.

### VI. Task-Specific Guidance

- **A. C++/CUDA Attention Work:**
  - Define kernels/APIs with explicit tensor shapes: `[batch, seq, heads, dim]`.
  - For sparse attention, prefer sliding-window/block-sparse indices with predictable memory footprints; document memory use via helpers (e.g., `SparseTensor::nnz`).
  - Provide `CUDA_CHECK`-style macros and optional stream parameters in host wrappers.

- **B. PyTorch Bridge Updates:**
  - Maintain `MACoreAttentionFunction` interface. When adding C++ backward, gate by availability and add test coverage for gradients (e.g., `torch.autograd.gradcheck`).
  - Implement efficient conversions by exposing contiguous buffers from `ma_core::Tensor` and binding them; avoid Python loops.

- **C. Build Integration:**
  - Keep `pybind11`-based module `ma_core` as the primary entry point. If adding Torch CUDA extensions, update `setup.py` to conditionally build them and document optional dependencies.

- **D. Performance Analysis:**
  - Quantify changes with microbenchmarks (latency/throughput, memory bandwidth). Highlight expected wins (e.g., “remove per-element Python loop, expect >10x copy speedup”).
  - Call out specific bottlenecks (copy-in/copy-out, softmax normalization, pattern generation) and propose code-level fixes.

### VII. Dependencies and Tooling

- **C++/Bindings:** C++17, `pybind11`.
- **Python/ML:** PyTorch.
- **Testing:** GoogleTest (`tests/cpp`), `unittest` (`tests/python`), `pytest` (`tests/integration`).
- **Build:** `setuptools` for `ma_core` extension; `Makefile`/`CMakeLists.txt` for C++ tests.

### VIII. How to Validate Changes

- Build the extension: `pip install -e .`
- Run tests: `python run_tests.py` or targeted suites under `tests/`.
- For CUDA additions, include smoke tests that skip gracefully if CUDA is unavailable.

### IX. Interaction Protocol

- **Explicit Instructions:** Prefer specific tasks (e.g., “implement C++ backward for dense attention”).
- **Iterative Refinement:** Expect follow-ups from maintainers; keep changes scoped and explain performance rationale briefly in PRs.
- **Context Awareness:** Refer to `README.md`, `.gemini/GEMINI.md`, and code in `src/csrc/` and `src/layers/` before proposing architectural changes.

By following these directives, AI agents can extend the engine, improve latency, and move the project toward robust, GPU-accelerated production readiness while preserving correctness and testability.
