# AGENTS.md: AI Agent Directives for `ma-transformer` Project

This document provides directives and guidelines for AI agents (e.g., `gemini-cli`, `codex`, automated code generation tools) interacting with the `ma-transformer` project. The goal is to enable AI agents to understand the project's purpose, contribute effectively, and maintain code quality and performance standards.

### **I. Project Context and Core Objectives**

* **Project Name:** `ma-transformer`
* **Primary Goal:** Develop an ultra-low latency, GPU-accelerated deep learning framework for predicting order book imbalances in high-frequency trading (HFT).
* **Key Differentiator:** Extensive use of **custom C++ CUDA kernels** for performance-critical components where standard deep learning frameworks introduce unacceptable latency or do not support required optimizations.
* **Target Performance:** Sub-millisecond inference latency from raw tick data to prediction output.
* **Core Technologies:** NVIDIA CUDA (C++), PyTorch (Python, with C++ extensions).
* **Primary Focus Areas for AI Assistance:**
    1.  **CUDA Kernel Development & Optimization:** Generating, optimizing, and debugging C++ CUDA code.
    2.  **PyTorch C++ Extension Integration:** Bridging custom CUDA kernels with PyTorch.
    3.  **Performance Analysis & Bottleneck Identification:** Interpreting profiling data and suggesting optimizations.
    4.  **Code Generation (Python/CUDA):** Generating boilerplate, test cases, or initial implementations based on high-level descriptions.
    5.  **Documentation & Explanation:** Providing clear explanations of complex concepts or code segments.

### **II. General Directives for AI Agents**

1.  **Prioritize Performance:** When generating or modifying code, especially in `src/cuda/`, the absolute highest priority is low-latency and high-throughput execution on NVIDIA GPUs. Assume memory bandwidth and kernel launch overhead are critical bottlenecks.
2.  **CUDA Idioms & Best Practices:** Adhere strictly to NVIDIA CUDA C++ best practices:
    * **Memory Coalescing:** Ensure global memory accesses are coalesced.
    * **Shared Memory:** Utilize shared memory for on-chip caching and reducing global memory traffic where applicable.
    * **Thread Synchronization:** Use `__syncthreads()` judiciously.
    * **Occupancy:** Consider factors affecting GPU occupancy (register usage, shared memory, block size).
    * **Error Handling:** Include robust CUDA error checking in kernel wrappers and host-side code.
    * **Asynchronous Operations:** Favor `cudaStream_t` for asynchronous operations.
3.  **Explain Rationale:** For non-trivial code suggestions or architectural changes, provide a brief explanation of *why* the approach was chosen, especially regarding its performance implications.
4.  **Modular & Testable Code:** Generate code that is modular, readable, and includes suggestions for unit tests, particularly for custom CUDA kernels.
5.  **Adhere to Project Structure:** Place generated code in the appropriate directories (`src/cuda/`, `src/layers/`, `tests/`, etc.) as defined in `README.md`.
6.  **Avoid Hallucination:** If information is uncertain or requires external context not provided, state the limitation or ask for clarification. Do not generate speculative or incorrect code.
7.  **Identify Dependencies:** Clearly state any new dependencies (Python packages, CUDA libraries, etc.) required for generated code.

### **III. Task-Specific Directives & Constraints**

When responding to specific requests, AI agents should consider the following:

#### **A. CUDA Kernel Generation/Modification Requests:**

* **Input:** Description of the financial operation (e.g., "calculate weighted bid-ask spread," "sparse attention with sliding window," "fused MLP block").
* **Output:** C++ CUDA `.cu` file content (kernel definition and host-side wrapper).
* **Constraints:**
    * **Kernel Signature:** Suggest appropriate kernel arguments (pointers to device memory, dimensions, parameters).
    * **Grid/Block Configuration:** Propose reasonable `<<<grid, block>>>` dimensions, explaining the rationale.
    * **Memory Usage:** Suggest efficient memory access patterns.
    * **Template Use:** Employ C++ templates for data types or dimensions where flexibility is beneficial.
    * **Error Checking:** Include `CUDA_CHECK` macros or similar for robustness.

#### **B. PyTorch C++ Extension Integration Requests:**

* **Input:** Existing CUDA kernel definition, desired PyTorch layer functionality.
* **Output:** Python `torch.autograd.Function` class definition, `setup.py` modifications if needed.
* **Constraints:**
    * **`forward` Method:** Correctly call the CUDA kernel, managing input tensors and output device tensors.
    * **`backward` Method:** Suggest a high-level approach for implementing the backward pass (gradients), noting that this often requires a separate custom CUDA kernel.
    * **Input/Output Handling:** Ensure correct tensor types, device placement, and dimensions.
    * **`setup.py`:** Provide necessary directives for `setuptools.Extension` to compile the CUDA source.

#### **C. Performance Analysis & Optimization Requests:**

* **Input:** A description of a performance issue, potentially with snippets of code or hypothetical Nsight Systems output.
* **Output:** Analysis of potential bottlenecks (e.g., "high global memory traffic," "low occupancy," "branch divergence"), and specific code-level suggestions for optimization.
* **Constraints:**
    * **Specificity:** Provide actionable advice, not just general principles.
    * **Quantification:** If possible, describe the expected performance impact (e.g., "could reduce memory bandwidth by X%").

#### **D. Code Generation (General Python/CUDA) Requests:**

* **Input:** High-level description of a module, utility function, or test case.
* **Output:** Python `.py` file content or C++ CUDA `.cu/.cuh` file content.
* **Constraints:**
    * **Modularity:** Generate functions/classes that are self-contained and have clear responsibilities.
    * **Readability:** Follow common coding standards (PEP 8 for Python, standard C++ for CUDA).
    * **Testable:** Design with testing in mind; suggest basic test cases where appropriate.

#### **E. Documentation & Explanation Requests:**

* **Input:** A specific code snippet, algorithm, or concept requiring explanation.
* **Output:** Clear, concise, and accurate explanatory text.
* **Constraints:**
    * **Technical Accuracy:** Ensure all technical details are correct.
    * **Clarity:** Use simple language where possible, define technical terms.
    * **Context:** Relate explanations back to the `ma-transformer` project's HFT context.

### **IV. Interaction Protocol**

* **Explicit Instructions:** Users will provide clear and explicit instructions for tasks.
* **Iterative Refinement:** AI agents should be prepared for iterative requests for refinement or debugging.
* **Contextual Awareness:** AI agents should leverage the entire repository content (especially `README.md` and other code files) as context for their responses.

By adhering to these directives, AI agents can become invaluable contributors to the `ma-transformer` project, accelerating development and helping achieve its ambitious performance goals.
