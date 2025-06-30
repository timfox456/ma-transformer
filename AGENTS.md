# AGENTS.md: Collaboration Guidelines for ma-transformer Project

This document outlines the roles, responsibilities, and collaborative workflows for individuals contributing to the `ma-transformer` project. Our aim is to foster an efficient, clear, and high-performance development environment.

## Core Principles

* **Performance First:** All contributions must prioritize ultra-low latency and computational efficiency.
* **CUDA Expertise:** A strong understanding of CUDA programming (C++ and kernel optimization) is essential for core development tasks.
* **Clear Communication:** Maintain open and transparent communication through issues, pull requests, and discussions.
* **Test-Driven Development:** Write tests for new features, especially for custom CUDA kernels, to ensure correctness and stability.
* **Documentation:** Document code, algorithms, and performance implications clearly.

## Roles & Responsibilities

For the initial phase, we envision the following primary "agents" or contributor types:

### 1. **Lead Architect / Project Owner (You!)**

* **Role:** Defines the overall system architecture, sets strategic direction, makes final technical decisions, and ensures project alignment with HFT performance goals.
* **Responsibilities:**
    * High-level design of the Transformer architecture and custom CUDA integrations.
    * Oversees the development roadmap and key milestones.
    * Reviews and approves major pull requests.
    * Manages project dependencies and infrastructure.
    * Acts as the primary point of contact for external inquiries (e.g., from WWT/NVIDIA).

### 2. **CUDA Kernel Developer(s)**

* **Role:** Specializes in writing, optimizing, and debugging C++ CUDA kernels for performance-critical components.
* **Responsibilities:**
    * Implement custom CUDA kernels for:
        * Raw tick data ingestion and preprocessing.
        * Microstructure feature extraction.
        * Sparse self-attention mechanisms (forward and backward passes).
        * Fused feed-forward layers.
        * Custom positional/temporal encodings.
    * Perform low-level memory optimization (shared memory, memory coalescing, thread block configuration).
    * Utilize NVIDIA profiling tools (Nsight Systems, Nsight Compute) to identify and resolve performance bottlenecks.
    * Write comprehensive unit tests for all custom kernels.
    * Work closely with the PyTorch Integrator to ensure seamless integration.

### 3. **PyTorch Integrator / Model Developer**

* **Role:** Focuses on integrating custom CUDA kernels into the PyTorch framework, defining the overall deep learning model, and managing the training pipeline.
* **Responsibilities:**
    * Develop PyTorch `torch.autograd.Function` wrappers for custom CUDA kernels.
    * Define the `ma_transformer` model architecture using PyTorch modules, incorporating custom layers.
    * Design and implement the training loop, including data loaders, optimizers, and loss functions.
    * Conduct hyperparameter tuning and model experimentation.
    * Evaluate model performance (prediction accuracy, generalization).
    * Ensure proper data flow between CPU (for non-GPU tasks) and GPU.

### 4. **Data Engineer / Generator**

* **Role:** Responsible for creating, managing, and preparing high-quality financial tick data for training and evaluation.
* **Responsibilities:**
    * Develop scripts for generating realistic synthetic tick data that mimics market microstructure.
    * Implement efficient data loading mechanisms, potentially leveraging GPU-direct storage or optimized I/O.
    * Ensure data integrity and consistency.
    * Document the data formats and generation process (`data/README.md`).

### 5. **Benchmarking & Profiling Specialist**

* **Role:** Dedicated to rigorous performance measurement, bottleneck identification, and reporting.
* **Responsibilities:**
    * Develop standardized benchmarking scripts for end-to-end latency and throughput.
    * Regularly profile the system using NVIDIA Nsight Systems and Nsight Compute.
    * Analyze profiling data to pinpoint performance bottlenecks at the kernel and system level.
    * Generate performance reports and suggest optimization strategies to the Lead Architect and CUDA Kernel Developers.
    * Collaborate on setting and tracking performance targets.

## Workflow & Collaboration

1.  **Issue Tracking:**
    * All work should stem from an issue in the GitHub repository.
    * Clearly define the problem, proposed solution, and acceptance criteria for each issue.
    * Assign issues to the relevant agent(s).

2.  **Branching Strategy:**
    * Use a feature branch workflow:
        * `main`: Stable, production-ready code. Only merge after thorough review and testing.
        * `develop`: Integration branch for new features.
        * `feature/your-feature-name`: Branch off `develop` for new features or bug fixes.

3.  **Pull Requests (PRs):**
    * Submit PRs from your feature branch to `develop`.
    * Provide a clear PR description, linking to the relevant issue.
    * Include:
        * **What:** What changes were made.
        * **Why:** The reasoning behind the changes.
        * **How:** Brief explanation of implementation details, especially for CUDA kernels.
        * **Tests:** Mention how the changes were tested (unit tests, integration tests, benchmarks).
        * **Performance Impact:** (Crucial for this project) Discuss expected and measured performance changes, including Nsight profiling snippets if applicable.
    * Request reviews from relevant agents (e.g., CUDA Kernel Developer review for CUDA code, PyTorch Integrator review for PyTorch models).

4.  **Code Review:**
    * Reviewers should focus on correctness, performance, clarity, adherence to coding standards, and maintainability.
    * Provide constructive feedback.

5.  **Testing:**
    * **Unit Tests:** Essential for custom CUDA kernels and PyTorch wrappers. Use Google Test for C++ CUDA tests, `unittest` or `pytest` for Python components.
    * **Integration Tests:** Verify interactions between different components (e.g., data ingestion -> feature engineering -> Transformer inference).
    * **Performance Benchmarks:** Run regular benchmarks to track performance regressions/improvements.

6.  **Communication:**
    * Use GitHub Issues for task-specific discussions.
    * Use PR comments for code-specific feedback.
    * Consider a shared chat channel (e.g., Slack, Discord) for real-time discussions and quick questions.
    * Regular sync-up meetings (e.g., weekly stand-ups) to coordinate efforts and address blockers.

## Coding Standards & Best Practices

* **C++ CUDA:**
    * Follow NVIDIA's CUDA C++ programming guide.
    * Prioritize memory coalescing and shared memory usage.
    * Minimize host-device transfers.
    * Handle error checking for CUDA API calls.
    * Document kernel launch parameters (grid, block dimensions) and assumptions.
    * Use `const` correctness.
* **Python:**
    * Adhere to PEP 8 guidelines.
    * Use type hints.
    * Write clear docstrings for functions and classes.
* **Documentation:** Keep READMEs, code comments, and docstrings up-to-date and comprehensive.

By following these guidelines, we can ensure the `ma-transformer` project evolves into a robust, high-performance, and well-documented reference for GPU-accelerated deep learning in HFT.
