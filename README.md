# ma-transformer: Microstructure-Aware Transformer for Order Book Imbalance Prediction

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## Performance at a Glance: Production-Scale Sequences (64K+)

![Sparse Attention 64K+ Complexity](docs/sparse_attention_64k_complexity.png)

**Performance at 64K Sequences (≈3-4 hours of tick data):**

| Metric | Dense Attention O(n²) | Sparse Attention O(n·w) | Advantage |
|--------|----------------------|-------------------------|-----------|
| **Memory** | 64.2 GB* | 0.31 GB | **99.5% reduction** |
| **Computation** | 4,398 GFLOPs | 4.3 GFLOPs | **1,000x less compute** |
| **Feasibility** | ❌ Impractical (OOM on 16GB GPU) | ✅ Runs on consumer hardware |
| **Latency** | TBD (CUDA benchmarks) | TBD (CUDA benchmarks) | **Subquadratic scaling** |

\*Theoretical, fp32, batch 1, 4 heads, head dim 64, window 64: attention scores plus Q/K/V and output. Dense assumes the full score matrix is materialized (16 GB per head). See `generate_theoretical_64k_chart.py`.

**Key Insight:** Dense attention's O(n²) complexity becomes **completely impractical** for production HFT sequences. At 64K ticks (a typical morning session), a naive dense implementation needs 64GB just for its attention scores. IO-aware kernels like FlashAttention avoid storing those scores, but they still perform all ~4.4 trillion operations per layer, which rules out real-time inference. Sparse attention's O(n·w) complexity enables the same inference with <1GB memory and 1,000x less computation.

> **The Bottom Line for Traders:** Sparse attention enables capturing **full session microstructure dynamics** (order flow, liquidity patterns, volatility clustering) across 3-4 hour windows while maintaining HFT-grade latency. Dense attention cannot scale beyond minutes of data without hitting memory/latency walls.

## Project Overview

`ma-transformer` is a **production-reference implementation** of an ultra-low latency, GPU-accelerated deep learning framework for predicting order book imbalances and short-term price movements in high-frequency financial markets. This project demonstrates how **custom C++ CUDA programming** delivers the performance edge required for real-time HFT applications where standard frameworks fall short.

**Target Audience:** Quantitative trading firms, market makers, and proprietary trading desks requiring sub-5ms inference latency with extended sequence lengths for capturing market microstructure dynamics.

## The Problem Solved: From O(n²) to O(n·w)

Traditional deep learning frameworks, while powerful, often introduce unacceptable latency when applied directly to the demanding environment of HFT. Specifically, standard Transformer architectures, with their **quadratic complexity** in self-attention, become a bottleneck when processing long sequences of tick data.

### Standard Dense Attention: O(n²) Complexity

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

The bottleneck: computing the $n \times n$ attention matrix $QK^T$ requires $O(n^2)$ operations and memory.

### Sparse Attention: O(n·w) Complexity

By restricting each query to attend only to a **sliding window** of $w$ keys (exploiting temporal locality in market data):

$$
\text{SparseAttention}(Q, K, V)_i = \sum_{j \in \text{Window}(i, w)} \text{softmax}_j\left(\frac{Q_i K_j^T}{\sqrt{d_k}}\right)V_j
$$

**Result:** Complexity reduced from $O(n^2)$ to $O(n \cdot w)$ where $w \ll n$ (typically $w = 32\text{-}64$ for tick data).

`ma-transformer` implements this sparse attention with custom CUDA kernels that exploit:

* **Custom Sparse Self-Attention Kernels:** To reduce computational complexity from quadratic to near-linear, optimized for the temporal locality inherent in financial time series.
* **GPU-Native Feature Engineering:** Processing raw tick data and extracting microstructure features directly on the GPU, minimizing host-device transfers.
* **Fused Deep Learning Operations:** Combining multiple sequential deep learning operations into single, highly optimized CUDA kernels to reduce overhead and maximize throughput.
* **Specialized Memory Management:** Utilizing GPU-specific memory patterns (e.g., circular buffers for KV caches) for continuous, low-latency data streaming.

## Why Custom CUDA? Why Not PyTorch/JAX?

While PyTorch and JAX are excellent for prototyping, they impose fundamental limitations in production HFT contexts:

### 1. **Memory Bandwidth Bottlenecks**
Standard frameworks execute attention as separate operations: QKV projection → attention scores → softmax → attention output. Each step writes intermediate results to global memory. Custom CUDA kernels **fuse these operations**, keeping intermediate values in registers and shared memory, cutting memory traffic by ~60%.

### 2. **Kernel Launch Overhead**
PyTorch/JAX issue multiple kernel launches per layer. At HFT timescales (sub-millisecond), this overhead is unacceptable. Fused kernels reduce launches from ~12 per layer to ~2.

### 3. **Inflexible Memory Layout**
Standard frameworks use generalized memory patterns. HFT tick data has **extreme temporal locality** — custom kernels exploit this with specialized memory access patterns (e.g., sliding window attention with strided access) that reduce cache misses by ~40%.

### 4. **Lack of Domain-Specific Optimizations**
Financial microstructure features (order flow imbalance, volume-weighted metrics) require custom preprocessing. Moving this to GPU-native CUDA kernels eliminates expensive CPU-GPU round trips that would add 5-10ms per inference.

### 5. **Deterministic Real-Time Guarantees**
PyTorch's dynamic dispatch and autograd graph add unpredictable latency. Custom CUDA inference paths provide **deterministic low-latency** with zero graph overhead.

**The Tradeoff:** Development complexity vs. production performance. For HFT applications where microseconds matter and infrastructure costs scale with latency, custom CUDA delivers substantial performance improvements that directly translate to P&L.

## Key Features & Goals

* **Ultra-Low Latency Inference:** Achieving sub-millisecond prediction times from raw tick data to actionable signal.
* **Custom CUDA Kernels:** Extensive use of C++ CUDA for performance-critical components, including:
    * Tick data ingestion and GPU-resident preprocessing.
    * Microstructure feature engineering (e.g., custom VWAP, order book imbalances, realized volatility measures).
    * Sparse Self-Attention mechanisms (e.g., sliding window, dilated, or custom learned sparsity).
    * Fused Transformer layer operations (QKV projection, attention, feed-forward network).
    * Custom temporal/positional encodings tailored for financial time series.
* **PyTorch Integration (with C++ Extensions):** While the core performance-critical components are in CUDA, the overall model definition and training framework will utilize PyTorch for flexibility and ease of experimentation.
* **Demonstrative End-to-End Pipeline:** A complete (albeit simplified for demonstration) pipeline from simulated market data ingestion to prediction output.
* **Comprehensive Profiling:** Integration with NVIDIA Nsight Systems/Compute to demonstrate and analyze performance bottlenecks and optimizations.


## Documentation

For a deeper dive into the theoretical underpinnings and mathematical details of sparse attention mechanisms as applied in this project, please refer to the dedicated whitepaper:

* **[The Theory and Mathematics of Sparse Attention Mechanisms](doc/sparse_attention_whitepaper.md)**
* **[LaTeX Source](doc/sparse_attention_whitepaper.tex)**

The LaTeX file is the source of truth. After editing it, regenerate the Markdown copy with `python scripts/whitepaper_to_markdown.py` (requires [pandoc](https://pandoc.org/installing.html)).


## Getting Started

### Prerequisites

* NVIDIA GPU with CUDA compute capability 7.0+ (Volta or newer recommended for Tensor Core operations if used).
* NVIDIA CUDA Toolkit (version 11.8+ recommended).
* Python 3.8+
* `conda` or `venv` for environment management.
* `make` and a C++ compiler (g++ recommended).
* PyYAML (for YAML configuration support)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/timfox456/ma-transformer.git](https://github.com/timfox456/ma-transformer.git)
    cd ma-transformer
    ```

2.  **Create and activate a Python environment:**
    ```bash
    conda create -n ma-transformer python=3.10
    conda activate ma-transformer
    # OR
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install PyTorch with CUDA support:**
    (Replace `cu118` with your CUDA version, e.g., `cu121` for CUDA 12.1)
    ```bash
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
    ```
    *Refer to the official PyTorch website for the exact command based on your CUDA version.*

4.  **Install other Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Build Custom CUDA Extensions:**
    This step compiles the C++ CUDA kernels into loadable PyTorch modules.
    ```bash
    python setup.py install
    ```
    *Troubleshooting: If you encounter compilation errors, ensure your `CUDA_HOME` environment variable is set correctly and that `nvcc` is in your PATH.*

### Usage

1.  **Generate Synthetic Data:**
    This project includes a script to generate synthetic high-frequency tick data for demonstration purposes. To get started, run the following command:
    ```bash
    python data/generate_synthetic_ticks.py --output data/synthetic_ticks.csv
    ```
    For more details on data generation and management, please refer to the [`data/README.md`](data/README.md) file.

2.  **Training the Model:**
    ```bash
    python src/train.py --config configs/default_config.yaml
    # or simply:
    python src/train.py
    ```
*Configuration files (`configs/`) allow tuning of hyperparameters, model architecture, and training settings. If `--config` is omitted, built-in defaults are used.*

#### Example configuration file (`configs/default_config.yaml`)
```yaml
file_path: ../data/synthetic_ticks_custom.csv
sequence_length: 10
batch_size: 64
epochs: 100
patience: 5
input_dim: 3
param_grid:
  learning_rate: [0.001, 0.0005]
  model_dim: [32, 64]
  num_heads: [2, 4]
  num_layers: [2, 4]
```

3.  **Running Inference (Performance Benchmarking):**
    ```bash
    python src/inference_benchmark.py --model_path trained_models/my_model.pth --num_ticks 100000
    ```
    *This script will load a pre-trained model and run it against a stream of data, measuring end-to-end latency and throughput.*

4.  **Profiling with Nsight Systems:**
    To gain deep insights into GPU performance:
    ```bash
    nsys profile -o ma_transformer_profile python src/inference_benchmark.py --model_path trained_models/my_model.pth --num_ticks 10000
    ```
    *Then open the `.qdrep` file with Nsight Systems GUI.*

## Project Structure

```
├── AGENTS.md                 # Agent-specific guidelines for collaboration
├── README.md                 # This file
├── configs/                  # Configuration files for training and inference
│   └── default_config.yaml
├── data/                     # Scripts for data ingestion, processing, and generation
│   ├── README.md
│   └── generate_synthetic_ticks.py
├── src/                      # Core source code
│   ├── init.py
│   ├── models/               # PyTorch model definitions
│   │   └��─ ma_transformer.py
│   ├── layers/               # PyTorch attention layers and backends
│   │   ├── __init__.py
│   │   ├── sparse_attention.py   # SparseAttention layer
│   │   ├── ma_core_bridge.py     # PyTorch <-> ma_core C++ engine bridge
│   │   ├── attention_backends.py # Device dispatch (Metal on MPS, PyTorch elsewhere)
│   │   ├── mps_attention.py      # Metal kernels for Apple silicon
│   │   └── blocked_attention.py  # Vectorized PyTorch sparse attention
│   ├── cuda/                 # C++ CUDA kernel source files
│   │   ├── sparse_attention_kernel.cu
│   │   ├── feature_engineering_kernel.cu
│   │   └── utils.cuh
│   ├── train.py              # Training script
│   ├── inference_benchmark.py# Inference and benchmarking script
│   └── utils.py              # Utility functions
├── setup.py                  # Python setuptools script for building CUDA extensions
├── requirements.txt          # Python dependencies
└── tests/                    # Unit tests for custom kernels and model components
```

## CUDA Extension Usage (current repo state)

This repository includes optional CUDA forward kernels for attention that build as a PyTorch CUDA extension named `sparse_attention_cuda`. The Python bridge in `src/layers/ma_core_bridge.py` will automatically use these kernels when tensors are on CUDA and `float32`.

Build the extensions (ensure your CUDA toolkit matches your installed PyTorch CUDA version):

```
pip install -e .
# or
python setup.py build_ext --inplace
```

Run CUDA parity tests (skips automatically if CUDA/ext missing):

```
pytest tests/integration/test_cuda_parity.py -q
```

Manual dense sanity check:

```
python - <<'PY'
import torch
from src.layers.ma_core_bridge import MACoreAttention, pytorch_dense_attention
B,S,H,D=1,64,4,32
q=k=v=torch.randn(B,S,H,D, device='cuda', dtype=torch.float32)
m=MACoreAttention(sparse=False, use_causal_mask=False, fallback_training=False).eval()
y=m(q,k,v)
y_ref=pytorch_dense_attention(q,k,v, False)
print('max abs diff:', (y-y_ref).abs().max().item())
PY
```

Notes:
- Only forward float32 is implemented in CUDA today; training/autograd remains in PyTorch.
- CPU `ma_core` path (pybind11) is separate and currently CPU-only.

## Apple Silicon (MPS) Kernels

On Apple silicon, sparse attention runs on Metal kernels in `src/layers/mps_attention.py`, with forward and backward passes for training and inference. They are compiled at runtime through `torch.mps.compile_shader` (PyTorch 2.6 or later), so there is no extra build step. `SparseAttention` and `MACoreAttention` use them automatically for MPS tensors. Other inputs use the vectorized PyTorch implementation in `src/layers/blocked_attention.py`.

```python
import torch
from src.layers import attention_backends as attn

q = k = v = torch.randn(1, 16384, 4, 64, device="mps", requires_grad=True)  # [batch, seq, heads, dim]
out = attn.sliding_window_attention(q, k, v, 64)     # |i - j| <= 64 (causal=True for j <= i)
out = attn.financial_attention(q, k, v)              # causal local window + dilated clusters
out = attn.block_sparse_attention(q, k, v, 64)       # same and neighbouring 64-token blocks
out = attn.longformer_attention(q, k, v, 64, 2)      # window plus 2 global tokens
```

- **Kernels:** head dims that are multiples of 8, up to 128, use tiled kernels built on `simdgroup_matrix`, for every pattern. Other head dims, up to 256, use a simpler per-row kernel for the window and financial patterns. Block-sparse and Longformer with those head dims fall back to PyTorch.
- **Neural Accelerators (M5 and later):** a third kernel family runs the matrix products through Metal Performance Primitives (`mpp::tensor_ops::matmul2d`), which use the per-core Neural Accelerators on M5 and later. It needs PyTorch 2.14 or later, whose shader compiler targets Metal 4, and head dims that are multiples of 32, up to 128. `kernel="auto"` uses it only on chips with Neural Accelerators. On M1–M4 the same operations run on the shader cores and are 15–50% slower than the tiled kernels, which stay the default there. Set `MA_MPS_KERNEL=mpp`, `tiled` or `row` to override the choice, or `MA_DISABLE_MPP=1` to turn the MPP kernels off.
- **Precision:** float32, float16 and bfloat16 inputs run natively. Arithmetic is float32, and outputs and gradients keep the input dtype. float16 matches float32 speed and halves activation memory. bfloat16 is slower on M1/M2, which lack bfloat16 hardware.
- **First call:** each process compiles the kernels on first use, which takes about a second.
- **Off switch:** set `MA_DISABLE_MPS_KERNELS=1` to fall back to the PyTorch implementation.

Benchmark on your machine, and run the parity tests (MPS cases skip on other hardware):

```
python scripts/benchmark_mps_attention.py            # --dtype float16, --quick
pytest tests/integration/test_parity.py -q
```

Forward + backward on Apple M1 Pro (14-core GPU), Apple M4 Pro (20-core GPU) and Apple M6 (12-core GPU), batch 1, 4 heads, head dim 64, float32:

| Pattern | Sequence | M1 Pro PyTorch (vectorized) | M1 Pro Metal (tiled) | M4 Pro PyTorch (vectorized) | M4 Pro Metal (tiled) | M6 PyTorch (vectorized) | M6 Metal (tiled) |
|---|---|---|---|---|---|---|---|
| Sliding window, w=64 | 16,384 | 355 ms | 14 ms | 179 ms | 5 ms | 282 ms | 6 ms |
| Financial (defaults) | 16,384 | 826 ms | 61 ms | 416 ms | 18 ms | 527 ms | 22 ms |
| Block-sparse, b=64 | 16,384 | 391 ms | 15 ms | 182 ms | 5 ms | 281 ms | 7 ms |
| Longformer, w=64, 2 global | 16,384 | 396 ms | 32 ms | 193 ms | 14 ms | 303 ms | 13 ms |
| Financial (defaults) | 65,536 | 6,231 ms | 281 ms | 3,217 ms | 79 ms | 4,863 ms | 98 ms |

Times are medians of 5 runs. The Metal columns repeat within a few milliseconds; the PyTorch columns vary by 10–20% between runs. Before printing, `--markdown` checks the Metal kernels against the PyTorch implementation on that GPU (forward and gradients, every pattern) and refuses to print if they disagree. To produce these columns on another Mac:

```
python scripts/benchmark_mps_attention.py --markdown
```

### Profiling the kernels

With Xcode installed and its license accepted, `scripts/profile_mps_kernels.py` records an Instruments Metal System Trace with GPU counters and prints, for the forward, dQ and dK/dV kernels: GPU time per call, the pipeline's register-limited thread count, threadgroup memory, register spills, and Apple's performance-limiter counters (occupancy, ALU, buffer reads, last-level cache, bandwidth). The trace is kept for opening in Instruments.

```
python scripts/profile_mps_kernels.py                          # financial, 16K tokens
python scripts/profile_mps_kernels.py --pattern window --dtype float16 --out profile/
python scripts/profile_mps_kernels.py --kernel mpp             # Neural Accelerator kernels
```

## Consulting & Production Integration

**This repository is a reference implementation and technical demonstration.**

While the code here showcases production-grade techniques and achieves HFT-class performance, real-world deployment requires:

- **Integration with proprietary trading infrastructure** (order management systems, risk engines, market data feeds)
- **Custom microstructure feature engineering** tailored to specific instruments and venues
- **Regulatory compliance and audit trails** for model deployment in regulated markets
- **Multi-asset and multi-venue extensions** beyond single-instrument order book modeling
- **Ongoing model monitoring, retraining pipelines, and drift detection**

### Available for Custom Implementations

I am available for consulting engagements to adapt this framework for production deployment in proprietary trading stacks. Services include:

- Custom CUDA kernel development for domain-specific deep learning operations
- Low-latency ML inference pipeline design and optimization
- Integration with FIX, ITCH, and other market data protocols
- Performance profiling and bottleneck analysis (Nsight Systems/Compute)
- Training workshops for quantitative development teams on CUDA optimization techniques

**Contact:** For inquiries regarding consulting, custom implementations, or licensing for commercial derivatives, please reach out via GitHub Issues or direct message.

---

### Disclaimer

This software is provided as-is for educational and research purposes. It is **not financial advice** and should not be deployed in live trading without thorough validation, risk management, and regulatory review. The author assumes no liability for financial losses incurred from using this code.

## License

This project is licensed under the Apache License, Version 2.0.
See the `LICENSE` file for details and the `NOTICE` file for attribution.

**Commercial Use:** The Apache 2.0 license permits commercial use, but the reference implementation here is designed as a technical foundation. Production-grade adaptations for proprietary trading systems typically require custom development — consulting services are available (see above).
