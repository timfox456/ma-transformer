# ma-transformer: Microstructure-Aware Transformer for Order Book Imbalance Prediction

## Project Overview

`ma-transformer` is an ambitious project focused on developing an ultra-low latency, GPU-accelerated deep learning framework for predicting order book imbalances and short-term price movements in high-frequency financial markets. At its core, this project leverages the power of NVIDIA CUDA to implement highly optimized, custom Transformer components, specifically tailored for the unique characteristics of tick-level market data.

The primary goal is to demonstrate how custom C++ CUDA programming can overcome the latency and computational challenges associated with applying sophisticated deep learning models, such as Transformers with self-attention, to real-time high-frequency trading (HFT) scenarios. This project aims to serve as a comprehensive reference implementation, showcasing best practices for performance-critical deep learning in finance.

## The Problem Solved

Traditional deep learning frameworks, while powerful, often introduce unacceptable latency when applied directly to the demanding environment of HFT. Specifically, standard Transformer architectures, with their quadratic complexity ($O(L^2)$) in self-attention, become a bottleneck when processing long sequences of tick data. `ma-transformer` addresses this by implementing:

* **Custom Sparse Self-Attention Kernels:** To reduce computational complexity from quadratic to near-linear, optimized for the temporal locality inherent in financial time series.
* **GPU-Native Feature Engineering:** Processing raw tick data and extracting microstructure features directly on the GPU, minimizing host-device transfers.
* **Fused Deep Learning Operations:** Combining multiple sequential deep learning operations into single, highly optimized CUDA kernels to reduce overhead and maximize throughput.
* **Specialized Memory Management:** Utilizing GPU-specific memory patterns (e.g., circular buffers for KV caches) for continuous, low-latency data streaming.

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


## Documetn

For a deeper dive into the theoretical underpinnings and mathematical details of sparse attention mechanisms as applied in this project, please refer to the dedicated whitepaper:

* **[The Theory and Mathematics of Sparse Attention Mechanisms](doc/sparse_attention_whitepaper.md)**
* **[Latex Source](doc/sparse_attention_whitepaper.tex)**


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
│   ├── layers/               # PyTorch wrappers for custom CUDA layers
│   │   ├── init.py
│   │   ├── sparse_attention.py
│   │   └── fused_mlp.py
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

