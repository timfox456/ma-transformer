#!/usr/bin/env bash
set -euo pipefail

echo "== NVIDIA driver / NVML =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || echo "nvidia-smi present but failed (driver/GPU unavailable)"
else
  echo "nvidia-smi: not found"
fi

echo
echo "== CUDA toolkit (nvcc) =="
echo "CUDA_HOME=${CUDA_HOME:-}"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "nvcc: not found"
fi

echo
echo "== PyTorch CUDA =="
python3 - << 'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    try:
        import sparse_attention_cuda  # optional extension
        print("sparse_attention_cuda: available")
    except Exception as e:
        print("sparse_attention_cuda: not available (", e.__class__.__name__, ")")
except Exception as e:
    print("torch: not importable (", e.__class__.__name__, ")")
PY

echo
echo "== Suggested next steps =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "- Install NVIDIA driver (Ubuntu): sudo ubuntu-drivers autoinstall && reboot"
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "- Install CUDA toolkit for building extensions (nvcc) if desired:"
  echo "  sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-1"
fi
echo "- For PyTorch with CUDA 12.1: pip/uv install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio"

