#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys

# Optional: Torch CUDA extension for sparse attention (guarded)
_has_torch = False
_cuda_ext = None
_cmdclass = {'build_ext': build_ext}
try:
    import torch  # noqa: F401
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension
    from torch.utils.cpp_extension import CUDA_HOME
    _has_torch = True
    if CUDA_HOME is not None:
        _cuda_ext = CUDAExtension(
            name='sparse_attention_cuda',
            sources=[
                'src/cuda/sparse_attention_kernel.cu',
                'src/cuda/dense_attention_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-O3', '-std=c++17'], 'nvcc': ['-O3']},
        )
        _cmdclass = {'build_ext': BuildExtension}
except Exception:
    # Torch not available or no CUDA toolchain; skip CUDA extension
    _has_torch = False


# Pure C++ extension without PyTorch dependencies
ext_modules = [
    Extension(
        'ma_core',
        sources=[
            'src/csrc/main.cpp', 
            'src/csrc/ma_core.cpp',
            'src/csrc/tensor.cpp'
        ],
        include_dirs=[pybind11.get_include(), 'src/csrc'],
        language='c++',
        extra_compile_args=['-std=c++17', '-O3'] + (['-stdlib=libc++'] if sys.platform == 'darwin' else []),
        extra_link_args=['-stdlib=libc++'] if sys.platform == 'darwin' else []
    )
]

if _cuda_ext is not None:
    ext_modules.append(_cuda_ext)

setup(
    ext_modules=ext_modules,
    cmdclass=_cmdclass,
)
