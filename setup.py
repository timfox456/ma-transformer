
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ext_modules = []
try:
    ext_modules.append(
        CUDAExtension('ma_transformer_cuda', [
            'src/cuda/sparse_attention_kernel.cu',
        ])
    )
except OSError:
    print("CUDA_HOME not set, skipping CUDA extension.")


setup(
    name='ma_transformer_cuda',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    })
