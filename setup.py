
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import pybind11
import sys


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

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext}
)
