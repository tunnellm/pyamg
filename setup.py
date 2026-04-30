#!/usr/bin/env python
"""PyAMG: Algebraic Multigrid Solvers in Python.

PyAMG is a library of Algebraic Multigrid (AMG)
solvers with a convenient Python interface.

PyAMG features implementations of:

- Ruge-Stuben (RS) or Classical AMG
- AMG based on Smoothed Aggregation (SA)
- Adaptive Smoothed Aggregation (aSA)
- Compatible Relaxation (CR)
- Krylov methods such as CG, GMRES, FGMRES, BiCGStab, MINRES, etc

PyAMG is primarily written in Python with
supporting C++ code for performance critical operations.
"""

import sys
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

amg_core_headers = ['air',
                    'evolution_strength',
                    'graph',
                    'krylov',
                    'linalg',
                    'relaxation',
                    'ruge_stuben',
                    'smoothed_aggregation']

ext_modules = [
    Pybind11Extension(f'pyamg.amg_core.{f}',
                      sources=[f'pyamg/amg_core/{f}_bind.cpp'],
                     )
    for f in amg_core_headers]

ext_modules += [
    Pybind11Extension('pyamg.amg_core.tests.bind_examples',
                      sources=['pyamg/amg_core/tests/bind_examples_bind.cpp'],
                     )
    ]

# pyamg/amg_core/sparse_blas: OpenMP-threaded SpMV / SpGEMM / RAP kernels.
# OpenMP flags are platform-specific; for non-MSVC compilers we use -fopenmp.
# If the platform lacks OpenMP, the kernels still build (guarded by _OPENMP)
# and run serially.
if sys.platform == 'win32':
    omp_compile_args = ['/openmp']
    omp_link_args = []
elif sys.platform == 'darwin':
    # macOS clang requires libomp via Homebrew; users may need to set
    # CC/CXX/CFLAGS appropriately. We try -fopenmp; if it fails, the user can
    # set PYAMG_DISABLE_OPENMP=1 before building.
    omp_compile_args = ['-Xpreprocessor', '-fopenmp']
    omp_link_args = ['-lomp']
else:
    omp_compile_args = ['-fopenmp']
    omp_link_args = ['-fopenmp']

ext_modules += [
    Pybind11Extension(
        'pyamg.amg_core.sparse_blas',
        sources=['pyamg/amg_core/sparse_blas/sparse_blas_bind.cpp'],
        include_dirs=['pyamg/amg_core/sparse_blas'],
        extra_compile_args=omp_compile_args,
        extra_link_args=omp_link_args,
    ),
]

setup(
    ext_modules=ext_modules,
    cmdclass={'build_ext': build_ext},
)
