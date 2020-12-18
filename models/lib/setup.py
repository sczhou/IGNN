# Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

# This file is part of the implementation as described in the NIPS 2018 paper:
# Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
# Please see the file LICENSE.txt for the license governing this code.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='matmul',
    ext_modules=[
        CUDAExtension('matmul_cuda', [
            'matmul.cpp',
            'matmul1_kernel.cu',
            'matmul1_bwd_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
