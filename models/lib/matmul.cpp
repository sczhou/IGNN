// Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

// This file is part of the implementation as described in the NIPS 2018 paper:
// Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
// Please see the file LICENSE.txt for the license governing this code.


#include <torch/extension.h>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

using namespace std;
void matmul1_cuda(at::Tensor mat_x, at::Tensor mat_y, at::Tensor mat_i, at::Tensor out, int n, int m, int e, int o, int b);
void matmul1_bwd_cuda(at::Tensor gradients, at::Tensor mat_x, at::Tensor mat_y, at::Tensor mat_i, at::Tensor out_x, at::Tensor out_y, int m, int n, int e, int o, int b);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("matmul1", &matmul1_cuda, "matmul1 forward(CUDA)");
	m.def("matmul1_bwd", &matmul1_bwd_cuda, "matmul1 backward (CUDA)");
}
