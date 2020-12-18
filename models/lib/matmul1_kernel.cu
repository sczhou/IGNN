// Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

// This file is part of the implementation as described in the NIPS 2018 paper:
// Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
// Please see the file LICENSE.txt for the license governing this code.

#include <math.h>
#include "stdio.h"
#include "iostream"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <ATen/ATen.h>

using namespace std;

const int N_THREADS_M = 256;
const int N_THREADS_O = 1024 / N_THREADS_M;

__global__
void matmul1_kernel(float *mat_x, float *mat_y, long *mat_i, float *mat_o, int m, int n, int e, int o, int batch_size)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int batch = blockIdx.z *blockDim.z + threadIdx.z;

	float sum = 0;

	if (batch >= batch_size || row >= m || col >= o){
		   return;
		}

	// Fetch x indices
	int pos_i = (batch * m * o) + (row * o) + col;
	int xind_col = mat_i[pos_i];

	// Mat mult
	for (int i = 0; i < e; i++) {
	    int pos_y = (batch * m * e) + (row * e + i);
	    int pos_x = (batch * n * e) + (xind_col * e + i);
	    sum += mat_y[pos_y] * mat_x[pos_x];
	}

	int pos = (batch * m * o) + (row * o + col);
	mat_o[pos] = sum;	 
}

void matmul1_cuda(at::Tensor mat_x, at::Tensor mat_y, at::Tensor mat_i, at::Tensor out, int n, int m, int e, int o, int b) {
		// Set array and CUDA block/grid sizes


		dim3 block(N_THREADS_O, N_THREADS_M, 1);
		dim3 grid((int)ceil(((float)o)/N_THREADS_O), (int)ceil(((float)m)/N_THREADS_M), b);
		
		// Call kernel
    	matmul1_kernel<<<grid, block>>>(mat_x.data<float>(), mat_y.data<float>(), mat_i.data<long>(), out.data<float>(), m, n, e, o, b);

		return;
}

