// Author: Tobias Plötz, TU Darmstadt (tobias.ploetz@visinf.tu-darmstadt.de)

// This file is part of the implementation as described in the NIPS 2018 paper:
// Tobias Plötz and Stefan Roth, Neural Nearest Neighbors Networks.
// Please see the file LICENSE.txt for the license governing this code.


#include <math.h>
#include <vector>
#include "stdio.h"
#include "iostream"
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>
#include <chrono>


using namespace std;

const int N_THREADS_N = 256;
const int N_THREADS_E = 1024 / N_THREADS_N;

__device__
void matmul1_xgrad(float *grad, float *mat_y, long *mat_i, float *mat_ox, int m, int n, int e, int o, int batch_size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z *blockDim.z + threadIdx.z;


    if (batch >= batch_size || row >= m || col >= o)
        return;

	int pos_i = (batch  * m * o) + (row * o) + col;	
	int idx = mat_i[pos_i];
	float g = grad[pos_i];

	for (int j = 0; j < e; j++) {
		int pos_y = (batch * m * e) + (row * e) + j;
	    int pos_ox = (batch * n * e) + (idx * e) + j;
	    atomicAdd(mat_ox + pos_ox, mat_y[pos_y] * g);
	}
}

__device__
void matmul1_ygrad(float *grad, float *mat_x, long *mat_i, float *mat_o, int m, int n, int e, int o, int batch_size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z *blockDim.z + threadIdx.z;


    if (batch >= batch_size || row >= m || col >= e)
        return;

    float sum = 0.0;

    for (int i = 0; i < o; i++) {
    	int pos_i = (batch * m * o) + (row * o) + i;
			int xind = mat_i[pos_i];
			int pos_x = (batch * n * e) + (xind * e) + col;

			int pos_g = (batch * m * o) + (row * o) + i;
			float g = grad[pos_g];

			sum = sum + (mat_x[pos_x] * g);
    }
    int pos_o = (batch * m * e) + (row * e) + col;
    mat_o[pos_o] = sum;
}


__global__
void matmul1_bwd_kernel_xgrad(float *gradients, float *mat_x, float *mat_y, long *mat_i, float *mat_ox, int m, int n,  int e, int o, int batch_size){
		matmul1_xgrad(gradients, mat_y, mat_i, mat_ox, m, n, e, o, batch_size);
}


__global__
void matmul1_bwd_kernel_ygrad(float *gradients, float *mat_x, float *mat_y, long *mat_i, float *mat_oy, int m, int n,  int e, int o, int batch_size){
    matmul1_ygrad(gradients, mat_x, mat_i, mat_oy, m, n, e, o, batch_size);
}


void matmul1_bwd_cuda(at::Tensor gradients, at::Tensor mat_x, at::Tensor mat_y, at::Tensor mat_i, at::Tensor out_x, at::Tensor out_y, int m, int n, int e, int o, int b){
	// Set array and CUDA block/grid sizes

	dim3 block(N_THREADS_E, N_THREADS_N, 1);
	dim3 grid((int)ceil(((float)e)/N_THREADS_E), (int)ceil(((float)std::max(n, m))/N_THREADS_N), b);

	// Call kernel
	matmul1_bwd_kernel_ygrad<<<grid, block>>>(gradients.data<float>(), mat_x.data<float>(), mat_y.data<float>(), mat_i.data<long>(), out_y.data<float>(), m, n, e, o, b);


	dim3 block_xgrad(N_THREADS_E, N_THREADS_N, 1);
	dim3 grid_xgrad((int)ceil(((float)e)/N_THREADS_E), (int)ceil(((float)std::max(n, m))/N_THREADS_N), b);

	// Call kernel
	matmul1_bwd_kernel_xgrad<<<grid_xgrad, block_xgrad>>>(gradients.data<float>(), mat_x.data<float>(), mat_y.data<float>(), mat_i.data<long>(), out_x.data<float>(), m, n, e, o, b);

	return;
}

