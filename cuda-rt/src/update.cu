// update.cu
#include <cuda_runtime.h>
#include <stdio.h> 
#include "rt_types.hpp"

// Kernel to update solution with Euler time step
__global__ void eulerStepKernel(float *d_q, float *d_q_rhs, float dt, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        d_q[idx] += dt * d_q_rhs[idx];
    }
}

// Host function to update all solution variables
void updateSolution(float *d_r, float *d_ru, float *d_rv, float *d_e,
                  float *d_r_rhs, float *d_ru_rhs, float *d_rv_rhs, float *d_e_rhs,
                  float dt, SimParams params) {
    // Calculate total size including ghost cells
    int size = (params.Nx + 2) * (params.Ny + 2);
    
    // Set up grid and block dimensions for 1D kernel
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    
    // Update each variable
    eulerStepKernel<<<gridSize, blockSize>>>(d_r, d_r_rhs, dt, size);
    eulerStepKernel<<<gridSize, blockSize>>>(d_ru, d_ru_rhs, dt, size);
    eulerStepKernel<<<gridSize, blockSize>>>(d_rv, d_rv_rhs, dt, size);
    eulerStepKernel<<<gridSize, blockSize>>>(d_e, d_e_rhs, dt, size);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in updateSolution: %s\n", cudaGetErrorString(err));
    }
}