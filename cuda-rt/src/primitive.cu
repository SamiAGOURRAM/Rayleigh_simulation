#include <stdio.h>
#include <cuda_runtime.h>
#include "rt_types.hpp"

// Kernel to convert from conservative to primitive variables
__global__ void primFromConsKernel(float *d_r, float *d_ru, float *d_rv, float *d_e,
    float *d_u, float *d_v, float *d_p, float *d_c,
    SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Modified: Process both interior AND ghost cells
    // Remove this line: if (i >= params.Nx || j >= params.Ny) return;

    // Process all cells including ghosts (-1 to Nx and -1 to Ny)
    if (i <= params.Nx && j <= params.Ny && i >= -1 && j >= -1) {
    int idx_ij = idx(i, j, params.Nx, params.Ny);

    // Get density
    float r = d_r[idx_ij];

    // Ensure density is positive
    r = fmaxf(r, 1e-6f);

    // Calculate velocities
    float u = d_ru[idx_ij] / r;
    float v = d_rv[idx_ij] / r;

    // Store velocities
    d_u[idx_ij] = u;
    d_v[idx_ij] = v;

    // Calculate pressure
    float kinetic_energy = 0.5f * r * (u*u + v*v);
    float p = (params.gamma - 1.0f) * (d_e[idx_ij] - kinetic_energy);

    // Ensure pressure is positive
    p = fmaxf(p, 1e-6f);

    // Store pressure
    d_p[idx_ij] = p;

    // Calculate sound speed
    d_c[idx_ij] = sqrtf(params.gamma * p / r);
    }
}

void computePrimitiveVariables(float *d_r, float *d_ru, float *d_rv, float *d_e,
    float *d_u, float *d_v, float *d_p, float *d_c,
    SimParams params) {
    // Modified: Include ghost cells in calculation
    dim3 blockDim(16, 16);
    dim3 gridDim((params.Nx + 2 + blockDim.x - 1) / blockDim.x,
    (params.Ny + 2 + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    primFromConsKernel<<<gridDim, blockDim>>>(d_r, d_ru, d_rv, d_e, d_u, d_v, d_p, d_c, params);

    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
    fprintf(stderr, "Error in computePrimitiveVariables: %s\n", cudaGetErrorString(err));
    }
}