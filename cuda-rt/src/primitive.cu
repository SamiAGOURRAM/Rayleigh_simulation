#include <stdio.h>
#include <cuda_runtime.h>
#include "rt_types.hpp"

// Kernel to convert from conservative to primitive variables
__global__ void primFromConsKernel(float *d_r, float *d_ru, float *d_rv, float *d_e,
    float *d_u, float *d_v, float *d_p, float *d_c,
    SimParams params) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
int j = blockIdx.y * blockDim.y + threadIdx.y;

// Include ghost cells in calculation (remove the boundary check)
if (i < params.Nx+2 && j < params.Ny+2) {
int ij = (i) + (j)*(params.Nx+2);  // Direct indexing including ghosts

// Get density (ensure positive)
float r = d_r[ij];
r = fmaxf(r, 1e-6f);

// Calculate velocities
float u = d_ru[ij] / r;
float v = d_rv[ij] / r;

// Store velocities
d_u[ij] = u;
d_v[ij] = v;

// Calculate pressure
float kinetic_energy = 0.5f * r * (u*u + v*v);
float p = (params.gamma - 1.0f) * (d_e[ij] - kinetic_energy);
p = fmaxf(p, 1e-6f);

d_p[ij] = p;
d_c[ij] = sqrtf(params.gamma * p / r);
}
}

// Host function to launch primitive calculation kernel
void computePrimitiveVariables(float *d_r, float *d_ru, float *d_rv, float *d_e,
                             float *d_u, float *d_v, float *d_p, float *d_c,
                             SimParams params) {
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((params.Nx + blockDim.x - 1) / blockDim.x,
                (params.Ny + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    primFromConsKernel<<<gridDim, blockDim>>>(d_r, d_ru, d_rv, d_e, d_u, d_v, d_p, d_c, params);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in computePrimitiveVariables: %s\n", cudaGetErrorString(err));
    }
}