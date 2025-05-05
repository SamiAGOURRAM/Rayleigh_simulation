#include <stdio.h>
// boundary.cu
#include <cuda_runtime.h>
#include "rt_types.hpp"

// Kernel for periodic X boundary condition
__global__ void bcPeriodicXKernel(float *d_q, int Nx, int Ny) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= Ny) return;
    
    // Left boundary: q[-1,j] = q[Nx-1,j]
    d_q[idx(-1, j, Nx, Ny)] = d_q[idx(Nx-1, j, Nx, Ny)];
    
    // Right boundary: q[Nx,j] = q[0,j]
    d_q[idx(Nx, j, Nx, Ny)] = d_q[idx(0, j, Nx, Ny)];
}

// Kernel for mirror Y boundary condition
__global__ void bcMirrorYKernel(float *d_q, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= Nx) return;
    
    // Bottom boundary: q[i,-1] = q[i,0]
    d_q[idx(i, -1, Nx, Ny)] = d_q[idx(i, 0, Nx, Ny)];
    
    // Top boundary: q[i,Ny] = q[i,Ny-1]
    d_q[idx(i, Ny, Nx, Ny)] = d_q[idx(i, Ny-1, Nx, Ny)];
}

// Special boundary for vertical velocity (v=0 at walls)
__global__ void bcVelocityYKernel(float *d_rv, float *d_r, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= Nx) return;
    
    // At boundaries, v = 0, so rv = 0
    d_rv[idx(i, -1, Nx, Ny)] = 0.0f;
    d_rv[idx(i, 0, Nx, Ny)] = 0.0f;
    d_rv[idx(i, Ny-1, Nx, Ny)] = 0.0f;
    d_rv[idx(i, Ny, Nx, Ny)] = 0.0f;
}

// Special kernel for primitive v=0 at walls
__global__ void bcVKernel(float *d_v, int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= Nx) return;
    
    // Set v=0 at walls
    d_v[idx(i, -1, Nx, Ny)] = 0.0f;
    d_v[idx(i, 0, Nx, Ny)] = 0.0f; 
    d_v[idx(i, Ny-1, Nx, Ny)] = 0.0f;
    d_v[idx(i, Ny, Nx, Ny)] = 0.0f;
}

// New function to apply boundary conditions to primitive variables
void applyPrimitiveBoundaryConditions(float *d_u, float *d_v, float *d_p, float *d_c,
                                   SimParams params) {
    int Nx = params.Nx;
    int Ny = params.Ny;
    
    // Set up grid and block dimensions for X boundaries
    dim3 blockDimX(256);
    dim3 gridDimX((Ny + blockDimX.x - 1) / blockDimX.x);
    
    // Set up grid and block dimensions for Y boundaries
    dim3 blockDimY(256);
    dim3 gridDimY((Nx + blockDimY.x - 1) / blockDimY.x);
    
    // Apply periodic boundary conditions in X
    bcPeriodicXKernel<<<gridDimX, blockDimX>>>(d_u, Nx, Ny);
    bcPeriodicXKernel<<<gridDimX, blockDimX>>>(d_v, Nx, Ny);
    bcPeriodicXKernel<<<gridDimX, blockDimX>>>(d_p, Nx, Ny);
    bcPeriodicXKernel<<<gridDimX, blockDimX>>>(d_c, Nx, Ny);
    
    // Apply mirror boundary conditions in Y
    bcMirrorYKernel<<<gridDimY, blockDimY>>>(d_u, Nx, Ny);
    bcMirrorYKernel<<<gridDimY, blockDimY>>>(d_p, Nx, Ny);
    bcMirrorYKernel<<<gridDimY, blockDimY>>>(d_c, Nx, Ny);
    
    // Special boundary for v (zero at walls)
    bcVKernel<<<gridDimY, blockDimY>>>(d_v, Nx, Ny);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in primitive boundary conditions: %s\n", cudaGetErrorString(err));
    }
}

// Host function to apply all boundary conditions
void applyBoundaryConditions(float *d_r, float *d_ru, float *d_rv, float *d_e,
                          SimParams params) {
    int Nx = params.Nx;
    int Ny = params.Ny;
    
    // Set up grid and block dimensions for X boundaries
    dim3 blockDimX(256);
    dim3 gridDimX((Ny + blockDimX.x - 1) / blockDimX.x);
    
    // Set up grid and block dimensions for Y boundaries
    dim3 blockDimY(256);
    dim3 gridDimY((Nx + blockDimY.x - 1) / blockDimY.x);
    
    // Apply periodic boundary conditions in X
    bcPeriodicXKernel<<<gridDimX, blockDimX>>>(d_r, Nx, Ny);
    bcPeriodicXKernel<<<gridDimX, blockDimX>>>(d_ru, Nx, Ny);
    bcPeriodicXKernel<<<gridDimX, blockDimX>>>(d_rv, Nx, Ny);
    bcPeriodicXKernel<<<gridDimX, blockDimX>>>(d_e, Nx, Ny);
    
    // Apply mirror boundary conditions in Y
    bcMirrorYKernel<<<gridDimY, blockDimY>>>(d_r, Nx, Ny);
    bcMirrorYKernel<<<gridDimY, blockDimY>>>(d_ru, Nx, Ny);
    bcMirrorYKernel<<<gridDimY, blockDimY>>>(d_e, Nx, Ny);
    
    // Special boundary for vertical velocity (v=0 at walls)
    bcVelocityYKernel<<<gridDimY, blockDimY>>>(d_rv, d_r, Nx, Ny);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in boundary conditions: %s\n", cudaGetErrorString(err));
    }
}