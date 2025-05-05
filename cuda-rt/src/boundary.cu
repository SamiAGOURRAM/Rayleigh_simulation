#include <stdio.h>
// boundary.cu
#include <cuda_runtime.h>
#include "rt_types.hpp"

// Kernel for periodic X boundary condition (original)
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


// New function to add to boundary.cu
void applyBoundaryToPrimitives(float *d_u, float *d_v, float *d_p, float *d_c, SimParams params) {
    int Nx = params.Nx;
    int Ny = params.Ny;
    
    // Set up grid and block dimensions
    dim3 blockDimX(256);
    dim3 gridDimX((Ny + blockDimX.x - 1) / blockDimX.x);
    
    dim3 blockDimY(256);
    dim3 gridDimY((Nx + blockDimY.x - 1) / blockDimY.x);
    
    // Apply periodic boundary in X for all primitive variables
    bcPeriodicXKernel<<<gridDimX, blockDimX>>>(d_u, Nx, Ny);
    bcPeriodicXKernel<<<gridDimX, blockDimX>>>(d_v, Nx, Ny);
    bcPeriodicXKernel<<<gridDimX, blockDimX>>>(d_p, Nx, Ny);
    bcPeriodicXKernel<<<gridDimX, blockDimX>>>(d_c, Nx, Ny);
    
    // Apply appropriate Y boundaries
    bcMirrorYKernel<<<gridDimY, blockDimY>>>(d_u, Nx, Ny);
    bcVelocityYKernel<<<gridDimY, blockDimY>>>(d_v, d_r, Nx, Ny); // v=0 at walls
    bcMirrorYKernel<<<gridDimY, blockDimY>>>(d_p, Nx, Ny);
    bcMirrorYKernel<<<gridDimY, blockDimY>>>(d_c, Nx, Ny);
}

// Then call this after computing primitive variables in main.cu
computePrimitiveVariables(d_r, d_ru, d_rv, d_e, d_u, d_v, d_p, d_c, params);
applyBoundaryToPrimitives(d_u, d_v, d_p, d_c, params);