// cfl.cu
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include "rt_types.hpp"

// Kernel to find maximum wave speed for CFL calculation
__global__ void cflKernel(float *d_u, float *d_v, float *d_c, float *d_maxWave, 
                        int Nx, int Ny) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    int blockIdx1D = blockIdx.y * gridDim.x + blockIdx.x;
    
    // Shared memory for block reduction
    __shared__ float maxWaveSpeed[256]; // Assumes blockDim.x * blockDim.y <= 256
    
    // Initialize to small value
    maxWaveSpeed[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
    
    if (i < Nx && j < Ny) {
        int idx_ij = idx(i, j, Nx, Ny);
        
        // Calculate wave speed: |u| + c, |v| + c
        float u_speed = fabsf(d_u[idx_ij]) + d_c[idx_ij];
        float v_speed = fabsf(d_v[idx_ij]) + d_c[idx_ij];
        
        // Use maximum of u and v directions
        float wave_speed = fmaxf(u_speed, v_speed);
        
        // Store in shared memory
        maxWaveSpeed[threadIdx.y * blockDim.x + threadIdx.x] = wave_speed;
    }
    
    __syncthreads();
    
    // Reduction in shared memory to find block maximum
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if ((threadIdx.y * blockDim.x + threadIdx.x) < s) {
            maxWaveSpeed[threadIdx.y * blockDim.x + threadIdx.x] = 
                fmaxf(maxWaveSpeed[threadIdx.y * blockDim.x + threadIdx.x],
                     maxWaveSpeed[(threadIdx.y * blockDim.x + threadIdx.x) + s]);
        }
        __syncthreads();
    }
    
    // First thread writes block result to global memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        d_maxWave[blockIdx1D] = maxWaveSpeed[0];
    }
}

// Host function to compute time step
float computeTimeStep(float *d_u, float *d_v, float *d_c, float *d_maxWave,
                    SimParams params) {
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((params.Nx + blockDim.x - 1) / blockDim.x,
                (params.Ny + blockDim.y - 1) / blockDim.y);
    
    // Calculate total number of blocks
    int numBlocks = gridDim.x * gridDim.y;
    
    // Launch kernel to find maximum wave speed per block
    cflKernel<<<gridDim, blockDim>>>(d_u, d_v, d_c, d_maxWave, params.Nx, params.Ny);
    
    // Use Thrust to find maximum across all blocks
    thrust::device_ptr<float> d_maxWave_thrust(d_maxWave);
    float maxWaveSpeed = thrust::reduce(d_maxWave_thrust, d_maxWave_thrust + numBlocks, 
                                      0.0f, thrust::maximum<float>());
    
    // Ensure we have a valid maximum (non-zero)
    if (maxWaveSpeed < 1e-6f) maxWaveSpeed = 1.0f;
    
    // Calculate time step from CFL condition
    float dt = params.CFL * params.dx / maxWaveSpeed;
    
    return dt;
}