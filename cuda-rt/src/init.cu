// init.cu
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h> 
#include "rt_types.hpp"

// Kernel for initial conditions
__global__ void initICsKernel(float *d_r, float *d_ru, float *d_rv, float *d_e,
                            float *d_u, float *d_v, float *d_p, float *d_c,
                            SimParams params, unsigned long seed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= params.Nx || j >= params.Ny) return;
    
    int idx_ij = idx(i, j, params.Nx, params.Ny);
    
    // Calculate position
    float x = (i + 0.5f) * params.dx;
    float y = (j + 0.5f) * params.dy;
    
    // Initialize density: ρ = 2.0 for y ≥ Ly/2, ρ = 1.0 for y < Ly/2
    float density = (y >= params.Ly/2.0f) ? 2.0f : 1.0f;
    d_r[idx_ij] = density;
    
    // Initialize velocity: u = 0 everywhere
    d_u[idx_ij] = 0.0f;
    d_ru[idx_ij] = 0.0f;
    
    // Initialize y-velocity: v = 0 with controlled perturbation near interface
    float v = 0.0f;
    
    // Calculate horizontal distance from center
    float center_x = params.Lx / 2.0f;
    float distance_from_center = fabsf(x - center_x);
    
    // Add controlled perturbation near the interface and center of domain
    if (fabsf(y - params.Ly/2.0f) <= 0.05f) {
        // Create a bell-shaped perturbation centered in the domain
        float width = 0.2f * params.Lx; // Width of perturbation (20% of domain)
        
        if (distance_from_center < width) {
            // Stronger in center, weaker toward edges
            float amplitude = 0.002f * expf(-8.0f * distance_from_center / params.Lx);
            
            // Downward velocity perturbation (negative)
            // This helps initiate the dense fluid moving down in the center
            v = -amplitude;
            
            // Add small single-mode sinusoidal perturbation (optional)
            // This creates a more controlled wavelength
            v += -0.001f * sinf(2.0f * M_PI * x / params.Lx);
        }
    }
    
    d_v[idx_ij] = v;
    d_rv[idx_ij] = density * v;
    
    // Initialize pressure: hydrostatic equilibrium
    float p = 40.0f + density * params.ga * (y - params.Ly/2.0f);
    
    // Ensure pressure doesn't go negative
    p = fmaxf(p, 1e-6f);
    
    d_p[idx_ij] = p;
    
    // Calculate sound speed
    d_c[idx_ij] = sqrtf(params.gamma * p / density);
    
    // Calculate total energy:
    // e = p/(γ-1) + 0.5*ρ*(u²+v²)
    d_e[idx_ij] = p / (params.gamma - 1.0f) + 0.5f * density * (v*v);
}

// Host function to launch initialization kernel
void initSimulation(float *d_r, float *d_ru, float *d_rv, float *d_e,
    float *d_u, float *d_v, float *d_p, float *d_c,
    SimParams params) {
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((params.Nx + blockDim.x - 1) / blockDim.x,
                (params.Ny + blockDim.y - 1) / blockDim.y);
    
    // Set random seed
    unsigned long seed = 12345;  // Fixed seed for reproducibility
    
    // Launch kernel
    initICsKernel<<<gridDim, blockDim>>>(d_r, d_ru, d_rv, d_e, d_u, d_v, d_p, d_c, 
                                        params, seed);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in initSimulation: %s\n", cudaGetErrorString(err));
    }
}