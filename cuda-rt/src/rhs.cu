// rhs.cu
#include <cuda_runtime.h>
#include <stdio.h> 
#include "rt_types.hpp"

// First derivative in x direction (central difference)
__device__ inline float diffX(float f_ip1, float f_im1, float dx) {
    return (f_ip1 - f_im1) / (2.0f * dx);
}

// First derivative in y direction (central difference)
__device__ inline float diffY(float f_jp1, float f_jm1, float dy) {
    return (f_jp1 - f_jm1) / (2.0f * dy);
}

// Second derivative in x direction
__device__ inline float diffX2(float f_ip1, float f_i, float f_im1, float dx2) {
    return (f_ip1 - 2.0f * f_i + f_im1) / dx2;
}

// Second derivative in y direction
__device__ inline float diffY2(float f_jp1, float f_j, float f_jm1, float dy2) {
    return (f_jp1 - 2.0f * f_j + f_jm1) / dy2;
}

// Kernel to compute the right-hand side of the equations
__global__ void computeRHSKernel(float *d_r, float *d_ru, float *d_rv, float *d_e,
                               float *d_u, float *d_v, float *d_p, float *d_c,
                               float *d_r_rhs, float *d_ru_rhs, float *d_rv_rhs, float *d_e_rhs,
                               float k1, float k2, float k3, SimParams params) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= params.Nx || j >= params.Ny) return;
    
    int ij = idx(i, j, params.Nx, params.Ny);
    int ip1j = idx(i+1, j, params.Nx, params.Ny);
    int im1j = idx(i-1, j, params.Nx, params.Ny);
    int ijp1 = idx(i, j+1, params.Nx, params.Ny);
    int ijm1 = idx(i, j-1, params.Nx, params.Ny);
    
    float dx = params.dx;
    float dy = params.dy;
    float dx2 = dx * dx;
    float dy2 = dy * dy;
    float ga = params.ga;
    
    // Local values
    float r = d_r[ij];
    float u = d_u[ij];
    float v = d_v[ij];
    float p = d_p[ij];
    
    // Density RHS:
    // ∂ρ/∂t + ∂(ρu)/∂x + ∂(ρv)/∂y = k1 * (∂²ρ/∂x² + ∂²ρ/∂y²)
    float drdt = 0.0f;
    
    // Advection terms
    drdt -= diffX(d_ru[ip1j], d_ru[im1j], dx);
    drdt -= diffY(d_rv[ijp1], d_rv[ijm1], dy);
    
    // Artificial diffusion
    drdt += k1 * (diffX2(d_r[ip1j], d_r[ij], d_r[im1j], dx2) + 
                 diffY2(d_r[ijp1], d_r[ij], d_r[ijm1], dy2));
    
    // X-momentum RHS:
    // ∂(ρu)/∂t + ∂(ρuu)/∂x + ∂(ρuv)/∂y + ∂p/∂x = k2 * ∂²(ρu)/∂x²
    float drudt = 0.0f;
    
    // Advection terms
    drudt -= diffX(d_ru[ip1j] * d_u[ip1j], d_ru[im1j] * d_u[im1j], dx);
    drudt -= diffY(d_rv[ijp1] * d_u[ijp1], d_rv[ijm1] * d_u[ijm1], dy);
    
    // Pressure gradient
    drudt -= diffX(d_p[ip1j], d_p[im1j], dx);
    
    // Artificial diffusion
    drudt += k2 * diffX2(d_ru[ip1j], d_ru[ij], d_ru[im1j], dx2);
    
    // Y-momentum RHS:
    // ∂(ρv)/∂t + ∂(ρuv)/∂x + ∂(ρvv)/∂y + ∂p/∂y = -gaρ + k2 * ∂²(ρv)/∂y²
    float drvdt = 0.0f;
    
    // Advection terms
    drvdt -= diffX(d_ru[ip1j] * d_v[ip1j], d_ru[im1j] * d_v[im1j], dx);
    drvdt -= diffY(d_rv[ijp1] * d_v[ijp1], d_rv[ijm1] * d_v[ijm1], dy);
    
    // Pressure gradient
    drvdt -= diffY(d_p[ijp1], d_p[ijm1], dy);
    
    // Gravity term
    drvdt += ga * r;
    
    // Artificial diffusion
    drvdt += k2 * diffY2(d_rv[ijp1], d_rv[ij], d_rv[ijm1], dy2);
    
    // Energy RHS:
    // ∂e/∂t + ∂(u(e+p))/∂x + ∂(v(e+p))/∂y = -gaρv + k3 * (∂²e/∂x² + ∂²e/∂y²)
    float dedt = 0.0f;
    
    // Energy flux terms
    float ep = d_e[ij] + p;
    float ep_ip1 = d_e[ip1j] + d_p[ip1j];
    float ep_im1 = d_e[im1j] + d_p[im1j];
    float ep_jp1 = d_e[ijp1] + d_p[ijp1];
    float ep_jm1 = d_e[ijm1] + d_p[ijm1];
    
    dedt -= diffX(d_u[ip1j] * ep_ip1, d_u[im1j] * ep_im1, dx);
    dedt -= diffY(d_v[ijp1] * ep_jp1, d_v[ijm1] * ep_jm1, dy);
    
    // Work due to gravity
    dedt += ga * r * v;
    
    // Artificial diffusion
    dedt += k3 * (diffX2(d_e[ip1j], d_e[ij], d_e[im1j], dx2) +
                 diffY2(d_e[ijp1], d_e[ij], d_e[ijm1], dy2));
    
    // Store RHS values
    d_r_rhs[ij] = drdt;
    d_ru_rhs[ij] = drudt;
    d_rv_rhs[ij] = drvdt;
    d_e_rhs[ij] = dedt;
}

// Host function to compute the right-hand side
void computeRightHandSide(float *d_r, float *d_ru, float *d_rv, float *d_e,
                         float *d_u, float *d_v, float *d_p, float *d_c,
                         float *d_r_rhs, float *d_ru_rhs, float *d_rv_rhs, float *d_e_rhs,
                         float k1, float k2, float k3, SimParams params) {
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((params.Nx + blockDim.x - 1) / blockDim.x,
                (params.Ny + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    computeRHSKernel<<<gridDim, blockDim>>>(d_r, d_ru, d_rv, d_e, d_u, d_v, d_p, d_c,
                                          d_r_rhs, d_ru_rhs, d_rv_rhs, d_e_rhs,
                                          k1, k2, k3, params);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in computeRightHandSide: %s\n", cudaGetErrorString(err));
    }
}