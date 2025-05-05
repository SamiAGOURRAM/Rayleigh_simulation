// rhs_shared.cu - Optimized RHS computation using shared memory
#include <cuda_runtime.h>
#include <stdio.h> 
#include "rt_types.hpp"

// Kernel to compute the right-hand side of the equations using shared memory
__global__ void computeRHSSharedKernel(float *d_r, float *d_ru, float *d_rv, float *d_e,
    float *d_u, float *d_v, float *d_p, float *d_c,
    float *d_r_rhs, float *d_ru_rhs, float *d_rv_rhs, float *d_e_rhs,
    float k1, float k2, float k3, SimParams params) {
// Block dimensions
const int blockDimX = blockDim.x;
const int blockDimY = blockDim.y;

// Thread indices
const int tx = threadIdx.x;
const int ty = threadIdx.y;

// Global indices
const int i = blockIdx.x * blockDimX + tx;
const int j = blockIdx.y * blockDimY + ty;

// Grid parameters
const int Nx = params.Nx;
const int Ny = params.Ny;
const float dx = params.dx;
const float dy = params.dy;
const float dx2 = dx * dx;
const float dy2 = dy * dy;
const float ga = params.ga;

// Shared memory arrays for each variable (including halos)
__shared__ float s_r[18][18];    // For 16x16 block
__shared__ float s_ru[18][18];
__shared__ float s_rv[18][18];
__shared__ float s_e[18][18];
__shared__ float s_u[18][18];
__shared__ float s_v[18][18];
__shared__ float s_p[18][18];

// Initialize shared memory with default values to avoid undefined behavior
s_r[ty+1][tx+1] = 0.0f;
s_ru[ty+1][tx+1] = 0.0f;
s_rv[ty+1][tx+1] = 0.0f;
s_e[ty+1][tx+1] = 0.0f;
s_u[ty+1][tx+1] = 0.0f;
s_v[ty+1][tx+1] = 0.0f;
s_p[ty+1][tx+1] = 0.0f;

// Load data into shared memory (including halos)
// Each thread loads its own point plus potentially a halo point

// Load interior points only if they are within the domain
if (i < Nx && j < Ny) {
int global_idx = idx(i, j, Nx, Ny);
s_r[ty+1][tx+1] = d_r[global_idx];
s_ru[ty+1][tx+1] = d_ru[global_idx];
s_rv[ty+1][tx+1] = d_rv[global_idx];
s_e[ty+1][tx+1] = d_e[global_idx];
s_u[ty+1][tx+1] = d_u[global_idx];
s_v[ty+1][tx+1] = d_v[global_idx];
s_p[ty+1][tx+1] = d_p[global_idx];
}

// Make sure all threads have loaded their center points
__syncthreads();

// Now load halo points, with proper boundary handling

// Left halo points - tx == 0 threads load them
if (tx == 0) {
int halo_i = i - 1;
// Handle periodic boundary at domain edge
if (halo_i < 0) halo_i = Nx - 1;

if (j < Ny) {
int global_idx = idx(halo_i, j, Nx, Ny);
s_r[ty+1][0] = d_r[global_idx];
s_ru[ty+1][0] = d_ru[global_idx];
s_rv[ty+1][0] = d_rv[global_idx];
s_e[ty+1][0] = d_e[global_idx];
s_u[ty+1][0] = d_u[global_idx];
s_v[ty+1][0] = d_v[global_idx];
s_p[ty+1][0] = d_p[global_idx];
}
}

// Right halo points - threads at the right edge load them
if (tx == blockDimX-1) {
int halo_i = i + 1;
// Handle periodic boundary at domain edge
if (halo_i >= Nx) halo_i = 0;

if (j < Ny) {
int global_idx = idx(halo_i, j, Nx, Ny);
s_r[ty+1][tx+2] = d_r[global_idx];
s_ru[ty+1][tx+2] = d_ru[global_idx];
s_rv[ty+1][tx+2] = d_rv[global_idx];
s_e[ty+1][tx+2] = d_e[global_idx];
s_u[ty+1][tx+2] = d_u[global_idx];
s_v[ty+1][tx+2] = d_v[global_idx];
s_p[ty+1][tx+2] = d_p[global_idx];
}
}

// Bottom halo points - ty == 0 threads load them
if (ty == 0) {
int halo_j = j - 1;
// Handle bottom boundary (mirror)
if (halo_j < 0) halo_j = 0;

if (i < Nx) {
int global_idx = idx(i, halo_j, Nx, Ny);
s_r[0][tx+1] = d_r[global_idx];
s_ru[0][tx+1] = d_ru[global_idx];
s_rv[0][tx+1] = 0.0f; // v=0 at bottom wall
s_e[0][tx+1] = d_e[global_idx];
s_u[0][tx+1] = d_u[global_idx];
s_v[0][tx+1] = 0.0f; // v=0 at bottom wall
s_p[0][tx+1] = d_p[global_idx];
}
}

// Top halo points - threads at the top edge load them
if (ty == blockDimY-1) {
int halo_j = j + 1;
// Handle top boundary (mirror)
if (halo_j >= Ny) halo_j = Ny - 1;

if (i < Nx) {
int global_idx = idx(i, halo_j, Nx, Ny);
s_r[ty+2][tx+1] = d_r[global_idx];
s_ru[ty+2][tx+1] = d_ru[global_idx];
s_rv[ty+2][tx+1] = 0.0f; // v=0 at top wall
s_e[ty+2][tx+1] = d_e[global_idx];
s_u[ty+2][tx+1] = d_u[global_idx];
s_v[ty+2][tx+1] = 0.0f; // v=0 at top wall
s_p[ty+2][tx+1] = d_p[global_idx];
}
}

// Synchronize again to make sure all halo points are loaded
__syncthreads();

// Compute RHS only for interior points that are within the domain
if (i < Nx && j < Ny) {
// Local indices in shared memory (including halo offset)
const int sx = tx + 1;
const int sy = ty + 1;

// Local values
float r = s_r[sy][sx];
float u = s_u[sy][sx];
float v = s_v[sy][sx];
float p = s_p[sy][sx];

// Density RHS
float drdt = 0.0f;

// Advection terms
drdt -= (s_ru[sy][sx+1] - s_ru[sy][sx-1]) / (2.0f * dx);
drdt -= (s_rv[sy+1][sx] - s_rv[sy-1][sx]) / (2.0f * dy);

// Artificial diffusion
drdt += k1 * ((s_r[sy][sx+1] - 2.0f * s_r[sy][sx] + s_r[sy][sx-1]) / dx2 + 
(s_r[sy+1][sx] - 2.0f * s_r[sy][sx] + s_r[sy-1][sx]) / dy2);

// X-momentum RHS
float drudt = 0.0f;

// Advection terms
drudt -= (s_ru[sy][sx+1] * s_u[sy][sx+1] - s_ru[sy][sx-1] * s_u[sy][sx-1]) / (2.0f * dx);
drudt -= (s_rv[sy+1][sx] * s_u[sy+1][sx] - s_rv[sy-1][sx] * s_u[sy-1][sx]) / (2.0f * dy);

// Pressure gradient
drudt -= (s_p[sy][sx+1] - s_p[sy][sx-1]) / (2.0f * dx);

// Artificial diffusion
drudt += k2 * (s_ru[sy][sx+1] - 2.0f * s_ru[sy][sx] + s_ru[sy][sx-1]) / dx2;

// Y-momentum RHS
float drvdt = 0.0f;

// Advection terms
drvdt -= (s_ru[sy][sx+1] * s_v[sy][sx+1] - s_ru[sy][sx-1] * s_v[sy][sx-1]) / (2.0f * dx);
drvdt -= (s_rv[sy+1][sx] * s_v[sy+1][sx] - s_rv[sy-1][sx] * s_v[sy-1][sx]) / (2.0f * dy);

// Pressure gradient
drvdt -= (s_p[sy+1][sx] - s_p[sy-1][sx]) / (2.0f * dy);

// Gravity term
drvdt += ga * r;

// Artificial diffusion
drvdt += k2 * (s_rv[sy+1][sx] - 2.0f * s_rv[sy][sx] + s_rv[sy-1][sx]) / dy2;

// Energy RHS
float dedt = 0.0f;

// Energy flux terms
float ep = s_e[sy][sx] + p;
float ep_ip1 = s_e[sy][sx+1] + s_p[sy][sx+1];
float ep_im1 = s_e[sy][sx-1] + s_p[sy][sx-1];
float ep_jp1 = s_e[sy+1][sx] + s_p[sy+1][sx];
float ep_jm1 = s_e[sy-1][sx] + s_p[sy-1][sx];

dedt -= (s_u[sy][sx+1] * ep_ip1 - s_u[sy][sx-1] * ep_im1) / (2.0f * dx);
dedt -= (s_v[sy+1][sx] * ep_jp1 - s_v[sy-1][sx] * ep_jm1) / (2.0f * dy);

// Work due to gravity
dedt += ga * r * v;

// Artificial diffusion
dedt += k3 * ((s_e[sy][sx+1] - 2.0f * s_e[sy][sx] + s_e[sy][sx-1]) / dx2 +
(s_e[sy+1][sx] - 2.0f * s_e[sy][sx] + s_e[sy-1][sx]) / dy2);

// Write RHS to global memory
int global_idx = idx(i, j, Nx, Ny);
d_r_rhs[global_idx] = drdt;
d_ru_rhs[global_idx] = drudt;
d_rv_rhs[global_idx] = drvdt;
d_e_rhs[global_idx] = dedt;
}
}

// Host function to compute the right-hand side using shared memory
void computeRightHandSideShared(float *d_r, float *d_ru, float *d_rv, float *d_e,
                              float *d_u, float *d_v, float *d_p, float *d_c,
                              float *d_r_rhs, float *d_ru_rhs, float *d_rv_rhs, float *d_e_rhs,
                              float k1, float k2, float k3, SimParams params) {
    // Set up grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((params.Nx + blockDim.x - 1) / blockDim.x,
                (params.Ny + blockDim.y - 1) / blockDim.y);
    
    // Launch kernel
    computeRHSSharedKernel<<<gridDim, blockDim>>>(d_r, d_ru, d_rv, d_e, d_u, d_v, d_p, d_c,
                                                d_r_rhs, d_ru_rhs, d_rv_rhs, d_e_rhs,
                                                k1, k2, k3, params);
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error in computeRightHandSideShared: %s\n", cudaGetErrorString(err));
    }
}