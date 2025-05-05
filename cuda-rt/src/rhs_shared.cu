// rhs_shared.cu - Optimized RHS computation using shared memory
#include <cuda_runtime.h>
#include <stdio.h> 
#include "rt_types.hpp"

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

// Shared memory - dynamically allocated to avoid size limits
extern __shared__ float shared[];

// Set up shared memory pointers
float (*s_r)[18] = (float (*)[18])&shared[0];
float (*s_ru)[18] = (float (*)[18])&shared[18*18];
float (*s_rv)[18] = (float (*)[18])&shared[2*18*18];
float (*s_e)[18] = (float (*)[18])&shared[3*18*18];
float (*s_u)[18] = (float (*)[18])&shared[4*18*18];
float (*s_v)[18] = (float (*)[18])&shared[5*18*18];
float (*s_p)[18] = (float (*)[18])&shared[6*18*18];

// Interior global idx
int global_idx = idx(i, j, Nx, Ny);

// Load interior points
if (i < Nx && j < Ny) {
s_r[ty+1][tx+1] = d_r[global_idx];
s_ru[ty+1][tx+1] = d_ru[global_idx];
s_rv[ty+1][tx+1] = d_rv[global_idx];
s_e[ty+1][tx+1] = d_e[global_idx];
s_u[ty+1][tx+1] = d_u[global_idx];
s_v[ty+1][tx+1] = d_v[global_idx];
s_p[ty+1][tx+1] = d_p[global_idx];
}
else {
// Initialize out-of-bounds points to avoid uninitialized memory
s_r[ty+1][tx+1] = 0.0f;
s_ru[ty+1][tx+1] = 0.0f;
s_rv[ty+1][tx+1] = 0.0f;
s_e[ty+1][tx+1] = 0.0f;
s_u[ty+1][tx+1] = 0.0f;
s_v[ty+1][tx+1] = 0.0f;
s_p[ty+1][tx+1] = 0.0f;
}

// Wait for all threads to load interior points
__syncthreads();

// Load halo points - left/right (x-direction)
if (tx == 0 && j < Ny) {
// Left halo
int halo_i = i - 1;
// Handle periodic boundary
if (halo_i < 0) halo_i = Nx - 1;

int halo_idx = idx(halo_i, j, Nx, Ny);
s_r[ty+1][0] = d_r[halo_idx];
s_ru[ty+1][0] = d_ru[halo_idx];
s_rv[ty+1][0] = d_rv[halo_idx];
s_e[ty+1][0] = d_e[halo_idx];
s_u[ty+1][0] = d_u[halo_idx];
s_v[ty+1][0] = d_v[halo_idx];
s_p[ty+1][0] = d_p[halo_idx];
}

if (tx == blockDimX-1 && j < Ny) {
// Right halo
int halo_i = i + 1;
// Handle periodic boundary
if (halo_i >= Nx) halo_i = 0;

int halo_idx = idx(halo_i, j, Nx, Ny);
s_r[ty+1][tx+2] = d_r[halo_idx];
s_ru[ty+1][tx+2] = d_ru[halo_idx];
s_rv[ty+1][tx+2] = d_rv[halo_idx];
s_e[ty+1][tx+2] = d_e[halo_idx];
s_u[ty+1][tx+2] = d_u[halo_idx];
s_v[ty+1][tx+2] = d_v[halo_idx];
s_p[ty+1][tx+2] = d_p[halo_idx];
}

// Load halo points - top/bottom (y-direction)
if (ty == 0 && i < Nx) {
// Bottom halo
int halo_j = j - 1;
// Mirror boundary condition at bottom wall
if (halo_j < 0) halo_j = 0;

int halo_idx = idx(i, halo_j, Nx, Ny);
s_r[0][tx+1] = d_r[halo_idx];
s_ru[0][tx+1] = d_ru[halo_idx];
s_rv[0][tx+1] = 0.0f; // v=0 at wall
s_e[0][tx+1] = d_e[halo_idx];
s_u[0][tx+1] = d_u[halo_idx];
s_v[0][tx+1] = 0.0f; // v=0 at wall
s_p[0][tx+1] = d_p[halo_idx];
}

if (ty == blockDimY-1 && i < Nx) {
// Top halo
int halo_j = j + 1;
// Mirror boundary condition at top wall
if (halo_j >= Ny) halo_j = Ny - 1;

int halo_idx = idx(i, halo_j, Nx, Ny);
s_r[ty+2][tx+1] = d_r[halo_idx];
s_ru[ty+2][tx+1] = d_ru[halo_idx];
s_rv[ty+2][tx+1] = 0.0f; // v=0 at wall
s_e[ty+2][tx+1] = d_e[halo_idx];
s_u[ty+2][tx+1] = d_u[halo_idx];
s_v[ty+2][tx+1] = 0.0f; // v=0 at wall
s_p[ty+2][tx+1] = d_p[halo_idx];
}

// Wait for all halo points to be loaded
__syncthreads();

// Compute RHS only for interior points
if (i < Nx && j < Ny) {
const int sx = tx + 1;
const int sy = ty + 1;

// Local values
float r = s_r[sy][sx];
float u = s_u[sy][sx];
float v = s_v[sy][sx];
float p = s_p[sy][sx];

// Density RHS
float drdt = 0.0f;
drdt -= (s_ru[sy][sx+1] - s_ru[sy][sx-1]) / (2.0f * dx);
drdt -= (s_rv[sy+1][sx] - s_rv[sy-1][sx]) / (2.0f * dy);
drdt += k1 * ((s_r[sy][sx+1] - 2.0f * r + s_r[sy][sx-1]) / dx2 + 
(s_r[sy+1][sx] - 2.0f * r + s_r[sy-1][sx]) / dy2);

// X-momentum RHS
float drudt = 0.0f;
drudt -= (s_ru[sy][sx+1] * s_u[sy][sx+1] - s_ru[sy][sx-1] * s_u[sy][sx-1]) / (2.0f * dx);
drudt -= (s_rv[sy+1][sx] * s_u[sy+1][sx] - s_rv[sy-1][sx] * s_u[sy-1][sx]) / (2.0f * dy);
drudt -= (s_p[sy][sx+1] - s_p[sy][sx-1]) / (2.0f * dx);
drudt += k2 * (s_ru[sy][sx+1] - 2.0f * s_ru[sy][sx] + s_ru[sy][sx-1]) / dx2;

// Y-momentum RHS
float drvdt = 0.0f;
drvdt -= (s_ru[sy][sx+1] * s_v[sy][sx+1] - s_ru[sy][sx-1] * s_v[sy][sx-1]) / (2.0f * dx);
drvdt -= (s_rv[sy+1][sx] * s_v[sy+1][sx] - s_rv[sy-1][sx] * s_v[sy-1][sx]) / (2.0f * dy);
drvdt -= (s_p[sy+1][sx] - s_p[sy-1][sx]) / (2.0f * dy);
drvdt += ga * r;  // Gravity term
drvdt += k2 * (s_rv[sy+1][sx] - 2.0f * s_rv[sy][sx] + s_rv[sy-1][sx]) / dy2;

// Energy RHS
float dedt = 0.0f;
float ep = s_e[sy][sx] + p;
float ep_ip1 = s_e[sy][sx+1] + s_p[sy][sx+1];
float ep_im1 = s_e[sy][sx-1] + s_p[sy][sx-1];
float ep_jp1 = s_e[sy+1][sx] + s_p[sy+1][sx];
float ep_jm1 = s_e[sy-1][sx] + s_p[sy-1][sx];

dedt -= (s_u[sy][sx+1] * ep_ip1 - s_u[sy][sx-1] * ep_im1) / (2.0f * dx);
dedt -= (s_v[sy+1][sx] * ep_jp1 - s_v[sy-1][sx] * ep_jm1) / (2.0f * dy);
dedt += ga * r * v;  // Work due to gravity 
dedt += k3 * ((s_e[sy][sx+1] - 2.0f * s_e[sy][sx] + s_e[sy][sx-1]) / dx2 +
(s_e[sy+1][sx] - 2.0f * s_e[sy][sx] + s_e[sy-1][sx]) / dy2);

// Write RHS to global memory
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

// Calculate shared memory size - 7 arrays of size 18×18
size_t sharedMemSize = 7 * 18 * 18 * sizeof(float);

// Launch kernel with shared memory allocation
computeRHSSharedKernel<<<gridDim, blockDim, sharedMemSize>>>(
d_r, d_ru, d_rv, d_e, d_u, d_v, d_p, d_c,
d_r_rhs, d_ru_rhs, d_rv_rhs, d_e_rhs,
k1, k2, k3, params);

// Check for errors
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
fprintf(stderr, "Error in computeRightHandSideShared: %s\n", cudaGetErrorString(err));
}
}