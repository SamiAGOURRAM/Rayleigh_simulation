
// rt_types.hpp
#pragma once

#include <cuda_runtime.h>

// Simulation parameters structure
struct SimParams {
    int Nx;            // Number of x grid points (interior)
    int Ny;            // Number of y grid points (interior)
    float Lx;          // Domain length in x
    float Ly;          // Domain length in y
    float dx;          // Grid spacing in x
    float dy;          // Grid spacing in y
    float CFL;         // CFL number for time stepping (typically 0.2)
    float ga;          // Gravitational acceleration (typically -10.0)
    float gamma;       // Ratio of specific heats (1.4 for air)
    int maxSteps;      // Maximum number of time steps
    int dumpFrequency; // Frequency of output dumps
};

// Inline index function for accessing 2D grid with ghost cells
__host__ __device__ inline int idx(int i, int j, int Nx, int Ny) {
    return (i+1) + (j+1)*(Nx+2);  // +1 offset for ghost cells
}