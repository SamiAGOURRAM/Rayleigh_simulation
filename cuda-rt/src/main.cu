// main.cu
#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <stdio.h> 

#include "rt_types.hpp"

// Forward declarations of kernel functions
void initSimulation(float *d_r, float *d_ru, float *d_rv, float *d_e,
                    float *d_u, float *d_v, float *d_p, float *d_c,
                    float *d_r_rhs, float *d_ru_rhs, float *d_rv_rhs, float *d_e_rhs,
                    SimParams params);
                    
void applyBoundaryConditions(float *d_r, float *d_ru, float *d_rv, float *d_e,
                             SimParams params);
                             
void computePrimitiveVariables(float *d_r, float *d_ru, float *d_rv, float *d_e,
                               float *d_u, float *d_v, float *d_p, float *d_c,
                               SimParams params);
                               
float computeTimeStep(float *d_u, float *d_v, float *d_c, float *d_maxWave,
                      SimParams params);
                      
void computeRightHandSide(float *d_r, float *d_ru, float *d_rv, float *d_e,
                          float *d_u, float *d_v, float *d_p, float *d_c,
                          float *d_r_rhs, float *d_ru_rhs, float *d_rv_rhs, float *d_e_rhs,
                          float k1, float k2, float k3, SimParams params);
                          
void updateSolution(float *d_r, float *d_ru, float *d_rv, float *d_e,
                    float *d_r_rhs, float *d_ru_rhs, float *d_rv_rhs, float *d_e_rhs,
                    float dt, SimParams params);

// CUDA error checking
#define CHECK_CUDA_ERROR(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Function to write density field to VTK file
void writeVTKFile(float *h_r, int step, SimParams params) {
    char filename[256];
    sprintf(filename, "rt_%04d.vtk", step);
    
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open output file: " << filename << std::endl;
        return;
    }
    
    // VTK file header
    outfile << "# vtk DataFile Version 3.0\n";
    outfile << "Rayleigh-Taylor instability simulation\n";
    outfile << "ASCII\n";
    outfile << "DATASET STRUCTURED_POINTS\n";
    outfile << "DIMENSIONS " << params.Nx << " " << params.Ny << " 1\n";
    outfile << "ORIGIN 0 0 0\n";
    outfile << "SPACING " << params.dx << " " << params.dy << " 1\n";
    outfile << "POINT_DATA " << params.Nx * params.Ny << "\n";
    outfile << "SCALARS density float 1\n";
    outfile << "LOOKUP_TABLE default\n";
    
    // Write density values (skip ghost cells)
    for (int j = 0; j < params.Ny; j++) {
        for (int i = 0; i < params.Nx; i++) {
            outfile << h_r[idx(i, j, params.Nx, params.Ny)] << "\n";
        }
    }
    
    outfile.close();
}

int main(int argc, char **argv) {
    // Default simulation parameters
    SimParams params;
    params.Nx = 1024;
    params.Ny = 512;
    params.Lx = 1.0;
    params.Ly = 1.0;
    params.CFL = 0.2;
    params.ga = -10.0;
    params.gamma = 1.4;
    params.maxSteps = 1000;
    params.dumpFrequency = 100;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i += 2) {
        std::string arg = argv[i];
        if (arg == "-nx" && i+1 < argc) params.Nx = std::stoi(argv[i+1]);
        else if (arg == "-ny" && i+1 < argc) params.Ny = std::stoi(argv[i+1]);
        else if (arg == "-steps" && i+1 < argc) params.maxSteps = std::stoi(argv[i+1]);
        else if (arg == "-dump" && i+1 < argc) params.dumpFrequency = std::stoi(argv[i+1]);
        else {
            std::cerr << "Unknown parameter: " << arg << std::endl;
            return 1;
        }
    }
    
    // Calculate grid spacing
    params.dx = params.Lx / params.Nx;
    params.dy = params.Ly / params.Ny;
    
    std::cout << "Rayleigh-Taylor simulation with " << params.Nx << "x" << params.Ny 
              << " grid points" << std::endl;
    
    // Allocate memory on the device
    size_t size = (params.Nx + 2) * (params.Ny + 2) * sizeof(float);
    
    float *d_r, *d_ru, *d_rv, *d_e;           // Conservative variables
    float *d_u, *d_v, *d_p, *d_c;             // Primitive variables
    float *d_r_rhs, *d_ru_rhs, *d_rv_rhs, *d_e_rhs;  // RHS arrays
    float *d_maxWave;                         // For CFL calculation
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_r, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ru, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_rv, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_e, size));
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_u, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_v, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_p, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, size));
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_r_rhs, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ru_rhs, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_rv_rhs, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_e_rhs, size));
    
    // For CFL reduction
    CHECK_CUDA_ERROR(cudaMalloc(&d_maxWave, 1024 * sizeof(float)));
    
    // Buffer for output (only density)
    float *h_r = new float[(params.Nx + 2) * (params.Ny + 2)];
    
    // Initialize simulation
    initSimulation(d_r, d_ru, d_rv, d_e, d_u, d_v, d_p, d_c, 
                   d_r_rhs, d_ru_rhs, d_rv_rhs, d_e_rhs, params);
                   
    // Time stepping loop
    float t = 0.0;
    float dt = 0.0;
    
    // Record start time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    for (int step = 0; step < params.maxSteps; step++) {
        // Apply boundary conditions
        applyBoundaryConditions(d_r, d_ru, d_rv, d_e, params);
        
        // Calculate primitive variables
        computePrimitiveVariables(d_r, d_ru, d_rv, d_e, d_u, d_v, d_p, d_c, params);
        
        // Compute time step from CFL condition
        dt = computeTimeStep(d_u, d_v, d_c, d_maxWave, params);
        
        // Update artificial diffusion coefficients
        float dx2 = params.dx * params.dx;
        float k1 = 0.0125f * dx2 / (2.0f * dt);  // For density
        float k2 = 0.125f * dx2 / (2.0f * dt);   // For momentum (10× stronger)
        float k3 = 0.0125f * dx2 / (2.0f * dt);  // For energy
        
        // Compute RHS terms
        computeRightHandSide(d_r, d_ru, d_rv, d_e, d_u, d_v, d_p, d_c,
                            d_r_rhs, d_ru_rhs, d_rv_rhs, d_e_rhs,
                            k1, k2, k3, params);
        
        // Update solution
        updateSolution(d_r, d_ru, d_rv, d_e, d_r_rhs, d_ru_rhs, d_rv_rhs, d_e_rhs, dt, params);
        
        // Output if needed
        if (step % params.dumpFrequency == 0) {
            std::cout << "Step " << step << ", t = " << t << ", dt = " << dt << std::endl;
            
            // Copy density to host for output
            CHECK_CUDA_ERROR(cudaMemcpy(h_r, d_r, size, cudaMemcpyDeviceToHost));
            writeVTKFile(h_r, step, params);
        }
        
        t += dt;
    }
    
    // Record end time and calculate total elapsed time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    std::cout << "Simulation completed in " << milliseconds / 1000.0 << " seconds" << std::endl;
    std::cout << "Average time per step: " << milliseconds / params.maxSteps << " ms" << std::endl;
    
    // Clean up
    delete[] h_r;
    
    cudaFree(d_r);
    cudaFree(d_ru);
    cudaFree(d_rv);
    cudaFree(d_e);
    
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_p);
    cudaFree(d_c);
    
    cudaFree(d_r_rhs);
    cudaFree(d_ru_rhs);
    cudaFree(d_rv_rhs);
    cudaFree(d_e_rhs);
    
    cudaFree(d_maxWave);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}