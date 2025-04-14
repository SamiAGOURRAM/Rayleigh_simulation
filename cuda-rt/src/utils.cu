// utils.cu - Utility functions for the Rayleigh-Taylor simulation
#include <iostream>
#include <fstream>
#include <string>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/reduce.h>
#include "rt_types.hpp"

// Timer class for performance measurement
class Timer {
private:
    cudaEvent_t start, stop;
    
public:
    Timer() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~Timer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void Start() {
        cudaEventRecord(start);
    }
    
    float Stop() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        return milliseconds / 1000.0f;  // Return time in seconds
    }
};

// Function to calculate total mass in the domain
float calculateTotalMass(float *d_r, SimParams params) {
    int size = (params.Nx + 2) * (params.Ny + 2);
    
    // Use Thrust to sum all density values
    thrust::device_ptr<float> d_r_thrust(d_r);
    float totalMass = thrust::reduce(d_r_thrust, d_r_thrust + size, 0.0f, thrust::plus<float>());
    
    // Scale by cell volume
    totalMass *= params.dx * params.dy;
    
    return totalMass;
}

// Function to calculate total energy in the domain
float calculateTotalEnergy(float *d_e, SimParams params) {
    int size = (params.Nx + 2) * (params.Ny + 2);
    
    // Use Thrust to sum all energy values
    thrust::device_ptr<float> d_e_thrust(d_e);
    float totalEnergy = thrust::reduce(d_e_thrust, d_e_thrust + size, 0.0f, thrust::plus<float>());
    
    // Scale by cell volume
    totalEnergy *= params.dx * params.dy;
    
    return totalEnergy;
}

// Function to find min/max density values
void findDensityExtrema(float *d_r, float &min_density, float &max_density, SimParams params) {
    int size = params.Nx * params.Ny;
    
    // Use Thrust to find min and max values (excluding ghost cells)
    thrust::device_ptr<float> d_r_thrust(d_r + (params.Nx + 2) + 1);  // Start after ghost cells
    
    min_density = *(thrust::min_element(d_r_thrust, d_r_thrust + size));
    max_density = *(thrust::max_element(d_r_thrust, d_r_thrust + size));
}

// Function to write simulation data in JSON format for testing
void writeJSONData(float *h_r, float *h_p, float total_mass, float total_energy,
                  int step, float time, SimParams params) {
    char filename[256];
    sprintf(filename, "rt_data_%04d.json", step);
    
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open JSON output file: " << filename << std::endl;
        return;
    }
    
    // Find min/max density and pressure values
    float min_r = h_r[idx(0, 0, params.Nx, params.Ny)];
    float max_r = min_r;
    float min_p = h_p[idx(0, 0, params.Nx, params.Ny)];
    float max_p = min_p;
    
    for (int j = 0; j < params.Ny; j++) {
        for (int i = 0; i < params.Nx; i++) {
            int idx_ij = idx(i, j, params.Nx, params.Ny);
            
            min_r = std::min(min_r, h_r[idx_ij]);
            max_r = std::max(max_r, h_r[idx_ij]);
            min_p = std::min(min_p, h_p[idx_ij]);
            max_p = std::max(max_p, h_p[idx_ij]);
        }
    }
    
    // Write data in JSON format
    outfile << "{\n";
    outfile << "  \"step\": " << step << ",\n";
    outfile << "  \"time\": " << time << ",\n";
    outfile << "  \"min_density\": " << min_r << ",\n";
    outfile << "  \"max_density\": " << max_r << ",\n";
    outfile << "  \"min_pressure\": " << min_p << ",\n";
    outfile << "  \"max_pressure\": " << max_p << ",\n";
    outfile << "  \"total_mass\": " << total_mass << ",\n";
    outfile << "  \"total_energy\": " << total_energy << "\n";
    outfile << "}\n";
    
    outfile.close();
}

// Enhanced VTK output function
void writeVTKFile(float *h_r, float *h_u, float *h_v, float *h_p, 
                int step, float time, SimParams params) {
    char filename[256];
    sprintf(filename, "rt_%04d.vtk", step);
    
    std::ofstream outfile(filename);
    if (!outfile.is_open()) {
        std::cerr << "Failed to open VTK output file: " << filename << std::endl;
        return;
    }
    
    // VTK file header
    outfile << "# vtk DataFile Version 3.0\n";
    outfile << "Rayleigh-Taylor instability simulation, t = " << time << "\n";
    outfile << "ASCII\n";
    outfile << "DATASET STRUCTURED_POINTS\n";
    outfile << "DIMENSIONS " << params.Nx << " " << params.Ny << " 1\n";
    outfile << "ORIGIN 0 0 0\n";
    outfile << "SPACING " << params.dx << " " << params.dy << " 1\n";
    outfile << "POINT_DATA " << params.Nx * params.Ny << "\n";
    
    // Write density
    outfile << "SCALARS density float 1\n";
    outfile << "LOOKUP_TABLE default\n";
    for (int j = 0; j < params.Ny; j++) {
        for (int i = 0; i < params.Nx; i++) {
            outfile << h_r[idx(i, j, params.Nx, params.Ny)] << "\n";
        }
    }
    
    // Write pressure
    outfile << "SCALARS pressure float 1\n";
    outfile << "LOOKUP_TABLE default\n";
    for (int j = 0; j < params.Ny; j++) {
        for (int i = 0; i < params.Nx; i++) {
            outfile << h_p[idx(i, j, params.Nx, params.Ny)] << "\n";
        }
    }
    
    // Write velocity as vectors
    outfile << "VECTORS velocity float\n";
    for (int j = 0; j < params.Ny; j++) {
        for (int i = 0; i < params.Nx; i++) {
            int idx_ij = idx(i, j, params.Nx, params.Ny);
            outfile << h_u[idx_ij] << " " << h_v[idx_ij] << " 0\n";
        }
    }
    
    outfile.close();
}