// derivatives.cu
#include <cuda_runtime.h>

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