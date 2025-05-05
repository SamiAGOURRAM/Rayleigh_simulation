# CMakeLists.txt
cmake_minimum_required(VERSION 3.8)
project(cuda_rt CUDA CXX)

# Find CUDA
find_package(CUDA 12.0 REQUIRED)

# Set CUDA architecture flags - adjust based on your GPU
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_75")

# Add debug info in debug mode
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g -lineinfo")

# Add optimizations in release mode
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")

# Add include directory
include_directories(include)

# Add source files
set(SOURCES
    src/main.cu
    src/init.cu
    src/boundary.cu
    src/primitive.cu
    src/cfl.cu
    src/rhs.cu
    src/rhs_shared.cu
    src/update.cu
)

# Create executable
add_executable(rt ${SOURCES})

# Link libraries
target_link_libraries(rt ${CUDA_LIBRARIES})

# Enable separate compilation
set_property(TARGET rt PROPERTY CUDA_SEPARABLE_COMPILATION ON)