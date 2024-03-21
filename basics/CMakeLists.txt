cmake_minimum_required(VERSION 3.10)
project(CUDA_Basics CUDA)

enable_language(CXX)

# CUDA settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

# Include directory for header-only files
include_directories(utils)

# Executable for vector addition
add_executable(vector_addition vector_addition.cu)

# Set CUDA architecture
set_property(TARGET vector_addition PROPERTY CUDA_ARCHITECTURES 35 37 50 52 60 61 70 75)

# Link CUDA libraries
target_link_libraries(vector_addition CUDA::cudart)

# Find glog package
find_package(glog REQUIRED)
target_link_libraries(vector_addition glog::glog)

# Executable for regression
add_executable(sum_vector sum_vector.cu)

# Set CUDA architecture for regression
set_property(TARGET sum_vector PROPERTY CUDA_ARCHITECTURES 35 37 50 52 60 61 70 75)

# Link CUDA libraries for regression
target_link_libraries(sum_vector CUDA::cudart)

# Link glog for regression
target_link_libraries(sum_vector glog::glog)
