cmake_minimum_required(VERSION 3.10)
project(CUDA_ML_BASIC CUDA)

enable_language(CXX)
enable_language(C)

# CUDA settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA package
find_package(CUDAToolkit REQUIRED)

# Executable for vector addition
add_executable(dot_product dot_product.cu)

# Set CUDA architecture
set_property(TARGET dot_product PROPERTY CUDA_ARCHITECTURES 35 37 50 52 60 61 70 75)

# Link CUDA libraries
target_link_libraries(dot_product CUDA::cudart)

# Find glog package
find_package(glog REQUIRED)
target_link_libraries(dot_product glog::glog)


