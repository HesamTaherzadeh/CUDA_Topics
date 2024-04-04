#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void printThreadAndBlockIndices() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
    int idz = threadIdx.z + blockIdx.z * blockDim.z;
    printf("ThreadIdx (%d, %d, %d), BlockIdx (%d, %d, %d), Total ThreadIdx (%d, %d, %d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           idx, idy, idz);
}

int main() {
    dim3 block(8, 8, 4); 
    dim3 grid(4, 4, 4); 

    printThreadAndBlockIndices<<<grid, block>>>();
    cudaDeviceSynchronize();

    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cuda_error));
    }

    cudaDeviceReset();
    return 0;
}
