#include <stdio.h>

#define N 16

__global__ void printArray(int *arr, int *output, int *threadIdx_x, int *threadIdx_y, int *blockDim_x, int *idx) {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int bdim_x = blockDim.x;
    int idx_val = tid_x + tid_y * bdim_x;
    
    output[idx_val] = arr[idx_val];
    threadIdx_x[idx_val] = tid_x;
    threadIdx_y[idx_val] = tid_y;
    blockDim_x[idx_val] = bdim_x;
    idx[idx_val] = idx_val;
}

int main() {
    int arr[N];
    int *d_arr;
    int *d_output;
    int *d_threadIdx_x;
    int *d_threadIdx_y;
    int *d_blockDim_x;
    int *d_idx;

    int output[N];
    int threadIdx_x[N];
    int threadIdx_y[N];
    int blockDim_x[N];
    int idx[N];

    // Initialize array on CPU
    for (int i = 0; i < N; i++) {
        arr[i] = i;
    }

    // Allocate memory on GPU for arrays
    cudaMalloc((void **)&d_arr, N * sizeof(int));
    cudaMalloc((void **)&d_output, N * sizeof(int));
    cudaMalloc((void **)&d_threadIdx_x, N * sizeof(int));
    cudaMalloc((void **)&d_threadIdx_y, N * sizeof(int));
    cudaMalloc((void **)&d_blockDim_x, N * sizeof(int));
    cudaMalloc((void **)&d_idx, N * sizeof(int));

    // Copy array from CPU to GPU
    cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(2, 2);
    dim3 gridDim(2, 2);

    // Launch kernel
    printArray<<<gridDim, blockDim>>>(d_arr, d_output, d_threadIdx_x, d_threadIdx_y, d_blockDim_x, d_idx);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy output arrays from GPU to CPU
    cudaMemcpy(output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(threadIdx_x, d_threadIdx_x, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(threadIdx_y, d_threadIdx_y, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(blockDim_x, d_blockDim_x, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(idx, d_idx, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the values returned from the GPU
    printf("Printing values returned from GPU:\n");
    printf("ThreadIdx_x | ThreadIdx_y | BlockDim_x | idx | Array Element\n");
    for (int i = 0; i < N; i++) {
        printf("%12d | %12d | %10d | %3d | %13d\n", threadIdx_x[i], threadIdx_y[i], blockDim_x[i], idx[i], output[i]);
    }

    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_output);
    cudaFree(d_threadIdx_x);
    cudaFree(d_threadIdx_y);
    cudaFree(d_blockDim_x);
    cudaFree(d_idx);

    return 0;
}
