#include <stdio.h>

#define N 16

__global__ void printArray(int *arr, int *output, int *threadIdx_x, int *threadIdx_y, int *blockDim_x, int *idx, int *row_offsets, int *block_offsets, int *tids) {
    int tid = blockDim.x * threadIdx.y + threadIdx.x;

    int num_threads_in_a_block = blockDim.x * blockDim.y;
    int block_offset = blockIdx.x * num_threads_in_a_block;

    int num_threads_in_a_row = num_threads_in_a_block * gridDim.x;
    int row_offset = num_threads_in_a_row * blockIdx.y;

    int idx_val = tid + block_offset + row_offset;

    row_offsets[idx_val] = row_offset;
    block_offsets[idx_val] = block_offset;
    tids[idx_val] = tid;
    
    output[idx_val] = arr[idx_val];
    threadIdx_x[idx_val] = threadIdx.x;
    threadIdx_y[idx_val] = threadIdx.y;
    blockDim_x[idx_val] = blockDim.x;
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
    int *d_row_offsets;
    int *d_block_offsets;
    int *d_tids;

    int output[N];
    int threadIdx_x[N];
    int threadIdx_y[N];
    int blockDim_x[N];
    int idx[N];
    int row_offsets[N];
    int block_offsets[N];
    int tids[N];

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
    cudaMalloc((void **)&d_row_offsets, N * sizeof(int));
    cudaMalloc((void **)&d_block_offsets, N * sizeof(int));
    cudaMalloc((void **)&d_tids, N * sizeof(int));

    // Copy array from CPU to GPU
    cudaMemcpy(d_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(2, 2);
    dim3 gridDim(2, 2);

    // Launch kernel
    printArray<<<gridDim, blockDim>>>(d_arr, d_output, d_threadIdx_x, d_threadIdx_y, d_blockDim_x, d_idx, d_row_offsets, d_block_offsets, d_tids);

    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // Copy output arrays from GPU to CPU
    cudaMemcpy(output, d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(threadIdx_x, d_threadIdx_x, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(threadIdx_y, d_threadIdx_y, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(blockDim_x, d_blockDim_x, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(idx, d_idx, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(row_offsets, d_row_offsets, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(block_offsets, d_block_offsets, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(tids, d_tids, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the values returned from the GPU
    printf("Printing values returned from GPU:\n");
    printf("ThreadIdx_x | ThreadIdx_y | BlockDim_x | Row Offset | Block Offset | TID | Array Index | Array Element\n");
    for (int i = 0; i < N; i++) {
        printf("%12d | %12d | %10d | %10d | %12d | %3d | %12d | %13d\n", 
            threadIdx_x[i], threadIdx_y[i], blockDim_x[i], row_offsets[i], block_offsets[i], tids[i], idx[i], output[i]);
    }


    // Free device memory
    cudaFree(d_arr);
    cudaFree(d_output);
    cudaFree(d_threadIdx_x);
    cudaFree(d_threadIdx_y);
    cudaFree(d_blockDim_x);
    cudaFree(d_idx);
    cudaFree(d_row_offsets);
    cudaFree(d_block_offsets);
    cudaFree(d_tids);

    return 0;
}
