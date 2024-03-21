#include <stdio.h>
#include "utils.hpp"

__global__ void sumVector(float *input, float *output, int N) {
    __shared__ float partialSum[256]; // Shared memory for partial sums
    int tid = threadIdx.x;
    int global_tid = blockIdx.x * blockDim.x + tid;

    partialSum[tid] = 0; // Initialize partial sum

    while (global_tid < N) {
        partialSum[tid] += input[global_tid];
        global_tid += blockDim.x * gridDim.x;
    }

    __syncthreads(); // Synchronize threads within the block

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partialSum[tid] += partialSum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = partialSum[0];
    }
}

float sumVectorCUDA(float *input, int N) {
    float *d_input, *d_output;
    float result = 0;

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    {
    Timer timer("Kernel execution Time");
    sumVector<<<gridSize, blockSize>>>(d_input, d_output, N);
    }

    float *blockSums = (float*)malloc(gridSize * sizeof(float));
    cudaMemcpy(blockSums, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < gridSize; ++i) {
        result += blockSums[i];
    }

    cudaFree(d_input);
    cudaFree(d_output);
    free(blockSums);

    return result;
}

int main() {
    google::InitGoogleLogging("VectorAdd"); 
    google::SetStderrLogging(google::GLOG_INFO);
    const int N = 1000000;
    float *input = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        input[i] = 1.0f;
    }
    {
    Timer timer("sumVectorCUDA execution Time");
          
    float sum = sumVectorCUDA(input, N);
    printf("Sum: %f\n", sum);

    }

    free(input);

    return 0;
}
