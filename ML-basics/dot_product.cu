#include <stdio.h>

__global__ void elementwise_product(int* a, int* b, int* c, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        c[idx] = a[idx] * b[idx];
    }
}

int main() {
    int N = 5;
    int arr1[N] = {1, 2, 3, 4, 5};
    int arr2[N] = {1, 2, 6, 4, 5};
    int h_result[N];  

    int *d_arr1, *d_arr2, *d_result;

    cudaMalloc((void **)&d_arr1, N * sizeof(int));
    cudaMalloc((void **)&d_arr2, N * sizeof(int));
    cudaMalloc((void **)&d_result, N * sizeof(int)); 

    cudaMemcpy(d_arr1, arr1, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr2, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    elementwise_product<<<blocks, threadsPerBlock>>>(d_arr1, d_arr2, d_result, N);
    cudaDeviceSynchronize();  

    cudaMemcpy(h_result, d_result, N * sizeof(int), cudaMemcpyDeviceToHost);

    int dot = 0;
    for (int i = 0; i < N; i++) {
        dot += h_result[i];
    }

    printf("Dot product: %d\n", dot);

    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_result);

    return 0;
}
