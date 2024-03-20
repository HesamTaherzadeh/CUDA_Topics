#include <iostream>
#include <vector>
#include "utils.hpp"

__global__ void vectorAddCUDA(int *a, int *b, int *c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

void vectorAddCPU(const int *a, const int *b, int *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    google::InitGoogleLogging("VectorAdd"); 
    google::SetStderrLogging(google::GLOG_INFO);

    const std::size_t sizes[] = {100, 1000000000};

    for (int size : sizes) {
        LOG(INFO) << "Vector Length: " << size; 

        // Host vectors
        std::vector<int> h_a(size), h_b(size), h_c(size);

        // Initialize host vectors
        for (int i = 0; i < size; ++i) {
            h_a[i] = i;
            h_b[i] = i * i;
        }

        // Timing for CUDA vector addition
        {
            Timer timer("CUDA Time");

            // Device vectors
            int *d_a, *d_b, *d_c;
            cudaMalloc((void **)&d_a, size * sizeof(int));
            cudaMalloc((void **)&d_b, size * sizeof(int));
            cudaMalloc((void **)&d_c, size * sizeof(int));

            cudaMemcpy(d_a, h_a.data(), size * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, h_b.data(), size * sizeof(int), cudaMemcpyHostToDevice);

            int threadsPerBlock = 256;
            int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
            vectorAddCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

            cudaMemcpy(h_c.data(), d_c, size * sizeof(int), cudaMemcpyDeviceToHost);
            
            cudaDeviceSynchronize();

            cudaFree(d_a);
            cudaFree(d_b);
            cudaFree(d_c);
        }

        {
            Timer timer("CPU Time");
            vectorAddCPU(h_a.data(), h_b.data(), h_c.data(), size);
        }

        LOG(INFO) << ""; 
    }

    google::ShutdownGoogleLogging(); 

    return 0;
}
