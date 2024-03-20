
# Build with CMake
```bash
mkdir build
cd build
cmake ..
make
```


# Vector Addition Benchmark with CUDA and CPU (vector_addition.cu)

This repository contains a simple program for benchmarking vector addition using both CUDA and CPU implementations. The program calculates the element-wise sum of two vectors and compares the performance of CUDA and CPU approaches.

```bash
./vector_addition
```

## How it Works

The main program consists of CUDA and CPU implementations for vector addition. It performs the following steps:

1. Initialize host vectors `h_a` and `h_b` with incremental and squared values, respectively.
2. Allocate memory for device vectors `d_a`, `d_b`, and `d_c` using `cudaMalloc`.
3. Copy host vectors to device memory using `cudaMemcpy`.
4. Launch the CUDA kernel `vectorAddCUDA` to perform vector addition on the GPU.
5. Copy the result vector `d_c` from device memory to host memory using `cudaMemcpy`.
6. Free device memory using `cudaFree`.
7. Benchmark the CPU vector addition by calling the function `vectorAddCPU`.
8. Repeat the process for different vector lengths specified in the `sizes` array.
9. Log the time taken for both CUDA and CPU implementations using Google Logging (glog).

## Benchmarking

The program benchmarks vector addition for two different vector lengths: 100 and 1,000,000,000. The timing for both CUDA and CPU implementations is logged using Google Logging.

| Vector Length    | CUDA Time (μs) | CPU Time (μs) |
|------------------|-----------------|---------------|
| 100              | 76416.838       | 0.302         |
| 1000000000       | 638450.007      | 1037285.598   |

From these results, we can observe that for smaller vector lengths (100), the CPU implementation significantly outperforms the CUDA implementation. This could be due to the overhead of transferring data to and from the GPU memory outweighing the computational gains of parallelization for such small data sizes.

However, for larger vector lengths (1000000000), the CUDA implementation outperforms the CPU implementation. This is expected as the parallel nature of CUDA allows it to handle large amounts of data more efficiently than the CPU.
