// CUDA 3D to 2D Gaussian Projection with Shared Memory
// This example assumes a pinhole camera model and projects 3D Gaussian means and covariances into 2D screen space.
// Author: ChatGPT

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cmath>

#define IDX3(i, j) ((i) * 3 + (j))
#define IDX2(i, j) ((i) * 2 + (j))

// -------------------------------
// Mathematical Overview
// -------------------------------
// Given a 3D Gaussian with mean \( \mu_3 \in \mathbb{R}^3 \) and covariance \( \Sigma_3 \in \mathbb{R}^{3\times3} \),
// we want to project it to a 2D Gaussian \( \mu_2, \Sigma_2 \) under a camera with intrinsics K and extrinsics [R|t].
//
// The projection of the mean is:
//   \( \mu_2 = \pi(K [R|t] \mu_3) \)
// where \( \pi \) is the perspective division: \( (x, y, z) \to (fx * x / z + cx, fy * y / z + cy) \).
//
// The projected covariance is:
//   \( \Sigma_2 = J \Sigma_3 J^T \),
// where J is the Jacobian of the projection function with respect to \( \mu_3 \).

// -------------------------------
// Shared Memory Explanation
// -------------------------------
// We use shared memory to load per-thread 3D means and covariance matrices into fast-access memory local to each CUDA block.
// Each thread copies its own Gaussian into shared memory. This reduces global memory latency during projection computation.
// The shared memory layout is:
//   - shared_mu3: blockDim.x * 3 floats for 3D means
//   - shared_sigma3: blockDim.x * 9 floats for 3x3 covariances

// -------------------------------
// Kernel
// -------------------------------
__global__ void project3DGaussiansTo2D(
    const float* d_mu3,       // [N x 3] array of 3D means
    const float* d_sigma3,    // [N x 9] array of 3x3 covariances
    const float* d_K,         // [3 x 3] camera intrinsics
    const float* d_Rt,        // [3 x 4] camera extrinsics (rotation + translation)
    float* d_mu2,             // [N x 2] output 2D projected means
    float* d_sigma2           // [N x 4] output 2D covariances (2x2 row-major)
) {
    extern __shared__ float shared[];
    float* shared_mu3 = shared;
    float* shared_sigma3 = shared_mu3 + blockDim.x * 3;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= gridDim.x * blockDim.x) return;

    // Load mean
    for (int i = 0; i < 3; ++i)
        shared_mu3[threadIdx.x * 3 + i] = d_mu3[idx * 3 + i];

    // Load 3x3 covariance
    for (int i = 0; i < 9; ++i)
        shared_sigma3[threadIdx.x * 9 + i] = d_sigma3[idx * 9 + i];

    __syncthreads();

    // Access local copies
    float mu3[3];
    float sigma3[9];

    for (int i = 0; i < 3; ++i)
        mu3[i] = shared_mu3[threadIdx.x * 3 + i];

    for (int i = 0; i < 9; ++i)
        sigma3[i] = shared_sigma3[threadIdx.x * 9 + i];

    // Apply extrinsics (R * X + t)
    float X_cam[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            X_cam[i] += d_Rt[IDX3(i,j)] * mu3[j];
        X_cam[i] += d_Rt[i * 4 + 3];
    }

    // Perspective projection
    float u = (d_K[0] * X_cam[0] + d_K[2] * X_cam[2]) / X_cam[2];
    float v = (d_K[4] * X_cam[1] + d_K[5] * X_cam[2]) / X_cam[2];

    d_mu2[idx * 2 + 0] = u;
    d_mu2[idx * 2 + 1] = v;

    // Jacobian J = d(Proj(X)) / dX at X = mu3
    float fx = d_K[0];
    float fy = d_K[4];
    float z_inv = 1.0f / X_cam[2];
    float z_inv2 = z_inv * z_inv;

    float J[6] = {
        fx * z_inv,            0,             -fx * X_cam[0] * z_inv2,
        0,              fy * z_inv,            -fy * X_cam[1] * z_inv2
    }; // [2x3]

    // Project covariance: Sigma2 = J * Sigma3 * J^T
    float temp[6] = {0}; // J * Sigma3 [2x3]
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                temp[i * 3 + j] += J[IDX2(i,k)] * sigma3[IDX3(k,j)];
            }
        }
    }

    float sigma2[4] = {0}; // [2x2]
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            for (int k = 0; k < 3; ++k) {
                sigma2[IDX2(i,j)] += temp[i * 3 + k] * J[IDX2(j,k)];
            }
        }
    }

    // Store 2D covariance
    for (int i = 0; i < 4; ++i)
        d_sigma2[idx * 4 + i] = sigma2[i];
}

// -------------------------------
// Host Main Function
// -------------------------------
int main() {
    const int N = 1;

    float h_mu3[N * 3] = {1.0f, 1.0f, 5.0f};
    float h_sigma3[N * 9] = {
        0.1f, 0.0f, 0.0f,
        0.0f, 0.1f, 0.0f,
        0.0f, 0.0f, 0.1f
    };

    float h_K[9] = {
        1000.0f, 0.0f, 320.0f,
        0.0f, 1000.0f, 240.0f,
        0.0f, 0.0f, 1.0f
    };

    float h_Rt[12] = {
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0
    };

    float* d_mu3; float* d_sigma3; float* d_K; float* d_Rt;
    float* d_mu2; float* d_sigma2;
    cudaMalloc(&d_mu3, N * 3 * sizeof(float));
    cudaMalloc(&d_sigma3, N * 9 * sizeof(float));
    cudaMalloc(&d_K, 9 * sizeof(float));
    cudaMalloc(&d_Rt, 12 * sizeof(float));
    cudaMalloc(&d_mu2, N * 2 * sizeof(float));
    cudaMalloc(&d_sigma2, N * 4 * sizeof(float));

    cudaMemcpy(d_mu3, h_mu3, N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma3, h_sigma3, N * 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rt, h_Rt, 12 * sizeof(float), cudaMemcpyHostToDevice);

    project3DGaussiansTo2D<<<1, N, N * (3 + 9) * sizeof(float)>>>(d_mu3, d_sigma3, d_K, d_Rt, d_mu2, d_sigma2);

    float h_mu2[2];
    float h_sigma2[4];
    cudaMemcpy(h_mu2, d_mu2, 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sigma2, d_sigma2, 4 * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Projected Mean: (" << h_mu2[0] << ", " << h_mu2[1] << ")\n";
    std::cout << "Projected Covariance: [" << h_sigma2[0] << ", " << h_sigma2[1] << "; "
              << h_sigma2[2] << ", " << h_sigma2[3] << "]\n";

    cudaFree(d_mu3); cudaFree(d_sigma3); cudaFree(d_K); cudaFree(d_Rt);
    cudaFree(d_mu2); cudaFree(d_sigma2);
    return 0;
}
