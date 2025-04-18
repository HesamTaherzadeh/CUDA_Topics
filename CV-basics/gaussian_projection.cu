#include "cuda_runtime.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>

#define IDX3(i, j) ((i) * 4 + (j))  
#define IDX2(i, j) ((i) * 2 + (j))

__device__ void matMul(const float* A, int rows_A, int cols_A,
            const float* B, int rows_B, int cols_B,
            float* C) {
    for (int i = 0; i < rows_A; ++i) {
        for (int j = 0; j < cols_B; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < cols_A; ++k) {
                sum += A[i * cols_A + k] * B[k * cols_B + j];
            }
            C[i * cols_B + j] = sum;
        }
    }
}

__device__ void transpose(const float* input, int rows, int cols, float* output) {
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}


__global__ void project3DGaussiansTo2D(const int N, const float* d_mu3, const float* d_sigma3, const float* d_K, const float* d_Rt, float* d_mu2, float* d_sigma2){
    extern __shared__ float shmem[];
    int idx{threadIdx.x + blockDim.x * blockIdx.x};
    float* shared_mu3 = shmem;
    float* shared_sigma3 = shared_mu3 + blockDim.x * 3;

    if (idx >= N) return;

    for (int i = 0; i < 3; ++i)
        shared_mu3[threadIdx.x * 3 + i] = d_mu3[idx * 3 + i];

    for (int i = 0; i < 9; ++i)
        shared_sigma3[threadIdx.x * 9 + i] = d_sigma3[idx * 9 + i];

    __syncthreads();

    float mu3[3];
    float sigma3[9];

    for (int i = 0; i < 3; ++i)
        mu3[i] = shared_mu3[threadIdx.x * 3 + i];

    for (int i = 0; i < 9; ++i)
        sigma3[i] = shared_sigma3[threadIdx.x * 9 + i];

    // x = RX + t
    float X_cam[3] = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            X_cam[i] += d_Rt[IDX3(i,j)] * mu3[j];
        X_cam[i] += d_Rt[i * 4 + 3];
    }

    constexpr float epsilon = 1e-6f;

    // u = (fx * x / z) + cx - v = (fy * y / z) + cy
    float u = (d_K[0] * X_cam[0] + d_K[2] * X_cam[2]) /(X_cam[2] + epsilon);
    float v = (d_K[4] * X_cam[1] + d_K[5] * X_cam[2]) / (X_cam[2] + epsilon);

    d_mu2[idx * 2 + 0] = u;
    d_mu2[idx * 2 + 1] = v;

    float fx{d_K[0]}; float fy{d_K[4]}; 
    float z_inv{1.0f / (X_cam[2] + epsilon)};

    float z_inv2{z_inv * z_inv};

    float J[6] = {
        fx * z_inv,            0,             -fx * X_cam[0] * z_inv2,
        0,              fy * z_inv,            -fy * X_cam[1] * z_inv2
    };

    float Sigma3_cam[9];

    float R[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            R[3 * i + j] = d_Rt[IDX3(i, j)];
        }
    }

    float intermediate[9];
    matMul(R, 3, 3, sigma3, 3, 3, intermediate);

    float R_transpose[9];
    transpose(R, 3, 3, R_transpose);

    matMul(intermediate, 3, 3, R_transpose, 3, 3, Sigma3_cam);

    float jcov3d[6]; float Jt[6];
    matMul(J, 2, 3, Sigma3_cam, 3, 3, jcov3d);  // J is 2x3
    
    transpose(J, 2, 3, Jt);
    matMul(jcov3d, 2, 3, Jt, 3, 2, &d_sigma2[idx * 4]);

}

int main(int argc, char** argv){

    const int N = 10000;               
    const float radius = 13.0f;      
    const float z_height = 1.0f;    

    std::vector<float> h_mu3(N * 3);
    std::vector<float> h_sigma3(N * 9);

    for (int i = 0; i < N; ++i) {
        float angle = 2.0f * M_PI * i / N;
        float x = radius * cosf(angle);
        float y = radius * sinf(angle);
        float z = z_height;

        h_mu3[i * 3 + 0] = x;
        h_mu3[i * 3 + 1] = y;
        h_mu3[i * 3 + 2] = z;
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < 9; ++j)
            h_sigma3[i * 9 + j] = 0.001f;
        h_sigma3[i * 9 + 0] = 0.1f;  
        h_sigma3[i * 9 + 4] = 0.1f;  
        h_sigma3[i * 9 + 8] = 0.1f;  
    }

    float h_K[9] = {
        5.0f, 0.0f, 240.0f,
        0.0f, 10.0f, 240.0f,
        0.0f, 0.0f, 1.0f
    };

    float h_Rt[12] = {
        1, 0, 0, 0,
        0, 0.5, 0, 0.3,
        0, 0, 0.7, 0.8
    };


    float* d_mu3; float* d_sigma3; float* d_K; float* d_Rt;
    float* d_mu2; float* d_sigma2;
    cudaMalloc(&d_mu3, N * 3 * sizeof(float));
    cudaMalloc(&d_sigma3, N * 9 * sizeof(float));
    cudaMalloc(&d_K, 9 * sizeof(float));
    cudaMalloc(&d_Rt, 12 * sizeof(float));
    cudaMalloc(&d_mu2, N * 2 * sizeof(float));
    cudaMalloc(&d_sigma2, N * 4 * sizeof(float));

    cudaMemcpy(d_mu3, h_mu3.data(), N * 3 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigma3, h_sigma3.data(), N * 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, 9 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Rt, h_Rt, 12 * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int numBlocks = (N + blockSize - 1) / blockSize;
    int sharedMemSize = (3 + 9) * blockSize * sizeof(float);

    project3DGaussiansTo2D<<<numBlocks, blockSize, sharedMemSize>>>(
    N, d_mu3, d_sigma3, d_K, d_Rt, d_mu2, d_sigma2
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel failed: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
    std::vector<float> h_mu2(N * 2);
    std::vector<float> h_sigma2(N * 4);
    cudaMemcpy(h_mu2.data(), d_mu2, N * 2 * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sigma2.data(), d_sigma2, N * 4 * sizeof(float), cudaMemcpyDeviceToHost);

    cv::Mat canvas(480, 480, CV_8UC3, cv::Scalar(255, 255, 255));

    for (int i = 0; i < N; ++i) {
        float u = h_mu2[i * 2 + 0];
        float v = h_mu2[i * 2 + 1];

        if (u < 0 || u >= canvas.cols || v < 0 || v >= canvas.rows)
            continue; 

        cv::circle(canvas, cv::Point2f(u, v), 2, cv::Scalar(0, 0, 255), -1);

        cv::Mat cov = (cv::Mat_<float>(2, 2) <<
            h_sigma2[i * 4 + 0], h_sigma2[i * 4 + 1],
            h_sigma2[i * 4 + 2], h_sigma2[i * 4 + 3]);

        cv::Mat eigvals, eigvecs;
        cv::eigen(cov, eigvals, eigvecs);

        float l1 = eigvals.at<float>(0);
        float l2 = eigvals.at<float>(1);
        if (l1 <= 0 || l2 <= 0 || std::isnan(l1) || std::isnan(l2))
            continue;

        float angle = std::atan2(eigvecs.at<float>(0,1), eigvecs.at<float>(0,0)) * 180.0 / CV_PI;
        float major = std::sqrt(l1) * 3.0f;
        float minor = std::sqrt(l2) * 3.0f;

        if (major > 1000 || minor > 300) {
            std::cout << "Skipping big ellipse: i=" << i << " major=" << major << " minor=" << minor << std::endl;
            continue;
        }


        cv::RNG rng(cv::getTickCount());  

        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));

        cv::ellipse(canvas, cv::Point2f(u, v), cv::Size(major, minor),
                    angle, 0, 360, color, 1);

    }

    cv::imshow("Projected 2D Gaussians", canvas);
    cv::waitKey(0);

    cudaFree(d_mu3); cudaFree(d_sigma3); cudaFree(d_K); cudaFree(d_Rt);
    cudaFree(d_mu2); cudaFree(d_sigma2);



}