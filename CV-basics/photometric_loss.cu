#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <cuda_runtime.h>
#include <numeric>

__global__ void photometricLossKernel(
    const float *d_left,
    const float *d_right,
    float *d_loss,
    const size_t width,
    const size_t height
) {
    int idx_x = threadIdx.x + blockDim.x * blockIdx.x;
    int idx_y = threadIdx.y + blockDim.y * blockIdx.y;

    if (idx_x >= width || idx_y >= height) return;

    int idx = idx_y * width + idx_x;

    float diff = d_left[idx] - d_right[idx];
    d_loss[idx] = diff * diff; 
}


int main(int argc, char** argv){
    cv::Mat left = cv::imread("images/L.png", cv::IMREAD_COLOR);
    cv::Mat right = cv::imread("images/R.png", cv::IMREAD_COLOR);

    cv::Mat left_gray, right_gray;
    cv::cvtColor(left, left_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(right, right_gray, cv::COLOR_BGR2GRAY);

    left_gray.convertTo(left_gray, CV_32F, 1.0 / 255.0);
    right_gray.convertTo(right_gray, CV_32F, 1.0 / 255.0);

    int width{left.cols}, height{left.rows};

    float *d_left, *d_right, *d_loss;
    size_t size{width * height * sizeof(float)};

    cudaMalloc((void**)&d_left, size);
    cudaMalloc((void**)&d_right, size);
    cudaMalloc((void**)&d_loss, size);


    cudaMemcpy(d_left, left_gray.ptr<float>(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_right, right_gray.ptr<float>(), size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocks((width + 15) / 16, (height + 15) / 16);
    photometricLossKernel<<<blocks, threadsPerBlock>>>(d_left, d_right, d_loss, width, height);


    float* h_loss = new float[width * height];
    cudaMemcpy(h_loss, d_loss, size, cudaMemcpyDeviceToHost);

    float total_loss = std::accumulate(h_loss, h_loss + width * height, 0.0);
    std::cout << "Total Photometric Loss: " << total_loss / (width * height) << std::endl;

    delete[] h_loss;
    cudaFree(d_left);
    cudaFree(d_right);
    cudaFree(d_loss);

}