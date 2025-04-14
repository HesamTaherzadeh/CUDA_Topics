#include <stdio.h>
#include <random>
#include <vector>
#include <iostream>

__global__ void single_step_gradient_descent(float *x, float *y, float *grad_w, float *grad_b, int N, float bias, float weight) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < N) {
        float pred = weight * x[idx] + bias;
        float diff = pred - y[idx];
        grad_w[idx] = 2.0f * diff * x[idx];
        grad_b[idx] = 2.0f * diff;
    }
}


void generate_data(float *x, float *y, const int sample_size){
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<float> dist_x(0.0f, 1.0f);
    std::uniform_real_distribution<float> noise(-0.02f, 0.02f);

    for (int i = 0; i < sample_size; ++i) {
        x[i] = dist_x(gen);
        y[i] = 2.0f * x[i] + 1.0f + noise(gen);
    }

}

int main() {
int N = 1000000;
float h_x[N], h_y[N];  
float weight = 0.0f, bias = 0.0f, lr = 0.02f;

generate_data(h_x, h_y, N);

float *d_x, *d_y, *d_grad_w, *d_grad_b;

cudaMalloc(&d_x, N * sizeof(float));
cudaMalloc(&d_y, N * sizeof(float));
cudaMalloc(&d_grad_w, N * sizeof(float));
cudaMalloc(&d_grad_b, N * sizeof(float));

cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice);

int threadsPerBlock = 256;
int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

single_step_gradient_descent<<<blocks, threadsPerBlock>>>(d_x, d_y, d_grad_w, d_grad_b, N, bias, weight);
cudaDeviceSynchronize();

std::vector<float> grad_w(N), grad_b(N);
cudaMemcpy(grad_w.data(), d_grad_w, N * sizeof(float), cudaMemcpyDeviceToHost);
cudaMemcpy(grad_b.data(), d_grad_b, N * sizeof(float), cudaMemcpyDeviceToHost);

float sum_dw = 0, sum_db = 0;
for (int i = 0; i < N; ++i) {
    sum_dw += grad_w[i];
    sum_db += grad_b[i];
}
sum_dw /= N;
sum_db /= N;

weight -= lr * sum_dw;
bias   -= lr * sum_db;

std::cout << "Updated weight: " << weight << ", bias: " << bias << "\n";

cudaFree(d_x);
cudaFree(d_y);
cudaFree(d_grad_w);
cudaFree(d_grad_b);
    
}
