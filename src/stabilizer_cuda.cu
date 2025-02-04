#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <iostream> 

// CUDA kernel for box filter convolution
__global__ void boxFilterKernel(const float* padded, float* result, int r, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float sum = 0.0f;
        for (int k = 0; k < r; ++k) {
            sum += padded[i + k];
        }
        result[i] = sum / r;
    }
}

std::vector<float> boxFilterConvolveCUDA(const std::vector<float> &path, int r) {
    if (path.empty()) return {};

    // Compute median on host
    std::vector<float> tmp = path;
    std::nth_element(tmp.begin(), tmp.begin() + tmp.size()/2, tmp.end());
    float med = tmp[tmp.size()/2];

    // Create padded array
    std::vector<float> padded(path.size() + 2*r, med);
    for(size_t i = 0; i < path.size(); ++i) {
        padded[r + i] = path[i];
    }

    // Allocate device memory
    float *d_padded = nullptr;
    float *d_result = nullptr;
    size_t padded_size = padded.size() * sizeof(float);
    size_t result_size = path.size() * sizeof(float);

    cudaMalloc(&d_padded, padded_size);
    cudaMalloc(&d_result, result_size);

    // Copy padded data to device
    cudaMemcpy(d_padded, padded.data(), padded_size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (path.size() + threadsPerBlock - 1) / threadsPerBlock;
    boxFilterKernel<<<blocksPerGrid, threadsPerBlock>>>(d_padded, d_result, r, path.size());

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        cudaFree(d_padded);
        cudaFree(d_result);
        return {};
    }

    // Copy result back to host
    std::vector<float> result(path.size());
    cudaMemcpy(result.data(), d_result, result_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_padded);
    cudaFree(d_result);

    return result;
}

