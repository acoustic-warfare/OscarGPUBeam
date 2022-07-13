#include <iostream>
#include <cuda_runtime.h>

__global__ void addGPU(unsigned int n, const float *x, const float *y, float *z) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < n; i += stride)
        z[i] = x[i] + y[i];
}

int main() {
    //error checking
    cudaError_t err = cudaSuccess;

    // init array size
    unsigned int *n;
    err = cudaMallocManaged(&n, sizeof(unsigned int));
    if (err != cudaSuccess) {
        std::cout << "Failed to allocate n on device: " << err << std::endl;
    }
    n[0] = 1 << 20;

    //init arrays
    float *x, *y, *z;
    cudaMallocManaged(&x, n[0] * sizeof(float));
    cudaMallocManaged(&y, n[0] * sizeof(float));
    cudaMallocManaged(&z, n[0] * sizeof(float));
    for (int i = 0; i < n[0]; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    unsigned int blockSize = 256;
    unsigned int NumBlocks = (n[0] + blockSize - 1) / blockSize;
    long start = clock();
    addGPU<<<NumBlocks, blockSize>>>(n[0], x, y, z);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "Failed to launch kernel: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cout << "Failed to sync threads: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
    long duration = clock() - start;


    double maxError = 0.0f;
    for (int i = 0; i < n[0]; i++) {
        maxError = fmax(maxError, fabs(z[i] - 3.0f));
    }

    std::cout << "Maximum error is: " << maxError << ", completed in " << duration << " µs" << std::endl;

    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(n);
    return 0;
}
