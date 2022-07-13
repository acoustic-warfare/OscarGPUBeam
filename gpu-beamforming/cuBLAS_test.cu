//
// Created by u080334 on 6/1/22.
//

#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <random>
#include <cassert>
#include <iostream>

/*void vector_init(cuDoubleComplex * a, int n) {
    std::random_device rd;
    for (int i = 0; i < n; i++)
        a[i] = make_cuDoubleComplex((double)(rd() % 100), (double)(rd() % 100));
}

void verify_result(const cuDoubleComplex *a, const cuDoubleComplex *b, const cuDoubleComplex *c, cuDoubleComplex factor, int n) {
    for (int i = 0; i < n; i++)
        assert(0 == cuCabs(cuCsub(c[i], cuCadd(cuCmul(factor, a[i]), b[i]))));
    std::cout << "verification success" << std::endl;
}

void run_vec() {
    // initialize variables
    int n = 1 << 10;
    size_t bytes = n * sizeof(cuDoubleComplex);

    // declare variables
    cuDoubleComplex *h_a, *h_b, *h_c, *d_a, *d_b;

    // allocate memory on host and device for variables
    h_a = (cuDoubleComplex*) malloc(bytes);
    h_b = (cuDoubleComplex*) malloc(bytes);
    h_c = (cuDoubleComplex*) malloc(bytes);
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);

    // initialise vectors
    vector_init(h_a, n);
    vector_init(h_b, n);

    // initialise cuBLAS context
    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    // copy numbers from host to device memory
    cublasSetVector(n, sizeof(cuDoubleComplex), h_a, 1, d_a, 1);
    cublasSetVector(n, sizeof(cuDoubleComplex), h_b, 1, d_b, 1);

    // use cublasSaxpy to add elements
    cuDoubleComplex scale = make_cuDoubleComplex(1.0f, 0);
    cublasZaxpy_v2(handle, n, &scale, d_a, 1, d_b, 1);

    // get result from device to host memory
    cublasGetVector(n, sizeof(cuDoubleComplex), d_b, 1, h_c, 1);

    // verify result
    verify_result(h_a, h_b, h_c, scale, n);

    // cleanup
    cublasDestroy_v2(handle);
    cudaFree(d_a);
    cudaFree(d_b);
    free(h_a);
    free(h_b);
    free(h_c);

}*/

void run_mat() {
    // initialize variables
    int N;
    std::cin >> N;
    size_t bytes = N * N * sizeof(cuDoubleComplex);
    std::random_device rd;

    // declare variables
    cuDoubleComplex *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;


    h_a = (cuDoubleComplex *) malloc(bytes);
    h_b = (cuDoubleComplex *) malloc(bytes);
    h_c = (cuDoubleComplex *) malloc(bytes);
    cudaMalloc((void**) &d_a, bytes);
    cudaMalloc((void**) &d_b, bytes);
    cudaMalloc((void**) &d_c, bytes);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            h_a[i + j * N] = make_cuDoubleComplex((double) (rd() % 9), (double) (rd() % 9));
            h_b[i + j * N] = make_cuDoubleComplex((double) (rd() % 9), (double) (rd() % 9));
        }
    }
/*
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << h_a[i + j * N].x << " + " << h_a[i + j * N].y << "i\t";
        std::cout << std::endl;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << h_b[i + j * N].x << " + " << h_b[i + j * N].y << "i\t";
        std::cout << std::endl;
    }*/

    cublasHandle_t handle;
    cublasCreate_v2(&handle);

    cublasSetMatrix(N, N, sizeof(cuDoubleComplex), h_a, N, d_a, N);
    cublasSetMatrix(N, N, sizeof(cuDoubleComplex), h_b, N, d_b, N);
    cuDoubleComplex alpha = {1, 0}, beta = {0, 0};
    cublasZgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_a, N, d_b, N, &beta, d_c, N);

    cublasGetMatrix(N, N, sizeof(cuDoubleComplex), d_c, N, h_c, N);

    /*for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            std::cout << h_c[i + j * N].x << " + " << h_c[i + j * N].y << "i\t";
        std::cout << std::endl;
    }
    cuDoubleComplex sum = {0, 0};
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++)
                sum = cuCadd(cuCmul(h_a[i + k * N], h_b[k + j * N]), sum);
            std::cout << sum.x << " + " << sum.y << "i\t";
            sum = {0, 0};
        }
        std::cout << std::endl;
    }*/

    cudaFree(&d_a);
    cudaFree(&d_b);
    cudaFree(&d_c);
    free(h_a);
    free(h_b);
    free(h_c);
}

int main() {
//    run_vec();
    run_mat();
    return 0;
}
