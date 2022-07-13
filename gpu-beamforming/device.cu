__global__ void
invert(const int length, const double *a, cuDoubleComplex *b) {
    if (threadIdx.x < length && fabs(a[threadIdx.x]) > 1.0e-10)
        b[threadIdx.x + threadIdx.x * length] = make_cuDoubleComplex(1 / a[threadIdx.x], 0);
}

__global__ void
get_power(const cuDoubleComplex *response, double *power_output, int &time_index) {
    power_output[time_index] = cuCreal(cuCmul(*response, cuConj(*response)));
    time_index++;
}

void cudaErrorChk(char* process) {
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Error during " << process << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}