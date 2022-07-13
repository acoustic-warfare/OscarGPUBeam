cmake_minimum_required(VERSION 3.21)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.0/bin/nvcc)
set(CMAKE_CUDA_HOST_COMPILER /opt/gcc-9.3.0/bin/g++)
project(GPUBeamforming CUDA)
set(CMAKE_CUDA_STANDARD 14)

add_executable(GPUBeamforming main.cu)
add_executable(intro_test test.cu)
set(LD_LIBRARY_PATH /opt/gcc-9.3.0/lib64/)
set_target_properties(GPUBeamforming PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(GPUBeamforming -lcublas -lcusolver)

project(CPUBeamforming CXX)
add_executable(CPUBeamforming serial_main.cpp)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_COMPILER /opt/gcc-9.3.0/bin/g++)

project(cuBLASTest CUDA)
set(LD_LIBRARY_PATH /opt/gcc-9.3.0/lib64/)
set(CMAKE_CUDA_STANDARD 11)
add_executable(cuBLASTest cuBLAS_test.cu)
target_link_libraries(cuBLASTest -lcublas)
set_target_properties(cuBLASTest PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
