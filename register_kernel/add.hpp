#pragma once
#include <cstdint>
#include <iostream>

#include "common/cuda_helper.hpp"

template <typename T>
__global__ void _add(T* input_left, T* input_right, T* result, std::size_t size) {
  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid_x >= size) {
    return;
  }
  result[tid_x] = input_left[tid_x] + input_right[tid_x];
}

#define REGISTER_ADD_CUDA_KERNEL(Type)                                                                  \
  void launch_kernel(Type* cpu_input_left, Type* cpu_input_right, Type* cpu_result, std::size_t size) { \
    const int bytes_size = size * sizeof(Type);                                                         \
    Type *d_x, *d_y, *d_z;                                                                              \
    CUDA_CHECK(cudaMalloc((void**)&d_x, bytes_size));                                                   \
    CUDA_CHECK(cudaMalloc((void**)&d_y, bytes_size));                                                   \
    CUDA_CHECK(cudaMalloc((void**)&d_z, bytes_size));                                                   \
    CUDA_CHECK(cudaMemcpy(d_x, cpu_input_left, bytes_size, cudaMemcpyHostToDevice));                    \
    CUDA_CHECK(cudaMemcpy(d_y, cpu_input_right, bytes_size, cudaMemcpyHostToDevice));                   \
    const int block_size = 128;                                                                         \
    const int grid_size = (size + block_size - 1) / block_size;                                         \
    _add<Type><<<grid_size, block_size>>>(d_x, d_y, d_z, size);                                         \
    CUDA_CHECK_LAST_ERROR;                                                                              \
    CUDA_CHECK(cudaMemcpy(cpu_result, d_z, bytes_size, cudaMemcpyDeviceToHost));                        \
    CUDA_CHECK(cudaFree(d_x));                                                                          \
    CUDA_CHECK(cudaFree(d_y));                                                                          \
    CUDA_CHECK(cudaFree(d_z));                                                                          \
  }
