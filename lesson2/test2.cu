
#include <iostream>

#include "common/cuda_helper.hpp"

template <typename T>
__global__ void _add_vector(const T* a, const T* b, T* c, int size) {
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  int bid = blockIdx.y * gridDim.x + blockIdx.x;

  // printf("tid = %d\n", tid);
  // printf("bid = %d\n", bid);

  int index = bid * blockDim.x * blockDim.y + tid;
  if (index >= size) {
    return;
  }

  c[index] = a[index] + b[index];
  // printf("c[%d] = %f\n", index, c[index]);
}

int main(int argc, char** argv) {
  using Type = float;

  const int N = 10000000;
  const int bytes_size = N * sizeof(Type);

  Type* h_x = new Type[N];
  Type* h_y = new Type[N];
  Type* h_z = new Type[N];

  for (int i = 0; i < N; ++i) {
    h_x[i] = i;
    h_y[i] = i - 1;
  }

  Type *d_x, *d_y, *d_z;

  CUDA_CHECK(cudaMalloc((void**)&d_x, bytes_size));
  CUDA_CHECK(cudaMalloc((void**)&d_y, bytes_size));
  CUDA_CHECK(cudaMalloc((void**)&d_z, bytes_size));

  CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes_size, cudaMemcpyHostToDevice));

  const int block_size = 128;
  const int grid_size = (N + block_size - 1) / block_size;

  std::cout << "block_size : " << block_size << std::endl;
  std::cout << "grid_size : " << grid_size << std::endl;
  _add_vector<Type><<<grid_size, block_size>>>(d_x, d_y, d_z, N);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK_LAST_ERROR;

  CUDA_CHECK(cudaMemcpy(h_z, d_z, bytes_size, cudaMemcpyDeviceToHost));

  free(h_x);
  free(h_y);
  free(h_z);

  CUDA_CHECK(cudaFree(d_x));
  CUDA_CHECK(cudaFree(d_y));
  CUDA_CHECK(cudaFree(d_z));
}