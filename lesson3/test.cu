#include <iostream>

#include "common/cuda_helper.hpp"

template <typename T, size_t N, size_t M>
class GpuMatrix {
 public:
  const size_t HeightDim = N;
  const size_t WidthDim = M;
  using DataType = T;

  const size_t ByteSize = HeightDim * WidthDim * sizeof(DataType);

  GpuMatrix() { CUDA_CHECK(cudaMalloc((void**)&_data, ByteSize)); }

  ~GpuMatrix() { CUDA_CHECK(cudaFree(_data)) }

  bool CopyFromCpuData(const DataType* cpu_data, size_t size) {
    CUDA_CHECK(cudaMemcpy(_data, cpu_data, size * sizeof(DataType),
                          cudaMemcpyHostToDevice));
    return true;
  }

  bool CopyToCpuData(DataType* cpu_data, size_t size) {
    CUDA_CHECK(cudaMemcpy(cpu_data, _data, size * sizeof(DataType),
                          cudaMemcpyDeviceToHost));
    return true;
  }
  __device__ size_t GetByteSize() { return ByteSize; }

  // not saft
  __device__ DataType& at(size_t y, size_t x) {
    return _data[y * WidthDim + x];
  }

 private:
  DataType* _data;
};

template <typename T, size_t N, size_t M, size_t Z>
__global__ void _multiplication_matrix(GpuMatrix<T, N, M> x,
                                       GpuMatrix<T, M, Z> y,
                                       GpuMatrix<T, N, Z> z) {
  int tid_x = blockIdx.x * gridDim.x + threadIdx.x;
  int tid_y = blockIdx.y * gridDim.y + threadIdx.y;

  if (tid_x >= Z || tid_y >= N) {
    return;
  }

  T temp = 0;
  for (int index = 0; index < M; ++index) {
    temp += x.at(tid_y, index) * x.at(index, tid_x);
  }
  z.at(tid_y, tid_x) = temp;
  // printf("(%d, %d) = %f\n", tid_y, tid_x, z.at(tid_y, tid_x));
}

int main(int argc, char** argv) {
  constexpr size_t N = 3;
  constexpr size_t M = 4;
  constexpr size_t Z = 5;
  using DataType = float;

  DataType *h_a, *h_b, *h_c;
  h_a = new DataType[N * M];
  h_b = new DataType[M * Z];
  h_c = new DataType[N * Z];

  for (size_t x = 0; x < N; ++x) {
    for (size_t y = 0; y < M; ++y) {
      h_a[x * M + y] = x + y;
    }
  }

  for (size_t x = 0; x < M; ++x) {
    for (size_t y = 0; y < Z; ++y) {
      h_b[x * Z + y] = x + y - 1;
    }
  }

  GpuMatrix<DataType, N, M> matrix_a;
  GpuMatrix<DataType, M, Z> matrix_b;
  GpuMatrix<DataType, N, Z> matrix_c;

  // Copy To Gpu From Cpu
  matrix_a.CopyFromCpuData(h_a, N * M);
  matrix_b.CopyFromCpuData(h_b, M * Z);

  dim3 block_size(32, 32);
  dim3 grid_size((N + block_size.x - 1) / block_size.x,
                 (Z + block_size.y - 1) / block_size.y);

  _multiplication_matrix<DataType, N, M, Z>
      <<<grid_size, block_size>>>(matrix_a, matrix_b, matrix_c);

  CUDA_CHECK_LAST_ERROR;

  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy To Cpu From Gpu
  matrix_c.CopyToCpuData(h_c, N * Z);

  // cout data
  std::cout << "[";
  for (size_t x = 0; x < N; ++x) {
    for (size_t y = 0; y < M; ++y) {
      std::cout << h_a[x * M + y] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "]" << std::endl;

  std::cout << std::endl;
  std::cout << "[";
  for (size_t x = 0; x < M; ++x) {
    for (size_t y = 0; y < Z; ++y) {
      std::cout << h_b[x * Z + y] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "]" << std::endl;

  std::cout << std::endl;
  std::cout << "[";
  for (size_t x = 0; x < N; ++x) {
    for (size_t y = 0; y < Z; ++y) {
      std::cout << h_c[x * Z + y] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "]" << std::endl;
  return 0;
}