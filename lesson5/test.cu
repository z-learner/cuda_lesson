#include <assert.h>

#include <common/cuda_helper.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>

#define BLOCK_SIZE 16

template <typename T, size_t N, size_t M>
class GpuMatrix {
 public:
  const size_t HeightDim = N;
  const size_t WidthDim = M;
  using DataType = T;

  const size_t ByteSize = HeightDim * WidthDim * sizeof(DataType);

  GpuMatrix() : _owner(true) {
    CUDA_CHECK(cudaMalloc((void**)&_data, ByteSize));
  };

  ~GpuMatrix() {
    if (_owner) {
      CUDA_CHECK(cudaFree(_data));
    }
  };

  GpuMatrix(const GpuMatrix& other) {
    _data = other._data;
    _owner = false;
  }

  GpuMatrix& operator==(const GpuMatrix& other) {
    _data = other._data;
    _owner = false;
  }

  bool CopyFromCpuData(const DataType* cpu_data, size_t size) {
    assert(size * sizeof(DataType) == ByteSize);
    CUDA_CHECK(cudaMemcpy(_data, cpu_data, size * sizeof(DataType),
                          cudaMemcpyHostToDevice));
    return true;
  }

  bool CopyToCpuData(DataType* cpu_data, size_t size) {
    assert(size * sizeof(DataType) == ByteSize);
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
  bool _owner;
};

template <typename T, size_t N, size_t M, size_t K>
__global__ void __conv(GpuMatrix<T, N, M> in, GpuMatrix<T, N, M> out,
                       GpuMatrix<T, K, K> kernel) {
  static_assert(K / 2 == 1);

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (tid_x >= M || tid_y >= N) {
    return;
  }
  // K is odd
  T temp = 0;
  int count = 0;
  for (int row = -K / 2; row < K / 2; ++row) {
    for (int col = -K / 2; col < K / 2; ++col) {
      int row_index = tid_y + row;
      int col_index = tid_x + col;
      if (row_index >= 0 && row_index < N && col_index >= 0 && col_index < M) {
        temp += kernel[row + K / 2][col + K / 2] * in[row_index][col_index];
        count++;
      }
    }
  }

  out[tid_y][tid_x] = temp / count;
}

template <typename T, size_t N, size_t M, size_t K>
__global__ void __conv_shared(GpuMatrix<T, N, M> in, GpuMatrix<T, N, M> out,
                              GpuMatrix<T, K, K> kernel) {
  static_assert(K / 2 == 1);
  __shared__ T sub_in[BLOCK_SIZE + K / 2 * 2][BLOCK_SIZE + K / 2 * 2];

  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
  int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

  sub_in[ty + K / 2][tx + K / 2] = in[tid_y][tid_x];

  // how to get margin ?

  // K is odd
  T temp = 0;
  int count = 0;
  for (int row = -K / 2; row < K / 2; ++row) {
    for (int col = -K / 2; col < K / 2; ++col) {
      int row_index = tid_y + row;
      int col_index = tid_x + col;
      if (row_index >= 0 && row_index < N && col_index >= 0 && col_index < M) {
        temp += kernel[row + K / 2][col + K / 2] * in[row_index][col_index];
        count++;
      }
    }
  }

  out[tid_y][tid_x] = temp / count;
}