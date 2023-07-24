#include <stdio.h>

__global__ void my_first_kernel() {
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int bid = blockIdx.x + blockIdx.y * gridDim.x;
  printf("Hello World From Gpu (%d, %d)\n", bid, tid);
}

int main() {
  printf("Hello World From Cpu\n");
  my_first_kernel<<<dim3(2, 3), dim3(4, 5)>>>();
  cudaDeviceSynchronize();
  return 0;
}