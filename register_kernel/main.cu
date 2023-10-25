#include <array>

#include "add.hpp"

REGISTER_ADD_CUDA_KERNEL(int)
REGISTER_ADD_CUDA_KERNEL(float)

using Type = int;

int main(int argc, char** argv) {
  std::array<Type, 100> data_left = {1};
  std::array<Type, 100> data_right = {2};
  std::array<Type, 100> data_result;

  launch_kernel(data_left.data(), data_right.data(), data_result.data(), data_result.size());
}