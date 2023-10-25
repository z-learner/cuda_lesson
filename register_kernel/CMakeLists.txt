cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(test)

include_directories(..)
# Find CUDA package
find_package(CUDA REQUIRED)

# Add your CUDA source files
file(GLOB CUDA_SOURCES *.cu *.cc)

# Set CUDA architecture (change to match your GPU)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_86,code=sm_86)

# Add your executable and link CUDA libraries
cuda_add_executable(test ${CUDA_SOURCES})