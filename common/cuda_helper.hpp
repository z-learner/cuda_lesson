/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __MY_CUDA_HELPER_H_
#define __MY_CUDA_HELPER_H_

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <exception>
#include <iostream>
#include <sstream>
#define CUDA_CHECK_LAST_ERROR                                          \
  {                                                                    \
    const cudaError_t __error = cudaGetLastError();                    \
    if (__error != cudaSuccess) {                                      \
      detail::throw_re(__FILE__, __LINE__, "[cuda] ",                  \
                       "error code : ", __error,                       \
                       ", error msg : ", cudaGetErrorString(__error)); \
    }                                                                  \
  }

#define CUDA_CHECK(call)                                               \
  {                                                                    \
    const cudaError_t __error = call;                                  \
    if (__error != cudaSuccess) {                                      \
      detail::throw_re(__FILE__, __LINE__, "[cuda] ",                  \
                       "error code : ", __error,                       \
                       ", error msg : ", cudaGetErrorString(__error)); \
    }                                                                  \
  }

namespace detail {
// Procedure: throw_se
// Throws the system error under a given error code.
template <typename... ArgsT>
// void throw_se(const char* fname, const size_t line, Error::Code c, ArgsT&&...
// args) {
void throw_re(const char* fname, const size_t line, ArgsT&&... args) {
  std::ostringstream oss;
  oss << "[" << fname << ":" << line << "] ";
  // ostreamize(oss, std::forward<ArgsT>(args)...);
  (oss << ... << args);
  throw std::runtime_error(oss.str());
}

size_t get_max_thread_pre_block(int device_id) {
  cudaDeviceProp device_prop;
  CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_id));
  return device_prop.maxThreadsPerBlock;
}

}  // namespace detail

/**
 * Execute a CUDA call and print out any errors
 * @return the original cudaError_t result
 * @ingroup cudaError
 */
#define CUDA(x) cudaCheckError((x), #x, __FILE__, __LINE__)

/**
 * Evaluates to true on success
 * @ingroup cudaError
 */
#define CUDA_SUCCESS(x) (CUDA(x) == cudaSuccess)

/**
 * Evaluates to true on failure
 * @ingroup cudaError
 */
#define CUDA_FAILED(x) (CUDA(x) != cudaSuccess)

/**
 * Return from the boolean function if CUDA call fails
 * @ingroup cudaError
 */
#define CUDA_VERIFY(x) \
  if (CUDA_FAILED(x)) return false;

/**
 * LOG_CUDA string.
 * @ingroup cudaError
 */
#define LOG_CUDA "[cuda]   "

/*
 * define this if you want all cuda calls to be printed
 * @ingroup cudaError
 */
// #define CUDA_TRACE

/**
 * cudaCheckError
 * @ingroup cudaError
 */
inline cudaError_t cudaCheckError(cudaError_t retval, const char* txt,
                                  const char* file, int line) {
#if !defined(CUDA_TRACE)
  if (retval == cudaSuccess) return cudaSuccess;
#endif

  // int activeDevice = -1;
  // cudaGetDevice(&activeDevice);

  printf(LOG_CUDA);
  if (retval == cudaSuccess)
    printf("%s\n", txt);
  else
    printf("%s\n", txt);

  if (retval != cudaSuccess) {
    printf("   %s (error %u) (hex 0x%02X)\n", cudaGetErrorString(retval),
           retval, retval);
  }

  return retval;
}

/**
 * Check for non-NULL pointer before freeing it, and then set the pointer to
 * NULL.
 * @ingroup cudaError
 */
#define CUDA_FREE(x) \
  if (x != NULL) {   \
    cudaFree(x);     \
    x = NULL;        \
  }

/**
 * Check for non-NULL pointer before freeing it, and then set the pointer to
 * NULL.
 * @ingroup cudaError
 */
#define CUDA_FREE_HOST(x) \
  if (x != NULL) {        \
    cudaFreeHost(x);      \
    x = NULL;             \
  }

/**
 * Check for non-NULL pointer before deleting it, and then set the pointer to
 * NULL.
 * @ingroup util
 */
#define SAFE_DELETE(x) \
  if (x != NULL) {     \
    delete x;          \
    x = NULL;          \
  }

/**
 * If a / b has a remainder, round up.  This function is commonly using when
 * launching CUDA kernels, to compute a grid size inclusive of the entire
 * dataset if it's dimensions aren't evenly divisible by the block size.
 *
 * For example:
 *
 *    const dim3 blockDim(8,8);
 *    const dim3 gridDim(iDivUp(imgWidth,blockDim.x),
 * iDivUp(imgHeight,blockDim.y));
 *
 * Then inside the CUDA kernel, there is typically a check that thread index is
 * in-bounds.
 *
 * Without the use of iDivUp(), if the data dimensions weren't evenly divisible
 * by the block size, parts of the data wouldn't be covered by the grid and not
 * processed.
 *
 * @ingroup cuda
 */
inline __device__ __host__ int iDivUp(int a, int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

#endif