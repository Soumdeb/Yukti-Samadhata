#ifndef YUTKI_CUDA_COMMON_CUH
#define YUTKI_CUDA_COMMON_CUH

#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "cuda_kernels.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return YUTKI_CUDA_ERROR_LAUNCH_FAILED; \
        } \
    } while (0)

#define WARP_SIZE 32
#define BLOCK_SIZE 256

/* Warp reduction for summation */
__inline__ __device__ double warp_reduce_sum(double val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

/* Warp reduction for maximum absolute value */
__inline__ __device__ double warp_reduce_max_abs(double val) {
    val = fabs(val);
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        double other = __shfl_down_sync(0xffffffff, val, offset);
        if (other > val) {
            val = other;
        }
    }
    return val;
}

#endif /* YUTKI_CUDA_COMMON_CUH */
