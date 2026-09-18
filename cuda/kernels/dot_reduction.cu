#include "cuda_common.cuh"

__global__ void dot_kernel(
    int n,
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ block_sums
) {
    __shared__ double sdata[BLOCK_SIZE / WARP_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    while (idx < n) {
        sum += x[idx] * y[idx];
        idx += stride;
    }

    /* Warp-level reduction */
    sum = warp_reduce_sum(sum);

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();

    /* First warp reduces the warp results */
    if (threadIdx.x < (BLOCK_SIZE / WARP_SIZE)) {
        double warp_sum = sdata[threadIdx.x];
        warp_sum = warp_reduce_sum(warp_sum);
        if (threadIdx.x == 0) {
            atomicAdd(block_sums, warp_sum);
        }
    }
}

extern "C" int yutki_cuda_dot(
    int n,
    const double* d_x,
    const double* d_y,
    double* host_result
) {
    if (n <= 0) {
        *host_result = 0.0;
        return YUTKI_CUDA_SUCCESS;
    }

    double* d_sum = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    dot_kernel<<<num_blocks, BLOCK_SIZE>>>(n, d_x, d_y, d_sum);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(host_result, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sum));

    return YUTKI_CUDA_SUCCESS;
}
