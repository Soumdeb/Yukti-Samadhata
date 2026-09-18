#include "cuda_common.cuh"

/* Vector update: z = alpha * x + beta * y */
__global__ void axpby_kernel(
    int n,
    double alpha,
    const double* __restrict__ x,
    double beta,
    const double* __restrict__ y,
    double* __restrict__ z
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = alpha * x[idx] + beta * y[idx];
    }
}

/* Coordinate-wise box projection: x = clamp(x, lower, upper) */
__global__ void box_project_kernel(
    int n,
    double* __restrict__ x,
    const double* __restrict__ lower,
    const double* __restrict__ upper
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double val = x[idx];
        double l = lower[idx];
        double u = upper[idx];
        if (val < l) {
            x[idx] = l;
        } else if (val > u) {
            x[idx] = u;
        }
    }
}

extern "C" int yutki_cuda_axpby(
    int n,
    double alpha,
    const double* d_x,
    double beta,
    const double* d_y,
    double* d_z
) {
    if (n <= 0) return YUTKI_CUDA_SUCCESS;

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    axpby_kernel<<<num_blocks, BLOCK_SIZE>>>(n, alpha, d_x, beta, d_y, d_z);

    CUDA_CHECK(cudaGetLastError());
    return YUTKI_CUDA_SUCCESS;
}

extern "C" int yutki_cuda_box_project(
    int n,
    double* d_x,
    const double* d_lower,
    const double* d_upper
) {
    if (n <= 0) return YUTKI_CUDA_SUCCESS;

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    box_project_kernel<<<num_blocks, BLOCK_SIZE>>>(n, d_x, d_lower, d_upper);

    CUDA_CHECK(cudaGetLastError());
    return YUTKI_CUDA_SUCCESS;
}
