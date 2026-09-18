#include "cuda_common.cuh"

/* Scalar CSR SpMV kernel: one thread per row */
__global__ void spmv_csr_scalar_kernel(
    int num_rows,
    const int* __restrict__ row_ptrs,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    double alpha,
    const double* __restrict__ x,
    double beta,
    double* __restrict__ y
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        int start = row_ptrs[row];
        int end = row_ptrs[row + 1];
        double sum = 0.0;
        for (int idx = start; idx < end; ++idx) {
            sum += values[idx] * x[col_indices[idx]];
        }
        if (beta == 0.0) {
            y[row] = alpha * sum;
        } else {
            y[row] = alpha * sum + beta * y[row];
        }
    }
}

/* Warp-per-row CSR SpMV kernel for higher non-zero densities */
__global__ void spmv_csr_vector_kernel(
    int num_rows,
    const int* __restrict__ row_ptrs,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    double alpha,
    const double* __restrict__ x,
    double beta,
    double* __restrict__ y
) {
    int global_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    if (global_warp_id < num_rows) {
        int row = global_warp_id;
        int start = row_ptrs[row];
        int end = row_ptrs[row + 1];

        double sum = 0.0;
        for (int idx = start + lane; idx < end; idx += WARP_SIZE) {
            sum += values[idx] * x[col_indices[idx]];
        }

        sum = warp_reduce_sum(sum);

        if (lane == 0) {
            if (beta == 0.0) {
                y[row] = alpha * sum;
            } else {
                y[row] = alpha * sum + beta * y[row];
            }
        }
    }
}

extern "C" int yutki_cuda_spmv(
    int num_rows,
    int num_cols,
    int nnz,
    const int* d_row_ptrs,
    const int* d_col_indices,
    const double* d_values,
    double alpha,
    const double* d_x,
    double beta,
    double* d_y
) {
    if (num_rows <= 0 || num_cols <= 0) {
        return YUTKI_CUDA_SUCCESS;
    }

    double avg_nnz_per_row = (double)nnz / (double)num_rows;

    if (avg_nnz_per_row >= 16.0) {
        /* Use warp-per-row */
        int warps_per_block = BLOCK_SIZE / WARP_SIZE;
        int num_blocks = (num_rows + warps_per_block - 1) / warps_per_block;
        spmv_csr_vector_kernel<<<num_blocks, BLOCK_SIZE>>>(
            num_rows, d_row_ptrs, d_col_indices, d_values, alpha, d_x, beta, d_y
        );
    } else {
        /* Use thread-per-row */
        int num_blocks = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
        spmv_csr_scalar_kernel<<<num_blocks, BLOCK_SIZE>>>(
            num_rows, d_row_ptrs, d_col_indices, d_values, alpha, d_x, beta, d_y
        );
    }

    CUDA_CHECK(cudaGetLastError());
    return YUTKI_CUDA_SUCCESS;
}
