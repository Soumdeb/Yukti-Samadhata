#include "cuda_common.cuh"

/* Scale vector x = beta * x before scatter accumulation */
__global__ void scale_vector_kernel(int n, double beta, double* __restrict__ x) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        x[idx] *= beta;
    }
}

/* Transpose SpMV kernel via CSR row traversal and atomic accumulation */
__global__ void spmv_csr_transpose_kernel(
    int num_rows,
    const int* __restrict__ row_ptrs,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    double alpha,
    const double* __restrict__ y,
    double* __restrict__ x
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < num_rows) {
        double y_val = y[row];
        if (y_val == 0.0) return;

        double scaled_y = alpha * y_val;
        int start = row_ptrs[row];
        int end = row_ptrs[row + 1];

        for (int idx = start; idx < end; ++idx) {
            int col = col_indices[idx];
            double contribution = scaled_y * values[idx];
            atomicAdd(&x[col], contribution);
        }
    }
}

extern "C" int yutki_cuda_spmv_transpose(
    int num_rows,
    int num_cols,
    int nnz,
    const int* d_row_ptrs,
    const int* d_col_indices,
    const double* d_values,
    double alpha,
    const double* d_y,
    double beta,
    double* d_x
) {
    if (num_rows <= 0 || num_cols <= 0) {
        return YUTKI_CUDA_SUCCESS;
    }

    /* 1. Handle beta scaling on x */
    if (beta == 0.0) {
        CUDA_CHECK(cudaMemset(d_x, 0, num_cols * sizeof(double)));
    } else if (beta != 1.0) {
        int blocks_scale = (num_cols + BLOCK_SIZE - 1) / BLOCK_SIZE;
        scale_vector_kernel<<<blocks_scale, BLOCK_SIZE>>>(num_cols, beta, d_x);
    }

    /* 2. Launch transpose scatter kernel */
    int num_blocks = (num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE;
    spmv_csr_transpose_kernel<<<num_blocks, BLOCK_SIZE>>>(
        num_rows, d_row_ptrs, d_col_indices, d_values, alpha, d_y, d_x
    );

    CUDA_CHECK(cudaGetLastError());
    return YUTKI_CUDA_SUCCESS;
}
