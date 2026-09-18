#ifndef YUTKI_CUDA_KERNELS_H
#define YUTKI_CUDA_KERNELS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Error codes returned by Yutki CUDA runtime functions */
#define YUTKI_CUDA_SUCCESS 0
#define YUTKI_CUDA_ERROR_NO_DEVICE 1
#define YUTKI_CUDA_ERROR_OUT_OF_MEMORY 2
#define YUTKI_CUDA_ERROR_INVALID_VALUE 3
#define YUTKI_CUDA_ERROR_LAUNCH_FAILED 4
#define YUTKI_CUDA_ERROR_UNINITIALIZED 5

/* Monitored timing profile for genuine GPU operations */
typedef struct {
    double h2d_ms;       /* Host-to-Device transfer time in milliseconds */
    double kernel_ms;    /* Pure GPU kernel execution time in milliseconds */
    double d2h_ms;       /* Device-to-Host transfer time in milliseconds */
    double total_ms;     /* Total end-to-end device time in milliseconds */
} YutkiCudaTiming;

/* Hardware discovery and device queries */
int yutki_cuda_device_count(int* count);
int yutki_cuda_device_get_name(int device_id, char* name, int max_len);
int yutki_cuda_device_get_info(int device_id, size_t* total_mem, int* sm_count, int* major, int* minor);

/* Device memory allocation and transfers */
int yutki_cuda_malloc(void** dev_ptr, size_t size_bytes);
int yutki_cuda_free(void* dev_ptr);
int yutki_cuda_memcpy_h2d(void* dev_dst, const void* host_src, size_t size_bytes);
int yutki_cuda_memcpy_d2h(void* host_dst, const void* dev_src, size_t size_bytes);
int yutki_cuda_memset(void* dev_ptr, int value, size_t size_bytes);
int yutki_cuda_synchronize(void);

/* Sparse Matrix Operations */
/* SpMV: d_y = alpha * A * d_x + beta * d_y */
int yutki_cuda_spmv(
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
);

/* Transpose SpMV: d_x = alpha * A^T * d_y + beta * d_x */
int yutki_cuda_spmv_transpose(
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
);

/* Vector arithmetic operations */
/* d_z = alpha * d_x + beta * d_y */
int yutki_cuda_axpby(
    int n,
    double alpha,
    const double* d_x,
    double beta,
    const double* d_y,
    double* d_z
);

/* Box projection: d_x[i] = clamp(d_x[i], d_lower[i], d_upper[i]) */
int yutki_cuda_box_project(
    int n,
    double* d_x,
    const double* d_lower,
    const double* d_upper
);

/* Inner product reduction: result = sum_i d_x[i] * d_y[i] */
int yutki_cuda_dot(
    int n,
    const double* d_x,
    const double* d_y,
    double* host_result
);

/* Euclidean (L2) norm reduction: result = sqrt(sum_i d_x[i]^2) */
int yutki_cuda_norm2(
    int n,
    const double* d_x,
    double* host_result
);

/* Infinity norm reduction: result = max_i |d_x[i]| */
int yutki_cuda_norm_inf(
    int n,
    const double* d_x,
    double* host_result
);

/* Timing query */
void yutki_cuda_get_last_timing(YutkiCudaTiming* timing);
void yutki_cuda_reset_timing(void);

#ifdef __cplusplus
}
#endif

#endif /* YUTKI_CUDA_KERNELS_H */
