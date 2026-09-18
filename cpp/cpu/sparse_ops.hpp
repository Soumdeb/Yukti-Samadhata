#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
  #define YUTKI_EXPORT __declspec(dllexport)
#else
  #define YUTKI_EXPORT
#endif

extern "C" {

/// SpMV: y = alpha * A * x + beta * y
/// A is stored in CSR format: num_rows, num_cols, row_ptrs, col_indices, values.
YUTKI_EXPORT void yutki_cpu_spmv(
    size_t num_rows,
    size_t num_cols,
    const size_t* row_ptrs,
    const size_t* col_indices,
    const double* values,
    double alpha,
    const double* x,
    double beta,
    double* y
);

/// Transpose SpMV: x = alpha * A^T * y + beta * x
YUTKI_EXPORT void yutki_cpu_spmv_transpose(
    size_t num_rows,
    size_t num_cols,
    const size_t* row_ptrs,
    const size_t* col_indices,
    const double* values,
    double alpha,
    const double* y,
    double beta,
    double* x
);

/// Vector update: z = alpha * x + beta * y
YUTKI_EXPORT void yutki_cpu_axpby(
    size_t len,
    double alpha,
    const double* x,
    double beta,
    const double* y,
    double* z
);

/// Coordinate-wise projection into box bounds [lower[i], upper[i]]
YUTKI_EXPORT void yutki_cpu_box_project(
    size_t len,
    double* x,
    const double* lower,
    const double* upper
);

/// Dot product: sum_{i=0}^{len-1} x[i] * y[i]
YUTKI_EXPORT double yutki_cpu_dot(
    size_t len,
    const double* x,
    const double* y
);

/// Euclidean (L2) norm: sqrt(sum_{i=0}^{len-1} x[i]^2)
YUTKI_EXPORT double yutki_cpu_norm2(
    size_t len,
    const double* x
);

/// Infinity norm: max_{i=0}^{len-1} |x[i]|
YUTKI_EXPORT double yutki_cpu_norm_inf(
    size_t len,
    const double* x
);

}
