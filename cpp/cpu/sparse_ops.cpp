#include "sparse_ops.hpp"
#include <algorithm>
#include <cmath>

extern "C" {

void yutki_cpu_spmv(
    size_t num_rows,
    size_t num_cols,
    const size_t* row_ptrs,
    const size_t* col_indices,
    const double* values,
    double alpha,
    const double* x,
    double beta,
    double* y
) {
    (void)num_cols;
    for (size_t i = 0; i < num_rows; ++i) {
        double sum = 0.0;
        const size_t start = row_ptrs[i];
        const size_t end = row_ptrs[i + 1];
        for (size_t k = start; k < end; ++k) {
            sum += values[k] * x[col_indices[k]];
        }
        if (beta == 0.0) {
            y[i] = alpha * sum;
        } else {
            y[i] = alpha * sum + beta * y[i];
        }
    }
}

void yutki_cpu_spmv_transpose(
    size_t num_rows,
    size_t num_cols,
    const size_t* row_ptrs,
    const size_t* col_indices,
    const double* values,
    double alpha,
    const double* y,
    double beta,
    double* x
) {
    if (beta == 0.0) {
        std::fill(x, x + num_cols, 0.0);
    } else if (beta != 1.0) {
        for (size_t j = 0; j < num_cols; ++j) {
            x[j] *= beta;
        }
    }

    for (size_t i = 0; i < num_rows; ++i) {
        const double y_val = alpha * y[i];
        if (y_val == 0.0) continue;
        const size_t start = row_ptrs[i];
        const size_t end = row_ptrs[i + 1];
        for (size_t k = start; k < end; ++k) {
            x[col_indices[k]] += values[k] * y_val;
        }
    }
}

void yutki_cpu_axpby(
    size_t len,
    double alpha,
    const double* x,
    double beta,
    const double* y,
    double* z
) {
    for (size_t i = 0; i < len; ++i) {
        z[i] = alpha * x[i] + beta * y[i];
    }
}

void yutki_cpu_box_project(
    size_t len,
    double* x,
    const double* lower,
    const double* upper
) {
    for (size_t i = 0; i < len; ++i) {
        double val = x[i];
        const double l = lower[i];
        const double u = upper[i];
        if (val < l) val = l;
        if (val > u) val = u;
        x[i] = val;
    }
}

double yutki_cpu_dot(
    size_t len,
    const double* x,
    const double* y
) {
    double sum = 0.0;
    for (size_t i = 0; i < len; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

double yutki_cpu_norm2(
    size_t len,
    const double* x
) {
    double sum_sq = 0.0;
    for (size_t i = 0; i < len; ++i) {
        sum_sq += x[i] * x[i];
    }
    return std::sqrt(sum_sq);
}

double yutki_cpu_norm_inf(
    size_t len,
    const double* x
) {
    double max_val = 0.0;
    for (size_t i = 0; i < len; ++i) {
        const double abs_v = std::abs(x[i]);
        if (abs_v > max_val) {
            max_val = abs_v;
        }
    }
    return max_val;
}

}
