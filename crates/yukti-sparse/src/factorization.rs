//! In-place LU Factorization with Scaled Partial Pivoting for square linear systems.
//! Provides high-performance forward/back solve and transpose solve operations
//! required by the Revised Simplex Basis Factorization Engine.

#![allow(clippy::needless_range_loop)]

use crate::SparseError;

/// LU Decomposition result representing P * A = L * U, where
/// P is a row permutation matrix, L is unit lower triangular, and U is upper triangular.
#[derive(Debug, Clone, PartialEq)]
pub struct LuDecomposition {
    pub n: usize,
    /// Compact n x n row-major storage: L (strictly below diagonal) and U (on and above diagonal).
    lu_data: Vec<f64>,
    /// Permutation mapping: row k was mapped from row pivots[k]
    pivots: Vec<usize>,
    /// Number of row permutations performed (used for determinant sign)
    num_swaps: usize,
}

impl LuDecomposition {
    /// Create a preallocated LuDecomposition structure for dimension n.
    pub fn with_dimension(n: usize) -> Self {
        Self {
            n,
            lu_data: vec![0.0f64; n * n],
            pivots: (0..n).collect(),
            num_swaps: 0,
        }
    }

    /// Factorize an n x n square matrix into a preallocated LuDecomposition and scale buffer.
    /// Avoids repeated memory allocations during simplex basis updates.
    pub fn factorize_dense_into(
        n: usize,
        matrix_row_major: &[f64],
        tol: f64,
        out: &mut Self,
        scales: &mut [f64],
    ) -> Result<(), SparseError> {
        if matrix_row_major.len() != n * n {
            return Err(SparseError::DimensionMismatch {
                expected: n * n,
                found: matrix_row_major.len(),
            });
        }
        if scales.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: scales.len(),
            });
        }

        out.n = n;
        out.num_swaps = 0;
        out.lu_data.resize(n * n, 0.0);
        out.lu_data.copy_from_slice(matrix_row_major);
        out.pivots.resize(n, 0);
        for i in 0..n {
            out.pivots[i] = i;
        }

        if n == 0 {
            return Ok(());
        }

        // Validate finite numbers
        for (idx, &val) in matrix_row_major.iter().enumerate() {
            if !val.is_finite() {
                let row = idx / n;
                let col = idx % n;
                return Err(SparseError::NonFiniteValue {
                    row,
                    col,
                    value: val,
                });
            }
        }

        // Compute row scale factors s_i = max_j |a_{i, j}| for scaled partial pivoting
        for i in 0..n {
            let mut row_max = 0.0f64;
            let row_offset = i * n;
            for j in 0..n {
                let abs_val = out.lu_data[row_offset + j].abs();
                if abs_val > row_max {
                    row_max = abs_val;
                }
            }
            if row_max < tol {
                return Err(SparseError::SingularMatrix {
                    step: i,
                    value: row_max,
                });
            }
            scales[i] = row_max;
        }

        for k in 0..n {
            // Find pivot row p >= k with largest scaled entry in column k
            let mut max_scaled = 0.0f64;
            let mut pivot_row = k;

            for i in k..n {
                let entry_abs = out.lu_data[i * n + k].abs();
                let scaled_entry = entry_abs / scales[i];
                if scaled_entry > max_scaled {
                    max_scaled = scaled_entry;
                    pivot_row = i;
                }
            }

            let pivot_val = out.lu_data[pivot_row * n + k].abs();
            if pivot_val < tol {
                return Err(SparseError::SingularMatrix {
                    step: k,
                    value: pivot_val,
                });
            }

            // Swap rows k and pivot_row if distinct using slice swaps
            if pivot_row != k {
                let (k_slice, p_slice) = if k < pivot_row {
                    let (left, right) = out.lu_data.split_at_mut(pivot_row * n);
                    (&mut left[k * n..k * n + n], &mut right[0..n])
                } else {
                    let (left, right) = out.lu_data.split_at_mut(k * n);
                    (
                        &mut right[0..n],
                        &mut left[pivot_row * n..pivot_row * n + n],
                    )
                };
                k_slice.swap_with_slice(p_slice);
                out.pivots.swap(k, pivot_row);
                scales.swap(k, pivot_row);
                out.num_swaps += 1;
            }

            let pivot_elem = out.lu_data[k * n + k];
            let inv_pivot = 1.0 / pivot_elem;

            // Eliminate column k in rows below k
            let (top, bottom) = out.lu_data.split_at_mut((k + 1) * n);
            let row_k = &top[k * n + (k + 1)..k * n + n];

            for i_rel in 0..(n - k - 1) {
                let row_i_full = &mut bottom[i_rel * n..i_rel * n + n];
                let mult = row_i_full[k] * inv_pivot;
                row_i_full[k] = mult; // Store L entry in strictly lower triangle

                if mult.abs() > 1e-16 {
                    let row_i = &mut row_i_full[(k + 1)..n];
                    for (elem_i, &elem_k) in row_i.iter_mut().zip(row_k.iter()) {
                        *elem_i -= mult * elem_k;
                    }
                }
            }
        }

        Ok(())
    }

    /// Factorize an n x n square matrix given as a flat row-major slice.
    pub fn factorize_dense(
        n: usize,
        matrix_row_major: &[f64],
        tol: f64,
    ) -> Result<Self, SparseError> {
        let mut decomp = Self::with_dimension(n);
        let mut scales = vec![0.0f64; n];
        Self::factorize_dense_into(n, matrix_row_major, tol, &mut decomp, &mut scales)?;
        Ok(decomp)
    }

    /// Construct LU factorization from dense columns (n columns of length n).
    pub fn from_dense_columns(n: usize, cols: &[Vec<f64>], tol: f64) -> Result<Self, SparseError> {
        if cols.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: cols.len(),
            });
        }
        let mut row_major = vec![0.0f64; n * n];
        for (col_idx, col) in cols.iter().enumerate() {
            if col.len() != n {
                return Err(SparseError::DimensionMismatch {
                    expected: n,
                    found: col.len(),
                });
            }
            for row_idx in 0..n {
                row_major[row_idx * n + col_idx] = col[row_idx];
            }
        }
        Self::factorize_dense(n, &row_major, tol)
    }

    /// Construct LU factorization from sparse columns where each column is a list of (row, value).
    pub fn from_sparse_columns(
        n: usize,
        cols: &[Vec<(usize, f64)>],
        tol: f64,
    ) -> Result<Self, SparseError> {
        if cols.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: cols.len(),
            });
        }
        let mut row_major = vec![0.0f64; n * n];
        for (col_idx, col) in cols.iter().enumerate() {
            for &(row_idx, val) in col {
                if row_idx >= n {
                    return Err(SparseError::IndexOutOfBounds {
                        row: row_idx,
                        col: col_idx,
                        num_rows: n,
                        num_cols: n,
                    });
                }
                row_major[row_idx * n + col_idx] += val;
            }
        }
        Self::factorize_dense(n, &row_major, tol)
    }

    /// Solve the linear system A * x = b for x in place into preallocated buffers.
    /// `x` receives the solution vector. `y` is a scratch buffer of length n.
    pub fn solve_into(&self, b: &[f64], x: &mut [f64], y: &mut [f64]) -> Result<(), SparseError> {
        let n = self.n;
        if b.len() != n || x.len() != n || y.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: b.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }

        // Apply permutation P to b: y initially holds P * b
        for i in 0..n {
            let orig_idx = self.pivots[i];
            let val = b[orig_idx];
            if !val.is_finite() {
                return Err(SparseError::NonFiniteVector {
                    target: "b".into(),
                    index: orig_idx,
                    value: val,
                });
            }
            y[i] = val;
        }

        // Forward substitution: L * y = P * b (L has 1.0 on diagonal)
        for i in 0..n {
            let row_offset = i * n;
            let mut sum = 0.0f64;
            let row_l = &self.lu_data[row_offset..row_offset + i];
            let y_slice = &y[0..i];
            for (&l_ij, &y_j) in row_l.iter().zip(y_slice.iter()) {
                sum += l_ij * y_j;
            }
            y[i] -= sum;
        }

        // Back substitution: U * x = y
        for i in (0..n).rev() {
            let row_offset = i * n;
            let mut sum = 0.0f64;
            let row_u = &self.lu_data[row_offset + (i + 1)..row_offset + n];
            let x_slice = &x[(i + 1)..n];
            for (&u_ij, &x_j) in row_u.iter().zip(x_slice.iter()) {
                sum += u_ij * x_j;
            }
            let diag = self.lu_data[row_offset + i];
            if diag.abs() < 1e-15 {
                return Err(SparseError::SingularMatrix {
                    step: i,
                    value: diag,
                });
            }
            x[i] = (y[i] - sum) / diag;
        }

        Ok(())
    }

    /// Solve the linear system A * x = b for x.
    /// Uses forward substitution on L and back substitution on U.
    pub fn solve(&self, b: &[f64]) -> Result<Vec<f64>, SparseError> {
        let n = self.n;
        let mut x = vec![0.0f64; n];
        let mut y = vec![0.0f64; n];
        self.solve_into(b, &mut x, &mut y)?;
        Ok(x)
    }

    /// Solve the transposed linear system A^T * lambda = c for lambda in place into preallocated buffers.
    /// `lambda` receives the solution vector. `work` is a scratch buffer of length n.
    pub fn solve_transpose_into(
        &self,
        c: &[f64],
        lambda: &mut [f64],
        work: &mut [f64],
    ) -> Result<(), SparseError> {
        let n = self.n;
        if c.len() != n || lambda.len() != n || work.len() != n {
            return Err(SparseError::DimensionMismatch {
                expected: n,
                found: c.len(),
            });
        }
        if n == 0 {
            return Ok(());
        }

        for (idx, &val) in c.iter().enumerate() {
            if !val.is_finite() {
                return Err(SparseError::NonFiniteVector {
                    target: "c".into(),
                    index: idx,
                    value: val,
                });
            }
        }

        // Step 1: Forward solve U^T * w = c into work buffer
        for i in 0..n {
            let mut sum = 0.0f64;
            for j in 0..i {
                sum += self.lu_data[j * n + i] * work[j];
            }
            let diag = self.lu_data[i * n + i];
            if diag.abs() < 1e-15 {
                return Err(SparseError::SingularMatrix {
                    step: i,
                    value: diag,
                });
            }
            work[i] = (c[i] - sum) / diag;
        }

        // Step 2: Backward solve L^T * v = w in place in work buffer
        for i in (0..n).rev() {
            let mut sum = 0.0f64;
            for j in (i + 1)..n {
                sum += self.lu_data[j * n + i] * work[j];
            }
            work[i] -= sum;
        }

        // Step 3: Permute lambda = P^T * v
        for k in 0..n {
            lambda[self.pivots[k]] = work[k];
        }

        Ok(())
    }

    /// Solve the transposed linear system A^T * lambda = c for lambda.
    /// Since P * A = L * U ==> A^T = U^T * L^T * P.
    /// Therefore U^T * L^T * (P * lambda) = c.
    /// Step 1: Solve U^T * w = c for w.
    /// Step 2: Solve L^T * v = w for v.
    /// Step 3: Compute lambda = P^T * v.
    pub fn solve_transpose(&self, c: &[f64]) -> Result<Vec<f64>, SparseError> {
        let n = self.n;
        let mut lambda = vec![0.0f64; n];
        let mut work = vec![0.0f64; n];
        self.solve_transpose_into(c, &mut lambda, &mut work)?;
        Ok(lambda)
    }

    /// Compute the determinant of matrix A.
    pub fn det(&self) -> f64 {
        if self.n == 0 {
            return 1.0;
        }
        let mut d = if self.num_swaps % 2 == 1 { -1.0 } else { 1.0 };
        for i in 0..self.n {
            d *= self.lu_data[i * self.n + i];
        }
        d
    }

    /// Return dimension n.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Find the minimum absolute diagonal pivot element (measure of conditioning).
    pub fn min_abs_pivot(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        let mut min_p = f64::INFINITY;
        for i in 0..self.n {
            let p = self.lu_data[i * self.n + i].abs();
            if p < min_p {
                min_p = p;
            }
        }
        min_p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lu_2x2_solve_and_transpose() {
        // A = [[1, 2], [3, 4]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let lu = LuDecomposition::factorize_dense(2, &a, 1e-12).unwrap();

        // Solve A * x = [5, 11] => x = [1, 2]
        let b = vec![5.0, 11.0];
        let x = lu.solve(&b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10, "x[0] was {}", x[0]);
        assert!((x[1] - 2.0).abs() < 1e-10, "x[1] was {}", x[1]);

        // Solve A^T * lambda = [7, 10] => lambda = [1, 2]
        let c = vec![7.0, 10.0];
        let lambda = lu.solve_transpose(&c).unwrap();
        assert!(
            (lambda[0] - 1.0).abs() < 1e-10,
            "lambda[0] was {}",
            lambda[0]
        );
        assert!(
            (lambda[1] - 2.0).abs() < 1e-10,
            "lambda[1] was {}",
            lambda[1]
        );

        // Det: 1*4 - 2*3 = -2
        assert!((lu.det() - (-2.0)).abs() < 1e-10);
    }

    #[test]
    fn test_lu_3x3_solve() {
        // A = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]
        let a = vec![2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0];
        let lu = LuDecomposition::factorize_dense(3, &a, 1e-12).unwrap();

        // Target x = [1, -2, 3]
        // b = A * x = [2*1 + 1*(-2) - 1*3 = -3,
        //              -3*1 - 1*(-2) + 2*3 = 5,
        //              -2*1 + 1*(-2) + 2*3 = 2]
        let b = vec![-3.0, 5.0, 2.0];
        let x = lu.solve(&b).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-10);
        assert!((x[1] - (-2.0)).abs() < 1e-10);
        assert!((x[2] - 3.0).abs() < 1e-10);

        // Transpose solve:
        // A^T * y = c
        let _y = [2.0, 1.0, -1.0];
        // c = A^T * y = [2*2 + (-3)*1 + (-2)*(-1) = 3,
        //                1*2 + (-1)*1 + 1*(-1) = 0,
        //                -1*2 + 2*1 + 2*(-1) = -2]
        let c = vec![3.0, 0.0, -2.0];
        let sol_y = lu.solve_transpose(&c).unwrap();
        assert!((sol_y[0] - 2.0).abs() < 1e-10);
        assert!((sol_y[1] - 1.0).abs() < 1e-10);
        assert!((sol_y[2] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_lu_singular_matrix() {
        // A = [[1, 2], [2, 4]] -> row 2 is 2 * row 1
        let a = vec![1.0, 2.0, 2.0, 4.0];
        let res = LuDecomposition::factorize_dense(2, &a, 1e-12);
        assert!(res.is_err());
    }

    #[test]
    fn test_lu_from_sparse_columns() {
        // 3x3 identity
        let cols = vec![vec![(0, 1.0)], vec![(1, 1.0)], vec![(2, 1.0)]];
        let lu = LuDecomposition::from_sparse_columns(3, &cols, 1e-12).unwrap();
        let b = vec![4.0, 5.0, 6.0];
        let x = lu.solve(&b).unwrap();
        assert_eq!(x, b);
        let y = lu.solve_transpose(&b).unwrap();
        assert_eq!(y, b);
    }
}
