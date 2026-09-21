//! High-performance sparse matrix data structures and primitives for Yutki-Samadhata.
//! Supports Coordinate (COO), Compressed Sparse Row (CSR), and Compressed Sparse Column (CSC) formats.

use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod factorization;
pub use factorization::LuDecomposition;

#[derive(Error, Debug, PartialEq)]
pub enum SparseError {
    #[error("Dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },

    #[error("Index out of bounds: row {row} >= {num_rows} or col {col} >= {num_cols}")]
    IndexOutOfBounds {
        row: usize,
        col: usize,
        num_rows: usize,
        num_cols: usize,
    },

    #[error("Malformed sparse structure: {0}")]
    MalformedStructure(String),

    #[error("Non-finite value (NaN or Inf) encountered at row {row}, col {col}: {value}")]
    NonFiniteValue { row: usize, col: usize, value: f64 },

    #[error(
        "Non-finite vector value (NaN or Inf) encountered in {target} at index {index}: {value}"
    )]
    NonFiniteVector {
        target: String,
        index: usize,
        value: f64,
    },

    #[error("Singular matrix encountered during factorization: pivot at step {step} is {value}")]
    SingularMatrix { step: usize, value: f64 },
}

/// Structural and numerical summary statistics for sparse matrices.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatrixStatistics {
    pub num_rows: usize,
    pub num_cols: usize,
    pub nnz: usize,
    pub density: f64,
    pub min_row_nnz: usize,
    pub max_row_nnz: usize,
    pub avg_row_nnz: f64,
    pub min_col_nnz: usize,
    pub max_col_nnz: usize,
    pub avg_col_nnz: f64,
    pub min_abs_val: f64,
    pub max_abs_val: f64,
    pub condition_ratio: f64,
}

/// Vector dot product: <x, y>
pub fn dot(x: &[f64], y: &[f64]) -> Result<f64, SparseError> {
    if x.len() != y.len() {
        return Err(SparseError::DimensionMismatch {
            expected: x.len(),
            found: y.len(),
        });
    }
    let mut sum = 0.0f64;
    for (i, (&a, &b)) in x.iter().zip(y.iter()).enumerate() {
        if !a.is_finite() {
            return Err(SparseError::NonFiniteVector {
                target: "x".into(),
                index: i,
                value: a,
            });
        }
        if !b.is_finite() {
            return Err(SparseError::NonFiniteVector {
                target: "y".into(),
                index: i,
                value: b,
            });
        }
        sum += a * b;
    }
    Ok(sum)
}

/// L1 vector norm: sum(|x_i|)
pub fn norm_l1(x: &[f64]) -> f64 {
    x.iter().map(|&v| v.abs()).sum()
}

/// Euclidean (L2) vector norm: sqrt(sum(x_i^2))
pub fn norm_l2(x: &[f64]) -> f64 {
    let sum_sq: f64 = x.iter().map(|&v| v * v).sum();
    sum_sq.sqrt()
}

/// Infinity vector norm: max(|x_i|)
pub fn norm_inf(x: &[f64]) -> f64 {
    x.iter().map(|&v| v.abs()).fold(0.0f64, f64::max)
}

/// Verify that all entries in a slice are finite numbers (not NaN, not Inf).
pub fn check_finite(target: &str, slice: &[f64]) -> Result<(), SparseError> {
    for (index, &value) in slice.iter().enumerate() {
        if !value.is_finite() {
            return Err(SparseError::NonFiniteVector {
                target: target.to_string(),
                index,
                value,
            });
        }
    }
    Ok(())
}

/// Coordinate list (COO) sparse matrix format, primarily used for problem ingestion and assembly.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CooMatrix {
    pub num_rows: usize,
    pub num_cols: usize,
    pub row_indices: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl CooMatrix {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            num_rows,
            num_cols,
            row_indices: Vec::new(),
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn with_capacity(num_rows: usize, num_cols: usize, capacity: usize) -> Self {
        Self {
            num_rows,
            num_cols,
            row_indices: Vec::with_capacity(capacity),
            col_indices: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        }
    }

    pub fn add_entry(&mut self, row: usize, col: usize, val: f64) -> Result<(), SparseError> {
        if row >= self.num_rows || col >= self.num_cols {
            return Err(SparseError::IndexOutOfBounds {
                row,
                col,
                num_rows: self.num_rows,
                num_cols: self.num_cols,
            });
        }
        if !val.is_finite() {
            return Err(SparseError::NonFiniteValue {
                row,
                col,
                value: val,
            });
        }
        self.row_indices.push(row);
        self.col_indices.push(col);
        self.values.push(val);
        Ok(())
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn density(&self) -> f64 {
        let total = self.num_rows * self.num_cols;
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    pub fn validate(&self) -> Result<(), SparseError> {
        let nnz = self.values.len();
        if self.row_indices.len() != nnz || self.col_indices.len() != nnz {
            return Err(SparseError::MalformedStructure(format!(
                "Vector lengths mismatch in COO: row_indices={}, col_indices={}, values={}",
                self.row_indices.len(),
                self.col_indices.len(),
                nnz
            )));
        }
        for i in 0..nnz {
            let r = self.row_indices[i];
            let c = self.col_indices[i];
            let v = self.values[i];
            if r >= self.num_rows || c >= self.num_cols {
                return Err(SparseError::IndexOutOfBounds {
                    row: r,
                    col: c,
                    num_rows: self.num_rows,
                    num_cols: self.num_cols,
                });
            }
            if !v.is_finite() {
                return Err(SparseError::NonFiniteValue {
                    row: r,
                    col: c,
                    value: v,
                });
            }
        }
        Ok(())
    }

    pub fn from_csr(csr: &CsrMatrix) -> Self {
        let nnz = csr.nnz();
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for i in 0..csr.num_rows {
            let start = csr.row_ptrs[i];
            let end = csr.row_ptrs[i + 1];
            for k in start..end {
                row_indices.push(i);
                col_indices.push(csr.col_indices[k]);
                values.push(csr.values[k]);
            }
        }

        Self {
            num_rows: csr.num_rows,
            num_cols: csr.num_cols,
            row_indices,
            col_indices,
            values,
        }
    }

    pub fn from_csc(csc: &CscMatrix) -> Self {
        let nnz = csc.nnz();
        let mut row_indices = Vec::with_capacity(nnz);
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for j in 0..csc.num_cols {
            let start = csc.col_ptrs[j];
            let end = csc.col_ptrs[j + 1];
            for k in start..end {
                col_indices.push(j);
                row_indices.push(csc.row_indices[k]);
                values.push(csc.values[k]);
            }
        }

        Self {
            num_rows: csc.num_rows,
            num_cols: csc.num_cols,
            row_indices,
            col_indices,
            values,
        }
    }
}

/// Compressed Sparse Row (CSR) matrix format, optimized for matrix-vector multiplication (A * x).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CsrMatrix {
    pub num_rows: usize,
    pub num_cols: usize,
    pub row_ptrs: Vec<usize>,
    pub col_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl CsrMatrix {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            num_rows,
            num_cols,
            row_ptrs: vec![0; num_rows + 1],
            col_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Convert from COO matrix with duplicate coordinate summation.
    pub fn from_coo(coo: &CooMatrix) -> Result<Self, SparseError> {
        coo.validate()?;
        let nnz = coo.nnz();
        let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(nnz);
        for i in 0..nnz {
            triplets.push((coo.row_indices[i], coo.col_indices[i], coo.values[i]));
        }

        // Sort primarily by row, then by column
        triplets.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        let mut row_ptrs = vec![0; coo.num_rows + 1];
        let mut col_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        let mut last_row = usize::MAX;
        let mut last_col = usize::MAX;

        for (r, c, val) in triplets {
            if r == last_row && c == last_col {
                // Coalesce duplicate coordinate
                if let Some(last_val) = values.last_mut() {
                    *last_val += val;
                }
            } else {
                col_indices.push(c);
                values.push(val);
                row_ptrs[r + 1] += 1;
                last_row = r;
                last_col = c;
            }
        }

        // Prefix sum to compute row_ptrs
        for i in 0..coo.num_rows {
            row_ptrs[i + 1] += row_ptrs[i];
        }

        Ok(Self {
            num_rows: coo.num_rows,
            num_cols: coo.num_cols,
            row_ptrs,
            col_indices,
            values,
        })
    }

    pub fn from_csc(csc: &CscMatrix) -> Result<Self, SparseError> {
        let coo = CooMatrix::from_csc(csc);
        Self::from_coo(&coo)
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn density(&self) -> f64 {
        let total = self.num_rows * self.num_cols;
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    pub fn norm_inf(&self) -> f64 {
        let mut max_row_sum = 0.0f64;
        for i in 0..self.num_rows {
            let start = self.row_ptrs[i];
            let end = self.row_ptrs[i + 1];
            let row_sum: f64 = self.values[start..end].iter().map(|v| v.abs()).sum();
            if row_sum > max_row_sum {
                max_row_sum = row_sum;
            }
        }
        max_row_sum
    }

    pub fn validate(&self) -> Result<(), SparseError> {
        if self.row_ptrs.len() != self.num_rows + 1 {
            return Err(SparseError::MalformedStructure(format!(
                "CSR row_ptrs length ({}) != num_rows + 1 ({})",
                self.row_ptrs.len(),
                self.num_rows + 1
            )));
        }
        if self.row_ptrs[0] != 0 {
            return Err(SparseError::MalformedStructure("row_ptrs[0] != 0".into()));
        }
        let nnz = self.values.len();
        if self.col_indices.len() != nnz {
            return Err(SparseError::MalformedStructure(format!(
                "col_indices length ({}) != values length ({})",
                self.col_indices.len(),
                nnz
            )));
        }
        if self.row_ptrs[self.num_rows] != nnz {
            return Err(SparseError::MalformedStructure(format!(
                "row_ptrs[last] ({}) != nnz ({})",
                self.row_ptrs[self.num_rows], nnz
            )));
        }

        for i in 0..self.num_rows {
            let start = self.row_ptrs[i];
            let end = self.row_ptrs[i + 1];
            if start > end {
                return Err(SparseError::MalformedStructure(format!(
                    "row_ptrs not non-decreasing at row {i}: start {start} > end {end}"
                )));
            }
            if end > nnz {
                return Err(SparseError::MalformedStructure(format!(
                    "row_ptrs[{}] ({}) exceeds nnz ({})",
                    i + 1,
                    end,
                    nnz
                )));
            }

            let mut prev_col = None;
            for k in start..end {
                let col = self.col_indices[k];
                if col >= self.num_cols {
                    return Err(SparseError::IndexOutOfBounds {
                        row: i,
                        col,
                        num_rows: self.num_rows,
                        num_cols: self.num_cols,
                    });
                }
                if let Some(pc) = prev_col {
                    if col <= pc {
                        return Err(SparseError::MalformedStructure(format!(
                            "Column indices in row {i} are not strictly increasing: prev {pc} >= curr {col}"
                        )));
                    }
                }
                prev_col = Some(col);

                let val = self.values[k];
                if !val.is_finite() {
                    return Err(SparseError::NonFiniteValue {
                        row: i,
                        col,
                        value: val,
                    });
                }
            }
        }
        Ok(())
    }

    /// Compute detailed matrix statistics
    pub fn statistics(&self) -> MatrixStatistics {
        let m = self.num_rows;
        let n = self.num_cols;
        let nnz = self.nnz();
        let density = self.density();

        let mut min_row_nnz = if m == 0 { 0 } else { usize::MAX };
        let mut max_row_nnz = 0;
        let mut col_counts = vec![0usize; n];

        let mut min_abs_val = f64::INFINITY;
        let mut max_abs_val = 0.0f64;

        for i in 0..m {
            let row_len = self.row_ptrs[i + 1] - self.row_ptrs[i];
            if row_len < min_row_nnz {
                min_row_nnz = row_len;
            }
            if row_len > max_row_nnz {
                max_row_nnz = row_len;
            }
            let start = self.row_ptrs[i];
            let end = self.row_ptrs[i + 1];
            for k in start..end {
                let col = self.col_indices[k];
                col_counts[col] += 1;
                let abs_v = self.values[k].abs();
                if abs_v > 0.0 && abs_v < min_abs_val {
                    min_abs_val = abs_v;
                }
                if abs_v > max_abs_val {
                    max_abs_val = abs_v;
                }
            }
        }

        if min_row_nnz == usize::MAX {
            min_row_nnz = 0;
        }
        if min_abs_val == f64::INFINITY {
            min_abs_val = 0.0;
        }

        let min_col_nnz = col_counts.iter().copied().min().unwrap_or(0);
        let max_col_nnz = col_counts.iter().copied().max().unwrap_or(0);
        let avg_row_nnz = if m == 0 { 0.0 } else { nnz as f64 / m as f64 };
        let avg_col_nnz = if n == 0 { 0.0 } else { nnz as f64 / n as f64 };

        let condition_ratio = if min_abs_val > 0.0 {
            max_abs_val / min_abs_val
        } else {
            1.0
        };

        MatrixStatistics {
            num_rows: m,
            num_cols: n,
            nnz,
            density,
            min_row_nnz,
            max_row_nnz,
            avg_row_nnz,
            min_col_nnz,
            max_col_nnz,
            avg_col_nnz,
            min_abs_val,
            max_abs_val,
            condition_ratio,
        }
    }

    /// Sparse matrix-vector product: y = alpha * A * x + beta * y
    pub fn spmv(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) -> Result<(), SparseError> {
        if x.len() != self.num_cols {
            return Err(SparseError::DimensionMismatch {
                expected: self.num_cols,
                found: x.len(),
            });
        }
        if y.len() != self.num_rows {
            return Err(SparseError::DimensionMismatch {
                expected: self.num_rows,
                found: y.len(),
            });
        }

        for (i, y_elem) in y.iter_mut().enumerate() {
            let mut sum = 0.0;
            let start = self.row_ptrs[i];
            let end = self.row_ptrs[i + 1];
            for k in start..end {
                let col = self.col_indices[k];
                sum += self.values[k] * x[col];
            }
            if beta == 0.0 {
                *y_elem = alpha * sum;
            } else {
                *y_elem = alpha * sum + beta * *y_elem;
            }
        }
        Ok(())
    }

    /// Convenience matrix-vector multiply: returns y = A * x
    pub fn mul_vec(&self, x: &[f64]) -> Result<Vec<f64>, SparseError> {
        let mut y = vec![0.0; self.num_rows];
        self.spmv(1.0, x, 0.0, &mut y)?;
        Ok(y)
    }

    /// Transpose sparse matrix-vector product: x = alpha * A^T * y + beta * x
    pub fn spmv_transpose(
        &self,
        alpha: f64,
        y: &[f64],
        beta: f64,
        x: &mut [f64],
    ) -> Result<(), SparseError> {
        if y.len() != self.num_rows {
            return Err(SparseError::DimensionMismatch {
                expected: self.num_rows,
                found: y.len(),
            });
        }
        if x.len() != self.num_cols {
            return Err(SparseError::DimensionMismatch {
                expected: self.num_cols,
                found: x.len(),
            });
        }

        if beta == 0.0 {
            for v in x.iter_mut() {
                *v = 0.0;
            }
        } else if beta != 1.0 {
            for v in x.iter_mut() {
                *v *= beta;
            }
        }

        for (i, &y_val) in y.iter().enumerate() {
            if y_val == 0.0 {
                continue;
            }
            let start = self.row_ptrs[i];
            let end = self.row_ptrs[i + 1];
            for k in start..end {
                let col = self.col_indices[k];
                x[col] += alpha * self.values[k] * y_val;
            }
        }
        Ok(())
    }

    /// Convenience transpose matrix-vector multiply: returns x = A^T * y
    pub fn mul_transpose_vec(&self, y: &[f64]) -> Result<Vec<f64>, SparseError> {
        let mut x = vec![0.0; self.num_cols];
        self.spmv_transpose(1.0, y, 0.0, &mut x)?;
        Ok(x)
    }
}

/// Compressed Sparse Column (CSC) matrix format, optimized for column-oriented and transpose computations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CscMatrix {
    pub num_rows: usize,
    pub num_cols: usize,
    pub col_ptrs: Vec<usize>,
    pub row_indices: Vec<usize>,
    pub values: Vec<f64>,
}

impl CscMatrix {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        Self {
            num_rows,
            num_cols,
            col_ptrs: vec![0; num_cols + 1],
            row_indices: Vec::new(),
            values: Vec::new(),
        }
    }

    pub fn from_coo(coo: &CooMatrix) -> Result<Self, SparseError> {
        coo.validate()?;
        let nnz = coo.nnz();
        let mut triplets: Vec<(usize, usize, f64)> = Vec::with_capacity(nnz);
        for i in 0..nnz {
            triplets.push((coo.row_indices[i], coo.col_indices[i], coo.values[i]));
        }

        // Sort primarily by column, then by row
        triplets.sort_unstable_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));

        let mut col_ptrs = vec![0; coo.num_cols + 1];
        let mut row_indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        let mut last_col = usize::MAX;
        let mut last_row = usize::MAX;

        for (r, c, val) in triplets {
            if c == last_col && r == last_row {
                if let Some(last_val) = values.last_mut() {
                    *last_val += val;
                }
            } else {
                row_indices.push(r);
                values.push(val);
                col_ptrs[c + 1] += 1;
                last_col = c;
                last_row = r;
            }
        }

        for j in 0..coo.num_cols {
            col_ptrs[j + 1] += col_ptrs[j];
        }

        Ok(Self {
            num_rows: coo.num_rows,
            num_cols: coo.num_cols,
            col_ptrs,
            row_indices,
            values,
        })
    }

    pub fn from_csr(csr: &CsrMatrix) -> Result<Self, SparseError> {
        let coo = CooMatrix::from_csr(csr);
        Self::from_coo(&coo)
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    pub fn density(&self) -> f64 {
        let total = self.num_rows * self.num_cols;
        if total == 0 {
            0.0
        } else {
            self.nnz() as f64 / total as f64
        }
    }

    pub fn validate(&self) -> Result<(), SparseError> {
        if self.col_ptrs.len() != self.num_cols + 1 {
            return Err(SparseError::MalformedStructure(format!(
                "CSC col_ptrs length ({}) != num_cols + 1 ({})",
                self.col_ptrs.len(),
                self.num_cols + 1
            )));
        }
        if self.col_ptrs[0] != 0 {
            return Err(SparseError::MalformedStructure("col_ptrs[0] != 0".into()));
        }
        let nnz = self.values.len();
        if self.row_indices.len() != nnz {
            return Err(SparseError::MalformedStructure(format!(
                "row_indices length ({}) != values length ({})",
                self.row_indices.len(),
                nnz
            )));
        }
        if self.col_ptrs[self.num_cols] != nnz {
            return Err(SparseError::MalformedStructure(format!(
                "col_ptrs[last] ({}) != nnz ({})",
                self.col_ptrs[self.num_cols], nnz
            )));
        }

        for j in 0..self.num_cols {
            let start = self.col_ptrs[j];
            let end = self.col_ptrs[j + 1];
            if start > end {
                return Err(SparseError::MalformedStructure(format!(
                    "col_ptrs not non-decreasing at col {j}: start {start} > end {end}"
                )));
            }
            if end > nnz {
                return Err(SparseError::MalformedStructure(format!(
                    "col_ptrs[{}] ({}) exceeds nnz ({})",
                    j + 1,
                    end,
                    nnz
                )));
            }

            let mut prev_row = None;
            for k in start..end {
                let row = self.row_indices[k];
                if row >= self.num_rows {
                    return Err(SparseError::IndexOutOfBounds {
                        row,
                        col: j,
                        num_rows: self.num_rows,
                        num_cols: self.num_cols,
                    });
                }
                if let Some(pr) = prev_row {
                    if row <= pr {
                        return Err(SparseError::MalformedStructure(format!(
                            "Row indices in col {j} are not strictly increasing: prev {pr} >= curr {row}"
                        )));
                    }
                }
                prev_row = Some(row);

                let val = self.values[k];
                if !val.is_finite() {
                    return Err(SparseError::NonFiniteValue {
                        row,
                        col: j,
                        value: val,
                    });
                }
            }
        }
        Ok(())
    }

    /// Matrix-vector multiplication using CSC column access: y = alpha * A * x + beta * y
    pub fn spmv(&self, alpha: f64, x: &[f64], beta: f64, y: &mut [f64]) -> Result<(), SparseError> {
        if x.len() != self.num_cols {
            return Err(SparseError::DimensionMismatch {
                expected: self.num_cols,
                found: x.len(),
            });
        }
        if y.len() != self.num_rows {
            return Err(SparseError::DimensionMismatch {
                expected: self.num_rows,
                found: y.len(),
            });
        }

        if beta == 0.0 {
            for val in y.iter_mut() {
                *val = 0.0;
            }
        } else if beta != 1.0 {
            for val in y.iter_mut() {
                *val *= beta;
            }
        }

        for (j, &xj) in x.iter().enumerate() {
            if xj == 0.0 {
                continue;
            }
            let start = self.col_ptrs[j];
            let end = self.col_ptrs[j + 1];
            for k in start..end {
                let r = self.row_indices[k];
                y[r] += alpha * self.values[k] * xj;
            }
        }
        Ok(())
    }

    /// Transpose matrix-vector multiplication using CSC: x = alpha * A^T * y + beta * x
    pub fn spmv_transpose(
        &self,
        alpha: f64,
        y: &[f64],
        beta: f64,
        x: &mut [f64],
    ) -> Result<(), SparseError> {
        if y.len() != self.num_rows {
            return Err(SparseError::DimensionMismatch {
                expected: self.num_rows,
                found: y.len(),
            });
        }
        if x.len() != self.num_cols {
            return Err(SparseError::DimensionMismatch {
                expected: self.num_cols,
                found: x.len(),
            });
        }

        for (j, xj) in x.iter_mut().enumerate() {
            let mut sum = 0.0;
            let start = self.col_ptrs[j];
            let end = self.col_ptrs[j + 1];
            for k in start..end {
                let r = self.row_indices[k];
                sum += self.values[k] * y[r];
            }
            if beta == 0.0 {
                *xj = alpha * sum;
            } else {
                *xj = alpha * sum + beta * *xj;
            }
        }
        Ok(())
    }

    pub fn mul_vec(&self, x: &[f64]) -> Result<Vec<f64>, SparseError> {
        let mut y = vec![0.0; self.num_rows];
        self.spmv(1.0, x, 0.0, &mut y)?;
        Ok(y)
    }

    pub fn mul_transpose_vec(&self, y: &[f64]) -> Result<Vec<f64>, SparseError> {
        let mut x = vec![0.0; self.num_cols];
        self.spmv_transpose(1.0, y, 0.0, &mut x)?;
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_to_csr_and_spmv() {
        let mut coo = CooMatrix::new(2, 3);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 2, 2.0).unwrap();
        coo.add_entry(1, 1, 3.0).unwrap();
        coo.add_entry(1, 2, 4.0).unwrap();

        let csr = CsrMatrix::from_coo(&coo).unwrap();
        assert_eq!(csr.num_rows, 2);
        assert_eq!(csr.num_cols, 3);
        assert_eq!(csr.nnz(), 4);
        assert!(csr.validate().is_ok());

        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0, 0.0];
        csr.spmv(1.0, &x, 0.0, &mut y).unwrap();
        assert_eq!(y, vec![7.0, 18.0]);

        let y_res = csr.mul_vec(&x).unwrap();
        assert_eq!(y_res, vec![7.0, 18.0]);

        let y_dual = vec![2.0, 1.0];
        let mut x_out = vec![0.0, 0.0, 0.0];
        csr.spmv_transpose(1.0, &y_dual, 0.0, &mut x_out).unwrap();
        assert_eq!(x_out, vec![2.0, 3.0, 8.0]);

        let x_res = csr.mul_transpose_vec(&y_dual).unwrap();
        assert_eq!(x_res, vec![2.0, 3.0, 8.0]);
    }

    #[test]
    fn test_csc_conversion_and_parity() {
        let mut coo = CooMatrix::new(2, 3);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 2, 2.0).unwrap();
        coo.add_entry(1, 1, 3.0).unwrap();
        coo.add_entry(1, 2, 4.0).unwrap();

        let csr = CsrMatrix::from_coo(&coo).unwrap();
        let csc = CscMatrix::from_csr(&csr).unwrap();
        assert!(csc.validate().is_ok());

        let x = vec![1.0, 2.0, 3.0];
        let y_csr = csr.mul_vec(&x).unwrap();
        let y_csc = csc.mul_vec(&x).unwrap();
        assert_eq!(y_csr, y_csc);

        let y = vec![2.0, 1.0];
        let xt_csr = csr.mul_transpose_vec(&y).unwrap();
        let xt_csc = csc.mul_transpose_vec(&y).unwrap();
        assert_eq!(xt_csr, xt_csc);

        // Conversion roundtrip: CSC -> CSR
        let csr_roundtrip = CsrMatrix::from_csc(&csc).unwrap();
        assert_eq!(csr, csr_roundtrip);
    }

    #[test]
    fn test_coalesce_duplicates() {
        let mut coo = CooMatrix::new(2, 2);
        coo.add_entry(0, 1, 1.5).unwrap();
        coo.add_entry(0, 1, 2.5).unwrap();

        let csr = CsrMatrix::from_coo(&coo).unwrap();
        assert_eq!(csr.nnz(), 1);
        assert_eq!(csr.values[0], 4.0);

        let csc = CscMatrix::from_coo(&coo).unwrap();
        assert_eq!(csc.nnz(), 1);
        assert_eq!(csc.values[0], 4.0);
    }

    #[test]
    fn test_matrix_statistics() {
        let mut coo = CooMatrix::new(3, 3);
        coo.add_entry(0, 0, 10.0).unwrap();
        coo.add_entry(0, 1, 2.0).unwrap();
        coo.add_entry(1, 1, 5.0).unwrap();
        coo.add_entry(2, 2, 100.0).unwrap();

        let csr = CsrMatrix::from_coo(&coo).unwrap();
        let stats = csr.statistics();
        assert_eq!(stats.num_rows, 3);
        assert_eq!(stats.num_cols, 3);
        assert_eq!(stats.nnz, 4);
        assert_eq!(stats.min_row_nnz, 1);
        assert_eq!(stats.max_row_nnz, 2);
        assert_eq!(stats.min_abs_val, 2.0);
        assert_eq!(stats.max_abs_val, 100.0);
        assert_eq!(stats.condition_ratio, 50.0);
    }

    #[test]
    fn test_vector_primitives() {
        let x = vec![3.0, -4.0, 0.0];
        let y = vec![2.0, 1.0, 5.0];

        assert_eq!(dot(&x, &y).unwrap(), 2.0); // 3*2 + (-4)*1 + 0*5 = 2
        assert_eq!(norm_l1(&x), 7.0);
        assert_eq!(norm_l2(&x), 5.0);
        assert_eq!(norm_inf(&x), 4.0);

        let diff_len = vec![1.0, 2.0];
        assert!(dot(&x, &diff_len).is_err());
    }

    #[test]
    fn test_nan_and_inf_validation() {
        let mut coo = CooMatrix::new(2, 2);
        assert!(coo.add_entry(0, 0, f64::NAN).is_err());
        assert!(coo.add_entry(0, 0, f64::INFINITY).is_err());
        assert!(coo.add_entry(2, 0, 1.0).is_err()); // out of bounds

        let v_nan = vec![1.0, f64::NAN];
        assert!(check_finite("test", &v_nan).is_err());

        let v_inf = vec![1.0, f64::NEG_INFINITY];
        assert!(check_finite("test", &v_inf).is_err());
    }

    #[test]
    fn test_coalesce_triple_duplicates() {
        let mut coo = CooMatrix::new(2, 2);
        coo.add_entry(1, 1, 1.0).unwrap();
        coo.add_entry(1, 1, 2.0).unwrap();
        coo.add_entry(1, 1, 3.0).unwrap();

        let csr = CsrMatrix::from_coo(&coo).unwrap();
        assert_eq!(csr.nnz(), 1);
        assert_eq!(csr.values[0], 6.0);
        assert!(csr.validate().is_ok());

        let csc = CscMatrix::from_coo(&coo).unwrap();
        assert_eq!(csc.nnz(), 1);
        assert_eq!(csc.values[0], 6.0);
        assert!(csc.validate().is_ok());
    }

    #[test]
    fn test_coalesce_duplicates_sum_to_zero() {
        let mut coo = CooMatrix::new(2, 2);
        coo.add_entry(0, 0, 5.5).unwrap();
        coo.add_entry(0, 0, -5.5).unwrap();

        let csr = CsrMatrix::from_coo(&coo).unwrap();
        assert_eq!(csr.nnz(), 1);
        assert_eq!(csr.values[0], 0.0);
        assert!(csr.validate().is_ok());
    }

    #[test]
    fn test_empty_rows_and_empty_columns() {
        // 4 rows, 4 cols.
        // Row 0 has (0, 1, 10.0), (0, 3, 20.0)
        // Row 1 is empty
        // Row 2 has (2, 3, 30.0)
        // Row 3 is empty
        // Note: Col 0 and Col 2 are completely empty
        let mut coo = CooMatrix::new(4, 4);
        coo.add_entry(0, 1, 10.0).unwrap();
        coo.add_entry(0, 3, 20.0).unwrap();
        coo.add_entry(2, 3, 30.0).unwrap();

        let csr = CsrMatrix::from_coo(&coo).unwrap();
        assert_eq!(csr.num_rows, 4);
        assert_eq!(csr.num_cols, 4);
        assert_eq!(csr.nnz(), 3);
        assert!(csr.validate().is_ok());

        let csc = CscMatrix::from_coo(&coo).unwrap();
        assert_eq!(csc.num_rows, 4);
        assert_eq!(csc.num_cols, 4);
        assert_eq!(csc.nnz(), 3);
        assert!(csc.validate().is_ok());

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = csr.mul_vec(&x).unwrap();
        // y[0] = 10*2 + 20*4 = 100
        // y[1] = 0 (empty row)
        // y[2] = 30*4 = 120
        // y[3] = 0 (empty row)
        assert_eq!(y, vec![100.0, 0.0, 120.0, 0.0]);

        let y_dual = vec![1.0, 5.0, 2.0, 9.0];
        let xt = csr.mul_transpose_vec(&y_dual).unwrap();
        // xt[0] = 0 (empty col)
        // xt[1] = 10*1 = 10
        // xt[2] = 0 (empty col)
        // xt[3] = 20*1 + 30*2 = 80
        assert_eq!(xt, vec![0.0, 10.0, 0.0, 80.0]);

        // Parity with CSC
        assert_eq!(csc.mul_vec(&x).unwrap(), y);
        assert_eq!(csc.mul_transpose_vec(&y_dual).unwrap(), xt);
    }

    #[test]
    fn test_zero_dimension_matrices() {
        let coo_0x0 = CooMatrix::new(0, 0);
        let csr_0x0 = CsrMatrix::from_coo(&coo_0x0).unwrap();
        assert!(csr_0x0.validate().is_ok());
        assert_eq!(csr_0x0.mul_vec(&[]).unwrap(), Vec::<f64>::new());
        assert_eq!(csr_0x0.mul_transpose_vec(&[]).unwrap(), Vec::<f64>::new());

        let coo_0x3 = CooMatrix::new(0, 3);
        let csr_0x3 = CsrMatrix::from_coo(&coo_0x3).unwrap();
        assert!(csr_0x3.validate().is_ok());
        assert_eq!(
            csr_0x3.mul_vec(&[1.0, 2.0, 3.0]).unwrap(),
            Vec::<f64>::new()
        );
        assert_eq!(csr_0x3.mul_transpose_vec(&[]).unwrap(), vec![0.0, 0.0, 0.0]);

        let coo_3x0 = CooMatrix::new(3, 0);
        let csr_3x0 = CsrMatrix::from_coo(&coo_3x0).unwrap();
        assert!(csr_3x0.validate().is_ok());
        assert_eq!(csr_3x0.mul_vec(&[]).unwrap(), vec![0.0, 0.0, 0.0]);
        assert_eq!(
            csr_3x0.mul_transpose_vec(&[1.0, 2.0, 3.0]).unwrap(),
            Vec::<f64>::new()
        );
    }
}
