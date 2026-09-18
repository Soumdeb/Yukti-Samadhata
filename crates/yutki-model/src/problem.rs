use crate::bounds::{Bound, Sense};
use crate::error::SolverError;
use crate::status::SolverStatus;
use serde::{Deserialize, Serialize};
use yutki_sparse::CsrMatrix;

/// General continuous Linear Program representation:
/// minimize / maximize c^T x + obj_offset
/// subject to:
///   row_bounds.lower <= A x <= row_bounds.upper
///   var_bounds.lower <= x <= var_bounds.upper
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LpProblem {
    pub name: String,
    pub sense: Sense,
    pub obj_offset: f64,
    pub c: Vec<f64>,
    pub var_names: Vec<String>,
    pub var_bounds: Vec<Bound>,
    pub row_names: Vec<String>,
    pub row_bounds: Vec<Bound>,
    pub a: CsrMatrix,
}

impl LpProblem {
    pub fn num_variables(&self) -> usize {
        self.var_names.len()
    }

    pub fn num_constraints(&self) -> usize {
        self.row_names.len()
    }

    pub fn num_nonzeros(&self) -> usize {
        self.a.nnz()
    }

    pub fn density(&self) -> f64 {
        self.a.density()
    }

    pub fn evaluate_objective(&self, x: &[f64]) -> f64 {
        let mut obj = self.obj_offset;
        for (ci, xi) in self.c.iter().zip(x.iter()) {
            obj += ci * xi;
        }
        obj
    }

    pub fn validate(&self) -> Result<(), SolverError> {
        let n = self.num_variables();
        let m = self.num_constraints();

        if n == 0 {
            return Err(SolverError::ValidationError("Model has 0 variables".into()));
        }
        if self.c.len() != n {
            return Err(SolverError::DimensionMismatch {
                expected: n,
                found: self.c.len(),
            });
        }
        if self.var_bounds.len() != n {
            return Err(SolverError::DimensionMismatch {
                expected: n,
                found: self.var_bounds.len(),
            });
        }
        if self.row_bounds.len() != m {
            return Err(SolverError::DimensionMismatch {
                expected: m,
                found: self.row_bounds.len(),
            });
        }
        if self.a.num_rows != m || self.a.num_cols != n {
            return Err(SolverError::ValidationError(format!(
                "Constraint matrix dimensions ({}, {}) do not match row/col count ({}, {})",
                self.a.num_rows, self.a.num_cols, m, n
            )));
        }

        for (i, b) in self.var_bounds.iter().enumerate() {
            if b.lower > b.upper {
                return Err(SolverError::ValidationError(format!(
                    "Variable '{}' lower bound {} > upper bound {}",
                    self.var_names[i], b.lower, b.upper
                )));
            }
        }

        for (i, b) in self.row_bounds.iter().enumerate() {
            if b.lower > b.upper {
                return Err(SolverError::ValidationError(format!(
                    "Constraint '{}' lower bound {} > upper bound {}",
                    self.row_names[i], b.lower, b.upper
                )));
            }
        }

        Ok(())
    }
}

/// Output solution structure produced by the optimization engine
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LpSolution {
    pub status: SolverStatus,
    pub primal: Vec<f64>,
    pub dual: Vec<f64>,
    pub objective: f64,
    pub iterations: usize,
    pub primal_residual: f64,
    pub dual_residual: f64,
    pub solve_time_secs: f64,
}

impl Default for LpSolution {
    fn default() -> Self {
        Self {
            status: SolverStatus::NotSolved,
            primal: Vec::new(),
            dual: Vec::new(),
            objective: 0.0,
            iterations: 0,
            primal_residual: f64::INFINITY,
            dual_residual: f64::INFINITY,
            solve_time_secs: 0.0,
        }
    }
}
