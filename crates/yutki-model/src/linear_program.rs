//! Mathematical Linear Programming representation and components.
//! Supports general continuous Linear Programs:
//!
//! minimize / maximize   c^T x + offset
//! subject to            row_lower <= A x <= row_upper
//!                       col_lower <=  x  <= col_upper

use crate::bounds::{Bound, Sense};
use crate::error::SolverError;
use crate::problem::LpProblem;
use serde::{Deserialize, Serialize};
use yutki_sparse::CsrMatrix;

/// Objective function for a linear program.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Objective {
    pub sense: Sense,
    pub c: Vec<f64>,
    pub offset: f64,
}

impl Objective {
    pub fn new(sense: Sense, c: Vec<f64>) -> Self {
        Self {
            sense,
            c,
            offset: 0.0,
        }
    }

    pub fn with_offset(sense: Sense, c: Vec<f64>, offset: f64) -> Self {
        Self { sense, c, offset }
    }

    pub fn len(&self) -> usize {
        self.c.len()
    }

    pub fn is_empty(&self) -> bool {
        self.c.is_empty()
    }

    pub fn evaluate(&self, x: &[f64]) -> Result<f64, SolverError> {
        if x.len() != self.c.len() {
            return Err(SolverError::DimensionMismatch {
                expected: self.c.len(),
                found: x.len(),
            });
        }
        let mut sum = self.offset;
        for (ci, xi) in self.c.iter().zip(x.iter()) {
            if !xi.is_finite() {
                return Err(SolverError::NumericalError(format!(
                    "Non-finite variable value in objective evaluation: {xi}"
                )));
            }
            sum += ci * xi;
        }
        Ok(sum)
    }

    pub fn validate(&self, expected_vars: usize) -> Result<(), SolverError> {
        if self.c.len() != expected_vars {
            return Err(SolverError::DimensionMismatch {
                expected: expected_vars,
                found: self.c.len(),
            });
        }
        if !self.offset.is_finite() {
            return Err(SolverError::ValidationError(format!(
                "Objective offset is non-finite: {}",
                self.offset
            )));
        }
        for (j, &cj) in self.c.iter().enumerate() {
            if !cj.is_finite() {
                return Err(SolverError::ValidationError(format!(
                    "Objective coefficient c[{j}] is non-finite: {cj}"
                )));
            }
        }
        Ok(())
    }
}

/// Variable bounds for an LP: col_lower <= x <= col_upper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VariableBounds {
    pub bounds: Vec<Bound>,
    pub names: Vec<String>,
}

impl VariableBounds {
    pub fn new(bounds: Vec<Bound>, names: Vec<String>) -> Self {
        Self { bounds, names }
    }

    pub fn from_bounds(bounds: Vec<Bound>) -> Self {
        let names = (0..bounds.len()).map(|i| format!("x{i}")).collect();
        Self { bounds, names }
    }

    pub fn len(&self) -> usize {
        self.bounds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bounds.is_empty()
    }

    pub fn project(&self, x: &mut [f64]) -> Result<(), SolverError> {
        if x.len() != self.bounds.len() {
            return Err(SolverError::DimensionMismatch {
                expected: self.bounds.len(),
                found: x.len(),
            });
        }
        for (xi, b) in x.iter_mut().zip(self.bounds.iter()) {
            *xi = b.project(*xi);
        }
        Ok(())
    }

    pub fn max_violation(&self, x: &[f64]) -> Result<f64, SolverError> {
        if x.len() != self.bounds.len() {
            return Err(SolverError::DimensionMismatch {
                expected: self.bounds.len(),
                found: x.len(),
            });
        }
        let mut max_viol = 0.0f64;
        for (&xi, b) in x.iter().zip(self.bounds.iter()) {
            let v = b.violation(xi);
            if v > max_viol {
                max_viol = v;
            }
        }
        Ok(max_viol)
    }

    pub fn validate(&self) -> Result<(), SolverError> {
        if self.bounds.len() != self.names.len() {
            return Err(SolverError::DimensionMismatch {
                expected: self.bounds.len(),
                found: self.names.len(),
            });
        }
        for (i, b) in self.bounds.iter().enumerate() {
            if b.lower.is_nan() || b.upper.is_nan() {
                return Err(SolverError::ValidationError(format!(
                    "Variable '{}' (index {i}) has NaN bound: lower={}, upper={}",
                    self.names[i], b.lower, b.upper
                )));
            }
            if b.lower > b.upper {
                return Err(SolverError::ValidationError(format!(
                    "Variable '{}' (index {i}) lower bound {} > upper bound {}",
                    self.names[i], b.lower, b.upper
                )));
            }
        }
        Ok(())
    }
}

/// Constraint bounds for an LP: row_lower <= A x <= row_upper.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstraintBounds {
    pub bounds: Vec<Bound>,
    pub names: Vec<String>,
}

impl ConstraintBounds {
    pub fn new(bounds: Vec<Bound>, names: Vec<String>) -> Self {
        Self { bounds, names }
    }

    pub fn from_bounds(bounds: Vec<Bound>) -> Self {
        let names = (0..bounds.len()).map(|i| format!("r{i}")).collect();
        Self { bounds, names }
    }

    pub fn len(&self) -> usize {
        self.bounds.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bounds.is_empty()
    }

    pub fn project(&self, ax: &mut [f64]) -> Result<(), SolverError> {
        if ax.len() != self.bounds.len() {
            return Err(SolverError::DimensionMismatch {
                expected: self.bounds.len(),
                found: ax.len(),
            });
        }
        for (axi, b) in ax.iter_mut().zip(self.bounds.iter()) {
            *axi = b.project(*axi);
        }
        Ok(())
    }

    pub fn max_violation(&self, ax: &[f64]) -> Result<f64, SolverError> {
        if ax.len() != self.bounds.len() {
            return Err(SolverError::DimensionMismatch {
                expected: self.bounds.len(),
                found: ax.len(),
            });
        }
        let mut max_viol = 0.0f64;
        for (&axi, b) in ax.iter().zip(self.bounds.iter()) {
            let v = b.violation(axi);
            if v > max_viol {
                max_viol = v;
            }
        }
        Ok(max_viol)
    }

    pub fn validate(&self) -> Result<(), SolverError> {
        if self.bounds.len() != self.names.len() {
            return Err(SolverError::DimensionMismatch {
                expected: self.bounds.len(),
                found: self.names.len(),
            });
        }
        for (i, b) in self.bounds.iter().enumerate() {
            if b.lower.is_nan() || b.upper.is_nan() {
                return Err(SolverError::ValidationError(format!(
                    "Constraint '{}' (index {i}) has NaN bound: lower={}, upper={}",
                    self.names[i], b.lower, b.upper
                )));
            }
            if b.lower > b.upper {
                return Err(SolverError::ValidationError(format!(
                    "Constraint '{}' (index {i}) lower bound {} > upper bound {}",
                    self.names[i], b.lower, b.upper
                )));
            }
        }
        Ok(())
    }
}

/// Sovereign Linear Program representation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LinearProgram {
    pub name: String,
    pub objective: Objective,
    pub variable_bounds: VariableBounds,
    pub constraint_bounds: ConstraintBounds,
    pub a: CsrMatrix,
}

impl LinearProgram {
    pub fn new(
        name: String,
        objective: Objective,
        variable_bounds: VariableBounds,
        constraint_bounds: ConstraintBounds,
        a: CsrMatrix,
    ) -> Result<Self, SolverError> {
        let lp = Self {
            name,
            objective,
            variable_bounds,
            constraint_bounds,
            a,
        };
        lp.validate()?;
        Ok(lp)
    }

    pub fn num_variables(&self) -> usize {
        self.variable_bounds.len()
    }

    pub fn num_constraints(&self) -> usize {
        self.constraint_bounds.len()
    }

    pub fn num_nonzeros(&self) -> usize {
        self.a.nnz()
    }

    pub fn density(&self) -> f64 {
        self.a.density()
    }

    pub fn evaluate_objective(&self, x: &[f64]) -> Result<f64, SolverError> {
        self.objective.evaluate(x)
    }

    pub fn validate(&self) -> Result<(), SolverError> {
        let n = self.num_variables();
        let m = self.num_constraints();

        if n == 0 {
            return Err(SolverError::ValidationError("Model has 0 variables".into()));
        }

        self.objective.validate(n)?;
        self.variable_bounds.validate()?;
        self.constraint_bounds.validate()?;

        if self.a.num_rows != m || self.a.num_cols != n {
            return Err(SolverError::ValidationError(format!(
                "Constraint matrix dimensions ({}, {}) do not match constraints ({m}) and variables ({n})",
                self.a.num_rows, self.a.num_cols
            )));
        }

        self.a
            .validate()
            .map_err(|e| SolverError::ValidationError(e.to_string()))?;

        Ok(())
    }

    pub fn to_lp_problem(&self) -> LpProblem {
        LpProblem {
            name: self.name.clone(),
            sense: self.objective.sense,
            obj_offset: self.objective.offset,
            c: self.objective.c.clone(),
            var_names: self.variable_bounds.names.clone(),
            var_bounds: self.variable_bounds.bounds.clone(),
            row_names: self.constraint_bounds.names.clone(),
            row_bounds: self.constraint_bounds.bounds.clone(),
            a: self.a.clone(),
        }
    }

    pub fn from_lp_problem(problem: LpProblem) -> Result<Self, SolverError> {
        let objective = Objective::with_offset(problem.sense, problem.c, problem.obj_offset);
        let variable_bounds = VariableBounds::new(problem.var_bounds, problem.var_names);
        let constraint_bounds = ConstraintBounds::new(problem.row_bounds, problem.row_names);
        Self::new(
            problem.name,
            objective,
            variable_bounds,
            constraint_bounds,
            problem.a,
        )
    }

    /// Load and parse a LinearProgram directly from an MPS file on disk.
    pub fn from_mps_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, SolverError> {
        crate::mps::parse_mps_file_to_lp(path)
    }

    /// Parse a LinearProgram directly from an MPS formatted string.
    pub fn from_mps_str(content: &str) -> Result<Self, SolverError> {
        crate::mps::parse_mps_str_to_lp(content)
    }
}

impl From<LpProblem> for LinearProgram {
    fn from(p: LpProblem) -> Self {
        Self::from_lp_problem(p).expect("LpProblem should be valid LinearProgram")
    }
}

impl From<LinearProgram> for LpProblem {
    fn from(lp: LinearProgram) -> Self {
        lp.to_lp_problem()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yutki_sparse::CooMatrix;

    fn sample_valid_matrix() -> CsrMatrix {
        let mut coo = CooMatrix::new(2, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 2.0).unwrap();
        coo.add_entry(1, 1, 3.0).unwrap();
        CsrMatrix::from_coo(&coo).unwrap()
    }

    #[test]
    fn test_valid_linear_program_creation() {
        let a = sample_valid_matrix();
        let obj = Objective::new(Sense::Minimize, vec![10.0, 20.0]);
        let var_bounds =
            VariableBounds::from_bounds(vec![Bound::non_negative(), Bound::new(0.0, 5.0)]);
        let con_bounds =
            ConstraintBounds::from_bounds(vec![Bound::upper_only(10.0), Bound::fixed(3.0)]);

        let lp = LinearProgram::new("test_lp".into(), obj, var_bounds, con_bounds, a).unwrap();
        assert_eq!(lp.num_variables(), 2);
        assert_eq!(lp.num_constraints(), 2);
        assert_eq!(lp.num_nonzeros(), 3);

        let x = vec![1.0, 2.0];
        assert_eq!(lp.evaluate_objective(&x).unwrap(), 50.0);
    }

    #[test]
    fn test_dimension_mismatch_detection() {
        let a = sample_valid_matrix(); // 2 rows, 2 cols
        let bad_obj = Objective::new(Sense::Minimize, vec![10.0]); // 1 cost entry instead of 2
        let var_bounds =
            VariableBounds::from_bounds(vec![Bound::non_negative(), Bound::non_negative()]);
        let con_bounds =
            ConstraintBounds::from_bounds(vec![Bound::upper_only(10.0), Bound::fixed(3.0)]);

        let res = LinearProgram::new("bad_dim".into(), bad_obj, var_bounds, con_bounds, a);
        assert!(res.is_err());
    }

    #[test]
    fn test_inverted_bounds_detection() {
        let a = sample_valid_matrix();
        let obj = Objective::new(Sense::Minimize, vec![1.0, 2.0]);
        // lower 5.0 > upper 2.0
        let bad_vars =
            VariableBounds::from_bounds(vec![Bound::new(5.0, 2.0), Bound::non_negative()]);
        let con_bounds =
            ConstraintBounds::from_bounds(vec![Bound::upper_only(10.0), Bound::fixed(3.0)]);

        let res = LinearProgram::new("inverted_bounds".into(), obj, bad_vars, con_bounds, a);
        assert!(res.is_err());
    }

    #[test]
    fn test_nan_in_bounds_detection() {
        let a = sample_valid_matrix();
        let obj = Objective::new(Sense::Minimize, vec![1.0, 2.0]);
        let bad_con =
            ConstraintBounds::from_bounds(vec![Bound::new(0.0, f64::NAN), Bound::fixed(3.0)]);
        let var_bounds =
            VariableBounds::from_bounds(vec![Bound::non_negative(), Bound::non_negative()]);

        let res = LinearProgram::new("nan_bounds".into(), obj, var_bounds, bad_con, a);
        assert!(res.is_err());
    }

    #[test]
    fn test_nan_in_objective_detection() {
        let a = sample_valid_matrix();
        let bad_obj = Objective::new(Sense::Minimize, vec![f64::NAN, 2.0]);
        let var_bounds =
            VariableBounds::from_bounds(vec![Bound::non_negative(), Bound::non_negative()]);
        let con_bounds =
            ConstraintBounds::from_bounds(vec![Bound::upper_only(10.0), Bound::fixed(3.0)]);

        let res = LinearProgram::new("nan_obj".into(), bad_obj, var_bounds, con_bounds, a);
        assert!(res.is_err());
    }

    #[test]
    fn test_roundtrip_conversion_with_lp_problem() {
        let a = sample_valid_matrix();
        let obj = Objective::with_offset(Sense::Maximize, vec![5.0, 7.0], 100.0);
        let var_bounds =
            VariableBounds::from_bounds(vec![Bound::non_negative(), Bound::new(1.0, 4.0)]);
        let con_bounds =
            ConstraintBounds::from_bounds(vec![Bound::upper_only(20.0), Bound::fixed(12.0)]);

        let lp = LinearProgram::new("roundtrip".into(), obj, var_bounds, con_bounds, a).unwrap();
        let problem = lp.to_lp_problem();
        let restored = LinearProgram::from_lp_problem(problem).unwrap();
        assert_eq!(lp, restored);
    }
}
