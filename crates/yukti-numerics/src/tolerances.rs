use serde::{Deserialize, Serialize};
use yukti_model::SolverError;

use crate::check_finite_slice;

/// Centralized configuration for all numerical thresholds in the solver.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NumericalTolerances {
    /// Relative primal feasibility tolerance (default: 1e-6)
    pub primal_tol: f64,
    /// Relative dual feasibility tolerance (default: 1e-6)
    pub dual_tol: f64,
    /// Relative duality gap tolerance (default: 1e-6)
    pub gap_tol: f64,
    /// Zero threshold below which numbers are treated as exact 0.0 (default: 1e-12)
    pub zero_tol: f64,
    /// Numerical threshold above which values are treated as infinity (default: 1e20)
    pub infinity_threshold: f64,
    /// Maximum condition ratio max(|A_ij|)/min(|A_ij|) before warning (default: 1e8)
    pub max_condition_ratio: f64,
    /// Stagnation progress window in iterations (default: 50)
    pub stagnation_window: usize,
    /// Minimum relative progress over stagnation window (default: 1e-4)
    pub stagnation_threshold: f64,
    /// Maximum allowed iterations before timeout / limit (default: 50,000)
    pub max_iterations: usize,
    /// Maximum wall-clock time limit in seconds (default: 300.0)
    pub time_limit_secs: f64,
}

impl Default for NumericalTolerances {
    fn default() -> Self {
        Self {
            primal_tol: 1e-6,
            dual_tol: 1e-6,
            gap_tol: 1e-6,
            zero_tol: 1e-12,
            infinity_threshold: 1e20,
            max_condition_ratio: 1e8,
            stagnation_window: 50,
            stagnation_threshold: 1e-4,
            max_iterations: 50_000,
            time_limit_secs: 300.0,
        }
    }
}

/// Tolerance policy enforcing feasibility, zero-thresholds, and infinity boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TolerancePolicy {
    pub tolerances: NumericalTolerances,
}

impl Default for TolerancePolicy {
    fn default() -> Self {
        Self::new(NumericalTolerances::default())
    }
}

impl TolerancePolicy {
    pub fn new(tolerances: NumericalTolerances) -> Self {
        Self { tolerances }
    }

    /// Check if a value is within zero tolerance.
    pub fn is_zero(&self, val: f64) -> bool {
        val.abs() <= self.tolerances.zero_tol
    }

    /// Check if a value exceeds infinity threshold.
    pub fn is_infinite(&self, val: f64) -> bool {
        val.abs() >= self.tolerances.infinity_threshold
    }

    /// Check finite numbers in a vector slice.
    pub fn check_finite(&self, target: &str, slice: &[f64]) -> Result<(), SolverError> {
        check_finite_slice(target, slice)
    }

    /// Check if relative residuals satisfy optimality/convergence criteria.
    pub fn is_converged(&self, primal_rel: f64, dual_rel: f64, gap_rel: f64) -> bool {
        primal_rel.is_finite()
            && dual_rel.is_finite()
            && gap_rel.is_finite()
            && primal_rel <= self.tolerances.primal_tol
            && dual_rel <= self.tolerances.dual_tol
            && gap_rel <= self.tolerances.gap_tol
    }
}
