//! Numerical tolerances, condition fingerprinting, Ruiz equilibration, and safety sentinels.

pub mod monitor;
pub mod restart;
pub mod stagnation;
pub mod tolerances;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use yukti_model::{LpProblem, SolverError};
use yukti_sparse::CsrMatrix;

pub use monitor::{MonitorMetrics, MonitorVerdict, NumericalMonitor};
pub use restart::{RestartAction, RestartManager, RestartReason};
pub use stagnation::StagnationDetector;
pub use tolerances::{NumericalTolerances, TolerancePolicy};

#[derive(Error, Debug, PartialEq)]
pub enum NumericalError {
    #[error("Non-finite value encountered in {target} at index {index}: {value}")]
    NonFiniteEncountered {
        target: String,
        index: usize,
        value: f64,
    },

    #[error(
        "Severe matrix ill-conditioning detected: condition ratio {ratio:.2e} > {max_ratio:.2e}"
    )]
    SevereIllConditioning { ratio: f64, max_ratio: f64 },
}

/// Dynamic range summary and conditioning fingerprint for a problem instance.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProblemFingerprint {
    pub num_rows: usize,
    pub num_cols: usize,
    pub num_nonzeros: usize,
    pub density: f64,
    pub min_abs_a: f64,
    pub max_abs_a: f64,
    pub condition_ratio: f64,
    pub min_abs_c: f64,
    pub max_abs_c: f64,
}

impl ProblemFingerprint {
    pub fn compute(problem: &LpProblem) -> Self {
        let mut min_abs_a = f64::INFINITY;
        let mut max_abs_a = 0.0f64;

        for &val in &problem.a.values {
            let abs_v = val.abs();
            if abs_v > 0.0 {
                if abs_v < min_abs_a {
                    min_abs_a = abs_v;
                }
                if abs_v > max_abs_a {
                    max_abs_a = abs_v;
                }
            }
        }

        if min_abs_a == f64::INFINITY {
            min_abs_a = 0.0;
        }

        let condition_ratio = if min_abs_a > 0.0 {
            max_abs_a / min_abs_a
        } else {
            1.0
        };

        let mut min_abs_c = f64::INFINITY;
        let mut max_abs_c = 0.0f64;
        for &val in &problem.c {
            let abs_v = val.abs();
            if abs_v > 0.0 {
                if abs_v < min_abs_c {
                    min_abs_c = abs_v;
                }
                if abs_v > max_abs_c {
                    max_abs_c = abs_v;
                }
            }
        }
        if min_abs_c == f64::INFINITY {
            min_abs_c = 0.0;
        }

        Self {
            num_rows: problem.num_constraints(),
            num_cols: problem.num_variables(),
            num_nonzeros: problem.num_nonzeros(),
            density: problem.density(),
            min_abs_a,
            max_abs_a,
            condition_ratio,
            min_abs_c,
            max_abs_c,
        }
    }
}

/// Verify that all elements of a slice are finite numbers (not NaN, not Inf).
pub fn check_finite_slice(name: &str, slice: &[f64]) -> Result<(), SolverError> {
    for (i, &val) in slice.iter().enumerate() {
        if !val.is_finite() {
            return Err(SolverError::NumericalError(format!(
                "Non-finite value ({val}) encountered in {name} at index {i}"
            )));
        }
    }
    Ok(())
}

/// Diagonal scaling vectors derived from Ruiz equilibration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RuizScaling {
    pub row_scale: Vec<f64>,
    pub col_scale: Vec<f64>,
}

/// Compute Ruiz equilibration diagonal scalers to normalize L_inf row and col norms.
pub fn compute_ruiz_scaling(a: &CsrMatrix, iterations: usize) -> RuizScaling {
    let m = a.num_rows;
    let n = a.num_cols;

    let mut row_scale = vec![1.0; m];
    let mut col_scale = vec![1.0; n];

    if a.nnz() == 0 {
        return RuizScaling {
            row_scale,
            col_scale,
        };
    }

    for _ in 0..iterations {
        // Row max norms
        for (i, r_scale) in row_scale.iter_mut().enumerate().take(m) {
            let start = a.row_ptrs[i];
            let end = a.row_ptrs[i + 1];
            let mut row_max = 0.0f64;
            for k in start..end {
                let j = a.col_indices[k];
                let scaled_val = a.values[k].abs() * *r_scale * col_scale[j];
                if scaled_val > row_max {
                    row_max = scaled_val;
                }
            }
            if row_max > 1e-12 {
                *r_scale *= 1.0 / row_max.sqrt();
            }
        }

        // Col max norms
        let mut col_max = vec![0.0f64; n];
        for (i, &r_scale) in row_scale.iter().enumerate().take(m) {
            let start = a.row_ptrs[i];
            let end = a.row_ptrs[i + 1];
            for k in start..end {
                let j = a.col_indices[k];
                let scaled_val = a.values[k].abs() * r_scale * col_scale[j];
                if scaled_val > col_max[j] {
                    col_max[j] = scaled_val;
                }
            }
        }

        for (c_scale, &c_max) in col_scale.iter_mut().zip(col_max.iter()) {
            if c_max > 1e-12 {
                *c_scale *= 1.0 / c_max.sqrt();
            }
        }
    }

    RuizScaling {
        row_scale,
        col_scale,
    }
}

/// Compute Pock-Chambolle diagonal step-sizes for guaranteed convergence:
/// sigma_i = 1 / sum_j |A_ij|, tau_j = 1 / sum_i |A_ij|
pub fn compute_pock_chambolle_steps(a: &CsrMatrix) -> (Vec<f64>, Vec<f64>) {
    let m = a.num_rows;
    let n = a.num_cols;

    let mut row_sums = vec![0.0f64; m];
    let mut col_sums = vec![0.0f64; n];

    for (i, r_sum) in row_sums.iter_mut().enumerate().take(m) {
        let start = a.row_ptrs[i];
        let end = a.row_ptrs[i + 1];
        for k in start..end {
            let abs_v = a.values[k].abs();
            *r_sum += abs_v;
            col_sums[a.col_indices[k]] += abs_v;
        }
    }

    let mut sigma = vec![1.0; m];
    for (s, &r_sum) in sigma.iter_mut().zip(row_sums.iter()) {
        if r_sum > 1e-12 {
            *s = 1.0 / r_sum;
        }
    }

    let mut tau = vec![1.0; n];
    for (t, &c_sum) in tau.iter_mut().zip(col_sums.iter()) {
        if c_sum > 1e-12 {
            *t = 1.0 / c_sum;
        }
    }

    (sigma, tau)
}

#[cfg(test)]
mod tests {
    use super::*;
    use yukti_sparse::CooMatrix;

    #[test]
    fn test_finite_check() {
        let good = vec![1.0, 2.0, -3.5, 0.0];
        assert!(check_finite_slice("test", &good).is_ok());

        let bad = vec![1.0, f64::NAN, 3.0];
        assert!(check_finite_slice("test", &bad).is_err());
    }

    #[test]
    fn test_ruiz_scaling() {
        let mut coo = CooMatrix::new(2, 2);
        coo.add_entry(0, 0, 100.0).unwrap();
        coo.add_entry(1, 1, 0.01).unwrap();
        let csr = CsrMatrix::from_coo(&coo).unwrap();

        let scaling = compute_ruiz_scaling(&csr, 10);
        assert_eq!(scaling.row_scale.len(), 2);
        assert_eq!(scaling.col_scale.len(), 2);
        assert!(scaling.row_scale[0] < 1.0);
        assert!(scaling.row_scale[1] > 1.0);
    }

    #[test]
    fn test_tolerance_policy() {
        let policy = TolerancePolicy::default();
        assert!(policy.is_zero(1e-15));
        assert!(!policy.is_zero(1e-6));
        assert!(policy.is_infinite(1e25));
        assert!(!policy.is_infinite(1e5));
        assert!(policy.is_converged(1e-7, 1e-7, 1e-7));
        assert!(!policy.is_converged(1e-5, 1e-7, 1e-7));
    }

    #[test]
    fn test_stagnation_detector() {
        let mut detector = StagnationDetector::new(5, 1e-4);
        for _ in 0..4 {
            assert!(!detector.record(0.1, 0.1, 0.1, 10.0, 0.01, 0.01));
        }
        // Fifth identical point triggers stagnation
        let stagnant = detector.record(0.1, 0.1, 0.1, 10.0, 0.01, 0.01);
        assert!(stagnant);
        assert_eq!(detector.consecutive_stagnations, 1);
    }

    #[test]
    fn test_restart_manager_and_safe_recovery() {
        let mut rm = RestartManager::new(2, 2, 2);
        rm.checkpoint_best(10, 0.05, &[1.0, 2.0], &[0.5, 0.5]);
        rm.set_baseline(0.05);

        // Routine acceleration on 20% progress
        let action = rm.evaluate(0.03, false);
        assert_eq!(
            action,
            RestartAction::RestartToAverage {
                reason: RestartReason::ProgressAcceleration
            }
        );

        // Safe recovery on stagnation
        let action2 = rm.evaluate(0.03, true);
        match action2 {
            RestartAction::RecoverToBest {
                best_iteration,
                best_err,
                reason,
            } => {
                assert_eq!(best_iteration, 10);
                assert!((best_err - 0.05).abs() < 1e-6);
                assert_eq!(reason, RestartReason::StagnationRecovery);
            }
            _ => panic!("Expected RecoverToBest"),
        }
    }

    #[test]
    fn test_numerical_monitor_nan_sentinel() {
        let policy = TolerancePolicy::default();
        let mut monitor = NumericalMonitor::new(policy, 2, 2, &[0.0, 0.0], &[0.0, 0.0]);

        // Healthy step
        let v1 = monitor.check_step(1, &[1.0, 1.0], &[0.5, 0.5], (0.1, 0.1, 0.1), 5.0);
        assert!(matches!(v1, MonitorVerdict::Healthy(_)));

        // Non-finite step
        let v2 = monitor.check_step(2, &[f64::NAN, 1.0], &[0.5, 0.5], (0.1, 0.1, 0.1), 5.0);
        assert!(matches!(v2, MonitorVerdict::NonFiniteDetected { .. }));
    }
}
