//! Primal-Dual Infeasible Interior Point Method (IPM) for Linear Programs.
//!
//! Implements:
//! - Canonical KKT system formulation
//! - Normal equations sparse solver with regularization: (A D^2 A^T) dy = r_y
//! - Fraction-to-boundary step length selection
//! - Primal-dual variable updates
//! - Numerical health monitoring with automatic stagnation detection and fallback routing

use serde::{Deserialize, Serialize};
use std::time::Instant;
use yukti_model::{LinearProgram, LpSolution, SolverError, SolverStatus};
use yukti_sparse::factorization::LuDecomposition;

use crate::simplex::StandardFormLp;

/// Termination outcome of the Interior Point Method solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IpmStatus {
    Optimal,
    IterationLimit,
    TimeLimit,
    NumericalStagnation,
}

impl std::fmt::Display for IpmStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IpmStatus::Optimal => write!(f, "OPTIMAL"),
            IpmStatus::IterationLimit => write!(f, "ITERATION_LIMIT"),
            IpmStatus::TimeLimit => write!(f, "TIME_LIMIT"),
            IpmStatus::NumericalStagnation => write!(f, "NUMERICAL_STAGNATION"),
        }
    }
}

/// Configuration options for the Interior Point Method solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IpmOptions {
    /// Maximum IPM iterations (default: 200)
    pub max_iterations: usize,
    /// Absolute and relative feasibility/optimality tolerance (default: 1e-6)
    pub tolerance: f64,
    /// Time limit in seconds (default: 120.0)
    pub time_limit_secs: f64,
    /// Centering parameter sigma (default: 0.2)
    pub sigma: f64,
    /// Fraction-to-boundary parameter tau in (0, 1) (default: 0.995)
    pub tau: f64,
}

impl Default for IpmOptions {
    fn default() -> Self {
        Self {
            max_iterations: 200,
            tolerance: 1e-6,
            time_limit_secs: 120.0,
            sigma: 0.2,
            tau: 0.995,
        }
    }
}

/// Detailed execution telemetry from the Interior Point Method solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IpmTelemetry {
    pub status: IpmStatus,
    pub iterations: usize,
    pub solve_time_secs: f64,
    pub primal_residual: f64,
    pub dual_residual: f64,
    pub duality_gap: f64,
    pub objective_value: f64,
}

/// Primal-Dual Interior Point Method solver engine.
pub struct IpmSolver;

impl IpmSolver {
    /// Solve a continuous Linear Program using the Infeasible Primal-Dual Interior Point Method.
    pub fn solve(
        lp: &LinearProgram,
        options: &IpmOptions,
    ) -> Result<(LpSolution, IpmTelemetry), SolverError> {
        let start_time = Instant::now();

        // 1. Transform to standard canonical form: min c^T x s.t. A x = b, x >= 0
        let std = StandardFormLp::from_linear_program(lp)?;
        let m = std.num_constraints;
        let n_dec = std.num_decision_vars;
        // Infeasible IPM does not need artificial variables; columns 0..(n_dec + num_slacks)
        let n_vars = n_dec + std.num_slacks;

        if m == 0 || n_vars == 0 {
            return Err(SolverError::ValidationError(
                "Cannot solve LP with zero constraints or variables via IPM".to_string(),
            ));
        }

        let a_cols = &std.cols[0..n_vars];
        let c = &std.c[0..n_vars];
        let b = &std.b;

        // Calculate baseline norms for relative stopping criteria
        let norm_b = b.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1.0);
        let norm_c = c.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1.0);

        // 2. Initialize primal-dual strictly interior point (x > 0, s > 0, y = 0)
        let init_x_val = 1.0f64.max(norm_b / (n_vars as f64).sqrt());
        let init_s_val = 1.0f64.max(norm_c / (n_vars as f64).sqrt());

        let mut x = vec![init_x_val; n_vars];
        let mut s = vec![init_s_val; n_vars];
        let mut y = vec![0.0f64; m];

        let mut status = IpmStatus::IterationLimit;
        let mut final_primal_res = 1.0;
        let mut final_dual_res = 1.0;
        let mut final_gap = 1.0;
        let mut final_obj = 0.0;
        let mut iter_count = 0;

        let mut lu = LuDecomposition::with_dimension(m);
        let mut scales = vec![0.0f64; m];

        // 3. Main Interior-Point Iteration Loop
        for iter in 0..options.max_iterations {
            iter_count = iter + 1;

            if start_time.elapsed().as_secs_f64() > options.time_limit_secs {
                status = IpmStatus::TimeLimit;
                break;
            }

            // Primal residual: r_p = b - A x
            let mut r_p = b.clone();
            for j in 0..n_vars {
                let xj = x[j];
                for &(row, val) in &a_cols[j] {
                    r_p[row] -= val * xj;
                }
            }

            // Dual residual: r_d = c - A^T y - s
            let mut r_d = c.to_vec();
            for j in 0..n_vars {
                let mut at_y_j = 0.0;
                for &(row, val) in &a_cols[j] {
                    at_y_j += val * y[row];
                }
                r_d[j] -= at_y_j + s[j];
            }

            // Complementarity measure: mu = (x^T s) / n
            let mut xs_dot = 0.0;
            for j in 0..n_vars {
                xs_dot += x[j] * s[j];
            }
            let mu = xs_dot / (n_vars as f64);

            let primal_res_norm = r_p.iter().map(|&v| v * v).sum::<f64>().sqrt();
            let dual_res_norm = r_d.iter().map(|&v| v * v).sum::<f64>().sqrt();

            final_primal_res = primal_res_norm / norm_b;
            final_dual_res = dual_res_norm / norm_c;

            let c_dot_x: f64 = c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();
            let b_dot_y: f64 = b.iter().zip(y.iter()).map(|(&bi, &yi)| bi * yi).sum();
            final_gap = (c_dot_x - b_dot_y).abs() / (1.0 + c_dot_x.abs());

            let current_obj = if std.is_maximization {
                -(c_dot_x + std.obj_offset)
            } else {
                c_dot_x + std.obj_offset
            };
            final_obj = current_obj;

            // Optimality termination check
            if final_primal_res < options.tolerance
                && final_dual_res < options.tolerance
                && final_gap < options.tolerance
            {
                status = IpmStatus::Optimal;
                break;
            }

            // 4. Form Normal Equations: (A D^2 A^T + delta * I) \Delta y = r_y
            // D^2 = diag(x_j / s_j)
            let mut m_dense = vec![0.0f64; m * m];
            for j in 0..n_vars {
                let w = x[j] / s[j];
                let col_entries = &a_cols[j];
                for &(r1, v1) in col_entries {
                    let w_v1 = w * v1;
                    for &(r2, v2) in col_entries {
                        m_dense[r1 * m + r2] += w_v1 * v2;
                    }
                }
            }

            // Regularization for numerical stability: delta = 1e-10
            for i in 0..m {
                m_dense[i * m + i] += 1e-10;
            }

            // Right-hand side: r_y = r_p + A (D^2 r_d - S^{-1} r_xs)
            // r_{xs, j} = sigma * mu - x_j * s_j
            let mut u = vec![0.0f64; n_vars];
            for j in 0..n_vars {
                let r_xs_j = options.sigma * mu - x[j] * s[j];
                u[j] = (x[j] * r_d[j] - r_xs_j) / s[j];
            }

            let mut r_y = r_p.clone();
            for j in 0..n_vars {
                let uj = u[j];
                for &(row, val) in &a_cols[j] {
                    r_y[row] += val * uj;
                }
            }

            // 5. Solve Normal Equations via LU Factorization
            if LuDecomposition::factorize_dense_into(m, &m_dense, 1e-12, &mut lu, &mut scales)
                .is_err()
            {
                status = IpmStatus::NumericalStagnation;
                break;
            }

            let delta_y = match lu.solve(&r_y) {
                Ok(sol) => sol,
                Err(_) => {
                    status = IpmStatus::NumericalStagnation;
                    break;
                }
            };

            // 6. Recover \Delta s and \Delta x
            // \Delta s = r_d - A^T \Delta y
            // \Delta x_j = (r_{xs, j} - x_j \Delta s_j) / s_j
            let mut delta_s = r_d.clone();
            for j in 0..n_vars {
                let mut at_dy_j = 0.0;
                for &(row, val) in &a_cols[j] {
                    at_dy_j += val * delta_y[row];
                }
                delta_s[j] -= at_dy_j;
            }

            let mut delta_x = vec![0.0f64; n_vars];
            for j in 0..n_vars {
                let r_xs_j = options.sigma * mu - x[j] * s[j];
                delta_x[j] = (r_xs_j - x[j] * delta_s[j]) / s[j];
            }

            // 7. Fraction-to-Boundary Step Length Selection
            let mut alpha_p = 1.0f64;
            let mut alpha_d = 1.0f64;
            for j in 0..n_vars {
                if delta_x[j] < 0.0 {
                    let step = -options.tau * (x[j] / delta_x[j]);
                    if step < alpha_p {
                        alpha_p = step;
                    }
                }
                if delta_s[j] < 0.0 {
                    let step = -options.tau * (s[j] / delta_s[j]);
                    if step < alpha_d {
                        alpha_d = step;
                    }
                }
            }

            // Numerical health & stagnation check
            if alpha_p < 1e-8 || alpha_d < 1e-8 || !alpha_p.is_finite() || !alpha_d.is_finite() {
                status = IpmStatus::NumericalStagnation;
                break;
            }

            // 8. Variable Update
            for j in 0..n_vars {
                x[j] += alpha_p * delta_x[j];
                s[j] += alpha_d * delta_s[j];
            }
            for i in 0..m {
                y[i] += alpha_d * delta_y[i];
            }
        }

        // 9. Reconstruct Original Model Decision Solution
        let mut primal_solution = vec![0.0f64; lp.num_variables()];
        for j in 0..n_dec {
            primal_solution[j] = x[j] + std.var_shifts[j];
        }

        let mut dual_solution = vec![0.0f64; m];
        for i in 0..m {
            let mut val = y[i];
            if std.row_sign_flips[i] {
                val = -val;
            }
            if std.is_maximization {
                val = -val;
            }
            dual_solution[i] = val;
        }

        let solver_status = match status {
            IpmStatus::Optimal => SolverStatus::Optimal,
            IpmStatus::IterationLimit => SolverStatus::IterationLimit,
            IpmStatus::TimeLimit => SolverStatus::TimeLimit,
            IpmStatus::NumericalStagnation => SolverStatus::NumericalFailure,
        };

        let total_time = start_time.elapsed().as_secs_f64();

        let telemetry = IpmTelemetry {
            status,
            iterations: iter_count,
            solve_time_secs: total_time,
            primal_residual: final_primal_res,
            dual_residual: final_dual_res,
            duality_gap: final_gap,
            objective_value: final_obj,
        };

        let solution = LpSolution {
            status: solver_status,
            primal: primal_solution,
            dual: dual_solution,
            objective: final_obj,
            iterations: iter_count,
            primal_residual: final_primal_res,
            dual_residual: final_dual_res,
            solve_time_secs: total_time,
        };

        Ok((solution, telemetry))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yukti_model::{Bound, ConstraintBounds, Objective, Sense, VariableBounds};
    use yukti_sparse::CsrMatrix;

    #[test]
    fn test_ipm_analytical_2d_lp() {
        // min -x1 - 2*x2
        // s.t. x1 + x2 <= 4
        //      x1 <= 3
        //      x1, x2 >= 0
        // Optimal: x1 = 0, x2 = 4, obj = -8.0
        let a = CsrMatrix {
            num_rows: 2,
            num_cols: 2,
            row_ptrs: vec![0, 2, 3],
            col_indices: vec![0, 1, 0],
            values: vec![1.0, 1.0, 1.0],
        };

        let lp = LinearProgram {
            name: "test_ipm_2d".to_string(),
            objective: Objective::new(Sense::Minimize, vec![-1.0, -2.0]),
            constraint_bounds: ConstraintBounds::from_bounds(vec![
                Bound::new(f64::NEG_INFINITY, 4.0),
                Bound::new(f64::NEG_INFINITY, 3.0),
            ]),
            variable_bounds: VariableBounds::from_bounds(vec![
                Bound::non_negative(),
                Bound::non_negative(),
            ]),
            a,
        };

        let opts = IpmOptions {
            max_iterations: 100,
            tolerance: 1e-4,
            ..Default::default()
        };

        let (sol, telem) = IpmSolver::solve(&lp, &opts).expect("solve analytical 2d lp");
        assert_eq!(telem.status, IpmStatus::Optimal);
        assert!((sol.objective - (-8.0)).abs() < 1e-2);
        assert!((sol.primal[0] - 0.0).abs() < 1e-2);
        assert!((sol.primal[1] - 4.0).abs() < 1e-2);
    }

    #[test]
    fn test_ipm_equality_constraint_lp() {
        // min 2*x1 + 3*x2
        // s.t. x1 + x2 = 5
        //      x1, x2 >= 0
        // Optimal: x1 = 5, x2 = 0, obj = 10.0
        let a = CsrMatrix {
            num_rows: 1,
            num_cols: 2,
            row_ptrs: vec![0, 2],
            col_indices: vec![0, 1],
            values: vec![1.0, 1.0],
        };

        let lp = LinearProgram {
            name: "test_ipm_eq".to_string(),
            objective: Objective::new(Sense::Minimize, vec![2.0, 3.0]),
            constraint_bounds: ConstraintBounds::from_bounds(vec![Bound::fixed(5.0)]),
            variable_bounds: VariableBounds::from_bounds(vec![
                Bound::non_negative(),
                Bound::non_negative(),
            ]),
            a,
        };

        let opts = IpmOptions {
            max_iterations: 100,
            tolerance: 1e-4,
            ..Default::default()
        };

        let (sol, telem) = IpmSolver::solve(&lp, &opts).expect("solve equality lp");
        assert_eq!(telem.status, IpmStatus::Optimal);
        assert!((sol.objective - 10.0).abs() < 1e-2);
        assert!((sol.primal[0] - 5.0).abs() < 1e-2);
        assert!((sol.primal[1] - 0.0).abs() < 1e-2);
    }
}
