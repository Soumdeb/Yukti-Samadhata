//! Primal-Dual Hybrid Gradient (PDHG / PDLP-style) optimization solver.
//! Indigenous first-order saddle-point engine for continuous linear programming.
//! Operates purely on the sparse matrix abstraction with zero CUDA dependency.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use yukti_gpu::{select_backend, BackendTimingProfile, ComputeBackend};
use yukti_model::{
    BackendType, Bound, LinearProgram, LpProblem, LpSolution, Sense, SolverError, SolverStatus,
};
use yukti_numerics::{MonitorVerdict, NumericalMonitor, NumericalTolerances, TolerancePolicy};
use yukti_sparse::{norm_l2, CsrMatrix};
use yukti_transform::{postsolve, presolve};

pub mod ipm;
pub use ipm::{IpmOptions, IpmSolver, IpmStatus, IpmTelemetry};

pub mod simplex;
pub use simplex::{RevisedSimplexSolver, SimplexOptions, SimplexStatus, SimplexTelemetry};

pub mod strategy;
pub use strategy::*;

/// Optimization solver algorithm selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SolverAlgorithm {
    /// Automatically select optimal algorithm based on problem dimensions
    #[default]
    Auto,
    /// Revised Simplex algorithm with 2-Phase method and Basis Factorization (exact corner BFS)
    Simplex,
    /// Primal-Dual Hybrid Gradient first-order engine (large-scale sparse, GPU acceleration)
    Pdhg,
    /// Interior Point Method (Barrier / Primal-Dual Path-Following)
    Ipm,
}

impl std::str::FromStr for SolverAlgorithm {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(SolverAlgorithm::Auto),
            "simplex" | "revised-simplex" | "revised_simplex" => Ok(SolverAlgorithm::Simplex),
            "pdhg" | "pdlp" => Ok(SolverAlgorithm::Pdhg),
            "ipm" | "interior-point" | "interior_point" => Ok(SolverAlgorithm::Ipm),
            other => Err(format!(
                "Unknown solver algorithm '{other}'. Valid options: auto, simplex, pdhg, ipm"
            )),
        }
    }
}

impl std::fmt::Display for SolverAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverAlgorithm::Auto => write!(f, "auto"),
            SolverAlgorithm::Simplex => write!(f, "simplex"),
            SolverAlgorithm::Pdhg => write!(f, "pdhg"),
            SolverAlgorithm::Ipm => write!(f, "ipm"),
        }
    }
}

/// Termination status outcomes from the PDHG solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PdhgStatus {
    /// Optimization converged to specified tolerances (primal, dual, and gap)
    Converged,
    /// Hit maximum iteration limit before reaching convergence
    IterationLimit,
    /// Hit wall-clock time limit
    TimeLimit,
    /// Numerical anomaly encountered (NaN, Inf, or denormal breakdown)
    NumericalFailure,
    /// Model contains unsupported structures or zero variables
    Unsupported,
}

impl std::fmt::Display for PdhgStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PdhgStatus::Converged => write!(f, "CONVERGED"),
            PdhgStatus::IterationLimit => write!(f, "ITERATION_LIMIT"),
            PdhgStatus::TimeLimit => write!(f, "TIME_LIMIT"),
            PdhgStatus::NumericalFailure => write!(f, "NUMERICAL_FAILURE"),
            PdhgStatus::Unsupported => write!(f, "UNSUPPORTED"),
        }
    }
}

/// Configuration options controlling the PDHG iterative loop.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PdhgOptions {
    /// Relative primal infeasibility tolerance (default: 1e-6)
    pub primal_tol: f64,
    /// Relative dual infeasibility tolerance (default: 1e-6)
    pub dual_tol: f64,
    /// Relative duality gap tolerance (default: 1e-6)
    pub gap_tol: f64,
    /// Maximum allowed iterations (default: 50,000)
    pub max_iterations: usize,
    /// Maximum wall-clock time in seconds (default: 300.0)
    pub time_limit_secs: f64,
    /// Frequency (iterations) of residual checking and convergence tests (default: 5)
    pub check_frequency: usize,
    /// Step size safety damping factor in (0, 1) (default: 0.95)
    pub step_damping: f64,
    /// Enable basic presolve transformations (default: true)
    pub enable_presolve: bool,
    /// Desired compute backend (CPU, GPU, AUTO) (default: Auto)
    pub backend_type: BackendType,
    /// Enable periodic progress logging to stdout (default: true)
    pub verbose: bool,
}

impl Default for PdhgOptions {
    fn default() -> Self {
        Self {
            primal_tol: 1e-6,
            dual_tol: 1e-6,
            gap_tol: 1e-6,
            max_iterations: 50_000,
            time_limit_secs: 300.0,
            check_frequency: 5,
            step_damping: 0.95,
            enable_presolve: true,
            backend_type: BackendType::Auto,
            verbose: true,
        }
    }
}

/// Infeasibility and optimality residuals for candidate primal-dual pairs.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Residuals {
    pub primal_abs: f64,
    pub primal_rel: f64,
    pub dual_abs: f64,
    pub dual_rel: f64,
    pub duality_gap_abs: f64,
    pub duality_gap_rel: f64,
}

/// Residual calculator evaluating KKT conditions against the original LP bounds and constraints.
pub struct ResidualCalculator;

impl ResidualCalculator {
    /// Calculate exact primal, dual, and gap residuals for point (x, y) on general LP:
    /// min c^T x s.t. row_bounds <= A x <= row_bounds, var_bounds <= x <= var_bounds
    pub fn compute(
        c: &[f64],
        var_bounds: &[Bound],
        row_bounds: &[Bound],
        a: &CsrMatrix,
        x: &[f64],
        y: &[f64],
    ) -> Result<Residuals, SolverError> {
        let m = row_bounds.len();
        let n = var_bounds.len();

        // 1. Primal residual: ||Ax - proj(Ax)||_2 and ||x - proj(x)||_2
        let ax = a
            .mul_vec(x)
            .map_err(|e| SolverError::NumericalError(e.to_string()))?;
        let mut s = vec![0.0f64; m];
        let mut diff_p = vec![0.0f64; m];
        for i in 0..m {
            s[i] = row_bounds[i].project(ax[i]);
            diff_p[i] = ax[i] - s[i];
        }

        let mut diff_x_sq = 0.0f64;
        for j in 0..n {
            let px = var_bounds[j].project(x[j]);
            let dx = x[j] - px;
            diff_x_sq += dx * dx;
        }

        let primal_row_abs_sq: f64 = diff_p.iter().map(|&d| d * d).sum();
        let primal_abs = (primal_row_abs_sq + diff_x_sq).sqrt();
        let s_norm = norm_l2(&s);
        let primal_rel = primal_abs / (1.0 + s_norm);

        // 2. Dual residual: ||c + A^T y - z||_2 / (1 + ||c||_2)
        // Note: With our saddle-point Lagrangian L = c^T x + y^T(Ax - s),
        // stationary condition is c + A^T y - z = 0, where z is in subgradient of box bounds.
        let at_y = a
            .mul_transpose_vec(y)
            .map_err(|e| SolverError::NumericalError(e.to_string()))?;
        let mut reduced_cost = vec![0.0f64; n];
        let mut z = vec![0.0f64; n];
        let mut diff_d = vec![0.0f64; n];

        for j in 0..n {
            let r_j = c[j] + at_y[j];
            reduced_cost[j] = r_j;
            let b = &var_bounds[j];
            // Compute subgradient projection for box [lower, upper]
            if (b.upper - b.lower).abs() <= 1e-12 {
                // Fixed variable
                z[j] = r_j;
            } else if b.upper.is_finite() && x[j] >= b.upper - 1e-10 {
                // At upper bound, z_j <= 0 allowed to reduce objective
                z[j] = r_j.min(0.0);
            } else if b.lower.is_finite() && x[j] <= b.lower + 1e-10 {
                // At lower bound, z_j >= 0 allowed
                z[j] = r_j.max(0.0);
            } else {
                // Interior coordinate, must be 0
                z[j] = 0.0;
            }
            diff_d[j] = r_j - z[j];
        }

        let dual_abs = norm_l2(&diff_d);
        let c_norm = norm_l2(c);
        let dual_rel = dual_abs / (1.0 + c_norm);

        // 3. Duality gap
        let primal_obj: f64 = c.iter().zip(x.iter()).map(|(&ci, &xi)| ci * xi).sum();
        let dual_obj: f64 = y
            .iter()
            .zip(s.iter())
            .map(|(&yi, &si)| -yi * si)
            .sum::<f64>()
            + z.iter()
                .zip(x.iter())
                .map(|(&zi, &xi)| zi * xi)
                .sum::<f64>();
        let duality_gap_abs = (primal_obj - dual_obj).abs();
        let duality_gap_rel = duality_gap_abs / (1.0 + primal_obj.abs() + dual_obj.abs());

        Ok(Residuals {
            primal_abs,
            primal_rel,
            dual_abs,
            dual_rel,
            duality_gap_abs,
            duality_gap_rel,
        })
    }
}

/// Termination policy assessing stopping conditions.
pub struct TerminationPolicy;

impl TerminationPolicy {
    pub fn evaluate(
        options: &PdhgOptions,
        residuals: &Residuals,
        iteration: usize,
        elapsed_secs: f64,
    ) -> Option<PdhgStatus> {
        if !residuals.primal_rel.is_finite()
            || !residuals.dual_rel.is_finite()
            || !residuals.duality_gap_rel.is_finite()
        {
            return Some(PdhgStatus::NumericalFailure);
        }

        if residuals.primal_rel <= options.primal_tol
            && residuals.dual_rel <= options.dual_tol
            && residuals.duality_gap_rel <= options.gap_tol
        {
            return Some(PdhgStatus::Converged);
        }

        if elapsed_secs >= options.time_limit_secs {
            return Some(PdhgStatus::TimeLimit);
        }

        if iteration >= options.max_iterations {
            return Some(PdhgStatus::IterationLimit);
        }

        None
    }
}

/// Dynamic state of the PDHG algorithm during execution.
pub struct PdhgState {
    pub x: Vec<f64>,
    pub x_bar: Vec<f64>,
    pub x_avg: Vec<f64>,
    pub y: Vec<f64>,
    pub y_avg: Vec<f64>,
    pub tau: Vec<f64>,
    pub sigma: Vec<f64>,
    pub iteration: usize,
    pub accumulated_weight: f64,
    pub last_restart_err: f64,
}

impl PdhgState {
    pub fn new(
        num_vars: usize,
        num_cons: usize,
        tau: Vec<f64>,
        sigma: Vec<f64>,
        init_x: Option<Vec<f64>>,
    ) -> Self {
        let x = init_x.unwrap_or_else(|| vec![0.0; num_vars]);
        let x_bar = x.clone();
        let x_avg = x.clone();
        let y = vec![0.0; num_cons];
        let y_avg = y.clone();

        Self {
            x,
            x_bar,
            x_avg,
            y,
            y_avg,
            tau,
            sigma,
            iteration: 0,
            accumulated_weight: 0.0,
            last_restart_err: f64::INFINITY,
        }
    }

    /// Reset ergodic iterate accumulation (adaptive restart)
    pub fn restart(&mut self, err: f64) {
        self.x.copy_from_slice(&self.x_avg);
        self.x_bar.copy_from_slice(&self.x_avg);
        self.y.copy_from_slice(&self.y_avg);
        self.accumulated_weight = 0.0;
        self.last_restart_err = err;
    }
}

/// Final result returned by the PDHG solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PdhgResult {
    pub status: PdhgStatus,
    pub primal_solution: Vec<f64>,
    pub dual_solution: Vec<f64>,
    pub objective_value: f64,
    pub iterations: usize,
    pub solve_time_secs: f64,
    pub residuals: Residuals,
    pub backend_name: String,
    pub timing_profile: BackendTimingProfile,
}

/// The core Primal-Dual Hybrid Gradient linear programming solver.
pub struct PdhgSolver {
    pub options: PdhgOptions,
    backend: Option<Arc<dyn ComputeBackend>>,
}

impl PdhgSolver {
    pub fn new(options: PdhgOptions) -> Self {
        Self {
            options,
            backend: None,
        }
    }

    pub fn new_with_backend(options: PdhgOptions, backend: Arc<dyn ComputeBackend>) -> Self {
        Self {
            options,
            backend: Some(backend),
        }
    }

    /// Solve a general LinearProgram using PDHG decoupled via ComputeBackend.
    pub fn solve(&self, lp: &LinearProgram) -> Result<PdhgResult, SolverError> {
        let start_time = Instant::now();

        // Resolve compute backend (either provided or dynamically selected)
        let (backend, info) = match &self.backend {
            Some(b) => (b.clone(), b.device_info()),
            None => {
                let (b, info) = select_backend(self.options.backend_type)?;
                (Arc::from(b), info)
            }
        };

        let n = lp.num_variables();
        let m = lp.num_constraints();

        if n == 0 {
            return Ok(PdhgResult {
                status: PdhgStatus::Unsupported,
                primal_solution: vec![],
                dual_solution: vec![],
                objective_value: 0.0,
                iterations: 0,
                solve_time_secs: start_time.elapsed().as_secs_f64(),
                residuals: Residuals {
                    primal_abs: 0.0,
                    primal_rel: 0.0,
                    dual_abs: 0.0,
                    dual_rel: 0.0,
                    duality_gap_abs: 0.0,
                    duality_gap_rel: 0.0,
                },
                backend_name: info.name,
                timing_profile: backend.timing_profile(),
            });
        }

        // Adjust cost vector for Maximize objective (min (-c)^T x)
        let is_maximize = lp.objective.sense == Sense::Maximize;
        let c: Vec<f64> = if is_maximize {
            lp.objective.c.iter().map(|&ci| -ci).collect()
        } else {
            lp.objective.c.clone()
        };

        let var_bounds = &lp.variable_bounds.bounds;
        let row_bounds = &lp.constraint_bounds.bounds;
        let a = &lp.a;

        // Compute Pock-Chambolle preconditioned diagonal step sizes:
        // sigma_i = gamma / sum_j |A_ij|, tau_j = gamma / sum_i |A_ij|
        let (raw_sigma, raw_tau) = compute_step_sizes(a, self.options.step_damping);

        let mut state = PdhgState::new(n, m, raw_tau, raw_sigma, None);

        // Project initial primal point into variable bounds
        for (j, b) in var_bounds.iter().enumerate().take(n) {
            state.x[j] = b.project(state.x[j]);
            state.x_bar[j] = state.x[j];
            state.x_avg[j] = state.x[j];
        }

        let mut last_residuals =
            ResidualCalculator::compute(&c, var_bounds, row_bounds, a, &state.x_avg, &state.y_avg)?;

        // Initialize Numerical Monitor with Tolerance Policy
        let tol_policy = TolerancePolicy::new(NumericalTolerances {
            primal_tol: self.options.primal_tol,
            dual_tol: self.options.dual_tol,
            gap_tol: self.options.gap_tol,
            max_iterations: self.options.max_iterations,
            time_limit_secs: self.options.time_limit_secs,
            ..Default::default()
        });
        let mut monitor = NumericalMonitor::new(tol_policy, n, m, &state.x_avg, &state.y_avg);
        let initial_err = last_residuals
            .primal_rel
            .max(last_residuals.dual_rel)
            .max(last_residuals.duality_gap_rel);
        monitor.restart_manager.set_baseline(initial_err);

        // Check initial point
        if let Some(status) = TerminationPolicy::evaluate(
            &self.options,
            &last_residuals,
            0,
            start_time.elapsed().as_secs_f64(),
        ) {
            let final_obj =
                compute_final_objective(&c, lp.objective.offset, &state.x_avg, is_maximize);
            if self.options.verbose {
                println!("\n--- Running Engine on: cpu ---");
                println!(
                    "Iter {:<4} | Objective = {:<12.6} | Gap = {:.3e} | |r_b| = {:.3e} | |r_c| = {:.3e}",
                    0,
                    final_obj,
                    last_residuals.duality_gap_rel,
                    last_residuals.primal_rel,
                    last_residuals.dual_rel
                );
            }
            return Ok(PdhgResult {
                status,
                primal_solution: state.x_avg,
                dual_solution: state.y_avg,
                objective_value: final_obj,
                iterations: 0,
                solve_time_secs: start_time.elapsed().as_secs_f64(),
                residuals: last_residuals,
                backend_name: info.name,
                timing_profile: backend.timing_profile(),
            });
        }

        if self.options.verbose {
            println!("\n--- Running Engine on: cpu ---");
            let initial_obj =
                compute_final_objective(&c, lp.objective.offset, &state.x_avg, is_maximize);
            println!(
                "Iter {:<4} | Objective = {:<12.6} | Gap = {:.3e} | |r_b| = {:.3e} | |r_c| = {:.3e}",
                0,
                initial_obj,
                last_residuals.duality_gap_rel,
                last_residuals.primal_rel,
                last_residuals.dual_rel
            );
        }

        let mut final_status = PdhgStatus::IterationLimit;
        let mut ax_bar = vec![0.0f64; m];
        let mut at_y = vec![0.0f64; n];

        while state.iteration < self.options.max_iterations {
            state.iteration += 1;

            // 1. Dual Step (proximal step on y)
            // v = y^k + sigma * A * x_bar^k
            // s = proj_[row_lower, row_upper]( v / sigma ) = proj( A * x_bar^k + y^k / sigma )
            // y^{k+1} = y^k + sigma * (A * x_bar^k - s)
            backend.spmv(1.0, a, &state.x_bar, 0.0, &mut ax_bar)?;

            for i in 0..m {
                let sigma_i = state.sigma[i];
                let v_div_sigma = ax_bar[i] + state.y[i] / sigma_i;
                let s_i = row_bounds[i].project(v_div_sigma);
                state.y[i] += sigma_i * (ax_bar[i] - s_i);
            }

            // 2. Primal Step (gradient descent on x)
            // x^{k+1} = proj_[col_lower, col_upper]( x^k - tau * (c + A^T * y^{k+1}) )
            backend.spmv_transpose(1.0, a, &state.y, 0.0, &mut at_y)?;

            let prev_x = state.x.clone();
            for j in 0..n {
                let tau_j = state.tau[j];
                let grad_x = c[j] + at_y[j];
                let x_next = state.x[j] - tau_j * grad_x;
                state.x[j] = var_bounds[j].project(x_next);
            }

            // 3. Primal Extrapolation
            // x_bar^{k+1} = 2 * x^{k+1} - x^k
            for (j, &px) in prev_x.iter().enumerate().take(n) {
                state.x_bar[j] = 2.0 * state.x[j] - px;
            }

            // 4. Ergodic Iterate Averaging
            let weight = 1.0;
            state.accumulated_weight += weight;
            let inv_total = 1.0 / state.accumulated_weight;
            for j in 0..n {
                state.x_avg[j] += (state.x[j] - state.x_avg[j]) * inv_total;
            }
            for i in 0..m {
                state.y_avg[i] += (state.y[i] - state.y_avg[i]) * inv_total;
            }

            // 5. Periodic Convergence and Health Checks
            if state.iteration.is_multiple_of(self.options.check_frequency) {
                let elapsed = start_time.elapsed().as_secs_f64();
                last_residuals = ResidualCalculator::compute(
                    &c,
                    var_bounds,
                    row_bounds,
                    a,
                    &state.x_avg,
                    &state.y_avg,
                )?;

                let current_obj =
                    compute_final_objective(&c, lp.objective.offset, &state.x_avg, is_maximize);

                if self.options.verbose {
                    println!(
                        "Iter {:<4} | Objective = {:<12.6} | Gap = {:.3e} | |r_b| = {:.3e} | |r_c| = {:.3e}",
                        state.iteration,
                        current_obj,
                        last_residuals.duality_gap_rel,
                        last_residuals.primal_rel,
                        last_residuals.dual_rel
                    );
                }

                let current_err = last_residuals
                    .primal_rel
                    .max(last_residuals.dual_rel)
                    .max(last_residuals.duality_gap_rel);

                // Check termination
                if let Some(status) = TerminationPolicy::evaluate(
                    &self.options,
                    &last_residuals,
                    state.iteration,
                    elapsed,
                ) {
                    final_status = status;
                    break;
                }

                // Run Numerical Monitor health checks (monitoring residuals, obj change, step norms, NaN/Inf, stagnation)
                let verdict = monitor.check_step(
                    state.iteration,
                    &state.x_avg,
                    &state.y_avg,
                    (
                        last_residuals.primal_rel,
                        last_residuals.dual_rel,
                        last_residuals.duality_gap_rel,
                    ),
                    current_obj,
                );

                match verdict {
                    MonitorVerdict::NonFiniteDetected { .. } => {
                        final_status = PdhgStatus::NumericalFailure;
                        break;
                    }
                    MonitorVerdict::StagnationExhausted(_) => {
                        final_status = PdhgStatus::IterationLimit;
                        break;
                    }
                    MonitorVerdict::RecoverToBest {
                        best_x,
                        best_y,
                        best_err,
                        ..
                    } => {
                        // Safe recovery rollback to best-known iterate
                        state.x.copy_from_slice(&best_x);
                        state.x_bar.copy_from_slice(&best_x);
                        state.x_avg.copy_from_slice(&best_x);
                        state.y.copy_from_slice(&best_y);
                        state.y_avg.copy_from_slice(&best_y);
                        state.accumulated_weight = 0.0;
                        state.last_restart_err = best_err;
                    }
                    MonitorVerdict::RestartToAverage { .. } => {
                        state.restart(current_err);
                    }
                    MonitorVerdict::Healthy(_) => {}
                }
            }
        }

        let elapsed = start_time.elapsed().as_secs_f64();
        // Compute final residuals on the returned solution
        last_residuals =
            ResidualCalculator::compute(&c, var_bounds, row_bounds, a, &state.x_avg, &state.y_avg)?;

        // Final check on termination status
        if let Some(status) =
            TerminationPolicy::evaluate(&self.options, &last_residuals, state.iteration, elapsed)
        {
            final_status = status;
        }

        let final_obj = compute_final_objective(&c, lp.objective.offset, &state.x_avg, is_maximize);

        if self.options.verbose && !state.iteration.is_multiple_of(self.options.check_frequency) {
            println!(
                "Iter {:<4} | Objective = {:<12.6} | Gap = {:.3e} | |r_b| = {:.3e} | |r_c| = {:.3e}",
                state.iteration,
                final_obj,
                last_residuals.duality_gap_rel,
                last_residuals.primal_rel,
                last_residuals.dual_rel
            );
        }

        // Invert dual multipliers back to standard LP shadow price convention (lambda = -y)
        let dual_solution: Vec<f64> = state.y_avg.iter().map(|&yi| -yi).collect();

        Ok(PdhgResult {
            status: final_status,
            primal_solution: state.x_avg,
            dual_solution,
            objective_value: final_obj,
            iterations: state.iteration,
            solve_time_secs: elapsed,
            residuals: last_residuals,
            backend_name: info.name,
            timing_profile: backend.timing_profile(),
        })
    }

    /// Convenience solver bridging from LpProblem with integrated presolve and postsolve reconstruction.
    pub fn solve_problem(&self, problem: &LpProblem) -> Result<LpSolution, SolverError> {
        if self.options.enable_presolve {
            let (presolved_problem, trans_map) = presolve(problem, 1e-12)?;
            if presolved_problem.num_variables() == 0 {
                // All variables eliminated (e.g. all variables fixed)
                let mock_sol = LpSolution {
                    status: SolverStatus::Optimal,
                    primal: vec![],
                    dual: vec![0.0; presolved_problem.num_constraints()],
                    objective: presolved_problem.obj_offset,
                    iterations: 0,
                    primal_residual: 0.0,
                    dual_residual: 0.0,
                    solve_time_secs: 0.0,
                };
                return Ok(postsolve(&mock_sol, &trans_map));
            }
            let lp = LinearProgram::from_lp_problem(presolved_problem)?;
            let res = self.solve(&lp)?;

            let solver_status = match res.status {
                PdhgStatus::Converged => SolverStatus::Optimal,
                PdhgStatus::IterationLimit => SolverStatus::IterationLimit,
                PdhgStatus::TimeLimit => SolverStatus::TimeLimit,
                PdhgStatus::NumericalFailure => SolverStatus::NumericalFailure,
                PdhgStatus::Unsupported => SolverStatus::NotSolved,
            };

            let presolved_sol = LpSolution {
                status: solver_status,
                primal: res.primal_solution,
                dual: res.dual_solution,
                objective: res.objective_value,
                iterations: res.iterations,
                primal_residual: res.residuals.primal_rel,
                dual_residual: res.residuals.dual_rel,
                solve_time_secs: res.solve_time_secs,
            };

            let final_sol = postsolve(&presolved_sol, &trans_map);
            Ok(final_sol)
        } else {
            let lp = LinearProgram::from_lp_problem(problem.clone())?;
            let res = self.solve(&lp)?;

            let solver_status = match res.status {
                PdhgStatus::Converged => SolverStatus::Optimal,
                PdhgStatus::IterationLimit => SolverStatus::IterationLimit,
                PdhgStatus::TimeLimit => SolverStatus::TimeLimit,
                PdhgStatus::NumericalFailure => SolverStatus::NumericalFailure,
                PdhgStatus::Unsupported => SolverStatus::NotSolved,
            };

            Ok(LpSolution {
                status: solver_status,
                primal: res.primal_solution,
                dual: res.dual_solution,
                objective: res.objective_value,
                iterations: res.iterations,
                primal_residual: res.residuals.primal_rel,
                dual_residual: res.residuals.dual_rel,
                solve_time_secs: res.solve_time_secs,
            })
        }
    }
}

/// Compute Pock-Chambolle step sizes with user damping factor gamma.
fn compute_step_sizes(a: &CsrMatrix, gamma: f64) -> (Vec<f64>, Vec<f64>) {
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
            *s = gamma / r_sum;
        } else {
            *s = 1.0;
        }
    }

    let mut tau = vec![1.0; n];
    for (t, &c_sum) in tau.iter_mut().zip(col_sums.iter()) {
        if c_sum > 1e-12 {
            *t = gamma / c_sum;
        } else {
            *t = 1.0;
        }
    }

    (sigma, tau)
}

fn compute_final_objective(c_internal: &[f64], offset: f64, x: &[f64], is_maximize: bool) -> f64 {
    let raw_sum: f64 = c_internal
        .iter()
        .zip(x.iter())
        .map(|(&ci, &xi)| ci * xi)
        .sum();
    if is_maximize {
        -raw_sum + offset
    } else {
        raw_sum + offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yukti_model::{ConstraintBounds, Objective, VariableBounds};
    use yukti_sparse::CooMatrix;

    #[test]
    fn test_small_analytical_1d_box_lp() {
        // min 2 * x  s.t. 1.0 <= x <= 4.0 (no row constraints)
        let coo = CooMatrix::new(0, 1);
        let a = CsrMatrix::from_coo(&coo).unwrap();
        let obj = Objective::new(Sense::Minimize, vec![2.0]);
        let var_bounds = VariableBounds::from_bounds(vec![Bound::new(1.0, 4.0)]);
        let con_bounds = ConstraintBounds::from_bounds(vec![]);

        let lp = LinearProgram::new("box_1d".into(), obj, var_bounds, con_bounds, a).unwrap();
        let solver = PdhgSolver::new(PdhgOptions {
            primal_tol: 1e-5,
            dual_tol: 1e-5,
            gap_tol: 1e-5,
            ..Default::default()
        });

        let result = solver.solve(&lp).unwrap();
        assert_eq!(result.status, PdhgStatus::Converged);
        assert!((result.primal_solution[0] - 1.0).abs() < 1e-3);
        assert!((result.objective_value - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_small_analytical_2d_equality_lp() {
        // min  x1 + 2 * x2
        // s.t. x1 + x2 = 3.0
        //      x1 >= 0, x2 >= 0
        // Optimal solution: x1 = 3.0, x2 = 0.0, optimal obj = 3.0
        let mut coo = CooMatrix::new(1, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let obj = Objective::new(Sense::Minimize, vec![1.0, 2.0]);
        let var_bounds =
            VariableBounds::from_bounds(vec![Bound::non_negative(), Bound::non_negative()]);
        let con_bounds = ConstraintBounds::from_bounds(vec![Bound::fixed(3.0)]);

        let lp = LinearProgram::new("eq_2d".into(), obj, var_bounds, con_bounds, a).unwrap();
        let solver = PdhgSolver::new(PdhgOptions {
            primal_tol: 1e-4,
            dual_tol: 1e-4,
            gap_tol: 1e-4,
            max_iterations: 10_000,
            ..Default::default()
        });

        let result = solver.solve(&lp).unwrap();
        assert_eq!(result.status, PdhgStatus::Converged);
        assert!((result.primal_solution[0] - 3.0).abs() < 0.05);
        assert!(result.primal_solution[1].abs() < 0.05);
        assert!((result.objective_value - 3.0).abs() < 0.05);
    }

    #[test]
    fn test_small_analytical_2d_inequality_lp() {
        // min -x1 - 2 * x2
        // s.t. x1 + x2 <= 4.0
        //      0 <= x1 <= 3.0, 0 <= x2 <= 3.0
        // Optimal solution: x2 = 3.0, x1 = 1.0, optimal obj = -7.0
        let mut coo = CooMatrix::new(1, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let obj = Objective::new(Sense::Minimize, vec![-1.0, -2.0]);
        let var_bounds =
            VariableBounds::from_bounds(vec![Bound::new(0.0, 3.0), Bound::new(0.0, 3.0)]);
        let con_bounds = ConstraintBounds::from_bounds(vec![Bound::upper_only(4.0)]);

        let lp = LinearProgram::new("ineq_2d".into(), obj, var_bounds, con_bounds, a).unwrap();
        let solver = PdhgSolver::new(PdhgOptions {
            primal_tol: 1e-4,
            dual_tol: 1e-4,
            gap_tol: 1e-4,
            max_iterations: 10_000,
            ..Default::default()
        });

        let result = solver.solve(&lp).unwrap();
        assert_eq!(result.status, PdhgStatus::Converged);
        assert!((result.primal_solution[0] - 1.0).abs() < 0.05);
        assert!((result.primal_solution[1] - 3.0).abs() < 0.05);
        assert!((result.objective_value - (-7.0)).abs() < 0.05);
    }

    #[test]
    fn test_iteration_limit_enforcement() {
        let mut coo = CooMatrix::new(1, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let obj = Objective::new(Sense::Minimize, vec![1.0, 2.0]);
        let var_bounds =
            VariableBounds::from_bounds(vec![Bound::non_negative(), Bound::non_negative()]);
        let con_bounds = ConstraintBounds::from_bounds(vec![Bound::fixed(3.0)]);

        let lp = LinearProgram::new("iter_limit".into(), obj, var_bounds, con_bounds, a).unwrap();
        // Set iteration limit artificially to 2
        let solver = PdhgSolver::new(PdhgOptions {
            max_iterations: 2,
            check_frequency: 1,
            primal_tol: 1e-12, // unreachable in 2 iters
            ..Default::default()
        });

        let result = solver.solve(&lp).unwrap();
        assert_eq!(result.status, PdhgStatus::IterationLimit);
        assert_eq!(result.iterations, 2);
    }

    #[test]
    fn test_small_analytical_maximization_lp() {
        // max 3 * x1 + 2 * x2
        // s.t. x1 + x2 <= 4.0
        //      x1 <= 2.0
        //      x1 >= 0, x2 >= 0
        // Optimal solution: x1 = 2.0, x2 = 2.0, optimal objective = 10.0
        let mut coo = CooMatrix::new(2, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 1.0).unwrap();
        coo.add_entry(1, 0, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let obj = Objective::new(Sense::Maximize, vec![3.0, 2.0]);
        let var_bounds =
            VariableBounds::from_bounds(vec![Bound::non_negative(), Bound::non_negative()]);
        let con_bounds =
            ConstraintBounds::from_bounds(vec![Bound::upper_only(4.0), Bound::upper_only(2.0)]);

        let lp = LinearProgram::new("max_2d".into(), obj, var_bounds, con_bounds, a).unwrap();
        let solver = PdhgSolver::new(PdhgOptions {
            primal_tol: 1e-4,
            dual_tol: 1e-4,
            gap_tol: 1e-4,
            max_iterations: 10_000,
            ..Default::default()
        });

        let result = solver.solve(&lp).unwrap();
        assert_eq!(result.status, PdhgStatus::Converged);
        assert!((result.primal_solution[0] - 2.0).abs() < 0.05);
        assert!((result.primal_solution[1] - 2.0).abs() < 0.05);
        assert!((result.objective_value - 10.0).abs() < 0.05);
    }

    #[test]
    fn test_small_analytical_range_constraint_lp() {
        // min 2 * x1 + x2
        // s.t. 3.0 <= x1 + x2 <= 6.0
        //      0 <= x1 <= 10.0, 0 <= x2 <= 10.0
        // Optimal solution: x1 = 0.0, x2 = 3.0, optimal objective = 3.0
        let mut coo = CooMatrix::new(1, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let obj = Objective::new(Sense::Minimize, vec![2.0, 1.0]);
        let var_bounds =
            VariableBounds::from_bounds(vec![Bound::new(0.0, 10.0), Bound::new(0.0, 10.0)]);
        let con_bounds = ConstraintBounds::from_bounds(vec![Bound::new(3.0, 6.0)]);

        let lp = LinearProgram::new("range_2d".into(), obj, var_bounds, con_bounds, a).unwrap();
        let solver = PdhgSolver::new(PdhgOptions {
            primal_tol: 1e-4,
            dual_tol: 1e-4,
            gap_tol: 1e-4,
            max_iterations: 10_000,
            ..Default::default()
        });

        let result = solver.solve(&lp).unwrap();
        assert_eq!(result.status, PdhgStatus::Converged);
        assert!(result.primal_solution[0].abs() < 0.05);
        assert!((result.primal_solution[1] - 3.0).abs() < 0.05);
        assert!((result.objective_value - 3.0).abs() < 0.05);
    }

    #[test]
    fn test_time_limit_enforcement() {
        let mut coo = CooMatrix::new(1, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let obj = Objective::new(Sense::Minimize, vec![1.0, 2.0]);
        let var_bounds =
            VariableBounds::from_bounds(vec![Bound::non_negative(), Bound::non_negative()]);
        let con_bounds = ConstraintBounds::from_bounds(vec![Bound::fixed(3.0)]);

        let lp = LinearProgram::new("time_limit".into(), obj, var_bounds, con_bounds, a).unwrap();
        // Artificially zero time limit and strict tolerance
        let solver = PdhgSolver::new(PdhgOptions {
            time_limit_secs: 0.0,
            check_frequency: 1,
            primal_tol: 1e-12,
            ..Default::default()
        });

        let result = solver.solve(&lp).unwrap();
        assert_eq!(result.status, PdhgStatus::TimeLimit);
        assert_ne!(result.status, PdhgStatus::Converged);
    }

    #[test]
    fn test_numerical_failure_detection() {
        let nan_res = Residuals {
            primal_abs: f64::NAN,
            primal_rel: f64::NAN,
            dual_abs: 0.0,
            dual_rel: 0.0,
            duality_gap_abs: 0.0,
            duality_gap_rel: 0.0,
        };
        let status = TerminationPolicy::evaluate(&PdhgOptions::default(), &nan_res, 10, 0.1);
        assert_eq!(status, Some(PdhgStatus::NumericalFailure));
    }

    #[test]
    fn test_unsupported_model_zero_variables() {
        let empty_lp = LinearProgram {
            name: "unsupported".into(),
            objective: Objective::new(Sense::Minimize, vec![]),
            variable_bounds: VariableBounds::from_bounds(vec![]),
            constraint_bounds: ConstraintBounds::from_bounds(vec![]),
            a: CsrMatrix::from_coo(&CooMatrix::new(0, 0)).unwrap(),
        };
        let solver = PdhgSolver::new(PdhgOptions::default());
        let result = solver.solve(&empty_lp).unwrap();
        assert_eq!(result.status, PdhgStatus::Unsupported);
        assert_ne!(result.status, PdhgStatus::Converged);
    }

    #[test]
    fn test_end_to_end_presolve_pdhg_solve_postsolve() {
        // Model with:
        // x1 fixed to 2.0 (5 * x1)
        // x2 (1 * x2)
        // x3 (2 * x3)
        // Row 0: empty row: 0 in [-1.0, 5.0]
        // Row 1: x1 + x2 + x3 = 6.0
        // Row 2: singleton: 2 * x2 <= 6.0 (implies x2 <= 3.0)
        // Bounds: x1 in [2, 2], x2 >= 0, x3 in [1, 10]
        // Analytical optimal solution:
        // x1 = 2.0 (fixed)
        // x2 = 3.0
        // x3 = 1.0
        // Objective = 5(2) + 1(3) + 2(1) = 15.0
        let mut coo = CooMatrix::new(3, 3);
        // Row 0 is empty (no entries added)
        // Row 1: 1*x1 + 1*x2 + 1*x3
        coo.add_entry(1, 0, 1.0).unwrap();
        coo.add_entry(1, 1, 1.0).unwrap();
        coo.add_entry(1, 2, 1.0).unwrap();
        // Row 2: 2*x2
        coo.add_entry(2, 1, 2.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let problem = LpProblem {
            name: "e2e_presolve".into(),
            sense: Sense::Minimize,
            obj_offset: 0.0,
            c: vec![5.0, 1.0, 2.0],
            var_names: vec!["x1".into(), "x2".into(), "x3".into()],
            var_bounds: vec![
                Bound::fixed(2.0),
                Bound::non_negative(),
                Bound::new(1.0, 10.0),
            ],
            row_names: vec!["r_empty".into(), "r_eq".into(), "r_singleton".into()],
            row_bounds: vec![
                Bound::new(-1.0, 5.0),
                Bound::fixed(6.0),
                Bound::upper_only(6.0),
            ],
            a,
        };

        let solver = PdhgSolver::new(PdhgOptions {
            primal_tol: 1e-4,
            dual_tol: 1e-4,
            gap_tol: 1e-4,
            max_iterations: 15_000,
            enable_presolve: true,
            ..Default::default()
        });

        let solution = solver.solve_problem(&problem).unwrap();
        assert_eq!(solution.status, SolverStatus::Optimal);
        assert_eq!(solution.primal.len(), 3);
        assert!((solution.primal[0] - 2.0).abs() < 0.05); // fixed x1 restored
        assert!((solution.primal[1] - 3.0).abs() < 0.05); // x2
        assert!((solution.primal[2] - 1.0).abs() < 0.05); // x3
        assert!((solution.objective - 15.0).abs() < 0.05);
    }

    #[test]
    fn test_backend_options_cpu_succeeds() {
        let mut coo = CooMatrix::new(1, 1);
        coo.add_entry(0, 0, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let lp = LinearProgram::new(
            "test_cpu".into(),
            Objective::new(Sense::Minimize, vec![2.0]),
            VariableBounds::from_bounds(vec![Bound::new(1.0, 5.0)]),
            ConstraintBounds::from_bounds(vec![Bound::non_negative()]),
            a,
        )
        .unwrap();

        let solver = PdhgSolver::new(PdhgOptions {
            backend_type: BackendType::Cpu,
            ..Default::default()
        });

        let res = solver.solve(&lp).unwrap();
        assert_eq!(res.status, PdhgStatus::Converged);
        assert_eq!(res.backend_name, "CPU (Rust Native)");
    }

    #[test]
    fn test_backend_options_auto_succeeds() {
        let mut coo = CooMatrix::new(1, 1);
        coo.add_entry(0, 0, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let lp = LinearProgram::new(
            "test_auto".into(),
            Objective::new(Sense::Minimize, vec![2.0]),
            VariableBounds::from_bounds(vec![Bound::new(1.0, 5.0)]),
            ConstraintBounds::from_bounds(vec![Bound::non_negative()]),
            a,
        )
        .unwrap();

        let solver = PdhgSolver::new(PdhgOptions {
            backend_type: BackendType::Auto,
            ..Default::default()
        });

        let res = solver.solve(&lp).unwrap();
        assert_eq!(res.status, PdhgStatus::Converged);
        assert!(res.backend_name.contains("CPU"));
    }

    #[test]
    fn test_backend_options_gpu_fails_gracefully() {
        let mut coo = CooMatrix::new(1, 1);
        coo.add_entry(0, 0, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let lp = LinearProgram::new(
            "test_gpu_fail".into(),
            Objective::new(Sense::Minimize, vec![2.0]),
            VariableBounds::from_bounds(vec![Bound::new(1.0, 5.0)]),
            ConstraintBounds::from_bounds(vec![Bound::non_negative()]),
            a,
        )
        .unwrap();

        let solver = PdhgSolver::new(PdhgOptions {
            backend_type: BackendType::Gpu,
            ..Default::default()
        });

        let res = solver.solve(&lp);
        assert!(res.is_err());
        if let Err(SolverError::BackendError(msg)) = res {
            assert!(msg.contains("ERR_GPU_UNAVAILABLE"));
        } else {
            panic!("Expected BackendError");
        }
    }

    #[test]
    fn test_solver_with_explicit_backend() {
        let mut coo = CooMatrix::new(1, 1);
        coo.add_entry(0, 0, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let lp = LinearProgram::new(
            "test_custom".into(),
            Objective::new(Sense::Minimize, vec![3.0]),
            VariableBounds::from_bounds(vec![Bound::new(2.0, 4.0)]),
            ConstraintBounds::from_bounds(vec![Bound::non_negative()]),
            a,
        )
        .unwrap();

        let custom_backend = Arc::new(yukti_gpu::CpuRustBackend::new());
        let solver = PdhgSolver::new_with_backend(PdhgOptions::default(), custom_backend);

        let res = solver.solve(&lp).unwrap();
        assert_eq!(res.status, PdhgStatus::Converged);
        assert_eq!(res.backend_name, "CPU (Rust Native)");
        assert!((res.primal_solution[0] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_solve_problem_all_fixed_variables_presolve() {
        // Model with 2 variables, both fixed
        // x1 = 3.0, x2 = 5.0
        // min 2*x1 + 4*x2 = 6 + 20 = 26
        // s.t. x1 + x2 <= 10.0 (3 + 5 = 8 <= 10)
        let mut coo = CooMatrix::new(1, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let problem = LpProblem {
            name: "all_fixed".into(),
            sense: Sense::Minimize,
            obj_offset: 0.0,
            c: vec![2.0, 4.0],
            var_names: vec!["x1".into(), "x2".into()],
            var_bounds: vec![Bound::fixed(3.0), Bound::fixed(5.0)],
            row_names: vec!["r1".into()],
            row_bounds: vec![Bound::upper_only(10.0)],
            a,
        };

        let solver = PdhgSolver::new(PdhgOptions {
            enable_presolve: true,
            ..Default::default()
        });

        let solution = solver.solve_problem(&problem).unwrap();
        assert_eq!(solution.status, SolverStatus::Optimal);
        assert_eq!(solution.primal, vec![3.0, 5.0]);
        assert_eq!(solution.objective, 26.0);
    }

    #[test]
    fn test_residual_calculator_catches_box_bound_violation() {
        // Model: min x1, s.t. 0 <= x1 (no row constraints)
        let coo = CooMatrix::new(0, 1);
        let a = CsrMatrix::from_coo(&coo).unwrap();
        let c = vec![1.0];
        let var_bounds = vec![Bound::new(0.0, 10.0)];
        let row_bounds = vec![];

        // x = [-5.0] violates lower bound by 5.0
        let res =
            ResidualCalculator::compute(&c, &var_bounds, &row_bounds, &a, &[-5.0], &[]).unwrap();
        assert!(res.primal_abs >= 5.0);
        assert!(res.primal_rel > 0.0);
    }
}
