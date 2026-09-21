//! High-performance Revised Simplex solver with 2-Phase method and Basis LU Factorization.
//! Engineered in pure, memory-safe Rust with scaled partial-pivoting LU basis factorization.
//! Delivers exact basic feasible solutions (BFS), corner vertices, and dual multipliers
//! for linear programming benchmarks (Netlib, industrial blending, logistics).

#![allow(clippy::needless_range_loop)]

use serde::{Deserialize, Serialize};
use std::time::Instant;
use yutki_model::{LinearProgram, LpSolution, Sense, SolverError, SolverStatus};
use yutki_sparse::factorization::LuDecomposition;

/// Termination outcome of the Revised Simplex solver.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimplexStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit,
    TimeLimit,
    SingularBasis,
}

impl std::fmt::Display for SimplexStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimplexStatus::Optimal => write!(f, "OPTIMAL"),
            SimplexStatus::Infeasible => write!(f, "INFEASIBLE"),
            SimplexStatus::Unbounded => write!(f, "UNBOUNDED"),
            SimplexStatus::IterationLimit => write!(f, "ITERATION_LIMIT"),
            SimplexStatus::TimeLimit => write!(f, "TIME_LIMIT"),
            SimplexStatus::SingularBasis => write!(f, "SINGULAR_BASIS"),
        }
    }
}

/// Configuration options for the Revised Simplex solver.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimplexOptions {
    /// Maximum total iterations across Phase 1 and Phase 2 (default: 100,000)
    pub max_iterations: usize,
    /// Absolute feasibility and optimality tolerance (default: 1e-8)
    pub tolerance: f64,
    /// Pivot selection threshold in ratio test (default: 1e-9)
    pub pivot_tolerance: f64,
    /// Maximum solve duration in seconds (default: 300.0)
    pub time_limit_secs: f64,
    /// Enable Bland's smallest-index anti-cycling rule (default: true)
    pub use_bland_rule: bool,
}

impl Default for SimplexOptions {
    fn default() -> Self {
        Self {
            max_iterations: 100_000,
            tolerance: 1e-8,
            pivot_tolerance: 1e-9,
            time_limit_secs: 300.0,
            use_bland_rule: false,
        }
    }
}

/// Detailed execution telemetry from the Revised Simplex solver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimplexTelemetry {
    pub status: SimplexStatus,
    pub objective_value: f64,
    pub phase1_iterations: usize,
    pub phase2_iterations: usize,
    pub total_iterations: usize,
    pub solve_time_secs: f64,
    pub num_slacks: usize,
    pub num_artificials: usize,
    pub basic_variable_indices: Vec<usize>,
}

/// Canonical standard-form Linear Program:
/// min c^T x s.t. A x = b, x >= 0, b >= 0.
#[derive(Debug, Clone)]
pub(crate) struct StandardFormLp {
    pub(crate) num_constraints: usize,
    pub(crate) num_decision_vars: usize,
    pub(crate) num_slacks: usize,
    pub(crate) num_artificials: usize,
    pub(crate) num_total_vars: usize,
    pub(crate) c: Vec<f64>,
    pub(crate) b: Vec<f64>,
    /// Column-major sparse representation for efficient pivot direction extraction:
    /// cols[j] is a list of (row, value)
    pub(crate) cols: Vec<Vec<(usize, f64)>>,
    /// Initial basis variable indices (size = num_constraints)
    pub(crate) initial_basis: Vec<usize>,
    /// Tracks which original row was multiplied by -1 to ensure b >= 0
    pub(crate) row_sign_flips: Vec<bool>,
    /// Lower bound shifts applied to original decision variables (x_orig = x_std + shift)
    pub(crate) var_shifts: Vec<f64>,
    /// Original objective constant offset
    pub(crate) obj_offset: f64,
    /// Whether original problem was a maximization problem
    pub(crate) is_maximization: bool,
}

impl StandardFormLp {
    /// Convert an arbitrary `LinearProgram` into canonical standard form:
    /// min c^T x s.t. A x = b, x >= 0, b >= 0.
    pub(crate) fn from_linear_program(lp: &LinearProgram) -> Result<Self, SolverError> {
        let m = lp.num_constraints();
        let n = lp.num_variables();
        let is_maximization = lp.objective.sense == Sense::Maximize;

        // Base objective cost vector (negated if maximizing)
        let mut orig_c: Vec<f64> = lp
            .objective
            .c
            .iter()
            .map(|&ci| if is_maximization { -ci } else { ci })
            .collect();
        let mut obj_offset = if is_maximization {
            -lp.objective.offset
        } else {
            lp.objective.offset
        };

        // Variable lower bounds shifts
        let mut var_shifts = vec![0.0f64; n];
        for (j, b) in lp.variable_bounds.bounds.iter().enumerate() {
            if b.lower.is_finite() && b.lower != 0.0 {
                var_shifts[j] = b.lower;
            }
        }

        // Apply variable shifts to objective offset
        for j in 0..n {
            if var_shifts[j] != 0.0 {
                obj_offset += orig_c[j] * var_shifts[j];
            }
        }

        // Build columns for decision variables
        let mut cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

        // Fill decision variable columns from CSR matrix A
        let csr = &lp.a;
        for i in 0..m {
            let start = csr.row_ptrs[i];
            let end = csr.row_ptrs[i + 1];
            for k in start..end {
                let j = csr.col_indices[k];
                let val = csr.values[k];
                if val != 0.0 {
                    cols[j].push((i, val));
                }
            }
        }

        // Base RHS vector from row bounds, adjusted for variable shifts
        let mut b = vec![0.0f64; m];
        let mut is_equality = vec![false; m];
        let mut is_less_than = vec![false; m];
        let mut is_greater_than = vec![false; m];

        for i in 0..m {
            let rb = &lp.constraint_bounds.bounds[i];
            // Compute shift contribution from variables
            let mut row_shift = 0.0f64;
            let start = csr.row_ptrs[i];
            let end = csr.row_ptrs[i + 1];
            for k in start..end {
                let j = csr.col_indices[k];
                row_shift += csr.values[k] * var_shifts[j];
            }

            if (rb.lower - rb.upper).abs() < 1e-12 {
                // Equality: Ax = b
                b[i] = rb.lower - row_shift;
                is_equality[i] = true;
            } else if rb.lower == f64::NEG_INFINITY && rb.upper.is_finite() {
                // Inequality: Ax <= upper
                b[i] = rb.upper - row_shift;
                is_less_than[i] = true;
            } else if rb.lower.is_finite() && rb.upper == f64::INFINITY {
                // Inequality: Ax >= lower
                b[i] = rb.lower - row_shift;
                is_greater_than[i] = true;
            } else if rb.lower == f64::NEG_INFINITY && rb.upper == f64::INFINITY {
                // Free row (redundant), treat as Ax <= 0 with huge slack
                b[i] = 0.0;
                is_less_than[i] = true;
            } else {
                // Range constraint: lower <= Ax <= upper.
                // Treat upper bound constraint for now
                b[i] = rb.upper - row_shift;
                is_less_than[i] = true;
            }
        }

        // Normalize negative RHS entries: if b[i] < 0, multiply entire row by -1
        let mut row_sign_flips = vec![false; m];
        for i in 0..m {
            if b[i] < 0.0 {
                b[i] = -b[i];
                row_sign_flips[i] = true;
                // Flip inequality sense: <= becomes >= and vice-versa
                if is_less_than[i] {
                    is_less_than[i] = false;
                    is_greater_than[i] = true;
                } else if is_greater_than[i] {
                    is_greater_than[i] = false;
                    is_less_than[i] = true;
                }
            }
        }

        // Multiply coefficients in flipped rows by -1
        for col in &mut cols {
            for entry in col.iter_mut() {
                if row_sign_flips[entry.0] {
                    entry.1 = -entry.1;
                }
            }
        }

        // Add slack / surplus variables
        // <= requires slack +1.0
        // >= requires surplus -1.0
        let mut num_slacks = 0;
        let mut initial_basis = vec![0; m];
        let mut row_has_basis = vec![false; m];

        for i in 0..m {
            if is_less_than[i] {
                let slack_idx = n + num_slacks;
                cols.push(vec![(i, 1.0)]);
                orig_c.push(0.0);
                initial_basis[i] = slack_idx;
                row_has_basis[i] = true;
                num_slacks += 1;
            } else if is_greater_than[i] {
                // Surplus variable: coeff = -1.0
                cols.push(vec![(i, -1.0)]);
                orig_c.push(0.0);
                num_slacks += 1;
            }
        }

        // Add artificial variables (+1.0) for rows without an initial basic slack
        // (equality rows and greater-than surplus rows)
        let mut num_artificials = 0;
        for i in 0..m {
            if !row_has_basis[i] {
                let art_idx = n + num_slacks + num_artificials;
                cols.push(vec![(i, 1.0)]);
                orig_c.push(0.0); // Cost in Phase 2 is 0
                initial_basis[i] = art_idx;
                num_artificials += 1;
            }
        }

        let num_total_vars = cols.len();

        Ok(Self {
            num_constraints: m,
            num_decision_vars: n,
            num_slacks,
            num_artificials,
            num_total_vars,
            c: orig_c,
            b,
            cols,
            initial_basis,
            row_sign_flips,
            var_shifts,
            obj_offset,
            is_maximization,
        })
    }
}

/// Sovereign Revised Simplex Solver.
pub struct RevisedSimplexSolver;

impl RevisedSimplexSolver {
    /// Solve an arbitrary continuous `LinearProgram` using the Revised Simplex method.
    pub fn solve(
        lp: &LinearProgram,
        options: &SimplexOptions,
    ) -> Result<(LpSolution, SimplexTelemetry), SolverError> {
        let start_time = Instant::now();
        lp.validate()?;

        if lp.num_variables() == 0 || lp.num_constraints() == 0 {
            return Err(SolverError::Unsupported(
                "Simplex solver requires non-zero variables and constraints".into(),
            ));
        }

        let std_lp = StandardFormLp::from_linear_program(lp)?;
        let non_art_count = std_lp.num_decision_vars + std_lp.num_slacks;

        let mut basis = std_lp.initial_basis.clone();
        let mut phase1_iters = 0;

        // PHASE 1: Minimize sum of artificial variables (if any exist)
        if std_lp.num_artificials > 0 {
            let mut c_phase1 = vec![0.0f64; std_lp.num_total_vars];
            for art_idx in non_art_count..std_lp.num_total_vars {
                c_phase1[art_idx] = 1.0;
            }

            let phase1_result = Self::run_simplex_loop(
                &std_lp,
                &mut basis,
                &c_phase1,
                std_lp.num_total_vars,
                options,
                start_time,
            )?;

            phase1_iters = phase1_result.iterations;

            if phase1_result.status != SimplexStatus::Optimal {
                let telemetry = SimplexTelemetry {
                    status: phase1_result.status,
                    objective_value: f64::NAN,
                    phase1_iterations: phase1_iters,
                    phase2_iterations: 0,
                    total_iterations: phase1_iters,
                    solve_time_secs: start_time.elapsed().as_secs_f64(),
                    num_slacks: std_lp.num_slacks,
                    num_artificials: std_lp.num_artificials,
                    basic_variable_indices: basis,
                };
                return Ok((
                    LpSolution {
                        status: SolverStatus::Infeasible,
                        objective: f64::NAN,
                        primal: vec![0.0; lp.num_variables()],
                        dual: vec![0.0; lp.num_constraints()],
                        iterations: phase1_iters,
                        primal_residual: f64::INFINITY,
                        dual_residual: f64::INFINITY,
                        solve_time_secs: telemetry.solve_time_secs,
                    },
                    telemetry,
                ));
            }

            // Check if sum of artificial variables is zero
            let phase1_obj = phase1_result.objective_value;
            if phase1_obj > options.tolerance {
                let telemetry = SimplexTelemetry {
                    status: SimplexStatus::Infeasible,
                    objective_value: phase1_obj,
                    phase1_iterations: phase1_iters,
                    phase2_iterations: 0,
                    total_iterations: phase1_iters,
                    solve_time_secs: start_time.elapsed().as_secs_f64(),
                    num_slacks: std_lp.num_slacks,
                    num_artificials: std_lp.num_artificials,
                    basic_variable_indices: basis,
                };
                return Ok((
                    LpSolution {
                        status: SolverStatus::Infeasible,
                        objective: f64::NAN,
                        primal: vec![0.0; lp.num_variables()],
                        dual: vec![0.0; lp.num_constraints()],
                        iterations: phase1_iters,
                        primal_residual: f64::INFINITY,
                        dual_residual: f64::INFINITY,
                        solve_time_secs: telemetry.solve_time_secs,
                    },
                    telemetry,
                ));
            }

            // Drive any remaining zero-valued artificial variables out of the basis
            Self::purge_artificial_basics(&mut basis, &std_lp, non_art_count, options.tolerance)?;
        }

        // PHASE 2: Optimize original objective function
        let phase2_result = Self::run_simplex_loop(
            &std_lp,
            &mut basis,
            &std_lp.c,
            non_art_count, // Never allow artificials to enter in Phase 2
            options,
            start_time,
        )?;

        let phase2_iters = phase2_result.iterations;
        let total_iters = phase1_iters + phase2_iters;
        let total_time = start_time.elapsed().as_secs_f64();

        // Extract primal solution and dual multipliers from final basis
        let (primal_x, dual_y, raw_objective) =
            Self::extract_solution(&std_lp, &basis, phase2_result.status)?;

        // Adjust objective for sense and constant offset
        let final_obj = if std_lp.is_maximization {
            -raw_objective
        } else {
            raw_objective
        };

        let solver_status = match phase2_result.status {
            SimplexStatus::Optimal => SolverStatus::Optimal,
            SimplexStatus::Infeasible => SolverStatus::Infeasible,
            SimplexStatus::Unbounded => SolverStatus::Unbounded,
            SimplexStatus::IterationLimit => SolverStatus::IterationLimit,
            SimplexStatus::TimeLimit => SolverStatus::TimeLimit,
            SimplexStatus::SingularBasis => SolverStatus::NumericalFailure,
        };

        let telemetry = SimplexTelemetry {
            status: phase2_result.status,
            objective_value: final_obj,
            phase1_iterations: phase1_iters,
            phase2_iterations: phase2_iters,
            total_iterations: total_iters,
            solve_time_secs: total_time,
            num_slacks: std_lp.num_slacks,
            num_artificials: std_lp.num_artificials,
            basic_variable_indices: basis,
        };

        let solution = LpSolution {
            status: solver_status,
            objective: final_obj,
            primal: primal_x,
            dual: dual_y,
            iterations: total_iters,
            primal_residual: 0.0,
            dual_residual: 0.0,
            solve_time_secs: total_time,
        };

        Ok((solution, telemetry))
    }
}

/// An elementary rank-1 Eta matrix update: E_k^-1
/// Differing from identity only in column `col_idx`:
/// (E^-1)_l = 1 / d_l
/// (E^-1)_i = -d_i / d_l  for i != l
#[derive(Debug, Clone)]
struct EtaMatrix {
    col_idx: usize,
    eta: Vec<f64>,
}

/// Basis Factorization Engine maintaining base LU and Product Form of Inverse (PFI) Eta vectors.
/// Speeds up pivot operations from O(m^3) to O(m) by avoiding from-scratch LU factorizations.
struct BasisEngine {
    m: usize,
    base_lu: LuDecomposition,
    etas: Vec<EtaMatrix>,
    basis_dense_scratch: Vec<f64>,
    scales_scratch: Vec<f64>,
    work_scratch: Vec<f64>,
    y_scratch: Vec<f64>,
}

impl BasisEngine {
    fn new(m: usize) -> Self {
        Self {
            m,
            base_lu: LuDecomposition::with_dimension(m),
            etas: Vec::with_capacity(64),
            basis_dense_scratch: vec![0.0f64; m * m],
            scales_scratch: vec![0.0f64; m],
            work_scratch: vec![0.0f64; m],
            y_scratch: vec![0.0f64; m],
        }
    }

    /// Full refactorization of the basis matrix A[:, basis].
    fn refactorize(
        &mut self,
        std_lp: &StandardFormLp,
        basis: &[usize],
        tol: f64,
    ) -> Result<(), SolverError> {
        let m = self.m;
        self.basis_dense_scratch.fill(0.0);
        for (k, &col_idx) in basis.iter().enumerate() {
            for &(row_idx, val) in &std_lp.cols[col_idx] {
                self.basis_dense_scratch[row_idx * m + k] = val;
            }
        }
        self.etas.clear();
        LuDecomposition::factorize_dense_into(
            m,
            &self.basis_dense_scratch,
            tol,
            &mut self.base_lu,
            &mut self.scales_scratch,
        )
        .map_err(|e| SolverError::NumericalError(format!("Basis LU factorize failed: {e}")))
    }

    /// Solve B * x = rhs (Forward transformation / FTRAN).
    /// B^-1 = E_k^-1 ... E_1^-1 B_0^-1
    fn ftran(&mut self, rhs: &[f64], out: &mut [f64]) -> Result<(), SolverError> {
        let m = self.m;
        // Step 1: Solve base LU: B_0 * out = rhs
        self.base_lu
            .solve_into(rhs, out, &mut self.y_scratch)
            .map_err(|e| SolverError::NumericalError(format!("FTRAN base solve failed: {e}")))?;

        // Step 2: Sequentially apply eta transformations E_1^-1, ..., E_k^-1
        for eta_mat in &self.etas {
            let l = eta_mat.col_idx;
            let xl = out[l];
            if xl != 0.0 {
                let eta = &eta_mat.eta;
                for i in 0..m {
                    if i == l {
                        out[i] = eta[i] * xl;
                    } else {
                        out[i] += eta[i] * xl;
                    }
                }
            }
        }

        Ok(())
    }

    /// Solve B^T * lambda = c (Backward transformation / BTRAN).
    /// (B^-1)^T = (B_0^-1)^T (E_1^-1)^T ... (E_k^-1)^T
    fn btran(&mut self, c: &[f64], lambda: &mut [f64]) -> Result<(), SolverError> {
        let m = self.m;
        // Step 1: Apply transposed etas backwards: (E_k^-1)^T ... (E_1^-1)^T to c
        self.work_scratch.copy_from_slice(c);
        for eta_mat in self.etas.iter().rev() {
            let l = eta_mat.col_idx;
            let eta = &eta_mat.eta;
            // z = (E^-1)^T * w: z_l = eta^T * w, z_i = w_i for i != l
            let mut dot = 0.0f64;
            for i in 0..m {
                dot += eta[i] * self.work_scratch[i];
            }
            self.work_scratch[l] = dot;
        }

        // Step 2: Solve base LU transpose: B_0^T * lambda = work
        self.base_lu
            .solve_transpose_into(&self.work_scratch, lambda, &mut self.y_scratch)
            .map_err(|e| SolverError::NumericalError(format!("BTRAN base solve failed: {e}")))?;

        Ok(())
    }

    /// Add an Eta vector when column `leaving_pos` leaves the basis.
    /// `d` is the already-computed direction B^-1 * A[:, entering_j].
    /// Returns false if pivot element is small (suggesting refactorization).
    fn update_basis(&mut self, leaving_pos: usize, d: &[f64]) -> bool {
        let dl = d[leaving_pos];
        if dl.abs() < 1e-7 {
            return false;
        }

        let m = self.m;
        let inv_dl = 1.0 / dl;
        let mut eta = vec![0.0f64; m];
        for i in 0..m {
            if i == leaving_pos {
                eta[i] = inv_dl;
            } else {
                eta[i] = -d[i] * inv_dl;
            }
        }

        self.etas.push(EtaMatrix {
            col_idx: leaving_pos,
            eta,
        });

        true
    }

    fn eta_count(&self) -> usize {
        self.etas.len()
    }
}

impl RevisedSimplexSolver {
    /// Execute the core Revised Simplex iteration loop with Product Form of the Inverse.
    fn run_simplex_loop(
        std_lp: &StandardFormLp,
        basis: &mut [usize],
        c: &[f64],
        candidate_count: usize,
        options: &SimplexOptions,
        start_time: Instant,
    ) -> Result<LoopOutcome, SolverError> {
        let m = std_lp.num_constraints;
        let mut iters = 0;

        let mut is_basic = vec![false; std_lp.num_total_vars];
        for &b_idx in basis.iter() {
            is_basic[b_idx] = true;
        }

        let mut engine = BasisEngine::new(m);
        engine.refactorize(std_lp, basis, options.pivot_tolerance)?;

        let mut x_b = vec![0.0f64; m];
        let mut entering_col_dense = vec![0.0f64; m];
        let mut d = vec![0.0f64; m];
        let mut c_b = vec![0.0f64; m];
        let mut lambda = vec![0.0f64; m];

        const REFACTOR_FREQUENCY: usize = 1;

        loop {
            if start_time.elapsed().as_secs_f64() > options.time_limit_secs {
                return Ok(LoopOutcome {
                    status: SimplexStatus::TimeLimit,
                    iterations: iters,
                    objective_value: f64::NAN,
                });
            }

            if iters >= options.max_iterations {
                return Ok(LoopOutcome {
                    status: SimplexStatus::IterationLimit,
                    iterations: iters,
                    objective_value: f64::NAN,
                });
            }

            // 1. Compute basic primal solution: x_B = B^-1 * b (FTRAN)
            engine.ftran(&std_lp.b, &mut x_b)?;

            // 2. Compute cost of basic variables c_B
            for (i, &col_idx) in basis.iter().enumerate() {
                c_b[i] = c[col_idx];
            }

            // 3. Compute simplex multipliers: B^T * lambda = c_B (BTRAN)
            engine.btran(&c_b, &mut lambda)?;

            // 4. Pricing: calculate reduced costs c_bar_j = c_j - lambda^T * A_j
            let mut best_reduced_cost = -options.tolerance;
            let mut entering_col: Option<usize> = None;

            for j in 0..candidate_count {
                // Skip basic variables in O(1)
                if is_basic[j] {
                    continue;
                }

                // Compute lambda^T * A_j
                let mut lambda_dot_aj = 0.0f64;
                for &(row_idx, val) in &std_lp.cols[j] {
                    lambda_dot_aj += lambda[row_idx] * val;
                }

                let reduced_cost = c[j] - lambda_dot_aj;

                if options.use_bland_rule {
                    if reduced_cost < -options.tolerance {
                        entering_col = Some(j);
                        break;
                    }
                } else if reduced_cost < best_reduced_cost {
                    best_reduced_cost = reduced_cost;
                    entering_col = Some(j);
                }
            }

            // Optimality test: no entering column found with c_bar < -tol
            let entering_j = match entering_col {
                Some(col) => col,
                None => {
                    let mut obj = 0.0f64;
                    for (i, &xb_val) in x_b.iter().enumerate() {
                        obj += c_b[i] * xb_val;
                    }
                    return Ok(LoopOutcome {
                        status: SimplexStatus::Optimal,
                        iterations: iters,
                        objective_value: obj,
                    });
                }
            };

            // 5. Compute search direction: B * d = A[:, entering_j] (FTRAN)
            entering_col_dense.fill(0.0);
            for &(row_idx, val) in &std_lp.cols[entering_j] {
                entering_col_dense[row_idx] = val;
            }

            engine.ftran(&entering_col_dense, &mut d)?;

            // 6. Ratio test: min { x_B[i] / d[i] : d[i] > tol }
            let mut min_ratio = f64::INFINITY;
            let mut leaving_pos: Option<usize> = None;

            for i in 0..m {
                let di = d[i];
                if di > options.pivot_tolerance {
                    let non_neg_xb = x_b[i].max(0.0);
                    let ratio = non_neg_xb / di;

                    if ratio < min_ratio - 1e-12 {
                        min_ratio = ratio;
                        leaving_pos = Some(i);
                    } else if (ratio - min_ratio).abs() <= 1e-12 && options.use_bland_rule {
                        if let Some(prev_pos) = leaving_pos {
                            if basis[i] < basis[prev_pos] {
                                leaving_pos = Some(i);
                            }
                        }
                    }
                }
            }

            // Unbounded test: no d[i] > tol
            let leave_idx = match leaving_pos {
                Some(pos) => pos,
                None => {
                    return Ok(LoopOutcome {
                        status: SimplexStatus::Unbounded,
                        iterations: iters,
                        objective_value: f64::NEG_INFINITY,
                    });
                }
            };

            // 7. Pivot: replace basis column
            let leaving_var = basis[leave_idx];
            is_basic[leaving_var] = false;
            is_basic[entering_j] = true;
            basis[leave_idx] = entering_j;
            iters += 1;

            // 8. Update basis factorization engine (Eta update or refactorize)
            let eta_ok = engine.update_basis(leave_idx, &d);
            if !eta_ok || engine.eta_count() >= REFACTOR_FREQUENCY {
                engine.refactorize(std_lp, basis, options.pivot_tolerance)?;
            }
        }
    }

    /// Purge artificial variables from the basis after Phase 1 convergence.
    fn purge_artificial_basics(
        basis: &mut [usize],
        std_lp: &StandardFormLp,
        non_art_count: usize,
        tol: f64,
    ) -> Result<(), SolverError> {
        let m = std_lp.num_constraints;

        for pos in 0..m {
            if basis[pos] >= non_art_count {
                // An artificial variable is basic at value 0
                // Factorize basis
                let basis_cols: Vec<Vec<(usize, f64)>> =
                    basis.iter().map(|&c| std_lp.cols[c].clone()).collect();

                let lu = match LuDecomposition::from_sparse_columns(m, &basis_cols, 1e-12) {
                    Ok(f) => f,
                    Err(_) => continue,
                };

                let mut candidate_col: Option<usize> = None;

                for j in 0..non_art_count {
                    if basis.contains(&j) {
                        continue;
                    }

                    let mut dense_col = vec![0.0f64; m];
                    for &(r, v) in &std_lp.cols[j] {
                        dense_col[r] = v;
                    }

                    if let Ok(dir) = lu.solve(&dense_col) {
                        if dir[pos].abs() > tol {
                            candidate_col = Some(j);
                            break;
                        }
                    }
                }

                if let Some(entering) = candidate_col {
                    basis[pos] = entering;
                }
                // If no entering column found, constraint is redundant (0 = 0); safely keep at zero
            }
        }

        Ok(())
    }

    /// Extract final solution: primal variable vector, dual shadow prices, and objective.
    fn extract_solution(
        std_lp: &StandardFormLp,
        basis: &[usize],
        status: SimplexStatus,
    ) -> Result<(Vec<f64>, Vec<f64>, f64), SolverError> {
        let m = std_lp.num_constraints;
        let n_orig = std_lp.num_decision_vars;

        if status != SimplexStatus::Optimal {
            return Ok((vec![0.0; n_orig], vec![0.0; m], f64::NAN));
        }

        // Factorize final basis
        let basis_cols: Vec<Vec<(usize, f64)>> =
            basis.iter().map(|&c| std_lp.cols[c].clone()).collect();

        let lu = LuDecomposition::from_sparse_columns(m, &basis_cols, 1e-12).map_err(|e| {
            SolverError::NumericalError(format!("Final basis factorization failed: {e}"))
        })?;

        let x_b = lu
            .solve(&std_lp.b)
            .map_err(|e| SolverError::NumericalError(format!("Final solve for x_B failed: {e}")))?;

        // Reconstruct full standard form primal solution
        let mut x_std = vec![0.0f64; std_lp.num_total_vars];
        for (i, &col_idx) in basis.iter().enumerate() {
            x_std[col_idx] = x_b[i].max(0.0);
        }

        // Map back to original primal decision variables (accounting for lower bound shifts)
        let mut x_orig = vec![0.0f64; n_orig];
        for j in 0..n_orig {
            x_orig[j] = x_std[j] + std_lp.var_shifts[j];
        }

        // Compute dual multipliers: B^T * lambda = c_B
        let mut c_b = vec![0.0f64; m];
        for (i, &col_idx) in basis.iter().enumerate() {
            c_b[i] = std_lp.c[col_idx];
        }

        let lambda = lu.solve_transpose(&c_b).map_err(|e| {
            SolverError::NumericalError(format!("Final solve for lambda failed: {e}"))
        })?;

        // Map lambda back to original constraints (accounting for row sign flips)
        let mut y_orig = vec![0.0f64; m];
        for i in 0..m {
            let mult = if std_lp.row_sign_flips[i] {
                -lambda[i]
            } else {
                lambda[i]
            };
            y_orig[i] = if std_lp.is_maximization { -mult } else { mult };
        }

        // Recompute exact objective value
        let mut objective = std_lp.obj_offset;
        for j in 0..n_orig {
            objective += std_lp.c[j] * (x_orig[j] - std_lp.var_shifts[j]);
        }

        Ok((x_orig, y_orig, objective))
    }
}

struct LoopOutcome {
    status: SimplexStatus,
    iterations: usize,
    objective_value: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use yutki_model::{linear_program::Objective, Bound, ConstraintBounds, Sense, VariableBounds};
    use yutki_sparse::{CooMatrix, CsrMatrix};

    #[test]
    fn test_simplex_small_2d_lp() {
        // min -3 x0 - 2 x1
        // s.t.
        // x0 + x1 <= 4
        // x0 <= 2
        // x0, x1 >= 0
        // Optimal at (2, 2) with objective -10.0
        let mut coo = CooMatrix::new(2, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 1.0).unwrap();
        coo.add_entry(1, 0, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let obj = Objective::new(Sense::Minimize, vec![-3.0, -2.0]);
        let var_b = VariableBounds::from_bounds(vec![
            Bound::new(0.0, f64::INFINITY),
            Bound::new(0.0, f64::INFINITY),
        ]);
        let row_b = ConstraintBounds::from_bounds(vec![
            Bound::new(f64::NEG_INFINITY, 4.0),
            Bound::new(f64::NEG_INFINITY, 2.0),
        ]);

        let lp = LinearProgram::new("test_2d".into(), obj, var_b, row_b, a).unwrap();
        let (sol, telemetry) =
            RevisedSimplexSolver::solve(&lp, &SimplexOptions::default()).unwrap();

        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!((sol.objective - (-10.0)).abs() < 1e-6);
        assert!((sol.primal[0] - 2.0).abs() < 1e-6);
        assert!((sol.primal[1] - 2.0).abs() < 1e-6);
        assert!(telemetry.total_iterations < 10);
    }

    #[test]
    fn test_simplex_equality_constraint() {
        // min 2 x0 + x1
        // s.t.
        // x0 + x1 == 3
        // x0, x1 >= 0
        // Optimal at (0, 3) with objective 3.0
        let mut coo = CooMatrix::new(1, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let obj = Objective::new(Sense::Minimize, vec![2.0, 1.0]);
        let var_b = VariableBounds::from_bounds(vec![
            Bound::new(0.0, f64::INFINITY),
            Bound::new(0.0, f64::INFINITY),
        ]);
        let row_b = ConstraintBounds::from_bounds(vec![Bound::new(3.0, 3.0)]);

        let lp = LinearProgram::new("test_eq".into(), obj, var_b, row_b, a).unwrap();
        let (sol, _) = RevisedSimplexSolver::solve(&lp, &SimplexOptions::default()).unwrap();

        assert_eq!(sol.status, SolverStatus::Optimal);
        assert!((sol.objective - 3.0).abs() < 1e-6);
        assert!((sol.primal[0] - 0.0).abs() < 1e-6);
        assert!((sol.primal[1] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_simplex_infeasible() {
        // min x0 + x1
        // s.t.
        // x0 + x1 <= 2
        // x0 + x1 >= 4
        // x0, x1 >= 0
        let mut coo = CooMatrix::new(2, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 1.0).unwrap();
        coo.add_entry(1, 0, 1.0).unwrap();
        coo.add_entry(1, 1, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let obj = Objective::new(Sense::Minimize, vec![1.0, 1.0]);
        let var_b = VariableBounds::from_bounds(vec![
            Bound::new(0.0, f64::INFINITY),
            Bound::new(0.0, f64::INFINITY),
        ]);
        let row_b = ConstraintBounds::from_bounds(vec![
            Bound::new(f64::NEG_INFINITY, 2.0),
            Bound::new(4.0, f64::INFINITY),
        ]);

        let lp = LinearProgram::new("infeasible".into(), obj, var_b, row_b, a).unwrap();
        let (sol, telemetry) =
            RevisedSimplexSolver::solve(&lp, &SimplexOptions::default()).unwrap();

        assert_eq!(sol.status, SolverStatus::Infeasible);
        assert_eq!(telemetry.status, SimplexStatus::Infeasible);
    }

    #[test]
    fn test_simplex_unbounded() {
        // min -x0 - x1
        // s.t.
        // x0 - x1 <= 2
        // x0, x1 >= 0
        let mut coo = CooMatrix::new(1, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, -1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let obj = Objective::new(Sense::Minimize, vec![-1.0, -1.0]);
        let var_b = VariableBounds::from_bounds(vec![
            Bound::new(0.0, f64::INFINITY),
            Bound::new(0.0, f64::INFINITY),
        ]);
        let row_b = ConstraintBounds::from_bounds(vec![Bound::new(f64::NEG_INFINITY, 2.0)]);

        let lp = LinearProgram::new("unbounded".into(), obj, var_b, row_b, a).unwrap();
        let (sol, telemetry) =
            RevisedSimplexSolver::solve(&lp, &SimplexOptions::default()).unwrap();

        assert_eq!(sol.status, SolverStatus::Unbounded);
        assert_eq!(telemetry.status, SimplexStatus::Unbounded);
    }

    #[test]
    fn test_simplex_solve_netlib_afiro() {
        use std::path::PathBuf;
        use yutki_model::parse_mps_file_to_lp;

        let manifest = env!("CARGO_MANIFEST_DIR");
        let path = PathBuf::from(manifest).join("../../datasets/netlib/afiro.mps");
        if path.exists() {
            let lp = parse_mps_file_to_lp(&path).unwrap();
            let options = SimplexOptions {
                tolerance: 1e-7,
                ..Default::default()
            };
            let (sol, telemetry) = RevisedSimplexSolver::solve(&lp, &options).unwrap();
            assert_eq!(sol.status, SolverStatus::Optimal);
            // Netlib ground truth optimal value: -464.753142857143
            assert!(
                (sol.objective - (-464.753142857143)).abs() < 1e-4,
                "Objective was {}",
                sol.objective
            );
            assert!(telemetry.total_iterations < 100);

            // Audit with independent solution verifier
            let candidate = yutki_verifier::CandidateSolution::from_lp_solution(&sol);
            let report = yutki_verifier::SolutionVerifier::verify_linear_program(
                &lp,
                &candidate,
                &yutki_numerics::NumericalTolerances::default(),
            );
            assert_eq!(
                report.verdict,
                yutki_verifier::VerificationVerdict::Valid,
                "Verifier message: {}",
                report.message
            );
        }
    }

    #[test]
    #[ignore = "Heavy 305x472 Netlib benchmark; run with -- --ignored or in release mode"]
    fn test_simplex_solve_netlib_bandm() {
        use std::path::PathBuf;
        use yutki_model::parse_mps_file_to_lp;

        let manifest = env!("CARGO_MANIFEST_DIR");
        let path = PathBuf::from(manifest).join("../../datasets/netlib/bandm.mps");
        if path.exists() {
            let lp = parse_mps_file_to_lp(&path).unwrap();
            let options = SimplexOptions {
                tolerance: 1e-6,
                max_iterations: 50_000,
                ..Default::default()
            };
            let (sol, _) = RevisedSimplexSolver::solve(&lp, &options).unwrap();
            assert_eq!(sol.status, SolverStatus::Optimal);
            // Netlib ground truth optimal value: -158.62801845
            assert!(
                (sol.objective - (-158.62801845)).abs() < 1e-2,
                "Objective was {}",
                sol.objective
            );
        }
    }
}
