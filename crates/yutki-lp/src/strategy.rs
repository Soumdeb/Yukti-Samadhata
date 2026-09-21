//! Sovereign Strategy Manager and Routing Pipeline.
//!
//! Implements the complete multi-stage solver architecture:
//! - Model Analysis & Structure Fingerprinting
//! - Problem Classification (LP, QP, MILP)
//! - LP Algorithm Strategy Routing (Revised Simplex, PDHG, IPM)
//! - Presolve Management & Coordinate Mapping
//! - Numerical Health Monitoring & Stagnation Fallback (PDHG -> Simplex)
//! - Invertible Postsolve Un-mapping
//! - Independent KKT / Residual Verification

use std::time::Instant;
use yutki_model::{BackendType, LinearProgram, LpProblem, LpSolution, SolverError, SolverStatus};
use yutki_verifier::{SolutionVerifier, VerificationReport};

use crate::ipm::{IpmOptions, IpmSolver, IpmStatus};
use crate::simplex::{RevisedSimplexSolver, SimplexOptions};
use crate::{PdhgOptions, PdhgSolver, PdhgStatus, SolverAlgorithm};

/// High-level mathematical optimization problem classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProblemType {
    /// Continuous Linear Program: min c^T x s.t. A x = b, l <= x <= u
    Lp,
    /// Quadratic Program: min 1/2 x^T Q x + c^T x s.t. A x = b, l <= x <= u
    Qp,
    /// Mixed-Integer Linear Program: LP with integrality constraints on subset of x
    Milp,
}

/// Structural fingerprint metrics extracted from the constraint matrix and vectors.
#[derive(Debug, Clone)]
pub struct ModelFingerprint {
    pub problem_type: ProblemType,
    pub num_rows: usize,
    pub num_cols: usize,
    pub num_nonzeros: usize,
    pub sparsity_density: f64,
    pub has_free_variables: bool,
    pub has_boxed_variables: bool,
    pub is_small_to_medium: bool,
}

impl ModelFingerprint {
    pub fn analyze(problem: &LpProblem) -> Self {
        let m = problem.num_constraints();
        let n = problem.num_variables();
        let nnz = problem.num_nonzeros();
        let total_entries = (m * n).max(1);
        let sparsity_density = nnz as f64 / total_entries as f64;

        let mut has_free = false;
        let mut has_boxed = false;
        for b in &problem.var_bounds {
            if b.lower.is_infinite() && b.upper.is_infinite() {
                has_free = true;
            } else if b.lower.is_finite() && b.upper.is_finite() {
                has_boxed = true;
            }
        }

        // Small-to-medium threshold: <= 2000 constraints and <= 5000 variables
        let is_small_to_medium = m <= 2000 && n <= 5000;

        Self {
            problem_type: ProblemType::Lp,
            num_rows: m,
            num_cols: n,
            num_nonzeros: nnz,
            sparsity_density,
            has_free_variables: has_free,
            has_boxed_variables: has_boxed,
            is_small_to_medium,
        }
    }
}

/// Active LP algorithm routing choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LpStrategy {
    /// Basis factorization, exact corner BFS (Simplex)
    RevisedSimplex,
    /// First-order saddle-point gradient (PDHG / PDLP)
    Pdhg,
    /// Primal-dual interior point barrier method (IPM)
    InteriorPoint,
}

/// Execution telemetry and audit from the Strategy Manager pipeline.
#[derive(Debug, Clone)]
pub struct StrategyExecutionReport {
    pub fingerprint: ModelFingerprint,
    pub selected_strategy: LpStrategy,
    pub fallback_triggered: bool,
    pub fallback_reason: Option<String>,
    pub presolve_reductions: usize,
    pub total_wall_time_secs: f64,
    pub verification: Option<VerificationReport>,
}

/// Strategy Manager configuring and executing the complete solver pipeline.
#[derive(Debug, Clone)]
pub struct StrategyManager {
    pub algorithm: SolverAlgorithm,
    pub backend: BackendType,
    pub tolerance: f64,
    pub max_iterations: usize,
    pub time_limit_secs: f64,
    pub enable_presolve: bool,
    pub enable_fallback: bool,
    pub enable_verification: bool,
}

impl Default for StrategyManager {
    fn default() -> Self {
        Self {
            algorithm: SolverAlgorithm::Auto,
            backend: BackendType::Auto,
            tolerance: 1e-6,
            max_iterations: 50_000,
            time_limit_secs: 120.0,
            enable_presolve: true,
            enable_fallback: true,
            enable_verification: true,
        }
    }
}

impl StrategyManager {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_algorithm(mut self, algo: SolverAlgorithm) -> Self {
        self.algorithm = algo;
        self
    }

    pub fn with_backend(mut self, backend: BackendType) -> Self {
        self.backend = backend;
        self
    }

    /// Select optimal LP algorithm given the problem fingerprint and configuration.
    pub fn select_strategy(&self, fp: &ModelFingerprint) -> LpStrategy {
        match self.algorithm {
            SolverAlgorithm::Simplex => LpStrategy::RevisedSimplex,
            SolverAlgorithm::Pdhg => LpStrategy::Pdhg,
            SolverAlgorithm::Ipm => LpStrategy::InteriorPoint,
            SolverAlgorithm::Auto => {
                if self.backend == BackendType::Gpu {
                    LpStrategy::Pdhg
                } else if fp.is_small_to_medium {
                    LpStrategy::RevisedSimplex
                } else {
                    LpStrategy::Pdhg
                }
            }
        }
    }

    /// Execute the end-to-end LP strategy pipeline according to the architecture flowchart.
    pub fn solve(
        &self,
        problem: &LpProblem,
    ) -> Result<(LpSolution, StrategyExecutionReport), SolverError> {
        let start_time = Instant::now();

        // 1. Model Validation
        problem.validate()?;

        // 2. Model Analysis & Fingerprinting
        let fingerprint = ModelFingerprint::analyze(problem);

        // 3. Strategy Selection
        let selected_strategy = self.select_strategy(&fingerprint);

        let mut fallback_triggered = false;
        let mut fallback_reason = None;
        let mut presolve_reductions = 0;

        let (mut solution, _strategy_used) = match selected_strategy {
            LpStrategy::InteriorPoint => {
                let lp = LinearProgram::from_lp_problem(problem.clone())?;
                let ipm_opts = IpmOptions {
                    max_iterations: self.max_iterations.min(500),
                    tolerance: self.tolerance,
                    time_limit_secs: self.time_limit_secs,
                    ..Default::default()
                };

                let ipm_res = IpmSolver::solve(&lp, &ipm_opts);

                let mut needs_fallback = false;
                let mut reason_str = String::new();

                match &ipm_res {
                    Ok((sol, telem)) => {
                        if telem.status == IpmStatus::NumericalStagnation
                            || sol.status == SolverStatus::NumericalFailure
                        {
                            needs_fallback = true;
                            reason_str =
                                "IPM encountered numerical stagnation or barrier singularity"
                                    .to_string();
                        }
                    }
                    Err(e) => {
                        needs_fallback = true;
                        reason_str = format!("IPM solve error: {e}");
                    }
                }

                // Numerical Health Stagnation Fallback to Revised Simplex
                if needs_fallback && self.enable_fallback {
                    fallback_triggered = true;
                    fallback_reason = Some(reason_str);

                    let raw_lp = LinearProgram::from_lp_problem(problem.clone())?;
                    let simplex_opts = SimplexOptions {
                        max_iterations: self.max_iterations.max(50_000),
                        tolerance: self.tolerance.min(1e-6),
                        time_limit_secs: self.time_limit_secs,
                        ..Default::default()
                    };
                    let (sol, _) = RevisedSimplexSolver::solve(&raw_lp, &simplex_opts)?;
                    (sol, LpStrategy::RevisedSimplex)
                } else {
                    let (sol, _) = ipm_res?;
                    (sol, LpStrategy::InteriorPoint)
                }
            }
            LpStrategy::RevisedSimplex => {
                let lp = LinearProgram::from_lp_problem(problem.clone())?;
                let opts = SimplexOptions {
                    max_iterations: self.max_iterations,
                    tolerance: self.tolerance,
                    time_limit_secs: self.time_limit_secs,
                    ..Default::default()
                };
                let (sol, _telem) = RevisedSimplexSolver::solve(&lp, &opts)?;
                (sol, LpStrategy::RevisedSimplex)
            }
            LpStrategy::Pdhg => {
                // Presolve Manager
                let (active_prob, transform_map) = if self.enable_presolve {
                    let (p_prob, t_map) = yutki_transform::presolve(problem, 1e-12)?;
                    presolve_reductions = t_map.fixed_vars.len();
                    (p_prob, Some(t_map))
                } else {
                    (problem.clone(), None)
                };

                let lp = LinearProgram::from_lp_problem(active_prob)?;
                let pdhg_opts = PdhgOptions {
                    backend_type: self.backend,
                    max_iterations: self.max_iterations,
                    time_limit_secs: self.time_limit_secs,
                    primal_tol: self.tolerance,
                    dual_tol: self.tolerance,
                    gap_tol: self.tolerance,
                    ..Default::default()
                };

                let solver = PdhgSolver::new(pdhg_opts);
                let pdhg_res = solver.solve(&lp);

                let mut needs_fallback = false;
                let mut reason_str = String::new();

                match &pdhg_res {
                    Ok(sol) => {
                        if sol.status == PdhgStatus::NumericalFailure {
                            needs_fallback = true;
                            reason_str = "PDHG encountered numerical stagnation / non-convergence"
                                .to_string();
                        }
                    }
                    Err(e) => {
                        needs_fallback = true;
                        reason_str = format!("PDHG solve failed with error: {e}");
                    }
                }

                // Stagnation Fallback to Revised Simplex
                if needs_fallback && self.enable_fallback {
                    fallback_triggered = true;
                    fallback_reason = Some(reason_str);

                    let raw_lp = LinearProgram::from_lp_problem(problem.clone())?;
                    let simplex_opts = SimplexOptions {
                        max_iterations: self.max_iterations.max(50_000),
                        tolerance: self.tolerance.min(1e-6),
                        time_limit_secs: self.time_limit_secs,
                        ..Default::default()
                    };
                    let (sol, _) = RevisedSimplexSolver::solve(&raw_lp, &simplex_opts)?;
                    (sol, LpStrategy::RevisedSimplex)
                } else {
                    let sol = pdhg_res?;
                    let presolved_sol = LpSolution {
                        status: SolverStatus::Optimal,
                        primal: sol.primal_solution,
                        dual: sol.dual_solution,
                        objective: sol.objective_value,
                        iterations: sol.iterations,
                        primal_residual: sol.residuals.primal_rel,
                        dual_residual: sol.residuals.dual_rel,
                        solve_time_secs: sol.solve_time_secs,
                    };

                    // Postsolve & Un-mapping
                    let restored = if let Some(t_map) = &transform_map {
                        yutki_transform::postsolve(&presolved_sol, t_map)
                    } else {
                        presolved_sol
                    };
                    (restored, LpStrategy::Pdhg)
                }
            }
        };

        // Recompute original objective value
        solution.objective = problem.evaluate_objective(&solution.primal);

        // Independent Solution Verification
        let verification = if self.enable_verification {
            let tols = yutki_numerics::NumericalTolerances {
                primal_tol: self.tolerance.max(1e-4),
                dual_tol: self.tolerance.max(1e-4),
                gap_tol: self.tolerance.max(1e-4),
                ..Default::default()
            };
            Some(SolutionVerifier::verify(problem, &solution, &tols))
        } else {
            None
        };

        let report = StrategyExecutionReport {
            fingerprint,
            selected_strategy,
            fallback_triggered,
            fallback_reason,
            presolve_reductions,
            total_wall_time_secs: start_time.elapsed().as_secs_f64(),
            verification,
        };

        Ok((solution, report))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_manager_afiro_simplex_and_verification() {
        let mps_path = if std::path::Path::new("datasets/netlib/afiro.mps").exists() {
            "datasets/netlib/afiro.mps"
        } else {
            "../../datasets/netlib/afiro.mps"
        };
        if !std::path::Path::new(mps_path).exists() {
            return;
        }

        let problem = yutki_model::parse_mps_file(mps_path).expect("parse afiro");
        let manager = StrategyManager::new().with_algorithm(SolverAlgorithm::Simplex);

        let (solution, report) = manager.solve(&problem).expect("solve afiro");
        assert_eq!(report.selected_strategy, LpStrategy::RevisedSimplex);
        assert!(!report.fallback_triggered);
        assert!((solution.objective - (-464.753142857143)).abs() < 1e-3);
        assert!(report.verification.is_some());
        let v = report.verification.unwrap();
        assert_eq!(v.verdict, yutki_verifier::VerificationVerdict::Valid);
    }

    #[test]
    fn test_strategy_manager_fingerprinting() {
        let mps_path = if std::path::Path::new("datasets/netlib/agg.mps").exists() {
            "datasets/netlib/agg.mps"
        } else {
            "../../datasets/netlib/agg.mps"
        };
        if !std::path::Path::new(mps_path).exists() {
            return;
        }

        let problem = yutki_model::parse_mps_file(mps_path).expect("parse agg");
        let fp = ModelFingerprint::analyze(&problem);
        assert_eq!(fp.problem_type, ProblemType::Lp);
        assert_eq!(fp.num_rows, 488);
        assert_eq!(fp.num_cols, 163);
        assert_eq!(fp.num_nonzeros, 2410);
        assert!(fp.is_small_to_medium);

        let manager = StrategyManager::new().with_algorithm(SolverAlgorithm::Auto);
        assert_eq!(manager.select_strategy(&fp), LpStrategy::RevisedSimplex);
    }

    #[test]
    fn test_strategy_manager_afiro_ipm() {
        let mps_path = if std::path::Path::new("datasets/netlib/afiro.mps").exists() {
            "datasets/netlib/afiro.mps"
        } else {
            "../../datasets/netlib/afiro.mps"
        };
        if !std::path::Path::new(mps_path).exists() {
            return;
        }

        let problem = yutki_model::parse_mps_file(mps_path).expect("parse afiro");
        let manager = StrategyManager::new().with_algorithm(SolverAlgorithm::Ipm);

        let (solution, report) = manager
            .solve(&problem)
            .expect("solve afiro via ipm pipeline");
        assert_eq!(report.selected_strategy, LpStrategy::InteriorPoint);
        // If solved directly via IPM or via graceful fallback on stagnation:
        // the objective must match the exact analytical ground truth (-464.753142857)
        assert!((solution.objective - (-464.753142857143)).abs() < 1e-2);
        assert!(report.verification.is_some());
    }
}
