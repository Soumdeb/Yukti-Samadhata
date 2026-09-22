//! Independent solution verifier.
//! Evaluates candidate primal-dual solutions directly against the raw unscaled
//! problem definition (LinearProgram or LpProblem) without trusting solver internal statuses.

use serde::{Deserialize, Serialize};
use yukti_model::{LinearProgram, LpProblem, LpSolution};
use yukti_numerics::NumericalTolerances;

/// Verification outcome verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationVerdict {
    Valid,
    NumericallyUncertain,
    Invalid,
}

impl std::fmt::Display for VerificationVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VerificationVerdict::Valid => write!(f, "VALID"),
            VerificationVerdict::NumericallyUncertain => write!(f, "NUMERICALLY_UNCERTAIN"),
            VerificationVerdict::Invalid => write!(f, "INVALID"),
        }
    }
}

/// Candidate solution submitted for independent verification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CandidateSolution {
    pub primal: Vec<f64>,
    pub dual: Option<Vec<f64>>,
    pub objective: Option<f64>,
}

impl CandidateSolution {
    pub fn new(primal: Vec<f64>) -> Self {
        Self {
            primal,
            dual: None,
            objective: None,
        }
    }

    pub fn with_objective(primal: Vec<f64>, objective: f64) -> Self {
        Self {
            primal,
            dual: None,
            objective: Some(objective),
        }
    }

    pub fn from_lp_solution(sol: &LpSolution) -> Self {
        Self {
            primal: sol.primal.clone(),
            dual: Some(sol.dual.clone()),
            objective: Some(sol.objective),
        }
    }
}

impl From<&LpSolution> for CandidateSolution {
    fn from(sol: &LpSolution) -> Self {
        Self::from_lp_solution(sol)
    }
}

impl From<LpSolution> for CandidateSolution {
    fn from(sol: LpSolution) -> Self {
        Self::from_lp_solution(&sol)
    }
}

/// Detailed independent verification report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationReport {
    pub verdict: VerificationVerdict,
    pub max_var_bound_violation: f64,
    pub max_constraint_violation: f64,
    pub max_violation: f64,
    pub ax: Vec<f64>,
    pub recomputed_objective: f64,
    pub objective_discrepancy: f64,
    pub message: String,
}

/// Completely decoupled, standalone LP solution verifier.
/// Independently evaluates primal bounds, SpMV Ax, row constraints, and objective.
pub struct SolutionVerifier;

impl SolutionVerifier {
    /// Independently verify a candidate solution against a LinearProgram.
    pub fn verify_linear_program(
        lp: &LinearProgram,
        candidate: &CandidateSolution,
        tolerances: &NumericalTolerances,
    ) -> VerificationReport {
        let x = &candidate.primal;
        let n = lp.num_variables();
        let m = lp.num_constraints();

        // 1. Dimension Check
        if x.len() != n {
            return VerificationReport {
                verdict: VerificationVerdict::Invalid,
                max_var_bound_violation: f64::INFINITY,
                max_constraint_violation: f64::INFINITY,
                max_violation: f64::INFINITY,
                ax: Vec::new(),
                recomputed_objective: f64::NAN,
                objective_discrepancy: f64::INFINITY,
                message: format!(
                    "Dimension mismatch: candidate x has {} entries, model requires {}",
                    x.len(),
                    n
                ),
            };
        }

        // 2. Finite Values Sentinel Check (NaN / Inf)
        for (j, &val) in x.iter().enumerate() {
            if !val.is_finite() {
                return VerificationReport {
                    verdict: VerificationVerdict::Invalid,
                    max_var_bound_violation: f64::INFINITY,
                    max_constraint_violation: f64::INFINITY,
                    max_violation: f64::INFINITY,
                    ax: Vec::new(),
                    recomputed_objective: f64::NAN,
                    objective_discrepancy: f64::INFINITY,
                    message: format!(
                        "Non-finite value ({val}) encountered in candidate variable at index {j}"
                    ),
                };
            }
        }

        // 3. Variable Bound Violations Check: max(0, l_v - x) + max(0, x - u_v)
        let mut max_var_viol = 0.0f64;
        for (b, &xj) in lp.variable_bounds.bounds.iter().zip(x.iter()) {
            let viol = b.violation(xj);
            if viol > max_var_viol {
                max_var_viol = viol;
            }
        }

        // 4. Independent SpMV: A * x via isolated scalar loop
        let mut ax = vec![0.0f64; m];
        let mut max_row_viol = 0.0f64;
        for (i, ax_i) in ax.iter_mut().enumerate().take(m) {
            let start = lp.a.row_ptrs[i];
            let end = lp.a.row_ptrs[i + 1];
            let mut sum_i = 0.0f64;
            for k in start..end {
                let col = lp.a.col_indices[k];
                sum_i += lp.a.values[k] * x[col];
            }
            *ax_i = sum_i;
            let viol = lp.constraint_bounds.bounds[i].violation(sum_i);
            if viol > max_row_viol {
                max_row_viol = viol;
            }
        }

        // 5. Independent Objective Recomputation: c^T x + offset
        let recomputed_obj = match lp.evaluate_objective(x) {
            Ok(val) => val,
            Err(e) => {
                return VerificationReport {
                    verdict: VerificationVerdict::Invalid,
                    max_var_bound_violation: max_var_viol,
                    max_constraint_violation: max_row_viol,
                    max_violation: max_var_viol.max(max_row_viol),
                    ax,
                    recomputed_objective: f64::NAN,
                    objective_discrepancy: f64::INFINITY,
                    message: format!("Objective evaluation failed: {e}"),
                };
            }
        };

        let obj_discrepancy = match candidate.objective {
            Some(reported_obj) => (recomputed_obj - reported_obj).abs(),
            None => 0.0,
        };

        let max_violation = max_var_viol.max(max_row_viol);

        // Discrepancy threshold for reported objective
        let obj_mismatch = obj_discrepancy > 10.0 * tolerances.primal_tol.max(tolerances.zero_tol);

        // Verdict Classification
        let verdict = if obj_mismatch {
            VerificationVerdict::Invalid
        } else if max_violation <= tolerances.primal_tol {
            VerificationVerdict::Valid
        } else if max_violation <= 100.0 * tolerances.primal_tol {
            VerificationVerdict::NumericallyUncertain
        } else {
            VerificationVerdict::Invalid
        };

        let message = format!(
            "Var viol: {:.2e}, Row viol: {:.2e}, Obj disc: {:.2e} (Verdict: {verdict})",
            max_var_viol, max_row_viol, obj_discrepancy
        );

        VerificationReport {
            verdict,
            max_var_bound_violation: max_var_viol,
            max_constraint_violation: max_row_viol,
            max_violation,
            ax,
            recomputed_objective: recomputed_obj,
            objective_discrepancy: obj_discrepancy,
            message,
        }
    }

    /// Convenience verifier evaluating an LpSolution against an LpProblem.
    pub fn verify(
        problem: &LpProblem,
        solution: &LpSolution,
        tolerances: &NumericalTolerances,
    ) -> VerificationReport {
        let lp = match LinearProgram::from_lp_problem(problem.clone()) {
            Ok(l) => l,
            Err(e) => {
                return VerificationReport {
                    verdict: VerificationVerdict::Invalid,
                    max_var_bound_violation: f64::INFINITY,
                    max_constraint_violation: f64::INFINITY,
                    max_violation: f64::INFINITY,
                    ax: Vec::new(),
                    recomputed_objective: f64::NAN,
                    objective_discrepancy: f64::INFINITY,
                    message: format!("Problem validation failed: {e}"),
                };
            }
        };
        let candidate = CandidateSolution::from_lp_solution(solution);
        Self::verify_linear_program(&lp, &candidate, tolerances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yukti_model::{Bound, ConstraintBounds, Objective, Sense, VariableBounds};
    use yukti_sparse::{CooMatrix, CsrMatrix};

    fn make_test_lp() -> LinearProgram {
        // min x1 + 2*x2
        // s.t. x1 + x2 <= 10.0
        //      0 <= x1 <= 5.0
        //      0 <= x2 <= 5.0
        let mut coo = CooMatrix::new(1, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();

        let obj = Objective::new(Sense::Minimize, vec![1.0, 2.0]);
        let var_bounds =
            VariableBounds::from_bounds(vec![Bound::new(0.0, 5.0), Bound::new(0.0, 5.0)]);
        let con_bounds = ConstraintBounds::from_bounds(vec![Bound::upper_only(10.0)]);

        LinearProgram::new("test_lp".into(), obj, var_bounds, con_bounds, a).unwrap()
    }

    #[test]
    fn test_verifier_valid_solution() {
        let lp = make_test_lp();
        let tols = NumericalTolerances::default();

        // Feasible: x1 = 3, x2 = 4 -> Ax = 7 <= 10, obj = 3*1 + 4*2 = 11.0
        let candidate = CandidateSolution::with_objective(vec![3.0, 4.0], 11.0);
        let rep = SolutionVerifier::verify_linear_program(&lp, &candidate, &tols);
        assert_eq!(rep.verdict, VerificationVerdict::Valid);
        assert_eq!(rep.ax, vec![7.0]);
        assert!((rep.recomputed_objective - 11.0).abs() < 1e-10);
        assert_eq!(rep.max_violation, 0.0);
    }

    #[test]
    fn test_verifier_catches_wrong_variable_count() {
        let lp = make_test_lp();
        let tols = NumericalTolerances::default();

        // 3 variables provided instead of 2
        let candidate = CandidateSolution::new(vec![1.0, 2.0, 3.0]);
        let rep = SolutionVerifier::verify_linear_program(&lp, &candidate, &tols);
        assert_eq!(rep.verdict, VerificationVerdict::Invalid);
        assert!(rep.message.contains("Dimension mismatch"));
    }

    #[test]
    fn test_verifier_catches_violated_constraint() {
        let tols = NumericalTolerances::default();

        let mut coo = CooMatrix::new(1, 2);
        coo.add_entry(0, 0, 1.0).unwrap();
        coo.add_entry(0, 1, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();
        let obj = Objective::new(Sense::Minimize, vec![1.0, 1.0]);
        let var_bounds =
            VariableBounds::from_bounds(vec![Bound::new(0.0, 10.0), Bound::new(0.0, 10.0)]);
        let con_bounds = ConstraintBounds::from_bounds(vec![Bound::upper_only(6.0)]);
        let lp_tight = LinearProgram::new("tight".into(), obj, var_bounds, con_bounds, a).unwrap();

        // x1=4, x2=4 -> Ax = 8 > 6
        let candidate = CandidateSolution::new(vec![4.0, 4.0]);
        let rep = SolutionVerifier::verify_linear_program(&lp_tight, &candidate, &tols);
        assert_eq!(rep.verdict, VerificationVerdict::Invalid);
        assert!(rep.max_constraint_violation > 1.99);
    }

    #[test]
    fn test_verifier_catches_bound_violation() {
        let lp = make_test_lp();
        let tols = NumericalTolerances::default();

        // x1 = 8.0 exceeds upper bound 5.0
        let candidate = CandidateSolution::new(vec![8.0, 1.0]);
        let rep = SolutionVerifier::verify_linear_program(&lp, &candidate, &tols);
        assert_eq!(rep.verdict, VerificationVerdict::Invalid);
        assert!(rep.max_var_bound_violation >= 3.0);
    }

    #[test]
    fn test_verifier_catches_nan() {
        let lp = make_test_lp();
        let tols = NumericalTolerances::default();

        let candidate = CandidateSolution::new(vec![f64::NAN, 1.0]);
        let rep = SolutionVerifier::verify_linear_program(&lp, &candidate, &tols);
        assert_eq!(rep.verdict, VerificationVerdict::Invalid);
        assert!(rep.message.contains("Non-finite"));
    }

    #[test]
    fn test_verifier_catches_inf() {
        let lp = make_test_lp();
        let tols = NumericalTolerances::default();

        let candidate_pos = CandidateSolution::new(vec![f64::INFINITY, 1.0]);
        let rep1 = SolutionVerifier::verify_linear_program(&lp, &candidate_pos, &tols);
        assert_eq!(rep1.verdict, VerificationVerdict::Invalid);

        let candidate_neg = CandidateSolution::new(vec![1.0, f64::NEG_INFINITY]);
        let rep2 = SolutionVerifier::verify_linear_program(&lp, &candidate_neg, &tols);
        assert_eq!(rep2.verdict, VerificationVerdict::Invalid);
    }

    #[test]
    fn test_verifier_catches_objective_corruption() {
        let lp = make_test_lp();
        let tols = NumericalTolerances::default();

        // Feasible point x = [2, 3] -> recomputed obj = 1*2 + 2*3 = 8.0
        // Candidate claims objective = 999.0
        let candidate = CandidateSolution::with_objective(vec![2.0, 3.0], 999.0);
        let rep = SolutionVerifier::verify_linear_program(&lp, &candidate, &tols);
        assert_eq!(rep.verdict, VerificationVerdict::Invalid);
        assert!(rep.objective_discrepancy > 900.0);
    }

    #[test]
    fn test_verifier_numerically_uncertain() {
        let mut coo = CooMatrix::new(1, 1);
        coo.add_entry(0, 0, 1.0).unwrap();
        let a = CsrMatrix::from_coo(&coo).unwrap();
        let obj = Objective::new(Sense::Minimize, vec![1.0]);
        let var_bounds = VariableBounds::from_bounds(vec![Bound::new(0.0, 10.0)]);
        let con_bounds = ConstraintBounds::from_bounds(vec![Bound::upper_only(5.0)]);
        let lp = LinearProgram::new("test".into(), obj, var_bounds, con_bounds, a).unwrap();

        let tols = NumericalTolerances {
            primal_tol: 1e-5,
            ..Default::default()
        };

        // Violation is 2e-5, which is > 1e-5 (tol) and <= 100 * 1e-5 (1e-3)
        let candidate = CandidateSolution::new(vec![5.0 + 2e-5]);
        let rep = SolutionVerifier::verify_linear_program(&lp, &candidate, &tols);
        assert_eq!(rep.verdict, VerificationVerdict::NumericallyUncertain);
    }

    #[test]
    fn test_verifier_ignores_pdhg_fake_optimal_status() {
        // Solution claims Optimal status, but point is violently infeasible
        let fake_solution = LpSolution {
            status: yukti_model::SolverStatus::Optimal, // solver claims optimal!
            primal: vec![100.0, 200.0],                 // totally infeasible
            dual: vec![0.0],
            objective: 500.0,
            iterations: 50,
            primal_residual: 0.0,
            dual_residual: 0.0,
            solve_time_secs: 0.1,
        };

        let lp = make_test_lp();
        let problem = lp.to_lp_problem();
        let tols = NumericalTolerances::default();

        let rep = SolutionVerifier::verify(&problem, &fake_solution, &tols);
        // Verifier must not trust solver status, must return Invalid!
        assert_eq!(rep.verdict, VerificationVerdict::Invalid);
    }

    #[test]
    fn test_verifier_catches_negative_variable_violating_non_negativity() {
        let lp = make_test_lp();
        let tols = NumericalTolerances::default();

        // x1 = -0.5 violates non-negativity lower bound 0.0
        let candidate = CandidateSolution::new(vec![-0.5, 2.0]);
        let rep = SolutionVerifier::verify_linear_program(&lp, &candidate, &tols);
        assert_eq!(rep.verdict, VerificationVerdict::Invalid);
        assert!(rep.max_var_bound_violation >= 0.5);
    }

    #[test]
    fn test_verifier_catches_sign_inversion_corruption() {
        let lp = make_test_lp();
        let tols = NumericalTolerances::default();

        // Legitimate feasible point is [3.0, 4.0] (obj = 11.0)
        // Corrupted inverted candidate: [-3.0, -4.0] claiming obj = 11.0
        let corrupted = CandidateSolution::with_objective(vec![-3.0, -4.0], 11.0);
        let rep = SolutionVerifier::verify_linear_program(&lp, &corrupted, &tols);
        assert_eq!(rep.verdict, VerificationVerdict::Invalid);
        // Discrepancy between recomputed obj (-11.0) and claimed obj (11.0) is 22.0
        assert!(rep.objective_discrepancy >= 20.0);
    }
}
