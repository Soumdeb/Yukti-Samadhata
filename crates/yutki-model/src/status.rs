use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SolverStatus {
    Optimal,
    Infeasible,
    Unbounded,
    IterationLimit,
    TimeLimit,
    NumericalStagnation,
    NumericalFailure,
    NotSolved,
}

impl std::fmt::Display for SolverStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolverStatus::Optimal => write!(f, "OPTIMAL"),
            SolverStatus::Infeasible => write!(f, "INFEASIBLE"),
            SolverStatus::Unbounded => write!(f, "UNBOUNDED"),
            SolverStatus::IterationLimit => write!(f, "ITERATION_LIMIT"),
            SolverStatus::TimeLimit => write!(f, "TIME_LIMIT"),
            SolverStatus::NumericalStagnation => write!(f, "NUMERICAL_STAGNATION"),
            SolverStatus::NumericalFailure => write!(f, "NUMERICAL_FAILURE"),
            SolverStatus::NotSolved => write!(f, "NOT_SOLVED"),
        }
    }
}
