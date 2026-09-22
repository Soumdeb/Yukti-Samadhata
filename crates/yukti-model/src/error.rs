use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum SolverError {
    #[error("I/O error: {0}")]
    Io(String),

    #[error("MPS parse error on line {line}: {message}")]
    ParseError { line: usize, message: String },

    #[error("Validation error: {0}")]
    ValidationError(String),

    #[error("Dimension mismatch: expected {expected}, found {found}")]
    DimensionMismatch { expected: usize, found: usize },

    #[error("Numerical error: {0}")]
    NumericalError(String),

    #[error("Backend error: {0}")]
    BackendError(String),

    #[error("Authentication error: {0}")]
    AuthError(String),

    #[error("Unsupported feature: {0}")]
    Unsupported(String),
}
