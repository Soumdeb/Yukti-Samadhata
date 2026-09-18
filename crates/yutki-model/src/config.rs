use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BackendType {
    #[default]
    Auto,
    Cpu,
    Gpu,
}

impl std::str::FromStr for BackendType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(BackendType::Auto),
            "cpu" => Ok(BackendType::Cpu),
            "gpu" => Ok(BackendType::Gpu),
            _ => Err(format!("Unknown backend '{s}'. Valid: auto, cpu, gpu")),
        }
    }
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::Auto => write!(f, "auto"),
            BackendType::Cpu => write!(f, "cpu"),
            BackendType::Gpu => write!(f, "gpu"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum LogFormat {
    #[default]
    Human,
    Json,
    Quiet,
}

impl std::str::FromStr for LogFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "human" => Ok(LogFormat::Human),
            "json" => Ok(LogFormat::Json),
            "quiet" => Ok(LogFormat::Quiet),
            _ => Err(format!(
                "Unknown log format '{s}'. Valid: human, json, quiet"
            )),
        }
    }
}

impl std::fmt::Display for LogFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogFormat::Human => write!(f, "human"),
            LogFormat::Json => write!(f, "json"),
            LogFormat::Quiet => write!(f, "quiet"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SolverConfig {
    pub backend: BackendType,
    pub max_iterations: usize,
    pub time_limit_secs: f64,
    pub log_format: LogFormat,
    pub primal_tol: f64,
    pub dual_tol: f64,
    pub gap_tol: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            backend: BackendType::Auto,
            max_iterations: 50_000,
            time_limit_secs: 300.0,
            log_format: LogFormat::Human,
            primal_tol: 1e-6,
            dual_tol: 1e-6,
            gap_tol: 1e-6,
        }
    }
}
