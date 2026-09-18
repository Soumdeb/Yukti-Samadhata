//! Structured solver lifecycle events, execution trace logger, and output formatters.

use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use yutki_model::{LogFormat, SolverStatus};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum SolverEvent {
    ModelLoaded {
        name: String,
        variables: usize,
        constraints: usize,
        nonzeros: usize,
    },
    ModelValidated {
        status: String,
    },
    FingerprintComputed {
        density: f64,
        condition_ratio: f64,
    },
    PresolveFinished {
        removed_variables: usize,
        eliminated_rows: usize,
    },
    BackendSelected {
        backend: String,
        device: String,
    },
    PdhgIteration {
        iter: usize,
        primal_res: f64,
        dual_res: f64,
        objective: f64,
    },
    PdhgRestart {
        iter: usize,
        reason: String,
    },
    NumericalWarning {
        warning: String,
    },
    VerificationFinished {
        verdict: String,
        max_violation: f64,
    },
    SolveFinished {
        status: SolverStatus,
        objective: f64,
        iterations: usize,
        wall_time_secs: f64,
    },
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SolverTrace {
    pub events: Vec<SolverEvent>,
}

impl SolverTrace {
    pub fn new() -> Self {
        Self { events: Vec::new() }
    }

    pub fn record(&mut self, event: SolverEvent) {
        self.events.push(event);
    }
}

/// Global/local trace sink with configurable formatting
pub struct EventLogger {
    format: LogFormat,
    trace: Arc<Mutex<SolverTrace>>,
}

impl EventLogger {
    pub fn new(format: LogFormat) -> Self {
        Self {
            format,
            trace: Arc::new(Mutex::new(SolverTrace::new())),
        }
    }

    pub fn emit(&self, event: SolverEvent) {
        if let Ok(mut t) = self.trace.lock() {
            t.record(event.clone());
        }

        match self.format {
            LogFormat::Quiet => {}
            LogFormat::Json => {
                if let Ok(json_line) = serde_json::to_string(&event) {
                    println!("{json_line}");
                }
            }
            LogFormat::Human => {
                Self::print_human(&event);
            }
        }
    }

    fn print_human(event: &SolverEvent) {
        match event {
            SolverEvent::ModelLoaded {
                name,
                variables,
                constraints,
                nonzeros,
            } => {
                println!("  [MODEL]       Loaded: {name} (Vars: {variables}, Cons: {constraints}, NNZ: {nonzeros})");
            }
            SolverEvent::ModelValidated { status } => {
                println!("  [VALIDATE]    Model coherence check: {status}");
            }
            SolverEvent::FingerprintComputed {
                density,
                condition_ratio,
            } => {
                println!(
                    "  [FINGERPRINT] Density: {:.2}%, Dynamic Range: {:.2e}",
                    density * 100.0,
                    condition_ratio
                );
            }
            SolverEvent::PresolveFinished {
                removed_variables,
                eliminated_rows,
            } => {
                println!(
                    "  [PRESOLVE]    Eliminated {removed_variables} fixed vars, {eliminated_rows} rows"
                );
            }
            SolverEvent::BackendSelected { backend, device } => {
                println!("  [BACKEND]     Active: {backend} ({device})");
            }
            SolverEvent::PdhgIteration {
                iter,
                primal_res,
                dual_res,
                objective,
            } => {
                if iter % 50 == 0 || *iter < 10 {
                    println!(
                        "  [PDHG]        Iter {iter:>5} | Obj: {objective:>12.6} | Pres: {primal_res:>9.2e} | Dres: {dual_res:>9.2e}"
                    );
                }
            }
            SolverEvent::PdhgRestart { iter, reason } => {
                println!("  [RESTART]     Iter {iter}: {reason}");
            }
            SolverEvent::NumericalWarning { warning } => {
                println!("  [WARNING]     {warning}");
            }
            SolverEvent::VerificationFinished {
                verdict,
                max_violation,
            } => {
                println!("  [VERIFY]      Verdict: {verdict} (Max violation: {max_violation:.2e})");
            }
            SolverEvent::SolveFinished {
                status,
                objective,
                iterations,
                wall_time_secs,
            } => {
                println!("  [RESULT]      Status: {status} | Objective: {objective:.6} | Iters: {iterations} | Time: {wall_time_secs:.4}s");
            }
        }
    }

    pub fn get_trace(&self) -> SolverTrace {
        self.trace.lock().map(|t| t.clone()).unwrap_or_default()
    }
}
