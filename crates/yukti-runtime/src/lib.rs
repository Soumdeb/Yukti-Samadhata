//! Cross-Cutting Runtime Services for the Yukti-Samadhata solver pipeline.
//!
//! Provides:
//! - Structured solver lifecycle events and event bus (`EventManager`)
//! - Execution trace logger with configurable formats (`EventLogger`, `SolverTrace`)
//! - Resource governance and termination management (`ResourceGovernor`)
//! - Runtime configuration abstraction (`RuntimeConfig`)
//! - Stage-by-stage monotonic performance profiling (`StageProfiler`)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use yukti_model::{LogFormat, SolverError, SolverStatus};

/// Structured lifecycle event emitted during optimization.
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
    AlgorithmDispatched {
        algorithm: String,
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
    IpmIteration {
        iter: usize,
        mu: f64,
        primal_res: f64,
        dual_res: f64,
        objective: f64,
    },
    NumericalWarning {
        warning: String,
    },
    FallbackTriggered {
        from: String,
        to: String,
        reason: String,
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

/// Audit log of events emitted during a solve.
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

/// Global/local trace sink with configurable formatting.
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
            SolverEvent::AlgorithmDispatched { algorithm } => {
                println!("  [ALGORITHM]   Routing to: {algorithm}");
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
            SolverEvent::IpmIteration {
                iter,
                mu,
                primal_res,
                dual_res,
                objective,
            } => {
                println!(
                    "  [IPM]         Iter {iter:>3} | Obj: {objective:>12.6} | Mu: {mu:>8.2e} | Pres: {primal_res:>9.2e} | Dres: {dual_res:>9.2e}"
                );
            }
            SolverEvent::NumericalWarning { warning } => {
                println!("  [WARNING]     {warning}");
            }
            SolverEvent::FallbackTriggered { from, to, reason } => {
                println!("  [FALLBACK]    Switched from {from} to {to}: {reason}");
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

pub type EventSubscriber = Box<dyn Fn(&SolverEvent) + Send + Sync>;

/// Centralized Event Manager supporting multiple dynamic subscribers and formatted logging.
pub struct EventManager {
    logger: EventLogger,
    subscribers: Mutex<Vec<EventSubscriber>>,
}

impl EventManager {
    pub fn new(format: LogFormat) -> Self {
        Self {
            logger: EventLogger::new(format),
            subscribers: Mutex::new(Vec::new()),
        }
    }

    pub fn subscribe<F>(&self, callback: F)
    where
        F: Fn(&SolverEvent) + Send + Sync + 'static,
    {
        if let Ok(mut subs) = self.subscribers.lock() {
            subs.push(Box::new(callback));
        }
    }

    pub fn publish(&self, event: SolverEvent) {
        // Dispatch to internal logger
        self.logger.emit(event.clone());

        // Dispatch to registered subscribers
        if let Ok(subs) = self.subscribers.lock() {
            for sub in subs.iter() {
                sub(&event);
            }
        }
    }

    pub fn get_trace(&self) -> SolverTrace {
        self.logger.get_trace()
    }
}

/// Resource and Termination Governor enforcing time limits, iteration budgets, and cancellation.
#[derive(Debug, Clone)]
pub struct ResourceGovernor {
    start_time: Instant,
    time_limit_secs: f64,
    max_iterations: usize,
    stop_signal: Arc<AtomicBool>,
}

impl ResourceGovernor {
    pub fn new(time_limit_secs: f64, max_iterations: usize) -> Self {
        Self {
            start_time: Instant::now(),
            time_limit_secs,
            max_iterations,
            stop_signal: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn stop_signal(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop_signal)
    }

    pub fn request_stop(&self) {
        self.stop_signal.store(true, Ordering::SeqCst);
    }

    pub fn is_stopped(&self) -> bool {
        self.stop_signal.load(Ordering::SeqCst)
    }

    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    pub fn check_limits(&self, current_iter: usize) -> Result<(), SolverError> {
        if self.is_stopped() {
            return Err(SolverError::NumericalError(
                "Execution aborted by stop signal".to_string(),
            ));
        }
        if self.time_limit_secs > 0.0 && self.elapsed_secs() > self.time_limit_secs {
            return Err(SolverError::NumericalError(format!(
                "Time limit exceeded: {:.2}s > {:.2}s",
                self.elapsed_secs(),
                self.time_limit_secs
            )));
        }
        if self.max_iterations > 0 && current_iter >= self.max_iterations {
            return Err(SolverError::NumericalError(format!(
                "Iteration limit reached: {} >= {}",
                current_iter, self.max_iterations
            )));
        }
        Ok(())
    }
}

/// Centralized runtime configuration for solver execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub algorithm: String,
    pub backend: String,
    pub tolerance: f64,
    pub time_limit_secs: f64,
    pub max_iterations: usize,
    pub log_format: LogFormat,
    pub enable_presolve: bool,
    pub enable_fallback: bool,
    pub enable_verification: bool,
    pub num_threads: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            algorithm: "auto".to_string(),
            backend: "auto".to_string(),
            tolerance: 1e-6,
            time_limit_secs: 120.0,
            max_iterations: 50_000,
            log_format: LogFormat::Human,
            enable_presolve: true,
            enable_fallback: true,
            enable_verification: true,
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        }
    }
}

/// Stage-by-stage monotonic performance profiling tracker.
#[derive(Debug, Default)]
pub struct StageProfiler {
    active_stages: Mutex<HashMap<String, Instant>>,
    completed_durations: Mutex<HashMap<String, f64>>,
}

impl StageProfiler {
    pub fn new() -> Self {
        Self {
            active_stages: Mutex::new(HashMap::new()),
            completed_durations: Mutex::new(HashMap::new()),
        }
    }

    pub fn start_stage(&self, name: &str) {
        if let Ok(mut active) = self.active_stages.lock() {
            active.insert(name.to_string(), Instant::now());
        }
    }

    pub fn stop_stage(&self, name: &str) -> f64 {
        let dur = if let Ok(mut active) = self.active_stages.lock() {
            active
                .remove(name)
                .map(|s| s.elapsed().as_secs_f64())
                .unwrap_or(0.0)
        } else {
            0.0
        };

        if let Ok(mut completed) = self.completed_durations.lock() {
            *completed.entry(name.to_string()).or_insert(0.0) += dur;
        }
        dur
    }

    pub fn get_durations(&self) -> HashMap<String, f64> {
        self.completed_durations
            .lock()
            .map(|m| m.clone())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_resource_governor_iteration_and_time() {
        let gov = ResourceGovernor::new(10.0, 100);
        assert!(gov.check_limits(50).is_ok());
        assert!(gov.check_limits(100).is_err());

        let timeout_gov = ResourceGovernor::new(0.001, 1000);
        std::thread::sleep(std::time::Duration::from_millis(10));
        assert!(timeout_gov.check_limits(1).is_err());
    }

    #[test]
    fn test_event_manager_subscribers() {
        let manager = EventManager::new(LogFormat::Quiet);
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = Arc::clone(&count);

        manager.subscribe(move |_| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });

        manager.publish(SolverEvent::ModelValidated {
            status: "OK".to_string(),
        });
        manager.publish(SolverEvent::NumericalWarning {
            warning: "Test".to_string(),
        });

        assert_eq!(count.load(Ordering::SeqCst), 2);
        assert_eq!(manager.get_trace().events.len(), 2);
    }

    #[test]
    fn test_stage_profiler() {
        let profiler = StageProfiler::new();
        profiler.start_stage("presolve");
        std::thread::sleep(std::time::Duration::from_millis(5));
        let dur = profiler.stop_stage("presolve");
        assert!(dur > 0.0);
        let durations = profiler.get_durations();
        assert!(durations.contains_key("presolve"));
    }
}
