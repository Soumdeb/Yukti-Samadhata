use serde::{Deserialize, Serialize};

/// Reason for triggering a PDHG restart or recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RestartReason {
    None,
    ProgressAcceleration,
    StagnationRecovery,
    DivergenceRecovery,
}

/// Action to be taken by the solver after restart evaluation.
#[derive(Debug, Clone, PartialEq)]
pub enum RestartAction {
    Continue,
    RestartToAverage {
        reason: RestartReason,
    },
    RecoverToBest {
        best_iteration: usize,
        best_err: f64,
        reason: RestartReason,
    },
    ExhaustedStagnation,
}

/// Safe restart and recovery manager for PDHG / PDLP.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RestartManager {
    pub best_err: f64,
    pub best_iteration: usize,
    pub best_x: Vec<f64>,
    pub best_y: Vec<f64>,
    pub last_restart_err: f64,
    pub routine_restarts: usize,
    pub recovery_restarts: usize,
    pub max_recovery_restarts: usize,
    pub progress_factor: f64, // default 0.8 (20% reduction)
}

impl RestartManager {
    pub fn new(num_vars: usize, num_cons: usize, max_recoveries: usize) -> Self {
        Self {
            best_err: f64::INFINITY,
            best_iteration: 0,
            best_x: vec![0.0; num_vars],
            best_y: vec![0.0; num_cons],
            last_restart_err: f64::INFINITY,
            routine_restarts: 0,
            recovery_restarts: 0,
            max_recovery_restarts: max_recoveries.max(1),
            progress_factor: 0.8,
        }
    }

    /// Checkpoint the current iterate as candidate best if error improved.
    pub fn checkpoint_best(&mut self, iteration: usize, err: f64, x: &[f64], y: &[f64]) {
        if err < self.best_err && err.is_finite() {
            self.best_err = err;
            self.best_iteration = iteration;
            self.best_x.clear();
            self.best_x.extend_from_slice(x);
            self.best_y.clear();
            self.best_y.extend_from_slice(y);
        }
    }

    /// Set baseline error from initial problem state.
    pub fn set_baseline(&mut self, initial_err: f64) {
        if initial_err.is_finite() {
            self.last_restart_err = initial_err;
            self.best_err = initial_err;
        }
    }

    /// Evaluate if an adaptive routine restart or safe recovery restart should be triggered.
    pub fn evaluate(&mut self, current_err: f64, is_stagnant: bool) -> RestartAction {
        if is_stagnant {
            if self.recovery_restarts < self.max_recovery_restarts {
                self.recovery_restarts += 1;
                self.last_restart_err = self.best_err;
                return RestartAction::RecoverToBest {
                    best_iteration: self.best_iteration,
                    best_err: self.best_err,
                    reason: RestartReason::StagnationRecovery,
                };
            } else {
                return RestartAction::ExhaustedStagnation;
            }
        }

        // Routine adaptive restart on progress (only if baseline has been established)
        if self.last_restart_err.is_finite() {
            if current_err <= self.progress_factor * self.last_restart_err {
                self.routine_restarts += 1;
                self.last_restart_err = current_err;
                return RestartAction::RestartToAverage {
                    reason: RestartReason::ProgressAcceleration,
                };
            }
        } else {
            self.last_restart_err = current_err;
        }

        RestartAction::Continue
    }
}
