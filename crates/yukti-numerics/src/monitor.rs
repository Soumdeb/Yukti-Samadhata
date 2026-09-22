use crate::restart::{RestartAction, RestartManager, RestartReason};
use crate::stagnation::StagnationDetector;
use crate::tolerances::TolerancePolicy;
use yukti_sparse::norm_l2;

/// Diagnostic metrics recorded by NumericalMonitor on a checkpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct MonitorMetrics {
    pub iteration: usize,
    pub primal_rel: f64,
    pub dual_rel: f64,
    pub gap_rel: f64,
    pub objective: f64,
    pub objective_change: f64,
    pub primal_step_norm: f64,
    pub dual_step_norm: f64,
}

/// Numerical health status verdict emitted by NumericalMonitor.
#[derive(Debug, Clone, PartialEq)]
pub enum MonitorVerdict {
    Healthy(MonitorMetrics),
    RestartToAverage {
        metrics: MonitorMetrics,
        reason: RestartReason,
    },
    RecoverToBest {
        metrics: MonitorMetrics,
        best_iteration: usize,
        best_err: f64,
        best_x: Vec<f64>,
        best_y: Vec<f64>,
    },
    StagnationExhausted(MonitorMetrics),
    NonFiniteDetected {
        target: String,
        index: usize,
        value: f64,
    },
}

/// Integrated numerical supervisor monitoring iterate health, residuals, step norms, and stagnation.
pub struct NumericalMonitor {
    pub tolerance_policy: TolerancePolicy,
    pub stagnation_detector: StagnationDetector,
    pub restart_manager: RestartManager,
    prev_obj: Option<f64>,
    prev_x: Vec<f64>,
    prev_y: Vec<f64>,
}

impl NumericalMonitor {
    pub fn new(
        tolerance_policy: TolerancePolicy,
        num_vars: usize,
        num_cons: usize,
        initial_x: &[f64],
        initial_y: &[f64],
    ) -> Self {
        let window = tolerance_policy.tolerances.stagnation_window;
        let threshold = tolerance_policy.tolerances.stagnation_threshold;
        let stagnation_detector = StagnationDetector::new(window, threshold);
        let restart_manager = RestartManager::new(num_vars, num_cons, 3);

        Self {
            tolerance_policy,
            stagnation_detector,
            restart_manager,
            prev_obj: None,
            prev_x: initial_x.to_vec(),
            prev_y: initial_y.to_vec(),
        }
    }

    /// Supervise the current solver step and evaluate numerical health.
    pub fn check_step(
        &mut self,
        iteration: usize,
        x: &[f64],
        y: &[f64],
        residuals: (f64, f64, f64),
        obj: f64,
    ) -> MonitorVerdict {
        let (primal_rel, dual_rel, gap_rel) = residuals;
        // 1. Sentinel NaN/Inf check on primal variables
        for (j, &val) in x.iter().enumerate() {
            if !val.is_finite() {
                return MonitorVerdict::NonFiniteDetected {
                    target: "primal_iterate_x".into(),
                    index: j,
                    value: val,
                };
            }
        }

        // 2. Sentinel NaN/Inf check on dual multipliers
        for (i, &val) in y.iter().enumerate() {
            if !val.is_finite() {
                return MonitorVerdict::NonFiniteDetected {
                    target: "dual_iterate_y".into(),
                    index: i,
                    value: val,
                };
            }
        }

        // 3. Sentinel check on residuals
        if !primal_rel.is_finite()
            || !dual_rel.is_finite()
            || !gap_rel.is_finite()
            || !obj.is_finite()
        {
            return MonitorVerdict::NonFiniteDetected {
                target: "residuals_or_objective".into(),
                index: 0,
                value: f64::NAN,
            };
        }

        // 4. Compute step norms
        let mut diff_x = vec![0.0; x.len()];
        for (j, dx) in diff_x.iter_mut().enumerate() {
            let prev = if j < self.prev_x.len() {
                self.prev_x[j]
            } else {
                0.0
            };
            *dx = x[j] - prev;
        }
        let primal_step_norm = norm_l2(&diff_x);

        let mut diff_y = vec![0.0; y.len()];
        for (i, dy) in diff_y.iter_mut().enumerate() {
            let prev = if i < self.prev_y.len() {
                self.prev_y[i]
            } else {
                0.0
            };
            *dy = y[i] - prev;
        }
        let dual_step_norm = norm_l2(&diff_y);

        // 5. Compute objective change
        let objective_change = match self.prev_obj {
            Some(po) => (obj - po).abs(),
            None => 0.0,
        };

        let metrics = MonitorMetrics {
            iteration,
            primal_rel,
            dual_rel,
            gap_rel,
            objective: obj,
            objective_change,
            primal_step_norm,
            dual_step_norm,
        };

        // Update previous vectors
        self.prev_obj = Some(obj);
        self.prev_x.clear();
        self.prev_x.extend_from_slice(x);
        self.prev_y.clear();
        self.prev_y.extend_from_slice(y);

        let current_err = primal_rel.max(dual_rel).max(gap_rel);

        // 6. Checkpoint best known state
        self.restart_manager
            .checkpoint_best(iteration, current_err, x, y);

        // 7. Stagnation detection
        let is_stagnant = self.stagnation_detector.record(
            primal_rel,
            dual_rel,
            gap_rel,
            obj,
            primal_step_norm,
            dual_step_norm,
        );

        // 8. Evaluate restart and recovery action
        let action = self.restart_manager.evaluate(current_err, is_stagnant);

        match action {
            RestartAction::Continue => MonitorVerdict::Healthy(metrics),
            RestartAction::RestartToAverage { reason } => {
                MonitorVerdict::RestartToAverage { metrics, reason }
            }
            RestartAction::RecoverToBest {
                best_iteration,
                best_err,
                reason: _,
            } => {
                self.stagnation_detector.reset_window();
                MonitorVerdict::RecoverToBest {
                    metrics,
                    best_iteration,
                    best_err,
                    best_x: self.restart_manager.best_x.clone(),
                    best_y: self.restart_manager.best_y.clone(),
                }
            }
            RestartAction::ExhaustedStagnation => MonitorVerdict::StagnationExhausted(metrics),
        }
    }
}
