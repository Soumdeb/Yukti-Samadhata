use serde::{Deserialize, Serialize};

/// Stagnation detector monitoring progress over a sliding iteration window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StagnationDetector {
    pub window_size: usize,
    pub threshold: f64,
    pub min_step_norm: f64,
    pub error_history: Vec<f64>,
    pub obj_history: Vec<f64>,
    pub primal_step_history: Vec<f64>,
    pub dual_step_history: Vec<f64>,
    pub consecutive_stagnations: usize,
    pub max_consecutive_stagnations: usize,
}

impl StagnationDetector {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        let w = window_size.max(5);
        Self {
            window_size: w,
            threshold,
            min_step_norm: 1e-12,
            error_history: Vec::with_capacity(w),
            obj_history: Vec::with_capacity(w),
            primal_step_history: Vec::with_capacity(w),
            dual_step_history: Vec::with_capacity(w),
            consecutive_stagnations: 0,
            max_consecutive_stagnations: 3,
        }
    }

    /// Record iteration progress metrics.
    /// Returns true if stagnation is detected on this checkpoint.
    pub fn record(
        &mut self,
        primal_rel: f64,
        dual_rel: f64,
        gap_rel: f64,
        obj: f64,
        primal_step_norm: f64,
        dual_step_norm: f64,
    ) -> bool {
        let current_err = primal_rel.max(dual_rel).max(gap_rel);

        self.error_history.push(current_err);
        self.obj_history.push(obj);
        self.primal_step_history.push(primal_step_norm);
        self.dual_step_history.push(dual_step_norm);

        if self.error_history.len() > self.window_size {
            self.error_history.remove(0);
            self.obj_history.remove(0);
            self.primal_step_history.remove(0);
            self.dual_step_history.remove(0);
        }

        if self.error_history.len() < self.window_size {
            return false;
        }

        let old_err = self.error_history[0];
        let rel_err_impr = if old_err > 1e-14 {
            (old_err - current_err) / old_err
        } else {
            0.0
        };

        let old_obj = self.obj_history[0];
        let rel_obj_change = (obj - old_obj).abs() / (1.0 + obj.abs());

        // Vanishing step norm check (steps virtually zero but residual not converged)
        let vanishing_steps =
            primal_step_norm < self.min_step_norm && dual_step_norm < self.min_step_norm;

        let is_stagnant = (rel_err_impr < self.threshold && rel_obj_change < self.threshold)
            || (vanishing_steps && current_err > 1e-4);

        if is_stagnant {
            self.consecutive_stagnations += 1;
        } else if rel_err_impr >= self.threshold {
            self.consecutive_stagnations = 0;
        }

        is_stagnant
    }

    /// Check if consecutive stagnation episodes have exhausted recovery limits.
    pub fn is_exhausted(&self) -> bool {
        self.consecutive_stagnations >= self.max_consecutive_stagnations
    }

    /// Reset stagnation history and counters after a recovery restart.
    pub fn reset_window(&mut self) {
        self.error_history.clear();
        self.obj_history.clear();
        self.primal_step_history.clear();
        self.dual_step_history.clear();
    }
}
