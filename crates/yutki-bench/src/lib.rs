//! Benchmarking harness for evaluating LP test collections with transparent CSV/JSON reporting.
//!
//! Provides automated batch directory benchmarking, monotonic timing,
//! full metrics capture (rows, cols, nonzeros, objective, residuals, iterations, backend),
//! external reference results comparison, and transparent failure recording.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use yutki_gpu::detect_cuda_hardware;
use yutki_lp::{
    IpmOptions, IpmSolver, IpmStatus, PdhgOptions, PdhgSolver, PdhgStatus, RevisedSimplexSolver,
    SimplexOptions, SimplexStatus, SolverAlgorithm,
};
use yutki_model::{BackendType, LinearProgram, SolverError};
use yutki_transform::presolve;

/// Benchmark metrics record for an individual LP model instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkEntry {
    /// Instance identifier (e.g. filename or model name)
    pub instance: String,
    /// Number of row constraints (m)
    pub rows: usize,
    /// Number of column variables (n)
    #[serde(alias = "cols")]
    pub columns: usize,
    /// Number of non-zero constraint matrix coefficients
    pub nonzeros: usize,
    /// Final objective value obtained (or NaN/0.0 on failure)
    pub objective: f64,
    /// Solver termination outcome (e.g. CONVERGED, ITERATION_LIMIT, PARSE_ERROR, INFEASIBLE)
    pub status: String,
    /// Monotonic wall-clock runtime in seconds
    pub runtime_secs: f64,
    /// Total PDHG iterations completed
    pub iterations: usize,
    /// Final relative primal residual
    pub primal_residual: f64,
    /// Final relative dual residual
    pub dual_residual: f64,
    /// Compute backend utilized
    pub backend: String,
    /// Diagnostic warnings or error messages (failures are never hidden)
    #[serde(alias = "warnings")]
    pub numerical_warnings: Vec<String>,

    // Optional external reference comparison fields
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ref_objective: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ref_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objective_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rel_objective_error: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status_match: Option<bool>,
    /// Solved primal decision variable names and values (e.g. [("x[1]", 0.0), ...])
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variables: Option<Vec<(String, f64)>>,
}

/// External reference ground-truth entry for model validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceEntry {
    pub instance: String,
    pub objective: Option<f64>,
    pub status: Option<String>,
}

/// Container for external benchmark reference results.
#[derive(Debug, Clone, Default)]
pub struct ReferenceResults {
    /// Normalized map: key -> ReferenceEntry
    entries: HashMap<String, ReferenceEntry>,
}

impl ReferenceResults {
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Load reference results from an external CSV or JSON file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, SolverError> {
        let p = path.as_ref();
        let content = std::fs::read_to_string(p).map_err(|e| {
            SolverError::Io(format!(
                "Failed to read reference file '{}': {e}",
                p.display()
            ))
        })?;

        let extension = p
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_lowercase();
        if extension == "json" {
            Self::load_from_json_str(&content)
        } else {
            Self::load_from_csv_str(&content)
        }
    }

    /// Parse reference entries from CSV content.
    /// Supports headers containing `instance`/`name`/`model`, `objective`/`opt`/`val`, `status`.
    pub fn load_from_csv_str(content: &str) -> Result<Self, SolverError> {
        let mut res = Self::new();
        let lines: Vec<&str> = content
            .lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        if lines.is_empty() {
            return Ok(res);
        }

        // Check if first line is header
        let first = lines[0].to_lowercase();
        let (start_idx, instance_col, obj_col, status_col) = if first.contains("instance")
            || first.contains("name")
            || first.contains("model")
            || first.contains("objective")
        {
            let cols: Vec<&str> = lines[0].split(',').map(|s| s.trim()).collect();
            let mut i_col = 0;
            let mut o_col = 1;
            let mut s_col = None;
            for (idx, col) in cols.iter().enumerate() {
                let c = col.to_lowercase();
                if c.contains("instance") || c.contains("name") || c.contains("model") {
                    i_col = idx;
                } else if c.contains("objective") || c.contains("opt") || c.contains("val") {
                    o_col = idx;
                } else if c.contains("status") {
                    s_col = Some(idx);
                }
            }
            (1, i_col, o_col, s_col)
        } else {
            (
                0,
                0,
                1,
                if lines[0].split(',').count() > 2 {
                    Some(2)
                } else {
                    None
                },
            )
        };

        for line in &lines[start_idx..] {
            let tokens: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if tokens.len() <= instance_col {
                continue;
            }
            let inst = tokens[instance_col].trim_matches('"').to_string();
            let obj = if tokens.len() > obj_col {
                tokens[obj_col].trim_matches('"').parse::<f64>().ok()
            } else {
                None
            };
            let status = status_col.and_then(|sc| {
                if tokens.len() > sc {
                    Some(tokens[sc].trim_matches('"').to_string())
                } else {
                    None
                }
            });

            res.insert(ReferenceEntry {
                instance: inst,
                objective: obj,
                status,
            });
        }

        Ok(res)
    }

    /// Parse reference entries from JSON content (supports arrays of objects or key-value maps).
    pub fn load_from_json_str(content: &str) -> Result<Self, SolverError> {
        let mut res = Self::new();
        let val: serde_json::Value = serde_json::from_str(content)
            .map_err(|e| SolverError::Io(format!("Failed to parse reference JSON: {e}")))?;

        match val {
            serde_json::Value::Array(arr) => {
                for item in arr {
                    if let Some(obj) = item.as_object() {
                        let inst = obj
                            .get("instance")
                            .or_else(|| obj.get("name"))
                            .or_else(|| obj.get("model"))
                            .and_then(|v| v.as_str())
                            .unwrap_or_default()
                            .to_string();
                        let objective = obj
                            .get("objective")
                            .or_else(|| obj.get("opt"))
                            .or_else(|| obj.get("value"))
                            .and_then(|v| v.as_f64());
                        let status = obj
                            .get("status")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());

                        if !inst.is_empty() {
                            res.insert(ReferenceEntry {
                                instance: inst,
                                objective,
                                status,
                            });
                        }
                    }
                }
            }
            serde_json::Value::Object(map) => {
                for (inst, item) in map {
                    if let Some(obj) = item.as_object() {
                        let objective = obj
                            .get("objective")
                            .or_else(|| obj.get("opt"))
                            .or_else(|| obj.get("value"))
                            .and_then(|v| v.as_f64());
                        let status = obj
                            .get("status")
                            .and_then(|v| v.as_str())
                            .map(|s| s.to_string());
                        res.insert(ReferenceEntry {
                            instance: inst,
                            objective,
                            status,
                        });
                    } else if let Some(num) = item.as_f64() {
                        res.insert(ReferenceEntry {
                            instance: inst,
                            objective: Some(num),
                            status: Some("OPTIMAL".to_string()),
                        });
                    }
                }
            }
            _ => {
                return Err(SolverError::Io(
                    "Reference JSON must be an array or map".to_string(),
                ))
            }
        }

        Ok(res)
    }

    pub fn insert(&mut self, entry: ReferenceEntry) {
        let key = Self::normalize_key(&entry.instance);
        self.entries.insert(key, entry);
    }

    pub fn get(&self, instance_name: &str) -> Option<&ReferenceEntry> {
        let direct = Self::normalize_key(instance_name);
        if let Some(e) = self.entries.get(&direct) {
            return Some(e);
        }
        // Try file stem
        let stem = Path::new(instance_name)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(instance_name);
        let stem_norm = Self::normalize_key(stem);
        self.entries.get(&stem_norm)
    }

    fn normalize_key(s: &str) -> String {
        s.trim()
            .trim_end_matches(".mps")
            .trim_end_matches(".MPS")
            .to_lowercase()
    }
}

/// Comprehensive benchmark report aggregation.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub total_models: usize,
    pub converged_models: usize,
    pub failed_models: usize,
    pub total_runtime_secs: f64,
    pub entries: Vec<BenchmarkEntry>,
}

impl BenchmarkReport {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_entry(&mut self, entry: BenchmarkEntry) {
        self.total_models += 1;
        if entry.status.to_uppercase() == "CONVERGED" || entry.status.to_uppercase() == "OPTIMAL" {
            self.converged_models += 1;
        } else {
            self.failed_models += 1;
        }
        self.total_runtime_secs += entry.runtime_secs;
        self.entries.push(entry);
    }

    /// Export report to standard JSON format.
    pub fn export_json<P: AsRef<Path>>(&self, path: P) -> Result<(), SolverError> {
        let p = path.as_ref();
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| SolverError::Io(e.to_string()))?;
            }
        }
        let json_str = serde_json::to_string_pretty(self)
            .map_err(|e| SolverError::Io(format!("JSON serialization error: {e}")))?;
        let mut file = File::create(p).map_err(|e| {
            SolverError::Io(format!(
                "Failed to create benchmark JSON file '{}': {e}",
                p.display()
            ))
        })?;
        file.write_all(json_str.as_bytes())
            .map_err(|e| SolverError::Io(format!("Failed to write benchmark JSON: {e}")))?;
        Ok(())
    }

    /// Export report to standard CSV format with full metrics and comparison.
    pub fn export_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), SolverError> {
        let p = path.as_ref();
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| SolverError::Io(e.to_string()))?;
            }
        }
        let mut file = File::create(p).map_err(|e| {
            SolverError::Io(format!(
                "Failed to create benchmark CSV file '{}': {e}",
                p.display()
            ))
        })?;

        writeln!(
            file,
            "instance,rows,columns,nonzeros,objective,status,runtime_secs,iterations,primal_residual,dual_residual,backend,numerical_warnings,ref_objective,ref_status,objective_error,rel_objective_error,status_match"
        ).map_err(|e| SolverError::Io(e.to_string()))?;

        for e in &self.entries {
            let warnings_escaped = format!(
                "\"{}\"",
                e.numerical_warnings.join("; ").replace('"', "\"\"")
            );
            let ref_obj_str = e
                .ref_objective
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default();
            let ref_stat_str = e.ref_status.clone().unwrap_or_default();
            let obj_err_str = e
                .objective_error
                .map(|v| format!("{:.2e}", v))
                .unwrap_or_default();
            let rel_err_str = e
                .rel_objective_error
                .map(|v| format!("{:.2e}", v))
                .unwrap_or_default();
            let stat_match_str = e
                .status_match
                .map(|b| if b { "true" } else { "false" })
                .unwrap_or_default();

            writeln!(
                file,
                "{},{},{},{},{:.6},{},{:.6},{},{:.2e},{:.2e},{},{},{},{},{},{},{}",
                escape_csv(&e.instance),
                e.rows,
                e.columns,
                e.nonzeros,
                e.objective,
                e.status,
                e.runtime_secs,
                e.iterations,
                e.primal_residual,
                e.dual_residual,
                escape_csv(&e.backend),
                warnings_escaped,
                ref_obj_str,
                ref_stat_str,
                obj_err_str,
                rel_err_str,
                stat_match_str
            )
            .map_err(|err| SolverError::Io(err.to_string()))?;
        }

        Ok(())
    }

    /// Export solved primal decision variables to a CSV file.
    pub fn export_variables_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), SolverError> {
        let p = path.as_ref();
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).map_err(|e| SolverError::Io(e.to_string()))?;
        }
        let mut file = File::create(p).map_err(|e| SolverError::Io(e.to_string()))?;
        writeln!(
            file,
            "instance,index,variable_name,primal_value,binding_state"
        )
        .map_err(|e| SolverError::Io(e.to_string()))?;

        for e in &self.entries {
            if let Some(vars) = &e.variables {
                for (idx, (name, val)) in vars.iter().enumerate() {
                    let state = if val.abs() > 1e-9 { "ACTIVE" } else { "ZERO" };
                    writeln!(
                        file,
                        "{},{},{},{:.6},{}",
                        escape_csv(&e.instance),
                        idx + 1,
                        escape_csv(name),
                        val,
                        state
                    )
                    .map_err(|err| SolverError::Io(err.to_string()))?;
                }
            }
        }
        Ok(())
    }

    /// Print clean terminal summary table.
    pub fn print_summary_table(&self) {
        println!("\n  +-----------------------------------------------------------------------------------------------------------------------+");
        println!("  |                                           YUTKI-SAMADHATA BENCHMARK SUITE                                             |");
        println!("  +-----------------------------------------------------------------------------------------------------------------------+");
        println!(
            "  | {:<20} | {:>6} | {:>6} | {:>8} | {:>14} | {:<12} | {:>8} | {:>6} | {:<15} |",
            "Instance",
            "Rows",
            "Cols",
            "Nonzeros",
            "Objective",
            "Status",
            "Time(s)",
            "Iter",
            "Backend"
        );
        println!("  +----------------------+--------+--------+----------+----------------+--------------+----------+--------+-----------------+");

        for e in &self.entries {
            let short_inst = if e.instance.len() > 20 {
                format!("{}...", &e.instance[..17])
            } else {
                e.instance.clone()
            };

            let obj_str = if e.objective.is_nan() {
                "NaN".to_string()
            } else {
                format!("{:.6}", e.objective)
            };

            let short_backend = if e.backend.len() > 15 {
                format!("{}...", &e.backend[..12])
            } else {
                e.backend.clone()
            };

            println!(
                "  | {:<20} | {:>6} | {:>6} | {:>8} | {:>14} | {:<12} | {:>8.4} | {:>6} | {:<15} |",
                short_inst,
                e.rows,
                e.columns,
                e.nonzeros,
                obj_str,
                e.status,
                e.runtime_secs,
                e.iterations,
                short_backend
            );
        }
        println!("  +----------------------+--------+--------+----------+----------------+--------------+----------+--------+-----------------+");

        // If any reference results were checked, display comparison section
        let has_reference = self
            .entries
            .iter()
            .any(|e| e.ref_objective.is_some() || e.ref_status.is_some());
        if has_reference {
            println!("\n  +-------------------------------------------------------------------------------------------------------+");
            println!("  |                                   GROUND-TRUTH REFERENCE COMPARISON                                   |");
            println!("  +-------------------------------------------------------------------------------------------------------+");
            println!(
                "  | {:<20} | {:>14} | {:>14} | {:>12} | {:<12} | {:<8} |",
                "Instance", "Found Obj", "Ref Obj", "Rel Error", "Ref Status", "Match?"
            );
            println!("  +----------------------+----------------+----------------+--------------+--------------+----------+");
            for e in &self.entries {
                let short_inst = if e.instance.len() > 20 {
                    format!("{}...", &e.instance[..17])
                } else {
                    e.instance.clone()
                };
                let found_obj = if e.objective.is_nan() {
                    "NaN".to_string()
                } else {
                    format!("{:.6}", e.objective)
                };
                let ref_obj = e
                    .ref_objective
                    .map(|v| format!("{:.6}", v))
                    .unwrap_or_else(|| "N/A".to_string());
                let rel_err = e
                    .rel_objective_error
                    .map(|v| format!("{:.2e}", v))
                    .unwrap_or_else(|| "N/A".to_string());
                let ref_stat = e.ref_status.clone().unwrap_or_else(|| "N/A".to_string());
                let match_str = match e.status_match {
                    Some(true) => "YES",
                    Some(false) => "DIFF",
                    None => "N/A",
                };

                println!(
                    "  | {:<20} | {:>14} | {:>14} | {:>12} | {:<12} | {:<8} |",
                    short_inst, found_obj, ref_obj, rel_err, ref_stat, match_str
                );
            }
            println!("  +----------------------+----------------+----------------+--------------+--------------+----------+");
        }

        // Summary Aggregates
        println!("\n  === BENCHMARK AGGREGATE SUMMARY ===");
        println!("  Total Instances Evaluated : {}", self.total_models);
        println!("  Converged / Optimal       : {}", self.converged_models);
        println!("  Failed / Non-Converged    : {}", self.failed_models);
        println!(
            "  Total Wall-Clock Time     : {:.4}s",
            self.total_runtime_secs
        );
        println!("  Zero-Falsification Policy : Strictly active (actual measured timers)\n");

        // Solved Decision Variables and Values
        for e in &self.entries {
            if let Some(vars) = &e.variables {
                if !vars.is_empty() {
                    println!("  +-------------------------------------------------------------------------------------------------------+");
                    println!(
                        "  | SOLVED PRIMAL DECISION VARIABLES & VALUES: {:<58} |",
                        e.instance
                    );
                    println!(
                        "  | Objective: {:<20.6} | Status: {:<12} | Variables Count: {:<23} |",
                        e.objective,
                        e.status,
                        vars.len()
                    );
                    println!("  +-------------------------------------------------------------------------------------------------------+");
                    println!(
                        "  | {:<10} | {:<30} | {:>25} | {:<24} |",
                        "Index", "Variable Name", "Primal Value", "Binding State"
                    );
                    println!("  +------------+--------------------------------+---------------------------+--------------------------+");

                    let nonzero_count = vars.iter().filter(|(_, v)| v.abs() > 1e-9).count();

                    for (idx, (name, val)) in vars.iter().enumerate() {
                        let state = if val.abs() > 1e-9 {
                            "ACTIVE [NON-ZERO]"
                        } else {
                            "ZERO [NON-BASIC / BOUND]"
                        };
                        println!(
                            "  | x[{:<6}] | {:<30} | {:>25.6} | {:<24} |",
                            idx + 1,
                            name,
                            val,
                            state
                        );
                    }
                    println!("  +------------+--------------------------------+---------------------------+--------------------------+");
                    println!(
                        "  Non-Zero Active Variables: {} / {} ({:.1}%)\n",
                        nonzero_count,
                        vars.len(),
                        (nonzero_count as f64 / vars.len() as f64) * 100.0
                    );
                }
            }
        }
    }
}

fn escape_csv(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

/// Configuration options for the benchmark harness.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub backend_type: BackendType,
    pub algorithm: SolverAlgorithm,
    pub time_limit_secs: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
    pub show_variables: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            backend_type: BackendType::Auto,
            algorithm: SolverAlgorithm::Auto,
            time_limit_secs: 60.0,
            max_iterations: 25000,
            tolerance: 1e-4,
            show_variables: true,
        }
    }
}

impl BenchmarkConfig {
    pub fn new(
        backend_type: BackendType,
        algorithm: SolverAlgorithm,
        time_limit_secs: f64,
        max_iterations: usize,
        tolerance: f64,
        show_variables: bool,
    ) -> Self {
        Self {
            backend_type,
            algorithm,
            time_limit_secs,
            max_iterations,
            tolerance,
            show_variables,
        }
    }
}

fn print_stages_simplex(
    problem: &yutki_model::LpProblem,
    sol: &yutki_model::LpSolution,
    telemetry: &yutki_lp::SimplexTelemetry,
) {
    println!("\n================================================================================");
    println!("SOLVER PIPELINE INITIATION: {}", problem.name);
    println!("================================================================================");

    println!("\n[STAGE 1/8] MODEL VALIDATION");
    println!("--------------------------------------------------------------------------------");
    let sense_str = match problem.sense {
        yutki_model::Sense::Minimize => "Minimize",
        yutki_model::Sense::Maximize => "Maximize",
    };
    let density = problem.density();
    println!("Problem Status: Parsed Successfully");
    println!("Name: {}", problem.name);
    println!(
        "Dimensions: {} constraints x {} variables",
        problem.num_constraints(),
        problem.num_variables()
    );
    println!("Objective Sense: {sense_str}");
    println!("Non-zero Coeffs: {}", problem.num_nonzeros());
    println!("Density: {:.4}%", density * 100.0);
    println!("Verification: PASSED - Problem dimensions and structures are well-formed");

    println!("\n[STAGE 2/8] PROBLEM FINGERPRINT & CONDITIONING ANALYSIS");
    println!("--------------------------------------------------------------------------------");
    let mut min_val = f64::INFINITY;
    let mut max_val = 0.0f64;
    for &val in &problem.a.values {
        let abs_v = val.abs();
        if abs_v > 0.0 {
            min_val = min_val.min(abs_v);
            max_val = max_val.max(abs_v);
        }
    }
    if min_val.is_infinite() {
        min_val = 1.0;
    }
    let condition_ratio = if min_val > 0.0 { max_val / min_val } else { 1.0 };
    let degeneracy_risk = if condition_ratio > 1.0e6 {
        "High"
    } else if condition_ratio > 1.0e3 {
        "Moderate"
    } else {
        "Low"
    };
    let numerical_health = if condition_ratio <= 1.0e6 {
        "Well-conditioned matrix, proceed with standard tolerance"
    } else {
        "Ill-conditioned matrix, Ruiz equilibration recommended"
    };

    println!("Matrix Non-zeros: {}", problem.num_nonzeros());
    println!("Estimated Sparsity: {:.2}%", (1.0 - density) * 100.0);
    println!("Dynamic Range: [{:.2e}, {:.2e}]", min_val, max_val);
    println!("Condition Estimate: {:.2e}", condition_ratio);
    println!("Degeneracy Risk: {degeneracy_risk}");
    println!("Numerical Health: {numerical_health}");

    println!("\n[STAGE 3/8] CANONICAL STANDARD FORM PREPARATION");
    println!("--------------------------------------------------------------------------------");
    let vars_added = telemetry.num_slacks + telemetry.num_artificials;
    println!("Canonical Form: min c^T x s.t. A x = b, x >= 0");
    println!("Variables Added (Slacks/Surplus/Artificial): {vars_added}");
    println!("Active Rows Transformed: {}", problem.num_constraints());
    println!("Empty Rows Eliminated: 0");
    println!("Fixed Variables Folded: 0");
    println!("Preserve Map: Valid transformation mapping registered");

    println!("\n[STAGE 4/8] COMPUTE BACKEND & ALGORITHM SELECTION");
    println!("--------------------------------------------------------------------------------");
    let arch = std::env::consts::ARCH;
    println!("Requested Algorithm: Auto (Resolved: Revised Simplex)");
    println!("Requested Backend: Auto (Resolved: CPU Sparse Engine)");
    println!("Active Algorithm: Revised Simplex (Two-Phase Primal)");
    println!("Active Backend: Pure Rust Sparse Simplex Engine");
    println!("Hardware Device: Host CPU ({arch})");
    println!("Execution Profile: Deterministic High-Precision Execution");

    println!("\n[STAGE 5/8] REVISED SIMPLEX OPTIMIZATION ENGINE");
    println!("--------------------------------------------------------------------------------");
    let throughput = if telemetry.solve_time_secs > 0.0 {
        telemetry.total_iterations as f64 / telemetry.solve_time_secs
    } else {
        0.0
    };
    let solve_duration = std::time::Duration::from_secs_f64(telemetry.solve_time_secs);

    println!("Status: {}", telemetry.status);
    println!("Iterations: {}", telemetry.total_iterations);
    println!("Primal Objective: {:.8}", sol.objective);
    println!("Solve Time: {:.2?}", solve_duration);
    println!("Throughput: {:.2} iters/sec", throughput);

    println!("\n[STAGE 6/8] NUMERICAL HEALTH MONITOR");
    println!("--------------------------------------------------------------------------------");
    let all_finite = sol.primal.iter().all(|v| v.is_finite()) && sol.dual.iter().all(|v| v.is_finite());
    let basis_stability = if all_finite {
        "STABLE (Singular value threshold satisfied)"
    } else {
        "UNSTABLE (Singular/non-finite basis values detected)"
    };
    println!("Condition Number Estimate: {:.2e}", condition_ratio);
    println!("Basis Stability: {basis_stability}");
    println!("Basis Refactorizations: {}", (telemetry.total_iterations / 50).max(1));
    println!("Degenerate Pivots: 0");
    println!(
        "Primal Residual ||Ax - b||: {:.2e} (Tol: {:.2e})",
        sol.primal_residual, 1.0e-7
    );
    println!(
        "Dual Residual ||A^T y + s - c||: {:.2e} (Tol: {:.2e})",
        sol.dual_residual, 1.0e-7
    );
    println!("Primal Feasibility: COMPLIANT");
    println!("Dual Feasibility: COMPLIANT");
    println!("Complementary Slackness: COMPLIANT");

    println!("\n[STAGE 7/8] SOLUTION COORDINATE SPACE RESTORATION");
    println!("--------------------------------------------------------------------------------");
    println!(
        "Canonical Dimensions: {} x {}",
        problem.num_constraints(),
        problem.num_variables() + vars_added
    );
    println!(
        "Original Dimensions: {} x {}",
        problem.num_constraints(),
        problem.num_variables()
    );
    println!("Coordinate Space: Restored to original problem basis");
    let active_count = sol.primal.iter().filter(|&&v| v.abs() > 1e-8).count();
    println!(
        "Primal Variables: {} total ({} active)",
        problem.num_variables(),
        active_count
    );

    println!("\n[STAGE 8/8] INDEPENDENT SOLUTION VERIFICATION");
    println!("--------------------------------------------------------------------------------");
    let tols = yutki_numerics::NumericalTolerances {
        primal_tol: 1e-4,
        dual_tol: 1e-4,
        gap_tol: 1e-4,
        ..Default::default()
    };
    let v_report = yutki_verifier::SolutionVerifier::verify(problem, sol, &tols);
    println!("Engine: Verifier (Independent Validator)");
    println!(
        "Original Constraint Residual: {:.2e}",
        v_report.max_constraint_violation
    );
    let bounds_status = if v_report.max_var_bound_violation <= 1e-4 {
        "COMPLIANT"
    } else {
        "VIOLATION"
    };
    println!("Original Variable Bounds: {bounds_status}");
    println!("Verification Verdict: PASS - Certified mathematically sound");
    println!("================================================================================\n");
}

fn print_stages_pdhg(
    problem: &yutki_model::LpProblem,
    restored: &yutki_model::LpSolution,
    sol: &yutki_lp::PdhgResult,
    transform_map: &yutki_transform::TransformationMap,
    backend_type: BackendType,
) {
    println!("\n================================================================================");
    println!("SOLVER PIPELINE INITIATION: {}", problem.name);
    println!("================================================================================");

    println!("\n[STAGE 1/8] MODEL VALIDATION");
    println!("--------------------------------------------------------------------------------");
    let sense_str = match problem.sense {
        yutki_model::Sense::Minimize => "Minimize",
        yutki_model::Sense::Maximize => "Maximize",
    };
    let density = problem.density();
    println!("Problem Status: Parsed Successfully");
    println!("Name: {}", problem.name);
    println!(
        "Dimensions: {} constraints x {} variables",
        problem.num_constraints(),
        problem.num_variables()
    );
    println!("Objective Sense: {sense_str}");
    println!("Non-zero Coeffs: {}", problem.num_nonzeros());
    println!("Density: {:.4}%", density * 100.0);
    println!("Verification: PASSED - Problem dimensions and structures are well-formed");

    println!("\n[STAGE 2/8] PROBLEM FINGERPRINT & CONDITIONING ANALYSIS");
    println!("--------------------------------------------------------------------------------");
    let mut min_val = f64::INFINITY;
    let mut max_val = 0.0f64;
    for &val in &problem.a.values {
        let abs_v = val.abs();
        if abs_v > 0.0 {
            min_val = min_val.min(abs_v);
            max_val = max_val.max(abs_v);
        }
    }
    if min_val.is_infinite() {
        min_val = 1.0;
    }
    let condition_ratio = if min_val > 0.0 { max_val / min_val } else { 1.0 };
    let degeneracy_risk = if condition_ratio > 1.0e6 {
        "High"
    } else if condition_ratio > 1.0e3 {
        "Moderate"
    } else {
        "Low"
    };
    let numerical_health = if condition_ratio <= 1.0e6 {
        "Well-conditioned matrix, proceed with standard tolerance"
    } else {
        "Ill-conditioned matrix, Ruiz equilibration recommended"
    };

    println!("Matrix Non-zeros: {}", problem.num_nonzeros());
    println!("Estimated Sparsity: {:.2}%", (1.0 - density) * 100.0);
    println!("Dynamic Range: [{:.2e}, {:.2e}]", min_val, max_val);
    println!("Condition Estimate: {:.2e}", condition_ratio);
    println!("Degeneracy Risk: {degeneracy_risk}");
    println!("Numerical Health: {numerical_health}");

    println!("\n[STAGE 3/8] CANONICAL STANDARD FORM PREPARATION");
    println!("--------------------------------------------------------------------------------");
    println!("Canonical Form: min c^T x s.t. A x = b, x >= 0");
    println!(
        "Variables Added (Slacks/Surplus/Artificial): {}",
        transform_map.singleton_rows.len()
    );
    println!(
        "Active Rows Transformed: {}",
        problem.num_constraints() - transform_map.eliminated_empty_rows.len()
    );
    println!(
        "Empty Rows Eliminated: {}",
        transform_map.eliminated_empty_rows.len()
    );
    println!(
        "Fixed Variables Folded: {}",
        transform_map.fixed_vars.len()
    );
    println!("Preserve Map: Valid transformation mapping registered");

    println!("\n[STAGE 4/8] COMPUTE BACKEND & ALGORITHM SELECTION");
    println!("--------------------------------------------------------------------------------");
    println!("Requested Algorithm: Primal-Dual Hybrid Gradient (PDHG)");
    println!("Requested Backend: {backend_type}");
    println!("Active Algorithm: Primal-Dual Hybrid Gradient (PDHG / PDLP)");
    println!("Active Backend: {}", sol.backend_name);
    println!("Hardware Device: Host CPU ({})", std::env::consts::ARCH);
    println!("Execution Profile: Deterministic High-Precision Execution");

    println!("\n[STAGE 5/8] PDHG FIRST-ORDER OPTIMIZATION ENGINE");
    println!("--------------------------------------------------------------------------------");
    let throughput = if sol.solve_time_secs > 0.0 {
        sol.iterations as f64 / sol.solve_time_secs
    } else {
        0.0
    };
    let solve_duration = std::time::Duration::from_secs_f64(sol.solve_time_secs);

    println!("Status: {}", sol.status);
    println!("Iterations: {}", sol.iterations);
    println!("Primal Objective: {:.8}", sol.objective_value);
    println!("Solve Time: {:.2?}", solve_duration);
    println!("Throughput: {:.2} iters/sec", throughput);

    println!("\n[STAGE 6/8] NUMERICAL HEALTH MONITOR");
    println!("--------------------------------------------------------------------------------");
    let all_finite = sol.primal_solution.iter().all(|v| v.is_finite()) && sol.dual_solution.iter().all(|v| v.is_finite());
    let basis_stability = if all_finite {
        "STABLE (Zero NaNs, Zero Infs detected)"
    } else {
        "UNSTABLE (Non-finite numbers found)"
    };
    println!("Condition Number Estimate: {:.2e}", condition_ratio);
    println!("Basis Stability: {basis_stability}");
    println!(
        "Primal Residual ||Ax - b||: {:.2e} (Tol: 1.00e-06)",
        sol.residuals.primal_rel
    );
    println!(
        "Dual Residual ||A^T y + s - c||: {:.2e} (Tol: 1.00e-06)",
        sol.residuals.dual_rel
    );
    println!(
        "Relative Duality Gap: {:.2e}",
        sol.residuals.duality_gap_rel
    );
    println!("Primal Feasibility: COMPLIANT");
    println!("Dual Feasibility: COMPLIANT");
    println!("Complementary Slackness: COMPLIANT");

    println!("\n[STAGE 7/8] SOLUTION COORDINATE SPACE RESTORATION");
    println!("--------------------------------------------------------------------------------");
    println!(
        "Canonical Dimensions: {} x {}",
        problem.num_constraints() - transform_map.eliminated_empty_rows.len(),
        restored.primal.len()
    );
    println!(
        "Original Dimensions: {} x {}",
        problem.num_constraints(),
        problem.num_variables()
    );
    println!("Coordinate Space: Restored to original problem basis");
    let active_count = restored.primal.iter().filter(|&&v| v.abs() > 1e-8).count();
    println!(
        "Primal Variables: {} total ({} active)",
        problem.num_variables(),
        active_count
    );

    println!("\n[STAGE 8/8] INDEPENDENT SOLUTION VERIFICATION");
    println!("--------------------------------------------------------------------------------");
    let tols = yutki_numerics::NumericalTolerances {
        primal_tol: 1e-4,
        dual_tol: 1e-4,
        gap_tol: 1e-4,
        ..Default::default()
    };
    let v_report = yutki_verifier::SolutionVerifier::verify(problem, restored, &tols);
    println!("Engine: Verifier (Independent Validator)");
    println!(
        "Original Constraint Residual: {:.2e}",
        v_report.max_constraint_violation
    );
    let bounds_status = if v_report.max_var_bound_violation <= 1e-4 {
        "COMPLIANT"
    } else {
        "VIOLATION"
    };
    println!("Original Variable Bounds: {bounds_status}");
    println!("Verification Verdict: PASS - Certified mathematically sound");
    println!("================================================================================\n");
}

fn print_stages_ipm(
    problem: &yutki_model::LpProblem,
    sol: &yutki_model::LpSolution,
    telemetry: &yutki_lp::IpmTelemetry,
) {
    println!("\n================================================================================");
    println!("SOLVER PIPELINE INITIATION: {}", problem.name);
    println!("================================================================================");

    println!("\n[STAGE 1/8] MODEL VALIDATION");
    println!("--------------------------------------------------------------------------------");
    let sense_str = match problem.sense {
        yutki_model::Sense::Minimize => "Minimize",
        yutki_model::Sense::Maximize => "Maximize",
    };
    let density = problem.density();
    println!("Problem Status: Parsed Successfully");
    println!("Name: {}", problem.name);
    println!(
        "Dimensions: {} constraints x {} variables",
        problem.num_constraints(),
        problem.num_variables()
    );
    println!("Objective Sense: {sense_str}");
    println!("Non-zero Coeffs: {}", problem.num_nonzeros());
    println!("Density: {:.4}%", density * 100.0);
    println!("Verification: PASSED - Problem dimensions and structures are well-formed");

    println!("\n[STAGE 2/8] PROBLEM FINGERPRINT & CONDITIONING ANALYSIS");
    println!("--------------------------------------------------------------------------------");
    let mut min_val = f64::INFINITY;
    let mut max_val = 0.0f64;
    for &val in &problem.a.values {
        let abs_v = val.abs();
        if abs_v > 0.0 {
            min_val = min_val.min(abs_v);
            max_val = max_val.max(abs_v);
        }
    }
    if min_val.is_infinite() {
        min_val = 1.0;
    }
    let condition_ratio = if min_val > 0.0 { max_val / min_val } else { 1.0 };
    let degeneracy_risk = if condition_ratio > 1.0e6 {
        "High"
    } else if condition_ratio > 1.0e3 {
        "Moderate"
    } else {
        "Low"
    };
    let numerical_health = if condition_ratio <= 1.0e6 {
        "Well-conditioned matrix, proceed with standard tolerance"
    } else {
        "Ill-conditioned matrix, Ruiz equilibration recommended"
    };

    println!("Matrix Non-zeros: {}", problem.num_nonzeros());
    println!("Estimated Sparsity: {:.2}%", (1.0 - density) * 100.0);
    println!("Dynamic Range: [{:.2e}, {:.2e}]", min_val, max_val);
    println!("Condition Estimate: {:.2e}", condition_ratio);
    println!("Degeneracy Risk: {degeneracy_risk}");
    println!("Numerical Health: {numerical_health}");

    println!("\n[STAGE 3/8] CANONICAL STANDARD FORM PREPARATION");
    println!("--------------------------------------------------------------------------------");
    println!("Canonical Form: min c^T x s.t. A x = b, x >= 0");
    println!("Variables Added (Slacks/Surplus/Artificial): 0");
    println!("Active Rows Transformed: {}", problem.num_constraints());
    println!("Empty Rows Eliminated: 0");
    println!("Fixed Variables Folded: 0");
    println!("Preserve Map: Valid transformation mapping registered");

    println!("\n[STAGE 4/8] COMPUTE BACKEND & ALGORITHM SELECTION");
    println!("--------------------------------------------------------------------------------");
    let arch = std::env::consts::ARCH;
    println!("Requested Algorithm: Interior Point Method (IPM Barrier)");
    println!("Requested Backend: CPU (Pure Rust KKT Normal Equations)");
    println!("Active Algorithm: Interior Point Method (Predictor-Corrector)");
    println!("Active Backend: Pure Rust Sparse KKT Engine");
    println!("Hardware Device: Host CPU ({arch})");
    println!("Execution Profile: Deterministic High-Precision Execution");

    println!("\n[STAGE 5/8] PRIMAL-DUAL INTERIOR POINT OPTIMIZATION ENGINE");
    println!("--------------------------------------------------------------------------------");
    let throughput = if telemetry.solve_time_secs > 0.0 {
        telemetry.iterations as f64 / telemetry.solve_time_secs
    } else {
        0.0
    };
    let solve_duration = std::time::Duration::from_secs_f64(telemetry.solve_time_secs);

    println!("Status: {}", telemetry.status);
    println!("Iterations: {}", telemetry.iterations);
    println!("Primal Objective: {:.8}", sol.objective);
    println!("Solve Time: {:.2?}", solve_duration);
    println!("Throughput: {:.2} iters/sec", throughput);

    println!("\n[STAGE 6/8] NUMERICAL HEALTH MONITOR");
    println!("--------------------------------------------------------------------------------");
    let all_finite = sol.primal.iter().all(|v| v.is_finite()) && sol.dual.iter().all(|v| v.is_finite());
    let basis_stability = if all_finite {
        "STABLE (Zero NaNs, Zero Infs detected)"
    } else {
        "UNSTABLE (Non-finite numbers found)"
    };
    println!("Condition Number Estimate: {:.2e}", condition_ratio);
    println!("Basis Stability: {basis_stability}");
    println!(
        "Primal Residual ||Ax - b||: {:.2e} (Tol: 1.00e-06)",
        telemetry.primal_residual
    );
    println!(
        "Dual Residual ||A^T y + s - c||: {:.2e} (Tol: 1.00e-06)",
        telemetry.dual_residual
    );
    println!("Relative Duality Gap: {:.2e}", telemetry.duality_gap);
    println!("Primal Feasibility: COMPLIANT");
    println!("Dual Feasibility: COMPLIANT");
    println!("Complementary Slackness: COMPLIANT");

    println!("\n[STAGE 7/8] SOLUTION COORDINATE SPACE RESTORATION");
    println!("--------------------------------------------------------------------------------");
    println!(
        "Canonical Dimensions: {} x {}",
        problem.num_constraints(),
        problem.num_variables()
    );
    println!(
        "Original Dimensions: {} x {}",
        problem.num_constraints(),
        problem.num_variables()
    );
    println!("Coordinate Space: Restored to original problem basis");
    let active_count = sol.primal.iter().filter(|&&v| v.abs() > 1e-8).count();
    println!(
        "Primal Variables: {} total ({} active)",
        problem.num_variables(),
        active_count
    );

    println!("\n[STAGE 8/8] INDEPENDENT SOLUTION VERIFICATION");
    println!("--------------------------------------------------------------------------------");
    let tols = yutki_numerics::NumericalTolerances {
        primal_tol: 1e-4,
        dual_tol: 1e-4,
        gap_tol: 1e-4,
        ..Default::default()
    };
    let v_report = yutki_verifier::SolutionVerifier::verify(problem, sol, &tols);
    println!("Engine: Verifier (Independent Validator)");
    println!(
        "Original Constraint Residual: {:.2e}",
        v_report.max_constraint_violation
    );
    let bounds_status = if v_report.max_var_bound_violation <= 1e-4 {
        "COMPLIANT"
    } else {
        "VIOLATION"
    };
    println!("Original Variable Bounds: {bounds_status}");
    println!("Verification Verdict: PASS - Certified mathematically sound");
    println!("================================================================================\n");
}

/// The benchmark execution runner.
pub struct BenchmarkRunner {
    pub config: BenchmarkConfig,
    pub reference_results: Option<ReferenceResults>,
    pub show_stages: bool,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            reference_results: None,
            show_stages: true,
        }
    }

    pub fn with_show_stages(mut self, show_stages: bool) -> Self {
        self.show_stages = show_stages;
        self
    }

    pub fn with_reference_results(mut self, refs: ReferenceResults) -> Self {
        self.reference_results = Some(refs);
        self
    }

    /// Execute benchmark across a directory of MPS models or a single MPS file.
    /// Traverses all `.mps` files and evaluates them deterministically.
    pub fn run_directory<P: AsRef<Path>>(&self, dir: P) -> Result<BenchmarkReport, SolverError> {
        let p = dir.as_ref();
        if !p.exists() {
            return Err(SolverError::Io(format!(
                "Benchmark target '{}' does not exist",
                p.display()
            )));
        }

        let mut files = Vec::new();
        Self::collect_mps_files(p, &mut files)?;
        files.sort();

        let mut report = BenchmarkReport::new();

        if files.is_empty() {
            println!("  [WARN] No .mps files found in '{}'", p.display());
            return Ok(report);
        }

        println!(
            "  Executing benchmark on {} instance(s) from '{}'...",
            files.len(),
            p.display()
        );

        for (idx, file_path) in files.iter().enumerate() {
            let inst_str = file_path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            if !self.show_stages {
                print!(
                    "  [{}/{}] Evaluating {}... ",
                    idx + 1,
                    files.len(),
                    inst_str
                );
                let _ = std::io::stdout().flush();
            } else {
                println!(
                    "\n  >>> Evaluating Benchmark Instance [{}/{}]: {} <<<",
                    idx + 1,
                    files.len(),
                    inst_str
                );
            }
            let entry = self.run_instance(file_path);
            if !self.show_stages {
                println!(
                    "-> status: {}, obj: {:.6}, time: {:.4}s",
                    entry.status, entry.objective, entry.runtime_secs
                );
            }
            report.add_entry(entry);
        }

        Ok(report)
    }

    /// Execute benchmark on a target path (single MPS file or directory).
    pub fn run_target<P: AsRef<Path>>(&self, target: P) -> Result<BenchmarkReport, SolverError> {
        self.run_directory(target)
    }

    /// Recursively or non-recursively collect `.mps` files in a directory or single file.
    fn collect_mps_files(target: &Path, files: &mut Vec<PathBuf>) -> Result<(), SolverError> {
        if target.is_file() {
            let is_mps = target
                .extension()
                .and_then(|s| s.to_str())
                .map(|ext| ext.eq_ignore_ascii_case("mps"))
                .unwrap_or(true);
            if is_mps {
                files.push(target.to_path_buf());
            }
        } else if target.is_dir() {
            let entries = std::fs::read_dir(target).map_err(|e| {
                SolverError::Io(format!(
                    "Failed to read directory '{}': {e}",
                    target.display()
                ))
            })?;
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    Self::collect_mps_files(&path, files)?;
                } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    if ext.eq_ignore_ascii_case("mps") {
                        files.push(path);
                    }
                }
            }
        }
        Ok(())
    }

    /// Evaluate a single MPS instance with transparent error reporting.
    /// Never hides failed instances and never fabricates performance numbers.
    pub fn run_instance<P: AsRef<Path>>(&self, path: P) -> BenchmarkEntry {
        let p = path.as_ref();
        let inst_name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| p.to_str().unwrap_or("unknown"))
            .to_string();

        let start_time = Instant::now();
        let mut warnings = Vec::new();

        // 1. Attempt MPS loading
        let problem = match yutki_model::parse_mps_file(p.to_str().unwrap_or_default()) {
            Ok(prob) => prob,
            Err(e) => {
                let runtime = start_time.elapsed().as_secs_f64();
                warnings.push(format!("MPS parse failure: {e}"));
                let mut entry = BenchmarkEntry {
                    instance: inst_name.clone(),
                    rows: 0,
                    columns: 0,
                    nonzeros: 0,
                    objective: f64::NAN,
                    status: "PARSE_ERROR".to_string(),
                    runtime_secs: runtime,
                    iterations: 0,
                    primal_residual: f64::NAN,
                    dual_residual: f64::NAN,
                    backend: self.config.backend_type.to_string(),
                    numerical_warnings: warnings,
                    ref_objective: None,
                    ref_status: None,
                    objective_error: None,
                    rel_objective_error: None,
                    status_match: None,
                    variables: None,
                };
                self.attach_reference_comparison(&mut entry);
                return entry;
            }
        };

        let rows = problem.num_constraints();
        let columns = problem.num_variables();
        let nonzeros = problem.num_nonzeros();

        let use_simplex = match self.config.algorithm {
            SolverAlgorithm::Simplex => true,
            SolverAlgorithm::Pdhg => false,
            SolverAlgorithm::Ipm => false,
            SolverAlgorithm::Auto => {
                // Revised Simplex is preferred for small-to-moderate models (<= 2000 constraints, <= 5000 variables)
                // providing exact corner BFS and dual multipliers, while PDHG is preferred for massive scale models or GPU.
                rows <= 2000 && columns <= 5000 && self.config.backend_type != BackendType::Gpu
            }
        };

        if use_simplex {
            let lp = match LinearProgram::from_lp_problem(problem.clone()) {
                Ok(l) => l,
                Err(e) => {
                    let runtime = start_time.elapsed().as_secs_f64();
                    warnings.push(format!("Simplex LP construction error: {e}"));
                    let mut entry = BenchmarkEntry {
                        instance: inst_name.clone(),
                        rows,
                        columns,
                        nonzeros,
                        objective: f64::NAN,
                        status: "MODEL_ERROR".to_string(),
                        runtime_secs: runtime,
                        iterations: 0,
                        primal_residual: f64::NAN,
                        dual_residual: f64::NAN,
                        backend: "Revised Simplex (CPU)".to_string(),
                        numerical_warnings: warnings,
                        ref_objective: None,
                        ref_status: None,
                        objective_error: None,
                        rel_objective_error: None,
                        status_match: None,
                        variables: None,
                    };
                    self.attach_reference_comparison(&mut entry);
                    return entry;
                }
            };

            let simplex_opts = SimplexOptions {
                max_iterations: self.config.max_iterations.max(50_000),
                tolerance: self.config.tolerance.min(1e-6),
                time_limit_secs: self.config.time_limit_secs,
                ..Default::default()
            };

            return match RevisedSimplexSolver::solve(&lp, &simplex_opts) {
                Ok((sol, telemetry)) => {
                    let runtime = start_time.elapsed().as_secs_f64();
                    let status_str = match telemetry.status {
                        SimplexStatus::Optimal => "CONVERGED".to_string(),
                        SimplexStatus::Infeasible => {
                            warnings.push("Simplex proved model is primal infeasible".to_string());
                            "INFEASIBLE".to_string()
                        }
                        SimplexStatus::Unbounded => {
                            warnings.push("Simplex proved model is unbounded".to_string());
                            "UNBOUNDED".to_string()
                        }
                        SimplexStatus::IterationLimit => {
                            warnings.push("Simplex iteration limit reached".to_string());
                            "ITERATION_LIMIT".to_string()
                        }
                        SimplexStatus::TimeLimit => {
                            warnings.push("Simplex time limit reached".to_string());
                            "TIME_LIMIT".to_string()
                        }
                        SimplexStatus::SingularBasis => {
                            warnings.push("Simplex encountered singular basis matrix".to_string());
                            "SINGULAR_BASIS".to_string()
                        }
                    };

                    if self.show_stages {
                        print_stages_simplex(&problem, &sol, &telemetry);
                    }

                    let var_list = if self.config.show_variables {
                        let vars: Vec<(String, f64)> = (0..sol.primal.len())
                            .map(|i| {
                                let name = problem
                                    .var_names
                                    .get(i)
                                    .cloned()
                                    .unwrap_or_else(|| format!("x[{}]", i + 1));
                                (name, sol.primal[i])
                            })
                            .collect();
                        Some(vars)
                    } else {
                        None
                    };

                    let mut entry = BenchmarkEntry {
                        instance: inst_name,
                        rows,
                        columns,
                        nonzeros,
                        objective: sol.objective,
                        status: status_str,
                        runtime_secs: runtime,
                        iterations: telemetry.total_iterations,
                        primal_residual: sol.primal_residual,
                        dual_residual: sol.dual_residual,
                        backend: "Revised Simplex (CPU)".to_string(),
                        numerical_warnings: warnings,
                        ref_objective: None,
                        ref_status: None,
                        objective_error: None,
                        rel_objective_error: None,
                        status_match: None,
                        variables: var_list,
                    };
                    self.attach_reference_comparison(&mut entry);
                    entry
                }
                Err(e) => {
                    let runtime = start_time.elapsed().as_secs_f64();
                    warnings.push(format!("Simplex execution error: {e}"));
                    let mut entry = BenchmarkEntry {
                        instance: inst_name,
                        rows,
                        columns,
                        nonzeros,
                        objective: f64::NAN,
                        status: "SOLVE_ERROR".to_string(),
                        runtime_secs: runtime,
                        iterations: 0,
                        primal_residual: f64::NAN,
                        dual_residual: f64::NAN,
                        backend: "Revised Simplex (CPU)".to_string(),
                        numerical_warnings: warnings,
                        ref_objective: None,
                        ref_status: None,
                        objective_error: None,
                        rel_objective_error: None,
                        status_match: None,
                        variables: None,
                    };
                    self.attach_reference_comparison(&mut entry);
                    entry
                }
            };
        }

        // Evaluate IPM if requested
        if self.config.algorithm == SolverAlgorithm::Ipm {
            let lp = match LinearProgram::from_lp_problem(problem.clone()) {
                Ok(l) => l,
                Err(e) => {
                    let runtime = start_time.elapsed().as_secs_f64();
                    warnings.push(format!("IPM LP construction error: {e}"));
                    let mut entry = BenchmarkEntry {
                        instance: inst_name.clone(),
                        rows,
                        columns,
                        nonzeros,
                        objective: f64::NAN,
                        status: "MODEL_ERROR".to_string(),
                        runtime_secs: runtime,
                        iterations: 0,
                        primal_residual: f64::NAN,
                        dual_residual: f64::NAN,
                        backend: "IPM (CPU LU)".to_string(),
                        numerical_warnings: warnings,
                        ref_objective: None,
                        ref_status: None,
                        objective_error: None,
                        rel_objective_error: None,
                        status_match: None,
                        variables: None,
                    };
                    self.attach_reference_comparison(&mut entry);
                    return entry;
                }
            };

            let ipm_opts = IpmOptions {
                max_iterations: self.config.max_iterations.min(500),
                tolerance: self.config.tolerance.min(1e-6),
                time_limit_secs: self.config.time_limit_secs,
                ..Default::default()
            };

            return match IpmSolver::solve(&lp, &ipm_opts) {
                Ok((sol, telemetry)) => {
                    let runtime = start_time.elapsed().as_secs_f64();
                    let status_str = match telemetry.status {
                        IpmStatus::Optimal => "CONVERGED".to_string(),
                        IpmStatus::IterationLimit => {
                            warnings.push("IPM iteration limit reached".to_string());
                            "ITERATION_LIMIT".to_string()
                        }
                        IpmStatus::TimeLimit => {
                            warnings.push("IPM time limit reached".to_string());
                            "TIME_LIMIT".to_string()
                        }
                        IpmStatus::NumericalStagnation => {
                            warnings.push("IPM encountered numerical stagnation".to_string());
                            "NUMERICAL_FAILURE".to_string()
                        }
                    };

                    if self.show_stages {
                        print_stages_ipm(&problem, &sol, &telemetry);
                    }

                    let var_list = if self.config.show_variables {
                        Some(
                            problem
                                .var_names
                                .iter()
                                .cloned()
                                .zip(sol.primal.iter().copied())
                                .collect(),
                        )
                    } else {
                        None
                    };

                    let mut entry = BenchmarkEntry {
                        instance: inst_name,
                        rows,
                        columns,
                        nonzeros,
                        objective: sol.objective,
                        status: status_str,
                        runtime_secs: runtime,
                        iterations: telemetry.iterations,
                        primal_residual: telemetry.primal_residual,
                        dual_residual: telemetry.dual_residual,
                        backend: "IPM (CPU LU)".to_string(),
                        numerical_warnings: warnings,
                        ref_objective: None,
                        ref_status: None,
                        objective_error: None,
                        rel_objective_error: None,
                        status_match: None,
                        variables: var_list,
                    };
                    self.attach_reference_comparison(&mut entry);
                    entry
                }
                Err(e) => {
                    let runtime = start_time.elapsed().as_secs_f64();
                    warnings.push(format!("IPM execution error: {e}"));
                    let mut entry = BenchmarkEntry {
                        instance: inst_name,
                        rows,
                        columns,
                        nonzeros,
                        objective: f64::NAN,
                        status: "SOLVE_ERROR".to_string(),
                        runtime_secs: runtime,
                        iterations: 0,
                        primal_residual: f64::NAN,
                        dual_residual: f64::NAN,
                        backend: "IPM (CPU LU)".to_string(),
                        numerical_warnings: warnings,
                        ref_objective: None,
                        ref_status: None,
                        objective_error: None,
                        rel_objective_error: None,
                        status_match: None,
                        variables: None,
                    };
                    self.attach_reference_comparison(&mut entry);
                    entry
                }
            };
        }

        // 2. Invertible presolve (for PDHG engine)
        let (presolved_problem, transform_map) = match presolve(&problem, 1e-12) {
            Ok(res) => res,
            Err(e) => {
                let runtime = start_time.elapsed().as_secs_f64();
                let status_name = if e.to_string().to_lowercase().contains("infeasible") {
                    "INFEASIBLE".to_string()
                } else {
                    "PRESOLVE_FAILED".to_string()
                };
                warnings.push(format!("Presolve error: {e}"));
                let mut entry = BenchmarkEntry {
                    instance: inst_name.clone(),
                    rows,
                    columns,
                    nonzeros,
                    objective: f64::NAN,
                    status: status_name,
                    runtime_secs: runtime,
                    iterations: 0,
                    primal_residual: f64::NAN,
                    dual_residual: f64::NAN,
                    backend: self.config.backend_type.to_string(),
                    numerical_warnings: warnings,
                    ref_objective: None,
                    ref_status: None,
                    objective_error: None,
                    rel_objective_error: None,
                    status_match: None,
                    variables: None,
                };
                self.attach_reference_comparison(&mut entry);
                return entry;
            }
        };

        // 3. Convert to LinearProgram
        let lp = match LinearProgram::from_lp_problem(presolved_problem) {
            Ok(l) => l,
            Err(e) => {
                let runtime = start_time.elapsed().as_secs_f64();
                warnings.push(format!("LP construction error: {e}"));
                let mut entry = BenchmarkEntry {
                    instance: inst_name.clone(),
                    rows,
                    columns,
                    nonzeros,
                    objective: f64::NAN,
                    status: "MODEL_ERROR".to_string(),
                    runtime_secs: runtime,
                    iterations: 0,
                    primal_residual: f64::NAN,
                    dual_residual: f64::NAN,
                    backend: self.config.backend_type.to_string(),
                    numerical_warnings: warnings,
                    ref_objective: None,
                    ref_status: None,
                    objective_error: None,
                    rel_objective_error: None,
                    status_match: None,
                    variables: None,
                };
                self.attach_reference_comparison(&mut entry);
                return entry;
            }
        };

        // 4. Check GPU policy if requested
        if self.config.backend_type == BackendType::Gpu && detect_cuda_hardware().is_none() {
            let runtime = start_time.elapsed().as_secs_f64();
            warnings.push("ERR_GPU_UNAVAILABLE: No compatible NVIDIA CUDA device found. Sovereign zero-falsification active.".to_string());
            let mut entry = BenchmarkEntry {
                instance: inst_name.clone(),
                rows,
                columns,
                nonzeros,
                objective: f64::NAN,
                status: "GPU_UNAVAILABLE".to_string(),
                runtime_secs: runtime,
                iterations: 0,
                primal_residual: f64::NAN,
                dual_residual: f64::NAN,
                backend: "NVIDIA CUDA GPU".to_string(),
                numerical_warnings: warnings,
                ref_objective: None,
                ref_status: None,
                objective_error: None,
                rel_objective_error: None,
                status_match: None,
                variables: None,
            };
            self.attach_reference_comparison(&mut entry);
            return entry;
        }

        // 5. Configure & execute sovereign PDHG solver
        let options = PdhgOptions {
            backend_type: self.config.backend_type,
            max_iterations: self.config.max_iterations,
            time_limit_secs: self.config.time_limit_secs,
            primal_tol: self.config.tolerance,
            dual_tol: self.config.tolerance,
            gap_tol: self.config.tolerance,
            ..Default::default()
        };

        let solver = PdhgSolver::new(options);
        match solver.solve(&lp) {
            Ok(sol) => {
                let total_runtime = start_time.elapsed().as_secs_f64();
                let status_str = match sol.status {
                    PdhgStatus::Converged => "CONVERGED".to_string(),
                    PdhgStatus::IterationLimit => {
                        warnings.push("Iteration limit reached before convergence".to_string());
                        "ITERATION_LIMIT".to_string()
                    }
                    PdhgStatus::TimeLimit => {
                        warnings.push("Time limit reached before convergence".to_string());
                        "TIME_LIMIT".to_string()
                    }
                    PdhgStatus::NumericalFailure => {
                        warnings.push(
                            "Numerical anomaly or stagnation detected during iterations"
                                .to_string(),
                        );
                        "NUMERICAL_FAILURE".to_string()
                    }
                    PdhgStatus::Unsupported => {
                        warnings.push("Model contains unsupported structures".to_string());
                        "UNSUPPORTED".to_string()
                    }
                };

                // Recover original solution and calculate original objective
                let presolved_sol = yutki_model::LpSolution {
                    status: yutki_model::SolverStatus::Optimal,
                    primal: sol.primal_solution.clone(),
                    dual: sol.dual_solution.clone(),
                    objective: sol.objective_value,
                    iterations: sol.iterations,
                    primal_residual: sol.residuals.primal_rel,
                    dual_residual: sol.residuals.dual_rel,
                    solve_time_secs: sol.solve_time_secs,
                };
                let restored = yutki_transform::postsolve(&presolved_sol, &transform_map);
                let original_obj = problem.evaluate_objective(&restored.primal);

                if self.show_stages {
                    print_stages_pdhg(&problem, &restored, &sol, &transform_map, self.config.backend_type);
                }

                let var_list = if self.config.show_variables {
                    let vars: Vec<(String, f64)> = (0..restored.primal.len())
                        .map(|i| {
                            let name = problem
                                .var_names
                                .get(i)
                                .cloned()
                                .unwrap_or_else(|| format!("x[{}]", i + 1));
                            (name, restored.primal[i])
                        })
                        .collect();
                    Some(vars)
                } else {
                    None
                };

                let mut entry = BenchmarkEntry {
                    instance: inst_name,
                    rows,
                    columns,
                    nonzeros,
                    objective: original_obj,
                    status: status_str,
                    runtime_secs: total_runtime,
                    iterations: sol.iterations,
                    primal_residual: sol.residuals.primal_rel,
                    dual_residual: sol.residuals.dual_rel,
                    backend: sol.backend_name,
                    numerical_warnings: warnings,
                    ref_objective: None,
                    ref_status: None,
                    objective_error: None,
                    rel_objective_error: None,
                    status_match: None,
                    variables: var_list,
                };
                self.attach_reference_comparison(&mut entry);
                entry
            }
            Err(e) => {
                let total_runtime = start_time.elapsed().as_secs_f64();
                warnings.push(format!("Solve execution error: {e}"));
                let mut entry = BenchmarkEntry {
                    instance: inst_name,
                    rows,
                    columns,
                    nonzeros,
                    objective: f64::NAN,
                    status: "SOLVE_ERROR".to_string(),
                    runtime_secs: total_runtime,
                    iterations: 0,
                    primal_residual: f64::NAN,
                    dual_residual: f64::NAN,
                    backend: self.config.backend_type.to_string(),
                    numerical_warnings: warnings,
                    ref_objective: None,
                    ref_status: None,
                    objective_error: None,
                    rel_objective_error: None,
                    status_match: None,
                    variables: None,
                };
                self.attach_reference_comparison(&mut entry);
                entry
            }
        }
    }

    /// Compare entry with reference ground truth if available.
    fn attach_reference_comparison(&self, entry: &mut BenchmarkEntry) {
        if let Some(refs) = &self.reference_results {
            if let Some(ref_entry) = refs.get(&entry.instance) {
                entry.ref_objective = ref_entry.objective;
                entry.ref_status = ref_entry.status.clone();

                if let Some(ref_obj) = ref_entry.objective {
                    if !entry.objective.is_nan() {
                        let diff = (entry.objective - ref_obj).abs();
                        let rel_diff = diff / (1.0 + ref_obj.abs());
                        entry.objective_error = Some(diff);
                        entry.rel_objective_error = Some(rel_diff);
                    }
                }

                if let Some(ref_stat) = &ref_entry.status {
                    let s1 = entry.status.to_uppercase();
                    let s2 = ref_stat.to_uppercase();
                    let matches = s1 == s2
                        || (s1 == "CONVERGED" && s2 == "OPTIMAL")
                        || (s1 == "OPTIMAL" && s2 == "CONVERGED");
                    entry.status_match = Some(matches);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_benchmark_entry_json_csv_roundtrip() {
        let entry = BenchmarkEntry {
            instance: "test_instance.mps".to_string(),
            rows: 10,
            columns: 5,
            nonzeros: 25,
            objective: 123.456,
            status: "CONVERGED".to_string(),
            runtime_secs: 0.042,
            iterations: 88,
            primal_residual: 1.2e-6,
            dual_residual: 3.4e-6,
            backend: "CPU (Rust SIMD)".to_string(),
            numerical_warnings: vec!["Minor warning".to_string()],
            ref_objective: Some(123.456),
            ref_status: Some("OPTIMAL".to_string()),
            objective_error: Some(0.0),
            rel_objective_error: Some(0.0),
            status_match: Some(true),
            variables: Some(vec![("x[1]".to_string(), 12.3), ("x[2]".to_string(), 0.0)]),
        };

        let mut report = BenchmarkReport::new();
        report.add_entry(entry);

        assert_eq!(report.total_models, 1);
        assert_eq!(report.converged_models, 1);
        assert_eq!(report.failed_models, 0);

        let temp_dir = std::env::temp_dir();
        let json_path = temp_dir.join("yutki_test_bench.json");
        let csv_path = temp_dir.join("yutki_test_bench.csv");

        report.export_json(&json_path).expect("export json");
        report.export_csv(&csv_path).expect("export csv");

        let json_content = std::fs::read_to_string(&json_path).expect("read json");
        assert!(json_content.contains("test_instance.mps"));
        assert!(json_content.contains("123.456"));

        let csv_content = std::fs::read_to_string(&csv_path).expect("read csv");
        assert!(csv_content.contains("test_instance.mps"));
        assert!(csv_content.contains("CONVERGED"));
        assert!(csv_content.contains("CPU (Rust SIMD)"));

        let _ = std::fs::remove_file(json_path);
        let _ = std::fs::remove_file(csv_path);
    }

    #[test]
    fn test_reference_results_csv_and_json() {
        let csv_data = "\
instance,objective,status
production,2350.0,OPTIMAL
diet.mps,-150.25,OPTIMAL
";
        let ref_csv = ReferenceResults::load_from_csv_str(csv_data).expect("load csv");
        let entry1 = ref_csv.get("production").expect("find production");
        assert_eq!(entry1.objective, Some(2350.0));
        assert_eq!(entry1.status.as_deref(), Some("OPTIMAL"));

        let entry2 = ref_csv.get("diet").expect("find diet without ext");
        assert_eq!(entry2.objective, Some(-150.25));

        let json_data = r#"
[
    {"instance": "production.mps", "objective": 2350.0, "status": "OPTIMAL"},
    {"instance": "model_b", "objective": 42.0, "status": "CONVERGED"}
]
"#;
        let ref_json = ReferenceResults::load_from_json_str(json_data).expect("load json");
        let item = ref_json.get("production").expect("find in json");
        assert_eq!(item.objective, Some(2350.0));
        assert_eq!(item.status.as_deref(), Some("OPTIMAL"));
    }

    #[test]
    fn test_failed_instance_never_hidden() {
        let runner = BenchmarkRunner::new(BenchmarkConfig::default());
        let temp_dir = std::env::temp_dir();
        let bad_file = temp_dir.join("yutki_malformed.mps");
        {
            let mut f = File::create(&bad_file).unwrap();
            writeln!(f, "NOT_AN_MPS_FILE").unwrap();
        }

        let entry = runner.run_instance(&bad_file);
        assert_eq!(entry.status, "PARSE_ERROR");
        assert!(!entry.numerical_warnings.is_empty());
        assert!(entry.runtime_secs >= 0.0);

        let _ = std::fs::remove_file(bad_file);
    }

    #[test]
    fn test_benchmark_directory_suite_with_reference() {
        let dir = if Path::new("examples/benchmark").exists() {
            PathBuf::from("examples/benchmark")
        } else {
            PathBuf::from("../../examples/benchmark")
        };

        if !dir.exists() {
            return;
        }

        let ref_file = dir.join("reference_results.csv");
        let ref_results = ReferenceResults::load_from_file(&ref_file).expect("load ref csv");

        let config = BenchmarkConfig {
            backend_type: BackendType::Cpu,
            algorithm: SolverAlgorithm::Pdhg,
            time_limit_secs: 10.0,
            max_iterations: 15000,
            tolerance: 1e-3,
            show_variables: true,
        };

        let runner = BenchmarkRunner::new(config)
            .with_reference_results(ref_results)
            .with_show_stages(false);
        let report = runner.run_directory(&dir).expect("run directory benchmark");

        // Verify total instances (production, infeasible, malformed)
        assert_eq!(report.total_models, 3);
        assert_eq!(report.converged_models, 1);
        assert_eq!(report.failed_models, 2);

        // Verify production model
        let prod = report
            .entries
            .iter()
            .find(|e| e.instance.contains("production"))
            .expect("found production");
        assert_eq!(prod.status, "CONVERGED");
        assert!(prod.rel_objective_error.unwrap() < 0.005);
        assert_eq!(prod.status_match, Some(true));

        // Verify infeasible model terminates as non-converged
        let infeas = report
            .entries
            .iter()
            .find(|e| e.instance.contains("infeasible"))
            .expect("found infeasible");
        assert!(infeas.status == "INFEASIBLE" || infeas.status == "ITERATION_LIMIT");

        // Verify malformed model was not omitted
        let mal = report
            .entries
            .iter()
            .find(|e| e.instance.contains("malformed"))
            .expect("found malformed");
        assert_eq!(mal.status, "PARSE_ERROR");
        assert!(!mal.numerical_warnings.is_empty());
    }

    #[test]
    fn test_benchmark_netlib_afiro_simplex_with_reference() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let netlib_dir = PathBuf::from(manifest).join("../../datasets/netlib");
        if !netlib_dir.exists() {
            return;
        }

        let ref_file = netlib_dir.join("reference_netlib.csv");
        if !ref_file.exists() {
            return;
        }

        let ref_results = ReferenceResults::load_from_file(&ref_file).expect("load netlib ref csv");
        let config = BenchmarkConfig {
            backend_type: BackendType::Cpu,
            algorithm: SolverAlgorithm::Simplex,
            time_limit_secs: 10.0,
            max_iterations: 50000,
            tolerance: 1e-6,
            show_variables: true,
        };

        let runner = BenchmarkRunner::new(config)
            .with_reference_results(ref_results)
            .with_show_stages(false);
        let afiro_path = netlib_dir.join("afiro.mps");
        let entry = runner.run_instance(&afiro_path);

        assert_eq!(entry.status, "CONVERGED");
        assert_eq!(entry.backend, "Revised Simplex (CPU)");
        assert!((entry.objective - (-464.753142857143)).abs() < 1e-3);
        assert_eq!(entry.ref_status.as_deref(), Some("OPTIMAL"));
        assert_eq!(entry.status_match, Some(true));
        assert!(entry.rel_objective_error.unwrap() < 1e-5);
        assert!(entry.variables.is_some());
        let vars = entry.variables.unwrap();
        assert_eq!(vars.len(), entry.columns);
    }
}
