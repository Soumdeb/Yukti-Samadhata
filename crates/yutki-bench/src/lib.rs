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
use yutki_lp::{PdhgOptions, PdhgSolver, PdhgStatus};
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
        let json_str = serde_json::to_string_pretty(self)
            .map_err(|e| SolverError::Io(format!("JSON serialization error: {e}")))?;
        let mut file = File::create(path.as_ref()).map_err(|e| {
            SolverError::Io(format!(
                "Failed to create benchmark JSON file '{}': {e}",
                path.as_ref().display()
            ))
        })?;
        file.write_all(json_str.as_bytes())
            .map_err(|e| SolverError::Io(format!("Failed to write benchmark JSON: {e}")))?;
        Ok(())
    }

    /// Export report to standard CSV format with full metrics and comparison.
    pub fn export_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), SolverError> {
        let mut file = File::create(path.as_ref()).map_err(|e| {
            SolverError::Io(format!(
                "Failed to create benchmark CSV file '{}': {e}",
                path.as_ref().display()
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
    pub time_limit_secs: f64,
    pub max_iterations: usize,
    pub tolerance: f64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            backend_type: BackendType::Auto,
            time_limit_secs: 60.0,
            max_iterations: 25000,
            tolerance: 1e-4,
        }
    }
}

/// The benchmark execution runner.
pub struct BenchmarkRunner {
    pub config: BenchmarkConfig,
    pub reference_results: Option<ReferenceResults>,
}

impl BenchmarkRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            reference_results: None,
        }
    }

    pub fn with_reference_results(mut self, refs: ReferenceResults) -> Self {
        self.reference_results = Some(refs);
        self
    }

    /// Execute benchmark across an entire directory of MPS models.
    /// Traverses all `.mps` files and evaluates them deterministically.
    pub fn run_directory<P: AsRef<Path>>(&self, dir: P) -> Result<BenchmarkReport, SolverError> {
        let p = dir.as_ref();
        if !p.exists() {
            return Err(SolverError::Io(format!(
                "Benchmark directory '{}' does not exist",
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

        for file_path in files {
            let entry = self.run_instance(&file_path);
            report.add_entry(entry);
        }

        Ok(report)
    }

    /// Recursively or non-recursively collect all `.mps` files in a directory.
    fn collect_mps_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<(), SolverError> {
        if dir.is_dir() {
            let entries = std::fs::read_dir(dir).map_err(|e| {
                SolverError::Io(format!("Failed to read directory '{}': {e}", dir.display()))
            })?;
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    Self::collect_mps_files(&path, files)?;
                } else if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    let ext_lower = ext.to_lowercase();
                    if ext_lower == "mps" {
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
                };
                self.attach_reference_comparison(&mut entry);
                return entry;
            }
        };

        let rows = problem.num_constraints();
        let columns = problem.num_variables();
        let nonzeros = problem.num_nonzeros();

        // 2. Invertible presolve
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
            time_limit_secs: 10.0,
            max_iterations: 15000,
            tolerance: 1e-3,
        };

        let runner = BenchmarkRunner::new(config).with_reference_results(ref_results);
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
}
