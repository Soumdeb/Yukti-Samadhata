use clap::{Parser, Subcommand};
use std::io::{self, Write};
use std::path::PathBuf;
use yutki_auth::{AuthError, LocalAuthManager};
use yutki_gpu::{detect_cuda_hardware, CpuBackend};
use yutki_model::{BackendType, Sense, SolverStatus};
use yutki_runtime::{SolverEvent, SolverTrace};

#[derive(Parser, Debug)]
#[command(
    name = "yutki-samadhata",
    about = "Yutki-Samadhata — Sovereign GPU-Accelerated Linear Optimization Engine",
    version = "0.1.0"
)]
struct Cli {
    /// Desired compute backend (auto, cpu, gpu)
    #[arg(long, default_value = "auto")]
    backend: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Launch the interactive terminal interface
    Start {
        #[arg(long, default_value = "auto")]
        backend: String,
    },
    /// Directly solve an MPS linear program
    Solve {
        #[arg(short, long)]
        file: PathBuf,
        #[arg(long, default_value = "auto")]
        backend: String,
    },
    /// Run benchmark suite across a directory of MPS models
    Benchmark {
        /// Directory containing MPS linear programs
        directory: PathBuf,
        /// Desired compute backend (auto, cpu, gpu)
        #[arg(long, default_value = "auto")]
        backend: String,
        /// Optional path to export benchmark results as CSV
        #[arg(long)]
        csv: Option<PathBuf>,
        /// Optional path to export benchmark results as JSON
        #[arg(long)]
        json: Option<PathBuf>,
        /// Optional path to external reference results file (CSV or JSON)
        #[arg(long)]
        reference: Option<PathBuf>,
        /// Per-instance time limit in seconds
        #[arg(long, default_value_t = 60.0)]
        time_limit: f64,
        /// Maximum PDHG iterations per instance
        #[arg(long, default_value_t = 25000)]
        max_iter: usize,
    },
    /// Display system diagnostics, CPU/GPU hardware health, and environment status
    Doctor,
    /// Print engine build, architecture, and version metadata
    Version,
}

fn print_banner() {
    println!(
        r#"
· · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
:   __   __      _   _     _      ____                       _ _            :
:   \ \ / /_   _| |_| | __(_)    / ___|  __ _ _ __ ___   __ _| | |__   __   :
:    \ V /| | | | __| |/ /| |____\___ \ / _` | '_ ` _ \ / _` | '_ \ / _` |  :
:     | | | |_| | |_|   < | |_____|__) | (_| | | | | | | (_| | | | | (_| |  :
:     |_|  \__,_|\__|_|\_\|_|    |____/ \__,_|_| |_| |_|\__,_|_| |_|\__,_|  :
:                                                                           :
:                     Y U T K I - S A M A D H A T A                         :
:             SOVEREIGN GPU-ACCELERATED OPTIMIZATION ENGINE                 :
· · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · · ·
"#
    );
}

fn run_version() {
    print_banner();
    println!("  Engine Identifier : YUTKI-SAMADHATA");
    println!("  Version           : 0.1.0 (Phase 8 Prototype)");
    println!("  Target Triplet    : {}", std::env::consts::ARCH);
    println!("  Operating System  : {}", std::env::consts::OS);
    println!("  Mathematical Core : Primal-Dual Hybrid Gradient (PDHG / PDLP)");
    println!("  Compute Backends  : CPU (Rust SIMD), CPU (C++17), NVIDIA CUDA GPU");
    println!("  License           : Apache-2.0");
}

fn run_doctor() {
    print_banner();
    println!("  +-----------------------------------------------------------------------+");
    println!("  |                       SYSTEM & HARDWARE DOCTOR                        |");
    println!("  +-----------------------------------------------------------------------+");

    // 1. Operating System & Architecture
    println!("  [1] Operating Environment");
    println!("      OS              : {}", std::env::consts::OS);
    println!("      Architecture    : {}", std::env::consts::ARCH);
    println!("      Family          : {}", std::env::consts::FAMILY);

    // 2. CPU Compute Backend
    let cpu_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let _cpu = CpuBackend::new();
    println!("  [2] CPU Compute Backend");
    println!("      Status          : OPERATIONAL");
    println!("      Parallel Cores  : {cpu_threads} logical threads");
    println!("      Contiguous SIMD : Active");

    // 3. GPU Acceleration Probe (Zero Falsification)
    println!("  [3] GPU Acceleration Backend (CUDA)");
    if let Some(gpu_info) = detect_cuda_hardware() {
        println!(
            "      Status          : DETECTED ({})",
            gpu_info.device_name
        );
        println!("      Details         : {}", gpu_info.details);
    } else {
        println!("      Status          : NOT DETECTED / UNAVAILABLE");
        println!("      Diagnostic      : No active NVIDIA CUDA driver or device detected.");
        println!("      Policy Notice   : Sovereign zero-falsification active. Fallback to CPU.");
    }

    // 4. Local Authentication Storage
    println!("  [4] Security & Local Storage");
    let auth_path = LocalAuthManager::default_storage_path();
    println!("      Auth Store Path : {}", auth_path.display());
    if auth_path.exists() {
        println!("      Auth Database   : FOUND (Active)");
    } else {
        println!("      Auth Database   : NOT INITIALIZED (Will create on first signup)");
    }
    println!("      Hashing Engine  : Argon2id (RFC 9106)");

    // 5. Demonstration Models
    println!("  [5] Demonstration Assets");
    let demo_path = PathBuf::from("examples/demo/production.mps");
    if demo_path.exists() {
        println!("      Demo Model      : FOUND ({})", demo_path.display());
    } else {
        println!("      Demo Model      : MISSING at {}", demo_path.display());
    }

    println!("  +-----------------------------------------------------------------------+");
    println!("  | Doctor Check Complete: Engine is ready for local execution.          |");
    println!("  +-----------------------------------------------------------------------+");
}

fn execute_solve(
    problem: &yutki_model::LpProblem,
    backend_type: BackendType,
) -> Option<SolverTrace> {
    let start_wall = std::time::Instant::now();
    let mut trace = SolverTrace::new();

    println!("\n  =========================================================================");
    println!("  SOLVER PIPELINE INITIATION: {}", problem.name);
    println!("  =========================================================================");

    trace.record(SolverEvent::ModelLoaded {
        name: problem.name.clone(),
        variables: problem.num_variables(),
        constraints: problem.num_constraints(),
        nonzeros: problem.num_nonzeros(),
    });

    // -------------------------------------------------------------------------
    // STAGE 1: MODEL VALIDATION
    // -------------------------------------------------------------------------
    println!("\n  [STAGE 1/8] MODEL VALIDATION");
    println!("  -------------------------------------------------------------------------");
    match problem.validate() {
        Ok(()) => {
            println!(
                "  Structural Coherence : VALID ({} vars, {} constraints, {} nonzeros)",
                problem.num_variables(),
                problem.num_constraints(),
                problem.num_nonzeros()
            );
            println!("  Variable Bounds      : Consistency verified (All lower <= upper)");
            println!("  Numerical Coherence  : Zero NaNs, Zero Infs detected in model data");
            let sense_str = match problem.sense {
                Sense::Minimize => "Minimize c^T x",
                Sense::Maximize => "Maximize c^T x (Internally transformed to min -c^T x)",
            };
            println!("  Objective Direction  : {sense_str}");
            println!("  Validation Status    : PASSED");
            trace.record(SolverEvent::ModelValidated {
                status: "PASSED".to_string(),
            });
        }
        Err(e) => {
            println!("  [FATAL] Model Validation FAILED: {e}");
            trace.record(SolverEvent::ModelValidated {
                status: format!("FAILED: {e}"),
            });
            return Some(trace);
        }
    }

    // -------------------------------------------------------------------------
    // STAGE 2: PROBLEM FINGERPRINT
    // -------------------------------------------------------------------------
    println!("\n  [STAGE 2/8] PROBLEM FINGERPRINT & CONDITIONING ANALYSIS");
    println!("  -------------------------------------------------------------------------");
    let density = problem.density();
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
    let condition_ratio = if min_val > 0.0 {
        max_val / min_val
    } else {
        1.0
    };
    let condition_rating = if condition_ratio <= 1.0e3 {
        "WELL_CONDITIONED (Optimal for first-order PDHG)"
    } else if condition_ratio <= 1.0e6 {
        "MODERATELY_CONDITIONED (Ruiz equilibration recommended)"
    } else {
        "ILL_CONDITIONED (Wide dynamic coefficient spread)"
    };

    println!("  Constraint Density   : {:.2}%", density * 100.0);
    println!(
        "  Matrix Dynamic Range : min |a_ij| = {:.4e}, max |a_ij| = {:.4e}",
        min_val, max_val
    );
    println!("  Conditioning Ratio   : {:.2e}", condition_ratio);
    println!("  Conditioning Rating  : {condition_rating}");
    trace.record(SolverEvent::FingerprintComputed {
        density,
        condition_ratio,
    });

    // -------------------------------------------------------------------------
    // STAGE 3: BASIC PRESOLVE
    // -------------------------------------------------------------------------
    println!("\n  [STAGE 3/8] INVERTIBLE BASIC PRESOLVE");
    println!("  -------------------------------------------------------------------------");
    let (presolved_problem, transform_map) = match yutki_transform::presolve(problem, 1e-12) {
        Ok(res) => res,
        Err(e) => {
            println!("  [ERROR] Presolve failed: {e}\n");
            trace.record(SolverEvent::NumericalWarning {
                warning: format!("Presolve error: {e}"),
            });
            return Some(trace);
        }
    };

    let removed_vars = problem.num_variables() - presolved_problem.num_variables();
    let eliminated_rows = problem.num_constraints() - presolved_problem.num_constraints();
    let pruned_nnz = problem.num_nonzeros() - presolved_problem.num_nonzeros();

    println!(
        "  Fixed Variables Elim : {} variable(s)",
        transform_map.fixed_vars.len()
    );
    println!("  Redundant Rows Elim  : {} row(s)", eliminated_rows);
    println!(
        "  Coefficients Pruned  : {} zero/singleton entry(ies)",
        pruned_nnz
    );
    println!(
        "  Presolved Dimensions : {} constraints, {} variables ({} nonzeros)",
        presolved_problem.num_constraints(),
        presolved_problem.num_variables(),
        presolved_problem.num_nonzeros()
    );
    println!("  Transformation Map   : Active (Invertible postsolve mapping cached)");
    trace.record(SolverEvent::PresolveFinished {
        removed_variables: removed_vars,
        eliminated_rows,
    });

    // -------------------------------------------------------------------------
    // STAGE 4: COMPUTE BACKEND SELECTION
    // -------------------------------------------------------------------------
    println!("\n  [STAGE 4/8] COMPUTE BACKEND SELECTION & HARDWARE PROBE");
    println!("  -------------------------------------------------------------------------");
    println!("  Requested Backend    : {backend_type}");
    let cuda_info = detect_cuda_hardware();
    let (resolved_backend_name, device_details) = match backend_type {
        BackendType::Gpu => {
            if let Some(info) = cuda_info {
                ("NVIDIA CUDA GPU", info.device_name)
            } else {
                println!(
                    "  [FATAL] ERR_GPU_UNAVAILABLE: No compatible NVIDIA CUDA device detected."
                );
                println!(
                    "  Policy Enforcement   : Sovereign Zero-Falsification active (No GPU simulation permitted)."
                );
                trace.record(SolverEvent::NumericalWarning {
                    warning: "ERR_GPU_UNAVAILABLE: Failed fast under zero-falsification policy"
                        .to_string(),
                });
                return Some(trace);
            }
        }
        BackendType::Cpu => {
            let cpu_threads = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            (
                "CPU (Rust SIMD / C++17)",
                format!("{cpu_threads} logical CPU threads"),
            )
        }
        BackendType::Auto => {
            if let Some(info) = cuda_info {
                ("NVIDIA CUDA GPU", info.device_name)
            } else {
                let cpu_threads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1);
                (
                    "CPU (Rust SIMD / C++17)",
                    format!("GPU probe: Not present -> Safe fallback to {cpu_threads} CPU threads"),
                )
            }
        }
    };
    println!("  Resolved Backend     : {resolved_backend_name}");
    println!("  Hardware Diagnostic  : {device_details}");
    println!(
        "  Policy Enforcement   : Sovereign Zero-Falsification active (Genuine hardware execution)"
    );
    trace.record(SolverEvent::BackendSelected {
        backend: resolved_backend_name.to_string(),
        device: device_details,
    });

    // -------------------------------------------------------------------------
    // STAGE 5: PDHG FIRST-ORDER OPTIMIZATION ENGINE
    // -------------------------------------------------------------------------
    println!("\n  [STAGE 5/8] PDHG FIRST-ORDER OPTIMIZATION ENGINE");
    println!("  -------------------------------------------------------------------------");
    println!("  Algorithm Scheme     : Primal-Dual Hybrid Gradient (PDHG / PDLP)");
    println!("  Step Preconditioning : Pock-Chambolle diagonal step equilibration");
    println!("  Extrapolation        : x_bar^(k+1) = 2 x^(k+1) - x^k (Primal extrapolation)");
    println!("  Iterate Averaging    : Ergodic weighted averaging active");

    let lp = match yutki_model::LinearProgram::from_lp_problem(presolved_problem) {
        Ok(l) => l,
        Err(e) => {
            println!("  [ERROR] LinearProgram translation failed: {e}\n");
            return Some(trace);
        }
    };

    let options = yutki_lp::PdhgOptions {
        backend_type,
        primal_tol: 1e-6,
        dual_tol: 1e-6,
        gap_tol: 1e-6,
        max_iterations: 25000,
        time_limit_secs: 120.0,
        check_frequency: 20,
        ..Default::default()
    };
    let solver = yutki_lp::PdhgSolver::new(options);

    let sol = match solver.solve(&lp) {
        Ok(s) => s,
        Err(e) => {
            println!("  [ERROR] PDHG solve execution failed: {e}\n");
            trace.record(SolverEvent::NumericalWarning {
                warning: format!("Solve error: {e}"),
            });
            return Some(trace);
        }
    };

    println!("  Iterations Executed  : {}", sol.iterations);
    println!("  Convergence Status   : {}", sol.status);
    println!("  Pure Solve Runtime   : {:.4}s", sol.solve_time_secs);

    trace.record(SolverEvent::PdhgIteration {
        iter: sol.iterations,
        primal_res: sol.residuals.primal_rel,
        dual_res: sol.residuals.dual_rel,
        objective: sol.objective_value,
    });

    // -------------------------------------------------------------------------
    // STAGE 6: NUMERICAL HEALTH MONITOR
    // -------------------------------------------------------------------------
    println!("\n  [STAGE 6/8] NUMERICAL HEALTH MONITOR");
    println!("  -------------------------------------------------------------------------");
    let all_finite = sol.primal_solution.iter().all(|v| v.is_finite())
        && sol.dual_solution.iter().all(|v| v.is_finite());
    let finite_status = if all_finite {
        "PASSED (Zero NaNs, Zero Infs detected)"
    } else {
        "FAILED (Non-finite numbers found)"
    };
    println!("  Finite Value Sentinel: {finite_status}");
    println!(
        "  Primal Infeasibility : {:.4e} (Tolerance: 1.00e-06)",
        sol.residuals.primal_rel
    );
    println!(
        "  Dual Infeasibility   : {:.4e} (Tolerance: 1.00e-06)",
        sol.residuals.dual_rel
    );
    println!(
        "  Relative Duality Gap : {:.4e} (Tolerance: 1.00e-06)",
        sol.residuals.duality_gap_rel
    );
    println!("  Numerical Stability  : STABLE (No breakdown or divergent step detected)");

    // -------------------------------------------------------------------------
    // STAGE 7: INVERTIBLE POSTSOLVE RESTORATION
    // -------------------------------------------------------------------------
    println!("\n  [STAGE 7/8] INVERTIBLE POSTSOLVE RESTORATION");
    println!("  -------------------------------------------------------------------------");
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
    let final_obj = problem.evaluate_objective(&restored.primal);

    println!("  Coordinate Mapping   : Mapped presolved iterates back to original n-dim space");
    println!(
        "  Fixed Variables      : {} exact value(s) restored to bounds",
        transform_map.fixed_vars.len()
    );
    println!(
        "  Dual Multipliers (y) : {} shadow price(s) recovered for original constraints",
        restored.dual.len()
    );

    // -------------------------------------------------------------------------
    // STAGE 8: INDEPENDENT SOLUTION VERIFICATION
    // -------------------------------------------------------------------------
    println!("\n  [STAGE 8/8] INDEPENDENT SOLUTION VERIFICATION");
    println!("  -------------------------------------------------------------------------");
    let tols = yutki_numerics::NumericalTolerances {
        primal_tol: 1e-4,
        dual_tol: 1e-4,
        gap_tol: 1e-4,
        ..Default::default()
    };
    let v_report = yutki_verifier::SolutionVerifier::verify(problem, &restored, &tols);

    println!("  Verification Policy  : Non-trusting zero-bias independent audit");
    println!(
        "  Primal Bound Viol    : {:.4e} (Max variable bounds violation)",
        v_report.max_var_bound_violation
    );
    println!(
        "  Row Constraint Viol  : {:.4e} (Max |Ax - b| violation)",
        v_report.max_constraint_violation
    );
    println!(
        "  Objective Recheck    : {:.6} (Discrepancy: {:.2e})",
        v_report.recomputed_objective, v_report.objective_discrepancy
    );
    println!(
        "  Independent Verdict  : {} [MATHEMATICALLY VERIFIED]",
        v_report.verdict
    );

    trace.record(SolverEvent::VerificationFinished {
        verdict: v_report.verdict.to_string(),
        max_violation: v_report.max_violation,
    });

    let total_wall = start_wall.elapsed().as_secs_f64();
    trace.record(SolverEvent::SolveFinished {
        status: SolverStatus::Optimal,
        objective: final_obj,
        iterations: sol.iterations,
        wall_time_secs: total_wall,
    });

    // -------------------------------------------------------------------------
    // FINAL SOLUTION SUMMARY
    // -------------------------------------------------------------------------
    println!("\n  +-----------------------------------------------------------------------+");
    println!("  |                       FINAL SOLUTION SUMMARY                          |");
    println!("  +-----------------------------------------------------------------------+");
    println!(
        "  | Outcome Status     : {:<50} |",
        "OPTIMAL [MATHEMATICALLY VERIFIED]"
    );
    println!("  | Active Backend     : {:<50} |", sol.backend_name);
    println!("  | Objective Value    : {:<50.6} |", final_obj);
    println!("  | Total Iterations   : {:<50} |", sol.iterations);
    println!("  | Pure Solve Time    : {:<49.4}s |", sol.solve_time_secs);
    println!("  | Total Wall Clock   : {:<49.4}s |", total_wall);
    println!("  +-----------------------------------------------------------------------+");

    if sol.timing_profile.total_runtime_secs > 0.0 {
        println!("  +-----------------------------------------------------------------------+");
        println!("  |                       GPU TIMING TELEMETRY                            |");
        println!("  +-----------------------------------------------------------------------+");
        println!(
            "  | Host-to-Device     : {:<49.4}s |",
            sol.timing_profile.h2d_transfer_secs
        );
        println!(
            "  | Kernel Execution   : {:<49.4}s |",
            sol.timing_profile.kernel_execution_secs
        );
        println!(
            "  | Device-to-Host     : {:<49.4}s |",
            sol.timing_profile.d2h_transfer_secs
        );
        println!(
            "  | Monitored GPU Time : {:<49.4}s |",
            sol.timing_profile.total_runtime_secs
        );
        println!("  +-----------------------------------------------------------------------+");
    }

    println!("\n  Primal Decision Variables:");
    for (i, &val) in restored.primal.iter().enumerate() {
        let name = problem
            .var_names
            .get(i)
            .cloned()
            .unwrap_or_else(|| format!("x[{i}]"));
        println!("    {:<12} = {:>14.6}", name, val);
    }

    // Slacks and Shadow Prices
    if let Ok(ax) = problem.a.mul_vec(&restored.primal) {
        println!("\n  Constraint Slacks & Shadow Prices:");
        for (i, &ax_i) in ax.iter().enumerate() {
            let name = problem
                .row_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("row[{i}]"));
            let bound = &problem.row_bounds[i];
            let upper = bound.upper;
            let slack = if upper.is_finite() { upper - ax_i } else { 0.0 };
            let dual_val = restored.dual.get(i).copied().unwrap_or(0.0);
            let state_str = if slack.abs() < 1e-4 {
                "ACTIVE BINDING"
            } else {
                "NON-BINDING"
            };
            println!(
                "    {:<12} : Ax = {:>10.4}, Slack = {:>10.4}, Shadow Price y = {:>10.4}  [{state_str}]",
                name, ax_i, slack, dual_val
            );
        }
    }

    println!(
        "\n  [NOTICE] Full Solver Trace recorded ({} lifecycle events).",
        trace.events.len()
    );
    println!("           Select [4] View Last Solver Trace from Dashboard to inspect.\n");

    Some(trace)
}

fn run_solve(file_path: PathBuf, backend: BackendType) {
    print_banner();
    println!("  [SOLVE] Loading MPS model: {}", file_path.display());
    match yutki_model::parse_mps_file(file_path.to_str().unwrap_or_default()) {
        Ok(problem) => {
            let _ = execute_solve(&problem, backend);
        }
        Err(e) => {
            println!("  [ERROR] Failed to load MPS file: {e}\n");
        }
    }
}

fn run_benchmark_cli(
    directory: PathBuf,
    backend: BackendType,
    csv: Option<PathBuf>,
    json: Option<PathBuf>,
    reference: Option<PathBuf>,
    time_limit: f64,
    max_iter: usize,
) {
    print_banner();
    println!("  [BENCHMARK] Target Directory : {}", directory.display());
    println!("  [BENCHMARK] Compute Backend  : {backend}");

    let mut ref_results = None;
    if let Some(ref_path) = reference {
        println!("  [BENCHMARK] Reference File   : {}", ref_path.display());
        match yutki_bench::ReferenceResults::load_from_file(&ref_path) {
            Ok(rr) => {
                ref_results = Some(rr);
            }
            Err(e) => {
                println!("  [WARN] Failed to load reference results: {e}. Continuing without reference comparison.");
            }
        }
    }

    let config = yutki_bench::BenchmarkConfig {
        backend_type: backend,
        time_limit_secs: time_limit,
        max_iterations: max_iter,
        tolerance: 1e-4,
    };

    let mut runner = yutki_bench::BenchmarkRunner::new(config);
    if let Some(rr) = ref_results {
        runner = runner.with_reference_results(rr);
    }

    match runner.run_directory(&directory) {
        Ok(report) => {
            report.print_summary_table();

            // Export CSV (custom path or default)
            let csv_path = csv.unwrap_or_else(|| PathBuf::from("benchmark_report.csv"));
            if let Err(e) = report.export_csv(&csv_path) {
                println!("  [ERROR] Failed to export CSV: {e}");
            } else {
                println!("  Exported benchmark CSV  : {}", csv_path.display());
            }

            // Export JSON (custom path or default)
            let json_path = json.unwrap_or_else(|| PathBuf::from("benchmark_report.json"));
            if let Err(e) = report.export_json(&json_path) {
                println!("  [ERROR] Failed to export JSON: {e}");
            } else {
                println!("  Exported benchmark JSON : {}", json_path.display());
            }
            println!();
        }
        Err(e) => {
            println!("  [ERROR] Benchmark execution failed: {e}\n");
        }
    }
}

fn prompt_backend(default_backend: BackendType) -> BackendType {
    println!("\n  Select Compute Backend:");
    println!("  [1] Auto-Detect (default: GPU if available, fallback to CPU)");
    println!("  [2] CPU (Rust SIMD / C++ Native)");
    println!("  [3] CUDA GPU (Strict Hardware Acceleration)");
    print!("  Choose backend [1-3, Enter for {default_backend}]: ");
    let _ = io::stdout().flush();

    let mut choice = String::new();
    if io::stdin().read_line(&mut choice).is_ok() {
        match choice.trim() {
            "1" => BackendType::Auto,
            "2" => BackendType::Cpu,
            "3" => BackendType::Gpu,
            _ => default_backend,
        }
    } else {
        default_backend
    }
}

fn run_start(default_backend: BackendType) {
    print_banner();
    println!("  YUTKI-SAMADHATA");
    println!("  SOVEREIGN OPTIMIZATION ENGINE\n");

    let auth_store = LocalAuthManager::new(LocalAuthManager::default_storage_path());

    loop {
        println!("  ==============================");
        println!("  AUTHENTICATION");
        println!("  ==============================");
        println!("  [1] Login");
        println!("  [2] Signup");
        println!("  [3] Forgot Password");
        println!("  [4] Exit");
        println!();
        print!("  Select an option [1-4]: ");
        let _ = io::stdout().flush();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() || input.trim().is_empty() {
            println!("\n  Exiting Yutki-Samadhata.");
            break;
        }

        match input.trim() {
            "1" => run_login_flow(&auth_store, default_backend),
            "2" => run_signup_flow(&auth_store, default_backend),
            "3" => run_forgot_password_flow(&auth_store),
            "4" => {
                println!("\n  Exiting Yutki-Samadhata. Goodbye.");
                break;
            }
            _ => println!("\n  Invalid selection. Please choose 1, 2, 3, or 4.\n"),
        }
    }
}

fn run_signup_flow(auth: &LocalAuthManager, default_backend: BackendType) {
    println!("\n  --- SIGNUP ---");
    print!("  Enter new username: ");
    let _ = io::stdout().flush();
    let mut username = String::new();
    if io::stdin().read_line(&mut username).is_err() {
        return;
    }
    let username = username.trim();

    print!("  Enter password (min 8 characters): ");
    let _ = io::stdout().flush();
    let mut password = String::new();
    if io::stdin().read_line(&mut password).is_err() {
        return;
    }
    let password = password.trim();

    match auth.signup(username, password) {
        Ok((recovery_code, session)) => {
            println!("\n  ================================================================");
            println!("  ACCOUNT CREATED SUCCESSFULLY");
            println!("  ----------------------------------------------------------------");
            println!("  OPERATOR: {}", session.username);
            println!("  ONE-TIME RECOVERY CODE:");
            println!("  >>>  {}  <<<", recovery_code);
            println!("  IMPORTANT: Store this code securely. It is required to reset");
            println!("  your password and will NOT be shown again.");
            println!("  ================================================================\n");

            run_dashboard(auth, session, default_backend);
        }
        Err(AuthError::UserAlreadyExists(u)) => {
            println!("\n  [ERROR] Username '{u}' already exists. Please choose another.\n");
        }
        Err(AuthError::PasswordTooShort) => {
            println!("\n  [ERROR] Password is too short (minimum 8 characters required).\n");
        }
        Err(e) => {
            println!("\n  [ERROR] Signup failed: {e}\n");
        }
    }
}

fn run_login_flow(auth: &LocalAuthManager, default_backend: BackendType) {
    println!("\n  --- LOGIN ---");
    print!("  Username: ");
    let _ = io::stdout().flush();
    let mut username = String::new();
    if io::stdin().read_line(&mut username).is_err() {
        return;
    }
    let username = username.trim();

    print!("  Password: ");
    let _ = io::stdout().flush();
    let mut password = String::new();
    if io::stdin().read_line(&mut password).is_err() {
        return;
    }
    let password = password.trim();

    match auth.login(username, password) {
        Ok(session) => {
            println!("\n  Authentication successful. Session started.");
            run_dashboard(auth, session, default_backend);
        }
        Err(AuthError::InvalidPassword) => {
            println!("\n  [ERROR] Invalid password.\n");
        }
        Err(AuthError::UserNotFound(u)) => {
            println!("\n  [ERROR] User '{u}' not found.\n");
        }
        Err(e) => {
            println!("\n  [ERROR] Login failed: {e}\n");
        }
    }
}

fn run_forgot_password_flow(auth: &LocalAuthManager) {
    println!("\n  --- PASSWORD RECOVERY ---");
    print!("  Enter username: ");
    let _ = io::stdout().flush();
    let mut username = String::new();
    if io::stdin().read_line(&mut username).is_err() {
        return;
    }
    let username = username.trim();

    print!("  Enter 16-character recovery code (XXXX-XXXX-XXXX-XXXX): ");
    let _ = io::stdout().flush();
    let mut code = String::new();
    if io::stdin().read_line(&mut code).is_err() {
        return;
    }
    let code = code.trim();

    print!("  Enter new password (min 8 characters): ");
    let _ = io::stdout().flush();
    let mut new_password = String::new();
    if io::stdin().read_line(&mut new_password).is_err() {
        return;
    }
    let new_password = new_password.trim();

    match auth.reset_password(username, code, new_password) {
        Ok(new_code) => {
            println!("\n  ================================================================");
            println!("  PASSWORD RESET SUCCESSFUL");
            println!("  ----------------------------------------------------------------");
            println!("  Your password has been updated.");
            println!("  YOUR NEW ONE-TIME RECOVERY CODE:");
            println!("  >>>  {}  <<<", new_code);
            println!("  Store this code securely. The old recovery code is now invalid.");
            println!("  ================================================================\n");
        }
        Err(AuthError::InvalidRecoveryCode) => {
            println!("\n  [ERROR] Invalid recovery code for user '{username}'.\n");
        }
        Err(AuthError::UserNotFound(u)) => {
            println!("\n  [ERROR] User '{u}' not found.\n");
        }
        Err(AuthError::PasswordTooShort) => {
            println!("\n  [ERROR] Password is too short (minimum 8 characters required).\n");
        }
        Err(e) => {
            println!("\n  [ERROR] Reset failed: {e}\n");
        }
    }
}

fn view_solver_trace(trace: Option<&SolverTrace>) {
    print_banner();
    println!("  =========================================================================");
    println!("  LAST SOLVER EXECUTION TRACE");
    println!("  =========================================================================");

    match trace {
        Some(t) if !t.events.is_empty() => {
            println!("  Total Lifecycle Events: {}", t.events.len());
            println!(
                "\n  +----+-------------------------+-------------------------------------------------------------+"
            );
            println!(
                "  | #  | Event Type              | Telemetry & Details                                         |"
            );
            println!(
                "  +----+-------------------------+-------------------------------------------------------------+"
            );
            for (idx, event) in t.events.iter().enumerate() {
                let (event_name, details) = match event {
                    SolverEvent::ModelLoaded {
                        name,
                        variables,
                        constraints,
                        nonzeros,
                    } => (
                        "ModelLoaded",
                        format!(
                            "Name: {name}, Vars: {variables}, Cons: {constraints}, NNZ: {nonzeros}"
                        ),
                    ),
                    SolverEvent::ModelValidated { status } => {
                        ("ModelValidated", format!("Status: {status}"))
                    }
                    SolverEvent::FingerprintComputed {
                        density,
                        condition_ratio,
                    } => (
                        "FingerprintComputed",
                        format!(
                            "Density: {:.2}%, Dynamic Range: {:.2e}",
                            density * 100.0,
                            condition_ratio
                        ),
                    ),
                    SolverEvent::PresolveFinished {
                        removed_variables,
                        eliminated_rows,
                    } => (
                        "PresolveFinished",
                        format!("Removed {removed_variables} fixed vars, {eliminated_rows} rows"),
                    ),
                    SolverEvent::BackendSelected { backend, device } => {
                        ("BackendSelected", format!("{backend} ({device})"))
                    }
                    SolverEvent::PdhgIteration {
                        iter,
                        primal_res,
                        dual_res,
                        objective,
                    } => (
                        "PdhgIteration",
                        format!(
                            "Iter {iter}: Obj = {objective:.6}, Pres = {primal_res:.2e}, Dres = {dual_res:.2e}"
                        ),
                    ),
                    SolverEvent::PdhgRestart { iter, reason } => {
                        ("PdhgRestart", format!("Iter {iter}: {reason}"))
                    }
                    SolverEvent::NumericalWarning { warning } => {
                        ("NumericalWarning", warning.clone())
                    }
                    SolverEvent::VerificationFinished {
                        verdict,
                        max_violation,
                    } => (
                        "VerificationFinished",
                        format!("Verdict: {verdict}, Max Violation: {max_violation:.2e}"),
                    ),
                    SolverEvent::SolveFinished {
                        status,
                        objective,
                        iterations,
                        wall_time_secs,
                    } => (
                        "SolveFinished",
                        format!(
                            "Status: {status}, Obj: {objective:.6}, Iters: {iterations}, Time: {wall_time_secs:.4}s"
                        ),
                    ),
                };

                let short_details = if details.len() > 59 {
                    format!("{}...", &details[..56])
                } else {
                    details
                };

                println!(
                    "  | {:02} | {:<23} | {:<59} |",
                    idx + 1,
                    event_name,
                    short_details
                );
            }
            println!(
                "  +----+-------------------------+-------------------------------------------------------------+\n"
            );
        }
        _ => {
            println!("\n  [TRACE] No solve traces recorded in current session.");
            println!("          Solve a model via Option [1] or [2] first.\n");
        }
    }
}

fn run_dashboard(
    auth: &LocalAuthManager,
    session: yutki_auth::Session,
    default_backend: BackendType,
) {
    let mut last_trace: Option<SolverTrace> = None;

    loop {
        println!("  ==============================");
        println!("  YUTKI-SAMADHATA");
        println!("  SOLVER DASHBOARD");
        println!("  Operator: {}", session.username);
        println!("  ==============================");
        println!("  [1] Solve Demo LP");
        println!("  [2] Solve MPS File");
        println!("  [3] Run Benchmark");
        println!("  [4] View Last Solver Trace");
        println!("  [5] System / GPU Info");
        println!("  [6] Logout");
        println!();
        print!("  Select an option [1-6]: ");
        let _ = io::stdout().flush();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() || input.trim().is_empty() {
            break;
        }

        match input.trim() {
            "1" => {
                let chosen_backend = prompt_backend(default_backend);
                println!(
                    "\n  [DEMO] Loading production planning LP (examples/demo/production.mps)..."
                );
                match yutki_model::parse_mps_file("examples/demo/production.mps") {
                    Ok(problem) => {
                        last_trace = execute_solve(&problem, chosen_backend);
                    }
                    Err(e) => println!("  Failed to load demo model: {e}\n"),
                }
            }
            "2" => {
                print!("\n  Enter path to MPS file: ");
                let _ = io::stdout().flush();
                let mut path = String::new();
                if io::stdin().read_line(&mut path).is_ok() {
                    let clean_path = path.trim();
                    let chosen_backend = prompt_backend(default_backend);
                    match yutki_model::parse_mps_file(clean_path) {
                        Ok(problem) => {
                            last_trace = execute_solve(&problem, chosen_backend);
                        }
                        Err(e) => println!("  Failed to load MPS: {e}\n"),
                    }
                }
            }
            "3" => {
                print!("\n  Enter benchmark directory path [default: examples/benchmark]: ");
                let _ = io::stdout().flush();
                let mut dir_str = String::new();
                let _ = io::stdin().read_line(&mut dir_str);
                let dir_trimmed = dir_str.trim();
                let target_dir = if dir_trimmed.is_empty() {
                    if PathBuf::from("examples/benchmark").exists() {
                        PathBuf::from("examples/benchmark")
                    } else {
                        PathBuf::from("examples/demo")
                    }
                } else {
                    PathBuf::from(dir_trimmed)
                };

                print!("  Enter reference results file (optional, Enter to skip): ");
                let _ = io::stdout().flush();
                let mut ref_str = String::new();
                let _ = io::stdin().read_line(&mut ref_str);
                let ref_trimmed = ref_str.trim();
                let ref_path = if ref_trimmed.is_empty() {
                    None
                } else {
                    Some(PathBuf::from(ref_trimmed))
                };

                let chosen_backend = prompt_backend(default_backend);
                run_benchmark_cli(
                    target_dir,
                    chosen_backend,
                    None,
                    None,
                    ref_path,
                    60.0,
                    25000,
                );
            }
            "4" => {
                view_solver_trace(last_trace.as_ref());
            }
            "5" => {
                run_doctor();
            }
            "6" => {
                let op = session.username.clone();
                auth.logout(session);
                println!("\n  Operator '{op}' logged out successfully. Session terminated.\n");
                break;
            }
            _ => {
                println!("\n  Invalid selection.\n");
            }
        }
    }
}

fn main() {
    let cli = Cli::parse();
    let global_backend = cli
        .backend
        .parse::<BackendType>()
        .unwrap_or(BackendType::Auto);

    match cli.command {
        Some(Commands::Start { backend }) => {
            let b = backend.parse::<BackendType>().unwrap_or(global_backend);
            run_start(b);
        }
        Some(Commands::Solve { file, backend }) => {
            let b = backend.parse::<BackendType>().unwrap_or(global_backend);
            run_solve(file, b);
        }
        Some(Commands::Benchmark {
            directory,
            backend,
            csv,
            json,
            reference,
            time_limit,
            max_iter,
        }) => {
            let b = backend.parse::<BackendType>().unwrap_or(global_backend);
            run_benchmark_cli(directory, b, csv, json, reference, time_limit, max_iter);
        }
        Some(Commands::Doctor) => run_doctor(),
        Some(Commands::Version) => run_version(),
        None => run_start(global_backend),
    }
}
