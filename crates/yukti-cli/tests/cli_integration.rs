use std::path::PathBuf;
use std::process::Command;
use yukti_auth::LocalAuthManager;
use yukti_model::parse_mps_file;
use yukti_verifier::SolutionVerifier;

#[test]
fn test_cli_auth_workflow_end_to_end() {
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let temp_dir = std::env::temp_dir().join(format!("yukti_cli_test_auth_{nonce}"));
    let auth_path = temp_dir.join("auth_store.json");
    let auth = LocalAuthManager::new(&auth_path);

    // 1. Signup
    let (rec_code, session) = auth.signup("operator_cli", "P@ssw0rd1234").unwrap();
    assert_eq!(session.username, "operator_cli");
    assert_eq!(rec_code.len(), 19);

    // 2. Login with correct password
    let login_session = auth.login("operator_cli", "P@ssw0rd1234").unwrap();
    assert_eq!(login_session.username, "operator_cli");

    // 3. Login with wrong password
    assert!(auth.login("operator_cli", "WrongPassword!").is_err());

    // 4. Reset password using recovery code
    let new_rec_code = auth
        .reset_password("operator_cli", &rec_code, "BrandNewP@ssw0rd5678")
        .unwrap();
    assert_ne!(new_rec_code, rec_code);

    // 5. Login with new password
    let updated_session = auth.login("operator_cli", "BrandNewP@ssw0rd5678").unwrap();
    assert_eq!(updated_session.username, "operator_cli");

    // 6. Old password fails
    assert!(auth.login("operator_cli", "P@ssw0rd1234").is_err());

    // 7. Old recovery code fails
    assert!(auth
        .reset_password("operator_cli", &rec_code, "AnotherPassword999")
        .is_err());

    let _ = std::fs::remove_dir_all(temp_dir);
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn test_cli_solve_netlib_afiro_mathematical_precision() {
    let afiro_path = workspace_root().join("datasets/netlib/afiro.mps");
    assert!(
        afiro_path.exists(),
        "Netlib file afiro.mps must exist at {}",
        afiro_path.display()
    );

    let problem = parse_mps_file(&afiro_path).unwrap();
    assert_eq!(problem.name, "AFIRO");
    assert_eq!(problem.num_variables(), 32);
    assert_eq!(problem.num_constraints(), 27);

    let lp = yukti_model::LinearProgram::from_lp_problem(problem.clone()).unwrap();
    let (solution, telemetry) =
        yukti_lp::RevisedSimplexSolver::solve(&lp, &yukti_lp::SimplexOptions::default()).unwrap();
    assert_eq!(telemetry.status, yukti_lp::SimplexStatus::Optimal);
    assert_eq!(solution.primal.len(), 32);
    assert!((solution.objective - (-464.753142857143)).abs() < 1e-3);

    // Independent verification with matching 1e-4 tolerance
    let tols = yukti_numerics::NumericalTolerances {
        primal_tol: 1e-4,
        dual_tol: 1e-4,
        gap_tol: 1e-4,
        ..Default::default()
    };
    let report = SolutionVerifier::verify(&problem, &solution, &tols);
    assert_eq!(
        report.verdict,
        yukti_verifier::VerificationVerdict::Valid,
        "Report: {}",
        report.message
    );
    assert!(report.max_violation < 1e-4);
}

#[test]
fn test_cli_binary_commands() {
    let bin_path = env!("CARGO_BIN_EXE_yukti-samadhata");
    let root = workspace_root();

    // 1. Help flag
    let output_help = Command::new(bin_path)
        .current_dir(&root)
        .arg("--help")
        .output()
        .expect("Failed to execute binary");
    assert!(output_help.status.success());
    let stdout_help = String::from_utf8_lossy(&output_help.stdout);
    assert!(stdout_help.contains("Yukti-Samadhata"));

    // 2. Version command
    let output_version = Command::new(bin_path)
        .current_dir(&root)
        .arg("version")
        .output()
        .expect("Failed to execute version");
    assert!(output_version.status.success());
    let stdout_ver = String::from_utf8_lossy(&output_version.stdout);
    assert!(stdout_ver.contains("YUKTI-SAMADHATA"));
    assert!(stdout_ver.contains("0.1.0"));

    // 3. Doctor command
    let output_doctor = Command::new(bin_path)
        .current_dir(&root)
        .arg("doctor")
        .output()
        .expect("Failed to execute doctor");
    assert!(output_doctor.status.success());
    let stdout_doc = String::from_utf8_lossy(&output_doctor.stdout);
    assert!(stdout_doc.contains("SYSTEM & HARDWARE DOCTOR"));
    assert!(stdout_doc.contains("CPU Compute Backend"));

    // 4. Direct solve command
    let output_solve = Command::new(bin_path)
        .current_dir(&root)
        .args([
            "solve",
            "--file",
            "datasets/netlib/afiro.mps",
            "--backend",
            "cpu",
        ])
        .output()
        .expect("Failed to execute solve");
    assert!(output_solve.status.success());
    let stdout_solve = String::from_utf8_lossy(&output_solve.stdout);
    assert!(stdout_solve.contains("OPTIMAL [MATHEMATICALLY VERIFIED]"));
    assert!(stdout_solve.contains("X01"));

    // 5. Benchmark command
    let output_bench = Command::new(bin_path)
        .current_dir(&root)
        .args([
            "benchmark",
            "datasets/netlib/afiro.mps",
            "--backend",
            "cpu",
            "--time-limit",
            "10",
            "--max-iter",
            "5000",
        ])
        .output()
        .expect("Failed to execute benchmark");
    assert!(output_bench.status.success());
    let stdout_bench = String::from_utf8_lossy(&output_bench.stdout);
    assert!(stdout_bench.contains("BENCHMARK"));
    assert!(stdout_bench.contains("afiro.mps"));
    assert!(stdout_bench.contains("SOLVED PRIMAL DECISION VARIABLES & VALUES"));
}
