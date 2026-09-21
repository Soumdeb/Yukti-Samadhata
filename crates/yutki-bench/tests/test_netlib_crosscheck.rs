use std::path::PathBuf;
use yutki_bench::{BenchmarkConfig, BenchmarkRunner, ReferenceResults};
use yutki_lp::SolverAlgorithm;
use yutki_model::BackendType;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn test_benchmark_single_mps_file_netlib_agg3() {
    let agg3_path = workspace_root().join("datasets/netlib/agg3.mps");
    assert!(agg3_path.exists(), "datasets/netlib/agg3.mps must exist");

    let config = BenchmarkConfig {
        backend_type: BackendType::Auto,
        algorithm: SolverAlgorithm::Pdhg,
        time_limit_secs: 5.0,
        max_iterations: 1000,
        tolerance: 1e-4,
        show_variables: true,
    };

    let runner = BenchmarkRunner::new(config).with_show_stages(false);
    let report = runner
        .run_directory(&agg3_path)
        .expect("Single file benchmark run should succeed");

    assert_eq!(
        report.entries.len(),
        1,
        "Exactly 1 instance should be evaluated"
    );
    let entry = &report.entries[0];
    assert_eq!(entry.instance, "agg3.mps");
    assert_eq!(entry.rows, 516);
    assert_eq!(entry.columns, 302);
    assert_eq!(entry.nonzeros, 4300);
    assert!(entry.runtime_secs >= 0.0);
}

#[test]
fn test_benchmark_single_mps_file_netlib_afiro_with_reference() {
    let afiro_path = workspace_root().join("datasets/netlib/afiro.mps");
    let ref_path = workspace_root().join("datasets/netlib/reference_netlib.csv");
    assert!(afiro_path.exists(), "datasets/netlib/afiro.mps must exist");
    assert!(ref_path.exists(), "reference_netlib.csv must exist");

    let refs = ReferenceResults::load_from_file(&ref_path).expect("Reference results must parse");

    let config = BenchmarkConfig {
        backend_type: BackendType::Cpu,
        algorithm: SolverAlgorithm::Simplex,
        time_limit_secs: 10.0,
        max_iterations: 5000,
        tolerance: 1e-6,
        show_variables: true,
    };

    let runner = BenchmarkRunner::new(config)
        .with_reference_results(refs)
        .with_show_stages(false);
    let report = runner
        .run_directory(&afiro_path)
        .expect("Single-file Afiro run should succeed");

    assert_eq!(report.entries.len(), 1);
    let entry = &report.entries[0];
    assert_eq!(entry.instance, "afiro.mps");
    assert!(entry.status == "CONVERGED" || entry.status == "OPTIMAL");
    assert!((entry.objective - (-464.753142857143)).abs() < 1e-4);
    assert_eq!(entry.status_match, Some(true));
    assert!(entry.rel_objective_error.unwrap_or(1.0) < 1e-6);
}

#[test]
fn test_netlib_datasets_directory_collection() {
    let dir_path = workspace_root().join("datasets/netlib");
    assert!(dir_path.exists(), "datasets/netlib must exist");

    let config = BenchmarkConfig {
        backend_type: BackendType::Cpu,
        algorithm: SolverAlgorithm::Auto,
        time_limit_secs: 0.1, // very short timeout per instance for quick collection test
        max_iterations: 10,
        tolerance: 1e-4,
        show_variables: true,
    };

    let runner = BenchmarkRunner::new(config).with_show_stages(false);
    let report = runner
        .run_directory(&dir_path)
        .expect("Directory benchmark must succeed");

    assert_eq!(
        report.entries.len(),
        7,
        "All 7 Netlib datasets in datasets/netlib must be collected and evaluated"
    );

    let instance_names: Vec<String> = report.entries.iter().map(|e| e.instance.clone()).collect();
    assert!(instance_names.contains(&"afiro.mps".to_string()));
    assert!(instance_names.contains(&"agg.mps".to_string()));
    assert!(instance_names.contains(&"agg2.mps".to_string()));
    assert!(instance_names.contains(&"agg3.mps".to_string()));
    assert!(instance_names.contains(&"bandm.mps".to_string()));
    assert!(instance_names.contains(&"bnl1.mps".to_string()));
    assert!(instance_names.contains(&"bnl2.mps".to_string()));
}

#[test]
fn test_benchmark_single_mps_file_netlib_agg_exact_parity() {
    let agg_path = workspace_root().join("datasets/netlib/agg.mps");
    let ref_path = workspace_root().join("datasets/netlib/reference_netlib.csv");
    assert!(agg_path.exists(), "datasets/netlib/agg.mps must exist");
    assert!(ref_path.exists(), "reference_netlib.csv must exist");

    let refs = ReferenceResults::load_from_file(&ref_path).expect("Reference results must parse");

    let config = BenchmarkConfig {
        backend_type: BackendType::Cpu,
        algorithm: SolverAlgorithm::Simplex,
        time_limit_secs: 30.0,
        max_iterations: 10_000,
        tolerance: 1e-6,
        show_variables: true,
    };

    let runner = BenchmarkRunner::new(config)
        .with_reference_results(refs)
        .with_show_stages(false);
    let report = runner
        .run_directory(&agg_path)
        .expect("Single-file Agg run should succeed");

    assert_eq!(report.entries.len(), 1);
    let entry = &report.entries[0];
    assert_eq!(entry.instance, "agg.mps");
    assert_eq!(entry.rows, 488);
    assert_eq!(entry.columns, 163);
    assert_eq!(entry.nonzeros, 2410);
    assert_eq!(entry.status, "CONVERGED");
    // Netlib ground truth optimal value: -35991767.287
    assert!(
        (entry.objective - (-35991767.286577)).abs() < 1e-2,
        "Expected -35991767.287, got {}",
        entry.objective
    );
    assert_eq!(entry.status_match, Some(true));
    assert!(
        entry.runtime_secs < 5.0,
        "Runtime should be sub-5s, got {:.4}s",
        entry.runtime_secs
    );
    assert!(entry.variables.is_some());
    let vars = entry.variables.as_ref().unwrap();
    assert_eq!(vars.len(), 163);
    let y00204 = vars.iter().find(|(n, _)| n == "Y00204");
    assert!(y00204.is_some(), "Y00204 should be in solved variables");
    assert!((y00204.unwrap().1 - 232200.0).abs() < 1e-3);
}
