# Yutki-Samadhata: Project Task Tracker

## Architectural & Design Foundations
- [x] Create comprehensive prototype scope document (`docs/PROTOTYPE-SCOPE.md`)
- [x] Formulate exact mathematical foundation for PDHG / PDLP (`docs/PDHG.md`)
- [x] Design GPU compute backend & zero-falsification policy (`docs/GPU-DESIGN.md`)
- [x] Specify numerical robustness invariants, tolerances & verification (`docs/NUMERICAL-DESIGN.md`)
- [x] Design local zero-network Argon2id authentication system (`docs/AUTH-DESIGN.md`)
- [x] Define master system architecture and crate decomposition (`docs/ARCHITECTURE.md`)
- [x] Detail phase-by-phase implementation plan with quality gates (`docs/IMPLEMENTATION-PLAN.md`)
- [x] Present architecture and implementation plan for user review

---

## Phase 1: Workspace Scaffolding, Core Models, Sparse Math & MPS Parser
- [x] Initialize Cargo workspace root `Cargo.toml`
- [x] Implement `yutki-sparse`:
  - [x] `CooMatrix` (triplet coordinate format)
  - [x] `CsrMatrix` (compressed sparse row format)
  - [x] `CscMatrix` (compressed sparse column format)
  - [x] Conversion functions (COO -> CSR, COO -> CSC)
  - [x] Contiguous memory vector and matrix allocation
  - [x] Sparse matrix unit tests and validation
- [x] Implement `yutki-model`:
  - [x] General LP data model (`LpProblem`, `Sense`, `Bound`)
  - [x] Standard MPS file parser (`NAME`, `ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS`)
  - [x] Support for continuous bounds (`LO`, `UP`, `FX`, `FR`)
  - [x] Model validation and consistency assertions
  - [x] Sample `examples/demo/production.mps` instance
  - [x] Model and MPS unit tests
- [x] Implement `yutki-cli` with `start`, `version`, `doctor` commands
- [x] Pass all quality gates: `cargo fmt --check`, `cargo check --workspace`, `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`

---

## Phase 3: Mathematical LP Model & Sparse Matrix Layer
- [x] Implement mathematical LP model components (`crates/yutki-model/src/linear_program.rs`):
  - [x] `LinearProgram` (sovereign LP container: $\min/\max c^T x$ s.t. $r^l \le A x \le r^u$, $c^l \le x \le c^u$)
  - [x] `Objective` (direction, cost vector $c$, scalar offset, objective evaluation)
  - [x] `VariableBounds` (bounds for each variable, projection, max violation)
  - [x] `ConstraintBounds` (bounds for each row constraint, projection, max violation)
  - [x] Validation: dimensions, indices, bounds ($l \le u$), NaN, and Inf detection
  - [x] Comprehensive unit tests (dimensions, inverted bounds, NaN bounds, NaN objective, roundtrip)
- [x] Implement sparse matrix layer (`crates/yutki-sparse/src/lib.rs`):
  - [x] Sparse formats: `CooMatrix`, `CsrMatrix`, `CscMatrix`
  - [x] Sparse operations: $A * x$, $A^T * y$ (both CSR and CSC implementations)
  - [x] Vector primitives: `dot(x, y)`, `norm_l1(x)`, `norm_l2(x)`, `norm_inf(x)`
  - [x] Format conversions: COO $\rightarrow$ CSR $\rightarrow$ CSC $\rightarrow$ COO round-trip
  - [x] Matrix statistics: non-zero distribution, row/col density, min/max absolute values, condition ratio
  - [x] Validation: dimensions, index out-of-bounds, monotonically sorted indices, NaN/Inf rejection
  - [x] Comprehensive unit tests across all sparse operations

## Phase 5: Numerical Robustness, Safe Restarts & Invertible Presolve
- [x] Implement `yutki-numerics`:
  - [x] Centralized `NumericalTolerances` structure & `TolerancePolicy`
  - [x] Matrix conditioning fingerprinting & dynamic range analysis
  - [x] NaN, Inf, and denormal sentinels (`check_finite_slice`, `NumericalMonitor`)
  - [x] `StagnationDetector` tracking residuals, objective change, step norms, and progress
  - [x] `RestartManager` with routine KKT acceleration and safe recovery to best-known iterate
  - [x] Ruiz matrix equilibration (iterative $L_\infty$ row/column scaling)
  - [x] Pock-Chambolle step-size preconditioning
  - [x] Numerics unit tests
- [x] Implement `yutki-transform`:
  - [x] Elimination of fixed variables ($l = u$) with row bound and objective shift adjustments
  - [x] Detection and removal of empty rows and infeasibility detection
  - [x] Zero coefficient purging
  - [x] Simple singleton row bound tightening and safe elimination
  - [x] `TransformationMap` tracking shifts, dropped indices, and reconstruction metadata
  - [x] Inverse `postsolve` transformation recovering original primal/dual coordinates
  - [x] Multi-pass presolve and transformation round-trip tests
  - [x] End-to-end integration test (original model -> presolve -> PDHG -> postsolve -> original solution)

---

## Hardware Compute Abstraction & C++ Numerical Backend (`yutki-gpu`)
- [x] Implement `yutki-gpu`:
  - [x] `ComputeBackend` trait definition
  - [x] `CpuRustBackend` implementation (multi-threaded, contiguous, cache-friendly)
  - [x] `CpuCppBackend` FFI implementation interfacing with high-performance native C++
  - [x] Native C++ numerical kernels (`cpp/cpu/sparse_ops.hpp`, `cpp/cpu/sparse_ops.cpp`)
  - [x] C ABI exports: `yutki_cpu_spmv`, `yutki_cpu_spmv_transpose`, `yutki_cpu_axpby`, `yutki_cpu_box_project`, `yutki_cpu_dot`, `yutki_cpu_norm2`, `yutki_cpu_norm_inf`
  - [x] Cargo C++ compilation integration via `cc` in `crates/yutki-gpu/build.rs`
  - [x] CUDA runtime discovery and detection mechanism
  - [x] Fail-fast validation against faked or simulated GPU execution (`ERR_GPU_UNAVAILABLE`)
  - [x] Numerical parity unit tests comparing Rust and C++ backend results (< 1e-14 discrepancy)

---

## Phase 8: Optional CUDA Support & Hardware Compute Architecture
- [x] Native CUDA kernels (`cuda/kernels/`):
  - [x] `cuda_kernels.h` (C API header and `YutkiCudaTiming` struct)
  - [x] `cuda_common.cuh` (warp-level and block-level reduction primitives)
  - [x] `spmv_csr.cu` (adaptive scalar and warp-per-row CSR SpMV)
  - [x] `spmv_csr_transpose.cu` (atomic accumulation CSR transpose SpMV)
  - [x] `vector_ops.cu` (grid-stride `axpby` and `box_project` kernels)
  - [x] `dot_reduction.cu` (warp-shuffle inner product reduction)
  - [x] `norm_reduction.cu` (L2 sum-of-squares and Linf maximum-absolute reductions)
- [x] Build script & FFI integration:
  - [x] Optional `nvcc` compilation via `build.rs` with `has_cuda_runtime` cfg
  - [x] FFI bindings for CUDA kernels in `crates/yutki-gpu/src/ffi.rs`
- [x] Clean Decoupled Architecture:
  - [x] PDHG interacts solely through `ComputeBackend` trait
  - [x] Zero CUDA code inside `PdhgSolver`
  - [x] `CudaBackend` implementation in `crates/yutki-gpu`
  - [x] `BackendTimingProfile` tracking host-to-device, kernel, device-to-host, and total runtime
  - [x] Persistent device memory management keeping vectors on device between iterations
- [x] Strict Zero-Falsification Policy:
  - [x] `--backend cpu`: Uses CPU without touching GPU
  - [x] `--backend gpu`: Fails fast with `ERR_GPU_UNAVAILABLE` if no compatible GPU exists
  - [x] `--backend auto`: Selects GPU only when genuinely functional, otherwise falls back gracefully to CPU
  - [x] Never simulate or mock GPU execution
  - [x] Never claim GPU speedup unless measured with monotonic high-precision timers
- [x] CLI & Interactive Dashboard Integration:
  - [x] `--backend <cpu|gpu|auto>` on top-level and `solve` subcommands
  - [x] `run_doctor` genuine dynamic probe of driver and devices
  - [x] Detailed timing breakdown display (H2D, Kernel, D2H, Total)

---

## Phase 4: CPU PDHG / PDLP Optimization Engine
- [x] Implement `yutki-lp`:
  - [x] Mathematical verification of `docs/PDHG.md` and saddle-point equations
  - [x] `PdhgSolver` CPU implementation using sparse matrix abstraction
  - [x] `PdhgState` maintaining iterates ($x, \bar{x}, x_{\text{avg}}, y, y_{\text{avg}}$) and step sizes
  - [x] `PdhgOptions` (tolerances, iteration/time limits, check frequency, damping)
  - [x] `PdhgResult` with solution vectors, objective, iterations, runtime, and residuals
  - [x] `ResidualCalculator` computing exact relative primal, dual, and duality gap residuals
  - [x] `TerminationPolicy` with statuses (`Converged`, `IterationLimit`, `TimeLimit`, `NumericalFailure`, `Unsupported`)
  - [x] Pock-Chambolle diagonal step sizes ($\tau, \sigma$) and primal extrapolation ($\bar{x} = 2 x^{k+1} - x^k$)
  - [x] Ergodic iterate averaging and adaptive restarts
  - [x] Comprehensive analytical LP unit tests with known optimal solutions
  - [x] Integration with CLI dashboard (`Solve Demo LP` and `Solve MPS File`)
- [x] Implement `yutki-verifier`:
  - [x] Decoupled `SolutionVerifier`
  - [x] Independent finite-value and bounds check
  - [x] Independent row feasibility check ($A x$)
  - [x] Recomputation of objective value
  - [x] Verdict emission (`VALID`, `NUMERICALLY_UNCERTAIN`, `INVALID`)
  - [x] Verifier unit tests with corrupted solution tests (wrong variable, violated constraint, bound violation, NaN, Inf)

---

## Local Sovereign Authentication Subsystem
- [x] Implement `yutki-auth`:
  - [x] Local JSON credential store with atomic disk persistence
  - [x] Argon2id password hashing with `rand::rngs::OsRng` salt
  - [x] Cryptographically secure 16-character recovery code generation and Argon2id hashing
  - [x] Signup workflow with one-time recovery code display
  - [x] Login workflow with timing-safe hash comparison
  - [x] Forgot-password workflow via recovery code verification
  - [x] In-memory session state management and clean logout
  - [x] Authentication unit tests (signup, duplicate username, wrong password, login, wrong recovery code, password reset, corrupted auth file, short password)

---

## Phase 9: Practical MPS Parser & Demonstration LP
- [x] Implement robust MPS parser in `crates/yutki-model/src/mps.rs`:
  - [x] Support all standard sections: `NAME`, `ROWS`, `COLUMNS`, `RHS`, `BOUNDS`, `ENDATA`
  - [x] Support all LP row types: `N` (free/objective), `L` ($\le$), `G` ($\ge$), `E` ($=$)
  - [x] Support all LP continuous bound types: `LO`, `UP`, `FX`, `FR`
  - [x] Provide descriptive error messages with 1-based source line numbers (`SolverError::ParseError`)
  - [x] Direct conversion functions: `parse_mps_file_to_lp`, `parse_mps_str_to_lp`
  - [x] Direct constructors on `LinearProgram`: `LinearProgram::from_mps_file`, `LinearProgram::from_mps_str`
- [x] Create analytical demonstration LP in `examples/demo/production.mps`:
  - [x] 3 variables (`CHAIRS`, `TABLES`, `DESKS`), 4 constraints (`LUMBER`, `CARPENTRY`, `FINISHING`, `DEMAND_T`)
  - [x] Exact analytical optimal primal solution: $(10, 10, 10)$
  - [x] Exact analytical optimal dual solution: $y = (0.875, 28.125, 6.25, 0.0)$
  - [x] Optimal objective value: $\text{profit} = 2350.0$, minimization cost = $-2350.0$
- [x] Comprehensive test suite:
  - [x] Valid MPS parsing with all bound types (`LO`, `UP`, `FX`, `FR`)
  - [x] Roundtrip direct conversion into `LinearProgram`
  - [x] Analytical solution verification for `examples/demo/production.mps`
  - [x] 10 malformed MPS rejection tests (unknown headers, invalid row types, duplicate rows, undefined columns/rows, malformed numbers, inverted bounds, missing sections)
- [x] Pass all quality gates: `cargo fmt --check`, `cargo check --workspace`, `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`

---

## Phase 10: Standalone Benchmark Suite & Reference Comparison (`yutki-bench`)
- [x] Implement `yutki-bench` library:
  - [x] `BenchmarkEntry` recording: instance, rows, columns, nonzeros, objective, status, runtime, iterations, primal residual, dual residual, backend, numerical warnings
  - [x] Transparent zero-falsification execution: failed/infeasible instances are never dropped or hidden
  - [x] Performance metrics measured with monotonic high-precision timers (`Instant::now()`)
  - [x] CSV export (`export_csv`) with properly formatted and escaped headers and records
  - [x] JSON export (`export_json`) formatted as pretty JSON
  - [x] External reference ground-truth loader (`ReferenceResults`) supporting CSV and JSON formats
  - [x] Objective error calculation: absolute difference and relative error $|\text{obj} - \text{ref\_obj}| / (1 + |\text{ref\_obj}|)$
  - [x] Status concordance check (`status_match`)
  - [x] Batch directory runner (`BenchmarkRunner`) scanning `.mps` instances deterministically
  - [x] Structured ASCII summary table (`print_summary_table`) with reference comparison matrix
- [x] CLI Subcommand (`yutki-cli`):
  - [x] `yutki-samadhata benchmark <directory>`
  - [x] Backend selection: `--backend cpu`, `--backend gpu`, `--backend auto`
  - [x] Custom export flags: `--csv <path>`, `--json <path>`
  - [x] Reference results flag: `--reference <path>`
  - [x] Solves models purely with internal sovereign `PdhgSolver` (no external solver invocation)
- [x] Interactive Dashboard:
  - [x] Wired option `[3] Run Benchmark` with directory, reference file, and backend prompts
- [x] Test Suite & Benchmark Assets:
  - [x] `examples/benchmark/` test collection (`production.mps`, `infeasible.mps`, `malformed.mps`, `reference_results.csv`, `reference_results.json`)
  - [x] Unit and integration tests in `crates/yutki-bench/src/lib.rs` verifying CSV/JSON round-trips, failure visibility, and ground-truth comparison
- [x] Pass all quality gates: `cargo fmt --check`, `cargo check --workspace`, `cargo test --workspace`, `cargo clippy --workspace --all-targets -- -D warnings`

---

## Phase 7: Sovereign Terminal Interface (`yutki-cli`) & Final Integration
- [x] Implement `yutki-cli`:
  - [x] Dotted/ASCII terminal visual styling (Yukti Samadhata sovereign visual identity)
  - [x] Top-level authentication loop (Login, Signup, Forgot Password, Exit) with Argon2id and atomic file storage
  - [x] Solver dashboard (Solve Demo LP, Solve MPS File, Run Benchmark, View Last Solver Trace, System/GPU Info, Logout)
  - [x] Full 10-step solver execution pipeline:
    - [x] Model Validation (dimensions, bounds sanity, NaN/Inf checks)
    - [x] Problem Fingerprint (density, min/max nonzeros, dynamic range ratio, conditioning rating)
    - [x] Basic Presolve (fixed variable elimination, redundant row removal, singleton tightening)
    - [x] Backend Selection (auto, cpu, gpu with zero-falsification enforcement)
    - [x] PDHG Optimization Engine (Pock-Chambolle step sizes, extrapolation, iterate averaging)
    - [x] Numerical Health Monitor (finite value sentinel, residuals, duality gap, stability)
    - [x] Invertible Postsolve (coordinate mapping, fixed variable restoration, shadow prices)
    - [x] Independent Verification (`SolutionVerifier` non-trusting audit, bounds/Ax check, verdict `VALID`)
    - [x] Final Solution Presentation (optimal objective, variables, slacks, shadow prices, telemetry)
    - [x] Solver Trace Recording (retained in session memory, viewable under Option 4)
  - [x] Standalone CLI commands (`start`, `solve`, `benchmark`, `doctor`, `version`)
  - [x] Final quality gates: `cargo fmt`, `cargo check`, `cargo test`, `cargo clippy -- -D warnings`
  - [x] Final project walkthrough documentation
