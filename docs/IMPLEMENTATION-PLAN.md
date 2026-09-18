# Yutki-Samadhata: Step-by-Step Implementation Plan

## 1. Phased Development Methodology
To maintain absolute stability, each phase must follow a strict lifecycle before proceeding to the next:
1. **Design & Scope Review**: Clarify types, traits, and error boundaries.
2. **Implementation**: Code only the target components.
3. **Automated Unit & Integration Tests**: Verify mathematical and algorithmic correctness.
4. **Code Quality Gates**:
   - `cargo fmt --all -- --check`
   - `cargo clippy --all-targets -- -D warnings`
   - `cargo test --all`
5. **Phase Verification Artifact**: Document evidence of tests and benchmark results.

---

## 2. Phase Breakdown

### Phase 1: Workspace Scaffolding, Core Model Representation, Sparse Math & MPS Parser
- **Target Crates:** `yutki-model`, `yutki-sparse`
- **Deliverables:**
  - Root `Cargo.toml` with workspace declaration.
  - `yutki-sparse`: `CooMatrix`, `CsrMatrix`, `CscMatrix` with conversion logic, contiguous storage, memory validation, and basic vector arithmetic.
  - `yutki-model`: General LP model struct (`LpProblem`), variable and row bound representations (`Bound`), direction (`Minimize`/`Maximize`).
  - Standard MPS parser supporting `NAME`, `ROWS`, `COLUMNS`, `RHS`, `RANGES`, and `BOUNDS` with support for continuous variable bounds (`LO`, `UP`, `FX`, `FR`).
  - Demo problem generator (`production.mps` sample LP).
- **Quality Gates:** Unit tests for MPS parsing, sparse matrix conversion round-trips, and matrix-vector dimension checks.

### Phase 2: Numerical Foundation, Preconditioning & Invertible Presolve
- **Target Crates:** `yutki-numerics`, `yutki-transform`
- **Deliverables:**
  - `yutki-numerics`:
    - Centralized `NumericalTolerances` configuration.
    - Dynamic range and problem conditioning fingerprinting.
    - NaN and Inf detection sentinels.
    - Ruiz matrix equilibration algorithm (iterative row/column scaling).
    - Pock-Chambolle step-size preconditioning.
  - `yutki-transform`:
    - Elementary presolve rules:
      - Elimination of fixed variables.
      - Detection of empty/zero rows and columns.
      - Cleaning of explicit zero coefficients.
      - Simple singleton row bound tightening.
    - `TransformationMap` capturing all column/row index changes, scale factors, and objective offsets.
    - Postsolve inverse mapping restoring candidate solutions back into the original LP coordinate space.
- **Quality Gates:** Unit tests verifying that presolve transforms followed by postsolve preserve exact solution equivalence on standard test problems.

### Phase 3: Hardware Compute Abstraction & CPU/GPU Execution Engines
- **Target Crate:** `yutki-gpu`
- **Deliverables:**
  - `ComputeBackend` trait defining uniform interfaces for vector allocation, buffer transfer, SpMV ($Ax$), transpose SpMV ($A^T y$), axpby, box projections, dot products, and norm reductions.
  - `CpuBackend`: High-performance, portable implementation with contiguous arrays and SIMD/multi-threaded capabilities.
  - `CudaBackend`: CUDA driver/runtime discovery, memory allocation primitives, and GPU kernel interfaces.
  - Automatic hardware detection (`BackendSelector`) for `--backend cpu`, `--backend gpu`, and `--backend auto`.
  - Fail-fast validation ensuring no synthetic or simulated GPU execution can ever occur.
- **Quality Gates:** Parity tests verifying that `CpuBackend` and mock/actual GPU execution produce numerically identical SpMV, dot products, and projections within floating-point epsilon.

### Phase 4: PDHG / PDLP Optimization Engine & Independent Verifier
- **Target Crates:** `yutki-lp`, `yutki-verifier`
- **Deliverables:**
  - `yutki-lp`:
    - Full mathematical PDHG algorithm with primal-dual updates and extrapolation.
    - Adaptive step-size scheduling.
    - Primal and dual ergodic iterate averaging.
    - Periodic normalized duality gap and residual checking with adaptive restart heuristics.
    - Stagnation and iteration/time limit safeguards.
  - `yutki-verifier`:
    - Decoupled `SolutionVerifier` operating on original unscaled LP models.
    - Precise check of finite values, variable bounds, constraint bounds ($Ax$), and recomputed objective.
    - Conclusive verdict generation (`VALID`, `NUMERICALLY_UNCERTAIN`, `INVALID`).
- **Quality Gates:** Solve test suite on synthetic and benchmark LP models verifying convergence to known optimal solutions and validation by `SolutionVerifier`.

### Phase 5: Local Sovereign Authentication Subsystem
- **Target Crate:** `yutki-auth`
- **Deliverables:**
  - Local credential store using JSON file with atomic write semantics.
  - Argon2id password hashing with cryptographically secure random salt (`rand::rngs::OsRng`).
  - Signup flow generating single-use cryptographically secure 16-character recovery codes (stored strictly as Argon2id hashes).
  - Login authentication with timing-safe comparison.
  - Forgot-password flow verifying recovery code hash and updating password.
  - Session lifecycle management (in-memory token, clean logout).
- **Quality Gates:** Unit tests for registration, authentication, invalid passwords, recovery code verification, replay prevention, and storage persistence.

### Phase 6: Runtime Lifecycle, Structured Trace Logging & Benchmarking Harness
- **Target Crates:** `yutki-runtime`, `yutki-bench`
- **Deliverables:**
  - `yutki-runtime`:
    - Event-driven logging infrastructure (`ModelLoaded`, `PresolveFinished`, `PdhgIteration`, `PdhgRestart`, `VerificationFinished`, `SolveFinished`).
    - Multiple output formatters: human-readable tabular ASCII, machine-readable JSON (NDJSON), and quiet mode.
  - `yutki-bench`:
    - Batch benchmark runner scanning directories of `.mps` files.
    - Metrics collection: problem dimensions, nonzeros, iterations, residuals, objective, wall-clock time, backend used, verification status.
    - CSV and JSON report exporters.
- **Quality Gates:** Benchmark execution on demo instances with verified CSV and JSON export output.

### Phase 7: Sovereign Terminal CLI & System Integration
- **Target Crate:** `yutki-cli`
- **Deliverables:**
  - Main binary entry point: `yutki-samadhata`.
  - Subcommands: `yutki-samadhata start`, `yutki-samadhata solve <file.mps>`, `yutki-samadhata benchmark <dir>`, `yutki-samadhata info`.
  - Dotted/ASCII terminal interface inspired by Yukti Samadhata design ethos.
  - Interactive top-level menu (Login, Signup, Forgot Password, Exit).
  - Interactive dashboard (Solve Demo LP, Solve MPS File, Run Benchmark, View Last Solver Trace, System/GPU Info, Logout).
  - Complete end-to-end integration tests.
- **Quality Gates:** End-to-end interactive and non-interactive CLI tests, full Clippy and formatting pass.
