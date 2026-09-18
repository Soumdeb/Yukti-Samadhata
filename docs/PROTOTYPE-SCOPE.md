# Yutki-Samadhata: Prototype Scope & Boundaries

## 1. Executive Mission
**Yutki-Samadhata** is an indigenous sovereign prototype for GPU-accelerated Linear Programming (LP). The objective is to demonstrate a robust, verifiable, first-principles mathematical optimization engine designed from the ground up in Rust (with low-level C++/CUDA acceleration backends), serving as a sovereign technological foundation and alternative to closed commercial solvers (Gurobi, CPLEX, FICO Xpress).

---

## 2. In-Scope Components
This project is an engineering and mathematical demonstration prototype. The scope is strictly bounded to the following 12 foundational areas:

1. **Standard & General Linear Programming (LP)**
   - Continuous variables with arbitrary box constraints: $l_j \le x_j \le u_j$, where $l_j \in [-\infty, \infty)$ and $u_j \in (-\infty, \infty]$.
   - Linear inequality and equality constraints: $r^l_i \le (Ax)_i \le r^u_i$.
   - Linear objective function: $\min c^T x$ (or $\max c^T x$, normalized internally).

2. **Sparse Matrix Storage & Computation**
   - Compressed Sparse Row (CSR) for high-performance matrix-vector multiplications ($Ax$).
   - Compressed Sparse Column (CSC) for transposed matrix-vector multiplications ($A^T y$) and column-wise scaling.
   - Coordinate List (COO) for efficient model ingestion, parsing, and dynamic problem assembly.

3. **Primal-Dual Hybrid Gradient (PDHG / PDLP)**
   - First-order saddle-point method suited for massive-scale parallelization on modern SIMD/GPU hardware.
   - Adaptive step-size scheduling (Malitsky-Pock or Barzilai-Borwein style / spectral norm estimates).
   - Primal and dual averaging / restarts based on normalized duality gap and residual trajectory.

4. **CPU Execution Engine**
   - High-performance, portable multi-threaded CPU backend using contiguously allocated memory.
   - Vectorized inner products, matrix-vector products, and coordinate-wise proximal projections.

5. **GPU Acceleration Architecture**
   - Clean hardware-agnostic abstraction (`ComputeBackend` trait).
   - CUDA / GPU kernels for dense vector updates, reduction operations (norms, dot products), and SpMV ($Ax$ and $A^T y$).
   - Dynamic hardware discovery (`--backend cpu`, `--backend gpu`, `--backend auto`) with zero falsification of GPU timing or presence.

6. **Elementary Linear Presolve & Transformation Mapping**
   - Clean isolated variable/row passes: fixed variable removal, empty row/column elimination, zero-coefficient purging, duplicate constraint checks, and simple bound tightening.
   - Bidirectional coordinate and value mapping (`TransformationMap`) enabling exact recovery of original-space solutions.

7. **Numerical Health & Invariant Monitoring**
   - Continuous NaN, Inf, and denormal checks at every iteration.
   - Primal feasibility residual $\|Ax - b_s\|$, dual feasibility residual $\|A^T y + z - c\|$, and duality gap tracking.
   - Stagnation and slow-progress detectors triggering restarts or bounded step-size damping.

8. **Independent Solution Verifier**
   - A completely decoupled module verifying candidate solutions $(x, y, s)$ directly against the unscaled, raw original LP model.
   - Categorical verdict: `VALID`, `INVALID`, or `NUMERICALLY_UNCERTAIN`.

9. **Standard MPS File Ingestion**
   - Support for fixed-column and free-format MPS (Mathematical Programming System) files: `NAME`, `ROWS`, `COLUMNS`, `RHS`, `RANGES`, `BOUNDS`, and `ENDATA`.
   - Handling of constraint types (`N`, `L`, `G`, `E`) and variable bound types (`LO`, `UP`, `FX`, `FR`).

10. **Systematic Benchmarking Harness**
    - CLI command `yutki-samadhata benchmark <dir>` generating structured CSV and JSON outputs.
    - Transparent metric recording: objective value, iterations, wall-clock time, residual norms, backend utilized, and numerical flags without filtering or concealing failed instances.

11. **Structured Solver Logs & Event Traces**
    - Configurable logging modes (`--log human`, `--log json`, `--log quiet`).
    - Standardized lifecycle events: `ModelLoaded`, `ModelValidated`, `PresolveFinished`, `PdhgIteration`, `VerificationFinished`, `SolveFinished`.

12. **Sovereign CLI & Local Authentication**
    - Rich interactive terminal UI with dotted/ASCII visual styling.
    - Secure zero-network local authentication: Argon2id password hashing, single-use cryptographically secure recovery codes, local encrypted profile storage.

---

## 3. Explicitly Unsupported Scope & Non-Claims

To maintain scientific integrity and mathematical rigor, the following domains are strictly **unsupported** and **NOT claimed**:

1. **Full MILP is Unsupported**: No branch-and-bound, branch-and-cut, Gomory cuts, or integer feasibility algorithms are included. This prototype solves continuous LPs only.
2. **Full QP is Unsupported**: Convex quadratic objectives, quadratic constraints, and semidefinite cones are not supported.
3. **No Commercial Solver Parity**: This project does not claim feature parity or benchmark parity with mature industrial commercial solvers (Gurobi, CPLEX, FICO Xpress, or COIN-OR HiGHS). It is an indigenous first-principles engineering research prototype.
4. **Not a Production Industrial Solver**: Yutki-Samadhata is an experimental, sovereign numerical optimization engine prototype, not a certified industrial mission-critical production solver.
5. **No GPU Speedup Without Measurement**: GPU acceleration is never simulated or assumed. Any claims of speedup must be backed by empirical hardware wall-clock measurements comparing identical instances against CPU baselines.
6. **No Optimality Without Evidence**: The engine never reports a solution as optimal based solely on loop completion. Optimality is only claimed when primal, dual, and duality gap residuals strictly satisfy specified tolerances and pass independent verification.
7. **No Commercial Solver Wrappers**: CBC, HiGHS, SCIP, CPLEX, Gurobi, Xpress, and Google OR-Tools are strictly prohibited from being utilized. All numerical kernels are indigenous.
8. **No Remote/Cloud Dependencies**: Web servers, REST/gRPC endpoints, remote telemetry, and cloud databases are excluded. Execution is 100% local and sovereign.

