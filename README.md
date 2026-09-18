# Yutki-Samadhata (युक्ति-समाधाता)

<p align="center">
  <img src="https://img.shields.io/badge/Rust-2021_Edition-orange?logo=rust&style=flat-square" alt="Rust 2021" />
  <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License" />
  <img src="https://img.shields.io/badge/Platform-Linux_%7C_Windows_%7C_macOS-lightgrey?style=flat-square" alt="Platform" />
  <img src="https://img.shields.io/badge/Hardware-NVIDIA_CUDA_%2B_C%2B%2B17_SIMD-green?logo=nvidia&style=flat-square" alt="Hardware Acceleration" />
  <img src="https://img.shields.io/badge/Math_Engine-PDHG_%2F_PDLP_First--Order-purple?style=flat-square" alt="Math Core" />
  <img src="https://img.shields.io/badge/Tests-84_Passed-success?style=flat-square" alt="Test Status" />
  <img src="https://img.shields.io/badge/Policy-Zero_Falsification-darkred?style=flat-square" alt="Zero Falsification" />
</p>

```
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
```

> **Yutki-Samadhata** (*युक्ति-समाधाता* — Sanskrit for *Strategic Reasoner & Problem Resolver*) is an indigenous, first-principles, high-performance Linear Programming (LP) optimization engine written in pure Rust with pluggable C++17 SIMD and native NVIDIA CUDA acceleration. Built specifically for massive-scale sparse optimization, it implements the **Primal-Dual Hybrid Gradient (PDHG / PDLP)** algorithm accompanied by independent post-solve verification and zero-falsification hardware integrity.

---

## 📑 Table of Contents

- [Key Highlights](#-key-highlights)
- [System Architecture](#-system-architecture)
- [Workspace Crate Topology](#-workspace-crate-topology)
- [Mathematical Foundations (PDHG / PDLP)](#-mathematical-foundations-pdhg--pdlp)
- [Zero-Falsification Policy](#-zero-falsification-policy)
- [Getting Started & Installation](#-getting-started--installation)
- [Command-Line Interface (CLI)](#-command-line-interface-cli)
- [Rust API Quickstart](#-rust-api-quickstart)
- [Benchmarking Suite](#-benchmarking-suite)
- [Local Sovereign Authentication](#-local-sovereign-authentication)
- [Quality Gates & Verification](#-quality-gates--verification)
- [License & Authors](#-license--authors)

---

## ⚡ Key Highlights

- **First-Principles PDHG/PDLP Numerical Core:**
  First-order primal-dual optimization designed for hyper-scale sparse linear programs where interior-point methods (IPM) or simplex stall due to memory and factorization bottlenecks.
- **Pluggable Hardware Compute Backends:**
  Interchangeable backends implementing a unified `ComputeBackend` trait:
  - `CpuRustBackend`: Cache-coherent, multi-threaded native Rust vector and sparse operations.
  - `CpuCppBackend`: High-performance C++17 SIMD matrix kernels via zero-overhead C FFI.
  - `CudaBackend`: Native CUDA kernels with warp-synchronous reductions and scalar/warp CSR sparse matrix-vector multiplication (SpMV).
- **Strict Zero-Falsification Policy:**
  Uncompromising hardware integrity. When requesting `--backend gpu`, the engine genuinely probes CUDA runtime and physical devices. If no functional device is detected, it **fails fast** (`ERR_GPU_UNAVAILABLE`) rather than emitting synthetic or mocked results.
- **10-Step Sovereign Pipeline:**
  End-to-end disciplined pipeline comprising model ingestion, matrix conditioning fingerprinting, elementary presolve, Ruiz equilibration, Pock-Chambolle step sizing, PDHG iteration, numerical stagnation sentinels, invertible post-solve, and decoupled verification.
- **Non-Trusting Independent Verifier (`yutki-verifier`):**
  Does not trust the solver's internal convergence flag. Recomputes primal feasibility, row bounds, finite value sanity, and dual objective values against the original unscaled model to emit an uncompromised verdict: `VALID`, `NUMERICALLY_UNCERTAIN`, or `INVALID`.
- **Production MPS Parser:**
  Standard MPS parser supporting `NAME`, `ROWS` (`N`, `L`, `G`, `E`), `COLUMNS`, `RHS`, `RANGES`, and `BOUNDS` (`LO`, `UP`, `FX`, `FR`) with strict 1-based syntax error diagnostics.
- **Air-Gapped Sovereign Authentication (`yutki-auth`):**
  Zero-network local security subsystem using `Argon2id` password hashing and cryptographically secure 16-character recovery codes for session tracing and audit logging.

---

## 🏛 System Architecture

The engine executes optimization workloads through a 10-step pipeline:

```mermaid
flowchart TD
    subgraph Ingestion & Modeling
        MPS[MPS File / Analytical LP] --> Ingestion[Model Ingestion & Parser]
        Ingestion --> LPModel[General LinearProgram Model]
        LPModel --> Fingerprint[Numerical Fingerprinting & Dynamic Range]
    end

    subgraph Transformation & Presolve
        Fingerprint --> Presolve[Elementary Presolver]
        Presolve --> TransMap[Invertible Transformation Map]
        Presolve --> PresolvedLP[Presolved LP Model]
    end

    subgraph Preconditioning
        PresolvedLP --> Scaling[Ruiz Equilibration & Pock-Chambolle Preconditioner]
        Scaling --> ScaledLP[Equilibrated Matrix & Vectors]
    end

    subgraph Optimization Engine
        ScaledLP --> PDHGController[PDHG / PDLP Engine]
        PDHGController <--> BackendTrait[ComputeBackend Trait]
        BackendTrait <--> CpuRust[CpuRustBackend]
        BackendTrait <--> CpuCpp[CpuCppBackend (C++17 FFI)]
        BackendTrait <--> CudaGPU[CudaBackend (NVIDIA GPU)]
        PDHGController --> NumMon[Numerical Health & Stagnation Monitor]
        NumMon -.->|Adaptive Restart / Fallback| PDHGController
    end

    subgraph Postsolve & Verification
        PDHGController --> RawSol[Presolved Space Solution]
        RawSol --> Postsolve[Invertible Postsolve Restoration]
        Postsolve --> FinalSol[Original Space Primal/Dual Solution]
        FinalSol --> Verifier[Independent Solution Verifier]
        LPModel -.->|Raw Unscaled Problem| Verifier
    end

    subgraph Output & Telemetry
        Verifier --> Verdict{Verdict: VALID / UNCERTAIN / INVALID}
        Verdict --> TraceLog[Structured Solver Trace]
        Verdict --> Benchmark[Benchmark / CLI Telemetry]
    end
```

---

## 📦 Workspace Crate Topology

The repository is organized as a unified Cargo workspace containing 11 decoupled crates:

| Crate | Path | Responsibility |
| :--- | :--- | :--- |
| **`yutki-cli`** | [`crates/yutki-cli`](crates/yutki-cli) | Main binary (`yutki-samadhata`), interactive TUI dashboard, commands (`start`, `solve`, `benchmark`, `doctor`, `version`). |
| **`yutki-lp`** | [`crates/yutki-lp`](crates/yutki-lp) | Core PDHG/PDLP solver, iteration loop, adaptive restarts, ergodic averaging, termination criteria. |
| **`yutki-gpu`** | [`crates/yutki-gpu`](crates/yutki-gpu) | `ComputeBackend` trait, Rust CPU backend, native C++ FFI, CUDA runtime probe & GPU kernels. |
| **`yutki-sparse`** | [`crates/yutki-sparse`](crates/yutki-sparse) | High-performance `CooMatrix`, `CsrMatrix`, `CscMatrix` sparse formats and vector BLAS-1 primitives. |
| **`yutki-model`** | [`crates/yutki-model`](crates/yutki-model) | Sovereign `LinearProgram` representation, bounds, objective sense, and full-featured standard MPS parser. |
| **`yutki-numerics`** | [`crates/yutki-numerics`](crates/yutki-numerics) | Centralized tolerances, matrix conditioning fingerprints, Ruiz scaling, and NaN/Inf sentinels. |
| **`yutki-transform`**| [`crates/yutki-transform`](crates/yutki-transform)| Elementary presolve passes (fixed variables, zero coefficients, singleton tightening) and inverse post-solve. |
| **`yutki-verifier`** | [`crates/yutki-verifier`](crates/yutki-verifier)| Decoupled, non-trusting solution validator computing primal/dual violations on unscaled models. |
| **`yutki-auth`** | [`crates/yutki-auth`](crates/yutki-auth) | Air-gapped local Argon2id authentication, atomic JSON storage, and secure recovery codes. |
| **`yutki-bench`** | [`crates/yutki-bench`](crates/yutki-bench) | Batch benchmark runner, ground-truth reference comparator, and CSV/JSON reporting engine. |
| **`yutki-runtime`** | [`crates/yutki-runtime`](crates/yutki-runtime) | Structured event logging, telemetry traces, and diagnostic formatting. |

---

## 🧮 Mathematical Foundations (PDHG / PDLP)

### 1. General Continuous Linear Program
Yutki-Samadhata solves the general bounded linear program:
$$
\begin{aligned}
\min_{x \in \mathbb{R}^n} \quad & c^T x \\
\text{subject to} \quad & l_c \le A x \le u_c \\
& l_v \le x \le u_v
\end{aligned}
$$
where $A \in \mathbb{R}^{m \times n}$ is the sparse constraint matrix, $c \in \mathbb{R}^n$ is the cost vector, and $l_c, u_c \in (\mathbb{R} \cup \{\pm\infty\})^m$, $l_v, u_v \in (\mathbb{R} \cup \{\pm\infty\})^n$ represent constraint and variable bounds.

### 2. Minimax Saddle-Point Reformulation
Introducing auxiliary slacks $s \in [l_c, u_c]$ such that $A x - s = 0$, the problem is cast into an unconstrained saddle-point problem:
$$
\min_{x \in [l_v, u_v],\, s \in [l_c, u_c]} \max_{y \in \mathbb{R}^m} \quad \mathcal{L}(x, s, y) = c^T x + y^T (A x - s)
$$

### 3. Iteration Equations (Pock-Chambolle Steps)
With diagonal preconditioning step sizes $\tau \in \mathbb{R}_{++}^n$ (primal) and $\sigma \in \mathbb{R}_{++}^m$ (dual):
1. **Primal Extrapolation**:
   $$\bar{x}^k = x^k + \theta (x^k - x^{k-1}) \quad (\theta = 1.0)$$
2. **Dual Ascent & Projection**:
   $$y^{k+1} = \operatorname{proj}_{\mathcal{Y}} \left( y^k + \sigma \odot (A \bar{x}^k - \operatorname{proj}_{[l_c, u_c]}(A \bar{x}^k)) \right)$$
3. **Primal Descent & Box Projection**:
   $$x^{k+1} = \operatorname{proj}_{[l_v, u_v]} \left( x^k - \tau \odot (c + A^T y^{k+1}) \right)$$
4. **Ergodic Averaging**:
   Iterates are averaged over time ($x_{\text{avg}} = \frac{1}{K} \sum_{k=1}^K x^k$) to guarantee $O(1/K)$ convergence rates.

### 4. Convergence & Stopping Criteria
The solver terminates when normalized primal residual, dual residual, and duality gap fall below specified tolerances:
$$
\frac{\| A x - \operatorname{proj}_{[l_c, u_c]}(A x) \|_\infty}{1 + \|u_c - l_c\|_\infty} \le \varepsilon_{\text{primal}}, \quad
\frac{\| \operatorname{proj}_{[l_v, u_v]}(x - (c + A^T y)) - x \|_\infty}{1 + \|c\|_\infty} \le \varepsilon_{\text{dual}}, \quad
\frac{|c^T x - \text{DualObj}(y)|}{1 + |c^T x| + |\text{DualObj}(y)|} \le \varepsilon_{\text{gap}}
$$

---

## 🛡 Zero-Falsification Policy

Yutki-Samadhata enforces a strict, verifiable engineering code of ethics:

```
[ ZERO-FALSIFICATION POLICY ENFORCED ]
1. NEVER simulate, mock, or fake GPU compute execution.
2. NEVER emit synthetic execution timings or unmeasured speedups.
3. FAIL FAST with ERR_GPU_UNAVAILABLE if `--backend gpu` is requested without genuine hardware.
4. Transparently report fallback warnings when `--backend auto` defaults to CPU.
5. NEVER suppress or conceal failed or numerically divergent benchmark instances.
```

---

## 🚀 Getting Started & Installation

### Prerequisites
- **Rust Toolchain:** Stable Rust $\ge$ 1.80 (`rustup update stable`)
- **C++ Compiler:** C++17 compatible compiler (`g++`, `clang++`, or MSVC)
- **Optional CUDA Toolkit:** NVIDIA CUDA $\ge$ 11.8 with `nvcc` in `PATH` (for GPU acceleration)

### Building from Source

```bash
# Clone the repository
git clone https://github.com/your-username/yutki-samadhata.git
cd yutki-samadhata

# Build release binary (automatically compiles C++ kernels and detects CUDA if present)
cargo build --release

# The compiled binary is located at:
# target/release/yutki-samadhata
```

---

## 💻 Command-Line Interface (CLI)

The top-level binary `yutki-samadhata` provides both direct command execution and an interactive sovereign terminal interface.

### 1. Interactive Sovereign Dashboard
```bash
cargo run --release -- start
# Or after installing:
yutki-samadhata start
```
*Launches the dotted-border terminal UI with local Argon2id authentication, problem inspection, real-time solver traces, and system diagnostics.*

### 2. Direct MPS Solver
```bash
# Solve with automatic backend selection (GPU if available, fallback to CPU)
yutki-samadhata solve -f examples/demo/production.mps --backend auto

# Solve forcing CPU execution (Rust native + C++ SIMD)
yutki-samadhata solve -f examples/demo/production.mps --backend cpu

# Solve on NVIDIA GPU (fails fast if no CUDA hardware is present)
yutki-samadhata solve -f examples/demo/production.mps --backend gpu
```

### 3. Batch Benchmarking Engine
```bash
# Run benchmark across an MPS directory with external reference comparison and JSON/CSV export
yutki-samadhata benchmark examples/benchmark \
  --backend cpu \
  --reference examples/benchmark/reference_results.csv \
  --csv benchmark_report.csv \
  --json benchmark_report.json
```

### 4. System & Hardware Diagnostics ("Doctor")
```bash
yutki-samadhata doctor
```
*Probes OS environment, memory topology, CPU core count, C++ compiler presence, NVIDIA driver, CUDA device count, compute capability, and VRAM.*

### 5. Version Metadata
```bash
yutki-samadhata version
```

---

## 🔬 Rust API Quickstart

You can use Yutki-Samadhata's crates programmatically in your own Rust applications:

```rust
use std::path::Path;
use yutki_model::LinearProgram;
use yutki_lp::{PdhgSolver, PdhgOptions};
use yutki_gpu::CpuBackend;
use yutki_verifier::SolutionVerifier;
use yutki_numerics::NumericalTolerances;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Ingest model from standard MPS format
    let lp = LinearProgram::from_mps_file(Path::new("examples/demo/production.mps"))?;
    println!("Loaded model '{}' ({} rows, {} cols)", lp.name, lp.num_constraints(), lp.num_variables());

    // 2. Configure PDHG solver options
    let mut options = PdhgOptions::default();
    options.tolerances.primal_feasibility = 1e-6;
    options.tolerances.dual_feasibility = 1e-6;
    options.max_iterations = 10_000;

    // 3. Initialize compute backend
    let backend = CpuBackend::new();

    // 4. Solve the linear program
    let result = PdhgSolver::solve(&lp, &options, &backend)?;
    println!("Solver status : {:?}", result.status);
    println!("Optimal cost  : {:.6}", result.primal_objective);
    println!("Iterations    : {}", result.iterations);

    // 5. Audit the solution with decoupled verifier
    let verifier = SolutionVerifier::new(NumericalTolerances::default());
    let audit = verifier.verify(&lp, &result.x, &result.y, result.primal_objective);
    println!("Verification  : {:?}", audit.verdict); // VALID, NUMERICALLY_UNCERTAIN, or INVALID

    Ok(())
}
```

---

## 📊 Benchmarking Suite

The engine includes a built-in benchmark harness with reference ground-truth comparison:

```
+---------------------------------------------------------------------------------------------------------+
|                                    BENCHMARK EXECUTION SUMMARY                                          |
+---------------------------------------------------------------------------------------------------------+
| Instance         | Status          | Iters  | Time (ms) | Obj Error | Rel Error | Status Match | Warnings   |
+------------------+-----------------+--------+-----------+-----------+-----------+--------------+------------+
| production.mps   | CONVERGED       |    340 |     0.665 |  0.739152 | 3.144e-04 | PASS         | None       |
| infeasible.mps   | ITERATION_LIMIT |   6640 |     6.247 |  3.500147 | 3.500e+00 | FAIL (INF)   | Divergent  |
| malformed.mps    | PARSE_ERROR     |      0 |     0.196 |       N/A |       N/A | N/A          | Header     |
+---------------------------------------------------------------------------------------------------------+
Total Models: 3 | Converged: 1 | Failed: 2 | Total Execution: 7.11 ms
```

> **Note:** Failed, infeasible, or divergent models are **never hidden or omitted** from benchmark metrics.

---

## 🔐 Local Sovereign Authentication

Designed for air-gapped workstations and sensitive enterprise networks:
- **Zero Network Footprint:** No external databases, telemetry pings, or cloud dependencies.
- **Argon2id Key Derivation:** Cryptographically robust hashing using `argon2` with operating-system entropy (`rand::rngs::OsRng`).
- **Cryptographic Recovery Code:** Generates a 16-character alphanumeric recovery key (`XXXX-XXXX-XXXX-XXXX`) displayed once at account creation.
- **Atomic File Persistence:** Safe write-to-temporary with atomic file swap prevents corruption on unexpected shutdowns.

---

## 🧪 Quality Gates & Verification

All code in this repository adheres to zero-warning compilation and comprehensive testing:

```bash
# 1. Format check
cargo fmt --check

# 2. Workspace compilation
cargo check --workspace

# 3. Comprehensive test suite (84 unit and integration tests)
cargo test --workspace

# 4. Strict clippy linter
cargo clippy --workspace --all-targets -- -D warnings
```

---

## 📄 License & Authors

Distributed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE) for details.

Developed by the **Yutki-Samadhata Engineering Team**.
Designed for indigenous high-performance mathematical computing, numerical transparency, and extreme-scale linear optimization.
