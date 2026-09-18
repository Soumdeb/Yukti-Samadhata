# Yutki-Samadhata: GPU Backend Architecture & Design

## 1. Architectural Principles

1. **Clean Separation of Concerns:**
   - PDHG is the mathematical optimization algorithm.
   - GPU is a pluggable hardware compute backend.
   - The PDHG solver core interacts exclusively through the `ComputeBackend` trait and never directly issues CUDA API calls or kernel launches.

2. **Absolute Transparency & Integrity (No Mock/Faked GPU Execution):**
   - If `--backend gpu` is requested and no genuine CUDA device is present, initialized, and functional, the engine **must fail fast** with a clear diagnostic error.
   - The engine **never** fakes GPU execution, **never** reports synthetic speedups, and **never** records mocked GPU timestamps.
   - When `--backend auto` is chosen, genuine device queries determine whether to select GPU or fallback to CPU with an explicit structured diagnostic warning.

---

## 2. ComputeBackend Abstraction (Rust Trait)

```rust
pub trait ComputeBackend: Send + Sync {
    /// Return the name and hardware identification of this backend
    fn name(&self) -> &'static str;
    
    /// Hardware details (e.g., CPU thread count or GPU Device Name, SM count, VRAM)
    fn device_info(&self) -> BackendInfo;

    /// Allocate or upload vector to the compute device
    fn create_vector(&self, initial: &[f64]) -> Result<Box<dyn VectorBuffer>, SolverError>;

    /// Allocate or upload sparse matrix in CSR format
    fn create_csr_matrix(&self, csr: &CsrMatrix) -> Result<Box<dyn MatrixBuffer>, SolverError>;

    /// Perform y = alpha * A * x + beta * y (SpMV)
    fn spmv(
        &self, 
        alpha: f64, 
        a: &dyn MatrixBuffer, 
        x: &dyn VectorBuffer, 
        beta: f64, 
        y: &mut dyn VectorBuffer
    ) -> Result<(), SolverError>;

    /// Perform x = alpha * A^T * y + beta * x (Transpose SpMV)
    fn spmv_transpose(
        &self, 
        alpha: f64, 
        a: &dyn MatrixBuffer, 
        y: &dyn VectorBuffer, 
        beta: f64, 
        x: &mut dyn VectorBuffer
    ) -> Result<(), SolverError>;

    /// Vector update: z = alpha * x + beta * y
    fn axpby(
        &self, 
        alpha: f64, 
        x: &dyn VectorBuffer, 
        beta: f64, 
        y: &dyn VectorBuffer, 
        z: &mut dyn VectorBuffer
    ) -> Result<(), SolverError>;

    /// Elementwise box projection: x = proj_{[lower, upper]}(x)
    fn box_project(
        &self, 
        x: &mut dyn VectorBuffer, 
        lower: &dyn VectorBuffer, 
        upper: &dyn VectorBuffer
    ) -> Result<(), SolverError>;

    /// Dot product: <x, y>
    fn dot(&self, x: &dyn VectorBuffer, y: &dyn VectorBuffer) -> Result<f64, SolverError>;

    /// Euclidean (L2) norm: ||x||_2
    fn norm2(&self, x: &dyn VectorBuffer) -> Result<f64, SolverError>;

    /// Maximum (L-infinity) norm: ||x||_inf
    fn norm_inf(&self, x: &dyn VectorBuffer) -> Result<f64, SolverError>;

    /// Synchronize device operations (no-op on synchronous CPU)
    fn synchronize(&self) -> Result<(), SolverError>;
}
```

---

## 3. Backend Implementations

### 3.1 CPU Backend (`CpuBackend`)
- **Memory Layout:** Contiguous `Vec<f64>` / aligned heap buffers to maximize cache locality and automatic SIMD auto-vectorization (AVX2 / AVX-512 / NEON).
- **Parallelism:** Multi-threaded row-partitioning for large sparse matrices using Rayon or thread pool worker slices.
- **Transposed Multiply:** Uses pre-indexed CSC representation for $A^T y$ to prevent random scatter writes and cache thrashing.
- **Availability:** 100% available on all platforms (Windows, Linux, macOS) without external driver requirements.

### 3.2 CUDA / GPU Backend (`CudaBackend`)
- **Native Interop:** Implemented via low-level C FFI or direct CUDA driver/runtime bindings (`cudaMalloc`, `cudaMemcpyAsync`, `cudaStreamSynchronize`).
- **Memory Management:** Managed device memory pointers (`DeviceBuffer<f64>`) allocated once during problem loading to eliminate allocation overhead inside the iterative loop.
- **Kernel Architecture:**
  1. `spmv_csr_kernel`: Warp-aggregated or thread-per-row SpMV based on matrix average row density.
  2. `spmv_csc_transpose_kernel`: Thread-per-column transpose SpMV using CSR column indices.
  3. `axpby_project_kernel`: Fused vector update and box projection to minimize memory bandwidth (PDHG is memory-bandwidth bound on GPUs).
  4. `reduction_kernels`: Parallel tree reduction in shared memory for norms and inner products.
- **Stream Concurrency:** Asynchronous kernel launches along a dedicated CUDA stream with host synchronization strictly isolated to residual checking checkpoints.

---

## 4. Hardware Discovery & Mode Selection

### 4.1 CLI Backend Flags
- `--backend cpu`: Force CPU backend. Never attempts GPU initialization.
- `--backend gpu`: Require GPU backend.
  - Checks: CUDA runtime dynamic library loading, driver capability check (`cuInit`), device count $> 0$, compute capability $\ge 6.0$, allocation test.
  - If any check fails: Engine outputs error `ERR_GPU_UNAVAILABLE` with exact diagnostics (e.g., `CUDA driver not found`, `No CUDA devices found`) and halts.
- `--backend auto`: Intelligent probe.
  - Attempts GPU discovery.
  - If successful, selects `CudaBackend` and outputs `BackendSelected { backend: "CUDA", device: "NVIDIA RTX ..." }`.
  - If unsuccessful, logs `BackendSelected { backend: "CPU", reason: "CUDA initialization failed: <details>" }` and executes seamlessly on `CpuBackend`.

---

## 5. Performance Instrumentation
- High-precision monotonic timers (`std::time::Instant`) measure:
  - Memory upload time (Host-to-Device).
  - Iteration loop time (GPU execution + periodic stream sync).
  - Memory download time (Device-to-Host).
- Trace logs explicitly differentiate between **wall-clock solve time**, **pure iteration time**, and **data transfer overhead**.
