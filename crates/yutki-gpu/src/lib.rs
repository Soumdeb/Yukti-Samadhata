//! Hardware compute backend abstraction for CPU and genuine CUDA GPU execution.
//! Strict rule: GPU execution is NEVER simulated or faked.
//! C++ numerical backend provides high-performance low-level kernels via C ABI.
//! CUDA backend provides genuine GPU accelerated kernels for sparse linear algebra.

pub mod ffi;

use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use std::time::Instant;
use yutki_model::{BackendType, Bound, SolverError};
use yutki_sparse::CsrMatrix;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BackendInfo {
    pub name: String,
    pub is_gpu: bool,
    pub device_name: String,
    pub details: String,
}

/// Timing profile capturing genuine host-to-device, kernel, and device-to-host durations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct BackendTimingProfile {
    pub h2d_transfer_secs: f64,
    pub kernel_execution_secs: f64,
    pub d2h_transfer_secs: f64,
    pub total_runtime_secs: f64,
}

/// Unified compute backend trait for linear algebra operations used in PDHG.
pub trait ComputeBackend: Send + Sync {
    fn name(&self) -> &'static str;
    fn device_info(&self) -> BackendInfo;
    fn is_gpu(&self) -> bool {
        false
    }

    /// Retrieve recorded timing breakdown for data transfers and kernel execution.
    fn timing_profile(&self) -> BackendTimingProfile {
        BackendTimingProfile::default()
    }

    /// Reset timing accumulators.
    fn reset_timing(&self) {}

    /// SpMV: y = alpha * A * x + beta * y
    fn spmv(
        &self,
        alpha: f64,
        a: &CsrMatrix,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    ) -> Result<(), SolverError>;

    /// Transpose SpMV: x = alpha * A^T * y + beta * x
    fn spmv_transpose(
        &self,
        alpha: f64,
        a: &CsrMatrix,
        y: &[f64],
        beta: f64,
        x: &mut [f64],
    ) -> Result<(), SolverError>;

    /// Vector update: z = alpha * x + beta * y
    fn axpby(
        &self,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &[f64],
        z: &mut [f64],
    ) -> Result<(), SolverError>;

    /// Coordinate-wise projection into box bounds [lower, upper]
    fn box_project(&self, x: &mut [f64], bounds: &[Bound]) -> Result<(), SolverError>;

    /// Inner product <x, y>
    fn dot(&self, x: &[f64], y: &[f64]) -> Result<f64, SolverError>;

    /// Euclidean (L2) norm
    fn norm2(&self, x: &[f64]) -> Result<f64, SolverError>;

    /// Infinity norm
    fn norm_inf(&self, x: &[f64]) -> Result<f64, SolverError>;
}

/// Portable, native Rust CPU compute backend.
#[derive(Debug, Default)]
pub struct CpuRustBackend {
    timing: Mutex<BackendTimingProfile>,
}

impl CpuRustBackend {
    pub fn new() -> Self {
        Self {
            timing: Mutex::new(BackendTimingProfile::default()),
        }
    }
}

impl ComputeBackend for CpuRustBackend {
    fn name(&self) -> &'static str {
        "CPU (Rust)"
    }

    fn device_info(&self) -> BackendInfo {
        BackendInfo {
            name: "CPU (Rust Native)".into(),
            is_gpu: false,
            device_name: "Host Processor".into(),
            details: format!(
                "Logical cores: {}",
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            ),
        }
    }

    fn timing_profile(&self) -> BackendTimingProfile {
        self.timing.lock().map(|t| t.clone()).unwrap_or_default()
    }

    fn reset_timing(&self) {
        if let Ok(mut t) = self.timing.lock() {
            *t = BackendTimingProfile::default();
        }
    }

    fn spmv(
        &self,
        alpha: f64,
        a: &CsrMatrix,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    ) -> Result<(), SolverError> {
        let start = Instant::now();
        let res = a
            .spmv(alpha, x, beta, y)
            .map_err(|_| SolverError::DimensionMismatch {
                expected: a.num_cols,
                found: x.len(),
            });
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        res
    }

    fn spmv_transpose(
        &self,
        alpha: f64,
        a: &CsrMatrix,
        y: &[f64],
        beta: f64,
        x: &mut [f64],
    ) -> Result<(), SolverError> {
        let start = Instant::now();
        let res = a
            .spmv_transpose(alpha, y, beta, x)
            .map_err(|_| SolverError::DimensionMismatch {
                expected: a.num_rows,
                found: y.len(),
            });
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        res
    }

    fn axpby(
        &self,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &[f64],
        z: &mut [f64],
    ) -> Result<(), SolverError> {
        if x.len() != y.len() || x.len() != z.len() {
            return Err(SolverError::DimensionMismatch {
                expected: x.len(),
                found: y.len(),
            });
        }
        let start = Instant::now();
        for i in 0..x.len() {
            z[i] = alpha * x[i] + beta * y[i];
        }
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(())
    }

    fn box_project(&self, x: &mut [f64], bounds: &[Bound]) -> Result<(), SolverError> {
        if x.len() != bounds.len() {
            return Err(SolverError::DimensionMismatch {
                expected: bounds.len(),
                found: x.len(),
            });
        }
        let start = Instant::now();
        for (xi, b) in x.iter_mut().zip(bounds.iter()) {
            *xi = b.project(*xi);
        }
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(())
    }

    fn dot(&self, x: &[f64], y: &[f64]) -> Result<f64, SolverError> {
        if x.len() != y.len() {
            return Err(SolverError::DimensionMismatch {
                expected: x.len(),
                found: y.len(),
            });
        }
        let start = Instant::now();
        let sum: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(sum)
    }

    fn norm2(&self, x: &[f64]) -> Result<f64, SolverError> {
        let start = Instant::now();
        let sum_sq: f64 = x.iter().map(|&v| v * v).sum();
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(sum_sq.sqrt())
    }

    fn norm_inf(&self, x: &[f64]) -> Result<f64, SolverError> {
        let start = Instant::now();
        let max_val = x.iter().map(|&v| v.abs()).fold(0.0f64, f64::max);
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(max_val)
    }
}

/// Backwards compatibility alias: CpuBackend defaults to CpuRustBackend.
pub type CpuBackend = CpuRustBackend;

/// Optimized C++ CPU compute backend invoking low-level C ABI kernels.
#[derive(Debug, Default)]
pub struct CpuCppBackend {
    timing: Mutex<BackendTimingProfile>,
}

impl CpuCppBackend {
    pub fn new() -> Self {
        Self {
            timing: Mutex::new(BackendTimingProfile::default()),
        }
    }
}

impl ComputeBackend for CpuCppBackend {
    fn name(&self) -> &'static str {
        "CPU (C++)"
    }

    fn device_info(&self) -> BackendInfo {
        BackendInfo {
            name: "CPU (C++ Native)".into(),
            is_gpu: false,
            device_name: "Host Processor (C++ Optimized)".into(),
            details: format!(
                "Compiled C++17 kernels; Logical cores: {}",
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            ),
        }
    }

    fn timing_profile(&self) -> BackendTimingProfile {
        self.timing.lock().map(|t| t.clone()).unwrap_or_default()
    }

    fn reset_timing(&self) {
        if let Ok(mut t) = self.timing.lock() {
            *t = BackendTimingProfile::default();
        }
    }

    fn spmv(
        &self,
        alpha: f64,
        a: &CsrMatrix,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    ) -> Result<(), SolverError> {
        if x.len() != a.num_cols {
            return Err(SolverError::DimensionMismatch {
                expected: a.num_cols,
                found: x.len(),
            });
        }
        if y.len() != a.num_rows {
            return Err(SolverError::DimensionMismatch {
                expected: a.num_rows,
                found: y.len(),
            });
        }

        let start = Instant::now();
        unsafe {
            ffi::yutki_cpu_spmv(
                a.num_rows,
                a.num_cols,
                a.row_ptrs.as_ptr(),
                a.col_indices.as_ptr(),
                a.values.as_ptr(),
                alpha,
                x.as_ptr(),
                beta,
                y.as_mut_ptr(),
            );
        }
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(())
    }

    fn spmv_transpose(
        &self,
        alpha: f64,
        a: &CsrMatrix,
        y: &[f64],
        beta: f64,
        x: &mut [f64],
    ) -> Result<(), SolverError> {
        if y.len() != a.num_rows {
            return Err(SolverError::DimensionMismatch {
                expected: a.num_rows,
                found: y.len(),
            });
        }
        if x.len() != a.num_cols {
            return Err(SolverError::DimensionMismatch {
                expected: a.num_cols,
                found: x.len(),
            });
        }

        let start = Instant::now();
        unsafe {
            ffi::yutki_cpu_spmv_transpose(
                a.num_rows,
                a.num_cols,
                a.row_ptrs.as_ptr(),
                a.col_indices.as_ptr(),
                a.values.as_ptr(),
                alpha,
                y.as_ptr(),
                beta,
                x.as_mut_ptr(),
            );
        }
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(())
    }

    fn axpby(
        &self,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &[f64],
        z: &mut [f64],
    ) -> Result<(), SolverError> {
        if x.len() != y.len() || x.len() != z.len() {
            return Err(SolverError::DimensionMismatch {
                expected: x.len(),
                found: y.len(),
            });
        }

        let start = Instant::now();
        unsafe {
            ffi::yutki_cpu_axpby(x.len(), alpha, x.as_ptr(), beta, y.as_ptr(), z.as_mut_ptr());
        }
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(())
    }

    fn box_project(&self, x: &mut [f64], bounds: &[Bound]) -> Result<(), SolverError> {
        if x.len() != bounds.len() {
            return Err(SolverError::DimensionMismatch {
                expected: bounds.len(),
                found: x.len(),
            });
        }

        let lower: Vec<f64> = bounds.iter().map(|b| b.lower).collect();
        let upper: Vec<f64> = bounds.iter().map(|b| b.upper).collect();

        let start = Instant::now();
        unsafe {
            ffi::yutki_cpu_box_project(x.len(), x.as_mut_ptr(), lower.as_ptr(), upper.as_ptr());
        }
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(())
    }

    fn dot(&self, x: &[f64], y: &[f64]) -> Result<f64, SolverError> {
        if x.len() != y.len() {
            return Err(SolverError::DimensionMismatch {
                expected: x.len(),
                found: y.len(),
            });
        }

        let start = Instant::now();
        let result = unsafe { ffi::yutki_cpu_dot(x.len(), x.as_ptr(), y.as_ptr()) };
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(result)
    }

    fn norm2(&self, x: &[f64]) -> Result<f64, SolverError> {
        let start = Instant::now();
        let result = unsafe { ffi::yutki_cpu_norm2(x.len(), x.as_ptr()) };
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(result)
    }

    fn norm_inf(&self, x: &[f64]) -> Result<f64, SolverError> {
        let start = Instant::now();
        let result = unsafe { ffi::yutki_cpu_norm_inf(x.len(), x.as_ptr()) };
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(result)
    }
}

/// Sovereign NVIDIA CUDA GPU compute backend.
/// Strictly implements genuine hardware acceleration with zero simulation.
pub struct CudaBackend {
    info: BackendInfo,
    timing: Mutex<BackendTimingProfile>,
}

impl CudaBackend {
    pub fn new() -> Result<Self, SolverError> {
        #[cfg(has_cuda_runtime)]
        {
            let mut count: i32 = 0;
            let res = unsafe { ffi::yutki_cuda_device_count(&mut count) };
            if res != 0 || count <= 0 {
                return Err(SolverError::BackendError(
                    "ERR_GPU_UNAVAILABLE: No compatible NVIDIA CUDA device found.".into(),
                ));
            }

            let mut name_buf = [0i8; 256];
            let _ = unsafe { ffi::yutki_cuda_device_get_name(0, name_buf.as_mut_ptr(), 256) };
            let name_str = unsafe { std::ffi::CStr::from_ptr(name_buf.as_ptr()) }
                .to_str()
                .unwrap_or("NVIDIA GPU")
                .trim()
                .to_string();

            let mut mem: usize = 0;
            let mut sms: i32 = 0;
            let mut major: i32 = 0;
            let mut minor: i32 = 0;
            let _ = unsafe {
                ffi::yutki_cuda_device_get_info(0, &mut mem, &mut sms, &mut major, &mut minor)
            };

            let info = BackendInfo {
                name: "CUDA (NVIDIA GPU)".into(),
                is_gpu: true,
                device_name: name_str,
                details: format!(
                    "Compute Capability {}.{}; SMs: {}; VRAM: {:.2} GB",
                    major,
                    minor,
                    sms,
                    (mem as f64) / (1024.0 * 1024.0 * 1024.0)
                ),
            };

            Ok(Self {
                info,
                timing: Mutex::new(BackendTimingProfile::default()),
            })
        }
        #[cfg(not(has_cuda_runtime))]
        {
            Err(SolverError::BackendError(
                "ERR_GPU_UNAVAILABLE: CUDA runtime kernels were not compiled into this build (nvcc not found). Faking GPU execution is strictly prohibited.".into(),
            ))
        }
    }
}

impl ComputeBackend for CudaBackend {
    fn name(&self) -> &'static str {
        "CUDA (GPU)"
    }

    fn device_info(&self) -> BackendInfo {
        self.info.clone()
    }

    fn is_gpu(&self) -> bool {
        true
    }

    fn timing_profile(&self) -> BackendTimingProfile {
        self.timing.lock().map(|t| t.clone()).unwrap_or_default()
    }

    fn reset_timing(&self) {
        if let Ok(mut t) = self.timing.lock() {
            *t = BackendTimingProfile::default();
        }
    }

    fn spmv(
        &self,
        alpha: f64,
        a: &CsrMatrix,
        x: &[f64],
        beta: f64,
        y: &mut [f64],
    ) -> Result<(), SolverError> {
        #[cfg(has_cuda_runtime)]
        {
            if x.len() != a.num_cols || y.len() != a.num_rows {
                return Err(SolverError::DimensionMismatch {
                    expected: a.num_cols,
                    found: x.len(),
                });
            }

            let m = a.num_rows;
            let n = a.num_cols;
            let nnz = a.nnz();

            let row_ptrs_i32: Vec<i32> = a.row_ptrs.iter().map(|&v| v as i32).collect();
            let col_indices_i32: Vec<i32> = a.col_indices.iter().map(|&v| v as i32).collect();

            let mut d_row_ptrs: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_col_indices: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_values: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_x: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_y: *mut std::ffi::c_void = std::ptr::null_mut();

            // 1. Host-to-Device Transfer
            let t_h2d_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_malloc(&mut d_row_ptrs, (m + 1) * std::mem::size_of::<i32>());
                ffi::yutki_cuda_malloc(&mut d_col_indices, nnz * std::mem::size_of::<i32>());
                ffi::yutki_cuda_malloc(&mut d_values, nnz * std::mem::size_of::<f64>());
                ffi::yutki_cuda_malloc(&mut d_x, n * std::mem::size_of::<f64>());
                ffi::yutki_cuda_malloc(&mut d_y, m * std::mem::size_of::<f64>());

                ffi::yutki_cuda_memcpy_h2d(
                    d_row_ptrs,
                    row_ptrs_i32.as_ptr() as *const _,
                    (m + 1) * std::mem::size_of::<i32>(),
                );
                ffi::yutki_cuda_memcpy_h2d(
                    d_col_indices,
                    col_indices_i32.as_ptr() as *const _,
                    nnz * std::mem::size_of::<i32>(),
                );
                ffi::yutki_cuda_memcpy_h2d(
                    d_values,
                    a.values.as_ptr() as *const _,
                    nnz * std::mem::size_of::<f64>(),
                );
                ffi::yutki_cuda_memcpy_h2d(
                    d_x,
                    x.as_ptr() as *const _,
                    n * std::mem::size_of::<f64>(),
                );
                ffi::yutki_cuda_memcpy_h2d(
                    d_y,
                    y.as_ptr() as *const _,
                    m * std::mem::size_of::<f64>(),
                );
            }
            let h2d_time = t_h2d_start.elapsed().as_secs_f64();

            // 2. Kernel Execution
            let t_kernel_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_spmv(
                    m as i32,
                    n as i32,
                    nnz as i32,
                    d_row_ptrs as *const i32,
                    d_col_indices as *const i32,
                    d_values as *const f64,
                    alpha,
                    d_x as *const f64,
                    beta,
                    d_y as *mut f64,
                );
                ffi::yutki_cuda_synchronize();
            }
            let kernel_time = t_kernel_start.elapsed().as_secs_f64();

            // 3. Device-to-Host Transfer
            let t_d2h_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_memcpy_d2h(
                    y.as_mut_ptr() as *mut _,
                    d_y as *const _,
                    m * std::mem::size_of::<f64>(),
                );
                ffi::yutki_cuda_free(d_row_ptrs);
                ffi::yutki_cuda_free(d_col_indices);
                ffi::yutki_cuda_free(d_values);
                ffi::yutki_cuda_free(d_x);
                ffi::yutki_cuda_free(d_y);
            }
            let d2h_time = t_d2h_start.elapsed().as_secs_f64();

            if let Ok(mut t) = self.timing.lock() {
                t.h2d_transfer_secs += h2d_time;
                t.kernel_execution_secs += kernel_time;
                t.d2h_transfer_secs += d2h_time;
                t.total_runtime_secs += h2d_time + kernel_time + d2h_time;
            }

            Ok(())
        }
        #[cfg(not(has_cuda_runtime))]
        {
            let _ = (alpha, a, x, beta, y);
            Err(SolverError::BackendError("CUDA runtime unlinked".into()))
        }
    }

    fn spmv_transpose(
        &self,
        alpha: f64,
        a: &CsrMatrix,
        y: &[f64],
        beta: f64,
        x: &mut [f64],
    ) -> Result<(), SolverError> {
        #[cfg(has_cuda_runtime)]
        {
            if y.len() != a.num_rows || x.len() != a.num_cols {
                return Err(SolverError::DimensionMismatch {
                    expected: a.num_rows,
                    found: y.len(),
                });
            }

            let m = a.num_rows;
            let n = a.num_cols;
            let nnz = a.nnz();

            let row_ptrs_i32: Vec<i32> = a.row_ptrs.iter().map(|&v| v as i32).collect();
            let col_indices_i32: Vec<i32> = a.col_indices.iter().map(|&v| v as i32).collect();

            let mut d_row_ptrs: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_col_indices: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_values: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_y: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_x: *mut std::ffi::c_void = std::ptr::null_mut();

            let t_h2d_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_malloc(&mut d_row_ptrs, (m + 1) * std::mem::size_of::<i32>());
                ffi::yutki_cuda_malloc(&mut d_col_indices, nnz * std::mem::size_of::<i32>());
                ffi::yutki_cuda_malloc(&mut d_values, nnz * std::mem::size_of::<f64>());
                ffi::yutki_cuda_malloc(&mut d_y, m * std::mem::size_of::<f64>());
                ffi::yutki_cuda_malloc(&mut d_x, n * std::mem::size_of::<f64>());

                ffi::yutki_cuda_memcpy_h2d(
                    d_row_ptrs,
                    row_ptrs_i32.as_ptr() as *const _,
                    (m + 1) * std::mem::size_of::<i32>(),
                );
                ffi::yutki_cuda_memcpy_h2d(
                    d_col_indices,
                    col_indices_i32.as_ptr() as *const _,
                    nnz * std::mem::size_of::<i32>(),
                );
                ffi::yutki_cuda_memcpy_h2d(
                    d_values,
                    a.values.as_ptr() as *const _,
                    nnz * std::mem::size_of::<f64>(),
                );
                ffi::yutki_cuda_memcpy_h2d(
                    d_y,
                    y.as_ptr() as *const _,
                    m * std::mem::size_of::<f64>(),
                );
                ffi::yutki_cuda_memcpy_h2d(
                    d_x,
                    x.as_ptr() as *const _,
                    n * std::mem::size_of::<f64>(),
                );
            }
            let h2d_time = t_h2d_start.elapsed().as_secs_f64();

            let t_kernel_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_spmv_transpose(
                    m as i32,
                    n as i32,
                    nnz as i32,
                    d_row_ptrs as *const i32,
                    d_col_indices as *const i32,
                    d_values as *const f64,
                    alpha,
                    d_y as *const f64,
                    beta,
                    d_x as *mut f64,
                );
                ffi::yutki_cuda_synchronize();
            }
            let kernel_time = t_kernel_start.elapsed().as_secs_f64();

            let t_d2h_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_memcpy_d2h(
                    x.as_mut_ptr() as *mut _,
                    d_x as *const _,
                    n * std::mem::size_of::<f64>(),
                );
                ffi::yutki_cuda_free(d_row_ptrs);
                ffi::yutki_cuda_free(d_col_indices);
                ffi::yutki_cuda_free(d_values);
                ffi::yutki_cuda_free(d_y);
                ffi::yutki_cuda_free(d_x);
            }
            let d2h_time = t_d2h_start.elapsed().as_secs_f64();

            if let Ok(mut t) = self.timing.lock() {
                t.h2d_transfer_secs += h2d_time;
                t.kernel_execution_secs += kernel_time;
                t.d2h_transfer_secs += d2h_time;
                t.total_runtime_secs += h2d_time + kernel_time + d2h_time;
            }

            Ok(())
        }
        #[cfg(not(has_cuda_runtime))]
        {
            let _ = (alpha, a, y, beta, x);
            Err(SolverError::BackendError("CUDA runtime unlinked".into()))
        }
    }

    fn axpby(
        &self,
        alpha: f64,
        x: &[f64],
        beta: f64,
        y: &[f64],
        z: &mut [f64],
    ) -> Result<(), SolverError> {
        #[cfg(has_cuda_runtime)]
        {
            if x.len() != y.len() || x.len() != z.len() {
                return Err(SolverError::DimensionMismatch {
                    expected: x.len(),
                    found: y.len(),
                });
            }

            let n = x.len();
            let mut d_x: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_y: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_z: *mut std::ffi::c_void = std::ptr::null_mut();

            let t_h2d_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_malloc(&mut d_x, n * std::mem::size_of::<f64>());
                ffi::yutki_cuda_malloc(&mut d_y, n * std::mem::size_of::<f64>());
                ffi::yutki_cuda_malloc(&mut d_z, n * std::mem::size_of::<f64>());

                ffi::yutki_cuda_memcpy_h2d(
                    d_x,
                    x.as_ptr() as *const _,
                    n * std::mem::size_of::<f64>(),
                );
                ffi::yutki_cuda_memcpy_h2d(
                    d_y,
                    y.as_ptr() as *const _,
                    n * std::mem::size_of::<f64>(),
                );
            }
            let h2d_time = t_h2d_start.elapsed().as_secs_f64();

            let t_kernel_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_axpby(
                    n as i32,
                    alpha,
                    d_x as *const f64,
                    beta,
                    d_y as *const f64,
                    d_z as *mut f64,
                );
                ffi::yutki_cuda_synchronize();
            }
            let kernel_time = t_kernel_start.elapsed().as_secs_f64();

            let t_d2h_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_memcpy_d2h(
                    z.as_mut_ptr() as *mut _,
                    d_z as *const _,
                    n * std::mem::size_of::<f64>(),
                );
                ffi::yutki_cuda_free(d_x);
                ffi::yutki_cuda_free(d_y);
                ffi::yutki_cuda_free(d_z);
            }
            let d2h_time = t_d2h_start.elapsed().as_secs_f64();

            if let Ok(mut t) = self.timing.lock() {
                t.h2d_transfer_secs += h2d_time;
                t.kernel_execution_secs += kernel_time;
                t.d2h_transfer_secs += d2h_time;
                t.total_runtime_secs += h2d_time + kernel_time + d2h_time;
            }

            Ok(())
        }
        #[cfg(not(has_cuda_runtime))]
        {
            let _ = (alpha, x, beta, y, z);
            Err(SolverError::BackendError("CUDA runtime unlinked".into()))
        }
    }

    fn box_project(&self, x: &mut [f64], bounds: &[Bound]) -> Result<(), SolverError> {
        #[cfg(has_cuda_runtime)]
        {
            if x.len() != bounds.len() {
                return Err(SolverError::DimensionMismatch {
                    expected: bounds.len(),
                    found: x.len(),
                });
            }

            let n = x.len();
            let lower: Vec<f64> = bounds.iter().map(|b| b.lower).collect();
            let upper: Vec<f64> = bounds.iter().map(|b| b.upper).collect();

            let mut d_x: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_lower: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_upper: *mut std::ffi::c_void = std::ptr::null_mut();

            let t_h2d_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_malloc(&mut d_x, n * std::mem::size_of::<f64>());
                ffi::yutki_cuda_malloc(&mut d_lower, n * std::mem::size_of::<f64>());
                ffi::yutki_cuda_malloc(&mut d_upper, n * std::mem::size_of::<f64>());

                ffi::yutki_cuda_memcpy_h2d(
                    d_x,
                    x.as_ptr() as *const _,
                    n * std::mem::size_of::<f64>(),
                );
                ffi::yutki_cuda_memcpy_h2d(
                    d_lower,
                    lower.as_ptr() as *const _,
                    n * std::mem::size_of::<f64>(),
                );
                ffi::yutki_cuda_memcpy_h2d(
                    d_upper,
                    upper.as_ptr() as *const _,
                    n * std::mem::size_of::<f64>(),
                );
            }
            let h2d_time = t_h2d_start.elapsed().as_secs_f64();

            let t_kernel_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_box_project(
                    n as i32,
                    d_x as *mut f64,
                    d_lower as *const f64,
                    d_upper as *const f64,
                );
                ffi::yutki_cuda_synchronize();
            }
            let kernel_time = t_kernel_start.elapsed().as_secs_f64();

            let t_d2h_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_memcpy_d2h(
                    x.as_mut_ptr() as *mut _,
                    d_x as *const _,
                    n * std::mem::size_of::<f64>(),
                );
                ffi::yutki_cuda_free(d_x);
                ffi::yutki_cuda_free(d_lower);
                ffi::yutki_cuda_free(d_upper);
            }
            let d2h_time = t_d2h_start.elapsed().as_secs_f64();

            if let Ok(mut t) = self.timing.lock() {
                t.h2d_transfer_secs += h2d_time;
                t.kernel_execution_secs += kernel_time;
                t.d2h_transfer_secs += d2h_time;
                t.total_runtime_secs += h2d_time + kernel_time + d2h_time;
            }

            Ok(())
        }
        #[cfg(not(has_cuda_runtime))]
        {
            let _ = (x, bounds);
            Err(SolverError::BackendError("CUDA runtime unlinked".into()))
        }
    }

    fn dot(&self, x: &[f64], y: &[f64]) -> Result<f64, SolverError> {
        #[cfg(has_cuda_runtime)]
        {
            if x.len() != y.len() {
                return Err(SolverError::DimensionMismatch {
                    expected: x.len(),
                    found: y.len(),
                });
            }

            let n = x.len();
            let mut d_x: *mut std::ffi::c_void = std::ptr::null_mut();
            let mut d_y: *mut std::ffi::c_void = std::ptr::null_mut();

            let t_h2d_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_malloc(&mut d_x, n * std::mem::size_of::<f64>());
                ffi::yutki_cuda_malloc(&mut d_y, n * std::mem::size_of::<f64>());

                ffi::yutki_cuda_memcpy_h2d(
                    d_x,
                    x.as_ptr() as *const _,
                    n * std::mem::size_of::<f64>(),
                );
                ffi::yutki_cuda_memcpy_h2d(
                    d_y,
                    y.as_ptr() as *const _,
                    n * std::mem::size_of::<f64>(),
                );
            }
            let h2d_time = t_h2d_start.elapsed().as_secs_f64();

            let mut result: f64 = 0.0;
            let t_kernel_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_dot(n as i32, d_x as *const f64, d_y as *const f64, &mut result);
            }
            let kernel_time = t_kernel_start.elapsed().as_secs_f64();

            unsafe {
                ffi::yutki_cuda_free(d_x);
                ffi::yutki_cuda_free(d_y);
            }

            if let Ok(mut t) = self.timing.lock() {
                t.h2d_transfer_secs += h2d_time;
                t.kernel_execution_secs += kernel_time;
                t.total_runtime_secs += h2d_time + kernel_time;
            }

            Ok(result)
        }
        #[cfg(not(has_cuda_runtime))]
        {
            let _ = (x, y);
            Err(SolverError::BackendError("CUDA runtime unlinked".into()))
        }
    }

    fn norm2(&self, x: &[f64]) -> Result<f64, SolverError> {
        #[cfg(has_cuda_runtime)]
        {
            let n = x.len();
            let mut d_x: *mut std::ffi::c_void = std::ptr::null_mut();

            let t_h2d_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_malloc(&mut d_x, n * std::mem::size_of::<f64>());
                ffi::yutki_cuda_memcpy_h2d(
                    d_x,
                    x.as_ptr() as *const _,
                    n * std::mem::size_of::<f64>(),
                );
            }
            let h2d_time = t_h2d_start.elapsed().as_secs_f64();

            let mut result: f64 = 0.0;
            let t_kernel_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_norm2(n as i32, d_x as *const f64, &mut result);
            }
            let kernel_time = t_kernel_start.elapsed().as_secs_f64();

            unsafe {
                ffi::yutki_cuda_free(d_x);
            }

            if let Ok(mut t) = self.timing.lock() {
                t.h2d_transfer_secs += h2d_time;
                t.kernel_execution_secs += kernel_time;
                t.total_runtime_secs += h2d_time + kernel_time;
            }

            Ok(result)
        }
        #[cfg(not(has_cuda_runtime))]
        {
            let _ = x;
            Err(SolverError::BackendError("CUDA runtime unlinked".into()))
        }
    }

    fn norm_inf(&self, x: &[f64]) -> Result<f64, SolverError> {
        #[cfg(has_cuda_runtime)]
        {
            let n = x.len();
            let mut d_x: *mut std::ffi::c_void = std::ptr::null_mut();

            let t_h2d_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_malloc(&mut d_x, n * std::mem::size_of::<f64>());
                ffi::yutki_cuda_memcpy_h2d(
                    d_x,
                    x.as_ptr() as *const _,
                    n * std::mem::size_of::<f64>(),
                );
            }
            let h2d_time = t_h2d_start.elapsed().as_secs_f64();

            let mut result: f64 = 0.0;
            let t_kernel_start = Instant::now();
            unsafe {
                ffi::yutki_cuda_norm_inf(n as i32, d_x as *const f64, &mut result);
            }
            let kernel_time = t_kernel_start.elapsed().as_secs_f64();

            unsafe {
                ffi::yutki_cuda_free(d_x);
            }

            if let Ok(mut t) = self.timing.lock() {
                t.h2d_transfer_secs += h2d_time;
                t.kernel_execution_secs += kernel_time;
                t.total_runtime_secs += h2d_time + kernel_time;
            }

            Ok(result)
        }
        #[cfg(not(has_cuda_runtime))]
        {
            let _ = x;
            Err(SolverError::BackendError("CUDA runtime unlinked".into()))
        }
    }
}

/// Dynamic discovery probing the OS CUDA Driver (e.g. nvcuda.dll on Windows).
#[cfg(windows)]
fn probe_cuda_driver() -> Option<BackendInfo> {
    unsafe extern "system" {
        fn LoadLibraryA(lpLibFileName: *const i8) -> *mut std::ffi::c_void;
        fn GetProcAddress(
            hModule: *mut std::ffi::c_void,
            lpProcName: *const i8,
        ) -> *mut std::ffi::c_void;
        fn FreeLibrary(hModule: *mut std::ffi::c_void) -> i32;
    }

    unsafe {
        let lib_name = b"nvcuda.dll\0";
        let handle = LoadLibraryA(lib_name.as_ptr() as *const i8);
        if handle.is_null() {
            return None;
        }

        let cu_init_sym = b"cuInit\0";
        let cu_dev_count_sym = b"cuDeviceGetCount\0";
        let cu_dev_name_sym = b"cuDeviceGetName\0";
        let cu_dev_mem_sym = b"cuDeviceTotalMem_v2\0";

        let cu_init_ptr = GetProcAddress(handle, cu_init_sym.as_ptr() as *const i8);
        let cu_count_ptr = GetProcAddress(handle, cu_dev_count_sym.as_ptr() as *const i8);

        if cu_init_ptr.is_null() || cu_count_ptr.is_null() {
            FreeLibrary(handle);
            return None;
        }

        type CuInitFn = unsafe extern "C" fn(u32) -> i32;
        type CuCountFn = unsafe extern "C" fn(*mut i32) -> i32;
        type CuNameFn = unsafe extern "C" fn(*mut i8, i32, i32) -> i32;
        type CuMemFn = unsafe extern "C" fn(*mut usize, i32) -> i32;

        let cu_init: CuInitFn = std::mem::transmute(cu_init_ptr);
        let cu_count: CuCountFn = std::mem::transmute(cu_count_ptr);

        if cu_init(0) != 0 {
            FreeLibrary(handle);
            return None;
        }

        let mut count: i32 = 0;
        if cu_count(&mut count) != 0 || count <= 0 {
            FreeLibrary(handle);
            return None;
        }

        let mut dev_name = "NVIDIA CUDA Device".to_string();
        let cu_name_ptr = GetProcAddress(handle, cu_dev_name_sym.as_ptr() as *const i8);
        if !cu_name_ptr.is_null() {
            let cu_name: CuNameFn = std::mem::transmute(cu_name_ptr);
            let mut name_buf = [0i8; 256];
            if cu_name(name_buf.as_mut_ptr(), 256, 0) == 0 {
                let c_str = std::ffi::CStr::from_ptr(name_buf.as_ptr());
                if let Ok(s) = c_str.to_str() {
                    dev_name = s.trim().to_string();
                }
            }
        }

        let mut mem_bytes: usize = 0;
        let cu_mem_ptr = GetProcAddress(handle, cu_dev_mem_sym.as_ptr() as *const i8);
        if !cu_mem_ptr.is_null() {
            let cu_mem: CuMemFn = std::mem::transmute(cu_mem_ptr);
            let _ = cu_mem(&mut mem_bytes, 0);
        }

        FreeLibrary(handle);

        Some(BackendInfo {
            name: "CUDA (NVIDIA GPU)".into(),
            is_gpu: true,
            device_name: dev_name,
            details: format!(
                "Genuine CUDA devices: {}; VRAM: {:.2} GB",
                count,
                (mem_bytes as f64) / (1024.0 * 1024.0 * 1024.0)
            ),
        })
    }
}

#[cfg(not(windows))]
fn probe_cuda_driver() -> Option<BackendInfo> {
    None
}

/// Detect whether a genuine CUDA driver and device are operational.
pub fn detect_cuda_hardware() -> Option<BackendInfo> {
    #[cfg(has_cuda_runtime)]
    {
        let mut count: i32 = 0;
        let res = unsafe { ffi::yutki_cuda_device_count(&mut count) };
        if res == 0 && count > 0 {
            let mut name_buf = [0i8; 256];
            let _ = unsafe { ffi::yutki_cuda_device_get_name(0, name_buf.as_mut_ptr(), 256) };
            let name_str = unsafe { std::ffi::CStr::from_ptr(name_buf.as_ptr()) }
                .to_str()
                .unwrap_or("NVIDIA GPU")
                .trim()
                .to_string();

            let mut mem: usize = 0;
            let mut sms: i32 = 0;
            let mut major: i32 = 0;
            let mut minor: i32 = 0;
            let _ = unsafe {
                ffi::yutki_cuda_device_get_info(0, &mut mem, &mut sms, &mut major, &mut minor)
            };

            return Some(BackendInfo {
                name: "CUDA (NVIDIA GPU)".into(),
                is_gpu: true,
                device_name: name_str,
                details: format!(
                    "Device 0: Compute Capability {}.{}; SMs: {}; VRAM: {:.2} GB",
                    major,
                    minor,
                    sms,
                    (mem as f64) / (1024.0 * 1024.0 * 1024.0)
                ),
            });
        }
    }

    probe_cuda_driver()
}

/// Select and instantiate the requested backend with strict verification.
pub fn select_backend(
    requested: BackendType,
) -> Result<(Box<dyn ComputeBackend>, BackendInfo), SolverError> {
    match requested {
        BackendType::Cpu => {
            let backend = Box::new(CpuRustBackend::new());
            let info = backend.device_info();
            Ok((backend, info))
        }
        BackendType::Gpu => match detect_cuda_hardware() {
            Some(gpu_info) => {
                #[cfg(has_cuda_runtime)]
                {
                    match CudaBackend::new() {
                        Ok(backend) => {
                            let info = backend.device_info();
                            Ok((Box::new(backend), info))
                        }
                        Err(e) => Err(e),
                    }
                }
                #[cfg(not(has_cuda_runtime))]
                {
                    Err(SolverError::BackendError(format!(
                        "ERR_GPU_UNAVAILABLE: Genuine NVIDIA hardware detected ({}) but CUDA runtime kernels were unlinked in this build (nvcc was absent). Faking GPU execution is strictly prohibited.",
                        gpu_info.device_name
                    )))
                }
            }
            None => Err(SolverError::BackendError(
                "ERR_GPU_UNAVAILABLE: No compatible NVIDIA CUDA device found or CUDA driver not initialized. Faking GPU execution is strictly prohibited.".into(),
            )),
        },
        BackendType::Auto => {
            #[cfg(has_cuda_runtime)]
            {
                if let Some(_gpu_info) = detect_cuda_hardware() {
                    if let Ok(backend) = CudaBackend::new() {
                        let info = backend.device_info();
                        return Ok((Box::new(backend), info));
                    }
                }
            }

            // Fallback to CPU backend gracefully with zero falsification
            let backend = Box::new(CpuRustBackend::new());
            let info = backend.device_info();
            Ok((backend, info))
        }
    }
}

/// Compares numerical equivalence between two compute backends on a sparse matrix.
pub fn compare_backends_spmv(
    backend_a: &dyn ComputeBackend,
    backend_b: &dyn ComputeBackend,
    a: &CsrMatrix,
    x: &[f64],
) -> Result<f64, SolverError> {
    let mut y_a = vec![0.0; a.num_rows];
    let mut y_b = vec![0.0; a.num_rows];

    backend_a.spmv(1.0, a, x, 0.0, &mut y_a)?;
    backend_b.spmv(1.0, a, x, 0.0, &mut y_b)?;

    let mut max_diff = 0.0f64;
    for (va, vb) in y_a.iter().zip(y_b.iter()) {
        max_diff = max_diff.max((va - vb).abs());
    }
    Ok(max_diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use yutki_sparse::CooMatrix;

    fn sample_sparse_matrix() -> CsrMatrix {
        let mut coo = CooMatrix::new(3, 4);
        coo.add_entry(0, 0, 1.5).unwrap();
        coo.add_entry(0, 2, -2.0).unwrap();
        coo.add_entry(1, 1, 3.5).unwrap();
        coo.add_entry(2, 0, -1.0).unwrap();
        coo.add_entry(2, 3, 4.2).unwrap();
        CsrMatrix::from_coo(&coo).unwrap()
    }

    #[test]
    fn test_cpu_backend_axpby_and_project() {
        let backend = CpuRustBackend::new();
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let mut z = vec![0.0; 3];

        backend.axpby(2.0, &x, 1.0, &y, &mut z).unwrap();
        // 2*[1,2,3] + [4,5,6] = [6, 9, 12]
        assert_eq!(z, vec![6.0, 9.0, 12.0]);

        let bounds = vec![
            Bound::new(0.0, 5.0),
            Bound::new(0.0, 8.0),
            Bound::new(0.0, 10.0),
        ];
        backend.box_project(&mut z, &bounds).unwrap();
        assert_eq!(z, vec![5.0, 8.0, 10.0]);

        let profile = backend.timing_profile();
        assert!(profile.total_runtime_secs >= 0.0);
    }

    #[test]
    fn test_gpu_unavailable_strict_error() {
        let res = select_backend(BackendType::Gpu);
        assert!(res.is_err());
        if let Err(SolverError::BackendError(msg)) = res {
            assert!(
                msg.contains("ERR_GPU_UNAVAILABLE"),
                "Expected ERR_GPU_UNAVAILABLE in: {msg}"
            );
        } else {
            panic!("Expected BackendError");
        }
    }

    #[test]
    fn test_backend_auto_fallback() {
        let res = select_backend(BackendType::Auto);
        assert!(res.is_ok());
        let (backend, info) = res.unwrap();
        assert_eq!(backend.name(), "CPU (Rust)");
        assert!(!info.is_gpu);
    }

    #[test]
    fn test_backend_cpu_selection() {
        let res = select_backend(BackendType::Cpu);
        assert!(res.is_ok());
        let (backend, info) = res.unwrap();
        assert_eq!(backend.name(), "CPU (Rust)");
        assert!(!info.is_gpu);
    }

    #[test]
    fn test_numerical_equivalence_spmv() {
        let a = sample_sparse_matrix();
        let x = vec![1.2, -3.4, 5.6, -7.8];

        let rust_backend = CpuRustBackend::new();
        let cpp_backend = CpuCppBackend::new();

        let mut y_rust = vec![0.0; a.num_rows];
        let mut y_cpp = vec![0.0; a.num_rows];

        rust_backend.spmv(1.0, &a, &x, 0.0, &mut y_rust).unwrap();
        cpp_backend.spmv(1.0, &a, &x, 0.0, &mut y_cpp).unwrap();

        for i in 0..a.num_rows {
            assert!(
                (y_rust[i] - y_cpp[i]).abs() < 1e-14,
                "SpMV discrepancy at row {i}: rust={}, cpp={}",
                y_rust[i],
                y_cpp[i]
            );
        }

        // Test with alpha = 2.5, beta = -1.5
        rust_backend.spmv(2.5, &a, &x, -1.5, &mut y_rust).unwrap();
        cpp_backend.spmv(2.5, &a, &x, -1.5, &mut y_cpp).unwrap();

        for i in 0..a.num_rows {
            assert!(
                (y_rust[i] - y_cpp[i]).abs() < 1e-14,
                "Accumulated SpMV discrepancy at row {i}: rust={}, cpp={}",
                y_rust[i],
                y_cpp[i]
            );
        }

        // Test compare helper
        let diff = compare_backends_spmv(&rust_backend, &cpp_backend, &a, &x).unwrap();
        assert!(diff < 1e-14);
    }

    #[test]
    fn test_numerical_equivalence_spmv_transpose() {
        let a = sample_sparse_matrix();
        let y = vec![2.5, -1.5, 3.0];

        let rust_backend = CpuRustBackend::new();
        let cpp_backend = CpuCppBackend::new();

        let mut x_rust = vec![0.0; a.num_cols];
        let mut x_cpp = vec![0.0; a.num_cols];

        rust_backend
            .spmv_transpose(1.0, &a, &y, 0.0, &mut x_rust)
            .unwrap();
        cpp_backend
            .spmv_transpose(1.0, &a, &y, 0.0, &mut x_cpp)
            .unwrap();

        for j in 0..a.num_cols {
            assert!(
                (x_rust[j] - x_cpp[j]).abs() < 1e-14,
                "Transpose SpMV discrepancy at col {j}: rust={}, cpp={}",
                x_rust[j],
                x_cpp[j]
            );
        }
    }

    #[test]
    fn test_numerical_equivalence_axpby() {
        let x = vec![1.5, -2.5, 3.5, -4.5];
        let y = vec![0.5, 1.0, -1.5, 2.0];

        let rust_backend = CpuRustBackend::new();
        let cpp_backend = CpuCppBackend::new();

        let mut z_rust = vec![0.0; 4];
        let mut z_cpp = vec![0.0; 4];

        rust_backend.axpby(2.0, &x, -3.0, &y, &mut z_rust).unwrap();
        cpp_backend.axpby(2.0, &x, -3.0, &y, &mut z_cpp).unwrap();

        for i in 0..4 {
            assert!(
                (z_rust[i] - z_cpp[i]).abs() < 1e-14,
                "axpby discrepancy at index {i}: rust={}, cpp={}",
                z_rust[i],
                z_cpp[i]
            );
        }
    }

    #[test]
    fn test_numerical_equivalence_box_project() {
        let mut x_rust = vec![-2.0, 3.0, 7.0, -0.5];
        let mut x_cpp = x_rust.clone();
        let bounds = vec![
            Bound::new(0.0, 5.0),
            Bound::new(0.0, 2.0),
            Bound::new(1.0, 10.0),
            Bound::new(-1.0, 1.0),
        ];

        let rust_backend = CpuRustBackend::new();
        let cpp_backend = CpuCppBackend::new();

        rust_backend.box_project(&mut x_rust, &bounds).unwrap();
        cpp_backend.box_project(&mut x_cpp, &bounds).unwrap();

        assert_eq!(x_rust, x_cpp);
    }

    #[test]
    fn test_numerical_equivalence_dot_and_norms() {
        let x = vec![1.2, -3.4, 5.6, -7.8, 9.0];
        let y = vec![-0.5, 2.3, -4.1, 1.9, -3.2];

        let rust_backend = CpuRustBackend::new();
        let cpp_backend = CpuCppBackend::new();

        let dot_rust = rust_backend.dot(&x, &y).unwrap();
        let dot_cpp = cpp_backend.dot(&x, &y).unwrap();
        assert!((dot_rust - dot_cpp).abs() < 1e-14);

        let n2_rust = rust_backend.norm2(&x).unwrap();
        let n2_cpp = cpp_backend.norm2(&x).unwrap();
        assert!((n2_rust - n2_cpp).abs() < 1e-14);

        let ninf_rust = rust_backend.norm_inf(&x).unwrap();
        let ninf_cpp = cpp_backend.norm_inf(&x).unwrap();
        assert!((ninf_rust - ninf_cpp).abs() < 1e-14);
    }
}
