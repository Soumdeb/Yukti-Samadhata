//! Hardware compute backend abstraction for CPU and genuine NVIDIA CUDA GPU execution.
//! Sovereign zero-falsification policy: genuine hardware detection and execution only.
//! CPU backend provides multi-threaded, portable pure Rust linear algebra kernels.
//! CUDA backend provides genuine GPU accelerated kernels via the pure Rust CUDA Driver API and PTX.

pub mod cuda_driver;
pub mod ffi;

use cuda_driver::{probe_nvidia_devices, CudaDeviceBuffer, CudaPipeline, CUDA_SUCCESS};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
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

/// Unified compute backend trait for linear algebra operations used in PDHG and Simplex.
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
            device_name: "Host Multi-Core Processor".into(),
            details: format!(
                "Logical cores: {} (Rayon Work-Stealing Pool)",
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
        if a.num_rows >= 256 {
            y.par_iter_mut().enumerate().for_each(|(i, y_elem)| {
                let start_idx = a.row_ptrs[i];
                let end_idx = a.row_ptrs[i + 1];
                let mut sum = 0.0;
                for k in start_idx..end_idx {
                    let col = a.col_indices[k];
                    sum += a.values[k] * x[col];
                }
                if beta == 0.0 {
                    *y_elem = alpha * sum;
                } else {
                    *y_elem = alpha * sum + beta * *y_elem;
                }
            });
        } else {
            let _ = a.spmv(alpha, x, beta, y);
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
        if x.len() >= 1024 {
            z.par_iter_mut().enumerate().for_each(|(i, z_elem)| {
                *z_elem = alpha * x[i] + beta * y[i];
            });
        } else {
            for i in 0..x.len() {
                z[i] = alpha * x[i] + beta * y[i];
            }
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
        if x.len() >= 1024 {
            x.par_iter_mut().zip(bounds.par_iter()).for_each(|(xi, b)| {
                *xi = b.project(*xi);
            });
        } else {
            for (xi, b) in x.iter_mut().zip(bounds.iter()) {
                *xi = b.project(*xi);
            }
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
        let sum: f64 = if x.len() >= 1024 {
            x.par_iter().zip(y.par_iter()).map(|(&a, &b)| a * b).sum()
        } else {
            x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum()
        };
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(sum)
    }

    fn norm2(&self, x: &[f64]) -> Result<f64, SolverError> {
        let start = Instant::now();
        let sum_sq: f64 = if x.len() >= 1024 {
            x.par_iter().map(|&v| v * v).sum()
        } else {
            x.iter().map(|&v| v * v).sum()
        };
        let dur = start.elapsed().as_secs_f64();
        if let Ok(mut t) = self.timing.lock() {
            t.kernel_execution_secs += dur;
            t.total_runtime_secs += dur;
        }
        Ok(sum_sq.sqrt())
    }

    fn norm_inf(&self, x: &[f64]) -> Result<f64, SolverError> {
        let start = Instant::now();
        let max_val = if x.len() >= 1024 {
            x.par_iter()
                .map(|&v| v.abs())
                .reduce(|| 0.0f64, |a, b| if a > b { a } else { b })
        } else {
            x.iter().map(|&v| v.abs()).fold(0.0f64, f64::max)
        };
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

/// Backwards compatibility alias: CpuCppBackend now maps to pure CpuRustBackend.
pub type CpuCppBackend = CpuRustBackend;

/// Sovereign NVIDIA CUDA GPU compute backend using pure Rust Driver API and PTX.
/// Strictly implements genuine hardware acceleration with zero simulation.
pub struct CudaBackend {
    info: BackendInfo,
    pipeline: Arc<CudaPipeline>,
    timing: Mutex<BackendTimingProfile>,
}

impl CudaBackend {
    pub fn new() -> Result<Self, SolverError> {
        let devices = probe_nvidia_devices().ok_or_else(|| {
            SolverError::BackendError(
                "ERR_GPU_UNAVAILABLE: No compatible NVIDIA CUDA device found or CUDA driver not initialized. Faking GPU execution is strictly prohibited.".into(),
            )
        })?;

        let dev = &devices[0];
        let pipeline = Arc::new(CudaPipeline::new(dev.device_id)?);

        let info = BackendInfo {
            name: "CUDA (NVIDIA GPU)".into(),
            is_gpu: true,
            device_name: dev.name.clone(),
            details: format!(
                "Compute Capability {}.{}; SMs: {}; VRAM: {:.2} GB",
                dev.major,
                dev.minor,
                dev.sm_count,
                (dev.total_memory_bytes as f64) / (1024.0 * 1024.0 * 1024.0)
            ),
        };

        Ok(Self {
            info,
            pipeline,
            timing: Mutex::new(BackendTimingProfile::default()),
        })
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
        if x.len() != a.num_cols || y.len() != a.num_rows {
            return Err(SolverError::DimensionMismatch {
                expected: a.num_cols,
                found: x.len(),
            });
        }

        let m = a.num_rows as u32;
        let nnz = a.nnz();

        let row_ptrs_i32: Vec<i32> = a.row_ptrs.iter().map(|&v| v as i32).collect();
        let col_indices_i32: Vec<i32> = a.col_indices.iter().map(|&v| v as i32).collect();

        let t_h2d_start = Instant::now();
        let d_row_ptrs =
            CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), (a.num_rows + 1) * 4)?;
        let d_col_indices = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), nnz * 4)?;
        let d_values = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), nnz * 8)?;
        let d_x = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), a.num_cols * 8)?;
        let d_y = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), a.num_rows * 8)?;

        d_row_ptrs.copy_from_host(unsafe {
            std::slice::from_raw_parts(row_ptrs_i32.as_ptr() as *const u8, row_ptrs_i32.len() * 4)
        })?;
        d_col_indices.copy_from_host(unsafe {
            std::slice::from_raw_parts(
                col_indices_i32.as_ptr() as *const u8,
                col_indices_i32.len() * 4,
            )
        })?;
        d_values.copy_from_host(unsafe {
            std::slice::from_raw_parts(a.values.as_ptr() as *const u8, a.values.len() * 8)
        })?;
        d_x.copy_from_host(unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 8)
        })?;
        if beta != 0.0 {
            d_y.copy_from_host(unsafe {
                std::slice::from_raw_parts(y.as_ptr() as *const u8, y.len() * 8)
            })?;
        } else {
            d_y.zero()?;
        }
        let h2d_time = t_h2d_start.elapsed().as_secs_f64();

        let t_kernel_start = Instant::now();
        let block_size = 256u32;
        let num_blocks = m.div_ceil(block_size);

        let mut p_m = m;
        let mut p_row_ptrs = d_row_ptrs.ptr;
        let mut p_col_indices = d_col_indices.ptr;
        let mut p_values = d_values.ptr;
        let mut p_alpha = alpha;
        let mut p_x = d_x.ptr;
        let mut p_beta = beta;
        let mut p_y = d_y.ptr;

        let mut args: [*mut std::ffi::c_void; 8] = [
            &mut p_m as *mut _ as *mut std::ffi::c_void,
            &mut p_row_ptrs as *mut _ as *mut std::ffi::c_void,
            &mut p_col_indices as *mut _ as *mut std::ffi::c_void,
            &mut p_values as *mut _ as *mut std::ffi::c_void,
            &mut p_alpha as *mut _ as *mut std::ffi::c_void,
            &mut p_x as *mut _ as *mut std::ffi::c_void,
            &mut p_beta as *mut _ as *mut std::ffi::c_void,
            &mut p_y as *mut _ as *mut std::ffi::c_void,
        ];

        let res = unsafe {
            (self.pipeline.driver.cu_launch_kernel)(
                self.pipeline.k_spmv,
                num_blocks,
                1,
                1,
                block_size,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != CUDA_SUCCESS {
            return Err(SolverError::BackendError(format!(
                "CUDA kernel launch failed for spmv (code {})",
                res
            )));
        }
        self.pipeline.synchronize()?;
        let kernel_time = t_kernel_start.elapsed().as_secs_f64();

        let t_d2h_start = Instant::now();
        d_y.copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(y.as_mut_ptr() as *mut u8, y.len() * 8)
        })?;
        let d2h_time = t_d2h_start.elapsed().as_secs_f64();

        if let Ok(mut t) = self.timing.lock() {
            t.h2d_transfer_secs += h2d_time;
            t.kernel_execution_secs += kernel_time;
            t.d2h_transfer_secs += d2h_time;
            t.total_runtime_secs += h2d_time + kernel_time + d2h_time;
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
        if y.len() != a.num_rows || x.len() != a.num_cols {
            return Err(SolverError::DimensionMismatch {
                expected: a.num_rows,
                found: y.len(),
            });
        }

        let m = a.num_rows as u32;
        let n = a.num_cols as u32;
        let nnz = a.nnz();

        let row_ptrs_i32: Vec<i32> = a.row_ptrs.iter().map(|&v| v as i32).collect();
        let col_indices_i32: Vec<i32> = a.col_indices.iter().map(|&v| v as i32).collect();

        let t_h2d_start = Instant::now();
        let d_row_ptrs =
            CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), (a.num_rows + 1) * 4)?;
        let d_col_indices = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), nnz * 4)?;
        let d_values = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), nnz * 8)?;
        let d_y = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), a.num_rows * 8)?;
        let d_x = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), a.num_cols * 8)?;

        d_row_ptrs.copy_from_host(unsafe {
            std::slice::from_raw_parts(row_ptrs_i32.as_ptr() as *const u8, row_ptrs_i32.len() * 4)
        })?;
        d_col_indices.copy_from_host(unsafe {
            std::slice::from_raw_parts(
                col_indices_i32.as_ptr() as *const u8,
                col_indices_i32.len() * 4,
            )
        })?;
        d_values.copy_from_host(unsafe {
            std::slice::from_raw_parts(a.values.as_ptr() as *const u8, a.values.len() * 8)
        })?;
        d_y.copy_from_host(unsafe {
            std::slice::from_raw_parts(y.as_ptr() as *const u8, y.len() * 8)
        })?;

        if beta == 0.0 {
            d_x.zero()?;
        } else {
            d_x.copy_from_host(unsafe {
                std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 8)
            })?;
        }
        let h2d_time = t_h2d_start.elapsed().as_secs_f64();

        let t_kernel_start = Instant::now();
        let block_size = 256u32;

        if beta != 0.0 && beta != 1.0 {
            let num_blocks_scale = n.div_ceil(block_size);
            let mut p_n = n;
            let mut p_beta = beta;
            let mut p_x = d_x.ptr;
            let mut scale_args: [*mut std::ffi::c_void; 3] = [
                &mut p_n as *mut _ as *mut std::ffi::c_void,
                &mut p_beta as *mut _ as *mut std::ffi::c_void,
                &mut p_x as *mut _ as *mut std::ffi::c_void,
            ];
            unsafe {
                let _ = (self.pipeline.driver.cu_launch_kernel)(
                    self.pipeline.k_scale,
                    num_blocks_scale,
                    1,
                    1,
                    block_size,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    scale_args.as_mut_ptr(),
                    std::ptr::null_mut(),
                );
            }
        }

        let num_blocks = m.div_ceil(block_size);
        let mut p_m = m;
        let mut p_row_ptrs = d_row_ptrs.ptr;
        let mut p_col_indices = d_col_indices.ptr;
        let mut p_values = d_values.ptr;
        let mut p_alpha = alpha;
        let mut p_y = d_y.ptr;
        let mut p_x = d_x.ptr;

        let mut args: [*mut std::ffi::c_void; 7] = [
            &mut p_m as *mut _ as *mut std::ffi::c_void,
            &mut p_row_ptrs as *mut _ as *mut std::ffi::c_void,
            &mut p_col_indices as *mut _ as *mut std::ffi::c_void,
            &mut p_values as *mut _ as *mut std::ffi::c_void,
            &mut p_alpha as *mut _ as *mut std::ffi::c_void,
            &mut p_y as *mut _ as *mut std::ffi::c_void,
            &mut p_x as *mut _ as *mut std::ffi::c_void,
        ];

        let res = unsafe {
            (self.pipeline.driver.cu_launch_kernel)(
                self.pipeline.k_spmv_transpose,
                num_blocks,
                1,
                1,
                block_size,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != CUDA_SUCCESS {
            return Err(SolverError::BackendError(format!(
                "CUDA kernel launch failed for spmv_transpose (code {})",
                res
            )));
        }
        self.pipeline.synchronize()?;
        let kernel_time = t_kernel_start.elapsed().as_secs_f64();

        let t_d2h_start = Instant::now();
        d_x.copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u8, x.len() * 8)
        })?;
        let d2h_time = t_d2h_start.elapsed().as_secs_f64();

        if let Ok(mut t) = self.timing.lock() {
            t.h2d_transfer_secs += h2d_time;
            t.kernel_execution_secs += kernel_time;
            t.d2h_transfer_secs += d2h_time;
            t.total_runtime_secs += h2d_time + kernel_time + d2h_time;
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

        let n = x.len() as u32;
        let t_h2d_start = Instant::now();
        let d_x = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), x.len() * 8)?;
        let d_y = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), y.len() * 8)?;
        let d_z = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), z.len() * 8)?;

        d_x.copy_from_host(unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 8)
        })?;
        d_y.copy_from_host(unsafe {
            std::slice::from_raw_parts(y.as_ptr() as *const u8, y.len() * 8)
        })?;
        let h2d_time = t_h2d_start.elapsed().as_secs_f64();

        let t_kernel_start = Instant::now();
        let block_size = 256u32;
        let num_blocks = n.div_ceil(block_size);

        let mut p_n = n;
        let mut p_alpha = alpha;
        let mut p_x = d_x.ptr;
        let mut p_beta = beta;
        let mut p_y = d_y.ptr;
        let mut p_z = d_z.ptr;

        let mut args: [*mut std::ffi::c_void; 6] = [
            &mut p_n as *mut _ as *mut std::ffi::c_void,
            &mut p_alpha as *mut _ as *mut std::ffi::c_void,
            &mut p_x as *mut _ as *mut std::ffi::c_void,
            &mut p_beta as *mut _ as *mut std::ffi::c_void,
            &mut p_y as *mut _ as *mut std::ffi::c_void,
            &mut p_z as *mut _ as *mut std::ffi::c_void,
        ];

        let res = unsafe {
            (self.pipeline.driver.cu_launch_kernel)(
                self.pipeline.k_axpby,
                num_blocks,
                1,
                1,
                block_size,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != CUDA_SUCCESS {
            return Err(SolverError::BackendError(format!(
                "CUDA kernel launch failed for axpby (code {})",
                res
            )));
        }
        self.pipeline.synchronize()?;
        let kernel_time = t_kernel_start.elapsed().as_secs_f64();

        let t_d2h_start = Instant::now();
        d_z.copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(z.as_mut_ptr() as *mut u8, z.len() * 8)
        })?;
        let d2h_time = t_d2h_start.elapsed().as_secs_f64();

        if let Ok(mut t) = self.timing.lock() {
            t.h2d_transfer_secs += h2d_time;
            t.kernel_execution_secs += kernel_time;
            t.d2h_transfer_secs += d2h_time;
            t.total_runtime_secs += h2d_time + kernel_time + d2h_time;
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

        let n = x.len() as u32;
        let lower: Vec<f64> = bounds.iter().map(|b| b.lower).collect();
        let upper: Vec<f64> = bounds.iter().map(|b| b.upper).collect();

        let t_h2d_start = Instant::now();
        let d_x = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), x.len() * 8)?;
        let d_lower = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), lower.len() * 8)?;
        let d_upper = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), upper.len() * 8)?;

        d_x.copy_from_host(unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 8)
        })?;
        d_lower.copy_from_host(unsafe {
            std::slice::from_raw_parts(lower.as_ptr() as *const u8, lower.len() * 8)
        })?;
        d_upper.copy_from_host(unsafe {
            std::slice::from_raw_parts(upper.as_ptr() as *const u8, upper.len() * 8)
        })?;
        let h2d_time = t_h2d_start.elapsed().as_secs_f64();

        let t_kernel_start = Instant::now();
        let block_size = 256u32;
        let num_blocks = n.div_ceil(block_size);

        let mut p_n = n;
        let mut p_x = d_x.ptr;
        let mut p_lower = d_lower.ptr;
        let mut p_upper = d_upper.ptr;

        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut p_n as *mut _ as *mut std::ffi::c_void,
            &mut p_x as *mut _ as *mut std::ffi::c_void,
            &mut p_lower as *mut _ as *mut std::ffi::c_void,
            &mut p_upper as *mut _ as *mut std::ffi::c_void,
        ];

        let res = unsafe {
            (self.pipeline.driver.cu_launch_kernel)(
                self.pipeline.k_box_project,
                num_blocks,
                1,
                1,
                block_size,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != CUDA_SUCCESS {
            return Err(SolverError::BackendError(format!(
                "CUDA kernel launch failed for box_project (code {})",
                res
            )));
        }
        self.pipeline.synchronize()?;
        let kernel_time = t_kernel_start.elapsed().as_secs_f64();

        let t_d2h_start = Instant::now();
        d_x.copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u8, x.len() * 8)
        })?;
        let d2h_time = t_d2h_start.elapsed().as_secs_f64();

        if let Ok(mut t) = self.timing.lock() {
            t.h2d_transfer_secs += h2d_time;
            t.kernel_execution_secs += kernel_time;
            t.d2h_transfer_secs += d2h_time;
            t.total_runtime_secs += h2d_time + kernel_time + d2h_time;
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

        let n = x.len() as u32;
        let t_h2d_start = Instant::now();
        let d_x = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), x.len() * 8)?;
        let d_y = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), y.len() * 8)?;
        let d_sum = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), 8)?;

        d_x.copy_from_host(unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 8)
        })?;
        d_y.copy_from_host(unsafe {
            std::slice::from_raw_parts(y.as_ptr() as *const u8, y.len() * 8)
        })?;
        d_sum.zero()?;
        let h2d_time = t_h2d_start.elapsed().as_secs_f64();

        let t_kernel_start = Instant::now();
        let block_size = 256u32;
        let mut num_blocks = n.div_ceil(block_size);
        if num_blocks > 1024 {
            num_blocks = 1024;
        }

        let mut p_n = n;
        let mut p_x = d_x.ptr;
        let mut p_y = d_y.ptr;
        let mut p_sum = d_sum.ptr;

        let mut args: [*mut std::ffi::c_void; 4] = [
            &mut p_n as *mut _ as *mut std::ffi::c_void,
            &mut p_x as *mut _ as *mut std::ffi::c_void,
            &mut p_y as *mut _ as *mut std::ffi::c_void,
            &mut p_sum as *mut _ as *mut std::ffi::c_void,
        ];

        let res = unsafe {
            (self.pipeline.driver.cu_launch_kernel)(
                self.pipeline.k_dot,
                num_blocks,
                1,
                1,
                block_size,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != CUDA_SUCCESS {
            return Err(SolverError::BackendError(format!(
                "CUDA kernel launch failed for dot (code {})",
                res
            )));
        }
        self.pipeline.synchronize()?;
        let kernel_time = t_kernel_start.elapsed().as_secs_f64();

        let mut result = 0.0f64;
        d_sum.copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(&mut result as *mut f64 as *mut u8, 8)
        })?;

        if let Ok(mut t) = self.timing.lock() {
            t.h2d_transfer_secs += h2d_time;
            t.kernel_execution_secs += kernel_time;
            t.total_runtime_secs += h2d_time + kernel_time;
        }

        Ok(result)
    }

    fn norm2(&self, x: &[f64]) -> Result<f64, SolverError> {
        let n = x.len() as u32;
        let t_h2d_start = Instant::now();
        let d_x = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), x.len() * 8)?;
        let d_sum = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), 8)?;

        d_x.copy_from_host(unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 8)
        })?;
        d_sum.zero()?;
        let h2d_time = t_h2d_start.elapsed().as_secs_f64();

        let t_kernel_start = Instant::now();
        let block_size = 256u32;
        let mut num_blocks = n.div_ceil(block_size);
        if num_blocks > 1024 {
            num_blocks = 1024;
        }

        let mut p_n = n;
        let mut p_x = d_x.ptr;
        let mut p_sum = d_sum.ptr;

        let mut args: [*mut std::ffi::c_void; 3] = [
            &mut p_n as *mut _ as *mut std::ffi::c_void,
            &mut p_x as *mut _ as *mut std::ffi::c_void,
            &mut p_sum as *mut _ as *mut std::ffi::c_void,
        ];

        let res = unsafe {
            (self.pipeline.driver.cu_launch_kernel)(
                self.pipeline.k_norm2,
                num_blocks,
                1,
                1,
                block_size,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != CUDA_SUCCESS {
            return Err(SolverError::BackendError(format!(
                "CUDA kernel launch failed for norm2 (code {})",
                res
            )));
        }
        self.pipeline.synchronize()?;
        let kernel_time = t_kernel_start.elapsed().as_secs_f64();

        let mut sum_sq = 0.0f64;
        d_sum.copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(&mut sum_sq as *mut f64 as *mut u8, 8)
        })?;

        if let Ok(mut t) = self.timing.lock() {
            t.h2d_transfer_secs += h2d_time;
            t.kernel_execution_secs += kernel_time;
            t.total_runtime_secs += h2d_time + kernel_time;
        }

        Ok(sum_sq.sqrt())
    }

    fn norm_inf(&self, x: &[f64]) -> Result<f64, SolverError> {
        let n = x.len() as u32;
        let t_h2d_start = Instant::now();
        let d_x = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), x.len() * 8)?;
        let d_max = CudaDeviceBuffer::allocate(self.pipeline.driver.clone(), 8)?;

        d_x.copy_from_host(unsafe {
            std::slice::from_raw_parts(x.as_ptr() as *const u8, x.len() * 8)
        })?;
        d_max.zero()?;
        let h2d_time = t_h2d_start.elapsed().as_secs_f64();

        let t_kernel_start = Instant::now();
        let block_size = 256u32;
        let mut num_blocks = n.div_ceil(block_size);
        if num_blocks > 1024 {
            num_blocks = 1024;
        }

        let mut p_n = n;
        let mut p_x = d_x.ptr;
        let mut p_max = d_max.ptr;

        let mut args: [*mut std::ffi::c_void; 3] = [
            &mut p_n as *mut _ as *mut std::ffi::c_void,
            &mut p_x as *mut _ as *mut std::ffi::c_void,
            &mut p_max as *mut _ as *mut std::ffi::c_void,
        ];

        let res = unsafe {
            (self.pipeline.driver.cu_launch_kernel)(
                self.pipeline.k_norm_inf,
                num_blocks,
                1,
                1,
                block_size,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if res != CUDA_SUCCESS {
            return Err(SolverError::BackendError(format!(
                "CUDA kernel launch failed for norm_inf (code {})",
                res
            )));
        }
        self.pipeline.synchronize()?;
        let kernel_time = t_kernel_start.elapsed().as_secs_f64();

        let mut max_val = 0.0f64;
        d_max.copy_to_host(unsafe {
            std::slice::from_raw_parts_mut(&mut max_val as *mut f64 as *mut u8, 8)
        })?;

        if let Ok(mut t) = self.timing.lock() {
            t.h2d_transfer_secs += h2d_time;
            t.kernel_execution_secs += kernel_time;
            t.total_runtime_secs += h2d_time + kernel_time;
        }

        Ok(max_val)
    }
}

/// Detect whether a genuine CUDA driver and device are operational.
pub fn detect_cuda_hardware() -> Option<BackendInfo> {
    let devices = probe_nvidia_devices()?;
    let dev = &devices[0];
    Some(BackendInfo {
        name: "CUDA (NVIDIA GPU)".into(),
        is_gpu: true,
        device_name: dev.name.clone(),
        details: format!(
            "Device 0: Compute Capability {}.{}; SMs: {}; VRAM: {:.2} GB",
            dev.major,
            dev.minor,
            dev.sm_count,
            (dev.total_memory_bytes as f64) / (1024.0 * 1024.0 * 1024.0)
        ),
    })
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
            Some(_) => match CudaBackend::new() {
                Ok(backend) => {
                    let info = backend.device_info();
                    Ok((Box::new(backend), info))
                }
                Err(e) => Err(e),
            },
            None => Err(SolverError::BackendError(
                "ERR_GPU_UNAVAILABLE: No compatible NVIDIA CUDA device found or CUDA driver not initialized. Faking GPU execution is strictly prohibited.".into(),
            )),
        },
        BackendType::Auto => {
            if detect_cuda_hardware().is_some() {
                if let Ok(backend) = CudaBackend::new() {
                    let info = backend.device_info();
                    return Ok((Box::new(backend), info));
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
    fn test_rust_spmv_precision() {
        let a = sample_sparse_matrix();
        let x = vec![1.2, -3.4, 5.6, -7.8];
        let rust_backend = CpuRustBackend::new();
        let mut y = vec![0.0; a.num_rows];

        rust_backend.spmv(1.0, &a, &x, 0.0, &mut y).unwrap();
        // Row 0: 1.5 * 1.2 + (-2.0) * 5.6 = 1.8 - 11.2 = -9.4
        // Row 1: 3.5 * (-3.4) = -11.9
        // Row 2: -1.0 * 1.2 + 4.2 * (-7.8) = -1.2 - 32.76 = -33.96
        assert!((y[0] - (-9.4)).abs() < 1e-12);
        assert!((y[1] - (-11.9)).abs() < 1e-12);
        assert!((y[2] - (-33.96)).abs() < 1e-12);

        // Test with alpha and beta
        rust_backend.spmv(2.0, &a, &x, -1.0, &mut y).unwrap();
        // y[0] = 2.0 * (-9.4) - (-9.4) = -9.4
        assert!((y[0] - (-9.4)).abs() < 1e-12);
    }

    #[test]
    fn test_rust_spmv_transpose_precision() {
        let a = sample_sparse_matrix();
        let y = vec![2.5, -1.5, 3.0];
        let rust_backend = CpuRustBackend::new();
        let mut x = vec![0.0; a.num_cols];

        rust_backend
            .spmv_transpose(1.0, &a, &y, 0.0, &mut x)
            .unwrap();
        // Col 0: 1.5 * 2.5 + (-1.0) * 3.0 = 3.75 - 3.0 = 0.75
        // Col 1: 3.5 * (-1.5) = -5.25
        // Col 2: -2.0 * 2.5 = -5.0
        // Col 3: 4.2 * 3.0 = 12.6
        assert!((x[0] - 0.75).abs() < 1e-12);
        assert!((x[1] - (-5.25)).abs() < 1e-12);
        assert!((x[2] - (-5.0)).abs() < 1e-12);
        assert!((x[3] - 12.6).abs() < 1e-12);
    }

    #[test]
    fn test_rust_dot_and_norms() {
        let x = vec![1.2, -3.4, 5.6, -7.8, 9.0];
        let y = vec![-0.5, 2.3, -4.1, 1.9, -3.2];
        let rust_backend = CpuRustBackend::new();

        let dot = rust_backend.dot(&x, &y).unwrap();
        // 1.2*(-0.5) + (-3.4)*2.3 + 5.6*(-4.1) + (-7.8)*1.9 + 9.0*(-3.2)
        // = -0.6 - 7.82 - 22.96 - 14.82 - 28.8 = -75.0
        assert!((dot - (-75.0)).abs() < 1e-12);

        let ninf = rust_backend.norm_inf(&x).unwrap();
        assert_eq!(ninf, 9.0);

        let n2 = rust_backend.norm2(&[3.0, 4.0]).unwrap();
        assert_eq!(n2, 5.0);
    }

    #[test]
    fn test_cpu_cpp_backend_alias_compatibility() {
        let backend = CpuCppBackend::new();
        assert_eq!(backend.name(), "CPU (Rust)");
        let x = vec![1.0, 2.0];
        let y = vec![3.0, 4.0];
        let mut z = vec![0.0; 2];
        backend.axpby(1.0, &x, 1.0, &y, &mut z).unwrap();
        assert_eq!(z, vec![4.0, 6.0]);
    }

    #[test]
    fn test_rayon_parallel_spmv() {
        // Construct a 300x50 diagonal sparse matrix (crosses parallel threshold 256)
        let rows = 300;
        let cols = 50;
        let mut row_ptrs = Vec::with_capacity(rows + 1);
        let mut col_indices = Vec::with_capacity(rows);
        let mut values = Vec::with_capacity(rows);

        for i in 0..rows {
            row_ptrs.push(values.len());
            let c = i % cols;
            col_indices.push(c);
            values.push((i + 1) as f64);
        }
        row_ptrs.push(values.len());

        let a = CsrMatrix {
            num_rows: rows,
            num_cols: cols,
            row_ptrs,
            col_indices,
            values,
        };

        let x = vec![1.0; cols];
        let mut y = vec![0.0; rows];

        let backend = CpuRustBackend::new();
        backend.spmv(2.0, &a, &x, 0.0, &mut y).unwrap();

        assert_eq!(y.len(), rows);
        assert!((y[0] - 2.0).abs() < 1e-12);
        assert!((y[299] - 600.0).abs() < 1e-12);
    }
}
