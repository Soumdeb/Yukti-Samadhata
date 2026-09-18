//! C ABI FFI bindings to the C++ and CUDA numerical kernels.

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct YutkiCudaTiming {
    pub h2d_ms: f64,
    pub kernel_ms: f64,
    pub d2h_ms: f64,
    pub total_ms: f64,
}

unsafe extern "C" {
    pub fn yutki_cpu_spmv(
        num_rows: usize,
        num_cols: usize,
        row_ptrs: *const usize,
        col_indices: *const usize,
        values: *const f64,
        alpha: f64,
        x: *const f64,
        beta: f64,
        y: *mut f64,
    );

    pub fn yutki_cpu_spmv_transpose(
        num_rows: usize,
        num_cols: usize,
        row_ptrs: *const usize,
        col_indices: *const usize,
        values: *const f64,
        alpha: f64,
        y: *const f64,
        beta: f64,
        x: *mut f64,
    );

    pub fn yutki_cpu_axpby(
        len: usize,
        alpha: f64,
        x: *const f64,
        beta: f64,
        y: *const f64,
        z: *mut f64,
    );

    pub fn yutki_cpu_box_project(len: usize, x: *mut f64, lower: *const f64, upper: *const f64);

    pub fn yutki_cpu_dot(len: usize, x: *const f64, y: *const f64) -> f64;

    pub fn yutki_cpu_norm2(len: usize, x: *const f64) -> f64;

    pub fn yutki_cpu_norm_inf(len: usize, x: *const f64) -> f64;
}

#[cfg(has_cuda_runtime)]
unsafe extern "C" {
    pub fn yutki_cuda_device_count(count: *mut i32) -> i32;
    pub fn yutki_cuda_device_get_name(device_id: i32, name: *mut i8, max_len: i32) -> i32;
    pub fn yutki_cuda_device_get_info(
        device_id: i32,
        total_mem: *mut usize,
        sm_count: *mut i32,
        major: *mut i32,
        minor: *mut i32,
    ) -> i32;
    pub fn yutki_cuda_malloc(dev_ptr: *mut *mut std::ffi::c_void, size_bytes: usize) -> i32;
    pub fn yutki_cuda_free(dev_ptr: *mut std::ffi::c_void) -> i32;
    pub fn yutki_cuda_memcpy_h2d(
        dev_dst: *mut std::ffi::c_void,
        host_src: *const std::ffi::c_void,
        size_bytes: usize,
    ) -> i32;
    pub fn yutki_cuda_memcpy_d2h(
        host_dst: *mut std::ffi::c_void,
        dev_src: *const std::ffi::c_void,
        size_bytes: usize,
    ) -> i32;
    pub fn yutki_cuda_memset(dev_ptr: *mut std::ffi::c_void, value: i32, size_bytes: usize) -> i32;
    pub fn yutki_cuda_synchronize() -> i32;

    pub fn yutki_cuda_spmv(
        num_rows: i32,
        num_cols: i32,
        nnz: i32,
        d_row_ptrs: *const i32,
        d_col_indices: *const i32,
        d_values: *const f64,
        alpha: f64,
        d_x: *const f64,
        beta: f64,
        d_y: *mut f64,
    ) -> i32;

    pub fn yutki_cuda_spmv_transpose(
        num_rows: i32,
        num_cols: i32,
        nnz: i32,
        d_row_ptrs: *const i32,
        d_col_indices: *const i32,
        d_values: *const f64,
        alpha: f64,
        d_y: *const f64,
        beta: f64,
        d_x: *mut f64,
    ) -> i32;

    pub fn yutki_cuda_axpby(
        n: i32,
        alpha: f64,
        d_x: *const f64,
        beta: f64,
        d_y: *const f64,
        d_z: *mut f64,
    ) -> i32;

    pub fn yutki_cuda_box_project(
        n: i32,
        d_x: *mut f64,
        d_lower: *const f64,
        d_upper: *const f64,
    ) -> i32;

    pub fn yutki_cuda_dot(n: i32, d_x: *const f64, d_y: *const f64, host_result: *mut f64) -> i32;

    pub fn yutki_cuda_norm2(n: i32, d_x: *const f64, host_result: *mut f64) -> i32;

    pub fn yutki_cuda_norm_inf(n: i32, d_x: *const f64, host_result: *mut f64) -> i32;

    pub fn yutki_cuda_get_last_timing(timing: *mut YutkiCudaTiming);
    pub fn yutki_cuda_reset_timing();
}
