//! Pure Rust CUDA Driver API abstraction and JIT-compiled PTX kernels.
//! Zero C++ dependencies; interacts directly with the NVIDIA display driver (nvcuda.dll / libcuda.so).
//! Strictly adheres to the Sovereign Zero-Falsification Policy.

use std::sync::Arc;
use yukti_model::SolverError;

pub type CUdevice = i32;
pub type CUcontext = *mut std::ffi::c_void;
pub type CUmodule = *mut std::ffi::c_void;
pub type CUfunction = *mut std::ffi::c_void;
pub type CUdeviceptr = u64;
pub type CUresult = i32;

pub const CUDA_SUCCESS: CUresult = 0;

/// Dynamically loaded CUDA Driver API function pointers
#[derive(Clone)]
pub struct CudaDriverFunctions {
    pub cu_init: unsafe extern "C" fn(flags: u32) -> CUresult,
    pub cu_device_get_count: unsafe extern "C" fn(count: *mut i32) -> CUresult,
    pub cu_device_get: unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult,
    pub cu_device_get_name:
        unsafe extern "C" fn(name: *mut i8, len: i32, dev: CUdevice) -> CUresult,
    pub cu_device_total_mem: unsafe extern "C" fn(bytes: *mut usize, dev: CUdevice) -> CUresult,
    pub cu_device_get_attribute:
        unsafe extern "C" fn(pi: *mut i32, attrib: i32, dev: CUdevice) -> CUresult,
    pub cu_ctx_create:
        unsafe extern "C" fn(pctx: *mut CUcontext, flags: u32, dev: CUdevice) -> CUresult,
    pub cu_ctx_destroy: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
    pub cu_ctx_synchronize: unsafe extern "C" fn() -> CUresult,
    pub cu_mem_alloc: unsafe extern "C" fn(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult,
    pub cu_mem_free: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
    pub cu_memcpy_htod: unsafe extern "C" fn(
        dst_device: CUdeviceptr,
        src_host: *const std::ffi::c_void,
        bytes: usize,
    ) -> CUresult,
    pub cu_memcpy_dtoh: unsafe extern "C" fn(
        dst_host: *mut std::ffi::c_void,
        src_device: CUdeviceptr,
        bytes: usize,
    ) -> CUresult,
    pub cu_memset_d8: unsafe extern "C" fn(dst_device: CUdeviceptr, uc: u8, n: usize) -> CUresult,
    pub cu_module_load_data:
        unsafe extern "C" fn(module: *mut CUmodule, image: *const std::ffi::c_void) -> CUresult,
    pub cu_module_unload: unsafe extern "C" fn(hmod: CUmodule) -> CUresult,
    pub cu_module_get_function:
        unsafe extern "C" fn(hfunc: *mut CUfunction, hmod: CUmodule, name: *const i8) -> CUresult,
    pub cu_launch_kernel: unsafe extern "C" fn(
        f: CUfunction,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        h_stream: *mut std::ffi::c_void,
        kernel_params: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void,
    ) -> CUresult,
    _lib_handle: *mut std::ffi::c_void,
}

unsafe impl Send for CudaDriverFunctions {}
unsafe impl Sync for CudaDriverFunctions {}

impl CudaDriverFunctions {
    /// Attempt to dynamically load the system CUDA driver library.
    #[allow(clippy::missing_transmute_annotations)]
    pub fn load() -> Option<Self> {
        #[cfg(windows)]
        unsafe {
            unsafe extern "system" {
                fn LoadLibraryA(lpLibFileName: *const i8) -> *mut std::ffi::c_void;
                fn GetProcAddress(
                    hModule: *mut std::ffi::c_void,
                    lpProcName: *const i8,
                ) -> *mut std::ffi::c_void;
                fn FreeLibrary(hModule: *mut std::ffi::c_void) -> i32;
            }

            let lib_name = b"nvcuda.dll\0";
            let handle = LoadLibraryA(lib_name.as_ptr() as *const i8);
            if handle.is_null() {
                return None;
            }

            macro_rules! get_sym {
                ($name:expr) => {{
                    let ptr = GetProcAddress(handle, $name.as_ptr() as *const i8);
                    if ptr.is_null() {
                        FreeLibrary(handle);
                        return None;
                    }
                    std::mem::transmute(ptr)
                }};
            }

            let funcs = Self {
                cu_init: get_sym!(b"cuInit\0"),
                cu_device_get_count: get_sym!(b"cuDeviceGetCount\0"),
                cu_device_get: get_sym!(b"cuDeviceGet\0"),
                cu_device_get_name: get_sym!(b"cuDeviceGetName\0"),
                cu_device_total_mem: get_sym!(b"cuDeviceTotalMem_v2\0"),
                cu_device_get_attribute: get_sym!(b"cuDeviceGetAttribute\0"),
                cu_ctx_create: get_sym!(b"cuCtxCreate_v2\0"),
                cu_ctx_destroy: get_sym!(b"cuCtxDestroy_v2\0"),
                cu_ctx_synchronize: get_sym!(b"cuCtxSynchronize\0"),
                cu_mem_alloc: get_sym!(b"cuMemAlloc_v2\0"),
                cu_mem_free: get_sym!(b"cuMemFree_v2\0"),
                cu_memcpy_htod: get_sym!(b"cuMemcpyHtoD_v2\0"),
                cu_memcpy_dtoh: get_sym!(b"cuMemcpyDtoH_v2\0"),
                cu_memset_d8: get_sym!(b"cuMemsetD8_v2\0"),
                cu_module_load_data: get_sym!(b"cuModuleLoadData\0"),
                cu_module_unload: get_sym!(b"cuModuleUnload\0"),
                cu_module_get_function: get_sym!(b"cuModuleGetFunction\0"),
                cu_launch_kernel: get_sym!(b"cuLaunchKernel\0"),
                _lib_handle: handle,
            };

            // Initialize CUDA driver
            if (funcs.cu_init)(0) != CUDA_SUCCESS {
                FreeLibrary(handle);
                return None;
            }

            Some(funcs)
        }

        #[cfg(not(windows))]
        unsafe {
            unsafe extern "C" {
                fn dlopen(filename: *const i8, flag: i32) -> *mut std::ffi::c_void;
                fn dlsym(handle: *mut std::ffi::c_void, symbol: *const i8)
                    -> *mut std::ffi::c_void;
                fn dlclose(handle: *mut std::ffi::c_void) -> i32;
            }

            const RTLD_NOW: i32 = 2;
            let lib_name = b"libcuda.so.1\0";
            let mut handle = dlopen(lib_name.as_ptr() as *const i8, RTLD_NOW);
            if handle.is_null() {
                let alt_name = b"libcuda.so\0";
                handle = dlopen(alt_name.as_ptr() as *const i8, RTLD_NOW);
            }
            if handle.is_null() {
                return None;
            }

            macro_rules! get_sym {
                ($name:expr) => {{
                    let ptr = dlsym(handle, $name.as_ptr() as *const i8);
                    if ptr.is_null() {
                        dlclose(handle);
                        return None;
                    }
                    std::mem::transmute(ptr)
                }};
            }

            let funcs = Self {
                cu_init: get_sym!(b"cuInit\0"),
                cu_device_get_count: get_sym!(b"cuDeviceGetCount\0"),
                cu_device_get: get_sym!(b"cuDeviceGet\0"),
                cu_device_get_name: get_sym!(b"cuDeviceGetName\0"),
                cu_device_total_mem: get_sym!(b"cuDeviceTotalMem_v2\0"),
                cu_device_get_attribute: get_sym!(b"cuDeviceGetAttribute\0"),
                cu_ctx_create: get_sym!(b"cuCtxCreate_v2\0"),
                cu_ctx_destroy: get_sym!(b"cuCtxDestroy_v2\0"),
                cu_ctx_synchronize: get_sym!(b"cuCtxSynchronize\0"),
                cu_mem_alloc: get_sym!(b"cuMemAlloc_v2\0"),
                cu_mem_free: get_sym!(b"cuMemFree_v2\0"),
                cu_memcpy_htod: get_sym!(b"cuMemcpyHtoD_v2\0"),
                cu_memcpy_dtoh: get_sym!(b"cuMemcpyDtoH_v2\0"),
                cu_memset_d8: get_sym!(b"cuMemsetD8_v2\0"),
                cu_module_load_data: get_sym!(b"cuModuleLoadData\0"),
                cu_module_unload: get_sym!(b"cuModuleUnload\0"),
                cu_module_get_function: get_sym!(b"cuModuleGetFunction\0"),
                cu_launch_kernel: get_sym!(b"cuLaunchKernel\0"),
                _lib_handle: handle,
            };

            if (funcs.cu_init)(0) != CUDA_SUCCESS {
                dlclose(handle);
                return None;
            }

            Some(funcs)
        }
    }
}

/// Discovered GPU Device Metadata
#[derive(Debug, Clone)]
pub struct CudaDeviceDetails {
    pub device_id: i32,
    pub name: String,
    pub total_memory_bytes: usize,
    pub sm_count: i32,
    pub major: i32,
    pub minor: i32,
}

/// Query connected physical NVIDIA devices using the genuine CUDA driver.
pub fn probe_nvidia_devices() -> Option<Vec<CudaDeviceDetails>> {
    let funcs = CudaDriverFunctions::load()?;
    let mut count: i32 = 0;
    let res = unsafe { (funcs.cu_device_get_count)(&mut count) };
    if res != CUDA_SUCCESS || count <= 0 {
        return None;
    }

    let mut devices = Vec::new();
    for i in 0..count {
        let mut dev: CUdevice = 0;
        if unsafe { (funcs.cu_device_get)(&mut dev, i) } != CUDA_SUCCESS {
            continue;
        }

        let mut name_buf = [0i8; 256];
        let _ = unsafe { (funcs.cu_device_get_name)(name_buf.as_mut_ptr(), 256, dev) };
        let name = unsafe { std::ffi::CStr::from_ptr(name_buf.as_ptr()) }
            .to_str()
            .unwrap_or("NVIDIA GPU")
            .trim()
            .to_string();

        let mut mem: usize = 0;
        let _ = unsafe { (funcs.cu_device_total_mem)(&mut mem, dev) };

        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        let mut sms: i32 = 0;
        let _ = unsafe { (funcs.cu_device_get_attribute)(&mut major, 75, dev) };
        let _ = unsafe { (funcs.cu_device_get_attribute)(&mut minor, 76, dev) };
        let _ = unsafe { (funcs.cu_device_get_attribute)(&mut sms, 16, dev) };

        devices.push(CudaDeviceDetails {
            device_id: i,
            name,
            total_memory_bytes: mem,
            sm_count: sms,
            major,
            minor,
        });
    }

    if devices.is_empty() {
        None
    } else {
        Some(devices)
    }
}

/// Safe RAII GPU Memory Buffer
pub struct CudaDeviceBuffer {
    driver: Arc<CudaDriverFunctions>,
    pub ptr: CUdeviceptr,
    pub size_bytes: usize,
}

impl CudaDeviceBuffer {
    pub fn allocate(
        driver: Arc<CudaDriverFunctions>,
        size_bytes: usize,
    ) -> Result<Self, SolverError> {
        let mut ptr: CUdeviceptr = 0;
        let res = unsafe { (driver.cu_mem_alloc)(&mut ptr, size_bytes) };
        if res != CUDA_SUCCESS {
            return Err(SolverError::BackendError(format!(
                "CUDA out of memory: failed to allocate {} bytes (code {})",
                size_bytes, res
            )));
        }
        Ok(Self {
            driver,
            ptr,
            size_bytes,
        })
    }

    pub fn copy_from_host(&self, host_slice: &[u8]) -> Result<(), SolverError> {
        if host_slice.len() > self.size_bytes {
            return Err(SolverError::BackendError(
                "CUDA copy buffer overflow".into(),
            ));
        }
        let res = unsafe {
            (self.driver.cu_memcpy_htod)(
                self.ptr,
                host_slice.as_ptr() as *const std::ffi::c_void,
                host_slice.len(),
            )
        };
        if res != CUDA_SUCCESS {
            return Err(SolverError::BackendError(format!(
                "CUDA Host-to-Device transfer failed (code {})",
                res
            )));
        }
        Ok(())
    }

    pub fn copy_to_host(&self, host_slice: &mut [u8]) -> Result<(), SolverError> {
        if host_slice.len() > self.size_bytes {
            return Err(SolverError::BackendError(
                "CUDA copy buffer underflow".into(),
            ));
        }
        let res = unsafe {
            (self.driver.cu_memcpy_dtoh)(
                host_slice.as_mut_ptr() as *mut std::ffi::c_void,
                self.ptr,
                host_slice.len(),
            )
        };
        if res != CUDA_SUCCESS {
            return Err(SolverError::BackendError(format!(
                "CUDA Device-to-Host transfer failed (code {})",
                res
            )));
        }
        Ok(())
    }

    pub fn zero(&self) -> Result<(), SolverError> {
        let res = unsafe { (self.driver.cu_memset_d8)(self.ptr, 0, self.size_bytes) };
        if res != CUDA_SUCCESS {
            return Err(SolverError::BackendError(format!(
                "CUDA memset failed (code {})",
                res
            )));
        }
        Ok(())
    }
}

impl Drop for CudaDeviceBuffer {
    fn drop(&mut self) {
        if self.ptr != 0 {
            unsafe {
                let _ = (self.driver.cu_mem_free)(self.ptr);
            }
        }
    }
}

/// Embedded, universal PTX numerical kernels for sparse linear algebra and PDHG vector primitives.
/// The NVIDIA driver JIT compiler compiles this at runtime into native hardware SASS.
pub const YUKTI_CUDA_PTX: &str = r#".version 6.0
.target sm_60
.address_size 64

// 1. SpMV CSR Scalar: y = alpha * A * x + beta * y
.visible .entry spmv_csr_kernel(
    .param .u32 num_rows,
    .param .u64 row_ptrs,
    .param .u64 col_indices,
    .param .u64 values,
    .param .f64 alpha,
    .param .u64 x,
    .param .f64 beta,
    .param .u64 y
) {
    .reg .pred %p<4>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<20>;
    .reg .f64 %f<10>;

    ld.param.u32 %r0, [num_rows];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    setp.ge.s32 %p0, %r4, %r0;
    @%p0 ret;

    cvt.s64.s32 %rd0, %r4;
    shl.b64 %rd1, %rd0, 2;
    ld.param.u64 %rd2, [row_ptrs];
    add.s64 %rd3, %rd2, %rd1;
    ld.global.s32 %r5, [%rd3];
    ld.global.s32 %r6, [%rd3 + 4];

    mov.f64 %f0, 0d0000000000000000;
    mov.s32 %r7, %r5;

$L_spmv_loop:
    setp.ge.s32 %p1, %r7, %r6;
    @%p1 bra $L_spmv_done;

    cvt.s64.s32 %rd4, %r7;
    shl.b64 %rd5, %rd4, 2;
    ld.param.u64 %rd6, [col_indices];
    add.s64 %rd7, %rd6, %rd5;
    ld.global.s32 %r8, [%rd7];

    shl.b64 %rd8, %rd4, 3;
    ld.param.u64 %rd9, [values];
    add.s64 %rd10, %rd9, %rd8;
    ld.global.f64 %f1, [%rd10];

    cvt.s64.s32 %rd11, %r8;
    shl.b64 %rd12, %rd11, 3;
    ld.param.u64 %rd13, [x];
    add.s64 %rd14, %rd13, %rd12;
    ld.global.f64 %f2, [%rd14];

    fma.rn.f64 %f0, %f1, %f2, %f0;

    add.s32 %r7, %r7, 1;
    bra $L_spmv_loop;

$L_spmv_done:
    ld.param.f64 %f3, [alpha];
    mul.f64 %f4, %f3, %f0;

    ld.param.f64 %f5, [beta];
    setp.eq.f64 %p2, %f5, 0d0000000000000000;

    ld.param.u64 %rd15, [y];
    shl.b64 %rd16, %rd0, 3;
    add.s64 %rd17, %rd15, %rd16;

    @%p2 bra $L_spmv_write_zero_beta;

    ld.global.f64 %f6, [%rd17];
    fma.rn.f64 %f4, %f5, %f6, %f4;

$L_spmv_write_zero_beta:
    st.global.f64 [%rd17], %f4;
    ret;
}

// 2. Vector scaling: x = beta * x
.visible .entry scale_vector_kernel(
    .param .u32 n,
    .param .f64 beta,
    .param .u64 x
) {
    .reg .pred %p0;
    .reg .b32 %r<5>;
    .reg .b64 %rd<6>;
    .reg .f64 %f<3>;

    ld.param.u32 %r0, [n];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    setp.ge.s32 %p0, %r4, %r0;
    @%p0 ret;

    cvt.s64.s32 %rd0, %r4;
    shl.b64 %rd1, %rd0, 3;
    ld.param.u64 %rd2, [x];
    add.s64 %rd3, %rd2, %rd1;
    ld.global.f64 %f0, [%rd3];
    ld.param.f64 %f1, [beta];
    mul.f64 %f2, %f0, %f1;
    st.global.f64 [%rd3], %f2;
    ret;
}

// 3. SpMV Transpose Scatter: atomic accumulation of scaled A^T * y
.visible .entry spmv_csr_transpose_kernel(
    .param .u32 num_rows,
    .param .u64 row_ptrs,
    .param .u64 col_indices,
    .param .u64 values,
    .param .f64 alpha,
    .param .u64 y,
    .param .u64 x
) {
    .reg .pred %p<3>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<20>;
    .reg .f64 %f<8>;

    ld.param.u32 %r0, [num_rows];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    setp.ge.s32 %p0, %r4, %r0;
    @%p0 ret;

    cvt.s64.s32 %rd0, %r4;
    shl.b64 %rd1, %rd0, 3;
    ld.param.u64 %rd2, [y];
    add.s64 %rd3, %rd2, %rd1;
    ld.global.f64 %f0, [%rd3];

    setp.eq.f64 %p1, %f0, 0d0000000000000000;
    @%p1 ret;

    ld.param.f64 %f1, [alpha];
    mul.f64 %f2, %f0, %f1;

    shl.b64 %rd4, %rd0, 2;
    ld.param.u64 %rd5, [row_ptrs];
    add.s64 %rd6, %rd5, %rd4;
    ld.global.s32 %r5, [%rd6];
    ld.global.s32 %r6, [%rd6 + 4];

    mov.s32 %r7, %r5;
$L_spt_loop:
    setp.ge.s32 %p2, %r7, %r6;
    @%p2 ret;

    cvt.s64.s32 %rd7, %r7;
    shl.b64 %rd8, %rd7, 2;
    ld.param.u64 %rd9, [col_indices];
    add.s64 %rd10, %rd9, %rd8;
    ld.global.s32 %r8, [%rd10];

    shl.b64 %rd11, %rd7, 3;
    ld.param.u64 %rd12, [values];
    add.s64 %rd13, %rd12, %rd11;
    ld.global.f64 %f3, [%rd13];

    mul.f64 %f4, %f2, %f3;

    cvt.s64.s32 %rd14, %r8;
    shl.b64 %rd15, %rd14, 3;
    ld.param.u64 %rd16, [x];
    add.s64 %rd17, %rd16, %rd15;

    atom.global.add.f64 %f5, [%rd17], %f4;

    add.s32 %r7, %r7, 1;
    bra $L_spt_loop;
}

// 4. AXPBY: z = alpha * x + beta * y
.visible .entry axpby_kernel(
    .param .u32 n,
    .param .f64 alpha,
    .param .u64 x,
    .param .f64 beta,
    .param .u64 y,
    .param .u64 z
) {
    .reg .pred %p0;
    .reg .b32 %r<5>;
    .reg .b64 %rd<10>;
    .reg .f64 %f<6>;

    ld.param.u32 %r0, [n];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    setp.ge.s32 %p0, %r4, %r0;
    @%p0 ret;

    cvt.s64.s32 %rd0, %r4;
    shl.b64 %rd1, %rd0, 3;

    ld.param.u64 %rd2, [x];
    add.s64 %rd3, %rd2, %rd1;
    ld.global.f64 %f0, [%rd3];

    ld.param.u64 %rd4, [y];
    add.s64 %rd5, %rd4, %rd1;
    ld.global.f64 %f1, [%rd5];

    ld.param.f64 %f2, [alpha];
    ld.param.f64 %f3, [beta];
    mul.f64 %f4, %f2, %f0;
    fma.rn.f64 %f5, %f3, %f1, %f4;

    ld.param.u64 %rd6, [z];
    add.s64 %rd7, %rd6, %rd1;
    st.global.f64 [%rd7], %f5;
    ret;
}

// 5. Box Projection: x = clamp(x, lower, upper)
.visible .entry box_project_kernel(
    .param .u32 n,
    .param .u64 x,
    .param .u64 lower,
    .param .u64 upper
) {
    .reg .pred %p<3>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<8>;
    .reg .f64 %f<4>;

    ld.param.u32 %r0, [n];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;
    setp.ge.s32 %p0, %r4, %r0;
    @%p0 ret;

    cvt.s64.s32 %rd0, %r4;
    shl.b64 %rd1, %rd0, 3;

    ld.param.u64 %rd2, [x];
    add.s64 %rd3, %rd2, %rd1;
    ld.global.f64 %f0, [%rd3];

    ld.param.u64 %rd4, [lower];
    add.s64 %rd5, %rd4, %rd1;
    ld.global.f64 %f1, [%rd5];

    ld.param.u64 %rd6, [upper];
    add.s64 %rd7, %rd6, %rd1;
    ld.global.f64 %f2, [%rd7];

    setp.lt.f64 %p1, %f0, %f1;
    selp.f64 %f3, %f1, %f0, %p1;

    setp.gt.f64 %p2, %f3, %f2;
    selp.f64 %f3, %f2, %f3, %p2;

    st.global.f64 [%rd3], %f3;
    ret;
}

// 6. Dot Reduction: atomic accumulator
.visible .entry dot_kernel(
    .param .u32 n,
    .param .u64 x,
    .param .u64 y,
    .param .u64 out_sum
) {
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    .reg .f64 %f<4>;

    ld.param.u32 %r0, [n];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;

    mov.u32 %r5, %nctaid.x;
    mul.lo.s32 %r5, %r5, %r2;

    mov.f64 %f0, 0d0000000000000000;

$L_dot_loop:
    setp.ge.s32 %p0, %r4, %r0;
    @%p0 bra $L_dot_accum;

    cvt.s64.s32 %rd0, %r4;
    shl.b64 %rd1, %rd0, 3;

    ld.param.u64 %rd2, [x];
    add.s64 %rd3, %rd2, %rd1;
    ld.global.f64 %f1, [%rd3];

    ld.param.u64 %rd4, [y];
    add.s64 %rd5, %rd4, %rd1;
    ld.global.f64 %f2, [%rd5];

    fma.rn.f64 %f0, %f1, %f2, %f0;

    add.s32 %r4, %r4, %r5;
    bra $L_dot_loop;

$L_dot_accum:
    ld.param.u64 %rd6, [out_sum];
    atom.global.add.f64 %f3, [%rd6], %f0;
    ret;
}

// 7. L2 Norm Reduction: atomic accumulator of squared sum
.visible .entry norm2_kernel(
    .param .u32 n,
    .param .u64 x,
    .param .u64 out_sum
) {
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    .reg .f64 %f<4>;

    ld.param.u32 %r0, [n];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;

    mov.u32 %r5, %nctaid.x;
    mul.lo.s32 %r5, %r5, %r2;

    mov.f64 %f0, 0d0000000000000000;

$L_norm2_loop:
    setp.ge.s32 %p0, %r4, %r0;
    @%p0 bra $L_norm2_accum;

    cvt.s64.s32 %rd0, %r4;
    shl.b64 %rd1, %rd0, 3;

    ld.param.u64 %rd2, [x];
    add.s64 %rd3, %rd2, %rd1;
    ld.global.f64 %f1, [%rd3];

    fma.rn.f64 %f0, %f1, %f1, %f0;

    add.s32 %r4, %r4, %r5;
    bra $L_norm2_loop;

$L_norm2_accum:
    ld.param.u64 %rd4, [out_sum];
    atom.global.add.f64 %f3, [%rd4], %f0;
    ret;
}

// 8. Infinity Norm Reduction: CAS atomic max on abs(x)
.visible .entry norm_inf_kernel(
    .param .u32 n,
    .param .u64 x,
    .param .u64 out_max
) {
    .reg .pred %p<3>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<10>;
    .reg .f64 %f<4>;

    ld.param.u32 %r0, [n];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.s32 %r4, %r1, %r2, %r3;

    mov.u32 %r5, %nctaid.x;
    mul.lo.s32 %r5, %r5, %r2;

    mov.f64 %f0, 0d0000000000000000;

$L_inf_loop:
    setp.ge.s32 %p0, %r4, %r0;
    @%p0 bra $L_inf_accum;

    cvt.s64.s32 %rd0, %r4;
    shl.b64 %rd1, %rd0, 3;

    ld.param.u64 %rd2, [x];
    add.s64 %rd3, %rd2, %rd1;
    ld.global.f64 %f1, [%rd3];
    abs.f64 %f2, %f1;

    setp.gt.f64 %p1, %f2, %f0;
    selp.f64 %f0, %f2, %f0, %p1;

    add.s32 %r4, %r4, %r5;
    bra $L_inf_loop;

$L_inf_accum:
    ld.param.u64 %rd4, [out_max];
$L_cas_loop:
    ld.global.f64 %f3, [%rd4];
    setp.le.f64 %p2, %f0, %f3;
    @%p2 ret;

    mov.b64 %rd5, %f3;
    mov.b64 %rd6, %f0;
    atom.global.cas.b64 %rd7, [%rd4], %rd5, %rd6;
    setp.eq.b64 %p2, %rd7, %rd5;
    @!%p2 bra $L_cas_loop;
    ret;
}
\0"#;

/// Compiled module and kernel pointers loaded into genuine NVIDIA GPU memory
pub struct CudaPipeline {
    pub driver: Arc<CudaDriverFunctions>,
    pub context: CUcontext,
    pub module: CUmodule,
    pub k_spmv: CUfunction,
    pub k_scale: CUfunction,
    pub k_spmv_transpose: CUfunction,
    pub k_axpby: CUfunction,
    pub k_box_project: CUfunction,
    pub k_dot: CUfunction,
    pub k_norm2: CUfunction,
    pub k_norm_inf: CUfunction,
}

impl CudaPipeline {
    pub fn new(device_id: i32) -> Result<Self, SolverError> {
        let funcs = CudaDriverFunctions::load().ok_or_else(|| {
            SolverError::BackendError(
                "ERR_GPU_UNAVAILABLE: CUDA driver not found or failed to initialize.".into(),
            )
        })?;
        let driver = Arc::new(funcs);

        let mut dev: CUdevice = 0;
        let res = unsafe { (driver.cu_device_get)(&mut dev, device_id) };
        if res != CUDA_SUCCESS {
            return Err(SolverError::BackendError(format!(
                "ERR_GPU_UNAVAILABLE: Failed to get CUDA device {} (code {})",
                device_id, res
            )));
        }

        let mut context: CUcontext = std::ptr::null_mut();
        let res = unsafe { (driver.cu_ctx_create)(&mut context, 0, dev) };
        if res != CUDA_SUCCESS || context.is_null() {
            return Err(SolverError::BackendError(format!(
                "ERR_GPU_UNAVAILABLE: Failed to create CUDA context on device {} (code {})",
                device_id, res
            )));
        }

        let mut module: CUmodule = std::ptr::null_mut();
        let res = unsafe {
            (driver.cu_module_load_data)(&mut module, YUKTI_CUDA_PTX.as_ptr() as *const _)
        };
        if res != CUDA_SUCCESS || module.is_null() {
            unsafe {
                (driver.cu_ctx_destroy)(context);
            }
            return Err(SolverError::BackendError(format!(
                "CUDA driver JIT compilation failed for PTX module (code {})",
                res
            )));
        }

        macro_rules! get_kernel {
            ($name:expr) => {{
                let mut f: CUfunction = std::ptr::null_mut();
                let res = unsafe {
                    (driver.cu_module_get_function)(&mut f, module, $name.as_ptr() as *const i8)
                };
                if res != CUDA_SUCCESS || f.is_null() {
                    unsafe {
                        (driver.cu_module_unload)(module);
                        (driver.cu_ctx_destroy)(context);
                    }
                    return Err(SolverError::BackendError(format!(
                        "Failed to locate kernel '{}' in compiled PTX module (code {})",
                        std::str::from_utf8($name).unwrap_or("?"),
                        res
                    )));
                }
                f
            }};
        }

        let k_spmv = get_kernel!(b"spmv_csr_kernel\0");
        let k_scale = get_kernel!(b"scale_vector_kernel\0");
        let k_spmv_transpose = get_kernel!(b"spmv_csr_transpose_kernel\0");
        let k_axpby = get_kernel!(b"axpby_kernel\0");
        let k_box_project = get_kernel!(b"box_project_kernel\0");
        let k_dot = get_kernel!(b"dot_kernel\0");
        let k_norm2 = get_kernel!(b"norm2_kernel\0");
        let k_norm_inf = get_kernel!(b"norm_inf_kernel\0");

        Ok(Self {
            driver,
            context,
            module,
            k_spmv,
            k_scale,
            k_spmv_transpose,
            k_axpby,
            k_box_project,
            k_dot,
            k_norm2,
            k_norm_inf,
        })
    }

    pub fn synchronize(&self) -> Result<(), SolverError> {
        let res = unsafe { (self.driver.cu_ctx_synchronize)() };
        if res != CUDA_SUCCESS {
            return Err(SolverError::BackendError(format!(
                "CUDA context synchronization failed (code {})",
                res
            )));
        }
        Ok(())
    }
}

impl Drop for CudaPipeline {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                let _ = (self.driver.cu_module_unload)(self.module);
            }
        }
        if !self.context.is_null() {
            unsafe {
                let _ = (self.driver.cu_ctx_destroy)(self.context);
            }
        }
    }
}

unsafe impl Send for CudaPipeline {}
unsafe impl Sync for CudaPipeline {}
unsafe impl Send for CudaDeviceBuffer {}
unsafe impl Sync for CudaDeviceBuffer {}
