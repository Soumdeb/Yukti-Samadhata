//! C ABI FFI timing structures and compatibility declarations.
//! Strictly pure Rust with zero C++ declarations.

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct YutkiCudaTiming {
    pub h2d_ms: f64,
    pub kernel_ms: f64,
    pub d2h_ms: f64,
    pub total_ms: f64,
}
