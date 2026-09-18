use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_cuda_runtime)");

    println!("cargo:rerun-if-changed=../../cpp/cpu/sparse_ops.hpp");
    println!("cargo:rerun-if-changed=../../cpp/cpu/sparse_ops.cpp");
    println!("cargo:rerun-if-changed=../../cuda/kernels/cuda_kernels.h");
    println!("cargo:rerun-if-changed=../../cuda/kernels/cuda_common.cuh");
    println!("cargo:rerun-if-changed=../../cuda/kernels/spmv_csr.cu");
    println!("cargo:rerun-if-changed=../../cuda/kernels/spmv_csr_transpose.cu");
    println!("cargo:rerun-if-changed=../../cuda/kernels/vector_ops.cu");
    println!("cargo:rerun-if-changed=../../cuda/kernels/dot_reduction.cu");
    println!("cargo:rerun-if-changed=../../cuda/kernels/norm_reduction.cu");

    // 1. Always build the native C++ CPU numerical kernels
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file("../../cpp/cpu/sparse_ops.cpp")
        .include("../../cpp/cpu")
        .compile("yutki_cpu_ops");

    // 2. Check for nvcc compiler and build CUDA kernels if available
    let nvcc_available = Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    if nvcc_available {
        println!("cargo:warning=NVIDIA CUDA Compiler (nvcc) detected. Compiling CUDA kernels...");
        let mut cuda_build = cc::Build::new();
        cuda_build
            .cuda(true)
            .file("../../cuda/kernels/spmv_csr.cu")
            .file("../../cuda/kernels/spmv_csr_transpose.cu")
            .file("../../cuda/kernels/vector_ops.cu")
            .file("../../cuda/kernels/dot_reduction.cu")
            .file("../../cuda/kernels/norm_reduction.cu")
            .include("../../cuda/kernels");

        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            cuda_build.include(format!("{}/include", cuda_path));
        }

        cuda_build.compile("yutki_cuda_kernels");
        println!("cargo:rustc-cfg=has_cuda_runtime");
        println!("cargo:rustc-link-lib=cudart");
    } else {
        println!("cargo:warning=CUDA compiler (nvcc) not found. Building in CPU-only mode with sovereign zero-falsification.");
    }
}
