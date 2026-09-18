#include "cuda_common.cuh"
#include <string.h>

__global__ void norm2_kernel(
    int n,
    const double* __restrict__ x,
    double* __restrict__ block_sums
) {
    __shared__ double sdata[BLOCK_SIZE / WARP_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double sum = 0.0;
    while (idx < n) {
        double val = x[idx];
        sum += val * val;
        idx += stride;
    }

    sum = warp_reduce_sum(sum);

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        sdata[warp_id] = sum;
    }
    __syncthreads();

    if (threadIdx.x < (BLOCK_SIZE / WARP_SIZE)) {
        double warp_sum = sdata[threadIdx.x];
        warp_sum = warp_reduce_sum(warp_sum);
        if (threadIdx.x == 0) {
            atomicAdd(block_sums, warp_sum);
        }
    }
}

/* Atomic max for double on CUDA via atomicCAS */
__device__ double atomicMaxDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        double current_val = __longlong_as_double(assumed);
        if (current_val >= val) break;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void norm_inf_kernel(
    int n,
    const double* __restrict__ x,
    double* __restrict__ global_max
) {
    __shared__ double sdata[BLOCK_SIZE / WARP_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    double max_val = 0.0;
    while (idx < n) {
        double val = fabs(x[idx]);
        if (val > max_val) {
            max_val = val;
        }
        idx += stride;
    }

    max_val = warp_reduce_max_abs(max_val);

    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    if (lane == 0) {
        sdata[warp_id] = max_val;
    }
    __syncthreads();

    if (threadIdx.x < (BLOCK_SIZE / WARP_SIZE)) {
        double warp_max = sdata[threadIdx.x];
        warp_max = warp_reduce_max_abs(warp_max);
        if (threadIdx.x == 0) {
            atomicMaxDouble(global_max, warp_max);
        }
    }
}

extern "C" int yutki_cuda_norm2(
    int n,
    const double* d_x,
    double* host_result
) {
    if (n <= 0) {
        *host_result = 0.0;
        return YUTKI_CUDA_SUCCESS;
    }

    double* d_sum = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_sum, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(double)));

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    norm2_kernel<<<num_blocks, BLOCK_SIZE>>>(n, d_x, d_sum);
    CUDA_CHECK(cudaGetLastError());

    double host_sum = 0.0;
    CUDA_CHECK(cudaMemcpy(&host_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sum));

    *host_result = sqrt(host_sum);
    return YUTKI_CUDA_SUCCESS;
}

extern "C" int yutki_cuda_norm_inf(
    int n,
    const double* d_x,
    double* host_result
) {
    if (n <= 0) {
        *host_result = 0.0;
        return YUTKI_CUDA_SUCCESS;
    }

    double* d_max = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_max, sizeof(double)));
    CUDA_CHECK(cudaMemset(d_max, 0, sizeof(double)));

    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (num_blocks > 1024) num_blocks = 1024;

    norm_inf_kernel<<<num_blocks, BLOCK_SIZE>>>(n, d_x, d_max);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(host_result, d_max, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_max));

    return YUTKI_CUDA_SUCCESS;
}

/* Device Management and Memory Allocation */
static YutkiCudaTiming g_last_timing = {0.0, 0.0, 0.0, 0.0};

extern "C" int yutki_cuda_device_count(int* count) {
    cudaError_t err = cudaGetDeviceCount(count);
    if (err != cudaSuccess) {
        *count = 0;
        return YUTKI_CUDA_ERROR_NO_DEVICE;
    }
    return YUTKI_CUDA_SUCCESS;
}

extern "C" int yutki_cuda_device_get_name(int device_id, char* name, int max_len) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return YUTKI_CUDA_ERROR_NO_DEVICE;
    }
    strncpy(name, prop.name, max_len - 1);
    name[max_len - 1] = '\0';
    return YUTKI_CUDA_SUCCESS;
}

extern "C" int yutki_cuda_device_get_info(int device_id, size_t* total_mem, int* sm_count, int* major, int* minor) {
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return YUTKI_CUDA_ERROR_NO_DEVICE;
    }
    *total_mem = prop.totalGlobalMem;
    *sm_count = prop.multiProcessorCount;
    *major = prop.major;
    *minor = prop.minor;
    return YUTKI_CUDA_SUCCESS;
}

extern "C" int yutki_cuda_malloc(void** dev_ptr, size_t size_bytes) {
    cudaError_t err = cudaMalloc(dev_ptr, size_bytes);
    if (err != cudaSuccess) {
        return YUTKI_CUDA_ERROR_OUT_OF_MEMORY;
    }
    return YUTKI_CUDA_SUCCESS;
}

extern "C" int yutki_cuda_free(void* dev_ptr) {
    if (!dev_ptr) return YUTKI_CUDA_SUCCESS;
    cudaError_t err = cudaFree(dev_ptr);
    if (err != cudaSuccess) {
        return YUTKI_CUDA_ERROR_INVALID_VALUE;
    }
    return YUTKI_CUDA_SUCCESS;
}

extern "C" int yutki_cuda_memcpy_h2d(void* dev_dst, const void* host_src, size_t size_bytes) {
    cudaError_t err = cudaMemcpy(dev_dst, host_src, size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        return YUTKI_CUDA_ERROR_LAUNCH_FAILED;
    }
    return YUTKI_CUDA_SUCCESS;
}

extern "C" int yutki_cuda_memcpy_d2h(void* host_dst, const void* dev_src, size_t size_bytes) {
    cudaError_t err = cudaMemcpy(host_dst, dev_src, size_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        return YUTKI_CUDA_ERROR_LAUNCH_FAILED;
    }
    return YUTKI_CUDA_SUCCESS;
}

extern "C" int yutki_cuda_memset(void* dev_ptr, int value, size_t size_bytes) {
    cudaError_t err = cudaMemset(dev_ptr, value, size_bytes);
    if (err != cudaSuccess) {
        return YUTKI_CUDA_ERROR_LAUNCH_FAILED;
    }
    return YUTKI_CUDA_SUCCESS;
}

extern "C" int yutki_cuda_synchronize(void) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return YUTKI_CUDA_ERROR_LAUNCH_FAILED;
    }
    return YUTKI_CUDA_SUCCESS;
}

extern "C" void yutki_cuda_get_last_timing(YutkiCudaTiming* timing) {
    if (timing) {
        *timing = g_last_timing;
    }
}

extern "C" void yutki_cuda_reset_timing(void) {
    g_last_timing.h2d_ms = 0.0;
    g_last_timing.kernel_ms = 0.0;
    g_last_timing.d2h_ms = 0.0;
    g_last_timing.total_ms = 0.0;
}
