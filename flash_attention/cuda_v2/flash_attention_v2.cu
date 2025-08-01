#include "flash_attention_v2.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

// CUDA kernel declarations
__global__ void flash_attention_v2_forward_kernel(
    const half* q, const half* k, const half* v, half* o, half* l,
    int batch_size, int seq_len, int num_heads, int head_dim,
    bool use_causal_mask, float scale
);

__global__ void flash_attention_v2_backward_kernel(
    const half* q, const half* k, const half* v, const half* o, const half* l,
    const half* do_ptr, half* dq, half* dk, half* dv,
    int batch_size, int seq_len, int num_heads, int head_dim,
    bool use_causal_mask, float scale
);

// Utility functions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared_mem) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Warp reduction
    val = warp_reduce_sum(val);
    
    // Write warp result to shared memory
    if (lane_id == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();
    
    // Reduce across warps
    if (warp_id == 0) {
        val = (tid < blockDim.x / 32) ? shared_mem[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// Flash Attention v2 forward implementation
cudaError_t flash_attention_v2_forward(
    const half* q, const half* k, const half* v, half* o, half* l,
    const FlashAttentionV2Config& config, cudaStream_t stream
) {
    // Validate configuration
    if (!is_config_valid(config)) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate scale factor
    float scale = 1.0f / sqrtf(static_cast<float>(config.head_dim));
    
    // Configure kernel launch parameters
    dim3 block(128, 1, 1);  // 128 threads per block
    dim3 grid(
        (config.seq_len + block.x - 1) / block.x,
        config.num_heads,
        config.batch_size
    );
    
    // No shared memory needed for simplified implementation
    int shared_mem_size = 0;
    
    // Launch forward kernel
    flash_attention_v2_forward_kernel<<<grid, block, shared_mem_size, stream>>>(
        q, k, v, o, l,
        config.batch_size, config.seq_len, config.num_heads, config.head_dim,
        config.use_causal_mask, scale
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    
    // Synchronize if no stream is provided
    if (stream == nullptr) {
        err = cudaDeviceSynchronize();
    }
    
    return err;
}

// Flash Attention v2 backward implementation
cudaError_t flash_attention_v2_backward(
    const half* q, const half* k, const half* v, const half* o, const half* l,
    const half* do_grad, half* dq, half* dk, half* dv,
    const FlashAttentionV2Config& config, cudaStream_t stream
) {
    // Validate configuration
    if (!is_config_valid(config)) {
        return cudaErrorInvalidValue;
    }
    
    // Calculate scale factor
    float scale = 1.0f / sqrtf(static_cast<float>(config.head_dim));
    
    // Configure kernel launch parameters
    dim3 block(128, 1, 1);
    dim3 grid(
        (config.seq_len + block.x - 1) / block.x,
        config.num_heads,
        config.batch_size
    );
    
    // No shared memory needed for simplified implementation
    int shared_mem_size = 0;
    
    // Launch backward kernel
    flash_attention_v2_backward_kernel<<<grid, block, shared_mem_size, stream>>>(
        q, k, v, o, l, do_grad, dq, dk, dv,
        config.batch_size, config.seq_len, config.num_heads, config.head_dim,
        config.use_causal_mask, scale
    );
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    
    // Synchronize if no stream is provided
    if (stream == nullptr) {
        err = cudaDeviceSynchronize();
    }
    
    return err;
}

// Configuration validation
bool is_config_valid(const FlashAttentionV2Config& config) {
    // Check basic parameters
    if (config.batch_size <= 0 || config.batch_size > 1024) {
        printf("Invalid batch_size: %d\n", config.batch_size);
        return false;
    }
    
    if (config.seq_len <= 0 || config.seq_len > 8192) {
        printf("Invalid seq_len: %d\n", config.seq_len);
        return false;
    }
    
    if (config.num_heads <= 0 || config.num_heads > 128) {
        printf("Invalid num_heads: %d\n", config.num_heads);
        return false;
    }
    
    if (config.head_dim <= 0 || config.head_dim > 256) {
        printf("Invalid head_dim: %d\n", config.head_dim);
        return false;
    }
    
    // Check dropout rate
    if (config.dropout_rate < 0.0f || config.dropout_rate >= 1.0f) {
        printf("Invalid dropout_rate: %f\n", config.dropout_rate);
        return false;
    }
    
    return true;
}


// Initialization and cleanup
cudaError_t flash_attention_v2_init() {
    // Initialize any required resources
    return cudaSuccess;
}

cudaError_t flash_attention_v2_cleanup() {
    // Clean up any allocated resources
    return cudaSuccess;
}