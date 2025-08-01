#include "../cuda_common/flash_attention_common.h"

// Shared memory for Q, K, V blocks
extern __shared__ float shared_mem[];

// Warp-level softmax reduction
__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Block-level reduction
__inline__ __device__ float block_reduce_max(float val) {
    static __shared__ float shared[32];
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_max(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    
    __syncthreads();
    
    val = (threadIdx.x < blockDim.y) ? shared[lane] : -INFINITY;
    val = warp_reduce_max(val);
    
    return val;
}

__inline__ __device__ float block_reduce_sum(float val) {
    static __shared__ float shared[32];
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared[wid] = val;
    }
    
    __syncthreads();
    
    val = (threadIdx.x < blockDim.y) ? shared[lane] : 0.0f;
    val = warp_reduce_sum(val);
    
    return val;
}

// Flash Attention v1 forward kernel (float32 version)
__global__ void flash_attention_v1_forward_kernel(
    const float* q, const float* k, const float* v, float* o,
    int batch_size, int seq_len, int head_dim,
    int block_size_m, int block_size_n,
    float scale,
    float* lse_ptr
) {
    // Block and thread indices
    int batch_idx = blockIdx.y;
    int row_block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    // Calculate row range for this block
    int row_start = row_block_idx * block_size_m;
    int row_end = min(row_start + block_size_m, seq_len);
    int num_rows = row_end - row_start;
    
    // Shared memory layout
    float* q_block = shared_mem;
    float* k_block = shared_mem + block_size_m * head_dim;
    float* v_block = shared_mem + block_size_m * head_dim + block_size_n * head_dim;
    
    // Registers for accumulation
    float acc_o[32]; // Maximum head dimension we handle in registers
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    // Initialize accumulator
    #pragma unroll
    for (int i = 0; i < 32 && i < head_dim; ++i) {
        acc_o[i] = 0.0f;
    }
    
    // Base pointers for this batch
    const float* batch_q = q + batch_idx * seq_len * head_dim;
    const float* batch_k = k + batch_idx * seq_len * head_dim;
    const float* batch_v = v + batch_idx * seq_len * head_dim;
    float* batch_o = o + batch_idx * seq_len * head_dim;
    float* batch_lse = lse_ptr + batch_idx * seq_len;
    
    // Load Q block for this row block
    if (thread_idx < num_rows * head_dim) {
        int local_row = thread_idx / head_dim;
        int local_col = thread_idx % head_dim;
        int global_row = row_start + local_row;
        q_block[local_row * head_dim + local_col] = batch_q[global_row * head_dim + local_col];
    }
    
    __syncthreads();
    
    // Process K and V in blocks
    for (int col_block_start = 0; col_block_start < seq_len; col_block_start += block_size_n) {
        int col_block_end = min(col_block_start + block_size_n, seq_len);
        int num_cols = col_block_end - col_block_start;
        
        // Load K and V blocks
        if (thread_idx < num_cols * head_dim) {
            int local_row = thread_idx / head_dim;
            int local_col = thread_idx % head_dim;
            int global_row = col_block_start + local_row;
            k_block[local_row * head_dim + local_col] = batch_k[global_row * head_dim + local_col];
            v_block[local_row * head_dim + local_col] = batch_v[global_row * head_dim + local_col];
        }
        
        __syncthreads();
        
        // Process this K, V block for all rows in our Q block
        if (thread_idx < num_rows) {
            int row_idx = row_start + thread_idx;
            
            // Compute Q*K^T for this row
            float s_ij[64]; // Maximum block size for K
            float m_ij = -INFINITY;
            
            // Compute attention scores
            for (int j = 0; j < num_cols; ++j) {
                float sum = 0.0f;
                
                // Dot product
                for (int d = 0; d < head_dim; ++d) {
                    sum += q_block[thread_idx * head_dim + d] * k_block[j * head_dim + d];
                }
                
                s_ij[j] = sum * scale;
                m_ij = fmaxf(m_ij, s_ij[j]);
            }
            
            // Update maximum and compute softmax
            float m_i_new = fmaxf(m_i, m_ij);
            float p_scale = expf(m_i - m_i_new);
            
            // Compute softmax and update output
            float l_i_new = 0.0f;
            
            for (int j = 0; j < num_cols; ++j) {
                float p_ij = expf(s_ij[j] - m_i_new);
                l_i_new += p_ij;
                
                // Update output accumulator
                for (int d = 0; d < head_dim && d < 32; ++d) {
                    acc_o[d] = acc_o[d] * p_scale + p_ij * v_block[j * head_dim + d];
                }
            }
            
            // Update statistics
            m_i = m_i_new;
            l_i = l_i * p_scale + l_i_new;
        }
        
        __syncthreads();
    }
    
    // Final normalization and write output
    if (thread_idx < num_rows) {
        int row_idx = row_start + thread_idx;
        
        // Normalize output
        float norm_factor = 1.0f / l_i;
        for (int d = 0; d < head_dim && d < 32; ++d) {
            acc_o[d] *= norm_factor;
            batch_o[row_idx * head_dim + d] = acc_o[d];
        }
        
        // Store log-sum-exp for numerical stability
        batch_lse[row_idx] = logf(l_i) + m_i;
    }
}

// Half precision version
__global__ void flash_attention_v1_forward_kernel_half(
    const __half* q, const __half* k, const __half* v, __half* o,
    int batch_size, int seq_len, int head_dim,
    int block_size_m, int block_size_n,
    float scale,
    float* lse_ptr
) {
    // Block and thread indices
    int batch_idx = blockIdx.y;
    int row_block_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    
    // Calculate row range for this block
    int row_start = row_block_idx * block_size_m;
    int row_end = min(row_start + block_size_m, seq_len);
    int num_rows = row_end - row_start;
    
    // Shared memory for half precision
    extern __shared__ __half shared_mem_half[];
    __half* q_block = shared_mem_half;
    __half* k_block = shared_mem_half + block_size_m * head_dim;
    __half* v_block = shared_mem_half + block_size_m * head_dim + block_size_n * head_dim;
    
    // Registers for accumulation
    float acc_o[32];
    float m_i = -INFINITY;
    float l_i = 0.0f;
    
    // Initialize accumulator
    #pragma unroll
    for (int i = 0; i < 32 && i < head_dim; ++i) {
        acc_o[i] = 0.0f;
    }
    
    // Base pointers for this batch
    const __half* batch_q = q + batch_idx * seq_len * head_dim;
    const __half* batch_k = k + batch_idx * seq_len * head_dim;
    const __half* batch_v = v + batch_idx * seq_len * head_dim;
    __half* batch_o = o + batch_idx * seq_len * head_dim;
    float* batch_lse = lse_ptr + batch_idx * seq_len;
    
    // Load Q block for this row block
    if (thread_idx < num_rows * head_dim) {
        int local_row = thread_idx / head_dim;
        int local_col = thread_idx % head_dim;
        int global_row = row_start + local_row;
        q_block[local_row * head_dim + local_col] = batch_q[global_row * head_dim + local_col];
    }
    
    __syncthreads();
    
    // Process K and V in blocks
    for (int col_block_start = 0; col_block_start < seq_len; col_block_start += block_size_n) {
        int col_block_end = min(col_block_start + block_size_n, seq_len);
        int num_cols = col_block_end - col_block_start;
        
        // Load K and V blocks
        if (thread_idx < num_cols * head_dim) {
            int local_row = thread_idx / head_dim;
            int local_col = thread_idx % head_dim;
            int global_row = col_block_start + local_row;
            k_block[local_row * head_dim + local_col] = batch_k[global_row * head_dim + local_col];
            v_block[local_row * head_dim + local_col] = batch_v[global_row * head_dim + local_col];
        }
        
        __syncthreads();
        
        // Process this K, V block for all rows in our Q block
        if (thread_idx < num_rows) {
            int row_idx = row_start + thread_idx;
            
            // Compute Q*K^T for this row
            float s_ij[64];
            float m_ij = -INFINITY;
            
            // Compute attention scores
            for (int j = 0; j < num_cols; ++j) {
                float sum = 0.0f;
                
                // Dot product with half precision
                for (int d = 0; d < head_dim; ++d) {
                    float q_val = __half2float(q_block[thread_idx * head_dim + d]);
                    float k_val = __half2float(k_block[j * head_dim + d]);
                    sum += q_val * k_val;
                }
                
                s_ij[j] = sum * scale;
                m_ij = fmaxf(m_ij, s_ij[j]);
            }
            
            // Update maximum and compute softmax
            float m_i_new = fmaxf(m_i, m_ij);
            float p_scale = expf(m_i - m_i_new);
            
            // Compute softmax and update output
            float l_i_new = 0.0f;
            
            for (int j = 0; j < num_cols; ++j) {
                float p_ij = expf(s_ij[j] - m_i_new);
                l_i_new += p_ij;
                
                // Update output accumulator
                for (int d = 0; d < head_dim && d < 32; ++d) {
                    float v_val = __half2float(v_block[j * head_dim + d]);
                    acc_o[d] = acc_o[d] * p_scale + p_ij * v_val;
                }
            }
            
            // Update statistics
            m_i = m_i_new;
            l_i = l_i * p_scale + l_i_new;
        }
        
        __syncthreads();
    }
    
    // Final normalization and write output
    if (thread_idx < num_rows) {
        int row_idx = row_start + thread_idx;
        
        // Normalize output and convert to half
        float norm_factor = 1.0f / l_i;
        for (int d = 0; d < head_dim && d < 32; ++d) {
            acc_o[d] *= norm_factor;
            batch_o[row_idx * head_dim + d] = __float2half(acc_o[d]);
        }
        
        // Store log-sum-exp for numerical stability
        batch_lse[row_idx] = logf(l_i) + m_i;
    }
}