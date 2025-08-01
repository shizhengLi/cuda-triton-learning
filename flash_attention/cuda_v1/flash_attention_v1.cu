#include "../cuda_common/flash_attention_common.h"
#include <iostream>
#include <vector>
#include <algorithm>

// Forward declarations for kernel functions
__global__ void flash_attention_v1_forward_kernel(
    const float* q, const float* k, const float* v, float* o,
    int batch_size, int seq_len, int head_dim,
    int block_size_m, int block_size_n,
    float scale,
    float* lse_ptr  // log sum exp for numerical stability
);

__global__ void flash_attention_v1_forward_kernel_half(
    const __half* q, const __half* k, const __half* v, __half* o,
    int batch_size, int seq_len, int head_dim,
    int block_size_m, int block_size_n,
    float scale,
    float* lse_ptr
);

class FlashAttentionV1 {
public:
    FlashAttentionV1(int batch_size, int seq_len, int head_dim, 
                    DataType dtype = DataType::FLOAT32)
        : batch_size_(batch_size), seq_len_(seq_len), head_dim_(head_dim),
          dtype_(dtype), is_initialized_(false) {
        
        // Set default block sizes
        block_size_m_ = 64;
        block_size_n_ = 64;
        
        // Calculate scale factor
        scale_ = 1.0f / sqrtf(static_cast<float>(head_dim));
        
        // Validate parameters
        validate_parameters();
        
        // Allocate device memory
        allocate_device_memory();
        
        is_initialized_ = true;
    }
    
    ~FlashAttentionV1() {
        if (is_initialized_) {
            free_device_memory();
        }
    }
    
    // Forward pass
    void forward(const void* q, const void* k, const void* v, void* o) {
        if (!is_initialized_) {
            throw std::runtime_error("FlashAttentionV1 not initialized");
        }
        
        // Copy input data to device
        copy_input_to_device(q, k, v);
        
        // Launch kernel
        launch_forward_kernel();
        
        // Copy output data from device
        copy_output_from_device(o);
    }
    
    // Set block sizes
    void set_block_sizes(int block_size_m, int block_size_n) {
        block_size_m_ = block_size_m;
        block_size_n_ = block_size_n;
        
        // Re-validate parameters
        validate_parameters();
    }
    
    // Get memory usage estimate
    size_t get_memory_usage_bytes() const {
        size_t input_size = batch_size_ * seq_len_ * head_dim_ * get_data_type_size(dtype_);
        size_t output_size = batch_size_ * seq_len_ * head_dim_ * get_data_type_size(dtype_);
        size_t lse_size = batch_size_ * seq_len_ * sizeof(float);
        size_t shared_memory = block_size_m_ * block_size_n_ * sizeof(float) * 3; // Q, K, V blocks
        
        return input_size * 3 + output_size + lse_size + shared_memory;
    }
    
    // Get theoretical operations count
    size_t get_theoretical_flops() const {
        // Attention: Q*K^T (M*N*K) + softmax + (softmax)*V (M*N*K)
        size_t qk_ops = static_cast<size_t>(batch_size_) * seq_len_ * seq_len_ * head_dim_;
        size_t softmax_ops = static_cast<size_t>(batch_size_) * seq_len_ * seq_len_;
        size_t pv_ops = static_cast<size_t>(batch_size_) * seq_len_ * seq_len_ * head_dim_;
        
        return 2 * (qk_ops + pv_ops) + softmax_ops;
    }
    
private:
    // Parameters
    int batch_size_;
    int seq_len_;
    int head_dim_;
    DataType dtype_;
    float scale_;
    
    // Block sizes
    int block_size_m_;
    int block_size_n_;
    
    // Device memory
    void* d_q_;
    void* d_k_;
    void* d_v_;
    void* d_o_;
    float* d_lse_;
    
    // State
    bool is_initialized_;
    
    void validate_parameters() {
        if (batch_size_ <= 0 || seq_len_ <= 0 || head_dim_ <= 0) {
            throw std::invalid_argument("Batch size, sequence length, and head dimension must be positive");
        }
        
        if (!is_power_of_two(block_size_m_) || !is_power_of_two(block_size_n_)) {
            throw std::invalid_argument("Block sizes must be powers of 2");
        }
        
        if (block_size_m_ > MAX_BLOCK_SIZE || block_size_n_ > MAX_BLOCK_SIZE) {
            throw std::invalid_argument("Block sizes too large");
        }
        
        // Check shared memory requirements
        size_t shared_mem_needed = block_size_m_ * block_size_n_ * sizeof(float) * 3;
        int max_shared_mem = get_max_shared_memory_per_block();
        
        if (shared_mem_needed > max_shared_mem) {
            throw std::invalid_argument("Block sizes require too much shared memory");
        }
    }
    
    void allocate_device_memory() {
        size_t elem_size = get_data_type_size(dtype_);
        size_t total_size = batch_size_ * seq_len_ * head_dim_ * elem_size;
        
        CUDA_CHECK(cudaMalloc(&d_q_, total_size));
        CUDA_CHECK(cudaMalloc(&d_k_, total_size));
        CUDA_CHECK(cudaMalloc(&d_v_, total_size));
        CUDA_CHECK(cudaMalloc(&d_o_, total_size));
        CUDA_CHECK(cudaMalloc(&d_lse_, batch_size_ * seq_len_ * sizeof(float)));
    }
    
    void free_device_memory() {
        CUDA_CHECK(cudaFree(d_q_));
        CUDA_CHECK(cudaFree(d_k_));
        CUDA_CHECK(cudaFree(d_v_));
        CUDA_CHECK(cudaFree(d_o_));
        CUDA_CHECK(cudaFree(d_lse_));
    }
    
    void copy_input_to_device(const void* q, const void* k, const void* v) {
        size_t total_size = batch_size_ * seq_len_ * head_dim_ * get_data_type_size(dtype_);
        
        CUDA_CHECK(cudaMemcpy(d_q_, q, total_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_k_, k, total_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_v_, v, total_size, cudaMemcpyHostToDevice));
    }
    
    void copy_output_from_device(void* o) {
        size_t total_size = batch_size_ * seq_len_ * head_dim_ * get_data_type_size(dtype_);
        CUDA_CHECK(cudaMemcpy(o, d_o_, total_size, cudaMemcpyDeviceToHost));
    }
    
    void launch_forward_kernel() {
        // Calculate grid dimensions
        dim3 block(block_size_m_, 1, 1);
        dim3 grid(ceil_div(seq_len_, block_size_m_), batch_size_, 1);
        
        // Launch appropriate kernel based on data type
        switch (dtype_) {
            case DataType::FLOAT32:
                flash_attention_v1_forward_kernel<<<grid, block>>>(
                    static_cast<const float*>(d_q_),
                    static_cast<const float*>(d_k_),
                    static_cast<const float*>(d_v_),
                    static_cast<float*>(d_o_),
                    batch_size_, seq_len_, head_dim_,
                    block_size_m_, block_size_n_,
                    scale_, d_lse_
                );
                break;
                
            case DataType::FLOAT16:
                flash_attention_v1_forward_kernel_half<<<grid, block>>>(
                    static_cast<const __half*>(d_q_),
                    static_cast<const __half*>(d_k_),
                    static_cast<const __half*>(d_v_),
                    static_cast<__half*>(d_o_),
                    batch_size_, seq_len_, head_dim_,
                    block_size_m_, block_size_n_,
                    scale_, d_lse_
                );
                break;
                
            default:
                throw std::runtime_error("Unsupported data type");
        }
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
};

// C interface for external use
extern "C" {
    
    FlashAttentionV1* create_flash_attention_v1(int batch_size, int seq_len, int head_dim, int dtype) {
        try {
            DataType data_type = static_cast<DataType>(dtype);
            return new FlashAttentionV1(batch_size, seq_len, head_dim, data_type);
        } catch (const std::exception& e) {
            std::cerr << "Error creating FlashAttentionV1: " << e.what() << std::endl;
            return nullptr;
        }
    }
    
    void destroy_flash_attention_v1(FlashAttentionV1* fa) {
        if (fa) {
            delete fa;
        }
    }
    
    int flash_attention_v1_forward(FlashAttentionV1* fa, const void* q, const void* k, const void* v, void* o) {
        if (!fa) {
            return -1;
        }
        
        try {
            fa->forward(q, k, v, o);
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Error in forward pass: " << e.what() << std::endl;
            return -1;
        }
    }
    
    int flash_attention_v1_set_block_sizes(FlashAttentionV1* fa, int block_size_m, int block_size_n) {
        if (!fa) {
            return -1;
        }
        
        try {
            fa->set_block_sizes(block_size_m, block_size_n);
            return 0;
        } catch (const std::exception& e) {
            std::cerr << "Error setting block sizes: " << e.what() << std::endl;
            return -1;
        }
    }
    
    size_t flash_attention_v1_get_memory_usage(FlashAttentionV1* fa) {
        if (!fa) {
            return 0;
        }
        return fa->get_memory_usage_bytes();
    }
    
    size_t flash_attention_v1_get_theoretical_flops(FlashAttentionV1* fa) {
        if (!fa) {
            return 0;
        }
        return fa->get_theoretical_flops();
    }
}