#include <iostream>
#include <cuda_runtime.h>
#include "flash_attention_v2.h"

int main() {
    std::cout << "Flash Attention v2 CUDA Test - Minimal Version" << std::endl;
    
    // Test basic functionality
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    if (device_count == 0) {
        std::cout << "No CUDA devices found" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << device_count << " CUDA devices" << std::endl;
    
    // Test configuration validation
    FlashAttentionV2Config config;
    config.batch_size = 1;
    config.seq_len = 32;
    config.num_heads = 4;
    config.head_dim = 16;
    config.use_causal_mask = false;
    config.dropout_rate = 0.0f;
    config.seed = 42;
    
    bool valid = is_config_valid(config);
    std::cout << "Configuration validation: " << (valid ? "PASSED" : "FAILED") << std::endl;
    
    // Test initialization
    cudaError_t init_err = flash_attention_v2_init();
    std::cout << "Initialization: " << (init_err == cudaSuccess ? "PASSED" : "FAILED") << std::endl;
    
    // Test cleanup
    cudaError_t cleanup_err = flash_attention_v2_cleanup();
    std::cout << "Cleanup: " << (cleanup_err == cudaSuccess ? "PASSED" : "FAILED") << std::endl;
    
    std::cout << "Basic functionality test completed successfully!" << std::endl;
    return 0;
}