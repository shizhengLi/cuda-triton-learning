#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include "flash_attention_v2.h"

// Test configuration validation
bool test_config_validation() {
    std::cout << "Testing configuration validation..." << std::endl;
    
    FlashAttentionV2Config config;
    config.batch_size = 1;
    config.seq_len = 32;
    config.num_heads = 4;
    config.head_dim = 16;
    config.use_causal_mask = false;
    config.dropout_rate = 0.0f;
    config.seed = 42;
    
    bool valid = is_config_valid(config);
    std::cout << "Valid configuration: " << (valid ? "PASSED" : "FAILED") << std::endl;
    
    // Test invalid configuration
    config.batch_size = -1;
    bool invalid = is_config_valid(config);
    std::cout << "Invalid configuration detection: " << (!invalid ? "PASSED" : "FAILED") << std::endl;
    
    return valid && !invalid;
}

// Test memory allocation and basic kernel launch
bool test_basic_functionality() {
    std::cout << "Testing basic functionality..." << std::endl;
    
    FlashAttentionV2Config config;
    config.batch_size = 1;
    config.seq_len = 32;
    config.num_heads = 4;
    config.head_dim = 16;
    config.use_causal_mask = false;
    config.dropout_rate = 0.0f;
    config.seed = 42;
    
    size_t total_size = config.batch_size * config.seq_len * config.num_heads * config.head_dim;
    
    // Allocate device memory
    half *d_q, *d_k, *d_v, *d_o, *d_l;
    cudaError_t err;
    
    err = cudaMalloc(&d_q, total_size * sizeof(half));
    if (err != cudaSuccess) return false;
    
    err = cudaMalloc(&d_k, total_size * sizeof(half));
    if (err != cudaSuccess) {
        cudaFree(d_q);
        return false;
    }
    
    err = cudaMalloc(&d_v, total_size * sizeof(half));
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        return false;
    }
    
    err = cudaMalloc(&d_o, total_size * sizeof(half));
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        return false;
    }
    
    err = cudaMalloc(&d_l, config.batch_size * config.num_heads * config.seq_len * sizeof(half));
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        return false;
    }
    
    // Test forward pass
    err = flash_attention_v2_forward(d_q, d_k, d_v, d_o, d_l, config);
    std::cout << "Forward pass: " << (err == cudaSuccess ? "PASSED" : "FAILED") << std::endl;
    
    // Test backward pass
    half *d_do, *d_dq, *d_dk, *d_dv;
    err = cudaMalloc(&d_do, total_size * sizeof(half));
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        cudaFree(d_l);
        return false;
    }
    
    err = cudaMalloc(&d_dq, total_size * sizeof(half));
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        cudaFree(d_l);
        cudaFree(d_do);
        return false;
    }
    
    err = cudaMalloc(&d_dk, total_size * sizeof(half));
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        cudaFree(d_l);
        cudaFree(d_do);
        cudaFree(d_dq);
        return false;
    }
    
    err = cudaMalloc(&d_dv, total_size * sizeof(half));
    if (err != cudaSuccess) {
        cudaFree(d_q);
        cudaFree(d_k);
        cudaFree(d_v);
        cudaFree(d_o);
        cudaFree(d_l);
        cudaFree(d_do);
        cudaFree(d_dq);
        cudaFree(d_dk);
        return false;
    }
    
    err = flash_attention_v2_backward(d_q, d_k, d_v, d_o, d_l, d_do, d_dq, d_dk, d_dv, config);
    std::cout << "Backward pass: " << (err == cudaSuccess ? "PASSED" : "FAILED") << std::endl;
    
    // Cleanup
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_l);
    cudaFree(d_do);
    cudaFree(d_dq);
    cudaFree(d_dk);
    cudaFree(d_dv);
    
    return err == cudaSuccess;
}

int main() {
    std::cout << "Flash Attention v2 CUDA Test - Comprehensive Version" << std::endl;
    std::cout << "====================================================" << std::endl;
    
    // Test basic CUDA functionality
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
    
    // Test initialization
    cudaError_t init_err = flash_attention_v2_init();
    std::cout << "Initialization: " << (init_err == cudaSuccess ? "PASSED" : "FAILED") << std::endl;
    
    // Run tests
    bool config_test = test_config_validation();
    bool functionality_test = test_basic_functionality();
    
    // Test cleanup
    cudaError_t cleanup_err = flash_attention_v2_cleanup();
    std::cout << "Cleanup: " << (cleanup_err == cudaSuccess ? "PASSED" : "FAILED") << std::endl;
    
    // Summary
    std::cout << "\nTest Summary:" << std::endl;
    std::cout << "=============" << std::endl;
    std::cout << "Configuration validation: " << (config_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Basic functionality: " << (functionality_test ? "PASSED" : "FAILED") << std::endl;
    std::cout << "Overall: " << (config_test && functionality_test ? "PASSED" : "FAILED") << std::endl;
    
    if (config_test && functionality_test) {
        std::cout << "🎉 All tests PASSED!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ Some tests FAILED!" << std::endl;
        return 1;
    }
}