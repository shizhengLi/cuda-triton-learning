#include "flash_attention_v2.h"
#include "../cuda_common/flash_attention_common.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <chrono>

// Reference CPU implementation for Flash Attention v2
void naive_attention_v2_cpu(const half* q, const half* k, const half* v, half* o,
                           int batch_size, int seq_len, int num_heads, int head_dim,
                           bool use_causal_mask = false) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                for (int d = 0; d < head_dim; ++d) {
                    float max_score = -INFINITY;
                    float sum_exp = 0.0f;
                    float sum_v = 0.0f;
                    
                    // Compute attention scores
                    for (int j = 0; j < seq_len; ++j) {
                        // Apply causal mask
                        if (use_causal_mask && j > i) {
                            continue;
                        }
                        
                        // Compute Q*K^T
                        float score = 0.0f;
                        for (int d2 = 0; d2 < head_dim; ++d2) {
                            float q_val = __half2float(q[b * num_heads * seq_len * head_dim + 
                                                         h * seq_len * head_dim + 
                                                         i * head_dim + d2]);
                            float k_val = __half2float(k[b * num_heads * seq_len * head_dim + 
                                                         h * seq_len * head_dim + 
                                                         j * head_dim + d2]);
                            score += q_val * k_val;
                        }
                        score *= scale;
                        
                        max_score = fmaxf(max_score, score);
                    }
                    
                    // Compute softmax and weighted sum
                    for (int j = 0; j < seq_len; ++j) {
                        // Apply causal mask
                        if (use_causal_mask && j > i) {
                            continue;
                        }
                        
                        // Compute Q*K^T
                        float score = 0.0f;
                        for (int d2 = 0; d2 < head_dim; ++d2) {
                            float q_val = __half2float(q[b * num_heads * seq_len * head_dim + 
                                                         h * seq_len * head_dim + 
                                                         i * head_dim + d2]);
                            float k_val = __half2float(k[b * num_heads * seq_len * head_dim + 
                                                         h * seq_len * head_dim + 
                                                         j * head_dim + d2]);
                            score += q_val * k_val;
                        }
                        score *= scale;
                        
                        float exp_score = expf(score - max_score);
                        sum_exp += exp_score;
                        
                        float v_val = __half2float(v[b * num_heads * seq_len * head_dim + 
                                                     h * seq_len * head_dim + 
                                                     j * head_dim + d]);
                        sum_v += exp_score * v_val;
                    }
                    
                    // Store result
                    o[b * num_heads * seq_len * head_dim + 
                      h * seq_len * head_dim + 
                      i * head_dim + d] = __float2half(sum_v / sum_exp);
                }
            }
        }
    }
}

// Generate random half precision data
void generate_random_half_data(half* data, size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = __float2half(dis(gen));
    }
}

// Compare half precision arrays
bool compare_half_arrays(const half* a, const half* b, size_t size, float tolerance = 1e-2f) {
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    
    for (size_t i = 0; i < size; ++i) {
        float diff = fabsf(__half2float(a[i]) - __half2float(b[i]));
        max_diff = fmaxf(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= size;
    
    printf("Max difference: %.6f, Average difference: %.6f\n", max_diff, avg_diff);
    
    return max_diff < tolerance;
}

// Test configurations for Flash Attention v2
struct TestConfigV2 {
    int batch_size;
    int seq_len;
    int num_heads;
    int head_dim;
    bool use_causal_mask;
    const char* name;
};

TestConfigV2 test_configs_v2[] = {
    {1, 32, 4, 16, false, "tiny_standard"},
    {1, 32, 4, 16, true, "tiny_causal"},
    {1, 64, 8, 32, false, "small_standard"},
    {1, 64, 8, 32, true, "small_causal"},
    {2, 128, 8, 32, false, "medium_standard"},
    {2, 128, 8, 32, true, "medium_causal"},
};

// Test Flash Attention v2 implementation
int main() {
    printf("Flash Attention v2 CUDA Test Suite\n");
    printf("===================================\n\n");
    
    // Print device info
    print_device_info();
    
    // Initialize Flash Attention v2
    cudaError_t init_err = flash_attention_v2_init();
    if (init_err != cudaSuccess) {
        printf("Failed to initialize Flash Attention v2: %s\n", cudaGetErrorString(init_err));
        return 1;
    }
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Test each configuration
    for (const auto& config : test_configs_v2) {
        printf("Testing configuration: %s (%dx%dx%dx%d, causal=%s)\n", 
               config.name, config.batch_size, config.seq_len, config.num_heads, config.head_dim,
               config.use_causal_mask ? "true" : "false");
        printf("----------------------------------------------------------------\n");
        
        total_tests++;
        
        try {
            // Create configuration
            FlashAttentionV2Config fa_config;
            fa_config.batch_size = config.batch_size;
            fa_config.seq_len = config.seq_len;
            fa_config.num_heads = config.num_heads;
            fa_config.head_dim = config.head_dim;
            fa_config.use_causal_mask = config.use_causal_mask;
            fa_config.dropout_rate = 0.0f;
            fa_config.seed = 42;
            
            // Allocate host memory
            size_t total_size = config.batch_size * config.seq_len * config.num_heads * config.head_dim;
            std::vector<half> h_q(total_size);
            std::vector<half> h_k(total_size);
            std::vector<half> h_v(total_size);
            std::vector<half> h_o_cpu(total_size);
            std::vector<half> h_o_cuda(total_size);
            std::vector<half> h_l(config.batch_size * config.num_heads * config.seq_len);
            
            // Generate random data
            generate_random_half_data(h_q.data(), total_size);
            generate_random_half_data(h_k.data(), total_size);
            generate_random_half_data(h_v.data(), total_size);
            
            // Compute CPU reference
            auto cpu_start = std::chrono::high_resolution_clock::now();
            naive_attention_v2_cpu(h_q.data(), h_k.data(), h_v.data(), h_o_cpu.data(),
                                   config.batch_size, config.seq_len, config.num_heads, config.head_dim,
                                   config.use_causal_mask);
            auto cpu_end = std::chrono::high_resolution_clock::now();
            auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
            
            printf("CPU reference computed in %ld ms\n", cpu_time);
            
            // Allocate device memory
            half *d_q, *d_k, *d_v, *d_o, *d_l;
            CUDA_CHECK(cudaMalloc(&d_q, total_size * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_k, total_size * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_v, total_size * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_o, total_size * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_l, config.batch_size * config.num_heads * config.seq_len * sizeof(half)));
            
            // Copy data to device
            CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), total_size * sizeof(half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), total_size * sizeof(half), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), total_size * sizeof(half), cudaMemcpyHostToDevice));
            
            // Launch Flash Attention v2 forward
            print_memory_usage("Before Flash Attention v2 kernel");
            
            auto cuda_start = std::chrono::high_resolution_clock::now();
            cudaError_t forward_err = flash_attention_v2_forward(
                d_q, d_k, d_v, d_o, d_l, fa_config
            );
            auto cuda_end = std::chrono::high_resolution_clock::now();
            auto cuda_time = std::chrono::duration_cast<std::chrono::milliseconds>(cuda_end - cuda_start).count();
            
            if (forward_err != cudaSuccess) {
                printf("✗ Test FAILED - Forward kernel error: %s\n", cudaGetErrorString(forward_err));
                continue;
            }
            
            print_memory_usage("After Flash Attention v2 kernel");
            
            printf("Flash Attention v2 computed in %ld ms\n", cuda_time);
            if (cpu_time > 0) {
                printf("Speedup: %.2fx\n", static_cast<double>(cpu_time) / cuda_time);
            }
            
            // Copy results back
            CUDA_CHECK(cudaMemcpy(h_o_cuda.data(), d_o, total_size * sizeof(half), cudaMemcpyDeviceToHost));
            
            // Compare results
            printf("Comparing results...\n");
            bool results_match = compare_half_arrays(h_o_cpu.data(), h_o_cuda.data(), total_size);
            
            if (results_match) {
                printf("✓ Test PASSED - Results match within tolerance\n");
                passed_tests++;
            } else {
                printf("✗ Test FAILED - Results differ\n");
            }
            
            // Test backward pass (optional)
            printf("Testing backward pass...\n");
            half *d_do, *d_dq, *d_dk, *d_dv;
            CUDA_CHECK(cudaMalloc(&d_do, total_size * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_dq, total_size * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_dk, total_size * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&d_dv, total_size * sizeof(half)));
            
            // Initialize output gradients
            std::vector<half> h_do(total_size);
            generate_random_half_data(h_do.data(), total_size);
            CUDA_CHECK(cudaMemcpy(d_do, h_do.data(), total_size * sizeof(half), cudaMemcpyHostToDevice));
            
            cudaError_t backward_err = flash_attention_v2_backward(
                d_q, d_k, d_v, d_o, d_l, d_do, d_dq, d_dk, d_dv, fa_config
            );
            
            if (backward_err != cudaSuccess) {
                printf("⚠ Backward pass failed: %s\n", cudaGetErrorString(backward_err));
            } else {
                printf("✓ Backward pass completed successfully\n");
            }
            
            // Free device memory
            CUDA_CHECK(cudaFree(d_q));
            CUDA_CHECK(cudaFree(d_k));
            CUDA_CHECK(cudaFree(d_v));
            CUDA_CHECK(cudaFree(d_o));
            CUDA_CHECK(cudaFree(d_l));
            CUDA_CHECK(cudaFree(d_do));
            CUDA_CHECK(cudaFree(d_dq));
            CUDA_CHECK(cudaFree(d_dk));
            CUDA_CHECK(cudaFree(d_dv));
            
        } catch (const std::exception& e) {
            printf("✗ Test FAILED - Exception: %s\n", e.what());
        }
        
        printf("\n");
    }
    
    // Cleanup
    cudaError_t cleanup_err = flash_attention_v2_cleanup();
    if (cleanup_err != cudaSuccess) {
        printf("Warning: Cleanup failed: %s\n", cudaGetErrorString(cleanup_err));
    }
    
    // Summary
    printf("Test Summary\n");
    printf("============\n");
    printf("Total tests: %d\n", total_tests);
    printf("Passed tests: %d\n", passed_tests);
    printf("Success rate: %.1f%%\n", 100.0f * passed_tests / total_tests);
    
    if (passed_tests == total_tests) {
        printf("🎉 All tests PASSED!\n");
        return 0;
    } else {
        printf("❌ Some tests FAILED!\n");
        return 1;
    }
}