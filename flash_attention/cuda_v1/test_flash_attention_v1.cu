#include "../cuda_common/flash_attention_common.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <cassert>
#include <chrono>

// Simple CPU implementation for testing
void naive_attention_cpu(const float* q, const float* k, const float* v, float* o,
                        int batch_size, int seq_len, int head_dim, float scale) {
    for (int b = 0; b < batch_size; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            float max_score = -INFINITY;
            float sum_exp = 0.0f;
            
            // First pass: find max score for numerical stability
            for (int j = 0; j < seq_len; ++j) {
                float score = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    score += q[b * seq_len * head_dim + i * head_dim + d] * 
                            k[b * seq_len * head_dim + j * head_dim + d];
                }
                score *= scale;
                max_score = fmaxf(max_score, score);
            }
            
            // Second pass: compute softmax and weighted sum for each dimension
            for (int d = 0; d < head_dim; ++d) {
                float sum = 0.0f;
                sum_exp = 0.0f;
                
                for (int j = 0; j < seq_len; ++j) {
                    float score = 0.0f;
                    for (int k_dim = 0; k_dim < head_dim; ++k_dim) {
                        score += q[b * seq_len * head_dim + i * head_dim + k_dim] * 
                                k[b * seq_len * head_dim + j * head_dim + k_dim];
                    }
                    score *= scale;
                    
                    float exp_score = expf(score - max_score);
                    sum_exp += exp_score;
                    sum += exp_score * v[b * seq_len * head_dim + j * head_dim + d];
                }
                
                // Normalize
                o[b * seq_len * head_dim + i * head_dim + d] = sum / sum_exp;
            }
        }
    }
}

// Generate random data
void generate_random_data(float* data, size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }
}

// Compare two arrays
bool compare_arrays(const float* a, const float* b, size_t size, float tolerance = 1e-3f) {
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    
    for (size_t i = 0; i < size; ++i) {
        float diff = fabsf(a[i] - b[i]);
        max_diff = fmaxf(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= size;
    
    printf("Max difference: %.6f, Average difference: %.6f\n", max_diff, avg_diff);
    
    return max_diff < tolerance;
}

// Simple CUDA kernel for testing
__global__ void simple_attention_kernel(
    const float* q, const float* k, const float* v, float* o,
    int batch_size, int seq_len, int head_dim, float scale
) {
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (seq_idx >= seq_len) return;
    
    const float* q_ptr = q + batch_idx * seq_len * head_dim + seq_idx * head_dim;
    const float* k_ptr = k + batch_idx * seq_len * head_dim;
    const float* v_ptr = v + batch_idx * seq_len * head_dim;
    float* o_ptr = o + batch_idx * seq_len * head_dim + seq_idx * head_dim;
    
    // Compute attention scores
    float scores[1024]; // Maximum sequence length for this test
    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    
    // Compute Q*K^T
    for (int j = 0; j < seq_len; ++j) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            score += q_ptr[d] * k_ptr[j * head_dim + d];
        }
        score *= scale;
        scores[j] = score;
        max_score = fmaxf(max_score, score);
    }
    
    // Compute softmax normalization factor
    sum_exp = 0.0f;
    for (int j = 0; j < seq_len; ++j) {
        float exp_score = expf(scores[j] - max_score);
        sum_exp += exp_score;
    }
    
    // Compute weighted sum for each dimension
    for (int d = 0; d < head_dim; ++d) {
        float sum = 0.0f;
        
        for (int j = 0; j < seq_len; ++j) {
            float exp_score = expf(scores[j] - max_score);
            sum += exp_score * v_ptr[j * head_dim + d];
        }
        
        o_ptr[d] = sum / sum_exp;
    }
}

// Test different configurations
struct TestConfig {
    int batch_size;
    int seq_len;
    int head_dim;
    const char* name;
};

TestConfig test_configs[] = {
    {1, 32, 16, "tiny"},
    {1, 64, 32, "small"},
    {2, 128, 32, "medium"},
};

int main() {
    printf("Flash Attention v1 CUDA Test Suite (Simple Version)\n");
    printf("==================================================\n\n");
    
    // Print device info
    print_device_info();
    
    int total_tests = 0;
    int passed_tests = 0;
    
    // Test each configuration
    for (const auto& config : test_configs) {
        printf("Testing configuration: %s (%dx%dx%d)\n", 
               config.name, config.batch_size, config.seq_len, config.head_dim);
        printf("------------------------------------------------\n");
        
        total_tests++;
        
        try {
            // Allocate host memory
            size_t total_size = config.batch_size * config.seq_len * config.head_dim;
            std::vector<float> h_q(total_size);
            std::vector<float> h_k(total_size);
            std::vector<float> h_v(total_size);
            std::vector<float> h_o_cpu(total_size);
            std::vector<float> h_o_cuda(total_size);
            
            // Generate random data
            generate_random_data(h_q.data(), total_size);
            generate_random_data(h_k.data(), total_size);
            generate_random_data(h_v.data(), total_size);
            
            // Compute CPU reference
            float scale = 1.0f / sqrtf(static_cast<float>(config.head_dim));
            
            auto cpu_start = std::chrono::high_resolution_clock::now();
            naive_attention_cpu(h_q.data(), h_k.data(), h_v.data(), h_o_cpu.data(),
                               config.batch_size, config.seq_len, config.head_dim, scale);
            auto cpu_end = std::chrono::high_resolution_clock::now();
            auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
            
            printf("CPU reference computed in %ld ms\n", cpu_time);
            
            // Allocate device memory
            float *d_q, *d_k, *d_v, *d_o;
            CUDA_CHECK(cudaMalloc(&d_q, total_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_k, total_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_v, total_size * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_o, total_size * sizeof(float)));
            
            // Copy data to device
            CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), total_size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), total_size * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), total_size * sizeof(float), cudaMemcpyHostToDevice));
            
            // Launch kernel
            dim3 block(256, 1, 1);
            dim3 grid(ceil_div(config.seq_len, block.x), config.batch_size, 1);
            
            print_memory_usage("Before CUDA kernel");
            
            auto cuda_start = std::chrono::high_resolution_clock::now();
            simple_attention_kernel<<<grid, block>>>(d_q, d_k, d_v, d_o,
                                                      config.batch_size, config.seq_len, config.head_dim, scale);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            auto cuda_end = std::chrono::high_resolution_clock::now();
            auto cuda_time = std::chrono::duration_cast<std::chrono::milliseconds>(cuda_end - cuda_start).count();
            
            print_memory_usage("After CUDA kernel");
            
            printf("CUDA kernel computed in %ld ms\n", cuda_time);
            printf("Speedup: %.2fx\n", static_cast<double>(cpu_time) / cuda_time);
            
            // Copy results back
            CUDA_CHECK(cudaMemcpy(h_o_cuda.data(), d_o, total_size * sizeof(float), cudaMemcpyDeviceToHost));
            
            // Compare results
            printf("Comparing results...\n");
            bool results_match = compare_arrays(h_o_cpu.data(), h_o_cuda.data(), total_size);
            
            if (results_match) {
                printf("✓ Test PASSED - Results match within tolerance\n");
                passed_tests++;
            } else {
                printf("✗ Test FAILED - Results differ\n");
            }
            
            // Free device memory
            CUDA_CHECK(cudaFree(d_q));
            CUDA_CHECK(cudaFree(d_k));
            CUDA_CHECK(cudaFree(d_v));
            CUDA_CHECK(cudaFree(d_o));
            
        } catch (const std::exception& e) {
            printf("✗ Test FAILED - Exception: %s\n", e.what());
        }
        
        printf("\n");
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