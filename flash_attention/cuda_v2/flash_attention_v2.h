#ifndef FLASH_ATTENTION_V2_H
#define FLASH_ATTENTION_V2_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Flash Attention v2 configuration
struct FlashAttentionV2Config {
    int batch_size;
    int seq_len;
    int num_heads;
    int head_dim;
    bool use_causal_mask;
    float dropout_rate;
    unsigned long long seed;
};

// Flash Attention v2 forward declaration
cudaError_t flash_attention_v2_forward(
    const half* q,           // Query [batch, seq_len, num_heads, head_dim]
    const half* k,           // Key [batch, seq_len, num_heads, head_dim]
    const half* v,           // Value [batch, seq_len, num_heads, head_dim]
    half* o,                 // Output [batch, seq_len, num_heads, head_dim]
    half* l,                 // Logsumexp [batch, num_heads, seq_len]
    const FlashAttentionV2Config& config,
    cudaStream_t stream = nullptr
);

// Flash Attention v2 backward declaration
cudaError_t flash_attention_v2_backward(
    const half* q,           // Query [batch, seq_len, num_heads, head_dim]
    const half* k,           // Key [batch, seq_len, num_heads, head_dim]
    const half* v,           // Value [batch, seq_len, num_heads, head_dim]
    const half* o,           // Output [batch, seq_len, num_heads, head_dim]
    const half* l,           // Logsumexp [batch, num_heads, seq_len]
    const half* do_grad,     // Output gradient [batch, seq_len, num_heads, head_dim]
    half* dq,                // Query gradient [batch, seq_len, num_heads, head_dim]
    half* dk,                // Key gradient [batch, seq_len, num_heads, head_dim]
    half* dv,                // Value gradient [batch, seq_len, num_heads, head_dim]
    const FlashAttentionV2Config& config,
    cudaStream_t stream = nullptr
);

// Utility functions
cudaError_t flash_attention_v2_init();
cudaError_t flash_attention_v2_cleanup();

// Configuration validation
bool is_config_valid(const FlashAttentionV2Config& config);

#endif // FLASH_ATTENTION_V2_H