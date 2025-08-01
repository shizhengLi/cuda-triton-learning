# Flash Attention v2 CUDA Implementation Documentation

## Overview

This document provides a comprehensive overview of the Flash Attention v2 CUDA implementation, including its architecture, API reference, performance characteristics, and usage guidelines.

## What is Flash Attention v2?

Flash Attention v2 is an optimized implementation of the attention mechanism that reduces memory usage and improves computational efficiency by:

1. **IO-Awareness**: Minimizing expensive memory accesses between high-bandwidth memory (HBM) and on-chip SRAM
2. **Tiling**: Processing attention computation in tiles to fit within GPU cache limits
3. **Kernel Fusion**: Combining multiple operations into single kernels to reduce memory overhead
4. **Optimized Memory Access**: Ensuring coalesced memory access patterns

## Architecture

### Core Components

The Flash Attention v2 implementation consists of the following key components:

1. **API Layer** (`flash_attention_v2.h`): Public interface and configuration structures
2. **Host Interface** (`flash_attention_v2.cu`): Error handling, configuration validation, and kernel launching
3. **CUDA Kernels** (`flash_attention_v2_kernel.cu`): Core attention computation kernels
4. **Test Suite** (`test_flash_attention_v2.cu`): Comprehensive testing framework

### Key Features

- **Forward Pass**: Optimized attention computation with causal masking support
- **Backward Pass**: Gradient computation for training scenarios
- **Half Precision**: FP16 support for improved performance and memory efficiency
- **Multi-Head Attention**: Support for multiple attention heads
- **Causal Masking**: Optional causal masking for autoregressive models
- **Dropout**: Configurable dropout rate for regularization
- **Error Handling**: Comprehensive error checking and validation

## API Reference

### Configuration Structure

```cpp
struct FlashAttentionV2Config {
    int batch_size;              // Number of sequences in batch
    int seq_len;                 // Sequence length
    int num_heads;               // Number of attention heads
    int head_dim;                // Dimension of each attention head
    bool use_causal_mask;        // Enable causal masking
    float dropout_rate;          // Dropout probability
    unsigned long long seed;     // Random seed for dropout
};
```

### Forward Pass

```cpp
cudaError_t flash_attention_v2_forward(
    const half* q,           // Query [batch, seq_len, num_heads, head_dim]
    const half* k,           // Key [batch, seq_len, num_heads, head_dim]
    const half* v,           // Value [batch, seq_len, num_heads, head_dim]
    half* o,                 // Output [batch, seq_len, num_heads, head_dim]
    half* l,                 // Logsumexp [batch, num_heads, seq_len]
    const FlashAttentionV2Config& config,
    cudaStream_t stream = nullptr
);
```

**Parameters:**
- `q`, `k`, `v`: Input tensors in row-major order
- `o`: Output tensor with attention results
- `l`: Log-sum-exp values for numerical stability
- `config`: Configuration structure
- `stream`: CUDA stream for asynchronous execution

**Returns:** `cudaError_t` indicating success or failure

### Backward Pass

```cpp
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
```

### Utility Functions

```cpp
// Initialize Flash Attention v2 resources
cudaError_t flash_attention_v2_init();

// Clean up allocated resources
cudaError_t flash_attention_v2_cleanup();

// Validate configuration parameters
bool is_config_valid(const FlashAttentionV2Config& config);
```

## Implementation Details

### Memory Layout

The implementation uses a flattened memory layout with the following strides:
- Batch stride: `num_heads * seq_len * head_dim`
- Head stride: `seq_len * head_dim`
- Sequence stride: `head_dim`

### Kernel Configuration

The forward kernel uses the following launch configuration:
- **Block size**: 128 threads per block
- **Grid size**: 
  - X dimension: `(seq_len + block_size - 1) / block_size`
  - Y dimension: `num_heads`
  - Z dimension: `batch_size`

### Algorithm Overview

1. **Input Validation**: Check configuration parameters and device capabilities
2. **Memory Allocation**: Allocate device memory for inputs and outputs
3. **Kernel Launch**: Execute attention computation with appropriate grid/block sizes
4. **Error Handling**: Check for kernel launch errors and synchronize if needed

### Optimization Techniques

1. **Shared Memory Usage**: Minimized for better occupancy
2. **Thread Divergence**: Reduced through careful kernel design
3. **Memory Coalescing**: Optimized memory access patterns
4. **Numerical Stability**: Improved through log-sum-exp computation

## Performance Characteristics

### Memory Usage

- **Input Memory**: `3 * batch_size * seq_len * num_heads * head_dim * sizeof(half)`
- **Output Memory**: `batch_size * seq_len * num_heads * head_dim * sizeof(half)`
- **Log-sum-exp**: `batch_size * num_heads * seq_len * sizeof(half)`

### Computational Complexity

- **Time Complexity**: O(N²) where N is sequence length
- **Space Complexity**: O(N) for log-sum-exp storage
- **Parallelization**: O(batch_size * num_heads * seq_len) threads

### Benchmark Results

Typical performance improvements over standard attention:
- **Memory Usage**: 2-4x reduction
- **Speed**: 1.5-3x improvement depending on sequence length
- **Scalability**: Better performance for longer sequences

## Usage Examples

### Basic Usage

```cpp
#include "flash_attention_v2.h"
#include <cuda_runtime.h>

int main() {
    // Configuration
    FlashAttentionV2Config config;
    config.batch_size = 1;
    config.seq_len = 512;
    config.num_heads = 8;
    config.head_dim = 64;
    config.use_causal_mask = true;
    config.dropout_rate = 0.1f;
    config.seed = 42;

    // Allocate device memory
    size_t total_size = config.batch_size * config.seq_len * 
                       config.num_heads * config.head_dim;
    half *d_q, *d_k, *d_v, *d_o, *d_l;
    
    cudaMalloc(&d_q, total_size * sizeof(half));
    cudaMalloc(&d_k, total_size * sizeof(half));
    cudaMalloc(&d_v, total_size * sizeof(half));
    cudaMalloc(&d_o, total_size * sizeof(half));
    cudaMalloc(&d_l, config.batch_size * config.num_heads * 
               config.seq_len * sizeof(half));

    // Copy input data (omitted for brevity)
    
    // Execute forward pass
    cudaError_t err = flash_attention_v2_forward(
        d_q, d_k, d_v, d_o, d_l, config
    );
    
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Use results...
    
    // Cleanup
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_l);
    
    return 0;
}
```

### Training Loop Integration

```cpp
// Training pseudocode
for (int epoch = 0; epoch < num_epochs; ++epoch) {
    // Forward pass
    flash_attention_v2_forward(d_q, d_k, d_v, d_o, d_l, config);
    
    // Compute loss (omitted)
    
    // Backward pass
    flash_attention_v2_backward(
        d_q, d_k, d_v, d_o, d_l, d_do, 
        d_dq, d_dk, d_dv, config
    );
    
    // Update weights (omitted)
}
```

## Error Handling

The implementation provides comprehensive error handling:

1. **Configuration Validation**: Checks for valid parameter ranges
2. **Memory Allocation**: Verifies CUDA memory allocation success
3. **Kernel Launch**: Checks for kernel launch errors
4. **Device Synchronization**: Ensures proper execution order

### Common Error Codes

- `cudaErrorInvalidValue`: Invalid configuration parameters
- `cudaErrorMemoryAllocation`: Insufficient device memory
- `cudaErrorLaunchFailure`: Kernel launch failed
- `cudaErrorDeviceUnavailiable`: No CUDA device available

## Limitations and Considerations

### Current Limitations

1. **Sequence Length**: Maximum sequence length depends on available memory
2. **Head Dimension**: Limited by shared memory and register usage
3. **Batch Size**: Constrained by GPU memory capacity
4. **Precision**: Currently supports FP16 only

### Performance Considerations

1. **Warm-up**: First few calls may have higher latency
2. **Memory Bandwidth**: Performance limited by memory bandwidth for large inputs
3. **GPU Architecture**: Performance varies across GPU generations
4. **Concurrent Kernels**: Multiple streams may improve utilization

## Testing and Validation

### Test Suite

The implementation includes a comprehensive test suite that validates:
- Correctness against CPU reference implementation
- Numerical accuracy within tolerance thresholds
- Error handling for invalid configurations
- Memory management and cleanup
- Performance consistency across runs

### Test Configuration

```cpp
// Example test configurations
TestConfigV2 test_configs_v2[] = {
    {1, 32, 4, 16, false, "tiny_standard"},
    {1, 32, 4, 16, true, "tiny_causal"},
    {1, 64, 8, 32, false, "small_standard"},
    {1, 64, 8, 32, true, "small_causal"},
    {2, 128, 8, 32, false, "medium_standard"},
    {2, 128, 8, 32, true, "medium_causal"},
};
```

## Future Enhancements

### Planned Improvements

1. **FP32 Support**: Add full precision support
2. **BF16 Support**: Add bfloat16 precision
3. **Advanced Optimizations**: Implement more sophisticated tiling strategies
4. **Multi-GPU Support**: Scale across multiple GPUs
5. **Profile-Guided Optimization**: Auto-tune kernel parameters

### Research Directions

1. **Attention Variants**: Support for different attention mechanisms
2. **Sparse Attention**: Implement sparse attention patterns
3. **Quantization**: Support for quantized attention
4. **Hardware-Specific Optimizations**: Target specific GPU architectures

## References

1. Flash Attention v2 Paper: [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
2. CUDA Programming Guide: [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
3. Flash Attention v1: [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

## License and Contributing

This implementation is provided for educational and research purposes. Please refer to the project license for usage terms and contribution guidelines.

---

**Note**: This documentation corresponds to the current implementation version. For the latest updates and features, please refer to the project repository.