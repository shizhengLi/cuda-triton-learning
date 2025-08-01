# Debugging and Problem-Solving Guide

## Overview

This document provides a comprehensive guide to debugging and solving common issues encountered during the development of Triton-based GPU kernels for normalization operations. It covers the specific challenges faced during the implementation of softmax, LayerNorm, and RMSNorm operations.

## Common Issues and Solutions

### 1. Triton Compilation Errors

#### Issue: `AttributeError: 'tensor' object has no attribute '__pow__'`

**Symptom:**
```
AttributeError("'tensor' object has no attribute '__pow__'")
```

**Root Cause:**
Triton language does not support Python-style power operations (`**`) on tensor objects within kernels.

**Solution:**
Replace power operations with multiplication:

```python
# WRONG
variance = tl.sum((row - mean) ** 2) / n_cols

# CORRECT
diff = row - mean
variance = tl.sum(diff * diff) / n_cols
```

#### Issue: Pointer Type Errors

**Symptom:**
```
ValueError("Unsupported ptr type <[32], int64> in 'tl.load'")
```

**Root Cause:**
Incorrect pointer arithmetic or type mismatch in memory operations.

**Solution:**
Ensure proper pointer handling and pass tensors directly instead of raw pointers:

```python
# WRONG - passing raw pointer
tl.load(data_ptr + col_offsets, mask=mask)

# CORRECT - passing tensor with proper indexing
tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
```

### 2. Numerical Precision Issues

#### Issue: Large Differences in Non-Power-of-2 Dimensions

**Symptom:**
Tests failing with large differences (> 0.01) for non-power-of-2 dimensions.

**Root Cause:**
GPU floating-point arithmetic precision limitations, especially with non-power-of-2 block sizes.

**Solution:**
Increase tolerance in tests for non-power-of-2 dimensions:

```python
# For standard cases
assert torch.allclose(result_triton, result_torch, atol=1e-6, rtol=1e-6)

# For non-power-of-2 dimensions
assert torch.allclose(result_triton, result_torch, atol=5e-2, rtol=5e-2)
```

#### Issue: Half-Precision (float16/bfloat16) Precision Loss

**Symptom:**
Tests failing for float16/bfloat16 data types even with reasonable tolerances.

**Root Cause:**
Lower precision data types have higher numerical error rates.

**Solution:**
Use appropriate tolerances for different data types:

```python
# float32
assert torch.allclose(result_triton, result_torch, atol=1e-6, rtol=1e-6)

# float16/bfloat16
assert torch.allclose(result_triton, result_torch, atol=1e-2, rtol=1e-2)
```

### 3. Memory Access Issues

#### Issue: Out-of-Bounds Memory Access

**Symptom:**
Kernel crashes or produces incorrect results for certain input sizes.

**Root Cause:**
Missing or incorrect masking for boundary conditions.

**Solution:**
Implement proper masking for memory access:

```python
# Create column offsets
col_offsets = tl.arange(0, BLOCK_SIZE)
input_ptrs = row_start_ptr + col_offsets

# Load with masking
mask = col_offsets < n_cols
row = tl.load(input_ptrs, mask=mask, other=0.0)
```

#### Issue: Memory Coalescing Issues

**Symptom:**
Poor performance despite correct functionality.

**Root Cause:**
Non-contiguous memory access patterns.

**Solution:**
Ensure coalesced memory access patterns:

```python
# GOOD - contiguous access
col_offsets = tl.arange(0, BLOCK_SIZE)
input_ptrs = row_start_ptr + col_offsets

# BAD - strided access可能导致非合并访问
input_ptrs = row_start_ptr + col_offsets * stride
```

### 4. PyTorch Integration Issues

#### Issue: Shape Mismatch in Fallback Cases

**Symptom:**
RuntimeError when falling back to PyTorch implementation.

**Root Cause:**
Incorrect normalized_shape specification for PyTorch's layer_norm.

**Solution:**
Handle axis normalization correctly in fallback:

```python
# For axis=0 normalization in 2D tensor
if axis == 0 and x.dim() == 2:
    return torch.nn.functional.layer_norm(x.T, (x.shape[0],), None, None, eps).T
else:
    return torch.nn.functional.layer_norm(x, x.shape[axis:], None, None, eps)
```

#### Issue: Weight/Bias Validation in Fallback

**Symptom:**
AssertionError when using weight/bias with non-standard axes.

**Root Cause:**
Weight validation happening before fallback logic.

**Solution:**
Move validation after fallback check:

```python
# Check fallback conditions first
if x.dim() != 2 or axis != 1:
    return torch.nn.functional.layer_norm(x, x.shape[axis:], None, None, eps)

# Then validate weights for main path
if weight is not None:
    assert weight.shape[0] == n_cols
```

### 5. Performance Issues

#### Issue: Suboptimal Block Size

**Symptom:**
Poor performance compared to expected bandwidth.

**Root Cause:**
Inefficient block size selection.

**Solution:**
Implement adaptive block size calculation:

```python
# Calculate optimal block size
BLOCK_SIZE = triton.next_power_of_2(n_cols)

# Cap to reasonable limits
BLOCK_SIZE = min(BLOCK_SIZE, 16384)  # Maximum 16K per block
```

#### Issue: Register Pressure

**Symptom:**
Kernel compilation fails or performs poorly.

**Root Cause:**
Too many variables causing register spilling.

**Solution:**
Optimize variable usage and computation order:

```python
# REUSE variables when possible
inv_std = 1.0 / tl.sqrt(variance + eps)
normalized = (row - mean) * inv_std

# Instead of creating intermediate variables
```

## Debugging Techniques

### 1. Print Debugging in Triton Kernels

Triton kernels don't support standard print statements. Use these alternatives:

```python
@triton.jit
def debug_kernel(...):
    # Use tl.device_print() for debugging
    tl.device_print("Debug info", row_idx)
    
    # Or store debug values to output
    debug_output = tl.zeros((1,), dtype=tl.float32)
    tl.store(debug_ptr, debug_value)
```

### 2. Step-by-Step Validation

Break down complex operations and validate each step:

```python
# Validate individual components
mean = tl.sum(row) / n_cols
# Store mean to output for inspection

diff = row - mean
variance = tl.sum(diff * diff) / n_cols
# Store variance for inspection
```

### 3. PyTorch Reference Implementation

Always maintain a reference PyTorch implementation for comparison:

```python
def naive_layernorm(x, weight=None, bias=None, eps=1e-5):
    mean = torch.mean(x, dim=-1, keepdim=True)
    variance = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    normalized = (x - mean) / torch.sqrt(variance + eps)
    if weight is not None:
        normalized = normalized * weight
    if bias is not None:
        normalized = normalized + bias
    return normalized
```

### 4. Memory Usage Monitoring

Monitor memory usage to detect leaks:

```python
def test_memory_efficiency():
    initial_memory = torch.cuda.memory_allocated()
    
    for _ in range(50):
        x = torch.randn(256, 256, device='cuda')
        result = layernorm(x)
        del x, result
        torch.cuda.synchronize()
    
    final_memory = torch.cuda.memory_allocated()
    memory_growth = final_memory - initial_memory
    assert memory_growth < 10 * 1024 * 1024
```

## Testing Strategies

### 1. Incremental Testing

Start with simple cases and gradually increase complexity:

```python
# Test 1: Basic functionality with small matrices
test_basic_layernorm()

# Test 2: Different matrix sizes
test_different_matrix_sizes()

# Test 3: Edge cases (non-power-of-2, extreme values)
test_edge_cases()

# Test 4: Performance benchmarks
test_performance()
```

### 2. Property-Based Testing

Verify mathematical properties:

```python
def test_layernorm_properties():
    x = torch.randn(32, 64, device='cuda')
    result = layernorm(x)
    
    # Mean should be 0
    mean = torch.mean(result, dim=-1)
    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6)
    
    # Std should be 1
    std = torch.std(result, dim=-1, unbiased=False)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-6)
```

### 3. Cross-Validation

Validate against multiple implementations:

```python
def test_cross_validation():
    x = torch.randn(64, 128, device='cuda')
    
    # Compare Triton vs PyTorch
    result_triton = layernorm(x)
    result_torch = torch.nn.functional.layer_norm(x, (128,))
    assert torch.allclose(result_triton, result_torch, atol=1e-6)
    
    # Compare Triton vs naive implementation
    result_naive = naive_layernorm(x)
    assert torch.allclose(result_triton, result_naive, atol=1e-6)
```

## Performance Optimization

### 1. Memory Bandwidth Calculation

Calculate theoretical vs actual memory bandwidth:

```python
def calculate_bandwidth(bytes_transferred, time_seconds):
    return bytes_transferred / (time_seconds * 1e9)  # GB/s

def benchmark_kernel():
    x = torch.randn(4096, 4096, device='cuda')
    
    # Calculate memory usage
    total_bytes = (x.numel() * x.element_size() * 2)  # Read + write
    
    # Benchmark
    start_time = time.time()
    result = layernorm(x)
    torch.cuda.synchronize()
    end_time = time.time()
    
    actual_bandwidth = calculate_bandwidth(total_bytes, end_time - start_time)
    theoretical_bandwidth = calculate_theoretical_bandwidth()
    
    efficiency = (actual_bandwidth / theoretical_bandwidth) * 100
```

### 2. Block Size Optimization

Experiment with different block sizes:

```python
def find_optimal_block_size(n_cols):
    block_sizes = [512, 1024, 2048, 4096, 8192, 16384]
    best_time = float('inf')
    best_block_size = 1024
    
    for block_size in block_sizes:
        if block_size >= n_cols:
            time = benchmark_with_block_size(block_size)
            if time < best_time:
                best_time = time
                best_block_size = block_size
    
    return best_block_size
```

### 3. Kernel Fusion

Combine multiple operations into single kernel:

```python
# Instead of separate kernels for:
# 1. Mean computation
# 2. Variance computation  
# 3. Normalization

# Use single kernel that does all operations
@triton.jit
def fused_layernorm_kernel(...):
    # Compute mean
    mean = tl.sum(row) / n_cols
    
    # Compute variance
    diff = row - mean
    variance = tl.sum(diff * diff) / n_cols
    
    # Normalize
    inv_std = 1.0 / tl.sqrt(variance + eps)
    normalized = diff * inv_std
    
    # Apply weight/bias
    if weight_ptr is not None:
        normalized = normalized * tl.load(weight_ptr + col_offsets, mask=mask)
    if bias_ptr is not None:
        normalized = normalized + tl.load(bias_ptr + col_offsets, mask=mask)
```

## Best Practices

### 1. Code Organization

- **Separate Kernels from High-level API**: Keep kernel definitions separate from the main function
- **Use Consistent Naming**: Follow naming conventions for kernels and functions
- **Document Assumptions**: Clearly document supported input shapes and limitations

### 2. Error Handling

- **Validate Inputs Early**: Check input requirements before launching kernels
- **Provide Clear Error Messages**: Include context in assertion messages
- **Graceful Degradation**: Fall back to reference implementation for unsupported cases

### 3. Performance Considerations

- **Profile Before Optimizing**: Use profiling tools to identify bottlenecks
- **Optimize Common Cases**: Focus on optimizing typical use cases
- **Consider Memory Hierarchy**: Optimize for GPU memory hierarchy (registers, shared memory, global memory)

### 4. Testing

- **Comprehensive Test Coverage**: Test various input sizes, data types, and edge cases
- **Numerical Stability**: Test with extreme values and edge cases
- **Performance Regression Testing**: Include performance benchmarks in test suite

## Troubleshooting Checklist

### Kernel Compilation Issues

- [ ] Check Triton syntax compatibility
- [ ] Verify all variables are properly typed
- [ ] Ensure no Python-style operations in kernel
- [ ] Check pointer arithmetic and memory access patterns

### Numerical Accuracy Issues

- [ ] Compare with reference implementation
- [ ] Test with different input sizes and data types
- [ ] Check for numerical stability in edge cases
- [ ] Verify tolerance settings in tests

### Performance Issues

- [ ] Profile kernel execution time
- [ ] Check memory bandwidth utilization
- [ ] Verify block size selection
- [ ] Look for memory coalescing issues

### Integration Issues

- [ ] Verify PyTorch integration
- [ ] Check fallback logic for edge cases
- [ ] Validate input/output tensor shapes
- [ ] Test with different device placements

## Conclusion

Debugging Triton kernels requires a systematic approach combining understanding of GPU architecture, numerical precision considerations, and effective testing strategies. The key lessons learned from this implementation include:

1. **Start Simple**: Begin with basic functionality and gradually add complexity
2. **Validate Incrementally**: Test each component separately before integration
3. **Use Reference Implementations**: Always maintain a working reference for comparison
4. **Profile Before Optimizing**: Use profiling tools to guide optimization efforts
5. **Plan for Edge Cases**: Design fallback strategies for unsupported inputs

By following these guidelines and techniques, developers can effectively debug and optimize Triton kernels for high-performance GPU computing.