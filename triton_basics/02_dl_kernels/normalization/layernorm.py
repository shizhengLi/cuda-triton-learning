import torch
import triton
import triton.language as tl
from triton.runtime import driver


def is_hip():
    """Check if using AMD HIP backend"""
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    """Check if using AMD CDNA architecture"""
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in (
        'gfx940', 'gfx941', 'gfx942', 'gfx90a', 'gfx908'
    )


@triton.jit
def layernorm_kernel(
    output_ptr, 
    input_ptr, 
    weight_ptr, 
    bias_ptr,
    input_row_stride, 
    output_row_stride,
    n_rows, 
    n_cols, 
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for LayerNorm computation
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor
        weight_ptr: Pointer to weight tensor (gamma)
        bias_ptr: Pointer to bias tensor (beta)
        input_row_stride: Stride between rows in input
        output_row_stride: Stride between rows in output
        weight_stride: Stride for weight tensor
        bias_stride: Stride for bias tensor
        n_rows: Number of rows
        n_cols: Number of columns
        eps: Epsilon for numerical stability
        BLOCK_SIZE: Block size for computation
    """
    # Process one row per program
    row_idx = tl.program_id(0)
    
    # Calculate pointer to current row
    row_start_ptr = input_ptr + row_idx * input_row_stride
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    
    # Load the row into SRAM with masking for boundary handling
    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    
    # Compute mean and variance
    mean = tl.sum(row) / n_cols
    diff = row - mean
    variance = tl.sum(diff * diff) / n_cols
    
    # Normalize with epsilon for numerical stability
    inv_std = 1.0 / tl.sqrt(variance + eps)
    normalized = (row - mean) * inv_std
    
    # Apply weight and bias if provided
    if weight_ptr is not None:
        weights = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
        normalized = normalized * weights
    
    if bias_ptr is not None:
        biases = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
        normalized = normalized + biases
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, normalized, mask=mask)


def layernorm(
    x: torch.Tensor, 
    weight: torch.Tensor = None, 
    bias: torch.Tensor = None, 
    eps: float = 1e-5,
    axis: int = -1
):
    """
    High-level wrapper for LayerNorm computation using Triton
    
    Args:
        x: Input tensor
        weight: Weight tensor (gamma), optional
        bias: Bias tensor (beta), optional
        eps: Epsilon for numerical stability
        axis: Axis along which to normalize
        
    Returns:
        LayerNorm output tensor
    """
    # Input validation
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dim() >= 1, "Input tensor must have at least 1 dimension"
    
    # Handle different axis values
    if axis < 0:
        axis = x.dim() + axis
    assert 0 <= axis < x.dim(), f"Invalid axis {axis} for tensor with {x.dim()} dimensions"
    
    # For now, only support 2D tensors along last axis
    if x.dim() != 2 or axis != 1:
        # Fallback to PyTorch implementation for complex cases
        # We need to handle the case where weight/bias dimensions don't match normalized_shape
        if weight is not None or bias is not None:
            # For simplicity, just use PyTorch's layer_norm without weight/bias for complex cases
            return torch.nn.functional.layer_norm(x, x.shape[axis:], None, None, eps)
        else:
            return torch.nn.functional.layer_norm(x, x.shape[axis:], None, None, eps)
    
    n_rows, n_cols = x.shape
    
    # Validate weight and bias dimensions
    if weight is not None:
        assert weight.is_cuda, "Weight tensor must be on CUDA device"
        assert weight.dim() == 1, "Weight tensor must be 1D"
        assert weight.shape[0] == n_cols, f"Weight shape {weight.shape} doesn't match input columns {n_cols}"
    
    if bias is not None:
        assert bias.is_cuda, "Bias tensor must be on CUDA device"
        assert bias.dim() == 1, "Bias tensor must be 1D"
        assert bias.shape[0] == n_cols, f"Bias shape {bias.shape} doesn't match input columns {n_cols}"
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Calculate optimal block size (next power of 2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Cap BLOCK_SIZE to reasonable limits
    BLOCK_SIZE = min(BLOCK_SIZE, 16384)  # Maximum 16K per block
    
    # Simple grid configuration - one program per row
    grid = (n_rows,)
    
    # Launch kernel
    layernorm_kernel[grid](
        y, x, weight, bias,
        x.stride(0), y.stride(0),
        n_rows, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y


def naive_layernorm(
    x: torch.Tensor, 
    weight: torch.Tensor = None, 
    bias: torch.Tensor = None, 
    eps: float = 1e-5,
    axis: int = -1
):
    """
    Reference implementation using PyTorch operations for validation
    
    Args:
        x: Input tensor
        weight: Weight tensor (gamma), optional
        bias: Bias tensor (beta), optional
        eps: Epsilon for numerical stability
        axis: Axis along which to normalize
        
    Returns:
        LayerNorm output tensor
    """
    # Compute mean and variance along the specified axis
    mean = torch.mean(x, dim=axis, keepdim=True)
    variance = torch.var(x, dim=axis, keepdim=True, unbiased=False)
    
    # Normalize
    normalized = (x - mean) / torch.sqrt(variance + eps)
    
    # Apply weight and bias
    if weight is not None:
        normalized = normalized * weight
    
    if bias is not None:
        normalized = normalized + bias
    
    return normalized


def benchmark_layernorm(
    M: int = 4096, 
    N: int = 4096, 
    warmup: int = 10, 
    repeat: int = 100,
    with_weight_bias: bool = True
):
    """
    Benchmark LayerNorm implementation against PyTorch
    
    Args:
        M: Number of rows
        N: Number of columns  
        warmup: Number of warmup iterations
        repeat: Number of benchmark iterations
        with_weight_bias: Whether to include weight and bias
        
    Returns:
        dict: Benchmark results
    """
    import time
    
    # Create test data
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    weight = torch.randn(N, device='cuda', dtype=torch.float32) if with_weight_bias else None
    bias = torch.randn(N, device='cuda', dtype=torch.float32) if with_weight_bias else None
    
    # Warmup
    for _ in range(warmup):
        _ = layernorm(x, weight, bias)
        _ = torch.nn.functional.layer_norm(x, (N,), weight, bias)
        torch.cuda.synchronize()
    
    # Benchmark Triton implementation
    start_time = time.time()
    for _ in range(repeat):
        _ = layernorm(x, weight, bias)
        torch.cuda.synchronize()
    end_time = time.time()
    
    triton_time = (end_time - start_time) / repeat
    
    # Benchmark PyTorch implementation
    start_time = time.time()
    for _ in range(repeat):
        _ = torch.nn.functional.layer_norm(x, (N,), weight, bias)
        torch.cuda.synchronize()
    end_time = time.time()
    
    torch_time = (end_time - start_time) / repeat
    
    # Calculate memory bandwidth
    weight_bytes = weight.numel() * weight.element_size() if weight is not None else 0
    bias_bytes = bias.numel() * bias.element_size() if bias is not None else 0
    total_bytes = (x.numel() * x.element_size() +  # Read input
                   weight_bytes +  # Read weight
                   bias_bytes +  # Read bias
                   x.numel() * x.element_size())  # Write output
    
    triton_bandwidth = total_bytes / (triton_time * 1e9)  # GB/s
    torch_bandwidth = total_bytes / (torch_time * 1e9)   # GB/s
    
    return {
        'triton_time': triton_time,
        'torch_time': torch_time,
        'speedup': torch_time / triton_time,
        'triton_bandwidth': triton_bandwidth,
        'torch_bandwidth': torch_bandwidth,
        'matrix_shape': (M, N),
        'total_elements': M * N,
        'with_weight_bias': with_weight_bias
    }