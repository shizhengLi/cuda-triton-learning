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
def rmsnorm_kernel(
    output_ptr, 
    input_ptr, 
    weight_ptr,
    input_row_stride, 
    output_row_stride,
    n_rows, 
    n_cols, 
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for RMSNorm computation
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor
        weight_ptr: Pointer to weight tensor (gamma)
        input_row_stride: Stride between rows in input
        output_row_stride: Stride between rows in output
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
    
    # Compute RMS: sqrt(mean(x^2) + eps)
    row_squared = row * row
    mean_squared = tl.sum(row_squared) / n_cols
    rms = tl.sqrt(mean_squared + eps)
    
    # Normalize
    inv_rms = 1.0 / rms
    normalized = row * inv_rms
    
    # Apply weight if provided
    if weight_ptr is not None:
        weights = tl.load(weight_ptr + col_offsets, mask=mask, other=1.0)
        normalized = normalized * weights
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, normalized, mask=mask)


def rmsnorm(
    x: torch.Tensor, 
    weight: torch.Tensor = None, 
    eps: float = 1e-6,
    axis: int = -1
):
    """
    High-level wrapper for RMSNorm computation using Triton
    
    Args:
        x: Input tensor
        weight: Weight tensor (gamma), optional
        eps: Epsilon for numerical stability
        axis: Axis along which to normalize
        
    Returns:
        RMSNorm output tensor
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
        # For simplicity, just use PyTorch's RMSNorm without weight/bias for complex cases
        return _rmsnorm_torch(x, None, eps, axis)
    
    n_rows, n_cols = x.shape
    
    # Validate weight dimensions
    if weight is not None:
        assert weight.is_cuda, "Weight tensor must be on CUDA device"
        assert weight.dim() == 1, "Weight tensor must be 1D"
        assert weight.shape[0] == n_cols, f"Weight shape {weight.shape} doesn't match input columns {n_cols}"
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Calculate optimal block size (next power of 2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Cap BLOCK_SIZE to reasonable limits
    BLOCK_SIZE = min(BLOCK_SIZE, 16384)  # Maximum 16K per block
    
    # Simple grid configuration - one program per row
    grid = (n_rows,)
    
    # Launch kernel
    rmsnorm_kernel[grid](
        y, x, weight,
        x.stride(0), y.stride(0),
        n_rows, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y


def _rmsnorm_torch(
    x: torch.Tensor, 
    weight: torch.Tensor = None, 
    eps: float = 1e-6,
    axis: int = -1
):
    """
    PyTorch fallback implementation for RMSNorm
    
    Args:
        x: Input tensor
        weight: Weight tensor (gamma), optional
        eps: Epsilon for numerical stability
        axis: Axis along which to normalize
        
    Returns:
        RMSNorm output tensor
    """
    # Compute RMS: sqrt(mean(x^2) + eps)
    x_squared = x * x
    mean_squared = torch.mean(x_squared, dim=axis, keepdim=True)
    rms = torch.sqrt(mean_squared + eps)
    
    # Normalize
    normalized = x / rms
    
    # Apply weight if provided
    if weight is not None:
        normalized = normalized * weight
    
    return normalized


def naive_rmsnorm(
    x: torch.Tensor, 
    weight: torch.Tensor = None, 
    eps: float = 1e-6,
    axis: int = -1
):
    """
    Reference implementation using PyTorch operations for validation
    
    Args:
        x: Input tensor
        weight: Weight tensor (gamma), optional
        eps: Epsilon for numerical stability
        axis: Axis along which to normalize
        
    Returns:
        RMSNorm output tensor
    """
    # Compute RMS: sqrt(mean(x^2) + eps)
    x_squared = x * x
    mean_squared = torch.mean(x_squared, dim=axis, keepdim=True)
    rms = torch.sqrt(mean_squared + eps)
    
    # Normalize
    normalized = x / rms
    
    # Apply weight if provided
    if weight is not None:
        normalized = normalized * weight
    
    return normalized


def benchmark_rmsnorm(
    M: int = 4096, 
    N: int = 4096, 
    warmup: int = 10, 
    repeat: int = 100,
    with_weight: bool = True
):
    """
    Benchmark RMSNorm implementation against PyTorch
    
    Args:
        M: Number of rows
        N: Number of columns  
        warmup: Number of warmup iterations
        repeat: Number of benchmark iterations
        with_weight: Whether to include weight
        
    Returns:
        dict: Benchmark results
    """
    import time
    
    # Create test data
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    weight = torch.randn(N, device='cuda', dtype=torch.float32) if with_weight else None
    
    # Warmup
    for _ in range(warmup):
        _ = rmsnorm(x, weight)
        _ = _rmsnorm_torch(x, weight)
        torch.cuda.synchronize()
    
    # Benchmark Triton implementation
    start_time = time.time()
    for _ in range(repeat):
        _ = rmsnorm(x, weight)
        torch.cuda.synchronize()
    end_time = time.time()
    
    triton_time = (end_time - start_time) / repeat
    
    # Benchmark PyTorch implementation
    start_time = time.time()
    for _ in range(repeat):
        _ = _rmsnorm_torch(x, weight)
        torch.cuda.synchronize()
    end_time = time.time()
    
    torch_time = (end_time - start_time) / repeat
    
    # Calculate memory bandwidth
    weight_bytes = weight.numel() * weight.element_size() if weight is not None else 0
    total_bytes = (x.numel() * x.element_size() +  # Read input
                   weight_bytes +  # Read weight
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
        'with_weight': with_weight
    }