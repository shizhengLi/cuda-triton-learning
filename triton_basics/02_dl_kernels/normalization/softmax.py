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
def softmax_kernel(
    output_ptr, 
    input_ptr, 
    input_row_stride, 
    output_row_stride, 
    n_rows, 
    n_cols, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for row-wise softmax computation
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor  
        input_row_stride: Stride between rows in input
        output_row_stride: Stride between rows in output
        n_rows: Number of rows
        n_cols: Number of columns
        BLOCK_SIZE: Block size for computation (power of 2)
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
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    
    # Numerical stability: subtract maximum value
    row_max = tl.max(row)
    row_minus_max = row - row_max
    
    # Exponentiation
    numerator = tl.exp(row_minus_max)
    
    # Sum for normalization
    denominator = tl.sum(numerator)
    
    # Avoid division by zero
    denominator_safe = tl.where(denominator == 0.0, 1.0, denominator)
    softmax_output = numerator / denominator_safe
    
    # Write back output to DRAM
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)


def softmax(x: torch.Tensor, axis: int = -1):
    """
    High-level wrapper for softmax computation using Triton
    
    Args:
        x: Input tensor
        axis: Axis along which to compute softmax (default: -1)
        
    Returns:
        Softmax output tensor
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
        return torch.softmax(x, dim=axis)
    
    n_rows, n_cols = x.shape
    
    # Allocate output tensor
    y = torch.empty_like(x)
    
    # Calculate optimal block size (next power of 2)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    # Cap BLOCK_SIZE to reasonable limits
    BLOCK_SIZE = min(BLOCK_SIZE, 16384)  # Maximum 16K per block
    
    # Simple grid configuration - one program per row
    grid = (n_rows,)
    
    # Launch kernel
    softmax_kernel[grid](
        y, x, x.stride(0), y.stride(0), n_rows, n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return y


def naive_softmax(x: torch.Tensor, axis: int = -1):
    """
    Reference implementation using PyTorch operations for validation
    
    Args:
        x: Input tensor
        axis: Axis along which to compute softmax
        
    Returns:
        Softmax output tensor
    """
    # Numerical stability: subtract maximum
    x_max = torch.max(x, dim=axis, keepdim=True).values
    z = x - x_max
    
    # Exponentiate and normalize
    numerator = torch.exp(z)
    denominator = torch.sum(numerator, dim=axis, keepdim=True)
    
    # Avoid division by zero
    denominator_safe = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    
    return numerator / denominator_safe


def benchmark_softmax(M: int = 4096, N: int = 4096, warmup: int = 10, repeat: int = 100):
    """
    Benchmark softmax implementation against PyTorch
    
    Args:
        M: Number of rows
        N: Number of columns  
        warmup: Number of warmup iterations
        repeat: Number of benchmark iterations
        
    Returns:
        dict: Benchmark results
    """
    import time
    
    # Create test data
    x = torch.randn(M, N, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(warmup):
        _ = softmax(x)
        _ = torch.softmax(x, dim=-1)
        torch.cuda.synchronize()
    
    # Benchmark Triton implementation
    start_time = time.time()
    for _ in range(repeat):
        _ = softmax(x)
        torch.cuda.synchronize()
    end_time = time.time()
    
    triton_time = (end_time - start_time) / repeat
    
    # Benchmark PyTorch implementation
    start_time = time.time()
    for _ in range(repeat):
        _ = torch.softmax(x, dim=-1)
        torch.cuda.synchronize()
    end_time = time.time()
    
    torch_time = (end_time - start_time) / repeat
    
    # Calculate memory bandwidth
    total_bytes = 2 * x.numel() * x.element_size()  # Read input + write output
    triton_bandwidth = total_bytes / (triton_time * 1e9)  # GB/s
    torch_bandwidth = total_bytes / (torch_time * 1e9)   # GB/s
    
    return {
        'triton_time': triton_time,
        'torch_time': torch_time,
        'speedup': torch_time / triton_time,
        'triton_bandwidth': triton_bandwidth,
        'torch_bandwidth': torch_bandwidth,
        'matrix_shape': (M, N),
        'total_elements': M * N
    }