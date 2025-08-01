import torch
import triton
import triton.language as tl


@triton.jit
def coalesced_access_kernel(
    output_ptr, input_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel demonstrating coalesced memory access pattern
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Number of elements processed by each thread block
    """
    # Calculate program ID and offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block - contiguous access pattern
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary handling
    mask = offsets < n_elements
    
    # Load data from global memory with coalesced access
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Simple operation: square the values
    result = data * data
    
    # Store result back to global memory with coalesced access
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def strided_access_kernel(
    output_ptr, input_ptr, 
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel demonstrating strided (non-coalesced) memory access pattern
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor
        n_rows: Number of rows
        n_cols: Number of columns
        BLOCK_SIZE: Block size for computation
    """
    # Process one row per program
    row_idx = tl.program_id(axis=0)
    
    # Create column offsets
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Calculate row start pointer (assuming contiguous layout)
    row_start_ptr = input_ptr + row_idx * n_cols
    output_row_start_ptr = output_ptr + row_idx * n_cols
    
    # Create mask for boundary handling (both row and column bounds)
    row_mask = row_idx < n_rows
    col_mask = col_offsets < n_cols
    mask = row_mask & col_mask
    
    # Load data with strided access pattern
    data = tl.load(row_start_ptr + col_offsets, mask=mask, other=0.0)
    
    # Simple operation
    result = data * 2.0
    
    # Store result with strided access pattern
    tl.store(output_row_start_ptr + col_offsets, result, mask=mask)


@triton.jit
def shared_memory_kernel(
    output_ptr, input_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simplified Triton kernel for 2D processing - treats 2D as 1D
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor
        n_rows: Number of rows
        n_cols: Number of columns
        BLOCK_SIZE: Block size for computation
    """
    # Treat 2D as 1D for simplicity
    pid = tl.program_id(axis=0)
    total_elements = n_rows * n_cols
    
    # Calculate offsets
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask
    mask = offsets < total_elements
    
    # Convert 1D offset to 2D coordinates
    row_indices = offsets // n_cols
    col_indices = offsets % n_cols
    
    # Load data with 2D indexing
    data = tl.load(input_ptr + row_indices * n_cols + col_indices, mask=mask, other=0.0)
    
    # Process data
    result = data * 2.0 + 1.0
    
    # Store result with 2D indexing
    tl.store(output_ptr + row_indices * n_cols + col_indices, result, mask=mask)


def coalesced_access(x: torch.Tensor):
    """
    Demonstrate coalesced memory access pattern
    
    Args:
        x: Input tensor
        
    Returns:
        output: x^2 (element-wise square)
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Determine launch configuration
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    coalesced_access_kernel[(grid_size,)](
        output,
        x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def strided_access(x: torch.Tensor):
    """
    Demonstrate strided memory access pattern
    
    Args:
        x: Input tensor (2D)
        
    Returns:
        output: x * 2 (element-wise multiplication by 2)
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dim() == 2, "Input tensor must be 2D"
    
    # Create output tensor
    output = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1024
    
    # Launch kernel
    grid_size = (n_rows,)
    strided_access_kernel[grid_size](
        output,
        x,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def shared_memory_access(x: torch.Tensor):
    """
    Demonstrate 2D processing with simplified kernel
    
    Args:
        x: Input tensor (2D)
        
    Returns:
        output: x * 2 + 1 (element-wise operation)
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    assert x.dim() == 2, "Input tensor must be 2D"
    
    # Create output tensor
    output = torch.empty_like(x)
    
    n_rows, n_cols = x.shape
    total_elements = n_rows * n_cols
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    shared_memory_kernel[(grid_size,)](
        output,
        x,
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def benchmark_memory_patterns(size: int = 4096, warmup: int = 10, repeat: int = 100):
    """
    Benchmark different memory access patterns
    
    Args:
        size: Size of the test matrix (size x size)
        warmup: Number of warmup iterations
        repeat: Number of benchmark iterations
        
    Returns:
        dict: Benchmark results
    """
    import time
    
    # Create test data
    x = torch.randn(size, size, device='cuda', dtype=torch.float32)
    
    results = {}
    
    # Benchmark coalesced access (1D)
    x_1d = x.flatten()
    
    # Warmup
    for _ in range(warmup):
        _ = coalesced_access(x_1d)
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(repeat):
        _ = coalesced_access(x_1d)
        torch.cuda.synchronize()
    end_time = time.time()
    
    results['coalesced_time'] = (end_time - start_time) / repeat
    
    # Benchmark strided access (2D)
    # Warmup
    for _ in range(warmup):
        _ = strided_access(x)
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(repeat):
        _ = strided_access(x)
        torch.cuda.synchronize()
    end_time = time.time()
    
    results['strided_time'] = (end_time - start_time) / repeat
    
    # Benchmark shared memory access (2D)
    # Warmup
    for _ in range(warmup):
        _ = shared_memory_access(x)
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(repeat):
        _ = shared_memory_access(x)
        torch.cuda.synchronize()
    end_time = time.time()
    
    results['shared_memory_time'] = (end_time - start_time) / repeat
    
    # Calculate theoretical memory bandwidth
    total_bytes = x.numel() * x.element_size() * 2  # Read + write
    
    results['coalesced_bandwidth'] = total_bytes / (results['coalesced_time'] * 1e9)
    results['strided_bandwidth'] = total_bytes / (results['strided_time'] * 1e9)
    results['shared_memory_bandwidth'] = total_bytes / (results['shared_memory_time'] * 1e9)
    
    results['matrix_size'] = (size, size)
    results['total_elements'] = size * size
    
    return results


def memory_efficiency_analysis(size: int = 2048):
    """
    Analyze memory efficiency of different access patterns
    
    Args:
        size: Size of the test matrix
        
    Returns:
        dict: Analysis results
    """
    # Create test data
    x = torch.randn(size, size, device='cuda', dtype=torch.float32)
    
    # Baseline PyTorch operations
    baseline_time = benchmark_pytorch_baseline(x)
    
    # Triton implementations
    triton_results = benchmark_memory_patterns(size, warmup=5, repeat=20)
    
    # Calculate efficiency ratios
    efficiency = {
        'coalesced_efficiency': baseline_time / triton_results['coalesced_time'],
        'strided_efficiency': baseline_time / triton_results['strided_time'],
        'shared_memory_efficiency': baseline_time / triton_results['shared_memory_time'],
        'bandwidth_coalesced_ratio': triton_results['coalesced_bandwidth'] / triton_results['strided_bandwidth'],
        'bandwidth_shared_ratio': triton_results['shared_memory_bandwidth'] / triton_results['strided_bandwidth'],
    }
    
    return {
        'triton_results': triton_results,
        'baseline_time': baseline_time,
        'efficiency': efficiency,
        'matrix_size': (size, size)
    }


def benchmark_pytorch_baseline(x: torch.Tensor, repeat: int = 20):
    """
    Benchmark PyTorch baseline for comparison
    
    Args:
        x: Input tensor
        repeat: Number of iterations
        
    Returns:
        float: Average time per iteration
    """
    import time
    
    # Warmup
    for _ in range(5):
        _ = x * x
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(repeat):
        _ = x * x
        torch.cuda.synchronize()
    end_time = time.time()
    
    return (end_time - start_time) / repeat