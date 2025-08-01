import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    x_ptr, 
    y_ptr, 
    output_ptr, 
    n_elements, 
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for vector addition: output = x + y
    
    Args:
        x_ptr: Pointer to input vector x
        y_ptr: Pointer to input vector y  
        output_ptr: Pointer to output vector
        n_elements: Total number of elements in vectors
        BLOCK_SIZE: Number of elements processed by each thread block
    """
    # Calculate program ID and offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary handling
    mask = offsets < n_elements
    
    # Load data from global memory
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform vector addition
    output = x + y
    
    # Store result back to global memory
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor):
    """
    High-level wrapper for vector addition using Triton
    
    Args:
        x: Input tensor
        y: Input tensor
        
    Returns:
        output: x + y
    """
    # Input validation
    assert x.shape == y.shape, "Input tensors must have same shape"
    assert x.is_cuda and y.is_cuda, "Input tensors must be on CUDA device"
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Determine launch configuration
    n_elements = x.numel()
    BLOCK_SIZE = 1024  # Elements per thread block
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    vector_add_kernel[grid_size,](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def benchmark_vector_add(size: int = 1000000, warmup: int = 10, repeat: int = 100):
    """
    Benchmark vector addition implementation
    
    Args:
        size: Size of vectors to test
        warmup: Number of warmup iterations
        repeat: Number of benchmark iterations
        
    Returns:
        dict: Benchmark results
    """
    import time
    
    # Create test data
    x = torch.randn(size, device='cuda')
    y = torch.randn(size, device='cuda')
    
    # Warmup
    for _ in range(warmup):
        _ = vector_add(x, y)
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(repeat):
        _ = vector_add(x, y)
        torch.cuda.synchronize()
    end_time = time.time()
    
    triton_time = (end_time - start_time) / repeat
    
    # PyTorch baseline
    start_time = time.time()
    for _ in range(repeat):
        _ = x + y
        torch.cuda.synchronize()
    end_time = time.time()
    
    torch_time = (end_time - start_time) / repeat
    
    return {
        'triton_time': triton_time,
        'torch_time': torch_time,
        'speedup': torch_time / triton_time,
        'bandwidth': (size * 4 * 3) / (triton_time * 1e9)  # GB/s (3 vectors * 4 bytes)
    }