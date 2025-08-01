import torch
import triton
import triton.language as tl
import math


@triton.jit
def relu_kernel(
    output_ptr, input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for ReLU activation function
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size for computation
    """
    # Calculate program ID and offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary handling
    mask = offsets < n_elements
    
    # Load data from global memory
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply ReLU activation: max(0, x)
    result = tl.maximum(x, 0.0)
    
    # Store result back to global memory
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def gelu_kernel(
    output_ptr, input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for GELU activation function (approximate version)
    
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size for computation
    """
    # Calculate program ID and offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary handling
    mask = offsets < n_elements
    
    # Load data from global memory
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Constants for GELU approximation
    sqrt_2_over_pi = tl.sqrt(2.0 / math.pi)
    coeff = 0.044715
    
    # Apply GELU activation
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + coeff * x_cubed)
    
    # Approximate tanh using sigmoid: tanh(x) = 2 * sigmoid(2*x) - 1
    sigmoid_2x = 1.0 / (1.0 + tl.exp(-2.0 * inner))
    tanh_approx = 2.0 * sigmoid_2x - 1.0
    
    result = 0.5 * x * (1.0 + tanh_approx)
    
    # Store result back to global memory
    tl.store(output_ptr + offsets, result, mask=mask)


@triton.jit
def silu_kernel(
    output_ptr, input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for SiLU activation function (Sigmoid-weighted Linear Unit)
    
    SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    
    Args:
        output_ptr: Pointer to output tensor
        input_ptr: Pointer to input tensor
        n_elements: Total number of elements
        BLOCK_SIZE: Block size for computation
    """
    # Calculate program ID and offsets
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Create offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for boundary handling
    mask = offsets < n_elements
    
    # Load data from global memory
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply SiLU activation: x * sigmoid(x)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    result = x * sigmoid
    
    # Store result back to global memory
    tl.store(output_ptr + offsets, result, mask=mask)


def relu(x: torch.Tensor):
    """
    ReLU activation function using Triton
    
    Args:
        x: Input tensor
        
    Returns:
        output: max(0, x)
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Determine launch configuration
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    relu_kernel[(grid_size,)](
        output,
        x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def gelu(x: torch.Tensor):
    """
    GELU activation function using Triton (approximate version)
    
    Args:
        x: Input tensor
        
    Returns:
        output: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Determine launch configuration
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    gelu_kernel[(grid_size,)](
        output,
        x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def silu(x: torch.Tensor):
    """
    SiLU activation function using Triton (Sigmoid-weighted Linear Unit)
    
    Args:
        x: Input tensor
        
    Returns:
        output: x * sigmoid(x)
    """
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Create output tensor
    output = torch.empty_like(x)
    
    # Determine launch configuration
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    silu_kernel[(grid_size,)](
        output,
        x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


# Reference implementations for validation
def relu_reference(x: torch.Tensor):
    """Reference ReLU implementation using PyTorch"""
    return torch.relu(x)


def gelu_reference(x: torch.Tensor):
    """Reference GELU implementation using PyTorch"""
    return torch.nn.functional.gelu(x)


def silu_reference(x: torch.Tensor):
    """Reference SiLU implementation using PyTorch"""
    return torch.nn.functional.silu(x)


def benchmark_activations(
    activation_func: str,
    size: int = 1000000,
    warmup: int = 10,
    repeat: int = 100
):
    """
    Benchmark activation function performance
    
    Args:
        activation_func: Name of activation function ('relu', 'gelu', 'silu')
        size: Size of input tensor
        warmup: Number of warmup iterations
        repeat: Number of benchmark iterations
        
    Returns:
        dict: Benchmark results
    """
    import time
    
    # Create test data
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    
    # Get Triton implementation
    if activation_func == 'relu':
        triton_func = relu
        torch_func = torch.relu
    elif activation_func == 'gelu':
        triton_func = gelu
        torch_func = torch.nn.functional.gelu
    elif activation_func == 'silu':
        triton_func = silu
        torch_func = torch.nn.functional.silu
    else:
        raise ValueError(f"Unknown activation function: {activation_func}")
    
    # Warmup
    for _ in range(warmup):
        _ = triton_func(x)
        _ = torch_func(x)
        torch.cuda.synchronize()
    
    # Benchmark Triton implementation
    start_time = time.time()
    for _ in range(repeat):
        _ = triton_func(x)
        torch.cuda.synchronize()
    end_time = time.time()
    
    triton_time = (end_time - start_time) / repeat
    
    # Benchmark PyTorch implementation
    start_time = time.time()
    for _ in range(repeat):
        _ = torch_func(x)
        torch.cuda.synchronize()
    end_time = time.time()
    
    torch_time = (end_time - start_time) / repeat
    
    # Calculate memory bandwidth
    total_bytes = x.numel() * x.element_size() * 2  # Read + write
    triton_bandwidth = total_bytes / (triton_time * 1e9)
    torch_bandwidth = total_bytes / (torch_time * 1e9)
    
    return {
        'activation': activation_func,
        'triton_time': triton_time,
        'torch_time': torch_time,
        'speedup': torch_time / triton_time,
        'triton_bandwidth': triton_bandwidth,
        'torch_bandwidth': torch_bandwidth,
        'input_size': size,
        'total_elements': size
    }


def benchmark_all_activations(size: int = 1000000, warmup: int = 10, repeat: int = 100):
    """
    Benchmark all activation functions
    
    Args:
        size: Size of input tensor
        warmup: Number of warmup iterations
        repeat: Number of benchmark iterations
        
    Returns:
        dict: Benchmark results for all activations
    """
    activations = ['relu', 'gelu', 'silu']
    results = {}
    
    for activation in activations:
        results[activation] = benchmark_activations(activation, size, warmup, repeat)
    
    return results