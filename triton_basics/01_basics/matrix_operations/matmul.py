import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Triton kernel for matrix multiplication: C = A @ B
    
    Args:
        a_ptr: Pointer to matrix A (M x K)
        b_ptr: Pointer to matrix B (K x N)
        c_ptr: Pointer to output matrix C (M x N)
        M, N, K: Matrix dimensions
        stride_*: Strides for each matrix dimension
        BLOCK_SIZE_*: Block sizes for tiling
        GROUP_SIZE_M: Group size for scheduling
    """
    # Program IDs
    pid = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    
    # Calculate number of blocks in M and N dimensions
    num_pid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_pid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Re-order program IDs for better L2 cache locality
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_m_in_group // num_pid_n
    pid_n = pid_m_in_group % num_pid_n
    
    # Create offsets for the blocks
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of A and B
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + (offs_k[None, :] + k) * stride_ak,
                   mask=offs_am[:, None] < M and (offs_k[None, :] + k) < K, other=0.0)
        b = tl.load(b_ptr + (offs_k[:, None] + k) * stride_bk + offs_bn[None, :] * stride_bn,
                   mask=(offs_k[:, None] + k) < K and offs_bn[None, :] < N, other=0.0)
        
        # Matrix multiplication
        accumulator += tl.dot(a, b)
    
    # Store result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c = accumulator.to(tl.float16)
    
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn,
             c, mask=offs_cm[:, None] < M and offs_cn[None, :] < N)


def matmul(a: torch.Tensor, b: torch.Tensor):
    """
    High-level wrapper for matrix multiplication using Triton
    
    Args:
        a: Input tensor A (M x K)
        b: Input tensor B (K x N)
        
    Returns:
        c: Output tensor C (M x N) = A @ B
    """
    # Input validation
    assert a.dim() == 2 and b.dim() == 2, "Input tensors must be 2D"
    assert a.shape[1] == b.shape[0], f"Matrix dimensions incompatible: {a.shape} @ {b.shape}"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on CUDA device"
    
    M, K = a.shape
    _, N = b.shape
    
    # Create output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Launch configuration
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # Calculate grid dimensions
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid = (GROUP_SIZE_M * grid_m * grid_n, grid_m, grid_n)
    
    # Launch kernel
    matmul_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )
    
    return c


def matmul_with_bias(a: torch.Tensor, b: torch.Tensor, bias: torch.Tensor):
    """
    Matrix multiplication with bias addition: C = A @ B + bias
    
    Args:
        a: Input tensor A (M x K)
        b: Input tensor B (K x N)
        bias: Bias tensor (N,)
        
    Returns:
        c: Output tensor C (M x N) = A @ B + bias
    """
    # Perform matrix multiplication
    c = matmul(a, b)
    
    # Add bias
    c += bias.unsqueeze(0)
    
    return c


def batched_matmul(a: torch.Tensor, b: torch.Tensor):
    """
    Batched matrix multiplication for 3D tensors
    
    Args:
        a: Input tensor A (B x M x K)
        b: Input tensor B (B x K x N)
        
    Returns:
        c: Output tensor C (B x M x N) = A @ B
    """
    assert a.dim() == 3 and b.dim() == 3, "Input tensors must be 3D"
    assert a.shape[0] == b.shape[0], "Batch dimensions must match"
    assert a.shape[2] == b.shape[1], f"Matrix dimensions incompatible: {a.shape} @ {b.shape}"
    
    B, M, K = a.shape
    _, K_b, N = b.shape
    assert K == K_b, f"Inner dimensions must match: {K} != {K_b}"
    
    # Process each batch separately
    results = []
    for i in range(B):
        a_batch = a[i]
        b_batch = b[i]
        c_batch = matmul(a_batch, b_batch)
        results.append(c_batch)
    
    # Stack results
    c = torch.stack(results, dim=0)
    
    return c


def benchmark_matmul(M: int = 1024, N: int = 1024, K: int = 1024, warmup: int = 10, repeat: int = 100):
    """
    Benchmark matrix multiplication implementation
    
    Args:
        M, N, K: Matrix dimensions
        warmup: Number of warmup iterations
        repeat: Number of benchmark iterations
        
    Returns:
        dict: Benchmark results
    """
    import time
    
    # Create test data
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    # Warmup
    for _ in range(warmup):
        _ = matmul(a, b)
        torch.cuda.synchronize()
    
    # Benchmark Triton implementation
    start_time = time.time()
    for _ in range(repeat):
        _ = matmul(a, b)
        torch.cuda.synchronize()
    end_time = time.time()
    
    triton_time = (end_time - start_time) / repeat
    
    # Benchmark PyTorch implementation
    start_time = time.time()
    for _ in range(repeat):
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
    end_time = time.time()
    
    torch_time = (end_time - start_time) / repeat
    
    # Calculate theoretical FLOPs
    total_flops = 2 * M * N * K  # Multiply-add operations
    
    return {
        'triton_time': triton_time,
        'torch_time': torch_time,
        'speedup': torch_time / triton_time,
        'triton_tflops': total_flops / (triton_time * 1e12),
        'torch_tflops': total_flops / (torch_time * 1e12),
        'matrix_shape': (M, N, K)
    }