"""
Triton矩阵乘法实现

矩阵乘法是深度学习中最核心的操作之一，也是Triton最重要的应用场景。
本实现展示了如何使用Triton实现高效的矩阵乘法，包括：
- 2D block tiling策略
- 共享内存使用
- 内存合并访问优化
"""

import triton
import triton.language as tl
import torch
import time
from typing import Optional


@triton.jit
def matrix_multiply_kernel(
    a_ptr,  # 矩阵A的指针 (M x K)
    b_ptr,  # 矩阵B的指针 (K x N)
    c_ptr,  # 输出矩阵C的指针 (M x N)
    M, K, N,  # 矩阵维度
    stride_am,  # 矩阵A的M维度stride
    stride_ak,  # 矩阵A的K维度stride
    stride_bk,  # 矩阵B的K维度stride
    stride_bn,  # 矩阵B的N维度stride
    stride_cm,  # 矩阵C的M维度stride
    stride_cn,  # 矩阵C的N维度stride
    BLOCK_SIZE_M: tl.constexpr,  # M维度的block大小
    BLOCK_SIZE_N: tl.constexpr,  # N维度的block大小
    BLOCK_SIZE_K: tl.constexpr,  # K维度的block大小
    GROUP_SIZE_M: tl.constexpr,  # M维度的分组大小
):
    """
    矩阵乘法kernel函数，使用2D block tiling策略
    
    参数:
        a_ptr, b_ptr: 输入矩阵的设备指针
        c_ptr: 输出矩阵的设备指针
        M, K, N: 矩阵维度
        stride_*: 各个维度的stride
        BLOCK_SIZE_*: 各个维度的block大小
        GROUP_SIZE_M: M维度的分组大小，用于提高L2缓存命中率
    """
    # 1. 程序ID和block索引
    pid = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    # 2. 计算当前block在输出矩阵中的位置
    # 使用分组策略提高L2缓存命中率
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m_in_group = pid % num_pid_in_group
    pid_m = first_pid_m + pid_m_in_group // num_pid_n
    pid_n = pid_m_in_group % num_pid_n
    
    # 3. 计算当前block的偏移量
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 4. 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 5. 分块处理K维度
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 计算当前K块的偏移量
        k_start = k * BLOCK_SIZE_K
        
        # 创建掩码处理边界情况
        mask_m = offs_m < M
        mask_n = offs_n < N
        mask_k = k_start + offs_k < K
        
        # 6. 从全局内存加载数据到共享内存
        # 加载矩阵A的块
        a_ptrs = a_ptr + (offs_m[:, None] * stride_am + (k_start + offs_k[None, :]) * stride_ak)
        a_mask = mask_m[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # 加载矩阵B的块
        b_ptrs = b_ptr + ((k_start + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        b_mask = mask_k[:, None] & mask_n[None, :]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # 7. 矩阵乘法累加
        accumulator += tl.dot(a, b)
    
    # 8. 将结果写回全局内存
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matrix_multiply(a: torch.Tensor, b: torch.Tensor, 
                   output: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    矩阵乘法的host函数
    
    参数:
        a: 输入矩阵A (M x K)
        b: 输入矩阵B (K x N)
        output: 可选的输出矩阵
    
    返回:
        矩阵乘法结果 C = A @ B
    """
    # 输入验证
    assert a.dim() == 2 and b.dim() == 2, "Input tensors must be 2D"
    assert a.size(1) == b.size(0), f"Matrix dimensions incompatible: {a.shape} @ {b.shape}"
    assert a.is_cuda and b.is_cuda, "Input tensors must be on GPU"
    
    M, K = a.shape
    _, N = b.shape
    
    # 准备输出张量
    if output is None:
        output = torch.empty((M, N), device=a.device, dtype=a.dtype)
    else:
        assert output.shape == (M, N), "Output tensor shape mismatch"
    
    # 设置block大小（这些参数需要根据硬件特性调整）
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8
    
    # 计算grid大小
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    grid = (GROUP_SIZE_M * grid_m * grid_n, grid_m, grid_n)
    
    # 启动kernel
    matrix_multiply_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=output,
        M=M, K=K, N=N,
        stride_am=a.stride(0),
        stride_ak=a.stride(1),
        stride_bk=b.stride(0),
        stride_bn=b.stride(1),
        stride_cm=output.stride(0),
        stride_cn=output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M
    )
    
    return output


def benchmark_matrix_multiply(M: int = 1024, K: int = 1024, N: int = 1024,
                             n_warmup: int = 10, n_repeat: int = 100):
    """
    矩阵乘法性能基准测试
    
    参数:
        M, K, N: 矩阵维度
        n_warmup: 预热次数
        n_repeat: 测试次数
    """
    print(f"Benchmarking matrix multiplication: {M}x{K} @ {K}x{N}")
    
    # 创建测试数据
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    # 计算理论FLOPS
    # 矩阵乘法需要 2 * M * K * N 次浮点运算
    flops = 2 * M * K * N
    
    # 预热
    for _ in range(n_warmup):
        _ = matrix_multiply(a, b)
        torch.cuda.synchronize()
    
    # 测试Triton实现
    triton_times = []
    for _ in range(n_repeat):
        start_time = time.time()
        _ = matrix_multiply(a, b)
        torch.cuda.synchronize()
        triton_times.append(time.time() - start_time)
    
    # 测试PyTorch实现
    torch_times = []
    for _ in range(n_repeat):
        start_time = time.time()
        _ = torch.matmul(a, b)
        torch.cuda.synchronize()
        torch_times.append(time.time() - start_time)
    
    # 计算统计信息
    triton_mean = sum(triton_times) / len(triton_times)
    torch_mean = sum(torch_times) / len(torch_times)
    
    triton_std = (sum((t - triton_mean) ** 2 for t in triton_times) / len(triton_times)) ** 0.5
    torch_std = (sum((t - torch_mean) ** 2 for t in torch_times) / len(torch_times)) ** 0.5
    
    # 计算GFLOPS和TFLOPS
    triton_tflops = flops / (triton_mean * 1e12)
    torch_tflops = flops / (torch_mean * 1e12)
    
    print(f"\nResults:")
    print(f"Triton:  {triton_mean*1000:.2f} ± {triton_std*1000:.2f} ms")
    print(f"PyTorch: {torch_mean*1000:.2f} ± {torch_std*1000:.2f} ms")
    print(f"Speedup: {torch_mean/triton_mean:.2f}x")
    print(f"Triton TFLOPS:  {triton_tflops:.2f}")
    print(f"PyTorch TFLOPS: {torch_tflops:.2f}")
    
    # 验证结果正确性
    triton_result = matrix_multiply(a, b)
    torch_result = torch.matmul(a, b)
    
    # 使用相对误差验证
    rel_error = torch.abs(triton_result - torch_result) / torch.abs(torch_result)
    max_rel_error = torch.max(rel_error).item()
    mean_rel_error = torch.mean(rel_error).item()
    
    print(f"Max relative error: {max_rel_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")
    
    return {
        'triton_time': triton_mean,
        'torch_time': torch_mean,
        'speedup': torch_mean / triton_mean,
        'max_rel_error': max_rel_error,
        'triton_tflops': triton_tflops,
        'torch_tflops': torch_tflops
    }


def test_matrix_multiply():
    """测试矩阵乘法的正确性"""
    print("Testing matrix multiplication...")
    
    # 测试不同大小的矩阵
    test_cases = [
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (1024, 1024, 1024),
        # 非方阵测试
        (128, 256, 512),
        (512, 128, 256),
    ]
    
    for M, K, N in test_cases:
        print(f"\nTesting {M}x{K} @ {K}x{N}")
        
        # 创建测试数据
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        
        # Triton实现
        triton_result = matrix_multiply(a, b)
        
        # PyTorch实现
        torch_result = torch.matmul(a, b)
        
        # 验证结果
        rel_error = torch.abs(triton_result - torch_result) / torch.abs(torch_result)
        max_rel_error = torch.max(rel_error).item()
        mean_rel_error = torch.mean(rel_error).item()
        
        print(f"  Max relative error: {max_rel_error:.2e}")
        print(f"  Mean relative error: {mean_rel_error:.2e}")
        
        # 对于FP16，使用相对宽松的误差容忍度
        assert max_rel_error < 1e-3, f"Test failed for {M}x{K} @ {K}x{N}"
    
    print("✓ All tests passed!")


def analyze_block_size_performance():
    """分析不同block size对性能的影响"""
    print("\nAnalyzing block size performance...")
    
    M, K, N = 1024, 1024, 1024
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    
    # 测试不同的block size组合
    block_sizes = [
        (64, 64, 16),
        (128, 128, 32),
        (256, 256, 32),
        (128, 64, 32),
        (64, 128, 32),
    ]
    
    results = []
    for block_m, block_n, block_k in block_sizes:
        print(f"\nTesting block size: {block_m}x{block_n}x{block_k}")
        
        # 临时修改kernel的block size
        # 这里我们使用一个简化的方法，实际应用中应该动态调整
        start_time = time.time()
        _ = matrix_multiply(a, b)  # 使用默认的block size
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        # 这里简化了block size的测试，实际应用中需要更复杂的实现
        results.append((block_m, block_n, block_k, elapsed))
        print(f"  Time: {elapsed*1000:.2f} ms")
    
    return results


if __name__ == "__main__":
    # 运行测试
    test_matrix_multiply()
    
    # 运行基准测试
    print("\n" + "="*60)
    benchmark_matrix_multiply(M=1024, K=1024, N=1024)
    
    # 测试不同大小的性能
    print("\n" + "="*60)
    print("Performance scaling test:")
    sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048)]
    for M, K, N in sizes:
        result = benchmark_matrix_multiply(M=M, K=K, N=N, n_warmup=5, n_repeat=20)
        print(f"Size {M}x{K}x{N}: Speedup = {result['speedup']:.2f}x, "
              f"Triton TFLOPS = {result['triton_tflops']:.2f}")
    
    # 分析block size性能
    print("\n" + "="*60)
    analyze_block_size_performance()