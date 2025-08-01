"""
Triton基础算子：向量加法

这是最简单的Triton算子实现，用于理解Triton的基本编程模型：
- kernel函数定义
- grid计算
- 内存访问模式
- 指针运算
"""

import triton
import triton.language as tl
import torch
import time
from typing import Optional


@triton.jit
def vector_add_kernel(
    x_ptr,  # 输入向量x的指针
    y_ptr,  # 输入向量y的指针
    output_ptr,  # 输出向量的指针
    n_elements,  # 向量长度
    BLOCK_SIZE: tl.constexpr,  # 每个block处理的元素数量
):
    """
    向量加法kernel函数
    
    参数:
        x_ptr, y_ptr: 输入向量的设备指针
        output_ptr: 输出向量的设备指针
        n_elements: 向量总长度
        BLOCK_SIZE: 每个block处理的元素数量 (编译时常量)
    """
    # 1. 计算当前线程的程序ID (pid)
    # tl.program_id(axis) 获取当前block在指定axis上的ID
    # 这里我们使用1D grid，所以axis=0
    pid = tl.program_id(0)
    
    # 2. 计算当前block处理的起始位置
    block_start = pid * BLOCK_SIZE
    
    # 3. 计算当前线程处理的元素偏移量
    # tl.arange(0, BLOCK_SIZE) 生成一个[0, 1, 2, ..., BLOCK_SIZE-1]的张量
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 4. 创建掩码，处理边界情况
    # 当向量长度不是BLOCK_SIZE的整数倍时，最后一个block可能不完整
    mask = offsets < n_elements
    
    # 5. 从全局内存加载数据
    # tl.load(pointer, mask) 从全局内存加载数据
    # mask确保我们不会越界访问
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 6. 执行向量加法
    output = x + y
    
    # 7. 将结果写回全局内存
    tl.store(output_ptr + offsets, output, mask=mask)


def vector_add(x: torch.Tensor, y: torch.Tensor, 
               output: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    向量加法的host函数
    
    参数:
        x: 输入张量1
        y: 输入张量2  
        output: 可选的输出张量
    
    返回:
        x + y的结果
    """
    # 输入验证
    assert x.shape == y.shape, "Input tensors must have the same shape"
    assert x.is_cuda and y.is_cuda, "Input tensors must be on GPU"
    
    # 准备输出张量
    if output is None:
        output = torch.empty_like(x)
    else:
        assert output.shape == x.shape, "Output tensor must have the same shape"
    
    # 计算向量长度
    n_elements = x.numel()
    
    # 设置block size (经验值：通常是warp size的倍数)
    BLOCK_SIZE = 1024
    
    # 计算grid size (需要的block数量)
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # 启动kernel
    vector_add_kernel[grid_size](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def benchmark_vector_add(size: int = 1000000, n_warmup: int = 10, n_repeat: int = 100):
    """
    性能基准测试
    
    参数:
        size: 向量长度
        n_warmup: 预热次数
        n_repeat: 测试次数
    """
    print(f"Benchmarking vector addition with size: {size:,}")
    
    # 创建测试数据
    x = torch.randn(size, device='cuda', dtype=torch.float32)
    y = torch.randn(size, device='cuda', dtype=torch.float32)
    
    # 预热
    for _ in range(n_warmup):
        _ = vector_add(x, y)
        torch.cuda.synchronize()
    
    # 测试Triton实现
    triton_times = []
    for _ in range(n_repeat):
        start_time = time.time()
        _ = vector_add(x, y)
        torch.cuda.synchronize()
        triton_times.append(time.time() - start_time)
    
    # 测试PyTorch实现
    torch_times = []
    for _ in range(n_repeat):
        start_time = time.time()
        _ = x + y
        torch.cuda.synchronize()
        torch_times.append(time.time() - start_time)
    
    # 计算统计信息
    triton_mean = sum(triton_times) / len(triton_times)
    torch_mean = sum(torch_times) / len(torch_times)
    
    triton_std = (sum((t - triton_mean) ** 2 for t in triton_times) / len(triton_times)) ** 0.5
    torch_std = (sum((t - torch_mean) ** 2 for t in torch_times) / len(torch_times)) ** 0.5
    
    # 计算GFLOPS
    flops = size  # 向量加法需要size次浮点运算
    triton_gflops = flops / (triton_mean * 1e9)
    torch_gflops = flops / (torch_mean * 1e9)
    
    print(f"\nResults:")
    print(f"Triton:  {triton_mean*1000:.2f} ± {triton_std*1000:.2f} ms")
    print(f"PyTorch: {torch_mean*1000:.2f} ± {torch_std*1000:.2f} ms")
    print(f"Speedup: {torch_mean/triton_mean:.2f}x")
    print(f"Triton GFLOPS:  {triton_gflops:.2f}")
    print(f"PyTorch GFLOPS: {torch_gflops:.2f}")
    
    # 验证结果正确性
    triton_result = vector_add(x, y)
    torch_result = x + y
    max_diff = torch.max(torch.abs(triton_result - torch_result)).item()
    print(f"Max difference: {max_diff:.2e}")
    
    return {
        'triton_time': triton_mean,
        'torch_time': torch_mean,
        'speedup': torch_mean / triton_mean,
        'max_diff': max_diff
    }


def test_vector_add():
    """测试向量加法的正确性"""
    print("Testing vector addition...")
    
    # 测试不同大小
    test_sizes = [100, 1000, 10000, 100000]
    
    for size in test_sizes:
        print(f"\nTesting size: {size}")
        
        # 创建测试数据
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        y = torch.randn(size, device='cuda', dtype=torch.float32)
        
        # Triton实现
        triton_result = vector_add(x, y)
        
        # PyTorch实现
        torch_result = x + y
        
        # 验证结果
        max_diff = torch.max(torch.abs(triton_result - torch_result)).item()
        mean_diff = torch.mean(torch.abs(triton_result - torch_result)).item()
        
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        
        # 断言正确性
        assert max_diff < 1e-5, f"Test failed for size {size}"
    
    print("✓ All tests passed!")


if __name__ == "__main__":
    # 运行测试
    test_vector_add()
    
    # 运行基准测试
    print("\n" + "="*50)
    benchmark_vector_add(size=1000000)
    
    # 测试不同大小的性能
    print("\n" + "="*50)
    print("Performance scaling test:")
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    for size in sizes:
        result = benchmark_vector_add(size=size, n_warmup=5, n_repeat=20)
        print(f"Size {size:,}: Speedup = {result['speedup']:.2f}x")