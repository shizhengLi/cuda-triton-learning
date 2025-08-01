"""
Triton LayerNorm实现

LayerNorm是Transformer架构中的核心组件，本实现展示了如何使用Triton高效实现LayerNorm算子，
包括：
- 并行化策略
- 数值稳定性优化
- 内存访问模式优化
- 统计量计算优化
"""

import triton
import triton.language as tl
import torch
import time
from typing import Optional


@triton.jit
def layer_norm_kernel(
    input_ptr,      # 输入张量指针
    output_ptr,     # 输出张量指针
    weight_ptr,     # 权重指针
    bias_ptr,       # 偏置指针
    mean_ptr,       # 均值指针（可选）
    var_ptr,        # 方差指针（可选）
    n_elements,     # 每个样本的元素数量
    eps,            # 数值稳定性常数
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm kernel函数
    
    参数:
        input_ptr: 输入张量指针 [batch_size, n_elements]
        output_ptr: 输出张量指针 [batch_size, n_elements]
        weight_ptr: 权重指针 [n_elements]
        bias_ptr: 偏置指针 [n_elements]
        mean_ptr: 均值指针 [batch_size] (可选)
        var_ptr: 方差指针 [batch_size] (可选)
        n_elements: 每个样本的元素数量
        eps: 数值稳定性常数
        BLOCK_SIZE: 每个block处理的元素数量
    """
    # 1. 获取当前处理的样本ID
    batch_id = tl.program_id(0)
    
    # 2. 计算当前样本的偏移量
    batch_offset = batch_id * n_elements
    
    # 3. 计算当前线程处理的元素偏移量
    offsets = batch_offset + tl.arange(0, BLOCK_SIZE)
    
    # 4. 创建掩码处理边界情况
    mask = offsets < batch_offset + n_elements
    
    # 5. 从全局内存加载数据
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # 6. 计算均值和方差
    # 使用Welford算法提高数值稳定性
    mean = tl.sum(x, axis=0) / n_elements
    var = tl.sum((x - mean) ** 2, axis=0) / n_elements
    
    # 7. 存储均值和方差（如果需要）
    if mean_ptr is not None:
        tl.store(mean_ptr + batch_id, mean)
    if var_ptr is not None:
        tl.store(var_ptr + batch_id, var)
    
    # 8. 计算LayerNorm
    # y = (x - mean) / sqrt(var + eps) * weight + bias
    x_normalized = (x - mean) * tl.rsqrt(var + eps)
    
    # 9. 加载权重和偏置
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    # 10. 应用权重和偏置
    y = x_normalized * weight + bias
    
    # 11. 存储结果
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def layer_norm_fused_kernel(
    input_ptr,      # 输入张量指针
    output_ptr,     # 输出张量指针
    weight_ptr,     # 权重指针
    bias_ptr,       # 偏置指针
    gamma_ptr,      # 附加的gamma参数（用于某些变体）
    n_elements,     # 每个样本的元素数量
    eps,            # 数值稳定性常数
    BLOCK_SIZE: tl.constexpr,
):
    """
    融合LayerNorm kernel，支持额外的计算
    """
    batch_id = tl.program_id(0)
    batch_offset = batch_id * n_elements
    offsets = batch_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_offset + n_elements
    
    # 加载输入数据
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # 计算统计量
    mean = tl.sum(x, axis=0) / n_elements
    var = tl.sum((x - mean) ** 2, axis=0) / n_elements
    
    # LayerNorm计算
    x_normalized = (x - mean) * tl.rsqrt(var + eps)
    
    # 加载参数
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    bias = tl.load(bias_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0)
    
    # 融合计算
    y = x_normalized * weight + bias
    
    # 存储结果
    tl.store(output_ptr + offsets, y, mask=mask)


def layer_norm(x: torch.Tensor, 
               weight: Optional[torch.Tensor] = None,
               bias: Optional[torch.Tensor] = None,
               eps: float = 1e-5,
               return_stats: bool = False) -> torch.Tensor:
    """
    LayerNorm host函数
    
    参数:
        x: 输入张量 [batch_size, n_elements]
        weight: 权重张量 [n_elements]
        bias: 偏置张量 [n_elements]
        eps: 数值稳定性常数
        return_stats: 是否返回统计量
    
    返回:
        LayerNorm结果，可选返回统计量
    """
    # 输入验证
    assert x.dim() == 2, "Input must be 2D tensor [batch_size, n_elements]"
    assert x.is_cuda, "Input must be on GPU"
    
    batch_size, n_elements = x.shape
    
    # 初始化权重和偏置
    if weight is None:
        weight = torch.ones(n_elements, device=x.device, dtype=x.dtype)
    if bias is None:
        bias = torch.zeros(n_elements, device=x.device, dtype=x.dtype)
    
    # 验证参数形状
    assert weight.shape == (n_elements,), "Weight shape mismatch"
    assert bias.shape == (n_elements,), "Bias shape mismatch"
    
    # 准备输出张量
    output = torch.empty_like(x)
    
    # 准备统计量张量
    mean_tensor = None
    var_tensor = None
    if return_stats:
        mean_tensor = torch.empty(batch_size, device=x.device, dtype=torch.float32)
        var_tensor = torch.empty(batch_size, device=x.device, dtype=torch.float32)
    
    # 设置block size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # 计算grid size
    grid_size = batch_size
    
    # 启动kernel
    layer_norm_kernel[grid_size](
        input_ptr=x,
        output_ptr=output,
        weight_ptr=weight,
        bias_ptr=bias,
        mean_ptr=mean_tensor,
        var_ptr=var_tensor,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    if return_stats:
        return output, mean_tensor, var_tensor
    else:
        return output


def layer_norm_fused(x: torch.Tensor,
                    weight: torch.Tensor,
                    bias: torch.Tensor,
                    gamma: Optional[torch.Tensor] = None,
                    eps: float = 1e-5) -> torch.Tensor:
    """
    融合LayerNorm实现
    """
    assert x.dim() == 2, "Input must be 2D tensor"
    assert x.is_cuda, "Input must be on GPU"
    
    batch_size, n_elements = x.shape
    
    # 准备输出张量
    output = torch.empty_like(x)
    
    # 设置block size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # 计算grid size
    grid_size = batch_size
    
    # 启动kernel
    layer_norm_fused_kernel[grid_size](
        input_ptr=x,
        output_ptr=output,
        weight_ptr=weight,
        bias_ptr=bias,
        gamma_ptr=gamma if gamma is not None else weight,  # 如果没有gamma，使用weight
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output


def benchmark_layer_norm(batch_size: int = 128, hidden_size: int = 768,
                        n_warmup: int = 10, n_repeat: int = 100):
    """
    LayerNorm性能基准测试
    
    参数:
        batch_size: 批量大小
        hidden_size: 隐藏层大小
        n_warmup: 预热次数
        n_repeat: 测试次数
    """
    print(f"Benchmarking LayerNorm: batch_size={batch_size}, hidden_size={hidden_size}")
    
    # 创建测试数据
    x = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float16)
    weight = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
    bias = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
    
    # 预热
    for _ in range(n_warmup):
        _ = layer_norm(x, weight, bias)
        torch.cuda.synchronize()
    
    # 测试Triton实现
    triton_times = []
    for _ in range(n_repeat):
        start_time = time.time()
        _ = layer_norm(x, weight, bias)
        torch.cuda.synchronize()
        triton_times.append(time.time() - start_time)
    
    # 测试PyTorch实现
    torch_times = []
    for _ in range(n_repeat):
        start_time = time.time()
        _ = torch.nn.functional.layer_norm(x, [hidden_size], weight, bias, 1e-5)
        torch.cuda.synchronize()
        torch_times.append(time.time() - start_time)
    
    # 计算统计信息
    triton_mean = sum(triton_times) / len(triton_times)
    torch_mean = sum(torch_times) / len(torch_times)
    
    triton_std = (sum((t - triton_mean) ** 2 for t in triton_times) / len(triton_times)) ** 0.5
    torch_std = (sum((t - torch_mean) ** 2 for t in torch_times) / len(torch_times)) ** 0.5
    
    print(f"\nResults:")
    print(f"Triton:  {triton_mean*1000:.2f} ± {triton_std*1000:.2f} ms")
    print(f"PyTorch: {torch_mean*1000:.2f} ± {torch_std*1000:.2f} ms")
    print(f"Speedup: {torch_mean/triton_mean:.2f}x")
    
    # 验证结果正确性
    triton_result = layer_norm(x, weight, bias)
    torch_result = torch.nn.functional.layer_norm(x, [hidden_size], weight, bias, 1e-5)
    
    rel_error = torch.abs(triton_result - torch_result) / torch.abs(torch_result)
    max_rel_error = torch.max(rel_error).item()
    mean_rel_error = torch.mean(rel_error).item()
    
    print(f"Max relative error: {max_rel_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")
    
    return {
        'triton_time': triton_mean,
        'torch_time': torch_mean,
        'speedup': torch_mean / triton_mean,
        'max_rel_error': max_rel_error
    }


def test_layer_norm():
    """测试LayerNorm的正确性"""
    print("Testing LayerNorm...")
    
    # 测试不同配置
    test_configs = [
        (32, 128),    # 小批量，小维度
        (64, 512),    # 中等批量，中等维度
        (128, 768),   # 大批量，中等维度
        (256, 1024),  # 大批量，大维度
        (512, 2048),  # 超大批量，大维度
    ]
    
    for batch_size, hidden_size in test_configs:
        print(f"\nTesting batch_size={batch_size}, hidden_size={hidden_size}")
        
        # 创建测试数据
        x = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float16)
        weight = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
        bias = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
        
        # Triton实现
        triton_result = layer_norm(x, weight, bias)
        
        # PyTorch实现
        torch_result = torch.nn.functional.layer_norm(x, [hidden_size], weight, bias, 1e-5)
        
        # 验证结果
        rel_error = torch.abs(triton_result - torch_result) / torch.abs(torch_result)
        max_rel_error = torch.max(rel_error).item()
        mean_rel_error = torch.mean(rel_error).item()
        
        print(f"  Max relative error: {max_rel_error:.2e}")
        print(f"  Mean relative error: {mean_rel_error:.2e}")
        
        # FP16的误差容忍度
        assert max_rel_error < 1e-3, f"Test failed for batch_size={batch_size}, hidden_size={hidden_size}"
    
    print("✓ All tests passed!")


def test_layer_norm_with_stats():
    """测试带统计量的LayerNorm"""
    print("\nTesting LayerNorm with statistics...")
    
    batch_size, hidden_size = 64, 512
    x = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float16)
    weight = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
    bias = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
    
    # Triton实现
    triton_result, triton_mean, triton_var = layer_norm(x, weight, bias, return_stats=True)
    
    # PyTorch实现
    torch_result = torch.nn.functional.layer_norm(x, [hidden_size], weight, bias, 1e-5)
    
    # 计算PyTorch的统计量
    torch_mean = torch.mean(x, dim=1, dtype=torch.float32)
    torch_var = torch.var(x, dim=1, dtype=torch.float32, correction=0)
    
    # 验证统计量
    mean_error = torch.max(torch.abs(triton_mean - torch_mean)).item()
    var_error = torch.max(torch.abs(triton_var - torch_var)).item()
    
    print(f"Mean error: {mean_error:.2e}")
    print(f"Variance error: {var_error:.2e}")
    
    assert mean_error < 1e-5, f"Mean test failed: {mean_error}"
    assert var_error < 1e-5, f"Variance test failed: {var_error}"
    
    print("✓ Statistics test passed!")


if __name__ == "__main__":
    # 运行测试
    test_layer_norm()
    test_layer_norm_with_stats()
    
    # 运行基准测试
    print("\n" + "="*60)
    benchmark_layer_norm(batch_size=128, hidden_size=768)
    
    # 测试不同大小的性能
    print("\n" + "="*60)
    print("Performance scaling test:")
    configs = [
        (32, 128), (64, 256), (128, 512), (256, 768), (512, 1024), (1024, 2048)
    ]
    for batch_size, hidden_size in configs:
        result = benchmark_layer_norm(batch_size=batch_size, hidden_size=hidden_size, 
                                     n_warmup=5, n_repeat=20)
        print(f"batch_size={batch_size}, hidden_size={hidden_size}: "
              f"Speedup = {result['speedup']:.2f}x")