"""
Triton RMSNorm实现

RMSNorm是LayerNorm的一个简化变体，在LLaMA等模型中被广泛使用。
相比LayerNorm，RMSNorm去掉了均值中心化步骤，只进行方差的归一化。
"""

import triton
import triton.language as tl
import torch
import time
from typing import Optional


@triton.jit
def rms_norm_kernel(
    input_ptr,      # 输入张量指针
    output_ptr,     # 输出张量指针
    weight_ptr,     # 权重指针
    rms_ptr,        # RMS值指针（可选）
    n_elements,     # 每个样本的元素数量
    eps,            # 数值稳定性常数
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm kernel函数
    
    RMSNorm公式: y = x / sqrt(mean(x²) + eps) * weight
    
    参数:
        input_ptr: 输入张量指针 [batch_size, n_elements]
        output_ptr: 输出张量指针 [batch_size, n_elements]
        weight_ptr: 权重指针 [n_elements]
        rms_ptr: RMS值指针 [batch_size] (可选)
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
    
    # 6. 计算RMS (Root Mean Square)
    # RMS = sqrt(mean(x²))
    x_squared = x * x
    mean_squared = tl.sum(x_squared, axis=0) / n_elements
    rms = tl.sqrt(mean_squared + eps)
    
    # 7. 存储RMS值（如果需要）
    if rms_ptr is not None:
        tl.store(rms_ptr + batch_id, rms)
    
    # 8. 计算RMSNorm
    # y = x / RMS * weight
    x_normalized = x / rms
    
    # 9. 加载权重
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    
    # 10. 应用权重
    y = x_normalized * weight
    
    # 11. 存储结果
    tl.store(output_ptr + offsets, y, mask=mask)


@triton.jit
def rms_norm_fused_backward_kernel(
    input_ptr,      # 输入张量指针
    output_grad_ptr, # 输出梯度指针
    weight_ptr,     # 权重指针
    input_grad_ptr, # 输入梯度指针
    weight_grad_ptr,# 权重梯度指针
    rms_ptr,        # RMS值指针
    n_elements,     # 每个样本的元素数量
    eps,            # 数值稳定性常数
    BLOCK_SIZE: tl.constexpr,
):
    """
    RMSNorm反向传播kernel
    
    这个kernel实现了RMSNorm的梯度计算，包括对输入和权重的梯度。
    """
    batch_id = tl.program_id(0)
    batch_offset = batch_id * n_elements
    offsets = batch_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_offset + n_elements
    
    # 加载数据
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    output_grad = tl.load(output_grad_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask, other=1.0)
    rms = tl.load(rms_ptr + batch_id)
    
    # 计算输入梯度
    # dL/dx = (dL/dy * w) * (1/rms - x²/rms³)
    rms_inv = 1.0 / rms
    rms_inv_cubed = rms_inv * rms_inv * rms_inv
    
    grad_y_w = output_grad * weight
    sum_grad_y_w_x = tl.sum(grad_y_w * x, axis=0) / n_elements
    
    input_grad = grad_y_w * (rms_inv - x * rms_inv_cubed * sum_grad_y_w_x)
    
    # 计算权重梯度
    # dL/dw = dL/dy * (x / rms)
    weight_grad = output_grad * (x * rms_inv)
    
    # 存储梯度
    tl.store(input_grad_ptr + offsets, input_grad, mask=mask)
    tl.store(weight_grad_ptr + tl.arange(0, BLOCK_SIZE), weight_grad, mask=mask)


def rms_norm(x: torch.Tensor, 
            weight: Optional[torch.Tensor] = None,
            eps: float = 1e-6,
            return_rms: bool = False) -> torch.Tensor:
    """
    RMSNorm host函数
    
    参数:
        x: 输入张量 [batch_size, n_elements]
        weight: 权重张量 [n_elements]
        eps: 数值稳定性常数
        return_rms: 是否返回RMS值
    
    返回:
        RMSNorm结果，可选返回RMS值
    """
    # 输入验证
    assert x.dim() == 2, "Input must be 2D tensor [batch_size, n_elements]"
    assert x.is_cuda, "Input must be on GPU"
    
    batch_size, n_elements = x.shape
    
    # 初始化权重
    if weight is None:
        weight = torch.ones(n_elements, device=x.device, dtype=x.dtype)
    
    # 验证参数形状
    assert weight.shape == (n_elements,), "Weight shape mismatch"
    
    # 准备输出张量
    output = torch.empty_like(x)
    
    # 准备RMS张量
    rms_tensor = None
    if return_rms:
        rms_tensor = torch.empty(batch_size, device=x.device, dtype=torch.float32)
    
    # 设置block size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # 计算grid size
    grid_size = batch_size
    
    # 启动kernel
    rms_norm_kernel[grid_size](
        input_ptr=x,
        output_ptr=output,
        weight_ptr=weight,
        rms_ptr=rms_tensor,
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    if return_rms:
        return output, rms_tensor
    else:
        return output


def rms_norm_backward(x: torch.Tensor, 
                     output_grad: torch.Tensor,
                     weight: torch.Tensor,
                     rms: torch.Tensor) -> tuple:
    """
    RMSNorm反向传播
    
    参数:
        x: 原始输入
        output_grad: 输出梯度
        weight: 权重
        rms: RMS值
    
    返回:
        input_grad, weight_grad
    """
    batch_size, n_elements = x.shape
    
    # 准备梯度张量
    input_grad = torch.empty_like(x)
    weight_grad = torch.zeros_like(weight)
    
    # 设置block size
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
    
    # 计算grid size
    grid_size = batch_size
    
    # 启动反向传播kernel
    rms_norm_fused_backward_kernel[grid_size](
        input_ptr=x,
        output_grad_ptr=output_grad,
        weight_ptr=weight,
        input_grad_ptr=input_grad,
        weight_grad_ptr=weight_grad,
        rms_ptr=rms,
        n_elements=n_elements,
        eps=1e-6,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return input_grad, weight_grad


def benchmark_rms_norm(batch_size: int = 128, hidden_size: int = 768,
                      n_warmup: int = 10, n_repeat: int = 100):
    """
    RMSNorm性能基准测试
    
    参数:
        batch_size: 批量大小
        hidden_size: 隐藏层大小
        n_warmup: 预热次数
        n_repeat: 测试次数
    """
    print(f"Benchmarking RMSNorm: batch_size={batch_size}, hidden_size={hidden_size}")
    
    # 创建测试数据
    x = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float16)
    weight = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
    
    # 预热
    for _ in range(n_warmup):
        _ = rms_norm(x, weight)
        torch.cuda.synchronize()
    
    # 测试Triton实现
    triton_times = []
    for _ in range(n_repeat):
        start_time = time.time()
        _ = rms_norm(x, weight)
        torch.cuda.synchronize()
        triton_times.append(time.time() - start_time)
    
    # 测试PyTorch实现（手动实现）
    torch_times = []
    for _ in range(n_repeat):
        start_time = time.time()
        # PyTorch RMSNorm实现
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-6)
        output = x / rms * weight
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
    triton_result = rms_norm(x, weight)
    rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-6)
    torch_result = x / rms * weight
    
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


def test_rms_norm():
    """测试RMSNorm的正确性"""
    print("Testing RMSNorm...")
    
    # 测试不同配置
    test_configs = [
        (32, 128),
        (64, 512),
        (128, 768),
        (256, 1024),
        (512, 2048),
    ]
    
    for batch_size, hidden_size in test_configs:
        print(f"\nTesting batch_size={batch_size}, hidden_size={hidden_size}")
        
        # 创建测试数据
        x = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float16)
        weight = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
        
        # Triton实现
        triton_result = rms_norm(x, weight)
        
        # PyTorch实现
        rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-6)
        torch_result = x / rms * weight
        
        # 验证结果
        rel_error = torch.abs(triton_result - torch_result) / torch.abs(torch_result)
        max_rel_error = torch.max(rel_error).item()
        mean_rel_error = torch.mean(rel_error).item()
        
        print(f"  Max relative error: {max_rel_error:.2e}")
        print(f"  Mean relative error: {mean_rel_error:.2e}")
        
        assert max_rel_error < 1e-3, f"Test failed for batch_size={batch_size}, hidden_size={hidden_size}"
    
    print("✓ All tests passed!")


def test_rms_norm_with_rms():
    """测试带RMS值的RMSNorm"""
    print("\nTesting RMSNorm with RMS values...")
    
    batch_size, hidden_size = 64, 512
    x = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float16)
    weight = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
    
    # Triton实现
    triton_result, triton_rms = rms_norm(x, weight, return_rms=True)
    
    # PyTorch实现
    torch_rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + 1e-6)
    torch_result = x / torch_rms * weight
    
    # 验证RMS值
    rms_error = torch.max(torch.abs(triton_rms - torch_rms.squeeze())).item()
    print(f"RMS error: {rms_error:.2e}")
    
    assert rms_error < 1e-5, f"RMS test failed: {rms_error}"
    
    print("✓ RMS test passed!")


def compare_layer_norm_rms_norm():
    """比较LayerNorm和RMSNorm的性能"""
    print("\nComparing LayerNorm vs RMSNorm...")
    
    batch_size, hidden_size = 128, 768
    x = torch.randn(batch_size, hidden_size, device='cuda', dtype=torch.float16)
    weight = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
    bias = torch.randn(hidden_size, device='cuda', dtype=torch.float16)
    
    # LayerNorm
    ln_times = []
    for _ in range(50):
        start_time = time.time()
        _ = torch.nn.functional.layer_norm(x, [hidden_size], weight, bias, 1e-5)
        torch.cuda.synchronize()
        ln_times.append(time.time() - start_time)
    
    # RMSNorm
    rms_times = []
    for _ in range(50):
        start_time = time.time()
        _ = rms_norm(x, weight)
        torch.cuda.synchronize()
        rms_times.append(time.time() - start_time)
    
    ln_mean = sum(ln_times) / len(ln_times)
    rms_mean = sum(rms_times) / len(rms_times)
    
    print(f"LayerNorm: {ln_mean*1000:.2f} ms")
    print(f"RMSNorm: {rms_mean*1000:.2f} ms")
    print(f"RMSNorm is {ln_mean/rms_mean:.2f}x faster than LayerNorm")


if __name__ == "__main__":
    # 运行测试
    test_rms_norm()
    test_rms_norm_with_rms()
    
    # 比较LayerNorm和RMSNorm
    compare_layer_norm_rms_norm()
    
    # 运行基准测试
    print("\n" + "="*60)
    benchmark_rms_norm(batch_size=128, hidden_size=768)
    
    # 测试不同大小的性能
    print("\n" + "="*60)
    print("Performance scaling test:")
    configs = [
        (32, 128), (64, 256), (128, 512), (256, 768), (512, 1024), (1024, 2048)
    ]
    for batch_size, hidden_size in configs:
        result = benchmark_rms_norm(batch_size=batch_size, hidden_size=hidden_size, 
                                   n_warmup=5, n_repeat=20)
        print(f"batch_size={batch_size}, hidden_size={hidden_size}: "
              f"Speedup = {result['speedup']:.2f}x")