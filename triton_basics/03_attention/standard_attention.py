"""
Triton标准Attention实现

本实现展示了如何使用Triton实现标准的自注意力机制，
包括：
- QKV矩阵乘法
- Attention Score计算
- Softmax计算
- 注意力权重应用
- 数值稳定性优化
"""

import triton
import triton.language as tl
import torch
import time
from typing import Optional


@triton.jit
def attention_score_kernel(
    q_ptr,          # Query矩阵指针 [batch_size, num_heads, seq_len, head_dim]
    k_ptr,          # Key矩阵指针 [batch_size, num_heads, seq_len, head_dim]
    score_ptr,      # Attention Score矩阵指针 [batch_size, num_heads, seq_len, seq_len]
    batch_size,     # 批量大小
    num_heads,      # 注意力头数
    seq_len,        # 序列长度
    head_dim,       # 头维度
    scale,          # 缩放因子
    BLOCK_SIZE_M: tl.constexpr,  # M维度block大小
    BLOCK_SIZE_N: tl.constexpr,  # N维度block大小
    BLOCK_SIZE_K: tl.constexpr,  # K维度block大小
):
    """
    计算Attention Score矩阵 S = Q @ K^T / sqrt(d_k)
    
    使用矩阵乘法的方式计算attention score
    """
    # 程序ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)
    pid_n = tl.program_id(3)
    
    # 计算当前block的起始位置
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # 计算当前线程处理的元素偏移量
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 计算Q和K的指针偏移
    q_offset = (pid_batch * num_heads * seq_len * head_dim + 
                pid_head * seq_len * head_dim + 
                offs_m[:, None] * head_dim + offs_k[None, :])
    
    k_offset = (pid_batch * num_heads * seq_len * head_dim + 
                pid_head * seq_len * head_dim + 
                offs_n[None, :] * head_dim + offs_k[:, None])
    
    # 计算Score矩阵的指针偏移
    score_offset = (pid_batch * num_heads * seq_len * seq_len + 
                    pid_head * seq_len * seq_len + 
                    offs_m[:, None] * seq_len + offs_n[None, :])
    
    # 创建掩码
    mask_m = offs_m < seq_len
    mask_n = offs_n < seq_len
    mask_k = offs_k < head_dim
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 分块计算
    for k in range(0, tl.cdiv(head_dim, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        mask_k_block = k_start + offs_k < head_dim
        
        # 加载Q和K的数据块
        q_ptrs = q_ptr + q_offset
        k_ptrs = k_ptr + k_offset
        
        q_mask = mask_m[:, None] & mask_k_block[None, :]
        k_mask = mask_k_block[:, None] & mask_n[None, :]
        
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        # 矩阵乘法累加
        accumulator += tl.dot(q, k)
    
    # 应用缩放因子并存储结果
    scores = accumulator * scale
    score_mask = mask_m[:, None] & mask_n[None, :]
    
    score_ptrs = score_ptr + score_offset
    tl.store(score_ptrs, scores, mask=score_mask)


@triton.jit
def softmax_kernel(
    score_ptr,      # Attention Score矩阵指针
    attn_ptr,       # Attention Weight矩阵指针
    batch_size,     # 批量大小
    num_heads,      # 注意力头数
    seq_len,        # 序列长度
    causal_mask,    # 是否使用因果掩码
    BLOCK_SIZE: tl.constexpr,
):
    """
    计算Softmax：Attention Weight = softmax(Score)
    
    支持因果掩码，用于自回归模型
    """
    # 程序ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_row = tl.program_id(2)
    
    # 计算当前行的偏移量
    row_offset = (pid_batch * num_heads * seq_len * seq_len + 
                  pid_head * seq_len * seq_len + 
                  pid_row * seq_len)
    
    # 计算当前线程处理的元素偏移量
    offsets = row_offset + tl.arange(0, BLOCK_SIZE)
    
    # 创建掩码
    mask = tl.arange(0, BLOCK_SIZE) < seq_len
    
    # 加载分数
    scores = tl.load(score_ptr + offsets, mask=mask, other=-float('inf'))
    
    # 应用因果掩码（如果需要）
    if causal_mask:
        causal_mask_vals = tl.arange(0, BLOCK_SIZE) <= pid_row
        scores = tl.where(causal_mask_vals, scores, -float('inf'))
    
    # 计算softmax
    # 数值稳定性：减去最大值
    max_score = tl.max(scores, axis=0)
    scores_stable = scores - max_score
    exp_scores = tl.exp(scores_stable)
    sum_exp = tl.sum(exp_scores, axis=0)
    attn_weights = exp_scores / sum_exp
    
    # 存储结果
    tl.store(attn_ptr + offsets, attn_weights, mask=mask)


@triton.jit
def attention_apply_kernel(
    attn_ptr,       # Attention Weight矩阵指针 [batch_size, num_heads, seq_len, seq_len]
    v_ptr,          # Value矩阵指针 [batch_size, num_heads, seq_len, head_dim]
    output_ptr,     # 输出矩阵指针 [batch_size, num_heads, seq_len, head_dim]
    batch_size,     # 批量大小
    num_heads,      # 注意力头数
    seq_len,        # 序列长度
    head_dim,       # 头维度
    BLOCK_SIZE_M: tl.constexpr,  # M维度block大小
    BLOCK_SIZE_N: tl.constexpr,  # N维度block大小
    BLOCK_SIZE_K: tl.constexpr,  # K维度block大小
):
    """
    应用注意力权重：Output = Attention Weight @ V
    """
    # 程序ID
    pid_batch = tl.program_id(0)
    pid_head = tl.program_id(1)
    pid_m = tl.program_id(2)
    pid_n = tl.program_id(3)
    
    # 计算当前block的起始位置
    m_start = pid_m * BLOCK_SIZE_M
    n_start = pid_n * BLOCK_SIZE_N
    
    # 计算当前线程处理的元素偏移量
    offs_m = m_start + tl.arange(0, BLOCK_SIZE_M)
    offs_n = n_start + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 计算Attention Weight和V的指针偏移
    attn_offset = (pid_batch * num_heads * seq_len * seq_len + 
                   pid_head * seq_len * seq_len + 
                   offs_m[:, None] * seq_len + offs_k[None, :])
    
    v_offset = (pid_batch * num_heads * seq_len * head_dim + 
                pid_head * seq_len * head_dim + 
                offs_k[:, None] * head_dim + offs_n[None, :])
    
    # 计算输出的指针偏移
    output_offset = (pid_batch * num_heads * seq_len * head_dim + 
                     pid_head * seq_len * head_dim + 
                     offs_m[:, None] * head_dim + offs_n[None, :])
    
    # 创建掩码
    mask_m = offs_m < seq_len
    mask_n = offs_n < head_dim
    mask_k = offs_k < seq_len
    
    # 初始化累加器
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 分块计算
    for k in range(0, tl.cdiv(seq_len, BLOCK_SIZE_K)):
        k_start = k * BLOCK_SIZE_K
        mask_k_block = k_start + offs_k < seq_len
        
        # 加载Attention Weight和V的数据块
        attn_ptrs = attn_ptr + attn_offset
        v_ptrs = v_ptr + v_offset
        
        attn_mask = mask_m[:, None] & mask_k_block[None, :]
        v_mask = mask_k_block[:, None] & mask_n[None, :]
        
        attn = tl.load(attn_ptrs, mask=attn_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        # 矩阵乘法累加
        accumulator += tl.dot(attn, v)
    
    # 存储结果
    output_mask = mask_m[:, None] & mask_n[None, :]
    output_ptrs = output_ptr + output_offset
    tl.store(output_ptrs, accumulator, mask=output_mask)


def standard_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal_mask: bool = False,
    scale: Optional[float] = None
) -> torch.Tensor:
    """
    标准Attention实现
    
    参数:
        q: Query矩阵 [batch_size, num_heads, seq_len, head_dim]
        k: Key矩阵 [batch_size, num_heads, seq_len, head_dim]
        v: Value矩阵 [batch_size, num_heads, seq_len, head_dim]
        causal_mask: 是否使用因果掩码
        scale: 缩放因子，默认为1/sqrt(head_dim)
    
    返回:
        Attention输出 [batch_size, num_heads, seq_len, head_dim]
    """
    # 输入验证
    assert q.dim() == 4, "Query must be 4D tensor"
    assert k.shape == q.shape, "Key shape must match Query shape"
    assert v.shape == q.shape, "Value shape must match Query shape"
    assert q.is_cuda, "Input tensors must be on GPU"
    
    batch_size, num_heads, seq_len, head_dim = q.shape
    
    # 计算缩放因子
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)
    
    # 准备中间张量
    scores = torch.empty(batch_size, num_heads, seq_len, seq_len, 
                         device=q.device, dtype=torch.float32)
    attn_weights = torch.empty_like(scores)
    output = torch.empty_like(q)
    
    # 设置block大小
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    
    # 计算grid大小
    grid_m = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (seq_len + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k = (head_dim + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    # 1. 计算Attention Score
    attention_score_kernel[(batch_size, num_heads, grid_m, grid_n)](
        q_ptr=q,
        k_ptr=k,
        score_ptr=scores,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        scale=scale,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    # 2. 计算Softmax
    BLOCK_SIZE_SOFTMAX = min(1024, triton.next_power_of_2(seq_len))
    grid_softmax = (batch_size, num_heads, seq_len)
    
    softmax_kernel[grid_softmax](
        score_ptr=scores,
        attn_ptr=attn_weights,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        causal_mask=causal_mask,
        BLOCK_SIZE=BLOCK_SIZE_SOFTMAX
    )
    
    # 3. 应用Attention权重
    grid_m_out = (seq_len + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n_out = (head_dim + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_k_out = (seq_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    
    attention_apply_kernel[(batch_size, num_heads, grid_m_out, grid_n_out)](
        attn_ptr=attn_weights,
        v_ptr=v,
        output_ptr=output,
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )
    
    return output


def benchmark_attention(batch_size: int = 8, num_heads: int = 8, 
                       seq_len: int = 512, head_dim: int = 64,
                       n_warmup: int = 5, n_repeat: int = 20):
    """
    Attention性能基准测试
    """
    print(f"Benchmarking Attention: batch_size={batch_size}, num_heads={num_heads}, "
          f"seq_len={seq_len}, head_dim={head_dim}")
    
    # 创建测试数据
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    device='cuda', dtype=torch.float16)
    
    # 计算理论FLOPS
    # Q@K^T: batch_size * num_heads * seq_len * seq_len * head_dim * 2
    # Softmax: batch_size * num_heads * seq_len * seq_len * 10 (approx)
    # Attn@V: batch_size * num_heads * seq_len * seq_len * head_dim * 2
    flops = (batch_size * num_heads * seq_len * seq_len * head_dim * 4 + 
             batch_size * num_heads * seq_len * seq_len * 10)
    
    # 预热
    for _ in range(n_warmup):
        _ = standard_attention(q, k, v)
        torch.cuda.synchronize()
    
    # 测试Triton实现
    triton_times = []
    for _ in range(n_repeat):
        start_time = time.time()
        _ = standard_attention(q, k, v)
        torch.cuda.synchronize()
        triton_times.append(time.time() - start_time)
    
    # 测试PyTorch实现
    torch_times = []
    for _ in range(n_repeat):
        start_time = time.time()
        # PyTorch实现
        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        torch.cuda.synchronize()
        torch_times.append(time.time() - start_time)
    
    # 计算统计信息
    triton_mean = sum(triton_times) / len(triton_times)
    torch_mean = sum(torch_times) / len(torch_times)
    
    triton_std = (sum((t - triton_mean) ** 2 for t in triton_times) / len(triton_times)) ** 0.5
    torch_std = (sum((t - torch_mean) ** 2 for t in torch_times) / len(torch_times)) ** 0.5
    
    # 计算GFLOPS
    triton_gflops = flops / (triton_mean * 1e9)
    torch_gflops = flops / (torch_mean * 1e9)
    
    print(f"\nResults:")
    print(f"Triton:  {triton_mean*1000:.2f} ± {triton_std*1000:.2f} ms")
    print(f"PyTorch: {torch_mean*1000:.2f} ± {torch_std*1000:.2f} ms")
    print(f"Speedup: {torch_mean/triton_mean:.2f}x")
    print(f"Triton GFLOPS:  {triton_gflops:.2f}")
    print(f"PyTorch GFLOPS: {torch_gflops:.2f}")
    
    # 验证结果正确性
    triton_result = standard_attention(q, k, v)
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weights = torch.softmax(scores, dim=-1)
    torch_result = torch.matmul(attn_weights, v)
    
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
        'triton_gflops': triton_gflops,
        'torch_gflops': torch_gflops
    }


def test_attention():
    """测试Attention的正确性"""
    print("Testing standard attention...")
    
    # 测试不同配置
    test_configs = [
        (2, 4, 64, 32),    # 小配置
        (4, 8, 128, 64),   # 中等配置
        (8, 8, 256, 64),   # 大配置
    ]
    
    for batch_size, num_heads, seq_len, head_dim in test_configs:
        print(f"\nTesting batch_size={batch_size}, num_heads={num_heads}, "
              f"seq_len={seq_len}, head_dim={head_dim}")
        
        # 创建测试数据
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                        device='cuda', dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                        device='cuda', dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                        device='cuda', dtype=torch.float16)
        
        # Triton实现
        triton_result = standard_attention(q, k, v)
        
        # PyTorch实现
        scale = 1.0 / (head_dim ** 0.5)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(scores, dim=-1)
        torch_result = torch.matmul(attn_weights, v)
        
        # 验证结果
        rel_error = torch.abs(triton_result - torch_result) / torch.abs(torch_result)
        max_rel_error = torch.max(rel_error).item()
        mean_rel_error = torch.mean(rel_error).item()
        
        print(f"  Max relative error: {max_rel_error:.2e}")
        print(f"  Mean relative error: {mean_rel_error:.2e}")
        
        assert max_rel_error < 1e-3, f"Test failed for the given configuration"
    
    print("✓ All tests passed!")


def test_causal_attention():
    """测试因果Attention"""
    print("\nTesting causal attention...")
    
    batch_size, num_heads, seq_len, head_dim = 2, 4, 64, 32
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    device='cuda', dtype=torch.float16)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    device='cuda', dtype=torch.float16)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, 
                    device='cuda', dtype=torch.float16)
    
    # Triton因果Attention
    triton_result = standard_attention(q, k, v, causal_mask=True)
    
    # PyTorch因果Attention
    scale = 1.0 / (head_dim ** 0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    
    # 创建因果掩码
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device)).view(1, 1, seq_len, seq_len)
    scores = scores.masked_fill(causal_mask == 0, float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    torch_result = torch.matmul(attn_weights, v)
    
    # 验证结果
    rel_error = torch.abs(triton_result - torch_result) / torch.abs(torch_result)
    max_rel_error = torch.max(rel_error).item()
    mean_rel_error = torch.mean(rel_error).item()
    
    print(f"Causal attention - Max relative error: {max_rel_error:.2e}")
    print(f"Causal attention - Mean relative error: {mean_rel_error:.2e}")
    
    assert max_rel_error < 1e-3, "Causal attention test failed"
    
    print("✓ Causal attention test passed!")


if __name__ == "__main__":
    # 运行测试
    test_attention()
    test_causal_attention()
    
    # 运行基准测试
    print("\n" + "="*60)
    benchmark_attention(batch_size=8, num_heads=8, seq_len=512, head_dim=64)
    
    # 测试不同大小的性能
    print("\n" + "="*60)
    print("Performance scaling test:")
    configs = [
        (4, 4, 256, 32),
        (8, 8, 512, 64),
        (16, 8, 1024, 64),
        (32, 16, 2048, 128),
    ]
    for batch_size, num_heads, seq_len, head_dim in configs:
        result = benchmark_attention(batch_size=batch_size, num_heads=num_heads, 
                                   seq_len=seq_len, head_dim=head_dim, 
                                   n_warmup=3, n_repeat=10)
        print(f"Config {batch_size}x{num_heads}x{seq_len}x{head_dim}: "
              f"Speedup = {result['speedup']:.2f}x, "
              f"GFLOPS = {result['triton_gflops']:.2f}")