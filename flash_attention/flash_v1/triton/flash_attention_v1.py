import torch
import triton
import triton.language as tl
import math
import time

# Flash Attention的核心思想:
# 1. 使用块级处理减少内存占用
# 2. 重组计算以减少HBM访问
# 3. 使用数学技巧保持数值稳定性

@triton.jit
def _flash_attention_kernel(
    # 查询、键、值张量指针
    q_ptr, k_ptr, v_ptr, output_ptr,
    # 张量的形状参数
    batch_size, seq_len, head_dim,
    # 块大小设置 (用于调优)
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    # 缩放因子
    scale,
):
    """Flash Attention内核实现"""
    # 获取当前程序的行和列块索引
    row_block_id = tl.program_id(0)
    col_block_id = tl.program_id(1)
    batch_id = tl.program_id(2)
    
    # 计算块大小和偏移量
    row_start = row_block_id * BLOCK_SIZE_M
    col_start = col_block_id * BLOCK_SIZE_N
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    col_offsets = col_start + tl.arange(0, BLOCK_SIZE_N)
    
    # 计算每个批次的基础偏移
    batch_offset = batch_id * seq_len * head_dim
    
    # 创建掩码以处理边界情况
    row_mask = row_offsets < seq_len
    col_mask = col_offsets < seq_len
    
    # 计算Q块的偏移量并加载
    q_block_ptr = q_ptr + batch_offset + row_offsets[:, None] * head_dim + tl.arange(0, head_dim)[None, :]
    q_block = tl.load(q_block_ptr, mask=row_mask[:, None], other=0.0)
    
    # 初始化累积器
    acc_o = tl.zeros([BLOCK_SIZE_M, head_dim], dtype=tl.float32)
    acc_p = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)  # softmax分母
    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")  # 每行的最大值
    
    # 初始化 L（用于softmax归一化）
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # 分块计算
    for k_col in range(0, tl.cdiv(seq_len, BLOCK_SIZE_N)):
        col_start_k = k_col * BLOCK_SIZE_N
        col_offsets_k = col_start_k + tl.arange(0, BLOCK_SIZE_N)
        col_mask_k = col_offsets_k < seq_len
        
        # 加载K、V块
        k_block_ptr = k_ptr + batch_offset + col_offsets_k[:, None] * head_dim + tl.arange(0, head_dim)[None, :]
        k_block = tl.load(k_block_ptr, mask=col_mask_k[:, None], other=0.0)
        
        v_block_ptr = v_ptr + batch_offset + col_offsets_k[:, None] * head_dim + tl.arange(0, head_dim)[None, :]
        v_block = tl.load(v_block_ptr, mask=col_mask_k[:, None], other=0.0)
        
        # 计算注意力分数 (S_ij = Q_i @ K_j^T * scale)
        s_qk = tl.dot(q_block, tl.trans(k_block)) * scale
        
        # 遮蔽无效位置
        if col_start_k + BLOCK_SIZE_N > seq_len:
            mask = row_offsets[:, None] < seq_len and col_offsets_k[None, :] < seq_len
            s_qk = tl.where(mask, s_qk, float("-inf"))
        
        # 计算当前块的最大值（用于数值稳定性）
        m_ij = tl.max(s_qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # 计算旧缩放因子
        p_scale = tl.exp(m_i - m_i_new)
        
        # 更新softmax分母
        l_i = l_i * p_scale + tl.sum(tl.exp(s_qk - m_i_new[:, None]), axis=1)
        
        # 更新输出
        for h in range(head_dim):
            v_h = v_block[:, h]
            s_qk_h = s_qk - m_i_new[:, None]
            p_ij_h = tl.exp(s_qk_h)
            acc_o[:, h] = acc_o[:, h] * p_scale + tl.sum(p_ij_h * v_h[None, :], axis=1)
        
        # 更新记录的最大值
        m_i = m_i_new
    
    # 最终归一化
    acc_o = acc_o / l_i[:, None]
    
    # 存储结果
    output_block_ptr = output_ptr + batch_offset + row_offsets[:, None] * head_dim + tl.arange(0, head_dim)[None, :]
    tl.store(output_block_ptr, acc_o, mask=row_mask[:, None])

def flash_attention(q, k, v, scale=None):
    """
    使用Flash Attention算法进行高效注意力计算
    
    参数:
        q: 查询张量 [batch_size, seq_len, head_dim]
        k: 键张量 [batch_size, seq_len, head_dim]
        v: 值张量 [batch_size, seq_len, head_dim]
        scale: 缩放因子，默认为1/sqrt(head_dim)
    
    返回:
        output: 注意力输出 [batch_size, seq_len, head_dim]
    """
    # 获取张量维度
    batch_size, seq_len, head_dim = q.shape
    
    # 默认缩放因子
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # 确保输入是连续的
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    # 分配输出张量
    output = torch.empty_like(q)
    
    # 确定块大小
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128
    
    # 计算网格维度
    grid = (
        triton.cdiv(seq_len, BLOCK_SIZE_M),  # 行块数
        triton.cdiv(seq_len, BLOCK_SIZE_N),  # 列块数
        batch_size,  # 批次数
    )
    
    # 启动内核
    _flash_attention_kernel[grid](
        q, k, v, output,
        batch_size, seq_len, head_dim,
        BLOCK_SIZE_M, BLOCK_SIZE_N,
        scale,
    )
    
    return output

def benchmark_compare(batch_size, seq_len, head_dim):
    """比较Flash Attention和朴素Attention的性能"""
    # 导入朴素Attention实现
    import sys
    sys.path.append("/data/lishizheng/cpp_projects/cuda-triton-learning/flash_attention/naive")
    from naive_attention import naive_attention
    
    # 创建随机输入
    q = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    k = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    v = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    
    # 预热
    _ = naive_attention(q, k, v)
    _ = flash_attention(q, k, v)
    torch.cuda.synchronize()
    
    # 测量朴素实现的时间
    torch.cuda.synchronize()
    start_time = time.time()
    naive_output = naive_attention(q, k, v)
    torch.cuda.synchronize()
    naive_time = (time.time() - start_time) * 1000  # 转为毫秒
    
    # 测量Flash Attention的时间
    torch.cuda.synchronize()
    start_time = time.time()
    flash_output = flash_attention(q, k, v)
    torch.cuda.synchronize()
    flash_time = (time.time() - start_time) * 1000  # 转为毫秒
    
    # 验证结果一致性
    is_close = torch.allclose(naive_output, flash_output, rtol=1e-2, atol=1e-2)
    
    return naive_time, flash_time, is_close

def main():
    """主函数：运行性能对比测试"""
    print("Flash Attention vs 朴素Attention性能对比")
    print("-------------------------------------")
    
    batch_size = 8
    head_dim = 64
    seq_lengths = [512, 1024, 2048, 4096, 8192]
    
    print("序列长度 | 朴素Attention (ms) | Flash Attention (ms) | 加速比 | 结果一致")
    print("--------|-------------------|---------------------|-------|----------")
    
    for seq_len in seq_lengths:
        try:
            naive_time, flash_time, is_close = benchmark_compare(batch_size, seq_len, head_dim)
            speedup = naive_time / flash_time
            print(f"{seq_len:8d} | {naive_time:17.2f} | {flash_time:19.2f} | {speedup:7.2f}x | {is_close}")
        except RuntimeError as e:
            print(f"{seq_len:8d} | 内存不足             | {flash_time:19.2f} | N/A     | N/A")
    
    print("\nFlash Attention的主要优势:")
    print("1. 减少了内存访问，通过块级计算避免存储完整的注意力矩阵")
    print("2. 重组计算以提高I/O效率，减少HBM访问")
    print("3. 在长序列上实现线性内存复杂度O(N)，而非朴素实现的平方复杂度O(N²)")

if __name__ == "__main__":
    main() 