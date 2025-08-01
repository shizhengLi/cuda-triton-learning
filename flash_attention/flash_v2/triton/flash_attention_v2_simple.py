import torch
import triton
import triton.language as tl
import math
import time

@triton.jit
def _flash_attention_v2_forward_kernel(
    # 查询、键、值张量指针
    q_ptr, k_ptr, v_ptr, output_ptr,
    # 张量的形状参数
    batch_size, seq_len, head_dim,
    # 块大小设置
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    # 缩放因子
    scale,
):
    """Flash Attention v2 前向内核实现"""
    # 获取程序ID
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)
    
    # 计算当前处理的行范围
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < seq_len
    
    # 计算批次偏移
    batch_offset = batch_id * seq_len * HEAD_DIM
    
    # 加载Q块
    head_offsets = tl.arange(0, HEAD_DIM)
    q_block_ptr = q_ptr + batch_offset + row_offsets[:, None] * HEAD_DIM + head_offsets[None, :]
    q_block = tl.load(q_block_ptr, mask=row_mask[:, None], other=0.0)
    
    # 初始化输出和softmax统计量
    acc_o = tl.zeros([BLOCK_SIZE_M, HEAD_DIM], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Flash Attention v2的核心改进：按列分块处理，减少HBM访问
    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        # 计算当前列块的范围
        end_n = min(start_n + BLOCK_SIZE_N, seq_len)
        col_offsets = start_n + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_offsets < end_n
        
        # 加载K块
        k_block_ptr = k_ptr + batch_offset + col_offsets[:, None] * HEAD_DIM + head_offsets[None, :]
        k_block = tl.load(k_block_ptr, mask=col_mask[:, None], other=0.0)
        
        # 加载V块
        v_block_ptr = v_ptr + batch_offset + col_offsets[:, None] * HEAD_DIM + head_offsets[None, :]
        v_block = tl.load(v_block_ptr, mask=col_mask[:, None], other=0.0)
        
        # 计算Q @ K^T
        qk = tl.dot(q_block, tl.trans(k_block)) * scale
        
        # 计算当前块的最大值
        m_ij = tl.max(qk, axis=1)
        m_i_new = tl.maximum(m_i, m_ij)
        
        # 计算缩放因子
        p_scale = tl.exp(m_i - m_i_new)
        
        # 更新softmax分母
        p_ij = tl.exp(qk - m_i_new[:, None])
        l_i_new = l_i * p_scale + tl.sum(p_ij, axis=1)
        
        # 更新输出
        acc_o = acc_o * p_scale[:, None] + tl.dot(p_ij, v_block)
        
        # 更新统计量
        m_i = m_i_new
        l_i = l_i_new
    
    # 最终归一化
    acc_o = acc_o / l_i[:, None]
    
    # 存储输出
    output_block_ptr = output_ptr + batch_offset + row_offsets[:, None] * HEAD_DIM + head_offsets[None, :]
    tl.store(output_block_ptr, acc_o, mask=row_mask[:, None])

def flash_attention_v2(q, k, v, scale=None):
    """
    使用Flash Attention v2算法进行高效注意力计算
    
    Flash Attention v2的改进：
    1. 更好的IO-awareness，减少HBM访问
    2. 优化的块大小和计算顺序
    3. 更好的数值稳定性
    
    参数:
        q: 查询张量 [batch_size, seq_len, head_dim]
        k: 键张量 [batch_size, seq_len, head_dim]
        v: 值张量 [batch_size, seq_len, head_dim]
        scale: 缩放因子，默认为1/sqrt(head_dim)
    
    返回:
        output: 注意力输出 [batch_size, seq_len, head_dim]
    """
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
    
    # 确定块大小 - Flash Attention v2优化
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_DMODEL = head_dim
    
    # 计算网格维度
    grid = (
        triton.cdiv(seq_len, BLOCK_SIZE_M),  # 行块数
        batch_size,  # 批次数
    )
    
    # 启动前向内核
    _flash_attention_v2_forward_kernel[grid](
        q, k, v, output,
        batch_size, seq_len, head_dim,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_DMODEL,
        scale,
    )
    
    return output

def benchmark_flash_attention_v2(batch_size, seq_len, head_dim):
    """测试Flash Attention v2性能"""
    # 创建随机输入
    q = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    k = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    v = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    
    # 预热
    _ = flash_attention_v2(q, k, v)
    torch.cuda.synchronize()
    
    # 测量前向时间
    torch.cuda.synchronize()
    start_time = time.time()
    output = flash_attention_v2(q, k, v)
    torch.cuda.synchronize()
    forward_time = (time.time() - start_time) * 1000
    
    return forward_time

def main():
    """主函数：运行Flash Attention v2性能测试"""
    print("Flash Attention v2 性能测试")
    print("========================")
    
    batch_size = 4
    head_dim = 64
    seq_lengths = [512, 1024, 2048, 4096]
    
    print("序列长度 | 前向时间 (ms)")
    print("--------|-------------")
    
    for seq_len in seq_lengths:
        try:
            forward_time = benchmark_flash_attention_v2(batch_size, seq_len, head_dim)
            print(f"{seq_len:8d} | {forward_time:13.2f}")
        except RuntimeError as e:
            print(f"{seq_len:8d} | 内存不足")
    
    print("\nFlash Attention v2的主要优势:")
    print("1. 更好的IO-awareness，减少HBM访问次数")
    print("2. 优化的块大小和计算顺序")
    print("3. 更好的数值稳定性")

if __name__ == "__main__":
    main()