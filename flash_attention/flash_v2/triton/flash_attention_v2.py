import torch
import triton
import triton.language as tl
import math
import time

@triton.jit
def _flash_attention_v2_forward_kernel(
    # 查询、键、值张量指针
    q_ptr, k_ptr, v_ptr, output_ptr,
    # softmax统计量指针 (用于backward)
    l_ptr, m_ptr,
    # 张量的形状参数
    batch_size, seq_len, head_dim,
    # 块大小设置
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_DMODEL: tl.constexpr,
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
    batch_offset = batch_id * seq_len * BLOCK_SIZE_DMODEL
    
    # 加载Q块
    q_offsets = tl.arange(0, BLOCK_SIZE_DMODEL)
    q_block_ptr = q_ptr + batch_offset + row_offsets[:, None] * BLOCK_SIZE_DMODEL + q_offsets[None, :]
    q_block = tl.load(q_block_ptr, mask=row_mask[:, None], other=0.0)
    
    # 初始化输出和softmax统计量
    acc_o = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_DMODEL], dtype=tl.float32)
    m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)
    
    # Flash Attention v2的核心改进：按列分块处理，减少HBM访问
    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        # 计算当前列块的范围
        end_n = min(start_n + BLOCK_SIZE_N, seq_len)
        col_offsets = start_n + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_offsets < end_n
        
        # 加载K块
        k_block_ptr = k_ptr + batch_offset + col_offsets[:, None] * BLOCK_SIZE_DMODEL + q_offsets[None, :]
        k_block = tl.load(k_block_ptr, mask=col_mask[:, None], other=0.0)
        
        # 加载V块
        v_block_ptr = v_ptr + batch_offset + col_offsets[:, None] * BLOCK_SIZE_DMODEL + q_offsets[None, :]
        v_block = tl.load(v_block_ptr, mask=col_mask[:, None], other=0.0)
        
        # 计算Q @ K^T
        qk = tl.dot(q_block, tl.trans(k_block)) * scale
        
        # 应用因果mask (如果需要)
        if start_n > row_start:
            causal_mask = col_offsets[None, :] <= row_offsets[:, None]
            qk = tl.where(causal_mask, qk, float('-inf'))
        
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
    output_block_ptr = output_ptr + batch_offset + row_offsets[:, None] * BLOCK_SIZE_DMODEL + q_offsets[None, :]
    tl.store(output_block_ptr, acc_o, mask=row_mask[:, None])
    
    # 存储softmax统计量 (用于backward)
    l_block_ptr = l_ptr + batch_offset + row_offsets
    m_block_ptr = m_ptr + batch_offset + row_offsets
    tl.store(l_block_ptr, l_i, mask=row_mask)
    tl.store(m_block_ptr, m_i, mask=row_mask)

@triton.jit
def _flash_attention_v2_backward_kernel(
    # 输出梯度指针
    doutput_ptr,
    # 输入梯度指针
    dq_ptr, dk_ptr, dv_ptr,
    # softmax统计量指针
    l_ptr, m_ptr,
    # 原始输入指针
    q_ptr, k_ptr, v_ptr,
    # 张量的形状参数
    batch_size, seq_len, head_dim,
    # 块大小设置
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_DMODEL: tl.constexpr,
    # 缩放因子
    scale,
):
    """Flash Attention v2 反向内核实现"""
    # 获取程序ID
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)
    
    # 计算当前处理的行范围
    row_start = pid * BLOCK_SIZE_M
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_M)
    row_mask = row_offsets < seq_len
    
    # 计算批次偏移
    batch_offset = batch_id * seq_len * BLOCK_SIZE_DMODEL
    
    # 加载softmax统计量
    l_i = tl.load(l_ptr + batch_offset + row_offsets, mask=row_mask)
    m_i = tl.load(m_ptr + batch_offset + row_offsets, mask=row_mask)
    
    # 加载输出梯度
    doutput_offsets = tl.arange(0, BLOCK_SIZE_DMODEL)
    doutput_block_ptr = doutput_ptr + batch_offset + row_offsets[:, None] * BLOCK_SIZE_DMODEL + doutput_offsets[None, :]
    doutput = tl.load(doutput_block_ptr, mask=row_mask[:, None], other=0.0)
    
    # 初始化梯度
    dq = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_DMODEL], dtype=tl.float32)
    
    # 反向传播
    for start_n in range(0, seq_len, BLOCK_SIZE_N):
        # 计算当前列块的范围
        end_n = min(start_n + BLOCK_SIZE_N, seq_len)
        col_offsets = start_n + tl.arange(0, BLOCK_SIZE_N)
        col_mask = col_offsets < end_n
        
        # 加载K, V块
        k_block_ptr = k_ptr + batch_offset + col_offsets[:, None] * BLOCK_SIZE_DMODEL + doutput_offsets[None, :]
        k_block = tl.load(k_block_ptr, mask=col_mask[:, None], other=0.0)
        
        v_block_ptr = v_ptr + batch_offset + col_offsets[:, None] * BLOCK_SIZE_DMODEL + doutput_offsets[None, :]
        v_block = tl.load(v_block_ptr, mask=col_mask[:, None], other=0.0)
        
        # 计算Q @ K^T
        qk = tl.dot(q_block, tl.trans(k_block)) * scale
        
        # 计算softmax
        p_ij = tl.exp(qk - m_i[:, None])
        p_ij = p_ij / l_i[:, None]
        
        # 计算dv
        dv += tl.dot(tl.trans(p_ij), doutput)
        
        # 计算dp
        dp = tl.dot(doutput, tl.trans(v_block))
        
        # 计算dq
        ds = (p_ij * (dp - tl.sum(p_ij * dp, axis=1, keepdims=True))) * scale
        dq += tl.dot(ds, k_block)
        
        # 计算dk
        dk += tl.dot(tl.trans(ds), q_block)
    
    # 存储梯度
    dq_block_ptr = dq_ptr + batch_offset + row_offsets[:, None] * BLOCK_SIZE_DMODEL + doutput_offsets[None, :]
    dk_block_ptr = dk_ptr + batch_offset + row_offsets[:, None] * BLOCK_SIZE_DMODEL + doutput_offsets[None, :]
    dv_block_ptr = dv_ptr + batch_offset + row_offsets[:, None] * BLOCK_SIZE_DMODEL + doutput_offsets[None, :]
    
    tl.store(dq_block_ptr, dq, mask=row_mask[:, None])
    tl.store(dk_block_ptr, dk, mask=row_mask[:, None])
    tl.store(dv_block_ptr, dv, mask=row_mask[:, None])

class FlashAttentionV2Function(torch.autograd.Function):
    """Flash Attention v2 的autograd函数"""
    
    @staticmethod
    def forward(ctx, q, k, v, scale=None):
        """
        Flash Attention v2 前向传播
        
        参数:
            q: 查询张量 [batch_size, seq_len, head_dim]
            k: 键张量 [batch_size, seq_len, head_dim]
            v: 值张量 [batch_size, seq_len, head_dim]
            scale: 缩放因子，默认为1/sqrt(head_dim)
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
        
        # 分配softmax统计量
        l_i = torch.empty(batch_size, seq_len, dtype=torch.float32, device=q.device)
        m_i = torch.empty(batch_size, seq_len, dtype=torch.float32, device=q.device)
        
        # 确定块大小 - Flash Attention v2优化
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_DMODEL = head_dim
        
        # 计算网格维度
        grid = (
            triton.cdiv(seq_len, BLOCK_SIZE_M),  # 行块数
            batch_size,  # 批次数
        )
        
        # 启动前向内核
        _flash_attention_v2_forward_kernel[grid](
            q, k, v, output,
            l_i, m_i,
            batch_size, seq_len, head_dim,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_DMODEL,
            scale,
        )
        
        # 保存用于backward的张量
        ctx.save_for_backward(q, k, v, l_i, m_i)
        ctx.scale = scale
        ctx.BLOCK_SIZE_M = BLOCK_SIZE_M
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.BLOCK_SIZE_DMODEL = BLOCK_SIZE_DMODEL
        
        return output
    
    @staticmethod
    def backward(ctx, doutput):
        """
        Flash Attention v2 反向传播
        
        参数:
            doutput: 输出梯度 [batch_size, seq_len, head_dim]
        """
        # 获取保存的张量
        q, k, v, l_i, m_i = ctx.saved_tensors
        scale = ctx.scale
        BLOCK_SIZE_M = ctx.BLOCK_SIZE_M
        BLOCK_SIZE_N = ctx.BLOCK_SIZE_N
        BLOCK_SIZE_DMODEL = ctx.BLOCK_SIZE_DMODEL
        
        batch_size, seq_len, head_dim = q.shape
        
        # 分配梯度张量
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        
        # 计算网格维度
        grid = (
            triton.cdiv(seq_len, BLOCK_SIZE_M),  # 行块数
            batch_size,  # 批次数
        )
        
        # 启动反向内核
        _flash_attention_v2_backward_kernel[grid](
            doutput,
            dq, dk, dv,
            l_i, m_i,
            q, k, v,
            batch_size, seq_len, head_dim,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_DMODEL,
            scale,
        )
        
        return dq, dk, dv, None

def flash_attention_v2(q, k, v, scale=None):
    """
    使用Flash Attention v2算法进行高效注意力计算
    
    Flash Attention v2的改进：
    1. 更好的IO-awareness，减少HBM访问
    2. 支持反向传播，可用于训练
    3. 优化的块大小和计算顺序
    4. 更好的数值稳定性
    
    参数:
        q: 查询张量 [batch_size, seq_len, head_dim]
        k: 键张量 [batch_size, seq_len, head_dim]
        v: 值张量 [batch_size, seq_len, head_dim]
        scale: 缩放因子，默认为1/sqrt(head_dim)
    
    返回:
        output: 注意力输出 [batch_size, seq_len, head_dim]
    """
    return FlashAttentionV2Function.apply(q, k, v, scale)

def benchmark_flash_attention_v2(batch_size, seq_len, head_dim):
    """测试Flash Attention v2性能"""
    # 创建随机输入
    q = torch.randn(batch_size, seq_len, head_dim, device='cuda', requires_grad=True)
    k = torch.randn(batch_size, seq_len, head_dim, device='cuda', requires_grad=True)
    v = torch.randn(batch_size, seq_len, head_dim, device='cuda', requires_grad=True)
    
    # 预热
    _ = flash_attention_v2(q, k, v)
    torch.cuda.synchronize()
    
    # 测量前向时间
    torch.cuda.synchronize()
    start_time = time.time()
    output = flash_attention_v2(q, k, v)
    torch.cuda.synchronize()
    forward_time = (time.time() - start_time) * 1000
    
    # 测量反向时间
    loss = output.sum()
    torch.cuda.synchronize()
    start_time = time.time()
    loss.backward()
    torch.cuda.synchronize()
    backward_time = (time.time() - start_time) * 1000
    
    return forward_time, backward_time

def main():
    """主函数：运行Flash Attention v2性能测试"""
    print("Flash Attention v2 性能测试")
    print("========================")
    
    batch_size = 4
    head_dim = 64
    seq_lengths = [512, 1024, 2048, 4096]
    
    print("序列长度 | 前向时间 (ms) | 反向时间 (ms) | 总时间 (ms)")
    print("--------|-------------|-------------|----------")
    
    for seq_len in seq_lengths:
        try:
            forward_time, backward_time = benchmark_flash_attention_v2(batch_size, seq_len, head_dim)
            total_time = forward_time + backward_time
            print(f"{seq_len:8d} | {forward_time:13.2f} | {backward_time:13.2f} | {total_time:10.2f}")
        except RuntimeError as e:
            print(f"{seq_len:8d} | 内存不足       | 内存不足       | 内存不足")
    
    print("\nFlash Attention v2的主要优势:")
    print("1. 支持反向传播，可用于模型训练")
    print("2. 更好的IO-awareness，减少HBM访问次数")
    print("3. 优化的块大小和计算顺序")
    print("4. 更好的数值稳定性")

if __name__ == "__main__":
    main()