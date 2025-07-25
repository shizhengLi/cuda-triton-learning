import torch
import time
import matplotlib.pyplot as plt

def naive_attention(q, k, v, scale=None):
    """
    朴素的注意力实现
    
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
        scale = 1.0 / (head_dim ** 0.5)
    
    # 计算注意力分数: Q @ K^T
    attn_weights = torch.bmm(q, k.transpose(1, 2))  # [batch_size, seq_len, seq_len]
    
    # 应用缩放
    attn_weights = attn_weights * scale
    
    # 应用softmax
    attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
    
    # 计算输出: attn_probs @ V
    output = torch.bmm(attn_probs, v)  # [batch_size, seq_len, head_dim]
    
    return output

def measure_memory(func, *args, **kwargs):
    """测量函数执行前后的GPU内存使用情况"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 记录初始内存
    start_mem = torch.cuda.memory_allocated()
    
    # 运行函数
    result = func(*args, **kwargs)
    
    # 记录峰值内存
    peak_mem = torch.cuda.max_memory_allocated()
    
    # 计算内存使用量
    memory_used = peak_mem - start_mem
    
    return result, memory_used / (1024**2)  # 转换为MB

def benchmark_attention(batch_size, seq_len, head_dim):
    """测量不同序列长度下的注意力计算性能"""
    # 创建随机输入
    q = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    k = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    v = torch.randn(batch_size, seq_len, head_dim, device='cuda')
    
    # 预热
    _ = naive_attention(q, k, v)
    torch.cuda.synchronize()
    
    # 测量时间
    start_time = time.time()
    _ = naive_attention(q, k, v)
    torch.cuda.synchronize()
    end_time = time.time()
    
    # 测量内存使用
    _, memory_used = measure_memory(naive_attention, q, k, v)
    
    return (end_time - start_time) * 1000, memory_used  # 时间转换为毫秒

def main():
    # 测试不同序列长度的性能和内存使用
    batch_size = 8
    head_dim = 64
    seq_lengths = [128, 256, 512, 1024, 2048, 4096]
    
    times = []
    memories = []
    
    print("序列长度 | 执行时间 (ms) | 内存使用 (MB)")
    print("--------|-------------|-------------")
    
    for seq_len in seq_lengths:
        try:
            time_ms, memory_mb = benchmark_attention(batch_size, seq_len, head_dim)
            times.append(time_ms)
            memories.append(memory_mb)
            print(f"{seq_len:8d} | {time_ms:13.2f} | {memory_mb:13.2f}")
        except RuntimeError as e:
            print(f"{seq_len:8d} | 内存不足       | 内存不足")
            times.append(None)
            memories.append(None)
    
    # 绘制性能图表
    valid_seq_lens = [seq_len for i, seq_len in enumerate(seq_lengths) if times[i] is not None]
    valid_times = [time for time in times if time is not None]
    valid_memories = [mem for mem in memories if mem is not None]
    
    if valid_seq_lens:
        plt.figure(figsize=(12, 5))
        
        # 执行时间图
        plt.subplot(1, 2, 1)
        plt.plot(valid_seq_lens, valid_times, 'o-', label='Naive Attention')
        plt.title('Attention执行时间')
        plt.xlabel('序列长度')
        plt.ylabel('执行时间 (ms)')
        plt.grid(True)
        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        
        # 内存使用图
        plt.subplot(1, 2, 2)
        plt.plot(valid_seq_lens, valid_memories, 'o-', label='Naive Attention')
        plt.title('Attention内存使用')
        plt.xlabel('序列长度')
        plt.ylabel('内存使用 (MB)')
        plt.grid(True)
        plt.xscale('log', base=2)
        plt.yscale('log', base=10)
        
        plt.tight_layout()
        plt.savefig('naive_attention_performance.png')
        print("\n性能图表已保存为 'naive_attention_performance.png'")
    
    print("\n分析结论:")
    print("1. 朴素Attention的时间复杂度为O(N²)，随序列长度增加，计算时间呈平方增长")
    print("2. 内存使用也为O(N²)，主要由注意力矩阵(N×N)决定")
    print("3. 这表明传统Attention在长序列上的局限性，为Flash Attention的必要性提供了基础")

if __name__ == "__main__":
    main() 