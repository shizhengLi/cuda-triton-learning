import torch
import triton
import triton.language as tl
import time

# 定义Triton矩阵乘法内核
@triton.jit
def matmul_kernel(
    # 指针参数
    a_ptr, b_ptr, c_ptr,
    # 矩阵维度
    M, N, K,
    # 块大小参数（用于调优）
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    # 可选: 分组大小
    GROUP_SIZE_M: tl.constexpr,
):
    """
    实现矩阵乘法 C = A @ B
    使用块级计算和共享内存优化
    """
    # 矩阵A是MxK，矩阵B是KxN，矩阵C是MxN
    
    # 计算程序ID
    pid = tl.program_id(axis=0)
    # 根据GROUP_SIZE_M拆分程序ID
    pid_m = pid // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % tl.cdiv(N, BLOCK_SIZE_N)
    
    # 计算当前块的起始位置
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # 为了避免额外的交换，我们使用积木编号初始化索引
    offs_a = offs_am[:, None] * K + offs_k[None, :]
    offs_b = offs_k[:, None] * N + offs_bn[None, :]
    
    # 初始化输出矩阵的累加寄存器
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # 迭代计算矩阵乘法
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # 边界检查
        a_mask = ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) < M) & ((k * BLOCK_SIZE_K + offs_k[None, :]) < K)
        b_mask = ((k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)) < K) & ((pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) < N)
        
        # 加载矩阵A和B的块
        a = tl.load(a_ptr + offs_a, mask=a_mask, other=0.0)
        b = tl.load(b_ptr + offs_b, mask=b_mask, other=0.0)
        
        # 执行矩阵乘法
        acc += tl.dot(a, b)
        
        # 更新偏移量
        offs_a += BLOCK_SIZE_K
        offs_b += BLOCK_SIZE_K * N
    
    # 计算结果矩阵的偏移量
    offs_c = offs_am[:, None] * N + offs_bn[None, :]
    
    # 边界检查
    c_mask = ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) < M) & ((pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) < N)
    
    # 存储结果
    tl.store(c_ptr + offs_c, acc, mask=c_mask)


# 自动调优的网格函数
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}),
    ],
    key=['M', 'N', 'K'],
)
def matmul(a, b):
    """
    计算矩阵乘法 a @ b
    """
    # 提取矩阵维度
    M, K = a.shape
    K, N = b.shape
    
    # 分配结果矩阵
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # 计算启动网格
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    # 启动内核
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
    
    return c


def benchmark(M, N, K, provider):
    """
    测量单次矩阵乘法的执行时间
    """
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    
    if provider == "torch":
        # PyTorch矩阵乘法
        torch.cuda.synchronize()
        start = time.time()
        c = a @ b
        torch.cuda.synchronize()
        end = time.time()
    elif provider == "triton":
        # Triton矩阵乘法
        torch.cuda.synchronize()
        start = time.time()
        c = matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
    
    return end - start


def main():
    # 测试精确度
    M, N, K = 1024, 1024, 1024
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    b = torch.randn((K, N), device='cuda', dtype=torch.float32)
    
    # 使用PyTorch计算参考结果
    torch_output = a @ b
    
    # 使用Triton计算结果
    triton_output = matmul(a, b)
    
    # 验证结果
    assert torch.allclose(torch_output, triton_output, atol=1e-2, rtol=1e-2)
    print("精确度验证通过!")
    
    # 比较不同矩阵大小的性能
    sizes = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), 
             (4096, 4096, 4096), (8192, 8192, 8192)]
    
    print("矩阵大小 | PyTorch (ms) | Triton (ms) | 加速比")
    print("---------|-------------|------------|--------")
    
    for size in sizes:
        M, N, K = size
        
        # 预热
        _ = benchmark(M, N, K, "torch")
        _ = benchmark(M, N, K, "triton")
        
        # 实际测试
        torch_time = benchmark(M, N, K, "torch") * 1000  # 转换为毫秒
        triton_time = benchmark(M, N, K, "triton") * 1000  # 转换为毫秒
        
        # 计算加速比
        speedup = torch_time / triton_time
        
        print(f"{M}x{N}x{K} | {torch_time:.2f} | {triton_time:.2f} | {speedup:.2f}x")


if __name__ == "__main__":
    main() 