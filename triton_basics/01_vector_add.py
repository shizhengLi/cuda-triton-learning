import torch
import triton
import triton.language as tl

# 定义Triton内核
@triton.jit
def vector_add_kernel(
    # 指针参数
    x_ptr,  # *fp32 输入向量 x
    y_ptr,  # *fp32 输入向量 y
    output_ptr,  # *fp32 输出向量
    # 非指针参数
    n_elements,  # 向量中的元素数
    BLOCK_SIZE: tl.constexpr,  # 每个程序处理的元素数
):
    # 程序ID
    pid = tl.program_id(axis=0)
    # 这个程序处理的元素的起始索引
    block_start = pid * BLOCK_SIZE
    # 这个程序处理的元素的偏移量
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建掩码来处理边界情况
    mask = offsets < n_elements
    # 加载输入向量
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # 计算结果
    output = x + y
    # 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)

# 封装成 PyTorch 友好的函数
def vector_add(x: torch.Tensor, y: torch.Tensor):
    # 输入验证
    assert x.shape == y.shape, "输入向量形状必须相同"
    assert x.is_cuda and y.is_cuda, "输入必须在 GPU 上"
    assert x.is_contiguous() and y.is_contiguous(), "输入必须是连续的"
    
    # 输出分配
    output = torch.empty_like(x)
    n_elements = output.numel()
    
    # 确定启动配置
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # 启动内核
    vector_add_kernel[grid](
        x, y, output, 
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def main():
    # 测试不同大小的向量
    sizes = [2**i for i in range(12, 28, 2)]  # 从 2^12 到 2^26
    
    for size in sizes:
        # 在 GPU 上创建随机向量
        x = torch.rand(size, device='cuda')
        y = torch.rand(size, device='cuda')
        
        # 使用 PyTorch 计算参考结果
        torch_output = x + y
        
        # 使用 Triton 计算结果
        triton_output = vector_add(x, y)
        
        # 验证结果
        assert torch.allclose(torch_output, triton_output)
        
        # 测试性能：PyTorch
        torch_time = triton.testing.do_bench(lambda: x + y)
        
        # 测试性能：Triton
        triton_time = triton.testing.do_bench(lambda: vector_add(x, y))
        
        # 计算加速比
        speedup = torch_time / triton_time
        
        print(f"向量大小: {size}, PyTorch: {torch_time:.3f} ms, Triton: {triton_time:.3f} ms, 加速比: {speedup:.2f}x")

if __name__ == "__main__":
    main() 