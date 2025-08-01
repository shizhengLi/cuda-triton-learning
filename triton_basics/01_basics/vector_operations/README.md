# Triton基础算子：向量操作

## 概述

本节介绍Triton编程的基础知识，通过实现向量加法算子来理解Triton的核心概念。

## 学习目标

1. 理解Triton的编程模型
2. 掌握kernel函数的定义和调用
3. 学习grid计算和线程分配
4. 理解内存访问模式
5. 学会性能基准测试

## 核心概念

### 1. Kernel函数

Triton kernel是用`@triton.jit`装饰的Python函数，它将在GPU上执行：

```python
@triton.jit
def kernel_function(pointer_x, pointer_y, ..., BLOCK_SIZE: tl.constexpr):
    # GPU代码
```

### 2. Grid和Block

- **Grid**: 所有线程的集合，可以是1D、2D或3D
- **Block**: Grid中的一组线程，通常同时执行
- **Thread**: 单个执行单元

### 3. 程序ID

- `tl.program_id(0)`: 获取当前block在x轴上的ID
- `tl.program_id(1)`: 获取当前block在y轴上的ID
- `tl.program_id(2)`: 获取当前block在z轴上的ID

### 4. 内存访问

- `tl.load(pointer, mask)`: 从全局内存加载数据
- `tl.store(pointer, value, mask)`: 向全局内存存储数据
- `mask`: 防止越界访问的布尔掩码

## 向量加法实现详解

### 1. Kernel函数分析

```python
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # 1. 获取当前block的ID
    pid = tl.program_id(0)
    
    # 2. 计算当前block处理的起始位置
    block_start = pid * BLOCK_SIZE
    
    # 3. 计算当前线程处理的元素偏移
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # 4. 创建掩码处理边界情况
    mask = offsets < n_elements
    
    # 5. 加载数据
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # 6. 执行计算
    output = x + y
    
    # 7. 存储结果
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 2. Host函数分析

```python
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # 1. 输入验证
    assert x.shape == y.shape
    assert x.is_cuda and y.is_cuda
    
    # 2. 准备输出张量
    output = torch.empty_like(x)
    
    # 3. 计算grid参数
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # 4. 启动kernel
    vector_add_kernel[grid_size](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
```

## 性能优化要点

### 1. Block Size选择

- 通常选择1024或512（warp size的倍数）
- 太小：kernel launch开销大
- 太大：资源竞争严重

### 2. 内存合并访问

- 确保连续线程访问连续内存地址
- 使用`tl.arange`生成连续偏移量

### 3. 边界处理

- 使用mask防止越界访问
- 确保所有线程都有工作做

## 基准测试结果

在不同向量长度下的性能表现：

```
Size: 1,000,000
Triton:  0.12 ± 0.02 ms
PyTorch: 0.08 ± 0.01 ms
Speedup: 0.67x
Triton GFLOPS:  8.33
PyTorch GFLOPS: 12.50
```

## 常见问题

### 1. 为什么Triton比PyTorch慢？

- 小规模计算：kernel launch开销占比大
- PyTorch使用了高度优化的底层库
- 需要更复杂的优化才能超越PyTorch

### 2. 如何调试Triton代码？

- 使用`print`在kernel中输出调试信息
- 使用`torch.allclose`验证结果正确性
- 使用`torch.cuda.synchronize()`确保计算完成

### 3. 内存访问错误

- 检查mask是否正确设置
- 确保指针运算正确
- 验证输入张量在GPU上

## 扩展练习

1. **实现向量减法、乘法、除法**
2. **实现向量点积**
3. **实现向量逐元素操作（sin, cos, exp等）**
4. **实现向量累加（reduce）**
5. **优化block size选择策略**

## 下一步

完成向量操作后，可以学习：

1. 矩阵操作（矩阵乘法）
2. 内存访问模式优化
3. 深度学习核心算子（Softmax, LayerNorm等）
4. Attention机制实现

## 相关资源

- [Triton官方文档](https://triton-lang.org/main/)
- [GPU编程基础](https://developer.nvidia.com/cuda-education)
- [PyTorch CUDA扩展](https://pytorch.org/tutorials/advanced/cpp_extension.html)