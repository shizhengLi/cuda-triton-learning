# CUDA & Triton Learning Project: Flash Attention 实现探索

这个项目旨在通过学习和实现不同版本的 Flash Attention 算法，掌握 CUDA 并行编程和 Triton 算子开发技能。

## 项目目标

1. 从零开始学习 CUDA 并行编程基础
2. 理解 Triton 编程模型及其优势
3. 分析并实现不同版本的 Flash Attention 算法
4. 比较 CUDA 原生实现与 Triton 实现的差异和性能

## 学习路径

### 阶段一：CUDA 基础 (2-3周)

1. CUDA 架构与编程模型
   - GPU 硬件架构基础
   - CUDA 线程模型：Grid, Block, Thread
   - 内存模型：全局内存、共享内存、寄存器等
   
2. 基础 CUDA 编程
   - 环境设置与编译工具链
   - 内核函数编写
   - 内存管理与数据传输
   - 同步机制
   
3. CUDA 优化技术
   - 内存合并访问
   - Bank 冲突优化
   - 占用率与延迟隐藏
   - 共享内存使用模式

### 阶段二：Triton 编程模型 (2周)

1. Triton 基础
   - Triton 设计理念与 CUDA 的区别
   - Triton IR 与编译流程
   - 核心编程抽象

2. Triton 编程实践
   - 基本算子实现
   - 内存访问模式
   - 自动调优技术
   - 与 PyTorch 集成

### 阶段三：Flash Attention 实现 (3-4周)

1. Attention 机制基础
   - 自注意力计算原理
   - 传统 Attention 实现及其内存瓶颈

2. Flash Attention 算法
   - Flash Attention v1 核心思想
   - Block-wise 计算与内存优化
   - 数值稳定性处理
   
3. Flash Attention v2 改进
   - 优化计算模式
   - 内存访问模式优化
   - 性能提升点分析

4. CUDA vs Triton 实现对比
   - 代码复杂度对比
   - 性能分析与优化
   - 可维护性与可扩展性评估

## 项目结构

```
cuda-triton-learning/
├── docs/                 # 学习笔记和教程
├── cuda_basics/          # CUDA 基础学习示例
├── triton_basics/        # Triton 基础学习示例
├── flash_attention/
│   ├── naive/            # 朴素 Attention 实现
│   ├── flash_v1/         # Flash Attention v1 实现
│   │   ├── cuda/         # CUDA 实现
│   │   └── triton/       # Triton 实现
│   └── flash_v2/         # Flash Attention v2 实现
│       ├── cuda/         # CUDA 实现
│       └── triton/       # Triton 实现
├── tiny-flash-attention/ # 参考项目 (git submodule)
├── benchmarks/           # 性能测试代码
└── utils/                # 工具函数
```

## 参考资源

### 官方文档与论文
- [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Flash Attention 论文](https://arxiv.org/abs/2205.14135)
- [Flash Attention 2 论文](https://arxiv.org/abs/2307.08691)
- [Triton 文档](https://triton-lang.org/)
- [OpenAI Triton 教程](https://github.com/openai/triton)
- [FlashAttention GitHub](https://github.com/HazyResearch/flash-attention)

### 参考项目
- [tiny-flash-attention](https://github.com/66RING/tiny-flash-attention) - 本项目的重要参考，提供了 Python、Triton、CUDA、CuTLASS 等多种 Flash Attention 实现

## 致谢

特别感谢 [@66RING](https://github.com/66RING) 开发的 [tiny-flash-attention](https://github.com/66RING/tiny-flash-attention) 项目。该项目为我们提供了宝贵的学习资源和参考实现，包含了从基础 Python 实现到高性能 CUDA/CuTLASS 实现的完整示例，对本学习项目的设计和开发具有重要的指导意义。

## 使用参考项目

本项目已将 tiny-flash-attention 作为 git submodule 集成：

```bash
# 初始化和更新 submodule
git submodule update --init --recursive

# 查看参考项目内容
cd tiny-flash-attention
ls
```

您可以通过学习 `tiny-flash-attention/` 目录下的代码来：
- 对比不同实现方式的优劣
- 理解从基础到高性能实现的演进过程
- 获得更多实现思路和优化技巧