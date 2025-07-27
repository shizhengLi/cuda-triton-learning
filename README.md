# CUDA & Triton Learning: Flash Attention 与高性能算子实现

本项目致力于深入学习高性能 GPU 算子的设计与实现，以 Flash Attention 为核心案例，通过分析多种实现方式来掌握 CUDA 并行编程和 Triton 算子开发技能。

## 项目特色

- **理论与实践结合**：从数学原理到工程实现的完整学习路径
- **多框架对比**：Python、Triton、CUDA 三种实现方式的深度分析
- **前沿技术探索**：集成 DeepSeek FlashMLA 等业界最新高性能实现
- **系统性优化**：从算法到硬件的多层次性能优化

## 核心内容

### Flash Attention 深度学习
基于 tiny-flash-attention 项目的系统性学习材料：
- **理论原理**：Online Softmax、分块计算、内存优化
- **版本演进**：Flash Attention v1 vs v2 的详细对比
- **实现分析**：Python/Triton/CUDA 三种实现的深度解析
- **优化指南**：性能调优和工程实践

### 高性能 Kernel 实现
基于 DeepSeek FlashMLA 的前沿技术学习：
- **MLA 架构**：Multi-Head Latent Attention 的设计原理
- **Hopper 优化**：针对最新 GPU 架构的专门优化
- **极致性能**：在 H800 SXM5 上达到 660 TFLOPS 计算性能
- **工程实践**：变长序列服务的生产级实现

## 项目结构

```
cuda-triton-learning/
├── docs/                         # 详细学习文档
│   ├── flash_attention_study/    # Flash Attention 深度学习指南
│   │   ├── 01_theory.md         # 理论原理详解
│   │   ├── 02_v1_vs_v2.md       # 版本对比分析
│   │   ├── 03_implementation.md  # 实现深度分析
│   │   └── 04_optimization.md   # 优化指南
│   └── learning_plan.md          # 11周学习计划
│
├── cuda_basics/                  # CUDA 编程基础
│   ├── 01_hello_cuda.cu         # CUDA 入门
│   ├── 02_vector_add.cu         # 向量加法
│   ├── 03_matrix_multiply.cu     # 矩阵乘法
│   ├── 04_parallel_reduction.cu  # 并行规约
│   └── 05_stream_overlap.cu     # 流重叠
│
├── triton_basics/                # Triton 编程基础
│   ├── 01_vector_add.py         # 向量加法
│   └── 02_matrix_multiply.py     # 矩阵乘法
│
├── flash_attention/              # Flash Attention 实现
│   ├── naive/                    # 朴素实现
│   ├── flash_v1/                 # Flash Attention v1
│   └── flash_v2/                 # Flash Attention v2 (Triton)
│
├── tiny-flash-attention/         # 教育性实现 (submodule)
├── FlashMLA/                     # DeepSeek 高性能实现 (submodule)
├── benchmarks/                   # 性能测试
├── utils/                        # 工具函数
└── requirements.txt              # Python 依赖
```

## 快速开始

### 环境配置
```bash
# 创建虚拟环境
conda create -n flash_attention python=3.8
conda activate flash_attention

# 安装依赖
pip install -r requirements.txt

# 初始化子模块
git submodule update --init --recursive
```

### 运行示例
```bash
# CUDA 基础示例
cd cuda_basics
nvcc 01_hello_cuda.cu -o hello_cuda && ./hello_cuda

# Triton 示例
cd triton_basics
python 01_vector_add.py

# Flash Attention 测试
cd tiny-flash-attention/flash_attention_py
make

# FlashMLA 性能测试
cd FlashMLA
python tests/test_flash_mla.py
```

## 核心参考项目

### 🎓 [tiny-flash-attention](https://github.com/66RING/tiny-flash-attention)
教育导向的 Flash Attention 多语言实现：
- **Python**：算法原理理解
- **Triton**：高性能 GPU 编程入门
- **CUDA**：底层优化实践
- **CuTLASS**：矩阵计算优化

### 🚀 [FlashMLA](https://github.com/deepseek-ai/FlashMLA)
DeepSeek 开源的生产级 MLA 实现：
- **极致性能**：H800 SXM5 上 660 TFLOPS
- **Hopper 优化**：专门针对最新 GPU 架构
- **变长序列**：高效的可变长度序列服务
- **工程实践**：生产环境的性能优化

## 学习资源

### 核心论文
- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)

### 技术文档
- [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Triton 文档](https://triton-lang.org/)
- [Flash Attention 官方实现](https://github.com/Dao-AILab/flash-attention)

### 性能对比

| 实现方式 | 目标场景 | 性能特点 | 学习价值 |
|----------|----------|----------|----------|
| **tiny-flash-attention** | 教育学习 | 清晰易懂 | ⭐⭐⭐⭐⭐ |
| **FlashMLA** | 生产部署 | 极致性能 | ⭐⭐⭐⭐ |
| **Flash Attention 官方** | 研究开发 | 功能完整 | ⭐⭐⭐ |

## 致谢

### 核心贡献者
- **[@66RING](https://github.com/66RING)** - [tiny-flash-attention](https://github.com/66RING/tiny-flash-attention) 项目作者，为本学习项目提供了宝贵的教育资源
- **[DeepSeek AI](https://github.com/deepseek-ai)** - [FlashMLA](https://github.com/deepseek-ai/FlashMLA) 的开发团队，展示了生产级高性能实现

### 学术致谢
- **Tri Dao** 等人的 Flash Attention 原创性工作
- **OpenAI Triton** 团队推动的 GPU 编程革新
- **NVIDIA** 在 CUDA 生态系统方面的持续投入

## 使用指南

### 学习建议
1. **初学者**：从 `docs/flash_attention_study/` 的理论文档开始
2. **进阶者**：分析 `tiny-flash-attention` 的多种实现
3. **专家级**：研究 `FlashMLA` 的性能优化技术

### 性能测试
```bash
# 对比不同实现的性能
cd benchmarks
python benchmark_all.py

# 分析性能瓶颈
nsys profile python flash_attention_benchmark.py
```

### 开发实践
```bash
# 实现自己的优化版本
cp tiny-flash-attention/flash_attention_py/tiny_flash_attn.py my_implementation.py
# 在此基础上进行优化和改进
```

---

🚀 **开始您的高性能 GPU 编程学习之旅！**

本项目将帮助您从基础概念到前沿实现，系统掌握现代 GPU 计算的核心技术。