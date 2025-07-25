# 环境设置指南

本项目用于学习CUDA并行编程和Triton算子开发，特别是通过Flash Attention实现。以下是设置开发环境的步骤。

## 系统要求

- NVIDIA GPU（支持CUDA的，Compute Capability 3.5+）
- CUDA工具包 11.3+
- Python 3.8+

## 安装步骤

### 1. 安装CUDA

如果您尚未安装CUDA，请从NVIDIA官方网站下载并安装适合您系统的CUDA工具包：
https://developer.nvidia.com/cuda-downloads

### 2. 创建Python虚拟环境（推荐）

```bash
# 使用conda（推荐）
conda create -n cuda-triton python=3.9
conda activate cuda-triton

# 或者使用venv
python -m venv cuda-triton-env
source cuda-triton-env/bin/activate  # Linux/Mac
# 或 cuda-triton-env\Scripts\activate  # Windows
```

### 3. 安装项目依赖

```bash
# 进入项目目录
cd /data/lishizheng/cpp_projects/cuda-triton-learning

# 安装依赖
pip install -r requirements.txt
```

## 编译和运行CUDA示例

对于CUDA示例程序，需要使用NVIDIA的nvcc编译器：

```bash
# 进入CUDA基础示例目录
cd cuda_basics

# 编译Hello World示例
nvcc -o hello_cuda 01_hello_cuda.cu

# 运行示例
./hello_cuda

# 编译向量加法示例
nvcc -o vector_add 02_vector_add.cu

# 运行示例
./vector_add

# 编译矩阵乘法示例
nvcc -o matrix_multiply 03_matrix_multiply.cu

# 运行示例
./matrix_multiply
```

## 运行Triton示例

Triton示例是Python程序，可以直接运行：

```bash
# 进入Triton基础示例目录
cd /data/lishizheng/cpp_projects/cuda-triton-learning/triton_basics

# 运行向量加法示例
python 01_vector_add.py

# 运行矩阵乘法示例
python 02_matrix_multiply.py
```

## 运行Attention实现对比

```bash
# 运行朴素Attention实现
cd /data/lishizheng/cpp_projects/cuda-triton-learning/flash_attention/naive
python naive_attention.py

# 运行基于Triton的Flash Attention v1实现
cd /data/lishizheng/cpp_projects/cuda-triton-learning/flash_attention/flash_v1/triton
python flash_attention_v1.py
```

## 故障排除

如果遇到以下问题：

### CUDA相关错误

- 确保CUDA路径已正确添加到系统环境变量中
- 检查CUDA版本与GPU驱动程序是否兼容

### Triton安装问题

- 确保PyTorch版本与Triton兼容（通常需要PyTorch 2.0+）
- 如果安装失败，尝试从源码安装：
  ```bash
  pip install ninja
  git clone https://github.com/openai/triton.git
  cd triton/python
  pip install -e .
  ```

### 内存不足错误

- 减少测试的序列长度或批量大小
- 确保没有其他程序占用大量GPU内存 