# Flash Attention v1 CUDA 开发指南

## 快速开始

### 1. 环境准备
```bash
# 检查 CUDA 版本
nvcc --version

# 检查 GPU 状态
nvidia-smi
```

### 2. 构建项目
```bash
# 进入项目目录
cd flash_attention

# 创建构建目录
mkdir -p build_cuda && cd build_cuda

# 配置项目
cmake ..

# 编译
make -j$(nproc)
```

### 3. 运行测试
```bash
# 运行所有测试
ctest --verbose

# 或直接运行可执行文件
./cuda_v1/test_flash_v1
```

## API 参考

### 主要函数

#### `simple_attention_kernel`
CUDA kernel 实现 Flash Attention

```cpp
__global__ void simple_attention_kernel(
    const float* q,    // Query 矩阵 [batch, seq_len, head_dim]
    const float* k,    // Key 矩阵 [batch, seq_len, head_dim] 
    const float* v,    // Value 矩阵 [batch, seq_len, head_dim]
    float* o,          // 输出矩阵 [batch, seq_len, head_dim]
    int batch_size,    // 批次大小
    int seq_len,       // 序列长度
    int head_dim,      // 注意力头维度
    float scale        // 缩放因子 (1.0 / sqrt(head_dim))
)
```

#### `naive_attention_cpu`
CPU 参考实现，用于验证结果

```cpp
void naive_attention_cpu(const float* q, const float* k, const float* v, float* o,
                        int batch_size, int seq_len, int head_dim, float scale)
```

### 辅助函数

#### `generate_random_data`
生成随机测试数据

```cpp
void generate_random_data(float* data, size_t size, 
                         float min_val = -1.0f, float max_val = 1.0f)
```

#### `compare_arrays`
比较两个数组的数值差异

```cpp
bool compare_arrays(const float* a, const float* b, size_t size, 
                   float tolerance = 1e-3f)
```

## 内存布局

### 输入/输出矩阵布局
所有矩阵都使用连续的内存布局，维度为 `[batch, seq_len, head_dim]`

```cpp
// 访问模式示例
float q_value = q[batch * seq_len * head_dim + seq * head_dim + dim];
```

### 线程分配策略
```cpp
// Kernel 启动配置
dim3 block(256, 1, 1);
dim3 grid(ceil_div(seq_len, block.x), batch_size, 1);

// 线程索引计算
int batch_idx = blockIdx.y;
int seq_idx = blockIdx.x * blockDim.x + threadIdx.x;
```

## 测试框架

### 测试配置
```cpp
TestConfig test_configs[] = {
    {1, 32, 16, "tiny"},      // 小规模测试
    {1, 64, 32, "small"},     // 中等规模测试
    {2, 128, 32, "medium"},   // 大规模测试
};
```

### 测试流程
1. **数据准备**: 生成随机 QKV 矩阵
2. **CPU 计算**: 运行参考实现
3. **CUDA 计算**: 运行 GPU 实现
4. **结果验证**: 比较数值结果
5. **性能统计**: 测量执行时间和加速比

## 性能优化建议

### 1. 内存访问优化
- 使用合并内存访问模式
- 考虑使用共享内存减少全局内存访问
- 使用 `__restrict__` 关键字帮助编译器优化

### 2. 计算优化
- 使用 CUDA 内置数学函数 (`__expf`, `__fmaxf`)
- 考虑使用 Tensor Core 进行矩阵乘法
- 使用 CUDA 流实现异步执行

### 3. 并行化策略
- 调整线程块大小以获得最佳占用率
- 考虑使用 2D 线程块布局
- 实现更细粒度的并行化

## 调试技巧

### 1. 调试输出
```cpp
// 在 kernel 中添加调试信息
printf("Thread (%d, %d): q=%.4f, k=%.4f\n", 
       blockIdx.x, threadIdx.x, q_ptr[0], k_ptr[0]);
```

### 2. 内存检查
```cpp
// 检查 CUDA 错误
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
```

### 3. 内存同步
```cpp
// 确保所有线程完成计算
__syncthreads();

// 等待 GPU 完成
cudaDeviceSynchronize();
```

## 常见问题

### 1. 编译错误
- **问题**: 找不到 CUDA 头文件
- **解决**: 确保 CUDA_PATH 环境变量正确设置

### 2. 运行时错误
- **问题**: CUDA out of memory
- **解决**: 减少批次大小或序列长度

### 3. 数值不匹配
- **问题**: CUDA 和 CPU 结果不一致
- **解决**: 检查 Softmax 实现和浮点精度

## 扩展功能

### 1. 支持更多数据类型
- half precision (FP16)
- bfloat16
- double precision (FP64)

### 2. 高级功能
- 注意力掩码 (attention mask)
- 因果注意力 (causal attention)
- 多头注意力 (multi-head attention)

### 3. 性能监控
- 使用 NVIDIA Nsight 进行性能分析
- 添加详细的性能计数器
- 实现自动性能调优

## 示例代码

### 基本使用
```cpp
// 准备数据
int batch_size = 1;
int seq_len = 64;
int head_dim = 32;
size_t total_size = batch_size * seq_len * head_dim;

float *d_q, *d_k, *d_v, *d_o;
cudaMalloc(&d_q, total_size * sizeof(float));
cudaMalloc(&d_k, total_size * sizeof(float));
cudaMalloc(&d_v, total_size * sizeof(float));
cudaMalloc(&d_o, total_size * sizeof(float));

// 复制数据到 GPU
cudaMemcpy(d_q, h_q, total_size * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_k, h_k, total_size * sizeof(float), cudaMemcpyHostToDevice);
cudaMemcpy(d_v, h_v, total_size * sizeof(float), cudaMemcpyHostToDevice);

// 启动 kernel
dim3 block(256, 1, 1);
dim3 grid(ceil_div(seq_len, block.x), batch_size, 1);
float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

simple_attention_kernel<<<grid, block>>>(d_q, d_k, d_v, d_o,
                                       batch_size, seq_len, head_dim, scale);

// 获取结果
cudaMemcpy(h_o, d_o, total_size * sizeof(float), cudaMemcpyDeviceToHost);

// 清理资源
cudaFree(d_q);
cudaFree(d_k);
cudaFree(d_v);
cudaFree(d_o);
```

---

**注意**: 本实现主要用于教学和研究目的。在生产环境中使用时，请考虑使用经过充分优化的库如 cuDNN 或 FlashAttention 官方实现。