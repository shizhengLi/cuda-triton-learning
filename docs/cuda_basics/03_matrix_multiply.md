# CUDA 学习笔记（3）：深入探索 `03_matrix_multiply.cu` 的矩阵乘法与共享内存优化

## 前言

在完成 `02_vector_add.cu` 的学习后，我进一步探索了 `03_matrix_multiply.cu`（代码路径：`cuda-triton-learning/cuda_basics/03_matrix_multiply.cu`），实现了一个矩阵乘法的 CUDA 程序。这个程序不仅展示了 CUDA 在二维数据处理中的应用，还通过朴素实现和共享内存优化的对比，突出了性能优化的重要性。本篇笔记将围绕矩阵乘法的核心公式、共享内存的优化原理以及我的困惑进行详细讲解，同时补充一些深入的知识点，记录我的第三步学习心得。

## 运行结果

运行 `./multiply` 后，输出如下：

```
朴素矩阵乘法执行时间: 0.640416 ms
共享内存矩阵乘法执行时间: 0.502784 ms
两种方法的结果一致
```

结果显示，共享内存版本比朴素版本更快，且两种方法计算结果一致。

## 知识点解析

### 1. 矩阵乘法的数学原理

矩阵乘法是线性代数中的核心运算，给定两个矩阵 `A`（维度 `M×K`）和 `B`（维度 `K×N`），计算结果矩阵 `C`（维度 `M×N`），其中：

```
C[i][j] = Σ(A[i][k] * B[k][j]), k 从 0 到 K-1
```

- 每个元素 `C[i][j]` 是 `A` 的第 `i` 行与 `B` 的第 `j` 列的点积。
- 在代码中，矩阵存储为行优先（Row-Major）的一维数组，`A[i][k]` 对应 `A[i * width + k]`，`B[k][j]` 对应 `B[k * width + j]`。

#### 困惑解答：`sum += A[row * width + k] * B[k * width + col]` 的公式

在朴素矩阵乘法内核中：

```cpp
sum += A[row * width + k] * B[k * width + col];
```

- **公式解释**：这行代码计算结果矩阵 `C` 中元素 `C[row][col]` 的值。`row` 是行索引，`col` 是列索引，`k` 是累加索引。
  - `A[row * width + k]`：访问矩阵 `A` 的第 `row` 行、第 `k` 列元素。
  - `B[k * width + col]`：访问矩阵 `B` 的第 `k` 行、第 `col` 列元素。
  - `sum += ...`：累加 `A[row][k] * B[k][col]`，`k` 从 0 到 `width-1`，得到 `C[row][col]`。
- **一行 × 一列**：这个公式本质上是矩阵乘法的点积运算，`A` 的第 `row` 行与 `B` 的第 `col` 列逐元素相乘后求和。
- **并行性**：每个线程负责计算 `C` 的一个元素（`C[row][col]`），通过二维线程索引（`blockIdx.y * blockDim.y + threadIdx.y` 和 `blockIdx.x * blockDim.x + threadIdx.x`）分配任务。例如，矩阵大小为 1024×1024，程序启动多个线程并行计算每个 `C[i][j]`。

### 2. 朴素矩阵乘法的实现

朴素实现（`matrixMultiplyNaive`）直接按照矩阵乘法公式计算：

- **线程分配**：每个线程计算 `C` 的一个元素，`row` 和 `col` 由线程索引确定。
- **内存访问**：每次计算需要从全局内存读取 `A[row][k]` 和 `B[k][col]`，共 `width` 次读取。
- **性能瓶颈**：
  - **全局内存访问**：全局内存延迟高（约 100-1000 周期），频繁访问导致性能低下。
  - **重复读取**：多个线程可能重复读取 `A` 和 `B` 的相同元素，浪费带宽。

### 3. 共享内存优化的原理

共享内存版本（`matrixMultiplyShared`）通过分块（Tiling）和共享内存（Shared Memory）优化性能。以下是详细分析：

#### 共享内存简介

- **共享内存**：每个线程块（Block）拥有一个快速的片上内存（Shared Memory），延迟低（约 1-2 周期），由块内所有线程共享。
- **用途**：通过将频繁访问的数据加载到共享内存，减少全局内存访问，提升性能。

#### 分块（Tiling）思想

- 矩阵乘法被拆分为多个小块（Tile），每个块大小为 `BLOCK_SIZE × BLOCK_SIZE`（代码中为 32×32）。
- 每个线程块负责计算 `C` 的一个子矩阵（Tile），通过循环加载 `A` 和 `B` 的子矩阵到共享内存。

#### 代码分析

```cpp
// 使用共享内存的矩阵乘法内核
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int width)
{
    // 定义共享内存数组
    __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];
    
    // 计算线程在结果矩阵中的位置
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    float sum = 0.0f;
    
    // 计算需要循环的tile数量
    int numTiles = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    for (int t = 0; t < numTiles; t++) 
    {
        // 计算当前tile在原始矩阵中的位置
        int tileRow = blockRow * BLOCK_SIZE + row;
        int tileCol = t * BLOCK_SIZE + col;
        
        // 加载数据到共享内存
        if (tileRow < width && tileCol < width)
            sharedA[row][col] = A[tileRow * width + tileCol];
        else
            sharedA[row][col] = 0.0f;
        
        tileRow = t * BLOCK_SIZE + row;
        tileCol = blockCol * BLOCK_SIZE + col;
        
        if (tileRow < width && tileCol < width)
            sharedB[row][col] = B[tileRow * width + tileCol];
        else
            sharedB[row][col] = 0.0f;
        
        // 确保所有线程都已加载完数据
        __syncthreads();
        
        // 使用共享内存计算当前tile的部分结果
        for (int k = 0; k < BLOCK_SIZE; k++) 
        {
            sum += sharedA[row][k] * sharedB[k][col];
        }
        
        // 确保所有线程都已完成计算，防止下一轮覆盖共享内存
        __syncthreads();
    }
    
    // 写回结果
    int globalRow = blockRow * BLOCK_SIZE + row;
    int globalCol = blockCol * BLOCK_SIZE + col;
    
    if (globalRow < width && globalCol < width)
        C[globalRow * width + globalCol] = sum;
}
```


在 `matrixMultiplyShared` 内核中：

- **共享内存定义**：

  ```cpp
  __shared__ float sharedA[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ float sharedB[BLOCK_SIZE][BLOCK_SIZE];
  ```

  每个线程块分配两个 32×32 的共享内存数组，分别存储 `A` 和 `B` 的子矩阵。

- **数据加载**：

  - 每个线程负责加载 `A` 和 `B` 的一个元素到 `sharedA` 和 `sharedB`。
  - 使用边界检查（如 `tileRow < width`）避免越界。

- **同步**：

  ```cpp
  __syncthreads();
  ```

  确保所有线程加载完数据后再计算，防止数据竞争。

- **计算**：

  - 每个线程使用共享内存中的 `sharedA[row][k]` 和 `sharedB[k][col]` 计算部分点积。
  - 通过循环（`t` 从 0 到 `numTiles`）累加所有子矩阵的贡献。

- **再次同步**：确保计算完成后，共享内存数据不会被下一轮覆盖。

#### 优化效果

- **减少全局内存访问**：每个子矩阵的 `A` 和 `B` 只从全局内存加载一次，存储到共享内存后，块内线程重复使用，显著降低全局内存访问次数。
- **内存合并**：线程按顺序加载连续的内存地址，符合 GPU 的合并访问（Coalesced Access）模式。
- **运行时间对比**：共享内存版本（0.502784 ms）比朴素版本（0.640416 ms）快约 20%，因为减少了全局内存访问的开销。

**深入知识**：

- **共享内存大小限制**：NVIDIA L20 GPU 每块共享内存通常为 48KB。本例中，两个 32×32 的 `float` 数组占用 `2 * 32 * 32 * 4 = 8192 字节`（8KB），远低于限制。
- **分块大小选择**：`BLOCK_SIZE = 32` 是经验值，匹配 GPU 线程束（Warp）大小（32）和共享内存容量。过大或过小的块可能降低性能。
- **银行冲突（Bank Conflict）**：共享内存分为多个存储体（Bank），如果多个线程同时访问同一存储体，可能导致冲突。二维数组 `sharedA` 和 `sharedB` 的访问模式通常能避免严重冲突。

### 4. 二维线程组织

矩阵乘法需要计算二维矩阵，使用二维线程索引：

```cpp
dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
dim3 gridDim((MATRIX_SIZE + blockDim.x - 1) / blockDim.x, 
             (MATRIX_SIZE + blockDim.y - 1) / blockDim.y);
```

- **块大小**：`blockDim(BLOCK_SIZE, BLOCK_SIZE)` 表示每个块有 32×32 = 1024 个线程。
- **网格大小**：`gridDim` 使用向上取整公式，确保覆盖整个 1024×1024 矩阵。例如，`MATRIX_SIZE = 1024`，`BLOCK_SIZE = 32`，则 `gridDim.x = gridDim.y = 32`。
- **索引计算**：
  - `row = blockIdx.y * blockDim.y + threadIdx.y`：计算全局行索引。
  - `col = blockIdx.x * blockDim.x + threadIdx.x`：计算全局列索引。

**深入知识**：

- **线程束调度**：GPU 以线程束（32 个线程）为单位调度，二维线程块需确保线程利用率高。32×32 是常见配置，平衡了并行性和资源占用。
- **占用率（Occupancy）**：每个块 1024 个线程可能限制占用率，后续可通过 NVIDIA 的 Occupancy Calculator 优化。

### 5. 时间测量与性能分析

代码使用 CUDA 事件（`cudaEvent_t`）测量内核执行时间：

```cpp
cudaEventRecord(start);
matrixMultiplyNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C_naive, MATRIX_SIZE);
cudaEventRecord(stop);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&milliseconds, start, stop);
```

- `cudaEventRecord`：记录时间戳。
- `cudaEventSynchronize`：确保事件完成。
- `cudaEventElapsedTime`：计算时间差（毫秒）。

**深入知识**：

- **性能瓶颈**：矩阵乘法是计算密集型任务（Compute-Bound），但全局内存访问仍占一定比例。共享内存优化主要减少内存访问时间。
- **分析工具**：使用 NVIDIA Nsight Systems 或 Visual Profiler 可视化时间线，分析内存传输、内核执行和同步的开销。例如，`nvprof ./multiply` 可以显示详细性能数据。
- **进一步优化**：
  - 使用向量化加载（如 `float4`）减少内存访问。
  - 调整 `BLOCK_SIZE`（如 16 或 64）测试性能变化。
  - 引入 CUDA 流（Streams）重叠数据传输和计算。

### 6. 结果验证

代码通过 `checkResult` 函数比较朴素和共享内存版本的结果，确保正确性：

- 使用 `fabs(a[i] - b[i]) > 1e-5` 检查浮点误差。
- 结果一致表明优化不影响计算精度。

### 7. 学习心得

通过 `03_matrix_multiply.cu`，我深入理解了以下内容：

- **矩阵乘法公式**：`C[i][j] = Σ(A[i][k] * B[k][j])` 通过行×列点积实现，每个线程并行计算一个元素。
- **共享内存优化**：通过分块和共享内存，显著减少全局内存访问，提升性能约 20%。
- **二维线程组织**：使用 `dim3` 配置网格和块，适应矩阵的二维结构。
- **性能分析**：CUDA 事件帮助量化优化效果，共享内存版本更快。

我的困惑得到了解答：矩阵乘法公式的 `A[row * width + k] * B[k * width + col]` 是行×列的点积，每个线程负责一个 `C[i][j]`，并行处理整个矩阵。

## 下一步计划

- 完成 `04_parallel_reduction.cu`，学习并行归约算法和优化技术。
- 使用 NVIDIA Nsight Systems 分析 `03_matrix_multiply.cu` 的性能瓶颈。
- 深入研究共享内存的银行冲突和占用率优化。
- 阅读《NVIDIA CUDA C 编程指南》第 4-6 章，掌握高级优化技巧。

## 总结

`03_matrix_multiply.cu` 是一个展示 CUDA 性能优化的优秀示例，通过对比朴素和共享内存版本，我理解了矩阵乘法的并行实现和共享内存的巨大优势。分块技术和共享内存减少了全局内存访问，显著提升性能。这篇笔记巩固了我的 CUDA 编程基础，也让我对性能优化有了更深的兴趣。下一阶段，我将探索更复杂的并行算法，继续优化 CUDA 程序！