# CuTLASS 编程基础

## 目录
- [编程模型概述](#编程模型概述)
- [CuTe 张量编程](#cute-张量编程)
- [Layout 系统详解](#layout-系统详解)
- [内存管理模式](#内存管理模式)
- [FlashMLA 编程模式分析](#flashmla-编程模式分析)
- [实践示例](#实践示例)

## 编程模型概述

CuTLASS 3.x 引入了全新的编程范式，核心是 **CuTe (Cu Tensor Extensions)**，它提供了比传统 CUDA 更高级的抽象。

### 传统 CUDA vs CuTe

| 特性 | 传统 CUDA | CuTe |
|------|-----------|------|
| **内存管理** | 手动指针操作 | 高级张量抽象 |
| **索引计算** | 手动计算偏移 | 自动布局管理 |
| **类型安全** | 运行时检查 | 编译时验证 |
| **代码可读性** | 较低 | 显著提升 |

### CuTe 核心组件

```cpp
// CuTe 的三个核心概念
1. Shape  - 张量的维度信息
2. Stride - 内存布局和步长
3. Layout - Shape + Stride 的组合
```

## CuTe 张量编程

### 1. 基本张量创建

```cpp
#include <cute/tensor.hpp>
using namespace cute;

// 创建 2D 张量布局 (8x16)
auto shape = make_shape(Int<8>{}, Int<16>{});
auto stride = make_stride(Int<16>{}, Int<1>{}); // 行主序
auto layout = make_layout(shape, stride);

// 创建张量
float* ptr = allocate_device_memory(8 * 16 * sizeof(float));
auto tensor = make_tensor(make_gmem_ptr(ptr), layout);
```

### 2. 编译时 vs 运行时

CuTe 强调编译时已知的信息：

```cpp
// 编译时已知 - 推荐方式
auto static_shape = make_shape(Int<64>{}, Int<128>{});

// 运行时确定 - 性能较差
auto dynamic_shape = make_shape(seq_len, head_dim);

// 混合模式 - 平衡灵活性和性能
auto hybrid_shape = make_shape(Int<64>{}, head_dim);
```

### 3. 张量分块 (Tiling)

FlashMLA 的核心技术之一：

```cpp
// 原始张量: 64x128
auto full_tensor = make_tensor(ptr, make_layout(
    make_shape(Int<64>{}, Int<128>{}),
    make_stride(Int<128>{}, Int<1>{})
));

// 分块大小: 16x32
auto tile_shape = make_shape(Int<16>{}, Int<32>{});

// 执行分块
auto tiled_tensor = zipped_divide(full_tensor, tile_shape);

// 结果: ((16,32), (4,4)) - 4x4 个 16x32 的块
```

### 4. 张量切片和操作

```cpp
// 获取特定的块
auto block_00 = tiled_tensor(make_coord(0, 0)); // 第一个块
auto block_11 = tiled_tensor(make_coord(1, 1)); // 对角线块

// 张量切片
auto first_row = slice(tensor, 0, _);           // 第一行
auto first_col = slice(tensor, _, 0);           // 第一列
auto sub_region = slice(tensor, 
                       make_coord(2, 4), 
                       make_coord(6, 8));        // 子区域
```

## Layout 系统详解

Layout 是 CuTe 的核心抽象，定义了逻辑索引到物理内存的映射。

### 1. 基础布局模式

```cpp
// 行主序 (Row Major)
auto row_major = make_layout(
    make_shape(Int<4>{}, Int<8>{}),     // 4x8 张量
    make_stride(Int<8>{}, Int<1>{})     // 步长: (8, 1)
);

// 列主序 (Column Major)  
auto col_major = make_layout(
    make_shape(Int<4>{}, Int<8>{}),     // 4x8 张量
    make_stride(Int<1>{}, Int<4>{})     // 步长: (1, 4)
);

// 带填充的布局 (Padding)
auto padded = make_layout(
    make_shape(Int<4>{}, Int<8>{}),     // 逻辑大小
    make_stride(Int<16>{}, Int<1>{})    // 物理步长有填充
);
```

### 2. Swizzled 布局

避免 Bank 冲突的关键技术：

```cpp
// 基础布局
auto base_layout = make_layout(
    make_shape(Int<32>{}, Int<32>{}),
    make_stride(Int<32>{}, Int<1>{})
);

// 应用 Swizzle
auto swizzled = composition(base_layout, Swizzle<3, 3, 3>{});

// Swizzle 的作用：
// - 重新排列内存访问模式
// - 减少共享内存 Bank 冲突
// - 提高内存带宽利用率
```

### 3. 分层布局 (Hierarchical Layout)

FlashMLA 使用的高级技术：

```cpp
// 创建分层布局
auto hierarchical = make_layout(
    make_shape(
        make_shape(Int<4>{}, Int<2>{}),  // Tile 形状
        make_shape(Int<8>{}, Int<16>{})  // 每个 Tile 内的形状
    ),
    make_stride(
        make_stride(Int<32>{}, Int<128>{}), // Tile 间步长
        make_stride(Int<16>{}, Int<1>{})    // Tile 内步长
    )
);
```

## 内存管理模式

### 1. 内存类型和指针

```cpp
// 全局内存指针
auto gmem_ptr = make_gmem_ptr(device_ptr);

// 共享内存指针
__shared__ float smem[1024];
auto smem_ptr = make_smem_ptr(smem);

// 寄存器内存指针
auto rmem_ptr = make_rmem_ptr<float>(nullptr);
```

### 2. 异步内存拷贝

CuTe 提供了高级的拷贝抽象：

```cpp
// 简单拷贝
cute::copy(src_tensor, dst_tensor);

// 异步拷贝 (需要 barrier 同步)
cute::copy_async(src_tensor, dst_tensor, barrier);

// 带谓词的拷贝 (条件拷贝)
cute::copy_if(predicate, src_tensor, dst_tensor);
```

### 3. TMA (Tensor Memory Accelerator)

Hopper 架构的专用内存加速器：

```cpp
// TMA 描述符创建
auto tma_desc = make_tma_copy(
    src_layout,        // 源布局
    dst_layout,        // 目标布局
    tile_shape,        // 传输块大小
    cluster_shape      // 集群形状
);

// TMA 异步加载
__global__ void tma_kernel() {
    // 启动 TMA 传输
    cute::copy(tma_desc, src_coord, dst_tensor);
    
    // 等待完成
    cute::wait(tma_desc);
}
```

## FlashMLA 编程模式分析

### 1. 核心数据结构

基于 FlashMLA 源码的分析：

```cpp
// FlashMLA 中的典型张量定义
template<typename Element>
struct FlashMLATensors {
    // Q 张量: (batch_size, num_heads, seq_len, head_dim)
    Tensor<Element> q_tensor;
    
    // KV Cache: 分页存储
    Tensor<Element> k_cache;  // (num_pages, PAGE_SIZE, head_dim)
    Tensor<Element> v_cache;  // (num_pages, PAGE_SIZE, head_dim)
    
    // 输出张量
    Tensor<Element> output;
    
    // 辅助张量
    Tensor<float> attention_scores;  // 临时注意力分数
    Tensor<float> softmax_stats;     // Softmax 统计量
};
```

### 2. 分块策略

```cpp
// FlashMLA 的分块模式
class FlashMLATiling {
public:
    static constexpr int PAGE_BLOCK_SIZE = 64;    // KV 页面大小
    static constexpr int HEAD_DIM_TILE = 64;      // 头维度分块
    static constexpr int SEQ_LEN_TILE = 128;      // 序列长度分块
    
    // 计算分块数量
    static auto compute_tile_count(int seq_len, int head_dim) {
        return make_shape(
            cute::ceil_div(seq_len, SEQ_LEN_TILE),
            cute::ceil_div(head_dim, HEAD_DIM_TILE)
        );
    }
};
```

### 3. Kernel 启动模式

```cpp
// FlashMLA 的典型 Kernel 启动
template<typename Config>
__global__ void flash_mla_kernel(
    typename Config::Params params
) {
    // 获取线程块和 Warp 信息
    auto thread_idx = cute::thread_idx_in_cluster();
    auto warp_idx = cute::canonical_warp_idx_sync();
    
    // 创建共享内存布局
    extern __shared__ char smem_buf[];
    auto smem_layout = Config::make_smem_layout();
    auto shared_tensors = Config::partition_smem(smem_buf, smem_layout);
    
    // 主计算循环
    for (auto tile_coord : params.tile_scheduler) {
        // 加载数据块
        load_tiles(params, tile_coord, shared_tensors);
        
        // 计算注意力分数
        compute_attention_scores(shared_tensors);
        
        // 应用 Softmax
        apply_online_softmax(shared_tensors);
        
        // 计算输出
        compute_attention_output(shared_tensors);
        
        // 存储结果
        store_output(params, tile_coord, shared_tensors);
    }
}
```

### 4. 内存访问优化

```cpp
// FlashMLA 的内存访问模式
template<typename TMA_Load, typename Barrier>
__device__ void optimized_load_pattern(
    TMA_Load const& tma_load,
    Barrier& barrier,
    auto const& src_tensor,
    auto& dst_tensor
) {
    // 1. 预取下一块数据
    if (threadIdx.x == 0) {
        cute::copy_async(tma_load, src_tensor, dst_tensor, barrier);
    }
    
    // 2. 处理当前块数据
    process_current_block();
    
    // 3. 等待预取完成
    cute::wait(barrier, 0);
    
    // 4. 同步所有线程
    __syncthreads();
}
```

## 实践示例

### 示例 1: 基础张量操作

```cpp
#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

__global__ void basic_tensor_demo() {
    // 创建 8x16 的张量布局
    auto layout = make_layout(
        make_shape(Int<8>{}, Int<16>{}),
        make_stride(Int<16>{}, Int<1>{})
    );
    
    if (threadIdx.x == 0) {
        printf("Layout: ");
        print(layout);
        printf("\n");
        
        printf("Shape: ");
        print(shape(layout));
        printf("\n");
        
        printf("Size: %d\n", size(layout));
    }
}
```

### 示例 2: 张量分块演示

```cpp
__global__ void tiling_demo() {
    // 原始张量: 32x64
    auto full_layout = make_layout(
        make_shape(Int<32>{}, Int<64>{}),
        make_stride(Int<64>{}, Int<1>{})
    );
    
    // 分块大小: 8x16
    auto tile_shape = make_shape(Int<8>{}, Int<16>{});
    
    // 执行分块
    auto tiled_layout = zipped_divide(full_layout, tile_shape);
    
    if (threadIdx.x == 0) {
        printf("Tiled layout: ");
        print(tiled_layout);
        printf("\n");
        
        // 分块后的形状: ((8,16), (4,4))
        // 表示: 4x4 个 8x16 的块
    }
}
```

### 示例 3: 内存拷贝模式

```cpp
__global__ void copy_pattern_demo() {
    // 共享内存缓冲区
    __shared__ float smem[16 * 32];
    
    // 创建共享内存张量
    auto smem_layout = make_layout(
        make_shape(Int<16>{}, Int<32>{}),
        make_stride(Int<32>{}, Int<1>{})
    );
    auto smem_tensor = make_tensor(make_smem_ptr(smem), smem_layout);
    
    // 模拟全局内存张量
    // (在实际使用中，这会是真实的设备内存)
    
    if (threadIdx.x == 0) {
        printf("Shared memory tensor layout: ");
        print(smem_layout);
        printf("\n");
    }
}
```

### 示例 4: Swizzling 演示

```cpp
__global__ void swizzling_demo() {
    // 基础布局: 16x16
    auto base_layout = make_layout(
        make_shape(Int<16>{}, Int<16>{}),
        make_stride(Int<16>{}, Int<1>{})
    );
    
    // 应用 Swizzle 避免 Bank 冲突
    auto swizzled_layout = composition(base_layout, Swizzle<2, 3, 3>{});
    
    if (threadIdx.x == 0) {
        printf("Base layout: ");
        print(base_layout);
        printf("\n");
        
        printf("Swizzled layout: ");
        print(swizzled_layout);
        printf("\n");
    }
}
```

## 编程最佳实践

### 1. 性能优化指南

```cpp
// ✅ 好的做法
// 1. 尽量使用编译时常量
auto static_shape = make_shape(Int<64>{}, Int<128>{});

// 2. 避免不必要的动态分配
template<int M, int N>
auto create_static_tensor() {
    return make_layout(make_shape(Int<M>{}, Int<N>{}), 
                      make_stride(Int<N>{}, Int<1>{}));
}

// 3. 使用合适的 Swizzle 模式
auto optimized_layout = composition(base_layout, Swizzle<3, 3, 3>{});

// ❌ 避免的做法
// 1. 过度使用运行时维度
auto bad_shape = make_shape(runtime_m, runtime_n);

// 2. 忽略内存对齐
auto unaligned_layout = make_stride(odd_number, 1);
```

### 2. 调试技巧

```cpp
// 启用详细输出
#define CUTE_DEBUG 1

// 打印布局信息
template<typename Layout>
void debug_layout(Layout const& layout) {
    print("Layout: "); print(layout); print("\n");
    print("Shape: "); print(shape(layout)); print("\n");
    print("Stride: "); print(stride(layout)); print("\n");
    print("Size: "); print(size(layout)); print("\n");
}

// 验证布局兼容性
template<typename LayoutA, typename LayoutB>
bool check_compatibility(LayoutA const& a, LayoutB const& b) {
    return compatible(a, b);
}
```

### 3. 错误处理

```cpp
// 编译时断言
template<typename Layout>
void validate_layout(Layout const& layout) {
    static_assert(rank(Layout{}) == 2, "Layout must be 2D");
    static_assert(size(Layout{}) > 0, "Layout must be non-empty");
}

// 运行时检查
template<typename Tensor>
bool validate_tensor(Tensor const& tensor) {
    return size(tensor) > 0 && 
           stride(tensor, 0) >= shape(tensor, 1);
}
```

## 总结

CuTe 编程范式的核心优势：

1. **类型安全**：编译时类型检查减少运行时错误
2. **高性能**：编译时优化和零开销抽象
3. **可读性**：代码意图更加清晰
4. **可维护性**：模块化设计便于扩展

### 与 FlashMLA 的关系

FlashMLA 充分利用了 CuTe 的特性：
- **张量抽象**：简化复杂的内存管理
- **分块操作**：高效的 Tile 算法实现
- **内存优化**：Swizzle 和 TMA 的深度集成
- **类型安全**：减少人为错误，提高代码质量

### 下一步学习

建议继续学习：
1. [CuTe 高级特性](./03_cute_advanced_features.md)
2. [TMA 内存加速器](./04_tma_programming.md)
3. [FlashMLA 源码解析](./05_flashmla_source_analysis.md)

通过这些基础知识，您将能够更好地理解和使用 FlashMLA，并开发自己的高性能 GPU 算子。 