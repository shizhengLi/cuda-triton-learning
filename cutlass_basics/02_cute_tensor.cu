/*
 * CuTe 张量操作示例
 * 
 * 本示例演示：
 * 1. CuTe 库的基本张量操作
 * 2. Layout 和 Tensor 的概念
 * 3. FlashMLA 中使用的张量分块技术
 */

#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <iostream>

using namespace cute;

// 演示基本张量操作
__global__ void basic_tensor_operations() {
    // 创建简单的张量布局
    auto layout = make_layout(make_shape(Int<8>{}, Int<16>{}),
                             make_stride(Int<16>{}, Int<1>{}));
    
    // 打印布局信息
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("张量布局: ");
        print(layout);
        printf("\n");
        
        printf("张量形状: ");
        print(shape(layout));
        printf("\n");
        
        printf("张量步长: ");
        print(stride(layout));
        printf("\n");
    }
}

// 演示张量分块操作（FlashMLA 核心技术）
__global__ void tensor_tiling_demo() {
    // 定义一个大张量的布局
    auto full_layout = make_layout(make_shape(Int<64>{}, Int<128>{}),
                                  make_stride(Int<128>{}, Int<1>{}));
    
    // 定义分块大小
    auto tile_shape = make_shape(Int<16>{}, Int<32>{});
    
    // 进行张量分块
    auto tiled_layout = zipped_divide(full_layout, tile_shape);
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== 张量分块演示 ===\n");
        printf("原始张量: ");
        print(full_layout);
        printf("\n");
        
        printf("分块形状: ");
        print(tile_shape);
        printf("\n");
        
        printf("分块后布局: ");
        print(tiled_layout);
        printf("\n");
    }
}

// 演示内存访问模式（类似 FlashMLA 中的 TMA 操作）
template<typename SrcTensor, typename DstTensor>
__device__ void copy_tensor_tile(SrcTensor const& src, DstTensor& dst) {
    // 使用 CuTe 的高级拷贝操作
    cute::copy(src, dst);
}

__global__ void memory_access_pattern() {
    // 模拟共享内存张量
    __shared__ float smem[16 * 32];
    
    // 创建共享内存张量布局
    auto smem_layout = make_layout(make_shape(Int<16>{}, Int<32>{}),
                                  make_stride(Int<32>{}, Int<1>{}));
    auto smem_tensor = make_tensor(make_smem_ptr(smem), smem_layout);
    
    // 模拟全局内存访问（在实际应用中这会是设备内存）
    auto thread_data = make_tensor(make_rmem_ptr<float>(nullptr), 
                                  make_layout(make_shape(Int<4>{})));
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== 内存访问模式演示 ===\n");
        printf("共享内存张量布局: ");
        print(smem_layout);
        printf("\n");
    }
}

// 演示 Swizzling（FlashMLA 性能优化关键技术）
__global__ void swizzling_demo() {
    // 创建带有 swizzling 的布局以避免 bank 冲突
    auto base_layout = make_layout(make_shape(Int<16>{}, Int<32>{}),
                                  make_stride(Int<32>{}, Int<1>{}));
    
    // 应用 swizzling 模式
    auto swizzled_layout = composition(base_layout, 
                                     Swizzle<3, 3, 3>{});
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("\n=== Swizzling 演示 ===\n");
        printf("基础布局: ");
        print(base_layout);
        printf("\n");
        
        printf("Swizzled 布局: ");
        print(swizzled_layout);
        printf("\n");
    }
}

// 主机端启动函数
int main() {
    std::cout << "CuTe 张量操作示例" << std::endl;
    std::cout << "=================" << std::endl;
    
    // 启动基本张量操作演示
    basic_tensor_operations<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    // 启动张量分块演示
    tensor_tiling_demo<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    // 启动内存访问模式演示
    memory_access_pattern<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    // 启动 Swizzling 演示
    swizzling_demo<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    std::cout << "\n✅ CuTe 张量操作演示完成!" << std::endl;
    
    return 0;
} 