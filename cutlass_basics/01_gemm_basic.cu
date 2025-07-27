/*
 * CuTLASS 基础 GEMM 示例
 * 
 * 本示例演示：
 * 1. CuTLASS 的基本使用方法
 * 2. Tile、Thread Block、Warp 的概念
 * 3. 与 FlashMLA 相关的优化技术
 */

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/tensor_fill.h>
#include <cutlass/util/reference/host/tensor_compare.h>
#include <cutlass/util/reference/host/gemm.h>

#include <iostream>
#include <vector>

// 定义 GEMM 配置
using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAccumulator = float;

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;

// CuTLASS GEMM 配置
using Gemm = cutlass::gemm::device::Gemm<
    ElementA, LayoutA,
    ElementB, LayoutB,
    ElementC, LayoutC,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,     // 使用 Tensor Core
    cutlass::arch::Sm80,                // Ampere 架构
    cutlass::gemm::GemmShape<128, 256, 64>,  // ThreadBlock tile shape
    cutlass::gemm::GemmShape<64, 64, 64>,    // Warp tile shape  
    cutlass::gemm::GemmShape<16, 8, 16>      // Instruction tile shape
>;

// 初始化矩阵数据
template<typename Element, typename Layout>
void initialize_matrix(cutlass::HostTensor<Element, Layout> &tensor, int seed = 2023) {
    cutlass::reference::host::TensorFillRandomUniform(
        tensor.host_view(),
        seed,
        Element(4),
        Element(-4),
        0  // bits
    );
}

// 验证结果正确性
template<typename Element, typename Layout>
bool verify_result(
    cutlass::HostTensor<Element, Layout> const& tensor_c,
    cutlass::HostTensor<Element, Layout> const& tensor_ref,
    ElementAccumulator tolerance = ElementAccumulator(0.1)
) {
    return cutlass::reference::host::TensorEquals(
        tensor_c.host_view(),
        tensor_ref.host_view(),
        tolerance
    );
}

int main() {
    std::cout << "CuTLASS 基础 GEMM 示例" << std::endl;
    std::cout << "======================" << std::endl;

    // 矩阵维度
    int M = 1024;
    int N = 1024; 
    int K = 512;

    // 分配主机内存
    cutlass::HostTensor<ElementA, LayoutA> tensor_a({M, K});
    cutlass::HostTensor<ElementB, LayoutB> tensor_b({K, N});
    cutlass::HostTensor<ElementC, LayoutC> tensor_c({M, N});
    cutlass::HostTensor<ElementC, LayoutC> tensor_ref({M, N});

    // 初始化矩阵
    initialize_matrix(tensor_a);
    initialize_matrix(tensor_b);
    cutlass::reference::host::TensorFill(tensor_c.host_view());
    cutlass::reference::host::TensorFill(tensor_ref.host_view());

    // 同步到设备
    tensor_a.sync_device();
    tensor_b.sync_device();
    tensor_c.sync_device();

    std::cout << "矩阵规模: A(" << M << "x" << K << ") × B(" << K << "x" << N 
              << ") = C(" << M << "x" << N << ")" << std::endl;

    // 配置 GEMM 参数
    ElementAccumulator alpha = ElementAccumulator(1);
    ElementAccumulator beta = ElementAccumulator(0);

    typename Gemm::Arguments arguments{
        {M, N, K},                                  // 问题规模
        {tensor_a.device_data(), tensor_a.stride(0)}, // A 矩阵
        {tensor_b.device_data(), tensor_b.stride(0)}, // B 矩阵  
        {tensor_c.device_data(), tensor_c.stride(0)}, // C 矩阵
        {tensor_c.device_data(), tensor_c.stride(0)}, // D 矩阵 (output)
        {alpha, beta}                               // 标量
    };

    // 初始化 GEMM 对象
    Gemm gemm_op;

    // 检查参数有效性
    cutlass::Status status = gemm_op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM 无法在当前硬件上执行!" << std::endl;
        return -1;
    }

    // 执行 GEMM
    status = gemm_op.initialize(arguments);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM 初始化失败!" << std::endl;
        return -1;
    }

    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    status = gemm_op();
    cudaEventRecord(stop);

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM 执行失败!" << std::endl;
        return -1;
    }

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // 同步结果到主机
    tensor_c.sync_host();

    // 计算参考结果进行验证
    cutlass::reference::host::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB, 
        ElementC, LayoutC,
        ElementAccumulator, ElementAccumulator
    > reference_gemm;

    reference_gemm(
        {M, N, K},
        alpha,
        tensor_a.host_ref(),
        tensor_b.host_ref(),
        beta,
        tensor_ref.host_ref(),
        tensor_ref.host_ref()
    );

    // 验证正确性
    bool passed = verify_result(tensor_c, tensor_ref);

    // 计算性能
    double gflops = (2.0 * M * N * K) / (milliseconds / 1000.0) / 1e9;

    std::cout << "\n性能结果:" << std::endl;
    std::cout << "执行时间: " << milliseconds << " ms" << std::endl;
    std::cout << "性能: " << gflops << " GFLOPS" << std::endl;
    std::cout << "正确性验证: " << (passed ? "通过" : "失败") << std::endl;

    if (passed) {
        std::cout << "\n✅ CuTLASS GEMM 示例执行成功!" << std::endl;
    } else {
        std::cout << "\n❌ 结果验证失败!" << std::endl;
        return -1;
    }

    // 清理资源
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
} 