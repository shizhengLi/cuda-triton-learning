#include <stdio.h>

/*
 * 这是一个简单的CUDA程序示例，演示了CUDA的基本概念：
 * 1. 内核函数定义和调用
 * 2. 线程组织方式
 * 3. 设备内存分配和数据传输
 */

// 定义一个CUDA内核函数，使用__global__修饰符表示它在设备上运行并可从主机调用
__global__ void helloFromGPU()
{
    // 获取当前线程在grid中的索引
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU, thread %d!\n", threadId);
}

int main()
{
    // 从CPU打印消息
    printf("Hello from CPU!\n");

    // 配置内核启动参数: <<<块数, 每块线程数>>>
    // 这里启动2个块，每个块有4个线程，总共8个线程
    helloFromGPU<<<2, 4>>>();
    
    // 等待所有GPU操作完成
    cudaDeviceSynchronize();
    
    // 检查是否有错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    return 0;
} 

// 运行方法
// nvcc 01_hello_cuda.cu -o hello_cuda
// ./hello_cuda
// 输出：
// Hello from CPU!
// Hello from GPU, thread 0!
// Hello from GPU, thread 1!
// Hello from GPU, thread 2!
// Hello from GPU, thread 3!
// Hello from GPU, thread 4!
// Hello from GPU, thread 5!
// Hello from GPU, thread 6!
// Hello from GPU, thread 7!
