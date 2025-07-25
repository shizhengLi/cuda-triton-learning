#include <stdio.h>
#include <stdlib.h>

// 向量大小
#define N 1000000

// CUDA内核函数 - 执行向量加法
__global__ void vectorAdd(float *a, float *b, float *c, int n)
{
    // 计算当前线程处理的元素索引
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    // 确保线程不会越界访问数组
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    // 主机内存指针
    float *h_a, *h_b, *h_c;
    // 设备内存指针
    float *d_a, *d_b, *d_c;
    
    size_t size = N * sizeof(float);
    
    // 分配主机内存
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c = (float *)malloc(size);
    
    // 初始化输入向量
    for (int i = 0; i < N; i++)
    {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }
    
    // 分配设备内存
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // 计算线程块和网格大小
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // 启动内核
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // 检查内核启动是否有错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    
    // 将结果从设备复制回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // 验证结果
    for (int i = 0; i < N; i++)
    {
        if (fabs(h_a[i] + h_b[i] - h_c[i]) > 1e-5)
        {
            printf("Result verification failed at element %d!\n", i);
            break;
        }
    }
    
    printf("Vector addition completed successfully!\n");
    
    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    // 释放主机内存
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
} 