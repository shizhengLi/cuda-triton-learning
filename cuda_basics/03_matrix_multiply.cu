#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// 矩阵维度
#define MATRIX_SIZE 1024
// 分块大小
#define BLOCK_SIZE 32

// 朴素的矩阵乘法内核（不使用共享内存）
__global__ void matrixMultiplyNaive(float *A, float *B, float *C, int width)
{
    // 计算当前线程对应的结果矩阵中的元素位置
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) 
    {
        float sum = 0.0f;
        // 计算一个元素的结果
        for (int k = 0; k < width; k++) 
        {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

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

void initializeMatrix(float *mat, int width)
{
    for (int i = 0; i < width * width; i++)
    {
        mat[i] = rand() / (float)RAND_MAX;
    }
}

// 检查两个矩阵是否相等
bool checkResult(float *a, float *b, int width)
{
    for (int i = 0; i < width * width; i++)
    {
        if (fabs(a[i] - b[i]) > 1e-5)
        {
            printf("Error at index %d: %f vs %f\n", i, a[i], b[i]);
            return false;
        }
    }
    return true;
}

int main()
{
    float *h_A, *h_B, *h_C_naive, *h_C_shared;
    float *d_A, *d_B, *d_C_naive, *d_C_shared;
    
    size_t size = MATRIX_SIZE * MATRIX_SIZE * sizeof(float);
    
    // 分配主机内存
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C_naive = (float*)malloc(size);
    h_C_shared = (float*)malloc(size);
    
    // 初始化矩阵
    initializeMatrix(h_A, MATRIX_SIZE);
    initializeMatrix(h_B, MATRIX_SIZE);
    
    // 分配设备内存
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C_naive, size);
    cudaMalloc((void**)&d_C_shared, size);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // 定义内核启动参数
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((MATRIX_SIZE + blockDim.x - 1) / blockDim.x, 
                 (MATRIX_SIZE + blockDim.y - 1) / blockDim.y);
    
    // 创建CUDA事件来测量执行时间
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    
    // 朴素矩阵乘法
    cudaEventRecord(start);
    matrixMultiplyNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C_naive, MATRIX_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("朴素矩阵乘法执行时间: %f ms\n", milliseconds);
    
    // 共享内存矩阵乘法
    cudaEventRecord(start);
    matrixMultiplyShared<<<gridDim, blockDim>>>(d_A, d_B, d_C_shared, MATRIX_SIZE);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("共享内存矩阵乘法执行时间: %f ms\n", milliseconds);
    
    // 将结果从设备复制回主机
    cudaMemcpy(h_C_naive, d_C_naive, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_shared, d_C_shared, size, cudaMemcpyDeviceToHost);
    
    // 检查两种方法的结果是否相同
    bool resultsMatch = checkResult(h_C_naive, h_C_shared, MATRIX_SIZE);
    printf("两种方法的结果%s\n", resultsMatch ? "一致" : "不一致");
    
    // 释放CUDA事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_naive);
    cudaFree(d_C_shared);
    
    // 释放主机内存
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_shared);
    
    return 0;
} 