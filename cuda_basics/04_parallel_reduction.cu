#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include <time.h>

// 朴素的并行规约实现（有分歧问题）
__global__ void reductionNaive(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 将全局内存数据拷贝到共享内存
    extern __shared__ float sdata[];
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // 朴素的规约：每次迭代减半
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 第0个线程写结果
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// 优化版本1：减少分歧，从外向内规约
__global__ void reductionOptimized1(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sdata[];
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // 从外向内规约，减少分歧
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// 优化版本2：连续访问模式
__global__ void reductionOptimized2(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    extern __shared__ float sdata[];
    
    // 每个线程加载两个元素并立即相加
    sdata[tid] = (idx < n ? input[idx] : 0.0f) + (idx + blockDim.x < n ? input[idx + blockDim.x] : 0.0f);
    __syncthreads();
    
    // 从外向内规约
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// 优化版本3：展开最后一个warp
__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void reductionOptimized3(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    extern __shared__ float sdata[];
    
    // 每个线程加载两个元素
    sdata[tid] = (idx < n ? input[idx] : 0.0f) + (idx + blockDim.x < n ? input[idx + blockDim.x] : 0.0f);
    __syncthreads();
    
    // 正常规约到32个元素
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 展开最后一个warp
    if (tid < 32) warpReduce(sdata, tid);
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// 使用shuffle指令的warp级规约
__device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void reductionShuffle(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp级规约
    val = warpReduceSum(val);
    
    // 收集每个warp的结果
    __shared__ float warpSums[32]; // 假设最多32个warp
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    
    if (laneId == 0) {
        warpSums[warpId] = val;
    }
    __syncthreads();
    
    // 最后一个warp处理所有warp的结果
    if (warpId == 0) {
        val = (laneId < (blockDim.x + warpSize - 1) / warpSize) ? warpSums[laneId] : 0.0f;
        val = warpReduceSum(val);
        if (laneId == 0) {
            output[blockIdx.x] = val;
        }
    }
}

// CPU端规约，用于验证结果
float cpuReduction(float *data, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum;
}

// 基准测试函数
double benchmarkKernel(void (*kernel)(float*, float*, int), float *d_input, float *d_output, 
                      int dataSize, int blockSize, const char* kernelName) {
    int gridSize = (dataSize + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(float);
    
    // 预热
    kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, dataSize);
    cudaDeviceSynchronize();
    
    // 计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel<<<gridSize, blockSize, sharedMemSize>>>(d_input, d_output, dataSize);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    
    printf("%s: %.3f ms (平均)\n", kernelName, elapsed / 100.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return elapsed / 100.0f;
}

int main() {
    const int dataSize = 1 << 20; // 1M 元素
    const int blockSize = 256;
    const int gridSize = (dataSize + blockSize - 1) / blockSize;
    
    printf("并行规约算法性能比较\n");
    printf("数据大小: %d 元素\n", dataSize);
    printf("块大小: %d 线程\n", blockSize);
    printf("网格大小: %d 块\n\n", gridSize);
    
    // 分配主机内存
    float *h_input = (float*)malloc(dataSize * sizeof(float));
    float *h_output = (float*)malloc(gridSize * sizeof(float));
    
    // 初始化数据
    srand(time(NULL));
    for (int i = 0; i < dataSize; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;
    }
    
    // 计算CPU结果作为参考
    float cpuResult = cpuReduction(h_input, dataSize);
    printf("CPU 规约结果: %.6f\n\n", cpuResult);
    
    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, dataSize * sizeof(float));
    cudaMalloc(&d_output, gridSize * sizeof(float));
    
    // 拷贝数据到设备
    cudaMemcpy(d_input, h_input, dataSize * sizeof(float), cudaMemcpyHostToDevice);
    
    // 测试各种规约实现
    
    // 朴素实现
    reductionNaive<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, dataSize);
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    float gpuResult1 = cpuReduction(h_output, gridSize);
    printf("朴素规约结果: %.6f (误差: %.6f)\n", gpuResult1, fabs(gpuResult1 - cpuResult));
    benchmarkKernel((void(*)(float*, float*, int))reductionNaive, d_input, d_output, dataSize, blockSize, "朴素规约");
    
    // 优化版本1
    reductionOptimized1<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, dataSize);
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    float gpuResult2 = cpuReduction(h_output, gridSize);
    printf("优化规约1结果: %.6f (误差: %.6f)\n", gpuResult2, fabs(gpuResult2 - cpuResult));
    benchmarkKernel((void(*)(float*, float*, int))reductionOptimized1, d_input, d_output, dataSize, blockSize, "优化规约1");
    
    // 优化版本2
    int gridSize2 = (dataSize + blockSize * 2 - 1) / (blockSize * 2);
    reductionOptimized2<<<gridSize2, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, dataSize);
    cudaMemcpy(h_output, d_output, gridSize2 * sizeof(float), cudaMemcpyDeviceToHost);
    float gpuResult3 = cpuReduction(h_output, gridSize2);
    printf("优化规约2结果: %.6f (误差: %.6f)\n", gpuResult3, fabs(gpuResult3 - cpuResult));
    
    // 优化版本3
    reductionOptimized3<<<gridSize2, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, dataSize);
    cudaMemcpy(h_output, d_output, gridSize2 * sizeof(float), cudaMemcpyDeviceToHost);
    float gpuResult4 = cpuReduction(h_output, gridSize2);
    printf("优化规约3结果: %.6f (误差: %.6f)\n", gpuResult4, fabs(gpuResult4 - cpuResult));
    
    // Shuffle版本
    reductionShuffle<<<gridSize, blockSize>>>(d_input, d_output, dataSize);
    cudaMemcpy(h_output, d_output, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    float gpuResult5 = cpuReduction(h_output, gridSize);
    printf("Shuffle规约结果: %.6f (误差: %.6f)\n", gpuResult5, fabs(gpuResult5 - cpuResult));
    benchmarkKernel((void(*)(float*, float*, int))reductionShuffle, d_input, d_output, dataSize, blockSize, "Shuffle规约");
    
    // 清理内存
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("\n并行规约优化总结:\n");
    printf("1. 朴素实现存在严重的线程分歧问题\n");
    printf("2. 从外向内规约减少分歧，提高性能\n");
    printf("3. 每线程处理多个元素提高带宽利用率\n");
    printf("4. 展开最后一个warp减少同步开销\n");
    printf("5. 使用shuffle指令可以避免共享内存使用\n");
    
    return 0;
} 