#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>

// 简单的计算密集型内核
__global__ void computeIntensive(float *input, float *output, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = input[idx];
        // 执行一些计算密集的操作
        for (int i = 0; i < iterations; i++) {
            val = sinf(val) + cosf(val);
        }
        output[idx] = val;
    }
}

// 向量加法内核
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// 测量时间的辅助函数
float measureTime(cudaEvent_t start, cudaEvent_t stop) {
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
}

// 初始化数据
void initializeData(float *data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = (float)(rand() % 100) / 100.0f;
    }
}

// 无重叠的同步执行
float sequentialExecution(float *h_input, float *h_output, int dataSize, int numStreams) {
    const int blockSize = 256;
    const int segmentSize = dataSize / numStreams;
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, dataSize * sizeof(float));
    cudaMalloc(&d_output, dataSize * sizeof(float));
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    for (int i = 0; i < numStreams; i++) {
        int offset = i * segmentSize;
        int currentSegmentSize = (i == numStreams - 1) ? dataSize - offset : segmentSize;
        
        // 数据传输到设备
        cudaMemcpy(d_input + offset, h_input + offset, 
                   currentSegmentSize * sizeof(float), cudaMemcpyHostToDevice);
        
        // 执行计算
        int gridSize = (currentSegmentSize + blockSize - 1) / blockSize;
        computeIntensive<<<gridSize, blockSize>>>(d_input + offset, d_output + offset, 
                                                 currentSegmentSize, 1000);
        
        // 数据传输回主机
        cudaMemcpy(h_output + offset, d_output + offset, 
                   currentSegmentSize * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed = measureTime(start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return elapsed;
}

// 使用多个流进行重叠执行
float overlappedExecution(float *h_input, float *h_output, int dataSize, int numStreams) {
    const int blockSize = 256;
    const int segmentSize = dataSize / numStreams;
    
    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, dataSize * sizeof(float));
    cudaMalloc(&d_output, dataSize * sizeof(float));
    
    // 分配页锁定主机内存（提高传输性能）
    float *h_input_pinned, *h_output_pinned;
    cudaMallocHost(&h_input_pinned, dataSize * sizeof(float));
    cudaMallocHost(&h_output_pinned, dataSize * sizeof(float));
    
    // 复制数据到页锁定内存
    memcpy(h_input_pinned, h_input, dataSize * sizeof(float));
    
    // 创建多个流
    cudaStream_t *streams = (cudaStream_t*)malloc(numStreams * sizeof(cudaStream_t));
    for (int i = 0; i < numStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // 启动所有流的操作
    for (int i = 0; i < numStreams; i++) {
        int offset = i * segmentSize;
        int currentSegmentSize = (i == numStreams - 1) ? dataSize - offset : segmentSize;
        
        // 异步内存传输到设备
        cudaMemcpyAsync(d_input + offset, h_input_pinned + offset, 
                        currentSegmentSize * sizeof(float), 
                        cudaMemcpyHostToDevice, streams[i]);
        
        // 异步执行内核
        int gridSize = (currentSegmentSize + blockSize - 1) / blockSize;
        computeIntensive<<<gridSize, blockSize, 0, streams[i]>>>(
            d_input + offset, d_output + offset, currentSegmentSize, 1000);
        
        // 异步内存传输回主机
        cudaMemcpyAsync(h_output_pinned + offset, d_output + offset, 
                        currentSegmentSize * sizeof(float), 
                        cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // 等待所有流完成
    for (int i = 0; i < numStreams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed = measureTime(start, stop);
    
    // 复制结果回原始内存
    memcpy(h_output, h_output_pinned, dataSize * sizeof(float));
    
    // 清理资源
    for (int i = 0; i < numStreams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    free(streams);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFreeHost(h_input_pinned);
    cudaFreeHost(h_output_pinned);
    cudaFree(d_input);
    cudaFree(d_output);
    
    return elapsed;
}

// 演示流优先级的使用
void demonstratePriority() {
    printf("\n=== 演示流优先级 ===\n");
    
    const int dataSize = 1024 * 1024;
    const int blockSize = 256;
    const int gridSize = (dataSize + blockSize - 1) / blockSize;
    
    // 分配内存
    float *d_data1, *d_data2, *d_result1, *d_result2;
    cudaMalloc(&d_data1, dataSize * sizeof(float));
    cudaMalloc(&d_data2, dataSize * sizeof(float));
    cudaMalloc(&d_result1, dataSize * sizeof(float));
    cudaMalloc(&d_result2, dataSize * sizeof(float));
    
    // 创建不同优先级的流
    cudaStream_t highPriorityStream, lowPriorityStream;
    
    int leastPriority, greatestPriority;
    cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
    
    cudaStreamCreateWithPriority(&highPriorityStream, cudaStreamNonBlocking, greatestPriority);
    cudaStreamCreateWithPriority(&lowPriorityStream, cudaStreamNonBlocking, leastPriority);
    
    printf("优先级范围: %d (最低) 到 %d (最高)\n", leastPriority, greatestPriority);
    
    // 创建事件来测量时间
    cudaEvent_t start1, start2, stop1, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&start2);
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);
    
    // 启动低优先级任务
    cudaEventRecord(start2, lowPriorityStream);
    computeIntensive<<<gridSize, blockSize, 0, lowPriorityStream>>>(d_data2, d_result2, dataSize, 2000);
    cudaEventRecord(stop2, lowPriorityStream);
    
    // 稍后启动高优先级任务
    cudaEventRecord(start1, highPriorityStream);
    computeIntensive<<<gridSize, blockSize, 0, highPriorityStream>>>(d_data1, d_result1, dataSize, 1000);
    cudaEventRecord(stop1, highPriorityStream);
    
    // 等待完成并测量时间
    cudaEventSynchronize(stop1);
    cudaEventSynchronize(stop2);
    
    float time1 = measureTime(start1, stop1);
    float time2 = measureTime(start2, stop2);
    
    printf("高优先级流执行时间: %.2f ms\n", time1);
    printf("低优先级流执行时间: %.2f ms\n", time2);
    
    // 清理
    cudaStreamDestroy(highPriorityStream);
    cudaStreamDestroy(lowPriorityStream);
    cudaEventDestroy(start1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);
    cudaFree(d_data1);
    cudaFree(d_data2);
    cudaFree(d_result1);
    cudaFree(d_result2);
}

// 演示多流计算和传输重叠
void demonstrateComputeTransferOverlap() {
    printf("\n=== 计算和传输重叠演示 ===\n");
    
    const int dataSize = 2 * 1024 * 1024; // 2M 元素
    const int numBatches = 4;
    const int batchSize = dataSize / numBatches;
    const int blockSize = 256;
    
    // 分配页锁定主机内存
    float *h_input, *h_output;
    cudaMallocHost(&h_input, dataSize * sizeof(float));
    cudaMallocHost(&h_output, dataSize * sizeof(float));
    
    // 分配设备内存
    float *d_input, *d_output;
    cudaMalloc(&d_input, dataSize * sizeof(float));
    cudaMalloc(&d_output, dataSize * sizeof(float));
    
    // 初始化数据
    initializeData(h_input, dataSize);
    
    // 创建流
    cudaStream_t computeStream, transferStream;
    cudaStreamCreate(&computeStream);
    cudaStreamCreate(&transferStream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    printf("处理 %d 个批次，每批次 %d 个元素\n", numBatches, batchSize);
    
    cudaEventRecord(start);
    
    for (int batch = 0; batch < numBatches; batch++) {
        int offset = batch * batchSize;
        int gridSize = (batchSize + blockSize - 1) / blockSize;
        
        // 数据传输到设备（使用传输流）
        cudaMemcpyAsync(d_input + offset, h_input + offset, 
                        batchSize * sizeof(float), 
                        cudaMemcpyHostToDevice, transferStream);
        
        // 等待传输完成后开始计算
        cudaStreamWaitEvent(computeStream, 0, 0);
        
        // 执行计算（使用计算流）
        computeIntensive<<<gridSize, blockSize, 0, computeStream>>>(
            d_input + offset, d_output + offset, batchSize, 1500);
        
        // 计算完成后传输结果回主机
        cudaMemcpyAsync(h_output + offset, d_output + offset, 
                        batchSize * sizeof(float), 
                        cudaMemcpyDeviceToHost, transferStream);
    }
    
    // 同步所有操作
    cudaStreamSynchronize(computeStream);
    cudaStreamSynchronize(transferStream);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed = measureTime(start, stop);
    printf("重叠执行总时间: %.2f ms\n", elapsed);
    
    // 清理
    cudaStreamDestroy(computeStream);
    cudaStreamDestroy(transferStream);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    printf("CUDA Stream 重叠技术演示\n");
    printf("========================\n");
    
    // 检查设备能力
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("设备: %s\n", prop.name);
    printf("支持并发内核执行: %s\n", prop.concurrentKernels ? "是" : "否");
    printf("支持重叠数据传输: %s\n", prop.deviceOverlap ? "是" : "否");
    printf("异步引擎数量: %d\n", prop.asyncEngineCount);
    printf("\n");
    
    const int dataSize = 4 * 1024 * 1024; // 4M 元素
    const int numStreams = 4;
    
    // 分配主机内存
    float *h_input = (float*)malloc(dataSize * sizeof(float));
    float *h_output1 = (float*)malloc(dataSize * sizeof(float));
    float *h_output2 = (float*)malloc(dataSize * sizeof(float));
    
    // 初始化数据
    initializeData(h_input, dataSize);
    
    printf("=== 性能比较：同步 vs 异步执行 ===\n");
    printf("数据大小: %d 元素\n", dataSize);
    printf("流数量: %d\n\n", numStreams);
    
    // 测试同步执行
    printf("1. 同步执行（无重叠）...\n");
    float seqTime = sequentialExecution(h_input, h_output1, dataSize, numStreams);
    printf("同步执行时间: %.2f ms\n\n", seqTime);
    
    // 测试异步重叠执行
    printf("2. 异步执行（重叠）...\n");
    float overlapTime = overlappedExecution(h_input, h_output2, dataSize, numStreams);
    printf("重叠执行时间: %.2f ms\n", overlapTime);
    
    // 计算加速比
    float speedup = seqTime / overlapTime;
    printf("加速比: %.2fx\n", speedup);
    printf("性能提升: %.1f%%\n", (speedup - 1) * 100);
    
    // 验证结果一致性
    bool resultsMatch = true;
    for (int i = 0; i < dataSize && resultsMatch; i++) {
        if (fabs(h_output1[i] - h_output2[i]) > 1e-5) {
            resultsMatch = false;
        }
    }
    printf("结果验证: %s\n", resultsMatch ? "一致" : "不一致");
    
    // 演示其他Stream技术
    demonstratePriority();
    demonstrateComputeTransferOverlap();
    
    printf("\n=== Stream重叠优化总结 ===\n");
    printf("1. 使用页锁定内存提高传输性能\n");
    printf("2. 多流并行可以重叠计算和数据传输\n");
    printf("3. 流优先级可以控制任务调度\n");
    printf("4. 合理分批处理可以隐藏传输延迟\n");
    printf("5. 事件同步可以精确控制流间依赖关系\n");
    
    // 清理内存
    free(h_input);
    free(h_output1);
    free(h_output2);
    
    return 0;
} 