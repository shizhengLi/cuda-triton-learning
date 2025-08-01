#ifndef FLASH_ATTENTION_COMMON_H
#define FLASH_ATTENTION_COMMON_H

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constants
constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 1024;
constexpr int MAX_SHARED_MEMORY = 48 * 1024; // 48KB per SM

// Data types
enum class DataType {
    FLOAT32,
    FLOAT16,
    BFLOAT16
};

// Flash Attention parameters
struct FlashAttentionParams {
    int batch_size;
    int seq_len;
    int head_dim;
    float scale;
    DataType data_type;
    
    // Block sizes
    int block_size_m;
    int block_size_n;
    int block_size_k;
    
    // Device pointers
    void* q_ptr;
    void* k_ptr;
    void* v_ptr;
    void* o_ptr;
    
    // Strides (in elements)
    size_t q_stride;
    size_t k_stride;
    size_t v_stride;
    size_t o_stride;
};

// Utility functions
inline int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

inline bool is_power_of_two(int x) {
    return (x & (x - 1)) == 0;
}

// Get data type size
inline size_t get_data_type_size(DataType dtype) {
    switch (dtype) {
        case DataType::FLOAT32:
            return sizeof(float);
        case DataType::FLOAT16:
            return sizeof(__half);
        case DataType::BFLOAT16:
            return sizeof(__nv_bfloat16);
        default:
            return sizeof(float);
    }
}

// Get max shared memory per block
inline int get_max_shared_memory_per_block() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    int max_shared_memory;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_memory, 
                                     cudaDevAttrMaxSharedMemoryPerBlock, 
                                     device));
    return max_shared_memory;
}

// Print device info
void print_device_info() {
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    printf("Device %d: %s\n", device, prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Max Threads Per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Shared Memory Per Block: %d KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Max Grid Size: %d x %d x %d\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Warp Size: %d\n", prop.warpSize);
    printf("  Memory Clock Rate: %.0f MHz\n", prop.memoryClockRate / 1000.0);
    printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
    printf("\n");
}

// Aligned memory allocation
void* aligned_alloc(size_t size, size_t alignment = 256) {
    void* ptr = nullptr;
    CUDA_CHECK(cudaMallocManaged(&ptr, size));
    return ptr;
}

// Aligned memory deallocation
void aligned_free(void* ptr) {
    CUDA_CHECK(cudaFree(ptr));
}

// Synchronization
void sync_device() {
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Timing utilities
class CudaTimer {
public:
    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    
    ~CudaTimer() {
        CUDA_CHECK(cudaEventDestroy(start_));
        CUDA_CHECK(cudaEventDestroy(stop_));
    }
    
    void start() {
        CUDA_CHECK(cudaEventRecord(start_));
    }
    
    void stop() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
    }
    
    float elapsed_milliseconds() {
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
    
private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};

// Memory usage utilities
size_t get_current_memory_usage() {
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return total - free;
}

void print_memory_usage(const char* label) {
    size_t free, total;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    size_t used = total - free;
    
    printf("%s: %.2f MB used, %.2f MB free, %.2f MB total\n", 
           label, 
           used / (1024.0 * 1024.0),
           free / (1024.0 * 1024.0),
           total / (1024.0 * 1024.0));
}

#endif // FLASH_ATTENTION_COMMON_H