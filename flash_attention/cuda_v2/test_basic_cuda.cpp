#include <iostream>
#include <cuda_runtime.h>

int main() {
    std::cout << "Flash Attention v2 CUDA Test - Basic CUDA" << std::endl;
    
    // Test basic CUDA functionality
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    if (device_count == 0) {
        std::cout << "No CUDA devices found" << std::endl;
        return 1;
    }
    
    std::cout << "Found " << device_count << " CUDA devices" << std::endl;
    
    // Test device properties
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    
    if (err != cudaSuccess) {
        std::cout << "Failed to get device properties: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    std::cout << "Device 0: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
    
    // Test basic memory allocation
    int *d_test;
    err = cudaMalloc(&d_test, sizeof(int));
    
    if (err != cudaSuccess) {
        std::cout << "Failed to allocate memory: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    
    std::cout << "Basic CUDA functionality test completed successfully!" << std::endl;
    
    // Cleanup
    cudaFree(d_test);
    
    return 0;
}