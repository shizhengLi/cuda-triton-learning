cmake_minimum_required(VERSION 3.18)
project(FlashAttentionCUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86) # Turing, Ampere, Lovelace

# Enable CUDA
enable_language(CUDA)

# Include directories
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../cuda_common)

# CUDA compilation flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Find Python for testing
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Testing
enable_testing()

# Add subdirectories
add_subdirectory(cuda_v1)
add_subdirectory(cuda_v2)

# Configure tests
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_runner.py.in"
    "${CMAKE_BINARY_DIR}/test_runner.py"
    @ONLY
)