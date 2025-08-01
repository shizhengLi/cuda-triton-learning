#!/bin/bash

# Build script for Flash Attention CUDA implementation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    print_error "CUDA compiler (nvcc) not found. Please install CUDA toolkit."
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | sed 's/,//')
print_status "CUDA version: $CUDA_VERSION"

# Create build directory
BUILD_DIR="build_cuda"
if [ -d "$BUILD_DIR" ]; then
    print_status "Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure with CMake
print_status "Configuring with CMake..."
if ! cmake .. -DCMAKE_BUILD_TYPE=Release; then
    print_error "CMake configuration failed"
    exit 1
fi

# Build the project
print_status "Building project..."
if ! make -j$(nproc); then
    print_error "Build failed"
    exit 1
fi

print_success "Build completed successfully"

# Run tests
print_status "Running Flash Attention v1 tests..."
if ./test_flash_v1; then
    print_success "All tests passed!"
else
    print_error "Some tests failed"
    exit 1
fi

cd ..

print_success "Flash Attention v1 CUDA implementation completed successfully!"
print_status "Check the test output above for detailed results."