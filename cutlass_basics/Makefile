# CuTLASS 示例编译 Makefile
# 
# 使用说明:
# 1. 确保已安装 CuTLASS 3.x
# 2. 设置正确的路径
# 3. 运行 make 编译

# 编译器设置
NVCC = nvcc
CXX = g++

# CuTLASS 路径 (需要根据实际安装路径修改)
CUTLASS_ROOT = /path/to/cutlass
CUTLASS_INCLUDE = $(CUTLASS_ROOT)/include
CUTLASS_TOOLS = $(CUTLASS_ROOT)/tools/util/include

# 编译标志
NVCC_FLAGS = -std=c++17 -arch=sm_80 -O3
INCLUDES = -I$(CUTLASS_INCLUDE) -I$(CUTLASS_TOOLS)

# 目标文件
TARGETS = gemm_basic cute_tensor tma_demo

# 默认目标
all: $(TARGETS)

# 基础 GEMM 示例
gemm_basic: 01_gemm_basic.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< -o $@

# CuTe 张量操作示例  
cute_tensor: 02_cute_tensor.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< -o $@

# TMA 内存操作示例
tma_demo: 03_tma_demo.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $< -o $@

# 清理
clean:
	rm -f $(TARGETS)

# 运行测试
test: all
	@echo "运行 CuTLASS 基础 GEMM 测试..."
	./gemm_basic
	@echo "\n运行 CuTe 张量操作测试..."
	./cute_tensor
	@echo "\n运行 TMA 演示..."
	./tma_demo

# 检查 CuTLASS 安装
check:
	@echo "检查 CuTLASS 安装..."
	@if [ -d "$(CUTLASS_ROOT)" ]; then \
		echo "✅ CuTLASS 路径找到: $(CUTLASS_ROOT)"; \
	else \
		echo "❌ CuTLASS 路径未找到: $(CUTLASS_ROOT)"; \
		echo "请修改 Makefile 中的 CUTLASS_ROOT 路径"; \
	fi

.PHONY: all clean test check 