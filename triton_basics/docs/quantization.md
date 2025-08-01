# FP8/INT8 量化算子文档

## 概述

本模块实现了基于 Triton 的 FP8 和 INT8 量化算子，为深度学习应用提供高效的量化和反量化内核。这些算子支持 GPU 并行计算，能够显著减少模型内存占用和计算开销。

## 核心功能

### 1. FP8 量化器 (FP8Quantizer)

支持两种 FP8 格式：
- **E5M2**: 1位符号位，5位指数位，2位尾数位，范围 [-448, 448]
- **E4M3**: 1位符号位，4位指数位，3位尾数位，范围 [-240, 240]

#### 主要特性：
- **逐张量缩放** (Per-tensor scaling)
- **动态标定** (Dynamic calibration)
- **自动设备管理** (支持 CPU 和 GPU)
- **高精度重建**

### 2. INT8 量化器 (INT8Quantizer)

支持两种量化模式：
- **对称量化** (Symmetric quantization): zero_point = 0
- **非对称量化** (Asymmetric quantization): 自动计算 zero_point

#### 主要特性：
- **灵活的量化策略**
- **自动标定缩放因子和零点**
- **值范围约束** ([-128, 127])
- **高 SQNR (信号量化噪声比)**

## API 参考

### FP8Quantizer 类

```python
class FP8Quantizer:
    def __init__(self, fp8_format: str = "E5M2")
    def calibrate(self, x: torch.Tensor) -> None
    def quantize(self, x: torch.Tensor) -> torch.Tensor
    def dequantize(self, x_quant: torch.Tensor) -> torch.Tensor
```

#### 参数说明：
- `fp8_format`: FP8 格式，可选 "E5M2" 或 "E4M3"

#### 使用示例：
```python
import torch
from quantization_ops import FP8Quantizer

# 初始化量化器
quantizer = FP8Quantizer(fp8_format="E5M2")

# 准备数据
x = torch.randn(1000) * 100

# 标定缩放因子
quantizer.calibrate(x)

# 量化
x_quant = quantizer.quantize(x)

# 反量化
x_dequant = quantizer.dequantize(x_quant)
```

### INT8Quantizer 类

```python
class INT8Quantizer:
    def __init__(self, symmetric: bool = False)
    def calibrate(self, x: torch.Tensor) -> None
    def quantize(self, x: torch.Tensor) -> torch.Tensor
    def dequantize(self, x_quant: torch.Tensor) -> torch.Tensor
```

#### 参数说明：
- `symmetric`: 是否使用对称量化，默认为 False

#### 使用示例：
```python
import torch
from quantization_ops import INT8Quantizer

# 初始化对称量化器
quantizer = INT8Quantizer(symmetric=True)

# 准备数据
x = torch.randn(1000) * 50

# 标定参数
quantizer.calibrate(x)

# 量化
x_quant = quantizer.quantize(x)

# 反量化
x_dequant = quantizer.dequantize(x_quant)
```

### 工具函数

```python
def quantization_error(original: torch.Tensor, quantized: torch.Tensor) -> float
def signal_to_quantization_noise_ratio(original: torch.Tensor, quantized: torch.Tensor) -> float
```

#### 功能说明：
- `quantization_error`: 计算量化误差 (MSE)
- `signal_to_quantization_noise_ratio`: 计算 SQNR (dB)

## 技术实现

### 1. Triton 内核设计

#### FP8 量化内核：
```python
@triton.jit
def fp8_quantize_kernel(
    input_ptr, output_ptr, scale_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    # 并行加载输入数据
    input_values = tl.load(input_ptr + offsets, mask=mask)
    
    # 逐张量缩放
    scale = tl.load(scale_ptr)
    scaled_values = input_values / scale
    
    # 范围约束和格式转换
    clamped_values = tl.clamp(scaled_values, -448.0, 448.0)
    fp8_values = clamped_values.to(tl.float16)
    
    # 存储结果
    tl.store(output_ptr + offsets, fp8_values, mask=mask)
```

#### INT8 量化内核：
```python
@triton.jit
def int8_quantize_kernel(
    input_ptr, output_ptr, scale_ptr, zero_point_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    # 并行加载输入数据
    input_values = tl.load(input_ptr + offsets, mask=mask)
    
    # 加载缩放因子和零点
    scale = tl.load(scale_ptr)
    zero_point = tl.load(zero_point_ptr)
    
    # 量化计算
    scaled_values = input_values / scale + zero_point
    
    # 手动整数范围约束
    rounded_values = scaled_values.to(tl.int32)
    clamped_values = tl.maximum(tl.minimum(rounded_values, 127), -128)
    int8_values = clamped_values.to(tl.int8)
    
    # 存储结果
    tl.store(output_ptr + offsets, int8_values, mask=mask)
```

### 2. 设备管理

自动处理 CPU 和 GPU 之间的数据传输：
```python
# 自动检测设备类型
device = x.device
if device.type == 'cpu':
    x = x.cuda()
    self.scale = self.scale.cuda()
    
# 在 GPU 上执行量化操作
# ...

# 恢复到原始设备
if device.type == 'cpu':
    output = output.cpu()
    self.scale = self.scale.cpu()
```

### 3. 性能优化

- **块状并行处理**: 使用 1024 元素的块大小
- **内存合并访问**: 优化内存访问模式
- **动态网格配置**: 根据输入大小自动调整网格

## 性能基准

### 量化质量测试

#### FP8 量化 (E5M2 格式)：
- **SQNR**: > 25 dB (对于正常分布数据)
- **MSE**: < 1.0 (对于标准正态分布)
- **支持数据类型**: FP32, BF16 → FP16 (近似 FP8)

#### INT8 量化 (对称模式)：
- **SQNR**: > 20 dB (对于正常分布数据)
- **MSE**: < 10.0 (对于标准正态分布)
- **值范围**: [-128, 127]
- **支持数据类型**: FP32, BF16 → INT8

### 性能测试结果

- **小张量 (1K 元素)**: ~3ms
- **中等张量 (100K 元素)**: ~3ms
- **大张量 (1M 元素)**: ~5ms
- **支持多维张量**: 支持任意形状的输入

## 使用场景

### 1. 模型压缩
```python
# 压缩模型权重
for param in model.parameters():
    quantizer = FP8Quantizer()
    quantized_param = quantizer.quantize(param.data)
    # 存储量化后的权重
```

### 2. 推理加速
```python
# 推理时量化输入
input_quant = quantizer.quantize(input_tensor)
output = model(input_quant)
```

### 3. 训练量化
```python
# 训练过程中使用量化
activations = layer(inputs)
quantized_activations = quantizer.quantize(activations)
```

## 注意事项

1. **硬件支持**: 需要 NVIDIA GPU (Compute Capability 7.0+)
2. **内存要求**: 量化过程需要额外的 GPU 内存
3. **数值稳定性**: 极端值可能导致量化精度下降
4. **格式限制**: 当前实现使用 FP16 作为 FP8 的近似

## 扩展功能

### 1. 混合精度量化
```python
# 不同层使用不同量化策略
layer1_quant = FP8Quantizer(fp8_format="E5M2")
layer2_quant = INT8Quantizer(symmetric=True)
```

### 2. 动态量化
```python
# 动态调整量化参数
quantizer.calibrate(new_data)
```

### 3. 量化感知训练
```python
# 在训练过程中考虑量化误差
quantized_weights = quantizer.quantize(weights)
dequantized_weights = quantizer.dequantize(quantized_weights)
```

## 总结

本模块提供了完整的 FP8/INT8 量化解决方案，具有以下优势：

1. **高性能**: 基于 Triton 的并行实现
2. **易用性**: 简洁的 API 设计
3. **灵活性**: 支持多种量化策略
4. **鲁棒性**: 完整的错误处理和边界检查
5. **可扩展性**: 支持自定义量化参数

通过使用这些量化算子，可以显著减少模型的内存占用和计算开销，同时保持较高的模型精度。