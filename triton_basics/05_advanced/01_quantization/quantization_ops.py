"""
FP8/INT8 Quantization Operators in Triton

This module implements quantization operators for FP8 and INT8 data types,
providing efficient quantization and dequantization kernels for deep learning applications.
"""

import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


@triton.jit
def fp8_quantize_kernel(
    input_ptr, output_ptr, scale_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    """
    FP8 quantization kernel
    Quantizes FP32/BF16 values to FP8 with per-tensor scaling
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    input_values = tl.load(input_ptr + offsets, mask=mask)
    
    # Load scale (per-tensor)
    scale = tl.load(scale_ptr)
    
    # Quantize to FP8 (E5M2 format)
    scaled_values = input_values / scale
    
    # Clamp to FP8 range (-448, 448)
    clamped_values = tl.clamp(scaled_values, -448.0, 448.0)
    
    # Convert to FP8 (simplified - in practice would use hardware FP8)
    # For now, use FP16 as approximation
    fp8_values = clamped_values.to(tl.float16)
    
    # Store output
    tl.store(output_ptr + offsets, fp8_values, mask=mask)


@triton.jit
def fp8_dequantize_kernel(
    input_ptr, output_ptr, scale_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    """
    FP8 dequantization kernel
    Dequantizes FP8 values back to FP32/BF16
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load FP8 values (as FP16 for now)
    fp8_values = tl.load(input_ptr + offsets, mask=mask)
    
    # Load scale
    scale = tl.load(scale_ptr)
    
    # Dequantize
    output_values = fp8_values * scale
    
    # Store output
    tl.store(output_ptr + offsets, output_values, mask=mask)


@triton.jit
def int8_quantize_kernel(
    input_ptr, output_ptr, scale_ptr, zero_point_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    """
    INT8 quantization kernel
    Quantizes FP32/BF16 values to INT8 with scaling and zero point
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    input_values = tl.load(input_ptr + offsets, mask=mask)
    
    # Load scale and zero point
    scale = tl.load(scale_ptr)
    zero_point = tl.load(zero_point_ptr)
    
    # Quantize to INT8
    scaled_values = input_values / scale + zero_point
    
    # Round and clamp to INT8 range (manual clamping for integers)
    rounded_values = scaled_values.to(tl.int32)
    clamped_values = tl.maximum(tl.minimum(rounded_values, 127), -128)
    int8_values = clamped_values.to(tl.int8)
    
    # Store output
    tl.store(output_ptr + offsets, int8_values, mask=mask)


@triton.jit
def int8_dequantize_kernel(
    input_ptr, output_ptr, scale_ptr, zero_point_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    """
    INT8 dequantization kernel
    Dequantizes INT8 values back to FP32/BF16
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load INT8 values
    int8_values = tl.load(input_ptr + offsets, mask=mask)
    
    # Load scale and zero point
    scale = tl.load(scale_ptr)
    zero_point = tl.load(zero_point_ptr)
    
    # Dequantize
    output_values = (int8_values.to(tl.float32) - zero_point) * scale
    
    # Store output
    tl.store(output_ptr + offsets, output_values, mask=mask)


class FP8Quantizer:
    """FP8 quantization with per-tensor scaling"""
    
    def __init__(self, fp8_format: str = "E5M2"):
        self.fp8_format = fp8_format
        self.scale = None
    
    def calibrate(self, x: torch.Tensor) -> None:
        """Calibrate scale factor based on input tensor statistics"""
        if self.fp8_format == "E5M2":
            max_val = 448.0  # FP8 E5M2 max value
        else:  # E4M3
            max_val = 240.0  # FP8 E4M3 max value
        
        # Calculate scale as max(abs(x)) / max_val
        self.scale = torch.max(torch.abs(x)) / max_val
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input tensor to FP8"""
        if self.scale is None:
            self.calibrate(x)
        
        # Move tensors to GPU if available
        device = x.device
        if device.type == 'cpu':
            x = x.cuda()
            self.scale = self.scale.cuda()
        
        output = torch.empty_like(x, dtype=torch.float16)
        n_elements = x.numel()
        
        # Configure grid
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        # Launch kernel
        fp8_quantize_kernel[grid](
            x, output, self.scale,
            n_elements, BLOCK_SIZE=1024
        )
        
        # Move back to original device
        if device.type == 'cpu':
            output = output.cpu()
            self.scale = self.scale.cpu()
        
        return output
    
    def dequantize(self, x_quant: torch.Tensor) -> torch.Tensor:
        """Dequantize FP8 tensor back to original precision"""
        # Move tensors to GPU if available
        device = x_quant.device
        if device.type == 'cpu':
            x_quant = x_quant.cuda()
            self.scale = self.scale.cuda()
        
        output = torch.empty_like(x_quant, dtype=torch.float32)
        n_elements = x_quant.numel()
        
        # Configure grid
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        # Launch kernel
        fp8_dequantize_kernel[grid](
            x_quant, output, self.scale,
            n_elements, BLOCK_SIZE=1024
        )
        
        # Move back to original device
        if device.type == 'cpu':
            output = output.cpu()
            self.scale = self.scale.cpu()
        
        return output


class INT8Quantizer:
    """INT8 quantization with scaling and zero point"""
    
    def __init__(self, symmetric: bool = False):
        self.symmetric = symmetric
        self.scale = None
        self.zero_point = None
    
    def calibrate(self, x: torch.Tensor) -> None:
        """Calibrate scale and zero point based on input tensor statistics"""
        if self.symmetric:
            # Symmetric quantization: zero_point = 0
            max_val = torch.max(torch.abs(x))
            self.scale = max_val / 127.0
            self.zero_point = torch.tensor(0.0, device=x.device)
        else:
            # Asymmetric quantization
            min_val = torch.min(x)
            max_val = torch.max(x)
            
            qmin = -128.0
            qmax = 127.0
            
            self.scale = (max_val - min_val) / (qmax - qmin)
            self.zero_point = qmin - min_val / self.scale
    
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input tensor to INT8"""
        if self.scale is None:
            self.calibrate(x)
        
        # Move tensors to GPU if available
        device = x.device
        if device.type == 'cpu':
            x = x.cuda()
            self.scale = self.scale.cuda()
            self.zero_point = self.zero_point.cuda()
        
        output = torch.empty_like(x, dtype=torch.int8)
        n_elements = x.numel()
        
        # Configure grid
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        # Launch kernel
        int8_quantize_kernel[grid](
            x, output, self.scale, self.zero_point,
            n_elements, BLOCK_SIZE=1024
        )
        
        # Move back to original device
        if device.type == 'cpu':
            output = output.cpu()
            self.scale = self.scale.cpu()
            self.zero_point = self.zero_point.cpu()
        
        return output
    
    def dequantize(self, x_quant: torch.Tensor) -> torch.Tensor:
        """Dequantize INT8 tensor back to original precision"""
        # Move tensors to GPU if available
        device = x_quant.device
        if device.type == 'cpu':
            x_quant = x_quant.cuda()
            self.scale = self.scale.cuda()
            self.zero_point = self.zero_point.cuda()
        
        output = torch.empty_like(x_quant, dtype=torch.float32)
        n_elements = x_quant.numel()
        
        # Configure grid
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        # Launch kernel
        int8_dequantize_kernel[grid](
            x_quant, output, self.scale, self.zero_point,
            n_elements, BLOCK_SIZE=1024
        )
        
        # Move back to original device
        if device.type == 'cpu':
            output = output.cpu()
            self.scale = self.scale.cpu()
            self.zero_point = self.zero_point.cpu()
        
        return output


def quantization_error(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Calculate quantization error (MSE)"""
    return torch.mean((original - quantized) ** 2).item()


def signal_to_quantization_noise_ratio(original: torch.Tensor, quantized: torch.Tensor) -> float:
    """Calculate SQNR in dB"""
    signal_power = torch.mean(original ** 2)
    noise_power = torch.mean((original - quantized) ** 2)
    return 10 * torch.log10(signal_power / noise_power).item()