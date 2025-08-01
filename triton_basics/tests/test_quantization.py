"""
Unit tests for quantization operators
"""

import pytest
import torch
import numpy as np
from typing import Tuple
import sys
import os

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '05_advanced', '01_quantization'))

from quantization_ops import FP8Quantizer, INT8Quantizer, quantization_error, signal_to_quantization_noise_ratio


class TestFP8Quantizer:
    """Test suite for FP8 quantization"""
    
    def test_fp8_quantizer_initialization(self):
        """Test FP8 quantizer initialization"""
        quantizer = FP8Quantizer(fp8_format="E5M2")
        assert quantizer.fp8_format == "E5M2"
        assert quantizer.scale is None
        
        quantizer = FP8Quantizer(fp8_format="E4M3")
        assert quantizer.fp8_format == "E4M3"
    
    def test_fp8_calibration(self):
        """Test FP8 scale calibration"""
        quantizer = FP8Quantizer(fp8_format="E5M2")
        
        # Test with uniform distribution
        x = torch.randn(1000)
        quantizer.calibrate(x)
        assert quantizer.scale is not None
        assert quantizer.scale > 0
        
        # Test with extreme values
        x_extreme = torch.tensor([100.0, -200.0, 50.0])
        quantizer.calibrate(x_extreme)
        expected_scale = 200.0 / 448.0  # max_abs / fp8_max
        assert torch.isclose(quantizer.scale, torch.tensor(expected_scale), rtol=1e-5)
    
    def test_fp8_quantize_dequantize(self):
        """Test FP8 quantization and dequantization cycle"""
        quantizer = FP8Quantizer(fp8_format="E5M2")
        
        # Test with random tensor
        x = torch.randn(1000) * 100  # Scale up to test quantization
        quantizer.calibrate(x)
        
        # Quantize and dequantize
        x_quant = quantizer.quantize(x)
        x_dequant = quantizer.dequantize(x_quant)
        
        # Check shapes
        assert x.shape == x_quant.shape
        assert x.shape == x_dequant.shape
        
        # Check data types
        assert x_quant.dtype == torch.float16
        assert x_dequant.dtype == torch.float32
        
        # Check quantization error is reasonable
        error = quantization_error(x, x_dequant)
        assert error < 1.0  # Should be relatively small for normal distribution
    
    def test_fp8_sqnr(self):
        """Test FP8 signal-to-quantization noise ratio"""
        quantizer = FP8Quantizer(fp8_format="E5M2")
        
        # Test with different signal amplitudes
        for amplitude in [10.0, 100.0, 1000.0]:
            x = torch.randn(1000) * amplitude
            quantizer.calibrate(x)
            
            x_quant = quantizer.quantize(x)
            x_dequant = quantizer.dequantize(x_quant)
            
            sqnr = signal_to_quantization_noise_ratio(x, x_dequant)
            assert sqnr > 20.0  # Should be at least 20 dB for reasonable quantization
    
    def test_fp8_edge_cases(self):
        """Test FP8 quantization edge cases"""
        quantizer = FP8Quantizer(fp8_format="E5M2")
        
        # Test with zeros
        x_zeros = torch.zeros(100)
        quantizer.calibrate(x_zeros)
        x_quant = quantizer.quantize(x_zeros)
        x_dequant = quantizer.dequantize(x_quant)
        assert torch.allclose(x_zeros, x_dequant, atol=1e-6)
        
        # Test with very small values
        x_small = torch.randn(100) * 1e-6
        quantizer.calibrate(x_small)
        x_quant = quantizer.quantize(x_small)
        x_dequant = quantizer.dequantize(x_quant)
        error = quantization_error(x_small, x_dequant)
        assert error < 1e-10  # Should be very small for small values


class TestINT8Quantizer:
    """Test suite for INT8 quantization"""
    
    def test_int8_quantizer_initialization(self):
        """Test INT8 quantizer initialization"""
        quantizer = INT8Quantizer(symmetric=True)
        assert quantizer.symmetric is True
        assert quantizer.scale is None
        assert quantizer.zero_point is None
        
        quantizer = INT8Quantizer(symmetric=False)
        assert quantizer.symmetric is False
    
    def test_int8_calibration_symmetric(self):
        """Test INT8 symmetric calibration"""
        quantizer = INT8Quantizer(symmetric=True)
        
        # Test with symmetric distribution
        x = torch.randn(1000)
        quantizer.calibrate(x)
        
        assert quantizer.scale is not None
        assert quantizer.zero_point is not None
        assert torch.isclose(quantizer.zero_point, torch.tensor(0.0))
        
        # Test calibration formula
        max_abs = torch.max(torch.abs(x))
        expected_scale = max_abs / 127.0
        assert torch.isclose(quantizer.scale, expected_scale, rtol=1e-5)
    
    def test_int8_calibration_asymmetric(self):
        """Test INT8 asymmetric calibration"""
        quantizer = INT8Quantizer(symmetric=False)
        
        # Test with asymmetric distribution
        x = torch.randn(1000) + 1.0  # Shift to make asymmetric
        quantizer.calibrate(x)
        
        assert quantizer.scale is not None
        assert quantizer.zero_point is not None
        
        # Test that zero point is not zero for asymmetric data
        assert not torch.isclose(quantizer.zero_point, torch.tensor(0.0), atol=1e-3)
    
    def test_int8_quantize_dequantize(self):
        """Test INT8 quantization and dequantization cycle"""
        quantizer = INT8Quantizer(symmetric=True)
        
        # Test with random tensor
        x = torch.randn(1000) * 100
        quantizer.calibrate(x)
        
        # Quantize and dequantize
        x_quant = quantizer.quantize(x)
        x_dequant = quantizer.dequantize(x_quant)
        
        # Check shapes
        assert x.shape == x_quant.shape
        assert x.shape == x_dequant.shape
        
        # Check data types
        assert x_quant.dtype == torch.int8
        assert x_dequant.dtype == torch.float32
        
        # Check quantization error
        error = quantization_error(x, x_dequant)
        assert error < 10.0  # Should be reasonable for INT8
    
    def test_int8_value_range(self):
        """Test INT8 quantization respects value range"""
        quantizer = INT8Quantizer(symmetric=True)
        
        # Test with extreme values
        x = torch.tensor([-1000.0, -500.0, 0.0, 500.0, 1000.0])
        quantizer.calibrate(x)
        
        x_quant = quantizer.quantize(x)
        
        # Check that quantized values are within INT8 range
        assert torch.all(x_quant >= -128)
        assert torch.all(x_quant <= 127)
    
    def test_int8_vs_fp8_comparison(self):
        """Compare INT8 and FP8 quantization quality"""
        # Generate test data
        x = torch.randn(1000) * 50
        
        # Test with symmetric quantization
        fp8_quantizer = FP8Quantizer(fp8_format="E5M2")
        int8_quantizer = INT8Quantizer(symmetric=True)
        
        fp8_quantizer.calibrate(x)
        int8_quantizer.calibrate(x)
        
        # Quantize and dequantize
        x_fp8_dequant = fp8_quantizer.dequantize(fp8_quantizer.quantize(x))
        x_int8_dequant = int8_quantizer.dequantize(int8_quantizer.quantize(x))
        
        # Compare errors
        fp8_error = quantization_error(x, x_fp8_dequant)
        int8_error = quantization_error(x, x_int8_dequant)
        
        # Both should have reasonable error
        assert fp8_error < 1.0
        assert int8_error < 10.0
        
        # Compare SQNR
        fp8_sqnr = signal_to_quantization_noise_ratio(x, x_fp8_dequant)
        int8_sqnr = signal_to_quantization_noise_ratio(x, x_int8_dequant)
        
        assert fp8_sqnr > 20.0
        assert int8_sqnr > 15.0


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_quantization_error(self):
        """Test quantization error calculation"""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.1, 2.1, 3.1])
        
        error = quantization_error(x, y)
        expected = torch.mean((x - y) ** 2).item()
        assert abs(error - expected) < 1e-6
    
    def test_sqnr_calculation(self):
        """Test SQNR calculation"""
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([1.01, 2.01, 3.01])  # Small noise
        
        sqnr = signal_to_quantization_noise_ratio(x, y)
        assert sqnr > 0  # Should be positive
        
        # Test with identical signals
        sqnr_perfect = signal_to_quantization_noise_ratio(x, x)
        assert np.isinf(sqnr_perfect)  # Should be infinite for perfect reconstruction


class TestQuantizationPerformance:
    """Performance tests for quantization operators"""
    
    def test_large_tensor_quantization(self):
        """Test quantization with large tensors"""
        quantizer = FP8Quantizer(fp8_format="E5M2")
        
        # Test with large tensor
        x = torch.randn(100000)  # 100K elements
        quantizer.calibrate(x)
        
        x_quant = quantizer.quantize(x)
        x_dequant = quantizer.dequantize(x_quant)
        
        # Check that it completed without error
        assert x_quant.shape == x.shape
        assert x_dequant.shape == x.shape
        
        # Check reasonable error
        error = quantization_error(x, x_dequant)
        assert error < 1.0
    
    def test_different_shapes(self):
        """Test quantization with different tensor shapes"""
        shapes = [(100,), (10, 10), (5, 5, 4), (2, 3, 4, 5)]
        
        for shape in shapes:
            quantizer = FP8Quantizer(fp8_format="E5M2")
            x = torch.randn(shape)
            quantizer.calibrate(x)
            
            x_quant = quantizer.quantize(x)
            x_dequant = quantizer.dequantize(x_quant)
            
            assert x_quant.shape == shape
            assert x_dequant.shape == shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])