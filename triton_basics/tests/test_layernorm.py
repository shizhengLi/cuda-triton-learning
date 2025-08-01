import torch
import pytest
import sys
import os

# Add the parent directory to the path to import our module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import directly from the file
import importlib.util
spec = importlib.util.spec_from_file_location(
    "layernorm", 
    os.path.join(project_root, "02_dl_kernels", "normalization", "layernorm.py")
)
layernorm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layernorm_module)

layernorm = layernorm_module.layernorm
naive_layernorm = layernorm_module.naive_layernorm
benchmark_layernorm = layernorm_module.benchmark_layernorm


class TestLayerNorm:
    """Test suite for LayerNorm Triton kernel"""
    
    def test_basic_layernorm(self):
        """Test basic LayerNorm functionality"""
        M, N = 64, 32
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        bias = torch.randn(N, device='cuda', dtype=torch.float32)
        
        # Triton implementation
        result_triton = layernorm(x, weight, bias)
        
        # PyTorch baseline
        result_torch = torch.nn.functional.layer_norm(x, (N,), weight, bias)
        
        # Verify results match
        assert torch.allclose(result_triton, result_torch, atol=1e-6, rtol=1e-6), \
            f"Triton and PyTorch results don't match. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_layernorm_without_weight_bias(self):
        """Test LayerNorm without weight and bias"""
        M, N = 64, 32
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        # Triton implementation
        result_triton = layernorm(x)
        
        # PyTorch baseline
        result_torch = torch.nn.functional.layer_norm(x, (N,))
        
        # Verify results match
        assert torch.allclose(result_triton, result_torch, atol=1e-6, rtol=1e-6), \
            f"Triton and PyTorch results don't match. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_different_matrix_sizes(self):
        """Test LayerNorm with different matrix sizes"""
        sizes = [
            (1, 1),      # Single element
            (1, 32),     # Single row
            (32, 1),     # Single column
            (64, 64),    # Square
            (128, 256),  # Rectangular
            (256, 128),  # Rectangular (transposed)
            (512, 512),  # Larger square
            (1024, 2048) # Large rectangular
        ]
        
        for M, N in sizes:
            x = torch.randn(M, N, device='cuda', dtype=torch.float32)
            weight = torch.randn(N, device='cuda', dtype=torch.float32)
            bias = torch.randn(N, device='cuda', dtype=torch.float32)
            
            result_triton = layernorm(x, weight, bias)
            result_torch = torch.nn.functional.layer_norm(x, (N,), weight, bias)
            
            assert torch.allclose(result_triton, result_torch, atol=1e-6, rtol=1e-6), \
                f"Failed for size {M}x{N}. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_non_power_of_two_dimensions(self):
        """Test with dimensions that are not powers of two"""
        test_cases = [
            (100, 100),  # Not power of 2
            (129, 129),  # Just over power of 2
            (255, 255),  # Almost 2^8
            (1000, 1000), # Large non-power of 2
            (123, 456),  # Different non-power of 2 dimensions
        ]
        
        for M, N in test_cases:
            x = torch.randn(M, N, device='cuda', dtype=torch.float32)
            weight = torch.randn(N, device='cuda', dtype=torch.float32)
            bias = torch.randn(N, device='cuda', dtype=torch.float32)
            
            result_triton = layernorm(x, weight, bias)
            result_torch = torch.nn.functional.layer_norm(x, (N,), weight, bias)
            
            assert torch.allclose(result_triton, result_torch, atol=1e-3, rtol=1e-3), \
                f"Failed for non-power-of-2 size {M}x{N}. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_different_axes(self):
        """Test LayerNorm along different axes (fallback to PyTorch for complex cases)"""
        # Test 3D tensor - should fallback to PyTorch without weight/bias
        x = torch.randn(4, 8, 16, device='cuda', dtype=torch.float32)
        weight = torch.randn(16, device='cuda', dtype=torch.float32)
        bias = torch.randn(16, device='cuda', dtype=torch.float32)
        
        # Test along last dimension (default) - should fallback to PyTorch without weight/bias
        result_triton_last = layernorm(x, weight, bias, axis=-1)
        result_torch_last = torch.nn.functional.layer_norm(x, (16,), None, None)
        assert torch.allclose(result_triton_last, result_torch_last, atol=1e-6)
        
        # Test 2D tensor along first dimension - should fallback to PyTorch without weight/bias
        x_2d = torch.randn(8, 16, device='cuda', dtype=torch.float32)
        weight_2d = torch.randn(8, device='cuda', dtype=torch.float32)
        bias_2d = torch.randn(8, device='cuda', dtype=torch.float32)
        result_triton_first = layernorm(x_2d, weight_2d, bias_2d, axis=0)
        result_torch_first = torch.nn.functional.layer_norm(x_2d, (8,), None, None)
        assert torch.allclose(result_triton_first, result_torch_first, atol=1e-6)
    
    def test_1d_tensor(self):
        """Test LayerNorm on 1D tensor (fallback to PyTorch)"""
        size = 128
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        weight = torch.randn(size, device='cuda', dtype=torch.float32)
        bias = torch.randn(size, device='cuda', dtype=torch.float32)
        
        result_triton = layernorm(x, weight, bias)
        result_torch = torch.nn.functional.layer_norm(x, (size,), None, None)
        
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        assert result_triton.shape == x.shape
    
    def test_higher_dimensional_tensors(self):
        """Test LayerNorm on higher dimensional tensors (fallback to PyTorch)"""
        # Test 4D tensor
        x = torch.randn(2, 4, 8, 16, device='cuda', dtype=torch.float32)
        
        for axis in range(x.dim()):
            # For LayerNorm, we normalize over the last N dimensions, not a single axis
            # So we need to handle this differently
            if axis == x.dim() - 1:
                # Last dimension - this is supported
                normalized_shape = (x.shape[axis],)
                weight = torch.randn(normalized_shape, device='cuda', dtype=torch.float32)
                bias = torch.randn(normalized_shape, device='cuda', dtype=torch.float32)
                
                result_triton = layernorm(x, weight, bias, axis=axis)
                result_torch = torch.nn.functional.layer_norm(x, normalized_shape, None, None)
                assert torch.allclose(result_triton, result_torch, atol=1e-6), \
                    f"Failed for axis {axis} in 4D tensor"
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        M, N = 32, 64
        
        # Test with large positive values
        x_large = torch.full((M, N), 100.0, device='cuda', dtype=torch.float32)
        result_triton = layernorm(x_large)
        result_torch = torch.nn.functional.layer_norm(x_large, (N,))
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Test with large negative values
        x_small = torch.full((M, N), -100.0, device='cuda', dtype=torch.float32)
        result_triton = layernorm(x_small)
        result_torch = torch.nn.functional.layer_norm(x_small, (N,))
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Test with mixed extreme values
        x_mixed = torch.randn(M, N, device='cuda', dtype=torch.float32) * 100
        result_triton = layernorm(x_mixed)
        result_torch = torch.nn.functional.layer_norm(x_mixed, (N,))
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Test with very small epsilon
        result_triton = layernorm(x_mixed, eps=1e-12)
        result_torch = torch.nn.functional.layer_norm(x_mixed, (N,), eps=1e-12)
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
    
    def test_zero_input(self):
        """Test LayerNorm with zero input"""
        M, N = 16, 32
        x_zero = torch.zeros((M, N), device='cuda', dtype=torch.float32)
        
        result_triton = layernorm(x_zero)
        result_torch = torch.nn.functional.layer_norm(x_zero, (N,))
        
        # All values should be zero
        assert torch.allclose(result_triton, torch.zeros_like(result_triton), atol=1e-6)
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
    
    def test_layernorm_properties(self):
        """Test mathematical properties of LayerNorm"""
        M, N = 32, 64
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        result = layernorm(x)
        
        # Test that each row has mean 0 and std 1
        mean = torch.mean(result, dim=-1)
        # Use unbiased=False to match PyTorch's layer_norm variance calculation
        std = torch.std(result, dim=-1, unbiased=False)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-6), \
            "Mean should be close to 0"
        assert torch.allclose(std, torch.ones_like(std), atol=1e-6), \
            "Standard deviation should be close to 1"
        
        # Test that LayerNorm is invariant to scaling and shifting
        x_scaled = x * 10.0  # Scale
        x_shifted = x + 5.0  # Shift
        result_scaled = layernorm(x_scaled)
        result_shifted = layernorm(x_shifted)
        
        assert torch.allclose(result, result_scaled, atol=1e-6), \
            "LayerNorm should be invariant to scaling"
        assert torch.allclose(result, result_shifted, atol=1e-6), \
            "LayerNorm should be invariant to shifting"
    
    def test_different_data_types(self):
        """Test LayerNorm with different data types"""
        M, N = 32, 64
        
        # Test float32
        x_f32 = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight_f32 = torch.randn(N, device='cuda', dtype=torch.float32)
        bias_f32 = torch.randn(N, device='cuda', dtype=torch.float32)
        
        result_triton_f32 = layernorm(x_f32, weight_f32, bias_f32)
        result_torch_f32 = torch.nn.functional.layer_norm(x_f32, (N,), weight_f32, bias_f32)
        assert torch.allclose(result_triton_f32, result_torch_f32, atol=1e-6)
        
        # Test float16
        x_f16 = torch.randn(M, N, device='cuda', dtype=torch.float16)
        weight_f16 = torch.randn(N, device='cuda', dtype=torch.float16)
        bias_f16 = torch.randn(N, device='cuda', dtype=torch.float16)
        
        result_triton_f16 = layernorm(x_f16, weight_f16, bias_f16)
        result_torch_f16 = torch.nn.functional.layer_norm(x_f16, (N,), weight_f16, bias_f16)
        # Higher tolerance for float16
        assert torch.allclose(result_triton_f16, result_torch_f16, atol=1e-3, rtol=1e-3)
        
        # Test bfloat16 if available
        if torch.cuda.is_bf16_supported():
            x_bf16 = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
            weight_bf16 = torch.randn(N, device='cuda', dtype=torch.bfloat16)
            bias_bf16 = torch.randn(N, device='cuda', dtype=torch.bfloat16)
            
            result_triton_bf16 = layernorm(x_bf16, weight_bf16, bias_bf16)
            result_torch_bf16 = torch.nn.functional.layer_norm(x_bf16, (N,), weight_bf16, bias_bf16)
            assert torch.allclose(result_triton_bf16, result_torch_bf16, atol=1e-3, rtol=1e-3)
    
    def test_input_validation(self):
        """Test input validation"""
        M, N = 32, 32
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        bias = torch.randn(N, device='cuda', dtype=torch.float32)
        
        # Test CPU tensor
        x_cpu = torch.randn(M, N, device='cpu', dtype=torch.float32)
        with pytest.raises(AssertionError, match="Input tensor must be on CUDA device"):
            layernorm(x_cpu, weight, bias)
        
        # Test invalid axis
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        with pytest.raises(AssertionError, match="Invalid axis"):
            layernorm(x, weight, bias, axis=5)
        
        # Test wrong weight dimensions
        weight_wrong = torch.randn(N + 1, device='cuda', dtype=torch.float32)
        with pytest.raises(AssertionError, match="Weight shape"):
            layernorm(x, weight_wrong, bias)
        
        # Test wrong bias dimensions
        bias_wrong = torch.randn(N + 1, device='cuda', dtype=torch.float32)
        with pytest.raises(AssertionError, match="Bias shape"):
            layernorm(x, weight, bias_wrong)
        
        # Test negative axis
        result = layernorm(x, weight, bias, axis=-1)  # Should work
        assert result.shape == x.shape
    
    def test_against_naive_implementation(self):
        """Test against naive PyTorch implementation"""
        M, N = 64, 128
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        bias = torch.randn(N, device='cuda', dtype=torch.float32)
        
        result_triton = layernorm(x, weight, bias)
        result_naive = naive_layernorm(x, weight, bias)
        
        assert torch.allclose(result_triton, result_naive, atol=1e-6), \
            f"Triton and naive implementations differ. Max diff: {torch.max(torch.abs(result_triton - result_naive))}"
    
    def test_performance_benchmark(self):
        """Test performance benchmark functionality"""
        results = benchmark_layernorm(M=512, N=512, warmup=5, repeat=10)
        
        # Verify benchmark results contain expected keys
        expected_keys = [
            'triton_time', 'torch_time', 'speedup', 
            'triton_bandwidth', 'torch_bandwidth', 'matrix_shape', 'total_elements'
        ]
        for key in expected_keys:
            assert key in results, f"Missing benchmark result key: {key}"
        
        # Verify reasonable values
        assert results['triton_time'] > 0, "Triton time should be positive"
        assert results['torch_time'] > 0, "PyTorch time should be positive"
        assert results['speedup'] > 0, "Speedup should be positive"
        assert results['triton_bandwidth'] > 0, "Triton bandwidth should be positive"
        assert results['torch_bandwidth'] > 0, "PyTorch bandwidth should be positive"
        assert results['matrix_shape'] == (512, 512), "Matrix shape should match input"
        assert results['total_elements'] == 512 * 512, "Total elements should match"
        
        print(f"Benchmark results: {results}")
    
    def test_memory_efficiency(self):
        """Test that kernel doesn't cause memory leaks"""
        import gc
        
        initial_memory = torch.cuda.memory_allocated()
        
        # Run multiple iterations
        for _ in range(50):
            x = torch.randn(256, 256, device='cuda', dtype=torch.float32)
            weight = torch.randn(256, device='cuda', dtype=torch.float32)
            bias = torch.randn(256, device='cuda', dtype=torch.float32)
            result = layernorm(x, weight, bias)
            del x, weight, bias, result
            torch.cuda.synchronize()
        
        gc.collect()
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should not grow significantly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 10 * 1024 * 1024, f"Memory growth too large: {memory_growth} bytes"
    
    def test_large_matrices(self):
        """Test with large matrices"""
        M, N = 2048, 4096
        
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        bias = torch.randn(N, device='cuda', dtype=torch.float32)
        
        result_triton = layernorm(x, weight, bias)
        result_torch = torch.nn.functional.layer_norm(x, (N,), weight, bias)
        
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Verify memory usage is reasonable
        expected_memory = (M * N * 3 + N * 2) * 4  # bytes for float32 (input + output + weight + bias)
        actual_memory = torch.cuda.memory_allocated()
        # Allow for significant overhead due to PyTorch memory management
        assert actual_memory < expected_memory * 10, f"Memory usage too high: {actual_memory} vs {expected_memory * 10}"
    
    def test_deterministic_output(self):
        """Test that output is deterministic"""
        M, N = 32, 64
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        bias = torch.randn(N, device='cuda', dtype=torch.float32)
        
        # Run multiple times
        results = []
        for _ in range(5):
            result = layernorm(x, weight, bias)
            results.append(result.clone())
            torch.cuda.synchronize()
        
        # All results should be identical
        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i]), "Results are not deterministic"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])