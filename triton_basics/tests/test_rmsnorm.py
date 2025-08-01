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
    "rmsnorm", 
    os.path.join(project_root, "02_dl_kernels", "normalization", "rmsnorm.py")
)
rmsnorm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rmsnorm_module)

rmsnorm = rmsnorm_module.rmsnorm
naive_rmsnorm = rmsnorm_module.naive_rmsnorm
benchmark_rmsnorm = rmsnorm_module.benchmark_rmsnorm


class TestRMSNorm:
    """Test suite for RMSNorm Triton kernel"""
    
    def test_basic_rmsnorm(self):
        """Test basic RMSNorm functionality"""
        M, N = 64, 32
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        
        # Triton implementation
        result_triton = rmsnorm(x, weight)
        
        # PyTorch baseline
        result_torch = rmsnorm_module._rmsnorm_torch(x, weight)
        
        # Verify results match
        assert torch.allclose(result_triton, result_torch, atol=1e-6, rtol=1e-6), \
            f"Triton and PyTorch results don't match. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_rmsnorm_without_weight(self):
        """Test RMSNorm without weight"""
        M, N = 64, 32
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        # Triton implementation
        result_triton = rmsnorm(x)
        
        # PyTorch baseline
        result_torch = rmsnorm_module._rmsnorm_torch(x)
        
        # Verify results match
        assert torch.allclose(result_triton, result_torch, atol=1e-6, rtol=1e-6), \
            f"Triton and PyTorch results don't match. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_different_matrix_sizes(self):
        """Test RMSNorm with different matrix sizes"""
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
            
            result_triton = rmsnorm(x, weight)
            result_torch = rmsnorm_module._rmsnorm_torch(x, weight)
            
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
            
            result_triton = rmsnorm(x, weight)
            result_torch = rmsnorm_module._rmsnorm_torch(x, weight)
            
            assert torch.allclose(result_triton, result_torch, atol=1e-5, rtol=1e-5), \
                f"Failed for non-power-of-2 size {M}x{N}. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_different_axes(self):
        """Test RMSNorm along different axes (fallback to PyTorch for complex cases)"""
        # Test 3D tensor - should fallback to PyTorch without weight/bias
        x = torch.randn(4, 8, 16, device='cuda', dtype=torch.float32)
        weight = torch.randn(16, device='cuda', dtype=torch.float32)
        
        # Test along last dimension (default) - should fallback to PyTorch without weight/bias
        result_triton_last = rmsnorm(x, weight, axis=-1)
        result_torch_last = rmsnorm_module._rmsnorm_torch(x, None, axis=-1)
        assert torch.allclose(result_triton_last, result_torch_last, atol=1e-6)
        
        # Test 2D tensor along first dimension - should fallback to PyTorch without weight/bias
        x_2d = torch.randn(8, 16, device='cuda', dtype=torch.float32)
        weight_2d = torch.randn(8, device='cuda', dtype=torch.float32)
        result_triton_first = rmsnorm(x_2d, weight_2d, axis=0)
        result_torch_first = rmsnorm_module._rmsnorm_torch(x_2d, None, axis=0)
        assert torch.allclose(result_triton_first, result_torch_first, atol=1e-6)
    
    def test_1d_tensor(self):
        """Test RMSNorm on 1D tensor (fallback to PyTorch)"""
        size = 128
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        weight = torch.randn(size, device='cuda', dtype=torch.float32)
        
        result_triton = rmsnorm(x, weight)
        result_torch = rmsnorm_module._rmsnorm_torch(x, None)
        
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        assert result_triton.shape == x.shape
    
    def test_higher_dimensional_tensors(self):
        """Test RMSNorm on higher dimensional tensors (fallback to PyTorch)"""
        # Test 4D tensor
        x = torch.randn(2, 4, 8, 16, device='cuda', dtype=torch.float32)
        
        for axis in range(x.dim()):
            weight = torch.randn(x.shape[axis], device='cuda', dtype=torch.float32)
            
            result_triton = rmsnorm(x, weight, axis=axis)
            result_torch = rmsnorm_module._rmsnorm_torch(x, None, axis=axis)
            assert torch.allclose(result_triton, result_torch, atol=1e-6), \
                f"Failed for axis {axis} in 4D tensor"
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        M, N = 32, 64
        
        # Test with large positive values
        x_large = torch.full((M, N), 100.0, device='cuda', dtype=torch.float32)
        result_triton = rmsnorm(x_large)
        result_torch = rmsnorm_module._rmsnorm_torch(x_large)
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Test with large negative values
        x_small = torch.full((M, N), -100.0, device='cuda', dtype=torch.float32)
        result_triton = rmsnorm(x_small)
        result_torch = rmsnorm_module._rmsnorm_torch(x_small)
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Test with mixed extreme values
        x_mixed = torch.randn(M, N, device='cuda', dtype=torch.float32) * 100
        result_triton = rmsnorm(x_mixed)
        result_torch = rmsnorm_module._rmsnorm_torch(x_mixed)
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Test with very small epsilon
        result_triton = rmsnorm(x_mixed, eps=1e-12)
        result_torch = rmsnorm_module._rmsnorm_torch(x_mixed, eps=1e-12)
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
    
    def test_zero_input(self):
        """Test RMSNorm with zero input"""
        M, N = 16, 32
        x_zero = torch.zeros((M, N), device='cuda', dtype=torch.float32)
        
        result_triton = rmsnorm(x_zero)
        result_torch = rmsnorm_module._rmsnorm_torch(x_zero)
        
        # All values should be zero (division by sqrt(eps) approaches 0)
        assert torch.allclose(result_triton, torch.zeros_like(result_triton), atol=1e-6)
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
    
    def test_rmsnorm_properties(self):
        """Test mathematical properties of RMSNorm"""
        M, N = 32, 64
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        result = rmsnorm(x)
        
        # Test that RMS is 1 (RMSNorm preserves RMS)
        x_squared = result * result
        mean_squared = torch.mean(x_squared, dim=-1)
        rms = torch.sqrt(mean_squared)
        
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-6), \
            "RMS should be close to 1"
        
        # Test that RMSNorm is invariant to scaling
        x_scaled = x * 10.0  # Scale
        result_scaled = rmsnorm(x_scaled)
        assert torch.allclose(result, result_scaled, atol=1e-6), \
            "RMSNorm should be invariant to scaling"
    
    def test_different_data_types(self):
        """Test RMSNorm with different data types"""
        M, N = 32, 64
        
        # Test float32
        x_f32 = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight_f32 = torch.randn(N, device='cuda', dtype=torch.float32)
        
        result_triton_f32 = rmsnorm(x_f32, weight_f32)
        result_torch_f32 = rmsnorm_module._rmsnorm_torch(x_f32, weight_f32)
        assert torch.allclose(result_triton_f32, result_torch_f32, atol=1e-6)
        
        # Test float16
        x_f16 = torch.randn(M, N, device='cuda', dtype=torch.float16)
        weight_f16 = torch.randn(N, device='cuda', dtype=torch.float16)
        
        result_triton_f16 = rmsnorm(x_f16, weight_f16)
        result_torch_f16 = rmsnorm_module._rmsnorm_torch(x_f16, weight_f16)
        # Higher tolerance for float16 due to precision differences
        assert torch.allclose(result_triton_f16, result_torch_f16, atol=1e-2, rtol=1e-2)
        
        # Test bfloat16 if available
        if torch.cuda.is_bf16_supported():
            x_bf16 = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
            weight_bf16 = torch.randn(N, device='cuda', dtype=torch.bfloat16)
            
            result_triton_bf16 = rmsnorm(x_bf16, weight_bf16)
            result_torch_bf16 = rmsnorm_module._rmsnorm_torch(x_bf16, weight_bf16)
            # Higher tolerance for bfloat16 due to precision differences
            assert torch.allclose(result_triton_bf16, result_torch_bf16, atol=1e-2, rtol=1e-2)
    
    def test_input_validation(self):
        """Test input validation"""
        M, N = 32, 32
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        
        # Test CPU tensor
        x_cpu = torch.randn(M, N, device='cpu', dtype=torch.float32)
        with pytest.raises(AssertionError, match="Input tensor must be on CUDA device"):
            rmsnorm(x_cpu, weight)
        
        # Test invalid axis
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        with pytest.raises(AssertionError, match="Invalid axis"):
            rmsnorm(x, weight, axis=5)
        
        # Test wrong weight dimensions
        weight_wrong = torch.randn(N + 1, device='cuda', dtype=torch.float32)
        with pytest.raises(AssertionError, match="Weight shape"):
            rmsnorm(x, weight_wrong)
        
        # Test negative axis
        result = rmsnorm(x, weight, axis=-1)  # Should work
        assert result.shape == x.shape
    
    def test_against_naive_implementation(self):
        """Test against naive PyTorch implementation"""
        M, N = 64, 128
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        
        result_triton = rmsnorm(x, weight)
        result_naive = naive_rmsnorm(x, weight)
        
        assert torch.allclose(result_triton, result_naive, atol=1e-6), \
            f"Triton and naive implementations differ. Max diff: {torch.max(torch.abs(result_triton - result_naive))}"
    
    def test_performance_benchmark(self):
        """Test performance benchmark functionality"""
        results = benchmark_rmsnorm(M=512, N=512, warmup=5, repeat=10)
        
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
            result = rmsnorm(x, weight)
            del x, weight, result
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
        
        result_triton = rmsnorm(x, weight)
        result_torch = rmsnorm_module._rmsnorm_torch(x, weight)
        
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Verify memory usage is reasonable
        expected_memory = (M * N * 2 + N) * 4  # bytes for float32 (input + output + weight)
        actual_memory = torch.cuda.memory_allocated()
        # Allow for significant overhead due to PyTorch memory management
        assert actual_memory < expected_memory * 10, f"Memory usage too high: {actual_memory} vs {expected_memory * 10}"
    
    def test_deterministic_output(self):
        """Test that output is deterministic"""
        M, N = 32, 64
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        weight = torch.randn(N, device='cuda', dtype=torch.float32)
        
        # Run multiple times
        results = []
        for _ in range(5):
            result = rmsnorm(x, weight)
            results.append(result.clone())
            torch.cuda.synchronize()
        
        # All results should be identical
        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i]), "Results are not deterministic"
    
    def test_comparison_with_layernorm(self):
        """Test that RMSNorm is different from LayerNorm"""
        M, N = 32, 64
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        # Add some constant to make means non-zero
        x = x + 2.0
        
        result_rmsnorm = rmsnorm(x)
        result_layernorm = torch.nn.functional.layer_norm(x, (N,))
        
        # Results should be different since RMSNorm doesn't subtract mean
        assert not torch.allclose(result_rmsnorm, result_layernorm, atol=1e-3), \
            "RMSNorm and LayerNorm should produce different results"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])