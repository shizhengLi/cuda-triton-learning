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
    "softmax", 
    os.path.join(project_root, "02_dl_kernels", "normalization", "softmax.py")
)
softmax_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(softmax_module)

softmax = softmax_module.softmax
naive_softmax = softmax_module.naive_softmax
benchmark_softmax = softmax_module.benchmark_softmax


class TestSoftmax:
    """Test suite for softmax Triton kernel"""
    
    def test_basic_softmax(self):
        """Test basic softmax functionality"""
        M, N = 64, 32
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        # Triton implementation
        result_triton = softmax(x)
        
        # PyTorch baseline
        result_torch = torch.softmax(x, dim=-1)
        
        # Verify results match
        assert torch.allclose(result_triton, result_torch, atol=1e-6, rtol=1e-6), \
            f"Triton and PyTorch results don't match. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_different_matrix_sizes(self):
        """Test softmax with different matrix sizes"""
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
            
            result_triton = softmax(x)
            result_torch = torch.softmax(x, dim=-1)
            
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
            
            result_triton = softmax(x)
            result_torch = torch.softmax(x, dim=-1)
            
            assert torch.allclose(result_triton, result_torch, atol=1e-6, rtol=1e-6), \
                f"Failed for non-power-of-2 size {M}x{N}. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_different_axes(self):
        """Test softmax along different axes (fallback to PyTorch for complex cases)"""
        # Test 3D tensor - should fallback to PyTorch
        x = torch.randn(4, 8, 16, device='cuda', dtype=torch.float32)
        
        # Test along last dimension (default) - should fallback to PyTorch
        result_triton_last = softmax(x, axis=-1)
        result_torch_last = torch.softmax(x, dim=-1)
        assert torch.allclose(result_triton_last, result_torch_last, atol=1e-6)
        
        # Test 2D tensor along first dimension - should fallback to PyTorch
        x_2d = torch.randn(8, 16, device='cuda', dtype=torch.float32)
        result_triton_first = softmax(x_2d, axis=0)
        result_torch_first = torch.softmax(x_2d, dim=0)
        assert torch.allclose(result_triton_first, result_torch_first, atol=1e-6)
    
    def test_1d_tensor(self):
        """Test softmax on 1D tensor (fallback to PyTorch)"""
        size = 128
        x = torch.randn(size, device='cuda', dtype=torch.float32)
        
        result_triton = softmax(x)
        result_torch = torch.softmax(x, dim=-1)
        
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        assert result_triton.shape == x.shape
    
    def test_higher_dimensional_tensors(self):
        """Test softmax on higher dimensional tensors (fallback to PyTorch)"""
        # Test 4D tensor
        x = torch.randn(2, 4, 8, 16, device='cuda', dtype=torch.float32)
        
        for axis in range(x.dim()):
            result_triton = softmax(x, axis=axis)
            result_torch = torch.softmax(x, dim=axis)
            assert torch.allclose(result_triton, result_torch, atol=1e-6), \
                f"Failed for axis {axis} in 4D tensor"
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        M, N = 32, 64
        
        # Test with large positive values
        x_large = torch.full((M, N), 100.0, device='cuda', dtype=torch.float32)
        result_triton = softmax(x_large)
        result_torch = torch.softmax(x_large, dim=-1)
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Test with large negative values
        x_small = torch.full((M, N), -100.0, device='cuda', dtype=torch.float32)
        result_triton = softmax(x_small)
        result_torch = torch.softmax(x_small, dim=-1)
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Test with mixed extreme values
        x_mixed = torch.randn(M, N, device='cuda', dtype=torch.float32) * 100
        result_triton = softmax(x_mixed)
        result_torch = torch.softmax(x_mixed, dim=-1)
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
    
    def test_zero_input(self):
        """Test softmax with zero input"""
        M, N = 16, 32
        x_zero = torch.zeros((M, N), device='cuda', dtype=torch.float32)
        
        result_triton = softmax(x_zero)
        result_torch = torch.softmax(x_zero, dim=-1)
        
        # All values should be equal (uniform distribution)
        expected_value = 1.0 / N
        assert torch.allclose(result_triton, torch.full_like(result_triton, expected_value), atol=1e-6)
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
    
    def test_softmax_properties(self):
        """Test mathematical properties of softmax"""
        M, N = 32, 64
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        result = softmax(x)
        
        # Test that all values are positive
        assert torch.all(result > 0), "Softmax output should be positive"
        
        # Test that each row sums to 1
        row_sums = torch.sum(result, dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
            "Each row should sum to 1"
        
        # Test that softmax is invariant to constant shifts
        x_shifted = x + 100.0  # Add large constant
        result_shifted = softmax(x_shifted)
        assert torch.allclose(result, result_shifted, atol=1e-6), \
            "Softmax should be invariant to constant shifts"
    
    def test_different_data_types(self):
        """Test softmax with different data types"""
        M, N = 32, 64
        
        # Test float32
        x_f32 = torch.randn(M, N, device='cuda', dtype=torch.float32)
        result_triton_f32 = softmax(x_f32)
        result_torch_f32 = torch.softmax(x_f32, dim=-1)
        assert torch.allclose(result_triton_f32, result_torch_f32, atol=1e-6)
        
        # Test float16
        x_f16 = torch.randn(M, N, device='cuda', dtype=torch.float16)
        result_triton_f16 = softmax(x_f16)
        result_torch_f16 = torch.softmax(x_f16, dim=-1)
        # Higher tolerance for float16
        assert torch.allclose(result_triton_f16, result_torch_f16, atol=1e-3, rtol=1e-3)
        
        # Test bfloat16 if available
        if torch.cuda.is_bf16_supported():
            x_bf16 = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
            result_triton_bf16 = softmax(x_bf16)
            result_torch_bf16 = torch.softmax(x_bf16, dim=-1)
            assert torch.allclose(result_triton_bf16, result_torch_bf16, atol=1e-3, rtol=1e-3)
    
    def test_input_validation(self):
        """Test input validation"""
        # Test CPU tensor
        x_cpu = torch.randn(32, 32, device='cpu', dtype=torch.float32)
        with pytest.raises(AssertionError, match="Input tensor must be on CUDA device"):
            softmax(x_cpu)
        
        # Test invalid axis
        x = torch.randn(32, 32, device='cuda', dtype=torch.float32)
        with pytest.raises(AssertionError, match="Invalid axis"):
            softmax(x, axis=5)
        
        # Test negative axis
        result = softmax(x, axis=-1)  # Should work
        assert result.shape == x.shape
    
    def test_against_naive_implementation(self):
        """Test against naive PyTorch implementation"""
        M, N = 64, 128
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        result_triton = softmax(x)
        result_naive = naive_softmax(x)
        
        assert torch.allclose(result_triton, result_naive, atol=1e-6), \
            f"Triton and naive implementations differ. Max diff: {torch.max(torch.abs(result_triton - result_naive))}"
    
    def test_performance_benchmark(self):
        """Test performance benchmark functionality"""
        results = benchmark_softmax(M=512, N=512, warmup=5, repeat=10)
        
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
            result = softmax(x)
            del x, result
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
        
        result_triton = softmax(x)
        result_torch = torch.softmax(x, dim=-1)
        
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Verify memory usage is reasonable
        expected_memory = (M * N * 2) * 4  # bytes for float32 (input + output)
        actual_memory = torch.cuda.memory_allocated()
        # Allow for significant overhead due to PyTorch memory management
        assert actual_memory < expected_memory * 10, f"Memory usage too high: {actual_memory} vs {expected_memory * 10}"
    
    def test_deterministic_output(self):
        """Test that output is deterministic"""
        M, N = 32, 64
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        # Run multiple times
        results = []
        for _ in range(5):
            result = softmax(x)
            results.append(result.clone())
            torch.cuda.synchronize()
        
        # All results should be identical
        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i]), "Results are not deterministic"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])