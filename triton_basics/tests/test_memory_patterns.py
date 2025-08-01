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
    "memory_access", 
    os.path.join(project_root, "01_basics", "memory_patterns", "memory_access.py")
)
memory_access_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memory_access_module)

coalesced_access = memory_access_module.coalesced_access
strided_access = memory_access_module.strided_access
shared_memory_access = memory_access_module.shared_memory_access
benchmark_memory_patterns = memory_access_module.benchmark_memory_patterns
memory_efficiency_analysis = memory_access_module.memory_efficiency_analysis


class TestMemoryPatterns:
    """Test suite for memory patterns Triton kernels"""
    
    def test_coalesced_access_basic(self):
        """Test basic coalesced memory access functionality"""
        size = 1024
        x = torch.randn(size, device='cuda')
        
        # Triton implementation
        result_triton = coalesced_access(x)
        
        # PyTorch baseline
        result_torch = x * x
        
        # Verify results match
        assert torch.allclose(result_triton, result_torch, atol=1e-6), \
            f"Triton and PyTorch results don't match. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_coalesced_access_different_sizes(self):
        """Test coalesced access with different sizes"""
        sizes = [1, 16, 512, 1024, 2048, 4096, 10000]
        
        for size in sizes:
            x = torch.randn(size, device='cuda')
            
            result_triton = coalesced_access(x)
            result_torch = x * x
            
            assert torch.allclose(result_triton, result_torch, atol=1e-6), \
                f"Failed for size {size}. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_strided_access_basic(self):
        """Test basic strided memory access functionality"""
        M, N = 64, 128
        x = torch.randn(M, N, device='cuda')
        
        # Triton implementation
        result_triton = strided_access(x)
        
        # PyTorch baseline
        result_torch = x * 2.0
        
        # Verify results match
        assert torch.allclose(result_triton, result_torch, atol=1e-6), \
            f"Triton and PyTorch results don't match. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_strided_access_different_shapes(self):
        """Test strided access with different matrix shapes"""
        shapes = [
            (1, 1),      # Single element
            (1, 1024),   # Single row
            (1024, 1),   # Single column
            (64, 64),    # Square
            (128, 256),  # Rectangular
            (256, 128),  # Rectangular (transposed)
            (512, 512),  # Larger square
        ]
        
        for M, N in shapes:
            x = torch.randn(M, N, device='cuda')
            
            result_triton = strided_access(x)
            result_torch = x * 2.0
            
            assert torch.allclose(result_triton, result_torch, atol=1e-6), \
                f"Failed for shape {M}x{N}. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_shared_memory_access_basic(self):
        """Test basic shared memory access functionality"""
        M, N = 32, 32
        x = torch.randn(M, N, device='cuda')
        
        # Triton implementation
        result_triton = shared_memory_access(x)
        
        # PyTorch baseline
        result_torch = x * 2.0 + 1.0
        
        # Verify results match
        assert torch.allclose(result_triton, result_torch, atol=1e-6), \
            f"Triton and PyTorch results don't match. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_shared_memory_access_different_shapes(self):
        """Test shared memory access with different matrix shapes"""
        shapes = [
            (16, 16),   # Small square
            (32, 32),   # Medium square
            (64, 64),   # Larger square
            (32, 64),   # Rectangular
            (64, 32),   # Rectangular (transposed)
        ]
        
        for M, N in shapes:
            x = torch.randn(M, N, device='cuda')
            
            result_triton = shared_memory_access(x)
            result_torch = x * 2.0 + 1.0
            
            assert torch.allclose(result_triton, result_torch, atol=1e-6), \
                f"Failed for shape {M}x{N}. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
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
            x = torch.randn(M, N, device='cuda')
            
            # Test strided access
            result_strided = strided_access(x)
            expected_strided = x * 2.0
            assert torch.allclose(result_strided, expected_strided, atol=1e-6), \
                f"Strided access failed for {M}x{N}"
            
            # Test shared memory access
            result_shared = shared_memory_access(x)
            expected_shared = x * 2.0 + 1.0
            assert torch.allclose(result_shared, expected_shared, atol=1e-6), \
                f"Shared memory access failed for {M}x{N}"
    
    def test_input_validation(self):
        """Test input validation"""
        # Test CPU tensor
        x_cpu = torch.randn(100, device='cpu')
        with pytest.raises(AssertionError, match="Input tensor must be on CUDA device"):
            coalesced_access(x_cpu)
        
        # Test wrong dimension for strided access
        x_1d = torch.randn(100, device='cuda')
        with pytest.raises(AssertionError, match="Input tensor must be 2D"):
            strided_access(x_1d)
        
        # Test wrong dimension for shared memory access
        with pytest.raises(AssertionError, match="Input tensor must be 2D"):
            shared_memory_access(x_1d)
    
    def test_performance_benchmark(self):
        """Test performance benchmark functionality"""
        results = benchmark_memory_patterns(size=512, warmup=3, repeat=5)
        
        # Verify benchmark results contain expected keys
        expected_keys = [
            'coalesced_time', 'strided_time', 'shared_memory_time',
            'coalesced_bandwidth', 'strided_bandwidth', 'shared_memory_bandwidth',
            'matrix_size', 'total_elements'
        ]
        for key in expected_keys:
            assert key in results, f"Missing benchmark result key: {key}"
        
        # Verify reasonable values
        assert results['coalesced_time'] > 0, "Coalesced time should be positive"
        assert results['strided_time'] > 0, "Strided time should be positive"
        assert results['shared_memory_time'] > 0, "Shared memory time should be positive"
        assert results['coalesced_bandwidth'] > 0, "Coalesced bandwidth should be positive"
        assert results['strided_bandwidth'] > 0, "Strided bandwidth should be positive"
        assert results['shared_memory_bandwidth'] > 0, "Shared memory bandwidth should be positive"
        assert results['matrix_size'] == (512, 512), "Matrix shape should match input"
        assert results['total_elements'] == 512 * 512, "Total elements should match"
        
        print(f"Benchmark results: {results}")
    
    def test_memory_efficiency_analysis(self):
        """Test memory efficiency analysis functionality"""
        results = memory_efficiency_analysis(size=256)
        
        # Verify analysis results contain expected keys
        assert 'triton_results' in results, "Missing triton results"
        assert 'baseline_time' in results, "Missing baseline time"
        assert 'efficiency' in results, "Missing efficiency analysis"
        assert 'matrix_size' in results, "Missing matrix size"
        
        # Verify efficiency metrics
        efficiency = results['efficiency']
        expected_efficiency_keys = [
            'coalesced_efficiency', 'strided_efficiency', 'shared_memory_efficiency',
            'bandwidth_coalesced_ratio', 'bandwidth_shared_ratio'
        ]
        for key in expected_efficiency_keys:
            assert key in efficiency, f"Missing efficiency metric: {key}"
        
        # Verify reasonable values
        assert results['baseline_time'] > 0, "Baseline time should be positive"
        assert efficiency['coalesced_efficiency'] > 0, "Coalesced efficiency should be positive"
        assert efficiency['strided_efficiency'] > 0, "Strided efficiency should be positive"
        assert efficiency['shared_memory_efficiency'] > 0, "Shared memory efficiency should be positive"
        
        print(f"Efficiency analysis results: {results}")
    
    def test_memory_efficiency(self):
        """Test that kernels don't cause memory leaks"""
        import gc
        
        initial_memory = torch.cuda.memory_allocated()
        
        # Run multiple iterations
        for _ in range(50):
            x = torch.randn(256, 256, device='cuda')
            
            # Test all three access patterns
            _ = coalesced_access(x.flatten())
            _ = strided_access(x)
            _ = shared_memory_access(x)
            
            del x
            torch.cuda.synchronize()
        
        gc.collect()
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should not grow significantly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 10 * 1024 * 1024, f"Memory growth too large: {memory_growth} bytes"
    
    def test_numeric_precision(self):
        """Test numeric precision with different data types"""
        # Test float32
        x_f32 = torch.randn(1000, device='cuda', dtype=torch.float32)
        
        result_coalesced = coalesced_access(x_f32)
        expected_coalesced = x_f32 * x_f32
        assert torch.allclose(result_coalesced, expected_coalesced, atol=1e-6)
        
        # Test float16
        x_f16 = torch.randn(1000, device='cuda', dtype=torch.float16)
        
        result_coalesced_f16 = coalesced_access(x_f16)
        expected_coalesced_f16 = x_f16 * x_f16
        # Higher tolerance for float16
        assert torch.allclose(result_coalesced_f16, expected_coalesced_f16, atol=1e-3)
    
    # def test_large_matrices(self):
    #     """Test with large matrices"""
    #     M, N = 2048, 2048
        
    #     x = torch.randn(M, N, device='cuda')
        
    #     # Test coalesced access
    #     result_coalesced = coalesced_access(x.flatten())
    #     expected_coalesced = x.flatten() * x.flatten()
    #     assert torch.allclose(result_coalesced, expected_coalesced, atol=1e-6)
        
    #     # Test strided access
    #     result_strided = strided_access(x)
    #     expected_strided = x * 2.0
    #     assert torch.allclose(result_strided, expected_strided, atol=1e-1)  # Slightly relaxed tolerance
        
    #     # Verify memory usage is reasonable
    #     assert torch.cuda.memory_allocated() < M * N * 4 * 4 * 2  # Some overhead allowed
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Empty tensor
        x_empty = torch.tensor([], device='cuda')
        result_empty = coalesced_access(x_empty)
        assert result_empty.shape == x_empty.shape
        
        # Single element
        x_single = torch.tensor([2.5], device='cuda')
        result_single = coalesced_access(x_single)
        expected_single = x_single * x_single
        assert torch.allclose(result_single, expected_single, atol=1e-6)
        assert result_single.item() == 6.25
        
        # Single element matrix
        x_matrix_single = torch.randn(1, 1, device='cuda')
        result_strided = strided_access(x_matrix_single)
        expected_strided = x_matrix_single * 2.0
        assert torch.allclose(result_strided, expected_strided, atol=1e-6)
    
    # def test_performance_comparison(self):
    #     """Test performance comparison between access patterns"""
    #     results = benchmark_memory_patterns(size=1024, warmup=5, repeat=10)
        
    #     # Coalesced access should generally be faster than strided access
    #     assert results['coalesced_time'] < results['strided_time'], \
    #         f"Coalesced access should be faster: coalesced={results['coalesced_time']:.6f}, strided={results['strided_time']:.6f}"
        
    #     # Coalesced access should have higher bandwidth
    #     assert results['coalesced_bandwidth'] > results['strided_bandwidth'], \
    #         f"Coalesced should have higher bandwidth: coalesced={results['coalesced_bandwidth']:.1f}, strided={results['strided_bandwidth']:.1f}"
        
    #     print(f"Performance comparison: {results}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])