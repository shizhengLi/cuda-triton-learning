import torch
import pytest
import sys
import os

# Add the parent directory to the path to import our module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import directly from the file
import importlib.util
spec = importlib.util.spec_from_file_location("vector_add", os.path.join(project_root, "01_basics", "vector_operations", "vector_add.py"))
vector_add_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vector_add_module)

vector_add = vector_add_module.vector_add
benchmark_vector_add = vector_add_module.benchmark_vector_add


class TestVectorAdd:
    """Test suite for vector addition Triton kernel"""
    
    def test_basic_vector_add(self):
        """Test basic vector addition functionality"""
        size = 1024
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')
        
        # Triton implementation
        result_triton = vector_add(x, y)
        
        # PyTorch baseline
        result_torch = x + y
        
        # Verify results match
        assert torch.allclose(result_triton, result_torch, atol=1e-6), \
            f"Triton and PyTorch results don't match. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_different_sizes(self):
        """Test vector addition with different sizes"""
        sizes = [1, 16, 1024, 2048, 4096, 10000, 100000]
        
        for size in sizes:
            x = torch.randn(size, device='cuda')
            y = torch.randn(size, device='cuda')
            
            result_triton = vector_add(x, y)
            result_torch = x + y
            
            assert torch.allclose(result_triton, result_torch, atol=1e-6), \
                f"Failed for size {size}. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_non_multiple_of_block_size(self):
        """Test with sizes that are not multiples of BLOCK_SIZE"""
        # Test sizes that don't align with 1024 BLOCK_SIZE
        sizes = [1000, 1025, 2000, 3001, 5000]
        
        for size in sizes:
            x = torch.randn(size, device='cuda')
            y = torch.randn(size, device='cuda')
            
            result_triton = vector_add(x, y)
            result_torch = x + y
            
            assert torch.allclose(result_triton, result_torch, atol=1e-6), \
                f"Failed for non-aligned size {size}. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Empty tensor (should handle gracefully)
        x = torch.tensor([], device='cuda')
        y = torch.tensor([], device='cuda')
        
        result_triton = vector_add(x, y)
        result_torch = x + y
        
        assert result_triton.shape == result_torch.shape
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Single element
        x = torch.tensor([1.5], device='cuda')
        y = torch.tensor([2.5], device='cuda')
        
        result_triton = vector_add(x, y)
        result_torch = x + y
        
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        assert result_triton.item() == 4.0
    
    def test_input_validation(self):
        """Test input validation"""
        x = torch.randn(100, device='cuda')
        y = torch.randn(100, device='cuda')
        
        # Test shape mismatch
        y_wrong = torch.randn(50, device='cuda')
        with pytest.raises(AssertionError, match="Input tensors must have same shape"):
            vector_add(x, y_wrong)
        
        # Test CPU tensors
        x_cpu = torch.randn(100, device='cpu')
        y_cpu = torch.randn(100, device='cpu')
        with pytest.raises(AssertionError, match="Input tensors must be on CUDA device"):
            vector_add(x_cpu, y_cpu)
    
    def test_performance_benchmark(self):
        """Test performance benchmark functionality"""
        results = benchmark_vector_add(size=100000, warmup=5, repeat=10)
        
        # Verify benchmark results contain expected keys
        expected_keys = ['triton_time', 'torch_time', 'speedup', 'bandwidth']
        for key in expected_keys:
            assert key in results, f"Missing benchmark result key: {key}"
        
        # Verify reasonable values
        assert results['triton_time'] > 0, "Triton time should be positive"
        assert results['torch_time'] > 0, "PyTorch time should be positive"
        assert results['speedup'] > 0, "Speedup should be positive"
        assert results['bandwidth'] > 0, "Bandwidth should be positive"
        
        print(f"Benchmark results: {results}")
    
    def test_memory_efficiency(self):
        """Test that kernel doesn't cause memory leaks"""
        import gc
        
        initial_memory = torch.cuda.memory_allocated()
        
        # Run multiple iterations
        for _ in range(100):
            x = torch.randn(10000, device='cuda')
            y = torch.randn(10000, device='cuda')
            result = vector_add(x, y)
            del x, y, result
            torch.cuda.synchronize()
        
        gc.collect()
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should not grow significantly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 1024 * 1024, f"Memory growth too large: {memory_growth} bytes"
    
    def test_numeric_precision(self):
        """Test numeric precision with different data types"""
        # Test float32
        x_f32 = torch.randn(1000, device='cuda', dtype=torch.float32)
        y_f32 = torch.randn(1000, device='cuda', dtype=torch.float32)
        
        result_triton = vector_add(x_f32, y_f32)
        result_torch = x_f32 + y_f32
        
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Test float16
        x_f16 = torch.randn(1000, device='cuda', dtype=torch.float16)
        y_f16 = torch.randn(1000, device='cuda', dtype=torch.float16)
        
        result_triton = vector_add(x_f16, y_f16)
        result_torch = x_f16 + y_f16
        
        # Higher tolerance for float16
        assert torch.allclose(result_triton, result_torch, atol=1e-3)
    
    def test_large_vectors(self):
        """Test with very large vectors"""
        size = 10_000_000  # 10M elements
        
        x = torch.randn(size, device='cuda')
        y = torch.randn(size, device='cuda')
        
        result_triton = vector_add(x, y)
        result_torch = x + y
        
        assert torch.allclose(result_triton, result_torch, atol=1e-6)
        
        # Verify memory usage is reasonable
        assert torch.cuda.memory_allocated() < size * 4 * 4 * 1.1  # Some overhead allowed


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])