import torch
import pytest
import sys
import os

# Add the parent directory to the path to import our module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import directly from the file
import importlib.util
spec = importlib.util.spec_from_file_location("matmul", os.path.join(project_root, "01_basics", "matrix_operations", "matmul.py"))
matmul_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(matmul_module)

matmul = matmul_module.matmul
matmul_with_bias = matmul_module.matmul_with_bias
batched_matmul = matmul_module.batched_matmul
benchmark_matmul = matmul_module.benchmark_matmul


class TestMatMul:
    """Test suite for matrix multiplication Triton kernel"""
    
    def test_basic_matmul(self):
        """Test basic matrix multiplication functionality"""
        M, N, K = 64, 64, 64
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        
        # Triton implementation
        result_triton = matmul(a, b)
        
        # PyTorch baseline
        result_torch = torch.matmul(a, b)
        
        # Verify results match
        assert torch.allclose(result_triton, result_torch, atol=1e-2, rtol=1e-2), \
            f"Triton and PyTorch results don't match. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_different_matrix_sizes(self):
        """Test matrix multiplication with different sizes"""
        sizes = [(32, 32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 512, 512)]
        
        for M, N, K in sizes:
            a = torch.randn(M, K, device='cuda', dtype=torch.float16)
            b = torch.randn(K, N, device='cuda', dtype=torch.float16)
            
            result_triton = matmul(a, b)
            result_torch = torch.matmul(a, b)
            
            assert torch.allclose(result_triton, result_torch, atol=1e-2, rtol=1e-2), \
                f"Failed for size {M}x{K} @ {K}x{N}. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_rectangular_matrices(self):
        """Test with rectangular matrices"""
        test_cases = [
            (100, 50, 75),  # M > N, K
            (50, 100, 75),  # N > M, K
            (75, 100, 50),  # K > M, N
            (128, 64, 256), # M, N < K
            (256, 128, 64), # M, N > K
        ]
        
        for M, N, K in test_cases:
            a = torch.randn(M, K, device='cuda', dtype=torch.float16)
            b = torch.randn(K, N, device='cuda', dtype=torch.float16)
            
            result_triton = matmul(a, b)
            result_torch = torch.matmul(a, b)
            
            assert torch.allclose(result_triton, result_torch, atol=1e-2, rtol=1e-2), \
                f"Failed for rectangular size {M}x{K} @ {K}x{N}. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_non_multiple_of_block_size(self):
        """Test with sizes that are not multiples of block sizes"""
        test_cases = [
            (100, 100, 100),  # Not multiple of 128
            (129, 129, 129),  # Just over block size
            (255, 255, 255),  # Almost 2x block size
            (1000, 1000, 33), # K not multiple of 32
        ]
        
        for M, N, K in test_cases:
            a = torch.randn(M, K, device='cuda', dtype=torch.float16)
            b = torch.randn(K, N, device='cuda', dtype=torch.float16)
            
            result_triton = matmul(a, b)
            result_torch = torch.matmul(a, b)
            
            assert torch.allclose(result_triton, result_torch, atol=1e-2, rtol=1e-2), \
                f"Failed for non-aligned size {M}x{K} @ {K}x{N}. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_matmul_with_bias(self):
        """Test matrix multiplication with bias"""
        M, N, K = 64, 32, 48
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        bias = torch.randn(N, device='cuda', dtype=torch.float16)
        
        # Triton implementation
        result_triton = matmul_with_bias(a, b, bias)
        
        # PyTorch baseline
        result_torch = torch.matmul(a, b) + bias.unsqueeze(0)
        
        assert torch.allclose(result_triton, result_torch, atol=1e-2, rtol=1e-2), \
            f"Matmul with bias failed. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_batched_matmul(self):
        """Test batched matrix multiplication"""
        B, M, N, K = 4, 32, 32, 32
        a = torch.randn(B, M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(B, K, N, device='cuda', dtype=torch.float16)
        
        # Triton implementation
        result_triton = batched_matmul(a, b)
        
        # PyTorch baseline
        result_torch = torch.bmm(a, b)
        
        assert torch.allclose(result_triton, result_torch, atol=1e-2, rtol=1e-2), \
            f"Batched matmul failed. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_different_batch_sizes(self):
        """Test batched matmul with different batch sizes"""
        batch_sizes = [1, 2, 4, 8, 16]
        
        for B in batch_sizes:
            M, N, K = 32, 32, 32
            a = torch.randn(B, M, K, device='cuda', dtype=torch.float16)
            b = torch.randn(B, K, N, device='cuda', dtype=torch.float16)
            
            result_triton = batched_matmul(a, b)
            result_torch = torch.bmm(a, b)
            
            assert torch.allclose(result_triton, result_torch, atol=1e-2, rtol=1e-2), \
                f"Failed for batch size {B}. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Small matrices
        a = torch.randn(1, 1, device='cuda', dtype=torch.float16)
        b = torch.randn(1, 1, device='cuda', dtype=torch.float16)
        
        result_triton = matmul(a, b)
        result_torch = torch.matmul(a, b)
        
        assert torch.allclose(result_triton, result_torch, atol=1e-2, rtol=1e-2)
        
        # Vector-matrix multiplication
        a = torch.randn(1, 32, device='cuda', dtype=torch.float16)
        b = torch.randn(32, 64, device='cuda', dtype=torch.float16)
        
        result_triton = matmul(a, b)
        result_torch = torch.matmul(a, b)
        
        assert torch.allclose(result_triton, result_torch, atol=1e-2, rtol=1e-2)
        
        # Matrix-vector multiplication
        a = torch.randn(32, 64, device='cuda', dtype=torch.float16)
        b = torch.randn(64, 1, device='cuda', dtype=torch.float16)
        
        result_triton = matmul(a, b)
        result_torch = torch.matmul(a, b)
        
        assert torch.allclose(result_triton, result_torch, atol=1e-2, rtol=1e-2)
    
    def test_input_validation(self):
        """Test input validation"""
        a = torch.randn(32, 32, device='cuda', dtype=torch.float16)
        b = torch.randn(32, 32, device='cuda', dtype=torch.float16)
        
        # Test dimension mismatch
        a_wrong = torch.randn(32, 16, device='cuda', dtype=torch.float16)
        with pytest.raises(AssertionError, match="Matrix dimensions incompatible"):
            matmul(a_wrong, b)
        
        # Test wrong dimensions
        a_3d = torch.randn(2, 32, 32, device='cuda', dtype=torch.float16)
        with pytest.raises(AssertionError, match="Input tensors must be 2D"):
            matmul(a_3d, b)
        
        # Test CPU tensors
        a_cpu = torch.randn(32, 32, device='cpu', dtype=torch.float16)
        with pytest.raises(AssertionError, match="Input tensors must be on CUDA device"):
            matmul(a_cpu, b)
    
    def test_performance_benchmark(self):
        """Test performance benchmark functionality"""
        results = benchmark_matmul(M=512, N=512, K=512, warmup=5, repeat=10)
        
        # Verify benchmark results contain expected keys
        expected_keys = ['triton_time', 'torch_time', 'speedup', 'triton_tflops', 'torch_tflops', 'matrix_shape']
        for key in expected_keys:
            assert key in results, f"Missing benchmark result key: {key}"
        
        # Verify reasonable values
        assert results['triton_time'] > 0, "Triton time should be positive"
        assert results['torch_time'] > 0, "PyTorch time should be positive"
        assert results['speedup'] > 0, "Speedup should be positive"
        assert results['triton_tflops'] > 0, "Triton TFLOPS should be positive"
        assert results['torch_tflops'] > 0, "PyTorch TFLOPS should be positive"
        
        print(f"Benchmark results: {results}")
    
    def test_memory_efficiency(self):
        """Test that kernel doesn't cause memory leaks"""
        import gc
        
        initial_memory = torch.cuda.memory_allocated()
        
        # Run multiple iterations
        for _ in range(50):
            a = torch.randn(256, 256, device='cuda', dtype=torch.float16)
            b = torch.randn(256, 256, device='cuda', dtype=torch.float16)
            result = matmul(a, b)
            del a, b, result
            torch.cuda.synchronize()
        
        gc.collect()
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should not grow significantly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 10 * 1024 * 1024, f"Memory growth too large: {memory_growth} bytes"
    
    def test_numeric_precision(self):
        """Test numeric precision with different data types"""
        M, N, K = 64, 64, 64
        
        # Test float16
        a_f16 = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b_f16 = torch.randn(K, N, device='cuda', dtype=torch.float16)
        
        result_triton = matmul(a_f16, b_f16)
        result_torch = torch.matmul(a_f16, b_f16)
        
        assert torch.allclose(result_triton, result_torch, atol=1e-2, rtol=1e-2)
        
        # Test with different random seeds but same data types
        torch.manual_seed(42)
        a_f16_test = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b_f16_test = torch.randn(K, N, device='cuda', dtype=torch.float16)
        
        result_triton_test = matmul(a_f16_test, b_f16_test)
        result_torch_test = torch.matmul(a_f16_test, b_f16_test)
        
        assert torch.allclose(result_triton_test, result_torch_test, atol=1e-2, rtol=1e-2)
    
    def test_large_matrices(self):
        """Test with large matrices"""
        M, N, K = 1024, 1024, 1024
        
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        
        result_triton = matmul(a, b)
        result_torch = torch.matmul(a, b)
        
        assert torch.allclose(result_triton, result_torch, atol=1e-2, rtol=1e-2)
        
        # Verify memory usage is reasonable (allow for PyTorch caching and overhead)
        expected_memory = (M * K + K * N + M * N) * 2  # bytes for float16
        actual_memory = torch.cuda.memory_allocated()
        # Allow for significant overhead due to PyTorch memory management
        assert actual_memory < expected_memory * 10, f"Memory usage too high: {actual_memory} vs {expected_memory * 10}"
    
    def test_deterministic_output(self):
        """Test that output is deterministic"""
        M, N, K = 64, 64, 64
        a = torch.randn(M, K, device='cuda', dtype=torch.float16)
        b = torch.randn(K, N, device='cuda', dtype=torch.float16)
        
        # Run multiple times
        results = []
        for _ in range(5):
            result = matmul(a, b)
            results.append(result.clone())
            torch.cuda.synchronize()
        
        # All results should be identical
        for i in range(1, len(results)):
            assert torch.equal(results[0], results[i]), "Results are not deterministic"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])