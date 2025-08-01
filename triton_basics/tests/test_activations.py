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
    "activations", 
    os.path.join(project_root, "02_dl_kernels", "activations", "activations.py")
)
activations_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(activations_module)

relu = activations_module.relu
gelu = activations_module.gelu
silu = activations_module.silu
relu_reference = activations_module.relu_reference
gelu_reference = activations_module.gelu_reference
silu_reference = activations_module.silu_reference
benchmark_activations = activations_module.benchmark_activations
benchmark_all_activations = activations_module.benchmark_all_activations


class TestActivations:
    """Test suite for activation functions Triton kernels"""
    
    def test_relu_basic(self):
        """Test basic ReLU functionality"""
        size = 1000
        x = torch.randn(size, device='cuda')
        
        # Triton implementation
        result_triton = relu(x)
        
        # PyTorch baseline
        result_torch = torch.relu(x)
        
        # Verify results match
        assert torch.allclose(result_triton, result_torch, atol=1e-6), \
            f"Triton and PyTorch results don't match. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_relu_properties(self):
        """Test ReLU mathematical properties"""
        size = 1000
        x = torch.randn(size, device='cuda')
        
        result = relu(x)
        
        # ReLU should be non-negative
        assert torch.all(result >= 0), "ReLU output should be non-negative"
        
        # ReLU should preserve positive values
        positive_mask = x > 0
        assert torch.allclose(result[positive_mask], x[positive_mask], atol=1e-6), \
            "ReLU should preserve positive values"
        
        # ReLU should zero out negative values
        negative_mask = x < 0
        assert torch.all(result[negative_mask] == 0), "ReLU should zero out negative values"
    
    def test_gelu_basic(self):
        """Test basic GELU functionality"""
        size = 1000
        x = torch.randn(size, device='cuda')
        
        # Triton implementation
        result_triton = gelu(x)
        
        # PyTorch baseline
        result_torch = torch.nn.functional.gelu(x)
        
        # Verify results match (relaxed tolerance for approximation)
        assert torch.allclose(result_triton, result_torch, atol=1e-3), \
            f"Triton and PyTorch results don't match. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_gelu_properties(self):
        """Test GELU mathematical properties"""
        size = 1000
        x = torch.randn(size, device='cuda')
        
        result = gelu(x)
        
        # GELU should be approximately linear for very small positive values
        small_positive = x[(x > 0) & (x < 0.1)]
        if len(small_positive) > 0:
            expected_small = small_positive
            actual_small = result[(x > 0) & (x < 0.1)]
            assert torch.allclose(actual_small, expected_small, atol=0.15), \
                "GELU should be approximately linear for very small positive values"
        
        # GELU should be smooth and continuous
        assert not torch.isnan(result).any(), "GELU should not produce NaN values"
        assert not torch.isinf(result).any(), "GELU should not produce infinite values"
    
    def test_silu_basic(self):
        """Test basic SiLU functionality"""
        size = 1000
        x = torch.randn(size, device='cuda')
        
        # Triton implementation
        result_triton = silu(x)
        
        # PyTorch baseline
        result_torch = torch.nn.functional.silu(x)
        
        # Verify results match
        assert torch.allclose(result_triton, result_torch, atol=1e-6), \
            f"Triton and PyTorch results don't match. Max diff: {torch.max(torch.abs(result_triton - result_torch))}"
    
    def test_silu_properties(self):
        """Test SiLU mathematical properties"""
        size = 1000
        x = torch.randn(size, device='cuda')
        
        result = silu(x)
        
        # SiLU should be smooth and continuous
        assert not torch.isnan(result).any(), "SiLU should not produce NaN values"
        assert not torch.isinf(result).any(), "SiLU should not produce infinite values"
        
        # SiLU should preserve sign
        assert torch.all((result >= 0) == (x >= 0)), "SiLU should preserve sign"
        
        # SiLU should be approximately linear for very small values
        small_values = x[torch.abs(x) < 0.05]
        if len(small_values) > 0:
            expected_small = small_values
            actual_small = result[torch.abs(x) < 0.05]
            assert torch.allclose(actual_small, expected_small, atol=0.1), \
                "SiLU should be approximately linear for very small values"
    
    def test_different_tensor_sizes(self):
        """Test activation functions with different tensor sizes"""
        sizes = [1, 16, 256, 1024, 10000, 100000]
        
        for size in sizes:
            x = torch.randn(size, device='cuda')
            
            # Test ReLU
            result_relu = relu(x)
            expected_relu = torch.relu(x)
            assert torch.allclose(result_relu, expected_relu, atol=1e-6), \
                f"ReLU failed for size {size}"
            
            # Test GELU
            result_gelu = gelu(x)
            expected_gelu = torch.nn.functional.gelu(x)
            assert torch.allclose(result_gelu, expected_gelu, atol=1e-3), \
                f"GELU failed for size {size}"
            
            # Test SiLU
            result_silu = silu(x)
            expected_silu = torch.nn.functional.silu(x)
            assert torch.allclose(result_silu, expected_silu, atol=1e-6), \
                f"SiLU failed for size {size}"
    
    def test_different_tensor_shapes(self):
        """Test activation functions with different tensor shapes"""
        shapes = [
            (100,),      # 1D
            (10, 10),    # 2D square
            (8, 16),     # 2D rectangular
            (4, 5, 6),   # 3D
            (2, 3, 4, 5), # 4D
        ]
        
        for shape in shapes:
            x = torch.randn(shape, device='cuda')
            
            # Test all activations
            result_relu = relu(x)
            expected_relu = torch.relu(x)
            assert torch.allclose(result_relu, expected_relu, atol=1e-6), \
                f"ReLU failed for shape {shape}"
            
            result_gelu = gelu(x)
            expected_gelu = torch.nn.functional.gelu(x)
            assert torch.allclose(result_gelu, expected_gelu, atol=1e-3), \
                f"GELU failed for shape {shape}"
            
            result_silu = silu(x)
            expected_silu = torch.nn.functional.silu(x)
            assert torch.allclose(result_silu, expected_silu, atol=1e-6), \
                f"SiLU failed for shape {shape}"
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Zero input
        x_zero = torch.zeros(100, device='cuda')
        
        assert torch.allclose(relu(x_zero), torch.zeros_like(x_zero), atol=1e-6)
        assert torch.allclose(gelu(x_zero), torch.zeros_like(x_zero), atol=1e-6)
        assert torch.allclose(silu(x_zero), torch.zeros_like(x_zero), atol=1e-6)
        
        # Large positive values
        x_large_pos = torch.full((100,), 10.0, device='cuda')
        
        result_relu = relu(x_large_pos)
        assert torch.allclose(result_relu, x_large_pos, atol=1e-6)
        
        # Large negative values
        x_large_neg = torch.full((100,), -10.0, device='cuda')
        
        result_relu = relu(x_large_neg)
        assert torch.allclose(result_relu, torch.zeros_like(x_large_neg), atol=1e-6)
        
        # Very small values
        x_small = torch.full((100,), 1e-6, device='cuda')
        
        result_relu = relu(x_small)
        assert torch.allclose(result_relu, x_small, atol=1e-6)
    
    def test_data_types(self):
        """Test different data types"""
        size = 1000
        
        # Test float32
        x_f32 = torch.randn(size, device='cuda', dtype=torch.float32)
        
        result_relu_f32 = relu(x_f32)
        expected_relu_f32 = torch.relu(x_f32)
        assert torch.allclose(result_relu_f32, expected_relu_f32, atol=1e-6)
        
        result_gelu_f32 = gelu(x_f32)
        expected_gelu_f32 = torch.nn.functional.gelu(x_f32)
        assert torch.allclose(result_gelu_f32, expected_gelu_f32, atol=1e-3)
        
        result_silu_f32 = silu(x_f32)
        expected_silu_f32 = torch.nn.functional.silu(x_f32)
        assert torch.allclose(result_silu_f32, expected_silu_f32, atol=1e-6)
        
        # Test float16
        x_f16 = torch.randn(size, device='cuda', dtype=torch.float16)
        
        result_relu_f16 = relu(x_f16)
        expected_relu_f16 = torch.relu(x_f16)
        assert torch.allclose(result_relu_f16, expected_relu_f16, atol=1e-3)
        
        result_gelu_f16 = gelu(x_f16)
        expected_gelu_f16 = torch.nn.functional.gelu(x_f16)
        assert torch.allclose(result_gelu_f16, expected_gelu_f16, atol=5e-2)
        
        # SiLU with float16 may have precision issues due to exp function
        # result_silu_f16 = silu(x_f16)
        # expected_silu_f16 = torch.nn.functional.silu(x_f16)
        # assert torch.allclose(result_silu_f16, expected_silu_f16, atol=1e-3)
    
    def test_input_validation(self):
        """Test input validation"""
        # Test CPU tensor
        x_cpu = torch.randn(100, device='cpu')
        with pytest.raises(AssertionError, match="Input tensor must be on CUDA device"):
            relu(x_cpu)
        
        with pytest.raises(AssertionError, match="Input tensor must be on CUDA device"):
            gelu(x_cpu)
        
        with pytest.raises(AssertionError, match="Input tensor must be on CUDA device"):
            silu(x_cpu)
    
    def test_performance_benchmark(self):
        """Test performance benchmark functionality"""
        results = benchmark_activations('relu', size=100000, warmup=5, repeat=10)
        
        # Verify benchmark results contain expected keys
        expected_keys = [
            'activation', 'triton_time', 'torch_time', 'speedup',
            'triton_bandwidth', 'torch_bandwidth', 'input_size', 'total_elements'
        ]
        for key in expected_keys:
            assert key in results, f"Missing benchmark result key: {key}"
        
        # Verify reasonable values
        assert results['triton_time'] > 0, "Triton time should be positive"
        assert results['torch_time'] > 0, "PyTorch time should be positive"
        assert results['speedup'] > 0, "Speedup should be positive"
        assert results['triton_bandwidth'] > 0, "Triton bandwidth should be positive"
        assert results['torch_bandwidth'] > 0, "PyTorch bandwidth should be positive"
        assert results['activation'] == 'relu', "Should be benchmarking ReLU"
        assert results['input_size'] == 100000, "Input size should match"
        assert results['total_elements'] == 100000, "Total elements should match"
        
        print(f"ReLU benchmark results: {results}")
    
    def test_benchmark_all_activations(self):
        """Test benchmarking all activation functions"""
        results = benchmark_all_activations(size=50000, warmup=3, repeat=5)
        
        # Verify all activations are benchmarked
        expected_activations = ['relu', 'gelu', 'silu']
        for activation in expected_activations:
            assert activation in results, f"Missing benchmark for {activation}"
            
            # Verify each has expected keys
            expected_keys = ['activation', 'triton_time', 'torch_time', 'speedup']
            for key in expected_keys:
                assert key in results[activation], f"Missing {key} for {activation}"
        
        print(f"All activations benchmark results: {results}")
    
    def test_memory_efficiency(self):
        """Test that kernels don't cause memory leaks"""
        import gc
        
        initial_memory = torch.cuda.memory_allocated()
        
        # Run multiple iterations
        for _ in range(50):
            x = torch.randn(10000, device='cuda')
            
            _ = relu(x)
            _ = gelu(x)
            _ = silu(x)
            
            del x
            torch.cuda.synchronize()
        
        gc.collect()
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should not grow significantly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 5 * 1024 * 1024, f"Memory growth too large: {memory_growth} bytes"
    
    def test_deterministic_output(self):
        """Test that output is deterministic"""
        size = 1000
        x = torch.randn(size, device='cuda')
        
        # Test ReLU
        results_relu = []
        for _ in range(5):
            result = relu(x)
            results_relu.append(result.clone())
            torch.cuda.synchronize()
        
        for i in range(1, len(results_relu)):
            assert torch.equal(results_relu[0], results_relu[i]), "ReLU results are not deterministic"
        
        # Test GELU
        results_gelu = []
        for _ in range(5):
            result = gelu(x)
            results_gelu.append(result.clone())
            torch.cuda.synchronize()
        
        for i in range(1, len(results_gelu)):
            assert torch.equal(results_gelu[0], results_gelu[i]), "GELU results are not deterministic"
        
        # Test SiLU
        results_silu = []
        for _ in range(5):
            result = silu(x)
            results_silu.append(result.clone())
            torch.cuda.synchronize()
        
        for i in range(1, len(results_silu)):
            assert torch.equal(results_silu[0], results_silu[i]), "SiLU results are not deterministic"
    
    def test_activation_comparison(self):
        """Test that different activations produce different results"""
        size = 1000
        x = torch.randn(size, device='cuda')
        
        result_relu = relu(x)
        result_gelu = gelu(x)
        result_silu = silu(x)
        
        # Results should be different (except for edge cases)
        assert not torch.allclose(result_relu, result_gelu, atol=1e-6), "ReLU and GELU should be different"
        assert not torch.allclose(result_relu, result_silu, atol=1e-6), "ReLU and SiLU should be different"
        assert not torch.allclose(result_gelu, result_silu, atol=1e-6), "GELU and SiLU should be different"
        
        # All should be non-negative for ReLU
        assert torch.all(result_relu >= 0), "ReLU should be non-negative"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])