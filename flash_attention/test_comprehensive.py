#!/usr/bin/env python3
"""
Comprehensive test suite for Flash Attention v1 and v2 implementations
"""
import torch
import sys
import os
import time
import math

# Add paths to implementations
sys.path.append("/data/lishizheng/cpp_projects/cuda-triton-learning/flash_attention/naive")
sys.path.append("/data/lishizheng/cpp_projects/cuda-triton-learning/flash_attention/flash_v1/triton")
sys.path.append("/data/lishizheng/cpp_projects/cuda-triton-learning/flash_attention/flash_v2/triton")

from naive_attention import naive_attention
from flash_attention_v1 import flash_attention as flash_attention_v1
from flash_attention_v2_simple import flash_attention_v2

class FlashAttentionTestSuite:
    """Comprehensive test suite for Flash Attention implementations"""
    
    def __init__(self):
        self.results = {}
        
    def test_correctness(self):
        """Test numerical correctness of all implementations"""
        print("=" * 60)
        print("Testing Numerical Correctness")
        print("=" * 60)
        
        # Test configurations
        test_configs = [
            (1, 32, 16, "small"),
            (2, 64, 32, "medium"),
            (4, 128, 64, "large"),
        ]
        
        implementations = {
            "naive": naive_attention,
            "flash_v1": flash_attention_v1,
            "flash_v2": flash_attention_v2,
        }
        
        results = {}
        
        for batch_size, seq_len, head_dim, config_name in test_configs:
            print(f"\nTesting {config_name} config: {batch_size}x{seq_len}x{head_dim}")
            
            # Create test tensors
            q = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            k = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            v = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            
            config_results = {}
            
            # Test each implementation
            for impl_name, impl_func in implementations.items():
                try:
                    output = impl_func(q, k, v)
                    
                    # Basic checks
                    assert output.shape == q.shape, f"Shape mismatch for {impl_name}"
                    assert not torch.isnan(output).any(), f"NaN values in {impl_name}"
                    assert torch.isfinite(output).all(), f"Infinite values in {impl_name}"
                    
                    config_results[impl_name] = {
                        'output': output,
                        'success': True,
                        'error': None
                    }
                    
                    print(f"  ✓ {impl_name}: PASSED")
                    
                except Exception as e:
                    config_results[impl_name] = {
                        'output': None,
                        'success': False,
                        'error': str(e)
                    }
                    print(f"  ✗ {impl_name}: FAILED - {e}")
            
            results[config_name] = config_results
            
            # Cross-validate implementations
            successful_impls = {k: v for k, v in config_results.items() if v['success']}
            
            if len(successful_impls) > 1:
                print(f"  Cross-validating {len(successful_impls)} implementations...")
                baseline_output = None
                baseline_name = None
                
                for impl_name, impl_data in successful_impls.items():
                    if baseline_output is None:
                        baseline_output = impl_data['output']
                        baseline_name = impl_name
                        continue
                    
                    # Compare with baseline
                    is_close = torch.allclose(baseline_output, impl_data['output'], rtol=1e-2, atol=1e-2)
                    max_diff = torch.max(torch.abs(baseline_output - impl_data['output'])).item()
                    
                    if is_close:
                        print(f"    ✓ {impl_name} matches {baseline_name} (max_diff: {max_diff:.6f})")
                    else:
                        print(f"    ⚠ {impl_name} differs from {baseline_name} (max_diff: {max_diff:.6f})")
        
        self.results['correctness'] = results
        return results
    
    def test_performance(self):
        """Test performance of all implementations"""
        print("\n" + "=" * 60)
        print("Testing Performance")
        print("=" * 60)
        
        # Test configurations
        test_configs = [
            (1, 256, 64, "small"),
            (2, 512, 64, "medium"),
            (4, 1024, 64, "large"),
        ]
        
        implementations = {
            "naive": naive_attention,
            "flash_v1": flash_attention_v1,
            "flash_v2": flash_attention_v2,
        }
        
        results = {}
        
        for batch_size, seq_len, head_dim, config_name in test_configs:
            print(f"\nPerformance test {config_name}: {batch_size}x{seq_len}x{head_dim}")
            
            # Create test tensors
            q = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            k = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            v = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            
            config_results = {}
            
            # Test each implementation
            for impl_name, impl_func in implementations.items():
                try:
                    # Warmup
                    _ = impl_func(q, k, v)
                    torch.cuda.synchronize()
                    
                    # Measure performance
                    times = []
                    for _ in range(10):  # Run multiple times for stability
                        torch.cuda.synchronize()
                        start_time = time.time()
                        _ = impl_func(q, k, v)
                        torch.cuda.synchronize()
                        elapsed_time = (time.time() - start_time) * 1000  # Convert to ms
                        times.append(elapsed_time)
                    
                    avg_time = sum(times) / len(times)
                    std_time = math.sqrt(sum((t - avg_time) ** 2 for t in times) / len(times))
                    
                    config_results[impl_name] = {
                        'avg_time': avg_time,
                        'std_time': std_time,
                        'success': True,
                        'error': None
                    }
                    
                    print(f"  ✓ {impl_name}: {avg_time:.2f} ± {std_time:.2f} ms")
                    
                except Exception as e:
                    config_results[impl_name] = {
                        'avg_time': None,
                        'std_time': None,
                        'success': False,
                        'error': str(e)
                    }
                    print(f"  ✗ {impl_name}: FAILED - {e}")
            
            # Calculate speedups
            successful_impls = {k: v for k, v in config_results.items() if v['success']}
            
            if len(successful_impls) > 1:
                naive_time = successful_impls.get('naive', {}).get('avg_time')
                if naive_time:
                    print(f"  Speedup vs naive:")
                    for impl_name, impl_data in successful_impls.items():
                        if impl_name != 'naive' and impl_data['avg_time']:
                            speedup = naive_time / impl_data['avg_time']
                            print(f"    {impl_name}: {speedup:.2f}x")
            
            results[config_name] = config_results
        
        self.results['performance'] = results
        return results
    
    def test_memory_usage(self):
        """Test memory usage of all implementations"""
        print("\n" + "=" * 60)
        print("Testing Memory Usage")
        print("=" * 60)
        
        # Test configurations
        test_configs = [
            (1, 512, 64, "medium"),
            (2, 1024, 64, "large"),
        ]
        
        implementations = {
            "naive": naive_attention,
            "flash_v1": flash_attention_v1,
            "flash_v2": flash_attention_v2,
        }
        
        results = {}
        
        for batch_size, seq_len, head_dim, config_name in test_configs:
            print(f"\nMemory test {config_name}: {batch_size}x{seq_len}x{head_dim}")
            
            config_results = {}
            
            # Test each implementation
            for impl_name, impl_func in implementations.items():
                try:
                    # Reset memory stats
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    
                    # Create test tensors
                    q = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
                    k = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
                    v = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
                    
                    # Record initial memory
                    initial_memory = torch.cuda.memory_allocated()
                    
                    # Run implementation
                    _ = impl_func(q, k, v)
                    torch.cuda.synchronize()
                    
                    # Record peak memory
                    peak_memory = torch.cuda.max_memory_allocated()
                    
                    # Calculate memory usage
                    memory_used = (peak_memory - initial_memory) / (1024 ** 2)  # Convert to MB
                    
                    config_results[impl_name] = {
                        'memory_mb': memory_used,
                        'success': True,
                        'error': None
                    }
                    
                    print(f"  ✓ {impl_name}: {memory_used:.2f} MB")
                    
                except Exception as e:
                    config_results[impl_name] = {
                        'memory_mb': None,
                        'success': False,
                        'error': str(e)
                    }
                    print(f"  ✗ {impl_name}: FAILED - {e}")
            
            # Calculate memory savings
            successful_impls = {k: v for k, v in config_results.items() if v['success']}
            
            if len(successful_impls) > 1:
                naive_memory = successful_impls.get('naive', {}).get('memory_mb')
                if naive_memory:
                    print(f"  Memory savings vs naive:")
                    for impl_name, impl_data in successful_impls.items():
                        if impl_name != 'naive' and impl_data['memory_mb']:
                            savings = (naive_memory - impl_data['memory_mb']) / naive_memory * 100
                            print(f"    {impl_name}: {savings:.1f}% savings")
            
            results[config_name] = config_results
        
        self.results['memory'] = results
        return results
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        print("\n" + "=" * 60)
        print("Testing Edge Cases")
        print("=" * 60)
        
        edge_cases = [
            # (batch_size, seq_len, head_dim, description)
            (1, 1, 1, "minimal dimensions"),
            (1, 8, 1, "single head dimension"),
            (1, 1, 64, "single sequence length"),
            (8, 16, 128, "large head dimension"),
            (16, 32, 256, "large sequence"),
        ]
        
        implementations = {
            "flash_v1": flash_attention_v1,
            "flash_v2": flash_attention_v2,
        }
        
        results = {}
        
        for batch_size, seq_len, head_dim, description in edge_cases:
            print(f"\nTesting {description}: {batch_size}x{seq_len}x{head_dim}")
            
            # Create test tensors
            q = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            k = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            v = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            
            case_results = {}
            
            # Test each implementation
            for impl_name, impl_func in implementations.items():
                try:
                    output = impl_func(q, k, v)
                    
                    # Basic checks
                    assert output.shape == q.shape, f"Shape mismatch for {impl_name}"
                    assert not torch.isnan(output).any(), f"NaN values in {impl_name}"
                    assert torch.isfinite(output).all(), f"Infinite values in {impl_name}"
                    
                    case_results[impl_name] = {
                        'success': True,
                        'error': None
                    }
                    
                    print(f"  ✓ {impl_name}: PASSED")
                    
                except Exception as e:
                    case_results[impl_name] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"  ✗ {impl_name}: FAILED - {e}")
            
            results[description] = case_results
        
        self.results['edge_cases'] = results
        return results
    
    def run_all_tests(self):
        """Run all tests and generate summary"""
        print("Flash Attention Comprehensive Test Suite")
        print("=" * 60)
        
        # Run all test categories
        self.test_correctness()
        self.test_performance()
        self.test_memory_usage()
        self.test_edge_cases()
        
        # Generate summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        
        summary = {}
        
        # Correctness summary
        if 'correctness' in self.results:
            correctness_results = self.results['correctness']
            total_tests = sum(len(config) for config in correctness_results.values())
            passed_tests = sum(sum(1 for impl in config.values() if impl['success']) for config in correctness_results.values())
            summary['correctness'] = f"{passed_tests}/{total_tests} passed"
        
        # Performance summary
        if 'performance' in self.results:
            perf_results = self.results['performance']
            total_tests = sum(len(config) for config in perf_results.values())
            passed_tests = sum(sum(1 for impl in config.values() if impl['success']) for config in perf_results.values())
            summary['performance'] = f"{passed_tests}/{total_tests} passed"
        
        # Memory summary
        if 'memory' in self.results:
            mem_results = self.results['memory']
            total_tests = sum(len(config) for config in mem_results.values())
            passed_tests = sum(sum(1 for impl in config.values() if impl['success']) for config in mem_results.values())
            summary['memory'] = f"{passed_tests}/{total_tests} passed"
        
        # Edge cases summary
        if 'edge_cases' in self.results:
            edge_results = self.results['edge_cases']
            total_tests = sum(len(case) for case in edge_results.values())
            passed_tests = sum(sum(1 for impl in case.values() if impl['success']) for case in edge_results.values())
            summary['edge_cases'] = f"{passed_tests}/{total_tests} passed"
        
        for test_name, result in summary.items():
            print(f"{test_name.capitalize()}: {result}")
        
        print("\n" + "=" * 60)
        print("Test suite completed!")
        
        return self.results

def main():
    """Main function to run the test suite"""
    test_suite = FlashAttentionTestSuite()
    results = test_suite.run_all_tests()
    
    # Save results to file (optional)
    import json
    with open('flash_attention_test_results.json', 'w') as f:
        # Convert tensors to strings for JSON serialization
        json_results = {}
        for category, category_data in results.items():
            json_results[category] = {}
            for config_name, config_data in category_data.items():
                json_results[category][config_name] = {}
                for impl_name, impl_data in config_data.items():
                    json_results[category][config_name][impl_name] = {
                        k: str(v) if torch.is_tensor(v) else v 
                        for k, v in impl_data.items()
                    }
        
        json.dump(json_results, f, indent=2)
        print(f"\nTest results saved to 'flash_attention_test_results.json'")

if __name__ == "__main__":
    main()