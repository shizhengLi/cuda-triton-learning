#!/usr/bin/env python3
"""
Test Flash Attention v2 implementation
"""
import torch
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flash_attention_v2_simple import flash_attention_v2

def test_flash_attention_v2_basic():
    """Test basic functionality of Flash Attention v2"""
    print("Testing Flash Attention v2 basic functionality...")
    
    # Test parameters
    batch_size = 2
    seq_len = 64
    head_dim = 32
    
    # Create test tensors
    q = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
    
    try:
        # Run Flash Attention v2
        output = flash_attention_v2(q, k, v)
        
        # Check output shape
        assert output.shape == q.shape, f"Expected shape {q.shape}, got {output.shape}"
        
        # Check output is not NaN
        assert not torch.isnan(output).any(), "Output contains NaN values"
        
        # Check output is finite
        assert torch.isfinite(output).all(), "Output contains infinite values"
        
        print(f"✓ Basic test passed: output shape {output.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
        return False

def test_flash_attention_v2_vs_naive():
    """Test Flash Attention v2 against naive implementation"""
    print("Testing Flash Attention v2 vs naive implementation...")
    
    # Import naive implementation
    sys.path.append("/data/lishizheng/cpp_projects/cuda-triton-learning/flash_attention/naive")
    from naive_attention import naive_attention
    
    # Test parameters
    batch_size = 2
    seq_len = 32
    head_dim = 16
    
    # Create test tensors (without requires_grad for comparison)
    q = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
    k = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
    v = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
    
    try:
        # Run both implementations
        naive_output = naive_attention(q, k, v)
        flash_output = flash_attention_v2(q, k, v)
        
        # Check numerical equivalence
        is_close = torch.allclose(naive_output, flash_output, rtol=1e-2, atol=1e-2)
        
        if is_close:
            print(f"✓ Numerical equivalence test passed")
            return True
        else:
            print(f"✗ Numerical equivalence test failed")
            print(f"  Naive output range: [{naive_output.min():.6f}, {naive_output.max():.6f}]")
            print(f"  Flash output range: [{flash_output.min():.6f}, {flash_output.max():.6f}]")
            print(f"  Max difference: {torch.max(torch.abs(naive_output - flash_output)):.6f}")
            return False
            
    except Exception as e:
        print(f"✗ Numerical equivalence test failed: {e}")
        return False

def test_flash_attention_v2_different_sizes():
    """Test Flash Attention v2 with different tensor sizes"""
    print("Testing Flash Attention v2 with different sizes...")
    
    test_cases = [
        (1, 16, 8),    # Small
        (2, 32, 16),   # Medium
        (4, 64, 32),   # Large
    ]
    
    passed = 0
    total = len(test_cases)
    
    for batch_size, seq_len, head_dim in test_cases:
        try:
            # Create test tensors
            q = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            k = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            v = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
            
            # Run Flash Attention v2
            output = flash_attention_v2(q, k, v)
            
            # Basic checks
            assert output.shape == q.shape
            assert not torch.isnan(output).any()
            assert torch.isfinite(output).all()
            
            passed += 1
            print(f"  ✓ Size {batch_size}x{seq_len}x{head_dim} passed")
            
        except Exception as e:
            print(f"  ✗ Size {batch_size}x{seq_len}x{head_dim} failed: {e}")
    
    success_rate = passed / total
    print(f"  Size tests: {passed}/{total} passed ({success_rate:.1%})")
    return success_rate >= 0.8

def test_flash_attention_v2_performance():
    """Test performance of Flash Attention v2"""
    print("Testing Flash Attention v2 performance...")
    
    # Test parameters
    batch_size = 2
    seq_len = 128
    head_dim = 32
    
    try:
        # Create test tensors
        q = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
        k = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
        v = torch.randn(batch_size, seq_len, head_dim, device='cuda', dtype=torch.float32)
        
        # Warmup
        _ = flash_attention_v2(q, k, v)
        torch.cuda.synchronize()
        
        # Measure performance
        import time
        start_time = time.time()
        output = flash_attention_v2(q, k, v)
        torch.cuda.synchronize()
        elapsed_time = (time.time() - start_time) * 1000
        
        print(f"  ✓ Performance test passed: {elapsed_time:.2f} ms")
        return True
        
    except Exception as e:
        print(f"  ✗ Performance test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Flash Attention v2 Test Suite")
    print("=" * 50)
    
    # Run all tests
    tests = [
        test_flash_attention_v2_basic,
        test_flash_attention_v2_vs_naive,
        test_flash_attention_v2_different_sizes,
        test_flash_attention_v2_performance,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
        print()
    
    print("=" * 50)
    print(f"Test Summary: {passed}/{total} tests passed")
    print("=" * 50)
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)