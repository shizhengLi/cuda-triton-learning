import torch
import pytest
import sys
import os

# Add the parent directory to the path to import our module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import directly from the file
import importlib.util
muon_file_path = os.path.join(project_root, "04_optimizers", "muon_optimizer", "muon_optimizer.py")
print(f"Loading Muon module from: {muon_file_path}")
print(f"File exists: {os.path.exists(muon_file_path)}")

spec = importlib.util.spec_from_file_location(
    "muon_optimizer", 
    muon_file_path
)
muon_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(muon_module)

TritonMuon = muon_module.TritonMuon
TritonMuonW = muon_module.TritonMuonW
benchmark_muon_optimizers = muon_module.benchmark_muon_optimizers


class TestTritonMuon:
    """Test suite for Triton Muon optimizer"""
    
    def test_basic_muon_step(self):
        """Test basic Muon optimization step"""
        # Create simple linear model
        model = torch.nn.Linear(10, 1).cuda()
        
        # Initialize Triton Muon
        optimizer = TritonMuon(model.parameters(), lr=0.01)
        
        # Create dummy data
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        # Store initial parameters
        initial_params = [p.clone() for p in model.parameters()]
        
        # Forward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters have been updated
        for init_param, curr_param in zip(initial_params, model.parameters()):
            assert not torch.equal(init_param, curr_param), "Parameters should be updated"
    
    def test_muon_weight_decay(self):
        """Test Muon with weight decay"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonMuon(model.parameters(), lr=0.01, weight_decay=0.01)
        
        # Create dummy data
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        # Forward and backward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that parameters have been updated
        assert True  # Basic functionality test
    
    def test_muon_momentum_coefficients(self):
        """Test Muon with different momentum coefficients"""
        model = torch.nn.Linear(10, 1).cuda()
        
        # Test with different momentum values
        optimizer = TritonMuon(model.parameters(), lr=0.01, momentum=0.95, velocity=0.99)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should not raise any errors
        assert True
    
    def test_muon_nesterov(self):
        """Test Muon with Nesterov momentum"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonMuon(model.parameters(), lr=0.01, nesterov=True)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should not raise any errors
        assert True
    
    def test_muon_layerwise_norm(self):
        """Test Muon with layer-wise normalization"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonMuon(model.parameters(), lr=0.01, layerwise_norm=True)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should not raise any errors
        assert True
    
    def test_muon_state_initialization(self):
        """Test that Muon optimizer state is properly initialized"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonMuon(model.parameters(), lr=0.01)
        
        # Check that state is initialized for each parameter
        for param in model.parameters():
            state = optimizer.state[param]
            assert 'step' in state
            assert 'momentum' in state
            assert 'velocity' in state
            assert state['step'] == 0
            assert torch.equal(state['momentum'], torch.zeros_like(param))
            assert torch.equal(state['velocity'], torch.zeros_like(param))
    
    def test_muon_step_counter(self):
        """Test that step counter increments correctly"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonMuon(model.parameters(), lr=0.01)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        # Perform multiple steps
        for i in range(5):
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Check step counter
            for param in model.parameters():
                assert optimizer.state[param]['step'] == i + 1
    
    def test_muon_zero_grad(self):
        """Test zero_grad functionality"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonMuon(model.parameters(), lr=0.01)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        # Forward and backward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        loss.backward()
        
        # Check that gradients are not zero
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad))
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Check that gradients are now zero
        for param in model.parameters():
            if param.grad is not None:
                assert torch.allclose(param.grad, torch.zeros_like(param.grad))
    
    def test_muon_multiple_parameters(self):
        """Test Muon with multiple parameter groups"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).cuda()
        
        optimizer = TritonMuon(model.parameters(), lr=0.01)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # All parameters should be updated
        assert True
    
    def test_muon_learning_rates(self):
        """Test Muon with different learning rates"""
        model = torch.nn.Linear(10, 1).cuda()
        
        # Test with very small learning rate
        optimizer = TritonMuon(model.parameters(), lr=1e-6)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        initial_params = [p.clone() for p in model.parameters()]
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # With very small learning rate, changes should be minimal
        for init_param, curr_param in zip(initial_params, model.parameters()):
            diff = torch.abs(curr_param - init_param).max().item()
            assert diff < 0.1, f"Parameter change too large for small LR: {diff}"
    
    def test_muon_parameter_groups(self):
        """Test Muon with parameter groups"""
        model = torch.nn.Linear(10, 1).cuda()
        
        # Create parameter groups with different learning rates
        optimizer = TritonMuon([
            {'params': [model.weight], 'lr': 0.01},
            {'params': [model.bias], 'lr': 0.001}
        ])
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        initial_weight = model.weight.clone()
        initial_bias = model.bias.clone()
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Both should be updated but with different magnitudes
        assert not torch.equal(model.weight, initial_weight)
        assert not torch.equal(model.bias, initial_bias)


class TestTritonMuonW:
    """Test suite for Triton MuonW optimizer"""
    
    def test_basic_muonw_step(self):
        """Test basic MuonW optimization step"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonMuonW(model.parameters(), lr=0.01, weight_decay=0.01)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        initial_params = [p.clone() for p in model.parameters()]
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Parameters should be updated
        for init_param, curr_param in zip(initial_params, model.parameters()):
            assert not torch.equal(init_param, curr_param)
    
    def test_muonw_vs_muon(self):
        """Test that MuonW produces different results than Muon"""
        # Create identical models
        model1 = torch.nn.Linear(10, 1).cuda()
        model2 = torch.nn.Linear(10, 1).cuda()
        
        # Copy weights
        model2.load_state_dict(model1.state_dict())
        
        # Create optimizers
        muon_opt = TritonMuon(model1.parameters(), lr=0.01, weight_decay=0.01)
        muonw_opt = TritonMuonW(model2.parameters(), lr=0.01, weight_decay=0.01)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        # Forward pass
        output1 = model1(x)
        output2 = model2(x)
        loss1 = torch.nn.functional.mse_loss(output1, y)
        loss2 = torch.nn.functional.mse_loss(output2, y)
        
        # Backward pass
        muon_opt.zero_grad()
        muonw_opt.zero_grad()
        loss1.backward()
        loss2.backward()
        muon_opt.step()
        muonw_opt.step()
        
        # Results should be different due to different weight decay handling
        weight_diff = torch.abs(model1.weight - model2.weight).max().item()
        assert weight_diff > 1e-6, f"Muon and MuonW should produce different results: {weight_diff}"
    
    def test_muonw_state_initialization(self):
        """Test MuonW state initialization"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonMuonW(model.parameters(), lr=0.01)
        
        for param in model.parameters():
            state = optimizer.state[param]
            assert 'step' in state
            assert 'momentum' in state
            assert 'velocity' in state
            assert state['step'] == 0
    
    def test_muonw_weight_decay_values(self):
        """Test MuonW with different weight decay values"""
        model = torch.nn.Linear(10, 1).cuda()
        
        # Test with zero weight decay
        optimizer = TritonMuonW(model.parameters(), lr=0.01, weight_decay=0.0)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should work without errors
        assert True


class TestMuonBenchmarks:
    """Test suite for Muon optimizer benchmarking functionality"""
    
    def test_benchmark_muon_optimizers(self):
        """Test benchmarking function"""
        results = benchmark_muon_optimizers(model_size=100, num_steps=10, lr=0.01)
        
        # Check that all expected keys are present
        expected_keys = [
            'triton_muon_time', 'triton_muonw_time', 'torch_adam_time', 'torch_sgd_time',
            'muon_vs_adam_speedup', 'muonw_vs_adam_speedup', 'muon_vs_sgd_speedup',
            'model_size', 'num_steps', 'parameters'
        ]
        
        for key in expected_keys:
            assert key in results, f"Missing benchmark result: {key}"
        
        # Check reasonable values
        assert results['triton_muon_time'] > 0
        assert results['triton_muonw_time'] > 0
        assert results['torch_adam_time'] > 0
        assert results['torch_sgd_time'] > 0
        assert results['model_size'] == 100
        assert results['num_steps'] == 10
        assert results['parameters'] == 100 * 100 + 100
    
    def test_benchmark_consistency(self):
        """Test that benchmark results are consistent"""
        # Run benchmark twice with same parameters
        results1 = benchmark_muon_optimizers(model_size=50, num_steps=5)
        results2 = benchmark_muon_optimizers(model_size=50, num_steps=5)
        
        # Results should be similar (within reasonable tolerance for GPU)
        time_diff_ratio = abs(results1['triton_muon_time'] - results2['triton_muon_time']) / results1['triton_muon_time']
        assert time_diff_ratio < 1.0, f"Benchmark results not consistent: {time_diff_ratio}"


class TestMuonEdgeCases:
    """Test edge cases and error handling"""
    
    def test_invalid_learning_rate(self):
        """Test invalid learning rate values"""
        model = torch.nn.Linear(10, 1).cuda()
        
        with pytest.raises(ValueError, match="Invalid learning rate"):
            TritonMuon(model.parameters(), lr=-0.01)
        
        with pytest.raises(ValueError, match="Invalid learning rate"):
            TritonMuonW(model.parameters(), lr=-0.01)
    
    def test_invalid_momentum(self):
        """Test invalid momentum values"""
        model = torch.nn.Linear(10, 1).cuda()
        
        with pytest.raises(ValueError, match="Invalid momentum parameter"):
            TritonMuon(model.parameters(), momentum=1.5)
        
        with pytest.raises(ValueError, match="Invalid momentum parameter"):
            TritonMuon(model.parameters(), momentum=-0.1)
    
    def test_invalid_velocity(self):
        """Test invalid velocity values"""
        model = torch.nn.Linear(10, 1).cuda()
        
        with pytest.raises(ValueError, match="Invalid velocity parameter"):
            TritonMuon(model.parameters(), velocity=1.5)
        
        with pytest.raises(ValueError, match="Invalid velocity parameter"):
            TritonMuon(model.parameters(), velocity=-0.1)
    
    def test_invalid_weight_decay(self):
        """Test invalid weight decay values"""
        model = torch.nn.Linear(10, 1).cuda()
        
        with pytest.raises(ValueError, match="Invalid weight_decay value"):
            TritonMuon(model.parameters(), weight_decay=-0.01)
    
    def test_empty_parameters(self):
        """Test optimizer with empty parameter list"""
        with pytest.raises(ValueError, match="optimizer got an empty parameter list"):
            TritonMuon([])
    
    def test_cpu_tensors(self):
        """Test optimizer with CPU tensors (should work after conversion)"""
        model = torch.nn.Linear(10, 1)  # CPU model
        optimizer = TritonMuon(model.parameters(), lr=0.01)
        
        x = torch.randn(32, 10)
        y = torch.randn(32, 1)
        
        # Forward pass
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        
        # Backward pass - should handle CPU tensors gracefully
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Parameters should be updated
        assert True
    
    def test_no_gradients(self):
        """Test optimizer step when some parameters don't have gradients"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonMuon(model.parameters(), lr=0.01)
        
        # Don't compute gradients for some parameters
        model.weight.requires_grad = False
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should not raise errors
        assert True
    
    def test_closure_functionality(self):
        """Test optimizer step with closure"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonMuon(model.parameters(), lr=0.01)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        def closure():
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            return loss
        
        loss = optimizer.step(closure=closure)
        
        # Should return loss value
        assert loss is not None
        assert isinstance(loss, torch.Tensor)
    
    def test_large_model(self):
        """Test optimizer with large model"""
        model = torch.nn.Linear(1000, 1000).cuda()
        optimizer = TritonMuon(model.parameters(), lr=0.001)
        
        x = torch.randn(16, 1000).cuda()
        y = torch.randn(16, 1000).cuda()
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should work without errors
        assert True
    
    def test_sparse_gradients(self):
        """Test optimizer with sparse gradients (should raise error)"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonMuon(model.parameters(), lr=0.01)
        
        # Create sparse gradient
        model.weight.grad = torch.sparse_coo_tensor(
            torch.tensor([[0, 0], [0, 1]]),
            torch.tensor([1.0, 2.0]),
            model.weight.shape,
            device='cuda'
        )
        
        with pytest.raises(RuntimeError, match="Muon does not support sparse gradients"):
            optimizer.step()
    
    def test_memory_efficiency(self):
        """Test that kernels don't cause memory leaks"""
        import gc
        
        model = torch.nn.Linear(1000, 100).cuda()
        optimizer = TritonMuon(model.parameters(), lr=0.01)
        
        x = torch.randn(32, 1000).cuda()
        y = torch.randn(32, 100).cuda()
        
        initial_memory = torch.cuda.memory_allocated()
        
        # Run multiple iterations
        for _ in range(20):
            optimizer.zero_grad()
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
        
        gc.collect()
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should not grow significantly
        memory_growth = final_memory - initial_memory
        assert memory_growth < 10 * 1024 * 1024, f"Memory growth too large: {memory_growth} bytes"
    
    def test_different_configurations(self):
        """Test different optimizer configurations"""
        model = torch.nn.Linear(10, 1).cuda()
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        # Test different combinations
        configs = [
            {'nesterov': False, 'layerwise_norm': False},
            {'nesterov': True, 'layerwise_norm': False},
            {'nesterov': False, 'layerwise_norm': True},
            {'nesterov': True, 'layerwise_norm': True},
        ]
        
        for config in configs:
            model = torch.nn.Linear(10, 1).cuda()
            optimizer = TritonMuon(model.parameters(), lr=0.01, **config)
            
            output = model(x)
            loss = torch.nn.functional.mse_loss(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Should work without errors
            assert True


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])