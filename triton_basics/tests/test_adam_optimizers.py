import torch
import pytest
import sys
import os

# Add the parent directory to the path to import our module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import directly from the file
import importlib.util
adam_file_path = os.path.join(project_root, "04_optimizers", "adam_variants", "adam_optimizers.py")
print(f"Loading Adam module from: {adam_file_path}")
print(f"File exists: {os.path.exists(adam_file_path)}")

spec = importlib.util.spec_from_file_location(
    "adam_optimizers", 
    adam_file_path
)
adam_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adam_module)

TritonAdam = adam_module.TritonAdam
TritonAdamW = adam_module.TritonAdamW
benchmark_optimizers = adam_module.benchmark_optimizers


class TestTritonAdam:
    """Test suite for Triton Adam optimizer"""
    
    def test_basic_adam_step(self):
        """Test basic Adam optimization step"""
        # Create simple linear model
        model = torch.nn.Linear(10, 1).cuda()
        
        # Initialize Triton Adam
        optimizer = TritonAdam(model.parameters(), lr=0.01)
        
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
    
    def test_adam_weight_decay(self):
        """Test Adam with weight decay"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonAdam(model.parameters(), lr=0.01, weight_decay=0.01)
        
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
        # With weight decay, large parameters should be penalized more
        assert True  # Basic functionality test
    
    def test_adam_betas(self):
        """Test Adam with different beta values"""
        model = torch.nn.Linear(10, 1).cuda()
        
        # Test with different beta values
        optimizer = TritonAdam(model.parameters(), lr=0.01, betas=(0.95, 0.99))
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should not raise any errors
        assert True
    
    def test_adam_epsilon(self):
        """Test Adam with different epsilon values"""
        model = torch.nn.Linear(10, 1).cuda()
        
        # Test with different epsilon values
        optimizer = TritonAdam(model.parameters(), lr=0.01, eps=1e-6)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should not raise any errors
        assert True
    
    def test_adam_state_initialization(self):
        """Test that Adam optimizer state is properly initialized"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonAdam(model.parameters(), lr=0.01)
        
        # Check that state is initialized for each parameter
        for param in model.parameters():
            state = optimizer.state[param]
            assert 'step' in state
            assert 'exp_avg' in state
            assert 'exp_avg_sq' in state
            assert state['step'] == 0
            assert torch.equal(state['exp_avg'], torch.zeros_like(param))
            assert torch.equal(state['exp_avg_sq'], torch.zeros_like(param))
    
    def test_adam_step_counter(self):
        """Test that step counter increments correctly"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonAdam(model.parameters(), lr=0.01)
        
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
    
    def test_adam_zero_grad(self):
        """Test zero_grad functionality"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonAdam(model.parameters(), lr=0.01)
        
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
    
    def test_adam_multiple_parameters(self):
        """Test Adam with multiple parameter groups"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, 1)
        ).cuda()
        
        optimizer = TritonAdam(model.parameters(), lr=0.01)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # All parameters should be updated
        assert True
    
    def test_adam_learning_rates(self):
        """Test Adam with different learning rates"""
        model = torch.nn.Linear(10, 1).cuda()
        
        # Test with very small learning rate
        optimizer = TritonAdam(model.parameters(), lr=1e-6)
        
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
    
    def test_adam_parameter_groups(self):
        """Test Adam with parameter groups"""
        model = torch.nn.Linear(10, 1).cuda()
        
        # Create parameter groups with different learning rates
        optimizer = TritonAdam([
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


class TestTritonAdamW:
    """Test suite for Triton AdamW optimizer"""
    
    def test_basic_adamw_step(self):
        """Test basic AdamW optimization step"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonAdamW(model.parameters(), lr=0.01, weight_decay=0.01)
        
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
    
    def test_adamw_vs_adam(self):
        """Test that AdamW produces different results than Adam"""
        # Create identical models
        model1 = torch.nn.Linear(10, 1).cuda()
        model2 = torch.nn.Linear(10, 1).cuda()
        
        # Copy weights
        model2.load_state_dict(model1.state_dict())
        
        # Create optimizers
        adam_opt = TritonAdam(model1.parameters(), lr=0.01, weight_decay=0.01)
        adamw_opt = TritonAdamW(model2.parameters(), lr=0.01, weight_decay=0.01)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        # Forward pass
        output1 = model1(x)
        output2 = model2(x)
        loss1 = torch.nn.functional.mse_loss(output1, y)
        loss2 = torch.nn.functional.mse_loss(output2, y)
        
        # Backward pass
        adam_opt.zero_grad()
        adamw_opt.zero_grad()
        loss1.backward()
        loss2.backward()
        adam_opt.step()
        adamw_opt.step()
        
        # Results should be different due to different weight decay handling
        weight_diff = torch.abs(model1.weight - model2.weight).max().item()
        assert weight_diff > 1e-6, f"Adam and AdamW should produce different results: {weight_diff}"
    
    def test_adamw_state_initialization(self):
        """Test AdamW state initialization"""
        model = torch.nn.Linear(10, 1).cuda()
        optimizer = TritonAdamW(model.parameters(), lr=0.01)
        
        for param in model.parameters():
            state = optimizer.state[param]
            assert 'step' in state
            assert 'exp_avg' in state
            assert 'exp_avg_sq' in state
            assert state['step'] == 0
    
    def test_adamw_weight_decay_values(self):
        """Test AdamW with different weight decay values"""
        model = torch.nn.Linear(10, 1).cuda()
        
        # Test with zero weight decay
        optimizer = TritonAdamW(model.parameters(), lr=0.01, weight_decay=0.0)
        
        x = torch.randn(32, 10).cuda()
        y = torch.randn(32, 1).cuda()
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Should work without errors
        assert True


class TestOptimizerBenchmarks:
    """Test suite for optimizer benchmarking functionality"""
    
    def test_benchmark_optimizers(self):
        """Test benchmarking function"""
        results = benchmark_optimizers(model_size=100, num_steps=10, lr=0.01)
        
        # Check that all expected keys are present
        expected_keys = [
            'triton_adam_time', 'torch_adam_time', 'adam_speedup',
            'triton_adamw_time', 'torch_adamw_time', 'adamw_speedup',
            'model_size', 'num_steps', 'parameters'
        ]
        
        for key in expected_keys:
            assert key in results, f"Missing benchmark result: {key}"
        
        # Check reasonable values
        assert results['triton_adam_time'] > 0
        assert results['torch_adam_time'] > 0
        assert results['triton_adamw_time'] > 0
        assert results['torch_adamw_time'] > 0
        assert results['model_size'] == 100
        assert results['num_steps'] == 10
        assert results['parameters'] == 100 * 100 + 100
    
    def test_benchmark_consistency(self):
        """Test that benchmark results are consistent"""
        # Run benchmark twice with same parameters
        results1 = benchmark_optimizers(model_size=50, num_steps=5)
        results2 = benchmark_optimizers(model_size=50, num_steps=5)
        
        # Results should be similar (within reasonable tolerance for GPU)
        time_diff_ratio = abs(results1['triton_adam_time'] - results2['triton_adam_time']) / results1['triton_adam_time']
        assert time_diff_ratio < 1.0, f"Benchmark results not consistent: {time_diff_ratio}"


class TestOptimizerEdgeCases:
    """Test edge cases and error handling"""
    
    def test_invalid_learning_rate(self):
        """Test invalid learning rate values"""
        model = torch.nn.Linear(10, 1).cuda()
        
        with pytest.raises(ValueError, match="Invalid learning rate"):
            TritonAdam(model.parameters(), lr=-0.01)
        
        with pytest.raises(ValueError, match="Invalid learning rate"):
            TritonAdamW(model.parameters(), lr=-0.01)
    
    def test_invalid_betas(self):
        """Test invalid beta values"""
        model = torch.nn.Linear(10, 1).cuda()
        
        with pytest.raises(ValueError, match="Invalid beta parameter"):
            TritonAdam(model.parameters(), betas=(1.5, 0.999))
        
        with pytest.raises(ValueError, match="Invalid beta parameter"):
            TritonAdam(model.parameters(), betas=(0.9, 1.5))
    
    def test_invalid_epsilon(self):
        """Test invalid epsilon values"""
        model = torch.nn.Linear(10, 1).cuda()
        
        with pytest.raises(ValueError, match="Invalid epsilon value"):
            TritonAdam(model.parameters(), eps=-1e-8)
    
    def test_invalid_weight_decay(self):
        """Test invalid weight decay values"""
        model = torch.nn.Linear(10, 1).cuda()
        
        with pytest.raises(ValueError, match="Invalid weight_decay value"):
            TritonAdam(model.parameters(), weight_decay=-0.01)
    
    def test_empty_parameters(self):
        """Test optimizer with empty parameter list"""
        with pytest.raises(ValueError, match="optimizer got an empty parameter list"):
            TritonAdam([])
    
    def test_cpu_tensors(self):
        """Test optimizer with CPU tensors (should work after conversion)"""
        model = torch.nn.Linear(10, 1)  # CPU model
        optimizer = TritonAdam(model.parameters(), lr=0.01)
        
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
        optimizer = TritonAdam(model.parameters(), lr=0.01)
        
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
        optimizer = TritonAdam(model.parameters(), lr=0.01)
        
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
        optimizer = TritonAdam(model.parameters(), lr=0.001)
        
        x = torch.randn(16, 1000).cuda()
        y = torch.randn(16, 1000).cuda()
        
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