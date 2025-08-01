import torch
import triton
import triton.language as tl
import math


@triton.jit
def muon_update_kernel(
    param_ptr, grad_ptr, momentum_ptr, velocity_ptr,
    lr, momentum_coef, velocity_coef, weight_decay, nesterov,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for Muon optimizer parameter update
    
    Muon optimizer combines momentum-based updates with adaptive learning rates
    and is designed for scaling LLM training to trillion parameters.
    
    Args:
        param_ptr: Pointer to parameters
        grad_ptr: Pointer to gradients  
        momentum_ptr: Pointer to momentum buffer
        velocity_ptr: Pointer to velocity buffer
        lr: Learning rate
        momentum_coef: Momentum coefficient
        velocity_coef: Velocity coefficient
        weight_decay: Weight decay
        nesterov: Whether to use Nesterov momentum
        n_elements: Number of parameters to update
        BLOCK_SIZE: Block size for parallel processing
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load parameters and gradients
    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    
    # Load momentum and velocity buffers
    momentum = tl.load(momentum_ptr + offsets, mask=mask)
    velocity = tl.load(velocity_ptr + offsets, mask=mask)
    
    # Apply weight decay if specified
    if weight_decay > 0.0:
        grad = grad + weight_decay * param
    
    # Update momentum
    momentum = momentum_coef * momentum + (1.0 - momentum_coef) * grad
    
    # Compute velocity (adaptive learning rate component)
    grad_squared = grad * grad
    velocity = velocity_coef * velocity + (1.0 - velocity_coef) * grad_squared
    
    # Compute adaptive learning rate
    adaptive_lr = lr / (tl.sqrt(velocity) + 1e-8)
    
    # Update parameters
    if nesterov:
        # Nesterov momentum: look ahead
        lookahead_param = param - adaptive_lr * momentum
        update = momentum_coef * momentum + (1.0 - momentum_coef) * grad
        param = param - adaptive_lr * update
    else:
        # Standard momentum update
        param = param - adaptive_lr * momentum
    
    # Store results
    tl.store(param_ptr + offsets, param, mask=mask)
    tl.store(momentum_ptr + offsets, momentum, mask=mask)
    tl.store(velocity_ptr + offsets, velocity, mask=mask)


@triton.jit
def muon_layerwise_kernel(
    param_ptr, grad_ptr, momentum_ptr, velocity_ptr,
    lr, momentum_coef, velocity_coef, weight_decay, nesterov,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for layer-wise Muon optimization
    
    This kernel implements layer-wise normalization of gradients
    for improved stability in very deep networks.
    
    Args:
        param_ptr: Pointer to parameters
        grad_ptr: Pointer to gradients  
        momentum_ptr: Pointer to momentum buffer
        velocity_ptr: Pointer to velocity buffer
        lr: Learning rate
        momentum_coef: Momentum coefficient
        velocity_coef: Velocity coefficient
        weight_decay: Weight decay
        nesterov: Whether to use Nesterov momentum
        n_elements: Number of parameters to update
        BLOCK_SIZE: Block size for parallel processing
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load parameters and gradients
    param = tl.load(param_ptr + offsets, mask=mask)
    grad = tl.load(grad_ptr + offsets, mask=mask)
    
    # Load momentum and velocity buffers
    momentum = tl.load(momentum_ptr + offsets, mask=mask)
    velocity = tl.load(velocity_ptr + offsets, mask=mask)
    
    # Apply weight decay if specified
    if weight_decay > 0.0:
        grad = grad + weight_decay * param
    
    # Layer-wise gradient normalization
    grad_norm = tl.sqrt(tl.sum(grad * grad))
    grad_scale = 1.0 / (grad_norm + 1e-8)
    normalized_grad = grad * grad_scale
    
    # Update momentum with normalized gradients
    momentum = momentum_coef * momentum + (1.0 - momentum_coef) * normalized_grad
    
    # Compute velocity with normalized gradients
    grad_squared = normalized_grad * normalized_grad
    velocity = velocity_coef * velocity + (1.0 - velocity_coef) * grad_squared
    
    # Compute adaptive learning rate
    adaptive_lr = lr / (tl.sqrt(velocity) + 1e-8)
    
    # Update parameters
    if nesterov:
        # Nesterov momentum with normalized gradients
        lookahead_param = param - adaptive_lr * momentum
        update = momentum_coef * momentum + (1.0 - momentum_coef) * normalized_grad
        param = param - adaptive_lr * update
    else:
        # Standard momentum update
        param = param - adaptive_lr * momentum
    
    # Store results
    tl.store(param_ptr + offsets, param, mask=mask)
    tl.store(momentum_ptr + offsets, momentum, mask=mask)
    tl.store(velocity_ptr + offsets, velocity, mask=mask)


class TritonMuon:
    """
    Muon optimizer implemented with Triton kernels
    
    Muon: Scaling LLM Training to Trillion Parameters
    This optimizer combines momentum-based updates with adaptive learning rates
    and layer-wise normalization for stable training of very large models.
    """
    
    def __init__(self, params, lr=1e-3, momentum=0.9, velocity=0.999, 
                 weight_decay=0.0, nesterov=False, layerwise_norm=False):
        """
        Initialize Muon optimizer
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (default: 1e-3)
            momentum: Momentum coefficient (default: 0.9)
            velocity: Velocity coefficient for adaptive learning rate (default: 0.999)
            weight_decay: Weight decay (L2 penalty) (default: 0.0)
            nesterov: Whether to use Nesterov momentum (default: False)
            layerwise_norm: Whether to use layer-wise gradient normalization (default: False)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum parameter: {momentum}")
        if not 0.0 <= velocity < 1.0:
            raise ValueError(f"Invalid velocity parameter: {velocity}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        self.defaults = dict(
            lr=lr, momentum=momentum, velocity=velocity, 
            weight_decay=weight_decay, nesterov=nesterov, layerwise_norm=layerwise_norm
        )
        self.state = {}
        self.param_groups = []
        
        # Initialize parameter groups
        if isinstance(params, torch.Tensor):
            params = [params]
        
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        
        for param_group in param_groups:
            self.add_param_group(param_group)
    
    def add_param_group(self, param_group):
        """Add a param group to the optimizer"""
        if not isinstance(param_group, dict):
            raise TypeError("param group must be a dict")
        
        params = param_group['params']
        if isinstance(params, torch.Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections')
        else:
            param_group['params'] = list(params)
        
        for param in param_group['params']:
            if not isinstance(param, torch.Tensor):
                raise TypeError("optimizer can only optimize Tensors")
            if not param.requires_grad:
                raise ValueError("can't optimize a parameter that doesn't require gradients")
            if param.is_sparse:
                raise ValueError("sparse parameters are not supported")
        
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
        
        self.param_groups.append(param_group)
        
        # Initialize state for new parameters
        for param in param_group['params']:
            if param not in self.state:
                self.state[param] = {}
                state = self.state[param]
                state['step'] = 0
                # Momentum buffer
                state['momentum'] = torch.zeros_like(param, device='cuda')
                # Velocity buffer for adaptive learning rate
                state['velocity'] = torch.zeros_like(param, device='cuda')
    
    def zero_grad(self, set_to_none: bool = False):
        """
        Sets the gradients of all optimized `torch.Tensor`s to zero.
        
        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        if p.grad.grad_fn is not None:
                            p.grad.detach_()
                        else:
                            p.grad.requires_grad_(False)
                        p.grad.zero_()
    
    def step(self, closure=None):
        """
        Perform a single optimization step
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Muon does not support sparse gradients')
                
                state = self.state.setdefault(p, {})
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p, device='cuda')
                    state['velocity'] = torch.zeros_like(p, device='cuda')
                
                # Ensure all tensors are on CUDA
                if not p.is_cuda:
                    p = p.cuda()
                if not grad.is_cuda:
                    grad = grad.cuda()
                
                momentum, velocity = state['momentum'], state['velocity']
                
                # Update step counter
                state['step'] += 1
                step = state['step']
                
                # Get hyperparameters
                lr = group['lr']
                momentum_coef = group['momentum']
                velocity_coef = group['velocity']
                weight_decay = group['weight_decay']
                nesterov = group['nesterov']
                layerwise_norm = group['layerwise_norm']
                
                # Flatten tensors for kernel processing
                param_flat = p.view(-1)
                grad_flat = grad.view(-1)
                momentum_flat = momentum.view(-1)
                velocity_flat = velocity.view(-1)
                
                n_elements = param_flat.numel()
                BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
                grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
                
                # Choose kernel based on configuration
                if layerwise_norm:
                    # Use layer-wise normalization kernel
                    muon_layerwise_kernel[(grid_size,)](
                        param_flat, grad_flat, momentum_flat, velocity_flat,
                        lr, momentum_coef, velocity_coef, weight_decay, nesterov,
                        n_elements,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                else:
                    # Use standard Muon kernel
                    muon_update_kernel[(grid_size,)](
                        param_flat, grad_flat, momentum_flat, velocity_flat,
                        lr, momentum_coef, velocity_coef, weight_decay, nesterov,
                        n_elements,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                
                # Use in-place operations for parameters that require gradients
                p_data = p.data
                p_data.copy_(param_flat.view(p.shape))
                momentum.copy_(momentum_flat.view(p.shape))
                velocity.copy_(velocity_flat.view(p.shape))
        
        return loss


class TritonMuonW(TritonMuon):
    """
    MuonW optimizer implemented with Triton kernels
    
    MuonW: Decoupled Weight Decay variant of Muon optimizer
    Similar to AdamW, this decouples weight decay from gradient-based updates.
    """
    
    def __init__(self, params, lr=1e-3, momentum=0.9, velocity=0.999, 
                 weight_decay=1e-2, nesterov=False, layerwise_norm=False):
        """
        Initialize MuonW optimizer
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (default: 1e-3)
            momentum: Momentum coefficient (default: 0.9)
            velocity: Velocity coefficient for adaptive learning rate (default: 0.999)
            weight_decay: Weight decay (L2 penalty) (default: 1e-2)
            nesterov: Whether to use Nesterov momentum (default: False)
            layerwise_norm: Whether to use layer-wise gradient normalization (default: False)
        """
        super().__init__(params, lr, momentum, velocity, weight_decay, nesterov, layerwise_norm)
    
    def step(self, closure=None):
        """
        Perform a single optimization step with decoupled weight decay
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('MuonW does not support sparse gradients')
                
                state = self.state.setdefault(p, {})
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum'] = torch.zeros_like(p, device='cuda')
                    state['velocity'] = torch.zeros_like(p, device='cuda')
                
                # Ensure all tensors are on CUDA
                if not p.is_cuda:
                    p = p.cuda()
                if not grad.is_cuda:
                    grad = grad.cuda()
                
                momentum, velocity = state['momentum'], state['velocity']
                
                # Update step counter
                state['step'] += 1
                step = state['step']
                
                # Get hyperparameters
                lr = group['lr']
                momentum_coef = group['momentum']
                velocity_coef = group['velocity']
                weight_decay = group['weight_decay']
                nesterov = group['nesterov']
                layerwise_norm = group['layerwise_norm']
                
                # Apply weight decay first (decoupled)
                if weight_decay > 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)
                
                # Flatten tensors for kernel processing
                param_flat = p.view(-1)
                grad_flat = grad.view(-1)
                momentum_flat = momentum.view(-1)
                velocity_flat = velocity.view(-1)
                
                n_elements = param_flat.numel()
                BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
                grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
                
                # Choose kernel based on configuration
                if layerwise_norm:
                    # Use layer-wise normalization kernel
                    muon_layerwise_kernel[(grid_size,)](
                        param_flat, grad_flat, momentum_flat, velocity_flat,
                        lr, momentum_coef, velocity_coef, 0.0, nesterov,  # No weight decay in kernel
                        n_elements,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                else:
                    # Use standard Muon kernel
                    muon_update_kernel[(grid_size,)](
                        param_flat, grad_flat, momentum_flat, velocity_flat,
                        lr, momentum_coef, velocity_coef, 0.0, nesterov,  # No weight decay in kernel
                        n_elements,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                
                # Use in-place operations for parameters that require gradients
                p_data = p.data
                p_data.copy_(param_flat.view(p.shape))
                momentum.copy_(momentum_flat.view(p.shape))
                velocity.copy_(velocity_flat.view(p.shape))
        
        return loss


def benchmark_muon_optimizers(model_size=1000, num_steps=100, lr=1e-3):
    """
    Benchmark Triton Muon optimizers against other optimizers
    
    Args:
        model_size: Number of parameters in the model
        num_steps: Number of optimization steps
        lr: Learning rate
        
    Returns:
        dict: Benchmark results
    """
    import time
    
    # Create dummy model and data
    model = torch.nn.Linear(model_size, model_size).cuda()
    x = torch.randn(32, model_size).cuda()
    y = torch.randn(32, model_size).cuda()
    criterion = torch.nn.MSELoss()
    
    # Benchmark Triton Muon
    triton_model = torch.nn.Linear(model_size, model_size).cuda()
    triton_muon = TritonMuon(triton_model.parameters(), lr=lr)
    
    start_time = time.time()
    for _ in range(num_steps):
        triton_muon.zero_grad()
        output = triton_model(x)
        loss = criterion(output, y)
        loss.backward()
        triton_muon.step()
    triton_time = time.time() - start_time
    
    # Benchmark Triton MuonW
    tritonw_model = torch.nn.Linear(model_size, model_size).cuda()
    triton_muonw = TritonMuonW(tritonw_model.parameters(), lr=lr)
    
    start_time = time.time()
    for _ in range(num_steps):
        triton_muonw.zero_grad()
        output = tritonw_model(x)
        loss = criterion(output, y)
        loss.backward()
        triton_muonw.step()
    tritonw_time = time.time() - start_time
    
    # Benchmark PyTorch Adam for comparison
    torch_model = torch.nn.Linear(model_size, model_size).cuda()
    torch_adam = torch.optim.Adam(torch_model.parameters(), lr=lr)
    
    start_time = time.time()
    for _ in range(num_steps):
        torch_adam.zero_grad()
        output = torch_model(x)
        loss = criterion(output, y)
        loss.backward()
        torch_adam.step()
    torch_time = time.time() - start_time
    
    # Benchmark PyTorch SGD with momentum
    torch_sgd_model = torch.nn.Linear(model_size, model_size).cuda()
    torch_sgd = torch.optim.SGD(torch_sgd_model.parameters(), lr=lr, momentum=0.9)
    
    start_time = time.time()
    for _ in range(num_steps):
        torch_sgd.zero_grad()
        output = torch_sgd_model(x)
        loss = criterion(output, y)
        loss.backward()
        torch_sgd.step()
    sgd_time = time.time() - start_time
    
    return {
        'triton_muon_time': triton_time,
        'triton_muonw_time': tritonw_time,
        'torch_adam_time': torch_time,
        'torch_sgd_time': sgd_time,
        'muon_vs_adam_speedup': torch_time / triton_time,
        'muonw_vs_adam_speedup': torch_time / tritonw_time,
        'muon_vs_sgd_speedup': sgd_time / triton_time,
        'model_size': model_size,
        'num_steps': num_steps,
        'parameters': model_size * model_size + model_size
    }