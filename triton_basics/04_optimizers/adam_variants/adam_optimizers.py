import torch
import triton
import triton.language as tl
import math


@triton.jit
def adam_update_kernel(
    param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
    lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for Adam optimizer parameter update
    
    Args:
        param_ptr: Pointer to parameters
        grad_ptr: Pointer to gradients  
        exp_avg_ptr: Pointer to exponential moving average of gradients
        exp_avg_sq_ptr: Pointer to exponential moving average of squared gradients
        lr: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        eps: Small constant for numerical stability
        weight_decay: Weight decay (L2 regularization)
        bias_correction1: Bias correction for first moment
        bias_correction2: Bias correction for second moment
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
    
    # Load exponential moving averages
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)
    
    # Apply weight decay if specified
    if weight_decay > 0.0:
        grad = grad + weight_decay * param
    
    # Update biased first moment estimate
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
    
    # Update biased second raw moment estimate
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
    
    # Compute bias-corrected moment estimates
    exp_avg_corrected = exp_avg / bias_correction1
    exp_avg_sq_corrected = exp_avg_sq / bias_correction2
    
    # Update parameters
    param_update = exp_avg_corrected / (tl.sqrt(exp_avg_sq_corrected) + eps)
    param = param - lr * param_update
    
    # Store results
    tl.store(param_ptr + offsets, param, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)


@triton.jit
def adamw_update_kernel(
    param_ptr, grad_ptr, exp_avg_ptr, exp_avg_sq_ptr,
    lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for AdamW optimizer parameter update
    AdamW decouples weight decay from the gradient-based update
    
    Args:
        param_ptr: Pointer to parameters
        grad_ptr: Pointer to gradients  
        exp_avg_ptr: Pointer to exponential moving average of gradients
        exp_avg_sq_ptr: Pointer to exponential moving average of squared gradients
        lr: Learning rate
        beta1: First moment decay rate
        beta2: Second moment decay rate
        eps: Small constant for numerical stability
        weight_decay: Weight decay (L2 regularization)
        bias_correction1: Bias correction for first moment
        bias_correction2: Bias correction for second moment
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
    
    # Load exponential moving averages
    exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask)
    exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask)
    
    # Update biased first moment estimate
    exp_avg = beta1 * exp_avg + (1.0 - beta1) * grad
    
    # Update biased second raw moment estimate
    exp_avg_sq = beta2 * exp_avg_sq + (1.0 - beta2) * grad * grad
    
    # Compute bias-corrected moment estimates
    exp_avg_corrected = exp_avg / bias_correction1
    exp_avg_sq_corrected = exp_avg_sq / bias_correction2
    
    # Update parameters (AdamW applies weight decay separately)
    param_update = exp_avg_corrected / (tl.sqrt(exp_avg_sq_corrected) + eps)
    param = param * (1.0 - lr * weight_decay) - lr * param_update
    
    # Store results
    tl.store(param_ptr + offsets, param, mask=mask)
    tl.store(exp_avg_ptr + offsets, exp_avg, mask=mask)
    tl.store(exp_avg_sq_ptr + offsets, exp_avg_sq, mask=mask)


class TritonAdam:
    """
    Adam optimizer implemented with Triton kernels
    
    Adam: A Method for Stochastic Optimization
    https://arxiv.org/abs/1412.6980
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, amsgrad=False):
        """
        Initialize Adam optimizer
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (default: 1e-3)
            betas: Coefficients for computing running averages (default: (0.9, 0.999))
            eps: Term added to denominator to improve numerical stability (default: 1e-8)
            weight_decay: Weight decay (L2 penalty) (default: 0)
            amsgrad: Whether to use the AMSGrad variant (default: False)
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        self.defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
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
                state['exp_avg'] = torch.zeros_like(param, device='cuda')
                state['exp_avg_sq'] = torch.zeros_like(param, device='cuda')
                if param_group.get('amsgrad', self.defaults.get('amsgrad', False)):
                    state['max_exp_avg_sq'] = torch.zeros_like(param, device='cuda')
    
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
                    raise RuntimeError('Adam does not support sparse gradients')
                
                state = self.state.setdefault(p, {})
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, device='cuda')
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, device='cuda')
                    if group['amsgrad']:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, device='cuda')
                
                # Ensure all tensors are on CUDA
                if not p.is_cuda:
                    p = p.cuda()
                if not grad.is_cuda:
                    grad = grad.cuda()
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # Update step counter
                state['step'] += 1
                step = state['step']
                
                # Get hyperparameters
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                amsgrad = group['amsgrad']
                
                # Flatten tensors for kernel processing
                param_flat = p.view(-1)
                grad_flat = grad.view(-1)
                exp_avg_flat = exp_avg.view(-1)
                exp_avg_sq_flat = exp_avg_sq.view(-1)
                
                n_elements = param_flat.numel()
                BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
                grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
                
                # Compute bias corrections on CPU
                bias_correction1 = 1.0 - (beta1 ** step)
                bias_correction2 = 1.0 - (beta2 ** step)
                
                if weight_decay > 0 and not amsgrad:
                    # Use standard Adam kernel
                    adam_update_kernel[(grid_size,)](
                        param_flat, grad_flat, exp_avg_flat, exp_avg_sq_flat,
                        lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2,
                        n_elements,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                elif weight_decay > 0 and amsgrad:
                    # AMSGrad variant - maintain maximum of second moment
                    # For simplicity, we'll implement this on CPU for now
                    self._step_amsgrad_cpu(p, grad, group, state)
                else:
                    # No weight decay
                    adam_update_kernel[(grid_size,)](
                        param_flat, grad_flat, exp_avg_flat, exp_avg_sq_flat,
                        lr, beta1, beta2, eps, 0.0, bias_correction1, bias_correction2,
                        n_elements,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                
                # Use in-place operations for parameters that require gradients
                p_data = p.data
                p_data.copy_(param_flat.view(p.shape))
                exp_avg.copy_(exp_avg_flat.view(p.shape))
                exp_avg_sq.copy_(exp_avg_sq_flat.view(p.shape))
        
        return loss
    
    def _step_amsgrad_cpu(self, p, grad, group, state):
        """Fallback CPU implementation for AMSGrad variant"""
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        max_exp_avg_sq = state['max_exp_avg_sq']
        
        state['step'] += 1
        step = state['step']
        
        lr = group['lr']
        beta1, beta2 = group['betas']
        eps = group['eps']
        weight_decay = group['weight_decay']
        
        # Apply weight decay
        if weight_decay != 0:
            grad = grad.add(p, alpha=weight_decay)
        
        # Update exponential moving averages
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Maintains the maximum of all 2nd moment running avg. till now
        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        
        # Use the max for normalizing
        denom = (max_exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** step)).add_(eps)
        
        # Bias correction
        step_size = lr / (1 - beta1 ** step)
        
        # Update parameters
        p.addcdiv_(exp_avg, denom, value=-step_size)


class TritonAdamW(TritonAdam):
    """
    AdamW optimizer implemented with Triton kernels
    
    AdamW: Decoupled Weight Decay Regularization
    https://arxiv.org/abs/1711.05101
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=1e-2, amsgrad=False):
        """
        Initialize AdamW optimizer
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate (default: 1e-3)
            betas: Coefficients for computing running averages (default: (0.9, 0.999))
            eps: Term added to denominator to improve numerical stability (default: 1e-8)
            weight_decay: Weight decay (L2 penalty) (default: 1e-2)
            amsgrad: Whether to use the AMSGrad variant (default: False)
        """
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
    
    def step(self, closure=None):
        """
        Perform a single optimization step
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
                    raise RuntimeError('AdamW does not support sparse gradients')
                
                state = self.state.setdefault(p, {})
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, device='cuda')
                    state['exp_avg_sq'] = torch.zeros_like(p, device='cuda')
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, device='cuda')
                
                # Ensure all tensors are on CUDA
                if not p.is_cuda:
                    p = p.cuda()
                if not grad.is_cuda:
                    grad = grad.cuda()
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                
                # Update step counter
                state['step'] += 1
                step = state['step']
                
                # Get hyperparameters
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                amsgrad = group['amsgrad']
                
                # Flatten tensors for kernel processing
                param_flat = p.view(-1)
                grad_flat = grad.view(-1)
                exp_avg_flat = exp_avg.view(-1)
                exp_avg_sq_flat = exp_avg_sq.view(-1)
                
                n_elements = param_flat.numel()
                BLOCK_SIZE = min(1024, triton.next_power_of_2(n_elements))
                grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
                
                # Compute bias corrections on CPU
                bias_correction1 = 1.0 - (beta1 ** step)
                bias_correction2 = 1.0 - (beta2 ** step)
                
                if not amsgrad:
                    # Use AdamW kernel
                    adamw_update_kernel[(grid_size,)](
                        param_flat, grad_flat, exp_avg_flat, exp_avg_sq_flat,
                        lr, beta1, beta2, eps, weight_decay, bias_correction1, bias_correction2,
                        n_elements,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                else:
                    # AMSGrad variant - implement on CPU for now
                    self._step_amsgrad_cpu_adamw(p, grad, group, state)
                
                # Use in-place operations for parameters that require gradients
                p_data = p.data
                p_data.copy_(param_flat.view(p.shape))
                exp_avg.copy_(exp_avg_flat.view(p.shape))
                exp_avg_sq.copy_(exp_avg_sq_flat.view(p.shape))
        
        return loss
    
    def _step_amsgrad_cpu_adamw(self, p, grad, group, state):
        """Fallback CPU implementation for AMSGrad AdamW variant"""
        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        max_exp_avg_sq = state['max_exp_avg_sq']
        
        state['step'] += 1
        step = state['step']
        
        lr = group['lr']
        beta1, beta2 = group['betas']
        eps = group['eps']
        weight_decay = group['weight_decay']
        
        # Update exponential moving averages
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Maintains the maximum of all 2nd moment running avg. till now
        torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        
        # Bias correction
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1
        
        # Apply weight decay (AdamW style)
        p.mul_(1 - lr * weight_decay)
        
        # Update parameters
        denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        p.addcdiv_(exp_avg, denom, value=-step_size)


def benchmark_optimizers(model_size=1000, num_steps=100, lr=1e-3):
    """
    Benchmark Triton optimizers against PyTorch optimizers
    
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
    
    # Benchmark Triton Adam
    triton_model = torch.nn.Linear(model_size, model_size).cuda()
    triton_adam = TritonAdam(triton_model.parameters(), lr=lr)
    
    start_time = time.time()
    for _ in range(num_steps):
        triton_adam.zero_grad()
        output = triton_model(x)
        loss = criterion(output, y)
        loss.backward()
        triton_adam.step()
    triton_time = time.time() - start_time
    
    # Benchmark PyTorch Adam
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
    
    # Benchmark Triton AdamW
    tritonw_model = torch.nn.Linear(model_size, model_size).cuda()
    triton_adamw = TritonAdamW(tritonw_model.parameters(), lr=lr)
    
    start_time = time.time()
    for _ in range(num_steps):
        triton_adamw.zero_grad()
        output = tritonw_model(x)
        loss = criterion(output, y)
        loss.backward()
        triton_adamw.step()
    tritonw_time = time.time() - start_time
    
    # Benchmark PyTorch AdamW
    torchw_model = torch.nn.Linear(model_size, model_size).cuda()
    torch_adamw = torch.optim.AdamW(torchw_model.parameters(), lr=lr)
    
    start_time = time.time()
    for _ in range(num_steps):
        torch_adamw.zero_grad()
        output = torchw_model(x)
        loss = criterion(output, y)
        loss.backward()
        torch_adamw.step()
    torchw_time = time.time() - start_time
    
    return {
        'triton_adam_time': triton_time,
        'torch_adam_time': torch_time,
        'adam_speedup': torch_time / triton_time,
        'triton_adamw_time': tritonw_time,
        'torch_adamw_time': torchw_time,
        'adamw_speedup': torchw_time / tritonw_time,
        'model_size': model_size,
        'num_steps': num_steps,
        'parameters': model_size * model_size + model_size
    }